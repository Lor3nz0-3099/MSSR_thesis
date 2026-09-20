"""Manual-only ROS-independent Snake8 teleoperation.

R2/L2 command locomotion for all eight modules.  The right stick commands
PAN or TILT only on the currently selected physical module; L1/R1 change
that selection.  There is no head-led trajectory or automatic backbone
shaping in this controller.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import time
from uuid import uuid4

from mssr_expert.behaviors.morphology_locomotion import (
    _planar_body_forward,
    coherent_planar_train_commands,
    validate_locomotion_dofs,
)
from mssr_expert.execution.primitive_protocol import (
    PrimitiveGoalRequest,
    parse_primitive_statuses,
)
from mssr_expert.graph.serialization import attributed_graph_from_dict
from mssr_expert.teleop.action_transport import ActionTransport, PostureDelivery
from mssr_expert.teleop.rc_car import RcCarObservation, finite
from mssr_expert.teleop.safety import SafetyDecision


def _unit(a, b):
    delta = tuple(y - x for x, y in zip(a, b))
    length = math.sqrt(sum(v * v for v in delta))
    return (
        tuple(v / length for v in delta)
        if length > 1e-9
        else (1.0, 0.0, 0.0)
    )


def _pose_yaw(node):
    """Validate the live physical orientation before granting control."""
    orientation = node.attributes["pose"]["orientation_xyzw"]
    if len(orientation) != 4 or not all(finite(v) for v in orientation):
        raise ValueError("Snake pose orientation is invalid")

    x, y, z, w = (float(v) for v in orientation)
    norm = math.sqrt(x*x + y*y + z*z + w*w)

    if norm < 1e-9:
        raise ValueError("Snake pose orientation is zero")

    x, y, z, w = (v / norm for v in (x, y, z, w))
    return math.atan2(
        2 * (w*z + x*y),
        1 - 2 * (y*y + z*z),
    )


@dataclass(frozen=True)
class SnakeObservation:
    graph: object
    assignments: tuple
    inventory: object
    positions: tuple
    root_yaw: float

    @classmethod
    def from_graph(cls, graph, target_graph):
        matched = RcCarObservation.from_graph(graph, target_graph)

        if matched is None:
            return None

        nodes = graph.node_by_id()

        try:
            positions = tuple(
                tuple(
                    float(v)
                    for v in nodes[item.module_id]
                    .attributes["pose"]["position"]
                )
                for item in matched.assignments
            )

            yaws = tuple(
                _pose_yaw(nodes[item.module_id])
                for item in matched.assignments
            )

            if any(
                len(point) != 3
                or not all(finite(v) for v in point)
                for point in positions
            ):
                return None

            dofs = {
                (d.module_id, d.name): d
                for d in matched.inventory.dofs
            }

            for item in matched.assignments:
                for name in ("left_wheel", "right_wheel"):
                    if not dofs[item.module_id, name].can_locomote:
                        return None

                for name in ("pan", "tilt"):
                    dof = dofs[item.module_id, name]
                    if not dof.can_shape or not finite(dof.position_rad):
                        return None

            return cls(
                graph,
                matched.assignments,
                matched.inventory,
                positions,
                yaws[3],
            )

        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class SnakeActions:
    module_actions: dict = field(default_factory=dict)
    joint_targets: tuple = ()
    intent: dict = field(default_factory=dict)
    module_roles: dict = field(default_factory=dict)
    allow_joint_updates: bool = False
    manual_override: bool = False


class SnakeManualController:
    """Global wheel drive plus one selected module for manual PAN/TILT."""

    def __init__(
        self,
        library,
        *,
        geometry,
        joint_rate_rad_s=0.5,
        joint_deadband_rad=0.025,
        max_dt_s=0.1,
    ):
        values = (
            joint_rate_rad_s,
            joint_deadband_rad,
            max_dt_s,
        )

        if any(not finite(v) or v <= 0 for v in values):
            raise ValueError(
                "Snake manual rates must be positive and finite"
            )

        # Keep the established constructor contract and validate geometry,
        # although manual locomotion does not generate a virtual backbone.
        spacing = float(geometry.top_to_bottom_spacing_m)

        if not finite(spacing) or spacing <= 0:
            raise ValueError("Snake module spacing must be positive")

        self.library = library
        self.joint_rate = float(joint_rate_rad_s)
        self.deadband = float(joint_deadband_rad)
        self.max_dt = float(max_dt_s)

    def step(
        self,
        input,
        observation,
        dt,
        *,
        safety=SafetyDecision("NONE", False, True, False),
    ):
        if not safety.motion_enabled or observation is None:
            return SnakeActions()

        try:
            r2 = float(input.r2)
            l2 = float(input.l2)
        except (AttributeError, TypeError, ValueError):
            return SnakeActions()

        if (
            not finite(dt)
            or dt < 0
            or not finite(r2)
            or not finite(l2)
            or not 0 <= r2 <= 1
            or not 0 <= l2 <= 1
        ):
            return SnakeActions()

        drive = self.library._profile("snake8")["drive"]

        speed = (
            r2 - l2
        ) * float(drive["max_linear_m_s"])

        commands = self.library.drive_commands(
            "snake8",
            observation.assignments,
            linear_m_s=speed,
            yaw_rate_rad_s=0.0,
        )

        commands = coherent_planar_train_commands(
            observation.graph,
            commands,
        )

        # Preserve one physical "forward" convention for the whole Snake,
        # independent of alternating local module frames.
        first = observation.assignments[0].module_id

        local_forward = _planar_body_forward(
            observation.graph.node_by_id()[first].attributes
        )

        tangent = _unit(
            observation.positions[-2],
            observation.positions[-1],
        )

        if (
            local_forward is not None
            and (
                local_forward[0] * tangent[0]
                + local_forward[1] * tangent[1]
            ) < 0
        ):
            for command in commands.values():
                command["vx"] = -command["vx"]

        validate_locomotion_dofs(
            commands,
            observation.inventory,
        )

        roles = {
            item.module_id: item.target_role
            for item in observation.assignments
        }

        return SnakeActions(
            module_actions=commands,
            intent={
                "longitudinal_m_s": speed,
            },
            module_roles=roles,
            allow_joint_updates=safety.allow_joint_updates,
            manual_override=True,
        )


class SnakePostureTransport:
    """One primitive-file write per tick, with per-module ownership and ACKs."""
    def __init__(self, *, retry_s=0.25, timeout_s=10.0,
                 target_deadband_rad=0.04, retarget_rad=0.2):
        self.retry_s = retry_s
        self.timeout_s = timeout_s
        self.target_deadband = target_deadband_rad
        self.retarget_rad = retarget_rad
        self._goals = {}
        self._ack = set()
        self._last_send = {}
        self._retired = {}
        self._desired = {}

    def observe(self, payload):
        for goal_id, status in parse_primitive_statuses(payload).items():
            for module, goal in tuple(self._goals.items()):
                if goal.goal_id != goal_id or status.module_ids != goal.module_ids or status.primitive != goal.primitive:
                    continue
                if status.terminal and status.code != "DUPLICATE_GOAL_ID":
                    self._goals.pop(module)
                    self._ack.discard(module)
                    self._retired[module] = goal if status.state == "succeeded" else None
                else:
                    self._ack.add(module)

    def effective_targets(self):
        active = set(self._goals) | {module for module, goal in self._retired.items()
                                     if goal is not None}
        return {module: self._desired[module] for module in active}

    def step(self, actions, *, now):
        if any(joint not in ("pan", "tilt")
               for _, joint, _, _ in actions.joint_targets):
            raise ValueError("Snake posture accepts PAN/TILT targets only")
        requested = {module: (joint, angle, measured)
                     for module, joint, angle, measured in actions.joint_targets}
        if not actions.allow_joint_updates:
            for module, goal in tuple(self._retired.items()):
                if goal is not None and module not in self._goals:
                    self._goals[module] = goal
            pending = tuple(self._goals.values())
            if pending:
                return PostureDelivery(cancel_goal_ids=tuple(goal.goal_id for goal in pending),
                                       blocked_module_ids=tuple(sorted(self._goals)))
            return PostureDelivery()
        for module, goal in tuple(self._goals.items()):
            wanted = requested.get(module)
            angle = self._desired[module][1]
            if (not actions.allow_joint_updates
                    or (wanted is not None and (
                        self._desired[module][0] != wanted[0]
                        or (abs(angle - wanted[1]) > self.retarget_rad
                            and now - self._last_send[module] >= self.retry_s)))):
                return PostureDelivery(cancel_goal_id=goal.goal_id,
                                       blocked_module_ids=(module,))
        # The primitive goal bridge also has one file slot. Wait for native
        # admission before writing another module goal into that slot.
        for module, goal in self._goals.items():
            if module not in self._ack:
                if now - self._last_send[module] >= self.retry_s:
                    self._last_send[module] = now
                    return PostureDelivery(goal=goal)
                return PostureDelivery()
        for module, (joint, angle, measured) in requested.items():
            if module in self._goals:
                continue
            previous = self._retired.get(module)
            if previous is not None:
                if self._desired[module][0] == joint and abs(self._desired[module][1] - angle) <= self.target_deadband:
                    continue
                # A retained reached target still owns the internal motors.
                # Native cancellation must retire it before another mode starts.
                self._goals[module] = previous
                return PostureDelivery(cancel_goal_id=previous.goal_id,
                                       blocked_module_ids=(module,))
            goal = PrimitiveGoalRequest(
                goal_id="teleop-snake-" + uuid4().hex,
                primitive="rotate_pan_by" if joint == "pan" else "set_tilt",
                module_ids=(module,),
                parameters={"delta_rad" if joint == "pan" else "angle_rad":
                            angle - measured if joint == "pan" else angle,
                            "retain_reached_on_interrupt": True},
                timeout_s=self.timeout_s,
            )
            self._goals[module] = goal
            self._desired[module] = (joint, angle)
            self._last_send[module] = now
            return PostureDelivery(goal=goal)
        return PostureDelivery()


@dataclass(frozen=True)
class SnakeTick:
    actions: SnakeActions
    posture: PostureDelivery
    envelope: str | None
    effective_actions: dict


class SnakeRuntime:
    def __init__(
        self,
        library,
        target_graph,
        *,
        geometry,
        observation_timeout_s=0.5,
        **controller_config,
    ):
        if (
            not finite(observation_timeout_s)
            or observation_timeout_s <= 0
        ):
            raise ValueError(
                "Snake observation timeout must be positive"
            )

        self.controller = SnakeManualController(
            library,
            geometry=geometry,
            **controller_config,
        )

        self.posture = SnakePostureTransport()
        self.transport = ActionTransport()
        self.target_graph = target_graph
        self.observation_timeout_s = observation_timeout_s

        self._observation = None
        self._received_at = None
        self._last_graph = None
        self._previous_tick = None
        self._motion_enabled = False

        self._command_id = (
            "teleop-snake-" + uuid4().hex
        )

        # Index zero in reversed assignment order is the Snake head.
        self._manual_index = 0
        self._manual_key = None
        self._manual_target = None
        self._manual_transition = False
        self._assignment_ids = ()

    def observe_graph(self, payload, *, now):
        try:
            graph = attributed_graph_from_dict(payload)

            if not finite(now):
                return False

            if (
                self._last_graph is not None
                and graph.stamp <= self._last_graph.stamp
            ):
                if graph == self._last_graph:
                    return self._observation is not None

                raise ValueError(
                    "Snake graph timestamp did not advance"
                )

            self._observation = SnakeObservation.from_graph(
                graph,
                self.target_graph,
            )

            if self._observation is not None:
                assignment_ids = tuple(
                    item.module_id
                    for item in self._observation.assignments
                )

                if (
                    self._assignment_ids
                    and assignment_ids != self._assignment_ids
                ):
                    self._manual_index = 0
                    self._manual_transition = True
                    self._manual_key = None
                    self._manual_target = None

                self._assignment_ids = assignment_ids
                self._last_graph = graph
                self._received_at = now

                return True

        except (
            AttributeError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            pass

        self._observation = None
        self._received_at = None
        self._manual_index = 0
        self._manual_transition = True
        self._manual_key = None
        self._manual_target = None

        return False

    @property
    def latest_graph(self):
        return self._last_graph

    def topology(self, now):
        return (
            "snake8"
            if (
                self._observation is not None
                and self._received_at is not None
                and 0
                <= now - self._received_at
                <= self.observation_timeout_s
            )
            else None
        )

    def observe_status(self, payload):
        self.posture.observe(payload)

    def _selected(self, observation):
        return tuple(
            reversed(observation.assignments)
        )[self._manual_index]

    def _manual_joint_target(
        self,
        input,
        observation,
        dt,
    ):
        try:
            x = float(input.right_x)
            y = float(input.right_y)
        except (AttributeError, TypeError, ValueError):
            return (), "invalid_stick"

        if not all(
            finite(value) and -1 <= value <= 1
            for value in (x, y)
        ):
            return (), "invalid_stick"

        # One physical joint at a time avoids competing primitive ownership.
        if abs(x) > 0.1 and abs(y) > 0.1:
            return (), "one_joint_axis_at_a_time"

        if max(abs(x), abs(y)) <= 0.1:
            return (), None

        joint, value = (
            ("pan", x)
            if abs(x) > 0.1
            else ("tilt", y)
        )

        selected = self._selected(observation)

        dof = next(
            (
                item
                for item in observation.inventory.dofs
                if item.module_id == selected.module_id
                and item.name == joint
            ),
            None,
        )

        if (
            dof is None
            or not dof.can_shape
            or not finite(dof.position_rad)
        ):
            return (), "joint_unavailable"

        key = (selected.module_id, joint)

        if key != self._manual_key:
            self._manual_key = key
            self._manual_target = dof.position_rad

        dt = max(
            0.0,
            min(float(dt), self.controller.max_dt),
        )

        target = (
            self._manual_target
            + value * self.controller.joint_rate * dt
        )

        if finite(dof.lower_limit_rad):
            target = max(target, dof.lower_limit_rad)

        if finite(dof.upper_limit_rad):
            target = min(target, dof.upper_limit_rad)

        self._manual_target = target

        if (
            abs(target - dof.position_rad)
            <= self.controller.deadband
        ):
            return (), None

        return (
            (
                selected.module_id,
                joint,
                target,
                dof.position_rad,
            ),
        ), None

    def step(self, input, *, safety, now):
        if not finite(now):
            raise ValueError(
                "Snake tick time must be finite"
            )

        fresh = (
            self._received_at is not None
            and 0
            <= now - self._received_at
            <= self.observation_timeout_s
        )

        observation = (
            self._observation
            if fresh
            else None
        )

        dt = (
            0.0
            if self._previous_tick is None
            else now - self._previous_tick
        )

        self._previous_tick = now

        events = tuple(
            getattr(input, "command_events", ())
        )

        stick_neutral = (
            abs(float(getattr(input, "right_x", 0.0))) < 0.05
            and abs(float(getattr(input, "right_y", 0.0))) < 0.05
        )

        manual_rejection = None

        if (
            observation is not None
            and safety.motion_enabled
            and (
                "next_module" in events
                or "previous_module" in events
            )
        ):
            if not stick_neutral:
                manual_rejection = (
                    "release_right_stick_before_module_switch"
                )

            elif (
                "next_module" in events
                and "previous_module" in events
            ):
                manual_rejection = (
                    "ambiguous_module_selection"
                )

            else:
                count = len(observation.assignments)

                delta = (
                    1
                    if "next_module" in events
                    else -1
                )

                self._manual_index = (
                    self._manual_index + delta
                ) % count

                self._manual_key = None
                self._manual_target = None
                self._manual_transition = True

        actions = self.controller.step(
            input,
            observation,
            dt,
            safety=safety,
        )

        if (
            observation is not None
            and safety.motion_enabled
        ):
            selected = self._selected(observation)

            if self._manual_transition:
                targets = ()
                rejected = None
            else:
                targets, rejected = (
                    self._manual_joint_target(
                        input,
                        observation,
                        dt,
                    )
                )

            manual_rejection = (
                manual_rejection or rejected
            )

            actions = replace(
                actions,
                joint_targets=targets,
                manual_override=True,
            )

            selected_id = selected.module_id
            selected_role = selected.target_role

        else:
            selected_id = None
            selected_role = None

        if (
            self._manual_transition
            or not safety.motion_enabled
        ):
            delivery = self.posture.step(
                replace(
                    actions,
                    joint_targets=(),
                    allow_joint_updates=False,
                ),
                now=now,
            )

            if (
                self._manual_transition
                and not self.posture.effective_targets()
            ):
                self._manual_transition = False

        else:
            delivery = self.posture.step(
                actions,
                now=now,
            )

        intent = dict(actions.intent)

        intent.update(
            control_mode="manual",
            selected_module_id=selected_id,
            selected_role=selected_role,
            manual_transition_pending=self._manual_transition,
            manual_rejection=manual_rejection,
        )

        actions = replace(
            actions,
            intent=intent,
        )

        enabled = bool(
            safety.motion_enabled
            and observation is not None
            and fresh
        )

        # E-stop/native ownership invalidates the previous behavior command
        # identity.  Match RcCarRuntime: a newly armed TELEOP epoch gets a
        # fresh command ID, while ordinary enabled ticks retain one stable ID.
        if enabled and not self._motion_enabled:
            self._command_id = (
                "teleop-snake-" + uuid4().hex
            )

        self._motion_enabled = enabled

        envelope = None

        if (
            actions.module_actions
            and enabled
        ):
            envelope = self.transport.serialize(
                actions,
                stamp=time.time(),
                command_id=self._command_id,
                morphology="snake8",
            )

        effective = {
            module: dict(command)
            for module, command
            in actions.module_actions.items()
        }

        for module, (joint, angle) in (
            self.posture.effective_targets().items()
        ):
            effective.setdefault(
                module,
                {},
            )[joint + "_target_rad"] = angle

        return SnakeTick(
            replace(
                actions,
                module_actions=effective,
            ),
            delivery,
            envelope,
            effective,
        )
