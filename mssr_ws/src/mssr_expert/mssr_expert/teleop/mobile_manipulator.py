"""Manual-only MobileManipulator8 teleoperation.

R2/L2 command longitudinal locomotion through the validated MM8 drive
profile.  The right stick commands PAN or TILT on exactly one selected
physical module; L1/R1 move that selection through the live role assignment.

There is no Cartesian end-effector controller in human teleoperation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import time
from uuid import uuid4

from mssr_expert.behaviors.morphology_locomotion import (
    coherent_planar_train_commands,
    validate_locomotion_dofs,
)
from mssr_expert.behaviors.morphology_library import JointTarget
from mssr_expert.execution.primitive_protocol import (
    PrimitiveGoalRequest,
    parse_primitive_statuses,
)
from mssr_expert.graph.serialization import attributed_graph_from_dict
from mssr_expert.teleop.action_transport import (
    ActionTransport,
    PostureDelivery,
)
from mssr_expert.teleop.rc_car import RcCarObservation, finite
from mssr_expert.teleop.safety import SafetyDecision


@dataclass(frozen=True)
class MobileManipulatorObservation:
    graph: object
    assignments: tuple
    inventory: object

    @classmethod
    def from_graph(cls, graph, target_graph):
        matched = RcCarObservation.from_graph(
            graph,
            target_graph,
        )

        if matched is None:
            return None

        roles = {
            item.target_role: item.module_id
            for item in matched.assignments
        }

        # T7 authoritative longitudinal pair.
        locomotor_roles = (
            "front_support",
            "arm_lift",
        )

        if any(role not in roles for role in locomotor_roles):
            return None

        dofs = {
            (item.module_id, item.name): item
            for item in matched.inventory.dofs
        }

        try:
            # Every module remains manually shapeable through PAN/TILT.
            for assignment in matched.assignments:
                for joint in ("pan", "tilt"):
                    dof = dofs[(assignment.module_id, joint)]

                    if (
                        not dof.can_shape
                        or not finite(dof.position_rad)
                    ):
                        return None

            # Only the validated MM8 longitudinal pair needs free wheels.
            for role in locomotor_roles:
                module_id = roles[role]

                for joint in ("left_wheel", "right_wheel"):
                    if not dofs[(module_id, joint)].can_locomote:
                        return None

        except KeyError:
            return None

        return cls(
            graph=graph,
            assignments=matched.assignments,
            inventory=matched.inventory,
        )


@dataclass(frozen=True)
class MobileManipulatorActions:
    module_actions: dict = field(default_factory=dict)
    joint_targets: tuple = ()
    intent: dict = field(default_factory=dict)
    module_roles: dict = field(default_factory=dict)
    allow_joint_updates: bool = False
    manual_override: bool = False


class MobileManipulatorManualController:
    """MM8 longitudinal drive plus one-module-at-a-time shape control."""

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

        if any(
            not finite(value) or value <= 0.0
            for value in values
        ):
            raise ValueError(
                "MM8 manual rates must be positive and finite"
            )

        spacing = float(
            geometry.top_to_bottom_spacing_m
        )

        if not finite(spacing) or spacing <= 0.0:
            raise ValueError(
                "MM8 module spacing must be positive"
            )

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
        safety=SafetyDecision(
            "NONE",
            False,
            True,
            False,
        ),
    ):
        if (
            not safety.motion_enabled
            or observation is None
        ):
            return MobileManipulatorActions()

        try:
            r2 = float(input.r2)
            l2 = float(input.l2)
        except (
            AttributeError,
            TypeError,
            ValueError,
        ):
            return MobileManipulatorActions()

        if (
            not finite(dt)
            or dt < 0.0
            or not finite(r2)
            or not finite(l2)
            or not 0.0 <= r2 <= 1.0
            or not 0.0 <= l2 <= 1.0
        ):
            return MobileManipulatorActions()

        profile = self.library._profile(
            "mobile_manipulator8"
        )["drive"]

        speed = (
            r2 - l2
        ) * float(
            profile["max_linear_m_s"]
        )

        if abs(speed) > 1.0e-12:
            commands = self.library.drive_commands(
                "mobile_manipulator8",
                observation.assignments,
                linear_m_s=speed,
                yaw_rate_rad_s=0.0,
            )

            commands = coherent_planar_train_commands(
                observation.graph,
                commands,
            )

            validate_locomotion_dofs(
                commands,
                observation.inventory,
            )

        else:
            # Dead-man release is an explicit native zero, not silence.
            # Keep this pair identical to the validated MM8 translation
            # selectors so the previous non-zero action cannot remain cached.
            by_role = {
                item.target_role: item.module_id
                for item in observation.assignments
            }

            commands = {
                by_role["front_support"]: {
                    "vx": 0.0,
                    "vy": 0.0,
                    "yaw_rate": 0.0,
                },
                by_role["arm_lift"]: {
                    "vx": 0.0,
                    "vy": 0.0,
                    "yaw_rate": 0.0,
                },
            }

        roles = {
            item.module_id: item.target_role
            for item in observation.assignments
        }

        return MobileManipulatorActions(
            module_actions=commands,
            intent={
                "longitudinal_m_s": speed,
            },
            module_roles=roles,
            allow_joint_updates=safety.allow_joint_updates,
            manual_override=True,
        )


class MobileManipulatorPostureTransport:
    """MM8 manual PAN/TILT primitive ownership with explicit retirement."""

    def __init__(
        self,
        *,
        retry_s=0.25,
        timeout_s=10.0,
        target_deadband_rad=0.04,
        retarget_rad=0.2,
    ):
        self.retry_s = float(retry_s)
        self.timeout_s = float(timeout_s)
        self.target_deadband = float(
            target_deadband_rad
        )
        self.retarget_rad = float(
            retarget_rad
        )

        self._goals = {}
        self._ack = set()
        self._last_send = {}
        self._retired = {}
        self._desired = {}
        self._cancel_pending = set()

    def observe(self, payload):
        for goal_id, status in (
            parse_primitive_statuses(payload).items()
        ):
            for module, goal in tuple(
                self._goals.items()
            ):
                if (
                    goal.goal_id != goal_id
                    or status.module_ids
                    != goal.module_ids
                    or status.primitive
                    != goal.primitive
                ):
                    continue

                if (
                    status.terminal
                    and status.code
                    != "DUPLICATE_GOAL_ID"
                ):
                    self._goals.pop(module)
                    self._ack.discard(module)
                    self._cancel_pending.discard(
                        goal.goal_id
                    )

                    self._retired[module] = (
                        goal
                        if status.state == "succeeded"
                        else None
                    )

                else:
                    self._ack.add(module)

    def effective_targets(self):
        active = (
            set(self._goals)
            | {
                module
                for module, goal
                in self._retired.items()
                if goal is not None
            }
        )

        return {
            module: self._desired[module]
            for module in active
        }

    def retire_modules(self, module_ids):
        """Cancel posture ownership only for the requested modules."""

        cancel_ids = []
        blocked = []

        for module in dict.fromkeys(module_ids):
            goal = self._goals.get(module)

            if goal is None:
                goal = self._retired.get(module)

                if goal is not None:
                    self._goals[module] = goal

            if goal is not None:
                self._cancel_pending.add(
                    goal.goal_id
                )
                cancel_ids.append(goal.goal_id)
                blocked.append(module)

        return PostureDelivery(
            cancel_goal_ids=tuple(cancel_ids),
            blocked_module_ids=tuple(blocked),
        )

    def step(self, actions, *, now):
        if any(
            joint not in ("pan", "tilt")
            for _, joint, _, _
            in actions.joint_targets
        ):
            raise ValueError(
                "MM8 manual posture accepts PAN/TILT only"
            )

        requested = {
            module: (
                joint,
                angle,
                measured,
            )
            for (
                module,
                joint,
                angle,
                measured,
            )
            in actions.joint_targets
        }

        if not actions.allow_joint_updates:
            for module, goal in tuple(
                self._retired.items()
            ):
                if (
                    goal is not None
                    and module not in self._goals
                ):
                    self._goals[module] = goal

            pending = tuple(
                self._goals.values()
            )

            if pending:
                self._cancel_pending.update(
                    goal.goal_id
                    for goal in pending
                )

                return PostureDelivery(
                    cancel_goal_ids=tuple(
                        goal.goal_id
                        for goal in pending
                    ),
                    blocked_module_ids=tuple(
                        sorted(self._goals)
                    ),
                )

            return PostureDelivery()

        for module, goal in tuple(
            self._goals.items()
        ):
            wanted = requested.get(module)
            angle = self._desired[module][1]

            if (
                wanted is not None
                and (
                    self._desired[module][0]
                    != wanted[0]
                    or (
                        abs(
                            angle
                            - wanted[1]
                        )
                        > self.retarget_rad
                        and (
                            now
                            - self._last_send[module]
                            >= self.retry_s
                        )
                    )
                )
            ):
                self._cancel_pending.add(
                    goal.goal_id
                )

                return PostureDelivery(
                    cancel_goal_id=goal.goal_id,
                    blocked_module_ids=(module,),
                )

        # One primitive file slot: wait until native admission before
        # publishing another module goal.
        for module, goal in self._goals.items():
            if goal.goal_id in self._cancel_pending:
                return PostureDelivery(
                    cancel_goal_id=goal.goal_id,
                    blocked_module_ids=(module,),
                )

            if module not in self._ack:
                if (
                    now
                    - self._last_send[module]
                    >= self.retry_s
                ):
                    self._last_send[module] = now

                    return PostureDelivery(
                        goal=goal
                    )

                return PostureDelivery()

        for module, (
            joint,
            angle,
            measured,
        ) in requested.items():
            if module in self._goals:
                continue

            previous = self._retired.get(
                module
            )

            if previous is not None:
                if (
                    self._desired[module][0]
                    == joint
                    and abs(
                        self._desired[module][1]
                        - angle
                    )
                    <= self.target_deadband
                ):
                    continue

                # A retained primitive still owns the internal motors.
                # Cancel it before changing joint/mode.
                self._goals[module] = previous
                self._cancel_pending.add(
                    previous.goal_id
                )

                return PostureDelivery(
                    cancel_goal_id=previous.goal_id,
                    blocked_module_ids=(module,),
                )

            goal = PrimitiveGoalRequest(
                goal_id=(
                    "teleop-mm8-"
                    + uuid4().hex
                ),
                primitive=(
                    "rotate_pan_by"
                    if joint == "pan"
                    else "set_tilt"
                ),
                module_ids=(module,),
                parameters={
                    (
                        "delta_rad"
                        if joint == "pan"
                        else "angle_rad"
                    ): (
                        angle - measured
                        if joint == "pan"
                        else angle
                    ),
                    "retain_reached_on_interrupt":
                        True,
                },
                timeout_s=self.timeout_s,
            )

            self._goals[module] = goal
            self._desired[module] = (
                joint,
                angle,
            )
            self._last_send[module] = now

            return PostureDelivery(
                goal=goal
            )

        return PostureDelivery()


@dataclass(frozen=True)
class MobileManipulatorTick:
    actions: MobileManipulatorActions
    posture: PostureDelivery
    envelope: str | None
    effective_actions: dict


class MobileManipulatorRuntime:
    """ROS-independent T7 MM8 manual teleoperation boundary."""

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
            or observation_timeout_s <= 0.0
        ):
            raise ValueError(
                "MM8 observation timeout must be positive"
            )

        self.controller = (
            MobileManipulatorManualController(
                library,
                geometry=geometry,
                **controller_config,
            )
        )

        self.posture = (
            MobileManipulatorPostureTransport()
        )

        self.transport = ActionTransport()
        self.target_graph = target_graph
        self.observation_timeout_s = float(
            observation_timeout_s
        )

        self._observation = None
        self._received_at = None
        self._last_graph = None
        self._previous_tick = None
        self._motion_enabled = False

        self._command_id = (
            "teleop-mm8-" + uuid4().hex
        )

        # MM8 starts in the validated Scorpion / drive-ready posture.
        # Drive and manipulation are mutually exclusive teleop modes.
        self._mode = "drive_ready"
        self._mode_transition_pending = False
        self._mode_transition_behavior = None
        self._mode_transition_destination = None
        self._mode_transition_targets = ()
        self._mode_transition_index = 0
        self._mode_transition_goal_id = None

        # Physical Scorpion produced by assembly.
        # Captured once from the first valid MM8 graph and used verbatim
        # whenever manipulation returns to drive_ready.
        self._assembly_drive_targets = None

        # Manual selection is restricted to the four-module arm chain.
        # end_effector is the initial human-selected role.
        self._manual_index = 0
        self._manual_key = None
        self._manual_target = None
        self._posture_reset_pending = False
        self._manual_transition = False
        self._manual_transition_module_id = None
        self._assignment_ids = ()

    def observe_graph(self, payload, *, now):
        try:
            if not finite(now):
                return False

            graph = attributed_graph_from_dict(
                payload
            )

            if (
                self._last_graph is not None
                and graph.stamp
                <= self._last_graph.stamp
            ):
                if graph == self._last_graph:
                    return (
                        self._observation
                        is not None
                    )

                raise ValueError(
                    "MM8 graph timestamp did not advance"
                )

            observation = (
                MobileManipulatorObservation.from_graph(
                    graph,
                    self.target_graph,
                )
            )

            if observation is None:
                raise ValueError(
                    "invalid MobileManipulator8 topology"
                )

            assignment_ids = tuple(
                item.module_id
                for item in observation.assignments
            )

            if self._assembly_drive_targets is None:
                dofs = {
                    (item.module_id, item.name): item
                    for item in observation.inventory.dofs
                }

                self._assembly_drive_targets = tuple(
                    JointTarget(
                        module_id=assignment.module_id,
                        joint=joint,
                        angle_rad=float(
                            dofs[
                                (
                                    assignment.module_id,
                                    joint,
                                )
                            ].position_rad
                        ),
                        target_vertex_id=(
                            assignment.target_vertex_id
                        ),
                        target_role=(
                            assignment.target_role
                        ),
                    )
                    for assignment in observation.assignments
                    for joint in ("pan", "tilt")
                )

            if (
                self._assignment_ids
                and assignment_ids
                != self._assignment_ids
            ):
                self._manual_index = 0
                self._posture_reset_pending = True
                self._manual_key = None
                self._manual_target = None

            self._assignment_ids = assignment_ids
            self._observation = observation
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
            self._observation = None
            self._received_at = None
            self._manual_index = 0
            self._posture_reset_pending = True
            self._manual_key = None
            self._manual_target = None

            return False

    @property
    def latest_graph(self):
        return self._last_graph

    def topology(self, now):
        return (
            "mobile_manipulator8"
            if (
                self._observation is not None
                and self._received_at is not None
                and 0.0
                <= now - self._received_at
                <= self.observation_timeout_s
            )
            else None
        )

    def observe_status(self, payload):
        statuses = parse_primitive_statuses(
            payload
        )

        self.posture.observe(payload)

        goal_id = self._mode_transition_goal_id

        if (
            not self._mode_transition_pending
            or goal_id is None
        ):
            return

        status = statuses.get(goal_id)

        if (
            status is None
            or not status.terminal
            or (
                status.state == "rejected"
                and status.code == "DUPLICATE_GOAL_ID"
            )
        ):
            return

        if status.state == "succeeded":
            self._mode_transition_goal_id = None
            self._mode_transition_index += 1
            return

        # Failure/cancel/rejection is never an implicit resume point.
        self._mode = "transition_interrupted"
        self._mode_transition_pending = False
        self._mode_transition_behavior = None
        self._mode_transition_destination = None
        self._mode_transition_targets = ()
        self._mode_transition_index = 0
        self._mode_transition_goal_id = None
        self._manual_key = None
        self._manual_target = None

    def _begin_mode_transition(
        self,
        observation,
        *,
        behavior,
        transient_mode,
        destination_mode,
    ):
        if behavior == "restore_drive":
            if self._assembly_drive_targets is None:
                raise RuntimeError(
                    "MM8 restore_drive has no captured assembly posture"
                )

            targets = self._assembly_drive_targets

        else:
            targets = tuple(
                self.controller.library.behavior_joint_targets(
                    "mobile_manipulator8",
                    behavior,
                    observation.assignments,
                )
            )

        if not targets:
            raise ValueError(
                f"MM8 behavior {behavior!r} has no posture targets"
            )

        self._mode = transient_mode
        self._mode_transition_pending = True
        self._mode_transition_behavior = behavior
        self._mode_transition_destination = destination_mode
        self._mode_transition_targets = targets
        self._mode_transition_index = 0
        self._mode_transition_goal_id = None

        # A mode transition starts a new posture authority epoch.
        self._manual_key = None
        self._manual_target = None
        self._manual_transition = False

    def _current_mode_joint_target(
        self,
        observation,
    ):
        if not self._mode_transition_pending:
            return ()

        if (
            self._mode_transition_goal_id is not None
            or self._mode_transition_index
            >= len(self._mode_transition_targets)
        ):
            return ()

        target = self._mode_transition_targets[
            self._mode_transition_index
        ]

        dof = next(
            (
                item
                for item in observation.inventory.dofs
                if (
                    item.module_id == target.module_id
                    and item.name == target.joint
                )
            ),
            None,
        )

        if (
            dof is None
            or not finite(dof.position_rad)
        ):
            raise ValueError(
                "MM8 transition target has no finite measured joint"
            )

        return (
            (
                target.module_id,
                target.joint,
                target.angle_rad,
                dof.position_rad,
            ),
        )

    def _finish_mode_transition_if_ready(self):
        if (
            not self._mode_transition_pending
            or self._mode_transition_goal_id is not None
            or self._mode_transition_index
            < len(self._mode_transition_targets)
        ):
            return False

        destination = self._mode_transition_destination

        if destination not in {
            "drive_ready",
            "manipulation_ready",
        }:
            raise RuntimeError(
                "MM8 transition has invalid destination"
            )

        self._mode = destination
        self._mode_transition_pending = False
        self._mode_transition_behavior = None
        self._mode_transition_destination = None
        self._mode_transition_targets = ()
        self._mode_transition_index = 0
        self._mode_transition_goal_id = None
        self._manual_key = None
        self._manual_target = None

        return True

    def _interrupt_mode_transition(self):
        if not self._mode_transition_pending:
            return

        self._mode = "transition_interrupted"
        self._mode_transition_pending = False
        self._mode_transition_behavior = None
        self._mode_transition_destination = None
        self._mode_transition_targets = ()
        self._mode_transition_index = 0
        self._mode_transition_goal_id = None
        self._manual_key = None
        self._manual_target = None

    def _manual_assignments(self, observation):
        by_role = {
            item.target_role: item
            for item in observation.assignments
        }

        arm_roles = (
            "end_effector",
            "arm_link",
            "arm_lift",
            "arm_ground_drive",
        )

        try:
            return tuple(
                by_role[role]
                for role in arm_roles
            )

        except KeyError as error:
            raise ValueError(
                "MobileManipulator8 arm role assignment is incomplete"
            ) from error

    def _selected(self, observation):
        return self._manual_assignments(
            observation
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

        except (
            AttributeError,
            TypeError,
            ValueError,
        ):
            return (), "invalid_stick"

        if not all(
            finite(value)
            and -1.0 <= value <= 1.0
            for value in (x, y)
        ):
            return (), "invalid_stick"

        # Same ownership rule as Snake: one internal joint at a time.
        if (
            abs(x) > 0.1
            and abs(y) > 0.1
        ):
            return (
                (),
                "one_joint_axis_at_a_time",
            )

        if max(
            abs(x),
            abs(y),
        ) <= 0.1:
            return (), None

        joint, value = (
            ("pan", x)
            if abs(x) > 0.1
            else ("tilt", y)
        )

        selected = self._selected(
            observation
        )

        dof = next(
            (
                item
                for item
                in observation.inventory.dofs
                if (
                    item.module_id
                    == selected.module_id
                    and item.name == joint
                )
            ),
            None,
        )

        if (
            dof is None
            or not dof.can_shape
            or not finite(dof.position_rad)
        ):
            return (), "joint_unavailable"

        key = (
            selected.module_id,
            joint,
        )

        if key != self._manual_key:
            self._manual_key = key
            self._manual_target = (
                dof.position_rad
            )

        dt = max(
            0.0,
            min(
                float(dt),
                self.controller.max_dt,
            ),
        )

        target = (
            self._manual_target
            + value
            * self.controller.joint_rate
            * dt
        )

        if finite(dof.lower_limit_rad):
            target = max(
                target,
                dof.lower_limit_rad,
            )

        if finite(dof.upper_limit_rad):
            target = min(
                target,
                dof.upper_limit_rad,
            )

        self._manual_target = target

        if (
            abs(
                target
                - dof.position_rad
            )
            <= self.controller.deadband
        ):
            return (), None

        return (
            (
                (
                    selected.module_id,
                    joint,
                    target,
                    dof.position_rad,
                ),
            ),
            None,
        )

    def step(
        self,
        input,
        *,
        safety,
        now,
    ):
        if not finite(now):
            raise ValueError(
                "MM8 tick time must be finite"
            )

        fresh = (
            self._received_at is not None
            and 0.0
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
            getattr(
                input,
                "command_events",
                (),
            )
        )

        stick_neutral = (
            abs(
                float(
                    getattr(
                        input,
                        "right_x",
                        0.0,
                    )
                )
            )
            < 0.05
            and abs(
                float(
                    getattr(
                        input,
                        "right_y",
                        0.0,
                    )
                )
            )
            < 0.05
        )

        manual_rejection = None

        if (
            self._mode_transition_pending
            and not safety.motion_enabled
        ):
            self._interrupt_mode_transition()

        if (
            observation is not None
            and safety.motion_enabled
            and "home" in events
            and not self._mode_transition_pending
        ):
            if self._mode == "drive_ready":
                self._begin_mode_transition(
                    observation,
                    behavior="prepare_manipulation",
                    transient_mode="to_manipulation_ready",
                    destination_mode="manipulation_ready",
                )

            elif self._mode == "manipulation_ready":
                self._begin_mode_transition(
                    observation,
                    behavior="restore_drive",
                    transient_mode="to_drive_ready",
                    destination_mode="drive_ready",
                )

        module_switch_requested = (
            "next_module" in events
            or "previous_module" in events
        )

        if (
            observation is not None
            and safety.motion_enabled
            and module_switch_requested
        ):
            if self._mode != "manipulation_ready":
                manual_rejection = (
                    "module_selection_requires_manipulation_ready"
                )

            elif not stick_neutral:
                manual_rejection = (
                    "release_right_stick_"
                    "before_module_switch"
                )

            elif (
                "next_module" in events
                and "previous_module"
                in events
            ):
                manual_rejection = (
                    "ambiguous_module_selection"
                )

            else:
                count = len(
                    self._manual_assignments(
                        observation
                    )
                )

                delta = (
                    1
                    if "next_module"
                    in events
                    else -1
                )

                outgoing = self._selected(
                    observation
                )

                self._manual_index = (
                    self._manual_index
                    + delta
                ) % count

                self._manual_key = None
                self._manual_target = None
                self._manual_transition = True
                self._manual_transition_module_id = (
                    outgoing.module_id
                )

        actions = self.controller.step(
            input,
            observation,
            dt,
            safety=safety,
        )

        locomotion_rejection = None

        if (
            observation is not None
            and safety.motion_enabled
            and self._mode != "drive_ready"
        ):
            r2 = float(
                getattr(
                    input,
                    "r2",
                    0.0,
                )
            )
            l2 = float(
                getattr(
                    input,
                    "l2",
                    0.0,
                )
            )

            if r2 > 0.04 or l2 > 0.04:
                locomotion_rejection = (
                    "locomotion_requires_drive_ready"
                )

            by_role = {
                item.target_role: item.module_id
                for item in observation.assignments
            }

            # Explicit zero is required: the native action channel can retain
            # the previous wheel command if the locomotors simply disappear.
            actions = replace(
                actions,
                module_actions={
                    by_role["front_support"]: {
                        "vx": 0.0,
                        "vy": 0.0,
                        "yaw_rate": 0.0,
                    },
                    by_role["arm_lift"]: {
                        "vx": 0.0,
                        "vy": 0.0,
                        "yaw_rate": 0.0,
                    },
                },
            )

        if (
            observation is not None
            and safety.motion_enabled
        ):
            selected = self._selected(
                observation
            )

            if self._mode != "manipulation_ready":
                targets = ()

                rejected = (
                    "manual_shape_requires_manipulation_ready"
                    if not stick_neutral
                    else None
                )

            elif self._manual_transition:
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
                manual_rejection
                or rejected
            )

            actions = replace(
                actions,
                joint_targets=targets,
                manual_override=True,
            )

            selected_id = (
                selected.module_id
            )
            selected_role = (
                selected.target_role
            )

        else:
            selected_id = None
            selected_role = None

        if (
            observation is not None
            and safety.motion_enabled
            and self._mode_transition_pending
        ):
            transition_targets = (
                self._current_mode_joint_target(
                    observation
                )
            )

            if transition_targets:
                actions = replace(
                    actions,
                    joint_targets=transition_targets,
                    allow_joint_updates=True,
                    manual_override=True,
                )

        self._finish_mode_transition_if_ready()

        if not safety.motion_enabled:
            delivery = self.posture.step(
                replace(
                    actions,
                    joint_targets=(),
                    allow_joint_updates=False,
                ),
                now=now,
            )

            self._manual_transition = False
            self._manual_transition_module_id = None

        elif self._posture_reset_pending:
            delivery = self.posture.step(
                replace(
                    actions,
                    joint_targets=(),
                    allow_joint_updates=False,
                ),
                now=now,
            )

            if not self.posture.effective_targets():
                self._posture_reset_pending = False

        elif self._manual_transition:
            module = self._manual_transition_module_id

            if module is None:
                raise RuntimeError(
                    "MM8 manual transition has no outgoing module"
                )

            delivery = self.posture.retire_modules(
                (module,)
            )

            if (
                module
                not in self.posture.effective_targets()
            ):
                self._manual_transition = False
                self._manual_transition_module_id = None

        else:
            delivery = self.posture.step(
                actions,
                now=now,
            )

        if (
            self._mode_transition_pending
            and self._mode_transition_goal_id is None
            and delivery.goal is not None
        ):
            self._mode_transition_goal_id = (
                delivery.goal.goal_id
            )

        intent = dict(
            actions.intent
        )

        intent.update(
            control_mode="manual",
            mode=self._mode,
            mode_transition_pending=(
                self._mode_transition_pending
            ),
            mode_transition_behavior=(
                self._mode_transition_behavior
            ),
            selected_module_id=selected_id,
            selected_role=selected_role,
            manual_transition_pending=(
                self._manual_transition
            ),
            manual_rejection=(
                manual_rejection
            ),
            locomotion_rejection=(
                locomotion_rejection
            ),
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

        # Leaving an armed TELEOP epoch invalidates only the manual
        # integration origin.  Selection and physical held posture stay
        # untouched; the next PAN/TILT command recaptures measured state.
        if (
            not enabled
            and self._motion_enabled
        ):
            self._manual_key = None
            self._manual_target = None

        # Same validated authority-epoch rule as RC/Snake:
        # the native E-stop quarantines the old behavior identity.
        if (
            enabled
            and not self._motion_enabled
        ):
            self._command_id = (
                "teleop-mm8-"
                + uuid4().hex
            )

        self._motion_enabled = enabled

        envelope = None

        # Shape commands travel through primitive transport.
        # Only effective wheel locomotion belongs in /mssr/actions.
        if (
            actions.module_actions
            and enabled
        ):
            envelope = (
                self.transport.serialize(
                    actions,
                    stamp=time.time(),
                    command_id=(
                        self._command_id
                    ),
                    morphology=(
                        "mobile_manipulator8"
                    ),
                )
            )

        effective = {
            module: dict(command)
            for module, command
            in actions.module_actions.items()
        }

        for module, (
            joint,
            angle,
        ) in (
            self.posture
            .effective_targets()
            .items()
        ):
            effective.setdefault(
                module,
                {},
            )[
                joint
                + "_target_rad"
            ] = angle

        actions = replace(
            actions,
            module_actions=effective,
        )

        return MobileManipulatorTick(
            actions=actions,
            posture=delivery,
            envelope=envelope,
            effective_actions=effective,
        )
