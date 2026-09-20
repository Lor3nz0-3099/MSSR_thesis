"""Head-led, ROS-independent Snake8 teleoperation and actuator delivery.

The trajectory is a body-relative polyline sampled at CAD module spacing.
Only the physical graph supplies module identities and available actuators.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import time
from types import SimpleNamespace
from uuid import uuid4

from mssr_expert.behaviors.morphology_locomotion import (
    _planar_body_forward, coherent_planar_train_commands, validate_locomotion_dofs,
)
from mssr_expert.execution.primitive_protocol import PrimitiveGoalRequest, parse_primitive_statuses
from mssr_expert.graph.serialization import attributed_graph_from_dict
from mssr_expert.teleop.action_transport import ActionTransport, PostureDelivery
from mssr_expert.teleop.rc_car import RcCarObservation, finite
from mssr_expert.teleop.safety import SafetyDecision


def _distance(a, b):
    return math.dist(a, b)


def _unit(a, b):
    delta = tuple(y - x for x, y in zip(a, b))
    length = math.sqrt(sum(v * v for v in delta))
    return tuple(v / length for v in delta) if length > 1e-9 else (1.0, 0.0, 0.0)


def _angle_delta(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def _angles(before, at, after):
    incoming = _unit(before, at)
    outgoing = _unit(at, after)
    yaw = _angle_delta(math.atan2(outgoing[1], outgoing[0]),
                       math.atan2(incoming[1], incoming[0]))
    pitch = _angle_delta(math.atan2(outgoing[2], math.hypot(*outgoing[:2])),
                         math.atan2(incoming[2], math.hypot(*incoming[:2])))
    return yaw, pitch


def _pose_yaw(node):
    orientation = node.attributes["pose"]["orientation_xyzw"]
    if len(orientation) != 4 or not all(finite(v) for v in orientation):
        raise ValueError("Snake pose orientation is invalid")
    x, y, z, w = (float(v) for v in orientation)
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm < 1e-9:
        raise ValueError("Snake pose orientation is zero")
    x, y, z, w = (v / norm for v in (x, y, z, w))
    return math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def _sample_back(path, distance):
    """Arc-length interpolation backwards from the current head."""
    remaining = distance
    for i in range(len(path) - 1, 0, -1):
        length = _distance(path[i], path[i - 1])
        if length >= remaining and length > 1e-9:
            fraction = remaining / length
            return tuple(path[i][k] + fraction * (path[i - 1][k] - path[i][k]) for k in range(3))
        remaining -= length
    return path[0]


def _retreat_head(path, distance, spacing):
    """Walk the head back along its route, keeping a full body behind it."""
    length = sum(_distance(a, b) for a, b in zip(path, path[1:]))
    required = distance + 8 * spacing
    if length < required:
        backward = _unit(path[1], path[0])
        path.insert(0, tuple(path[0][i] + backward[i] * (required - length)
                             for i in range(3)))
    remaining = distance
    while remaining > 1e-9:
        segment = _distance(path[-2], path[-1])
        if segment <= remaining + 1e-9:
            path.pop()
            remaining -= segment
        else:
            path[-1] = tuple(path[-1][i] + remaining / segment *
                             (path[-2][i] - path[-1][i]) for i in range(3))
            remaining = 0.0


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
            positions = tuple(tuple(float(v) for v in
                                    nodes[item.module_id].attributes["pose"]["position"])
                              for item in matched.assignments)
            yaws = tuple(_pose_yaw(nodes[item.module_id]) for item in matched.assignments)
            if any(len(point) != 3 or not all(finite(v) for v in point) for point in positions):
                return None
            dofs = {(d.module_id, d.name): d for d in matched.inventory.dofs}
            for item in matched.assignments:
                for name in ("left_wheel", "right_wheel"):
                    if not dofs[item.module_id, name].can_locomote:
                        return None
                for name in ("tilt",):
                    dof = dofs[item.module_id, name]
                    if not dof.can_shape or not finite(dof.position_rad):
                        return None
            return cls(graph, matched.assignments, matched.inventory, positions, yaws[3])
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class SnakeActions:
    module_actions: dict = field(default_factory=dict)
    joint_targets: tuple = ()  # (physical ID, pan|tilt, absolute target, measured angle)
    intent: dict = field(default_factory=dict)
    module_roles: dict = field(default_factory=dict)
    allow_joint_updates: bool = False
    manual_override: bool = False


class SnakeTeleopController:
    def __init__(self, library, *, geometry, vertical_rate_m_s=0.025,
                 recenter_rate_m_s=0.01,
                 max_head_offset_m=0.2, max_curvature_rad=0.8,
                 min_head_height_m=0.02, max_head_height_m=0.3,
                 joint_rate_rad_s=0.5,
                 joint_deadband_rad=0.025,
                 max_dt_s=0.1):
        values = (vertical_rate_m_s, recenter_rate_m_s,
                  max_head_offset_m, max_curvature_rad, joint_rate_rad_s,
                  joint_deadband_rad, max_dt_s)
        if any(not finite(v) or v <= 0 for v in values):
            raise ValueError("Snake control rates and bounds must be positive and finite")
        if (not finite(min_head_height_m) or not finite(max_head_height_m)
                or not 0 <= min_head_height_m < max_head_height_m):
            raise ValueError("Snake head height limits must be ordered")
        self.library = library
        self.spacing = float(geometry.top_to_bottom_spacing_m)
        if not finite(self.spacing) or self.spacing <= 0:
            raise ValueError("Snake module spacing must be positive")
        self.vertical_rate = vertical_rate_m_s
        self.recenter_rate = recenter_rate_m_s
        self.max_offset = max_head_offset_m
        self.max_curvature = max_curvature_rad
        self.joint_rate = joint_rate_rad_s
        self.min_head_height = min_head_height_m
        self.max_head_height = max_head_height_m
        self.deadband = joint_deadband_rad
        self.max_dt = max_dt_s
        self._assignments = ()
        self._path = []
        self._baseline = ()
        self._joint_baseline = {}
        self._joint_target = {}
        self._vertical_offset = 0.0
        self._home = False
        self.recapture = True
        self._root = None
        self._root_yaw = None

    def home(self):
        self._home = True

    def _capture(self, observation):
        self._assignments = observation.assignments
        self._baseline = observation.positions
        self._path = list(observation.positions)
        dofs = {(d.module_id, d.name): d for d in observation.inventory.dofs}
        self._joint_baseline = {(a.module_id, "tilt"): dofs[a.module_id, "tilt"].position_rad
                                for a in observation.assignments}
        self._joint_target = dict(self._joint_baseline)
        self._vertical_offset = 0.0
        self._home = False
        self.recapture = False
        self._root = observation.positions[3]
        self._root_yaw = observation.root_yaw

    def _follow_body_frame(self, observation):
        root = observation.positions[3]
        yaw = observation.root_yaw
        rotation = _angle_delta(yaw, self._root_yaw)
        c, s = math.cos(rotation), math.sin(rotation)
        if root != self._root or abs(rotation) > 1e-9:
            self._path = [(root[0] + c*(point[0]-self._root[0]) - s*(point[1]-self._root[1]),
                           root[1] + s*(point[0]-self._root[0]) + c*(point[1]-self._root[1]),
                           root[2] + point[2] - self._root[2])
                          for point in self._path]
        self._root, self._root_yaw = root, yaw

    def step(self, input, observation, dt, *, safety=SafetyDecision("NONE", False, True, False)):
        if not safety.motion_enabled or observation is None:
            self.recapture = True
            self._home = False
            return SnakeActions()
        if observation.assignments != self._assignments or self.recapture:
            self._capture(observation)
        else:
            self._follow_body_frame(observation)
        values = {name: getattr(input, name, None) for name in ("r2", "l2", "right_x", "right_y")}
        if not finite(dt) or dt < 0 or any(not finite(v) or not (0 <= v <= 1 if k in ("r2", "l2") else -1 <= v <= 1)
                                             for k, v in values.items()):
            self.recapture = True
            return SnakeActions()
        dt = min(dt, self.max_dt)
        forward = _unit(self._path[-2], self._path[-1])
        if values["right_y"]:
            self._home = False
        previous_offset = self._vertical_offset
        if self._home:
            self._vertical_offset += max(-self.recenter_rate * dt,
                                         min(self.recenter_rate * dt, -self._vertical_offset))
            if abs(self._vertical_offset) < 1e-9:
                self._home = False
        else:
            self._vertical_offset = max(-self.max_offset, min(self.max_offset,
                self._vertical_offset + values["right_y"] * self.vertical_rate * dt))
        drive = self.library._profile("snake8")["drive"]
        speed = (values["r2"] - values["l2"]) * float(drive["max_linear_m_s"])
        # Head progression follows the current tangent. The vertical target
        # is held when the right stick is released.
        if speed < 0:
            _retreat_head(self._path, -speed * dt, self.spacing)
        old_head = self._path[-1]
        vertical_step = self._vertical_offset - previous_offset
        forward_step = max(0.0, speed * dt)
        candidate = (old_head[0] + forward[0] * forward_step,
                     old_head[1] + forward[1] * forward_step,
                     max(self.min_head_height, min(self.max_head_height,
                         old_head[2] + forward[2] * forward_step + vertical_step)))
        if _distance(candidate, old_head) > 1e-9:
            if speed < 0:
                self._path[-1] = candidate
            else:
                self._path.append(candidate)
        # Keep slightly more than seven spacings so interpolation remains
        # well-defined when the head advances for a long episode.
        while len(self._path) > 9 and sum(_distance(self._path[i], self._path[i-1])
                                           for i in range(1, len(self._path)-1)) > 8 * self.spacing:
            self._path.pop(0)
        backbone = tuple(_sample_back(self._path, self.spacing * (7 - i)) for i in range(8))
        roles = {a.module_id: a.target_role for a in observation.assignments}
        dofs = {(d.module_id, d.name): d for d in observation.inventory.dofs}
        targets = []
        for i in range(1, 7):
            module = observation.assignments[i].module_id
            base_tilt = _angles(*self._baseline[i-1:i+2])[1]
            tilt = _angles(*backbone[i-1:i+2])[1]
            target = self._joint_baseline[module, "tilt"] + max(
                -self.max_curvature, min(self.max_curvature, _angle_delta(tilt, base_tilt)))
            dof = dofs[module, "tilt"]
            if finite(dof.lower_limit_rad):
                target = max(target, dof.lower_limit_rad)
            if finite(dof.upper_limit_rad):
                target = min(target, dof.upper_limit_rad)
            key = (module, "tilt")
            error = max(abs(target - dof.position_rad),
                        abs(self._joint_target[key] - dof.position_rad))
            if error > self.deadband:
                previous = self._joint_target[key]
                delta = max(-self.joint_rate * dt,
                            min(self.joint_rate * dt, target - previous))
                self._joint_target[key] = previous + delta
                if abs(self._joint_target[key] - dof.position_rad) > self.deadband:
                    targets.append((module, "tilt", self._joint_target[key], dof.position_rad))
        commands = self.library.drive_commands("snake8", observation.assignments,
                                               linear_m_s=speed, yaw_rate_rad_s=0.0)
        commands = coherent_planar_train_commands(observation.graph, commands)
        # The generic train helper makes all modules agree with its first
        # locomotor. Orient that common direction along the current head
        # tangent, including when assembly flipped the tail's local +X axis.
        first = observation.assignments[0].module_id
        local_forward = _planar_body_forward(observation.graph.node_by_id()[first].attributes)
        tangent = _unit(self._path[-2], self._path[-1])
        if local_forward is not None and (local_forward[0]*tangent[0] +
                                          local_forward[1]*tangent[1]) < 0:
            for command in commands.values():
                command["vx"] = -command["vx"]
        validate_locomotion_dofs(commands, observation.inventory)
        intent = {"head_target_m": list(self._path[-1]),
                  "backbone_m": [list(point) for point in backbone],
                  "head_tangent": list(_unit(backbone[-2], backbone[-1])),
                  "longitudinal_m_s": speed, "lateral_offset_m": 0.0,
                  "vertical_offset_m": self._vertical_offset}
        return SnakeActions(commands, tuple(targets), intent, roles, safety.allow_joint_updates)


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
        if any(joint not in ("pan", "tilt") or
               (joint == "pan" and not actions.manual_override)
               for _, joint, _, _ in actions.joint_targets):
            raise ValueError("Snake automatic posture accepts TILT targets only")
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
    def __init__(self, library, target_graph, *, geometry, observation_timeout_s=0.5, **controller_config):
        if not finite(observation_timeout_s) or observation_timeout_s <= 0:
            raise ValueError("Snake observation timeout must be positive")
        self.controller = SnakeTeleopController(library, geometry=geometry, **controller_config)
        self.posture = SnakePostureTransport()
        self.transport = ActionTransport()
        self.target_graph = target_graph
        self.observation_timeout_s = observation_timeout_s
        self._observation = None
        self._received_at = None
        self._last_graph = None
        self._previous_tick = None
        self._command_id = "teleop-snake-" + uuid4().hex
        self._manual_mode = False
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
            if self._last_graph is not None and graph.stamp <= self._last_graph.stamp:
                if graph == self._last_graph:
                    return self._observation is not None
                raise ValueError("Snake graph timestamp did not advance")
            self._observation = SnakeObservation.from_graph(graph, self.target_graph)
            if self._observation is not None:
                assignment_ids = tuple(item.module_id for item in self._observation.assignments)
                if self._assignment_ids and assignment_ids != self._assignment_ids:
                    self._manual_mode = False
                    self._manual_transition = True
                    self._manual_key = self._manual_target = None
                    self.controller.recapture = True
                self._assignment_ids = assignment_ids
                self._last_graph = graph
                self._received_at = now
                return True
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            pass
        self._observation = None
        self._received_at = None
        self.controller.recapture = True
        self._manual_mode = False
        self._manual_transition = True
        self._manual_key = self._manual_target = None
        return False

    @property
    def latest_graph(self):
        return self._last_graph

    def topology(self, now):
        return "snake8" if (self._observation is not None and
                            self._received_at is not None and
                            0 <= now - self._received_at <= self.observation_timeout_s) else None

    def observe_status(self, payload):
        self.posture.observe(payload)

    def _manual_joint_target(self, input, observation, dt):
        x, y = float(input.right_x), float(input.right_y)
        if not all(finite(value) and -1 <= value <= 1 for value in (x, y)):
            return (), "invalid_stick"
        if abs(x) > 0.1 and abs(y) > 0.1:
            return (), "one_joint_axis_at_a_time"
        if max(abs(x), abs(y)) <= 0.1:
            return (), None
        joint, value = ("pan", x) if abs(x) > 0.1 else ("tilt", y)
        selected = tuple(reversed(observation.assignments))[self._manual_index]
        dof = next((item for item in observation.inventory.dofs
                    if item.module_id == selected.module_id and item.name == joint), None)
        if dof is None or not dof.can_shape or not finite(dof.position_rad):
            return (), "joint_unavailable"
        key = (selected.module_id, joint)
        if key != self._manual_key:
            self._manual_key = key
            self._manual_target = dof.position_rad
        step = value * 0.4 * max(0.0, min(dt, 0.1))
        target = self._manual_target + step
        if finite(dof.lower_limit_rad):
            target = max(target, dof.lower_limit_rad)
        if finite(dof.upper_limit_rad):
            target = min(target, dof.upper_limit_rad)
        self._manual_target = target
        if abs(target - dof.position_rad) <= self.controller.deadband:
            return (), None
        return ((selected.module_id, joint, target, dof.position_rad),), None

    def step(self, input, *, safety, now):
        if not finite(now):
            raise ValueError("Snake tick time must be finite")
        fresh = (self._received_at is not None and
                 0 <= now - self._received_at <= self.observation_timeout_s)
        observation = self._observation if fresh else None
        dt = 0.0 if self._previous_tick is None else now - self._previous_tick
        self._previous_tick = now
        events = tuple(getattr(input, "command_events", ()))
        stick_neutral = (abs(float(getattr(input, "right_x", 0.0))) < 0.05 and
                         abs(float(getattr(input, "right_y", 0.0))) < 0.05)
        manual_rejection = None
        if observation is not None and safety.motion_enabled:
            if "override" in events:
                if stick_neutral:
                    self._manual_mode = not self._manual_mode
                    self._manual_index = 0 if self._manual_mode else self._manual_index
                    self._manual_key = self._manual_target = None
                    self._manual_transition = True
                    self.controller.recapture = True
                else:
                    manual_rejection = "release_right_stick_before_mode_switch"
            elif self._manual_mode and ("next_module" in events or "previous_module" in events):
                if not stick_neutral:
                    manual_rejection = "release_right_stick_before_module_switch"
                elif "next_module" in events and "previous_module" in events:
                    manual_rejection = "ambiguous_module_selection"
                else:
                    count = len(observation.assignments)
                    delta = 1 if "next_module" in events else -1
                    self._manual_index = (self._manual_index + delta) % count
                    self._manual_key = self._manual_target = None
                    self._manual_transition = True
        if "home" in events and safety.allow_joint_updates and not self._manual_mode:
            self.controller.home()
        wheel_input = (SimpleNamespace(r2=input.r2, l2=input.l2,
                                       right_x=0.0, right_y=0.0)
                       if self._manual_mode else input)
        actions = self.controller.step(wheel_input, observation, dt, safety=safety)
        if self._manual_mode and observation is not None and safety.motion_enabled:
            selected = tuple(reversed(observation.assignments))[self._manual_index]
            targets, rejected = (self._manual_joint_target(input, observation, dt)
                                 if not self._manual_transition else ((), None))
            manual_rejection = manual_rejection or rejected
            actions = replace(actions, joint_targets=targets, manual_override=True)
            selected_id, selected_role = selected.module_id, selected.target_role
        else:
            selected_id = selected_role = None
        if self._manual_transition or not safety.motion_enabled:
            delivery = self.posture.step(replace(actions, joint_targets=(),
                                                 allow_joint_updates=False), now=now)
            if self._manual_transition and not self.posture.effective_targets():
                self._manual_transition = False
                self.controller.recapture = True
        else:
            delivery = self.posture.step(actions, now=now)
        intent = dict(actions.intent)
        intent.update(control_mode="single_module" if self._manual_mode else "head_led",
                      selected_module_id=selected_id, selected_role=selected_role,
                      manual_transition_pending=self._manual_transition,
                      manual_rejection=manual_rejection)
        actions = replace(actions, intent=intent)
        envelope = None
        if actions.module_actions and safety.motion_enabled and fresh:
            envelope = self.transport.serialize(actions, stamp=time.time(),
                                                command_id=self._command_id,
                                                morphology="snake8")
        effective = {module: dict(command) for module, command in actions.module_actions.items()}
        for module, (joint, angle) in self.posture.effective_targets().items():
            effective.setdefault(module, {})[joint + "_target_rad"] = angle
        return SnakeTick(replace(actions, module_actions=effective), delivery, envelope, effective)
