"""Existing action envelope and acknowledged RC posture primitive delivery."""
from dataclasses import dataclass, replace
import json
import time
from uuid import uuid4

from mssr_expert.execution.primitive_protocol import PrimitiveGoalRequest, parse_primitive_statuses
from mssr_expert.graph.serialization import attributed_graph_from_dict
from mssr_expert.teleop.rc_car import RcCarObservation, RcCarTeleopController, finite


class ActionTransport:
    topic = "/mssr/actions"

    def serialize(self, result, *, stamp, command_id, morphology="rc_car8"):
        if not finite(stamp):
            raise ValueError("Action timestamp must be finite")
        # Validate intent as well: diagnostics must never contaminate the
        # existing finite-only combined action stream.
        json.dumps(result.intent, allow_nan=False)
        return json.dumps({
            "schema_version": "mssr.actions.v2", "stamp": stamp, "stage_id": 0,
            "task_type": "morphology_behavior", "reset": False,
            "locomotion": result.module_actions, "magnetic": [],
            "expert": {"debug": {"command_id": command_id, "message": f"{morphology} teleoperation"}, "fsm_state": "TELEOP",
                       "active_primitive": morphology, "primitive_params": {},
                       "module_roles": result.module_roles,
                       "pan_traction_module_ids": sorted(module for module, role in result.module_roles.items()
                                                         if role.startswith("wheel_")),
                       "task_metrics": {"progress": 0.0, "phase": f"{morphology}_teleop"},
                       "success": False, "done": False}}, allow_nan=False)


@dataclass(frozen=True)
class PostureDelivery:
    goal: object = None
    cancel_goal_id: str | None = None
    blocked_module_ids: tuple = ()
    cancel_goal_ids: tuple = ()


class RcCarPostureTransport:
    """One unacknowledged file write at a time; PAN waits for TILT retirement.

    Each batch freezes the coordinated target snapshot. Only an Isaac status
    admits the next goal or retires a canceled goal; wall time cannot grant
    motion while a primitive may still own the module.
    """
    def __init__(self, *, retry_s=0.25, timeout_s=10.0):
        if any(not finite(value) or value <= 0 for value in (retry_s, timeout_s)):
            raise ValueError("Posture timeout and retry must be positive finite values")
        self.retry_s, self.timeout_s = retry_s, timeout_s
        self._active = {}
        self._retained = {}
        self._queue = []
        self._pending = None
        self._last_send = None
        self._canceling = False
        self._cancel_pending = None

    def observe(self, payload):
        for goal_id, status in parse_primitive_statuses(payload).items():
            goal = self._active.get(goal_id)
            if goal is None or status.primitive != goal.primitive or status.module_ids != goal.module_ids:
                continue
            # Native retries can be rejected as DUPLICATE_GOAL_ID while the
            # original goal still owns the motors. This acknowledges delivery,
            # but cannot prove retirement or authorize PAN on that module.
            duplicate = status.state == "rejected" and status.code == "DUPLICATE_GOAL_ID"
            if not status.failed or duplicate:
                self._retained.pop(goal.module_ids[0], None)
            if status.terminal and not duplicate:
                self._active.pop(goal_id)
                if status.state == "succeeded":
                    # A completion ACK can leave a loaded servo moving toward
                    # its retained target. Keep its identity for a safe yield.
                    self._retained[goal.module_ids[0]] = goal
                if self._cancel_pending == goal_id:
                    self._cancel_pending = None
                if status.failed:
                    self._canceling = True
                    self._queue.clear()
            if self._pending == goal_id:
                self._pending = None

    def step(self, result, *, now):
        if not finite(now):
            raise ValueError("Posture receipt time must be finite")
        pan_modules = {module for module, command in result.module_actions.items()
                       if abs(command.get("pan_rate_rad_s", 0)) > 1e-12}
        owned = {goal.module_ids[0] for goal in self._active.values()}
        queued_modules = {goal.module_ids[0] for goal in self._queue}
        if not result.allow_joint_updates or (owned | queued_modules | set(self._retained)).intersection(pan_modules):
            self._canceling = True
            self._queue.clear()
        if self._canceling:
            self._active.update((goal.goal_id, goal) for goal in self._retained.values())
            self._retained.clear()
            owned = {goal.module_ids[0] for goal in self._active.values()}
            if self._active:
                goal_id = self._cancel_pending or next(iter(self._active))
                self._cancel_pending = goal_id
                return PostureDelivery(cancel_goal_id=goal_id, blocked_module_ids=tuple(sorted(owned)))
            self._canceling = False
            self._pending = None
        if self._pending is not None:
            retry = self._last_send is None or now - self._last_send >= self.retry_s
            if retry:
                self._last_send = now
            return PostureDelivery(goal=self._active[self._pending] if retry else None,
                                   blocked_module_ids=tuple(sorted(owned)))
        if not self._active and not self._queue and result.allow_joint_updates and result.joint_targets:
            group = "teleop-rc-" + uuid4().hex
            chassis = sorted(module for module, role in result.module_roles.items() if not role.startswith("wheel_"))
            self._queue = [PrimitiveGoalRequest(
                goal_id=f"{group}-{index}", primitive="set_tilt", module_ids=(target.module_id,),
                parameters={"angle_rad": target.angle_rad,
                            **({"tolerance_rad": target.tolerance_rad} if target.tolerance_rad is not None else {}),
                            "coordination_group": group, "coordination_size": len(result.joint_targets),
                            "retain_reached_on_interrupt": True,
                            "structural_hold_module_ids": chassis}, timeout_s=self.timeout_s)
                for index, target in enumerate(result.joint_targets)]
        if self._queue:
            goal = self._queue.pop(0)
            self._active[goal.goal_id] = goal
            self._pending = goal.goal_id
            self._last_send = now
            owned.add(goal.module_ids[0])
            return PostureDelivery(goal=goal, blocked_module_ids=tuple(sorted(owned)))
        return PostureDelivery(blocked_module_ids=tuple(sorted(owned)))


@dataclass(frozen=True)
class RcCarTick:
    actions: object
    posture: PostureDelivery
    envelope: str | None


class RcCarRuntime:
    """ROS-independent T3 boundary: live graph, permissions and delivered actions."""
    def __init__(self, library, target_graph, *, geometry, observation_timeout_s=0.5,
                 height_rate_m_s=0.005, max_dt_s=0.1):
        if not finite(observation_timeout_s) or observation_timeout_s <= 0:
            raise ValueError("RC observation timeout must be positive and finite")
        self.controller = RcCarTeleopController(library, geometry=geometry,
                                               height_rate_m_s=height_rate_m_s, max_dt_s=max_dt_s)
        self.posture = RcCarPostureTransport()
        self.transport = ActionTransport()
        self.target_graph = target_graph
        self.observation_timeout_s = observation_timeout_s
        self._observation = None
        self._received_at = None
        self._last_graph = None
        self._previous_tick = None
        self._motion_enabled = False
        self._command_id = "teleop-rc-" + uuid4().hex

    def observe_graph(self, payload, *, now):
        try:
            if not finite(now):
                return False
            graph = attributed_graph_from_dict(payload)
            if self._last_graph is not None and graph.stamp <= self._last_graph.stamp:
                if graph == self._last_graph:
                    # The file bridge republishes unchanged files. Receipt of
                    # cached state cannot renew the physical observation lease.
                    return self._observation is not None
                raise ValueError("RC graph timestamp did not advance")
            self._observation = RcCarObservation.from_graph(graph, self.target_graph)
            if self._observation is not None:
                self._last_graph = graph
        except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
            self._observation = None
        if self._observation is None:
            self._received_at = None
            self.controller.recapture = True
            return False
        self._received_at = now
        return True

    def _fresh_observation(self, now):
        if self._received_at is None or not 0 <= now - self._received_at <= self.observation_timeout_s:
            self.controller.recapture = True
            return None
        return self._observation

    @property
    def latest_graph(self):
        """Return the latest accepted physical graph for dataset recording."""
        return self._last_graph

    def topology(self, now):
        return "rc_car8" if self._fresh_observation(now) is not None else None

    def observe_status(self, payload):
        self.posture.observe(payload)

    def step(self, input, *, safety, now):
        if not finite(now):
            raise ValueError("RC tick time must be finite")
        observation = self._fresh_observation(now)
        dt = 0.0 if self._previous_tick is None else now - self._previous_tick
        self._previous_tick = now
        if "home" in getattr(input, "command_events", ()) and safety.allow_joint_updates:
            self.controller.home()
        actions = self.controller.step(input, observation, dt, safety=safety)
        delivery = self.posture.step(actions, now=now)
        # The existing Isaac primitive owns these internal motors until a
        # terminal status. Publish the actual gated commands, not the desired
        # propulsion intent, so downstream action consumers see effective data.
        commands = {module: dict(command) for module, command in actions.module_actions.items()}
        for module in delivery.blocked_module_ids:
            if module in commands:
                commands[module]["pan_rate_rad_s"] = 0.0
        actions = replace(actions, module_actions=commands)
        enabled = safety.motion_enabled and observation is not None
        if enabled and not self._motion_enabled:
            self._command_id = "teleop-rc-" + uuid4().hex
        self._motion_enabled = enabled
        envelope = self.transport.serialize(actions, stamp=time.time(), command_id=self._command_id) if commands else None
        return RcCarTick(actions, delivery, envelope)
