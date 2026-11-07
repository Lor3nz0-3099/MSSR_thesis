"""Conservative C1 redundancy evidence, independent of physics and rewards.

Only newly admitted SET_PAN/SET_TILT goals can incur discrete counts. ALIGN,
DOCK (which already has failure feedback), UNDOCK and recovery are excluded.
Plain dispatcher continuous commands have no request identity: observing a
held command cannot prove repeated requests and never incurs a count.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
import hashlib
import json
import math

from drl.MARL.c1_structure import canonical_signature


def _finite(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _target(metadata):
    arguments = metadata.get("arguments", metadata.get("parameters", ()))
    if isinstance(arguments, Mapping):
        return _finite(arguments.get("angle_rad"))
    if not isinstance(arguments, (tuple, list)):
        return None
    for argument in arguments:
        if _field(argument, "name") == "angle_rad":
            return _finite(_field(argument, "value"))
        if isinstance(argument, (tuple, list)) and len(argument) == 2 and argument[0] == "angle_rad":
            return _finite(argument[1])
    return None


class RedundancyMonitor:
    """Return per-owner event counts and JSON-compatible diagnostics.

    Settings live in ``settings.values['redundancy']``; absent means disabled.
    ``already_charged_keys`` accepts goal IDs or emitted ``event_key`` strings.
    Continuous request evidence, when available, is an owner mapping containing
    ``{'request_id': unique_policy_request_id, 'command': SmoresCommand}``.
    A request ID must change only for independent policy requests, never merely
    for actuator refresh ticks. Without that provenance continuous is diagnostic.
    ``useful_progress`` must include task/goal approach as well as graph gains.
    """

    def __init__(self, owner_ids, settings):
        self.owner_ids = frozenset(owner_ids)
        self.config = getattr(settings, "values", {}).get("redundancy", {})
        self.enabled = bool(self.config.get("enabled", False))
        self.started = set()
        self.terminal = set()
        self.charged = set()
        self.targets = {}
        self.failed = {}
        self.history = {}
        self.previous_signature = None
        self.previous_frame = None
        self.continuous = {}
        self.cooldowns = {}
        self._topology_cache_key = None
        self._topology_cache_value = None

    def _number(self, name, default):
        value = _finite(self.config.get(name, default))
        return value if value is not None and value > 0 else default

    def _signature(self, frame):
        if not frame.evidence_complete or len(frame.modules) != len(self.owner_ids):
            return None
        quantum = self._number("joint_quantization_rad", .05)
        incident = {module.module_id: [] for module in frame.modules}
        connections = tuple(c for c in frame.connections if c.joint_valid)
        topology_key = (tuple(incident), tuple(sorted((c.module_a, c.connector_a, c.module_b, c.connector_b)
                                                    for c in connections)))
        if topology_key != self._topology_cache_key:
            self._topology_cache_value = canonical_signature(replace(frame, connections=connections))
            self._topology_cache_key = topology_key
        if self._topology_cache_value is None:
            return None
        for connection in connections:
            for owner, face in connection.endpoints:
                if owner in incident:
                    incident[owner].append(face)
        posture = []
        for module in frame.modules:
            joints = dict(module.joints)
            angles = [_finite(joints.get(f"{name}.position_rad")) for name in ("pan", "tilt")]
            if any(angle is None for angle in angles):
                return None
            posture.append((tuple(sorted(incident[module.module_id])), tuple(round(angle / quantum) for angle in angles)))
        return hashlib.sha256(json.dumps((self._topology_cache_value, sorted(posture)),
                                         separators=(",", ":")).encode()).hexdigest()

    def _unchanged(self, previous, current):
        if previous is None or not previous.evidence_complete or not current.evidence_complete:
            return False
        old = {module.module_id: module for module in previous.modules}
        if not old or set(old) != {module.module_id for module in current.modules}:
            return False
        for module in current.modules:
            before = old[module.module_id]
            if not all(_finite(v) is not None for v in (*before.position, *module.position,
                                                        *before.orientation, *module.orientation)):
                return False
            if math.dist(before.position, module.position) > self._number("motion_tolerance_m", .015):
                return False
            norm_a = math.sqrt(sum(v * v for v in before.orientation))
            norm_b = math.sqrt(sum(v * v for v in module.orientation))
            if not norm_a or not norm_b:
                return False
            cosine = min(1., abs(sum(a * b for a, b in zip(before.orientation, module.orientation))) / (norm_a * norm_b))
            if 2 * math.acos(cosine) > self._number("angular_motion_tolerance_rad", .05):
                return False
            before_joints, current_joints = dict(before.joints), dict(module.joints)
            for name in ("pan.position_rad", "tilt.position_rad"):
                a, b = _finite(before_joints.get(name)), _finite(current_joints.get(name))
                if a is None or b is None or abs(a - b) > self._number("angular_motion_tolerance_rad", .05):
                    return False
        return True

    def _eligible(self, metadata):
        if not isinstance(metadata, Mapping):
            return False
        owner = metadata.get("owner_module_id")
        scope = metadata.get("module_ids", (owner,))
        # A shared owner is insufficient to establish that every actuated
        # member of a group already reached its target.
        if not isinstance(scope, (tuple, list)) or len(scope) != 1 or scope[0] != owner:
            return False
        return (metadata.get("primitive") in ("set_pan", "set_tilt")
                and isinstance(metadata.get("owner_module_id"), str)
                and metadata.get("owner_module_id") in self.owner_ids
                and not metadata.get("recovery", False)
                and metadata.get("purpose") not in ("recovery", "reposition", "correction"))

    def step(self, frame, action_goals, *, useful_progress=False,
             continuous_commands=None, already_charged_keys=()):
        counts = Counter()
        diagnostic = dict(enabled=self.enabled, events=[], state_signature=None,
                          suppressed_existing_charge_count=0, missing_target_metadata_count=0,
                          availability=dict(action_goal_metadata=isinstance(action_goals, Mapping),
                                            continuous_payload=continuous_commands is not None,
                                            continuous_request_identity=False), noninstrumented=[])
        if not self.enabled:
            return {}, diagnostic
        goals = action_goals if isinstance(action_goals, Mapping) else {}
        ignored = set(already_charged_keys or ())
        signature = self._signature(frame)
        diagnostic["state_signature"] = signature
        if signature is None:
            diagnostic["noninstrumented"].append("canonical_measured_state")
            # A measurement gap cannot establish a closed state/action cycle
            # or that a repeated target was already reached before admission.
            self.history.clear()
            self.targets.clear()
            self.failed.clear()
            self.continuous.clear()
        if useful_progress:
            self.targets.clear()
            self.failed.clear()
            self.history.clear()
            self.continuous.clear()

        def charge(owner, reason, goal_id=None, *, cooldown=False, identity=None):
            event_key = f"redundancy:{reason}:{identity if identity is not None else goal_id}"
            if goal_id in self.charged or event_key in self.charged:
                return
            if goal_id in ignored or event_key in ignored:
                diagnostic["suppressed_existing_charge_count"] += 1
                self.charged.add(event_key)
                if goal_id is not None:
                    self.charged.add(goal_id)
                return
            cooldown_key = (owner, reason)
            if cooldown and frame.time_s - self.cooldowns.get(cooldown_key, -math.inf) < self._number("cooldown_s", 10.):
                return
            counts[owner] += 1
            self.cooldowns[cooldown_key] = frame.time_s
            self.charged.add(event_key)
            if goal_id is not None:
                self.charged.add(goal_id)
            diagnostic["events"].append(dict(event_key=event_key, owner=owner, reason=reason,
                                               goal_id=goal_id, time_s=frame.time_s))

        for goal_id, metadata in goals.items():
            if goal_id in self.started:
                continue
            self.started.add(goal_id)
            if not self._eligible(metadata):
                continue
            owner, action = metadata["owner_module_id"], metadata["primitive"]
            target = _target(metadata)
            if target is None:
                diagnostic["missing_target_metadata_count"] += 1
                continue
            key = (owner, action, target)
            if signature is None:
                continue
            # The previous sample precedes native execution, unlike this sample
            # which may already reflect a newly completed short primitive.
            start = self.previous_frame or frame
            module = next((m for m in start.modules if m.module_id == owner), None)
            position = _finite(dict(module.joints).get(action.removeprefix("set_") + ".position_rad")) if module else None
            previous_target = self.targets.get((owner, action))
            tolerance = self._number("joint_target_tolerance_rad", math.radians(2.))
            if (not useful_progress and position is not None and abs(position - target) <= tolerance
                    and previous_target is not None and abs(previous_target - target) <= tolerance):
                charge(owner, "repeated_reached_target", goal_id)
            elif (not useful_progress and self.config.get("failed_non_align_repeat", False)
                  and key in self.failed and self.failed[key][0] == signature
                  and self._unchanged(self.failed[key][1], frame)):
                charge(owner, "failed_non_align_repeat", goal_id)
            self.targets[(owner, action)] = target

        # Cycle attribution requires a fresh, successful eligible primitive.
        window = self._number("cycle_window_s", 60.)
        self.history = {key: value for key, value in self.history.items() if frame.time_s - value[0] <= window}
        for event in frame.primitive_events:
            if event.state not in ("failed", "succeeded", "canceled", "rejected") or event.goal_id in self.terminal:
                continue
            self.terminal.add(event.goal_id)
            metadata = goals.get(event.goal_id)
            if (not self._eligible(metadata) or signature is None
                    or event.action != metadata["primitive"]):
                continue
            owner = metadata["owner_module_id"]
            target = _target(metadata)
            key = (owner, metadata["primitive"], target)
            if event.state == "failed" and target is not None:
                self.failed[key] = (signature, frame)
            elif event.state == "succeeded":
                self.failed.pop(key, None)
                if (not useful_progress and signature != self.previous_signature and signature in self.history
                        and self._unchanged(self.history[signature][1], frame)):
                    charge(owner, "equivalent_state_cycle", event.goal_id, cooldown=True)
        if signature is not None:
            self.history[signature] = (frame.time_s, frame)

        commands = continuous_commands if isinstance(continuous_commands, Mapping) else {}
        current_owners = set()
        for owner, payload in commands.items():
            if owner not in self.owner_ids:
                continue
            request_id = _field(payload, "request_id")
            independent_request = request_id is not None
            if independent_request:
                diagnostic["availability"]["continuous_request_identity"] = True
                command = _field(payload, "command")
            else:
                # Physical refresh ticks preserve evidence, but never count
                # as independent policy requests. Still check measured motion.
                item = self.continuous.get(owner)
                if item is None:
                    continue
                command = payload
                request_id = item['request_id']
            if command is None or _field(payload, "recovery", False) or _field(payload, "purpose") in ("recovery", "reposition", "correction"):
                continue
            values = tuple(_finite(_field(command, name, 0.)) for name in
                           ("linear_x_m_s", "angular_z_rad_s", "pan_velocity_rad_s"))
            if any(value is None for value in values) or not any(abs(value) > 1e-6 for value in values) or signature is None:
                continue
            current_owners.add(owner)
            item = self.continuous.get(owner)
            if item is None or item["command"] != values or item["signature"] != signature or not self._unchanged(item["frame"], frame) or useful_progress:
                item = dict(command=values, signature=signature, frame=frame, start_s=frame.time_s,
                            request_id=request_id, request_count=int(independent_request))
                self.continuous[owner] = item
            elif independent_request and request_id != item["request_id"]:
                item["request_count"] += 1
                item["request_id"] = request_id
                if item["request_count"] >= 3 and frame.time_s - item["start_s"] >= self._number("continuous_window_s", 10.):
                    charge(owner, "sustained_ineffectual_requests", cooldown=True, identity=f"{owner}:{request_id}")
        self.continuous = {owner: item for owner, item in self.continuous.items() if owner in current_owners}
        if not diagnostic["availability"]["continuous_request_identity"]:
            diagnostic["noninstrumented"].append("continuous_request_identity")
        if diagnostic["missing_target_metadata_count"]:
            diagnostic["noninstrumented"].append("target_metadata")
        self.previous_signature = signature
        self.previous_frame = frame
        return dict(counts), diagnostic
