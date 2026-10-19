"""Pure-data SMORES boundary: current physical inputs, independently joined labels.

Config authority: smores_ep/config/geometry.py (calibrated radius 0.03150 m,
not the older backend manifest's 0.03106), config/physics.py, and
isaac/physics_asset.py docking frames (+X normal, +Y tangent). Config port
poses describe ZERO joint angles. Articulated nominal geometry is invalid as
a current observation when measured geometry is missing. B1 does not select
a global graph frame or implement kinematics, dispatch, lifecycle or feasibility.

Decoded mappings follow preprocessing's command/module_ids/parameters data
convention. They are not ROS messages. Evidence reads only the exact referenced
primitive_goal as label metadata, never the source graph or its post-state.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .bc_v1_contract import (
    ActionChoice, ConnectorRef, ConnectorState, ConnectionState, CoordinateState,
    LabelEvidence, ModuleRef, ModuleState, NumericFeature, ParameterValue,
    ResourceRef, ResourceState, ResourceTarget, RobotAdapter, RobotSpec, StateGraph,
    StructuralTarget, SupervisionBundle, validate_robot_spec,
)


_INTERNAL_COMMANDS = {
    "pan_velocity": ("pan_velocity", "rate_rad_s"),
    "tilt_velocity": ("tilt_velocity", "rate_rad_s"),
    "pan_target": ("set_pan", "angle_rad"),
    "tilt_target": ("set_tilt", "angle_rad"),
    "pan_delta": ("rotate_pan_by", "delta_rad"),
    "tilt_delta": ("rotate_tilt_by", "delta_rad"),
}
_PAIR_ACTIONS = {"align_faces", "dock", "undock"}
_SCOPE_KEYS = {"module_ids", "participants", "peer_module_id"}
_LIFECYCLE_KEYS = {"execution_phase"}


class LabelEvidenceRequired(ValueError):
    """A canonical label is incomplete; explicitly request a label-only join."""


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _id(value):
    if not isinstance(value, str) or not value or any(c.isspace() for c in value):
        raise ValueError(f"Invalid reference ID {value!r}")
    return value


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Expected finite numeric value, got {value!r}")
    return float(value)


def _vector(value, size):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"Expected vector of size {size}")
    return tuple(_number(v) for v in value)


def _quaternion(value):
    q = _vector(value, 4)
    norm = math.hypot(*q)
    if norm < 1e-12:
        raise ValueError("Zero orientation quaternion")
    return tuple(v / norm for v in q)


def _rotate(q, vector):
    # Unit quaternion xyzw; uses no planar/yaw approximation.
    x, y, z, w = q
    vx, vy, vz = vector
    tx, ty, tz = 2*(y*vz-z*vy), 2*(z*vx-x*vz), 2*(x*vy-y*vx)
    return (vx+w*tx+y*tz-z*ty, vy+w*ty+z*tx-x*tz, vz+w*tz+x*ty-y*tx)


def _module_ids(value):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("Missing nonempty module scope")
    ids = tuple(_id(v) for v in value)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate modules in scope")
    return ids


@dataclass
class _SourceIndex:
    signature: tuple[int, int, int, int]
    sha256: str
    offsets: list[int]


_SOURCE_INDEX: dict[Path, _SourceIndex] = {}


def _file_signature(path):
    stat = path.stat()
    # atime may change on a read. It is not a source-content identity.
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _source_index(path):
    """Hash/index once per unchanged file; cache offsets, not huge source graphs."""
    signature = _file_signature(path)
    cached = _SOURCE_INDEX.get(path)
    if cached is not None and cached.signature == signature:
        return cached
    digest = hashlib.sha256()
    offsets, offset = [], 0
    with path.open("rb") as stream:
        for line in stream:
            offsets.append(offset)
            digest.update(line)
            offset += len(line)
    if _file_signature(path) != signature:
        raise ValueError(f"Source changed while indexing: {path}")
    indexed = _SourceIndex(signature, digest.hexdigest(), offsets)
    _SOURCE_INDEX[path] = indexed
    return indexed


def _source_goal(path, record, expected_hash=None):
    if type(record) is not int or record < 1:
        raise ValueError("Source record must be a positive, one-based line index")
    indexed = _source_index(path)
    if expected_hash is not None and indexed.sha256 != expected_hash:
        raise ValueError(f"Source integrity hash changed: {path}")
    if record > len(indexed.offsets):
        raise ValueError(f"Source record {record} out of bounds: {path}")
    with path.open("rb") as stream:
        stream.seek(indexed.offsets[record - 1])
        row = json.loads(stream.readline())
    action = _mapping(row, "source record").get("expert_action") or {}
    goal = _mapping(action, "expert_action").get("primitive_goal")
    return None if goal is None else _mapping(goal, "primitive_goal")


def read_label_evidence(row: Mapping, repo_root: Path) -> tuple[LabelEvidence, ...]:
    """Exact provenance joins (one-based JSONL lines); no heuristic nearest stamp.

    Call lazily, only if canonical scope is incomplete. Missing source files
    fail here; fully canonical labels need neither this function nor RAW files.
    Hashing/indexing visits file bytes once, parsing only referenced records.
    """
    provenance = _mapping(row.get("provenance") or {}, "provenance")
    refs = provenance.get("source_records", ())
    evidence = []
    seen = set()
    for ref in refs:
        ref = _mapping(ref, "source reference")
        source = ref.get("_loader_source_path")
        if not isinstance(source, str) or not source:
            raise ValueError("Source reference is missing its exact path")
        path = Path(source)
        if not path.is_absolute():
            path = repo_root / path
        path = path.resolve()
        record = ref.get("_loader_source_row")
        goal = _source_goal(path, record)
        if (path, record) in seen:
            continue
        seen.add((path, record))
        if goal is None:
            continue
        params = _mapping(goal.get("parameters") or {}, "goal parameters")
        evidence.append(LabelEvidence(str(path), record, _source_index(path).sha256,
                                      goal.get("goal_id"), params.get("execution_phase")))
    return tuple(evidence)


class SMORESAdapter:
    def __init__(self, spec: RobotSpec):
        validate_robot_spec(spec)
        # Canonical SMORES nodes have no type field. One backend type removes
        # guesswork; heterogeneous backend support belongs to another adapter.
        if len(spec.module_types) != 1:
            raise ValueError("SMORES Canonical adapter requires one module type")
        self.spec = spec
        self.module_type = spec.module_types[0]
        self.actions = {a.action_id: a for a in spec.actions}
        self.ports = {p.connector_id: p for p in self.module_type.connectors}
        self.modes = {}
        for resource in self.module_type.resources:
            for mode in resource.modes:
                if mode.mode_id in self.modes:
                    raise ValueError(f"Ambiguous configured mode {mode.mode_id}")
                self.modes[mode.mode_id] = (resource, mode)

    def encode_state(self, graph: Mapping[str, object]) -> StateGraph:
        graph = _mapping(graph, "graph")
        modules, connectors, resources, connections = [], [], [], []
        refs = set()
        for node in graph.get("nodes", ()):
            node = _mapping(node, "node")
            ref = ModuleRef(_id(node.get("module_id")))
            if ref in refs:
                raise ValueError("Duplicate module reference")
            refs.add(ref)
            position_present, orientation_present = node.get("position") is not None, node.get("orientation") is not None
            position = _vector(node["position"], 3) if position_present else (0., 0., 0.)
            orientation = _quaternion(node["orientation"]) if orientation_present else (0., 0., 0., 1.)
            valid = position_present and orientation_present
            modules.append(ModuleState(ref, self.module_type.type_id, position, orientation,
                _vector(node["linear_velocity"], 3) if node.get("linear_velocity") is not None else None,
                _vector(node["angular_velocity"], 3) if node.get("angular_velocity") is not None else None,
                pose_valid=valid))
            actuators = _mapping(node.get("actuators") or {}, "actuators")
            for resource in self.module_type.resources:
                coordinates = []
                for coordinate in resource.coordinates:
                    observed = _mapping(actuators.get(coordinate.coordinate_id) or {}, "actuator")
                    coordinates.append(CoordinateState(coordinate.coordinate_id,
                        _number(observed["position_rad"]) if observed.get("position_rad") is not None else None,
                        _number(observed["velocity_rad_s"]) if observed.get("velocity_rad_s") is not None else None))
                resources.append(ResourceState(ResourceRef(ref, resource.resource_id), tuple(coordinates)))
            faces = _mapping(node.get("faces") or {}, "faces")
            if set(faces) - self.ports.keys():
                raise ValueError("Unknown current connector")
            inverse = (-orientation[0], -orientation[1], -orientation[2], orientation[3])
            for port in self.module_type.connectors:
                face = _mapping(faces.get(port.connector_id) or {}, "face")
                nominal = {"position": port.local_position,
                           "normal": _rotate(port.local_orientation_xyzw, (1., 0., 0.)),
                           "tangent": _rotate(port.local_orientation_xyzw, (0., 1., 0.))}
                features = []
                for field, name in (("position", "position_body_m"), ("normal", "normal_body"), ("tangent", "tangent_body")):
                    if face.get(field) is not None and (valid if field == "position" else orientation_present):
                        value = _vector(face[field], 3)
                        if field == "position":
                            value = tuple(a-b for a, b in zip(value, position))
                        features.append(NumericFeature(name, _rotate(inverse, value)))
                    else:
                        features.append(NumericFeature(name, nominal[field], valid="body_fixed" in port.capabilities))
                status = {"free": "available", "available": "available", "connected": "connected"}.get(face.get("state"), "unknown")
                connectors.append(ConnectorState(ConnectorRef(ref, port.connector_id), port.kind,
                                                 status=status, features=tuple(features)))
        by_ref = {port.ref: i for i, port in enumerate(connectors)}
        for edge in graph.get("edges", ()):
            edge = _mapping(edge, "edge")
            a = ConnectorRef(ModuleRef(_id(edge.get("module_a_id"))), _id(edge.get("face_a")))
            b = ConnectorRef(ModuleRef(_id(edge.get("module_b_id"))), _id(edge.get("face_b")))
            if a not in by_ref or b not in by_ref or a.module == b.module:
                raise ValueError("Physical edge references invalid endpoints")
            relation = edge.get("relation")
            if relation not in {"current_connection", "contact"}:
                raise ValueError("Unknown physical edge relation")
            features = []
            pose = _mapping(edge.get("relative_pose") or {}, "relative_pose")
            if pose.get("position") is not None:
                features.append(NumericFeature("relative_position_m", _vector(pose["position"], 3)))
            if pose.get("orientation") is not None:
                features.append(NumericFeature("relative_orientation_xyzw", _quaternion(pose["orientation"])))
            if edge.get("clocking_quarter_turns") is not None:
                features.append(NumericFeature("clocking_quarter_turns", (_number(edge["clocking_quarter_turns"]),)))
            connections.append(ConnectionState(a, b, relation, features=tuple(features)))
            if relation == "current_connection":
                for endpoint, peer in ((a, b), (b, a)):
                    index = by_ref[endpoint]
                    old = connectors[index]
                    if old.connected_to is not None and old.connected_to != peer:
                        raise ValueError("Connector has contradictory physical connections")
                    connectors[index] = ConnectorState(old.ref, old.kind, peer, "connected", old.features)
        return StateGraph(tuple(modules), tuple(connectors), tuple(resources), tuple(connections))

    def _action(self, name):
        if name not in self.actions:
            raise ValueError(f"Unknown configured action {name!r}")
        return self.actions[name]

    def _resource_target(self, initiator, mode_id, payload, unit=None):
        if mode_id not in self.modes:
            raise ValueError(f"Unknown configured resource mode {mode_id!r}")
        resource, mode = self.modes[mode_id]
        payload = _vector(payload, mode.payload_size)
        if unit is not None and mode.units != (unit,) * mode.payload_size:
            raise ValueError("Resource label units disagree with configured mode")
        return ResourceTarget(ResourceRef(ModuleRef(initiator), resource.resource_id), mode_id, payload)

    def _parameters(self, primitive, params):
        catalog = {p.name: p for p in self._action(primitive).parameters}
        targets = []
        for name, value in params.items():
            if name in _SCOPE_KEYS | _LIFECYCLE_KEYS | {"face_a", "face_b"}:
                continue
            if name not in catalog:
                raise ValueError(f"Unknown configured parameter {primitive}.{name}")
            spec = catalog[name]
            if spec.kind == "scalar":
                value = _number(value)
                if spec.limits is not None and not spec.limits[0] <= value <= spec.limits[1]:
                    raise ValueError(f"Parameter {name} out of limits")
            elif spec.kind == "boolean":
                if type(value) is not bool:
                    raise ValueError(f"Parameter {name} must be boolean")
            elif spec.kind == "category":
                value = str(value)
                if value not in spec.categories:
                    raise ValueError(f"Parameter {name} has invalid category")
            elif spec.kind == "module_refs":
                value = tuple(ModuleRef(mid) for mid in _module_ids(value))
            elif spec.kind == "module_ref":
                value = ModuleRef(_id(value))
            else:
                raise ValueError(f"Unsupported SMORES parameter codec {spec.kind}")
            targets.append(ParameterValue(name, value))
        return tuple(sorted(targets, key=lambda p: p.name))

    def _canonical_scope(self, initiator, primitive, structural, modules, params):
        candidates = []
        for container in (structural, params):
            for key in ("module_ids", "participants"):
                if container.get(key) is not None:
                    scope = _module_ids(container[key])
                    if scope[0] != initiator:
                        raise ValueError("Contradictory canonical initiator scope")
                    candidates.append(scope)
            if container.get("peer_module_id") is not None:
                candidates.append((initiator, _id(container["peer_module_id"])))
        peers = []
        for mid, action in modules.items():
            for participation in action.get("participates_in", ()):
                participation = _mapping(participation, "participation")
                if participation.get("initiator_module_id") != initiator or participation.get("primitive") != primitive:
                    continue
                if primitive in _PAIR_ACTIONS:
                    for source, key in (("initiator_face", "face_a"), ("own_face", "face_b")):
                        observed = participation.get(source)
                        if observed is None:
                            continue
                        if params.get(key) is not None and observed != params[key]:
                            raise ValueError("Contradictory canonical connector parameters")
                        if params.get(key) is None:
                            params[key] = observed
                    peers.append(mid)
        if len(set(peers)) > 1:
            raise ValueError("Ambiguous canonical peer evidence")
        if peers:
            candidates.append((initiator, peers[0]))
        if len(set(candidates)) > 1:
            raise ValueError("Contradictory canonical module scope")
        return candidates[0] if candidates else None

    def _evidence_scope(self, initiator, primitive, params, evidence):
        if not evidence:
            raise LabelEvidenceRequired("Missing structural peer/scope evidence")
        candidates = {}
        for ref in evidence:
            try:
                goal = _source_goal(Path(ref.source_path), ref.source_record, ref.source_sha256)
            except OSError as error:
                raise ValueError(f"Missing source evidence: {ref.source_path}") from error
            if goal is None or goal.get("primitive") != primitive:
                continue
            scope = _module_ids(goal.get("module_ids"))
            if scope[0] != initiator:
                continue
            goal_params = _mapping(goal.get("parameters") or {}, "evidence parameters")
            comparable = {k: v for k, v in params.items()
                          if k not in _SCOPE_KEYS | _LIFECYCLE_KEYS
                          and not (k in {"face_a", "face_b"} and v is None)}
            if any(goal_params.get(k) != v for k, v in comparable.items()):
                raise ValueError("Contradictory source evidence parameters")
            relevant = {k: v for k, v in goal_params.items() if k not in _LIFECYCLE_KEYS | _SCOPE_KEYS}
            signature = json.dumps([scope, relevant], sort_keys=True, allow_nan=False)
            candidates[signature] = (scope, goal_params)
        if len(candidates) != 1:
            raise ValueError("Missing or ambiguous structural peer/scope evidence")
        return next(iter(candidates.values()))

    def _structural_target(self, initiator, structural, modules, evidence):
        primitive = structural.get("primitive")
        params = dict(_mapping(structural.get("parameters") or {}, "structural parameters"))
        if primitive in _PAIR_ACTIONS:
            scope = self._canonical_scope(initiator, primitive, structural, modules, params)
            if scope is None or params.get("face_a") is None or params.get("face_b") is None:
                recovered, goal_params = self._evidence_scope(initiator, primitive, params, evidence)
                if scope is not None and scope != recovered:
                    raise ValueError("Contradictory canonical and evidence scope")
                scope = recovered
                # Only missing face arguments are recovered; canonical params
                # remain authoritative. Evidence never adds executor settings.
                for key in ("face_a", "face_b"):
                    if params.get(key) is None:
                        params[key] = goal_params.get(key)
            if len(scope) != 2:
                raise ValueError("Structural connector pair requires two modules")
            connectors = tuple(ConnectorRef(ModuleRef(mid), _id(params.get(face)))
                               for mid, face in zip(scope, ("face_a", "face_b")))
            if any(port.connector_id not in self.ports for port in connectors):
                raise ValueError("Unknown structural connector")
        elif primitive == "gravity_settle":
            passive_peers = [mid for mid, command in modules.items()
                if any(p.get("primitive") == primitive and p.get("initiator_module_id") == initiator
                       and p.get("role") == "passive_module" for p in command.get("participates_in", ()))]
            if params.get("passive_module_ids") is None:
                if passive_peers:
                    params["passive_module_ids"] = [initiator] + [mid for mid in passive_peers if mid != initiator]
                else:
                    _, recovered = self._evidence_scope(initiator, primitive, params, evidence)
                    params["passive_module_ids"] = recovered.get("passive_module_ids")
                    params.setdefault("duration_s", recovered.get("duration_s"))
            passive = _module_ids(params["passive_module_ids"])
            if not set(passive_peers).issubset(passive):
                raise ValueError("Contradictory canonical gravity_settle passive scope")
            if initiator not in passive or _number(params.get("duration_s")) <= 0:
                raise ValueError("Invalid gravity_settle passive scope/duration")
            scope = (initiator,) + tuple(mid for mid in passive if mid != initiator)
            params["passive_module_ids"] = list(scope)
            connectors = ()
        else:
            raise ValueError(f"Unsupported structural primitive {primitive!r}")
        if any(mid not in modules for mid in scope):
            raise ValueError("Structural scope references an absent module")
        parameters = self._parameters(primitive, params)
        if any(isinstance(p.value, ModuleRef) and p.value.module_id not in modules for p in parameters):
            raise ValueError("Structural parameter references an absent module")
        return StructuralTarget(ModuleRef(initiator), primitive, tuple(ModuleRef(mid) for mid in scope),
                                connectors, parameters)

    def encode_labels(self, action: Mapping[str, object], supervision: Mapping[str, object],
                      evidence: Sequence[LabelEvidence]) -> SupervisionBundle:
        modules = _mapping(action.get("modules"), "action modules")
        masks = _mapping(supervision.get("module_resource_masks") or {}, "loss masks")
        resources, structural_targets, exclusions = [], [], []
        for initiator, commands in modules.items():
            _id(initiator)
            commands = _mapping(commands, "module commands")
            mask = _mapping(masks.get(initiator) or {}, "module loss masks")
            for resource in ("wheels", "internal", "structural"):
                if type(mask.get(resource, False)) is not bool:
                    raise ValueError("Loss mask must be boolean")
                if not mask.get(resource, False):
                    continue
                command = _mapping(commands.get(resource) or {}, "command")
                if command.get("commanded") is not True:
                    raise ValueError("Supervised resource has no commanded label")
                name = ("wheel_velocity" if resource == "wheels" else
                        command.get("mode") if resource == "internal" else command.get("primitive"))
                if not self._action(name).enabled:
                    exclusions.append(f"disabled_action:{name}")
                    continue
                if resource == "wheels":
                    resources.append(self._resource_target(initiator, name,
                        (command.get("left_velocity_rad_s"), command.get("right_velocity_rad_s")), "rad/s"))
                elif resource == "internal":
                    if command.get("unit") is None:
                        raise ValueError("Internal label is missing its units")
                    resources.append(self._resource_target(initiator, name, (command.get("value"),), command.get("unit")))
                else:
                    structural_targets.append(self._structural_target(initiator, command, modules, evidence))
        return SupervisionBundle(tuple(resources), tuple(structural_targets), tuple(exclusions), tuple(evidence))

    def decode_action(self, choice: ActionChoice) -> Mapping[str, object]:
        """SMORES command data only; no runtime or lifecycle side effects."""
        if not self._action(choice.action_id).enabled:
            raise ValueError("Action explicitly disabled by RobotSpec")
        initiator = _id(choice.initiator.module_id)
        if choice.action_id in self.modes:
            target = self._resource_target(initiator, choice.action_id, choice.payload)
            if choice.resource != target.resource:
                raise ValueError("Action resource does not match configured mode/initiator")
            if choice.action_id == "wheel_velocity":
                command, parameters = "wheel_velocity", dict(zip(("left_rad_s", "right_rad_s"), target.payload))
            else:
                command, key = _INTERNAL_COMMANDS[choice.action_id]
                parameters = {key: target.payload[0]}
            return {"command": command, "module_ids": [initiator], "parameters": parameters}
        raw_params = {}
        for p in choice.parameters:
            if p.name in raw_params:
                raise ValueError("Duplicate action parameter")
            if isinstance(p.value, ModuleRef):
                raw_params[p.name] = p.value.module_id
            elif isinstance(p.value, tuple) and all(isinstance(v, ModuleRef) for v in p.value):
                raw_params[p.name] = [v.module_id for v in p.value]
            else:
                raw_params[p.name] = p.value
        if choice.action_id in _PAIR_ACTIONS:
            if len(choice.connectors) != 2 or choice.connectors[0].module != choice.initiator:
                raise ValueError("Action requires initiator and peer connectors")
            scope = tuple(p.module.module_id for p in choice.connectors)
            if choice.participants and choice.participants != tuple(ModuleRef(mid) for mid in scope):
                raise ValueError("Contradictory action participant scope")
            raw_params.update(face_a=choice.connectors[0].connector_id, face_b=choice.connectors[1].connector_id)
            module_ids = list(scope)
        elif choice.action_id == "gravity_settle":
            if choice.connectors:
                raise ValueError("gravity_settle uses modules, not connectors")
            if "passive_module_ids" not in raw_params and choice.participants:
                raw_params["passive_module_ids"] = [mid.module_id for mid in choice.participants]
            module_ids = [initiator]
        else:
            raise ValueError(f"Unsupported SMORES action {choice.action_id}")
        # Validate decoded scope and arguments through the same structural path.
        references = {raw_params[p.name] for p in self._action(choice.action_id).parameters
                      if p.kind == "module_ref" and p.name in raw_params}
        mids = set(module_ids) | set(raw_params.get("passive_module_ids", [])) | references
        dummy_modules = {mid: {} for mid in mids}
        structural = {"primitive": choice.action_id, "module_ids": module_ids, "parameters": raw_params}
        target = self._structural_target(initiator, structural, dummy_modules, ())
        if choice.action_id == "gravity_settle" and choice.participants:
            if set(choice.participants) != set(target.participants):
                raise ValueError("Contradictory gravity_settle action scope")
        parameters = {}
        for p in target.parameters:
            if isinstance(p.value, ModuleRef):
                parameters[p.name] = p.value.module_id
            elif isinstance(p.value, tuple):
                parameters[p.name] = [v.module_id for v in p.value]
            elif p.name == "clocking_quarter_turns":
                parameters[p.name] = int(p.value)
            else:
                parameters[p.name] = p.value
        if target.connectors:
            parameters.update(face_a=target.connectors[0].connector_id, face_b=target.connectors[1].connector_id)
        return {"command": choice.action_id, "module_ids": module_ids, "parameters": parameters}


def _source_snapshot(root):
    """Read-only identity: exclude atime (which reads legitimately update)."""
    files = {}
    for path in sorted(root.rglob("*.jsonl")):
        signature = _file_signature(path)
        files[path.relative_to(root).as_posix()] = {
            "inode": signature[0], "bytes": signature[1],
            "mtime_ns": signature[2], "ctime_ns": signature[3],
        }
    manifest = root / "manifest.json"
    signature = _file_signature(manifest)
    return {"manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "manifest_stat": {"inode": signature[0], "bytes": signature[1],
                              "mtime_ns": signature[2], "ctime_ns": signature[3]},
            "files": files}


def _manifest_entries(root, manifest, snapshot):
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Manifest requires a nonempty episode list")
    ids, files = set(), set()
    for entry in episodes:
        entry = _mapping(entry, "episode entry")
        eid = _id(entry.get("episode_id"))
        filename = entry.get("file")
        if not isinstance(filename, str) or Path(filename).is_absolute():
            raise ValueError("Manifest episode file must be a relative path")
        if (root / filename).resolve().is_relative_to(root) is False:
            raise ValueError("Episode path escapes source root")
        if filename in files or eid in ids:
            raise ValueError("Duplicate episode/file in source inventory")
        files.add(filename)
        ids.add(eid)
        if filename not in snapshot["files"]:
            raise ValueError(f"Episode file missing from source inventory: {filename}")
        if type(entry.get("bytes")) is not int or entry["bytes"] != snapshot["files"][filename]["bytes"]:
            raise ValueError(f"Episode payload size differs from manifest: {filename}")
        digest = entry.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(v not in "0123456789abcdef" for v in digest):
            raise ValueError("Invalid manifest-declared payload SHA256")
        _id(entry.get("kind"))
        _id(entry.get("task"))
    if files != set(snapshot["files"]):
        raise ValueError("Source episode inventory differs from manifest")
    return episodes


def _count_canonical_labels(action, supervision):
    """Count command visibility separately from the exact loss-eligible keys."""
    modules = _mapping(action.get("modules"), "joint action modules")
    masks = _mapping(supervision.get("module_resource_masks"), "module loss masks")
    if set(masks) - set(modules):
        raise ValueError("Loss mask references absent action module")
    valid = supervision.get("valid_for_behavior_cloning")
    if type(valid) is not bool:
        raise ValueError("BC-valid flag must be boolean")
    commanded, expected_resources, expected_structural = Counter(), {}, set()
    for mid, commands in modules.items():
        _id(mid)
        commands = _mapping(commands, "module commands")
        mask = _mapping(masks.get(mid) or {}, "module mask")
        if set(mask) - {"wheels", "internal", "structural"}:
            raise ValueError("Unknown Canonical resource loss mask")
        for resource in ("wheels", "internal", "structural"):
            command = _mapping(commands.get(resource) or {}, "command")
            supervised = mask.get(resource, False)
            if type(supervised) is not bool:
                raise ValueError("Loss mask must be boolean")
            name = ("wheel_velocity" if resource == "wheels" else command.get("mode")
                    if resource == "internal" else command.get("primitive"))
            if command.get("commanded") is True:
                commanded[_id(name)] += 1
            if not supervised:
                continue
            if not valid or command.get("commanded") is not True:
                raise ValueError("Supervised target contradicts BC-valid/commanded flags")
            name = _id(name)
            if resource == "structural":
                expected_structural.add((mid, name))
            else:
                payload = ((command.get("left_velocity_rad_s"), command.get("right_velocity_rad_s"))
                           if resource == "wheels" else (command.get("value"),))
                expected_resources[mid, name] = tuple(_number(v) for v in payload)
    if valid != bool(expected_resources or expected_structural):
        raise ValueError("BC-valid flag does not match supervised target presence")
    return commanded, expected_resources, expected_structural


def _check_generic_targets(bundle, expected_resources, expected_structural, graph, adapter):
    """Certify labels, not features: only current module/connector references.

    B2 does not run feature extraction on 197k graphs. B1 tested state encoding;
    here graph nodes are used solely to validate label routing references.
    """
    if not isinstance(bundle, SupervisionBundle) or bundle.exclusions:
        raise ValueError("Supervised label excluded/discarded by adapter")
    modules = {_id(n.get("module_id")): n for n in graph.get("nodes", ())}
    actual_resources, actual_structural = {}, set()
    for target in bundle.resource_targets:
        if not isinstance(target, ResourceTarget) or target.supervised is not True:
            raise ValueError("Invalid/unsupervised generic resource target")
        if not isinstance(target.resource, ResourceRef) or target.resource.module.module_id not in modules:
            raise ValueError("Resource target references absent physical module")
        _id(target.resource.resource_id)
        key = target.resource.module.module_id, _id(target.mode_id)
        if key in actual_resources or not target.payload:
            raise ValueError("Duplicate/empty generic resource target")
        actual_resources[key] = tuple(_number(v) for v in target.payload)
    for target in bundle.structural_targets:
        if not isinstance(target, StructuralTarget) or target.supervised is not True:
            raise ValueError("Invalid/unsupervised generic structural target")
        key = target.initiator.module_id, _id(target.operation_id)
        if key in actual_structural:
            raise ValueError("Duplicate generic structural target")
        actual_structural.add(key)
        scope = tuple(mid.module_id for mid in target.participants)
        if not scope or scope[0] != target.initiator.module_id or len(set(scope)) != len(scope) or not set(scope).issubset(modules):
            raise ValueError("Incomplete/invalid structural participant scope")
        for endpoint in target.connectors:
            if not isinstance(endpoint, ConnectorRef) or endpoint.module.module_id not in scope:
                raise ValueError("Structural connector outside participant scope")
            node = modules[endpoint.module.module_id]
            if endpoint.connector_id not in (node.get("faces") or {}):
                raise ValueError("Structural connector absent from current physical module")
        for parameter in target.parameters:
            if isinstance(parameter.value, ModuleRef) and parameter.value.module_id not in modules:
                raise ValueError("Structural parameter references absent physical module")
        if target.operation_id in _PAIR_ACTIONS:
            if len(scope) != 2 or len(target.connectors) != 2 or tuple(p.module.module_id for p in target.connectors) != scope:
                raise ValueError("Incomplete structural connector pair")
        elif target.operation_id == "gravity_settle":
            parameters = {p.name: p.value for p in target.parameters}
            passive = parameters.get("passive_module_ids")
            if target.connectors or not isinstance(passive, tuple) or set(passive) != set(target.participants) or _number(parameters.get("duration_s")) <= 0:
                raise ValueError("Incomplete gravity_settle scope/duration")
        else:
            raise ValueError("Executor phase/unsupported primitive is not a BC structural action")
        decoded = adapter.decode_action(ActionChoice(target.operation_id, target.initiator,
            participants=target.participants, connectors=target.connectors, parameters=target.parameters))
        _mapping(decoded, "decoded action")
        json.dumps(decoded, allow_nan=False)
    if actual_resources != expected_resources or actual_structural != expected_structural:
        raise ValueError("Adapter discarded/invented/changed a supervised target")


def _targets_requiring_evidence(adapter, action, supervision, structural_keys):
    """Identify dependencies only on fallback rows, retaining the joint label."""
    needed = set()
    for initiator, primitive in structural_keys:
        masks = {mid: {"wheels": False, "internal": False, "structural": mid == initiator}
                 for mid in action["modules"]}
        isolated = {**supervision, "module_resource_masks": masks}
        try:
            adapter.encode_labels(action, isolated, ())
        except LabelEvidenceRequired:
            needed.add((initiator, primitive))
    if not needed:
        raise ValueError("Adapter requested evidence without an incomplete structural target")
    return needed


def audit_canonical_source(root: Path, adapter: RobotAdapter) -> Mapping:
    """One streaming READ-ONLY census, with independently checkable loss counts.

    No filters, repeat weighting, tensors or dataset writes. Memory consists of
    one decoded row, counters, per-episode summaries/stat snapshots and at most
    100 diagnostic examples. Payload SHA256s are manifest declarations; this
    audit compares inventory, sizes and stable stat metadata, not all 10 GB hashes.
    Exact source evidence is requested only for LabelEvidenceRequired. Other
    label failures are never retried/hidden using provenance or future states.
    """
    root = Path(root).resolve()
    before = _source_snapshot(root)
    manifest = _mapping(json.loads((root / "manifest.json").read_bytes()), "manifest")
    entries = _manifest_entries(root, manifest, before)
    totals, kinds, target_counts, command_counts, command_rows = (Counter() for _ in range(5))
    schemas, morphology_rows, bc_morphology_rows = (Counter() for _ in range(3))
    resource_counts, structural_counts, per_episode, evidence_files = {}, Counter(), [], {}
    diagnostics = {"resource_conflicts": 0, "conflicts_on_bc_rows": 0, "conflicting_resources_supervised": 0,
                   "unresolved_sequences": 0, "unresolved_sequences_on_bc_rows": 0,
                   "unresolved_sequences_by_episode": {}}
    conflict_examples, fallback_rows, with_evidence, without_evidence, settle_checked = [], 0, 0, 0, 0
    # Canonical provenance paths use the repository layout datasets/<collection>.
    repo_root = root.parent.parent
    for entry in entries:
        eid = entry["episode_id"]
        local, local_targets, local_commands, local_command_rows, local_morph, local_bc_morph = (Counter() for _ in range(6))
        with (root / entry["file"]).open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if not line.strip():
                    raise ValueError(f"Source changed or invalid blank row: {eid} transition {index}")
                try:
                    row = _mapping(json.loads(line), "Canonical row")
                    graph = _mapping(row.get("graph_t"), "current physical graph")
                    action = _mapping(row.get("action_t"), "joint action")
                    supervision = _mapping(row.get("supervision"), "supervision")
                    commands, expected_resources, expected_structural = _count_canonical_labels(action, supervision)
                    local["canonical_rows"] += 1
                    valid = supervision["valid_for_behavior_cloning"]
                    local["bc_valid_rows"] += valid
                    schemas[_id(row.get("schema_version"))] += 1
                    morphology = graph.get("morphology") or "missing"
                    _id(morphology)
                    local_morph[morphology] += 1
                    local_bc_morph[morphology] += valid
                    local_commands.update(commands)
                    local_command_rows.update(commands.keys())
                    for conflict in supervision.get("conflicts", ()):
                        conflict = _mapping(conflict, "conflict")
                        resource_mask = supervision["module_resource_masks"].get(conflict.get("module_id"), {}).get(conflict.get("resource"), False)
                        diagnostics["resource_conflicts"] += 1
                        diagnostics["conflicts_on_bc_rows"] += valid
                        diagnostics["conflicting_resources_supervised"] += resource_mask
                        if len(conflict_examples) < 100:
                            conflict_examples.append({"episode_id": eid, "transition_index": index,
                                "module_id": conflict.get("module_id"), "resource": conflict.get("resource"),
                                "bc_valid": valid, "resource_supervised": resource_mask,
                                "reason": conflict.get("reason")})
                        if resource_mask:
                            raise ValueError("Conflicting resource still has supervised loss mask")
                    unresolved = (row.get("provenance") or {}).get("unresolved_recorded_sequences") or ()
                    if unresolved:
                        diagnostics["unresolved_sequences"] += len(unresolved)
                        diagnostics["unresolved_sequences_on_bc_rows"] += len(unresolved) * valid
                        by_episode = diagnostics["unresolved_sequences_by_episode"]
                        by_episode[eid] = by_episode.get(eid, 0) + len(unresolved)
                    if not valid:
                        continue
                    evidence, evidence_targets = (), set()
                    try:
                        bundle = adapter.encode_labels(action, supervision, ())
                    except LabelEvidenceRequired:
                        evidence_targets = _targets_requiring_evidence(adapter, action, supervision, expected_structural)
                        evidence = read_label_evidence(row, repo_root)
                        bundle = adapter.encode_labels(action, supervision, evidence)
                        fallback_rows += 1
                        for ref in evidence:
                            path = Path(ref.source_path)
                            display = path.relative_to(repo_root).as_posix() if path.is_relative_to(repo_root) else str(path)
                            evidence_files[display] = ref.source_sha256
                    _check_generic_targets(bundle, expected_resources, expected_structural, graph, adapter)
                    for target in bundle.resource_targets:
                        local_targets[target.mode_id] += 1
                        modes = resource_counts.setdefault(target.resource.resource_id, Counter())
                        modes[target.mode_id] += 1
                    for target in bundle.structural_targets:
                        local_targets[target.operation_id] += 1
                        structural_counts[target.operation_id] += 1
                        settle_checked += target.operation_id == "gravity_settle"
                    with_evidence += len(evidence_targets)
                    without_evidence += len(bundle.structural_targets) - len(evidence_targets)
                except (ValueError, TypeError, KeyError, OSError) as error:
                    raise ValueError(f"{eid} transition {index}: {error}") from error
        for field, key in (("canonical_transitions", "canonical_rows"), ("bc_valid_transitions", "bc_valid_rows")):
            if local[key] != entry.get(field):
                raise ValueError(f"Episode row count differs from manifest: {eid} {key}")
        for key, measured in (("command_instances", local_commands), ("command_types", local_command_rows)):
            if dict(measured) != entry.get(key):
                raise ValueError(f"Episode command count differs from manifest: {eid} {key}")
        totals.update(local)
        kinds[entry["kind"]] += 1
        target_counts.update(local_targets)
        command_counts.update(local_commands)
        command_rows.update(local_command_rows)
        morphology_rows.update(local_morph)
        bc_morphology_rows.update(local_bc_morph)
        per_episode.append({"episode_id": eid, "kind": entry["kind"], "task": entry["task"],
            "file": entry["file"], "sha256": entry["sha256"], "bytes": entry["bytes"],
            **local, "supervised_instances": dict(local_targets), "command_instances": dict(local_commands),
            "command_types": dict(local_command_rows), "morphology_rows": dict(local_morph),
            "bc_morphology_rows": dict(local_bc_morph)})
    after = _source_snapshot(root)
    if before != after:
        raise ValueError("Canonical source changed/mutated during audit")
    inventory = [{k: e[k] for k in ("episode_id", "file", "sha256", "bytes")}
                 for e in sorted(entries, key=lambda e: e["episode_id"])]
    inventory_sha = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    result = {"schema_version": "mssr.bc_v1.source_audit.v1",
        "source": {"dataset_name": manifest["dataset_name"], "collection_schema": manifest["schema_version"],
                   "manifest_sha256": before["manifest_sha256"], "episode_inventory_sha256": inventory_sha,
                   "transition_schemas": dict(schemas)},
        "episodes": len(entries), "episodes_by_kind": dict(kinds),
        "bc_useful_episodes": sum(e["bc_valid_rows"] > 0 for e in per_episode),
        **totals, "supervised_instances": dict(target_counts), "command_instances": dict(command_counts),
        "command_types": dict(command_rows), "generic_resource_instances": {r: dict(m) for r, m in resource_counts.items()},
        "structural_counts": dict(structural_counts), "structural_targets_checked": sum(structural_counts.values()),
        "unresolved_supervised_structural_targets": 0, "settle_targets_checked": settle_checked,
        "evidence_fallback_rows": fallback_rows, "structural_resolved_with_evidence": with_evidence,
        "structural_resolved_without_evidence": without_evidence,
        "source_evidence_files": [{"path": p, "sha256": h} for p, h in sorted(evidence_files.items())],
        "morphology_rows": dict(morphology_rows), "bc_morphology_rows": dict(bc_morphology_rows),
        "per_episode": per_episode, "diagnostics": diagnostics, "conflict_examples": conflict_examples,
        "immutability": {"before": before, "after": after},
        "warnings": ["Payload hashes are manifest declarations; payload bytes were not rehashed."]}
    manifest_totals = manifest.get("totals", {})
    for key, measured in (("episodes", result["episodes"]), ("episodes_by_kind", result["episodes_by_kind"]),
                          ("canonical_transitions", result["canonical_rows"]), ("bc_valid_transitions", result["bc_valid_rows"]),
                          ("command_instances", result["command_instances"]), ("command_types", result["command_types"]),
                          ("resource_conflicts", diagnostics["resource_conflicts"]),
                          ("unresolved_recorded_sequences", diagnostics["unresolved_sequences"])):
        if key in manifest_totals and manifest_totals[key] != measured:
            raise ValueError(f"Collection total differs from manifest: {key}")
    return result
