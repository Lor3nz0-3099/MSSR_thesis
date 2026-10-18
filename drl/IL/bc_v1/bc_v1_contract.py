"""Stdlib-only value types at the BC-v1 policy boundary.

IDs resolve references for adapters and batching; they are not learned features.
Inputs contain current physical state and online task context only. Labels and
their reconstruction evidence live in separate containers. Robot catalogs have
explicit stdlib validation; feature selection, tensorization and runtime
feasibility remain later steps.

Collection fields use tuples: callers construct these typed values after parsing
their source data. These dataclasses do not validate untrusted JSON themselves.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Literal, Mapping, Protocol, Sequence, TypeAlias


Vector3: TypeAlias = tuple[float, float, float]
Quaternion: TypeAlias = tuple[float, float, float, float]  # xyzw
ConnectorKind: TypeAlias = Literal["fixed_port", "surface"]
ParameterKind: TypeAlias = Literal[
    "scalar", "vector", "boolean", "category", "module_ref", "connector_ref",
    "module_refs", "connector_refs",
]


@dataclass(frozen=True)
class ModuleRef:
    module_id: str


@dataclass(frozen=True)
class ConnectorRef:
    module: ModuleRef
    connector_id: str


@dataclass(frozen=True)
class ResourceRef:
    module: ModuleRef
    resource_id: str


@dataclass(frozen=True)
class NumericFeature:
    """Named scalar/vector with explicit validity; feature whitelist is external."""

    name: str
    values: tuple[float, ...]
    valid: bool = True


@dataclass(frozen=True)
class CoordinateSpec:
    coordinate_id: str
    units: str
    limits: tuple[float, float] | None = None
    periodic: bool = False
    axis: Vector3 | None = None


@dataclass(frozen=True)
class ResourceModeSpec:
    mode_id: str
    payload_size: int
    units: tuple[str, ...]
    kind: Literal["continuous", "discrete"] = "continuous"
    codec: str = "vector"
    categories: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResourceSpec:
    resource_id: str
    coordinates: tuple[CoordinateSpec, ...] = ()
    modes: tuple[ResourceModeSpec, ...] = ()
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConnectorSpec:
    connector_id: str
    kind: ConnectorKind
    local_position: Vector3 = (0.0, 0.0, 0.0)
    local_orientation_xyzw: Quaternion = (0.0, 0.0, 0.0, 1.0)
    geometry: tuple[NumericFeature, ...] = ()
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModuleTypeSpec:
    type_id: str
    connectors: tuple[ConnectorSpec, ...] = ()
    resources: tuple[ResourceSpec, ...] = ()
    geometry: tuple[NumericFeature, ...] = ()
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    kind: ParameterKind
    units: str | None = None
    size: int | None = None
    categories: tuple[str, ...] = ()
    limits: tuple[float, float] | None = None
    learnable: bool = True
    # Catalog constraints name module TYPES, never runtime module instances.
    # None codec means the codec implied by kind; explicit values must agree.
    codec: str | None = None
    module_type_id: str | None = None
    resource_id: str | None = None
    connector_id: str | None = None


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    kind: Literal[
        "continuous", "discrete", "prepare_connection", "connect", "disconnect", "skill",
    ]
    scope: Literal["resource", "module", "modules", "connector_pair"]
    parameters: tuple[ParameterSpec, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    enabled: bool = True


@dataclass(frozen=True)
class RobotSpec:
    schema_version: str
    robot_id: str
    module_types: tuple[ModuleTypeSpec, ...] = ()
    actions: tuple[ActionSpec, ...] = ()


@dataclass(frozen=True)
class ModuleState:
    ref: ModuleRef
    module_type_id: str
    position: Vector3
    orientation_xyzw: Quaternion
    linear_velocity: Vector3 | None = None
    angular_velocity: Vector3 | None = None
    pose_valid: bool = False
    features: tuple[NumericFeature, ...] = ()


@dataclass(frozen=True)
class ConnectorState:
    ref: ConnectorRef
    kind: ConnectorKind
    connected_to: ConnectorRef | None = None
    status: Literal["unknown", "available", "connected"] = "unknown"
    features: tuple[NumericFeature, ...] = ()


@dataclass(frozen=True)
class CoordinateState:
    coordinate_id: str
    position: float | None = None
    velocity: float | None = None


@dataclass(frozen=True)
class ResourceState:
    ref: ResourceRef
    coordinates: tuple[CoordinateState, ...] = ()
    features: tuple[NumericFeature, ...] = ()


@dataclass(frozen=True)
class ConnectionState:
    endpoint_a: ConnectorRef
    endpoint_b: ConnectorRef
    relation_id: str
    allowed_relative_dofs: tuple[str, ...] = ()
    features: tuple[NumericFeature, ...] = ()


@dataclass(frozen=True)
class StateGraph:
    modules: tuple[ModuleState, ...] = ()
    connectors: tuple[ConnectorState, ...] = ()
    resources: tuple[ResourceState, ...] = ()
    connections: tuple[ConnectionState, ...] = ()


@dataclass(frozen=True)
class ObservationBundle:
    features: tuple[NumericFeature, ...] = ()


@dataclass(frozen=True)
class TaskContext:
    """Externally assigned current problem and its static physical geometry.

    Task ontology is independent of the robot; None means unobserved.
    """

    task_id: str | None = None
    static_geometry: tuple[NumericFeature, ...] = ()


@dataclass(frozen=True)
class PolicyInputs:
    graph: StateGraph
    observation: ObservationBundle
    context: TaskContext

    def __post_init__(self) -> None:
        # Lazy import keeps the value-type module independent at import time.
        # Direct construction must enforce the same boundary as D2's selector.
        from .policy_inputs import _validate_selected_inputs

        _validate_selected_inputs(self)


ParameterPayload: TypeAlias = (
    float | bool | str | tuple[float, ...] | ModuleRef | ConnectorRef
    | tuple[ModuleRef, ...] | tuple[ConnectorRef, ...]
)


@dataclass(frozen=True)
class ParameterValue:
    name: str
    value: ParameterPayload


@dataclass(frozen=True)
class ResourceTarget:
    resource: ResourceRef
    mode_id: str
    payload: tuple[float, ...]
    supervised: bool = True  # Loss eligibility, never online feasibility.


@dataclass(frozen=True)
class StructuralTarget:
    """One initiator decision, not a mutually exclusive alternative to its peers."""

    initiator: ModuleRef
    operation_id: str
    participants: tuple[ModuleRef, ...] = ()
    connectors: tuple[ConnectorRef, ...] = ()
    parameters: tuple[ParameterValue, ...] = ()
    supervised: bool = True


@dataclass(frozen=True)
class LabelEvidence:
    """Source provenance for reconstructing labels; never a policy input."""

    source_path: str
    source_record: int
    source_sha256: str
    goal_id: str | None = None
    phase: str | None = None


@dataclass(frozen=True)
class SupervisionBundle:
    resource_targets: tuple[ResourceTarget, ...] = ()
    structural_targets: tuple[StructuralTarget, ...] = ()
    exclusions: tuple[str, ...] = ()
    label_evidence: tuple[LabelEvidence, ...] = ()


@dataclass(frozen=True)
class SampleAudit:
    episode_id: str
    transition_index: int
    physical_stamp: float | None = None
    provenance: str | None = None
    task_stratum: str | None = None
    morphology_stratum: str | None = None
    source_hashes: tuple[tuple[str, str], ...] = ()
    joint_decision_id: str | None = None
    event_order: int | None = None
    recovery_confidence: str | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PolicySample:
    inputs: PolicyInputs
    supervision: SupervisionBundle
    audit: SampleAudit


@dataclass(frozen=True)
class ActionChoice:
    """Configured action and typed arguments for an adapter to decode."""

    action_id: str
    initiator: ModuleRef
    resource: ResourceRef | None = None
    participants: tuple[ModuleRef, ...] = ()
    connectors: tuple[ConnectorRef, ...] = ()
    parameters: tuple[ParameterValue, ...] = ()
    payload: tuple[float, ...] = ()


class RobotAdapter(Protocol):
    def encode_state(self, graph: Mapping[str, object]) -> StateGraph:
        ...

    def encode_labels(
        self,
        action: Mapping[str, object],
        supervision: Mapping[str, object],
        evidence: Sequence[LabelEvidence],
    ) -> SupervisionBundle:
        ...

    def decode_action(self, choice: ActionChoice) -> Mapping[str, object]:
        ...


# A2 catalog boundary. Runtime state, label support and feasibility are absent.
_ROBOT_SPEC_SCHEMA = "mssr.bc_v1.robot_spec.v1"
_ARGUMENT_CODECS = {
    "scalar": "continuous", "vector": "continuous", "boolean": "categorical",
    "category": "categorical", "module_ref": "module_ref",
    "connector_ref": "connector_ref", "module_refs": "reference_set",
    "connector_refs": "reference_set",
}
# Catalog order has no meaning. Category and numeric component order DOES.
_CATALOG_KEYS = {
    "module_types": "type_id", "connectors": "connector_id",
    "resources": "resource_id", "coordinates": "coordinate_id",
    "modes": "mode_id", "actions": "action_id", "parameters": "name",
    "geometry": "name", "capabilities": None, "required_capabilities": None,
}
# Closed deserialization tree for the A1 catalog dataclasses.
_SPEC_CHILDREN = {
    RobotSpec: {"module_types": ModuleTypeSpec, "actions": ActionSpec},
    ModuleTypeSpec: {"connectors": ConnectorSpec, "resources": ResourceSpec,
                     "geometry": NumericFeature},
    ConnectorSpec: {"geometry": NumericFeature},
    ResourceSpec: {"coordinates": CoordinateSpec, "modes": ResourceModeSpec},
    ActionSpec: {"parameters": ParameterSpec},
}
_TUPLE_FIELDS = {
    "capabilities", "required_capabilities", "units", "categories", "limits",
    "axis", "local_position", "local_orientation_xyzw", "values",
}


def _spec_error(path: str, message: str) -> None:
    raise ValueError(f"{path}: {message}")


def _identifier(value: object, path: str) -> None:
    if not isinstance(value, str) or not value or any(ch.isspace() for ch in value):
        _spec_error(path, "expected a nonempty string without whitespace")


def _boolean(value: object, path: str) -> None:
    if type(value) is not bool:
        _spec_error(path, "expected a boolean")


def _number(value: object, path: str) -> None:
    if type(value) not in (int, float):
        _spec_error(path, "expected a finite number, not a boolean")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        _spec_error(path, "expected a finite number")


def _numeric_tuple(value: object, path: str, size: int | None = None) -> None:
    if not isinstance(value, tuple) or not value or (size is not None and len(value) != size):
        _spec_error(path, f"expected a numeric tuple of length {size or '>=1'}")
    for i, item in enumerate(value):
        _number(item, f"{path}[{i}]")


def _limits(value: object, path: str) -> None:
    if value is None:
        return
    _numeric_tuple(value, path, 2)
    if value[0] > value[1]:
        _spec_error(path, "lower bound exceeds upper bound")


def _enum(value: object, allowed: object, path: str) -> None:
    if not isinstance(value, str) or value not in allowed:
        _spec_error(path, f"unsupported value {value!r}")


def _catalog(value: object, cls: type, key: str, path: str) -> dict:
    if not isinstance(value, tuple):
        _spec_error(path, "expected an immutable tuple")
    result = {}
    for i, item in enumerate(value):
        if not isinstance(item, cls):
            _spec_error(f"{path}[{i}]", f"expected {cls.__name__}")
        identity = getattr(item, key)
        _identifier(identity, f"{path}[{i}].{key}")
        if identity in result:
            _spec_error(path, f"duplicate {key} {identity!r}")
        result[identity] = item
    return result


def _names(value: object, path: str) -> None:
    if not isinstance(value, tuple):
        _spec_error(path, "expected an immutable tuple")
    seen = set()
    for item in value:
        _identifier(item, path)
        if item in seen:
            _spec_error(path, f"duplicate identifier {item!r}")
        seen.add(item)


def _geometry(value: object, path: str) -> None:
    for name, feature in _catalog(value, NumericFeature, "name", path).items():
        _numeric_tuple(feature.values, f"{path}[{name}].values")
        _boolean(feature.valid, f"{path}[{name}].valid")


def _validate_parameter(p: ParameterSpec, path: str, modules: dict) -> None:
    _enum(p.kind, _ARGUMENT_CODECS, f"{path}.kind")
    expected = _ARGUMENT_CODECS[p.kind]
    if p.codec is not None and p.codec != expected:
        _spec_error(f"{path}.codec", f"expected {expected!r} for kind {p.kind!r}")
    _boolean(p.learnable, f"{path}.learnable")
    _names(p.categories, f"{path}.categories")
    if p.size is not None and (type(p.size) is not int or p.size < 1):
        _spec_error(f"{path}.size", "expected a positive integer or None")
    if p.kind == "vector" and p.size is None:
        _spec_error(f"{path}.size", "vector requires a positive payload dimension")
    if p.kind not in {"vector", "module_refs", "connector_refs"} and p.size not in (None, 1):
        _spec_error(f"{path}.size", "scalar/category/single reference has dimension one")
    if p.units is not None:
        _identifier(p.units, f"{path}.units")
    _limits(p.limits, f"{path}.limits")
    if p.kind in {"scalar", "vector"}:
        if p.categories:
            _spec_error(f"{path}.categories", "numeric arguments cannot declare categories")
    else:
        if p.units is not None or p.limits is not None:
            _spec_error(path, "nonnumeric arguments must omit units and limits")
        if (p.kind == "category") != bool(p.categories):
            _spec_error(f"{path}.categories", "only category arguments require nonempty categories")

    for field in ("module_type_id", "resource_id", "connector_id"):
        value = getattr(p, field)
        if value is not None:
            _identifier(value, f"{path}.{field}")
    if p.module_type_id is None:
        if p.resource_id is not None or p.connector_id is not None:
            _spec_error(f"{path}.module_type_id", "an owner type is required for resource/connector references")
        return
    owner = modules.get(p.module_type_id)
    if owner is None:
        _spec_error(f"{path}.module_type_id", f"dangling reference {p.module_type_id!r}")
    if p.resource_id is not None:
        if p.kind not in {"scalar", "vector", "category"} or p.connector_id is not None:
            _spec_error(f"{path}.resource_id", "resource binding requires a numeric/category argument without connector binding")
        resources = {r.resource_id: r for r in owner.resources}
        if p.resource_id not in resources:
            _spec_error(f"{path}.resource_id", f"dangling reference {p.resource_id!r}")
        if p.kind in {"scalar", "vector"}:
            resource = resources[p.resource_id]
            units = {x.units for x in resource.coordinates}
            units.update(u for mode in resource.modes for u in mode.units)
            if (p.units or "1") not in units:
                _spec_error(f"{path}.units", "unit is not declared by the bound resource")
    if p.connector_id is not None:
        if p.kind not in {"connector_ref", "connector_refs"}:
            _spec_error(f"{path}.connector_id", "connector binding requires connector_ref(s)")
        if p.connector_id not in {x.connector_id for x in owner.connectors}:
            _spec_error(f"{path}.connector_id", f"dangling reference {p.connector_id!r}")


def validate_robot_spec(spec: RobotSpec) -> None:
    """Validate a type/capability catalog using stdlib only.

    Coordinates are observables; modes have independent ordered payload units.
    A surface declares nonempty named geometry/domain features, whose physical
    interpretation belongs to its adapter. Runtime instance references and
    feasibility cannot be checked here. All failures are path-qualified ValueError.
    """
    if not isinstance(spec, RobotSpec):
        _spec_error("RobotSpec", "expected a RobotSpec")
    if spec.schema_version != _ROBOT_SPEC_SCHEMA:
        _spec_error("schema_version", f"expected {_ROBOT_SPEC_SCHEMA!r}")
    _identifier(spec.robot_id, "robot_id")
    modules = _catalog(spec.module_types, ModuleTypeSpec, "type_id", "module_types")
    if not modules:
        _spec_error("module_types", "at least one module type is required")
    capabilities = set()
    for mid, module in modules.items():
        path = f"module_types[{mid}]"
        _names(module.capabilities, f"{path}.capabilities")
        capabilities.update(module.capabilities)
        _geometry(module.geometry, f"{path}.geometry")
        connectors = _catalog(module.connectors, ConnectorSpec, "connector_id", f"{path}.connectors")
        for cid, connector in connectors.items():
            cp = f"{path}.connectors[{cid}]"
            _enum(connector.kind, {"fixed_port", "surface"}, f"{cp}.kind")
            _numeric_tuple(connector.local_position, f"{cp}.local_position", 3)
            _numeric_tuple(connector.local_orientation_xyzw, f"{cp}.local_orientation_xyzw", 4)
            if not math.isclose(math.hypot(*connector.local_orientation_xyzw), 1.0, rel_tol=1e-6, abs_tol=1e-6):
                _spec_error(f"{cp}.local_orientation_xyzw", "expected a unit quaternion")
            _geometry(connector.geometry, f"{cp}.geometry")
            if connector.kind == "surface" and not connector.geometry:
                _spec_error(f"{cp}.geometry", "surface requires geometry/domain features")
            _names(connector.capabilities, f"{cp}.capabilities")
            capabilities.update(connector.capabilities)
        resources = _catalog(module.resources, ResourceSpec, "resource_id", f"{path}.resources")
        for rid, resource in resources.items():
            rp = f"{path}.resources[{rid}]"
            _names(resource.capabilities, f"{rp}.capabilities")
            capabilities.update(resource.capabilities)
            coordinates = _catalog(resource.coordinates, CoordinateSpec, "coordinate_id", f"{rp}.coordinates")
            for cid, coordinate in coordinates.items():
                cp = f"{rp}.coordinates[{cid}]"
                _identifier(coordinate.units, f"{cp}.units")
                _limits(coordinate.limits, f"{cp}.limits")
                _boolean(coordinate.periodic, f"{cp}.periodic")
                if coordinate.axis is not None:
                    _numeric_tuple(coordinate.axis, f"{cp}.axis", 3)
                    if not math.isclose(math.hypot(*coordinate.axis), 1.0, rel_tol=1e-6, abs_tol=1e-6):
                        _spec_error(f"{cp}.axis", "expected a unit axis")
            modes = _catalog(resource.modes, ResourceModeSpec, "mode_id", f"{rp}.modes")
            for mode_id, mode in modes.items():
                mp = f"{rp}.modes[{mode_id}]"
                _enum(mode.kind, {"continuous", "discrete"}, f"{mp}.kind")
                _enum(mode.codec, {"vector", "continuous", "categorical"}, f"{mp}.codec")
                if type(mode.payload_size) is not int or mode.payload_size < 1:
                    _spec_error(f"{mp}.payload_size", "expected a positive integer")
                if not isinstance(mode.units, tuple) or len(mode.units) != mode.payload_size:
                    _spec_error(f"{mp}.units", "one unit is required per ordered payload component")
                for unit in mode.units:
                    _identifier(unit, f"{mp}.units")
                _names(mode.categories, f"{mp}.categories")
                if mode.codec == "categorical":
                    if mode.kind != "discrete" or mode.payload_size != 1 or mode.units != ("1",) or not mode.categories:
                        _spec_error(mp, "categorical requires discrete kind, dimension 1, units ('1',) and categories")
                elif mode.categories:
                    _spec_error(f"{mp}.categories", "numeric codecs cannot declare categories")
    actions = _catalog(spec.actions, ActionSpec, "action_id", "actions")
    for aid, action in actions.items():
        path = f"actions[{aid}]"
        _enum(action.kind, {"continuous", "discrete", "prepare_connection", "connect", "disconnect", "skill"}, f"{path}.kind")
        _enum(action.scope, {"resource", "module", "modules", "connector_pair"}, f"{path}.scope")
        _boolean(action.enabled, f"{path}.enabled")
        _names(action.required_capabilities, f"{path}.required_capabilities")
        for name in action.required_capabilities:
            if name not in capabilities:
                _spec_error(f"{path}.required_capabilities", f"undefined capability {name!r}")
        params = _catalog(action.parameters, ParameterSpec, "name", f"{path}.parameters")
        for name, parameter in params.items():
            _validate_parameter(parameter, f"{path}.parameters[{name}]", modules)


def _decode_spec(cls: type, raw: object, path: str):
    if not isinstance(raw, dict):
        _spec_error(path, "expected a JSON object")
    definitions = {f.name: f for f in fields(cls)}
    unknown = sorted(set(raw) - set(definitions))
    if unknown:
        _spec_error(f"{path}.{unknown[0]}", "unknown field")
    values = {}
    children = _SPEC_CHILDREN.get(cls, {})
    for name, field in definitions.items():
        location = f"{path}.{name}"
        if name not in raw:
            if field.default is MISSING and field.default_factory is MISSING:
                _spec_error(location, "missing required field")
            continue
        value = raw[name]
        if name in children:
            if not isinstance(value, list):
                _spec_error(location, "expected a JSON array")
            value = tuple(_decode_spec(children[name], v, f"{location}[{i}]")
                          for i, v in enumerate(value))
        elif name in _TUPLE_FIELDS and value is not None:
            # ParameterSpec/CoordinateSpec.units is scalar, mode.units is an array.
            if name != "units" or cls is ResourceModeSpec:
                if not isinstance(value, list):
                    _spec_error(location, "expected a JSON array")
                value = tuple(value)
        elif name in {"size", "payload_size"} and type(value) is float and math.isfinite(value) and value.is_integer():
            # JSON Schema integer is mathematical, so 3.0 normalizes to int 3.
            value = int(value)
        values[name] = value
    return cls(**values)


def _semantic_value(value: object, field_name: str = ""):
    if is_dataclass(value):
        result = {f.name: _semantic_value(getattr(value, f.name), f.name) for f in fields(value)}
        if isinstance(value, ParameterSpec):
            result["codec"] = value.codec or _ARGUMENT_CODECS[value.kind]
        return result
    if isinstance(value, tuple):
        items = value
        if field_name in _CATALOG_KEYS:
            key = _CATALOG_KEYS[field_name]
            items = sorted(items, key=(lambda x: getattr(x, key)) if key else None)
        if field_name in {"local_position", "local_orientation_xyzw", "axis", "limits", "values"}:
            # Normalize 1 == 1.0 and signed zero without rounding large integers.
            return [int(x) if type(x) is float and x.is_integer() else x for x in items]
        return [_semantic_value(x) for x in items]
    return value


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            _spec_error("JSON", f"duplicate object key {key!r}")
        result[key] = value
    return result


def load_robot_spec(path: Path) -> RobotSpec:
    """Read strict UTF-8 JSON and return a validated, normalized frozen catalog."""
    def reject_constant(value: str):
        _spec_error("JSON", f"non-finite constant {value!r}")

    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"),
                         object_pairs_hook=_unique_json_object,
                         parse_constant=reject_constant)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"RobotSpec JSON: {error}") from error
    spec = _decode_spec(RobotSpec, raw, "RobotSpec")
    validate_robot_spec(spec)
    # Reconstruct canonical immutable catalog order and explicit argument codecs.
    normalized = _decode_spec(RobotSpec, _semantic_value(spec), "RobotSpec")
    validate_robot_spec(normalized)
    return normalized


def robot_spec_fingerprint(spec: RobotSpec) -> str:
    """SHA256 of the validated semantic catalog; excludes path/provenance/order."""
    validate_robot_spec(spec)
    payload = json.dumps(_semantic_value(spec), ensure_ascii=True, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
