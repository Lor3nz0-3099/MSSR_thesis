"""D2 closed, stdlib-only policy input view (mssr.bc_v1.policy_inputs.v1).

This boundary reads typed CURRENT physical/task objects, never Canonical,
RobotSpec, labels, evidence, source metadata, executor or future records.
Unknown NumericFeature names are omitted; malformed admitted values fail.
IDs and module_type_id remain opaque routing/catalog references. No ID-derived
numeric feature, embedding, sorting, frame transform or tensor is produced.

The task vocabulary/geometry below freezes D1.2's task contract independently
of robot hardware. New measured channels require an explicit whitelist change.
Module/resource extension features are empty in v1: their current physical
measurements already have dedicated pose/coordinate fields. Connector and
connection extension features admit only the physical channels listed below.
The caller certifies current-state and current-task ownership; numeric values
cannot themselves prove which timestep or task their source belongs to.
"""

from __future__ import annotations

from dataclasses import fields
import math
import re
from types import MappingProxyType

from .bc_v1_contract import (
    ConnectionState, ConnectorRef, ConnectorState, CoordinateState, ModuleRef,
    ModuleState, NumericFeature, ObservationBundle, PolicyInputs, ResourceRef,
    ResourceState, StateGraph, TaskContext,
)


POLICY_INPUTS_SCHEMA_VERSION = "mssr.bc_v1.policy_inputs.v1"
_TASKS = frozenset({"navigation", "gap", "stairs", "button"})
_FRAMES = frozenset({"world", "map", "stage_local", "body_local", "unframed"})
_CONNECTOR_FEATURES = MappingProxyType({
    "position_body_m": 3, "normal_body": 3, "tangent_body": 3,
})
_CONNECTION_FEATURES = MappingProxyType({
    "relative_position_m": 3, "relative_orientation_xyzw": 4,
    "clocking_quarter_turns": 1,
})
_RELATIONS = frozenset({"current_connection", "contact", "rolling_contact"})
_RELATIVE_DOFS = frozenset({"x", "y", "z", "roll", "pitch", "yaw", "normal", "tangent"})
# Entries: (number of components, spatial source frame required). None = >=1.
_DYNAMIC = MappingProxyType({
    "button.depression_m": (1, False),
    "button.current_center_xyz_m": (3, True),
})
_COMMON_STATIC = MappingProxyType({
    "environment.clear_landing_length_m": (1, False),
    "environment.floor_height_m": (1, True),
    "environment.upper_deck_height_m": (1, True),
    "environment.platform_bounds_xy_m": (4, True),
    "task_frame.origin_xyz_m": (3, True), "task_frame.yaw_rad": (1, True),
})
_TASK_STATIC = MappingProxyType({
    "gap": MappingProxyType({
        "gap.width_m": (1, False), "gap.bank_width_m": (1, False),
        "gap.bank_thickness_m": (1, False), "gap.landing_length_m": (1, False),
        "gap.near_edge_x_m": (1, True), "gap.far_edge_x_m": (1, True),
        "gap.near_edge_world_xyz_m": (3, True), "gap.far_edge_world_xyz_m": (3, True),
        "gap.near_edge_center_world_xyz_m": (3, True),
        "gap.far_edge_center_world_xyz_m": (3, True),
        "gap.crossing_direction_world_xy": (2, True),
    }),
    "stairs": MappingProxyType({
        "stairs.riser_depth_m": (1, False), "stairs.rise_m": (1, False),
        "stairs.tread_depth_m": (1, False), "stairs.width_m": (1, False),
        "stairs.step_count": (1, False), "stairs.first_riser_x_m": (1, True),
        "stairs.base_height_m": (1, True), "stairs.top_heights_m": (None, True),
        "stairs.first_riser_center_world_xyz_m": (3, True),
        "stairs.ascent_direction_world_xy": (2, True),
        "stairs.top_heights_world_m": (None, True),
    }),
    "button": MappingProxyType({
        "button.face_size_tangent_z_m": (2, False), "button.plunger_depth_m": (1, False),
        "button.plunger_stroke_m": (1, False), "button.center_xyz_m": (3, True),
        "button.press_direction_world_xy": (2, True),
        "button.press_direction_world_xyz": (3, True), "button.press_yaw_rad": (1, True),
    }),
    "navigation": MappingProxyType({
        "cones.cone_radius_m": (1, False), "cones.cone_height_m": (1, False),
        "cones.corridor_width_m": (1, False), "goal.center_xyz_m": (3, True),
        "goal.position_xyz_m": (3, True), "goal.xyyaw": (3, True),
    }),
})
_OBJECT_FEATURE = re.compile(r"(obstacles|environment_cones|cones)\.[0-9]+\.([a-z_]+)")
_OBSTACLE_FIELDS = MappingProxyType({
    "center_xyz_m": (3, True), "size_xyz_m": (3, True),
    "yaw_deg": (1, True), "pitch_deg": (1, True),
})
_CONE_FIELDS = MappingProxyType({
    "center_xyz_m": (3, True), "radius_m": (1, False), "height_m": (1, False),
})


def _instance(value, cls, strict):
    if type(value) is not cls:
        raise TypeError(f"Expected typed {cls.__name__}; raw mappings/subclasses are not policy inputs")
    if strict and set(vars(value)) != {field.name for field in fields(cls)}:
        raise ValueError(f"Unselected attributes in {cls.__name__}; use select_policy_inputs")
    return value


def _tuple(value):
    if type(value) is not tuple:
        raise TypeError("Policy collections must be immutable typed tuples")
    return value


def _number(value):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("Policy physical values must be finite numbers, not booleans")
    return value


def _numbers(value, size):
    value = _tuple(value)
    if not value or size is not None and len(value) != size:
        raise ValueError(f"Invalid physical feature/vector shape: expected length {size or '>=1'}")
    return tuple(_number(component) for component in value)


def _boolean(value):
    if type(value) is not bool:
        raise ValueError("Policy validity must be boolean")
    return value


def _identifier(value):
    if type(value) is not str or not value or any(char.isspace() for char in value):
        raise ValueError("Invalid opaque policy routing reference")
    return value


def _module_ref(value, strict):
    value = _instance(value, ModuleRef, strict)
    return ModuleRef(_identifier(value.module_id))


def _connector_ref(value, strict):
    value = _instance(value, ConnectorRef, strict)
    return ConnectorRef(_module_ref(value.module, strict), _identifier(value.connector_id))


def _resource_ref(value, strict):
    value = _instance(value, ResourceRef, strict)
    return ResourceRef(_module_ref(value.module, strict), _identifier(value.resource_id))


def _features(values, rule, strict):
    selected, seen = [], set()
    for feature in _tuple(values):
        feature = _instance(feature, NumericFeature, strict)
        if type(feature.name) is not str:
            raise TypeError("NumericFeature name must be a whitelisted string")
        spec = rule(feature.name)
        if spec is None:
            if strict:
                raise ValueError(f"Unselected feature {feature.name!r}; use select_policy_inputs whitelist")
            continue
        if feature.name in seen:
            raise ValueError(f"Duplicate policy feature {feature.name!r}")
        seen.add(feature.name)
        size, unframed = spec
        valid = _boolean(feature.valid)
        if unframed and valid:
            raise ValueError("Unframed policy geometry must have valid=False")
        selected.append(NumericFeature(feature.name, _numbers(feature.values, size), valid))
    return tuple(selected)


def _physical_rule(catalog):
    return lambda name: (catalog[name], False) if name in catalog else None


def _task_rule(name, task, dynamic=False):
    first, separator, rest = name.partition(".")
    frame = first if separator and first in _FRAMES else None
    base = rest if frame else name
    spec = _DYNAMIC.get(base) if dynamic else _COMMON_STATIC.get(base) or _TASK_STATIC.get(task, {}).get(base)
    if spec is None and not dynamic:
        object_field = _OBJECT_FEATURE.fullmatch(base)
        if object_field:
            group, field = object_field.groups()
            if group == "obstacles":
                spec = _OBSTACLE_FIELDS.get(field)
            elif task == "navigation" and group == "environment_cones":
                spec = _CONE_FIELDS.get(field)
            elif task == "navigation" and group == "cones" and field == "center_xy_m":
                spec = (2, True)
    if spec is None or spec[1] != (frame is not None):
        return None
    return spec[0], frame == "unframed"


def _select_graph(graph, strict):
    graph = _instance(graph, StateGraph, strict)
    modules, connectors, resources, connections = [], [], [], []
    for module in _tuple(graph.modules):
        module = _instance(module, ModuleState, strict)
        modules.append(ModuleState(
            ref=_module_ref(module.ref, strict), module_type_id=_identifier(module.module_type_id),
            position=_numbers(module.position, 3), orientation_xyzw=_numbers(module.orientation_xyzw, 4),
            linear_velocity=None if module.linear_velocity is None else _numbers(module.linear_velocity, 3),
            angular_velocity=None if module.angular_velocity is None else _numbers(module.angular_velocity, 3),
            pose_valid=_boolean(module.pose_valid), features=_features(module.features, _physical_rule({}), strict),
        ))
    for connector in _tuple(graph.connectors):
        connector = _instance(connector, ConnectorState, strict)
        if (type(connector.kind) is not str or connector.kind not in {"fixed_port", "surface"}
                or type(connector.status) is not str or connector.status not in {"unknown", "available", "connected"}):
            raise ValueError("Unknown current physical connector kind/status")
        connectors.append(ConnectorState(
            ref=_connector_ref(connector.ref, strict), kind=connector.kind,
            connected_to=None if connector.connected_to is None else _connector_ref(connector.connected_to, strict),
            status=connector.status, features=_features(connector.features, _physical_rule(_CONNECTOR_FEATURES), strict),
        ))
    for resource in _tuple(graph.resources):
        resource = _instance(resource, ResourceState, strict)
        coordinates = []
        for coordinate in _tuple(resource.coordinates):
            coordinate = _instance(coordinate, CoordinateState, strict)
            coordinates.append(CoordinateState(
                _identifier(coordinate.coordinate_id),
                None if coordinate.position is None else _number(coordinate.position),
                None if coordinate.velocity is None else _number(coordinate.velocity)))
        resources.append(ResourceState(_resource_ref(resource.ref, strict), tuple(coordinates),
                                       _features(resource.features, _physical_rule({}), strict)))
    for connection in _tuple(graph.connections):
        connection = _instance(connection, ConnectionState, strict)
        if type(connection.relation_id) is not str or connection.relation_id not in _RELATIONS:
            raise ValueError("Unknown current physical connection relation")
        dofs = _tuple(connection.allowed_relative_dofs)
        if any(type(dof) is not str or dof not in _RELATIVE_DOFS for dof in dofs):
            raise ValueError("Unknown physical relative degree of freedom")
        connections.append(ConnectionState(
            _connector_ref(connection.endpoint_a, strict), _connector_ref(connection.endpoint_b, strict),
            connection.relation_id, tuple(dof for dof in dofs),
            _features(connection.features, _physical_rule(_CONNECTION_FEATURES), strict)))
    return StateGraph(tuple(modules), tuple(connectors), tuple(resources), tuple(connections))


def _select_parts(graph, observation, context, *, strict):
    observation = _instance(observation, ObservationBundle, strict)
    context = _instance(context, TaskContext, strict)
    if context.task_id is not None and (type(context.task_id) is not str or context.task_id not in _TASKS):
        raise ValueError("Unknown externally assigned task; behavioral stages are not task IDs")
    return (
        _select_graph(graph, strict),
        ObservationBundle(_features(observation.features, lambda name: _task_rule(name, context.task_id, True), strict)),
        TaskContext(context.task_id, _features(context.static_geometry, lambda name: _task_rule(name, context.task_id), strict)),
    )


def _validate_selected_inputs(inputs: PolicyInputs) -> None:
    """Constructor guard: typed callers cannot bypass the feature whitelist."""
    _instance(inputs, PolicyInputs, True)
    _select_parts(inputs.graph, inputs.observation, inputs.context, strict=True)


def select_policy_inputs(
    graph: StateGraph,
    observation: ObservationBundle,
    context: TaskContext,
) -> PolicyInputs:
    """Explicitly copy the frozen physical/task whitelist into a detached view.

    Unknown feature names/undeclared attributes are omitted. Wrong contracts,
    admitted malformed values and unknown task/physical enums fail explicitly.
    No label/audit/raw argument exists. The output's direct constructor also
    validates this same whitelist, so raw dictionaries cannot bypass it.
    """
    selected_graph, selected_observation, selected_context = _select_parts(graph, observation, context, strict=False)
    return PolicyInputs(selected_graph, selected_observation, selected_context)
