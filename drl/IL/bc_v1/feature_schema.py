"""Framework-neutral E2 schema contract and current-robot extraction.

Import/load/validation remain stdlib-only. Extraction lazily uses the existing
E1 codecs and frozen D2 boundary; no transforms,
candidates, dataset IO, tensors or statistics are implemented here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from .bc_v1_contract import ConnectorRef, ModuleRef, ResourceRef

if TYPE_CHECKING:
    from .bc_v1_contract import PolicyInputs, RobotSpec
    from .features import FrameGeometryView, PhysicalFrame
import hashlib
import json
import math
import re
from pathlib import Path


_VERSION = 'mssr.bc_v1.features.v1'
_TOP_KEYS = {
    'schema_version', 'input_contract', 'geometry_contract', 'frame_policy',
    'r6_convention', 'missingness', 'cardinality', 'robot_spec_policy',
    'unit_resolvers', 'source_shape_policy', 'routing_metadata', 'geometry_kinds', 'entities',
    'dedup_policy', 'relation_policy', 'candidate_policy',
}
_ROUTING = {
    'module_ref', 'module_type_ref', 'connector_ref', 'resource_ref',
    'coordinate_ref', 'connection_ref', 'object_ref', 'source_scope',
    'source_name', 'geometry_kind', 'primitive_part_ref', 'action_id',
    'scope_members', 'parameter_binding_ref',
}
_CODEC_WIDTHS = {'point3': 3, 'vector3': 3, 'vector2': 2, 'r6': 6,
                 'quaternion_r6': 6, 'scalar': 1, 'boolean': 1}
# (codec, unit, existing physical frame, symbolic dimension/category set).
_SOURCES = {
    'module': {
        'module.position': ('point3', 'm', 'robot_centroid', None),
        'module.orientation_matrix': ('r6', '1', 'robot_centroid', None),
        'module.linear_velocity': ('vector3', 'm/s', 'robot_centroid', None),
        'module.angular_velocity': ('vector3', 'rad/s', 'robot_centroid', None),
    },
    'connector': {
        'connector.position_body_m': ('point3', 'm', 'module_body', None),
        'connector.normal_body': ('vector3', '1', 'module_body', None),
        'connector.tangent_body': ('vector3', '1', 'module_body', None),
        'connector.kind': ('one_hot', '1', 'intrinsic', {'fixed_port', 'surface'}),
        'connector.status': ('one_hot', '1', 'intrinsic', {'available', 'connected'}),
        'connector.capabilities': ('multi_hot', '1', 'intrinsic', 'robot_spec.connector_capabilities'),
    },
    'resource': {
        'resource.capabilities': ('multi_hot', '1', 'intrinsic', 'robot_spec.resource_capabilities'),
    },
    'actuator_coordinate': {
        'coordinate.position': ('scalar', 'coordinate_units', 'resource_local', None),
        'coordinate.velocity': ('scalar', 'coordinate_units_per_s', 'resource_local', None),
        'coordinate.limits': ('vector2', 'coordinate_units', 'resource_local', None),
        'coordinate.periodic': ('boolean', '1', 'intrinsic', None),
    },
    'connection': {
        'connection.relative_position_m': ('point3', 'm', 'endpoint_a_body', None),
        'connection.relative_orientation_xyzw': ('quaternion_r6', '1', 'endpoint_a_body', None),
        'connection.clocking_quarter_turns': ('scalar', 'quarter_turn', 'connection_intrinsic', None),
        'connection.allowed_relative_dofs': ('multi_hot', '1', 'intrinsic', {'x','y','z','roll','pitch','yaw','normal','tangent'}),
        'connection.relation_id': ('one_hot', '1', 'intrinsic', {'current_connection','contact','rolling_contact'}),
    },
    'environment': {
        'environment.geometry': ('geometry_kind', 'geometry_kind_units', 'from_e1_output_frame', 'geometry_kind'),
        'environment.object_type': ('one_hot', '1', 'intrinsic', {'button','cone','goal','gap_boundary','stair_riser','collision_box','task_frame','environment_boundary','intrinsic_quantity'}),
    },
    'module_environment_relation': {
        'relation.delta_xyz': ('vector3', 'm', 'robot_centroid', None),
        'relation.distance': ('scalar', 'm', 'intrinsic', None),
        'relation.normalized_direction': ('vector3', '1', 'robot_centroid', None),
    },
    'structural_candidate_metadata': {},
}
_SCOPES = {
    name: {'g_current'} for name in _SOURCES
}
_SCOPES['environment'] = {'c_static', 'o_dynamic'}
_SCOPES['module_environment_relation'] = {'g_current', 'c_static', 'o_dynamic'}
_ENTITY_ROUTING = {
    'module': {'module_ref','module_type_ref'},
    'connector': {'module_ref','connector_ref'},
    'resource': {'module_ref','resource_ref'},
    'actuator_coordinate': {'module_ref','resource_ref','coordinate_ref'},
    'connection': {'connection_ref','connector_ref','module_ref'},
    'environment': {'object_ref','source_scope','source_name','geometry_kind','primitive_part_ref'},
    'module_environment_relation': {'module_ref','object_ref','source_scope','source_name','primitive_part_ref'},
    'structural_candidate_metadata': {'action_id','module_ref','connector_ref','resource_ref','scope_members','parameter_binding_ref'},
}
# Fixed physical meanings/arity; learned packing widths come from config fields.
_KINDS = {
    'point3': (['point'], 3, ['m']*3, 'one_row', 'always'),
    'vector3': (['vector'], 3, ['1']*3, 'one_row', 'always'),
    'line': (['line'], 6, ['m']*3+['1']*3, 'one_row_unknown_height_preserved', 'always'),
    'plane': (['planes'], 4, ['1']*3+['m'], 'one_plane_per_row_preserve_source_group', 'always'),
    'heading': (['heading'], 3, ['1']*3, 'one_row', 'always'),
    'planar_pose': (['planar_pose'], 9, ['m']*3+['1']*6, 'one_row_unknown_height_preserved', 'always'),
    'angle_constraint': (['angle_constraint'], 10, ['rad']+['1']*9, 'one_row_partial_attitude_preserved', 'always'),
    'intrinsic_scalar': (['intrinsic'], 1, ['source_quantity_units'], 'one_row', 'length_eq_1'),
    'intrinsic_vector': (['intrinsic','dimensions'], 'source_components', 'source_quantity_units', 'one_row', 'length_gt_1'),
}
_CANDIDATES = {
    'align_faces': ('prepare_connection','compatible_connector_pair','from_robot_spec'),
    'dock': ('connect','compatible_connector_pair','from_robot_spec'),
    'undock': ('disconnect','current_connection_pair','from_robot_spec'),
    'gravity_settle': ('skill','none','from_robot_spec_module_scope'),
}
_SET_LISTS = {'source_scopes','routing_fields','allowed_inputs','e1_kinds','dimensionless_names'}


def _error(path, message):
    raise ValueError(f'{path}: {message}')


def _json_domain(value, path='schema'):
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                _error(path, 'keys must be strings')
            _json_domain(item, f'{path}.{key}')
    elif type(value) is list:
        for i, item in enumerate(value):
            _json_domain(item, f'{path}[{i}]')
    elif type(value) not in (str, bool, int, float, type(None)):
        _error(path, 'expected plain JSON configuration values')
    elif type(value) is float and not math.isfinite(value):
        _error(path, 'nonfinite configuration value')


def _object(value, keys, path):
    if not isinstance(value, Mapping):
        _error(path, 'expected an object')
    if set(value) != set(keys):
        _error(path, f'unknown/missing keys: {sorted(set(value)^set(keys))}')


def _literal(value, expected, path):
    """Exact JSON literals, including bool/int distinction."""
    if isinstance(expected, dict):
        _object(value, expected, path)
        for key, item in expected.items():
            _literal(value[key], item, f'{path}.{key}')
    elif type(expected) is list:
        if type(value) is not list or len(value) != len(expected):
            _error(path, f'expected {expected!r}')
        for i,(actual,want) in enumerate(zip(value,expected)):
            _literal(actual,want,f'{path}[{i}]')
    elif type(value) is not type(expected) or value != expected:
        _error(path, f'expected {expected!r}')


def _names(value, expected, path):
    if type(value) is not list or any(type(v) is not str or not v for v in value):
        _error(path, 'expected a list of named entries')
    if len(value) != len(set(value)) or set(value) != set(expected):
        _error(path, f'expected distinct entries {sorted(expected)!r}')


def _validate_entities(schema):
    _object(schema['entities'], _SOURCES, 'entities')
    for family, sources in _SOURCES.items():
        path = f'entities.{family}'
        entity = schema['entities'][family]
        _object(entity, {'source_scopes','routing_fields','packing','fields'}, path)
        _names(entity['source_scopes'], _SCOPES[family], path+'.source_scopes')
        _names(entity['routing_fields'], _ENTITY_ROUTING[family], path+'.routing_fields')
        _literal(entity['packing'], 'routing_only' if not sources else 'all_values_then_matching_validity', path+'.packing')
        fields = entity['fields']
        if type(fields) is not list:
            _error(path+'.fields', 'expected an ordered list')
        seen_names, seen_sources = set(), set()
        for i, field in enumerate(fields):
            fp = f'{path}.fields[{i}]'
            if not isinstance(field, Mapping):
                _error(fp, 'expected field object')
            _object(field, {'name','source','codec','dimension','units','frame','validity'} | ({'categories'} if 'categories' in field else set()), fp)
            name = field['name']
            if type(name) is not str or not name or not name.isidentifier() or name in seen_names or name in _ROUTING:
                _error(fp+'.name', 'invalid/duplicate learned field or routing ID')
            seen_names.add(name)
            source = field['source']
            if type(source) is not str or source not in sources or source in seen_sources:
                _error(fp+'.source', 'unknown, noncausal or duplicate physical source')
            seen_sources.add(source)
            codec,units,frame,dimension = sources[source]
            _literal(field['codec'],codec,fp+'.codec')
            _literal(field['units'],units,fp+'.units')
            _literal(field['frame'],frame,fp+'.frame')
            _literal(field['validity'],'per_component',fp+'.validity')
            if isinstance(dimension,set):
                _literal(field['dimension'],'categories',fp+'.dimension')
                _names(field.get('categories'),dimension,fp+'.categories')
            else:
                if 'categories' in field:
                    _error(fp+'.categories','only declared categorical sources have categories')
                _literal(field['dimension'], _CODEC_WIDTHS[codec] if dimension is None else dimension, fp+'.dimension')
        if seen_sources != set(sources):
            _error(path+'.fields', 'missing required physical sources')
    module = schema['entities']['module']['fields']
    _literal([f['name'] for f in module], ['relative_position','orientation_r6','linear_velocity','angular_velocity'], 'entities.module.fields.order')
    _literal([f['source'] for f in module], list(_SOURCES['module']), 'entities.module.fields.source_order')


def validate_feature_schema(schema: Mapping[str, object]) -> None:
    """Validate closed v1 semantics; errors are path-qualified ValueError."""
    _json_domain(schema)
    _object(schema, _TOP_KEYS, 'schema')
    _literal(schema['schema_version'],_VERSION,'schema_version')
    _literal(schema['input_contract'],'mssr.bc_v1.policy_inputs.v1','input_contract')
    _literal(schema['geometry_contract'],'FrameGeometryView','geometry_contract')
    _literal(schema['frame_policy'], {
        'owner':'E1','origin':'centroid_all_current_valid_modules',
        'preference':['certified_current_physical_task_frame','certified_map_or_world_frame'],
        'same_for_training_and_rollout':True,'missing_certification':'invalid_no_inferred_frame',
    }, 'frame_policy')
    _literal(schema['r6_convention'], {'owner':'E1.matrix_to_r6','columns':[0,1],'flatten':'column_order'}, 'r6_convention')
    _literal(schema['missingness'], {'value_fill':'zero_only_with_invalid_channel','validity':'per_component',
        'absent':'invalid','nonfinite_valid':'error','unknown_category':'invalid'}, 'missingness')
    _literal(schema['cardinality'], {k:'variable' for k in ('modules','connectors','resources','coordinates')}, 'cardinality')
    _literal(schema['robot_spec_policy'], {'capability_vocabulary':'sorted_union_by_entity_kind',
        'resource_rows':'one_per_current_resource','coordinate_rows':'one_per_current_coordinate',
        'argument_contract':'robot_spec_action_parameters','instance_ids':'routing_only'}, 'robot_spec_policy')
    _literal(schema['unit_resolvers'], {'coordinate_units':'RobotSpec.coordinate.units',
        'coordinate_units_per_s':'RobotSpec.coordinate.units_divided_by_seconds',
        'geometry_kind_units':'geometry_kinds.units',
        'source_quantity_units':{'rule':'approved_source_name_unit','metric_suffix':'_m','dimensionless_names':['stairs.step_count']}}, 'unit_resolvers')
    _literal(schema['source_shape_policy'], {
        'owner':'validated_policy_inputs',
        'scope_sources':{'c_static':'context.static_geometry','o_dynamic':'observation.features'},
        'lookup':'source_scope_and_source_name','read':'component_count_only','learned':False,
        'intrinsic_dispatch':'source_arity_even_when_values_missing',
        'intrinsic_vector_width':'source_arity','plane_row_count':'source_arity',
        'unknown_shape':'error_no_zero_or_scalar_guess',
    }, 'source_shape_policy')
    _object(schema['routing_metadata'],_ROUTING,'routing_metadata')
    for name, entry in schema['routing_metadata'].items():
        _literal(entry, {'learned':False,'encoding':'opaque_routing'}, f'routing_metadata.{name}')
    _object(schema['geometry_kinds'],_KINDS,'geometry_kinds')
    for kind, (aliases, dimension, units, policy, condition) in _KINDS.items():
        kp = f'geometry_kinds.{kind}'; entry = schema['geometry_kinds'][kind]
        _object(entry, {'e1_kinds','dimension','units','row_policy','when'},kp)
        _names(entry['e1_kinds'],aliases,kp+'.e1_kinds')
        for key,want in (('dimension',dimension),('units',units),('row_policy',policy),('when',condition)):
            _literal(entry[key],want,kp+'.'+key)
    _validate_entities(schema)
    dedup = schema['dedup_policy']
    _object(dedup, {'keep_distinct','canonicalize_equivalent'},'dedup_policy')
    _literal(dedup['keep_distinct'], {'button_nominal_current':'preserve_nominal_and_current_roles',
        'goal_point_support_volume':'preserve_destination_and_support_roles',
        'module_state_relative_edge':'preserve_absolute_and_relative_semantics'}, 'dedup_policy.keep_distinct')
    rules = {'cone_centers':'certified_3d_over_2d','cone_dimensions':'per_object_observed_over_shared_default',
             'gap_edges':'most_informative_certified_geometry','stairs':'most_informative_certified_geometry'}
    _object(dedup['canonicalize_equivalent'],rules,'dedup_policy.canonicalize_equivalent')
    for name, preference in rules.items():
        _literal(dedup['canonicalize_equivalent'][name], {'requires_physical_equivalence':True,
            'prefer':preference,'partial_geometry':'never_invent_missing_components'}, f'dedup_policy.canonicalize_equivalent.{name}')
    relation = schema['relation_policy']
    _object(relation, {'allowed_inputs','selection','object_types_by_task','geometry_policy','undefined_direction'},'relation_policy')
    _names(relation['allowed_inputs'], {'current_object_type','current_task_semantics','current_physical_geometry'}, 'relation_policy.allowed_inputs')
    selection = relation['selection']
    _object(selection, {'rule','max_neighbors_per_type','limit_semantics','tie_policy','distance_metric','unknown_task'}, 'relation_policy.selection')
    limit = selection['max_neighbors_per_type']
    if type(limit) is not int or limit < 1:
        _error('relation_policy.selection.max_neighbors_per_type','expected positive integer schema budget')
    for key,want in {'rule':'nearest_valid_per_module_per_object_type', 'limit_semantics':'soft_at_equal_physical_distance',
            'tie_policy':'retain_boundary_ties_no_instance_id_tiebreak','distance_metric':'kind_aware_closest_geometry',
            'unknown_task':'no_relations'}.items():
        _literal(selection[key],want,'relation_policy.selection.'+key)
    types = {'button':{'button','collision_box'},'navigation':{'cone','goal','collision_box'},
             'gap':{'gap_boundary','collision_box'},'stairs':{'stair_riser','collision_box'}}
    _object(relation['object_types_by_task'],types,'relation_policy.object_types_by_task')
    for task, names in types.items():
        _names(relation['object_types_by_task'][task],names,f'relation_policy.object_types_by_task.{task}')
    _literal(relation['geometry_policy'],'kind_aware_no_partial_point_cast','relation_policy.geometry_policy')
    _literal(relation['undefined_direction'],'invalid_at_zero_distance','relation_policy.undefined_direction')
    candidate = schema['candidate_policy']
    _object(candidate, {'allowed_inputs','robot_spec_enabled_only','argument_contract','feasibility_owner','actions'}, 'candidate_policy')
    _names(candidate['allowed_inputs'], {'current_state_graph','robot_spec','connector_compatibility',
        'current_connection_state','action_argument_requirements'},'candidate_policy.allowed_inputs')
    for key,want in {'robot_spec_enabled_only':True,'argument_contract':'robot_spec_action_parameters','feasibility_owner':'runtime_executor'}.items():
        _literal(candidate[key],want,'candidate_policy.'+key)
    _object(candidate['actions'],_CANDIDATES,'candidate_policy.actions')
    for name,(kind,endpoints,scope) in _CANDIDATES.items():
        _literal(candidate['actions'][name], {'kind':kind,'endpoints':endpoints,'scope':scope,
            'distance_only_pruning':False,'enabled_by_default':True},'candidate_policy.actions.'+name)


def _unique_object(pairs):
    result = {}
    for name,value in pairs:
        if name in result:
            _error(name,'duplicate JSON key')
        result[name] = value
    return result


def _reject_constant(value):
    _error('schema',f'nonfinite JSON literal {value}')


def load_feature_schema(path: Path | str) -> Mapping[str, object]:
    """Load unambiguous JSON and validate; no side effects or source readers."""
    schema = json.loads(Path(path).read_text(encoding='utf-8'),
                        object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    validate_feature_schema(schema)
    return schema


def _canonical(value, path=()):
    if isinstance(value,Mapping):
        return {k:_canonical(v,(*path,k)) for k,v in value.items()}
    if type(value) is list:
        items = [_canonical(v,path) for v in value]
        return sorted(items) if path and (path[-1] in _SET_LISTS or 'object_types_by_task' in path) else items
    return value


def feature_schema_fingerprint(schema: Mapping[str, object]) -> str:
    """SHA256 of validated semantic JSON, sorted keys, UTF-8 compact encoding.

    Unordered input/routing/alias sets canonicalized; field/category/physical
    component order preserved. No path, whitespace, timestamp or mutation.
    """
    validate_feature_schema(schema)
    encoded = json.dumps(_canonical(schema), sort_keys=True, separators=(',',':'),
                         ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def module_base_width(schema: Mapping[str, object]) -> int:
    """Values plus one explicit validity channel for each configured component."""
    validate_feature_schema(schema)
    fields = schema['entities']['module']['fields']
    values = sum(field['dimension'] for field in fields)
    validity = sum(field['dimension'] for field in fields if field['validity']=='per_component')
    return values + validity


@dataclass(frozen=True)
class FeatureField:
    """Immutable resolved column layout; names/units/frames are metadata.

    channels records configured categories or sorted RobotSpec capabilities.
    No instance ID or catalog ID is encoded as a learned column.
    """

    name: str
    width: int
    units: tuple[str, ...]
    frame: str
    channels: tuple[str, ...] = ()


EnvironmentRef = tuple[str, str, str, int]
RowRef = ModuleRef | ConnectorRef | ResourceRef | tuple[ResourceRef, str] | tuple[ConnectorRef, ConnectorRef] | EnvironmentRef | tuple[ModuleRef, tuple[EnvironmentRef, ...]]


@dataclass(frozen=True)
class FeatureRow:
    """All value components followed by matching validity in ``numeric``.

    ref and catalog_ref are opaque routing metadata. fields describes only
    value columns; validity has exactly the same order and component count.
    Missing numeric components are zero with validity False.
    """

    ref: RowRef
    values: tuple[float, ...]
    validity: tuple[bool, ...]
    fields: tuple[FeatureField, ...]
    catalog_ref: str | None = None
    source_scope: str = 'g_current'
    object_ref: str | None = None
    object_type: str | None = None
    source_name: str | None = None
    geometry_kind: str | None = None
    output_frame: str | None = None
    primitive_part_ref: int | None = None
    module_ref: ModuleRef | None = None
    relation_kind: str | None = None
    geometry_sources: tuple[EnvironmentRef, ...] = ()

    @property
    def numeric(self) -> tuple[float, ...]:
        return self.values + tuple(float(v) for v in self.validity)


@dataclass(frozen=True)
class FeatureView:
    """Current robot/environment/relation rows plus the exact E1 physical frame.

    Row order follows current PolicyInputs collections, never instance-ID
    sorting. Coordinate refs are (ResourceRef, coordinate_id); connection refs
    are ordered (endpoint_a, endpoint_b). No batch indices or tensor packing.
    """

    schema_version: str
    schema_fingerprint: str
    frame: PhysicalFrame
    modules: tuple[FeatureRow, ...]
    connectors: tuple[FeatureRow, ...]
    resources: tuple[FeatureRow, ...]
    coordinates: tuple[FeatureRow, ...]
    connections: tuple[FeatureRow, ...]
    environment: tuple[FeatureRow, ...] = ()
    relations: tuple[FeatureRow, ...] = ()


def _index(items, key, path):
    result = {}
    for item in items:
        ref = key(item)
        if ref in result:
            _error(path, 'duplicate current routing reference')
        result[ref] = item
    return result


def _lookup(index, ref, path):
    if ref not in index:
        _error(path, 'unresolved current routing/catalog reference')
    return index[ref]


def _physical_values(values, width, path):
    if type(values) is not tuple or len(values) != width:
        _error(path, 'invalid physical component shape')
    if any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
        _error(path, 'valid physical components must be finite numbers')
    return tuple(float(v) for v in values)


def _resolved_fields(schema, family, vocabularies, coordinate_spec=None):
    fields = []
    for field in schema['entities'][family]['fields']:
        dimension = field['dimension']
        channels = tuple(field['categories']) if dimension == 'categories' else vocabularies.get(dimension, ())
        width = dimension if type(dimension) is int else len(channels)
        units = field['units']
        if units in ('coordinate_units', 'coordinate_units_per_s'):
            units = coordinate_spec.units + ('/s' if units == 'coordinate_units_per_s' else '')
        fields.append(FeatureField(field['name'], width, (units,)*width, field['frame'], channels))
    return tuple(fields)


def _robot_row(ref, family, sources, schema, vocabularies, coordinate_spec=None, catalog_ref=None):
    # E1 import stays lazy: schema validation never imports NumPy/geometry.
    from .features import matrix_to_r6, quaternion_to_r6

    fields = _resolved_fields(schema, family, vocabularies, coordinate_spec)
    values, validity = [], []
    for declaration, layout in zip(schema['entities'][family]['fields'], fields):
        source = sources[declaration['source']]
        codec = declaration['codec']
        if source is not None and codec == 'one_hot':
            source = tuple(float(source == category) for category in layout.channels) if source in layout.channels else None
        elif source is not None and codec == 'multi_hot':
            source = tuple(float(category in source) for category in layout.channels) if set(source) <= set(layout.channels) else None
        elif source is not None and codec == 'r6':
            source = matrix_to_r6(source)
        elif source is not None and codec == 'quaternion_r6':
            source = quaternion_to_r6(source)
        if source is None:
            components, valid = (0.,)*layout.width, (False,)*layout.width
        else:
            components = _physical_values(source, layout.width, family+'.'+layout.name)
            valid = (True,)*layout.width
        values.extend(components)
        validity.extend(valid)
    return FeatureRow(ref, tuple(values), tuple(validity), fields, catalog_ref)


def _current_features(state, prefix, names):
    features = {f.name: f for f in state.features}
    return {prefix+'.'+name: features[name].values if name in features and features[name].valid else None
            for name in names}



def _validate_frame_metadata(frame):
    """Validate the carried E1 record without constructing/selecting a frame.

    Reuse E1's physical identifier/proper-rotation checks via its config
    validator. Nothing calls physical_frame or transforms any coordinates.
    Enforce immutable tuples before E1 can normalize mutable containers.
    """
    from .features import PhysicalFrameConfig

    if type(frame.valid) is not bool or frame.valid != (frame.origin is not None):
        _error('geometry.frame', 'explicit boolean validity must match origin availability')
    if frame.origin is not None:
        _physical_values(frame.origin, 3, 'geometry.frame.origin')
    _physical_values(frame.rotation_source_from_axes, 9, 'geometry.frame.rotation')
    PhysicalFrameConfig(graph_frame=frame.graph_frame, axes_id=frame.axes_id,
                        rotation_source_from_axes=frame.rotation_source_from_axes)


def extract_features(
    inputs: PolicyInputs,
    geometry: FrameGeometryView,
    spec: RobotSpec,
    schema: Mapping[str, object],
) -> FeatureView:
    """Encode current robot and approved current-task environment physical state.

    Caller must supply the E1 view of these same current inputs. We certify
    reference correspondence and retained local state, not recompute module
    transforms to compare their numbers. E1 owns source/axes certification and
    selection; its exact PhysicalFrame object is carried unchanged.
    """
    from .bc_v1_contract import validate_robot_spec
    from .features import FrameGeometryView, ModuleGeometry, PhysicalFrame
    from .policy_inputs import _validate_selected_inputs

    _validate_selected_inputs(inputs)
    validate_robot_spec(spec)
    fingerprint = feature_schema_fingerprint(schema)
    if type(geometry) is not FrameGeometryView or type(geometry.frame) is not PhysicalFrame:
        raise TypeError('Expected typed E1 FrameGeometryView/PhysicalFrame')
    _validate_frame_metadata(geometry.frame)
    if type(geometry.modules) is not tuple or any(type(m) is not ModuleGeometry for m in geometry.modules):
        raise TypeError('Expected immutable E1 module geometry rows')
    graph = inputs.graph
    modules = _index(graph.modules, lambda m:m.ref, 'modules')
    derived = _index(geometry.modules, lambda m:m.ref, 'geometry.modules')
    if set(modules) != set(derived):
        _error('geometry.modules', 'must describe exactly the current modules')
    for name in ('connectors','resources','connections'):
        if getattr(geometry,name) != getattr(graph,name):
            _error('geometry.'+name, 'local state must match the current inputs')
    types = {t.type_id:t for t in spec.module_types}
    catalog = {m.ref:_lookup(types,m.module_type_id,'module_type') for m in graph.modules}
    connectors = _index(graph.connectors, lambda x:x.ref, 'connectors')
    _index(graph.resources, lambda x:x.ref, 'resources')
    _index(graph.connections, lambda x:(x.endpoint_a,x.endpoint_b), 'connections')
    vocabularies = {
        'robot_spec.connector_capabilities': tuple(sorted({cap for t in spec.module_types for port in t.connectors for cap in port.capabilities})),
        'robot_spec.resource_capabilities': tuple(sorted({cap for t in spec.module_types for resource in t.resources for cap in resource.capabilities})),
    }
    module_rows, connector_rows, resource_rows, coordinate_rows, connection_rows = [], [], [], [], []
    for module in graph.modules:
        physical = derived[module.ref]
        if physical.source_scope != 'g_current' or physical.output_frame != 'robot_centroid':
            _error('geometry.modules', 'expected E1 current robot frame geometry')
        if type(physical.pose_valid) is not bool:
            _error('geometry.modules', 'pose validity must be explicit')
        if physical.pose_valid != (module.pose_valid and geometry.frame.valid):
            _error('geometry.modules', 'inconsistent E1 pose/frame validity')
        if (physical.position is not None) != physical.pose_valid or (physical.orientation_matrix is not None) != physical.pose_valid:
            _error('geometry.modules', 'pose components must match E1 validity')
        for name in ('linear_velocity','angular_velocity'):
            if (getattr(physical,name) is None) != (getattr(module,name) is None):
                _error('geometry.modules', 'measured twist missingness must be preserved')
        sources = {'module.'+name:getattr(physical,name) for name in
                   ('position','orientation_matrix','linear_velocity','angular_velocity')}
        module_rows.append(_robot_row(module.ref,'module',sources,schema,vocabularies,catalog_ref=module.module_type_id))
    for connector in geometry.connectors:
        mt = _lookup(catalog,connector.ref.module,'connector.module')
        port = _lookup({p.connector_id:p for p in mt.connectors},connector.ref.connector_id,'connector.catalog')
        if port.kind != connector.kind:
            _error('connector.kind', 'current kind disagrees with RobotSpec')
        if connector.connected_to is not None:
            _lookup(connectors,connector.connected_to,'connector.connected_to')
        sources = _current_features(connector,'connector',('position_body_m','normal_body','tangent_body'))
        sources.update({'connector.kind':connector.kind, 'connector.status':connector.status,
                        'connector.capabilities':port.capabilities})
        connector_rows.append(_robot_row(connector.ref,'connector',sources,schema,vocabularies))
    for resource in geometry.resources:
        mt = _lookup(catalog,resource.ref.module,'resource.module')
        rs = _lookup({r.resource_id:r for r in mt.resources},resource.ref.resource_id,'resource.catalog')
        resource_rows.append(_robot_row(resource.ref,'resource',{'resource.capabilities':rs.capabilities},schema,vocabularies))
        coordinates = {x.coordinate_id:x for x in rs.coordinates}
        _index(resource.coordinates,lambda x:x.coordinate_id,'coordinates')
        for coordinate in resource.coordinates:
            cs = _lookup(coordinates,coordinate.coordinate_id,'coordinate.catalog')
            sources = {'coordinate.position':None if coordinate.position is None else (coordinate.position,),
                       'coordinate.velocity':None if coordinate.velocity is None else (coordinate.velocity,),
                       'coordinate.limits':cs.limits, 'coordinate.periodic':(float(cs.periodic),)}
            coordinate_rows.append(_robot_row((resource.ref,coordinate.coordinate_id),'actuator_coordinate',sources,schema,vocabularies,cs))
    for connection in geometry.connections:
        for endpoint in (connection.endpoint_a,connection.endpoint_b):
            _lookup(connectors,endpoint,'connection.endpoint')
        sources = _current_features(connection,'connection',('relative_position_m','relative_orientation_xyzw','clocking_quarter_turns'))
        sources.update({'connection.allowed_relative_dofs':connection.allowed_relative_dofs,
                        'connection.relation_id':connection.relation_id})
        connection_rows.append(_robot_row((connection.endpoint_a,connection.endpoint_b),'connection',sources,schema,vocabularies))
    environment = _environment_rows(inputs, geometry, schema)
    relations = _module_environment_relations(tuple(module_rows), environment, inputs.context.task_id, schema)
    return FeatureView(schema['schema_version'], fingerprint, geometry.frame,
        tuple(module_rows), tuple(connector_rows), tuple(resource_rows), tuple(coordinate_rows), tuple(connection_rows),
        environment, relations)


_SOURCE_FRAMES = frozenset({'world','map','stage_local','unframed','body_local'})
_ENV_OBJECT = re.compile(r'(cones|environment_cones|obstacles)\.([0-9]+)\.(.+)')


def _source_parts(name):
    first, separator, rest = name.partition('.')
    return (first,rest) if separator and first in _SOURCE_FRAMES else (None,name)


def _environment_identity(base):
    """Closed physical source semantics, never numeric instance ordinals."""
    match = _ENV_OBJECT.fullmatch(base)
    if match:
        group, ref, _ = match.groups()
        if group == 'obstacles':
            return 'obstacles.'+ref,'collision_box'
        # D1 independently enumerates legacy XY and physical cone inventories.
        # Their ordinals are distinct routing namespaces, never object identity.
        return ('cone_xy.' if group == 'cones' else 'cone.')+ref,'cone'
    group, _, field = base.partition('.')
    if group == 'cones':
        return ('cone.shared','cone') if field in ('cone_radius_m','cone_height_m') else ('environment.corridor','environment_boundary')
    if group in ('button','goal','task_frame'):
        return group,group
    if group == 'gap':
        role = 'near_edge' if field.startswith('near_edge') else 'far_edge' if field.startswith('far_edge') else None
        return ('gap.'+role if role else 'gap','gap_boundary')
    if group == 'stairs':
        role = 'first_riser' if field.startswith('first_riser') else 'top_heights' if field.startswith('top_heights') else field
        return 'stairs.'+role,'stair_riser'
    if group == 'environment':
        return base,'intrinsic_quantity' if field == 'clear_landing_length_m' else 'environment_boundary'
    _error('environment.source_name','unresolved approved physical object semantics')


def _environment_layout(quantity, source, schema):
    kind = next((name for name,entry in schema['geometry_kinds'].items()
                 if quantity.kind in entry['e1_kinds'] and
                 (entry['when']=='always' or entry['when']=='length_eq_1' and len(source.values)==1
                  or entry['when']=='length_gt_1' and len(source.values)>1)), None)
    if kind is None:
        _error('environment.geometry_kind','unknown geometry or source shape; no guessing')
    entry = schema['geometry_kinds'][kind]
    width = len(source.values) if entry['dimension']=='source_components' else entry['dimension']
    units = entry['units']
    if units == 'source_quantity_units' or units == ['source_quantity_units']:
        _,base = _source_parts(quantity.source_name)
        resolver = schema['unit_resolvers']['source_quantity_units']
        unit = 'm' if base.endswith(resolver['metric_suffix']) else '1' if base in resolver['dimensionless_names'] else None
        if unit is None:
            _error('environment.units','unresolved approved physical units')
        units = (unit,)*width
    else:
        units = tuple(units)
    parts = len(source.values) if kind == 'plane' else 1
    return kind,width,units,parts


def _environment_row(quantity, object_ref, object_type, kind, width, units, part, schema):
    values, validity, fields = [], [], []
    for declaration in schema['entities']['environment']['fields']:
        if declaration['codec']=='geometry_kind':
            layout = FeatureField(declaration['name'],width,units,quantity.output_frame)
            payload = (0.,)*width if not quantity.valid else quantity.values[part*width:(part+1)*width]
            valid = (quantity.valid,)*width
        else:
            channels = tuple(declaration['categories'])
            layout = FeatureField(declaration['name'],len(channels),('1',)*len(channels),'intrinsic',channels)
            payload = tuple(float(t==object_type) for t in channels)
            valid = (True,)*len(channels)
        fields.append(layout); values.extend(payload); validity.extend(valid)
    return FeatureRow((object_ref,quantity.source_scope,quantity.source_name,part),
        tuple(values),tuple(validity),tuple(fields),source_scope=quantity.source_scope,
        object_ref=object_ref,object_type=object_type,source_name=quantity.source_name,
        geometry_kind=kind,output_frame=quantity.output_frame,primitive_part_ref=part)


def _environment_rows(inputs, geometry, schema):
    """Encode already transformed E1 quantities, preserving missing-source arity."""
    from .features import DerivedQuantity, _geometry_kind

    sources = {(scope,f.name):f for scope,features in
               (('c_static',inputs.context.static_geometry),('o_dynamic',inputs.observation.features)) for f in features}
    if type(geometry.quantities) is not tuple or any(type(q) is not DerivedQuantity for q in geometry.quantities):
        raise TypeError('Expected immutable typed E1 environment quantities')
    quantities = _index(geometry.quantities,lambda q:(q.source_scope,q.source_name),'environment.quantities')
    if set(quantities) != set(sources):
        _error('environment.quantities','must describe exactly the approved current c/o sources')
    rows = []
    for quantity in geometry.quantities:
        source = sources[quantity.source_scope,quantity.source_name]
        frame,base = _source_parts(quantity.source_name)
        expected_kind = _geometry_kind(base) if frame is not None else 'intrinsic'
        expected_frame = 'intrinsic' if expected_kind in ('intrinsic','dimensions') else 'robot_centroid'
        if (quantity.kind != expected_kind or quantity.source_frame != frame or quantity.output_frame != expected_frame
                or quantity.entity_ref is not None or quantity.coordinate_id is not None):
            _error('environment.geometry','E1 source/frame/codec metadata mismatch')
        if type(quantity.valid) is not bool or quantity.valid and not source.valid:
            _error('environment.validity','invalid input cannot become observed geometry')
        kind,width,units,parts = _environment_layout(quantity,source,schema)
        if quantity.valid:
            _physical_values(quantity.values,width*parts,'environment.geometry')
        elif quantity.values is not None:
            _error('environment.geometry','invalid E1 quantity must keep values unavailable')
        object_ref,object_type = _environment_identity(base)
        rows.extend(_environment_row(quantity,object_ref,object_type,kind,width,units,part,schema) for part in range(parts))
    # Schema semantics are closed/validated; these are the four declared rules.
    return _deduplicate_environment(tuple(rows))


def _geometry_payload(row):
    offset = 0
    for field in row.fields:
        if field.name == 'geometry':
            payload = row.values[offset:offset+field.width]
            valid = row.validity[offset:offset+field.width]
            return payload if all(valid) else None
        offset += field.width
    _error('environment.fields','missing geometry layout')


def _same_numbers(a,b):
    return a is not None and b is not None and len(a)==len(b) and all(math.isclose(x,y,rel_tol=1e-9,abs_tol=1e-9) for x,y in zip(a,b))


def _dot(a,b):
    return math.fsum(x*y for x,y in zip(a,b))


def _parallel(a,b):
    norm_a = math.hypot(*a); norm_b = math.hypot(*b)
    if norm_a<=0 or norm_b<=0 or not math.isfinite(norm_a) or not math.isfinite(norm_b):
        return False
    cosine = abs(_dot(tuple(x/norm_a for x in a),tuple(x/norm_b for x in b)))
    return math.isclose(cosine,1.,rel_tol=1e-9,abs_tol=1e-9)


def _line_contains_point(line,point):
    direction = line[3:]
    length2 = _dot(direction,direction)
    if length2<=0:
        return False
    delta = tuple(p-o for p,o in zip(point,line[:3]))
    parameter = _dot(delta,direction)/length2
    return _same_numbers(point,tuple(o+parameter*d for o,d in zip(line[:3],direction)))


def _cone_center_certificate(line,point,rows):
    """Scoped physical incidence plus floor/height fixes center-height semantics.

    Incidence alone proves only compatible XY, never equivalent 3D geometry.
    No coordinates are filled; a certified existing 3D center replaces the
    partial legacy line only with a unique physical match in both directions.
    Dimensions belong to the matched physical point, not a legacy ordinal.
    """
    l,p = _geometry_payload(line),_geometry_payload(point)
    if l is None or p is None or not _line_contains_point(l,p):
        return False
    heights = [r for r in rows if r.source_scope==line.source_scope and
        r.object_ref in (point.object_ref,'cone.shared') and
        _source_parts(r.source_name)[1].endswith(('height_m','cone_height_m'))]
    # Prefer current per-object height; shared height is only a fallback.
    local = [r for r in heights if r.object_ref==point.object_ref and _geometry_payload(r) is not None]
    heights = local or [r for r in heights if r.object_ref=='cone.shared']
    floor_planes = [r for r in rows if r.source_scope==line.source_scope and
        _source_parts(r.source_name)[1]=='environment.floor_height_m' and r.geometry_kind=='plane']
    for height in heights:
        h = _geometry_payload(height)
        if h is None or len(h)!=1 or h[0]<=0:
            continue
        for floor in floor_planes:
            plane = _geometry_payload(floor)
            if plane is None or floor.output_frame!=point.output_frame or not _parallel(l[3:],plane[:3]):
                continue
            n = math.sqrt(_dot(plane[:3],plane[:3]))
            if n>0 and math.isclose((_dot(plane[:3],p)-plane[3])/n,h[0]/2,rel_tol=1e-9,abs_tol=1e-9):
                return True
    return False


def _deduplicate_environment(rows):
    """Semantic equivalence only, no generic equal-vector deletion/ID ordering.

    Conservative missing-frame behavior: invalid alternates remain invalid.
    A point may supersede its boundary plane only together with the already
    admitted boundary-normal direction; planes never become fabricated points.
    """
    discard = set()
    points = [r for r in rows if r.object_type=='cone' and r.geometry_kind=='point3']
    lines = [r for r in rows if r.object_type=='cone' and r.geometry_kind=='line']
    matches, owners = {}, {}
    for line in lines:
        candidates = {p.object_ref for p in points if p.source_scope==line.source_scope and
                      p.output_frame==line.output_frame and _cone_center_certificate(line,p,rows)}
        matches[line.ref] = candidates
        for ref in candidates:
            owners.setdefault((line.source_scope,ref),set()).add(line.object_ref)
    for line in lines:
        candidates = matches[line.ref]
        if len(candidates)==1 and len(owners[line.source_scope,next(iter(candidates))])==1:
            discard.add(line.ref)
    groups = {}
    for row in rows:
        groups.setdefault((row.source_scope,row.object_ref),[]).append(row)
    for (scope,ref), group in groups.items():
        if ref in ('gap.near_edge','gap.far_edge','stairs.first_riser'):
            points = [r for r in group if r.geometry_kind=='point3' and _geometry_payload(r) is not None]
            preferred = [r for r in points if '_center_world_xyz_m' in _source_parts(r.source_name)[1]]
            for point in points:
                if point not in preferred and any(p.output_frame==point.output_frame and _same_numbers(_geometry_payload(p),_geometry_payload(point)) for p in preferred):
                    discard.add(point.ref)
            direction_name = 'gap.crossing_direction_world_xy' if ref.startswith('gap.') else 'stairs.ascent_direction_world_xy'
            directions = [r for r in rows if r.source_scope==scope and _source_parts(r.source_name)[1]==direction_name and _geometry_payload(r) is not None]
            for plane in (r for r in group if r.geometry_kind=='plane'):
                v = _geometry_payload(plane)
                if v is None:
                    continue
                if any(p.output_frame==plane.output_frame and math.isclose(_dot(v[:3],_geometry_payload(p)),v[3],rel_tol=1e-9,abs_tol=1e-9)
                       and any(d.output_frame==plane.output_frame and _parallel(v[:3],_geometry_payload(d)) for d in directions) for p in points):
                    discard.add(plane.ref)
        if ref=='stairs.top_heights':
            preferred = [r for r in group if _source_parts(r.source_name)[1]=='stairs.top_heights_world_m']
            for row in group:
                if _source_parts(row.source_name)[1]=='stairs.top_heights_m' and any(p.primitive_part_ref==row.primitive_part_ref and p.output_frame==row.output_frame and _same_numbers(_geometry_payload(p),_geometry_payload(row)) for p in preferred):
                    discard.add(row.ref)
    # Remove shared dimensions only if every routed cone has its valid equivalent.
    for shared in (r for r in rows if r.object_ref=='cone.shared'):
        _,base = _source_parts(shared.source_name)
        dimension = 'radius_m' if base=='cones.cone_radius_m' else 'height_m' if base=='cones.cone_height_m' else None
        if dimension is None or _geometry_payload(shared) is None:
            continue
        objects = {r.object_ref for r in rows if r.source_scope==shared.source_scope and r.object_type=='cone' and r.object_ref!='cone.shared' and r.ref not in discard}
        if objects and all(any(r.object_ref==ref and r.source_scope==shared.source_scope and _source_parts(r.source_name)[1].endswith('.'+dimension)
                              and _same_numbers(_geometry_payload(r),_geometry_payload(shared)) for r in rows) for ref in objects):
            discard.add(shared.ref)
    return tuple(row for row in rows if row.ref not in discard)


_RELATION_EPS_M = 1e-12


@dataclass(frozen=True)
class _GeometryTarget:
    """Internal physical primitive, not another public feature contract.

    row identifies its scoped physical object. sources lists all physical inputs
    (including a boundary normal), never learned IDs. Multiple plane parts of
    one source group are closest components of the same selection object.
    """

    row: FeatureRow
    kind: str
    geometry: tuple[float, ...]
    relation_kind: str
    sources: tuple[EnvironmentRef, ...]


def _field_payload(row, name):
    offset = 0
    for field in row.fields:
        if field.name == name:
            valid = row.validity[offset:offset+field.width]
            return row.values[offset:offset+field.width] if all(valid) else None
        offset += field.width
    _error('relation.fields', 'missing declared physical field')


def _unit_vector(vector):
    scale = max(abs(v) for v in vector)
    if scale<=0 or not math.isfinite(scale):
        return None
    scaled = tuple(v/scale for v in vector)
    length = math.hypot(*scaled)
    return tuple(v/length for v in scaled)


def _unit_plane(geometry):
    normal = geometry[:3]
    unit = _unit_vector(normal)
    if unit is None:
        return None
    scale = max(abs(v) for v in normal)
    offset = (geometry[3]/scale)/math.hypot(*(v/scale for v in normal))
    return unit,offset


def _equivalent_relation_geometry(first, second):
    """Same physical primitive, allowing roundoff and parameterization aliases.

    Used ONLY within the same current object/scope/type group. Plane normal
    sign and a line's parameter origin do not identify different surfaces.
    """
    if first.kind!=second.kind:
        return False
    a,b = first.geometry,second.geometry
    if first.kind=='point3':
        return math.hypot(*(x-y for x,y in zip(a,b)))<=_RELATION_EPS_M
    if first.kind=='plane':
        pa,pb = _unit_plane(a),_unit_plane(b)
        if pa is None or pb is None:
            return False
        na,da = pa
        nb,db = pb
        sign = 1 if _dot(na,nb)>=0 else -1
        return (math.hypot(*(x-sign*y for x,y in zip(na,nb)))<=1e-12 and
                abs(da-sign*db)<=_RELATION_EPS_M)
    ua,ub = _unit_vector(a[3:]),_unit_vector(b[3:])
    if ua is None or ub is None:
        return False
    sign = 1 if _dot(ua,ub)>=0 else -1
    if math.hypot(*(x-sign*y for x,y in zip(ua,ub)))>1e-12:
        return False
    displacement = tuple(x-y for x,y in zip(a[:3],b[:3]))
    along = _dot(displacement,ua)
    return math.hypot(*(v-along*u for v,u in zip(displacement,ua)))<=_RELATION_EPS_M


def _relation_targets(environment, allowed_types):
    """Bind current task geometry, then group actual objects, not feature rows.

    Task-ontology boundary roles use already-valid anchor+normal; robot hardware
    plays no role. Intrinsic/attitude fields are not independent spatial targets.
    A collision center is a point target, NOT an inferred fully oriented box.
    """
    boundary_normals = {'gap.near_edge':'gap.crossing_direction_world_xy',
        'gap.far_edge':'gap.crossing_direction_world_xy',
        'stairs.first_riser':'stairs.ascent_direction_world_xy'}
    objects = {}
    for row in environment:
        if row.object_type not in allowed_types or row.output_frame!='robot_centroid':
            continue
        payload = _geometry_payload(row)
        if payload is None or row.geometry_kind not in ('point3','plane','line'):
            continue
        targets = []
        if row.geometry_kind=='point3' and row.object_ref in boundary_normals:
            for normal in environment:
                if (normal.source_scope!=row.source_scope or normal.output_frame!=row.output_frame or
                        normal.geometry_kind!='vector3' or _source_parts(normal.source_name)[1]!=boundary_normals[row.object_ref]):
                    continue
                vector = _geometry_payload(normal)
                unit = None if vector is None else _unit_vector(vector)
                if unit is not None:
                    plane = (*unit,_dot(unit,payload))
                    targets.append(_GeometryTarget(row,'plane',plane,'anchor_normal_plane',(row.ref,normal.ref)))
            # An unpaired boundary anchor is not silently made a plane target.
        else:
            targets.append(_GeometryTarget(row,row.geometry_kind,payload,row.geometry_kind,(row.ref,)))
        objects.setdefault((row.object_type,row.source_scope,row.object_ref),[]).extend(targets)
    # Equivalent spatial aliases of the same object form one primitive; their
    # complete provenance remains in sources. No representative ID is selected.
    grouped = {}
    for key, targets in objects.items():
        primitives = []
        # Physical numeric ordering only chooses a stable representation of
        # equivalent aliases; it never ranks objects or breaks distance ties.
        for target in sorted(targets,key=lambda t:(t.kind,t.geometry)):
            match = next((i for i,p in enumerate(primitives)
                          if _equivalent_relation_geometry(p,target)),None)
            if match is not None:
                previous = primitives[match]
                sources = tuple(dict.fromkeys((*previous.sources,*target.sources)))
                primitives[match] = replace(previous,sources=sources)
            else:
                primitives.append(target)
        if primitives:
            grouped[key] = tuple(primitives)
    return grouped


def _closest_displacement(position, target):
    """Closest surface minus module, in the EXISTING common physical axes."""
    geometry = target.geometry
    if target.kind=='point3':
        delta = tuple(p-m for p,m in zip(geometry,position))
    elif target.kind=='plane':
        plane = _unit_plane(geometry)
        if plane is None:
            return None
        unit,offset = plane
        # n.p=d, closest_point - module = (d/||n|| - n_hat.module)*n_hat.
        amount = offset-_dot(unit,position)
        delta = tuple(amount*v for v in unit)
    else:  # E1 line=(parameter origin, direction), unknown height preserved.
        unit = _unit_vector(geometry[3:])
        if unit is None:
            return None
        to_origin = tuple(p-m for p,m in zip(geometry[:3],position))
        along = _dot(unit,to_origin)
        delta = tuple(v-along*d for v,d in zip(to_origin,unit))
    delta = _physical_values(delta,3,'relation.delta_xyz')
    distance = math.hypot(*delta)
    if not math.isfinite(distance):
        _error('relation.distance','valid closest metric must be finite')
    return delta,distance


def _distance_tie(a,b):
    # Absolute metre-scale tolerance handles roundoff under common rigid axes.
    return abs(a-b)<=_RELATION_EPS_M


def _relation_row(module, target, delta, distance, schema):
    direction = tuple(v/distance for v in delta) if distance>_RELATION_EPS_M else None
    sources = {'relation.delta_xyz':delta,'relation.distance':(distance,),
               'relation.normalized_direction':direction}
    row = _robot_row((module.ref,target.sources),'module_environment_relation',sources,schema,{})
    # Single-source routing fields remain exact. Composite geometry/provenance
    # is explicit in geometry_sources, including anchor/normal and aliases.
    primary = target.row
    has_primary = len(target.sources)==1 or (
        target.relation_kind=='anchor_normal_plane' and len(target.sources)==2)
    parts = {source[3] for source in target.sources if source[0]==primary.object_ref}
    return replace(row,source_scope=primary.source_scope,object_ref=primary.object_ref,
        object_type=primary.object_type,source_name=primary.source_name if has_primary else None,
        geometry_kind=target.kind,output_frame='robot_centroid',
        primitive_part_ref=next(iter(parts)) if len(parts)==1 else None,module_ref=module.ref,
        relation_kind=target.relation_kind,geometry_sources=target.sources)


def _module_environment_relations(modules, environment, task, schema):
    """Nearest valid physical OBJECTS per module/type, soft schema budget.

    No N×M feature matrix or unconditional Cartesian relation allocation.
    Scan current object metrics per module; materialize only selected pairs.
    Compound source groups use their closest component(s), retaining ties.
    """
    policy = schema['relation_policy']
    allowed = policy['object_types_by_task'].get(task)
    if allowed is None:
        return ()
    objects = _relation_targets(environment,allowed)
    budget = policy['selection']['max_neighbors_per_type']
    result = []
    for module in modules:
        position = _field_payload(module,'relative_position')
        if position is None:
            continue
        by_type = {}
        for (object_type,_,_),targets in objects.items():
            closest = []
            for target in targets:
                metric = _closest_displacement(position,target)
                if metric is not None:
                    closest.append((target,*metric))
            if not closest:
                continue
            nearest = min(item[2] for item in closest)
            components = tuple(item for item in closest if _distance_tie(item[2],nearest))
            by_type.setdefault(object_type,[]).append((nearest,components))
        for candidates in by_type.values():
            threshold = sorted(distance for distance,_ in candidates)[min(budget,len(candidates))-1]
            for distance,components in candidates:
                if distance<threshold or _distance_tie(distance,threshold):
                    result.extend(_relation_row(module,target,delta,metric,schema)
                        for target,delta,metric in components)
    return tuple(result)
