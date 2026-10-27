"""E2 Tasks 1–2: schema contract and current robot rows; no candidates."""

import copy
import importlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / 'configs/bc_v1/features.json'
FAMILIES = {'module', 'connector', 'resource', 'actuator_coordinate', 'connection',
            'environment', 'module_environment_relation', 'structural_candidate_metadata'}


def api():
    name = 'drl.IL.bc_v1.feature_schema'
    assert importlib.util.find_spec(name) is not None, 'Missing E2 executable feature schema'
    return importlib.import_module(name)


def schema():
    module = api()
    assert CONFIG.is_file(), 'Missing E2 feature configuration'
    return module.load_feature_schema(CONFIG)


def test_features_schema_has_exact_version_v1():
    value = schema()
    assert value['schema_version'] == 'mssr.bc_v1.features.v1'
    assert value['input_contract'] == 'mssr.bc_v1.policy_inputs.v1'
    assert value['geometry_contract'] == 'FrameGeometryView'
    api().validate_feature_schema(value)


def test_features_schema_declares_frame_policy():
    rule = schema()['frame_policy']
    assert rule['owner'] == 'E1'
    assert rule['origin'] == 'centroid_all_current_valid_modules'
    assert rule['preference'] == ['certified_current_physical_task_frame', 'certified_map_or_world_frame']
    assert rule['same_for_training_and_rollout'] is True
    assert rule['missing_certification'] == 'invalid_no_inferred_frame'


def test_features_schema_declares_entity_field_order_units_codecs_and_validity():
    value = schema()
    assert set(value['entities']) == FAMILIES
    for name, entity in value['entities'].items():
        fields = entity['fields']
        assert isinstance(fields, list)
        assert len({f['name'] for f in fields}) == len(fields)
        for field in fields:
            assert field['units'] and field['codec'] and field['dimension']
            assert field['validity'] == 'per_component'
            assert field['source'] and field['frame']
        if name != 'structural_candidate_metadata':
            assert fields
    assert value['missingness']['value_fill'] == 'zero_only_with_invalid_channel'
    assert value['missingness']['nonfinite_valid'] == 'error'


def test_features_schema_declares_r6_convention():
    assert schema()['r6_convention'] == {'owner': 'E1.matrix_to_r6', 'columns': [0, 1], 'flatten': 'column_order'}


def test_features_schema_declares_dedup_relation_and_candidate_rules():
    value = schema()
    dedup = value['dedup_policy']
    assert set(dedup['keep_distinct']) == {'button_nominal_current', 'goal_point_support_volume', 'module_state_relative_edge'}
    assert set(dedup['canonicalize_equivalent']) == {'cone_centers', 'cone_dimensions', 'gap_edges', 'stairs'}
    assert dedup['canonicalize_equivalent']['cone_centers']['requires_physical_equivalence'] is True
    assert dedup['canonicalize_equivalent']['cone_centers']['prefer'] == 'certified_3d_over_2d'
    # XY equality does not prove a 3D point equals another with different z.
    assert dedup['canonicalize_equivalent']['cone_centers']['partial_geometry'] == 'never_invent_missing_components'
    assert value['relation_policy']['selection']['rule'] == 'nearest_valid_per_module_per_object_type'
    assert value['relation_policy']['selection']['max_neighbors_per_type'] == 2
    assert set(value['candidate_policy']['actions']) == {'align_faces', 'dock', 'undock', 'gravity_settle'}


def reordered(value):
    if isinstance(value, dict):
        return {k: reordered(value[k]) for k in reversed(value)}
    if isinstance(value, list):
        return [reordered(v) for v in value]
    return value


def test_feature_schema_fingerprint_is_deterministic(tmp_path):
    value = schema()
    a = tmp_path/'a.json'; b = tmp_path/'nested'/'b.json'; b.parent.mkdir()
    a.write_text(json.dumps(value, indent=4))
    b.write_text(json.dumps(reordered(value), separators=(',', ':')))
    fp = api().feature_schema_fingerprint(value)
    assert len(fp) == 64 and all(c in '0123456789abcdef' for c in fp)
    assert api().feature_schema_fingerprint(api().load_feature_schema(str(a))) == fp
    assert api().feature_schema_fingerprint(api().load_feature_schema(b)) == fp
    assert value == schema(), 'Fingerprint mutated source schema'


def test_schema_fingerprint_canonicalizes_set_lists_but_preserves_column_order():
    value = schema(); changed = copy.deepcopy(value)
    changed['candidate_policy']['allowed_inputs'].reverse()
    changed['entities']['environment']['source_scopes'].reverse()
    changed['entities']['connector']['routing_fields'].reverse()
    assert api().feature_schema_fingerprint(value) == api().feature_schema_fingerprint(changed)
    changed = copy.deepcopy(value)
    changed['entities']['connector']['fields'][3]['categories'].reverse()
    assert api().feature_schema_fingerprint(value) != api().feature_schema_fingerprint(changed)


def test_module_base_width_is_derived_as_30():
    value = schema(); fields = value['entities']['module']['fields']
    assert [f['name'] for f in fields] == ['relative_position', 'orientation_r6', 'linear_velocity', 'angular_velocity']
    assert [f['dimension'] for f in fields] == [3, 6, 3, 3]
    values = sum(f['dimension'] for f in fields)
    assert values == 15
    assert api().module_base_width(value) == values + values == 30
    assert value['entities']['module']['packing'] == 'all_values_then_matching_validity'


def test_instance_ids_are_not_declared_as_learned_features():
    value = schema()
    for entry in value['routing_metadata'].values():
        assert entry['learned'] is False and entry['encoding'] == 'opaque_routing'
    for entity in value['entities'].values():
        assert all(f['name'] not in value['routing_metadata'] for f in entity['fields'])
    assert value['entities']['structural_candidate_metadata']['fields'] == []


def test_partial_geometry_kinds_remain_distinct():
    kinds = schema()['geometry_kinds']
    assert set(kinds) == {'point3', 'vector3', 'line', 'plane', 'heading', 'planar_pose',
                          'angle_constraint', 'intrinsic_scalar', 'intrinsic_vector'}
    assert kinds['point3']['e1_kinds'] == ['point'] and kinds['point3']['dimension'] == 3
    assert kinds['line']['e1_kinds'] == ['line'] and kinds['line']['dimension'] == 6
    assert kinds['plane']['e1_kinds'] == ['planes'] and kinds['plane']['dimension'] == 4
    assert kinds['plane']['row_policy'] == 'one_plane_per_row_preserve_source_group'
    assert kinds['angle_constraint']['dimension'] == 10


def test_static_dynamic_provenance_is_not_numeric_feature():
    value = schema()
    assert value['entities']['environment']['source_scopes'] == ['c_static', 'o_dynamic']
    assert value['routing_metadata']['source_scope']['learned'] is False
    assert value['routing_metadata']['source_name']['learned'] is False
    assert value['dedup_policy']['keep_distinct']['button_nominal_current'] == 'preserve_nominal_and_current_roles'


def test_schema_declares_missing_geometry_shape_without_reading_invalid_values():
    policy = schema().get('source_shape_policy')
    assert policy == {
        'owner': 'validated_policy_inputs',
        'scope_sources': {'c_static': 'context.static_geometry', 'o_dynamic': 'observation.features'},
        'lookup': 'source_scope_and_source_name',
        'read': 'component_count_only',
        'learned': False,
        'intrinsic_dispatch': 'source_arity_even_when_values_missing',
        'intrinsic_vector_width': 'source_arity',
        'plane_row_count': 'source_arity',
        'unknown_shape': 'error_no_zero_or_scalar_guess',
    }


@pytest.mark.parametrize('task,name,values,kind,arity', [
    ('button', 'button.plunger_depth_m', (.1,), 'intrinsic', 1),
    ('button', 'button.face_size_tangent_z_m', (.1,.2), 'intrinsic', 2),
    ('button', 'world.obstacles.0.size_xyz_m', (.1,.2,.3), 'dimensions', 3),
    ('stairs', 'world.stairs.top_heights_m', (.1,.2,.3,.4), 'planes', 4),
])
def test_schema_missing_shape_policy_matches_frozen_d2_e1(task,name,values,kind,arity):
    # A future extractor can resolve the layout from the approved input shape;
    # this test does not implement extraction, filling, or row creation.
    assert schema().get('source_shape_policy',{}).get('owner') == 'validated_policy_inputs'
    from drl.IL.bc_v1 import bc_v1_contract as c
    from drl.IL.bc_v1.policy_inputs import select_policy_inputs
    from drl.IL.bc_v1.features import transform_inputs, PhysicalFrameConfig
    from drl.IL.tests.bc_v1 import fixtures as f
    source = select_policy_inputs(f.state_graph(2,0),c.ObservationBundle(),
                                 c.TaskContext(task,(c.NumericFeature(name,values,valid=False),)))
    derived = transform_inputs(source,PhysicalFrameConfig()).quantities[0]
    assert derived.kind == kind and not derived.valid and derived.values is None
    assert derived.source_scope == 'c_static' and derived.source_name == name
    assert len(source.context.static_geometry[0].values) == arity


@pytest.mark.parametrize('key,bad', [('read','numeric_values'),('learned',True),
                                  ('plane_row_count','one'),('unknown_shape','guess_scalar')])
def test_schema_rejects_unsafe_missing_shape_policy(key,bad):
    value = schema()
    assert 'source_shape_policy' in value
    value['source_shape_policy'][key] = bad
    with pytest.raises(ValueError): api().validate_feature_schema(value)


def test_schema_has_no_smores_specific_pan_tilt_or_fixed_module_count():
    value = schema()
    assert value['cardinality'] == {'modules': 'variable', 'connectors': 'variable', 'resources': 'variable', 'coordinates': 'variable'}
    sources = [f['source'] for e in value['entities'].values() for f in e['fields']]
    assert 'resource.capabilities' in sources
    assert 'coordinate.position' in sources and 'coordinate.velocity' in sources
    assert all(f['dimension'] == 'robot_spec.resource_capabilities' for f in value['entities']['resource']['fields'])


def test_align_faces_rule_does_not_use_distance_only_pruning():
    rule = schema()['candidate_policy']['actions']['align_faces']
    assert rule['distance_only_pruning'] is False
    assert rule['endpoints'] == 'compatible_connector_pair'
    assert rule['kind'] == 'prepare_connection'


def test_gravity_settle_rule_requires_no_connector_endpoint():
    rule = schema()['candidate_policy']['actions']['gravity_settle']
    assert rule['endpoints'] == 'none'
    assert rule['scope'] == 'from_robot_spec_module_scope'
    assert rule['enabled_by_default'] is True


@pytest.mark.parametrize('name', ['stage', 'target_morphology', 'future_state', 'action_t', 'supervision', 'executor', 'nominal_route', 'root_module'])
def test_schema_rejects_forbidden_learned_source(name):
    value = schema()
    value['entities']['module']['fields'][0]['source'] = name
    with pytest.raises(ValueError, match='source'): api().validate_feature_schema(value)


@pytest.mark.parametrize('path,replacement', [
    (('schema_version',), 'mssr.bc_v1.features.v2'),
    (('input_contract',), 'open_inputs'),
    (('geometry_contract',), 'PolicyInputs'),
    (('frame_policy', 'preference'), ['first_module']),
    (('frame_policy', 'origin'), 'min_id'),
    (('frame_policy', 'same_for_training_and_rollout'), False),
    (('r6_convention', 'flatten'), 'row_order'),
    (('geometry_kinds', 'line', 'dimension'), 3),
    (('geometry_kinds', 'plane', 'e1_kinds'), ['point']),
    (('routing_metadata', 'source_scope', 'learned'), True),
    (('candidate_policy', 'actions', 'align_faces', 'distance_only_pruning'), True),
    (('candidate_policy', 'actions', 'gravity_settle', 'endpoints'), 'compatible_connector_pair'),
    (('candidate_policy', 'actions', 'gravity_settle', 'enabled_by_default'), False),
    (('relation_policy', 'selection', 'rule'), 'cartesian_all_pairs'),
    (('relation_policy', 'selection', 'max_neighbors_per_type'), 0),
    (('relation_policy', 'allowed_inputs'), ['expert_label']),
    (('candidate_policy', 'allowed_inputs'), ['future_graph']),
    (('entities', 'module', 'fields', 0, 'dimension'), True),
    (('entities', 'module', 'fields', 0, 'dimension'), 0),
    (('entities', 'module', 'fields', 0, 'units'), 'rad'),
    (('entities', 'module', 'fields', 0, 'validity'), 'implicit_zero'),
    (('entities', 'connector', 'fields', 0, 'frame'), 'robot_centroid'),
    (('entities', 'connection', 'fields', 0, 'frame'), 'robot_centroid'),
    (('entities', 'resource', 'fields', 0, 'dimension'), 2),
    (('entities', 'environment', 'source_scopes'), ['provenance_numeric_id']),
])
def test_schema_rejects_invalid_or_noncausal_contracts(path, replacement):
    value = schema(); parent = value
    for key in path[:-1]: parent = parent[key]
    parent[path[-1]] = replacement
    with pytest.raises(ValueError): api().validate_feature_schema(value)


@pytest.mark.parametrize('change', ['unknown_key', 'unknown_field_key', 'missing_field', 'duplicate_field',
                                   'learned_module_id', 'executor_phase', 'dedup_false', 'empty_family'])
def test_schema_rejects_unknown_incomplete_or_unsafe_extensions(change):
    value = schema()
    if change == 'unknown_key': value['module_count'] = 8
    elif change == 'unknown_field_key': value['entities']['module']['fields'][0]['episode_id'] = 'hint'
    elif change == 'missing_field': del value['entities']['module']['fields'][0]['units']
    elif change == 'duplicate_field': value['entities']['connector']['fields'].append(copy.deepcopy(value['entities']['connector']['fields'][0]))
    elif change == 'learned_module_id': value['entities']['module']['fields'][0]['name'] = 'module_ref'
    elif change == 'executor_phase': value['candidate_policy']['actions']['reach'] = copy.deepcopy(value['candidate_policy']['actions']['align_faces'])
    elif change == 'dedup_false': value['dedup_policy']['canonicalize_equivalent']['cone_centers']['requires_physical_equivalence'] = False
    else: value['entities']['connector']['fields'] = []
    with pytest.raises(ValueError): api().validate_feature_schema(value)
    with pytest.raises(ValueError): api().feature_schema_fingerprint(value)


@pytest.mark.parametrize('text', ['{"schema_version":1,"schema_version":2}', '{"threshold":NaN}',
                                '{"threshold":Infinity}', '[]'])
def test_schema_loader_rejects_ambiguous_or_nonfinite_json(tmp_path, text):
    path = tmp_path/'bad.json'; path.write_text(text)
    with pytest.raises(ValueError): api().load_feature_schema(path)


def test_feature_schema_import_isolated_from_hardware_tensors_and_readers():
    api()
    script = '''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'rclpy','isaacsim','isaaclab','omni','torch','numpy','smores_ep','mssr_expert','freebot_docking'} or fullname in {'drl.IL.bc_v1.smores_adapter','drl.IL.bc_v1.task_context','drl.IL.preprocessing','drl.IL.bc_v1.features'}:
   raise ImportError('schema isolation: '+fullname)
sys.meta_path.insert(0,Block())
from drl.IL.bc_v1.feature_schema import load_feature_schema,module_base_width
assert module_base_width(load_feature_schema('configs/bc_v1/features.json'))==30
'''
    result = subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr


# E2 Task 2. Expectations are physical literals, independently hand-derived.
from dataclasses import FrozenInstanceError, replace
from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.bc_v1 import features as e1
from drl.IL.tests.bc_v1 import fixtures as f


def extract(graph=None, spec=None, config=None, inputs=None, geometry=None, config_schema=None):
    assert hasattr(api(), 'extract_features'), 'Missing Task-2 structured extraction'
    if inputs is None:
        inputs = c.PolicyInputs(f.state_graph(2, 1) if graph is None else graph,
                                c.ObservationBundle(), c.TaskContext())
    if geometry is None:
        geometry = e1.transform_inputs(inputs, config or e1.PhysicalFrameConfig())
    return api().extract_features(inputs, geometry, spec or f.robot_spec(), config_schema or schema())


def physical_graph():
    graph = f.state_graph(2, 1)
    modules = tuple(replace(m, position=(10.+2*i, 4., 3.),
                            linear_velocity=(1., 2., 3.), angular_velocity=(4., 5., 6.))
                    for i, m in enumerate(graph.modules))
    return replace(graph, modules=modules)


def axes():
    return e1.PhysicalFrameConfig(axes_id='stage_local',
                                 rotation_source_from_axes=(0.,-1.,0.,1.,0.,0.,0.,0.,1.))


def renamed_modules(graph):
    refs = {m.ref: c.ModuleRef('opaque-'+str(99-i)) for i,m in enumerate(graph.modules)}
    def connector(ref):
        return c.ConnectorRef(refs[ref.module], ref.connector_id)
    return c.StateGraph(
        tuple(replace(m,ref=refs[m.ref]) for m in graph.modules),
        tuple(replace(x,ref=connector(x.ref),connected_to=None if x.connected_to is None else connector(x.connected_to)) for x in graph.connectors),
        tuple(replace(x,ref=c.ResourceRef(refs[x.ref.module],x.ref.resource_id)) for x in graph.resources),
        tuple(replace(x,endpoint_a=connector(x.endpoint_a),endpoint_b=connector(x.endpoint_b)) for x in graph.connections))


def local_graph():
    graph = physical_graph()
    connectors = tuple(replace(x,status='available',features=(
        c.NumericFeature('position_body_m',(2.,3.,4.)),
        c.NumericFeature('normal_body',(1.,0.,0.)),
        c.NumericFeature('tangent_body',(0.,1.,0.)))) for x in graph.connectors)
    edge = c.ConnectionState(connectors[0].ref,connectors[1].ref,'current_connection',('x','yaw'),(
        c.NumericFeature('relative_position_m',(7.,8.,9.)),
        c.NumericFeature('relative_orientation_xyzw',(0.,0.,0.,1.)),
        c.NumericFeature('clocking_quarter_turns',(2.,))))
    return replace(graph,connectors=connectors,connections=(edge,))


def test_module_base_width_is_schema_derived_30():
    view = extract()
    assert len(view.modules[0].numeric) == api().module_base_width(schema()) == 30
    assert tuple(field.width for field in view.modules[0].fields) == (3,6,3,3)


def test_module_row_contains_relative_position_r6_twists_and_validity():
    view = extract(physical_graph(),config=axes())
    assert view.modules[0].values == (0.,1.,0., 0.,-1.,0.,1.,0.,0., 2.,-1.,3.,5.,-4.,6.)
    assert view.modules[0].validity == (True,)*15
    assert view.modules[0].numeric == view.modules[0].values+(1.,)*15


def test_module_reorder_changes_only_row_order():
    graph = physical_graph()
    a = extract(graph); b = extract(replace(graph,modules=tuple(reversed(graph.modules))))
    assert tuple(r.numeric for r in a.modules) == tuple(r.numeric for r in reversed(b.modules))
    assert tuple(r.ref for r in b.modules) == tuple(m.ref for m in reversed(graph.modules))


def test_module_id_rename_does_not_change_numeric_features():
    a = extract(physical_graph()); b = extract(renamed_modules(physical_graph()))
    assert tuple(r.numeric for r in a.modules) == tuple(r.numeric for r in b.modules)
    assert a.modules[0].ref != b.modules[0].ref


def test_q_and_minus_q_have_same_r6():
    graph = physical_graph(); q = (0.,0.,2**-.5,2**-.5)
    a = extract(replace(graph,modules=tuple(replace(m,orientation_xyzw=q) for m in graph.modules)))
    b = extract(replace(graph,modules=tuple(replace(m,orientation_xyzw=tuple(-v for v in q)) for m in graph.modules)))
    assert a.modules[0].values[3:9] == b.modules[0].values[3:9]
    assert a.modules[0].values[3:9] == pytest.approx((0.,1.,0.,-1.,0.,0.))


def test_missing_linear_velocity_remains_invalid_not_observed_zero():
    graph = physical_graph()
    graph = replace(graph,modules=(replace(graph.modules[0],linear_velocity=None),replace(graph.modules[1],linear_velocity=(0.,0.,0.))))
    rows = extract(graph).modules
    assert rows[0].values[9:12] == rows[1].values[9:12] == (0.,)*3
    assert rows[0].validity[9:12] == (False,)*3 and rows[1].validity[9:12] == (True,)*3


def test_missing_angular_velocity_remains_invalid_not_observed_zero():
    graph = physical_graph()
    graph = replace(graph,modules=(replace(graph.modules[0],angular_velocity=None),replace(graph.modules[1],angular_velocity=(0.,0.,0.))))
    rows = extract(graph).modules
    assert rows[0].values[12:15] == rows[1].values[12:15] == (0.,)*3
    assert rows[0].validity[12:15] == (False,)*3 and rows[1].validity[12:15] == (True,)*3


def test_module_numeric_vector_contains_no_instance_id_encoding():
    graph = physical_graph()
    a = extract(graph); b = extract(renamed_modules(graph))
    assert a.modules[0].numeric == b.modules[0].numeric
    assert a.modules[0].catalog_ref == 'unit'
    spec = replace(f.robot_spec(),module_types=(replace(f.robot_spec().module_types[0],type_id='arbitrary-type'),))
    rows = extract(replace(graph,modules=tuple(replace(m,module_type_id='arbitrary-type') for m in graph.modules)),spec).modules
    assert rows[0].numeric == a.modules[0].numeric
    assert rows[0].catalog_ref == 'arbitrary-type'


def test_connector_body_local_geometry_is_not_transformed_twice():
    row = extract(local_graph(),config=axes()).connectors[0]
    assert row.values[:9] == (2.,3.,4.,1.,0.,0.,0.,1.,0.)
    assert row.validity[:9] == (True,)*9
    assert tuple(x.frame for x in row.fields[:3]) == ('module_body',)*3


@pytest.mark.parametrize('count,surface',[(0,False),(1,True),(7,False)])
def test_connector_count_is_variable(count,surface):
    view = extract(f.state_graph(2,count,surface=surface),f.robot_spec(count,surface=surface))
    assert len(view.connectors) == 2*(count+surface)
    if surface:
        assert view.connectors[-1].values[9:11] == (0.,1.)


def test_connector_instance_id_is_routing_only():
    graph = local_graph(); spec = f.robot_spec()
    ports = lambda ref:c.ConnectorRef(ref.module,'arbitrary-port')
    changed = replace(graph,connectors=tuple(replace(x,ref=ports(x.ref)) for x in graph.connectors),
                      connections=tuple(replace(x,endpoint_a=ports(x.endpoint_a),endpoint_b=ports(x.endpoint_b)) for x in graph.connections))
    changed_spec = replace(spec,module_types=(replace(spec.module_types[0],connectors=(replace(spec.module_types[0].connectors[0],connector_id='arbitrary-port'),)),))
    a,b = extract(graph),extract(changed,changed_spec)
    assert a.connectors[0].numeric == b.connectors[0].numeric
    assert a.connectors[0].ref != b.connectors[0].ref


def test_connection_relative_geometry_is_not_transformed_twice():
    row = extract(local_graph(),config=axes()).connections[0]
    assert row.values[:10] == (7.,8.,9.,1.,0.,0.,0.,1.,0.,2.)
    assert row.validity == (True,)*21 and len(row.numeric) == 42
    assert row.values[10:18] == (1.,0.,0.,0.,0.,1.,0.,0.)
    assert row.values[18:] == (1.,0.,0.)


def test_connection_endpoint_refs_are_routing_only():
    a,b = extract(local_graph()),extract(renamed_modules(local_graph()))
    assert a.connections[0].numeric == b.connections[0].numeric
    assert a.connections[0].ref != b.connections[0].ref


def test_resource_coordinate_width_is_schema_and_spec_driven():
    spec = f.robot_spec(); resource = replace(spec.module_types[0].resources[0],capabilities=('rotate','translate'))
    spec = replace(spec,module_types=(replace(spec.module_types[0],resources=(resource,)),))
    row = extract(spec=spec).resources[0]
    assert row.values == (1.,1.) and row.validity == (True,True) and len(row.numeric) == 4
    assert row.fields[0].channels == ('rotate','translate')
    coord = extract(spec=spec).coordinates[0]
    assert coord.values == (.1,0.,0.,0.,1.) and coord.validity == (True,True,False,False,True)
    assert len(coord.numeric) == 10
    assert tuple(field.units for field in coord.fields) == (('rad',),('rad/s',),('rad','rad'),('1',))


@pytest.mark.parametrize('count',[0,1,3])
def test_resource_count_is_variable(count):
    base = f.robot_spec(); module = base.module_types[0]
    resources = tuple(replace(module.resources[0],resource_id='motor-'+str(i)) for i in range(count))
    spec = replace(base,module_types=(replace(module,resources=resources),))
    graph = f.state_graph(2,1)
    graph = replace(graph,resources=tuple(c.ResourceState(c.ResourceRef(m.ref,r.resource_id),
                           (c.CoordinateState('rotation',0.,0.),)) for m in graph.modules for r in resources))
    assert len(extract(graph,spec).resources) == 2*count


@pytest.mark.parametrize('count',[0,1,4])
def test_coordinate_count_is_variable(count):
    base = f.robot_spec(); module = base.module_types[0]
    coords = tuple(c.CoordinateSpec('axis-'+str(i),'m',limits=(-2.,2.)) for i in range(count))
    spec = replace(base,module_types=(replace(module,resources=(replace(module.resources[0],coordinates=coords),)),))
    graph = f.state_graph(2,1)
    graph = replace(graph,resources=tuple(replace(r,coordinates=tuple(c.CoordinateState(x.coordinate_id,.2,0.) for x in coords)) for r in graph.resources))
    rows = extract(graph,spec).coordinates
    assert len(rows) == 2*count
    if rows:
        assert rows[0].values == (.2,0.,-2.,2.,0.) and rows[0].validity == (True,)*5
        assert rows[0].fields[1].units == ('m/s',)


def test_coordinate_ids_are_routing_only():
    graph = f.state_graph(2,1); spec = f.robot_spec(); mt = spec.module_types[0]; rs = mt.resources[0]
    changed = replace(graph,resources=tuple(replace(r,coordinates=(replace(r.coordinates[0],coordinate_id='something'),)) for r in graph.resources))
    spec = replace(spec,module_types=(replace(mt,resources=(replace(rs,coordinates=(replace(rs.coordinates[0],coordinate_id='something'),)),)),))
    a,b = extract(graph),extract(changed,spec)
    assert a.coordinates[0].numeric == b.coordinates[0].numeric and a.coordinates[0].ref != b.coordinates[0].ref


def test_missing_coordinate_position_velocity_preserve_validity():
    graph = f.state_graph(2,1)
    graph = replace(graph,resources=(replace(graph.resources[0],coordinates=(c.CoordinateState('rotation'),)),graph.resources[1]))
    a,b = extract(graph).coordinates
    assert a.values[:2] == (0.,0.) and a.validity[:2] == (False,False)
    assert b.values[:2] == (.1,0.) and b.validity[:2] == (True,True)


def test_no_pan_tilt_branch_exists_in_generic_e2():
    base = f.robot_spec(); mt=base.module_types[0]; rs=mt.resources[0]; graph=f.state_graph(2,1)
    a=extract(graph)
    for name in ('PAN','TILT','arbitrary-actuator'):
        spec=replace(base,module_types=(replace(mt,resources=(replace(rs,resource_id=name),)),))
        changed=replace(graph,resources=tuple(replace(r,ref=c.ResourceRef(r.ref.module,name)) for r in graph.resources))
        b=extract(changed,spec)
        assert tuple(r.numeric for r in b.resources)==tuple(r.numeric for r in a.resources)
        assert tuple(r.numeric for r in b.coordinates)==tuple(r.numeric for r in a.coordinates)


@pytest.mark.parametrize('n',[0,1,2,5,12])
def test_no_fixed_eight_module_assumption(n):
    view=extract(f.state_graph(n,1))
    assert len(view.modules)==n and len(view.resources)==n
    if n==0: assert not view.frame.valid and view.frame.origin is None


@pytest.mark.parametrize('n',[0,2,7])
def test_no_fixed_four_connector_assumption(n):
    assert len(extract(f.state_graph(3,n),f.robot_spec(n)).connectors)==3*n


def test_environment_extension_preserves_task2_robot_rows():
    inputs=c.PolicyInputs(f.state_graph(2,1),c.ObservationBundle(),
                          c.TaskContext('button',(c.NumericFeature('world.button.center_xyz_m',(8.,0.,0.)),)))
    view=extract(inputs=inputs)
    assert view.environment
    robot_only=extract(inputs=c.PolicyInputs(inputs.graph,c.ObservationBundle(),c.TaskContext()))
    assert (view.modules,view.connectors,view.resources,view.coordinates,view.connections)==(robot_only.modules,robot_only.connectors,robot_only.resources,robot_only.coordinates,robot_only.connections)


@pytest.mark.parametrize('forbidden',['labels','future_state','semantic_morphology'])
def test_extract_features_does_not_access_forbidden_information(forbidden):
    # Unknown attributes rejected at the typed boundary, never traversed/read.
    class Trap:
        def __getattribute__(self,name): raise AssertionError('forbidden payload read')
    inputs=c.PolicyInputs(f.state_graph(2,1),c.ObservationBundle(),c.TaskContext())
    geometry=e1.transform_inputs(inputs,e1.PhysicalFrameConfig())
    object.__setattr__(inputs,forbidden,Trap())
    with pytest.raises(ValueError): extract(inputs=inputs,geometry=geometry)


def test_extract_features_is_immutable_and_does_not_mutate_sources():
    graph=physical_graph(); inputs=c.PolicyInputs(graph,c.ObservationBundle(),c.TaskContext())
    geometry=e1.transform_inputs(inputs,axes()); spec=f.robot_spec(); config=schema()
    before=copy.deepcopy((inputs,geometry,spec,config))
    view=extract(inputs=inputs,geometry=geometry,spec=spec,config_schema=config)
    assert (inputs,geometry,spec,config)==before
    with pytest.raises(FrozenInstanceError): view.modules=()
    with pytest.raises(FrozenInstanceError): view.modules[0].values=()
    assert isinstance(view.modules,tuple) and isinstance(view.modules[0].values,tuple)


def test_extract_carries_exact_e1_frame_metadata_without_selecting_frame():
    inputs=c.PolicyInputs(physical_graph(),c.ObservationBundle(),c.TaskContext())
    geometry=e1.transform_inputs(inputs,axes()); view=extract(inputs=inputs,geometry=geometry)
    assert view.frame is geometry.frame
    assert (view.frame.origin,view.frame.axes_id,view.frame.graph_frame,view.frame.valid)==((11.,4.,3.),'stage_local','world',True)
    assert view.schema_fingerprint=='43ab1b93c21b2645176e614ab1af7437f37163ec2b6047c0671d18a354b67d57'


def test_extract_missing_module_pose_does_not_destroy_measured_twist():
    graph=physical_graph(); graph=replace(graph,modules=tuple(replace(m,pose_valid=False) for m in graph.modules))
    row=extract(graph).modules[0]
    assert row.values[:9]==(0.,)*9 and row.validity[:9]==(False,)*9
    assert row.values[9:]==(1.,2.,3.,4.,5.,6.) and row.validity[9:]==(True,)*6


def test_extract_unknown_connector_status_is_invalid_not_available():
    row=extract().connectors[0]
    assert row.values[11:13]==(0.,0.) and row.validity[11:13]==(False,False)


def test_extract_reordered_schema_categories_control_column_order():
    value=schema(); value['entities']['connection']['fields'][4]['categories'].reverse()
    row=extract(local_graph(),config_schema=value).connections[0]
    assert row.values[18:]==(0.,0.,1.)


@pytest.mark.parametrize('case',['catalog','geometry_refs','local_state','nonfinite','wrong_scope'])
def test_extract_rejects_inconsistent_current_contract(case):
    inputs=c.PolicyInputs(physical_graph(),c.ObservationBundle(),c.TaskContext()); geometry=e1.transform_inputs(inputs,axes()); spec=f.robot_spec()
    if case=='catalog': spec=replace(spec,module_types=(replace(spec.module_types[0],type_id='other'),))
    elif case=='geometry_refs': geometry=replace(geometry,modules=geometry.modules[:1])
    elif case=='local_state': geometry=replace(geometry,resources=())
    elif case=='nonfinite': geometry=replace(geometry,modules=(replace(geometry.modules[0],linear_velocity=(float('nan'),0.,0.)),geometry.modules[1]))
    else: geometry=replace(geometry,modules=(replace(geometry.modules[0],source_scope='c_static'),geometry.modules[1]))
    with pytest.raises(ValueError): extract(inputs=inputs,geometry=geometry,spec=spec)


def test_extract_import_isolation_from_hardware_tensors_and_readers():
    assert hasattr(api(),'extract_features'), 'Missing Task-2 extraction'
    script='''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'rclpy','isaacsim','isaaclab','omni','torch','smores_ep','mssr_expert','freebot_docking'} or fullname in {'drl.IL.bc_v1.smores_adapter','drl.IL.bc_v1.task_context','drl.IL.preprocessing'}:
   raise ImportError('extract isolation: '+fullname)
sys.meta_path.insert(0,Block())
from drl.IL.tests.bc_v1.fixtures import state_graph,robot_spec
from drl.IL.bc_v1.bc_v1_contract import PolicyInputs,ObservationBundle,TaskContext
from drl.IL.bc_v1.features import transform_inputs,PhysicalFrameConfig
from drl.IL.bc_v1.feature_schema import extract_features,load_feature_schema
p=PolicyInputs(state_graph(2,1),ObservationBundle(),TaskContext())
v=extract_features(p,transform_inputs(p,PhysicalFrameConfig()),robot_spec(),load_feature_schema('configs/bc_v1/features.json'))
assert len(v.modules)==2 and len(v.modules[0].numeric)==30
'''
    result=subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


@pytest.fixture(scope='module')
def task2_real_rows():
    # Only two real rows; no full scan, no six-case E2 regression, no artifacts in dataset.
    from drl.IL.bc_v1.smores_adapter import SMORESAdapter
    from drl.IL.bc_v1.task_context import read_task_metadata, build_task_view
    root=ROOT/'datasets/expert_v1_canonical_v6'
    if not (root/'manifest.json').is_file(): pytest.skip('Canonical v6 unavailable')
    manifest=json.loads((root/'manifest.json').read_text())
    paths={e['episode_id']:root/e['file'] for e in manifest['episodes']}
    config=json.loads((ROOT/'configs/bc_v1/task_context.json').read_text())
    spec=c.load_robot_spec(ROOT/'configs/bc_v1/smores_ep.robot.json')
    adapter=SMORESAdapter(spec)
    selected={}
    for case,episode,index in [('assembly','expert:assembly:seed-006217',0),
                               ('connected','teleop-1bea8f28c1c04323a4c3ec7e2829d07c',7228)]:
        with paths[episode].open() as stream:
            for i,line in enumerate(stream):
                if i==index:
                    raw=json.loads(line); break
        context,observation=build_task_view(raw,read_task_metadata(raw,ROOT),config)
        inputs=c.PolicyInputs(adapter.encode_state(raw['graph_t']),observation,context)
        selected[case]=(inputs,e1.transform_inputs(inputs,e1.PhysicalFrameConfig()),spec)
    return selected


@pytest.mark.parametrize('case,edge_count',[('assembly',0),('connected',7)])
def test_extract_targeted_real_current_robot_rows(task2_real_rows,case,edge_count):
    inputs,geometry,spec=task2_real_rows[case]
    view=extract(inputs=inputs,geometry=geometry,spec=spec)
    assert len(view.modules)==8 and len(view.connections)==edge_count
    assert len(view.connectors)==len(inputs.graph.connectors)==32
    assert len(view.resources)==len(inputs.graph.resources)
    assert len(view.coordinates)==sum(len(r.coordinates) for r in inputs.graph.resources)
    assert all(all(row.validity[:9]) for row in view.modules)
    assert all(len(row.numeric)==30 for row in view.modules)
    assert all(__import__('math').isfinite(v) for rows in (view.modules,view.connectors,view.resources,view.coordinates,view.connections) for row in rows for v in row.numeric)
    assert view.environment


@pytest.mark.parametrize('change',[
    {'valid':1},
    {'valid':True,'origin':None},
    {'valid':False},
    {'axes_id':'semantic-head'},
    {'graph_frame':'root'},
    {'rotation_source_from_axes':(0.,)*9},
    {'rotation_source_from_axes':[1.,0.,0.,0.,1.,0.,0.,0.,1.]},
    {'origin':[11.,4.,3.]},
    {'origin':(float('nan'),4.,3.)},
])
def test_extract_rejects_invalid_or_mutable_e1_frame_metadata(change):
    inputs=c.PolicyInputs(physical_graph(),c.ObservationBundle(),c.TaskContext())
    geometry=e1.transform_inputs(inputs,axes())
    geometry=replace(geometry,frame=replace(geometry.frame,**change))
    with pytest.raises((ValueError,TypeError)):
        extract(inputs=inputs,geometry=geometry)


# E2 Task 3 — immutable environment entities; no module/environment relations.
def environment_view(task='navigation',static=(),dynamic=(),config=None):
    inputs=c.PolicyInputs(f.state_graph(2,1),c.ObservationBundle(dynamic),c.TaskContext(task,static))
    return extract(inputs=inputs,config=config)


def quantity(name,values,valid=True):
    return c.NumericFeature(name,values,valid)


def environment_row(view,name):
    rows=[r for r in view.environment if getattr(r,'source_name',None)==name]
    assert len(rows)==1, 'Missing/duplicate typed environment source '+name
    return rows[0]


@pytest.mark.parametrize('task,name,values,kind,width,object_type',[
    ('navigation','world.goal.center_xyz_m',(7.,0.,2.),'point3',3,'goal'),
    ('gap','world.gap.crossing_direction_world_xy',(0.,1.),'vector3',3,'gap_boundary'),
    ('navigation','world.cones.0.center_xy_m',(7.,2.),'line',6,'cone'),
    ('gap','world.gap.near_edge_x_m',(7.,),'plane',4,'gap_boundary'),
    ('button','world.button.press_yaw_rad',(0.,),'heading',3,'button'),
    ('navigation','world.goal.xyyaw',(7.,2.,0.),'planar_pose',9,'goal'),
    ('navigation','world.obstacles.0.yaw_deg',(90.,),'angle_constraint',10,'collision_box'),
    ('button','button.plunger_stroke_m',(.004,),'intrinsic_scalar',1,'button'),
    ('button','button.face_size_tangent_z_m',(.08,.09),'intrinsic_vector',2,'button'),
])
def test_environment_geometry_kind_preserves_declared_codec(task,name,values,kind,width,object_type):
    row=environment_row(environment_view(task,(quantity(name,values),)),name)
    assert row.geometry_kind==kind and row.object_type==object_type
    assert row.fields[0].width==width and len(row.numeric)==2*(width+9)
    assert row.fields[0].name=='geometry' and all(row.validity)
    assert row.values[width:]==tuple(float(t==object_type) for t in schema()['entities']['environment']['fields'][1]['categories'])
    if kind=='point3': assert row.values[:3]==(6.5,0.,2.)
    if kind=='vector3': assert row.values[:3]==(0.,1.,0.)
    if kind=='line': assert row.values[:6]==(6.5,2.,0.,0.,0.,1.)
    if kind=='plane': assert row.values[:4]==(1.,0.,0.,6.5)
    if kind=='intrinsic_vector': assert row.values[:2]==values and row.output_frame=='intrinsic'
    if kind=='intrinsic_scalar': assert row.values[0]==.004 and row.fields[0].units==('m',)


@pytest.mark.parametrize('name,values,kind,parts,width',[
    ('button.plunger_depth_m',(.04,),'intrinsic_scalar',1,1),
    ('button.face_size_tangent_z_m',(.08,.09),'intrinsic_vector',1,2),
    ('unframed.obstacles.0.size_xyz_m',(1.,2.,3.),'intrinsic_vector',1,3),
    ('unframed.environment.platform_bounds_xy_m',(0.,1.,2.,3.),'plane',4,4),
])
def test_invalid_environment_quantity_keeps_validity_false_and_known_shape(name,values,kind,parts,width):
    view=environment_view('button',(quantity(name,values,False),))
    rows=[r for r in view.environment if getattr(r,'source_name',None)==name]
    assert len(rows)==parts
    assert all(r.geometry_kind==kind and r.values[:width]==(0.,)*width and r.validity[:width]==(False,)*width for r in rows)
    assert all(r.validity[width:]==(True,)*9 for r in rows)


def test_environment_plane_source_splits_parts_without_point_cast():
    name='world.environment.platform_bounds_xy_m'
    rows=environment_view('navigation',(quantity(name,(0.,2.,3.,4.)),)).environment
    assert len(rows)==4 and [r.primitive_part_ref for r in rows]==[0,1,2,3]
    assert [r.values[:4] for r in rows]==[(1.,0.,0.,-.5),(1.,0.,0.,1.5),(0.,1.,0.,3.),(0.,1.,0.,4.)]
    assert all(r.geometry_kind=='plane' for r in rows)


def test_button_nominal_and_current_centers_are_both_retained_with_scopes():
    view=environment_view('button',(quantity('world.button.center_xyz_m',(1.,2.,3.)),),
                           (quantity('world.button.current_center_xyz_m',(1.,2.,3.)),quantity('button.depression_m',(0.,))))
    nominal=environment_row(view,'world.button.center_xyz_m');current=environment_row(view,'world.button.current_center_xyz_m')
    assert nominal.values==current.values
    assert nominal.source_scope=='c_static' and current.source_scope=='o_dynamic'
    assert nominal.ref!=current.ref and nominal.object_type==current.object_type=='button'
    assert environment_row(view,'button.depression_m').source_scope=='o_dynamic'


def cone_sources(z=.2, *,certified=True,ids=('0',),radius=.1,height=.4):
    items=[quantity('world.environment.floor_height_m',(0.,))] if certified else []
    for index,i in enumerate(ids):
        items.extend((quantity('world.cones.'+i+'.center_xy_m',(2.+10*index,3.)),
                      quantity('world.environment_cones.'+i+'.center_xyz_m',(2.+10*index,3.,z)),
                      quantity('environment_cones.'+i+'.height_m',(height,)),
                      quantity('environment_cones.'+i+'.radius_m',(radius,))))
    return tuple(items)


def test_navigation_cone_2d_3d_duplicate_prefers_declared_canonical_representation():
    view=environment_view(static=cone_sources())
    centers=[r for r in view.environment if getattr(r,'object_type',None)=='cone' and r.geometry_kind in ('point3','line')]
    assert len(centers)==1 and centers[0].geometry_kind=='point3'
    assert centers[0].source_name=='world.environment_cones.0.center_xyz_m'


@pytest.mark.parametrize('z,certified',[(.7,True),(.2,False)])
def test_navigation_cone_xy_match_alone_does_not_prove_3d_equivalence(z,certified):
    view=environment_view(static=cone_sources(z,certified=certified))
    assert sorted(r.geometry_kind for r in view.environment if getattr(r,'object_type',None)=='cone' and r.geometry_kind in ('point3','line'))==['line','point3']


def test_repeated_cone_dimensions_follow_schema_dedup_rule():
    items=cone_sources(ids=('3','9'))+(quantity('cones.cone_height_m',(.4,)),quantity('cones.cone_radius_m',(.1,)))
    view=environment_view(static=items)
    names={getattr(r,'source_name',None) for r in view.environment}
    assert 'cones.cone_radius_m' not in names and 'cones.cone_height_m' not in names
    assert 'environment_cones.3.radius_m' in names and 'environment_cones.9.radius_m' in names


def test_cone_dedup_keeps_shared_dimension_if_any_object_lacks_equivalent_measurement():
    items=tuple(x for x in cone_sources(ids=('3','9')) if x.name!='environment_cones.9.radius_m')+(quantity('cones.cone_radius_m',(.1,)),)
    assert environment_row(environment_view(static=items),'cones.cone_radius_m').values[0]==.1


def test_goal_point_and_goal_platform_box_remain_distinct_when_semantics_differ():
    view=environment_view(static=(quantity('world.goal.center_xyz_m',(1.,2.,3.)),
                                  quantity('world.obstacles.0.center_xyz_m',(1.,2.,3.)),
                                  quantity('world.obstacles.0.size_xyz_m',(1.,1.,.2))))
    goal=environment_row(view,'world.goal.center_xyz_m');box=environment_row(view,'world.obstacles.0.center_xyz_m')
    assert goal.values[:3]==box.values[:3] and goal.object_type=='goal' and box.object_type=='collision_box'
    assert goal.object_ref!=box.object_ref and len(view.environment)==3


def test_gap_alternate_encodings_follow_declared_precedence():
    view=environment_view('gap',(quantity('world.gap.near_edge_world_xyz_m',(2.,0.,0.)),
        quantity('world.gap.near_edge_center_world_xyz_m',(2.,0.,0.)),
        quantity('world.gap.near_edge_x_m',(2.,)),
        quantity('world.gap.crossing_direction_world_xy',(1.,0.))))
    rows=[r for r in view.environment if getattr(r,'object_ref',None)=='gap.near_edge']
    assert len(rows)==1 and rows[0].source_name=='world.gap.near_edge_center_world_xyz_m'
    assert rows[0].geometry_kind=='point3'


def test_gap_dedup_retains_physically_conflicting_alternate_boundary():
    view=environment_view('gap',(quantity('world.gap.near_edge_center_world_xyz_m',(2.,0.,0.)),
                               quantity('world.gap.near_edge_x_m',(3.,)),quantity('world.gap.crossing_direction_world_xy',(1.,0.))))
    assert len([r for r in view.environment if getattr(r,'object_ref',None)=='gap.near_edge'])==2


def test_stairs_alternate_encodings_follow_declared_precedence():
    view=environment_view('stairs',(quantity('world.stairs.first_riser_x_m',(2.,)),
        quantity('world.stairs.first_riser_center_world_xyz_m',(2.,0.,0.)),
        quantity('world.stairs.ascent_direction_world_xy',(1.,0.)),
        quantity('world.stairs.top_heights_m',(.1,.2)),quantity('world.stairs.top_heights_world_m',(.1,.2))))
    risers=[r for r in view.environment if getattr(r,'object_ref',None)=='stairs.first_riser']
    assert len(risers)==1 and risers[0].geometry_kind=='point3'
    tops=[r for r in view.environment if getattr(r,'object_ref',None)=='stairs.top_heights']
    assert len(tops)==2 and all(r.source_name=='world.stairs.top_heights_world_m' for r in tops)


def test_dedup_is_independent_of_object_id_order_and_source_order():
    a=environment_view(static=cone_sources(ids=('3','9')))
    b=environment_view(static=tuple(reversed(cone_sources(ids=('42','1')))))
    def payloads(view): return sorted((r.object_type,r.geometry_kind,r.numeric) for r in view.environment)
    assert payloads(a)==payloads(b)
    assert all('3' not in field.channels for r in a.environment for field in r.fields)


def test_environment_instance_ids_are_routing_only():
    a=environment_view(static=cone_sources(ids=('1',)))
    b=environment_view(static=cone_sources(ids=('99',)))
    assert sorted((r.geometry_kind,r.numeric) for r in a.environment)==sorted((r.geometry_kind,r.numeric) for r in b.environment)
    assert {r.ref for r in a.environment}!={r.ref for r in b.environment}


@pytest.mark.parametrize('payload',['labels','target_morphology','graph_t_plus_1','nav2_path','waypoints','route'])
def test_environment_rows_ignore_unapproved_metadata(payload):
    from drl.IL.bc_v1.policy_inputs import select_policy_inputs
    graph=f.state_graph(2,1); context=c.TaskContext('navigation',cone_sources());obs=c.ObservationBundle()
    expected=extract(inputs=select_policy_inputs(graph,obs,context)).environment
    class Trap:
        def __getattribute__(self,name): raise AssertionError('unapproved metadata read')
    object.__setattr__(context,payload,Trap())
    actual=extract(inputs=select_policy_inputs(graph,obs,context)).environment
    assert actual==expected and actual


def test_current_physical_cone_change_changes_environment_row():
    a=environment_view(static=(quantity('world.environment_cones.0.center_xyz_m',(1.,2.,3.)),))
    b=environment_view(static=(quantity('world.environment_cones.0.center_xyz_m',(2.,2.,3.)),))
    assert a.environment and b.environment and a.environment[0].numeric!=b.environment[0].numeric


def test_environment_extension_preserves_rows_without_candidates():
    view=environment_view(static=cone_sources())
    assert view.environment
    assert not hasattr(view,'candidates') and not hasattr(view,'batch_indices')


def test_environment_extraction_never_calls_e1_transforms_again(monkeypatch):
    inputs=c.PolicyInputs(f.state_graph(2,1),c.ObservationBundle(),c.TaskContext('button',(quantity('world.button.center_xyz_m',(2.,3.,4.)),)))
    geometry=e1.transform_inputs(inputs,axes())
    def forbidden(*a,**k): raise AssertionError('E1 transform called by E2')
    for name in ('transform_inputs','physical_frame','transform_point','transform_vector','transform_orientation','transform_twist'):
        monkeypatch.setattr(e1,name,forbidden)
    view=extract(inputs=inputs,geometry=geometry)
    assert environment_row(view,'world.button.center_xyz_m').values[:3]==(3.,-1.5,4.)


@pytest.mark.parametrize('case',['unapproved_quantity','missing_quantity','wrong_shape','intrinsic_shape_missing'])
def test_environment_extractor_rejects_unresolved_or_malformed_e1_sources(case):
    inputs=c.PolicyInputs(f.state_graph(2,1),c.ObservationBundle(),c.TaskContext('button',(quantity('button.plunger_depth_m',(.04,)),)))
    geometry=e1.transform_inputs(inputs,e1.PhysicalFrameConfig());q=geometry.quantities[0]
    if case=='unapproved_quantity': geometry=replace(geometry,quantities=(replace(q,source_name='future.stairs'),))
    elif case=='missing_quantity': geometry=replace(geometry,quantities=())
    elif case=='wrong_shape': geometry=replace(geometry,quantities=(replace(q,values=(1.,2.)),))
    else: geometry=replace(geometry,quantities=(replace(q,source_scope='c_static',source_name='not_approved',values=None,valid=False),))
    with pytest.raises(ValueError): extract(inputs=inputs,geometry=geometry)


@pytest.fixture(scope='module')
def task3_real_rows(tmp_path_factory):
    from drl.IL.tests.bc_v1.test_policy_inputs import real_policy_rows
    yield from real_policy_rows.__wrapped__(tmp_path_factory)


@pytest.mark.parametrize('case',['assembly','gap','stairs','button','navigation','terminal-goal'])
def test_environment_six_targeted_real_cases(task3_real_rows,case):
    inputs=task3_real_rows[case][4]
    spec=c.load_robot_spec(ROOT/'configs/bc_v1/smores_ep.robot.json')
    view=extract(inputs=inputs,spec=spec)
    assert view.environment
    assert all(r.source_scope in ('c_static','o_dynamic') and r.source_name and r.geometry_kind for r in view.environment)
    if case in ('assembly','button'):
        nominal=environment_row(view,'world.button.center_xyz_m');current=environment_row(view,'world.button.current_center_xyz_m')
        assert nominal.source_scope=='c_static' and current.source_scope=='o_dynamic'
    if case=='navigation':
        centers=[r for r in view.environment if r.object_type=='cone' and r.geometry_kind in ('point3','line')]
        assert len(centers)==8 and all(r.geometry_kind=='point3' for r in centers)
        assert not any('waypoint' in r.source_name or 'route' in r.source_name for r in view.environment)
    if case=='terminal-goal':
        assert any(r.object_type=='goal' for r in view.environment) and any(r.object_type=='collision_box' for r in view.environment)
    assert all(__import__('math').isfinite(v) for r in view.environment for v in r.numeric)


def test_gap_dedup_requires_true_normal_alignment_for_small_nonzero_direction():
    view=environment_view('gap',(quantity('world.gap.near_edge_center_world_xyz_m',(2.,0.,0.)),
        quantity('world.gap.near_edge_x_m',(2.,)),
        quantity('world.gap.crossing_direction_world_xy',(0.,1e-12))))
    assert len([r for r in view.environment if r.object_ref=='gap.near_edge'])==2


def test_navigation_environment_rows_ignore_raw_nav2_path_waypoint_and_route_changes():
    from drl.IL.bc_v1.task_context import build_task_view
    row=f.task_row('navigation')
    metadata={'course':{'frame_id':'world','navigation':{
        'waypoints_xyyaw':[[9.,8.,0.]],'nominal_route':{'waypoints_xyyaw':[[7.,6.,0.]]}},
        'physical_track':{'frame_id':'world','cone_centers_xy_m':[[1.,2.]],
                          'centerline_xy_m':[[20.,20.]],'cone_radius_m':.05}}}
    config=json.loads((ROOT/'configs/bc_v1/task_context.json').read_text())
    def encode(raw,source):
        context,observation=build_task_view(raw,source,config)
        view=extract(inputs=c.PolicyInputs(f.state_graph(2,1),observation,context))
        return view.environment,view.relations
    a=encode(row,metadata);changed=copy.deepcopy(metadata)
    changed['course']['navigation']={'waypoints_xyyaw':[[999.,-999.,3.]],'nav2_path':[[99.,99.]]}
    changed['course']['physical_track']['centerline_xy_m']=[[-999.,999.]]
    perturbed={**row,'action_t':{'arbitrary':'label'},'target_morphology':{'final_graph':'forbidden'},'graph_t_plus_1':{'future':'forbidden'}}
    assert a[0] and a[1] and encode(perturbed,changed)==a


def test_environment_import_isolation_during_actual_extraction():
    script='''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'rclpy','isaacsim','isaaclab','omni','torch','smores_ep','mssr_expert','freebot_docking'} or fullname in {'drl.IL.bc_v1.smores_adapter','drl.IL.bc_v1.task_context','drl.IL.preprocessing'}:
   raise ImportError('environment isolation: '+fullname)
sys.meta_path.insert(0,Block())
from drl.IL.tests.bc_v1.fixtures import state_graph,robot_spec
from drl.IL.bc_v1.bc_v1_contract import PolicyInputs,ObservationBundle,TaskContext,NumericFeature
from drl.IL.bc_v1.features import transform_inputs,PhysicalFrameConfig
from drl.IL.bc_v1.feature_schema import extract_features,load_feature_schema
p=PolicyInputs(state_graph(2,1),ObservationBundle((NumericFeature('world.button.current_center_xyz_m',(1.,2.,3.)),)),TaskContext('button',(NumericFeature('world.button.center_xyz_m',(1.,2.,3.)),)))
v=extract_features(p,transform_inputs(p,PhysicalFrameConfig()),robot_spec(),load_feature_schema('configs/bc_v1/features.json'))
assert len(v.environment)==2 and {r.source_scope for r in v.environment}=={'c_static','o_dynamic'}
assert len(v.relations)==4
assert all(r.source_scope in {'c_static','o_dynamic'} for r in v.relations)
'''
    result=subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_navigation_cone_dedup_is_safe_under_independent_collection_reorder():
    from drl.IL.bc_v1.task_context import build_task_view
    row=f.task_row('navigation')
    metadata={'course':{'frame_id':'world','floor_height_m':0.,
        'cone_centers_xy_m':[[1.,2.],[4.,5.]],'navigation_cones':[
            {'center_xyz_m':[1.,2.,.2],'height_m':.4,'radius_m':.1},
            {'center_xyz_m':[4.,5.,.5],'height_m':1.,'radius_m':.2}]}}
    config=json.loads((ROOT/'configs/bc_v1/task_context.json').read_text())
    def rows(source):
        context,observation=build_task_view(row,source,config)
        return extract(inputs=c.PolicyInputs(f.state_graph(2,1),observation,context)).environment
    baseline=rows(metadata)
    assert len(baseline)==7
    for key in ('navigation_cones','cone_centers_xy_m'):
        changed=copy.deepcopy(metadata); changed['course'][key].reverse()
        other=rows(changed)
        assert sorted((r.object_type,r.geometry_kind,r.numeric) for r in other)==sorted((r.object_type,r.geometry_kind,r.numeric) for r in baseline)
        assert len([r for r in other if r.object_type=='cone' and r.geometry_kind in ('point3','line')])==2


def test_cone_dedup_retains_ambiguous_cross_collection_matches_without_id_tiebreak():
    items=cone_sources(ids=('3','9'))
    items=tuple(replace(x,values=(2.,3.)) if x.name.endswith('center_xy_m') else
                replace(x,values=(2.,3.,.2)) if x.name.endswith('center_xyz_m') else x for x in items)
    view=environment_view(static=items)
    centers=[r for r in view.environment if r.object_type=='cone' and r.geometry_kind in ('point3','line')]
    assert len(centers)==4
    assert sorted(r.geometry_kind for r in centers)==['line','line','point3','point3']


# E2 Task4 — per-module closest geometry, schema budgets, no action candidates.
def relation_view(task='button',static=(),dynamic=(),graph=None,config=None,config_schema=None):
    inputs=c.PolicyInputs(f.state_graph(2,1) if graph is None else graph,c.ObservationBundle(dynamic),c.TaskContext(task,static))
    return extract(inputs=inputs,config=config,config_schema=config_schema)


def relation(view,module=0,object_ref=None,scope=None):
    refs=[r for r in view.relations if getattr(r,'module_ref',None)==view.modules[module].ref and
          (object_ref is None or r.object_ref==object_ref) and (scope is None or r.source_scope==scope)]
    assert len(refs)==1,'Missing/duplicate relation for module/object'
    return refs[0]


def point_scene(point=(4.,3.,0.),graph=None,config=None):
    return relation_view(static=(quantity('world.button.center_xyz_m',point),),graph=graph,config=config)


def test_point_relation_delta_xyz_is_object_minus_module():
    assert relation(point_scene()).values[:3]==(4.,3.,0.)


def test_point_relation_distance_is_euclidean_norm():
    row=relation(point_scene())
    assert row.values[3]==5. and row.validity[3]
    assert len(row.numeric)==14 and tuple(x.width for x in row.fields)==(3,1,3)


def test_point_relation_direction_is_normalized():
    assert relation(point_scene()).values[4:]==pytest.approx((.8,.6,0.))


def test_zero_distance_keeps_distance_valid_but_direction_invalid():
    row=relation(point_scene((0.,0.,0.)))
    assert row.values==(0.,)*7 and row.validity==(True,)*4+(False,)*3


def test_each_module_gets_its_own_point_relation_geometry():
    view=point_scene()
    assert relation(view,0).values[:3]==(4.,3.,0.)
    assert relation(view,1).values[:3]==(3.,3.,0.)
    assert relation(view,0).values[3]!=relation(view,1).values[3]


def test_centroid_is_not_used_as_module_substitute():
    view=point_scene();assert len(view.relations)==2
    assert [relation(view,i).values[0] for i in range(2)]==[4.,3.]
    assert not any(r.values[0]==3.5 for r in view.relations)


def boundary_scene(graph=None,config=None):
    return relation_view('gap',(quantity('world.gap.near_edge_center_world_xyz_m',(2.,0.,0.)),
        quantity('world.gap.far_edge_center_world_xyz_m',(3.,0.,0.)),
        quantity('world.gap.crossing_direction_world_xy',(1.,0.))),graph=graph,config=config)


def test_plane_relation_uses_perpendicular_distance_not_anchor_distance():
    graph=f.state_graph(2,1);graph=replace(graph,modules=(replace(graph.modules[0],position=(0.,100.,50.)),replace(graph.modules[1],position=(1.,-8.,-3.))))
    row=relation(boundary_scene(graph),0,'gap.near_edge')
    assert row.geometry_kind=='plane' and row.values[:4]==(2.,0.,0.,2.)
    assert row.values[3]!=pytest.approx((2**2+100**2+50**2)**.5)


def test_plane_relation_preserves_signed_side_without_adding_unsigned_schema_channel():
    graph=f.state_graph(2,1);graph=replace(graph,modules=(replace(graph.modules[0],position=(1.,4.,0.)),replace(graph.modules[1],position=(3.,4.,0.))))
    view=boundary_scene(graph)
    assert relation(view,0,'gap.near_edge').values[:4]==(1.,0.,0.,1.)
    assert relation(view,1,'gap.near_edge').values[:4]==(-1.,0.,0.,1.)
    assert [field.name for field in relation(view,0,'gap.near_edge').fields]==['delta_xyz','distance','direction']


def test_valid_direct_plane_relation_uses_e1_plane_codec():
    view=relation_view('gap',(quantity('world.gap.near_edge_x_m',(2.,)),))
    assert relation(view,0,'gap.near_edge').values[:4]==(2.,0.,0.,2.)


def test_line_relation_is_not_treated_as_point3():
    graph=f.state_graph(2,1);graph=replace(graph,modules=(replace(graph.modules[0],position=(0.,0.,10.)),replace(graph.modules[1],position=(1.,0.,-8.))))
    view=relation_view('navigation',(quantity('world.cones.0.center_xy_m',(2.,0.)),),graph=graph)
    row=relation(view,0)
    assert row.geometry_kind=='line' and row.values[:4]==(2.,0.,0.,2.)


@pytest.mark.parametrize('task,name,values',[
    ('button','button.plunger_depth_m',(.04,)),('button','button.face_size_tangent_z_m',(.08,.08)),
    ('button','world.button.press_yaw_rad',(0.,)),('navigation','world.goal.xyyaw',(1.,2.,0.)),
    ('navigation','world.obstacles.0.yaw_deg',(90.,)),
])
def test_partial_or_intrinsic_geometry_does_not_automatically_create_spatial_relation(task,name,values):
    view=relation_view(task,(quantity(name,values),))
    assert view.environment and view.relations==()


def test_invalid_environment_geometry_placeholder_is_never_used_as_measurement():
    view=relation_view(static=(quantity('world.button.center_xyz_m',(0.,0.,0.),False),))
    assert view.environment and view.relations==()


def test_invalid_module_position_is_not_used_as_physical_zero_in_relations():
    graph=f.state_graph(2,1);graph=replace(graph,modules=(replace(graph.modules[0],pose_valid=False),graph.modules[1]))
    view=point_scene(graph=graph)
    assert len(view.relations)==1 and view.relations[0].module_ref==graph.modules[1].ref


def test_invalid_gap_local_planes_do_not_create_fake_zero_plane_relations():
    static=(quantity('stage_local.gap.near_edge_x_m',(0.,)),quantity('stage_local.gap.far_edge_x_m',(.178,)))
    view=relation_view('gap',static)
    assert len(view.environment)==2 and not view.relations


def test_gap_valid_anchor_normal_geometry_produces_boundary_relation():
    view=boundary_scene();row=relation(view,0,'gap.near_edge')
    assert row.geometry_kind=='plane' and row.relation_kind=='anchor_normal_plane'
    assert {r[2] for r in row.geometry_sources}=={'world.gap.near_edge_center_world_xyz_m','world.gap.crossing_direction_world_xy'}


def test_gap_near_and_far_boundary_relations_are_distinct():
    view=boundary_scene()
    assert relation(view,0,'gap.near_edge').values[3]==2.
    assert relation(view,0,'gap.far_edge').values[3]==3.


def stairs_scene():
    return relation_view('stairs',(
        quantity('stage_local.stairs.base_height_m',(0.,)),quantity('stage_local.stairs.first_riser_x_m',(2.,)),quantity('stage_local.stairs.top_heights_m',(.1,.2,.3,.4)),
        quantity('world.environment.floor_height_m',(0.,)),
        quantity('world.stairs.first_riser_center_world_xyz_m',(2.,0.,0.)),quantity('world.stairs.ascent_direction_world_xy',(1.,0.)),quantity('world.stairs.top_heights_world_m',(.1,.2,.3,.4))))


def test_stairs_invalid_local_planes_are_not_used():
    view=stairs_scene();assert view.relations
    assert all(not r.source_name.startswith('stage_local.') for r in view.relations)


def test_stairs_valid_world_first_riser_geometry_produces_relation():
    assert relation(stairs_scene(),0,'stairs.first_riser').values[:4]==(2.,0.,0.,2.)


def test_stairs_valid_top_plane_geometry_selects_closest_source_group_component():
    row=relation(stairs_scene(),0,'stairs.top_heights')
    assert row.geometry_kind=='plane' and row.values[:4]==pytest.approx((0.,0.,.1,.1))
    assert row.primitive_part_ref==0
    assert not any(r.object_type=='environment_boundary' for r in stairs_scene().relations)


def cone_relation_scene(xs=(1.,3.,5.,7.),ids=None,config_schema=None):
    graph=f.state_graph(1,1);ids=tuple(str(i) for i in range(len(xs))) if ids is None else ids
    return relation_view('navigation',tuple(quantity('world.environment_cones.'+i+'.center_xyz_m',(x,0.,0.)) for i,x in zip(ids,xs)),graph=graph,config_schema=config_schema)


def test_relation_generation_is_schema_rule_driven():
    value=schema();value['relation_policy']['selection']['max_neighbors_per_type']=1
    assert len(cone_relation_scene(config_schema=value).relations)==1
    assert len(cone_relation_scene().relations)==2


def test_relation_nearest_budget_is_two_when_no_ties():
    view=cone_relation_scene()
    assert sorted(r.values[3] for r in view.relations)==[1.,3.]


def test_relation_boundary_distance_ties_are_preserved_without_id_tiebreak():
    view=cone_relation_scene((1.,2.,-2.,2.),('99','1','7','0'))
    assert len(view.relations)==4 and sorted(r.values[3] for r in view.relations)==[1.,2.,2.,2.]


def test_relation_selection_is_invariant_to_object_row_reorder_and_rename():
    a=cone_relation_scene((1.,2.,-2.,2.),('99','1','7','0'))
    b=cone_relation_scene((2.,-2.,2.,1.),('42','55','66','77'))
    assert a.relations and sorted(r.numeric for r in a.relations)==sorted(r.numeric for r in b.relations)


def test_module_id_rename_does_not_change_learned_relation_values():
    graph=f.state_graph(2,1);a=point_scene(graph=graph);b=point_scene(graph=renamed_modules(graph))
    assert a.relations and tuple(r.numeric for r in a.relations)==tuple(r.numeric for r in b.relations)
    assert tuple(r.module_ref for r in a.relations)!=tuple(r.module_ref for r in b.relations)


def test_relation_selection_uses_physical_object_not_each_collision_feature_row():
    items=[]
    for i in range(8):
        items.extend((quantity(f'world.obstacles.{i}.center_xyz_m',(2.+i,0.,0.)),quantity(f'world.obstacles.{i}.size_xyz_m',(1.,1.,1.)),quantity(f'world.obstacles.{i}.yaw_deg',(0.,)),quantity(f'world.obstacles.{i}.pitch_deg',(0.,))))
    view=relation_view('navigation',tuple(items))
    assert len(view.environment)==32 and len(view.relations)==4
    assert all(r.geometry_kind=='point3' for r in view.relations)
    assert len(view.relations)<len(view.modules)*len(view.environment)


@pytest.mark.parametrize('payload',['expert_action','supervision','module_resource_masks','behavioral_stage','source_morphology','target_morphology','final_graph','G_t_plus_1','reward','success','executor_fsm','nav2_path','waypoints','centerline'])
def test_relation_output_independent_of_forbidden_metadata(payload):
    from drl.IL.bc_v1.policy_inputs import select_policy_inputs
    graph=f.state_graph(2,1);context=c.TaskContext('button',(quantity('world.button.center_xyz_m',(4.,3.,0.)),));obs=c.ObservationBundle()
    expected=extract(inputs=select_policy_inputs(graph,obs,context)).relations
    object.__setattr__(context,payload,{'forbidden':999})
    other=extract(inputs=select_policy_inputs(graph,obs,context)).relations
    assert expected and other==expected


def test_physical_object_motion_changes_relation_values():
    a=point_scene();b=point_scene((5.,3.,0.))
    assert a.relations and relation(a).numeric!=relation(b).numeric


def test_unknown_task_produces_no_task_specific_relations():
    view=relation_view(None,(quantity('world.obstacles.0.center_xyz_m',(2.,0.,0.)),))
    assert view.environment and view.relations==()


def test_relation_object_type_filter_respects_current_task():
    view=relation_view('navigation',(quantity('world.environment_cones.0.center_xyz_m',(2.,0.,0.)),),
                       (quantity('world.button.current_center_xyz_m',(1.,0.,0.)),))
    assert view.relations and all(r.object_type=='cone' for r in view.relations)


def test_relation_engine_never_calls_e1_frame_or_coordinate_transforms(monkeypatch):
    inputs=c.PolicyInputs(f.state_graph(2,1),c.ObservationBundle(),c.TaskContext('button',(quantity('world.button.center_xyz_m',(4.,3.,0.)),)))
    geometry=e1.transform_inputs(inputs,axes())
    def forbidden(*args,**kw):raise AssertionError('second coordinate transform')
    for name in ('physical_frame','transform_inputs','transform_point','transform_vector','transform_orientation','transform_twist'):
        monkeypatch.setattr(e1,name,forbidden)
    view=extract(inputs=inputs,geometry=geometry)
    assert relation(view).values[:4]==(3.,-4.,0.,5.)


def test_relation_frame_change_preserves_point_and_plane_distances():
    a=point_scene();b=point_scene(config=axes())
    assert relation(a).values[3]==relation(b).values[3]==5.
    assert relation(b).values[:3]==(3.,-4.,0.)
    plane_a=boundary_scene();plane_b=boundary_scene(config=axes())
    assert relation(plane_a,0,'gap.near_edge').values[3]==relation(plane_b,0,'gap.near_edge').values[3]==2.
    assert relation(plane_b,0,'gap.near_edge').values[:3]==(0.,-2.,0.)


def test_relation_rows_are_immutable_and_task4_does_not_create_candidates():
    view=point_scene();assert view.relations and not hasattr(view,'candidates')
    with pytest.raises(FrozenInstanceError):view.relations[0].values=()
    assert isinstance(view.relations,tuple) and isinstance(view.relations[0].geometry_sources,tuple)


@pytest.mark.parametrize('case',['assembly','gap','stairs','button','navigation','terminal-goal'])
def test_relation_six_targeted_real_cases(task3_real_rows,case):
    inputs=task3_real_rows[case][4];spec=c.load_robot_spec(ROOT/'configs/bc_v1/smores_ep.robot.json');view=extract(inputs=inputs,spec=spec)
    assert view.relations and all(r.module_ref in {m.ref for m in view.modules} for r in view.relations)
    assert all(not r.source_name.startswith('stage_local.') for r in view.relations)
    if case in ('assembly','button'):
        assert {r.source_scope for r in view.relations if r.object_type=='button'}=={'c_static','o_dynamic'}
        distances={r.values[3] for r in view.relations if r.object_type=='button'};assert len(distances)>1
    if case=='gap':
        assert {'gap.near_edge','gap.far_edge'}<={r.object_ref for r in view.relations}
        assert all(r.geometry_kind=='plane' for r in view.relations if r.object_type=='gap_boundary')
    if case=='stairs':
        assert {'stairs.first_riser','stairs.top_heights'}<={r.object_ref for r in view.relations}
        assert all(r.geometry_kind=='plane' for r in view.relations if r.object_type=='stair_riser')
    if case=='navigation':
        assert len(view.relations)<len(view.modules)*len(view.environment)/4
        for module in view.modules:assert len([r for r in view.relations if r.module_ref==module.ref and r.object_type=='cone'])==2
    if case=='terminal-goal':assert any(r.object_type=='goal' for r in view.relations)


def test_relation_aliases_of_same_geometry_keep_all_sources_without_arbitrary_primary():
    items=(quantity('world.goal.center_xyz_m',(2.,0.,0.)),quantity('world.goal.position_xyz_m',(2.,0.,0.)))
    view=relation_view('navigation',items)
    assert len(view.relations)==2
    for row in view.relations:
        assert row.source_name is None
        assert {ref[2] for ref in row.geometry_sources}=={'world.goal.center_xyz_m','world.goal.position_xyz_m'}
    other=relation_view('navigation',tuple(reversed(items)))
    assert tuple(r.numeric for r in view.relations)==tuple(r.numeric for r in other.relations)
    assert all(r.source_name is None for r in other.relations)


def test_relation_tied_top_components_are_kept_without_expert_next_step():
    view=relation_view('stairs',(quantity('world.stairs.top_heights_world_m',(.1,.3,.8)),),
                        graph=replace(f.state_graph(1,1),modules=(replace(f.state_graph(1,1).modules[0],position=(0.,0.,.2)),)))
    assert len(view.relations)==2
    assert sorted(r.values[2] for r in view.relations)==pytest.approx([-.1,.1])
    assert {r.primitive_part_ref for r in view.relations}=={0,1}


def test_relation_boundary_normal_scale_does_not_change_plane_geometry():
    def scene(scale):
        return relation_view('gap',(quantity('world.gap.near_edge_center_world_xyz_m',(2.,2.,0.)),quantity('world.gap.crossing_direction_world_xy',(scale,scale))))
    a,b=scene(1.),scene(1.7e308)
    assert a.relations and b.relations
    assert relation(a,0,'gap.near_edge').numeric==pytest.approx(relation(b,0,'gap.near_edge').numeric)


def test_relation_equivalent_lines_with_shifted_parameter_origins_merge():
    items=(quantity('world.cones.0.center_xy_m',(2.,0.)),
           quantity('map.cones.0.center_xy_m',(2.,0.)))
    config=e1.PhysicalFrameConfig(transforms=(e1.SourceFrameTransform('map',(0.,0.,10.)),))
    view=relation_view('navigation',items,config=config)
    assert len(view.environment)==2 and len(view.relations)==2
    for row in view.relations:
        assert row.geometry_kind=='line' and row.source_name is None
        assert len(row.geometry_sources)==2 and row.values[2]==0.
    other=relation_view('navigation',items[::-1],config=config)
    assert tuple(r.numeric for r in view.relations)==tuple(r.numeric for r in other.relations)


def test_relation_equivalent_boundary_normals_merge_despite_bridge_roundoff():
    cs,sn=math.cos(.1),math.sin(.1)
    items=(quantity('world.gap.near_edge_center_world_xyz_m',(2.,2.,0.)),
           quantity('world.gap.crossing_direction_world_xy',(1.,2.)),
           quantity('map.gap.crossing_direction_world_xy',(cs+2*sn,-sn+2*cs)))
    config=e1.PhysicalFrameConfig(transforms=(e1.SourceFrameTransform('map',(0.,0.,0.),(cs,-sn,0.,sn,cs,0.,0.,0.,1.)),))
    view=relation_view('gap',items,config=config)
    assert len(view.environment)==3 and len(view.relations)==2
    for i in range(2):
        row=relation(view,i,'gap.near_edge')
        assert len(row.geometry_sources)==3 and row.source_name is None
        assert row.values[3]==pytest.approx((6-i)/math.sqrt(5))
    other=relation_view('gap',items[::-1],config=config)
    assert tuple(r.numeric for r in view.relations)==tuple(r.numeric for r in other.relations)


def test_relation_merged_plane_alias_parts_have_no_arbitrary_singular_part():
    items=(quantity('world.stairs.top_heights_world_m',(.1,.3)),
           quantity('stage_local.stairs.top_heights_m',(.3,.1)))
    config=e1.PhysicalFrameConfig(transforms=(e1.SourceFrameTransform('stage_local',(0.,0.,0.)),))
    a=relation_view('stairs',items,config=config)
    b=relation_view('stairs',items[::-1],config=config)
    assert len(a.relations)==len(b.relations)==2
    for row in (*a.relations,*b.relations):
        assert row.source_name is None and row.primitive_part_ref is None
        assert {source[3] for source in row.geometry_sources}=={0,1}
    assert tuple(r.numeric for r in a.relations)==tuple(r.numeric for r in b.relations)


def test_relation_equivalent_boundary_normals_merge_with_opposite_direction():
    items=(quantity('world.gap.near_edge_center_world_xyz_m',(2.,2.,0.)),
           quantity('world.gap.crossing_direction_world_xy',(1.,2.)),
           quantity('map.gap.crossing_direction_world_xy',(-1.,-2.)))
    config=e1.PhysicalFrameConfig(transforms=(e1.SourceFrameTransform('map',(0.,0.,0.)),))
    view=relation_view('gap',items,config=config)
    assert len(view.relations)==2
    assert all(len(r.geometry_sources)==3 for r in view.relations)


def test_relation_physically_distinct_parallel_line_components_are_not_merged():
    items=(quantity('world.cones.0.center_xy_m',(2.,0.)),
           quantity('map.cones.0.center_xy_m',(2.,0.)))
    config=e1.PhysicalFrameConfig(transforms=(e1.SourceFrameTransform('map',(-4.,0.,10.)),))
    view=relation_view('navigation',items,graph=f.state_graph(1,1),config=config)
    assert len(view.relations)==2
    assert sorted(r.values[0] for r in view.relations)==[-2.,2.]
