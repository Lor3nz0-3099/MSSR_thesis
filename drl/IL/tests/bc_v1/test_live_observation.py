"""L1b semantic parity on publisher-shaped synthetic physical snapshots."""
import copy
from dataclasses import replace
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from drl.IL.bc_v1 import bc_v1_contract as c, normalization as f2
from drl.IL.bc_v1.batching import collate_policy_samples
from drl.IL.bc_v1.dataset import CachedSample, CurrentCandidateCache, derive_physical_inputs
from drl.IL.bc_v1.feature_schema import load_feature_schema
from drl.IL.bc_v1.features import quaternion_to_matrix
from drl.IL.bc_v1.smores_adapter import SMORESAdapter
from drl.IL.bc_v1.source_frame_bridges import BridgeResolution, SourceFrameBridgeBatch
from drl.IL.bc_v1.tensorization import load_tensor_layout, tensorize_policy_sample

ROOT = Path(__file__).resolve().parents[4]
F2_SHA = 'ea8efbd9062acb8625e0618bcc5f86b975f20839bf3132a684a626c744e52003'


def api():
    name = 'drl.IL.bc_v1.live_observation'
    assert importlib.util.find_spec(name), 'Missing L1b live observation bridge'
    return importlib.import_module(name)


@pytest.fixture(autouse=True, scope='module')
def threads():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def builder():
    # Pure existing runtime graph code, installed normally in the ROS runtime.
    path = str(ROOT / 'mssr_ws/src/mssr_expert')
    if path not in sys.path: sys.path.insert(0, path)
    from mssr_expert.graph.graph_builder import GraphBuilder
    return GraphBuilder()


def settings():
    cfg = ROOT / 'configs/bc_v1'
    spec = c.load_robot_spec(cfg / 'smores_ep.robot.json')
    schema = load_feature_schema(cfg / 'features.json')
    policy = f2.load_normalization_policy(cfg / 'normalization.json', schema)
    path = ROOT / 'artifacts/bc_v1/normalization/f2_v1_optimized/normalizers.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == F2_SHA
    # Independent SHA pin covers the complete metadata/scalers, no fitting.
    stats = f2.read_normalizers(path, expected_metadata=json.loads(path.read_text())['metadata'])
    return dict(spec=spec, feature_schema=schema, normalization_policy=policy,
        normalizers=stats, tensor_layout=load_tensor_layout(cfg / 'tensor_layout.v1.json'))


def world_bridges():
    return SourceFrameBridgeBatch((BridgeResolution('world', 'world', 'current-live',
        None, True, 'current publisher WORLD contract', None, ()),),
        (('provider_id', 'synthetic-current-physical'), ('revision', '1')))


def snapshot(n=2, *, attached=False, rc=False):
    spec = settings()['spec']; mt = spec.module_types[0]; modules = []
    for i in range(n):
        position = [1.+2.*i, 2., 0.]
        q = [0., 0., math.sqrt(.5), math.sqrt(.5)] if i == 0 else [0., 0., 0., 1.]
        rotation = np.array(quaternion_to_matrix(q)).reshape(3, 3)
        ports = []
        for port in mt.connectors:
            local = np.array(quaternion_to_matrix(port.local_orientation_xyzw)).reshape(3, 3)
            ports.append(dict(connector_id=port.connector_id, face_name=port.connector_id,
                status='available', connected_module_id=None, connected_connector_id=None,
                position_world=(np.array(position)+rotation@port.local_position).tolist(),
                outward_normal_world=(rotation@local[:, 0]).tolist(), tangent_world=(rotation@local[:, 1]).tolist()))
        actuators = {co.coordinate_id: {'position_rad': .1*(j+1), 'velocity_rad_s': .2*(j+1)}
            for resource in mt.resources for j, co in enumerate(resource.coordinates)}
        modules.append(dict(module_id=f'module-{i}', position=position, orientation=q,
            pose={'frame_id':'world', 'position':position[:], 'orientation':q[:], 'orientation_xyzw':q[:]},
            linear_velocity=[1., 2., 3.], angular_velocity=[0., 0., .5],
            twist={'linear':[1., 2., 3.], 'angular':[0., 0., .5]},
            actuators=actuators, connectors=ports, current_role='unassigned', target_role='ignored'))
    edges = []
    pairs = [(0, 'TOP', 1, 'BOTTOM')] if attached else []
    if rc:
        assert n == 8
        # Literal CURRENT RC8 connectivity fixture, no target graph in inputs.
        pairs = [(1,'TOP',0,'BOTTOM'), (0,'TOP',2,'BOTTOM'), (2,'TOP',7,'BOTTOM'),
                 (3,'BOTTOM',1,'LEFT'), (4,'BOTTOM',1,'RIGHT'),
                 (5,'BOTTOM',7,'LEFT'), (6,'BOTTOM',7,'RIGHT')]
    for a, fa, b, fb in pairs:
        edges.append(dict(module_a_id=modules[a]['module_id'], module_b_id=modules[b]['module_id'],
            connector_a_id=fa, connector_b_id=fb, face_a=fa, face_b=fb,
            is_attached=True, is_contact=True, clocking_quarter_turns=0,
            edge_type='docking_connection', connection_state='latched'))
        for own, face, peer, pf in ((a,fa,b,fb),(b,fb,a,fa)):
            port = next(p for p in modules[own]['connectors'] if p['connector_id'] == face)
            port.update(status='connected', connected_module_id=modules[peer]['module_id'], connected_connector_id=pf)
    return dict(schema_version='mssr.state_graph.v2', stamp=12.5,
        state=dict(schema_version='mssr.module_states.v2', stamp=12.5, robot_family='smores_ep',
            modules=modules, contacts=[], attachments=copy.deepcopy(edges)),
        graph=dict(schema_version='mssr.robot_graph.v2', stamp=12.5,
            nodes=[{'module_id':m['module_id'], 'attributes':copy.deepcopy(m)} for m in modules],
            edges=copy.deepcopy(edges), global_attributes={}))


def context():
    return c.TaskContext('navigation', tuple(sorted((
        c.NumericFeature('world.goal.position_xyz_m', (5., 4., 0.)),
        c.NumericFeature('world.cones.0.center_xy_m', (4., 3.)),
        c.NumericFeature('cones.cone_radius_m', (.05,)),
    ), key=lambda f: f.name)))


def prepare(raw=None, ctx=None, **overrides):
    a = api(); kw = settings(); kw.update(overrides)
    state = a.encode_smores_snapshot(snapshot() if raw is None else raw,
        adapter=SMORESAdapter(kw['spec']), graph_builder=builder())
    return a.prepare_live_observation(state, context=context() if ctx is None else ctx,
        observation=c.ObservationBundle(), bridges=world_bridges(), **kw)


def values(row, name):
    offset = 0
    for field in row.fields:
        if field.name == name: return tuple(row.values[offset:offset+field.width])
        offset += field.width
    raise AssertionError(name)


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape
        torch.testing.assert_close(a, b, rtol=0., atol=0.)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: equal_tree(a[k], b[k])
    else: assert a == b


def test_current_snapshot_preserves_physical_state_and_joint_signs():
    a = api(); kw = settings(); raw = snapshot()
    state = a.encode_smores_snapshot(raw, adapter=SMORESAdapter(kw['spec']), graph_builder=builder())
    assert state.stamp == 12.5 and len(state.graph.modules) == 2
    assert state.graph.modules[0].position == (1., 2., 0.)
    coordinates = {co.coordinate_id:co for r in state.graph.resources if r.ref.module.module_id == 'module-0' for co in r.coordinates}
    for name, co in coordinates.items():
        assert co.position == raw['state']['modules'][0]['actuators'][name]['position_rad']
        assert co.velocity == raw['state']['modules'][0]['actuators'][name]['velocity_rad_s']


def test_centroid_frame_and_orientation_match_hand_computed_training_semantics():
    result = prepare()
    assert result.geometry.frame.origin == (2., 2., 0.)
    assert values(result.inputs.features.modules[0], 'relative_position') == (-1., 0., 0.)
    assert values(result.inputs.features.modules[0], 'orientation_r6') == pytest.approx((0.,1.,0.,-1.,0.,0.), abs=1e-12)


def test_twists_remain_world_vectors_without_centroid_velocity_subtraction():
    result = prepare()
    assert values(result.inputs.features.modules[0], 'linear_velocity') == (1., 2., 3.)
    assert values(result.inputs.features.modules[0], 'angular_velocity') == (0., 0., .5)


def test_current_dock_has_endpoint_a_relative_geometry_and_occupied_ports():
    result = prepare(snapshot(attached=True)); graph = result.inputs.policy.graph
    assert len(graph.connections) == 1 and graph.connections[0].relation_id == 'current_connection'
    assert sum(p.status == 'connected' for p in graph.connectors) == 2
    edge = result.inputs.features.connections[0]
    assert values(edge, 'relative_position') == pytest.approx((0., -2., 0.), abs=1e-12)
    assert values(edge, 'relative_orientation_r6') == pytest.approx((0.,-1.,0.,1.,0.,0.), abs=1e-12)


def test_goal_and_xy_cones_use_frozen_geometry_without_route_or_invented_height():
    result = prepare()
    goal = next(q for q in result.geometry.quantities if q.source_name == 'world.goal.position_xyz_m')
    cone = next(q for q in result.geometry.quantities if q.source_name == 'world.cones.0.center_xy_m')
    assert goal.kind == 'point' and goal.values == (3., 2., 0.)
    assert cone.kind == 'line' and cone.values[-3:] == (0., 0., 1.)
    assert result.batch['inputs']['task_one_hot'].tolist() == [[0., 1., 0., 0., 0.]]
    assert result.inputs.policy.observation.features == ()


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_current_pose_is_rejected_before_builder_fallback(bad):
    a = api(); raw = snapshot()
    raw['state']['modules'][0]['pose']['position'][0] = bad
    with pytest.raises(ValueError, match='finite'):
        a.encode_smores_snapshot(raw, adapter=SMORESAdapter(settings()['spec']), graph_builder=builder())


@pytest.mark.parametrize('field', ['pose', 'linear_velocity', 'actuators', 'connectors'])
def test_missing_required_physical_source_is_not_silently_zero_filled(field):
    a = api(); raw = snapshot(); del raw['state']['modules'][0][field]
    with pytest.raises(ValueError):
        a.encode_smores_snapshot(raw, adapter=SMORESAdapter(settings()['spec']), graph_builder=builder())


def test_incoherent_snapshot_stamps_and_duplicate_modules_fail_closed():
    a = api(); kw = settings()
    for mutation in ('stamp', 'duplicate'):
        raw = snapshot()
        if mutation == 'stamp': raw['graph']['stamp'] = 13.
        else: raw['state']['modules'].append(copy.deepcopy(raw['state']['modules'][0]))
        with pytest.raises(ValueError): a.encode_smores_snapshot(raw, adapter=SMORESAdapter(kw['spec']), graph_builder=builder())


def test_unsupported_pose_frame_and_missing_actuator_measurement_fail_closed():
    a = api()
    for mutation in ('frame', 'joint'):
        raw = snapshot()
        if mutation == 'frame': raw['state']['modules'][0]['pose']['frame_id'] = 'map'
        else: next(iter(raw['state']['modules'][0]['actuators'].values())).pop('velocity_rad_s')
        with pytest.raises(ValueError): a.encode_smores_snapshot(raw, adapter=SMORESAdapter(settings()['spec']), graph_builder=builder())


def test_future_topology_is_rejected_as_physical_connectivity():
    a = api(); raw = snapshot(attached=True)
    raw['graph']['edges'][0].update(is_target_edge=True, relation_type='target_connection')
    with pytest.raises(ValueError): a.encode_smores_snapshot(raw, adapter=SMORESAdapter(settings()['spec']), graph_builder=builder())


def test_forbidden_context_fields_are_not_accepted():
    a = api(); result = prepare()
    for field in ('stage', 'control_state', 'configuration_goal', 'future_graph', 'cmd_vel', 'nav2_route'):
        with pytest.raises(TypeError):
            a.prepare_live_observation(result.state, context=context(), observation=c.ObservationBundle(),
                bridges=world_bridges(), **settings(), **{field: object()})
    with pytest.raises(ValueError): prepare(ctx=c.TaskContext('navigation', (c.NumericFeature('stage', (1.,)),)))


def test_expert_roles_and_provenance_metadata_do_not_enter_learned_inputs():
    api(); raw = snapshot(); expected = prepare(raw)
    raw['state'].update(stage='secret', control_state={'fsm':'answer'}, cmd_vel=[99.,99.],
        course={'navigation':{'waypoints_xyyaw':[[999.,999.,0.]]}}, provenance={'expert':True})
    for module in raw['state']['modules']: module.update(target_role='answer', role_source='expert', current_role='answer')
    for node in raw['graph']['nodes']: node['attributes'].update(target_role='answer', current_role='answer')
    equal_tree(expected.batch['inputs'], prepare(raw).batch['inputs'])


def test_unknown_named_geometry_frame_does_not_become_valid_by_default():
    api()
    with pytest.raises(ValueError): prepare(ctx=c.TaskContext('navigation', (c.NumericFeature('map.goal.position_xyz_m', (5.,4.,0.)),)))


def test_golden_full_g1_offline_path_equals_live_path_for_current_rc8_snapshot():
    api(); raw = snapshot(8, rc=True); kw = settings(); live = prepare(raw)
    from drl.IL.preprocessing import compact_robot_graph
    from drl.IL.tests.bc_v1.test_dataset import world_runtime
    old = builder().build(raw['state'], raw['graph']).to_dict()
    row = {'graph_t':compact_robot_graph(old), 'task_type':'navigation'}
    metadata = {'course':{'frame_id':'world', 'current_goal':{'position_xyz_m':[5.,4.,0.]},
        'cone_centers_xy_m':[[4.,3.]], 'cone_radius_m':.05}}
    cfg = dict(world_runtime(), spec=kw['spec'], schema=kw['feature_schema'],
        metadata_reader=lambda row:metadata,
        task_config=json.loads((ROOT/'configs/bc_v1/task_context.json').read_text()))
    derived = derive_physical_inputs(row, SMORESAdapter(kw['spec']), cfg, CurrentCandidateCache(kw['spec'], kw['feature_schema']))
    offline = CachedSample(derived.inputs, c.SupervisionBundle(), c.SampleAudit('offline-golden',0))
    batch = collate_policy_samples([tensorize_policy_sample(offline, **kw)])
    assert derived.inputs.policy == live.inputs.policy
    assert derived.features == live.inputs.features
    assert len(live.inputs.policy.graph.connections) == 7
    equal_tree(batch['inputs'], live.batch['inputs'])
    assert batch['routing']['layout_fingerprint'] == live.batch['routing']['layout_fingerprint']
    assert batch['routing']['normalizer_fingerprint'] == live.batch['routing']['normalizer_fingerprint']
    for name in ('module_refs','connector_refs','resource_refs','module_types','mode_catalog','parameter_catalog'):
        assert batch['routing']['samples'][0][name] == live.batch['routing']['samples'][0][name]


def test_id_renaming_and_input_order_only_permute_entity_rows():
    api(); raw = snapshot(8, rc=True); before = prepare(raw)
    mapping = {f'module-{i}':f'opaque-{20-i}' for i in range(8)}
    def rename(value):
        if isinstance(value, dict): return {k:rename(v) for k,v in value.items()}
        if isinstance(value, list): return [rename(v) for v in value]
        return mapping.get(value, value) if isinstance(value,str) else value
    renamed = rename(raw); renamed['state']['modules'].reverse(); renamed['graph']['nodes'].reverse()
    after = prepare(renamed)
    inverse = {v:k for k,v in mapping.items()}
    def unref(ref):
        if isinstance(ref,c.ModuleRef): return c.ModuleRef(inverse.get(ref.module_id,ref.module_id))
        if isinstance(ref,c.ResourceRef): return c.ResourceRef(unref(ref.module),ref.resource_id)
        if isinstance(ref,c.ConnectorRef): return c.ConnectorRef(unref(ref.module),ref.connector_id)
        if isinstance(ref,tuple): return tuple(unref(v) for v in ref)
        return ref
    for family in ('modules','connectors','resources','coordinates','connections','relations'):
        left = {r.ref:r for r in getattr(before.inputs.features,family)}
        right = {unref(r.ref):replace(r,ref=unref(r.ref),module_ref=unref(r.module_ref) if r.module_ref else None)
            for r in getattr(after.inputs.features,family)}
        assert left.keys() == right.keys()
        for ref in left:
            assert left[ref].values == right[ref].values
            assert left[ref].validity == right[ref].validity
    # Normalized entity rows use only the corresponding reference permutation.
    a,b = before.batch['routing']['samples'][0],after.batch['routing']['samples'][0]
    for family in ('module','connector','resource','coordinate','connection'):
        refs = [unref(r) for r in b[family+'_refs']]
        order = [refs.index(ref) for ref in a[family+'_refs']]
        for suffix in ('_x','_validity'):
            torch.testing.assert_close(before.batch['inputs'][family+suffix],after.batch['inputs'][family+suffix][order],rtol=0.,atol=0.)


def test_frozen_normalizer_identity_and_empty_supervision_are_preserved():
    result = prepare()
    assert result.batch['routing']['normalizer_fingerprint'] == '890eca0b9f6d4deca2cac81d3a41bf535aa3d03ac16e0ff4b81cbc0ca163abf5'
    assert result.batch['supervision']['resource']['resource'].numel() == 0
    assert result.batch['supervision']['structural']['candidate'].numel() == 0


@pytest.mark.parametrize('missing', ['pose', 'twist', 'coordinate'])
def test_typed_state_cannot_bypass_required_live_measurement_gate(missing):
    a = api(); result = prepare(); graph = result.state.graph
    if missing == 'pose': graph = replace(graph, modules=(replace(graph.modules[0], pose_valid=False),)+graph.modules[1:])
    elif missing == 'twist': graph = replace(graph, modules=(replace(graph.modules[0], linear_velocity=None),)+graph.modules[1:])
    else:
        resource = graph.resources[0]
        coordinate = replace(resource.coordinates[0], velocity=None)
        graph = replace(graph, resources=(replace(resource, coordinates=(coordinate,)+resource.coordinates[1:]),)+graph.resources[1:])
    state = a.LivePhysicalState(result.state.stamp, graph)
    with pytest.raises(ValueError, match='required'):
        a.prepare_live_observation(state, context=context(), observation=c.ObservationBundle(), bridges=world_bridges(), **settings())


def test_bridge_does_not_mutate_snapshot_or_task_geometry():
    api(); raw = snapshot(8, rc=True); before = copy.deepcopy(raw); ctx = context()
    prepare(raw, ctx)
    assert raw == before and ctx == context()


def test_pure_bridge_import_has_no_isaac_ros_or_runtime_package_dependency():
    import subprocess
    script = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('rclpy','isaacsim','omni','pxr','smores_ep','mssr_expert'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from drl.IL.bc_v1.live_observation import encode_smores_snapshot, prepare_live_observation
assert callable(encode_smores_snapshot) and callable(prepare_live_observation)
'''
    result = subprocess.run([sys.executable,'-B','-c',script],capture_output=True,text=True,cwd=ROOT)
    assert result.returncode == 0, result.stderr
