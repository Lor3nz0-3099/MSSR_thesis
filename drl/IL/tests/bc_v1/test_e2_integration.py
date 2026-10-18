"""Final E2 integration gates; no training statistics or source-label inputs.

Real fixtures read fixed prefixes only. Adversarial metadata is injected before
D2 selection, while positive controls change permitted current physical state.
"""

from collections import Counter
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.bc_v1 import candidates, feature_schema as e2, features as e1
from drl.IL.bc_v1.policy_inputs import select_policy_inputs
from drl.IL.tests.bc_v1.test_candidates import catalog, graph, attach

ROOT = Path(__file__).resolve().parents[4]
FINGERPRINT = '43ab1b93c21b2645176e614ab1af7437f37163ec2b6047c0671d18a354b67d57'
MANIFEST = '31765272b6ecd7506c54b25e214b75d7b1b28691803ddfc286dafcce8b4ec562'
RESOURCE_ACTIONS = {'wheel_velocity', 'pan_velocity', 'tilt_velocity', 'pan_target',
                    'tilt_target', 'pan_delta', 'tilt_delta'}
STRUCTURAL_ACTIONS = {'align_faces', 'dock', 'undock', 'gravity_settle'}
EXPECTED = {
    # task, connection/environment/relation rows, ALIGN/DOCK pairs, invalid geometry
    'assembly': ('button', 0, 21, 32, 448, 0),
    'gap': ('gap', 7, 26, 32, 139, 2),
    'stairs': ('stairs', 7, 50, 32, 139, 6),
    'button': ('button', 7, 24, 32, 139, 0),
    'navigation': ('navigation', 7, 156, 32, 141, 0),
    'terminal-goal': ('navigation', 7, 9, 24, 138, 0),
}
FORBIDDEN = (
    'behavioral_stage', 'source_morphology', 'target_morphology', 'final_morphology',
    'configuration_goal', 'target_graph', 'graph_t_plus_1', 'G_t_plus_1',
    'future_state', 'expert_action', 'action_t', 'supervision_target',
    'supervision', 'module_resource_masks', 'loss_masks', 'reward', 'success',
    'outcome', 'training_split', 'provenance', 'executor_fsm', 'control_state',
    'nav2_path', 'nav2_route', 'nav2_waypoints', 'nav2_centerline',
)
FAMILIES = ('modules', 'connectors', 'resources', 'coordinates', 'connections',
            'environment', 'relations')


def schema():
    return e2.load_feature_schema(ROOT / 'configs/bc_v1/features.json')


def smores_spec():
    return c.load_robot_spec(ROOT / 'configs/bc_v1/smores_ep.robot.json')


def pipeline(g, observation, context, spec, config=None):
    inputs = select_policy_inputs(g, observation, context)
    geometry = e1.transform_inputs(inputs, config or e1.PhysicalFrameConfig())
    view = e2.extract_features(inputs, geometry, spec, schema())
    # Independent branch: neither FeatureView nor expert targets are arguments.
    table = candidates.enumerate_candidates(inputs.graph, spec, schema())
    return inputs, geometry, view, table


def numeric(view):
    return tuple(tuple(row.numeric for row in getattr(view, name)) for name in FAMILIES)


@pytest.fixture(scope='module')
def real_rows(tmp_path_factory):
    from drl.IL.tests.bc_v1.test_policy_inputs import real_policy_rows
    yield from real_policy_rows.__wrapped__(tmp_path_factory)


@pytest.mark.parametrize('case', EXPECTED)
def test_complete_pipeline_on_six_fixed_real_rows(real_rows, case):
    _, g, o, ctx, approved = real_rows[case]
    inputs, geometry, view, table = pipeline(g, o, ctx, smores_spec())
    task, edges, env, relations, pairs, invalid = EXPECTED[case]
    assert inputs == approved and inputs.context.task_id == task
    assert view.frame is geometry.frame and view.frame.axes_id == 'world'
    assert view.frame.valid and view.schema_fingerprint == FINGERPRINT
    assert tuple(len(getattr(view, name)) for name in FAMILIES) == (8, 32, 16, 32, edges, env, relations)
    assert len(table.resource_candidates) == 56
    assert {x.action.action_id for x in table.resource_candidates} == RESOURCE_ACTIONS
    counts = Counter(x.action.action_id for x in table.structural_candidates)
    assert counts == Counter({'align_faces': pairs, 'dock': pairs, 'undock': edges, 'gravity_settle': 1})
    assert sum(not all(r.validity[:r.fields[0].width]) for r in view.environment) == invalid
    assert all(all(r.validity) for r in view.relations)
    assert {r.module_ref for r in view.relations} == {m.ref for m in view.modules}
    assert all(len(r.numeric) == 30 for r in view.modules)
    assert all(r.output_frame == 'robot_centroid' for r in view.relations)
    assert all(not source[2].startswith('stage_local.') for r in view.relations for source in r.geometry_sources)
    if case in ('assembly', 'button'):
        button = [r for r in view.environment if r.object_type == 'button' and r.geometry_kind == 'point3']
        assert {r.source_scope for r in button} == {'c_static', 'o_dynamic'}
        assert len(button) == 2
        assert len({r.values[3] for r in view.relations if r.object_type == 'button'}) > 1
    if case in ('gap', 'stairs'):
        kind = 'gap_boundary' if case == 'gap' else 'stair_riser'
        assert all(r.geometry_kind == 'plane' for r in view.relations if r.object_type == kind)
        bad = [r for r in view.environment if not all(r.validity[:r.fields[0].width])]
        assert all(r.source_name.startswith('stage_local.') for r in bad)
        assert all(r.values[:r.fields[0].width] == (0.,)*r.fields[0].width for r in bad)
    if case == 'navigation':
        assert len(view.relations) == 32 < 8*156/4
        assert len([r for r in view.environment if r.object_type == 'cone' and r.geometry_kind == 'point3']) == 8
        for m in view.modules:
            assert len([r for r in view.relations if r.module_ref == m.ref and r.object_type == 'cone']) == 2
        assert not any(word in r.source_name for r in view.environment for word in ('nav2', 'route', 'waypoint', 'centerline'))
    if case == 'terminal-goal':
        assert any(r.object_type == 'goal' and r.geometry_kind == 'point3' for r in view.environment)
        assert any(r.object_type == 'collision_box' for r in view.environment)


@pytest.mark.parametrize('case', EXPECTED)
@pytest.mark.parametrize('field', FORBIDDEN)
def test_end_to_end_numeric_features_and_candidates_ignore_forbidden_metadata(real_rows, case, field):
    _, g, o, ctx, _ = real_rows[case]
    before = pipeline(g, o, ctx, smores_spec())
    g, o, ctx = copy.deepcopy((g, o, ctx))
    for obj in (g, *g.modules, *g.connectors, *g.resources, *g.connections, o, ctx):
        object.__setattr__(obj, field, {'forbidden_future_hint': 123456789})
    # Open numeric slots must not smuggle the same metadata into learned columns.
    object.__setattr__(o, 'features', o.features + (c.NumericFeature(field, (999.,)),))
    object.__setattr__(ctx, 'static_geometry', ctx.static_geometry + (c.NumericFeature(field, (-999.,)),))
    # Check direct candidate input as well as the strict selector composition.
    assert candidates.enumerate_candidates(g, smores_spec(), schema()) == before[3]
    after = pipeline(g, o, ctx, smores_spec())
    assert after[0] == before[0]
    assert numeric(after[2]) == numeric(before[2]) and after[3] == before[3]


@pytest.mark.parametrize('task', ['button', 'navigation'])
def test_forbidden_raw_metadata_perturbation_survives_caller_composition(task):
    from drl.IL.tests.bc_v1.test_policy_inputs import raw_view_fixture, from_raw
    raw, metadata = raw_view_fixture(task)
    approved = from_raw(raw, metadata)
    def encode(p):
        return pipeline(p.graph, p.observation, p.context, smores_spec())
    baseline = encode(approved)
    changed, source = copy.deepcopy((raw, metadata))
    for field in FORBIDDEN:
        changed[field] = {'forbidden': 999.}
        changed['graph_t'][field] = {'forbidden': -999.}
        source['course'][field] = {'forbidden': 123.}
    source['course']['navigation'] = {'waypoints_xyyaw': [[999., -999., 3.]],
                                      'route': [[-888., 888.]], 'path': [[-777., 777.]]}
    source['course'].setdefault('physical_track', {})['centerline_xy_m'] = [[-999., 999.]]
    after = encode(from_raw(changed, source))
    assert after[0] == baseline[0]
    assert numeric(after[2]) == numeric(baseline[2]) and after[3] == baseline[3]


@pytest.mark.parametrize('case,scope,suffix', [
    ('navigation', 'c_static', 'environment_cones.0.center_xyz_m'),
    ('button', 'o_dynamic', 'button.current_center_xyz_m'),
])
def test_current_environment_movement_changes_features_and_relations(real_rows, case, scope, suffix):
    _, g, o, ctx, _ = real_rows[case]
    baseline = pipeline(g, o, ctx, smores_spec())
    items = ctx.static_geometry if scope == 'c_static' else o.features
    source = next(f for f in items if f.name.endswith(suffix))
    moved = replace(source, values=(source.values[0]+.07, *source.values[1:]))
    items = tuple(moved if f is source else f for f in items)
    if scope == 'c_static': ctx = replace(ctx, static_geometry=items)
    else: o = replace(o, features=items)
    after = pipeline(g, o, ctx, smores_spec())
    assert after[2].environment != baseline[2].environment
    assert after[2].relations != baseline[2].relations
    assert after[3] == baseline[3]  # Environment is not candidate availability.


def test_current_module_pose_changes_module_rows_and_relations(real_rows):
    _, g, o, ctx, _ = real_rows['button']
    before = pipeline(g, o, ctx, smores_spec())
    m = g.modules[0]
    moved = replace(m, position=(m.position[0]+.17, *m.position[1:]))
    after = pipeline(replace(g, modules=(moved, *g.modules[1:])), o, ctx, smores_spec())
    assert after[2].modules != before[2].modules
    assert after[2].relations != before[2].relations
    assert after[3] == before[3]  # No distance-only candidate pruning.


def test_current_attachment_changes_connection_features_and_dock_undock_choices():
    g = graph(2, 2); spec = catalog(2)
    before = pipeline(g, c.ObservationBundle(), c.TaskContext(), spec)
    after = pipeline(attach(g, 0, 2), c.ObservationBundle(), c.TaskContext(), spec)
    assert not before[2].connections and len(after[2].connections) == 1
    assert Counter(r.action.action_id for r in before[3].structural_candidates) == {'align_faces': 4, 'dock': 4, 'gravity_settle': 1}
    assert Counter(r.action.action_id for r in after[3].structural_candidates) == {'align_faces': 1, 'dock': 1, 'undock': 1, 'gravity_settle': 1}


def test_enabled_actions_remain_available_with_zero_positive_supervision():
    g = graph(); spec = catalog()
    historical_labels = ['pace'] * 100
    before = candidates.enumerate_candidates(g, spec, schema())
    assert historical_labels.count('aim') == historical_labels.count('gravity_settle') == 0
    assert {r.mode.mode_id for r in before.resource_candidates} == {'pace', 'aim'}
    assert any(r.action.action_id == 'gravity_settle' for r in before.structural_candidates)
    historical_labels.clear()
    assert candidates.enumerate_candidates(g, spec, schema()) == before
    assert not hasattr(before, 'label_frequency')


def test_all_eleven_smores_families_are_available_without_any_labels(real_rows):
    spec = smores_spec()
    assert {a.action_id for a in spec.actions if a.enabled} == RESOURCE_ACTIONS | STRUCTURAL_ACTIONS
    families = set()
    for case in ('assembly', 'navigation'):
        table = candidates.enumerate_candidates(real_rows[case][1], spec, schema())
        families.update(x.action.action_id for x in (*table.resource_candidates, *table.structural_candidates))
    assert families == RESOURCE_ACTIONS | STRUCTURAL_ACTIONS
    assert not families.intersection({'REACH', 'ALIGN', 'CLOCKING', 'APPROACH', 'WAIT', 'NO_DISPATCH', 'no_dispatch'})


def test_common_rigid_upstream_frame_preserves_payloads_and_physical_distances():
    g = graph(2, 1)
    g = replace(g, modules=tuple(replace(m, position=(10.+2*i, 4., 3.),
        linear_velocity=(1., 2., 3.), angular_velocity=(4., 5., 6.)) for i,m in enumerate(g.modules)))
    ctx = c.TaskContext('navigation', (
        c.NumericFeature('world.environment_cones.0.center_xyz_m', (15., 7., 2.)),
        c.NumericFeature('world.cones.1.center_xy_m', (14., 6.)),))
    before = pipeline(g, c.ObservationBundle(), ctx, catalog())
    # Same physical sample expressed upstream in map = R*world + translation.
    rotation = (0., -1., 0., 1., 0., 0., 0., 0., 1.)
    translation = (100., 200., 30.)
    def vector(p): return (-p[1], p[0], p[2])
    def point(p): return tuple(a+b for a,b in zip(vector(p), translation))
    moved = replace(g, modules=tuple(replace(m, position=point(m.position),
        orientation_xyzw=(0., 0., 2**-.5, 2**-.5),
        linear_velocity=vector(m.linear_velocity), angular_velocity=vector(m.angular_velocity)) for m in g.modules))
    config = e1.PhysicalFrameConfig(graph_frame='map', axes_id='world',
        rotation_source_from_axes=rotation,
        transforms=(e1.SourceFrameTransform('world', translation, rotation),))
    after = pipeline(moved, c.ObservationBundle(), ctx, catalog(), config)
    for a,b in zip(numeric(before[2]), numeric(after[2])):
        assert len(a) == len(b)
        for x,y in zip(a,b): assert y == pytest.approx(x, abs=1e-12)
    assert before[3] == after[3]
    assert {r.geometry_kind for r in after[2].environment} == {'point3', 'line'}
    assert {r.geometry_kind for r in after[2].relations} == {'point3', 'line'}


def test_final_pipeline_does_not_retransform_e1_or_use_module_ids_as_root(monkeypatch):
    from drl.IL.tests.bc_v1.test_feature_schema import renamed_modules
    g = graph(); ctx = c.TaskContext('button', (c.NumericFeature('world.button.center_xyz_m', (4.,3.,0.)),))
    inputs, geometry, before, table = pipeline(g, c.ObservationBundle(), ctx, catalog())
    after = pipeline(renamed_modules(g), c.ObservationBundle(), ctx, catalog())
    assert numeric(after[2]) == numeric(before)
    assert geometry.frame.origin == (.5, 0., 0.)
    assert tuple(m.position for m in geometry.modules) == ((-.5,0.,0.),(.5,0.,0.))
    def forbidden(*args, **kwargs): raise AssertionError('E2 attempted a second transform')
    for name in ('physical_frame', 'transform_inputs', 'transform_point', 'transform_vector', 'transform_orientation', 'transform_twist'):
        monkeypatch.setattr(e1, name, forbidden)
    assert e2.extract_features(inputs, geometry, catalog(), schema()) == before
    assert candidates.enumerate_candidates(inputs.graph, catalog(), schema()) == table


def test_complete_pipeline_quaternion_sign_invariance():
    g = graph(); ctx = c.TaskContext('button', (c.NumericFeature('world.button.center_xyz_m', (4.,3.,0.)),))
    negative = replace(g, modules=tuple(replace(m, orientation_xyzw=tuple(-v for v in m.orientation_xyzw)) for m in g.modules))
    a = pipeline(g, c.ObservationBundle(), ctx, catalog())
    b = pipeline(negative, c.ObservationBundle(), ctx, catalog())
    assert numeric(a[2]) == numeric(b[2]) and a[3] == b[3]


def test_generic_pipeline_import_isolation_during_actual_execution():
    script = '''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'rclpy','rospy','isaacsim','isaaclab','omni','torch','smores_ep','mssr_expert','freebot','freebot_docking'} or fullname in {'drl.IL.bc_v1.smores_adapter','drl.IL.bc_v1.task_context','drl.IL.bc_v1.canonical_reader','drl.IL.preprocessing','drl.IL.trainer'}:
   raise ImportError('isolation: '+fullname)
sys.meta_path.insert(0,Block())
from drl.IL.bc_v1 import bc_v1_contract as c,features as e1,feature_schema as e2,candidates
from drl.IL.bc_v1.policy_inputs import select_policy_inputs
port=c.ConnectorSpec('socket','fixed_port')
res=c.ResourceSpec('motor',modes=(c.ResourceModeSpec('rate',1,('rad/s',)),))
spec=c.RobotSpec('mssr.bc_v1.robot_spec.v1','kit',(c.ModuleTypeSpec('unit',(port,),(res,)),),
 (c.ActionSpec('rate','continuous','resource'),c.ActionSpec('gravity_settle','skill','modules')))
m=c.ModuleRef('arbitrary')
g=c.StateGraph((c.ModuleState(m,'unit',(2.,0.,0.),(0.,0.,0.,1.),pose_valid=True),),
 (c.ConnectorState(c.ConnectorRef(m,'socket'),'fixed_port'),),(c.ResourceState(c.ResourceRef(m,'motor')),))
p=select_policy_inputs(g,c.ObservationBundle(),c.TaskContext('button',(c.NumericFeature('world.button.center_xyz_m',(3.,0.,0.)),)))
s=e2.load_feature_schema('configs/bc_v1/features.json')
v=e2.extract_features(p,e1.transform_inputs(p,e1.PhysicalFrameConfig()),spec,s)
t=candidates.enumerate_candidates(p.graph,spec,s)
assert len(v.modules)==len(v.environment)==len(v.relations)==1
assert len(t.resource_candidates)==len(t.structural_candidates)==1
assert not any(x in sys.modules for x in ('torch','rclpy','isaaclab','drl.IL.bc_v1.smores_adapter'))
'''
    result = subprocess.run([sys.executable, '-B', '-c', script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_frozen_schema_and_canonical_manifest():
    assert schema()['input_contract'] == 'mssr.bc_v1.policy_inputs.v1'
    assert schema()['schema_version'] == 'mssr.bc_v1.features.v1'
    assert schema()['geometry_contract'] == 'FrameGeometryView'
    assert e2.feature_schema_fingerprint(schema()) == FINGERPRINT
    path = ROOT / 'datasets/expert_v1_canonical_v6/manifest.json'
    if not path.is_file(): pytest.skip('Canonical v6 unavailable')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == MANIFEST
