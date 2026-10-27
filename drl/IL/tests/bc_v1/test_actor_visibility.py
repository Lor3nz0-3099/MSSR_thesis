"""Actor information projection, using real E1/E2/G2 (CPU synthetic only)."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path

import pytest

from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.bc_v1.features import transform_inputs, PhysicalFrameConfig
from drl.IL.bc_v1.feature_schema import extract_features
from drl.IL.bc_v1.policy_inputs import select_policy_inputs
from drl.IL.tests.bc_v1.g2_fixtures import fixture, tensorize, with_structural, ROOT
from drl.IL.tests.bc_v1.test_tensorization import assert_tree_equal


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.actor_visibility'), 'Missing actor visibility preprocessing contract'
    from drl.IL.bc_v1 import actor_visibility
    return actor_visibility


def policy():
    return api().load_visibility_policy(ROOT/'configs/bc_v1/actor_visibility_r1m.json')


def sample_for(task, static=(), dynamic=(), *, positions=(0., .7)):
    sample, kw=fixture(len(positions))
    graph=replace(sample.inputs.policy.graph,modules=tuple(replace(m,position=(x,0.,0.)) for m,x in zip(sample.inputs.policy.graph.modules,positions)))
    inputs=select_policy_inputs(graph,c.ObservationBundle(tuple(c.NumericFeature(k,tuple(v)) for k,v in dynamic)),
        c.TaskContext(task,tuple(c.NumericFeature(k,tuple(v)) for k,v in static)))
    view=extract_features(inputs,transform_inputs(inputs,PhysicalFrameConfig()),kw['spec'],kw['feature_schema'])
    return replace(sample,inputs=replace(sample.inputs,policy=inputs,features=view)),kw


def project(sample,kw,**kwargs):
    return api().project_actor_features(sample.inputs.features,sample.inputs.policy.context.task_id,
        policy(),kw['feature_schema'],**kwargs)


def names(view):return {r.source_name.split('.',1)[1] if r.source_name.startswith(('world.','map.','stage_local.','unframed.')) else r.source_name for r in view.environment}


def test_versioned_radius_contract_provenance_and_fail_closed_config():
    p=policy();assert p.document['schema_version']=='mssr.bc_v1.actor_visibility.v1'
    assert p.document['radius_m']==1.0 and p.document['threshold']=='<='
    assert p.document['scope']=='team_union_wireless'
    assert p.document['empirical_reference_mean_m']==0.47550612841561374
    bad={**p.document,'threshold':'<'}
    with pytest.raises(ValueError):api().VisibilityPolicy.from_dict(bad)
    assert api().VisibilityPolicy.from_dict({**p.document,'radius_m':2.}).fingerprint!=p.fingerprint


@pytest.mark.parametrize('x,visible',[(1.5,True),(1.7,True),(1.700001,False)])
def test_team_union_inclusive_boundary_and_full_absence(x,visible):
    sample,kw=sample_for('navigation',[
        ('world.environment_cones.0.center_xyz_m',(x,0,0)),('environment_cones.0.radius_m',(.2,)),('environment_cones.0.height_m',(.5,))])
    v=project(sample,kw)
    assert bool(v.environment)==visible and bool(v.relations)==visible
    out=tensorize(replace(sample,inputs=replace(sample.inputs,features=v)),kw)
    assert int(out['inputs']['environment_ptr'][-1])==(3 if visible else 0)
    assert int(out['inputs']['object_ptr'][-1])==int(visible)
    assert bool(out['inputs']['environment'])==visible
    assert out['inputs']['relation_geometry_index'].numel()>0 if visible else out['inputs']['relation_x'].numel()==0
    refs={r.ref for r in v.environment}
    assert all(set(r.geometry_sources)<=refs for r in v.relations)


@pytest.mark.parametrize('task,center,local,critic',[
    ('gap','world.gap.near_edge_x_m',{'gap.width_m':(.3,),'gap.bank_width_m':(.2,)},{'environment.clear_landing_length_m':(3,)}),
    ('stairs','world.stairs.first_riser_x_m',{'stairs.rise_m':(.1,),'stairs.tread_depth_m':(.2,)},{'stairs.step_count':(7,)}),
    ('navigation','world.environment_cones.0.center_xyz_m',{'cones.cone_radius_m':(.2,),'cones.cone_height_m':(.3,)},{'cones.corridor_width_m':(3,)}),
    ('button','world.button.center_xyz_m',{'button.face_size_tangent_z_m':(.08,.08)},{'button.plunger_depth_m':(.04,),'button.plunger_stroke_m':(.004,)}),
])
@pytest.mark.parametrize('near',[False,True])
def test_local_attributes_follow_physical_visibility_and_critic_fields_never_survive(task,center,local,critic,near):
    x=.8 if near else 5.
    value=(x,0,0) if center.endswith('xyz_m') else (x,)
    static=[(center,value),*local.items(),*critic.items()]
    sample,kw=sample_for(task,static)
    view=project(sample,kw)
    assert set(local)<=names(view) if near else not set(local)&names(view)
    assert not set(critic)&names(view)
    assert not view.environment if not near else view.environment


def test_current_button_geometry_gates_measured_depression_and_overrides_stale_nominal_visibility():
    for current,visible in [(5.,False),(.8,True)]:
        sample,kw=sample_for('button',[('world.button.center_xyz_m',(.8,0,0)),('button.face_size_tangent_z_m',(.1,.1))],
            [('world.button.current_center_xyz_m',(current,0,0)),('button.depression_m',(.003,))])
        v=project(sample,kw)
        assert ('button.depression_m' in names(v))==visible
        assert ('button.current_center_xyz_m' in names(v))==visible
        assert bool(v.environment)==visible


def test_explicit_assigned_goal_is_exempt_but_presence_alone_does_not_certify():
    sample,kw=sample_for('navigation',[('world.goal.position_xyz_m',(10,0,0))])
    assert not project(sample,kw).environment
    a=api();cert=a.certify_assigned_goal(sample.inputs.features,provider_identity='mission-manager.v1',
        source_contract_id='current-physical-goal.v1',evidence=('assigned goal id/current declaration',))
    view=project(sample,kw,assigned_goal=cert)
    assert 'goal.position_xyz_m' in names(view) and view.relations
    changed,_=sample_for('navigation',[('world.goal.position_xyz_m',(11,0,0))])
    with pytest.raises(ValueError,match='goal'):project(changed,kw,assigned_goal=cert)
    with pytest.raises(ValueError):a.certify_assigned_goal(sample.inputs.features,provider_identity='',source_contract_id='x',evidence=())


def test_task_frame_and_global_support_do_not_reach_actor_tensor_or_change_features():
    static=[('world.task_frame.origin_xyz_m',(10,2,0)),('world.task_frame.yaw_rad',(.4,)),
        ('world.environment.floor_height_m',(0,)),('world.environment.upper_deck_height_m',(1,)),
        ('world.environment.platform_bounds_xy_m',(-10,10,-10,10)),('environment.clear_landing_length_m',(3,))]
    full,kw=sample_for('navigation',static);plain,_=sample_for('navigation')
    before=full.inputs.features
    out=project(full,kw);assert before.environment and not out.environment
    x=tensorize(replace(full,inputs=replace(full.inputs,features=out)),kw)
    y=tensorize(replace(plain,inputs=replace(plain.inputs,features=project(plain,kw))),kw)
    assert_tree_equal(x['inputs'],y['inputs'])
    assert full.inputs.features==before


def test_compound_planes_keep_only_local_parts_and_no_whole_inventory_shortcut():
    sample,kw=sample_for('stairs',[('world.stairs.top_heights_m',(.1,4.)),('stairs.step_count',(2,)),('stairs.rise_m',(.1,))])
    v=project(sample,kw)
    planes=[r for r in v.environment if r.geometry_kind=='plane']
    assert len(planes)==1 and 'stairs.step_count' not in names(v)
    assert 'stairs.rise_m' in names(v)
    assert all(set(r.geometry_sources)<={e.ref for e in v.environment} for r in v.relations)


def test_no_robot_state_or_candidate_changes_and_no_target_or_id_embedding():
    sample,kw=sample_for('navigation',[('world.environment_cones.99.center_xyz_m',(8,0,0))])
    sample=with_structural(sample)
    v=project(sample,kw)
    for field in ('modules','connectors','resources','coordinates','connections'):assert getattr(v,field) is getattr(sample.inputs.features,field)
    changed=replace(sample,supervision=c.SupervisionBundle(),audit=replace(sample.audit,notes=('stage FSM configuration_goal target',)))
    a=tensorize(replace(sample,inputs=replace(sample.inputs,features=v)),kw)
    b=tensorize(replace(changed,inputs=replace(changed.inputs,features=project(changed,kw))),kw)
    assert_tree_equal(a['inputs'],b['inputs'])
    assert a['routing']['module_refs']==tuple(m.ref for m in sample.inputs.policy.graph.modules)
    assert a['inputs']['task_valid'].tolist()==[True]
    # A far fixture is indistinguishable from absent environment in H1's learned tensor path.
    empty,_=sample_for('navigation')
    z=tensorize(replace(empty,inputs=replace(empty.inputs,features=project(empty,kw))),kw)
    assert_tree_equal(a['inputs'],z['inputs'])
