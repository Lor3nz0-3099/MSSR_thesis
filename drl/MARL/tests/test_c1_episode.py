from dataclasses import replace
import pytest

from drl.MARL.c1_config import C1Settings
from drl.MARL.c1_episode import AssemblyProbeEvaluator
from drl.MARL.c1_structure import C1PrimitiveEvent
from drl.MARL.tests.test_c1_structure import frame
from drl.MARL.tests.test_c1_probe import move
from drl.MARL.c1_runner import make_course
from drl.MARL.phase1_reward import ActionOutcomeSignals


def evaluator(task='navigation',max_steps=1800):
    return AssemblyProbeEvaluator(task,make_course(task,{'navigation':7017,'stairs':12005,'gap':16022,'manipulation':20004}[task],'train'),
        tuple(f'm{i}' for i in range(8)),settings=C1Settings.load(),max_steps=max_steps)


def settle(e,base=None):
    base=base or frame();result=None
    for i in range(19):result=e.step(replace(base,time_s=i/30))
    assert result.assembly_valid_now and result.phase=='FUNCTIONAL_PROBE'
    return replace(base,time_s=18/30),result


def test_stability_window_resets_on_break_and_between_step_topology_churn():
    e=evaluator()
    for i in range(10):assert not e.step(frame(i/30)).assembly_valid_now
    assert not e.step(frame(10/30,edges=6)).assembly_valid_now
    for i in range(11,20):assert not e.step(frame(i/30)).assembly_valid_now
    assert not e.step(replace(frame(20/30),topology_changed=True)).assembly_valid_now
    assert not e.step(frame(21/30)).assembly_valid_now


def test_valid_assembly_without_probe_is_not_success_and_geometry_before_valid_is_ignored():
    e=evaluator()
    f=frame(edges=0)
    for i in range(20):
        result=e.step(replace(move(f,dx=i*.003),time_s=i/30))
        assert not result.success
        assert result.reward_breakdown['reward_task_probe_progress']==0.
    e=evaluator();_,result=settle(e)
    assert not result.success


def test_valid_novel_cycle_can_complete_navigation_probe_without_known_topology():
    e=evaluator();base,result=settle(e,frame(edges=8))
    for i in range(1,91):
        result=e.step(replace(move(base,dx=.25*i/90),time_s=base.time_s+i/30))
        if result.success:break
    assert result.success and result.terminated and not result.truncated
    record=e.episode_record()
    assert record['assembly_valid_timestamp'] is not None
    assert record['probe_success_timestamp'] is not None
    assert record['valid_connection_count']==8
    assert record['reward_breakdown']['reward_probe_success']==5.


def test_probe_invalidates_immediately_when_connection_breaks():
    e=evaluator();base,_=settle(e)
    result=e.step(replace(move(base,dx=.003),time_s=base.time_s+1/30,connections=base.connections[:-1]))
    assert not result.success and not result.assembly_valid_now
    assert result.phase=='ASSEMBLY_NOT_VALID'
    assert result.reward_breakdown['penalty_instability']<0.


def test_dock_undock_and_stability_oscillation_cannot_farm_positive_reward():
    e=evaluator();total_positive=0.
    for i in range(180):
        f=frame(i/30,edges=7 if (i//20)%2==0 else 0)
        r=e.step(f)
        total_positive+=sum(v for v in r.reward_breakdown.values() if v>0)
    assert total_positive<=3.+1e-9
    assert e.episode_record()['reward_breakdown']['reward_structural_stability']<=1.


def test_alignment_only_cannot_earn_positive_assembly_reward():
    e=evaluator()
    for i in range(40):
        r=e.step(replace(frame(i/30,edges=0),primitive_events=(C1PrimitiveEvent(str(i),'align_faces','succeeded','ALIGNED','m0'),)))
        assert not any(value>0 for value in r.reward_breakdown.values())


def test_conflicts_invalids_and_failed_docking_keep_existing_penalty_events():
    e=evaluator()
    r=e.step(replace(frame(edges=0),primitive_events=(C1PrimitiveEvent('x','dock','failed','DOCK_REJECTED','m2'),)),
        signals=ActionOutcomeSignals(conflict_owner_ids=frozenset(('m0',)),invalid_owner_ids=frozenset(('m1',))))
    assert r.owner_rewards['m0']<r.owner_rewards['m7']
    assert r.reward_breakdown['penalty_conflict']==pytest.approx(-1/8)
    assert r.reward_breakdown['penalty_invalid']==pytest.approx(-1/8)
    assert r.reward_breakdown['penalty_failed_docking']==pytest.approx(-1/8)
    r=e.step(replace(frame(1/30,edges=0),primitive_events=(C1PrimitiveEvent('x','dock','failed','DOCK_REJECTED','m2'),)))
    assert r.reward_breakdown['penalty_failed_docking']==0.


def test_nan_or_lost_module_terminates_and_timeout_is_truncated():
    e=evaluator();r=e.step(replace(frame(),modules=frame().modules[:-1]))
    assert r.terminated and not r.success
    e=evaluator(max_steps=2);e.step(frame(edges=0));r=e.step(frame(1/30,edges=0))
    assert r.truncated and not r.terminated and r.reason=='time_limit'


def test_observation_gap_does_not_count_as_continuous_stability():
    e=evaluator();e.step(frame());r=e.step(frame(1.))
    assert not r.assembly_valid_now and not r.success


def test_runtime_pathological_collision_invalidates_an_ongoing_probe():
    e=evaluator();base,_=settle(e)
    result=e.step(replace(base,time_s=base.time_s+1/30),
        signals=ActionOutcomeSignals(collision_owner_ids=frozenset(('m0',))))
    assert not result.assembly_valid_now
    assert result.reward_breakdown['penalty_collision']<0


def test_brief_dock_undock_inside_one_decision_is_logged_without_reward():
    e=evaluator()
    result=e.step(replace(frame(edges=0),topology_changed=True,connection_events=(('added','temp'),('removed','temp'))))
    assert result.reward_breakdown['reward_valid_docking']==0
    record=e.episode_record()
    assert record['docking_count']==1 and record['undocking_count']==1


def test_fast_connector_motion_cannot_hide_behind_a_stationary_module_root():
    f=frame()
    e=evaluator();base,_=settle(e,f)
    fast=replace(base,time_s=base.time_s+1/30,
        modules=(replace(base.modules[0],connectors=tuple((name,(point[0]+.02,*point[1:])) if name=='LEFT' else (name,point) for name,point in base.modules[0].connectors)),*base.modules[1:]))
    result=e.step(fast)
    assert not result.assembly_valid_now
    assert result.assessment.reason=='excessive_dynamics'


@pytest.mark.parametrize('task',['stairs','gap','manipulation'])
def test_each_task_can_complete_its_short_probe_with_a_novel_connected_graph(task):
    e=evaluator(task);base=frame(edges=8)
    dx=dy=dz=0.
    if task=='stairs':base=move(base,dx=.65,dz=.032);dx=.1;dz=.02
    elif task=='gap':base=move(base,dx=.59);dx=.04
    else:
        course=e.course;n=course.press_direction_world_xy;c=course.button_center_xyz_m
        front=(c[0]-n[0]*.02,c[1]-n[1]*.02,c[2])
        root_shift=(c[0]-n[0]*.3,c[1]-n[1]*.3)
        base=move(base,dx=root_shift[0],dy=root_shift[1])
        modules=[]
        for index,module in enumerate(base.modules):
            ports=tuple((name,(front[0]-n[0]*(.09 if index==0 and name=='LEFT' else .2),
                front[1]-n[1]*(.09 if index==0 and name=='LEFT' else .2),front[2])) for name,_ in module.connectors)
            modules.append(replace(module,connectors=ports))
        base=replace(base,modules=tuple(modules),button_depression_m=0.)
        dx,dy=n[0]*.05,n[1]*.05
    anchor,_=settle(e,base)
    result=None
    for i in range(1,91):
        result=e.step(replace(move(anchor,dx=dx*i/90,dy=dy*i/90,dz=dz*i/90),time_s=anchor.time_s+i/30))
        if result.terminated:break
    assert result.success,result
