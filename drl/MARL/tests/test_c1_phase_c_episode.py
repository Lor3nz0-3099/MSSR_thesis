from dataclasses import replace
from copy import deepcopy
import pytest

from drl.MARL.c1_config import C1Settings
from drl.MARL.c1_episode import AssemblyProbeEvaluator
from drl.MARL.c1_guided_feedback import load_target
from drl.MARL.c1_runner import make_course
from drl.MARL.tests.test_c1_structure import frame
from drl.MARL.tests.test_c1_probe import move
from drl.MARL.tests.test_c1_guided_success import measured_target


def settings(mode='guided', *, monitor=True):
    s=C1Settings.load(f'configs/marl/c1_{mode}_phase_c_v1.json')
    values=deepcopy(s.values);values['stagnation']['enabled']=monitor
    return replace(s,values=values)


def evaluator(mode='guided', *, monitor=True):
    return AssemblyProbeEvaluator('navigation',make_course('navigation',7017,'train'),
        tuple(f'm{i}' for i in range(8)),settings=settings(mode,monitor=monitor),max_steps=3600,
        target_graph=load_target('navigation') if mode=='guided' else None)


def guided_frame(t=0.,folded=True):
    measured,_,_=measured_target(load_target('navigation'),folded=folded,owners=tuple(f'm{i}' for i in range(8)))
    modules=[]
    for physical,fold in zip(frame().modules,measured.modules):
        angles=dict(fold.joints)
        modules.append(replace(physical,joints=tuple((k,angles.get(k,v)) for k,v in physical.joints)))
    return replace(frame(t),modules=tuple(modules),connections=measured.connections)


def test_seven_wrong_connections_cannot_start_guided_probe_or_earn_generic_bonuses():
    e=evaluator()
    for i in range(25):r=e.step(frame(i/30))
    assert not r.assembly_valid_now and not r.success
    assert not r.target_graph_complete
    assert r.reward_breakdown['reward_structural_progress']==0.
    assert r.reward_breakdown['reward_valid_docking']==0.


def test_target_topology_without_fold_never_starts_probe():
    e=evaluator()
    for i in range(25):r=e.step(guided_frame(i/30,folded=False))
    assert r.target_graph_complete and not r.fold_complete
    assert not r.assembly_stable and r.probe is None and not r.episode_success


def test_all_guided_gates_are_required_for_success_and_reported_separately():
    e=evaluator();base=guided_frame()
    for i in range(19):r=e.step(replace(base,time_s=i/30))
    assert r.target_graph_complete and r.fold_complete and r.assembly_stable
    assert not r.functional_probe_success and not r.episode_success
    for i in range(1,181):
        r=e.step(replace(move(base,dx=i*.002),time_s=(18+i)/30))
        if r.success:break
    assert r.episode_success and r.functional_probe_success and r.terminated and not r.truncated
    record=e.episode_record()
    assert record['episode_success'] and record['target_graph_complete'] and record['fold_complete']


def test_discovery_accepts_novel_functional_graph_and_never_loads_target():
    e=evaluator('discovery');base=frame()
    for i in range(19):r=e.step(replace(base,time_s=i/30))
    assert r.assembly_stable and e.target_progress is None
    for i in range(1,181):
        r=e.step(replace(move(base,dx=i*.002),time_s=(18+i)/30))
        if r.success:break
    assert r.success and r.episode_success
    assert r.reward_breakdown['reward_target_graph_progress']==0.


def test_timeout_120_seconds_truncates_without_stagnation_failure():
    e=evaluator(monitor=False)
    for i in range(1,3601):r=e.step(frame(i/30,edges=0))
    assert r.truncated and not r.terminated and r.reason=='time_limit'


def test_global_stagnation_terminates_with_reason_after_real_opportunities():
    e=evaluator()
    for i in range(1,901):
        r=e.step(frame(i/30,edges=0),decision_opportunity=i%15==0)
        if r.terminated:break
    assert r.terminated and not r.truncated and r.reason=='stagnation'
    assert not r.success and r.stagnation['decision_opportunities']>=3


def test_redundant_joint_cost_reaches_only_responsible_owner_and_team_mean():
    e=evaluator('discovery');f=frame(edges=0)
    goal=lambda:dict(primitive='set_tilt',module_ids=('m0',),owner_module_id='m0',arguments=(('angle_rad',0.),))
    e.step(f,action_goals={'first':goal()})
    r=e.step(replace(f,time_s=1/30),action_goals={'first':goal(),'second':goal()})
    assert r.reward_breakdown['penalty_redundant_action']==pytest.approx(-.1/8)
    assert r.owner_rewards['m0']-r.owner_rewards['m1']==pytest.approx(-.1)
    assert sum(r.owner_rewards.values())/8==pytest.approx(r.team_reward)


def test_transient_invalid_connector_constraint_is_logged_without_target_shaper_crash():
    e=evaluator();f=guided_frame()
    bad=replace(f.connections[0],connector_a='INVALID')
    goal=dict(primitive='align_faces',owner_module_id='m0',module_ids=('m0','m1'),connectors=(('m0','TOP'),('m1','BOTTOM')))
    r=e.step(replace(f,connections=(bad,*f.connections[1:])),action_goals={'g':goal},active_goals={'g':goal})
    assert not r.success and not r.target_graph_complete
    assert r.assessment.reason=='invalid_docking_constraint_or_connector_ownership'
    assert r.reward_breakdown['penalty_instability']<0


def test_loss_of_folding_resets_stability_and_probe_before_recovery():
    e=evaluator()
    for i in range(19):r=e.step(guided_frame(i/30))
    assert r.assembly_stable
    r=e.step(guided_frame(19/30,folded=False))
    assert not r.assembly_stable and r.probe is None and not r.fold_complete
    r=e.step(guided_frame(20/30))
    assert not r.assembly_stable and not r.success


def test_stagnation_failure_preserves_independent_achieved_assembly_flags():
    e=evaluator()
    for i in range(29*30):
        r=e.step(guided_frame(i/30),decision_opportunity=True)
        if r.terminated:break
    assert r.reason=='stagnation' and r.target_graph_complete and r.fold_complete
    assert r.assembly_stable and not r.functional_probe_success and not r.episode_success


def test_time_limit_retains_target_potential_for_smdp_bootstrap():
    e=evaluator(monitor=False);e.max_steps=1
    r=e.step(guided_frame())
    assert r.truncated and not r.terminated
    target_sum=sum(v for k,v in r.reward_breakdown.items() if k.startswith('reward_target_'))
    assert target_sum==pytest.approx(6*.99)


def test_pan_correction_outside_executor_tolerance_is_not_already_reached():
    e=evaluator('discovery');f=frame(edges=0)
    first=f.modules[0];joints=dict(first.joints);joints['pan.position_rad']=.04
    f=replace(f,modules=(replace(first,joints=tuple(joints.items())),*f.modules[1:]))
    goal=dict(primitive='set_pan',module_ids=('m0',),owner_module_id='m0',arguments=(('angle_rad',0.),))
    e.step(f,action_goals={'a':goal})
    r=e.step(replace(f,time_s=1/30),action_goals={'a':goal,'b':goal})
    assert r.reward_breakdown['penalty_redundant_action']==0.


def test_owner_cost_survives_real_smdp_accumulation_with_team_mean():
    import torch
    from drl.MARL.c1_smdp import DiscountedOptionAccumulator
    e=evaluator('discovery');f=frame(edges=0)
    goal=dict(primitive='set_tilt',module_ids=('m0',),owner_module_id='m0',arguments=(('angle_rad',0.),))
    acc=DiscountedOptionAccumulator(.99)
    for i,goals in enumerate(({'a':goal},{'a':goal,'b':goal})):
        r=e.step(replace(f,time_s=i/30),action_goals=goals)
        owners={k:torch.tensor(v,dtype=torch.float64) for k,v in r.owner_rewards.items()}
        acc.add(team_reward=torch.tensor(r.team_reward,dtype=torch.float64),owner_rewards=owners,mean_field_rewards=owners)
    assert (acc.owner_rewards['m0']-acc.owner_rewards['m1']).item()==pytest.approx(-.099)
    assert acc.team_reward.item()==pytest.approx(sum(v.item() for v in acc.owner_rewards.values())/8)
