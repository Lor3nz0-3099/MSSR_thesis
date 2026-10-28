"""Unit contracts only; these tests do not claim physical calibration evidence."""
from types import SimpleNamespace
from pathlib import Path
import json
import pytest


def test_calibration_cli_rejects_nontrain_before_runtime():
    from drl.MARL.c1_runner import parse_args
    with pytest.raises(SystemExit):
        parse_args(['calibrate', '--output', '/tmp/c1-unit', '--split', 'validation'])


def test_launch_uses_domain_course_seed_and_240hz(tmp_path):
    from drl.MARL.c1_runner import runtime_command
    for domain, flag in [('navigation', 'phase1_marl_seed'), ('stairs', 'phase1_stairs_seed'), ('gap', 'phase1_gap_seed'), ('manipulation', 'phase1_manipulation_seed')]:
        args = SimpleNamespace(repository_root=tmp_path, gui=False)
        command = runtime_command(args, tmp_path/'run', domain, 7000, 'train')
        assert 'physics_hz:=240' in command
        assert f'{flag}:=7000' in command


def test_transport_reuses_native_command_id_and_physically_completes_empty(tmp_path):
    from drl.MARL.c1_runner import LockstepTransport
    status = dict(session='unit', epoch='native-epoch', armed=True, failure=None, rejections=[], last_completed=None)
    (tmp_path/'bc_status.json').write_text(json.dumps(status))
    expected = []
    sync = SimpleNamespace(expect_action=lambda env, command: expected.append((env, command)))
    transport = LockstepTransport(tmp_path, 'unit', sync, step_ticks=8)
    transport.stage((), state_stamp=12.)
    action = transport.step(0)
    packet = json.loads((tmp_path/'bc_request.json').read_text())
    assert packet['command_id'] == 'bc:unit:1'
    assert expected == [(0, packet['command_id'])]
    assert packet['state_stamp'] == 12.
    assert packet['step_ticks'] == 8
    assert packet['continuous'] == packet['goals'] == []
    status['last_completed'] = dict(command_id=packet['command_id'], outcome='empty')
    (tmp_path/'bc_status.json').write_text(json.dumps(status))
    assert transport.completed_outcome(action).code == 'EMPTY'


def test_transport_uses_decoded_commands_without_resampling(tmp_path):
    from drl.MARL.c1_runner import LockstepTransport
    (tmp_path/'bc_status.json').write_text(json.dumps(dict(session='unit', epoch='e', failure=None)))
    t = LockstepTransport(tmp_path, 'unit', SimpleNamespace(expect_action=lambda *a:None), step_ticks=8)
    decoded = SimpleNamespace(command=dict(command='wheel_velocity', module_ids=['m0'], parameters=dict(left_rad_s=.2, right_rad_s=.3)))
    t.stage((decoded,), state_stamp=1.)
    t.step(0)
    packet = json.loads((tmp_path/'bc_request.json').read_text())
    assert packet['continuous'][0]['payload'] == decoded.command
    assert packet['continuous'][0]['key'] == ['m0', 'locomotion']


def test_transition_scheduler_balances_transitions_not_episodes():
    from drl.MARL.c1_runner import TransitionMix
    scheduler = TransitionMix()
    counts = {name:0 for name in ('navigation', 'stairs', 'gap', 'manipulation')}
    for _ in range(100):
        domain = scheduler.next_domain()
        counts[domain] += 1
        scheduler.record(domain, 1)
    assert counts == dict(navigation=70, stairs=10, gap=10, manipulation=10)


def test_raw_frame_adapter_rejects_missing_z_and_wrong_intended_connections():
    from drl.MARL.c1_runner import captured_observation
    with pytest.raises(ValueError, match='XYZ'):
        captured_observation('gap', dict(state=dict(modules=[dict(module_id='m0', position=[0,0])])), ())


def test_new_stochastic_heads_init_without_mutating_bc_checkpoint():
    from drl.MARL.c1_runner import STOCHASTIC_LOG_STD_INIT
    assert STOCHASTIC_LOG_STD_INIT == -2.


def test_course_adapter_reads_publisher_state_course_and_rejects_wrong_pairs():
    from drl.MARL.c1_runner import captured_observation
    modules = [dict(module_id=f'm{i}', position=[0,0,.1], connectors=[dict(connector_id='BOTTOM',position_world=[0,0,0],outward_normal_world=[0,0,-1])]) for i in range(8)]
    raw = dict(state=dict(modules=modules,course=dict(button=dict(depression_m=.0035))),graph=dict(edges=[]))
    obs = captured_observation('manipulation', raw, frozenset({(('m0','TOP'),('m1','BOTTOM'))}),end_effector_id='m7')
    assert obs.button_depression_m == .0035
    assert obs.connection_count == 0


def test_runtime_context_preserves_decision_graphs_through_post_state():
    from drl.MARL.c1_runner import C1RuntimeContext
    assert callable(C1RuntimeContext.current_action_contexts)


def test_thin_calibration_builder_has_no_optimizer_or_memory():
    from drl.MARL.c1_runner import InferenceDecisionBuilder
    b = InferenceDecisionBuilder(actor=object(), critic=object(), owners=('m0',),device='cpu')
    assert not hasattr(b,'memories')
    assert not hasattr(b,'actor_optimizer')


def test_completed_native_rejection_preserves_actual_code_and_owner(tmp_path):
    from drl.MARL.c1_runner import LockstepTransport
    status = dict(session='s',epoch='e',failure=None,rejections=[])
    (tmp_path/'bc_status.json').write_text(json.dumps(status))
    transport = LockstepTransport(tmp_path,'s',SimpleNamespace(expect_action=lambda *a:None),step_ticks=8)
    transport.stage((),state_stamp=1.)
    action = transport.step(0)
    status.update(last_completed=dict(command_id=action.command_id,outcome='rejected'),
                  rejections=[dict(code='INVALID_GOAL',owner_ids=['m0'])])
    (tmp_path/'bc_status.json').write_text(json.dumps(status))
    outcome = transport.completed_outcome(action)
    assert not outcome.accepted
    assert outcome.code == 'INVALID_GOAL'
    assert outcome.owner_ids == ('m0',)


def test_runner_entrypoint_exists_and_writes_no_learning_calibration_provenance():
    from drl.MARL.c1_runner import main, calibration_policy
    assert callable(main)
    assert calibration_policy(0,16) == 'bc_v4_stochastic'
    assert 'reference' in calibration_policy(14,16)
    assert 'perturbation' in calibration_policy(15,16)


def test_real_bc_model_and_appo_learner_construct_without_physics(tmp_path):
    """Architecture-only check; fixture is synthetic and never calibration data."""
    import torch
    from pathlib import Path
    from drl.MARL.c1_runner import build_observation_bridge,build_models,create_learner,parse_args
    torch.set_num_threads(2)
    args=parse_args(['smoke','--output',str(tmp_path),'--reward-config',str(tmp_path/'unused.json')])
    bridge=build_observation_bridge(args.repository_root)
    actor,critic=build_models(args,bridge)
    learner,provider=create_learner(args,bridge,actor,critic,tuple(f'module-{i}' for i in range(8)))
    assert learner.shared_actor is actor
    assert learner.altruistic_alpha == .5
    assert len(learner.memories['module-0']) == 0
    assert not actor.training
    assert learner.altruistic_owner_critic is not learner.altruistic_mean_field_critic


def test_reference_decision_reuses_bc_means_and_masks_structural_candidates(tmp_path):
    """Synthetic current frame checks model plumbing only, without collection."""
    import torch
    from drl.IL.bc_v1 import bc_v1_contract as c
    from drl.IL.tests.bc_v1.test_live_observation import snapshot
    from drl.MARL.c1_runner import (build_observation_bridge,build_models,parse_args,
        CalibrationDecisionBuilder)
    args=parse_args(['calibrate','--output',str(tmp_path)])
    bridge=build_observation_bridge(args.repository_root)
    actor,critic=build_models(args,bridge)
    prepared=bridge.prepare_snapshot(snapshot(n=8,rc=True),task_context=c.TaskContext('navigation'))
    owners=tuple(ref.module_id for ref in prepared.routing_snapshot['module_refs'])
    builder=CalibrationDecisionBuilder(actor=actor,critic=critic,owners=owners,device='cpu')
    builder.policy='bc_v4_deterministic_reference'
    route=prepared.routing_snapshot
    mask=torch.tensor([False]*len(route['resource_candidates'])+[True]*len(route['structural_bindings']))
    kwargs=dict(actor_graph=prepared.actor_graph,critic_graph=prepared.privileged_graph,routing_snapshot=route,hard_mask=mask)
    first=builder.build_online_rollout_decision(**kwargs)
    second=builder.build_online_rollout_decision(**kwargs)
    assert torch.equal(first.action.presence,second.action.presence)
    assert not first.action.presence[mask].any()
    assert all(torch.equal(first.action.resource_numeric[key],second.action.resource_numeric[key]) for key in first.action.resource_numeric)


def test_calibration_manifest_policy_is_fixed_for_the_whole_episode():
    from drl.MARL.c1_runner import calibration_collection_label
    assert calibration_collection_label('calibrate') == 'bc_v4_stochastic_reference_perturbation'
    assert calibration_collection_label('smoke') == 'bc_v4_stochastic'


def test_task_context_exposes_explicit_physical_goal_and_no_route(tmp_path):
    from drl.MARL.c1_runner import physical_task_view,parse_args
    args=parse_args(['calibrate','--output',str(tmp_path)])
    context,observation=physical_task_view('navigation',
        dict(frame_id='world',goal=dict(center_xyz_m=[1,2,.1]),route=[[9,9]],centerline=[[8,8]]),
        json.loads((args.repository_root/'configs/bc_v1/task_context.json').read_text()))
    names=[feature.name for feature in context.static_geometry]
    assert 'world.goal.position_xyz_m' in names
    assert all('route' not in name and 'centerline' not in name for name in names)


def test_launch_selects_c1_courses_and_forwards_split(tmp_path):
    from drl.MARL.c1_runner import runtime_command
    args=SimpleNamespace(repository_root=tmp_path,gui=False)
    for domain,prefix in [('navigation','phase1_marl'),('stairs','phase1_stairs'),('gap','phase1_gap'),('manipulation','phase1_manipulation')]:
        command=runtime_command(args,tmp_path/domain,domain,1234,'validation')
        assert f'{prefix}_course:=true' in command
        assert f'{prefix}_seed:=1234' in command
        assert f'{prefix}_split:=validation' in command


def test_real_models_record_and_update_two_unit_frames_without_calibration_rows(tmp_path):
    """UNIT ONLY: fabricated task rewards are never empirical calibration."""
    import torch
    from drl.IL.bc_v1 import bc_v1_contract as c
    from drl.IL.tests.bc_v1.test_live_observation import snapshot
    from drl.MARL.c1_runner import (build_observation_bridge,build_models,parse_args,
        create_learner,record_learning_step,update_learner)
    args=parse_args(['smoke','--output',str(tmp_path),'--reward-config',str(tmp_path/'unit-only.json')])
    args.reward_config.write_text('{"unit_only":true}')
    bridge=build_observation_bridge(args.repository_root)
    actor,critic=build_models(args,bridge)
    raw=snapshot(n=8,rc=True)
    prepared=bridge.prepare_snapshot(raw,task_context=c.TaskContext('navigation'))
    route=prepared.routing_snapshot
    owners=tuple(ref.module_id for ref in route['module_refs'])
    learner,provider=create_learner(args,bridge,actor,critic,owners)
    mask=torch.tensor([False]*len(route['resource_candidates'])+[True]*len(route['structural_bindings']))
    kwargs=dict(actor_graph=prepared.actor_graph,critic_graph=prepared.privileged_graph,routing_snapshot=route,hard_mask=mask)
    torch.manual_seed(23)
    gap=bridge.prepare_snapshot(raw,task_context=c.TaskContext('gap'))
    for index in range(3):
        current_kwargs=kwargs if index<2 else dict(kwargs,actor_graph=gap.actor_graph,
            critic_graph=gap.privileged_graph,routing_snapshot=gap.routing_snapshot)
        decision=learner.build_online_rollout_decision(**current_kwargs)
        context=SimpleNamespace(routing_snapshot=route)
        transition=SimpleNamespace(previous_state=raw,evaluation=SimpleNamespace(terminated=index==2,truncated=index==1))
        step=SimpleNamespace(transitions=(transition,),contexts=(context,),decisions=(decision,),
            env_actions={owner:torch.zeros((1,1)) for owner in owners},next_critic_graphs=(prepared.privileged_graph,))
        reward=SimpleNamespace(team=.2+index,owners={owner:.2+index+i*.1 for i,owner in enumerate(owners)})
        record_learning_step(args,learner,provider,step,SimpleNamespace(),reward)
    assert len(learner._structured_rollout_sidecar)==3
    assert len(learner._altruistic_rollout_sidecar)==3
    metrics=update_learner(args,learner,3,1)
    assert metrics['actor_optimizer_steps']==1
    assert metrics['owner_critic_optimizer_steps']==1
    assert metrics['mean_field_critic_optimizer_steps']==1
    assert not (tmp_path/'transitions.jsonl').exists()


def test_layout_limit_keeps_default_pilot_and_selects_one_per_domain(tmp_path):
    from drl.MARL.c1_runner import parse_args,selected_layouts
    default=parse_args(['calibrate','--output',str(tmp_path)])
    assert len(selected_layouts(default))==24
    small=parse_args(['calibrate','--output',str(tmp_path),'--layouts-per-domain','1'])
    assert [(x['domain'],x['seed']) for x in selected_layouts(small)] == [
        ('navigation',7000),('stairs',12000),('gap',16000),('manipulation',20000)]


def test_population_view_excludes_unfilled_real_memory_capacity():
    import torch
    from drl.MARL.c1_runner import make_rollout_memory
    memory=make_rollout_memory(memory_size=16,num_envs=1,device='cpu')
    memory.create_tensor(name='terminated',size=1,dtype=torch.bool)
    memory.add_samples(terminated=torch.tensor([[False]]))
    memory.add_samples(terminated=torch.tensor([[True]]))
    assert memory.get_tensor_by_name('terminated').shape == (2,1,1)
    assert memory.get_tensor_by_name('terminated')[:,0,0].tolist()==[False,True]
    assert memory.tensors['terminated'].shape == (16,1,1)


def test_cleanup_terminates_surviving_group_child_after_leader_exits(tmp_path):
    import os,signal,subprocess,sys,time
    from drl.MARL.c1_runner import terminate_process_group
    pid_path=tmp_path/'child.pid'
    code='import subprocess,sys; from pathlib import Path; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);Path(sys.argv[1]).write_text(str(p.pid))'
    leader=subprocess.Popen([sys.executable,'-c',code,str(pid_path)],start_new_session=True)
    try:
        leader.wait(timeout=5)
        child_pid=int(pid_path.read_text())
        assert os.getpgid(child_pid)==leader.pid
        terminate_process_group(leader,term_timeout_s=.5,kill_timeout_s=.5)
        stat=Path(f'/proc/{child_pid}/stat')
        assert not stat.exists() or stat.read_text().split(') ',1)[1].split()[0]=='Z'
    finally:
        try:os.killpg(leader.pid,signal.SIGKILL)
        except ProcessLookupError:pass


def test_resource_conflict_attribution_marks_only_competing_selected_owners():
    import torch
    from drl.MARL.c1_runner import sampled_resource_conflict_owners
    candidate=lambda owner,resource:SimpleNamespace(resource_ref=SimpleNamespace(module=SimpleNamespace(module_id=owner),resource_id=resource))
    route=dict(resource_candidates=[candidate('m0','locomotion'),candidate('m0','locomotion'),candidate('m1','locomotion')])
    assert sampled_resource_conflict_owners(SimpleNamespace(presence=torch.tensor([1.,1.,1.])),route)==('m0',)
    assert sampled_resource_conflict_owners(SimpleNamespace(presence=torch.tensor([0.,1.,1.])),route)==()


def test_result_partial_calibration_is_bootstrap_only_and_requires_real_actuation():
    from drl.MARL.c1_runner import run_result
    partial=run_result('calibrate',8,0,[],dict(navigation=8),0)
    assert partial['status']=='BOOTSTRAP_ONLY'
    assert not partial['complete_four_domain_evidence']
    assert not partial['calibration_fit_complete']
    all_domains={d:8 for d in ('navigation','stairs','gap','manipulation')}
    rejected=run_result('smoke',32,1,[],all_domains,0)
    assert rejected['status']=='INCOMPLETE_ACTUATION'
    accepted=run_result('smoke',32,1,[],all_domains,1)
    assert accepted['status']=='GREEN'


def test_runtime_mailboxes_default_to_tmpfs_separate_from_durable_output(tmp_path):
    from drl.MARL.c1_runner import parse_args,PhysicalEpisode
    args=parse_args(['calibrate','--output',str(tmp_path)])
    episode=PhysicalEpisode(args,'navigation',7000,'unit-owned-episode',object())
    assert episode.directory == Path('/dev/shm/mssr-c1/unit-owned-episode')
    assert episode.evidence_directory == tmp_path/'runtime'/'unit-owned-episode'


def test_runtime_evidence_copy_bounds_log_and_preserves_causal_files(tmp_path):
    from drl.MARL.c1_runner import copy_runtime_evidence
    source,dest=tmp_path/'ephemeral',tmp_path/'durable'
    source.mkdir()
    (source/'runtime.log').write_bytes(b'a'*1000+b'last causal event\n')
    (source/'bc_status.json').write_text('{"last_completed":{"command_id":"bc:s:1"}}')
    copy_runtime_evidence(source,dest,max_log_bytes=100)
    assert (dest/'runtime.log').stat().st_size <= 100
    assert (dest/'runtime.log').read_text().endswith('last causal event\n')
    assert json.loads((dest/'bc_status.json').read_text())['last_completed']['command_id']=='bc:s:1'
    assert json.loads((dest/'evidence.json').read_text())['log_tail_truncated']


def test_sigterm_requests_python_unwind_for_owned_runtime_cleanup():
    import signal
    from drl.MARL.c1_runner import shutdown_on_signal
    with pytest.raises(SystemExit) as caught:
        shutdown_on_signal(signal.SIGTERM,None)
    assert caught.value.code==143


@pytest.mark.parametrize('domain,seed',[('navigation',7000),('stairs',12000),('gap',16000),('manipulation',20000)])
def test_all_c1_course_task_views_pass_strict_frozen_bc_bridge(tmp_path,domain,seed):
    """Generated current geometry and synthetic modules are UNIT plumbing only."""
    from drl.MARL.c1_runner import parse_args,physical_task_view,make_course,build_observation_bridge
    from drl.IL.tests.bc_v1.test_live_observation import snapshot
    args=parse_args(['calibrate','--output',str(tmp_path)])
    course=make_course(domain,seed,'train').to_observation()
    context,observation=physical_task_view(domain,course,json.loads((args.repository_root/'configs/bc_v1/task_context.json').read_text()))
    if domain!='navigation':
        assert all('goal.' not in feature.name for feature in context.static_geometry)
    prepared=build_observation_bridge(args.repository_root).prepare_snapshot(snapshot(n=8,rc=True),
        task_context=context,observation=observation)
    assert prepared.physical_state.stamp==12.5


def test_saved_actual_stairs_frame_passes_frozen_bc_bridge_after_context_selection(tmp_path):
    """Reprocess saved physical evidence; no simulator or calibration collection."""
    from drl.MARL.c1_runner import parse_args,physical_task_view,build_observation_bridge
    args=parse_args(['calibrate','--output',str(tmp_path)])
    paths=list((args.repository_root/'logs/appo_smoke/calibration_stairs_train12000_v3/runtime').glob('*/state_graph.json'))
    if not paths:pytest.skip('saved local physical stairs evidence is unavailable')
    raw=json.loads(paths[0].read_text())
    context,observation=physical_task_view('stairs',raw['state']['course'],
        json.loads((args.repository_root/'configs/bc_v1/task_context.json').read_text()))
    prepared=build_observation_bridge(args.repository_root).prepare_snapshot(raw,task_context=context,observation=observation)
    assert prepared.physical_state.stamp==raw['stamp']


@pytest.mark.parametrize('command',['smoke','train'])
def test_learning_runtime_exception_stops_before_next_episode_or_update(tmp_path,monkeypatch,command):
    """Fault-injection UNIT plumbing: no simulator or physical evidence."""
    import torch
    import drl.MARL.c1_runner as runner
    import drl.MARL.phase1_vertical_slice as vertical
    import drl.MARL.reward_calibration as calibration
    import drl.MARL.reward_config as configuration
    opened,recorded,updates=[],[],[]
    status=dict(last_completed=dict(outcome='accepted',command_id='bc:unit:1'),last_applied={})
    class Episode:
        def __init__(self,args,domain,seed,episode_id,bridge):
            self.domain,self.seed,self.episode_id=domain,seed,episode_id
            self.owners=('m0',)
            self.env=self.context=self.synchronizer=object()
            self.transport=SimpleNamespace(status=lambda:status)
            self.availability={}
        def __enter__(self):
            opened.append(self.domain)
            return self
        def __exit__(self,*args):pass
    class Slice:
        def __init__(self,**kwargs):self.calls=0
        def step(self):
            self.calls+=1
            if self.calls>1:raise RuntimeError('unit physics fault')
            transition=SimpleNamespace(previous_state=dict(stamp=0.),episode_step=1,
                evaluation=SimpleNamespace(raw_reward=SimpleNamespace(team=SimpleNamespace(success=0.)),terminated=False,truncated=False),
                actuation_result=SimpleNamespace(command_id='bc:unit:1',proposal_diagnostics={},decoded=(object(),)))
            return SimpleNamespace(
                active_env_indices=(0,),
                transitions=(transition,),
                contexts=(SimpleNamespace(actor_graph={},critic_graph={}),),
                decisions=(SimpleNamespace(owner_log_probs={},global_value=torch.tensor(0.)),))
    class Recorder:
        def __init__(self,*args):pass
        def observe(self,*args,**kwargs):pass
    learner=SimpleNamespace(memories={'m0':recorded},possible_agents=('m0',))
    monkeypatch.setattr(runner,'build_observation_bridge',lambda root:object())
    monkeypatch.setattr(runner,'build_models',lambda *args:(object(),object()))
    monkeypatch.setattr(runner,'PhysicalEpisode',Episode)
    monkeypatch.setattr(vertical,'Phase1VerticalSliceRunner',Slice)
    monkeypatch.setattr(calibration,'TransitionRecorder',Recorder)
    monkeypatch.setattr(runner,'episode_manifest',lambda *args:object())
    monkeypatch.setattr(runner,'create_learner',lambda *args:(learner,object()))
    monkeypatch.setattr(runner,'record_learning_batch',lambda *args:recorded.append('valid unit transition'))
    def update(*args):
        updates.append(True)
        (tmp_path/'APPO_SMOKE.pt').write_text('unexpected unit checkpoint')
    monkeypatch.setattr(runner,'update_learner',update)
    config=SimpleNamespace(scalarize=lambda *args,**kwargs:SimpleNamespace(team=.1,owners={'m0':.1}))
    monkeypatch.setattr(configuration.RewardConfig,'load',lambda path:config)
    monkeypatch.setattr(runner.signal,'signal',lambda *args:None)
    with pytest.raises(RuntimeError,match='unit physics fault'):
        runner.main([command,'--output',str(tmp_path),'--reward-config',str(tmp_path/'unused.json'),
                     '--steps-per-layout','3','--rollout-transitions','64'])
    assert opened==['navigation']
    assert len(recorded)==1
    assert not updates
    assert not (tmp_path/'APPO_SMOKE.pt').exists()
    assert not (tmp_path/'result.json').exists()


def test_irrecoverable_fall_requires_all_eight_below_actual_support_surface():
    from drl.MARL.c1_runner import irrecoverable_fall,minimum_support_height_m
    box=SimpleNamespace(center_xyz_m=(0,0,.0),size_xyz_m=(2,2,.02),pitch_deg=0.,collidable=True)
    course=SimpleNamespace(support_boxes=(box,))
    floor=minimum_support_height_m(course)
    raw=dict(state=dict(modules=[dict(module_id=f'm{i}',position=[0,0,floor-.151]) for i in range(8)]))
    assert irrecoverable_fall(course,raw)
    raw['state']['modules'][0]['position'][2]=floor+.03
    assert not irrecoverable_fall(course,raw)
    raw['state']['modules'][0]['position'][2]=floor-.15
    assert not irrecoverable_fall(course,raw)


def test_support_floor_uses_lowest_point_of_pitched_top_face():
    import math
    from drl.MARL.c1_runner import minimum_support_height_m
    box=SimpleNamespace(center_xyz_m=(0,0,1.),size_xyz_m=(2,1,.2),pitch_deg=30.,collidable=True)
    assert minimum_support_height_m(SimpleNamespace(support_boxes=(box,))) == pytest.approx(1+.1*math.cos(math.pi/6)-math.sin(math.pi/6))


@pytest.mark.parametrize('coordinate',[float('nan'),float('inf'),100.001,-100.001])
def test_invalid_world_geometry_is_infrastructure_failure(coordinate):
    from drl.MARL.c1_runner import captured_world_positions
    raw=dict(state=dict(modules=[dict(module_id=f'm{i}',position=[0,0,coordinate]) for i in range(8)]))
    with pytest.raises(RuntimeError,match='world XYZ'):
        captured_world_positions(raw)


def test_navigation_whole_body_fall_over_goal_is_failure_with_absorbing_potential(tmp_path):
    from drl.MARL.c1_runner import PhysicalEpisode,parse_args
    from drl.MARL.runtime_signals import StagnationDetector
    from drl.MARL.phase1_reward import compute_phase1_raw_reward
    args=parse_args(['calibrate','--output',str(tmp_path),'--steps-per-layout','256'])
    episode=PhysicalEpisode(args,'navigation',7000,'unit-fall',object())
    episode.owners=tuple(f'm{i}' for i in range(8))
    episode.stagnation=StagnationDetector()
    episode.transport=SimpleNamespace(completed_outcome=lambda action:SimpleNamespace(accepted=True,code='EMPTY',owner_ids=()))
    before_xy=episode.course.start_platform.center_xy_m
    after_xy=episode.course.target_platform.center_xy_m
    raw=lambda stamp,xy,z:dict(stamp=stamp,state=dict(modules=[dict(module_id=owner,position=[*xy,z]) for owner in episode.owners]))
    pre,post=raw(0.,before_xy,.04),raw(1/30,after_xy,-.2)
    result=episode.evaluate_transition(env_index=0,previous_state=pre,next_state=post,
        actuation_result=SimpleNamespace(local_rejection=None),physics_result=None,episode_step=1)
    assert result.terminated and not result.truncated
    assert not result.success
    assert result.failure_reason=='irrecoverable_fall'
    assert result.raw_reward.team.success==0.
    assert all(terms.success==0 for terms in result.raw_reward.owners.values())
    absorbing=compute_phase1_raw_reward(episode.course,{o:before_xy for o in episode.owners},
        {o:after_xy for o in episode.owners},signals=episode.last_signals.signals,shaping_gamma=.99,terminated=True)
    assert result.raw_reward.team.progress==pytest.approx(absorbing.team.progress)


def test_stairs_fall_records_terminal_failure_without_deep_fall_tail(tmp_path):
    """UNIT transition exercises the completed observer; no empirical fitting."""
    from drl.MARL.c1_runner import PhysicalEpisode,parse_args,minimum_support_height_m,BC_SHA256
    from drl.MARL.direct_marl_env import DirectMARLTransition
    from drl.MARL.reward_calibration import EpisodeManifest,TransitionRecorder,CalibrationDataset
    from drl.MARL.runtime_signals import StagnationDetector
    args=parse_args(['calibrate','--output',str(tmp_path),'--steps-per-layout','256'])
    episode=PhysicalEpisode(args,'stairs',12000,'unit-stairs-fall',object())
    episode.owners=tuple(f'm{i}' for i in range(8))
    episode.stagnation=StagnationDetector()
    episode.transport=SimpleNamespace(completed_outcome=lambda action:SimpleNamespace(accepted=True,code='EMPTY',owner_ids=()))
    floor=minimum_support_height_m(episode.course)
    raw=lambda stamp,z:dict(stamp=stamp,state=dict(modules=[dict(module_id=owner,position=[0,0,z]) for owner in episode.owners]))
    pre,post=raw(0.,floor+.04),raw(1/30,floor-.151)
    action=SimpleNamespace(local_rejection=None)
    evaluation=episode.evaluate_transition(env_index=0,previous_state=pre,next_state=post,
        actuation_result=action,physics_result=None,episode_step=1)
    assert evaluation.terminated and not evaluation.truncated
    assert evaluation.failure_reason=='irrecoverable_fall'
    # Existing raw below_support is normalized by wheel radius, not metres.
    assert evaluation.raw_reward.team.below_support<10.
    manifest=EpisodeManifest(run_id='unit',episode_id='unit-stairs-fall',domain='stairs',morphology='snake8',
        seed=12000,split='train',difficulty='easy',family=None,collection_policy='unit-only',
        checkpoint_path='unit',checkpoint_sha256=BC_SHA256,base_commit='unit',working_tree_dirty=True,
        physics_hz=240.,decision_hz=30.,source='synthetic')
    transition=DirectMARLTransition(env_index=0,previous_state=pre,actuation_result=action,physics_result=None,
        next_state=post,evaluation=evaluation,episode_step=1)
    recorder=TransitionRecorder(tmp_path/'unit-observer.jsonl',manifest)
    row=recorder.observe(transition,availability=episode.availability)
    assert row['failure_reason']=='irrecoverable_fall'
    assert row['terminated'] and not row['success'] and not row['truncated']
    assert CalibrationDataset([row]).report()['episodes'][0]['failure']
    with pytest.raises(ValueError,match='terminal'):
        recorder.observe(transition,availability=episode.availability)


@pytest.mark.parametrize('domain,seed',[('navigation',7000),('stairs',12000),('gap',16000),('manipulation',20000)])
def test_actual_domain_reward_dataclasses_terminate_whole_body_fall_and_preserve_signals(tmp_path,domain,seed):
    """UNIT geometry exercises all real evaluators, without physical collection."""
    from drl.MARL.c1_runner import PhysicalEpisode,parse_args,minimum_support_height_m,attached_connections
    from drl.MARL.runtime_signals import StagnationDetector
    args=parse_args(['calibrate','--output',str(tmp_path)])
    episode=PhysicalEpisode(args,domain,seed,'unit-fall-'+domain,object())
    episode.owners=tuple(f'm{i}' for i in range(8))
    episode.end_effector_id='m7'
    episode.stagnation=StagnationDetector()
    episode.transport=SimpleNamespace(completed_outcome=lambda action:SimpleNamespace(accepted=False,code='INVALID_GOAL',owner_ids=('m0',)))
    floor=minimum_support_height_m(episode.course)
    edges=[dict(module_a_id=f'm{i}',connector_a_id='TOP',module_b_id=f'm{i+1}',connector_b_id='BOTTOM',is_attached=True) for i in range(7)]
    raw=lambda stamp,z:dict(stamp=stamp,graph=dict(edges=edges),state=dict(course=dict(button=dict(depression_m=0.)),
        modules=[dict(module_id=owner,position=[0,0,z],connectors=[dict(connector_id='BOTTOM',position_world=[0,0,z-.04],outward_normal_world=[0,1,0])]) for owner in episode.owners]))
    pre,post=raw(0.,floor+.04),raw(1/30,floor-.151)
    episode.intended_connections=attached_connections(pre)
    result=episode.evaluate_transition(env_index=0,previous_state=pre,next_state=post,
        actuation_result=SimpleNamespace(local_rejection=None),physics_result=None,episode_step=1)
    assert result.terminated and not result.truncated and not result.success
    assert result.failure_reason=='irrecoverable_fall'
    assert result.raw_reward.team.success==0.
    assert result.raw_reward.owners['m0'].invalid==1.
    assert result.raw_reward.owners['m1'].invalid==0.
    assert result.raw_reward.team.time==1.
    if domain=='gap':
        assert not hasattr(result.raw_reward.owners['m0'],'success')
    else:
        assert all(terms.success==0 for terms in result.raw_reward.owners.values())


def test_completed_episode_outcome_distinguishes_success_failure_and_truncation():
    """Completed UNIT transitions verify logging only, without physical evidence."""
    from drl.MARL.c1_runner import completed_episode_outcome,aggregate_episode_outcomes
    from drl.MARL.direct_marl_env import DirectMARLTransition
    make=lambda success,terminated,truncated,reason:DirectMARLTransition(env_index=0,
        previous_state=dict(stamp=.1),actuation_result=None,physics_result=None,next_state=dict(stamp=4/30),
        evaluation=SimpleNamespace(raw_reward=SimpleNamespace(team=SimpleNamespace(success=success)),
            terminated=terminated,truncated=truncated,failure_reason=reason),episode_step=4)
    failure=completed_episode_outcome(make(0.,True,False,'irrecoverable_fall'),episode_id='unit-fail',
        domain='stairs',seed=13000,split='validation',episode_start_stamp=0.)
    success=completed_episode_outcome(make(1.,True,False,None),episode_id='unit-success',
        domain='stairs',seed=13001,split='validation',episode_start_stamp=0.)
    timeout=completed_episode_outcome(make(0.,False,True,None),episode_id='unit-timeout',
        domain='stairs',seed=13002,split='validation',episode_start_stamp=0.)
    assert failure['failure_reason']=='irrecoverable_fall' and not failure['success']
    assert failure['length_steps']==4 and failure['length_seconds']==pytest.approx(4/30)
    assert failure['seed']==13000 and failure['split']=='validation'
    summary=aggregate_episode_outcomes([failure,success,timeout])['stairs']
    assert summary==dict(episodes=3,success_count=1,failure_count=1,truncation_count=1,success_rate=1/3)


def test_scalar_log_outcome_uses_raw_success_and_completed_failure_reason():
    from drl.MARL.c1_runner import scalar_outcome_fields
    evaluation=SimpleNamespace(raw_reward=SimpleNamespace(team=SimpleNamespace(success=0.)),
        terminated=True,truncated=False,failure_reason='irrecoverable_fall')
    assert scalar_outcome_fields(evaluation)==dict(success=False,failure_reason='irrecoverable_fall')
    evaluation.raw_reward.team.success=1.
    evaluation.failure_reason=None
    assert scalar_outcome_fields(evaluation)==dict(success=True,failure_reason=None)


def test_future_train_uses_full_reproducibly_shuffled_frozen_train_seed_pools(tmp_path):
    from drl.MARL.c1_runner import parse_args,selected_layouts
    args=parse_args(['train','--output',str(tmp_path),'--reward-config',str(tmp_path/'unused.json')])
    rows=selected_layouts(args)
    assert rows==selected_layouts(args)
    assert len(rows)==4000
    for domain,start in [('navigation',7000),('stairs',12000),('gap',16000),('manipulation',20000)]:
        pool=[row for row in rows if row['domain']==domain]
        assert len(pool)==1000
        assert {row['seed'] for row in pool}==set(range(start,start+1000))
        assert all(row['split']=='train' for row in pool)
        assert {row['difficulty'] for row in pool[:24]}=={'easy','medium','hard'}
        assert [row['seed'] for row in pool]!=list(range(start,start+1000))
    capped=selected_layouts(parse_args(['train','--output',str(tmp_path),'--reward-config',str(tmp_path/'unused.json'),'--layouts-per-domain','3']))
    assert len(capped)==12
    for domain in ('navigation','stairs','gap','manipulation'):
        assert [row for row in capped if row['domain']==domain]==[row for row in rows if row['domain']==domain][:3]


def test_calibration_smoke_and_validation_layout_selection_remain_unchanged(tmp_path):
    from drl.MARL.c1_runner import parse_args,selected_layouts
    from drl.MARL.reward_calibration import pilot_layouts
    assert selected_layouts(parse_args(['calibrate','--output',str(tmp_path)]))==pilot_layouts()
    smoke=selected_layouts(parse_args(['smoke','--output',str(tmp_path),'--reward-config',str(tmp_path/'unused.json')]))
    assert [(row['domain'],row['seed']) for row in smoke]==[('navigation',7000),('stairs',12000),('gap',16000),('manipulation',20000)]
    validation=selected_layouts(parse_args(['evaluate','--split','validation','--output',str(tmp_path),'--reward-config',str(tmp_path/'unused.json')]))
    assert [(row['domain'],row['seed']) for row in validation]==[('navigation',8000),('stairs',13000),('gap',17000),('manipulation',21000)]


def test_training_layout_cursor_advances_each_domain_pool_independently():
    from drl.MARL.c1_runner import TrainingLayoutPool
    rows=[dict(domain='navigation',seed=7001),dict(domain='navigation',seed=7000),
          dict(domain='stairs',seed=12001),dict(domain='stairs',seed=12000)]
    pool=TrainingLayoutPool(rows)
    assert pool.next_layout('navigation')['seed']==7001
    assert pool.next_layout('navigation')['seed']==7000
    assert pool.next_layout('stairs')['seed']==12001
    assert pool.next_layout('navigation')['seed']==7001
    assert pool.next_layout('stairs')['seed']==12000



def test_lockstep_transport_keeps_two_env_mailboxes_sessions_and_actions_independent(
    tmp_path,
):
    """One transport may serve N Isaac envs without env-0 aliasing."""
    import json

    from types import SimpleNamespace

    from drl.MARL.c1_runner import (
        LockstepTransport,
    )

    env0 = tmp_path / "env_0"
    env1 = tmp_path / "env_1"

    env0.mkdir()
    env1.mkdir()

    status0 = {
        "session": "session-0",
        "epoch": "epoch-0",
        "armed": True,
        "failure": None,
        "rejections": [],
        "last_completed": None,
    }

    status1 = {
        "session": "session-1",
        "epoch": "epoch-1",
        "armed": True,
        "failure": None,
        "rejections": [],
        "last_completed": None,
    }

    (env0 / "bc_status.json").write_text(
        json.dumps(status0)
    )

    (env1 / "bc_status.json").write_text(
        json.dumps(status1)
    )

    expected_actions = []

    synchronizer = SimpleNamespace(
        expect_action=lambda env_index, command_id:
            expected_actions.append(
                (
                    env_index,
                    command_id,
                )
            )
    )

    transport = LockstepTransport(
        (
            env0,
            env1,
        ),
        (
            "session-0",
            "session-1",
        ),
        synchronizer,
        step_ticks=8,
    )

    decoded0 = SimpleNamespace(
        command={
            "command": "wheel_velocity",
            "module_ids": [
                "env0_module",
            ],
            "parameters": {
                "left_rad_s": 0.1,
                "right_rad_s": 0.2,
            },
        }
    )

    decoded1 = SimpleNamespace(
        command={
            "command": "wheel_velocity",
            "module_ids": [
                "env1_module",
            ],
            "parameters": {
                "left_rad_s": 0.3,
                "right_rad_s": 0.4,
            },
        }
    )

    # Desired vector API. Default env_index=0 will preserve
    # existing single-env call sites.
    transport.stage(
        (
            decoded0,
        ),
        state_stamp=10.0,
        env_index=0,
    )

    transport.stage(
        (
            decoded1,
        ),
        state_stamp=20.0,
        env_index=1,
    )

    action0 = transport.step(
        0
    )

    action1 = transport.step(
        1
    )

    request0 = json.loads(
        (
            env0
            / "bc_request.json"
        ).read_text()
    )

    request1 = json.loads(
        (
            env1
            / "bc_request.json"
        ).read_text()
    )

    # Each environment writes only to its own native mailbox.
    assert request0[
        "session"
    ] == "session-0"

    assert request1[
        "session"
    ] == "session-1"

    assert request0[
        "state_stamp"
    ] == 10.0

    assert request1[
        "state_stamp"
    ] == 20.0

    assert request0[
        "continuous"
    ][0][
        "payload"
    ] == decoded0.command

    assert request1[
        "continuous"
    ][0][
        "payload"
    ] == decoded1.command

    # Environment identity reaches the already-vectorized
    # FileBackedIsaacControlStep.
    assert expected_actions == [
        (
            0,
            action0.command_id,
        ),
        (
            1,
            action1.command_id,
        ),
    ]

    assert action0.command_id.startswith(
        "bc:session-0:"
    )

    assert action1.command_id.startswith(
        "bc:session-1:"
    )

    assert action0.command_id != action1.command_id

    # Completing env 1 must inspect env 1's status mailbox,
    # never env 0's.
    status1[
        "last_completed"
    ] = {
        "command_id":
            action1.command_id,
        "outcome":
            "empty",
    }

    (
        env1
        / "bc_status.json"
    ).write_text(
        json.dumps(
            status1
        )
    )

    assert (
        transport
        .completed_outcome(
            action1
        )
        .code
        == "EMPTY"
    )

    assert (
        status0[
            "last_completed"
        ]
        is None
    )
