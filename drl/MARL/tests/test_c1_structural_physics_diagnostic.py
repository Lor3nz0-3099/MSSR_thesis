"""Offline contracts for the user-run frozen physical diagnostic."""
from types import SimpleNamespace
import json
import pytest
import torch
from drl.MARL.tests.test_align_face_claims import goal,route,capture
from drl.MARL.tests.test_c1_feasibility import executor_fixture,simple_output,gravity_fixture
from drl.MARL.c1_feasibility_cache import NativeAvailabilityCache
from drl.MARL.policy import bind_marl_candidates,sample_marl_action_bundle,MARLCandidateRecord
from smores_ep.primitives.model import PrimitiveState


def recorder(tmp_path):
    from drl.MARL.c1_primitive_lifecycle import PrimitiveLifecycleRecorder
    return PrimitiveLifecycleRecorder(tmp_path,('navigation','stairs','gap','manipulation'))


def observe(rec,executor,cache,stamp=0.,generation=1):
    availability=capture(cache,executor,generation)
    records=bind_marl_candidates(routing=route(),presence_logits=torch.zeros(3),presence_threshold_logit=4.5)
    rec.observe_context(0,generation,records,availability,stamp)
    return availability


def test_cli_defaults_are_frozen_four_env_40_seconds_and_reject_invalid_bounds():
    from drl.MARL.c1_structural_physics_diagnostic import parse_args,configuration
    args=parse_args(['--headless'])
    cfg=configuration(args)
    assert cfg['simulated_seconds_per_env']==40.
    assert cfg['num_envs']==4 and cfg['optimizer_updates']==0 and cfg['learner_constructed'] is False
    assert cfg['primitive_timeout_s']==30. and cfg['episode_seconds']==60.
    assert cfg['presence_threshold_logit']==4.5 and cfg['sampling']=='stochastic'
    for values in (['--steps','0'],['--steps','1201'],['--max-wall-seconds','nan']):
        with pytest.raises(SystemExit):parse_args(values)


@pytest.mark.parametrize('terminal',['succeeded','failed','canceled','timeout'])
def test_precise_lifecycle_release_and_cache_reopen_are_passive(tmp_path,terminal):
    rec=recorder(tmp_path);executor=executor_fixture();cache=NativeAvailabilityCache()
    rec.attach_executor(executor,0,1);rec.decision_step=7
    before=torch.random.get_rng_state()
    observe(rec,executor,cache)
    status=executor.submit(goal('left','module_a','LEFT'),.125)
    assert status.state is PrimitiveState.ACCEPTED
    assert observe(rec,executor,cache,.14).candidate_mask==(True,False,True)
    version=executor.active_claim_version
    if terminal=='canceled':executor.cancel('left',.625)
    elif terminal=='timeout':
        # Native timeout branch, no physical execution.
        executor._active['left'].started_at_s=-30.
        executor.step(.625)
    else:executor._finish(executor._active['left'],PrimitiveState(terminal),.625,'TEST','terminal')
    assert executor.active_claim_version==version+1
    observe(rec,executor,cache,.633)
    assert torch.equal(before,torch.random.get_rng_state())
    summary=rec.summary();row=summary['goals'][0]
    assert row['start_timestamp_s']==.125 and row['terminal_timestamp_s']==.625
    assert row['duration_s']==pytest.approx(.5)
    assert row['claims_released'] is True
    assert row['claims']['connector:module_b:LEFT']=='exclusive'
    label='TIMEOUT' if terminal=='timeout' else {'succeeded':'COMPLETE','failed':'FAILED','canceled':'CANCELLED'}[terminal]
    assert summary['by_env']['0']['align_faces'][label]==1
    assert row['mask_after_release']['reopened_count']==2
    assert summary['success_criteria']['align_real_terminal'] is True
    assert (tmp_path/'primitive_lifecycle.jsonl').exists()


def test_cutoff_is_censored_and_reset_is_not_reported_as_reopen(tmp_path):
    rec=recorder(tmp_path);executor=executor_fixture();cache=NativeAvailabilityCache()
    rec.attach_executor(executor,0,1);executor.submit(goal('left','module_a','LEFT'),0.)
    observe(rec,executor,cache)
    rec.censor_open(2.,'diagnostic_cutoff')
    row=rec.summary()['goals'][0]
    assert row['terminal_state'] is None and row['censored'] is True
    assert rec.summary()['success_criteria']['align_real_terminal'] is False
    executor.cancel('left',2.1)
    fresh=executor_fixture();observe(rec,fresh,cache,2.13,generation=2)
    assert rec.summary()['goals'][0]['mask_after_release']['state']=='reset_epoch_changed'
    assert rec.summary()['by_env']['0']['mask_reopen_after_release_count']==0


def test_lifecycle_observer_retains_sibling_claims_and_isolates_goal_ids_by_env(tmp_path):
    rec=recorder(tmp_path);a=executor_fixture();b=executor_fixture();cache=NativeAvailabilityCache()
    rec.attach_executor(a,0,1);rec.attach_executor(b,1,1)
    a.submit(goal('same','module_a','LEFT'),0.);b.submit(goal('same','module_a','LEFT'),0.)
    a.submit(goal('right','module_c','RIGHT'),0.)
    observe(rec,a,cache)
    a.cancel('same',.5);observe(rec,a,cache,.533)
    assert len(a.active_goals)==1 and len(b.active_goals)==1
    rows=rec.summary()['goals'];released=next(r for r in rows if r['env_index']==0 and r['goal_id']=='same')
    assert released['claims_released'] and released['mask_after_release']['reopened_count']==2
    assert next(r for r in rows if r['env_index']==1)['terminal_state'] is None


def test_sampling_reuses_same_rng_and_optional_availability_support():
    from drl.MARL.c1_structural_physics_diagnostic import sample_decision
    from drl.MARL.c1_feasibility import apply_availability
    from drl.MARL.tests.test_c1_feasibility import capture as full_capture,align_goal
    executor=executor_fixture();executor.submit(align_goal(),0.)
    routing,graph,payload=gravity_fixture();availability=full_capture(executor,routing,graph)
    output=simple_output(torch.tensor([-10.]*4))
    output.update(structural_payloads={4:payload},prediction_template={'presence_logits':torch.zeros(5)})
    output['records']=(*output['records'],MARLCandidateRecord(4,'structural','gravity','module_c',(),torch.tensor(20.),4.5))
    actor=SimpleNamespace(forward_all_owners=lambda **kwargs:output)
    context=SimpleNamespace(actor_graph=graph,routing_snapshot=routing,hard_mask=availability.mask_tensor(),availability_snapshot=availability)
    torch.manual_seed(87);before=torch.random.get_rng_state()
    filtered=apply_availability(output,availability)
    expected=sample_marl_action_bundle(**{k:filtered[k] for k in ('records','resource_payloads','structural_payloads','resource_log_std','structural_log_std')},hard_mask=context.hard_mask,deterministic=False)
    after=torch.random.get_rng_state();torch.random.set_rng_state(before)
    decision,_=sample_decision(actor,context,('module_c',))
    assert torch.equal(after,torch.random.get_rng_state())
    assert torch.equal(expected.presence,decision.action.presence)
    assert torch.equal(expected.structural_scope[4],decision.action.structural_scope[4])
    assert decision.action_set_prediction_template is output['prediction_template']


def test_monitor_reads_progress_without_isaac(tmp_path):
    from drl.MARL.c1_structural_physics_diagnostic import format_progress
    assert 'awaiting' in format_progress(tmp_path)
    (tmp_path/'progress.json').write_text(json.dumps(dict(status='running',decisions_completed=30,simulated_seconds_per_env=1.,wall_seconds=100.,by_env={'0':dict(task='navigation',align_faces={'START':2,'COMPLETE':1,'TIMEOUT':0},dock={'START':1,'COMPLETE':0})},errors=[])))
    value=format_progress(tmp_path)
    assert 'sim/env=1.000' in value and 'align=2/1/0' in value and 'dock=1/0' in value


def test_short_collector_uses_native_provider_cache_and_no_learning(monkeypatch,tmp_path):
    from collections import deque
    from dataclasses import replace
    from drl.MARL.tests.test_native_policy_bridge import make_bridge
    from drl.MARL.tests.test_c1_feasibility import routing_fixture
    from drl.MARL.c1_structural_physics_diagnostic import rollout_frozen
    bridge,env,_,decoded=make_bridge()
    executors=[executor_fixture(),executor_fixture()];routing=routing_fixture()
    for candidate in routing['resource_candidates']:
        candidate.action=SimpleNamespace(action_id='drive')
        candidate.mode=SimpleNamespace(mode_id='velocity')
    prepared=bridge.provider.observation_source.prepare_env(0)
    bridge.provider.observation_source=SimpleNamespace(num_envs=2,
        prepare_env=lambda i:replace(prepared,routing_snapshot=routing,
            actor_graph={'stamp':env.stamp},privileged_graph={'stamp':env.stamp}),reset=lambda:None)
    bridge.provider._hard_mask_builder=lambda g,r:torch.zeros(6,dtype=torch.bool)
    env.cfg=SimpleNamespace(initialization_mode='loose_modules');env.generation=[1,1]
    env.native_dispatcher=SimpleNamespace(executor_provider=lambda i:executors[i])
    env._sim_step_counter=0;env.physics_dt=1/240
    env.c1_runtime=SimpleNamespace(recent_transitions=deque(maxlen=64),evaluators={})
    frame=SimpleNamespace(pathological_collision_owner_ids=(),evidence_complete=True,connections=())
    calls=[]
    def forward(**kw):
        assert not torch.is_grad_enabled()
        calls.append(kw['graph']['stamp'])
        return dict(simple_output(torch.full((6,),-100.)),prediction_template={'presence_logits':torch.full((6,),-100.)})
    def step(carrier):
        assert carrier['m0'].shape==(2,1)
        env._sim_step_counter+=8;env.stamp+=1/30
        for i in range(2):
            env.c1_runtime.recent_transitions.append(dict(env_index=i,generation=1,action_counts={},assessment=dict(valid_connections=0,safe=True),timestamp_s=env.stamp))
            env.c1_runtime.evaluators[i]=(1,SimpleNamespace(last_frame=frame))
        return {},{'m0':torch.zeros(2)},None,None,None
    env.step=step
    def forbidden(*a,**kw):pytest.fail('Learning/critic invoked in frozen diagnostic')
    monkeypatch.setattr(torch.optim.Optimizer,'__init__',forbidden)
    monkeypatch.setattr(torch.autograd,'backward',forbidden)
    # Terminal critic tensorization is deliberately discarded by provider.reset.
    bridge.provider.current_critic_graphs=forbidden
    rec=recorder(tmp_path);progress=[]
    result=rollout_frozen(env,SimpleNamespace(forward_all_owners=forward),bridge.provider,rec,('module_c',),
        steps=3,max_wall_seconds=60.,progress_callback=lambda n,*a:progress.append(n))
    assert result['decisions_completed']==3 and result['cutoff_reason']=='horizon_reached'
    assert progress==[1,2,3] and len(decoded)==6 and len(calls)==6
    assert calls[0]!=calls[2]!=calls[4]  # Fresh state each decision.
    assert bridge.provider.feasibility_metrics()['totals']['mask_cache_hits']==6
    assert bridge.provider.feasibility_metrics()['totals']['mask_cache_misses']==2


def test_ideal_chain_requires_same_env_generation_pair_and_valid_joint_after_completion(tmp_path):
    from drl.MARL.c1_structure import C1Connection
    from smores_ep.primitives.model import PrimitiveGoal
    rec=recorder(tmp_path);executor=executor_fixture();cache=NativeAvailabilityCache()
    rec.attach_executor(executor,0,1)
    executor.submit(goal('align','module_a','LEFT'),0.);observe(rec,executor,cache)
    executor._finish(executor._active['align'],PrimitiveState.SUCCEEDED,.5,'DONE','complete')
    observe(rec,executor,cache,.533)
    dock=PrimitiveGoal.from_dict(dict(goal_id='dock',primitive='dock',module_ids=['module_a','module_b'],parameters=dict(face_a='BOTTOM',face_b='LEFT')))
    executor.submit(dock,.6)
    frame=SimpleNamespace(time_s=.7,pathological_collision_owner_ids=(),evidence_complete=True,
        connections=(C1Connection('joint','module_a','BOTTOM','module_b','LEFT',True),))
    transition=dict(env_index=0,generation=1,action_counts={},assessment=dict(valid_connections=1,safe=True))
    rec.observe_transition(transition,frame)  # Joint can exist before dock's terminal ACK.
    executor._finish(executor._active['dock'],PrimitiveState.SUCCEEDED,.8,'DONE','complete')
    assert rec.summary()['success_criteria']['strong'] is True
    assert rec.summary()['success_criteria']['ideal'] is False
    frame.time_s=.833;rec.observe_transition(transition,frame)
    assert rec.summary()['success_criteria']['ideal'] is True
    assert rec.summary()['valid_connections'][0]['last_observed_timestamp_s']==.833


def test_observer_can_be_detached_before_cleanup_without_inventing_cutoff_terminals(tmp_path):
    rec=recorder(tmp_path);executor=executor_fixture();rec.attach_executor(executor,0,1)
    executor.submit(goal('live','module_a','LEFT'),0.);rec.censor_open(2.,'horizon')
    rec.detach_executor(executor)
    executor.cancel('live',2.)
    assert rec.summary()['goals'][0]['terminal_state'] is None


def test_missing_candidate_after_release_does_not_count_as_reopened(tmp_path):
    rec=recorder(tmp_path);executor=executor_fixture();cache=NativeAvailabilityCache()
    rec.attach_executor(executor,0,1);executor.submit(goal('a','module_a','LEFT'),0.)
    observe(rec,executor,cache);executor.cancel('a',.5)
    availability=capture(cache,executor)
    rec.observe_context(0,1,(),availability,.533)
    finding=rec.summary()['goals'][0]['mask_after_release']
    assert finding['reopened_count']==0 and finding['missing_from_catalog_count']==2


def test_monitor_displays_startup_error_tail_when_progress_not_created(tmp_path):
    from drl.MARL.c1_structural_physics_diagnostic import format_progress
    (tmp_path/'stdout.log').write_text('startup\n[Error] PhysX failed\n')
    assert 'PhysX failed' in format_progress(tmp_path)


def test_real_checkpoint_factory_is_frozen_cpu_actor_only(monkeypatch):
    from drl.MARL.c1_structural_physics_diagnostic import build_frozen_actor,configuration,parse_args,actor_digest
    from drl.MARL.c1_runner import build_observation_bridge
    from drl.MARL.c1_structural_physics_diagnostic import ROOT
    from drl.MARL.global_critic import GlobalCritic
    def forbidden(*args,**kwargs):pytest.fail('No optimizer or critic may be constructed')
    monkeypatch.setattr(torch.optim.Optimizer,'__init__',forbidden)
    monkeypatch.setattr(GlobalCritic,'from_repository_config',forbidden)
    config=configuration(parse_args(['--preflight','--device','cpu']))
    actor=build_frozen_actor(build_observation_bridge(ROOT),config)
    assert actor.presence_threshold_logit==4.5 and not actor.training
    assert all(not p.requires_grad and p.grad is None for p in actor.parameters())
    assert len(actor_digest(actor))==64


def test_failed_offline_run_keeps_actual_simulated_span_and_error_artifact(monkeypatch,tmp_path):
    """Main failure/cleanup contract with Kit substituted; no physical launch."""
    import sys
    from drl.MARL import c1_structural_physics_diagnostic as diag
    from drl.MARL import c1_runner,isaaclab_c1_runner
    args=diag.parse_args(['--headless','--device','cpu','--output',str(tmp_path)])
    config=dict(diag.configuration(args),source_sha256={},initial_train_seeds=[1,2,3,4])
    monkeypatch.setattr(diag,'preflight',lambda args:config)
    monkeypatch.setattr(c1_runner,'build_observation_bridge',lambda root:object())
    monkeypatch.setattr(isaaclab_c1_runner,'seed_native_training',lambda seed:None)
    monkeypatch.setattr(diag,'build_frozen_actor',lambda *a:torch.nn.Linear(1,1).requires_grad_(False))
    app=SimpleNamespace(close=lambda:None)
    monkeypatch.setitem(sys.modules,'isaaclab.app',SimpleNamespace(AppLauncher=lambda **kw:SimpleNamespace(app=app)))
    class Env:
        def __init__(self,cfg):
            self.cfg=cfg;self._sim_step_counter=0;self.physics_dt=1/240;self.seed_by_env=(1,2,3,4)
            self.physical_expert_backends=[];self.c1_runtime=SimpleNamespace(flush_open_episodes=lambda *a:None)
        def reset(self):pass
        def close(self):pass
        def _ensure_native_actuation_backends(self):pass
        def initialize_policy_bridge(self,**kw):return SimpleNamespace(provider=SimpleNamespace(feasibility_metrics=lambda:{}))
    monkeypatch.setitem(sys.modules,'drl.MARL.isaaclab_c1_env',SimpleNamespace(MSSRC1IsaacLabEnv=Env,make_c1_isaaclab_env_cfg=lambda **kw:SimpleNamespace()))
    monkeypatch.setattr(diag,'runtime_settings',lambda env:dict(render_enabled=False,has_gui=False,sensors=[]))
    def fail(env,*a,**kw):
        env._sim_step_counter=16
        raise RuntimeError('physical failure evidence')
    monkeypatch.setattr(diag,'rollout_frozen',fail)
    assert diag.run(args)==1
    result=json.loads((tmp_path/'summary.json').read_text())
    assert result['status']=='error' and 'physical failure evidence' in result['error']
    assert result['simulated_seconds_per_env']==pytest.approx(2/30)


def test_offline_shell_preserves_stdout_against_concurrent_invocations(tmp_path):
    """Execute a shell copy with /usr/bin/time replaced: NEVER invokes Isaac."""
    import subprocess
    from pathlib import Path
    from drl.MARL.c1_structural_physics_diagnostic import OUTPUT
    run_folder=tmp_path/'run';run_folder.mkdir()
    fake_time=tmp_path/'fake_time'
    fake_time.write_text('#!/usr/bin/env python3\nfrom pathlib import Path\nwith Path('+repr(str(run_folder/'calls'))+').open("a") as f:f.write("call\\n")\nprint("preserved output")\n')
    fake_time.chmod(0o755)
    barrier=tmp_path/'barrier.py'
    barrier.write_text('import os,time\nfrom pathlib import Path\np=Path('+repr(str(run_folder))+')\n(p/str(os.getpid())).touch()\nstart=time.monotonic()\nwhile len([x for x in p.iterdir() if x.name.isdecimal()])<2:\n if time.monotonic()-start>3:raise RuntimeError("barrier timeout")\n time.sleep(.01)\n')
    source=(OUTPUT/'run_offline.sh').read_text()
    source=source.replace('run_directory='+str(OUTPUT),'run_directory='+str(run_folder))
    source=source.replace('exec /usr/bin/time',f'python3 {barrier}\nexec {fake_time}')
    script=tmp_path/'offline.sh';script.write_text(source)
    processes=[subprocess.Popen(['bash',str(script)],stdout=subprocess.PIPE,stderr=subprocess.PIPE) for _ in range(2)]
    try:
        for process in processes:process.communicate(timeout=8)
    finally:
        for process in processes:
            if process.poll() is None:process.kill();process.wait()
    assert sorted(process.returncode for process in processes)==[0,1]
    assert (run_folder/'calls').read_text().splitlines()==['call']
    assert (run_folder/'stdout.log').read_text()=='preserved output\n'


def test_start_claim_evidence_reads_acquired_snapshot_not_a_new_preview(monkeypatch,tmp_path):
    rec=recorder(tmp_path);executor=executor_fixture();rec.attach_executor(executor,0,1)
    def forbidden(*a,**kw):pytest.fail('Acquired claims must come from native holder snapshot')
    monkeypatch.setattr(executor,'resource_claims_for',forbidden)
    executor.submit(goal('actual','module_a','LEFT'),.125)
    row=rec.summary()['goals'][0]
    acquired,=executor.active_resource_snapshot()
    assert row['claims']==dict(acquired.resource_modes)
