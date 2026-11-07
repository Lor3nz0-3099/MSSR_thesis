"""CPU-only viewer contracts: diagnostic topology must never control an action."""
import io
import json
from types import SimpleNamespace
import pytest
import torch
from drl.MARL.c1_structure import C1Connection


def target():
    return dict(nodes=[dict(node_id='r',attributes=dict(is_target_root=True)),
        dict(node_id='a'),dict(node_id='b')],edges=[
        dict(module_a_id='a',module_b_id='r',attributes=dict(face_a='BOTTOM',face_b='LEFT')),
        dict(module_a_id='b',module_b_id='r',attributes=dict(face_a='BOTTOM',face_b='RIGHT'))])


def classifier():
    from drl.MARL.c1_relation_diagnostics import TargetRelationDiagnostic
    return TargetRelationDiagnostic(target(),('smores_01','smores_02','smores_03'))


def test_classifier_five_labels_valid_joint_only_retry_and_no_fixed_assignment():
    c=classifier();pair=(('smores_02','BOTTOM'),('smores_01','LEFT'))
    before=torch.random.get_rng_state()
    assert c.classify(pair,(),1)['classification']=='TARGET_RELATION_UNSATISFIED'
    c.record_attempt(pair,1,'NO_COLLISION_FREE_STAGING_PATH',.5)
    result=c.classify(pair,(),1)
    assert result['classification']=='RETRY_SAME_RELATION'
    assert result['prior_attempts'][0]['failure_code']=='NO_COLLISION_FREE_STAGING_PATH'
    invalid=C1Connection('j',*pair[0],*pair[1],False)
    assert c.classify(pair,(invalid,),1)['classification']=='RETRY_SAME_RELATION'
    valid=C1Connection('j',*pair[0],*pair[1],True)
    assert c.classify(pair,(valid,),1)['classification']=='TARGET_RELATION_ALREADY_SATISFIED'
    assert c.classify((('smores_02','TOP'),('smores_01','LEFT')),(),1)['classification']=='NON_TARGET_OR_INCOMPATIBLE'
    assert c.classify((('unknown','BOTTOM'),('smores_01','LEFT')),(),1)['classification']=='UNKNOWN'
    assert c.classify(pair,(),2)['classification']=='TARGET_RELATION_UNSATISFIED'
    assert torch.equal(before,torch.random.get_rng_state())
    assert 'module_assignment' not in result


def test_classifier_unanchored_edge_ambiguity_is_unknown_and_keeps_both_graphs():
    c=classifier()
    result=c.classify((('smores_02','BOTTOM'),('smores_03','LEFT')),(),1)
    # With no connection to the root neither of these can take the occupied
    # anchored root slot, hence there is no compatible target edge.
    assert result['classification']=='NON_TARGET_OR_INCOMPATIBLE'
    doc=target();doc['nodes'].append(dict(node_id='c'))
    doc['edges'].append(dict(module_a_id='c',module_b_id='a',attributes=dict(face_a='BOTTOM',face_b='LEFT')))
    doc['edges'].append(dict(module_a_id='b',module_b_id='a',attributes=dict(face_a='BOTTOM',face_b='LEFT')))
    from drl.MARL.c1_relation_diagnostics import TargetRelationDiagnostic
    c=TargetRelationDiagnostic(doc,('smores_01','smores_02','smores_03','smores_04'))
    result=c.classify((('smores_03','BOTTOM'),('smores_02','LEFT')),(),1)
    assert result['classification']=='UNKNOWN' and result['ambiguous']
    assert result['target_topology']==doc and result['current_topology']==[]


def test_viewer_defaults_preflight_without_isaac_and_cli_bounds(tmp_path):
    from drl.MARL.c1_gui_viewer import parse_args,preflight
    args=parse_args(['--preflight','--device','cpu','--output',str(tmp_path)])
    cfg=preflight(args)
    assert cfg['task']=='navigation' and cfg['num_envs']==1 and cfg['headless'] is False
    assert cfg['presence_threshold_logit']==4.5 and cfg['sampling']=='stochastic'
    assert cfg['actor_only'] and cfg['runtime_mode']=='accelerated'
    assert cfg['physics_hz']==240 and cfg['decision_hz']==30
    assert cfg['optimizer_updates']==0 and cfg['training_memory_constructed'] is False
    assert cfg['initial_train_seed']==7033
    for task in ('stairs','gap','manipulation'):
        assert preflight(parse_args(['--preflight','--task',task,'--device','cpu']))['task']==task
    for argv in (['--steps','0'],['--status-seconds','.1'],['--task','other']):
        with pytest.raises(SystemExit):parse_args(argv)


def test_sparse_observer_native_events_timeout_and_zero_per_tick_serialization(tmp_path,monkeypatch):
    from drl.MARL.c1_viewer_events import ViewerEvents
    from drl.MARL.tests.test_c1_feasibility import executor_fixture
    from drl.MARL.tests.test_align_face_claims import goal,capture
    from drl.MARL.c1_feasibility_cache import NativeAvailabilityCache
    from drl.MARL.policy import bind_marl_candidates
    from drl.MARL.tests.test_align_face_claims import route
    from smores_ep.primitives.model import PrimitiveState,PrimitiveStatus
    out=io.StringIO();rec=ViewerEvents(tmp_path,'navigation',classifier=classifier(),console=out)
    executor=executor_fixture();rec.attach_executor(executor,0,1)
    cache=NativeAvailabilityCache();records=bind_marl_candidates(routing=route(),presence_logits=torch.zeros(3))
    def observe(stamp):rec.observe_context(0,1,records,capture(cache,executor),stamp)
    observe(0.);before=torch.random.get_rng_state()
    executor.submit(goal('a','module_a','LEFT'),.125);observe(.2)
    def running(runtime,now):
        return {},executor._make_status(runtime.goal,PrimitiveState.RUNNING,now,phase='staging',code='RUNNING',message='')
    monkeypatch.setattr(executor,'_align_faces',running)
    monkeypatch.setattr(PrimitiveStatus,'to_dict',lambda self:pytest.fail('per-tick serialization'))
    executor.step(.3);executor.step(.4)
    executor._finish(executor._active['a'],PrimitiveState.FAILED,.625,'CONTACT_TIMEOUT','contact not reached')
    observe(.633);rec.flush();rec.detach_executor(executor);rec.close()
    rows=[json.loads(s) for s in (tmp_path/'events.jsonl').read_text().splitlines()]
    labels=[r['event'] for r in rows]
    assert labels.count('ALIGN START')==1 and labels.count('ALIGN TIMEOUT')==1
    assert 'CLAIMS ACQUIRED' in labels and 'CLAIMS RELEASED' in labels and 'MASK REOPEN' in labels
    terminal=next(r for r in rows if r['event']=='ALIGN TIMEOUT')
    assert terminal['timestamp_s']==.625 and terminal['failure_code']=='CONTACT_TIMEOUT'
    assert terminal['candidate_id']=='structural|align_faces|module_a|module_a:BOTTOM>module_b:LEFT'
    assert 'module_a:BOTTOM -> module_b:LEFT' in out.getvalue()
    assert torch.equal(before,torch.random.get_rng_state())


def test_camera_bounds_cover_all_modules():
    from drl.MARL.c1_gui_viewer import camera_for_positions
    positions=[(0.,0.,.1),(2.,1.,.1),(-1.,-.5,.1)]
    eye,target=camera_for_positions(positions)
    assert target==pytest.approx((.5,.25,.1))
    assert eye[2]>2. and eye[0]>target[0]


def test_single_env_cpu_rollout_has_no_training_or_critic_and_uses_real_provider(tmp_path,monkeypatch):
    from collections import deque
    from drl.MARL.bc_v4_runtime_observation import BCV4RuntimeObservation
    from drl.MARL.native_policy_bridge import build_native_policy_bridge
    from drl.MARL.tests.test_native_policy_bridge import NativeEnv
    from drl.MARL.c1_initialization import PolicyControlledHandoff
    from drl.MARL.tests.test_c1_feasibility import executor_fixture,routing_fixture,simple_output
    from drl.MARL.c1_viewer_events import ViewerEvents
    from drl.MARL.c1_gui_viewer import rollout_viewer
    from drl.MARL.c1_structure import C1Frame
    executor=executor_fixture();routing=routing_fixture();env=NativeEnv();env.num_envs=1
    env.cfg=SimpleNamespace(initialization_mode='loose_modules');env.generation=[1]
    env.native_dispatcher=SimpleNamespace(executor_provider=lambda i:executor)
    env._sim_step_counter=0;env.physics_dt=1/240
    env.c1_runtime=SimpleNamespace(recent_transitions=deque(maxlen=64),evaluators={})
    for candidate in routing['resource_candidates']:
        candidate.action=SimpleNamespace(action_id='drive');candidate.mode=SimpleNamespace(mode_id='velocity')
    decoded=[]
    def prepare(i):
        return BCV4RuntimeObservation(SimpleNamespace(stamp=env.stamp),{'stamp':env.stamp},routing,{'inputs':{},'routing':[routing]},None)
    def decode(**kwargs):
        decoded.append(kwargs);return ()
    handoff=PolicyControlledHandoff();handoff.activate_policy()
    native=build_native_policy_bridge(env,observation_bridge=SimpleNamespace(normalizers={}),task_context_provider=None,
        binding=SimpleNamespace(decode_marl_action=decode),handoffs=[handoff],
        hard_mask_builder=lambda a,r:torch.zeros(6,dtype=torch.bool),placeholder_action=lambda i,c:{'m0':torch.zeros(1)},
        observation_source=SimpleNamespace(num_envs=1,prepare_env=prepare,reset=lambda:None),actor_only=True)
    calls=[]
    def forward(**kwargs):
        assert kwargs['lazy_payloads'] is True and not torch.is_grad_enabled()
        calls.append(kwargs['graph']['stamp'])
        logits=torch.full((6,),-100.);logits[0]=100.
        return dict(simple_output(logits),prediction_template={'presence_logits':logits})
    def step(carrier):
        assert carrier['m0'].shape==(1,1)
        env._sim_step_counter+=8;env.stamp+=1/30
        env.c1_runtime.recent_transitions.append(dict(env_index=0,generation=1,action_counts={},assessment=dict(valid_connections=0,safe=True),timestamp_s=env.stamp))
        env.c1_runtime.evaluators[0]=(1,SimpleNamespace(last_frame=C1Frame(env.stamp,(),())))
        return {},{'m0':torch.zeros(1)},None,None,None
    env.step=step
    def forbidden(*a,**k):pytest.fail('Training/critic invoked')
    monkeypatch.setattr(torch.optim.Optimizer,'__init__',forbidden)
    monkeypatch.setattr(torch.autograd,'backward',forbidden)
    native.provider.current_critic_graphs=forbidden
    rec=ViewerEvents(tmp_path,'navigation',classifier=classifier(),console=io.StringIO())
    result=rollout_viewer(env,SimpleNamespace(forward_all_owners=forward),native.provider,rec,('module_c',),
        steps=3,max_wall_seconds=60.,accelerated=True)
    rec.close()
    assert result['decisions_completed']==3 and result['cutoff_reason']=='horizon_reached'
    assert len(decoded)==3 and calls[0]!=calls[1]!=calls[2]
    assert all(c.critic_graph is None for c in native.provider.current_action_contexts())


def test_actor_only_wrapper_reset_epoch_and_default_training_critic():
    from drl.MARL.c1_gui_viewer import ActorOnlyObservationBridge
    from drl.MARL.tests.test_bc_v4_runtime_observation import make_bridge
    from drl.IL.tests.bc_v1 import test_live_observation as fixture
    from drl.IL.bc_v1.bc_v1_contract import TaskContext
    bridge=make_bridge();epoch=[1];viewer=ActorOnlyObservationBridge(bridge,lambda:epoch[0])
    args=dict(task_context=TaskContext('navigation'));raw=fixture.snapshot(n=8)
    a=viewer.prepare_snapshot(raw,**args);b=viewer.prepare_snapshot(raw,**args)
    assert a.privileged_graph is None and b.privileged_graph is None
    assert a.routing_snapshot['runtime_candidate_layout_token']==b.routing_snapshot['runtime_candidate_layout_token']
    epoch[0]+=1;c=viewer.prepare_snapshot(raw,**args)
    assert c.routing_snapshot['runtime_candidate_layout_token']!=b.routing_snapshot['runtime_candidate_layout_token']
    assert bridge.prepare_snapshot(raw,**args).privileged_graph is not None


def test_slow_status_is_rate_limited_and_valid_connections_are_joint_valid_only(tmp_path):
    from drl.MARL.c1_viewer_events import ViewerEvents
    from drl.MARL.c1_structure import C1Frame
    rec=ViewerEvents(tmp_path,'navigation',classifier=classifier(),console=io.StringIO(),status_seconds=2.)
    row=dict(env_index=0,generation=1,action_counts={},assessment=dict(valid_connections=1,safe=True),timestamp_s=1.)
    connections=(C1Connection('yes','smores_02','BOTTOM','smores_01','LEFT',True),
                 C1Connection('no','smores_03','BOTTOM','smores_01','RIGHT',False))
    rec.observe_transition(row,C1Frame(1.,(),connections));rec.observe_transition(row,C1Frame(1.1,(),connections))
    first=rec.slow_status(1.1,now=10.);assert first['connections']==1
    assert rec.slow_status(1.2,now=10.1) is None
    assert rec.slow_status(1.3,now=12.) is not None
    rec.close()
    rows=[json.loads(s) for s in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert [r['event'] for r in rows]==['VALID CONNECTION']


def test_viewer_reset_discards_previous_topology_without_false_mask_reopen(tmp_path):
    from drl.MARL.c1_viewer_events import ViewerEvents
    from drl.MARL.c1_structure import C1Frame
    from drl.MARL.tests.test_align_face_claims import route,capture
    from drl.MARL.tests.test_c1_feasibility import executor_fixture
    from drl.MARL.c1_feasibility_cache import NativeAvailabilityCache
    from drl.MARL.policy import bind_marl_candidates
    rec=ViewerEvents(tmp_path,'navigation',classifier=classifier(),console=io.StringIO())
    records=bind_marl_candidates(routing=route(),presence_logits=torch.zeros(3))
    availability=capture(NativeAvailabilityCache(),executor_fixture())
    rec.observe_context(0,1,records,availability,0.)
    joint=C1Connection('old','smores_02','BOTTOM','smores_01','LEFT',True)
    rec.observe_transition(dict(env_index=0,generation=1,action_counts={},assessment=dict(valid_connections=1,safe=True)),C1Frame(.1,(),(joint,)))
    assert rec.latest_connections
    rec.observe_context(0,2,records,availability,.2)
    assert rec.latest_connections==()
    rec.close()
    events=[json.loads(s)['event'] for s in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert events.count('RESET')==1 and 'MASK REOPEN' not in events


def test_existing_unmarked_output_directory_never_overwrites_artifacts(tmp_path,monkeypatch):
    from drl.MARL import c1_gui_viewer as viewer
    folder=tmp_path/'existing';folder.mkdir()
    old={'run_config.json':'old config','source_sha256.json':'old hashes','runtime.log':'old log'}
    for name,content in old.items():(folder/name).write_text(content)
    monkeypatch.setattr(viewer,'preflight',lambda args:{'source_sha256':{}})
    with pytest.raises(FileExistsError):viewer.run(viewer.parse_args(['--device','cpu','--output',str(folder)]))
    assert {name:(folder/name).read_text() for name in old}==old
    assert not (folder/'RUN_STARTED.json').exists()


@pytest.mark.parametrize('failure',['censor','status','detach','recorder_close','digest','reset_records','env_close'])
def test_finalization_failure_always_closes_both_resources_preserves_primary_error(tmp_path,failure):
    from drl.MARL.c1_gui_viewer import finalize_viewer
    calls=[]
    def operation(name,result=None):
        calls.append(name)
        if name==failure:raise OSError('diagnostic/storage failure')
        return result
    rec=SimpleNamespace(censor_open=lambda *a:operation('censor'),slow_status=lambda *a,**k:operation('status'),
        detach_environment=lambda:operation('detach'),close=lambda:operation('recorder_close'),event_counts={})
    env=SimpleNamespace(_sim_step_counter=24,physics_dt=1/240,reset_records=[],close=lambda:operation('env_close'))
    app=SimpleNamespace(close=lambda:operation('app_close'))
    actor=SimpleNamespace(parameters=lambda:())
    result=dict(status='error',error='original rollout failure')
    finalize_viewer(env=env,application=app,actor=actor,recorder=rec,before='sha',sim_start=0.,result=result,folder=tmp_path,
        write_json=lambda *a:operation('reset_records'),actor_digest=lambda a:operation('digest','sha'))
    assert calls.count('env_close')==calls.count('app_close')==1
    assert calls.index('env_close')<calls.index('app_close')
    assert result['error']=='original rollout failure' and result['status']=='error'
    assert result['finalization_errors']


def test_compact_camera_reuses_validated_teleop_radius():
    import math
    from drl.MARL.c1_gui_viewer import TELEOP_CAMERA_RADIUS_M, camera_for_positions
    positions=[(-.18,-.18,.03),(.18,.18,.03)]
    eye,target=camera_for_positions(positions)
    radius=math.sqrt(sum((eye[i]-target[i])**2 for i in range(3)))
    assert radius == pytest.approx(TELEOP_CAMERA_RADIUS_M)
    assert TELEOP_CAMERA_RADIUS_M == pytest.approx(1.1885363940578346)
