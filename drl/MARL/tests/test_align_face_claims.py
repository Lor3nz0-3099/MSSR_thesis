"""Native connector-level exclusion with independent stationary-root readers."""
from types import SimpleNamespace
import pytest
import torch

from drl.MARL.tests.test_c1_feasibility import executor_fixture,align_goal
from drl.MARL.c1_feasibility import capture_native_availability
from drl.MARL.c1_feasibility_cache import NativeAvailabilityCache
from smores_ep.primitives.model import PrimitiveGoal,PrimitiveState


def goal(name,source,target_face):
    return align_goal(name,source,'module_b','BOTTOM',target_face)


def route():
    from drl.IL.bc_v1.bc_v1_contract import ModuleRef,ConnectorRef
    modules=tuple(ModuleRef(m) for m in ('module_a','module_b','module_c','module_d'))
    rows=[]
    for source,target_face in ((0,'LEFT'),(2,'RIGHT'),(3,'LEFT')):
        ports=(ConnectorRef(modules[source],'BOTTOM'),ConnectorRef(modules[1],target_face))
        members=frozenset(p.module for p in ports)
        rows.append((SimpleNamespace(action_id='align_faces'),SimpleNamespace(initiator=modules[source],
            connectors=ports,required_scope_members=members,scope_members=members)))
    return dict(module_refs=modules,resource_candidates=(),structural_bindings=tuple(rows),argument_domains=())


def capture(cache,executor,generation=1):
    args=dict(routing=route(),graph={},hard_mask=torch.zeros(3,dtype=torch.bool),executor=executor)
    value=cache.capture(**args,generation=generation)
    assert value==capture_native_availability(**args)
    return value


def test_distinct_root_faces_are_admitted_at_same_time_with_shared_locomotion():
    executor=executor_fixture()
    assert executor.submit(goal('left','module_a','LEFT'),0.).state is PrimitiveState.ACCEPTED
    assert executor.submit(goal('right','module_c','RIGHT'),0.).state is PrimitiveState.ACCEPTED
    assert len(executor.active_goals)==2
    assert executor._resource_owners['locomotion:module_b']=={'left':'shared','right':'shared'}
    assert executor._resource_owners['connector:module_b:LEFT']=={'left':'exclusive'}
    assert executor._resource_owners['connector:module_b:RIGHT']=={'right':'exclusive'}
    assert 'internal_motion:module_b' not in executor._resource_owners


@pytest.mark.parametrize('face',['LEFT','RIGHT','TOP','BOTTOM'])
def test_same_target_face_rejected_with_exact_connector_reason_and_no_version_change(face):
    executor=executor_fixture();a=goal('a','module_a',face);c=goal('c','module_c',face)
    assert executor.submit(a,0.).state is PrimitiveState.ACCEPTED
    version=executor.active_claim_version
    modes=dict(executor.resource_claims_for(a))
    assert modes[f'connector:module_b:{face}']=='exclusive'
    rejected=executor.submit(c,.1)
    assert rejected.state is PrimitiveState.REJECTED and rejected.code=='RESOURCE_BUSY'
    assert f'connector:module_b:{face} owned by a' in rejected.message
    assert executor.active_claim_version==version and len(executor.active_goals)==1
    assert executor.check_admission(c,.1).code=='RESOURCE_BUSY'


def test_source_control_already_blocks_align_and_dock_using_free_source_face():
    executor=executor_fixture();executor.submit(goal('a','module_a','LEFT'),0.)
    modes=dict(executor.resource_claims_for(goal('a','module_a','LEFT')))
    assert modes['locomotion:module_a']=='exclusive'
    assert 'connector:module_a:BOTTOM' not in modes  # Deliberate minimal source decision.
    for primitive in ('align_faces','dock'):
        other=PrimitiveGoal.from_dict(dict(goal_id=primitive,primitive=primitive,
            module_ids=['module_c','module_a'],parameters=dict(face_a='BOTTOM',face_b='BOTTOM')))
        rejected=executor.check_admission(other,.1)
        assert rejected.code=='RESOURCE_BUSY' and 'locomotion:module_a owned by a' in rejected.message


@pytest.mark.parametrize('primitive',['dock','undock'])
def test_target_face_claim_uses_existing_dock_undock_connector_vocabulary(primitive):
    executor=executor_fixture();executor.submit(goal('align','module_a','LEFT'),0.)
    other=PrimitiveGoal.from_dict(dict(goal_id='other',primitive=primitive,module_ids=['module_c','module_b'],
        parameters=dict(face_a='BOTTOM',face_b='LEFT')))
    modes=dict(executor.resource_claims_for(other))
    assert modes['connector:module_b:LEFT']=='exclusive'
    assert 'connector:module_b:LEFT' in executor.resource_claim_conflicts(modes)
    assert executor.check_admission(other,.1).code=='RESOURCE_BUSY'


@pytest.mark.parametrize('terminal',['succeeded','failed','canceled','timeout'])
def test_release_reopens_face_without_releasing_other_root_face(terminal,monkeypatch):
    executor=executor_fixture();cache=NativeAvailabilityCache()
    first=capture(cache,executor);assert first.candidate_mask==(False,False,False)
    executor.submit(goal('left','module_a','LEFT'),0.)
    held=capture(cache,executor)
    assert held.candidate_mask==(True,False,True)
    assert capture(cache,executor)==held  # BUSY hit must retain exact connector support.
    executor.submit(goal('right','module_c','RIGHT'),.01)
    both=capture(cache,executor);assert both.candidate_mask==(True,True,True)
    before=executor.active_claim_version
    if terminal=='canceled':executor.cancel('left',.1)
    elif terminal=='timeout':
        # Exercise the real timeout branch without executing physical alignment.
        executor._active['left'].started_at_s=-31.
        monkeypatch.setattr(executor,'_align_faces',lambda runtime,now: ({},executor._make_status(
            runtime.goal,PrimitiveState.RUNNING,now,phase='unit_no_physics',code='RUNNING',message='unexpired sibling')))
        statuses={status.goal_id:status for status in executor.step(.1).statuses}
        assert statuses['left'].code=='TIMEOUT' and statuses['right'].state is PrimitiveState.RUNNING
    else:executor._finish(executor._active['left'],PrimitiveState(terminal),.1,'TEST_END','terminal test')
    assert executor.active_claim_version==before+1
    assert executor._resource_owners['connector:module_b:RIGHT']=={'right':'exclusive'}
    assert executor._resource_owners['locomotion:module_b']=={'right':'shared'}
    assert 'connector:module_b:LEFT' not in executor._resource_owners
    reopened=capture(cache,executor)
    assert reopened.candidate_mask==(False,True,False)
    assert held.candidate_mask==(True,False,True)  # Historical PPO support immutable.
    assert executor.submit(goal('replacement','module_d','LEFT'),.2).state is PrimitiveState.ACCEPTED
    assert cache.metrics()['mask_cache_hits']>=1 and cache.metrics()['mask_recomputes']==4


def test_connector_claim_reset_epoch_and_parallel_env_isolation():
    a,b=executor_fixture(),executor_fixture();ca,cb=NativeAvailabilityCache(),NativeAvailabilityCache()
    a.submit(goal('left','module_a','LEFT'),0.);b.submit(goal('left','module_a','LEFT'),0.)
    assert capture(ca,a).candidate_mask[2] and capture(cb,b).candidate_mask[2]
    # Existing selected reset contract: cancel old goals and create a fresh executor.
    for active in tuple(a.active_goals):a.cancel(active.goal_id,.1)
    fresh=executor_fixture()
    assert not capture(ca,fresh,generation=2).candidate_mask[2]
    assert capture(cb,b).candidate_mask[2]
    assert ca.metrics()['claim_epoch_changes']==1
    assert cb.metrics()['mask_cache_hits']==1 and cb.metrics()['mask_recomputes']==1
    assert not a._resource_owners and len(b.active_goals)==1


def test_different_faces_do_not_override_actual_source_locomotion_conflict():
    executor=executor_fixture();executor.submit(goal('left','module_a','LEFT'),0.)
    repeated_source=goal('right','module_a','RIGHT')
    rejected=executor.check_admission(repeated_source,.1)
    assert rejected.code=='RESOURCE_BUSY' and 'locomotion:module_a owned by left' in rejected.message


def test_policy_busy_mask_keeps_other_face_executable_without_conflict_penalty():
    from drl.IL.bc_v1.bc_v1_contract import ActionChoice
    from drl.IL.bc_v1.live_policy import smores_claims
    from drl.MARL.policy import bind_marl_candidates,sample_marl_action_bundle,StructuralPayloadView
    from drl.MARL.action_projection import ProjectionCandidate,project_executable_action_set
    from drl.MARL.smores_runtime_binding import DecodedSMORESAction
    from drl.MARL.runtime_signals import collect_runtime_signals
    from drl.MARL.tests.test_smores_runtime_binding import _native_dispatcher_fixture
    executor=executor_fixture();executor.submit(goal('left','module_a','LEFT'),0.)
    available=capture(NativeAvailabilityCache(),executor)
    records=bind_marl_candidates(routing=route(),presence_logits=torch.full((3,),20.),presence_threshold_logit=4.5)
    action=sample_marl_action_bundle(records=records,resource_payloads={},structural_payloads={i:StructuralPayloadView() for i in range(3)},
        hard_mask=available.mask_tensor(),resource_log_std=torch.tensor(-4.),structural_log_std=torch.tensor(-4.),deterministic=True)
    assert action.presence.tolist()==[0.,1.,0.]  # Other LEFT candidate is masked, RIGHT survives.
    spec,binding=route()['structural_bindings'][1]
    choice=ActionChoice(spec.action_id,binding.initiator,participants=tuple(p.module for p in binding.connectors),connectors=binding.connectors)
    decoded=DecodedSMORESAction(choice,dict(command='align_faces',module_ids=['module_c','module_b'],parameters=dict(face_a='BOTTOM',face_b='RIGHT')))
    candidate=ProjectionCandidate(records[1].candidate_id,1,20.,('module_c',),choice,smores_claims(choice),decoded)
    projection=project_executable_action_set((candidate,),raw_sampled_action=action)
    dispatcher,_,_=_native_dispatcher_fixture();dispatcher.executor_provider=lambda i:executor
    dispatcher.primitive_goal_factory=PrimitiveGoal.from_dict;dispatcher.accepted_state=PrimitiveState.ACCEPTED
    dispatcher.dispatch_projected(env_index=0,projection=projection)
    outcome=dispatcher.proposal_outcome(0)
    assert outcome.executed_candidate_ids==(candidate.candidate_id,) and not outcome.physical_drops
    assert len(executor.active_goals)==2
    signals=collect_runtime_signals(owner_ids=('module_a','module_b','module_c','module_d'),delta_seconds=1/30,proposal_outcome=outcome)
    assert not signals.signals.conflict_owner_ids and not signals.signals.invalid_owner_ids
    assert outcome.projection.raw_sampled_action is action
