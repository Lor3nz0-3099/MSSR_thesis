"""Native claim versioning, exact BUSY cache support and lazy evidence contracts."""
from dataclasses import asdict,replace
from types import SimpleNamespace
import json

import pytest
import torch

from drl.MARL.tests.test_c1_feasibility import executor_fixture,align_goal,routing_fixture,gravity_fixture
from drl.MARL.c1_feasibility import capture_native_availability


def inputs(executor,route=None,graph=None,hard_mask=None,generation=1):
    route=route or routing_fixture()
    return dict(executor=executor,routing=route,graph=graph or {},
        hard_mask=hard_mask if hard_mask is not None else torch.zeros(len(route['resource_candidates'])+len(route['structural_bindings']),dtype=torch.bool),generation=generation)


def oracle(args):
    return capture_native_availability(**{k:v for k,v in args.items() if k!='generation'})


def cache():
    from drl.MARL.c1_feasibility_cache import NativeAvailabilityCache
    return NativeAvailabilityCache()


@pytest.mark.parametrize('terminal',['succeeded','failed','canceled'])
def test_native_version_acquisition_release_and_noop(terminal):
    from smores_ep.primitives.model import PrimitiveState
    executor=executor_fixture();assert executor.active_claim_version==0
    executor.submit(align_goal(),0.);assert executor.active_claim_version==1
    held=executor.active_resource_snapshot();assert executor.active_resource_snapshot() is held
    executor.submit(align_goal(),.1)  # duplicate admission
    assert executor.active_claim_version==1
    if terminal=='canceled':executor.cancel('active',.2)
    else:executor._finish(executor._active['active'],PrimitiveState(terminal),.2,'END','test terminal')
    assert executor.active_claim_version==2
    assert not executor.active_resource_snapshot() and held
    executor.cancel('active',.3);assert executor.active_claim_version==2


def test_hit_skips_claim_resolver_and_precompute_is_once(monkeypatch):
    executor=executor_fixture();state=cache();args=inputs(executor)
    first=state.capture(**args)
    def forbidden(*a,**kw):raise AssertionError('cache hit rebuilt claims or BUSY checks')
    monkeypatch.setattr(executor,'resource_claims_for',forbidden)
    monkeypatch.setattr(executor,'resource_claim_conflicts',forbidden)
    second=state.capture(**args)
    assert second==first
    stats=state.metrics()
    assert stats['mask_cache_hits']==1 and stats['mask_cache_misses']==stats['mask_recomputes']==1
    assert stats['static_claim_precompute_build_count']==1
    assert stats['static_candidate_claim_build_count']==6
    assert stats['feasibility_masking_s']>0


@pytest.mark.parametrize('terminal',['succeeded','failed','canceled'])
def test_start_and_release_recompute_and_reopen_identical_uncached_support(terminal):
    from smores_ep.primitives.model import PrimitiveState
    executor=executor_fixture();state=cache();args=inputs(executor)
    empty=state.capture(**args);assert empty==oracle(args)
    executor.submit(align_goal(),0.)
    held=state.capture(**args);assert held==oracle(args)
    assert held.candidate_mask==(True,True,False,False,True,False)
    assert state.capture(**args)==held
    if terminal=='canceled':executor.cancel('active',.1)
    else:executor._finish(executor._active['active'],PrimitiveState(terminal),.1,'END','test')
    reopened=state.capture(**args);assert reopened==empty==oracle(args)
    assert held.candidate_mask[4]  # Old PPO snapshot remains immutable.
    stats=state.metrics()
    assert stats['mask_recomputes']==3 and stats['static_claim_precompute_build_count']==1
    assert stats['active_claim_version_changes']==2


def test_reset_new_executor_same_version_and_generation_epoch_do_not_reuse_old_busy():
    old=executor_fixture();old.submit(align_goal(),0.);state=cache()
    held=state.capture(**inputs(old));assert held.candidate_mask[4]
    new=executor_fixture();new.submit(align_goal('different','module_c','module_d'),0.)
    assert new.active_claim_version==old.active_claim_version
    fresh=state.capture(**inputs(new,generation=2))
    assert fresh==oracle(inputs(new,generation=2)) and not fresh.candidate_mask[4] and fresh.candidate_mask[5]
    assert state.metrics()['claim_epoch_changes']==1


def test_parallel_caches_isolate_native_versions():
    a,b=executor_fixture(),executor_fixture();ca,cb=cache(),cache()
    ca.capture(**inputs(a));cb.capture(**inputs(b));a.submit(align_goal(),0.)
    assert ca.capture(**inputs(a)).candidate_mask[4]
    assert not cb.capture(**inputs(b)).candidate_mask[4]
    assert cb.metrics()['mask_cache_hits']==1 and cb.metrics()['mask_recomputes']==1


def test_fresh_dynamic_base_mask_changes_even_when_busy_cache_hits():
    executor=executor_fixture();state=cache();args=inputs(executor)
    first=state.capture(**args)
    dynamic=torch.tensor([False,False,True,False,False,False])
    changed=state.capture(**{**args,'hard_mask':dynamic})
    assert changed.candidate_mask[2] and not first.candidate_mask[2]
    assert changed.reasons[2].code=='masked_unavailable'
    assert changed==oracle({**args,'hard_mask':dynamic})
    dynamic[:]=False
    assert state.capture(**args)==first
    assert state.metrics()['mask_cache_hits']==2


def test_optional_layout_and_claim_changes_never_reuse_stale_support():
    executor=executor_fixture();executor.submit(align_goal(),0.)
    route,graph,_=gravity_fixture();state=cache();args=inputs(executor,route,graph)
    first=state.capture(**args);assert first==oracle(args) and first.optional_factors
    second=state.capture(**args);assert second==first
    changed={k:v.clone() for k,v in graph['structural'].items()}
    changed['scope_members']=torch.tensor([3,1,2,0])
    args2={**args,'graph':{'structural':changed}}
    moved=state.capture(**args2);assert moved==oracle(args2)
    assert moved.scope_blocked!=first.scope_blocked
    executor.cancel('active',.1)
    free=state.capture(**args2);assert free==oracle(args2) and not free.optional_factors
    assert state.metrics()['static_claim_precompute_build_count']==2


def test_static_endpoint_change_with_same_candidate_count_invalidates_layout():
    from drl.IL.bc_v1.bc_v1_contract import ConnectorRef
    executor=executor_fixture();executor.submit(align_goal(),0.)
    route=routing_fixture();state=cache();state.capture(**inputs(executor,route))
    action,binding=route['structural_bindings'][1]
    route['structural_bindings'][1]=(action,SimpleNamespace(**{**vars(binding),'connectors':
        (ConnectorRef(route['module_refs'][0],'LEFT'),ConnectorRef(route['module_refs'][3],'BOTTOM'))}))
    args=inputs(executor,route)
    assert state.capture(**args)==oracle(args)
    assert state.metrics()['static_claim_precompute_build_count']==2


def test_lazy_mask_detail_builds_only_bounded_payloads_and_preserves_exact_order(monkeypatch):
    from drl.MARL.c1_action_events import C1ActionEvents
    from drl.MARL.policy import MARLCandidateRecord
    executor=executor_fixture();route,graph,_=gravity_fixture();executor.submit(align_goal(),0.)
    availability=oracle(inputs(executor,route,graph))
    records=tuple(MARLCandidateRecord(i,'resource' if i<4 else 'structural',
        f'resource|m{i}|locomotion|drive' if i<4 else 'structural|gravity_settle|module_c|none','module_c',(),torch.tensor(9.),4.5) for i in range(5))
    recorder=C1ActionEvents(masked_detail_limit=1)
    original=recorder._score;calls=[]
    def score(value):calls.append(value);return original(value)
    monkeypatch.setattr(recorder,'_score',score)
    before=torch.random.get_rng_state()
    rows=recorder.mask_events(records=records,availability=availability,action=SimpleNamespace(presence=torch.zeros(5)),
        identity=dict(env_index=0,generation=1,task='navigation',seed=7017,timestamp_s=0.))
    assert torch.equal(before,torch.random.get_rng_state())
    assert len(rows)==len(calls)==1
    assert recorder.totals['masked_busy_count']==2
    assert recorder.totals['masked_optional_membership_count']==4
    assert rows[0]['candidate_id']=='resource|m0|locomotion|drive'
    assert recorder.summary()['mask_detail_payload_build_count']==1


def test_zero_detail_builds_no_scores_or_payloads_but_keeps_all_counts(monkeypatch):
    from drl.MARL.c1_action_events import C1ActionEvents
    from drl.MARL.policy import MARLCandidateRecord
    executor=executor_fixture();executor.submit(align_goal(),0.);availability=oracle(inputs(executor))
    records=tuple(MARLCandidateRecord(i,'resource',str(i),'module_c',(),torch.tensor(0.),4.5) for i in range(6))
    recorder=C1ActionEvents(masked_detail_limit=0)
    monkeypatch.setattr(recorder,'_score',lambda *a:pytest.fail('unemitted score built'))
    assert recorder.mask_events(records=records,availability=availability,action=SimpleNamespace(presence=torch.zeros(6)),
        identity=dict(env_index=0,generation=1,task='navigation',seed=7017,timestamp_s=0.))==[]
    assert recorder.totals['masked_busy_count']==3


def test_native_provider_reuses_per_env_cache_and_reports_metrics_without_observation_features():
    from drl.MARL.tests.test_native_policy_bridge import make_bridge
    bridge,env,_,_=make_bridge();route=routing_fixture();executors=[executor_fixture(),executor_fixture()]
    prepared=bridge.provider.observation_source.prepare_env(0)
    bridge.provider.observation_source=SimpleNamespace(num_envs=2,prepare_env=lambda i:replace(prepared,routing_snapshot=route),reset=lambda:None)
    bridge.provider._hard_mask_builder=lambda g,r:torch.zeros(6,dtype=torch.bool)
    env.cfg=SimpleNamespace(initialization_mode='loose_modules');env.generation=[1,1]
    env.native_dispatcher=SimpleNamespace(executor_provider=lambda i:executors[i])
    bridge.provider.reset();first=bridge.provider.current_action_contexts()
    bridge.provider.reset();second=bridge.provider.current_action_contexts()
    executors[0].submit(align_goal(),0.)
    bridge.provider.reset();third=bridge.provider.current_action_contexts()
    assert first[0].availability_snapshot==second[0].availability_snapshot
    assert third[0].hard_mask[4] and not third[1].hard_mask[4]
    assert set(third[0].actor_graph)==set(prepared.actor_graph)
    metrics=bridge.provider.feasibility_metrics()
    assert metrics['totals']['mask_cache_hits']==3 and metrics['totals']['mask_cache_misses']==3
    assert metrics['by_env']['0']['active_claim_version_changes']==1
    assert metrics['by_env']['1']['mask_recomputes']==1


def test_cache_metrics_propagate_to_native_transition_and_action_summary(tmp_path):
    from drl.MARL.c1_action_events import C1ActionEvents
    from drl.MARL.c1_native_runtime import NativeC1RewardRuntime
    from drl.MARL.tests.test_c1_native_runtime import env_fixture
    from drl.MARL.tests.test_c1_structure import frame
    env=env_fixture();env.c1_action_events=C1ActionEvents(tmp_path)
    metrics=dict(totals=dict(mask_cache_hits=8,mask_cache_misses=1,mask_recomputes=1,
        active_claim_version_changes=0,static_claim_precompute_build_count=1,feasibility_masking_s=.125),by_env={'0':{}})
    env.runtime_context_provider=SimpleNamespace(feasibility_metrics=lambda:metrics)
    NativeC1RewardRuntime(evidence_directory=tmp_path).evaluate(env,0,frame(edges=0))
    transition=json.loads((tmp_path/'transitions.jsonl').read_text())
    assert transition['feasibility_cache']==metrics
    summary=json.loads((tmp_path/'action_summary.json').read_text())
    assert summary['mask_cache_hits']==8 and summary['feasibility_masking_s']==.125


def test_cached_support_disjoint_admission_does_not_reintroduce_executor_busy():
    from drl.MARL.tests.test_smores_runtime_binding import _native_dispatcher_fixture
    from drl.IL.bc_v1.bc_v1_contract import ActionChoice
    from drl.IL.bc_v1.live_policy import smores_claims
    from drl.MARL.smores_runtime_binding import DecodedSMORESAction
    from drl.MARL.action_projection import ProjectionCandidate,project_executable_action_set
    from smores_ep.primitives.model import PrimitiveGoal,PrimitiveState
    executor=executor_fixture();executor.submit(align_goal(),0.)
    route=routing_fixture();state=cache();args=inputs(executor,route)
    state.capture(**args);available=state.capture(**args)
    assert available.candidate_mask[4] and not available.candidate_mask[5]
    spec,binding=route['structural_bindings'][1]
    choice=ActionChoice(spec.action_id,binding.initiator,participants=tuple(p.module for p in binding.connectors),connectors=binding.connectors)
    decoded=DecodedSMORESAction(choice,dict(command='align_faces',module_ids=['module_c','module_d'],parameters=dict(face_a='TOP',face_b='BOTTOM')))
    candidate=ProjectionCandidate('structural|align_faces|module_c|ports',5,9.,('module_c',),choice,smores_claims(choice),decoded)
    dispatcher,_,_=_native_dispatcher_fixture();dispatcher.executor_provider=lambda i:executor
    dispatcher.primitive_goal_factory=PrimitiveGoal.from_dict;dispatcher.accepted_state=PrimitiveState.ACCEPTED
    dispatcher.dispatch_projected(env_index=0,projection=project_executable_action_set((candidate,)))
    outcome=dispatcher.proposal_outcome(0)
    assert outcome.executed_candidate_ids==(candidate.candidate_id,) and not outcome.physical_drops
    assert not outcome.conflict_owner_ids
    assert state.capture(**args)==oracle(args)


def test_generation_reset_same_executor_counts_epoch_once_without_recounting_history():
    executor=executor_fixture();executor.submit(align_goal(),0.);state=cache()
    state.capture(**inputs(executor,generation=1))
    assert state.metrics()['active_claim_version_changes']==1
    state.capture(**inputs(executor,generation=2))
    assert state.metrics()['active_claim_version_changes']==2  # one acquisition + one epoch change
    assert state.metrics()['claim_epoch_changes']==1


def test_prepared_long_smoke_cannot_stop_at_two_updates_and_keeps_minibatch_memory(tmp_path):
    from pathlib import Path
    from drl.MARL.isaaclab_c1_runner import build_parser,configure_run_arguments
    root=Path(__file__).resolve().parents[3]
    path=root/'logs/isaaclab_c1_native/redesign/c1_task5a_cached_mb8_40s_r1/preparation.json'
    if not path.exists():pytest.skip('local human-run preparation unavailable')
    preparation=json.loads(path.read_text())
    args,extra=build_parser().parse_known_args(preparation['argv'])
    assert extra==['--headless','--device','cuda:0']
    args.output=tmp_path/'new-run';args=configure_run_arguments(args)
    assert (args.num_envs,args.rollout_steps,args.mini_batches,args.learning_epochs)==(4,32,8,2)
    assert args.total_transitions==4864 and args.min_updates==38 and args.max_vector_steps==1216
    assert args.episode_seconds==60 and args.min_completed_seeds_per_domain==0
    assert args.total_transitions/args.num_envs/30>40
    assert preparation['executed'] is False and preparation['mixed_precision_enabled'] is False
