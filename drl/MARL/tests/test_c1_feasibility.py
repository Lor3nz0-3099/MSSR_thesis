"""Native temporal feasibility is external policy support, never an observation."""
from dataclasses import replace
from types import SimpleNamespace
import json

import pytest
import torch

from drl.IL.bc_v1.bc_v1_contract import ModuleRef, ResourceRef
from drl.MARL.policy import (MARLCandidateRecord, ResourcePayloadView, StructuralPayloadView,
    StructuralScopeView, StructuralReferenceSetArgumentView, sample_marl_action_bundle,
    compute_owner_action_log_probs, compute_owner_action_entropies)


def executor_fixture():
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor
    from scripts.smores_ep.tests.test_primitives import _FakeDocking, _FakeStateReader
    ids=('module_a','module_b','module_c','module_d')
    docking=_FakeDocking();docking.module_ids=ids
    return IsaacPrimitiveExecutor(stage=object(),module_roots={m:'/'+m for m in ids},
        states={m:_FakeStateReader() for m in ids},docking=docking)


def align_goal(gid='active',a='module_a',b='module_b',fa='TOP',fb='BOTTOM'):
    from smores_ep.primitives.model import PrimitiveGoal
    return PrimitiveGoal.from_dict(dict(goal_id=gid,primitive='align_faces',module_ids=[a,b],
        parameters=dict(face_a=fa,face_b=fb)))


def routing_fixture():
    from drl.IL.bc_v1.bc_v1_contract import ConnectorRef
    modules=tuple(ModuleRef(m) for m in ('module_a','module_b','module_c','module_d'))
    def pair(a,b):
        ports=(ConnectorRef(modules[a],'TOP'),ConnectorRef(modules[b],'BOTTOM'))
        return SimpleNamespace(action_id='align_faces'),SimpleNamespace(initiator=modules[a],connectors=ports,
            required_scope_members=frozenset((modules[a],modules[b])),scope_members=frozenset((modules[a],modules[b])))
    resources=[SimpleNamespace(resource_ref=ResourceRef(m,'locomotion')) for m in modules]
    return dict(module_refs=modules,resource_candidates=resources,structural_bindings=[pair(0,1),pair(2,3)],argument_domains=())


def capture(executor,route=None,graph=None):
    from drl.MARL.c1_feasibility import capture_native_availability
    route=route or routing_fixture()
    return capture_native_availability(routing=route,graph=graph or {},hard_mask=torch.zeros(
        len(route['resource_candidates'])+len(route['structural_bindings']),dtype=torch.bool),executor=executor)


def test_active_native_snapshot_claim_lifetime_is_read_only_and_shared_modes_exact():
    executor=executor_fixture();assert not executor.active_resource_snapshot()
    goal=align_goal();executor.submit(goal,0.)
    before=dict(executor._resource_owners)
    snapshot=executor.active_resource_snapshot()
    row,=snapshot
    assert row.goal_id=='active' and row.primitive=='align_faces'
    assert row.module_ids==('module_a','module_b')
    assert dict(row.resource_modes)=={'locomotion:module_a':'exclusive','locomotion:module_b':'shared',
        'internal_motion:module_a':'exclusive','connector:module_b:BOTTOM':'exclusive'}
    assert row.faces==('TOP','BOTTOM')
    assert all('module_c' not in key and 'module_d' not in key for key,_ in row.resource_modes)
    assert executor._resource_owners==before
    executor.cancel(goal.goal_id,.1)
    assert not executor.active_resource_snapshot()
    assert snapshot[0]==row  # Independent immutable historical value.


def test_busy_masks_only_held_claims_and_disjoint_pair_survives():
    executor=executor_fixture();executor.submit(align_goal(),0.)
    availability=capture(executor)
    assert availability.candidate_mask==(True,True,False,False,True,False)
    assert availability.reasons[4].code=='masked_busy'
    assert availability.reasons[4].conflicts_with[0].goal_id=='active'
    executor.cancel('active',.2)
    assert capture(executor).candidate_mask==(False,)*6
    assert availability.candidate_mask[4] is True


def test_native_shared_target_can_still_be_read_by_disjoint_initiator():
    executor=executor_fixture();executor.submit(align_goal(),0.)
    route=routing_fixture()
    action,binding=route['structural_bindings'][1]
    from drl.IL.bc_v1.bc_v1_contract import ConnectorRef
    modules=route['module_refs']
    binding=SimpleNamespace(initiator=modules[2],connectors=(ConnectorRef(modules[2],'BOTTOM'),ConnectorRef(modules[1],'LEFT')),
        required_scope_members=frozenset((modules[2],modules[1])))
    route['structural_bindings'][1]=(action,binding)
    assert not capture(executor,route).candidate_mask[5]


def gravity_fixture():
    route=routing_fixture();modules=route['module_refs']
    parameter=SimpleNamespace(name='passive_module_ids',kind='module_refs')
    route['argument_domains']=(SimpleNamespace(parameter=parameter),)
    route['structural_bindings']=[(SimpleNamespace(action_id='gravity_settle'),
        SimpleNamespace(initiator=modules[2],connectors=(),required_scope_members=frozenset((modules[2],))))]
    graph={'structural':dict(scope_ptr=torch.tensor([0,4]),scope_members=torch.arange(4),
        scope_required=torch.tensor([False,False,True,False]),argument_binding=torch.tensor([0]),
        argument_domain_ptr=torch.tensor([0,4]),argument_domain_index=torch.arange(4),
        argument_domain_required=torch.tensor([False]*4))}
    payload=StructuralPayloadView(scope=StructuralScopeView(torch.tensor([20.,20.,20.],requires_grad=True),torch.tensor([0,1,3]),torch.tensor([2])),
        reference_set=(StructuralReferenceSetArgumentView(0,torch.tensor([20.]*4,requires_grad=True),torch.arange(4),torch.tensor([],dtype=torch.long)),))
    return route,graph,payload


def test_optional_busy_memberships_removed_before_sampling_not_afterward():
    from drl.MARL.c1_feasibility import apply_availability
    executor=executor_fixture();executor.submit(align_goal(),0.)
    route,graph,payload=gravity_fixture();availability=capture(executor,route,graph)
    assert availability.candidate_mask[-1] is False  # free required initiator
    records=(MARLCandidateRecord(4,'structural','gravity','module_c',(),torch.tensor(20.),4.5),)
    resources=simple_output(torch.tensor([-10.]*4))
    output=dict(records=(*resources['records'],*records),resource_payloads=resources['resource_payloads'],structural_payloads={4:payload},
        resource_log_std=torch.tensor(-4.),structural_log_std=torch.tensor(-4.))
    masked=apply_availability(output,availability)
    action=sample_marl_action_bundle(**masked,hard_mask=availability.mask_tensor(),deterministic=True)
    assert action.structural_scope[4].tolist()==[3,2]
    assert action.structural_reference_set[4][0].tolist()==[2,3]
    assert payload.scope.optional_domain_indices.tolist()==[0,1,3]  # Raw H2 views unchanged.
    lp=compute_owner_action_log_probs(**masked,action=action,hard_mask=availability.mask_tensor(),possible_agents=('module_c',))
    ent=compute_owner_action_entropies(**masked,hard_mask=availability.mask_tensor(),possible_agents=('module_c',))
    (lp['module_c']+ent['module_c']).backward()
    assert payload.scope.optional_logits.grad[:2].tolist()==[0.,0.]
    assert payload.reference_set[0].optional_logits.grad[:2].tolist()==[0.,0.]
    assert torch.isfinite(lp['module_c']) and torch.isfinite(ent['module_c'])


def simple_output(logits):
    return dict(records=tuple(MARLCandidateRecord(i,'resource',f'resource|{i}','module_c',(),v,4.5) for i,v in enumerate(logits)),
        resource_payloads={i:ResourcePayloadView(numeric=v.reshape(1)) for i,v in enumerate(logits)},
        structural_payloads={},resource_log_std=torch.tensor(-4.),structural_log_std=torch.tensor(-4.))


def test_stored_candidate_mask_is_cloned_and_reevaluation_never_uses_live_state():
    from drl.MARL.mappo import MSSRMAPPO
    logits=torch.nn.Parameter(torch.tensor([6.,7.]))
    agent=object.__new__(MSSRMAPPO);agent.device='cpu';agent.possible_agents=['module_c']
    agent.shared_actor=SimpleNamespace(forward_all_owners=lambda **kw:simple_output(logits))
    agent.shared_critic=SimpleNamespace(act=lambda *a,**kw:(torch.zeros(1,1),{}))
    mask=torch.tensor([True,False]);agent.deterministic_policy_actions=True
    decision=agent.build_online_rollout_decision(actor_graph={},critic_graph={},routing_snapshot={},hard_mask=mask)
    mask[:]=~mask  # executor/state-owned input changes after sampling
    lp,entropy,_=agent.reevaluate_structured_minibatch(structured_batch=[decision.structured_sample],owner_tensor_batch={},evaluate_global_critic=False)
    assert decision.structured_sample.hard_mask.tolist()==[True,False]
    assert torch.allclose(lp['module_c'].flatten(),decision.owner_log_probs['module_c'].reshape(1))
    assert torch.exp(lp['module_c'].flatten()-decision.owner_log_probs['module_c']).item()==pytest.approx(1.)
    assert torch.isfinite(entropy['module_c']).all()


def test_masked_candidate_all_factors_zero_log_prob_entropy_and_rng_draws():
    output=simple_output(torch.tensor([100.],requires_grad=True));mask=torch.tensor([True])
    before=torch.random.get_rng_state()
    action=sample_marl_action_bundle(**output,hard_mask=mask)
    assert torch.equal(before,torch.random.get_rng_state())
    assert action.presence.tolist()==[0.] and not action.resource_numeric
    assert compute_owner_action_log_probs(**output,action=action,hard_mask=mask,possible_agents=('module_c',))['module_c']==0
    assert compute_owner_action_entropies(**output,hard_mask=mask,possible_agents=('module_c',))['module_c']==0


def test_action_diagnostics_distinguish_busy_and_intra_step_conflicts_without_rng(tmp_path):
    from drl.MARL.c1_action_events import C1ActionEvents
    from drl.MARL.tests.test_action_projection import candidate
    from drl.MARL.action_projection import project_executable_action_set
    # Projection rules unchanged: shared logical resource, descending raw logit.
    a=candidate('a',8.,('module_a',));b=candidate('b',7.,('module_a',))
    projection=project_executable_action_set((a,b))
    events=C1ActionEvents(tmp_path)
    before=torch.random.get_rng_state()
    rows=events.projection_events(projection,identity=dict(env_index=0,generation=2,task='navigation',seed=7017,timestamp_s=.1))
    assert torch.equal(before,torch.random.get_rng_state())
    dropped=next(r for r in rows if r['candidate_id']=='b')
    assert dropped['outcome']=='dropped_intra_step_conflict'
    assert dropped['conflicts_with'] and dropped['resource_claims']
    assert 'intra_step_conflict_drop_count' in events.totals
    assert json.loads((tmp_path/'action_events.jsonl').read_text().splitlines()[-1])['outcome'] in {'selected_executable','dropped_intra_step_conflict'}


def test_structured_optional_support_is_stored_for_ppo_after_live_busy_changes():
    from drl.MARL.c1_feasibility import apply_availability
    from drl.MARL.mappo import MSSRMAPPO
    executor=executor_fixture();executor.submit(align_goal(),0.)
    route,graph,payload=gravity_fixture();availability=capture(executor,route,graph)
    output=dict(records=(MARLCandidateRecord(4,'structural','gravity','module_c',(),torch.tensor(20.),4.5),),
        resource_payloads={},structural_payloads={4:payload},resource_log_std=torch.tensor(-4.),structural_log_std=torch.tensor(-4.))
    # Four fixed resource rows precede the structural row in the real routing.
    resources=simple_output(torch.tensor([-10.]*4))
    output['records']=(*resources['records'],*output['records'])
    output['resource_payloads']=resources['resource_payloads']
    agent=object.__new__(MSSRMAPPO);agent.device='cpu';agent.possible_agents=['module_c']
    agent.shared_actor=SimpleNamespace(forward_all_owners=lambda **kw:output)
    agent.shared_critic=SimpleNamespace(act=lambda *a,**kw:(torch.zeros(1,1),{}));agent.deterministic_policy_actions=True
    decision=agent.build_online_rollout_decision(actor_graph=graph,critic_graph={},routing_snapshot=route,
        hard_mask=torch.zeros(5,dtype=torch.bool),availability_snapshot=availability)
    executor.cancel('active',.1)
    assert not any(capture(executor,route,graph).candidate_mask)
    lp,_,_=agent.reevaluate_structured_minibatch(structured_batch=[decision.structured_sample],owner_tensor_batch={},evaluate_global_critic=False)
    assert decision.structured_sample.availability_snapshot==availability
    assert torch.allclose(lp['module_c'].flatten(),decision.owner_log_probs['module_c'].reshape(1))
    with torch.no_grad():
        payload.scope.optional_logits[:2].fill_(-100.)
        payload.reference_set[0].optional_logits[:2].fill_(-100.)
    after,_,_=agent.reevaluate_structured_minibatch(structured_batch=[decision.structured_sample],owner_tensor_batch={},evaluate_global_critic=False)
    assert torch.equal(after['module_c'],lp['module_c'])


def test_busy_mask_events_have_scores_exact_claims_and_no_fake_sample(tmp_path):
    from drl.MARL.c1_action_events import C1ActionEvents
    from drl.MARL.policy import bind_marl_candidates
    executor=executor_fixture();executor.submit(align_goal(),0.)
    route=routing_fixture()
    # The small synthetic resource rows need the same opaque routing identifiers.
    for r in route['resource_candidates']:
        r.action=SimpleNamespace(action_id='drive');r.mode=SimpleNamespace(mode_id='velocity')
    availability=capture(executor,route)
    logits=torch.tensor([6.,7.,-10.,-10.,9.,-10.])
    action=SimpleNamespace(presence=torch.zeros(6))
    recorder=C1ActionEvents(tmp_path,masked_detail_limit=1)
    identity=dict(env_index=0,generation=0,task='gap',seed=16022,timestamp_s=0.)
    before=torch.random.get_rng_state()
    rows=recorder.mask_events(records=bind_marl_candidates(routing=route,presence_logits=logits,presence_threshold_logit=4.5),
        availability=availability,action=action,identity=identity)
    assert torch.equal(before,torch.random.get_rng_state())
    assert recorder.totals['masked_busy_count']==3
    details=[r for r in rows if r['outcome']=='masked_busy']
    assert len(details)==1
    row=details[0]
    assert row['raw_presence_logit']==9. and row['calibrated_presence_logit']==4.5
    assert row['above_presence_boundary'] is True and row['sampled'] is False
    assert row['conflicts_with'][0]['goal_id']=='active'
    assert row['resource_claims'] and row['availability_state']=='masked_busy'


def test_masked_busy_no_conflict_penalty_but_intra_step_and_executor_fallback_keep_it(tmp_path):
    from drl.MARL.c1_action_events import C1ActionEvents
    from drl.MARL.tests.test_action_projection import candidate
    from drl.MARL.projected_runtime import projection_outcome
    from drl.MARL.action_projection import project_executable_action_set
    from drl.MARL.runtime_signals import collect_runtime_signals
    owners=('a','b')
    empty=project_executable_action_set(())
    assert not collect_runtime_signals(owner_ids=owners,delta_seconds=1/30,proposal_outcome=projection_outcome(empty)).signals.conflict_owner_ids
    projection=project_executable_action_set((candidate('a',8.,('x',)),candidate('b',7.,('x',))))
    assert collect_runtime_signals(owner_ids=owners,delta_seconds=1/30,proposal_outcome=projection_outcome(projection)).signals.conflict_owner_ids==frozenset(('b',))
    # Late executor BUSY remains a safety penalty, classified outside intra-step.
    busy=project_executable_action_set((candidate('a',8.,('x',)),),feasibility=lambda c:('RESOURCE_BUSY','locomotion:x owned by active'))
    assert collect_runtime_signals(owner_ids=owners,delta_seconds=1/30,proposal_outcome=projection_outcome(busy)).signals.conflict_owner_ids==frozenset(('a',))
    recorder=C1ActionEvents(tmp_path)
    rows=recorder.projection_events(busy,identity=dict(env_index=0,generation=0,task='navigation',seed=7017,timestamp_s=0.))
    assert rows[0]['outcome']=='executor_rejected_busy'
    assert recorder.totals['executor_busy_rejection_count']==1 and recorder.totals['intra_step_conflict_drop_count']==0


def test_native_context_applies_external_mask_without_changing_graph_schema():
    from drl.MARL.tests.test_native_policy_bridge import make_bridge
    bridge,env,_,_=make_bridge()
    # Use an actual captured physical routing; no network input extension.
    route=routing_fixture();executor=executor_fixture();executor.submit(align_goal(),0.)
    prepared=bridge.provider.observation_source.prepare_env(0)
    source=SimpleNamespace(num_envs=2,prepare_env=lambda i:replace(prepared,routing_snapshot=route),reset=lambda:None)
    bridge.provider.observation_source=source;bridge.provider.reset()
    bridge.provider._hard_mask_builder=lambda g,r:torch.zeros(6,dtype=torch.bool)
    env.cfg=SimpleNamespace(initialization_mode='loose_modules')
    env.native_dispatcher=SimpleNamespace(executor_provider=lambda i:executor)
    contexts=bridge.provider.current_action_contexts()
    assert contexts[0].hard_mask.tolist()==[True,True,False,False,True,False]
    assert set(contexts[0].actor_graph)==set(prepared.actor_graph)
    assert torch.equal(contexts[0].actor_graph['visible'],prepared.actor_graph['visible'])
    assert contexts[0].availability_snapshot.reasons[4].code=='masked_busy'


def test_primitive_lifecycle_events_are_deduplicated_across_physics_ticks(tmp_path):
    from drl.MARL.c1_action_events import C1ActionEvents
    from drl.MARL.c1_structure import C1PrimitiveEvent
    recorder=C1ActionEvents(tmp_path)
    identity=dict(env_index=0,generation=0,task='stairs',seed=12005,timestamp_s=.1)
    metadata=dict(candidate_id='structural|align_faces|module_a|ports',primitive='align_faces',module_ids=['module_a','module_b'],faces=['TOP','BOTTOM'])
    recorder.register_goal('g',metadata,identity)
    event=C1PrimitiveEvent('g','align_faces','succeeded','DONE','module_a')
    recorder.primitive_events((event,event),identity=identity)
    recorder.primitive_events((event,),identity={**identity,'timestamp_s':.2})
    assert recorder.totals['structural_started_count']==1
    assert recorder.totals['primitive_completed_count']==1
    rows=[json.loads(line) for line in (tmp_path/'action_events.jsonl').read_text().splitlines()]
    assert [r['outcome'] for r in rows]==['primitive_started','primitive_completed']
    assert rows[-1]['module_ids']==['module_a','module_b']


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16])
def test_masked_structural_candidate_zero_for_every_payload_head(dtype):
    from drl.MARL.policy import (StructuralNumericArgumentView,StructuralBooleanArgumentView,
        StructuralCategoricalArgumentView,StructuralReferenceArgumentView)
    raw=torch.tensor(100.,dtype=dtype,requires_grad=True)
    record=MARLCandidateRecord(0,'structural','structural|align_faces|a|a:TOP>b:BOTTOM','a',('b',),raw,4.5)
    payload=StructuralPayloadView(numeric=(StructuralNumericArgumentView(0,raw.reshape(1)),),
        boolean=(StructuralBooleanArgumentView(1,raw),),categorical=(StructuralCategoricalArgumentView(2,raw.repeat(2)),),
        reference=(StructuralReferenceArgumentView(3,raw.repeat(2),torch.arange(2)),),
        scope=StructuralScopeView(raw.reshape(1),torch.tensor([1]),torch.tensor([0])),
        reference_set=(StructuralReferenceSetArgumentView(4,raw.reshape(1),torch.tensor([1]),torch.tensor([0])),))
    out=dict(records=(record,),resource_payloads={},structural_payloads={0:payload},resource_log_std=raw,structural_log_std=raw)
    mask=torch.tensor([True]);before=torch.random.get_rng_state()
    action=sample_marl_action_bundle(**out,hard_mask=mask)
    assert torch.equal(before,torch.random.get_rng_state())
    lp=compute_owner_action_log_probs(**out,action=action,hard_mask=mask,possible_agents=('a',))['a']
    ent=compute_owner_action_entropies(**out,hard_mask=mask,possible_agents=('a',))['a']
    assert lp==ent==0 and torch.isfinite(lp+ent)


def test_action_progress_totals_are_unbounded_and_written_atomically(tmp_path):
    from drl.MARL.c1_action_events import C1ActionEvents
    recorder=C1ActionEvents(tmp_path)
    identity=dict(env_index=0,generation=0,task='gap',seed=16022,timestamp_s=.1)
    recorder._count('masked_busy_count',identity,50)
    recorder.record_transition_result(team_reward=-.25,valid_connections=3,success=False)
    recorder.record_transition_result(team_reward=.5,valid_connections=5,success=True)
    report=json.loads((tmp_path/'action_summary.json').read_text())
    assert report['masked_busy_count']==50 and report['valid_connection_max']==5
    assert report['row_transitions']==2 and report['C1_success_count']==1
    assert report['mean_reward']==pytest.approx(.125)
    assert report['by_task']['gap']['masked_busy_count']==50


def test_conflict_viewer_and_progress_print_separate_causes(tmp_path,capsys):
    from drl.MARL.c1_action_diagnostics import main
    discovery=tmp_path/'c1_discovery';discovery.mkdir()
    (discovery/'action_summary.json').write_text(json.dumps(dict(row_transitions=12,masked_busy_count=30,structural_sampled_count=8)))
    (tmp_path/'updates.jsonl').write_text(json.dumps(dict(update=1))+'\n')
    rows=[dict(outcome=k,primitive='align_faces',module_ids=['m1','m3'],faces=['TOP','LEFT'],task='gap',conflicts_with=[dict(candidate_id='peer',shared_claims=['locomotion:m1'])])
          for k in ('masked_busy','dropped_intra_step_conflict','executor_rejected_busy')]
    (discovery/'action_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    assert main(['--run',str(tmp_path),'--progress'])==0
    printed=capsys.readouterr().out
    assert '12 / 256' in printed and '1 / 2' in printed and '30' in printed
    assert main(['--run',str(tmp_path),'--details'])==0
    printed=capsys.readouterr().out
    assert all(k in printed for k in ('masked_busy','dropped_intra_step_conflict','executor_rejected_busy','m1','TOP','locomotion:m1'))


def test_real_frozen_bcv4_forward_mask_and_ppo_replay_on_finished_physical_capture():
    from pathlib import Path
    from drl.MARL.c1_runner import CHECKPOINT, build_observation_bridge, build_models
    ROOT=Path(__file__).resolve().parents[3]
    from drl.MARL.mappo import MSSRMAPPO
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor
    from scripts.smores_ep.tests.test_primitives import _FakeDocking, _FakeStateReader
    path=ROOT/'logs/isaaclab_c1_native/redesign/structural_conflict_diagnostic/navigation/decision_000.pt'
    if not path.exists():pytest.skip('finished local physical capture unavailable')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    bridge=build_observation_bridge(ROOT)
    actor,critic=build_models(SimpleNamespace(repository_root=ROOT,checkpoint=ROOT/CHECKPOINT,device='cpu'),bridge)
    assert actor.presence_threshold_logit==4.5
    route=saved['routing'];ids=tuple(m.module_id for m in route['module_refs'])
    docking=_FakeDocking();docking.module_ids=ids
    executor=IsaacPrimitiveExecutor(stage=object(),module_roots={m:'/'+m for m in ids},states={m:_FakeStateReader() for m in ids},docking=docking)
    executor.submit(align_goal(a=ids[0],b=ids[2]),0.)
    availability=capture(executor,route,saved['actor_graph'])
    assert any(availability.candidate_mask) and not all(availability.candidate_mask)
    agent=object.__new__(MSSRMAPPO);agent.device='cpu';agent.possible_agents=list(ids)
    agent.shared_actor=actor;agent.shared_critic=critic;agent.deterministic_policy_actions=True
    # Global critic is irrelevant to this actor-support integration contract.
    agent.shared_critic=SimpleNamespace(act=lambda *a,**kw:(torch.zeros(1,1),{}))
    decision=agent.build_online_rollout_decision(actor_graph=saved['actor_graph'],critic_graph={},routing_snapshot=route,
        hard_mask=saved['hard_mask'],availability_snapshot=availability)
    assert not decision.action.presence[availability.mask_tensor()].any()
    executor.cancel('active',.1)
    new,ent,_=agent.reevaluate_structured_minibatch(structured_batch=[decision.structured_sample],owner_tensor_batch={},evaluate_global_critic=False)
    for owner in ids:
        assert torch.allclose(new[owner].flatten(),decision.owner_log_probs[owner].reshape(1),atol=1e-5)
        assert torch.isfinite(ent[owner]).all()
    assert all('busy' not in str(k).lower() and 'desired_morphology' not in str(k).lower() for k in saved['actor_graph'])


def test_preflight_busy_fallback_records_actual_active_goal_even_if_it_started_after_snapshot(tmp_path):
    from drl.MARL.c1_action_events import C1ActionEvents
    from drl.MARL.action_projection import ProjectionCandidate,project_executable_action_set
    from drl.IL.bc_v1.bc_v1_contract import ActionChoice
    from drl.IL.bc_v1.live_policy import smores_claims
    executor=executor_fixture();route=routing_fixture()
    for r in route['resource_candidates']:
        r.action=SimpleNamespace(action_id='drive');r.mode=SimpleNamespace(mode_id='velocity')
    availability=capture(executor,route)
    action,binding=route['structural_bindings'][0]
    choice=ActionChoice(action.action_id,binding.initiator,participants=tuple(p.module for p in binding.connectors),connectors=binding.connectors)
    row=ProjectionCandidate('structural|align_faces|module_a|module_a:TOP>module_b:BOTTOM',4,9.,('module_a',),choice,smores_claims(choice))
    executor.submit(align_goal(),0.)  # Runtime fallback after the earlier free snapshot.
    projection=project_executable_action_set((row,),feasibility=lambda c:('RESOURCE_BUSY','locomotion:module_a owned by active'))
    env=SimpleNamespace(cfg=SimpleNamespace(),generation=[0],domain_by_env=['navigation'],seed_by_env=[7017],_sim_step_counter=0,physics_dt=1/240,
        native_dispatcher=SimpleNamespace(executor_provider=lambda i:executor))
    recorder=C1ActionEvents(tmp_path)
    recorder.record_decision(env,0,SimpleNamespace(action=SimpleNamespace(presence=torch.tensor([0.,0.,0.,0.,1.,0.])),
        action_set_prediction_template=dict(presence_logits=torch.tensor([-10.,-10.,-10.,-10.,9.,-10.]))),
        SimpleNamespace(routing_snapshot=route,availability_snapshot=availability),projection)
    rows=[json.loads(line) for line in (tmp_path/'action_events.jsonl').read_text().splitlines()]
    rejected=next(r for r in rows if r['outcome']=='executor_rejected_busy')
    assert rejected['conflicts_with'][0]['goal_id']=='active'
    assert rejected['conflicts_with'][0]['resource_modes']


def test_native_action_counters_reach_transitions_and_preserve_legacy_aliases(tmp_path):
    from drl.MARL.c1_action_events import C1ActionEvents
    from drl.MARL.c1_native_runtime import NativeC1RewardRuntime
    from drl.MARL.tests.test_c1_native_runtime import env_fixture
    from drl.MARL.tests.test_c1_structure import frame
    env=env_fixture();env.c1_action_events=C1ActionEvents(tmp_path)
    identity=dict(env_index=0,generation=1,task='navigation',seed=7017,timestamp_s=0.)
    env.c1_action_events._count('masked_busy_count',identity,17)
    runtime=NativeC1RewardRuntime(evidence_directory=tmp_path)
    runtime.evaluate(env,0,frame(edges=0))
    row=json.loads((tmp_path/'transitions.jsonl').read_text())
    assert row['action_counts']['masked_busy_count']==17
    assert row['action_counts']['intra_step_conflict_drop_count']==0
    assert row['reward_breakdown']['penalty_conflict']==0
    progress=json.loads((tmp_path/'action_summary.json').read_text())
    assert progress['masked_busy_count']==17


def test_prepared_task5a_mb8_command_uses_current_cli_and_explicit_no_coverage_gate(tmp_path):
    from pathlib import Path
    from drl.MARL.isaaclab_c1_runner import build_parser,configure_run_arguments
    from skrl.multi_agents.torch.mappo import MAPPO_CFG
    root=Path(__file__).resolve().parents[3]
    path=root/'logs/isaaclab_c1_native/redesign/c1_task5a_masked_mb8_r1/preparation.json'
    if not path.exists():pytest.skip('local human-run preparation unavailable')
    preparation=json.loads(path.read_text())
    args,extra=build_parser().parse_known_args(preparation['argv'])
    assert extra==['--headless','--device','cuda:0']
    assert args.output.name=='c1_task5a_masked_mb8_r1'
    # The human has now completed this run; validate argv in a fresh test output
    # rather than requiring deletion/overwrite of authoritative finished evidence.
    args.output=tmp_path/'unexecuted-smoke'
    args=configure_run_arguments(args)
    assert args.smoke_train and not args.train and args.curriculum=='c1' and args.initialization_mode=='loose_modules'
    assert (args.num_envs,args.rollout_steps,args.learning_epochs,args.mini_batches,args.total_transitions,args.min_updates,args.episode_seconds,args.global_seed)==(4,32,2,8,256,2,40.,3103)
    assert args.min_completed_seeds_per_domain==0 and args.max_vector_steps==64
    assert preparation['environment']['PYTORCH_ALLOC_CONF']=='expandable_segments:True'
    assert preparation['mixed_precision_enabled'] is False and MAPPO_CFG().mixed_precision is False
    assert not preparation['executed']


@pytest.mark.parametrize('rejection',['RESOURCE_BUSY','INVALID_GOAL'])
def test_native_rejection_attributes_only_actual_conflicting_holders(rejection):
    from drl.MARL.tests.test_smores_runtime_binding import _native_dispatcher_fixture
    from drl.MARL.smores_runtime_binding import DecodedSMORESAction
    from drl.MARL.action_projection import ProjectionCandidate,project_executable_action_set
    from drl.IL.bc_v1.bc_v1_contract import ActionChoice
    from smores_ep.primitives.model import PrimitiveGoal,PrimitiveState
    executor=executor_fixture();executor.submit(align_goal(),0.)
    executor.submit(align_goal('independent','module_c','module_d'),0.)
    dispatcher,_,_=_native_dispatcher_fixture()
    dispatcher.executor_provider=lambda i:executor
    dispatcher.primitive_goal_factory=PrimitiveGoal.from_dict
    dispatcher.accepted_state=PrimitiveState.ACCEPTED
    if rejection=='INVALID_GOAL':
        executor.submit=lambda *a,**kw:SimpleNamespace(state=PrimitiveState.REJECTED,code='INVALID_GOAL',message='invalid argument')
    ref=ResourceRef(ModuleRef('module_a'),'internal_motion')
    choice=ActionChoice('pan_target',ref.module,resource=ref,payload=(.1,))
    candidate=ProjectionCandidate('resource|module_a|internal_motion|pan_target',0,8.,('module_a',),choice,(ref,),
        DecodedSMORESAction(choice,dict(command='set_pan',module_ids=['module_a'],parameters=dict(angle_rad=.1))))
    dispatcher.dispatch_projected(env_index=0,projection=project_executable_action_set((candidate,)))
    event,=dispatcher.proposal_outcome(0).admission_events
    assert not event['accepted'] and event['code']==rejection
    assert [c.goal_id for c in event['active_claims']]==(['active'] if rejection=='RESOURCE_BUSY' else [])


def test_optional_busy_details_identify_factor_module_and_actual_holders_without_rng(tmp_path):
    from drl.MARL.c1_action_events import C1ActionEvents
    executor=executor_fixture();executor.submit(align_goal(),0.)
    route,graph,_=gravity_fixture();availability=capture(executor,route,graph)
    resources=simple_output(torch.tensor([-10.]*4))
    records=(*resources['records'],MARLCandidateRecord(4,'structural','structural|gravity_settle|module_c|none','module_c',(),torch.tensor(9.),4.5))
    recorder=C1ActionEvents(tmp_path)
    before=torch.random.get_rng_state()
    rows=recorder.mask_events(records=records,availability=availability,action=SimpleNamespace(presence=torch.tensor([0.,0.,0.,0.,1.])),
        identity=dict(env_index=0,generation=0,task='gap',seed=16022,timestamp_s=0.))
    assert torch.equal(before,torch.random.get_rng_state())
    optional=[r for r in rows if r.get('mask_granularity')=='optional_membership']
    assert len(optional)==4
    assert {r['factor'] for r in optional}=={'scope','reference_set'}
    assert {tuple(r['module_ids']) for r in optional}=={('module_a',),('module_b',)}
    assert all(r['outcome']=='masked_busy' and r['sampled'] is False and r['candidate_sampled'] is True for r in optional)
    assert all(r['conflicts_with'][0]['goal_id']=='active' and r['resource_claims'] and r['domain_index']>=0 for r in optional)
    assert recorder.totals['masked_optional_membership_count']==4
