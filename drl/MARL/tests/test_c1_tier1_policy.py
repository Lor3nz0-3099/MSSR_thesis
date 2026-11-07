"""Seed-exact sampler, lazy routing and native projection/dispatch oracles."""
from dataclasses import fields,is_dataclass
from pathlib import Path
from types import SimpleNamespace
import copy
import pytest
import torch
from drl.MARL import policy as p
from drl.MARL.tests.reference_policy_sampler import sample_reference


def equal(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif is_dataclass(a):
        for field in fields(a):equal(getattr(a,field.name),getattr(b,field.name))
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def mixed():
    records=tuple(p.MARLCandidateRecord(i,'resource' if i in (0,2) else 'structural',str(i),'a' if i%2==0 else 'b',(),torch.tensor(4.+i/2),4.5) for i in range(6))
    payload=p.StructuralPayloadView(numeric=(p.StructuralNumericArgumentView(0,torch.tensor([.1,.2])),),
        boolean=(p.StructuralBooleanArgumentView(1,torch.tensor(.2)),),
        categorical=(p.StructuralCategoricalArgumentView(2,torch.tensor([.3,.1])),),
        reference=(p.StructuralReferenceArgumentView(3,torch.tensor([.4,-.2]),torch.tensor([1,3])),),
        reference_set=(p.StructuralReferenceSetArgumentView(4,torch.tensor([.3,-.1]),torch.tensor([0,2]),torch.tensor([1])),),
        scope=p.StructuralScopeView(torch.tensor([.2,-.7]),torch.tensor([0,2]),torch.tensor([1])))
    return dict(records=records,resource_payloads={0:p.ResourcePayloadView(numeric=torch.tensor([.4,.2])),2:p.ResourcePayloadView(categorical_logits=torch.tensor([.1,.2,.3]))},
        structural_payloads={i:payload for i in (1,3,4,5)},resource_log_std=torch.tensor(-2.),structural_log_std=torch.tensor(-2.))


@pytest.mark.parametrize('seed',[0,1,7,41,291])
@pytest.mark.parametrize('mask',[[True]*6,[False]*6,[True,False,False,True,False,True]])
def test_seed_exact_bundle_rng_probabilities_logprob_and_entropy(seed,mask):
    output=mixed();mask=torch.tensor(mask)
    torch.manual_seed(seed);reference=sample_reference(**output,hard_mask=mask);rng=torch.random.get_rng_state()
    torch.manual_seed(seed);actual=p.sample_marl_action_bundle(**output,hard_mask=mask,bulk_host_reads=True)
    equal(actual,reference);assert torch.equal(rng,torch.random.get_rng_state())
    for fn,extra in ((p.compute_owner_action_log_probs,dict(action=actual)),(p.compute_owner_action_entropies,{})):
        a=fn(**output,hard_mask=mask,possible_agents=('a','b'),**extra)
        if extra:extra=dict(action=reference)
        b=fn(**output,hard_mask=mask,possible_agents=('a','b'),**extra)
        equal(a,b)
    assert not actual.presence[mask].any()


def test_bulk_transfer_removes_per_candidate_item_reads(monkeypatch):
    from drl.MARL.tests.test_c1_feasibility import simple_output
    output=simple_output(torch.full((1856,),-100.));mask=torch.zeros(1856,dtype=torch.bool)
    calls=[];original=torch.Tensor.item
    def observed(tensor,*a):calls.append(1);return original(tensor,*a)
    monkeypatch.setattr(torch.Tensor,'item',observed)
    sample_reference(**output,hard_mask=mask);before=len(calls);calls.clear()
    p.sample_marl_action_bundle(**output,hard_mask=mask,bulk_host_reads=True)
    assert len(calls)<10 and before>=3*1856


@pytest.fixture(scope='module')
def checkpoint_case():
    from drl.MARL.tests.test_bc_v4_runtime_observation import make_bridge
    from drl.IL.tests.bc_v1 import test_live_observation as data
    from drl.IL.bc_v1.bc_v1_contract import TaskContext
    from drl.IL.bc_v1.runtime_static_cache import RuntimeStaticCache
    from drl.MARL.c1_structural_physics_diagnostic import build_frozen_actor,configuration,parse_args
    bridge=make_bridge();cache=RuntimeStaticCache();raw=data.snapshot(n=8)
    prepared=bridge.prepare_snapshot(raw,task_context=TaskContext('navigation'),include_privileged=False,static_cache=cache)
    actor=build_frozen_actor(bridge,configuration(parse_args(['--preflight','--device','cpu'])))
    return actor,bridge,prepared,cache,raw


def test_real_checkpoint_lazy_materialization_is_exact_and_only_selected(checkpoint_case):
    actor,bridge,prepared,cache,raw=checkpoint_case
    with torch.no_grad():
        eager=actor.forward_all_owners(graph=prepared.actor_graph,routing=prepared.routing_snapshot,possible_agents=[])
        lazy=actor.forward_all_owners(graph=prepared.actor_graph,routing=prepared.routing_snapshot,possible_agents=[],lazy_payloads=True)
    equal(eager['prediction_template'],lazy['prediction_template'])
    assert lazy['structural_payloads'].materialized_count==0
    mask=torch.zeros(len(eager['records']),dtype=torch.bool);mask[::3]=True
    seen={'resource':set(),'structural':set()}
    for seed in (0,19,37):
        keys=('records','resource_payloads','structural_payloads','resource_log_std','structural_log_std')
        torch.manual_seed(seed);reference=sample_reference(**{k:eager[k] for k in keys},hard_mask=mask);rng=torch.random.get_rng_state()
        torch.manual_seed(seed);actual=p.sample_marl_action_bundle(**{k:lazy[k] for k in keys},hard_mask=mask)
        equal(actual,reference);assert torch.equal(rng,torch.random.get_rng_state())
        for record in lazy['records']:
            if actual.presence[record.candidate_index]:seen[record.candidate_kind].add(record.candidate_index)
        for kind in seen:
            assert lazy[kind+'_payloads'].materialized_count==len(seen[kind])
    for key in ('resource_payloads','structural_payloads'):
        # Full inspection is an oracle only; viewer never materializes unselected rows.
        for index,view in eager[key].items():equal(lazy[key][index],view)
    # New H2 values with identical catalog must never reuse old payload tensors.
    from drl.IL.bc_v1.bc_v1_contract import TaskContext
    changed=copy.deepcopy(raw);changed['state']['modules'][0]['linear_velocity'][0]=6.
    changed['state']['modules'][0]['twist']['linear'][0]=6.;changed['graph']['nodes'][0]['attributes']=copy.deepcopy(changed['state']['modules'][0])
    changed=bridge.prepare_snapshot(changed,task_context=TaskContext('navigation'),include_privileged=False,static_cache=cache)
    with torch.no_grad():
        new_eager=actor.forward_all_owners(graph=changed.actor_graph,routing=changed.routing_snapshot,possible_agents=[])
        new_lazy=actor.forward_all_owners(graph=changed.actor_graph,routing=changed.routing_snapshot,possible_agents=[],lazy_payloads=True)
    for index,view in new_eager['structural_payloads'].items():equal(new_lazy['structural_payloads'][index],view)


def test_lazy_optional_membership_restriction_is_exact(checkpoint_case):
    from drl.MARL.c1_feasibility import NativeAvailability,CandidateAvailability,apply_availability
    actor,_,prepared,_,_=checkpoint_case
    with torch.no_grad():
        eager=actor.forward_all_owners(graph=prepared.actor_graph,routing=prepared.routing_snapshot,possible_agents=[])
        lazy=actor.forward_all_owners(graph=prepared.actor_graph,routing=prepared.routing_snapshot,possible_agents=[],lazy_payloads=True)
    indices=[i for i,v in eager['structural_payloads'].items() if v.scope is not None and len(v.scope.optional_domain_indices)]
    assert indices;index=indices[0];view=eager['structural_payloads'][index]
    availability=NativeAvailability((False,)*len(eager['records']),tuple(CandidateAvailability('available') for _ in eager['records']),scope_blocked=((index,(int(view.scope.optional_domain_indices[0]),)),))
    a=apply_availability(eager,availability);b=apply_availability(lazy,availability)
    assert b['structural_payloads'].materialized_count==0
    for index,view in a['structural_payloads'].items():equal(b['structural_payloads'][index],view)


def test_projection_and_native_dispatch_match_frozen_baseline(checkpoint_case):
    from drl.MARL.projected_runtime import project_marl_action
    from drl.MARL.smores_runtime_binding import SMORESMARLRuntimeBinding,SMORESNativeDispatcher
    from drl.IL.bc_v1.live_policy import smores_claims
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor
    from scripts.smores_ep.tests.test_primitives import _FakeDocking,_FakeStateReader
    actor,bridge,prepared,_,_=checkpoint_case
    with torch.no_grad():out=actor.forward_all_owners(graph=prepared.actor_graph,routing=prepared.routing_snapshot,possible_agents=[])
    keys=('records','resource_payloads','structural_payloads','resource_log_std','structural_log_std');mask=torch.zeros(len(out['records']),dtype=torch.bool)
    torch.manual_seed(43);old=sample_reference(**{k:out[k] for k in keys},hard_mask=mask)
    torch.manual_seed(43);new=p.sample_marl_action_bundle(**{k:out[k] for k in keys},hard_mask=mask,bulk_host_reads=True)
    binding=SMORESMARLRuntimeBinding(adapter=bridge.adapter,claim_resolver=smores_claims)
    projected=[];active=[];commands=[]
    owners=tuple(r.module_id for r in prepared.routing_snapshot['module_refs'])
    for action in (old,new):
        result=project_marl_action(binding=binding,action=action,prediction_template=out['prediction_template'],routing=prepared.routing_snapshot,batch=prepared.decoder_batch,normalizers=bridge.normalizers)
        docking=_FakeDocking();docking.module_ids=owners
        executor=IsaacPrimitiveExecutor(stage=object(),module_roots={m:'/'+m for m in owners},states={m:_FakeStateReader() for m in owners},docking=docking)
        dispatcher=SMORESNativeDispatcher(executor_provider=lambda i:executor,now_provider=lambda i:0.)
        dispatcher.dispatch_projected(env_index=0,projection=result)
        projected.append((tuple(c.candidate_id for c in result.accepted_candidates),tuple((d.candidate_id,d.code,d.reason) for d in result.dropped)))
        active.append(executor.active_resource_snapshot());commands.append(dispatcher.continuous_commands(0))
    assert projected[0]==projected[1] and active[0]==active[1] and commands[0]==commands[1]


def test_projection_static_decoder_maps_oracle(checkpoint_case):
    from drl.MARL.projected_runtime import project_marl_action
    from drl.MARL.smores_runtime_binding import SMORESMARLRuntimeBinding
    from drl.IL.bc_v1.live_policy import smores_claims
    actor,bridge,prepared,_,_=checkpoint_case
    assert 'runtime_decoder_layout' in prepared.decoder_batch
    with torch.no_grad():out=actor.forward_all_owners(graph=prepared.actor_graph,routing=prepared.routing_snapshot,possible_agents=[])
    kwargs={k:out[k] for k in ('records','resource_payloads','structural_payloads','resource_log_std','structural_log_std')}
    torch.manual_seed(8);action=p.sample_marl_action_bundle(**kwargs,hard_mask=torch.zeros(len(out['records']),dtype=torch.bool))
    binding=SMORESMARLRuntimeBinding(adapter=bridge.adapter,claim_resolver=smores_claims)
    results=[]
    for cached in (False,True):
        batch=dict(prepared.decoder_batch)
        if not cached:batch.pop('runtime_decoder_layout')
        result=project_marl_action(binding=binding,action=action,prediction_template=out['prediction_template'],routing=prepared.routing_snapshot,batch=batch,normalizers=bridge.normalizers)
        results.append((result.projected_bundle,tuple((d.candidate_id,d.code,d.reason) for d in result.dropped)))
    assert results[0]==results[1]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_seed_exact_rng_and_payloads():
    from dataclasses import replace
    def move(value):
        if torch.is_tensor(value):return value.cuda()
        if is_dataclass(value):return replace(value,**{f.name:move(getattr(value,f.name)) for f in fields(value)})
        if isinstance(value,dict):return {k:move(v) for k,v in value.items()}
        if isinstance(value,tuple):return tuple(move(v) for v in value)
        return value
    output=move(mixed());mask=torch.tensor([False,True,False,False,True,False],device='cuda')
    for seed in (0,19,87):
        torch.manual_seed(seed);reference=sample_reference(**output,hard_mask=mask);rng=torch.cuda.get_rng_state()
        torch.manual_seed(seed);actual=p.sample_marl_action_bundle(**output,hard_mask=mask,bulk_host_reads=True)
        equal(actual,reference);assert torch.equal(rng,torch.cuda.get_rng_state())
