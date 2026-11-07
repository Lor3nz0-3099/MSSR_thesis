"""Exact deterministic policy math with scalar versus bulk support reads."""
from dataclasses import fields, is_dataclass, replace
import pytest
import torch
from drl.MARL import policy
from drl.MARL.tests.test_c1_tier1_policy import mixed


def devices():
    return ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])


def move(value, device):
    if torch.is_tensor(value):
        result=value.detach().clone().to(device)
        return result.requires_grad_(result.is_floating_point())
    if is_dataclass(value):
        return replace(value, **{f.name:move(getattr(value,f.name),device) for f in fields(value)})
    if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
    if isinstance(value,tuple):return tuple(move(v,device) for v in value)
    return value


def tensors(value):
    if torch.is_tensor(value):yield value
    elif is_dataclass(value):
        for f in fields(value):yield from tensors(getattr(value,f.name))
    elif isinstance(value,dict):
        for item in value.values():yield from tensors(item)
    elif isinstance(value,(tuple,list)):
        for item in value:yield from tensors(item)


@pytest.mark.parametrize('device',devices())
@pytest.mark.parametrize('mask_values',[[False]*6,[True]*6,[False,True,False,False,True,False]])
@pytest.mark.parametrize('presence_offset',[0.,100.])
def test_likelihood_entropy_and_all_gradients_exact(device,mask_values,presence_offset,monkeypatch):
    data=move(mixed(),device);mask=torch.tensor(mask_values,device=device)
    data['records']=tuple(replace(record,presence_logit=record.presence_logit+presence_offset)
                          for record in data['records'])
    torch.manual_seed(7033)
    with torch.no_grad():action=policy.sample_marl_action_bundle(**data,hard_mask=mask)
    cpu_rng=torch.random.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state().clone() if device.startswith('cuda') else None
    host=policy._host_scalar_values
    def evaluate(bulk):
        monkeypatch.setattr(policy,'_host_scalar_values',host if bulk else lambda value:None)
        values=[*policy.compute_owner_action_log_probs(**data,hard_mask=mask,action=action,possible_agents=('a','b','unused')).values(),
                *policy.compute_owner_action_entropies(**data,hard_mask=mask,possible_agents=('a','b','unused')).values()]
        loss=sum(values)
        leaves=list(tensors(data))
        params=[v for v in leaves if v.requires_grad]
        grads=torch.autograd.grad(loss,params,allow_unused=True) if loss.requires_grad else [None]*len(params)
        return values,grads
    before,gb=evaluate(False);after,ga=evaluate(True)
    assert all(torch.equal(a,b) for a,b in zip(before,after))
    assert all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)) for a,b in zip(gb,ga))
    assert torch.equal(cpu_rng,torch.random.get_rng_state())
    if cuda_rng is not None:assert torch.equal(cuda_rng,torch.cuda.get_rng_state())


def test_bulk_statistics_remove_candidate_scalar_reads(monkeypatch):
    data=mixed();mask=torch.ones(6,dtype=torch.bool)
    with torch.no_grad():action=policy.sample_marl_action_bundle(**data,hard_mask=mask)
    calls=[];original=torch.Tensor.item
    def observed(value,*args):calls.append(value.device.type);return original(value,*args)
    monkeypatch.setattr(torch.Tensor,'item',observed)
    policy.compute_owner_action_log_probs(**data,action=action,hard_mask=mask,possible_agents=('a','b'))
    policy.compute_owner_action_entropies(**data,hard_mask=mask,possible_agents=('a','b'))
    assert len(calls)==0


def test_nonflat_support_keeps_legacy_scalar_validation():
    assert policy._host_scalar_values(torch.ones(2,2)) is None
    assert len(policy._host_scalar_values(torch.ones(0)))==0


def test_complete_ppo_objective_and_actor_gradients_exact(monkeypatch):
    from drl.MARL.tests.test_5b_ppo_masking_parity import (
        TinyStructuredActor, _agent, _sample, _loss_with_ppo_clipping, OWNERS)
    samples=[
        _sample([.8,-1.3],[False,True],[1.,0.],{0:.1},{'module_0'}),
        _sample([-.2,1.6],[True,False],[0.,0.],{},{'module_1'}),
        _sample([1.1,1.4],[False,False],[0.,1.],{1:-.3},set(OWNERS)),
        _sample([8.,-9.],[True,True],[0.,0.],{},set(),rollout_active=False)]
    actor=TinyStructuredActor();agent=_agent(actor);host=policy._host_scalar_values
    old_probs={uid:torch.tensor([[.1],[-.2],[.3],[0.]]) for uid in OWNERS}
    results=[]
    for bulk in (False,True):
        monkeypatch.setattr(policy,'_host_scalar_values',host if bulk else lambda value:None)
        probs,entropy,_=agent.reevaluate_structured_minibatch(
            structured_batch=samples,owner_tensor_batch={},evaluate_global_critic=False)
        loss=_loss_with_ppo_clipping(probs,entropy,old_probs,samples)
        gradients=torch.autograd.grad(loss,tuple(actor.parameters()),allow_unused=True)
        results.append((loss.detach(),gradients))
    assert torch.equal(results[0][0],results[1][0])
    assert all(torch.equal(a,b) for a,b in zip(results[0][1],results[1][1]))
