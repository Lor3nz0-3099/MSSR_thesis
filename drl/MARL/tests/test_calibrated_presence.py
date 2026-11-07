from types import SimpleNamespace
import pytest
import torch

from drl.MARL.policy import (MARLCandidateRecord, ResourcePayloadView, StructuralPayloadView,
    sample_marl_action_bundle, compute_owner_presence_log_probs, compute_owner_presence_entropies)


def records(kind):
    return tuple(MARLCandidateRecord(i,kind,str(i),'m',(),torch.tensor(x,requires_grad=True),
        presence_threshold_logit=4.5) for i,x in enumerate((4.49,4.50,4.51)))


@pytest.mark.parametrize('kind',['resource','structural'])
def test_calibrated_deterministic_boundary_and_payload_unchanged(kind):
    rows=records(kind)
    payloads={i:ResourcePayloadView(numeric=torch.tensor([19.25])) for i in range(3)} if kind=='resource' else {}
    result=sample_marl_action_bundle(records=rows,resource_payloads=payloads,
        structural_payloads={i:StructuralPayloadView() for i in range(3)} if kind=='structural' else {},
        hard_mask=torch.zeros(3,dtype=torch.bool),resource_log_std=torch.tensor(-2.),
        structural_log_std=torch.tensor(-2.),deterministic=True)
    assert result.presence.tolist()==[0.,1.,1.]
    if kind=='resource':
        assert result.resource_numeric[1].item()==19.25
        assert result.resource_numeric[2].item()==19.25


@pytest.mark.parametrize('kind',['resource','structural'])
def test_sampling_likelihood_entropy_all_use_shifted_logits(kind,monkeypatch):
    rows=records(kind)
    seen=[]
    def draw(distribution):
        if isinstance(distribution,torch.distributions.Bernoulli):
            seen.append(distribution.logits.detach().clone())
        return torch.ones_like(distribution.mean)
    monkeypatch.setattr(torch.distributions.Bernoulli,'sample',draw)
    sampled=sample_marl_action_bundle(records=rows,resource_payloads={},structural_payloads={},
        hard_mask=torch.zeros(3,dtype=torch.bool),resource_log_std=torch.tensor(-2.),structural_log_std=torch.tensor(-2.))
    logits=torch.stack([r.presence_logit for r in rows])
    effective=logits-4.5
    assert torch.allclose(torch.stack(seen),effective)
    assert torch.sigmoid(seen[1]).item()==.5
    assert all(r.presence_logit.item()>4 for r in rows)  # raw confidence retained
    expected=torch.distributions.Bernoulli(logits=effective)
    kwargs=dict(records=rows,hard_mask=torch.zeros(3,dtype=torch.bool),possible_agents=('m',))
    lp=compute_owner_presence_log_probs(sampled_presence=sampled.presence,**kwargs)['m']
    assert torch.allclose(lp,expected.log_prob(torch.ones(3)).sum())
    assert torch.allclose(compute_owner_presence_entropies(**kwargs)['m'],expected.entropy().sum())
    lp.backward()
    assert all(r.presence_logit.grad is not None and r.presence_logit.grad.item()!=0 for r in rows)


def test_repository_actor_loads_calibration_and_provenance():
    from drl.MARL.tests.test_bc_v4_runtime_observation import make_bridge, ROOT
    from drl.MARL.c1_runner import build_models,CHECKPOINT
    actor,_=build_models(SimpleNamespace(repository_root=ROOT,checkpoint=ROOT/CHECKPOINT,device='cpu'),make_bridge())
    assert actor.presence_threshold_logit==4.5
    assert len(actor.presence_calibration_sha256)==64
    assert actor.presence_calibration_path==str(ROOT/'configs/bc_v1/runtime_inference_v1.json')


@pytest.mark.parametrize('kind',['resource','structural'])
def test_payload_entropy_uses_calibrated_presence_probability(kind):
    from drl.MARL.policy import (StructuralNumericArgumentView,compute_owner_resource_payload_entropies,
        compute_owner_structural_payload_entropies)
    row=MARLCandidateRecord(0,kind,'x','m',(),torch.tensor(4.5),presence_threshold_logit=4.5)
    kwargs=dict(records=(row,),hard_mask=torch.zeros(1,dtype=torch.bool),possible_agents=('m',))
    mean=torch.tensor([19.25]);log_std=torch.tensor(-2.)
    if kind=='resource':
        actual=compute_owner_resource_payload_entropies(**kwargs,resource_payloads={0:ResourcePayloadView(numeric=mean)},log_std=log_std)['m']
    else:
        actual=compute_owner_structural_payload_entropies(**kwargs,
            structural_payloads={0:StructuralPayloadView(numeric=(StructuralNumericArgumentView(0,mean),))},log_std=log_std)['m']
    assert torch.allclose(actual,.5*torch.distributions.Normal(mean,log_std.exp()).entropy().sum())
