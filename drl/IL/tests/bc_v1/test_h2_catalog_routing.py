"""Structural performance invariants, without wall-clock thresholds."""
import torch

from drl.IL.tests.bc_v1.test_resource_heads_v3 import example, threads
from drl.IL.bc_v1.batching import collate_policy_samples


def test_resource_routing_scalar_reads_do_not_scale_with_candidates(monkeypatch):
    batch, model, _, _ = example()
    row = dict(inputs=batch['inputs'], supervision=batch['supervision'],
               routing=batch['routing']['samples'][0])
    large = collate_policy_samples([row]*8)
    reads = []
    original = torch.Tensor.__int__

    def counted(value):
        reads.append(value.numel())
        return original(value)

    latents = model.encode(large['inputs'])
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, '__int__', counted)
        output = model.action_heads.forward_resource(latents, large['inputs'])
    assert len(output['outcome_logits']) == 8*12
    assert len(reads) <= 16, f'Per-candidate scalar routing: {len(reads)} reads'


def test_catalog_metadata_is_not_added_to_checkpoint_state():
    _, model, _, _ = example()
    assert not any('_catalog_' in key for key in model.state_dict())
    # Catalog tables follow normal device/dtype movement but have no gradients.
    tables = [(name,value) for name,value in model.action_heads.named_buffers()
              if name.startswith('_catalog_')]
    assert tables, 'Static catalog routing should be initialized once'
    assert all(not value.requires_grad and value.dtype in (torch.int64,torch.bool)
               for _,value in tables)


def test_structural_option_linear_work_is_batched():
    from drl.IL.tests.bc_v1.g2_fixtures import fixture,tensorize
    from drl.IL.tests.bc_v1.test_action_heads import heads_for,model_for
    sample,kw=fixture(5,0)
    x=tensorize(sample,kw)['inputs'];heads=heads_for(kw)
    latents=model_for(kw).encode(x)
    calls={'scope':0,'reference_set':0}
    handles=[]
    for key,scorer in [('scope',heads.scope_scorer),('reference_set',heads.reference_set_scorer)]:
        def count(module,args,output,key=key):calls[key]+=1
        handles.append(scorer[0].register_forward_hook(count))
    try:
        out=heads.forward_structural(latents,x)
    finally:
        for h in handles:h.remove()
    assert len(out['scope']['logits'])==20
    assert len(out['arguments']['reference_set']['logits'])==25
    assert calls=={'scope':1,'reference_set':1},calls


def test_segmented_scorer_keeps_training_dropout_rng_and_gradients():
    from drl.IL.bc_v1.action_heads import _segmented_score
    scorer=torch.nn.Sequential(torch.nn.Linear(4,7),torch.nn.SiLU(),
                               torch.nn.Dropout(.1),torch.nn.Linear(7,1)).train()
    values=torch.randn(9,4,requires_grad=True);ptr=torch.tensor([0,2,2,5,9])
    torch.manual_seed(143)
    expected=torch.cat([scorer(values[a:b]) for a,b in zip(ptr[:-1],ptr[1:])])
    rng=torch.get_rng_state().clone()
    expected_grad=torch.autograd.grad(expected.sum(),values)[0]
    torch.manual_seed(143)
    actual=_segmented_score(scorer,values,ptr)
    torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)
    assert torch.equal(rng,torch.get_rng_state())
    torch.testing.assert_close(torch.autograd.grad(actual.sum(),values)[0],expected_grad,
                               atol=1e-7,rtol=1e-6)
