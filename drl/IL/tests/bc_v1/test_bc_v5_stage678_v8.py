"""Stage 6/7/8 v8: candidate mask is loss-only; no DATA/G2 scans."""
from dataclasses import replace
import pytest
import torch

from drl.IL.tests.bc_v1.g2_fixtures import fixture,tensorize
from drl.IL.tests.bc_v1.test_candidates import attach
from drl.IL.bc_v1.candidates import enumerate_candidates
from drl.IL.tests.bc_v1.test_candidates import attach
from drl.IL.bc_v1.bc_v5_binary import candidate_validity, binary_presence_loss
from drl.IL.bc_v1.bc_v5_v8_readiness import load_readiness
from drl.IL.bc_v1.bc_v5_v8_sampling import AdaptivePrimitiveBatchSampler


def test_align_mask_never_changes_candidate_catalogue():
    src,kw=fixture(3,2)
    graph=attach(src.inputs.policy.graph,0,2)
    choices=enumerate_candidates(graph,kw['spec'],kw['feature_schema'])
    src=replace(src,inputs=replace(src.inputs,
        policy=replace(src.inputs.policy,graph=graph),candidates=choices))
    sample=tensorize(src,kw)
    route=dict(sample['routing'])
    route['bc_v5_masked_align_initiators']=('unit-0',)
    route['bc_v5_masked_align_count']=1
    original=len(route['resource_candidates'])+len(route['structural_bindings'])
    audit=route['audit']
    overlay={'episode_id':audit.episode_id,'transition_index':audit.transition_index,
             'blocked':[],'exclude':False}
    valid=candidate_validity(route,overlay)
    assert len(valid)==original
    start=len(route['resource_candidates'])
    checked=0
    for i,(action,binding) in enumerate(route['structural_bindings']):
        if action.action_id=='align_faces' and binding.initiator.module_id=='unit-0':
            assert not valid[start+i]
            checked+=1
    assert checked>0,'Fixture must exercise a real candidate mask'
    # No mutation: the model still has the same number of candidate logits.
    assert len(route['resource_candidates'])+len(route['structural_bindings'])==original


def test_invalid_align_logit_has_no_gradient():
    logits=torch.tensor([2.,3.,-2.],requires_grad=True)
    positive=torch.tensor([True,False,False])
    valid=torch.tensor([True,False,True])
    loss=binary_presence_loss(logits,positive,valid,torch.zeros(3,dtype=torch.int64),1)
    loss.backward()
    assert logits.grad[1]==0
    assert logits.grad[0]<0
    assert logits.grad[2]>0


def test_no_positive_graph_is_not_treated_as_all_negative():
    with pytest.raises(ValueError, match='without positive'):
        binary_presence_loss(torch.zeros(2),torch.tensor([False,False]),
                             torch.tensor([True,True]),torch.zeros(2,dtype=torch.int64),1)


def test_sampler_uses_only_train_and_grows_exact_positive_pools(tmp_path):
    from drl.IL.tests.bc_v1.test_bc_v5_census_sampling import _source
    from drl.IL.tests.bc_v1.test_sidecar import identity
    from drl.IL.bc_v1.bc_v5_train import FilteredTrainingDataset
    from drl.IL.bc_v1.training_data import TrainingDataset
    root=_source(tmp_path)
    with TrainingDataset(root,'train',expected_identity=identity()) as base:
        rows={}
        for _,eid,transition in base._data.db.execute(
                "SELECT idx,episode,transition_idx FROM samples WHERE split='train' ORDER BY idx"):
            rows[(eid,transition)]={'exclude':False}
        dataset=FilteredTrainingDataset(base,rows)
        sampler=AdaptivePrimitiveBatchSampler(dataset,batch_size=2,balanced_quota=1,
            seed=4,bootstrap_uniform=2,bootstrap_structural=1)
        initial=sampler.summary()
        assert initial['train_rows']==3
        assert initial['bootstrap_exact_rows']<=3
        assert len(list(sampler))==2
        assert all(0<=idx<3 for idx in next(iter(sampler)))
        sampler.observe_batch([0],[dataset.get_for_collate(0)])
        assert len(sampler.observed)>=initial['bootstrap_exact_rows']
