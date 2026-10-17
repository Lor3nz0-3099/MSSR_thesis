from dataclasses import replace
import importlib
import importlib.util

import pytest
import torch

from drl.IL.tests.bc_v1.g2_fixtures import fixture, tensorize, with_structural


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.batching'), 'Missing variable G2 collate'
    return importlib.import_module('drl.IL.bc_v1.batching')


def pair():
    a,ka=fixture(2,2,attached=True)
    b,kb=fixture(5,2,geometry=False)
    return tensorize(a,ka),tensorize(with_structural(b,skill=True),kb)


def test_offsets_boundaries_and_no_cross_graph_edges():
    batcher=api(); a,b=pair(); batch=batcher.collate_policy_samples([a,b]); x=batch['inputs']
    assert x['graph_ptr'].tolist()==[0,2,7]
    assert x['module_x'].shape==(7,30)
    assert x['connector_to_module'].tolist()==[0,0,1,1,2,2,3,3,4,4,5,5,6,6]
    assert x['resource_to_module'].tolist()==list(range(7))
    assert x['coordinate_to_resource'].tolist()==[i for i in range(7) for _ in range(2)]
    assert x['connection_connectors'].tolist()==[[0,2]]
    owner=x['connector_to_module'][x['connection_connectors']]
    assert (owner<2).all()
    assert x['module_batch'].tolist()==[0,0,1,1,1,1,1]
    assert len(batch['routing']['samples'])==2
    assert 'audit' not in x and 'routing' not in x


def test_candidate_group_argument_and_supervision_offsets_with_sentinels():
    batcher=api(); a,b=pair(); batch=batcher.collate_policy_samples([a,b]); x=batch['inputs']; y=batch['supervision']
    qa=len(a['inputs']['structural']['initiator']); qr=len(a['inputs']['resource_candidates']['resource'])
    assert y['resource']['candidate'].tolist()==[0,qr]
    assert y['resource']['resource'].tolist()==[0,2]
    assert y['structural']['candidate'].tolist()==[qa+int(i) for i in b['supervision']['structural']['candidate']]
    assert y['structural']['group'].tolist()==[2,3]
    sb=x['structural']; original=b['inputs']['structural']
    assert torch.equal(sb['peer'][qa:],torch.where(original['peer']<0,original['peer'],original['peer']+2))
    assert torch.equal(sb['connector_a'][qa:],torch.where(original['connector_a']<0,original['connector_a'],original['connector_a']+4))
    assert sb['group_ptr'].numel()==8
    assert sb['argument_binding'].min()>=0
    for i,(kind,start,end) in enumerate(zip(sb['argument_kind'],sb['argument_domain_ptr'][:-1],sb['argument_domain_ptr'][1:])):
        values=sb['argument_domain_index'][start:end]
        if values.numel():assert values.max() < (14 if int(kind) in (5,7) else 7)
    assert y['structural']['membership_domain'].max() < sb['argument_domain_index'].numel()


def test_empty_optional_connectors_edges_and_environment_are_legal():
    batcher=api(); sample,kwargs=fixture(2,2,geometry=False)
    from drl.IL.bc_v1.candidates import enumerate_candidates
    graph=replace(sample.inputs.policy.graph,connectors=())
    sample=replace(sample,inputs=replace(sample.inputs,
        policy=replace(sample.inputs.policy,graph=graph),features=replace(sample.inputs.features,connectors=()),
        candidates=enumerate_candidates(graph,kwargs['spec'],kwargs['feature_schema'])))
    t=tensorize(sample,kwargs); out=batcher.collate_policy_samples([t,t])
    assert out['inputs']['connector_x'].shape[0]==0
    assert out['inputs']['connection_connectors'].shape==(0,2)
    assert out['inputs']['environment']=={}
    assert out['inputs']['structural']['connector_a'].tolist()==[-1]*4


def test_batch_rejects_incompatible_layout_and_empty_batch():
    batcher=api(); a,b=pair(); b['routing']['layout_fingerprint']='wrong'
    with pytest.raises(ValueError,match='incompatible'):batcher.collate_policy_samples([a,b])
    with pytest.raises(ValueError):batcher.collate_policy_samples([])
def test_participant_scope_membership_shifts_to_its_graph_domain():
    from drl.IL.tests.bc_v1.g2_fixtures import fixture, tensorize, with_structural
    from drl.IL.bc_v1.batching import collate_policy_samples
    first, k1 = fixture(2, 0)
    second, k2 = fixture(5, 0)
    a = tensorize(with_structural(first, skill=True), k1)
    b = tensorize(with_structural(second, skill=True), k2)
    merged = collate_policy_samples([a,b])
    y = merged['supervision']['structural']
    assert y['scope_membership_ptr'].tolist() == [0,2,4,9,14]
    offset = len(a['inputs']['structural']['scope_members'])
    assert torch.equal(y['scope_membership_index'][4:], b['supervision']['structural']['scope_membership_index']+offset)
    assert torch.equal(y['scope_membership_value'][4:], b['supervision']['structural']['scope_membership_value'])
    assert torch.equal(y['scope_membership_valid'][4:], b['supervision']['structural']['scope_membership_valid'])
