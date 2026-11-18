"""BC-v5: immutable-source audit + binary membership mask regression."""
from dataclasses import replace
import math

import pytest
import torch

from drl.IL.bc_v1.bc_v5_binary import (
    binary_presence_loss, candidate_validity, collate_binary_v5,
    compute_binary_v5_loss, index_overlay, binary_presence_metrics)
from drl.IL.bc_v1.bc_v5_integrity import canonical_decision_flags
from drl.IL.tests.bc_v1.g2_fixtures import fixture, tensorize


def _row(conflicted=False, *, r16=False):
    return {'supervision': {
        'valid_for_behavior_cloning': True,
        'module_resource_masks': {
            'm0': {'wheels': True,'internal':not conflicted,'structural':False},
            'm1': {'wheels':False,'internal':True,'structural':False}},
        'conflicts': ([{'module_id':'m0','resource':'internal','reason':'multiple_discrete_decisions_without_intermediate_state'}]
                      if conflicted else []) +
                     ([{'reason':'r16_unresolved_logical_dispatch_order'}] if r16 else [])},
        'action_t': {'modules': {
            'm0': {'wheels':{'commanded':True},'internal':{'commanded':conflicted},'structural':{'commanded':False}},
            'm1': {'wheels':{'commanded':False},'internal':{'commanded':True},'structural':{'commanded':False}}}},
        'provenance': {}}


def test_only_contradictory_commanded_resource_is_masked():
    status=canonical_decision_flags(_row(conflicted=True))
    assert status['eligible'] is True
    assert status['blocked']==[['m0','internal']]
    assert status['bad_temporal'] is False
    status2=canonical_decision_flags(_row(conflicted=False))
    assert status2['blocked']==[]
    assert status2['bad_temporal'] is False
    assert canonical_decision_flags(_row(conflicted=True,r16=True))['bad_temporal'] is True


def test_partial_bc_valid_cannot_be_interpreted_as_full_negative():
    row=_row(conflicted=True)
    assert canonical_decision_flags(row)['blocked'] == [['m0','internal']]
    row['supervision']['module_resource_masks']['m1']['internal']=False
    assert sorted(canonical_decision_flags(row)['blocked']) == [['m0','internal'],['m1','internal']]


def test_bad_conflict_without_localization_forces_graph_exclusion():
    row=_row()
    row['supervision']['conflicts']=[{'reason':'nonsensical_conflict'}]
    assert canonical_decision_flags(row)['bad_temporal']


def test_binary_loss_penalizes_extra_actions_and_ignores_only_masked():
    logits=torch.tensor([3.,3.,-3.,3.],requires_grad=True)
    labels=torch.tensor([True,False,False,False])
    valid=torch.tensor([True,True,True,False])
    loss=binary_presence_loss(logits,labels,valid,torch.zeros(4,dtype=torch.long),1)
    loss.backward()
    # Extra unselected candidate with high logit is punished, as requested.
    assert logits.grad[1] > 0
    assert logits.grad[2] > 0
    assert logits.grad[3] == 0
    assert logits.grad[0] < 0
    assert loss.item() > 1.


def test_all_present_binary_loss_matches_existing_bc_v4():
    from drl.IL.bc_v1.action_set_loss import presence_loss
    logits=torch.tensor([.1,-.2,.3,.4,-.5],requires_grad=True)
    labels=torch.tensor([True,False,False,True,False])
    owners=torch.tensor([0,0,0,1,1])
    valid=torch.ones_like(labels)
    old=presence_loss(logits,labels,owners,2)
    new=binary_presence_loss(logits,labels,valid,owners,2)
    torch.testing.assert_close(old,new)


def test_masking_positive_raises():
    with pytest.raises(ValueError,match='Masked positive'):
        binary_presence_loss(torch.zeros(1),torch.tensor([True]),
                             torch.tensor([False]),torch.tensor([0]),1)


def _entry(sample, *, blocked=(),exclude=False):
    audit=sample['routing']['audit']
    return {'episode_id':audit.episode_id,'transition_index':audit.transition_index,
            'blocked':[list(x) for x in blocked],'exclude':exclude,
            'temporal_kind':'single_decision'}


def test_collation_invariant_and_extra_actions_still_negative():
    src,kwargs=fixture()
    sample=tensorize(src,kwargs)
    route=sample['routing']; audit=route['audit']
    key=(audit.episode_id,audit.transition_index)
    data=collate_binary_v5([sample],{key:_entry(sample)})
    assert data['supervision']['action_set']['valid'].all()
    positive=data['supervision']['action_set']['presence']
    assert positive.sum()==1
    assert (~positive).any()  # explicit negatives retained
    # Unknown-resource masking is not an acceptable fallback.
    false_group=next((r.resource_ref.module.module_id,r.resource_ref.resource_id)
                     for r in route['resource_candidates'] if r.resource_ref.resource_id=='rotor')
    with pytest.raises(ValueError,match='unknown resource category'):
        candidate_validity(route,_entry(sample,blocked=((false_group[0],'internal'),)))


def test_temporal_exclusion_prevents_collation():
    src,kwargs=fixture(); sample=tensorize(src,kwargs)
    with pytest.raises(ValueError,match='Temporally unsafe'):
        candidate_validity(sample['routing'],_entry(sample,exclude=True))


def test_incomplete_overlay_rejected_and_split_isolated():
    with pytest.raises(ValueError,match='complete'):
        index_overlay({'ready_for_training':False},split='train')
    report={'ready_for_training':True,'rows':[dict(split='train',episode_id='e',transition_index=0,
        temporal_kind='joint_single_point',exclude=False,blocked=[])],
        'expected_rows':{'train':1,'validation':0}}
    assert ('e',0) in index_overlay(report,split='train')
    assert index_overlay(report,split='validation')=={}


def test_full_positive_codecs_unchanged_under_binary_v5_mask():
    from drl.IL.tests.bc_v1.test_action_set_loss import parameter_example
    prediction,batch=parameter_example()
    labels=batch['supervision']['action_set']['presence']
    valid=torch.ones_like(labels)
    # Ignore one negative while retaining all positive targets and codecs.
    valid[int((~labels).nonzero()[0])] = False
    batch['supervision']['action_set']['valid']=valid
    total,stats=compute_binary_v5_loss(prediction,batch)
    assert torch.isfinite(total)
    assert stats['resource_payload']['support'] == 1
    assert stats['arg_numeric']['support'] == 2
    total.backward()
    assert (prediction['presence_logits'].grad[labels]<0).all()
    assert prediction['presence_logits'].grad[~valid].abs().sum() == 0
    assert (prediction['presence_logits'].grad[~labels & valid]>0).all()
    metric=binary_presence_metrics(prediction,batch)
    assert metric['masked_candidates']==1
    assert metric['valid_candidates']==int(valid.sum())


def test_full_cross_source_overlay_authenticates_and_never_reads_test(tmp_path):
    import hashlib,json
    from drl.IL.bc_v1 import dataset as g1
    from drl.IL.bc_v1.bc_v5_integrity import inspect,write_report,read_report
    from drl.IL.bc_v1.temporal_supervision import (
        TemporalSupervisionAudit,write_temporal_audit)
    from drl.IL.bc_v1.sidecar import SidecarWriter
    from drl.IL.tests.bc_v1.test_sidecar import identity,row as g2row

    source=tmp_path/'canonical';(source/'episodes').mkdir(parents=True)
    names=[]
    for eid in ('a','v'):
        specimen=_row(conflicted=(eid=='a'))
        specimen['supervision']['module_resource_masks']['m1']['internal']=True
        # BC eligibility for only canonical index 7; the other 7 are invalid.
        records=[]
        for idx in range(8):
            item=json.loads(json.dumps(specimen))
            if idx!=7:item['supervision']['valid_for_behavior_cloning']=False
            records.append(item)
        raw=b''.join((json.dumps(r)+'\n').encode() for r in records)
        path=source/'episodes'/f'{eid}.jsonl';path.write_bytes(raw)
        names.append({'episode_id':eid,'file':f'episodes/{eid}.jsonl',
                      'sha256':hashlib.sha256(raw).hexdigest()})
    (source/'manifest.json').write_text(json.dumps({'episodes':names})+'\n')
    canonical_sha=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()
    ids={**identity(),'canonical_sha':canonical_sha}
    inventory={'train':[{'episode_id':'a','samples':1,'transition_indices_fingerprint':g1._hash([7])}],
               'validation':[{'episode_id':'v','samples':1,'transition_indices_fingerprint':g1._hash([7])}],
               'test':[]}
    # Test membership is deliberately empty.
    writer=SidecarWriter(tmp_path/'g2',ids,inventory)
    for split,eid in (('train','a'),('validation','v')):
        sample=g2row(eid,7)
        sample['routing']['layout_fingerprint']=ids['layout_fingerprint']
        writer.add(sample,split,0)
    writer.finalize()
    temporal=tmp_path/'temporal.json'
    entries=(TemporalSupervisionAudit('a',7,'single_decision'),
             TemporalSupervisionAudit('v',7,'temporally_coalesced_sequential'))
    write_temporal_audit(temporal,entries,{'manifest_sha256':canonical_sha},
                         expected_keys={('a',7),('v',7)})
    audit=inspect(tmp_path/'g2',source,temporal_path=temporal)
    assert audit['ready_for_training'] and audit['test_payload_access']==0
    assert audit['counts']['rows']==2
    assert audit['counts']['excluded_temporally']==1
    assert audit['blocked_resources']['internal']==1
    assert audit['expected_rows']=={'train':1,'validation':1}
    saved=write_report(tmp_path/'result.json',audit)
    validated=read_report(tmp_path/'result.json',expected_g2_fingerprint=\
                          saved['g2_manifest_fingerprint'])
    assert validated['counts']==audit['counts']
    assert index_overlay(validated,split='train')[('a',7)]['exclude'] is False
    assert index_overlay(validated,split='validation')[('v',7)]['exclude'] is True
    without_temporal=inspect(tmp_path/'g2',source)
    assert without_temporal['ready_for_training'] is True
    assert without_temporal['temporal_audit_provided'] is False
    assert without_temporal['counts'].get('excluded_temporally', 0) == 0
    saved_native=write_report(tmp_path/'native.json', without_temporal)
    loaded_native=read_report(tmp_path/'native.json', expected_g2_fingerprint=saved_native['g2_manifest_fingerprint'])
    assert index_overlay(loaded_native,split='validation')[('v',7)]['temporal_kind'] is None


def test_filtered_train_subset_integrates_with_balanced_sampler(tmp_path):
    from drl.IL.tests.bc_v1.test_bc_v5_census_sampling import _source
    from drl.IL.tests.bc_v1.test_sidecar import identity
    from drl.IL.bc_v1.bc_v5_train import FilteredTrainingDataset
    from drl.IL.bc_v1.bc_v5_sampling import build_primitive_train_index, BalancedPrimitiveBatchSampler
    from drl.IL.bc_v1.training_data import TrainingDataset
    root=_source(tmp_path)
    with TrainingDataset(root,'train',expected_identity=identity()) as data:
        rows={}
        for _,e,transition,_,_ in data._data.db.execute(
                "SELECT idx,episode,transition_idx,local_idx,structural FROM samples WHERE split='train' ORDER BY idx"):
            rows[(e,transition)]={'exclude':e=='b'}
        safe=FilteredTrainingDataset(data,rows)
        assert len(safe)==2
        assert safe.get_for_collate(0)['routing']['audit'].episode_id=='a'
        assert safe.get_for_collate(1)['routing']['audit'].episode_id=='a'
        index=build_primitive_train_index(safe)
        assert index.sample_count==2
        sampler=BalancedPrimitiveBatchSampler(index,batch_size=4,balanced_quota=2,seed=7)
        for batch in sampler:
            assert all(0<=i<2 for i in batch)
