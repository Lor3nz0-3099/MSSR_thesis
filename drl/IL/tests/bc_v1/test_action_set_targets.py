"""Positive-only views over immutable G2, no legacy absence targets."""
import importlib.util
from dataclasses import replace
import copy
import pytest
import torch
from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.tests.bc_v1.g2_fixtures import fixture,tensorize,with_structural


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.action_set'), 'Missing unified ActionBundle ADT'
    from drl.IL.bc_v1 import action_set,action_set_targets
    return action_set,action_set_targets


def test_align_only_bundle_has_no_fabricated_resource_actions():
    adt,view=api();s,kw=fixture();s=with_structural(s)
    s=replace(s,supervision=c.SupervisionBundle(structural_targets=s.supervision.structural_targets[:1]))
    row=tensorize(s,kw);before=copy.deepcopy(row)
    target=view.target_bundle(row)
    assert len(target.actions)==1 and target.actions[0].action_id=='align_faces'
    out=view.collate_action_set_samples([row]);labels=out['supervision']['action_set']['presence']
    assert labels.dtype==torch.bool and int(labels.sum())==1
    assert torch.equal(row['inputs']['module_x'],before['inputs']['module_x'])
    assert row['routing']==before['routing']


def test_mixed_two_dock_four_internal_targets_preserve_all_bindings_payloads():
    adt,view=api();s,kw=fixture(n=6)
    bindings=[(r.action,b) for r in s.inputs.candidates.structural_candidates if r.action.action_id=='dock' for b in r.bindings]
    chosen=[next((a,b) for a,b in bindings if b.initiator==m.ref) for m in s.inputs.policy.graph.modules[:2]]
    targets=tuple(c.StructuralTarget(b.initiator,a.action_id,tuple(p.module for p in b.connectors),b.connectors,(c.ParameterValue('turn','quarter'),)) for a,b in chosen)
    resources=tuple(c.ResourceTarget(r.ref,'aim',(-1.367,)) for r in s.inputs.policy.graph.resources[2:])
    s=replace(s,supervision=c.SupervisionBundle(resources,targets));row=tensorize(s,kw)
    bundle=view.target_bundle(row)
    assert len(bundle.actions)==6
    assert [a.action_id for a in bundle.actions].count('dock')==2
    assert [a.action_id for a in bundle.actions].count('aim')==4
    assert all(a.payload==(-1.367,) for a in bundle.actions if a.resource)
    b=view.collate_action_set_samples([row,row]);assert int(b['supervision']['action_set']['presence'].sum())==12
    assert b['supervision']['structural']['candidate'].numel()==4


def test_seventeen_actions_no_fixed_cardinality_and_order_is_semantic_set():
    adt,view=api();s,kw=fixture(n=17,ports=0)
    s=replace(s,supervision=c.SupervisionBundle(tuple(c.ResourceTarget(r.ref,'pace',(1.,2.)) for r in s.inputs.policy.graph.resources)))
    row=tensorize(s,kw);bundle=view.target_bundle(row)
    assert len(bundle.actions)==17
    assert bundle==adt.ActionBundle(tuple(reversed(bundle.actions)))
    assert hash(bundle)==hash(adt.ActionBundle(tuple(reversed(bundle.actions))))
    assert int(view.collate_action_set_samples([row])['supervision']['action_set']['presence'].sum())==17
    with pytest.raises(ValueError,match='Duplicate'):adt.ActionBundle((bundle.actions[0],bundle.actions[0]))


def test_explicit_zero_is_present_and_absence_is_empty_and_missing_mapping_fails():
    adt,view=api();s,kw=fixture();zero=replace(s.supervision.resource_targets[0],payload=(0.,0.))
    row=tensorize(replace(s,supervision=c.SupervisionBundle((zero,))),kw)
    assert view.target_bundle(row).actions[0].payload==(0.,0.)
    empty=tensorize(replace(s,supervision=c.SupervisionBundle()),kw)
    assert view.target_bundle(row)!=view.target_bundle(empty)==adt.ActionBundle(())
    assert not view.collate_action_set_samples([empty])['supervision']['action_set']['presence'].any()
    bad=copy.deepcopy(row);bad['routing']['resource_candidates']=()
    with pytest.raises(ValueError,match='unique'):view.target_bundle(bad)


def test_missing_normalized_positive_is_not_silently_lost():
    adt,view=api();s,kw=fixture();row=tensorize(s,kw)
    damaged={**row,'routing':{**row['routing'],'resource_targets':()}}
    with pytest.raises(ValueError,match='positive'):view.target_view(damaged)


@pytest.mark.parametrize('index,mode,payload',[(3354,'pan_velocity',(0.,)),(3064,'wheel_velocity',(0.,0.))])
def test_frozen_train_zero_targets_restored_without_legacy_fixed_classes(index,mode,payload):
    import json
    from drl.IL.tests.bc_v1.g2_fixtures import ROOT
    from drl.IL.bc_v1.sidecar import SidecarDataset
    adt,view=api();path=ROOT/'artifacts/bc_v1/sidecar/g2_v3_resource_contract'
    if not (path/'manifest.json').is_file():pytest.skip('external TRAIN fixture unavailable')
    manifest=json.loads((path/'manifest.json').read_text())
    with SidecarDataset(path,'train',expected_identity=manifest['identity']) as data:row=data[index]
    bundle=view.target_bundle(row)
    assert any(a.action_id==mode and a.payload==payload for a in bundle.actions)
    rebuilt=view.target_view(row)
    assert len(rebuilt['supervision']['resource']['candidate'])==len(row['routing']['physical_supervision'].resource_targets)
    assert not {'decision_candidate','decision_supervised'} & rebuilt['supervision']['resource'].keys()


def test_frozen_train_mixed_bundle_is_exactly_two_dock_and_four_tilt():
    import json
    from collections import Counter
    from drl.IL.tests.bc_v1.g2_fixtures import ROOT
    from drl.IL.bc_v1.sidecar import SidecarDataset
    adt,view=api();path=ROOT/'artifacts/bc_v1/sidecar/g2_v3_resource_contract'
    if not (path/'manifest.json').is_file():pytest.skip('external TRAIN fixture unavailable')
    manifest=json.loads((path/'manifest.json').read_text())
    with SidecarDataset(path,'train',expected_identity=manifest['identity']) as data:row=data[3063]
    bundle=view.target_bundle(row)
    assert Counter(a.action_id for a in bundle.actions)=={'dock':2,'tilt_target':4}
    assert len(bundle.actions)==6
    assert int(view.collate_action_set_samples([row])['supervision']['action_set']['presence'].sum())==6
