import importlib.util
import copy
from dataclasses import replace
import pytest
import torch
from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.tests.bc_v1.g2_fixtures import fixture,tensorize,with_structural,ROOT
from drl.IL.bc_v1.action_set_targets import collate_action_set_samples
from drl.IL.bc_v1.action_set_head import ActionSetPolicy,load_action_set_head_config
from drl.IL.bc_v1.model import load_model_config


@pytest.fixture(autouse=True)
def threads():
    previous=torch.get_num_threads();torch.set_num_threads(1);yield;torch.set_num_threads(previous)


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.action_set_decoder'),'Missing ActionSet pure decoder'
    from drl.IL.bc_v1 import action_set_decoder
    return action_set_decoder


def model(kw):
    return ActionSetPolicy(kw['spec'],kw['feature_schema'],load_model_config(ROOT/'configs/bc_v1/gnn_v1.json'),load_action_set_head_config(ROOT/'configs/bc_v1/action_set_head_v1.json')).eval()


def test_decode_all_present_candidates_not_one_per_initiator_and_empty_is_empty():
    decoder=api();s,kw=fixture();s=with_structural(s)
    targets=s.supervision.structural_targets
    same=replace(targets[1],initiator=targets[0].initiator)
    # Choose two valid, distinct bindings with the SAME initiator.
    candidates=[(r.action,b) for r in s.inputs.candidates.structural_candidates if r.action.action_id=='align_faces' for b in r.bindings if b.initiator==targets[0].initiator]
    actions=tuple(c.StructuralTarget(b.initiator,a.action_id,tuple(p.module for p in b.connectors),b.connectors,(c.ParameterValue('turn','quarter'),)) for a,b in candidates[:2])
    s=replace(s,supervision=c.SupervisionBundle(structural_targets=actions));batch=collate_action_set_samples([tensorize(s,kw)])
    p=model(kw)(batch['inputs']);p['presence_logits']=torch.where(batch['supervision']['action_set']['presence'],10.,-10.)
    bundles=decoder.decode_action_sets(p,batch,kw['normalizers'])
    assert len(bundles)==1 and len(bundles[0].actions)==2
    assert len({a.initiator for a in bundles[0].actions})==1
    p['presence_logits'].fill_(-10.)
    assert decoder.decode_action_sets(p,batch,kw['normalizers'])[0].actions==()
    with pytest.raises(ValueError):decoder.decode_action_sets(p,batch,kw['normalizers'],threshold=float('nan'))
    bad={**p,'candidate_graph':p['candidate_graph'].clone()};bad['candidate_graph'][0]=99
    with pytest.raises(ValueError):decoder.decode_action_sets(bad,batch,kw['normalizers'])


def live_case():
    from drl.IL.tests.bc_v1.test_live_observation import settings,builder,snapshot,context
    from drl.IL.bc_v1.live_observation import encode_smores_snapshot,prepare_live_observation
    from drl.IL.bc_v1.smores_adapter import SMORESAdapter
    from drl.IL.bc_v1.source_frame_bridges import BridgeResolution,SourceFrameBridgeBatch
    kw=settings();state=encode_smores_snapshot(snapshot(),adapter=SMORESAdapter(kw['spec']),graph_builder=builder())
    v=prepare_live_observation(state,context=context(),observation=c.ObservationBundle(),bridges=SourceFrameBridgeBatch((BridgeResolution('world','world','live',None,True,'world',None,()),),(('id','test'),)),**kw)
    return v.batch,kw


def test_pan_wheels_allowed_pan_tilt_conflict_rejected_without_substitution():
    decoder=api();b,kw=live_case();p=model(kw)(b['inputs']);route=b['routing']['samples'][0]
    p['presence_logits'].fill_(-10.)
    for i,row in enumerate(route['resource_candidates']):
        if row.mode.mode_id in ('pan_velocity','wheel_velocity'):p['presence_logits'][i]=10.
    p['parameters']['resource']['payload'].fill_(1.)
    from drl.IL.bc_v1.live_policy import smores_claims
    bundle=decoder.decode_action_sets(p,b,kw['normalizers'])[0]
    assert {a.action_id for a in bundle.actions}=={'pan_velocity','wheel_velocity'}
    assert all(all(v!=0 for v in a.payload) for a in bundle.actions)
    assert decoder.assert_executable(bundle,smores_claims)==bundle
    for i,row in enumerate(route['resource_candidates']):
        if row.mode.mode_id=='tilt_target':p['presence_logits'][i]=10.
    proposed=decoder.decode_action_sets(p,b,kw['normalizers'])[0]
    assert len(proposed.actions)>len(bundle.actions)  # network proposal is preserved for diagnostics/loss.
    with pytest.raises(ValueError,match='Conflicting'):decoder.assert_executable(proposed,smores_claims)
    assert 'tilt_target' in {a.action_id for a in proposed.actions}
