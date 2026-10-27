"""Synthetic pre-build converter/training contracts. No long jobs or real samples."""
import importlib.util
import json
from pathlib import Path
import torch
import pytest
from drl.IL.tests.bc_v1.test_sidecar import identity,catalog,row
from drl.IL.bc_v1.sidecar import SidecarWriter,SidecarDataset
ROOT=Path(__file__).resolve().parents[4]


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.no_structural_prepare'), 'Missing audited TRAIN/VAL-only v2 converter'
    from drl.IL.bc_v1 import no_structural_prepare
    return no_structural_prepare


def test_converter_never_opens_test_and_preserves_resource_targets_and_inputs(tmp_path):
    a=api();ids=identity();source=tmp_path/'v1';writer=SidecarWriter(source,ids,catalog())
    for split,eid in [('train','a'),('validation','v'),('test','t')]:writer.add(row(eid,structural=split=='train'),split,0)
    manifest=writer.finalize();opened=[]
    def reader(root,split,**kw):
        assert split!='test','TEST payload access forbidden'
        opened.append(split);return SidecarDataset(root,split,**kw)
    out=tmp_path/'v2'
    report=a.convert_sidecar(source,out,source_manifest=manifest,dataset_factory=reader)
    assert opened==['train','validation']
    assert report['counts']['samples']==2 and report['counts']['groups']==6
    assert report['counts']['physical']==2 and report['counts']['no_structural']==4
    assert report['sealed_test']['samples']==1
    doc=json.loads((out/'manifest.json').read_text())
    with SidecarDataset(source,'train',expected_identity=ids) as old,SidecarDataset(out,'train',expected_identity=doc['identity']) as new:
        from drl.IL.bc_v1.sidecar import assert_exact
        assert_exact(new[0]['inputs'],old[0]['inputs'])
        assert_exact(new[0]['supervision']['resource'],old[0]['supervision']['resource'])
        assert new[0]['supervision']['structural']['decision_candidate'].tolist().count(-1)==1
    assert doc['inventory']['test']==[]
    with pytest.raises(ValueError):a.convert_sidecar(source,out,source_manifest=manifest,dataset_factory=reader)


def test_k2_configuration_doubles_positive_quota_without_increasing_batch_size():
    path=ROOT/'configs/bc_v1/training_no_structural_v2.json'
    assert path.is_file(),'Missing prepared K2 training configuration'
    cfg=json.loads(path.read_text())
    assert cfg['batch_size']==64 and cfg['structural_quota']==16 and cfg['general_quota']==48
    assert cfg['early_stopping_patience']==8
    assert cfg['learning_rate']==.0003 and cfg['weight_decay']==.0001


def test_warm_start_reuses_every_existing_weight_and_only_initializes_none():
    assert importlib.util.find_spec('drl.IL.bc_v1.k2_no_structural'),'Missing versioned K2 manual entrypoint'
    from drl.IL.bc_v1.k2_no_structural import warm_start
    from drl.IL.tests.bc_v1.g2_fixtures import fixture
    from drl.IL.tests.bc_v1.test_action_heads import complete_model
    from drl.IL.tests.bc_v1.test_no_structural import v2_model
    _,kw=fixture();old=complete_model(kw);new=v2_model(kw)
    report=warm_start(new,old.state_dict())
    assert report['new_parameters'] and all('no_structural_scorer' in name for name in report['new_parameters'])
    for name,value in old.state_dict().items():assert torch.equal(new.state_dict()[name],value)


def test_new_preparation_entrypoints_reject_test_before_reading_any_payload(tmp_path):
    a=api()
    assert 'test' not in a.MATERIALIZED_SPLITS


def test_prebuild_positive_count_mismatch_fails_before_publishing(tmp_path):
    a=api();source=tmp_path/'v1';ids=identity();writer=SidecarWriter(source,ids,catalog())
    for split,eid in [('train','a'),('validation','v'),('test','t')]:writer.add(row(eid,structural=split=='train'),split,0)
    doc=writer.finalize();out=tmp_path/'bad-v2'
    assert 'expected_positive_counts' in __import__('inspect').signature(a.convert_sidecar).parameters, 'Positive census must be checked before publication'
    with pytest.raises(ValueError,match='Positive'):
        a.convert_sidecar(source,out,source_manifest=doc,expected_positive_counts={'align_faces':99})
    assert not out.exists()


def test_checkpoint_safety_gate_rejects_always_none_and_retains_positive_model():
    from drl.IL.bc_v1.k2_no_structural import collapsed
    from drl.IL.bc_v1.structural_decisions import StructuralDecisionMetrics
    missed=StructuralDecisionMetrics(('align_faces',));missed.add('align_faces',None,False)
    correct=StructuralDecisionMetrics(('align_faces',));correct.add('align_faces','align_faces',True)
    assert collapsed(missed.compute()) and not collapsed(correct.compute())


def test_warm_start_rejects_loss_of_preexisting_head_weight():
    from drl.IL.bc_v1.k2_no_structural import warm_start
    from drl.IL.tests.bc_v1.g2_fixtures import fixture
    from drl.IL.tests.bc_v1.test_action_heads import complete_model
    from drl.IL.tests.bc_v1.test_no_structural import v2_model
    _,kw=fixture();state=dict(complete_model(kw).state_dict());state.pop(next(iter(state)))
    with pytest.raises(ValueError,match='exact'):warm_start(v2_model(kw),state)


def test_early_stopping_patience_tracks_eligible_candidates_not_collapsed_low_objectives():
    from drl.IL.bc_v1 import k2_no_structural as k2
    selection_type=getattr(k2,'CandidateSelection',None)
    assert selection_type is not None,'K2 patience must track eligible checkpoint improvement'
    selector=selection_type(patience=8,min_delta=1e-4)
    bad={'positive':{'support':1,'recall':0.,'accuracy':0.}}
    good={'positive':{'support':1,'recall':1.,'accuracy':1.}}
    assert not selector.observe_candidate(0,.1,bad)
    assert selector.observe_candidate(1,3.,good)
    assert selector.observe_candidate(2,2.,good)
    assert selector.best_epoch==2 and selector.counter==0
    for epoch in range(3,11):selector.observe_candidate(epoch,.01,bad)
    assert selector.should_stop and selector.best_epoch==2
