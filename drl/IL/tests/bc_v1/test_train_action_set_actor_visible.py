"""Actor training admission and orchestration, CPU-only and no optimizer steps."""
import copy
import json
from pathlib import Path

import pytest

from drl.IL.bc_v1 import train_action_set as runner
from drl.IL.bc_v1.normalization import normalizer_fingerprint
from drl.IL.bc_v1.sidecar import _manifest_fingerprint
from drl.IL.tests.bc_v1.test_actor_prepare import tiny, fit, policy


@pytest.fixture
def actor_metadata(tiny, tmp_path, monkeypatch):
    from drl.IL.bc_v1.actor_prepare import publish_actor_normalizers
    split, _, kw, _, _, old = tiny
    stats, _ = fit(tiny)
    normalizers = tmp_path/'actor_f2'
    context = publish_actor_normalizers(stats, normalizers, source_root=tmp_path/'physical', visibility=policy())
    manifest = copy.deepcopy(old)
    manifest['identity'].update(f2_fingerprint=normalizer_fingerprint(stats),
        normalizer_sha=context['normalizer_sha256'], canonical_sha=split['dataset_identity']['manifest_sha256'])
    counts = {'train': 2, 'validation': 1}
    manifest['actor_visibility'] = dict(schema_version='mssr.bc_v1.actor_inputs.v1',
        fingerprint=policy().fingerprint, policy=policy().document, sample_counts=counts)
    manifest['manifest_fingerprint'] = _manifest_fingerprint(manifest)
    source = tmp_path/'actor_source'; source.mkdir()
    (source/'manifest.json').write_text(json.dumps(manifest))
    success = dict(manifest_fingerprint=manifest['manifest_fingerprint'], fingerprint=policy().fingerprint,
        sample_counts=counts, normalizer_context=context, TEST_payload_access=0,
        verification=dict(integrity='PASS', manifest_fingerprint=manifest['manifest_fingerprint'], TEST_payload_access=0))
    (source/'SUCCESS.json').write_text(json.dumps(success))
    monkeypatch.setattr(runner, 'load_robot_spec', lambda path: kw['spec'], raising=False)
    monkeypatch.setattr(runner, 'load_feature_schema', lambda path: kw['feature_schema'], raising=False)
    monkeypatch.setattr(runner, 'load_normalization_policy', lambda path, schema: kw['normalization_policy'], raising=False)
    def forbidden(*args, **kwargs): raise AssertionError('Legacy preprocessing, dataset or model accessed in static actor preflight')
    for name in ('load_sources','build_scratch_actor','open_dataset'): monkeypatch.setattr(runner, name, forbidden)
    return source, normalizers, manifest, kw, stats


def test_actor_preflight_authenticates_explicit_f2_without_legacy_sources(actor_metadata):
    source, normalizers, manifest, _, stats = actor_metadata
    cfg = runner.load_action_set_training_config(runner.DEFAULT_CONFIG)
    report, kw, _, _, checked = runner.preflight(source, cfg, normalizer_dir=normalizers)
    assert checked == manifest
    assert normalizer_fingerprint(kw['normalizers']) == normalizer_fingerprint(stats)
    assert report['source'] == str(source.resolve())
    assert report['normalizer_directory'] == str(normalizers.resolve())
    assert report['actor_visibility']['fingerprint'] == policy().fingerprint
    assert report['splits']['train']['samples'] == 2 and report['splits']['validation']['samples'] == 1
    assert report['TEST_payload_access'] == 0


def test_actor_cli_dry_run_accepts_explicit_normalizer_directory(actor_metadata, tmp_path, capsys):
    source, normalizers, _, _, _ = actor_metadata
    output = tmp_path/'never_created'
    assert runner.main(['--g2',str(source),'--normalizers',str(normalizers),'--output',str(output)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['mode'] == 'DRY_RUN' and report['initialization']['mode'] == 'scratch'
    assert not output.exists()


@pytest.mark.parametrize('mutation', ['missing_success','foreign_success','foreign_visibility','wrong_normalizer','missing_normalizers','foreign_fit_context'])
def test_actor_preflight_rejects_missing_or_foreign_certificates(actor_metadata, tmp_path, mutation):
    source, normalizers, manifest, _, _ = actor_metadata
    if mutation == 'missing_success': (source/'SUCCESS.json').unlink()
    elif mutation == 'foreign_success':
        doc=json.loads((source/'SUCCESS.json').read_text()); doc['manifest_fingerprint']='foreign'
        (source/'SUCCESS.json').write_text(json.dumps(doc))
    elif mutation == 'foreign_visibility':
        manifest['actor_visibility']['fingerprint']='foreign'; manifest['manifest_fingerprint']=_manifest_fingerprint(manifest)
        (source/'manifest.json').write_text(json.dumps(manifest))
    elif mutation == 'wrong_normalizer':
        (normalizers/'normalizers.json').write_text('{}')
    elif mutation == 'missing_normalizers': normalizers=None
    else:
        doc=json.loads((normalizers/'fit_context.json').read_text()); doc['TRAIN_only']=False
        (normalizers/'fit_context.json').write_text(json.dumps(doc))
    with pytest.raises(ValueError): runner.preflight(source,runner.load_action_set_training_config(runner.DEFAULT_CONFIG),normalizer_dir=normalizers)


def test_production_selection_strict_absolute_boundary_reset_and_eight_bad_epochs():
    cfg=runner.load_action_set_training_config(runner.DEFAULT_CONFIG)
    assert cfg==dict(schema_version='mssr.bc_v1.training_action_set.v1',model_contract='action_set_v1',initialization='scratch',seed=20261021,lr=.0003,weight_decay=.0001,batch_size=64,max_epochs=30,patience=8,min_delta=.002,threshold_logit=0.)
    for value,promoted in ((.0981,False),(.0980,False),(.0979,True)):
        state=runner.selection_state(cfg)
        assert state.observe(0,.1)
        assert state.observe(1,value) is promoted
        assert state.counter==(0 if promoted else 1)
    state=runner.selection_state(cfg);state.observe(0,.1)
    for epoch in range(1,4): assert not state.observe(epoch,.099)
    assert state.counter==3
    assert state.observe(4,.0979) and state.counter==0
    for epoch in range(5,13):
        assert not state.observe(epoch,.0979)
        assert state.counter==epoch-4 and state.should_stop==(epoch==12)


def simulated_run(tmp_path,monkeypatch,losses):
    """Exercise orchestration with in-memory training/VAL/checkpoint boundaries.

    Real gradient updates and full evaluation are prohibited in preparation.
    SelectionState, history persistence, loop and checkpoint decisions are real.
    """
    from contextlib import nullcontext
    train_calls=[];saves=[];evaluated=[]
    class Samples:
        def __len__(self):return 1
        def train_index(self):return None
        def get_for_collate(self,i):return None
    cfg=runner.load_action_set_training_config(runner.DEFAULT_CONFIG)
    monkeypatch.setattr(runner.torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(runner,'build_scratch_actor',lambda *a,**k:(None,None,{}))
    monkeypatch.setattr(runner,'open_dataset',lambda *a,**k:nullcontext(Samples()))
    monkeypatch.setattr(runner,'JointActionBatchSampler',lambda *a,**k:[[0]])
    monkeypatch.setattr(runner,'collate_action_set_samples',lambda x:x)
    monkeypatch.setattr(runner,'move_batch_to_device',lambda x,d:x)
    def train(*args):
        train_calls.append(1)
        return {'families':{k:{'sum':0.,'support':1} for k in runner.FAMILIES}}
    def evaluate(*args,**kwargs):
        epoch=len(evaluated);evaluated.append(epoch)
        return {'total_loss':losses[epoch],'metrics':{'epoch_witness':epoch,'set_f1':1.-epoch/100.,'target_cardinality_min':1,'target_cardinality_max':3}}
    def save(path,model,optimizer,identity,config,summary,*,step):
        saves.append((path.name,copy.deepcopy(summary),step))
    monkeypatch.setattr(runner,'train_batch',train)
    monkeypatch.setattr(runner,'evaluate',evaluate)
    monkeypatch.setattr(runner,'save_action_set_checkpoint',save)
    report={'sampler_quotas':{'ordinary':48,'structural':16},'initialization':{'mode':'scratch'},'runtime':runner.RUNTIME,'dropout':{'h1':.1,'h2':.1},'actor_visibility':{'fingerprint':'certified-visible'}}
    output=tmp_path/'new_run'
    assert runner.run(output,tmp_path/'source',cfg,(report,{'spec':None,'feature_schema':None},{},{},{'identity':{}}))==0
    history=[json.loads(s) for s in (output/'history.jsonl').read_text().splitlines()]
    return train_calls,saves,evaluated,history,json.loads((output/'status.json').read_text())


def test_initial_val_then_thirty_trained_epochs_and_same_epoch_checkpoint_snapshot(tmp_path,monkeypatch):
    trained,saved,evaluated,history,status=simulated_run(tmp_path,monkeypatch,[1.-.02*i for i in range(31)])
    assert len(trained)==30 and evaluated==list(range(31))
    assert history[0]['train'] is None and history[0]['epoch_kind']=='INITIAL_RANDOM_VAL'
    assert history[0]['trained_epochs']==0 and history[-1]['trained_epochs']==30
    assert history[-1]['global_step']==30
    assert len([s for s in saved if s[0]=='last.pt'])==31
    for name,row,step in saved:
        epoch=row['epoch']
        assert row['validation']['metrics']['epoch_witness']==epoch
        assert row['validation']['total_loss']==1.-.02*epoch
        assert row['actor_visibility']['fingerprint']=='certified-visible' and step==epoch
    assert status['selection']['best_epoch']==30 and status['phase']=='COMPLETE'


def test_real_selection_loop_stops_at_eighth_bad_epoch_and_never_selects_on_f1(tmp_path,monkeypatch):
    trained,saved,_,history,status=simulated_run(tmp_path,monkeypatch,[.1,.09]+[.089]*8)
    assert len(trained)==9 and len(history)==10
    assert [(name,row['epoch']) for name,row,_ in saved if name=='best.pt']==[('best.pt',0),('best.pt',1)]
    assert saved[-1][0]=='last.pt' and saved[-1][1]['epoch']==9
    assert status['selection']['counter']==8 and status['selection']['best_epoch']==1
