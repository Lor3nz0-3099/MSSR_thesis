"""CPU-only scratch/selection/checkpoint and TRAIN/VAL admission gates."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from drl.IL.bc_v1.bc_v1_contract import robot_spec_fingerprint
from drl.IL.bc_v1.feature_schema import feature_schema_fingerprint
from drl.IL.bc_v1.model import load_model_config
from drl.IL.bc_v1.sidecar import VERSION, _hash, _manifest_fingerprint
from drl.IL.bc_v1.tensorization import load_tensor_layout
from drl.IL.tests.bc_v1.g2_fixtures import fixture, tensorize
from drl.IL.tests.bc_v1.test_model import equal_trees
from drl.IL.tests.bc_v1.test_sidecar import identity as source_identity

ROOT=Path(__file__).resolve().parents[4]


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.action_set_training'), 'Missing ActionSet scratch/checkpoint APIs'
    return importlib.import_module('drl.IL.bc_v1.action_set_training')


def case():
    sample,kw=fixture(2)
    h1=load_model_config(ROOT/'configs/bc_v1/gnn_v1.json')
    h2=json.loads((ROOT/'configs/bc_v1/action_set_head_v1.json').read_text())
    ids={**source_identity(),'robot_spec_fingerprint':robot_spec_fingerprint(kw['spec']),
        'feature_schema_fingerprint':feature_schema_fingerprint(kw['feature_schema']),
        'layout_fingerprint':load_tensor_layout(ROOT/h1['tensor_layout'])['fingerprint']}
    inventory={'train':[{'episode_id':'synthetic','samples':2,'transition_indices_fingerprint':_hash([0,1])}],
        'validation':[],'test':[]}
    manifest=dict(schema_version=VERSION,complete=True,identity=ids,inventory=inventory,membership_fingerprint=_hash(inventory))
    manifest['manifest_fingerprint']=_manifest_fingerprint(manifest)
    return sample,kw,h1,h2,manifest


def config():
    return api().load_action_set_training_config(ROOT/'configs/bc_v1/training_action_set_v1.json')


def build():
    sample,kw,h1,h2,manifest=case()
    model,opt,identity=api().build_scratch_actor(kw['spec'],kw['feature_schema'],h1,h2,manifest,config())
    return sample,kw,h1,h2,manifest,model,opt,identity


def test_scratch_is_reproducible_and_optimizer_covers_every_parameter_once():
    *_,model,opt,identity=build()
    *_,other,_,other_identity=build()
    equal_trees(model.state_dict(),other.state_dict())
    assert identity==other_identity
    assert identity['model_contract']=='action_set_v1'
    assert identity['source_manifest_fingerprint']==case()[-1]['manifest_fingerprint']
    assert identity['source_identity']==case()[-1]['identity']
    assert identity['decode_config']=={'threshold_logit':0.0,'configured_arguments':{}}
    parameters=[p for group in opt.param_groups for p in group['params']]
    assert len(parameters)==len({id(p) for p in parameters})==len(list(model.parameters()))
    assert {id(p) for p in parameters}=={id(p) for p in model.parameters()}
    assert isinstance(opt,torch.optim.AdamW)
    assert all(p.device.type=='cpu' for p in model.parameters())


@pytest.mark.parametrize('field,value',[('schema_version','legacy'),('initialization','pretrained'),
    ('model_contract','resource_v3'),('batch_size',0),('threshold_logit',float('nan')),
    ('lr',0.),('weight_decay',-1.),('patience',0),('min_delta',float('nan')),('extra',True)])
def test_training_config_fails_closed(field,value):
    sample,kw,h1,h2,manifest=case()
    with pytest.raises(ValueError):
        api().build_scratch_actor(kw['spec'],kw['feature_schema'],h1,h2,manifest,{**config(),field:value})


@pytest.mark.parametrize('mutation',['incomplete','schema','checksum','test','robot','layout','feature','missing_f2'])
def test_scratch_rejects_incompatible_or_unauthenticated_source(mutation):
    _,kw,h1,h2,manifest=case()
    if mutation=='incomplete':manifest['complete']=False
    elif mutation=='schema':manifest['schema_version']='old'
    elif mutation=='checksum':manifest['identity']['f2_fingerprint']='tampered'
    elif mutation=='test':manifest['inventory']['test']=[{'episode_id':'forbidden','samples':1,'transition_indices_fingerprint':'x'}]
    elif mutation=='missing_f2':manifest['identity'].pop('f2_fingerprint')
    else:manifest['identity'][{'robot':'robot_spec_fingerprint','layout':'layout_fingerprint','feature':'feature_schema_fingerprint'}[mutation]]='incompatible'
    if mutation!='checksum':
        manifest['membership_fingerprint']=_hash(manifest['inventory'])
        manifest['manifest_fingerprint']=_manifest_fingerprint(manifest)
    with pytest.raises(ValueError):api().build_scratch_actor(kw['spec'],kw['feature_schema'],h1,h2,manifest,config())


def test_selection_uses_configured_patience_and_strict_min_delta():
    cfg={**config(),'patience':2,'min_delta':.25};state=api().selection_state(cfg)
    assert state.observe(0,2.) and not state.trained_improvement
    assert not state.observe(1,1.75) and not state.should_stop
    assert state.observe(2,1.5) and state.trained_improvement
    assert not state.observe(3,1.4) and not state.should_stop
    assert not state.observe(4,1.3) and state.should_stop


def test_test_split_is_rejected_before_dataset_constructor(monkeypatch):
    module=api();calls=[]
    def forbidden(*args,**kwargs):calls.append((args,kwargs));raise AssertionError('constructor reached')
    monkeypatch.setattr(module,'TrainingDataset',forbidden)
    for split in ('test','TEST','all'):
        with pytest.raises(ValueError,match='TEST'):module.open_dataset('/never/open',split,{})
    assert calls==[]


def checkpoint(tmp_path):
    sample,kw,h1,h2,manifest,model,opt,identity=build()
    model.eval();x=tensorize(sample,kw)['inputs']
    # Exercise moments as well as initial model tensors, without a BC loop.
    sum(v.square().mean() for v in (model(x)['presence_logits'],)).backward();opt.step();opt.zero_grad()
    path=tmp_path/'actor.pt'
    api().save_action_set_checkpoint(path,model,opt,identity,config(),{'check':'cpu-roundtrip'},step=1)
    return path,model,kw,h1,h2,identity,x


def test_cpu_checkpoint_roundtrip_has_exact_outputs_and_portable_state(tmp_path):
    path,model,kw,h1,h2,identity,x=checkpoint(tmp_path)
    payload=torch.load(path,map_location='cpu',weights_only=False)
    assert payload['schema_version']=='mssr.bc_v1.checkpoint.action_set.v1'
    assert payload['step']==1 and payload['summary']=={'check':'cpu-roundtrip'}
    assert all(v.device.type=='cpu' for v in payload['model'].values())
    assert payload['rng_state']['torch'].device.type=='cpu'
    restored=api().load_action_set_checkpoint(path,kw['spec'],kw['feature_schema'],h1,h2,expected_identity=identity)
    equal_trees(model(x),restored.eval()(x))
    from drl.IL.bc_v1.training import load_checkpoint,build_optimizer
    with pytest.raises(ValueError,match='schema'):
        load_checkpoint(path,restored,build_optimizer(restored),expected_identity=identity)


@pytest.mark.parametrize('mutation',['schema','identity','config','head','nan','missing_key','wrong_dtype','source_f2','source_layout'])
def test_checkpoint_rejects_corruption_before_returning_actor(tmp_path,mutation):
    path,model,kw,h1,h2,identity,x=checkpoint(tmp_path)
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if mutation=='schema':payload['schema_version']='mssr.bc_v1.training_checkpoint.v1'
    elif mutation=='identity':payload['identity']['model_contract']='resource_v3'
    elif mutation=='config':payload['config']['lr']=.5
    elif mutation=='head':h2={**h2,'schema_version':'mssr.bc_v1.action_heads.v3'}
    elif mutation=='source_f2':identity={**identity,'source_identity':{**identity['source_identity'],'f2_fingerprint':'wrong-fold'}}
    elif mutation=='source_layout':identity={**identity,'source_identity':{**identity['source_identity'],'layout_fingerprint':'wrong-layout'}}
    else:
        key=next(iter(payload['model']))
        if mutation=='nan':payload['model'][key].flatten()[0]=float('nan')
        elif mutation=='missing_key':del payload['model'][key]
        else:payload['model'][key]=payload['model'][key].double()
    torch.save(payload,path)
    with pytest.raises((ValueError,RuntimeError)):
        api().load_action_set_checkpoint(path,kw['spec'],kw['feature_schema'],h1,h2,expected_identity=identity)


def test_failed_atomic_save_preserves_previous_checkpoint(tmp_path,monkeypatch):
    path,model,kw,h1,h2,identity,x=checkpoint(tmp_path);before=path.read_bytes()
    def interrupted(*args,**kwargs):raise OSError('simulated interrupted write')
    monkeypatch.setattr(torch,'save',interrupted)
    with pytest.raises(OSError):api().save_action_set_checkpoint(path,model,None,identity,config(),{},step=2)
    assert path.read_bytes()==before
    assert list(tmp_path.iterdir())==[path]


def test_config_accepts_explicit_positive_batch_size_and_finite_threshold():
    _,kw,h1,h2,manifest=case()
    cfg={**config(),'batch_size':32,'threshold_logit':.5}
    model,optimizer,identity=api().build_scratch_actor(kw['spec'],kw['feature_schema'],h1,h2,manifest,cfg)
    assert identity['decode_config']['threshold_logit']==.5
    assert identity['training_config_fingerprint']==_hash(cfg)


@pytest.mark.parametrize('initialized',[False,True])
def test_checkpoint_captures_cuda_rng_only_if_cuda_already_initialized(tmp_path,monkeypatch,initialized):
    *_,model,optimizer,identity=build()
    calls=[]
    expected=[torch.tensor([3,1,4],dtype=torch.uint8)]
    def cuda_rng():
        calls.append('read-existing-state')
        return expected
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:initialized)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',cuda_rng)
    path=tmp_path/'rng.pt'
    api().save_action_set_checkpoint(path,model,optimizer,identity,config(),{},step=0)
    rng=torch.load(path,map_location='cpu',weights_only=False)['rng_state']
    if initialized:
        assert calls==['read-existing-state']
        assert len(rng['cuda'])==1 and torch.equal(rng['cuda'][0],expected[0])
    else:
        assert calls==[] and 'cuda' not in rng
