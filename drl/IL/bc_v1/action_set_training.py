"""Scratch ActionSet admission and portable checkpoints; no training loop.

F2/normalizer identities come from the authenticated supplied G2 manifest and
are pinned exactly in the checkpoint. RobotSpec, feature schema and layout are
also checked against the actual caller inputs. Checkpoints are trusted local
artifacts, deserialized on CPU before validation or device transfer.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from . import action_set
from .action_set_head import ActionSetPolicy, _validate_config as _validate_head
from .bc_v1_contract import robot_spec_fingerprint
from .feature_schema import feature_schema_fingerprint
from .model import _validate_config as _validate_h1
from .sidecar import VERSION as G2_VERSION, _hash, _identity, _catalog, _manifest_fingerprint
from .tensorization import load_tensor_layout
from .trainer import SelectionState
from .training import build_optimizer
from .training_data import TrainingDataset

CHECKPOINT_SCHEMA='mssr.bc_v1.checkpoint.action_set.v1'
TRAINING_SCHEMA='mssr.bc_v1.training_action_set.v1'
MODEL_CONTRACT='action_set_v1'
ROOT=Path(__file__).resolve().parents[3]


def _validate_config(config):
    expected={'schema_version','model_contract','initialization','seed','lr','weight_decay',
        'batch_size','max_epochs','patience','min_delta','threshold_logit'}
    if (set(config)!=expected or config['schema_version']!=TRAINING_SCHEMA
            or config['model_contract']!=MODEL_CONTRACT or config['initialization']!='scratch'
            or type(config['seed']) is not int or not 0<=config['seed']<2**32
            or type(config['batch_size']) is not int or config['batch_size']<1
            or any(type(config[k]) is not int or config[k]<1 for k in ('max_epochs','patience'))
            or any(type(config[k]) not in (float,int) or not math.isfinite(config[k])
                for k in ('lr','weight_decay','min_delta','threshold_logit'))
            or config['lr']<=0 or config['weight_decay']<0 or config['min_delta']<0
            ):
        raise ValueError('Unsupported ActionSet training configuration')


def load_action_set_training_config(path):
    config=json.loads(Path(path).read_text());_validate_config(config)
    return config


def selection_state(config):
    _validate_config(config)
    return SelectionState(patience=config['patience'],min_delta=config['min_delta'])


def _source(manifest):
    if (manifest.get('schema_version')!=G2_VERSION or manifest.get('complete') is not True
            or manifest.get('manifest_fingerprint')!=_manifest_fingerprint(manifest)):
        raise ValueError('Unsupported/incomplete or unauthenticated source G2 manifest')
    ids=_identity(manifest['identity'])
    inventory=_catalog(manifest['inventory'])
    if inventory['test'] or manifest.get('membership_fingerprint')!=_hash(inventory):
        raise ValueError('Source G2 must exclude TEST and authenticate membership')
    return ids


def _model_identity(spec,schema,h1,h2,source,source_fp,config):
    _validate_config(config);_validate_h1(h1);_validate_head(h2)
    source=_identity(source)
    robot=robot_spec_fingerprint(spec)
    feature=feature_schema_fingerprint(schema)
    layout=load_tensor_layout(ROOT/h1['tensor_layout'])['fingerprint']
    for key,value in (('robot_spec_fingerprint',robot),('feature_schema_fingerprint',feature),('layout_fingerprint',layout)):
        if source[key]!=value:
            raise ValueError('Source G2 mismatch: '+key)
    return {'model_contract':MODEL_CONTRACT,'action_set_version':action_set.VERSION,
        'action_set_fingerprint':hashlib.sha256(Path(action_set.__file__).read_bytes()).hexdigest(),
        'head_config':copy.deepcopy(h2),'head_config_fingerprint':_hash(h2),
        'h1_config_fingerprint':_hash(h1),'source_schema_version':G2_VERSION,
        'source_manifest_fingerprint':source_fp,'source_identity':source,
        'robot_spec_fingerprint':robot,'catalog_fingerprint':robot,
        'feature_schema_fingerprint':feature,'layout_fingerprint':layout,
        'training_config_fingerprint':_hash(config),
        'decode_config':{'threshold_logit':config['threshold_logit'],'configured_arguments':{}}}


def build_scratch_actor(spec,schema,h1,h2,manifest,config,*,device='cpu'):
    """Return freshly seeded (ActionSetPolicy, AdamW, immutable-by-copy identity)."""
    source=_source(manifest)
    identity=_model_identity(spec,schema,h1,h2,source,manifest['manifest_fingerprint'],config)
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    model=ActionSetPolicy(spec,schema,h1,h2).to(device)
    optimizer=build_optimizer(model,lr=config['lr'],weight_decay=config['weight_decay'])
    return model,optimizer,identity


def open_dataset(root,split,identity,**kwargs):
    """Reject TEST before constructing even a dataset or opening a file."""
    if split not in ('train','validation'):
        raise ValueError('ActionSet forbids TEST: only train/validation may be opened')
    return TrainingDataset(root,split,expected_identity=identity,**kwargs)


def _cpu(value):
    if isinstance(value,torch.Tensor):
        tensor=value.detach().cpu().clone()
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
            raise ValueError('Nonfinite checkpoint tensor')
        return tensor
    if isinstance(value,dict):return {k:_cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [_cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(_cpu(v) for v in value)
    if isinstance(value,float) and not math.isfinite(value):raise ValueError('Nonfinite checkpoint scalar')
    return copy.deepcopy(value)


def _check_identity(identity,spec,schema,h1,h2,config):
    try:
        current=_model_identity(spec,schema,h1,h2,identity['source_identity'],
            identity['source_manifest_fingerprint'],config)
    except (KeyError,TypeError) as exc:
        raise ValueError('Incomplete ActionSet checkpoint identity') from exc
    if identity!=current:
        raise ValueError('Incompatible ActionSet checkpoint identity/configuration')


def save_action_set_checkpoint(path,model,optimizer,identity,config,summary,*,step=0):
    """Atomically publish CPU state, optimizer moments, config, identity and RNG."""
    _check_identity(identity,model.head.spec,model.backbone.schema,model.backbone.config,model.head.config,config)
    if type(step) is not int or step<0:raise ValueError('Invalid checkpoint step')
    rng={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()}
    if torch.cuda.is_initialized():
        rng['cuda']=torch.cuda.get_rng_state_all()
    payload=_cpu({'schema_version':CHECKPOINT_SCHEMA,'model_contract':MODEL_CONTRACT,
        'model':model.state_dict(),'optimizer':optimizer.state_dict() if optimizer is not None else None,
        'identity':identity,'config':config,'step':step,'summary':summary,'rng_state':rng})
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='wb',dir=path.parent,suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name);torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def load_action_set_checkpoint(path,spec,schema,h1,h2,*,expected_identity,device='cpu'):
    """Return ActionSetPolicy after exact identity/schema/state validation on CPU.

    This loader does not resume optimization or restore the caller's RNG. The
    portable optimizer/RNG records remain available in the checkpoint payload.
    """
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if (not isinstance(payload,dict) or payload.get('schema_version')!=CHECKPOINT_SCHEMA
            or payload.get('model_contract')!=MODEL_CONTRACT):
        raise ValueError('Incompatible ActionSet checkpoint schema')
    if payload.get('identity')!=expected_identity:
        raise ValueError('Incompatible ActionSet checkpoint identity')
    _check_identity(expected_identity,spec,schema,h1,h2,payload['config'])
    _cpu(payload)  # Reject nonfinite state before loading or device transfer.
    model=ActionSetPolicy(spec,schema,h1,h2)
    expected=model.state_dict();actual=payload['model']
    if actual.keys()!=expected.keys():raise ValueError('Incompatible ActionSet state keys')
    if any(not isinstance(actual[k],torch.Tensor) or actual[k].dtype!=v.dtype
            or actual[k].shape!=v.shape for k,v in expected.items()):
        raise ValueError('Incompatible ActionSet state dtype/shape')
    model.load_state_dict(actual,strict=True)
    return model.to(device)
