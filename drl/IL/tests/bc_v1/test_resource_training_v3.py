import importlib.util
import torch
import pytest
from drl.IL.tests.bc_v1.test_resource_heads_v3 import example,threads
from drl.IL.tests.bc_v1.g2_fixtures import ROOT


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.resource_training'),'Resource v3 training composition missing'
    from drl.IL.bc_v1 import resource_training
    return resource_training


def test_v3_scratch_train_step_and_checkpoint_roundtrip_cpu(tmp_path):
    from drl.IL.bc_v1.training import train_step,save_checkpoint,load_checkpoint
    from drl.IL.bc_v1.losses import load_loss_config
    from drl.IL.bc_v1.sidecar import _hash
    b,model,kw,_=example('inactive');a=api()
    head=model.action_heads.config;c=model.action_heads.resource_contract
    cfg={'schema_version':'mssr.bc_v1.training_resource.v3','initialization':'scratch','seed':41,
        'resource_contract_fingerprint':c.fingerprint,'learning_rate':.0003,'weight_decay':.0001}
    manifest={'resource_decision_schema':'mssr.bc_v1.resource_decisions.v1','manifest_fingerprint':'tiny',
        'identity':{'resource_contract_fingerprint':c.fingerprint},'inventory':{'test':[]}}
    m,opt,identity=a.build_scratch_actor(kw['spec'],kw['feature_schema'],model.config,head,manifest,cfg,device='cpu')
    assert identity['resource_contract_fingerprint']==c.fingerprint
    assert identity['training_config_fingerprint']==_hash(cfg)
    result=train_step(m,b,opt,load_loss_config(ROOT/'configs/bc_v1/loss_v1.json'),audit=True)
    assert result['optimizer_step_completed'] and result['gradients']['resource_mode']['nonzero']
    path=tmp_path/'checkpoint.pt'
    save_checkpoint(path,m,opt,step=1,identity=identity,config=cfg,selection={},seed=41,summary={'epoch':1})
    restored,optimizer,expected=a.build_scratch_actor(kw['spec'],kw['feature_schema'],model.config,head,manifest,cfg,device='cpu')
    load_checkpoint(path,restored,optimizer,expected_identity=expected)
    assert all(torch.equal(value,restored.state_dict()[key]) for key,value in m.state_dict().items())
    with pytest.raises(ValueError):a.build_scratch_actor(kw['spec'],kw['feature_schema'],model.config,head,manifest,{**cfg,'initialization':'warm_start'},device='cpu')
