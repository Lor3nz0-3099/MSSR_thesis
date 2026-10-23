"""Resource v3 scratch actor composition for existing CPU/GPU training plumbing.

No training loop/launch on import; no warm-start or legacy checkpoint rewriting.
The next milestone supplies its reviewed schedule after the manual G2-v3 build.
"""
import torch
from .model import BCGNN
from .resource_decisions import ResourceDecisionContract,VERSION
from .structural_decisions import VERSION as STRUCTURAL_VERSION
from .sidecar import _hash
from .training import build_optimizer


def build_scratch_actor(spec,feature_schema,h1,h2,manifest,config,*,device='cpu'):
    from .bc_v1_contract import robot_spec_fingerprint
    from .feature_schema import feature_schema_fingerprint
    if config.get('schema_version')!='mssr.bc_v1.training_resource.v3' or config.get('initialization')!='scratch':raise ValueError('Resource v3 baseline requires explicit scratch initialization')
    if h2.get('schema_version')!='mssr.bc_v1.action_heads.v3':raise ValueError('Resource v3 H2 required')
    contract=ResourceDecisionContract.from_dict(h2['resource_contract'],spec)
    if (manifest.get('resource_decision_schema')!=VERSION or manifest['identity'].get('resource_contract_fingerprint')!=contract.fingerprint
            or config.get('resource_contract_fingerprint')!=contract.fingerprint or manifest['inventory'].get('test')):
        raise ValueError('Resource v3 training identity/TEST boundary mismatch')
    for k,v in [('robot_spec_fingerprint',robot_spec_fingerprint(spec)),('feature_schema_fingerprint',feature_schema_fingerprint(feature_schema))]:
        if k in manifest['identity'] and manifest['identity'][k]!=v:raise ValueError('Physical training identity mismatch: '+k)
    torch.manual_seed(config['seed'])
    model=BCGNN(spec,feature_schema,h1,action_head_config=h2).to(device)
    identity={**manifest['identity'],'sidecar_manifest_fingerprint':manifest['manifest_fingerprint'],
        'resource_decision_schema':VERSION,'resource_contract_fingerprint':contract.fingerprint,
        'structural_decision_schema':STRUCTURAL_VERSION,'h1_config_fingerprint':_hash(h1),
        'h2_config_fingerprint':_hash(h2),'training_config_fingerprint':_hash(config)}
    return model,build_optimizer(model,lr=config['learning_rate'],weight_decay=config['weight_decay']),identity
