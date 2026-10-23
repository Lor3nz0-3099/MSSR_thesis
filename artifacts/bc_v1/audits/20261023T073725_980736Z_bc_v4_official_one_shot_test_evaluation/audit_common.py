"""Audit-local frozen evaluation composition. No source/TEST IO on import."""
from pathlib import Path
import hashlib
import json
import datetime
import sys

A = Path(__file__).resolve().parent
R = Path('/home/lorenzo/MSSR_thesis')
sys.path.insert(0, str(R))
RUN = R/'artifacts/bc_v1/runs/20261022T232532_854802073Z_207309_bc_v4_actor_visible_r1m_scratch_final'
G2 = R/'artifacts/bc_v1/sidecar/g2_v4_actor_visible_r1m'
G1 = R/'artifacts/bc_v1/cache/g1_v1_optimized'
F2 = R/'artifacts/bc_v1/normalization/f2_actor_visible_r1m_v1'
BEST = RUN/'best.pt'

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def load(path):
    return json.loads(Path(path).read_text())

def save(name, value):
    p=A/name; temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temp.replace(p)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def model_hash(model):
    h=hashlib.sha256()
    for key,t in sorted(model.state_dict().items()):
        t=t.detach().cpu().contiguous()
        h.update(json.dumps([key,str(t.dtype),list(t.shape)],separators=(',',':')).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()

def runtime():
    import torch
    from drl.IL.bc_v1 import train_action_set as runner, dataset as g1
    from drl.IL.bc_v1.actor_prepare import FrozenD1GoalProvider
    from drl.IL.bc_v1.actor_visibility import load_visibility_policy
    from drl.IL.bc_v1.resource_decisions import load_resource_contract
    from drl.IL.bc_v1.normalization import load_normalization_policy
    from drl.IL.bc_v1.tensorization import load_tensor_layout
    from drl.IL.bc_v1.action_set_training import load_action_set_checkpoint
    cfg=load(RUN/'config.json')
    assert cfg['initialization']=='scratch' and cfg['threshold_logit']==0.0 and cfg['batch_size']==64
    report,kw,h1,h2,manifest=runner.preflight(G2,cfg,normalizer_dir=F2)
    identity=load(RUN/'identity.json')
    assert identity==report['identity']
    assert identity['source_manifest_fingerprint']=='d80b0774ed1793dedfa101cbaa82e45b7cb460384bfdd36a4291a592e9d97a3f'
    assert identity['source_identity']['f2_fingerprint']=='bdb5304cd620faa7461c448794044af4757b85e77e672ca7022a1083da9f37cd'
    visibility=load_visibility_policy(R/'configs/bc_v1/actor_visibility_r1m.json')
    assert visibility.fingerprint=='e6b819c2e23e1713f70649d83b547d7701d78ef8c0ec7a21668c803c7c8f6234'
    assert visibility.document['radius_m']==1.0 and visibility.document['scope']=='team_union_wireless'
    physical=g1.read_cache_manifest(G1)
    split=load(R/'artifacts/bc_v1/splits/f1_v1/split.json')
    assert physical['cache_fingerprint']==identity['source_identity']['g1_fingerprint']
    assert physical['splits']==split['splits'] and split['manifest_fingerprint']==identity['source_identity']['split_fingerprint']
    assert physical['counts']['splits']['test']=={'samples':18536,'episodes':12}
    assert len(split['splits']['test'])==12
    assert physical['identity']['source_manifest_sha256']==identity['source_identity']['canonical_sha']
    assert sha(R/'artifacts/bc_v1/splits/f1_v1/split.json')==identity['source_identity']['f1_sha']
    assert load(RUN/'status.json')['phase']=='COMPLETE'
    payload=torch.load(BEST,map_location='cpu',weights_only=False)
    history=[json.loads(x) for x in (RUN/'history.jsonl').read_text().splitlines()]
    row=next(x for x in history if x['epoch']==5)
    assert payload['summary']==row
    assert payload['summary']['epoch']==5 and payload['step']==6120
    assert payload['summary']['validation']['total_loss']==0.06868529843473761
    assert payload['summary']['initialization']==report['initialization']
    model=load_action_set_checkpoint(BEST,kw['spec'],kw['feature_schema'],h1,h2,expected_identity=identity,device='cpu')
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    assert all(torch.isfinite(t).all() for t in model.state_dict().values())
    contract=load_resource_contract(R/'configs/bc_v1/resource_decisions_v1.json',kw['spec'])
    assert contract.fingerprint==identity['source_identity']['resource_contract_fingerprint']
    goals=FrozenD1GoalProvider(physical,R/'configs/bc_v1/task_context.json')
    assert kw['normalizers'].metadata['goal_provider_identity']==goals.identity
    return dict(model=model,cfg=cfg,report=report,kw=kw,h1=h1,h2=h2,manifest=manifest,
        identity=identity,physical=physical,split=split,history=history,summary=row,
        visibility=visibility,contract=contract,goals=goals,
        policy=load_normalization_policy(R/'configs/bc_v1/normalization.json',kw['feature_schema']),
        layout=load_tensor_layout(R/'configs/bc_v1/tensor_layout.v1.json'))

def tensorize(sample, rt):
    from drl.IL.bc_v1.actor_prepare import tensorize_actor_sample
    from drl.IL.bc_v1.resource_decisions import add_resource_decisions
    from drl.IL.bc_v1.structural_decisions import add_no_structural_supervision
    t=tensorize_actor_sample(sample,spec=rt['kw']['spec'],feature_schema=rt['kw']['feature_schema'],
        normalization_policy=rt['policy'],normalizers=rt['kw']['normalizers'],tensor_layout=rt['layout'],
        visibility=rt['visibility'],assigned_goal=rt['goals'](sample))
    # Existing v3 ABI only; v4 target_view discards dense absence decisions.
    # No authority inferred from absence. Candidate universe/order stay current-G1.
    return add_resource_decisions(add_no_structural_supervision(t),rt['contract'])
