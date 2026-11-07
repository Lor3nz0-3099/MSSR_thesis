"""Compare SAME H1-batched BC-v4 actor with serial and batched H2.
No Isaac startup, no training, no checkpoint modifications.
"""
import json
from pathlib import Path
from statistics import median
from time import perf_counter
import torch
from drl.IL.tests.bc_v1.g2_fixtures import fixture,tensorize
from drl.IL.bc_v1.action_set_head import ActionSetPolicy,load_action_set_head_config
from drl.IL.bc_v1.model import load_model_config
from drl.MARL.bc_v4_actor import BCV4SharedActorWrapper

ROOT=Path(__file__).resolve().parent
# Caller runs from repository root, not Home.
ROOT=Path.cwd()
assert (ROOT/'drl/MARL/bc_v4_actor.py').is_file(), 'Run from ~/MSSR_thesis'
assert torch.cuda.is_available(),'CUDA unavailable'
DEVICE=torch.device('cuda:0')
torch.manual_seed(3103)
_, kw=fixture(8,2,attached=False)
actor=BCV4SharedActorWrapper(policy=ActionSetPolicy(
    kw['spec'],kw['feature_schema'],
    load_model_config(ROOT/'configs/bc_v1/gnn_v1.json'),
    load_action_set_head_config(ROOT/'configs/bc_v1/action_set_head_v1.json')),
    device=DEVICE).eval()

def to_gpu(v):
    if torch.is_tensor(v):return v.to(DEVICE)
    if isinstance(v,dict):return {k:to_gpu(x) for k,x in v.items()}
    return v

sample=tensorize(fixture(8,2,attached=False)[0],kw)
sample_graph=to_gpu(sample['inputs'])
routing=sample['routing']

results=[]
for n in (8,16,32):
    graphs=[sample_graph]*n
    routings=[routing]*n
    def run(batched):
        actor.prefer_batched_h2=batched
        return actor.forward_all_owners_batch(
            graphs=graphs,routings=routings,
            possible_agents=['module_0'],lazy_payloads=True)

    with torch.inference_mode():
        serial=run(False)
        batch=run(True)
        for j,(a,b) in enumerate(zip(serial,batch)):
            x=a['prediction_template']['presence_logits'];y=b['prediction_template']['presence_logits']
            torch.testing.assert_close(x,y,atol=3e-5,rtol=1e-4)
            assert len(a['records'])==len(b['records']),f'env {j}'
        measured={}
        for batched in (False,True):
            samples=[]
            for _ in range(2):run(batched)
            for _ in range(7):
                torch.cuda.synchronize(DEVICE)
                start=perf_counter()
                run(batched)
                torch.cuda.synchronize(DEVICE)
                samples.append((perf_counter()-start)*1000)
            measured[batched]=median(samples)
    speed=measured[False]/measured[True]
    results.append({'envs':n,'full_actor_h2_serial_ms':round(measured[False],3),
                    'full_actor_h2_batch_ms':round(measured[True],3),
                    'speedup':round(speed,3)})
    print(f'{n} env | FULL ACTOR H2 serial {measured[False]:.1f}ms | '
          f'H2 batched {measured[True]:.1f}ms | {speed:.2f}x',flush=True)

out=ROOT/'logs'/'mssr_5b_h2_actor_profile.json'
if out.exists():raise SystemExit(f'Output exists, not overwriting: {out}')
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({'mode':'synthetic, lazy, H1/H2, no Isaac',
                          'results':results},indent=2)+'\n')
print('Saved:',out)
