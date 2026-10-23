"""Explicit manual v3 FP32 run. Default is metadata-only preflight.

No TEST, preparation, resume, HPO, AMP or compile path. A new output directory
is mandatory for --run. One epoch is the default manual performance gate.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import resource
import time

import numpy as np
import torch

from .sidecar import _read_manifest
from .sidecar_build import ROOT, load_sources
from .resource_training import build_scratch_actor
from .training_data import TrainingDataset
from .sampling import JointActionBatchSampler
from .losses import load_loss_config
from .losses_bulk import compute_bc_loss_bulk
from .metrics import BCMetrics, load_metrics_config
from .trainer import train_epoch, evaluate_validation, SelectionState, check_supports, support_snapshot
from .training import save_checkpoint

G2=ROOT/'artifacts/bc_v1/sidecar/g2_v3_resource_contract'
PIN='4874065f8c7adc7894224d07692d21b289def12562543706cab0f420366bf3be'
CONTRACT='7a77f9118c81a7ca7a9b80f8f1df78011c92492be9c53a17d3fc00188268d194'


def read(path):return json.loads(Path(path).read_text())


def preflight():
    sources=load_sources()
    doc=read(G2/'manifest.json');doc=_read_manifest(G2,doc['identity'])
    if (doc['manifest_fingerprint']!=PIN or doc.get('resource_contract_fingerprint')!=CONTRACT
            or doc['inventory'].get('test') or
            {k:doc['split_counts'][k]['samples'] for k in ('train','validation')}!={'train':78302,'validation':22762}):
        raise ValueError('Frozen G2-v3 identity/counts/TEST boundary mismatch')
    kw=sources['kwargs']
    from .normalization import normalizer_fingerprint
    from .bc_v1_contract import robot_spec_fingerprint
    from .feature_schema import feature_schema_fingerprint
    expected={'f2_fingerprint':normalizer_fingerprint(kw['normalizers']),
        'robot_spec_fingerprint':robot_spec_fingerprint(kw['spec']),
        'feature_schema_fingerprint':feature_schema_fingerprint(kw['feature_schema']),
        'layout_fingerprint':kw['tensor_layout']['fingerprint']}
    if any(doc['identity'][k]!=v for k,v in expected.items()):raise ValueError('Frozen preprocessing identity mismatch')
    return {'g2':str(G2),'manifest_fingerprint':PIN,'resource_contract_fingerprint':CONTRACT,
        'samples':{'train':78302,'validation':22762},'preprocessing':expected,'TEST_payload_access':0,
        'numeric_mode':'FP32 eager','initialization':'scratch','dropout':.1,
        'batch_size':64,'general_quota':48,'structural_quota':16,
        'learning_rate':3e-4,'weight_decay':1e-4,'seed':20261006,
        'cache_budgets_bytes':{'train_dynamic':2*1024**3,'train_template':4*1024**3,
            'validation_dynamic':256*1024**2,'validation_template':512*1024**2},
        'prefetch':False,'normalization':'frozen F2; fold views are separately opt-in'}


def _write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as f:
        json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def run(output, *, epochs, cpu_threads, report):
    if output.exists():raise ValueError('Output exists; no overwrite/resume')
    # Reserve room for Python/SQLite/host batches, CUDA context and other apps.
    available=next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    if available<sum(report['cache_budgets_bytes'].values())+4*1024**3:
        raise ValueError('Insufficient available RAM for bounded cache plus working headroom')
    if not torch.cuda.is_available():raise ValueError('Manual run requires CUDA; no CPU fallback')
    output.mkdir(parents=True,exist_ok=False)
    _write(output/'preflight.json',report)
    cfg={**report,'schema_version':'mssr.bc_v1.training_resource.v3','initialization':'scratch',
        'cpu_threads':cpu_threads,'epochs':epochs,'gradient_clip':1.,'patience':8,'min_delta':1e-4}
    _write(output/'config.json',cfg)
    torch.set_num_threads(cpu_threads);torch.set_num_interop_threads(cpu_threads)
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cuda:0';sources=load_sources();kw=sources['kwargs'];doc=read(G2/'manifest.json')
    model,optimizer,identity=build_scratch_actor(kw['spec'],kw['feature_schema'],
        read(ROOT/'configs/bc_v1/gnn_v1.json'),read(ROOT/'configs/bc_v1/action_heads_resource_v3.json'),doc,cfg,device=device)
    loss_cfg=load_loss_config(ROOT/'configs/bc_v1/loss_v1.json')
    metric_cfg=load_metrics_config(ROOT/'configs/bc_v1/metrics_v1.json')
    metrics=BCMetrics(kw['spec'],kw['normalizers'],metric_cfg)
    selection=SelectionState(patience=8,min_delta=1e-4,last_epoch=0)
    global_step=0;expected_supports=None;start=time.perf_counter()
    budgets=report['cache_budgets_bytes']
    try:
        with TrainingDataset(G2,'train',expected_identity=doc['identity'],
                dynamic_budget_bytes=budgets['train_dynamic'],template_budget_bytes=budgets['train_template']) as train, \
             TrainingDataset(G2,'validation',expected_identity=doc['identity'],
                dynamic_budget_bytes=budgets['validation_dynamic'],template_budget_bytes=budgets['validation_template']) as val, \
             (output/'history.jsonl').open('x') as history:
            sampler=JointActionBatchSampler(train.train_index(),batch_size=64,ordinary_quota=48,structural_quota=16,seed=cfg['seed'])
            for epoch in range(1,epochs+1):
                preflight()  # Metadata/fingerprint boundary, no TEST payload.
                _write(output/'status.json',{'phase':'TRAIN','epoch':epoch,'global_step':global_step})
                training=train_epoch(model,train,sampler,optimizer,loss_cfg,device=device,epoch=epoch,
                    global_step=global_step,gradient_clip=1.,loss_fn=compute_bc_loss_bulk)
                if training['steps']!=len(sampler):raise ValueError('Incomplete TRAIN epoch')
                global_step=training['global_step']
                _write(output/'status.json',{'phase':'VAL','epoch':epoch,'global_step':global_step})
                validation=evaluate_validation(model,val,loss_cfg,metrics,device=device,batch_size=64,
                    epoch=epoch,loss_fn=compute_bc_loss_bulk)
                if validation['samples']!=22762:raise ValueError('Incomplete VAL coverage')
                if expected_supports is None:expected_supports=support_snapshot(validation)
                else:check_supports(expected_supports,validation)
                if epoch==1:selection.initial_val=validation['total_loss']
                improved=selection.observe(epoch,validation['total_loss'])
                summary={'epoch':epoch,'global_step':global_step,'best_epoch':selection.best_epoch,'best_val_loss':selection.best_val}
                row={**summary,'train':training,'validation':validation,'cache':{'train':train.stats(),'validation':val.stats()},
                    'ram_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    'vram_peak_bytes':torch.cuda.max_memory_allocated(),'TEST_payload_access':0}
                history.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');history.flush();os.fsync(history.fileno())
                for name in (['best_checkpoint.pt','last_checkpoint.pt'] if improved else ['last_checkpoint.pt']):
                    save_checkpoint(output/name,model,optimizer,step=global_step,identity=identity,config=cfg,
                        selection=asdict(selection),seed=cfg['seed'],summary=summary)
                _write(output/'sampler_state.json',sampler.state_dict())
                if selection.should_stop:break
            _write(output/'SUCCESS.json',{'epochs':epoch,'global_step':global_step,'wall_s':time.perf_counter()-start,
                'selection':asdict(selection),'TEST_payload_access':0,'identity':identity})
            _write(output/'status.json',{'phase':'COMPLETED','epoch':epoch,'global_step':global_step})
    except BaseException as exc:
        _write(output/'status.json',{'phase':'FAILED','error':repr(exc),'global_step':global_step})
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true');parser.add_argument('--output',type=Path)
    parser.add_argument('--epochs',type=int,default=1);parser.add_argument('--cpu-threads',type=int,default=1)
    args=parser.parse_args(argv)
    if not 1<=args.epochs<=100 or not 1<=args.cpu_threads<=16:parser.error('Bounded epoch/thread values required')
    if args.run and args.output is None:parser.error('--run requires a new --output directory')
    report=preflight()
    if not args.run:
        print(json.dumps({'mode':'PREFLIGHT_ONLY',**report},indent=2,sort_keys=True));return
    if args.output.exists():raise ValueError('Output exists; no overwrite')
    run(args.output,epochs=args.epochs,cpu_threads=args.cpu_threads,report=report)


if __name__=='__main__':main()
