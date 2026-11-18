"""Explicit BC-v5 scratch runner: authenticated binary labels + primitive sampler.

BC-v4 runner, G2, H1/H2, Canonical, F1/F2 and TEST remain untouched.
A complete authenticated Canonical/G2 overlay is mandatory; a separate temporal
audit is optional. Canonical conflict masks and R16 exclusions remain enforced.
Default = DRY_RUN. No checkpoint is ever initialized from BC-v4.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
# BC_V5_V8_MONITOR_PATCH_V1

import torch

from . import train_action_set as base
from .action_set_training import build_scratch_actor, open_dataset, selection_state
from .bc_v5_binary import (index_overlay, collate_binary_v5, compute_binary_v5_loss,
    binary_presence_metrics, aggregate_binary_metrics)
from .bc_v5_integrity import read_report
from .bc_v5_sampling import build_primitive_train_index, BalancedPrimitiveBatchSampler
from .bc_v5_v8_sampling import AdaptivePrimitiveBatchSampler
# BC_V5_V8_PRIMITIVE_FIRST_SAMPLER_V2
from .bc_v5_v8_readiness import load_readiness
from .sidecar import _read_manifest
from .trainer import FamilyAccumulator, ensure_finite_parameters, validation_batches
from .training import family_statistics, move_batch_to_device

SCHEMA='mssr.bc_v5.checkpoint.action_set.v1'


class FilteredTrainingDataset:
    """Safe TRAIN/VAL subset, indexed from G2 SQLite *metadata* not TEST tensors."""
    def __init__(self, underlying, rows, drop_keys=()):
        drop_keys=set(drop_keys)
        if underlying.split not in ('train','validation'):
            raise ValueError('BC-v5 cannot inspect TEST')
        self.underlying = underlying
        self.split, self.manifest = underlying.split, underlying.manifest
        self.episode_ids = underlying._data.episode_ids
        indexed = []
        hit = set()
        for idx,eid,transition in underlying._data.db.execute(
                'SELECT idx,episode,transition_idx FROM samples WHERE split=? ORDER BY idx',
                (underlying.split,)):
            key=eid,transition
            if key not in rows or idx in hit:
                raise ValueError('Incomplete BC-v5 subset membership or duplicate index')
            hit.add(idx)
            if not rows[key]['exclude'] and key not in drop_keys:
                indexed.append(idx)
        if len(hit) != len(underlying) or len(rows) != len(hit):
            raise ValueError('Overlay does not cover the frozen source')
        self.positions = tuple(indexed)
        if not self.positions:
            raise ValueError('No safe BC-v5 samples in '+self.split)

    def __len__(self):return len(self.positions)
    def get_for_collate(self,idx):return self.underlying.get_for_collate(self.positions[idx])
    def read_parts(self,idx):return self.underlying._data.read_parts(self.positions[idx])


def _step(model,batch,optimizer):
    model.train();optimizer.zero_grad(set_to_none=True)
    prediction=model(batch['inputs'])
    loss,stats=compute_binary_v5_loss(prediction,batch)
    if not torch.isfinite(loss):raise ValueError('Nonfinite BC-v5 objective')
    loss.backward()
    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step();ensure_finite_parameters(model)
    return {'loss':float(loss.detach()),'grad_norm':float(grad),'families':family_statistics(stats)}


@torch.no_grad()
def evaluate(model, data, overlay, device, *, batch_size, threshold, progress=None):
    model.eval()
    accumulator=FamilyAccumulator({k:1. for k in base.FAMILIES})
    metric_rows=[]
    total_batches=math.ceil(len(data)/batch_size)
    for batch_number,indices in enumerate(validation_batches(len(data),batch_size),start=1):
        batch=move_batch_to_device(collate_binary_v5(
            [data.get_for_collate(i) for i in indices],overlay),device)
        pred=model(batch['inputs'])
        loss,stats=compute_binary_v5_loss(pred,batch)
        if not torch.isfinite(loss):raise ValueError('Nonfinite BC-v5 validation')
        accumulator.add(family_statistics(stats))
        metric_rows.append(binary_presence_metrics(pred,batch,threshold=threshold))
        if progress is not None:progress(batch_number,total_batches)
    return {**accumulator.compute(), 'binary_membership':aggregate_binary_metrics(metric_rows)}


def _checkpoint(path,model,optimizer,identity,config,summary,step):
    """Distinct BC-v5 checkpoint schema; overlay/sampler fingerprint is mandatory."""
    if identity.get('schema_version')!='mssr.bc_v5.training_identity.v1' or not identity.get('audit_fingerprint'):
        raise ValueError('Missing BC-v5 experiment identity')
    payload={'schema_version':SCHEMA,'model_contract':'action_set_v1',
             'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
             'optimizer':optimizer.state_dict(),'identity':identity,
             'config':config,'step':step,'summary':summary}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='wb',dir=path.parent,suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name)
            torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)



def load_bc_v5_checkpoint(path, spec, schema, h1, h2, *, expected_identity, device='cpu'):
    """Load a BC-v5 checkpoint only under its exact audited training identity.

    The model architecture is the original ActionSetPolicy, but the checkpoint
    format is deliberately incompatible with the legacy BC-v4 loader.
    """
    from .action_set_head import ActionSetPolicy
    from .action_set_training import _model_identity
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if (not isinstance(payload,dict) or payload.get('schema_version')!=SCHEMA
            or payload.get('model_contract')!='action_set_v1'
            or payload.get('identity')!=expected_identity):
        raise ValueError('BC-v5 checkpoint schema/identity mismatch')
    source_identity=expected_identity.get('model_source')
    config=payload['config']
    expected=_model_identity(spec,schema,h1,h2,source_identity['source_identity'],
                             source_identity['source_manifest_fingerprint'],config)
    if source_identity!=expected:
        raise ValueError('BC-v5 source/model architecture mismatch')
    model=ActionSetPolicy(spec,schema,h1,h2)
    expected_state=model.state_dict()
    actual=payload['model']
    if (not isinstance(actual,dict) or actual.keys()!=expected_state.keys()
            or any(not isinstance(actual[k],torch.Tensor)
                   or actual[k].shape!=v.shape or actual[k].dtype!=v.dtype
                   or not torch.isfinite(actual[k]).all()
                   for k,v in expected_state.items())):
        raise ValueError('Invalid BC-v5 checkpoint weights')
    model.load_state_dict(actual,strict=True)
    return model.to(device)


def _write(path,doc):
    path.write_text(json.dumps(doc,sort_keys=True,indent=2,allow_nan=False)+'\n')


def _monitor_status(output, status):
    path=Path(output)/"status.json"
    temporary=path.with_name("status.monitor.tmp")
    temporary.write_text(json.dumps(status,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(temporary,path)

def run(g2,normalizers,overlay_path,output,config,*,run_training=False,
        balanced_quota=16,frequency_exponent=.5,readiness_path=None,progress_every=50):
    if type(progress_every) is not int or progress_every<1:
        raise ValueError("progress_every must be positive")
    if (type(balanced_quota) is not int or not 0 <= balanced_quota <= config['batch_size']
            or type(frequency_exponent) not in (int,float)
            or not math.isfinite(frequency_exponent) or not 0<=frequency_exponent<=1):
        raise ValueError('Invalid BC-v5 sampling parameters')
    # Authenticate the same G2 actor-visible model/source contract as BC-v4.
    prepared=base.preflight(g2,config,normalizer_dir=normalizers)
    preflight,kwargs,h1,h2,manifest=prepared
    # Stage11 G2 records a cryptographic provenance link to its immutable
    # Stage5 source overlay. Never relabel/rebuild that 106279-row audit.
    topology=manifest.get('bc_v5_topology')
    if topology is not None:
        if readiness_path is None:
            raise ValueError('Stage11 training requires --ready-v8 certificate')
        ready,ready_fingerprint=load_readiness(readiness_path,manifest,overlay_path)
        overlay_source=topology['source_manifest_fingerprint']
        zero_keys={s:{(row['episode_id'],row['transition_index'])
                      for row in ready['zero_positive_graphs'][s]}
                   for s in ('train','validation')}
    else:
        ready_fingerprint=None
        overlay_source=manifest['manifest_fingerprint']
        zero_keys={'train':set(),'validation':set()}
    report=read_report(overlay_path,expected_g2_fingerprint=overlay_source)
    train_overlay=index_overlay(report,split='train')
    val_overlay=index_overlay(report,split='validation')
    output=base._output_path(output,g2)
    if output.resolve()==Path(overlay_path).resolve() or output.resolve().is_relative_to(Path(overlay_path).resolve()):
        raise ValueError('Output conflicts with BC-v5 audit')
    plan={'schema_version':'mssr.bc_v5.training_plan.v1',
          'mode':'RUN_REQUESTED' if run_training else 'DRY_RUN',
          'source_g2':str(Path(g2).resolve()),
          'manifest_fingerprint':manifest['manifest_fingerprint'],
          'audit_fingerprint':json.loads(Path(overlay_path).read_text())['fingerprint'],
          'stage678_readiness_fingerprint':ready_fingerprint,
          'label_contract':'binary_presence + explicit partial-resource validity mask',
          'temporal_exclusion':('Canonical R16/unresolved sequence/unlocalizable conflict'
              + (' plus external temporal classifications'
                 if report['temporal_audit_provided'] else ' (no external classification)')),
          'test_payload_access':0,'scratch':True,
          'sampling':{'balanced_quota':balanced_quota,
                      'frequency_exponent':frequency_exponent},
          'samples':{s:sum(1 for k,x in rows.items() if not x['exclude'] and k not in zero_keys[s])
                     for s,rows in (('train',train_overlay),('validation',val_overlay))},
          'zero_positive_excluded':{s:len(zero_keys[s]) for s in ('train','validation')},
          'excluded':{s:sum(1 for x in rows.values() if x['exclude'])
                      for s,rows in (('train',train_overlay),('validation',val_overlay))}}
    if not run_training:
        return plan
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    model,optimizer,v4_identity=build_scratch_actor(kwargs['spec'],kwargs['feature_schema'],
        h1,h2,manifest,config,device=device)
    identity={'schema_version':'mssr.bc_v5.training_identity.v1',
              'audit_fingerprint':plan['audit_fingerprint'],
              'sample_counts':plan['samples'],'excluded_counts':plan['excluded'],
              'sampling':plan['sampling'],'model_source':v4_identity,
              'stage678_readiness_fingerprint':ready_fingerprint}
    state=selection_state(config)
    if state.patience!=8 or config['patience']!=8:
        raise ValueError('BC-v5 v8 production requires patience=8')
    step=0
    run_started=time.monotonic()
    print(f'[MONITOR] v8 | batch_size={config["batch_size"]} '
          f'max_epochs={config["max_epochs"]} patience={state.patience} '
          f'min_delta={config["min_delta"]} lr={config["lr"]} '
          f'balanced_quota={balanced_quota} progress_every={progress_every}',flush=True)
    with open_dataset(g2,'train',manifest['identity'],
            dynamic_budget_bytes=base.CACHE_BUDGETS['train_dynamic'],
            template_budget_bytes=base.CACHE_BUDGETS['train_template']) as tr, \
         open_dataset(g2,'validation',manifest['identity'],
            dynamic_budget_bytes=base.CACHE_BUDGETS['validation_dynamic'],
            template_budget_bytes=base.CACHE_BUDGETS['validation_template']) as va:
        train=FilteredTrainingDataset(tr,train_overlay,zero_keys['train'])
        validation=FilteredTrainingDataset(va,val_overlay,zero_keys['validation'])
        if (len(train)!=plan['samples']['train'] or len(validation)!=plan['samples']['validation']):
            raise ValueError('BC-v5 safe subset count mismatch')
        if topology is not None:
            # No full TRAIN tensor/route scan: metadata-only SQL plus a small
            # exact-positive bootstrap. Pools grow as minibatches are loaded.
            # Reuse Stage11 TRAIN positive census, no G2/Canonical rescan.
            report_path=Path('artifacts/bc_v1/audits/bc_v5_v8/stage11_full_report.json')
            stage11=json.loads(report_path.read_text())
            if (stage11.get('new_g2') != str(Path(g2).resolve())
                    or stage11.get('source_rows') != {'train':82528,'validation':23751}
                    or stage11.get('test_payload_access') != 0):
                raise ValueError('Stage11 TRAIN primitive prior/source mismatch')
            prior={name:int(stats['positive'])
                   for name,stats in stage11['new_candidate_counts']['train'].items()
                   if int(stats.get('positive',0))>0}
            expected={'resource:wheel_velocity','resource:pan_velocity','resource:tilt_target',
                      'resource:pan_target','structural:align_faces','structural:dock',
                      'structural:undock','structural:gravity_settle'}
            if set(prior)!=expected:
                raise ValueError('Unexpected Stage11 primitive prior')
            sampler=AdaptivePrimitiveBatchSampler(train,batch_size=config['batch_size'],
                balanced_quota=balanced_quota,frequency_exponent=frequency_exponent,
                seed=config['seed'],primitive_prior=prior)
            print('[SAMPLER] 48 uniform + 16 primitive-first balanced; '
                  'rare priority: 1 of 16, exact TRAIN positive pools only', flush=True)
            print('[SAMPLER] discovered='+', '.join(sampler.summary()['known_primitives'])+
                  ' | bootstrap_rows='+str(sampler.bootstrap_sample_count),flush=True)
            identity['train_sampler_index_fingerprint']=sampler.bootstrap_signature
            plan['v8_sampler_bootstrap']=sampler.summary()
        else:
            index=build_primitive_train_index(train)
            sampler=BalancedPrimitiveBatchSampler(index,batch_size=config['batch_size'],
                balanced_quota=balanced_quota,frequency_exponent=frequency_exponent,
                seed=config['seed'])
            identity['train_sampler_index_fingerprint']=index.fingerprint
        output.mkdir(parents=True,exist_ok=False)
        _write(output/'plan.json',plan)
        _write(output/'identity.json',identity)
        n_train_batches=len(sampler)
        n_val_batches=math.ceil(len(validation)/config['batch_size'])
        with (output/'history.jsonl').open('x') as history:
            for epoch in range(config['max_epochs']+1):
                epoch_started=time.monotonic()
                accumulator=FamilyAccumulator({k:1. for k in base.FAMILIES})
                monitor={'phase':'TRAIN' if epoch else 'VAL','epoch':epoch,
                         'max_epochs':config['max_epochs'],'global_step':step,
                         'train_batches_total':n_train_batches,'validation_batches_total':n_val_batches,
                         'patience':state.patience,'bad_epochs':state.counter,
                         'best_epoch':state.best_epoch,'best_validation_loss':None if not math.isfinite(state.best_val) else state.best_val,
                         'test_payload_access':0}
                _monitor_status(output,monitor)
                if epoch:
                    loss_sum=0.0
                    for batch_number,indices in enumerate(sampler,start=1):
                        samples=[train.get_for_collate(i) for i in indices]
                        if topology is not None:
                            sampler.observe_batch(indices,samples)
                        batch=move_batch_to_device(collate_binary_v5(
                            samples,train_overlay),device)
                        result=_step(model,batch,optimizer)
                        accumulator.add(result['families']);step+=1
                        loss_sum+=result['loss']
                        if batch_number%progress_every==0 or batch_number==n_train_batches:
                            seconds=time.monotonic()-epoch_started
                            eta=seconds*max(0,n_train_batches-batch_number)/batch_number
                            monitor.update(phase='TRAIN',global_step=step,
                                train_batches_done=batch_number,train_batches_total=n_train_batches,
                                train_batch_loss_last=round(result['loss'],6),
                                train_batch_loss_mean=round(loss_sum/batch_number,6),
                                elapsed_epoch_seconds=round(seconds,1),eta_train_seconds=round(eta,1))
                            _monitor_status(output,monitor)
                            print(f'[TRAIN] epoch={epoch}/{config["max_epochs"]} '
                                  f'batch={batch_number}/{n_train_batches} '
                                  f'loss_batch={result["loss"]:.5f} '
                                  f'loss_mean={loss_sum/batch_number:.5f} '
                                  f'elapsed={seconds:.0f}s eta_train~{eta:.0f}s',flush=True)
                val_started=time.monotonic()
                monitor.update(phase='VAL',validation_batches_done=0,
                    validation_batches_total=n_val_batches,global_step=step)
                _monitor_status(output,monitor)
                print(f'[VAL] epoch={epoch}/{config["max_epochs"]} '
                      f'start {len(validation)} rows / {n_val_batches} batches',flush=True)
                def report_val(batch_number,total_batches):
                    if batch_number%progress_every!=0 and batch_number!=total_batches:return
                    sec=time.monotonic()-val_started
                    eta=sec*max(0,total_batches-batch_number)/batch_number
                    monitor.update(phase='VAL',validation_batches_done=batch_number,
                        validation_batches_total=total_batches,
                        elapsed_validation_seconds=round(sec,1),eta_validation_seconds=round(eta,1))
                    _monitor_status(output,monitor)
                    print(f'[VAL] epoch={epoch}/{config["max_epochs"]} '
                          f'batch={batch_number}/{total_batches} '
                          f'elapsed={sec:.0f}s eta_val~{eta:.0f}s',flush=True)
                val=evaluate(model,validation,val_overlay,device,
                    batch_size=config['batch_size'],threshold=config['threshold_logit'],
                    progress=report_val)
                improved=state.observe(epoch,val['total_loss'])
                row={'epoch':epoch,'global_step':step,'validation':val,
                     'train':accumulator.compute() if epoch else None,
                     'selection':{'best_epoch':state.best_epoch,'best_value':state.best_val,
                                  'bad_epochs':state.counter},'test_payload_access':0,
                     'audit_fingerprint':identity['audit_fingerprint']}
                history.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');history.flush()
                _checkpoint(output/'last.pt',model,optimizer,identity,config,row,step)
                if improved:_checkpoint(output/'best.pt',model,optimizer,identity,config,row,step)
                if epoch and topology is not None:
                    primitive_draws=dict(sorted(sampler.last_balanced.items()))
                    monitor.update(balanced_primitive_draws=primitive_draws,
                                   sampler_uniform_slots=sampler.last_uniform,
                                   sampler_balanced_slots=sampler.last_balanced_total)
                    print('[SAMPLER] epoch='+str(epoch)+' balanced_draws='+
                          json.dumps(primitive_draws,sort_keys=True),flush=True)
                monitor.update(phase='EPOCH_END',epoch=epoch,global_step=step,
                    train_loss_mean=(round(loss_sum/n_train_batches,7) if epoch else None),
                    validation_f1=round(val['binary_membership']['f1'],7),
                    validation_precision=round(val['binary_membership']['precision'],7),
                    validation_recall=round(val['binary_membership']['recall'],7),
                    validation_loss=round(val['total_loss'],7),
                    best_validation_loss=round(state.best_val,7),
                    best_epoch=state.best_epoch,bad_epochs=state.counter,
                    epoch_seconds=round(time.monotonic()-epoch_started,1),
                    total_elapsed_seconds=round(time.monotonic()-run_started,1),
                    stopped_early=state.should_stop)
                _monitor_status(output,monitor)
                print(f'[EPOCH] {epoch}/{config["max_epochs"]} '
                      f'train_loss={(loss_sum/n_train_batches if epoch else float("nan")):.5f} '
                      f'val_loss={val["total_loss"]:.5f} '
                      f'F1={val["binary_membership"]["f1"]:.3f} '
                      f'precision={val["binary_membership"]["precision"]:.3f} '
                      f'recall={val["binary_membership"]["recall"]:.3f} '
                      f'best={state.best_val:.6f} '
                      f'best_epoch={state.best_epoch} '
                      f'patience={state.counter}/{state.patience} '
                      f'epoch_seconds={time.monotonic()-epoch_started:.0f} '
                      f'checkpoint={"best+last" if improved else "last"}',flush=True)
                if state.should_stop:
                    print(f'[EARLY_STOP] patience={state.patience} exhausted '
                          f'at epoch={epoch}; best epoch={state.best_epoch}',flush=True)
                    break
        _monitor_status(output,{'phase':'COMPLETE','global_step':step,
            'max_epochs':config['max_epochs'],'best_epoch':state.best_epoch,
            'best_validation_loss':state.best_val,'bad_epochs':state.counter,
            'patience':state.patience,
            'elapsed_seconds':round(time.monotonic()-run_started,1),
            'test_payload_access':0})
    return {'schema_version':'mssr.bc_v5.train_result.v1','output':str(output),
            'global_step':step,'test_payload_access':0}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--g2',type=Path,required=True)
    parser.add_argument('--normalizers',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=base.DEFAULT_CONFIG)
    parser.add_argument('--balanced-quota',type=int,default=16)
    parser.add_argument('--frequency-exponent',type=float,default=.5)
    parser.add_argument('--ready-v8',type=Path,default=None,
                        help='Stage678 readiness certificate for Stage11-v8 G2')
    parser.add_argument('--progress-every',type=int,default=50,
                        help='Emit atomic status and stdout progress every N TRAIN/VAL batches')
    parser.add_argument('--run',action='store_true',help='Explicitly enable full scratch BC-v5 training')
    args=parser.parse_args(argv)
    config=base.load_action_set_training_config(args.config)
    print(json.dumps(run(args.g2,args.normalizers,args.audit,args.output,config,
        run_training=args.run,balanced_quota=args.balanced_quota,
        frequency_exponent=args.frequency_exponent,readiness_path=args.ready_v8,
        progress_every=args.progress_every),sort_keys=True,allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
