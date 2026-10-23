"""Manual-only K2 small-overfit / TRAIN+VAL retraining. Never opens TEST.

Nothing executes at import. Run only after the audited G2-v2 build and review.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch

from .action_heads import load_action_head_config
from .batching import collate_policy_samples
from .losses import load_loss_config,compute_bc_loss
from .metrics import BCMetrics,load_metrics_config
from .model import BCGNN,load_model_config
from .sampling import JointActionBatchSampler
from .sidecar import _hash
from .trainer import open_dataset,train_epoch,evaluate_validation,SelectionState
from .training import build_optimizer,move_batch_to_device,train_step,save_checkpoint
from .no_structural_prepare import ROOT,DEFAULT_CONFIG,frozen_identity,write_json,file_sha
from .bc_v1_contract import load_robot_spec,robot_spec_fingerprint
from .feature_schema import load_feature_schema,feature_schema_fingerprint
from .normalization import read_normalizers,normalizer_fingerprint

CONFIG=ROOT/'configs/bc_v1/training_no_structural_v2.json'
K1=Path('/home/lorenzo/MSSR_thesis/artifacts/bc_v1/checkpoints/k1_baseline_v1/best_checkpoint.pt')


def warm_start(model,state):
    """Only the new NONE scorer may be absent; never silently drop old weights."""
    expected={k for k in model.state_dict() if 'action_heads.no_structural_scorer.' in k}
    missing=set(model.state_dict())-set(state)
    if missing!=expected or set(state)-set(model.state_dict()):raise ValueError('K1 warm-start requires exact old architecture plus new NONE scorer')
    result=model.load_state_dict(state,strict=False)
    if set(result.missing_keys)!=expected or result.unexpected_keys:raise ValueError('Unexpected initialization mismatch')
    return {'reused_tensors':len(state),'new_parameters':sorted(expected)}


def load_config(path=CONFIG):
    cfg=json.loads(Path(path).read_text())
    baseline=json.loads((ROOT/'configs/bc_v1/training_baseline_v1.json').read_text())
    allowed={'schema_version','general_quota','structural_quota','early_stopping_patience','audit_dir','checkpoint_dir','run_dir'}
    if any(cfg.get(k)!=v for k,v in baseline.items() if k not in allowed):raise ValueError('Unexpected baseline hyperparameter change')
    if (cfg['schema_version']!='mssr.bc_v1.training_no_structural.v2' or cfg['early_stopping_patience']!=8
            or cfg['structural_quota'] not in (16,24) or cfg['general_quota']+cfg['structural_quota']!=64):
        raise ValueError('K2 requires patience8, batch64 and structural quota16 (optional single retry24)')
    for key in ('audit_dir','checkpoint_dir','run_dir'):
        if 'k2_no_structural' not in cfg[key] or not (ROOT/cfg[key]).resolve().is_relative_to(ROOT.resolve()):raise ValueError('Unsafe K2 output path')
    return cfg


def components(cfg):
    frozen=frozen_identity()
    if not torch.cuda.is_available():raise ValueError('Prepared K2 run requires CUDA; no silent CPU fallback')
    torch.set_num_threads(cfg['cpu_threads']);random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    torch.use_deterministic_algorithms(False)
    prepare=json.loads(DEFAULT_CONFIG.read_text());cache=ROOT/prepare['output'];doc=json.loads((cache/'manifest.json').read_text())
    from .structural_decisions import VERSION
    if doc.get('structural_decision_schema')!=VERSION or doc['inventory']['test']:raise ValueError('Expected reviewed decision-v2 TRAIN/VAL-only artifact')
    for name,digest in doc['decision_audit']['code_hashes'].items():
        if file_sha(Path(__file__).with_name(name+'.py'))!=digest:raise ValueError('G2-v2 code identity changed: '+name)
    if doc['decision_audit']['physical_operations']!=prepare['expected_train_validation_positive_counts']:raise ValueError('Positive structural counts changed')
    if doc['split_counts']['train']['samples']!=78302 or doc['split_counts']['validation']['samples']!=22762:raise ValueError('Incomplete TRAIN/VAL v2 artifact')
    pinned={k:doc['identity'][k] for k in prepare['source_pins']}
    if pinned!=prepare['source_pins']:raise ValueError('G2-v2 frozen upstream pins changed')
    f2path=ROOT/'artifacts/bc_v1/normalization/f2_v1_optimized/normalizers.json'
    stats=read_normalizers(f2path,expected_metadata=json.loads(f2path.read_text())['metadata'])
    spec=load_robot_spec(ROOT/'configs/bc_v1/smores_ep.robot.json');schema=load_feature_schema(ROOT/'configs/bc_v1/features.json')
    h1=load_model_config(ROOT/'configs/bc_v1/gnn_v1.json');h2=load_action_head_config(ROOT/'configs/bc_v1/action_heads_no_structural_v2.json')
    if (robot_spec_fingerprint(spec)!=doc['identity']['robot_spec_fingerprint'] or feature_schema_fingerprint(schema)!=doc['identity']['feature_schema_fingerprint']
            or normalizer_fingerprint(stats)!=doc['identity']['f2_fingerprint']):raise ValueError('Learned inputs/F2 identity changed')
    model=BCGNN(spec,schema,h1,action_head_config=h2).to(cfg['device'])
    payload=torch.load(K1,map_location='cpu',weights_only=False)
    for key,current in [('h1_config_fingerprint',_hash(h1)),('robot_spec_fingerprint',robot_spec_fingerprint(spec)),('feature_schema_fingerprint',feature_schema_fingerprint(schema)),('f2_fingerprint',normalizer_fingerprint(stats))]:
        if payload['identity'].get(key)!=current:raise ValueError('K1 initialization contract changed: '+key)
    initialized=warm_start(model,payload['model'])
    identity={**doc['identity'],'sidecar_manifest_fingerprint':doc['manifest_fingerprint'],
        'source_initialization_sha':frozen['k1'],'structural_decision_schema':VERSION,
        'h1_config_fingerprint':_hash(h1),'h2_config_fingerprint':_hash(h2),'training_config_fingerprint':_hash(cfg)}
    loss=load_loss_config(ROOT/'configs/bc_v1/loss_v1.json')
    metrics=BCMetrics(spec,stats,{**load_metrics_config(ROOT/'configs/bc_v1/metrics_v1.json'),'breakdowns':cfg['metrics_breakdowns']})
    return model,build_optimizer(model,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']),loss,metrics,identity,cache,doc,initialized,frozen


def collapsed(decisions,*,k1_positive_accuracy=.9095477386934674):
    """Declared anti-collapse gate, not an HPO objective: reject loss of >half baseline."""
    positive=decisions['positive']
    return (not positive['support'] or positive['recall'] is None or positive['recall']<.5
        or positive['accuracy'] is None or positive['accuracy']<.5*k1_positive_accuracy)


class CandidateSelection(SelectionState):
    """Patience and objective apply to non-collapsed trained candidates only."""
    def observe_candidate(self,epoch,val_total,decisions):
        if not math.isfinite(val_total) or epoch!=self.last_epoch+1:
            raise ValueError('Finite consecutive validation epochs required')
        self.last_epoch=epoch
        if epoch==0:self.initial_val=val_total
        improved=epoch>0 and not collapsed(decisions) and val_total<self.best_val-self.min_delta
        if improved:self.best_epoch,self.best_val,self.counter=epoch,val_total,0
        elif epoch>0:self.counter+=1
        return improved


def small_overfit(cfg,*,output):
    output=Path(output)
    if output.exists():raise ValueError('Small-overfit output already exists')
    model,optimizer,loss,metrics,identity,cache,doc,initialized,frozen=components(cfg)
    output.mkdir(parents=True)
    with open_dataset(cache,'train',doc['identity']) as data:
        index=data.train_index();positive=[i for rows in index.structural_rows.values() for i in rows]
        if not positive:raise ValueError('No TRAIN structural positive')
        # Whole original samples, not isolated/fabricated targets; regular + positive mix.
        selected=list(dict.fromkeys([0,positive[0],positive[len(positive)//3],positive[2*len(positive)//3],positive[-1]]))
        batch=move_batch_to_device(collate_policy_samples([data[i] for i in selected]),cfg['device'])
        history=[];success=False
        for step in range(501):
            if step:
                result=train_step(model,batch,optimizer,loss,max_grad_norm=cfg['gradient_clip'])
                if not math.isfinite(result['grad_norm_before_clip']):raise ValueError('Nonfinite overfit gradient')
            if step%25==0:
                model.eval()
                with torch.inference_mode():
                    p=model(batch['inputs']);total,_=compute_bc_loss(p,batch,loss);metrics.reset();metrics.update(p,batch)
                    decisions=metrics.compute()['structural']['decisions']
                history.append({'step':step,'loss':float(total),'decisions':decisions})
                print('K2_OVERFIT '+json.dumps(history[-1]),flush=True)
                if step and float(total)<history[0]['loss'] and decisions['no_structural']['recall'] is not None and decisions['no_structural']['recall']>=.8 and decisions['positive']['accuracy'] is not None and decisions['positive']['accuracy']>=.8:
                    success=True;break
        report={'success':success,'history':history,'train_indices':selected,'finite_gradients':True,'initialization':initialized,
            'identity':identity,'frozen_before':frozen,'frozen_after':frozen_identity(),'TEST_access':0}
        write_json(output/'report.json',report)
        if not success:raise ValueError('Small-overfit failed: do not start full training')
    return report


def train(cfg,*,overfit_report):
    proof=json.loads(Path(overfit_report).read_text())
    if not proof.get('success') or proof.get('TEST_access')!=0:raise ValueError('Successful TRAIN-only small-overfit report required')
    model,optimizer,loss,metrics,identity,cache,doc,initialized,frozen=components(cfg)
    if proof['identity']!=identity:raise ValueError('Overfit identity differs from current training setup')
    audit=ROOT/cfg['audit_dir'];checkpoints=ROOT/cfg['checkpoint_dir'];run=ROOT/cfg['run_dir']
    for path in (audit,checkpoints,run):
        if path.exists() and any(path.iterdir()):raise ValueError('Refusing to overwrite previous K2 run')
    for path in (audit,checkpoints,run):path.mkdir(parents=True,exist_ok=True)
    write_json(audit/'config.json',cfg);write_json(audit/'initialization.json',initialized)
    status={'state':'RUNNING','epoch':0,'step':0,'TEST_access':0};write_json(run/'status.json',status)
    selection=CandidateSelection(patience=8,min_delta=cfg['min_delta']);best=None;global_step=0;no_valid_epochs=0
    with open_dataset(cache,'train',doc['identity']) as train_data,open_dataset(cache,'validation',doc['identity']) as val_data:
        sampler=JointActionBatchSampler(train_data.train_index(),batch_size=64,ordinary_quota=cfg['general_quota'],structural_quota=cfg['structural_quota'],seed=cfg['seed'])
        with (audit/'history.jsonl').open('x',buffering=1) as history:
            for epoch in range(cfg['max_epochs']+1):
                tr=None
                if epoch:
                    tr=train_epoch(model,train_data,sampler,optimizer,loss,device=cfg['device'],epoch=epoch,global_step=global_step,
                        gradient_clip=cfg['gradient_clip'],log_interval=cfg['logging_cadence'])
                    if tr['steps']!=cfg['steps_per_epoch']:raise ValueError('Incomplete full TRAIN epoch')
                    global_step=tr['global_step']
                val=evaluate_validation(model,val_data,loss,metrics,device=cfg['device'],batch_size=64,
                    epoch=epoch,log_interval=cfg['logging_cadence'])
                if val['samples']!=22762:raise ValueError('Incomplete full VAL')
                decisions=val['metrics']['structural']['decisions'];invalid=collapsed(decisions)
                # Ineligible epochs consume patience without becoming the
                # objective reference for a valid deployment candidate.
                improved=selection.observe_candidate(epoch,val['total_loss'],decisions)
                eligible=(epoch>0 and not invalid)
                if improved:
                    best={'epoch':epoch,'total_loss':val['total_loss'],'decisions':decisions}
                    save_checkpoint(checkpoints/'best_checkpoint.pt',model,optimizer,step=global_step,identity=identity,config=cfg,
                        selection={'metric':cfg['selection_metric'],'anti_collapse_gate':'recall>=0.5, accuracy>=0.5*K1 VAL positive baseline'},seed=cfg['seed'],summary=best)
                    write_json(audit/'best_val_metrics.json',{'epoch':epoch,**val})
                no_valid_epochs=no_valid_epochs+1 if epoch and not eligible else 0
                summary={'epoch':epoch,'global_step':global_step,'best':best,'validation_collapsed':invalid,'TEST_access':0}
                save_checkpoint(checkpoints/'last_checkpoint.pt',model,optimizer,step=global_step,identity=identity,config=cfg,
                    selection={'metric':cfg['selection_metric']},seed=cfg['seed'],summary=summary)
                row={'epoch':epoch,'train':tr,'validation':val,'collapsed':invalid,'eligible':eligible,'objective_improved':improved}
                history.write(json.dumps(row,allow_nan=False)+'\n');history.flush();os.fsync(history.fileno())
                print('K2_EPOCH '+json.dumps({'epoch':epoch,'loss':val['total_loss'],'best':best,'decisions':decisions}),flush=True)
                write_json(run/'status.json',dict(status,epoch=epoch,step=global_step,best=best))
                if selection.should_stop or no_valid_epochs>=8:break
    if frozen_identity()!=frozen:raise ValueError('Frozen artifacts changed')
    report={'success':best is not None,'best':best,'epochs':epoch,'patience':8,'structural_quota':cfg['structural_quota'],
        'initialization':initialized,'identity':identity,'frozen_before':frozen,'frozen_after':frozen,'TEST_access':0,
        'checkpoint_sha256':file_sha(checkpoints/'best_checkpoint.pt') if best else None,'status':'VAL-selected candidate; TEST not evaluated'}
    write_json(audit/'report.json',report);write_json(run/'status.json',dict(status,state='COMPLETED' if best else 'FAILED_COLLAPSE',epoch=epoch,best=best))
    if best is None:raise ValueError('No anti-collapse-valid checkpoint; review report before at most one quota24 retry')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('small-overfit','train'));parser.add_argument('--config',type=Path,default=CONFIG)
    parser.add_argument('--overfit-report',type=Path,default=ROOT/'artifacts/bc_v1/audits/k2_no_structural_overfit_v1/report.json')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/bc_v1/audits/k2_no_structural_overfit_v1')
    args=parser.parse_args();cfg=load_config(args.config)
    if args.mode=='small-overfit':small_overfit(cfg,output=args.output)
    else:train(cfg,overfit_report=args.overfit_report)


if __name__=='__main__':main()
