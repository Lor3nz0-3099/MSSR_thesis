#!/usr/bin/env python3
"""Evaluate a frozen BC-v5 v8 best.pt or last.pt on unresampled VALIDATION.

Outputs detailed, reproducible metrics for the thesis. No training, no optimizer,
no dataset generation, no TEST access, no parameter updates or calibration.

From MSSR_thesis root:
 PYTHONPATH="$PWD" ~/.local/share/mssr-bc-v1/venv/bin/python -B \
   scripts/bc_v5/evaluate_checkpoint.py --check
 PYTHONPATH="$PWD" ~/.local/share/mssr-bc-v1/venv/bin/python -B \
   scripts/bc_v5/evaluate_checkpoint.py --validation
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_DEFAULT = ROOT / 'artifacts/bc_v1/runs/bc_v5_v8_primitive_first_20261117T143857Z_118314'
G2 = ROOT / 'artifacts/bc_v1/sidecar/g2_v8_bc_v5_stage11_full'
F2 = ROOT / 'artifacts/bc_v1/normalization/f2_v8_actor_visible_r1m'
STAGE5 = ROOT / 'artifacts/bc_v1/audits/bc_v5_v8/stage5_integrity.json'
READY = ROOT / 'artifacts/bc_v1/audits/bc_v5_v8/stage678_readiness.json'
DEFAULT_REPORT_BASE = ROOT / 'artifacts/bc_v1/evaluations'
EXPECTED_VALIDATION = 23749
EXPECTED_TRAIN = 82518
SCHEMA = 'mssr.bc_v5.validation_evaluation.v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256(path):
    hash_ = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            hash_.update(block)
    return hash_.hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def scores(row):
    tp, fp, fn = (int(row.get(k, 0)) for k in ('tp', 'fp', 'fn'))
    return {'precision': tp/(tp+fp) if tp+fp else None,
            'recall': tp/(tp+fn) if tp+fn else None,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'positive_support': tp+fn}


class NumericAccumulator:
    """Aggregates identical components/units; never mixes unlike physical units."""
    def __init__(self):
        self.bins = {}

    def add(self, condition, primitive, parameter, unit, pred, truth):
        p, t = float(pred), float(truth)
        require(math.isfinite(p) and math.isfinite(t), 'Nonfinite regression value')
        key = (str(condition), str(primitive), str(parameter), str(unit))
        b = self.bins.setdefault(key, dict(n=0, absolute=0., squared=0., target_sum=0.,
                                           target_squared=0., min_target=math.inf, max_target=-math.inf))
        error = p-t
        b['n'] += 1
        b['absolute'] += abs(error)
        b['squared'] += error*error
        b['target_sum'] += t
        b['target_squared'] += t*t
        b['min_target'] = min(t,b['min_target'])
        b['max_target'] = max(t,b['max_target'])
        require(math.isfinite(b['squared']), 'Overflow in squared regression error')

    def rows(self):
        output = []
        for (condition, primitive, parameter, unit), b in sorted(self.bins.items()):
            n=b['n']; rmse=math.sqrt(b['squared']/n)
            rng=b['max_target']-b['min_target']
            output.append(dict(condition=condition, primitive=primitive, parameter=parameter,
                               unit=unit, count=n, mae=b['absolute']/n, rmse=rmse,
                               nrmse_range=rmse/rng if rng>0 else None,
                               target_min=b['min_target'], target_max=b['max_target']))
        return output


class MembershipAccumulator:
    """Exact-set is per original graph, *after* excluding masked candidate slots."""
    def __init__(self):
        self.totals = Counter()
        self.per_primitive = defaultdict(Counter)
        self.per_task = defaultdict(Counter)
        self.per_morphology = defaultdict(Counter)
        self.numeric = NumericAccumulator()
        self.examples = 0

    def update(self, logits, labels, valid, owners, primitive_tags, group_task, group_morphology, threshold):
        import torch
        logits, labels, valid, owners = [v.detach().cpu() for v in (logits, labels, valid, owners)]
        require(logits.ndim == 1 and len(logits)==len(labels)==len(valid)==len(owners)==len(primitive_tags),
                'Mismatched membership tensors/tags')
        require(labels.dtype == torch.bool and valid.dtype == torch.bool, 'Invalid binary ground truth')
        require(not bool((labels & ~valid).any()), 'Positive candidate excluded by supervision mask')
        require(torch.isfinite(logits).all().item(), 'Nonfinite candidate logit')
        n = len(group_task)
        require(n == len(group_morphology) and n >= 1, 'Invalid graph metadata')
        require(not bool(((owners < 0)|(owners >= n)).any()), 'Invalid candidate graph ownership')
        prediction = logits >= float(threshold)
        t = self.totals
        t['valid_candidates'] += int(valid.sum())
        t['masked_candidates'] += int((~valid).sum())
        for i in range(len(labels)):
            if not bool(valid[i]):
                continue
            graph=int(owners[i]); tag=primitive_tags[i]
            truth=bool(labels[i]); proposed=bool(prediction[i])
            field='tp' if truth and proposed else 'fp' if proposed else 'fn' if truth else 'tn'
            t[field]+=1
            self.per_primitive[tag][field]+=1
            self.per_task[group_task[graph]][field]+=1
            self.per_morphology[group_morphology[graph]][field]+=1
        for group in range(n):
            group_mask=(owners == group) & valid
            a=labels[group_mask];p=prediction[group_mask]
            difference=int((a != p).sum())
            tp=int((a & p).sum());fp=int((p & ~a).sum());fn=int((a & ~p).sum())
            n_pred=int(p.sum());n_true=int(a.sum())
            t['examples']+=1
            t['exact_sets']+=int(difference == 0)
            t['target_actions']+=n_true
            t['predicted_actions']+=n_pred
            t['cardinality_absolute_error_sum']+=abs(n_pred-n_true)
            t['jaccard_sum']+=tp/(tp+fp+fn) if tp+fp+fn else 1.
            for key,bucket in ((group_task[group],self.per_task), (group_morphology[group],self.per_morphology)):
                bucket[key]['examples']+=1
                bucket[key]['exact_sets']+=int(difference == 0)
                bucket[key]['target_actions']+=n_true
                bucket[key]['predicted_actions']+=n_pred
        self.examples+=n
    def report(self):
        def enriched(counter):
            row=dict(counter); row.update(scores(counter))
            n=counter.get('examples',0)
            if n:
                row.update(exact_set_accuracy=counter.get('exact_sets',0)/n,
                           mean_target_actions=counter.get('target_actions',0)/n,
                           mean_predicted_actions=counter.get('predicted_actions',0)/n)
                if 'cardinality_absolute_error_sum' in counter:
                    row['mean_cardinality_absolute_error']=counter['cardinality_absolute_error_sum']/n
                if 'jaccard_sum' in counter:row['mean_jaccard']=counter['jaccard_sum']/n
            return row
        return {'global': enriched(self.totals),
                'primitive': {k:enriched(v) for k,v in sorted(self.per_primitive.items())},
                'task': {k:enriched(v) for k,v in sorted(self.per_task.items())},
                'morphology': {k:enriched(v) for k,v in sorted(self.per_morphology.items())},
                'regression': self.numeric.rows()}


def batch_tags(samples):
    """Exact ordering: all resource candidates in batch, then all structural ones."""
    resources, structural=[],[]
    tasks, morphologies=[] ,[]
    r_map, s_map, argument_map = [], [], []
    resource_offset = structural_offset = argument_offset = 0
    for idx,sample in enumerate(samples):
        route=sample['routing']
        r = route['resource_candidates']; s=route['structural_bindings']
        for local, candidate in enumerate(r):
            resources.append('resource:' + candidate.mode.mode_id)
            r_map.append((idx,local))
        for local,(action,_binding) in enumerate(s):
            structural.append('structural:' + action.action_id)
            s_map.append((idx,local))
        for local, domain in enumerate(route['argument_domains']):
            argument_map.append((idx,local))
        audit=route['audit']
        task=audit.task_stratum
        if not task:
            onehot=sample['inputs']['task_one_hot'][0]
            vocab=route['task_vocabulary']
            task = str(vocab[int(onehot.argmax())]) if len(vocab) else 'unknown'
        tasks.append(str(task))
        morphologies.append(str(audit.morphology_stratum or 'unknown'))
        resource_offset+=len(r);structural_offset+=len(s);argument_offset+=len(route['argument_domains'])
    return resources+structural,tasks,morphologies,r_map,s_map,argument_map


def resource_regression(model_prediction, batch, samples, r_map, membership, normalizers, pred_selected):
    """Read only positive, supervised numeric payload components from BC-v5 loss ABI."""
    y=batch['supervision']['resource']; p=model_prediction['parameters']['resource']
    candidate_rows={int(cid):j for j,cid in enumerate(p['payload_candidate'].detach().cpu().tolist())}
    pred_ptr=p['payload_ptr'].detach().cpu().tolist()
    target_ptr=y['payload_ptr'].detach().cpu().tolist()
    target_cand=y['candidate'].detach().cpu().tolist()
    supervised=y['supervised'].detach().cpu().tolist()
    target_mask=y['payload_valid'].detach().cpu().tolist()
    target_values=y['payload'].detach().cpu().tolist()
    pred_values=p['payload'].detach().cpu().tolist()
    bindings={(b.module_type_id,b.resource_id,b.mode_id):b for b in normalizers.resource_bindings}
    for j,cid in enumerate(target_cand):
        if not supervised[j]:continue
        require(0<=cid<len(r_map), 'Positive resource candidate index outside batch')
        if cid not in candidate_rows:continue  # categorical payload: no scalar regression
        sample_index,local_cid = r_map[cid]
        route=samples[sample_index]['routing']
        candidate=route['resource_candidates'][local_cid]
        module_types=dict(zip(route['module_refs'],route['module_types']))
        type_id=module_types[candidate.resource_ref.module]
        b=bindings.get((type_id,candidate.resource_ref.resource_id,candidate.mode.mode_id))
        require(b is not None and b.codec != 'categorical','Missing physical F2 binding for resource payload')
        p_row=candidate_rows[cid]
        p_a,p_b=pred_ptr[p_row:p_row+2]; t_a,t_b=target_ptr[j:j+2]
        require(p_b-p_a==t_b-t_a==len(b.keys), 'Resource payload width mismatch')
        primitive='resource:'+candidate.mode.mode_id
        physical_target={ (x.resource,x.mode_id):x for x in route['physical_supervision'].resource_targets }
        target=physical_target.get((candidate.resource_ref,candidate.mode.mode_id))
        require(target is not None and len(target.payload)==len(b.keys), 'Missing original physical positive payload')
        for component in range(t_b-t_a):
            if not target_mask[t_a+component]:continue
            observed=target_values[t_a+component]; predicted=pred_values[p_a+component]
            require(math.isfinite(observed) and math.isfinite(predicted),'Nonfinite resource regression')
            name='payload['+str(component)+']'
            membership.numeric.add('oracle_positive_normalized',primitive,name,'normalized',predicted,observed)
            if pred_selected[cid]:
                membership.numeric.add('selected_true_positive_normalized',primitive,name,'normalized',predicted,observed)
            scaler=normalizers.get(b.keys[component]);unit=b.units[component]
            pred_phys=predicted*scaler.scale+scaler.offset
            truth_phys=float(target.payload[component]);require(math.isfinite(pred_phys) and math.isfinite(truth_phys),'Nonfinite physical resource values')
            recovered=observed*scaler.scale+scaler.offset
            require(abs(recovered-truth_phys) <= 2e-4*max(1.,abs(truth_phys)),
                    'F2 resource inverse disagrees with original physical expert target')
            membership.numeric.add('oracle_positive_physical',primitive,name,unit,pred_phys,truth_phys)
            if pred_selected[cid]:
                membership.numeric.add('selected_true_positive_physical',primitive,name,unit,pred_phys,truth_phys)


def structural_regression(model_prediction, batch, samples, s_map, argument_map, resource_count, membership, normalizers, pred_selected):
    from drl.IL.bc_v1.losses import _argument_rows
    p=model_prediction['parameters']['structural']['arguments']['numeric']
    y=batch['supervision']['structural']; x=batch['inputs']['structural']
    p_rows={int(arg):j for j,arg in enumerate(p['argument'].detach().cpu().tolist())}
    p_ptr=p['ptr'].detach().cpu().tolist();p_val=p['value'].detach().cpu().tolist()
    y_ptr=y['argument_value_ptr'].detach().cpu().tolist()
    y_val=y['argument_value'].detach().cpu().tolist();y_mask=y['argument_value_valid'].detach().cpu().tolist()
    arg_bindings={(b.action_id,b.name):b for b in normalizers.argument_bindings}
    # _argument_rows uses the exact same train-time supervision filter and kind IDs.
    host={ 'inputs': {'structural': {k:v.detach().cpu() for k,v in x.items()}},
           'supervision': {'structural': {k:v.detach().cpu() for k,v in y.items()}} }
    for row,global_argument in _argument_rows(host,('scalar','vector')):
        if global_argument not in p_rows:
            raise ValueError('Missing BC-v5 numeric structural prediction')
        sample_index,local_argument=argument_map[global_argument]
        route=samples[sample_index]['routing']
        domain=route['argument_domains'][local_argument]
        candidate_glob=int(x['argument_binding'][global_argument].detach().cpu())
        require(0<=candidate_glob<len(s_map), 'Argument binding outside structural candidate set')
        s_index,s_local=s_map[candidate_glob]
        require(s_index==sample_index, 'Structural candidate/argument split mismatch')
        operation,_binding=route['structural_bindings'][s_local]
        param_name=domain.parameter.name
        binding=arg_bindings.get((operation.action_id,param_name))
        require(binding is not None and binding.keys, 'Missing structural numeric F2 binding')
        pp=p_rows[global_argument]; p_a,p_b=p_ptr[pp:pp+2]; t_a,t_b=y_ptr[row:row+2]
        require(p_b-p_a==t_b-t_a==len(binding.keys),'Structural numeric width mismatch')
        physical_source=route['physical_supervision'].structural_targets
        target=next((t for t in physical_source if t.operation_id==operation.action_id and
                     t.initiator==_binding.initiator and t.connectors==_binding.connectors),None)
        require(target is not None, 'Original physical structural target absent')
        parameter=next((a for a in target.parameters if a.name==param_name),None)
        require(parameter is not None,'Original physical structural parameter absent')
        original=[parameter.value] if binding.kind=='scalar' else list(parameter.value)
        for dim in range(t_b-t_a):
            if not y_mask[t_a+dim]:continue
            predicted=p_val[p_a+dim];observed=y_val[t_a+dim]
            primitive='structural:'+operation.action_id
            name=param_name+'['+str(dim)+']'
            membership.numeric.add('oracle_positive_normalized',primitive,name,'normalized',predicted,observed)
            if pred_selected[resource_count+candidate_glob]:
                membership.numeric.add('selected_true_positive_normalized',primitive,name,'normalized',predicted,observed)
            scaler=normalizers.get(binding.keys[dim]); physical=predicted*scaler.scale+scaler.offset
            truth=float(original[dim]);unit=json.loads(binding.keys[dim])[-1]
            require(abs(observed*scaler.scale+scaler.offset-truth)<=2e-4*max(1.,abs(truth)),
                    'F2 structural inverse disagrees with original physical expert parameter')
            membership.numeric.add('oracle_positive_physical',primitive,name,unit,physical,truth)
            if pred_selected[resource_count+candidate_glob]:
                membership.numeric.add('selected_true_positive_physical',primitive,name,unit,physical,truth)


def write_csv(path, rows, fieldnames):
    with path.open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def verify_preflight(run_path, checkpoint_name='best'):
    require(checkpoint_name in ('best', 'last'), 'Only best or last checkpoints are allowed')
    checkpoint_filename=checkpoint_name+'.pt'
    import torch
    from drl.IL.bc_v1 import train_action_set as base, bc_v5_train as v5
    from drl.IL.bc_v1.action_set_training import load_action_set_training_config
    from drl.IL.bc_v1.bc_v5_integrity import read_report
    from drl.IL.bc_v1.bc_v5_v8_readiness import load_readiness
    from drl.IL.bc_v1.bc_v5_binary import index_overlay

    require(run_path.is_dir(),'Training run not found: '+str(run_path))
    for p in (G2/'manifest.json', F2/'normalizers.json', STAGE5, READY,
              run_path/checkpoint_filename,run_path/'identity.json',run_path/'plan.json',run_path/'history.jsonl'):
        require(p.is_file(),'Required certified input absent: '+str(p))
    plan=json_read(run_path/'plan.json')
    require(plan.get('test_payload_access')==0 and plan.get('source_g2')==str(G2),
            'Run plan source/test policy differs')
    checkpoint=torch.load(run_path/checkpoint_filename,map_location='cpu',weights_only=False)
    require(checkpoint.get('schema_version')==v5.SCHEMA and checkpoint.get('model_contract')=='action_set_v1',
            'Not the certified BC-v5 checkpoint format')
    require(checkpoint.get('identity')==json_read(run_path/'identity.json'), 'Checkpoint/run identity mismatch')
    ck_config=checkpoint['config']
    require(float(ck_config['threshold_logit'])==0.0, 'Expected original decoder threshold 0.0')
    require(ck_config['patience']==8, 'Unexpected patience, do not confuse runs')
    preflight,kwargs,h1,h2,manifest=base.preflight(G2,ck_config,normalizer_dir=F2)
    require(checkpoint['identity']['model_source']['source_manifest_fingerprint']==manifest['manifest_fingerprint'],
            'Checkpoint trained on a different G2')
    topo=manifest.get('bc_v5_topology');require(isinstance(topo,dict),'Not the Stage11 final G2')
    ready,ready_fp=load_readiness(READY,manifest,STAGE5)
    require(ready_fp==checkpoint['identity']['stage678_readiness_fingerprint'], 'Stage678 identity mismatch')
    report=read_report(STAGE5,expected_g2_fingerprint=topo['source_manifest_fingerprint'])
    require(json_read(STAGE5)['fingerprint']==checkpoint['identity']['audit_fingerprint'], 'Stage5 identity mismatch')
    require(manifest['inventory']['test']==[], 'TEST data present in G2, refuse to run')
    overlay=index_overlay(report,split='validation')
    zero={(r['episode_id'],r['transition_index']) for r in ready['zero_positive_graphs']['validation']}
    val_eligible=sum(1 for k,v in overlay.items() if not v['exclude'] and k not in zero)
    require(val_eligible==EXPECTED_VALIDATION, 'Unexpected VALIDATION supervision count')
    require(checkpoint['identity']['sample_counts']['train']==EXPECTED_TRAIN and
            checkpoint['identity']['sample_counts']['validation']==EXPECTED_VALIDATION,
            'Checkpoint sample counts differ from final v8')
    summary=checkpoint.get('summary',{})
    require(isinstance(summary,dict) and isinstance(summary.get('epoch'),int) and summary['epoch']>=1,
            'No trained best epoch in checkpoint')
    model=v5.load_bc_v5_checkpoint(run_path/checkpoint_filename,kwargs['spec'],kwargs['feature_schema'],h1,h2,
                                   expected_identity=checkpoint['identity'],device='cpu')
    return dict(model=model, overlay=overlay,zero=zero,manifest=manifest,kwargs=kwargs,
                summary=summary, config=ck_config,ready_fpr=ready_fp,
                checkpoint_identity=checkpoint['identity'],checkpoint_name=checkpoint_name)


def evaluate(run_path, state, output, *, batch_size, threshold, device, progress_every, max_batches):
    import torch
    from drl.IL.bc_v1.action_set_training import open_dataset
    from drl.IL.bc_v1 import bc_v5_train as v5, train_action_set as base
    from drl.IL.bc_v1.bc_v5_binary import collate_binary_v5,compute_binary_v5_loss
    from drl.IL.bc_v1.trainer import FamilyAccumulator,validation_batches
    from drl.IL.bc_v1.training import family_statistics,move_batch_to_device

    # preflight already verified the actor-specific F2; reuse same authenticated instance.
    normalizers=state['kwargs']['normalizers']
    model=state['model'].to(device);model.eval()
    torch.set_num_threads(1)
    from drl.IL.bc_v1.action_set_loss import _candidate_graph
    metrics=MembershipAccumulator()
    family=FamilyAccumulator({k:1.0 for k in base.FAMILIES})
    start=time.monotonic();total_batches=math.ceil(EXPECTED_VALIDATION/batch_size)
    print(f'[START] {state["checkpoint_name"].upper()} checkpoint ep={state["summary"]["epoch"]} | VALIDATION {EXPECTED_VALIDATION} rows '
          f'| {total_batches} batches | threshold={threshold} | device={device}',flush=True)
    with open_dataset(G2,'validation',state['manifest']['identity'],
            dynamic_budget_bytes=256*1024**2,template_budget_bytes=128*1024**2) as raw:
        ds=v5.FilteredTrainingDataset(raw,state['overlay'],state['zero'])
        require(len(ds)==EXPECTED_VALIDATION,'Filtered VALIDATION differs from training')
        for i,indices in enumerate(validation_batches(len(ds),batch_size),start=1):
            samples=[ds.get_for_collate(k) for k in indices]
            tags,tasks,morphs,r_map,s_map,arg_map=batch_tags(samples)
            batch=move_batch_to_device(collate_binary_v5(samples,state['overlay']),device)
            with torch.inference_mode():
                pred=model(batch['inputs'])
                loss,stats=compute_binary_v5_loss(pred,batch)
            require(torch.isfinite(loss).item(),'Nonfinite validation loss')
            family.add(family_statistics(stats))
            logits=pred['presence_logits'];labels=batch['supervision']['action_set']['presence'];valid=batch['supervision']['action_set']['valid']
            expected_graph,n_graphs=_candidate_graph(batch['inputs'])
            require(n_graphs==len(samples) and torch.equal(pred['candidate_graph'].detach().cpu(),expected_graph),
                    'Candidate ownership differs from physical graph')
            metrics.update(logits,labels,valid,expected_graph,tags,tasks,morphs,threshold)
            selection=(logits.detach().cpu() >= threshold).tolist()
            resource_regression(pred,batch,samples,r_map,metrics,normalizers,selection)
            structural_regression(pred,batch,samples,s_map,arg_map,len(r_map),metrics,normalizers,selection)
            if i%progress_every==0 or i==total_batches or i==max_batches:
                elapsed=time.monotonic()-start
                eta=elapsed*(total_batches-i)/i
                print(f'[VAL] {i}/{total_batches} batches | {metrics.examples}/{EXPECTED_VALIDATION} graphs '
                      f'| elapsed={elapsed:.0f}s eta~{eta:.0f}s',flush=True)
            if max_batches and i>=max_batches:break
    result=metrics.report()
    result['global']['validation_loss']=family.compute()['total_loss']
    result['global']['batches_evaluated']=i
    result['global']['complete']=i==total_batches
    if i==total_batches:
        recorded=state['summary']['validation']['total_loss']
        require(abs(recorded-result['global']['validation_loss'])<3e-5,
                'Recomputed validation loss disagrees with checkpoint history')
    output.mkdir(parents=True,exist_ok=False)
    write_csv(output/'primitive_metrics.csv', [dict(primitive=key,**entry) for key,entry in result['primitive'].items()],
              ['primitive','tp','fp','fn','tn','positive_support','precision','recall','f1'])
    write_csv(output/'task_metrics.csv',[dict(task=key,**entry) for key,entry in result['task'].items()],
              ['task','tp','fp','fn','tn','examples','exact_sets','exact_set_accuracy','precision','recall','f1','positive_support','mean_target_actions','mean_predicted_actions'])
    write_csv(output/'morphology_metrics.csv',[dict(morphology=key,**entry) for key,entry in result['morphology'].items()],
              ['morphology','tp','fp','fn','tn','examples','exact_sets','exact_set_accuracy','precision','recall','f1','positive_support','mean_target_actions','mean_predicted_actions'])
    write_csv(output/'regression_metrics.csv',result['regression'],
              ['condition','primitive','parameter','unit','count','mae','rmse','nrmse_range','target_min','target_max'])
    identity={'schema_version':SCHEMA,'split':'validation','status':'COMPLETE' if i==total_batches else 'PARTIAL',
              'checkpoint':str(run_path/(state['checkpoint_name']+'.pt')),'checkpoint_sha256':sha256(run_path/(state['checkpoint_name']+'.pt')),
              'checkpoint_epoch':state['summary']['epoch'],'g2_manifest_fingerprint':state['manifest']['manifest_fingerprint'],
              'stage678_fingerprint':state['ready_fpr'],'f2_fingerprint':state['manifest']['identity']['f2_fingerprint'],
              'audit_fingerprint':state['checkpoint_identity']['audit_fingerprint'],
              'threshold_logit':threshold,'batch_size':batch_size,'mask_policy':'exact-set over valid candidate slots only',
              'regression_policy':'oracle positives and correctly selected positives; masked candidates excluded; physical original expert targets',
              'test_payload_access':0,'elapsed_seconds':round(time.monotonic()-start,2),
              'rows':metrics.examples,'expected_rows':EXPECTED_VALIDATION,
              'source_code_sha256':sha256(Path(__file__))}
    (output/'metrics.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
    (output/'identity.json').write_text(json.dumps(identity,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(f'[PASS] VALIDATION report: {output}',flush=True)
    print(f'[METRICS] loss={result["global"]["validation_loss"]:.6f} '
          f'P={result["global"]["precision"]:.3f} R={result["global"]["recall"]:.3f} '
          f'F1={result["global"]["f1"]:.3f} ExactSet={result["global"]["exact_set_accuracy"]:.3f}',flush=True)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true',help='Metadata/checkpoint identity only: no G2 samples opened')
    mode.add_argument('--validation',action='store_true',help='Evaluate full VALIDATION and save reports')
    parser.add_argument('--run-dir',type=Path,default=RUN_DEFAULT)
    parser.add_argument('--checkpoint',choices=('best','last'),default='best',help='Which existing checkpoint to evaluate')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--threshold',type=float,default=0.)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--progress-every',type=int,default=25)
    parser.add_argument('--max-batches',type=int,default=0,help='Limited smoke: report clearly marked PARTIAL')
    parser.add_argument('--output',type=Path,default=None,help='New output directory under artifacts/bc_v1/evaluations')
    args=parser.parse_args(argv)
    require(Path.cwd().resolve()==ROOT,'Run from MSSR_thesis root: cd ~/MSSR_thesis')
    require(args.batch_size>=1 and args.progress_every>=1 and args.max_batches>=0, 'Invalid batch/progress argument')
    require(math.isfinite(args.threshold),'Threshold must be finite')
    run_dir=args.run_dir.resolve()
    state=verify_preflight(run_dir,args.checkpoint)
    print(f'[PASS] Checkpoint {args.checkpoint}.pt epoch={state["summary"]["epoch"]} | audited G2/F2/Stage5/Stage678',flush=True)
    print(f'[PASS] TEST sealed | VALIDATION={EXPECTED_VALIDATION} | threshold={args.threshold}',flush=True)
    if args.check:
        print('[NEXT] --validation runs evaluation without changing training weights or data',flush=True)
        return 0
    import torch
    device=('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else ('cuda:0' if args.device=='cuda' else 'cpu')
    require(not device.startswith('cuda') or torch.cuda.is_available(), 'Requested CUDA not available')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination=(args.output or DEFAULT_REPORT_BASE/f'bc_v5_v8_{args.checkpoint}_epoch{state["summary"]["epoch"]}_validation_{stamp}').resolve()
    require(destination.is_relative_to(DEFAULT_REPORT_BASE.resolve()),'Output must be under evaluation artifact root')
    require(not destination.exists(),'Refusing to overwrite evaluation result directory')
    evaluate(run_dir,state,destination,batch_size=args.batch_size,threshold=args.threshold,
             device=device,progress_every=args.progress_every,max_batches=args.max_batches)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (RuntimeError,ValueError,KeyError,IndexError,AssertionError,FileNotFoundError) as exc:
        print(f'[STOP] {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        raise SystemExit(2)
