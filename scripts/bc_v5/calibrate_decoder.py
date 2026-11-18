#!/usr/bin/env python3
"""BC-v5 logit decoder threshold calibration on TRAIN-fitted-F2 VALIDATION ONLY.

Runs one immutable forward pass per validation graph for ALL requested logit thresholds.
Uses the same membership rule as BC-v4's action-set decoder: logits >= threshold.
Does not mutate checkpoint, training config, GUI, G2/F2, or TEST.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
import evaluate_checkpoint as ev  # intentional path-based local module, NOT ROS's `scripts`

DEFAULT_GRID=(-1.,-0.5,0.,0.25,0.5,0.75,1.,1.25,1.5,2.,2.5,3.,4.,5.)


def thresholds_from_text(text):
    try:
        values=tuple(float(x.strip()) for x in text.split(','))
    except ValueError as exc:
        raise ValueError('Thresholds must be comma-separated finite numbers') from exc
    ev.require(len(values)>=2 and all(math.isfinite(x) for x in values),
               'Need at least two finite logit thresholds')
    ev.require(0. in values,'Threshold grid must include baseline 0.0')
    ev.require(len(values)==len(set(values)),'Duplicate decoder thresholds')
    return tuple(sorted(values))


def _scores(tp,fp,fn):
    return dict(precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


class DecoderSweep:
    """Numpy counters for exact set and per-primitive diagnostics; only valid candidate slots."""
    def __init__(self, thresholds):
        self.thresholds=tuple(thresholds)
        self.buckets={t:dict(tp=0,fp=0,fn=0,tn=0,exact_sets=0,predicted_actions=0)
                      for t in self.thresholds}
        self.rows=0
        self.target_actions=0
        self.per_primitive=defaultdict(list)
        self.primitive_counts=defaultdict(int)
        self.masked=0

    def update(self, logits, labels, valid, owners, tags, graphs):
        logits=np.asarray(logits,dtype=np.float32)
        labels=np.asarray(labels,dtype=np.bool_)
        valid=np.asarray(valid,dtype=np.bool_)
        owners=np.asarray(owners,dtype=np.int64)
        ev.require(logits.ndim==labels.ndim==valid.ndim==owners.ndim==1 and
                   len(logits)==len(labels)==len(valid)==len(owners)==len(tags),
                   'Inconsistent calibration candidate layout')
        ev.require(graphs>=1 and np.all(owners>=0) and np.all(owners<graphs),
                   'Invalid candidate-to-graph indices')
        ev.require(np.isfinite(logits).all() and not np.any(labels & ~valid),
                   'Nonfinite logit or masked positive candidate')
        self.rows+=graphs
        self.target_actions+=int(np.count_nonzero(labels & valid))
        self.masked+=int(np.count_nonzero(~valid))
        # Store only (valid logits, valid labels) by legal primitive, in RAM; no G2 disk copy.
        # This is a bounded few-MB index, not the full G2 tensors or predictions.
        names=set(tags)
        for name in names:
            indices=np.fromiter((i for i,n in enumerate(tags) if n==name and valid[i]),
                                dtype=np.int64)
            if len(indices):
                self.per_primitive[name].append((logits[indices].copy(),labels[indices].copy()))
                self.primitive_counts[name]+=len(indices)
        for threshold in self.thresholds:
            proposed=logits>=threshold
            positives=proposed & valid
            tp=int(np.count_nonzero(positives & labels))
            fp=int(np.count_nonzero(positives & ~labels))
            fn=int(np.count_nonzero(~proposed & valid & labels))
            tn=int(np.count_nonzero(~proposed & valid & ~labels))
            wrong=(proposed != labels)&valid
            # One mismatch is enough to make an observed action set non-exact.
            per_graph=np.bincount(owners[wrong],minlength=graphs)
            bucket=self.buckets[threshold]
            bucket['tp']+=tp; bucket['fp']+=fp; bucket['fn']+=fn; bucket['tn']+=tn
            bucket['exact_sets']+=int(np.count_nonzero(per_graph==0))
            bucket['predicted_actions']+=tp+fp

    def summary(self):
        ev.require(self.rows>0, 'No VALIDATION examples processed')
        rows=[]
        for threshold,bucket in self.buckets.items():
            row=dict(threshold_logit=threshold,**bucket,**_scores(bucket['tp'],bucket['fp'],bucket['fn']))
            row.update(rows=self.rows,exact_set_accuracy=bucket['exact_sets']/self.rows,
                       mean_predicted_actions=bucket['predicted_actions']/self.rows,
                       mean_expert_actions=self.target_actions/self.rows)
            rows.append(row)
        return rows

    def per_primitive_summary(self):
        results=[]
        for primitive, parts in sorted(self.per_primitive.items()):
            logits=np.concatenate([a for a,_ in parts]); labels=np.concatenate([b for _,b in parts])
            for threshold in self.thresholds:
                proposed=logits>=threshold
                tp=int(np.count_nonzero(proposed & labels))
                fp=int(np.count_nonzero(proposed & ~labels))
                fn=int(np.count_nonzero(~proposed & labels))
                tn=int(np.count_nonzero(~proposed & ~labels))
                results.append(dict(primitive=primitive,threshold_logit=threshold,tp=tp,fp=fp,fn=fn,tn=tn,
                                    positive_support=tp+fn,**_scores(tp,fp,fn)))
        return results


def recommend_threshold(rows, max_recall_drop=0.01):
    """Select maximum VALIDATION exact set under global recall degradation guard.

    Threshold 0.0 is always considered as fallback; never changes model/GUI automatically.
    """
    ev.require(0 <= max_recall_drop < 1, 'Invalid recall drop constraint')
    baseline=next(r for r in rows if r['threshold_logit']==0.)
    ev.require(baseline['recall'] is not None, 'No positive validation candidate')
    floor=max(0.,baseline['recall']-max_recall_drop)
    eligible=[r for r in rows if r['recall'] is not None and r['recall']+1e-12>=floor]
    ev.require(bool(eligible),'No threshold satisfies the recall guard')
    selected=max(eligible,key=lambda r:(r['exact_set_accuracy'],r['f1'] or 0.,-abs(r['threshold_logit'])))
    return baseline,selected,floor


def write_csv(path, rows, fieldnames):
    with path.open('x',encoding='utf-8',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=fieldnames,extrasaction='ignore')
        w.writeheader();w.writerows(rows)


def run(args):
    import torch
    from drl.IL.bc_v1.action_set_training import open_dataset
    from drl.IL.bc_v1 import bc_v5_train as v5
    from drl.IL.bc_v1.bc_v5_binary import collate_binary_v5
    from drl.IL.bc_v1.trainer import validation_batches
    from drl.IL.bc_v1.training import move_batch_to_device
    from drl.IL.bc_v1.action_set_loss import _candidate_graph

    ev.require(Path.cwd().resolve()==ev.ROOT,'Run from MSSR_thesis root')
    ev.require(args.batch_size>0 and args.progress_every>0,'Invalid batch/progress')
    ev.require(0<=args.max_recall_drop<1,'Invalid recall guard')
    thresholds=thresholds_from_text(args.thresholds)
    path=args.run_dir.resolve()
    state=ev.verify_preflight(path,args.checkpoint)
    print(f'[PASS] {args.checkpoint}.pt epoch={state["summary"]["epoch"]} | authenticated G2/F2/supervision | TEST sealed',flush=True)
    device=('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else ('cuda:0' if args.device=='cuda' else 'cpu')
    ev.require(not device.startswith('cuda') or torch.cuda.is_available(),'CUDA unavailable')
    torch.set_num_threads(1)
    model=state['model'].to(device);model.eval()
    n=ev.EXPECTED_VALIDATION
    total_batches=math.ceil(n/args.batch_size)
    results=DecoderSweep(thresholds)
    start=time.monotonic()
    print(f'[START] VALIDATION {n} rows, {total_batches} batches; {len(thresholds)} thresholds in ONE forward pass',flush=True)
    with open_dataset(ev.G2,'validation',state['manifest']['identity'],
                      dynamic_budget_bytes=256*1024**2,template_budget_bytes=128*1024**2) as raw:
        ds=v5.FilteredTrainingDataset(raw,state['overlay'],state['zero'])
        ev.require(len(ds)==n,'VALIDATION count changed')
        for step,indices in enumerate(validation_batches(n,args.batch_size),start=1):
            samples=[ds.get_for_collate(i) for i in indices]
            tags,tasks,morphs,*_=ev.batch_tags(samples)
            batch=move_batch_to_device(collate_binary_v5(samples,state['overlay']),device)
            with torch.inference_mode(): pred=model(batch['inputs'])
            expected_graph,graph_count=_candidate_graph(batch['inputs'])
            ev.require(graph_count==len(samples) and torch.equal(pred['candidate_graph'].detach().cpu(),expected_graph),
                       'Candidate ownership mismatch')
            logits=pred['presence_logits'].detach().cpu().numpy()
            labels=batch['supervision']['action_set']['presence'].detach().cpu().numpy()
            valid=batch['supervision']['action_set']['valid'].detach().cpu().numpy()
            owners=expected_graph.numpy()
            results.update(logits,labels,valid,owners,tags,graph_count)
            if step%args.progress_every==0 or step==total_batches:
                elapsed=time.monotonic()-start
                eta=elapsed*(total_batches-step)/step
                print(f'[SWEEP] {step}/{total_batches} batches | {results.rows}/{n} graphs | elapsed={elapsed:.0f}s eta~{eta:.0f}s',flush=True)
    sweep=results.summary()
    baseline,selected,recall_floor=recommend_threshold(sweep,args.max_recall_drop)
    primitives=results.per_primitive_summary()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=(args.output or ev.DEFAULT_REPORT_BASE/f'bc_v5_v8_{args.checkpoint}_epoch{state["summary"]["epoch"]}_decoder_calibration_{stamp}').resolve()
    ev.require(output.is_relative_to(ev.DEFAULT_REPORT_BASE.resolve()),'Output must be under evaluation root')
    output.mkdir(parents=True,exist_ok=False)
    write_csv(output/'threshold_sweep.csv',sweep,
              ['threshold_logit','precision','recall','f1','exact_set_accuracy','tp','fp','fn','tn','rows','exact_sets','mean_predicted_actions','mean_expert_actions'])
    write_csv(output/'primitive_by_threshold.csv',primitives,
              ['primitive','threshold_logit','tp','fp','fn','tn','positive_support','precision','recall','f1'])
    identity={'schema_version':'mssr.bc_v5.decoder_validation_calibration.v1','checkpoint':str(path/(args.checkpoint+'.pt')),
              'checkpoint_sha256':ev.sha256(path/(args.checkpoint+'.pt')),
              'checkpoint_epoch':state['summary']['epoch'],
              'g2_manifest_fingerprint':state['manifest']['manifest_fingerprint'],
              'f2_fingerprint':state['manifest']['identity']['f2_fingerprint'],
              'thresholds_logit':list(thresholds),'validation_rows':results.rows,
              'masked_candidates':results.masked,'test_payload_access':0,
              'source_code_sha256':ev.sha256(Path(__file__)),
              'decoder_rule':'presence_logits >= threshold_logit',
              'mask_policy':'VALID candidate slots ONLY; no masked supervision counted',
              'elapsed_seconds':round(time.monotonic()-start,3)}
    config={'schema_version':'mssr.bc_v5.decoder_calibration_result.v1',
            'baseline_threshold_logit':0.,'suggested_threshold_logit':selected['threshold_logit'],
            'max_global_recall_drop':args.max_recall_drop,'minimum_global_recall':recall_floor,
            'objective':'maximize exact-set accuracy with global recall guard on VALIDATION',
            'baseline':baseline,'suggested':selected,
            'warning':'Proposal only. Verify per-primitive recall and GUI before applying; never use TEST to tune.',
            'gui_decoder_changed':False,'checkpoint_changed':False,'test_payload_access':0}
    (output/'calibration.json').write_text(json.dumps(config,indent=2,sort_keys=True,allow_nan=False)+'\n')
    (output/'identity.json').write_text(json.dumps(identity,indent=2,sort_keys=True,allow_nan=False)+'\n')
    (output/'gui_decoder_proposal.json').write_text(json.dumps({
        'checkpoint':str(path/(args.checkpoint+'.pt')),'threshold_logit':selected['threshold_logit'],
        'original_threshold_logit':0.,'apply_automatically':False,
        'note':'Review primitive_by_threshold.csv and GUI runs before applying this proposal.'
    },indent=2,sort_keys=True)+'\n')
    print(f'[PASS] Calibration: {output}',flush=True)
    print('[BASELINE] threshold=0.0 ExactSet={:.3f} F1={:.3f} P={:.3f} R={:.3f}'.format(
          baseline['exact_set_accuracy'],baseline['f1'],baseline['precision'],baseline['recall']),flush=True)
    print('[PROPOSAL] threshold={:.2f} ExactSet={:.3f} F1={:.3f} P={:.3f} R={:.3f}; recall_floor={:.3f}'.format(
          selected['threshold_logit'],selected['exact_set_accuracy'],selected['f1'],selected['precision'],selected['recall'],recall_floor),flush=True)
    print('[NOTE] GUI remains at threshold=0.0; TEST sealed; no data or weights modified',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=ev.RUN_DEFAULT)
    parser.add_argument('--checkpoint',choices=('best','last'),default='last')
    parser.add_argument('--thresholds',default=','.join(str(x) for x in DEFAULT_GRID))
    parser.add_argument('--max-recall-drop',type=float,default=0.01,
                        help='Max drop in global candidate recall relative to 0.0 (default 0.01 = one percentage point)')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--progress-every',type=int,default=25)
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args()
    run(args)


if __name__=='__main__':
    try:main()
    except (RuntimeError,ValueError,KeyError,IndexError,AssertionError,FileNotFoundError) as exc:
        print(f'[STOP] {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        raise SystemExit(2)
