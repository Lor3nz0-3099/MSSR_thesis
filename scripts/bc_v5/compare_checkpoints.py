#!/usr/bin/env python3
"""Compare existing complete best.pt and last.pt VALIDATION reports at logit 0.0.

Read-only: writes no dataset, and never opens TEST or model checkpoints.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import evaluate_checkpoint as ev


def open_eval(path, expected_ckpt):
    info=ev.json_read(path/'identity.json'); result=ev.json_read(path/'metrics.json')
    ev.require(info['status']=='COMPLETE' and info['split']=='validation' and
               info['threshold_logit']==0. and info['rows']==ev.EXPECTED_VALIDATION and
               info['test_payload_access']==0 and
               Path(info['checkpoint']).name==expected_ckpt+'.pt',
               f'Incompatible/incomplete {expected_ckpt} VALIDATION report: {path}')
    ev.require(result['global']['complete'] is True, 'Evaluation not complete')
    return info,result


def find_report(checkpoint_name):
    paths=sorted(ev.DEFAULT_REPORT_BASE.glob(f'bc_v5_v8_{checkpoint_name}_epoch*_validation_*/metrics.json'),
                 key=lambda p:p.parent.name)
    ev.require(bool(paths), 'Missing '+checkpoint_name+' VALIDATION report')
    return paths[-1].parent


def _num(x):
    if x is None:return '-'
    return f'{float(x):.5f}'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--best-report',type=Path,default=None)
    ap.add_argument('--last-report',type=Path,default=None)
    args=ap.parse_args()
    best=args.best_report or find_report('best')
    last=args.last_report or find_report('last')
    a,am=open_eval(best,'best'); b,bm=open_eval(last,'last')
    for key in ('g2_manifest_fingerprint','stage678_fingerprint','f2_fingerprint','audit_fingerprint'):
        ev.require(a[key]==b[key],f'Best/last use different {key}')
    names=[('validation_loss','VAL loss'),('precision','Precision'),('recall','Recall'),('f1','F1'),
           ('exact_set_accuracy','Exact-Set'),('mean_predicted_actions','Predicted actions / graph'),
           ('mean_target_actions','Expert actions / graph')]
    print('[COMPARE] Strict threshold=0.0; same F2/G2/VALIDATION, checkpoint only',flush=True)
    print(f'{"METRIC":31s} {"BEST epoch 12":>15s} {"LAST epoch 20":>15s}  delta(last-best)',flush=True)
    for key,label in names:
        x=am['global'].get(key);y=bm['global'].get(key)
        print(f'{label:31s} {_num(x):>15s} {_num(y):>15s}  {_num(y-x if x is not None and y is not None else None)}',flush=True)
    print('\n[PRIMITIVES] F1 at threshold=0.0:',flush=True)
    for name in sorted(set(am['primitive'])|set(bm['primitive'])):
        x=am['primitive'].get(name,{}).get('f1'); y=bm['primitive'].get(name,{}).get('f1')
        print(f' {name:32s} best={_num(x):>10s} last={_num(y):>10s}',flush=True)
    print('[NOTE] MAE/RMSE and other details are preserved in each checkpoint evaluation report.',flush=True)
    print(f'[BEST] {best}',flush=True)
    print(f'[LAST] {last}',flush=True)

if __name__=='__main__':
    try:main()
    except (RuntimeError,ValueError,KeyError,IndexError,AssertionError,FileNotFoundError) as exc:
        print(f'[STOP] {type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        raise SystemExit(2)
