"""Canonical-v8 Stage-10 mask disposition. No legacy candidate restoration.

This is a provenance record of unrepresentable positive align_faces labels,
NOT a license to train. Stage 11 and the binary loss must implement masks
and prove the excluded positives do not become negative labels.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

SCHEMA = 'mssr.bc_v5.v8.align_mask_disposition.v1'
STAGE7_SCHEMA = 'mssr.bc_v5.topology_positive_preflight.v1'
STAGE8_SCHEMA = 'mssr.bc_v5.align_joint_history.v1'

def sha(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()

def key(item):
    split, eid, idx, mid = (item.get('split'),item.get('episode_id'),
                             item.get('transition_index'),item.get('initiator'))
    if (split not in ('train','validation') or not isinstance(eid,str) or not eid
            or type(idx) is not int or idx < 0 or not isinstance(mid,str) or not mid):
        raise ValueError('Malformed align_faces exception identity')
    return split,eid,idx,mid

def build(config, stage7, stage8):
    pin = config['pins']; expected = config['stage10']['expected_missing_align']
    if set(expected) != {'train','validation'} or any(
            type(n) is not int or n < 0 for n in expected.values()):
        raise ValueError('Invalid frozen Stage-10 v8 expected counts')
    if (stage7.get('schema_version') != STAGE7_SCHEMA or stage7.get('complete') is not True
            or stage7.get('test_payload_access') != 0
            or stage8.get('schema_version') != STAGE8_SCHEMA
            or stage8.get('complete') is not True
            or stage8.get('no_test_payload_access') is not True):
        raise ValueError('Full Stage-7 and Stage-8 TRAIN/VAL reports required')
    canonical = pin['canonical_manifest_sha256']
    root = Path(__file__).resolve().parents[3]
    actor_path = (root / config['stage3']['actor_g2']).resolve()
    if not actor_path.is_relative_to(root / 'artifacts/bc_v1/sidecar'):
        raise ValueError('Actor G2 path escapes expected artifact directory')
    actor_doc = json.loads((actor_path / 'manifest.json').read_text())
    success = json.loads((actor_path / 'SUCCESS.json').read_text())
    actor_fpr = actor_doc.get('manifest_fingerprint')
    if (not isinstance(actor_fpr,str) or len(actor_fpr)!=64
            or actor_doc.get('complete') is not True
            or not actor_doc.get('actor_visibility')
            or actor_doc.get('inventory',{}).get('test') != []
            or actor_doc.get('identity',{}).get('canonical_sha') != canonical
            or actor_doc.get('identity',{}).get('g1_fingerprint') != pin['g1_cache_fingerprint']
            or success.get('manifest_fingerprint') != actor_fpr
            or success.get('TEST_payload_access') != 0):
        raise ValueError('Stage-3 TEST-free actor G2 certification/identity mismatch')
    if any(sum(e['samples'] for e in actor_doc['inventory'][split]) != n for
            split,n in config['expected']['actor_g2'].items()):
        raise ValueError('Actor G2 TRAIN/VALIDATION sample count mismatch')
    if (stage7.get('source_manifest_sha256') != canonical
            or stage8.get('canonical_sha256') != canonical
            or stage7.get('g2_manifest_fingerprint') != actor_fpr
            or stage8.get('g2_manifest_fingerprint') != actor_fpr):
        raise ValueError('Stage-7/8 reports do not match v8 source identity')
    violations = stage8.get('violations')
    if (not isinstance(violations,list) or len(violations) != stage7.get('align_violations')
            or len(violations) != stage8.get('summary',{}).get('total_violations')):
        raise ValueError('Stage-7/8 violation-count mismatch')
    ids = [key(v) for v in violations]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate align-positive exception identity')
    by_split = dict(Counter(x[0] for x in ids))
    for part,n in expected.items():
        if (by_split.get(part,0)!=n or
            stage7.get('counts',{}).get(part,{}).get('align_singleton_violation',0)!=n):
            raise ValueError('Stage-7/8 mismatch with independently pinned v8 counts: '+part)
    rows = [dict(split=s,episode_id=e,transition_index=i,initiator=m,
                 primitive='align_faces', supervised_positive=False,
                 permitted_as_negative=False,
                 treatment='mask_exact_structural_label_in_stage11_and_binary_loss')
            for s,e,i,m in sorted(ids)]
    report = dict(schema_version=SCHEMA,complete=True,
        canonical_sha256=canonical,actor_g2_manifest_fingerprint=actor_fpr,
        expected_counts=expected,observed_counts={k:by_split.get(k,0) for k in expected},
        masked_align_positive_instances=rows,
        mask_policy='mask_unrepresentable_positive_not_negative',
        legacy_exception_restoration=False,runtime_exception=False,
        test_payload_access=0,ready_for_training=False,
        stage11_mask_integration_verified=False,
        note='This disposition is not a Stage-11 training certification.')
    report['fingerprint']=sha(report)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','stage7','stage8','output'):
        p.add_argument('--'+name,required=True,type=Path)
    args=p.parse_args()
    if args.output.exists():
        p.error('Refusing to overwrite existing report')
    config=json.loads(args.config.read_text())
    if config.get('schema_version')!='mssr.bc_v5.stage2_v8_config.v1':
        p.error('Wrong Stage-2 configuration schema')
    result=build(config,json.loads(args.stage7.read_text()),json.loads(args.stage8.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf8') as f:
        f.write(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(masked=len(result['masked_align_positive_instances']),
        ready_for_training=False,report=str(args.output)),sort_keys=True))
if __name__ == '__main__':
    main()
