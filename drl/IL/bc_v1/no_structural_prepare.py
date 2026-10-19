"""Explicit, lossless G2-v1 -> decision-v2 conversion; TRAIN/VAL only.

No invocation at import. The physical tensors/F2 and positive labels are reused.
TEST membership is recorded from manifests, never opened or materialized.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import time

from .sidecar import SidecarDataset,SidecarWriter,_hash,_publish_manifest
from .structural_decisions import VERSION,add_no_structural_supervision

ROOT=Path(__file__).resolve().parents[3]
MATERIALIZED_SPLITS=('train','validation')
DEFAULT_CONFIG=ROOT/'configs/bc_v1/no_structural_v2.json'


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def frozen_identity():
    original=Path('/home/lorenzo/MSSR_thesis')
    paths={'canonical':original/'datasets/expert_v1_canonical_v6/manifest.json',
        'f1':ROOT/'artifacts/bc_v1/splits/f1_v1/split.json',
        'f2':ROOT/'artifacts/bc_v1/normalization/f2_v1_optimized/normalizers.json',
        'k1':original/'artifacts/bc_v1/checkpoints/k1_baseline_v1/best_checkpoint.pt'}
    expected={'canonical':'31765272b6ecd7506c54b25e214b75d7b1b28691803ddfc286dafcce8b4ec562',
        'f1':'f6759e6af89d732078ac656133d40236e1fd3cec8b43c92708907ed90ac965c7',
        'f2':'ea8efbd9062acb8625e0618bcc5f86b975f20839bf3132a684a626c744e52003',
        'k1':'35737c025e7fa53492cb12e59ccf4c4f3f6210a6d6f0882c245bf1847975d06d'}
    actual={k:file_sha(p) for k,p in paths.items()}
    if actual!=expected:raise ValueError('Frozen Canonical/F1/F2/K1 identity mismatch')
    return actual


def convert_sidecar(source_root,output,*,source_manifest=None,dataset_factory=SidecarDataset,progress=None,expected_positive_counts=None):
    source_root,output=Path(source_root),Path(output)
    if source_root.resolve()==output.resolve() or source_root.resolve() in output.resolve().parents:
        raise ValueError('Output must be separate from immutable source')
    doc=source_manifest or json.loads((source_root/'manifest.json').read_text())
    if doc.get('structural_decision_schema'):raise ValueError('Expected a v1 source, not already converted')
    code_hashes={name:file_sha(Path(__file__).with_name(name+'.py')) for name in ('structural_decisions','batching','no_structural_prepare')}
    identity={**doc['identity'],'tensorizer_fingerprint':_hash({'source':doc['identity']['tensorizer_fingerprint'],
        'decision_schema':VERSION,'code':code_hashes}),
        'storage_fingerprint':_hash({'source':doc['identity']['storage_fingerprint'],'decision_schema':VERSION,'code':code_hashes})}
    inventory={s:(doc['inventory'][s] if s in MATERIALIZED_SPLITS else []) for s in doc['inventory']}
    counts=Counter();ops=Counter();breakdowns={k:defaultdict(Counter) for k in ('task','morphology','source','split')}
    start=time.monotonic();writer=SidecarWriter(output,identity,inventory)
    try:
        for split in MATERIALIZED_SPLITS:
            with dataset_factory(source_root,split,expected_identity=doc['identity']) as data:
                for index in range(len(data)):
                    old=data[index];new=add_no_structural_supervision(old)
                    y=new['supervision']['structural'];positive=int((y['decision_candidate']>=0).sum());groups=len(y['decision_group'])
                    if positive!=int(old['supervision']['structural']['supervised'].sum()):raise ValueError('Positive structural labels lost')
                    if new['inputs'] is not old['inputs'] or new['supervision']['resource'] is not old['supervision']['resource']:
                        raise ValueError('Physical inputs or resource targets replaced')
                    counts.update(samples=1,groups=groups,physical=positive,no_structural=groups-positive)
                    vocabulary=new['routing']['task_vocabulary'];x=new['inputs'];audit=new['routing']['audit']
                    task=vocabulary[int(x['task_one_hot'][0].argmax())] if bool(x['task_valid'][0]) else 'UNKNOWN'
                    labels={'task':task,'morphology':audit.morphology_stratum or 'UNKNOWN','source':audit.provenance or 'UNKNOWN','split':split}
                    for dimension,label in labels.items():breakdowns[dimension][label].update(samples=1,groups=groups,physical=positive,no_structural=groups-positive)
                    for q in y['decision_candidate'][y['decision_candidate']>=0].tolist():ops[new['routing']['structural_bindings'][q][0].action_id]+=1
                    writer.add(new,split,index)
                    if progress and (index+1)%1000==0:progress(dict(split=split,split_samples=index+1,**counts,elapsed_s=time.monotonic()-start))
        if expected_positive_counts is not None and dict(ops)!=expected_positive_counts:
            raise ValueError('Positive target counts differ from certified TRAIN+VAL baseline')
        result=writer.finalize()
    finally:writer.close()
    report={'schema_version':VERSION,'counts':dict(counts),'physical_operations':dict(ops),
        'breakdowns':{k:{name:dict(v) for name,v in sorted(rows.items())} for k,rows in breakdowns.items()},
        'sealed_test':doc['split_counts']['test'],'source_declared_samples':doc['sample_count'],
        'source_manifest_fingerprint':doc['manifest_fingerprint'],'code_hashes':code_hashes,
        'TEST_payload_access':0,'preserved_inputs':True,'preserved_resource_targets':True,'preserved_positive_targets':True,
        'build_seconds':time.monotonic()-start}
    result=_publish_manifest(output,dict(result,structural_decision_schema=VERSION,decision_audit=report))
    report.update(artifact_fingerprint=result['manifest_fingerprint'],identity=identity)
    write_json(output/'decision_audit.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--verify',action='store_true',help='Verify published metadata/storage without reading TEST')
    args=parser.parse_args();cfg=json.loads(args.config.read_text());before=frozen_identity()
    source=ROOT/cfg['source_sidecar'];output=ROOT/cfg['output']
    if not output.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Output escapes approved worktree')
    src=json.loads((source/'manifest.json').read_text())
    if src['manifest_fingerprint']!=cfg['source_manifest_fingerprint']:raise ValueError('Source sidecar identity changed')
    expected={k:src['identity'][k] for k in ('canonical_sha','f1_sha','g1_fingerprint','f2_fingerprint','normalizer_sha','layout_fingerprint')}
    if expected!=cfg['source_pins']:raise ValueError('Frozen upstream identity changed')
    if args.verify:
        doc=json.loads((output/'manifest.json').read_text());report=doc['decision_audit']
        if doc['inventory']['test'] or doc['structural_decision_schema']!=VERSION:raise ValueError('TEST must remain sealed')
        with SidecarDataset(output,'train',expected_identity=doc['identity']) as data:
            if data.db.execute('PRAGMA integrity_check').fetchone()!=('ok',):raise ValueError('Storage integrity failure')
            if data.db.execute("SELECT count(*) FROM samples WHERE split='test'").fetchone()[0]:raise ValueError('Unexpected TEST payload')
    else:
        report=convert_sidecar(source,output,source_manifest=src,progress=lambda r:print(json.dumps(r),flush=True),
            expected_positive_counts=cfg['expected_train_validation_positive_counts'])
    if before!=frozen_identity():raise ValueError('Frozen source changed during operation')
    # Certified global baseline is inherited, not presented as measured TEST data.
    expected_positive=cfg['expected_train_validation_positive_counts']
    if report['physical_operations']!=expected_positive:raise ValueError('Positive target counts differ from certified TRAIN+VAL baseline')
    write_json(output/'frozen_integrity.json',{'before':before,'after':before,'TEST_payload_access':0})
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
