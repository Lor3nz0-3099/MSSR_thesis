"""Explicit immutable G2-v2 -> G2-v3 conversion. TRAIN/VAL only, no TEST reader.

No work on import. CLI defaults to preflight; --run is an explicit full build.
The old sidecar is read-only. No normalization, source rewrite or new features.
"""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import time
from .sidecar import SidecarDataset,SidecarWriter,_hash,_publish_manifest,assert_exact
from .resource_decisions import VERSION,validate_outcome_inputs,load_resource_contract,validate_decision_targets
from .no_structural_prepare import ROOT,file_sha,write_json

SPLITS=('train','validation')
CONFIG=ROOT/'configs/bc_v1/resource_prepare_v3.json'
TENSOR_SCHEMA='mssr.bc_v1.tensor_sample.v3'


def support_counts(sample,contract):
    y=sample['supervision']['resource'];x=sample['inputs']['resource_candidates'];out=Counter()
    for q,valid in zip(y['decision_candidate'].tolist(),y['decision_supervised'].tolist()):
        if not valid:out['unsupervised']+=1;continue
        o=contract.outcomes[int(x['outcome_catalog'][q])]
        out[{0:'INACTIVE',1:'FIXED',2:'PARAMETERIZED'}[o.kind]]+=1
        out[f'{o.module_type_id}/{o.resource_id}/{o.outcome_id}']+=1
    return out


def convert_resource_sidecar(source_root,output,contract,*,label_provider,dataset_factory=SidecarDataset,indices_by_split=None,progress=None):
    source_root,output=Path(source_root),Path(output)
    if output.exists() or output.with_name(output.name+'.building').exists():raise ValueError('Output collision; no overwrite/resume')
    if source_root.resolve()==output.resolve() or source_root.resolve() in output.resolve().parents:raise ValueError('Output overlaps immutable source')
    doc=json.loads((source_root/'manifest.json').read_text())
    if doc.get('resource_decision_schema'):raise ValueError('Expected legacy v2 source')
    code={name:file_sha(Path(__file__).with_name(name+'.py')) for name in ('resource_decisions','smores_resource_labels','resource_prepare','sidecar','batching')}
    semantics={'schema':VERSION,'tensor_schema':TENSOR_SCHEMA,'resource_contract_fingerprint':contract.fingerprint,'authority_identity':getattr(label_provider,'identity','injected'),'code':code}
    identity={**doc['identity'],'resource_contract_fingerprint':contract.fingerprint,
        'tensorizer_fingerprint':_hash({'source':doc['identity']['tensorizer_fingerprint'],**semantics}),
        'storage_fingerprint':_hash({'source':doc['identity']['storage_fingerprint'],**semantics})}
    inventory={s:(doc['inventory'][s] if s in SPLITS else []) for s in doc['inventory']}
    if indices_by_split is not None:
        if set(indices_by_split)!=set(SPLITS):raise ValueError('Tiny build requires TRAIN and VAL selections only')
        inventory={s:[] for s in doc['inventory']}
        for split in SPLITS:
            groups={}
            with dataset_factory(source_root,split,expected_identity=doc['identity']) as data:
                for index in indices_by_split[split]:
                    a=data[index]['routing']['audit'];groups.setdefault(a.episode_id,[]).append(a.transition_index)
            inventory[split]=[dict(episode_id=e,samples=len(ids),transition_indices_fingerprint=_hash(ids)) for e,ids in groups.items()]
    counts={s:Counter() for s in SPLITS};sample_counts={s:0 for s in SPLITS};structural=Counter();start=time.monotonic()
    writer=SidecarWriter(output,identity,inventory)
    try:
        for split in SPLITS:
            with dataset_factory(source_root,split,expected_identity=doc['identity']) as data:
                indices=range(len(data)) if indices_by_split is None else indices_by_split[split]
                for local,index in enumerate(indices):
                    old=data[index]
                    if 'decision_group' not in old['supervision']['structural']:raise ValueError('Missing K2 structural decisions')
                    new=label_provider(old)
                    for k in old['inputs']:
                        if k=='resource_candidates':
                            for name,value in old['inputs'][k].items():assert_exact(new['inputs'][k][name],value)
                        else:assert_exact(new['inputs'][k],old['inputs'][k])
                    assert_exact(new['supervision']['structural'],old['supervision']['structural'])
                    assert_exact(new['routing']['physical_supervision'],old['routing']['physical_supervision'])
                    validate_outcome_inputs(new['inputs']['resource_candidates'],contract,len(new['inputs']['resource_x']))
                    validate_decision_targets(new)
                    counts[split].update(support_counts(new,contract));sample_counts[split]+=1
                    for q in old['supervision']['structural']['candidate'][old['supervision']['structural']['supervised']].tolist():structural[old['routing']['structural_bindings'][q][0].action_id]+=1
                    writer.add(new,split,local)
                    if progress and (local+1)%1000==0:progress(dict(split=split,samples=local+1,elapsed_s=time.monotonic()-start))
        result=writer.finalize()
    finally:writer.close()
    report=dict(schema_version=VERSION,tensor_schema=TENSOR_SCHEMA,resource_contract_fingerprint=contract.fingerprint,
        source_manifest_fingerprint=doc['manifest_fingerprint'],sample_counts=sample_counts,
        support={s:dict(c) for s,c in counts.items()},physical_structural_counts=dict(structural),code_hashes=code,
        TEST_payload_access=0,sealed_test=doc.get('decision_audit',{}).get('sealed_test'),build_seconds=time.monotonic()-start,
        authority_identity=getattr(label_provider,'identity','injected'),
        authority_policy='smores.complete_nav2_snapshot.v1' if hasattr(label_provider,'source_hashes') else 'injected',
        authority_sources=getattr(label_provider,'source_hashes',{}),physical_inputs_preserved=True,structural_supervision_preserved=True)
    result=_publish_manifest(output,dict(result,tensor_sample_schema=TENSOR_SCHEMA,resource_decision_schema=VERSION,
        resource_contract_fingerprint=contract.fingerprint,structural_decision_schema=doc.get('structural_decision_schema','mssr.bc_v1.structural_decisions.v2'),resource_audit=report))
    report['manifest_fingerprint']=result['manifest_fingerprint'];write_json(output/'resource_support.json',report)
    return report


def verify_resource_sidecar(root,contract):
    root=Path(root);doc=json.loads((root/'manifest.json').read_text())
    if (doc.get('resource_decision_schema')!=VERSION or doc.get('tensor_sample_schema')!=TENSOR_SCHEMA
            or doc['identity'].get('resource_contract_fingerprint')!=contract.fingerprint or doc['inventory']['test']):raise ValueError('Incompatible v3 manifest/TEST membership')
    counts={s:Counter() for s in SPLITS};seen={}
    for split in SPLITS:
        with SidecarDataset(root,split,expected_identity=doc['identity']) as data:
            if data.db.execute('PRAGMA integrity_check').fetchone()!=('ok',):raise ValueError('SQLite integrity failure')
            if data.db.execute("SELECT count(*) FROM samples WHERE split='test'").fetchone()[0]:raise ValueError('Forbidden TEST payload')
            for i in range(len(data)):
                row=data[i];x=row['inputs']['resource_candidates'];validate_outcome_inputs(x,contract,len(row['inputs']['resource_x']))
                if row['routing']['resource_contract_fingerprint']!=contract.fingerprint:raise ValueError('Sample contract mismatch')
                validate_decision_targets(row)
                counts[split].update(support_counts(row,contract))
            seen[split]=len(data)
    if {s:dict(c) for s,c in counts.items()}!=doc['resource_audit']['support'] or seen!=doc['resource_audit']['sample_counts']:raise ValueError('Deep support/inventory mismatch')
    report=dict(integrity='PASS',sample_counts=seen,manifest_fingerprint=doc['manifest_fingerprint'],resource_contract_fingerprint=contract.fingerprint,TEST_payload_access=0)
    write_json(root/'verification.json',report);return report


def preflight(config=CONFIG):
    from .bc_v1_contract import load_robot_spec
    cfg=json.loads(Path(config).read_text())
    if cfg.get('schema_version')!='mssr.bc_v1.resource_prepare.v3':raise ValueError('Unsupported prepare configuration schema')
    source=ROOT/cfg['source_sidecar'];out=ROOT/cfg['output']
    if not out.resolve().is_relative_to(ROOT) or source.resolve()==out.resolve():raise ValueError('Unsafe output path')
    spec=load_robot_spec(ROOT/'configs/bc_v1/smores_ep.robot.json');contract=load_resource_contract(ROOT/cfg['resource_contract'],spec)
    if contract.fingerprint!=cfg['resource_contract_fingerprint']:raise ValueError('Resource contract pin mismatch')
    doc=json.loads((source/'manifest.json').read_text())
    if doc['manifest_fingerprint']!=cfg['source_manifest_fingerprint']:raise ValueError('Frozen G2-v2 identity mismatch')
    if doc['identity']!=cfg['source_identity']:raise ValueError('Frozen physical identity mismatch')
    for path,sha in cfg.get('code_pins',{}).items():
        if file_sha(ROOT/path)!=sha:raise ValueError('Build code identity mismatch: '+path)
    for path,sha in cfg['file_pins'].items():
        if file_sha(ROOT/path)!=sha:raise ValueError('Frozen upstream file mismatch: '+path)
    return cfg,source,out,contract,doc


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=CONFIG)
    p.add_argument('--run',action='store_true');p.add_argument('--verify',action='store_true')
    args=p.parse_args(argv);cfg,source,out,c,doc=preflight(args.config)
    if args.verify:report=verify_resource_sidecar(out,c)
    elif args.run:
        from .smores_resource_labels import SmoresResourceLabelProvider
        allowed={e['episode_id'] for s in SPLITS for e in doc['inventory'][s]}
        provider=SmoresResourceLabelProvider(ROOT,c,allowed_episodes=allowed,source_certificates=cfg['authority_sources'])
        report=convert_resource_sidecar(source,out,c,label_provider=provider,progress=lambda row:print(json.dumps(row),flush=True))
        if report['sample_counts']!=cfg['expected_counts']:raise ValueError('Full TRAIN/VAL counts mismatch')
        if report['physical_structural_counts']!=cfg['expected_structural_counts']:raise ValueError('Preserved K2 structural counts mismatch')
        report={**report,'verification':verify_resource_sidecar(out,c)}
        preflight(args.config)
        write_json(out/'SUCCESS.json',report)
    else:
        if out.exists() or out.with_name(out.name+'.building').exists():raise ValueError('Output already exists')
        report=dict(mode='DRY_RUN',output=str(out),source=str(source),resource_contract_fingerprint=c.fingerprint,expected_counts=cfg['expected_counts'],TEST_payload_access=0)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
