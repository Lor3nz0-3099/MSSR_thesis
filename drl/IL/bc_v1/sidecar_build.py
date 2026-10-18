"""Reproducible P0 build/pilot/verify/benchmark tool, independent of training.

python -m drl.IL.bc_v1.sidecar_build --help
Frozen sources are read only. Each episode is opened once per build; only its
existing G2 tensorizer determines content. Partial builds require explicit resume.
"""
import argparse
import gc
import json
from pathlib import Path
import random
import resource
import time

import torch

from . import dataset as g1,normalization as f2,tensorization as g2
from .bc_v1_contract import load_robot_spec,robot_spec_fingerprint
from .feature_schema import load_feature_schema,feature_schema_fingerprint
from .sidecar import SidecarWriter,SidecarDataset,verify_sidecar,SPLITS,_hash

ROOT=Path(__file__).resolve().parents[3]
DEFAULT_CONFIG=ROOT/'configs/bc_v1/sidecar_v1.json'


def load_sources(config=DEFAULT_CONFIG):
    cfg=json.loads(Path(config).read_text())
    if cfg['schema_version']!='mssr.bc_v1.sidecar_build_config.v1':raise ValueError('Invalid sidecar build config')
    paths={k:(ROOT/v).resolve() for k,v in cfg['paths'].items()}
    pins=cfg['pins']
    for key,pathkey in [('canonical_sha','canonical_manifest'),('f1_sha','split'),('normalizer_sha','normalizers')]:
        if g1.file_hash(paths[pathkey])!=pins[key]:raise ValueError('Frozen source SHA mismatch: '+key)
    manifest=g1.read_cache_manifest(paths['g1']);split=json.loads(paths['split'].read_text());ids=manifest['identity']
    if manifest['cache_fingerprint']!=pins['g1_fingerprint']:raise ValueError('Frozen G1 mismatch')
    if manifest['splits']!=split['splits'] or ids['split_fingerprint']!=split['manifest_fingerprint']:
        raise ValueError('Frozen F1 membership mismatch')
    spec=load_robot_spec(paths['robot_spec']);schema=load_feature_schema(paths['feature_schema'])
    policy=f2.load_normalization_policy(paths['normalization_policy'],schema)
    actual={'robot_spec_fingerprint':robot_spec_fingerprint(spec),'feature_schema_fingerprint':feature_schema_fingerprint(schema),
        'normalization_policy_fingerprint':f2.policy_fingerprint(policy,schema)}
    if any(ids[k]!=v for k,v in actual.items()):raise ValueError('Frozen G1 spec/schema/policy mismatch')
    expected={'split_fingerprint':ids['split_fingerprint'],'dataset_manifest_sha256':pins['canonical_sha'],
        'train_membership_fingerprint':_hash(sorted(manifest['splits']['train'])),
        'robot_spec_fingerprint':actual['robot_spec_fingerprint'],'feature_schema_fingerprint':actual['feature_schema_fingerprint'],
        'policy_fingerprint':actual['normalization_policy_fingerprint'],'complete_train_coverage':True}
    normalizers=f2.read_normalizers(paths['normalizers'],expected_metadata=expected)
    layout=g2.load_tensor_layout(paths['tensor_layout'])
    if f2.normalizer_fingerprint(normalizers)!=pins['f2_fingerprint'] or layout['fingerprint']!=pins['layout_fingerprint']:
        raise ValueError('Frozen F2/G2 identity mismatch')
    identity={**pins,'split_fingerprint':ids['split_fingerprint'],
        'feature_schema_fingerprint':actual['feature_schema_fingerprint'],'robot_spec_fingerprint':actual['robot_spec_fingerprint'],
        'tensorizer_fingerprint':g1.file_hash(Path(g2.__file__)),
        'storage_fingerprint':_hash({name:g1.file_hash(Path(__file__).with_name(name+'.py')) for name in ('sidecar','sidecar_build')})}
    inventory={part:[{'episode_id':eid,'samples':manifest['episodes'][eid]['samples'],
        'transition_indices_fingerprint':manifest['episodes'][eid]['transition_indices_fingerprint']}
        for eid in manifest['splits'][part]] for part in SPLITS}
    return {'cache_root':paths['g1'],'manifest':manifest,'identity':identity,'inventory':inventory,
        'kwargs':dict(spec=spec,feature_schema=schema,normalization_policy=policy,normalizers=normalizers,tensor_layout=layout),
        'paths':paths,'config':cfg}


def build_sidecar(sources,output,*,resume=False,episode_reader=None,progress=None,workers=1):
    if type(workers) is not int or not 1<=workers<=2:raise ValueError('Builder permits one or two bounded episode workers')
    if workers>1 and episode_reader is not None:raise ValueError('Injected readers require serial build')
    output=Path(output).resolve();cache=Path(sources['cache_root']).resolve()
    for upstream in [cache,*[p.parent for k,p in sources.get('paths',{}).items() if k in ('canonical_manifest','split','normalizers')]]:
        if output.is_relative_to(upstream) or upstream.is_relative_to(output):raise ValueError('Sidecar output overlaps frozen source')
    reader=episode_reader or g1._load_episode
    kwargs=dict(sources['kwargs'])
    if 'tensor_layout' not in kwargs:kwargs['tensor_layout']=g2.load_tensor_layout(ROOT/'configs/bc_v1/tensor_layout.v1.json')
    writer=SidecarWriter(output,sources['identity'],sources['inventory'],resume=resume)
    start=time.perf_counter();completed=0;episodes=0
    try:
        if writer.publication_ready is not None:
            return {**writer.finalize(),'build_elapsed_s':time.perf_counter()-start}
        if workers>1 and any(0<writer.db.execute('SELECT COUNT(*) FROM samples WHERE episode=?',
                (entry['episode_id'],)).fetchone()[0]<entry['samples'] for entries in sources['inventory'].values() for entry in entries):
            # A serial partial episode cannot be reimported wholesale. Reuse its
            # committed rows and tensorize only missing rows via the same path.
            workers=1
            if progress:progress({'resume_strategy':'serial_missing_rows'})
        if workers>1:
            _parallel_episodes(sources,writer,workers,progress)
            return {**writer.finalize(),'build_elapsed_s':time.perf_counter()-start}
        for split,entries in sources['inventory'].items():
            offset=0
            for entry in entries:
                eid=entry['episode_id'];count=entry['samples']
                if all(writer.has(split,i) for i in range(offset,offset+count)):
                    completed+=count;offset+=count;episodes+=1;continue
                samples=reader(cache,eid,sources['manifest']['episodes'][eid])
                if len(samples)!=count:raise ValueError('Incomplete episode source '+eid)
                for local,sample in enumerate(samples):
                    if not writer.has(split,offset+local):
                        tensor=g2.tensorize_policy_sample(sample,**kwargs);writer.add(tensor,split,offset+local)
                    completed+=1
                    if progress and completed%1000==0:
                        progress({'episodes':episodes,'samples':completed,'elapsed_s':time.perf_counter()-start,
                            'storage_bytes':sum(p.stat().st_size for p in writer.stage.glob('data.sqlite3*'))})
                del samples;gc.collect();episodes+=1;offset+=count
                if progress:progress({'episodes':episodes,'samples':completed,'elapsed_s':time.perf_counter()-start})
        _clean_completed_workers(sources,writer)
        raw={'dynamic':writer.raw_dynamic_bytes,'unique_templates':writer.raw_template_bytes}
        doc=writer.finalize()
        return {**doc,'build_elapsed_s':time.perf_counter()-start,'raw_bytes':raw}
    finally:writer.close()


def _clean_completed_workers(sources,writer):
    import shutil
    directory=writer.stage/'workers'
    if not directory.exists():return
    for entries in sources['inventory'].values():
        for entry in entries:
            count=writer.db.execute('SELECT COUNT(*) FROM samples WHERE episode=?',(entry['episode_id'],)).fetchone()[0]
            if count!=entry['samples']:continue
            path=directory/_hash(entry['episode_id'])
            for owned in (path,path.with_name(path.name+'.building')):
                if owned.exists():shutil.rmtree(owned)
    if not any(directory.iterdir()):directory.rmdir()


def _episode_job(sources,split,entry,path):
    torch.set_num_threads(1)
    subset={**sources,'inventory':{s:([entry] if s==split else []) for s in SPLITS}}
    if Path(path).exists():
        with SidecarDataset(path,split,expected_identity=sources['identity']) as data:
            if data.manifest['inventory']!=subset['inventory']:raise ValueError('Worker resume membership mismatch')
        return
    resume=Path(path).with_name(Path(path).name+'.building').exists()
    build_sidecar(subset,path,resume=resume,
        progress=lambda d:print(json.dumps({'worker_episode':entry['episode_id'],**d}),flush=True))


def _parallel_episodes(sources,writer,workers,progress):
    """Two spawn workers maximum; memory admission uses the measured G1 ratio.

    Workers exit after each episode so Python arenas cannot accumulate across
    large episodes. They return only a disk artifact, never episode-sized RAM
    results. The parent merges indexed records transactionally, then removes
    only that owned worker artifact. No second G1 pass or tensor reconstruction.
    """
    import multiprocessing as mp
    import shutil
    _clean_completed_workers(sources,writer)
    gib=1024**3
    available=next((int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
        if line.startswith('MemAvailable:')),4*gib)
    budget=min(18*gib,available-3*gib)
    if budget<gib:raise ValueError('Insufficient available RAM for episode builder')
    jobs=[];done=0;completed=0
    for split,entries in sources['inventory'].items():
        offset=0
        for e in entries:
            if all(writer.has(split,i) for i in range(offset,offset+e['samples'])):
                done+=1;completed+=e['samples']
            else:
                path=Path(sources['cache_root'])/sources['manifest']['episodes'][e['episode_id']]['cache_file']
                weight=int(path.stat().st_size*9.5)+512*1024**2
                if weight>budget:raise ValueError('Episode exceeds safe memory budget: '+e['episode_id'])
                jobs.append((weight,split,e))
            offset+=e['samples']
    jobs.sort(key=lambda j:-j[0]);active={};directory=writer.stage/'workers';directory.mkdir(exist_ok=True)
    start=time.perf_counter()
    with mp.get_context('spawn').Pool(processes=workers,maxtasksperchild=1) as pool:
        while jobs or active:
            while len(active)<workers:
                used=sum(v[0] for v in active.values())
                position=next((i for i,j in enumerate(jobs) if used+j[0]<=budget),None)
                if position is None:break
                weight,split,entry=jobs.pop(position);path=directory/_hash(entry['episode_id'])
                future=pool.apply_async(_episode_job,(sources,split,entry,path))
                active[future]=(weight,split,entry,path)
            deadline=time.monotonic()+30
            finished=[f for f in active if f.ready()]
            while not finished and time.monotonic()<deadline:
                time.sleep(.2);finished=[f for f in active if f.ready()]
            for future in finished:
                _,split,entry,path=active.pop(future);future.get()
                writer.import_episode(path,split,entry['episode_id']);shutil.rmtree(path)
                done+=1;completed+=entry['samples']
            if progress:progress({'episodes':done,'samples':completed,'elapsed_s':time.perf_counter()-start,
                'active_episodes':[v[2]['episode_id'] for v in active.values()],'memory_budget_bytes':budget})
    directory.rmdir()


def pilot_sources(sources):
    """Bounded representative selection, including every requested structural row.

    Only six selected TRAIN episode files are opened. Retain selected samples,
    not whole episodes, after extracting the bounded pilot. No derived G1 cache.
    """
    from dataclasses import replace
    selected={};inventory={part:[] for part in SPLITS}
    choices=[('expert:assembly:seed-006307',None),('expert:gap:seed-004303',None),
        ('expert:rc_car:seed-005100',None),('expert:stairs:seed-003102',32),
        ('teleop-1f8b406b56f74377834cd863ebccf696',64),('teleop-0e4e34fe3a5f4ed1b6a86642acf4fac8',64)]
    for eid,limit in choices:
        if eid not in sources['manifest']['splits']['train']:raise ValueError('Pilot must use TRAIN only')
        rows=g1._load_episode(sources['cache_root'],eid,sources['manifest']['episodes'][eid])
        indices=set(range(len(rows))) if limit is None else {i*(len(rows)-1)//(limit-1) for i in range(limit)}
        indices.update(i for i,r in enumerate(rows) if any(t.supervised for t in r.supervision.structural_targets))
        selected[eid]=tuple(rows[i] for i in sorted(indices));del rows;gc.collect()
        inventory['train'].append({'episode_id':eid,'samples':len(selected[eid]),
            'transition_indices_fingerprint':_hash([s.audit.transition_index for s in selected[eid]])})
    return {**sources,'inventory':inventory},selected


def benchmark_requests(sources,sidecar_root,requests):
    """Exact same (episode, local G1 row) lists as the frozen direct benchmark."""
    from .batching import collate_policy_samples
    results={}
    with SidecarDataset(sidecar_root,'train',expected_identity=sources['identity']) as data:
        # Full source row->Canonical mapping is supplied separately by the caller
        # via exact logical keys. This routine never rebuilds semantic domains.
        for name,keys in requests.items():
            samples=[];latencies=[];start=time.perf_counter()
            for eid,transition in keys:
                t=time.perf_counter();samples.append(data.get(eid,transition));latencies.append(time.perf_counter()-t)
            fetch=time.perf_counter()-start;t=time.perf_counter();collate_policy_samples(samples);collate=time.perf_counter()-t
            ordered=sorted(latencies)
            results[name]={'samples':len(keys),'fetch_s':fetch,'collate_s':collate,'total_s':fetch+collate,
                'samples_per_s':len(keys)/fetch,'p50_s':ordered[len(ordered)//2],'p95_s':ordered[int((len(ordered)-1)*.95)]}
            del samples;gc.collect()
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('pilot','build','verify','benchmark'))
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--workers',type=int,default=2,help='build only: one/two memory-admitted episode processes')
    parser.add_argument('--requests',type=Path,help='JSON mapping benchmark names to [episode,Canonical transition] requests')
    args=parser.parse_args();torch.set_num_threads(1);sources=load_sources(args.config)
    output=args.output or sources['paths']['output']
    emit=lambda v:print(json.dumps(v,sort_keys=True),flush=True)
    if args.command=='pilot':
        output=args.output or output.with_name(output.name+'_pilot')
        pilot,selected=pilot_sources(sources)
        emit(build_sidecar(pilot,output,resume=args.resume,episode_reader=lambda root,eid,entry:selected[eid],progress=emit))
    elif args.command=='build':emit(build_sidecar(sources,output,resume=args.resume,progress=emit,workers=args.workers))
    elif args.command=='verify':emit(verify_sidecar(output,expected_identity=sources['identity']))
    elif args.command=='benchmark':
        if args.requests is None:parser.error('benchmark requires --requests')
        emit(benchmark_requests(sources,output,json.loads(args.requests.read_text())))


if __name__=='__main__':main()
