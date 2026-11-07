"""Matched CPU-only Tier1 component probes; never launches a simulator.

Uses the same eight-module test fixture and real frozen BC weights for both
paths. Timings do not predict GUI/GPU speedup. The fake native view measures
redundant boundary calls, not GPU transfer latency.
"""
import argparse
import json
from pathlib import Path
import statistics
import tempfile
import time
import sys


def benchmark(*, repeats=7):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from drl.MARL.tests.test_bc_v4_runtime_observation import make_bridge
    from drl.IL.tests.bc_v1.test_live_observation import snapshot
    from drl.IL.bc_v1.bc_v1_contract import TaskContext
    from drl.IL.bc_v1.candidates import enumerate_candidates
    from drl.IL.bc_v1.tensorization import _candidate_inputs
    from drl.IL.bc_v1.runtime_static_cache import RuntimeStaticCache
    from drl.MARL.c1_structural_physics_diagnostic import build_frozen_actor,configuration,parse_args
    from drl.MARL.policy import sample_marl_action_bundle
    from drl.MARL.tests.test_c1_tier1_policy import equal
    from drl.MARL.projected_runtime import project_marl_action
    from drl.MARL.smores_runtime_binding import SMORESMARLRuntimeBinding
    from drl.IL.bc_v1.live_policy import smores_claims
    from drl.MARL.c1_viewer_events import CompactEventWriter
    from drl.MARL.isaaclab_articulation import ReadonlyNativeView
    bridge=make_bridge();raw=snapshot(n=8);context=TaskContext('navigation');cache=RuntimeStaticCache()
    actor=build_frozen_actor(bridge,configuration(parse_args(['--preflight','--device','cpu'])))
    def prepare(optimized):
        return bridge.prepare_snapshot(raw,task_context=context,
            **dict(include_privileged=False,static_cache=cache) if optimized else {})
    old,new=prepare(False),prepare(True)
    equal(old.actor_graph,new.actor_graph)
    indices={name:{ref:i for i,ref in enumerate(old.routing_snapshot[name+'_refs'])}
             for name in ('module','connector','resource')}
    def static(optimized):
        graph=old.physical_state.graph
        table=(cache.enumerate if optimized else enumerate_candidates)(graph,bridge.spec,bridge.feature_schema)
        route=dict(old.routing_snapshot)
        return _candidate_inputs(table,bridge.spec,indices,route,static_cache=cache if optimized else None)
    equal(static(False),static(True))
    mask=torch.zeros(len(old.routing_snapshot['resource_candidates'])+len(old.routing_snapshot['structural_bindings']),dtype=torch.bool)
    keys=('records','resource_payloads','structural_payloads','resource_log_std','structural_log_std')
    def infer(optimized):
        item=new if optimized else old
        with torch.no_grad():
            out=actor.forward_all_owners(graph=item.actor_graph,routing=item.routing_snapshot,possible_agents=(),lazy_payloads=optimized)
            action=sample_marl_action_bundle(**{k:out[k] for k in keys},hard_mask=mask,bulk_host_reads=optimized)
        return action
    torch.manual_seed(13);expected=infer(False);rng=torch.random.get_rng_state()
    torch.manual_seed(13);actual=infer(True);equal(expected,actual);assert torch.equal(rng,torch.random.get_rng_state())
    with torch.no_grad():out=actor.forward_all_owners(graph=new.actor_graph,routing=new.routing_snapshot,possible_agents=())
    binding=SMORESMARLRuntimeBinding(adapter=bridge.adapter,claim_resolver=smores_claims)
    def project(optimized):
        batch=dict(new.decoder_batch)
        if not optimized:batch.pop('runtime_decoder_layout',None)
        return project_marl_action(binding=binding,action=expected,prediction_template=out['prediction_template'],
            routing=new.routing_snapshot,batch=batch,normalizers=bridge.normalizers,bulk_host_reads=optimized)
    a,b=project(False),project(True)
    assert a.projected_bundle==b.projected_bundle and a.dropped==b.dropped
    results={}
    def measure(name,fn,*,units='one decision'):
        # Warm each path, then alternate order across repeats to reduce drift.
        for _ in range(2):
            for optimized in (False,True):torch.manual_seed(13);fn(optimized)
        samples={False:[],True:[]}
        for repeat in range(repeats):
            for optimized in ((False,True) if repeat%2==0 else (True,False)):
                torch.manual_seed(13);start=time.perf_counter();fn(optimized)
                samples[optimized].append(time.perf_counter()-start)
        before,after=(statistics.median(samples[k]) for k in (False,True))
        results[name]=dict(before_median_s=before,after_median_s=after,speedup=before/after,
            before_samples_s=samples[False],after_samples_s=samples[True],units=units,equivalent=True)
    measure('context_construction',prepare)
    measure('candidate_catalog_and_layout',static)
    measure('inference_and_sampling',infer)
    measure('projection_empty_policy_bundle',project)
    from dataclasses import replace
    record=next(r for r in out['records'] if '|align_faces|' in r.candidate_id)
    forced_records=tuple(replace(r,presence_logit=torch.tensor(100.)) if r is record
                         else replace(r,presence_logit=torch.tensor(-100.)) for r in out['records'])
    projection_fixture=sample_marl_action_bundle(**{k:(forced_records if k=='records' else out[k]) for k in keys},hard_mask=mask)
    assert int(projection_fixture.presence.sum())==1
    # Only the CPU projection workload uses this deliberately selected member.
    # The real actor benchmark above retains its actual zero-selection bundle.
    policy_action=expected
    expected=projection_fixture
    a,b=project(False),project(True)
    assert a.projected_bundle==b.projected_bundle and a.dropped==b.dropped
    measure('projection_selected_align_fixture',project)
    projection_fixture_outcome=dict(selected=record.candidate_id,accepted=len(a.accepted_candidates),
        dropped=len(a.dropped),note='Explicit CPU unit workload, not a policy decision or live action')
    expected=policy_action
    with tempfile.TemporaryDirectory(prefix='mssr-tier1-cpu-') as folder:
        paths=[Path(folder)/name for name in ('before.jsonl','after.jsonl')]
        rows=[dict(event='ALIGN COMPLETE',timestamp_s=i/240.,goal_id=f'g{i}',claims={'connector:smores_01:TOP':'exclusive'}) for i in range(100)]
        def logging(optimized):
            path=paths[int(optimized)]
            if path.exists():path.unlink()
            if optimized:
                writer=CompactEventWriter(path)
                for row in rows:writer.append(row)
                writer.flush();writer.close()
            else:
                for row in rows:
                    with path.open('a') as handle:handle.write(json.dumps(row,allow_nan=False)+'\n')
        measure('logging_100_events',logging,units='100 identical event records (batch flush)')
        assert [json.loads(s) for s in paths[0].read_text().splitlines()]==[json.loads(s) for s in paths[1].read_text().splitlines()]==rows
    class Array:
        def __init__(self):self.values=np.arange(32,dtype=np.float32)
        def numpy(self):return self.values
    class View:
        def __init__(self):self.reads=0;self.array=Array()
        def get_dof_positions(self):self.reads+=1;return self.array
    view=View();stamp=[0];cached=ReadonlyNativeView(view,lambda:stamp[0])
    def readback(optimized):
        stamp[0]+=1
        for _ in range(128):
            values=(cached if optimized else view).get_dof_positions().numpy()
        return values
    equal(readback(False).tolist(),readback(True).tolist())
    measure('fake_native_readback_128_consumers',readback,units='128 same-tick reads; CPU fake, no transfer latency')
    view.reads=0;readback(False);before_reads=view.reads
    view.reads=0;readback(True);after_reads=view.reads
    results['fake_native_readback_128_consumers'].update(before_native_calls=before_reads,after_native_calls=after_reads)
    assert not any(name.startswith(('isaaclab.','isaacsim.')) for name in sys.modules)
    return dict(schema='mssr.c1.tier1.cpu_benchmark.v1',device='cpu',torch_threads=1,envs=1,modules=8,
        candidates=len(mask),warmup_per_path=2,repeats=repeats,physics_executed=False,training=False,
        selected_candidates=int(expected.presence.sum()),projection_fixture=projection_fixture_outcome,results=results,
        note='Matched CPU fixture only. Logging timing excludes console and uses decision-batch flush; viewer flushes immediate important events. Fake readback speed is Python overhead, not PhysX/CUDA latency. No GUI/GPU end-to-end speedup claim.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repeats',type=int,default=7)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.repeats<3:parser.error('At least3 repeats required')
    result=benchmark(repeats=args.repeats)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{n:v[n] for n in ('before_median_s','after_median_s','speedup')} for k,v in result['results'].items()},indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
