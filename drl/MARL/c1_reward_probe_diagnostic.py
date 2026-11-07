"""Bounded user-run C1 reward/probe interface verification, no optimizer/Expert.

This tests native instrumentation, not autonomous assembly success. Scientific
success cases are covered by synthetic unit contracts; physical task utility
must be established from completed runs before training readiness is claimed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

from drl.MARL.structural_conflict_diagnostic import ROOT,TASK_SEEDS,write_json,emit
from drl.MARL.c1_native_runtime import sanitized_record

OUTPUT=ROOT/'logs/isaaclab_c1_native/redesign/c1_reward_probe'


def summarize(output):
    results={};episodes=[]
    for task in TASK_SEEDS:
        folder=output/task
        if (folder/'result.json').exists():results[task]=json.loads((folder/'result.json').read_text())
        path=folder/'discovery/episodes.jsonl'
        if path.exists():episodes.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    with (output/'episodes.jsonl').open('w') as handle:
        for record in episodes:emit(handle,record)
    from drl.MARL.c1_discovery import classify_records_offline
    write_json(output/'offline_topology_classification.json',classify_records_offline(episodes,ROOT))
    summary=dict(status='native_interface_check_complete_pending_analysis' if len(results)==4 and all(v['status']=='pass' for v in results.values()) else 'awaiting_offline_verification',
        results=results,episodes=len(episodes),assembly_validity_observed=sum(r['assembly_valid_timestamp'] is not None for r in episodes) if episodes else None,
        functional_probe_observed=sum(r['probe_success_timestamp'] is not None for r in episodes) if episodes else None,C1_successes=sum(r['success'] for r in episodes) if episodes else None,
        physics_verification_executed=bool(results),
        optimizer_updates=0,expert_actions=0,training_ready=False,task5_started=False,
        note='A short loose-module run need not complete assembly. Interface completion is not empirical C1 success or training readiness.')
    write_json(output/'summary.json',summary)
    return summary


def run(args):
    folder=args.output/args.task;folder.mkdir(parents=True,exist_ok=False)
    write_json(folder/'arguments.json',vars(args))
    sources=sorted((ROOT/'drl/MARL').glob('*.py'))+sorted((ROOT/'scripts/smores_ep/src/smores_ep/isaac').glob('*.py'))
    sources+=[ROOT/'configs/marl/c1_assembly_probe.json',ROOT/'configs/bc_v1/runtime_inference_v1.json']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    write_json(folder/'source_sha256.json',hashes)
    env=application=None;start=time.monotonic()
    result=dict(status='fail',task=args.task,seed=TASK_SEEDS[args.task],num_envs=1,initialization_mode='loose_modules',
        optimizer_updates=0,expert_actions=0,decisions_completed=0)
    try:
        from isaaclab.app import AppLauncher
        application=AppLauncher(args).app
        import torch
        from drl.MARL.c1_runner import build_observation_bridge,build_models,CHECKPOINT,BC_SHA256
        from drl.MARL.isaaclab_c1_env import MSSRC1IsaacLabEnv,make_c1_isaaclab_env_cfg
        from drl.MARL.isaaclab_c1_runner import seed_native_training
        from drl.MARL.policy import sample_marl_action_bundle
        seed_native_training(args.global_seed)
        owners=tuple(f'smores_{i:02d}' for i in range(1,9))
        cfg=make_c1_isaaclab_env_cfg(possible_agents=owners,num_envs=1,device=args.device)
        cfg.initialization_mode='loose_modules';cfg.domains=(args.task,);cfg.seeds=(TASK_SEEDS[args.task],)
        cfg.seed=args.global_seed;cfg.resample_train=False
        cfg.reward_config_path=str(ROOT/'logs/appo_smoke/calibration_validated_train/reward_config.initial.json')
        cfg.c1_evidence_directory=str(folder/'discovery')
        env=MSSRC1IsaacLabEnv(cfg);env.reset()
        bridge=build_observation_bridge(ROOT)
        actor,_=build_models(SimpleNamespace(repository_root=ROOT,checkpoint=ROOT/CHECKPOINT,device=args.device),bridge)
        native=env.initialize_policy_bridge(observation_bridge=bridge)
        def digest():
            h=hashlib.sha256()
            for key,value in sorted(actor.state_dict().items()):h.update(key.encode());h.update(value.detach().cpu().numpy().tobytes())
            return h.hexdigest()
        before=digest()
        with (folder/'physical_states.jsonl').open('w',buffering=1) as states:
            for decision_step in range(args.steps):
                context,=native.provider.current_action_contexts()
                with torch.no_grad():
                    out=actor.forward_all_owners(graph=context.actor_graph,routing=context.routing_snapshot,possible_agents=owners)
                    action=sample_marl_action_bundle(**{k:out[k] for k in ('records','resource_payloads','structural_payloads','resource_log_std','structural_log_std')},hard_mask=context.hard_mask)
                    presence=action.presence.clone()
                    carrier=native.provider.encode_env_actions([SimpleNamespace(action=action,action_set_prediction_template=out['prediction_template'])])
                    if env.staged[0][1].projection.raw_sampled_action is not action:raise AssertionError('Raw policy action lost')
                    _,reward,*_=env.step(carrier)
                    if not all(torch.isfinite(v).all() for v in reward.values()):raise AssertionError('Nonfinite C1 reward')
                    if not torch.equal(presence,action.presence):raise AssertionError('Evaluator changed sampled action')
                    emit(states,sanitized_record(dict(decision_step=decision_step,generation=env.c1_runtime.recent_transitions[-1]['generation'],
                        before=env._pre_raw[0],after=env._post_raw[0],evaluation=env.c1_runtime.recent_transitions[-1])))
                native.provider.current_critic_graphs();result['decisions_completed']+=1
        env.c1_runtime.flush_open_episodes('bounded_diagnostic_end')
        failures=[r for r in env.c1_episode_records if r['terminated'] and not r['success']]
        result.update(status='pass' if not failures else 'fail',actor_unchanged=before==digest(),
            bc_sha256=BC_SHA256,settings_sha256=env.c1_runtime.settings.sha256,
            c1_successes=sum(r['success'] for r in env.c1_episode_records),physical_failures=[r['success_failure_reason'] for r in failures],
            expert_assembly_records=len(env.assembly_records),reward_finite=True)
        if not result['actor_unchanged'] or env.assembly_records:raise AssertionError('Actor changed or Expert entered loop')
    except Exception as error:result.update(status='fail',error=str(error),traceback=traceback.format_exc())
    finally:
        result['wall_seconds']=time.monotonic()-start
        result['sources_unchanged']=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
        if not result['sources_unchanged']:result['status']='fail'
        write_json(folder/'result.json',result)
        print('C1_REWARD_PROBE_RESULT',json.dumps(result),flush=True)
        if env is not None:env.close()
        if application is not None:application.close()
    return 0 if result['status']=='pass' else 1


def main():
    import sys
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=tuple(TASK_SEEDS));parser.add_argument('--steps',type=int,default=32)
    parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--global-seed',type=int,default=20261029)
    parser.add_argument('--summarize',action='store_true')
    if '--summarize' not in sys.argv:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    if args.summarize:
        args.output.mkdir(parents=True,exist_ok=True)
        result=summarize(args.output);print(json.dumps(result));return 0 if len(result['results'])==4 and all(r['status']=='pass' for r in result['results'].values()) else 1
    if args.task is None or not 1<=args.steps<=120:parser.error('Requires --task and 1..120 decisions; no long run')
    return run(args)


if __name__=='__main__':raise SystemExit(main())
