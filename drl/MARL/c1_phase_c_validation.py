"""User-run bounded native Phase C validation; --preflight never imports Isaac.

Frozen MAPPO actor only. Training provenance is checked against its ORIGINAL
reward/probe hashes; the intentionally changed evaluation contract is recorded
separately. No learner, critic, optimizer, memory or backward is constructed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
CHECKPOINT='logs/isaaclab_c1_native/redesign/c1_guided_resume_u13_20261105_113402/c1_resume_u000078_v000067.pt'


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=('navigation','stairs','gap','manipulation'),default='navigation')
    p.add_argument('--mode',choices=('guided','discovery'),default='guided')
    p.add_argument('--seed',type=int,default=7033)
    p.add_argument('--global-seed',type=int,default=3103)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--checkpoint',type=Path,default=ROOT/CHECKPOINT)
    p.add_argument('--episode-seconds',type=float,default=120.)
    p.add_argument('--steps',type=int,default=3600)
    p.add_argument('--max-wall-seconds',type=float,default=900.)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preflight',action='store_true')
    a=p.parse_args(argv)
    if not 0<a.steps<=5400:p.error('--steps requires1..5400')
    if not math.isfinite(a.episode_seconds) or not 0<a.episode_seconds<=180:p.error('invalid episode duration')
    if not math.isfinite(a.max_wall_seconds) or not 0<a.max_wall_seconds<=1800:p.error('invalid wall limit')
    if not 0<=a.global_seed<2**32:p.error('invalid global seed')
    return a


def preflight(args):
    from drl.MARL.c1_structural_physics_diagnostic import configuration, source_hashes
    from drl.MARL.c1_viewer_policy import inspect_policy_checkpoint
    from drl.MARL.c1_config import C1Settings
    from drl.MARL.c1_train_sampling import catalog_seeds
    cfg=configuration(args)
    if args.output.exists():raise ValueError('Output must be a NEW nonexistent directory')
    if args.seed not in catalog_seeds(args.task,'train'):raise ValueError('Seed outside the selected TRAIN catalog')
    cfg['bc_checkpoint']=cfg['checkpoint']
    if hashlib.sha256(Path(cfg['bc_checkpoint']).read_bytes()).hexdigest()!=cfg['expected_bc_sha256']:
        raise ValueError('Original BC architecture reference hash mismatch')
    cfg['probe_config']=str(ROOT/'configs/marl/c1_assembly_probe_guided_v1.json')
    cfg.update(inspect_policy_checkpoint(args.checkpoint,cfg))
    profile=C1Settings.load(ROOT/f'configs/marl/c1_{args.mode}_phase_c_v1.json')
    cfg.update(schema='mssr.c1.phase_c_validation.v1',num_envs=1,tasks=[args.task],
        initial_train_seed=args.seed,headless=True,rendering=False,
        episode_mode=args.mode,episode_seconds=args.episode_seconds,
        runtime_probe_config=profile.path,runtime_probe_sha256=profile.sha256,
        source_sha256=source_hashes(),structural_scheduler='event',structural_fallback_hz=2.,
        aggregate_environment_seconds=args.steps/30.,
        actor_only=True,training_provenance_probe_config=cfg['probe_config'],
        note='Intentional Phase C evaluation contract change. Original checkpoint loader and provenance checks preserved.')
    from drl.MARL.c1_guided_feedback import DOMAIN_TARGETS
    target=ROOT/'mssr_ws/src/mssr_expert/config'/DOMAIN_TARGETS[args.task]
    cfg.update(target_graph_path=str(target),target_graph_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        target_role='external guided evaluator' if args.mode=='guided' else 'unused by discovery evaluator')
    return cfg


def run(args):
    config=preflight(args)
    folder=args.output.resolve();folder.mkdir(parents=True,exist_ok=False)
    from drl.MARL.c1_structural_physics_diagnostic import write_json,actor_digest
    from drl.MARL.c1_gui_rc_car_180s import native_log,ActorOnlyObservationBridge,sample_contexts
    from drl.MARL.c1_viewer_policy import build_viewer_actor
    write_json(folder/'run_config.json',config)
    env=app=actor=None;before=None;started=time.perf_counter()
    summary=dict(status='error',optimizer_updates=0,backward_calls=0,learner_constructed=False,
        optimizer_constructed=False,critic_model_constructed=False,training_memory_constructed=False,
        expert_actions=0,vector_steps=0)
    with native_log(folder/'runtime.log') as console:
        try:
            from isaaclab.app import AppLauncher
            app=AppLauncher(headless=True,device=args.device,enable_cameras=False,visualizer='none',livestream=0).app
            import torch
            from drl.MARL.c1_runner import build_observation_bridge
            from drl.MARL.isaaclab_c1_runner import seed_native_training
            from drl.MARL.isaaclab_c1_env import MSSRC1IsaacLabEnv,make_c1_isaaclab_env_cfg
            from drl.MARL.c1_action_events import C1ActionEvents
            seed_native_training(args.global_seed)
            owners=tuple(f'smores_{i:02d}' for i in range(1,9))
            bridge=build_observation_bridge(ROOT)
            actor=build_viewer_actor(bridge,config);before=actor_digest(actor)
            write_json(folder/'actor_restore_audit.json',actor.viewer_restore_audit)
            cfg=make_c1_isaaclab_env_cfg(possible_agents=owners,num_envs=1,device=args.device)
            cfg.initialization_mode='loose_modules';cfg.domains=(args.task,);cfg.seed=args.global_seed
            cfg.seeds=(args.seed,);cfg.resample_train=True;cfg.train_sampler_seed=args.global_seed;cfg.smoke_seed_cycle=True
            cfg.reward_config_path=config['reward_config'];cfg.c1_probe_config_path=config['runtime_probe_config']
            cfg.c1_evidence_directory=str(folder/'c1_evidence');cfg.episode_length_s=args.episode_seconds
            cfg.c1_readonly_state_cache=True
            env=MSSRC1IsaacLabEnv(cfg);env.reset()
            if tuple(env.seed_by_env)!=(args.seed,):raise RuntimeError('Unexpected initial course seed')
            env.c1_action_events=C1ActionEvents(folder/'actions',masked_detail_limit=0)
            bridge=ActorOnlyObservationBridge(bridge,lambda:env.generation[0])
            native=env.initialize_policy_bridge(observation_bridge=bridge,actor_only=True,
                bulk_host_reads=True,structural_scheduler='event')
            generation=env.generation[0];sim_start=env._sim_step_counter*env.physics_dt
            reason='step_limit'
            with torch.no_grad():
                for i in range(args.steps):
                    if time.perf_counter()-started>=args.max_wall_seconds:reason='wall_limit';break
                    if not app.is_running():reason='application_closed';break
                    contexts=native.provider.current_action_contexts()
                    decisions=[sample_contexts(actor,[c],owners,accelerated=True)[0][0]
                        if c.rollout_active else SimpleNamespace(action=None,action_set_prediction_template=None)
                        for c in contexts]
                    carrier=native.provider.encode_env_actions(decisions)
                    _,rewards,*_=env.step(carrier)
                    if any(not torch.isfinite(r).all() for r in rewards.values()):raise RuntimeError('Nonfinite reward')
                    summary['vector_steps']=i+1
                    if env.generation[0]!=generation:reason='first_episode_ended';break
                    native.provider.reset()
                    if (i+1)%300==0:print(f'frames={i+1} simulated_s={(i+1)/30:.1f} wall_s={time.perf_counter()-started:.1f}',file=console,flush=True)
            summary.update(status='completed',cutoff_reason=reason,
                simulated_seconds=env._sim_step_counter*env.physics_dt-sim_start,
                physics_hz=1/env.physics_dt,control_hz=1/env.step_dt,
                episode_records=list(env.c1_runtime.episode_records),
                scheduler=native.provider.structural_option_metrics(),action_counts=dict(env.c1_action_events.totals))
            if env.assembly_records:raise RuntimeError('Unexpected Expert control')
        except BaseException as error:
            summary.update(status='error',error=str(error),traceback=traceback.format_exc())
        finally:
            try:
                if actor is not None:
                    summary['actor_unchanged']=actor_digest(actor)==before and all(p.grad is None for p in actor.parameters())
                    if not summary['actor_unchanged']:summary['status']='error'
                if env is not None:
                    env.c1_runtime.flush_open_episodes('validation_cutoff')
                    summary['episode_records']=list(env.c1_runtime.episode_records)
                    write_json(folder/'reset_records.json',dict(records=env.reset_records))
            finally:
                try:
                    if env is not None:env.close()
                finally:
                    if app is not None:app.close()
                    summary['wall_seconds']=time.perf_counter()-started
                    write_json(folder/'summary.json',summary)
                    print(json.dumps(summary,default=str),file=console,flush=True)
    return 1 if summary['status']=='error' else 0


def main(argv=None):
    args=parse_args(argv)
    if args.preflight:
        print(json.dumps(preflight(args),indent=2));return 0
    return run(args)


if __name__=='__main__':raise SystemExit(main())
