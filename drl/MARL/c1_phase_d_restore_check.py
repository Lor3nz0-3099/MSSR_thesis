"""MANUAL-ONLY native snapshot restore smoke; frozen/no policy/no training.

Checks one reset for each of S0/S1/S2/S3 and four known C1 task domains.
Does not claim structural stability beyond this bounded physical replay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-bank',type=Path,required=True)
    parser.add_argument('--reward-config',type=Path,required=True)
    parser.add_argument('--c1-probe-config',type=Path,required=True)
    parser.add_argument('--steps-per-stage',type=int,default=15)
    parser.add_argument('--global-seed',type=int,default=3103)
    return parser


def check(args):
    import torch
    from drl.MARL.isaaclab_c1_env import MSSRC1IsaacLabEnv,make_c1_isaaclab_env_cfg
    from drl.MARL.c1_native_observation import initial_frame_from_native
    from drl.MARL.c1_structure import assess_structure
    from drl.MARL.c1_config import C1Settings,physical_bounds
    from drl.MARL.c1_guided_feedback import TargetGraphProgress,load_target
    owners=tuple(f'smores_{i:02d}' for i in range(1,9))
    domains=('navigation','stairs','gap','manipulation')
    cfg=make_c1_isaaclab_env_cfg(possible_agents=owners,num_envs=4,device=args.device)
    cfg.reward_config_path=str(args.reward_config.resolve())
    cfg.c1_probe_config_path=str(args.c1_probe_config.resolve())
    cfg.domains=domains
    cfg.course_split='train'
    cfg.initialization_mode='assembled_state'
    cfg.snapshot_bank_path=str(args.snapshot_bank.resolve())
    cfg.snapshot_stage_weights=dict(S0=1.,S1=0.,S2=0.,S3=0.)
    cfg.resample_train=True
    cfg.train_sampler_seed=args.global_seed
    cfg.seed=args.global_seed
    cfg.episode_length_s=120.
    env=MSSRC1IsaacLabEnv(cfg)
    settings=C1Settings.load(args.c1_probe_config)
    zeros={owner:torch.zeros(env.num_envs,1,device=env.device) for owner in owners}
    records=[]
    try:
        for stage in ('S0','S1','S2','S3'):
            env.snapshot_sampler.weights={key:float(key==stage) for key in ('S0','S1','S2','S3')}
            env.reset()
            current=env._local_snapshots()
            for i,domain in enumerate(domains):
                frame=initial_frame_from_native(env,i,current[i])
                assessed=assess_structure(frame,physical_bounds(env.courses[i],settings,owners))
                progress=TargetGraphProgress(load_target(domain),owners)
                progress.potential(frame)
                expected=env.snapshot_records[i]['stage']
                actual=int(progress._last_match)
                if expected!=stage or actual!=len(frame.connections) or not assessed.safe:
                    raise RuntimeError(f'Invalid restored {domain}/{stage}: '
                        f'{actual} target edges, {len(frame.connections)} connections, '
                        f'assessment={assessed.reason}')
                records.append(dict(domain=domain,stage=stage,
                    restored_edges=actual,seed=env.seed_by_env[i]))
            for _ in range(args.steps_per_stage):
                env.step(zeros)
            current=env._local_snapshots()
            for i,domain in enumerate(domains):
                frame=initial_frame_from_native(env,i,current[i])
                assessed=assess_structure(frame,physical_bounds(env.courses[i],settings,owners))
                if not assessed.safe:
                    raise RuntimeError(f'Native snapshot unstable after replay: {domain}/{stage} '
                        f'({assessed.reason})')
            print('C1_SNAPSHOT_RESTORE_PASS: '+json.dumps(dict(stage=stage,
                domains=list(domains),steps=args.steps_per_stage)),flush=True)
        print('C1_SNAPSHOT_VALIDATION_PASS: '+json.dumps(records),flush=True)
    finally:
        env.close()


def main(argv=None):
    from isaaclab.app import AppLauncher
    parser=build_parser();AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args(argv)
    from drl.MARL.c1_expert_snapshots import SnapshotBank,current_snapshot_physics
    SnapshotBank(args.snapshot_bank,'train',current_snapshot_physics())
    if args.steps_per_stage<1 or args.steps_per_stage>120:
        parser.error('bounded steps-per-stage must be in 1..120')
    if not args.reward_config.is_file() or not args.c1_probe_config.is_file():
        parser.error('missing reward/probe configuration')
    launcher=AppLauncher(args)
    try:
        check(args)
    finally:
        launcher.app.close()


if __name__=='__main__':
    main()
