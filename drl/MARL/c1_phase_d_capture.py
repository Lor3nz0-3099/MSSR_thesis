"""MANUAL-ONLY native C1 Expert partial-state capture (no training).

Run this explicitly in the user's Isaac Lab environment.  Does not fabricate
module positions or use policy outputs.  Missing stages are a hard failure;
no partial bank is silently promoted to a training asset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path


def stage_for_count(count):
    if count == 0:
        return 'S0'
    if count <= 2:
        return 'S1'
    if count <= 4:
        return 'S2'
    if count <= 6:
        return 'S3'
    return None


def stable_connection_signature(connections):
    """Canonical undirected port-labelled graph signature from C1Connection."""
    return tuple(sorted(tuple(sorted(connection.endpoints)) for connection in connections))



def record_capture_diagnostic(args, phase, **details):
    """Persist the phase BEFORE Kit shutdown (which can terminate Python)."""
    payload = dict(phase=phase, **details)
    print('EXPERT_SNAPSHOT_LIFECYCLE: '+json.dumps(payload, sort_keys=True), flush=True)
    diagnostic = args.output.parent / 'capture.diagnostic.json'
    try:
        diagnostic.write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n')
    except OSError as error:
        print('EXPERT_SNAPSHOT_DIAGNOSTIC_WRITE_ERROR: '+str(error),
              file=sys.stderr, flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reward-config', type=Path, required=True)
    parser.add_argument('--domains', nargs=4,
                        choices=('navigation','stairs','gap','manipulation'),
                        default=['navigation','stairs','gap','manipulation'])
    parser.add_argument('--seeds', nargs=4, type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-vector-steps', type=int, default=9000)
    parser.add_argument('--max-wall-seconds', type=float, default=1800.)
    parser.add_argument('--min-stable-frames', type=int, default=3)
    parser.add_argument('--progress-every', type=int, default=60)
    parser.add_argument('--global-seed', type=int, default=3103)
    return parser


def _check_args(args):
    if set(args.domains) != {'navigation','stairs','gap','manipulation'}:
        raise ValueError('Exactly one of each C1 domain is required')
    if args.output.exists():
        raise ValueError('Capture output must be a NEW directory')
    if not args.reward_config.is_file():
        raise ValueError('Missing reward config')
    if args.max_vector_steps < 1 or not 0 < args.max_wall_seconds <= 86400.:
        raise ValueError('Invalid bounded capture budget')
    if args.min_stable_frames < 1:
        raise ValueError('Stable-frame count must be positive')
    if args.progress_every < 1:
        raise ValueError('Progress interval must be positive')
    from drl.MARL.c1_train_sampling import catalog_seeds
    for domain, seed in zip(args.domains,args.seeds,strict=True):
        if seed not in catalog_seeds(domain,'train'):
            raise ValueError(f'{domain} seed {seed} is outside TRAIN split')


def capture(args):
    """Called only after AppLauncher is initialized by main()."""
    import torch
    from drl.MARL.isaaclab_c1_env import MSSRC1IsaacLabEnv,make_c1_isaaclab_env_cfg
    from drl.MARL.c1_expert_snapshots import (
        BANK_SCHEMA,capture_native_snapshot,current_snapshot_physics,
        current_snapshot_identity,save_snapshot)
    from drl.MARL.c1_native_observation import initial_frame_from_native
    from drl.MARL.c1_structure import assess_structure
    from drl.MARL.c1_guided_feedback import TargetGraphProgress,load_target
    from drl.MARL.c1_config import C1Settings,physical_bounds

    owners=tuple(f'smores_{i:02d}' for i in range(1,9))
    cfg=make_c1_isaaclab_env_cfg(possible_agents=owners,num_envs=4,device=args.device)
    cfg.reward_config_path=str(args.reward_config.resolve())
    cfg.domains=tuple(args.domains)
    cfg.seeds=tuple(args.seeds)
    cfg.course_split='train'
    cfg.resample_train=False
    cfg.seed=args.global_seed
    cfg.initialization_mode='expert_validation'
    cfg.episode_length_s=180.
    record_capture_diagnostic(args,'environment_construct_begin', num_envs=4)
    env=MSSRC1IsaacLabEnv(cfg)
    record_capture_diagnostic(args,'environment_constructed', num_envs=env.num_envs)
    records=[]
    streak=[(None,0) for _ in range(env.num_envs)]
    last_edge_counts=[None]*env.num_envs
    step=-1
    targets=[TargetGraphProgress(load_target(domain),owners) for domain in args.domains]
    settings=C1Settings.load()
    physics=current_snapshot_physics()
    started=time.monotonic()
    zeros={owner:torch.zeros(env.num_envs,1,device=env.device) for owner in owners}
    try:
        record_capture_diagnostic(args,'environment_reset_begin')
        env.reset()
        record_capture_diagnostic(args,'environment_reset_complete')
        for step in range(args.max_vector_steps):
            if time.monotonic()-started>args.max_wall_seconds:
                break
            env.native_transforms.invalidate()
            raw=env._local_snapshots()
            for i,domain in enumerate(args.domains):
                captured={row['stage'] for row in records if row['identity']['domain']==domain}
                if len(captured)==4:
                    continue
                frame=initial_frame_from_native(env,i,raw[i])
                assessment=assess_structure(frame,physical_bounds(env.courses[i],settings,owners))
                # Read real graph matching with global role assignment.  Merely
                # counting physical joint connections is never sufficient.
                tracker=targets[i]
                tracker.potential(frame)
                count=int(tracker._last_match)
                stage=stage_for_count(count)
                compatible=(stage is not None and len(frame.connections)==count
                    and assessment.safe and assessment.reason in ('valid','disconnected_assembly')
                    and all(j.joint_valid for j in frame.connections))
                if step % args.progress_every == 0 or len(frame.connections) != last_edge_counts[i]:
                    print('EXPERT_SNAPSHOT_PROGRESS: '+json.dumps(dict(
                        step=step,domain=domain,connections=len(frame.connections),
                        matching=count,stage=stage,reason=assessment.reason,
                        safe=assessment.safe,invalid_joints=sum(not j.joint_valid for j in frame.connections),
                        stable_frames=streak[i][1],captured=sorted(captured))),flush=True)
                last_edge_counts[i]=len(frame.connections)
                if not compatible or stage in captured:
                    streak[i]=(None,0)
                    continue
                signature=stable_connection_signature(frame.connections)
                current=(stage,signature)
                previous,consecutive=streak[i]
                streak[i]=(current,consecutive+1 if current==previous else 1)
                if streak[i][1]<args.min_stable_frames:
                    continue
                identity=current_snapshot_identity(env,i)
                certificate=dict(expert_generated=True,target_compatible=True,
                    target_edges=count,source='native Expert verified by measured C1 graph and physics',
                    expert_step=step,stable_frames=args.min_stable_frames)
                payload=capture_native_snapshot(env,i,stage,identity=identity,
                    physics=physics,physical_certificate=certificate)
                relative=Path('snapshots')/f'{domain}_{identity["seed"]}_{stage}.json'
                path=args.output/relative
                save_snapshot(path,payload)
                records.append(dict(path=relative.as_posix(),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    identity=identity,stage=stage))
                print('EXPERT_SNAPSHOT_CAPTURED: '+json.dumps(dict(domain=domain,stage=stage,
                    matching=count,step=step,path=str(path))),flush=True)
            if len(records)==16:
                break
            env.step(zeros)
        summary={domain:sorted(row['stage'] for row in records
                if row['identity']['domain']==domain) for domain in args.domains}
        print('EXPERT_SNAPSHOT_COVERAGE: '+json.dumps(summary,sort_keys=True),flush=True)
        missing={domain:sorted(set(('S0','S1','S2','S3'))-set(stages))
                 for domain,stages in summary.items() if len(stages)<4}
        if missing:
            raise RuntimeError(f'Expert snapshot capture incomplete (NOT trainable): {missing}')
        manifest=args.output/'bank.json'
        with manifest.open('x') as stream:
            json.dump(dict(schema=BANK_SCHEMA,records=sorted(records,key=lambda r:(
                r['identity']['domain'],r['stage']))),stream,sort_keys=True,indent=2)
            stream.write('\n')
        print('EXPERT_SNAPSHOT_BANK_READY: '+str(manifest),flush=True)
    except BaseException as error:
        record_capture_diagnostic(args,'capture_failed',step=step,
            error_type=type(error).__name__,message=str(error),
            traceback=traceback.format_exc(),
            captured=[f'{r["identity"]["domain"]}/{r["stage"]}' for r in records])
        print('EXPERT_SNAPSHOT_ABORT: '+json.dumps(dict(
            step=step,error_type=type(error).__name__,message=str(error),
            captured=[f'{r["identity"]["domain"]}/{r["stage"]}' for r in records])),flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        env.close()


def main(argv=None):
    # CPU preflight before initializing the graphical/simulation runtime.
    parser=build_parser()
    # AppLauncher injects optional simulator CLI flags.
    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args(argv)
    _check_args(args)
    args.output.mkdir(parents=True,exist_ok=False)
    launcher=None
    record_capture_diagnostic(args,'launcher_begin')
    try:
        launcher=AppLauncher(args)
        record_capture_diagnostic(args,'launcher_ready')
        capture(args)
        if not (args.output/'bank.json').is_file():
            raise RuntimeError('capture returned without bank.json: incomplete result')
        record_capture_diagnostic(args,'bank_ready')
    except BaseException as error:
        record_capture_diagnostic(args,'fatal_before_shutdown',
            error_type=type(error).__name__,message=str(error),
            traceback=traceback.format_exc())
        print('EXPERT_SNAPSHOT_FATAL_BEFORE_SHUTDOWN: '
              +repr(error),file=sys.stderr,flush=True)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    finally:
        if launcher is not None:
            print('EXPERT_SNAPSHOT_APP_CLOSE_BEGIN',flush=True)
            launcher.app.close()


if __name__=='__main__':
    main()
