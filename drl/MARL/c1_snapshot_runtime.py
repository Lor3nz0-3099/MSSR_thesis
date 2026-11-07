"""Selected-row snapshot initialization; no Expert planning or stepping."""
from dataclasses import asdict


def initialize_snapshot_sampling(env):
    from drl.MARL.c1_expert_snapshots import SnapshotBank,current_snapshot_physics
    from drl.MARL.c1_snapshot_sampling import SnapshotSampler
    path=getattr(env.cfg,'snapshot_bank_path',None)
    if not path:raise ValueError('assembled_state requires explicit snapshot-bank')
    env.snapshot_bank=SnapshotBank(path,env.cfg.course_split,current_snapshot_physics())
    env.snapshot_sampler=SnapshotSampler(env.snapshot_bank,env.domain_by_env,env.cfg.train_sampler_seed,
        weights=getattr(env.cfg,'snapshot_stage_weights',None),split=env.cfg.course_split)
    env.snapshot_records=[env.snapshot_sampler.sample(i) for i in range(env.num_envs)]
    env.train_sampler=env.snapshot_sampler if env.cfg.course_split=='train' else None
    env.seed_by_env=tuple(r['identity']['seed'] for r in env.snapshot_records)


def resample_snapshot_course(env,index):
    from drl.MARL.c1_runner import make_course
    from drl.MARL.isaaclab_course_reset import update_native_course
    if env.generation[index]==0:return
    record=env.snapshot_sampler.sample(index)
    identity=record['identity']
    course=make_course(identity['domain'],identity['seed'],identity['split'])
    update_native_course(env.sim.stage,env_root=f'/World/envs/env_{index}',domain=identity['domain'],
        course=course,origin=env.scene.env_origins[index].cpu().tolist())
    seeds,courses=list(env.seed_by_env),list(env.courses)
    seeds[index],courses[index]=identity['seed'],course
    env.seed_by_env,env.courses=tuple(seeds),tuple(courses)
    env.snapshot_records[index]=record


def restore_snapshot_roots(env,index):
    from drl.MARL.c1_expert_snapshots import restore_native_snapshot
    value=env.snapshot_bank.load(env.snapshot_records[index])
    return restore_native_snapshot(env,index,value,restore_connections=False)


def prime_snapshot_episode(env,index,payload):
    from drl.MARL.c1_native_observation import initial_frame_from_native
    from drl.MARL.c1_native_runtime import NativeC1RewardRuntime
    env.native_transforms.invalidate()
    # Real readback after exact-state writes and joint reconstruction, prior
    # to the first policy action. No fabricated interval/ticks are introduced.
    frame=initial_frame_from_native(env,index,env._local_snapshots()[index])
    if not hasattr(env,'c1_runtime'):
        env.c1_runtime=NativeC1RewardRuntime(settings_path=env.cfg.c1_probe_config_path,
            evidence_directory=getattr(env.cfg,'c1_evidence_directory',None))
        env.c1_episode_records=env.c1_runtime.episode_records
    env.c1_runtime.prime(env,index,frame)
    _,evaluator=env.c1_runtime.evaluators[index]
    if round(evaluator.last_target_fraction*7)!=payload['actual_target_edges']:
        raise ValueError('Restored graph disagrees with measured snapshot target count')
    record=env.snapshot_records[index]
    env.c1_runtime._emit('initial_states',dict(env_index=index,generation=env.generation[index],
        snapshot_path=record['path'],snapshot_sha256=record['sha256'],stage=payload['stage'],
        actual_target_edges=payload['actual_target_edges'],identity=payload['identity'],frame=asdict(frame)))
    return dict(stage=payload['stage'],actual_target_edges=payload['actual_target_edges'],
        snapshot_path=record['path'],snapshot_sha256=record['sha256'])
