"""User-run frozen BC C1 physical diagnostic. No learner, critic model or optimizer.

--preflight and --monitor-once are CPU-only; no AppLauncher import or launch.
Only the normal execution path starts Isaac. Never starts a training runner.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'logs/isaaclab_c1_native/redesign/c1_structural_physics_40s_r1'
TASKS = ('navigation', 'stairs', 'gap', 'manipulation')
REWARD = ROOT / 'logs/appo_smoke/calibration_validated_train/reward_config.initial.json'


def write_json(path, value):
    from drl.MARL.c1_native_runtime import sanitized_record
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(sanitized_record(value), indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=1200, help='Vector decisions, 30Hz: default 40s/env; max1200')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--global-seed', type=int, default=3103)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--headless', action='store_true', help='Required for physical execution; cameras disabled')
    parser.add_argument('--max-wall-seconds', type=float, default=86400., help='Bounded cutoff between decisions')
    parser.add_argument('--preflight', action='store_true', help='Validate/hash artifacts without starting Isaac')
    parser.add_argument('--monitor-once', action='store_true', help='Read progress/results once, without Isaac')
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 1200:
        parser.error('--steps requires 1..1200')
    if not math.isfinite(args.max_wall_seconds) or not 0 < args.max_wall_seconds <= 86400:
        parser.error('--max-wall-seconds requires a finite value in (0,86400]')
    if args.global_seed < 0 or args.global_seed >= 2**32:
        parser.error('--global-seed requires 0..2**32-1')
    if not args.preflight and not args.monitor_once and not args.headless:
        parser.error('Physical execution requires --headless')
    return args


def configuration(args):
    from drl.MARL.c1_runner import CHECKPOINT, BC_SHA256
    from drl.MARL.bc_v4_runtime_inference import load_bc_v4_runtime_inference_config
    calibration = load_bc_v4_runtime_inference_config(ROOT)
    threshold = float(calibration['deterministic_presence_threshold_logit'])
    if threshold != 4.5:
        raise ValueError('Diagnostic requires the validated BC-v4 presence threshold 4.5')
    return dict(schema='mssr.c1.frozen_physics_diagnostic.v1', num_envs=4, tasks=list(TASKS),
        initialization_mode='loose_modules', global_seed=args.global_seed, device=args.device,
        physics_hz=240, decimation=8, decision_hz=30, steps=args.steps,
        simulated_seconds_per_env=args.steps/30., aggregate_environment_seconds=4*args.steps/30.,
        primitive_timeout_s=30., episode_seconds=60., train_sampler='smoke-cycle',
        checkpoint=str(ROOT/CHECKPOINT), expected_bc_sha256=BC_SHA256,
        reward_config=str(REWARD), probe_config=str(ROOT/'configs/marl/c1_assembly_probe.json'),
        presence_threshold_logit=threshold, sampling='stochastic', gaussian_log_std_init=-2.,
        learner_constructed=False, critic_model_constructed=False, optimizer_constructed=False,
        optimizer_updates=0, backward_calls=0, minibatches=0, expert_actions=0,
        realtime_pacing=False, cameras=False, headless=True, max_wall_seconds=args.max_wall_seconds,
        output=str(args.output.resolve()), note='Prepared experiment; success flags are observational, not learning.')


def source_hashes():
    files = sorted((ROOT/'drl/MARL').glob('*.py'))
    files += sorted((ROOT/'scripts/smores_ep/src/smores_ep').rglob('*.py'))
    files += sorted((ROOT/'configs/marl').glob('*.json'))
    files += [ROOT/'configs/bc_v1/runtime_inference_v1.json', REWARD,
        ROOT/'assets/smores-ep/usd_physics/smores_ep_physics_v1.usd']
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def preflight(args):
    config = configuration(args)
    actual = hashlib.sha256(Path(config['checkpoint']).read_bytes()).hexdigest()
    if actual != config['expected_bc_sha256']:
        raise ValueError('Frozen BC checkpoint hash mismatch')
    for filename in ('reward_config', 'probe_config'):
        json.loads(Path(config[filename]).read_text())
    # Construct only the CPU seed sampler, not a model or environment.
    from drl.MARL.c1_train_sampling import NativeC1TrainSampler
    sampler = NativeC1TrainSampler(TASKS, args.global_seed, cycle=True)
    config['initial_train_seeds'] = [sampler.sample(i).seed for i in range(4)]
    config['bc_sha256_verified'] = actual
    config['source_sha256'] = source_hashes()
    return config


def build_frozen_actor(bridge, config):
    from drl.MARL.bc_v4_actor import BCV4SharedActorWrapper
    actor = BCV4SharedActorWrapper.from_repository_checkpoint(spec=bridge.spec,
        feature_schema=bridge.feature_schema, repo_root=ROOT,
        checkpoint_path=Path(config['checkpoint']), expected_sha256=config['expected_bc_sha256'],
        map_location=config['device'], device=config['device'],
        resource_log_std_init=-2., structural_log_std_init=-2.)
    actor.to(config['device']).eval().requires_grad_(False)
    if actor.presence_threshold_logit != config['presence_threshold_logit']:
        raise ValueError('Actor decode configuration differs from diagnostic')
    return actor


def actor_digest(actor):
    result = hashlib.sha256()
    for key, tensor in sorted(actor.state_dict().items()):
        result.update(key.encode())
        result.update(tensor.detach().cpu().numpy().tobytes())
    return result.hexdigest()


def sample_decision(actor, context, owners):
    import torch
    from drl.MARL.c1_feasibility import apply_availability
    from drl.MARL.policy import sample_marl_action_bundle
    with torch.no_grad():
        output = actor.forward_all_owners(graph=context.actor_graph,
            routing=context.routing_snapshot, possible_agents=owners)
        filtered = apply_availability(output, context.availability_snapshot)
        mask = torch.as_tensor(context.hard_mask, dtype=torch.bool, device=output['records'][0].presence_logit.device)
        mask = (mask | context.availability_snapshot.mask_tensor(device=mask.device)).detach().clone()
        action = sample_marl_action_bundle(**{key:filtered[key] for key in (
            'records','resource_payloads','structural_payloads','resource_log_std','structural_log_std')},
            hard_mask=mask, deterministic=False)
    return SimpleNamespace(action=action, action_set_prediction_template=output['prediction_template']), output


def rollout_frozen(env, actor, provider, recorder, owners, *, steps, max_wall_seconds, progress_callback):
    """Physical collection only. No value evaluation, rollout memory or learner."""
    import torch
    timings = dict(context_s=0., inference_sampling_s=0., encode_projection_s=0., environment_step_s=0., diagnostic_logging_s=0.)
    started = time.perf_counter()
    completed = 0
    cutoff = 'horizon_reached'
    for decision_step in range(steps):
        if time.perf_counter() - started >= max_wall_seconds:
            cutoff = 'wall_limit_before_horizon'
            break
        recorder.decision_step = decision_step
        tick = time.perf_counter()
        contexts = provider.current_action_contexts()
        timings['context_s'] += time.perf_counter()-tick
        decisions = []
        for i, context in enumerate(contexts):
            tick = time.perf_counter()
            decision, output = sample_decision(actor, context, owners)
            timings['inference_sampling_s'] += time.perf_counter()-tick
            tick = time.perf_counter()
            recorder.observe_context(i, env.generation[i], output['records'], context.availability_snapshot,
                env._sim_step_counter*env.physics_dt)
            timings['diagnostic_logging_s'] += time.perf_counter()-tick
            decisions.append(decision)
        tick = time.perf_counter()
        carrier = provider.encode_env_actions(decisions)
        timings['encode_projection_s'] += time.perf_counter()-tick
        tick = time.perf_counter()
        with torch.no_grad():
            _, rewards, *_ = env.step(carrier)
        timings['environment_step_s'] += time.perf_counter()-tick
        if any(not torch.isfinite(reward).all() for reward in rewards.values()):
            raise RuntimeError('Nonfinite reward in unchanged native C1 evaluator')
        tick = time.perf_counter()
        for row in list(env.c1_runtime.recent_transitions)[-env.num_envs:]:
            generation, evaluator = env.c1_runtime.evaluators[row['env_index']]
            if generation != row['generation']:
                raise RuntimeError('Transition attribution lost across native reset')
            recorder.observe_transition(row, evaluator.last_frame)
        # No learner needs terminal critic sidecars. Drop the consumed context;
        # next act captures current/reset state, retaining per-env claim caches.
        provider.reset()
        completed += 1
        progress_callback(completed, timings, time.perf_counter()-started)
        timings['diagnostic_logging_s'] += time.perf_counter()-tick
    # Observe the final real support after release, including the final tick.
    # No final actor forward or shadow action sample.
    from drl.MARL.policy import bind_marl_candidates
    tick = time.perf_counter()
    for i, context in enumerate(provider.current_action_contexts()):
        count = len(context.routing_snapshot['resource_candidates']) + len(context.routing_snapshot['structural_bindings'])
        records = bind_marl_candidates(routing=context.routing_snapshot, presence_logits=torch.zeros(count), presence_threshold_logit=4.5)
        recorder.observe_context(i, env.generation[i], records, context.availability_snapshot,
            env._sim_step_counter*env.physics_dt, count_masks=False)
    timings['diagnostic_logging_s'] += time.perf_counter()-tick
    return dict(decisions_completed=completed, cutoff_reason=cutoff, timings_s=timings,
        rollout_wall_seconds=time.perf_counter()-started)


def runtime_settings(env):
    """Read authored scene settings and configuration, no physical step."""
    from pxr import Usd, UsdPhysics
    scene = env.sim.stage.GetPrimAtPath(env.cfg.sim.physics_prim_path)
    roots = env.module_roots_by_env
    modules = []
    for i, row in enumerate(roots):
        for owner, root in row.items():
            prim = env.sim.stage.GetPrimAtPath(root)
            bodies = [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            modules.append(dict(env_index=i, module=owner, articulation_attributes={a.GetName():a.Get()
                for a in prim.GetAttributes() if a.GetName().startswith('physxArticulation:')},
                rigid_body_attributes=[dict(path=str(p.GetPath()), **{a.GetName():a.Get() for a in p.GetAttributes()
                    if a.GetName().startswith('physxRigidBody:')}) for p in bodies],
                wheel_material={a.GetName():a.Get() for a in env.sim.stage.GetPrimAtPath(root+'/materials/wheel').GetAttributes()
                    if 'Friction' in a.GetName() or 'friction' in a.GetName()}))
    return dict(physics_dt=env.physics_dt, step_dt=env.step_dt, decimation=env.cfg.decimation,
        requested_ccd=bool(env.cfg.sim.physics.enable_ccd),
        requested_solver_type=env.cfg.sim.physics.solver_type,
        physics_device=env.cfg.sim.device, use_fabric=env.cfg.sim.use_fabric,
        gravity=list(env.cfg.sim.gravity),
        render_interval=env.cfg.sim.render_interval, render_enabled=bool(env.sim.is_rendering),
        has_gui=bool(env.sim.has_gui), sensors=list(env.scene.sensors),
        scene_attributes={a.GetName():a.Get() for a in scene.GetAttributes()
            if a.GetName().startswith('physxScene:') and isinstance(a.Get(), (str,int,float,bool,type(None)))},
        actuator_profile=vars(next(iter(env.state_readers_by_env[0].values()))._drive), modules=modules,
        seed_by_env=list(env.seed_by_env), native_primitive_timeout_s=env.native_dispatcher.primitive_timeout_s)


def run(args):
    config = preflight(args)
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    # Exclusive create protects concurrent invocation as well as reruns.
    with (folder/'RUN_STARTED.json').open('x') as handle:
        handle.write(json.dumps(dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), configuration=config), indent=2))
    write_json(folder/'run_config.json', config)
    write_json(folder/'source_sha256.json', config['source_sha256'])
    env = application = actor = recorder = None
    before = None
    sim_start = None
    started = time.perf_counter()
    result = dict(status='error', configuration={k:v for k,v in config.items() if k != 'source_sha256'},
        optimizer_updates=0, backward_calls=0, optimizer_constructed=False, learner_constructed=False, expert_actions=0,
        decisions_completed=0, errors=[])
    try:
        from isaaclab.app import AppLauncher
        application = AppLauncher(headless=True, device=args.device, enable_cameras=False).app
        import torch
        from drl.MARL.c1_runner import build_observation_bridge
        from drl.MARL.isaaclab_c1_runner import seed_native_training
        from drl.MARL.isaaclab_c1_env import MSSRC1IsaacLabEnv, make_c1_isaaclab_env_cfg
        from drl.MARL.c1_primitive_lifecycle import PrimitiveLifecycleRecorder
        seed_native_training(args.global_seed)
        owners = tuple(f'smores_{i:02d}' for i in range(1,9))
        bridge = build_observation_bridge(ROOT)
        actor = build_frozen_actor(bridge, config)
        before = actor_digest(actor)
        cfg = make_c1_isaaclab_env_cfg(possible_agents=owners, num_envs=4, device=args.device)
        cfg.initialization_mode='loose_modules';cfg.domains=TASKS;cfg.seed=args.global_seed
        cfg.resample_train=True;cfg.train_sampler_seed=args.global_seed;cfg.smoke_seed_cycle=True
        cfg.episode_length_s=60.;cfg.reward_config_path=str(REWARD)
        cfg.c1_evidence_directory=str(folder/'c1_discovery')
        env = MSSRC1IsaacLabEnv(cfg)
        env.reset()
        native = env.initialize_policy_bridge(observation_bridge=bridge)
        recorder = PrimitiveLifecycleRecorder(folder, TASKS)
        recorder.attach_environment(env)
        settings = runtime_settings(env)
        write_json(folder/'runtime_settings.json', settings)
        if settings['render_enabled'] or settings['has_gui'] or settings['sensors']:
            raise RuntimeError('Headless diagnostic unexpectedly enabled rendering/GUI/sensors')
        if tuple(env.seed_by_env) != tuple(config['initial_train_seeds']):
            raise RuntimeError('Native seed assignment differs from CPU preflight')
        result['startup_wall_seconds'] = time.perf_counter()-started
        sim_start = env._sim_step_counter*env.physics_dt
        if torch.cuda.is_available() and str(args.device).startswith('cuda'):
            torch.cuda.synchronize(args.device)
            torch.cuda.reset_peak_memory_stats(args.device)

        def progress(completed, timings, rollout_wall):
            result['decisions_completed'] = completed
            summary = recorder.summary()
            span = env._sim_step_counter*env.physics_dt-sim_start
            write_json(folder/'progress.json', dict(status='running', decisions_completed=completed,
                decisions_planned=args.steps, simulated_seconds_per_env=span, aggregate_environment_seconds=4*span,
                wall_seconds=time.perf_counter()-started, rollout_wall_seconds=rollout_wall,
                timings_s=timings, by_env=summary['by_env'], errors=summary['errors'],
                feasibility_cache=native.provider.feasibility_metrics()))

        progress(0, {}, 0.)
        collected = rollout_frozen(env, actor, native.provider, recorder, owners,
            steps=args.steps, max_wall_seconds=max(0.,args.max_wall_seconds-(time.perf_counter()-started)),
            progress_callback=progress)
        if torch.cuda.is_available() and str(args.device).startswith('cuda'):
            sync = time.perf_counter();torch.cuda.synchronize(args.device)
            collected['final_cuda_sync_seconds'] = time.perf_counter()-sync
            collected['rollout_wall_seconds'] += collected['final_cuda_sync_seconds']
            result['peak_gpu_allocated_bytes'] = torch.cuda.max_memory_allocated(args.device)
            result['peak_gpu_reserved_bytes'] = torch.cuda.max_memory_reserved(args.device)
        result.update(collected)
        result['simulated_seconds_per_env'] = env._sim_step_counter*env.physics_dt-sim_start
        result['aggregate_environment_seconds'] = 4*result['simulated_seconds_per_env']
        result['feasibility_cache'] = native.provider.feasibility_metrics()
        result['action_summary'] = env.c1_action_events.summary() if hasattr(env,'c1_action_events') else {}
        result['expert_assembly_records'] = len(env.assembly_records)
        result['status'] = 'completed' if collected['cutoff_reason']=='horizon_reached' else 'incomplete'
        if env.assembly_records:
            raise RuntimeError('Expert assembly entered the diagnostic')
    except BaseException as error:
        result.update(status='error', error=str(error), traceback=traceback.format_exc())
    finally:
        if env is not None and sim_start is not None:
            result['simulated_seconds_per_env'] = env._sim_step_counter*env.physics_dt-sim_start
            result['aggregate_environment_seconds'] = 4*result['simulated_seconds_per_env']
            write_json(folder/'reset_records.json', dict(records=getattr(env,'reset_records',[]),
                final_seed_by_env=list(env.seed_by_env)))
        if env is not None and hasattr(env,'c1_runtime'):
            env.c1_runtime.flush_open_episodes('frozen_physics_diagnostic_end')
        if recorder is not None:
            recorder.censor_open(env._sim_step_counter*env.physics_dt, result.get('cutoff_reason',result['status']))
            result.update(recorder.summary())
            recorder.detach_environment()
        if actor is not None and before is not None:
            result['actor_unchanged'] = actor_digest(actor)==before
            if not result['actor_unchanged'] or any(p.grad is not None for p in actor.parameters()):
                result.update(status='error', error='Frozen actor state/gradient contract violated')
        result['sources_unchanged'] = all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in config['source_sha256'].items())
        if not result['sources_unchanged']:
            result.update(status='error', error='Source files changed during diagnostic')
        result['bc_checkpoint_file_unchanged'] = hashlib.sha256(Path(config['checkpoint']).read_bytes()).hexdigest()==config['expected_bc_sha256']
        if not result['bc_checkpoint_file_unchanged']:
            result.update(status='error', error='Frozen checkpoint file changed during diagnostic')
        if env is not None:
            try:env.close()
            except Exception as error:result.update(status='error', cleanup_error=str(error))
        if application is not None:
            try:application.close()
            except Exception as error:result.update(status='error', application_cleanup_error=str(error))
        result['wall_seconds'] = time.perf_counter()-started
        span = result.get('simulated_seconds_per_env',0.)
        result['rtf_per_env_total_wall'] = span/result['wall_seconds']
        result['aggregate_env_seconds_per_wall_second'] = 4*span/result['wall_seconds']
        result['rtf_per_env_rollout_wall'] = span/result['rollout_wall_seconds'] if result.get('rollout_wall_seconds') else None
        result['timing_notes'] = 'Host wall per callback; CUDA may overlap. Only rollout boundaries synchronize; no per-step synchronize. environment_step includes native physics/executor/evaluation/existing logs. Total includes startup and cleanup; per-env RTF uses one env span, aggregate rate is 4x. No optimizer/critic model/rollout memory.'
        write_json(folder/'summary.json', result)
        write_json(folder/'progress.json', {k:v for k,v in result.items() if k not in ('goals','configuration','action_summary')})
        print('C1_FROZEN_PHYSICS_DIAGNOSTIC_RESULT', json.dumps(dict(status=result['status'],
            decisions_completed=result['decisions_completed'], success_criteria=result.get('success_criteria'), output=str(folder))), flush=True)
    return 0 if result['status']=='completed' else 1


def format_progress(folder):
    folder=Path(folder)
    log_tail = ''
    log_path = folder/'stdout.log'
    if log_path.exists():
        with log_path.open('rb') as handle:
            handle.seek(max(0,log_path.stat().st_size-8192))
            log_tail='\nstdout tail:\n'+'\n'.join(handle.read().decode(errors='replace').splitlines()[-8:])
    path=folder/'progress.json'
    if not path.exists():
        return 'awaiting progress.json (startup or run not started)' + log_tail
    try:row=json.loads(path.read_text())
    except (ValueError,OSError) as error:return f'progress temporarily unavailable: {error}'
    lines=[f"status={row['status']} decisions={row.get('decisions_completed',0)} sim/env={row.get('simulated_seconds_per_env',0.):.3f}s wall={row.get('wall_seconds',0.):.1f}s"]
    for index, metrics in row.get('by_env',{}).items():
        a,d=metrics['align_faces'],metrics['dock']
        lines.append(f"env{index} {metrics['task']} align={a['START']}/{a['COMPLETE']}/{a['TIMEOUT']} (START/COMPLETE/TIMEOUT) dock={d['START']}/{d['COMPLETE']} claims={metrics.get('claim_acquire_goal_count',0)}/{metrics.get('claim_release_goal_count',0)} reopened={metrics.get('mask_reopen_after_release_count',0)} physical_errors={metrics.get('physical_error_observation_count',0)} collisions={metrics.get('collision_observation_count',0)}")
    if row.get('error'):lines.append('ERROR: '+row['error'])
    if row.get('errors'):lines.append('ERRORS: '+json.dumps(row['errors']))
    lines.append(f'progress age={max(0.,time.time()-path.stat().st_mtime):.1f}s')
    return '\n'.join(lines)+log_tail


def main(argv=None):
    args=parse_args(argv)
    if args.monitor_once:
        print(format_progress(args.output));return 0
    if args.preflight:
        print(json.dumps(preflight(args), indent=2));return 0
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
