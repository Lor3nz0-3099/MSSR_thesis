"""User-run single-env frozen BC GUI. --preflight never imports Isaac.

No learner, critic model, optimizer or training memory. Acceleration is an
explicit viewer opt-in; --runtime-mode legacy is the manual timing control.
"""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
TASKS=('navigation','stairs','gap','manipulation')
INITIAL_SEEDS=dict(navigation=7033,stairs=12142,gap=16076,manipulation=20134)
TARGET_FILES=dict(navigation='smores_rc_car8.json',stairs='smores_snake8.json',
                  gap='smores_snake8.json',manipulation='smores_mobile_manipulator8.json')
TELEOP_CAMERA_RADIUS_M=1.1885363940578346
VIEWER_WIDTH=960
VIEWER_HEIGHT=540
STRUCTURAL_CONTROL_HZ=30.0
STRUCTURAL_FALLBACK_HZ=2.0


@dataclass(frozen=True)
class StructuralOptionState:
    active_goal_ids: tuple[str, ...]
    prepared_relations: tuple[tuple, ...]
    connection_count: int


class StructuralOptionScheduler:
    """Viewer-only option scheduler for long-running structural primitives.

    Physics/render still advance every 30 Hz control frame.  Only the expensive
    structured policy evaluation is event-driven.  Continuous low-level control
    is intentionally outside this scheduler and keeps its own fixed-rate clock.
    """

    def __init__(self, *, mode, control_hz=STRUCTURAL_CONTROL_HZ,
                 fallback_hz=STRUCTURAL_FALLBACK_HZ):
        if mode not in ('event', 'fixed'):
            raise ValueError('structural scheduler mode must be event or fixed')
        control_hz=float(control_hz);fallback_hz=float(fallback_hz)
        if not math.isfinite(control_hz) or control_hz<=0:
            raise ValueError('control_hz must be finite and positive')
        if not math.isfinite(fallback_hz) or fallback_hz<=0 or fallback_hz>control_hz:
            raise ValueError('fallback_hz must be finite, positive and <= control_hz')
        self.mode=mode
        self.fallback_frames=max(1, int(round(control_hz/fallback_hz)))
        self._force_decision=True
        self._idle_frames=0
        self._continuous_control=False

    def set_continuous_control(self, active):
        was_continuous=self._continuous_control
        self._continuous_control=bool(active)
        if self._continuous_control or was_continuous:
            # Fixed-rate resource control already evaluates every frame.  Do not
            # carry a stale structural wave-fill request across its boundary.
            self._force_decision=False
            self._idle_frames=0

    def should_decide(self, state):
        if self.mode=='fixed' or self._continuous_control:
            return True
        if self._force_decision:
            self._force_decision=False
            self._idle_frames=0
            return True
        if state.active_goal_ids:
            return False
        # An unresolved ALIGN->DOCK handoff is not generic idle.  Keep the
        # structural policy hot until a matching DOCK/re-ALIGN actually
        # starts; otherwise a single stochastic miss falls back to 2 Hz and
        # can strand a prepared wave for seconds.
        if any(row and not bool(row[-1]) for row in state.prepared_relations):
            self._idle_frames=0
            return True
        if self._idle_frames>=self.fallback_frames:
            self._idle_frames=0
            return True
        return False

    def observe_step(self, before, after):
        if self.mode=='fixed':
            return
        if self._continuous_control:
            self._force_decision=False
            self._idle_frames=0
            return
        if after.active_goal_ids:
            # Build a parallel structural wave over consecutive policy frames.
            # A first/next option joining the active set earns one more H1/H2
            # evaluation so another compatible option can join.  As soon as a
            # policy frame adds no new active goal, the wave is sealed and H1/H2
            # sleeps until the in-flight wave terminates or changes topology.
            added=set(after.active_goal_ids)-set(before.active_goal_ids)
            self._force_decision=bool(added)
            self._idle_frames=0
            return
        changed=(before.active_goal_ids!=after.active_goal_ids
                 or before.prepared_relations!=after.prepared_relations
                 or before.connection_count!=after.connection_count)
        if changed:
            self._force_decision=True
            self._idle_frames=0
        else:
            self._idle_frames+=1


def seal_active_align_waves(env):
    """Seal saturated C1 ALIGN waves without touching independent components."""

    rows=[]
    dispatcher=getattr(env, 'native_dispatcher', None)
    if dispatcher is None:
        return ()
    for env_index in range(int(getattr(env, 'num_envs', 1))):
        executor=dispatcher.executor_provider(env_index)
        seal=getattr(executor, 'seal_active_align_wave', None)
        if not callable(seal):
            continue
        for group in seal():
            rows.append((env_index, tuple(group)))
    return tuple(rows)


def structural_wave_fill_saturated(decide, before, after):
    if not decide or not after.active_goal_ids:
        return False
    return not (set(after.active_goal_ids) - set(before.active_goal_ids))


def structural_option_state(env):
    active=[];prepared=[];connections=0
    dispatcher=getattr(env, 'native_dispatcher', None)
    managers=getattr(env, 'docking_managers', ())
    for env_index in range(int(getattr(env, 'num_envs', 1))):
        if dispatcher is None:
            continue
        executor=dispatcher.executor_provider(env_index)
        for goal in tuple(getattr(executor, 'active_goals', ())):
            active.append(f'{env_index}:{getattr(goal, "goal_id", repr(goal))}')
        snapshot=getattr(executor, 'prepared_relation_snapshot', None)
        if callable(snapshot):
            for relation in snapshot():
                prepared.append((env_index, relation.source_module, relation.source_face,
                                 relation.target_module, relation.target_face,
                                 bool(getattr(relation, 'docked', False))))
        if env_index<len(managers):
            connections+=len(tuple(getattr(managers[env_index], 'connections', ())))
    return StructuralOptionState(tuple(sorted(active)), tuple(sorted(prepared)), connections)


def sampled_resource_control(sampled, contexts):
    """True when the sampled bundle contains any low-level/resource candidate."""
    for (decision, _output), context in zip(sampled, contexts, strict=True):
        resource_count=len(context.routing_snapshot.get('resource_candidates', ()))
        if resource_count<=0:
            continue
        presence=getattr(decision.action, 'presence', None)
        if presence is None:
            continue
        selected=presence[:resource_count]
        if bool(selected.any().item()):
            return True
    return False


def neutral_viewer_carrier(env, owners):
    """Return the DirectMARLEnv ABI carrier without staging a new proposal."""
    import torch
    device=getattr(env, 'device', 'cpu')
    return {owner:torch.zeros((int(getattr(env, 'num_envs', 1)), 1),
                              dtype=torch.float32, device=device)
            for owner in owners}


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=TASKS,default='navigation')
    parser.add_argument('--steps',type=int,default=1200,help='Control frames at30Hz; maximum1200 (40sim seconds)')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--global-seed',type=int,default=3103)
    parser.add_argument('--headless',action='store_true',help='Manual microbenchmark only; default GUI')
    parser.add_argument('--runtime-mode',choices=('legacy','accelerated'),default='accelerated')
    parser.add_argument('--structural-scheduler',choices=('event','fixed'),default=None,
                        help='Viewer structural decision clock; accelerated defaults event, legacy fixed')
    parser.add_argument('--status-seconds',type=float,default=2.,help='Slow wall-time telemetry, minimum0.5s; no pacing')
    parser.add_argument('--max-wall-seconds',type=float,default=86400.)
    parser.add_argument('--preflight',action='store_true',help='CPU configuration/hash validation only')
    args=parser.parse_args(argv)
    if not 1<=args.steps<=1200:parser.error('--steps requires1..1200')
    if not math.isfinite(args.status_seconds) or args.status_seconds<.5:parser.error('--status-seconds requires finite >=0.5')
    if not math.isfinite(args.max_wall_seconds) or not 0<args.max_wall_seconds<=86400:parser.error('--max-wall-seconds requires(0,86400]')
    if not 0<=args.global_seed<2**32:parser.error('--global-seed requires0..2**32-1')
    if args.structural_scheduler is None:
        args.structural_scheduler='event' if args.runtime_mode=='accelerated' else 'fixed'
    if args.output is None:
        args.output=ROOT/'logs/isaaclab_c1_native/redesign'/f'c1_gui_{args.task}_{time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())}'
    return args


def launcher_options(args):
    """Request the Kit window BEFORE AppLauncher selects its experience.

    This IsaacLab checkout forces headless without explicit Kit visualizer
    intent, even when headless=False. No camera sensor is needed for a viewport.
    """
    return dict(headless=bool(args.headless),device=args.device,enable_cameras=False,
                visualizer='none' if args.headless else 'kit',livestream=0)


def configure_viewer_display(application,settings,*,requested_gui):
    """Publish the resolved window state before SimulationContext caches it.

    The local AppLauncher publishes viewport/visualizer settings but does not
    publish /isaaclab/has_gui, which SimulationContext reads at construction.
    Derive it from the REAL SimulationApp config, never from GUI intent alone.
    """
    headless=bool(application.config['headless'])
    rendering=bool(settings.get('/isaaclab/render/active_viewport'))
    if requested_gui==headless:
        raise RuntimeError('Viewer GUI/headless request differs from resolved SimulationApp mode; '
                           'check HEADLESS environment override in runtime.log')
    if requested_gui and not rendering:
        raise RuntimeError('Kit GUI requested but resolved viewport rendering is disabled')
    settings.set_bool('/isaaclab/has_gui',not headless)
    if requested_gui:
        for key, value in (('/app/window/width', VIEWER_WIDTH),
                           ('/app/window/height', VIEWER_HEIGHT),
                           ('/app/renderer/resolution/width', VIEWER_WIDTH),
                           ('/app/renderer/resolution/height', VIEWER_HEIGHT)):
            settings.set_int(key, value)
    return dict(requested_gui=bool(requested_gui),effective_headless=headless,
                rendering_enabled=rendering,visualizer='kit' if requested_gui else 'none',
                window_width=VIEWER_WIDTH if requested_gui else None,
                window_height=VIEWER_HEIGHT if requested_gui else None)


def preflight(args):
    from drl.MARL.c1_structural_physics_diagnostic import configuration,source_hashes
    cfg=configuration(args)
    target_path=ROOT/'mssr_ws/src/mssr_expert/config'/TARGET_FILES[args.task]
    target=json.loads(target_path.read_text())
    from drl.MARL.c1_relation_diagnostics import TargetRelationDiagnostic
    TargetRelationDiagnostic(target,tuple(f'smores_{i:02d}' for i in range(1,9)))
    actual=hashlib.sha256(Path(cfg['checkpoint']).read_bytes()).hexdigest()
    if actual!=cfg['expected_bc_sha256']:raise ValueError('Frozen BC checkpoint hash mismatch')
    for key in ('reward_config','probe_config'):json.loads(Path(cfg[key]).read_text())
    from drl.MARL.c1_train_sampling import NativeC1TrainSampler
    seed=INITIAL_SEEDS[args.task]
    sampler=NativeC1TrainSampler((args.task,),args.global_seed,cycle=True,initial_seeds=(seed,))
    if sampler.sample(0).seed!=seed:raise ValueError('Viewer initial TRAIN seed not admitted')
    cfg.update(schema='mssr.c1.gui_viewer.v1',num_envs=1,tasks=[args.task],task=args.task,
        headless=args.headless,cameras=False,gui=not args.headless,runtime_mode=args.runtime_mode,
        requested_gui=not args.headless,effective_headless=bool(args.headless),
        rendering_enabled=not args.headless,launcher_options=launcher_options(args),
        actor_only=args.runtime_mode=='accelerated',readonly_state_cache=args.runtime_mode=='accelerated',
        training_memory_constructed=False,aggregate_environment_seconds=args.steps/30.,
        initial_train_seed=seed,target_graph_path=str(target_path),
        bc_sha256_verified=actual,status_wall_seconds=args.status_seconds,
        structural_scheduler=args.structural_scheduler,structural_fallback_hz=STRUCTURAL_FALLBACK_HZ,
        source_sha256=source_hashes(),note='Single-env viewer; TRAIN resampling and C1 episode semantics unchanged. Structural options scheduler is viewer-only; continuous control keeps its own fixed-rate clock.')
    return cfg


class ActorOnlyObservationBridge:
    """Viewer-local wrapper; never changes the default training bridge."""
    def __init__(self, bridge, generation):
        from drl.IL.bc_v1.runtime_static_cache import RuntimeStaticCache
        self.bridge,self.generation=bridge,generation
        self.static_cache=RuntimeStaticCache()

    def __getattr__(self,name):return getattr(self.bridge,name)

    def prepare_snapshot(self,*args,**kwargs):
        self.static_cache.set_epoch(self.generation())
        return self.bridge.prepare_snapshot(*args,**kwargs,include_privileged=False,static_cache=self.static_cache)


def sample_contexts(actor,contexts,owners,*,accelerated):
    """Batch-shaped boundary for Task5B; currently exact original env order."""
    import torch
    from drl.MARL.c1_feasibility import apply_availability
    from drl.MARL.policy import sample_marl_action_bundle
    results=[]
    with torch.no_grad():
        for context in contexts:
            kwargs=dict(graph=context.actor_graph,routing=context.routing_snapshot,possible_agents=owners)
            if accelerated:kwargs['lazy_payloads']=True
            output=actor.forward_all_owners(**kwargs)
            filtered=apply_availability(output,context.availability_snapshot)
            mask=torch.as_tensor(context.hard_mask,dtype=torch.bool,device=output['records'][0].presence_logit.device)
            mask=(mask|context.availability_snapshot.mask_tensor(device=mask.device)).detach().clone()
            action=sample_marl_action_bundle(**{key:filtered[key] for key in (
                'records','resource_payloads','structural_payloads','resource_log_std','structural_log_std')},
                hard_mask=mask,deterministic=False,bulk_host_reads=accelerated)
            results.append((SimpleNamespace(action=action,action_set_prediction_template=output['prediction_template']),output))
    return results


def configure_viewer_scene(env, *, headless):
    """Apply viewer-only lights/materials without touching headless runs."""
    if headless:
        return {'enabled': False, 'reason': 'headless'}
    from drl.MARL.c1_viewer_visuals import configure_viewer_visuals
    return configure_viewer_visuals(env.sim.stage, env.module_roots_by_env[0])


def summarize_timing_samples(samples):
    values = sorted(float(value) for value in samples)
    if not values:
        return dict(count=0, mean=0.0, p50=0.0, p95=0.0, max=0.0)
    count = len(values)
    p50 = (values[count // 2] if count % 2 else
           0.5 * (values[count // 2 - 1] + values[count // 2]))
    p95 = values[max(0, math.ceil(0.95 * count) - 1)]
    return dict(
        count=count, mean=sum(values) / count, p50=p50, p95=p95, max=values[-1]
    )


def rollout_viewer(env,actor,provider,recorder,owners,*,steps,max_wall_seconds,accelerated,application=None,structural_scheduler='fixed'):
    import torch
    timings=dict(context_s=0.,inference_sampling_s=0.,encode_projection_s=0.,environment_step_s=0.,diagnostic_logging_s=0.)
    timing_samples={key:[] for key in timings};decision_samples=[]
    scheduler=StructuralOptionScheduler(mode=structural_scheduler)
    idle_carrier=neutral_viewer_carrier(env,owners)
    start=time.perf_counter();completed=0;policy_evaluations=0;idle_control_frames=0;reason='horizon_reached'
    for step in range(steps):
        if application is not None and not application.is_running():reason='gui_closed';break
        if time.perf_counter()-start>=max_wall_seconds:reason='wall_limit';break
        recorder.decision_step=step
        decision_started=time.perf_counter()
        structural_before=structural_option_state(env)
        decide=scheduler.should_decide(structural_before)
        if decide:
            tick=time.perf_counter();contexts=provider.current_action_contexts();elapsed=time.perf_counter()-tick;timings['context_s']+=elapsed;timing_samples['context_s'].append(elapsed)
            tick=time.perf_counter();sampled=sample_contexts(actor,contexts,owners,accelerated=accelerated);elapsed=time.perf_counter()-tick;timings['inference_sampling_s']+=elapsed;timing_samples['inference_sampling_s'].append(elapsed)
            tick=time.perf_counter()
            for i,(_,output) in enumerate(sampled):
                recorder.observe_context(i,env.generation[i],output['records'],contexts[i].availability_snapshot,env._sim_step_counter*env.physics_dt)
            elapsed=time.perf_counter()-tick;timings['diagnostic_logging_s']+=elapsed;timing_samples['diagnostic_logging_s'].append(elapsed)
            tick=time.perf_counter();carrier=provider.encode_env_actions([d for d,_ in sampled]);elapsed=time.perf_counter()-tick;timings['encode_projection_s']+=elapsed;timing_samples['encode_projection_s'].append(elapsed)
            scheduler.set_continuous_control(sampled_resource_control(sampled,contexts))
            policy_evaluations+=1
        else:
            carrier=idle_carrier
            idle_control_frames+=1
        tick=time.perf_counter()
        with torch.no_grad():_,rewards,*_=env.step(carrier)
        elapsed=time.perf_counter()-tick;timings['environment_step_s']+=elapsed;timing_samples['environment_step_s'].append(elapsed)
        structural_after=structural_option_state(env)
        scheduler.observe_step(structural_before,structural_after)
        if structural_wave_fill_saturated(decide, structural_before, structural_after):
            seal_active_align_waves(env)
        if any(not torch.isfinite(r).all() for r in rewards.values()):raise RuntimeError('Nonfinite unchanged C1 reward')
        tick=time.perf_counter()
        for row in list(env.c1_runtime.recent_transitions)[-env.num_envs:]:
            generation,evaluator=env.c1_runtime.evaluators[row['env_index']]
            if generation!=row['generation']:raise RuntimeError('Native transition generation mismatch')
            recorder.observe_transition(row,evaluator.last_frame)
        provider.reset()  # no terminal critic/storage; retain native claim caches
        completed+=1
        recorder.slow_status(env._sim_step_counter*env.physics_dt)
        elapsed=time.perf_counter()-tick;timings['diagnostic_logging_s']+=elapsed;timing_samples['diagnostic_logging_s'].append(elapsed)
        decision_samples.append(time.perf_counter()-decision_started)
    if reason=='horizon_reached':
        # Final real post-release mask, without another actor forward or RNG draw.
        from drl.MARL.policy import bind_marl_candidates
        tick=time.perf_counter()
        for i,context in enumerate(provider.current_action_contexts()):
            count=len(context.routing_snapshot['resource_candidates'])+len(context.routing_snapshot['structural_bindings'])
            records=bind_marl_candidates(routing=context.routing_snapshot,presence_logits=torch.zeros(count),presence_threshold_logit=4.5)
            recorder.observe_context(i,env.generation[i],records,context.availability_snapshot,env._sim_step_counter*env.physics_dt,count_masks=False)
        timings['final_context_and_mask_s']=time.perf_counter()-tick
    policy_keys=('context_s','inference_sampling_s','encode_projection_s')
    return dict(decisions_completed=completed,control_frames_completed=completed,
        policy_evaluations=policy_evaluations,idle_control_frames=idle_control_frames,
        structural_scheduler=structural_scheduler,cutoff_reason=reason,timings_s=timings,
        mean_decision_timings_s={k:v/completed for k,v in timings.items() if k!='final_context_and_mask_s'} if completed else {},
        mean_policy_timings_s={k:timings[k]/policy_evaluations for k in policy_keys} if policy_evaluations else {},
        timing_distributions_s={k:summarize_timing_samples(v) for k,v in timing_samples.items()},
        decision_wall_distribution_s=summarize_timing_samples(decision_samples),
        rollout_wall_seconds=time.perf_counter()-start)


def camera_for_positions(positions):
    if not positions:raise ValueError('Camera needs module positions')
    low=[min(float(p[k]) for p in positions) for k in range(3)]
    high=[max(float(p[k]) for p in positions) for k in range(3)]
    center=tuple((a+b)/2 for a,b in zip(low,high))
    extent=max(math.sqrt(sum((float(p[k])-center[k])**2 for k in range(3))) for p in positions)
    radius=max(TELEOP_CAMERA_RADIUS_M,3.0*extent)
    direction=(1.30,-1.20,0.85)
    norm=math.sqrt(sum(value*value for value in direction))
    eye=tuple(center[k]+radius*direction[k]/norm for k in range(3))
    return eye,center


@contextmanager
def native_log(path):
    """Keep Kit/native output out of the sparse terminal; preserve it on disk."""
    sys.stdout.flush();sys.stderr.flush()
    saved_out,saved_err=os.dup(1),os.dup(2)
    console=os.fdopen(os.dup(saved_out),'w',buffering=1)
    try:
        with Path(path).open('x') as handle:
            os.dup2(handle.fileno(),1);os.dup2(handle.fileno(),2)
            yield console
    finally:
        sys.stdout.flush();sys.stderr.flush()
        os.dup2(saved_out,1);os.dup2(saved_err,2)
        os.close(saved_out);os.close(saved_err);console.close()


def finalize_viewer(*,env,application,actor,recorder,before,sim_start,result,folder,write_json,actor_digest,before_app_close=None):
    """Diagnostic failures must never bypass native resource cleanup.

    Preserve the original rollout error; report each secondary failure separately.
    No observer is allowed to synthesize a terminal during normal close.
    """
    def attempt(operation,callback):
        try:
            return callback()
        except BaseException as error:
            result['status']='error'
            result.setdefault('error',f'Finalization failed: {operation}: {error}')
            result.setdefault('finalization_errors',[]).append(dict(operation=operation,
                error=str(error),exception_type=type(error).__name__))
            return None
    try:
        if env is not None and sim_start is not None:
            stamp=attempt('final_timestamp',lambda:env._sim_step_counter*env.physics_dt)
            if stamp is not None:
                result['simulated_seconds']=stamp-sim_start
                if recorder is not None:
                    attempt('censor',lambda:recorder.censor_open(stamp,result.get('cutoff_reason',result['status'])))
                    attempt('status',lambda:recorder.slow_status(stamp,force=True))
        if recorder is not None:
            attempt('detach',recorder.detach_environment)
            attempt('recorder_close',recorder.close)
            counts=attempt('event_counts',lambda:dict(recorder.event_counts))
            if counts is not None:result['events']=counts
        if actor is not None and before is not None:
            def verify_actor():
                result['actor_unchanged']=actor_digest(actor)==before
                if not result['actor_unchanged'] or any(p.grad is not None for p in actor.parameters()):
                    raise RuntimeError('Frozen actor state/gradient contract violated')
            attempt('digest',verify_actor)
        if env is not None:
            attempt('reset_records',lambda:write_json(folder/'reset_records.json',dict(records=getattr(env,'reset_records',[]))))
    finally:
        # Nested cleanup covers even an unexpected failure outside a diagnostic
        # callback. Application close remains unconditional if env.close fails.
        try:
            if env is not None:attempt('env_close',env.close)
        finally:
            try:
                if before_app_close is not None:attempt('before_app_close',before_app_close)
            finally:
                if application is not None:attempt('app_close',application.close)


def run(args):
    config=preflight(args);folder=args.output.resolve();folder.mkdir(parents=True,exist_ok=False)
    from drl.MARL.c1_structural_physics_diagnostic import write_json,build_frozen_actor,actor_digest
    with (folder/'RUN_STARTED.json').open('x') as handle:json.dump(dict(started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),handle)
    write_json(folder/'run_config.json',{k:v for k,v in config.items() if k!='source_sha256'})
    write_json(folder/'source_sha256.json',config['source_sha256'])
    env=application=actor=recorder=None;before=None;sim_start=None;start=time.perf_counter()
    result=dict(status='error',task=args.task,num_envs=1,optimizer_updates=0,backward_calls=0,
        learner_constructed=False,critic_model_constructed=False,optimizer_constructed=False,
        training_memory_constructed=False,expert_actions=0,decisions_completed=0,runtime_mode=args.runtime_mode)
    with native_log(folder/'runtime.log') as console:
        print(f'C1 GUI task={args.task} num_envs=1 training=OFF output={folder}',file=console,flush=True)
        try:
            from isaaclab.app import AppLauncher
            application=AppLauncher(**launcher_options(args)).app
            from isaaclab.app.settings_manager import get_settings_manager
            display=configure_viewer_display(application,get_settings_manager(),requested_gui=not args.headless)
            write_json(folder/'launcher_settings.json',display)
            print('KIT MODE '+ ' '.join(f'{k}={v}' for k,v in display.items()),file=console,flush=True)
            import torch
            from drl.MARL.c1_runner import build_observation_bridge
            from drl.MARL.isaaclab_c1_runner import seed_native_training
            from drl.MARL.isaaclab_c1_env import MSSRC1IsaacLabEnv,make_c1_isaaclab_env_cfg
            from drl.MARL.c1_relation_diagnostics import TargetRelationDiagnostic
            from drl.MARL.c1_viewer_events import ViewerEvents
            from drl.MARL.c1_action_events import C1ActionEvents
            seed_native_training(args.global_seed)
            owners=tuple(f'smores_{i:02d}' for i in range(1,9));accelerated=args.runtime_mode=='accelerated'
            bridge=build_observation_bridge(ROOT);actor=build_frozen_actor(bridge,config);before=actor_digest(actor)
            cfg=make_c1_isaaclab_env_cfg(possible_agents=owners,num_envs=1,device=args.device)
            cfg.initialization_mode='loose_modules';cfg.domains=(args.task,);cfg.seed=args.global_seed
            cfg.seeds=(config['initial_train_seed'],);cfg.resample_train=True;cfg.train_sampler_seed=args.global_seed;cfg.smoke_seed_cycle=True
            cfg.reward_config_path=config['reward_config'];cfg.c1_evidence_directory=None
            cfg.c1_readonly_state_cache=accelerated
            env=MSSRC1IsaacLabEnv(cfg);env.reset()
            viewer_visuals=configure_viewer_scene(env,headless=args.headless)
            if tuple(env.seed_by_env)!=(config['initial_train_seed'],):raise RuntimeError('Unexpected viewer initial seed')
            # Keep reward counters, avoid constructing detailed masked-candidate log payloads.
            env.c1_action_events=C1ActionEvents(None,masked_detail_limit=0,threshold_logit=4.5)
            if accelerated:bridge=ActorOnlyObservationBridge(bridge,lambda:env.generation[0])
            native=env.initialize_policy_bridge(observation_bridge=bridge,actor_only=accelerated,bulk_host_reads=accelerated)
            classifier=TargetRelationDiagnostic(json.loads(Path(config['target_graph_path']).read_text()),owners)
            write_json(folder/'target_topology.json',dict(target=classifier.target,anchor={classifier.physical_root:classifier.root},
                note='diagnostic-only partial mapping, no fixed assignment of remaining modules'))
            recorder=ViewerEvents(folder,args.task,classifier=classifier,console=console,status_seconds=args.status_seconds,low_latency=True)
            recorder.attach_environment(env)
            raw=env.cached_raw_snapshot(0)
            eye,lookat=camera_for_positions([m['position'] for m in raw['state']['modules']])
            if not args.headless:env.sim.set_camera_view(eye=eye,target=lookat)
            write_json(folder/'viewer_settings.json',dict(eye=eye,lookat=lookat,has_gui=bool(env.sim.has_gui),
                rendering=bool(env.sim.is_rendering),physics_dt=env.physics_dt,decimation=cfg.decimation,
                simple_visuals=True,viewer_visuals=viewer_visuals,readonly_state_cache=accelerated,actor_only=accelerated))
            if not args.headless and not env.sim.has_gui:raise RuntimeError('GUI requested but Isaac has no GUI')
            sim_start=env._sim_step_counter*env.physics_dt
            if str(args.device).startswith('cuda'):torch.cuda.synchronize(args.device)
            result['startup_wall_seconds']=time.perf_counter()-start
            result.update(rollout_viewer(env,actor,native.provider,recorder,owners,steps=args.steps,
                max_wall_seconds=max(0.,args.max_wall_seconds-(time.perf_counter()-start)),accelerated=accelerated,
                application=application,structural_scheduler=args.structural_scheduler))
            if str(args.device).startswith('cuda'):torch.cuda.synchronize(args.device)
            result['feasibility_cache']=native.provider.feasibility_metrics()
            if accelerated:result['static_cache']=bridge.static_cache.metrics()
            result['status']='completed' if result['cutoff_reason']=='horizon_reached' else result['cutoff_reason']
            if env.assembly_records:raise RuntimeError('Expert assembly entered frozen viewer')
        except BaseException as error:
            result.update(status='error',error=str(error),traceback=traceback.format_exc())
            print(f"VIEWER ERROR: {error}; details={folder/'summary.json'}",file=console,flush=True)
        finally:
            def persist_result(announce=True):
                result['wall_seconds']=time.perf_counter()-start
                try:
                    result['sources_unchanged']=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in config['source_sha256'].items())
                    result['checkpoint_unchanged']=hashlib.sha256(Path(config['checkpoint']).read_bytes()).hexdigest()==config['expected_bc_sha256']
                    if not result['sources_unchanged'] or not result['checkpoint_unchanged']:
                        result['status']='error';result.setdefault('error','Source/checkpoint changed during viewer')
                except BaseException as error:
                    result['status']='error';result.setdefault('error',str(error))
                    result.setdefault('finalization_errors',[]).append(dict(operation='source_checkpoint_verification',error=str(error)))
                result['timing_notes']='Host wall callbacks; CUDA overlap possible; synchronize only rollout boundaries. Summary is persisted before Kit close, which may terminate the process; wall_seconds can exclude final Kit shutdown. Use matched manual legacy/accelerated micro-runs, not old4env timings.'
                try:write_json(folder/'summary.json',result)
                except BaseException as error:
                    result['status']='error';result.setdefault('error',str(error))
                    print(f'VIEWER SUMMARY WRITE FAILED: {error}',file=console,flush=True)
                if announce:print(f"VIEWER END status={result['status']} decisions={result['decisions_completed']} output={folder}",file=console,flush=True)
            finalize_viewer(env=env,application=application,actor=actor,recorder=recorder,
                before=before,sim_start=sim_start,result=result,folder=folder,
                write_json=write_json,actor_digest=actor_digest,before_app_close=persist_result)
            # Graceful-close versions return; refresh any close-error evidence.
            # Fast shutdown versions already have a complete durable summary.
            persist_result(announce=False)
    return 1 if result['status']=='error' else 0


def main(argv=None):
    args=parse_args(argv)
    if args.preflight:
        cfg=preflight(args);print(json.dumps({k:v for k,v in cfg.items() if k!='source_sha256'},indent=2));return 0
    return run(args)


if __name__=='__main__':raise SystemExit(main())
