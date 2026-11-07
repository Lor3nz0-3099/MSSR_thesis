"""Bounded native training through the unmodified SKRL SequentialTrainer.

Observers call original instance methods and restore them afterwards. Native
_get_dones owns transition identity and sends it to env.training_telemetry.
"""
from collections import defaultdict
import csv
import json
from pathlib import Path
import time
import traceback

import torch
from skrl.trainers.torch import SequentialTrainer

from drl.MARL.c1_training_telemetry import TrainingTelemetry
from drl.MARL.c1_incremental_diagnostics import StructuredSampleDiagnostics
from drl.MARL.c1_early_stopping import ValidationEarlyStopping, WallBudget, validate_frozen_evaluation


class BudgetReached(Exception):
    """Normal observer stop, raised after a completed PPO update."""


def _make_validation_callback(args, output):
    # Import only for an explicitly enabled campaign; benchmark/smoke paths
    # never instantiate or launch a validation child.
    from drl.MARL.c1_validation_callback import FixedBankValidationCallback
    return FixedBankValidationCallback(args, output)


def _coverage_ready(episodes, domains, minimum):
    return bool(domains) and all(len({int(episode['seed']) for episode in episodes
        if episode['domain'] == domain}) >= minimum for domain in domains)


def _synchronize(device):
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def enable_static_preparation_cache(env):
    """Opt native training into existing immutable catalog/layout caching.

    Physical features and normalized tensors are rebuilt every observation.
    Viewers keep their separately configured preprocessing path.
    """
    env._c1_runtime_static_caches = {}
    env._c1_static_preparation_enabled = True


def run_native_training(args, env, wrapper, learner, *, startup_s, assembly_s, run_started):
    """Train a bounded benchmark/smoke and persist both successful and failed evidence.

    Active budget is inspected only at update boundaries. Smoke coverage can stop
    sooner when minimum updates and distinct ended TRAIN seeds are observed; an
    exhausted active or vector budget without that coverage yields failure.
    """
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'updates.jsonl').write_text('')
    telemetry = TrainingTelemetry()
    env.training_telemetry = telemetry
    enable_static_preparation_cache(env)
    original = []
    diagnostics = dict(sidecar_alignment_checks=0, structured_samples_checked=0,
        nonfinite_structured_samples=0, rollout_active_transitions=0, actor_weights_changed=False)
    structured_diagnostics = StructuredSampleDiagnostics(diagnostics)
    counts = dict(vector_steps=0, updates=0, inactive_update_calls=0, actor_steps_in_session=0)
    update_times = []
    checkpoint_times, validation_times, validation_events = [], [], []
    campaign = bool(getattr(args, 'train', False))
    staged_schedule = staged_step = None
    if campaign and getattr(args, 'c1_staged_training', False):
        from drl.MARL.c1_longtrain_schedule import C1LongtrainSchedule
        staged_schedule = C1LongtrainSchedule()
    patience = int(getattr(args, 'early_stopping_patience', 0))
    validation_enabled = campaign and patience > 0
    max_wall_s = getattr(args, 'max_wall_seconds', None) if campaign else None
    if validation_enabled and max_wall_s is None:
        max_wall_s = 86400.
    wall_budget = WallBudget(max_wall_s) if max_wall_s is not None else None
    early_stopping = (ValidationEarlyStopping(patience,
        getattr(args, 'early_stopping_min_delta', 0.),
        preserve_domain_floor=True) if validation_enabled else None)
    checkpoint_reserve = float(getattr(args, 'checkpoint_reserve_seconds', 60.))
    if wall_budget is not None:
        wall_budget.record_costs(update_s=float(getattr(args, 'update_reserve_seconds', 600.)),
                                 checkpoint_s=checkpoint_reserve)
    validation_callback = None
    last_validation_elapsed = 0.
    wall_stop_info = None
    row_wall = defaultdict(float)
    episode_started = {}
    episode_wall = {}
    actor_before = [parameter.detach().cpu().clone() for parameter in learner.shared_actor.parameters()]
    stop_reason = 'vector_step_limit'
    error = None
    last_checkpoint = 0
    last_resume_artifact = None
    minimum_updates = int(getattr(args, 'min_updates', 2))
    minimum_seeds = int(getattr(args, 'min_completed_seeds_per_domain', 4))
    domains = tuple(getattr(args, 'domains', ()) or
        {getattr(task, 'domain', '') for task in getattr(env, 'tasks', ())})
    domains = tuple(domain for domain in domains if domain)
    initial_manifest = None
    sampler = next((getattr(env, name) for name in ('train_sampler', 'course_sampler', 'sampler')
        if getattr(env, name, None) is not None), None)
    if sampler is not None and hasattr(sampler, 'manifest'):
        initial_manifest = sampler.manifest()
    staged_resume_enabled = False
    if staged_schedule is not None:
        from drl.MARL.c1_longtrain_resume_artifacts import supports_staged_resume_artifact
        staged_resume_enabled = supports_staged_resume_artifact(initial_manifest)
    resume = getattr(args, '_c1_resume_context', None)
    if resume is not None:
        counts['updates'] = resume['counters']['completed_updates']
        counts['vector_steps'] = resume['counters']['vector_steps']
        if not validation_enabled or resume['validation_state'] is None:
            raise RuntimeError('Resume requires saved frozen-validation early-stopping state')
        restored_es = ValidationEarlyStopping.from_state_dict(resume['validation_state'])
        if (restored_es.patience != early_stopping.patience
                or restored_es.min_delta != early_stopping.min_delta
                or restored_es.preserve_domain_floor != early_stopping.preserve_domain_floor):
            raise ValueError('Resume validation controller configuration changed')
        early_stopping = restored_es

    critic_gate = None
    if staged_schedule is not None:
        from drl.MARL.c1_critic_quality_gate import C1CriticQualityGate
        resume_context = getattr(args, "_c1_resume_context", None)
        critic_gate = (
            C1CriticQualityGate.from_state_dict(
                resume_context["critic_gate_state"]
            )
            if resume_context is not None
            else C1CriticQualityGate()
        )
        if critic_gate.last_completed != counts["updates"]:
            raise RuntimeError(
                "Resume PPO update and critic gate disagree"
            )

    def elapsed_wall():
        return time.monotonic() - run_started

    def save_checkpoint(path):
        started = time.perf_counter()
        try:
            learner.save(str(path))
        finally:
            duration = time.perf_counter() - started
            checkpoint_times.append(duration)
            if wall_budget is not None:
                wall_budget.record_costs(checkpoint_s=max(checkpoint_reserve, duration))
        return path

    def check_wall():
        nonlocal wall_stop_info
        if wall_budget is not None:
            decision = wall_budget.decision(elapsed_wall())
            if decision['stop']:
                wall_stop_info = decision
                raise BudgetReached('wall_budget')

    def evaluate_validation(checkpoint):
        nonlocal validation_callback, last_validation_elapsed, wall_stop_info
        check_wall()
        if validation_callback is None:
            validation_callback = _make_validation_callback(args, output)
        # Optimizer state and frozen policy weights stay resident. Retire only
        # completed update gradients before measuring room for the child.
        seen = set()
        optimizers = list(getattr(learner, 'optimizers', {}).values())
        optimizers.extend(getattr(learner, name, None) for name in (
            'actor_optimizer', 'critic_optimizer', 'owner_critic_optimizer',
            'mean_field_critic_optimizer'))
        for optimizer in optimizers:
            if optimizer is not None and id(optimizer) not in seen:
                optimizer.zero_grad(set_to_none=True)
                seen.add(id(optimizer))
        if str(args.device).startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
        remaining = max(0., wall_budget.max_wall_s - elapsed_wall() -
                        (wall_budget.costs['checkpoint'] or checkpoint_reserve))
        started = time.perf_counter()
        try:
            evidence = validation_callback.evaluate(checkpoint, len(validation_events), remaining)
        finally:
            duration = time.perf_counter() - started
            validation_times.append(duration)
            wall_budget.record_costs(validation_s=duration)
        validation_events.append(evidence)
        last_validation_elapsed = elapsed_wall()
        if evidence.get('skipped'):
            wall_stop_info = dict(reason=evidence.get('reason'), remaining_s=remaining,
                                  elapsed_s=elapsed_wall())
            raise BudgetReached('validation_unavailable')
        verified = validate_frozen_evaluation(evidence, expected_seed_manifest=
            getattr(validation_callback, 'expected_seed_manifest', None))
        decision = early_stopping.update(verified['checkpoint_path'], verified['split'],
                                          evidence['score_by_domain'],
                                          success_by_domain=evidence.get('success_by_domain'))
        evidence['early_stopping'] = decision
        (output / 'early_stopping_state.json').write_text(json.dumps(early_stopping.state_dict(), indent=2)+'\n')
        if decision['stop']:
            raise BudgetReached('validation_patience')
        check_wall()

    def install(obj, name, replacement):
        original.append((obj, name, name in vars(obj), vars(obj).get(name)))
        setattr(obj, name, replacement)

    def timed(name, function):
        def observed(*positional, **keywords):
            started = time.perf_counter()
            try:
                return function(*positional, **keywords)
            finally:
                telemetry.record_timing(name, time.perf_counter()-started)
        return observed

    def checked_record(*positional, **keywords):
        result = original_record(*positional, **keywords)
        structured_diagnostics.observe(learner._structured_rollout_sidecar)
        return result

    def observed_step(*positional, **keywords):
        started = time.perf_counter()
        old_episodes = len(telemetry.episodes)
        before = {domain: row['row_transitions'] for domain, row in telemetry.domains.items()}
        result = original_step(*positional, **keywords)
        elapsed = time.perf_counter()-started
        telemetry.record_timing('environment_step', elapsed)
        counts['vector_steps'] += 1
        # Attribute this vector wall span by observed row share, never sum the
        # full parallel span into every domain.
        increments = {domain: row['row_transitions']-before.get(domain, 0)
            for domain, row in telemetry.domains.items()}
        total = sum(increments.values())
        for domain, increment in increments.items():
            row_wall[domain] += elapsed * increment / total if total else 0.
        def episode_key(episode):
            return (episode['env_index'], episode['domain'], episode['seed'], episode['generation'])
        for episode in list(telemetry.open_episodes.values()) + telemetry.episodes[old_episodes:]:
            key = episode_key(episode)
            episode_started.setdefault(key, started)
        for episode in telemetry.episodes[old_episodes:]:
            key = episode_key(episode)
            episode_wall[key] = time.perf_counter()-episode_started[key]
        return result

    def observed_update(*positional, **keywords):
        nonlocal staged_step
        samples = list(getattr(learner._structured_rollout_sidecar, '_samples', ()))
        altruistic = getattr(learner, '_altruistic_rollout_sidecar', None)
        if altruistic is not None and len(altruistic.flatten()) != len(samples):
            raise ValueError('structured/altruistic rollout sidecar alignment failed')
        for memory in learner.memories.values():
            if len(memory) != len(samples):
                raise ValueError('SKRL memory/structured rollout sidecar alignment failed')
        diagnostics['sidecar_alignment_checks'] += 1
        active_rows = sum(any(sample.owner_active_mask.values()) for sample in samples)
        diagnostics['rollout_active_transitions'] += active_rows
        _synchronize(args.device)
        if staged_schedule is not None:
            staged_step = staged_schedule.apply(
                learner, critic_gate.step_for_update(counts["updates"] + 1)
            )
        started = time.perf_counter()
        try:
            return original_update(*positional, **keywords)
        finally:
            _synchronize(args.device)
            elapsed = time.perf_counter()-started
            telemetry.record_timing('update', elapsed)
            update_times.append(elapsed)
            if wall_budget is not None:
                wall_budget.record_costs(update_s=max(
                    float(getattr(args, 'update_reserve_seconds', 600.)), 2 * elapsed))

    def observed_post(*positional, **keywords):
        nonlocal last_checkpoint, last_resume_artifact
        # Inspect even all-inactive assembly rows before SKRL can start an update.
        check_wall()
        before_updates = len(update_times)
        result = original_post(*positional, **keywords)
        if len(update_times) == before_updates:
            return result
        metrics = learner.last_update_metrics
        if not metrics:
            raise ValueError('update call has no diagnostics')
        actual_steps = sum(float(value) for name, value in metrics.items()
            if name.endswith('optimizer_steps'))
        if not metrics.get('minibatches', 0) or not actual_steps:
            counts['inactive_update_calls'] += 1
            telemetry.sample_resources(args.device)
            return result
        counts['updates'] += 1
        counts['actor_steps_in_session'] += int(metrics.get('actor_optimizer_steps', 0))
        if critic_gate is not None:
            critic_gate.observe_completed(counts['updates'], metrics)
            metrics.update(critic_gate.metrics())
        if staged_step is not None:
            metrics.update(
                c1_phase_id=staged_step.phase_id,
                c1_actor_lr=staged_step.actor_lr,
                c1_critic_lr=staged_step.critic_lr
            )
        telemetry.record_update(metrics)
        with (output / 'updates.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(update=counts['updates'],
                vector_step=counts['vector_steps'], active_marl_transitions=active_count(),
                c1_phase=staged_step.phase if staged_step is not None else None,
                metrics=metrics), allow_nan=False)+'\n')
        telemetry.sample_resources(args.device)

        # Legacy catalog-only resumable bundles require its sampler schema.
        # Phase-D snapshot training uses separate periodic model checkpoints;
        # its sampler is not yet compatible with historical resume artifacts.
        if staged_resume_enabled:
            update_number = counts["updates"]

            should_save = (
                update_number in (1, 4, 5, 12, 13)
                or update_number % 2 == 0
            )

            if should_save:
                from drl.MARL.c1_longtrain_resume_artifacts import (
                    save_update_artifact,
                )

                checkpoint_started = time.perf_counter()

                try:
                    last_resume_artifact = save_update_artifact(
                        args,
                        output,
                        learner,
                        sampler,
                        counts={
                            "completed_updates": update_number,
                            "vector_steps": counts["vector_steps"],
                            "smdp_transition_count": int(
                                getattr(
                                    learner,
                                    "smdp_transition_count",
                                    0,
                                )
                            ),
                            "smdp_event_step_count": int(
                                getattr(
                                    learner,
                                    "smdp_event_step_count",
                                    0,
                                )
                            ),
                            "smdp_policy_version": int(
                                getattr(
                                    learner,
                                    "smdp_policy_version",
                                    0,
                                )
                            ),
                        },
                        validation_state=(
                            early_stopping.state_dict()
                            if early_stopping is not None
                            else None
                        ),
                        critic_gate_state=critic_gate.state_dict(),
                    )
                finally:
                    duration = (
                        time.perf_counter()
                        - checkpoint_started
                    )

                    checkpoint_times.append(duration)

                    if wall_budget is not None:
                        wall_budget.record_costs(
                            checkpoint_s=max(
                                checkpoint_reserve,
                                duration,
                            )
                        )

        if critic_gate is not None and critic_gate.exhausted:
            raise BudgetReached('critic_quality_gate_unmet')
        active = active_count()
        cadence = int(getattr(args, 'checkpoint_every_transitions', 0))
        if cadence and active-last_checkpoint >= cadence:
            save_checkpoint(output / f'checkpoint_{active}.pt')
            last_checkpoint = active
        if any(check['nonfinite'] for check in telemetry.finite_checks.values()) or diagnostics['nonfinite_structured_samples']:
            raise ValueError('nonfinite rollout data observed')
        check_wall()
        if validation_enabled and elapsed_wall()-last_validation_elapsed >= float(getattr(args, 'validation_every_seconds', 14400.)):
            checkpoint = save_checkpoint(output / f'validation_candidate_{active}.pt')
            evaluate_validation(checkpoint)
        smoke_covered = _coverage_ready(telemetry.episodes, domains, minimum_seeds)
        if bool(getattr(args, 'smoke_train', False)) and counts['updates'] >= minimum_updates and smoke_covered:
            raise BudgetReached('smoke_coverage')
        if active >= int(args.total_transitions):
            raise BudgetReached('active_transition_budget')
        return result

    def active_count():
        if bool(getattr(learner, "_smdp_options_enabled", False)):
            return int(getattr(learner, "smdp_transition_count", 0))
        return sum(row['active_marl_transitions'] for row in telemetry.domains.values())

    original_record, original_step = learner.record_transition, wrapper.step
    original_update, original_post = learner.coordinated_update, learner.post_interaction
    install(learner, 'act', timed('actor_inference', learner.act))
    install(learner, 'record_transition', timed('record_transition', checked_record))
    install(learner, 'coordinated_update', observed_update)
    install(learner, 'post_interaction', observed_post)
    install(wrapper, 'step', observed_step)
    _synchronize(args.device)
    training_started = time.perf_counter()
    try:
        telemetry.sample_resources(args.device)
        if validation_enabled and resume is None and not getattr(args, 'c1_integration_probe', False):
            check_wall()
            checkpoint = save_checkpoint(output / 'validation_initial.pt')
            evaluate_validation(checkpoint)
        trainer = SequentialTrainer(env=wrapper, agents=learner, cfg=dict(
            timesteps=int(args.max_vector_steps), headless=bool(getattr(args, 'headless', True)),
            disable_progressbar=True, close_environment_at_exit=False))
        trainer.train()
    except BudgetReached as stopped:
        stop_reason = str(stopped)
    except Exception as caught:
        error = dict(type=type(caught).__name__, message=str(caught), traceback=traceback.format_exc())
        stop_reason = 'error'
    finally:
        _synchronize(args.device)
        training_wall = time.perf_counter()-training_started
        telemetry.sample_resources(args.device)
        for obj, name, existed, previous in reversed(original):
            if existed:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)
    after = list(learner.shared_actor.parameters())
    diagnostics['actor_weights_changed'] = len(after) == len(actor_before) and any(
        not torch.equal(before, parameter.detach().cpu()) for before, parameter in zip(actor_before, after))
    diagnostics['actor_weights_finite'] = all(bool(torch.isfinite(parameter).all().item()) for parameter in after)
    diagnostics['last_update_metrics'] = dict(learner.last_update_metrics)
    action_events = getattr(env, 'c1_action_events', None)
    provider = getattr(env, 'runtime_context_provider', None)
    if hasattr(provider, 'feasibility_metrics'):
        diagnostics['feasibility_cache'] = provider.feasibility_metrics()
    if action_events is not None:
        if 'feasibility_cache' in diagnostics:
            action_events.feasibility_cache_metrics = diagnostics['feasibility_cache']
        diagnostics['action_diagnostics'] = action_events.summary()
    training_checkpoint_wall = sum(checkpoint_times)
    checkpoint_path = output / 'diagnostic_final.pt'
    checkpoint_bytes = 0
    try:
        # One final artifact is outside rollout/update timing, including for the
        # benchmark so measured checkpoint size can support storage estimates.
        save_checkpoint(checkpoint_path)
        checkpoint_bytes = checkpoint_path.stat().st_size
    except Exception as caught:
        if error is None:
            error = dict(type=type(caught).__name__, message=str(caught), traceback=traceback.format_exc())
            stop_reason = 'checkpoint_error'
    snapshot = telemetry.snapshot()
    ended_coverage = _coverage_ready(telemetry.episodes, domains, minimum_seeds)
    finite = not any(row['nonfinite'] for row in snapshot['finite_checks'].values()) and not diagnostics['nonfinite_structured_samples']
    actor_change_expected = (
        staged_schedule is None
        or counts['actor_steps_in_session'] > 0
    )
    actor_change_valid = diagnostics['actor_weights_changed'] == actor_change_expected
    passed = error is None and counts['updates'] > 0 and finite and diagnostics['actor_weights_finite'] and actor_change_valid
    if getattr(args, 'smoke_train', False):
        passed = passed and counts['updates'] >= minimum_updates and ended_coverage
    elif not (campaign and stop_reason in ('wall_budget', 'validation_patience', 'validation_unavailable')):
        passed = passed and active_count() >= int(args.total_transitions)
    domain_results = {}
    for domain, row in snapshot['domains'].items():
        episodes = []
        for episode in snapshot['episodes']:
            if episode['domain'] == domain:
                key = (episode['env_index'], episode['domain'], episode['seed'], episode['generation'])
                wall = episode_wall.get(key, time.perf_counter()-episode_started.get(key, training_started))
                episodes.append(dict(episode, transitions=episode['active_length'],
                    simulated_s=episode['elapsed_s'], wall_s=wall,
                    termination='horizon' if episode['censored'] else 'terminated'))
        domain_results[domain] = dict(row, wall_s=row_wall[domain], episodes=episodes)
    reset_s = float(getattr(env, 'training_reset_wall_s', getattr(env, 'reset_wall_s', 0.)))
    timings = dict(startup=float(startup_s), assembly=float(assembly_s), reset=reset_s,
        rollout=max(0., training_wall-sum(update_times)-sum(validation_times)-training_checkpoint_wall),
        update=sum(update_times), validation=sum(validation_times), checkpoint=sum(checkpoint_times),
        total=time.monotonic()-run_started)
    diagnostics['runtime_static_cache'] = {
        str(index): cache.metrics()
        for index, cache in env._c1_runtime_static_caches.items()
    }
    result = dict(schema='mssr.c1_native_training.v1', status='pass' if passed else 'fail',
        training_harness='SKRL SequentialTrainer', num_envs=int(args.num_envs),
        active_marl_transitions=active_count(), row_transitions=snapshot['row_transitions'],
        smdp_event_steps=int(getattr(learner, "smdp_event_step_count", 0)),
        smdp_transitions=int(getattr(learner, "smdp_transition_count", 0)),
        smdp_policy_version=int(getattr(learner, "smdp_policy_version", 0)),
        smdp_max_policy_lag=int(getattr(learner, "smdp_max_policy_lag", 0)),
        requested_total_transitions=int(args.total_transitions), vector_steps=counts['vector_steps'],
        stop_reason=stop_reason, error=error, timings_s=timings, domains=domain_results,
        optimizer=dict(rollout_active_transitions=diagnostics['rollout_active_transitions'],
            learning_epochs=int(args.learning_epochs), mini_batches=int(args.mini_batches),
            completed_ppo_updates=counts['updates'], inactive_update_calls=counts['inactive_update_calls']),
        last_resumable_checkpoint=str(last_resume_artifact) if last_resume_artifact else None,
        resumed_from_checkpoint=resume['checkpoint_path'] if resume is not None else None,
        session_start_updates=resume['counters']['completed_updates'] if resume is not None else 0,
        session_start_smdp_transitions=resume['counters']['smdp_transition_count'] if resume is not None else 0,
        checkpoint_bytes=checkpoint_bytes, checkpoint_path=str(checkpoint_path), memory_bytes=snapshot['resources'],
        diagnostics=diagnostics, seed_manifest=initial_manifest,
        final_seed_manifest=sampler.manifest() if sampler is not None and hasattr(sampler, 'manifest') else None,
        assembly_records=getattr(env, 'assembly_records', []), smoke_seed_coverage_ready=ended_coverage,
        best_validation_checkpoint=early_stopping.best_checkpoint if early_stopping is not None else None,
        early_stopping_state=early_stopping.state_dict() if early_stopping is not None else None,
        wall_budget_state=wall_budget.state_dict() if wall_budget is not None else None,
        wall_stop_info=wall_stop_info, validation_events=validation_events,
        timing_notes='Update and total boundaries synchronize CUDA; callback timings measure host wall. Reset/reassembly overlaps rollout. Episode wall spans first observed row decision to end, including intervening updates; domain wall apportions vector stepping by row share.')
    if wall_budget is not None or early_stopping is not None:
        (output / 'campaign_control.json').write_text(json.dumps(dict(
            early_stopping_state=result['early_stopping_state'], wall_budget_state=result['wall_budget_state'],
            wall_stop_info=wall_stop_info, stop_reason=stop_reason,
            best_validation_checkpoint=result['best_validation_checkpoint']), indent=2, allow_nan=False)+'\n')
    (output / 'training_telemetry.json').write_text(json.dumps(snapshot, indent=2, allow_nan=False)+'\n')
    (output / 'episodes.jsonl').write_text(''.join(json.dumps(episode, allow_nan=False)+'\n' for episode in snapshot['episodes']))
    with (output / 'episodes.csv').open('w', newline='') as stream:
        if snapshot['episodes']:
            writer = csv.DictWriter(stream, fieldnames=list(snapshot['episodes'][0]))
            writer.writeheader()
            writer.writerows(snapshot['episodes'])
    (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result
