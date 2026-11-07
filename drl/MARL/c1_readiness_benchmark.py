"""Offline accounting and optional sequential orchestration for native C1 smokes.

One active environment transition is one environment's learned decision, never
an agent action, inactive assembly row, physics substep, or vector iteration.
Importing this module does not import Isaac Lab, torch, or start a simulator.
Campaign exposure choices are planning scenarios, not convergence predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

DOMAINS = ('navigation', 'stairs', 'gap', 'manipulation')
DIFFICULTY_COUNTS = dict(easy=250, medium=400, hard=350)
FAIL_SENTINELS = ('NATIVE_C1_PRETRAIN_FAIL', 'NATIVE_C1_BENCHMARK_FAIL', 'NATIVE_C1_TRAIN_FAIL')


def _positive(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def load_run_result(path, *, returncode=0, stdout=''):
    """Require explicit machine-readable success; Isaac can exit zero on error."""
    for sentinel in FAIL_SENTINELS:
        if sentinel in stdout:
            raise ValueError(f'{sentinel} present in process output')
    if returncode:
        raise ValueError(f'runner exited {returncode}')
    result = json.loads(Path(path).read_text())
    if result.get('status') != 'pass':
        raise ValueError('runner result status must explicitly be pass')
    return result


def scaling_comparison(runs):
    rows = []
    for run in runs:
        timing = run.get('timings_s', {})
        transitions = run.get('active_marl_transitions')
        total, rollout = timing.get('total'), timing.get('rollout')
        decisions = run.get('vector_decision_steps', run.get('vector_steps'))
        row = dict(num_envs=run.get('num_envs'), status=run.get('status'),
                   requested_total_transitions=run.get('requested_total_transitions'),
                   active_marl_transitions=transitions,
                   active_transitions_per_s=transitions / rollout if _positive(transitions) and _positive(rollout) else None,
                   end_to_end_transitions_per_s=transitions / total if _positive(transitions) and _positive(total) else None,
                   vector_decisions_per_s=decisions / rollout if _positive(decisions) and _positive(rollout) else None,
                   error=run.get('error'))
        row.update({name + '_s': timing.get(name) for name in ('startup', 'assembly', 'reset', 'rollout', 'update', 'total')})
        row.update(run.get('memory_bytes', {}))
        rows.append(row)
    budgets = {row['active_marl_transitions'] for row in rows}
    comparable = bool(rows) and len(budgets) == 1 and all(
        row['status'] == 'pass' and _positive(row['active_marl_transitions']) and
        row['active_marl_transitions'] == row['requested_total_transitions'] for row in rows)
    return dict(comparable=comparable, reason=None if comparable else
                'Require successful runs at equal requested and actual active environment transitions.', rows=rows)


def estimate_campaigns(run, *, episode_exposures=None, checkpoint_every_transitions=None):
    """Extrapolate TRAIN catalog passes using measured natural episode lengths.

    Fail closed when measurements are absent or contain horizon-only episodes.
    The scenario interval (0.5x–2x) is an explicit planning sensitivity, never a
    statistical confidence interval; a brief smoke cannot establish convergence.
    """
    observations, missing = {}, []
    for domain in DOMAINS:
        source = run.get('domains', {}).get(domain, {})
        episodes = source.get('episodes', [])
        natural = [episode for episode in episodes if episode.get('termination') == 'terminated']
        valid = [episode for episode in natural if all(_positive(episode.get(key)) for key in
                                                       ('transitions', 'simulated_s', 'wall_s'))]
        observations[domain] = dict(natural_completed_episodes=len(natural),
            horizon_episodes=sum(episode.get('termination') == 'horizon' for episode in episodes),
            censored_episodes=source.get('censored_episodes'),
            mean_transitions=sum(episode['transitions'] for episode in valid) / len(valid) if valid else None,
            mean_simulated_s=sum(episode['simulated_s'] for episode in valid) / len(valid) if valid else None,
            mean_episode_wall_s=sum(episode['wall_s'] for episode in valid) / len(valid) if valid else None)
        if not valid or len(valid) != len(natural) or not _positive(source.get('reset_assembly_s')):
            missing.append(domain)
    timing = run.get('timings_s', {})
    active = run.get('active_marl_transitions')
    optimizer = run.get('optimizer', {})
    if run.get('status') != 'pass' or not _positive(active):
        missing.append('successful active transition measurement')
    for name in ('rollout', 'update'):
        if not _positive(timing.get(name)):
            missing.append(name + ' timing')
    for name in ('startup', 'assembly'):
        if not isinstance(timing.get(name), (float, int)) or not math.isfinite(timing[name]) or timing[name] < 0:
            missing.append(name + ' timing')
    for name in ('rollout_active_transitions', 'learning_epochs', 'mini_batches', 'completed_ppo_updates'):
        if not _positive(optimizer.get(name)):
            missing.append(name)
    if missing:
        return dict(available=False, reason='Insufficient measured evidence: ' + ', '.join(missing),
                    observations=observations, budgets={})
    exposures = episode_exposures or dict(SHORT=1, RECOMMENDED=3, EXTENDED=6)
    budgets = {}
    for label, exposure in exposures.items():
        if not _positive(exposure):
            raise ValueError('episode exposure must be positive')
        domain_episodes = {domain: math.ceil(sum(DIFFICULTY_COUNTS.values()) * exposure) for domain in DOMAINS}
        episodes_total = sum(domain_episodes.values())
        transitions = math.ceil(sum(domain_episodes[domain] * observations[domain]['mean_transitions'] for domain in DOMAINS))
        updates = math.ceil(transitions / optimizer['rollout_active_transitions'])
        # Reset overhead is measured wall occupancy per completed domain episode;
        # caller must remove overlap with rollout/update timing before supplying it.
        reset_s = sum(domain_episodes[domain] * run['domains'][domain]['reset_assembly_s'] for domain in DOMAINS)
        rollout_s = transitions * timing['rollout'] / active
        update_s = updates * timing['update'] / optimizer['completed_ppo_updates']
        seconds = timing['startup'] + timing['assembly'] + reset_s + rollout_s + update_s
        checkpoint_count = (math.ceil(transitions / checkpoint_every_transitions) + 1
                            if _positive(checkpoint_every_transitions) else None)
        checkpoint_bytes = run.get('checkpoint_bytes')
        budgets[label] = dict(episode_exposure_per_train_seed=exposure,
            active_environment_transitions=transitions, episodes_total=episodes_total,
            episodes_per_domain=domain_episodes,
            episodes_per_difficulty={difficulty: math.ceil(count * exposure) * len(DOMAINS)
                                     for difficulty, count in DIFFICULTY_COUNTS.items()},
            ppo_updates=updates,
            optimizer_steps_per_owner=updates * optimizer['learning_epochs'] * optimizer['mini_batches'],
            wall_seconds_point_estimate=seconds,
            timing_components_s=dict(startup=timing['startup'], assembly=timing['assembly'],
                                     reset=reset_s, rollout=rollout_s, update=update_s),
            uncertainty=dict(kind='scenario_range_not_confidence_interval', lower_s=seconds * .5,
                             upper_s=seconds * 2., note='Short-smoke extrapolation; task failures, curriculum and throughput can change.'),
            checkpoint_count=checkpoint_count,
            checkpoint_storage_bytes=checkpoint_count * checkpoint_bytes
            if checkpoint_count and _positive(checkpoint_bytes) else None)
    return dict(available=True, reason=None, observations=observations, budgets=budgets,
                caveats=['Catalog exposures are proposed episode budgets, not evidence of convergence.',
                         'Natural completed episodes only; horizon and in-flight censoring reported separately.',
                         'Reset wall occupancy must exclude overlap with rollout and update timing.',
                         'Optimizer step count is per owner; multiply only by actual independently optimized owners.',
                         'Difficulty counts use existing TRAIN catalog 250/400/350 per domain.'])


def estimate_horizon_bounds(run, *, production_horizon_s, decision_dt_s,
                            episode_scenarios, difficulty_counts=None,
                            throughput_run=None, checkpoint_every_transitions=None):
    """Planning envelopes from short diagnostic evidence, never natural-duration means.

    Each scenario specifies actual episode draws per domain (integer or mapping).
    A diagnostic timeout measures reset mechanics; the production horizon supplies
    the maximum active decisions per episode. Assembly latency is env occupancy,
    so fully parallel occupancy and serialized occupancy form separate scenarios.
    These are conditional scheduling/cost envelopes, not guaranteed runtime bounds.
    Prefer a no-reset equal-budget benchmark as throughput_run: learning rollout
    timing includes assembly and would conservatively charge that overlap twice.
    """
    baseline = throughput_run if throughput_run is not None else run
    timing, optimizer = baseline.get('timings_s', {}), baseline.get('optimizer', {})
    missing = []
    if not _positive(production_horizon_s) or not _positive(decision_dt_s):
        raise ValueError('production horizon and decision interval must be positive')
    horizon = math.ceil(production_horizon_s / decision_dt_s)
    counts = difficulty_counts if difficulty_counts is not None else DIFFICULTY_COUNTS
    if not counts or any(not _positive(count) for count in counts.values()):
        raise ValueError('difficulty catalog counts must be positive')
    catalog_size = sum(counts.values())
    if baseline.get('status') != 'pass' or not _positive(baseline.get('active_marl_transitions')):
        missing.append('successful active transition measurement')
    for name in ('rollout', 'update'):
        if not _positive(timing.get(name)):
            missing.append(name + ' timing')
    for name in ('startup', 'assembly'):
        value = run.get('timings_s', {}).get(name)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            missing.append(name + ' timing')
    for name in ('rollout_active_transitions', 'completed_ppo_updates', 'learning_epochs', 'mini_batches'):
        if not _positive(optimizer.get(name)):
            missing.append(name)
    slots, latencies = {}, {}
    records = run.get('assembly_records', [])
    for domain in DOMAINS:
        domain_records = [record for record in records if record.get('domain') == domain]
        envs = {record['env_index'] for record in domain_records if 'env_index' in record}
        samples = [record['wall_s'] for record in domain_records if _positive(record.get('wall_s'))]
        slots[domain], latencies[domain] = len(envs), samples
        if not envs or not samples:
            missing.append(domain + ' assembly latency/concurrency')
    if missing:
        return dict(available=False, reason='Insufficient measured evidence: ' + ', '.join(missing), budgets={})
    if sum(slots.values()) != run.get('num_envs'):
        return dict(available=False, reason='Assembly env concurrency does not match num_envs', budgets={})
    budgets = {}
    seconds_per_transition = timing['rollout'] / baseline['active_marl_transitions']
    seconds_per_update = timing['update'] / optimizer['completed_ppo_updates']
    initial_s = sum(run['timings_s'][name] for name in ('startup', 'assembly'))
    for label, scenario in episode_scenarios.items():
        episodes = dict.fromkeys(DOMAINS, scenario) if isinstance(scenario, int) else dict(scenario)
        if set(episodes) != set(DOMAINS) or any(not isinstance(n, int) or n < 1 for n in episodes.values()):
            raise ValueError('Each scenario requires positive integer episode counts for all four domains')
        total_episodes = sum(episodes.values())
        transitions_bounds = [total_episodes, horizon * total_episodes]
        update_bounds = [math.ceil(transitions / optimizer['rollout_active_transitions']) for transitions in transitions_bounds]
        # Initial parallel assembly is measured separately. The future reset
        # model charges only episodes beyond the initially assembled env slots.
        resets = {domain: max(0, episodes[domain] - slots[domain]) for domain in DOMAINS}
        reset_bounds = [max(resets[domain] * min(latencies[domain]) / slots[domain] for domain in DOMAINS),
                        sum(resets[domain] * max(latencies[domain]) for domain in DOMAINS)]
        wall_bounds = [initial_s + transitions_bounds[i] * seconds_per_transition +
                       update_bounds[i] * seconds_per_update + reset_bounds[i] for i in (0, 1)]
        checkpoint_bounds = ([math.ceil(n / checkpoint_every_transitions) + 1 for n in transitions_bounds]
                             if _positive(checkpoint_every_transitions) else None)
        checkpoint_bytes = run.get('checkpoint_bytes')
        budgets[label] = dict(episodes_total=total_episodes, episodes_per_domain=episodes,
            episodes_per_difficulty={difficulty: total_episodes * count / catalog_size for difficulty, count in counts.items()},
            expected_catalog_coverage_fraction_per_domain={domain: -math.expm1(-n / catalog_size) for domain, n in episodes.items()},
            expected_distinct_train_seeds_per_domain={domain: catalog_size * -math.expm1(-n / catalog_size) for domain, n in episodes.items()},
            active_environment_transitions_bounds=transitions_bounds,
            ppo_updates_bounds=update_bounds,
            optimizer_steps_per_owner_bounds=[n * optimizer['learning_epochs'] * optimizer['mini_batches'] for n in update_bounds],
            reset_wall_seconds_scenario_bounds=reset_bounds,
            wall_seconds_scenario_bounds=wall_bounds,
            checkpoint_count_bounds=checkpoint_bounds,
            checkpoint_storage_bytes_bounds=[n * checkpoint_bytes for n in checkpoint_bounds]
            if checkpoint_bounds and _positive(checkpoint_bytes) else None)
    return dict(available=True, reason=None, schema='mssr.c1_horizon_planning_bounds.v1',
        kind='conditional_planning_scenarios_not_measured_natural_episode_duration',
        production_horizon_s=production_horizon_s, production_horizon_decisions=horizon,
        decision_dt_s=decision_dt_s, domain_env_concurrency=slots,
        assembly_latency_samples_s=latencies, difficulty_catalog_counts=dict(counts),
        measured_rollout_seconds_per_active_transition=seconds_per_transition,
        measured_update_seconds_per_ppo_update=seconds_per_update, budgets=budgets,
        assumptions=['Episode decisions span 1 to the configured production horizon; short diagnostic endings do not estimate natural duration.',
            'Reset lower scenario assumes full parallelism within fixed domain slots; upper scenario serializes all measured maximum latencies.',
            'Observed assembly latency range and throughput are extrapolated; neither is a guaranteed future runtime limit.',
            'Rollout/update rates stay fixed at the measured hardware operating point.',
            'Difficulty draws follow supplied catalog proportions; coverage uses uniform independent sampling approximation 1-exp(-draws/catalog_size).',
            'Initial assembly is charged once; subsequent reset estimates exclude initially assembled environment slots.',
            'No statistical convergence or success-rate improvement is implied.'],
        rollout_reset_overlap_caveat=(None if throughput_run is not None else
            'Rate source includes any learning-time reset/assembly; adding reset occupancy is conservative and may double count overlap.'))


def collect_metadata(repository_root):
    """Capture exact worktree provenance and host/GPU metadata without assumptions."""
    root = Path(repository_root)
    def capture(command):
        try:
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
            return dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except OSError as error:
            return dict(error=str(error))
    diff = capture(['git', 'diff', '--binary', 'HEAD'])
    untracked = capture(['git', 'ls-files', '--others', '--exclude-standard'])
    hashes = {}
    for filename in untracked.get('stdout', '').splitlines():
        path = root / filename
        if path.is_file():
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(created_unix_s=time.time(), python=sys.version, executable=sys.executable,
                platform=platform.platform(), cpu=platform.processor(),
                git_head=capture(['git', 'rev-parse', 'HEAD']),
                git_status=capture(['git', 'status', '--porcelain=v1']), git_diff=diff,
                git_diff_sha256=hashlib.sha256(diff.get('stdout', '').encode()).hexdigest(),
                untracked_sha256=hashes,
                gpu=capture(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free', '--format=csv,noheader']),
                host_memory=capture(['free', '-b']))


def write_reports(output, runs, *, metadata=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    comparison = scaling_comparison(runs)
    report = dict(schema='mssr.c1_readiness_benchmark.v1', metadata=metadata, runs=runs,
                  scaling=comparison)
    (output / 'benchmark.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    fields = list(dict.fromkeys(key for row in comparison['rows'] for key in row))
    with (output / 'scaling.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison['rows'])
    (output / 'seed_manifest.json').write_text(json.dumps([run.get('seed_manifest') for run in runs], indent=2, allow_nan=False) + '\n')
    (output / 'campaign_estimates.json').write_text(json.dumps([estimate_campaigns(run) for run in runs], indent=2, allow_nan=False) + '\n')
    return report


def execute_run(command, output, *, timeout_s=None):
    """One subprocess at a time; retain failures and logs even after exit zero."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / 'result.json'
    if result_path.exists():
        raise ValueError(f'Refusing stale result file: {result_path}')
    started = time.monotonic()
    with (output / 'stdout.log').open('w') as stdout, (output / 'stderr.log').open('w') as stderr:
        try:
            process = subprocess.run(command, stdout=stdout, stderr=stderr, text=True, timeout=timeout_s, check=False)
            returncode = process.returncode
            launch_error = None
        except (subprocess.TimeoutExpired, OSError) as error:
            returncode, launch_error = None, str(error)
    logs = (output / 'stdout.log').read_text() + (output / 'stderr.log').read_text()
    try:
        if launch_error:
            raise ValueError(launch_error)
        result = load_run_result(result_path, returncode=returncode, stdout=logs)
    except (OSError, ValueError) as error:
        result = dict(status='fail', error=str(error))
    result.update(command=list(command), returncode=returncode, process_wall_s=time.monotonic() - started)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input', type=Path, nargs='*', default=[], help='existing run result JSONs; offline mode')
    parser.add_argument('--runner-mode', choices=('benchmark', 'smoke-train'), help='explicit opt-in to native runner subprocesses')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--num-envs', type=int, nargs='+', default=[4, 8, 16])
    parser.add_argument('--total-transitions', type=int, default=128)
    parser.add_argument('--timeout-s', type=float, default=1800.)
    parser.add_argument('--repository-root', type=Path, default=Path(__file__).resolve().parents[2])
    args, runner_extra = parser.parse_known_args(argv)
    if args.total_transitions <= 0 or any(count <= 0 for count in args.num_envs):
        parser.error('environment counts and active transition budget must be positive')
    if runner_extra and not args.runner_mode:
        parser.error('runner arguments require --runner-mode')
    metadata = collect_metadata(args.repository_root)
    runs = [json.loads(path.read_text()) for path in args.input]
    if args.runner_mode:
        for count in args.num_envs:
            destination = args.output / f'{args.runner_mode}_{count}'
            command = [args.python, '-m', 'drl.MARL.isaaclab_c1_runner', '--' + args.runner_mode,
                       '--num-envs', str(count), '--total-transitions', str(args.total_transitions),
                       '--output', str(destination), *runner_extra]
            run = execute_run(command, destination, timeout_s=args.timeout_s)
            run.setdefault('num_envs', count)
            run.setdefault('requested_total_transitions', args.total_transitions)
            runs.append(run)
            write_reports(args.output, runs, metadata=metadata)
    write_reports(args.output, runs, metadata=metadata)
    return 0 if runs and all(run.get('status') == 'pass' for run in runs) else 1


if __name__ == '__main__':
    raise SystemExit(main())
