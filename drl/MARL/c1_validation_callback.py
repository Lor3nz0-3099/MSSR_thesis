"""Frozen validation child callback for a paused, wall-bounded C1 campaign.

No callback runs at import/construction. Training keeps its native scene and
learner; evaluate starts only its own finite fixed-bank native subprocess.
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from drl.MARL.c1_early_stopping import validate_frozen_evaluation
from drl.MARL.c1_train_sampling import catalog_seeds

DOMAINS = ('navigation', 'stairs', 'gap', 'manipulation')
VALIDATION_BANK = tuple(dict(env_index=i, domain=domain, seed=seed, split='validation')
    for i, (domain, seed) in enumerate(zip(DOMAINS, (8057, 13091, 17018, 21063))))
MODELS = ('shared_actor', 'shared_critic', 'altruistic_owner_critic', 'altruistic_mean_field_critic')


def _fixed_window_success(episodes, bank):
    """Average any-completed-success indicators per bank slot, then per domain."""
    by_index = {row['env_index']: row for row in bank}
    if len(by_index) != len(bank) or {row['domain'] for row in bank} != set(DOMAINS):
        raise ValueError('success bank must have unique slots and all four domains')
    succeeded, observed = set(), set()
    counts = {domain: dict(bank_slots=sum(row['domain'] == domain for row in bank),
        observed_bank_slots=0, successful_bank_slots=0, completed_episodes=0,
        successful_completed_episodes=0, censored_prefixes=0) for domain in DOMAINS}
    for episode in episodes:
        index = episode.get('env_index')
        expected = by_index.get(index)
        if expected is None or any(episode.get(key) != expected[key] for key in ('domain', 'seed')):
            raise ValueError('success episode identity differs from fixed validation bank')
        if any(type(episode.get(key)) is not bool for key in ('success', 'terminated', 'censored')):
            raise ValueError('success evidence requires explicit boolean episode outcomes')
        observed.add(index)
        row = counts[expected['domain']]
        completed = episode['terminated'] and not episode['censored']
        successful = completed and episode['success']
        row['completed_episodes'] += int(completed)
        row['successful_completed_episodes'] += int(successful)
        row['censored_prefixes'] += int(episode['censored'])
        if successful:
            succeeded.add(index)
    if observed != set(by_index):
        raise ValueError('success evidence must observe every validation bank slot')
    success = {}
    for domain, row in counts.items():
        indices = {index for index, item in by_index.items() if item['domain'] == domain}
        row['observed_bank_slots'] = len(observed & indices)
        row['successful_bank_slots'] = len(succeeded & indices)
        success[domain] = row['successful_bank_slots'] / row['bank_slots']
    if set(success) != set(DOMAINS) or any(not math.isfinite(value) or not 0 <= value <= 1
                                         for value in success.values()):
        raise ValueError('all four validation success scores must be finite in [0, 1]')
    return success, math.fsum(success.values()) / len(DOMAINS), counts


def _finite(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return float(value)


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _available_gpu_bytes(device='cuda:0'):
    index = str(device).split(':')[-1] if ':' in str(device) else '0'
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if visible != [''] and index.isdigit() and int(index) < len(visible):
        index = visible[int(index)]
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free',
        '--format=csv,noheader,nounits', '-i', index], capture_output=True, text=True,
        timeout=5, check=True)
    return int(result.stdout.strip()) * 1024**2


def _run_scoped(command, *, cwd, timeout):
    """A timeout signals only the new child's process group, including descendants."""
    with subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, start_new_session=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # The leader may have exited while a descendant still owns pipes.
            # Signal its independent group even if process.poll() is non-None.
            for sig, grace in ((signal.SIGTERM, 2.), (signal.SIGKILL, 2.)):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    process.communicate(timeout=grace)
                    break
                except subprocess.TimeoutExpired:
                    continue
            # Kill any descendant that ignored TERM even if the leader and
            # its captured pipes have already closed.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise subprocess.TimeoutExpired(command, timeout)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


class FixedBankValidationCallback:
    def __init__(self, args, output, *, launcher=None, memory_fn=None):
        self.args = args
        self.output = Path(output)
        self.launcher = launcher or _run_scoped
        self.memory_fn = memory_fn or (lambda: _available_gpu_bytes(getattr(args, 'device', 'cuda:0')))
        self.seed_manifest = deepcopy(list(VALIDATION_BANK))
        self.expected_seed_manifest = deepcopy(self.seed_manifest)

    def evaluate(self, checkpoint_path, event_index, remaining_wall_s):
        """Return frozen proof accepted by validate_frozen_evaluation.

        Scores sum active returns within one identical vector decision window
        per domain. Open/time-limited episodes contribute their observed prefix;
        the metric never extrapolates a full-episode return or rescales rewards.
        """
        if type(event_index) is not int or event_index < 0:
            raise ValueError('validation event_index must be a nonnegative integer')
        out = self.output / 'validation' / f'event_{event_index:06d}'
        out.mkdir(parents=True, exist_ok=False)
        evidence_path = out / 'validation_evidence.json'
        evidence = dict(schema='mssr.c1_fixed_bank_validation.v1', status='fail',
            checkpoint_path=str(Path(checkpoint_path).resolve()), split='validation',
            seed_manifest=deepcopy(list(VALIDATION_BANK)), evidence_path=str(evidence_path),
            available_gpu_bytes=None, peak_gpu_bytes=None)
        started = time.monotonic()
        try:
            remaining = _finite(remaining_wall_s, 'remaining wall time')
            reserve = _finite(getattr(self.args, 'checkpoint_reserve_seconds', 60.), 'checkpoint reserve')
            timeout = min(_finite(getattr(self.args, 'validation_timeout_seconds', 1800.),
                                  'validation timeout', positive=True), remaining - reserve)
            if timeout <= 0:
                evidence.update(status='skipped', skipped=True, reason='insufficient_wall_reserve')
                return evidence
            checkpoint = Path(checkpoint_path).resolve()
            if not checkpoint.is_file():
                raise ValueError('validation candidate checkpoint does not exist')
            candidate_sha = _sha_file(checkpoint)
            steps = getattr(self.args, 'validation_vector_steps', 300)
            if type(steps) is not int or steps <= 0:
                raise ValueError('validation_vector_steps must be a positive integer')
            repo = Path(self.args.repository_root).resolve()
            command = [sys.executable, '-m', 'drl.MARL.isaaclab_c1_runner', '--evaluate',
                '--evaluation-split', 'validation', '--train-sampler', 'fixed', '--num-envs', '4',
                '--domains', *DOMAINS, '--seeds', *(str(row['seed']) for row in VALIDATION_BANK),
                '--global-seed', str(self.args.global_seed), '--max-vector-steps', str(steps),
                '--episode-seconds', '60', '--checkpoint', str(Path(self.args.checkpoint).resolve()),
                '--policy-checkpoint', str(checkpoint), '--reward-config', str(Path(self.args.reward_config).resolve()),
                '--repository-root', str(repo), '--output', str(out.resolve()), '--headless']
            if getattr(self.args, 'device', None):
                command.extend(('--device', str(self.args.device)))
            evidence.update(command=command, candidate_sha256=candidate_sha,
                            timeout_seconds=timeout, checkpoint_reserve_seconds=reserve)
            for row in VALIDATION_BANK:
                if row['seed'] not in catalog_seeds(row['domain'], 'validation'):
                    raise ValueError('fixed bank seed is outside the authoritative validation catalog')
            available = _finite(self.memory_fn(), 'available GPU bytes', positive=True)
            required = _finite(getattr(self.args, 'validation_memory_bytes', 3 * 1024**3),
                               'validation memory headroom', positive=True)
            evidence['available_gpu_bytes'] = int(available)
            evidence['required_gpu_headroom_bytes'] = int(required)
            if available < required:
                evidence.update(status='skipped', skipped=True, reason='insufficient_gpu_headroom')
                return evidence
            # Include hashing/GPU-query startup in the caller's remaining budget.
            timeout = min(timeout, remaining - reserve - (time.monotonic() - started))
            if timeout <= 0:
                evidence.update(status='skipped', skipped=True, reason='insufficient_wall_reserve')
                return evidence
            evidence['timeout_seconds'] = timeout
            completed = self.launcher(command, cwd=str(repo), timeout=timeout)
            (out / 'stdout.log').write_text(getattr(completed, 'stdout', '') or '')
            (out / 'stderr.log').write_text(getattr(completed, 'stderr', '') or '')
            evidence['child_returncode'] = completed.returncode
            if completed.returncode != 0:
                raise RuntimeError(f'validation child failed with exit code {completed.returncode}')
            result = json.loads((out / 'result.json').read_text())
            memory = json.loads((out / 'gpu_monitor.json').read_text())
            if result.get('status') != 'pass' or result.get('split') != 'validation':
                raise ValueError('validation child did not report an explicit validation PASS')
            if result.get('seed_bank') != list(VALIDATION_BANK):
                raise ValueError('child seed bank differs from the fixed validation bank')
            if result.get('vector_steps') != steps:
                raise ValueError('validation child did not complete the fixed vector window')
            if result.get('optimizer_updates') != 0 or isinstance(result.get('optimizer_updates'), bool) or result.get('optimizer_attempts', 0) != 0:
                raise ValueError('frozen validation child attempted optimizer updates')
            loaded = result.get('policy_checkpoint', {})
            if loaded.get('sha256') != candidate_sha or Path(loaded.get('path', '')).resolve() != checkpoint:
                raise ValueError('validation child did not evaluate the exact candidate checkpoint')
            if _sha_file(checkpoint) != candidate_sha:
                raise ValueError('candidate checkpoint changed during frozen validation')
            proof = result.get('frozen_parameter_proof', {})
            models = proof.get('models', {})
            if not proof.get('all_unchanged') or set(models) != set(MODELS):
                raise ValueError('validation child lacks the complete four-model frozen proof')
            before, after = {}, {}
            for name in MODELS:
                row = models[name]
                old, new = row.get('before_sha256'), row.get('after_sha256')
                if not row.get('unchanged') or not isinstance(old, str) or len(old) != 64 or any(c not in '0123456789abcdef' for c in old) or new != old:
                    raise ValueError(f'validation child model changed: {name}')
                before[name], after[name] = old, new
            peak = _finite(memory.get('process_gpu_sampled_peak_bytes'), 'measured child peak GPU bytes', positive=True)
            if peak > available:
                raise ValueError('validation child peak GPU memory exceeded measured available memory')
            scores = {domain: 0. for domain in DOMAINS}
            seen = set()
            episodes = result.get('episodes', [])
            for episode in episodes:
                index = episode.get('env_index')
                if type(index) is not int or not 0 <= index < 4:
                    raise ValueError('validation episode has an invalid native environment slot')
                expected = VALIDATION_BANK[index]
                if episode.get('domain') != expected['domain'] or episode.get('seed') != expected['seed']:
                    raise ValueError('validation episode identity differs from fixed bank')
                value = episode.get('active_return')
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError('validation active return must be finite')
                scores[expected['domain']] += float(value)
                seen.add(expected['domain'])
            if seen != set(DOMAINS):
                raise ValueError('validation child has no observed return for every bank domain')
            success_by_domain, macro_success, success_counts = _fixed_window_success(episodes, VALIDATION_BANK)
            combine = lambda values: hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
            evidence.update(optimizer_updates=0, peak_gpu_bytes=int(peak), score_by_domain=scores,
                success_by_domain=success_by_domain, macro_success=macro_success,
                success_metric=dict(name='macro_fixed_window_bank_slot_success_lower_bound',
                    description='Observed-window success lower bound: each bank slot contributes one iff any completed noncensored episode succeeded during the fixed vector window. Domain values average bank-slot indicators; macro success averages the four domains equally. This is not an eventual full-task success rate.',
                    vector_steps=steps, source='frozen child result.json episodes: env_index, domain, seed, success, terminated, censored',
                    censored_prefixes_count_as_success=False,
                    censoring_rule='No completed success observed in a censored prefix contributes zero; no eventual-success claim or extrapolation.',
                    weighting='equal bank-slot weight within domain; equal domain weight in macro; episode counts never weight success',
                    source_counts_by_domain=success_counts),
                parameter_sha256_before=combine(before), parameter_sha256_after=combine(after),
                model_parameter_proof=models, episodes=episodes,
                metric=dict(name='fixed_window_sum_active_return', vector_steps=steps,
                    environments_per_domain=1, reward_rescaling='none',
                    censored_prefixes_included=True,
                    censoring_rule='Use observed active-return prefixes for time_limit and observation_end; never extrapolate full-episode return.'),
                child_result_path=str(out/'result.json'), gpu_monitor_path=str(out/'gpu_monitor.json'))
            validate_frozen_evaluation(evidence, expected_seed_manifest=list(VALIDATION_BANK))
            evidence['status'] = 'pass'
        except Exception as error:
            evidence['error'] = dict(type=type(error).__name__, message=str(error))
            raise
        finally:
            evidence['wall_seconds'] = time.monotonic() - started
            evidence_path.write_text(json.dumps(evidence, indent=2, allow_nan=False) + '\n')
        return evidence
