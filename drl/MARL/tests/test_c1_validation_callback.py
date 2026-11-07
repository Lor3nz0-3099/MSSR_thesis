"""Bounded fixed-bank child validation without launching Isaac."""
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from drl.MARL.c1_validation_callback import FixedBankValidationCallback, VALIDATION_BANK
from drl.MARL.c1_early_stopping import validate_frozen_evaluation

GIB = 1024**3
MODELS = ('shared_actor', 'shared_critic', 'altruistic_owner_critic', 'altruistic_mean_field_critic')


def setup(tmp_path, bad=None):
    checkpoint = tmp_path / 'candidate.pt'
    checkpoint.write_bytes(b'candidate')
    args = SimpleNamespace(repository_root=tmp_path, checkpoint=tmp_path/'bc.pt',
        reward_config=tmp_path/'reward.json', global_seed=42, validation_vector_steps=7,
        validation_timeout_seconds=100, checkpoint_reserve_seconds=20)
    calls = []
    def launcher(command, **kwargs):
        calls.append((command, kwargs))
        out = Path(command[command.index('--output') + 1])
        if bad == 'timeout':
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        if bad == 'oom':
            return subprocess.CompletedProcess(command, 1, '', 'CUDA out of memory')
        result = dict(status='pass', split='validation', optimizer_updates=0, optimizer_attempts=0,
            vector_steps=7, seed_bank=[dict(row) for row in VALIDATION_BANK],
            policy_checkpoint=dict(path=str(checkpoint), sha256=hashlib.sha256(b'candidate').hexdigest()),
            frozen_parameter_proof=dict(all_unchanged=True, models={name:
                dict(unchanged=True, before_sha256='a'*64, after_sha256='a'*64) for name in MODELS}),
            episodes=[dict(**row, active_return=float(i+1), active_length=7, success=False, terminated=False, censored=True,
                           censor_reason='observation_end') for i, row in enumerate(VALIDATION_BANK)])
        if bad == 'train_bank': result['seed_bank'][0]['seed'] = 7000
        if bad == 'optimizer': result['optimizer_updates'] = 1
        if bad == 'hash': result['frozen_parameter_proof']['models']['shared_actor']['after_sha256'] = 'b'*64
        (out/'result.json').write_text(json.dumps(result))
        (out/'gpu_monitor.json').write_text(json.dumps(dict(process_gpu_sampled_peak_bytes=5*GIB if bad == 'peak' else GIB)))
        return subprocess.CompletedProcess(command, 0, '', '')
    callback = FixedBankValidationCallback(args, tmp_path/'run', launcher=launcher, memory_fn=lambda: 4*GIB)
    return callback, checkpoint, calls


def test_callback_adapts_actual_child_formats_with_fixed_window_metric(tmp_path):
    callback, checkpoint, calls = setup(tmp_path)
    evidence = callback.evaluate(checkpoint, 0, 90)
    accepted = validate_frozen_evaluation(evidence, expected_seed_manifest=list(VALIDATION_BANK))
    assert accepted['balanced_score'] == 2.5
    assert evidence['score_by_domain'] == dict(navigation=1., stairs=2., gap=3., manipulation=4.)
    assert evidence['metric']['censored_prefixes_included']
    assert evidence['peak_gpu_bytes'] == GIB
    command, kwargs = calls[0]
    assert '--evaluate' in command and '--train' not in command
    assert command[command.index('--seeds')+1:command.index('--seeds')+5] == ['8057','13091','17018','21063']
    assert 0 < kwargs['timeout'] <= 70
    assert kwargs['cwd'] == str(tmp_path)
    assert 'shell' not in kwargs or kwargs['shell'] is False
    assert json.loads(Path(evidence['evidence_path']).read_text()) == evidence


@pytest.mark.parametrize('bad', ['train_bank', 'optimizer', 'hash', 'oom', 'timeout', 'peak'])
def test_invalid_child_proof_is_rejected_and_failure_saved(tmp_path, bad):
    callback, checkpoint, calls = setup(tmp_path, bad)
    with pytest.raises((RuntimeError, ValueError, subprocess.TimeoutExpired)):
        callback.evaluate(checkpoint, 0, 90)
    evidence_path = tmp_path/'run/validation/event_000000/validation_evidence.json'
    assert json.loads(evidence_path.read_text())['status'] == 'fail'


def test_no_headroom_or_wall_reserve_prevents_child_launch(tmp_path):
    callback, checkpoint, calls = setup(tmp_path)
    callback.memory_fn = lambda: GIB
    skipped = callback.evaluate(checkpoint, 0, 90)
    assert skipped['skipped'] and skipped['reason'] == 'insufficient_gpu_headroom'
    assert not calls
    callback.memory_fn = lambda: 4*GIB
    skipped = callback.evaluate(checkpoint, 1, 10)
    assert skipped['skipped'] and skipped['reason'] == 'insufficient_wall_reserve'
    assert not calls


def test_scoped_timeout_kills_its_descendant_and_preserves_unrelated_process(tmp_path):
    import os
    import sys
    import time
    from drl.MARL.c1_validation_callback import _run_scoped
    marker = tmp_path/'descendant.pid'
    child_code = (
        "import os,signal,time,pathlib;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        "os.close(1);os.close(2);"
        f"pathlib.Path({str(marker)!r}).write_text(str(os.getpid()));"
        "time.sleep(30)"
    )
    leader_code = f'import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",{child_code!r}]);time.sleep(30)'
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            _run_scoped([sys.executable, '-c', leader_code], cwd=str(tmp_path), timeout=.8)
        assert unrelated.poll() is None
        assert marker.exists()
        pid = int(marker.read_text())
        def alive():
            try:
                return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1][0] != 'Z'
            except FileNotFoundError:
                return False
        deadline = time.monotonic() + 2
        while alive() and time.monotonic() < deadline:
            time.sleep(.02)
        assert not alive()
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_callback_reports_four_domain_observed_window_success_lower_bound(tmp_path):
    callback, checkpoint, calls = setup(tmp_path)
    original = callback.launcher
    def child(command, **kwargs):
        completed = original(command, **kwargs)
        out = Path(command[command.index('--output') + 1])
        result = json.loads((out/'result.json').read_text())
        for i, episode in enumerate(result['episodes']):
            episode.update(success=i < 2, terminated=i < 3, censored=i == 3)
        (out/'result.json').write_text(json.dumps(result))
        return completed
    callback.launcher = child
    evidence = callback.evaluate(checkpoint, 0, 90)
    assert evidence['success_by_domain'] == dict(navigation=1., stairs=1., gap=0., manipulation=0.)
    assert evidence['macro_success'] == .5
    assert evidence['success_metric']['censored_prefixes_count_as_success'] is False
    assert 'lower bound' in evidence['success_metric']['description']
    counts = evidence['success_metric']['source_counts_by_domain']
    assert counts['manipulation']['bank_slots'] == 1
    assert counts['manipulation']['successful_bank_slots'] == 0
    assert counts['manipulation']['censored_prefixes'] == 1
    assert evidence['score_by_domain'] == dict(navigation=1., stairs=2., gap=3., manipulation=4.)


def test_success_averages_bank_slots_not_fast_reset_episode_counts():
    from drl.MARL.c1_validation_callback import _fixed_window_success
    bank = [dict(row) for row in VALIDATION_BANK]
    bank.append(dict(env_index=4, domain='navigation', seed=8058, split='validation'))
    def episode(index, success, terminated=True, censored=False):
        return dict(bank[index], success=success, terminated=terminated, censored=censored)
    episodes = [episode(0, True)] * 10 + [episode(4, False)]
    episodes += [episode(i, False, terminated=False, censored=True) for i in (1, 2, 3)]
    scores, macro, counts = _fixed_window_success(episodes, bank)
    assert scores['navigation'] == .5
    assert macro == .125
    assert counts['navigation']['successful_completed_episodes'] == 10
    assert counts['navigation']['successful_bank_slots'] == 1
    assert counts['navigation']['bank_slots'] == 2


@pytest.mark.parametrize('field,value', [('success', 'true'), ('terminated', None), ('censored', float('nan'))])
def test_malformed_child_success_outcomes_rejected_before_controller(tmp_path, field, value):
    callback, checkpoint, calls = setup(tmp_path)
    original = callback.launcher
    def child(command, **kwargs):
        completed = original(command, **kwargs)
        out = Path(command[command.index('--output') + 1])
        result = json.loads((out/'result.json').read_text())
        result['episodes'][0][field] = value
        (out/'result.json').write_text(json.dumps(result))
        return completed
    callback.launcher = child
    with pytest.raises(ValueError, match='boolean'):
        callback.evaluate(checkpoint, 0, 90)


def test_censored_or_unterminated_claim_does_not_count_as_completed_success():
    from drl.MARL.c1_validation_callback import _fixed_window_success
    episodes = [dict(row, success=True, terminated=False, censored=True) for row in VALIDATION_BANK]
    scores, macro, counts = _fixed_window_success(episodes, VALIDATION_BANK)
    assert scores == dict.fromkeys(('navigation', 'stairs', 'gap', 'manipulation'), 0.)
    assert macro == 0.
    assert all(row['completed_episodes'] == 0 for row in counts.values())
