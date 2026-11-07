"""Telemetry must describe actual targets and distinguish padding from experience."""
import json
import math
from types import SimpleNamespace

import pytest
import torch

from drl.MARL.mappo import MSSRMAPPO
from drl.MARL.tests.test_mssr_mappo import (
    _make_two_step_altruistic_sidecar, _zero_altruistic_values, _zero_altruistic_bootstrap,
)


def test_computed_target_stats_use_only_active_owner_samples():
    agents = ('module_0', 'module_1')
    targets = MSSRMAPPO.compute_altruistic_rollout_targets(
        sidecar=_make_two_step_altruistic_sidecar(), possible_agents=agents,
        owner_values=_zero_altruistic_values(), mean_field_values=_zero_altruistic_values(),
        owner_bootstrap_values=_zero_altruistic_bootstrap(),
        mean_field_bootstrap_values=_zero_altruistic_bootstrap(),
        dones=torch.zeros(2, 1, 1, dtype=torch.bool), discount_factor=0.,
        lambda_coefficient=.95, alpha=.5)
    # First row is Expert; second has just module_0 active.
    samples = [SimpleNamespace(owner_active_mask={uid: False for uid in agents}),
               SimpleNamespace(owner_active_mask={'module_0': True, 'module_1': False})]
    metrics = MSSRMAPPO.altruistic_target_diagnostics(
        targets=targets, structured_samples=samples, possible_agents=agents, alpha=.5)
    for name, value in [('owner_advantage', 2.), ('mean_field_advantage', 4.),
                        ('scaled_mean_field_advantage', 2.), ('combined_advantage', 4.)]:
        assert metrics[name + '_count'] == 1
        assert metrics[name + '_mean'] == pytest.approx(value)
        assert metrics[name + '_std'] == 0.
        assert metrics[name + '_min'] == metrics[name + '_max'] == value
    assert all(isinstance(value, (int, float)) and math.isfinite(value) for value in metrics.values())


def test_empty_active_target_stats_are_numeric_and_zero():
    target = SimpleNamespace(**{name: {'m': torch.tensor([[[float('nan')]]])} for name in
        ('owner_advantages', 'mean_field_advantages', 'altruistic_advantages')})
    metrics = MSSRMAPPO.altruistic_target_diagnostics(targets=target,
        structured_samples=[SimpleNamespace(owner_active_mask={'m': False})], possible_agents=['m'], alpha=.5)
    assert metrics and all(value == 0 for value in metrics.values())


def test_recorder_counts_active_and_completed_separately_and_censors_open_rows():
    from drl.MARL.c1_training_telemetry import TrainingTelemetry
    recorder = TrainingTelemetry()
    recorder.observe_transition(domain='stairs', seed=12, generation=0, reward=100.,
        terminated=False, truncated=False, success=False, active=False, delta_s=.1)
    recorder.observe_transition(domain='stairs', seed=12, generation=0, reward=2.,
        terminated=True, truncated=False, success=True, active=True, delta_s=.2,
        actions={'x': torch.ones(2)}, graphs={'x': torch.ones(2)})
    recorder.observe_transition(domain='gap', seed=15, generation=0, reward=-1.,
        terminated=False, truncated=True, success=False, active=True, delta_s=.3)
    recorder.observe_transition(domain='stairs', seed=19, generation=1, reward=3.,
        terminated=False, truncated=False, success=False, active=True)
    recorder.record_update({'actor_loss': 1., 'minibatches': 2})
    result = recorder.snapshot()
    assert result['row_transitions'] == 4
    assert result['active_marl_transitions'] == 3
    assert result['completed_episodes'] == 1
    assert result['censored_episodes'] == 2
    episode = result['episodes'][0]
    assert episode['return'] == 102. and episode['active_return'] == 2.
    assert episode['length'] == 2 and episode['active_length'] == 1
    assert episode['elapsed_s'] == pytest.approx(.3)
    assert result['domains']['stairs']['active_marl_transitions'] == 2
    assert result['domains']['stairs']['successes'] == 1
    json.dumps(result, allow_nan=False)


def test_recorder_records_nonfinite_data_and_rejects_bad_update_metrics():
    from drl.MARL.c1_training_telemetry import TrainingTelemetry
    recorder = TrainingTelemetry()
    recorder.observe_transition(domain='gap', seed=1, generation=0, reward=float('nan'),
        terminated=False, truncated=False, success=False, active=True,
        actions=torch.tensor([float('inf')]), graphs={'x': torch.tensor([float('nan')])})
    result = recorder.snapshot()
    assert result['finite_checks']['reward']['nonfinite'] == 1
    assert result['finite_checks']['actions']['nonfinite'] == 1
    assert result['finite_checks']['graphs']['nonfinite'] == 1
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match='nonfinite'):
        recorder.record_update({'actor_loss': float('nan')})


def test_target_stats_support_per_owner_alpha_without_changing_targets_or_rng():
    tensors = {'m0': torch.tensor([[[2.]]]), 'm1': torch.tensor([[[4.]]])}
    target = SimpleNamespace(owner_advantages=tensors, mean_field_advantages=tensors,
        altruistic_advantages={'m0': torch.tensor([[[3.]]]), 'm1': torch.tensor([[[12.]]])})
    before = {uid: value.clone() for uid, value in tensors.items()}
    rng = torch.random.get_rng_state().clone()
    metrics = MSSRMAPPO.altruistic_target_diagnostics(targets=target,
        structured_samples=[SimpleNamespace(owner_active_mask={'m0': True, 'm1': True})],
        possible_agents=['m0', 'm1'], alpha={'m0': .5, 'm1': 2.})
    assert metrics['scaled_mean_field_advantage_mean'] == 4.5
    assert metrics['combined_advantage_mean'] == 7.5
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert all(torch.equal(tensors[uid], before[uid]) for uid in tensors)
    target.owner_advantages['m0'] = torch.full((1, 1, 1), float('nan'))
    with pytest.raises(ValueError, match='nonfinite active owner_advantage'):
        MSSRMAPPO.altruistic_target_diagnostics(targets=target,
            structured_samples=[SimpleNamespace(owner_active_mask={'m0': True, 'm1': True})],
            possible_agents=['m0', 'm1'], alpha=.5)


def test_recorder_env_identity_and_snapshot_do_not_finalize_live_episodes():
    from drl.MARL.c1_training_telemetry import TrainingTelemetry
    recorder = TrainingTelemetry()
    for index in (0, 1):
        recorder.observe_transition(domain='gap', seed=7, generation=0, env_index=index,
            reward=1., terminated=False, truncated=False, success=False, active=True)
    assert recorder.snapshot()['censored_episodes'] == 2
    assert recorder.snapshot()['censored_episodes'] == 2
    recorder.observe_transition(domain='gap', seed=7, generation=0, env_index=0,
        reward=-2., terminated=True, truncated=False, success=False, active=True)
    result = recorder.snapshot()
    assert result['completed_episodes'] == result['censored_episodes'] == 1
    assert result['domains']['gap']['failures'] == 1
    assert result['episodes'][0]['return'] == -1.
    result['domains']['gap']['failures'] = 99
    assert recorder.snapshot()['domains']['gap']['failures'] == 1


def test_resource_sampler_uses_sampled_peak_and_reports_allocator_and_rss(monkeypatch):
    from drl.MARL import c1_training_telemetry as telemetry
    monkeypatch.setattr(telemetry.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(telemetry.torch.cuda, 'max_memory_allocated', lambda device: 123)
    monkeypatch.setattr(telemetry.torch.cuda, 'max_memory_reserved', lambda device: 456)
    samples = iter([f'{telemetry.os.getpid()}, 20', f'{telemetry.os.getpid()}, 10'])
    monkeypatch.setattr(telemetry.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(stdout=next(samples)))
    recorder = telemetry.TrainingTelemetry()
    recorder.sample_resources()
    result = recorder.sample_resources()
    assert result['torch_peak_allocated_bytes'] == 123
    assert result['torch_peak_reserved_bytes'] == 456
    assert result['process_gpu_sampled_peak_bytes'] == 20 * 1024**2
    assert result['rss_peak_bytes'] > 0


def test_cpu_resource_sample_skips_cuda_allocator_even_when_cuda_available(monkeypatch):
    from drl.MARL import c1_training_telemetry as telemetry
    monkeypatch.setattr(telemetry.torch.cuda, 'is_available', lambda: True)
    def unexpected_counter(device):
        raise AssertionError('CPU mode must not read CUDA allocator counters')
    monkeypatch.setattr(telemetry.torch.cuda, 'max_memory_allocated', unexpected_counter)
    monkeypatch.setattr(telemetry.torch.cuda, 'max_memory_reserved', unexpected_counter)
    monkeypatch.setattr(telemetry.subprocess, 'run', lambda *args, **kwargs:
        SimpleNamespace(stdout=f'{telemetry.os.getpid()}, 10'))
    result = telemetry.TrainingTelemetry().sample_resources(torch.device('cpu'))
    assert result['rss_peak_bytes'] > 0
    assert result['process_gpu_sampled_peak_bytes'] == 10 * 1024**2
    assert 'torch_peak_allocated_bytes' not in result
    assert 'torch_peak_reserved_bytes' not in result
