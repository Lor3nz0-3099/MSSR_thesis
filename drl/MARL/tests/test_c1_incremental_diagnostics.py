"""Offline regressions for append-only diagnostics, including SMDP pre/post rows."""
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from drl.MARL.mappo import StructuredRolloutSidecar


def _sample(value=1.):
    return SimpleNamespace(action={'owner': torch.tensor([value])},
        graph_snapshot={'x': torch.ones(2)}, critic_graph_snapshot=None,
        next_critic_graph_snapshot=None)


def _checker():
    from drl.MARL.c1_incremental_diagnostics import StructuredSampleDiagnostics
    counters = {}
    return StructuredSampleDiagnostics(counters), counters


def test_simultaneous_pre_post_checks_first_infinite_sample_and_every_new_row():
    checker, counters = _checker()
    sidecar = StructuredRolloutSidecar()
    sidecar.append_step([_sample(float('inf'))] + [_sample() for _ in range(7)])
    sidecar.append_step([_sample() for _ in range(8)])
    checker.observe(sidecar)
    assert counters['structured_samples_checked'] == 16
    assert counters['nonfinite_structured_samples'] == 1
    assert counters['structured_tensor_checks'] == 32
    assert counters['structured_device_reductions'] == 1
    for _ in range(40):
        checker.observe(sidecar)
    assert counters['structured_samples_checked'] == 16
    assert counters['structured_tensor_checks'] == 32
    assert counters['structured_device_reductions'] == 1
    assert counters['structured_no_new_sample_calls'] == 40
    assert counters['structured_repeat_samples_skipped'] == 640


def test_clear_reappend_equal_length_reused_object_is_a_new_event():
    checker, counters = _checker()
    sidecar = StructuredRolloutSidecar()
    repeated = _sample()
    sidecar.append_step([repeated])
    checker.observe(sidecar)
    sidecar.clear()
    repeated.action['owner'].fill_(float('nan'))
    sidecar.append_step([repeated])
    checker.observe(sidecar)
    assert counters['structured_samples_checked'] == 2
    assert counters['nonfinite_structured_samples'] == 1
    sidecar.clear()
    sidecar.append_step([_sample()])
    checker.observe(sidecar)
    assert counters['structured_samples_checked'] == 3
    assert counters['nonfinite_structured_samples'] == 1


def test_replaced_list_and_replaced_sidecar_restart_cursor():
    checker, counters = _checker()
    sidecar = StructuredRolloutSidecar()
    sample = _sample()
    sidecar.append_step([sample])
    checker.observe(sidecar)
    sidecar._samples = [sample]
    checker.observe(sidecar)
    replacement = StructuredRolloutSidecar()
    replacement._samples = sidecar._samples
    checker.observe(replacement)
    assert counters['structured_samples_checked'] == 3


def test_append_same_sample_twice_checks_both_positions():
    checker, counters = _checker()
    sidecar = StructuredRolloutSidecar()
    sample = _sample(float('nan'))
    sidecar.append_step([sample])
    checker.observe(sidecar)
    sidecar.append_step([sample])
    checker.observe(sidecar)
    assert counters['structured_samples_checked'] == 2
    assert counters['nonfinite_structured_samples'] == 2


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_batched_finite_matches_original_nested_tree_semantics(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable; no simulator is required')
    from drl.MARL.c1_incremental_diagnostics import finite_sample_trees
    from drl.MARL.c1_training_telemetry import _finite_tree
    @dataclass
    class Payload:
        values: object
    trees = [Payload({'a': torch.ones(3, device=device), 'nested': [None, True, 2.]}),
        [torch.tensor([], device=device), float('inf')],
        {'valid': torch.ones(2, device=device), 'bad': torch.tensor([float('nan')], device=device)},
        (torch.tensor([False, True], device=device), 0),
        [float('nan'), torch.ones(1, device=device)]]
    counters = {}
    assert finite_sample_trees(trees, counters) == [_finite_tree(tree) for tree in trees]
    assert counters['structured_tensor_checks'] == 6
    assert counters['structured_device_reductions'] == 1
    assert counters['structured_host_transfers'] == int(device == 'cuda')


def test_generation_token_supports_lightweight_sidecar_without_object_deduplication():
    checker, counters = _checker()
    sample = _sample()
    sidecar = SimpleNamespace(_samples=[sample], _diagnostic_generation=0)
    checker.observe(sidecar)
    sidecar._samples.clear()
    sidecar._diagnostic_generation += 1
    sidecar._samples.append(sample)
    checker.observe(sidecar)
    assert counters['structured_samples_checked'] == 2


def test_training_hook_fails_for_infinite_first_pre_event(tmp_path, monkeypatch):
    from drl.MARL.tests.test_isaaclab_c1_training import _campaign_fixture
    training, args, env, wrapper, learner, clock = _campaign_fixture(
        tmp_path, monkeypatch, train=False)
    args.total_transitions = 4
    learner._structured_rollout_sidecar = StructuredRolloutSidecar()
    calls = []
    def record_transition(**kwargs):
        rows = [_sample(float('inf') if not calls and i == 0 else 1.) for i in range(4)]
        for row in rows:
            row.owner_active_mask = {'m': True}
        learner._structured_rollout_sidecar.append_step(rows)
        calls.append(True)
    learner.record_transition = record_transition
    result = training.run_native_training(args, env, wrapper, learner,
        startup_s=0., assembly_s=0., run_started=0.)
    assert result['status'] == 'fail'
    assert result['diagnostics']['nonfinite_structured_samples'] == 1
    assert result['diagnostics']['structured_samples_checked'] == 8
