"""K1 orchestration contracts; synthetic examples never touch split payloads."""
import importlib
import math

import pytest
import torch

from drl.IL.bc_v1.sampling import JointActionBatchSampler, TrainRowIndex


def api():
    name = 'drl.IL.bc_v1.trainer'
    assert importlib.util.find_spec(name) is not None, 'K1 trainer contracts are missing'
    return importlib.import_module(name)


def test_epoch_length_and_sampler_stream_are_continuous():
    m = api()
    assert m.steps_per_epoch(78302, 64) == 1224
    index = TrainRowIndex({'a': tuple(range(130))}, {'a': (1, 2)}, {'split_name': 'train'})
    sampler = JointActionBatchSampler(index, batch_size=64, ordinary_quota=56, structural_quota=8, seed=20261006)
    first = list(sampler)
    second = list(sampler)
    replay = JointActionBatchSampler(index, batch_size=64, ordinary_quota=56, structural_quota=8, seed=20261006)
    assert first == list(replay)
    assert second == list(replay)
    assert first != second and len(first) == 3


def test_validation_sequential_tail_and_test_guard_precedes_open():
    m = api()
    assert list(m.validation_batches(130, 64)) == [tuple(range(64)), tuple(range(64, 128)), (128, 129)]
    opened = []
    def factory(*args, **kwargs):
        opened.append(args[1]); return object()
    with pytest.raises(ValueError, match='TEST|split'):
        m.open_dataset('unused', 'test', {}, factory=factory)
    assert opened == []
    m.open_dataset('unused', 'train', {}, factory=factory)
    m.open_dataset('unused', 'validation', {}, factory=factory)
    assert opened == ['train', 'validation']


def test_global_family_aggregation_is_support_weighted():
    m = api()
    accumulator = m.FamilyAccumulator({'a': 2., 'b': .5})
    accumulator.add({'a': {'sum': 2., 'support': 1}, 'b': {'sum': 0., 'support': 0}})
    accumulator.add({'a': {'sum': 18., 'support': 9}, 'b': {'sum': 12., 'support': 3}})
    result = accumulator.compute()
    assert result['total_loss'] == 6.
    assert result['families']['a']['support'] == 10
    assert result['families']['b']['mean'] == 4.
    # A different-support example explicitly distinguishes the wrong batch mean.
    accumulator = m.FamilyAccumulator({'a': 1.})
    accumulator.add({'a': {'sum': 2., 'support': 1}})
    accumulator.add({'a': {'sum': 0., 'support': 9}})
    assert accumulator.compute()['total_loss'] == .2
    assert accumulator.compute()['total_loss'] != (2. + 0.) / 2
    with pytest.raises(ValueError, match='Nonfinite'):
        accumulator.add({'a': {'sum': float('nan'), 'support': 1}})


def test_selection_val_only_delta_ties_patience_and_epoch_zero():
    m = api()
    selection = m.SelectionState()
    assert selection.patience == 15 and selection.min_delta == 1e-4
    assert selection.observe(0, 2.)
    assert not selection.trained_improvement
    assert not selection.observe(1, 2.)  # Keep epoch zero on ties.
    assert not selection.observe(2, 2. - 5e-5)
    assert selection.best_epoch == 0 and selection.counter == 2
    assert selection.observe(3, 1.9)
    assert selection.best_epoch == 3 and selection.counter == 0 and selection.trained_improvement
    for epoch in range(4, 18):
        assert not selection.observe(epoch, 1.9 + .01*epoch)
        assert not selection.should_stop
    assert not selection.observe(18, 1.9)
    assert selection.should_stop and selection.counter == 15
    with pytest.raises(ValueError, match='Nonfinite'):
        selection.observe(19, float('nan'))


def test_all_supports_preserved_and_drift_rejected():
    m = api()
    original = {'families': {'a': {'support': 2, 'mean': 9.}}, 'metrics': {
        'mode': {'support': 4, 'class_support': [1, 3], 'value': .1},
        'numeric': {'decision_support': 5}, 'tuple': {'excluded_incomplete': 2}}}
    snapshot = m.support_snapshot(original)
    changed_values = {'families': {'a': {'support': 2, 'mean': 1.}}, 'metrics': {
        'mode': {'support': 4, 'class_support': [1, 3], 'value': .9},
        'numeric': {'decision_support': 5}, 'tuple': {'excluded_incomplete': 2}}}
    m.check_supports(snapshot, changed_values)
    changed_values['metrics']['mode']['class_support'] = [2, 2]
    with pytest.raises(ValueError, match='support'):
        m.check_supports(snapshot, changed_values)


def test_evaluation_modes_no_grad_global_aggregation_and_no_updates(monkeypatch):
    m = api()
    class Toy(torch.nn.Module):
        def __init__(self): super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(2.)); self.modes = []
        def forward(self, inputs):
            self.modes.append((self.training, torch.is_grad_enabled()))
            return inputs['x'] * self.weight
    class Metrics:
        def reset(self): self.n = 0
        def update(self, prediction, batch): self.n += prediction.numel()
        def compute(self): return {'toy': {'support': self.n, 'value': 1.}}
    def collate(rows): return {'inputs': {'x': torch.tensor(rows, dtype=torch.float32)}, 'supervision': {}}
    def loss(pred, batch, cfg):
        s = pred.sum(); n = torch.tensor(pred.numel())
        return s/n, {'a': {'sum': s, 'support': n, 'mean': s/n, 'weighted': s/n}}
    monkeypatch.setattr(m, 'compute_bc_loss', loss)
    class Data(list): split = 'validation'
    model = Toy(); before = model.weight.detach().clone()
    result = m.evaluate_validation(model, Data([1, 2, 3]), {'weights': {'a': 1.}}, Metrics(),
                                   device='cpu', batch_size=2, collate=collate)
    assert result['total_loss'] == 4. and result['samples'] == 3 and result['batches'] == 2
    assert result['metrics']['toy']['support'] == 3
    assert model.modes == [(False, False), (False, False)]
    assert model.training and torch.equal(model.weight, before) and model.weight.grad is None
    data = Data([1]); data.split = 'test'
    with pytest.raises(ValueError, match='TEST|validation'):
        m.evaluate_validation(model, data, {}, Metrics(), device='cpu')


def runner():
    name = 'drl.IL.bc_v1.train_baseline'
    assert importlib.util.find_spec(name) is not None, 'K1 runner is missing'
    return importlib.import_module(name)


def test_runner_fixed_config_no_hpo_or_test_entrypoint(tmp_path):
    import json
    from pathlib import Path
    r = runner()
    cfg = r.load_config()
    assert cfg['seed'] == 20261006 and cfg['steps_per_epoch'] == 1224
    assert cfg['batch_size'] == 64 and cfg['general_quota'] == 56 and cfg['structural_quota'] == 8
    assert cfg['early_stopping_patience'] == 15 and cfg['min_delta'] == 1e-4
    assert cfg['amp'] is False and cfg['scheduler'] is None
    for key, value in [('learning_rate', 1e-3), ('seed', 1), ('max_epochs', 2), ('amp', True)]:
        path = tmp_path/'bad.json'
        path.write_text(json.dumps({**cfg, key: value}))
        with pytest.raises(ValueError, match='fixed baseline'):
            r.load_config(path)
    path.write_text(json.dumps({**cfg, 'test_split': 'test'}))
    with pytest.raises(ValueError, match='fixed baseline'):
        r.load_config(path)


def test_runner_validation_epoch_zero_selection_checkpoints_and_stop(tmp_path, monkeypatch):
    r = runner()
    cfg = r.load_config()
    events = []
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    values = iter([2., 1.] + [1.] * 15)
    def evaluate(*args, **kwargs):
        events.append('val')
        return {'total_loss': next(values), 'families': {'a': {'sum': 1., 'support': 1, 'mean': 1., 'weighted': 1.}},
                'metrics': {}, 'samples': 22762, 'batches': 356, 'wall_seconds': .01}
    def train(*args, **kwargs):
        events.append('train')
        assert kwargs['global_step'] == (kwargs['epoch']-1)*1224
        return {'total_loss': .5, 'families': {}, 'steps': 1224,
                'global_step': kwargs['global_step']+1224, 'wall_seconds': .1,
                'mean_step_seconds': .01, 'median_step_seconds': .01, 'samples_per_second': 64.,
                'first_step_audit': {'optimizer_step_completed': True}}
    monkeypatch.setattr(r, 'evaluate_validation', evaluate)
    monkeypatch.setattr(r, 'train_epoch', train)
    monkeypatch.setattr(r, 'capture_reference', lambda *a: {'test': torch.tensor(1.)})
    monkeypatch.setattr(r, 'memory_stats', lambda: {})
    checkpoint_calls = []
    original_save = r.save_checkpoint
    def save(path, *args, **kwargs):
        checkpoint_calls.append((path.name, kwargs['summary']['epoch']))
        original_save(path, *args, **kwargs)
    monkeypatch.setattr(r, 'save_checkpoint', save)
    audit, checkpoints = tmp_path/'audit', tmp_path/'checkpoints'
    audit.mkdir(); checkpoints.mkdir()
    state = r.fit(model, optimizer, train_data=object(), val_data=object(), sampler=object(),
                  cfg=cfg, components={'i1': {}, 'identity': {'fixture': 'v1'}, 'metrics': object()},
                  audit=audit, checkpoints=checkpoints, identity_check=lambda: None)
    assert events[0] == 'val' and events[1:3] == ['train', 'val']
    assert state['selection'].best_epoch == 1 and state['selection'].counter == 15
    assert state['epoch'] == 16 and state['global_step'] == 16*1224
    assert [epoch for name, epoch in checkpoint_calls if name == 'best_checkpoint.pt'] == [0, 1]
    import json
    history = [json.loads(line) for line in (audit/'history.jsonl').read_text().splitlines()]
    assert len(history) == 17 and history[0]['epoch'] == 0 and history[-1]['patience_counter'] == 15
    payload = torch.load(checkpoints/'best_checkpoint.pt', weights_only=False)
    assert payload['summary']['best_epoch'] == 1 and payload['summary']['epoch'] == 1
    assert payload['identity'] == {'fixture': 'v1'} and 'rng_state' in payload
    assert json.loads((checkpoints/'manifest.json').read_text())['best_epoch'] == 1


def test_runner_support_drift_stops_before_next_epoch(tmp_path, monkeypatch):
    r = runner()
    cfg = r.load_config()
    calls = []
    def evaluate(*args, **kwargs):
        n = len(calls)+1
        return {'total_loss': 2./n, 'families': {'a': {'support': n}}, 'metrics': {},
                'samples': 22762, 'batches': 356, 'wall_seconds': 0.}
    def train(*args, **kwargs):
        calls.append(kwargs['epoch'])
        return {'steps': 1224, 'global_step': 1224, 'first_step_audit': {'optimizer_step_completed': True}}
    monkeypatch.setattr(r, 'evaluate_validation', evaluate)
    monkeypatch.setattr(r, 'train_epoch', train)
    monkeypatch.setattr(r, 'capture_reference', lambda *a: {})
    monkeypatch.setattr(r, 'memory_stats', lambda: {})
    for name in ('a', 'c'): (tmp_path/name).mkdir()
    model = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError, match='support drift'):
        r.fit(model, torch.optim.AdamW(model.parameters()), train_data=object(), val_data=object(), sampler=object(),
              cfg=cfg, components={'i1': {}, 'identity': {}, 'metrics': object()},
              audit=tmp_path/'a', checkpoints=tmp_path/'c', identity_check=lambda: None)
    assert calls == [1] and (tmp_path/'c'/'last_checkpoint.pt').exists()


def test_runtime_status_is_atomic_machine_readable_and_failure_visible(tmp_path):
    import json
    import os
    r = runner()
    status = r.RunStatus(tmp_path/'status.json', device='cuda:0')
    status.update('STARTING', phase='PREFLIGHT')
    initial = json.loads((tmp_path/'status.json').read_text())
    assert initial['pid'] == os.getpid() and initial['run_state'] == 'STARTING'
    status.update('RUNNING', epoch=1, global_step=50, phase='TRAINING')
    status.update('FAILED', stop_reason='fixture nonfinite loss')
    final = json.loads((tmp_path/'status.json').read_text())
    assert final['global_step'] == 50 and final['epoch'] == 1
    assert final['stop_reason'] == 'fixture nonfinite loss' and final['last_update_timestamp']
    assert list(tmp_path.iterdir()) == [tmp_path/'status.json']
    with pytest.raises(ValueError, match='state'):
        status.update('RESTARTING')


def test_training_reports_actual_optimization_mean_separately_from_global_families(monkeypatch):
    m = api()
    class Data(list): split = 'train'
    class Sampler:
        batch_size = 1
        def __len__(self): return 2
        def __iter__(self): return iter([[0], [1]])
    model = torch.nn.Linear(1, 1)
    results = iter([
        {'total_loss': 2., 'families': {'a': {'sum': 2., 'support': 1, 'mean': 2., 'weighted': 2.}},
         'grad_norm_before_clip': 1., 'optimizer_step_completed': True},
        {'total_loss': 0., 'families': {'a': {'sum': 0., 'support': 9, 'mean': 0., 'weighted': 0.}},
         'grad_norm_before_clip': 1., 'optimizer_step_completed': True}])
    monkeypatch.setattr(m, 'train_step', lambda *args, **kwargs: next(results))
    monkeypatch.setattr(m, 'collate_policy_samples', lambda rows: {'inputs': {}, 'supervision': {}})
    result = m.train_epoch(model, Data([0, 1]), Sampler(), torch.optim.AdamW(model.parameters()),
                           {'weights': {'a': 1.}}, device='cpu', epoch=1)
    assert result['total_loss'] == 1.  # Actual objective averaged across two updates.
    assert result['global_support_total_loss'] == .2
    assert result['families']['a']['support'] == 10


def test_runner_exception_marks_failed_and_never_restarts(tmp_path, monkeypatch):
    import json
    r = runner()
    config_path = r.DEFAULT_CONFIG
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    attempts = []
    def fail(cfg):
        attempts.append(1)
        raise ValueError('fixture CUDA unavailable')
    monkeypatch.setattr(r, 'initialize', fail)
    with pytest.raises(ValueError, match='CUDA unavailable'):
        r.run(config_path)
    assert attempts == [1]
    runtime = tmp_path/'artifacts/bc_v1/runs/k1_baseline_v1'
    status = json.loads((runtime/'status.json').read_text())
    assert status['run_state'] == 'FAILED' and 'CUDA unavailable' in status['stop_reason']
    assert (runtime/'failure.json').is_file()
    assert not (tmp_path/'artifacts/bc_v1/checkpoints/k1_baseline_v1').exists()


def test_failure_diagnostics_do_not_mask_original_error_or_prevent_failed_status(tmp_path, monkeypatch):
    import json
    r = runner()
    config_path = r.DEFAULT_CONFIG
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    def fail(cfg): raise ValueError('original CUDA failure')
    monkeypatch.setattr(r, 'initialize', fail)
    original_write = r._write_json
    def write(path, value):
        if path.name == 'failure.json': raise OSError('diagnostic write failed')
        return original_write(path, value)
    monkeypatch.setattr(r, '_write_json', write)
    with pytest.raises(ValueError, match='original CUDA failure'):
        r.run(config_path)
    doc = json.loads((tmp_path/'artifacts/bc_v1/runs/k1_baseline_v1/status.json').read_text())
    assert doc['run_state'] == 'FAILED' and 'original CUDA failure' in doc['stop_reason']


@pytest.mark.parametrize('branch', ['feature/bc-v1-training-pipeline', 'nav2-integration'])
def test_initialize_accepts_branch_as_provenance_not_scientific_identity(monkeypatch, tmp_path, branch):
    """Exercise the real initialization boundary without CUDA or frozen payloads.

    Hardware, artifact storage and model allocation are simulated; the actual
    runner must pass the branch boundary and return the observed Git provenance.
    """
    from types import SimpleNamespace
    r = runner()
    cfg = r.load_config()
    monkeypatch.setattr(r.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(r.torch.cuda, 'get_device_name', lambda device: 'NVIDIA GeForce RTX 4070 Laptop GPU')
    monkeypatch.setattr(r.torch.cuda, 'manual_seed_all', lambda seed: None)
    monkeypatch.setattr(r.subprocess, 'check_output', lambda *args, **kwargs: branch+'\n')
    sources = {'paths': {'output': tmp_path}, 'identity': {'fixture': 'v1'},
               'kwargs': {'spec': object(), 'feature_schema': object(), 'normalizers': object()}}
    monkeypatch.setattr(r, 'load_sources', lambda: sources)
    class Data:
        def __init__(self, split):
            self.split = split
            self.episode_ids = tuple(str(i) for i in range(49 if split == 'train' else 12))
            self.manifest = {'manifest_fingerprint': r.SIDECAR_PIN, 'complete': True}
        def __len__(self): return 78302 if self.split == 'train' else 22762
        def train_index(self):
            return TrainRowIndex({'fixture': tuple(range(78302))}, {'fixture': (0,)}, {'split_name': 'train'})
        def close(self): pass
    monkeypatch.setattr(r, 'open_dataset', lambda root, split, identity: Data(split))
    class CPUModel(torch.nn.Linear):
        def to(self, device): return self  # No device allocation in this boundary test.
    monkeypatch.setattr(r, 'BCGNN', lambda *args, **kwargs: CPUModel(2, 1))
    monkeypatch.setattr(r, 'build_optimizer', lambda *args, **kwargs:
                        SimpleNamespace(param_groups=[{'betas': (.9, .999), 'eps': 1e-8}]))
    monkeypatch.setattr(r, 'BCMetrics', lambda *args: object())
    initialized = r.initialize(cfg)
    try:
        assert initialized[-1] == branch  # The same value becomes run environment.branch.
        assert initialized[0] is sources
        assert initialized[4].seed == 20261006 and len(initialized[4]) == 1224
    finally:
        initialized[5].close()
        initialized[6].close()
