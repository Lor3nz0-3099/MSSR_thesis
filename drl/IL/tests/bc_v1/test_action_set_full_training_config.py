"""Full-run admission and checkpoint protocol; never optimize any weights."""
import copy
import json

import pytest
import torch

from drl.IL.bc_v1 import action_set_training as training
from drl.IL.bc_v1 import train_action_set as runner
from drl.IL.bc_v1.action_set_metrics import ActionSetMetrics
from drl.IL.bc_v1.action_set_targets import target_bundle
from drl.IL.tests.bc_v1.g2_fixtures import ROOT, tensorize
from drl.IL.tests.bc_v1.test_action_set_training import case
from drl.IL.tests.bc_v1.test_train_action_set import static_fixture


def config():
    return training.load_action_set_training_config(ROOT/'configs/bc_v1/training_action_set_v1.json')


def test_approved_full_run_configuration_is_frozen():
    assert config() == {'schema_version': 'mssr.bc_v1.training_action_set.v1',
        'model_contract': 'action_set_v1', 'initialization': 'scratch', 'seed': 20261021,
        'lr': 3e-4, 'weight_decay': 1e-4, 'batch_size': 64, 'max_epochs': 30,
        'patience': 8, 'min_delta': .002, 'threshold_logit': 0.}


def test_absolute_delta_examples_boundary_reset_and_eight_bad_epochs():
    state = training.selection_state(config())
    assert state.observe(0, .1840)
    assert not state.observe(1, .1835) and state.counter == 1
    assert not state.observe(2, .1825) and state.counter == 2
    # Use the exact floating-point boundary used by SelectionState, no rounding.
    assert not state.observe(3, state.best_val - .002) and state.counter == 3
    assert state.observe(4, .1819) and state.counter == 0 and state.best_epoch == 4
    for epoch in range(5, 13):
        assert not state.observe(epoch, .1818)
        assert state.should_stop == (epoch == 12)


def test_fresh_initialization_has_no_deserialization_or_state_restore(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Checkpoint/state restore attempted during scratch initialization')
    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(torch.nn.Module, 'load_state_dict', forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, 'load_state_dict', forbidden)
    monkeypatch.setattr(training, 'load_action_set_checkpoint', forbidden)
    _, kw, h1, h2, manifest = case()
    first, optimizer, identity = training.build_scratch_actor(
        kw['spec'], kw['feature_schema'], h1, h2, manifest, config())
    second, other_optimizer, _ = training.build_scratch_actor(
        kw['spec'], kw['feature_schema'], h1, h2, manifest, config())
    assert first is not second and first.backbone is not second.backbone and first.head is not second.head
    assert optimizer is not other_optimizer and not optimizer.state and not other_optimizer.state
    assert not set(map(id, first.parameters())) & set(map(id, second.parameters()))
    assert first.backbone.config['dropout'] == first.head.config['dropout'] == .1
    assert all(p.dtype == torch.float32 and p.device.type == 'cpu' for p in first.parameters())
    assert identity['model_contract'] == 'action_set_v1'
    state, other = training.selection_state(config()), training.selection_state(config())
    state.observe(0, 1.)
    assert other.last_epoch == -1 and other.best_epoch == -1 and other.counter == 0


@pytest.mark.parametrize('option', ['--resume', '--checkpoint', '--init-checkpoint', '--pretrained', '--load-model'])
def test_cli_has_no_checkpoint_input_or_resume_option(tmp_path, monkeypatch, option):
    def forbidden(*args, **kwargs): pytest.fail('Rejected checkpoint CLI reached preflight')
    monkeypatch.setattr(runner, 'preflight', forbidden)
    with pytest.raises(SystemExit) as exc:
        runner.main(['--output', str(tmp_path/'newrun'), option, '/forbidden.pt'])
    assert exc.value.code == 2 and not (tmp_path/'newrun').exists()


def test_dry_run_explicitly_certifies_scratch_splits_dropout_and_numeric_mode(tmp_path, monkeypatch, capsys):
    module, source, _ = static_fixture(tmp_path, monkeypatch)
    assert module.main(['--g2', str(source), '--output', str(tmp_path/'unused')]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['initialization'] == {'mode': 'scratch', 'pretrained_checkpoint': None,
        'resume_checkpoint': None, 'fresh_model': True, 'fresh_optimizer': True, 'fresh_selection_state': True}
    assert report['splits']['train']['enabled'] and report['splits']['validation']['enabled']
    assert report['splits']['test'] == {'enabled': False, 'samples': 0}
    assert report['dropout'] == {'h1': .1, 'h2': .1}
    assert report['runtime']['amp'] is False and report['runtime']['compile'] is False
    assert report['runtime']['tf32'] is False and report['runtime']['prefetch'] is False
    assert report['config'] == config() and report['TEST_payload_access'] == 0
    assert not (tmp_path/'unused').exists()


def protocol(tmp_path, monkeypatch, losses):
    """Exercise the real run/save protocol with synthetic losses and no train/eval computation."""
    sample, kw, h1, h2, manifest = case()
    row = tensorize(sample, kw)
    report = {'sampler_quotas': {'ordinary': 48, 'structural': 16},
        'initialization': {'mode': 'scratch', 'pretrained_checkpoint': None, 'resume_checkpoint': None,
            'fresh_model': True, 'fresh_optimizer': True, 'fresh_selection_state': True},
        'runtime': {**runner.RUNTIME, 'cache_budgets_bytes': dict(runner.CACHE_BUDGETS)},
        'dropout': {'h1': .1, 'h2': .1}}
    class Data:
        def __init__(self, split): self.split = split
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __len__(self): return 1
        def train_index(self): return ()
        def get_for_collate(self, index): return row
    monkeypatch.setattr(runner, 'open_dataset', lambda root, split, *a, **k: Data(split))
    monkeypatch.setattr(runner, 'JointActionBatchSampler', lambda *a, **k: [(0,)])
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    calls = {'train_stubs': 0, 'validation_stubs': 0, 'saved': []}
    def train_stub(*args, **kwargs):
        calls['train_stubs'] += 1
        return {'families': {name: {'sum': 0., 'support': int(name == 'presence'),
            'mean': 0., 'weighted': 0.} for name in runner.FAMILIES}}
    def validation_stub(*args, **kwargs):
        epoch = calls['validation_stubs']; calls['validation_stubs'] += 1
        metrics = ActionSetMetrics()
        metrics.update(target_bundle(row), target_bundle(row))
        metrics.update_numeric('payload', [epoch/100.], [0.], '1', normalized=True)
        return {'total_loss': losses[epoch], 'families': {'presence': {'mean': losses[epoch], 'support': 1}},
            'metrics': metrics.compute()}
    monkeypatch.setattr(runner, 'train_batch', train_stub)
    monkeypatch.setattr(runner, 'evaluate', validation_stub)
    real_save = runner.save_action_set_checkpoint
    def save(path, model, optimizer, identity, cfg, summary, **kwargs):
        assert not optimizer.state  # No step or restored optimizer moments.
        calls['saved'].append((path.name, copy.deepcopy(summary)))
        real_save(path, model, optimizer, identity, cfg, summary, **kwargs)
    monkeypatch.setattr(runner, 'save_action_set_checkpoint', save)
    # Observe actual serialized checkpoint payloads without loading any checkpoint.
    real_torch_save = torch.save
    def serialize(payload, *args, **kwargs):
        calls['last_payload'] = {k: copy.deepcopy(payload[k]) for k in ('summary', 'config', 'identity', 'optimizer')}
        return real_torch_save(payload, *args, **kwargs)
    monkeypatch.setattr(torch, 'save', serialize)
    output = tmp_path/'protocol_only'
    assert runner.run(output, tmp_path/'synthetic_g2', config(), (report, kw, h1, h2, manifest)) == 0
    calls['output'] = output
    return calls


def test_best_last_same_epoch_metrics_and_complete_scratch_provenance(tmp_path, monkeypatch):
    calls = protocol(tmp_path, monkeypatch, [.1840, .1835, .1825, .1819] + [.1818]*8)
    best = [r for name, r in calls['saved'] if name == 'best.pt']
    last = [r for name, r in calls['saved'] if name == 'last.pt']
    assert [r['epoch'] for r in best] == [0, 3]
    assert [r['epoch'] for r in last] == list(range(12))
    assert calls['train_stubs'] == 11 and calls['validation_stubs'] == 12
    assert best[-1]['validation']['total_loss'] == .1819
    assert best[-1]['validation']['metrics']['normalized_payload_rmse'] == .03
    assert last[-1]['validation']['metrics']['normalized_payload_rmse'] == .11
    metrics = best[-1]['validation']['metrics']
    assert {'set_exact_match', 'set_precision', 'set_recall', 'set_f1', 'set_jaccard',
        'target_cardinality_mean', 'predicted_cardinality_mean', 'cardinality_absolute_error',
        'false_positive_actions', 'false_negative_actions'} <= metrics.keys()
    assert metrics['per_action']['pace']['support'] == 1
    assert metrics['per_action']['pace']['precision'] == metrics['per_action']['pace']['recall'] == 1.
    assert metrics['normalized_numeric']['payload']['1']['mae'] == .03
    assert metrics['normalized_numeric']['payload']['1']['within_tolerance'] is None
    assert best[-1]['selection']['counter'] == 0 and last[-1]['selection']['counter'] == 8
    payload = calls['last_payload']
    assert payload['summary']['initialization']['mode'] == 'scratch'
    assert payload['summary']['runtime']['numeric_mode'] == 'FP32 eager'
    assert payload['summary']['dropout'] == {'h1': .1, 'h2': .1}
    assert payload['config'] == config() and payload['optimizer']['state'] == {}
    assert payload['identity']['source_manifest_fingerprint']
    assert (calls['output']/'best.pt').is_file() and (calls['output']/'last.pt').is_file()


def test_epoch_zero_is_diagnostic_and_limit_allows_thirty_trained_epochs(tmp_path, monkeypatch):
    calls = protocol(tmp_path, monkeypatch, [1. - i*.01 for i in range(31)])
    assert calls['train_stubs'] == 30 and calls['validation_stubs'] == 31
    assert [r['epoch'] for name, r in calls['saved'] if name == 'last.pt'] == list(range(31))
    assert calls['saved'][-1][1]['epoch'] == 30


def test_cli_prints_scratch_banner_and_config_before_handing_off_to_run(tmp_path, monkeypatch, capsys):
    report = {'config': config()}
    monkeypatch.setattr(runner, 'preflight', lambda *a, normalizer_dir=None: (report, None, None, None, None))
    def no_training(*args, **kwargs):
        printed = capsys.readouterr().out
        assert 'INITIALIZATION: FROM SCRATCH' in printed
        assert 'PRETRAINED CHECKPOINT: NONE' in printed and 'RESUME CHECKPOINT: NONE' in printed
        assert 'TEST: disabled' in printed
        assert json.loads(printed.splitlines()[-1])['config'] == config()
        return 0
    monkeypatch.setattr(runner, 'run', no_training)
    assert runner.main(['--output', str(tmp_path/'uncreated'), '--run']) == 0
    assert not (tmp_path/'uncreated').exists()
