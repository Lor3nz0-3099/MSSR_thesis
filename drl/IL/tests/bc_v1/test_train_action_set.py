"""Safe future runner: synthetic steps only, no real-data epochs or GPU."""
import importlib
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.train_action_set'), 'Missing ActionSet training entrypoint'
    return importlib.import_module('drl.IL.bc_v1.train_action_set')


def case():
    from drl.IL.tests.bc_v1.test_action_set_training import build
    from drl.IL.tests.bc_v1.g2_fixtures import tensorize
    from drl.IL.bc_v1.action_set_targets import collate_action_set_samples
    sample, kw, _, _, _, model, optimizer, _ = build()
    return collate_action_set_samples([tensorize(sample, kw)]), model, optimizer


def test_train_batch_updates_scratch_actor_with_finite_clipped_gradient():
    module = api(); batch, model, optimizer = case()
    before = model.head.presence_scorer[-1].weight.detach().clone()
    result = module.train_batch(model, batch, optimizer)
    assert math.isfinite(result['total_loss']) and math.isfinite(result['grad_norm'])
    assert result['families']['presence']['support'] == 1
    assert not torch.equal(before, model.head.presence_scorer[-1].weight)
    norm = torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None))
    assert norm <= 1.00001


def test_measure_batch_reports_raw_conflicting_candidates_and_matched_normalized_payload():
    module = api(); batch, model, _ = case()
    prediction = model(batch['inputs'])
    prediction['presence_logits'] = torch.full_like(prediction['presence_logits'], 1.)
    row = module.measure_batch(prediction, batch).compute()
    assert row['false_positive_actions'] > 0 and row['false_negative_actions'] == 0
    assert row['predicted_cardinality_mean'] == len(prediction['presence_logits'])
    assert math.isfinite(row['normalized_payload_rmse'])
    prediction['presence_logits'].fill_(-1.)
    empty = module.measure_batch(prediction, batch).compute()
    assert empty['predicted_cardinality_mean'] == 0
    assert empty['false_negative_actions'] == 1
    assert empty['normalized_payload_rmse'] is None


def test_measure_batch_includes_zero_threshold_and_empty_target_convention():
    module = api(); batch, model, _ = case()
    prediction = model(batch['inputs'])
    prediction['presence_logits'] = torch.full_like(prediction['presence_logits'], -1.)
    prediction['presence_logits'][batch['supervision']['resource']['candidate'][0]] = 0.
    row = module.measure_batch(prediction, batch).compute()
    assert row['set_exact_match'] == 1 and row['predicted_cardinality_mean'] == 1


@pytest.mark.parametrize('max_norm', [0., -1., float('nan'), True])
def test_train_batch_rejects_invalid_gradient_clip(max_norm):
    module = api()
    with pytest.raises(ValueError): module.train_batch(None, None, None, max_norm)


@pytest.mark.parametrize('relation', ['same', 'child', 'ancestor'])
def test_output_source_collisions_fail_before_creation(tmp_path, relation):
    module = api(); source = tmp_path/'source'
    output = {'same': source, 'child': source/'run', 'ancestor': tmp_path}[relation]
    with pytest.raises(ValueError, match='overlap'):
        module.main(['--g2', str(source), '--output', str(output)])
    assert not source.exists()


def static_fixture(tmp_path, monkeypatch):
    module = api()
    from drl.IL.tests.bc_v1.test_action_set_training import case
    from drl.IL.bc_v1.sidecar import _manifest_fingerprint
    from drl.IL.bc_v1.normalization import normalizer_fingerprint
    _, kw, _, _, manifest = case()
    manifest['identity']['f2_fingerprint'] = normalizer_fingerprint(kw['normalizers'])
    manifest['manifest_fingerprint'] = _manifest_fingerprint(manifest)
    source = tmp_path/'source'; source.mkdir(); (source/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(module, 'load_sources', lambda: {'kwargs': kw, 'identity': manifest['identity']})
    monkeypatch.setattr(module, '_read_manifest', lambda root, identity: manifest)
    def forbidden(*args, **kwargs): raise AssertionError('model/GPU/dataset accessed in dry-run')
    monkeypatch.setattr(module, 'build_scratch_actor', forbidden)
    monkeypatch.setattr(module, 'open_dataset', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    return module, source, manifest


def test_default_cli_is_static_dry_run_without_output_directory(tmp_path, monkeypatch, capsys):
    module, source, _ = static_fixture(tmp_path, monkeypatch)
    output = tmp_path/'newrun'
    assert module.main(['--g2', str(source), '--output', str(output)]) == 0
    row = json.loads(capsys.readouterr().out)
    assert row['mode'] == 'DRY_RUN' and row['model_contract'] == 'action_set_v1'
    assert row['TEST_payload_access'] == 0 and row['sampler_quotas'] == {'ordinary': 48, 'structural': 16}
    assert not output.exists()


def test_static_test_membership_is_sealed_before_dataset_or_model(tmp_path, monkeypatch):
    module, source, manifest = static_fixture(tmp_path, monkeypatch)
    from drl.IL.bc_v1.sidecar import _hash, _manifest_fingerprint
    manifest['inventory']['test'] = [{'episode_id': 'sealed', 'samples': 1, 'transition_indices_fingerprint': 'x'}]
    manifest['membership_fingerprint'] = _hash(manifest['inventory'])
    manifest['manifest_fingerprint'] = _manifest_fingerprint(manifest)
    (source/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='TEST'): module.main(['--g2', str(source), '--output', str(tmp_path/'newrun')])


def test_static_f2_mismatch_fails_before_dataset_or_model(tmp_path, monkeypatch):
    module, source, manifest = static_fixture(tmp_path, monkeypatch)
    from drl.IL.bc_v1.sidecar import _manifest_fingerprint
    manifest['identity']['f2_fingerprint'] = 'wrong'
    manifest['manifest_fingerprint'] = _manifest_fingerprint(manifest)
    (source/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='preprocessing'): module.main(['--g2', str(source), '--output', str(tmp_path/'newrun')])


def test_help_exits_without_static_sources(monkeypatch, capsys):
    module = api()
    def forbidden(*args, **kwargs): raise AssertionError('help read sources')
    monkeypatch.setattr(module, 'load_sources', forbidden)
    with pytest.raises(SystemExit) as result: module.main(['--help'])
    assert result.value.code == 0 and '--run' in capsys.readouterr().out


def test_raw_metrics_empty_graph_has_perfect_empty_set_and_no_numeric_support():
    module = api()
    from drl.IL.tests.bc_v1.g2_fixtures import fixture, tensorize
    from drl.IL.bc_v1.action_set_targets import collate_action_set_samples
    sample, kwargs = fixture(0, 0)
    batch = collate_action_set_samples([tensorize(sample, kwargs)])
    prediction = {'presence_logits': torch.empty(0), 'candidate_graph': torch.empty(0, dtype=torch.int64),
        'parameters': {'resource': {'payload_candidate': torch.empty(0, dtype=torch.int64)},
                       'structural': {'arguments': {'numeric': {'argument': torch.empty(0, dtype=torch.int64)}}}}}
    row = module.measure_batch(prediction, batch).compute()
    assert row['samples'] == 1 and row['set_exact_match'] == row['set_jaccard'] == 1.
    assert row['predicted_cardinality_mean'] == row['target_cardinality_mean'] == 0.
    assert row['normalized_payload_rmse'] is None


def test_synthetic_validation_aggregates_every_sample_once_and_keeps_model_in_eval():
    module = api()
    from drl.IL.tests.bc_v1.test_action_set_training import build
    from drl.IL.tests.bc_v1.g2_fixtures import tensorize
    sample, kwargs, _, _, _, model, _, _ = build()
    class Samples(list):
        def get_for_collate(self, index): return self[index]
    data = Samples([tensorize(sample, kwargs)] * 3)
    grouped = module.evaluate(model, data, 'cpu', batch_size=2)
    singles = module.evaluate(model, data, 'cpu', batch_size=1)
    assert grouped['total_loss'] == pytest.approx(singles['total_loss'], rel=1e-5)
    assert grouped['families']['presence']['support'] == 3
    assert grouped['families']['resource_payload']['support'] == 3
    assert grouped['metrics']['samples'] == 3 and not model.training
    with pytest.raises(ValueError, match='Empty validation'): module.evaluate(model, Samples(), 'cpu')


def test_raw_membership_with_configured_physical_metrics_does_not_regress_omitted_values():
    module = api(); batch, model, _ = case()
    from drl.IL.bc_v1.action_set_metrics import ActionSetMetrics
    prediction = model(batch['inputs']); prediction['presence_logits'] = torch.ones_like(prediction['presence_logits'])
    metrics = ActionSetMetrics(parameter_units={'payload': 'rad/s'})
    result = module.measure_batch(prediction, batch, metrics).compute()
    assert result['numeric'] == {} and math.isfinite(result['normalized_payload_rmse'])


def test_run_retains_optimized_train_val_cache_budgets_before_any_epoch(tmp_path, monkeypatch):
    """Catch accidental fallback to tiny TrainingDataset defaults in real run wiring."""
    from drl.IL.bc_v1.training_data import TrainingDataset
    from drl.IL.tests.bc_v1.test_sidecar import built, identity
    module = api(); source = built(tmp_path)
    budgets = {'train': (2*1024**3, 4*1024**3), 'validation': (256*1024**2, 512*1024**2)}
    opened = []
    class BeforeEpoch(Exception): pass
    real_open = module.open_dataset
    def open_checked(root, split, ids, **kwargs):
        data = real_open(root, split, ids, **kwargs)
        try:
            assert (data._dynamic.budget, data._templates.budget) == budgets[split]
        except BaseException:
            data.close(); raise
        opened.append(data)
        if split == 'validation':
            data.close(); raise BeforeEpoch
        return data
    monkeypatch.setattr(module, 'open_dataset', open_checked)
    monkeypatch.setattr(module, 'build_scratch_actor', lambda *a, **k: (None, None, {}))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    config = json.loads((Path(module.ROOT)/'configs/bc_v1/training_action_set_v1.json').read_text())
    prepared = ({'sampler_quotas': {'ordinary': 48, 'structural': 16}}, {'spec': None, 'feature_schema': None}, {}, {}, {'identity': identity()})
    with pytest.raises(BeforeEpoch): module.run(tmp_path/'run', source, config, prepared)
    assert len(opened) == 2 and all(d._closed for d in opened)
    assert not (tmp_path/'run').exists()


def test_run_sets_proven_cpu_thread_and_fp32_policy_before_model_factory(tmp_path, monkeypatch):
    module = api()
    previous = (torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    class BeforeModel(Exception): pass
    def factory(*args, **kwargs):
        assert torch.get_num_threads() == 1
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        raise BeforeModel
    monkeypatch.setattr(module, 'build_scratch_actor', factory)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    config = json.loads((Path(module.ROOT)/'configs/bc_v1/training_action_set_v1.json').read_text())
    try:
        with pytest.raises(BeforeModel): module.run(tmp_path/'newrun', tmp_path/'g2', config, ({}, {'spec': None, 'feature_schema': None}, {}, {}, {}))
    finally:
        torch.set_num_threads(previous[0])
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous[1:]
    assert not (tmp_path/'newrun').exists()
