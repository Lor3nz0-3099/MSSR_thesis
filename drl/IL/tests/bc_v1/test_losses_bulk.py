"""Synthetic CPU parity and fail-closed contracts for opt-in bulk BC losses."""
import importlib.util
import math

import pytest
import torch

from drl.IL.bc_v1.batching import collate_policy_samples
from drl.IL.bc_v1.losses import compute_bc_loss, load_loss_config
from drl.IL.tests.bc_v1.g2_fixtures import ROOT
from drl.IL.tests.bc_v1.test_losses import rich_joint
from drl.IL.tests.bc_v1.test_resource_heads_v3 import example


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def bulk():
    assert importlib.util.find_spec('drl.IL.bc_v1.losses_bulk'), 'Missing opt-in bulk BC losses'
    from drl.IL.bc_v1.losses_bulk import compute_bc_loss_bulk
    return compute_bc_loss_bulk


def leaves(prediction):
    result = {}
    for key, value in prediction.items():
        result[key] = (leaves(value) if isinstance(value, dict) else
                       value.detach().clone().requires_grad_(value.is_floating_point()))
    return result


def floats(tree, prefix=''):
    for key, value in tree.items():
        path = prefix + '/' + key
        if isinstance(value, dict):
            yield from floats(value, path)
        elif value.is_floating_point():
            yield path, value


def parity(prediction, batch):
    config = load_loss_config(ROOT / 'configs/bc_v1/loss_v1.json')
    expected, actual = leaves(prediction), leaves(prediction)
    want_total, want = compute_bc_loss(expected, batch, config)
    got_total, got = bulk()(actual, batch, config)
    torch.testing.assert_close(got_total, want_total, rtol=2e-6, atol=2e-7)
    for family in want:
        for field in ('sum', 'support', 'mean', 'weighted'):
            torch.testing.assert_close(got[family][field], want[family][field], rtol=2e-6, atol=2e-7)
    want_total.backward()
    got_total.backward()
    for (name, wanted), (actual_name, value) in zip(floats(expected), floats(actual)):
        assert name == actual_name
        assert wanted.grad is not None and value.grad is not None
        torch.testing.assert_close(value.grad, wanted.grad, rtol=3e-6, atol=2e-7, msg=name)
    return got


@pytest.mark.parametrize('kind,mode,payload', [
    ('parameterized', 1, 1), ('inactive', 1, 0), ('fixed', 1, 0), ('unknown', 0, 0)])
def test_v3_outcome_support_values_and_all_head_gradients_match_reference(kind, mode, payload):
    batch, model, _, _ = example(kind)
    result = parity(model(batch['inputs']), batch)
    assert result['resource_mode']['support'] == mode
    assert result['resource_payload']['support'] == payload
    assert result['structural_choice']['support'] == 3


def test_mixed_v3_batch_preserves_each_decision_and_masks_nan_components():
    batches = [example(kind)[0] for kind in ('parameterized', 'inactive', 'fixed', 'unknown')]
    batch = collate_policy_samples(batches)
    _, model, _, _ = example()
    prediction = model(batch['inputs'])
    batch['supervision']['resource']['payload_valid'][1] = False
    batch['supervision']['resource']['payload'][1] = float('nan')
    prediction['resource']['payload'][1] = float('nan')
    # Unknown outcomes and unrelated payload components never enter arithmetic.
    prediction['resource']['outcome_logits'][36:] = float('nan')
    result = parity(prediction, batch)
    assert result['resource_mode']['support'] == 3
    assert result['resource_payload']['support'] == 1


@pytest.mark.parametrize('dense', [False, True])
def test_legacy_and_dense_structural_arguments_match_reference(dense):
    prediction, batch, model = rich_joint()
    if dense:
        from drl.IL.bc_v1.structural_decisions import add_no_structural_supervision
        batch = add_no_structural_supervision(batch)
        prediction['structural']['no_structural_logits'] = torch.tensor([-.7, .2, .5])
    result = parity(prediction, batch)
    assert result['arg_numeric']['support'] == 2
    assert result['arg_reference_set']['support'] == 2
    assert result['arg_reference']['support'] == 2


def test_all_masks_false_and_nonfinite_heads_produce_exact_zero_and_zero_gradients():
    prediction, batch, _ = rich_joint()
    for family in batch['supervision'].values():
        for key, value in family.items():
            if value.dtype == torch.bool:
                value.fill_(False)
    for _, value in floats(prediction):
        value.data.fill_(float('nan'))
    result = parity(prediction, batch)
    assert all(row['support'] == 0 and row['sum'] == 0 for row in result.values())


@pytest.mark.parametrize('ptr,wanted', [([0, 0, 0, 0], 0.), ([0, 0, 2, 3], math.log(6.))])
def test_dense_structural_empty_and_unequal_domains_keep_none_decision_support(ptr, wanted):
    batch, model, _, _ = example('inactive')
    prediction = model(batch['inputs'])
    prediction['structural']['group_ptr'] = torch.tensor(ptr)
    prediction['structural']['candidate_logits'] = torch.zeros(ptr[-1])
    prediction['structural']['no_structural_logits'] = torch.zeros(3)
    result = parity(prediction, batch)
    assert result['structural_choice']['support'] == 3
    assert float(result['structural_choice']['sum'].detach()) == pytest.approx(wanted)


def test_mixed_categorical_and_numeric_payloads_share_decision_denominator():
    from drl.IL.tests.bc_v1.test_losses import categorical_resource_example
    prediction, batch = categorical_resource_example()
    batch['supervision']['resource'].update(
        resource=torch.tensor([0, 1]), candidate=torch.tensor([1, 2]),
        supervised=torch.tensor([True, True]), payload_ptr=torch.tensor([0, 1, 3]),
        payload=torch.tensor([float('nan'), 2., 3.]),
        payload_category=torch.tensor([1, -1, -1]), payload_valid=torch.ones(3, dtype=torch.bool))
    result = parity(prediction, batch)
    assert result['resource_payload']['support'] == 2


@pytest.mark.parametrize('fault', ['numeric_arity_masked', 'nonfinite_mode', 'nonfinite_payload',
                                  'bad_category', 'duplicate_argument', 'duplicate_membership',
                                  'truncated_payload_masked', 'truncated_argument_masked',
                                  'truncated_category'])
def test_active_argument_and_payload_integrity_matches_reference_rejections(fault):
    prediction, batch, _ = rich_joint()
    y = batch['supervision']['structural']
    if fault == 'numeric_arity_masked':
        r = batch['supervision']['resource']
        r['payload_valid'].fill_(False)
        prediction['resource']['payload_ptr'][1] += 1
    elif fault == 'nonfinite_mode':
        prediction['resource']['mode_logits'][0] = float('nan')
    elif fault == 'nonfinite_payload':
        prediction['resource']['payload'][0] = float('nan')
    elif fault == 'bad_category':
        y['argument_category'][y['argument_category'] >= 0] = 99
    elif fault == 'duplicate_argument':
        y['argument_domain'][1] = y['argument_domain'][0]
    elif fault == 'truncated_payload_masked':
        batch['supervision']['resource']['payload_valid'].fill_(False)
        prediction['resource']['payload'] = prediction['resource']['payload'][:1]
    elif fault == 'truncated_argument_masked':
        y['argument_value_valid'].fill_(False)
        records = prediction['structural']['arguments']['numeric']
        records['value'] = records['value'][:0]
    elif fault == 'truncated_category':
        records = prediction['structural']['arguments']['category']
        records['logits'] = records['logits'][:0]
    else:
        y['membership_domain'][1] = y['membership_domain'][0]
    config = load_loss_config(ROOT / 'configs/bc_v1/loss_v1.json')
    with pytest.raises(ValueError):
        compute_bc_loss(prediction, batch, config)
    with pytest.raises(ValueError):
        bulk()(prediction, batch, config)


@pytest.mark.parametrize('fault', ['extra_payload', 'unknown_target', 'dense_mapping',
                                  'duplicate_payload', 'target_grad', 'missing_payload_route'])
def test_malformed_v3_targets_fail_closed(fault):
    batch, model, _, _ = example()
    prediction = model(batch['inputs'])
    resource = batch['supervision']['resource']
    if fault == 'extra_payload':
        resource['decision_candidate'][0] = 0
    elif fault == 'unknown_target':
        resource['decision_candidate'][1] = 0
    elif fault == 'dense_mapping':
        batch['supervision']['structural']['decision_candidate'][0] = 0
    elif fault == 'duplicate_payload':
        resource['resource'][0] = 1
    elif fault == 'target_grad':
        resource['payload'].requires_grad_()
    else:
        prediction['resource']['payload_candidate'] = torch.empty(0, dtype=torch.int64)
    config = load_loss_config(ROOT / 'configs/bc_v1/loss_v1.json')
    with pytest.raises(ValueError):
        compute_bc_loss(prediction, batch, config)
    with pytest.raises(ValueError):
        bulk()(prediction, batch, config)


@pytest.mark.parametrize('mask', ['argument_supervised', 'supervised'])
def test_membership_owner_validation_precedes_parent_mask(mask):
    prediction, batch, _ = rich_joint()
    structural = batch['inputs']['structural']
    target = batch['supervision']['structural']
    target[mask].fill_(False)
    wrong = next(i for i, kind in enumerate(structural['argument_kind'].tolist()) if kind == 4)
    start, end = structural['argument_domain_ptr'][wrong:wrong+2].tolist()
    target['membership_domain'][:3] = torch.arange(start, end)
    config = load_loss_config(ROOT / 'configs/bc_v1/loss_v1.json')
    with pytest.raises(ValueError):
        bulk()(prediction, batch, config)
