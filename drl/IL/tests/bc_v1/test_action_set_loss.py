"""CPU contracts for balanced set presence and positive-only F2 regression."""
import importlib
import importlib.util
import math

import pytest
import torch


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.action_set_loss'), 'Missing ActionSet loss'
    return importlib.import_module('drl.IL.bc_v1.action_set_loss')


def test_missing_and_extra_actions_have_opposite_gradients_with_graph_balancing():
    logits = torch.tensor([0., 0., 0., 0., 0.], requires_grad=True)
    value = api().presence_loss(logits, torch.tensor([True, False, False, False, True]),
                                torch.tensor([0, 0, 0, 0, 1]), 2)
    assert float(value.detach()) == pytest.approx(1.5 * math.log(2.))
    value.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([-.25, 1/12, 1/12, 1/12, -.25]))


def test_negative_duplication_and_candidate_permutation_do_not_change_balanced_loss():
    loss = api().presence_loss
    expected = loss(torch.tensor([-2., 3.]), torch.tensor([True, False]), torch.tensor([0, 0]), 1)
    actual = loss(torch.tensor([3., -2., 3., 3.]), torch.tensor([False, True, False, False]),
                  torch.zeros(4, dtype=torch.int64), 1)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize('labels,groups,count,wanted', [
    ([False, False], [0, 0], 1, math.log(2.)),
    ([True, True], [0, 0], 1, math.log(2.)),
    ([], [], 2, 0.), ([], [], 0, 0.),
    ([False], [0], 2, math.log(2.)/2)])
def test_empty_positive_negative_and_candidate_groups(labels, groups, count, wanted):
    logits = torch.zeros(len(labels), requires_grad=True)
    value = api().presence_loss(logits, torch.tensor(labels, dtype=torch.bool),
                                torch.tensor(groups, dtype=torch.int64), count)
    assert float(value.detach()) == pytest.approx(wanted)
    value.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


@pytest.mark.parametrize('fault', ['labels_float', 'labels_shape', 'graph_dtype', 'graph_shape',
                                  'negative_graph', 'outside_graph', 'logits_shape', 'nonfinite', 'bad_count'])
def test_presence_rejects_malformed_binary_abi(fault):
    z = torch.zeros(2)
    y = torch.tensor([True, False]); g = torch.tensor([0, 0]); n = 1
    if fault == 'labels_float': y = y.float()
    elif fault == 'labels_shape': y = y[:1]
    elif fault == 'graph_dtype': g = g.float()
    elif fault == 'graph_shape': g = g[:, None]
    elif fault == 'negative_graph': g[0] = -1
    elif fault == 'outside_graph': g[0] = 1
    elif fault == 'logits_shape': z = z[:, None]
    elif fault == 'nonfinite': z[0] = float('nan')
    else: n = True
    with pytest.raises(ValueError): api().presence_loss(z, y, g, n)


def parameter_example():
    from drl.IL.tests.bc_v1.test_losses import rich_joint
    legacy, batch, _ = rich_joint()
    resource = batch['inputs']['resource_candidates']; structural = batch['inputs']['structural']
    count = len(resource['resource']) + len(structural['initiator'])
    labels = torch.zeros(count, dtype=torch.bool)
    labels[batch['supervision']['resource']['candidate']] = True
    labels[len(resource['resource']) + batch['supervision']['structural']['candidate']] = True
    batch['supervision']['action_set'] = {'presence': labels}
    parameters = {'resource': {k: v for k, v in legacy['resource'].items() if not k.startswith('mode_')},
                  'structural': {k: legacy['structural'][k] for k in ('arguments', 'scope')}}
    def leaves(tree):
        return {k: leaves(v) if isinstance(v, dict) else v.detach().clone().requires_grad_(v.is_floating_point())
                for k, v in tree.items()}
    prediction = {'presence_logits': torch.zeros(count, requires_grad=True),
                  'candidate_graph': torch.zeros(count, dtype=torch.int64), 'parameters': leaves(parameters)}
    return prediction, batch


def test_positive_parameter_codecs_keep_existing_support_and_absent_gradients_zero():
    prediction, batch = parameter_example()
    total, stats = api().compute_action_set_loss(prediction, batch)
    assert stats['presence']['support'] == 1
    assert stats['resource_payload']['support'] == 1
    assert stats['arg_numeric']['support'] == 2
    assert stats['arg_reference_set']['support'] == 2
    assert 'resource_mode' not in stats and 'structural_choice' not in stats
    assert total == sum(row['mean'] for row in stats.values())
    total.backward()
    resource = prediction['parameters']['resource']; selected = set(batch['supervision']['resource']['candidate'].tolist())
    for i, q in enumerate(resource['payload_candidate'].tolist()):
        lo, hi = resource['payload_ptr'][i:i+2].tolist()
        if q not in selected: assert torch.equal(resource['payload'].grad[lo:hi], torch.zeros(hi-lo))
    labels = batch['supervision']['action_set']['presence']
    assert (prediction['presence_logits'].grad[labels] < 0).all()
    assert (prediction['presence_logits'].grad[~labels] > 0).all()


@pytest.mark.parametrize('fault', ['cross_graph', 'truncated', 'negative_parameter', 'unbacked_positive'])
def test_full_loss_rejects_misaligned_physical_candidates_and_parameter_targets(fault):
    prediction, batch = parameter_example()
    if fault == 'cross_graph': prediction['candidate_graph'][0] = 1
    elif fault == 'truncated': prediction['presence_logits'] = prediction['presence_logits'][:-1]
    elif fault == 'negative_parameter':
        batch['supervision']['action_set']['presence'][batch['supervision']['resource']['candidate'][0]] = False
    else:
        y = batch['supervision']['action_set']['presence']; y[torch.nonzero(~y)[0]] = True
    with pytest.raises(ValueError): api().compute_action_set_loss(prediction, batch)


def test_actual_action_set_collator_and_policy_preserve_two_graph_support_and_backbone_gradient():
    from drl.IL.bc_v1.action_set_targets import collate_action_set_samples
    from drl.IL.tests.bc_v1.g2_fixtures import fixture, tensorize, with_structural
    from drl.IL.tests.bc_v1.test_action_set_head import policy_for
    a, kwargs = fixture(2)
    b, _ = fixture(4)
    batch = collate_action_set_samples([tensorize(with_structural(a), kwargs), tensorize(b, kwargs)])
    policy = policy_for(kwargs)
    prediction = policy(batch['inputs'])
    total, stats = api().compute_action_set_loss(prediction, batch)
    assert torch.isfinite(total)
    assert stats['presence']['support'] == 2
    assert stats['resource_payload']['support'] == 2
    assert stats['arg_category']['support'] == 2
    total.backward()
    assert policy.head.presence_scorer[-1].weight.grad.abs().sum() > 0
    assert policy.backbone.module_fusion[0].weight.grad.abs().sum() > 0


def test_absent_nonfinite_payload_never_enters_parameter_regression():
    prediction, batch = parameter_example()
    payload = prediction['parameters']['resource']
    target = batch['supervision']['resource']
    selected = set(target['candidate'].tolist())
    for row, candidate in enumerate(payload['payload_candidate'].tolist()):
        start, end = payload['payload_ptr'][row:row+2].tolist()
        if candidate not in selected:
            with torch.no_grad(): payload['payload'][start:end] = float('nan')
    total, stats = api().compute_action_set_loss(prediction, batch)
    assert torch.isfinite(total) and stats['resource_payload']['support'] == 1
    total.backward()
    assert torch.isfinite(payload['payload'].grad).all()
