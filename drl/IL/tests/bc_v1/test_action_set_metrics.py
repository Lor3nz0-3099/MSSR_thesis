"""Streaming membership and unit-separated matched numeric diagnostics."""
from dataclasses import replace
import importlib
import importlib.util
import math

import pytest

from drl.IL.bc_v1.bc_v1_contract import ActionChoice, ConnectorRef, ModuleRef, ParameterValue


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.action_set_metrics'), 'Missing ActionSet metrics'
    return importlib.import_module('drl.IL.bc_v1.action_set_metrics')


def bundle(*actions):
    from drl.IL.bc_v1.action_set import ActionBundle
    return ActionBundle(tuple(actions))


def choice(peer=1, action='connect', value=2.):
    a, b = ModuleRef('m0'), ModuleRef(f'm{peer}')
    return ActionChoice(action, a, participants=(a, b),
                        connectors=(ConnectorRef(a, 'front'), ConnectorRef(b, 'back')),
                        parameters=(ParameterValue('angle', value),))


def test_missing_metrics_module_is_an_assertion_failure():
    assert api().ActionSetMetrics


def test_wrong_peer_counts_one_false_negative_and_one_false_positive():
    metrics = api().ActionSetMetrics()
    metrics.update(bundle(choice(1)), bundle(choice(2)))
    result = metrics.compute()
    assert (result['true_positive'], result['false_positive'], result['false_negative']) == (0, 1, 1)
    assert result['precision'] == result['recall'] == result['f1'] == 0
    assert result['exact_match'] == result['jaccard'] == 0
    assert result['per_action']['connect']['support'] == 1


def test_membership_ignores_values_and_order_but_numeric_uses_only_matched_candidates():
    metrics = api().ActionSetMetrics(parameter_units={('connect', 'angle'): 'rad'}, tolerances={'rad': .5})
    target = (choice(1, value=2.), choice(2, value=4.))
    predicted = (choice(3, value=1000.), choice(1, value=3.))
    metrics.update(bundle(*target), bundle(*predicted))
    metrics.update(bundle(*reversed(target)), bundle(*target))
    row = metrics.compute()
    assert (row['true_positive'], row['false_positive'], row['false_negative']) == (3, 1, 1)
    assert row['precision'] == row['recall'] == row['f1'] == .75
    assert row['exact_match'] == .5
    assert row['jaccard'] == pytest.approx(2/3)
    error = row['numeric']['connect.angle']['rad']
    assert error['support'] == 3 and error['mae'] == pytest.approx(1/3)
    assert error['rmse'] == pytest.approx(math.sqrt(1/3))
    assert error['within_tolerance'] == pytest.approx(2/3)


def test_empty_set_conventions_cardinality_and_streaming_reset():
    metrics = api().ActionSetMetrics()
    metrics.update(bundle(), bundle())
    row = metrics.compute()
    assert row['precision'] == row['recall'] == row['f1'] == row['jaccard'] == row['exact_match'] == 1.
    metrics.update(bundle(choice()), bundle())
    row = metrics.compute()
    assert row['precision'] == row['recall'] == row['f1'] == 0.
    assert row['target_cardinality'] == .5 and row['predicted_cardinality'] == 0
    assert row['cardinality_absolute_error'] == .5
    metrics.reset()
    assert metrics.compute()['samples'] == 0


def test_identity_ignores_participants_payload_and_parameters_but_orders_connectors():
    action = choice()
    changed = replace(action, participants=(), parameters=(), payload=(99.,))
    assert api().action_identity(action) == api().action_identity(changed)
    assert api().action_identity(action) != api().action_identity(replace(action, connectors=action.connectors[::-1]))


def test_numeric_units_tolerance_and_normalized_payload_are_kept_separate():
    metrics = api().ActionSetMetrics()
    metrics.update_numeric('payload', [1., 3.], [0., 1.], 'm', tolerance=1.)
    metrics.update_numeric('payload', [4.], [1.], 'rad', tolerance=.1)
    metrics.update_numeric('payload', [2.], [0.], '1', normalized=True)
    row = metrics.compute()
    assert row['numeric']['payload']['m']['mae'] == 1.5
    assert row['numeric']['payload']['m']['within_tolerance'] == .5
    assert row['numeric']['payload']['rad']['rmse'] == 3.
    assert row['normalized_payload_rmse'] == 2.


@pytest.mark.parametrize('predicted,target,unit,tolerance', [([float('nan')], [0.], 'm', 1.),
    ([1.], [0., 1.], 'm', 1.), ([1.], [0.], '', 1.), ([1.], [0.], 'm', -1.)])
def test_numeric_rejects_invalid_values_shapes_units_and_tolerances(predicted, target, unit, tolerance):
    with pytest.raises(ValueError): api().ActionSetMetrics().update_numeric('payload', predicted, target, unit, tolerance)


def test_public_set_metrics_names_expose_streaming_values_and_per_action_precision_recall():
    metrics = api().ActionSetMetrics()
    metrics.update(bundle(choice(action='alpha')), bundle(choice(action='alpha'), choice(action='beta')))
    result = metrics.compute()
    assert result['set_exact_match'] == 0
    assert result['set_precision'] == .5 and result['set_recall'] == 1.
    assert result['set_f1'] == pytest.approx(2/3) and result['set_jaccard'] == .5
    assert result['target_cardinality_mean'] == 1 and result['predicted_cardinality_mean'] == 2
    assert result['false_positive_actions'] == 1 and result['false_negative_actions'] == 0
    assert result['per_action']['alpha']['precision'] == result['per_action']['alpha']['recall'] == 1
    assert result['per_action']['beta']['precision'] == result['per_action']['beta']['recall'] == 0


def test_streaming_cardinality_ranges_include_empty_sets_and_reset():
    metrics=api().ActionSetMetrics()
    empty=metrics.compute()
    for key in ('target_cardinality_min','target_cardinality_max','predicted_cardinality_min','predicted_cardinality_max'):
        assert empty[key] is None
    metrics.update(bundle(),bundle(choice(1),choice(2)))
    metrics.update(bundle(choice(1),choice(2),choice(3)),bundle())
    result=metrics.compute()
    assert (result['target_cardinality_min'],result['target_cardinality_max'])==(0,3)
    assert (result['predicted_cardinality_min'],result['predicted_cardinality_max'])==(0,2)
    assert result['target_cardinality_mean']==1.5 and result['predicted_cardinality_mean']==1.
    metrics.reset()
    assert metrics.compute()['target_cardinality_min'] is None
