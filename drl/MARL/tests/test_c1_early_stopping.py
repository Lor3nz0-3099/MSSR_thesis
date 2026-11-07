"""Validation-only stop decisions and measured wall-budget reservations."""
import json

import pytest

from drl.MARL.c1_early_stopping import ValidationEarlyStopping, WallBudget, validate_frozen_evaluation


SCORES = dict(navigation=1., stairs=2., gap=3., manipulation=4.)


def test_patience_counts_balanced_validation_scores_and_restores_best_checkpoint():
    stop = ValidationEarlyStopping(patience=2)
    first = stop.update('first.pt', 'validation', SCORES)
    assert first['balanced_score'] == 2.5
    assert first['improved'] and first['best_checkpoint'] == 'first.pt'
    assert not stop.update('equal.pt', 'validation', SCORES)['stop']
    resumed = ValidationEarlyStopping.from_state_dict(json.loads(json.dumps(stop.state_dict())))
    last = resumed.update('worse.pt', 'validation', dict.fromkeys(SCORES, 0.))
    assert last['stop']
    assert last['best_checkpoint'] == 'first.pt'
    assert last['bad_validations'] == 2


@pytest.mark.parametrize('split', ['train', 'test', 'challenge', 'TRAIN', 'val'])
def test_only_explicit_validation_split_is_accepted(split):
    stop = ValidationEarlyStopping(patience=2)
    with pytest.raises(ValueError, match='validation'):
        stop.update('candidate.pt', split, SCORES)
    assert stop.state_dict()['validation_count'] == 0


def test_scores_require_all_four_finite_domains_and_equal_domain_weight():
    stop = ValidationEarlyStopping(patience=2)
    for scores in ({'navigation': 1}, dict(SCORES, gap=float('nan')), dict(SCORES, extra=1)):
        with pytest.raises(ValueError):
            stop.update('bad.pt', 'validation', scores)
    assert stop.state_dict()['validation_count'] == 0
    assert stop.update('ok.pt', 'validation', SCORES)['balanced_score'] == 2.5


def test_min_delta_is_explicit_and_improvement_resets_patience():
    stop = ValidationEarlyStopping(patience=2, min_delta=.1)
    stop.update('first.pt', 'validation', SCORES)
    assert not stop.update('tiny.pt', 'validation', {d: v + .05 for d, v in SCORES.items()})['improved']
    result = stop.update('better.pt', 'validation', {d: v + .2 for d, v in SCORES.items()})
    assert result['improved'] and result['bad_validations'] == 0
    assert result['best_checkpoint'] == 'better.pt'


def test_budget_blocks_new_update_before_reserved_checkpoint_exceeds_cap():
    budget = WallBudget(max_wall_s=86400.)
    # Elapsed includes process startup, assembly, optimization and validation.
    assert budget.decision(elapsed_s=100.)['stop']  # no measured costs: fail closed
    budget.record_costs(update_s=100., checkpoint_s=20., validation_s=200.)
    decision = budget.decision(elapsed_s=86290.)
    assert decision['stop']
    assert decision['reserved_s'] == 120.
    assert not budget.decision(elapsed_s=86279.)['stop']
    assert budget.decision(elapsed_s=86190., next_operation='validation')['stop']
    assert budget.decision(elapsed_s=86400., next_operation='checkpoint')['stop']


def test_budget_resumes_measured_costs_without_ignoring_elapsed_time():
    budget = WallBudget(max_wall_s=3600.)
    budget.record_costs(update_s=10., checkpoint_s=5.)
    restored = WallBudget.from_state_dict(json.loads(json.dumps(budget.state_dict())))
    assert restored.decision(elapsed_s=3590.)['stop']
    assert not restored.decision(elapsed_s=3500.)['stop']


def test_frozen_validation_evidence_rejects_updates_wrong_split_or_missing_proof():
    result = dict(split='validation', optimizer_updates=0,
                  parameter_sha256_before='a' * 64, parameter_sha256_after='a' * 64,
                  checkpoint_path='frozen.pt', score_by_domain=SCORES,
                  seed_manifest={'navigation': [8001]}, command=['python', 'evaluate'],
                  peak_gpu_bytes=100, available_gpu_bytes=200)
    assert validate_frozen_evaluation(result)['balanced_score'] == 2.5
    for changes in (dict(optimizer_updates=1), dict(split='test'),
                    dict(parameter_sha256_after='b' * 64), dict(seed_manifest={}),
                    dict(available_gpu_bytes=50)):
        with pytest.raises(ValueError):
            validate_frozen_evaluation(dict(result, **changes))


def test_domain_floor_rejects_mean_gain_that_collapses_one_bc_skill():
    stop = ValidationEarlyStopping(patience=2, preserve_domain_floor=True)
    stop.update('bc.pt', 'validation', SCORES, success_by_domain=dict.fromkeys(SCORES, 0.))
    result = stop.update('masked-collapse.pt', 'validation', dict(SCORES, navigation=100., gap=2.),
                         success_by_domain=dict.fromkeys(SCORES, 0.))
    assert result['balanced_score'] > 2.5
    assert not result['improved'] and result['best_checkpoint'] == 'bc.pt'
    assert result['guard_violations'] == {'gap': 'return_below_bc_baseline_at_equal_success'}
    assert result['bad_validations'] == 1


def test_macro_success_gain_allows_own_return_drop_but_never_other_domain_collapse():
    stop = ValidationEarlyStopping(patience=2, preserve_domain_floor=True)
    success = dict.fromkeys(SCORES, .5)
    stop.update('bc.pt', 'validation', SCORES, success_by_domain=success)
    candidate_success = dict(success, gap=.75)
    gain = stop.update('gain.pt', 'validation', dict(SCORES, gap=-10.), success_by_domain=candidate_success)
    assert gain['improved'] and gain['macro_success'] == .5625
    collapse = stop.update('collapse.pt', 'validation', dict(SCORES, gap=-5., stairs=100.),
                           success_by_domain=dict(candidate_success, stairs=.25))
    assert not collapse['improved']
    assert collapse['guard_violations'] == {'stairs': 'success_below_bc_baseline'}
    assert collapse['best_checkpoint'] == 'gain.pt'


def test_missing_or_zero_success_uses_guarded_return_tiebreak_and_serializes_baseline():
    stop = ValidationEarlyStopping(patience=2, preserve_domain_floor=True)
    stop.update('bc.pt', 'validation', SCORES)
    better = stop.update('better.pt', 'validation', {d: value+1 for d, value in SCORES.items()})
    assert better['improved'] and better['macro_success'] == 0.
    restored = ValidationEarlyStopping.from_state_dict(json.loads(json.dumps(stop.state_dict())))
    result = restored.update('unsafe.pt', 'validation', dict(SCORES, gap=2., navigation=50.))
    assert not result['improved'] and result['best_checkpoint'] == 'better.pt'
    assert restored.state_dict()['baseline_score_by_domain'] == SCORES
    assert restored.state_dict()['baseline_success_by_domain'] == dict.fromkeys(SCORES, 0.)
    assert restored.state_dict()['preserve_domain_floor']


def test_success_is_primary_metric_and_must_be_four_probabilities():
    stop = ValidationEarlyStopping(patience=2, preserve_domain_floor=True)
    stop.update('bc.pt', 'validation', SCORES, success_by_domain=dict.fromkeys(SCORES, 0.))
    for invalid in ({'gap': 0.}, dict.fromkeys(SCORES, 1.1), dict.fromkeys(SCORES, float('nan'))):
        with pytest.raises(ValueError):
            stop.update('invalid.pt', 'validation', SCORES, success_by_domain=invalid)
    assert stop.state_dict()['validation_count'] == 1
    stop.update('success.pt', 'validation', SCORES, success_by_domain=dict.fromkeys(SCORES, .5))
    result = stop.update('returns-only.pt', 'validation', dict.fromkeys(SCORES, 100.),
                         success_by_domain=dict.fromkeys(SCORES, .25))
    assert not result['improved'] and result['best_checkpoint'] == 'success.pt'
