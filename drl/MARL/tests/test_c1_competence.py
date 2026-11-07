"""CPU-only checks for independent target-stage competence evidence."""
from types import SimpleNamespace
import json

import pytest


def policy(**kwargs):
    from drl.MARL.c1_competence import CompetencePatience
    return CompetencePatience(**kwargs)


def add_attempts(p, successes, failures=0, *, elapsed=10., failure_elapsed=120.,
                 domain='train-layout', stage='S0'):
    for _ in range(successes):
        p.observe(domain, stage, elapsed_s=elapsed, success=True)
    for _ in range(failures):
        p.observe(domain, stage, elapsed_s=failure_elapsed, success=False, censored=True)


def test_absent_and_unsuccessful_stages_remain_permissive():
    p = policy()
    assert p.patience('hard', 'S0') == 60.
    for _ in range(100):
        p.observe('hard', 'S0', elapsed_s=25., success=False, censored=True,
                  primitive_failures=3, recoveries=0, opportunities=4)
    s = p.snapshot('hard', 'S0')
    assert p.patience('hard', 'S0') == 60.
    assert s['attempts'] == 64 and s['successes'] == 0
    assert s['primitive_failures'] == 192 and s['opportunities'] == 256
    assert s['success_fraction'] == 0. and not s['mastered']


def test_mastery_requires_observations_and_robust_success_rate():
    p = policy()
    add_attempts(p, 19)
    assert p.patience('train-layout', 'S0') == 60.
    add_attempts(p, 1)
    assert p.patience('train-layout', 'S0') == 25.
    weak = policy()
    add_attempts(weak, 16, 4, failure_elapsed=1.)
    assert weak.snapshot('train-layout', 'S0')['wilson_lower_bound'] < .6
    assert weak.patience('train-layout', 'S0') == 60.


@pytest.mark.parametrize(('elapsed', 'expected'), [(1., 20.), (100., 90.)])
def test_mastered_duration_estimate_obeys_time_bounds(elapsed, expected):
    p = policy()
    add_attempts(p, 20, elapsed=elapsed)
    assert p.patience('train-layout', 'S0') == expected


def test_configurable_success_requirement_is_independent_of_success_rate():
    p = policy(config=dict(min_successes=19))
    add_attempts(p, 18, 2, failure_elapsed=1.)
    assert p.snapshot('train-layout', 'S0')['wilson_lower_bound'] >= .6
    assert p.patience('train-layout', 'S0') == 60.


def test_upper_success_duration_tail_sets_patience_instead_of_fast_majority():
    p = policy()
    add_attempts(p, 17, elapsed=1.)
    add_attempts(p, 3, elapsed=20.)
    assert p.patience('train-layout', 'S0') == 45.


def test_long_censored_tail_cannot_be_hidden_by_fast_successes():
    p = policy()
    add_attempts(p, 24, 4)
    s = p.snapshot('train-layout', 'S0')
    assert s['wilson_lower_bound'] >= .6
    assert s['successful_duration_quantile_s'] == 10.
    assert s['censor_aware_duration_quantile_s'] is None
    assert p.patience('train-layout', 'S0') == 60.


def test_domain_and_stage_statistics_do_not_leak_and_window_forgets_mastery():
    p = policy()
    add_attempts(p, 64)
    assert p.patience('train-layout', 'S0') == 25.
    assert p.patience('other-layout', 'S0') == 60.
    assert p.patience('train-layout', 'S1') == 60.
    add_attempts(p, 0, 64)
    assert p.patience('train-layout', 'S0') == 60.
    assert p.snapshot('train-layout', 'S0')['successes'] == 0


def test_evaluation_observations_and_queries_leave_training_state_unchanged():
    p = policy()
    add_attempts(p, 20)
    before = p.state_dict()
    for split in ('VALIDATION', 'test'):
        assert not p.observe('train-layout', 'S0', elapsed_s=100., success=False, split=split)
    for _ in range(100):
        p.patience('unseen', 'S3')
    assert p.state_dict() == before
    frozen = type(p).from_state_dict(before, training=False)
    assert not frozen.observe('train-layout', 'S0', elapsed_s=1., success=True)
    assert frozen.patience('train-layout', 'S0') == 25.


def test_serialized_state_roundtrip_preserves_counters_and_independent_windows():
    p = policy()
    add_attempts(p, 20)
    p.observe('x', 'S2', elapsed_s=0., success=False, censored=True,
              primitive_failures=2, recoveries=1, opportunities=7)
    restored = type(p).from_state_dict(json.loads(json.dumps(p.state_dict())))
    assert restored.snapshot() == p.snapshot()
    assert restored.snapshot('x', 'S2')['recoveries'] == 1
    assert restored.snapshot('x', 'S2')['successes'] == 0


@pytest.mark.parametrize('kwargs', [dict(elapsed_s=-1., success=False),
    dict(elapsed_s=float('nan'), success=False), dict(elapsed_s=0., success=True),
    dict(elapsed_s=1., success=True, censored=True),
    dict(elapsed_s=1., success=False, opportunities=-1)])
def test_invalid_observations_cannot_contaminate_learning(kwargs):
    p = policy()
    with pytest.raises(ValueError):
        p.observe('x', 'S0', **kwargs)
    assert p.snapshot()['stages'] == []


def test_invalid_configuration_cannot_create_unbounded_or_premature_patience():
    for config in (dict(window_size=0), dict(min_observations=65),
                   dict(min_successes=0), dict(fallback_patience_s=121.),
                   dict(wilson_z=0.), dict(duration_quantile=1.1)):
        with pytest.raises(ValueError):
            policy(config=config)


def tracker(p, **kwargs):
    from drl.MARL.c1_competence import CompetenceEpisodeTracker
    return CompetenceEpisodeTracker('layout', 0, 0., p, **kwargs)


def test_tracker_records_real_edge_progress_without_retry_or_regression_resets():
    p = policy()
    t = tracker(p)
    failure = SimpleNamespace(goal_id='g1', state='failed')
    assert t.observe(10., 0, primitive_events=(failure,), decision_opportunity=True) == []
    assert t.observe(20., 0, primitive_events=(failure,), decision_opportunity=True) == []
    first, = t.observe(30., 2, decision_opportunity=True)
    assert (first['stage'], first['elapsed_s'], first['target_count_before'],
            first['target_count_after']) == ('S0', 30., 0, 2)
    assert first['primitive_failures'] == 1 and first['recoveries'] == 1
    assert first['opportunities'] == 3 and first['success']
    assert t.observe(35., 1) == []
    assert t.observe(40., 2) == []
    second, = t.observe(50., 3)
    assert (second['stage'], second['elapsed_s'], second['target_count_before']) == ('S1', 20., 2)
    assert p.snapshot('layout', 'S0')['attempts'] == 1


def test_tracker_terminal_censor_is_once_only_and_evaluation_is_frozen():
    p = policy()
    t = tracker(p, split='validation')
    attempt, = t.observe(60., 0, finished=True, truncated=True)
    assert attempt['censored'] and not attempt['success']
    assert t.observe(61., 0, finished=True) == []
    assert p.snapshot()['stages'] == []
    train = tracker(p)
    train.observe(60., 0, finished=True)
    assert p.snapshot('layout', 'S0')['censored'] == 1


def test_tracker_initial_baseline_and_terminal_edge_do_not_invent_failures():
    from drl.MARL.c1_competence import CompetenceEpisodeTracker
    p = policy()
    t = CompetenceEpisodeTracker('layout', 5, 10., p)
    record, = t.observe(25., 7, finished=True)
    assert record['stage'] == 'S3' and record['elapsed_s'] == 15.
    assert record['success'] and not record['censored']
    assert p.snapshot('layout', 'S3')['attempts'] == 1
    assert p.snapshot('layout', 'FOLD')['attempts'] == 0


@pytest.mark.parametrize(('count', 'stage'), [(0, 'S0'), (1, 'S1'), (2, 'S1'),
    (3, 'S2'), (4, 'S2'), (5, 'S3'), (6, 'S3'), (7, 'FOLD')])
def test_tracker_uses_actual_target_count_stage(count, stage):
    from drl.MARL.c1_competence import CompetenceEpisodeTracker
    p = policy()
    t = CompetenceEpisodeTracker('layout', count, 0., p)
    t.observe(30., count, finished=True)
    assert p.snapshot('layout', stage)['attempts'] == 1


def test_tracker_rejects_backwards_time_and_out_of_range_count():
    t = tracker(policy())
    t.observe(2., 0)
    with pytest.raises(ValueError):
        t.observe(1., 0)
    with pytest.raises(ValueError):
        t.observe(3., 8)
