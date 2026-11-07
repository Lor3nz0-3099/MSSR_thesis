"""CPU reward contracts for a real six-of-seven initial target snapshot."""
from dataclasses import replace

import pytest

from drl.MARL.c1_guided_feedback import TargetGraphProgress, load_target
from drl.MARL.tests.test_c1_phase_c_episode import evaluator, guided_frame
from drl.MARL.tests.test_c1_probe import move


def partial_frame(time_s=0.):
    full = guided_frame(time_s)
    return replace(full, connections=full.connections[:6])


def target_reward(result):
    return sum(v for k, v in result.reward_breakdown.items()
               if k.startswith('reward_target_'))


def test_prime_records_six_edges_without_an_episode_step_or_reward():
    e = evaluator(monitor=False)
    initial = partial_frame()
    e.prime(initial)
    assert e.steps == 0 and e.return_ == 0. and not e.breakdown and not e.counts
    assert e.last_result is None and not e.terminal
    assert e.last_frame == initial and e.last_assessment.valid_connections == 6
    assert len(e.seen_joint_ids) == 6
    assert e.target_progress.current_potential == pytest.approx(6/7)
    assert e.target_progress.best_fraction == pytest.approx(6/7)
    assert e.last_target_fraction == pytest.approx(6/7)
    assert e.best_lcc > 0 and e.best_merges == pytest.approx(6/7)
    assert e.stable_since is None and e.probe_anchor is None
    assert e.assembly_valid_timestamp is None and e.probe_start_timestamp is None
    assert e.probe_success_timestamp is None and not e.probe_attempts
    assert not e.stability_paid and e.best_probe == 0.
    assert e.stagnation_monitor.started is None
    assert e.episode_record() == {}


def test_first_holding_step_pays_discount_only_and_no_existing_docking():
    e = evaluator(monitor=False)
    e.prime(partial_frame())
    r = e.step(partial_frame(1/30))
    assert target_reward(r) == pytest.approx(-.06*6/7)
    assert r.reward_breakdown['reward_target_graph_progress'] == 0.
    assert r.reward_breakdown['reward_structural_progress'] == 0.
    assert r.reward_breakdown['reward_valid_docking'] == 0.
    assert e.counts['docking_count'] == 0 and e.steps == 1
    assert not r.assembly_stable and r.probe is None and not r.success


@pytest.mark.parametrize('next_edges, expected', [(7, 6*(.99-6/7)), (5, 6*(.99*5/7-6/7))])
def test_first_physical_completion_or_regression_uses_snapshot_baseline(next_edges, expected):
    e = evaluator(monitor=False)
    e.prime(partial_frame())
    full = guided_frame(1/30)
    r = e.step(replace(full, connections=full.connections[:next_edges]))
    assert target_reward(r) == pytest.approx(expected)
    assert e.counts['docking_count'] == int(next_edges == 7)
    assert e.counts['undocking_count'] == int(next_edges == 5)


@pytest.mark.parametrize('fault', ['missing_module', 'invalid_joint', 'dynamics', 'incomplete', 'collision'])
def test_prime_rejects_invalid_initial_physics_without_mutating_reward_state(fault):
    e = evaluator(monitor=False)
    f = partial_frame()
    if fault == 'missing_module':
        f = replace(f, modules=f.modules[:-1])
    elif fault == 'invalid_joint':
        f = replace(f, connections=(replace(f.connections[0], joint_valid=False), *f.connections[1:]))
    elif fault == 'dynamics':
        f = replace(f, peak_linear_speed=.5)
    elif fault == 'incomplete':
        f = replace(f, evidence_complete=False)
    else:
        f = replace(f, pathological_collision_owner_ids=('m0',))
    with pytest.raises(ValueError):
        e.prime(f)
    assert e.last_frame is None and e.steps == 0 and e.return_ == 0.
    assert e.target_progress.current_potential == 0.


def test_priming_cannot_reset_the_potential_between_physical_steps():
    e = evaluator(monitor=False)
    e.prime(partial_frame())
    with pytest.raises(RuntimeError):
        e.prime(partial_frame())
    e.step(partial_frame(1/30))
    with pytest.raises(RuntimeError):
        e.prime(partial_frame(2/30))


@pytest.mark.parametrize('kind', ['success', 'stagnation', 'unsafe'])
def test_true_terminal_target_return_is_minus_initial_potential_for_every_outcome(kind):
    e = evaluator(monitor=kind == 'stagnation')
    e.prime(partial_frame())
    total = 0.
    base = guided_frame()
    for i in range(1, 1001):
        f = partial_frame(i/30) if kind == 'stagnation' else replace(base, time_s=i/30)
        if kind == 'success' and i > 19:
            f = replace(move(base, dx=(i-19)*.002), time_s=i/30)
        if kind == 'unsafe' and i == 12:
            f = replace(f, modules=f.modules[:-1])
        r = e.step(f, decision_opportunity=True)
        total += .99**(i-1)*target_reward(r)
        if r.terminated:
            break
    assert r.terminated and not r.truncated
    assert r.success == (kind == 'success')
    assert e.target_progress.current_potential == 0.
    assert total == pytest.approx(-36/7, abs=1e-10)


def test_snapshot_truncation_retains_correct_shaped_value_bootstrap():
    e = evaluator(monitor=False)
    e.max_steps = 3
    e.prime(partial_frame())
    total = 0.
    for i in range(1, 4):
        r = e.step(replace(guided_frame(), time_s=i/30))
        total += .99**(i-1)*target_reward(r)
    assert r.truncated and not r.terminated
    assert total == pytest.approx(6*(.99**3-6/7))
    # V_shaped(s_T) = V_base(s_T) - 6 Phi(s_T).
    base_bootstrap = 2.
    shaped_bootstrap = base_bootstrap - 6.
    assert total + .99**3*shaped_bootstrap == pytest.approx(-36/7 + .99**3*base_bootstrap)


def test_real_smdp_options_telescope_from_partial_snapshot_with_one_discount_per_step():
    import torch
    from drl.MARL.c1_smdp import DiscountedOptionAccumulator
    full = guided_frame()
    partial = partial_frame()
    tracker = TargetGraphProgress(load_target('navigation'), tuple(m.module_id for m in full.modules))
    tracker.prime(partial)
    options = []
    for frames in ((partial, full), (full, partial, None)):
        accumulator = DiscountedOptionAccumulator(.99)
        for f in frames:
            reward = 6*sum(tracker.shaping(f or partial, discount_factor=.99, terminal=f is None).values())
            value = torch.tensor(reward, dtype=torch.float64)
            accumulator.add(team_reward=value, owner_rewards={'m0': value}, mean_field_rewards={'m0': value})
        options.append(accumulator)
    assert options[0].duration_steps == 2 and options[1].duration_steps == 3
    combined = options[0].team_reward + .99**2*options[1].team_reward
    assert combined.item() == pytest.approx(-36/7, abs=1e-12)


def test_target_tracker_prime_seeds_incompatible_edges_and_highwater_without_a_gain():
    f = partial_frame()
    wrong = replace(f.connections[0], connector_a='BOTTOM', connector_b='BOTTOM')
    f = replace(f, connections=(wrong,))
    tracker = TargetGraphProgress(load_target('navigation'), tuple(m.module_id for m in f.modules))
    tracker.prime(f, incompatible_weight=2.)
    assert tracker.current_potential == pytest.approx(-2/7)
    assert tracker.observe(f) == (0., 0.)
    parts = tracker.shaping(f, discount_factor=.99, incompatible_weight=2.)
    assert sum(parts.values()) == pytest.approx(.02/7)


@pytest.mark.parametrize('weight', [-1., float('nan'), float('inf')])
def test_target_tracker_prime_rejects_invalid_weight(weight):
    f = partial_frame()
    tracker = TargetGraphProgress(load_target('navigation'), tuple(m.module_id for m in f.modules))
    with pytest.raises(ValueError):
        tracker.prime(f, incompatible_weight=weight)


def test_primed_complete_target_must_still_observe_stability_and_functional_motion():
    e = evaluator(monitor=False)
    e.prime(guided_frame())
    first = e.step(guided_frame(1/30))
    assert first.target_graph_complete and first.fold_complete
    assert not first.assembly_stable and not first.functional_probe_success and not first.episode_success
    assert first.reward_breakdown['reward_task_probe_progress'] == 0.
    assert first.reward_breakdown['reward_structural_stability'] == 0.


def test_primed_unfolded_target_never_starts_probe_from_initial_topology_alone():
    e = evaluator(monitor=False)
    e.prime(guided_frame(folded=False))
    for i in range(1, 25):
        r = e.step(guided_frame(i/30, folded=False))
    assert r.target_graph_complete and not r.fold_complete
    assert not r.assembly_stable and r.probe is None and not r.episode_success


def test_legacy_highwater_rewards_do_not_credit_initial_six_edges():
    from drl.MARL.tests.test_c1_episode import evaluator as legacy_evaluator
    from drl.MARL.tests.test_c1_structure import frame
    e = legacy_evaluator()
    e.prime(frame(edges=6))
    r = e.step(frame(1/30, edges=6))
    assert r.reward_breakdown['reward_structural_progress'] == 0.
    assert r.reward_breakdown['reward_valid_docking'] == 0.
    r = e.step(frame(2/30, edges=7))
    assert r.reward_breakdown['reward_structural_progress'] == pytest.approx(1/7)
    assert r.reward_breakdown['reward_valid_docking'] == pytest.approx(1/7)


@pytest.fixture(scope='module')
def audited_wait():
    from docs.experiments.c1_phase_d_20261106_kq4rg8dq.reward.audit import evaluate_case
    return evaluate_case('wait')


@pytest.mark.parametrize('delay_s', [1, 5, 10, 20, 30, 60])
def test_actual_profile_delayed_success_and_wait_have_same_terminal_target_offset(delay_s, audited_wait):
    from docs.experiments.c1_phase_d_20261106_kq4rg8dq.reward.audit import evaluate_case
    successful = evaluate_case('success', delay_s)
    assert successful['success'] and successful['terminated'] and not successful['truncated']
    assert successful['returns']['target'] == pytest.approx(-36/7, abs=1e-10)
    assert audited_wait['returns']['target'] == pytest.approx(-36/7, abs=1e-10)
    assert successful['returns']['base'] > audited_wait['returns']['base']
    # Complete-episode reward ranking comes entirely from the base terms.
    assert successful['returns']['total']-audited_wait['returns']['total'] == pytest.approx(
        successful['returns']['base']-audited_wait['returns']['base'], abs=1e-10)


def test_late_completion_with_one_new_failed_align_can_lose_to_immediate_wait(audited_wait):
    from docs.experiments.c1_phase_d_20261106_kq4rg8dq.reward.audit import evaluate_case
    successful = evaluate_case('success', 20, failed_attempts=1)
    assert successful['success']
    assert successful['discounted_terms']['penalty_failed_alignment'] == pytest.approx(-.125)
    assert successful['returns']['base'] == pytest.approx(-1.117891078379, abs=1e-10)
    assert audited_wait['returns']['base'] == pytest.approx(-1.000537919542, abs=1e-10)
    assert successful['returns']['base'] < audited_wait['returns']['base']
    same_errors_wait = evaluate_case('wait', failed_attempts=1)
    # A mistake already paid by both continuations is a sunk cost.
    assert successful['returns']['base'] > same_errors_wait['returns']['base']


def test_repeated_failed_alignment_cannot_beat_wait_or_change_terminal_potential(audited_wait):
    from docs.experiments.c1_phase_d_20261106_kq4rg8dq.reward.audit import evaluate_case
    failed = evaluate_case('failed_align')
    assert failed['terminated'] and not failed['success']
    assert failed['reason'] == 'stagnation'
    assert failed['align_feedback_counts']['align_failed'] > 1
    assert failed['align_feedback_counts']['unproductive_retry'] > 1
    assert failed['returns']['target'] == pytest.approx(-36/7, abs=1e-10)
    assert failed['returns']['base'] < audited_wait['returns']['base']
