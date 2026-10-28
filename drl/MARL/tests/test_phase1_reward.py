from __future__ import annotations

import pytest

from smores_ep.isaac.phase1_training_course import (
    phase1_rc_car_training_course,
)


def _owners_at(x, y):
    return {
        f"smores_{index:02d}": (
            float(x),
            float(y),
        )
        for index in range(1, 9)
    }


def test_forward_progress_is_positive_and_backward_progress_is_negative():
    from drl.MARL.phase1_reward import (
        compute_phase1_raw_reward,
    )

    course = phase1_rc_car_training_course(5102)

    previous = _owners_at(0.50, 0.0)
    forward = _owners_at(0.60, 0.0)
    backward = _owners_at(0.40, 0.0)

    positive = compute_phase1_raw_reward(
        course,
        previous,
        forward,
    )

    negative = compute_phase1_raw_reward(
        course,
        previous,
        backward,
    )

    assert positive.team.progress > 0.0
    assert negative.team.progress < 0.0

    assert all(
        term.progress > 0.0
        for term in positive.owners.values()
    )


def test_progress_is_normalized_by_route_length():
    from drl.MARL.phase1_reward import (
        compute_phase1_raw_reward,
    )

    course = phase1_rc_car_training_course(5102)

    result = compute_phase1_raw_reward(
        course,
        _owners_at(0.50, 0.0),
        _owners_at(0.60, 0.0),
    )

    assert 0.0 < result.team.progress < 1.0


def test_target_entry_is_potential_difference_and_success_requires_8_of_8():
    from drl.MARL.phase1_reward import (
        compute_phase1_raw_reward,
    )

    course = phase1_rc_car_training_course(5101)

    cx, cy = course.target_platform.center_xy_m

    previous = {
        f"smores_{index:02d}": (
            cx,
            cy,
        )
        for index in range(1, 8)
    }

    previous["smores_08"] = (
        cx - 1.0,
        cy,
    )

    next_state = {
        f"smores_{index:02d}": (
            cx,
            cy,
        )
        for index in range(1, 9)
    }

    result = compute_phase1_raw_reward(
        course,
        previous,
        next_state,
    )

    assert result.team.target_entry == pytest.approx(
        1.0 / 8.0
    )

    assert result.owners["smores_08"].target_entry == pytest.approx(
        1.0
    )

    assert result.team.success == pytest.approx(1.0)

    assert all(
        term.success == pytest.approx(1.0)
        for term in result.owners.values()
    )


def test_leaving_target_produces_negative_target_entry():
    from drl.MARL.phase1_reward import (
        compute_phase1_raw_reward,
    )

    course = phase1_rc_car_training_course(5101)

    cx, cy = course.target_platform.center_xy_m

    previous = _owners_at(cx, cy)
    next_state = dict(previous)

    next_state["smores_08"] = (
        cx - 1.0,
        cy,
    )

    result = compute_phase1_raw_reward(
        course,
        previous,
        next_state,
    )

    assert result.team.target_entry == pytest.approx(
        -1.0 / 8.0
    )

    assert result.owners["smores_08"].target_entry == pytest.approx(
        -1.0
    )

    assert result.team.success == pytest.approx(0.0)


def test_outside_road_penalty_is_zero_inside_and_positive_outside():
    from drl.MARL.phase1_reward import (
        compute_phase1_raw_reward,
    )

    course = phase1_rc_car_training_course(5102)

    previous = _owners_at(0.60, 0.0)

    inside = compute_phase1_raw_reward(
        course,
        previous,
        previous,
    )

    outside_positions = dict(previous)

    outside_positions["smores_08"] = (
        0.60,
        course.road_width_m,
    )

    outside = compute_phase1_raw_reward(
        course,
        previous,
        outside_positions,
    )

    assert inside.owners["smores_08"].outside_road == pytest.approx(
        0.0
    )

    assert outside.owners["smores_08"].outside_road > 0.0
    assert outside.team.outside_road > 0.0


def test_runtime_events_remain_owner_local_and_have_team_fractions():
    from drl.MARL.phase1_reward import (
        ActionOutcomeSignals,
        compute_phase1_raw_reward,
    )

    course = phase1_rc_car_training_course(5102)

    positions = _owners_at(0.60, 0.0)

    signals = ActionOutcomeSignals(
        collision_owner_ids=frozenset({
            "smores_03",
        }),
        conflict_owner_ids=frozenset({
            "smores_04",
            "smores_06",
        }),
        invalid_owner_ids=frozenset({
            "smores_08",
        }),
    )

    result = compute_phase1_raw_reward(
        course,
        positions,
        positions,
        signals=signals,
    )

    assert result.owners["smores_03"].collision == pytest.approx(1.0)
    assert result.owners["smores_04"].collision == pytest.approx(0.0)

    assert result.owners["smores_04"].conflict == pytest.approx(1.0)
    assert result.owners["smores_06"].conflict == pytest.approx(1.0)

    assert result.owners["smores_08"].invalid == pytest.approx(1.0)

    assert result.team.collision == pytest.approx(1.0 / 8.0)
    assert result.team.conflict == pytest.approx(2.0 / 8.0)
    assert result.team.invalid == pytest.approx(1.0 / 8.0)


def test_redundant_action_term_counts_only_runtime_attributed_extra_actions():
    from drl.MARL.phase1_reward import (
        ActionOutcomeSignals,
        compute_phase1_raw_reward,
    )

    course = phase1_rc_car_training_course(5102)

    positions = _owners_at(0.60, 0.0)

    baseline = compute_phase1_raw_reward(
        course,
        positions,
        positions,
    )

    assert all(
        term.redundant_action == pytest.approx(0.0)
        for term in baseline.owners.values()
    )

    # The reward engine does NOT infer that "many active actions"
    # are automatically bad. Only runtime-attributed redundant
    # actions enter this term.
    signals = ActionOutcomeSignals(
        redundant_action_count_by_owner={
            "smores_03": 2,
            "smores_04": 1,
        },
    )

    result = compute_phase1_raw_reward(
        course,
        positions,
        positions,
        signals=signals,
    )

    assert result.owners["smores_03"].redundant_action == pytest.approx(
        2.0
    )

    assert result.owners["smores_04"].redundant_action == pytest.approx(
        1.0
    )

    assert result.owners["smores_06"].redundant_action == pytest.approx(
        0.0
    )

    assert result.team.redundant_action == pytest.approx(
        3.0 / 8.0
    )


def test_control_effort_is_explicit_not_inferred_from_action_count():
    from drl.MARL.phase1_reward import (
        ActionOutcomeSignals,
        compute_phase1_raw_reward,
    )

    course = phase1_rc_car_training_course(5102)

    positions = _owners_at(0.60, 0.0)

    signals = ActionOutcomeSignals(
        control_effort_by_owner={
            "smores_03": 0.25,
            "smores_04": 0.50,
        },
    )

    result = compute_phase1_raw_reward(
        course,
        positions,
        positions,
        signals=signals,
    )

    assert result.owners["smores_03"].control_effort == pytest.approx(
        0.25
    )

    assert result.owners["smores_04"].control_effort == pytest.approx(
        0.50
    )

    assert result.team.control_effort == pytest.approx(
        0.75 / 8.0
    )


def test_time_cost_is_raw_constant_and_stagnation_is_explicit():
    from drl.MARL.phase1_reward import (
        ActionOutcomeSignals,
        compute_phase1_raw_reward,
    )

    course = phase1_rc_car_training_course(5102)

    positions = _owners_at(0.60, 0.0)

    result = compute_phase1_raw_reward(
        course,
        positions,
        positions,
        signals=ActionOutcomeSignals(
            stagnating=True,
        ),
    )

    assert result.team.time == pytest.approx(1.0)
    assert result.team.stagnation == pytest.approx(1.0)

    assert all(
        term.time == pytest.approx(1.0)
        for term in result.owners.values()
    )

    assert all(
        term.stagnation == pytest.approx(1.0)
        for term in result.owners.values()
    )


def test_reward_contract_rejects_missing_module_or_unknown_event_owner():
    from drl.MARL.phase1_reward import (
        ActionOutcomeSignals,
        compute_phase1_raw_reward,
    )

    course = phase1_rc_car_training_course(5102)

    positions = _owners_at(0.60, 0.0)

    missing = dict(positions)
    missing.pop("smores_08")

    with pytest.raises(
        ValueError,
        match="exactly 8",
    ):
        compute_phase1_raw_reward(
            course,
            positions,
            missing,
        )

    with pytest.raises(
        ValueError,
        match="unknown owner",
    ):
        compute_phase1_raw_reward(
            course,
            positions,
            positions,
            signals=ActionOutcomeSignals(
                conflict_owner_ids=frozenset({
                    "smores_99",
                }),
            ),
        )


@pytest.mark.parametrize("gamma", [0.5, 0.99, 1.0])
def test_rc_progress_target_entry_and_outside_road_closed_cycles_do_not_pay(gamma):
    from drl.MARL.phase1_reward import compute_phase1_raw_reward
    course = phase1_rc_car_training_course(5102)
    cx,cy = course.target_platform.center_xy_m
    states = [_owners_at(0.5,0), _owners_at(0.6,0),
              _owners_at(0.6,course.road_width_m), _owners_at(cx,cy)]
    frames = [compute_phase1_raw_reward(course,a,b,shaping_gamma=gamma)
              for a,b in zip(states, states[1:]+states[:1])]
    assert any(frame.team.outside_road > 0 for frame in frames)
    for field in ("progress", "target_entry"):
        assert sum(getattr(frame.team, field) for frame in frames) <= 1e-12
    assert frames[-1].team.target_entry < 0


def test_rc_success_absorbs_shaping_without_erasing_sparse_success():
    from drl.MARL.phase1_reward import compute_phase1_raw_reward
    course = phase1_rc_car_training_course(5102)
    cx,cy = course.target_platform.center_xy_m
    frame = compute_phase1_raw_reward(course, _owners_at(0.5,0), _owners_at(cx,cy),
                                     shaping_gamma=0.99, terminated=True)
    assert frame.team.success == 1.0
    assert frame.team.progress <= 0.0
    assert frame.team.target_entry == 0.0


@pytest.mark.parametrize("gamma", [-0.1, 1.1, float("nan"), float("inf")])
def test_shaping_gamma_must_be_a_finite_discount(gamma):
    from drl.MARL.phase1_reward import compute_phase1_raw_reward
    course = phase1_rc_car_training_course(5102)
    state = _owners_at(0.5,0)
    with pytest.raises(ValueError, match="gamma"):
        compute_phase1_raw_reward(course, state, state, shaping_gamma=gamma)


@pytest.mark.parametrize("domain", ["navigation", "stairs", "gap", "manipulation"])
def test_task_evaluators_apply_discount_and_absorb_only_true_terminal(domain):
    from drl.MARL.phase1_reward import ActionOutcomeSignals
    neutral_signals = lambda *_args, **_kwargs: ActionOutcomeSignals()
    if domain == "navigation":
        from drl.MARL.phase1_transition import Phase1TransitionEvaluator
        course = phase1_rc_car_training_course(5102)
        evaluator = Phase1TransitionEvaluator(courses=(course,), max_episode_steps=1,
            signals_provider=neutral_signals, shaping_gamma=0.99)
        before = _owners_at(0.5, 0)
        cx,cy = course.target_platform.center_xy_m
        goal = _owners_at(cx,cy)
        field = "progress"
    elif domain == "stairs":
        from drl.MARL.stairs_transition import StairsTransitionEvaluator
        from smores_ep.isaac.phase1_stairs_curriculum import phase1_stairs_curriculum_course
        course = phase1_stairs_curriculum_course(12000, split="train")
        evaluator = StairsTransitionEvaluator(courses=(course,), max_episode_steps=1,
            signals_provider=neutral_signals, shaping_gamma=0.99)
        before = {owner:(0,0,0.0315) for owner in _owners_at(0,0)}
        goal = {owner:(course.goal_platform_start_x_m+0.1,0,course.goal_height_m+0.0315)
                for owner in before}
        field = "profile_progress"
    elif domain == "gap":
        from drl.MARL.gap_transition import GapTransitionEvaluator
        from smores_ep.isaac.phase1_gap_curriculum import phase1_gap_curriculum_course
        course = phase1_gap_curriculum_course(16000, split="train")
        evaluator = GapTransitionEvaluator(courses=(course,), max_episode_steps=1,
            signals_provider=neutral_signals, shaping_gamma=0.99)
        before = {owner:(course.near_edge_x_m-0.1,0,0.0315) for owner in _owners_at(0,0)}
        goal = {owner:(course.far_edge_x_m+0.1,course.landing_center_y_m,0.0315) for owner in before}
        field = "approach_progress"
    else:
        from drl.MARL.manipulation_transition import ManipulationTransitionEvaluator
        from drl.MARL.tests.test_manipulation_reward import _course, _obs
        course = _course()
        evaluator = ManipulationTransitionEvaluator(course=course,
            signal_builder=neutral_signals, shaping_gamma=0.99)
        before = _obs(course, standoff_m=0.325)
        goal = _obs(course, standoff_m=0.325, depression_m=0.0035)
        field = "approach_progress"
    steady = evaluator(env_index=0, previous_state=before, next_state=before,
        actuation_result=None, physics_result=None, episode_step=1)
    # A time-limit truncation keeps the next potential: only (gamma-1)*Phi.
    steady_value = getattr(steady.raw_reward.team, field)
    assert -0.02 < steady_value < 0
    if domain != "manipulation":
        assert steady.truncated and not steady.terminated
    terminal = evaluator(env_index=0, previous_state=before, next_state=goal,
        actuation_result=None, physics_result=None, episode_step=1)
    assert terminal.terminated
    assert terminal.raw_reward.team.success == 1
    assert getattr(terminal.raw_reward.team, field) == pytest.approx(100 * steady_value)


def test_manipulation_connection_loss_absorbs_potentials_as_failure():
    from drl.MARL.manipulation_transition import ManipulationTransitionEvaluator
    from drl.MARL.tests.test_manipulation_reward import _course, _obs
    from drl.MARL.phase1_reward import ActionOutcomeSignals
    course = _course()
    evaluator = ManipulationTransitionEvaluator(course=course, shaping_gamma=0.99,
        signal_builder=lambda **_kwargs: ActionOutcomeSignals())
    result = evaluator(env_index=0,
        previous_state=_obs(course, standoff_m=0.325),
        next_state=_obs(course, standoff_m=0.325, connection_count=6),
        actuation_result=None, physics_result=None, episode_step=1)
    assert result.terminated
    assert result.failure_reason == "connection_loss"
    assert result.raw_reward.team.approach_progress == pytest.approx(-1)
    assert result.raw_reward.team.success == 0
