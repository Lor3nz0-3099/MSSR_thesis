from __future__ import annotations

import pytest

from drl.MARL.phase1_reward import (
    ActionOutcomeSignals,
)
from smores_ep.isaac.phase1_stairs_curriculum import (
    phase1_stairs_curriculum_course,
)


OWNERS = tuple(
    f"smores_{index:02d}"
    for index in range(1, 9)
)


def _states(
    *,
    x: float,
    y: float,
    z: float,
):
    return {
        owner: (x, y, z)
        for owner in OWNERS
    }


def test_ascent_forward_and_upward_motion_are_positive():
    from drl.MARL.stairs_reward import (
        compute_stairs_raw_reward,
    )

    course = phase1_stairs_curriculum_course(
        12000,
        split="train",
    )

    previous = _states(
        x=course.first_transition_x_m - 0.05,
        y=0.0,
        z=0.0315,
    )

    next_state = _states(
        x=course.first_transition_x_m + 0.03,
        y=0.0,
        z=(
            course.step_top_heights_m[0]
            + 0.0315
        ),
    )

    reward = compute_stairs_raw_reward(
        course,
        previous,
        next_state,
    )

    assert reward.team.profile_progress > 0.0
    assert reward.team.vertical_progress > 0.0
    assert reward.team.level_progress > 0.0


def test_descent_forward_and_downward_motion_are_positive():
    from drl.MARL.stairs_reward import (
        compute_stairs_raw_reward,
    )

    course = phase1_stairs_curriculum_course(
        12650,
        split="train",
    )

    assert course.direction == "descent"

    previous = _states(
        x=course.first_transition_x_m - 0.04,
        y=0.0,
        z=(
            course.start_height_m
            + 0.0315
        ),
    )

    next_state = _states(
        x=course.first_transition_x_m + 0.04,
        y=0.0,
        z=(
            course.step_top_heights_m[0]
            + 0.0315
        ),
    )

    reward = compute_stairs_raw_reward(
        course,
        previous,
        next_state,
    )

    assert reward.team.profile_progress > 0.0
    assert reward.team.vertical_progress > 0.0
    assert reward.team.level_progress > 0.0


def test_wrong_vertical_direction_is_negative():
    from drl.MARL.stairs_reward import (
        compute_stairs_raw_reward,
    )

    ascent = phase1_stairs_curriculum_course(
        12000,
        split="train",
    )

    reward = compute_stairs_raw_reward(
        ascent,
        _states(
            x=0.0,
            y=0.0,
            z=0.060,
        ),
        _states(
            x=0.0,
            y=0.0,
            z=0.040,
        ),
    )

    assert reward.team.vertical_progress < 0.0


def test_wheel_spin_without_physical_motion_has_zero_progress():
    from drl.MARL.stairs_reward import (
        compute_stairs_raw_reward,
    )

    course = phase1_stairs_curriculum_course(
        12653,
        split="train",
    )

    state = _states(
        x=course.first_transition_x_m - 0.02,
        y=0.0,
        z=0.0315,
    )

    reward = compute_stairs_raw_reward(
        course,
        state,
        state,
        signals=ActionOutcomeSignals(
            stagnating=True,
        ),
    )

    assert reward.team.profile_progress == pytest.approx(0.0)
    assert reward.team.vertical_progress == pytest.approx(0.0)
    assert reward.team.level_progress == pytest.approx(0.0)
    assert reward.team.stagnation == pytest.approx(1.0)


def test_vertical_lift_before_riser_does_not_complete_level():
    from drl.MARL.stairs_reward import (
        compute_stairs_raw_reward,
    )

    course = phase1_stairs_curriculum_course(
        12000,
        split="train",
    )

    x = course.first_transition_x_m - 0.08

    previous = _states(
        x=x,
        y=0.0,
        z=0.0315,
    )

    next_state = _states(
        x=x,
        y=0.0,
        z=(
            course.step_top_heights_m[0]
            + 0.0315
        ),
    )

    reward = compute_stairs_raw_reward(
        course,
        previous,
        next_state,
    )

    assert reward.team.vertical_progress > 0.0
    assert reward.team.level_progress == pytest.approx(0.0)


def test_goal_entry_requires_position_and_goal_deck_height():
    from drl.MARL.stairs_reward import (
        compute_stairs_raw_reward,
    )

    course = phase1_stairs_curriculum_course(
        12653,
        split="train",
    )

    goal_x = (
        course.goal_platform_start_x_m
        + 0.15
    )

    previous = _states(
        x=course.goal_platform_start_x_m - 0.03,
        y=0.0,
        z=(
            course.goal_height_m
            + 0.0315
        ),
    )

    next_state = _states(
        x=goal_x,
        y=0.0,
        z=(
            course.goal_height_m
            + 0.0315
        ),
    )

    reward = compute_stairs_raw_reward(
        course,
        previous,
        next_state,
    )

    assert reward.team.goal_entry == pytest.approx(1.0)
    assert reward.team.success == pytest.approx(1.0)

    too_low = _states(
        x=goal_x,
        y=0.0,
        z=course.goal_height_m - 0.05,
    )

    failed = compute_stairs_raw_reward(
        course,
        previous,
        too_low,
    )

    assert failed.team.success == pytest.approx(0.0)


def test_outside_width_is_continuous_positive_penalty_term():
    from drl.MARL.stairs_reward import (
        compute_stairs_raw_reward,
    )

    course = phase1_stairs_curriculum_course(
        12000,
        split="train",
    )

    previous = _states(
        x=0.0,
        y=0.0,
        z=0.0315,
    )

    next_state = _states(
        x=0.0,
        y=0.70,
        z=0.0315,
    )

    reward = compute_stairs_raw_reward(
        course,
        previous,
        next_state,
    )

    assert reward.team.outside_width > 0.0


def test_below_support_detects_real_drop_but_allows_transition_clearance():
    from drl.MARL.stairs_reward import (
        compute_stairs_raw_reward,
    )

    course = phase1_stairs_curriculum_course(
        12000,
        split="train",
    )

    x = course.first_transition_x_m + 0.15

    normal = _states(
        x=x,
        y=0.0,
        z=(
            course.step_top_heights_m[0]
            + 0.0315
        ),
    )

    dropped = _states(
        x=x,
        y=0.0,
        z=(
            course.step_top_heights_m[0]
            - 0.050
        ),
    )

    ok = compute_stairs_raw_reward(
        course,
        normal,
        normal,
    )

    bad = compute_stairs_raw_reward(
        course,
        normal,
        dropped,
    )

    assert ok.team.below_support == pytest.approx(0.0)
    assert bad.team.below_support > 0.0


def test_common_action_outcome_signals_are_preserved():
    from drl.MARL.stairs_reward import (
        compute_stairs_raw_reward,
    )

    course = phase1_stairs_curriculum_course(
        12000,
        split="train",
    )

    state = _states(
        x=0.0,
        y=0.0,
        z=0.0315,
    )

    reward = compute_stairs_raw_reward(
        course,
        state,
        state,
        signals=ActionOutcomeSignals(
            collision_owner_ids=frozenset(
                {"smores_01"}
            ),
            conflict_owner_ids=frozenset(
                {"smores_02"}
            ),
            invalid_owner_ids=frozenset(
                {"smores_03"}
            ),
            redundant_action_count_by_owner={
                "smores_04": 2,
            },
            control_effort_by_owner={
                "smores_05": 0.75,
            },
            stagnating=True,
        ),
    )

    assert reward.owners["smores_01"].collision == 1.0
    assert reward.owners["smores_02"].conflict == 1.0
    assert reward.owners["smores_03"].invalid == 1.0
    assert reward.owners["smores_04"].redundant_action == 2.0
    assert reward.owners["smores_05"].control_effort == pytest.approx(0.75)
    assert reward.team.stagnation == 1.0


def test_eight_module_contract_is_fail_closed():
    from drl.MARL.stairs_reward import (
        compute_stairs_raw_reward,
    )

    course = phase1_stairs_curriculum_course(
        12000,
        split="train",
    )

    bad = {
        owner: (0.0, 0.0, 0.0315)
        for owner in OWNERS[:-1]
    }

    with pytest.raises(
        ValueError,
        match="exactly 8",
    ):
        compute_stairs_raw_reward(
            course,
            bad,
            bad,
        )


def test_descent_fall_cannot_earn_vertical_or_level_progress():
    from drl.MARL.stairs_reward import compute_stairs_raw_reward
    course = phase1_stairs_curriculum_course(12650, split="train")
    x = course.first_transition_x_m - 0.10
    before = _states(x=x, y=0, z=course.start_height_m+0.0315)
    fallen = _states(x=x+0.03, y=0, z=-0.10)
    frame = compute_stairs_raw_reward(course, before, fallen)
    assert frame.team.below_support > 0
    assert frame.team.profile_progress <= 0
    assert frame.team.vertical_progress <= 0
    assert frame.team.level_progress <= 0


@pytest.mark.parametrize("gamma", [0.5, 0.99, 1.0])
@pytest.mark.parametrize("seed", [12000,12650])
def test_stairs_support_loss_recovery_and_goal_loops_do_not_pay(seed, gamma):
    from drl.MARL.stairs_reward import compute_stairs_raw_reward
    course = phase1_stairs_curriculum_course(seed, split="train")
    states = [
        _states(x=course.first_transition_x_m-0.06, y=0, z=course.start_height_m+0.0315),
        _states(x=course.first_transition_x_m+0.06, y=0, z=course.step_top_heights_m[0]+0.0315),
        _states(x=course.goal_platform_start_x_m+0.1, y=0, z=course.goal_height_m+0.0315),
        _states(x=course.first_transition_x_m-0.06, y=0, z=-0.10),
    ]
    for field in ("profile_progress", "vertical_progress", "level_progress", "goal_entry"):
        total = sum(getattr(compute_stairs_raw_reward(course, a, b, shaping_gamma=gamma).team, field)
                    for a,b in zip(states, states[1:] + states[:1]))
        assert total <= 1e-12


def test_stairs_goal_entry_exit_is_signed():
    from drl.MARL.stairs_reward import compute_stairs_raw_reward
    course = phase1_stairs_curriculum_course(12000, split="train")
    a = _states(x=course.goal_platform_start_x_m-0.03, y=0, z=course.goal_height_m+0.0315)
    b = _states(x=course.goal_platform_start_x_m+0.03, y=0, z=course.goal_height_m+0.0315)
    assert compute_stairs_raw_reward(course, a, b).team.goal_entry == pytest.approx(1)
    assert compute_stairs_raw_reward(course, b, a).team.goal_entry == pytest.approx(-1)
