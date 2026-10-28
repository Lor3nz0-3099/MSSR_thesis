from __future__ import annotations

from dataclasses import asdict

import pytest

from drl.MARL.phase1_reward import (
    ActionOutcomeSignals,
)
from smores_ep.config.geometry import (
    SmoresGeometry,
)
from smores_ep.isaac.phase1_gap_curriculum import (
    phase1_gap_curriculum_course,
)


OWNERS = tuple(
    f"smores_{index:02d}"
    for index in range(1, 9)
)


def _state(
    *,
    x: float,
    y: float,
    z: float,
):
    return {
        owner: (x, y, z)
        for owner in OWNERS
    }


def _per_owner_state(
    xyz_by_owner,
):
    return {
        owner: tuple(xyz_by_owner[owner])
        for owner in OWNERS
    }


def test_neutral_transition_has_only_time_term():
    from drl.MARL.gap_reward import (
        compute_gap_raw_reward,
    )

    course = phase1_gap_curriculum_course(
        18527,
        split="challenge",
    )

    r = SmoresGeometry().wheel_radius_m

    state = _state(
        x=course.near_edge_x_m - 0.20,
        y=0.0,
        z=r,
    )

    frame = compute_gap_raw_reward(
        course,
        state,
        state,
    )

    assert frame.team.approach_progress == pytest.approx(0.0)
    assert frame.team.crossing_progress == pytest.approx(0.0)
    assert frame.team.landing_progress == pytest.approx(0.0)
    assert frame.team.goal_entry == pytest.approx(0.0)
    assert frame.team.success == pytest.approx(0.0)
    assert frame.team.fall == pytest.approx(0.0)
    assert frame.team.time == pytest.approx(1.0)


def test_forward_motion_on_approach_rewards_only_approach_progress():
    from drl.MARL.gap_reward import (
        compute_gap_raw_reward,
    )

    course = phase1_gap_curriculum_course(
        18527,
        split="challenge",
    )

    r = SmoresGeometry().wheel_radius_m

    previous = _state(
        x=course.near_edge_x_m - 0.20,
        y=0.0,
        z=r,
    )

    next_state = _state(
        x=course.near_edge_x_m - 0.15,
        y=0.0,
        z=r,
    )

    frame = compute_gap_raw_reward(
        course,
        previous,
        next_state,
    )

    assert frame.team.approach_progress == pytest.approx(
        0.05
    )

    assert frame.team.crossing_progress == pytest.approx(0.0)
    assert frame.team.landing_progress == pytest.approx(0.0)
    assert frame.team.goal_entry == pytest.approx(0.0)
    assert frame.team.fall == pytest.approx(0.0)


def test_crossing_progress_is_normalized_over_gap_width():
    from drl.MARL.gap_reward import (
        compute_gap_raw_reward,
    )

    course = phase1_gap_curriculum_course(
        18527,
        split="challenge",
    )

    r = SmoresGeometry().wheel_radius_m

    previous_x = (
        course.near_edge_x_m
        + 0.25 * course.gap_width_m
    )

    next_x = (
        course.near_edge_x_m
        + 0.75 * course.gap_width_m
    )

    previous = _state(
        x=previous_x,
        y=0.0,
        z=r,
    )

    next_state = _state(
        x=next_x,
        y=0.0,
        z=r,
    )

    frame = compute_gap_raw_reward(
        course,
        previous,
        next_state,
    )

    assert frame.team.approach_progress == pytest.approx(0.0)

    assert frame.team.crossing_progress == pytest.approx(
        0.50
    )

    assert frame.team.landing_progress == pytest.approx(0.0)
    assert frame.team.fall == pytest.approx(0.0)


def test_falling_forward_does_not_earn_crossing_progress():
    from drl.MARL.gap_reward import (
        compute_gap_raw_reward,
    )

    course = phase1_gap_curriculum_course(
        18527,
        split="challenge",
    )

    r = SmoresGeometry().wheel_radius_m

    previous = _state(
        x=(
            course.near_edge_x_m
            + 0.20 * course.gap_width_m
        ),
        y=0.0,
        z=r,
    )

    next_state = _state(
        x=(
            course.near_edge_x_m
            + 0.80 * course.gap_width_m
        ),
        y=0.0,
        z=r - 0.05,
    )

    frame = compute_gap_raw_reward(
        course,
        previous,
        next_state,
    )

    # Falling removes the endpoint potential, debiting prior progress.
    # Returning to support can therefore never farm a fresh progress bonus.
    assert frame.team.crossing_progress == pytest.approx(-0.20)

    assert frame.team.fall == pytest.approx(
        float(len(OWNERS))
    )


def test_entering_landing_generates_goal_entry():
    from drl.MARL.gap_reward import (
        compute_gap_raw_reward,
    )

    course = phase1_gap_curriculum_course(
        18527,
        split="challenge",
    )

    r = SmoresGeometry().wheel_radius_m

    previous = _state(
        x=course.far_edge_x_m - 0.01,
        y=course.landing_center_y_m,
        z=r,
    )

    next_state = _state(
        x=course.far_edge_x_m + 0.03,
        y=course.landing_center_y_m,
        z=r,
    )

    frame = compute_gap_raw_reward(
        course,
        previous,
        next_state,
    )

    assert frame.team.goal_entry == pytest.approx(
        1.0
    )

    assert frame.team.crossing_progress > 0.0
    assert frame.team.fall == pytest.approx(0.0)


def test_forward_motion_on_landing_generates_landing_progress():
    from drl.MARL.gap_reward import (
        compute_gap_raw_reward,
    )

    course = phase1_gap_curriculum_course(
        18527,
        split="challenge",
    )

    r = SmoresGeometry().wheel_radius_m

    previous = _state(
        x=course.far_edge_x_m + 0.10,
        y=course.landing_center_y_m,
        z=r,
    )

    next_state = _state(
        x=course.far_edge_x_m + 0.15,
        y=course.landing_center_y_m,
        z=r,
    )

    frame = compute_gap_raw_reward(
        course,
        previous,
        next_state,
    )

    assert frame.team.approach_progress == pytest.approx(0.0)
    assert frame.team.crossing_progress == pytest.approx(0.0)

    assert frame.team.landing_progress == pytest.approx(
        0.05
    )

    assert frame.team.fall == pytest.approx(0.0)


def test_success_requires_every_module_supported_on_landing():
    from drl.MARL.gap_reward import (
        compute_gap_raw_reward,
    )

    course = phase1_gap_curriculum_course(
        18527,
        split="challenge",
    )

    r = SmoresGeometry().wheel_radius_m

    previous = _state(
        x=course.far_edge_x_m - 0.02,
        y=course.landing_center_y_m,
        z=r,
    )

    next_state = _state(
        x=course.far_edge_x_m + 0.20,
        y=course.landing_center_y_m,
        z=r,
    )

    frame = compute_gap_raw_reward(
        course,
        previous,
        next_state,
    )

    assert frame.team.success == pytest.approx(
        1.0
    )

    broken = dict(next_state)

    broken["smores_08"] = (
        course.near_edge_x_m - 0.10,
        0.0,
        r,
    )

    frame = compute_gap_raw_reward(
        course,
        previous,
        broken,
    )

    assert frame.team.success == pytest.approx(
        0.0
    )


def test_landing_offset_affects_physical_support_not_reward_target():
    from drl.MARL.gap_reward import (
        compute_gap_raw_reward,
    )

    course = phase1_gap_curriculum_course(
        18527,
        split="challenge",
    )

    r = SmoresGeometry().wheel_radius_m

    x = course.far_edge_x_m + 0.10

    previous = _state(
        x=x - 0.02,
        y=course.landing_center_y_m,
        z=r,
    )

    next_state = _state(
        x=x,
        y=course.landing_center_y_m,
        z=r,
    )

    frame = compute_gap_raw_reward(
        course,
        previous,
        next_state,
    )

    owner_fields = asdict(
        frame.owners["smores_01"]
    )

    team_fields = asdict(
        frame.team
    )

    assert all(
        "lateral" not in key.lower()
        for key in owner_fields
    )

    assert all(
        "lateral" not in key.lower()
        for key in team_fields
    )


def test_runtime_action_outcome_signals_are_preserved():
    from drl.MARL.gap_reward import (
        compute_gap_raw_reward,
    )

    course = phase1_gap_curriculum_course(
        18527,
        split="challenge",
    )

    r = SmoresGeometry().wheel_radius_m

    state = _state(
        x=course.near_edge_x_m - 0.10,
        y=0.0,
        z=r,
    )

    signals = ActionOutcomeSignals(
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
    )

    frame = compute_gap_raw_reward(
        course,
        state,
        state,
        signals=signals,
    )

    assert frame.owners["smores_01"].collision == pytest.approx(1.0)
    assert frame.owners["smores_02"].conflict == pytest.approx(1.0)
    assert frame.owners["smores_03"].invalid == pytest.approx(1.0)

    assert (
        frame
        .owners["smores_04"]
        .redundant_action
        == pytest.approx(2.0)
    )

    assert (
        frame
        .owners["smores_05"]
        .control_effort
        == pytest.approx(0.75)
    )

    assert frame.team.collision == pytest.approx(1.0)
    assert frame.team.conflict == pytest.approx(1.0)
    assert frame.team.invalid == pytest.approx(1.0)
    assert frame.team.redundant_action == pytest.approx(2.0)
    assert frame.team.control_effort == pytest.approx(0.75)
    assert frame.team.stagnation == pytest.approx(1.0)


def test_backward_crossing_progress_is_signed():
    from drl.MARL.gap_reward import (
        compute_gap_raw_reward,
    )

    course = phase1_gap_curriculum_course(
        18527,
        split="challenge",
    )

    r = SmoresGeometry().wheel_radius_m

    previous = _state(
        x=(
            course.near_edge_x_m
            + 0.75 * course.gap_width_m
        ),
        y=0.0,
        z=r,
    )

    next_state = _state(
        x=(
            course.near_edge_x_m
            + 0.25 * course.gap_width_m
        ),
        y=0.0,
        z=r,
    )

    frame = compute_gap_raw_reward(
        course,
        previous,
        next_state,
    )

    assert frame.team.crossing_progress == pytest.approx(
        -0.50
    )


def test_owner_sets_must_match():
    from drl.MARL.gap_reward import (
        compute_gap_raw_reward,
    )

    course = phase1_gap_curriculum_course(
        18527,
        split="challenge",
    )

    r = SmoresGeometry().wheel_radius_m

    previous = _state(
        x=0.0,
        y=0.0,
        z=r,
    )

    next_state = dict(
        previous
    )

    next_state.pop(
        "smores_08"
    )

    with pytest.raises(
        ValueError,
        match="owner",
    ):
        compute_gap_raw_reward(
            course,
            previous,
            next_state,
        )


def test_wheel_radius_and_tolerance_fail_closed():
    from drl.MARL.gap_reward import (
        compute_gap_raw_reward,
    )

    course = phase1_gap_curriculum_course(
        18527,
        split="challenge",
    )

    r = SmoresGeometry().wheel_radius_m

    state = _state(
        x=0.0,
        y=0.0,
        z=r,
    )

    with pytest.raises(ValueError):
        compute_gap_raw_reward(
            course,
            state,
            state,
            wheel_radius_m=0.0,
        )

    with pytest.raises(ValueError):
        compute_gap_raw_reward(
            course,
            state,
            state,
            support_tolerance_m=-0.01,
        )


def test_gap_one_owner_cannot_succeed():
    from drl.MARL.gap_reward import compute_gap_raw_reward
    course = phase1_gap_curriculum_course(16000, split="train")
    state = {"smores_01": (course.far_edge_x_m + 0.1, course.landing_center_y_m, 0.0315)}
    with pytest.raises(ValueError, match="8"):
        compute_gap_raw_reward(course, state, state)


@pytest.mark.parametrize("signals", [
    ActionOutcomeSignals(collision_owner_ids=frozenset({"ghost"})),
    ActionOutcomeSignals(conflict_owner_ids=frozenset({"ghost"})),
    ActionOutcomeSignals(invalid_owner_ids=frozenset({"ghost"})),
    ActionOutcomeSignals(control_effort_by_owner={"smores_01": float("nan")}),
    ActionOutcomeSignals(control_effort_by_owner={"smores_01": -1.0}),
    ActionOutcomeSignals(redundant_action_count_by_owner={"smores_01": -1}),
])
def test_gap_invalid_runtime_signals_fail_closed(signals):
    from drl.MARL.gap_reward import compute_gap_raw_reward
    course = phase1_gap_curriculum_course(16000, split="train")
    state = _state(x=0, y=0, z=0.0315)
    with pytest.raises(ValueError):
        compute_gap_raw_reward(course, state, state, signals=signals)


def test_gap_goal_exit_debits_entry_even_on_support_loss():
    from drl.MARL.gap_reward import compute_gap_raw_reward
    course = phase1_gap_curriculum_course(16000, split="train")
    outside = _state(x=course.far_edge_x_m-0.01, y=course.landing_center_y_m, z=0.0315)
    inside = _state(x=course.far_edge_x_m+0.10, y=course.landing_center_y_m, z=0.0315)
    fallen = _state(x=course.far_edge_x_m+0.10, y=course.landing_center_y_m, z=-0.05)
    entry = compute_gap_raw_reward(course, outside, inside)
    for leave in (outside, fallen):
        exit_frame = compute_gap_raw_reward(course, inside, leave)
        assert exit_frame.team.goal_entry == pytest.approx(-1.0)
        assert entry.team.goal_entry + exit_frame.team.goal_entry <= 1e-12


@pytest.mark.parametrize("gamma", [0.5, 0.99, 1.0])
def test_gap_support_loss_and_recovery_cannot_farm_crossing(gamma):
    from drl.MARL.gap_reward import compute_gap_raw_reward
    course = phase1_gap_curriculum_course(16000, split="train")
    states = [
        _state(x=course.near_edge_x_m+0.2*course.gap_width_m, y=0, z=0.0315),
        _state(x=course.near_edge_x_m+0.8*course.gap_width_m, y=0, z=0.0315),
        _state(x=course.near_edge_x_m+0.2*course.gap_width_m, y=0, z=-0.05),
    ]
    for field in ("approach_progress", "crossing_progress", "landing_progress", "goal_entry"):
        total = sum(getattr(compute_gap_raw_reward(course, a, b, shaping_gamma=gamma).team, field)
                    for a,b in zip(states, states[1:] + states[:1]))
        assert total <= 1e-12


@pytest.mark.parametrize("bad", [(float("nan"), 0, 0), (0, float("inf"), 0), (0, 0), None])
def test_gap_malformed_observations_are_rejected(bad):
    from drl.MARL.gap_reward import compute_gap_raw_reward
    course = phase1_gap_curriculum_course(16000, split="train")
    state = _state(x=0, y=0, z=0.0315)
    broken = dict(state, smores_01=bad)
    with pytest.raises(ValueError):
        compute_gap_raw_reward(course, state, broken)
