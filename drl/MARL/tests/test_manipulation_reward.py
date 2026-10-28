from __future__ import annotations

import dataclasses
import math

import pytest

from drl.MARL.phase1_reward import (
    ActionOutcomeSignals,
)


def _course(
    seed=20650,
    split="train",
):
    from smores_ep.isaac.phase1_manipulation_curriculum import (
        phase1_manipulation_curriculum_course,
    )

    return phase1_manipulation_curriculum_course(
        seed,
        split=split,
    )


_OFFSETS = (
    (-0.03, -0.03),
    (-0.03,  0.00),
    (-0.03, +0.03),
    ( 0.00, -0.03),
    ( 0.00, +0.03),
    (+0.03, -0.03),
    (+0.03,  0.00),
    (+0.03, +0.03),
)


def _modules(
    center_xy,
):
    cx, cy = center_xy

    return {
        f"smores_{index:02d}":
            (
                cx + dx,
                cy + dy,
                0.05,
            )
        for index, (dx, dy)
        in enumerate(
            _OFFSETS,
            start=1,
        )
    }


def _base_at_standoff(
    course,
    standoff_m,
):
    bx, by, _ = (
        course.button_center_xyz_m
    )

    nx, ny = (
        course.press_direction_world_xy
    )

    return (
        bx - nx * standoff_m,
        by - ny * standoff_m,
    )


def _task_axes(
    course,
):
    nx, ny = (
        course.press_direction_world_xy
    )

    return (
        (nx, ny, 0.0),
        (-ny, nx, 0.0),
    )


def _rest_front(
    course,
):
    from smores_ep.isaac.obstacle_course import (
        BUTTON_PLUNGER_DEPTH_M,
    )

    button = (
        course.button_center_xyz_m
    )

    normal, _ = (
        _task_axes(course)
    )

    return tuple(
        button[index]
        - 0.5
        * BUTTON_PLUNGER_DEPTH_M
        * normal[index]
        for index in range(3)
    )


def _point_from_face(
    course,
    *,
    normal_offset_m=0.0,
    tangent_offset_m=0.0,
    z_offset_m=0.0,
):
    face = _rest_front(
        course
    )

    normal, tangent = (
        _task_axes(course)
    )

    return (
        face[0]
        + normal_offset_m * normal[0]
        + tangent_offset_m * tangent[0],

        face[1]
        + normal_offset_m * normal[1]
        + tangent_offset_m * tangent[1],

        face[2]
        + z_offset_m,
    )


def _normal_with_error(
    course,
    error_deg,
):
    normal, tangent = (
        _task_axes(course)
    )

    angle = math.radians(
        error_deg
    )

    return (
        math.cos(angle) * normal[0]
        + math.sin(angle) * tangent[0],

        math.cos(angle) * normal[1]
        + math.sin(angle) * tangent[1],

        0.0,
    )


def _obs(
    course,
    *,
    standoff_m,
    ee_center=None,
    normal_error_deg=0.0,
    depression_m=0.0,
    connection_count=7,
):
    from drl.MARL.manipulation_reward import (
        ManipulationObservation,
    )

    if ee_center is None:
        ee_center = _point_from_face(
            course,
            normal_offset_m=-0.03,
        )

    return ManipulationObservation(
        module_xyz_m=_modules(
            _base_at_standoff(
                course,
                standoff_m,
            )
        ),
        end_effector_bottom_center_xyz_m=(
            ee_center
        ),
        end_effector_bottom_normal_world=(
            _normal_with_error(
                course,
                normal_error_deg,
            )
        ),
        button_depression_m=(
            depression_m
        ),
        connection_count=(
            connection_count
        ),
    )


def test_approach_progress_starts_immediately_after_assembly():
    from drl.MARL.manipulation_reward import (
        compute_manipulation_raw_reward,
    )

    course = _course()

    previous = _obs(
        course,
        standoff_m=1.20,
    )

    next_state = _obs(
        course,
        standoff_m=1.00,
    )

    reward = (
        compute_manipulation_raw_reward(
            course,
            previous,
            next_state,
        )
    )

    assert (
        reward.team.approach_progress
        > 0.0
    )

    # Still far from the manipulation region:
    # do not double-shape with EE terms yet.
    assert (
        reward.team.ee_surface_progress
        == pytest.approx(0.0)
    )

    assert (
        reward.team.orientation_progress
        == pytest.approx(0.0)
    )


def test_moving_away_from_button_has_negative_approach_progress():
    from drl.MARL.manipulation_reward import (
        compute_manipulation_raw_reward,
    )

    course = _course()

    reward = (
        compute_manipulation_raw_reward(
            course,
            _obs(
                course,
                standoff_m=0.90,
            ),
            _obs(
                course,
                standoff_m=1.00,
            ),
        )
    )

    assert (
        reward.team.approach_progress
        < 0.0
    )


def test_approach_term_turns_off_inside_manipulation_region():
    from drl.MARL.manipulation_reward import (
        compute_manipulation_raw_reward,
    )

    course = _course()

    reward = (
        compute_manipulation_raw_reward(
            course,
            _obs(
                course,
                standoff_m=0.34,
            ),
            _obs(
                course,
                standoff_m=0.33,
            ),
        )
    )

    assert (
        reward.team.approach_progress
        == pytest.approx(0.0)
    )


def test_entering_manipulation_region_activates_surface_progress():
    from drl.MARL.manipulation_reward import (
        compute_manipulation_raw_reward,
    )

    course = _course()

    previous = _obs(
        course,
        standoff_m=0.40,
        ee_center=_point_from_face(
            course,
            normal_offset_m=-0.050,
        ),
    )

    next_state = _obs(
        course,
        standoff_m=0.36,
        ee_center=_point_from_face(
            course,
            normal_offset_m=-0.025,
        ),
    )

    reward = (
        compute_manipulation_raw_reward(
            course,
            previous,
            next_state,
        )
    )

    assert (
        reward.team.approach_progress
        > 0.0
    )

    assert (
        reward.team.ee_surface_progress
        > 0.0
    )


def test_finite_button_surface_does_not_force_center_hit():
    from drl.MARL.manipulation_reward import (
        compute_manipulation_raw_reward,
    )

    course = _course()

    previous = _obs(
        course,
        standoff_m=0.325,
        ee_center=_point_from_face(
            course,
            normal_offset_m=-0.020,
            tangent_offset_m=+0.050,
            z_offset_m=+0.050,
        ),
    )

    next_state = _obs(
        course,
        standoff_m=0.325,
        ee_center=_point_from_face(
            course,
            normal_offset_m=-0.020,
            tangent_offset_m=-0.050,
            z_offset_m=-0.050,
        ),
    )

    reward = (
        compute_manipulation_raw_reward(
            course,
            previous,
            next_state,
        )
    )

    # Both points overlap the finite 80x80 mm button
    # after the conservative 20 mm EE contact expansion.
    # Only normal depth matters, and it did not change.
    assert (
        reward.team.ee_surface_progress
        == pytest.approx(0.0)
    )


def test_surface_progress_rewards_reduction_of_outside_tangent_error():
    from drl.MARL.manipulation_reward import (
        compute_manipulation_raw_reward,
    )

    course = _course()

    previous = _obs(
        course,
        standoff_m=0.325,
        ee_center=_point_from_face(
            course,
            normal_offset_m=-0.020,
            tangent_offset_m=+0.090,
        ),
    )

    next_state = _obs(
        course,
        standoff_m=0.325,
        ee_center=_point_from_face(
            course,
            normal_offset_m=-0.020,
            tangent_offset_m=+0.070,
        ),
    )

    reward = (
        compute_manipulation_raw_reward(
            course,
            previous,
            next_state,
        )
    )

    assert (
        reward.team.ee_surface_progress
        > 0.0
    )


def test_orientation_progress_exists_only_outside_30_degree_cone():
    from drl.MARL.manipulation_reward import (
        compute_manipulation_raw_reward,
    )

    course = _course()

    outside = (
        compute_manipulation_raw_reward(
            course,
            _obs(
                course,
                standoff_m=0.325,
                normal_error_deg=60.0,
            ),
            _obs(
                course,
                standoff_m=0.325,
                normal_error_deg=40.0,
            ),
        )
    )

    assert (
        outside.team.orientation_progress
        > 0.0
    )

    inside = (
        compute_manipulation_raw_reward(
            course,
            _obs(
                course,
                standoff_m=0.325,
                normal_error_deg=20.0,
            ),
            _obs(
                course,
                standoff_m=0.325,
                normal_error_deg=10.0,
            ),
        )
    )

    assert (
        inside.team.orientation_progress
        == pytest.approx(0.0)
    )


def test_depression_progress_is_physical_and_normalized_to_success():
    from drl.MARL.manipulation_reward import (
        BUTTON_SUCCESS_DEPRESSION_M,
        compute_manipulation_raw_reward,
    )

    course = _course()

    half = (
        compute_manipulation_raw_reward(
            course,
            _obs(
                course,
                standoff_m=0.325,
                depression_m=0.0,
            ),
            _obs(
                course,
                standoff_m=0.325,
                depression_m=(
                    0.5
                    * BUTTON_SUCCESS_DEPRESSION_M
                ),
            ),
        )
    )

    assert (
        half.team.depression_progress
        == pytest.approx(0.5)
    )

    assert (
        half.team.success
        == pytest.approx(0.0)
    )

    success = (
        compute_manipulation_raw_reward(
            course,
            _obs(
                course,
                standoff_m=0.325,
                depression_m=(
                    0.5
                    * BUTTON_SUCCESS_DEPRESSION_M
                ),
            ),
            _obs(
                course,
                standoff_m=0.325,
                depression_m=(
                    BUTTON_SUCCESS_DEPRESSION_M
                ),
            ),
        )
    )

    assert (
        success.team.success
        == pytest.approx(1.0)
    )


def test_releasing_button_gives_negative_depression_progress():
    from drl.MARL.manipulation_reward import (
        BUTTON_SUCCESS_DEPRESSION_M,
        compute_manipulation_raw_reward,
    )

    course = _course()

    reward = (
        compute_manipulation_raw_reward(
            course,
            _obs(
                course,
                standoff_m=0.325,
                depression_m=(
                    BUTTON_SUCCESS_DEPRESSION_M
                ),
            ),
            _obs(
                course,
                standoff_m=0.325,
                depression_m=(
                    0.5
                    * BUTTON_SUCCESS_DEPRESSION_M
                ),
            ),
        )
    )

    assert (
        reward.team.depression_progress
        < 0.0
    )


def test_broken_topology_blocks_success():
    from drl.MARL.manipulation_reward import (
        BUTTON_SUCCESS_DEPRESSION_M,
        compute_manipulation_raw_reward,
    )

    course = _course()

    reward = (
        compute_manipulation_raw_reward(
            course,
            _obs(
                course,
                standoff_m=0.325,
                depression_m=0.0,
            ),
            _obs(
                course,
                standoff_m=0.325,
                depression_m=(
                    BUTTON_SUCCESS_DEPRESSION_M
                ),
                connection_count=6,
            ),
        )
    )

    assert (
        reward.team.success
        == pytest.approx(0.0)
    )

    assert (
        reward.team.connection_loss
        == pytest.approx(
            1.0 / 7.0
        )
    )


def test_action_outcome_signals_remain_owner_attributed():
    from drl.MARL.manipulation_reward import (
        compute_manipulation_raw_reward,
    )

    course = _course()

    signals = ActionOutcomeSignals(
        collision_owner_ids=frozenset(
            {
                "smores_01",
            }
        ),
        conflict_owner_ids=frozenset(
            {
                "smores_02",
            }
        ),
        invalid_owner_ids=frozenset(
            {
                "smores_03",
            }
        ),
        redundant_action_count_by_owner={
            "smores_04": 2,
        },
        control_effort_by_owner={
            "smores_05": 4.0,
        },
        stagnating=True,
    )

    reward = (
        compute_manipulation_raw_reward(
            course,
            _obs(
                course,
                standoff_m=0.325,
            ),
            _obs(
                course,
                standoff_m=0.325,
            ),
            signals=signals,
        )
    )

    assert (
        reward.owners[
            "smores_01"
        ].collision
        == 1.0
    )

    assert (
        reward.owners[
            "smores_04"
        ].redundant_action
        == 2.0
    )

    assert (
        reward.team.collision
        == pytest.approx(
            1.0 / 8.0
        )
    )

    assert (
        reward.team.conflict
        == pytest.approx(
            1.0 / 8.0
        )
    )

    assert (
        reward.team.invalid
        == pytest.approx(
            1.0 / 8.0
        )
    )

    assert (
        reward.team.redundant_action
        == pytest.approx(
            2.0 / 8.0
        )
    )

    assert (
        reward.team.control_effort
        == pytest.approx(
            4.0 / 8.0
        )
    )

    assert (
        reward.team.stagnation
        == 1.0
    )


def test_unknown_signal_owner_fails_closed():
    from drl.MARL.manipulation_reward import (
        compute_manipulation_raw_reward,
    )

    course = _course()

    with pytest.raises(
        ValueError,
        match="unknown owner",
    ):
        compute_manipulation_raw_reward(
            course,
            _obs(
                course,
                standoff_m=0.325,
            ),
            _obs(
                course,
                standoff_m=0.325,
            ),
            signals=ActionOutcomeSignals(
                collision_owner_ids=frozenset(
                    {
                        "ghost",
                    }
                )
            ),
        )


def test_invalid_physical_observation_fails_closed():
    from drl.MARL.manipulation_reward import (
        ManipulationObservation,
        compute_manipulation_raw_reward,
    )

    course = _course()

    bad = ManipulationObservation(
        module_xyz_m=_modules(
            _base_at_standoff(
                course,
                0.325,
            )
        ),
        end_effector_bottom_center_xyz_m=(
            _point_from_face(
                course
            )
        ),
        end_effector_bottom_normal_world=(
            0.0,
            0.0,
            0.0,
        ),
        button_depression_m=0.0,
        connection_count=7,
    )

    with pytest.raises(
        ValueError
    ):
        compute_manipulation_raw_reward(
            course,
            bad,
            bad,
        )


def test_reward_contract_contains_no_weights():
    from drl.MARL.manipulation_reward import (
        ManipulationOwnerRewardTerms,
        ManipulationTeamRewardTerms,
    )

    names = {
        field.name.lower()
        for cls in (
            ManipulationOwnerRewardTerms,
            ManipulationTeamRewardTerms,
        )
        for field
        in dataclasses.fields(cls)
    }

    assert not any(
        "weight" in name
        for name in names
    )


def test_arbitrary_yaw_fixture_uses_same_reward_geometry():
    from drl.MARL.manipulation_reward import (
        compute_manipulation_raw_reward,
    )

    course = _course(
        seed=22521,
        split="challenge",
    )

    # Challenge 22521 is intentionally oblique.
    nx, ny = (
        course.press_direction_world_xy
    )

    assert (
        abs(nx) > 0.1
        and abs(ny) > 0.1
    )

    reward = (
        compute_manipulation_raw_reward(
            course,
            _obs(
                course,
                standoff_m=0.325,
                ee_center=_point_from_face(
                    course,
                    normal_offset_m=-0.04,
                ),
            ),
            _obs(
                course,
                standoff_m=0.325,
                ee_center=_point_from_face(
                    course,
                    normal_offset_m=-0.02,
                ),
            ),
        )
    )

    assert (
        reward.team.ee_surface_progress
        > 0.0
    )


def test_plunger_numerical_noise_is_clamped_to_physical_bounds():
    from drl.MARL.manipulation_reward import (
        compute_manipulation_raw_reward,
        manipulation_task_success,
    )

    course = _course()

    zero = _obs(
        course,
        standoff_m=0.325,
        depression_m=0.0,
    )

    tiny_negative = _obs(
        course,
        standoff_m=0.325,
        depression_m=-3.459605933956159e-08,
    )

    zero_to_noise = compute_manipulation_raw_reward(
        course,
        zero,
        tiny_negative,
    )

    assert zero_to_noise.team.depression_progress == pytest.approx(0.0)
    assert not manipulation_task_success(tiny_negative)

    full_stroke = _obs(
        course,
        standoff_m=0.325,
        depression_m=0.004,
    )

    tiny_overshoot = _obs(
        course,
        standoff_m=0.325,
        depression_m=0.0040005,
    )

    stroke_to_noise = compute_manipulation_raw_reward(
        course,
        full_stroke,
        tiny_overshoot,
    )

    assert stroke_to_noise.team.depression_progress == pytest.approx(0.0)
    assert manipulation_task_success(tiny_overshoot)



def test_impossible_plunger_depression_is_rejected():
    from drl.MARL.manipulation_reward import compute_manipulation_raw_reward
    course = _course()
    state = _obs(course, standoff_m=0.325)
    for impossible in (-1.1e-6, 0.0041, 0.040):
        with pytest.raises(ValueError, match="depression"):
            compute_manipulation_raw_reward(
                course, state, dataclasses.replace(state, button_depression_m=impossible)
            )


@pytest.mark.parametrize("field", ["approach_progress", "ee_surface_progress", "orientation_progress"])
def test_manipulation_region_closed_loops_do_not_create_progress(field):
    from drl.MARL.manipulation_reward import compute_manipulation_raw_reward
    course = _course()
    # Changing quality inside and outside the old pairwise gate must cancel.
    states = [
        _obs(course, standoff_m=0.40, normal_error_deg=60,
             ee_center=_point_from_face(course, normal_offset_m=-0.06)),
        _obs(course, standoff_m=0.325, normal_error_deg=30,
             ee_center=_point_from_face(course, normal_offset_m=-0.03)),
        _obs(course, standoff_m=0.36, normal_error_deg=0,
             ee_center=_point_from_face(course)),
        _obs(course, standoff_m=0.40, normal_error_deg=0,
             ee_center=_point_from_face(course)),
    ]
    rewards = [compute_manipulation_raw_reward(course, a, b).team
               for a, b in zip(states, states[1:] + states[:1])]
    assert sum(getattr(reward, field) for reward in rewards) <= 1e-12


def test_contact_start_and_release_closed_loop_does_not_create_ee_progress():
    from drl.MARL.manipulation_reward import compute_manipulation_raw_reward
    course = _course()
    states = [
        _obs(course, standoff_m=0.325,
             ee_center=_point_from_face(course, normal_offset_m=-0.03)),
        _obs(course, standoff_m=0.325, ee_center=_point_from_face(course)),
        _obs(course, standoff_m=0.325, depression_m=0.001,
             ee_center=_point_from_face(course, normal_offset_m=-0.03)),
    ]
    total = sum(compute_manipulation_raw_reward(course, a, b).team.ee_surface_progress
                for a, b in zip(states, states[1:] + states[:1]))
    assert total <= 1e-12


def test_manipulation_shaping_is_continuous_at_region_and_contact_boundaries():
    from drl.MARL.manipulation_reward import compute_manipulation_raw_reward
    course = _course()
    for before, after in [
        (_obs(course, standoff_m=0.375000001), _obs(course, standoff_m=0.374999999)),
        (_obs(course, standoff_m=0.325, depression_m=0.000099999),
         _obs(course, standoff_m=0.325, depression_m=0.000100001)),
    ]:
        frame = compute_manipulation_raw_reward(course, before, after)
        for field in ("approach_progress", "ee_surface_progress", "orientation_progress"):
            assert abs(getattr(frame.team, field)) < 1e-4


@pytest.mark.parametrize("gamma", [0.0, 0.5, 0.99, 1.0])
def test_discounted_potentials_do_not_pay_for_closed_loops_or_terminal_resets(gamma):
    from drl.MARL.manipulation_reward import compute_manipulation_raw_reward
    course = _course()
    a = _obs(course, standoff_m=0.40, normal_error_deg=60)
    b = _obs(course, standoff_m=0.325, normal_error_deg=10,
             ee_center=_point_from_face(course), depression_m=0.001)
    forward = compute_manipulation_raw_reward(course, a, b, shaping_gamma=gamma)
    backward = compute_manipulation_raw_reward(course, b, a, shaping_gamma=gamma)
    terminal = compute_manipulation_raw_reward(course, b, a, shaping_gamma=gamma, terminated=True)
    for field in ("approach_progress", "ee_surface_progress", "orientation_progress", "depression_progress"):
        assert getattr(forward.team, field) + getattr(backward.team, field) <= 1e-12
        assert getattr(forward.team, field) + gamma * getattr(terminal.team, field) <= 1e-12
