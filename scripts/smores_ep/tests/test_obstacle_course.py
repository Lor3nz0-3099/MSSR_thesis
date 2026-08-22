
from __future__ import annotations

import pytest

from smores_ep.isaac.obstacle_course import (
    manual_obstacle_course,
    snake8_stair_test_course,
)


def test_manual_obstacle_course_exports_world_landmarks() -> None:
    course = manual_obstacle_course()

    assert course.to_observation() == {
        "frame_id": "world",
        "gap": {"near_edge_x_m": 0.65, "far_edge_x_m": 0.85},
        "ramp": {
            "entry_x_m": -1.55,
            "exit_x_m": -1.10,
            "top_height_m": 0.0,
        },
        "stairs": {
            "top_heights_m": [0.065, 0.13, 0.195],
            "first_riser_x_m": 1.25,
            "riser_depth_m": 0.28,
        },
        "button": {"center_xyz_m": [2.65, 0.455, 0.365]},
        "exit": {"center_xyz_m": [3.55, 0.0, 0.385]},
    }


def test_manual_course_has_a_real_gap_and_monotonic_stairs() -> None:
    course = manual_obstacle_course()
    gap_start, gap_end = course.gap_interval_x_m

    assert gap_end > gap_start
    for element in course.boxes:
        if not element.collidable:
            continue
        half_x = 0.5 * element.size_xyz_m[0]
        element_start = element.center_xyz_m[0] - half_x
        element_end = element.center_xyz_m[0] + half_x
        assert element_end <= gap_start or element_start >= gap_end
    assert course.stair_top_heights_m == tuple(
        sorted(course.stair_top_heights_m)
    )
    assert course.stair_top_heights_m == pytest.approx((0.065, 0.13, 0.195))


def test_manual_course_exposes_future_task_landmarks() -> None:
    course = manual_obstacle_course()
    semantics = {element.semantic for element in course.boxes}

    assert {
        "button",
        "exit_marker",
        "stair",
        "gap_landing",
        "approach_ramp",
        "rear_start_platform",
    } <= semantics
    assert course.button_center_xyz_m[2] > course.stair_top_heights_m[-1]
    assert course.exit_center_xyz_m[0] > course.button_center_xyz_m[0]


def test_snake8_stair_test_course_has_three_wheel_high_risers() -> None:
    course = snake8_stair_test_course()

    assert course.first_riser_x_m == pytest.approx(0.65)
    assert course.riser_depth_m == pytest.approx(0.28)
    assert course.stair_top_heights_m == pytest.approx(
        (0.065, 0.13, 0.195)
    )
    assert tuple(
        second - first
        for first, second in zip(
            (0.0, *course.stair_top_heights_m[:-1]),
            course.stair_top_heights_m,
        )
    ) == pytest.approx((0.065, 0.065, 0.065))


def test_snake8_stair_test_observation_is_isolated_from_full_course() -> None:
    course = snake8_stair_test_course()

    assert course.to_observation() == {
        "frame_id": "world",
        "course_profile": "snake8_stair_test",
        "stairs": {
            "top_heights_m": [0.065, 0.13, 0.195],
            "first_riser_x_m": 0.65,
            "riser_depth_m": 0.28,
        },
    }
    assert {box.semantic for box in course.boxes} == {
        "stair_test_start",
        "stair_test_riser",
        "stair_test_upper_deck",
    }
