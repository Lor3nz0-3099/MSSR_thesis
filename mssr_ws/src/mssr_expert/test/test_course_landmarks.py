"""Tests for the Isaac-to-ROS obstacle-course landmark contract."""

import pytest

from mssr_expert.planning.smores_ep.course_landmarks import (
    CourseLandmarkError,
    CourseLandmarks,
)


def _observation() -> dict:
    return {
        "course": {
            "frame_id": "world",
            "gap": {"near_edge_x_m": 0.65, "far_edge_x_m": 0.85},
            "ramp": {
                "entry_x_m": -1.55,
                "exit_x_m": -1.10,
                "top_height_m": 0.0,
            },
            "stairs": {
                "top_heights_m": [0.04, 0.08, 0.12],
                "first_riser_x_m": 1.25,
                "riser_depth_m": 0.30,
            },
            "button": {"center_xyz_m": [2.65, 0.455, 0.29]},
            "exit": {"center_xyz_m": [3.55, 0.0, 0.31]},
        }
    }


def test_landmarks_parse_the_isaac_world_observation() -> None:
    landmarks = CourseLandmarks.from_observation(_observation())

    assert landmarks.gap_near_x_m == pytest.approx(0.65)
    assert landmarks.ramp_exit_x_m == pytest.approx(-1.10)
    assert landmarks.stair_top_heights_m == pytest.approx((0.04, 0.08, 0.12))
    assert landmarks.button_center_xyz_m == pytest.approx((2.65, 0.455, 0.29))


def test_landmarks_reject_missing_isaac_course_metadata() -> None:
    with pytest.raises(CourseLandmarkError, match="unavailable"):
        CourseLandmarks.from_observation({})