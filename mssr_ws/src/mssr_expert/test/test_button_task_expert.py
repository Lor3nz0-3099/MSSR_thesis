"""Focused tests for the dynamic camera-guided button sub-expert."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from mssr_expert.planning.smores_ep.button_task_expert import (
    button_base_target,
    button_is_physically_pressed,
    longitudinal_tracking_error,
    observe_button,
)
from mssr_expert.planning.smores_ep.obstacle_course_policy import (
    MissionTask,
    ObstacleCoursePolicy,
)
from mssr_expert.planning.smores_ep.composite_mission import (
    CompositeMissionPlanner,
)


def test_live_button_pose_overrides_nominal_seed_pose() -> None:
    observation = {
        "course": {
            "button": {
                "center_xyz_m": [0.85, 0.465, 0.170],
                "current_center_xyz_m": [1.17, 0.72, 0.285],
                "depression_m": 0.0012,
            }
        }
    }

    button = observe_button(observation)

    assert button.center_xyz_m == pytest.approx(
        (1.17, 0.72, 0.285)
    )
    assert button.depression_m == pytest.approx(0.0012)


def test_button_relative_base_target_tracks_dynamic_xy_and_height() -> None:
    target = button_base_target(
        (1.17, 0.72, 0.285),
        standoff_m=0.55,
        future_forward_yaw_rad=-math.pi / 2.0,
    )

    # Forward points toward -Y, therefore the root sits 0.55 m
    # below the button and its rear/arm side faces +Y toward it.
    assert target.root_xy_m == pytest.approx(
        (1.17, 0.17),
        abs=1e-9,
    )

    # Height is deliberately preserved for the future arm IK.
    assert target.button_center_xyz_m[2] == pytest.approx(0.285)


def test_mm8_longitudinal_error_has_correct_reverse_sign() -> None:
    target = button_base_target(
        (1.0, 0.70, 0.20),
        standoff_m=0.30,
        future_forward_yaw_rad=-math.pi / 2.0,
    )

    # Robot is farther from the wall than the target.
    # Since its forward axis is -Y, reaching larger Y means REVERSE.
    error = longitudinal_tracking_error(
        (1.0, 0.10, 0.03),
        -math.pi / 2.0,
        target,
    )

    assert error.along_m < 0.0
    assert error.lateral_m == pytest.approx(0.0, abs=1e-9)
    assert error.heading_error_rad == pytest.approx(
        0.0,
        abs=1e-9,
    )


def test_button_success_is_physical_depression() -> None:
    observation = {
        "course": {
            "button": {
                "current_center_xyz_m": [0.85, 0.465, 0.170],
                "depression_m": 0.00349,
            }
        }
    }

    assert not button_is_physically_pressed(
        observation,
        threshold_m=0.0035,
    )

    observation["course"]["button"]["depression_m"] = 0.00350

    assert button_is_physically_pressed(
        observation,
        threshold_m=0.0035,
    )


def test_planner_uses_complete_validated_button_macro() -> None:
    policy = ObstacleCoursePolicy(
        {
            "rc_car8": {"flat_navigation"},
            "snake8": {"cross_gap", "climb_stairs"},
            "mobile_manipulator8": {"press_button"},
        }
    )
    stages = CompositeMissionPlanner(policy).build(
        (
            MissionTask("button-a", "button", {"seed": 6101}),
            MissionTask("goal", "goal", {"center_xyz_m": [2.0, 0.0, 0.0]}),
        )
    )

    button = [stage for stage in stages if stage.task_id == "button-a"]
    assert [stage.kind for stage in button] == [
        "assembly",
        "button_rc_alignment",
        "reconfiguration",
        "button_expert",
        "reconfiguration",
    ]
    assert button[-1].target_morphology == "rc_car8"


def test_mm8_restore_drive_is_only_validated_base_transition() -> None:
    package_root = Path(__file__).parents[1]

    data = json.loads(
        (
            package_root
            / "config"
            / "smores_morphology_behaviors.json"
        ).read_text()
    )

    mm8 = data["morphologies"]["mobile_manipulator8"]

    assert mm8["behaviors"]["restore_drive"] == [
        "retreat_drive_base"
    ]

    posture = mm8["postures"]["retreat_drive_base"]

    assert [
        item["target_role"]
        for item in posture
    ] == [
        "chassis_center",
        "front_support",
        "arm_ground_drive",
    ]

    assert all(
        "coordination_group" not in item
        for item in posture
    )

    assert posture[0]["angle_rad"] == pytest.approx(
        -math.pi / 4.0
    )
    assert posture[1]["angle_rad"] == pytest.approx(
        math.radians(10.0)
    )
    assert posture[2]["angle_rad"] == pytest.approx(
        math.pi / 4.0
    )
