"""Tests for collider-aware Snake8 stair path IK and tracking."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import pytest

from mssr_expert.behaviors.morphology_library import (
    AssignedModule,
    BehaviorProgramStep,
    LongitudinalPositionGoal,
    MorphologyLibrary,
)
from mssr_expert.behaviors.snake_stair_concertina import (
    SnakeStairConcertinaPlanner,
)
from mssr_expert.behaviors.snake_stair_concertina_geometry import (
    ConcertinaStaircase,
)
from mssr_expert.behaviors.snake_stair_gait import SnakeStairGaitError
from mssr_expert.behaviors.snake_stair_path_ik import (
    WheelCenterPath,
    reconstruct_centers,
    relative_tilt_ik,
)
from mssr_expert.execution.morphology_behavior_executor import (
    MorphologyBehaviorExecutor,
    MorphologyCommand,
)
from mssr_expert.graph.attributed_robot_graph import (
    AttributedRobotGraph,
    GraphNode,
)


ROLES = (
    "snake_tail",
    "snake_rear",
    "snake_hip",
    "snake_center_rear",
    "snake_center_front",
    "snake_shoulder",
    "snake_neck",
    "snake_head",
)
SPACING_M = 0.07777
WHEEL_RADIUS_M = 0.03106
FORWARD_EXTENT_M = 0.043771
CORNER_RADIUS_M = FORWARD_EXTENT_M + 0.005


def _assignments() -> tuple[AssignedModule, ...]:
    return tuple(
        AssignedModule(f"m{index}", f"v{index}", role)
        for index, role in enumerate(ROLES)
    )


def _course(
    *,
    rise_m: float = 0.060,
    tread_depth_m: float = 0.272,
    stair_count: int = 3,
) -> dict:
    collision_boxes = [
        {
            "name": f"Stair{index:02d}",
            "center_xyz_m": [
                0.65 + (index - 0.5) * tread_depth_m,
                0.0,
                0.5 * rise_m * index,
            ],
            "size_xyz_m": [
                tread_depth_m,
                1.2,
                rise_m * index,
            ],
            "semantic": "stair_test_riser",
            "pitch_deg": 0.0,
        }
        for index in range(1, stair_count + 1)
    ]
    return {
        "frame_id": "world",
        "stairs": {
            "first_riser_x_m": 0.65,
            "riser_depth_m": tread_depth_m,
            "top_heights_m": [
                rise_m * index for index in range(1, stair_count + 1)
            ],
        },
        "collision_boxes": collision_boxes,
    }


def _graph(
    *, neutral_tilts: Mapping[str, float] | None = None
) -> AttributedRobotGraph:
    return AttributedRobotGraph(
        nodes=tuple(
            GraphNode(
                f"m{index}",
                {
                    "position": [
                        index * SPACING_M,
                        0.0,
                        WHEEL_RADIUS_M,
                    ],
                    "actuators": {
                        "tilt": {
                            "position_rad": (
                                neutral_tilts.get(f"m{index}", 0.0)
                                if neutral_tilts is not None
                                else 0.0
                            ),
                            "lower_limit_rad": -math.pi / 2.0,
                            "upper_limit_rad": math.pi / 2.0,
                        }
                    },
                },
            )
            for index in range(8)
        ),
        global_attributes={
            "course": _course(),
            "module_geometry": {
                "wheel_radius_m": WHEEL_RADIUS_M,
                "forward_collision_extent_m": FORWARD_EXTENT_M,
                "pan_face_radius_m": 0.03140,
            },
        },
    )


def _path() -> WheelCenterPath:
    return WheelCenterPath(
        staircase=ConcertinaStaircase.from_course(_course()),
        wheel_radius_m=WHEEL_RADIUS_M,
        corner_clearance_radius_m=CORNER_RADIUS_M,
        approach_run_m=0.135,
        landing_run_m=0.105,
    )


def _library() -> MorphologyLibrary:
    return MorphologyLibrary.load(
        Path(__file__).parents[1]
        / "config"
        / "smores_morphology_behaviors.json"
    )


def test_course_profile_is_cross_checked_against_collision_boxes() -> None:
    course = _course()
    course["collision_boxes"][0]["center_xyz_m"][0] += 0.010

    with pytest.raises(SnakeStairGaitError, match="collision boxes"):
        ConcertinaStaircase.from_course(course)


def test_curve_has_a_clearance_apex_and_settles_on_each_tread() -> None:
    path = _path()
    for edge_x, top_z in zip(
        path.riser_edges_m, path.staircase.top_heights_m
    ):
        assert path.height_m(edge_x) == pytest.approx(
            top_z + CORNER_RADIUS_M
        )
        assert path.height_m(edge_x + path.landing_run_m) == pytest.approx(
            top_z + WHEEL_RADIUS_M
        )


def test_curve_clears_the_full_corner_exclusion_radius() -> None:
    path = _path()
    for edge_x, top_z in zip(
        path.riser_edges_m, path.staircase.top_heights_m
    ):
        minimum = min(
            math.hypot(x_m - edge_x, path.height_m(x_m) - top_z)
            for x_m in (
                edge_x - CORNER_RADIUS_M
                + 2.0 * CORNER_RADIUS_M * index / 1000
                for index in range(1001)
            )
        )
        assert minimum >= CORNER_RADIUS_M - 1.0e-9


def test_eight_centres_use_true_rigid_link_spacing() -> None:
    points = _path().sample_module_centers(
        head_x_m=0.922,
        module_count=8,
        link_length_m=SPACING_M,
    )

    assert len(points) == 8
    assert all(
        math.hypot(upper.x_m - lower.x_m, upper.z_m - lower.z_m)
        == pytest.approx(SPACING_M, abs=1.0e-10)
        for lower, upper in zip(points, points[1:])
    )


def test_relative_ik_reconstructs_all_eight_centres() -> None:
    points = _path().sample_module_centers(
        head_x_m=1.05,
        module_count=8,
        link_length_m=SPACING_M,
    )
    tilts = relative_tilt_ik(points)
    reconstructed = reconstruct_centers(points[0], tilts, SPACING_M)

    assert len(tilts) == 8
    assert tilts[-1] == pytest.approx(0.0)
    for actual, expected in zip(reconstructed, points):
        assert actual.x_m == pytest.approx(expected.x_m, abs=1.0e-10)
        assert actual.z_m == pytest.approx(expected.z_m, abs=1.0e-10)


def test_seed3000_geometry_stays_well_inside_ninety_degrees() -> None:
    path = _path()
    worst = 0.0
    for index in range(1001):
        head_x_m = 0.54439 + index * (1.85 - 0.54439) / 1000
        points = path.sample_module_centers(
            head_x_m=head_x_m,
            module_count=8,
            link_length_m=SPACING_M,
        )
        worst = max(worst, *(abs(value) for value in relative_tilt_ik(points)))

    assert worst < math.radians(50.0)


def test_planner_generates_only_global_path_ik_tracking() -> None:
    program = SnakeStairConcertinaPlanner().plan(
        _graph(), _assignments(), {}
    )
    phases = tuple(step.phase for step in program)

    assert phases[0] == "PATH_IK_PRELOAD"
    assert phases[-1] == "PATH_IK_UPPER_DECK_SETTLE"
    assert all(
        token not in phase
        for phase in phases
        for token in ("BUILD", "GROW", "SHIFT", "ADVANCE")
    )
    tracking = program[1:-1]
    assert tracking
    assert all(step.kind == "posture_drive" for step in tracking)
    assert all(len(step.posture_targets) == 8 for step in program)
    assert all(step.position_tracking_kp_s_inv == 2.0 for step in tracking)
    assert all(step.position_tracking_kd == 0.25 for step in tracking)
    assert all(
        target.angle_reference == "captured_neutral"
        for step in program
        for target in step.posture_targets
    )


def test_final_solution_is_flat_and_landed_beyond_the_last_corner() -> None:
    program = SnakeStairConcertinaPlanner().plan(
        _graph(), _assignments(), {}
    )
    final = program[-1]

    assert all(
        target.angle_rad == pytest.approx(0.0, abs=1.0e-8)
        for target in final.posture_targets
    )
    final_track = program[-2]
    assert final_track.position_goal is not None
    expected_head_x = 0.65 + 2 * 0.272 + 0.105 + 7 * SPACING_M
    assert final_track.position_goal.target_x_m == pytest.approx(
        expected_head_x, abs=1.0e-7
    )


@pytest.mark.parametrize(
    ("parameters", "message"),
    (
        ({"path_corner_safety_m": 0.001}, "corner_safety"),
        ({"trajectory_step_m": 0.050}, "trajectory_step"),
        ({"trajectory_tracking_kd": -1.0}, "tracking_kd"),
        ({"path_tail_landing_inset_m": 0.050}, "tail_landing"),
    ),
)
def test_invalid_path_or_controller_parameters_are_rejected(
    parameters: dict, message: str
) -> None:
    with pytest.raises(SnakeStairGaitError, match=message):
        SnakeStairConcertinaPlanner().plan(
            _graph(), _assignments(), parameters
        )


def test_pd_tracking_decelerates_from_live_position_and_velocity() -> None:
    executor = MorphologyBehaviorExecutor(_library())
    step = BehaviorProgramStep(
        phase="PATH_IK_TRACK_TEST",
        linear_m_s=0.040,
        active_target_roles=ROLES,
        position_goal=LongitudinalPositionGoal("m7", 1.012, 0.001),
        position_tracking_kp_s_inv=2.0,
        position_tracking_kd=0.25,
        minimum_tracking_linear_m_s=0.005,
    )
    executor.start(
        MorphologyCommand(
            "path-pd-test",
            "snake8",
            "crawl_stairs_spatial_concertina",
        ),
        _assignments(),
        program_steps_override=(step,),
    )

    first = executor._position_tracking_speed(
        step, 0.0, {"m7": (1.000, 0.0, 0.03)}, speed_limit_m_s=0.040
    )
    second = executor._position_tracking_speed(
        step, 0.1, {"m7": (1.002, 0.0, 0.03)}, speed_limit_m_s=0.040
    )

    assert first == pytest.approx(0.024)
    assert 0.005 <= second < first
