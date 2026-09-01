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
        approach_run_m=0.272 - SPACING_M - 0.012 - 0.060,
        landing_run_m=0.060,
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


def test_corner_clearance_profile_is_continuous_with_continuous_slope() -> None:
    path = _path()
    assert 0.0 < path.transition_bias <= 1.0

    sample_dx = 1.0e-5
    for edge_x in path.riser_edges_m:
        for boundary_x in (
            edge_x - path.approach_run_m,
            edge_x - path.corner_clearance_radius_m,
            edge_x,
            edge_x + path.corner_clearance_radius_m,
            edge_x + path.landing_run_m,
        ):
            left_slope = (
                path.height_m(boundary_x)
                - path.height_m(boundary_x - sample_dx)
            ) / sample_dx
            right_slope = (
                path.height_m(boundary_x + sample_dx)
                - path.height_m(boundary_x)
            ) / sample_dx

            assert math.isfinite(left_slope)
            assert math.isfinite(right_slope)
            assert abs(right_slope - left_slope) < 0.02


def test_high_clearance_path_clears_twenty_mm_safety_envelope() -> None:
    high_radius = FORWARD_EXTENT_M + 0.020
    approach = 0.272 - SPACING_M - 0.012 - 0.060

    path = WheelCenterPath(
        staircase=ConcertinaStaircase.from_course(_course()),
        wheel_radius_m=WHEEL_RADIUS_M,
        corner_clearance_radius_m=high_radius,
        approach_run_m=approach,
        landing_run_m=0.060,
    )

    for edge_x, top_z in zip(
        path.riser_edges_m,
        path.staircase.top_heights_m,
    ):
        minimum = min(
            math.hypot(
                x_m - edge_x,
                path.height_m(x_m) - top_z,
            )
            for x_m in (
                edge_x - high_radius
                + 2.0 * high_radius * i / 1000
                for i in range(1001)
            )
        )

        assert minimum >= high_radius - 1.0e-9



def test_seed3000_treads_leave_two_module_support_plateau() -> None:
    path = _path()

    flat_run = (
        path.staircase.tread_depth_m
        - path.approach_run_m
        - path.landing_run_m
    )

    assert flat_run == pytest.approx(SPACING_M + 0.012)
    assert flat_run >= SPACING_M + 0.012 - 1.0e-9

    # Between successive risers there is now a genuine horizontal support
    # interval long enough for two consecutive rigidly-spaced centres.
    for edge_x, top_z in zip(
        path.riser_edges_m[:-1],
        path.staircase.top_heights_m[:-1],
    ):
        flat_start = edge_x + path.landing_run_m
        flat_end = (
            edge_x
            + path.staircase.tread_depth_m
            - path.approach_run_m
        )

        assert (
            flat_end - flat_start
            >= SPACING_M + 0.012 - 1.0e-9
        )
        assert path.height_m(flat_start) == pytest.approx(
            top_z + WHEEL_RADIUS_M
        )
        assert path.height_m(flat_end) == pytest.approx(
            top_z + WHEEL_RADIUS_M
        )


def test_old_long_transitions_are_accepted_for_single_module_support() -> None:
    program = SnakeStairConcertinaPlanner().plan(
        _graph(),
        _assignments(),
        {
            "path_approach_run_m": 0.135,
            "path_landing_run_m": 0.105,
        },
    )

    assert program
    assert any(
        step.phase.startswith("PATH_IK_TRACK_")
        for step in program
    )

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

    assert worst < math.radians(60.0)


@pytest.mark.parametrize(
    ("rise_m", "tread_depth_m", "stair_count"),
    (
        (0.050, 0.320, 2),
        (0.060, 0.272, 3),
        # This geometry exposed the former discontinuous corner envelope.
        (0.064, 0.251, 4),
    ),
)
def test_global_path_ik_plans_the_robust_stair_envelope(
    rise_m: float,
    tread_depth_m: float,
    stair_count: int,
) -> None:
    base = _graph()
    graph = AttributedRobotGraph(
        nodes=base.nodes,
        edges=base.edges,
        global_attributes={
            **base.global_attributes,
            "course": _course(
                rise_m=rise_m,
                tread_depth_m=tread_depth_m,
                stair_count=stair_count,
            ),
        },
    )

    program = SnakeStairConcertinaPlanner().plan(
        graph, _assignments(), {}
    )

    assert program[0].phase == "PATH_IK_PRELOAD"
    assert program[-1].phase == "PATH_IK_UPPER_DECK_SETTLE"
    assert any(step.phase == "PATH_IK_LIFT_TAIL" for step in program)


def test_planner_generates_only_global_path_ik_tracking() -> None:
    program = SnakeStairConcertinaPlanner().plan(
        _graph(), _assignments(), {}
    )
    phases = tuple(step.phase for step in program)

    assert phases[0] == "PATH_IK_PRELOAD"
    assert phases[-1] == "PATH_IK_UPPER_DECK_SETTLE"
    assert phases.count("PATH_IK_LIFT_TAIL") == 1
    assert all(
        token not in phase
        for phase in phases
        for token in ("BUILD", "GROW", "SHIFT", "ADVANCE")
    )

    tracking = tuple(
        step
        for step in program
        if step.phase.startswith("PATH_IK_TRACK_")
    )
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


def test_terminal_tail_lift_reverses_only_q0_against_supported_chain() -> None:
    assignments = _assignments()
    program = SnakeStairConcertinaPlanner().plan(
        _graph(), assignments, {}
    )
    phases = tuple(step.phase for step in program)

    lift_index = phases.index("PATH_IK_LIFT_TAIL")
    assert lift_index > 0

    previous_track = program[lift_index - 1]
    lift = program[lift_index]
    next_track = program[lift_index + 1]

    assert previous_track.phase.startswith("PATH_IK_TRACK_")
    assert next_track.phase.startswith("PATH_IK_TRACK_")

    # LIFT_TAIL is a pure posture transition: no wheel motion or spatial goal.
    assert lift.kind == "posture"
    assert lift.position_goal is None
    assert lift.displacement_goal is None

    before = {
        target.module_id: target.angle_rad
        for target in previous_track.posture_targets
    }
    lifted = {
        target.module_id: target.angle_rad
        for target in lift.posture_targets
    }

    assert set(before) == set(lifted)
    tail_id = assignments[0].module_id

    # q0 alone changes sign: the support side of the serial chain has changed.
    assert abs(before[tail_id]) > math.radians(3.0)
    assert lifted[tail_id] == pytest.approx(-before[tail_id])

    for assignment in assignments[1:]:
        assert lifted[assignment.module_id] == pytest.approx(
            before[assignment.module_id]
        )

    # Normal IK resumes after the special terminal reaction maneuver.
    assert any(
        step.phase.startswith("PATH_IK_TRACK_")
        for step in program[lift_index + 1 : -1]
    )


def test_path_translation_slows_while_loaded_tilts_are_moving() -> None:
    program = SnakeStairConcertinaPlanner().plan(
        _graph(), _assignments(), {}
    )

    tracking = [
        step for step in program
        if step.phase.startswith("PATH_IK_TRACK_")
    ]

    assert tracking
    assert any(step.linear_m_s < 0.040 for step in tracking)

    for step in tracking:
        assert 0.0 < step.linear_m_s <= 0.040
        assert step.posture_reached_linear_m_s == pytest.approx(0.040)
        assert (
            step.minimum_tracking_linear_m_s
            <= step.linear_m_s + 1.0e-12
        )


def test_path_ik_keeps_all_eight_modules_in_traction() -> None:
    program = SnakeStairConcertinaPlanner().plan(
        _graph(), _assignments(), {}
    )

    tracking = [
        step
        for step in program
        if step.phase.startswith("PATH_IK_TRACK_")
    ]

    assert tracking

    for step in tracking:
        assert step.active_target_roles == ROLES
        assert step.posture_reached_active_target_roles == ROLES



def test_tracking_tilts_are_time_synchronized() -> None:
    assignments = _assignments()

    targets = SnakeStairConcertinaPlanner._posture_targets(
        phase="PATH_IK_TRACK_SYNC_TEST",
        previous_tilts=(0.0,) * 8,
        tilts=(0.20, 0.10, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0),
        assignments=assignments,
    )

    by_module = {
        target.module_id: target
        for target in targets
    }

    fast = by_module["m0"]
    slow = by_module["m1"]
    tiny = by_module["m2"]

    assert fast.max_servo_speed_rad_s == pytest.approx(0.45)
    assert slow.max_servo_speed_rad_s == pytest.approx(0.225)

    # Both meaningful motions have the same nominal completion time.
    assert (
        0.20 / fast.max_servo_speed_rad_s
        == pytest.approx(
            0.10 / slow.max_servo_speed_rad_s
        )
    )

    # Tiny changes already within tolerance must not slow the entire group.
    assert tiny.max_servo_speed_rad_s == pytest.approx(0.45)

    # PATH-IK uses the narrow 0.015-rad band only for servo-speed
    # synchronization. Micro-waypoint completion deliberately allows
    # a 5-degree physical tracking tolerance under load.
    assert all(
        target.tolerance_rad == pytest.approx(math.radians(5.0))
        for target in targets
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
    # Final tail placement must clear the enlarged corner envelope.
    # With the default 20 mm safety:
    #   corner radius = forward extent + safety
    #   tail inset    = max(landing run, corner radius + 10 mm)
    expected_tail_inset = max(
        0.105,
        FORWARD_EXTENT_M + 0.020 + 0.010,
    )
    expected_head_x = (
        0.65
        + 2 * 0.272
        + expected_tail_inset
        + 7 * SPACING_M
    )
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
