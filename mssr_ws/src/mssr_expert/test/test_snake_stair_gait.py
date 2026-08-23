"""Tests for the deterministic Snake8 staircase follower."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from mssr_expert.behaviors.morphology_library import (
    AssignedModule,
    BehaviorProgramStep,
    LongitudinalDisplacementGoal,
    MorphologyLibrary,
)
from mssr_expert.behaviors.snake_stair_gait import (
    SnakeStairGaitError,
    SnakeStairGaitPlanner,
    UniformStaircase,
)
from mssr_expert.graph.attributed_robot_graph import (
    AttributedRobotGraph,
    GraphNode,
)
from mssr_expert.execution.morphology_behavior_executor import (
    MorphologyBehaviorExecutor,
    MorphologyCommand,
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


def _assignments() -> tuple[AssignedModule, ...]:
    return tuple(
        AssignedModule(f"m{index}", f"v{index}", role)
        for index, role in enumerate(ROLES)
    )


def _course() -> dict:
    return {
        "frame_id": "world",
        "course_profile": "snake8_stair_test",
        "stairs": {
            "first_riser_x_m": 0.65,
            "riser_depth_m": 0.28,
            "top_heights_m": [0.065, 0.13, 0.195],
        },
    }


def _graph(*, lateral_step_m: float = 0.0) -> AttributedRobotGraph:
    spacing = 0.07777
    return AttributedRobotGraph(
        nodes=tuple(
            GraphNode(
                f"m{index}",
                {
                    "position": [
                        index * spacing,
                        index * lateral_step_m,
                        0.031,
                    ]
                },
            )
            for index in range(8)
        ),
        global_attributes={
            "course": _course(),
            "module_geometry": {"wheel_radius_m": 0.03106},
        },
    )


def test_uniform_staircase_recognizes_the_shared_course_geometry() -> None:
    stairs = UniformStaircase.from_course(_course())

    assert stairs.rise_m == pytest.approx(0.065)
    assert stairs.tread_depth_m == pytest.approx(0.28)
    assert stairs.top_heights_m == pytest.approx((0.065, 0.13, 0.195))


def test_profiles_hold_two_risers_when_chain_spans_two_steps() -> None:
    planner = SnakeStairGaitPlanner()
    angle = math.asin(0.065 / 0.07777)

    assert planner.profile_offsets(
        phase=0, stair_count=3, stride=4, bend_angle=angle
    ) == pytest.approx((0, 0, 0, 0, angle, -angle, 0, 0))
    assert planner.profile_offsets(
        phase=2, stair_count=3, stride=4, bend_angle=angle
    ) == pytest.approx((0, 0, angle, -angle, 0, 0, 0, 0))
    assert planner.profile_offsets(
        phase=3, stair_count=3, stride=4, bend_angle=angle
    ) == pytest.approx((0, angle, -angle, 0, 0, angle, -angle, 0))


def test_plan_micro_interleaves_conforming_postures_and_crawl() -> None:
    program = SnakeStairGaitPlanner().plan(
        _graph(),
        _assignments(),
        {
            "profile_substeps": 2,
            "linear_m_s": 0.030,
        },
    )
    phases = [step.phase for step in program]
    drive_steps = [step for step in program if step.kind == "drive"]

    assert phases[:3] == [
        "LIFT_FIRST_RISER",
        "APPROACH_FIRST_RISER",
        "CONFORM_PROFILE_00",
    ]
    assert phases[3:7] == [
        "PROFILE_00_01",
        "CRAWL_00_01",
        "PROFILE_00_02",
        "CRAWL_00_02",
    ]
    assert phases[-1] == "UPPER_DECK_ADVANCE"
    assert drive_steps
    assert all(step.duration_s is None for step in drive_steps)
    crawl = next(step for step in program if step.phase == "CRAWL_02_01")
    approach = program[1]
    assert approach.position_goal is not None
    assert approach.position_goal.module_id == "m5"
    assert approach.position_goal.target_x_m == pytest.approx(0.65 - 0.03106)
    assert approach.position_goal.tolerance_m == pytest.approx(0.010)
    assert approach.duration_s is None
    assert approach.kind == "drive"
    assert crawl.active_target_roles == (
        "snake_tail",
        "snake_rear",
        "snake_center_front",
        "snake_shoulder",
        "snake_head",
    )
    assert crawl.duration_s is None
    assert crawl.displacement_goal is not None
    assert crawl.displacement_goal.module_ids == (
        "m0",
        "m1",
        "m4",
        "m5",
        "m7",
    )
    assert crawl.displacement_goal.distance_m == pytest.approx(0.07777 / 2)
    assert crawl.displacement_goal.tolerance_m == pytest.approx(0.004)
    upper_deck = program[-1]
    assert upper_deck.duration_s is None
    assert upper_deck.displacement_goal is not None
    assert upper_deck.displacement_goal.module_ids == tuple(
        f"m{index}" for index in range(8)
    )
    assert upper_deck.displacement_goal.distance_m == pytest.approx(0.07777)


def test_first_crawl_uses_ground_tail_and_upper_tread_supports() -> None:
    program = SnakeStairGaitPlanner().plan(
        _graph(),
        _assignments(),
        {"profile_substeps": 1},
    )

    first_crawl = next(
        step for step in program if step.phase == "CRAWL_00_01"
    )

    assert first_crawl.active_target_roles == (
        "snake_tail",
        "snake_rear",
        "snake_hip",
        "snake_center_rear",
        "snake_neck",
        "snake_head",
    )
    assert first_crawl.displacement_goal is not None
    assert first_crawl.displacement_goal.module_ids == (
        "m0",
        "m1",
        "m2",
        "m3",
        "m6",
        "m7",
    )


def test_plan_rejects_a_snake_not_aligned_with_the_known_stairs() -> None:
    with pytest.raises(SnakeStairGaitError, match="not aligned"):
        SnakeStairGaitPlanner().plan(
            _graph(lateral_step_m=0.04),
            _assignments(),
            {},
        )


def test_executor_accepts_generated_program_without_library_entry() -> None:
    assignments = _assignments()
    program = SnakeStairGaitPlanner().plan(
        _graph(), assignments, {"profile_substeps": 1}
    )
    library = MorphologyLibrary.load(
        Path(__file__).parents[1]
        / "config"
        / "smores_morphology_behaviors.json"
    )
    executor = MorphologyBehaviorExecutor(library)
    executor.start(
        MorphologyCommand("crawl-test", "snake8", "crawl_stairs"),
        assignments,
        {item.module_id: -0.1 for item in assignments},
        program,
    )

    first = executor.step(0.0)
    assert first.phase == "LIFT_FIRST_RISER"
    assert first.primitive_goal is not None
    assert first.primitive_goal.module_ids == ("m4",)


def test_approach_drives_until_the_live_world_x_goal() -> None:
    assignments = _assignments()
    program = SnakeStairGaitPlanner().plan(
        _graph(), assignments, {"profile_substeps": 1}
    )
    executor = MorphologyBehaviorExecutor(
        MorphologyLibrary.load(
            Path(__file__).parents[1]
            / "config"
            / "smores_morphology_behaviors.json"
        )
    )
    executor.start(
        MorphologyCommand("crawl-goal", "snake8", "crawl_stairs"),
        assignments,
        {item.module_id: 0.0 for item in assignments},
        program,
    )
    first = executor.step(0.0)
    completed_lift = executor.step(
        0.1,
        {
            "schema_version": "mssr.primitive_status.v1",
            "goal_id": first.primitive_goal.goal_id,
            "primitive": first.primitive_goal.primitive,
            "module_ids": list(first.primitive_goal.module_ids),
            "state": "succeeded",
            "phase": "terminal",
            "progress": 1.0,
            "code": "JOINT_TARGET_REACHED",
            "message": "done",
        },
    )
    assert completed_lift.phase == "LIFT_FIRST_RISER_COMPLETE"

    goal = program[1].position_goal
    assert goal is not None
    moving = executor.step(
        0.2,
        module_positions={"m5": (goal.target_x_m - 0.10, 0.0, 0.096)},
    )
    assert moving.phase == "APPROACH_FIRST_RISER"
    assert moving.locomotion
    assert "error=0.100m" in moving.message

    reached = executor.step(
        5.0,
        module_positions={
            "m4": (0.585, 0.0, 0.031),
            "m5": (0.636, 0.0, 0.096),
        },
    )
    assert reached.phase == "APPROACH_FIRST_RISER_GOAL_REACHED"
    assert not reached.locomotion
    assert not reached.done


def test_approach_has_no_time_limit_while_position_keeps_progressing() -> None:
    assignments = _assignments()
    program = SnakeStairGaitPlanner().plan(
        _graph(),
        assignments,
        {"profile_substeps": 1},
    )
    executor = MorphologyBehaviorExecutor(
        MorphologyLibrary.load(
            Path(__file__).parents[1]
            / "config"
            / "smores_morphology_behaviors.json"
        )
    )
    executor.start(
        MorphologyCommand("crawl-slow", "snake8", "crawl_stairs"),
        assignments,
        {item.module_id: 0.0 for item in assignments},
        program,
    )
    first = executor.step(0.0)
    executor.step(
        0.1,
        {
            "schema_version": "mssr.primitive_status.v1",
            "goal_id": first.primitive_goal.goal_id,
            "primitive": first.primitive_goal.primitive,
            "module_ids": list(first.primitive_goal.module_ids),
            "state": "succeeded",
            "phase": "terminal",
            "progress": 1.0,
            "code": "JOINT_TARGET_REACHED",
            "message": "done",
        },
    )
    goal = program[1].position_goal
    executor.step(
        0.2,
        module_positions={"m5": (goal.target_x_m - 0.10, 0.0, 0.096)},
    )
    still_moving = executor.step(
        3600.0,
        module_positions={"m5": (goal.target_x_m - 0.08, 0.0, 0.096)},
    )

    assert not still_moving.done
    assert still_moving.state == "RUNNING_PROGRAM_DRIVE"
    assert still_moving.phase == "APPROACH_FIRST_RISER"
    assert still_moving.locomotion


def test_approach_waits_without_driving_when_live_pose_is_missing() -> None:
    assignments = _assignments()
    program = SnakeStairGaitPlanner().plan(
        _graph(), assignments, {"profile_substeps": 1}
    )
    executor = MorphologyBehaviorExecutor(
        MorphologyLibrary.load(
            Path(__file__).parents[1]
            / "config"
            / "smores_morphology_behaviors.json"
        )
    )
    executor.start(
        MorphologyCommand("crawl-no-pose", "snake8", "crawl_stairs"),
        assignments,
        {item.module_id: 0.0 for item in assignments},
        program,
    )
    first = executor.step(0.0)
    executor.step(
        0.1,
        {
            "schema_version": "mssr.primitive_status.v1",
            "goal_id": first.primitive_goal.goal_id,
            "primitive": first.primitive_goal.primitive,
            "module_ids": list(first.primitive_goal.module_ids),
            "state": "succeeded",
            "phase": "terminal",
            "progress": 1.0,
            "code": "JOINT_TARGET_REACHED",
            "message": "done",
        },
    )

    waiting = executor.step(1000.0, module_positions={})

    assert not waiting.done
    assert waiting.state == "WAITING_PROGRAM_POSITION"
    assert not waiting.locomotion
    assert "locomotion stopped" in waiting.message


def test_crawl_displacement_uses_live_support_centroid_without_timer() -> None:
    assignments = _assignments()
    executor = MorphologyBehaviorExecutor(
        MorphologyLibrary.load(
            Path(__file__).parents[1]
            / "config"
            / "smores_morphology_behaviors.json"
        )
    )
    goal = LongitudinalDisplacementGoal(
        module_ids=("m4", "m5", "m7"),
        distance_m=0.040,
        tolerance_m=0.003,
    )
    executor.start(
        MorphologyCommand("crawl-centroid", "snake8", "crawl_stairs"),
        assignments,
        {item.module_id: 0.0 for item in assignments},
        (
            BehaviorProgramStep(
                phase="CRAWL_GEOMETRIC",
                linear_m_s=0.030,
                active_target_roles=(
                    "snake_center_front",
                    "snake_shoulder",
                    "snake_head",
                ),
                displacement_goal=goal,
            ),
        ),
    )
    origin = {
        "m4": (0.40, 0.0, 0.031),
        "m5": (0.50, 0.0, 0.096),
        "m7": (0.70, 0.0, 0.096),
    }

    moving = executor.step(0.0, module_positions=origin)
    assert moving.phase == "CRAWL_GEOMETRIC"
    assert moving.locomotion
    assert "traveled=0.000m" in moving.message

    slow = executor.step(
        3600.0,
        module_positions={
            module_id: (position[0] + 0.020, position[1], position[2])
            for module_id, position in origin.items()
        },
    )
    assert slow.locomotion
    assert not slow.done
    assert "traveled=0.020m" in slow.message

    reached = executor.step(
        7200.0,
        module_positions={
            module_id: (position[0] + 0.038, position[1], position[2])
            for module_id, position in origin.items()
        },
    )
    assert reached.phase == "CRAWL_GEOMETRIC_GOAL_REACHED"
    assert not reached.locomotion


def test_crawl_displacement_waits_if_one_support_pose_is_missing() -> None:
    assignments = _assignments()
    executor = MorphologyBehaviorExecutor(
        MorphologyLibrary.load(
            Path(__file__).parents[1]
            / "config"
            / "smores_morphology_behaviors.json"
        )
    )
    executor.start(
        MorphologyCommand("crawl-missing-support", "snake8", "crawl_stairs"),
        assignments,
        {item.module_id: 0.0 for item in assignments},
        (
            BehaviorProgramStep(
                phase="CRAWL_GEOMETRIC",
                linear_m_s=0.030,
                active_target_roles=("snake_center_front", "snake_head"),
                displacement_goal=LongitudinalDisplacementGoal(
                    module_ids=("m4", "m7"),
                    distance_m=0.040,
                    tolerance_m=0.003,
                ),
            ),
        ),
    )

    waiting = executor.step(
        1000.0,
        module_positions={"m4": (0.4, 0.0, 0.031)},
    )

    assert waiting.state == "WAITING_PROGRAM_POSITION"
    assert not waiting.locomotion
    assert "m7" in waiting.message


def test_recognizer_rejects_nonuniform_risers() -> None:
    course = _course()
    course["stairs"]["top_heights_m"] = [0.065, 0.15, 0.195]

    with pytest.raises(SnakeStairGaitError, match="uniform"):
        UniformStaircase.from_course(course)


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"profile_substeps": "three"}, "must be an integer"),
        ({"profile_substeps": float("inf")}, "must be an integer"),
        ({"max_alignment_error_rad": 0.0}, "must be in"),
        ({"riser_approach_tolerance_m": 0.001}, "must be in"),
        ({"crawl_goal_tolerance_m": 0.020}, "must be in"),
        ({"profile_substeps": 6, "crawl_goal_tolerance_m": 0.010}, "half"),
        ({"upper_deck_advance_distance_m": 0.010}, "half one link"),
        ({"slip_compensation": 1.5}, "does not accept timed parameters"),
        ({"tread_advance_duration_s": 4.0}, "does not accept timed"),
    ],
)
def test_plan_rejects_invalid_runtime_parameters(
    parameters: dict, message: str
) -> None:
    with pytest.raises(SnakeStairGaitError, match=message):
        SnakeStairGaitPlanner().plan(
            _graph(),
            _assignments(),
            parameters,
        )
