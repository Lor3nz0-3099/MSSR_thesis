"""Tests for the deterministic Snake8 staircase follower."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from mssr_expert.behaviors.morphology_library import (
    AssignedModule,
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
        global_attributes={"course": _course()},
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
            "slip_compensation": 1.0,
            "linear_m_s": 0.030,
        },
    )
    phases = [step.phase for step in program]

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
    crawl = next(step for step in program if step.phase == "CRAWL_02_01")
    approach = program[1]
    assert approach.position_goal is not None
    assert approach.position_goal.module_id == "m4"
    assert approach.position_goal.target_x_m == pytest.approx(
        0.65 - 0.5 * math.sqrt(0.07777**2 - 0.065**2)
    )
    assert approach.position_goal.tolerance_m == pytest.approx(0.010)
    assert approach.duration_s is None
    assert approach.kind == "drive"
    assert crawl.active_target_roles == (
        "snake_center_front",
        "snake_shoulder",
        "snake_head",
    )
    assert crawl.duration_s == pytest.approx((0.07777 / 2) / 0.030)


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
        module_positions={"m4": (goal.target_x_m - 0.10, 0.0, 0.031)},
    )
    assert moving.phase == "APPROACH_FIRST_RISER"
    assert moving.locomotion
    assert "error=0.100m" in moving.message

    reached = executor.step(
        5.0,
        module_positions={
            "m4": (goal.target_x_m - 0.005, 0.0, 0.031)
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
        module_positions={"m4": (goal.target_x_m - 0.10, 0.0, 0.031)},
    )
    still_moving = executor.step(
        3600.0,
        module_positions={"m4": (goal.target_x_m - 0.08, 0.0, 0.031)},
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
