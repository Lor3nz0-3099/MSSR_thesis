"""Tests for the geometric Snake8 gap-crossing gait."""

from __future__ import annotations

import math

import pytest

from mssr_expert.behaviors.morphology_library import AssignedModule
from mssr_expert.behaviors.snake_gap_gait import (
    FlatGap,
    SnakeGapGaitError,
    SnakeGapGaitPlanner,
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


def _assignments() -> tuple[AssignedModule, ...]:
    return tuple(
        AssignedModule(f"m{index}", f"v{index}", role)
        for index, role in enumerate(ROLES)
    )


def _course(near: float = 0.55, far: float = 0.75) -> dict:
    return {
        "frame_id": "world",
        "course_profile": "snake8_gap_test",
        "gap": {
            "near_edge_x_m": near,
            "far_edge_x_m": far,
            "width_m": far - near,
        },
    }


def _graph(
    *,
    near: float = 0.55,
    far: float = 0.75,
    lateral_step_m: float = 0.0,
) -> AttributedRobotGraph:
    spacing = 0.07777
    return AttributedRobotGraph(
        nodes=tuple(
            GraphNode(
                f"m{index}",
                {
                    "position": [
                        -0.25 + index * spacing,
                        index * lateral_step_m,
                        0.031,
                    ],
                    "actuators": {
                        "tilt": {
                            "lower_limit_rad": -math.pi / 2,
                            "upper_limit_rad": math.pi / 2,
                        }
                    },
                },
            )
            for index in range(8)
        ),
        global_attributes={
            "course": _course(near, far),
            "module_geometry": {"wheel_radius_m": 0.03106},
        },
    )


def _state_at(program, phase: str) -> dict[str, float]:
    state = {f"m{index}": 0.0 for index in range(8)}
    for step in program:
        for target in step.posture_targets:
            state[target.module_id] = target.angle_rad
        if step.phase == phase:
            return state
    raise AssertionError(f"Missing phase {phase}")


def test_flat_gap_recognizes_consistent_world_landmarks() -> None:
    gap = FlatGap.from_course(_course())

    assert gap.near_edge_x_m == pytest.approx(0.55)
    assert gap.far_edge_x_m == pytest.approx(0.75)
    assert gap.width_m == pytest.approx(0.20)


def test_gap_program_has_requested_geometric_sequence_without_timers() -> None:
    program = SnakeGapGaitPlanner().plan(_graph(), _assignments(), {})

    assert tuple(step.phase for step in program) == (
        "RESTORE_GAP_NEUTRAL",
        "APPROACH_NEAR_EDGE",
        "LIFT_HEAD",
        "EXTEND_HEAD_OVER_GAP",
        "LOWER_HEAD_ON_FAR_BANK",
        "ADVANCE_BODY",
        "LIFT_TAIL",
        "PULL_TAIL_OVER_GAP",
        "LOWER_TAIL_ON_FAR_BANK",
        "CLEAR_FAR_EDGE",
    )
    drives = tuple(step for step in program if step.kind == "drive")
    assert all(step.duration_s is None for step in program)
    assert all(step.position_goal is not None for step in drives)
    assert tuple(step.phase for step in drives) == (
        "APPROACH_NEAR_EDGE",
        "EXTEND_HEAD_OVER_GAP",
        "ADVANCE_BODY",
        "PULL_TAIL_OVER_GAP",
        "CLEAR_FAR_EDGE",
    )


def test_head_lands_before_body_advances_and_tail_is_lifted() -> None:
    program = SnakeGapGaitPlanner().plan(_graph(), _assignments(), {})
    head = _state_at(program, "LIFT_HEAD")
    landed = _state_at(program, "LOWER_HEAD_ON_FAR_BANK")
    tail = _state_at(program, "LIFT_TAIL")
    lowered = _state_at(program, "LOWER_TAIL_ON_FAR_BANK")

    assert head["m4"] > 0.0
    assert head["m6"] == pytest.approx(-head["m4"])
    assert all(value == pytest.approx(0.0) for value in landed.values())
    assert tail["m0"] < 0.0
    assert tail["m2"] == pytest.approx(-tail["m0"])
    assert all(value == pytest.approx(0.0) for value in lowered.values())

    extend = next(step for step in program if step.phase == "EXTEND_HEAD_OVER_GAP")
    advance = next(step for step in program if step.phase == "ADVANCE_BODY")
    pull = next(step for step in program if step.phase == "PULL_TAIL_OVER_GAP")
    assert extend.active_target_roles == ROLES[:5]
    assert advance.active_target_roles == ROLES
    assert pull.active_target_roles == ROLES[3:]
    assert extend.position_goal.module_id == "m7"
    assert advance.position_goal.module_id == "m4"
    assert pull.position_goal.module_id == "m0"


@pytest.mark.parametrize(
    "parameters",
    (
        {"duration_s": 1.0},
        {"span_duration_s": 1.0},
        {"tail_clear_duration_s": 1.0},
    ),
)
def test_gap_program_rejects_timed_parameters(parameters: dict) -> None:
    with pytest.raises(SnakeGapGaitError, match="timed parameters"):
        SnakeGapGaitPlanner().plan(_graph(), _assignments(), parameters)


def test_gap_program_rejects_excessive_width_and_misalignment() -> None:
    with pytest.raises(SnakeGapGaitError, match="exceeds"):
        SnakeGapGaitPlanner().plan(
            _graph(far=0.90),
            _assignments(),
            {},
        )
    with pytest.raises(SnakeGapGaitError, match="not aligned"):
        SnakeGapGaitPlanner().plan(
            _graph(lateral_step_m=0.05),
            _assignments(),
            {},
        )


def test_gap_program_requires_live_course_and_geometry_metadata() -> None:
    graph = _graph()
    without_course = AttributedRobotGraph(
        nodes=graph.nodes,
        global_attributes={"module_geometry": {"wheel_radius_m": 0.03106}},
    )
    with pytest.raises(SnakeGapGaitError, match="no course"):
        SnakeGapGaitPlanner().plan(without_course, _assignments(), {})
