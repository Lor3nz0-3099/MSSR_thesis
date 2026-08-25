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
        "LIFT_HEAD_DRAWBRIDGE",
        "ADVANCE_HEAD_PIVOT_TO_EDGE",
        "LOWER_HEAD_ACROSS_GAP",
        "ADVANCE_BODY_TO_FAR_SUPPORT",
        "LIFT_TAIL_DRAWBRIDGE",
        "PULL_TAIL_TO_SAFE_LANDING",
        "LOWER_TAIL_ON_FAR_BANK",
        "CLEAR_FAR_EDGE",
    )
    drives = tuple(step for step in program if step.kind == "drive")
    assert all(step.duration_s is None for step in program)
    assert all(step.position_goal is not None for step in drives)
    assert tuple(step.phase for step in drives) == (
        "ADVANCE_HEAD_PIVOT_TO_EDGE",
        "ADVANCE_BODY_TO_FAR_SUPPORT",
        "PULL_TAIL_TO_SAFE_LANDING",
        "CLEAR_FAR_EDGE",
    )


def test_head_lands_before_body_advances_and_tail_is_lifted() -> None:
    program = SnakeGapGaitPlanner().plan(_graph(), _assignments(), {})
    head = _state_at(program, "LIFT_HEAD_DRAWBRIDGE")
    landed = _state_at(program, "LOWER_HEAD_ACROSS_GAP")
    tail = _state_at(program, "LIFT_TAIL_DRAWBRIDGE")
    lowered = _state_at(program, "LOWER_TAIL_ON_FAR_BANK")

    # A 200 mm gap plus two wheel supports requires four raised modules.
    # The single v3 hinge makes a real drawbridge; there is no cancelling bend.
    assert head["m3"] == pytest.approx(1.20)
    assert sum(abs(value) > 1e-9 for value in head.values()) == 1
    assert all(value == pytest.approx(0.0) for value in landed.values())
    assert tail["m3"] == pytest.approx(-1.20)
    assert sum(abs(value) > 1e-9 for value in tail.values()) == 1
    assert all(value == pytest.approx(0.0) for value in lowered.values())

    approach = next(
        step for step in program if step.phase == "ADVANCE_HEAD_PIVOT_TO_EDGE"
    )
    advance = next(
        step for step in program if step.phase == "ADVANCE_BODY_TO_FAR_SUPPORT"
    )
    pull = next(
        step for step in program if step.phase == "PULL_TAIL_TO_SAFE_LANDING"
    )
    assert approach.active_target_roles == ROLES[:4]
    assert advance.active_target_roles == ROLES
    assert pull.active_target_roles == ROLES[4:]
    assert approach.position_goal.module_id == "m3"
    assert advance.position_goal.module_id == "m4"
    assert pull.position_goal.module_id == "m4"
    assert pull.position_goal.target_x_m == pytest.approx(
        0.75 + 4 * 0.07777 + 0.03106 + 0.006
    )
    head_target = next(
        step for step in program if step.phase == "LIFT_HEAD_DRAWBRIDGE"
    ).posture_targets[0]
    tail_target = next(
        step for step in program if step.phase == "LIFT_TAIL_DRAWBRIDGE"
    ).posture_targets[0]
    assert head_target.pusher_module_id == "m2"
    assert head_target.pusher_linear_m_s == pytest.approx(0.020)
    assert tail_target.pusher_module_id == "m4"
    assert tail_target.pusher_linear_m_s == pytest.approx(-0.020)


def test_drawbridge_module_count_scales_with_gap_width() -> None:
    narrow = SnakeGapGaitPlanner().plan(
        _graph(near=0.55, far=0.65),
        _assignments(),
        {},
    )
    head = _state_at(narrow, "LIFT_HEAD_DRAWBRIDGE")
    tail = _state_at(narrow, "LIFT_TAIL_DRAWBRIDGE")
    approach = next(
        step for step in narrow if step.phase == "ADVANCE_HEAD_PIVOT_TO_EDGE"
    )

    # 100 mm gap plus wheel support margins requires three terminal modules.
    assert head["m4"] == pytest.approx(1.20)
    assert tail["m2"] == pytest.approx(-1.20)
    assert approach.position_goal.module_id == "m4"
    assert approach.active_target_roles == ROLES[:5]


def test_drawbridge_lift_cannot_degenerate_into_a_flat_arch() -> None:
    with pytest.raises(SnakeGapGaitError, match="at least 0.85"):
        SnakeGapGaitPlanner().plan(
            _graph(),
            _assignments(),
            {"drawbridge_lift_angle_rad": 0.11},
        )


def test_top_bottom_chain_raises_head_before_tail() -> None:
    program = SnakeGapGaitPlanner().plan(_graph(), _assignments(), {})
    head_step = next(
        step for step in program if step.phase == "LIFT_HEAD_DRAWBRIDGE"
    )
    tail_step = next(
        step for step in program if step.phase == "LIFT_TAIL_DRAWBRIDGE"
    )

    assert len(head_step.posture_targets) == 1
    assert head_step.posture_targets[0].target_role == "snake_center_rear"
    assert head_step.posture_targets[0].angle_rad > 0.0
    assert len(tail_step.posture_targets) == 1
    assert tail_step.posture_targets[0].target_role == "snake_center_rear"
    assert tail_step.posture_targets[0].angle_rad < 0.0
    assert program.index(head_step) < program.index(tail_step)


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
