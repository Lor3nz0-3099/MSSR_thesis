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


def _center_heights_from_tilts(
    state: dict[str, float], spacing: float
) -> tuple[float, ...]:
    heights = [0.0]
    link_angle = 0.0
    for index in range(7):
        link_angle += state[f"m{index}"]
        heights.append(heights[-1] + spacing * math.sin(link_angle))
    return tuple(heights)


def test_flat_gap_recognizes_consistent_world_landmarks() -> None:
    gap = FlatGap.from_course(_course())

    assert gap.near_edge_x_m == pytest.approx(0.55)
    assert gap.far_edge_x_m == pytest.approx(0.75)
    assert gap.width_m == pytest.approx(0.20)


def test_gap_program_has_requested_geometric_sequence_without_timers() -> None:
    program = SnakeGapGaitPlanner().plan(_graph(), _assignments(), {})

    phases = tuple(step.phase for step in program)
    assert phases[:5] == (
        "RESTORE_GAP_NEUTRAL",
        "PRELIFT_HEAD_DRAWBRIDGE",
        "LIFT_HEAD_DRAWBRIDGE",
        "STRAIGHTEN_HEAD_DRAWBRIDGE",
        "ADVANCE_HEAD_PIVOT_TO_EDGE",
    )
    profile_postures = tuple(
        phase for phase in phases if phase.startswith("CONFORM_GAP_PROFILE_")
    )
    profile_drives = tuple(
        phase for phase in phases if phase.startswith("FOLLOW_GAP_PROFILE_")
    )
    assert profile_postures[0] == "CONFORM_GAP_PROFILE_00"
    assert len(profile_postures) == len(profile_drives) + 1
    assert phases[-4:] == (
        "LIFT_TAIL_DRAWBRIDGE",
        "PULL_TAIL_TO_SAFE_LANDING",
        "LOWER_TAIL_ON_FAR_BANK",
        "CLEAR_FAR_EDGE",
    )
    drives = tuple(step for step in program if step.kind == "drive")
    assert all(step.duration_s is None for step in program)
    assert all(step.position_goal is not None for step in drives)
    assert drives[0].phase == "ADVANCE_HEAD_PIVOT_TO_EDGE"
    assert all(
        step.phase.startswith("FOLLOW_GAP_PROFILE_")
        for step in drives[1:-2]
    )
    assert tuple(step.phase for step in drives[-2:]) == (
        "PULL_TAIL_TO_SAFE_LANDING",
        "CLEAR_FAR_EDGE",
    )


def test_head_lands_before_body_advances_and_tail_is_lifted() -> None:
    program = SnakeGapGaitPlanner().plan(_graph(), _assignments(), {})
    prelift = _state_at(program, "PRELIFT_HEAD_DRAWBRIDGE")
    head = _state_at(program, "LIFT_HEAD_DRAWBRIDGE")
    straight = _state_at(program, "STRAIGHTEN_HEAD_DRAWBRIDGE")
    profile_phases = tuple(
        step.phase
        for step in program
        if step.phase.startswith("CONFORM_GAP_PROFILE_")
    )
    landed = _state_at(program, profile_phases[0])
    final_arch = _state_at(program, profile_phases[-1])
    tail = _state_at(program, "LIFT_TAIL_DRAWBRIDGE")
    lowered = _state_at(program, "LOWER_TAIL_ON_FAR_BANK")

    # A 200 mm gap plus two wheel supports requires four raised modules.
    # v4 first selects the head side of the symmetric chain.  v3 then becomes
    # the positive pivot and v4 is restored, leaving one real drawbridge hinge.
    assert prelift["m4"] == pytest.approx(0.45)
    assert head["m3"] == pytest.approx(1.20)
    assert head["m4"] == pytest.approx(0.45)
    assert straight["m3"] == pytest.approx(1.20)
    assert sum(abs(value) > 1e-9 for value in straight.values()) == 1
    assert max(_center_heights_from_tilts(landed, 0.07777)) > 0.02
    assert min(_center_heights_from_tilts(landed, 0.07777)) >= -1e-9
    assert max(_center_heights_from_tilts(final_arch, 0.07777)) > 0.02
    assert min(_center_heights_from_tilts(final_arch, 0.07777)) >= -1e-9
    assert tail["m3"] == pytest.approx(-1.20)
    assert sum(abs(value) > 1e-9 for value in tail.values()) == 1
    assert all(value == pytest.approx(0.0) for value in lowered.values())

    approach = next(
        step for step in program if step.phase == "ADVANCE_HEAD_PIVOT_TO_EDGE"
    )
    profile_drives = tuple(
        step
        for step in program
        if step.phase.startswith("FOLLOW_GAP_PROFILE_")
    )
    advance = profile_drives[-1]
    pull = next(
        step for step in program if step.phase == "PULL_TAIL_TO_SAFE_LANDING"
    )
    assert approach.active_target_roles == ROLES[:4]
    assert advance.active_target_roles == ROLES
    assert pull.active_target_roles == ROLES[4:]
    assert approach.position_goal.module_id == "m3"
    assert advance.position_goal.module_id == "m4"
    assert advance.position_goal.target_x_m == pytest.approx(
        0.75 + 0.03106 + 0.006
    )
    assert pull.position_goal.module_id == "m4"
    assert pull.position_goal.target_x_m == pytest.approx(
        0.75 + 4 * 0.07777 + 0.03106 + 0.006
    )


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


def test_wider_gap_migrates_fold_to_geometry_selected_pivot() -> None:
    program = SnakeGapGaitPlanner().plan(
        _graph(near=0.55, far=0.82),
        _assignments(),
        {},
    )

    phases = tuple(step.phase for step in program)
    assert phases[:7] == (
        "RESTORE_GAP_NEUTRAL",
        "PRELIFT_HEAD_DRAWBRIDGE",
        "MIGRATE_HEAD_DRAWBRIDGE_V3",
        "RELEASE_HEAD_DRAWBRIDGE_V4",
        "LIFT_HEAD_DRAWBRIDGE",
        "STRAIGHTEN_HEAD_DRAWBRIDGE",
        "ADVANCE_HEAD_PIVOT_TO_EDGE",
    )
    final_fold = _state_at(program, "STRAIGHTEN_HEAD_DRAWBRIDGE")
    assert final_fold["m2"] == pytest.approx(1.20)
    assert sum(abs(value) > 1e-9 for value in final_fold.values()) == 1
    approach = next(
        step for step in program if step.phase == "ADVANCE_HEAD_PIVOT_TO_EDGE"
    )
    assert approach.position_goal.module_id == "m2"
    assert approach.active_target_roles == ROLES[:3]


def test_drawbridge_lift_cannot_degenerate_into_a_flat_arch() -> None:
    with pytest.raises(SnakeGapGaitError, match="at least 0.85"):
        SnakeGapGaitPlanner().plan(
            _graph(),
            _assignments(),
            {"drawbridge_lift_angle_rad": 0.11},
        )


def test_landing_arch_clearance_is_geometry_derived_and_bounded() -> None:
    program = SnakeGapGaitPlanner().plan(_graph(), _assignments(), {})
    landed = _state_at(program, "CONFORM_GAP_PROFILE_00")
    spacing = 0.07777
    expected_clearance = 0.03106 + 0.006
    pivot_x = 0.55 - 0.03106 - 0.006
    far_support_x = 0.75 + 0.03106 + 0.006
    nominal_x = tuple(pivot_x + (index - 3) * spacing for index in range(8))
    expected_heights = tuple(
        (
            expected_clearance
            * math.sin(
                math.pi
                * (x_m - pivot_x)
                / (far_support_x - pivot_x)
            )
            if pivot_x < x_m < far_support_x
            else 0.0
        )
        for x_m in nominal_x
    )
    reconstructed = _center_heights_from_tilts(landed, spacing)

    assert reconstructed == pytest.approx(expected_heights)
    assert min(reconstructed) >= -1e-9
    assert max(reconstructed) > 0.0

    profile_phases = tuple(
        step.phase
        for step in program
        if step.phase.startswith("CONFORM_GAP_PROFILE_")
    )
    peak_indices: list[int] = []
    for phase in profile_phases:
        heights = _center_heights_from_tilts(
            _state_at(program, phase), spacing
        )
        assert min(heights) >= -1e-9
        assert max(heights) <= expected_clearance + 1e-9
        peak_indices.append(max(range(8), key=heights.__getitem__))
    assert peak_indices == sorted(peak_indices, reverse=True)
    assert peak_indices[0] > peak_indices[-1]

    with pytest.raises(SnakeGapGaitError, match="landing_arch_clearance_m"):
        SnakeGapGaitPlanner().plan(
            _graph(),
            _assignments(),
            {"landing_arch_clearance_m": 0.001},
        )
    with pytest.raises(SnakeGapGaitError, match="gap_profile_substeps"):
        SnakeGapGaitPlanner().plan(
            _graph(),
            _assignments(),
            {"gap_profile_substeps": 2.5},
        )


def test_top_bottom_chain_raises_head_before_tail() -> None:
    program = SnakeGapGaitPlanner().plan(_graph(), _assignments(), {})
    head_step = next(
        step for step in program if step.phase == "LIFT_HEAD_DRAWBRIDGE"
    )
    straight_step = next(
        step for step in program if step.phase == "STRAIGHTEN_HEAD_DRAWBRIDGE"
    )
    tail_step = next(
        step for step in program if step.phase == "LIFT_TAIL_DRAWBRIDGE"
    )

    assert len(head_step.posture_targets) == 1
    assert head_step.posture_targets[0].target_role == "snake_center_rear"
    assert head_step.posture_targets[0].angle_rad > 0.0
    assert len(straight_step.posture_targets) == 1
    assert straight_step.posture_targets[0].target_role == "snake_center_front"
    assert straight_step.posture_targets[0].angle_rad == pytest.approx(0.0)
    tail_pivot = next(
        target
        for target in tail_step.posture_targets
        if target.target_role == "snake_center_rear"
    )
    assert tail_pivot.angle_rad < 0.0
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
