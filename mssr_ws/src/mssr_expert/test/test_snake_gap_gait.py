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
    assert phases[:4] == (
        "RESTORE_GAP_NEUTRAL",
        "APPROACH_HEAD_TO_NEAR_EDGE",
        "CONFORM_GAP_PROFILE_01",
        "FOLLOW_GAP_PROFILE_01",
    )
    profile_postures = tuple(
        phase for phase in phases if phase.startswith("CONFORM_GAP_PROFILE_")
    )
    profile_drives = tuple(
        phase for phase in phases if phase.startswith("FOLLOW_GAP_PROFILE_")
    )
    assert profile_postures[0] == "CONFORM_GAP_PROFILE_01"
    assert len(profile_postures) == len(profile_drives)
    assert phases[-1] == "RESTORE_GAP_NEUTRAL_FINAL"
    drives = tuple(step for step in program if step.kind == "drive")
    assert all(step.duration_s is None for step in program)
    assert all(step.position_goal is not None for step in drives)
    assert drives[0].phase == "APPROACH_HEAD_TO_NEAR_EDGE"
    assert all(
        step.phase.startswith("FOLLOW_GAP_PROFILE_")
        for step in drives[1:]
    )


def test_backbone_wave_approaches_flat_then_carries_head_to_tail() -> None:
    program = SnakeGapGaitPlanner().plan(_graph(), _assignments(), {})
    approach = next(
        step for step in program if step.phase == "APPROACH_HEAD_TO_NEAR_EDGE"
    )
    profile_drives = tuple(
        step
        for step in program
        if step.phase.startswith("FOLLOW_GAP_PROFILE_")
    )
    final_neutral = _state_at(program, "RESTORE_GAP_NEUTRAL_FINAL")

    near_support_x = 0.55 - 0.03106 - 0.006
    far_support_x = 0.75 + 0.03106 + 0.006
    assert approach.active_target_roles == ROLES
    assert approach.position_goal.module_id == "m7"
    assert approach.position_goal.target_x_m == pytest.approx(near_support_x)
    assert all(step.active_target_roles == ROLES for step in profile_drives)
    assert all(step.position_goal.module_id == "m7" for step in profile_drives)
    assert profile_drives[-1].position_goal.target_x_m == pytest.approx(
        far_support_x + 8 * 0.07777
    )
    assert all(value == pytest.approx(0.0) for value in final_neutral.values())
    assert not any(
        forbidden in step.phase
        for step in program
        for forbidden in ("DRAWBRIDGE", "PULL_TAIL", "LOWER_TAIL")
    )


def test_gap_width_scales_wave_travel_and_profile_count() -> None:
    narrow = SnakeGapGaitPlanner().plan(
        _graph(near=0.55, far=0.65),
        _assignments(),
        {},
    )
    nominal = SnakeGapGaitPlanner().plan(_graph(), _assignments(), {})
    narrow_drives = tuple(
        step
        for step in narrow
        if step.phase.startswith("FOLLOW_GAP_PROFILE_")
    )
    nominal_drives = tuple(
        step
        for step in nominal
        if step.phase.startswith("FOLLOW_GAP_PROFILE_")
    )
    assert len(nominal_drives) > len(narrow_drives)
    assert (
        nominal_drives[-1].position_goal.target_x_m
        - narrow_drives[-1].position_goal.target_x_m
    ) == pytest.approx(0.10)


def test_wider_admissible_gap_still_uses_only_the_backbone_wave() -> None:
    wider = SnakeGapGaitPlanner().plan(
        _graph(near=0.55, far=0.78),
        _assignments(),
        {},
    )
    nominal = SnakeGapGaitPlanner().plan(_graph(), _assignments(), {})
    wider_profiles = tuple(
        step
        for step in wider
        if step.phase.startswith("CONFORM_GAP_PROFILE_")
    )
    nominal_profiles = tuple(
        step
        for step in nominal
        if step.phase.startswith("CONFORM_GAP_PROFILE_")
    )
    assert len(wider_profiles) > len(nominal_profiles)
    assert all("DRAWBRIDGE" not in step.phase for step in wider)


@pytest.mark.parametrize(
    "parameter",
    (
        "drawbridge_lift_angle_rad",
        "drawbridge_prelift_angle_rad",
        "drawbridge_bias_linear_m_s",
    ),
)
def test_removed_drawbridge_parameters_are_rejected(parameter: str) -> None:
    with pytest.raises(
        SnakeGapGaitError,
        match="vertical drawbridge gait was removed",
    ):
        SnakeGapGaitPlanner().plan(
            _graph(),
            _assignments(),
            {parameter: 0.11},
        )


def test_landing_arch_clearance_is_geometry_derived_and_bounded() -> None:
    program = SnakeGapGaitPlanner().plan(_graph(), _assignments(), {})
    spacing = 0.07777
    expected_clearance = 0.03106 + 0.006
    near_support_x = 0.55 - 0.03106 - 0.006
    far_support_x = 0.75 + 0.03106 + 0.006
    far_arch_x = far_support_x + spacing
    initial_x = tuple(
        near_support_x - (7 - index) * spacing for index in range(8)
    )
    body_travel = far_arch_x - initial_x[0]
    step_count = math.ceil(body_travel / (spacing / 3))
    first_x = tuple(x_m + body_travel / step_count for x_m in initial_x)
    expected_heights = SnakeGapGaitPlanner._traveling_arch_heights(
        first_x,
        near_support_x,
        far_arch_x,
        expected_clearance,
    )
    landed = _state_at(program, "CONFORM_GAP_PROFILE_01")
    reconstructed = _center_heights_from_tilts(landed, spacing)

    assert reconstructed == pytest.approx(
        tuple(height - expected_heights[0] for height in expected_heights)
    )
    assert min(expected_heights) >= -1e-9
    assert max(expected_heights) > 0.0

    profile_phases = tuple(
        step.phase
        for step in program
        if step.phase.startswith("CONFORM_GAP_PROFILE_")
    )
    peak_indices: list[int] = []
    for profile_index, phase in enumerate(profile_phases, start=1):
        translation = profile_index * body_travel / step_count
        nominal_x = tuple(x_m + translation for x_m in initial_x)
        heights = SnakeGapGaitPlanner._traveling_arch_heights(
            nominal_x,
            near_support_x,
            far_arch_x,
            expected_clearance,
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
    with pytest.raises(SnakeGapGaitError, match="far_bank_transition_links"):
        SnakeGapGaitPlanner().plan(
            _graph(),
            _assignments(),
            {"far_bank_transition_links": 0.25},
        )


def test_far_bank_transition_keeps_the_head_high_past_the_edge() -> None:
    spacing = 0.07777
    wheel_radius = 0.03106
    near_support_x = 0.55 - wheel_radius - 0.006
    far_support_x = 0.75 + wheel_radius + 0.006
    far_arch_x = far_support_x + spacing

    far_edge_height = SnakeGapGaitPlanner._traveling_arch_heights(
        (0.75,),
        near_support_x,
        far_arch_x,
        wheel_radius + 0.006,
    )[0]
    old_far_edge_height = SnakeGapGaitPlanner._traveling_arch_heights(
        (0.75,),
        near_support_x,
        far_support_x,
        wheel_radius + 0.006,
    )[0]

    assert far_edge_height > wheel_radius
    assert far_edge_height > old_far_edge_height


def test_wave_is_low_bidirectional_and_migrates_from_head_to_tail() -> None:
    program = SnakeGapGaitPlanner().plan(_graph(), _assignments(), {})
    profile_states = tuple(
        _state_at(program, step.phase)
        for step in program
        if step.phase.startswith("CONFORM_GAP_PROFILE_")
    )

    assert max(
        abs(value) for state in profile_states for value in state.values()
    ) < 0.75
    assert any(
        any(value > 1e-6 for value in state.values())
        and any(value < -1e-6 for value in state.values())
        for state in profile_states
    )


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
