"""Tests for the deterministic Snake8 staircase follower."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import pytest

from mssr_expert.behaviors.morphology_library import (
    AssignedModule,
    BehaviorProgramStep,
    JointTarget,
    LongitudinalDisplacementGoal,
    LongitudinalPositionGoal,
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


def _course(
    *,
    rise_m: float = 0.065,
    tread_depth_m: float = 0.28,
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
        "course_profile": "snake8_stair_test",
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
    *,
    lateral_step_m: float = 0.0,
    rise_m: float = 0.065,
    tread_depth_m: float = 0.28,
    stair_count: int = 3,
    neutral_tilts: Mapping[str, float] | None = None,
) -> AttributedRobotGraph:
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
                    ],
                    **(
                        {
                            "actuators": {
                                "tilt": {
                                    "position_rad": neutral_tilts.get(
                                        f"m{index}", 0.0
                                    ),
                                    "lower_limit_rad": -math.pi / 2.0,
                                    "upper_limit_rad": math.pi / 2.0,
                                }
                            }
                        }
                        if neutral_tilts is not None
                        else {}
                    ),
                },
            )
            for index in range(8)
        ),
        global_attributes={
            "course": _course(
                rise_m=rise_m,
                tread_depth_m=tread_depth_m,
                stair_count=stair_count,
            ),
            "module_geometry": {
                "wheel_radius_m": 0.03106,
                "forward_collision_extent_m": 0.043771,
            },
        },
    )


def _posture_state_at(
    program: tuple[BehaviorProgramStep, ...],
    phase: str,
) -> dict[str, float]:
    state = {f"m{index}": 0.0 for index in range(8)}
    for step in program:
        for target in step.posture_targets:
            state[target.module_id] = target.angle_rad
        if step.phase == phase:
            return state
    raise AssertionError(f"Missing program phase {phase}")


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


def test_next_riser_lifts_head_then_migrates_hook_after_contact() -> None:
    program = SnakeStairGaitPlanner().plan(
        _graph(),
        _assignments(),
        {"profile_substeps": 6, "transition_clearance_m": 0.0065},
    )
    before_targets = _posture_state_at(program, "PROFILE_00_06")
    start_targets = _posture_state_at(program, "PROFILE_01_01")
    lifted_targets = _posture_state_at(program, "PROFILE_01_04")
    hook_targets = _posture_state_at(program, "PROFILE_01_05")
    merged_targets = _posture_state_at(program, "PROFILE_02_06")
    angle = math.asin(0.065 / 0.07777)
    clearance_angle = math.asin((0.065 + 0.010) / 0.07777)

    assert before_targets["m7"] == pytest.approx(0.0)
    assert start_targets["m6"] > 0.0
    assert start_targets["m7"] < 0.0
    assert lifted_targets["m6"] == pytest.approx(clearance_angle)
    assert lifted_targets["m7"] == pytest.approx(-clearance_angle)
    assert hook_targets["m5"] > 0.0
    assert hook_targets["m7"] > -clearance_angle
    assert merged_targets["m5"] == pytest.approx(angle)
    assert merged_targets["m6"] == pytest.approx(-angle)
    assert merged_targets["m7"] == pytest.approx(0.0)

    support_drive = next(
        step for step in program if step.phase == "CRAWL_00_06"
    )
    assert support_drive.position_goal is not None
    assert support_drive.position_goal.module_id == "m5"
    assert (
        support_drive.position_goal.target_x_m
        - support_drive.position_goal.tolerance_m
        > 0.65 + 0.03106
    )


def test_head_overstep_cycle_repeats_for_third_riser() -> None:
    program = SnakeStairGaitPlanner().plan(
        _graph(),
        _assignments(),
        {"profile_substeps": 6, "transition_clearance_m": 0.0065},
    )
    before_targets = _posture_state_at(program, "PROFILE_04_06")
    start_targets = _posture_state_at(program, "PROFILE_05_01")
    lifted_targets = _posture_state_at(program, "PROFILE_05_04")
    hook_targets = _posture_state_at(program, "PROFILE_05_05")
    clearance_angle = math.asin((0.065 + 0.010) / 0.07777)

    assert before_targets["m5"] == pytest.approx(0.0)
    assert before_targets["m6"] == pytest.approx(0.0)
    assert before_targets["m7"] == pytest.approx(0.0)
    assert start_targets["m6"] > 0.0
    assert start_targets["m7"] < 0.0
    assert lifted_targets["m6"] == pytest.approx(clearance_angle)
    assert lifted_targets["m7"] == pytest.approx(-clearance_angle)
    assert hook_targets["m5"] > 0.0
    assert hook_targets["m7"] > -clearance_angle

    support_drive = next(
        step for step in program if step.phase == "CRAWL_04_06"
    )
    assert support_drive.position_goal is not None
    assert support_drive.position_goal.module_id == "m5"
    assert (
        support_drive.position_goal.target_x_m
        - support_drive.position_goal.tolerance_m
        > 0.93 + 0.03106
    )


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
        "snake_hip",
        "snake_center_rear",
        "snake_center_front",
        "snake_shoulder",
        "snake_neck",
        "snake_head",
    )
    assert crawl.duration_s is None
    assert crawl.displacement_goal is None
    assert crawl.position_goal is not None
    assert crawl.position_goal.module_id == "m7"
    expected_clearance = 0.10 * 0.065
    assert crawl.position_goal.target_x_m == pytest.approx(
        0.93
        - 0.03106
        + 0.5 * 0.07777
        - expected_clearance
    )
    assert crawl.position_goal.tolerance_m == pytest.approx(0.004)
    upper_deck = program[-1]
    assert upper_deck.duration_s is None
    assert upper_deck.displacement_goal is not None
    assert upper_deck.displacement_goal.module_ids == tuple(
        f"m{index}" for index in range(8)
    )
    assert upper_deck.displacement_goal.distance_m == pytest.approx(0.07777)


def test_first_crawl_also_drives_wheels_transitioning_over_riser() -> None:
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
        "snake_center_front",
        "snake_shoulder",
        "snake_neck",
        "snake_head",
    )
    assert first_crawl.position_goal is not None
    assert first_crawl.position_goal.module_id == "m5"


def test_next_riser_crawl_keeps_every_wheel_commanded() -> None:
    program = SnakeStairGaitPlanner().plan(
        _graph(),
        _assignments(),
        {"profile_substeps": 6},
    )

    crawl = next(step for step in program if step.phase == "CRAWL_01_04")

    assert crawl.active_target_roles == tuple(
        assignment.target_role for assignment in _assignments()
    )


def test_transition_lead_straightens_front_and_moves_bend_rearward() -> None:
    planner = SnakeStairGaitPlanner()
    program = planner.plan(
        _graph(),
        _assignments(),
        {"profile_substeps": 2, "transition_clearance_m": 0.008},
    )
    angle = math.asin(0.065 / 0.07777)
    middle = next(step for step in program if step.phase == "PROFILE_00_01")
    middle_targets = {
        target.module_id: target.angle_rad
        for target in middle.posture_targets
    }
    endpoint = next(step for step in program if step.phase == "PROFILE_00_02")
    endpoint_targets = {
        target.module_id: target.angle_rad
        for target in endpoint.posture_targets
    }

    assert middle_targets["m3"] > 0.5 * angle
    assert middle_targets["m4"] < 0.0
    assert middle_targets["m5"] > -0.5 * angle
    assert endpoint_targets["m3"] == pytest.approx(angle)
    assert endpoint_targets["m4"] == pytest.approx(-angle)
    assert endpoint_targets["m5"] == pytest.approx(0.0)
    assert "m6" not in endpoint_targets
    assert "m7" not in endpoint_targets


def test_upper_riser_release_preserves_first_wave_and_repeats_above() -> None:
    planner = SnakeStairGaitPlanner()
    baseline = planner.plan(
        _graph(),
        _assignments(),
        {
            "profile_substeps": 6,
            "transition_clearance_m": 0.0065,
            "upper_riser_edge_release_lead_m": 0.0,
        },
    )
    released = planner.plan(
        _graph(),
        _assignments(),
        {
            "profile_substeps": 6,
            "transition_clearance_m": 0.0065,
        },
    )

    # Every posture in the first-riser wave remains exactly as it was before
    # the upper-riser correction was introduced.
    for substep in range(1, 7):
        phase = f"PROFILE_00_{substep:02d}"
        baseline_state = _posture_state_at(baseline, phase)
        released_state = _posture_state_at(released, phase)
        assert released_state == pytest.approx(baseline_state)

    # The local release starts on the second riser and repeats on the third.
    for phase in ("PROFILE_04_03", "PROFILE_08_03"):
        baseline_state = _posture_state_at(baseline, phase)
        released_state = _posture_state_at(released, phase)
        assert released_state["m5"] == pytest.approx(0.0)
        assert abs(released_state["m5"]) < abs(baseline_state["m5"])
        assert released_state["m3"] == pytest.approx(
            baseline_state["m3"]
        )
        assert released_state["m4"] == pytest.approx(
            baseline_state["m4"]
        )

    endpoint = _posture_state_at(released, "PROFILE_04_06")
    baseline_endpoint = _posture_state_at(baseline, "PROFILE_04_06")
    assert endpoint == pytest.approx(baseline_endpoint)


def test_crawl_uses_world_edge_targets_with_temporary_lead() -> None:
    program = SnakeStairGaitPlanner().plan(
        _graph(),
        _assignments(),
        {"profile_substeps": 6, "transition_clearance_m": 0.0065},
    )
    fourth = next(step for step in program if step.phase == "CRAWL_00_04")
    endpoint = next(step for step in program if step.phase == "CRAWL_00_06")

    assert fourth.position_goal is not None
    assert fourth.position_goal.module_id == "m5"
    assert fourth.position_goal.target_x_m == pytest.approx(
        0.65
        - 0.03106
        + 4.0 * 0.07777 / 6.0
        - 0.0065 * math.sin(4.0 * math.pi / 6.0)
    )
    assert endpoint.position_goal is not None
    assert endpoint.position_goal.target_x_m == pytest.approx(
        0.65 - 0.03106 + 0.07777
    )


def test_default_profile_uses_six_progressive_microsteps() -> None:
    program = SnakeStairGaitPlanner().plan(_graph(), _assignments(), {})
    phases = {step.phase for step in program}

    assert "PROFILE_00_06" in phases
    assert "CRAWL_00_06" in phases


def test_arch_wave_keeps_the_flat_captured_neutral_approach() -> None:
    program = SnakeStairGaitPlanner().plan_arch_wave(
        _graph(),
        _assignments(),
        {"profile_substeps": 6},
    )
    approach = program[0]

    assert approach.phase == "GEOM_APPROACH_FIRST_RISER"
    assert approach.kind == "posture_drive"
    assert len(approach.posture_targets) == 8
    assert all(
        target.joint == "tilt"
        and target.angle_rad == pytest.approx(0.0)
        and target.angle_reference == "captured_neutral"
        for target in approach.posture_targets
    )
    assert approach.position_goal is not None
    assert approach.position_goal.module_id == "m7"


def test_first_riser_uses_a_broad_clearance_arch() -> None:
    spacing = 0.07777
    rise = 0.065
    wheel_radius = 0.03106
    clearance = 0.58 * wheel_radius
    program = SnakeStairGaitPlanner().plan_arch_wave(
        _graph(),
        _assignments(),
        {"profile_substeps": 6},
    )
    lift = next(
        step for step in program if step.phase == "APPROACH_FIRST_RISER"
    )
    targets = {target.module_id: target for target in lift.posture_targets}
    expected = math.asin((rise + clearance) / (2.0 * spacing))

    assert set(targets) == {"m3", "m5"}
    assert targets["m3"].angle_rad == pytest.approx(expected)
    assert targets["m5"].angle_rad == pytest.approx(-expected)
    assert 2.0 * spacing * math.sin(expected) == pytest.approx(
        rise + clearance
    )


def test_arch_rail_passes_the_broad_cell_one_module_toward_the_tail() -> None:
    program = SnakeStairGaitPlanner().plan_arch_wave(
        _graph(),
        _assignments(),
        {"profile_substeps": 6},
    )
    first = _posture_state_at(program, "ARCH_HEAD_GATE_01")
    second = _posture_state_at(program, "ARCH_DRIVE_01_06")
    third = _posture_state_at(program, "ARCH_DRIVE_02_06")

    assert first["m2"] > 0.0 and first["m4"] < 0.0
    assert second["m1"] > 0.0 and second["m3"] < 0.0
    assert third["m0"] > 0.0 and third["m2"] < 0.0
    assert first["m2"] == pytest.approx(second["m1"])
    assert second["m1"] == pytest.approx(third["m0"])


def test_arch_rail_moves_the_rigid_support_partition_with_the_cell() -> None:
    assignments = _assignments()
    program = SnakeStairGaitPlanner().plan_arch_wave(
        _graph(),
        assignments,
        {"profile_substeps": 6},
    )
    expected = {
        "APPROACH_FIRST_RISER": (
            {"m3", "m5"},
            {"m0", "m1", "m2", "m4", "m6", "m7"},
        ),
        "ARCH_DRIVE_00_03": (
            {"m2", "m3", "m4", "m5"},
            {"m0", "m1", "m6", "m7"},
        ),
        "ARCH_DRIVE_01_03": (
            {"m1", "m2", "m3", "m4", "m5", "m6", "m7"},
            {"m0"},
        ),
    }

    for phase, (moving_ids, support_ids) in expected.items():
        step = next(item for item in program if item.phase == phase)
        assert {target.module_id for target in step.posture_targets} == (
            moving_ids
        )
        executor = MorphologyBehaviorExecutor(
            MorphologyLibrary.load(
                Path(__file__).parents[1]
                / "config"
                / "smores_morphology_behaviors.json"
            )
        )
        executor.start(
            MorphologyCommand(f"partition-{phase}", "snake8", "crawl_stairs"),
            assignments,
            {item.module_id: 0.0 for item in assignments},
            (step,),
        )
        decision = executor.step(
            0.0,
            module_positions={
                item.module_id: (0.0, 0.0, 0.031)
                for item in assignments
            },
        )
        assert decision.primitive_goal is not None
        assert set(
            decision.primitive_goal.parameters[
                "structural_hold_module_ids"
            ]
        ) == support_ids


def test_arch_wave_only_overlays_concurrent_drive_on_validated_shape() -> None:
    program = SnakeStairGaitPlanner().plan_arch_wave(
        _graph(),
        _assignments(),
        {
            "profile_substeps": 6,
            "synchronized_linear_m_s": 0.020,
            "max_wave_tilt_speed_rad_s": 0.45,
        },
    )
    wave = [
        step
        for step in program
        if step.phase.startswith("ARCH_") and step.position_goal is not None
    ]

    assert wave
    assert all(step.kind == "posture_drive" for step in wave)
    assert all(step.linear_m_s == pytest.approx(0.020) for step in wave)
    assert all(
        step.posture_reached_linear_m_s == pytest.approx(0.040)
        for step in wave
    )
    assert all(not step.hold_locomotion_until_admitted for step in wave)
    assert all(step.continuous_with_next for step in wave[:-1])
    assert not wave[-1].continuous_with_next
    assert all(
        target.max_servo_speed_rad_s == pytest.approx(0.45)
        and target.tolerance_rad == pytest.approx(0.025)
        and target.angle_reference == "captured_neutral"
        for step in wave
        for target in step.posture_targets
    )


def test_tail_boundary_keeps_the_distributed_arch_and_finishes_flat() -> None:
    program = SnakeStairGaitPlanner().plan_arch_wave(
        _graph(),
        _assignments(),
        {"profile_substeps": 6},
    )
    final_step = program[-1]
    final_state = _posture_state_at(program, final_step.phase)

    assert final_step.phase == "ARCH_TAIL_LIFT_COMPLETE"
    assert final_step.kind == "posture_drive"
    assert final_state == pytest.approx(
        {f"m{index}": 0.0 for index in range(8)}
    )
    maximum_clearance_angle = math.asin(
        (0.065 + 0.58 * 0.03106) / (2.0 * 0.07777)
    )
    assert max(
        abs(target.angle_rad)
        for step in program
        for target in step.posture_targets
        if target.module_id in {"m0", "m1"}
    ) == pytest.approx(maximum_clearance_angle)


def test_loaded_tilt_tolerance_is_configurable_but_safely_bounded() -> None:
    program = SnakeStairGaitPlanner().plan_arch_wave(
        _graph(),
        _assignments(),
        {"loaded_tilt_tolerance_rad": 0.030},
    )
    moving_targets = tuple(
        target
        for step in program
        if step.kind == "posture_drive"
        for target in step.posture_targets
    )
    assert moving_targets
    assert all(
        target.tolerance_rad == pytest.approx(0.030)
        for target in moving_targets
    )

    with pytest.raises(
        SnakeStairGaitError,
        match="loaded_tilt_tolerance_rad",
    ):
        SnakeStairGaitPlanner().plan_arch_wave(
            _graph(),
            _assignments(),
            {"loaded_tilt_tolerance_rad": 0.050},
        )


@pytest.mark.parametrize(
    ("rise_m", "tread_depth_m", "stair_count"),
    (
        (0.050, 0.250, 2),
        (0.060, 0.272, 3),
        (0.065, 0.280, 3),
        (0.065, 0.320, 4),
    ),
)
def test_continuous_geometric_wave_scales_across_seed_geometries(
    rise_m: float,
    tread_depth_m: float,
    stair_count: int,
) -> None:
    program = SnakeStairGaitPlanner().plan_arch_wave(
        _graph(
            rise_m=rise_m,
            tread_depth_m=tread_depth_m,
            stair_count=stair_count,
        ),
        _assignments(),
        {"profile_substeps": 6},
    )
    final_step = program[-1]
    final_riser_x = 0.65 + (stair_count - 1) * tread_depth_m
    assert final_step.phase == "ARCH_TAIL_LIFT_COMPLETE"
    assert final_step.position_goal is not None
    assert final_step.position_goal.module_id == "m1"
    assert final_step.position_goal.target_x_m == pytest.approx(
        final_riser_x + 0.03106
    )
    assert all(
        abs(target.angle_rad) <= math.pi / 2.0 - 0.030 + 1e-9
        for step in program
        for target in step.posture_targets
    )
    state = {f"m{index}": 0.0 for index in range(8)}
    for step in program:
        for target in step.posture_targets:
            state[target.module_id] = target.angle_rad
    assert sum(state.values()) == pytest.approx(0.0, abs=1e-6)


def test_wave_targets_respect_limits_around_captured_neutral() -> None:
    neutrals = {
        f"m{index}": (-0.115 if index % 2 == 0 else 0.095)
        for index in range(8)
    }
    program = SnakeStairGaitPlanner().plan_arch_wave(
        _graph(neutral_tilts=neutrals),
        _assignments(),
        {"profile_substeps": 6},
        neutrals,
    )

    for step in program:
        for target in step.posture_targets:
            absolute = neutrals[target.module_id] + target.angle_rad
            assert absolute >= -math.pi / 2.0 + 0.030 - 1e-9
            assert absolute <= math.pi / 2.0 - 0.030 + 1e-9


def test_plan_rejects_a_snake_not_aligned_with_the_known_stairs() -> None:
    with pytest.raises(SnakeStairGaitError, match="not aligned"):
        SnakeStairGaitPlanner().plan(
            _graph(lateral_step_m=0.04),
            _assignments(),
            {},
        )


def test_plan_rejects_stair_landmarks_that_disagree_with_world_boxes() -> None:
    graph = _graph()
    course = graph.global_attributes["course"]
    course["collision_boxes"][0]["center_xyz_m"][0] += 0.010

    with pytest.raises(SnakeStairGaitError, match="collision boxes"):
        SnakeStairGaitPlanner().plan_arch_wave(
            graph,
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


def test_executor_starts_drive_only_after_ramped_tilt_is_admitted() -> None:
    assignments = _assignments()
    executor = MorphologyBehaviorExecutor(
        MorphologyLibrary.load(
            Path(__file__).parents[1]
            / "config"
            / "smores_morphology_behaviors.json"
        )
    )
    executor.start(
        MorphologyCommand("bridge-continuous", "snake8", "crawl_stairs"),
        assignments,
        {item.module_id: 0.0 for item in assignments},
        (
            BehaviorProgramStep(
                phase="BRIDGE_WAVE_TEST",
                posture_targets=(
                    JointTarget(
                        module_id="m5",
                        joint="tilt",
                        angle_rad=0.3,
                        target_vertex_id="v5",
                        target_role="snake_shoulder",
                        coordination_group="stair:test",
                        max_servo_speed_rad_s=0.4,
                        angle_reference="captured_neutral",
                    ),
                ),
                linear_m_s=0.011,
                active_target_roles=ROLES,
                position_goal=LongitudinalPositionGoal(
                    module_id="m7",
                    target_x_m=0.70,
                    tolerance_m=0.004,
                ),
                posture_reached_linear_m_s=0.025,
            ),
        ),
    )
    poses = {"m7": (0.60, 0.0, 0.031)}

    dispatched = executor.step(0.0, module_positions=poses)
    assert dispatched.primitive_goal is not None
    assert not dispatched.locomotion
    assert dispatched.primitive_goal.parameters[
        "max_servo_speed_rad_s"
    ] == pytest.approx(0.4)

    awaiting_admission = executor.step(0.1, module_positions=poses)
    assert awaiting_admission.state == "WAITING_JOINT_ADMISSION"
    assert not awaiting_admission.locomotion

    goal = dispatched.primitive_goal
    admitted = executor.step(
        0.2,
        {
            "schema_version": "mssr.primitive_status.v1",
            "goal_id": goal.goal_id,
            "primitive": goal.primitive,
            "module_ids": list(goal.module_ids),
            "state": "accepted",
            "phase": "accepted",
            "progress": 0.0,
            "code": "ACCEPTED",
            "message": "admitted",
        },
        poses,
    )
    assert set(admitted.locomotion) == {
        f"m{index}" for index in range(8)
    }

    posture_lagging = executor.step(
        0.25,
        {
            "schema_version": "mssr.primitive_status.v1",
            "goal_id": goal.goal_id,
            "primitive": goal.primitive,
            "module_ids": list(goal.module_ids),
            "state": "running",
            "phase": "tilt",
            "progress": 0.8,
            "code": "MOVING_JOINT",
            "message": "moving",
        },
        {"m7": (0.70, 0.0, 0.031)},
    )
    assert not posture_lagging.locomotion
    assert "position barrier reached" in posture_lagging.message

    settled_traction = executor.step(
        0.3,
        {
            "schema_version": "mssr.primitive_status.v1",
            "goal_id": goal.goal_id,
            "primitive": goal.primitive,
            "module_ids": list(goal.module_ids),
            "state": "succeeded",
            "phase": "terminal",
            "progress": 1.0,
            "code": "JOINT_TARGET_REACHED",
            "message": "done",
        },
        {"m7": (0.65, 0.0, 0.031)},
    )
    assert settled_traction.state == "RUNNING_PROGRAM_DRIVE"
    assert "settled traction speed 0.025m/s" in settled_traction.message
    assert all(
        command["vx"] == pytest.approx(0.025)
        for command in settled_traction.locomotion.values()
    )

    completed = executor.step(
        0.4,
        module_positions={"m7": (0.70, 0.0, 0.031)},
    )
    assert completed.phase == "BRIDGE_WAVE_TEST_GOAL_REACHED"
    assert not completed.locomotion


def test_executor_keeps_wheels_commanded_between_rail_segments() -> None:
    assignments = _assignments()
    executor = MorphologyBehaviorExecutor(
        MorphologyLibrary.load(
            Path(__file__).parents[1]
            / "config"
            / "smores_morphology_behaviors.json"
        )
    )
    executor.start(
        MorphologyCommand("rail-continuous", "snake8", "crawl_stairs"),
        assignments,
        {item.module_id: 0.0 for item in assignments},
        (
            BehaviorProgramStep(
                phase="RAIL_HOLD",
                linear_m_s=0.02,
                active_target_roles=ROLES,
                position_goal=LongitudinalPositionGoal(
                    module_id="m7", target_x_m=0.60, tolerance_m=0.004
                ),
                continuous_with_next=True,
            ),
            BehaviorProgramStep(
                phase="RAIL_SHIFT",
                posture_targets=(
                    JointTarget(
                        module_id="m6",
                        joint="tilt",
                        angle_rad=1.0,
                        target_vertex_id="v6",
                        target_role="snake_neck",
                        coordination_group="rail:shift",
                        max_servo_speed_rad_s=0.4,
                        angle_reference="captured_neutral",
                    ),
                ),
                linear_m_s=0.01,
                active_target_roles=ROLES,
                position_goal=LongitudinalPositionGoal(
                    module_id="m7", target_x_m=0.68, tolerance_m=0.004
                ),
                hold_locomotion_until_admitted=False,
            ),
        ),
    )

    decision = executor.step(
        0.0, module_positions={"m7": (0.60, 0.0, 0.031)}
    )
    assert decision.primitive_goal is not None
    assert decision.phase == "RAIL_SHIFT"
    assert set(decision.locomotion) == {
        f"m{index}" for index in range(8)
    }


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
        ({"transition_clearance_m": 0.020}, "must be in"),
        (
            {"upper_riser_edge_release_lead_m": 0.040},
            "upper_riser_edge_release_lead_m",
        ),
        ({"head_prelift_lookahead_m": 0.020}, "must be in"),
        (
            {
                "head_prelift_lookahead_m": 0.060,
                "head_prelift_ramp_m": 0.080,
            },
            "head_prelift_ramp_m",
        ),
        ({"head_hook_transfer_m": 0.005}, "head_hook_transfer_m"),
        (
            {"head_overstep_clearance_m": 0.020},
            "head_overstep_clearance_m",
        ),
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


@pytest.mark.parametrize("clearance", (0.003, 0.046))
def test_arch_wave_rejects_invalid_clearance(clearance: float) -> None:
    with pytest.raises(SnakeStairGaitError, match="arch_clearance_m"):
        SnakeStairGaitPlanner().plan_arch_wave(
            _graph(),
            _assignments(),
            {"arch_clearance_m": clearance},
        )
