"""Tests for operational behaviors of assembled SMORES-EP morphologies."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from mssr_expert.behaviors.morphology_library import (
    AssignedModule,
    MorphologyLibrary,
    MorphologyLibraryError,
)
from mssr_expert.execution.morphology_behavior_executor import (
    MorphologyBehaviorExecutor,
    MorphologyCommand,
)
from mssr_expert.nodes.smores_morphology_behavior_node import (
    assembly_readiness,
    load_behavior_morphology_catalog,
)
from mssr_expert.nodes.smores_self_reconfiguration_node import (
    load_morphology_catalog,
)
from mssr_expert.primitives.common import logical_tilt_positions


def _library() -> MorphologyLibrary:
    return MorphologyLibrary.load(
        Path(__file__).parents[1]
        / "config"
        / "smores_morphology_behaviors.json"
    )


def _assignments(roles: tuple[str, ...]) -> tuple[AssignedModule, ...]:
    return tuple(
        AssignedModule(
            module_id=f"m{index}",
            target_vertex_id=f"v{index}",
            target_role=role,
        )
        for index, role in enumerate(roles)
    )


RC_CAR_ROLES = (
    "chassis_center",
    "chassis_left",
    "chassis_right",
    "wheel_left_front",
    "wheel_left_rear",
    "wheel_right_front",
    "wheel_right_rear",
)

RC_CAR8_ROLES = (
    "chassis_center_left",
    "chassis_left",
    "chassis_center_right",
    "chassis_right",
    "wheel_left_front",
    "wheel_left_rear",
    "wheel_right_front",
    "wheel_right_rear",
)

MANIPULATOR_ROLES = (
    "chassis_center",
    "right_drive",
    "front_support",
    "left_drive",
    "arm_base",
    "arm_link",
    "end_effector",
)

MANIPULATOR8_ROLES = (
    "chassis_center",
    "right_drive",
    "front_support",
    "left_drive",
    "arm_ground_drive",
    "arm_lift",
    "arm_link",
    "end_effector",
)

SNAKE8_ROLES = (
    "snake_tail",
    "snake_rear",
    "snake_hip",
    "snake_center_rear",
    "snake_center_front",
    "snake_shoulder",
    "snake_neck",
    "snake_head",
)

BRIDGE8_ROLES = (
    "bridge_rear_support",
    "bridge_rear_riser",
    "bridge_rear_span",
    "bridge_center_rear",
    "bridge_center_front",
    "bridge_front_span",
    "bridge_front_riser",
    "bridge_front_support",
)

HOLONOMIC_ROLES = (
    "holonomic_center",
    "holonomic_inner_west",
    "holonomic_inner_north",
    "holonomic_inner_east",
    "holonomic_inner_south",
    "holonomic_drive_west",
    "holonomic_drive_north",
    "holonomic_drive_east",
    "holonomic_drive_south",
)


def _succeeded(goal) -> dict:
    return {
        "schema_version": "mssr.primitive_status.v1",
        "goal_id": goal.goal_id,
        "primitive": goal.primitive,
        "state": "succeeded",
        "module_ids": list(goal.module_ids),
        "phase": "terminal",
        "progress": 1.0,
        "code": "JOINT_TARGET_REACHED",
        "message": "done",
    }


def _finish_posture(executor: MorphologyBehaviorExecutor, now_s: float):
    decision = executor.step(now_s)
    while decision.primitive_goal is not None:
        decision = executor.step(
            now_s,
            _succeeded(decision.primitive_goal),
        )
    return decision


def _finish_program_posture(
    executor: MorphologyBehaviorExecutor,
    now_s: float,
):
    goals = []
    decision = executor.step(now_s)
    while decision.primitive_goal is not None:
        goals.append(decision.primitive_goal)
        decision = executor.step(
            now_s,
            _succeeded(decision.primitive_goal),
        )
    return decision, goals


def _angles_by_module(goals) -> dict[str, float]:
    return {
        goal.module_ids[0]: float(goal.parameters["angle_rad"])
        for goal in goals
    }


def test_reference_morphology_library_contains_all_profiles() -> None:
    library = _library()
    assert library.morphology_names == (
        "bridge8",
        "holonomic9",
        "mobile_manipulator7",
        "mobile_manipulator8",
        "rc_car7",
        "rc_car7_reference",
        "rc_car8",
        "snake7",
        "snake8",
    )


def test_reconfiguration_catalog_discovers_all_morphologies() -> None:
    catalog = load_morphology_catalog(
        Path(__file__).parents[1] / "config"
    )

    assert tuple(sorted(catalog)) == (
        "bridge8",
        "holonomic9",
        "mobile_manipulator7",
        "mobile_manipulator8",
        "rc_car7",
        "rc_car7_reference",
        "rc_car8",
        "snake7",
        "snake8",
    )

    behavior_catalog = load_behavior_morphology_catalog(
        Path(__file__).parents[1] / "config"
    )
    assert tuple(sorted(behavior_catalog)) == tuple(sorted(catalog))


def test_rc_car_drive_resolves_physical_modules_by_target_role() -> None:
    library = _library()
    commands = library.drive_commands(
        "rc_car7",
        _assignments(RC_CAR_ROLES),
        linear_m_s=0.06,
        yaw_rate_rad_s=0.2,
    )

    assert set(commands) == {"m3", "m4", "m5", "m6"}
    assert all(command["vx"] == 0.0 for command in commands.values())
    saturation_scale = 2.0 / (0.084 / 0.0314)
    assert commands["m3"]["pan_rate_rad_s"] == pytest.approx(
        0.036 / 0.0314 * saturation_scale
    )
    assert commands["m4"]["pan_rate_rad_s"] == pytest.approx(-2.0)
    assert commands["m5"]["pan_rate_rad_s"] == pytest.approx(
        0.036 / 0.0314 * saturation_scale
    )
    assert commands["m6"]["pan_rate_rad_s"] == pytest.approx(-2.0)


def test_rc_car_ready_posture_lowers_all_four_tops_to_forty_five_degrees() -> None:
    targets = _library().ready_joint_targets(
        "rc_car7",
        _assignments(RC_CAR_ROLES),
    )

    assert len(targets) == 4
    assert {target.target_role for target in targets} == set(
        RC_CAR_ROLES[3:]
    )
    assert all(
        target.angle_rad == pytest.approx(-math.radians(45.0), abs=1.0e-6)
        for target in targets
    )
    assert all(
        target.tolerance_rad == pytest.approx(0.08)
        for target in targets
    )


def test_rc_car8_folds_all_four_wheels_as_one_group() -> None:
    assignments = _assignments(RC_CAR8_ROLES)
    wheels_down = _library().ready_joint_targets(
        "rc_car8",
        assignments,
    )
    wheels_stowed = _library().behavior_joint_targets(
        "rc_car8",
        "stow",
        assignments,
    )

    assert [target.target_role for target in wheels_down] == [
        "wheel_left_front",
        "wheel_left_rear",
        "wheel_right_front",
        "wheel_right_rear",
    ]
    assert [target.coordination_group for target in wheels_down] == [
        "wheels_down:drive_supports",
        "wheels_down:drive_supports",
        "wheels_down:drive_supports",
        "wheels_down:drive_supports",
    ]
    assert all(
        target.tolerance_rad == pytest.approx(0.12)
        for target in wheels_down
    )
    assert all(
        target.max_servo_error_rad is None
        for target in wheels_down
    )
    assert [target.target_role for target in wheels_stowed] == [
        "wheel_left_front",
        "wheel_left_rear",
        "wheel_right_front",
        "wheel_right_rear",
    ]
    assert all(
        target.max_servo_error_rad is None
        for target in wheels_stowed
    )
    assert len({
        target.coordination_group for target in wheels_stowed
    }) == 1


def test_rc_car_drive_prepares_wheels_then_runs_until_duration() -> None:
    executor = MorphologyBehaviorExecutor(_library())
    executor.start(
        MorphologyCommand(
            command_id="car-drive-1",
            morphology="rc_car7",
            behavior="drive",
            parameters={
                "linear_m_s": 0.05,
                "yaw_rate_rad_s": 0.0,
                "duration_s": 2.0,
            },
        ),
        _assignments(RC_CAR_ROLES),
    )

    first_posture = executor.step(10.0)
    assert first_posture.primitive_goal is not None
    assert first_posture.primitive_goal.parameters[
        "tolerance_rad"
    ] == pytest.approx(0.08)
    assert "pusher_module_id" not in first_posture.primitive_goal.parameters
    assert (
        "hold_after_group_module_ids"
        not in first_posture.primitive_goal.parameters
    )
    assert (
        "stabilize_during_group_module_ids"
        not in first_posture.primitive_goal.parameters
    )

    driving = executor.step(
        10.0,
        _succeeded(first_posture.primitive_goal),
    )
    while driving.primitive_goal is not None:
        driving = executor.step(
            10.0,
            _succeeded(driving.primitive_goal),
        )
    assert driving.phase == "DRIVE"
    assert len(driving.locomotion) == 4
    assert all(
        "pan_rate_rad_s" in command
        for command in driving.locomotion.values()
    )
    assert all(
        command["vx"] == pytest.approx(0.0)
        for command in driving.locomotion.values()
    )
    assert not driving.done

    finished = executor.step(12.01)
    assert finished.done
    assert finished.success
    assert finished.locomotion == {}


def test_prepare_remains_a_legacy_ready_posture_command() -> None:
    executor = MorphologyBehaviorExecutor(_library())
    executor.start(
        MorphologyCommand(
            command_id="car-prepare-1",
            morphology="rc_car7",
            behavior="prepare",
        ),
        _assignments(RC_CAR_ROLES),
    )

    first = executor.step(0.0)

    assert first.primitive_goal is not None
    assert first.primitive_goal.primitive == "set_tilt"


def test_mobile_manipulator_raise_arm_dispatches_joint_targets() -> None:
    executor = MorphologyBehaviorExecutor(_library())
    executor.start(
        MorphologyCommand(
            command_id="arm-up-1",
            morphology="mobile_manipulator8",
            behavior="raise_arm",
        ),
        _assignments(MANIPULATOR8_ROLES),
    )

    first = executor.step(0.0)
    assert first.primitive_goal is not None
    assert first.primitive_goal.primitive == "set_tilt"
    decision = executor.step(0.1, _succeeded(first.primitive_goal))
    while decision.primitive_goal is not None:
        decision = executor.step(
            0.1, _succeeded(decision.primitive_goal)
        )
    finished = decision
    assert finished.done
    assert finished.success


def test_mobile_manipulator_button_press_holds_until_release() -> None:
    library = _library()
    press_targets = library.behavior_joint_targets(
        "mobile_manipulator8",
        "press_button",
        _assignments(MANIPULATOR8_ROLES),
    )
    release_targets = library.behavior_joint_targets(
        "mobile_manipulator8",
        "release_button",
        _assignments(MANIPULATOR8_ROLES),
    )
    lift_angles = [
        target.angle_rad
        for target in press_targets
        if target.target_role == "arm_lift"
    ]
    link_angles = [
        target.angle_rad
        for target in press_targets
        if target.target_role == "arm_link"
    ]
    release_lift_angles = [
        target.angle_rad
        for target in release_targets
        if target.target_role == "arm_lift"
    ]

    assert lift_angles == pytest.approx([0.75, 0.45, 0.30])
    assert link_angles == pytest.approx([0.65, -0.35, -0.55])
    assert release_lift_angles == pytest.approx([0.45, 0.75])


def test_mobile_ready_posture_leaves_transverse_drives_at_neutral_tilt() -> None:
    targets = _library().ready_joint_targets(
        "mobile_manipulator8",
        _assignments(MANIPULATOR8_ROLES),
    )

    by_role = {
        (target.target_role, target.joint): target.angle_rad
        for target in targets
    }
    assert by_role == {
        ("chassis_center", "tilt"): pytest.approx(0.0),
        ("left_drive", "tilt"): pytest.approx(-0.20),
        ("right_drive", "tilt"): pytest.approx(-0.20),
        ("front_support", "tilt"): pytest.approx(0.0),
        ("arm_ground_drive", "tilt"): pytest.approx(0.0),
        ("arm_lift", "tilt"): pytest.approx(0.75),
        ("arm_link", "tilt"): pytest.approx(0.65),
        ("end_effector", "pan"): pytest.approx(0.0),
    }
    assert all(
        target.tolerance_rad == pytest.approx(0.2)
        for target in targets
        if target.joint == "tilt"
    )


def test_mobile_manipulator_translates_on_longitudinal_left_right_wheels() -> None:
    library = _library()
    commands = library.drive_commands(
        "mobile_manipulator8",
        _assignments(MANIPULATOR8_ROLES),
        linear_m_s=0.04,
        yaw_rate_rad_s=0.0,
    )

    assert set(commands) == {"m2", "m4"}
    assert commands["m2"]["vx"] == pytest.approx(0.04)
    assert commands["m4"]["vx"] == pytest.approx(0.04)
    assert all(command["yaw_rate"] == 0.0 for command in commands.values())


def test_mobile_drive_does_not_dispatch_automatic_arm_motion() -> None:
    executor = MorphologyBehaviorExecutor(_library())
    executor.start(
        MorphologyCommand(
            command_id="manip-drive-group-1",
            morphology="mobile_manipulator8",
            behavior="drive",
            parameters={
                "linear_m_s": 0.04,
                "yaw_rate_rad_s": 0.0,
                "duration_s": 2.0,
            },
        ),
        _assignments(MANIPULATOR8_ROLES),
    )

    driving = executor.step(0.0)
    assert driving.primitive_goal is None
    assert driving.state == "RUNNING_DRIVE"
    assert set(driving.locomotion) == {"m2", "m4"}


def test_mobile_manipulator_drive_preserves_current_tilt_posture() -> None:
    library = _library()
    assignments = _assignments(MANIPULATOR8_ROLES)

    translating = library.drive_joint_targets(
        "mobile_manipulator8", assignments, 0.04, 0.0
    )
    spinning = library.drive_joint_targets(
        "mobile_manipulator8", assignments, 0.0, 0.4
    )

    assert translating == ()
    assert spinning == ()
    spin_commands = library.drive_commands(
        "mobile_manipulator8",
        assignments,
        linear_m_s=0.0,
        yaw_rate_rad_s=0.4,
    )
    assert set(spin_commands) == {"m2", "m4"}
    assert spin_commands["m2"]["vx"] == pytest.approx(0.04)
    assert spin_commands["m4"]["vx"] == pytest.approx(-0.04)


def test_holonomic_drive_keeps_pods_straight_and_projects_translation() -> None:
    roles = (
        "holonomic_center",
        "holonomic_inner_west",
        "holonomic_inner_north",
        "holonomic_inner_east",
        "holonomic_inner_south",
        "holonomic_drive_west",
        "holonomic_drive_north",
        "holonomic_drive_east",
        "holonomic_drive_south",
    )
    library = _library()
    assignments = _assignments(roles)
    posture = library.drive_joint_targets(
        "holonomic9",
        assignments,
        linear_m_s=0.04,
        yaw_rate_rad_s=0.0,
        lateral_m_s=0.02,
    )
    commands = library.drive_commands(
        "holonomic9",
        assignments,
        linear_m_s=0.04,
        yaw_rate_rad_s=0.0,
        lateral_m_s=0.02,
    )

    assert posture == ()
    assert commands["m5"]["vx"] == pytest.approx(-0.02)
    assert commands["m6"]["vx"] == pytest.approx(-0.04)
    assert commands["m7"]["vx"] == pytest.approx(0.02)
    assert commands["m8"]["vx"] == pytest.approx(0.04)
    assert {command["yaw_rate"] for command in commands.values()} == {0.0}


def test_holonomic_drive_preserves_completed_fold_without_redeploying() -> None:
    executor = MorphologyBehaviorExecutor(_library())
    executor.start(
        MorphologyCommand(
            command_id="holonomic-drive-preserve-fold",
            morphology="holonomic9",
            behavior="drive",
            parameters={
                "linear_m_s": 0.02,
                "lateral_m_s": 0.0,
                "yaw_rate_rad_s": 0.0,
                "duration_s": 1.0,
            },
        ),
        _assignments(HOLONOMIC_ROLES),
    )

    moving = executor.step(0.0)

    assert moving.primitive_goal is None
    assert moving.phase == "DRIVE"
    assert set(moving.locomotion) == {"m5", "m6", "m7", "m8"}


def test_holonomic_deploy_dispatches_external_pusher_with_inner_tilt() -> None:
    executor = MorphologyBehaviorExecutor(_library())
    executor.start(
        MorphologyCommand(
            command_id="deploy-physical",
            morphology="holonomic9",
            behavior="deploy",
        ),
        _assignments(HOLONOMIC_ROLES),
    )

    first = executor.step(0.0).primitive_goal

    assert first is not None
    assert first.module_ids == ("m1",)
    assert first.parameters["angle_rad"] == pytest.approx(-1.35)
    assert first.parameters["pusher_module_id"] == "m5"
    assert first.parameters["pusher_linear_m_s"] == pytest.approx(0.025)
    assert set(first.parameters["hold_after_group_module_ids"]) == {
        f"m{index}" for index in range(9)
    }
    assert first.parameters["stabilize_during_group_module_ids"] == ["m0"]


def test_snake_rejects_unvalidated_worm_gait() -> None:
    roles = (
        "snake_head",
        "snake_link",
        "snake_link",
        "snake_center",
        "snake_link",
        "snake_link",
        "snake_tail",
    )
    with pytest.raises(MorphologyLibraryError, match="not defined"):
        _library().behavior_joint_targets(
            "snake7", "worm", _assignments(roles)
        )


def test_snake_drive_remains_wheel_only_without_holonomic_fold_metadata() -> None:
    roles = (
        "snake_head",
        "snake_link",
        "snake_link",
        "snake_center",
        "snake_link",
        "snake_link",
        "snake_tail",
    )
    executor = MorphologyBehaviorExecutor(_library())
    executor.start(
        MorphologyCommand(
            command_id="snake-drive-regression",
            morphology="snake7",
            behavior="drive",
            parameters={
                "linear_m_s": 0.04,
                "yaw_rate_rad_s": 0.0,
                "duration_s": 1.0,
            },
        ),
        _assignments(roles),
    )

    posture = executor.step(0.0)
    assert posture.primitive_goal is not None
    assert set(posture.primitive_goal.parameters) <= {
        "angle_rad", "structural_hold_module_ids"
    }
    assert len(posture.primitive_goal.parameters["structural_hold_module_ids"]) == 6

    driving = executor.step(0.0, _succeeded(posture.primitive_goal))
    while driving.primitive_goal is not None:
        assert set(driving.primitive_goal.parameters) <= {
            "angle_rad", "structural_hold_module_ids"
        }
        driving = executor.step(
            0.0, _succeeded(driving.primitive_goal)
        )

    assert driving.phase == "DRIVE"
    assert len(driving.locomotion) == 7
    assert all(
        "pan_rate_rad_s" not in command
        for command in driving.locomotion.values()
    )
    assert all(
        command["vx"] == pytest.approx(0.04)
        for command in driving.locomotion.values()
    )


def test_snake_rejects_turning_until_a_turn_gait_is_defined() -> None:
    roles = (
        "snake_head",
        "snake_link",
        "snake_link",
        "snake_center",
        "snake_link",
        "snake_link",
        "snake_tail",
    )
    with pytest.raises(MorphologyLibraryError, match="yaw rate"):
        _library().drive_commands(
            "snake7",
            _assignments(roles),
            linear_m_s=0.04,
            yaw_rate_rad_s=0.1,
        )


def test_snake8_stair_postures_are_coordinated_and_drive_preserves_them() -> None:
    assignments = _assignments(SNAKE8_ROLES)
    assert _library().uses_captured_neutral("snake8")
    assert not _library().uses_captured_neutral("bridge8")
    targets = _library().behavior_joint_targets(
        "snake8", "lift_head", assignments
    )

    assert len(targets) == 5
    assert [target.angle_rad for target in targets] == pytest.approx(
        [0.18, 0.18, 0.18, 0.18, 0.18]
    )
    assert [target.target_vertex_id for target in targets] == [
        "v2", "v3", "v4", "v5", "v6"
    ]
    assert len({target.coordination_group for target in targets}) == 3
    assert all(
        target.max_servo_error_rad == pytest.approx(0.18)
        for target in targets
    )
    hook = _library().behavior_joint_targets(
        "snake8", "hook_step", assignments
    )
    assert [target.target_vertex_id for target in hook if target.angle_rad] == [
        "v2", "v3", "v4", "v5", "v6"
    ]
    straight = _library().behavior_joint_targets(
        "snake8", "straighten", assignments
    )
    assert [target.target_vertex_id for target in straight] == [
        "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"
    ]
    assert all(target.angle_rad == pytest.approx(0.0) for target in straight)
    assert all(
        target.angle_reference == "captured_neutral"
        for target in straight
    )
    assert _library().drive_joint_targets(
        "snake8", assignments, 0.03, 0.0
    ) == ()
    commands = _library().drive_commands(
        "snake8", assignments, 0.03, 0.0
    )
    assert len(commands) == 8
    assert all(command["vx"] == pytest.approx(0.03) for command in commands.values())
    assert all(
        command["yaw_rate"] == pytest.approx(0.0)
        for command in commands.values()
    )

    curved_commands = _library().drive_commands(
        "snake8", assignments, 0.03, 0.2
    )
    assert len(curved_commands) == 8
    assert all(
        command["vx"] == pytest.approx(0.03)
        for command in curved_commands.values()
    )
    assert all(
        command["yaw_rate"] == pytest.approx(0.2)
        for command in curved_commands.values()
    )

    train = _library().composite_behavior_steps(
        "snake8", "train", assignments, {"duration_s": 9.0}
    )
    assert len(train) == 1
    assert train[0].duration_s == pytest.approx(9.0)
    assert set(train[0].active_target_roles) == set(SNAKE8_ROLES)

    pull = _library().composite_behavior_steps(
        "snake8", "pull_over_step", assignments
    )
    assert [step.phase for step in pull] == [
        "SET_HOOK", "PULL_FRONT", "TRANSFER_LOAD", "PULL_BODY",
        "RETURN_FLAT",
    ]
    assert set(pull[1].active_target_roles) == set(SNAKE8_ROLES)
    assert set(pull[3].active_target_roles) == set(SNAKE8_ROLES)


def test_snake8_straighten_restores_captured_neutral_only() -> None:
    assignments = _assignments(SNAKE8_ROLES)
    neutral = {
        assignment.module_id: 0.08 + 0.005 * index
        for index, assignment in enumerate(assignments)
    }
    executor = MorphologyBehaviorExecutor(_library())
    executor.start(
        MorphologyCommand(
            command_id="snake-neutral-1",
            morphology="snake8",
            behavior="straighten",
        ),
        assignments,
        neutral,
    )

    finished, goals = _finish_program_posture(executor, 0.0)

    assert finished.done
    assert finished.success
    assert _angles_by_module(goals) == pytest.approx(neutral)

    lift_executor = MorphologyBehaviorExecutor(_library())
    lift_executor.start(
        MorphologyCommand(
            command_id="snake-lift-1",
            morphology="snake8",
            behavior="lift_head",
        ),
        assignments,
        neutral,
    )
    first_lift = lift_executor.step(0.0).primitive_goal
    assert first_lift is not None
    assert first_lift.parameters["angle_rad"] == pytest.approx(0.18)


def test_snake8_neutral_requires_a_captured_tilt_for_every_module() -> None:
    with pytest.raises(
        MorphologyLibraryError,
        match="No captured neutral tilt is available for m7",
    ):
        MorphologyBehaviorExecutor(_library()).start(
            MorphologyCommand(
                command_id="snake-neutral-missing",
                morphology="snake8",
                behavior="straighten",
            ),
            _assignments(SNAKE8_ROLES),
            {f"m{index}": 0.1 for index in range(7)},
        )


def test_module_state_tilts_are_converted_to_primitive_coordinates() -> None:
    assert logical_tilt_positions(
        {
            "modules": [
                {
                    "module_id": "smores_01",
                    "actuators": {
                        "tilt": {"position_rad": -0.11},
                    },
                },
                {"module_id": "missing-actuator"},
            ]
        }
    ) == {"smores_01": pytest.approx(0.11)}


def test_coordinated_posture_targets_do_not_hold_each_other() -> None:
    executor = MorphologyBehaviorExecutor(_library())
    executor.start(
        MorphologyCommand(
            command_id="bridge-flat-test",
            morphology="bridge8",
            behavior="prepare",
        ),
        _assignments(BRIDGE8_ROLES),
    )

    first = executor.step(0.0).primitive_goal

    assert first is not None
    assert first.parameters["coordination_size"] == 8
    assert "structural_hold_module_ids" not in first.parameters


def test_bridge8_uses_distributed_drive_and_internal_lever_joints() -> None:
    assignments = _assignments(BRIDGE8_ROLES)
    targets = _library().behavior_joint_targets(
        "bridge8", "deploy_span", assignments
    )
    commands = _library().drive_commands(
        "bridge8", assignments, 0.02, 0.0
    )

    assert len(targets) == 7
    assert [target.target_vertex_id for target in targets] == [
        "v1", "v2", "v3", "v1", "v2", "v3", "v4",
    ]
    assert all(target.target_vertex_id not in {"v0", "v7"} for target in targets)
    assert sorted(commands) == [f"m{index}" for index in range(8)]
    assert all(command["vx"] == pytest.approx(0.02) for command in commands.values())


def test_bridge8_cross_gap_program_is_sequential_and_parameterized() -> None:
    steps = _library().composite_behavior_steps(
        "bridge8",
        "cross_gap",
        _assignments(BRIDGE8_ROLES),
        {
            "linear_m_s": 0.02,
            "approach_duration_s": 1.2,
            "rear_push_duration_s": 1.0,
            "front_transfer_duration_s": 2.0,
            "rear_clear_duration_s": 1.5,
        },
    )

    assert [step.phase for step in steps] == [
        "APPROACH_EDGE",
        "LIFT_FRONT_PRELOAD",
        "LIFT_FRONT",
        "ADVANCE_REAR",
        "LOWER_FRONT",
        "LAND_FRONT",
        "TRANSFER_FRONT",
        "LIFT_REAR_PRELOAD",
        "LIFT_REAR",
        "CLEAR_REAR",
        "RETURN_FLAT",
    ]
    assert [step.kind for step in steps] == [
        "drive", "posture", "posture", "drive", "posture",
        "posture", "drive", "posture", "posture", "drive", "posture",
    ]
    assert steps[0].duration_s == pytest.approx(1.2)
    assert steps[3].duration_s == pytest.approx(1.0)
    assert steps[6].duration_s == pytest.approx(2.0)
    assert steps[9].duration_s == pytest.approx(1.5)
    assert steps[3].linear_m_s == pytest.approx(0.02)
    assert set(steps[0].active_target_roles) == set(BRIDGE8_ROLES)
    assert steps[3].active_target_roles == (
        "bridge_rear_support",
        "bridge_rear_riser",
        "bridge_rear_span",
        "bridge_center_rear",
    )
    assert steps[6].active_target_roles == (
        "bridge_center_front",
        "bridge_front_span",
        "bridge_front_riser",
        "bridge_front_support",
    )
    assert steps[9].active_target_roles == steps[6].active_target_roles
    assert all(
        target.angle_rad == pytest.approx(0.0)
        for target in steps[-1].posture_targets
    )


def test_bridge8_cross_gap_executes_support_handover_in_order() -> None:
    executor = MorphologyBehaviorExecutor(_library())
    executor.start(
        MorphologyCommand(
            command_id="bridge-cross-test",
            morphology="bridge8",
            behavior="cross_gap",
            parameters={
                "linear_m_s": 0.02,
                "approach_duration_s": 1.2,
                "rear_push_duration_s": 1.0,
                "front_transfer_duration_s": 2.0,
                "rear_clear_duration_s": 1.5,
            },
        ),
        _assignments(BRIDGE8_ROLES),
    )

    approach = executor.step(0.0)
    assert approach.phase == "APPROACH_EDGE"
    assert set(approach.locomotion) == {f"m{index}" for index in range(8)}
    assert executor.step(1.21).phase == "APPROACH_EDGE_STOP"

    lift_preload, lift_preload_goals = _finish_program_posture(executor, 1.22)
    assert lift_preload.phase == "LIFT_FRONT_PRELOAD_COMPLETE"
    assert _angles_by_module(lift_preload_goals) == {
        "m1": pytest.approx(0.18),
        "m2": pytest.approx(0.14),
        "m3": pytest.approx(0.08),
    }

    lift_front, lift_front_goals = _finish_program_posture(executor, 1.22)
    assert lift_front.phase == "LIFT_FRONT_COMPLETE"
    assert _angles_by_module(lift_front_goals) == {
        "m1": pytest.approx(0.36),
        "m2": pytest.approx(0.30),
        "m3": pytest.approx(0.20),
        "m4": pytest.approx(0.10),
    }
    assert all(
        goal.parameters["max_servo_error_rad"] == pytest.approx(0.18)
        for goal in lift_front_goals
    )

    advance_rear = executor.step(1.23)
    assert advance_rear.phase == "ADVANCE_REAR"
    assert set(advance_rear.locomotion) == {"m0", "m1", "m2", "m3"}
    assert advance_rear.locomotion["m0"]["vx"] == pytest.approx(0.02)
    assert executor.step(2.24).phase == "ADVANCE_REAR_STOP"

    lower_front, lower_front_goals = _finish_program_posture(executor, 2.25)
    assert lower_front.phase == "LOWER_FRONT_COMPLETE"
    assert _angles_by_module(lower_front_goals) == {
        "m1": pytest.approx(0.16),
        "m2": pytest.approx(0.10),
        "m3": pytest.approx(0.04),
        "m4": pytest.approx(0.0),
    }

    land_front, land_front_goals = _finish_program_posture(executor, 2.25)
    assert land_front.phase == "LAND_FRONT_COMPLETE"
    assert _angles_by_module(land_front_goals) == {
        "m0": pytest.approx(0.0),
        "m1": pytest.approx(0.0),
        "m2": pytest.approx(0.0),
        "m3": pytest.approx(0.0),
        "m4": pytest.approx(0.0),
        "m5": pytest.approx(0.0),
        "m6": pytest.approx(0.0),
        "m7": pytest.approx(0.0),
    }

    transfer = executor.step(2.26)
    assert transfer.phase == "TRANSFER_FRONT"
    assert set(transfer.locomotion) == {"m4", "m5", "m6", "m7"}
    assert transfer.locomotion["m4"]["vx"] == pytest.approx(0.02)
    assert executor.step(4.27).phase == "TRANSFER_FRONT_STOP"

    lift_rear_preload, lift_rear_preload_goals = _finish_program_posture(
        executor, 4.28
    )
    assert lift_rear_preload.phase == "LIFT_REAR_PRELOAD_COMPLETE"
    assert _angles_by_module(lift_rear_preload_goals) == {
        "m4": pytest.approx(-0.08),
        "m5": pytest.approx(-0.14),
        "m6": pytest.approx(-0.18),
    }

    lift_rear, lift_rear_goals = _finish_program_posture(executor, 4.28)
    assert lift_rear.phase == "LIFT_REAR_COMPLETE"
    assert _angles_by_module(lift_rear_goals) == {
        "m3": pytest.approx(-0.06),
        "m4": pytest.approx(-0.14),
        "m5": pytest.approx(-0.22),
        "m6": pytest.approx(-0.28),
    }

    clear_rear = executor.step(4.29)
    assert clear_rear.phase == "CLEAR_REAR"
    assert set(clear_rear.locomotion) == {"m4", "m5", "m6", "m7"}
    assert clear_rear.locomotion["m7"]["vx"] == pytest.approx(0.02)
    assert executor.step(5.8).phase == "CLEAR_REAR_STOP"

    returned_flat, flat_goals = _finish_program_posture(executor, 5.81)
    assert returned_flat.phase == "RETURN_FLAT_COMPLETE"
    assert all(
        angle == pytest.approx(0.0)
        for angle in _angles_by_module(flat_goals).values()
    )
    completed = executor.step(5.82)
    assert completed.done
    assert completed.success
    assert completed.locomotion == {}


def test_assignment_count_is_checked_before_actuation() -> None:
    with pytest.raises(MorphologyLibraryError, match="needs 7"):
        _library().ready_joint_targets(
            "rc_car7",
            _assignments(RC_CAR_ROLES[:-1]),
        )


def test_morphology_commands_wait_for_successful_self_assembly() -> None:
    assert assembly_readiness(
        {"execution_state": {"state": "POSTURE", "done": False}}
    ) == (False, "POSTURE")
    assert assembly_readiness(
        {
            "execution_state": {
                "state": "SUCCEEDED",
                "done": True,
                "success": True,
            }
        }
    ) == (True, "SUCCEEDED")
    assert assembly_readiness(
        {
            "execution_state": {
                "state": "FAILED",
                "done": True,
                "success": False,
            }
        }
    ) == (False, "FAILED")
