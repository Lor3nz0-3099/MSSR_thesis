"""Static integration contracts for the ROS/Isaac button script.

These tests intentionally avoid importing the script because the unit-test
container does not provide rclpy/Isaac.  The pure geometry is tested separately.
"""
from __future__ import annotations

from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "run_button_expert_to_ik.py"
SOURCE = SCRIPT.read_text()


def _block(start: str, end: str) -> str:
    return SOURCE.split(start, 1)[1].split(end, 1)[0]


def test_ik_pan_execution_uses_relative_physical_delta() -> None:
    block = _block("def _ik_command_configuration(", "def _ik_clik_press(")
    assert '"rotate_pan_by"' in block
    assert '"delta_rad"' in block
    assert "shortest_angular_delta" in block


def test_ik_residual_has_continuous_orientation_branch_term() -> None:
    block = _block("def _ik_residual(", "def _ik_jacobian(")
    assert "orientation_residual(" in block
    assert "normal @ _IK_FACE_TANGENT" not in block


def test_pre_is_closed_loop_and_relinearized_from_live_state() -> None:
    assert "def _ik_clik_pre(" in SOURCE
    block = _block("def _ik_clik_pre(", "def _ik_clik_press(")
    assert "_ik_read_state()" in block
    assert "bounded_position_waypoint(" in block
    assert "bounded_normal_waypoint(" in block
    assert "_ik_solve(" in block
    assert "_ik_command_configuration(" in block
    assert "One absolute movement to q_pre." not in SOURCE

def test_ik_waits_for_terminal_goal_before_reusing_internal_motion() -> None:
    block = _block(
        "def _ik_command_configuration(",
        "def _ik_clik_pre(",
    )
    assert "_ik_wait_goal_terminal(" in block
    assert "_IK_TERMINAL_STATES" in SOURCE

def test_ik_pan_requests_periodic_equivalent_relative_motion() -> None:
    block = _block(
        "def _ik_command_configuration(",
        "def _ik_clik_pre(",
    )
    assert '"periodic_equivalent": True' in block


def test_pre_orientation_uses_task_deadband() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "run_button_expert_to_ik.py"
    )
    text = script.read_text(encoding="utf-8")

    assert "_IK_NORMAL_DEADBAND_DEG = 30.0" in text

    start = text.index("        def _ik_clik_pre(")
    end = text.index("        def _ik_clik_press(", start)
    pre = text[start:end]

    # PRE considers orientation good enough inside the task cone.
    assert (
        "final_normal_error\n"
        "                    <= _IK_NORMAL_DEADBAND_DEG"
    ) in pre

    # Once inside the cone we no longer chase the perfectly parallel
    # button normal.  The current admissible normal becomes the local
    # reference while position continues converging.
    assert "orientation_in_deadband = (" in pre
    assert "desired_normal = normal.copy()" in pre

    # Progress must not reward polishing 20 deg -> 0 deg once the face
    # is already admissible for button pressing.
    assert "normal_excess_deg = max(" in pre

    # Ignore source formatting/newlines: test the expression contract,
    # not Black-style line wrapping.
    compact_pre = " ".join(pre.split())
    assert (
        "normal_after_deg - _IK_NORMAL_DEADBAND_DEG"
        in compact_pre
    )


def test_preorient_end_effector_is_tilt_only() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "run_button_expert_to_ik.py"
    )
    text = script.read_text(encoding="utf-8")

    assert "_PREORIENT_TILT_SCAN_STEP_DEG = 2.0" in text
    assert "_PREORIENT_MIN_IMPROVEMENT_DEG = 5.0" in text
    assert "_PREORIENT_MIN_MOVE_DEG = 1.0" in text

    planner_start = text.index(
        "        def _ik_plan_end_effector_tilt_preorientation("
    )
    planner_end = text.index(
        "        def _ik_clik_pre(",
        planner_start,
    )
    planner = text[planner_start:planner_end]

    # Planning must be purely virtual and TILT-only.
    assert "_ik_build_reference(state)" in planner
    compact_planner = "".join(planner.split())
    assert "_ik_fk(reference,q_trial," in compact_planner
    assert '("end_effector", "tilt")' in planner
    assert '("end_effector", "pan")' not in planner
    assert '"should_move"' in planner
    assert '"predicted_normal_error_deg"' in planner

    # Existing configuration executor must support selecting joints,
    # otherwise giving it the full q vector would also emit PAN goals.
    command_start = text.index(
        "        def _ik_command_configuration("
    )
    command_end = text.index(
        "        def _ik_plan_end_effector_tilt_preorientation(",
        command_start,
    )
    command = text[command_start:command_end]

    assert "only_dofs=None" in command
    # The executor normalizes the requested subset once into `selected`,
    # then uses that same list both for physical dispatch and final settle.
    assert "selected = [" in command
    assert "if selected_dofs is None" in command
    assert "or (role, joint) in selected_dofs" in command
    assert "for primitive_index, (index, role, joint) in enumerate(" in command
    assert "for index, role, joint in selected:" in command

    # Exactly one physical preorientation configuration command and
    # explicitly only end_effector.tilt.
    call_start = text.index(
        "        preorient_plan = "
        "_ik_plan_end_effector_tilt_preorientation("
    )
    call_end = text.index(
        "        pre_result = _ik_clik_pre(",
        call_start,
    )
    call = text[call_start:call_end]

    assert call.count("_ik_command_configuration(") == 1
    assert 'only_dofs={("end_effector", "tilt")}' in call
    assert "PRE-ORIENT TILT" in call



def test_button_runner_uses_direct_mm8_assembly_without_nav2_or_reconfiguration() -> None:
    main = SOURCE[SOURCE.index("def main():"):]

    assert "config/smores_mobile_manipulator8.json" in main
    assert "config/smores_rc_car8.json" not in main
    assert "smores_nav2.launch.py" not in main
    assert "mssr_smores_self_reconfiguration_node" not in main


def test_button_runner_phase_order_is_direct_mm8_then_reposition_manipulation() -> None:
    main = SOURCE[SOURCE.index("def main():"):]

    assembly = main.index("PHASE 2 — MOBILEMANIPULATOR8 ASSEMBLY")
    reposition = main.index("MM8 BUTTON-RELATIVE REPOSITION")
    manipulation = main.index("PREPARE MANIPULATION")

    assert assembly < reposition < manipulation

    # Direct-MM8 flow must not insert the legacy Scorpion posture transition.
    assert "SCORPION DRIVE POSTURE" not in main
    assert '"restore_drive"' not in main

def test_button_success_path_ends_in_mm8_press_posture() -> None:
    terminal = SOURCE[SOURCE.index("# 6C. TERMINAL PHYSICAL SUCCESS"):]

    assert '"final_morphology": "mobile_manipulator8"' in terminal
    assert '"restore_drive"' not in terminal
    assert "MM8 RETREAT" not in terminal
    assert "target_morphology:=rc_car8" not in terminal
    assert "pre-return" not in terminal
    assert "post-press-clearance" not in terminal


def test_manipulation_recorder_supervises_one_primitive_goal_only() -> None:
    recorder = _block(
        "def _ik_record_primitive_transition(",
        "def _ik_read_state(",
    )

    assert '"primitive_goal": primitive_goal' in recorder
    assert '"joint_configuration"' not in recorder
    assert '"primitive_sequence"' not in recorder
    assert '"primitive_index"' in recorder
    assert '"primitive_count"' in recorder


def test_configuration_executor_records_physical_endpoint_per_primitive() -> None:
    block = _block(
        "def _ik_command_configuration(",
        "def _ik_plan_end_effector_tilt_preorientation(",
    )

    assert "_ik_read_transition_endpoint(" in block
    assert block.count("_ik_record_primitive_transition(") >= 2
    assert "carried_state = state_after" in block
    assert "carried_graph = graph_after" in block
    assert "primitive_index < primitive_count" in block
    assert "now - stable_since >= 0.6" in block
    assert "terminal_anchor_graph = _ik_read_robot_graph()" in block
    assert "settle_anchor_graph = _ik_read_robot_graph()" in block


def test_transition_endpoint_requires_strictly_newer_graph_sample() -> None:
    helper = _block(
        "def _ik_read_transition_endpoint(",
        "def _ik_button_depression(",
    )
    assert "last_graph.stamp > graph_before.stamp" in helper
    assert "last_graph.stamp != graph_before.stamp" not in helper


def test_failed_press_does_not_dispatch_recovery_training_action() -> None:
    start = SOURCE.index('if not clik_result["success"]:')
    end = SOURCE.index("press_bottom = _ik_connector(", start)
    failure = SOURCE[start:end]

    assert "_ik_command_configuration(" not in failure
    assert "No recovery primitive is recorded" in failure


def test_direct_button_dataset_layout_uses_only_active_streams() -> None:
    main = SOURCE[SOURCE.index("def main():"):]

    for name in (
        "assembly_dataset_path",
        "mm8_behavior_dataset_path",
        "manipulation_dataset_path",
    ):
        assert name in main

    assert '"dataset_path:=" + str(assembly_dataset_path)' in main
    assert "_ik_dataset_path = manipulation_dataset_path" in main



def test_mm8_reposition_runner_is_translation_only_on_fixed_axis() -> None:
    from pathlib import Path as _Path

    source = (
        _Path(__file__).parents[1]
        / "run_button_expert_to_ik.py"
    ).read_text()

    import_start = source.index(
        "from smores_ep.control.button_ik_math import ("
    )
    import_end = source.index(
        "\n)",
        import_start,
    )
    import_block = source[import_start:import_end]

    assert "plan_mm8_reposition_command" in import_block

    start = source.index("def approach_mm8(")
    end = source.index("\ndef parse_args(", start)
    approach = source[start:end]

    assert "plan_mm8_reposition_command(" in approach

    # Direct MM8 assembly fixes the longitudinal drive direction:
    # forward is opposite to the button press direction.
    assert "drive_axis_x=-press_nx" in approach
    assert "drive_axis_y=-press_ny" in approach

    # MM8 cannot steer.
    assert "align_heading" not in approach
    assert "final_align" not in approach
    assert "final_yaw" not in approach
    assert "TURN ->" not in approach

    # Runtime must publish translation only.
    assert 'command["linear_x_m_s"]' in approach
    assert "cmd.linear.x" in approach
    assert "cmd.angular.z = 0.0" in approach

    # Completion still comes from the planner.
    assert 'phase == "done"' in approach

    # The old lateral-error abort must stay removed.
    assert "Lateral error too large" not in approach


def test_direct_mm8_button_flow_does_not_restore_drive_posture() -> None:
    from pathlib import Path as _Path

    source = (
        _Path(__file__).parents[1]
        / "run_button_expert_to_ik.py"
    ).read_text()

    start = source.index("def main(")
    main = source[start:]

    # Direct MM8 assembly must keep the assembled posture.
    assert '"restore_drive"' not in main
    assert "SCORPION DRIVE POSTURE" not in main
