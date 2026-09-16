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
    assert command.count("selected_dofs is not None") >= 2

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



def test_composite_button_dataset_layout_declares_distinct_phase_paths() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    # New composite interface. Legacy --dataset-path remains supported
    # for standalone/backward-compatible runs.
    assert '"--dataset-layout-json"' in text

    # A composite button stage must resolve independent raw streams.
    for name in (
        "rc_behavior_dataset_path",
        "rc_to_mm8_dataset_path",
        "mm8_behavior_dataset_path",
        "manipulation_dataset_path",
        "mm8_to_rc_dataset_path",
    ):
        assert name in text

    # Required layout keys.
    for key in (
        '"rc_behavior"',
        '"rc_to_mm8_reconfiguration"',
        '"mm8_behavior"',
        '"manipulation"',
        '"mm8_to_rc_reconfiguration"',
    ):
        assert key in text


def test_rc_to_mm8_reconfiguration_uses_dedicated_dataset_path() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"target_morphology:=mobile_manipulator8"' in text
    assert (
        '"dataset_path:=" + str(rc_to_mm8_dataset_path)'
        in text
    )


def test_manipulation_ik_uses_dedicated_dataset_path() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "_ik_dataset_path = manipulation_dataset_path" in text


def test_mm8_to_rc_reconfiguration_uses_dedicated_dataset_path() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"target_morphology:=rc_car8"' in text
    assert (
        '"dataset_path:=" + str(mm8_to_rc_dataset_path)'
        in text
    )


def test_button_runtime_starts_with_rc_behavior_dataset_path() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert (
        '"behavior_dataset_path:="\n'
        '                + str(rc_behavior_dataset_path)'
        in text
    )


def test_button_behavior_stream_barrier_surrounds_rc_to_mm8_reconfiguration() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    rc_context = text.index(
        "dataset_path=rc_behavior_dataset_path"
    )
    reconfiguration = text.index(
        '"target_morphology:=mobile_manipulator8"'
    )
    clear_context = text.rfind(
        "dataset_path=None",
        rc_context,
        reconfiguration,
    )
    mm8_context = text.index(
        "dataset_path=mm8_behavior_dataset_path",
        reconfiguration,
    )
    approach = text.index(
        "result_approach = approach_mm8(",
        mm8_context,
    )

    assert rc_context < clear_context
    assert clear_context < reconfiguration
    assert reconfiguration < mm8_context
    assert mm8_context < approach
