"""Static T7 integration contracts for MobileManipulator8 teleoperation."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]

NODE = (
    ROOT
    / "mssr_ws/src/mssr_expert/"
      "mssr_expert/nodes/smores_teleop_node.py"
)

TELEOP_CONFIG = (
    ROOT
    / "mssr_ws/src/mssr_expert/"
      "config/smores_teleop.yaml"
)


def node_source() -> str:
    return NODE.read_text(encoding="utf-8")


def config_source() -> str:
    return TELEOP_CONFIG.read_text(encoding="utf-8")


def test_node_declares_all_three_validated_runtime_controllers():
    text = node_source()

    assert (
        'controller_morphologies={'
        '"rc_car8", "snake8", "mobile_manipulator8"}'
        in text
    )

    assert (
        "from mssr_expert.teleop.mobile_manipulator "
        "import MobileManipulatorRuntime"
        in text
    )


def test_node_constructs_mm8_runtime_from_existing_library_and_target_graph():
    text = node_source()

    assert 'get("mobile_manipulator", {})' in text
    assert "self._mm8 = MobileManipulatorRuntime(" in text
    assert '"smores_mobile_manipulator8.json"' in text


def test_shipped_teleop_yaml_has_mm8_manual_runtime_parameters():
    text = config_source()

    assert "mobile_manipulator:" in text
    assert "joint_rate_rad_s:" in text
    assert "joint_deadband_rad:" in text
    assert "observation_timeout_s:" in text


def test_robot_graph_is_forwarded_to_mm8_runtime():
    text = node_source()

    assert "self._mm8.observe_graph(payload, now=now)" in text


def test_mm8_runtime_freshness_is_required_for_mm8_topology_authority():
    text = node_source()

    assert (
        'self._topology_name == "mobile_manipulator8"'
        in text
    )

    assert (
        "self._mm8.topology(now) is None"
        in text
    )


def test_primitive_status_is_forwarded_to_mm8_posture_transport():
    text = node_source()

    assert "self._mm8.observe_status(" in text


def test_tick_steps_mm8_with_same_safety_decision_pattern():
    text = node_source()

    assert "mm8_output = self._mm8.step(" in text

    assert (
        'decision if controller == "mobile_manipulator8" '
        "else disabled"
        in text
    )


def test_active_output_selection_contains_mm8():
    text = node_source()

    assert '"mobile_manipulator8": mm8_output' in text

    # T7 must not silently fall back to RC when MM8 is active.
    assert (
        'output = outputs.get(controller)'
        in text
    )


def test_mm8_posture_delivery_uses_common_goal_cancel_path():
    text = node_source()

    assert (
        "for delivery in "
        "(rc_output.posture, snake_output.posture, mm8_output.posture):"
        in text
    )


def test_recording_selects_authoritative_graph_for_all_three_morphologies():
    text = node_source()

    assert "recording_graphs =" in text
    assert '"rc_car8": self._rc.latest_graph' in text
    assert '"snake8": self._snake.latest_graph' in text
    assert (
        '"mobile_manipulator8": self._mm8.latest_graph'
        in text
    )

    assert (
        'graph=recording_graphs.get(status["active_controller"])'
        in text
    )


def test_status_exposes_mm8_intent_and_effective_actions():
    text = node_source()

    assert "mm8_intent=mm8_output.actions.intent" in text
    assert (
        "mm8_effective_actions="
        "mm8_output.actions.module_actions"
        in text
    )


def test_startup_log_mentions_all_three_controllers():
    text = node_source()

    assert "RC-Car8/Snake8/MobileManipulator8 teleop" in text
