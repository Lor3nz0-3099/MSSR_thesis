"""Input/state coordination and strict shell configuration without ROS."""
import importlib
import importlib.util
from pathlib import Path

import pytest
import yaml

from mssr_expert.teleop.input import InputConfig


CONFIG_DIR = Path(__file__).parents[1] / "config"


def component(name):
    path = "mssr_expert.teleop." + name
    assert importlib.util.find_spec(path) is not None, f"missing T1 {name}"
    return importlib.import_module(path)


def session():
    mapping = yaml.safe_load((CONFIG_DIR / "smores_dualsense.yaml").read_text())
    # Synthetic test bindings only; shipped deferred assignments stay null.
    mapping["commands"].update(select_rc="cross", select_snake="triangle",
                               select_mm8="circle", home="square",
                               estop="ps", resume="share")
    return component("session").TeleopSession(InputConfig.from_mapping(mapping))


def send(core, pressed=(), at=10.0):
    buttons = [0] * 21
    for index in pressed:
        buttons[index] = 1
    return core.update_joy([0.0] * 6, buttons, at)


def test_shell_starts_ready_without_claiming_topology_or_actuator_authority():
    core = session()
    payload = core.tick(10.0)
    assert payload["phase"] == "READY"
    assert payload["authority"] == "NONE"
    assert payload["active_controller"] is None
    assert not payload["controller_connected"]
    assert payload["controller_input"]["command_events"] == []


def test_snapshot_consumes_start_edges_once_and_preserves_multiple_toggles():
    core = session()
    send(core)
    send(core, [6], 10.01)
    send(core, [], 10.02)
    send(core, [6], 10.03)
    payload = core.tick(10.04)
    assert not payload["recording"]
    assert payload["controller_input"]["command_events"] == ["record_toggle", "record_toggle"]
    assert core.tick(10.05)["controller_input"]["command_events"] == []


def test_estop_dominates_resume_in_the_same_packet():
    core = session()
    send(core)
    send(core, [4, 5, 6], 10.1)
    payload = core.tick(10.11)
    assert payload["phase"] == "ESTOP_PAUSED"
    assert payload["authority"] == "ESTOP"
    assert payload["recording"]
    send(core, [], 10.2)
    send(core, [4], 10.3)
    assert core.tick(10.31)["phase"] == "READY"


def test_selection_sets_intent_but_does_not_infer_physical_topology():
    core = session()
    send(core)
    send(core, [3], 10.1)
    payload = core.tick(10.11)
    assert payload["requested_morphology"] == "snake8"
    assert payload["active_controller"] is None
    assert payload["authority"] == "NONE"


def test_ambiguous_simultaneous_morphology_buttons_do_not_choose_arbitrarily():
    core = session()
    send(core)
    send(core, [0, 3], 10.1)
    payload = core.tick(10.11)
    assert payload["requested_morphology"] is None
    assert payload["rejected_commands"] == ["ambiguous_morphology_selection"]


def test_disconnect_changes_connectivity_once_preserving_recording():
    core = session()
    send(core)
    assert core.tick(10.01)["events"] == ["controller_connected"]
    send(core, [6], 10.1)
    assert core.tick(10.11)["recording"]
    payload = core.tick(10.6)
    assert payload["events"] == ["controller_disconnected"]
    assert payload["recording"]
    assert core.tick(10.7)["events"] == []
    send(core, [6], 10.8)
    payload = core.tick(10.81)
    assert payload["events"] == ["controller_connected"]
    assert payload["recording"]


def test_bad_joy_cannot_refresh_connectivity_or_change_state():
    core = session()
    send(core)
    assert not core.update_joy([0] * 5, [0] * 21, 10.4)
    payload = core.tick(10.5)
    assert not payload["controller_connected"]
    assert payload["valid_joy_packets"] == 1
    assert payload["invalid_joy_packets"] == 1


def test_timeout_does_not_cancel_an_active_macro_or_deferred_stop():
    core = session()
    send(core)
    send(core, [6], 10.01)
    core.tick(10.02)
    core.state.request_morphology("rc_car8")
    core.state.begin_macro()
    send(core, [], 10.03)
    send(core, [6], 10.04)
    core.tick(10.05)
    payload = core.tick(11.0)
    assert payload["phase"] == "STRUCTURAL_MACRO"
    assert payload["recording"] and payload["recording_stop_pending"]


def test_shell_config_produces_driver_parameters_without_double_deadzone():
    config = component("config").load_teleop_config(CONFIG_DIR / "smores_teleop.yaml")
    assert config.joy_parameters()["deadzone"] == 0.0
    assert config.joy_parameters()["sticky_buttons"] is False
    assert config.control_rate_hz == 50.0
    assert config.dataset_rate_hz == 25.0
    assert config.joy_topic == "/joy"


@pytest.mark.parametrize("field,value", [
    ("control_rate_hz", 0), ("control_rate_hz", float("nan")),
    ("dataset_rate_hz", -1), ("dataset_rate_hz", float("inf")),
])
def test_invalid_rates_fail_before_shell_startup(tmp_path, field, value):
    mapping = yaml.safe_load((CONFIG_DIR / "smores_teleop.yaml").read_text())
    mapping[field] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(mapping))
    with pytest.raises(ValueError):
        component("config").load_teleop_config(path)


@pytest.mark.parametrize("field,value", [
    ("deadzone", 0.1), ("sticky_buttons", True), ("autorepeat_rate", 0),
    ("device_id", -1), ("device_id", 1.5), ("device_name", None),
    ("topic", ""), ("topic", "relative"),
])
def test_invalid_driver_config_fails_before_shell_startup(tmp_path, field, value):
    mapping = yaml.safe_load((CONFIG_DIR / "smores_teleop.yaml").read_text())
    mapping["joy"][field] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(mapping))
    with pytest.raises(ValueError):
        component("config").load_teleop_config(path)
