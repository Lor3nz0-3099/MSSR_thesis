"""Input/state coordination and strict shell configuration without ROS."""
import importlib
import importlib.util
import math
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
    # Isolate synthetic bindings from the approved physical toggle mapping.
    mapping["commands"].update(select_rc="cross", select_snake="triangle",
                               select_mm8="circle", home="square",
                               estop="ps", resume="share", estop_toggle=None)
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
    assert config.dataset_rate_hz == 10.0
    assert config.joy_topic == "/joy"


def test_shipped_camera_radius_matches_initial_rc_assembly_view():
    config = component("config").load_teleop_config(CONFIG_DIR / "smores_teleop.yaml")
    # Existing scene at the CLI's default 0.34 m spawn radius.
    startup_distance = math.dist((0.7956, -0.7344, 0.5202), (0.0, 0.0, 0.03))
    assert getattr(config, "camera_radius_m", None) == pytest.approx(startup_distance)


@pytest.mark.parametrize("radius", [0.6, 1.4])
def test_configured_camera_radius_reaches_runtime_intent_and_orbit_geometry(tmp_path, radius):
    mapping = yaml.safe_load((CONFIG_DIR / "smores_teleop.yaml").read_text())
    mapping["camera"] = {"radius_m": radius}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(mapping))
    config = component("config").load_teleop_config(path)
    assert getattr(config, "camera_radius_m", None) == radius
    controller = component("camera").CameraController(radius_m=config.camera_radius_m)
    core = component("coordinator").RuntimeCoordinator(session(), camera=controller)
    send(core.session)
    _, request = core.tick(10.0)
    assert request["camera"]["radius_m"] == radius
    eye, target = controller.step(core.session.input.snapshot(10.0), 0.0).view((1, 2, 0.1))
    assert math.dist(eye, target) == pytest.approx(radius)


@pytest.mark.parametrize("radius", [0, -1, float("nan"), float("inf"), True, 101])
def test_invalid_camera_radius_fails_before_shell_startup(tmp_path, radius):
    mapping = yaml.safe_load((CONFIG_DIR / "smores_teleop.yaml").read_text())
    mapping["camera"] = {"radius_m": radius}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(mapping))
    with pytest.raises(ValueError, match="camera"):
        component("config").load_teleop_config(path)


def test_legacy_configuration_without_camera_preserves_old_radius(tmp_path):
    mapping = yaml.safe_load((CONFIG_DIR / "smores_teleop.yaml").read_text())
    mapping.pop("camera", None)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(mapping))
    config = component("config").load_teleop_config(path)
    assert getattr(config, "camera_radius_m", None) == 2.0


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



def test_different_morphology_selection_emits_one_shot_structural_macro_request():
    """T5: selection requests a macro, but launch owns STRUCTURAL_MACRO entry."""

    core = session()

    # Controller online and verified current morphology.
    send(core, at=10.0)
    core.state.observe_topology("rc_car8")
    baseline = core.tick(10.01)

    assert baseline["authority"] == "TELEOP"
    assert baseline["active_controller"] == "rc_car8"

    # Synthetic test binding: triangle -> select_snake.
    send(core, [3], 10.10)
    payload = core.tick(10.11)

    assert payload["requested_morphology"] == "snake8"

    # New T5 contract:
    # the input edge exposes one structural launch request.
    assert payload["structural_macro_request"] == "snake8"

    # Merely requesting the launcher must NOT claim macro authority yet.
    assert payload["authority"] == "TELEOP"
    assert payload["macro_active"] is False

    # It is an edge/event, not a continuously repeated request.
    following = core.tick(10.12)
    assert following["structural_macro_request"] is None



def test_detected_morphology_without_runtime_controller_has_no_teleop_authority():
    state = component("state").TeleopState(
        controller_morphologies={"rc_car8"},
    )

    state.start_ready()
    state.set_connected(True)

    # The physical topology is authoritative and must still be reported.
    state.observe_topology("snake8")

    assert state.detected_morphology == "snake8"

    # But no Snake teleop runtime exists yet.
    assert state.active_controller is None
    assert state.authority == "NONE"

    # Structural intent must remain available even without locomotion authority.
    assert state.request_morphology("rc_car8")
    assert state.requested_morphology == "rc_car8"


def test_detected_morphology_with_runtime_controller_claims_teleop_authority():
    state = component("state").TeleopState(
        controller_morphologies={"rc_car8"},
    )

    state.start_ready()
    state.set_connected(True)
    state.observe_topology("rc_car8")

    assert state.detected_morphology == "rc_car8"
    assert state.active_controller == "rc_car8"
    assert state.authority == "TELEOP"



def test_session_propagates_available_controller_morphologies():
    mapping = yaml.safe_load(
        (CONFIG_DIR / "smores_dualsense.yaml").read_text()
    )

    core = component("session").TeleopSession(
        InputConfig.from_mapping(mapping),
        controller_morphologies={"rc_car8"},
    )

    send(core, at=10.0)
    core.state.observe_topology("snake8")

    payload = core.tick(10.01)

    assert payload["detected_morphology"] == "snake8"
    assert payload["active_controller"] is None
    assert payload["authority"] == "NONE"

    core.state.observe_topology("rc_car8")
    payload = core.tick(10.02)

    assert payload["detected_morphology"] == "rc_car8"
    assert payload["active_controller"] == "rc_car8"
    assert payload["authority"] == "TELEOP"
