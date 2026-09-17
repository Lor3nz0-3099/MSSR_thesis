"""Approved DualSense buttons through actual input edges and native ACKs."""
import importlib.util
from pathlib import Path

import pytest
import yaml

from mssr_expert.teleop.coordinator import RuntimeCoordinator
from mssr_expert.teleop.input import InputConfig, load_input_config
from mssr_expert.teleop.session import TeleopSession


CONFIG = Path(__file__).parents[1] / "config/smores_dualsense.yaml"


def core():
    result = RuntimeCoordinator(TeleopSession(load_input_config(CONFIG)))
    result.session.state.observe_topology("rc_car8")
    send(result, 10)
    ack(result, 10, False)
    result.tick(10)
    return result


def send(result, at, *buttons, r2=0):
    packet = [0] * 21
    for button in buttons:
        packet[button] = 1
    assert result.session.update_joy([0, 0, 0, 0, 0, -r2], packet, at)


def ack(result, at, stopped, request=None):
    result.observe_runtime({
        "schema_version": "mssr.teleop_runtime_status.v1",
        "stamp_monotonic": at, "structure_stopped": stopped,
        "structure_stop_ack": {**request, "applied": True} if request else None,
        "error": None,
    }, at)


def request(result, at):
    return result.tick(at)[1]["structure_stop_request"]


def press(result, at, *, r2=0):
    send(result, at - 0.001, r2=r2)
    send(result, at, 3, r2=r2)
    return request(result, at)


def test_approved_home_and_toggle_bindings():
    config = load_input_config(CONFIG)
    assert config.commands["home"] == "circle"
    assert config.commands.get("estop_toggle") == "triangle"
    assert config.commands["record_toggle"] == "start"


def test_circle_emits_home_and_dpad_does_not_launch_future_macros():
    result = core()
    send(result, 10.01, 1, 11, 12, 13, 14)
    status, payload = result.tick(10.01)
    assert status["controller_input"]["command_events"] == ["home"]
    assert payload["structure_stop_request"] is None
    assert not result.session.state.macro_active


def test_triangle_held_requests_only_one_stop_then_release_press_resumes():
    result = core()
    stop = press(result, 10.01)
    assert stop and stop["active"] is True
    ack(result, 10.02, True, stop)
    send(result, 10.03, 3)
    assert request(result, 10.03) is None
    clear = press(result, 10.04)
    assert clear and clear["active"] is False
    assert result.tick(10.04)[0]["safety"]["authority"] == "ESTOP"


def test_second_press_before_stop_ack_queues_clear_and_third_press_discards_it():
    result = core()
    stop = press(result, 10.01)
    assert stop and stop["active"] is True
    assert press(result, 10.02) == stop
    ack(result, 10.03, True, stop)
    clear = request(result, 10.03)
    assert clear and clear["active"] is False
    new_stop = press(result, 10.04)
    assert new_stop and new_stop["active"] is True
    ack(result, 10.05, False, clear)
    assert request(result, 10.05) == new_stop
    assert result.tick(10.05)[0]["safety"]["authority"] == "ESTOP"


@pytest.mark.parametrize("presses", [2, 3])
def test_multiple_buffered_triangle_presses_preserve_stop_delivery(presses):
    result = core()
    for index in range(presses):
        send(result, 10.01 + index * 0.02, 3)
        send(result, 10.02 + index * 0.02)
    stop = request(result, 10.1)
    assert stop and stop["active"] is True
    ack(result, 10.11, True, stop)
    following = request(result, 10.11)
    if presses == 2:
        assert following and following["active"] is False
    else:
        assert following is None
    assert result.tick(10.11)[0]["safety"]["authority"] == "ESTOP"


def test_resume_waits_actual_ack_and_new_neutral_input():
    result = core()
    stop = press(result, 10.01, r2=0.4)
    assert stop and stop["active"] is True
    ack(result, 10.02, True, stop)
    clear = press(result, 10.03, r2=0.4)
    assert clear and clear["active"] is False
    send(result, 10.04)
    ack(result, 10.05, False, clear)
    assert not result.tick(10.05)[0]["safety"]["motion_enabled"]
    send(result, 10.06, r2=0.4)
    assert not result.tick(10.06)[0]["safety"]["motion_enabled"]
    send(result, 10.07)
    assert result.tick(10.07)[0]["safety"]["motion_enabled"]


def test_triangle_resumes_an_authoritative_native_stop():
    result = core()
    ack(result, 10.01, True)
    clear = press(result, 10.02)
    assert clear and clear["active"] is False


def test_legacy_explicit_estop_dominates_toggle_in_same_packet():
    mapping = yaml.safe_load(CONFIG.read_text())
    mapping["commands"]["estop"] = "ps"
    result = RuntimeCoordinator(TeleopSession(InputConfig.from_mapping(mapping)))
    send(result, 10)
    send(result, 10.01, 3, 5)
    stop = request(result, 10.01)
    assert stop and stop["active"] is True
    ack(result, 10.02, True, stop)
    assert request(result, 10.02) is None


def smoke():
    root = Path(__file__).parents[4]
    spec = importlib.util.spec_from_file_location("rc_button_preflight", root / "scripts/teleop/check_rc_car.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shipped_bindings_pass_preflight_without_launching_runtime():
    assert smoke().preflight(CONFIG).commands.get("estop_toggle") == "triangle"


@pytest.mark.parametrize("estop,resume", [(None, None), ("ps", None), (None, "share")])
def test_preflight_still_rejects_incomplete_legacy_stop_pair(tmp_path, estop, resume):
    mapping = yaml.safe_load(CONFIG.read_text())
    mapping["commands"].pop("estop_toggle", None)
    mapping["commands"].update(home="circle", estop=estop, resume=resume)
    path = tmp_path / "incomplete.yaml"
    path.write_text(yaml.safe_dump(mapping))
    with pytest.raises(ValueError, match="deferred"):
        smoke().preflight(path)
