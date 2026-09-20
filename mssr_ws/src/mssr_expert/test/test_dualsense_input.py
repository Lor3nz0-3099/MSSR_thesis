"""Behavioral contracts for ROS Joy normalization before actuator control."""
from copy import deepcopy
from pathlib import Path

import pytest

from mssr_expert.teleop.input import DualSenseInput, InputConfig, load_input_config


@pytest.fixture
def mapping():
    return {
        "deadzone": 0.1,
        "trigger_deadzone": 0.05,
        "disconnect_timeout_s": 0.5,
        "axes": {
            "left_x": {"index": 0, "sign": -1},
            "left_y": {"index": 1, "sign": 1},
            "right_x": {"index": 2, "sign": -1},
            "right_y": {"index": 3, "sign": 1},
        },
        "triggers": {
            "l2": {"index": 4, "released": 0.0, "pressed": -1.0},
            "r2": {"index": 5, "released": 0.0, "pressed": -1.0},
        },
        "buttons": {"cross": 0, "circle": 1, "square": 2,
                    "triangle": 3, "start": 6, "l1": 9, "r1": 10},
        "commands": {"record_toggle": "start", "select_rc": None,
                     "select_snake": None, "select_mm8": None,
                     "home": None, "estop": None, "resume": None,
                     "override": None, "previous_module": None,
                     "next_module": None, "manual_pan_positive": None,
                     "manual_pan_negative": None,
                     "manual_tilt_positive": None,
                     "manual_tilt_negative": None},
    }


@pytest.fixture
def reader(mapping):
    return DualSenseInput(InputConfig.from_mapping(mapping))


def packet(reader, *, axes=None, pressed=(), at=10.0):
    buttons = [0] * 15
    for index in pressed:
        buttons[index] = 1
    return reader.update([0.0] * 6 if axes is None else axes, buttons, at)


def test_no_input_is_disconnected_without_a_receipt_timestamp(reader):
    sample = reader.snapshot(10.0)
    assert not sample.connected
    assert sample.last_message_at is None
    assert sample.command_edges == frozenset()


@pytest.mark.parametrize("raw,want", [(0, 0), (-1, 1), (-0.525, 0.5),
                                          (-0.03, 0), (0.2, 0), (-1.2, 1)])
def test_sdl_triggers_have_zero_release_and_analog_travel(reader, raw, want):
    assert packet(reader, axes=[0, 0, 0, 0, raw, raw])
    sample = reader.snapshot(10.01)
    assert sample.l2 == pytest.approx(want)
    assert sample.r2 == pytest.approx(want)


def test_generic_joy_trigger_endpoints_are_configurable(mapping):
    mapping["triggers"]["l2"].update(released=1.0, pressed=-1.0)
    mapping["triggers"]["r2"].update(released=1.0, pressed=-1.0)
    reader = DualSenseInput(InputConfig.from_mapping(mapping))
    packet(reader, axes=[0, 0, 0, 0, 1, -1])
    sample = reader.snapshot(10.01)
    assert (sample.l2, sample.r2) == (0.0, 1.0)


@pytest.mark.parametrize("raw,want", [(0.1, 0), (-0.09, 0), (0.55, -0.5),
                                          (-1, 1), (2, -1)])
def test_sticks_rescale_deadzone_apply_sign_and_clamp(reader, raw, want):
    packet(reader, axes=[raw, raw, raw, raw, 0, 0])
    sample = reader.snapshot(10.01)
    assert sample.left_x == pytest.approx(want)
    assert sample.right_x == pytest.approx(want)
    assert sample.left_y == pytest.approx(-want)
    assert sample.right_y == pytest.approx(-want)


def test_start_edges_toggle_once_and_survive_intermediate_joy_messages(reader):
    packet(reader)
    packet(reader, pressed=[6], at=10.02)
    packet(reader, pressed=[6], at=10.03)
    packet(reader, at=10.04)
    sample = reader.snapshot(10.05)
    assert sample.rising_edges == frozenset({"start"})
    assert sample.command_edges == frozenset({"record_toggle"})
    assert "start" not in sample.buttons
    assert reader.snapshot(10.06).command_edges == frozenset()
    packet(reader, pressed=[6], at=10.07)
    assert reader.snapshot(10.08).command_edges == frozenset({"record_toggle"})


def test_multiple_button_edges_are_preserved_until_consumed(reader):
    packet(reader)
    packet(reader, pressed=[0], at=10.01)
    packet(reader, pressed=[9], at=10.02)
    sample = reader.snapshot(10.03)
    assert sample.rising_edges == frozenset({"cross", "l1"})
    assert sample.command_edges == frozenset()


def test_repeated_start_edges_keep_order_and_multiplicity(reader):
    packet(reader)
    packet(reader, pressed=[6], at=10.01)
    packet(reader, at=10.02)
    packet(reader, pressed=[6], at=10.03)
    sample = reader.snapshot(10.04)
    assert sample.command_events == ("record_toggle", "record_toggle")
    assert reader.snapshot(10.05).command_events == ()


def test_button_event_retains_modifier_context_even_after_release(mapping):
    mapping["commands"].update(home="square", select_rc="cross")
    reader = DualSenseInput(InputConfig.from_mapping(mapping))
    packet(reader)
    packet(reader, pressed=[2], at=10.01)
    packet(reader, pressed=[0, 2], at=10.02)
    packet(reader, at=10.03)
    events = reader.snapshot(10.04).button_events
    assert events[1].button == "cross"
    assert events[1].pressed_buttons == frozenset({"cross", "square"})


def test_startup_held_start_does_not_open_recording(reader):
    packet(reader, pressed=[6])
    sample = reader.snapshot(10.01)
    assert sample.buttons == frozenset({"start"})
    assert sample.command_edges == frozenset()
    packet(reader, at=10.02)
    packet(reader, pressed=[6], at=10.03)
    assert reader.snapshot(10.04).command_edges == frozenset({"record_toggle"})


def test_timeout_uses_receipt_time_and_drops_unconsumed_edges(reader):
    packet(reader)
    packet(reader, pressed=[6], at=10.02)
    assert reader.snapshot(10.51).connected
    assert not reader.snapshot(10.52).connected
    assert reader.snapshot(10.53).command_edges == frozenset()
    assert reader.snapshot(10.53).last_message_at == 10.02


def test_reconnect_with_held_command_requires_release_and_repress(reader):
    packet(reader)
    packet(reader, pressed=[6], at=11.0)
    assert reader.snapshot(11.01).connected
    assert reader.snapshot(11.01).command_edges == frozenset()
    packet(reader, at=11.02)
    packet(reader, pressed=[6], at=11.03)
    assert reader.snapshot(11.04).command_edges == frozenset({"record_toggle"})


def test_invalid_packet_does_not_keep_controller_connected(reader):
    packet(reader)
    assert not reader.update([0] * 5, [0] * 15, 10.4)
    assert not reader.snapshot(10.5).connected
    assert reader.snapshot(10.5).last_message_at == 10.0


@pytest.mark.parametrize("axes,buttons", [
    ([0] * 5, [0] * 15), ([0] * 6, [0] * 6),
    ([0, 0, float("nan"), 0, 0, 0], [0] * 15),
    ([0, 0, 0, 0, float("inf"), 0], [0] * 15),
    ([0] * 6, [0] * 6 + [2] + [0] * 8),
    ([0] * 6, [0] * 6 + [float("nan")] + [0] * 8),
])
def test_malformed_data_is_rejected_atomically(reader, axes, buttons):
    packet(reader, axes=[0, 0, -1, 0, 0, -1])
    assert not reader.update(axes, buttons, 10.2)
    sample = reader.snapshot(10.21)
    assert sample.right_x == 1.0
    assert sample.r2 == 1.0
    assert sample.last_message_at == 10.0


def test_out_of_order_or_nonfinite_receipt_timestamp_is_rejected(reader):
    packet(reader)
    for stamp in (9.0, float("nan"), float("inf")):
        assert not packet(reader, pressed=[6], at=stamp)
    assert reader.snapshot(10.1).command_edges == frozenset()


def test_snapshot_is_immutable_and_configuration_is_copied(reader, mapping):
    packet(reader, axes=[0, 0, -1, 0, 0, 0])
    sample = reader.snapshot(10.1)
    with pytest.raises(AttributeError):
        sample.right_x = 0
    config = InputConfig.from_mapping(mapping)
    mapping["axes"]["right_x"]["sign"] = 1
    other = DualSenseInput(config)
    packet(other, axes=[0, 0, -1, 0, 0, 0])
    assert other.snapshot(10.1).right_x == 1


def test_semantic_face_assignment_requires_explicit_configuration(mapping):
    configured = deepcopy(mapping)
    configured["commands"]["select_snake"] = "triangle"
    reader = DualSenseInput(InputConfig.from_mapping(configured))
    packet(reader)
    packet(reader, pressed=[3], at=10.1)
    assert reader.snapshot(10.2).command_edges == frozenset({"select_snake"})


@pytest.mark.parametrize("key,value", [
    ("deadzone", -0.1), ("deadzone", 1),
    ("trigger_deadzone", float("nan")),
    ("disconnect_timeout_s", 0), ("disconnect_timeout_s", float("inf")),
])
def test_invalid_safety_config_fails_before_any_input(mapping, key, value):
    mapping[key] = value
    with pytest.raises(ValueError):
        InputConfig.from_mapping(mapping)


@pytest.mark.parametrize("kind", ["negative_index", "fractional_index", "sign",
                                  "trigger_endpoints", "unknown_button",
                                  "duplicate_command", "start_reassigned",
                                  "overlapping_axis", "missing_stick"])
def test_invalid_bindings_fail_before_any_input(mapping, kind):
    if kind == "negative_index":
        mapping["axes"]["right_x"]["index"] = -1
    elif kind == "fractional_index":
        mapping["buttons"]["start"] = 1.5
    elif kind == "sign":
        mapping["axes"]["right_y"]["sign"] = 0
    elif kind == "trigger_endpoints":
        mapping["triggers"]["r2"]["pressed"] = 0
    elif kind == "unknown_button":
        mapping["commands"]["estop"] = "mystery"
    elif kind == "duplicate_command":
        mapping["commands"].update(estop="cross", resume="cross")
    elif kind == "start_reassigned":
        mapping["commands"]["record_toggle"] = "cross"
    elif kind == "overlapping_axis":
        mapping["triggers"]["r2"]["index"] = 2
    else:
        del mapping["axes"]["left_x"]
    with pytest.raises(ValueError):
        InputConfig.from_mapping(mapping)


def test_shipped_configuration_enables_only_approved_commands():
    path = Path(__file__).parents[1] / "config/smores_dualsense.yaml"
    reader = DualSenseInput(load_input_config(path))
    packet(reader)
    packet(
        reader,
        pressed=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        at=10.1,
    )

    assert reader.snapshot(10.2).command_edges == frozenset({
        "record_toggle",
        "home",
        "estop_toggle",
        "select_rc",
        "select_snake",
        "select_mm8",
        "previous_module",
        "next_module",
    })


def test_configuration_loader_reports_malformed_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("[not: a mapping", encoding="utf-8")
    with pytest.raises(ValueError):
        load_input_config(path)
