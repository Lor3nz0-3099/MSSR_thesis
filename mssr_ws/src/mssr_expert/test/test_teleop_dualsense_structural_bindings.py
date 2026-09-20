"""DualSense D-pad bindings for T5 morphology selection."""

from pathlib import Path

import pytest

from mssr_expert.teleop.input import DualSenseInput, load_input_config


CONFIG = (
    Path(__file__).parents[1]
    / "config"
    / "smores_dualsense.yaml"
)


def test_structural_morphology_commands_use_three_distinct_dpad_buttons():
    config = load_input_config(CONFIG)

    assert config.commands["select_rc"] == "dpad_left"
    assert config.commands["select_snake"] == "dpad_up"
    assert config.commands["select_mm8"] == "dpad_right"

    assert config.commands["select_rc"] != config.commands["select_snake"]
    assert config.commands["select_rc"] != config.commands["select_mm8"]
    assert config.commands["select_snake"] != config.commands["select_mm8"]

    assert config.commands["override"] == "dpad_down"
    assert config.commands["previous_module"] == "l1"
    assert config.commands["next_module"] == "r1"


@pytest.mark.parametrize(
    ("button_name", "expected_command"),
    (
        ("dpad_left", "select_rc"),
        ("dpad_up", "select_snake"),
        ("dpad_right", "select_mm8"),
    ),
)
def test_dpad_press_emits_expected_structural_command(
    button_name,
    expected_command,
):
    config = load_input_config(CONFIG)
    controller = DualSenseInput(config)

    axis_values = [0.0] * 6
    neutral_buttons = [0] * 15

    # First Joy packet only establishes connectivity/state.
    assert controller.update(
        axis_values,
        neutral_buttons,
        received_at=10.0,
    )
    assert controller.snapshot(10.01).command_events == ()

    pressed = list(neutral_buttons)
    pressed[config.buttons[button_name]] = 1

    assert controller.update(
        axis_values,
        pressed,
        received_at=10.1,
    )

    snapshot = controller.snapshot(10.11)

    assert snapshot.command_events == (expected_command,)
