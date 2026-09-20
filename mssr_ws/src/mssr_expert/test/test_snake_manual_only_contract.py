"""Contract for the final Snake8 manual-only teleoperation mode."""
from pathlib import Path

import yaml

from test_snake_teleop import ENABLED, graph, runtime, sample


CONFIG = Path(__file__).parents[1] / "config"


def ready():
    snake = runtime()
    assert snake.observe_graph(graph().to_dict(), now=1.0)
    return snake


def test_snake_starts_directly_in_manual_mode_with_head_selected():
    snake = ready()

    out = snake.step(sample(), safety=ENABLED, now=1.0)

    assert out.actions.intent["control_mode"] == "manual"
    assert out.actions.intent["selected_module_id"] == "physical_v7"
    assert out.actions.intent["selected_role"] == "snake_head"

    # The removed head-led controller must leave no latent trajectory state.
    assert "head_target_m" not in out.actions.intent
    assert "backbone_m" not in out.actions.intent
    assert "vertical_offset_m" not in out.actions.intent


def test_dpad_down_is_not_required_for_snake_manual_control():
    mapping = yaml.safe_load(
        (CONFIG / "smores_dualsense.yaml").read_text(encoding="utf-8")
    )

    assert mapping["commands"]["override"] is None
    assert mapping["commands"]["previous_module"] == "l1"
    assert mapping["commands"]["next_module"] == "r1"


def test_right_stick_pan_controls_selected_module_without_mode_toggle():
    snake = ready()

    snake.step(sample(), safety=ENABLED, now=1.0)
    out = snake.step(
        sample(right_x=1.0),
        safety=ENABLED,
        now=1.1,
    )

    assert out.actions.intent["control_mode"] == "manual"
    assert out.actions.intent["selected_module_id"] == "physical_v7"
    assert out.posture.goal is not None
    assert out.posture.goal.primitive == "rotate_pan_by"
    assert out.posture.goal.module_ids == ("physical_v7",)


def test_right_stick_tilt_controls_only_selected_module():
    snake = ready()

    snake.step(sample(), safety=ENABLED, now=1.0)
    out = snake.step(
        sample(right_y=1.0),
        safety=ENABLED,
        now=1.1,
    )

    assert out.posture.goal is not None
    assert out.posture.goal.primitive == "set_tilt"
    assert out.posture.goal.module_ids == ("physical_v7",)

    targeted = [
        module
        for module, command in out.actions.module_actions.items()
        if "tilt_target_rad" in command
    ]
    assert targeted == ["physical_v7"]


def test_r1_and_l1_cycle_selected_module_without_mode_switch():
    snake = ready()

    first = snake.step(sample(), safety=ENABLED, now=1.0)
    assert first.actions.intent["selected_module_id"] == "physical_v7"

    nxt = snake.step(
        sample(command_events=("next_module",)),
        safety=ENABLED,
        now=1.1,
    )
    assert nxt.actions.intent["selected_module_id"] == "physical_v6"

    prev = snake.step(
        sample(command_events=("previous_module",)),
        safety=ENABLED,
        now=1.2,
    )
    assert prev.actions.intent["selected_module_id"] == "physical_v7"


def test_r2_l2_keep_all_eight_modules_driving_during_manual_joint_control():
    snake = ready()

    snake.step(sample(), safety=ENABLED, now=1.0)

    forward = snake.step(
        sample(r2=0.5, right_x=1.0),
        safety=ENABLED,
        now=1.1,
    )

    assert len(forward.actions.module_actions) == 8
    assert all(
        abs(command.get("vx", 0.0)) > 0
        for command in forward.actions.module_actions.values()
    )

    # Retire the PAN primitive before changing axis.
    goal = forward.posture.goal
    assert goal is not None

    switched = snake.step(
        sample(command_events=("next_module",), l2=0.5),
        safety=ENABLED,
        now=1.2,
    )

    assert len(switched.actions.module_actions) == 8
    assert all(
        abs(command.get("vx", 0.0)) > 0
        for command in switched.actions.module_actions.values()
    )
    assert switched.actions.intent["selected_module_id"] == "physical_v6"
