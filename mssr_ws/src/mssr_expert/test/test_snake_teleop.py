"""T6a head-led Snake8 control at the pure runtime boundary."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from mssr_expert.behaviors.morphology_library import MorphologyLibrary
from mssr_expert.graph.serialization import attributed_graph_from_dict, load_attributed_graph
from mssr_expert.teleop.rc_car import load_geometry
from mssr_expert.teleop.safety import SafetyDecision
from mssr_expert.teleop.snake import SnakeActions, SnakePostureTransport, SnakeRuntime
from smores_ep.primitives.file_channel import ActionFileChannel


CONFIG = Path(__file__).parents[1] / "config"
ENABLED = SafetyDecision("TELEOP", True, False, True)
STOPPED = SafetyDecision("ESTOP", False, True, False)
MACRO = SafetyDecision("STRUCTURAL_MACRO", False, False, False)


def graph(*, stamp=1.0, missing=False):
    payload = json.loads((CONFIG / "smores_snake8.json").read_text())
    spacing = load_geometry().top_to_bottom_spacing_m
    ids = {node["node_id"]: "physical_" + node["node_id"] for node in payload["nodes"]}
    for i, node in enumerate(payload["nodes"]):
        node["node_id"] = ids[node["node_id"]]
        node["attributes"].update(
            node_type="physical_module", is_target_node=False,
            pose={"position": [i * spacing, 0.0, 0.05],
                  "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]},
            actuators={name: {"position_rad": 0.0, "lower_limit_rad": -1.2,
                              "upper_limit_rad": 1.2}
                       for name in ("pan", "tilt")},
        )
    for edge in payload["edges"]:
        edge["module_a_id"] = ids[edge["module_a_id"]]
        edge["module_b_id"] = ids[edge["module_b_id"]]
        edge["attributes"].update(is_target_edge=False, is_attached=True,
                                  relation_type="current_connection")
    if missing:
        payload["edges"].pop()
    payload["stamp"] = stamp
    return attributed_graph_from_dict(payload)


def runtime():
    configuration = yaml.safe_load((CONFIG / "smores_teleop.yaml").read_text())["snake"]
    return SnakeRuntime(
        MorphologyLibrary.load(CONFIG / "smores_morphology_behaviors.json"),
        load_attributed_graph(CONFIG / "smores_snake8.json"),
        geometry=load_geometry(),
        **configuration,
    )


def sample(**values):
    return SimpleNamespace(**{**dict(left_x=0.0, left_y=0.0, right_x=0.0,
                                    right_y=0.0, r2=0.0, l2=0.0,
                                    command_events=()), **values})


def test_physical_assignment_and_neutral_are_safe():
    snake = runtime()
    assert snake.observe_graph(json.loads(json.dumps(graph().to_dict())), now=1.0)
    out = snake.step(sample(), safety=ENABLED, now=1.0)
    assert out.actions.module_roles["physical_v7"] == "snake_head"
    assert all(all(abs(value) < 1e-12 for value in command.values())
               for command in out.actions.module_actions.values())
    assert out.actions.joint_targets == ()
    assert snake.latest_graph is not None


def test_snake_ignores_lateral_stick_and_sends_no_pan_primitive():
    snake = runtime()
    snake.observe_graph(graph().to_dict(), now=1.0)
    neutral = snake.step(sample(), safety=ENABLED, now=1.0)
    for tick in range(1, 16):
        now = 1.0 + 0.1 * tick
        snake.observe_graph(graph(stamp=1.0 + tick).to_dict(), now=now)
        out = snake.step(sample(right_x=1.0), safety=ENABLED, now=now)
        assert out.actions.intent["head_target_m"][1] == pytest.approx(
            neutral.actions.intent["head_target_m"][1])
        assert out.actions.intent["lateral_offset_m"] == 0.0
        assert all(joint == "tilt" for _, joint, _, _ in out.actions.joint_targets)
        assert all("pan_target_rad" not in command
                   for command in out.actions.module_actions.values())
        assert out.posture.goal is None


def test_snake_right_stick_up_and_down_changes_height_with_tilt_only():
    snake = runtime()
    snake.observe_graph(graph().to_dict(), now=1.0)
    snake.step(sample(), safety=ENABLED, now=1.0)
    for tick in range(1, 16):
        now = 1.0 + 0.1 * tick
        snake.observe_graph(graph(stamp=1.0 + tick).to_dict(), now=now)
        raised = snake.step(sample(right_y=1.0), safety=ENABLED, now=now)
    raised_z = raised.actions.intent["head_target_m"][2]
    assert raised_z > 0.05
    for tick in range(16, 31):
        now = 1.0 + 0.1 * tick
        snake.observe_graph(graph(stamp=1.0 + tick).to_dict(), now=now)
        lowered = snake.step(sample(right_y=-1.0, right_x=1.0),
                             safety=ENABLED, now=now)
        assert all(joint == "tilt" for _, joint, _, _ in lowered.actions.joint_targets)
        assert all("pan_target_rad" not in command
                   for command in lowered.actions.module_actions.values())
    assert lowered.actions.intent["head_target_m"][2] < raised_z


def test_manual_module_mode_keeps_all_wheels_and_cycles_from_head():
    snake = runtime()
    snake.observe_graph(graph().to_dict(), now=1.0)
    snake.step(sample(), safety=ENABLED, now=1.0)
    entered = snake.step(sample(command_events=("override",), r2=0.5),
                         safety=ENABLED, now=1.1)
    assert entered.actions.intent["control_mode"] == "single_module"
    assert entered.actions.intent["selected_module_id"] == "physical_v7"
    assert all(command["vx"] > 0 for command in entered.actions.module_actions.values())
    next_module = snake.step(sample(command_events=("next_module",), r2=0.5),
                             safety=ENABLED, now=1.2)
    assert next_module.actions.intent["selected_module_id"] == "physical_v6"
    assert all(command["vx"] > 0 for command in next_module.actions.module_actions.values())
    previous = snake.step(sample(command_events=("previous_module",), l2=0.5),
                          safety=ENABLED, now=1.3)
    assert previous.actions.intent["selected_module_id"] == "physical_v7"
    assert all(command["vx"] < 0 for command in previous.actions.module_actions.values())
    exited = snake.step(sample(command_events=("override",), r2=0.5),
                        safety=ENABLED, now=1.4)
    assert exited.actions.intent["control_mode"] == "head_led"
    assert all(command["vx"] > 0 for command in exited.actions.module_actions.values())


def test_manual_pan_targets_only_selected_module_while_all_wheels_drive():
    snake = runtime()
    snake.observe_graph(graph().to_dict(), now=1.0)
    snake.step(sample(), safety=ENABLED, now=1.0)
    snake.step(sample(command_events=("override",)), safety=ENABLED, now=1.1)
    out = snake.step(sample(right_x=1.0, r2=0.5), safety=ENABLED, now=1.2)
    assert out.posture.goal is not None
    assert out.posture.goal.primitive == "rotate_pan_by"
    assert out.posture.goal.module_ids == ("physical_v7",)
    assert all(command["vx"] > 0 for command in out.actions.module_actions.values())
    assert "pan_target_rad" in out.actions.module_actions["physical_v7"]
    assert sum("pan_target_rad" in command for command in out.actions.module_actions.values()) == 1
    assert "pan_target_rad" not in json.loads(out.envelope)["locomotion"]["physical_v7"]


def test_manual_tilt_and_auto_to_manual_transition_cancel_old_goal():
    snake = runtime()
    snake.observe_graph(graph().to_dict(), now=1.0)
    snake.step(sample(), safety=ENABLED, now=1.0)
    automatic = snake.step(sample(right_y=1.0), safety=ENABLED, now=1.1)
    assert automatic.posture.goal is not None
    snake.step(sample(), safety=ENABLED, now=1.12)
    entered = snake.step(sample(command_events=("override",), r2=0.5),
                         safety=ENABLED, now=1.14)
    assert automatic.posture.goal.goal_id in entered.posture.cancel_goal_ids
    assert entered.posture.goal is None
    assert all(command["vx"] > 0 for command in entered.actions.module_actions.values())


def test_manual_module_switch_retires_old_pan_before_tilt_on_next_module():
    snake = runtime()
    snake.observe_graph(graph().to_dict(), now=1.0)
    snake.step(sample(), safety=ENABLED, now=1.0)
    snake.step(sample(command_events=("override",)), safety=ENABLED, now=1.1)
    pan = snake.step(sample(right_x=1.0, r2=0.5), safety=ENABLED, now=1.2)
    goal = pan.posture.goal
    assert goal is not None
    switched = snake.step(sample(command_events=("next_module",), l2=0.5),
                          safety=ENABLED, now=1.22)
    assert switched.actions.intent["selected_module_id"] == "physical_v6"
    assert goal.goal_id in switched.posture.cancel_goal_ids
    assert all(command["vx"] < 0 for command in switched.actions.module_actions.values())
    assert snake.step(sample(right_y=1.0, l2=0.5), safety=ENABLED, now=1.24).posture.goal is None
    snake.observe_status({"goal_id": goal.goal_id, "primitive": "rotate_pan_by",
                          "state": "canceled", "module_ids": ["physical_v7"]})
    snake.step(sample(right_y=1.0, l2=0.5), safety=ENABLED, now=1.26)
    tilt = snake.step(sample(right_y=1.0, l2=0.5), safety=ENABLED, now=1.36)
    assert tilt.posture.goal is not None
    assert tilt.posture.goal.primitive == "set_tilt"
    assert tilt.posture.goal.module_ids == ("physical_v6",)
    assert all(command["vx"] < 0 for command in tilt.actions.module_actions.values())


def test_head_target_integrates_height_and_arc_length_body_follows():
    snake = runtime()
    assert snake.observe_graph(graph().to_dict(), now=1.0)
    first = snake.step(sample(), safety=ENABLED, now=1.0)
    moved = snake.step(sample(r2=0.5, right_x=1.0, right_y=1.0),
                       safety=ENABLED, now=1.1)
    head = moved.actions.intent["head_target_m"]
    assert head[0] > first.actions.intent["head_target_m"][0]
    assert head[1] == pytest.approx(first.actions.intent["head_target_m"][1])
    assert head[2] > first.actions.intent["head_target_m"][2]
    assert len(moved.actions.intent["backbone_m"]) == 8
    assert moved.actions.intent["backbone_m"][-1] == pytest.approx(head)
    held = snake.step(sample(), safety=ENABLED, now=1.2)
    assert held.actions.intent["head_target_m"] == pytest.approx(head)
    assert all(command.get("yaw_rate", 0.0) == 0.0 for command in moved.actions.module_actions.values())
    assert all(command.get("vx", 0.0) >= 0.0 for command in moved.actions.module_actions.values())


def test_same_module_wheels_are_paired_and_release_stops_them_immediately():
    snake = runtime()
    snake.observe_graph(graph().to_dict(), now=1.0)
    moving = snake.step(sample(r2=0.7), safety=ENABLED, now=1.1)
    assert all(command["vx"] > 0 and command["yaw_rate"] == 0
               for command in moving.actions.module_actions.values())
    stopped = snake.step(sample(), safety=ENABLED, now=1.2)
    assert all(command["vx"] == 0 for command in stopped.actions.module_actions.values())


def test_l2_retraces_straight_path_without_flipping_wheels_or_bending_body():
    snake = runtime()
    snake.observe_graph(graph().to_dict(), now=1.0)
    snake.step(sample(), safety=ENABLED, now=1.0)
    for tick in range(1, 21):
        now = 1.0 + 0.1 * tick
        snake.observe_graph(graph(stamp=1.0 + tick).to_dict(), now=now)
        snake.step(sample(r2=0.5), safety=ENABLED, now=now)

    previous_head_x = snake.step(sample(), safety=ENABLED, now=3.01).actions.intent["head_target_m"][0]
    for tick in range(1, 121):
        now = 3.01 + 0.1 * tick
        snake.observe_graph(graph(stamp=21.0 + tick).to_dict(), now=now)
        moving = snake.step(sample(l2=0.5), safety=ENABLED, now=now)
        intent = moving.actions.intent
        assert all(command["vx"] < 0 for command in moving.actions.module_actions.values())
        assert intent["head_target_m"][0] < previous_head_x
        assert all(a[0] < b[0] for a, b in zip(intent["backbone_m"], intent["backbone_m"][1:]))
        assert moving.actions.joint_targets == ()
        previous_head_x = intent["head_target_m"][0]


def test_stale_or_invalid_graph_and_authority_change_emit_no_snake_motion():
    snake = runtime()
    snake.observe_graph(graph().to_dict(), now=1.0)
    assert snake.step(sample(r2=1.0), safety=MACRO, now=1.1).envelope is None
    assert snake.step(sample(r2=1.0), safety=STOPPED, now=1.2).actions.module_actions == {}
    assert snake.step(sample(r2=1.0), safety=ENABLED, now=1.8).envelope is None
    assert not snake.observe_graph(graph(stamp=2.0, missing=True).to_dict(), now=1.9)
    assert snake.step(sample(r2=1.0), safety=ENABLED, now=1.9).envelope is None


def test_effective_actions_and_morphology_are_in_existing_envelope():
    snake = runtime()
    snake.observe_graph(graph().to_dict(), now=1.0)
    out = snake.step(sample(r2=0.5), safety=ENABLED, now=1.1)
    assert out.envelope is not None
    payload = json.loads(out.envelope)
    assert payload["expert"]["active_primitive"] == "snake8"
    assert payload["locomotion"] == out.actions.module_actions
    assert out.actions.module_roles["physical_v0"] == "snake_tail"
    parsed, diagnostics = ActionFileChannel._parse(out.envelope)
    assert diagnostics.phase == "snake8_teleop"
    assert all(command.angular_z_rad_s == 0 for command in parsed.values())


def test_home_softly_reduces_vertical_target():
    snake = runtime()
    snake.observe_graph(graph().to_dict(), now=1.0)
    snake.step(sample(), safety=ENABLED, now=1.0)
    before = snake.step(sample(right_x=1.0, right_y=1.0), safety=ENABLED, now=1.1)
    after = snake.step(sample(command_events=("home",)), safety=ENABLED, now=1.2)
    assert after.actions.intent["head_target_m"][1] == pytest.approx(0.0)
    assert 0.05 < after.actions.intent["head_target_m"][2] < before.actions.intent["head_target_m"][2]


def test_backbone_replays_vertical_head_path_with_tilt_only():
    snake = runtime()
    snake.observe_graph(graph().to_dict(), now=1.0)
    snake.step(sample(), safety=ENABLED, now=1.0)
    outputs = []
    for i in range(1, 25):
        snake.observe_graph(graph(stamp=i + 1.0).to_dict(), now=1.0 + 0.1 * i)
        outputs.append(snake.step(sample(r2=1.0, right_x=1.0 if i < 10 else 0.0,
                                         right_y=1.0 if i >= 10 else 0.0),
                                  safety=ENABLED, now=1.0 + 0.1 * i))
    assert any(any(joint == "tilt" for _, joint, _, _ in out.actions.joint_targets)
               for out in outputs)
    assert all(all(joint == "tilt" for _, joint, _, _ in out.actions.joint_targets)
               for out in outputs)
    final = outputs[-1].actions.intent["backbone_m"]
    assert all(abs(point[1]) < 1e-9 for point in final)
    assert final[7][2] > final[6][2]


def test_posture_transport_runs_tilt_on_two_modules():
    transport = SnakePostureTransport()
    actions = SnakeActions(
        joint_targets=(("physical_v5", "tilt", 0.4, 0.1),
                       ("physical_v6", "tilt", -0.3, -0.1)),
        allow_joint_updates=True,
    )
    first = transport.step(actions, now=1.0)
    assert first.goal.primitive == "set_tilt"
    assert first.goal.parameters["angle_rad"] == pytest.approx(0.4)
    assert transport.step(actions, now=1.01).goal is None
    transport.observe({"goal_id": first.goal.goal_id,
                       "primitive": "set_tilt", "state": "accepted",
                       "module_ids": ["physical_v5"]})
    second = transport.step(actions, now=1.01)
    assert second.goal.primitive == "set_tilt"
    assert second.goal.module_ids == ("physical_v6",)
    stopped = transport.step(SnakeActions(), now=1.02)
    assert set(stopped.cancel_goal_ids) == {
        first.goal.goal_id, second.goal.goal_id}


def test_snake_posture_transport_rejects_pan_targets():
    transport = SnakePostureTransport()
    with pytest.raises(ValueError, match="TILT"):
        transport.step(SnakeActions(
            joint_targets=(("physical_v5", "pan", 0.4, 0.1),),
            allow_joint_updates=True,
        ), now=1.0)
    allowed = transport.step(SnakeActions(
        joint_targets=(("physical_v5", "pan", 0.4, 0.1),),
        allow_joint_updates=True, manual_override=True,
    ), now=1.1)
    assert allowed.goal.primitive == "rotate_pan_by"
    assert allowed.goal.parameters["delta_rad"] == pytest.approx(0.3)


def test_primitive_target_is_recorded_as_effective_action_only_when_sent():
    snake = runtime()
    snake.observe_graph(graph().to_dict(), now=1.0)
    snake.step(sample(), safety=ENABLED, now=1.0)
    out = snake.step(sample(right_y=1.0, r2=1.0), safety=ENABLED, now=1.1)
    assert out.posture.goal is not None
    module = out.posture.goal.module_ids[0]
    assert "tilt_target_rad" in out.actions.module_actions[module]
    assert "tilt_target_rad" not in json.loads(out.envelope)["locomotion"][module]
    again = snake.step(sample(right_y=1.0, r2=1.0), safety=ENABLED, now=1.12)
    assert "tilt_target_rad" in again.actions.module_actions[module]


def test_body_relative_head_target_tracks_live_root_translation():
    snake = runtime()
    snake.observe_graph(graph().to_dict(), now=1.0)
    snake.step(sample(), safety=ENABLED, now=1.0)
    before = snake.step(sample(right_y=1.0), safety=ENABLED, now=1.1)
    payload = graph(stamp=2.0).to_dict()
    for node in payload["nodes"]:
        node["attributes"]["pose"]["position"][0] += 0.1
    assert snake.observe_graph(payload, now=1.2)
    after = snake.step(sample(), safety=ENABLED, now=1.2)
    assert after.actions.intent["head_target_m"][0] == pytest.approx(
        before.actions.intent["head_target_m"][0] + 0.1)


def test_forward_wheels_follow_head_even_when_tail_local_axis_is_reversed():
    snake = runtime()
    payload = graph().to_dict()
    payload["nodes"][0]["attributes"]["pose"]["orientation_xyzw"] = [0, 0, 1, 0]
    assert snake.observe_graph(payload, now=1.0)
    moving = snake.step(sample(r2=0.5), safety=ENABLED, now=1.0)
    assert moving.actions.module_actions["physical_v0"]["vx"] < 0
    assert moving.actions.module_actions["physical_v1"]["vx"] > 0


def test_invalid_pose_fails_closed_before_snake_claims_authority():
    snake = runtime()
    payload = graph().to_dict()
    payload["nodes"][7]["attributes"]["pose"]["orientation_xyzw"] = [0, 0, 0, 0]
    assert not snake.observe_graph(payload, now=1.0)
    assert snake.topology(1.0) is None
    assert snake.step(sample(r2=1.0), safety=ENABLED, now=1.0).envelope is None


def test_joint_target_slews_and_head_height_is_bounded():
    snake = runtime()
    snake.observe_graph(graph().to_dict(), now=1.0)
    snake.step(sample(), safety=ENABLED, now=1.0)
    first = snake.step(sample(right_y=1.0, r2=1.0), safety=ENABLED, now=1.1)
    assert all(abs(target) <= 0.05 + 1e-9
               for _, _, target, _ in first.actions.joint_targets)
    for i in range(2, 30):
        snake.observe_graph(graph(stamp=i + 1.0).to_dict(), now=1.0 + i * 0.1)
        last = snake.step(sample(right_y=-1.0), safety=ENABLED, now=1.0 + i * 0.1)
    assert last.actions.intent["head_target_m"][2] >= 0.02
