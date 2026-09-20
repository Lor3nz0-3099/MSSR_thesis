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


def test_manual_pan_targets_only_selected_module_while_all_wheels_drive():
    snake = runtime()
    snake.observe_graph(graph().to_dict(), now=1.0)
    snake.step(sample(), safety=ENABLED, now=1.0)
    out = snake.step(sample(right_x=1.0, r2=0.5), safety=ENABLED, now=1.1)
    assert out.posture.goal is not None
    assert out.posture.goal.primitive == "rotate_pan_by"
    assert out.posture.goal.module_ids == ("physical_v7",)
    assert all(command["vx"] > 0 for command in out.actions.module_actions.values())
    assert "pan_target_rad" in out.actions.module_actions["physical_v7"]
    assert sum("pan_target_rad" in command for command in out.actions.module_actions.values()) == 1
    assert "pan_target_rad" not in json.loads(out.envelope)["locomotion"]["physical_v7"]


def test_manual_module_switch_retires_old_pan_before_tilt_on_next_module():
    snake = runtime()
    snake.observe_graph(graph().to_dict(), now=1.0)
    snake.step(sample(), safety=ENABLED, now=1.0)
    pan = snake.step(sample(right_x=1.0, r2=0.5), safety=ENABLED, now=1.1)
    goal = pan.posture.goal
    assert goal is not None
    switched = snake.step(sample(command_events=("next_module",), l2=0.5),
                          safety=ENABLED, now=1.12)
    assert switched.actions.intent["selected_module_id"] == "physical_v6"
    assert goal.goal_id in switched.posture.cancel_goal_ids
    assert all(command["vx"] < 0 for command in switched.actions.module_actions.values())
    assert snake.step(sample(right_y=1.0, l2=0.5), safety=ENABLED, now=1.14).posture.goal is None
    snake.observe_status({"goal_id": goal.goal_id, "primitive": "rotate_pan_by",
                          "state": "canceled", "module_ids": ["physical_v7"]})
    snake.step(sample(right_y=1.0, l2=0.5), safety=ENABLED, now=1.16)
    tilt = snake.step(sample(right_y=1.0, l2=0.5), safety=ENABLED, now=1.26)
    assert tilt.posture.goal is not None
    assert tilt.posture.goal.primitive == "set_tilt"
    assert tilt.posture.goal.module_ids == ("physical_v6",)
    assert all(command["vx"] < 0 for command in tilt.actions.module_actions.values())


def test_same_module_wheels_are_paired_and_release_stops_them_immediately():
    snake = runtime()
    snake.observe_graph(graph().to_dict(), now=1.0)
    moving = snake.step(sample(r2=0.7), safety=ENABLED, now=1.1)
    assert all(command["vx"] > 0 and command["yaw_rate"] == 0
               for command in moving.actions.module_actions.values())
    stopped = snake.step(sample(), safety=ENABLED, now=1.2)
    assert all(command["vx"] == 0 for command in stopped.actions.module_actions.values())


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



def test_manual_posture_transport_accepts_pan_target():
    transport = SnakePostureTransport()
    delivery = transport.step(
        SnakeActions(
            joint_targets=(("physical_v7", "pan", 0.4, 0.1),),
            allow_joint_updates=True,
            manual_override=True,
        ),
        now=1.0,
    )

    assert delivery.goal is not None
    assert delivery.goal.primitive == "rotate_pan_by"
    assert delivery.goal.module_ids == ("physical_v7",)
    assert delivery.goal.parameters["delta_rad"] == pytest.approx(0.3)


def test_l2_drives_all_modules_in_reverse_without_automatic_shape_target():
    snake = runtime()
    snake.observe_graph(graph().to_dict(), now=1.0)
    snake.step(sample(), safety=ENABLED, now=1.0)

    reverse = snake.step(
        sample(l2=0.5),
        safety=ENABLED,
        now=1.1,
    )

    assert len(reverse.actions.module_actions) == 8
    assert all(
        command["vx"] < 0
        for command in reverse.actions.module_actions.values()
    )
    assert reverse.actions.joint_targets == ()
