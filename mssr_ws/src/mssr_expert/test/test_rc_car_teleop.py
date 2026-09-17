"""RC-Car8 behavior against real role matching, DOFs and backend parsers."""
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from mssr_expert.behaviors.morphology_library import MorphologyLibrary
from mssr_expert.graph.serialization import attributed_graph_from_dict, load_attributed_graph
from mssr_expert.teleop.safety import SafetyDecision, SafetyGate
from smores_ep.config.geometry import SmoresGeometry
from smores_ep.primitives.file_channel import ActionFileChannel
from smores_ep.primitives.model import PrimitiveGoal


CONFIG = Path(__file__).parents[1] / "config"
ENABLED = SafetyDecision("TELEOP", True, False, True)
GEOMETRY = SmoresGeometry()


def component(name):
    path = "mssr_expert.teleop." + name
    assert importlib.util.find_spec(path) is not None, f"missing T3 {name}"
    return importlib.import_module(path)


def graph(tilt=-0.785398, lower=-math.pi / 2, upper=math.pi / 2):
    payload = json.loads((CONFIG / "smores_rc_car8.json").read_text())
    ids = {item["node_id"]: "physical_" + item["node_id"] for item in payload["nodes"]}
    for item in payload["nodes"]:
        item["node_id"] = ids[item["node_id"]]
        item["attributes"].update(node_type="physical_module", is_target_node=False,
                                  actuators={"tilt": {"position_rad": tilt,
                                                      "lower_limit_rad": lower,
                                                      "upper_limit_rad": upper}})
    for edge in payload["edges"]:
        edge["module_a_id"] = ids[edge["module_a_id"]]
        edge["module_b_id"] = ids[edge["module_b_id"]]
        edge["attributes"].update(is_target_edge=False, is_attached=True,
                                  relation_type="current_connection")
    return attributed_graph_from_dict(payload)


def setup_controller(**graph_kwargs):
    rc = component("rc_car")
    target = load_attributed_graph(CONFIG / "smores_rc_car8.json")
    observation = rc.RcCarObservation.from_graph(graph(**graph_kwargs), target)
    assert observation is not None
    controller = rc.RcCarTeleopController(
        MorphologyLibrary.load(CONFIG / "smores_morphology_behaviors.json"),
        geometry=GEOMETRY, height_rate_m_s=0.005, max_dt_s=0.1)
    return controller, observation


def sample(**values):
    return SimpleNamespace(
        **{**dict(left_x=0.0, left_y=0.0, right_x=0.0, right_y=0.0, l2=0.0, r2=0.0), **values})


def step(controller, observation, dt=0.1, safety=ENABLED, **values):
    return controller.step(sample(**values), observation, dt, safety=safety)


def rates(result):
    return {module: command["pan_rate_rad_s"] for module, command in result.module_actions.items()}


@pytest.mark.parametrize("trigger,value,front,rear", [
    ("r2", 0.1, 0.3184713376, -0.3184713376),
    ("r2", 0.2, 0.6369426752, -0.6369426752),
    ("l2", 0.1, -0.3184713376, 0.3184713376),
    ("l2", 0.2, -0.6369426752, 0.6369426752),
])
def test_analog_triggers_use_existing_pan_traction_signs(trigger, value, front, rear):
    controller, observation = setup_controller()
    result = step(controller, observation, **{trigger: value})
    assert rates(result) == pytest.approx({"physical_v3": front, "physical_v4": rear,
                                          "physical_v5": front, "physical_v6": rear})


def test_trigger_release_zeroes_traction_in_same_tick():
    controller, observation = setup_controller()
    step(controller, observation, r2=1.0)
    assert all(rate == 0 for rate in rates(step(controller, observation, dt=0)).values())


@pytest.mark.parametrize("x,front,rear", [(0.1, 0.6242038217, -0.0127388535),
                                       (-0.1, 0.0127388535, -0.6242038217)])
def test_right_x_turns_right_with_existing_role_and_yaw_conventions(x, front, rear):
    controller, observation = setup_controller()
    result = step(controller, observation, right_x=x, r2=0.1)
    assert rates(result) == pytest.approx({"physical_v3": front, "physical_v4": rear,
                                          "physical_v5": front, "physical_v6": rear})


def test_releasing_triggers_zeroes_traction_even_with_steering_stick_held():
    controller, observation = setup_controller()
    step(controller, observation, r2=0.1, right_x=0.5)
    released = step(controller, observation, right_x=0.5)
    assert all(rate == 0 for rate in rates(released).values())


def test_height_input_is_integrated_in_meters_with_dt_and_held_on_release():
    controller, observation = setup_controller()
    initial = step(controller, observation).intent["chassis_height_m"]
    changed = step(controller, observation, dt=0.04, right_y=0.5)
    assert changed.intent["chassis_height_m"] == pytest.approx(initial + 0.0001)
    assert step(controller, observation).intent["chassis_height_m"] == changed.intent["chassis_height_m"]


def test_large_dt_does_not_jump_after_scheduler_stall():
    controller, observation = setup_controller()
    initial = step(controller, observation).intent["chassis_height_m"]
    result = step(controller, observation, dt=100, right_y=-1)
    assert result.intent["chassis_height_m"] == pytest.approx(initial - 0.0005)


@pytest.mark.parametrize("direction,bound", [(1, -0.85), (-1, -0.70)])
def test_common_height_target_respects_live_tilt_bounds(direction, bound):
    controller, observation = setup_controller(lower=-0.85, upper=-0.70)
    for _ in range(100):
        result = step(controller, observation, right_y=direction)
    assert result.intent["chassis_height_m"] == pytest.approx(GEOMETRY.ground_contact_height_m(bound))
    assert all(-0.85 <= item.angle_rad <= -0.70 for item in result.joint_targets)


def test_height_never_crosses_the_nonmonotonic_contact_geometry_branch():
    controller, observation = setup_controller()
    for _ in range(100):
        result = step(controller, observation, right_y=1)
    assert all(-1.05 < item.angle_rad <= 0 for item in result.joint_targets)
    assert result.intent["chassis_height_m"] < 0.052


def test_home_slews_to_library_nominal_without_teleport():
    controller, observation = setup_controller(tilt=-0.4)
    before = step(controller, observation).intent["chassis_height_m"]
    controller.home()
    first = step(controller, observation)
    nominal = GEOMETRY.ground_contact_height_m(-0.785398)
    assert before < first.intent["chassis_height_m"] < nominal
    assert first.intent["chassis_height_m"] - before <= 0.000500001
    for _ in range(100):
        result = step(controller, observation)
    assert result.intent["chassis_height_m"] == pytest.approx(nominal)


def test_pan_defers_tilt_per_module_and_release_resumes_held_height():
    controller, observation = setup_controller()
    # vx=.0096 and yaw=-.08 cancel PAN only for rear support modules.
    busy = step(controller, observation, r2=0.096, right_x=0.1, right_y=1)
    assert {item.module_id for item in busy.joint_targets} == {"physical_v4", "physical_v6"}
    desired = busy.intent["chassis_height_m"]
    released = step(controller, observation)
    assert released.intent["chassis_height_m"] == desired
    assert {item.module_id for item in released.joint_targets} == {
        "physical_v3", "physical_v4", "physical_v5", "physical_v6"}


def test_left_stick_never_changes_any_robot_action_or_intent():
    first, observation = setup_controller()
    second, _ = setup_controller()
    assert step(first, observation) == step(second, observation, left_x=1, left_y=-1)


def test_initial_neutral_captures_reached_tilt_without_posture_jump():
    controller, observation = setup_controller(tilt=-0.4)
    result = step(controller, observation)
    assert not result.joint_targets
    assert result.intent["chassis_height_m"] == pytest.approx(GEOMETRY.ground_contact_height_m(-0.4))


def test_missing_or_wrong_topology_cannot_produce_traction_or_joint_targets():
    controller, observation = setup_controller()
    step(controller, observation, r2=1)
    result = step(controller, None, r2=1, right_y=1)
    assert all(rate == 0 for rate in rates(result).values())
    assert not result.joint_targets
    rc = component("rc_car")
    target = load_attributed_graph(CONFIG / "smores_rc_car8.json")
    assert rc.RcCarObservation.from_graph(load_attributed_graph(CONFIG / "smores_snake8.json"), target) is None
    assert rc.RcCarObservation.from_graph(target, target) is None


@pytest.mark.parametrize("safety", [SafetyDecision("ESTOP", False, True, False),
                                    SafetyDecision("NONE", False, True, False)])
def test_safety_denial_zeroes_traction_and_emits_no_new_pose_goal(safety):
    controller, observation = setup_controller()
    result = step(controller, observation, safety=safety, r2=1, right_y=1)
    assert all(rate == 0 for rate in rates(result).values())
    assert not result.joint_targets


def test_macro_authority_never_receives_teleop_safe_hold():
    controller, observation = setup_controller()
    result = step(controller, observation, safety=SafetyDecision("STRUCTURAL_MACRO", False, False, False))
    assert result.module_actions == {} and not result.joint_targets


def test_estop_captures_reached_posture_and_cancels_home_intent():
    controller, observation = setup_controller(tilt=-0.4)
    step(controller, observation, right_y=1)
    controller.home()
    step(controller, observation, safety=SafetyDecision("ESTOP", False, True, False))
    result = step(controller, observation)
    assert not result.joint_targets
    assert result.intent["chassis_height_m"] == pytest.approx(GEOMETRY.ground_contact_height_m(-0.4))


def test_resume_gate_needs_new_neutral_then_allows_new_rc_input():
    controller, observation = setup_controller()
    gate = SafetyGate()
    gate.update(connected=True, l2=0, r2=0, received_at=10, topology_supported=True)
    gate.pause()
    step(controller, observation, safety=gate.update(connected=True, l2=0, r2=1, received_at=10.1))
    gate.resume(resumed_at=11)
    denied = gate.update(connected=True, l2=0, r2=0, received_at=10.5, topology_supported=True)
    assert all(rate == 0 for rate in rates(step(controller, observation, safety=denied, r2=1)).values())
    armed = gate.update(connected=True, l2=0, r2=0, received_at=11.1, topology_supported=True)
    assert any(rates(step(controller, observation, safety=armed, r2=1)).values())


@pytest.mark.parametrize("field,value", [("r2", float("nan")), ("l2", float("inf")),
                                        ("right_x", 2), ("right_y", float("nan"))])
def test_invalid_controller_input_fails_closed(field, value):
    controller, observation = setup_controller()
    result = step(controller, observation, **{field: value})
    assert all(rate == 0 for rate in rates(result).values()) and not result.joint_targets


@pytest.mark.parametrize("dt", [float("nan"), float("inf"), -1])
def test_invalid_dt_fails_closed(dt):
    controller, observation = setup_controller()
    result = step(controller, observation, dt=dt, r2=1)
    assert all(rate == 0 for rate in rates(result).values())


def test_exact_action_envelope_round_trips_effective_commands_to_native_parser():
    controller, observation = setup_controller()
    result = step(controller, observation, r2=0.1)
    transport = component("action_transport").ActionTransport()
    payload = transport.serialize(result, stamp=123.0, command_id="rc-test")
    decoded = json.loads(payload)
    assert transport.topic == "/mssr/actions"
    assert decoded["schema_version"] == "mssr.actions.v2"
    assert set(decoded) == {"schema_version", "stamp", "stage_id", "task_type", "reset",
                            "locomotion", "magnetic", "expert"}
    assert decoded["locomotion"] == result.module_actions
    assert decoded["expert"]["module_roles"]["physical_v3"] == "wheel_left_front"
    assert decoded["expert"]["pan_traction_module_ids"] == ["physical_v3", "physical_v4", "physical_v5", "physical_v6"]
    assert "cmd_vel" not in payload
    native, diagnostics = ActionFileChannel._parse(payload)
    assert native["physical_v3"].pan_velocity_rad_s == pytest.approx(0.3184713376)
    assert diagnostics.command_id == "rc-test"


@pytest.mark.parametrize("where", ["stamp", "actions", "intent"])
def test_transport_rejects_all_nonfinite_values(where):
    controller, observation = setup_controller()
    result = step(controller, observation)
    stamp = float("nan") if where == "stamp" else 1.0
    if where == "actions":
        result.module_actions["physical_v3"]["pan_rate_rad_s"] = float("inf")
    if where == "intent":
        result.intent["chassis_height_m"] = float("nan")
    with pytest.raises(ValueError):
        component("action_transport").ActionTransport().serialize(result, stamp=stamp, command_id="rc-test")


def test_height_primitive_delivery_waits_for_admission_and_cancels_before_pan():
    controller, observation = setup_controller()
    height = step(controller, observation, right_y=1)
    delivery = component("action_transport").RcCarPostureTransport()
    first = delivery.step(height, now=10)
    goal = first.goal
    assert goal is not None
    assert PrimitiveGoal.from_json(json.dumps(goal.to_dict())).primitive.value == "set_tilt"
    assert delivery.step(height, now=10.01).goal is None
    delivery.observe({"schema_version": "mssr.primitive_status.v1", "goal_id": goal.goal_id,
                      "primitive": "set_tilt", "module_ids": list(goal.module_ids),
                      "state": "running"})
    busy = step(controller, observation, r2=0.1)
    canceled = delivery.step(busy, now=10.02)
    assert canceled.cancel_goal_id == goal.goal_id
    assert goal.module_ids[0] in canceled.blocked_module_ids
    delivery.observe({"schema_version": "mssr.primitive_status.v1", "goal_id": goal.goal_id,
                      "primitive": "set_tilt", "module_ids": list(goal.module_ids),
                      "state": "canceled"})
    assert goal.module_ids[0] not in delivery.step(busy, now=10.03).blocked_module_ids


def test_posture_transport_does_not_send_goals_after_safety_loss():
    controller, observation = setup_controller()
    height = step(controller, observation, right_y=1)
    delivery = component("action_transport").RcCarPostureTransport()
    goal = delivery.step(height, now=10).goal
    denied = step(controller, observation, safety=SafetyDecision("NONE", False, True, False))
    result = delivery.step(denied, now=10.1)
    assert result.goal is None and result.cancel_goal_id == goal.goal_id


def test_unacknowledged_pose_goal_is_retried_with_same_id_not_overwritten():
    controller, observation = setup_controller()
    height = step(controller, observation, right_y=1)
    delivery = component("action_transport").RcCarPostureTransport()
    first = delivery.step(height, now=10).goal
    assert delivery.step(height, now=11).goal == first


def test_queued_unadmitted_tilt_cannot_take_a_module_requested_by_pan():
    from dataclasses import replace
    controller, observation = setup_controller()
    height = step(controller, observation, right_y=1)
    delivery = component("action_transport").RcCarPostureTransport()
    goal = delivery.step(height, now=10).goal
    delivery.observe({"schema_version": "mssr.primitive_status.v1", "goal_id": goal.goal_id,
                      "primitive": "set_tilt", "module_ids": list(goal.module_ids), "state": "running"})
    commands = {module: dict(command) for module, command in height.module_actions.items()}
    other = next(module for module in commands if module != goal.module_ids[0])
    commands[other]["pan_rate_rad_s"] = 1
    result = delivery.step(replace(height, module_actions=commands), now=10.1)
    assert result.goal is None
    assert result.cancel_goal_id == goal.goal_id


def runtime():
    return component("action_transport").RcCarRuntime(
        MorphologyLibrary.load(CONFIG / "smores_morphology_behaviors.json"),
        load_attributed_graph(CONFIG / "smores_rc_car8.json"), geometry=GEOMETRY,
        observation_timeout_s=0.5)


def test_runtime_rejects_stale_and_invalid_graphs_and_holds_last_supports():
    core = runtime()
    assert core.observe_graph(graph().to_dict(), now=10)
    assert core.topology(10.1) == "rc_car8"
    assert any(rates(core.step(sample(r2=1), safety=ENABLED, now=10.1).actions).values())
    assert core.topology(10.6) is None
    assert all(rate == 0 for rate in rates(core.step(sample(r2=1), safety=ENABLED, now=10.6).actions).values())
    assert not core.observe_graph({"nodes": "broken"}, now=10.7)
    assert core.topology(10.7) is None


def test_runtime_cannot_publish_pan_until_active_pose_is_retired():
    core = runtime()
    core.observe_graph(graph().to_dict(), now=10)
    core.step(sample(), safety=ENABLED, now=10)
    height = core.step(sample(right_y=1), safety=ENABLED, now=10.1)
    goal = height.posture.goal
    driving = core.step(sample(r2=0.1), safety=ENABLED, now=10.2)
    assert driving.posture.cancel_goal_id == goal.goal_id
    assert driving.actions.module_actions[goal.module_ids[0]]["pan_rate_rad_s"] == 0
    core.observe_status({"schema_version": "mssr.primitive_status.v1", "goal_id": goal.goal_id,
                         "primitive": "set_tilt", "module_ids": list(goal.module_ids), "state": "canceled"})
    assert core.step(sample(r2=0.1), safety=ENABLED, now=10.3).actions.module_actions[goal.module_ids[0]]["pan_rate_rad_s"] != 0


def test_runtime_macro_does_not_publish_any_teleop_action_envelope():
    core = runtime()
    core.observe_graph(graph().to_dict(), now=10)
    output = core.step(sample(r2=1), safety=SafetyDecision("STRUCTURAL_MACRO", False, False, False), now=10)
    assert output.envelope is None


def test_runtime_rotates_action_identity_after_stop_to_escape_native_quarantine():
    core = runtime()
    core.observe_graph(graph().to_dict(), now=10)
    before = json.loads(core.step(sample(), safety=ENABLED, now=10).envelope)
    core.step(sample(), safety=SafetyDecision("ESTOP", False, True, False), now=10.1)
    after = json.loads(core.step(sample(), safety=ENABLED, now=10.2).envelope)
    assert before["expert"]["debug"]["command_id"] != after["expert"]["debug"]["command_id"]


def test_hardware_smoke_refuses_deferred_bindings_before_any_runtime_launch():
    root = Path(__file__).resolve().parents[4]
    result = subprocess.run([sys.executable, str(root / "scripts/teleop/check_rc_car.py"),
                             "--input-config", str(CONFIG / "smores_dualsense.yaml")],
                            cwd=root, capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert "deferred" in result.stderr and "home" in result.stderr
    assert "estop" in result.stderr and "resume" in result.stderr


def test_duplicate_goal_rejection_does_not_retire_actual_tilt_ownership():
    core = runtime()
    core.observe_graph(graph().to_dict(), now=10)
    core.step(sample(), safety=ENABLED, now=10)
    goal = core.step(sample(right_y=1), safety=ENABLED, now=10.1).posture.goal
    core.observe_status({"schema_version": "mssr.primitive_status.v1", "goal_id": goal.goal_id,
                         "primitive": "set_tilt", "module_ids": list(goal.module_ids),
                         "state": "rejected", "code": "DUPLICATE_GOAL_ID"})
    driving = core.step(sample(r2=0.1), safety=ENABLED, now=10.2)
    assert driving.posture.cancel_goal_id == goal.goal_id
    assert driving.actions.module_actions[goal.module_ids[0]]["pan_rate_rad_s"] == 0


def test_macro_authority_stays_exclusive_even_when_live_tilt_is_outside_rc_branch():
    controller, observation = setup_controller()
    step(controller, observation)
    _, changed = setup_controller(tilt=-1.2)
    result = step(controller, changed, safety=SafetyDecision("STRUCTURAL_MACRO", False, False, False))
    assert result.module_actions == {} and not result.joint_targets


def smoke():
    path = Path(__file__).resolve().parents[4] / "scripts/teleop/check_rc_car.py"
    spec = importlib.util.spec_from_file_location("mssr_rc_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_topics_are_exclusive_and_driver_is_the_verified_selected_device(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[4] / "scripts/teleop"))
    commands = smoke().runtime_commands(tmp_path, CONFIG / "smores_dualsense.yaml", "abc123", 2)
    assert commands["assembly"][3] == "mssr_smores_self_assembly_node"
    assert "joy:=/mssr/teleop_probe/run_abc123/joy" in commands["joy"]
    assert "device_id:=2" in commands["joy"]
    assert "start_joy:=false" in commands["teleop"]
    assert "joy_topic:=/mssr/teleop_probe/run_abc123/joy" in commands["teleop"]
    assert "status_topic:=/mssr/teleop_probe/run_abc123/status" in commands["teleop"]


def test_smoke_held_height_captures_after_actual_stick_release():
    probe = smoke().HeldHeightProbe()
    status = {"controller_input": {"right_y": -1, "l2": 0, "r2": 0},
              "rc_car_intent": {"chassis_height_m": 0.04}}
    assert not probe.observe(status, now=10)
    status["rc_car_intent"]["chassis_height_m"] = 0.039
    status["controller_input"]["right_y"] = 0
    assert not probe.observe(status, now=10.5)
    assert probe.observe(status, now=11.6)


def test_pan_release_slews_from_reached_posture_instead_of_preemption_target():
    controller, observation = setup_controller()
    step(controller, observation, right_y=-1)
    _, reached = setup_controller(tilt=-0.4)
    busy = step(controller, reached, r2=0.2)
    desired = busy.intent["chassis_height_m"]
    released = step(controller, reached)
    assert released.intent["chassis_height_m"] == desired
    current = GEOMETRY.ground_contact_height_m(-0.4)
    assert released.joint_targets
    assert all(abs(GEOMETRY.ground_contact_height_m(target.angle_rad) - current) <= 0.000500001
               for target in released.joint_targets)


def test_bridge_republishing_cached_graph_cannot_keep_topology_fresh():
    core = runtime()
    payload = graph().to_dict()
    assert core.observe_graph(payload, now=10)
    assert core.observe_graph(payload, now=10.4)
    assert core.topology(10.6) is None
    payload["stamp"] = 1
    assert core.observe_graph(payload, now=10.7)
    assert core.topology(10.8) == "rc_car8"


def test_ros_shell_can_reuse_cad_geometry_without_isaac_package_on_pythonpath(tmp_path):
    environment = {**os.environ, "PYTHONPATH": str(CONFIG.parent)}
    result = subprocess.run([sys.executable, "-c",
                             "from mssr_expert.teleop.rc_car import load_geometry; "
                             "print(load_geometry().ground_contact_height_m(-0.785398))"],
                            cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert float(result.stdout) == pytest.approx(0.05032, abs=0.00001)


def test_home_requested_on_first_observation_is_not_lost_at_initial_capture():
    controller, observation = setup_controller(tilt=-0.4)
    controller.home()
    result = step(controller, observation)
    assert result.intent["chassis_height_m"] == pytest.approx(GEOMETRY.ground_contact_height_m(-0.4) + 0.0005)


def test_runtime_uses_existing_unix_wall_timestamp_in_action_envelope():
    core = runtime()
    core.observe_graph(graph().to_dict(), now=10)
    before = time.time()
    envelope = json.loads(core.step(sample(), safety=ENABLED, now=10).envelope)
    assert before <= envelope["stamp"] <= time.time()
