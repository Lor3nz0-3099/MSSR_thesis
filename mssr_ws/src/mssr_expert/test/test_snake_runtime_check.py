"""Acceptance checker must use physical Snake data and fail before launch."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from mssr_expert.graph.serialization import load_attributed_graph
from test_snake_teleop import graph


ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "mssr_ws/src/mssr_expert/config"


def checker(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts/teleop"))
    spec = importlib.util.spec_from_file_location(
        "mssr_snake_runtime_check", ROOT / "scripts/teleop/check_snake.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_uses_physical_assignment_and_rejects_broken_chain(monkeypatch):
    smoke = checker(monkeypatch)
    target = load_attributed_graph(CONFIG / "smores_snake8.json")
    metrics = smoke.physical_metrics(graph().to_dict(), target)
    assert metrics["roles"]["snake_head"] == "physical_v7"
    assert metrics["forward"] == pytest.approx((1.0, 0.0))
    with pytest.raises(ValueError, match="valid Snake8"):
        smoke.physical_metrics(graph(missing=True).to_dict(), target)


def test_wheel_invariant_and_recording_require_actual_snake_actions(tmp_path, monkeypatch):
    smoke = checker(monkeypatch)
    assert smoke.wheel_invariant({"physical_v7": {"vx": 0.03, "yaw_rate": 0.0}})
    assert not smoke.wheel_invariant({"physical_v7": {"vx": 0.03, "yaw_rate": 0.1}})
    human = tmp_path / "human_behavior.jsonl"
    human.write_text("\n".join(json.dumps(row) for row in (
        {"task_type": "rc_car8_teleop", "expert_action": {"locomotion": {"a": {"vx": 1}}}},
        {"task_type": "snake8_teleop", "observation": {"intent": {
            "control_mode": "single_module", "selected_module_id": "physical_v6"}},
         "expert_action": {"locomotion": {
            "physical_v6": {"vx": 0.0, "pan_target_rad": 0.1}}}},
        {"task_type": "snake8_teleop", "expert_action": {"locomotion": {
            "physical_v6": {"vx": 0.03, "tilt_target_rad": 0.1}}}},
    )) + "\n")
    assert smoke.recorded_snake_actions(human) == {"rows": 2, "wheel_rows": 1,
                                                    "shape_rows": 1, "manual_pan_rows": 1}


def test_reverse_gate_rejects_alternating_wheel_direction(monkeypatch):
    smoke = checker(monkeypatch)
    forward = {"a": {"vx": 0.03}, "b": {"vx": -0.03}}
    reverse = {"a": {"vx": -0.02}, "b": {"vx": 0.02}}
    alternating = {"a": {"vx": 0.02}, "b": {"vx": -0.02}}
    window = smoke.OppositeWheelWindow(forward, hold_s=0.3)
    assert not window.observe(1.0, reverse)
    assert not window.observe(1.1, alternating)
    assert not window.observe(1.2, reverse)
    assert not window.observe(1.4, reverse)
    assert window.observe(1.5, reverse)
    assert not window.observe(1.6, alternating)


def test_runtime_commands_assemble_snake_directly(tmp_path, monkeypatch):
    smoke = checker(monkeypatch)
    commands = smoke.snake_runtime_commands(
        tmp_path, CONFIG / "smores_dualsense.yaml", "snake-probe", 0,
        runtime_dir=tmp_path / "runtime",
    )
    assert f"target_graph_path:={CONFIG / 'smores_snake8.json'}" in commands["assembly"]
    assert "execution_id:=t6a-snake-probe" in commands["assembly"]
    assert not any("smores_rc_car8.json" in arg for arg in commands["assembly"])
    assert any("mssr_snake_teleop_snake-probe" in arg for arg in commands["teleop"])


def test_direct_snake_preflight_does_not_require_reconfiguration_button(tmp_path):
    mapping = yaml.safe_load((CONFIG / "smores_dualsense.yaml").read_text())
    mapping["commands"]["select_snake"] = None
    path = tmp_path / "deferred.yaml"
    path.write_text(yaml.safe_dump(mapping))
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/teleop/check_snake.py"),
         "--input-config", str(path), "--device-id", "-1"], cwd=ROOT,
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 2
    assert "device ID" in result.stderr


def test_missing_recording_binding_fails_preflight_before_runtime(tmp_path):
    mapping = yaml.safe_load((CONFIG / "smores_dualsense.yaml").read_text())
    mapping["commands"]["record_toggle"] = None
    path = tmp_path / "deferred.yaml"
    path.write_text(yaml.safe_dump(mapping))
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/teleop/check_snake.py"),
         "--input-config", str(path)], cwd=ROOT, capture_output=True,
        text=True, timeout=5)
    assert result.returncode == 2
    assert "record_toggle must use the approved start button" in result.stderr
