"""File bridge runtime routing with ROS methods replaced; no DDS initialization."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from rclpy.clock import ClockType


ROOT = Path(__file__).resolve().parents[4]


def bridge_module():
    spec = importlib.util.spec_from_file_location("teleop_file_bridge_test", ROOT / "ros2_bridge/mssr_file_bridge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bridge(tmp_path, monkeypatch):
    module = bridge_module()
    assert hasattr(module.MssrFileBridge, "_on_runtime_request"), "missing runtime channel in file bridge"
    publishers, subscriptions, timers = {}, {}, []
    monkeypatch.setattr(module.Node, "__init__", lambda self, name: None)
    monkeypatch.setattr(module.Node, "get_logger", lambda self: SimpleNamespace(info=lambda message: None))

    def publisher(self, message_type, topic, depth):
        values = []
        publishers[topic] = values
        return SimpleNamespace(publish=lambda message: values.append(message.data))

    def subscriber(self, message_type, topic, callback, depth):
        subscriptions[topic] = callback
        return object()

    def timer(self, period, callback, **kwargs):
        timers.append(kwargs)
        return object()

    monkeypatch.setattr(module.Node, "create_publisher", publisher)
    monkeypatch.setattr(module.Node, "create_subscription", subscriber)
    monkeypatch.setattr(module.Node, "create_timer", timer)
    node = module.MssrFileBridge(tmp_path / "state", tmp_path / "actions.json",
        tmp_path / "goal.json", tmp_path / "cancel.json", tmp_path / "primitive_status.json", 0.02,
        runtime_request_file=tmp_path / "runtime_request.json", runtime_status_file=tmp_path / "runtime_status.json")
    return node, publishers, subscriptions, timers


def test_runtime_request_never_writes_robot_or_primitive_files(tmp_path, monkeypatch):
    node, _, subscriptions, _ = bridge(tmp_path, monkeypatch)
    payload = '{"schema_version":"mssr.teleop_runtime.v1","timeline_request":{"id":"p1","operation":"pause"}}'
    subscriptions["/mssr/teleop/runtime_request"](SimpleNamespace(data=payload))
    assert (tmp_path / "runtime_request.json").read_text() == payload
    assert not any((tmp_path / name).exists() for name in ("actions.json", "goal.json", "cancel.json"))


def test_runtime_status_has_its_own_topic(tmp_path, monkeypatch):
    node, publishers, _, _ = bridge(tmp_path, monkeypatch)
    payload = '{"schema_version":"mssr.teleop_runtime_status.v1","timeline_playing":false}'
    (tmp_path / "runtime_status.json").write_text(payload)
    node._publish_files()
    assert publishers["/mssr/teleop/runtime_status"] == [payload]
    assert publishers["/mssr/primitives/status"] == []


def test_file_bridge_polling_is_independent_of_frozen_sim_time(tmp_path, monkeypatch):
    _, _, _, timers = bridge(tmp_path, monkeypatch)
    assert timers[0]["clock"].clock_type == ClockType.STEADY_TIME
