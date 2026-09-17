"""Timeline requests and camera channel, using a controlled timeline fixture."""
import importlib
import importlib.util
import json

import pytest


def runtime(tmp_path, playing=True):
    module = "smores_ep.isaac.teleop_runtime"
    assert importlib.util.find_spec(module) is not None, "missing T2 Isaac runtime adapter"
    timeline = Timeline(playing)
    cameras = []
    bridge = importlib.import_module(module).TeleopRuntimeBridge(
        timeline, tmp_path / "request.json", tmp_path / "status.json",
        camera_callback=lambda eye, target: cameras.append((eye, target)),
        center_callback=lambda: (3.0, 4.0, 0.2))
    return bridge, timeline, cameras


class Timeline:
    def __init__(self, playing):
        self.playing = playing
        self.calls = []

    def is_playing(self):
        return self.playing

    def pause(self):
        self.calls.append("pause")
        self.playing = False

    def play(self):
        self.calls.append("resume")
        self.playing = True


def write(bridge, operation=None, request_id="r1", camera=None):
    payload = {"schema_version": "mssr.teleop_runtime.v1", "camera": camera}
    if operation is not None:
        payload["timeline_request"] = {"id": request_id, "operation": operation}
    bridge.request_file.write_text(json.dumps(payload))


def test_pause_is_acknowledged_and_duplicate_request_is_idempotent(tmp_path):
    bridge, timeline, _ = runtime(tmp_path)
    write(bridge, "pause")
    status = bridge.poll()
    assert not status["timeline_playing"]
    assert status["timeline_ack"] == {"id": "r1", "operation": "pause", "applied": True}
    bridge.poll()
    assert timeline.calls == ["pause"]
    assert json.loads(bridge.status_file.read_text())["timeline_ack"] == status["timeline_ack"]


def test_resume_can_be_processed_while_physics_is_paused(tmp_path):
    bridge, timeline, _ = runtime(tmp_path, playing=False)
    write(bridge, "resume")
    assert bridge.poll()["timeline_playing"]
    assert timeline.calls == ["resume"]


def test_old_pause_replay_cannot_pause_after_a_new_resume(tmp_path):
    bridge, timeline, _ = runtime(tmp_path)
    write(bridge, "pause", "r1")
    bridge.poll()
    write(bridge, "resume", "r2")
    bridge.poll()
    write(bridge, "pause", "r1")
    bridge.poll()
    assert timeline.playing and timeline.calls == ["pause", "resume"]


def test_camera_changes_follow_live_center_even_while_paused(tmp_path):
    bridge, timeline, cameras = runtime(tmp_path, playing=False)
    write(bridge, camera={"azimuth_rad": 0.0, "elevation_rad": 0.5, "radius_m": 2.0})
    assert bridge.poll()["camera_applied"]
    assert cameras[-1][1] == (3.0, 4.0, 0.2)
    assert not timeline.playing and not timeline.calls


def test_invalid_camera_does_not_prevent_estop(tmp_path):
    bridge, timeline, cameras = runtime(tmp_path)
    write(bridge, "pause", camera={"radius_m": -1})
    status = bridge.poll()
    assert not timeline.playing and not cameras
    assert status["timeline_ack"]["applied"] and status["error"]


@pytest.mark.parametrize("payload", ["{", "[]", "null", '{"schema_version":"wrong"}',
    '{"schema_version":"mssr.teleop_runtime.v1","timeline_request":{"id":"r1","operation":"stop"}}'])
def test_malformed_or_unsupported_request_cannot_control_timeline(tmp_path, payload):
    bridge, timeline, _ = runtime(tmp_path)
    bridge.request_file.write_text(payload)
    assert bridge.poll()["error"]
    assert timeline.playing and not timeline.calls


def test_no_request_still_reports_real_timeline_state(tmp_path):
    bridge, timeline, _ = runtime(tmp_path)
    assert bridge.poll()["timeline_playing"]
    timeline.playing = False
    assert not bridge.poll()["timeline_playing"]
