"""Structure-only E-STOP runtime protocol.

These tests deliberately contain no Isaac dependency. They define the
transport contract that replaces the old timeline pause/resume protocol.
"""
import json

from mssr_expert.teleop.runtime_channel import RuntimeChannel
from smores_ep.isaac.teleop_runtime import TeleopRuntimeBridge


def runtime_ack(request, *, stopped, stamp=10.0):
    return {
        "schema_version": "mssr.teleop_runtime_status.v1",
        "stamp_monotonic": stamp,
        "structure_stopped": stopped,
        "structure_stop_ack": {
            **request,
            "applied": True,
        },
        "camera_applied": False,
        "error": None,
    }


def test_runtime_channel_uses_structure_stop_not_timeline():
    channel = RuntimeChannel()

    channel.request_stop(True)
    payload = channel.payload(None)

    request = payload["structure_stop_request"]
    assert request["active"] is True

    assert "timeline_request" not in payload
    assert "timeline_playing" not in payload
    assert "timeline_ack" not in payload


def test_matching_stop_and_clear_acknowledgments_are_ordered():
    channel = RuntimeChannel()

    channel.request_stop(True)
    stop = channel.payload(None)["structure_stop_request"]

    assert channel.observe(
        runtime_ack(stop, stopped=True),
        10.0,
    ) == "stop"

    channel.request_stop(False)
    clear = channel.payload(None)["structure_stop_request"]

    assert clear["active"] is False
    assert clear["id"] != stop["id"]

    assert channel.observe(
        runtime_ack(clear, stopped=False),
        10.1,
    ) == "clear"

    assert channel.payload(None)["structure_stop_request"] is None


def test_new_emergency_stop_supersedes_pending_clear():
    channel = RuntimeChannel()

    channel.request_stop(False)
    pending_clear = channel.payload(None)["structure_stop_request"]
    assert pending_clear["active"] is False

    channel.request_stop(True)
    stop = channel.payload(None)["structure_stop_request"]

    assert stop["active"] is True
    assert stop["id"] != pending_clear["id"]

    assert channel.observe(
        runtime_ack(pending_clear, stopped=False),
        10.0,
    ) is None

    assert channel.payload(None)["structure_stop_request"] == stop


def test_runtime_bridge_applies_structure_stop_without_timeline(tmp_path):
    calls = []

    bridge = TeleopRuntimeBridge(
        tmp_path / "request.json",
        tmp_path / "status.json",
        structure_stop_callback=lambda active: calls.append(active),
    )

    request = {
        "id": "stop-1",
        "active": True,
    }
    bridge.request_file.write_text(
        json.dumps(
            {
                "schema_version": "mssr.teleop_runtime.v1",
                "structure_stop_request": request,
                "camera": None,
            }
        )
    )

    status = bridge.poll()

    assert calls == [True]
    assert status["structure_stopped"] is True
    assert status["structure_stop_ack"] == {
        "id": "stop-1",
        "active": True,
        "applied": True,
    }

    assert "timeline_playing" not in status
    assert "timeline_ack" not in status

    # Re-reading the same request must not apply the stop twice.
    bridge.poll()
    assert calls == [True]


def test_runtime_status_without_timeline_fields_can_be_ready():
    channel = RuntimeChannel()

    payload = {
        "schema_version": "mssr.teleop_runtime_status.v1",
        "stamp_monotonic": 10.0,
        "structure_stopped": False,
        "structure_stop_ack": None,
        "camera_applied": False,
        "error": None,
    }

    assert channel.observe(payload, 10.0) is None
    assert channel.ready(10.1)
    assert channel.structure_stopped is False
