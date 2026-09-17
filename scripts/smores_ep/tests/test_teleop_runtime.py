"""Structure-stop and camera runtime channel without Isaac timeline control."""
import importlib
import importlib.util
import json

import pytest


def runtime(tmp_path):
    module = "smores_ep.isaac.teleop_runtime"
    assert importlib.util.find_spec(module) is not None

    stop_calls = []
    cameras = []

    request_file = tmp_path / "request.json"
    status_file = tmp_path / "status.json"

    bridge = importlib.import_module(module).TeleopRuntimeBridge(
        request_file,
        status_file,
        structure_stop_callback=lambda active: stop_calls.append(active),
        camera_callback=lambda eye, target: cameras.append((eye, target)),
        center_callback=lambda: (3.0, 4.0, 0.2),
    )
    return bridge, request_file, status_file, stop_calls, cameras


def write(
    request_file,
    *,
    active=None,
    request_id="r1",
    camera=None,
):
    payload = {
        "schema_version": "mssr.teleop_runtime.v1",
        "camera": camera,
    }
    if active is not None:
        payload["structure_stop_request"] = {
            "id": request_id,
            "active": active,
        }
    request_file.write_text(json.dumps(payload))


def test_structure_stop_is_acknowledged_and_duplicate_is_idempotent(tmp_path):
    bridge, request_file, status_file, stop_calls, _ = runtime(tmp_path)

    write(request_file, active=True)
    status = bridge.poll()

    assert status["structure_stopped"] is True
    assert status["structure_stop_ack"] == {
        "id": "r1",
        "active": True,
        "applied": True,
    }
    assert stop_calls == [True]

    bridge.poll()
    assert stop_calls == [True]

    persisted = json.loads(status_file.read_text())
    assert persisted["structure_stop_ack"] == status["structure_stop_ack"]


def test_structure_stop_can_be_explicitly_cleared(tmp_path):
    bridge, request_file, _, stop_calls, _ = runtime(tmp_path)

    write(request_file, active=True, request_id="stop-1")
    bridge.poll()

    write(request_file, active=False, request_id="clear-1")
    status = bridge.poll()

    assert status["structure_stopped"] is False
    assert status["structure_stop_ack"] == {
        "id": "clear-1",
        "active": False,
        "applied": True,
    }
    assert stop_calls == [True, False]


def test_camera_remains_available_while_structure_is_stopped(tmp_path):
    bridge, request_file, _, _, cameras = runtime(tmp_path)

    write(request_file, active=True, request_id="stop-1")
    bridge.poll()

    write(
        request_file,
        camera={
            "azimuth_rad": 0.0,
            "elevation_rad": 0.5,
            "radius_m": 2.0,
        },
    )
    status = bridge.poll()

    assert status["structure_stopped"] is True
    assert status["camera_applied"] is True
    assert cameras[-1][1] == (3.0, 4.0, 0.2)


def test_invalid_camera_does_not_prevent_structure_stop(tmp_path):
    bridge, request_file, _, stop_calls, cameras = runtime(tmp_path)

    write(
        request_file,
        active=True,
        request_id="stop-1",
        camera={"radius_m": -1},
    )
    status = bridge.poll()

    assert status["structure_stopped"] is True
    assert status["structure_stop_ack"]["applied"] is True
    assert stop_calls == [True]
    assert cameras == []
    assert status["error"]


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        "[]",
        "null",
        '{"schema_version":"wrong"}',
        (
            '{"schema_version":"mssr.teleop_runtime.v1",'
            '"structure_stop_request":{"id":"r1","active":"yes"}}'
        ),
    ],
)
def test_malformed_request_cannot_change_structure_state(tmp_path, payload):
    bridge, request_file, _, stop_calls, _ = runtime(tmp_path)

    request_file.write_text(payload)
    status = bridge.poll()

    assert status["error"]
    assert status["structure_stopped"] is False
    assert stop_calls == []


def test_no_request_reports_current_structure_state(tmp_path):
    bridge, _, _, stop_calls, _ = runtime(tmp_path)

    status = bridge.poll()

    assert status["structure_stopped"] is False
    assert status["structure_stop_ack"] is None
    assert stop_calls == []
