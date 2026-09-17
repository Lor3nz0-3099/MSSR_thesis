"""Structure-stop delivery, acknowledgments and stale runtime gating."""
import importlib
import importlib.util

import pytest


def channel():
    module = "mssr_expert.teleop.runtime_channel"
    assert importlib.util.find_spec(module) is not None
    return importlib.import_module(module).RuntimeChannel()


def acknowledgment(request, *, stopped, stamp=10.0):
    return {
        "schema_version": "mssr.teleop_runtime_status.v1",
        "stamp_monotonic": stamp,
        "structure_stopped": stopped,
        "structure_stop_ack": {
            **request,
            "applied": True,
        },
        "error": None,
    }


def test_stop_retries_keep_same_id_and_clear_waits_for_stop_ack():
    core = channel()

    core.request_stop(True)
    first = core.payload(None)["structure_stop_request"]

    core.request_stop(False)

    assert core.payload(None)["structure_stop_request"] == first

    assert (
        core.observe(
            acknowledgment(first, stopped=True),
            10.0,
        )
        == "stop"
    )

    second = core.payload(None)["structure_stop_request"]

    assert second["active"] is False
    assert second["id"] != first["id"]

    assert (
        core.observe(
            acknowledgment(second, stopped=False),
            10.0,
        )
        == "clear"
    )

    assert core.payload(None)["structure_stop_request"] is None


def test_duplicate_ack_cannot_consume_next_command():
    core = channel()

    core.request_stop(True)
    first = core.payload(None)["structure_stop_request"]

    core.request_stop(False)

    core.observe(
        acknowledgment(first, stopped=True),
        10.0,
    )

    assert (
        core.observe(
            acknowledgment(first, stopped=True),
            10.0,
        )
        is None
    )

    assert (
        core.payload(None)["structure_stop_request"]["active"]
        is False
    )


def test_wrong_observed_structure_state_cannot_acknowledge_stop():
    core = channel()

    core.request_stop(True)
    first = core.payload(None)["structure_stop_request"]

    assert (
        core.observe(
            acknowledgment(first, stopped=False),
            10.0,
        )
        is None
    )

    assert core.payload(None)["structure_stop_request"] == first


def test_stale_file_republication_does_not_keep_runtime_ready():
    core = channel()

    payload = {
        "schema_version": "mssr.teleop_runtime_status.v1",
        "stamp_monotonic": 10.0,
        "structure_stopped": False,
        "structure_stop_ack": None,
        "error": None,
    }

    core.observe(payload, 10.0)
    assert core.ready(10.1)

    core.observe(payload, 11.0)
    assert not core.ready(11.0)


def test_camera_payload_has_no_robot_command_fields():
    core = channel()

    camera = {
        "azimuth_rad": 0.0,
        "elevation_rad": 0.5,
        "radius_m": 2.0,
    }

    assert core.payload(camera) == {
        "schema_version": "mssr.teleop_runtime.v1",
        "structure_stop_request": None,
        "camera": camera,
    }


def test_estop_supersedes_unacknowledged_clear_and_ignores_late_ack():
    core = channel()

    core.request_stop(False)
    clear = core.payload(None)["structure_stop_request"]

    core.request_stop(True)
    stop = core.payload(None)["structure_stop_request"]

    assert stop["active"] is True

    assert (
        core.observe(
            acknowledgment(clear, stopped=False),
            10.0,
        )
        is None
    )

    assert core.payload(None)["structure_stop_request"] == stop


def test_new_estop_discards_clear_queued_behind_pending_stop():
    core = channel()

    core.request_stop(True)
    first = core.payload(None)["structure_stop_request"]

    core.request_stop(False)
    core.request_stop(True)

    assert core.payload(None)["structure_stop_request"] == first

    assert (
        core.observe(
            acknowledgment(first, stopped=True),
            10.0,
        )
        == "stop"
    )

    assert core.payload(None)["structure_stop_request"] is None


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {},
        {"schema_version": "wrong"},
        {
            "schema_version": "mssr.teleop_runtime_status.v1",
            "stamp_monotonic": float("nan"),
            "structure_stopped": False,
        },
        {
            "schema_version": "mssr.teleop_runtime_status.v1",
            "stamp_monotonic": 10.0,
            "structure_stopped": "false",
        },
    ],
)
def test_invalid_runtime_status_cannot_grant_readiness(payload):
    core = channel()

    assert core.observe(payload, 10.0) is None
    assert not core.ready(10.0)
