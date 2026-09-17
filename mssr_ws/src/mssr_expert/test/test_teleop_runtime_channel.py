"""Runtime delivery retries, ordered acknowledgments and stale bridge gating."""
import importlib
import importlib.util

import pytest


def channel():
    module = "mssr_expert.teleop.runtime_channel"
    assert importlib.util.find_spec(module) is not None, "missing T2 runtime channel"
    return importlib.import_module(module).RuntimeChannel()


def acknowledgment(request, *, playing, stamp=10.0):
    return {"schema_version": "mssr.teleop_runtime_status.v1", "stamp_monotonic": stamp,
            "timeline_playing": playing, "timeline_ack": {**request, "applied": True},
            "error": None}


def test_pause_retries_keep_same_id_and_resume_waits_for_pause_ack():
    core = channel()
    core.request("pause")
    first = core.payload(None)["timeline_request"]
    core.request("resume")
    assert core.payload(None)["timeline_request"] == first
    assert core.observe(acknowledgment(first, playing=False), 10.0) == "pause"
    second = core.payload(None)["timeline_request"]
    assert second["operation"] == "resume" and second["id"] != first["id"]
    assert core.observe(acknowledgment(second, playing=True), 10.0) == "resume"
    assert core.payload(None)["timeline_request"] is None


def test_duplicate_ack_cannot_consume_next_command():
    core = channel()
    core.request("pause")
    first = core.payload(None)["timeline_request"]
    core.request("resume")
    core.observe(acknowledgment(first, playing=False), 10.0)
    assert core.observe(acknowledgment(first, playing=False), 10.0) is None
    assert core.payload(None)["timeline_request"]["operation"] == "resume"


def test_wrong_observed_timeline_cannot_acknowledge_pause():
    core = channel()
    core.request("pause")
    first = core.payload(None)["timeline_request"]
    assert core.observe(acknowledgment(first, playing=True), 10.0) is None
    assert core.payload(None)["timeline_request"] == first


def test_stale_file_republication_does_not_keep_runtime_ready():
    core = channel()
    payload = {"schema_version": "mssr.teleop_runtime_status.v1", "stamp_monotonic": 10.0,
               "timeline_playing": True, "timeline_ack": None, "error": None}
    core.observe(payload, 10.0)
    assert core.ready(10.1)
    core.observe(payload, 11.0)
    assert not core.ready(11.0)


def test_camera_payload_has_no_robot_command_fields():
    core = channel()
    camera = {"azimuth_rad": 0.0, "elevation_rad": 0.5, "radius_m": 2.0}
    assert core.payload(camera) == {"schema_version": "mssr.teleop_runtime.v1",
                                    "timeline_request": None, "camera": camera}


def test_estop_supersedes_unacknowledged_resume_and_ignores_its_late_ack():
    core = channel()
    core.request("resume")
    resume = core.payload(None)["timeline_request"]
    core.request("pause")
    pause = core.payload(None)["timeline_request"]
    assert pause["operation"] == "pause"
    assert core.observe(acknowledgment(resume, playing=True), 10.0) is None
    assert core.payload(None)["timeline_request"] == pause


def test_new_estop_discards_resume_queued_behind_pending_pause():
    core = channel()
    core.request("pause")
    first = core.payload(None)["timeline_request"]
    core.request("resume")
    core.request("pause")
    assert core.payload(None)["timeline_request"] == first
    assert core.observe(acknowledgment(first, playing=False), 10.0) == "pause"
    assert core.payload(None)["timeline_request"] is None


@pytest.mark.parametrize("payload", [None, [], {}, {"schema_version": "wrong"},
    {"schema_version": "mssr.teleop_runtime_status.v1", "stamp_monotonic": float("nan"), "timeline_playing": True},
    {"schema_version": "mssr.teleop_runtime_status.v1", "stamp_monotonic": 10.0, "timeline_playing": "true"}])
def test_invalid_runtime_status_cannot_grant_readiness(payload):
    core = channel()
    assert core.observe(payload, 10.0) is None
    assert not core.ready(10.0)
