"""T2 input/state/safety/runtime coordination without starting ROS or Isaac."""
import importlib
import importlib.util
from pathlib import Path

import yaml

from mssr_expert.teleop.input import InputConfig
from mssr_expert.teleop.session import TeleopSession


def coordinator():
    name = "mssr_expert.teleop.coordinator"
    assert importlib.util.find_spec(name) is not None, "missing T2 runtime coordinator"
    mapping = yaml.safe_load((Path(__file__).parents[1] / "config/smores_dualsense.yaml").read_text())
    # These assignments are synthetic test fixtures, not shipped physical decisions.
    mapping["commands"].update(estop="ps", resume="share")
    session = TeleopSession(InputConfig.from_mapping(mapping))
    return importlib.import_module(name).RuntimeCoordinator(session)


def send(core, at, pressed=(), left_x=0.0, r2=0.0):
    axes = [-left_x, 0.0, 0.0, 0.0, 0.0, -r2]
    buttons = [0] * 21
    for index in pressed:
        buttons[index] = 1
    assert core.session.update_joy(axes, buttons, at)


def ack(core, at, *, playing, request=None):
    payload = {"schema_version": "mssr.teleop_runtime_status.v1", "stamp_monotonic": at,
               "timeline_playing": playing, "timeline_ack":
               {**request, "applied": True} if request else None, "error": None}
    core.observe_runtime(payload, at)


def test_no_runtime_ack_cannot_enable_motion_even_with_verified_topology():
    core = coordinator()
    core.session.state.observe_topology("rc_car8")
    send(core, 10.0)
    status, runtime = core.tick(10.01)
    assert not status["runtime_bridge_ready"]
    assert not status["safety"]["motion_enabled"] and status["safety"]["zero_wheels"]
    assert runtime["schema_version"] == "mssr.teleop_runtime.v1"


def test_estop_edge_requests_pause_once_and_hold_retries_same_id():
    core = coordinator()
    send(core, 10.0)
    core.tick(10.01)
    send(core, 10.02, [5])
    status, runtime = core.tick(10.03)
    request = runtime["timeline_request"]
    assert request["operation"] == "pause" and status["phase"] == "ESTOP_PAUSED"
    send(core, 10.04, [5])
    assert core.tick(10.05)[1]["timeline_request"] == request


def test_resume_keeps_estop_until_matching_actual_timeline_ack():
    core = coordinator()
    send(core, 10.0)
    core.tick(10.01)
    send(core, 10.02, [5])
    pause = core.tick(10.03)[1]["timeline_request"]
    ack(core, 10.04, playing=False, request=pause)
    send(core, 10.05)
    core.tick(10.06)
    send(core, 10.07, [4])
    status, runtime = core.tick(10.08)
    assert status["phase"] == "ESTOP_PAUSED" and not status["safety"]["motion_enabled"]
    resume = runtime["timeline_request"]
    assert resume["operation"] == "resume"
    ack(core, 10.09, playing=True, request=resume)
    assert core.tick(10.10)[0]["phase"] == "READY"


def test_resume_ack_fences_cached_neutral_until_post_ack_joy_arrives():
    core = coordinator()
    core.session.state.observe_topology("rc_car8")
    send(core, 10.0)
    ack(core, 10.0, playing=True)
    core.tick(10.01)
    send(core, 10.02, [5])
    pause = core.tick(10.03)[1]["timeline_request"]
    ack(core, 10.04, playing=False, request=pause)
    send(core, 10.05)
    core.tick(10.06)
    send(core, 10.07, [4])
    resume = core.tick(10.08)[1]["timeline_request"]
    # Packet arrived before ACK, but is first consumed after it.
    send(core, 10.09)
    ack(core, 10.10, playing=True, request=resume)
    assert not core.tick(10.11)[0]["safety"]["motion_enabled"]
    send(core, 10.12)
    assert core.tick(10.13)[0]["safety"]["motion_enabled"]


def test_new_estop_during_resume_delivery_cannot_be_cleared_by_late_resume_ack():
    core = coordinator()
    send(core, 10.0)
    core.tick(10.01)
    send(core, 10.02, [5])
    pause = core.tick(10.03)[1]["timeline_request"]
    ack(core, 10.04, playing=False, request=pause)
    send(core, 10.05)
    core.tick(10.06)
    send(core, 10.07, [4])
    resume = core.tick(10.08)[1]["timeline_request"]
    send(core, 10.09)
    core.tick(10.10)
    send(core, 10.11, [5])
    new_pause = core.tick(10.12)[1]["timeline_request"]
    assert new_pause["operation"] == "pause"
    ack(core, 10.13, playing=True, request=resume)
    status, runtime = core.tick(10.14)
    assert status["phase"] == "ESTOP_PAUSED" and runtime["timeline_request"] == new_pause


def test_simultaneous_estop_resume_keeps_pause_priority():
    core = coordinator()
    send(core, 10.0)
    core.tick(10.01)
    send(core, 10.02, [4, 5])
    status, runtime = core.tick(10.03)
    assert status["phase"] == "ESTOP_PAUSED"
    assert runtime["timeline_request"]["operation"] == "pause"


def test_disconnect_preserves_macro_recording_and_does_not_overwrite_macro_wheels():
    core = coordinator()
    state = core.session.state
    state.request_morphology("snake8")
    state.begin_macro()
    state.toggle_recording()
    send(core, 10.0)
    core.tick(10.01)
    ack(core, 10.6, playing=True)
    status, _ = core.tick(10.61)
    assert status["macro_active"] and status["recording"]
    assert status["safety"]["authority"] == "STRUCTURAL_MACRO"
    assert not status["safety"]["zero_wheels"] and not status["safety"]["allow_joint_updates"]
    assert "controller_disconnected" in status["events"]


def test_left_stick_changes_only_separate_camera_intent():
    core = coordinator()
    send(core, 10.0)
    _, initial = core.tick(10.01)
    send(core, 10.02, left_x=1.0)
    _, moved = core.tick(10.03)
    assert moved["camera"]["azimuth_rad"] != initial["camera"]["azimuth_rad"]
    assert moved["timeline_request"] is None
    assert not {"actions", "commands", "modules"}.intersection(moved)


def test_external_pause_requires_explicit_resume_not_fresh_neutral_alone():
    core = coordinator()
    send(core, 10.0)
    ack(core, 10.0, playing=False)
    status, runtime = core.tick(10.01)
    assert status["phase"] == "ESTOP_PAUSED"
    assert not status["safety"]["motion_enabled"] and runtime["timeline_request"] is None
