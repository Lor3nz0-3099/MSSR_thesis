"""T2 input/state/safety/structure-stop coordination without ROS or Isaac."""
import importlib
import importlib.util
from pathlib import Path

import yaml

from mssr_expert.teleop.input import InputConfig
from mssr_expert.teleop.session import TeleopSession


def coordinator():
    name = "mssr_expert.teleop.coordinator"
    assert importlib.util.find_spec(name) is not None

    mapping = yaml.safe_load(
        (
            Path(__file__).parents[1]
            / "config/smores_dualsense.yaml"
        ).read_text()
    )

    # Synthetic test assignments only, not physical button decisions.
    mapping["commands"].update(
        estop="ps",
        resume="share",
    )

    session = TeleopSession(
        InputConfig.from_mapping(mapping)
    )

    return importlib.import_module(
        name
    ).RuntimeCoordinator(session)


def send(
    core,
    at,
    pressed=(),
    left_x=0.0,
    r2=0.0,
):
    axes = [
        -left_x,
        0.0,
        0.0,
        0.0,
        0.0,
        -r2,
    ]

    buttons = [0] * 21

    for index in pressed:
        buttons[index] = 1

    assert core.session.update_joy(
        axes,
        buttons,
        at,
    )


def ack(
    core,
    at,
    *,
    stopped,
    request=None,
):
    payload = {
        "schema_version": "mssr.teleop_runtime_status.v1",
        "stamp_monotonic": at,
        "structure_stopped": stopped,
        "structure_stop_ack": (
            {
                **request,
                "applied": True,
            }
            if request
            else None
        ),
        "error": None,
    }

    core.observe_runtime(payload, at)


def test_no_runtime_status_cannot_enable_motion_even_with_verified_topology():
    core = coordinator()
    core.session.state.observe_topology("rc_car8")

    send(core, 10.0)

    status, runtime = core.tick(10.01)

    assert not status["runtime_bridge_ready"]
    assert not status["safety"]["motion_enabled"]
    assert status["safety"]["zero_wheels"]

    assert (
        runtime["schema_version"]
        == "mssr.teleop_runtime.v1"
    )


def test_estop_edge_requests_structure_stop_once_and_retries_same_id():
    core = coordinator()

    send(core, 10.0)
    core.tick(10.01)

    send(core, 10.02, [5])
    status, runtime = core.tick(10.03)

    request = runtime["structure_stop_request"]

    assert request["active"] is True
    assert status["phase"] == "ESTOP_PAUSED"

    send(core, 10.04, [5])

    assert (
        core.tick(10.05)[1]["structure_stop_request"]
        == request
    )


def test_resume_keeps_estop_until_matching_structure_clear_ack():
    core = coordinator()

    send(core, 10.0)
    core.tick(10.01)

    send(core, 10.02, [5])
    stop = core.tick(
        10.03
    )[1]["structure_stop_request"]

    ack(
        core,
        10.04,
        stopped=True,
        request=stop,
    )

    send(core, 10.05)
    core.tick(10.06)

    send(core, 10.07, [4])
    status, runtime = core.tick(10.08)

    assert status["phase"] == "ESTOP_PAUSED"
    assert not status["safety"]["motion_enabled"]

    clear = runtime["structure_stop_request"]

    assert clear["active"] is False

    ack(
        core,
        10.09,
        stopped=False,
        request=clear,
    )

    assert (
        core.tick(10.10)[0]["phase"]
        == "READY"
    )


def test_clear_ack_fences_cached_neutral_until_post_ack_joy_arrives():
    core = coordinator()

    core.session.state.observe_topology("rc_car8")

    send(core, 10.0)
    ack(core, 10.0, stopped=False)
    core.tick(10.01)

    send(core, 10.02, [5])
    stop = core.tick(
        10.03
    )[1]["structure_stop_request"]

    ack(
        core,
        10.04,
        stopped=True,
        request=stop,
    )

    send(core, 10.05)
    core.tick(10.06)

    send(core, 10.07, [4])
    clear = core.tick(
        10.08
    )[1]["structure_stop_request"]

    # Neutral packet received before the actual clear ACK.
    send(core, 10.09)

    ack(
        core,
        10.10,
        stopped=False,
        request=clear,
    )

    assert not core.tick(
        10.11
    )[0]["safety"]["motion_enabled"]

    send(core, 10.12)

    assert core.tick(
        10.13
    )[0]["safety"]["motion_enabled"]


def test_new_estop_during_clear_delivery_cannot_be_cleared_by_late_ack():
    core = coordinator()

    send(core, 10.0)
    core.tick(10.01)

    send(core, 10.02, [5])
    stop = core.tick(
        10.03
    )[1]["structure_stop_request"]

    ack(
        core,
        10.04,
        stopped=True,
        request=stop,
    )

    send(core, 10.05)
    core.tick(10.06)

    send(core, 10.07, [4])
    clear = core.tick(
        10.08
    )[1]["structure_stop_request"]

    send(core, 10.09)
    core.tick(10.10)

    send(core, 10.11, [5])
    new_stop = core.tick(
        10.12
    )[1]["structure_stop_request"]

    assert new_stop["active"] is True

    ack(
        core,
        10.13,
        stopped=False,
        request=clear,
    )

    status, runtime = core.tick(10.14)

    assert status["phase"] == "ESTOP_PAUSED"
    assert (
        runtime["structure_stop_request"]
        == new_stop
    )


def test_simultaneous_estop_resume_keeps_stop_priority():
    core = coordinator()

    send(core, 10.0)
    core.tick(10.01)

    send(core, 10.02, [4, 5])
    status, runtime = core.tick(10.03)

    assert status["phase"] == "ESTOP_PAUSED"
    assert (
        runtime["structure_stop_request"]["active"]
        is True
    )


def test_disconnect_preserves_macro_recording_and_does_not_overwrite_macro_wheels():
    core = coordinator()
    state = core.session.state

    state.request_morphology("snake8")
    state.begin_macro()
    state.toggle_recording()

    send(core, 10.0)
    core.tick(10.01)

    ack(
        core,
        10.6,
        stopped=False,
    )

    status, _ = core.tick(10.61)

    assert status["macro_active"]
    assert status["recording"]
    assert (
        status["safety"]["authority"]
        == "STRUCTURAL_MACRO"
    )
    assert not status["safety"]["zero_wheels"]
    assert not status["safety"]["allow_joint_updates"]
    assert (
        "controller_disconnected"
        in status["events"]
    )


def test_left_stick_changes_only_separate_camera_intent():
    core = coordinator()

    send(core, 10.0)
    _, initial = core.tick(10.01)

    send(
        core,
        10.02,
        left_x=1.0,
    )

    _, moved = core.tick(10.03)

    assert (
        moved["camera"]["azimuth_rad"]
        != initial["camera"]["azimuth_rad"]
    )

    assert moved["structure_stop_request"] is None

    assert not {
        "actions",
        "commands",
        "modules",
    }.intersection(moved)


def test_external_structure_stop_requires_explicit_resume_not_neutral_alone():
    core = coordinator()

    send(core, 10.0)

    ack(
        core,
        10.0,
        stopped=True,
    )

    status, runtime = core.tick(10.01)

    assert status["phase"] == "ESTOP_PAUSED"
    assert not status["safety"]["motion_enabled"]
    assert runtime["structure_stop_request"] is None

    # Fresh neutral input alone must not clear an externally observed stop.
    send(core, 10.02)

    status, _ = core.tick(10.03)

    assert status["phase"] == "ESTOP_PAUSED"
