"""Structural launch must claim authority before actuator safety is evaluated."""

from pathlib import Path

import yaml

from mssr_expert.teleop.coordinator import RuntimeCoordinator
from mssr_expert.teleop.input import InputConfig
from mssr_expert.teleop.session import TeleopSession


CONFIG_DIR = Path(__file__).parents[1] / "config"


def configured_session():
    mapping = yaml.safe_load(
        (CONFIG_DIR / "smores_dualsense.yaml").read_text()
    )

    mapping["commands"].update(
        select_rc="cross",
        select_snake="triangle",
        select_mm8="square",
        estop_toggle=None,
    )

    return TeleopSession(
        InputConfig.from_mapping(mapping),
        controller_morphologies={"rc_car8"},
    )


def send(core, pressed=(), at=10.0):
    buttons = [0] * 21

    for index in pressed:
        buttons[index] = 1

    return core.update_joy(
        [0.0] * 6,
        buttons,
        at,
    )


def test_structural_request_handler_runs_before_same_tick_safety_decision():
    session = configured_session()

    launched = []

    def launch(target):
        launched.append(target)

        assert session.state.requested_morphology == target
        assert session.state.begin_macro()

        return True

    coordinator = RuntimeCoordinator(
        session,
        structural_request_handler=launch,
    )

    # First packet establishes controller connectivity.
    send(session, at=10.0)
    session.state.observe_topology("rc_car8")
    coordinator.tick(10.01)

    # Triangle is the synthetic select_snake binding in this test.
    send(session, [3], 10.10)

    status, _ = coordinator.tick(10.11)

    assert launched == ["snake8"]

    assert status["requested_morphology"] == "snake8"
    assert status["phase"] == "STRUCTURAL_MACRO"
    assert status["authority"] == "STRUCTURAL_MACRO"

    # Critical contract: no final TELEOP actuator tick is allowed to leak
    # after the structural expert has accepted ownership.
    assert status["safety"]["authority"] == "STRUCTURAL_MACRO"
    assert not status["safety"]["motion_enabled"]
    assert not status["safety"]["allow_joint_updates"]
