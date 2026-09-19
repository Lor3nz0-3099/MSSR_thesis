"""Unexpected structural-expert exit must not latch macro authority forever."""

from pathlib import Path

from mssr_expert.teleop.state import TeleopState
from mssr_expert.teleop.structural_macro import StructuralMacroLauncher


class FakeProcess:
    def __init__(self):
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True


def started_macro():
    state = TeleopState()
    state.start_ready()
    assert state.request_morphology("snake8")

    process = FakeProcess()
    launcher = StructuralMacroLauncher(
        spawn=lambda command: process,
    )

    launcher.start(
        state=state,
        target_morphology="snake8",
        execution_id="watchdog-test",
        episode_id="watchdog-episode",
        dataset_path=Path("/tmp/watchdog-test.jsonl"),
    )

    assert state.macro_active

    return state, launcher, process


def running_payload():
    return {
        "schema_version": "mssr.self_reconfiguration_state.v1",
        "target_morphology": "snake8",
        "active_goal_ids": ["goal-b", "goal-a"],
        "done": False,
        "success": False,
    }


def terminal_payload():
    return {
        "schema_version": "mssr.self_reconfiguration_state.v1",
        "target_morphology": "snake8",
        "active_goal_ids": [],
        "done": True,
        "success": True,
    }


def test_process_exit_waits_for_terminal_packet_then_fails_after_grace():
    state, launcher, process = started_macro()

    assert not launcher.observe_expert_state(
        state=state,
        payload=running_payload(),
    )

    process.returncode = 17

    # First observation of process death only starts the grace period.
    assert launcher.check_process(
        state=state,
        now=100.0,
        exit_grace_s=0.5,
    ) is None
    assert state.macro_active
    assert state.last_macro_success is None

    # Still inside the grace period: keep structural ownership.
    assert launcher.check_process(
        state=state,
        now=100.49,
        exit_grace_s=0.5,
    ) is None
    assert state.macro_active

    # No authoritative terminal packet arrived: fail closed.
    cancel_goal_ids = launcher.check_process(
        state=state,
        now=100.51,
        exit_grace_s=0.5,
    )

    assert cancel_goal_ids == ("goal-a", "goal-b")
    assert not state.macro_active
    assert state.last_macro_success is False
    assert launcher.process is None
    assert launcher.target_morphology is None


def test_terminal_packet_during_exit_grace_remains_authoritative():
    state, launcher, process = started_macro()

    process.returncode = 0

    # Process is gone, but ROS may still deliver its last state packet.
    assert launcher.check_process(
        state=state,
        now=200.0,
        exit_grace_s=0.5,
    ) is None
    assert state.macro_active

    consumed = launcher.observe_expert_state(
        state=state,
        payload=terminal_payload(),
    )

    assert consumed
    assert not state.macro_active
    assert state.last_macro_success is True
    assert launcher.process is None

    # The watchdog must now be inert.
    assert launcher.check_process(
        state=state,
        now=201.0,
        exit_grace_s=0.5,
    ) is None
