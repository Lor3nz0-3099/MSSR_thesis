"""T5 structural-macro launcher contract."""

from pathlib import Path

import pytest

from mssr_expert.teleop.state import TeleopState
from mssr_expert.teleop.structural_macro import StructuralMacroLauncher


class FakeProcess:
    def __init__(self, returncode=None):
        self.returncode = returncode

    def poll(self):
        return self.returncode


def ready_state(target: str = "snake8") -> TeleopState:
    state = TeleopState()
    state.start_ready()
    state.set_connected(True)
    state.observe_topology("rc_car8")

    assert state.request_morphology(target)
    assert state.authority == "TELEOP"
    assert not state.macro_active

    return state


def parameter_value(command, name):
    needle = f"{name}:="

    for item in command:
        if isinstance(item, str) and item.startswith(needle):
            return item[len(needle):]

    raise AssertionError(
        f"missing ROS parameter {name!r} in command: {command!r}"
    )


def test_successful_spawn_claims_structural_macro_and_builds_real_reconfiguration_command(
    tmp_path,
):
    calls = []

    def spawn(command):
        calls.append(tuple(command))
        return FakeProcess()

    state = ready_state("snake8")
    dataset = tmp_path / "structural" / "reconfiguration.jsonl"

    launcher = StructuralMacroLauncher(spawn=spawn)

    process = launcher.start(
        state=state,
        target_morphology="snake8",
        execution_id="teleop-macro-0001",
        episode_id="teleop-episode-0001",
        dataset_path=dataset,
    )

    assert isinstance(process, FakeProcess)
    assert len(calls) == 1

    command = calls[0]

    assert command[:4] == (
        "ros2",
        "run",
        "mssr_expert",
        "mssr_smores_self_reconfiguration_node",
    )

    assert "--ros-args" in command

    assert parameter_value(
        command,
        "source_graph_path",
    ) == "auto"

    assert parameter_value(
        command,
        "target_morphology",
    ) == "snake8"

    assert parameter_value(
        command,
        "execution_id",
    ) == "teleop-macro-0001"

    assert parameter_value(
        command,
        "episode_id",
    ) == "teleop-episode-0001"

    assert parameter_value(
        command,
        "dataset_path",
    ) == str(dataset)

    # Authority changes only after a live process has been returned.
    assert state.macro_active
    assert state.phase == "STRUCTURAL_MACRO"
    assert state.authority == "STRUCTURAL_MACRO"

    assert launcher.active
    assert launcher.process is process
    assert launcher.target_morphology == "snake8"


def test_failed_spawn_does_not_claim_structural_macro_authority(tmp_path):
    def spawn(_command):
        raise OSError("simulated launch failure")

    state = ready_state("snake8")

    launcher = StructuralMacroLauncher(spawn=spawn)

    with pytest.raises(OSError, match="simulated launch failure"):
        launcher.start(
            state=state,
            target_morphology="snake8",
            execution_id="teleop-macro-fail",
            episode_id="teleop-episode-fail",
            dataset_path=tmp_path / "failed.jsonl",
        )

    assert not state.macro_active
    assert state.phase == "READY"
    assert state.authority == "TELEOP"

    assert not launcher.active
    assert launcher.process is None



def test_nonterminal_expert_state_keeps_structural_macro_active(tmp_path):
    state = ready_state("snake8")
    process = FakeProcess()

    launcher = StructuralMacroLauncher(
        spawn=lambda _command: process,
    )

    launcher.start(
        state=state,
        target_morphology="snake8",
        execution_id="macro-live",
        episode_id="episode-live",
        dataset_path=tmp_path / "live.jsonl",
    )

    consumed = launcher.observe_expert_state(
        state=state,
        payload={
            "schema_version": "mssr.self_reconfiguration_state.v1",
            "target_morphology": "snake8",
            "done": False,
            "success": False,
        },
    )

    assert consumed is False
    assert state.macro_active
    assert state.authority == "STRUCTURAL_MACRO"
    assert launcher.active


@pytest.mark.parametrize("success", [True, False])
def test_matching_terminal_expert_state_finishes_macro(
    tmp_path,
    success,
):
    state = ready_state("snake8")
    process = FakeProcess()

    launcher = StructuralMacroLauncher(
        spawn=lambda _command: process,
    )

    launcher.start(
        state=state,
        target_morphology="snake8",
        execution_id="macro-terminal",
        episode_id="episode-terminal",
        dataset_path=tmp_path / "terminal.jsonl",
    )

    consumed = launcher.observe_expert_state(
        state=state,
        payload={
            "schema_version": "mssr.self_reconfiguration_state.v1",
            "target_morphology": "snake8",
            "done": True,
            "success": success,
        },
    )

    assert consumed is True

    assert not state.macro_active
    assert state.phase == "READY"
    assert state.authority == "TELEOP"
    assert state.last_macro_success is success

    assert launcher.process is None
    assert launcher.target_morphology is None
    assert not launcher.active


def test_terminal_state_for_wrong_target_is_ignored(tmp_path):
    state = ready_state("snake8")
    process = FakeProcess()

    launcher = StructuralMacroLauncher(
        spawn=lambda _command: process,
    )

    launcher.start(
        state=state,
        target_morphology="snake8",
        execution_id="macro-target",
        episode_id="episode-target",
        dataset_path=tmp_path / "target.jsonl",
    )

    consumed = launcher.observe_expert_state(
        state=state,
        payload={
            "schema_version": "mssr.self_reconfiguration_state.v1",
            "target_morphology": "mobile_manipulator8",
            "done": True,
            "success": True,
        },
    )

    assert consumed is False
    assert state.macro_active
    assert launcher.active



class TerminableFakeProcess(FakeProcess):
    def __init__(self):
        super().__init__()
        self.terminated = False

    def terminate(self):
        self.terminated = True
        self.returncode = -15


def test_estop_interrupt_terminates_macro_and_returns_active_goals(tmp_path):
    state = ready_state("snake8")
    process = TerminableFakeProcess()

    launcher = StructuralMacroLauncher(
        spawn=lambda _command: process,
    )

    launcher.start(
        state=state,
        target_morphology="snake8",
        execution_id="macro-estop",
        episode_id="episode-estop",
        dataset_path=tmp_path / "estop.jsonl",
    )

    assert state.authority == "STRUCTURAL_MACRO"

    # The structural expert tells us which native primitives it currently owns.
    consumed = launcher.observe_expert_state(
        state=state,
        payload={
            "schema_version": "mssr.self_reconfiguration_state.v1",
            "target_morphology": "snake8",
            "active_goal_ids": [
                "macro-estop-tilt-0",
                "macro-estop-align-1",
            ],
            "done": False,
            "success": False,
        },
    )

    assert consumed is False

    # E-STOP remains the authoritative state transition.
    state.pause()

    assert state.authority == "ESTOP"
    assert not state.macro_active
    assert state.last_macro_success is None

    cancel_goal_ids = launcher.interrupt()

    assert process.terminated

    assert cancel_goal_ids == (
        "macro-estop-align-1",
        "macro-estop-tilt-0",
    )

    assert launcher.process is None
    assert launcher.target_morphology is None
    assert not launcher.active

    # An old terminal packet from the interrupted expert must be inert.
    consumed = launcher.observe_expert_state(
        state=state,
        payload={
            "schema_version": "mssr.self_reconfiguration_state.v1",
            "target_morphology": "snake8",
            "active_goal_ids": [],
            "done": True,
            "success": True,
        },
    )

    assert consumed is False
    assert state.authority == "ESTOP"
    assert state.last_macro_success is None


def test_interrupted_macro_does_not_auto_resume_after_estop_clear(tmp_path):
    state = ready_state("snake8")
    process = TerminableFakeProcess()

    launcher = StructuralMacroLauncher(
        spawn=lambda _command: process,
    )

    launcher.start(
        state=state,
        target_morphology="snake8",
        execution_id="macro-no-resume",
        episode_id="episode-no-resume",
        dataset_path=tmp_path / "no-resume.jsonl",
    )

    state.pause()
    launcher.interrupt()

    assert state.authority == "ESTOP"
    assert not state.macro_active
    assert not launcher.active

    # Runtime ACK would eventually authorize this resume.
    assert state.resume()

    assert state.phase == "READY"
    assert state.authority == "TELEOP"
    assert not state.macro_active
    assert not launcher.active

    # The requested target may remain informational, but it must not
    # spontaneously restart the structural expert.
    assert state.requested_morphology == "snake8"



def test_terminal_packet_is_authoritative_even_if_process_already_exited(
    tmp_path,
):
    """ROS terminal state may arrive just after the expert process exits."""

    state = ready_state("snake8")
    process = FakeProcess()

    launcher = StructuralMacroLauncher(
        spawn=lambda _command: process,
    )

    launcher.start(
        state=state,
        target_morphology="snake8",
        execution_id="macro-terminal-race",
        episode_id="episode-terminal-race",
        dataset_path=tmp_path / "terminal-race.jsonl",
    )

    assert state.authority == "STRUCTURAL_MACRO"
    assert launcher.active

    # The expert publishes its terminal state and exits immediately.
    # The ROS callback can observe the message only after poll() reports exit.
    process.returncode = 0

    assert not launcher.active
    assert launcher.process is process

    consumed = launcher.observe_expert_state(
        state=state,
        payload={
            "schema_version": "mssr.self_reconfiguration_state.v1",
            "target_morphology": "snake8",
            "active_goal_ids": [],
            "done": True,
            "success": True,
        },
    )

    assert consumed is True
    assert not state.macro_active
    assert state.phase == "READY"
    assert state.last_macro_success is True

    assert launcher.process is None
    assert launcher.target_morphology is None
