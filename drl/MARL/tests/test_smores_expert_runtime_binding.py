from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize("terminal", [None, True, False])
def test_owned_expert_is_reaped_and_can_restart(tmp_path, monkeypatch, terminal):
    import signal
    from drl.MARL import smores_expert_runtime_binding as runtime_binding

    config = tmp_path / "repo/mssr_ws/src/mssr_expert/config"
    config.mkdir(parents=True)
    (config / "smores_rc_car8.json").write_text("{}")
    processes = []
    signals = []

    class Process:
        returncode = None

        def __init__(self):
            self.pid = 100 + len(processes)

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return self.returncode

    def spawn(*args, **kwargs):
        process = Process()
        processes.append(process)
        return process

    monkeypatch.setattr("os.killpg", lambda pid, sig: signals.append((pid, sig)))
    payload = None if terminal is None else {
        "schema_version": "mssr.self_assembly_state.v1",
        "done": True, "success": terminal, "message": "finished",
    }
    binding = runtime_binding.SMORESExpertRuntimeBinding(
        repository_root=tmp_path / "repo",
        runtime_dirs=(tmp_path / "runtime",),
        expert_state_sources=(lambda: payload,),
        current_graph_sources=(lambda: None,),
        morphology_catalog={"rc_car8": object()},
        topology_matcher=object(),
        process_factory=spawn,
    )
    assert binding.launch_expert(0, "rc_car8")
    log = binding._log_streams[0]

    if terminal is None:
        binding.close()
    else:
        assert binding.expert_status(0) is terminal

    assert signals == [(processes[0].pid, signal.SIGTERM)]
    assert processes[0].poll() is not None
    assert log.closed
    assert binding.launch_expert(0, "rc_car8")
    binding.close()
    binding.close()
    assert all(process.poll() is not None for process in processes)


def test_builds_existing_rc_car8_self_assembly_command(tmp_path):
    from drl.MARL.smores_expert_runtime_binding import (
        build_self_assembly_command,
    )

    repo = tmp_path / "repo"

    config = (
        repo
        / "mssr_ws/src/mssr_expert/config"
    )

    config.mkdir(
        parents=True
    )

    (
        config
        / "smores_rc_car8.json"
    ).write_text(
        "{}"
    )

    runtime = (
        tmp_path
        / "runtime"
    )

    command = (
        build_self_assembly_command(
            repository_root=repo,
            runtime_dir=runtime,
            target_morphology="rc_car8",
            episode_id="marl-env0",
        )
    )

    assert command[:4] == (
        "ros2",
        "run",
        "mssr_expert",
        "mssr_smores_self_assembly_node",
    )

    assert (
        "target_graph_path:="
        + str(
            config
            / "smores_rc_car8.json"
        )
        in command
    )

    assert (
        "execution_id:=marl-env0-assembly"
        in command
    )

    assert (
        "episode_id:=marl-env0"
        in command
    )

    assert (
        "dataset_path:="
        + str(
            runtime
            / "assembly_dataset.jsonl"
        )
        in command
    )


def test_self_assembly_state_is_none_until_terminal():
    from drl.MARL.smores_expert_runtime_binding import (
        parse_self_assembly_terminal,
    )

    assert (
        parse_self_assembly_terminal(
            {
                "schema_version":
                    "mssr.self_assembly_state.v1",
                "done": False,
                "success": False,
                "message": "RUNNING",
            }
        )
        is None
    )


@pytest.mark.parametrize(
    (
        "success",
        "expected",
    ),
    (
        (
            True,
            True,
        ),
        (
            False,
            False,
        ),
    ),
)
def test_self_assembly_state_returns_real_terminal_success(
    success,
    expected,
):
    from drl.MARL.smores_expert_runtime_binding import (
        parse_self_assembly_terminal,
    )

    result = (
        parse_self_assembly_terminal(
            {
                "schema_version":
                    "mssr.self_assembly_state.v1",
                "done": True,
                "success": success,
                "message":
                    "completed"
                    if success
                    else "failed",
            }
        )
    )

    assert result is expected


def test_topology_detection_uses_existing_configuration_assignment():
    from drl.MARL.smores_expert_runtime_binding import (
        detect_morphology_name,
    )

    class Matcher:
        def __init__(self):
            self.calls = []

        def configuration_assignment(
            self,
            current_graph,
            target_graph,
        ):
            self.calls.append(
                (
                    current_graph,
                    target_graph,
                )
            )

            if (
                current_graph["shape"]
                == target_graph["shape"]
            ):
                return object()

            return None

    matcher = Matcher()

    current = {
        "shape": "rc",
    }

    catalog = {
        "rc_car8": {
            "shape": "rc",
        },
        "snake8": {
            "shape": "snake",
        },
        "mobile_manipulator8": {
            "shape": "arm",
        },
    }

    assert (
        detect_morphology_name(
            current_graph=current,
            morphology_catalog=catalog,
            topology_matcher=matcher,
        )
        == "rc_car8"
    )

    assert len(
        matcher.calls
    ) == 3


def test_topology_detection_returns_none_if_no_known_morphology_matches():
    from drl.MARL.smores_expert_runtime_binding import (
        detect_morphology_name,
    )

    class Matcher:
        def configuration_assignment(
            self,
            current_graph,
            target_graph,
        ):
            return None

    assert (
        detect_morphology_name(
            current_graph=object(),
            morphology_catalog={
                "rc_car8": object(),
            },
            topology_matcher=Matcher(),
        )
        is None
    )


def test_topology_detection_rejects_ambiguous_match():
    from drl.MARL.smores_expert_runtime_binding import (
        detect_morphology_name,
    )

    class Matcher:
        def configuration_assignment(
            self,
            current_graph,
            target_graph,
        ):
            return object()

    with pytest.raises(
        ValueError,
        match="ambiguous",
    ):
        detect_morphology_name(
            current_graph=object(),
            morphology_catalog={
                "rc_car8": object(),
                "snake8": object(),
            },
            topology_matcher=Matcher(),
        )


def test_binding_plugs_directly_into_direct_marl_env(tmp_path, monkeypatch):
    from drl.MARL.direct_marl_env import (
        DirectMARLEnv,
    )
    from drl.MARL.runtime_handoff import (
        DirectMARLHandoff,
    )
    from drl.MARL.smores_expert_runtime_binding import (
        SMORESExpertRuntimeBinding,
    )

    repo = tmp_path / "repo"

    config = (
        repo
        / "mssr_ws/src/mssr_expert/config"
    )

    config.mkdir(
        parents=True
    )

    (
        config
        / "smores_rc_car8.json"
    ).write_text(
        "{}"
    )

    runtime = (
        tmp_path
        / "runtime"
    )

    terminal = {
        "payload": {
            "schema_version":
                "mssr.self_assembly_state.v1",
            "done": False,
            "success": False,
            "message": "RUNNING",
        }
    }

    graph = {
        "payload": {
            "shape": "none",
        }
    }

    monkeypatch.setattr("os.killpg", lambda *_: None)

    class Process:
        pid = 100

        def poll(
            self,
        ):
            return None

        def wait(self, timeout=None):
            return 0

    spawned = []

    def process_factory(
        command,
        **kwargs,
    ):
        spawned.append(
            (
                tuple(
                    command
                ),
                kwargs,
            )
        )

        return Process()

    class Matcher:
        def configuration_assignment(
            self,
            current_graph,
            target_graph,
        ):
            if (
                current_graph["shape"]
                == target_graph["shape"]
            ):
                return object()

            return None

    binding = (
        SMORESExpertRuntimeBinding(
            repository_root=repo,
            runtime_dirs=(
                runtime,
            ),
            expert_state_sources=(
                lambda:
                    terminal[
                        "payload"
                    ],
            ),
            current_graph_sources=(
                lambda:
                    graph[
                        "payload"
                    ],
            ),
            morphology_catalog={
                "rc_car8": {
                    "shape": "rc",
                },
                "snake8": {
                    "shape": "snake",
                },
            },
            topology_matcher=Matcher(),
            process_factory=(
                process_factory
            ),
            episode_id_factory=(
                lambda env_index:
                    f"marl-env{env_index}"
            ),
        )
    )

    class Bridge:
        def __init__(
            self,
        ):
            self.calls = []

        def step(
            self,
            env_index,
        ):
            self.calls.append(
                env_index
            )

            return "APPLIED"

    bridge = Bridge()

    handoff = (
        DirectMARLHandoff()
    )

    env = DirectMARLEnv(
        handoffs=(
            handoff,
        ),
        expert_launcher=(
            binding.launch_expert
        ),
        expert_status_reader=(
            binding.expert_status
        ),
        topology_observer=(
            binding.observe_topology
        ),
        actuation_bridge=bridge,
        target_morphology="rc_car8",
    )

    env.reset()

    assert len(
        spawned
    ) == 1

    assert handoff.marl_active is False

    # Expert reaches terminal SUCCESS.
    terminal[
        "payload"
    ] = {
        "schema_version":
            "mssr.self_assembly_state.v1",
        "done": True,
        "success": True,
        "message": "completed",
    }

    env.advance_handoff(
        0
    )

    # Success alone is NOT enough.
    assert handoff.marl_active is False

    # Physical graph now independently matches rc_car8.
    graph[
        "payload"
    ] = {
        "shape": "rc",
    }

    env.advance_handoff(
        0
    )

    assert handoff.marl_active is True

    assert (
        env.step_marl_actuation(
            0
        )
        == "APPLIED"
    )

    assert bridge.calls == [
        0
    ]


# --- FILE-BACKED REAL RUNTIME SOURCES ---

def test_file_sources_read_expert_state_and_decode_robot_graph(
    tmp_path,
):
    import json

    from drl.MARL.smores_expert_runtime_binding import (
        make_expert_state_file_source,
        make_robot_graph_file_source,
    )

    runtime = (
        tmp_path
        / "runtime"
    )

    runtime.mkdir()

    expert_payload = {
        "schema_version":
            "mssr.self_assembly_state.v1",
        "done": True,
        "success": True,
        "message": "completed",
    }

    graph_payload = {
        "schema_version":
            "mssr.attributed_robot_graph.v1",
        "sentinel": 42,
    }

    (
        runtime
        / "self_assembly_state.json"
    ).write_text(
        json.dumps(
            expert_payload
        )
    )

    (
        runtime
        / "robot_graph.json"
    ).write_text(
        json.dumps(
            graph_payload
        )
    )

    expert_source = (
        make_expert_state_file_source(
            runtime
        )
    )

    seen = []

    def decoder(
        payload,
    ):
        seen.append(
            payload
        )

        return {
            "decoded":
                payload[
                    "sentinel"
                ]
        }

    graph_source = (
        make_robot_graph_file_source(
            runtime,
            graph_decoder=decoder,
        )
    )

    assert (
        expert_source()
        == expert_payload
    )

    assert graph_source() == {
        "decoded": 42,
    }

    assert seen == [
        graph_payload
    ]


def test_missing_runtime_files_are_not_false_success_or_topology(
    tmp_path,
):
    from drl.MARL.smores_expert_runtime_binding import (
        make_expert_state_file_source,
        make_robot_graph_file_source,
    )

    runtime = (
        tmp_path
        / "runtime"
    )

    runtime.mkdir()

    assert (
        make_expert_state_file_source(
            runtime
        )()
        is None
    )

    assert (
        make_robot_graph_file_source(
            runtime,
            graph_decoder=lambda payload:
                payload,
        )()
        is None
    )


def test_corrupt_runtime_json_fails_explicitly(
    tmp_path,
):
    from drl.MARL.smores_expert_runtime_binding import (
        make_expert_state_file_source,
    )

    runtime = (
        tmp_path
        / "runtime"
    )

    runtime.mkdir()

    (
        runtime
        / "self_assembly_state.json"
    ).write_text(
        "{not-json"
    )

    source = (
        make_expert_state_file_source(
            runtime
        )
    )

    with pytest.raises(
        ValueError,
        match="Invalid JSON",
    ):
        source()


def test_launch_expert_removes_stale_terminal_state_before_spawn(
    tmp_path,
):
    import json

    from drl.MARL.smores_expert_runtime_binding import (
        SMORESExpertRuntimeBinding,
    )

    repo = (
        tmp_path
        / "repo"
    )

    config = (
        repo
        / "mssr_ws/src/mssr_expert/config"
    )

    config.mkdir(
        parents=True
    )

    (
        config
        / "smores_rc_car8.json"
    ).write_text(
        "{}"
    )

    runtime = (
        tmp_path
        / "runtime"
    )

    runtime.mkdir()

    stale = (
        runtime
        / "self_assembly_state.json"
    )

    stale.write_text(
        json.dumps(
            {
                "schema_version":
                    "mssr.self_assembly_state.v1",
                "done": True,
                "success": True,
                "message":
                    "OLD EPISODE",
            }
        )
    )

    class Process:
        def poll(
            self,
        ):
            return None

    observed = []

    def process_factory(
        command,
        **kwargs,
    ):
        # Stale terminal state must already be gone
        # before the new Expert is started.
        observed.append(
            stale.exists()
        )

        return Process()

    binding = (
        SMORESExpertRuntimeBinding(
            repository_root=repo,
            runtime_dirs=(
                runtime,
            ),
            expert_state_sources=(
                lambda:
                    None,
            ),
            current_graph_sources=(
                lambda:
                    None,
            ),
            morphology_catalog={
                "rc_car8": object(),
            },
            topology_matcher=object(),
            process_factory=(
                process_factory
            ),
            episode_id_factory=(
                lambda env_index:
                    "fresh-episode"
            ),
        )
    )

    assert (
        binding.launch_expert(
            0,
            "rc_car8",
        )
        is True
    )

    assert observed == [
        False
    ]

    assert not stale.exists()


def test_ros_file_bridge_contract_mirrors_expert_topic_to_runtime_file():
    from pathlib import Path

    source = Path(
        "ros2_bridge/mssr_file_bridge.py"
    ).read_text()

    assert (
        '"/mssr/expert/self_assembly/state"'
        in source
    )

    assert (
        "self._on_expert_state"
        in source
    )

    assert (
        "self._expert_state_file"
        in source
    )

    assert (
        '"--expert-state-file"'
        in source
    )

    assert (
        "expert_state_file="
        in source
    )


def test_smores_runtime_launch_wires_same_expert_state_file():
    from pathlib import Path

    source = Path(
        "mssr_ws/src/mssr_expert/launch/"
        "smores_runtime.launch.py"
    ).read_text()

    assert (
        '"self_assembly_state.json"'
        in source
    )

    assert (
        '"--expert-state-file"'
        in source
    )
