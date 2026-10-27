from __future__ import annotations

from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from pathlib import Path
import json
import os
import signal
import subprocess
import time
from typing import Any


def _read_json_mapping(
    path: Path,
) -> Mapping[str, Any] | None:
    """Read one atomic runtime JSON mapping.

    Missing files mean that the producer has not published yet.
    Invalid JSON is an explicit runtime error rather than a silent
    false observation.
    """

    try:
        raw = path.read_text(
            encoding="utf-8"
        )
    except FileNotFoundError:
        return None
    except OSError as error:
        raise RuntimeError(
            f"Cannot read runtime JSON {path}: {error}"
        ) from error

    try:
        payload = json.loads(
            raw
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in runtime file {path}"
        ) from error

    if not isinstance(
        payload,
        Mapping,
    ):
        raise ValueError(
            f"Runtime JSON {path} must contain an object"
        )

    return payload


def make_expert_state_file_source(
    runtime_dir: Path,
):
    """Return a callable reading the mirrored assembly Expert state."""

    path = (
        Path(
            runtime_dir
        )
        .expanduser()
        .resolve()
        / "self_assembly_state.json"
    )

    def source():
        return _read_json_mapping(
            path
        )

    return source


def make_robot_graph_file_source(
    runtime_dir: Path,
    *,
    graph_decoder=None,
):
    """Return a callable reading the current physical attributed graph."""

    path = (
        Path(
            runtime_dir
        )
        .expanduser()
        .resolve()
        / "robot_graph.json"
    )

    def decode(
        payload,
    ):
        if graph_decoder is not None:
            return graph_decoder(
                payload
            )

        # Lazy import: the MARL module itself remains ROS-independent.
        from mssr_expert.graph.serialization import (
            attributed_graph_from_dict,
        )

        return (
            attributed_graph_from_dict(
                payload
            )
        )

    def source():
        payload = (
            _read_json_mapping(
                path
            )
        )

        if payload is None:
            return None

        return decode(
            payload
        )

    return source


SELF_ASSEMBLY_STATE_SCHEMA = (
    "mssr.self_assembly_state.v1"
)


def build_self_assembly_command(
    *,
    repository_root: Path,
    runtime_dir: Path,
    target_morphology: str,
    episode_id: str,
) -> tuple[str, ...]:
    """Build the repository's existing SMORES assembly command."""

    repository_root = (
        Path(
            repository_root
        )
        .expanduser()
        .resolve()
    )

    runtime_dir = (
        Path(
            runtime_dir
        )
        .expanduser()
        .resolve()
    )

    if (
        type(
            target_morphology
        )
        is not str
        or not target_morphology
    ):
        raise ValueError(
            "target_morphology must be a non-empty string"
        )

    if (
        type(
            episode_id
        )
        is not str
        or not episode_id
    ):
        raise ValueError(
            "episode_id must be a non-empty string"
        )

    target_graph_path = (
        repository_root
        / "mssr_ws"
        / "src"
        / "mssr_expert"
        / "config"
        / f"smores_{target_morphology}.json"
    )

    if not target_graph_path.is_file():
        raise FileNotFoundError(
            "Missing SMORES target morphology: "
            f"{target_graph_path}"
        )

    runtime_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        "ros2",
        "run",
        "mssr_expert",
        "mssr_smores_self_assembly_node",
        "--ros-args",
        "-p",
        (
            "target_graph_path:="
            + str(
                target_graph_path
            )
        ),
        "-p",
        (
            "execution_id:="
            + episode_id
            + "-assembly"
        ),
        "-p",
        (
            "episode_id:="
            + episode_id
        ),
        "-p",
        (
            "dataset_path:="
            + str(
                runtime_dir
                / "assembly_dataset.jsonl"
            )
        ),
    )


def parse_self_assembly_terminal(
    payload: Mapping[str, Any] | None,
) -> bool | None:
    """Return None while running, otherwise terminal success/failure."""

    if payload is None:
        return None

    if not isinstance(
        payload,
        Mapping,
    ):
        raise TypeError(
            "self-assembly state must be a mapping or None"
        )

    schema = payload.get(
        "schema_version"
    )

    if (
        schema
        != SELF_ASSEMBLY_STATE_SCHEMA
    ):
        raise ValueError(
            "Unsupported self-assembly state schema: "
            f"{schema!r}"
        )

    done = payload.get(
        "done"
    )

    if type(
        done
    ) is not bool:
        raise ValueError(
            "self-assembly state 'done' must be bool"
        )

    if not done:
        return None

    success = payload.get(
        "success"
    )

    if type(
        success
    ) is not bool:
        raise ValueError(
            "terminal self-assembly state "
            "must contain boolean success"
        )

    return success


def detect_morphology_name(
    *,
    current_graph: Any,
    morphology_catalog: Mapping[
        str,
        Any,
    ],
    topology_matcher: Any,
) -> str | None:
    """Identify the physical morphology using the existing matcher."""

    if not morphology_catalog:
        raise ValueError(
            "morphology_catalog must not be empty"
        )

    configuration_assignment = getattr(
        topology_matcher,
        "configuration_assignment",
        None,
    )

    if not callable(
        configuration_assignment
    ):
        raise TypeError(
            "topology_matcher must expose "
            "configuration_assignment()"
        )

    matches = []

    for (
        name,
        target_graph,
    ) in sorted(
        morphology_catalog.items()
    ):
        assignment = (
            configuration_assignment(
                current_graph,
                target_graph,
            )
        )

        if assignment is not None:
            matches.append(
                name
            )

    if not matches:
        return None

    if len(
        matches
    ) > 1:
        raise ValueError(
            "Physical topology match is ambiguous: "
            f"{matches}"
        )

    return matches[
        0
    ]


class SMORESExpertRuntimeBinding:
    """Bind DirectMARLEnv callbacks to existing SMORES runtime APIs."""

    def __init__(
        self,
        *,
        repository_root: Path,
        runtime_dirs: Sequence[
            Path
        ],
        expert_state_sources: Sequence[
            Callable[
                [],
                Mapping[
                    str,
                    Any,
                ]
                | None,
            ]
        ],
        current_graph_sources: Sequence[
            Callable[
                [],
                Any,
            ]
        ],
        morphology_catalog: Mapping[
            str,
            Any,
        ],
        topology_matcher: Any,
        process_factory: Callable[..., Any] = (
            subprocess.Popen
        ),
        episode_id_factory: (
            Callable[
                [int],
                str,
            ]
            | None
        ) = None,
    ) -> None:
        self.repository_root = (
            Path(
                repository_root
            )
            .expanduser()
            .resolve()
        )

        self.runtime_dirs = tuple(
            Path(
                path
            )
            .expanduser()
            .resolve()
            for path
            in runtime_dirs
        )

        self.expert_state_sources = tuple(
            expert_state_sources
        )

        self.current_graph_sources = tuple(
            current_graph_sources
        )

        if not self.runtime_dirs:
            raise ValueError(
                "at least one runtime directory is required"
            )

        if (
            len(
                self.expert_state_sources
            )
            != len(
                self.runtime_dirs
            )
            or len(
                self.current_graph_sources
            )
            != len(
                self.runtime_dirs
            )
        ):
            raise ValueError(
                "runtime sources must share environment count"
            )

        if not all(
            callable(
                source
            )
            for source
            in (
                *self.expert_state_sources,
                *self.current_graph_sources,
            )
        ):
            raise TypeError(
                "runtime sources must be callable"
            )

        self.morphology_catalog = dict(
            morphology_catalog
        )

        self.topology_matcher = (
            topology_matcher
        )

        self.process_factory = (
            process_factory
        )

        self.episode_id_factory = (
            episode_id_factory
            if episode_id_factory is not None
            else (
                lambda env_index:
                    "marl-"
                    f"env{env_index}-"
                    f"{time.time_ns()}"
            )
        )

        self._processes = [
            None
            for _
            in self.runtime_dirs
        ]

        self._log_streams = [
            None
            for _
            in self.runtime_dirs
        ]

    @property
    def num_envs(
        self,
    ) -> int:
        return len(
            self.runtime_dirs
        )

    def _validate_env_index(
        self,
        env_index: int,
    ) -> int:
        if (
            isinstance(
                env_index,
                bool,
            )
            or not isinstance(
                env_index,
                int,
            )
            or env_index < 0
            or env_index >= self.num_envs
        ):
            raise IndexError(
                f"invalid env_index: {env_index}"
            )

        return env_index

    def stop_expert(self, env_index: int) -> None:
        """Stop and reap this binding's Expert session before releasing it."""
        env_index = self._validate_env_index(env_index)
        process = self._processes[env_index]
        if process is not None:
            if process.poll() is None:
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(process.pid, sig)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5.0)
                        break
                    except subprocess.TimeoutExpired:
                        if sig == signal.SIGKILL:
                            raise
            else:
                process.wait()
            self._processes[env_index] = None
        log_stream = self._log_streams[env_index]
        if log_stream is not None:
            log_stream.close()
            self._log_streams[env_index] = None

    def close(self) -> None:
        """Release every owned Expert process and log, including on timeout."""
        errors = []
        for env_index in range(self.num_envs):
            try:
                self.stop_expert(env_index)
            except Exception as error:
                errors.append(error)
        if errors:
            raise RuntimeError("Failed to stop an owned self-assembly Expert") from errors[0]

    def launch_expert(
        self,
        env_index: int,
        target_morphology: str,
    ) -> bool:
        """Launch the repository's real self-assembly expert."""

        env_index = (
            self._validate_env_index(
                env_index
            )
        )

        previous = (
            self._processes[
                env_index
            ]
        )

        if (
            previous is not None
            and previous.poll()
            is None
        ):
            raise RuntimeError(
                "self-assembly expert is already running "
                f"for env {env_index}"
            )

        # Reap an exited previous launch and release its retained log stream.
        self.stop_expert(env_index)

        episode_id = (
            self.episode_id_factory(
                env_index
            )
        )

        runtime_dir = (
            self.runtime_dirs[
                env_index
            ]
        )

        runtime_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # A retained terminal state belongs to the previous episode.
        # Remove it before launching the new Expert so it can never
        # produce a false immediate success/failure.
        (
            runtime_dir
            / "self_assembly_state.json"
        ).unlink(
            missing_ok=True
        )

        command = (
            build_self_assembly_command(
                repository_root=(
                    self.repository_root
                ),
                runtime_dir=(
                    runtime_dir
                ),
                target_morphology=(
                    target_morphology
                ),
                episode_id=(
                    episode_id
                ),
            )
        )

        log_stream = (
            runtime_dir
            / "marl_assembly_expert.log"
        ).open(
            "a",
            encoding="utf-8",
        )

        try:
            process = (
                self.process_factory(
                    command,
                    cwd=str(
                        self.repository_root
                    ),
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        except OSError:
            log_stream.close()
            return False

        self._processes[
            env_index
        ] = process

        self._log_streams[
            env_index
        ] = log_stream

        return True

    def expert_status(
        self,
        env_index: int,
    ) -> bool | None:
        """Read the real structured Expert terminal signal."""

        env_index = (
            self._validate_env_index(
                env_index
            )
        )

        payload = (
            self.expert_state_sources[
                env_index
            ]()
        )

        terminal = (
            parse_self_assembly_terminal(
                payload
            )
        )

        if terminal is not None:
            self.stop_expert(env_index)
            return terminal

        process = (
            self._processes[
                env_index
            ]
        )

        if (
            process is not None
            and process.poll()
            is not None
        ):
            # Expert died without publishing a terminal state.
            self.stop_expert(env_index)
            return False

        return None

    def observe_topology(
        self,
        env_index: int,
    ) -> str | None:
        """Independently match the physical graph against known morphologies."""

        env_index = (
            self._validate_env_index(
                env_index
            )
        )

        current_graph = (
            self.current_graph_sources[
                env_index
            ]()
        )

        if current_graph is None:
            return None

        return (
            detect_morphology_name(
                current_graph=(
                    current_graph
                ),
                morphology_catalog=(
                    self.morphology_catalog
                ),
                topology_matcher=(
                    self.topology_matcher
                ),
            )
        )
