from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

from drl.MARL.direct_marl_env import (
    DirectMARLEnv,
)
from drl.MARL.runtime_handoff import (
    DirectMARLHandoff,
    RuntimePhase,
)
from drl.MARL.smores_expert_runtime_binding import (
    SMORESExpertRuntimeBinding,
    make_expert_state_file_source,
    make_robot_graph_file_source,
)

from mssr_expert.behaviors.morphology_library import (
    AssignedModule,
    MorphologyLibrary,
)
from mssr_expert.behaviors.morphology_locomotion import (
    coherent_planar_train_commands,
)
from mssr_expert.graph.serialization import (
    load_attributed_graph,
)
from mssr_expert.planning.smores_ep.attributed_adapter import (
    target_roles_from_graph,
)
from mssr_expert.planning.smores_ep.self_reconfiguration_planner import (
    SmoresSelfReconfigurationPlanner,
)


ACTIVE_MORPHOLOGIES = (
    "rc_car8",
    "snake8",
    "mobile_manipulator8",
)


def _write_atomic(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            separators=(
                ",",
                ":",
            ),
        )
        + "\n",
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def _wait_for_file(
    path: Path,
    *,
    timeout_s: float,
) -> None:
    deadline = (
        time.monotonic()
        + timeout_s
    )

    while time.monotonic() < deadline:
        if (
            path.is_file()
            and path.stat().st_size > 0
        ):
            return

        time.sleep(
            0.1
        )

    raise TimeoutError(
        f"Runtime file not produced: {path}"
    )


def _wait_for_log(
    path: Path,
    token: str,
    *,
    timeout_s: float,
) -> str:
    deadline = (
        time.monotonic()
        + timeout_s
    )

    while time.monotonic() < deadline:
        try:
            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError:
            text = ""

        if token in text:
            lines = [
                line
                for line in text.splitlines()
                if token in line
            ]

            return (
                lines[-1]
                if lines
                else token
            )

        time.sleep(
            0.1
        )

    raise TimeoutError(
        f"Runtime log did not contain {token!r}"
    )


def _terminate_group(
    process: subprocess.Popen,
) -> None:
    if process.poll() is not None:
        return

    try:
        os.killpg(
            process.pid,
            signal.SIGINT,
        )

        process.wait(
            timeout=10.0
        )

        return

    except (
        ProcessLookupError,
        subprocess.TimeoutExpired,
    ):
        pass

    if process.poll() is None:
        try:
            os.killpg(
                process.pid,
                signal.SIGTERM,
            )

            process.wait(
                timeout=5.0
            )

            return

        except (
            ProcessLookupError,
            subprocess.TimeoutExpired,
        ):
            pass

    if process.poll() is None:
        os.killpg(
            process.pid,
            signal.SIGKILL,
        )

        process.wait(
            timeout=5.0
        )


class FileBackedRcCarProbe:
    """One safe external MARL actuation probe.

    This is only the transport used by the first real smoke.

    DirectMARLEnv still controls whether step() may be called.
    Once allowed, the command follows the existing Isaac path:

        actions.json
        -> ActionFileChannel
        -> primitive compose
        -> IsaacMultiModuleCommandRouter.apply()
    """

    def __init__(
        self,
        *,
        runtime_dir: Path,
        graph_source,
        target_graph,
        topology_matcher,
        library: MorphologyLibrary,
    ) -> None:
        self._runtime_dir = (
            runtime_dir
        )

        self._graph_source = (
            graph_source
        )

        self._target_graph = (
            target_graph
        )

        self._topology_matcher = (
            topology_matcher
        )

        self._library = (
            library
        )

        self.calls = 0

    def _assignments(
        self,
    ) -> tuple[
        AssignedModule,
        ...,
    ]:
        current_graph = (
            self._graph_source()
        )

        if current_graph is None:
            raise RuntimeError(
                "Physical graph unavailable "
                "during MARL actuation"
            )

        assignment = (
            self
            ._topology_matcher
            .configuration_assignment(
                current_graph,
                self._target_graph,
            )
        )

        if assignment is None:
            raise RuntimeError(
                "Physical graph stopped matching "
                "rc_car8 before actuation"
            )

        target_roles = (
            target_roles_from_graph(
                self._target_graph
            )
        )

        result = []

        for target_vertex in sorted(
            assignment.target_to_module
        ):
            role_attributes = (
                target_roles[
                    target_vertex
                ]
            )

            target_role = str(
                role_attributes.get(
                    "target_role",
                    "",
                )
            )

            if not target_role:
                raise RuntimeError(
                    "Target role missing for "
                    f"{target_vertex}"
                )

            result.append(
                AssignedModule(
                    module_id=(
                        assignment
                        .target_to_module[
                            target_vertex
                        ]
                    ),
                    target_vertex_id=(
                        target_vertex
                    ),
                    target_role=(
                        target_role
                    ),
                )
            )

        return tuple(
            result
        )

    def step(
        self,
        env_index: int,
    ):
        if env_index != 0:
            raise IndexError(
                env_index
            )

        current_graph = (
            self._graph_source()
        )

        if current_graph is None:
            raise RuntimeError(
                "Physical graph unavailable"
            )

        assignments = (
            self._assignments()
        )

        # Very small straight-ahead RC-Car command.
        # MorphologyLibrary resolves the real physical module IDs
        # and the validated per-role actuator signs/scales.
        locomotion = (
            self._library
            .drive_commands(
                "rc_car8",
                assignments,
                0.015,
                0.0,
            )
        )

        locomotion = (
            coherent_planar_train_commands(
                current_graph,
                locomotion,
            )
        )

        if not locomotion:
            raise RuntimeError(
                "RC-Car probe produced no locomotion commands"
            )

        for (
            module_id,
            command,
        ) in locomotion.items():
            vx = float(
                command.get(
                    "vx",
                    0.0,
                )
            )

            if not math.isfinite(
                vx
            ):
                raise RuntimeError(
                    f"Non-finite vx for {module_id}"
                )

        payload = {
            "schema_version":
                "mssr.actions.v2",
            "stamp":
                time.time(),
            "locomotion":
                locomotion,
        }

        _write_atomic(
            self._runtime_dir
            / "actions.json",
            payload,
        )

        self.calls += 1

        return {
            "command_count":
                len(
                    locomotion
                ),
            "module_ids":
                tuple(
                    sorted(
                        locomotion
                    )
                ),
            "locomotion":
                locomotion,
        }

    def stop(
        self,
    ) -> None:
        _write_atomic(
            self._runtime_dir
            / "actions.json",
            {
                "schema_version":
                    "mssr.actions.v2",
                "stamp":
                    time.time(),
                "locomotion":
                    {},
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
    )

    parser.add_argument(
        "--runtime-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--startup-timeout-s",
        type=float,
        default=180.0,
    )

    parser.add_argument(
        "--assembly-timeout-s",
        type=float,
        default=600.0,
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run Isaac with the viewport enabled.",
    )

    args = parser.parse_args()

    repository_root = (
        args.repository_root
        .expanduser()
        .resolve()
    )

    runtime_dir = (
        args.runtime_dir
        .expanduser()
        .resolve()
    )

    runtime_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config_dir = (
        repository_root
        / "mssr_ws/src/"
        "mssr_expert/config"
    )

    catalog = {
        name:
            load_attributed_graph(
                config_dir
                / f"smores_{name}.json"
            )
        for name
        in ACTIVE_MORPHOLOGIES
    }

    matcher = (
        SmoresSelfReconfigurationPlanner()
    )

    graph_source = (
        make_robot_graph_file_source(
            runtime_dir
        )
    )

    expert_state_source = (
        make_expert_state_file_source(
            runtime_dir
        )
    )

    behavior_library = (
        MorphologyLibrary.load(
            config_dir
            / "smores_morphology_behaviors.json"
        )
    )

    runtime_log_path = (
        runtime_dir
        / "runtime.log"
    )

    runtime_log = (
        runtime_log_path.open(
            "w",
            encoding="utf-8",
        )
    )

    runtime_command = [
        "ros2",
        "launch",
        "mssr_expert",
        "smores_runtime.launch.py",
        f"repository_root:={repository_root}",
        f"runtime_dir:={runtime_dir}",
        "module_count:=8",
        "clean_runtime:=true",
        f"headless:={'false' if args.gui else 'true'}",
        "performance:=true",
        "simple_visuals:=true",
        "physics_hz:=240",
        "state_publish_hz:=30",
        "simulation_speed_factor:=1.0",
        "actuator_effort_scale:=4.0",
        "tilt_effort_scale:=8.0",
        "wheel_friction_scale:=1.50",
        "simulation_steps:=300000",
        (
            "ros_domain_id:="
            + os.environ.get(
                "ROS_DOMAIN_ID",
                "0",
            )
        ),
        (
            "rmw_implementation:="
            + os.environ.get(
                "RMW_IMPLEMENTATION",
                "rmw_cyclonedds_cpp",
            )
        ),
    ]

    print(
        "RUNTIME_LAUNCH:",
        " ".join(
            runtime_command
        ),
        flush=True,
    )

    runtime = subprocess.Popen(
        runtime_command,
        cwd=repository_root,
        env=dict(
            os.environ
        ),
        stdout=runtime_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    probe = None
    binding = None

    try:
        # This proves Isaac + file bridge are alive before
        # DirectMARLEnv is allowed to launch the Expert.
        _wait_for_file(
            runtime_dir
            / "robot_graph.json",
            timeout_s=(
                args.startup_timeout_s
            ),
        )

        if runtime.poll() is not None:
            raise RuntimeError(
                "SMORES runtime exited during startup"
            )

        print(
            "ISAAC_RUNTIME_READY",
            flush=True,
        )

        binding = (
            SMORESExpertRuntimeBinding(
                repository_root=(
                    repository_root
                ),
                runtime_dirs=(
                    runtime_dir,
                ),
                expert_state_sources=(
                    expert_state_source,
                ),
                current_graph_sources=(
                    graph_source,
                ),
                morphology_catalog=(
                    catalog
                ),
                topology_matcher=(
                    matcher
                ),
                episode_id_factory=(
                    lambda env_index:
                        f"marl-smoke-env{env_index}"
                ),
            )
        )

        probe = (
            FileBackedRcCarProbe(
                runtime_dir=(
                    runtime_dir
                ),
                graph_source=(
                    graph_source
                ),
                target_graph=(
                    catalog[
                        "rc_car8"
                    ]
                ),
                topology_matcher=(
                    matcher
                ),
                library=(
                    behavior_library
                ),
            )
        )

        handoff = (
            DirectMARLHandoff()
        )

        env = (
            DirectMARLEnv(
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
                actuation_bridge=(
                    probe
                ),
                target_morphology=(
                    "rc_car8"
                ),
            )
        )

        env.reset()

        print(
            "PHASE:",
            handoff.phase.name,
            flush=True,
        )

        deadline = (
            time.monotonic()
            + args.assembly_timeout_s
        )

        previous_phase = (
            handoff.phase
        )

        while (
            time.monotonic()
            < deadline
        ):
            if runtime.poll() is not None:
                raise RuntimeError(
                    "SMORES runtime exited "
                    "before MARL handoff"
                )

            phase = (
                env.advance_handoff(
                    0
                )
            )

            if (
                phase
                is not previous_phase
            ):
                print(
                    "PHASE:",
                    phase.name,
                    flush=True,
                )

                previous_phase = (
                    phase
                )

            if (
                phase
                is RuntimePhase.FAILED
            ):
                raise RuntimeError(
                    "Expert assembly failed"
                )

            if handoff.marl_active:
                break

            time.sleep(
                0.1
            )

        if not handoff.marl_active:
            raise TimeoutError(
                "MARL authority was not reached"
            )

        print(
            "MARL_AUTHORITY_OK",
            flush=True,
        )

        # This is the critical authority-gated call.
        result = (
            env.step_marl_actuation(
                0
            )
        )

        if (
            result is None
            or probe.calls != 1
        ):
            raise RuntimeError(
                "MARL actuation bridge "
                "was not executed exactly once"
            )

        print(
            "MARL_ACTION_WRITTEN:",
            result[
                "module_ids"
            ],
            flush=True,
        )

        # Isaac itself must confirm that ActionFileChannel
        # reached the real native command router.
        active_line = (
            _wait_for_log(
                runtime_log_path,
                "[behavior] ACTIVE",
                timeout_s=10.0,
            )
        )

        print(
            "ISAAC_ROUTER_ACTIVE:",
            active_line,
            flush=True,
        )

        probe.stop()

        stopped_line = (
            _wait_for_log(
                runtime_log_path,
                "[behavior] STOPPED",
                timeout_s=10.0,
            )
        )

        print(
            "ISAAC_ROUTER_STOPPED:",
            stopped_line,
            flush=True,
        )

        if runtime.poll() is not None:
            raise RuntimeError(
                "Runtime exited during "
                "MARL actuation probe"
            )

        print()
        print(
            "REAL_AUTHORITY_ACTUATION_SMOKE_SUCCEEDED",
            flush=True,
        )

        return 0

    finally:
        if probe is not None:
            try:
                probe.stop()
            except Exception:
                pass

        try:
            if binding is not None:
                binding.close()
        finally:
            try:
                _terminate_group(runtime)
            finally:
                runtime_log.close()


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
