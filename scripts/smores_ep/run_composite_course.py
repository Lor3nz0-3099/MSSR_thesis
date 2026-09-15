#!/usr/bin/env python3
"""Validate, materialize, and launch one composite obstacle-course episode."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
SMORES_EP_SRC = ROOT / "scripts" / "smores_ep" / "src"
EXPERT_SRC = ROOT / "mssr_ws" / "src" / "mssr_expert"
sys.path.insert(0, str(SMORES_EP_SRC))
sys.path.insert(0, str(EXPERT_SRC))

from smores_ep.isaac.obstacle_course import composite_obstacle_course
from mssr_expert.planning.smores_ep.composite_mission import (
    CompositeMissionPlanner,
)
from mssr_expert.planning.smores_ep.obstacle_course_policy import (
    ObstacleCoursePolicy,
    ValidatedSeedCatalog,
)
from mssr_expert.graph.serialization import (
    attributed_graph_from_dict,
    load_attributed_graph,
)
from mssr_expert.planning.smores_ep.self_reconfiguration_planner import (
    SmoresSelfReconfigurationPlanner,
)


DEFAULT_CAMPAIGN = (
    EXPERT_SRC / "config" / "smores_composite_campaign16.json"
)
DEFAULT_SEED_CATALOG = (
    EXPERT_SRC / "config" / "smores_composite_seed_catalog.json"
)
TARGET_CONFIGS = {
    "rc_car8": EXPERT_SRC / "config" / "smores_rc_car8.json",
    "snake8": EXPERT_SRC / "config" / "smores_snake8.json",
    "mobile_manipulator8": (
        EXPERT_SRC / "config" / "smores_mobile_manipulator8.json"
    ),
}


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch one validated multi-obstacle SMORES-EP course. The "
            "episode order is taken from the campaign; no new seeds are sampled."
        )
    )
    parser.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    parser.add_argument("--seed-catalog", type=Path, default=DEFAULT_SEED_CATALOG)
    parser.add_argument("--episode", default="composite-0001")
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--stop-after-stage",
        type=int,
        default=None,
        help=(
            "Diagnostic mode: stop stage execution immediately after "
            "the selected stage ID. In GUI mode Isaac remains open "
            "for inspection."
        ),
    )

    execution_mode = parser.add_mutually_exclusive_group()
    execution_mode.add_argument(
        "--execute",
        dest="execute",
        action="store_true",
        help=(
            "Execute the planned assembly/reconfiguration/expert stages "
            "(default; retained for backwards compatibility)."
        ),
    )
    execution_mode.add_argument(
        "--preview-only",
        dest="execute",
        action="store_false",
        help="Launch the composite world without executing the planned stages.",
    )
    parser.set_defaults(execute=True)

    parser.add_argument("--ros-domain-id", default="0")
    parser.add_argument(
        "--reconfiguration-executor",
        choices=("legacy", "v2"),
        default="legacy",
        help="Select the reconfiguration node explicitly; legacy remains default.",
    )
    return parser


def transition_executable(is_assembly: bool, reconfiguration_executor: str) -> str:
    if is_assembly:
        return "mssr_smores_self_assembly_node"
    if reconfiguration_executor == "v2":
        return "mssr_smores_deterministic_reconfiguration_node"
    return "mssr_smores_self_reconfiguration_node"


def _object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root in {path} must be an object")
    return payload


def select_episode(campaign: Mapping[str, Any], episode_id: str) -> dict[str, Any]:
    if campaign.get("schema_version") != "mssr.composite_campaign.v1":
        raise ValueError("Unsupported composite campaign schema")
    episodes = campaign.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError("Composite campaign has no episodes array")
    matches = [
        episode
        for episode in episodes
        if isinstance(episode, Mapping)
        and str(episode.get("episode_id", "")) == episode_id
    ]
    if len(matches) != 1:
        available = sorted(
            str(item.get("episode_id"))
            for item in episodes
            if isinstance(item, Mapping)
        )
        raise ValueError(
            f"Episode {episode_id!r} was not found exactly once; "
            f"available={available}"
        )
    tasks = matches[0].get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"Episode {episode_id!r} has no tasks")
    return {
        "schema_version": "mssr.composite_mission.v1",
        "episode_id": episode_id,
        "tasks": [dict(task) for task in tasks],
    }


def morphology_capabilities() -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for expected_name, path in TARGET_CONFIGS.items():
        payload = _object(path)
        attributes = payload.get("global_attributes")
        if not isinstance(attributes, Mapping):
            raise ValueError(f"Target graph {path} has no global_attributes")
        name = str(attributes.get("morphology_name", ""))
        if name != expected_name:
            raise ValueError(f"Target graph {path} names morphology {name!r}")
        raw = attributes.get("capabilities")
        if not isinstance(raw, list):
            raise ValueError(f"Target graph {path} has no capability array")
        result[name] = tuple(str(item) for item in raw)
    return result


def stop_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=8)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass


def wait_for_file(path: Path, process: subprocess.Popen[Any], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Runtime exited before creating {path.name}")
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                return
        time.sleep(0.25)
    raise TimeoutError(f"Timed out waiting for {path}")


def wait_for_topology(
    graph_path: Path,
    target_name: str,
    runtime: subprocess.Popen[Any],
    timeout_s: float = 540.0,
    *,
    transition_log_path: Path | None = None,
    transition_process: subprocess.Popen[Any] | None = None,
    is_assembly: bool = False,
) -> None:
    target = load_attributed_graph(TARGET_CONFIGS[target_name])
    matcher = SmoresSelfReconfigurationPlanner()
    deadline = time.monotonic() + timeout_s
    stable = 0
    while time.monotonic() < deadline:
        if runtime.poll() is not None:
            raise RuntimeError("Runtime exited during topology transition")

        # A transition may fail before the target topology is ever reached.
        # Do not wait for the topology timeout if the expert has already
        # reported a terminal failure.
        if transition_log_path is not None:
            try:
                transition_content = transition_log_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except OSError:
                transition_content = ""

            failure_marker = (
                "Parallel self-assembly failed:"
                if is_assembly
                else "Self-reconfiguration failed:"
            )

            if failure_marker in transition_content:
                failure_lines = [
                    line.strip()
                    for line in transition_content.splitlines()
                    if failure_marker in line
                ]
                detail = (
                    failure_lines[-1]
                    if failure_lines
                    else failure_marker
                )
                raise RuntimeError(
                    "Transition expert reported failure before target "
                    f"topology {target_name!r}: {detail}"
                )

        if (
            transition_process is not None
            and transition_process.poll() is not None
        ):
            raise RuntimeError(
                "Transition expert exited before target topology "
                f"{target_name!r} was reached"
            )
        try:
            payload = json.loads(graph_path.read_text(encoding="utf-8"))
            graph = attributed_graph_from_dict(payload)
            matched = matcher.configuration_assignment(graph, target) is not None
        except (OSError, ValueError, json.JSONDecodeError):
            matched = False
        stable = stable + 1 if matched else 0
        if stable >= 5:
            return
        time.sleep(0.5)
    raise TimeoutError(f"Physical topology did not become {target_name!r}")


def run_logged(
    command: list[str],
    *,
    environment: Mapping[str, str],
    log_path: Path,
    timeout_s: float | None = None,
) -> None:
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=dict(environment),
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {result.returncode}; see {log_path}"
        )


def wait_for_transition_terminal(
    log_path: Path,
    process: subprocess.Popen[Any],
    *,
    is_assembly: bool,
    timeout_s: float = 180.0,
) -> None:
    """Wait until the transition expert reports its real terminal state.

    Topology can become correct before final PAN/TILT posture primitives
    complete.  The composite must not kill the owning expert at that point,
    otherwise those backend primitives remain active and leak resources into
    the following behavior stage.
    """

    success_marker = (
        "Parallel self-assembly completed."
        if is_assembly
        else "Self-reconfiguration completed."
    )
    failure_marker = (
        "Parallel self-assembly failed:"
        if is_assembly
        else "Self-reconfiguration failed:"
    )

    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        try:
            content = log_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            content = ""

        if success_marker in content:
            return

        if failure_marker in content:
            tail = "\n".join(content.splitlines()[-30:])
            raise RuntimeError(
                "Transition expert reported failure:\n" + tail
            )

        time.sleep(0.2)

    raise TimeoutError(
        "Timed out waiting for transition expert terminal state: "
        f"{success_marker}"
    )


def start_transition(
    stage: Any,
    *,
    runtime: subprocess.Popen[Any],
    runtime_dir: Path,
    dataset_path: Path,
    episode_id: str,
    environment: Mapping[str, str],
    reconfiguration_executor: str = "legacy",
) -> None:
    is_assembly = stage.kind == "assembly"
    executable = transition_executable(is_assembly, reconfiguration_executor)
    command = ["ros2", "run", "mssr_expert", executable, "--ros-args"]
    if is_assembly:
        command.extend(
            ("-p", f"target_graph_path:={TARGET_CONFIGS[stage.target_morphology]}")
        )
    else:
        command.extend(
            (
                "-p", "source_graph_path:=auto",
                "-p", f"target_morphology:={stage.target_morphology}",
            )
        )
    command.extend(
        (
            "-p", f"execution_id:={episode_id}-stage-{stage.stage_id:02d}",
            "-p", f"episode_id:={episode_id}",
            "-p", f"dataset_path:={dataset_path}",
        )
    )
    log_path = runtime_dir / f"stage-{stage.stage_id:02d}-{stage.kind}.log"
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=dict(environment),
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_for_topology(
                runtime_dir / "robot_graph.json",
                stage.target_morphology,
                runtime,
                transition_log_path=log_path,
                transition_process=process,
                is_assembly=is_assembly,
            )
            # Topology becomes valid at the final docking event, but final
            # posture primitives (PAN/TILT normalization/folding) may still
            # own backend resources.  Do not advance to the next composite
            # stage until the expert itself reports terminal success.
            wait_for_transition_terminal(
                log_path,
                process,
                is_assembly=is_assembly,
            )
        finally:
            stop_process(process)


def wait_nav2(environment: Mapping[str, str], timeout_s: float = 75.0) -> None:
    deadline = time.monotonic() + timeout_s
    required = ("/bt_navigator", "/planner_server", "/controller_server")
    last_status = "no lifecycle response received"

    while time.monotonic() < deadline:
        active = True

        for node in required:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                active = False
                break

            try:
                result = subprocess.run(
                    ["ros2", "lifecycle", "get", node],
                    cwd=ROOT,
                    env=dict(environment),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=min(5.0, remaining_s),
                    check=False,
                )
            except subprocess.TimeoutExpired:
                last_status = f"{node}: lifecycle probe timed out"
                active = False
                break

            output = result.stdout.strip()
            if result.returncode != 0 or "active" not in output.lower():
                last_status = (
                    f"{node}: returncode={result.returncode}, "
                    f"output={output!r}"
                )
                active = False
                break

        if active:
            return

        remaining_s = deadline - time.monotonic()
        if remaining_s > 0.0:
            time.sleep(min(1.0, remaining_s))

    raise TimeoutError(
        f"Nav2 did not become active within {timeout_s:.1f}s; "
        f"last_status={last_status}"
    )


def execute_navigation(
    stage: Any,
    *,
    runtime_dir: Path,
    dataset_path: Path,
    episode_id: str,
    environment: Mapping[str, str],
) -> None:
    nav_log = runtime_dir / f"stage-{stage.stage_id:02d}-nav2.log"
    with nav_log.open("w", encoding="utf-8") as stream:
        nav2 = subprocess.Popen(
            [
                "ros2", "launch", "mssr_expert", "smores_nav2.launch.py",
                "autostart:=true", "log_level:=warn",
            ],
            cwd=ROOT,
            env=dict(environment),
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            wait_nav2(environment)
            result_path = runtime_dir / f"stage-{stage.stage_id:02d}-nav2.json"
            route_command = [
                sys.executable,
                str(ROOT / "scripts" / "smores_ep" / "run_rc_car_nav2_route.py"),
                "--seed", str(stage.parameters.get("seed", 0)),
                "--action-timeout-s", "900",
                "--result-json", str(result_path),
                "--dataset-path", str(dataset_path),
                "--episode-id", episode_id,
                "--task-id", stage.task_id,
            ]
            if stage.kind == "nav2":
                route_path = runtime_dir / f"stage-{stage.stage_id:02d}-route.json"
                route_path.write_text(
                    json.dumps(
                        {
                            "waypoints_xyyaw": stage.parameters[
                                "waypoints_xyyaw"
                            ],
                            "cone_centers_xy_m": stage.parameters.get(
                                "cone_centers_xy_m", []
                            ),
                            "cone_radius_m": stage.parameters.get(
                                "cone_radius_m", 0.05
                            ),
                            "corridor_width_m": stage.parameters.get(
                                "corridor_width_m", 1.10
                            ),
                            "platform_bounds_xy_m": stage.parameters.get(
                                "platform_bounds_xy_m"
                            ),
                            "start_pad_bounds_xy_m": stage.parameters.get(
                                "start_pad_bounds_xy_m"
                            ),
                            "vehicle_footprint": stage.parameters.get(
                                "vehicle_footprint"
                            ),
                            "free_rectangles_xy_m": stage.parameters.get(
                                "navigation_free_rectangles_xy_m",
                                [],
                            ),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                navigation_goal = stage.parameters.get(
                    "navigation_goal_xyyaw",
                    stage.parameters["waypoints_xyyaw"][-1],
                )

                if (
                    not isinstance(navigation_goal, (list, tuple))
                    or len(navigation_goal) != 3
                ):
                    raise ValueError(
                        "navigation_goal_xyyaw must contain x, y, yaw"
                    )

                gx, gy, gyaw = (
                    float(value) for value in navigation_goal
                )

                print(
                    "  Nav2 explicit goal: "
                    f"({gx:.3f}, {gy:.3f}, yaw={gyaw:.3f})",
                    flush=True,
                )

                route_command.extend(
                    (
                        "--route-json", str(route_path),
                        "--goal-x", str(gx),
                        "--goal-y", str(gy),
                        "--goal-yaw", str(gyaw),
                    )
                )

                accept_position = stage.parameters.get(
                    "navigation_accept_position_m"
                )
                accept_yaw = stage.parameters.get(
                    "navigation_accept_yaw_rad"
                )

                if (
                    accept_position is not None
                    and accept_yaw is not None
                ):
                    route_command.extend(
                        (
                            "--accept-position-m",
                            str(float(accept_position)),
                            "--accept-yaw-rad",
                            str(float(accept_yaw)),
                        )
                    )

                    print(
                        "  Nav2 coarse acceptance: "
                        f"position<={float(accept_position):.3f}m, "
                        f"yaw<={float(accept_yaw):.3f}rad",
                        flush=True,
                    )
            else:
                raw_goal = stage.parameters.get("goal_xyyaw")

                if (
                    isinstance(raw_goal, (list, tuple))
                    and len(raw_goal) == 3
                ):
                    gx, gy, gyaw = (
                        float(value) for value in raw_goal
                    )
                else:
                    gx, gy, _ = stage.parameters[
                        "center_xyz_m"
                    ]
                    gyaw = 0.0

                navigation_map = stage.parameters.get(
                    "navigation_map"
                )

                if isinstance(navigation_map, Mapping):
                    route_path = (
                        runtime_dir
                        / f"stage-{stage.stage_id:02d}-route.json"
                    )
                    route_path.write_text(
                        json.dumps(
                            dict(navigation_map),
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )

                    # run_rc_car_nav2_route supports using a
                    # composite map together with an explicit
                    # NavigateToPose goal.
                    route_command.extend(
                        ("--route-json", str(route_path))
                    )

                route_command.extend(
                    (
                        "--goal-x", str(gx),
                        "--goal-y", str(gy),
                        "--goal-yaw", str(gyaw),
                    )
                )
            run_logged(
                route_command,
                environment=environment,
                log_path=runtime_dir / f"stage-{stage.stage_id:02d}-route.log",
                timeout_s=930,
            )
        finally:
            stop_process(nav2)


def normalize_dataset(
    path: Path,
    episode_id: str,
    success: bool,
) -> None:
    """Normalize an episode JSONL with bounded memory usage."""
    if not path.is_file():
        return

    temp_path = path.with_name(path.name + ".normalizing")

    def valid_records():
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if isinstance(record, dict):
                    yield record

    records = iter(valid_records())
    record = next(records, None)
    index = 0

    try:
        with temp_path.open("w", encoding="utf-8") as destination:
            while record is not None:
                following = next(records, None)
                terminal = following is None

                record["skill_done"] = bool(
                    record.get("done", False)
                )
                record["skill_success"] = bool(
                    record.get("success", False)
                )
                record["episode_id"] = episode_id
                record["timestep"] = index
                record["is_first"] = index == 0
                record["is_last"] = terminal
                record["is_terminal"] = terminal
                record["episode_done"] = terminal
                record["episode_success"] = (
                    bool(success) if terminal else False
                )
                record["done"] = terminal
                record["success"] = (
                    bool(success) if terminal else False
                )
                record["reward"] = (
                    1.0 if terminal and success else 0.0
                )

                if following is not None:
                    following_graph = following.get("graph_t")

                    if isinstance(following_graph, Mapping):
                        record["graph_t_plus_1"] = following_graph
                        record["next_graph"] = following_graph

                    following_observation = following.get(
                        "observation_t",
                        following.get("observation"),
                    )

                    if isinstance(
                        following_observation,
                        Mapping,
                    ):
                        record["observation_t_plus_1"] = (
                            following_observation
                        )
                        record["next_observation"] = (
                            following_observation
                        )

                destination.write(
                    json.dumps(
                        record,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

                record = following
                index += 1

        temp_path.replace(path)

    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise

def execute_stages(
    stages: tuple[Any, ...],
    *,
    runtime: subprocess.Popen[Any],
    runtime_dir: Path,
    dataset_path: Path,
    episode_id: str,
    environment: Mapping[str, str],
    headless: bool,
    stop_after_stage: int | None = None,
    reconfiguration_executor: str = "legacy",
) -> bool:
    completed_button_tasks: set[str] = set()
    for stage in stages:
        print(
            f"EXECUTE {stage.stage_id:02d}/{len(stages) - 1:02d} "
            f"{stage.task_id} {stage.kind}",
            flush=True,
        )
        if stage.task_id in completed_button_tasks:
            continue
        if stage.kind in {"assembly", "reconfiguration"}:
            start_transition(
                stage,
                runtime=runtime,
                runtime_dir=runtime_dir,
                dataset_path=dataset_path,
                episode_id=episode_id,
                environment=environment,
                reconfiguration_executor=reconfiguration_executor,
            )
        elif stage.kind == "behavior":
            run_logged(
                [
                    "ros2", "run", "mssr_expert",
                    "mssr_smores_morphology_command_client",
                    "--morphology", stage.target_morphology,
                    "--behavior", str(stage.behavior),
                    "--command-id", f"{episode_id}-stage-{stage.stage_id:02d}",
                    "--parameters-json", json.dumps(dict(stage.parameters)),
                    "--timeout-s", "900",
                ],
                environment=environment,
                log_path=runtime_dir / f"stage-{stage.stage_id:02d}-behavior.log",
                timeout_s=930,
            )
        elif stage.kind in {"nav2", "nav2_goal"}:
            execute_navigation(
                stage,
                runtime_dir=runtime_dir,
                dataset_path=dataset_path,
                episode_id=episode_id,
                environment=environment,
            )
        elif stage.kind == "gap_rc_alignment":
            command = [
                sys.executable,
                str(
                    ROOT
                    / "scripts"
                    / "smores_ep"
                    / "run_gap_rc_alignment.py"
                ),
                "--target-yaw-rad",
                str(stage.parameters.get("target_yaw_rad", 0.0)),
            ]

            run_logged(
                command,
                environment=environment,
                log_path=(
                    runtime_dir
                    / f"stage-{stage.stage_id:02d}-gap-alignment.log"
                ),
                timeout_s=120,
            )

        elif stage.kind == "button_rc_alignment":
            command = [
                sys.executable,
                str(ROOT / "scripts" / "smores_ep" / "run_button_expert_to_ik.py"),
                "--seed", str(stage.parameters["seed"]),
                "--external-runtime-dir", str(runtime_dir),
                "--button-task-id", stage.task_id,
                "--dataset-path", str(dataset_path),
                "--episode-id", episode_id,
            ]
            if headless:
                command.append("--headless")
            run_logged(
                command,
                environment=environment,
                log_path=runtime_dir / f"stage-{stage.stage_id:02d}-button.log",
                timeout_s=1800,
            )
            completed_button_tasks.add(stage.task_id)
        elif stage.kind != "button_expert":
            raise RuntimeError(f"Unsupported composite stage kind {stage.kind!r}")

        if (
            stop_after_stage is not None
            and stage.stage_id == stop_after_stage
        ):
            print(
                f"STOP-AFTER-STAGE reached: {stage.stage_id:02d} "
                f"{stage.task_id} {stage.kind}",
                flush=True,
            )
            return False

    return True


def main() -> int:
    args = argument_parser().parse_args()
    mission = select_episode(_object(args.campaign), args.episode)
    catalog = ValidatedSeedCatalog.load(args.seed_catalog.resolve())
    course = composite_obstacle_course(mission, catalog.seeds_by_task_type)
    planner = CompositeMissionPlanner(
        ObstacleCoursePolicy(morphology_capabilities())
    )
    stages = planner.build(course.tasks)

    if (
        args.stop_after_stage is not None
        and args.stop_after_stage
        not in {stage.stage_id for stage in stages}
    ):
        raise ValueError(
            f"Unknown --stop-after-stage {args.stop_after_stage}; "
            f"valid IDs are 0..{len(stages) - 1}"
        )

    print(
        f"episode={args.episode} obstacles={len(mission['tasks'])} "
        f"stages={len(stages)} final_height={course.final_floor_height_m:.3f}m"
    )
    for stage in stages:
        transition = (
            f" {stage.source_morphology or 'loose'}->{stage.target_morphology}"
            if stage.kind in {"assembly", "reconfiguration"}
            else f" morphology={stage.target_morphology}"
        )
        print(
            f"  {stage.stage_id:02d} {stage.task_id}: {stage.kind}{transition}"
        )

    if args.plan_only:
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    runtime_dir = (
        args.runtime_dir.resolve()
        if args.runtime_dir is not None
        else ROOT / "logs" / "composite_course" / f"{args.episode}-{stamp}"
    )
    runtime_dir.mkdir(parents=True, exist_ok=False)
    mission_path = runtime_dir / "mission.json"
    mission_path.write_text(
        json.dumps(mission, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (runtime_dir / "stage_plan.json").write_text(
        json.dumps(
            {
                "schema_version": "mssr.composite_stage_plan.v1",
                "episode_id": args.episode,
                "stages": [
                    {
                        "stage_id": stage.stage_id,
                        "task_id": stage.task_id,
                        "task_type": stage.task_type,
                        "kind": stage.kind,
                        "source_morphology": stage.source_morphology,
                        "target_morphology": stage.target_morphology,
                        "behavior": stage.behavior,
                        "parameters": dict(stage.parameters),
                    }
                    for stage in stages
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    command = [
        "ros2",
        "launch",
        "mssr_expert",
        "smores_runtime.launch.py",
        f"runtime_dir:={runtime_dir}",
        "module_count:=8",
        f"composite_mission:={mission_path}",
        f"composite_seed_catalog:={args.seed_catalog.resolve()}",
        f"headless:={'true' if args.headless else 'false'}",
        f"simulation_steps:={300000 if args.headless and args.execute else 0}",
        "performance:=true",
        "simple_visuals:=true",
        "actuator_effort_scale:=4.0",
        "wheel_friction_scale:=1.50",
        "tilt_effort_scale:=8.0",
        f"behavior_dataset_path:={runtime_dir / 'dataset.jsonl'}",
        f"behavior_dataset_episode_id:={args.episode}",
        f"ros_domain_id:={args.ros_domain_id}",
        "rmw_implementation:=rmw_cyclonedds_cpp",
    ]
    environment = dict(os.environ)
    environment["ROS_DOMAIN_ID"] = str(args.ros_domain_id)
    environment["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
    environment["ROS_LOG_DIR"] = str(runtime_dir / "ros_logs")
    print(f"runtime_dir={runtime_dir}")
    runtime_log = (runtime_dir / "runtime.log").open("w", encoding="utf-8")
    runtime = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=runtime_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    success = False
    try:
        wait_for_file(runtime_dir / "state_graph.json", runtime, 240.0)
        if not args.execute:
            print("Composite GUI ready; close Isaac to end the command.")
            return runtime.wait()
        dataset_path = runtime_dir / "dataset.jsonl"
        completed_all_stages = execute_stages(
            stages,
            runtime=runtime,
            runtime_dir=runtime_dir,
            dataset_path=dataset_path,
            episode_id=args.episode,
            environment=environment,
            headless=args.headless,
            stop_after_stage=args.stop_after_stage,
            reconfiguration_executor=args.reconfiguration_executor,
        )

        if not completed_all_stages:
            print(
                "Composite execution intentionally paused after "
                f"stage {args.stop_after_stage:02d}.",
                flush=True,
            )
            if not args.headless:
                print(
                    "Isaac remains open for inspection; "
                    "close the GUI to exit."
                )
                return runtime.wait()
            return 0

        success = True
        normalize_dataset(dataset_path, args.episode, True)
        print(f"COMPOSITE MISSION SUCCEEDED: final goal reached; dataset={dataset_path}")
        if not args.headless:
            print("Isaac remains open for inspection; close the GUI to exit.")
            return runtime.wait()
        return 0
    finally:
        if not success:
            normalize_dataset(runtime_dir / "dataset.jsonl", args.episode, False)
        if args.headless or not success:
            stop_process(runtime)
        runtime_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
