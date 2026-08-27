#!/usr/bin/env python3
"""Run reproducible Snake8 coplanar-gap episodes without the Isaac GUI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from smores_ep.isaac.obstacle_course import (  # noqa: E402
    CoplanarGapSpec,
    sample_coplanar_gap_spec,
)


DEFAULT_BEHAVIOR_PARAMETERS = {
    "approach_linear_m_s": 0.050,
    "linear_m_s": 0.040,
    "gap_profile_substeps": 3,
    "far_bank_transition_links": 1.0,
    "arch_clearance_wheel_radii": 2.0,
    "landing_release_support_modules": 3,
    "landing_release_ramp_links": 1.0,
    "far_bank_traction_preload_wheel_radii": 0.25,
    "gap_goal_tolerance_m": 0.004,
}


def parse_seed_expression(text: str) -> tuple[int, ...]:
    """Parse comma-separated integers and inclusive ``start:end`` ranges."""

    seeds: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            seeds.append(int(token))
            continue
        start_text, end_text = token.split(":", 1)
        start = int(start_text)
        end = int(end_text)
        step = 1 if end >= start else -1
        seeds.extend(range(start, end + step, step))
    if not seeds:
        raise ValueError("At least one gap seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Gap seeds must be unique")
    return tuple(seeds)


def build_episode_specs(
    seeds: Iterable[int],
    *,
    include_reference: bool,
) -> tuple[tuple[str, CoplanarGapSpec], ...]:
    episodes: list[tuple[str, CoplanarGapSpec]] = []
    if include_reference:
        episodes.append(("reference", CoplanarGapSpec()))
    episodes.extend(
        (f"seed-{seed:06d}", sample_coplanar_gap_spec(seed))
        for seed in seeds
    )
    return tuple(episodes)


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, Mapping) else None


def graph_connection_count(payload: Mapping[str, Any]) -> int:
    global_attributes = payload.get("global_attributes", {})
    if isinstance(global_attributes, Mapping):
        reported = global_attributes.get("latched_connection_count")
        if reported is not None:
            return int(reported)
    edges = payload.get("edges", ())
    return len(edges) if isinstance(edges, list) else 0


def evaluate_far_bank_result(
    payload: Mapping[str, Any],
    spec: CoplanarGapSpec,
    *,
    vertical_tolerance_m: float = 0.020,
    maximum_tilt_spread_rad: float = 0.15,
) -> dict[str, Any]:
    """Apply an independent geometric terminal check to the final graph."""

    global_attributes = payload.get("global_attributes", {})
    wheel_radius_m = 0.03106
    if isinstance(global_attributes, Mapping):
        geometry = global_attributes.get("module_geometry", {})
        if isinstance(geometry, Mapping):
            wheel_radius_m = float(
                geometry.get("wheel_radius_m", wheel_radius_m)
            )
    positions: list[tuple[float, float, float]] = []
    tilt_positions: list[float] = []
    nodes = payload.get("nodes", ())
    if isinstance(nodes, list):
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            attributes = node.get("attributes", {})
            if not isinstance(attributes, Mapping):
                continue
            position = attributes.get("position")
            if isinstance(position, list) and len(position) >= 3:
                positions.append(tuple(float(value) for value in position[:3]))
            actuators = attributes.get("actuators", {})
            if isinstance(actuators, Mapping):
                tilt = actuators.get("tilt", {})
                if isinstance(tilt, Mapping) and "position_rad" in tilt:
                    tilt_positions.append(float(tilt["position_rad"]))

    minimum_center_x_m = min(
        (position[0] for position in positions),
        default=float("-inf"),
    )
    maximum_bank_height_error_m = max(
        (abs(position[2] - wheel_radius_m) for position in positions),
        default=float("inf"),
    )
    tilt_spread_rad = (
        max(tilt_positions) - min(tilt_positions)
        if len(tilt_positions) == len(positions) and tilt_positions
        else float("inf")
    )
    connections = graph_connection_count(payload)
    all_modules_on_far_bank = (
        len(positions) == 8
        and connections == 7
        and minimum_center_x_m >= spec.far_edge_x_m
        and maximum_bank_height_error_m <= vertical_tolerance_m
        and tilt_spread_rad <= maximum_tilt_spread_rad
    )
    return {
        "module_count": len(positions),
        "connection_count": connections,
        "wheel_radius_m": wheel_radius_m,
        "required_minimum_center_x_m": spec.far_edge_x_m,
        "minimum_center_x_m": minimum_center_x_m,
        "maximum_bank_height_error_m": maximum_bank_height_error_m,
        "vertical_tolerance_m": vertical_tolerance_m,
        "tilt_spread_rad": tilt_spread_rad,
        "maximum_tilt_spread_rad": maximum_tilt_spread_rad,
        "all_modules_on_far_bank": all_modules_on_far_bank,
    }


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=10.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5.0)


def _wait_for_assembly(
    graph_path: Path,
    processes: Sequence[subprocess.Popen[Any]],
    timeout_s: float,
) -> Mapping[str, Any]:
    deadline = time.monotonic() + timeout_s
    stable_samples = 0
    while time.monotonic() < deadline:
        exited = [
            returncode
            for process in processes
            if (returncode := process.poll()) is not None
        ]
        if exited:
            raise RuntimeError(
                f"Runtime exited before assembly completed: {exited}"
            )
        payload = _read_json(graph_path)
        if payload is not None and graph_connection_count(payload) == 7:
            stable_samples += 1
            if stable_samples >= 3:
                return payload
        else:
            stable_samples = 0
        time.sleep(0.5)
    raise TimeoutError("Wall-clock safety guard reached during self-assembly")


def _run_behavior_client(
    episode_id: str,
    *,
    timeout_s: float,
    log_path: Path,
) -> subprocess.CompletedProcess[str]:
    command = [
        "ros2",
        "run",
        "mssr_expert",
        "mssr_smores_morphology_command_client",
        "--morphology",
        "snake8",
        "--command-id",
        f"{episode_id}-gap-crossing",
        "--behavior",
        "gap_crossing",
        "--parameters-json",
        json.dumps(DEFAULT_BEHAVIOR_PARAMETERS, separators=(",", ":")),
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    return completed


def run_episode(
    episode_id: str,
    spec: CoplanarGapSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    runtime_dir = args.output_dir / episode_id
    runtime_dir.mkdir(parents=True, exist_ok=False)
    curriculum_level = str(getattr(args, "curriculum_level", "robust"))
    curriculum_difficulty = float(
        getattr(args, "curriculum_difficulty", 0.0)
    )
    manifest = {
        "schema_version": "mssr.gap_headless_episode.v1",
        "episode_id": episode_id,
        "gap": spec.to_dict(),
        "behavior": "gap_crossing",
        "behavior_parameters": DEFAULT_BEHAVIOR_PARAMETERS,
        "curriculum_level": curriculum_level,
        "curriculum_difficulty": curriculum_difficulty,
        "validated_baseline_commit": "eb7836e",
    }
    (runtime_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.plan_only:
        return {**manifest, "state": "PLANNED", "success": None}

    ros_domain_id = os.environ.get("ROS_DOMAIN_ID", "0")
    rmw_implementation = os.environ.get(
        "RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp"
    )
    launch_command = [
        "ros2",
        "launch",
        "mssr_expert",
        "smores_runtime.launch.py",
        f"runtime_dir:={runtime_dir}",
        "module_count:=8",
        "gap_test_course:=true",
        "headless:=true",
        "performance:=true",
        "simple_visuals:=true",
        f"simulation_steps:={args.simulation_steps}",
        f"simulation_speed_factor:={args.simulation_speed_factor}",
        f"gap_width_m:={spec.width_m}",
        f"gap_near_edge_x_m:={spec.near_edge_x_m}",
        "behavior_dataset_path:="
        + str(runtime_dir / "behavior_dataset.jsonl"),
        f"behavior_dataset_episode_id:={episode_id}",
        "behavior_dataset_stage_name:="
        + f"snake8_gap_{curriculum_level}",
        f"behavior_dataset_difficulty:={curriculum_difficulty}",
        "behavior_dataset_log_period:=1",
        f"ros_domain_id:={ros_domain_id}",
        f"rmw_implementation:={rmw_implementation}",
    ]
    if spec.seed is not None:
        launch_command.append(f"gap_seed:={spec.seed}")
    assembly_command = [
        "ros2",
        "run",
        "mssr_expert",
        "mssr_smores_self_assembly_node",
        "--ros-args",
        "-p",
        "target_graph_path:="
        + str(
            REPOSITORY_ROOT
            / "mssr_ws/src/mssr_expert/config/smores_snake8.json"
        ),
        "-p",
        f"execution_id:={episode_id}-assembly",
        "-p",
        f"episode_id:={episode_id}",
        "-p",
        "dataset_path:=" + str(runtime_dir / "assembly_dataset.jsonl"),
    ]
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    ros_log_dir = runtime_dir / "ros_logs"
    ros_log_dir.mkdir(parents=True, exist_ok=True)
    environment["ROS_LOG_DIR"] = str(ros_log_dir)
    processes: list[subprocess.Popen[Any]] = []
    start_s = time.monotonic()
    try:
        launch_log = (runtime_dir / "runtime.log").open("w", encoding="utf-8")
        assembly_log = (runtime_dir / "assembly.log").open(
            "w", encoding="utf-8"
        )
        launch = subprocess.Popen(
            launch_command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=launch_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append(launch)
        assembly = subprocess.Popen(
            assembly_command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=assembly_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append(assembly)
        _wait_for_assembly(
            runtime_dir / "robot_graph.json",
            processes,
            args.assembly_wall_timeout_s,
        )
        time.sleep(1.0)
        behavior = _run_behavior_client(
            episode_id,
            timeout_s=args.behavior_wall_timeout_s,
            log_path=runtime_dir / "behavior.log",
        )
        final_graph = _read_json(runtime_dir / "robot_graph.json") or {}
        geometry = evaluate_far_bank_result(final_graph, spec)
        success = behavior.returncode == 0 and bool(
            geometry["all_modules_on_far_bank"]
        )
        result = {
            **manifest,
            "state": "SUCCEEDED" if success else "FAILED",
            "success": success,
            "behavior_returncode": behavior.returncode,
            "geometry": geometry,
            "wall_duration_s": time.monotonic() - start_s,
        }
    except (RuntimeError, TimeoutError, subprocess.TimeoutExpired) as error:
        result = {
            **manifest,
            "state": "FAILED",
            "success": False,
            "error": str(error),
            "wall_duration_s": time.monotonic() - start_s,
        }
    finally:
        for process in reversed(processes):
            _terminate_process(process)
        for stream_name in ("launch_log", "assembly_log"):
            stream = locals().get(stream_name)
            if stream is not None:
                stream.close()
    (runtime_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        default="",
        help="Seed list/ranges; may be empty when --include-reference is used",
    )
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "logs/gap_headless_batch",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--simulation-steps", type=int, default=240_000)
    parser.add_argument("--simulation-speed-factor", type=float, default=1.0)
    parser.add_argument("--assembly-wall-timeout-s", type=float, default=600.0)
    parser.add_argument("--behavior-wall-timeout-s", type=float, default=600.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if shutil.which("ros2") is None and not args.plan_only:
        raise SystemExit("ros2 is unavailable; source ROS and mssr_ws first")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(
            f"Output directory is not empty: {args.output_dir}; use a new path"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.seeds.strip():
        try:
            seeds = parse_seed_expression(args.seeds)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    else:
        seeds = ()
    if not seeds and not args.include_reference:
        raise SystemExit("Provide --seeds and/or --include-reference")
    episodes = build_episode_specs(
        seeds,
        include_reference=args.include_reference,
    )
    results: list[dict[str, Any]] = []
    for episode_id, spec in episodes:
        print(
            f"[{episode_id}] width={spec.width_m:.3f}m "
            f"near={spec.near_edge_x_m:.3f}m "
            f"far={spec.far_edge_x_m:.3f}m",
            flush=True,
        )
        result = run_episode(episode_id, spec, args)
        results.append(result)
        print(f"[{episode_id}] {result['state']}", flush=True)
        if result.get("success") is False and not args.continue_on_failure:
            break
    summary = {
        "schema_version": "mssr.gap_headless_batch.v1",
        "episode_count": len(results),
        "success_count": sum(item.get("success") is True for item in results),
        "failure_count": sum(item.get("success") is False for item in results),
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0 if summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
