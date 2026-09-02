#!/usr/bin/env python3
"""Run seeded RC-Car8 planar Nav2 expert episodes headlessly."""

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
    RCPlanarSpec,
    sample_rc_car_planar_spec,
)


def parse_seed_expression(text: str) -> tuple[int, ...]:
    seeds: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            seeds.append(int(token))
            continue
        a, b = token.split(":", 1)
        start = int(a)
        end = int(b)
        step = 1 if end >= start else -1
        seeds.extend(range(start, end + step, step))

    if not seeds:
        raise ValueError("At least one RC-Car seed is required")
    if len(seeds) != len(set(seeds)):
        raise ValueError("RC-Car seeds must be unique")
    return tuple(seeds)


def _read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text())
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


def _dataset_terminal_success(path: Path) -> bool:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return False

    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if bool(payload.get("done", False)):
            return bool(payload.get("success", False))
    return False


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
    stable = 0

    while time.monotonic() < deadline:
        exited = [
            process.poll()
            for process in processes
            if process.poll() is not None
        ]
        if exited:
            raise RuntimeError(
                f"Runtime exited during RC-Car assembly: {exited}"
            )

        graph = _read_json(graph_path)
        if graph is not None and graph_connection_count(graph) == 7:
            stable += 1
            if stable >= 4:
                return graph
        else:
            stable = 0

        time.sleep(0.5)

    raise TimeoutError("RC-Car8 assembly wall timeout")


def _wait_for_nav2_active(
    *,
    timeout_s: float,
    environment: Mapping[str, str],
    log_path: Path,
) -> None:
    """Wait until the Nav2 lifecycle stack is genuinely ACTIVE.

    On ROS2 Humble an action server may be discoverable before its lifecycle
    node is active, so wait_for_server() alone is not a sufficient readiness
    test.
    """

    nodes = (
        "/bt_navigator",
        "/planner_server",
        "/controller_server",
    )

    deadline = time.monotonic() + timeout_s
    history: list[str] = []

    while time.monotonic() < deadline:
        all_active = True
        snapshot: list[str] = []

        for node in nodes:
            result = subprocess.run(
                ["ros2", "lifecycle", "get", node],
                cwd=REPOSITORY_ROOT,
                env=dict(environment),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=5.0,
                check=False,
            )

            text = result.stdout.strip()
            snapshot.append(f"{node}: {text}")

            if (
                result.returncode != 0
                or "active" not in text.lower()
            ):
                all_active = False

        history.extend(snapshot)

        if all_active:
            log_path.write_text(
                "\n".join(snapshot) + "\n"
            )
            return

        time.sleep(1.0)

    log_path.write_text(
        "\n".join(history[-120:]) + "\n"
    )

    raise TimeoutError(
        "Nav2 lifecycle did not reach ACTIVE state "
        "for bt_navigator/planner_server/controller_server"
    )


def _wait_for_odom(
    *,
    timeout_s: float,
    environment: Mapping[str, str],
    log_path: Path,
) -> None:
    command = [
        "ros2",
        "topic",
        "echo",
        "/odom",
        "nav_msgs/msg/Odometry",
        "--once",
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=dict(environment),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
    log_path.write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(
            "RC-Car8 virtual /odom was not available after assembly"
        )


def run_episode(
    episode_id: str,
    spec: RCPlanarSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    runtime_dir = args.output_dir / episode_id
    runtime_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "schema_version": "mssr.rc_car_nav2_episode.v1",
        "episode_id": episode_id,
        "morphology": "rc_car8",
        "task": "nav2_planar_route",
        "route": spec.to_dict(),
        "controller": "Nav2 NavigateThroughPoses + DWB",
        "pan_contact_profile": {
            "active_only_for_rc_car8_wheel_roles": True,
            "static_friction": 1.20,
            "dynamic_friction": 1.00,
        },
    }
    (runtime_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    if args.plan_only:
        return {**manifest, "state": "PLANNED", "success": None}

    ros_domain_id = os.environ.get("ROS_DOMAIN_ID", "223")
    rmw = os.environ.get(
        "RMW_IMPLEMENTATION",
        "rmw_cyclonedds_cpp",
    )

    launch_command = [
        "ros2",
        "launch",
        "mssr_expert",
        "smores_runtime.launch.py",
        f"runtime_dir:={runtime_dir}",
        "module_count:=8",
        "rc_car_planar_test_course:=true",
        f"rc_car_seed:={spec.seed}",
        "headless:=true",
        "performance:=true",
        "simple_visuals:=true",
        f"simulation_steps:={args.simulation_steps}",
        "simulation_speed_factor:=1.0",
        "actuator_effort_scale:=4.0",
        "wheel_friction_scale:=1.50",
        "tilt_effort_scale:=8.0",
        "behavior_dataset_path:="
        + str(runtime_dir / "behavior_dataset.jsonl"),
        f"behavior_dataset_episode_id:={episode_id}",
        "behavior_dataset_stage_name:=rc_car8_planar_nav2",
        "behavior_dataset_difficulty:=0.0",
        "behavior_dataset_log_period:=1",
        "behavior_control_rate_hz:=10.0",
        f"ros_domain_id:={ros_domain_id}",
        f"rmw_implementation:={rmw}",
    ]

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
            / "mssr_ws/src/mssr_expert/config/smores_rc_car8.json"
        ),
        "-p",
        f"execution_id:={episode_id}-assembly",
        "-p",
        f"episode_id:={episode_id}",
        "-p",
        "dataset_path:="
        + str(runtime_dir / "assembly_dataset.jsonl"),
    ]

    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    ros_log_dir = runtime_dir / "ros_logs"
    ros_log_dir.mkdir(parents=True, exist_ok=True)
    environment["ROS_LOG_DIR"] = str(ros_log_dir)

    processes: list[subprocess.Popen[Any]] = []
    streams: list[Any] = []
    start_s = time.monotonic()

    try:
        runtime_log = (runtime_dir / "runtime.log").open("w")
        assembly_log = (runtime_dir / "assembly.log").open("w")
        streams.extend((runtime_log, assembly_log))

        runtime = subprocess.Popen(
            launch_command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=runtime_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append(runtime)

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

        # RC-Car8 has a final coordinated four-wheel fold after topology
        # closure. Give that posture time to finish before Nav2 takes control.
        time.sleep(args.post_assembly_settle_s)

        _wait_for_odom(
            timeout_s=30.0,
            environment=environment,
            log_path=runtime_dir / "odom_ready.log",
        )

        nav2_log = (runtime_dir / "nav2.log").open("w")
        streams.append(nav2_log)
        nav2 = subprocess.Popen(
            [
                "ros2",
                "launch",
                "mssr_expert",
                "smores_nav2.launch.py",
                "autostart:=true",
                "log_level:=warn",
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            stdout=nav2_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append(nav2)

        _wait_for_nav2_active(
            timeout_s=45.0,
            environment=environment,
            log_path=runtime_dir / "nav2_lifecycle_ready.log",
        )

        route_command = [
            sys.executable,
            str(SCRIPT_DIR / "run_rc_car_nav2_route.py"),
            "--seed",
            str(spec.seed),
            "--action-timeout-s",
            str(args.route_wall_timeout_s),
            "--result-json",
            str(runtime_dir / "route_result.json"),
        ]

        route = subprocess.run(
            route_command,
            cwd=REPOSITORY_ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.route_wall_timeout_s + 90.0,
            check=False,
        )
        (runtime_dir / "route.log").write_text(route.stdout)

        # Let the terminal route-status sample enter the 10 Hz graph logger.
        time.sleep(1.0)

        final_graph = (
            _read_json(runtime_dir / "robot_graph.json") or {}
        )
        connections = graph_connection_count(final_graph)
        dataset_terminal = _dataset_terminal_success(
            runtime_dir / "behavior_dataset.jsonl"
        )
        route_result = (
            _read_json(runtime_dir / "route_result.json") or {}
        )

        success = (
            route.returncode == 0
            and bool(route_result.get("success", False))
            and connections == 7
            and dataset_terminal
        )

        result = {
            **manifest,
            "state": "SUCCEEDED" if success else "FAILED",
            "success": success,
            "route_returncode": route.returncode,
            "route_result": dict(route_result),
            "final_connection_count": connections,
            "dataset_terminal_success": dataset_terminal,
            "wall_duration_s": time.monotonic() - start_s,
        }

    except (
        RuntimeError,
        TimeoutError,
        subprocess.TimeoutExpired,
    ) as error:
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
        for stream in streams:
            stream.close()

    (runtime_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "logs/rc_car_nav2_batch",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--simulation-steps", type=int, default=240_000)
    parser.add_argument(
        "--assembly-wall-timeout-s",
        type=float,
        default=600.0,
    )
    parser.add_argument(
        "--post-assembly-settle-s",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--route-wall-timeout-s",
        type=float,
        default=360.0,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if shutil.which("ros2") is None and not args.plan_only:
        raise SystemExit(
            "ros2 unavailable; source ROS and mssr_ws first"
        )

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(
            f"Output directory is not empty: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_seed_expression(args.seeds)
    results: list[dict[str, Any]] = []

    for seed in seeds:
        episode_id = f"seed-{seed:06d}"
        spec = sample_rc_car_planar_spec(seed)

        print(
            f"[{episode_id}] "
            f"route={spec.route_kind} "
            f"poses={len(spec.waypoints_xyyaw)}",
            flush=True,
        )

        result = run_episode(
            episode_id,
            spec,
            args,
        )
        results.append(result)

        print(
            f"[{episode_id}] {result['state']}",
            flush=True,
        )

        if (
            result.get("success") is False
            and not args.continue_on_failure
        ):
            break

    summary = {
        "schema_version": "mssr.rc_car_nav2_batch.v1",
        "episode_count": len(results),
        "success_count": sum(
            item.get("success") is True for item in results
        ),
        "failure_count": sum(
            item.get("success") is False for item in results
        ),
        "results": results,
    }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )

    return 0 if summary["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
