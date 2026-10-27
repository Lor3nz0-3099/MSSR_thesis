#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String


ROOT = Path(__file__).resolve().parents[2]

# Make the repository-local smores_ep package available even when
# the caller has not exported scripts/smores_ep/src in PYTHONPATH.
#
# This keeps direct GUI runs and future headless campaign workers
# independent from the interactive shell environment.
SMORES_EP_SRC = ROOT / "scripts" / "smores_ep" / "src"

if str(SMORES_EP_SRC) not in sys.path:
    sys.path.insert(0, str(SMORES_EP_SRC))

from smores_ep.control.button_ik_math import (
    bounded_normal_waypoint,
    bounded_position_waypoint,
    orientation_residual,
    plan_mm8_reposition_command,
    shortest_angular_delta,
    vector_angle_deg,
)

DEFAULT_SEED = 6100
SEED = DEFAULT_SEED
BUTTON_TASK_ID = ""

# Headless self_assembly_cli converts --steps 0 into only 7200
# physics steps (= 30 simulated seconds at 240 Hz).  That is too
# short for the complete button episode: wave 2 of RC-Car8 assembly
# can begin near t=30 s.  Give headless runs the same explicit long
# episode guard used by the existing batch workflows.  The expert
# terminates the runtime normally as soon as the episode completes.
HEADLESS_SIMULATION_STEPS = 240_000
DOMAIN = os.environ["ROS_DOMAIN_ID"]

RC_STANDOFF_M = 0.70

# Region where we stop the folded MM8 and unfold for manipulation.
MM8_STANDOFF_M = 0.3250
MM8_LONG_TOL_M = 0.005
MM8_MAX_LATERAL_M = 0.12

# Nav2 does NOT need exact action success.
RC_ACCEPT_POSITION_M = 0.20
RC_ACCEPT_YAW_RAD = math.radians(35.0)


def angle_error(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


def read_json(path: Path):
    try:
        obj = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def button_candidates(state):
    """Yield legacy and task-addressed composite button observations."""

    candidates = [
        state.get("state", {}).get("course", {}).get("button"),
        state.get("course", {}).get("button"),
        state.get("global_attributes", {})
             .get("course", {}).get("button"),
    ]
    courses = (
        state.get("state", {}).get("course", {}),
        state.get("course", {}),
        state.get("global_attributes", {}).get("course", {}),
    )
    for course in courses:
        if not isinstance(course, dict):
            continue
        mission = course.get("mission", {})
        tasks = mission.get("tasks", []) if isinstance(mission, dict) else []
        if not isinstance(tasks, list):
            continue
        for task in tasks:
            if not isinstance(task, dict) or task.get("type") != "button":
                continue
            if BUTTON_TASK_ID and str(task.get("task_id", "")) != BUTTON_TASK_ID:
                continue
            parameters = task.get("parameters", {})
            if isinstance(parameters, dict):
                candidates.append(parameters.get("button"))
    return tuple(candidates)


def selected_button(state):
    """Return the addressed live button object in either course schema."""

    for button in button_candidates(state):
        if not isinstance(button, dict):
            continue
        xyz = button.get("current_center_xyz_m") or button.get("center_xyz_m")
        if isinstance(xyz, (list, tuple)) and len(xyz) >= 3:
            return button
    raise RuntimeError(
        f"Button {BUTTON_TASK_ID!r} not found in the live course observation"
    )


def button_xyz(state):
    button = selected_button(state)
    xyz = button.get("current_center_xyz_m") or button.get("center_xyz_m")
    return tuple(float(v) for v in xyz[:3])


def button_press_direction_xy(state):
    """Return normalized world-frame button press direction."""

    candidates = button_candidates(state)

    for button in candidates:

        if not isinstance(
            button,
            dict,
        ):
            continue

        direction = (
            button.get(
                "press_direction_world_xy"
            )
        )

        if (
            isinstance(
                direction,
                (list, tuple),
            )
            and len(direction) >= 2
        ):
            nx = float(
                direction[0]
            )
            ny = float(
                direction[1]
            )

            norm = math.hypot(
                nx,
                ny,
            )

            if norm > 1.0e-12:
                return (
                    nx / norm,
                    ny / norm,
                )

    # Backwards compatibility with old +Y button runs.
    return (
        0.0,
        1.0,
    )


def graph_connections(graph):
    if not graph:
        return 0

    ga = graph.get("global_attributes", {})
    if isinstance(ga, dict):
        n = ga.get("latched_connection_count")
        if n is not None:
            return int(n)

    edges = graph.get("edges", [])
    return len(edges) if isinstance(edges, list) else 0


def stop_process(proc):
    if proc is None or proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=8)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass

    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass


class Monitor(Node):

    def __init__(self):
        super().__init__("button_expert_monitor")

        self.odom = None
        self.actions = None
        self.reconfiguration = None

        self.cmd_pub = self.create_publisher(
            Twist, "/cmd_vel", 10
        )

        self.create_subscription(
            Odometry, "/odom", self.on_odom, 20
        )
        self.create_subscription(
            String, "/mssr/actions", self.on_actions, 10
        )
        self.create_subscription(
            String,
            "/mssr/expert/self_reconfiguration/state",
            self.on_reconfig,
            10,
        )

    def on_odom(self, msg):
        self.odom = msg

    def on_actions(self, msg):
        try:
            self.actions = json.loads(msg.data)
        except Exception:
            pass

    def on_reconfig(self, msg):
        try:
            self.reconfiguration = json.loads(msg.data)
        except Exception:
            pass

    def spin_until(self, predicate, timeout, message):
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.10)

            if predicate():
                return

        raise TimeoutError(message)

    def pose(self):
        if self.odom is None:
            raise RuntimeError("/odom unavailable")

        p = self.odom.pose.pose.position
        q = self.odom.pose.pose.orientation

        yaw = math.atan2(
            2.0 * (q.w*q.z + q.x*q.y),
            1.0 - 2.0 * (q.y*q.y + q.z*q.z),
        )

        return float(p.x), float(p.y), yaw


def set_behavior_dataset_context(
    env,
    *,
    dataset_path,
    stage_name=None,
):
    parameters = [
        (
            "behavior_dataset_path",
            (
                "''"
                if dataset_path is None
                else str(dataset_path)
            ),
        ),
    ]

    if stage_name is not None:
        parameters.append(
            (
                "behavior_dataset_stage_name",
                str(stage_name),
            )
        )

    for name, value in parameters:
        result = subprocess.run(
            [
                "ros2",
                "param",
                "set",
                "/smores_morphology_behavior_node",
                name,
                value,
            ],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=15,
        )

        if (
            result.returncode != 0
            or "successful" not in result.stdout.lower()
        ):
            raise RuntimeError(
                "Could not set morphology behavior dataset "
                f"parameter {name!r}: {result.stdout.strip()}"
            )


def behavior(env, name, timeout=90.0):

    command = [
        "ros2", "run", "mssr_expert",
        "mssr_smores_morphology_command_client",
        "--morphology", "mobile_manipulator8",
        "--behavior", name,
        "--command-id",
        f"button-{name}-{time.time_ns()}",
        "--parameters-json", "{}",
        "--timeout-s", str(timeout),
    ]

    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    print(result.stdout, flush=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"Behavior {name!r} failed."
        )


def wait_nav2(env):
    deadline = time.monotonic() + 60

    required = (
        "/bt_navigator",
        "/planner_server",
        "/controller_server",
    )

    last_status = "no lifecycle response"

    while time.monotonic() < deadline:
        okay = True

        for node in required:
            remaining_s = deadline - time.monotonic()

            if remaining_s <= 0.0:
                okay = False
                break

            try:
                result = subprocess.run(
                    ["ros2", "lifecycle", "get", node],
                    cwd=ROOT,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=min(5.0, remaining_s),
                )
            except subprocess.TimeoutExpired:
                last_status = f"{node}: lifecycle probe timed out"
                okay = False
                break

            output = result.stdout.strip()

            if (
                result.returncode != 0
                or "active" not in output.lower()
            ):
                last_status = (
                    f"{node}: returncode={result.returncode}, "
                    f"output={output!r}"
                )
                okay = False
                break

        if okay:
            return

        remaining_s = deadline - time.monotonic()

        if remaining_s > 0.0:
            time.sleep(min(1.0, remaining_s))

    raise TimeoutError(
        "Nav2 did not become ACTIVE within 60 s; "
        f"last_status={last_status}"
    )


def wait_assembly(graph_path, runtime):
    deadline = time.monotonic() + 300
    stable = 0

    while time.monotonic() < deadline:

        if runtime is not None and runtime.poll() is not None:
            raise RuntimeError("Runtime exited during assembly.")

        graph = read_json(graph_path)

        if graph_connections(graph) == 7:
            stable += 1

            if stable >= 5:
                return
        else:
            stable = 0

        time.sleep(0.5)

    raise TimeoutError("Eight-module assembly timeout.")



def align_rc_rear_to_button(monitor, state_path):
    """Bounded reverse-arc alignment: RC rear points toward live button."""
    state = read_json(state_path)

    if not state:
        raise RuntimeError("State graph unavailable for RC alignment.")

    bx, by, bz = button_xyz(state)

    (
        press_nx,
        press_ny,
    ) = button_press_direction_xy(
        state
    )

    monitor.spin_until(
        lambda: monitor.odom is not None,
        10,
        "/odom unavailable for RC final alignment.",
    )

    x0, y0, yaw0 = monitor.pose()

    def desired_yaw(x, y):
        bearing = math.atan2(by - y, bx - x)
        a = bearing + math.pi
        return math.atan2(math.sin(a), math.cos(a))

    desired0 = desired_yaw(x0, y0)
    err0 = angle_error(desired0, yaw0)

    print()
    print("============================================================")
    print(" RC-CAR FINAL REVERSE-ARC ALIGNMENT")
    print("============================================================")
    print(
        f"button      = ({bx:+.3f},{by:+.3f},{bz:+.3f})"
    )
    print(
        f"yaw before  = {math.degrees(yaw0):+.2f} deg"
    )
    print(
        f"desired yaw = {math.degrees(desired0):+.2f} deg"
    )
    print(
        f"yaw error   = {math.degrees(err0):+.2f} deg"
    )

    tol = math.radians(0.5)
    start = time.monotonic()
    stable = 0
    last_print = 0.0

    try:
        while time.monotonic() - start < 75.0:
            rclpy.spin_once(monitor, timeout_sec=0.04)

            x, y, yaw = monitor.pose()
            desired = desired_yaw(x, y)
            err = angle_error(desired, yaw)

            # Final RC positioning is a coarse pre-reconfiguration
            # alignment. As soon as the physical yaw error enters the
            # accepted region, stop instead of continuing the reverse arc.
            if abs(err) <= tol:
                break

            wz = max(
                -0.30,
                min(0.30, 1.2 * err),
            )

            if abs(wz) < 0.07 and abs(err) > tol:
                wz = math.copysign(0.07, err)

            # RC-Car8 is non-holonomic: pure angular.z eventually
            # stalls. Use a small reverse arc instead.
            #
            # The rear already points approximately toward the button,
            # therefore negative linear.x also reduces the standoff.
            travel = math.hypot(x - x0, y - y0)


            cmd = Twist()
            cmd.linear.x = -0.025
            cmd.angular.z = wz
            monitor.cmd_pub.publish(cmd)

            now = time.monotonic()

            if now - last_print >= 1.0:
                print(
                    f"yaw={math.degrees(yaw):+7.2f} deg  "
                    f"desired={math.degrees(desired):+7.2f} deg  "
                    f"error={math.degrees(err):+6.2f} deg  "
                    f"wz={wz:+.3f}"
                )
                last_print = now
        else:
            raise RuntimeError(
                "RC final yaw alignment timeout."
            )

    finally:
        stop = Twist()
        for _ in range(15):
            monitor.cmd_pub.publish(stop)
            rclpy.spin_once(monitor, timeout_sec=0.05)

    x1, y1, yaw1 = monitor.pose()
    desired1 = desired_yaw(x1, y1)
    err1 = angle_error(desired1, yaw1)

    print()
    print(
        f"yaw after   = {math.degrees(yaw1):+.2f} deg"
    )
    print(
        f"final error = {math.degrees(err1):+.2f} deg"
    )
    print(
        f"XY drift    = "
        f"{math.hypot(x1-x0,y1-y0)*1000:.1f} mm"
    )

    return {
        "button_xyz_m": [bx, by, bz],

        "press_direction_world_xy": [
            press_nx,
            press_ny,
        ],

        "xy_before_m": [x0, y0],
        "xy_after_m": [x1, y1],
        "yaw_before_rad": yaw0,
        "desired_yaw_before_rad": desired0,
        "yaw_error_before_rad": err0,
        "yaw_after_rad": yaw1,
        "desired_yaw_after_rad": desired1,
        "yaw_error_after_rad": err1,
    }


def validate_mm8_routing(actions):
    if not isinstance(actions, dict):
        return None

    roles = (
        actions.get("expert", {})
        .get("module_roles", {})
    )

    if not roles:
        return None

    role_to_module = {
        role: module
        for module, role in roles.items()
    }

    front = role_to_module.get("front_support")
    lift = role_to_module.get("arm_lift")
    ground = role_to_module.get("arm_ground_drive")

    if front is None or lift is None or ground is None:
        return None

    locomotion = actions.get("locomotion", {})

    if not locomotion:
        return {
            "roles": roles,
            "front": front,
            "lift": lift,
            "ground": ground,
            "active": set(),
        }

    active = set(locomotion)

    expected = {front, lift}

    if active != expected:
        raise RuntimeError(
            "\nWRONG MM8 ROUTING\n"
            f" expected front_support + arm_lift = {sorted(expected)}\n"
            f" got                              = {sorted(active)}\n"
            f" arm_ground_drive                 = {ground}"
        )

    return {
        "roles": roles,
        "front": front,
        "lift": lift,
        "ground": ground,
        "active": active,
    }


def approach_mm8(
    monitor,
    state_path,
    standoff_m=None,
    label="MM8 APPROACH",
):

    state = read_json(state_path)

    if not state:
        raise RuntimeError("State graph unavailable.")

    if standoff_m is None:
        standoff_m = MM8_STANDOFF_M

    standoff_m = float(standoff_m)

    bx, by, bz = button_xyz(state)

    (
        press_nx,
        press_ny,
    ) = button_press_direction_xy(
        state
    )

    monitor.spin_until(
        lambda: monitor.odom is not None,
        10,
        "/odom unavailable before MM8 approach.",
    )

    # Desired base center before unfolding.
    tx = bx - press_nx * standoff_m
    ty = by - press_ny * standoff_m

    # Direct MM8 assembly fixes the locomotion axis.
    # MM8 forward is opposite to the button press direction.
    drive_axis_x = -press_nx
    drive_axis_y = -press_ny

    print()
    print("============================================================")
    print(f" {label}")
    print("============================================================")
    print("Motion policy: translation only")
    print(
        "Semantic translation pair: "
        "front_support + arm_lift"
    )
    print(
        f"button = ({bx:+.3f}, {by:+.3f}, {bz:+.3f})"
    )
    print(
        f"target = ({tx:+.3f}, {ty:+.3f})"
    )
    print(
        f"drive axis = "
        f"({drive_axis_x:+.3f}, {drive_axis_y:+.3f})"
    )
    print()

    start = time.monotonic()
    best_error = float("inf")
    last_improvement = start
    last_print = 0.0
    routing_verified = False

    try:
        while time.monotonic() - start < 180:

            rclpy.spin_once(
                monitor,
                timeout_sec=0.03,
            )

            x, y, yaw = monitor.pose()

            dx = tx - x
            dy = ty - y

            longitudinal_error = (
                dx * drive_axis_x
                + dy * drive_axis_y
            )

            lateral_error = (
                -dx * drive_axis_y
                + dy * drive_axis_x
            )

            command = plan_mm8_reposition_command(
                current_x_m=x,
                current_y_m=y,
                current_yaw_rad=yaw,
                target_x_m=tx,
                target_y_m=ty,
                drive_axis_x=-press_nx,
                drive_axis_y=-press_ny,
                position_tolerance_m=MM8_LONG_TOL_M,
            )

            phase = str(command["phase"])

            if phase == "done":
                print()
                print("MM8 manipulation region reached.")
                break

            if phase != "translate":
                raise RuntimeError(
                    f"Unexpected MM8 reposition phase: {phase}"
                )

            # MM8 is not steerable.
            if abs(float(command["angular_z_rad_s"])) > 1.0e-12:
                raise RuntimeError(
                    "MM8 planner requested forbidden angular motion."
                )

            error = abs(longitudinal_error)

            if error < best_error - 0.002:
                best_error = error
                last_improvement = time.monotonic()

            if (
                routing_verified
                and time.monotonic() - last_improvement > 12
            ):
                raise RuntimeError(
                    "MM8 locomotion stalled for 12 s."
                )

            cmd = Twist()
            cmd.linear.x = float(
                command["linear_x_m_s"]
            )
            cmd.angular.z = 0.0

            monitor.cmd_pub.publish(cmd)

            routing = validate_mm8_routing(
                monitor.actions
            )

            if (
                routing is not None
                and routing["active"]
            ):
                if not routing_verified:
                    print()
                    print("ROUTING VERIFIED:")
                    print(
                        "  front_support    = "
                        f"{routing['front']}"
                    )
                    print(
                        "  arm_lift         = "
                        f"{routing['lift']}"
                    )
                    print(
                        "  arm_ground_drive = "
                        f"{routing['ground']} "
                        "(NOT driven)"
                    )
                    print()

                routing_verified = True

            now = time.monotonic()

            if now - last_print >= 1.0:
                print(
                    f"root=({x:+.3f},{y:+.3f}) "
                    f"yaw={math.degrees(yaw):+.1f}° | "
                    f"phase={phase} "
                    f"long={longitudinal_error*1000:+.1f} mm "
                    f"lat={lateral_error*1000:+.1f} mm "
                    f"v={cmd.linear.x:+.3f} "
                    f"w={cmd.angular.z:+.3f}"
                )
                last_print = now

            time.sleep(0.04)

        else:
            raise RuntimeError(
                "MM8 reposition timeout."
            )

    finally:
        stop = Twist()

        for _ in range(15):
            monitor.cmd_pub.publish(stop)
            rclpy.spin_once(
                monitor,
                timeout_sec=0.05,
            )

    if not routing_verified:
        raise RuntimeError(
            "Could not verify live MM8 drive routing."
        )

    x, y, yaw = monitor.pose()

    return {
        "button_xyz_m": [bx, by, bz],
        "press_direction_world_xy": [
            press_nx,
            press_ny,
        ],
        "drive_axis_world_xy": [
            drive_axis_x,
            drive_axis_y,
        ],
        "target_xy_m": [tx, ty],
        "root_xy_m": [x, y],
        "root_yaw_rad": yaw,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the direct-assembly MobileManipulator8 button-press expert."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=(
            "Deterministic button-target seed "
            f"(default: {DEFAULT_SEED})."
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "Run Isaac without the GUI and close the runtime "
            "automatically when the episode terminates."
        ),
    )
    parser.add_argument(
        "--mm8-settle-distance-m",
        type=float,
        default=None,
        help=(
            "Override the post-assembly MM8/Scorpion "
            "reposition distance from the button. Default: use the "
            "validated legacy MM8_STANDOFF_M."
        ),
    )

    parser.add_argument(
        "--stop-before-ik",
        action="store_true",
        help=(
            "Stop after prepare_manipulation and save the complete "
            "physical PRE-IK state graph, before issuing any IK "
            "joint command."
        ),
    )
    parser.add_argument(
        "--external-runtime-dir",
        type=Path,
        help=(
            "Attach to an already-running runtime whose robot already has "
            "the validated MobileManipulator8 topology. Standalone "
            "rerecording should omit this option so direct assembly is logged."
        ),
    )
    parser.add_argument(
        "--button-task-id",
        default="",
        help="Select one button from course.mission.tasks in composite mode.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Append every button sub-stage to this composite JSONL dataset.",
    )
    parser.add_argument(
        "--dataset-layout-json",
        type=Path,
        help=(
            "Optional JSON mapping direct button sub-stages to distinct raw "
            "dataset paths. Supported keys: assembly, mm8_behavior, "
            "manipulation. Legacy --dataset-path remains supported."
        ),
    )
    parser.add_argument(
        "--episode-id",
        default="",
        help="Episode identifier used when appending to a composite dataset.",
    )
    parser.add_argument(
        "--dataset-log-period",
        type=int,
        default=1,
        help=(
            "Log every Nth control tick instead of every tick. Raise this "
            "for a smoother GUI inspection run; keep it at 1 for real "
            "dataset campaigns."
        ),
    )

    return parser.parse_args()


def main():
    global BUTTON_TASK_ID, SEED

    args = parse_args()
    SEED = int(args.seed)
    BUTTON_TASK_ID = str(args.button_task_id).strip()
    if args.external_runtime_dir is not None and not BUTTON_TASK_ID:
        raise SystemExit(
            "--external-runtime-dir requires --button-task-id so repeated "
            "buttons cannot be confused"
        )

    run_id = time.strftime(
        f"seed-{SEED:06d}-%Y%m%d-%H%M%S"
    )

    run = (
        ROOT
        / "logs"
        / "button_expert_to_ik"
        / run_id
    )

    run.mkdir(parents=True, exist_ok=False)
    runtime_dir = (
        args.external_runtime_dir.expanduser().resolve()
        if args.external_runtime_dir is not None
        else run / "runtime"
    )
    if args.external_runtime_dir is None:
        runtime_dir.mkdir(parents=True, exist_ok=False)
    elif not (runtime_dir / "state_graph.json").is_file():
        raise SystemExit(
            f"External runtime has no state_graph.json: {runtime_dir}"
        )
    episode_id = str(args.episode_id).strip() or run_id
    dataset_path = (
        args.dataset_path.expanduser().resolve()
        if args.dataset_path is not None
        else run / "button_dataset.jsonl"
    )

    explicit_dataset_path = args.dataset_path is not None

    if args.dataset_layout_json is not None:
        dataset_layout_path = (
            args.dataset_layout_json.expanduser().resolve()
        )
        dataset_layout = json.loads(
            dataset_layout_path.read_text(encoding="utf-8")
        )
        if not isinstance(dataset_layout, dict):
            raise SystemExit(
                "--dataset-layout-json must contain a JSON object"
            )

        def _optional_layout_path(key, fallback):
            raw = dataset_layout.get(key)
            if not isinstance(raw, str) or not raw.strip():
                return Path(fallback).resolve()
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = dataset_layout_path.parent / candidate
            return candidate.resolve()

        assembly_dataset_path = _optional_layout_path(
            "assembly",
            dataset_path,
        )
        mm8_behavior_dataset_path = _optional_layout_path(
            "mm8_behavior",
            dataset_path,
        )
        manipulation_dataset_path = _optional_layout_path(
            "manipulation",
            dataset_path,
        )
    elif explicit_dataset_path:
        assembly_dataset_path = dataset_path
        mm8_behavior_dataset_path = dataset_path
        manipulation_dataset_path = dataset_path
    else:
        assembly_dataset_path = run / "assembly_dataset.jsonl"
        mm8_behavior_dataset_path = run / "behavior_dataset.jsonl"
        manipulation_dataset_path = run / "manipulation_dataset.jsonl"

    Path(
        "/tmp/mssr_button_current_run"
    ).write_text(str(run) + "\n")

    env = dict(os.environ)

    env["PYTHONUNBUFFERED"] = "1"
    env["ROS_DOMAIN_ID"] = DOMAIN
    env["RMW_IMPLEMENTATION"] = (
        "rmw_cyclonedds_cpp"
    )
    env["ROS_LOG_DIR"] = str(run / "ros_logs")

    Path(env["ROS_LOG_DIR"]).mkdir()

    runtime = None
    assembly = None

    success = False

    rclpy.init()
    monitor = Monitor()

    try:

        # ========================================================
        # 1. RUNTIME
        # ========================================================

        print()
        print("============================================================")
        print(" PHASE 1 — FRESH RUNTIME")
        print("============================================================")

        runtime_log = (
            run / "runtime.log"
        ).open("w")

        runtime = (
            None
            if args.external_runtime_dir is not None
            else subprocess.Popen(
            [
                "ros2", "launch",
                "mssr_expert",
                "smores_runtime.launch.py",

                f"runtime_dir:={runtime_dir}",
                "module_count:=8",

                "button_test_course:=true",
                f"button_seed:={SEED}",

                (
                    "headless:=true"
                    if args.headless
                    else "headless:=false"
                ),
                "performance:=true",
                "simple_visuals:=true",

                (
                    f"simulation_steps:={HEADLESS_SIMULATION_STEPS}"
                    if args.headless
                    else "simulation_steps:=0"
                ),
                "simulation_speed_factor:=1.0",

                "actuator_effort_scale:=4.0",
                "wheel_friction_scale:=1.50",
                "tilt_effort_scale:=8.0",

                "behavior_dataset_path:="
                + str(mm8_behavior_dataset_path),
                "behavior_dataset_episode_id:="
                + episode_id,
                "behavior_dataset_stage_name:="
                + "button_expert",
                "behavior_dataset_difficulty:=0.0",
                f"behavior_dataset_log_period:={args.dataset_log_period}",
                "behavior_control_rate_hz:=10.0",

                f"ros_domain_id:={DOMAIN}",
                "rmw_implementation:="
                + "rmw_cyclonedds_cpp",
            ],
            cwd=ROOT,
            env=env,
            stdout=runtime_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            )
        )

        # Isaac Sim cold startup can legitimately take longer than
        # 90 s while Kit extensions are being initialized.  Killing the
        # launch during that import phase can leave misleading secondary
        # Python-extension / torch import errors in runtime.log.
        deadline = time.monotonic() + 180

        while time.monotonic() < deadline:

            if runtime is not None and runtime.poll() is not None:
                raise RuntimeError(
                    "Runtime exited during startup."
                )

            if read_json(
                runtime_dir / "state_graph.json"
            ):
                break

            time.sleep(0.25)
        else:
            raise TimeoutError(
                "Runtime state graph timeout after 180 s."
            )

        print("Runtime READY.")

        # ========================================================
        # 2. DIRECT MOBILEMANIPULATOR8 ASSEMBLY
        # ========================================================

        print()
        print("============================================================")
        print(" PHASE 2 — MOBILEMANIPULATOR8 ASSEMBLY")
        print("============================================================")

        assembly_log = (run / "assembly.log").open("w")

        assembly = (
            None
            if args.external_runtime_dir is not None
            else subprocess.Popen(
                [
                    "ros2", "run",
                    "mssr_expert",
                    "mssr_smores_self_assembly_node",
                    "--ros-args",
                    "-p",
                    "target_graph_path:="
                    + str(
                        ROOT
                        / "mssr_ws/src/mssr_expert/"
                          "config/smores_mobile_manipulator8.json"
                    ),
                    "-p",
                    "execution_id:=" + run_id + "-assembly",
                    "-p",
                    "episode_id:=" + episode_id,
                    "-p",
                    "dataset_path:=" + str(assembly_dataset_path),
                ],
                cwd=ROOT,
                env=env,
                stdout=assembly_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        )

        wait_assembly(
            runtime_dir / "robot_graph.json",
            runtime,
        )

        time.sleep(1 if args.external_runtime_dir is not None else 8)
        stop_process(assembly)
        assembly = None

        set_behavior_dataset_context(
            env,
            dataset_path=mm8_behavior_dataset_path,
            stage_name="button_mm8_behavior",
        )

        print("MobileManipulator8 READY.")

        monitor.spin_until(
            lambda: monitor.odom is not None,
            15,
            "MobileManipulator8 /odom unavailable.",
        )

        # ========================================================
        # 4. BUTTON-RELATIVE REPOSITION / BACK-AWAY
        # ========================================================

        print()
        print("============================================================")
        print(" PHASE 4 — MM8 BUTTON-RELATIVE REPOSITION")
        print(" front_support + arm_lift")
        print("============================================================")

        result_reposition = approach_mm8(
            monitor,
            runtime_dir / "state_graph.json",
            standoff_m=args.mm8_settle_distance_m,
            label="MM8 REPOSITION / BACK-AWAY",
        )

        (run / "mm8_reposition.json").write_text(
            json.dumps(result_reposition, indent=2) + "\n"
        )

        # ========================================================
        # 5. UNFOLD FOR MANIPULATION
        # ========================================================

        print()
        print("============================================================")
        print(" PHASE 5 — PREPARE MANIPULATION")
        print("============================================================")

        behavior(
            env,
            "prepare_manipulation",
        )
        time.sleep(1.0)

        state = read_json(
            runtime_dir / "state_graph.json"
        )

        bx, by, bz = button_xyz(state)

        x, y, yaw = monitor.pose()

        routing = validate_mm8_routing(
            monitor.actions
        )

        roles = (
            routing["roles"]
            if routing is not None
            else {}
        )

        role_to_module = {
            role: module
            for module, role in roles.items()
        }

        modules = {
            m["module_id"]: m
            for m in state["state"]["modules"]
        }

        chain = (
            "arm_ground_drive",
            "arm_lift",
            "arm_link",
            "end_effector",
        )

        q = {}

        for role in chain:

            module = role_to_module.get(role)

            if module not in modules:
                continue

            actuators = modules[module][
                "actuators"
            ]

            q[role] = {
                "module_id": module,
                "tilt_rad": float(
                    actuators["tilt"][
                        "position_rad"
                    ]
                ),
                "pan_rad": float(
                    actuators["pan"][
                        "position_rad"
                    ]
                ),
            }

        ready = {
            "schema_version":
                "mssr.button_ready_for_ik.v1",

            "seed": SEED,

            "state":
                "READY_FOR_5DOF_IK",

            "success": True,

            "button_xyz_m":
                [bx, by, bz],

            "root_xy_m":
                [x, y],

            "root_yaw_rad":
                yaw,

            "scorpion_drive_roles": [
                "front_support",
                "arm_lift",
            ],

            "ik_chain_roles": list(chain),

            "ik_dofs": [
                "arm_lift.tilt",
                "arm_link.tilt",
                "arm_link.pan",
                "end_effector.tilt",
                "end_effector.pan",
            ],

            "live_role_assignment":
                roles,

            "q_current": q,

            "reposition":
                result_reposition,
        }

        (
            run / "ready_for_ik.json"
        ).write_text(
            json.dumps(
                ready,
                indent=2,
            ) + "\n"
        )

        pre_ik_state_path = (
            run / "pre_ik_state_graph.json"
        )

        pre_ik_state_path.write_text(
            json.dumps(
                state,
                indent=2,
            ) + "\n"
        )

        print(
            f"PRE-IK full state graph = "
            f"{pre_ik_state_path}"
        )

        print()
        print("============================================================")
        print(" READY FOR 5-DOF IK / LOCAL PRESS")
        print("============================================================")
        print(
            f"button = "
            f"({bx:+.3f},{by:+.3f},{bz:+.3f})"
        )
        print(
            f"root   = "
            f"({x:+.3f},{y:+.3f})"
        )
        print(
            f"yaw    = "
            f"{math.degrees(yaw):+.1f}°"
        )

        print()
        print("IK chain:")
        print(
            "  arm_ground_drive"
            " -> arm_lift"
            " -> arm_link"
            " -> end_effector"
        )

        print()
        print("Current IK state:")

        for role in chain:
            item = q.get(role)

            if item is None:
                print(
                    f"  {role}: unavailable"
                )
                continue

            print(
                f"  {role:<18} "
                f"{item['module_id']}: "
                f"T={math.degrees(item['tilt_rad']):+7.2f}° "
                f"P={math.degrees(item['pan_rad']):+7.2f}°"
            )

        print()
        print(f"RUN={run}")
        print()
        print(
            "Isaac + bridge + behavior node remain OPEN."
        )
        print(
            "Assembly is STOPPED; MM8 behavior/primitive executor remain active."
        )
        if args.stop_before_ik:
            print(
                "5-DoF BOTTOM-face IK intentionally SKIPPED."
            )
            print(
                "No IK joint command has been sent."
            )
            print(
                "PRE-IK snapshot acquisition: COMPLETE"
            )
            return

        print("5-DoF BOTTOM-face IK follows.")

        # ========================================================
        # 6. 5-DOF BOTTOM-FACE IK
        #
        # The free task face is end_effector.BOTTOM.
        # No physical finite-difference probing is performed.
        #
        # q_pre:
        #   BOTTOM centre aligned with button centre,
        #   BOTTOM normal = task press normal N,
        #   3 mm before the resting plunger face.
        #
        # q_press:
        #   same geometric constraint,
        #   advanced along task press normal N to depress the plunger.
        # ========================================================

        print()
        print("============================================================")
        print(" PHASE 6 — 5-DOF BOTTOM-FACE IK")
        print(" no physical probes")
        print("============================================================")

        import numpy as _np

        from std_msgs.msg import String as _IKString
        from mssr_expert.execution.primitive_protocol import (
            PrimitiveGoalRequest as _IKPrimitiveGoalRequest,
            TERMINAL_STATES as _IK_TERMINAL_STATES,
        )
        from mssr_expert.behaviors.morphology_dof_model import (
            SmoresMorphologyDofAnalyzer as _IKDofAnalyzer,
        )
        from mssr_expert.graph.graph_features import (
            graph_to_features as _ik_graph_to_features,
        )
        from mssr_expert.graph.serialization import (
            attributed_graph_from_dict as _ik_graph_from_dict,
            load_attributed_graph as _ik_load_graph,
        )
        from mssr_expert.planning.smores_ep.attributed_adapter import (
            target_roles_from_graph as _ik_target_roles,
        )
        from mssr_expert.primitives.common import (
            module_position as _ik_module_position,
        )
        from smores_ep.config.geometry import (
            SmoresGeometry as _IKGeometry,
        )

        _ik_state_path = runtime_dir / "state_graph.json"

        # --------------------------------------------------------
        # Button-local task frame.
        #
        # N: physical press direction in world XY.
        # T: horizontal tangent to the button face.
        # Z: world vertical.
        #
        # Historical +Y button:
        #   N=(0,+1,0), T=(-1,0,0)
        #
        # The same IK below now works unchanged in concept for
        # +/-X and +/-Y fixtures.
        # --------------------------------------------------------

        _ik_task_state = read_json(
            _ik_state_path
        )

        if not _ik_task_state:
            raise RuntimeError(
                "state_graph unavailable before IK task-frame setup"
            )

        (
            _ik_press_nx,
            _ik_press_ny,
        ) = button_press_direction_xy(
            _ik_task_state
        )

        _IK_TARGET_NORMAL = _np.asarray(
            [
                _ik_press_nx,
                _ik_press_ny,
                0.0,
            ],
            dtype=float,
        )

        _IK_FACE_TANGENT = _np.asarray(
            [
                -_ik_press_ny,
                +_ik_press_nx,
                0.0,
            ],
            dtype=float,
        )

        print()
        print(
            "Button IK task frame:"
        )
        print(
            "  N = "
            f"[{_IK_TARGET_NORMAL[0]:+.1f}, "
            f"{_IK_TARGET_NORMAL[1]:+.1f}, 0.0]"
        )
        print(
            "  T = "
            f"[{_IK_FACE_TANGENT[0]:+.1f}, "
            f"{_IK_FACE_TANGENT[1]:+.1f}, 0.0]"
        )

        # Kinematic order from upstream TOP toward free BOTTOM.
        #
        # arm_lift.pan is deliberately frozen because it belongs
        # to the stable support configuration.
        #
        # For arm_link and end_effector, PAN is upstream of TILT
        # when traversing TOP -> BOTTOM.
        _IK_DOF = (
            ("arm_lift", "tilt"),
            ("arm_link", "pan"),
            ("arm_link", "tilt"),
            ("end_effector", "pan"),
            ("end_effector", "tilt"),
        )

        _IK_PRE_GAP_M = 0.0300
        _IK_PRESS_DEPTH_M = 0.0038


        # --------------------------------------------------------
        # Effective BOTTOM contact patch.
        #
        # The task is surface-to-surface overlap, not forcing the
        # geometric centre of the BOTTOM connector onto the button.
        #
        # With the BOTTOM normal constrained close to +Y, the two
        # contact surfaces are approximately parallel to the X-Z
        # button plane.  The admissible BOTTOM-centre interval is
        # therefore the Minkowski expansion of the button rectangle
        # by this conservative effective contact half-size.
        # --------------------------------------------------------
        _IK_EE_CONTACT_HALF_TANGENT_M = 0.0200
        _IK_EE_CONTACT_HALF_Z_M = 0.0200

        # --------------------------------------------------------
        # Closed-loop inverse kinematics / resolved-rate controller.
        #
        # Outer loop:
        #
        #   xdot_cmd = xdot_d - Kp * r(x)
        #
        #   qdot = J# xdot_cmd
        #          + (I - J#J) qdot_null
        #
        # Damped pseudoinverse:
        #
        #   J# = J.T (J J.T + lambda^2 I)^-1
        #
        # The present primitive transport is position-based, so qdot
        # is integrated into small absolute q targets and the REAL
        # state graph is read again after every step.
        # --------------------------------------------------------
        _CLIK_CARTESIAN_STEP_M = 0.0020
        _CLIK_WAYPOINT_TOL_M = 0.0015

        _CLIK_FREE_SPEED_M_S = 0.0060
        _CLIK_CONTACT_SPEED_M_S = 0.0015

        _CLIK_KP_S_INV = 2.0
        _CLIK_DAMPING = 0.020

        # Secondary objective in the Jacobian nullspace:
        # remain near the mechanically comfortable q_pre while also
        # softly avoiding joint limits.
        _CLIK_NULL_PRE_GAIN_S_INV = 0.35
        _CLIK_LIMIT_GAIN_S_INV = 0.06

        _CLIK_INTEGRATION_DT_S = 0.25
        _CLIK_QDOT_MAX_RAD_S = math.radians(15.0)

        _CLIK_MAX_CONTROL_STEPS = 80
        _CLIK_STALL_STEPS = 12
        _CLIK_MIN_PROGRESS_M = 0.00025
        _CLIK_POST_COMMAND_SETTLE_S = 0.08

        # PRESS is a local motion from the physically reached PRE
        # configuration.  The regional X/Z residual is deliberately
        # zero inside the button rectangle, which leaves a nullspace
        # in which the two PAN joints can drift into cancelling,
        # non-local configurations.  Keep PRESS PAN motion local to
        # the actual PRE state while leaving PRE itself fully 5-DoF.
        _IK_PRESS_PAN_DELTA_LIMIT_RAD = math.radians(5.0)
        _IK_PLUNGER_DEPTH_M = 0.040

        # Button front face is an 80 x 80 mm rectangle in X-Z.
        # The task does NOT require the BOTTOM-face centre to hit
        # the button centre. Any point on this usable surface is valid.
        _IK_BUTTON_HALF_TANGENT_M = 0.040
        _IK_BUTTON_HALF_Z_M = 0.040

        _IK_SUCCESS_DEPRESSION_M = 0.0035

        _IK_ORIENTATION_LEVER_M = 0.055
        _IK_POSITION_TOL_M = 0.0020
        _IK_NORMAL_TOL_DEG = 3.0
        _IK_NORMAL_DEADBAND_DEG = 30.0
        _PREORIENT_TILT_SCAN_STEP_DEG = 2.0
        _PREORIENT_MIN_IMPROVEMENT_DEG = 5.0
        _PREORIENT_MIN_MOVE_DEG = 1.0

        # PRE is reached through local task-space waypoints, always
        # re-linearized from the physical state after the previous
        # command.  This makes the approach independent from the
        # post-reconfiguration PAN reference captured at docking.
        _PRE_POSITION_STEP_M = 0.015
        _PRE_NORMAL_STEP_RAD = math.radians(15.0)
        _PRE_JOINT_STEP_MAX_RAD = math.radians(6.0)
        _PRE_MAX_CONTROL_STEPS = 40
        _PRE_STALL_STEPS = 8
        _PRE_PROGRESS_EPS = 1.0e-5
        _PRE_POST_COMMAND_SETTLE_S = 0.10

        _ik_geom = _IKGeometry()

        # --------------------------------------------------------
        # Raw manipulation dataset.
        #
        # One record now represents ONE actually admitted low-level
        # internal-motion primitive.  The complete q_target / CLIK decision
        # remains provenance, while the supervised action is the single
        # primitive_goal that physically produced G_t -> G_t+1.
        # Busy/poll states are never emitted as WAIT labels.
        # --------------------------------------------------------

        _ik_robot_graph_path = (
            runtime_dir / "robot_graph.json"
        )

        _ik_dataset_path = manipulation_dataset_path

        _ik_dataset_timestep = 0

        _ik_dof_analyzer = _IKDofAnalyzer()

        _ik_target_graph = _ik_load_graph(
            ROOT
            / "mssr_ws/src/mssr_expert/config/"
              "smores_mobile_manipulator8.json"
        )

        _ik_target_role_specs = (
            _ik_target_roles(
                _ik_target_graph
            )
        )

        _ik_assignment = {}

        # Runtime robot_graph nodes do not necessarily carry the
        # target-role labels themselves. Preserve the validated
        # role assignment explicitly in the RAW manipulation data.
        _ik_module_to_role = {}

        # ``roles`` is normally module_id -> target_role in the
        # validated button runner. Accept the reverse orientation too
        # so RAW role annotation does not depend on mapping direction.
        for key, value in roles.items():
            key = str(key)
            value = str(value)

            if key.startswith("smores_"):
                _ik_module_to_role[key] = value
            elif value.startswith("smores_"):
                _ik_module_to_role[value] = key

        if len(_ik_module_to_role) != 8:
            raise RuntimeError(
                "Incomplete MobileManipulator8 module-to-role "
                f"mapping for RAW dataset: {_ik_module_to_role}"
            )

        for (
            target_vertex_id,
            role_spec,
        ) in _ik_target_role_specs.items():

            target_role = str(
                role_spec.get(
                    "target_role",
                    "",
                )
            )

            module_id = (
                role_to_module.get(
                    target_role
                )
            )

            if module_id:
                _ik_assignment[
                    str(target_vertex_id)
                ] = str(module_id)

        if (
            len(_ik_assignment)
            != len(
                _ik_target_role_specs
            )
        ):
            raise RuntimeError(
                "Cannot build complete MobileManipulator8 "
                "target-to-module assignment for manipulation "
                "dataset recording."
            )

        print(
            "Manipulation raw dataset = "
            f"{_ik_dataset_path}"
        )

        def _ik_read_robot_graph():

            payload = read_json(
                _ik_robot_graph_path
            )

            if not payload:
                raise RuntimeError(
                    "robot_graph unavailable during IK "
                    "dataset recording"
                )

            return _ik_graph_from_dict(
                payload
            )

        def _ik_read_transition_endpoint(
            graph_before,
            *,
            timeout_s=2.0,
        ):
            """Return a real post-action sample with a newer physical stamp."""

            deadline = time.monotonic() + float(timeout_s)
            last_state = None
            last_graph = None

            while time.monotonic() < deadline:
                last_state = _ik_read_state()
                last_graph = _ik_read_robot_graph()
                if last_graph.stamp > graph_before.stamp:
                    return last_state, last_graph
                rclpy.spin_once(_ik_node, timeout_sec=0.02)
                time.sleep(0.01)

            raise RuntimeError(
                "IK primitive completed but no newer physical graph sample "
                f"arrived after stamp {graph_before.stamp!r}; "
                f"last_stamp={getattr(last_graph, 'stamp', None)!r}"
            )

        def _ik_button_depression(
            state,
        ):

            try:
                return float(selected_button(state).get("depression_m", 0.0))
            except (RuntimeError, TypeError, ValueError):
                return 0.0

        def _ik_dof_payload(
            graph,
        ):

            inventory = (
                _ik_dof_analyzer
                .analyze(
                    graph
                )
            )

            result = []

            for module in inventory.modules:

                result.append(
                    {
                        "module_id":
                            module.module_id,

                        "target_role":
                            _ik_module_to_role.get(
                                module.module_id,
                                module.target_role,
                            ),

                        "connected_faces":
                            sorted(
                                module.connected_faces
                            ),

                        "body_is_directly_attached":
                            module.body_is_directly_attached,

                        "ground_support_anchor":
                            module.ground_support_anchor,

                        "dofs": [
                            {
                                "name":
                                    dof.name,

                                "joint_kind":
                                    dof.joint_kind,

                                "affected_face":
                                    dof.affected_face,

                                "mode":
                                    dof.mode,

                                "connected":
                                    dof.connected,

                                "position_rad":
                                    dof.position_rad,

                                "lower_limit_rad":
                                    dof.lower_limit_rad,

                                "upper_limit_rad":
                                    dof.upper_limit_rad,

                                "max_effort_nm":
                                    dof.max_effort_nm,

                                "motor_mix": [
                                    list(item)
                                    for item
                                    in dof.motor_mix
                                ],

                                "locomotion_capable":
                                    dof.locomotion_capable,

                                "shape_capable":
                                    dof.shape_capable,
                            }
                            for dof in module.dofs
                        ],
                    }
                )

            return result

        def _ik_dataset_observation(
            state,
            graph,
            label,
        ):

            positions = {}

            for node in graph.nodes:

                try:
                    positions[
                        node.module_id
                    ] = list(
                        _ik_module_position(
                            node.attributes
                        )
                    )
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                ):
                    continue

            course = (
                graph.global_attributes
                .get(
                    "course",
                    {},
                )
            )

            if not isinstance(
                course,
                dict,
            ):
                course = {}

            module_geometry = (
                graph.global_attributes
                .get(
                    "module_geometry",
                    {},
                )
            )

            if not isinstance(
                module_geometry,
                dict,
            ):
                module_geometry = {}

            runtime_state = (
                state.get(
                    "state",
                    {},
                )
                if isinstance(
                    state,
                    dict,
                )
                else {}
            )

            return {
                "schema_version":
                    "mssr.button_manipulation_observation.v1",

                "command_id":
                    f"button-ik-{label}",

                "morphology":
                    "mobile_manipulator8",

                "behavior":
                    "button_manipulation_ik",

                "task_context": {
                    "schema_version":
                        "mssr.behavior_task_context.v1",

                    "instruction":
                        (
                            "Press the seeded physical button "
                            "with end_effector.BOTTOM using the "
                            "validated 5-DoF closed-loop IK."
                        ),

                    "morphology":
                        "mobile_manipulator8",

                    "behavior":
                        "button_manipulation_ik",

                    "success_criteria": {
                        "physical_button_depression_m":
                            _IK_SUCCESS_DEPRESSION_M,

                        "end_effector_face":
                            "BOTTOM",

                        "preserve_connection_count":
                            7,
                    },
                },

                # Preserve the same stage/course information used
                # by the other raw expert datasets.
                "environment": {
                    "schema_version":
                        "mssr.environment_observation.v1",

                    "source":
                        "isaac_world_ground_truth",

                    "frame_id":
                        str(
                            course.get(
                                "frame_id",
                                "world",
                            )
                        ),

                    "stage_name":
                        "button_expert",

                    "difficulty":
                        0.0,

                    "course":
                        dict(course),

                    "module_geometry":
                        dict(module_geometry),
                },

                # Direct aliases retained deliberately in RAW.
                "course":
                    dict(course),

                "global_attributes":
                    dict(
                        graph.global_attributes
                    ),

                "module_positions_world_m":
                    positions,

                "operational_dofs":
                    _ik_dof_payload(
                        graph
                    ),

                # Do NOT compact this now.  It contains the detailed
                # Isaac state used by the physical CLIK, including
                # modules/connectors/button state.
                "runtime_state":
                    dict(runtime_state),

                "controller": {
                    "configuration_label":
                        str(label),

                    "end_effector_face":
                        "BOTTOM",
                },
            }

        def _ik_dataset_phase(
            label,
        ):

            if label == "pre":
                return (
                    "BUTTON_IK_PRE",
                    "absolute_joint_configuration",
                )

            if str(label).startswith(
                "clik-"
            ):
                return (
                    "BUTTON_CLIK_PRESS",
                    "closed_loop_resolved_rate_dls",
                )

            if label == "pre-return":
                return (
                    "BUTTON_IK_RETURN",
                    "absolute_joint_configuration",
                )

            if (
                label
                == "post-press-clearance"
            ):
                return (
                    "BUTTON_IK_CLEARANCE",
                    "absolute_joint_configuration",
                )

            return (
                "BUTTON_IK",
                "absolute_joint_configuration",
            )

        def _ik_record_primitive_transition(
            *,
            label,
            q_target,
            primitive_goal,
            primitive_index,
            primitive_count,
            state_before,
            graph_before,
            state_after,
            graph_after,
            context,
        ):
            """Write exactly one physical primitive transition for BC."""

            nonlocal _ik_dataset_timestep

            primitive_goal = dict(primitive_goal or {})
            if not primitive_goal.get("primitive"):
                raise RuntimeError(
                    "Cannot record an IK transition without primitive_goal"
                )

            phase, controller = _ik_dataset_phase(str(label))
            q_target_map = {
                f"{role}.{joint}": float(q_target[index])
                for index, (role, joint) in enumerate(_IK_DOF)
            }

            depression_before = _ik_button_depression(state_before)
            depression_after = _ik_button_depression(state_after)
            progress = max(
                0.0,
                min(
                    1.0,
                    depression_after
                    / max(_IK_SUCCESS_DEPRESSION_M, 1.0e-12),
                ),
            )
            context = dict(context or {})

            primitive_params = {
                "configuration_label": str(label),
                "controller": controller,
                "q_target_rad": q_target_map,
                "primitive_index": int(primitive_index),
                "primitive_count": int(primitive_count),
                **context,
            }
            task_metrics = {
                "phase": phase,
                "progress": progress,
                "button_depression_before_m": depression_before,
                "button_depression_after_m": depression_after,
                "button_success_threshold_m": _IK_SUCCESS_DEPRESSION_M,
                "controller": controller,
                **context,
            }

            observation_before = _ik_dataset_observation(
                state_before,
                graph_before,
                str(label),
            )
            observation_after = _ik_dataset_observation(
                state_after,
                graph_after,
                str(label),
            )

            active_primitive = str(primitive_goal["primitive"])
            record = {
                "schema_version": "mssr.expert_transition.v3",
                "episode_id": episode_id,
                "timestep": int(_ik_dataset_timestep),
                "stamp": graph_before.stamp,
                "stage_id": int(_ik_dataset_timestep),
                "stage_name": "button_expert",
                "task_type": "button_manipulation_ik",
                "difficulty": 0.0,
                "fsm_state": phase,
                "is_first": _ik_dataset_timestep == 0,
                "is_last": False,
                "is_terminal": False,
                "action_valid": True,
                "reward": 0.0,
                "discount": 1.0,
                "observation": observation_before,
                "observation_t_plus_1": observation_after,
                "graph_t": graph_before.to_dict(),
                "target_graph": _ik_target_graph.to_dict(),
                "task_graph_t": graph_before.to_dict(),
                "assignment_target_to_module": dict(_ik_assignment),
                "graph_t_plus_1": graph_after.to_dict(),
                "attributed_graph": graph_before.to_dict(),
                "attributed_task_graph": graph_before.to_dict(),
                "graph_features": _ik_graph_to_features(graph_before),
                "expert_action": {
                    "locomotion": {},
                    "magnetic": [],
                    "primitive_goal": primitive_goal,
                },
                "supervision": {
                    "label_source": "deterministic_expert",
                    "executed_action_source": "deterministic_expert",
                    "expert_intervention": False,
                    "valid_for_behavior_cloning": True,
                },
                "expert_annotation": {
                    "fsm_state": phase,
                    "active_primitive": active_primitive,
                    "primitive_params": primitive_params,
                    "task_metrics": task_metrics,
                    "debug": {
                        "message": "One accepted low-level IK primitive.",
                        "command_id": str(primitive_goal.get("goal_id", "")),
                        "morphology": "mobile_manipulator8",
                        "behavior": "button_manipulation_ik",
                    },
                },
                "active_primitive": active_primitive,
                "primitive_params": primitive_params,
                "module_roles": dict(roles),
                "attachment_modes": {},
                "task_metrics": task_metrics,
                "success": False,
                "done": False,
                "debug": {
                    "message": "One accepted low-level IK primitive.",
                    "command_id": str(primitive_goal.get("goal_id", "")),
                    "morphology": "mobile_manipulator8",
                    "behavior": "button_manipulation_ik",
                },
            }

            with _ik_dataset_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(record, separators=(",", ":")) + "\n"
                )

            print(
                "IL manipulation sample "
                f"{_ik_dataset_timestep}: {phase} | "
                f"primitive={active_primitive} | "
                f"index={primitive_index}/{primitive_count} | "
                f"depression={depression_after*1000:.3f} mm"
            )
            _ik_dataset_timestep += 1

        def _ik_read_state():

            state = read_json(
                _ik_state_path
            )

            if not state:
                raise RuntimeError(
                    "state_graph unavailable during IK"
                )

            return state

        def _ik_goal_status(goal_id):
            """Look up one goal's admission/terminal status by id.

            The previous joint's own goal can still be releasing its
            ``internal_motion:<module>`` resource by the time the next
            joint is published; without this check a REJECTED
            (RESOURCE_BUSY) goal was silently treated as accepted and
            the caller just watched a joint that was never actually
            commanded until it timed out.
            """

            payload = read_json(
                runtime_dir / "primitive_status.json"
            )

            if not payload:
                return None

            for status in payload.get(
                "statuses", ()
            ):
                if status.get("goal_id") == goal_id:
                    return status

            return None

        def _ik_wait_goal_terminal(
            goal_id,
            label,
            role,
            joint,
            timeout_s=20.0,
        ):
            """Wait until Isaac has released the primitive resource.

            Reaching the requested physical joint angle is not enough:
            the primitive may still own ``internal_motion:<module>``.
            The following joint/configuration must not be admitted until
            the previous goal has published a terminal status.
            """

            deadline = time.monotonic() + float(timeout_s)

            while time.monotonic() < deadline:

                rclpy.spin_once(
                    _ik_node,
                    timeout_sec=0.05,
                )

                status = _ik_goal_status(goal_id)

                if status is None:
                    continue

                state = str(
                    status.get("state", "")
                ).lower()

                if state not in _IK_TERMINAL_STATES:
                    continue

                if state == "succeeded":
                    return status

                raise RuntimeError(
                    f"{label}: {role}.{joint} "
                    f"primitive ended in {state}: "
                    f"{status.get('message', '')}"
                )

            latest = _ik_goal_status(goal_id)

            raise RuntimeError(
                f"{label}: {role}.{joint} reached its "
                "physical target but primitive did not become "
                f"terminal within {timeout_s:.1f}s; "
                f"latest_status={latest}"
            )

        def _ik_modules(state):

            return {
                item["module_id"]: item
                for item in state["state"]["modules"]
            }

        def _ik_module(state, role):

            module_id = role_to_module.get(role)

            if not module_id:
                raise RuntimeError(
                    f"Missing role assignment: {role}"
                )

            modules = _ik_modules(state)

            if module_id not in modules:
                raise RuntimeError(
                    f"Module unavailable for role {role}: "
                    f"{module_id}"
                )

            return modules[module_id]

        def _ik_connector(state, role, face):

            module = _ik_module(
                state,
                role,
            )

            for connector in module.get(
                "connectors",
                (),
            ):
                if (
                    str(
                        connector.get(
                            "connector_id",
                            "",
                        )
                    ).upper()
                    == face
                ):
                    return connector

            raise RuntimeError(
                f"{role}.{face} connector unavailable"
            )

        def _ik_normalize(vector):

            vector = _np.asarray(
                vector,
                dtype=float,
            )

            norm = float(
                _np.linalg.norm(vector)
            )

            if norm < 1.0e-12:
                raise RuntimeError(
                    "Zero-length IK vector"
                )

            return vector / norm

        def _ik_quaternion_matrix(q):

            x, y, z, w = (
                float(value)
                for value in q
            )

            norm = math.sqrt(
                x*x + y*y + z*z + w*w
            )

            if norm <= 0.0:
                raise RuntimeError(
                    "Invalid module quaternion"
                )

            x /= norm
            y /= norm
            z /= norm
            w /= norm

            return _np.asarray(
                [
                    [
                        1.0 - 2.0*(y*y + z*z),
                        2.0*(x*y - z*w),
                        2.0*(x*z + y*w),
                    ],
                    [
                        2.0*(x*y + z*w),
                        1.0 - 2.0*(x*x + z*z),
                        2.0*(y*z - x*w),
                    ],
                    [
                        2.0*(x*z - y*w),
                        2.0*(y*z + x*w),
                        1.0 - 2.0*(x*x + y*y),
                    ],
                ],
                dtype=float,
            )

        def _ik_revolute_transform(
            axis,
            pivot,
            angle,
        ):

            axis = _ik_normalize(axis)
            pivot = _np.asarray(
                pivot,
                dtype=float,
            )

            skew = _np.asarray(
                [
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ],
                dtype=float,
            )

            rotation = (
                _np.eye(3)
                + math.sin(angle) * skew
                + (
                    1.0 - math.cos(angle)
                ) * (skew @ skew)
            )

            transform = _np.eye(4)

            transform[:3, :3] = rotation
            transform[:3, 3] = (
                pivot
                - rotation @ pivot
            )

            return transform

        def _ik_joint_value(
            state,
            role,
            joint,
        ):

            module = _ik_module(
                state,
                role,
            )

            return float(
                module["actuators"]
                [joint]["position_rad"]
            )

        def _ik_normal_angle_deg(
            normal,
            target_normal=None,
        ):

            target = (
                _IK_TARGET_NORMAL
                if target_normal is None
                else _ik_normalize(target_normal)
            )

            return vector_angle_deg(
                normal,
                target,
            )


        def _ik_build_reference(state):

            # Reference state is the actual current physical
            # configuration, including any small compliant
            # displacement of the support structure.
            q_ref = _np.asarray(
                [
                    _ik_joint_value(
                        state,
                        role,
                        joint,
                    )
                    for role, joint in _IK_DOF
                ],
                dtype=float,
            )

            axes = []
            pivots = []
            limits = []

            for role, joint in _IK_DOF:

                module = _ik_module(
                    state,
                    role,
                )

                actuator = (
                    module["actuators"][joint]
                )

                if joint == "tilt":

                    body_rotation = (
                        _ik_quaternion_matrix(
                            module["orientation"]
                        )
                    )

                    # Positive ROS TILT raises TOP.
                    # With TOP fixed upstream, BODY/BOTTOM
                    # therefore moves with +body-Y rotation.
                    axis = (
                        body_rotation
                        @ _np.asarray(
                            [0.0, 1.0, 0.0],
                            dtype=float,
                        )
                    )

                    pivot = _np.asarray(
                        module["position"],
                        dtype=float,
                    )

                else:

                    top = _ik_connector(
                        state,
                        role,
                        "TOP",
                    )

                    top_normal = (
                        _ik_normalize(
                            top[
                                "outward_normal_world"
                            ]
                        )
                    )

                    # PAN axis is pan-link local +X.
                    # TOP is the fixed upstream side,
                    # so downstream BODY motion is the
                    # inverse rotation.
                    axis = -top_normal

                    top_center = _np.asarray(
                        top["position_world"],
                        dtype=float,
                    )

                    pivot = (
                        top_center
                        - top_normal
                        * (
                            0.5
                            * _ik_geom
                              .pan_face_thickness_m
                        )
                    )

                axes.append(
                    _ik_normalize(axis)
                )

                pivots.append(
                    _np.asarray(
                        pivot,
                        dtype=float,
                    )
                )

                if joint == "tilt":

                    lower = actuator.get(
                        "lower_limit_rad"
                    )

                    upper = actuator.get(
                        "upper_limit_rad"
                    )

                    limits.append(
                        (
                            None
                            if lower is None
                            else float(lower),

                            None
                            if upper is None
                            else float(upper),
                        )
                    )

                else:

                    # PAN is continuous, but the local IK
                    # is deliberately kept in the nearby
                    # branch to avoid unnecessary winding.
                    limits.append(
                        (
                            q_ref[len(limits)]
                            - math.radians(100.0),

                            q_ref[len(limits)]
                            + math.radians(100.0),
                        )
                    )

            bottom = _ik_connector(
                state,
                "end_effector",
                "BOTTOM",
            )

            bottom_center = _np.asarray(
                bottom["position_world"],
                dtype=float,
            )

            bottom_normal = _ik_normalize(
                bottom[
                    "outward_normal_world"
                ]
            )

            return {
                "q_ref":
                    q_ref,

                "axes":
                    tuple(axes),

                "pivots":
                    tuple(pivots),

                "limits":
                    tuple(limits),

                "bottom_center":
                    bottom_center,

                "bottom_normal":
                    bottom_normal,
            }

        def _ik_fk(reference, q):

            transform = _np.eye(4)

            # Space product of exponentials.
            #
            # Order is the physical TOP -> BOTTOM
            # serial order defined in _IK_DOF.
            for index in range(
                len(_IK_DOF)
            ):

                delta = float(
                    q[index]
                    - reference["q_ref"][index]
                )

                transform = (
                    transform
                    @ _ik_revolute_transform(
                        reference["axes"][index],
                        reference["pivots"][index],
                        delta,
                    )
                )

            rotation = transform[:3, :3]

            center = (
                rotation
                @ reference[
                    "bottom_center"
                ]
                + transform[:3, 3]
            )

            normal = _ik_normalize(
                rotation
                @ reference[
                    "bottom_normal"
                ]
            )

            return center, normal

        def _ik_region_position_residual(
            center,
            target_center,
        ):
            """Residual to the admissible finite button-face region.

            T and Z are surface-overlap inequality constraints.
            N is the PRE/PRESS depth equality.
            """

            center = _np.asarray(
                center,
                dtype=float,
            )

            target_center = _np.asarray(
                target_center,
                dtype=float,
            )

            tangent = float(
                center
                @ _IK_FACE_TANGENT
            )

            normal_coordinate = float(
                center
                @ _IK_TARGET_NORMAL
            )

            z = float(
                center[2]
            )

            contact_t_min = (
                button_t_min
                - _IK_EE_CONTACT_HALF_TANGENT_M
            )

            contact_t_max = (
                button_t_max
                + _IK_EE_CONTACT_HALF_TANGENT_M
            )

            contact_z_min = (
                button_z_min
                - _IK_EE_CONTACT_HALF_Z_M
            )

            contact_z_max = (
                button_z_max
                + _IK_EE_CONTACT_HALF_Z_M
            )

            if tangent < contact_t_min:
                error_t = (
                    tangent
                    - contact_t_min
                )

            elif tangent > contact_t_max:
                error_t = (
                    tangent
                    - contact_t_max
                )

            else:
                error_t = 0.0

            if z < contact_z_min:
                error_z = (
                    z
                    - contact_z_min
                )

            elif z > contact_z_max:
                error_z = (
                    z
                    - contact_z_max
                )

            else:
                error_z = 0.0

            target_normal_coordinate = float(
                target_center
                @ _IK_TARGET_NORMAL
            )

            error_n = (
                normal_coordinate
                - target_normal_coordinate
            )

            return _np.asarray(
                [
                    error_t,
                    error_n,
                    error_z,
                ],
                dtype=float,
            )


        def _ik_region_position_error(
            center,
            target_center,
        ):
            return float(
                _np.linalg.norm(
                    _ik_region_position_residual(
                        center,
                        target_center,
                    )
                )
            )


        def _ik_residual(
            reference,
            q,
            target_center,
            target_normal=None,
        ):

            center, normal = _ik_fk(
                reference,
                q,
            )

            error = (
                _ik_region_position_residual(
                    center,
                    target_center,
                )
            )

            target = (
                _IK_TARGET_NORMAL
                if target_normal is None
                else _ik_normalize(target_normal)
            )

            # Six residual components for five DoFs are intentional:
            # three finite-region position constraints plus the full
            # 3-D normal-vector difference.  Unlike the former
            # tangent/Z-only orientation residual, this has a unique
            # zero at +target_normal and therefore does not confuse
            # the desired and antiparallel normal branches.
            orientation = orientation_residual(
                normal,
                target,
                lever_m=_IK_ORIENTATION_LEVER_M,
            )

            return _np.concatenate(
                (error, orientation)
            ).astype(float)

        def _ik_jacobian(
            reference,
            q,
            target_center,
            target_normal=None,
        ):

            step = math.radians(0.25)

            residual_size = int(
                _ik_residual(
                    reference,
                    q,
                    target_center,
                    target_normal,
                ).shape[0]
            )

            jacobian = _np.zeros(
                (residual_size, 5),
                dtype=float,
            )

            for index in range(5):

                plus = q.copy()
                minus = q.copy()

                plus[index] += step
                minus[index] -= step

                jacobian[:, index] = (
                    _ik_residual(
                        reference,
                        plus,
                        target_center,
                        target_normal,
                    )
                    - _ik_residual(
                        reference,
                        minus,
                        target_center,
                        target_normal,
                    )
                ) / (2.0 * step)

            return jacobian

        def _ik_solve(
            state,
            target_center,
            label,
            target_normal=None,
        ):

            target = (
                _IK_TARGET_NORMAL
                if target_normal is None
                else _ik_normalize(target_normal)
            )

            reference = _ik_build_reference(
                state
            )

            q = reference["q_ref"].copy()

            damping = 0.010
            max_step = math.radians(7.0)

            best_q = q.copy()
            best_cost = float("inf")
            initial_cost = None

            first_rank = None
            first_condition = None

            for iteration in range(120):

                residual = _ik_residual(
                    reference,
                    q,
                    target_center,
                    target,
                )

                cost = float(
                    residual @ residual
                )

                if initial_cost is None:
                    initial_cost = cost

                if cost < best_cost:
                    best_cost = cost
                    best_q = q.copy()

                center, normal = _ik_fk(
                    reference,
                    q,
                )

                position_error = float(
                    _ik_region_position_error(
                        center,
                        target_center,
                    )
                )

                normal_error = (
                    _ik_normal_angle_deg(
                        normal,
                        target,
                    )
                )

                if (
                    position_error
                    <= _IK_POSITION_TOL_M
                    and normal_error
                    <= _IK_NORMAL_TOL_DEG
                ):
                    return {
                        "success": True,
                        "label": label,
                        "iterations":
                            iteration + 1,
                        "initial_cost":
                            float(initial_cost),
                        "best_cost":
                            float(cost),
                        "q":
                            q.copy(),
                        "reference":
                            reference,
                        "center":
                            center,
                        "normal":
                            normal,
                        "target_normal":
                            target.copy(),
                        "position_error_m":
                            position_error,
                        "normal_error_deg":
                            normal_error,
                        "jacobian_rank":
                            first_rank,
                        "jacobian_condition":
                            first_condition,
                    }

                jacobian = _ik_jacobian(
                    reference,
                    q,
                    target_center,
                    target,
                )

                if first_rank is None:

                    first_rank = int(
                        _np.linalg.matrix_rank(
                            jacobian
                        )
                    )

                    first_condition = float(
                        _np.linalg.cond(
                            jacobian
                        )
                    )

                system = (
                    jacobian.T @ jacobian
                    + (
                        damping * damping
                    ) * _np.eye(5)
                )

                gradient = (
                    jacobian.T @ residual
                )

                try:
                    delta = -_np.linalg.solve(
                        system,
                        gradient,
                    )
                except _np.linalg.LinAlgError:
                    break

                delta = _np.clip(
                    delta,
                    -max_step,
                    +max_step,
                )

                accepted = False

                for alpha in (
                    1.0,
                    0.5,
                    0.25,
                    0.125,
                    0.0625,
                ):

                    trial = (
                        q
                        + alpha * delta
                    )

                    for index, (
                        lower,
                        upper,
                    ) in enumerate(
                        reference["limits"]
                    ):

                        if lower is not None:
                            trial[index] = max(
                                trial[index],
                                lower,
                            )

                        if upper is not None:
                            trial[index] = min(
                                trial[index],
                                upper,
                            )

                    if (
                        str(label)
                        .strip()
                        .upper()
                        .startswith("PRESS")
                    ):
                        for pan_index in (1, 3):
                            pan_reference = float(
                                reference["q_ref"][
                                    pan_index
                                ]
                            )
                            trial[pan_index] = float(
                                _np.clip(
                                    trial[pan_index],
                                    pan_reference
                                    - _IK_PRESS_PAN_DELTA_LIMIT_RAD,
                                    pan_reference
                                    + _IK_PRESS_PAN_DELTA_LIMIT_RAD,
                                )
                            )

                    trial_residual = (
                        _ik_residual(
                            reference,
                            trial,
                            target_center,
                            target,
                        )
                    )

                    trial_cost = float(
                        trial_residual
                        @ trial_residual
                    )

                    if trial_cost < cost:

                        q = trial
                        accepted = True
                        damping = max(
                            0.004,
                            damping * 0.75,
                        )
                        break

                if not accepted:

                    damping = min(
                        0.20,
                        damping * 2.0,
                    )

            center, normal = _ik_fk(
                reference,
                best_q,
            )

            return {
                "success": False,
                "label": label,
                "iterations": 120,
                "initial_cost": float(
                    initial_cost
                    if initial_cost is not None
                    else best_cost
                ),
                "best_cost": float(best_cost),
                "q": best_q,
                "reference": reference,
                "center": center,
                "normal": normal,
                "target_normal": target.copy(),
                "position_error_m":
                    float(
                        _ik_region_position_error(
                            center,
                            target_center,
                        )
                    ),
                "normal_error_deg":
                    _ik_normal_angle_deg(
                        normal,
                        target,
                    ),
                "jacobian_rank":
                    first_rank,
                "jacobian_condition":
                    first_condition,
            }

        def _ik_plan_payload(
            solution,
            target,
        ):

            return {
                "success":
                    bool(
                        solution["success"]
                    ),

                "iterations":
                    solution[
                        "iterations"
                    ],

                "target_center_xyz_m":
                    [
                        float(value)
                        for value in target
                    ],

                "q_rad": {
                    (
                        f"{role}.{joint}"
                    ):
                        float(
                            solution["q"][index]
                        )
                    for index, (
                        role,
                        joint,
                    ) in enumerate(
                        _IK_DOF
                    )
                },

                "bottom_center_xyz_m":
                    [
                        float(value)
                        for value in
                        solution["center"]
                    ],

                "bottom_normal":
                    [
                        float(value)
                        for value in
                        solution["normal"]
                    ],

                "position_error_m":
                    float(
                        solution[
                            "position_error_m"
                        ]
                    ),

                "normal_error_deg":
                    float(
                        solution[
                            "normal_error_deg"
                        ]
                    ),

                "jacobian_rank":
                    solution[
                        "jacobian_rank"
                    ],

                "jacobian_condition":
                    solution[
                        "jacobian_condition"
                    ],
            }

        def _ik_print_solution(
            solution,
            target,
        ):

            print()
            print(
                "------------------------------------------------------------"
            )
            print(
                f"IK {solution['label']}"
            )
            print(
                "------------------------------------------------------------"
            )

            reference_q = (
                solution[
                    "reference"
                ][
                    "q_ref"
                ]
            )

            for index, (
                role,
                joint,
            ) in enumerate(_IK_DOF):

                print(
                    f"{role}.{joint:<4} "
                    f"{math.degrees(reference_q[index]):+7.2f}°"
                    f" -> "
                    f"{math.degrees(solution['q'][index]):+7.2f}°"
                )

            print(
                f"target center = "
                f"({target[0]:+.5f},"
                f"{target[1]:+.5f},"
                f"{target[2]:+.5f})"
            )

            center = solution["center"]
            normal = solution["normal"]

            print(
                f"BOTTOM center = "
                f"({center[0]:+.5f},"
                f"{center[1]:+.5f},"
                f"{center[2]:+.5f})"
            )

            print(
                f"BOTTOM normal = "
                f"[{normal[0]:+.4f},"
                f"{normal[1]:+.4f},"
                f"{normal[2]:+.4f}]"
            )

            print(
                f"position err  = "
                f"{solution['position_error_m']*1000:.2f} mm"
            )

            print(
                f"normal err    = "
                f"{solution['normal_error_deg']:.2f}°"
            )

            print(
                f"J rank/cond   = "
                f"{solution['jacobian_rank']}/"
                f"{solution['jacobian_condition']}"
            )

            print(
                f"CONVERGED     = "
                f"{solution['success']}"
            )

        # --------------------------------------------------------
        # Absolute joint-target execution.
        # No +/- probe is ever sent to Isaac.
        # --------------------------------------------------------

        _ik_node = rclpy.create_node(
            "mssr_button_bottom_face_ik"
        )

        _ik_goal_pub = (
            _ik_node.create_publisher(
                _IKString,
                "/mssr/primitives/goal",
                10,
            )
        )

        _discover_deadline = (
            time.monotonic() + 3.0
        )

        while (
            _ik_goal_pub
            .get_subscription_count()
            == 0
            and time.monotonic()
            < _discover_deadline
        ):
            rclpy.spin_once(
                _ik_node,
                timeout_sec=0.1,
            )

        def _ik_joint_error(
            actual,
            target,
            joint,
        ):

            if joint == "pan":

                return math.atan2(
                    math.sin(
                        actual - target
                    ),
                    math.cos(
                        actual - target
                    ),
                )

            return actual - target

        def _ik_command_configuration(
            label,
            q_target,
            dataset_context=None,
            only_dofs=None,
        ):
            """Execute one configuration while logging each physical primitive."""

            selected_dofs = (
                None
                if only_dofs is None
                else {
                    (str(role), str(joint))
                    for role, joint in only_dofs
                }
            )
            dataset_context = dict(dataset_context or {})

            selected = [
                (index, role, joint)
                for index, (role, joint) in enumerate(_IK_DOF)
                if selected_dofs is None
                or (role, joint) in selected_dofs
            ]
            if not selected:
                raise RuntimeError(
                    f"{label}: configuration selected no executable DoFs"
                )

            primitive_count = len(selected)
            carried_state = None
            carried_graph = None
            final_primitive = None

            print()
            print(f"COMMAND ABSOLUTE CONFIGURATION: {label}")

            for primitive_index, (index, role, joint) in enumerate(
                selected,
                start=1,
            ):
                target = float(q_target[index])
                accepted_state_before = None
                accepted_graph_before = None
                accepted_goal = None
                status = None

                for attempt in range(20):
                    if attempt == 0 and carried_state is not None:
                        state_before = carried_state
                        graph_before = carried_graph
                    else:
                        state_before = _ik_read_state()
                        graph_before = _ik_read_robot_graph()

                    if joint == "pan":
                        pan_module = _ik_module(state_before, role)
                        pan_actual = float(
                            pan_module["actuators"]["pan"]["position_rad"]
                        )
                        pan_delta = shortest_angular_delta(
                            pan_actual,
                            target,
                        )
                        primitive_name = "rotate_pan_by"
                        primitive_parameters = {
                            "delta_rad": pan_delta,
                            "periodic_equivalent": True,
                            "tolerance_rad": math.radians(0.6),
                            "max_servo_speed_rad_s": 0.12,
                        }
                    else:
                        primitive_name = "set_tilt"
                        primitive_parameters = {
                            "angle_rad": target,
                            "tolerance_rad": math.radians(0.6),
                        }

                    request = _IKPrimitiveGoalRequest(
                        goal_id=(
                            f"button-ik-{label}-{role}-{joint}-"
                            f"{time.time_ns()}"
                        ),
                        primitive=primitive_name,
                        module_ids=(role_to_module[role],),
                        parameters=primitive_parameters,
                        timeout_s=20.0,
                    )

                    message = _IKString()
                    message.data = json.dumps(request.to_dict())
                    _ik_goal_pub.publish(message)

                    admission_deadline = time.monotonic() + 2.0
                    status = None
                    while time.monotonic() < admission_deadline:
                        rclpy.spin_once(_ik_node, timeout_sec=0.05)
                        status = _ik_goal_status(request.goal_id)
                        if status is not None:
                            break

                    if (
                        status is not None
                        and str(status.get("state", "")).lower() == "rejected"
                    ):
                        print(
                            f"  {role}.{joint}: goal rejected "
                            f"({status.get('message', '')}); retrying"
                        )
                        carried_state = None
                        carried_graph = None
                        time.sleep(0.2)
                        continue

                    accepted_state_before = state_before
                    accepted_graph_before = graph_before
                    accepted_goal = request.to_dict()
                    break
                else:
                    raise RuntimeError(
                        f"{label}: {role}.{joint} goal was repeatedly rejected"
                    )

                joint_deadline = time.monotonic() + 20.0
                while time.monotonic() < joint_deadline:
                    rclpy.spin_once(_ik_node, timeout_sec=0.05)
                    joint_state = _ik_read_state()
                    joint_module = _ik_module(joint_state, role)
                    joint_actuator = joint_module["actuators"][joint]
                    joint_actual = float(joint_actuator["position_rad"])
                    joint_velocity = abs(
                        float(joint_actuator.get("velocity_rad_s", 0.0))
                    )
                    joint_error = abs(
                        _ik_joint_error(
                            joint_actual,
                            target,
                            joint,
                        )
                    )
                    if joint_error <= math.radians(1.2):
                        print(
                            f"  {role}.{joint}: target reached "
                            f"(error={math.degrees(joint_error):.2f}°, "
                            f"velocity={joint_velocity:.3f} rad/s)"
                        )
                        break
                    time.sleep(0.05)
                else:
                    raise RuntimeError(
                        f"{label}: {role}.{joint} did not reach its target"
                    )

                _ik_wait_goal_terminal(
                    request.goal_id,
                    label,
                    role,
                    joint,
                    timeout_s=20.0,
                )

                # The graph file may lag behind primitive status. Anchor
                # after terminal success, then require a subsequently
                # published physical graph before supervising this action.
                terminal_anchor_graph = _ik_read_robot_graph()

                final_primitive = (
                    primitive_index,
                    accepted_goal,
                    accepted_state_before,
                    accepted_graph_before,
                )

                if primitive_index < primitive_count:
                    state_after, graph_after = _ik_read_transition_endpoint(
                        terminal_anchor_graph
                    )
                    _ik_record_primitive_transition(
                        label=str(label),
                        q_target=q_target,
                        primitive_goal=accepted_goal,
                        primitive_index=primitive_index,
                        primitive_count=primitive_count,
                        state_before=accepted_state_before,
                        graph_before=accepted_graph_before,
                        state_after=state_after,
                        graph_after=graph_after,
                        context=dataset_context,
                    )
                    # No independent sample is inserted before the next
                    # immediate dispatch: exact physical continuity is kept.
                    carried_state = state_after
                    carried_graph = graph_after

            # The last physical primitive owns the configuration-level settle
            # interval.  Its G_t+1 is therefore the stable state used by the
            # next PRE/CLIK decision.
            deadline = time.monotonic() + 20.0
            stable_since = None
            while time.monotonic() < deadline:
                rclpy.spin_once(_ik_node, timeout_sec=0.05)
                state = _ik_read_state()
                errors = []
                velocities = []

                for index, role, joint in selected:
                    module = _ik_module(state, role)
                    actuator = module["actuators"][joint]
                    actual = float(actuator["position_rad"])
                    velocity = abs(
                        float(actuator.get("velocity_rad_s", 0.0))
                    )
                    errors.append(
                        abs(_ik_joint_error(actual, float(q_target[index]), joint))
                    )
                    velocities.append(velocity)

                within = max(errors) <= math.radians(1.2)
                now = time.monotonic()
                if within:
                    if stable_since is None:
                        stable_since = now
                    elif now - stable_since >= 0.6:
                        print(
                            f"{label}: reached "
                            f"(max error {math.degrees(max(errors)):.2f}°, "
                            f"max velocity {max(velocities):.3f} rad/s)"
                        )

                        # Stable is a controller condition observed
                        # now. Require one graph publication after that
                        # condition before closing the final transition.
                        settle_anchor_graph = _ik_read_robot_graph()
                        state_after, graph_after = _ik_read_transition_endpoint(
                            settle_anchor_graph
                        )
                        (
                            primitive_index,
                            primitive_goal,
                            state_before,
                            graph_before,
                        ) = final_primitive

                        _ik_record_primitive_transition(
                            label=str(label),
                            q_target=q_target,
                            primitive_goal=primitive_goal,
                            primitive_index=primitive_index,
                            primitive_count=primitive_count,
                            state_before=state_before,
                            graph_before=graph_before,
                            state_after=state_after,
                            graph_after=graph_after,
                            context=dataset_context,
                        )
                        return
                else:
                    stable_since = None

                time.sleep(0.05)

            raise RuntimeError(
                f"{label} absolute configuration did not settle"
            )

        def _ik_plan_end_effector_tilt_preorientation(
            state,
        ):
            """Virtual scan of end-effector TILT only."""

            reference = _ik_build_reference(state)

            tilt_dof = ("end_effector", "tilt")
            tilt_index = _IK_DOF.index(
                tilt_dof
            )

            q_ref = _np.asarray(
                reference["q_ref"],
                dtype=float,
            ).copy()

            current_tilt = float(
                q_ref[tilt_index]
            )

            lower = -0.5 * math.pi
            upper = +0.5 * math.pi

            current_tilt = float(
                _np.clip(
                    current_tilt,
                    lower,
                    upper,
                )
            )

            def score(candidate):
                q_trial = q_ref.copy()

                q_trial[tilt_index] = float(
                    candidate
                )

                _, normal_trial = _ik_fk(
                    reference,
                    q_trial,
                )

                return float(
                    _ik_normal_angle_deg(
                        _ik_normalize(
                            _np.asarray(
                                normal_trial,
                                dtype=float,
                            )
                        ),
                        _IK_TARGET_NORMAL,
                    )
                )

            current_error = float(
                score(current_tilt)
            )

            step = math.radians(
                _PREORIENT_TILT_SCAN_STEP_DEG
            )

            candidates = {
                float(lower),
                float(upper),
                float(current_tilt),
            }

            value = float(lower)

            while value <= upper + 1.0e-12:
                candidates.add(
                    float(
                        _np.clip(
                            value,
                            lower,
                            upper,
                        )
                    )
                )
                value += step

            scored = sorted(
                (
                    float(score(candidate)),
                    abs(
                        float(candidate)
                        - current_tilt
                    ),
                    float(candidate),
                )
                for candidate in candidates
            )

            (
                best_error,
                _,
                best_tilt,
            ) = scored[0]

            improvement = float(
                current_error - best_error
            )

            movement_deg = abs(
                math.degrees(
                    best_tilt
                    - current_tilt
                )
            )

            should_move = bool(
                current_error
                > _IK_NORMAL_DEADBAND_DEG
                and improvement
                >= _PREORIENT_MIN_IMPROVEMENT_DEG
                and movement_deg
                >= _PREORIENT_MIN_MOVE_DEG
            )

            q_target = q_ref.copy()
            q_target[tilt_index] = float(
                best_tilt
            )

            return {
                "should_move":
                    should_move,
                "current_tilt_rad":
                    float(current_tilt),
                "target_tilt_rad":
                    float(best_tilt),
                "movement_deg":
                    float(movement_deg),
                "current_normal_error_deg":
                    float(current_error),
                "predicted_normal_error_deg":
                    float(best_error),
                "improvement_deg":
                    float(improvement),
                "q_target":
                    q_target,
            }

        def _ik_clik_pre(
            pre_target,
        ):
            """Reach PRE through live task-space waypoints.

            Every control step starts from the physical state read back
            from Isaac, builds a bounded Cartesian/normal waypoint, solves
            a local 5-DoF IK problem, executes only a bounded joint step,
            then re-reads and re-linearizes.  No stale post-reconfiguration
            PAN reference is used.
            """

            samples = []
            best_metric = float("inf")
            stall_steps = 0
            last_state = _ik_read_state()

            print()
            print(
                "------------------------------------------------------------"
            )
            print(
                "CLOSED-LOOP PRE TRAJECTORY"
            )
            print(
                "------------------------------------------------------------"
            )
            print(
                f"position waypoint max = "
                f"{_PRE_POSITION_STEP_M*1000:.1f} mm"
            )
            print(
                f"normal waypoint max   = "
                f"{math.degrees(_PRE_NORMAL_STEP_RAD):.1f} deg"
            )
            print(
                f"joint step max        = "
                f"{math.degrees(_PRE_JOINT_STEP_MAX_RAD):.1f} deg"
            )

            for control_step in range(
                _PRE_MAX_CONTROL_STEPS
            ):
                state = _ik_read_state()
                last_state = state

                bottom = _ik_connector(
                    state,
                    "end_effector",
                    "BOTTOM",
                )
                center = _np.asarray(
                    bottom["position_world"],
                    dtype=float,
                )
                normal = _ik_normalize(
                    _np.asarray(
                        bottom["outward_normal_world"],
                        dtype=float,
                    )
                )

                final_position_error = float(
                    _ik_region_position_error(
                        center,
                        pre_target,
                    )
                )
                final_normal_error = float(
                    _ik_normal_angle_deg(
                        normal,
                        _IK_TARGET_NORMAL,
                    )
                )

                if (
                    final_position_error
                    <= _IK_POSITION_TOL_M
                    and final_normal_error
                    <= _IK_NORMAL_DEADBAND_DEG
                ):
                    return {
                        "success": True,
                        "reason": "physical PRE reached",
                        "state": state,
                        "samples": samples,
                    }

                desired_center = bounded_position_waypoint(
                    center,
                    pre_target,
                    max_step_m=_PRE_POSITION_STEP_M,
                )
                orientation_in_deadband = (
                    final_normal_error
                    <= _IK_NORMAL_DEADBAND_DEG
                )

                if orientation_in_deadband:
                    # Task-space cone: once the BOTTOM face points
                    # sufficiently toward the button, perfect
                    # parallelism is unnecessary.  Preserve the
                    # admissible live normal instead of rewarding
                    # further alignment toward zero angular error.
                    desired_normal = normal.copy()
                else:
                    desired_normal = bounded_normal_waypoint(
                        normal,
                        _IK_TARGET_NORMAL,
                        max_angle_rad=_PRE_NORMAL_STEP_RAD,
                    )

                solution = _ik_solve(
                    state,
                    desired_center,
                    f"PRE-WP-{control_step:02d}",
                    target_normal=desired_normal,
                )

                # At a rank-deficient pose a local solve may not satisfy
                # the waypoint tolerance exactly in one mathematical pass.
                # It is still useful if its best residual is strictly lower
                # than the residual at the live starting state; execute only
                # a bounded physical step and re-linearize afterwards.
                solver_progress = (
                    math.isfinite(float(solution["best_cost"]))
                    and float(solution["best_cost"])
                    < float(solution["initial_cost"])
                    - 1.0e-12
                )

                if (
                    not solution["success"]
                    and not solver_progress
                ):
                    return {
                        "success": False,
                        "reason": (
                            "local PRE IK made no residual progress "
                            f"at control step {control_step}"
                        ),
                        "state": state,
                        "samples": samples,
                    }

                q_actual = _np.asarray(
                    solution["reference"]["q_ref"],
                    dtype=float,
                ).copy()
                q_candidate = _np.asarray(
                    solution["q"],
                    dtype=float,
                ).copy()
                q_step = q_actual.copy()

                for index, (role, joint) in enumerate(_IK_DOF):
                    if joint == "pan":
                        delta = shortest_angular_delta(
                            float(q_actual[index]),
                            float(q_candidate[index]),
                        )
                    else:
                        delta = float(
                            q_candidate[index]
                            - q_actual[index]
                        )

                    delta = float(
                        _np.clip(
                            delta,
                            -_PRE_JOINT_STEP_MAX_RAD,
                            +_PRE_JOINT_STEP_MAX_RAD,
                        )
                    )
                    q_step[index] = (
                        float(q_actual[index])
                        + delta
                    )

                    lower, upper = solution["reference"]["limits"][index]
                    if lower is not None:
                        q_step[index] = max(
                            float(q_step[index]),
                            float(lower),
                        )
                    if upper is not None:
                        q_step[index] = min(
                            float(q_step[index]),
                            float(upper),
                        )

                max_delta_deg = float(
                    _np.max(
                        _np.abs(
                            _np.degrees(q_step - q_actual)
                        )
                    )
                )

                print(
                    f"PRE {control_step:02d} "
                    f"pos={final_position_error*1000:.1f}mm "
                    f"normal={final_normal_error:.1f}deg "
                    f"wp_normal={vector_angle_deg(desired_normal, _IK_TARGET_NORMAL):.1f}deg "
                    f"rank={solution['jacobian_rank']} "
                    f"dqmax={max_delta_deg:.2f}deg"
                )

                _ik_command_configuration(
                    f"pre-{control_step:02d}",
                    q_step,
                    dataset_context={
                        "control_step": int(control_step),
                        "controller": "closed_loop_pre_waypoint",
                        "desired_center_xyz_m": [
                            float(value)
                            for value in desired_center
                        ],
                        "desired_normal_world": [
                            float(value)
                            for value in desired_normal
                        ],
                        "final_position_error_before_m":
                            final_position_error,
                        "final_normal_error_before_deg":
                            final_normal_error,
                        "jacobian_rank":
                            solution["jacobian_rank"],
                        "jacobian_condition":
                            solution["jacobian_condition"],
                        "max_delta_deg": max_delta_deg,
                    },
                )

                time.sleep(
                    _PRE_POST_COMMAND_SETTLE_S
                )

                state_after = _ik_read_state()
                last_state = state_after
                bottom_after = _ik_connector(
                    state_after,
                    "end_effector",
                    "BOTTOM",
                )
                center_after = _np.asarray(
                    bottom_after["position_world"],
                    dtype=float,
                )
                normal_after = _ik_normalize(
                    _np.asarray(
                        bottom_after["outward_normal_world"],
                        dtype=float,
                    )
                )
                position_after = float(
                    _ik_region_position_error(
                        center_after,
                        pre_target,
                    )
                )
                normal_after_deg = float(
                    _ik_normal_angle_deg(
                        normal_after,
                        _IK_TARGET_NORMAL,
                    )
                )
                normal_excess_deg = max(
                    0.0,
                    normal_after_deg
                    - _IK_NORMAL_DEADBAND_DEG,
                )

                metric = (
                    position_after
                    + _IK_ORIENTATION_LEVER_M
                    * math.radians(normal_excess_deg)
                )

                samples.append({
                    "control_step": int(control_step),
                    "center_xyz_m": [
                        float(value)
                        for value in center_after
                    ],
                    "normal_world": [
                        float(value)
                        for value in normal_after
                    ],
                    "position_error_m": position_after,
                    "normal_error_deg": normal_after_deg,
                    "normal_excess_deg": normal_excess_deg,
                    "orientation_in_deadband": bool(
                        normal_after_deg
                        <= _IK_NORMAL_DEADBAND_DEG
                    ),
                    "max_delta_deg": max_delta_deg,
                    "solver_success": bool(solution["success"]),
                    "solver_initial_cost":
                        float(solution["initial_cost"]),
                    "solver_best_cost":
                        float(solution["best_cost"]),
                })

                if metric < best_metric - _PRE_PROGRESS_EPS:
                    best_metric = metric
                    stall_steps = 0
                else:
                    stall_steps += 1

                if stall_steps >= _PRE_STALL_STEPS:
                    return {
                        "success": False,
                        "reason": "closed-loop PRE progress stalled",
                        "state": state_after,
                        "samples": samples,
                    }

            return {
                "success": False,
                "reason": "closed-loop PRE maximum control steps reached",
                "state": last_state,
                "samples": samples,
            }

        def _ik_clik_press(
            pre_state,
            q_pre_actual,
            press_target,
        ):
            """Closed-loop Cartesian press along button normal N."""

            q_pre_actual = _np.asarray(
                q_pre_actual,
                dtype=float,
            ).copy()

            initial_bottom = _ik_connector(
                pre_state,
                "end_effector",
                "BOTTOM",
            )

            initial_center = _np.asarray(
                initial_bottom[
                    "position_world"
                ],
                dtype=float,
            )

            start_s = float(
                initial_center
                @ _IK_TARGET_NORMAL
            )

            goal_s = float(
                _np.asarray(
                    press_target,
                    dtype=float,
                )
                @ _IK_TARGET_NORMAL
            )

            if goal_s <= start_s:
                return {
                    "success": False,
                    "reason":
                        "press target is not ahead of PRE "
                        "along button normal N",
                    "depression_m":
                        0.0,
                    "state":
                        pre_state,
                    "samples":
                        [],
                }

            # 2-mm Cartesian references along N.
            path_s = []

            s = (
                start_s
                + _CLIK_CARTESIAN_STEP_M
            )

            while s < goal_s:
                path_s.append(
                    float(s)
                )

                s += (
                    _CLIK_CARTESIAN_STEP_M
                )

            path_s.append(
                float(goal_s)
            )

            path_index = 0
            samples = []

            best_s = start_s
            stall_steps = 0

            last_state = pre_state
            last_depression = 0.0

            print()
            print(
                "------------------------------------------------------------"
            )
            print(
                "CLIK PRESS TRAJECTORY"
            )
            print(
                "------------------------------------------------------------"
            )
            print(
                f"physical PRE N = "
                f"{start_s:+.5f} m"
            )
            print(
                f"PRESS goal N   = "
                f"{goal_s:+.5f} m"
            )
            print(
                f"path distance   = "
                f"{(goal_s-start_s)*1000:.2f} mm"
            )
            print(
                f"trajectory refs = "
                f"{len(path_s)}"
            )
            print(
                "controller      = "
                "resolved-rate DLS + nullspace + physical feedback"
            )

            for control_step in range(
                _CLIK_MAX_CONTROL_STEPS
            ):

                state = _ik_read_state()
                last_state = state

                button_live = selected_button(state)

                depression = float(
                    button_live.get(
                        "depression_m",
                        0.0,
                    )
                )

                last_depression = (
                    depression
                )

                bottom_live = _ik_connector(
                    state,
                    "end_effector",
                    "BOTTOM",
                )

                center_live = _np.asarray(
                    bottom_live[
                        "position_world"
                    ],
                    dtype=float,
                )

                normal_live = _ik_normalize(
                    _np.asarray(
                        bottom_live[
                            "outward_normal_world"
                        ],
                        dtype=float,
                    )
                )

                if (
                    depression
                    >= _IK_SUCCESS_DEPRESSION_M
                ):
                    print(
                        f"CLIK physical success at step "
                        f"{control_step}: "
                        f"depression="
                        f"{depression*1000:.3f} mm"
                    )

                    return {
                        "success": True,
                        "reason":
                            "physical button depression reached",
                        "depression_m":
                            depression,
                        "state":
                            state,
                        "samples":
                            samples,
                    }

                if float(
                    normal_live
                    @ _IK_TARGET_NORMAL
                ) <= 0.0:
                    return {
                        "success": False,
                        "reason":
                            "BOTTOM normal left target hemisphere",
                        "depression_m":
                            depression,
                        "state":
                            state,
                        "samples":
                            samples,
                    }

                current_s = float(
                    center_live
                    @ _IK_TARGET_NORMAL
                )

                while (
                    path_index
                    < len(path_s) - 1
                    and current_s
                    >= (
                        float(
                            path_s[
                                path_index
                            ]
                        )
                        - _CLIK_WAYPOINT_TOL_M
                    )
                ):
                    path_index += 1

                desired_s = float(
                    path_s[
                        path_index
                    ]
                )

                # Keep target tangent/Z coordinates and change
                # only its coordinate along N.
                desired = _np.asarray(
                    press_target,
                    dtype=float,
                ).copy()

                desired += (
                    desired_s
                    - float(
                        desired
                        @ _IK_TARGET_NORMAL
                    )
                ) * _IK_TARGET_NORMAL

                # Re-linearize about REAL current state.
                reference = (
                    _ik_build_reference(
                        state
                    )
                )

                q = (
                    reference[
                        "q_ref"
                    ].copy()
                )

                residual = _ik_residual(
                    reference,
                    q,
                    desired,
                )

                jacobian = _ik_jacobian(
                    reference,
                    q,
                    desired,
                )

                rank = int(
                    _np.linalg.matrix_rank(
                        jacobian
                    )
                )

                condition_raw = float(
                    _np.linalg.cond(
                        jacobian
                    )
                )

                condition = (
                    condition_raw
                    if math.isfinite(
                        condition_raw
                    )
                    else None
                )

                task_rate = (
                    -_CLIK_KP_S_INV
                    * residual
                )

                contact_started = (
                    depression
                    > 0.00010
                )

                forward_speed = (
                    _CLIK_CONTACT_SPEED_M_S
                    if contact_started
                    else _CLIK_FREE_SPEED_M_S
                )

                # Residual component 1 is coordinate N.
                if current_s < desired_s:
                    task_rate[1] += (
                        forward_speed
                    )

                task_system = (
                    jacobian
                    @ jacobian.T
                    + (
                        _CLIK_DAMPING
                        * _CLIK_DAMPING
                    )
                    * _np.eye(
                        jacobian.shape[0]
                    )
                )

                try:
                    j_hash = (
                        jacobian.T
                        @ _np.linalg.solve(
                            task_system,
                            _np.eye(
                                jacobian.shape[0]
                            ),
                        )
                    )

                except _np.linalg.LinAlgError:
                    return {
                        "success": False,
                        "reason":
                            "DLS task system solve failed",
                        "depression_m":
                            depression,
                        "state":
                            state,
                        "samples":
                            samples,
                    }

                qdot_primary = (
                    j_hash
                    @ task_rate
                )

                qdot_null = (
                    -_CLIK_NULL_PRE_GAIN_S_INV
                    * (
                        q
                        - q_pre_actual
                    )
                )

                for index, (
                    lower,
                    upper,
                ) in enumerate(
                    reference[
                        "limits"
                    ]
                ):

                    if (
                        lower is None
                        or upper is None
                    ):
                        continue

                    lower = float(lower)
                    upper = float(upper)

                    half_range = (
                        0.5
                        * (
                            upper
                            - lower
                        )
                    )

                    if half_range <= 1.0e-6:
                        continue

                    midpoint = (
                        0.5
                        * (
                            lower
                            + upper
                        )
                    )

                    qdot_null[index] += (
                        _CLIK_LIMIT_GAIN_S_INV
                        * (
                            midpoint
                            - float(q[index])
                        )
                        / half_range
                    )

                null_projector = (
                    _np.eye(
                        len(_IK_DOF)
                    )
                    - j_hash
                    @ jacobian
                )

                qdot = (
                    qdot_primary
                    + null_projector
                    @ qdot_null
                )

                qdot = _np.clip(
                    qdot,
                    -_CLIK_QDOT_MAX_RAD_S,
                    +_CLIK_QDOT_MAX_RAD_S,
                )

                delta_q = (
                    qdot
                    * _CLIK_INTEGRATION_DT_S
                )

                q_next = None

                for scale in (
                    1.0,
                    0.5,
                    0.25,
                    0.125,
                ):

                    trial = (
                        q
                        + scale
                        * delta_q
                    )

                    for index, (
                        lower,
                        upper,
                    ) in enumerate(
                        reference[
                            "limits"
                        ]
                    ):

                        if lower is not None:
                            trial[index] = max(
                                float(
                                    trial[index]
                                ),
                                float(lower),
                            )

                        if upper is not None:
                            trial[index] = min(
                                float(
                                    trial[index]
                                ),
                                float(upper),
                            )

                    (
                        _,
                        trial_normal,
                    ) = _ik_fk(
                        reference,
                        trial,
                    )

                    if float(
                        trial_normal
                        @ _IK_TARGET_NORMAL
                    ) <= 0.0:
                        continue

                    q_next = trial
                    break

                if q_next is None:
                    return {
                        "success": False,
                        "reason":
                            "no safe CLIK integration step",
                        "depression_m":
                            depression,
                        "state":
                            state,
                        "samples":
                            samples,
                    }

                max_delta_deg = float(
                    _np.max(
                        _np.abs(
                            _np.degrees(
                                q_next
                                - q
                            )
                        )
                    )
                )

                print(
                    f"CLIK {control_step:02d} "
                    f"wp={path_index+1:02d}/"
                    f"{len(path_s):02d} "
                    f"N={current_s:+.5f}->"
                    f"{desired_s:+.5f} "
                    f"dep={depression*1000:.2f}mm "
                    f"rank={rank} "
                    f"dqmax={max_delta_deg:.2f}deg"
                )

                _ik_command_configuration(
                    f"clik-{control_step:02d}",
                    q_next,
                    dataset_context={
                        "control_step":
                            int(control_step),

                        "waypoint_index":
                            int(path_index),

                        "desired_normal_coordinate_m":
                            float(
                                desired_s
                            ),

                        "press_direction_world_xy": [
                            float(
                                _IK_TARGET_NORMAL[0]
                            ),
                            float(
                                _IK_TARGET_NORMAL[1]
                            ),
                        ],

                        "depression_before_m":
                            float(depression),

                        "jacobian_rank":
                            int(rank),

                        "jacobian_condition":
                            condition,

                        "qdot_rad_s": [
                            float(value)
                            for value
                            in qdot
                        ],

                        "max_delta_deg":
                            max_delta_deg,
                    },
                )

                time.sleep(
                    _CLIK_POST_COMMAND_SETTLE_S
                )

                state_after = (
                    _ik_read_state()
                )

                last_state = state_after

                bottom_after = _ik_connector(
                    state_after,
                    "end_effector",
                    "BOTTOM",
                )

                center_after = _np.asarray(
                    bottom_after[
                        "position_world"
                    ],
                    dtype=float,
                )

                normal_after = _ik_normalize(
                    _np.asarray(
                        bottom_after[
                            "outward_normal_world"
                        ],
                        dtype=float,
                    )
                )

                button_after = selected_button(state_after)

                depression_after = float(
                    button_after.get(
                        "depression_m",
                        0.0,
                    )
                )

                last_depression = (
                    depression_after
                )

                current_s_after = float(
                    center_after
                    @ _IK_TARGET_NORMAL
                )

                samples.append(
                    {
                        "control_step":
                            int(control_step),

                        "waypoint_index":
                            int(path_index),

                        "desired_normal_coordinate_m":
                            float(
                                desired_s
                            ),

                        "normal_coordinate_m":
                            current_s_after,

                        "press_direction_world_xy": [
                            float(
                                _IK_TARGET_NORMAL[0]
                            ),
                            float(
                                _IK_TARGET_NORMAL[1]
                            ),
                        ],

                        "center_xyz_m": [
                            float(value)
                            for value
                            in center_after
                        ],

                        "normal_world": [
                            float(value)
                            for value
                            in normal_after
                        ],

                        "depression_m":
                            float(
                                depression_after
                            ),

                        "jacobian_rank":
                            int(rank),

                        "jacobian_condition":
                            condition,

                        "qdot_rad_s": [
                            float(value)
                            for value
                            in qdot
                        ],

                        "max_delta_deg":
                            max_delta_deg,
                    }
                )

                if (
                    depression_after
                    >= _IK_SUCCESS_DEPRESSION_M
                ):
                    print(
                        f"CLIK physical success after "
                        f"command {control_step}: "
                        f"depression="
                        f"{depression_after*1000:.3f} mm"
                    )

                    return {
                        "success": True,
                        "reason":
                            "physical button depression reached",
                        "depression_m":
                            depression_after,
                        "state":
                            state_after,
                        "samples":
                            samples,
                    }

                while (
                    path_index
                    < len(path_s) - 1
                    and current_s_after
                    >= (
                        float(
                            path_s[
                                path_index
                            ]
                        )
                        - _CLIK_WAYPOINT_TOL_M
                    )
                ):
                    path_index += 1

                if (
                    current_s_after
                    > best_s
                    + _CLIK_MIN_PROGRESS_M
                ):
                    best_s = (
                        current_s_after
                    )
                    stall_steps = 0

                else:
                    stall_steps += 1

                if (
                    stall_steps
                    >= _CLIK_STALL_STEPS
                ):
                    return {
                        "success": False,
                        "reason":
                            "CLIK forward progress stalled",
                        "depression_m":
                            depression_after,
                        "state":
                            state_after,
                        "samples":
                            samples,
                    }

            return {
                "success": False,
                "reason":
                    "CLIK maximum control steps reached",
                "depression_m":
                    last_depression,
                "state":
                    last_state,
                "samples":
                    samples,
            }


        # --------------------------------------------------------
        # Target geometry
        # --------------------------------------------------------

        state0 = _ik_read_state()

        button0 = selected_button(state0)

        button_center = _np.asarray(
            button0[
                "center_xyz_m"
            ],
            dtype=float,
        )

        # Robot-side resting face of the physical plunger.
        rest_front_center = (
            button_center
            - (
                0.5
                * _IK_PLUNGER_DEPTH_M
            )
            * _IK_TARGET_NORMAL
        )

        rest_front_s = float(
            rest_front_center
            @ _IK_TARGET_NORMAL
        )

        bottom0 = _ik_connector(
            state0,
            "end_effector",
            "BOTTOM",
        )

        bottom0_center = _np.asarray(
            bottom0[
                "position_world"
            ],
            dtype=float,
        )

        button_t_center = float(
            button_center
            @ _IK_FACE_TANGENT
        )

        button_t_min = (
            button_t_center
            - _IK_BUTTON_HALF_TANGENT_M
        )

        button_t_max = (
            button_t_center
            + _IK_BUTTON_HALF_TANGENT_M
        )

        button_z_min = (
            button_center[2]
            - _IK_BUTTON_HALF_Z_M
        )

        button_z_max = (
            button_center[2]
            + _IK_BUTTON_HALF_Z_M
        )

        bottom_t = float(
            bottom0_center
            @ _IK_FACE_TANGENT
        )

        contact_t = float(
            _np.clip(
                bottom_t,
                button_t_min,
                button_t_max,
            )
        )

        contact_z = float(
            _np.clip(
                bottom0_center[2],
                button_z_min,
                button_z_max,
            )
        )

        # Build the closest admissible point on the finite
        # resting button face.
        contact_face = (
            rest_front_center.copy()
        )

        contact_face += (
            contact_t
            - float(
                contact_face
                @ _IK_FACE_TANGENT
            )
        ) * _IK_FACE_TANGENT

        contact_face[2] = (
            contact_z
        )

        pre_target = (
            contact_face
            - _IK_PRE_GAP_M
            * _IK_TARGET_NORMAL
        )

        press_target = (
            contact_face
            + _IK_PRESS_DEPTH_M
            * _IK_TARGET_NORMAL
        )

        print()
        print(
            "BUTTON CONTACT REGION:"
        )
        print(
            "  N = "
            f"[{_IK_TARGET_NORMAL[0]:+.1f}, "
            f"{_IK_TARGET_NORMAL[1]:+.1f}, "
            "0.0]"
        )
        print(
            "  T = "
            f"[{_IK_FACE_TANGENT[0]:+.1f}, "
            f"{_IK_FACE_TANGENT[1]:+.1f}, "
            "0.0]"
        )
        print(
            f"  T interval = "
            f"[{button_t_min:+.5f}, "
            f"{button_t_max:+.5f}]"
        )
        print(
            f"  Z interval = "
            f"[{button_z_min:+.5f}, "
            f"{button_z_max:+.5f}]"
        )
        print(
            f"  current BOTTOM T/Z = "
            f"({bottom_t:+.5f}, "
            f"{bottom0_center[2]:+.5f})"
        )
        print(
            f"  selected contact T/Z = "
            f"({contact_t:+.5f}, "
            f"{contact_z:+.5f})"
        )

        # --------------------------------------------------------
        # 6A. CLOSED-LOOP PRE TRAJECTORY
        #
        # PRE is not a one-shot q target anymore.  The controller
        # repeatedly reads the REAL state, creates a bounded position
        # and normal waypoint, solves locally, executes a bounded step,
        # and re-linearizes from the new physical state.
        # --------------------------------------------------------

        plan = {
            "schema_version":
                "mssr.bottom_face_clik.v5",

            "dofs": [
                f"{role}.{joint}"
                for role, joint
                in _IK_DOF
            ],

            "bottom_face":
                "BOTTOM",

            "contact_model":
                "finite_surface_overlap",

            "bottom_contact_half_extent_tangent_m":
                _IK_EE_CONTACT_HALF_TANGENT_M,

            "bottom_contact_half_extent_z_m":
                _IK_EE_CONTACT_HALF_Z_M,

            "target_normal": [
                float(_IK_TARGET_NORMAL[0]),
                float(_IK_TARGET_NORMAL[1]),
                0.0,
            ],

            "button_face_tangent_world": [
                float(_IK_FACE_TANGENT[0]),
                float(_IK_FACE_TANGENT[1]),
                0.0,
            ],

            "pre_gap_m":
                _IK_PRE_GAP_M,

            "press_depth_m":
                _IK_PRESS_DEPTH_M,
        }

        preorient_state = _ik_read_state()

        preorient_plan = _ik_plan_end_effector_tilt_preorientation(
            preorient_state
        )

        print()
        print(
            "------------------------------------------------------------"
        )
        print(
            "PRE-ORIENT TILT — END EFFECTOR ONLY"
        )
        print(
            "------------------------------------------------------------"
        )
        print("PAN command = NONE")
        print(
            "current TILT = "
            f"{math.degrees(preorient_plan['current_tilt_rad']):+.2f} deg"
        )
        print(
            "selected TILT = "
            f"{math.degrees(preorient_plan['target_tilt_rad']):+.2f} deg"
        )
        print(
            "normal error current   = "
            f"{preorient_plan['current_normal_error_deg']:.2f} deg"
        )
        print(
            "normal error predicted = "
            f"{preorient_plan['predicted_normal_error_deg']:.2f} deg"
        )
        print(
            "predicted improvement  = "
            f"{preorient_plan['improvement_deg']:.2f} deg"
        )

        if preorient_plan["should_move"]:
            pan_before_preorient = float(
                _ik_module(
                    preorient_state,
                    "end_effector",
                )["actuators"]["pan"][
                    "position_rad"
                ]
            )

            _ik_command_configuration(
                "PRE-ORIENT TILT",
                preorient_plan["q_target"],
                only_dofs={("end_effector", "tilt")},
            )

            preorient_after = _ik_read_state()

            pan_after_preorient = float(
                _ik_module(
                    preorient_after,
                    "end_effector",
                )["actuators"]["pan"][
                    "position_rad"
                ]
            )

            bottom_after = _ik_connector(
                preorient_after,
                "end_effector",
                "BOTTOM",
            )

            actual_error = float(
                _ik_normal_angle_deg(
                    _ik_normalize(
                        _np.asarray(
                            bottom_after[
                                "outward_normal_world"
                            ],
                            dtype=float,
                        )
                    ),
                    _IK_TARGET_NORMAL,
                )
            )

            pan_drift_deg = math.degrees(
                shortest_angular_delta(
                    pan_before_preorient,
                    pan_after_preorient,
                )
            )

            print(
                "normal error actual    = "
                f"{actual_error:.2f} deg"
            )
            print(
                "PAN physical drift     = "
                f"{pan_drift_deg:+.3f} deg "
                "(NO PAN primitive)"
            )

        else:
            print(
                "PRE-ORIENT skipped: TILT-only does not "
                "provide enough useful improvement, or the "
                "face is already inside the 30 deg cone."
            )

        print(
            "PRE-ORIENT complete -> CLOSED-LOOP PRE"
        )
        print()

        pre_result = _ik_clik_pre(
            pre_target,
        )

        plan["pre"] = {
            "controller":
                "closed_loop_task_space_waypoints",

            "success":
                bool(pre_result["success"]),

            "reason":
                pre_result["reason"],

            "target_center_xyz_m": [
                float(value)
                for value in pre_target
            ],

            "target_normal": [
                float(value)
                for value in _IK_TARGET_NORMAL
            ],

            "position_waypoint_max_m":
                _PRE_POSITION_STEP_M,

            "normal_waypoint_max_deg":
                math.degrees(_PRE_NORMAL_STEP_RAD),

            "joint_step_max_deg":
                math.degrees(_PRE_JOINT_STEP_MAX_RAD),

            "samples":
                pre_result["samples"],
        }

        if not pre_result["success"]:

            (
                run
                / "bottom_face_5dof_ik_plan.json"
            ).write_text(
                json.dumps(
                    plan,
                    indent=2,
                ) + "\n"
            )

            print()
            print(
                "CLOSED-LOOP PRE DID NOT CONVERGE."
            )
            print(
                "reason = "
                + str(pre_result["reason"])
            )
            print(
                "Runtime left open for inspection."
            )

            _ik_node.destroy_node()

            success = False
            if not args.headless:
                runtime = None
            raise SystemExit(1)

        # Record the REAL settled pre-contact joint state.
        pre_state = pre_result["state"]

        pre_state = _ik_read_state()

        q_pre_actual = _np.asarray(
            [
                _ik_joint_value(
                    pre_state,
                    role,
                    joint,
                )
                for role, joint
                in _IK_DOF
            ],
            dtype=float,
        )

        pre_bottom = _ik_connector(
            pre_state,
            "end_effector",
            "BOTTOM",
        )

        print()
        print(
            "PHYSICAL PRE-CONTACT BOTTOM:"
        )

        print(
            " center = "
            + str(
                pre_bottom[
                    "position_world"
                ]
            )
        )

        print(
            " normal = "
            + str(
                pre_bottom[
                    "outward_normal_world"
                ]
            )
        )

        # --------------------------------------------------------
        # 6B. CLOSED-LOOP PRESS TRAJECTORY
        #
        # The old architecture solved one q_press and then executed
        # it open-loop.  This phase now follows the Cartesian
        # approach using resolved-rate DLS with physical feedback
        # after every small joint-space integration step.
        #
        # Success is NOT a geometric guess: the controller stops
        # when the physical plunger depression reaches threshold.
        # --------------------------------------------------------

        clik_result = _ik_clik_press(
            pre_state,
            q_pre_actual,
            press_target,
        )

        plan["press"] = {
            "controller":
                "closed_loop_resolved_rate_dls",

            "surface_contact":
                "button_patch_overlap_bottom_patch",

            "success":
                bool(
                    clik_result[
                        "success"
                    ]
                ),

            "reason":
                clik_result[
                    "reason"
                ],

            "samples":
                clik_result[
                    "samples"
                ],
        }

        (
            run
            / "bottom_face_5dof_ik_plan.json"
        ).write_text(
            json.dumps(
                plan,
                indent=2,
            ) + "\n"
        )

        depression = float(
            clik_result[
                "depression_m"
            ]
        )

        state_press = (
            clik_result[
                "state"
            ]
        )

        if not clik_result["success"]:

            print()
            print(
                "CLIK PRESS DID NOT REACH PHYSICAL SUCCESS."
            )

            print(
                "reason = "
                + str(
                    clik_result[
                        "reason"
                    ]
                )
            )

            print(
                f"depression = "
                f"{depression*1000:.3f} mm"
            )

            print(
                "No recovery primitive is recorded after a failed press."
            )

            _ik_node.destroy_node()

            success = False
            if not args.headless:
                runtime = None
            raise SystemExit(1)

        press_bottom = _ik_connector(
            state_press,
            "end_effector",
            "BOTTOM",
        )

        print()
        print("============================================================")

        if (
            depression
            >= _IK_SUCCESS_DEPRESSION_M
        ):
            print(
                " BUTTON PRESSED — PHYSICAL SUCCESS"
            )
        else:
            print(
                " BUTTON PRESS FAILED"
            )

        print("============================================================")

        print(
            f"depression = "
            f"{depression*1000:.3f} mm"
        )

        print(
            "BOTTOM center = "
            + str(
                press_bottom[
                    "position_world"
                ]
            )
        )

        print(
            "BOTTOM normal = "
            + str(
                press_bottom[
                    "outward_normal_world"
                ]
            )
        )

        # --------------------------------------------------------
        # 6C. TERMINAL PHYSICAL SUCCESS
        # --------------------------------------------------------

        final_state = _ik_read_state()
        final_bottom = _ik_connector(
            final_state,
            "end_effector",
            "BOTTOM",
        )

        result = {
            "schema_version": "mssr.button_bottom_face_press.v5",
            "press_direction_world_xy": [
                float(_IK_TARGET_NORMAL[0]),
                float(_IK_TARGET_NORMAL[1]),
            ],
            "success": bool(
                depression >= _IK_SUCCESS_DEPRESSION_M
            ),
            "depression_m": depression,
            "threshold_m": _IK_SUCCESS_DEPRESSION_M,
            "pre_gap_m": _IK_PRE_GAP_M,
            "press_depth_m": _IK_PRESS_DEPTH_M,
            "press_bottom_center_xyz_m": press_bottom["position_world"],
            "press_bottom_normal": press_bottom["outward_normal_world"],
            "terminal_bottom_center_xyz_m": final_bottom["position_world"],
            "terminal_bottom_normal": final_bottom["outward_normal_world"],
        }

        (run / "button_press.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )

        _ik_node.destroy_node()

        for _ in range(10):
            rclpy.spin_once(monitor, timeout_sec=0.1)

        final_x, final_y, final_yaw = monitor.pose()
        final_course_state = read_json(
            runtime_dir / "state_graph.json"
        )
        final_button = (
            button_xyz(final_course_state)
            if final_course_state
            else None
        )

        final_summary = {
            "schema_version": "mssr.button_expert_complete.v2",
            "seed": SEED,
            "success": True,
            "button_pressed": True,
            "final_morphology": "mobile_manipulator8",
            "reposition": result_reposition,
            "button_press": result,
            "final_root_xy_m": [float(final_x), float(final_y)],
            "final_root_yaw_rad": float(final_yaw),
            "button_xyz_m": (
                list(final_button)
                if final_button is not None
                else None
            ),
        }

        (run / "expert_complete.json").write_text(
            json.dumps(final_summary, indent=2) + "\n"
        )

        print()
        print("============================================================")
        print(" BUTTON EXPERT — TERMINAL SUCCESS")
        print("============================================================")
        print("direct MM8 assembly = SUCCEEDED")
        print("Scorpion/reposition = SUCCEEDED")
        print("button press         = SUCCEEDED")
        print(
            f"final root           = ({final_x:+.3f},{final_y:+.3f})"
        )
        print(
            f"final yaw            = {math.degrees(final_yaw):+.2f} deg"
        )
        print(
            f"expert_complete.json = {run / 'expert_complete.json'}"
        )
        print()
        print("Runtime left OPEN in MobileManipulator8 press posture.")

        success = True

        # Interactive reference runs preserve Isaac for inspection.
        # Headless campaign runs keep the process handle so the
        # finally block can terminate it cleanly.
        if not args.headless:
            runtime = None

    finally:

        stop_process(assembly)

        if args.headless or not success:
            stop_process(runtime)

        monitor.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
