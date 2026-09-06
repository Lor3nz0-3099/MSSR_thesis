#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import os
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

SEED = 6100
DOMAIN = os.environ["ROS_DOMAIN_ID"]

RC_STANDOFF_M = 0.55
RC_GOAL_YAW_RAD = -math.pi / 2.0

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


def button_xyz(state):
    candidates = (
        state.get("state", {}).get("course", {}).get("button"),
        state.get("course", {}).get("button"),
        state.get("global_attributes", {})
             .get("course", {}).get("button"),
    )

    for button in candidates:
        if not isinstance(button, dict):
            continue

        xyz = (
            button.get("current_center_xyz_m")
            or button.get("center_xyz_m")
        )

        if isinstance(xyz, (list, tuple)) and len(xyz) >= 3:
            return tuple(float(v) for v in xyz[:3])

    raise RuntimeError("Button center not found.")


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

    while time.monotonic() < deadline:
        okay = True

        for node in required:
            result = subprocess.run(
                ["ros2", "lifecycle", "get", node],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=5,
            )

            if (
                result.returncode != 0
                or "active" not in result.stdout.lower()
            ):
                okay = False
                break

        if okay:
            return

        time.sleep(1)

    raise TimeoutError("Nav2 did not become ACTIVE.")


def wait_assembly(graph_path, runtime):
    deadline = time.monotonic() + 300
    stable = 0

    while time.monotonic() < deadline:

        if runtime.poll() is not None:
            raise RuntimeError("Runtime exited during assembly.")

        graph = read_json(graph_path)

        if graph_connections(graph) == 7:
            stable += 1

            if stable >= 5:
                return
        else:
            stable = 0

        time.sleep(0.5)

    raise TimeoutError("RC-Car8 assembly timeout.")



def align_rc_rear_to_button(monitor, state_path):
    """Bounded reverse-arc alignment: RC rear points toward live button."""
    state = read_json(state_path)

    if not state:
        raise RuntimeError("State graph unavailable for RC alignment.")

    bx, by, bz = button_xyz(state)

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

    monitor.spin_until(
        lambda: monitor.odom is not None,
        10,
        "/odom unavailable before MM8 approach.",
    )

    # Desired base center immediately before unfolding.
    tx = bx
    ty = by - standoff_m

    print()
    print("============================================================")
    print(f" {label}")
    print("============================================================")
    print(
        "Semantic drive pair: "
        "front_support + arm_lift"
    )
    print(f"button = ({bx:+.3f}, {by:+.3f}, {bz:+.3f})")
    print(f"target = ({tx:+.3f}, {ty:+.3f})")
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

            longitudinal = (
                dx * math.cos(yaw)
                + dy * math.sin(yaw)
            )

            lateral = (
                -dx * math.sin(yaw)
                + dy * math.cos(yaw)
            )

            error = abs(longitudinal)

            if error <= MM8_LONG_TOL_M:
                print()
                print("MM8 manipulation region reached.")
                break

            if abs(lateral) > MM8_MAX_LATERAL_M:
                raise RuntimeError(
                    "Lateral error too large: "
                    f"{lateral*1000:.1f} mm"
                )

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

            if error > 0.12:
                speed = 0.050
            else:
                # Keep enough wheel command near the target.
                # At 0.020 m/s the scorpion stopped making
                # measurable progress at ~50 mm residual error.
                speed = 0.035

            speed = math.copysign(
                speed,
                longitudinal,
            )

            cmd = Twist()
            cmd.linear.x = speed
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
                    f"long={longitudinal*1000:+.1f} mm "
                    f"lat={lateral*1000:+.1f} mm "
                    f"cmd={speed:+.3f}"
                )

                last_print = now

            time.sleep(0.04)

        else:
            raise RuntimeError(
                "MM8 approach wall timeout."
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
        "target_xy_m": [tx, ty],
        "root_xy_m": [x, y],
        "root_yaw_rad": yaw,
    }


def main():

    run_id = time.strftime(
        f"seed-{SEED:06d}-%Y%m%d-%H%M%S"
    )

    run = (
        ROOT
        / "logs"
        / "button_expert_to_ik"
        / run_id
    )

    runtime_dir = run / "runtime"

    runtime_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

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
    nav2 = None
    reconfig = None

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

        runtime = subprocess.Popen(
            [
                "ros2", "launch",
                "mssr_expert",
                "smores_runtime.launch.py",

                f"runtime_dir:={runtime_dir}",
                "module_count:=8",

                "button_test_course:=true",
                f"button_seed:={SEED}",

                "headless:=false",
                "performance:=true",
                "simple_visuals:=true",

                "simulation_steps:=0",
                "simulation_speed_factor:=1.0",

                "actuator_effort_scale:=4.0",
                "wheel_friction_scale:=1.50",
                "tilt_effort_scale:=8.0",

                "behavior_dataset_path:="
                + str(
                    run / "behavior_dataset.jsonl"
                ),
                "behavior_dataset_episode_id:="
                + run_id,
                "behavior_dataset_stage_name:="
                + "button_expert",
                "behavior_dataset_difficulty:=0.0",
                "behavior_dataset_log_period:=1",
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

        # Isaac Sim cold startup can legitimately take longer than
        # 90 s while Kit extensions are being initialized.  Killing the
        # launch during that import phase can leave misleading secondary
        # Python-extension / torch import errors in runtime.log.
        deadline = time.monotonic() + 180

        while time.monotonic() < deadline:

            if runtime.poll() is not None:
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
        # 2. RC-CAR8 ASSEMBLY
        # ========================================================

        print()
        print("============================================================")
        print(" PHASE 2 — RC-CAR8 ASSEMBLY")
        print("============================================================")

        assembly_log = (
            run / "assembly.log"
        ).open("w")

        assembly = subprocess.Popen(
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
                      "config/smores_rc_car8.json"
                ),

                "-p",
                "execution_id:="
                + run_id + "-assembly",

                "-p",
                "episode_id:=" + run_id,

                "-p",
                "dataset_path:="
                + str(
                    run / "assembly_dataset.jsonl"
                ),
            ],
            cwd=ROOT,
            env=env,
            stdout=assembly_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        wait_assembly(
            runtime_dir / "robot_graph.json",
            runtime,
        )

        # Let final post-assembly posture settle.
        time.sleep(8)

        stop_process(assembly)
        assembly = None

        print("RC-Car8 READY.")

        monitor.spin_until(
            lambda: monitor.odom is not None,
            15,
            "RC-Car /odom unavailable.",
        )

        # ========================================================
        # 3. BUTTON-DERIVED NAV2 GOAL
        # ========================================================

        state = read_json(
            runtime_dir / "state_graph.json"
        )

        bx, by, bz = button_xyz(state)

        gx = bx
        gy = by - RC_STANDOFF_M
        gyaw = RC_GOAL_YAW_RAD

        print()
        print("============================================================")
        print(" PHASE 3 — RC-CAR NAV2 PREALIGNMENT")
        print("============================================================")
        print(
            f"button = ({bx:+.3f},"
            f"{by:+.3f},{bz:+.3f})"
        )
        print(
            f"goal   = ({gx:+.3f},"
            f"{gy:+.3f},"
            f"{math.degrees(gyaw):+.1f}°)"
        )

        nav2_log = (
            run / "nav2.log"
        ).open("w")

        nav2 = subprocess.Popen(
            [
                "ros2", "launch",
                "mssr_expert",
                "smores_nav2.launch.py",
                "autostart:=true",
                "log_level:=warn",
            ],
            cwd=ROOT,
            env=env,
            stdout=nav2_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        wait_nav2(env)

        route_cmd = [
            "python3",
            str(
                ROOT
                / "scripts/smores_ep/"
                  "run_rc_car_nav2_route.py"
            ),
            "--seed", str(SEED),
            "--goal-x", str(gx),
            "--goal-y", str(gy),
            "--goal-yaw", str(gyaw),
            "--action-timeout-s", "120",
            "--result-json",
            str(run / "nav2_result.json"),
        ]

        try:
            route = subprocess.run(
                route_cmd,
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=135,
            )

            (
                run / "nav2_route.log"
            ).write_text(route.stdout)

            route_rc = route.returncode

        except subprocess.TimeoutExpired as exc:

            output = (
                exc.stdout.decode()
                if isinstance(exc.stdout, bytes)
                else (exc.stdout or "")
            )

            (
                run / "nav2_route.log"
            ).write_text(output)

            route_rc = 124

        # Allow final /odom update.
        for _ in range(10):
            rclpy.spin_once(
                monitor,
                timeout_sec=0.1,
            )

        x, y, yaw = monitor.pose()

        pos_err = math.hypot(
            gx - x,
            gy - y,
        )

        yaw_err = abs(
            angle_error(yaw, gyaw)
        )

        accepted = (
            pos_err <= RC_ACCEPT_POSITION_M
            and yaw_err <= RC_ACCEPT_YAW_RAD
        )

        print()
        print(
            f"Nav2 result code = {route_rc}"
        )
        print(
            f"physical position error = "
            f"{pos_err*1000:.1f} mm"
        )
        print(
            f"physical yaw error = "
            f"{math.degrees(yaw_err):.1f}°"
        )

        if route_rc == 0:
            print("Nav2 terminal success.")

        elif accepted:
            print(
                "NAV2 ACCEPTED BY PHYSICAL REGION."
            )

        else:
            raise RuntimeError(
                "Nav2 stopped outside the "
                "pre-reconfiguration acceptance region."
            )

        stop_process(nav2)
        nav2 = None

        # Ensure Nav2 no longer owns /cmd_vel.
        time.sleep(2)

        # ========================================================
        # 3B. FINAL RC-CAR YAW ALIGNMENT
        # Rear of RC-Car points directly toward the live button.
        # ========================================================

        rc_align = align_rc_rear_to_button(
            monitor,
            runtime_dir / "state_graph.json",
        )

        (
            run / "rc_final_alignment.json"
        ).write_text(
            json.dumps(rc_align, indent=2) + "\n"
        )

        time.sleep(1)

        # ========================================================
        # 4. RC-CAR -> MM8
        # ========================================================

        print()
        print("============================================================")
        print(" PHASE 4 — RC-CAR8 -> MOBILEMANIPULATOR8")
        print("============================================================")

        reconfig_log = (
            run / "reconfiguration.log"
        ).open("w")

        monitor.reconfiguration = None

        reconfig = subprocess.Popen(
            [
                "ros2", "run",
                "mssr_expert",
                "mssr_smores_self_reconfiguration_node",
                "--ros-args",
                "-p", "source_graph_path:=auto",
                "-p",
                "target_morphology:=mobile_manipulator8",
            ],
            cwd=ROOT,
            env=env,
            stdout=reconfig_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        monitor.spin_until(
            lambda: (
                isinstance(
                    monitor.reconfiguration,
                    dict,
                )
                and monitor.reconfiguration.get(
                    "done"
                ) is True
            ),
            480,
            "Self-reconfiguration timeout.",
        )

        terminal = monitor.reconfiguration

        print(json.dumps(
            terminal,
            indent=2,
        ))

        if terminal.get("success") is not True:
            raise RuntimeError(
                "RC-Car8 -> MM8 failed."
            )

        stop_process(reconfig)
        reconfig = None

        time.sleep(2)

        # Measure how much yaw the reconfiguration itself changed.
        for _ in range(15):
            rclpy.spin_once(monitor, timeout_sec=0.1)

        px, py, pyaw = monitor.pose()

        bx, by, _ = rc_align["button_xyz_m"]
        pbearing = math.atan2(by - py, bx - px)
        pdesired = math.atan2(
            math.sin(pbearing + math.pi),
            math.cos(pbearing + math.pi),
        )

        yaw_shift = angle_error(
            pyaw,
            rc_align["yaw_after_rad"],
        )

        post = {
            "rc_alignment": rc_align,
            "mm8_xy_after_reconfiguration_m": [px, py],
            "mm8_yaw_after_reconfiguration_rad": pyaw,
            "reconfiguration_yaw_shift_rad": yaw_shift,
            "rear_to_button_yaw_error_after_reconfiguration_rad":
                angle_error(pdesired, pyaw),
        }

        (
            run / "pre_post_reconfiguration_alignment.json"
        ).write_text(
            json.dumps(post, indent=2) + "\n"
        )

        print()
        print("============================================================")
        print(" PRE / POST RECONFIGURATION YAW")
        print("============================================================")
        print(
            f"RC aligned yaw = "
            f"{math.degrees(rc_align['yaw_after_rad']):+.2f} deg"
        )
        print(
            f"MM8 yaw        = "
            f"{math.degrees(pyaw):+.2f} deg"
        )
        print(
            f"yaw shift      = "
            f"{math.degrees(yaw_shift):+.2f} deg"
        )
        print(
            f"MM8 rear-to-button error = "
            f"{math.degrees(angle_error(pdesired, pyaw)):+.2f} deg"
        )

        # ========================================================
        # 5. SCORPION
        # ========================================================

        print()
        print("============================================================")
        print(" PHASE 5 — SCORPION DRIVE POSTURE")
        print("============================================================")

        behavior(
            env,
            "restore_drive",
        )

        time.sleep(1)

        # ========================================================
        # 6. APPROACH WITH CORRECT SEMANTIC DRIVE PAIR
        # ========================================================

        print()
        print("============================================================")
        print(" PHASE 6 — MM8 APPROACH")
        print(" front_support + arm_lift")
        print("============================================================")

        result_approach = approach_mm8(
            monitor,
            runtime_dir / "state_graph.json",
        )

        (
            run / "mm8_approach.json"
        ).write_text(
            json.dumps(
                result_approach,
                indent=2,
            ) + "\n"
        )

        # ========================================================
        # 7. UNFOLD FOR MANIPULATION
        # ========================================================

        print()
        print("============================================================")
        print(" PHASE 7 — PREPARE MANIPULATION")
        print("============================================================")

        behavior(
            env,
            "prepare_manipulation",
        )

        time.sleep(1)

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

            "approach":
                result_approach,
        }

        (
            run / "ready_for_ik.json"
        ).write_text(
            json.dumps(
                ready,
                indent=2,
            ) + "\n"
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
            "Nav2 / assembly / reconfiguration are STOPPED."
        )
        print("5-DoF BOTTOM-face IK follows.")

        # ========================================================
        # 8. 5-DOF BOTTOM-FACE IK
        #
        # The free task face is end_effector.BOTTOM.
        # No physical finite-difference probing is performed.
        #
        # q_pre:
        #   BOTTOM centre aligned with button centre,
        #   BOTTOM normal = +Y,
        #   3 mm before the resting plunger face.
        #
        # q_press:
        #   same geometric constraint,
        #   advanced along +Y to depress the plunger.
        # ========================================================

        print()
        print("============================================================")
        print(" PHASE 8 — 5-DOF BOTTOM-FACE IK")
        print(" no physical probes")
        print("============================================================")

        import numpy as _np

        from std_msgs.msg import String as _IKString
        from mssr_expert.execution.primitive_protocol import (
            PrimitiveGoalRequest as _IKPrimitiveGoalRequest,
        )
        from smores_ep.config.geometry import (
            SmoresGeometry as _IKGeometry,
        )

        _ik_state_path = runtime_dir / "state_graph.json"

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

        # After the physical press and return to q_pre, retract the
        # end-effector farther from the wall before folding back into
        # the scorpion posture.  This prevents the scorpion recovery
        # sweep from crossing the button again.
        _IK_POST_PRESS_CLEARANCE_M = 0.0600

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
        _IK_EE_CONTACT_HALF_X_M = 0.0200
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
        _IK_PLUNGER_DEPTH_Y_M = 0.040

        # Button front face is an 80 x 80 mm rectangle in X-Z.
        # The task does NOT require the BOTTOM-face centre to hit
        # the button centre. Any point on this usable surface is valid.
        _IK_BUTTON_HALF_X_M = 0.040
        _IK_BUTTON_HALF_Z_M = 0.040

        _IK_SUCCESS_DEPRESSION_M = 0.0035

        _IK_ORIENTATION_LEVER_M = 0.055
        _IK_POSITION_TOL_M = 0.0020
        _IK_NORMAL_TOL_DEG = 3.0

        _ik_geom = _IKGeometry()

        def _ik_read_state():

            state = read_json(
                _ik_state_path
            )

            if not state:
                raise RuntimeError(
                    "state_graph unavailable during IK"
                )

            return state

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
        ):

            normal = _ik_normalize(normal)

            return math.degrees(
                math.acos(
                    float(
                        _np.clip(
                            normal[1],
                            -1.0,
                            +1.0,
                        )
                    )
                )
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
            """Residual to the admissible button-face region.

            X and Z are surface-overlap inequality constraints.
            The BOTTOM centre may lie outside the button rectangle
            as long as its effective contact patch still overlaps it.

            Y remains an equality constraint because it defines
            PRE-CONTACT / PRESS depth.
            """

            x = float(center[0])
            y = float(center[1])
            z = float(center[2])

            contact_center_x_min = (
                button_x_min
                - _IK_EE_CONTACT_HALF_X_M
            )
            contact_center_x_max = (
                button_x_max
                + _IK_EE_CONTACT_HALF_X_M
            )

            contact_center_z_min = (
                button_z_min
                - _IK_EE_CONTACT_HALF_Z_M
            )
            contact_center_z_max = (
                button_z_max
                + _IK_EE_CONTACT_HALF_Z_M
            )

            if x < contact_center_x_min:
                error_x = (
                    x
                    - contact_center_x_min
                )
            elif x > contact_center_x_max:
                error_x = (
                    x
                    - contact_center_x_max
                )
            else:
                error_x = 0.0

            if z < contact_center_z_min:
                error_z = (
                    z
                    - contact_center_z_min
                )
            elif z > contact_center_z_max:
                error_z = (
                    z
                    - contact_center_z_max
                )
            else:
                error_z = 0.0

            error_y = (
                y
                - float(target_center[1])
            )

            return _np.asarray(
                [
                    error_x,
                    error_y,
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

            # Five task residual components:
            #
            #   x: zero anywhere inside button X interval
            #   y: PRE/PRESS plane equality
            #   z: zero anywhere inside button Z interval
            #   nx = 0
            #   nz = 0
            #
            # ny is constrained to the +Y hemisphere
            # by the line search.
            return _np.asarray(
                [
                    error[0],
                    error[1],
                    error[2],
                    (
                        _IK_ORIENTATION_LEVER_M
                        * normal[0]
                    ),
                    (
                        _IK_ORIENTATION_LEVER_M
                        * normal[2]
                    ),
                ],
                dtype=float,
            )

        def _ik_jacobian(
            reference,
            q,
            target_center,
        ):

            step = math.radians(0.25)

            jacobian = _np.zeros(
                (5, 5),
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
                    )
                    - _ik_residual(
                        reference,
                        minus,
                        target_center,
                    )
                ) / (2.0 * step)

            return jacobian

        def _ik_solve(
            state,
            target_center,
            label,
        ):

            reference = _ik_build_reference(
                state
            )

            q = reference["q_ref"].copy()

            damping = 0.010
            max_step = math.radians(7.0)

            best_q = q.copy()
            best_cost = float("inf")

            first_rank = None
            first_condition = None

            for iteration in range(120):

                residual = _ik_residual(
                    reference,
                    q,
                    target_center,
                )

                cost = float(
                    residual @ residual
                )

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
                        normal
                    )
                )

                if (
                    position_error
                    <= _IK_POSITION_TOL_M
                    and normal_error
                    <= _IK_NORMAL_TOL_DEG
                    and normal[1] > 0.0
                ):
                    return {
                        "success": True,
                        "label": label,
                        "iterations":
                            iteration + 1,
                        "q":
                            q.copy(),
                        "reference":
                            reference,
                        "center":
                            center,
                        "normal":
                            normal,
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

                    _, trial_normal = _ik_fk(
                        reference,
                        trial,
                    )

                    # Reject the antiparallel normal branch.
                    if trial_normal[1] <= 0.0:
                        continue

                    trial_residual = (
                        _ik_residual(
                            reference,
                            trial,
                            target_center,
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
                "q": best_q,
                "reference": reference,
                "center": center,
                "normal": normal,
                "position_error_m":
                    float(
                        _ik_region_position_error(
                                center,
                                target_center,
                            )
                    ),
                "normal_error_deg":
                    _ik_normal_angle_deg(
                        normal
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
        ):

            print()
            print(
                f"COMMAND ABSOLUTE CONFIGURATION: "
                f"{label}"
            )

            for index, (
                role,
                joint,
            ) in enumerate(_IK_DOF):

                target = float(
                    q_target[index]
                )

                request = (
                    _IKPrimitiveGoalRequest(
                        goal_id=(
                            f"button-ik-"
                            f"{label}-"
                            f"{role}-"
                            f"{joint}-"
                            f"{time.time_ns()}"
                        ),
                        primitive=(
                            "set_tilt"
                            if joint == "tilt"
                            else "set_pan"
                        ),
                        module_ids=(
                            role_to_module[
                                role
                            ],
                        ),
                        parameters={
                            "angle_rad":
                                target,

                            "tolerance_rad":
                                math.radians(
                                    0.6
                                ),
                        },
                        timeout_s=20.0,
                    )
                )

                message = _IKString()

                message.data = json.dumps(
                    request.to_dict()
                )

                _ik_goal_pub.publish(
                    message
                )

                # PAN and TILT of one SMORES module share the
                # ``internal_motion:<module>`` primitive resource.
                # Therefore another primitive for the same module
                # must not be admitted while the previous one is
                # still active.
                #
                # q_target is still ONE mathematically computed
                # absolute IK configuration.  We simply execute its
                # primitive goals resource-safely; there are no
                # physical probes or IK recomputations here.
                joint_deadline = (
                    time.monotonic()
                    + 20.0
                )

                joint_stable_since = None

                while (
                    time.monotonic()
                    < joint_deadline
                ):

                    rclpy.spin_once(
                        _ik_node,
                        timeout_sec=0.05,
                    )

                    joint_state = (
                        _ik_read_state()
                    )

                    joint_module = (
                        _ik_module(
                            joint_state,
                            role,
                        )
                    )

                    joint_actuator = (
                        joint_module[
                            "actuators"
                        ][joint]
                    )

                    joint_actual = float(
                        joint_actuator[
                            "position_rad"
                        ]
                    )

                    joint_velocity = abs(
                        float(
                            joint_actuator.get(
                                "velocity_rad_s",
                                0.0,
                            )
                        )
                    )

                    joint_error = abs(
                        _ik_joint_error(
                            joint_actual,
                            float(
                                q_target[
                                    index
                                ]
                            ),
                            joint,
                        )
                    )

                    joint_now = (
                        time.monotonic()
                    )

                    # The primitive executor itself declares
                    # JOINT_TARGET_REACHED on position tolerance.
                    #
                    # Do NOT require the individual joint to become
                    # dynamically settled here: this is a loaded,
                    # compliant serial chain and a joint may still
                    # have non-zero velocity after reaching its
                    # commanded angle.
                    #
                    # Full configuration settling (position +
                    # velocity for all 5 DoFs) is checked below,
                    # after every absolute target has been issued.
                    if (
                        joint_error
                        <= math.radians(1.2)
                    ):
                        print(
                            f"  {role}.{joint}: "
                            f"target reached "
                            f"(error="
                            f"{math.degrees(joint_error):.2f}°, "
                            f"velocity="
                            f"{joint_velocity:.3f} rad/s)"
                        )
                        break

                    time.sleep(0.05)

                else:
                    raise RuntimeError(
                        f"{label}: "
                        f"{role}.{joint} "
                        f"did not reach its absolute target"
                    )

                # Allow the primitive executor to publish its
                # terminal state and release internal_motion before
                # admitting another joint primitive.
                time.sleep(0.15)

            deadline = (
                time.monotonic()
                + 20.0
            )

            stable_since = None

            while (
                time.monotonic()
                < deadline
            ):

                rclpy.spin_once(
                    _ik_node,
                    timeout_sec=0.05,
                )

                state = _ik_read_state()

                errors = []
                velocities = []

                for index, (
                    role,
                    joint,
                ) in enumerate(_IK_DOF):

                    module = _ik_module(
                        state,
                        role,
                    )

                    actuator = (
                        module[
                            "actuators"
                        ][joint]
                    )

                    actual = float(
                        actuator[
                            "position_rad"
                        ]
                    )

                    velocity = abs(
                        float(
                            actuator.get(
                                "velocity_rad_s",
                                0.0,
                            )
                        )
                    )

                    errors.append(
                        abs(
                            _ik_joint_error(
                                actual,
                                float(
                                    q_target[
                                        index
                                    ]
                                ),
                                joint,
                            )
                        )
                    )

                    velocities.append(
                        velocity
                    )

                within = (
                    max(errors)
                    <= math.radians(1.2)
                )

                now = time.monotonic()

                # The loaded MM8 arm is compliant and individual
                # actuator velocity estimates remain non-zero even
                # while the commanded configuration is being held.
                #
                # For the absolute IK configuration, require the
                # complete 5-DoF pose to remain inside the joint
                # position tolerance continuously for 0.6 s.
                #
                # The task-space PRE/PRESS geometry is validated
                # separately by the IK and by the physical button
                # depression criterion.
                if within:

                    if stable_since is None:
                        stable_since = now

                    elif (
                        now - stable_since
                        >= 0.6
                    ):
                        print(
                            f"{label}: reached "
                            f"(max error "
                            f"{math.degrees(max(errors)):.2f}°, "
                            f"max velocity "
                            f"{max(velocities):.3f} rad/s)"
                        )
                        return

                else:
                    stable_since = None

                time.sleep(0.05)

            raise RuntimeError(
                f"{label} absolute configuration "
                f"did not settle"
            )

        def _ik_clik_press(
            pre_state,
            q_pre_actual,
            press_target,
        ):
            """Closed-loop Cartesian press from the physical q_pre.

            This is a resolved-rate CLIK outer loop implemented over
            the existing absolute joint-position primitive transport.

            Every iteration:

              1. reads the REAL state graph;
              2. advances a small Cartesian trajectory reference;
              3. builds a local Jacobian around the REAL q;
              4. computes DLS resolved-rate motion;
              5. adds a nullspace posture objective;
              6. integrates qdot into one small q target;
              7. executes it;
              8. reads physical feedback again.

            Physical button depression is the terminal success signal.
            """

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
                initial_bottom["position_world"],
                dtype=float,
            )

            start_y = float(
                initial_center[1]
            )

            goal_y = float(
                press_target[1]
            )

            if goal_y <= start_y:
                return {
                    "success": False,
                    "reason":
                        "press target is not ahead of physical PRE",
                    "depression_m": 0.0,
                    "state": pre_state,
                    "samples": [],
                }

            # ----------------------------------------------------
            # Desired geometric trajectory.
            #
            # A sequence of 2 mm Cartesian references represents
            # one continuous approach along +Y.  Unlike the old
            # waypoint IK, these are not solved independently:
            # q evolves continuously through the feedback loop.
            # ----------------------------------------------------

            path_y = []

            y = (
                start_y
                + _CLIK_CARTESIAN_STEP_M
            )

            while y < goal_y:
                path_y.append(
                    float(y)
                )
                y += _CLIK_CARTESIAN_STEP_M

            path_y.append(
                float(goal_y)
            )

            path_index = 0
            samples = []

            best_y = start_y
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
                f"physical PRE y = "
                f"{start_y:+.5f} m"
            )
            print(
                f"PRESS goal y   = "
                f"{goal_y:+.5f} m"
            )
            print(
                f"path distance   = "
                f"{(goal_y-start_y)*1000:.2f} mm"
            )
            print(
                f"trajectory refs = "
                f"{len(path_y)}"
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

                button_live = (
                    state[
                        "state"
                    ][
                        "course"
                    ][
                        "button"
                    ]
                )

                depression = float(
                    button_live.get(
                        "depression_m",
                        0.0,
                    )
                )

                last_depression = depression

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

                q_live = _np.asarray(
                    [
                        _ik_joint_value(
                            state,
                            role,
                            joint,
                        )
                        for role, joint
                        in _IK_DOF
                    ],
                    dtype=float,
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

                if normal_live[1] <= 0.0:
                    return {
                        "success": False,
                        "reason":
                            "BOTTOM normal left +Y hemisphere",
                        "depression_m":
                            depression,
                        "state":
                            state,
                        "samples":
                            samples,
                    }

                # The PRESS trajectory is monotonic along +Y.
                #
                # A physical command can legitimately overshoot a
                # 2-mm Cartesian reference.  Do not require the EE
                # to land inside a narrow symmetric tolerance around
                # each waypoint: once a waypoint has been reached or
                # crossed, advance to the next one.
                while (
                    path_index
                    < len(path_y) - 1
                    and float(center_live[1])
                    >= (
                        float(path_y[path_index])
                        - _CLIK_WAYPOINT_TOL_M
                    )
                ):
                    path_index += 1

                desired_y = float(
                    path_y[path_index]
                )

                desired = _np.asarray(
                    [
                        float(press_target[0]),
                        desired_y,
                        float(press_target[2]),
                    ],
                    dtype=float,
                )

                # ------------------------------------------------
                # Re-linearize around the REAL current robot state.
                # This is the feedback step that the previous
                # open-loop q_press solution did not have.
                # ------------------------------------------------

                reference = _ik_build_reference(
                    state
                )

                q = reference[
                    "q_ref"
                ].copy()

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

                # ------------------------------------------------
                # Cartesian closed-loop desired rate.
                #
                # residual = current - desired.
                # Therefore -Kp*residual reduces pose error.
                #
                # +Y feed-forward supplies the desired trajectory
                # velocity.  Once physical contact is detected the
                # feed-forward speed is reduced.
                # ------------------------------------------------

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

                if (
                    float(center_live[1])
                    < desired_y
                ):
                    task_rate[1] += (
                        forward_speed
                    )

                # ------------------------------------------------
                # Damped pseudoinverse.
                # ------------------------------------------------

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

                # ------------------------------------------------
                # Nullspace redundancy objective.
                #
                # 1. Stay close to the comfortable physical q_pre.
                #    With the base moved closer, q_pre should be
                #    naturally less extended.
                #
                # 2. Softly bias joints away from hard limits.
                # ------------------------------------------------

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
                    reference["limits"]
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

                # ------------------------------------------------
                # Small line search purely for kinematic safety:
                # do not cross into the opposite normal branch.
                # ------------------------------------------------

                q_next = None
                predicted_center = None
                predicted_normal = None

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
                        reference["limits"]
                    ):
                        if lower is not None:
                            trial[index] = max(
                                float(trial[index]),
                                float(lower),
                            )

                        if upper is not None:
                            trial[index] = min(
                                float(trial[index]),
                                float(upper),
                            )

                    (
                        trial_center,
                        trial_normal,
                    ) = _ik_fk(
                        reference,
                        trial,
                    )

                    if (
                        trial_normal[1]
                        <= 0.0
                    ):
                        continue

                    q_next = trial
                    predicted_center = (
                        trial_center
                    )
                    predicted_normal = (
                        trial_normal
                    )
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
                    f"{len(path_y):02d} "
                    f"y={center_live[1]:+.5f}->"
                    f"{desired_y:+.5f} "
                    f"dep={depression*1000:.2f}mm "
                    f"rank={rank} "
                    f"dqmax={max_delta_deg:.2f}deg"
                )

                _ik_command_configuration(
                    f"clik-{control_step:02d}",
                    q_next,
                )

                time.sleep(
                    _CLIK_POST_COMMAND_SETTLE_S
                )

                # ------------------------------------------------
                # Physical feedback after executing this q step.
                # ------------------------------------------------

                state_after = _ik_read_state()
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

                button_after = (
                    state_after[
                        "state"
                    ][
                        "course"
                    ][
                        "button"
                    ]
                )

                depression_after = float(
                    button_after.get(
                        "depression_m",
                        0.0,
                    )
                )

                last_depression = (
                    depression_after
                )

                samples.append(
                    {
                        "control_step":
                            int(control_step),

                        "waypoint_index":
                            int(path_index),

                        "desired_y_m":
                            float(desired_y),

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

                # ------------------------------------------------
                # Progress the geometric trajectory only when the
                # REAL EE has reached the current local reference.
                # ------------------------------------------------

                while (
                    path_index
                    < len(path_y) - 1
                    and float(center_after[1])
                    >= (
                        float(path_y[path_index])
                        - _CLIK_WAYPOINT_TOL_M
                    )
                ):
                    path_index += 1

                # ------------------------------------------------
                # Stall detector.
                # ------------------------------------------------

                current_y = float(
                    center_after[1]
                )

                if (
                    current_y
                    > best_y
                    + _CLIK_MIN_PROGRESS_M
                ):
                    best_y = current_y
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

        button0 = (
            state0[
                "state"
            ][
                "course"
            ][
                "button"
            ]
        )

        button_center = _np.asarray(
            button0[
                "center_xyz_m"
            ],
            dtype=float,
        )

        rest_front_y = (
            button_center[1]
            - 0.5
            * _IK_PLUNGER_DEPTH_Y_M
        )

        # --------------------------------------------------------
        # Contact target = REGION on the button face, not its centre.
        #
        # Project the CURRENT BOTTOM-face centre onto the button's
        # X-Z rectangle. Therefore:
        #
        #   - if BOTTOM x/z is already over the button, preserve it;
        #   - if it lies outside, use the nearest point on the face.
        #
        # This avoids wasting arm reach trying to hit the geometric
        # centre when an off-centre face-to-face press is equally valid.
        # --------------------------------------------------------

        bottom0 = _ik_connector(
            state0,
            "end_effector",
            "BOTTOM",
        )

        bottom0_center = _np.asarray(
            bottom0["position_world"],
            dtype=float,
        )

        button_x_min = (
            button_center[0]
            - _IK_BUTTON_HALF_X_M
        )

        button_x_max = (
            button_center[0]
            + _IK_BUTTON_HALF_X_M
        )

        button_z_min = (
            button_center[2]
            - _IK_BUTTON_HALF_Z_M
        )

        button_z_max = (
            button_center[2]
            + _IK_BUTTON_HALF_Z_M
        )

        contact_x = float(
            _np.clip(
                bottom0_center[0],
                button_x_min,
                button_x_max,
            )
        )

        contact_z = float(
            _np.clip(
                bottom0_center[2],
                button_z_min,
                button_z_max,
            )
        )

        pre_target = _np.asarray(
            [
                contact_x,
                (
                    rest_front_y
                    - _IK_PRE_GAP_M
                ),
                contact_z,
            ],
            dtype=float,
        )

        press_target = _np.asarray(
            [
                contact_x,
                (
                    rest_front_y
                    + _IK_PRESS_DEPTH_M
                ),
                contact_z,
            ],
            dtype=float,
        )

        print()
        print("BUTTON CONTACT REGION:")
        print(
            f"  X = "
            f"[{button_x_min:+.5f}, "
            f"{button_x_max:+.5f}]"
        )
        print(
            f"  Z = "
            f"[{button_z_min:+.5f}, "
            f"{button_z_max:+.5f}]"
        )
        print(
            f"  current BOTTOM x/z = "
            f"({bottom0_center[0]:+.5f}, "
            f"{bottom0_center[2]:+.5f})"
        )
        print(
            f"  selected contact x/z = "
            f"({contact_x:+.5f}, "
            f"{contact_z:+.5f})"
        )

        # --------------------------------------------------------
        # 8A. Solve PRE from the actual manipulation_ready state.
        # --------------------------------------------------------

        pre_solution = _ik_solve(
            state0,
            pre_target,
            "PRE-CONTACT",
        )

        _ik_print_solution(
            pre_solution,
            pre_target,
        )

        plan = {
            "schema_version":
                "mssr.bottom_face_clik.v3",

            "dofs": [
                f"{role}.{joint}"
                for role, joint
                in _IK_DOF
            ],

            "bottom_face":
                "BOTTOM",

            "contact_model":
                "finite_surface_overlap",

            "bottom_contact_half_extent_x_m":
                _IK_EE_CONTACT_HALF_X_M,

            "bottom_contact_half_extent_z_m":
                _IK_EE_CONTACT_HALF_Z_M,

            "target_normal":
                [0.0, 1.0, 0.0],

            "pre_gap_m":
                _IK_PRE_GAP_M,

            "press_depth_m":
                _IK_PRESS_DEPTH_M,

            "pre":
                _ik_plan_payload(
                    pre_solution,
                    pre_target,
                ),
        }

        if not pre_solution["success"]:

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
                "PRE IK DID NOT CONVERGE."
            )

            print(
                "No IK joint target was sent."
            )

            print(
                "Runtime left open for inspection."
            )

            _ik_node.destroy_node()

            success = False
            runtime = None
            raise SystemExit(1)

        # One absolute movement to q_pre.
        _ik_command_configuration(
            "pre",
            pre_solution["q"],
        )

        time.sleep(0.8)

        # Record the REAL settled pre-contact joint state.
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
        # 8B. CLOSED-LOOP PRESS TRAJECTORY
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
                "Returning to physical q_pre."
            )

            _ik_command_configuration(
                "pre-return",
                q_pre_actual,
            )

            _ik_node.destroy_node()

            success = False
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
        # 8C. Absolute return to the REAL q_pre.
        # --------------------------------------------------------

        _ik_command_configuration(
            "pre-return",
            q_pre_actual,
        )

        time.sleep(0.8)

        # --------------------------------------------------------
        # 8D. POST-PRESS ARM CLEARANCE
        #
        # q_pre is about 30 mm from the resting button face.  That
        # is sufficient for manipulation, but the large scorpion
        # recovery motion can sweep the arm back through the button.
        #
        # Before restore_drive, move the BOTTOM face farther away
        # from the wall while preserving its current X/Z location
        # and +Y-facing orientation.
        #
        # This motion is AWAY from the obstacle; the validated CLIK
        # remains responsible for the actual contact/press phase.
        # --------------------------------------------------------

        clearance_state = _ik_read_state()

        clearance_bottom0 = _ik_connector(
            clearance_state,
            "end_effector",
            "BOTTOM",
        )

        clearance_center0 = _np.asarray(
            clearance_bottom0[
                "position_world"
            ],
            dtype=float,
        )

        clearance_target = _np.asarray(
            [
                float(
                    clearance_center0[0]
                ),
                float(
                    rest_front_y
                    - _IK_POST_PRESS_CLEARANCE_M
                ),
                float(
                    clearance_center0[2]
                ),
            ],
            dtype=float,
        )

        print()
        print(
            "------------------------------------------------------------"
        )
        print(
            "POST-PRESS ARM CLEARANCE"
        )
        print(
            "------------------------------------------------------------"
        )
        print(
            f"current BOTTOM y = "
            f"{clearance_center0[1]:+.5f} m"
        )
        print(
            f"clearance target = "
            f"{_IK_POST_PRESS_CLEARANCE_M*1000:.1f} mm "
            f"from resting button face"
        )
        print(
            f"target BOTTOM y  = "
            f"{clearance_target[1]:+.5f} m"
        )

        clearance_solution = _ik_solve(
            clearance_state,
            clearance_target,
            "POST-PRESS CLEARANCE",
        )

        _ik_print_solution(
            clearance_solution,
            clearance_target,
        )

        plan[
            "post_press_clearance"
        ] = _ik_plan_payload(
            clearance_solution,
            clearance_target,
        )

        plan[
            "post_press_clearance"
        ][
            "clearance_m"
        ] = (
            _IK_POST_PRESS_CLEARANCE_M
        )

        (
            run
            / "bottom_face_5dof_ik_plan.json"
        ).write_text(
            json.dumps(
                plan,
                indent=2,
            ) + "\n"
        )

        if not clearance_solution["success"]:

            print()
            print(
                "POST-PRESS CLEARANCE IK DID NOT CONVERGE."
            )
            print(
                "Scorpion restore intentionally NOT executed."
            )
            print(
                "Runtime left open for inspection."
            )

            _ik_node.destroy_node()

            success = False
            runtime = None
            raise SystemExit(1)

        _ik_command_configuration(
            "post-press-clearance",
            clearance_solution["q"],
        )

        time.sleep(0.8)

        final_state = _ik_read_state()

        final_bottom = _ik_connector(
            final_state,
            "end_effector",
            "BOTTOM",
        )

        result = {
            "schema_version":
                "mssr.button_bottom_face_press.v3",

            "success":
                bool(
                    depression
                    >= _IK_SUCCESS_DEPRESSION_M
                ),

            "depression_m":
                depression,

            "threshold_m":
                _IK_SUCCESS_DEPRESSION_M,

            "pre_gap_m":
                _IK_PRE_GAP_M,

            "press_depth_m":
                _IK_PRESS_DEPTH_M,

            "post_press_clearance_m":
                _IK_POST_PRESS_CLEARANCE_M,

            "press_bottom_center_xyz_m":
                press_bottom[
                    "position_world"
                ],

            "press_bottom_normal":
                press_bottom[
                    "outward_normal_world"
                ],

            "returned_bottom_center_xyz_m":
                final_bottom[
                    "position_world"
                ],

            "returned_bottom_normal":
                final_bottom[
                    "outward_normal_world"
                ],
        }

        (
            run / "button_press.json"
        ).write_text(
            json.dumps(
                result,
                indent=2,
            ) + "\n"
        )

        _ik_node.destroy_node()

        # ========================================================
        # 9. RETURN TO VALIDATED SCORPION POSTURE
        # ========================================================

        print()
        print("============================================================")
        print(" PHASE 9 — RETURN TO SCORPION")
        print("============================================================")

        behavior(
            env,
            "restore_drive",
        )

        time.sleep(1.0)

        print()
        print(
            f"button_press.json = "
            f"{run / 'button_press.json'}"
        )

        print(
            f"physical success = "
            f"{result['success']}"
        )

        # ========================================================
        # 10. RETREAT IN VALIDATED SCORPION LOCOMOTION
        #
        # Reuse the exact same longitudinal controller and semantic
        # drive pair used by PHASE 6.  The only difference is the
        # desired standoff: move away from the button until the
        # connected MM8 root reaches the RC navigation standoff.
        # ========================================================

        print()
        print("============================================================")
        print(" PHASE 10 — SCORPION RETREAT")
        print(" front_support + arm_lift")
        print("============================================================")
        print(
            f"Retreat target standoff = "
            f"{RC_STANDOFF_M*1000:.1f} mm"
        )

        result_retreat = approach_mm8(
            monitor,
            runtime_dir / "state_graph.json",
            standoff_m=RC_STANDOFF_M,
            label="MM8 RETREAT",
        )

        (
            run / "mm8_retreat.json"
        ).write_text(
            json.dumps(
                result_retreat,
                indent=2,
            ) + "\n"
        )

        # Stop command is already handled by approach_mm8().
        time.sleep(1.0)

        # ========================================================
        # 11. MOBILEMANIPULATOR8 -> RC-CAR8
        #
        # Same self-reconfiguration contract already validated in
        # PHASE 4, but with the target morphology reversed.
        # ========================================================

        print()
        print("============================================================")
        print(" PHASE 11 — MOBILEMANIPULATOR8 -> RC-CAR8")
        print("============================================================")

        return_reconfig_log = (
            run / "return_reconfiguration.log"
        ).open("w")

        monitor.reconfiguration = None

        reconfig = subprocess.Popen(
            [
                "ros2", "run",
                "mssr_expert",
                "mssr_smores_self_reconfiguration_node",
                "--ros-args",
                "-p", "source_graph_path:=auto",
                "-p", "target_morphology:=rc_car8",
            ],
            cwd=ROOT,
            env=env,
            stdout=return_reconfig_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        monitor.spin_until(
            lambda: (
                isinstance(
                    monitor.reconfiguration,
                    dict,
                )
                and monitor.reconfiguration.get(
                    "done"
                ) is True
            ),
            480,
            "MM8 -> RC-Car8 self-reconfiguration timeout.",
        )

        return_terminal = (
            monitor.reconfiguration
        )

        print(
            json.dumps(
                return_terminal,
                indent=2,
            )
        )

        (
            run / "return_reconfiguration.json"
        ).write_text(
            json.dumps(
                return_terminal,
                indent=2,
            ) + "\n"
        )

        if (
            return_terminal.get(
                "success"
            )
            is not True
        ):
            raise RuntimeError(
                "MobileManipulator8 -> RC-Car8 failed."
            )

        stop_process(reconfig)
        reconfig = None

        time.sleep(2)

        # Let /odom and the state graph settle after the topology
        # transition, then record the final connected-robot pose.
        for _ in range(15):
            rclpy.spin_once(
                monitor,
                timeout_sec=0.1,
            )

        final_x, final_y, final_yaw = (
            monitor.pose()
        )

        final_state = read_json(
            runtime_dir / "state_graph.json"
        )

        final_button = (
            button_xyz(final_state)
            if final_state
            else None
        )

        final_summary = {
            "schema_version":
                "mssr.button_expert_complete.v1",

            "seed":
                SEED,

            "success":
                True,

            "button_pressed":
                True,

            "retreat":
                result_retreat,

            "return_reconfiguration":
                return_terminal,

            "final_morphology":
                "rc_car8",

            "final_root_xy_m": [
                float(final_x),
                float(final_y),
            ],

            "final_root_yaw_rad":
                float(final_yaw),

            "button_xyz_m": (
                list(final_button)
                if final_button is not None
                else None
            ),
        }

        (
            run / "expert_complete.json"
        ).write_text(
            json.dumps(
                final_summary,
                indent=2,
            ) + "\n"
        )

        print()
        print("============================================================")
        print(" BUTTON EXPERT — END-TO-END SUCCESS")
        print("============================================================")
        print("button press        = SUCCEEDED")
        print("pre-return          = SUCCEEDED")
        print("scorpion restore    = SUCCEEDED")
        print("scorpion retreat    = SUCCEEDED")
        print("MM8 -> RC-Car8      = SUCCEEDED")
        print(
            f"final root          = "
            f"({final_x:+.3f},{final_y:+.3f})"
        )
        print(
            f"final yaw           = "
            f"{math.degrees(final_yaw):+.2f} deg"
        )
        print(
            f"expert_complete.json = "
            f"{run / 'expert_complete.json'}"
        )
        print()
        print(
            "Runtime left OPEN in RC-Car8 posture."
        )

        success = True

        # Deliberately leave runtime process alive.
        runtime = None

    finally:

        stop_process(assembly)
        stop_process(nav2)
        stop_process(reconfig)

        if not success:
            stop_process(runtime)

        monitor.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
