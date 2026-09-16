#!/usr/bin/env python3
"""Focused RC-Car8 -> MobileManipulator8 reconfiguration timeline probe.

This is a diagnostic-only scenario.  It intentionally does not record IL
datasets and does not change morphology/reconfiguration policies.

Sequence:
  empty stage
  -> RC-Car8 self assembly
  -> short curved Nav2 route
  -> RC-Car8 -> MobileManipulator8 self reconfiguration
  -> observe untouched Scorpion posture
  -> retreat about 10 cm while still folded
  -> prepare_manipulation
  -> observe manipulation-ready posture

All diagnostic evidence is merged into one human-readable timeline.txt:
primitive goals/statuses/cancels, action payloads, cmd_vel, state snapshots,
PAN/TILT angles, declared structural-hold parameters, subprocess output,
reconfiguration state, and stage-boundary summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Mapping

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String


ROOT = Path(__file__).resolve().parents[2]
DOMAIN = os.environ.get("ROS_DOMAIN_ID", "0")

HOLD_PARAMETER_KEYS = (
    "structural_hold_module_ids",
    "stabilize_during_group_module_ids",
    "hold_after_group_module_ids",
    "passive_module_ids",
    "pusher_module_id",
    "coordination_group",
    "coordination_size",
    "max_coordination_lead_rad",
)


def structural_hold_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    parameters = payload.get("parameters")
    if not isinstance(parameters, Mapping):
        return {}
    return {
        key: parameters[key]
        for key in HOLD_PARAMETER_KEYS
        if key in parameters
    }


def build_curved_route(
    x_m: float,
    y_m: float,
    yaw_rad: float,
) -> tuple[tuple[float, float, float], ...]:
    """Two short local-frame waypoints that force a real left curve."""

    local = (
        (0.25, 0.08, math.radians(15.0)),
        (0.50, 0.25, math.radians(30.0)),
    )
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    route = []
    for dx, dy, dyaw in local:
        route.append(
            (
                x_m + c * dx - s * dy,
                y_m + s * dx + c * dy,
                math.atan2(
                    math.sin(yaw_rad + dyaw),
                    math.cos(yaw_rad + dyaw),
                ),
            )
        )
    return tuple(route)


def json_object(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json_object(path.read_text(encoding="utf-8"))
    except OSError:
        return None


def stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=8)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
            return
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class TimelineWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._start_wall = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("w", encoding="utf-8", buffering=1)

    def close(self) -> None:
        with self._lock:
            self._stream.flush()
            self._stream.close()

    def line(
        self,
        phase: str,
        kind: str,
        message: str,
        *,
        sim_s: float | None = None,
        echo: bool = False,
    ) -> None:
        wall = time.time()
        elapsed = wall - self._start_wall
        sim_text = "sim=       ?" if sim_s is None else f"sim={sim_s:8.3f}"
        prefix = (
            f"[wall={wall:.3f}] [dt={elapsed:8.3f}] "
            f"[{sim_text}] [{phase}] [{kind}]"
        )
        rendered = f"{prefix} {message}"
        with self._lock:
            self._stream.write(rendered + "\n")
        if echo:
            print(rendered, flush=True)

    def block(
        self,
        phase: str,
        kind: str,
        payload: Any,
        *,
        sim_s: float | None = None,
        echo: bool = False,
    ) -> None:
        if isinstance(payload, str):
            text = payload
        else:
            text = json.dumps(payload, indent=2, sort_keys=True, default=str)
        for index, line in enumerate(text.splitlines() or [""]):
            self.line(
                phase,
                kind if index == 0 else kind + "+",
                line,
                sim_s=sim_s,
                echo=echo,
            )


class TimelineMonitor(Node):
    def __init__(self, writer: TimelineWriter) -> None:
        super().__init__("mssr_reconfiguration_timeline_debug")
        self.writer = writer
        self.phase = "BOOT"
        self.latest_module_states: dict[str, Any] | None = None
        self.latest_state_graph: dict[str, Any] | None = None
        self.latest_robot_graph: dict[str, Any] | None = None
        self.latest_actions: dict[str, Any] | None = None
        self.latest_reconfiguration: dict[str, Any] | None = None
        self.latest_task_graph: dict[str, Any] | None = None
        self.odom: tuple[float, float, float] | None = None
        self.role_by_module: dict[str, str] = {}
        self.goal_by_id: dict[str, dict[str, Any]] = {}
        self.active_goal_ids: set[str] = set()
        self.persistent_declared_holds: set[str] = set()
        self.stage_snapshots: dict[str, dict[str, dict[str, Any]]] = {}
        self._last_raw: dict[str, str] = {}
        self._last_cmd_vel_log = 0.0
        self._last_cmd_vel: tuple[float, float] | None = None
        self._last_odom_log = 0.0

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        for topic, callback in (
            ("/mssr/module_states", self._on_module_states),
            ("/mssr/state_graph", self._on_state_graph),
            ("/mssr/robot_graph", self._on_robot_graph),
            ("/mssr/primitives/goal", self._on_primitive_goal),
            ("/mssr/primitives/status", self._on_primitive_status),
            ("/mssr/primitives/cancel", self._on_primitive_cancel),
            ("/mssr/actions", self._on_actions),
            (
                "/mssr/expert/self_reconfiguration/state",
                self._on_reconfiguration,
            ),
            ("/mssr/expert/task_graph", self._on_task_graph),
        ):
            self.create_subscription(String, topic, callback, 20)

        self.create_subscription(Odometry, "/odom", self._on_odom, 20)
        self.create_subscription(Twist, "/cmd_vel", self._on_cmd_vel, 20)

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        self.writer.line(
            self.phase,
            "STAGE",
            "=" * 18 + f" {phase} " + "=" * 18,
            sim_s=self.sim_time(),
            echo=True,
        )

    def sim_time(self) -> float | None:
        payload = self.latest_module_states
        if isinstance(payload, Mapping):
            raw = payload.get("stamp", payload.get("timestamp"))
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
        payload = self.latest_state_graph
        if isinstance(payload, Mapping):
            try:
                return float(payload.get("stamp"))
            except (TypeError, ValueError):
                pass
        return None

    def source_hint(self) -> str:
        return {
            "ASSEMBLY": "mssr_smores_self_assembly_node",
            "RC_CURVE": "Nav2 + morphology_behavior_node",
            "RECONFIG": "mssr_smores_self_reconfiguration_node",
            "SCORPION_OBSERVE": "no new command expected",
            "MM8_RETREAT": "cmd_vel -> morphology_behavior_node",
            "MANIPULATION_READY": "morphology_command_client -> behavior_node",
        }.get(self.phase, "unknown/current-stage")

    def _dedup_json(
        self,
        key: str,
        msg: String,
    ) -> dict[str, Any] | None:
        raw = msg.data
        if self._last_raw.get(key) == raw:
            return None
        self._last_raw[key] = raw
        return json_object(raw)

    def _on_state_graph(self, msg: String) -> None:
        payload = self._dedup_json("state_graph", msg)
        if payload is not None:
            self.latest_state_graph = payload

    def _on_robot_graph(self, msg: String) -> None:
        payload = self._dedup_json("robot_graph", msg)
        if payload is not None:
            self.latest_robot_graph = payload

    def _on_task_graph(self, msg: String) -> None:
        payload = self._dedup_json("task_graph", msg)
        if payload is None:
            return
        self.latest_task_graph = payload
        self.writer.block(
            self.phase,
            "TASK_GRAPH",
            payload,
            sim_s=self.sim_time(),
        )

    def _on_reconfiguration(self, msg: String) -> None:
        payload = self._dedup_json("reconfiguration", msg)
        if payload is None:
            return
        self.latest_reconfiguration = payload
        self.writer.block(
            self.phase,
            "RECONFIG_STATE",
            payload,
            sim_s=self.sim_time(),
        )

    def _on_actions(self, msg: String) -> None:
        payload = self._dedup_json("actions", msg)
        if payload is None:
            return
        self.latest_actions = payload
        expert = payload.get("expert")
        if isinstance(expert, Mapping):
            module_roles = expert.get("module_roles")
            if isinstance(module_roles, Mapping):
                self.role_by_module = {
                    str(module): str(role)
                    for module, role in module_roles.items()
                }

        self.writer.block(
            self.phase,
            "ACTIONS",
            {
                "source_hint": self.source_hint(),
                "payload": payload,
            },
            sim_s=self.sim_time(),
        )

    def _on_primitive_goal(self, msg: String) -> None:
        payload = self._dedup_json("primitive_goal:" + msg.data, msg)
        if payload is None:
            return
        goal_id = str(payload.get("goal_id", ""))
        if goal_id:
            self.goal_by_id[goal_id] = payload
            self.active_goal_ids.add(goal_id)

        self.writer.block(
            self.phase,
            "PRIMITIVE_GOAL",
            {
                "source_hint": self.source_hint(),
                "goal": payload,
                "structural_directives": structural_hold_fields(payload),
            },
            sim_s=self.sim_time(),
            echo=True,
        )

        hold_fields = structural_hold_fields(payload)
        hold_modules: set[str] = set()
        for key in (
            "structural_hold_module_ids",
            "stabilize_during_group_module_ids",
            "hold_after_group_module_ids",
            "passive_module_ids",
        ):
            value = hold_fields.get(key)
            if isinstance(value, (list, tuple)):
                hold_modules.update(str(item) for item in value)
        pusher = hold_fields.get("pusher_module_id")
        if pusher is not None:
            hold_modules.add(str(pusher))
        if hold_modules:
            self._log_angles_for_modules(
                "HOLD_ANGLES_AT_GOAL_DISPATCH",
                sorted(hold_modules),
            )

    def _status_items(self, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        statuses = payload.get("statuses")
        if isinstance(statuses, list):
            return [item for item in statuses if isinstance(item, dict)]
        if payload.get("goal_id") is not None:
            return [dict(payload)]
        return []

    def _on_primitive_status(self, msg: String) -> None:
        payload = self._dedup_json("primitive_status", msg)
        if payload is None:
            return

        changed_items = self._status_items(payload)
        self.writer.block(
            self.phase,
            "PRIMITIVE_STATUS",
            payload,
            sim_s=self.sim_time(),
        )
        terminal = {"succeeded", "failed", "canceled", "rejected"}
        for status in changed_items:
            goal_id = str(status.get("goal_id", ""))
            state = str(status.get("state", "")).lower()
            if not goal_id:
                continue
            if state in {"accepted", "running"}:
                self.active_goal_ids.add(goal_id)
            elif state in terminal:
                self.active_goal_ids.discard(goal_id)
                goal = self.goal_by_id.get(goal_id, {})
                holds = structural_hold_fields(goal)
                if state == "succeeded":
                    for module in holds.get(
                        "hold_after_group_module_ids",
                        (),
                    ):
                        self.persistent_declared_holds.add(str(module))
                for module in holds.get("passive_module_ids", ()):
                    self.persistent_declared_holds.discard(str(module))

    def _on_primitive_cancel(self, msg: String) -> None:
        payload = self._dedup_json("primitive_cancel:" + msg.data, msg)
        if payload is None:
            return
        self.writer.block(
            self.phase,
            "PRIMITIVE_CANCEL",
            {
                "source_hint": self.source_hint(),
                "payload": payload,
            },
            sim_s=self.sim_time(),
            echo=True,
        )

    def _on_module_states(self, msg: String) -> None:
        payload = self._dedup_json("module_states", msg)
        if payload is None:
            return
        self.latest_module_states = payload
        self._log_module_snapshot("STATE_SAMPLE", store=False)

    def _on_odom(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.odom = (float(p.x), float(p.y), float(yaw))
        now = time.monotonic()
        if now - self._last_odom_log >= 1.0:
            self._last_odom_log = now
            self.writer.line(
                self.phase,
                "ODOM",
                (
                    f"x={p.x:+.4f} y={p.y:+.4f} "
                    f"yaw_deg={math.degrees(yaw):+.3f}"
                ),
                sim_s=self.sim_time(),
            )

    def _on_cmd_vel(self, msg: Twist) -> None:
        command = (float(msg.linear.x), float(msg.angular.z))
        now = time.monotonic()
        changed = (
            self._last_cmd_vel is None
            or abs(command[0] - self._last_cmd_vel[0]) >= 0.005
            or abs(command[1] - self._last_cmd_vel[1]) >= 0.02
        )
        if changed or now - self._last_cmd_vel_log >= 0.5:
            self._last_cmd_vel_log = now
            self._last_cmd_vel = command
            self.writer.line(
                self.phase,
                "CMD_VEL",
                f"linear_x={command[0]:+.4f} angular_z={command[1]:+.4f}",
                sim_s=self.sim_time(),
            )

    def pose(self) -> tuple[float, float, float]:
        if self.odom is None:
            raise RuntimeError("/odom unavailable")
        return self.odom

    def _module_records(self) -> list[dict[str, Any]]:
        payload = self.latest_module_states or {}
        modules = payload.get("modules")
        if not isinstance(modules, list):
            return []
        return [item for item in modules if isinstance(item, dict)]

    def _module_angle_record(self, module: Mapping[str, Any]) -> dict[str, Any]:
        module_id = str(module.get("module_id", "?"))
        actuators = module.get("actuators")
        actuators = actuators if isinstance(actuators, Mapping) else {}
        pan = actuators.get("pan")
        tilt = actuators.get("tilt")
        pan = pan if isinstance(pan, Mapping) else {}
        tilt = tilt if isinstance(tilt, Mapping) else {}
        position = module.get("position")
        if not isinstance(position, (list, tuple)):
            position = ()
        role = (
            self.role_by_module.get(module_id)
            or str(module.get("current_role", module.get("role", "unassigned")))
        )

        def number(mapping: Mapping[str, Any], key: str) -> float | None:
            try:
                return float(mapping[key])
            except (KeyError, TypeError, ValueError):
                return None

        pan_rad = number(pan, "position_rad")
        tilt_rad = number(tilt, "position_rad")
        pan_vel = number(pan, "velocity_rad_s")
        tilt_vel = number(tilt, "velocity_rad_s")

        return {
            "module_id": module_id,
            "role": role,
            "pan_rad": pan_rad,
            "pan_deg": None if pan_rad is None else math.degrees(pan_rad),
            "pan_velocity_rad_s": pan_vel,
            "tilt_rad": tilt_rad,
            "tilt_deg": None if tilt_rad is None else math.degrees(tilt_rad),
            "tilt_velocity_rad_s": tilt_vel,
            "position_xyz_m": list(position[:3]) if len(position) >= 3 else None,
        }

    def active_structural_context(self) -> dict[str, Any]:
        active: dict[str, Any] = {}
        for goal_id in sorted(self.active_goal_ids):
            goal = self.goal_by_id.get(goal_id)
            if not isinstance(goal, Mapping):
                continue
            fields = structural_hold_fields(goal)
            if fields:
                active[goal_id] = fields
        return {
            "active_declared_hold_goals": active,
            "persistent_declared_hold_modules": sorted(
                self.persistent_declared_holds
            ),
            "note": (
                "active/persistent sets are reconstructed from public goal "
                "parameters. Full /mssr/actions payloads and measured joint "
                "angles are logged separately for correlation."
            ),
        }

    def _log_angles_for_modules(
        self,
        kind: str,
        module_ids: list[str],
    ) -> None:
        by_id = {
            str(module.get("module_id")): self._module_angle_record(module)
            for module in self._module_records()
        }
        selected = [by_id[module_id] for module_id in module_ids if module_id in by_id]
        self.writer.block(
            self.phase,
            kind,
            selected,
            sim_s=self.sim_time(),
        )

    def _log_module_snapshot(self, kind: str, *, store: bool) -> None:
        records = [
            self._module_angle_record(module)
            for module in self._module_records()
        ]
        payload = {
            "structural_hold_context": self.active_structural_context(),
            "modules": records,
        }
        self.writer.block(
            self.phase,
            kind,
            payload,
            sim_s=self.sim_time(),
        )
        if store:
            self.stage_snapshots[kind] = {
                record["module_id"]: record for record in records
            }

    def boundary_snapshot(self, name: str) -> None:
        # Spin briefly so the boundary captures a fresh physical sample.
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.05)
        records = {
            record["module_id"]: record
            for record in (
                self._module_angle_record(module)
                for module in self._module_records()
            )
        }
        self.stage_snapshots[name] = records
        self.writer.block(
            self.phase,
            f"BOUNDARY_SNAPSHOT:{name}",
            {
                "structural_hold_context": self.active_structural_context(),
                "modules": list(records.values()),
            },
            sim_s=self.sim_time(),
            echo=True,
        )

    def write_summary(self) -> None:
        self.writer.line(
            self.phase,
            "SUMMARY",
            "=" * 20 + " PAN/TILT BOUNDARY TIMELINE " + "=" * 20,
            sim_s=self.sim_time(),
            echo=True,
        )
        names = list(self.stage_snapshots)
        module_ids = sorted(
            {
                module_id
                for snapshot in self.stage_snapshots.values()
                for module_id in snapshot
            }
        )
        for module_id in module_ids:
            self.writer.line(
                self.phase,
                "SUMMARY",
                f"MODULE {module_id}",
                sim_s=self.sim_time(),
            )
            for name in names:
                record = self.stage_snapshots[name].get(module_id)
                if record is None:
                    continue
                self.writer.line(
                    self.phase,
                    "SUMMARY",
                    (
                        f"  {name:28s} role={record['role']:<22s} "
                        f"PAN={record['pan_deg']!s:>12} deg "
                        f"TILT={record['tilt_deg']!s:>12} deg"
                    ),
                    sim_s=self.sim_time(),
                )

        self.writer.line(
            self.phase,
            "SUMMARY",
            "Declared structural-hold parameters are searchable as "
            "STRUCTURAL_DIRECTIVES / HOLD_ANGLES_AT_GOAL_DISPATCH.",
            sim_s=self.sim_time(),
        )


def start_process(
    label: str,
    command: list[str],
    *,
    env: Mapping[str, str],
    writer: TimelineWriter,
    monitor: TimelineMonitor,
) -> subprocess.Popen[str]:
    writer.block(
        monitor.phase,
        f"PROC_START:{label}",
        {"command": command},
        sim_s=monitor.sim_time(),
        echo=True,
    )
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=dict(env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        start_new_session=True,
    )

    def pump() -> None:
        assert process.stdout is not None
        for raw in process.stdout:
            writer.line(
                monitor.phase,
                f"PROC:{label}",
                raw.rstrip(),
                sim_s=monitor.sim_time(),
            )

    threading.Thread(
        target=pump,
        name=f"timeline-{label}",
        daemon=True,
    ).start()
    return process


def spin_for(monitor: TimelineMonitor, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(monitor, timeout_sec=0.05)


def wait_until(
    monitor: TimelineMonitor,
    predicate,
    timeout_s: float,
    message: str,
    *,
    process: subprocess.Popen[str] | None = None,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"{message}: owning process exited rc={process.returncode}"
            )
        rclpy.spin_once(monitor, timeout_sec=0.05)
        if predicate():
            return
    raise TimeoutError(message)


def wait_process(
    monitor: TimelineMonitor,
    process: subprocess.Popen[str],
    timeout_s: float,
    label: str,
) -> int:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rclpy.spin_once(monitor, timeout_sec=0.05)
        rc = process.poll()
        if rc is not None:
            monitor.writer.line(
                monitor.phase,
                f"PROC_END:{label}",
                f"returncode={rc}",
                sim_s=monitor.sim_time(),
                echo=True,
            )
            return int(rc)
    raise TimeoutError(f"{label} wall timeout after {timeout_s:.1f}s")


def graph_connection_count(monitor: TimelineMonitor) -> int:
    payload = monitor.latest_robot_graph
    if not isinstance(payload, Mapping):
        return 0
    ga = payload.get("global_attributes")
    if isinstance(ga, Mapping):
        try:
            return int(ga.get("latched_connection_count", 0))
        except (TypeError, ValueError):
            pass
    edges = payload.get("edges")
    if not isinstance(edges, list):
        return 0
    return sum(
        1
        for edge in edges
        if isinstance(edge, Mapping)
        and edge.get("connection_state") == "latched"
    )


def wait_assembled(
    monitor: TimelineMonitor,
    process: subprocess.Popen[str],
) -> None:
    deadline = time.monotonic() + 360.0
    stable = 0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"assembly exited before stable topology rc={process.returncode}"
            )
        rclpy.spin_once(monitor, timeout_sec=0.10)
        stable = stable + 1 if graph_connection_count(monitor) == 7 else 0
        if stable >= 5:
            return
    raise TimeoutError("RC-Car8 assembly timeout")


def wait_nav2(
    monitor: TimelineMonitor,
    env: Mapping[str, str],
    timeout_s: float = 90.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    required = ("/bt_navigator", "/planner_server", "/controller_server")
    while time.monotonic() < deadline:
        rclpy.spin_once(monitor, timeout_sec=0.05)
        all_active = True
        for node in required:
            remaining = max(1.0, deadline - time.monotonic())
            try:
                result = subprocess.run(
                    ["ros2", "lifecycle", "get", node],
                    cwd=ROOT,
                    env=dict(env),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=min(5.0, remaining),
                )
            except subprocess.TimeoutExpired:
                all_active = False
                break
            if result.returncode != 0 or "active" not in result.stdout.lower():
                all_active = False
                break
        if all_active:
            return
        time.sleep(0.5)
    raise TimeoutError("Nav2 did not become ACTIVE")


def drive_curve_for_sim_time(
    monitor: TimelineMonitor,
    route_process: subprocess.Popen[str],
    *,
    curve_sim_s: float,
) -> None:
    """Let Nav2 excite the RC-Car PANs for a fixed simulation-time window.

    This diagnostic does not care whether Nav2 reaches the exact final
    pose.  We only need a genuine curved drive before reconfiguration.
    """

    wait_until(
        monitor,
        lambda: monitor.sim_time() is not None,
        20.0,
        "simulation clock unavailable before RC curve",
        process=route_process,
    )

    start_sim = monitor.sim_time()
    assert start_sim is not None

    monitor.writer.line(
        monitor.phase,
        "CURVE_TIMER",
        f"start_sim={start_sim:.3f} duration={curve_sim_s:.3f}s",
        sim_s=start_sim,
        echo=True,
    )

    last_sim = start_sim
    no_sim_progress_wall = time.monotonic()

    while True:
        rclpy.spin_once(monitor, timeout_sec=0.05)

        current_sim = monitor.sim_time()

        if current_sim is not None:
            if current_sim > last_sim + 1e-6:
                last_sim = current_sim
                no_sim_progress_wall = time.monotonic()

            elapsed_sim = current_sim - start_sim

            if elapsed_sim >= curve_sim_s:
                monitor.writer.line(
                    monitor.phase,
                    "CURVE_TIMER",
                    (
                        f"timer reached: elapsed_sim={elapsed_sim:.3f}s; "
                        "exact Nav2 goal intentionally ignored"
                    ),
                    sim_s=current_sim,
                    echo=True,
                )
                break

        # If the route process exits after motion has already begun,
        # that is acceptable for this diagnostic.  We still proceed.
        rc = route_process.poll()
        if rc is not None:
            current_sim = monitor.sim_time()
            elapsed_sim = (
                0.0
                if current_sim is None
                else current_sim - start_sim
            )

            monitor.writer.line(
                monitor.phase,
                "CURVE_ROUTE_EARLY_EXIT",
                (
                    f"route rc={rc}; elapsed_sim={elapsed_sim:.3f}s; "
                    "continuing diagnostic"
                ),
                sim_s=current_sim,
                echo=True,
            )

            # An immediate failure means we never exercised the RC-Car.
            if elapsed_sim < 1.0:
                raise RuntimeError(
                    "Nav2 route exited before producing at least "
                    "1.0 s of curved simulated motion"
                )
            break

        # Only a watchdog for a genuinely frozen simulation.
        if time.monotonic() - no_sim_progress_wall > 30.0:
            raise RuntimeError(
                "simulation time stopped advancing during RC curve"
            )

    # Explicit physical stop before shutting Nav2 down.
    zero = Twist()

    for _ in range(20):
        monitor.cmd_pub.publish(zero)
        rclpy.spin_once(monitor, timeout_sec=0.03)


def retreat_mm8(
    monitor: TimelineMonitor,
    distance_m: float = 0.10,
    speed_m_s: float = 0.035,
) -> None:
    wait_until(
        monitor,
        lambda: monitor.odom is not None,
        15.0,
        "/odom unavailable before MM8 retreat",
    )
    x0, y0, yaw0 = monitor.pose()
    c = math.cos(yaw0)
    s = math.sin(yaw0)
    start = time.monotonic()
    last_progress = -1.0
    last_improvement = start

    try:
        while time.monotonic() - start < 90.0:
            rclpy.spin_once(monitor, timeout_sec=0.03)
            x, y, _ = monitor.pose()
            backward = -((x - x0) * c + (y - y0) * s)
            if backward >= distance_m:
                monitor.writer.line(
                    monitor.phase,
                    "MM8_RETREAT",
                    f"target reached backward={backward:.4f} m",
                    sim_s=monitor.sim_time(),
                    echo=True,
                )
                return

            if backward > last_progress + 0.002:
                last_progress = backward
                last_improvement = time.monotonic()
            if time.monotonic() - last_improvement > 15.0:
                raise RuntimeError(
                    f"MM8 retreat stalled at {backward:.4f} m"
                )

            cmd = Twist()
            cmd.linear.x = -abs(speed_m_s)
            monitor.cmd_pub.publish(cmd)
            time.sleep(0.04)
    finally:
        stop = Twist()
        for _ in range(15):
            monitor.cmd_pub.publish(stop)
            rclpy.spin_once(monitor, timeout_sec=0.03)

    raise TimeoutError("MM8 retreat wall timeout")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Trace RC curve -> reconfiguration -> Scorpion -> manipulation."
    )
    p.add_argument("--headless", action="store_true")
    p.add_argument("--seed", type=int, default=7101)
    p.add_argument("--observe-s", type=float, default=5.0)
    p.add_argument(
        "--curve-sim-s",
        type=float,
        default=12.0,
        help=(
            "How many simulated seconds Nav2 is allowed to curve the "
            "RC-Car before the diagnostic forcibly continues."
        ),
    )
    return p


def main() -> int:
    args = parser().parse_args()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run = (
        ROOT
        / "logs"
        / "reconfiguration_timeline"
        / f"seed-{args.seed:06d}-{stamp}"
    )
    runtime_dir = run / "runtime"
    run.mkdir(parents=True, exist_ok=False)
    runtime_dir.mkdir(parents=True, exist_ok=False)
    timeline_path = run / "timeline.txt"

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["ROS_DOMAIN_ID"] = DOMAIN
    env["RMW_IMPLEMENTATION"] = "rmw_cyclonedds_cpp"
    env["ROS_LOG_DIR"] = str(run / "ros_logs")
    Path(env["ROS_LOG_DIR"]).mkdir()

    writer = TimelineWriter(timeline_path)
    runtime = None
    assembly = None
    nav2 = None
    route = None
    reconfig = None
    behavior_proc = None

    rclpy.init()
    monitor = TimelineMonitor(writer)

    writer.line(
        "BOOT",
        "INFO",
        f"timeline_path={timeline_path}",
        echo=True,
    )
    writer.line(
        "BOOT",
        "INFO",
        (
            "Diagnostic only: no IL dataset. Structural directives are logged "
            "verbatim from primitive goals together with measured PAN/TILT."
        ),
        echo=True,
    )

    try:
        # ------------------------------------------------------------
        # 1. EMPTY STAGE / RUNTIME
        # ------------------------------------------------------------
        monitor.set_phase("EMPTY_STAGE")
        runtime_cmd = [
            "ros2",
            "launch",
            "mssr_expert",
            "smores_runtime.launch.py",
            f"runtime_dir:={runtime_dir}",
            "module_count:=8",
            f"headless:={'true' if args.headless else 'false'}",
            "performance:=true",
            "simple_visuals:=true",
            f"simulation_steps:={240000 if args.headless else 0}",
            "simulation_speed_factor:=1.0",
            "actuator_effort_scale:=4.0",
            "wheel_friction_scale:=1.50",
            "tilt_effort_scale:=8.0",
            "behavior_control_rate_hz:=10.0",
            f"ros_domain_id:={DOMAIN}",
            "rmw_implementation:=rmw_cyclonedds_cpp",
        ]
        runtime = start_process(
            "runtime",
            runtime_cmd,
            env=env,
            writer=writer,
            monitor=monitor,
        )
        wait_until(
            monitor,
            lambda: monitor.latest_module_states is not None,
            240.0,
            "runtime state timeout",
            process=runtime,
        )
        monitor.boundary_snapshot("00_EMPTY_STAGE")

        # ------------------------------------------------------------
        # 2. RC-CAR8 ASSEMBLY
        # ------------------------------------------------------------
        monitor.set_phase("ASSEMBLY")
        assembly = start_process(
            "assembly",
            [
                "ros2",
                "run",
                "mssr_expert",
                "mssr_smores_self_assembly_node",
                "--ros-args",
                "-p",
                (
                    "target_graph_path:="
                    + str(
                        ROOT
                        / "mssr_ws/src/mssr_expert/config/smores_rc_car8.json"
                    )
                ),
                "-p",
                f"execution_id:=timeline-{args.seed}-assembly",
                "-p",
                f"episode_id:=timeline-{args.seed}",
                "-p",
                "dataset_path:=/dev/null",
            ],
            env=env,
            writer=writer,
            monitor=monitor,
        )
        wait_assembled(monitor, assembly)
        spin_for(monitor, 6.0)
        stop_process(assembly)
        assembly = None
        wait_until(
            monitor,
            lambda: monitor.odom is not None,
            20.0,
            "/odom unavailable after assembly",
        )
        monitor.boundary_snapshot("01_RC_CAR_ASSEMBLED")

        # ------------------------------------------------------------
        # 3. SHORT CURVED RC-CAR NAV2 ROUTE
        # ------------------------------------------------------------
        monitor.set_phase("RC_CURVE")
        x0, y0, yaw0 = monitor.pose()
        waypoints = build_curved_route(x0, y0, yaw0)
        route_path = run / "curve_route.json"
        route_payload = {
            "waypoints_xyyaw": [list(item) for item in waypoints],
            "corridor_width_m": 1.20,
            "cone_centers_xy_m": [],
            "cone_radius_m": 0.05,
        }
        route_path.write_text(
            json.dumps(route_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        writer.block(
            monitor.phase,
            "CURVE_ROUTE",
            {
                "start_xyyaw": [x0, y0, yaw0],
                **route_payload,
            },
            sim_s=monitor.sim_time(),
            echo=True,
        )

        nav2 = start_process(
            "nav2",
            [
                "ros2",
                "launch",
                "mssr_expert",
                "smores_nav2.launch.py",
                "autostart:=true",
                "log_level:=warn",
            ],
            env=env,
            writer=writer,
            monitor=monitor,
        )
        wait_nav2(monitor, env)

        route = start_process(
            "route",
            [
                sys.executable,
                str(ROOT / "scripts/smores_ep/run_rc_car_nav2_route.py"),
                "--seed",
                str(args.seed),
                "--route-json",
                str(route_path),
                "--action-timeout-s",
                "300",
                "--result-json",
                str(run / "curve_result.json"),
            ],
            env=env,
            writer=writer,
            monitor=monitor,
        )
        drive_curve_for_sim_time(
            monitor,
            route,
            curve_sim_s=args.curve_sim_s,
        )

        # Goal completion is intentionally irrelevant here.  Stop the
        # route and Nav2 after the requested amount of simulated motion.
        stop_process(route)
        route = None

        spin_for(monitor, 1.0)

        stop_process(nav2)
        nav2 = None

        spin_for(monitor, 1.0)
        monitor.boundary_snapshot("02_AFTER_RC_CURVE")

        # ------------------------------------------------------------
        # 4. RC-CAR8 -> MOBILEMANIPULATOR8
        # ------------------------------------------------------------
        monitor.set_phase("RECONFIG")
        monitor.latest_reconfiguration = None
        reconfig = start_process(
            "reconfiguration",
            [
                "ros2",
                "run",
                "mssr_expert",
                "mssr_smores_self_reconfiguration_node",
                "--ros-args",
                "-p",
                "source_graph_path:=auto",
                "-p",
                "target_morphology:=mobile_manipulator8",
                "-p",
                f"execution_id:=timeline-{args.seed}-reconfig",
                "-p",
                f"episode_id:=timeline-{args.seed}",
                "-p",
                "dataset_path:=/dev/null",
            ],
            env=env,
            writer=writer,
            monitor=monitor,
        )
        wait_until(
            monitor,
            lambda: (
                isinstance(monitor.latest_reconfiguration, Mapping)
                and monitor.latest_reconfiguration.get("done") is True
            ),
            600.0,
            "self-reconfiguration timeout",
            process=reconfig,
        )
        terminal = monitor.latest_reconfiguration or {}
        writer.block(
            monitor.phase,
            "RECONFIG_TERMINAL",
            terminal,
            sim_s=monitor.sim_time(),
            echo=True,
        )
        if terminal.get("success") is not True:
            raise RuntimeError("RC-Car8 -> MobileManipulator8 failed")
        stop_process(reconfig)
        reconfig = None

        # ------------------------------------------------------------
        # 5. OBSERVE UNTOUCHED SCORPION
        # ------------------------------------------------------------
        monitor.set_phase("SCORPION_OBSERVE")
        monitor.boundary_snapshot("03_SCORPION_ENTRY")
        spin_for(monitor, args.observe_s)
        monitor.boundary_snapshot("04_SCORPION_AFTER_OBSERVE")

        # ------------------------------------------------------------
        # 6. RETREAT 10 CM WHILE STILL FOLDED
        # ------------------------------------------------------------
        monitor.set_phase("MM8_RETREAT")
        retreat_mm8(monitor, distance_m=0.10, speed_m_s=0.035)
        spin_for(monitor, 2.0)
        monitor.boundary_snapshot("05_AFTER_MM8_RETREAT")

        # ------------------------------------------------------------
        # 7. SCORPION -> MANIPULATION READY
        # ------------------------------------------------------------
        monitor.set_phase("MANIPULATION_READY")
        behavior_proc = start_process(
            "prepare_manipulation",
            [
                "ros2",
                "run",
                "mssr_expert",
                "mssr_smores_morphology_command_client",
                "--morphology",
                "mobile_manipulator8",
                "--behavior",
                "prepare_manipulation",
                "--command-id",
                f"timeline-prepare-{time.time_ns()}",
                "--parameters-json",
                "{}",
                "--timeout-s",
                "180",
            ],
            env=env,
            writer=writer,
            monitor=monitor,
        )
        behavior_rc = wait_process(
            monitor,
            behavior_proc,
            210.0,
            "prepare_manipulation",
        )
        behavior_proc = None
        if behavior_rc != 0:
            raise RuntimeError(
                f"prepare_manipulation failed rc={behavior_rc}"
            )
        spin_for(monitor, args.observe_s)
        monitor.boundary_snapshot("06_MANIPULATION_READY")
        monitor.write_summary()

        writer.line(
            monitor.phase,
            "COMPLETE",
            f"diagnostic sequence complete; attach {timeline_path}",
            sim_s=monitor.sim_time(),
            echo=True,
        )
        print()
        print("============================================================")
        print(" DIAGNOSTIC COMPLETE")
        print("============================================================")
        print(f"Attach this file to ChatGPT:\n  {timeline_path}")
        print()
        if not args.headless:
            print("Isaac remains open for inspection. Close the GUI or Ctrl-C to exit.")
            return runtime.wait() if runtime is not None else 0
        return 0

    except KeyboardInterrupt:
        writer.line(
            monitor.phase,
            "INTERRUPTED",
            "KeyboardInterrupt",
            sim_s=monitor.sim_time(),
            echo=True,
        )
        try:
            monitor.write_summary()
        except Exception:
            pass
        return 130

    except Exception as error:
        writer.line(
            monitor.phase,
            "ERROR",
            f"{type(error).__name__}: {error}",
            sim_s=monitor.sim_time(),
            echo=True,
        )
        writer.block(
            monitor.phase,
            "TRACEBACK",
            traceback.format_exc(),
            sim_s=monitor.sim_time(),
        )
        try:
            monitor.boundary_snapshot("ERROR_BOUNDARY")
            monitor.write_summary()
        except Exception as summary_error:
            writer.line(
                monitor.phase,
                "SUMMARY_ERROR",
                repr(summary_error),
                sim_s=monitor.sim_time(),
            )
        print()
        print(f"Diagnostic failed; timeline retained at:\n  {timeline_path}")
        if not args.headless and runtime is not None and runtime.poll() is None:
            print("Isaac remains open for inspection. Close the GUI or Ctrl-C to exit.")
            try:
                runtime.wait()
            except KeyboardInterrupt:
                pass
        return 1

    finally:
        stop_process(route)
        stop_process(nav2)
        stop_process(assembly)
        stop_process(reconfig)
        stop_process(behavior_proc)
        if args.headless:
            stop_process(runtime)
        monitor.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
