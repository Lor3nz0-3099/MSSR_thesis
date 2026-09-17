"""T0 hardware evidence using ROS joy; no robot actuator publisher."""
from __future__ import annotations

import argparse
from collections.abc import MutableMapping
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time
from uuid import uuid4

from mssr_expert.teleop.input import DualSenseInput, InputConfig, STICKS, TRIGGERS, load_input_config


ROOT = Path(__file__).resolve().parents[2]
PHASES = (
    ("neutral", 5.0, "Lascia tutti gli stick e R2/L2 a riposo."),
    ("travel", 25.0, "Muovi ENTRAMBI gli stick in tutte le direzioni fino al fondo; premi R2 e L2 lentamente fino al fondo e rilascia."),
    ("start", 10.0, "Premi e rilascia OPTIONS/START due volte."),
    ("disconnect", 15.0, "Disconnetti il controller (USB: stacca il cavo; Bluetooth: spegnilo). Lascialo disconnesso fino alla fase successiva."),
    ("reconnect", 20.0, "Riconnetti il controller con stick e trigger a riposo."),
    ("final_neutral", 5.0, "Lascia tutti i controlli a riposo."),
)


class ProbeReport:
    """Aggregate observed evidence; zero samples and missing phases cannot pass."""

    def __init__(self, config: InputConfig) -> None:
        self.input = DualSenseInput(config)
        self.count = 0
        self.invalid = 0
        self.ranges: dict[str, list[float]] = {}
        self.raw_ranges: list[list[float]] = []
        self.analog = set()
        self.neutral_phases = set()
        self.neutral_since: dict[str, float | None] = {}
        self.record_edges = 0
        self.connected = False
        self.disconnected_in_phase = False
        self.reconnected_in_phase = False
        self.events: list[dict] = []

    def _connectivity(self, connected: bool, now: float, phase: str) -> None:
        if not connected:
            self.neutral_since[phase] = None
        if connected != self.connected:
            event = "controller_connected" if connected else "controller_disconnected"
            self.events.append({"event": event, "received_at": now, "phase": phase})
            if not connected and phase == "disconnect":
                self.disconnected_in_phase = True
            if connected and phase == "reconnect" and self.disconnected_in_phase:
                self.reconnected_in_phase = True
            self.connected = connected

    def accept(self, axes, buttons, received_at: float, phase: str):
        self.poll(received_at, phase)
        if not self.input.update(axes, buttons, received_at):
            self.invalid += 1
            return None
        sample = self.input.snapshot(received_at)
        self.count += 1
        self._connectivity(sample.connected, received_at, phase)
        for index, value in enumerate(axes):
            value = float(value)
            if index >= len(self.raw_ranges):
                self.raw_ranges.append([value, value])
            else:
                self.raw_ranges[index][0] = min(self.raw_ranges[index][0], value)
                self.raw_ranges[index][1] = max(self.raw_ranges[index][1], value)
        values = {name: getattr(sample, name) for name in (*STICKS, *TRIGGERS)}
        if phase in {"neutral", "final_neutral"}:
            if all(value == 0 for value in values.values()) and not sample.buttons:
                if self.neutral_since.get(phase) is None:
                    self.neutral_since[phase] = received_at
            else:
                self.neutral_since[phase] = None
        if phase == "travel":
            for name, value in values.items():
                bounds = self.ranges.setdefault(name, [value, value])
                bounds[0] = min(bounds[0], value)
                bounds[1] = max(bounds[1], value)
                if name in TRIGGERS and 0.15 <= value <= 0.85:
                    self.analog.add(name)
        if phase == "start":
            self.record_edges += sample.command_events.count("record_toggle")
        return sample

    def poll(self, now: float, phase: str) -> None:
        self._connectivity(self.input.snapshot(now).connected, now, phase)

    def end_phase(self, now: float, phase: str) -> None:
        self.poll(now, phase)
        since = self.neutral_since.get(phase)
        if phase in {"neutral", "final_neutral"}:
            self.neutral_phases.discard(phase)
            if self.connected and since is not None and now - since >= 0.5:
                self.neutral_phases.add(phase)

    def summary(self) -> dict:
        checks = {
            "joy_observed": self.count > 0,
            "initial_neutral": "neutral" in self.neutral_phases,
            "stick_travel": all(self.ranges.get(name, [0, 0])[0] <= -0.8 and
                                self.ranges.get(name, [0, 0])[1] >= 0.8 for name in STICKS),
            "trigger_travel": all(self.ranges.get(name, [0, 0])[0] <= 0.05 and
                                  self.ranges.get(name, [0, 0])[1] >= 0.95 for name in TRIGGERS),
            "trigger_analog": self.analog == set(TRIGGERS),
            "start_edges": self.record_edges >= 2,
            "disconnect": self.disconnected_in_phase,
            "reconnect": self.reconnected_in_phase,
            "final_neutral": "final_neutral" in self.neutral_phases,
            "no_invalid_packets": self.invalid == 0,
        }
        return {"checks": checks, "input_checks_passed": all(checks.values()),
                "valid_packets": self.count, "invalid_packets": self.invalid,
                "record_toggle_edges": self.record_edges,
                "raw_axes_range": self.raw_ranges, "normalized_travel": self.ranges,
                "events": self.events}


def parse_devices(text: str) -> list[dict]:
    devices = []
    for line in text.splitlines():
        match = re.fullmatch(r"\s*(\d+)\s*:\s*([a-fA-F0-9]{32})\s*:\s*(true|false)\s*:\s*(true|false)\s*:\s*(.+)", line)
        if match:
            index, guid, gamepad, mapped, name = match.groups()
            vendor = int.from_bytes(bytes.fromhex(guid[8:12]), "little")
            product = int.from_bytes(bytes.fromhex(guid[16:20]), "little")
            devices.append({"device_id": int(index), "guid": guid, "name": name,
                            "gamepad": gamepad == "true", "mapped": mapped == "true",
                            "dualsense": vendor == 0x054c and product in {0x0ce6, 0x0df2}})
    return devices


def _json_default(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"unsupported report value: {type(value).__name__}")


def build_driver_command(device_id: int, topic: str) -> list[str]:
    if not topic.startswith("/mssr/teleop_probe/run_") or not topic.endswith("/joy"):
        raise ValueError("the probe requires an exclusive per-run Joy topic")
    return ["/opt/ros/humble/lib/joy/game_controller_node", "--ros-args",
            "-r", "joy:=" + topic, "-p", f"device_id:={device_id}",
            "-p", "deadzone:=0.0", "-p", "autorepeat_rate:=50.0",
            "-p", "sticky_buttons:=false"]


def configure_probe_environment(environment: MutableMapping[str, str], output: Path) -> None:
    """Validate DDS port range before cleanup or middleware initialization."""
    value = environment.get("ROS_DOMAIN_ID", "42")
    # Humble rcl uses strtoul(base=0): leading zeros are octal, unlike Python int.
    if not isinstance(value, str) or re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise ValueError("ROS_DOMAIN_ID must be canonical ASCII decimal in [0,232]")
    try:
        domain = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError("ROS_DOMAIN_ID must be an integer in [0,232]") from error
    if not 0 <= domain <= 232:
        raise ValueError(f"ROS_DOMAIN_ID={value} exceeds DDS port limits; use an integer in [0,232]")
    environment.setdefault("ROS_DOMAIN_ID", "42")
    environment["ROS_LOG_DIR"] = str(output / "ros_logs")


def run(args, output: Path, metadata: dict) -> tuple[dict, int]:
    from runtime_cleanup import scoped_cleanup
    metadata["cleanup"] = scoped_cleanup(ROOT)
    enumeration = subprocess.run(["/opt/ros/humble/lib/joy/joy_enumerate_devices"],
                                 capture_output=True, text=True, timeout=15, check=False)
    metadata["enumeration_stdout"] = enumeration.stdout
    metadata["enumeration_stderr"] = enumeration.stderr
    metadata["enumeration_returncode"] = enumeration.returncode
    metadata["devices"] = parse_devices(enumeration.stdout)
    print(enumeration.stdout, flush=True)
    device = next((item for item in metadata["devices"] if item["device_id"] == args.device_id), None)
    report = ProbeReport(load_input_config(args.config))
    if enumeration.returncode or device is None or not device["gamepad"] or not device["dualsense"]:
        metadata["terminal_reason"] = "selected_dualsense_not_detected"
        return report.summary(), 2
    if args.preflight_only:
        metadata["terminal_reason"] = "hardware_acceptance_pending"
        return report.summary(), 2

    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Joy
    rclpy.init(args=[])
    node = rclpy.create_node("mssr_dualsense_probe")
    phase = "waiting"
    process = None
    try:
        with (output / "joy_raw_normalized.jsonl").open("x", encoding="utf-8") as raw_log, (
            output / "game_controller.log"
        ).open("x", encoding="utf-8") as driver_log:
            def callback(message):
                stamp = time.monotonic()
                sample = report.accept(message.axes, message.buttons, stamp, phase)
                row = {"phase": phase, "received_at": stamp,
                       "joy_stamp": {"sec": message.header.stamp.sec, "nanosec": message.header.stamp.nanosec},
                       "axes": list(message.axes), "buttons": list(message.buttons),
                       "valid": sample is not None,
                       "normalized": asdict(sample) if sample is not None else None}
                raw_log.write(json.dumps(row, default=_json_default, allow_nan=False) + "\n")

            node.create_subscription(Joy, metadata["joy_topic"], callback, qos_profile_sensor_data)
            command = build_driver_command(args.device_id, metadata["joy_topic"])
            metadata["driver_command"] = command
            process = subprocess.Popen(command, cwd=ROOT, stdout=driver_log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 10.0
            while report.count == 0 and time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"game_controller_node exited {process.returncode}")
                rclpy.spin_once(node, timeout_sec=0.05)
            if report.count == 0:
                metadata["terminal_reason"] = "joy_timeout"
                return report.summary(), 2
            for phase, duration, instruction in PHASES:
                print(f"\n[{phase}: {duration:.0f}s] {instruction}", flush=True)
                deadline = time.monotonic() + duration
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise RuntimeError(f"game_controller_node exited {process.returncode}")
                    rclpy.spin_once(node, timeout_sec=0.05)
                    report.poll(time.monotonic(), phase)
                report.end_phase(time.monotonic(), phase)
            result = report.summary()
            metadata["terminal_reason"] = "passed" if result["input_checks_passed"] else "input_checks_failed"
            return result, 0 if result["input_checks_passed"] else 1
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        node.destroy_node()
        rclpy.shutdown()
        metadata["driver_stopped"] = process is None or process.poll() is not None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--config", type=Path, default=ROOT / "mssr_ws/src/mssr_expert/config/smores_dualsense.yaml")
    parser.add_argument("--preflight-only", action="store_true",
                        help="cleanup and enumerate devices without starting timed input validation")
    args = parser.parse_args()
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = ROOT / "logs/teleop/hardware_checks" / now
    try:
        configure_probe_environment(os.environ, output)
    except ValueError as error:
        parser.error(str(error))
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"schema": "mssr.teleop_input_verification.v1", "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "started_at": now,
        "ros_domain_id": os.environ["ROS_DOMAIN_ID"],
        "joy_topic": "/mssr/teleop_probe/run_" + uuid4().hex + "/joy",
        "input_config": args.config.read_text(encoding="utf-8")}
    result = {}
    try:
        result, exit_code = run(args, output, metadata)
    except (Exception, KeyboardInterrupt) as error:
        metadata["terminal_reason"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "runtime_error"
        metadata["error"] = repr(error)
        exit_code = 2
    metadata.update(result)
    metadata["passed"] = exit_code == 0
    metadata["ended_at"] = datetime.now(timezone.utc).isoformat()
    path = output / "report.json"
    path.write_text(json.dumps(metadata, indent=2, default=_json_default, allow_nan=False) + "\n", encoding="utf-8")
    print("T0_HARDWARE_RESULT=" + json.dumps({"passed": metadata["passed"],
          "terminal_reason": metadata["terminal_reason"], "checks": result.get("checks", {})}), flush=True)
    print(f"REPORT={path}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
