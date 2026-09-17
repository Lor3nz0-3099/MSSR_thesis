"""Interactive T3 gate: existing Isaac assembly scene and real DualSense.

The shipped mapping uses user-approved Circle HOME and Triangle stop/resume.
Preflight rejects incomplete bindings before cleanup or launch. Evidence is written separately
from any training dataset; software tests cannot mark this acceptance passed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "mssr_ws/src/mssr_expert/config"
sys.path[:0] = [str(ROOT / "mssr_ws/src/mssr_expert"), str(ROOT / "scripts/smores_ep/src")]


def preflight(path):
    from mssr_expert.teleop.input import load_input_config
    config = load_input_config(path)
    missing = ["home"] if config.commands.get("home") is None else []
    if config.commands.get("estop_toggle") is None:
        missing.extend(name for name in ("estop", "resume") if config.commands.get(name) is None)
    if missing:
        raise ValueError("deferred physical bindings: " + ", ".join(missing) +
                         "; use --input-config with user-approved assignments")
    return config


def physical_metrics(payload, target):
    """Use detected role assignment and observed physical pose/joints only."""
    from mssr_expert.graph.serialization import attributed_graph_from_dict
    from mssr_expert.teleop.rc_car import RcCarObservation
    graph = attributed_graph_from_dict(payload)
    observation = RcCarObservation.from_graph(graph, target)
    if observation is None:
        raise ValueError("physical topology is not rc_car8")
    nodes = graph.node_by_id()
    positions = {item.target_role: nodes[item.module_id].attributes["position"] for item in observation.assignments}
    center = [sum(point[index] for role, point in positions.items() if role.startswith("chassis_")) / 4
              for index in range(3)]
    direction = [sum(point[index] for role, point in positions.items() if role.startswith("wheel_right_")) / 2 -
                 sum(point[index] for role, point in positions.items() if role.startswith("wheel_left_")) / 2
                 for index in range(2)]
    norm = math.hypot(*direction)
    if norm < 1e-6:
        raise ValueError("degenerate physical RC heading")
    tilt = {item.module_id: nodes[item.module_id].attributes["actuators"]["tilt"]["position_rad"]
            for item in observation.assignments if item.target_role.startswith("wheel_")}
    if not all(math.isfinite(value) for value in (*center, *direction, *tilt.values())):
        raise ValueError("nonfinite physical RC observation")
    return {"center": center, "forward": [value / norm for value in direction],
            "heading": math.atan2(direction[1], direction[0]), "tilt": tilt, "stamp": graph.stamp}


class HeldHeightProbe:
    def __init__(self):
        self.height = None
        self.since = None

    def observe(self, status, *, now):
        sample = status["controller_input"]
        if any(sample[name] != 0 for name in ("right_y", "l2", "r2")):
            self.height = self.since = None
            return False
        height = status["rc_car_intent"]["chassis_height_m"]
        if self.height is None:
            self.height, self.since = height, now
        return now - self.since >= 1 and abs(height - self.height) < 1e-9


def runtime_commands(output, input_path, run_id, device_id):
    from check_dualsense import build_driver_command
    action = output / "actions.json"
    goal, cancel, status = [output / name for name in ("goal.json", "cancel.json", "primitive_status.json")]
    topic = f"/mssr/teleop_probe/run_{run_id}"
    return {
        "isaac": ["bash", "scripts/smores_ep/run_self_assembly.sh", "--module-count", "8",
                  "--performance", "--physics-hz", "240", "--actuator-effort-scale", "4.0",
                  "--tilt-effort-scale", "8.0", "--action-file", str(action),
                  "--primitive-goal-file", str(goal), "--primitive-cancel-file", str(cancel),
                  "--primitive-status-file", str(status)],
        "bridge": [sys.executable, "ros2_bridge/mssr_file_bridge.py", "--state-graph-dir", str(output),
                   "--action-file", str(action), "--primitive-goal-file", str(goal),
                   "--primitive-cancel-file", str(cancel), "--primitive-status-file", str(status)],
        "assembly": ["ros2", "run", "mssr_expert", "mssr_smores_self_assembly_node", "--ros-args",
                     "-p", f"target_graph_path:={CONFIG / 'smores_rc_car8.json'}",
                     "-p", f"dataset_path:={output / 'assembly.jsonl'}", "-p", f"execution_id:=t3-{run_id}"],
        "joy": build_driver_command(device_id, topic + "/joy"),
        "teleop": ["ros2", "launch", "mssr_expert", "smores_teleop.launch.py", "start_joy:=false",
                   f"joy_topic:={topic}/joy", f"status_topic:={topic}/status",
                   f"node_name:=mssr_rc_teleop_{run_id}",
                   f"input_config_path:={input_path.resolve()}", f"teleop_config_path:={CONFIG / 'smores_teleop.yaml'}"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-config", type=Path, default=CONFIG / "smores_dualsense.yaml")
    parser.add_argument("--assembly-timeout", type=float, default=900)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    try:
        input_config = preflight(args.input_config)
        if not math.isfinite(args.assembly_timeout) or args.assembly_timeout <= 0:
            raise ValueError("assembly timeout must be positive and finite")
        if args.device_id < 0:
            raise ValueError("device ID must be nonnegative")
    except (OSError, ValueError) as error:
        print(f"T3 hardware preflight: {error}", file=sys.stderr)
        return 2

    # All external effects occur only after the physical mapping preflight.
    from check_dualsense import configure_probe_environment, parse_devices
    from runtime_cleanup import scoped_cleanup
    from mssr_expert.graph.serialization import load_attributed_graph
    import rclpy
    from std_msgs.msg import String

    output = ROOT / "logs/teleop/rc_car_checks" / uuid4().hex
    configure_probe_environment(os.environ, output)
    output.mkdir(parents=True)
    commands = runtime_commands(output, args.input_config, output.name, args.device_id)
    summary = {"passed": False, "source": "real_dualsense_and_native_isaac", "checks": {},
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    processes = []
    observer = None
    latest = {"status": None, "assembly": None}
    target = load_attributed_graph(CONFIG / "smores_rc_car8.json")
    records = (output / "observations.jsonl").open("w")

    def launch(name, command):
        with (output / f"{name}.log").open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        processes.append((name, process))
        return process

    def receive(kind, message):
        payload = json.loads(message.data)
        latest[kind] = payload
        records.write(json.dumps({"received_at": time.monotonic(), "kind": kind, "payload": payload}, allow_nan=False) + "\n")

    def read(name):
        try:
            return json.loads((output / name).read_text())
        except (OSError, ValueError):
            return {}

    def wait(label, instruction, predicate, timeout=30):
        print(f"[{label}] {instruction}", flush=True)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for name, process in processes:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited {process.returncode}; inspect {output / (name + '.log')}")
            rclpy.spin_once(observer, timeout_sec=0.02)
            if predicate():
                summary["checks"][label] = True
                return
        raise TimeoutError(f"{label} did not produce observed acceptance evidence")

    def state():
        status = latest["status"]
        if status is None or not 0 <= time.monotonic() - status["stamp_monotonic"] <= 0.5:
            raise ValueError("stale teleop status")
        return status

    def metrics():
        return physical_metrics(read("robot_graph.json"), target)

    def zero():
        commands = state()["rc_car_effective_actions"]
        return bool(commands) and all(abs(command["pan_rate_rad_s"]) < 1e-12 for command in commands.values())

    def drive(direction):
        before = metrics()
        trigger = "r2" if direction == 1 else "l2"
        def observed():
            current = metrics()
            displacement = sum((current["center"][index] - before["center"][index]) * before["forward"][index]
                               for index in range(2))
            return (state()["controller_input"][trigger] > 0.15 and not zero() and direction * displacement > 0.01)
        return observed

    try:
        summary["cleanup"] = scoped_cleanup(ROOT)
        enumeration = subprocess.run(["/opt/ros/humble/lib/joy/joy_enumerate_devices"],
                                     capture_output=True, text=True, timeout=15)
        summary["enumeration"] = enumeration.stdout
        device = next((item for item in parse_devices(enumeration.stdout) if item["device_id"] == args.device_id), None)
        if enumeration.returncode or device is None or not device["gamepad"] or not device["dualsense"]:
            raise RuntimeError("selected Sony DualSense not detected")
        summary["device"] = device
        rclpy.init(args=[])
        observer = rclpy.create_node("mssr_rc_car_acceptance")
        observer.create_subscription(String, f"/mssr/teleop_probe/run_{output.name}/status", lambda message: receive("status", message), 100)
        observer.create_subscription(String, "/mssr/expert/self_assembly/state", lambda message: receive("assembly", message), 100)
        launch("isaac", commands["isaac"])
        launch("bridge", commands["bridge"])
        wait("native_ready", "Attendi Isaac; lascia il controller a riposo.",
             lambda: 0 <= time.monotonic() - read("smores_teleop_runtime_status.json").get("stamp_monotonic", -1e12) <= 0.5,
             timeout=240)
        assembly = launch("assembly", commands["assembly"])
        def assembled():
            status = latest["assembly"]
            if status and status.get("done") and not status.get("success"):
                raise RuntimeError("existing RC assembly failed: " + status.get("message", ""))
            return bool(status and status.get("done") and status.get("success"))
        wait("assembly", "Attendi la self-assembly RC-Car8 esistente.", assembled, timeout=args.assembly_timeout)
        assembly.terminate()
        assembly.wait(timeout=5)
        processes[:] = [(name, process) for name, process in processes if process is not assembly]
        metrics()  # Verify detected physical topology independently of success.
        launch("joy", commands["joy"])
        launch("teleop", commands["teleop"])
        wait("neutral", "DualSense reale: L2/R2 e stick a riposo.",
             lambda: latest["status"] is not None and state()["safety"]["motion_enabled"] and zero())
        wait("forward", "Premi lentamente R2 fino a circa 1/3; stick neutri.", drive(1))
        wait("release", "Rilascia L2/R2: verifica arresto.", lambda: state()["controller_input"]["r2"] == 0 and zero())
        wait("reverse", "Premi lentamente L2 fino a circa 1/3; stick neutri.", drive(-1))
        wait("release_reverse", "Rilascia L2/R2.", lambda: state()["controller_input"]["l2"] == 0 and zero())
        for sign, label in ((1, "steer_right"), (-1, "steer_left")):
            before = metrics()["heading"]
            def turned():
                delta = metrics()["heading"] - before
                return (sign * state()["controller_input"]["right_x"] > 0.4 and
                        state()["controller_input"]["r2"] > 0.1 and sign * math.atan2(math.sin(delta), math.cos(delta)) < -0.03)
            wait(label, f"R2 circa 1/3 e right-X {'destra' if sign == 1 else 'sinistra'}.", turned)
            wait(label + "_release", "Mantieni right-X e rilascia entrambi i trigger.", zero)
        for sign, label in ((1, "raise"), (-1, "lower")):
            before = metrics()
            def reshaped():
                after = metrics()
                return (sign * state()["controller_input"]["right_y"] > 0.5 and zero() and
                        sign * (after["center"][2] - before["center"][2]) > 0.0003 and
                        any(-sign * (after["tilt"][module] - angle) > 0.01 for module, angle in before["tilt"].items()))
            wait(label, f"Trigger neutri, right-Y {'su' if sign == 1 else 'giù'}; right-X neutro.", reshaped)
        held = HeldHeightProbe()
        wait("held_height", "Rilascia right-Y e mantieni tutti i controlli neutri.",
             lambda: held.observe(state(), now=time.monotonic()))
        seen_home = False
        def homed():
            nonlocal seen_home
            seen_home |= "home" in state()["controller_input"]["command_events"]
            return seen_home and all(abs(angle + 0.785398) < 0.01 for angle in metrics()["tilt"].values())
        home_button = input_config.commands["home"]
        stop_button = input_config.commands.get("estop_toggle") or input_config.commands["estop"]
        resume_button = input_config.commands.get("estop_toggle") or input_config.commands["resume"]
        wait("home", f"Premi {home_button} (HOME); attendi il ritorno graduale.", homed)
        wait("pre_stop_drive", "Premi R2 circa 1/3.", drive(1))
        wait("estop", f"Mantieni R2 e premi {stop_button} (E-stop).",
             lambda: state()["safety"]["authority"] == "ESTOP" and state()["runtime_structure_stopped"] and zero())
        stopped = metrics()
        since = time.monotonic()
        wait("stop_physics_hold", "Resta in E-stop e muovi left stick: camera continua.",
             lambda: time.monotonic() - since > 1 and metrics()["stamp"] > stopped["stamp"] and
                     any(abs(state()["controller_input"][name]) > 0.3 for name in ("left_x", "left_y")) and
                     read("smores_teleop_runtime_status.json").get("structure_stopped") is True and
                     read("smores_teleop_runtime_status.json").get("camera_applied") is True and zero() and
                     all(abs(metrics()["tilt"][module] - angle) < 0.02 for module, angle in stopped["tilt"].items()))
        wait("resume_held_fence", f"Mantieni R2 premuto, rilascia e premi {resume_button} (RESUME).",
             lambda: state()["runtime_structure_stopped"] is False and state()["controller_input"]["r2"] > 0.1 and
                     not state()["safety"]["motion_enabled"] and zero())
        wait("fresh_neutral", "Rilascia trigger e stick: input nuovo e neutro per rearm.",
             lambda: state()["safety"]["motion_enabled"] and state()["controller_input"]["l2"] == 0 and
                     state()["controller_input"]["r2"] == 0 and zero())
        wait("post_resume_forward", "Premi di nuovo R2 circa 1/3.", drive(1))
        wait("final_zero", "Rilascia tutti i controlli.", zero)
        summary["passed"] = True
    except (Exception, KeyboardInterrupt) as error:
        summary["error"] = repr(error)
    finally:
        records.close()
        if observer is not None:
            observer.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        try:
            # UID, checkout cwd, argv and PID start-time scope; TERM/wait,
            # KILL only pertinent survivors, daemon stop, final verification.
            summary["final_cleanup"] = scoped_cleanup(ROOT)
        except Exception as error:
            summary["passed"] = False
            summary["cleanup_error"] = repr(error)
        (output / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print("T3_RC_RESULT=" + json.dumps(summary), flush=True)
    print(f"REPORT={output / 'report.json'}", flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
