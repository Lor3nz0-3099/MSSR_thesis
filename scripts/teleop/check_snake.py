"""Interactive T6a gate: direct Snake8 assembly and real DualSense.

Only native physical observations can pass this gate. It leaves a report and
logs even when a phase fails, and cleans only scoped MSSR runtime processes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from uuid import uuid4

from check_rc_car import CONFIG, ROOT, finalize_runtime, preflight, runtime_commands


def snake_runtime_commands(output, input_path, run_id, device_id, *, runtime_dir):
    commands = runtime_commands(output, input_path, run_id, device_id,
                                runtime_dir=runtime_dir)
    commands["assembly"] = [
        "ros2", "run", "mssr_expert", "mssr_smores_self_assembly_node", "--ros-args",
        "-p", f"target_graph_path:={CONFIG / 'smores_snake8.json'}",
        "-p", f"dataset_path:={output / 'assembly.jsonl'}",
        "-p", f"execution_id:=t6a-{run_id}",
    ]
    commands["teleop"] = [part.replace("mssr_rc_teleop_", "mssr_snake_teleop_")
                          for part in commands["teleop"]]
    return commands


def physical_metrics(payload, target):
    from mssr_expert.graph.serialization import attributed_graph_from_dict
    from mssr_expert.teleop.snake import SnakeObservation

    graph = attributed_graph_from_dict(payload)
    observation = SnakeObservation.from_graph(graph, target)
    if observation is None:
        raise ValueError("physical topology is not a valid Snake8")
    nodes = graph.node_by_id()
    by_role = {item.target_role: item.module_id for item in observation.assignments}
    head = tuple(float(v) for v in nodes[by_role["snake_head"]].attributes["pose"]["position"])
    neck = tuple(float(v) for v in nodes[by_role["snake_neck"]].attributes["pose"]["position"])
    tangent = tuple(head[i] - neck[i] for i in range(3))
    planar = math.hypot(*tangent[:2])
    if planar <= 1e-6:
        raise ValueError("degenerate observed Snake head tangent")
    pan = {item.module_id: float(nodes[item.module_id].attributes["actuators"]["pan"]["position_rad"])
           for item in observation.assignments}
    tilt = {item.module_id: float(nodes[item.module_id].attributes["actuators"]["tilt"]["position_rad"])
            for item in observation.assignments}
    if not all(math.isfinite(v) for v in (*head, *neck, *pan.values(), *tilt.values())):
        raise ValueError("nonfinite physical Snake observation")
    return {"stamp": graph.stamp, "head": head, "neck": neck,
            "forward": (tangent[0] / planar, tangent[1] / planar),
            "pan": pan, "tilt": tilt, "roles": by_role}


def wheel_invariant(actions):
    """The native differential-drive parser gives equal wheels for yaw=0."""
    return bool(actions) and all(
        math.isfinite(float(command.get("vx", 0.0)))
        and abs(float(command.get("vy", 0.0))) < 1e-9
        and abs(float(command.get("yaw_rate", 0.0))) < 1e-9
        for command in actions.values()
    )


class OppositeWheelWindow:
    """Require sustained reverse of every forward wheel, not one lucky tick."""
    def __init__(self, forward_commands, *, hold_s=0.4):
        self.forward = {module: float(command.get("vx", 0.0))
                        for module, command in forward_commands.items()}
        self.hold_s = hold_s
        self.first_at = None
        self.last_at = None
        self.confirmed = False

    def observe(self, stamp, commands):
        if self.last_at == stamp:
            return self.confirmed
        if self.last_at is not None and stamp < self.last_at:
            self.first_at = None
        self.last_at = stamp
        opposite = bool(self.forward) and set(commands) == set(self.forward) and all(
            math.isfinite(float(commands[module].get("vx", 0.0)))
            and abs(reference) > 0.005
            and abs(float(commands[module].get("vx", 0.0))) > 0.005
            and float(commands[module]["vx"]) * reference < 0
            for module, reference in self.forward.items()
        )
        if not opposite:
            self.first_at = None
            self.confirmed = False
            return False
        if self.first_at is None:
            self.first_at = stamp
        self.confirmed = stamp - self.first_at >= self.hold_s
        return self.confirmed


def recorded_snake_actions(path):
    rows = wheel_rows = shape_rows = manual_pan_rows = 0
    if not path.is_file():
        return {"rows": 0, "wheel_rows": 0, "shape_rows": 0, "manual_pan_rows": 0}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("task_type") != "snake8_teleop":
            continue
        rows += 1
        actions = row.get("expert_action", {}).get("locomotion", {})
        if any(abs(float(command.get("vx", 0.0))) > 1e-5 for command in actions.values()):
            wheel_rows += 1
        if any("tilt_target_rad" in command for command in actions.values()):
            shape_rows += 1
        selected = row.get("observation", {}).get("intent", {}).get("selected_module_id")
        if (row.get("observation", {}).get("intent", {}).get("control_mode") == "single_module"
                and selected in actions and "pan_target_rad" in actions[selected]):
            manual_pan_rows += 1
    return {"rows": rows, "wheel_rows": wheel_rows, "shape_rows": shape_rows,
            "manual_pan_rows": manual_pan_rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-config", type=Path, default=CONFIG / "smores_dualsense.yaml")
    parser.add_argument("--assembly-timeout", type=float, default=900)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()
    try:
        mapping = preflight(args.input_config)
        if mapping.commands.get("record_toggle") is None:
            raise ValueError("recording button must be configured")
        if args.device_id < 0 or not math.isfinite(args.assembly_timeout) or args.assembly_timeout <= 0:
            raise ValueError("device ID and timeouts are invalid")
    except (OSError, ValueError) as error:
        print(f"T6a hardware preflight: {error}", file=sys.stderr)
        return 2

    from check_dualsense import configure_probe_environment, parse_devices
    from runtime_cleanup import scoped_cleanup
    from mssr_expert.graph.serialization import load_attributed_graph
    import rclpy
    from std_msgs.msg import String

    output = ROOT / "logs/teleop/snake_checks" / uuid4().hex
    configure_probe_environment(os.environ, output)
    output.mkdir(parents=True)
    runtime_dir = Path(tempfile.mkdtemp(prefix=f"mssr-snake-{output.name}-", dir="/dev/shm"))
    commands = snake_runtime_commands(output, args.input_config, output.name,
                                      args.device_id, runtime_dir=runtime_dir)
    summary = {"passed": False, "source": "real_dualsense_and_native_isaac",
               "checks": {}, "runtime_dir": str(runtime_dir),
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                     cwd=ROOT, text=True).strip(),
               "git_worktree_status": subprocess.check_output(
                   ["git", "status", "--short", "--untracked-files=all"],
                   cwd=ROOT, text=True).splitlines()}
    target = load_attributed_graph(CONFIG / "smores_snake8.json")
    processes = []
    latest = {"status": None, "assembly": None}
    observer = None
    records = (output / "observations.jsonl").open("w", encoding="utf-8")

    def launch(name, command):
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT)
        processes.append((name, process))
        return process

    def receive(kind, message):
        payload = json.loads(message.data)
        latest[kind] = payload
        records.write(json.dumps({"received_at": time.monotonic(), "kind": kind,
                                  "payload": payload}, allow_nan=False) + "\n")

    def read(name):
        try:
            return json.loads((runtime_dir / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def state():
        status = latest["status"]
        if status is None or not 0 <= time.monotonic() - status["stamp_monotonic"] <= 0.5:
            raise ValueError("stale teleop status")
        return status

    def metrics():
        return physical_metrics(read("robot_graph.json"), target)

    def actions():
        return state().get("snake_effective_actions", {})

    def zero():
        return all(abs(float(command.get("vx", 0.0))) < 1e-9 for command in actions().values())

    def wait(label, instruction, predicate, timeout=45):
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
        summary["failure_context"] = {"phase": label, "teleop_status": latest["status"],
                                      "native_runtime_status": read("smores_teleop_runtime_status.json"),
                                      "robot_graph_stamp": read("robot_graph.json").get("stamp")}
        raise TimeoutError(f"{label} did not produce observed acceptance evidence")

    def drive(direction):
        before = metrics()
        trigger = "r2" if direction > 0 else "l2"
        reverse_window = OppositeWheelWindow(forward_wheels) if direction < 0 else None
        def observed():
            current, status = metrics(), state()
            displacement = sum((current["head"][i] - before["head"][i]) * before["forward"][i]
                               for i in range(2))
            commands_now = actions()
            summary["drive_evidence"] = {"direction": direction, "displacement_m": displacement,
                                         "trigger_value": status["controller_input"][trigger],
                                         "commands": commands_now, "before": before["head"],
                                         "after": current["head"]}
            direction_stable = (reverse_window is None or reverse_window.observe(
                status["stamp_monotonic"], commands_now))
            return (status["controller_input"][trigger] > 0.15
                    and status["safety"]["motion_enabled"]
                    and wheel_invariant(commands_now)
                    and direction_stable
                    and any(abs(command.get("vx", 0.0)) > 0.005 for command in commands_now.values())
                    and direction * displacement > 0.01)
        return observed

    try:
        summary["cleanup"] = scoped_cleanup(ROOT)
        enumeration = subprocess.run(["/opt/ros/humble/lib/joy/joy_enumerate_devices"],
                                     capture_output=True, text=True, timeout=15)
        device = next((item for item in parse_devices(enumeration.stdout)
                       if item["device_id"] == args.device_id), None)
        if enumeration.returncode or device is None or not device["gamepad"] or not device["dualsense"]:
            raise RuntimeError("selected Sony DualSense not detected")
        summary["device"] = device
        rclpy.init(args=[])
        observer = rclpy.create_node("mssr_snake_acceptance")
        observer.create_subscription(String, f"/mssr/teleop_probe/run_{output.name}/status",
                                     lambda message: receive("status", message), 100)
        observer.create_subscription(String, "/mssr/expert/self_assembly/state",
                                     lambda message: receive("assembly", message), 100)
        launch("isaac", commands["isaac"])
        launch("bridge", commands["bridge"])
        wait("native_ready", "Attendi Isaac; lascia il DualSense a riposo.",
             lambda: 0 <= time.monotonic() - read("smores_teleop_runtime_status.json").get("stamp_monotonic", -1e12) <= 0.5,
             timeout=240)
        assembly = launch("assembly", commands["assembly"])
        def assembled():
            status = latest["assembly"]
            if status and status.get("done") and not status.get("success"):
                raise RuntimeError("Snake assembly failed: " + status.get("message", ""))
            return bool(status and status.get("done") and status.get("success"))
        wait("snake_assembly", "Attendi la self-assembly Snake8 diretta.", assembled,
             timeout=args.assembly_timeout)
        assembly.terminate()
        assembly.wait(timeout=5)
        processes[:] = [(name, process) for name, process in processes if process is not assembly]
        def snake_topology_ready():
            try:
                return metrics()["stamp"] > 0
            except (KeyError, ValueError):
                return False
        wait("snake_topology", "Verifica la topologia fisica Snake8 assemblata.",
             snake_topology_ready, timeout=30)
        launch("joy", commands["joy"])
        launch("teleop", commands["teleop"])
        wait("snake_neutral", "Rilascia tutti i controlli; attendi il fresh-neutral fence.",
             lambda: latest["status"] is not None
             and state()["safety"]["motion_enabled"] and zero()
             and state()["active_controller"] == "snake8")
        wait("recording_start", "Premi START una volta per registrare la dimostrazione.",
             lambda: state()["recording"] and bool(state()["recording_episode_id"])
             and state()["recording_backend_ready"])
        episode_id = state()["recording_episode_id"]
        wait("forward", "Premi R2 circa 1/3: la testa deve avanzare.", drive(1), timeout=60)
        forward_wheels = {module: dict(command) for module, command in actions().items()}
        wait("forward_release", "Rilascia R2: tutte le ruote devono fermarsi.",
             lambda: state()["controller_input"]["r2"] == 0 and zero())
        wait("reverse", "Premi L2 circa 1/3: la testa deve arretrare.", drive(-1), timeout=60)
        wait("reverse_release", "Rilascia L2.",
             lambda: state()["controller_input"]["l2"] == 0 and zero())
        before = metrics()
        vertical_start = state()["snake_intent"]["vertical_offset_m"]
        def vertical():
            status, current = state(), metrics()
            module_tilt = max(abs(current["tilt"][module] - value)
                              for module, value in before["tilt"].items())
            dz = current["head"][2] - before["head"][2]
            summary["vertical_evidence"] = {"tilt_delta_max_rad": module_tilt,
                                             "head_delta_z_m": dz,
                                             "offset_m": status["snake_intent"]["vertical_offset_m"]}
            return (status["controller_input"]["right_y"] > 0.45
                    and status["snake_intent"]["vertical_offset_m"] > vertical_start + 0.002
                    and module_tilt > 0.01 and dz > 0.001)
        wait("vertical_tilt_up", "Trigger neutri; tieni lo stick destro verso ALTO finché TILT alza la testa.",
             vertical, timeout=60)
        wait("vertical_release", "Rilascia completamente lo stick destro.",
             lambda: abs(state()["controller_input"]["right_y"]) < 0.05)
        held_offset = state()["snake_intent"]["vertical_offset_m"]
        held_since = time.monotonic()
        wait("vertical_hold", "Lascia lo stick neutro: l'altezza impostata deve restare ferma.",
             lambda: abs(state()["controller_input"]["right_y"]) < 0.05
             and abs(state()["snake_intent"]["vertical_offset_m"] - held_offset) < 0.002
             and time.monotonic() - held_since > 1)
        before = metrics()
        vertical_start = state()["snake_intent"]["vertical_offset_m"]
        def downward():
            status, current = state(), metrics()
            module_tilt = max(abs(current["tilt"][module] - value)
                              for module, value in before["tilt"].items())
            dz = current["head"][2] - before["head"][2]
            summary["downward_evidence"] = {"tilt_delta_max_rad": module_tilt,
                                             "head_delta_z_m": dz,
                                             "offset_m": status["snake_intent"]["vertical_offset_m"]}
            return (status["controller_input"]["right_y"] < -0.45
                    and status["snake_intent"]["vertical_offset_m"] < vertical_start - 0.002
                    and module_tilt > 0.01 and dz < -0.001)
        wait("vertical_tilt_down", "Tieni lo stick destro verso BASSO finché TILT abbassa la testa.",
             downward, timeout=60)
        wait("down_release", "Rilascia completamente lo stick destro.",
             lambda: abs(state()["controller_input"]["right_y"]) < 0.05)
        before = metrics()
        def whole_body():
            status, current = state(), metrics()
            moved = [module for module, angle in before["tilt"].items()
                     if abs(current["tilt"][module] - angle) > 0.015]
            summary["whole_body_evidence"] = {"tilt_modules_moved": moved,
                                              "head": current["head"],
                                              "backbone": status["snake_intent"].get("backbone_m")}
            return (status["controller_input"]["r2"] > 0.15
                    and status["controller_input"]["right_y"] > 0.4
                    and wheel_invariant(actions()) and len(moved) >= 2)
        wait("whole_body_follow", "Tieni R2 circa 1/3 e stick destro ALTO: almeno due TILT devono seguire.",
             whole_body, timeout=90)
        wait("whole_body_release", "Rilascia R2 e lo stick destro.",
             lambda: state()["controller_input"]["r2"] == 0
             and abs(state()["controller_input"]["right_y"]) < 0.05 and zero())
        before_home = abs(state()["snake_intent"]["vertical_offset_m"])
        seen_home = False
        def homed():
            nonlocal seen_home
            status = state()
            seen_home |= "home" in status["controller_input"]["command_events"]
            return seen_home and abs(status["snake_intent"]["vertical_offset_m"]) < max(0.001, before_home - 0.002)
        wait("soft_home", "Premi CERCHIO una volta e lascia gli stick: ritorno graduale.",
             homed, timeout=45)
        wait("pre_estop_drive", "Premi R2 circa 1/3.", drive(1), timeout=60)
        wait("estop", "Mantieni R2 e premi TRIANGOLO (E-stop).",
             lambda: state()["safety"]["authority"] == "ESTOP"
             and state()["runtime_structure_stopped"] is True and zero())
        stopped = metrics()
        since = time.monotonic()
        wait("stop_physics_camera", "In E-stop muovi lo stick SINISTRO: camera e fisica devono continuare.",
             lambda: time.monotonic() - since > 1 and metrics()["stamp"] > stopped["stamp"]
             and read("smores_teleop_runtime_status.json").get("camera_applied") is True
             and any(abs(state()["controller_input"][axis]) > 0.3 for axis in ("left_x", "left_y"))
             and zero())
        wait("resume_fence", "Tieni R2 premuto; premi di nuovo TRIANGOLO per Resume.",
             lambda: state()["runtime_structure_stopped"] is False
             and state()["controller_input"]["r2"] > 0.1
             and not state()["safety"]["motion_enabled"] and zero())
        wait("fresh_neutral", "Rilascia trigger e stick per riarmare con un input neutro nuovo.",
             lambda: state()["safety"]["motion_enabled"] and zero()
             and state()["controller_input"]["r2"] == 0)
        wait("post_resume_forward", "Premi di nuovo R2 circa 1/3.", drive(1), timeout=60)
        wait("final_zero", "Rilascia tutti i controlli.", zero)
        wait("recording_stop", "Premi START una seconda volta per chiudere la registrazione.",
             lambda: not state()["recording"])
        human = ROOT / "logs/teleop/recordings" / episode_id / "human_behavior.jsonl"
        manifest_path = human.parent / "manifest.json"
        def recording_written():
            if not manifest_path.is_file():
                return False
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") == "failed":
                raise RuntimeError(f"Snake recording failed: {manifest.get('error')}")
            if manifest.get("status") != "completed":
                return False
            counts = recorded_snake_actions(human)
            summary["recording_evidence"] = {**counts, "path": str(human),
                                             "manifest": str(manifest_path),
                                             "eligible_for_import": manifest.get("eligible_for_import")}
            return (manifest.get("eligible_for_import") is True and counts["rows"] > 0
                    and counts["wheel_rows"] > 0 and counts["shape_rows"] > 0
                    and counts["manual_pan_rows"] > 0)
        wait("recorded_effective_actions", "Attendi la finalizzazione del dataset Snake.",
             recording_written, timeout=30)
        summary["passed"] = True
    except (Exception, KeyboardInterrupt) as error:
        summary["error"] = repr(error)
    finally:
        records.close()
        if observer is not None:
            observer.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        stopped = False
        try:
            summary["final_cleanup"] = scoped_cleanup(ROOT)
            stopped = True
        except Exception as error:
            summary["passed"] = False
            summary["cleanup_error"] = repr(error)
        try:
            summary.update(finalize_runtime(runtime_dir, output, writers_stopped=stopped))
        except Exception as error:
            summary["passed"] = False
            summary["runtime_finalize_error"] = repr(error)
            summary["runtime_preserved"] = str(runtime_dir)
        (output / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                            encoding="utf-8")
    print("T6A_SNAKE_RESULT=" + json.dumps(summary), flush=True)
    print(f"REPORT={output / 'report.json'}", flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
