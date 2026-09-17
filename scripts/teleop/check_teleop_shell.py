"""T1 installed ROS launch acceptance with synthetic Joy and frozen sim time.

Run after sourcing Humble and mssr_ws/install/setup.bash. No Isaac required.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from uuid import uuid4

from check_dualsense import configure_probe_environment
from runtime_cleanup import scoped_cleanup


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    assert importlib.util.find_spec("mssr_expert.nodes.smores_teleop_node") is not None, "missing T1 ROS shell"
    run_id = uuid4().hex
    output = ROOT / "logs/teleop/shell_checks" / run_id
    configure_probe_environment(os.environ, output)
    output.mkdir(parents=True, exist_ok=False)
    cleanup = scoped_cleanup(ROOT)
    joy_topic = f"/mssr/teleop_shell_check/run_{run_id}/joy"
    status_topic = f"/mssr/teleop_shell_check/run_{run_id}/status"
    shell_name = f"mssr_teleop_shell_{run_id}"

    import rclpy
    from sensor_msgs.msg import Joy
    from std_msgs.msg import String
    rclpy.init(args=[])
    observer = rclpy.create_node(f"mssr_teleop_observer_{run_id}")
    messages = []
    observer.create_subscription(String, status_topic,
        lambda message: messages.append(json.loads(message.data)), 100)
    publisher = observer.create_publisher(Joy, joy_topic, 10)
    process = None
    summary = {"passed": False, "cleanup": cleanup, "source": "synthetic_joy",
               "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    command = ["ros2", "launch", "mssr_expert", "smores_teleop.launch.py",
               "start_joy:=false", "use_sim_time:=true", f"joy_topic:={joy_topic}",
               f"status_topic:={status_topic}", f"node_name:={shell_name}"]
    command.extend([f"runtime_request_topic:=/mssr/teleop_shell_check/run_{run_id}/runtime_request",
                    f"runtime_status_topic:=/mssr/teleop_shell_check/run_{run_id}/runtime_status"])

    def wait_for(predicate, pressed=None, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"launch exited {process.returncode}; inspect {output / 'launch.log'}")
            if pressed is not None:
                message = Joy()
                message.axes = [0.0] * 6
                message.buttons = [0] * 21
                if pressed:
                    message.buttons[6] = 1
                publisher.publish(message)
            rclpy.spin_once(observer, timeout_sec=0.02)
            if messages and predicate(messages[-1]):
                return
        raise AssertionError("shell acceptance condition timed out")

    try:
        with (output / "launch.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            summary["stage"] = "launch_startup"
            wait_for(lambda status: status["phase"] == "READY", timeout=20.0)
            summary["stage"] = "joy_connection"
            assert messages[-1]["active_controller"] is None
            wait_for(lambda status: status["controller_connected"], pressed=False)
            summary["stage"] = "recording_edges"
            wait_for(lambda status: status["recording"], pressed=True)
            # Hold START long enough to cover many control-loop ticks.
            before = time.monotonic()
            wait_for(lambda status: time.monotonic() - before >= 0.25, pressed=True)
            assert messages[-1]["recording"]
            wait_for(lambda status: "start" not in status["controller_input"]["buttons"], pressed=False)
            wait_for(lambda status: not status["recording"], pressed=True)
            wait_for(lambda status: "start" not in status["controller_input"]["buttons"], pressed=False)
            summary["stage"] = "joy_timeout"
            wait_for(lambda status: not status["controller_connected"])
            assert messages[-1]["phase"] == "READY" and messages[-1]["authority"] == "NONE"
            publishers = observer.get_publisher_names_and_types_by_node(shell_name, "/")
            robot_topics = {"/mssr/actions", "/mssr/primitives/goal", "/mssr/primitives/cancel"}
            assert not robot_topics.intersection(topic for topic, _ in publishers)
            elapsed = messages[-1]["stamp_monotonic"] - messages[0]["stamp_monotonic"]
            observed_hz = (len(messages) - 1) / elapsed
            assert 35 <= observed_hz <= 65, observed_hz
            assert all(not status["actuator_commands_enabled"] for status in messages)
            assert all(status["stamp_ros"] == 0.0 for status in messages)
            summary.update(passed=True, statuses=len(messages), measured_control_hz=observed_hz,
                           joy_timeout_observed=True, recording_edge_state_verified=True,
                           frozen_sim_time=True, robot_publishers=[], publishers=publishers)
            summary["stage"] = "verified"
    except Exception as error:
        summary["error"] = repr(error)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
        observer.destroy_node()
        rclpy.shutdown()
        # Verify that launch children stopped, and stop the daemon for this domain.
        summary["final_cleanup"] = scoped_cleanup(ROOT)
        (output / "statuses.jsonl").write_text("".join(json.dumps(item) + "\n" for item in messages))
        (output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("T1_SHELL_RESULT=" + json.dumps(summary), flush=True)
    print(f"REPORT={output / 'report.json'}", flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
