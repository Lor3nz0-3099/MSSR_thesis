"""External teleoperation shell with separate structure-stop and camera channels."""
from __future__ import annotations

import json
from pathlib import Path
import time
from types import SimpleNamespace

import yaml

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Joy
from std_msgs.msg import String

from mssr_expert.teleop.config import control_rate, load_teleop_config
from mssr_expert.teleop.coordinator import RuntimeCoordinator
from mssr_expert.teleop.input import load_input_config
from mssr_expert.teleop.session import TeleopSession
from mssr_expert.teleop.safety import SafetyDecision
from mssr_expert.teleop.action_transport import RcCarRuntime
from mssr_expert.teleop.rc_car import load_geometry
from mssr_expert.behaviors.morphology_library import MorphologyLibrary
from mssr_expert.graph.serialization import load_attributed_graph


class SmoresTeleopNode(Node):
    def __init__(self) -> None:
        super().__init__("mssr_smores_teleop_node")
        config_dir = Path(get_package_share_directory("mssr_expert")) / "config"
        input_path = self.declare_parameter("input_config_path", str(config_dir / "smores_dualsense.yaml")).value
        teleop_path = self.declare_parameter("teleop_config_path", str(config_dir / "smores_teleop.yaml")).value
        config = load_teleop_config(teleop_path)
        self.session = TeleopSession(load_input_config(input_path))
        self.coordinator = RuntimeCoordinator(self.session)
        rc_config = yaml.safe_load(Path(teleop_path).read_text()).get("rc_car", {})
        self._rc = RcCarRuntime(MorphologyLibrary.load(config_dir / "smores_morphology_behaviors.json"),
                                load_attributed_graph(config_dir / "smores_rc_car8.json"),
                                geometry=load_geometry(str(teleop_path)), **rc_config)
        self._actions = self.create_publisher(String, "/mssr/actions", 10)
        self._goal = self.create_publisher(String, "/mssr/primitives/goal", 10)
        self._cancel = self.create_publisher(String, "/mssr/primitives/cancel", 10)
        self._graph = self.create_subscription(String, "/mssr/robot_graph", self._on_graph, 10)
        self._primitive_status = self.create_subscription(String, "/mssr/primitives/status", self._on_primitive_status, 10)
        joy_topic = self.declare_parameter("joy_topic", config.joy_topic).value
        status_topic = self.declare_parameter("status_topic", "/mssr/teleop/status").value
        rate = control_rate(self.declare_parameter("control_rate_hz", config.control_rate_hz).value)
        self._status = self.create_publisher(String, status_topic, 10)
        runtime_request_topic = self.declare_parameter("runtime_request_topic", "/mssr/teleop/runtime_request").value
        runtime_status_topic = self.declare_parameter("runtime_status_topic", "/mssr/teleop/runtime_status").value
        self._runtime_request = self.create_publisher(String, runtime_request_topic, 10)
        self._runtime_status = self.create_subscription(String, runtime_status_topic, self._on_runtime_status, 10)
        self._joy = self.create_subscription(Joy, joy_topic, self._on_joy, qos_profile_sensor_data)
        # Input age and diagnostics use a steady wall clock independent of Isaac simulation time.
        self._wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self._timer = self.create_timer(1.0 / rate, self._tick, clock=self._wall_clock)
        self.get_logger().info(f"RC-Car8 teleop at {rate:g} Hz; live topology and runtime safety required")

    def _on_graph(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (ValueError, TypeError):
            payload = None
        self._rc.observe_graph(payload, now=time.monotonic())

    def _on_primitive_status(self, message: String) -> None:
        try:
            self._rc.observe_status(json.loads(message.data))
        except (KeyError, RuntimeError, TypeError, ValueError):
            return

    def _on_joy(self, message: Joy) -> None:
        self.session.update_joy(message.axes, message.buttons, time.monotonic())

    def _on_runtime_status(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (ValueError, TypeError):
            return
        self.coordinator.observe_runtime(payload, time.monotonic())

    def _tick(self) -> None:
        now = time.monotonic()
        self.session.state.observe_topology(self._rc.topology(now))
        status, runtime_request = self.coordinator.tick(now)
        output = self._rc.step(SimpleNamespace(**status["controller_input"]),
                               safety=SafetyDecision(**status["safety"]), now=now)
        if output.envelope is not None:
            self._actions.publish(String(data=output.envelope))
        if output.posture.goal is not None:
            self._goal.publish(String(data=json.dumps(output.posture.goal.to_dict(), allow_nan=False)))
        if output.posture.cancel_goal_id is not None:
            self._cancel.publish(String(data=json.dumps({"goal_id": output.posture.cancel_goal_id})))
        status["stamp_ros"] = self.get_clock().now().nanoseconds * 1e-9
        status.update(actuator_commands_enabled=bool(status["safety"]["motion_enabled"] and output.envelope),
                      topology_verification_ready=self._rc.topology(now) is not None,
                      recording_backend_ready=False,
                      rc_car_intent=output.actions.intent,
                      rc_car_effective_actions=output.actions.module_actions)
        message = String()
        message.data = json.dumps(status, allow_nan=False)
        self._status.publish(message)
        request = String()
        request.data = json.dumps(runtime_request, allow_nan=False)
        self._runtime_request.publish(request)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = SmoresTeleopNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
