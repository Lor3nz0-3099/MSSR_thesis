"""External teleoperation shell with a separate timeline/camera channel."""
from __future__ import annotations

import json
from pathlib import Path
import time

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


class SmoresTeleopNode(Node):
    def __init__(self) -> None:
        super().__init__("mssr_smores_teleop_node")
        config_dir = Path(get_package_share_directory("mssr_expert")) / "config"
        input_path = self.declare_parameter("input_config_path", str(config_dir / "smores_dualsense.yaml")).value
        teleop_path = self.declare_parameter("teleop_config_path", str(config_dir / "smores_teleop.yaml")).value
        config = load_teleop_config(teleop_path)
        self.session = TeleopSession(load_input_config(input_path))
        self.coordinator = RuntimeCoordinator(self.session)
        joy_topic = self.declare_parameter("joy_topic", config.joy_topic).value
        status_topic = self.declare_parameter("status_topic", "/mssr/teleop/status").value
        rate = control_rate(self.declare_parameter("control_rate_hz", config.control_rate_hz).value)
        self._status = self.create_publisher(String, status_topic, 10)
        runtime_request_topic = self.declare_parameter("runtime_request_topic", "/mssr/teleop/runtime_request").value
        runtime_status_topic = self.declare_parameter("runtime_status_topic", "/mssr/teleop/runtime_status").value
        self._runtime_request = self.create_publisher(String, runtime_request_topic, 10)
        self._runtime_status = self.create_subscription(String, runtime_status_topic, self._on_runtime_status, 10)
        self._joy = self.create_subscription(Joy, joy_topic, self._on_joy, qos_profile_sensor_data)
        # Input age and diagnostics must progress while Isaac's simulation clock is paused.
        self._wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self._timer = self.create_timer(1.0 / rate, self._tick, clock=self._wall_clock)
        self.get_logger().info(f"Teleop diagnostics at {rate:g} Hz; actuator transport disabled")

    def _on_joy(self, message: Joy) -> None:
        self.session.update_joy(message.axes, message.buttons, time.monotonic())

    def _on_runtime_status(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (ValueError, TypeError):
            return
        self.coordinator.observe_runtime(payload, time.monotonic())

    def _tick(self) -> None:
        status, runtime_request = self.coordinator.tick(time.monotonic())
        status["stamp_ros"] = self.get_clock().now().nanoseconds * 1e-9
        status.update(actuator_commands_enabled=False,
                      topology_verification_ready=False, recording_backend_ready=False)
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
