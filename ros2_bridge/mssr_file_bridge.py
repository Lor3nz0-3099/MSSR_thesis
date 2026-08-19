"""ROS 2 bridge between Isaac JSON files and ROS 2 string topics.

Run this script with the system ROS 2 Python environment, not with Isaac's
``python.sh``. It keeps ``rclpy`` outside Isaac Sim while still exposing the
simulation state, graph, and actions through ROS 2 topics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from std_msgs.msg import String


class MssrFileBridge(Node):
    """Publish Isaac state files and write incoming action messages."""

    def __init__(
        self,
        state_graph_dir: Path,
        action_file: Path,
        primitive_goal_file: Path,
        primitive_cancel_file: Path,
        primitive_status_file: Path,
        publish_period: float,
    ) -> None:
        """Create publishers, subscriber, and periodic file polling."""
        super().__init__("mssr_file_bridge")
        self._state_graph_dir = state_graph_dir
        self._action_file = action_file
        self._primitive_goal_file = primitive_goal_file
        self._primitive_cancel_file = primitive_cancel_file
        self._primitive_status_file = primitive_status_file

        self._state_pub = self.create_publisher(String, "/mssr/module_states", 10)
        self._graph_pub = self.create_publisher(String, "/mssr/robot_graph", 10)
        self._combined_pub = self.create_publisher(String, "/mssr/state_graph", 10)
        self._task_metrics_pub = self.create_publisher(String, "/mssr/task_metrics", 10)
        self._primitive_status_pub = self.create_publisher(
            String,
            "/mssr/primitives/status",
            10,
        )
        self._action_sub = self.create_subscription(
            String,
            "/mssr/actions",
            self._on_actions,
            10,
        )
        self._primitive_goal_sub = self.create_subscription(
            String,
            "/mssr/primitives/goal",
            self._on_primitive_goal,
            10,
        )
        self._primitive_cancel_sub = self.create_subscription(
            String,
            "/mssr/primitives/cancel",
            self._on_primitive_cancel,
            10,
        )
        self._timer = self.create_timer(publish_period, self._publish_files)
        self.get_logger().info(f"Reading Isaac JSON payloads from {state_graph_dir}")
        self.get_logger().info(f"Writing incoming actions to {action_file}")
        self.get_logger().info(
            "Primitive protocol: "
            f"goal={primitive_goal_file}, cancel={primitive_cancel_file}, "
            f"status={primitive_status_file}"
        )

    def _publish_files(self) -> None:
        """Publish the latest JSON files on ROS 2 topics."""
        self._publish_file("module_states.json", self._state_pub)
        self._publish_file("robot_graph.json", self._graph_pub)
        self._publish_file("state_graph.json", self._combined_pub)
        self._publish_file("task_metrics.json", self._task_metrics_pub)
        self._publish_path(
            self._primitive_status_file,
            self._primitive_status_pub,
        )

    def _publish_file(self, filename: str, publisher: object) -> None:
        """Publish one JSON file if it exists."""
        self._publish_path(self._state_graph_dir / filename, publisher)

    def _publish_path(self, path: Path, publisher: object) -> None:
        """Publish one JSON payload from an explicit path."""
        try:
            payload = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return

        message = String()
        message.data = payload
        publisher.publish(message)

    def _on_actions(self, message: String) -> None:
        """Write the latest ROS 2 action payload for Isaac to consume."""
        self._write_atomic(self._action_file, message.data)

    def _on_primitive_goal(self, message: String) -> None:
        """Forward a new action-like primitive goal to Isaac."""
        self._write_atomic(self._primitive_goal_file, message.data)

    def _on_primitive_cancel(self, message: String) -> None:
        """Forward a primitive cancellation request to Isaac."""
        self._write_atomic(self._primitive_cancel_file, message.data)

    @staticmethod
    def _write_atomic(path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(payload, encoding="utf-8")
        temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    """Parse bridge options."""
    parser = argparse.ArgumentParser(description="Bridge MSSR Isaac JSON files to ROS 2.")
    parser.add_argument(
        "--state-graph-dir",
        default="logs/bridge",
        help="Directory where Isaac writes state/graph JSON files.",
    )
    parser.add_argument(
        "--action-file",
        default="configs/actions.json",
        help="JSON file written from /mssr/actions and read by Isaac.",
    )
    parser.add_argument(
        "--publish-period",
        type=float,
        default=0.05,
        help="Polling period in seconds for publishing changed Isaac JSON files.",
    )
    parser.add_argument(
        "--primitive-goal-file",
        default="configs/smores_primitive_goal.json",
        help="Goal file consumed by the SMORES-EP primitive executor.",
    )
    parser.add_argument(
        "--primitive-cancel-file",
        default="configs/smores_primitive_cancel.json",
        help="Cancellation file consumed by the primitive executor.",
    )
    parser.add_argument(
        "--primitive-status-file",
        default="logs/bridge/smores_primitive_status.json",
        help="Status file produced by Isaac and published to ROS 2.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the ROS 2 file bridge node."""
    args = parse_args()
    rclpy.init()
    node = MssrFileBridge(
        state_graph_dir=Path(args.state_graph_dir),
        action_file=Path(args.action_file),
        primitive_goal_file=Path(args.primitive_goal_file),
        primitive_cancel_file=Path(args.primitive_cancel_file),
        primitive_status_file=Path(args.primitive_status_file),
        publish_period=args.publish_period,
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
