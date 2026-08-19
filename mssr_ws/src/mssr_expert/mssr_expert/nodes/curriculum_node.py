"""ROS 2 node that publishes current curriculum state."""
from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from mssr_expert.curriculum.stage_config import StageConfig
from mssr_expert.utils.json_io import dict_to_string_msg


class CurriculumNode(Node):
    """Publish stage metadata for other nodes and dataset consumers."""

    def __init__(self) -> None:
        super().__init__("mssr_curriculum_node")
        self.declare_parameter("stage_id", 0)
        self.declare_parameter("stage_name", "gap_crossing")
        self.declare_parameter("task_type", "gap_crossing_temporary_bridge")
        self.declare_parameter("expert_name", "stage0_gap_crossing")
        self.declare_parameter("difficulty", 0.1)
        self.declare_parameter("scenario_name", "stage0_gap_crossing")
        self.declare_parameter("module_count", 6)
        self.declare_parameter("curriculum_state_topic", "/mssr/curriculum/state")
        self.declare_parameter("publish_rate_hz", 2.0)

        self._publisher = self.create_publisher(
            String,
            str(self.get_parameter("curriculum_state_topic").value),
            10,
        )
        period = 1.0 / max(1e-6, float(self.get_parameter("publish_rate_hz").value))
        self._timer = self.create_timer(period, self._publish)

    def _publish(self) -> None:
        stage = StageConfig(
            stage_id=int(self.get_parameter("stage_id").value),
            stage_name=str(self.get_parameter("stage_name").value),
            task_type=str(self.get_parameter("task_type").value),
            expert_name=str(self.get_parameter("expert_name").value),
            difficulty=float(self.get_parameter("difficulty").value),
            scenario_name=str(self.get_parameter("scenario_name").value),
            module_count=int(self.get_parameter("module_count").value),
        )
        self._publisher.publish(dict_to_string_msg(stage.to_dict()))


def main(args: list[str] | None = None) -> None:
    """Run the curriculum node."""
    rclpy.init(args=args)
    node = CurriculumNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
