"""Launch the MSSR deterministic expert node."""
from __future__ import annotations

from os.path import join

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Create launch description."""
    config_path = join(get_package_share_directory("mssr_expert"), "config", "expert.yaml")
    return LaunchDescription(
        [
            Node(
                package="mssr_expert",
                executable="mssr_expert_node",
                name="mssr_expert_node",
                output="screen",
                parameters=[config_path],
            )
        ]
    )
