"""Launch the MSSR curriculum node."""
from __future__ import annotations

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Create launch description."""
    return LaunchDescription(
        [
            Node(
                package="mssr_expert",
                executable="mssr_curriculum_node",
                name="mssr_curriculum_node",
                output="screen",
            )
        ]
    )
