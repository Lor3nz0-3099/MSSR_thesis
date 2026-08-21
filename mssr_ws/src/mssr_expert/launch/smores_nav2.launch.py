"""Launch Nav2 against the virtual base exported by the MSSR morphology node."""

from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Launch navigation only; the morphology controller is started separately."""

    package_share = Path(get_package_share_directory("mssr_expert"))
    nav2_share = Path(get_package_share_directory("nav2_bringup"))
    default_params = package_share / "config" / "nav2_smores.yaml"

    params_file = LaunchConfiguration("params_file")
    autostart = LaunchConfiguration("autostart")
    log_level = LaunchConfiguration("log_level")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=str(default_params),
                description="Nav2 parameters for the current MSSR morphology.",
            ),
            DeclareLaunchArgument(
                "autostart",
                default_value="true",
                description="Automatically activate the Nav2 lifecycle nodes.",
            ),
            DeclareLaunchArgument(
                "log_level",
                default_value="info",
                description="Nav2 log level.",
            ),
            # The virtual /odom pose is expressed directly in Isaac's world
            # coordinates, so map and odom deliberately coincide for now.
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="mssr_map_to_odom",
                output="screen",
                arguments=[
                    "0", "0", "0",
                    "0", "0", "0",
                    "map", "odom",
                ],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    str(nav2_share / "launch" / "navigation_launch.py")
                ),
                launch_arguments={
                    "namespace": "",
                    "use_sim_time": "false",
                    "autostart": autostart,
                    "params_file": params_file,
                    "use_composition": "False",
                    "use_respawn": "false",
                    "log_level": log_level,
                }.items(),
            ),
        ]
    )
