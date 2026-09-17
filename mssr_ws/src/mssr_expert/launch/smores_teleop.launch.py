"""DualSense and external teleoperation shell, without starting Isaac."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from mssr_expert.teleop.config import load_teleop_config


def _nodes(context):
    def argument(name):
        return LaunchConfiguration(name).perform(context)

    config = load_teleop_config(argument("teleop_config_path"))
    joy_topic = argument("joy_topic") or config.joy_topic
    nodes = []
    start_joy = argument("start_joy").lower()
    use_sim_time = argument("use_sim_time").lower()
    if start_joy not in {"true", "false"} or use_sim_time not in {"true", "false"}:
        raise ValueError("start_joy and use_sim_time must be true or false")
    if start_joy == "true":
        nodes.append(Node(package="joy", executable="game_controller_node",
                          name="mssr_dualsense", output="screen",
                          parameters=[config.joy_parameters()], remappings=[("joy", joy_topic)]))
    nodes.append(Node(package="mssr_expert", executable="mssr_smores_teleop_node",
                      name=argument("node_name"), output="screen",
                      parameters=[{"input_config_path": argument("input_config_path"),
                                   "teleop_config_path": argument("teleop_config_path"),
                                   "joy_topic": joy_topic, "status_topic": argument("status_topic"),
                                   "use_sim_time": use_sim_time == "true"}]))
    return nodes


def generate_launch_description():
    config_dir = Path(get_package_share_directory("mssr_expert")) / "config"
    return LaunchDescription([
        DeclareLaunchArgument("input_config_path", default_value=str(config_dir / "smores_dualsense.yaml")),
        DeclareLaunchArgument("teleop_config_path", default_value=str(config_dir / "smores_teleop.yaml")),
        DeclareLaunchArgument("joy_topic", default_value=""),
        DeclareLaunchArgument("status_topic", default_value="/mssr/teleop/status"),
        DeclareLaunchArgument("node_name", default_value="mssr_smores_teleop_node"),
        DeclareLaunchArgument("start_joy", default_value="true"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        OpaqueFunction(function=_nodes),
    ])
