"""Start the SMORES simulation, file bridge, and morphology controller.

Self-assembly and Nav2 are deliberately separate lifecycle phases.  Start the
self-assembly expert after this launch is ready, then start Nav2 only after the
assembled morphology publishes ``/odom`` and ``odom -> base_link``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


_CURRENT_STATE_FILES = (
    "module_states.json",
    "robot_graph.json",
    "state_graph.json",
    "task_metrics.json",
    "primitive_status.json",
)


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean launch value, got {value!r}")


def _runtime_path(repository_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repository_root / path
    return path.resolve()


def _prepare_runtime_files(repository_root: Path, runtime_dir: Path) -> None:
    if runtime_dir in {Path("/"), repository_root}:
        raise ValueError(
            "runtime_dir must be a dedicated directory, not the repository root"
        )
    runtime_dir.mkdir(parents=True, exist_ok=True)
    idle_actions = repository_root / "configs" / "idle_actions.json"
    (runtime_dir / "actions.json").write_text(
        idle_actions.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    for name in ("primitive_goal.json", "primitive_cancel.json"):
        (runtime_dir / name).write_text("{}\n", encoding="utf-8")
    for name in _CURRENT_STATE_FILES:
        (runtime_dir / name).unlink(missing_ok=True)


def _launch_runtime(context: LaunchContext) -> list[object]:
    repository_root = Path(
        LaunchConfiguration("repository_root").perform(context)
    ).expanduser().resolve()
    runtime_dir = _runtime_path(
        repository_root,
        LaunchConfiguration("runtime_dir").perform(context),
    )
    if _as_bool(LaunchConfiguration("clean_runtime").perform(context)):
        _prepare_runtime_files(repository_root, runtime_dir)

    action_file = runtime_dir / "actions.json"
    primitive_goal_file = runtime_dir / "primitive_goal.json"
    primitive_cancel_file = runtime_dir / "primitive_cancel.json"
    primitive_status_file = runtime_dir / "primitive_status.json"

    simulation_command = [
        str(repository_root / "scripts" / "smores_ep" / "run_self_assembly.sh"),
        "--module-count",
        LaunchConfiguration("module_count").perform(context),
        "--action-file",
        str(action_file),
        "--primitive-goal-file",
        str(primitive_goal_file),
        "--primitive-cancel-file",
        str(primitive_cancel_file),
        "--primitive-status-file",
        str(primitive_status_file),
        "--steps",
        LaunchConfiguration("simulation_steps").perform(context),
        "--simulation-speed-factor",
        LaunchConfiguration("simulation_speed_factor").perform(context),
    ]
    stair_seed = LaunchConfiguration("stair_seed").perform(context).strip()
    if stair_seed:
        simulation_command.extend(("--stair-seed", stair_seed))
    for launch_name, option_name in (
        ("stair_rise_m", "--stair-rise-m"),
        ("stair_depth_m", "--stair-depth-m"),
        ("stair_count", "--stair-count"),
        ("stair_first_riser_x_m", "--stair-first-riser-x-m"),
    ):
        value = LaunchConfiguration(launch_name).perform(context).strip()
        if value:
            simulation_command.extend((option_name, value))
    gap_seed = LaunchConfiguration("gap_seed").perform(context).strip()
    if gap_seed:
        simulation_command.extend(("--gap-seed", gap_seed))
    for launch_name, option_name in (
        ("gap_width_m", "--gap-width-m"),
        ("gap_near_edge_x_m", "--gap-near-edge-x-m"),
    ):
        value = LaunchConfiguration(launch_name).perform(context).strip()
        if value:
            simulation_command.extend((option_name, value))
    if _as_bool(LaunchConfiguration("performance").perform(context)):
        simulation_command.append("--performance")
    if _as_bool(LaunchConfiguration("simple_visuals").perform(context)):
        simulation_command.append("--simple-visuals")
    if _as_bool(LaunchConfiguration("obstacle_course").perform(context)):
        simulation_command.append("--obstacle-course")
    if _as_bool(LaunchConfiguration("stair_test_course").perform(context)):
        simulation_command.append("--stair-test-course")
    if _as_bool(LaunchConfiguration("button_test_course").perform(context)):
        simulation_command.append("--button-test-course")
    if _as_bool(LaunchConfiguration("gap_test_course").perform(context)):
        simulation_command.append("--gap-test-course")
    if _as_bool(LaunchConfiguration("headless").perform(context)):
        simulation_command.append("--headless")

    bridge_command = [
        sys.executable,
        str(repository_root / "ros2_bridge" / "mssr_file_bridge.py"),
        "--state-graph-dir",
        str(runtime_dir),
        "--action-file",
        str(action_file),
        "--primitive-goal-file",
        str(primitive_goal_file),
        "--primitive-cancel-file",
        str(primitive_cancel_file),
        "--primitive-status-file",
        str(primitive_status_file),
    ]

    return [
        ExecuteProcess(
            cmd=bridge_command,
            cwd=str(repository_root),
            name="mssr_file_bridge",
            output="screen",
            emulate_tty=True,
        ),
        Node(
            package="mssr_expert",
            executable="mssr_smores_morphology_behavior_node",
            name="smores_morphology_behavior_node",
            output="screen",
            emulate_tty=True,
            parameters=[
                {
                    "behavior_dataset_path": LaunchConfiguration(
                        "behavior_dataset_path"
                    ),
                    "behavior_dataset_episode_id": LaunchConfiguration(
                        "behavior_dataset_episode_id"
                    ),
                    "behavior_dataset_stage_name": LaunchConfiguration(
                        "behavior_dataset_stage_name"
                    ),
                    "behavior_dataset_difficulty": LaunchConfiguration(
                        "behavior_dataset_difficulty"
                    ),
                    "behavior_dataset_log_period": LaunchConfiguration(
                        "behavior_dataset_log_period"
                    ),
                }
            ],
        ),
        ExecuteProcess(
            cmd=simulation_command,
            cwd=str(repository_root),
            name="smores_isaac_sim",
            output="both",
            emulate_tty=True,
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    """Create the pre-assembly SMORES runtime launch description."""

    repository_root = Path(__file__).resolve().parents[4]
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "repository_root",
                default_value=str(repository_root),
                description="Root of the MSSR_thesis checkout.",
            ),
            DeclareLaunchArgument(
                "runtime_dir",
                default_value="logs/nav2_smoke",
                description="Shared Isaac/ROS JSON channel directory.",
            ),
            DeclareLaunchArgument(
                "module_count",
                default_value="8",
                description="Number of SMORES-EP modules spawned in Isaac.",
            ),
            DeclareLaunchArgument(
                "clean_runtime",
                default_value="true",
                description="Clear stale commands and current-state payloads.",
            ),
            DeclareLaunchArgument(
                "performance",
                default_value="false",
                description="Enable Isaac realtime performance settings.",
            ),
            DeclareLaunchArgument(
                "simple_visuals",
                default_value="false",
                description="Use collision proxies instead of full CAD.",
            ),
            DeclareLaunchArgument(
                "obstacle_course",
                default_value="false",
                description="Spawn the physical gap and stair course in Isaac.",
            ),
            DeclareLaunchArgument(
                "stair_test_course",
                default_value="false",
                description="Spawn the isolated three-step Snake8 test stage.",
            ),
            DeclareLaunchArgument(
                "button_test_course",
                default_value="false",
                description=(
                    "Spawn the isolated MobileManipulator8 button test stage."
                ),
            ),
            DeclareLaunchArgument(
                "gap_test_course",
                default_value="false",
                description="Spawn the isolated Snake8 gap test stage.",
            ),
            DeclareLaunchArgument(
                "headless",
                default_value="false",
                description="Run Isaac without its GUI.",
            ),
            DeclareLaunchArgument(
                "simulation_steps",
                default_value="0",
                description=(
                    "Maximum Isaac physics steps; headless batch runs should "
                    "set an explicit episode guard."
                ),
            ),
            DeclareLaunchArgument(
                "simulation_speed_factor",
                default_value="1.0",
                description=(
                    "Simulated seconds per wall second when performance "
                    "pacing is enabled."
                ),
            ),
            DeclareLaunchArgument(
                "stair_seed",
                default_value="",
                description=(
                    "Optional seed for the conservative uniform-stair "
                    "generator."
                ),
            ),
            DeclareLaunchArgument(
                "stair_rise_m",
                default_value="",
                description=(
                    "Optional uniform stair rise override in metres; empty "
                    "uses the reference or seeded value."
                ),
            ),
            DeclareLaunchArgument(
                "stair_depth_m",
                default_value="",
                description=(
                    "Optional uniform tread-depth override in metres."
                ),
            ),
            DeclareLaunchArgument(
                "stair_count",
                default_value="",
                description="Optional number of uniform stair risers.",
            ),
            DeclareLaunchArgument(
                "stair_first_riser_x_m",
                default_value="",
                description=(
                    "Optional world-X coordinate override for the first riser."
                ),
            ),
            DeclareLaunchArgument(
                "gap_seed",
                default_value="",
                description=(
                    "Optional seed for the conservative coplanar-gap "
                    "generator."
                ),
            ),
            DeclareLaunchArgument(
                "gap_width_m",
                default_value="",
                description=(
                    "Optional isolated gap width override in metres."
                ),
            ),
            DeclareLaunchArgument(
                "gap_near_edge_x_m",
                default_value="",
                description=(
                    "Optional world-X coordinate of the near gap edge."
                ),
            ),
            DeclareLaunchArgument(
                "behavior_dataset_path",
                default_value="",
                description=(
                    "Optional JSONL path for graph-conditioned morphology "
                    "behavior transitions; empty disables recording."
                ),
            ),
            DeclareLaunchArgument(
                "behavior_dataset_episode_id",
                default_value="",
                description="Episode ID written into behavior transitions.",
            ),
            DeclareLaunchArgument(
                "behavior_dataset_stage_name",
                default_value="morphology_behavior",
                description="Stage label written into behavior transitions.",
            ),
            DeclareLaunchArgument(
                "behavior_dataset_difficulty",
                default_value="0.0",
                description="Numeric curriculum difficulty stored in JSONL.",
            ),
            DeclareLaunchArgument(
                "behavior_dataset_log_period",
                default_value="1",
                description="Record one behavior transition every N ticks.",
            ),
            DeclareLaunchArgument(
                "ros_domain_id",
                default_value="0",
                description="ROS domain shared by bridge and behavior node.",
            ),
            DeclareLaunchArgument(
                "rmw_implementation",
                default_value="rmw_cyclonedds_cpp",
                description="ROS middleware shared by runtime processes.",
            ),
            SetEnvironmentVariable(
                "ROS_DOMAIN_ID",
                LaunchConfiguration("ros_domain_id"),
            ),
            SetEnvironmentVariable(
                "RMW_IMPLEMENTATION",
                LaunchConfiguration("rmw_implementation"),
            ),
            OpaqueFunction(function=_launch_runtime),
        ]
    )
