"""Entry point for the MSSR Isaac Sim 6.0 framework.

Legacy status: this launches the spherical FreeBOT prototype scenario only
(``robots/spherical_robot.py`` and ``graphs/robot_graph.py``). The SMORES-EP
backend is launched instead through ``scripts/smores_ep/run_*.sh``. See
``context/LEGACY_SPHERE_MIGRATION.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaacsim import SimulationApp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    """Parse standalone simulation options."""
    parser = argparse.ArgumentParser(description="Run the MSSR Isaac Sim framework.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without the GUI.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop after this many SimulationApp updates. Use 0 to run until the app closes.",
    )
    parser.add_argument(
        "--demo-forward-speed",
        type=float,
        default=0.0,
        help="Command a constant forward speed for the spherical robot.",
    )
    parser.add_argument(
        "--demo-yaw-rate",
        type=float,
        default=0.0,
        help="Command a constant yaw rate for the spherical robot.",
    )
    parser.add_argument(
        "--demo-control-scope",
        choices=("controlled", "all"),
        default="controlled",
        help="Choose whether demo commands control only sphere_0 or all spawned modules.",
    )
    parser.add_argument(
        "--module-count",
        type=int,
        default=1,
        help="Number of spherical modules to spawn.",
    )
    parser.add_argument(
        "--module-spacing",
        type=float,
        default=1.2,
        help="Distance between neighboring module centers at spawn time.",
    )
    parser.add_argument(
        "--scenario-file",
        default="",
        help="Load module spawn and reset settings from this scenario JSON file.",
    )
    parser.add_argument(
        "--gravity-magnitude",
        type=float,
        default=19.62,
        help="World gravity magnitude in m/s^2. Use 9.81 for Earth gravity.",
    )
    parser.add_argument(
        "--attachment-mode",
        choices=("continuous", "six_axis"),
        default="continuous",
        help="Contact-site model for magnetic attachment detection.",
    )
    parser.add_argument(
        "--command-source",
        choices=("demo", "ros2", "attach-test", "json", "json-file"),
        default="demo",
        help="Choose where robot velocity commands come from.",
    )
    parser.add_argument(
        "--ros2-cmd-vel-topic",
        default="/cmd_vel",
        help="ROS 2 topic used when --command-source=ros2.",
    )
    parser.add_argument(
        "--ros2-command-timeout",
        type=float,
        default=0.5,
        help="Stop the robot if no ROS 2 command is received for this many seconds.",
    )
    parser.add_argument(
        "--ros2-odom-topic",
        default="/odom",
        help="ROS 2 odometry topic published when --command-source=ros2.",
    )
    parser.add_argument(
        "--ros2-tf-topic",
        default="/tf",
        help="ROS 2 TF topic published when --command-source=ros2.",
    )
    parser.add_argument(
        "--ros2-clock-topic",
        default="/clock",
        help="ROS 2 clock topic published when --command-source=ros2.",
    )
    parser.add_argument(
        "--json-action-file",
        default="",
        help="Read multi-module locomotion and magnetic actions from this JSON file.",
    )
    parser.add_argument(
        "--publish-json-dir",
        default="",
        help="Write module state and robot graph JSON payloads to this directory.",
    )
    parser.add_argument(
        "--json-publish-interval",
        type=int,
        default=1,
        help="Publish JSON payloads every N simulation steps when --publish-json-dir is set.",
    )
    parser.add_argument(
        "--publish-json-history",
        action="store_true",
        help="Append combined state/graph payloads to state_graph_history.jsonl.",
    )
    parser.add_argument(
        "--reset-every-steps",
        type=int,
        default=0,
        help="Reset modules and detach magnetic joints every N simulation steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--debug-state",
        action="store_true",
        help="Print module registry snapshots while the simulation is running.",
    )
    parser.add_argument(
        "--debug-state-interval",
        type=int,
        default=60,
        help="Print one state snapshot every N simulation steps when --debug-state is enabled.",
    )
    parser.add_argument(
        "--debug-graph",
        action="store_true",
        help="Print graph snapshots while the simulation is running.",
    )
    parser.add_argument(
        "--debug-task",
        action="store_true",
        help="Print task evaluation metrics while the simulation is running.",
    )
    parser.add_argument(
        "--stop-on-task-done",
        action="store_true",
        help="Stop the simulation when the task succeeds or reaches timeout.",
    )
    parser.add_argument(
        "--auto-attach-on-contact",
        action="store_true",
        help="Testing mode: create magnetic attach actions automatically when modules are in contact.",
    )
    parser.add_argument(
        "--auto-attach-joint-type",
        choices=("rigid", "spherical", "hinge"),
        default="spherical",
        help="Joint type used by --auto-attach-on-contact testing mode.",
    )
    return parser.parse_args()


def run_simulation(
    headless: bool,
    max_steps: int,
    demo_forward_speed: float,
    demo_yaw_rate: float,
    demo_control_scope: str,
    module_count: int,
    module_spacing: float,
    scenario_file: str,
    gravity_magnitude: float,
    attachment_mode: str,
    command_source_name: str,
    ros2_cmd_vel_topic: str,
    ros2_command_timeout: float,
    ros2_odom_topic: str,
    ros2_tf_topic: str,
    ros2_clock_topic: str,
    json_action_file: str,
    publish_json_dir: str,
    json_publish_interval: int,
    publish_json_history: bool,
    reset_every_steps: int,
    debug_state: bool,
    debug_state_interval: int,
    debug_graph: bool,
    debug_task: bool,
    stop_on_task_done: bool,
    auto_attach_on_contact: bool,
    auto_attach_joint_type: str,
) -> None:
    """Launch Isaac Sim, build the default world, and run the update loop."""
    simulation_app = SimulationApp({"headless": headless})

    import isaacsim.core.experimental.utils.app as app_utils
    from graphs.debug import format_robot_graph
    from graphs.robot_graph import RobotGraphBuilder
    from robots.action_sources import JsonFileActionSource, LocomotionCommandActionAdapter
    from robots.actions import create_attach_actions_from_contacts
    from robots.command_sources import (
        ConstantCommandSource,
        MultiConstantCommandSource,
        Ros2CmdVelSource,
        SingleModuleCommandAdapter,
        create_attachment_pull_test_command_source,
    )
    from robots.control import (
        MultiModuleVelocityController,
        PlanarVelocityCommand,
        SphericalVelocityController,
    )
    from robots.joints import JointType
    from robots.json_publishers import JsonFileStateGraphPublisher
    from robots.magnetic_attachment import MagneticAttachmentManager
    from robots.reset import SimulationResetManager
    from robots.state_debug import format_state_snapshot
    from robots.state_readers import (
        IsaacRigidBodyStateReader,
        ModuleStateTracker,
        planar_command_to_twist,
    )
    from robots.state_publishers import Ros2ClockPublisher, Ros2OdometryPublisher, Ros2TfPublisher
    from robots.spherical_robot import SphericalRobotBuilder
    from robots.surface_attachment import SurfaceAttachmentConfig, SurfaceAttachmentMode
    from worlds.basic_world import BasicWorldBuilder, BasicWorldConfig
    from worlds.scenario_config import (
        create_linear_scenario,
        create_spherical_robot_configs,
        load_scenario_config,
    )
    from worlds.scenario_obstacles import ScenarioObstacleBuilder
    from worlds.task_evaluator import TaskEvaluator

    if command_source_name == "ros2":
        app_utils.enable_extension("isaacsim.ros2.nodes")
        app_utils.enable_extension("isaacsim.ros2.bridge")
        simulation_app.update()

    if gravity_magnitude <= 0.0:
        raise ValueError("--gravity-magnitude must be greater than zero.")

    world_builder = BasicWorldBuilder(BasicWorldConfig(gravity_magnitude=gravity_magnitude))
    world_builder.build()

    if debug_state_interval <= 0:
        raise ValueError("--debug-state-interval must be greater than zero.")
    if json_publish_interval <= 0:
        raise ValueError("--json-publish-interval must be greater than zero.")
    if scenario_file:
        scenario = load_scenario_config(scenario_file)
        attachment_mode = scenario.attachment_mode
        if reset_every_steps == 0:
            reset_every_steps = scenario.reset_every_steps
    else:
        scenario = create_linear_scenario(
            module_count=module_count,
            module_spacing=module_spacing,
            attachment_mode=attachment_mode,
            reset_every_steps=reset_every_steps,
        )
    if reset_every_steps < 0:
        raise ValueError("reset_every_steps must be zero or greater.")
    if attachment_mode not in ("continuous", "six_axis"):
        raise ValueError("attachment_mode must be 'continuous' or 'six_axis'.")

    ScenarioObstacleBuilder(scenario.box_obstacles).build()

    robots = tuple(
        SphericalRobotBuilder(config).build()
        for config in create_spherical_robot_configs(scenario)
    )
    controlled_robot = robots[0]
    controller = MultiModuleVelocityController(
        {
            robot.module_id: SphericalVelocityController(
                body_path=robot.body_path,
                radius=robot.radius,
            )
            for robot in robots
        }
    )
    state_tracker = ModuleStateTracker(
        readers=tuple(IsaacRigidBodyStateReader(robot.initial_state) for robot in robots),
        attachment_config=SurfaceAttachmentConfig(mode=SurfaceAttachmentMode(attachment_mode)),
    )
    attachment_manager = MagneticAttachmentManager()
    if command_source_name in ("json", "json-file") and not json_action_file:
        raise ValueError("--command-source=json requires --json-action-file.")

    if command_source_name == "ros2":
        locomotion_source = SingleModuleCommandAdapter(
            controlled_robot.module_id,
            Ros2CmdVelSource(
                topic_name=ros2_cmd_vel_topic,
                command_timeout=ros2_command_timeout,
            ),
        )
        state_publisher = Ros2OdometryPublisher(
            body_path=controlled_robot.body_path,
            topic_name=ros2_odom_topic,
            odom_frame_id=controlled_robot.world_frame_id,
            chassis_frame_id=controlled_robot.body_frame_id,
        )
        tf_publisher = Ros2TfPublisher(
            body_path=controlled_robot.body_path,
            topic_name=ros2_tf_topic,
            parent_frame_id=controlled_robot.world_frame_id,
            child_frame_id=controlled_robot.body_frame_id,
        )
        clock_publisher = Ros2ClockPublisher(topic_name=ros2_clock_topic)
    elif command_source_name == "attach-test":
        locomotion_source = SingleModuleCommandAdapter(
            controlled_robot.module_id,
            create_attachment_pull_test_command_source(),
        )
        state_publisher = None
        tf_publisher = None
        clock_publisher = None
    elif command_source_name in ("json", "json-file"):
        locomotion_source = None
        state_publisher = None
        tf_publisher = None
        clock_publisher = None
    else:
        command = PlanarVelocityCommand(vx=demo_forward_speed, yaw_rate=demo_yaw_rate)
        if demo_control_scope == "all":
            locomotion_source = MultiConstantCommandSource(
                {robot.module_id: command for robot in robots}
            )
        else:
            locomotion_source = SingleModuleCommandAdapter(
                controlled_robot.module_id,
                ConstantCommandSource(command),
            )
        state_publisher = None
        tf_publisher = None
        clock_publisher = None

    if command_source_name in ("json", "json-file") or json_action_file:
        action_source = JsonFileActionSource(json_action_file)
    else:
        assert locomotion_source is not None
        action_source = LocomotionCommandActionAdapter(locomotion_source)
    graph_builder = RobotGraphBuilder()
    task_evaluator = TaskEvaluator(scenario)
    json_publisher = (
        JsonFileStateGraphPublisher(
            publish_json_dir,
            write_history=publish_json_history,
        )
        if publish_json_dir
        else None
    )
    reset_manager = SimulationResetManager(
        robots=robots,
        controller=controller,
        attachment_manager=attachment_manager,
    )

    app_utils.play()

    print("MSSR stage created.")
    print(f"Gravity magnitude: {gravity_magnitude:.2f} m/s^2.")
    print(f"Scenario: {scenario.name}.")
    print(
        f"Task: {scenario.curriculum.task_type}, "
        f"stage={scenario.curriculum.stage}, "
        f"difficulty={scenario.curriculum.difficulty:.2f}."
    )
    if scenario.goal is not None:
        print(f"Goal: {scenario.goal.name} at {scenario.goal.position}.")
    print(f"Episode timeout: {scenario.episode_timeout_steps} steps.")
    print(f"Static obstacles created: {len(scenario.box_obstacles)}.")
    print(f"Spherical modules created: {len(robots)}.")
    print(f"Controlled module: {controlled_robot.module_id} at {controlled_robot.body_path}.")
    print(f"Controlled robot frames: {controlled_robot.world_frame_id} -> {controlled_robot.body_frame_id}.")
    print("Module state tracker enabled.")
    print(f"Attachment mode: {attachment_mode}.")
    print(f"Command source: {command_source_name}.")
    if json_action_file:
        print(f"Reading simulation actions from {json_action_file}.")
    if json_publisher is not None:
        print(f"Publishing JSON state/graph payloads to {publish_json_dir}.")
        if publish_json_history:
            print("Appending JSON state/graph history.")
    if reset_every_steps > 0:
        print(f"Reset enabled every {reset_every_steps} steps.")
    if state_publisher is not None:
        print(f"Publishing odometry on {ros2_odom_topic}.")
    if tf_publisher is not None:
        print(f"Publishing TF on {ros2_tf_topic}.")
    if clock_publisher is not None:
        print(f"Publishing clock on {ros2_clock_topic}.")
    if debug_state:
        print(f"Debug state enabled every {debug_state_interval} steps.")
    if debug_graph:
        print(f"Debug graph enabled every {debug_state_interval} steps.")
    if debug_task:
        print(f"Debug task evaluation enabled every {debug_state_interval} steps.")
    if auto_attach_on_contact:
        print("Testing mode enabled: contacts will create physical magnetic joints.")

    step = 0
    try:
        while simulation_app.is_running():
            if max_steps > 0 and step >= max_steps:
                break

            action_source.update()
            actions = action_source.get_actions()
            if actions.reset_requested:
                reset_manager.reset()
            controller.apply(actions.locomotion, dt=1.0 / 60.0)
            simulation_app.update()
            snapshot = state_tracker.update(
                timestamp=step / 60.0,
                last_commands={
                    module_id: planar_command_to_twist(
                        vx=command.vx,
                        vy=command.vy,
                        yaw_rate=command.yaw_rate,
                    )
                    for module_id, command in actions.locomotion.items()
                },
                connected_pairs=attachment_manager.connected_pairs,
                connected_joint_types=attachment_manager.connected_joint_types,
                connected_attachment_metadata=attachment_manager.connected_attachment_metadata,
            )
            magnetic_actions = actions.magnetic
            if auto_attach_on_contact:
                magnetic_actions = magnetic_actions + create_attach_actions_from_contacts(
                    snapshot,
                    joint_type=JointType(auto_attach_joint_type),
                )
            attachment_manager.apply_actions(magnetic_actions, snapshot)
            graph = graph_builder.build(snapshot)
            task_evaluation = task_evaluator.evaluate(snapshot, graph, step)
            if json_publisher is not None and step % json_publish_interval == 0:
                json_publisher.publish(snapshot, graph)
                json_publisher.publish_task_metrics(task_evaluation.to_dict())
            if debug_state and step % debug_state_interval == 0:
                print(format_state_snapshot(snapshot))
            if debug_graph and step % debug_state_interval == 0:
                print(format_robot_graph(graph))
            if debug_task and step % debug_state_interval == 0:
                print(
                    "Task: "
                    f"assembled={task_evaluation.assembled_ratio:.2f} "
                    f"distance_to_goal={task_evaluation.distance_to_goal} "
                    f"success={task_evaluation.is_success} "
                    f"timeout={task_evaluation.is_timeout}"
                )
            if stop_on_task_done and task_evaluation.is_done:
                break
            if reset_every_steps > 0 and step > 0 and step % reset_every_steps == 0:
                reset_manager.reset()
            step += 1
    finally:
        if state_publisher is not None:
            state_publisher.close()
        if tf_publisher is not None:
            tf_publisher.close()
        if clock_publisher is not None:
            clock_publisher.close()
        action_source.close()
        simulation_app.close()


def main() -> None:
    """Run the command-line entry point."""
    args = parse_args()
    run_simulation(
        headless=args.headless,
        max_steps=args.max_steps,
        demo_forward_speed=args.demo_forward_speed,
        demo_yaw_rate=args.demo_yaw_rate,
        demo_control_scope=args.demo_control_scope,
        module_count=args.module_count,
        module_spacing=args.module_spacing,
        scenario_file=args.scenario_file,
        gravity_magnitude=args.gravity_magnitude,
        attachment_mode=args.attachment_mode,
        command_source_name=args.command_source,
        ros2_cmd_vel_topic=args.ros2_cmd_vel_topic,
        ros2_command_timeout=args.ros2_command_timeout,
        ros2_odom_topic=args.ros2_odom_topic,
        ros2_tf_topic=args.ros2_tf_topic,
        ros2_clock_topic=args.ros2_clock_topic,
        json_action_file=args.json_action_file,
        publish_json_dir=args.publish_json_dir,
        json_publish_interval=args.json_publish_interval,
        publish_json_history=args.publish_json_history,
        reset_every_steps=args.reset_every_steps,
        debug_state=args.debug_state,
        debug_state_interval=args.debug_state_interval,
        debug_graph=args.debug_graph,
        debug_task=args.debug_task,
        stop_on_task_done=args.stop_on_task_done,
        auto_attach_on_contact=args.auto_attach_on_contact,
        auto_attach_joint_type=args.auto_attach_joint_type,
    )


if __name__ == "__main__":
    main()
