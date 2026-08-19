from __future__ import annotations

import math
from typing import Any

from smores_ep.config.simulation import DockingSimulationConfig
from smores_ep.control.docking_teleop import Ros2DockingCommandSubscriber
from smores_ep.control.teleop import Ros2SmoresTeleop, SmoresCommand
from smores_ep.isaac.docking import IsaacDockingManager
from smores_ep.isaac.command_router import IsaacMultiModuleCommandRouter
from smores_ep.isaac.dynamic_stage import (
    ArticulationStateReader,
    DynamicDriveController,
    MechanismVisualController,
    configure_dynamic_stage,
)
from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor
from smores_ep.isaac.state_graph_publisher import (
    SmoresStateGraphPublisher,
)
from smores_ep.isaac.multi_module_stage import clone_module, set_module_pose
from smores_ep.isaac.physics_asset import PHYSICS_ROOT
from smores_ep.isaac.support_anchor import IsaacGroundSupportAnchor
from smores_ep.primitives.file_channel import PrimitiveFileChannel
from smores_ep.primitives.pose_control import PoseControllerConfig


ACTIVE_ROOT = PHYSICS_ROOT
PASSIVE_ROOT = "/World/smores_ep_passive"


def _world_position(stage: Any, prim_path: str) -> tuple[float, float, float]:
    from pxr import Usd, UsdGeom

    matrix = UsdGeom.Xformable(
        stage.GetPrimAtPath(prim_path)
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return tuple(float(value) for value in matrix.ExtractTranslation())


def _world_yaw_deg(stage: Any, prim_path: str) -> float:
    from pxr import Gf, Usd, UsdGeom

    matrix = UsdGeom.Xformable(
        stage.GetPrimAtPath(prim_path)
    ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    forward = matrix.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
    return math.degrees(math.atan2(float(forward[1]), float(forward[0])))


def _initial_passive_position(
    docking: IsaacDockingManager,
    config: DockingSimulationConfig,
) -> tuple[float, float, float]:
    active_face = next(
        pose
        for pose in docking.face_poses_for(config.active_module_id)
        if pose.face.face_name == config.initial_active_face.upper()
    )
    passive_face = next(
        pose
        for pose in docking.face_poses_for(config.passive_module_id)
        if pose.face.face_name == config.initial_passive_face.upper()
    )
    desired_face_position = tuple(
        position + config.initial_face_gap_m * normal
        for position, normal in zip(
            active_face.position_world_m,
            active_face.outward_normal_world,
        )
    )
    return (
        desired_face_position[0] - passive_face.position_world_m[0],
        desired_face_position[1] - passive_face.position_world_m[1],
        config.spawn_height_m,
    )


def _connected_faces_for_module(
    docking: IsaacDockingManager,
    module_id: str,
) -> set[str]:
    return {
        face.face_name
        for connection in docking.connections
        for face in (connection.first_face, connection.second_face)
        if face.module_id == module_id
    }


def run_two_module_docking_scenario(
    config: DockingSimulationConfig,
    simulation_app: object,
) -> None:
    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.rendering_manager import ViewportManager
    from isaacsim.core.simulation_manager import SimulationManager

    if config.ros2_enabled:
        app_utils.enable_extension("isaacsim.ros2.nodes")
        app_utils.enable_extension("isaacsim.ros2.bridge")
        simulation_app.update()

    physics_path = config.physics_usd.resolve()
    if not physics_path.is_file():
        raise FileNotFoundError(f"Physics USD does not exist: {physics_path}")
    success, stage = stage_utils.open_stage(str(physics_path))
    if not success:
        raise RuntimeError(f"Could not open physics USD: {physics_path}")

    clone_module(stage, ACTIVE_ROOT, PASSIVE_ROOT)
    configure_dynamic_stage(
        stage,
        config.spawn_height_m,
        config.initial_pitch_deg,
        ACTIVE_ROOT,
    )
    set_module_pose(
        stage,
        PASSIVE_ROOT,
        (0.0, 0.0, config.spawn_height_m),
        pitch_deg=config.initial_pitch_deg,
        yaw_deg=config.passive_yaw_deg,
    )

    module_roots = {
        config.active_module_id: ACTIVE_ROOT,
        config.passive_module_id: PASSIVE_ROOT,
    }
    docking = IsaacDockingManager(stage, module_roots)
    active_support = IsaacGroundSupportAnchor(
        stage,
        f"{ACTIVE_ROOT}/body_link",
        yaw_max_effort_nm=config.active_actuators.tilt_max_effort_nm,
    )
    set_module_pose(
        stage,
        PASSIVE_ROOT,
        _initial_passive_position(docking, config),
        pitch_deg=config.initial_pitch_deg,
        yaw_deg=config.passive_yaw_deg,
    )

    active_gears = MechanismVisualController(
        stage,
        config.geometry.spur_to_pinion_ratio,
        ACTIVE_ROOT,
    )
    passive_gears = MechanismVisualController(
        stage,
        config.geometry.spur_to_pinion_ratio,
        PASSIVE_ROOT,
    )
    teleop = (
        Ros2SmoresTeleop(
            config.cmd_vel_topic,
            config.pan_topic,
            config.pan_delta_topic,
            config.tilt_topic,
            graph_path="/ActionGraph/SmoresEPActiveCommands",
        )
        if config.ros2_enabled
        else None
    )
    docking_commands = (
        Ros2DockingCommandSubscriber(
            config.docking_command_topic,
            graph_path="/ActionGraph/SmoresEPDockingCommands",
        )
        if config.ros2_enabled
        else None
    )

    SimulationManager.set_physics_dt(1.0 / config.physics_hz)
    app_utils.play()
    simulation_app.update()
    active_state = ArticulationStateReader(
        ACTIVE_ROOT,
        config.active_actuators,
        stage,
    )
    passive_state = ArticulationStateReader(
        PASSIVE_ROOT,
        config.active_actuators,
        stage,
    )
    states = {
        config.active_module_id: active_state,
        config.passive_module_id: passive_state,
    }
    drives = {
        module_id: DynamicDriveController(
            state,
            config.geometry,
            config.max_wheel_speed_rad_s,
        )
        for module_id, state in states.items()
    }
    command_router = IsaacMultiModuleCommandRouter(
        states,
        drives,
        docking,
    )
    primitive_channel = PrimitiveFileChannel(
        config.primitive_goal_file,
        config.primitive_cancel_file,
        config.primitive_status_file,
    )
    primitive_executor = IsaacPrimitiveExecutor(
        stage,
        module_roots,
        states,
        docking,
        motion_module_ids=tuple(module_roots),
        geometry=config.geometry,
        pose_controller=PoseControllerConfig(
            max_linear_speed_m_s=(
                config.geometry.wheel_radius_m
                * config.max_wheel_speed_rad_s
            ),
        ),
    )
    state_graph_publisher = SmoresStateGraphPublisher(
        stage,
        module_roots,
        states,
        docking,
        output_dir=config.primitive_status_file.parent,
        geometry=config.geometry,
        actuator_profiles={
            config.active_module_id: config.active_actuators,
            config.passive_module_id: config.active_actuators,
        },
        roles={
            config.active_module_id: {
                "current_role": "mobile_docker",
                "target_role": "mobile_docker",
                "role_confidence": 1.0,
                "role_source": "scenario_expert",
                "functional_role": {
                    "name": "locomotor",
                    "effective_dof_count": 4,
                    "responsibilities": [
                        "cluster_locomotion",
                        "face_alignment",
                        "payload_lifting",
                    ],
                },
            },
            config.passive_module_id: {
                "current_role": "dock_target",
                "target_role": "attached_payload",
                "role_confidence": 1.0,
                "role_source": "scenario_expert",
                "functional_role": {
                    "name": "payload",
                    "effective_dof_count": 0,
                    "responsibilities": ["receive_docking"],
                },
            },
        },
    )

    if not config.headless:
        ViewportManager.set_camera_view(
            "/OmniverseKit_Persp",
            eye=[0.42, -0.48, 0.30],
            target=[0.04, 0.0, 0.04],
        )

    print("SMORES-EP two-module rigid docking scenario")
    print(
        f"modules: {config.active_module_id}={ACTIVE_ROOT} [ROS active], "
        f"{config.passive_module_id}={PASSIVE_ROOT} [dynamic passive]"
    )
    print(
        "faces/module: LEFT, RIGHT, TOP, BOTTOM(base-chassis); "
        "attach requires contact, opposed normals and 90deg-compatible clocking"
    )
    print(
        f"initial approach: active {config.initial_active_face.upper()} "
        f"toward passive {config.initial_passive_face.upper()}; "
        f"face_gap={1e3*config.initial_face_gap_m:.1f}mm"
    )
    print(
        "active payload overdrive: "
        f"wheel={config.active_actuators.wheel_max_effort_nm:.1f}Nm "
        f"tilt={config.active_actuators.tilt_max_effort_nm:.1f}Nm "
        f"pan={config.active_actuators.pan_max_effort_nm:.1f}Nm"
    )
    print(
        "active anti-tip support on attach: "
        f"{'enabled' if config.anchor_active_on_attach else 'disabled'}"
    )
    if config.ros2_enabled:
        print(
            f"motion topics: {config.cmd_vel_topic}, {config.pan_topic}, "
            f"{config.pan_delta_topic}, {config.tilt_topic}"
        )
        print(
            f"docking topic: {config.docking_command_topic} [String]; "
            f"'attach {config.active_module_id} "
            f"{config.passive_module_id}' / "
            f"'detach {config.active_module_id} "
            f"{config.passive_module_id}'"
        )
        print(
            "action-like primitive topics are exposed by "
            "ros2_bridge/mssr_file_bridge.py: "
            "/mssr/primitives/goal, /mssr/primitives/cancel, "
            "/mssr/primitives/status"
        )

    maximum_steps = config.steps if config.steps > 0 else None
    initial_step = SimulationManager.get_num_physics_steps()
    initial_time = SimulationManager.get_simulation_time()
    next_log_step = 0
    command = SmoresCommand()
    connection_signature: frozenset[tuple[str, str]] = frozenset()
    last_primitive_status = ""
    state_publish_interval = max(1, config.physics_hz // 20)
    while simulation_app.is_running():
        physics_step = SimulationManager.get_num_physics_steps() - initial_step
        if maximum_steps is not None and physics_step >= maximum_steps:
            break

        if teleop is not None:
            command = teleop.update()
        if docking_commands is not None:
            docking_text = docking_commands.poll()
            if docking_text is not None:
                result = docking.handle_text(docking_text)
                status = "ACCEPTED" if result.accepted else "REJECTED"
                support = (
                    " anti_tip_support=ON"
                    if active_support.engaged
                    else " anti_tip_support=OFF"
                )
                print(f"[dock] {status}: {result.message};{support}")

        now_s = SimulationManager.get_simulation_time()
        try:
            primitive_goal = primitive_channel.poll_goal()
            if primitive_goal is not None:
                primitive_status = primitive_executor.submit(
                    primitive_goal,
                    now_s,
                )
                primitive_channel.publish(primitive_status)
                print(
                    f"[primitive] {primitive_status.state.value.upper()} "
                    f"{primitive_status.goal_id}: "
                    f"{primitive_status.message}"
                )
            cancel_goal_id = primitive_channel.poll_cancel()
            if cancel_goal_id is not None:
                canceled = primitive_executor.cancel(cancel_goal_id, now_s)
                if canceled is not None:
                    primitive_channel.publish(canceled)
                    print(
                        f"[primitive] CANCELED {cancel_goal_id}: "
                        f"{canceled.message}"
                    )
        except (KeyError, TypeError, ValueError) as error:
            print(f"[primitive] REJECTED malformed payload: {error}")

        primitive_step = primitive_executor.step(now_s)
        primitive_statuses = primitive_step.statuses
        if primitive_statuses:
            serialized_status = "|".join(
                status.to_json() for status in primitive_statuses
            )
            status_due = (
                any(status.state.terminal for status in primitive_statuses)
                or physics_step % state_publish_interval == 0
            )
            if status_due and serialized_status != last_primitive_status:
                primitive_channel.publish_many(primitive_statuses, now_s)
                last_primitive_status = serialized_status
                for primitive_status in primitive_statuses:
                    if primitive_status.state.terminal:
                        print(
                            f"[primitive] "
                            f"{primitive_status.state.value.upper()} "
                            f"{primitive_status.goal_id}: "
                            f"{primitive_status.code} "
                            f"{primitive_status.message}"
                        )
        routed_commands = primitive_executor.compose_with_baseline(
            {config.active_module_id: command},
            primitive_step.commands,
        )

        current_connection_signature = frozenset(
            face.key
            for connection in docking.connections
            for face in (connection.first_face, connection.second_face)
        )
        if current_connection_signature != connection_signature:
            connection_signature = current_connection_signature
            print(
                "[router] docking topology changed; actuator modes will "
                "be recomputed per commanded module"
            )
            if not docking.connections:
                active_support.release()

        if (
            config.anchor_active_on_attach
            and docking.connections
            and not active_support.engaged
            and abs(
                routed_commands[config.active_module_id].tilt_target_rad
            )
            > 1.0e-4
        ):
            active_support.engage()
            print("[support] anti_tip_support=ON (tilt requested)")
        active_support.set_yaw_velocity_target(
            routed_commands[config.active_module_id].angular_z_rad_s
        )

        routed_rates = command_router.apply(routed_commands)
        active_rates = routed_rates.get(
            config.active_module_id,
            (0.0, 0.0),
        )
        simulation_app.update()
        active_joint_state = active_state.read()
        passive_joint_state = passive_state.read()
        active_gears.update(active_joint_state)
        passive_gears.update(passive_joint_state)

        physics_step = SimulationManager.get_num_physics_steps() - initial_step
        if physics_step % state_publish_interval == 0:
            state_graph_publisher.publish(
                SimulationManager.get_simulation_time(),
                support_engaged_by_module={
                    config.active_module_id: active_support.engaged,
                },
                experiment_profile=(
                    "enhanced_payload_supported"
                    if config.anchor_active_on_attach
                    else "enhanced_payload_free_body"
                ),
            )
        if config.log_interval and physics_step >= next_log_step:
            active_position = _world_position(
                stage,
                f"{ACTIVE_ROOT}/body_link",
            )
            active_yaw_deg = _world_yaw_deg(
                stage,
                f"{ACTIVE_ROOT}/body_link",
            )
            passive_position = _world_position(
                stage,
                f"{PASSIVE_ROOT}/body_link",
            )
            elapsed = SimulationManager.get_simulation_time() - initial_time
            print(
                f"t={elapsed:7.3f}s "
                f"active=({active_position[0]:+.3f},"
                f"{active_position[1]:+.3f},"
                f"{active_position[2]:+.3f}) "
                f"passive=({passive_position[0]:+.3f},"
                f"{passive_position[1]:+.3f},"
                f"{passive_position[2]:+.3f}) "
                f"wheel_cmd=({active_rates[0]:+.2f},"
                f"{active_rates[1]:+.2f})rad/s "
                f"wheel_actual=({active_joint_state.left_wheel_rad_s:+.2f},"
                f"{active_joint_state.right_wheel_rad_s:+.2f})rad/s "
                f"yaw={active_yaw_deg:+.1f}deg "
                f"tilt={-math.degrees(active_joint_state.tilt_joint_rad):+.1f}deg "
                f"passive_left="
                f"{math.degrees(passive_joint_state.left_wheel_rad):+.1f}deg "
                f"passive_wheels=("
                f"{passive_joint_state.left_wheel_rad_s:+.2f},"
                f"{passive_joint_state.right_wheel_rad_s:+.2f})rad/s "
                f"connections={len(docking.connections)}"
            )
            next_log_step += config.log_interval

    print(f"final_connections={len(docking.connections)}")
    app_utils.stop()
