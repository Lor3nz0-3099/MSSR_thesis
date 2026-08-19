from __future__ import annotations

import math
from typing import Any

from smores_ep.config.simulation import MultiModuleLiftSimulationConfig
from smores_ep.control.docking_teleop import Ros2DockingCommandSubscriber
from smores_ep.control.teleop import Ros2SmoresTeleop, SmoresCommand
from smores_ep.isaac.command_router import IsaacMultiModuleCommandRouter
from smores_ep.isaac.docking import IsaacDockingManager
from smores_ep.isaac.dynamic_stage import (
    ArticulationStateReader,
    DynamicDriveController,
    MechanismVisualController,
    configure_dynamic_stage,
)
from smores_ep.isaac.multi_module_stage import clone_module, set_module_pose
from smores_ep.isaac.physics_asset import PHYSICS_ROOT
from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor
from smores_ep.isaac.state_graph_publisher import SmoresStateGraphPublisher
from smores_ep.isaac.support_anchor import IsaacGroundSupportAnchor
from smores_ep.primitives.file_channel import PrimitiveFileChannel
from smores_ep.primitives.pose_control import PoseControllerConfig


ACTIVE_ROOT = PHYSICS_ROOT
CHAIN_ROOT_PREFIX = "/World/smores_ep_chain"


def chain_module_ids(prefix: str, count: int) -> tuple[str, ...]:
    if not prefix or count < 1:
        raise ValueError("A chain needs a non-empty prefix and positive count")
    return tuple(f"{prefix}_{index:02d}" for index in range(1, count + 1))


def chain_module_roots(count: int) -> tuple[str, ...]:
    if count < 1:
        raise ValueError("A chain needs at least one module")
    return tuple(
        f"{CHAIN_ROOT_PREFIX}_{index:02d}"
        for index in range(1, count + 1)
    )


def _world_position(
    stage: Any,
    prim_path: str,
) -> tuple[float, float, float]:
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


def _preconnect_chain(
    docking: IsaacDockingManager,
    chain_ids: tuple[str, ...],
) -> None:
    for first_id, second_id in zip(chain_ids, chain_ids[1:]):
        result = docking.handle_text(f"attach {first_id} {second_id}")
        if not result.accepted or result.connection is None:
            raise RuntimeError(
                f"Could not pre-connect {first_id} to {second_id}: "
                f"{result.message}"
            )
        connection = result.connection
        faces = (
            connection.first_face.face_name,
            connection.second_face.face_name,
        )
        if faces != ("TOP", "BOTTOM"):
            raise RuntimeError(
                f"Unexpected pre-connection {first_id}:{faces[0]} to "
                f"{second_id}:{faces[1]}"
            )


def run_multi_module_lift_scenario(
    config: MultiModuleLiftSimulationConfig,
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

    chain_ids = chain_module_ids(
        config.chain_module_prefix,
        config.chain_module_count,
    )
    chain_roots = chain_module_roots(config.chain_module_count)
    for root in chain_roots:
        clone_module(stage, ACTIVE_ROOT, root)

    configure_dynamic_stage(
        stage,
        config.spawn_height_m,
        config.initial_pitch_deg,
        ACTIVE_ROOT,
    )
    root_spacing_m = config.geometry.top_to_bottom_spacing_m
    for index, root in enumerate(chain_roots, start=1):
        set_module_pose(
            stage,
            root,
            (
                index * root_spacing_m + config.active_to_chain_gap_m,
                0.0,
                config.spawn_height_m,
            ),
            pitch_deg=config.initial_pitch_deg,
        )

    module_roots = {
        config.active_module_id: ACTIVE_ROOT,
        **dict(zip(chain_ids, chain_roots)),
    }
    docking = IsaacDockingManager(stage, module_roots)
    _preconnect_chain(docking, chain_ids)

    active_support = IsaacGroundSupportAnchor(
        stage,
        f"{ACTIVE_ROOT}/body_link",
        yaw_max_effort_nm=config.active_actuators.tilt_max_effort_nm,
    )
    mechanism_visuals = {
        module_id: MechanismVisualController(
            stage,
            config.geometry.spur_to_pinion_ratio,
            root,
        )
        for module_id, root in module_roots.items()
    }
    teleop = (
        Ros2SmoresTeleop(
            config.cmd_vel_topic,
            config.pan_topic,
            config.pan_delta_topic,
            config.tilt_topic,
            graph_path="/ActionGraph/SmoresEPMultiLiftActiveCommands",
        )
        if config.ros2_enabled
        else None
    )
    docking_commands = (
        Ros2DockingCommandSubscriber(
            config.docking_command_topic,
            graph_path="/ActionGraph/SmoresEPMultiLiftDockingCommands",
        )
        if config.ros2_enabled
        else None
    )

    SimulationManager.set_physics_dt(1.0 / config.physics_hz)
    app_utils.play()
    simulation_app.update()

    states = {
        module_id: ArticulationStateReader(
            root,
            config.active_actuators,
            stage,
        )
        for module_id, root in module_roots.items()
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
    roles = {
        config.active_module_id: {
            "current_role": "mobile_lifter",
            "target_role": "mobile_lifter",
            "role_confidence": 1.0,
            "role_source": "scenario_expert",
            "functional_role": {
                "name": "support_and_actuator",
                "effective_dof_count": 4,
                "responsibilities": [
                    "cluster_locomotion",
                    "chain_lifting",
                ],
            },
        },
        **{
            module_id: {
                "current_role": "structural_link",
                "target_role": "structural_link",
                "role_confidence": 1.0,
                "role_source": "scenario_expert",
                "functional_role": {
                    "name": "payload_link",
                    "effective_dof_count": 0,
                    "responsibilities": ["transmit_load"],
                },
            }
            for module_id in chain_ids
        },
    }
    state_graph_publisher = SmoresStateGraphPublisher(
        stage,
        module_roots,
        states,
        docking,
        output_dir=config.primitive_status_file.parent,
        geometry=config.geometry,
        actuator_profiles={
            module_id: config.active_actuators for module_id in module_roots
        },
        roles=roles,
    )

    if not config.headless:
        ViewportManager.set_camera_view(
            "/OmniverseKit_Persp",
            eye=[0.68, -0.62, 0.34],
            target=[0.20, 0.0, 0.055],
        )

    first_chain_id = chain_ids[0]
    last_chain_id = chain_ids[-1]
    first_chain_root = chain_roots[0]
    last_chain_root = chain_roots[-1]
    print("SMORES-EP multi-module cantilever lift scenario")
    print(
        f"active: {config.active_module_id}={ACTIVE_ROOT} [ROS controlled]"
    )
    print(
        f"chain: {', '.join(chain_ids)} [dynamic]; "
        f"{len(docking.connections)} pre-connected TOP(UP)<->BOTTOM joints"
    )
    print(
        f"root spacing={1e3*root_spacing_m:.2f}mm; "
        f"active:TOP to {first_chain_id}:BOTTOM "
        f"gap={1e3*config.active_to_chain_gap_m:.1f}mm"
    )
    print(
        "exaggerated actuator profile: "
        f"wheel={config.active_actuators.wheel_max_effort_nm:.1f}Nm "
        f"tilt={config.active_actuators.tilt_max_effort_nm:.1f}Nm "
        f"pan={config.active_actuators.pan_max_effort_nm:.1f}Nm"
    )
    print(
        "active anti-tip support after docking: "
        f"{'enabled' if config.anchor_active_on_attach else 'disabled'}"
    )
    if config.ros2_enabled:
        print(
            f"attach command: 'attach {config.active_module_id} "
            f"{first_chain_id}' on {config.docking_command_topic}"
        )
        print(
            f"motion topics: {config.cmd_vel_topic}, {config.pan_topic}, "
            f"{config.pan_delta_topic}, {config.tilt_topic}"
        )

    maximum_steps = config.steps if config.steps > 0 else None
    initial_step = SimulationManager.get_num_physics_steps()
    initial_time = SimulationManager.get_simulation_time()
    next_log_step = 0
    command = SmoresCommand()
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
                if result.accepted:
                    active_faces = _connected_faces_for_module(
                        docking,
                        config.active_module_id,
                    )
                    if not active_faces:
                        active_support.release()
                status = "ACCEPTED" if result.accepted else "REJECTED"
                support = (
                    "ON" if active_support.engaged else "OFF"
                )
                print(
                    f"[dock] {status}: {result.message}; "
                    f"anti_tip_support={support}"
                )

        now_s = SimulationManager.get_simulation_time()
        try:
            primitive_goal = primitive_channel.poll_goal()
            if primitive_goal is not None:
                accepted = primitive_executor.submit(primitive_goal, now_s)
                primitive_channel.publish(accepted)
                print(
                    f"[primitive] {accepted.state.value.upper()} "
                    f"{accepted.goal_id}: {accepted.message}"
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
        if primitive_step.statuses:
            serialized_status = "|".join(
                status.to_json() for status in primitive_step.statuses
            )
            status_due = (
                any(
                    status.state.terminal
                    for status in primitive_step.statuses
                )
                or physics_step % state_publish_interval == 0
            )
            if status_due and serialized_status != last_primitive_status:
                primitive_channel.publish_many(
                    primitive_step.statuses,
                    now_s,
                )
                last_primitive_status = serialized_status
                for status in primitive_step.statuses:
                    if status.state.terminal:
                        print(
                            f"[primitive] {status.state.value.upper()} "
                            f"{status.goal_id}: {status.code} "
                            f"{status.message}"
                        )
        routed_commands = primitive_executor.compose_with_baseline(
            {config.active_module_id: command},
            primitive_step.commands,
        )
        active_faces = _connected_faces_for_module(
            docking,
            config.active_module_id,
        )
        if (
            config.anchor_active_on_attach
            and active_faces
            and not active_support.engaged
            and abs(
                routed_commands[config.active_module_id].tilt_target_rad
            )
            > 1.0e-4
        ):
            active_support.engage()
            print("[support] anti_tip_support=ON (chain lift requested)")
        active_support.set_yaw_velocity_target(
            routed_commands[config.active_module_id].angular_z_rad_s
        )

        routed_rates = command_router.apply(routed_commands)
        active_rates = routed_rates.get(
            config.active_module_id,
            (0.0, 0.0),
        )
        simulation_app.update()
        joint_states = {
            module_id: reader.read()
            for module_id, reader in states.items()
        }
        for module_id, visual in mechanism_visuals.items():
            visual.update(joint_states[module_id])

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
            first_position = _world_position(
                stage,
                f"{first_chain_root}/body_link",
            )
            last_position = _world_position(
                stage,
                f"{last_chain_root}/body_link",
            )
            active_joint_state = joint_states[config.active_module_id]
            elapsed = SimulationManager.get_simulation_time() - initial_time
            print(
                f"t={elapsed:7.3f}s "
                f"active=({active_position[0]:+.3f},"
                f"{active_position[1]:+.3f},"
                f"{active_position[2]:+.3f}) "
                f"{first_chain_id}_z={first_position[2]:+.3f}m "
                f"{last_chain_id}_z={last_position[2]:+.3f}m "
                f"wheel_cmd=({active_rates[0]:+.2f},"
                f"{active_rates[1]:+.2f})rad/s "
                f"wheel_actual=("
                f"{active_joint_state.left_wheel_rad_s:+.2f},"
                f"{active_joint_state.right_wheel_rad_s:+.2f})rad/s "
                f"yaw="
                f"{_world_yaw_deg(stage, f'{ACTIVE_ROOT}/body_link'):+.1f}deg "
                f"tilt="
                f"{-math.degrees(active_joint_state.tilt_joint_rad):+.1f}deg "
                f"active_faces={sorted(active_faces)} "
                f"connections={len(docking.connections)}"
            )
            next_log_step += config.log_interval

    print(f"final_connections={len(docking.connections)}")
    app_utils.stop()
