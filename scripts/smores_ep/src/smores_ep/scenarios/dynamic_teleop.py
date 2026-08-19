from __future__ import annotations

import math
from typing import Any

from smores_ep.config.physics import SmoresActuatorConfig
from smores_ep.config.simulation import DynamicSimulationConfig
from smores_ep.control.teleop import (
    InternalMotionMode,
    Ros2SmoresTeleop,
    SmoresCommand,
)
from smores_ep.isaac.dynamic_stage import (
    ArticulationStateReader,
    BODY_PATH,
    DynamicDriveController,
    MechanismVisualController,
    configure_dynamic_stage,
)


def _demo_command(time_s: float) -> SmoresCommand:
    """Settle, drive on the wheels, then exercise TOP pan/tilt."""

    if time_s < 1.0:
        return SmoresCommand()
    if time_s < 3.0:
        return SmoresCommand(linear_x_m_s=0.06)
    if time_s < 5.0:
        return SmoresCommand(
            tilt_target_rad=-math.pi / 4.0,
            internal_motion=InternalMotionMode.TILT,
        )
    return SmoresCommand(
        angular_z_rad_s=0.5,
        pan_target_rad=0.5,
        internal_motion=InternalMotionMode.PAN,
    )


def _body_pose(stage: Any) -> tuple[tuple[float, float, float], float]:
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(BODY_PATH)
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    translation = matrix.ExtractTranslation()
    body_up = matrix.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0)).GetNormalized()
    return (
        tuple(float(value) for value in translation),
        float(body_up[2]),
    )


def run_dynamic_scenario(
    config: DynamicSimulationConfig,
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
    configure_dynamic_stage(
        stage,
        config.spawn_height_m,
        config.initial_pitch_deg,
    )
    gears = MechanismVisualController(
        stage,
        config.geometry.spur_to_pinion_ratio,
    )
    teleop = (
        Ros2SmoresTeleop(
            config.cmd_vel_topic,
            config.pan_topic,
            config.pan_delta_topic,
            config.tilt_topic,
        )
        if config.ros2_enabled
        else None
    )

    dt_s = 1.0 / config.physics_hz
    SimulationManager.set_physics_dt(dt_s)
    app_utils.play()
    simulation_app.update()
    state_reader = ArticulationStateReader(
        actuators=SmoresActuatorConfig(
            wheel_max_speed_rad_s=config.max_wheel_speed_rad_s,
        )
    )
    drives = DynamicDriveController(
        state_reader,
        config.geometry,
        config.max_wheel_speed_rad_s,
    )

    if not config.headless:
        ViewportManager.set_camera_view(
            "/OmniverseKit_Persp",
            eye=[0.34, -0.42, 0.25],
            target=[0.0, 0.0, 0.04],
        )

    print("SMORES-EP fully dynamic scenario")
    print(f"physics_usd={physics_path}")
    print(
        "root=free_6DoF gravity=9.81m/s^2 contacts=PhysX "
        f"physics_hz={config.physics_hz}"
    )
    print(f"dofs={state_reader.dof_paths}")
    if config.ros2_enabled:
        print(
            f"topics: {config.cmd_vel_topic} [Twist], "
            f"{config.pan_topic} [Float32 rad absolute], "
            f"{config.pan_delta_topic} [Float32 rad relative], "
            f"{config.tilt_topic} [Float32 rad]"
        )

    command = SmoresCommand()
    maximum_steps = config.steps if config.steps > 0 else None
    initial_physics_step = SimulationManager.get_num_physics_steps()
    initial_simulation_time = SimulationManager.get_simulation_time()
    next_log_step = 0
    while simulation_app.is_running():
        physics_step = (
            SimulationManager.get_num_physics_steps() - initial_physics_step
        )
        if maximum_steps is not None and physics_step >= maximum_steps:
            break
        elapsed_s = (
            SimulationManager.get_simulation_time()
            - initial_simulation_time
        )
        if teleop is not None:
            command = teleop.update()
        elif config.demo_enabled:
            command = _demo_command(elapsed_s)
        else:
            command = SmoresCommand()

        left_rate, right_rate = drives.apply(command)
        # One application update advances the configured 240 Hz physics by the
        # required number of substeps for the 60 Hz Kit/render update. Calling
        # SimulationManager.step() here as well used to advance physics twice.
        simulation_app.update()
        joint_state = state_reader.read()
        gears.update(joint_state)

        physics_step = (
            SimulationManager.get_num_physics_steps() - initial_physics_step
        )
        elapsed_s = (
            SimulationManager.get_simulation_time()
            - initial_simulation_time
        )
        if config.log_interval and physics_step >= next_log_step:
            position, body_up_z = _body_pose(stage)
            tilt_target, pan_target = state_reader.target_positions()
            physx_tilt_position, physx_pan_velocity = (
                state_reader.physx_drive_targets()
            )
            print(
                f"t={elapsed_s:7.3f}s "
                f"body=({position[0]:+.3f},{position[1]:+.3f},"
                f"{position[2]:+.3f}) up_z={body_up_z:+.3f} "
                f"wheel_cmd=({left_rate:+.2f},{right_rate:+.2f})rad/s "
                f"wheel_meas=({joint_state.left_wheel_rad_s:+.2f},"
                f"{joint_state.right_wheel_rad_s:+.2f})rad/s "
                f"pan={drives.pan_position_rad:+.3f} "
                f"pan_raw={joint_state.pan_joint_rad:+.3f} "
                f"tilt={-joint_state.tilt_joint_rad:+.3f} "
                f"joint_targets=({tilt_target:+.3f},{pan_target:+.3f}) "
                f"physx_drives=(tilt_pos={physx_tilt_position:+.3f},"
                f"pan_vel={physx_pan_velocity:+.3f})"
            )
            next_log_step += config.log_interval

    final_position, final_up_z = _body_pose(stage)
    print(
        "final_body="
        f"({final_position[0]:+.6f},{final_position[1]:+.6f},"
        f"{final_position[2]:+.6f}) final_up_z={final_up_z:+.6f}"
    )
    app_utils.stop()
