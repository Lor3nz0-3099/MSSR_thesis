from __future__ import annotations

import math

from smores_ep.config.simulation import KinematicSimulationConfig
from smores_ep.control.differential_drive import (
    PlanarPose,
    integrate_planar_pose,
    twist_to_wheel_rates,
)
from smores_ep.control.pan_tilt import clamp_tilt, move_toward
from smores_ep.control.teleop import (
    InternalMotionMode,
    Ros2SmoresTeleop,
    SmoresCommand,
)
from smores_ep.isaac.kinematic_stage import build_kinematic_stage


def _demo_command(time_s: float) -> SmoresCommand:
    """Exercise every output without requiring a ROS installation."""

    phase = time_s % 8.0
    if phase < 2.0:
        linear, yaw = 0.08, 0.0
    elif phase < 4.0:
        linear, yaw = 0.0, 0.7
    elif phase < 6.0:
        linear, yaw = -0.05, 0.0
    else:
        linear, yaw = 0.0, -0.7
    return SmoresCommand(
        linear_x_m_s=linear,
        angular_z_rad_s=yaw,
        pan_target_rad=0.9 * math.sin(0.6 * time_s),
        tilt_target_rad=0.7 * math.sin(0.35 * time_s),
        internal_motion=(
            InternalMotionMode.PAN
            if phase < 4.0
            else InternalMotionMode.TILT
        ),
    )


def run_kinematic_scenario(
    config: KinematicSimulationConfig,
    simulation_app: object,
) -> None:
    import isaacsim.core.experimental.utils.app as app_utils
    import isaacsim.core.experimental.utils.stage as stage_utils
    from isaacsim.core.rendering_manager import (
        RenderingManager,
        ViewportManager,
    )
    from isaacsim.core.simulation_manager import SimulationManager

    if config.ros2_enabled:
        app_utils.enable_extension("isaacsim.ros2.nodes")
        app_utils.enable_extension("isaacsim.ros2.bridge")
        simulation_app.update()

    built = build_kinematic_stage(
        stage_utils,
        config.visual_usd,
        config.geometry,
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

    dt_s = 1.0 / config.update_hz
    SimulationManager.set_physics_dt(dt_s)
    app_utils.play()
    simulation_app.update()
    if not config.headless:
        ViewportManager.set_camera_view(
            "/OmniverseKit_Persp",
            eye=[0.34, -0.42, 0.28],
            target=[0.0, 0.0, 0.04],
        )

    pose = PlanarPose()
    left_angle = 0.0
    right_angle = 0.0
    pan_angle = 0.0
    tilt_angle = 0.0
    command = SmoresCommand()
    maximum_steps = config.steps if config.steps > 0 else None

    print("SMORES-EP kinematic scenario")
    print(f"visual_usd={config.visual_usd.resolve()}")
    print(
        "frame=ROS_REP_103 (+X forward, +Y left, +Z up) "
        f"track_width={config.geometry.track_width_m:.6f}m "
        f"wheel_radius={config.geometry.wheel_radius_m:.6f}m"
    )
    if config.ros2_enabled:
        print(
            f"topics: {config.cmd_vel_topic} [Twist], "
            f"{config.pan_topic} [Float32 rad absolute], "
            f"{config.pan_delta_topic} [Float32 rad relative], "
            f"{config.tilt_topic} [Float32 rad]"
        )

    step = 0
    while simulation_app.is_running():
        if maximum_steps is not None and step >= maximum_steps:
            break
        SimulationManager.step()
        simulation_app.update()
        if teleop is not None:
            command = teleop.update()
        elif config.demo_enabled:
            command = _demo_command(step * dt_s)
        else:
            command = SmoresCommand()

        if command.internal_motion is InternalMotionMode.PAN:
            pan_angle = move_toward(
                pan_angle,
                command.pan_target_rad,
                config.max_pan_speed_rad_s,
                dt_s,
            )
        elif command.internal_motion is InternalMotionMode.TILT:
            tilt_target = clamp_tilt(
                command.tilt_target_rad,
                config.geometry.tilt_min_rad,
                config.geometry.tilt_max_rad,
            )
            tilt_angle = move_toward(
                tilt_angle,
                tilt_target,
                config.max_tilt_speed_rad_s,
                dt_s,
            )
        rates = twist_to_wheel_rates(
            command.linear_x_m_s,
            command.angular_z_rad_s,
            config.geometry.wheel_radius_m,
            config.geometry.track_width_m,
        )
        left_angle += rates.left_rad_s * dt_s
        right_angle += rates.right_rad_s * dt_s
        pose = integrate_planar_pose(
            pose,
            command.linear_x_m_s,
            command.angular_z_rad_s,
            dt_s,
        )
        built.model.set_state(
            pose,
            config.geometry.ground_contact_height_m(tilt_angle),
            left_angle,
            right_angle,
            pan_angle,
            tilt_angle,
        )

        if not config.headless:
            RenderingManager.render()
        if config.log_interval and step % config.log_interval == 0:
            print(
                f"t={step * dt_s:7.3f}s "
                f"pose=({pose.x_m:+.3f},{pose.y_m:+.3f},"
                f"{pose.yaw_rad:+.3f}) "
                f"pan={pan_angle:+.3f} tilt={tilt_angle:+.3f}"
            )
        step += 1

    app_utils.stop()
