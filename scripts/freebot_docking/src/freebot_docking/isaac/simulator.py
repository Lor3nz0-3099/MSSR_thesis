from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import numpy as np

from freebot_docking.config.geometry import ShellGeometry
from freebot_docking.config.magnet import MagnetConfig
from freebot_docking.config.simulation import ShellContactFrictionConfig
from freebot_docking.control.teleop import Ros2TwistTeleop, TwistCommand
from freebot_docking.control.wheel_drive import (
    WheelDriveConfig,
    WheelTorqueLimits,
    WheelVelocityTargets,
    apply_climb_heading_correction,
    apply_wheel_targets,
    motor_torque_limits,
    signed_heading_error_rad,
    twist_to_wheel_targets,
)
from freebot_docking.diagnostics.contacts import (
    IsaacContactDiagnostics,
    figure9_balance_residual,
    paper_required_ground_friction,
)
from freebot_docking.isaac.force_applier import apply_action_reaction_pair
from freebot_docking.isaac.debug_draw import IsaacForceDebugDraw
from freebot_docking.isaac.module_handles import FreebotModuleHandles
from freebot_docking.isaac.stage_builder import (
    IsaacStageConfig,
    build_freebot_stage,
)
from freebot_docking.physics.external_magnet import (
    compute_external_magnetic_interaction,
)
from freebot_docking.scenarios.two_module_docking import (
    compute_internal_magnetic_preload_interaction,
)


@dataclass(frozen=True)
class IsaacSimulationConfig:
    """Runtime controls for the first force-driven Isaac validation."""

    stage: IsaacStageConfig
    physics_hz: int = 240
    steps: int = 12_000
    log_interval: int = 240
    headless: bool = False
    ros2_teleop: bool = True
    cmd_vel_topic: str = "/cmd_vel"
    cmd_timeout_s: float = 0.5
    debug_draw: bool = False
    debug_force_scale_m_per_n: float = 0.003
    internal_preload_force_n: float = 9.5
    external_force_target: Literal["active-shell", "active-carrier"] = (
        "active-carrier"
    )
    shell_geometry: ShellGeometry = field(default_factory=ShellGeometry)
    magnet_config: MagnetConfig = field(default_factory=MagnetConfig)
    wheel_drive: WheelDriveConfig = field(default_factory=WheelDriveConfig)
    contact_friction: ShellContactFrictionConfig = field(
        default_factory=ShellContactFrictionConfig
    )

    def __post_init__(self) -> None:
        if self.external_force_target not in {"active-shell", "active-carrier"}:
            raise ValueError(
                "External force target must be 'active-shell' or "
                "'active-carrier'"
            )


def run_isaac_simulation(config: IsaacSimulationConfig) -> None:
    """Run state readback, magnetic evaluation and wrench application."""

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": config.headless})
    teleop: Ros2TwistTeleop | None = None
    try:
        import isaacsim.core.experimental.utils.app as app_utils
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.experimental.prims import RigidPrim, XformPrim
        from isaacsim.core.rendering_manager import RenderingManager
        from isaacsim.core.simulation_manager import SimulationManager

        if config.ros2_teleop:
            app_utils.enable_extension("isaacsim.ros2.nodes")
            app_utils.enable_extension("isaacsim.ros2.bridge")
            simulation_app.update()

        stage_config = replace(
            config.stage,
            shell_geometry=config.shell_geometry,
            materials=replace(
                config.stage.materials,
                shell_static_friction=(
                    config.contact_friction.static_friction_coefficient
                ),
                shell_dynamic_friction=(
                    config.contact_friction.dynamic_friction_coefficient
                ),
            ),
        )
        built = build_freebot_stage(stage_utils, stage_config)
        wheel_drive_instance = "angular"
        zero_targets = WheelVelocityTargets(0.0, 0.0)
        apply_wheel_targets(
            built.stage,
            built.active_root,
            zero_targets,
            config.wheel_drive,
            drive_instance=wheel_drive_instance,
        )
        apply_wheel_targets(
            built.stage,
            built.passive_root,
            zero_targets,
            WheelDriveConfig(
                linear_scale_deg_s=config.wheel_drive.linear_scale_deg_s,
                yaw_scale_deg_s=config.wheel_drive.yaw_scale_deg_s,
                damping=0.0,
                no_load_speed_deg_s=config.wheel_drive.no_load_speed_deg_s,
                stall_torque_nm=0.0,
                armature_kg_m2=config.wheel_drive.armature_kg_m2,
                zero_command_brake_torque_nm=0.0,
                climb_heading_enabled=False,
            ),
            WheelTorqueLimits(0.0, 0.0),
            drive_instance=wheel_drive_instance,
        )

        for _ in range(5):
            simulation_app.update()
        active = FreebotModuleHandles.create(
            built.active_root,
            RigidPrim,
            XformPrim,
            stage_config.wheel_radial_compliance.enabled,
        )
        passive = FreebotModuleHandles.create(
            built.passive_root,
            RigidPrim,
            XformPrim,
            stage_config.wheel_radial_compliance.enabled,
        )
        if stage_config.wheel_radial_compliance.enabled:
            radial_axes_local = tuple(
                np.asarray(
                    built.stage.GetPrimAtPath(
                        f"{built.active_root}/joints/"
                        f"{side}_wheel_radial_joint"
                    )
                    .GetAttribute("freebot:radialComplianceAxis")
                    .Get(),
                    dtype=np.float64,
                )
                for side in ("left", "right")
            )
        else:
            radial_axes_local = (
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
            )
        radial_reference_m = active.wheel_radial_projections_m(
            radial_axes_local
        )
        contact_diagnostics = IsaacContactDiagnostics(
            RigidPrim,
            (
                (
                    "left_wheel-active_shell",
                    f"{built.active_root}/left_wheel_link",
                    f"{built.active_root}/shell_link",
                ),
                (
                    "right_wheel-active_shell",
                    f"{built.active_root}/right_wheel_link",
                    f"{built.active_root}/shell_link",
                ),
                (
                    "caster_1-active_shell",
                    f"{built.active_root}/caster_1_ball_link",
                    f"{built.active_root}/shell_link",
                ),
                (
                    "caster_2-active_shell",
                    f"{built.active_root}/caster_2_ball_link",
                    f"{built.active_root}/shell_link",
                ),
                (
                    "active_shell-passive_shell",
                    f"{built.active_root}/shell_link",
                    f"{built.passive_root}/shell_link",
                ),
                (
                    "active_shell-ground",
                    f"{built.active_root}/shell_link",
                    "/World/freebot_ground",
                ),
            ),
        )
        force_debug = IsaacForceDebugDraw(
            built.stage,
            config.debug_draw,
            config.debug_force_scale_m_per_n,
        )

        time_step = 1.0 / float(config.physics_hz)
        SimulationManager.set_physics_dt(time_step)
        app_utils.play()
        simulation_app.update()
        if config.ros2_teleop:
            teleop = Ros2TwistTeleop(
                topic_name=config.cmd_vel_topic,
                command_timeout_s=config.cmd_timeout_s,
                physics_hz=config.physics_hz,
            )

        print(
            "FreeBOT Isaac scenario: two identical CAD modules; "
            f"active=dynamic passive_fixed={config.stage.passive_fixed}"
        )
        print(
            f"external=FreeBOT Fig.4+Fig.5 internal_preload="
            f"{config.internal_preload_force_n:.3f}N "
            f"external_target={config.external_force_target} "
            "contact_friction=PhysX"
        )
        print(
            f"ROS2={'on' if config.ros2_teleop else 'off'} "
            f"topic={config.cmd_vel_topic} physics_hz={config.physics_hz}"
        )
        print(
            f"colliders: duplicate_removed="
            f"{built.removed_duplicate_colliders} "
            f"wheel_SDF_replaced={built.replaced_wheel_colliders} "
            f"caster_SDF_replaced={built.replaced_caster_colliders}"
        )
        wheel_mount_mass_kg = (
            stage_config.wheel_radial_compliance.mount_mass_kg
            if stage_config.wheel_radial_compliance.enabled
            else 0.0
        )
        print(
            "masses: "
            f"shell={config.stage.masses.shell_kg*1e3:.3f}g "
            f"internal_link={active.body_mass_kg(active.internal_body)*1e3:.3f}g "
            f"wheel_mount_each="
            f"{wheel_mount_mass_kg*1e3:.3f}g "
            f"wheel_each={config.stage.masses.wheel_kg*1e3:.3f}g "
            f"caster_each={config.stage.masses.caster_ball_kg*1e3:.3f}g "
            f"module={active.total_mass_kg()*1e3:.3f}g"
        )
        print(
            "wheel_motor: "
            f"no_load={config.wheel_drive.no_load_speed_deg_s:.1f}deg/s "
            f"stall={config.wheel_drive.stall_torque_nm:.3f}Nm "
            f"armature={config.wheel_drive.armature_kg_m2:.6f}kgm2 "
            f"brake={config.wheel_drive.zero_command_brake_torque_nm:.3f}Nm "
            f"vertical_plane_control={config.wheel_drive.climb_heading_enabled}"
        )
        radial = stage_config.wheel_radial_compliance
        print(
            "wheel_radial_compliance: "
            f"enabled={radial.enabled} "
            f"joints={built.radial_suspension_joints} "
            f"travel=[{-1e3*radial.inward_travel_m:+.2f},"
            f"{1e3*radial.outward_travel_m:+.2f}]mm "
            f"rest={1e3*radial.rest_position_m:+.2f}mm "
            f"stiffness={radial.stiffness_n_per_m:.1f}N/m "
            f"damping={radial.damping_n_s_per_m:.1f}Ns/m "
            f"max_force={radial.max_force_n:.1f}N"
        )
        print(
            "wheel_tire: "
            f"loaded_radius={1e3*stage_config.running_gear.tire_outer_radius_m:.3f}mm "
            f"envelope_radius={1e3*stage_config.running_gear.tire_collision_radius_m:.3f}mm "
            f"precompression={1e3*stage_config.running_gear.tire_precompression_m:.3f}mm "
            f"stiffness={stage_config.materials.wheel_contact_stiffness_n_per_m:.1f}N/m "
            f"damping={stage_config.materials.wheel_contact_damping_n_s_per_m:.1f}Ns/m "
            f"caster_clearance={1e3*stage_config.running_gear.caster_nominal_clearance_m:.3f}mm "
            f"caster_envelope={1e3*stage_config.running_gear.caster_collision_radius_m:.3f}mm "
            f"caster_precompression={1e3*stage_config.running_gear.caster_precompression_m:.3f}mm "
            f"caster_stiffness={stage_config.materials.caster_contact_stiffness_n_per_m:.1f}N/m "
            f"caster_damping={stage_config.materials.caster_contact_damping_n_s_per_m:.1f}Ns/m"
        )
        print(
            "friction(static,dynamic): "
            f"shell=({stage_config.materials.shell_static_friction:.3f},"
            f"{stage_config.materials.shell_dynamic_friction:.3f}) "
            f"wheel=({stage_config.materials.wheel_static_friction:.3f},"
            f"{stage_config.materials.wheel_dynamic_friction:.3f}) "
            f"caster=({stage_config.materials.caster_static_friction:.3f},"
            f"{stage_config.materials.caster_dynamic_friction:.3f}) "
            f"ground=({stage_config.materials.ground_static_friction:.3f},"
            f"{stage_config.materials.ground_dynamic_friction:.3f}) "
            "combine=max"
        )
        if config.debug_draw:
            print(
                "debug_draw: yellow=magnet_axis white=magnet_to_patch "
                "green=external_active orange=external_reaction "
                "blue=internal_carrier cyan=internal_shell "
                "red=contact_normal magenta=contact_friction "
                f"force_scale={config.debug_force_scale_m_per_n:.4f}m/N"
            )

        targets = zero_targets
        torque_limits = WheelTorqueLimits(0.0, 0.0)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        initial_center_line = (
            np.asarray(config.stage.passive_shell_center_world, dtype=np.float64)
            - np.asarray(config.stage.active_shell_center_world, dtype=np.float64)
        )
        initial_center_line -= np.dot(initial_center_line, world_up) * world_up
        desired_climb_plane_normal = np.cross(world_up, initial_center_line)
        desired_climb_plane_normal /= np.linalg.norm(desired_climb_plane_normal)
        for step in range(config.steps):
            command = teleop.update() if teleop is not None else TwistCommand()
            active_shell = active.shell_state(config.shell_geometry)
            passive_shell = passive.shell_state(config.shell_geometry)
            active_magnet = active.magnet_state(config.magnet_config)

            desired_targets = twist_to_wheel_targets(
                command.linear_x,
                command.angular_z,
                config.wheel_drive,
            )
            heading_error_rad = 0.0
            heading_control_active = False
            center_distance = float(
                np.linalg.norm(
                    passive_shell.center_world - active_shell.center_world
                )
            )
            shell_gap_for_control = max(
                0.0,
                center_distance - 2.0 * config.shell_geometry.outer_radius_m,
            )
            if (
                config.wheel_drive.climb_heading_enabled
                and shell_gap_for_control
                <= config.wheel_drive.climb_heading_capture_gap_m
                and abs(command.linear_x) > 1.0e-6
                and abs(command.angular_z) <= 1.0e-6
            ):
                carrier_axle = active.carrier_direction_world([0.0, 1.0, 0.0])
                heading_error_rad = signed_heading_error_rad(
                    carrier_axle,
                    desired_climb_plane_normal,
                    -active_magnet.axis_world,
                )
                desired_targets = apply_climb_heading_correction(
                    desired_targets,
                    heading_error_rad,
                    config.wheel_drive,
                )
                heading_control_active = True

            actual_left_before, actual_right_before = active.wheel_speeds_deg_s()
            targets = desired_targets
            torque_limits = motor_torque_limits(
                targets,
                actual_left_before,
                actual_right_before,
                config.wheel_drive,
            )
            apply_wheel_targets(
                built.stage,
                built.active_root,
                targets,
                config.wheel_drive,
                torque_limits,
                drive_instance=wheel_drive_instance,
            )

            internal = compute_internal_magnetic_preload_interaction(
                active_shell,
                config.shell_geometry,
                active_magnet,
                config.magnet_config,
                config.internal_preload_force_n,
            )
            external = compute_external_magnetic_interaction(
                active_shell,
                config.shell_geometry,
                passive_shell,
                config.shell_geometry,
                active_magnet,
                config.magnet_config,
            )
            apply_action_reaction_pair(
                active.internal_body,
                internal.carrier_wrench,
                active.shell_body,
                internal.shell_wrench,
            )
            if config.external_force_target == "active-shell":
                external_active_body = active.shell_body
                external_active_wrench = external.active_shell_wrench
            else:
                external_active_body = active.internal_body
                external_active_wrench = external.active_carrier_wrench
            apply_action_reaction_pair(
                external_active_body,
                external_active_wrench,
                passive.shell_body,
                external.passive_shell_wrench,
                apply_second=not config.stage.passive_fixed,
            )

            passive_magnet = passive.magnet_state(config.magnet_config)
            passive_internal = compute_internal_magnetic_preload_interaction(
                passive_shell,
                config.shell_geometry,
                passive_magnet,
                config.magnet_config,
                config.internal_preload_force_n,
            )
            apply_action_reaction_pair(
                passive.internal_body,
                passive_internal.carrier_wrench,
                passive.shell_body,
                passive_internal.shell_wrench,
                apply_second=not config.stage.passive_fixed,
            )

            SimulationManager.step()
            if not config.headless:
                RenderingManager.render()
            simulation_app.update()

            should_log = (
                config.log_interval > 0
                and step % config.log_interval == 0
            )
            contact_readings = (
                contact_diagnostics.snapshots(time_step)
                if config.debug_draw or should_log
                else None
            )
            if config.debug_draw:
                assert contact_readings is not None
                force_debug.update(
                    active_magnet=active_magnet,
                    passive_magnet=passive_magnet,
                    external=external,
                    external_force_target=config.external_force_target,
                    active_internal=internal,
                    passive_internal=passive_internal,
                    contacts=contact_readings,
                )

            if should_log:
                assert contact_readings is not None
                total_external = float(
                    np.linalg.norm(external.force_on_active_world)
                )
                center_normal = (
                    passive_shell.center_world - active_shell.center_world
                )
                center_normal /= np.linalg.norm(center_normal)
                magnet_radial = (
                    active_magnet.center_world - active_shell.center_world
                )
                magnet_radial /= np.linalg.norm(magnet_radial)
                radial_angle_deg = float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                np.dot(magnet_radial, center_normal),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                )
                axis_radial_error_deg = float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                np.dot(
                                    active_magnet.axis_world,
                                    magnet_radial,
                                ),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                )
                patch_normal_offset_deg = float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                np.dot(
                                    external.passive_surface_normal_world,
                                    center_normal,
                                ),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                )
                perpendicular_force_world = (
                    external.force_on_active_world
                    - external.parallel_force_n * center_normal
                )
                perpendicular_up_n = float(
                    np.dot(perpendicular_force_world, world_up)
                )
                external_up_n = float(
                    np.dot(external.force_on_active_world, world_up)
                )
                carrier_applied_force = (
                    internal.force_on_carrier_world
                    + (
                        external.force_on_active_world
                        if config.external_force_target == "active-carrier"
                        else np.zeros(3, dtype=np.float64)
                    )
                )
                carrier_radial_load_n = float(
                    np.dot(carrier_applied_force, magnet_radial)
                )
                carrier_tangent_load_n = float(
                    np.linalg.norm(
                        carrier_applied_force
                        - carrier_radial_load_n * magnet_radial
                    )
                )
                actual_left, actual_right = active.wheel_speeds_deg_s()
                clearances = active.inner_shell_clearances_m(
                    active_shell.center_world,
                    config.shell_geometry,
                    stage_config.running_gear,
                )
                radial_positions_m = active.wheel_radial_projections_m(
                    radial_axes_local
                )
                radial_displacements_m = (
                    radial_positions_m[0] - radial_reference_m[0],
                    radial_positions_m[1] - radial_reference_m[1],
                )
                left_wheel_reading = contact_readings[
                    "left_wheel-active_shell"
                ]
                right_wheel_reading = contact_readings[
                    "right_wheel-active_shell"
                ]
                caster_readings = tuple(
                    contact_readings[label]
                    for label in (
                        "caster_1-active_shell",
                        "caster_2-active_shell",
                    )
                )
                left_wheel_normal_n = float(
                    np.linalg.norm(left_wheel_reading.normal_force_world)
                )
                right_wheel_normal_n = float(
                    np.linalg.norm(right_wheel_reading.normal_force_world)
                )
                wheel_normal_n = left_wheel_normal_n + right_wheel_normal_n
                wheel_compressions_m = (
                    max(0.0, -clearances[0]),
                    max(0.0, -clearances[1]),
                )
                caster_compressions_m = (
                    max(0.0, -clearances[2]),
                    max(0.0, -clearances[3]),
                )
                caster_normal_n = sum(
                    float(np.linalg.norm(reading.normal_force_world))
                    for reading in caster_readings
                )
                shell_contact_reading = contact_readings[
                    "active_shell-passive_shell"
                ]
                ground_reading = contact_readings["active_shell-ground"]
                gravity_force, gravity_moment = active.gravity_wrench_about(
                    active_shell.center_world
                )
                magnetic_moment = (
                    external.active_carrier_wrench.expressed_at(
                        active_shell.center_world
                    ).torque
                )
                shell_contact_moment = (
                    shell_contact_reading.total_moment_about(
                        active_shell.center_world
                    )
                )
                ground_contact_moment = ground_reading.total_moment_about(
                    active_shell.center_world
                )
                # Wheel/caster sensors report the wrench acting on the
                # running gear.  Its opposite is the actual contact wrench
                # transmitted to the shell; unlike motor maxForce, this is a
                # resolved PhysX quantity rather than an actuator limit.
                wheel_to_shell_moment = -(
                    left_wheel_reading.total_moment_about(
                        active_shell.center_world
                    )
                    + right_wheel_reading.total_moment_about(
                        active_shell.center_world
                    )
                )
                caster_to_shell_moment = -sum(
                    (
                        reading.total_moment_about(
                            active_shell.center_world
                        )
                        for reading in caster_readings
                    ),
                    start=np.zeros(3, dtype=np.float64),
                )
                internal_shell_moment = internal.shell_wrench.expressed_at(
                    active_shell.center_world
                ).torque
                external_shell_moment = (
                    external.active_shell_wrench.expressed_at(
                        active_shell.center_world
                    ).torque
                    if config.external_force_target == "active-shell"
                    else np.zeros(3, dtype=np.float64)
                )
                shell_gravity_force = np.array(
                    [
                        0.0,
                        0.0,
                        -active.body_mass_kg(active.shell_body) * 9.81,
                    ],
                    dtype=np.float64,
                )
                shell_gravity_moment = np.cross(
                    active.body_com_world(active.shell_body)
                    - active_shell.center_world,
                    shell_gravity_force,
                )
                shell_net_moment = (
                    internal_shell_moment
                    + external_shell_moment
                    + wheel_to_shell_moment
                    + caster_to_shell_moment
                    + shell_contact_moment
                    + ground_contact_moment
                    + shell_gravity_moment
                )
                static_force_residual = (
                    external.force_on_active_world
                    + shell_contact_reading.total_force_world
                    + ground_reading.total_force_world
                    + gravity_force
                )
                static_moment_residual = (
                    magnetic_moment
                    + shell_contact_moment
                    + ground_contact_moment
                    + gravity_moment
                )
                climb_moments_nm = {
                    "mag": float(
                        np.dot(magnetic_moment, desired_climb_plane_normal)
                    ),
                    "grav": float(
                        np.dot(gravity_moment, desired_climb_plane_normal)
                    ),
                    "wheel_shell": float(
                        np.dot(
                            wheel_to_shell_moment,
                            desired_climb_plane_normal,
                        )
                    ),
                    "caster_shell": float(
                        np.dot(
                            caster_to_shell_moment,
                            desired_climb_plane_normal,
                        )
                    ),
                    "passive": float(
                        np.dot(
                            shell_contact_moment,
                            desired_climb_plane_normal,
                        )
                    ),
                    "ground": float(
                        np.dot(
                            ground_contact_moment,
                            desired_climb_plane_normal,
                        )
                    ),
                    "net": float(
                        np.dot(
                            static_moment_residual,
                            desired_climb_plane_normal,
                        )
                    ),
                }
                shell_net_climb_moment_nm = float(
                    np.dot(shell_net_moment, desired_climb_plane_normal)
                )
                drive_state = (
                    "brake"
                    if abs(targets.left_deg_s)
                    <= config.wheel_drive.zero_command_threshold_deg_s
                    and abs(targets.right_deg_s)
                    <= config.wheel_drive.zero_command_threshold_deg_s
                    else "drive"
                )
                module_com_radius_m = float(
                    np.linalg.norm(
                        active.module_com_world()
                        - active_shell.center_world
                    )
                )
                required_ground_mu = (
                    paper_required_ground_friction(
                        angle_deg=external.lifting_angle_deg,
                        parallel_force_n=external.parallel_force_n,
                        perpendicular_force_n=external.perpendicular_force_n,
                        gravity_force_n=float(np.linalg.norm(gravity_force)),
                        shell_radius_m=config.shell_geometry.outer_radius_m,
                        mechanism_com_radius_m=module_com_radius_m,
                        shell_shell_friction_coefficient=(
                            config.contact_friction.static_friction_coefficient
                        ),
                    )
                    if external.lifting_angle_deg <= 90.0
                    else float("inf")
                )
                shell_tangent_force = (
                    shell_contact_reading.friction_force_world
                )
                paper_normal = center_normal
                paper_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                equation_5 = figure9_balance_residual(
                    gravity_force_n=float(np.linalg.norm(gravity_force)),
                    perpendicular_force_n=float(
                        np.dot(external.force_on_active_world, paper_up)
                    ),
                    parallel_force_n=float(
                        np.dot(external.force_on_active_world, paper_normal)
                    ),
                    shell_friction_n=float(
                        np.dot(shell_tangent_force, paper_up)
                    ),
                    shell_normal_n=float(
                        -np.dot(
                            shell_contact_reading.normal_force_world,
                            paper_normal,
                        )
                    ),
                    ground_friction_n=float(
                        np.dot(
                            ground_reading.friction_force_world,
                            paper_normal,
                        )
                    ),
                    ground_normal_n=float(
                        np.dot(
                            ground_reading.normal_force_world,
                            paper_up,
                        )
                    ),
                    shell_radius_m=config.shell_geometry.outer_radius_m,
                    com_radius_m=module_com_radius_m,
                    angle_deg=min(external.lifting_angle_deg, 90.0),
                )
                print(
                    f"t={step*time_step:7.3f}s "
                    f"gap={1e3*external.shell_gap_m:+7.2f}mm "
                    f"theta_axis={external.lifting_angle_deg:6.2f}deg "
                    f"theta_Apar={external.parallel_curve_angle_deg:6.2f}deg "
                    f"theta_radial={radial_angle_deg:6.2f}deg "
                    f"axis_radial_err={axis_radial_error_deg:5.2f}deg "
                    f"patch_offset={patch_normal_offset_deg:5.2f}deg "
                    f"patch_ray={'ok' if external.line_of_action_valid else 'off'} "
                    f"mag_surface_gap={1e3*external.magnet_surface_gap_m:6.2f}mm "
                    f"Fint={internal.preload_force_n:6.3f}N "
                    f"Fext={total_external:6.3f}N "
                    f"Apar={external.parallel_force_n:6.3f}N "
                    f"Aperp={external.perpendicular_force_n:6.3f}N "
                    f"Aperp_up={perpendicular_up_n:+6.3f}N "
                    f"Aup={external_up_n:+6.3f}N "
                    f"Fcarrier_radial={carrier_radial_load_n:6.3f}N "
                    f"Fcarrier_tangent={carrier_tangent_load_n:6.3f}N "
                    "contact=PhysX "
                    f"wheel=({targets.left_deg_s:+6.1f},"
                    f"{targets.right_deg_s:+6.1f})deg/s "
                    f"wheel_req=({desired_targets.left_deg_s:+6.1f},"
                    f"{desired_targets.right_deg_s:+6.1f})deg/s "
                    f"actual=({actual_left:+6.1f},"
                    f"{actual_right:+6.1f})deg/s "
                    f"drive_state={drive_state} "
                    f"motor_tau_limit=({torque_limits.left_nm:.3f},"
                    f"{torque_limits.right_nm:.3f})Nm "
                    f"shell_z={1e3*active_shell.center_world[2]:7.2f}mm "
                    "shell_omega=("
                    f"{np.degrees(active_shell.angular_velocity_world[0]):+6.1f},"
                    f"{np.degrees(active_shell.angular_velocity_world[1]):+6.1f},"
                    f"{np.degrees(active_shell.angular_velocity_world[2]):+6.1f}"
                    ")deg/s "
                    f"heading={'on' if heading_control_active else 'off'} "
                    f"heading_err={np.degrees(heading_error_rad):+5.1f}deg "
                    f"proxy_gap_mm=(wheel {1e3*clearances[0]:+.2f},"
                    f"{1e3*clearances[1]:+.2f}; caster "
                    f"{1e3*clearances[2]:+.2f},"
                    f"{1e3*clearances[3]:+.2f}) "
                    f"tire_comp_mm=({1e3*wheel_compressions_m[0]:.2f},"
                    f"{1e3*wheel_compressions_m[1]:.2f}) "
                    f"caster_comp_mm=({1e3*caster_compressions_m[0]:.2f},"
                    f"{1e3*caster_compressions_m[1]:.2f}) "
                    f"wheel_radial_mm=({1e3*radial_displacements_m[0]:+.2f},"
                    f"{1e3*radial_displacements_m[1]:+.2f}) "
                    f"Nwheel=({left_wheel_normal_n:.3f},"
                    f"{right_wheel_normal_n:.3f})N "
                    f"Ncaster={caster_normal_n:.3f}N "
                    f"N1={np.linalg.norm(shell_contact_reading.normal_force_world):.3f}N "
                    f"f1={np.linalg.norm(shell_tangent_force):.3f}N "
                    f"N2={np.linalg.norm(ground_reading.normal_force_world):.3f}N "
                    f"f2={np.linalg.norm(ground_reading.friction_force_world):.3f}N "
                    f"mu2_req={required_ground_mu:.3f} "
                    f"Fbalance={np.linalg.norm(static_force_residual):.3f}N "
                    f"Tbalance={np.linalg.norm(static_moment_residual):.4f}Nm "
                    f"Tshell_climb={shell_net_climb_moment_nm:+.4f}Nm "
                    "Tclimb=("
                    f"mag {climb_moments_nm['mag']:+.4f},"
                    f"grav {climb_moments_nm['grav']:+.4f},"
                    f"wheelShell {climb_moments_nm['wheel_shell']:+.4f},"
                    f"casterShell {climb_moments_nm['caster_shell']:+.4f},"
                    f"passive {climb_moments_nm['passive']:+.4f},"
                    f"ground {climb_moments_nm['ground']:+.4f},"
                    f"net {climb_moments_nm['net']:+.4f})Nm "
                    f"eq5=({equation_5.vertical_force_n:+.3f},"
                    f"{equation_5.horizontal_force_n:+.3f}N;"
                    f"{equation_5.moment_nm:+.4f}Nm)"
                )
                for line in contact_diagnostics.formatted_lines(
                    time_step,
                    contact_readings,
                ):
                    print(line)
    finally:
        if teleop is not None:
            teleop.close()
        simulation_app.close()
