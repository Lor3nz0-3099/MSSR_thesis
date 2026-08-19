from __future__ import annotations

import argparse
from pathlib import Path

from freebot_docking.config.geometry import (
    RunningGearGeometry,
    ShellGeometry,
    WheelRadialComplianceConfig,
)
from freebot_docking.config.mass import ModuleMassConfig
from freebot_docking.control.wheel_drive import WheelDriveConfig
from freebot_docking.isaac.materials import IsaacMaterialConfig
from freebot_docking.isaac.simulator import (
    IsaacSimulationConfig,
    run_isaac_simulation,
)
from freebot_docking.isaac.stage_builder import IsaacStageConfig


def _default_usd_path() -> Path:
    repository_root = Path(__file__).resolve().parents[4]
    return (
        repository_root
        / "assets/freebot/usd_physics/"
        "freebot_cad_full_nearer_wheels_rigid.usd"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Two-module FreeBOT magnetic docking in Isaac Sim"
    )
    parser.add_argument("--usd", type=Path, default=_default_usd_path())
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-ros2", action="store_true")
    parser.add_argument("--dynamic-passive", action="store_true")
    parser.add_argument("--steps", type=int, default=12_000)
    parser.add_argument("--physics-hz", type=int, default=240)
    parser.add_argument("--log-interval", type=int, default=240)
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--cmd-timeout", type=float, default=0.5)
    parser.add_argument(
        "--debug-draw",
        action="store_true",
        help="Draw display-only magnetic and PhysX contact force vectors",
    )
    parser.add_argument(
        "--debug-force-scale",
        type=float,
        default=0.003,
        help="Displayed force-vector length in metres per newton",
    )
    parser.add_argument(
        "--initial-shell-gap-mm",
        type=float,
        default=20.0,
        help="Initial shell gap inside the measured 0--30 mm capture range",
    )
    parser.add_argument("--internal-preload", type=float, default=9.5)
    parser.add_argument(
        "--external-target",
        choices=("active-shell", "active-carrier"),
        default="active-carrier",
        help=(
            "Body receiving the published external resultant; "
            "active-shell is the reversible shell-patch experiment"
        ),
    )
    parser.add_argument(
        "--mass-scale",
        type=float,
        default=1.0,
        help="Diagnostic uniform scale for every module body mass",
    )
    parser.add_argument(
        "--tire-precompression-mm",
        type=float,
        default=0.9,
        help="Unloaded tire-envelope growth used as elastic compression [mm]",
    )
    parser.add_argument(
        "--tire-contact-stiffness",
        type=float,
        default=8_000.0,
        help="Force-based compliant tire stiffness [N/m]",
    )
    parser.add_argument(
        "--tire-contact-damping",
        type=float,
        default=40.0,
        help="Force-based compliant tire damping [N*s/m]",
    )
    parser.add_argument(
        "--wheel-radial-compliance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable limited spring-damped radial wheel travel; use "
            "--no-wheel-radial-compliance to restore the previous rigid joints"
        ),
    )
    parser.add_argument(
        "--wheel-radial-inward-mm",
        type=float,
        default=0.6,
        help="Allowed wheel travel toward the carrier [mm]",
    )
    parser.add_argument(
        "--wheel-radial-outward-mm",
        type=float,
        default=2.1,
        help="Allowed wheel travel toward the shell [mm]",
    )
    parser.add_argument(
        "--wheel-radial-rest-mm",
        type=float,
        default=1.7,
        help="Unloaded spring position toward the shell [mm]",
    )
    parser.add_argument(
        "--wheel-radial-stiffness",
        type=float,
        default=3_500.0,
        help="Wheel-support radial stiffness [N/m]",
    )
    parser.add_argument(
        "--wheel-radial-damping",
        type=float,
        default=12.0,
        help="Wheel-support radial damping [N*s/m]",
    )
    parser.add_argument(
        "--wheel-radial-max-force",
        type=float,
        default=15.0,
        help="Maximum internal spring force per wheel [N]",
    )
    parser.add_argument(
        "--caster-clearance-mm",
        type=float,
        default=0.0,
        help="Nominal caster-to-inner-shell clearance [mm] (0=contact)",
    )
    parser.add_argument(
        "--caster-precompression-mm",
        type=float,
        default=0.1,
        help="Effective compliant caster/shell envelope growth [mm]",
    )
    parser.add_argument(
        "--caster-contact-stiffness",
        type=float,
        default=2_000.0,
        help="Force-based compliant caster stiffness [N/m]",
    )
    parser.add_argument(
        "--caster-contact-damping",
        type=float,
        default=15.0,
        help="Force-based compliant caster damping [N*s/m]",
    )
    parser.add_argument(
        "--ground-static-friction",
        type=float,
        default=1.25,
        help=(
            "Static shell-ground coefficient; 1.25 is just below the "
            "FreeBOT Eq. (7) threshold after the carrier starts lifting"
        ),
    )
    parser.add_argument(
        "--ground-dynamic-friction",
        type=float,
        default=1.00,
        help="Dynamic shell-ground friction coefficient",
    )
    parser.add_argument("--linear-scale", type=float, default=720.0)
    parser.add_argument("--yaw-scale", type=float, default=360.0)
    parser.add_argument("--motor-no-load-speed", type=float, default=360.0)
    parser.add_argument(
        "--motor-stall-torque",
        type=float,
        default=7.0 * 0.0980665,
    )
    parser.add_argument(
        "--motor-armature",
        type=float,
        default=0.003,
        help="Output-side reflected motor/gear inertia [kg*m^2]",
    )
    parser.add_argument(
        "--motor-brake-torque",
        type=float,
        default=0.12,
        help="Static zero-command brake torque per wheel [N*m]",
    )
    parser.add_argument("--climb-heading-gain", type=float, default=2.0)
    parser.add_argument("--climb-heading-max-turn", type=float, default=90.0)
    parser.add_argument("--climb-heading-gap-mm", type=float, default=5.0)
    parser.add_argument(
        "--climb-heading",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Opt-in differential steering correction near shell contact",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.steps < 1 or args.physics_hz < 1:
        raise ValueError("Steps and physics frequency must be positive")
    if not 0.0 <= args.initial_shell_gap_mm <= 30.0:
        raise ValueError(
            "Initial shell gap must lie in the calibrated 0--30 mm range"
        )
    if args.tire_precompression_mm > 0.0 and args.tire_contact_stiffness <= 0.0:
        raise ValueError(
            "A precompressed tire requires positive compliant-contact stiffness"
        )
    if (
        args.caster_precompression_mm > 0.0
        and args.caster_contact_stiffness <= 0.0
    ):
        raise ValueError(
            "A precompressed caster requires positive contact stiffness"
        )
    if args.debug_force_scale <= 0.0:
        raise ValueError("Debug force scale must be positive")

    geometry = ShellGeometry()
    active_center = (0.0199584, 0.060, geometry.outer_radius_m)
    passive_center = (
        active_center[0]
        + 2.0 * geometry.outer_radius_m
        + 1.0e-3 * args.initial_shell_gap_mm,
        active_center[1],
        active_center[2],
    )
    stage = IsaacStageConfig(
        usd_path=args.usd,
        active_shell_center_world=active_center,
        passive_shell_center_world=passive_center,
        passive_fixed=not args.dynamic_passive,
        running_gear=RunningGearGeometry(
            tire_precompression_m=1.0e-3 * args.tire_precompression_mm,
            caster_nominal_clearance_m=1.0e-3 * args.caster_clearance_mm,
            caster_precompression_m=(
                1.0e-3 * args.caster_precompression_mm
            ),
        ),
        wheel_radial_compliance=WheelRadialComplianceConfig(
            enabled=args.wheel_radial_compliance,
            inward_travel_m=1.0e-3 * args.wheel_radial_inward_mm,
            outward_travel_m=1.0e-3 * args.wheel_radial_outward_mm,
            rest_position_m=1.0e-3 * args.wheel_radial_rest_mm,
            stiffness_n_per_m=args.wheel_radial_stiffness,
            damping_n_s_per_m=args.wheel_radial_damping,
            max_force_n=args.wheel_radial_max_force,
        ),
        materials=IsaacMaterialConfig(
            wheel_contact_stiffness_n_per_m=args.tire_contact_stiffness,
            wheel_contact_damping_n_s_per_m=args.tire_contact_damping,
            caster_contact_stiffness_n_per_m=(
                args.caster_contact_stiffness
            ),
            caster_contact_damping_n_s_per_m=args.caster_contact_damping,
            ground_static_friction=args.ground_static_friction,
            ground_dynamic_friction=args.ground_dynamic_friction,
        ),
        masses=ModuleMassConfig().scaled(args.mass_scale),
    )
    simulation = IsaacSimulationConfig(
        stage=stage,
        physics_hz=args.physics_hz,
        steps=args.steps,
        log_interval=args.log_interval,
        headless=args.headless,
        ros2_teleop=not args.no_ros2,
        cmd_vel_topic=args.cmd_vel_topic,
        cmd_timeout_s=args.cmd_timeout,
        debug_draw=args.debug_draw,
        debug_force_scale_m_per_n=args.debug_force_scale,
        internal_preload_force_n=args.internal_preload,
        external_force_target=args.external_target,
        wheel_drive=WheelDriveConfig(
            linear_scale_deg_s=args.linear_scale,
            yaw_scale_deg_s=args.yaw_scale,
            no_load_speed_deg_s=args.motor_no_load_speed,
            stall_torque_nm=args.motor_stall_torque,
            armature_kg_m2=args.motor_armature,
            zero_command_brake_torque_nm=args.motor_brake_torque,
            climb_heading_enabled=args.climb_heading,
            climb_heading_gain_s_inv=args.climb_heading_gain,
            climb_heading_max_turn_deg_s=args.climb_heading_max_turn,
            climb_heading_capture_gap_m=1.0e-3 * args.climb_heading_gap_mm,
        ),
    )
    run_isaac_simulation(simulation)


if __name__ == "__main__":
    main()
