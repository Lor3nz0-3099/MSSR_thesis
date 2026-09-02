from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from smores_ep.config.physics import (
    SMORES_EP_MAX_WHEEL_SPEED_RAD_S,
    SmoresActuatorConfig,
)
from smores_ep.config.simulation import SelfAssemblySimulationConfig
from smores_ep.isaac.obstacle_course import (
    CoplanarGapSpec,
    UniformStairSpec,
    sample_coplanar_gap_spec,
    sample_uniform_stair_spec,
)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_physics_asset() -> Path:
    return (
        _repository_root()
        / "assets/smores-ep/usd_physics/smores_ep_physics_v1.usd"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic SMORES-EP parallel self-assembly"
        )
    )
    parser.add_argument(
        "--physics-usd",
        type=Path,
        default=_default_physics_asset(),
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--performance",
        action="store_true",
        help=(
            "Use paced simulation, 240 Hz physics, 30 FPS updates and 5 Hz "
            "state publication while keeping the full CAD"
        ),
    )
    parser.add_argument(
        "--simulation-speed-factor",
        type=float,
        default=1.0,
        help=(
            "Paced simulated seconds per wall second when --performance is "
            "enabled; useful for ROS-coordinated headless runs"
        ),
    )
    parser.add_argument(
        "--simple-visuals",
        action="store_true",
        help="Render collision proxies instead of the full CAD",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="0 keeps a GUI run alive until its window is closed",
    )
    parser.add_argument("--physics-hz", type=int, default=None)
    parser.add_argument("--render-hz", type=int, default=None)
    parser.add_argument("--state-publish-hz", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=240)
    parser.add_argument(
        "--primitive-goal-file",
        type=Path,
        default=Path("configs/smores_primitive_goal.json"),
    )
    parser.add_argument(
        "--primitive-cancel-file",
        type=Path,
        default=Path("configs/smores_primitive_cancel.json"),
    )
    parser.add_argument(
        "--primitive-status-file",
        type=Path,
        default=Path("logs/bridge/smores_primitive_status.json"),
    )
    parser.add_argument(
        "--action-file",
        type=Path,
        default=Path("configs/actions.json"),
        help="Composed morphology commands forwarded from /mssr/actions",
    )
    parser.add_argument(
        "--action-command-timeout",
        type=float,
        default=0.5,
        help="Dead-man timeout for cluster locomotion commands",
    )
    parser.add_argument("--left-module-id", default="smores_01")
    parser.add_argument("--center-module-id", default="smores_02")
    parser.add_argument("--right-module-id", default="smores_03")
    parser.add_argument(
        "--module-count",
        type=int,
        default=3,
        help=(
            "Number of modules. Counts other than 3 generate IDs from "
            "--module-prefix; the legacy three-module IDs remain supported"
        ),
    )
    parser.add_argument("--module-prefix", default="smores_")
    parser.add_argument("--spawn-height", type=float, default=0.0316)
    parser.add_argument("--initial-pitch-deg", type=float, default=0.0)
    parser.add_argument("--spawn-half-width", type=float, default=0.18)
    parser.add_argument("--outer-y", type=float, default=-0.10)
    parser.add_argument("--center-y", type=float, default=0.14)
    parser.add_argument("--outer-yaw-deg", type=float, default=25.0)
    parser.add_argument("--spawn-radius", type=float, default=0.34)
    course_group = parser.add_mutually_exclusive_group()
    course_group.add_argument(
        "--obstacle-course",
        action="store_true",
        help=(
            "Replace the infinite floor with a manual gap/stairs/button/exit "
            "task-achievement course"
        ),
    )
    parser.add_argument(
        "--stair-seed",
        type=int,
        default=None,
        help=(
            "Sample a reproducible uniform staircase from the conservative "
            "Snake8 validation envelope"
        ),
    )
    parser.add_argument("--stair-rise-m", type=float, default=None)
    parser.add_argument("--stair-depth-m", type=float, default=None)
    parser.add_argument("--stair-count", type=int, default=None)
    parser.add_argument("--stair-first-riser-x-m", type=float, default=None)
    course_group.add_argument(
        "--stair-test-course",
        action="store_true",
        help=(
            "Replace the infinite floor with an isolated three-step Snake8 "
            "behavior test course"
        ),
    )
    course_group.add_argument(
        "--button-test-course",
        action="store_true",
        help=(
            "Replace the infinite floor with an isolated flat "
            "MobileManipulator8 button test course"
        ),
    )
    course_group.add_argument(
        "--gap-test-course",
        action="store_true",
        help=(
            "Replace the infinite floor with an isolated equal-bank "
            "Snake8 gap test course"
        ),
    )
    course_group.add_argument(
        "--rc-car-planar-test-course",
        action="store_true",
        help=(
            "Replace the infinite floor with the seeded flat RC-Car8 "
            "Nav2 route stage"
        ),
    )
    parser.add_argument(
        "--rc-car-seed",
        type=int,
        default=None,
        help="Seed selecting the RC-Car8 S-curve/slalom/loop route",
    )
    parser.add_argument(
        "--gap-seed",
        type=int,
        default=None,
        help=(
            "Sample a reproducible coplanar gap from the conservative "
            "Snake8 validation envelope"
        ),
    )
    parser.add_argument("--gap-width-m", type=float, default=None)
    parser.add_argument("--gap-near-edge-x-m", type=float, default=None)
    parser.add_argument(
        "--disable-staging-collision-avoidance",
        action="store_true",
        help="Disable assembly-only planar obstacle avoidance",
    )
    parser.add_argument(
        "--staging-center-clearance",
        type=float,
        default=0.110,
        help="Minimum centre-to-centre clearance during free-space staging",
    )
    parser.add_argument(
        "--staging-waypoint-margin",
        type=float,
        default=0.015,
        help="Extra clearance around generated avoidance waypoints",
    )
    parser.add_argument(
        "--max-wheel-speed",
        type=float,
        default=SMORES_EP_MAX_WHEEL_SPEED_RAD_S,
    )
    parser.add_argument(
        "--wheel-friction-scale",
        type=float,
        default=1.50,
        help=(
            "Scale wheel-only static and dynamic friction at runtime; "
            "body and passive-skid friction are unchanged"
        ),
    )
    parser.add_argument(
        "--actuator-effort-scale",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--tilt-effort-scale",
        type=float,
        default=8.0,
        help=(
            "TILT-only effort scale for cantilevered chains; defaults to "
            "the Snake8 single-hinge payload profile (8.0)"
        ),
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.headless and args.steps == 0:
        args.steps = 7200

    stair_spec = (
        sample_uniform_stair_spec(args.stair_seed)
        if args.stair_seed is not None
        else UniformStairSpec()
    )
    stair_overrides = {
        field_name: value
        for field_name, value in (
            ("rise_m", args.stair_rise_m),
            ("tread_depth_m", args.stair_depth_m),
            ("step_count", args.stair_count),
            ("first_riser_x_m", args.stair_first_riser_x_m),
        )
        if value is not None
    }
    if stair_overrides:
        stair_spec = replace(stair_spec, **stair_overrides)

    gap_spec = (
        sample_coplanar_gap_spec(args.gap_seed)
        if args.gap_seed is not None
        else CoplanarGapSpec()
    )
    gap_overrides = {
        field_name: value
        for field_name, value in (
            ("width_m", args.gap_width_m),
            ("near_edge_x_m", args.gap_near_edge_x_m),
        )
        if value is not None
    }
    if gap_overrides:
        gap_spec = replace(gap_spec, **gap_overrides)

    physics_hz = args.physics_hz
    if physics_hz is None:
        physics_hz = 240
    render_hz = args.render_hz
    if render_hz is None:
        render_hz = 30 if args.performance else 20
    state_publish_hz = args.state_publish_hz
    if state_publish_hz is None:
        state_publish_hz = 5 if args.performance else 10

    if args.module_count == 3:
        module_ids = (
            args.left_module_id,
            args.center_module_id,
            args.right_module_id,
        )
    else:
        module_ids = tuple(
            f"{args.module_prefix}{index:02d}"
            for index in range(1, args.module_count + 1)
        )

    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": args.headless,
            "width": 960,
            "height": 540,
        }
    )
    try:
        from smores_ep.scenarios.parallel_self_assembly import (
            run_parallel_self_assembly_scenario,
        )

        run_parallel_self_assembly_scenario(
            SelfAssemblySimulationConfig(
                physics_usd=args.physics_usd,
                headless=args.headless,
                steps=args.steps,
                physics_hz=physics_hz,
                render_hz=render_hz,
                state_publish_hz=state_publish_hz,
                log_interval=args.log_interval,
                simple_visuals=args.simple_visuals,
                realtime_pacing=args.performance,
                simulation_speed_factor=args.simulation_speed_factor,
                include_contact_candidates=not args.performance,
                primitive_goal_file=args.primitive_goal_file,
                primitive_cancel_file=args.primitive_cancel_file,
                primitive_status_file=args.primitive_status_file,
                action_file=args.action_file,
                action_command_timeout_s=args.action_command_timeout,
                module_ids=module_ids,
                spawn_height_m=args.spawn_height,
                initial_pitch_deg=args.initial_pitch_deg,
                spawn_half_width_m=args.spawn_half_width,
                outer_y_m=args.outer_y,
                center_y_m=args.center_y,
                outer_yaw_deg=args.outer_yaw_deg,
                spawn_radius_m=args.spawn_radius,
                manual_obstacle_course=args.obstacle_course,
                stair_test_course=args.stair_test_course,
                button_test_course=args.button_test_course,
                gap_test_course=args.gap_test_course,
                rc_car_planar_test_course=args.rc_car_planar_test_course,
                rc_car_seed=args.rc_car_seed,
                stair_rise_m=stair_spec.rise_m,
                stair_depth_m=stair_spec.tread_depth_m,
                stair_count=stair_spec.step_count,
                stair_first_riser_x_m=stair_spec.first_riser_x_m,
                stair_seed=stair_spec.seed,
                gap_width_m=gap_spec.width_m,
                gap_near_edge_x_m=gap_spec.near_edge_x_m,
                gap_seed=gap_spec.seed,
                staging_collision_avoidance=(
                    not args.disable_staging_collision_avoidance
                ),
                staging_center_clearance_m=args.staging_center_clearance,
                staging_waypoint_margin_m=args.staging_waypoint_margin,
                wheel_friction_scale=args.wheel_friction_scale,
                max_wheel_speed_rad_s=args.max_wheel_speed,
                actuators=SmoresActuatorConfig.payload_overdrive(
                    args.actuator_effort_scale,
                    wheel_max_speed_rad_s=args.max_wheel_speed,
                    tilt_effort_scale=args.tilt_effort_scale,
                ),
            ),
            simulation_app,
        )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
