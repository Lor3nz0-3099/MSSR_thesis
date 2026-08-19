from __future__ import annotations

import argparse
from pathlib import Path

from smores_ep.config.physics import SMORES_EP_MAX_WHEEL_SPEED_RAD_S
from smores_ep.config.simulation import DynamicSimulationConfig


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_physics_asset() -> Path:
    return (
        _repository_root()
        / "assets/smores-ep/usd_physics/smores_ep_physics_v1.usd"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SMORES-EP fully dynamic ROS 2 teleoperation in Isaac Sim"
    )
    parser.add_argument(
        "--physics-usd",
        type=Path,
        default=_default_physics_asset(),
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-ros2", action="store_true")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a deterministic physical demo when ROS 2 is disabled",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="0 keeps a GUI run alive until its window is closed",
    )
    parser.add_argument("--physics-hz", type=int, default=240)
    parser.add_argument("--render-hz", type=int, default=60)
    parser.add_argument("--log-interval", type=int, default=240)
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--pan-topic", default="/smores_ep/pan_angle")
    parser.add_argument(
        "--pan-delta-topic",
        default="/smores_ep/pan_delta",
    )
    parser.add_argument("--tilt-topic", default="/smores_ep/tilt_angle")
    parser.add_argument("--spawn-height", type=float, default=0.0316)
    parser.add_argument("--initial-pitch-deg", type=float, default=-0.25)
    parser.add_argument(
        "--max-wheel-speed",
        type=float,
        default=SMORES_EP_MAX_WHEEL_SPEED_RAD_S,
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.headless and args.steps == 0:
        args.steps = 1200

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})
    try:
        from smores_ep.scenarios.dynamic_teleop import run_dynamic_scenario

        run_dynamic_scenario(
            DynamicSimulationConfig(
                physics_usd=args.physics_usd,
                headless=args.headless,
                ros2_enabled=not args.no_ros2,
                demo_enabled=args.demo,
                steps=args.steps,
                physics_hz=args.physics_hz,
                render_hz=args.render_hz,
                log_interval=args.log_interval,
                cmd_vel_topic=args.cmd_vel_topic,
                pan_topic=args.pan_topic,
                pan_delta_topic=args.pan_delta_topic,
                tilt_topic=args.tilt_topic,
                spawn_height_m=args.spawn_height,
                initial_pitch_deg=args.initial_pitch_deg,
                max_wheel_speed_rad_s=args.max_wheel_speed,
            ),
            simulation_app,
        )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
