from __future__ import annotations

import argparse
from pathlib import Path

from smores_ep.config.simulation import KinematicSimulationConfig


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_visual() -> Path:
    return (
        _repository_root()
        / "assets/smores-ep/usd_visual/smores_ep_usd_visual_v1.usd"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SMORES-EP ROS 2 kinematic teleoperation in Isaac Sim"
    )
    parser.add_argument("--visual-usd", type=Path, default=_default_visual())
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-ros2", action="store_true")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a deterministic motion demo when ROS 2 is disabled",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="0 keeps a GUI run alive until its window is closed",
    )
    parser.add_argument("--update-hz", type=int, default=120)
    parser.add_argument("--log-interval", type=int, default=120)
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--pan-topic", default="/smores_ep/pan_angle")
    parser.add_argument(
        "--pan-delta-topic",
        default="/smores_ep/pan_delta",
    )
    parser.add_argument("--tilt-topic", default="/smores_ep/tilt_angle")
    parser.add_argument("--max-pan-speed", type=float, default=2.0)
    parser.add_argument("--max-tilt-speed", type=float, default=1.25)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.headless and args.steps == 0:
        args.steps = 600

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})
    try:
        from smores_ep.scenarios.kinematic_teleop import (
            run_kinematic_scenario,
        )

        run_kinematic_scenario(
            KinematicSimulationConfig(
                visual_usd=args.visual_usd,
                headless=args.headless,
                ros2_enabled=not args.no_ros2,
                demo_enabled=args.demo,
                steps=args.steps,
                update_hz=args.update_hz,
                log_interval=args.log_interval,
                cmd_vel_topic=args.cmd_vel_topic,
                pan_topic=args.pan_topic,
                pan_delta_topic=args.pan_delta_topic,
                tilt_topic=args.tilt_topic,
                max_pan_speed_rad_s=args.max_pan_speed,
                max_tilt_speed_rad_s=args.max_tilt_speed,
            ),
            simulation_app,
        )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
