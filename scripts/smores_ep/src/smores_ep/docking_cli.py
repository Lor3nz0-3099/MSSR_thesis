from __future__ import annotations

import argparse
from pathlib import Path

from smores_ep.config.physics import (
    SMORES_EP_MAX_WHEEL_SPEED_RAD_S,
    SmoresActuatorConfig,
)
from smores_ep.config.simulation import DockingSimulationConfig


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_physics_asset() -> Path:
    return (
        _repository_root()
        / "assets/smores-ep/usd_physics/smores_ep_physics_v1.usd"
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SMORES-EP two-module rigid face docking"
    )
    parser.add_argument(
        "--physics-usd",
        type=Path,
        default=_default_physics_asset(),
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-ros2", action="store_true")
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
    parser.add_argument(
        "--docking-command-topic",
        default="/smores_ep/docking_command",
    )
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
    parser.add_argument("--active-module-id", default="active")
    parser.add_argument("--passive-module-id", default="passive")
    parser.add_argument(
        "--initial-active-face",
        choices=("LEFT", "RIGHT", "TOP", "BOTTOM"),
        default="TOP",
    )
    parser.add_argument(
        "--initial-passive-face",
        choices=("LEFT", "RIGHT", "TOP", "BOTTOM"),
        default="LEFT",
    )
    parser.add_argument("--initial-face-gap-mm", type=float, default=12.0)
    parser.add_argument("--spawn-height", type=float, default=0.0316)
    parser.add_argument("--initial-pitch-deg", type=float, default=0.0)
    parser.add_argument("--passive-yaw-deg", type=float, default=90.0)
    parser.add_argument(
        "--max-wheel-speed",
        type=float,
        default=SMORES_EP_MAX_WHEEL_SPEED_RAD_S,
    )
    parser.add_argument(
        "--actuator-effort-scale",
        type=float,
        default=3.0,
        help="active-module effort multiplier over nominal hardware",
    )
    parser.add_argument(
        "--no-active-ground-anchor",
        action="store_true",
        help="do not brace the active body after a successful attach",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.headless and args.steps == 0:
        args.steps = 2400

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})
    try:
        from smores_ep.scenarios.two_module_docking import (
            run_two_module_docking_scenario,
        )

        run_two_module_docking_scenario(
            DockingSimulationConfig(
                physics_usd=args.physics_usd,
                headless=args.headless,
                ros2_enabled=not args.no_ros2,
                steps=args.steps,
                physics_hz=args.physics_hz,
                render_hz=args.render_hz,
                log_interval=args.log_interval,
                cmd_vel_topic=args.cmd_vel_topic,
                pan_topic=args.pan_topic,
                pan_delta_topic=args.pan_delta_topic,
                tilt_topic=args.tilt_topic,
                docking_command_topic=args.docking_command_topic,
                primitive_goal_file=args.primitive_goal_file,
                primitive_cancel_file=args.primitive_cancel_file,
                primitive_status_file=args.primitive_status_file,
                active_module_id=args.active_module_id,
                passive_module_id=args.passive_module_id,
                initial_active_face=args.initial_active_face,
                initial_passive_face=args.initial_passive_face,
                initial_face_gap_m=1.0e-3 * args.initial_face_gap_mm,
                spawn_height_m=args.spawn_height,
                initial_pitch_deg=args.initial_pitch_deg,
                passive_yaw_deg=args.passive_yaw_deg,
                max_wheel_speed_rad_s=args.max_wheel_speed,
                active_actuators=SmoresActuatorConfig.payload_overdrive(
                    args.actuator_effort_scale,
                    wheel_max_speed_rad_s=args.max_wheel_speed,
                ),
                anchor_active_on_attach=not args.no_active_ground_anchor,
            ),
            simulation_app,
        )
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
