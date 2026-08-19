from __future__ import annotations

import argparse
from pathlib import Path

from smores_ep.config.physics import (
    SMORES_EP_MAX_WHEEL_SPEED_RAD_S,
    SmoresActuatorConfig,
)
from smores_ep.config.simulation import MultiModuleLiftSimulationConfig


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
            "SMORES-EP active module docking to a pre-connected "
            "TOP-to-BOTTOM chain"
        )
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
    parser.add_argument("--chain-module-prefix", default="chain")
    parser.add_argument(
        "--chain-count",
        type=int,
        default=5,
        help="number of modules already connected TOP-to-BOTTOM",
    )
    parser.add_argument(
        "--active-to-chain-gap-mm",
        type=float,
        default=0.0,
        help="initial gap between active:TOP and chain_01:BOTTOM",
    )
    parser.add_argument("--spawn-height", type=float, default=0.0316)
    parser.add_argument("--initial-pitch-deg", type=float, default=0.0)
    parser.add_argument(
        "--max-wheel-speed",
        type=float,
        default=SMORES_EP_MAX_WHEEL_SPEED_RAD_S,
    )
    parser.add_argument(
        "--actuator-effort-scale",
        type=float,
        default=6.0,
        help=(
            "effort multiplier over nominal hardware for the intentionally "
            "exaggerated five-module lift"
        ),
    )
    parser.add_argument(
        "--no-active-ground-anchor",
        action="store_true",
        help="do not brace active roll/pitch after it docks to the chain",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.headless and args.steps == 0:
        args.steps = 7200

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})
    try:
        from smores_ep.scenarios.multi_module_lift import (
            run_multi_module_lift_scenario,
        )

        run_multi_module_lift_scenario(
            MultiModuleLiftSimulationConfig(
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
                chain_module_prefix=args.chain_module_prefix,
                chain_module_count=args.chain_count,
                active_to_chain_gap_m=(
                    1.0e-3 * args.active_to_chain_gap_mm
                ),
                spawn_height_m=args.spawn_height,
                initial_pitch_deg=args.initial_pitch_deg,
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
