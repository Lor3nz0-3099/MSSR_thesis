"""C1 RC-Car GUI test: frozen BC-v4 actor, no training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time

from pathlib import Path


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--reward-config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7977,
    )

    parser.add_argument(
        "--global-seed",
        type=int,
        default=3103,
    )

    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=180.0,
    )

    parser.add_argument(
        "--inspect-seconds",
        type=float,
        default=60.0,
    )

    parser.add_argument(
        "--wall-limit-seconds",
        type=float,
        default=10800.0,
    )

    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args()

    if (
        args.episode_seconds != 180.0
        or args.seed < 0
        or args.wall_limit_seconds <= 0
    ):
        parser.error(
            "Diagnostic requires 180 simulated seconds "
            "and a positive wall limit"
        )

    if (
        not args.reward_config.is_file()
        or not args.checkpoint.is_file()
    ):
        parser.error(
            "Reward config or BC-v4 checkpoint missing"
        )

    if args.output.exists():
        parser.error(
            "Output directory already exists"
        )

    args.output.mkdir(parents=True)

    (args.output / "config.json").write_text(
        json.dumps(
            {
                "mode": "frozen_actor_no_optimizer",
                "domain": "navigation",
                "morphology": "rc_car8",
                "num_envs": 1,
                "seed": args.seed,
                "global_seed": args.global_seed,
                "episode_seconds": args.episode_seconds,
                "checkpoint_sha256": hashlib.sha256(
                    args.checkpoint.read_bytes()
                ).hexdigest(),
            },
            indent=2,
        ) + "\n"
    )

    app = AppLauncher(args).app

    env = None
    runtime = None

    steps = 0
    outcome = "not_started"

    timeline = []
    last_progress = None

    peak_valid = 0
    peak_component = 1

    started = time.monotonic()

    try:
        import torch

        from skrl.utils import set_seed
        from isaaclab_rl.skrl import SkrlVecEnvWrapper

        root = Path(__file__).resolve().parents[2]

        for source in (
            root / "mssr_ws/src/mssr_expert",
            root / "scripts/smores_ep/src",
        ):
            sys.path.insert(0, str(source))

        from drl.MARL.c1_runner import (
            build_observation_bridge,
            build_models,
        )

        from drl.MARL.isaaclab_c1_env import (
            MSSRC1IsaacLabEnv,
            make_c1_isaaclab_env_cfg,
        )

        from drl.MARL.mappo import MSSRMAPPO

        set_seed(args.global_seed)

        owners = tuple(
            f"smores_{i:02d}"
            for i in range(1, 9)
        )

        cfg = make_c1_isaaclab_env_cfg(
            possible_agents=owners,
            num_envs=1,
            device=args.device,
        )

        cfg.domains = ("navigation",)
        cfg.seeds = (args.seed,)

        cfg.course_split = "train"
        cfg.resample_train = False

        cfg.initialization_mode = "loose_modules"

        cfg.reward_config_path = str(
            args.reward_config.resolve()
        )

        cfg.episode_length_s = args.episode_seconds

        cfg.c1_evidence_directory = str(
            args.output / "c1_evidence"
        )

        cfg.c1_readonly_state_cache = True

        env = MSSRC1IsaacLabEnv(cfg)

        wrapper = SkrlVecEnvWrapper(
            env,
            ml_framework="torch",
        )

        obs, _ = wrapper.reset()

        bridge = build_observation_bridge(root)

        model_args = argparse.Namespace(
            checkpoint=args.checkpoint.resolve(),
            repository_root=root,
            device=args.device,
        )

        actor, critic = build_models(
            model_args,
            bridge,
        )

        actor.eval().requires_grad_(False)
        critic.eval().requires_grad_(False)

        env.initialize_policy_bridge(
            observation_bridge=bridge,
            structural_scheduler="event",
            bulk_host_reads=True,
        )

        # Riutilizziamo l'inferenza MAPPO.
        # Nessun learner, optimizer o buffer PPO.
        class FrozenActorInference:

            forward_shared_actor = staticmethod(
                MSSRMAPPO.forward_shared_actor
            )

            forward_shared_actor_batch = staticmethod(
                MSSRMAPPO.forward_shared_actor_batch
            )

            build_online_rollout_decision = (
                MSSRMAPPO.build_online_rollout_decision
            )

            act = MSSRMAPPO.act

            def __init__(self):
                self.shared_actor = actor
                self.shared_critic = critic

                self.possible_agents = owners
                self.device = args.device

                self.skip_expert_policy_inference = True

                self.deterministic_policy_actions = False

                self._rollout_context_provider = (
                    env.runtime_context_provider
                )

                self._current_online_decisions = None

        policy = FrozenActorInference()

        try:
            env.sim.set_camera_view(
                [2.7, -2.8, 2.3],
                [0.0, 0.0, 0.15],
            )
        except Exception as exc:
            print(
                "Camera view not set:",
                type(exc).__name__,
                str(exc)[:160],
                flush=True,
            )

        print(
            "GUI_RC_CAR_START:",
            json.dumps(
                {
                    "seed": args.seed,
                    "episode_seconds": args.episode_seconds,
                    "num_envs": 1,
                    "morphology": "rc_car8",
                    "policy": "BC-v4 frozen",
                    "training": False,
                }
            ),
            flush=True,
        )

        goal_steps = int(
            round(args.episode_seconds / env.step_dt)
        )

        initial_generation = int(
            env.generation[0]
        )

        while (
            steps < goal_steps
            and app.is_running()
        ):

            if (
                time.monotonic() - started
                >= args.wall_limit_seconds
            ):
                outcome = "wall_limit_observation_end"
                break

            with torch.no_grad():

                actions, _ = policy.act(
                    obs,
                    wrapper.state(),
                    timestep=steps,
                    timesteps=goal_steps,
                )

                obs, _, terminated, truncated, _ = (
                    wrapper.step(actions)
                )

            steps += 1

            runtime = getattr(
                env,
                "c1_runtime",
                None,
            )

            if (
                runtime is None
                or 0 not in runtime.evaluators
            ):
                raise RuntimeError(
                    "C1 evaluator did not initialize"
                )

            evaluator = runtime.evaluators[0][1]
            result = evaluator.last_result

            if result is None:
                raise RuntimeError(
                    "Missing C1 assessment"
                )

            valid = int(
                result.assessment.valid_connections
            )

            components = list(
                result.assessment.component_sizes
            )

            largest = max(
                components,
                default=1,
            )

            peak_valid = max(
                peak_valid,
                valid,
            )

            peak_component = max(
                peak_component,
                largest,
            )

            simulated_seconds = round(
                steps * env.step_dt,
                3,
            )

            progress = (
                valid,
                tuple(components),
                result.phase,
                evaluator.counts["docking_count"],
                evaluator.counts["undocking_count"],
            )

            report_interval = max(
                1,
                int(round(10 / env.step_dt)),
            )

            if (
                progress != last_progress
                or steps == 1
                or steps % report_interval == 0
                or result.terminated
                or result.truncated
            ):

                row = {
                    "time_s": simulated_seconds,
                    "steps": steps,
                    "phase": result.phase,
                    "valid_connections": valid,
                    "component_sizes": components,
                    "peak_valid_connections": peak_valid,
                    "peak_component_size": peak_component,
                    "dockings": evaluator.counts[
                        "docking_count"
                    ],
                    "undockings": evaluator.counts[
                        "undocking_count"
                    ],
                    "assembly_valid_ever": (
                        evaluator.assembly_valid_timestamp
                        is not None
                    ),
                    "success": result.success,
                    "terminated": result.terminated,
                    "truncated": result.truncated,
                    "reason": result.reason,
                }

                timeline.append(row)

                print(
                    "GUI_RC_CAR_PROGRESS:",
                    json.dumps(row),
                    flush=True,
                )

                last_progress = progress

            if (
                result.terminated
                or result.truncated
            ):

                outcome = (
                    "success"
                    if result.success
                    else "failure"
                    if result.terminated
                    else "time_limit"
                )

                break

            if (
                int(env.generation[0])
                != initial_generation
            ):
                raise RuntimeError(
                    "Unexpected episode reset"
                )

        else:
            outcome = (
                "observation_end"
                if steps >= goal_steps
                else "viewer_closed"
            )

    except KeyboardInterrupt:
        outcome = "user_interrupted"

        print(
            "GUI_RC_CAR_INTERRUPTED",
            flush=True,
        )

    finally:

        if runtime is not None:
            runtime.flush_open_episodes(
                reason="observation_end"
            )

        final = {
            "outcome": outcome,
            "steps": steps,
            "simulated_seconds": round(
                steps / 30.0,
                3,
            ),
            "peak_valid_connections": peak_valid,
            "peak_component_size": peak_component,
            "time_wall_s": round(
                time.monotonic() - started,
                3,
            ),
            "episodes": (
                runtime.episode_records
                if runtime is not None
                else []
            ),
        }

        (
            args.output / "summary.json"
        ).write_text(
            json.dumps(
                final,
                indent=2,
                default=str,
            ) + "\n"
        )

        with (
            args.output / "timeline.jsonl"
        ).open("w") as handle:

            for row in timeline:
                handle.write(
                    json.dumps(row) + "\n"
                )

        print(
            "GUI_RC_CAR_RESULT:",
            json.dumps(
                {
                    key: final[key]
                    for key in (
                        "outcome",
                        "steps",
                        "simulated_seconds",
                        "peak_valid_connections",
                        "peak_component_size",
                    )
                }
            ),
            flush=True,
        )

        # Manteniamo la GUI aperta per ispezionare la scena.
        if (
            env is not None
            and args.inspect_seconds > 0
            and outcome != "viewer_closed"
        ):

            try:
                env.sim.pause()
            except Exception:
                pass

            print(
                "VIEWER_HOLD:",
                f"GUI aperta per {args.inspect_seconds:g} s.",
                "Ctrl+C per chiudere.",
                flush=True,
            )

            try:
                end = (
                    time.monotonic()
                    + args.inspect_seconds
                )

                while (
                    time.monotonic() < end
                    and app.is_running()
                ):
                    app.update()

            except KeyboardInterrupt:
                pass

        if env is not None:
            env.close()

        app.close()


if __name__ == "__main__":
    main()
