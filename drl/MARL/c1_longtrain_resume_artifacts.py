"""Completed-update checkpoint integration for C1."""

from pathlib import Path

from drl.MARL.c1_longtrain_checkpoint import (
    save_bundle,
    read_manifest,
    sha256_file,
)


def supports_staged_resume_artifact(sampler_manifest):
    """Only the legacy catalog sampler can be persisted by save_update_artifact.

    Phase-D snapshot episodes use a different, non-resumable sampler schema.
    They still receive the ordinary timed learner checkpoints and final model.
    Unknown schemas fail closed instead of silently suppressing recovery.
    """
    if not isinstance(sampler_manifest, dict):
        raise ValueError("C1 staged training requires a sampler manifest")
    schema = sampler_manifest.get("schema")
    if schema == "mssr.c1_train_sampler.v1":
        return True
    if schema == "mssr.c1.snapshot_sampler.v1":
        return False
    raise ValueError(f"Unsupported C1 staged sampler schema: {schema!r}")


def training_fingerprint(args):
    """Identity of the training configuration and implementation."""

    repo = Path(args.repository_root).resolve()

    sources = (
        "drl/MARL/mappo.py",
        "drl/MARL/bc_v4_actor.py",
        "drl/MARL/h2_batching.py",
        "drl/MARL/c1_longtrain_schedule.py",
        "drl/MARL/c1_episode.py",
        "drl/MARL/c1_native_runtime.py",
        "drl/MARL/c1_guided_feedback.py",
        "mssr_ws/src/mssr_expert/config/smores_rc_car8.json",
        "mssr_ws/src/mssr_expert/config/smores_snake8.json",
        "mssr_ws/src/mssr_expert/config/smores_mobile_manipulator8.json",
        "drl/MARL/c1_critic_quality.py",
        "drl/MARL/c1_critic_quality_gate.py",
        "drl/MARL/c1_longtrain_checkpoint.py",
        "drl/MARL/c1_longtrain_resume_artifacts.py",
        "drl/MARL/c1_longtrain_resume_core.py",
        "drl/MARL/c1_longtrain_resume_integration.py",
        "drl/MARL/c1_train_sampling.py",
        "drl/MARL/isaaclab_c1_tasks.py",
        "drl/MARL/c1_early_stopping.py",
        "drl/MARL/c1_validation_callback.py",
        "drl/MARL/isaaclab_c1_training.py",
        "drl/MARL/isaaclab_c1_runner.py",
    )

    return {
        "schema": "mssr.c1.longtrain_fingerprint.v1",
        "staged_training": bool(args.c1_staged_training),
        "integration_probe": bool(getattr(args, "c1_integration_probe", False)),
        "num_envs": int(args.num_envs),
        "rollout_steps": int(args.rollout_steps),
        "learning_epochs": int(args.learning_epochs),
        "mini_batches": int(args.mini_batches),
        "curriculum": str(args.curriculum),
        "initialization_mode": str(args.initialization_mode),
        "structural_scheduler": str(args.structural_scheduler),
        "domains": list(args.domains),
        "global_seed": int(args.global_seed),
        "device": str(args.device),
        "episode_seconds": float(args.episode_seconds),
        "c1_probe_config_sha256": sha256_file(
            args.c1_probe_config
            or (repo / "configs/marl/c1_assembly_probe.json")
        ),
        "alpha": float(args.alpha),
        "gamma": float(args.gamma),
        "altruism_radius_m": float(args.altruism_radius_m),
        "bc_v4_checkpoint_sha256": sha256_file(args.checkpoint),
        "reward_config_sha256": sha256_file(args.reward_config),
        "source_sha256": {
            source: sha256_file(repo / source)
            for source in sources
        },
    }


def save_update_artifact(
    args,
    output,
    learner,
    sampler,
    *,
    counts,
    validation_state=None,
    critic_gate_state=None,
):
    """Save only after a completed PPO update."""

    if not (
        getattr(args, "train", False)
        and getattr(args, "c1_staged_training", False)
    ):
        raise ValueError(
            "Explicit staged C1 training is required"
        )

    if critic_gate_state is None:
        raise RuntimeError(
            'Staged C1 checkpoint requires critic gate state'
        )
    if sampler is None or not callable(
        getattr(sampler, "manifest", None)
    ):
        raise RuntimeError(
            "TRAIN sampler manifest is required"
        )

    manifest = sampler.manifest()
    environments = manifest.get("environments")

    if (
        manifest.get("schema")
        != "mssr.c1_train_sampler.v1"
        or not isinstance(environments, list)
        or len(environments) != args.num_envs
    ):
        raise ValueError(
            "Incomplete TRAIN sampler state"
        )

    for row in environments:
        for key in (
            "rng_state",
            "generation",
            "pending_cycle",
        ):
            if key not in row:
                raise ValueError(
                    f"Missing sampler field: {key}"
                )

    update = counts.get("completed_updates")
    version = counts.get("smdp_policy_version")

    if type(update) is not int or update < 1:
        raise ValueError(
            "Checkpoint requires a completed PPO update"
        )

    if (
        type(version) is not int
        or not 0 <= version <= update
    ):
        raise ValueError(
            "Invalid SMDP policy version"
        )

    output = Path(output)

    path = output / (
        f"c1_resume_u{update:06d}_v{version:06d}.pt"
    )

    fingerprint = training_fingerprint(args)

    save_bundle(
        learner,
        path,
        counters=counts,
        fingerprint=fingerprint,
        sampler_manifest=manifest,
        validation_state=validation_state,
        critic_gate_state=critic_gate_state,
    )

    read_manifest(
        path,
        expected_fingerprint=fingerprint,
    )

    return path
