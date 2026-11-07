"""Restore C1 learner state at a fresh-episode boundary.

Physical episodes and sampler RNG must be reconciled separately.
"""

from drl.MARL.c1_longtrain_checkpoint import (
    load_bundle,
    read_manifest,
    validate_counters,
)
from drl.MARL.c1_longtrain_schedule import C1LongtrainSchedule


_SAMPLER_KEYS = (
    "schema",
    "seed",
    "domains_by_env",
    "cycle",
    "initial_seeds",
    "seed_pools",
    "catalog_difficulty_counts",
    "difficulty_weights",
)


def _check_empty_rollout(learner):
    if int(getattr(learner, "_rollout", 0)) != 0:
        raise RuntimeError(
            "Resume requires an empty PPO rollout counter"
        )

    for memory in getattr(learner, "memories", {}).values():
        if len(memory) != 0:
            raise RuntimeError(
                "Resume requires empty SKRL rollout memories"
            )

    structured = getattr(
        learner, "_structured_rollout_sidecar", None
    )

    if structured is not None and len(structured) != 0:
        raise RuntimeError(
            "Resume requires an empty structured sidecar"
        )

    altruistic = getattr(
        learner, "_altruistic_rollout_sidecar", None
    )

    if altruistic is not None and len(altruistic.flatten()) != 0:
        raise RuntimeError(
            "Resume requires an empty altruistic sidecar"
        )

    pending = getattr(learner, "_smdp_pending", None)

    if pending is not None and any(
        value is not None for value in pending
    ):
        raise RuntimeError(
            "Resume requires no in-flight SMDP options"
        )


def restore_learner_boundary(
    learner,
    checkpoint,
    *,
    fingerprint,
    fresh_sampler_manifest,
):
    """Restore weights, Adam and global counters.

    The caller must start fresh physical episodes before
    collecting new PPO transitions.

    RNG restoration must happen after scene reconstruction.
    """

    state = read_manifest(
        checkpoint,
        expected_fingerprint=fingerprint,
    )

    counters = validate_counters(state["counters"])

    saved_sampler = state["sampler_manifest"]

    if saved_sampler.get("schema") != "mssr.c1_train_sampler.v1":
        raise ValueError(
            "Resume requires a TRAIN catalog sampler"
        )

    for key in _SAMPLER_KEYS:
        if (
            fresh_sampler_manifest.get(key)
            != saved_sampler.get(key)
        ):
            raise ValueError(
                f"Sampler identity changed: {key}"
            )

    saved_envs = saved_sampler.get("environments")
    fresh_envs = fresh_sampler_manifest.get("environments")

    if (
        not isinstance(saved_envs, list)
        or not isinstance(fresh_envs, list)
        or len(saved_envs) != len(fresh_envs)
    ):
        raise ValueError(
            "Number of sampler slots changed"
        )

    for index, row in enumerate(saved_envs):
        if (
            row.get("env_index") != index
            or row.get("domain")
            != saved_sampler["domains_by_env"][index]
        ):
            raise ValueError(
                "Saved sampler environment slots inconsistent"
            )

    updates = counters["completed_updates"]
    version = counters["smdp_policy_version"]

    if (
        updates < 1
        or version > updates
        or (updates <= 4 and version != 0)
    ):
        raise ValueError(
            "Inconsistent staged PPO update/policy version"
        )

    # Controlla che nessun vecchio rollout venga riutilizzato.
    _check_empty_rollout(learner)

    # Ripristina i modelli e gli optimizer dal checkpoint.
    state = load_bundle(
        learner,
        checkpoint,
        expected_fingerprint=fingerprint,
    )

    # Reject any rollout data unexpectedly restored by SKRL.
    _check_empty_rollout(learner)

    # Recupera la progressione globale del training SMDP.
    learner._smdp_policy_version = version

    learner._smdp_transition_count = (
        counters["smdp_transition_count"]
    )

    learner._smdp_event_step_count = (
        counters["smdp_event_step_count"]
    )

    # Nuovi episodi: nessuna opzione fisica pendente.
    learner._smdp_event_rows_last_interaction = 0
    learner._smdp_pending = None
    learner._smdp_max_policy_lag = 0
    learner._rollout = 0

    # Restore the actual stage from the saved quality gate.
    from drl.MARL.c1_critic_quality_gate import C1CriticQualityGate

    gate = C1CriticQualityGate.from_state_dict(
        state.get("critic_gate_state")
    )

    if gate.last_completed != updates:
        raise ValueError(
            "Critic gate/update checkpoint mismatch"
        )

    next_step = gate.step_for_update(updates + 1)

    return {
        "counters": counters,
        "next_update": updates + 1,
        "next_phase": next_step.phase,
        "next_actor_lr": next_step.actor_lr,
        "next_critic_lr": next_step.critic_lr,
        "validation_state": state.get("validation_state"),
        "critic_gate_state": gate.state_dict(),
        "sampler_to_reconcile_with_fresh_physics": saved_sampler,
        "rng_to_restore_after_physics_reset": state["rng"],
        "physical_episode_reused": False,
    }
