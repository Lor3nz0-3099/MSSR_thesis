"""Restore a C1 catalog sampler and reinitialize native physical episodes."""

import random

from drl.MARL.c1_longtrain_checkpoint import restore_rng
from drl.MARL.c1_longtrain_resume_artifacts import training_fingerprint
from drl.MARL.c1_longtrain_resume_core import restore_learner_boundary
from drl.MARL.c1_train_sampling import C1TrainSample
from drl.MARL.reward_calibration import MORPHOLOGIES, curriculum_identity


IDENTITY = (
    "schema", "seed", "domains_by_env", "cycle", "initial_seeds",
    "seed_pools", "catalog_difficulty_counts", "difficulty_weights",
)


def _tuple(value):
    return tuple(map(_tuple, value)) if isinstance(value, list) else value


def restore_sampler(sampler, saved):
    """Restore exact per-environment seed generations and RNG states."""

    fresh = sampler.manifest()

    if any(fresh.get(k) != saved.get(k) for k in IDENTITY):
        raise ValueError("TRAIN sampler configuration mismatch")

    rows = saved.get("environments")

    if not isinstance(rows, list) or len(rows) != len(sampler.domains_by_env):
        raise ValueError("TRAIN sampler slot mismatch")

    rngs, generations, currents, pending = [], [], [], []

    for i, row in enumerate(rows):
        domain = sampler.domains_by_env[i]
        generation = row.get("generation")
        current = row.get("current")

        if row.get("env_index") != i or row.get("domain") != domain:
            raise ValueError(f"Sampler domain/slot mismatch: {i}")

        if (
            type(generation) is not int
            or generation < 1
            or not isinstance(current, dict)
        ):
            raise ValueError(f"Invalid sampler generation: {i}")

        if (
            current.get("generation") != generation
            or current.get("domain") != domain
            or current.get("split") != "train"
        ):
            raise ValueError(f"Invalid TRAIN sample: {i}")

        seed = current.get("seed")

        if type(seed) is not int or seed not in sampler._pools[domain]:
            raise ValueError(f"Out-of-catalog seed: {i}")

        identity = curriculum_identity(domain, seed)

        if (
            current.get("difficulty") != identity.difficulty
            or current.get("morphology") != MORPHOLOGIES[domain]
        ):
            raise ValueError(f"Invalid sample geometry identity: {i}")

        if (
            row.get("initial_pending") is not False
            or not isinstance(row.get("pending_cycle"), list)
            or (not sampler.cycle and row["pending_cycle"])
        ):
            raise ValueError(f"Invalid pending sampler state: {i}")

        try:
            rng = random.Random()
            rng.setstate(_tuple(row["rng_state"]))
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Invalid sampler RNG: {i}") from exc

        rngs.append(rng)
        generations.append(generation)
        currents.append(C1TrainSample(**current))
        pending.append(list(row["pending_cycle"]))

    sampler._rngs[:] = rngs
    sampler._generation[:] = generations
    sampler._current[:] = currents
    sampler._pending[:] = pending

    if sampler.manifest() != saved:
        raise RuntimeError("Sampler RNG roundtrip failed")

    return currents



def force_wrapped_native_reset(wrapper):
    """Force a physical reset and refresh SKRL cached observations."""

    if getattr(wrapper, "_reset_once", None) is not False:
        raise RuntimeError(
            "Unexpected SKRL reset contract; unsafe C1 resume refused"
        )

    wrapper._reset_once = True
    result = wrapper.reset()

    if getattr(wrapper, "_reset_once", None) is not False:
        raise RuntimeError(
            "SKRL wrapper did not complete its physical reset"
        )

    return result


def resume_fresh_native_episodes(args, env, wrapper, learner):
    """Restore training state and restart every environment physically."""

    sampler = getattr(env, "train_sampler", None)

    if (
        sampler is None
        or sampler.cycle
        or sampler.initial_seeds is not None
    ):
        raise RuntimeError(
            "Resume requires the full TRAIN catalog sampler"
        )

    context = restore_learner_boundary(
        learner,
        args.resume_checkpoint,
        fingerprint=training_fingerprint(args),
        fresh_sampler_manifest=sampler.manifest(),
    )

    previous = restore_sampler(
        sampler,
        context["sampler_to_reconcile_with_fresh_physics"],
    )

    # Align the native episode generation before physical reset.
    for i, sample in enumerate(previous):
        env.generation[i] = sample.generation

    # Calls the native environment reset, including course resampling.
    force_wrapped_native_reset(wrapper)

    rows = sampler.manifest()["environments"]

    if len(rows) != env.num_envs:
        raise RuntimeError("World count differs after resume reset")

    for i, row in enumerate(rows):
        sample = row["current"]
        generation = previous[i].generation + 1

        if (
            row["generation"] != generation
            or env.generation[i] != generation
            or sample["generation"] != generation
            or env.seed_by_env[i] != sample["seed"]
            or sample["seed"] == previous[i].seed
        ):
            raise RuntimeError(
                f"World / TRAIN sampler mismatch in env {i}"
            )

    provider = getattr(env, "runtime_context_provider", None)

    if callable(getattr(provider, "reset", None)):
        provider.reset()

    # Restore global RNG only after rebuilding the physical scene.
    restore_rng(
        context["rng_to_restore_after_physics_reset"]
    )

    args._c1_resume_context = {
        "counters": context["counters"],
        "validation_state": context["validation_state"],
        "critic_gate_state": context["critic_gate_state"],
        "checkpoint_path": str(args.resume_checkpoint),
        "fresh_seeds": [
            row["current"]["seed"] for row in rows
        ],
    }

    print(
        "C1_RESUME_FRESH_EPISODES:",
        {
            "next_update": context["next_update"],
            "next_phase": context["next_phase"],
            "fresh_seeds": args._c1_resume_context["fresh_seeds"],
        },
        flush=True,
    )

    return args._c1_resume_context
