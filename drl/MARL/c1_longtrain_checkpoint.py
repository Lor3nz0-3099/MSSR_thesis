"""Atomic checkpoint and resume metadata for C1 staged training."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


SCHEMA = "mssr.c1.staged_resume.v1"

COUNTER_NAMES = (
    "completed_updates",
    "vector_steps",
    "smdp_transition_count",
    "smdp_event_step_count",
    "smdp_policy_version",
)


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def manifest_path(path):
    return Path(str(path) + ".resume.json")


def validate_counters(counters):
    result = {}

    for name in COUNTER_NAMES:
        value = counters.get(name)

        if type(value) is not int or value < 0:
            raise ValueError(
                f"Invalid checkpoint counter: {name}"
            )

        result[name] = value

    return result


def capture_rng():
    numpy_state = np.random.get_state()

    cuda_states = []

    if torch.cuda.is_available():
        cuda_states = [
            state.tolist()
            for state in torch.cuda.get_rng_state_all()
        ]

    return {
        "python": random.getstate(),
        "numpy": [
            numpy_state[0],
            numpy_state[1].tolist(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ],
        "torch_cpu": torch.get_rng_state().tolist(),
        "torch_cuda": cuda_states,
    }


def _restore_tuples(value):
    if isinstance(value, list):
        return tuple(
            _restore_tuples(x) for x in value
        )

    return value


def restore_rng(state):
    random.setstate(
        _restore_tuples(state["python"])
    )

    n = state["numpy"]

    np.random.set_state((
        n[0],
        np.asarray(n[1], dtype=np.uint32),
        n[2],
        n[3],
        n[4],
    ))

    torch.set_rng_state(
        torch.tensor(
            state["torch_cpu"],
            dtype=torch.uint8,
        )
    )

    cuda_states = state["torch_cuda"]

    if cuda_states:
        if (
            not torch.cuda.is_available()
            or len(cuda_states) != torch.cuda.device_count()
        ):
            raise RuntimeError(
                "CUDA configuration differs from checkpoint"
            )

        torch.cuda.set_rng_state_all([
            torch.tensor(s, dtype=torch.uint8)
            for s in cuda_states
        ])


def save_bundle(
    learner,
    path,
    *,
    counters,
    fingerprint,
    sampler_manifest,
    validation_state=None,
    critic_gate_state=None,
):
    """Save models/optimizers and publish metadata atomically.

    Call only at completed PPO-update boundaries.
    """

    path = Path(path).resolve()
    metadata = manifest_path(path)

    counters = validate_counters(counters)

    if not isinstance(fingerprint, dict) or not fingerprint:
        raise ValueError(
            "Configuration fingerprint is required"
        )

    if (
        not isinstance(sampler_manifest, dict)
        or sampler_manifest.get("schema")
        != "mssr.c1_train_sampler.v1"
    ):
        raise ValueError(
            "Valid TRAIN sampler manifest required"
        )

    if path.exists() or metadata.exists():
        raise FileExistsError(
            f"Checkpoint already exists: {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.stem + ".incomplete.pt"
    )

    temporary_metadata = Path(
        str(metadata) + ".incomplete"
    )

    if temporary.exists() or temporary_metadata.exists():
        raise FileExistsError(
            "Incomplete checkpoint already exists"
        )

    try:
        learner.save(str(temporary))

        if (
            not temporary.is_file()
            or temporary.stat().st_size == 0
        ):
            raise RuntimeError(
                "SKRL checkpoint was not created"
            )

        state = {
            "schema": SCHEMA,
            "checkpoint_name": path.name,
            "checkpoint_sha256": sha256_file(
                temporary
            ),
            "checkpoint_bytes": (
                temporary.stat().st_size
            ),
            "counters": counters,
            "fingerprint": fingerprint,
            "sampler_manifest": sampler_manifest,
            "validation_state": validation_state,
            "critic_gate_state": critic_gate_state,
            "rng": capture_rng(),
            "physics_resume": (
                "new_episode_not_bitwise_continuation"
            ),
        }

        with temporary_metadata.open(
            "x", encoding="utf-8"
        ) as stream:
            json.dump(
                state,
                stream,
                sort_keys=True,
                allow_nan=False,
            )

            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        # Publish the checkpoint first.
        os.replace(temporary, path)

        # Publish metadata last.
        # A complete pair identifies a resumable artifact.
        os.replace(
            temporary_metadata,
            metadata,
        )

        return metadata

    except BaseException:
        temporary.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
        raise


def read_manifest(
    path,
    *,
    expected_fingerprint,
):
    """Validate checkpoint before touching model state."""

    path = Path(path).resolve()

    state = json.loads(
        manifest_path(path).read_text(
            encoding="utf-8"
        )
    )

    if state.get("schema") != SCHEMA:
        raise ValueError(
            "Unsupported checkpoint schema"
        )

    if state.get("checkpoint_name") != path.name:
        raise ValueError(
            "Checkpoint filename mismatch"
        )

    if state.get("fingerprint") != expected_fingerprint:
        raise ValueError(
            "Configuration changed: unsafe resume"
        )

    validate_counters(
        state.get("counters", {})
    )

    sampler = state.get("sampler_manifest", {})

    if (
        sampler.get("schema")
        != "mssr.c1_train_sampler.v1"
    ):
        raise ValueError(
            "TRAIN sampler state missing"
        )

    if (
        state.get("checkpoint_bytes")
        != path.stat().st_size
        or state.get("checkpoint_sha256")
        != sha256_file(path)
    ):
        raise ValueError(
            "Checkpoint incomplete or corrupted"
        )

    if not isinstance(state.get("rng"), dict):
        raise ValueError(
            "RNG state missing"
        )

    return state


def load_bundle(
    learner,
    path,
    *,
    expected_fingerprint,
    restore_random_state=False,
):
    """Restore SKRL weights/optimizers after integrity checks.

    The runner must separately restore counters, sampler,
    validation controller and fresh physics episodes.
    """

    state = read_manifest(
        path,
        expected_fingerprint=expected_fingerprint,
    )

    learner.load(
        str(Path(path).resolve())
    )

    if restore_random_state:
        restore_rng(state["rng"])

    return state
