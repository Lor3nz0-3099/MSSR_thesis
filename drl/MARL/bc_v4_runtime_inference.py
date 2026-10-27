"""Deterministic BC-v4 runtime inference contract.

This module is deliberately separate from MAPPO/APPO sampling.
The threshold configured here is used only when evaluating or
executing the frozen BC-v4 policy deterministically.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from drl.IL.bc_v1.action_set_decoder import (
    decode_action_sets,
)


_SCHEMA = "mssr.bc_v1.runtime_inference.v1"


def load_bc_v4_runtime_inference_config(
    repo_root: str | Path,
):
    root = Path(repo_root).resolve()

    path = (
        root
        / "configs/bc_v1/runtime_inference_v1.json"
    )

    document = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if document.get("schema_version") != _SCHEMA:
        raise ValueError(
            "Unsupported BC-v4 runtime inference schema"
        )

    threshold = document.get(
        "deterministic_presence_threshold_logit"
    )

    if (
        type(threshold) not in (int, float)
        or not math.isfinite(threshold)
    ):
        raise ValueError(
            "Invalid deterministic BC-v4 threshold"
        )

    selection = document.get("selection")

    if not isinstance(selection, dict):
        raise ValueError(
            "Missing BC-v4 validation selection metadata"
        )

    if selection.get("split") != "validation":
        raise ValueError(
            "BC-v4 runtime threshold must be selected on VAL"
        )

    if (
        selection.get("metric")
        != "set_exact_match"
    ):
        raise ValueError(
            "Unexpected BC-v4 runtime selection metric"
        )

    if document.get("TEST_payload_access") != 0:
        raise ValueError(
            "BC-v4 runtime threshold selection accessed TEST"
        )

    return document


def decode_bc_v4_deterministic(
    *,
    prediction,
    decoder_batch,
    normalizers,
    repo_root: str | Path,
):
    """Decode frozen BC-v4 using the VAL-selected runtime threshold.

    This function must not be used for MAPPO/APPO stochastic action
    sampling or PPO action reevaluation.
    """

    config = (
        load_bc_v4_runtime_inference_config(
            repo_root
        )
    )

    threshold = float(
        config[
            "deterministic_presence_threshold_logit"
        ]
    )

    return decode_action_sets(
        prediction,
        decoder_batch,
        normalizers,
        threshold=threshold,
    )
