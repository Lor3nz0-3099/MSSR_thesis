from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def test_repository_runtime_inference_threshold_is_val_selected():
    from drl.MARL.bc_v4_runtime_inference import (
        load_bc_v4_runtime_inference_config,
    )

    config = load_bc_v4_runtime_inference_config(
        ROOT
    )

    assert (
        config["deterministic_presence_threshold_logit"]
        == 4.5
    )

    assert config["selection"]["split"] == "validation"
    assert config["selection"]["metric"] == "set_exact_match"
    assert config["TEST_payload_access"] == 0


def test_deterministic_bc_decoder_uses_frozen_runtime_threshold(
    monkeypatch,
):
    import drl.MARL.bc_v4_runtime_inference as runtime

    seen = {}

    def fake_decode(
        prediction,
        batch,
        normalizers,
        *,
        threshold,
    ):
        seen["threshold"] = threshold
        return ["decoded"]

    monkeypatch.setattr(
        runtime,
        "decode_action_sets",
        fake_decode,
    )

    result = runtime.decode_bc_v4_deterministic(
        prediction="prediction",
        decoder_batch="batch",
        normalizers="normalizers",
        repo_root=ROOT,
    )

    assert result == ["decoded"]
    assert seen["threshold"] == 4.5


def test_runtime_threshold_is_not_a_mappo_sampling_parameter():
    from drl.MARL.bc_v4_runtime_inference import (
        load_bc_v4_runtime_inference_config,
    )

    config = load_bc_v4_runtime_inference_config(
        ROOT
    )

    assert (
        "mappo" not in
        " ".join(config.keys()).lower()
    )
