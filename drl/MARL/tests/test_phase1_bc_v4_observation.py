from __future__ import annotations

import copy
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from drl.IL.bc_v1 import (
    bc_v1_contract as c,
)
from drl.IL.tests.bc_v1 import (
    test_live_observation as live_test,
)


ROOT = Path(
    __file__
).resolve().parents[3]


def _api():
    spec = importlib.util.find_spec(
        "drl.MARL.phase1_bc_v4_observation"
    )

    assert (
        spec is not None
    ), "TASK4C2_RED_SYNC_BCV4_SOURCE_MISSING"

    module = importlib.import_module(
        "drl.MARL.phase1_bc_v4_observation"
    )

    return (
        module.SynchronizedBCV4ObservationSource,
    )


def _bridge():
    from drl.MARL.bc_v4_runtime_observation import (
        BCV4RuntimeObservationBridge,
    )

    return (
        BCV4RuntimeObservationBridge
        .from_repository(
            repo_root=ROOT,
            graph_builder=live_test.builder(),
            bridges=live_test.world_bridges(),
        )
    )


def _write_raw(
    runtime_dir: Path,
    raw: dict,
) -> None:
    runtime_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        runtime_dir
        / "state_graph.json"
    )

    temporary = (
        runtime_dir
        / "state_graph.json.tmp"
    )

    temporary.write_text(
        json.dumps(raw),
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def _snapshot(
    *,
    stamp: float,
):
    raw = copy.deepcopy(
        live_test.snapshot(
            n=8,
            rc=True,
        )
    )

    raw["stamp"] = float(
        stamp
    )

    raw["state"]["stamp"] = float(
        stamp
    )

    raw["state"]["timestamp"] = float(
        stamp
    )

    raw["graph"]["stamp"] = float(
        stamp
    )

    return raw


def test_task4c2_uses_exact_synchronizer_cached_frame_not_latest_file(
    tmp_path,
):
    (
        SynchronizedBCV4ObservationSource,
    ) = _api()

    from drl.MARL.isaac_control_step import (
        FileBackedIsaacControlStep,
    )

    runtime = (
        tmp_path
        / "env0"
    )

    raw_10 = _snapshot(
        stamp=10.0,
    )

    _write_raw(
        runtime,
        raw_10,
    )

    sync = FileBackedIsaacControlStep(
        runtime_dirs=(runtime,),
        expected_module_count=8,
    )

    # Arms the exact s_t.
    sync.state_reader(0)

    source = (
        SynchronizedBCV4ObservationSource(
            frame_synchronizer=sync,
            observation_bridge=_bridge(),
            task_context_provider=(
                lambda env_index:
                    c.TaskContext(
                        "navigation"
                    )
            ),
        )
    )

    first = source.prepare_env(
        0
    )

    assert (
        first.physical_state.stamp
        == pytest.approx(10.0)
    )

    assert (
        first.actor_graph[
            "graph_ptr"
        ].tolist()
        == [0, 8]
    )

    assert len(
        first.routing_snapshot[
            "module_refs"
        ]
    ) == 8

    # Producer advances independently AFTER the MARL frame was armed.
    raw_11 = _snapshot(
        stamp=11.0,
    )

    _write_raw(
        runtime,
        raw_11,
    )

    second = source.prepare_env(
        0
    )

    # Must still be the exact cached s_t=10.0.
    assert (
        second.physical_state.stamp
        == pytest.approx(10.0)
    )

    assert (
        source.prepared_stamp(0)
        == pytest.approx(10.0)
    )


def test_task4c2_preprocesses_each_physical_stamp_once(
    tmp_path,
):
    (
        SynchronizedBCV4ObservationSource,
    ) = _api()

    from drl.MARL.isaac_control_step import (
        FileBackedIsaacControlStep,
    )

    runtime = (
        tmp_path
        / "env0"
    )

    _write_raw(
        runtime,
        _snapshot(
            stamp=20.0,
        ),
    )

    sync = FileBackedIsaacControlStep(
        runtime_dirs=(runtime,),
        expected_module_count=8,
    )

    sync.state_reader(0)

    bridge = _bridge()

    calls = []
    original = (
        bridge.prepare_snapshot
    )

    def observed(
        raw_snapshot,
        **kwargs,
    ):
        calls.append(
            float(
                raw_snapshot[
                    "stamp"
                ]
            )
        )

        return original(
            raw_snapshot,
            **kwargs,
        )

    bridge.prepare_snapshot = observed

    source = (
        SynchronizedBCV4ObservationSource(
            frame_synchronizer=sync,
            observation_bridge=bridge,
            task_context_provider=(
                lambda env_index:
                    c.TaskContext(
                        "navigation"
                    )
            ),
        )
    )

    first = source.prepare_env(
        0
    )

    second = source.prepare_env(
        0
    )

    assert first is second

    assert calls == [
        20.0,
    ]

    source.reset()

    # Source reset drops preprocessing cache only.
    # Synchronizer still owns the physical frame.
    third = source.prepare_env(
        0
    )

    assert (
        third.physical_state.stamp
        == pytest.approx(20.0)
    )

    assert calls == [
        20.0,
        20.0,
    ]


def test_task4c2_real_checkpoint_consumes_exact_live_actor_graph(
    tmp_path,
):
    (
        SynchronizedBCV4ObservationSource,
    ) = _api()

    from drl.MARL.bc_v4_actor import (
        BCV4SharedActorWrapper,
        BC_V4_REFERENCE_SHA256,
    )

    from drl.MARL.isaac_control_step import (
        FileBackedIsaacControlStep,
    )

    runtime = (
        tmp_path
        / "env0"
    )

    _write_raw(
        runtime,
        _snapshot(
            stamp=30.0,
        ),
    )

    sync = FileBackedIsaacControlStep(
        runtime_dirs=(runtime,),
        expected_module_count=8,
    )

    sync.state_reader(0)

    bridge = _bridge()

    source = (
        SynchronizedBCV4ObservationSource(
            frame_synchronizer=sync,
            observation_bridge=bridge,
            task_context_provider=(
                lambda env_index:
                    c.TaskContext(
                        "navigation"
                    )
            ),
        )
    )

    prepared = source.prepare_env(
        0
    )

    actor = (
        BCV4SharedActorWrapper
        .from_repository_checkpoint(
            spec=bridge.spec,
            feature_schema=(
                bridge.feature_schema
            ),
            repo_root=ROOT,
            checkpoint_path=(
                ROOT
                / "artifacts/bc_v1/runs/"
                "20261022T232532_854802073Z_207309_"
                "bc_v4_actor_visible_r1m_scratch_final/"
                "best.pt"
            ),
            expected_sha256=(
                BC_V4_REFERENCE_SHA256
            ),
            map_location="cpu",
            device="cpu",
        )
    )

    actor.eval()

    possible_agents = [
        ref.module_id
        for ref
        in prepared.routing_snapshot[
            "module_refs"
        ]
    ]

    with torch.no_grad():
        output = actor.forward_all_owners(
            graph=prepared.actor_graph,
            routing=(
                prepared.routing_snapshot
            ),
            possible_agents=(
                possible_agents
            ),
        )

    prediction = output[
        "prediction_template"
    ]

    assert prediction is not None

    assert (
        prediction[
            "presence_logits"
        ].numel()
        > 0
    )

    logits = torch.stack(
        [
            record
            .presence_logit
            .reshape(())
            for record
            in output["records"]
        ]
    )

    assert torch.isfinite(
        logits
    ).all()

    assert (
        prepared.physical_state.stamp
        == pytest.approx(30.0)
    )

    # Critical contract: this is stochastic-MAPPO actor preparation.
    # No deterministic 4.5 validation threshold is used here.
