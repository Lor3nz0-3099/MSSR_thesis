from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.tests.bc_v1 import test_live_observation as live_test


ROOT = Path(__file__).resolve().parents[3]


def make_bridge():
    from drl.MARL.bc_v4_runtime_observation import (
        BCV4RuntimeObservationBridge,
    )

    return BCV4RuntimeObservationBridge.from_repository(
        repo_root=ROOT,
        graph_builder=live_test.builder(),
        bridges=live_test.world_bridges(),
    )


def test_repository_bridge_prepares_actor_visible_rc8_batch():
    bridge = make_bridge()

    raw = live_test.snapshot(
        n=8,
        rc=True,
    )

    result = bridge.prepare_snapshot(
        raw,
        task_context=c.TaskContext(
            "navigation",
        ),
        observation=c.ObservationBundle(),
    )

    assert result.physical_state.stamp == 12.5
    assert len(result.physical_state.graph.modules) == 8

    assert result.actor_graph["graph_ptr"].tolist() == [
        0,
        8,
    ]

    assert (
        result.routing_snapshot[
            "actor_visibility_fingerprint"
        ]
        == bridge.visibility.fingerprint
    )

    assert (
        result.routing_snapshot[
            "actor_input_schema"
        ]
        == "mssr.bc_v1.actor_inputs.v1"
    )

    assert len(
        result.routing_snapshot["module_refs"]
    ) == 8

    assert result.decoder_batch[
        "inputs"
    ]["graph_ptr"].tolist() == [
        0,
        8,
    ]


def test_repository_bridge_runs_real_bc_v4_checkpoint_forward():
    from drl.MARL.bc_v4_actor import (
        BCV4SharedActorWrapper,
        BC_V4_REFERENCE_SHA256,
    )

    bridge = make_bridge()

    result = bridge.prepare_snapshot(
        live_test.snapshot(
            n=8,
            rc=True,
        ),
        task_context=c.TaskContext(
            "navigation",
        ),
        observation=c.ObservationBundle(),
    )

    actor = (
        BCV4SharedActorWrapper
        .from_repository_checkpoint(
            spec=bridge.spec,
            feature_schema=bridge.feature_schema,
            repo_root=ROOT,
            checkpoint_path=(
                ROOT
                / "artifacts/bc_v1/runs/"
                "20261022T232532_854802073Z_207309_"
                "bc_v4_actor_visible_r1m_scratch_final/"
                "best.pt"
            ),
            expected_sha256=BC_V4_REFERENCE_SHA256,
            map_location="cpu",
            device="cpu",
        )
    )

    actor.eval()

    possible_agents = [
        ref.module_id
        for ref in result.routing_snapshot[
            "module_refs"
        ]
    ]

    with torch.no_grad():
        output = actor.forward_all_owners(
            graph=result.actor_graph,
            routing=result.routing_snapshot,
            possible_agents=possible_agents,
        )

    assert output["prediction_template"] is not None

    prediction = output["prediction_template"]

    assert prediction["presence_logits"].numel() > 0

    assert len(output["records"]) == (
        prediction["presence_logits"].numel()
    )

    logits = torch.stack(
        [
            record.presence_logit.reshape(())
            for record in output["records"]
        ]
    )

    assert torch.isfinite(logits).all()

    assert output["resource_payloads"] is not None
    assert output["structural_payloads"] is not None


def test_repository_bridge_fails_closed_on_missing_physical_measurement():
    bridge = make_bridge()

    raw = copy.deepcopy(
        live_test.snapshot(
            n=8,
            rc=True,
        )
    )

    del raw["state"]["modules"][0]["pose"]

    with pytest.raises(ValueError):
        bridge.prepare_snapshot(
            raw,
            task_context=c.TaskContext(
                "navigation",
            ),
            observation=c.ObservationBundle(),
        )


def test_privileged_current_goal_changes_critic_view_without_actor_visibility_leak():
    bridge = make_bridge()
    raw = live_test.snapshot(n=8, rc=True)
    def prepare(goal_x):
        return bridge.prepare_snapshot(raw, task_context=c.TaskContext('navigation', (
            c.NumericFeature('world.goal.position_xyz_m', (goal_x, 4., 0.)),)),
            observation=c.ObservationBundle())
    first, second = prepare(5.), prepare(25.)
    assert first.privileged_graph is not None
    assert first.privileged_graph is not first.actor_graph
    def equal_tree(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:equal_tree(left[key], right[key])
        else:assert left == right
    equal_tree(first.actor_graph, second.actor_graph)
    assert first.actor_graph['environment']=={}
    key='c_static:goal.position_xyz_m:point3:3'
    assert not torch.equal(first.privileged_graph['environment'][key]['x'],
        second.privileged_graph['environment'][key]['x'])
    assert not torch.equal(first.privileged_graph['relation_x'],second.privileged_graph['relation_x'])
    for result in (first, second):
        assert result.physical_state.stamp == 12.5
        assert result.routing_snapshot['audit'].physical_stamp == 12.5
        assert result.privileged_graph['graph_ptr'].tolist() == [0,8]
        assert result.privileged_graph['module_batch'].tolist() == [0]*8
        assert [ref.module_id for ref in result.routing_snapshot['module_refs']] == [f'module-{i}' for i in range(8)]
        equal_tree(result.privileged_graph['module_x'], result.actor_graph['module_x'])
        equal_tree(result.privileged_graph['connector_to_module'], result.actor_graph['connector_to_module'])


def test_privileged_missing_normalizer_fails_with_exact_key():
    from dataclasses import replace
    from drl.IL.bc_v1.normalization import feature_key
    bridge=make_bridge()
    key=feature_key('environment','geometry',0,'m',geometry_kind='point3')
    assert key in dict(bridge.normalizers.scalers)
    bridge.normalizers=replace(bridge.normalizers,scalers=tuple((name,value) for name,value in bridge.normalizers.scalers if name!=key))
    with pytest.raises(ValueError,match='Missing privileged normalizer coverage') as error:
        bridge.prepare_snapshot(live_test.snapshot(n=8,rc=True),task_context=c.TaskContext('navigation',(
            c.NumericFeature('world.goal.position_xyz_m',(5.,4.,0.)),)))
    assert key in str(error.value)
