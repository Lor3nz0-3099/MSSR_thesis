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


@pytest.mark.parametrize('connected', [False, True])
def test_inactive_candidate_omission_keeps_real_actor_and_critic_physical_features(monkeypatch, connected):
    from drl.MARL import bc_v4_runtime_observation as runtime
    bridge = make_bridge()
    raw = live_test.snapshot(n=8, rc=connected)
    context = c.TaskContext('navigation', (c.NumericFeature('world.goal.position_xyz_m', (5., 4., 0.)),))
    full = bridge.prepare_snapshot(raw, task_context=context)
    explicit_full = bridge.prepare_snapshot(raw, task_context=context, include_action_candidates=True)
    assert full.routing_snapshot['structural_bindings']
    def equal_tree(left, right):
        if torch.is_tensor(left):
            assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left: equal_tree(left[key], right[key])
        else:
            assert left == right
    equal_tree(full.actor_graph, explicit_full.actor_graph)
    equal_tree(full.privileged_graph, explicit_full.privileged_graph)
    def forbidden(*args, **kwargs):
        raise AssertionError('Expert row must not enumerate action candidates')
    monkeypatch.setattr(runtime, 'enumerate_candidates', forbidden)
    inactive = bridge.prepare_snapshot(raw, task_context=context, include_action_candidates=False)
    assert inactive.physical_state == full.physical_state
    assert inactive.routing_snapshot['resource_candidates'] == ()
    assert inactive.routing_snapshot['structural_bindings'] == ()
    for graph_name in ('actor_graph', 'privileged_graph'):
        original, reduced = getattr(full, graph_name), getattr(inactive, graph_name)
        assert reduced['module_x'].shape[0] == 8
        for key in original:
            if key not in ('resource_candidates', 'structural'):
                equal_tree(original[key], reduced[key])
        assert len(reduced['resource_candidates']['resource']) == 0
        assert len(reduced['structural']['initiator']) == 0


def test_real_bc_v4_loose_structural_samples_materialize_all_argument_families():
    """CPU synthetic physical snapshot; actual BC checkpoint and MARL sampler."""
    from types import SimpleNamespace
    from drl.MARL.c1_runner import CHECKPOINT,build_models
    from drl.MARL.policy import sample_marl_action_bundle
    from drl.MARL.smores_runtime_binding import materialize_marl_action_set_prediction
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261029)
            bridge=make_bridge()
            prepared=bridge.prepare_snapshot(live_test.snapshot(n=8),task_context=c.TaskContext('navigation'))
            actor,_=build_models(SimpleNamespace(repository_root=ROOT,checkpoint=ROOT/CHECKPOINT,device='cpu'),bridge)
            with torch.no_grad():
                output=actor.forward_all_owners(graph=prepared.actor_graph,routing=prepared.routing_snapshot,
                    possible_agents=[r.module_id for r in prepared.routing_snapshot['module_refs']])
                # Force one categorical branch for UNIT coverage only. These
                # synthetic positions are not a BC behavior/physical smoke.
                from dataclasses import replace
                category=output['prediction_template']['parameters']['structural']['arguments']['category']
                argument=int(category['argument'][0])
                binding=int(prepared.actor_graph['structural']['argument_binding'][argument])
                chosen=len(prepared.routing_snapshot['resource_candidates'])+binding
                records=tuple(replace(record,presence_logit=torch.tensor(100. if i==chosen else -100.))
                    for i,record in enumerate(output['records']))
                action=sample_marl_action_bundle(records=records,resource_payloads=output['resource_payloads'],
                    structural_payloads=output['structural_payloads'],hard_mask=torch.zeros(len(output['records']),dtype=torch.bool),
                    resource_log_std=output['resource_log_std'],structural_log_std=output['structural_log_std'])
                resource_count=len(prepared.routing_snapshot['resource_candidates'])
                assert action.presence[resource_count:].sum()>0
                materialized=materialize_marl_action_set_prediction(action=action,
                    prediction_template=output['prediction_template'],routing=prepared.routing_snapshot,batch=prepared.decoder_batch)
                assert torch.isfinite(materialized['presence_logits']).all()
    finally:torch.set_num_threads(previous)
