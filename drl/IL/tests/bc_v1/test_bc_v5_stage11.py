"""Stage 11: actual H2 tensor repack, all positives, and live fail-closed gate."""
from dataclasses import replace

import pytest
import torch

from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.bc_v1.bc_v5_stage11 import _make_row, _TopologyLRU, row_count, _restore_historical_align_choices
from drl.IL.bc_v1.bc_v5_live import gate_v5_bundle
from drl.IL.bc_v1.bc_v5_topology_candidates import enumerate_candidates_v5
from drl.IL.bc_v1.action_set import ActionBundle, ActionInstance
from drl.IL.bc_v1.action_set_targets import positive_indices, collate_action_set_samples
from drl.IL.bc_v1.candidates import enumerate_candidates
from drl.IL.bc_v1.live_observation import LivePhysicalState
from drl.IL.tests.bc_v1.g2_fixtures import fixture, tensorize, with_structural
from drl.IL.tests.bc_v1.test_candidates import attach, graph, catalog, schema, structural


def _repack(sample, kw):
    original = tensorize(sample, kw)
    fresh, exception_used = _make_row(original, sample, split='train', registry={}, exceptional_keys=set(),
        spec=kw['spec'], schema=kw['feature_schema'],
        lru=_TopologyLRU(kw['spec'], kw['feature_schema']))
    return original, fresh, exception_used


def test_repack_copies_all_h1_features_and_reduces_strict_structural_negatives():
    s, kw = fixture(3,2)
    g = attach(s.inputs.policy.graph, 0, 2)
    from drl.IL.bc_v1.dataset import CachedInputs
    # Keep original resource supervision, only switch graph/current catalog.
    s = replace(s, inputs=replace(s.inputs, policy=replace(s.inputs.policy, graph=g),
        candidates=enumerate_candidates(g, kw['spec'], kw['feature_schema'])))
    old, new, exceptional = _repack(s, kw)
    assert exceptional is False
    assert len(new['routing']['structural_bindings']) < len(old['routing']['structural_bindings'])
    assert (len(new['routing']['resource_candidates']) == len(old['routing']['resource_candidates']))
    assert len(positive_indices(new['routing'])) == len(positive_indices(old['routing']))
    for name in old['inputs']:
        if name != 'structural':
            assert torch.equal(new['inputs'][name], old['inputs'][name]) if isinstance(new['inputs'][name],torch.Tensor) else new['inputs'][name] == old['inputs'][name]
    batch = collate_action_set_samples([new])
    assert int(batch['supervision']['action_set']['presence'].sum()) == 1
    cnt = row_count(new['routing'])
    assert cnt['resource:pace']['positive'] == 1
    assert cnt['structural:align_faces']['negative'] < row_count(old['routing'])['structural:align_faces']['negative']


def test_offline_historical_exception_restores_full_align_alternatives_not_single_label():
    s,kw = fixture(3,2)
    g = attach(s.inputs.policy.graph, 0, 2)
    candidate = enumerate_candidates(g, kw['spec'], kw['feature_schema'])
    positives = [(item.action,b) for item in candidate.structural_candidates
                 if item.action.action_id == 'align_faces'
                 for b in item.bindings if b.initiator.module_id == 'unit-0']
    assert positives
    action, binding = positives[0]
    target = c.StructuralTarget(binding.initiator, action.action_id,
        tuple(p.module for p in binding.connectors), binding.connectors,
        (c.ParameterValue('turn', 'quarter'),))
    s = replace(s, inputs=replace(s.inputs, policy=replace(s.inputs.policy, graph=g), candidates=candidate),
                supervision=replace(s.supervision, structural_targets=(target,)))
    old = tensorize(s,kw)
    # Strict catalog loses this positive on an attached initiator.
    with pytest.raises(ValueError,match='Structural target has no unique binding'):
        _make_row(old,s,split='train',registry={},exceptional_keys=set(),
             spec=kw['spec'],schema=kw['feature_schema'],lru=_TopologyLRU(kw['spec'],kw['feature_schema']))
    row, restored = _make_row(old,s,split='train',registry={},
        exceptional_keys={('train',s.audit.episode_id,s.audit.transition_index,binding.initiator.module_id)},
        spec=kw['spec'],schema=kw['feature_schema'],lru=_TopologyLRU(kw['spec'],kw['feature_schema']))
    assert restored
    assert len(positive_indices(row['routing'])) == 2
    assert len(row['routing']['structural_bindings']) > 1
    # Entire historic align domain must be present for exceptional timestep.
    old_align = {(a.action_id,b.initiator,b.connectors) for a,b in old['routing']['structural_bindings']
                 if a.action_id == 'align_faces'}
    new_align = {(a.action_id,b.initiator,b.connectors) for a,b in row['routing']['structural_bindings']
                 if a.action_id == 'align_faces'}
    assert old_align <= new_align


def _claims(action):
    if action.resource is not None:
        return (action.resource,)
    return action.connectors if action.connectors else (c.ConnectorRef(action.initiator,'socket-0'),)


def test_live_rechecks_all_actions_and_blocks_joint_before_dispatch():
    spec, feature_schema = catalog(2),schema()
    physical = attach(graph(3,2),0,2)
    detached = graph(3,2)
    before = LivePhysicalState(10., detached)
    stale = LivePhysicalState(11., physical)
    # This reference is only a mock physical snapshot; no offline exception.
    align = next((a,b) for cand in structural(enumerate_candidates_v5(detached,spec,feature_schema),'align_faces')
                 for a,b in ((cand.action, bind) for bind in cand.bindings)
                 if b.initiator.module_id == 'unit-0' and b.connectors[1].module.module_id == 'unit-2')
    proposal = ActionBundle((ActionInstance(align[0].action_id, align[1].initiator,
                              participants=(align[1].initiator,), connectors=align[1].connectors),))
    with pytest.raises(ValueError,match='snapshot'):
        gate_v5_bundle(proposal,decision_state=before,fresh_state=stale,
                       physical_snapshot_confirmed=False,spec=spec,schema=feature_schema,claim_resolver=_claims)
    # Other free module remains allowed, but this particular initiator acquired a joint.
    with pytest.raises(ValueError,match='rejects|FixedJoint'):
        gate_v5_bundle(proposal,decision_state=before,fresh_state=stale,
                       physical_snapshot_confirmed=True,spec=spec,schema=feature_schema,claim_resolver=_claims)
    same = LivePhysicalState(10., detached)
    assert gate_v5_bundle(proposal,decision_state=before,fresh_state=same,
            physical_snapshot_confirmed=True,spec=spec,schema=feature_schema,claim_resolver=_claims) == proposal
    with pytest.raises(ValueError,match='regressed'):
        gate_v5_bundle(proposal,decision_state=before,fresh_state=LivePhysicalState(9.,detached),
            physical_snapshot_confirmed=True,spec=spec,schema=feature_schema,claim_resolver=_claims)


def test_live_v5_uses_actor_visibility_and_topology_in_tensorized_h2_batch(monkeypatch):
    import json
    from drl.IL.tests.bc_v1 import test_live_observation as live_fixture
    from drl.IL.tests.bc_v1.test_live_observation import (
        snapshot, builder, context, world_bridges, ROOT,
    )
    from drl.IL.bc_v1.smores_adapter import SMORESAdapter
    from drl.IL.bc_v1.live_observation import encode_smores_snapshot
    from drl.IL.bc_v1.bc_v5_live import prepare_v5_live
    from drl.IL.bc_v1.actor_visibility import load_visibility_policy
    from drl.IL.bc_v1.normalization import read_normalizers
    cfg = ROOT/'configs/bc_v1'
    from drl.IL.bc_v1.normalization import load_normalization_policy
    from drl.IL.bc_v1.feature_schema import load_feature_schema
    from drl.IL.bc_v1.tensorization import load_tensor_layout
    spec = c.load_robot_spec(cfg/'smores_ep.robot.json')
    feature_schema = load_feature_schema(cfg/'features.json')
    kw = {'spec': spec, 'feature_schema': feature_schema,
          'normalization_policy': load_normalization_policy(cfg/'normalization.json',feature_schema),
          'tensor_layout': load_tensor_layout(cfg/'tensor_layout.v1.json')}
    path = ROOT/'artifacts/bc_v1/normalization/f2_actor_visible_r1m_v1/normalizers.json'
    kw['normalizers'] = read_normalizers(path,
        expected_metadata=json.loads(path.read_text())['metadata'])
    visibility = load_visibility_policy(cfg/'actor_visibility_r1m.json')
    # The legacy F2 fixture is not bundled; the real actor F2 is available.
    monkeypatch.setattr(live_fixture, 'settings', lambda: kw)
    state = encode_smores_snapshot(snapshot(n=2, attached=True),
        adapter=SMORESAdapter(kw['spec']), graph_builder=builder())
    view = prepare_v5_live(state, visibility=visibility,
        context=context(),observation=c.ObservationBundle(),bridges=world_bridges(),**kw)
    assert view.inputs.candidates == enumerate_candidates_v5(state.graph,kw['spec'],kw['feature_schema'])
    assert len(view.batch['inputs']['structural']['initiator']) < 2*100
    assert len(view.batch['routing']['samples'][0]['structural_bindings']) == len(
        view.batch['inputs']['structural']['initiator'])
    # H1 receives exactly the actor-visible projection, not the full critic view.
    assert view.batch['routing']['samples'][0]['normalizer_fingerprint']
    with pytest.raises(ValueError, match='normalizer/visibility'):
        prepare_v5_live(state, visibility=visibility, context=context(),
            observation=c.ObservationBundle(),bridges=world_bridges(),
            **{**kw, 'normalizers': replace(kw['normalizers'], metadata_items=(('actor_visibility_fingerprint','bad'),))})


def test_regenerated_g2_roundtrip_and_h2_membership(tmp_path):
    from drl.IL.bc_v1.sidecar import SidecarWriter, SidecarDataset, verify_sidecar, _hash
    s, kw = fixture(3,2)
    original,fresh,exceptional = _repack(s,kw)
    identity = {name:'certified-fixture-identity' for name in (
        'canonical_sha','f1_sha','g1_fingerprint','f2_fingerprint','normalizer_sha',
        'layout_fingerprint','feature_schema_fingerprint','robot_spec_fingerprint',
        'tensorizer_fingerprint','storage_fingerprint','split_fingerprint')}
    identity['f2_fingerprint'] = fresh['routing']['normalizer_fingerprint']
    identity['layout_fingerprint'] = fresh['routing']['layout_fingerprint']
    eid = s.audit.episode_id
    entry = {'episode_id':eid,'samples':1,
             'transition_indices_fingerprint':_hash([s.audit.transition_index])}
    inventory = {'train':[entry],'validation':[],'test':[]}
    out = tmp_path/'g2_stage11'
    writer=SidecarWriter(out,identity,inventory)
    try:
        writer.add(fresh,'train',0)
        manifest=writer.finalize()
    finally:
        writer.close()
    with SidecarDataset(out,'train',expected_identity=identity) as data:
        saved=data[0]
        assert saved['inputs']['structural']['initiator'].shape == fresh['inputs']['structural']['initiator'].shape
        assert torch.equal(saved['inputs']['module_x'], original['inputs']['module_x'])
        batch=collate_action_set_samples([saved])
        assert int(batch['supervision']['action_set']['presence'].sum()) == 1
    assert verify_sidecar(out,expected_identity=identity)['samples'] == 1
