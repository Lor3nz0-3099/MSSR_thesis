"""Stage 10: opt-in supervised exceptions never weaken physical runtime rules."""
import copy
from dataclasses import replace

import pytest

from drl.IL.bc_v1.bc_v1_contract import ModuleRef, StructuralTarget
from drl.IL.bc_v1.bc_v5_offline_align_exceptions import (
    EXPECTED_SPLITS, _digest, build_registry, offline_supervised_candidates_for_row,
)
from drl.IL.bc_v1.bc_v5_joint_safe_candidates import live_align_is_physically_safe
from drl.IL.bc_v1.candidates import enumerate_candidates
from drl.IL.bc_v1.bc_v5_topology_candidates import enumerate_candidates_v5
from drl.IL.tests.bc_v1.test_candidates import attach, catalog, graph, schema


CANONICAL = 'sha256-of-test-canonical'
G2 = 'test-g2-fingerprint'


def violation(split, i, *, changes=None):
    result = {
        'split': split, 'episode_id': 'test-episode-'+split,
        'transition_index': i, 'initiator': 'unit-0',
        'status': 'physical_edge_but_executor_override_detached',
        'physical_links': [{'own_face':'socket-0','peer':'unit-1',
                            'peer_face':'socket-0',
                            'previous_event_on_face': {
                                'primitive': 'undock',
                                'transition_index': i-1,
                                'supervised': True}}],
        'next_physical_change': {'subsequent_buckets': 1,
                                 'transition_index': i+1,
                                 'all_previous_links_gone': True,
                                 'fully_isolated': True},
    }
    if changes:
        result.update(changes)
    return result


def reports():
    violations = ([violation('train', i+1) for i in range(EXPECTED_SPLITS['train'])] +
                  [violation('validation', i+1) for i in range(EXPECTED_SPLITS['validation'])])
    history = {'schema_version':'mssr.bc_v5.align_joint_history.v1',
               'complete':True,'no_test_payload_access':True,
               'canonical_sha256':CANONICAL,
               'g2_manifest_fingerprint':G2,
               'summary':{'total_violations': len(violations)},
               'violations':violations}
    preflight = {'schema_version':'mssr.bc_v5.topology_positive_preflight.v1',
                 'complete':True,'test_payload_access':0,
                 'source_manifest_sha256':CANONICAL,
                 'g2_manifest_fingerprint':G2,
                 'align_violations':len(violations),
                 'counts':{
                    split:{'align_singleton_violation':count}
                    for split,count in EXPECTED_SPLITS.items()}}
    return history, preflight


def _registry():
    return build_registry(*reports())


def _call(registry, *, split='train', mid='unit-0', target=None):
    g = attach(graph(3, 2), 0, 2)
    spec, policy = catalog(2), schema()
    old = enumerate_candidates(g, spec, policy)
    # The selected initiator unit-0 is still connected to unit-1; align
    # to the separate singleton unit-2 through unit-0's free socket-1.
    binding = next(b for c in old.structural_candidates
                   if c.action.action_id == 'align_faces'
                   for b in c.bindings
                   if b.initiator == ModuleRef('unit-0')
                   and b.connectors[1].module == ModuleRef('unit-2'))
    target = target or StructuralTarget(ModuleRef(mid), 'align_faces',
                                       (ModuleRef(mid), ModuleRef('unit-2')),
                                       binding.connectors)
    result = offline_supervised_candidates_for_row(
        g, spec, policy, split=split, episode_id='test-episode-train',
        transition_index=1, targets=(target,), exception_registry=registry)
    return g, result, target


def _contains_exact(table, target):
    return sum(b.initiator == target.initiator and b.connectors == target.connectors
               for c in table.structural_candidates
               if c.action.action_id == 'align_faces' for b in c.bindings)


def test_all_47_strict_evidence_create_only_offline_exceptions():
    doc = _registry()
    assert doc['complete']
    assert not doc['needs_review']
    assert len(doc['exceptions']) == 47
    assert doc['counts']['accepted'] == EXPECTED_SPLITS
    assert doc['ready_for_training'] is False
    assert doc['runtime_exception'] is False
    assert doc['temporal_authenticated'] is False
    assert doc['test_payload_access'] == 0
    assert doc['fingerprint'] == _digest({k:v for k,v in doc.items() if k!='fingerprint'})


@pytest.mark.parametrize('changes,reason', [
    ({'status':'also_connected_for_executor'},'executor_has_not_marked_module_detached'),
    ({'physical_links':[]},'no_recorded_physical_links'),
    ({'physical_links':[{'own_face':'socket-0','peer':'unit-1','peer_face':'socket-0',
                         'previous_event_on_face':None}]},'not_every_connected_face_has_prior_undock'),
    ({'physical_links':[{'own_face':'socket-0','peer':'unit-1','peer_face':'socket-0',
                         'previous_event_on_face':{'primitive':'undock',
                                                   'transition_index':0,'supervised':False}}]},
     'prior_undock_not_supervised'),
    ({'next_physical_change':None},'no_next_bucket_confirmed_isolation'),
    ({'next_physical_change':{'subsequent_buckets':2, 'transition_index':3,
                              'all_previous_links_gone': True,'fully_isolated':True}},
     'no_next_bucket_confirmed_isolation'),
])
def test_proof_fails_closed_without_changing_any_source(changes,reason):
    history,preflight = reports()
    history['violations'][0].update(changes)
    result = build_registry(history,preflight)
    assert not result['complete']
    assert result['counts']['accepted']['train'] == 33
    assert result['needs_review'][0]['reason'] == reason


def test_rejects_duplicate_or_cross_split_or_mismatched_sources():
    history,preflight = reports()
    history['violations'][1] = copy.deepcopy(history['violations'][0])
    with pytest.raises(ValueError,match='Duplicate'):
        build_registry(history,preflight)
    history,preflight = reports()
    preflight['source_manifest_sha256'] = 'other'
    with pytest.raises(ValueError,match='identity mismatch'):
        build_registry(history,preflight)
    history,preflight = reports()
    history['violations'][0]['split'] = 'test'
    with pytest.raises(ValueError,match='Invalid split'):
        build_registry(history,preflight)


def test_all_legacy_align_options_restored_not_only_supervised_binding():
    registry = _registry()
    g,result,target = _call(registry)
    current = enumerate_candidates_v5(g,catalog(2),schema())
    assert _contains_exact(current,target) == 0
    assert _contains_exact(result,target) == 1
    assert result.resource_candidates == current.resource_candidates
    # Restore ALL legacy align possibilities, not only the positive;
    # otherwise target-specific candidate presence would leak the label.
    legacy = enumerate_candidates(g,catalog(2),schema())
    legacy_align = {(c.endpoints,b) for c in legacy.structural_candidates
                    if c.action.action_id == 'align_faces' for b in c.bindings}
    restored_align = {(c.endpoints,b) for c in result.structural_candidates
                      if c.action.action_id == 'align_faces' for b in c.bindings}
    assert legacy_align <= restored_align
    assert len(restored_align) > 1
    assert {(c.action.action_id,c.endpoints,c.bindings) for c in result.structural_candidates
            if c.action.action_id!='align_faces'} == {
                (c.action.action_id,c.endpoints,c.bindings) for c in current.structural_candidates
                if c.action.action_id!='align_faces'}
    # Physical execution must still refuse to align a connected initiator.
    assert not live_align_is_physically_safe(ModuleRef('unit-0'),g,
                                              physical_snapshot_confirmed=True)


def test_live_test_and_unapproved_rows_cannot_use_the_exception():
    registry = _registry()
    g = attach(graph(3,2),0,2)
    for split in ('test','live','inference'):
        with pytest.raises(ValueError,match='forbidden'):
            offline_supervised_candidates_for_row(
                g,catalog(2),schema(), split=split, episode_id='test-episode-train',
                transition_index=1, targets=(), exception_registry=registry)
    current = enumerate_candidates_v5(g,catalog(2),schema())
    not_exception = offline_supervised_candidates_for_row(
        g,catalog(2),schema(),split='train',episode_id='other',
        transition_index=1,targets=(),exception_registry=registry)
    assert current == not_exception
    registry['exceptions'][0]['initiator'] = 'other'
    with pytest.raises(ValueError,match='fingerprint'):
        offline_supervised_candidates_for_row(
            g,catalog(2),schema(),split='train',episode_id='test-episode-train',
            transition_index=1,targets=(),exception_registry=registry)


def test_exceptions_do_not_allow_same_component_targets():
    registry = _registry()
    g = attach(graph(3,2),0,2)
    old = enumerate_candidates(g,catalog(2),schema())
    b = next(b for c in old.structural_candidates
             if c.action.action_id=='align_faces' for b in c.bindings
             if b.initiator == ModuleRef('unit-0')
             and b.connectors[1].module == ModuleRef('unit-1'))
    target = StructuralTarget(ModuleRef('unit-0'), 'align_faces',
                              (ModuleRef('unit-0'),ModuleRef('unit-1')),
                              b.connectors)
    with pytest.raises(ValueError, match='distinct physical component'):
        offline_supervised_candidates_for_row(
            g,catalog(2),schema(),split='train',episode_id='test-episode-train',
            transition_index=1,targets=(target,),exception_registry=registry)


def test_registry_prohibits_unlisted_initiator_and_targets():
    registry = _registry()
    g = attach(graph(3,2),0,2)
    with pytest.raises(ValueError,match='Exact positive align'):
        offline_supervised_candidates_for_row(
            g,catalog(2),schema(),split='train',episode_id='test-episode-train',
            transition_index=1,targets=(),exception_registry=registry)


def test_cli_report_is_bound_to_sources_and_refuses_overwrite(tmp_path, capsys):
    from json import dumps, loads
    from drl.IL.bc_v1.bc_v5_offline_align_exceptions import main, read_registry
    history, preflight = reports()
    old_history, old_preflight = copy.deepcopy(history), copy.deepcopy(preflight)
    a, b, out = (tmp_path/'history.json', tmp_path/'preflight.json', tmp_path/'report.json')
    a.write_text(dumps(history))
    b.write_text(dumps(preflight))
    main(['--history', str(a), '--preflight', str(b), '--output', str(out)])
    assert loads(capsys.readouterr().out)['complete'] is True
    assert history == old_history and preflight == old_preflight
    loaded = read_registry(out, canonical_sha256=CANONICAL, g2_manifest_fingerprint=G2)
    assert len(loaded['exceptions']) == 47
    with pytest.raises(ValueError, match='G2 identity mismatch'):
        read_registry(out, g2_manifest_fingerprint='wrong')
    with pytest.raises(SystemExit):
        main(['--history', str(a), '--preflight', str(b), '--output', str(out)])
    loaded['exceptions'][0]['initiator'] = 'tampered'
    out.write_text(dumps(loaded))
    with pytest.raises(ValueError, match='corrupted'):
        read_registry(out)


def test_prior_undock_must_be_in_previous_one_or_two_buckets():
    history, preflight = reports()
    history['violations'][0]['physical_links'][0]['previous_event_on_face']['transition_index'] = -4
    doc = build_registry(history, preflight)
    assert doc['complete'] is False
    assert doc['needs_review'][0]['reason'] == 'undock_not_one_or_two_prior_buckets'


def test_legacy_fallback_never_changes_other_rows_or_existing_v5_actions():
    registry = _registry()
    g = attach(graph(3,2),0,2)
    v5 = enumerate_candidates_v5(g,catalog(2),schema())
    v5_without_exception = offline_supervised_candidates_for_row(
        g,catalog(2),schema(),split='validation',episode_id='test-episode-train',
        transition_index=1,targets=(),exception_registry=registry)
    assert v5_without_exception == v5
    # Completely disconnected row must not be "repaired" using a special label.
    free = graph(3,2)
    t = StructuralTarget(ModuleRef('unit-0'), 'align_faces',
                         (ModuleRef('unit-0'),ModuleRef('unit-2')),
                         (free.connectors[0].ref,free.connectors[4].ref))
    with pytest.raises(ValueError,match='physically isolated'):
        offline_supervised_candidates_for_row(
            free,catalog(2),schema(),split='train',episode_id='test-episode-train',
            transition_index=1,targets=(t,),exception_registry=registry)
