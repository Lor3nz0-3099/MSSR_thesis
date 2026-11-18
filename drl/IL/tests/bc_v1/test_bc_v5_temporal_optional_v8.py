"""BC-v5 native temporal policy: no external audit is required, unsafe rows still excluded."""
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from drl.IL.bc_v1 import bc_v5_integrity as integrity
from drl.IL.bc_v1.bc_v5_binary import index_overlay


def _row(*, unresolved=False, conflict=False):
    masks = {'m0': {'wheels': True, 'internal': True, 'structural': True}}
    return {
        'supervision': {'valid_for_behavior_cloning': True,
                        'module_resource_masks': masks,
                        'conflicts': ([{'module_id': 'm0', 'resource': 'internal',
                                       'reason': 'source_conflict'}] if conflict else [])},
        'action_t': {'modules': {'m0': {k: {'commanded': False}
                                       for k in ('wheels', 'internal', 'structural')}}},
        'provenance': {'unresolved_recorded_sequences': unresolved},
    }


def test_native_canonical_flags_are_not_removed():
    assert integrity.canonical_decision_flags(_row())['bad_temporal'] is False
    conflicted = integrity.canonical_decision_flags(_row(conflict=True))
    assert conflicted['bad_temporal'] is False
    assert conflicted['blocked'] == [['m0', 'internal']]
    assert 'source_conflict' in conflicted['reasons']
    assert integrity.canonical_decision_flags(_row(unresolved=True))['bad_temporal'] is True


def test_index_overlay_accepts_no_external_temporal_audit():
    row = {'split': 'train', 'episode_id': 'e0', 'transition_index': 0,
           'temporal_kind': None, 'exclude': False, 'blocked': []}
    report = {'ready_for_training': True, 'temporal_audit_provided': False,
              'rows': [row], 'expected_rows': {'train': 1, 'validation': 0}}
    assert index_overlay(report, split='train') == {('e0', 0): row}
    with pytest.raises(ValueError, match='Unexpected external'):
        index_overlay({**report, 'rows': [{**row, 'temporal_kind': 'single_decision'}]},
                      split='train')
    with pytest.raises(ValueError, match='Missing/invalid'):
        index_overlay({**report, 'temporal_audit_provided': True}, split='train')


def test_inspect_single_canonical_row_without_external_audit(tmp_path):
    canonical, g2 = tmp_path / 'canonical', tmp_path / 'g2'
    canonical.mkdir(); g2.mkdir()
    payload = (json.dumps(_row(), sort_keys=True) + '\n').encode('utf-8')
    (canonical / 'episode.jsonl').write_bytes(payload)
    source = {'episodes': [{'episode_id': 'e0', 'file': 'episode.jsonl',
                            'sha256': hashlib.sha256(payload).hexdigest()}]}
    (canonical / 'manifest.json').write_text(json.dumps(source))
    source_sha = integrity.file_hash(canonical / 'manifest.json')
    manifest = {'identity': {'canonical_sha': source_sha},
                'manifest_fingerprint': 'test-actor-g2',
                'inventory': {'train': [{'episode_id': 'e0', 'samples': 1}],
                              'validation': [], 'test': []}}
    (g2 / 'manifest.json').write_text(json.dumps(manifest))

    class FakeDataset:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def sample_keys(self):
            return [('e0', 0)] if current_split[0] == 'train' else []

    # Replace only external I/O boundaries. Real Canonical flags/loop are executed.
    current_split = ['train']
    class SplitDataset(FakeDataset):
        def __init__(self, _root, split, **kwargs):
            current_split[0] = split

    with patch.object(integrity, '_read_manifest', return_value=manifest), \
         patch.object(integrity, '_source'), \
         patch.object(integrity, 'SidecarDataset', SplitDataset):
        report = integrity.inspect(g2, canonical, temporal_path=None)

    assert report['complete'] is True
    assert report['ready_for_training'] is True
    assert report['temporal_audit_provided'] is False
    assert report['rows'][0]['temporal_kind'] is None
    assert report['rows'][0]['exclude'] is False
    assert report['counts']['rows'] == 1
