"""Regression tests for read-only per-candidate BC-v5 counts (no production G2)."""
from collections import Counter
import json

import pytest

from drl.IL.bc_v1 import bc_v5_candidate_census as candidate
from drl.IL.bc_v1.bc_v5_integrity import write_report
from drl.IL.bc_v1.sidecar import SidecarDataset
from drl.IL.tests.bc_v1.test_bc_v5_census_sampling import _source
from drl.IL.tests.bc_v1.test_sidecar import identity


def _overlay(root, out, *, completed=True, excluded=None, blocked=None):
    excluded = excluded or set()
    blocked = blocked or {}
    raw = json.loads((root / 'manifest.json').read_text())
    rows = []
    expected = {}
    for split in ('train', 'validation'):
        with SidecarDataset(root, split, expected_identity=identity()) as data:
            expected[split] = len(data)
            for eid, idx in data.sample_keys():
                rows.append({'split': split, 'episode_id': eid,
                             'transition_index': idx,
                             'temporal_kind': ('ambiguous' if (eid, idx) in excluded else 'joint_single_point')
                             if completed else None,
                             'blocked': blocked.get((eid, idx), []),
                             'exclude': (eid, idx) in excluded})
    report = {
        'schema_version': 'mssr.bc_v5.binary_decision_audit.v2',
        'g2_manifest_fingerprint': raw['manifest_fingerprint'],
        'canonical_manifest_sha256': 'canonical-test',
        'temporal_audit_provided': completed,
        'complete': completed,
        'ready_for_training': completed,
        'expected_rows': expected,
        'rows': rows,
    }
    write_report(out, report)
    return out


def test_counts_positive_negative_and_temporal_exclusions(tmp_path):
    root = _source(tmp_path)
    overlay = _overlay(root, tmp_path/'overlay.json', excluded={('b', 7)})
    counts = candidate.analyze(root, overlay)
    train = counts['splits']['train']
    assert counts['complete'] and counts['ready_for_training']
    assert counts['test_payload_access'] == 0
    assert train['summary']['inspected_rows'] == 3
    assert train['summary']['admitted_rows'] == 2
    assert train['summary']['temporally_excluded_rows'] == 1
    assert train['summary']['positive'] > 0
    assert train['summary']['negative'] > 0
    assert train['summary']['temporal_excluded_candidates'] > 0
    assert sum(t.get('positive', 0) for t in train['per_primitive'].values()) == train['summary']['positive']
    assert sum(t.get('negative', 0) for t in train['per_primitive'].values()) == train['summary']['negative']
    assert train['per_primitive']['resource:pace']['positive'] == 2
    assert train['per_primitive']['resource:pace']['temporal_excluded_positive'] == 1
    assert counts['splits']['validation']['summary']['inspected_rows'] == 1


def test_no_temporal_audit_means_provisional_census_only(tmp_path):
    root = _source(tmp_path)
    overlay = _overlay(root, tmp_path/'overlay.json', completed=False)
    with pytest.raises(ValueError, match='complete'):
        candidate.analyze(root, overlay)
    report = candidate.analyze(root, overlay, pilot_rows=1)
    assert not report['ready_for_training'] and not report['complete']
    assert report['splits']['train']['summary']['inspected_rows'] == 1
    assert report['splits']['validation']['summary']['inspected_rows'] == 1


def test_full_native_only_overlay_is_training_ready(tmp_path):
    root = _source(tmp_path)
    overlay = _overlay(root, tmp_path/'overlay.json', completed=True)
    doc = json.loads(overlay.read_text())
    doc['temporal_audit_provided'] = False
    doc['ready_for_training'] = True
    for row in doc['rows']:
        row['temporal_kind'] = None
    from drl.IL.bc_v1.bc_v5_integrity import audit_fingerprint
    doc.pop('fingerprint')
    doc['fingerprint'] = audit_fingerprint(doc)
    overlay.write_text(json.dumps(doc))
    result = candidate.analyze(root, overlay)
    assert result['complete'] and result['ready_for_training']
    assert not result['temporal_authenticated']


def test_mask_math_without_fabricating_negative_labels():
    info = [('resource:tilt_target', 'm0', 'internal'),
            ('resource:pan_velocity', 'm0', 'internal'),
            ('resource:wheel_velocity', 'm1', 'wheels')]
    per_resource = {('m0', 'internal'): Counter({'resource:tilt_target': 1, 'resource:pan_velocity': 1})}
    assert candidate._blocked_counts(info, per_resource, [['m0', 'internal']]) == per_resource[('m0', 'internal')]
    with pytest.raises(ValueError, match='Unknown G2'):
        candidate._blocked_counts(info + [('resource:unexpected', 'm0', None)], per_resource,
                                  [['m0', 'internal']])


def test_output_collision_fails_and_cannot_read_test(tmp_path):
    root = _source(tmp_path, test_count=1)
    overlay = tmp_path/'not-relevant.json'
    with pytest.raises(ValueError, match='TEST'):
        candidate.analyze(root, overlay)
