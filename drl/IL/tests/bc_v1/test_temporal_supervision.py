"""Audit-only metadata: reject drift, never widen the policy boundary."""
import importlib
import importlib.util
import json
from dataclasses import FrozenInstanceError, fields
import subprocess
import sys

import pytest


KINDS = ('single_decision', 'joint_single_point',
         'temporally_coalesced_sequential', 'ambiguous')
SOURCE = {'manifest_sha256': 'a' * 64}
KEYS = frozenset({('episode-A', 3), ('episode-B', 7)})


def api():
    name = 'drl.IL.bc_v1.temporal_supervision'
    assert importlib.util.find_spec(name) is not None, 'Missing Task 1 temporal audit contract'
    return importlib.import_module(name)


def entries():
    m = api()
    return (m.TemporalSupervisionAudit('episode-A', 3, KINDS[2]),
            m.TemporalSupervisionAudit('episode-B', 7, KINDS[3]))


@pytest.mark.parametrize('kind', KINDS)
def test_audit_contract_exact_enum_and_frozen(kind):
    x = api().TemporalSupervisionAudit('episode', 0, kind)
    with pytest.raises(FrozenInstanceError):
        x.kind = 'joint_single_point'
    api().validate_temporal_audit((x,), frozenset({('episode', 0)}))


@pytest.mark.parametrize('episode,index,kind', [('', 0, KINDS[0]),
    ('e', -1, KINDS[0]), ('e', True, KINDS[0]), ('e', 1.0, KINDS[0]),
    ('e', 0, 'sequential'), ('e', 0, 'UNKNOWN')])
def test_invalid_identity_and_unknown_kind_rejected(episode, index, kind):
    with pytest.raises(ValueError):
        api().TemporalSupervisionAudit(episode, index, kind)


def test_audit_is_not_policy_input():
    m = api()
    from drl.IL.bc_v1.bc_v1_contract import PolicyInputs, TaskContext
    assert tuple(f.name for f in fields(PolicyInputs)) == ('graph', 'observation', 'context')
    assert tuple(f.name for f in fields(TaskContext)) == ('task_id', 'static_geometry')
    assert m.TemporalSupervisionAudit not in [f.type for f in fields(PolicyInputs)]


@pytest.mark.parametrize('fault', ['duplicate', 'missing', 'foreign'])
def test_duplicate_missing_foreign_keys_fail(fault):
    m = api(); values = entries()
    if fault == 'duplicate': values += (values[0],)
    elif fault == 'missing': values = values[:1]
    else: values += (m.TemporalSupervisionAudit('other', 8, KINDS[0]),)
    with pytest.raises(ValueError):
        m.validate_temporal_audit(values, KEYS)


def test_temporal_fingerprint_order_invariant_content_sensitive():
    m = api(); values = entries()
    before = m.temporal_audit_fingerprint(values, SOURCE)
    assert len(before) == 64
    assert before == m.temporal_audit_fingerprint(tuple(reversed(values)), dict(SOURCE))
    assert before != m.temporal_audit_fingerprint(values, {'manifest_sha256': 'b' * 64})
    different = (m.TemporalSupervisionAudit('episode-A', 3, KINDS[0]), values[1])
    assert before != m.temporal_audit_fingerprint(different, SOURCE)


def test_writer_roundtrip_deterministic_bytes(tmp_path):
    m = api(); first = tmp_path/'first.json'; second = tmp_path/'second.json'
    m.write_temporal_audit(first, entries(), SOURCE, expected_keys=KEYS)
    m.write_temporal_audit(second, tuple(reversed(entries())), SOURCE, expected_keys=KEYS)
    assert first.read_bytes() == second.read_bytes()
    assert m.read_temporal_audit(first, expected_source_sha256='a'*64, expected_keys=KEYS) == entries()
    with pytest.raises(ValueError):
        m.read_temporal_audit(first, expected_source_sha256='b'*64, expected_keys=KEYS)


@pytest.mark.parametrize('fault', ['kind', 'version', 'duplicate', 'fingerprint'])
def test_reader_rejects_corrupt_or_stale_artifact(tmp_path, fault):
    m = api(); path = tmp_path/'audit.json'
    m.write_temporal_audit(path, entries(), SOURCE, expected_keys=KEYS)
    doc = json.loads(path.read_text())
    if fault == 'kind': doc['entries'][0]['kind'] = KINDS[0]
    elif fault == 'version': doc['schema_version'] = 'other'
    elif fault == 'duplicate': doc['entries'].append(doc['entries'][0])
    else: doc['fingerprint'] = '0'*64
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        m.read_temporal_audit(path, expected_source_sha256='a'*64, expected_keys=KEYS)


def test_audit_import_without_policy_tensor_or_hardware_dependencies():
    api()
    code = '''
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, name, *args):
  if name.startswith(('numpy', 'torch', 'rclpy', 'isaac', 'smores', 'drl.IL.bc_v1.policy_inputs', 'drl.IL.bc_v1.bc_v1_contract')):
   raise RuntimeError(name)
sys.meta_path.insert(0, Block())
import drl.IL.bc_v1.temporal_supervision
'''
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
