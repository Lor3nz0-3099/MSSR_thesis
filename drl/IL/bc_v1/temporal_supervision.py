"""Versioned temporal grouping audit; independent of policy and tensor types."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping


VERSION = 'mssr.bc_v1.temporal_supervision_audit.v1'
KINDS = frozenset({'single_decision', 'joint_single_point',
                  'temporally_coalesced_sequential', 'ambiguous'})


@dataclass(frozen=True)
class TemporalSupervisionAudit:
    episode_id: str
    transition_index: int
    kind: str

    def __post_init__(self):
        if type(self.episode_id) is not str or not self.episode_id.strip():
            raise ValueError('Missing audit episode identity')
        if type(self.transition_index) is not int or self.transition_index < 0:
            raise ValueError('Invalid canonical transition index')
        if type(self.kind) is not str or self.kind not in KINDS:
            raise ValueError('Unknown temporal supervision kind')


def validate_temporal_audit(entries, expected_keys) -> None:
    seen = set()
    for entry in entries:
        if type(entry) is not TemporalSupervisionAudit:
            raise ValueError('Expected typed temporal audit record')
        key = (entry.episode_id, entry.transition_index)
        if key in seen:
            raise ValueError('Duplicate temporal audit key')
        seen.add(key)
    if seen != set(expected_keys):
        raise ValueError('Missing or foreign temporal audit keys')


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False)


def _payload(entries, source_identity: Mapping):
    entries = tuple(entries)
    validate_temporal_audit(entries, {(x.episode_id, x.transition_index)
                                     for x in entries if type(x) is TemporalSupervisionAudit})
    if not isinstance(source_identity, Mapping):
        raise ValueError('Expected source identity mapping')
    return {'schema_version': VERSION, 'source_identity': dict(source_identity),
            'entries': [asdict(x) for x in sorted(entries,
                         key=lambda x: (x.episode_id, x.transition_index))]}


def temporal_audit_fingerprint(entries, source_identity: Mapping) -> str:
    return hashlib.sha256(_json(_payload(entries, source_identity)).encode()).hexdigest()


def write_temporal_audit(path: Path, entries, source_identity: Mapping, *, expected_keys) -> None:
    entries = tuple(entries)
    validate_temporal_audit(entries, expected_keys)
    payload = _payload(entries, source_identity)
    payload['fingerprint'] = temporal_audit_fingerprint(entries, source_identity)
    Path(path).write_text(_json(payload) + '\n', encoding='utf-8')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON object key')
        result[key] = value
    return result


def read_temporal_audit(path: Path, *, expected_source_sha256: str, expected_keys):
    doc = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_unique_object)
    if not isinstance(doc, dict) or set(doc) != {'schema_version', 'source_identity', 'entries', 'fingerprint'}:
        raise ValueError('Invalid temporal audit artifact')
    if doc['schema_version'] != VERSION:
        raise ValueError('Incompatible temporal audit version')
    if not isinstance(doc['source_identity'], dict) or doc['source_identity'].get('manifest_sha256') != expected_source_sha256:
        raise ValueError('Incompatible temporal audit source')
    try:
        entries = tuple(TemporalSupervisionAudit(**row) for row in doc['entries'])
    except (TypeError, KeyError) as exc:
        raise ValueError('Malformed temporal audit entries') from exc
    validate_temporal_audit(entries, expected_keys)
    if temporal_audit_fingerprint(entries, doc['source_identity']) != doc['fingerprint']:
        raise ValueError('Temporal audit fingerprint mismatch')
    return tuple(sorted(entries, key=lambda x: (x.episode_id, x.transition_index)))
