"""BC-v5: immutable Canonical ↔ G2 TRAIN/VAL integrity and decision audit.

A BC-valid row may contain unlabelled COMMANDS. This audit preserves the
binary set-membership objective on trustworthy resources and refuses to turn
masked commands into H2 negatives. An independent temporal classification is optional. Canonical v8's native
conflict masks, R16 and unresolved-sequence indicators remain authoritative.

Does not read TEST payload, write to Canonical/G2, or change excluded episodes.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .dataset import file_hash
from .sidecar import SidecarDataset, _read_manifest
from .action_set_training import _source
from .temporal_supervision import (TemporalSupervisionAudit,
    temporal_audit_fingerprint)

SCHEMA = 'mssr.bc_v5.binary_decision_audit.v2'
GOOD_TEMPORAL = frozenset(('single_decision', 'joint_single_point'))
BAD_TEMPORAL = frozenset(('temporally_coalesced_sequential', 'ambiguous'))
RESOURCES = frozenset(('wheels', 'internal', 'structural'))


def canonical_decision_flags(row):
    """Inspect supervision completeness without inferring negatives from masks.

    A commanded but *unlabelled* resource cannot be used to supervise absence.
    Even if its selected command was suppressed by a structural skill's claim,
    it must not be treated as a confidently absent action.
    """
    sup = row['supervision']
    if type(sup.get('valid_for_behavior_cloning')) is not bool:
        raise ValueError('Missing Canonical BC eligibility flag')
    if not sup['valid_for_behavior_cloning']:
        return {'eligible': False, 'blocked': [], 'bad_temporal': False, 'reasons': []}
    modules = row['action_t']['modules']
    masks = sup['module_resource_masks']
    if not isinstance(modules, dict) or not isinstance(masks, dict):
        raise ValueError('Invalid Canonical module/mask contract')
    blocked = set()
    reasons = set()
    for mid, decisions in modules.items():
        if mid not in masks or not isinstance(masks[mid], dict):
            raise ValueError('Missing per-module supervision mask')
        for resource in RESOURCES:
            command = decisions[resource]
            flagged = command.get('commanded')
            valid = masks[mid].get(resource)
            if type(flagged) is not bool or type(valid) is not bool:
                raise ValueError('Invalid commanded/mask bool')
            if flagged and not valid:
                blocked.add((mid, resource))
                reasons.add('unlabelled_command')
    # Contradictory-source conflicts must never create false absent actions.
    for conflict in sup.get('conflicts') or []:
        mid, res = conflict.get('module_id'), conflict.get('resource')
        if mid is None or res not in RESOURCES:
            reasons.add('unlocalizable_conflict')
        else:
            if mid not in modules:
                raise ValueError('Conflict references absent module')
            blocked.add((mid, res))
            reasons.add('source_conflict')
    provenance = row.get('provenance') or {}
    if provenance.get('unresolved_recorded_sequences'):
        reasons.add('unresolved_recorded_sequence')
    if any(c.get('reason') == 'r16_unresolved_logical_dispatch_order'
           for c in sup.get('conflicts') or []):
        reasons.add('r16_unresolved_logical_dispatch_order')
    bad = bool(reasons.intersection((
        'unlocalizable_conflict', 'unresolved_recorded_sequence',
        'r16_unresolved_logical_dispatch_order')))
    return {'eligible': True, 'blocked': [list(x) for x in sorted(blocked)],
            'bad_temporal': bad, 'reasons': sorted(reasons)}


def load_temporal(path, canonical_sha):
    """Optional legacy temporal classification; never required to train."""
    if path is None:
        return None
    doc = json.loads(Path(path).read_text(encoding='utf-8'))
    if (set(doc) != {'schema_version', 'source_identity', 'entries', 'fingerprint'}
            or doc['schema_version'] != 'mssr.bc_v1.temporal_supervision_audit.v1'
            or doc['source_identity'].get('manifest_sha256') != canonical_sha):
        raise ValueError('Temporal audit source/schema mismatch')
    rows = tuple(TemporalSupervisionAudit(**item) for item in doc['entries'])
    if temporal_audit_fingerprint(rows, doc['source_identity']) != doc['fingerprint']:
        raise ValueError('Temporal audit fingerprint mismatch')
    mapping = {}
    for record in rows:
        key = (record.episode_id, record.transition_index)
        if key in mapping:
            raise ValueError('Duplicate temporal audit key')
        mapping[key] = record.kind
    return mapping


def inspect(g2, canonical, *, temporal_path=None, max_episodes=0):
    """Inspect exact G2 membership using index metadata, no TEST tensor reads.

    max_episodes limits diagnostic work and marks the overlay incomplete. Full
    audit hashes each included Canonical episode and checks every eligible row
    against its G2 TRAIN/VAL membership. Only a full verified overlay may train.
    """
    if type(max_episodes) is not int or max_episodes < 0:
        raise ValueError('Invalid episode limit')
    g2, canonical = Path(g2).resolve(), Path(canonical).resolve()
    raw = json.loads((g2/'manifest.json').read_text())
    manifest = _read_manifest(g2, raw['identity'])
    _source(manifest)  # Reject any TEST membership before dataset access.
    source_sha = file_hash(canonical/'manifest.json')
    if source_sha != manifest['identity']['canonical_sha']:
        raise ValueError('Canonical source is not the exact G2 source (manifest SHA mismatch)')
    source = json.loads((canonical/'manifest.json').read_text())
    entries = {x['episode_id']: x for x in source['episodes']}
    if len(entries) != len(source['episodes']):
        raise ValueError('Duplicate Canonical episode')
    wanted = {}
    split_counts = {}
    for split in ('train', 'validation'):
        with SidecarDataset(g2, split, expected_identity=manifest['identity']) as ds:
            keys = list(ds.sample_keys())  # Metadata only; authenticated G2 index.
            wanted.update({key: split for key in keys})
            split_counts[split] = len(keys)
    if len(wanted) != sum(split_counts.values()):
        raise ValueError('TRAIN/VAL G2 duplicate identities')
    episode_ids = {entry['episode_id'] for s in ('train', 'validation')
                   for entry in manifest['inventory'][s]}
    if not episode_ids <= set(entries):
        raise ValueError('Missing Canonical episodes for G2')
    listed = sorted(episode_ids)
    selected = listed[:max_episodes] if max_episodes else listed
    temporal = load_temporal(temporal_path, source_sha)
    by_episode = defaultdict(dict)
    for (eid, transition), split in wanted.items():
        if eid in selected:
            by_episode[eid][transition] = split
    counts, by_split, per_resource, per_reason = Counter(), defaultdict(Counter), Counter(), Counter()
    examples, rows = [], []
    for eid in selected:
        entry = entries[eid]
        path = (canonical/entry['file']).resolve()
        if not path.is_relative_to(canonical) or not path.is_file():
            raise ValueError('Unsafe/missing Canonical episode path')
        digest = hashlib.sha256()
        matched = 0
        with path.open('rb') as stream:
            for idx, line in enumerate(stream):
                digest.update(line)
                if idx not in by_episode[eid]:
                    continue
                row = json.loads(line)
                flags = canonical_decision_flags(row)
                if not flags['eligible']:
                    raise ValueError(f'G2 includes non-BC-valid row: {eid}/{idx}')
                split = by_episode[eid][idx]
                kind = None if temporal is None else temporal.get((eid, idx))
                if temporal is not None and kind is None:
                    raise ValueError(f'Temporal audit missing TRAIN/VAL key: {eid}/{idx}')
                reject = flags['bad_temporal'] or kind in BAD_TEMPORAL
                blocked = flags['blocked']
                counts['rows'] += 1
                by_split[split]['rows'] += 1
                if reject:
                    by_split[split]['excluded_temporally'] += 1
                    counts['excluded_temporally'] += 1
                else:
                    counts['admitted'] += 1
                    by_split[split]['admitted'] += 1
                if blocked:
                    counts['rows_with_partial_supervision'] += 1
                    by_split[split]['rows_with_partial_supervision'] += 1
                for _, resource in blocked:
                    per_resource[resource] += 1
                for reason in flags['reasons']:
                    per_reason[reason] += 1
                if reject or blocked:
                    if len(examples) < 25:
                        examples.append({'split': split, 'episode_id': eid,
                                         'transition_index': idx, 'temporal_kind': kind,
                                         'reasons': flags['reasons'], 'blocked': blocked})
                rows.append({'split': split, 'episode_id': eid, 'transition_index': idx,
                             'temporal_kind': kind, 'exclude': reject, 'blocked': blocked})
                matched += 1
        if matched != len(by_episode[eid]):
            raise ValueError('G2 indices not present in Canonical episode '+eid)
        if digest.hexdigest() != entry['sha256']:
            raise ValueError('Canonical episode changed: '+eid)
    complete = len(selected) == len(listed)
    if complete and counts['rows'] != sum(split_counts.values()):
        raise ValueError('Incomplete TRAIN/VAL Canonical ↔ G2 audit')
    return {'schema_version': SCHEMA,
            'g2_manifest_fingerprint': manifest['manifest_fingerprint'],
            'canonical_manifest_sha256': source_sha,
            'temporal_audit_provided': temporal is not None,
            'complete': complete, 'test_payload_access': 0,
            'splits': {k: dict(v) for k, v in by_split.items()},
            'expected_rows': split_counts, 'counts': dict(counts),
            'blocked_resources': dict(per_resource), 'reasons': dict(per_reason),
            'examples': examples, 'rows': rows,
            'ready_for_training': complete,
            'temporal_policy': 'canonical_v8_intrinsic_flags_with_optional_external_audit',
            'note': ('Complete Canonical/G2 overlay: native resource conflict masks and '
                     'unresolved R16/sequence exclusion enforced; no independent '
                     'temporal classification is required or inferred.') }


def audit_fingerprint(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def write_report(output, report):
    path = Path(output).resolve()
    if path.exists():
        raise ValueError('Output collision: report may not be overwritten')
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(report)
    data['fingerprint'] = audit_fingerprint(report)
    path.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False)+'\n')
    return data


def read_report(path, *, expected_g2_fingerprint, ready=True):
    data = json.loads(Path(path).read_text())
    fingerprint = data.pop('fingerprint', None)
    if fingerprint != audit_fingerprint(data):
        raise ValueError('Invalid BC-v5 overlay fingerprint')
    if data.get('schema_version') != SCHEMA or data.get('g2_manifest_fingerprint') != expected_g2_fingerprint:
        raise ValueError('Incompatible BC-v5 overlay identity')
    if data.get('schema_version') == SCHEMA and data.get('temporal_audit_provided') is False:
        if any(row.get('temporal_kind') is not None for row in data.get('rows', ())):
            raise ValueError('Native-only overlay must not invent external temporal categories')
    if ready and not data.get('ready_for_training'):
        raise ValueError('BC-v5 Canonical/G2 audit is incomplete or not training-ready')
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--g2', type=Path, required=True)
    parser.add_argument('--canonical', type=Path, required=True)
    parser.add_argument('--temporal-audit', type=Path)
    parser.add_argument('--max-episodes', type=int, default=0,
                        help='Diagnostic only, 0=full TRAIN/VAL (no TEST payload)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.output.resolve()
    if out.exists() or any(out == p.resolve() or out.is_relative_to(p.resolve())
                           for p in (args.g2, args.canonical)):
        parser.error('Refusing overwrite or writes inside source datasets')
    result = write_report(out, inspect(args.g2, args.canonical,
        temporal_path=args.temporal_audit, max_episodes=args.max_episodes))
    print(json.dumps({'report': str(out), 'complete': result['complete'],
                      'ready_for_training': result['ready_for_training'],
                      'counts': result['counts'], 'blocked_resources': result['blocked_resources'],
                      'reasons': result['reasons'], 'test_payload_access': 0}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
