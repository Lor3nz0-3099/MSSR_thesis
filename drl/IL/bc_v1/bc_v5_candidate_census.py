"""BC-v5: exact TRAIN/VALIDATION candidate/label counts, using Canonical overlay.

Read-only. Performs *no* temporal inference. An independent temporal audit is
optional; Canonical v8 conflict/exclusion flags remain mandatory. Reports positive, negative,
conflict-masked, and temporal-excluded candidates separately by primitive,
morphology/task/source. All negative counts assume the reliable portions of an
expert action set are complete, as in the approved BC-v5 binary contract.

The Canonical/G2 overlay is created by ``bc_v5_integrity``; this reader only
counts its authenticated rows. Does not read G2 TEST records or alter datasets.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .action_set_targets import candidate_index, positive_indices
from .action_set_training import _source
from .bc_v5_integrity import read_report
from .bc_v5_census import _known_primitives
from .sidecar import SidecarDataset, _read_manifest

SCHEMA = 'mssr.bc_v5.candidate_census.v1'
RESOURCE_CANONICAL = {'locomotion': 'wheels', 'internal_motion': 'internal'}


def _primitive(candidate, *, structural=False):
    return ('structural:' + candidate.action_id if structural else
            'resource:' + candidate.mode.mode_id)


def _static_index(template):
    """Once per G2 template; the category matches actual Stage-5 masking."""
    route = template['routing']
    info = []
    for candidate in route['resource_candidates']:
        info.append((_primitive(candidate),
                     candidate.resource_ref.module.module_id,
                     RESOURCE_CANONICAL.get(candidate.resource_ref.resource_id)))
    for action, binding in route['structural_bindings']:
        info.append((_primitive(action, structural=True),
                     binding.initiator.module_id, 'structural'))
    per_type = Counter(name for name, _, _ in info)
    per_resource = defaultdict(Counter)
    for name, module_id, kind in info:
        per_resource[(module_id, kind)][name] += 1
    return (info, per_type, per_resource, candidate_index(route))


def _audit_rows(report, split):
    rows = {}
    for row in report['rows']:
        if row['split'] != split:
            continue
        key = row['episode_id'], row['transition_index']
        if key in rows:
            raise ValueError('Duplicate overlay identity')
        rows[key] = row
    if report['complete'] and len(rows) != report['expected_rows'][split]:
        raise ValueError('Incomplete overlay for '+split)
    return rows


def _blocked_counts(info, per_resource, blocked):
    masked = Counter()
    seen = set()
    for pair in blocked:
        if (not isinstance(pair, list) or len(pair) != 2 or
                any(type(v) is not str or not v for v in pair)):
            raise ValueError('Invalid blocked resource')
        key = tuple(pair)
        if key in seen:
            raise ValueError('Repeated blocked resource')
        seen.add(key)
        module_id, resource = key
        if resource not in ('internal', 'wheels', 'structural'):
            raise ValueError('Unknown Canonical resource')
        # Match Stage-5 fail-closed unknown category behavior.
        if any(mid == module_id and category is None for _, mid, category in info):
            raise ValueError('Unknown G2 resource for blocked module')
        masked.update(per_resource.get(key, {}))
    return masked


def _row_counts(template, dynamic, entry, static):
    info, all_counts, per_resource, identity_index = static
    route = {**template['routing'], **dynamic['routing']}
    positive = Counter(info[index][0] for index in positive_indices(route, identity_index))
    if sum(positive.values()) == 0:
        raise ValueError('BC sample with no positive target')
    if any(positive[p] > all_counts[p] for p in positive):
        raise ValueError('More positives than available candidates')
    masked = _blocked_counts(info, per_resource, entry['blocked'])
    # Positive candidates may never be masked (same invariant as Stage-5 collation).
    positive_indices_set = set(positive_indices(route, identity_index))
    blocked_pairs = {tuple(x) for x in entry['blocked']}
    for index in positive_indices_set:
        _, mid, category = info[index]
        if (mid, category) in blocked_pairs:
            raise ValueError('Overlay attempts to mask a positive candidate')
    for p, count in masked.items():
        if count > all_counts[p]:
            raise ValueError('Masked count exceeds candidate count')
    return all_counts, positive, masked


def analyze(g2, overlay, *, pilot_rows=0):
    """Count only authenticated G2 TRAIN/VAL. pilot_rows=0 counts all overlay rows.

    Without a temporal audit, counts are explicitly PRELIMINARY: old coalesced
    labels are counted as currently present, not automatically called safe.
    """
    if type(pilot_rows) is not int or pilot_rows < 0:
        raise ValueError('Invalid pilot row count')
    root = Path(g2).resolve()
    raw = json.loads((root/'manifest.json').read_text())
    manifest = _read_manifest(root, raw['identity'])
    _source(manifest)
    report = read_report(overlay,
                         expected_g2_fingerprint=manifest['manifest_fingerprint'],
                         ready=False)
    if not report.get('complete') and pilot_rows == 0:
        raise ValueError('Full census requires complete Canonical/G2 overlay')
    result = {'schema_version': SCHEMA,
              'g2_manifest_fingerprint': manifest['manifest_fingerprint'],
              'canonical_sha256': report['canonical_manifest_sha256'],
              'overlay_fingerprint': json.loads(Path(overlay).read_text())['fingerprint'],
              'temporal_authenticated': report['temporal_audit_provided'],
              'complete': bool(report['complete'] and pilot_rows == 0),
              'ready_for_training': bool(report['ready_for_training'] and pilot_rows == 0),
              'test_payload_access': 0,
              'count_interpretation': ('Candidate membership is binary. Unselected candidates '
                 'are negative unless explicitly masked; temporally excluded rows are NOT training '
                 'examples. In native-only mode, temporal exclusions come from Canonical v8 flags.'),
              'splits': {}}
    for split in ('train', 'validation'):
        overlay_rows = _audit_rows(report, split)
        by_primitive = defaultdict(Counter)
        by_context = defaultdict(Counter)
        summary = Counter()
        cache = {}
        matched = set()
        with SidecarDataset(root, split, expected_identity=manifest['identity'],
                            template_cache_size=16) as data:
            # Pilot reads only the indexed overlay identities, not a prefix
            # of potentially tens of thousands of unrelated G2 records.
            if pilot_rows:
                keys = list(overlay_rows)[:pilot_rows]
                indices = []
                for eid, transition in keys:
                    match = data.db.execute(
                        'SELECT idx FROM samples WHERE split=? AND episode=? AND transition_idx=?',
                        (split, eid, transition)).fetchone()
                    if match is None:
                        raise ValueError('Overlay key absent from G2 '+str((eid, transition)))
                    indices.append(match[0])
            else:
                indices = range(len(data))
            for idx in indices:
                tid, template, dynamic = data.read_parts(idx)
                audit = dynamic['routing']['audit']
                key = audit.episode_id, audit.transition_index
                if key not in overlay_rows or key in matched:
                    if report['complete'] or key in matched:
                        raise ValueError('Missing/duplicate Canonical overlay key '+str(key))
                    continue
                matched.add(key)
                entry = overlay_rows[key]
                if tid not in cache:
                    cache[tid] = _static_index(template)
                all_counts, positive, masked = _row_counts(template, dynamic, entry, cache[tid])
                summary['inspected_rows'] += 1
                is_temporal = bool(entry['exclude'])
                if is_temporal:
                    summary['temporally_excluded_rows'] += 1
                else:
                    summary['admitted_rows'] += 1
                    if masked:
                        summary['partially_masked_rows'] += 1
                morph = (audit.morphology_stratum or '<missing>')
                task = (audit.task_stratum or '<missing>')
                provenance = (audit.provenance or '<missing>')
                for primitive, candidates in all_counts.items():
                    pos = positive[primitive]
                    unknown = masked[primitive]
                    if pos + unknown > candidates:
                        raise ValueError('Incorrect candidate partition')
                    dest = by_primitive[primitive]
                    if is_temporal:
                        dest['temporal_excluded'] += candidates
                        summary['temporal_excluded_candidates'] += candidates
                        dest['temporal_excluded_positive'] += pos
                    else:
                        dest['positive'] += pos
                        dest['negative'] += candidates - pos - unknown
                        dest['masked'] += unknown
                        summary['positive'] += pos
                        summary['negative'] += candidates - pos - unknown
                        summary['masked'] += unknown
                        dest['candidate_rows'] += candidates
                    # Always track how many rows offer this primitive, even if it is never selected.
                    dest['offered_in_rows'] += 1
                    if pos and not is_temporal:
                        dest['positive_in_rows'] += 1
                    context = by_context[(morph, task, provenance, primitive)]
                    if is_temporal:
                        context['temporal_excluded'] += candidates
                    else:
                        context['positive'] += pos
                        context['negative'] += candidates-pos-unknown
                        context['masked'] += unknown
        if report['complete'] and not pilot_rows:
            if len(matched) != report['expected_rows'][split]:
                raise ValueError('Incomplete candidate census '+split)
        summary['distinct_episodes'] = len({e for e, _ in matched})
        all_primitive_names = sorted(set(_known_primitives()) | set(by_primitive))
        result['splits'][split] = {
            'summary': dict(summary),
            'per_primitive': {k: dict(by_primitive[k]) for k in all_primitive_names},
            'by_morphology_task_provenance_primitive': [
                {'morphology': m, 'task': t, 'provenance': s, 'primitive': p, **dict(counts)}
                for (m,t,s,p), counts in sorted(by_context.items())]}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--g2', type=Path, required=True)
    parser.add_argument('--overlay', type=Path, required=True,
                        help='Authenticated output of bc_v5_integrity (no temporal audit needed for provisional counts)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot-rows', type=int, default=0,
                        help='Diagnostic only; samples per split, 0=complete overlay required')
    args = parser.parse_args(argv)
    destination, g2 = args.output.resolve(), args.g2.resolve()
    source_overlay = args.overlay.resolve()
    if (destination.exists() or destination == source_overlay or destination == g2 or
            destination.is_relative_to(g2)):
        parser.error('Output collision/unsafe destination')
    data = analyze(g2, args.overlay, pilot_rows=args.pilot_rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, sort_keys=True, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'report': str(destination),
                      'complete': data['complete'], 'ready_for_training': data['ready_for_training'],
                      'split_rows': {k: v['summary']['inspected_rows'] for k,v in data['splits'].items()},
                      'test_payload_access': 0}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
