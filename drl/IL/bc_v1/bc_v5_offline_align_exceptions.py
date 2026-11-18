"""Stage 10: auditable, TRAIN/VAL-only exceptions for stale-edge align labels.

Do NOT use this module to make live physical decisions. A recorded physical edge
still blocks the Stage-9 runtime alignment gate until a fresh joint check passes.

A past ``undock`` alone is insufficient. The *offline exception registry* checks
all recorded attached faces, the executor's detached override, and a one-bucket
later physically isolated state. The latter is used ONLY to review a historical
label, not as an H1/H2 feature or an online candidate-generation rule.

CAVEAT: optional offline candidate retention is label/row-specific, and hence
creates a controlled train-vs-live candidate-distribution difference. It must
be measured in a v5 positive-retention audit and an ablation, NOT silently
turned into an online candidate rule or a training-readiness certificate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from .bc_v1_contract import ModuleRef, StateGraph, StructuralTarget
from .bc_v5_topology_candidates import enumerate_candidates_v5, topology_index
from .candidates import CandidateTables, StructuralCandidate, enumerate_candidates


SCHEMA = 'mssr.bc_v5.offline_align_exceptions.v1'
# Source-specific expected positive exceptions, from FULL Stage-7/8 audits.
EXPECTED_SPLITS = {'train': 34, 'validation': 13}


def _digest(payload):
    data = json.dumps(payload, sort_keys=True, ensure_ascii=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _key(item):
    split = item.get('split')
    eid = item.get('episode_id')
    idx = item.get('transition_index')
    mid = item.get('initiator')
    if (split not in ('train', 'validation') or type(eid) is not str or not eid
            or type(idx) is not int or idx < 0 or type(mid) is not str or not mid):
        raise ValueError('Invalid split/episode/transition/initiator for align exception')
    return split, eid, idx, mid


def _review(violation):
    """Return per-link evidence, or a precise refusal reason; never infer proof."""
    if violation.get('status') != 'physical_edge_but_executor_override_detached':
        return None, 'executor_has_not_marked_module_detached'
    links = violation.get('physical_links')
    if not isinstance(links, list) or not links:
        return None, 'no_recorded_physical_links'
    index = violation['transition_index']
    corroborated = []
    seen = set()
    for link in links:
        own, peer, peer_face = (link.get('own_face'), link.get('peer'),
                               link.get('peer_face'))
        if (not all(type(x) is str and x for x in (own, peer, peer_face))
                or own in seen):
            return None, 'invalid_or_duplicated_connected_face'
        seen.add(own)
        event = link.get('previous_event_on_face')
        if not isinstance(event, dict) or event.get('primitive') != 'undock':
            return None, 'not_every_connected_face_has_prior_undock'
        previous = event.get('transition_index')
        if type(previous) is not int or index - previous not in (1, 2):
            return None, 'undock_not_one_or_two_prior_buckets'
        if event.get('supervised') is not True:
            return None, 'prior_undock_not_supervised'
        corroborated.append({'own_face': own, 'peer': peer,
                             'peer_face': peer_face, 'undock_distance': index-previous})
    following = violation.get('next_physical_change')
    if (not isinstance(following, dict)
            or following.get('subsequent_buckets') != 1
            or following.get('transition_index') != index + 1
            or following.get('all_previous_links_gone') is not True
            or following.get('fully_isolated') is not True):
        return None, 'no_next_bucket_confirmed_isolation'
    return sorted(corroborated, key=lambda x: x['own_face']), None


def build_registry(history, preflight):
    """Build exact row-scoped label exceptions; never train from this alone."""
    if (history.get('schema_version') != 'mssr.bc_v5.align_joint_history.v1'
            or history.get('complete') is not True
            or history.get('no_test_payload_access') is not True):
        raise ValueError('Full TRAIN/VAL Stage-8 audit required')
    if (preflight.get('schema_version') != 'mssr.bc_v5.topology_positive_preflight.v1'
            or preflight.get('complete') is not True
            or preflight.get('test_payload_access') != 0):
        raise ValueError('Full TRAIN/VAL Stage-7 preflight required')
    canonical = preflight.get('source_manifest_sha256')
    g2 = preflight.get('g2_manifest_fingerprint')
    if (not isinstance(canonical, str) or not canonical
            or not isinstance(g2, str) or not g2
            or history.get('canonical_sha256') != canonical
            or history.get('g2_manifest_fingerprint') != g2):
        raise ValueError('Canonical/G2 source identity mismatch')
    violations = history.get('violations')
    if not isinstance(violations, list):
        raise ValueError('Stage-8 violations list missing')
    if (len(violations) != preflight.get('align_violations')
            or len(violations) != history.get('summary', {}).get('total_violations')):
        raise ValueError('Stage-7/8 violation count mismatch')
    seen = set()
    admitted, refused = [], []
    input_counts = Counter()
    for entry in violations:
        key = _key(entry)
        if key in seen:
            raise ValueError('Duplicate exact align exception identity')
        seen.add(key)
        input_counts[key[0]] += 1
        evidence, why = _review(entry)
        common = {'split': key[0], 'episode_id': key[1],
                  'transition_index': key[2], 'initiator': key[3]}
        if why is not None:
            refused.append({**common, 'reason': why})
        else:
            admitted.append({**common, 'prior_undock_links': evidence,
                             'later_physical_isolation_for_offline_audit_only': True})
    for split, expected in EXPECTED_SPLITS.items():
        if (input_counts[split] != expected
                or preflight.get('counts', {}).get(split, {}).get('align_singleton_violation') != expected):
            raise ValueError('Unexpected source-specific Stage-7/8 split totals')
    admitted.sort(key=lambda x: _key(x))
    refused.sort(key=lambda x: _key(x))
    result = {
        'schema_version': SCHEMA,
        'canonical_sha256': canonical,
        'g2_manifest_fingerprint': g2,
        'complete': not refused and len(admitted) == sum(EXPECTED_SPLITS.values()),
        'exceptions': admitted,
        'needs_review': refused,
        'counts': {'input_violations': len(violations),
                   'accepted': dict(Counter(x['split'] for x in admitted)),
                   'refused': dict(Counter(x['split'] for x in refused))},
        'exception_scope': 'OFFLINE TRAIN/VALIDATION positive label only',
        'runtime_exception': False,
        'dataset_or_g2_mutated': False,
        'temporal_authenticated': False,
        'ready_for_training': False,
        'test_payload_access': 0,
        'caveat': ('A later observation is used only to audit these historical labels. '
                   'No new online rule or causal physical completion is inferred. '
                   'Offline row-level fallback restores all legacy align choices, not just '
                   'the supervised positive, but row selection is still label-dependent. '
                   'This train/inference mismatch requires an ablation.'),
    }
    result['fingerprint'] = _digest(result)
    return result


def read_registry(path, *, canonical_sha256=None, g2_manifest_fingerprint=None):
    data = _read_json(path)
    fingerprint = data.pop('fingerprint', None)
    if fingerprint != _digest(data) or data.get('schema_version') != SCHEMA:
        raise ValueError('Offline exception registry is corrupted or incompatible')
    if data.get('complete') is not True or data.get('needs_review'):
        raise ValueError('Offline align exception audit requires manual review')
    if (data.get('runtime_exception') is not False or data.get('ready_for_training') is not False
            or data.get('test_payload_access') != 0):
        raise ValueError('Exception registry cannot authorize live runtime or training')
    if (canonical_sha256 is not None and data.get('canonical_sha256') != canonical_sha256):
        raise ValueError('Canonical identity mismatch')
    if (g2_manifest_fingerprint is not None
            and data.get('g2_manifest_fingerprint') != g2_manifest_fingerprint):
        raise ValueError('G2 identity mismatch')
    keys = [_key(x) for x in data['exceptions']]
    if len(keys) != len(set(keys)) or len(keys) != sum(EXPECTED_SPLITS.values()):
        raise ValueError('Exception registry identities invalid')
    data['fingerprint'] = fingerprint
    return data


def offline_supervised_candidates_for_row(graph, spec, schema, *, split,
                                          episode_id, transition_index,
                                          targets, exception_registry):
    """EXPERIMENTAL offline routing, never imported by the live controller.

    Exact audited exceptions restore the FULL historical BC-v4 align domain
    for that row (rather than injecting only the positive action, which would
    leak the correct answer through candidate admission). The other structural
    and resource domains are unchanged; no new pair is invented. This supervised
    exception deliberately uses label metadata and MUST NOT become a runtime
    rule. It does not authorize training without a separate temporal audit.
    """
    if split not in ('train', 'validation'):
        raise ValueError('Offline exception routing is forbidden for TEST/live/inference')
    if not isinstance(graph, StateGraph):
        raise TypeError('Expected StateGraph')
    if exception_registry.get('schema_version') != SCHEMA or exception_registry.get('complete') is not True:
        raise ValueError('Validated, full offline registry required')
    # Fail closed against callers passing an unverified dict instead of a
    # content-validated Stage-10 registry.
    fingerprint = exception_registry.get('fingerprint')
    if fingerprint != _digest({k: v for k, v in exception_registry.items() if k != 'fingerprint'}):
        raise ValueError('Invalid offline exception registry fingerprint')
    applicable = {x['initiator'] for x in exception_registry['exceptions']
                  if (x['split'], x['episode_id'], x['transition_index']) ==
                     (split, episode_id, transition_index)}
    if len(applicable) != sum(
            (x['split'], x['episode_id'], x['transition_index']) ==
            (split, episode_id, transition_index)
            for x in exception_registry['exceptions']):
        raise ValueError('Duplicated initiator in offline exception row')
    v5 = enumerate_candidates_v5(graph, spec, schema)
    if not applicable:
        return v5
    old = enumerate_candidates(graph, spec, schema)
    ports = {port.ref: port for port in graph.connectors}
    components, _, isolated, _ = topology_index(graph, ports)
    rows = list(v5.structural_candidates)
    for mid in applicable:
        if ModuleRef(mid) in isolated:
            raise ValueError('Exception no longer needed: initiator is physically isolated')
        matching_targets = [t for t in targets
                            if isinstance(t, StructuralTarget)
                            and t.operation_id == 'align_faces'
                            and t.initiator == ModuleRef(mid)]
        if len(matching_targets) != 1:
            raise ValueError('Exact positive align target required for each exception')
        target = matching_targets[0]
        if (len(target.connectors) != 2
                or target.connectors[0].module != target.initiator
                or target.connectors[1].module == target.initiator
                or components[target.initiator] == components[target.connectors[1].module]):
            raise ValueError('Offline align must target a distinct physical component')
        # This is ONLY a validity assertion on an existing legacy choice.
        # Target labels are NEVER used to decide which candidates to add.
        historical = [(cand, binding) for cand in old.structural_candidates
                      if cand.action.action_id == 'align_faces'
                      for binding in cand.bindings
                      if binding.initiator == target.initiator
                      and binding.connectors == target.connectors]
        if len(historical) != 1:
            raise ValueError('No unique historical BC-v4 positive align binding')
    # For an exceptional historical ROW, restore the FULL legacy align domain,
    # rather than injecting ONLY the correct target, which would leak its
    # identity into H2 through the existence of a unique candidate.
    # Other 10 primitive domains remain precisely the BC-v5 topology result.
    for cand in old.structural_candidates:
        if cand.action.action_id != 'align_faces':
            continue
        indices = [i for i, row in enumerate(rows)
                   if row.action.action_id == 'align_faces'
                   and row.endpoints == cand.endpoints]
        if len(indices) > 1:
            raise ValueError('Duplicate v5 structural candidate pair')
        if indices:
            i = indices[0]
            rows[i] = replace(rows[i], bindings=rows[i].bindings | cand.bindings)
        else:
            rows.append(cand)
    return CandidateTables(v5.resource_candidates, tuple(rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', required=True)
    parser.add_argument('--preflight', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    result = build_registry(_read_json(args.history), _read_json(args.preflight))
    out = Path(args.output)
    if out.exists():
        parser.error('Refusing to overwrite existing Stage-10 report')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({'complete': result['complete'], 'counts': result['counts'],
                      'needs_review': len(result['needs_review']), 'output': str(out),
                      'ready_for_training': False, 'runtime_exception': False,
                      'test_payload_access': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
