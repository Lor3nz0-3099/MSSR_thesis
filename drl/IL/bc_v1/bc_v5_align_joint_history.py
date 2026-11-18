"""Read-only Stage-8 audit for BC-v5 align_faces vs recorded physical joints.

Important: a current_connection is a RECORDED physical docking edge; it is not
an instantaneous USD query at the moment of command execution. Executor face
availability is separate inferred evidence, not proof that a FixedJoint is gone.
No G2 rebuild, supervision mutation, test payload access or temporal certification.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from .bc_v1_contract import ModuleRef, load_robot_spec
from .bc_v5_integrity import read_report
from .bc_v5_topology_candidates import topology_index
from .action_set_training import _source
from .sidecar import _read_manifest
from .smores_adapter import SMORESAdapter


SCHEMA = 'mssr.bc_v5.align_joint_history.v1'


def _links(graph):
    """Preserve physical edge identity, including both concrete connector faces."""
    by_module = defaultdict(list)
    for edge in graph.get('edges', []):
        if edge.get('relation') != 'current_connection':
            continue
        a, b = edge.get('module_a_id'), edge.get('module_b_id')
        fa, fb = edge.get('face_a'), edge.get('face_b')
        if not all(isinstance(x, str) and x for x in (a, b, fa, fb)):
            raise ValueError('Malformed physical docking edge')
        by_module[a].append({'own_face': fa, 'peer': b, 'peer_face': fb})
        by_module[b].append({'own_face': fb, 'peer': a, 'peer_face': fa})
    return {m: sorted(v, key=lambda x: (x['own_face'], x['peer'], x['peer_face']))
            for m, v in by_module.items()}


def _structural_actions(row):
    result = []
    masks = row['supervision']['module_resource_masks']
    modules = row['action_t']['modules']
    for mid, action in modules.items():
        structural = action['structural']
        if not structural['commanded']:
            continue
        result.append({'module': mid, 'primitive': structural['primitive'],
                       'parameters': structural.get('parameters') or {},
                       'supervised': masks[mid]['structural'] is True})
    return result


def _face_state(graph, module):
    for node in graph.get('nodes', []):
        if node.get('module_id') == module:
            return (node.get('control_state') or {}).get('faces') or {}
    raise ValueError('Unknown initiator in graph: ' + module)


def _status(physical_links, face_states):
    """Classify evidence, never assert that a USD FixedJoint was removed."""
    if not physical_links:
        return 'no_current_connection_edge'
    if not face_states:
        return 'executor_evidence_missing'
    values = []
    for link in physical_links:
        state = face_states.get(link['own_face'])
        if not isinstance(state, dict) or type(state.get('connected_for_executor')) is not bool:
            return 'executor_evidence_missing'
        values.append(state['connected_for_executor'])
    # A new DOCK may be known only to the executor (no physical edge yet).
    # Never call the initiator detached while ANY other connector is logically attached.
    if any(state.get('connected_for_executor') is True
           for state in face_states.values() if isinstance(state, dict)):
        return 'also_connected_for_executor'
    if any(state.get('connected_for_executor') is not False
           for state in face_states.values() if isinstance(state, dict)):
        return 'executor_evidence_missing'
    return 'physical_edge_but_executor_override_detached'


def _event_faces(action, modules):
    """Map undock/dock faces to both endpoints, using participant metadata."""
    mid = action['module']
    params = action['parameters']
    if not isinstance(params.get('face_a'), str):
        return []
    result = [(mid, params['face_a'])]
    for peer_id, peer in modules.items():
        for item in peer.get('participates_in') or []:
            if (item.get('primitive') == action['primitive']
                    and item.get('initiator_module_id') == mid
                    and item.get('own_face') == params.get('face_b')):
                result.append((peer_id, params['face_b']))
    return result


def _row_snapshot(row, idx):
    graph = row['graph_t']
    return {'index': idx, 'stamp': (row.get('provenance') or {}).get('physical_stamp', graph.get('stamp')),
            'links': _links(graph),
            'actions': _structural_actions(row),
            'face_states': {node['module_id']:(node.get('control_state') or {}).get('faces') or {}
                            for node in graph.get('nodes', [])},
            'modules': row['action_t']['modules'],
            'morphology': graph.get('morphology'),
            'source_records': (row.get('provenance') or {}).get('source_records') or []}


def _first_link_disappearance(sequence, index, initiator, baseline, lookahead):
    """Future snapshots are diagnostic-only; no future state enters H1/H2."""
    wanted = {(x['own_face'], x['peer'], x['peer_face']) for x in baseline}
    for pos in range(index+1, min(len(sequence), index+lookahead+1)):
        current = {(x['own_face'], x['peer'], x['peer_face'])
                   for x in sequence[pos]['links'].get(initiator, [])}
        if not wanted <= current:
            return {'subsequent_buckets': pos-index,
                    'transition_index': sequence[pos]['index'],
                    'stamp': sequence[pos]['stamp'],
                    'all_previous_links_gone': not bool(wanted & current),
                    'fully_isolated': not bool(current)}
    return None


def analyze_sequence(sequence, eligible_by_index, adapter, *, lookahead):
    """Audit one TRAIN/VAL episode, without touching its original rows."""
    last_event = {}
    all_positives = Counter()
    violations = []
    for pos, s in enumerate(sequence):
        idx = s['index']
        split = eligible_by_index.get(idx)
        if split is not None:
            for act in s['actions']:
                if act['primitive'] != 'align_faces' or not act['supervised']:
                    continue
                all_positives[split] += 1
                mid = act['module']
                # Stage-7's exact rule: singleton in the ADAPTER graph,
                # including uncertain connector connected_to/status evidence.
                raw = s['_graph']
                graph = adapter.encode_state(raw)
                ports = {p.ref: p for p in graph.connectors}
                _, _, singletons, _ = topology_index(graph, ports)
                if ModuleRef(mid) in singletons:
                    continue
                physical = s['links'].get(mid, [])
                faces = s['face_states'].get(mid, {})
                state = _status(physical, faces)
                attached = []
                for link in physical:
                    f = link['own_face']
                    prior = last_event.get((mid, f))
                    attached.append({**link, 'executor_face': faces.get(f),
                                     'previous_event_on_face': prior})
                violations.append({'split': split, 'episode_id': s['_episode_id'],
                                   'transition_index': idx, 'physical_stamp': s['stamp'],
                                   'initiator': mid, 'morphology': s['morphology'],
                                   'status': state, 'physical_links': attached,
                                   'same_bucket_undocks': [a for a in s['actions'] if a['primitive']=='undock'],
                                   'align_parameters': act['parameters'],
                                   'source_records': s['source_records'][:6],
                                   'next_physical_change': _first_link_disappearance(
                                       sequence, pos, mid, physical, lookahead)})
        # Record dispatched structural commands AFTER evaluating this policy
        # observation: no leaking a_t into the inferred state at t.
        for act in s['actions']:
            if act['primitive'] not in ('dock', 'undock'):
                continue
            event = {'primitive': act['primitive'], 'transition_index': idx,
                     'physical_stamp': s['stamp'], 'supervised': act['supervised'],
                     'initiator': act['module']}
            for key in _event_faces(act, s['modules']):
                last_event[key] = event
    return all_positives, violations


def inspect(canonical, g2, overlay, preflight, *, lookahead=30):
    canonical, g2 = Path(canonical).resolve(), Path(g2).resolve()
    if type(lookahead) is not int or lookahead < 1:
        raise ValueError('lookahead must be a positive integer')
    g2_raw = json.loads((g2 / 'manifest.json').read_text())
    g2_manifest = _read_manifest(g2, g2_raw['identity'])
    _source(g2_manifest)
    canonical_sha = hashlib.sha256((canonical / 'manifest.json').read_bytes()).hexdigest()
    if canonical_sha != g2_manifest['identity']['canonical_sha']:
        raise ValueError('Canonical/G2 identity mismatch')
    overlay_report = read_report(overlay, expected_g2_fingerprint=g2_manifest['manifest_fingerprint'], ready=False)
    if not overlay_report['complete'] or overlay_report['canonical_manifest_sha256'] != canonical_sha:
        raise ValueError('Full authenticated Stage-5 overlay required')
    pre = json.loads(Path(preflight).read_text())
    if (pre.get('schema_version') != 'mssr.bc_v5.topology_positive_preflight.v1'
            or pre.get('complete') is not True
            or pre.get('source_manifest_sha256') != canonical_sha
            or pre.get('g2_manifest_fingerprint') != g2_manifest['manifest_fingerprint']):
        raise ValueError('Stage-7 preflight identity/fullness mismatch')
    m = json.loads((canonical / 'manifest.json').read_text())
    entries = {x['episode_id']: x for x in m['episodes']}
    wanted = defaultdict(dict)
    for entry in overlay_report['rows']:
        split = entry['split']
        if split not in ('train', 'validation'):
            raise ValueError('Unexpected split (TEST must not be inspected)')
        key = entry['transition_index']
        eid = entry['episode_id']
        if key in wanted[eid]:
            raise ValueError('Duplicate admitted identity')
        wanted[eid][key] = split
    spec = load_robot_spec(Path(__file__).resolve().parents[3] / 'configs/bc_v1/smores_ep.robot.json')
    adapter = SMORESAdapter(spec)
    positives, violations = Counter(), []
    for eid in sorted(wanted):
        entry = entries[eid]
        path = (canonical / entry['file']).resolve()
        if not path.is_relative_to(canonical) or not path.is_file():
            raise ValueError('Unsafe or missing Canonical episode')
        digest = hashlib.sha256()
        sequence = []
        with path.open('rb') as f:
            for idx, line in enumerate(f):
                digest.update(line)
                row = json.loads(line)
                s = _row_snapshot(row, idx)
                # Keep full graph for Stage-7 identity on relevant rows only.
                if idx in wanted[eid] and any(
                    a['primitive'] == 'align_faces' and a['supervised'] for a in s['actions']):
                    s['_graph'] = row['graph_t']
                s['_episode_id'] = eid
                sequence.append(s)
        if digest.hexdigest() != entry['sha256']:
            raise ValueError('Canonical episode SHA mismatch: '+eid)
        if not set(wanted[eid]) <= {x['index'] for x in sequence}:
            raise ValueError('Missing G2 row identities')
        n, v = analyze_sequence(sequence, wanted[eid], adapter, lookahead=lookahead)
        positives.update(n)
        violations.extend(v)
    for split in ('train', 'validation'):
        prev = pre['counts'][split]
        if (positives[split] != prev['positive:align_faces']
                or sum(x['split']==split for x in violations) != prev['align_singleton_violation']):
            raise ValueError(f'Stage-7 positive/violation mismatch in {split}')
    by_status = {split: dict(Counter(x['status'] for x in violations if x['split']==split))
                 for split in ('train', 'validation')}
    with_prev = sum(any(link['previous_event_on_face'] is not None
                        and link['previous_event_on_face']['primitive']=='undock'
                        for link in x['physical_links']) for x in violations)
    return {'schema_version': SCHEMA, 'complete': True,
            'canonical_sha256': canonical_sha,
            'g2_manifest_fingerprint': g2_manifest['manifest_fingerprint'],
            'lookahead_buckets': lookahead,
            'no_test_payload_access': True,
            'note': ('Diagnostic only; executor override is NOT proof of USD FixedJoint removal. '
                     'Future graph is used only for audit, never model input; '
                     'does NOT authenticate the overall temporal audit or enable training.'),
            'positive_align_faces': dict(positives),
            'violations': violations,
            'summary': {'total_violations': len(violations),
                        'by_executor_status': by_status,
                        'with_prior_undock_on_connected_face': with_prev,
                        'with_physical_change_within_lookahead': sum(
                            x['next_physical_change'] is not None for x in violations)}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ('canonical', 'g2', 'overlay', 'preflight', 'output'):
        p.add_argument('--'+arg, required=True)
    p.add_argument('--lookahead', type=int, default=30)
    args = p.parse_args()
    result = inspect(args.canonical, args.g2, args.overlay, args.preflight, lookahead=args.lookahead)
    out = Path(args.output)
    if out.exists():
        raise SystemExit('Refusing to overwrite existing audit: '+str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False)+'\n')
    print(json.dumps({'summary': result['summary'], 'report': str(out),
                      'no_test_payload_access': True}, sort_keys=True))
    for x in result['violations'][:12]:
        print(f"{x['split']:10} {x['episode_id']}:{x['transition_index']} "
              f"{x['initiator']:12} {x['status']:42} "
              f"prior_undock={any(y['previous_event_on_face'] and y['previous_event_on_face']['primitive']=='undock' for y in x['physical_links'])} "
              f"next_change={x['next_physical_change'] is not None}")


if __name__ == '__main__':
    main()
