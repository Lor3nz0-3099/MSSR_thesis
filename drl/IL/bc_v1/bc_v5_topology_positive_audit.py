"""Read-only BC-v5 *preflight*, not a rebuilt G2 or training certification.

Inspect exact TRAIN/VALIDATION Canonical rows admitted by the Stage-5 overlay.
For every positive align_faces, require a certified singleton initiator in
current state. Any violation must be investigated, not silently discarded.
Other structural primitives are counted for orientation; complete exact
positive-retention must later be proven on rebuilt v5 candidate routing.
"""
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


def inspect(canonical, g2, overlay, *, max_episodes=0):
    canonical, g2 = Path(canonical).resolve(), Path(g2).resolve()
    raw_manifest = json.loads((g2 / 'manifest.json').read_text())
    g2_manifest = _read_manifest(g2, raw_manifest['identity'])
    _source(g2_manifest)  # refuses a G2 including TEST membership
    source_sha = hashlib.sha256((canonical / 'manifest.json').read_bytes()).hexdigest()
    if source_sha != g2_manifest['identity']['canonical_sha']:
        raise ValueError('Canonical manifest not identical to G2 source')
    overlay_report = read_report(overlay,
        expected_g2_fingerprint=g2_manifest['manifest_fingerprint'], ready=False)
    if not overlay_report['complete'] or overlay_report['canonical_manifest_sha256'] != source_sha:
        raise ValueError('Complete Stage-5 overlay bound to exact Canonical required')
    names = json.loads((canonical / 'manifest.json').read_text())['episodes']
    entries = {x['episode_id']: x for x in names}
    rows_by_episode = defaultdict(dict)
    for item in overlay_report['rows']:
        if item['split'] not in ('train', 'validation'):
            raise ValueError('Overlay includes a non-TRAIN/VALIDATION row')
        key = item['transition_index']
        if key in rows_by_episode[item['episode_id']]:
            raise ValueError('Duplicate Canonical key in Stage-5 overlay')
        rows_by_episode[item['episode_id']][key] = item['split']
    episode_ids = sorted(rows_by_episode)
    if max_episodes:
        episode_ids = episode_ids[:max_episodes]
    spec = load_robot_spec(Path(__file__).resolve().parents[3] / 'configs/bc_v1/smores_ep.robot.json')
    adapter = SMORESAdapter(spec)
    counts, examples = defaultdict(Counter), []
    for eid in episode_ids:
        entry = entries[eid]
        path = (canonical / entry['file']).resolve()
        if not path.is_relative_to(canonical):
            raise ValueError('Unsafe Canonical episode path')
        digest = hashlib.sha256()
        for idx, line in enumerate(path.open('rb')):
            digest.update(line)
            split = rows_by_episode[eid].get(idx)
            if split is None:
                continue
            counts[split]['rows'] += 1
            item = json.loads(line)
            if item['supervision']['valid_for_behavior_cloning'] is not True:
                raise ValueError('Non-BC-valid row in authenticated G2')
            structural = [(mid, decision['structural']['primitive'])
                          for mid, decision in item['action_t']['modules'].items()
                          if item['supervision']['module_resource_masks'][mid]['structural']
                          and decision['structural']['commanded']]
            if not structural:
                continue
            graph = adapter.encode_state(item['graph_t'])
            ports = {p.ref: p for p in graph.connectors}
            _, attached, singleton, _ = topology_index(graph, ports)
            for initiator, name in structural:
                counts[split][f'positive:{name}'] += 1
                if name != 'align_faces':
                    continue
                if ModuleRef(initiator) in singleton:
                    counts[split]['align_singleton_pass'] += 1
                else:
                    counts[split]['align_singleton_violation'] += 1
                    if len(examples) < 30:
                        examples.append({'episode_id': eid, 'transition_index': idx,
                                         'split': split, 'initiator': initiator,
                                         'connected': ModuleRef(initiator) in attached,
                                         'morphology': item['graph_t'].get('morphology')})
        if digest.hexdigest() != entry['sha256']:
            raise ValueError('Canonical episode SHA changed during audit: ' + eid)
    all_counts = {s:dict(sorted(c.items())) for s,c in sorted(counts.items())}
    return {'schema_version':'mssr.bc_v5.topology_positive_preflight.v1',
            'source_manifest_sha256':source_sha,
            'g2_manifest_fingerprint':g2_manifest['manifest_fingerprint'],
            'complete':not bool(max_episodes),
            'candidate_generator_applied_to_g2':False,
            'exact_positive_retention_certified':False,
            'test_payload_access':0,
            'counts':all_counts,
            'align_violations':sum(c.get('align_singleton_violation',0) for c in counts.values()),
            'violation_examples':examples}


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--canonical', required=True)
    cli.add_argument('--g2', required=True)
    cli.add_argument('--overlay', required=True)
    cli.add_argument('--output', required=True)
    cli.add_argument('--max-episodes', type=int, default=0)
    args=cli.parse_args()
    if args.max_episodes < 0:
        cli.error('--max-episodes must be nonnegative')
    result=inspect(args.canonical,args.g2,args.overlay,max_episodes=args.max_episodes)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'complete':result['complete'],
                      'align_violations':result['align_violations'],
                      'counts':result['counts'],
                      'report':str(output),
                      'test_payload_access':0}, sort_keys=True))
    # Violations are reported, not silently hidden. A separate approval step
    # must inspect them before this candidate generator can replace G2/live.


if __name__=='__main__':
    main()
