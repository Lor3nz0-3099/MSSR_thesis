"""BC-v5 P1: immutable TRAIN/VALIDATION census over the authenticated G2 sidecar.

This is a *diagnostic*, not a dataset rewrite, sampler or training run.
No TEST payload is opened. Only BC-eligible records already admitted to G2
are counted. In particular, excluded button episodes are never recovered.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from .sidecar import SidecarDataset, _read_manifest
from .action_set_training import _source

SCHEMA = 'mssr.bc_v5.primitive_census.v1'
ZERO_TOL = 1e-9  # Numeric equality diagnostic, NOT a semantic stop detector.


def _primitive(target, structural=False):
    return ('structural:' + target.operation_id if structural else
            'resource:' + target.mode_id)


def _groups(audit):
    return (audit.morphology_stratum or '<missing>',
            audit.task_stratum or '<missing>',
            audit.provenance or '<missing>')


def _rounded(value):
    """Small deterministic target signature for temporal repetition diagnostics."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError('Non-finite physical action payload')
        return round(value, 4)
    if isinstance(value, (list, tuple)):
        return tuple(_rounded(x) for x in value)
    if hasattr(value, 'module_id'):
        return str(value.module_id)
    if hasattr(value, 'connector_id'):
        return (_rounded(value.module), str(value.connector_id))
    if hasattr(value, 'resource_id'):
        return (_rounded(value.module), str(value.resource_id))
    return value if isinstance(value, (str, int, bool, type(None))) else repr(value)


def _target_signature(supervision):
    entries = []
    for t in supervision.resource_targets:
        entries.append((_primitive(t), _rounded(t.resource), _rounded(t.payload)))
    for t in supervision.structural_targets:
        entries.append((_primitive(t, True), _rounded(t.initiator),
                        _rounded(t.connectors),
                        tuple((p.name, _rounded(p.value)) for p in t.parameters)))
    return tuple(sorted(entries, key=repr))


def _known_primitives():
    from .action_set_training import ROOT
    robot = json.loads((ROOT/'configs/bc_v1/smores_ep.robot.json').read_text())
    modes = {'resource:' + m['mode_id'] for mod in robot['module_types']
             for r in mod['resources'] for m in r['modes']}
    ops = {'structural:' + a['action_id'] for a in robot['actions']
           if a['scope'] != 'resource'}
    return sorted(modes | ops)


def _summarize(counter):
    return dict(sorted(counter.items()))


def analyze(g2_root, *, include_validation=True):
    """One streaming scan of frozen G2, never TEST; does not mutate the source."""
    g2_root = Path(g2_root).resolve()
    raw = json.loads((g2_root/'manifest.json').read_text())
    manifest = _read_manifest(g2_root, raw['identity'])
    _source(manifest)  # Enforce G2 TEST membership empty before opening payload.
    known = _known_primitives()
    result = {'schema_version': SCHEMA,
              'g2_manifest_fingerprint': manifest['manifest_fingerprint'],
              'g2_identity': manifest['identity'],
              'selection': 'G2 BC-eligible TRAIN/VALIDATION samples only',
              'test_payload_access': 0,
              'excluded_episodes_policy': 'unchanged; no RAW or excluded episodes opened',
              'known_primitives': known, 'splits': {}}
    for split in ('train', 'validation') if include_validation else ('train',):
        samples_by_morph, samples_by_task, samples_by_source = Counter(), Counter(), Counter()
        primitive_samples, primitive_instances = Counter(), Counter()
        cross = Counter(); episode_coverage = defaultdict(set)
        zero_numeric, nonzero_numeric = Counter(), Counter()
        action_sets, repeated = Counter(), Counter()
        episodes = Counter(); per_episode_primitive = defaultdict(Counter)
        former = {}
        with SidecarDataset(g2_root, split, expected_identity=manifest['identity'],
                            template_cache_size=8) as data:
            for idx in range(len(data)):
                # Read authenticated dynamic routing without cloning full H1/H2 tensors.
                _, _, dynamic = data.read_parts(idx)
                route = dynamic['routing']
                a = route['audit']; sup = route['physical_supervision']
                if a.episode_id not in data.episode_ids:
                    raise ValueError('Unrecognized G2 episode in '+split)
                morph, task, source = _groups(a)
                episodes[a.episode_id] += 1
                samples_by_morph[morph] += 1
                samples_by_task[task] += 1
                samples_by_source[source] += 1
                types = set()
                for target in sup.resource_targets:
                    if target.supervised is not True:
                        raise ValueError('Positive resource target not supervised at '+str(idx))
                    primitive = _primitive(target)
                    types.add(primitive); primitive_instances[primitive] += 1
                    if all(abs(float(v)) <= ZERO_TOL for v in target.payload):
                        zero_numeric[primitive] += 1
                    else:
                        nonzero_numeric[primitive] += 1
                for target in sup.structural_targets:
                    if target.supervised is not True:
                        raise ValueError('Positive structural target not supervised at '+str(idx))
                    primitive = _primitive(target, True)
                    types.add(primitive); primitive_instances[primitive] += 1
                if not types:
                    raise ValueError('G2 BC sample without positive supervision at '+str(idx))
                for primitive in types:
                    primitive_samples[primitive] += 1
                    cross[(morph, task, source, primitive)] += 1
                    episode_coverage[primitive].add(a.episode_id)
                    per_episode_primitive[a.episode_id][primitive] += 1
                action_sets[' + '.join(sorted(types))] += 1
                signature = _target_signature(sup)
                if former.get(a.episode_id) == signature:
                    repeated['consecutive_identical_target_signature'] += 1
                former[a.episode_id] = signature
            if sum(episodes.values()) != len(data):
                raise ValueError('G2 census coverage mismatch')
        observed = sorted(set(known) | set(primitive_samples))
        result['splits'][split] = {
            'samples': sum(episodes.values()), 'episodes': len(episodes),
            'samples_by_morphology': _summarize(samples_by_morph),
            'samples_by_task': _summarize(samples_by_task),
            'samples_by_provenance': _summarize(samples_by_source),
            'primitive_sample_rows': {k: primitive_samples[k] for k in observed},
            'primitive_target_instances': {k: primitive_instances[k] for k in observed},
            'primitive_episode_coverage': {k: len(episode_coverage[k]) for k in observed},
            'resource_payload_all_zero': _summarize(zero_numeric),
            'resource_payload_not_all_zero': _summarize(nonzero_numeric),
            'cross_morphology_task_provenance_primitive': [
                {'morphology': m, 'task': t, 'provenance': s, 'primitive': p, 'sample_rows': n}
                for (m,t,s,p),n in sorted(cross.items())],
            'top_joint_action_sets': [
                {'primitive_set': key, 'sample_rows': count}
                for key,count in action_sets.most_common(30)],
            'temporal_diagnostics': {
                'consecutive_identical_target_signature': repeated['consecutive_identical_target_signature'],
                'note': 'Repeated labels alone do not prove repeated observations or invalid data.'},
            'per_episode': [
                {'episode_id': eid, 'samples': episodes[eid],
                 'primitive_sample_rows': _summarize(per_episode_primitive[eid])}
                for eid in sorted(episodes)],
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--g2', type=Path, required=True, help='Complete, frozen local G2 sidecar')
    parser.add_argument('--output', type=Path, required=True, help='New JSON report, cannot overwrite')
    parser.add_argument('--train-only', action='store_true')
    args = parser.parse_args(argv)
    output, g2 = args.output.resolve(), args.g2.resolve()
    if output == g2 or g2 in output.parents or output in g2.parents:
        parser.error('Output must not overlap frozen G2 source')
    if output.exists():
        parser.error('Output exists; refusing overwrite')
    if not output.parent.is_dir():
        parser.error('Output parent must already exist')
    result = analyze(g2, include_validation=not args.train_only)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({'report': str(output),
                      'g2_manifest_fingerprint': result['g2_manifest_fingerprint'],
                      'splits': {k: {'samples': v['samples'], 'episodes': v['episodes']}
                                 for k, v in result['splits'].items()},
                      'test_payload_access': 0}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
