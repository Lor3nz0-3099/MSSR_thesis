"""BC-v5 TRAIN-only sampler exposure simulation, without starting training.

Produces a compact JSON comparison: original uniform-row exposure versus
proposed primitive/morphology/task/provenance soft-balanced exposure. It never
reads TEST or changes G2, H1, H2, targets or checkpoints.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .action_set_training import _source
from .sidecar import SidecarDataset, _read_manifest
from .bc_v5_sampling import build_primitive_train_index, BalancedPrimitiveBatchSampler

SCHEMA = 'mssr.bc_v5.sampler_probe.v1'


def _count_by_dimension(index, draws):
    row_to_groups = defaultdict(set)
    for (morph, task, provenance, primitive), episode_rows in index.group_episode_rows.items():
        for group in episode_rows.values():
            for row in group:
                row_to_groups[row].add((morph, task, provenance, primitive))
    if set(row_to_groups) != set(range(index.sample_count)):
        raise ValueError('Missing TRAIN row/group membership')
    counts = {'primitive': Counter(), 'morphology': Counter(),
              'task': Counter(), 'provenance': Counter()}
    for row, weight in draws.items():
        if row not in row_to_groups or weight < 0:
            raise ValueError('Invalid sampled row')
        groups = row_to_groups[row]
        morph, task, provenance, _ = next(iter(groups))
        counts['morphology'][morph] += weight
        counts['task'][task] += weight
        counts['provenance'][provenance] += weight
        for p in {g[-1] for g in groups}:
            counts['primitive'][p] += weight
    return {k: dict(sorted(v.items())) for k, v in counts.items()}


def probe(g2_root, *, batch_size=64, balanced_quota=16, frequency_exponent=0.5,
          epochs=1, seed=20261112):
    if type(epochs) is not int or not 1 <= epochs <= 10:
        raise ValueError('Probe epochs must be between 1 and 10')
    root = Path(g2_root).resolve()
    raw = json.loads((root/'manifest.json').read_text())
    manifest = _read_manifest(root, raw['identity'])
    _source(manifest)
    with SidecarDataset(root, 'train', expected_identity=manifest['identity'], template_cache_size=8) as data:
        index = build_primitive_train_index(data)
    sampler = BalancedPrimitiveBatchSampler(index, batch_size=batch_size,
        balanced_quota=balanced_quota, frequency_exponent=frequency_exponent, seed=seed)
    observed = Counter({i: 1 for i in range(index.sample_count)})
    draws = Counter()
    for _ in range(epochs):
        for batch in sampler:
            draws.update(batch)
    original = _count_by_dimension(index, observed)
    sampled = _count_by_dimension(index, draws)
    return {'schema_version': SCHEMA,
            'g2_manifest_fingerprint': manifest['manifest_fingerprint'],
            'train_index_fingerprint': index.fingerprint,
            'test_payload_access': 0,
            'source_unchanged': True,
            'sampler_settings': {'batch_size': batch_size, 'balanced_quota': balanced_quota,
                                 'frequency_exponent': frequency_exponent,
                                 'seed': seed, 'epochs': epochs},
            'train_rows': index.sample_count,
            'sampled_slots': sum(draws.values()),
            'unique_rows_sampled': len(draws),
            'max_repeat_of_one_row': max(draws.values()),
            'original_one_pass': original,
            'simulated_exposure': sampled,
            'primitive_exposure': [
                {'primitive': key, 'source_rows': original['primitive'].get(key, 0),
                 'draws': sampled['primitive'].get(key, 0),
                 'repeat_factor': round(sampled['primitive'].get(key, 0) /
                                        original['primitive'][key], 4)}
                for key in sorted(original['primitive'])],
            'note': 'Diagnostic label sampling only. No BC training or generalization claim.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--g2', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--balanced-quota', type=int, default=16)
    parser.add_argument('--frequency-exponent', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20261112)
    args = parser.parse_args(argv)
    root, out = args.g2.resolve(), args.output.resolve()
    if out == root or out in root.parents or root in out.parents:
        parser.error('Output cannot overlap frozen G2')
    if out.exists() or not out.parent.is_dir():
        parser.error('Output exists or parent directory is missing')
    report = probe(root, batch_size=args.batch_size, balanced_quota=args.balanced_quota,
                   frequency_exponent=args.frequency_exponent, epochs=args.epochs, seed=args.seed)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({'output': str(out), 'train_rows': report['train_rows'],
                      'sampled_slots': report['sampled_slots'],
                      'unique_rows_sampled': report['unique_rows_sampled'],
                      'test_payload_access': 0}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
