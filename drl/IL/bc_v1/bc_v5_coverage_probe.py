"""Fast, bounded TRAIN-only S1 supervision coverage probe (never TEST)."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .sidecar import SidecarDataset, _read_manifest
from .action_set_training import _source
from .bc_v5_supervision import membership_with_evidence
from .bc_v5_membership import POSITIVE, NEGATIVE, UNKNOWN


def probe(g2, *, limit=320):
    if type(limit) is not int or limit < 1:
        raise ValueError('Invalid sample limit')
    root = Path(g2).resolve()
    raw = json.loads((root / 'manifest.json').read_text())
    manifest = _read_manifest(root, raw['identity'])
    _source(manifest)  # Enforce zero TEST membership and full manifest identity.
    counts = Counter()
    by_evidence = Counter()
    with SidecarDataset(root, 'train', expected_identity=manifest['identity'],
                        template_cache_size=4) as ds:
        if not len(ds):
            raise ValueError('Empty TRAIN')
        # Spread the audit over TRAIN rather than biasing toward early episodes.
        indices = sorted({min(len(ds) - 1, i * len(ds) // min(limit, len(ds)))
                          for i in range(min(limit, len(ds)))})
        for idx in indices:
            # G2 stores invariant routing (resource_refs, resource_candidates,
            # structural_bindings) in the shared template. The dynamic record
            # alone is NOT a complete route. Shallow merge without cloning
            # large input tensors; read_parts authenticated both components.
            _, template, dynamic = ds.read_parts(idx)
            route = {**template['routing'], **dynamic['routing']}
            status, proofs = membership_with_evidence(route)
            counts['samples'] += 1
            counts['graphs_with_certified_negative'] += int((status == NEGATIVE).any())
            counts['positive'] += int((status == POSITIVE).sum())
            counts['certified_negative'] += int((status == NEGATIVE).sum())
            counts['unknown'] += int((status == UNKNOWN).sum())
            by_evidence.update(p.evidence_kind for p in proofs)
    return {'schema_version': 'mssr.bc_v5.s1_coverage_probe.v1',
            'g2_manifest_fingerprint': manifest['manifest_fingerprint'],
            'split': 'train', 'sampled_train_rows': counts['samples'],
            'total_train_rows': sum(e['samples'] for e in manifest['inventory']['train']),
            'membership': dict(counts), 'negative_evidence_kinds': dict(by_evidence),
            'note': 'Bounded diagnostic, NOT complete TRAIN counts or certification of full coverage',
            'test_payload_access': 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--g2', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=320)
    args = parser.parse_args(argv)
    path, root = args.output.resolve(), args.g2.resolve()
    if path.exists() or path == root or root in path.parents or path in root.parents:
        parser.error('Output exists or overlaps frozen G2 source')
    if not path.parent.is_dir():
        parser.error('Output directory does not exist')
    result = probe(root, limit=args.limit)
    path.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
