"""Explicit train-only F2 fitting from a completed, immutable physical G1 cache."""
from pathlib import Path
import os
import shutil
import tempfile

from . import dataset
from .bc_v1_contract import robot_spec_fingerprint
from .feature_schema import feature_schema_fingerprint
from .normalization import (
    NormalizationSample, fit_normalizers, normalizer_fingerprint,
    policy_fingerprint, read_normalizers, write_normalizers)
from .split import serialize_split_manifest

VERSION = 'mssr.bc_v1.normalization_fit_context.v1'


def fit_prepared_normalizers(cache_root, output_dir, *, split_manifest, spec,
                             schema, policy, expected_identity):
    """Read only train payloads; publish both F2 files in one directory transaction.

    F1 and asset expectations are supplied independently. No Canonical source,
    hardware adapter or runtime provider is required. Existing output is never
    overwritten, including byte-identical artifacts.
    """
    cache_root, output_dir = Path(cache_root).resolve(), Path(output_dir).resolve()
    if output_dir.exists(): raise ValueError('F2 output collision: directory already exists')
    if output_dir.is_relative_to(cache_root) or cache_root.is_relative_to(output_dir):
        raise ValueError('F2 output must be separate from physical G1 cache')
    manifest_path = cache_root/'manifest.json'
    original_manifest_hash = dataset.file_hash(manifest_path)
    manifest = dataset.read_cache_manifest(cache_root, expected_identity=expected_identity)
    serialize_split_manifest(split_manifest)
    source_root = Path(manifest['source_identity']['source_root']).resolve()
    if output_dir.is_relative_to(source_root) or source_root.is_relative_to(output_dir):
        raise ValueError('F2 output forbidden inside/over source dataset')
    actual = dict(split_fingerprint=split_manifest['manifest_fingerprint'],
        source_manifest_sha256=split_manifest['dataset_identity']['manifest_sha256'],
        robot_spec_fingerprint=robot_spec_fingerprint(spec),
        feature_schema_fingerprint=feature_schema_fingerprint(schema),
        normalization_policy_fingerprint=policy_fingerprint(policy, schema))
    if any(manifest['identity'].get(k) != value for k, value in actual.items()):
        raise ValueError('G1 incompatible with independent F1/spec/schema/normalization policy')
    source_identity = {k: v for k, v in manifest['source_identity'].items() if k != 'source_root'}
    if (source_identity != split_manifest['dataset_identity']
            or manifest['splits'] != split_manifest['splits']
            or manifest['excluded_episode_ids'] != split_manifest['excluded_episode_ids']):
        raise ValueError('G1 source/split membership differs from frozen F1')
    episodes = split_manifest['episode_metadata']
    if manifest['source_episode_hashes'] != {eid: row['sha256'] for eid, row in episodes.items()}:
        raise ValueError('G1 source episode inventory differs from frozen F1')
    count_field = split_manifest['strata_config']['eligible_count_field']
    if any(entry['samples'] != episodes[eid][count_field] for eid, entry in manifest['episodes'].items()):
        raise ValueError('G1 episode sample counts differ from complete F1 coverage')
    train = sorted(split_manifest['splits']['train'])
    signatures = {eid: dataset._signature(dataset._safe_file(cache_root,
        manifest['episodes'][eid]['cache_file'])) for eid in train}
    opened = []
    def stream(eid):
        opened.append(eid)
        for sample in dataset._load_episode(cache_root, eid, manifest['episodes'][eid]):
            yield NormalizationSample(eid, sample.audit.transition_index,
                                      sample.inputs.features, sample.supervision)
    stats = fit_normalizers({eid: lambda eid=eid: stream(eid) for eid in train},
        split_manifest, spec, schema, policy, expected_split_fingerprint=actual['split_fingerprint'])
    metadata = stats.metadata
    counts = [[eid, episodes[eid][count_field]] for eid in train]
    if opened != train or metadata['complete_train_coverage'] is not True or metadata['sample_counts'] != counts:
        raise ValueError('Incomplete/non-train normalizer coverage or sample counts')
    expected_metadata = dict(split_fingerprint=actual['split_fingerprint'],
        dataset_manifest_sha256=actual['source_manifest_sha256'],
        train_membership_fingerprint=dataset._hash(train),
        robot_spec_fingerprint=actual['robot_spec_fingerprint'],
        feature_schema_fingerprint=actual['feature_schema_fingerprint'],
        policy_fingerprint=actual['normalization_policy_fingerprint'], complete_train_coverage=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output_dir.name+'-', dir=output_dir.parent))
    reserved = published = False
    try:
        write_normalizers(stats, staging/'normalizers.json', source_root=source_root)
        read_normalizers(staging/'normalizers.json', expected_metadata=expected_metadata)
        context = dict(schema_version=VERSION, g1_cache_fingerprint=manifest['cache_fingerprint'],
            g1_manifest_sha256=original_manifest_hash, source_identity=manifest['source_identity'],
            split_fingerprint=actual['split_fingerprint'], robot_spec_fingerprint=actual['robot_spec_fingerprint'],
            feature_schema_fingerprint=actual['feature_schema_fingerprint'],
            normalization_policy_fingerprint=actual['normalization_policy_fingerprint'],
            train_episode_ids=train, train_episode_count=len(train), train_sample_count=sum(n for _, n in counts),
            sample_counts=counts, complete_train_coverage=True,
            normalizer_fingerprint=normalizer_fingerprint(stats),
            normalizer_sha256=dataset.file_hash(staging/'normalizers.json'),
            normalizer_expected_metadata=expected_metadata)
        (staging/'fit_context.json').write_text(dataset._json(context)+'\n')
        if dataset.file_hash(manifest_path) != original_manifest_hash or signatures != {
                eid: dataset._signature(dataset._safe_file(cache_root, manifest['episodes'][eid]['cache_file'])) for eid in train}:
            raise ValueError('Physical G1 cache changed during F2 fitting')
        dataset.read_cache_manifest(cache_root, expected_identity=expected_identity)
        # Reserve an absent destination without clobbering a concurrent publisher.
        # Only the complete directory is renamed over our owned empty reservation.
        output_dir.mkdir()
        reserved = True
        os.rename(staging, output_dir)
        published = True
        return context
    finally:
        if staging.exists(): shutil.rmtree(staging)
        if not published and reserved and output_dir.is_dir() and not any(output_dir.iterdir()):
            output_dir.rmdir()
