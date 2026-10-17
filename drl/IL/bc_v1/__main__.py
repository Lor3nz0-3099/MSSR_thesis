"""Side-effect-free help; hardware boundary loaded only for explicit prepare."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description='BC-v1 offline physical dataset preparation')
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--source', type=Path, required=True)
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--config', type=Path, default=Path('configs/bc_v1/training.json'))
    prepare.add_argument('--prepare-config', type=Path, default=Path('configs/bc_v1/prepare.json'))
    fit = sub.add_parser('fit-normalizers', help='Fit train-only F2 from a complete physical G1 cache')
    fit.add_argument('--cache', type=Path, required=True)
    fit.add_argument('--output', type=Path, required=True, help='F2 directory containing both JSON artifacts')
    fit.add_argument('--config', type=Path, default=Path('configs/bc_v1/training.json'))
    fit.add_argument('--prepare-config', type=Path, default=Path('configs/bc_v1/prepare.json'))
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        run_prepare(args)
    else:
        run_fit_normalizers(args)


def encode_smores_labels(row, adapter, root):
    from . import smores_adapter as sm
    _, resources, structural = sm._count_canonical_labels(row['action_t'], row['supervision'])
    try:
        bundle = adapter.encode_labels(row['action_t'], row['supervision'], ())
    except sm.LabelEvidenceRequired:
        bundle = adapter.encode_labels(row['action_t'], row['supervision'], sm.read_label_evidence(row, root))
    sm._check_generic_targets(bundle, resources, structural, row['graph_t'], adapter)
    return bundle


def _read_indexed_source_records(path, indices):
    """Reuse the process-local source offset index; decode only exact rows."""
    from . import smores_adapter as sm
    if any(type(index) is not int or index < 1 for index in indices):
        raise ValueError('Source metadata record must be a positive one-based index')
    indexed = sm._source_index(path)
    with path.open('rb') as stream:
        for index in indices:
            if index > len(indexed.offsets): raise ValueError('Source metadata record out of bounds')
            stream.seek(indexed.offsets[index-1])
            yield index, json.loads(stream.readline())


def compose_runtime_providers(root, dataset_identity, assets, episode_metadata):
    """Load serializable reviewed assets; inject callables only in memory.

    Topology/frame assets are required; temporal audit is optional. Configured
    audits remain strict. No dataset scan guesses run links or classifications.
    """
    from collections import Counter
    from .smores_disconnect_evidence import (
        HistoricalSmoresDisconnectProvider, read_causal_repairs_catalog, read_disconnect_source_refs)
    from .smores_source_frames import SmoresSourceFrameProvider, read_frame_source_context
    from .temporal_supervision import read_temporal_audit
    root = Path(root).resolve()
    def artifact(relative):
        path = root/relative
        if not path.is_file(): raise ValueError(f'Required certified provider artifact missing: {path}')
        return path
    topology_policy_path = artifact(assets['topology']['policy'])
    catalog_path = artifact(assets['topology']['source_catalog'])
    frame_policy_path = artifact(assets['source_frames']['policy'])
    temporal_setting = assets.get('temporal_audit')
    temporal_path = None if temporal_setting is None else artifact(temporal_setting)
    topology_policy = json.loads(topology_policy_path.read_text())
    catalog = read_causal_repairs_catalog(catalog_path, dataset_identity)
    frame_policy = json.loads(frame_policy_path.read_text())
    topology_provider = HistoricalSmoresDisconnectProvider(root, topology_policy)
    frame_provider = SmoresSourceFrameProvider(root, frame_policy)
    def topology(provenance, published_graph):
        refs = read_disconnect_source_refs(provenance, repo_root=root, source_catalog=catalog)
        return None if refs is None else topology_provider.for_bucket(refs, published_graph)
    def frame_reader(provenance, metadata):
        return read_frame_source_context(provenance, metadata, repo_root=root, policy=frame_policy,
            record_reader=_read_indexed_source_records)
    # Artifact integrity/enum validation uses the existing codec. Counts/ranges
    # are checked independently against frozen F1; exact BC keys at prepare.
    temporal = None
    if temporal_path is not None:
        document = json.loads(temporal_path.read_text())
        keys = {(x['episode_id'], x['transition_index']) for x in document['entries']}
        temporal = read_temporal_audit(temporal_path,
            expected_source_sha256=dataset_identity['manifest_sha256'], expected_keys=keys)
        if document['source_identity'] != dataset_identity:
            raise ValueError('Certified temporal source identity mismatch')
        counts = Counter(x.episode_id for x in temporal)
        eligible = {e: v['bc_valid_transitions'] for e, v in episode_metadata.items() if v['bc_valid_transitions']}
        if dict(counts) != eligible or any(x.transition_index >= episode_metadata[x.episode_id]['canonical_transitions'] for x in temporal):
            raise ValueError('Certified temporal inventory differs from frozen episode counts/ranges')
    return dict(dataset_identity=dict(dataset_identity), topology_evidence_provider=topology,
        source_frame_provider=frame_provider, frame_source_reader=frame_reader, temporal_audit=temporal,
        provider_config_files=(topology_policy_path, catalog_path, frame_policy_path)
            + (() if temporal_path is None else (temporal_path,)))


def run_prepare(args):
    from .bc_v1_contract import load_robot_spec, robot_spec_fingerprint
    from .dataset import prepare_dataset, file_hash
    from .feature_schema import load_feature_schema, feature_schema_fingerprint
    from .normalization import load_normalization_policy, policy_fingerprint
    from .split import read_split_manifest
    from .task_context import read_task_metadata
    config = json.loads(args.config.read_text())
    assets = json.loads(args.prepare_config.read_text())
    root = Path.cwd().resolve()
    source = args.source.resolve()
    if source != (root/config['source']['path']).resolve() or file_hash(source/'manifest.json') != config['source']['manifest_sha256']:
        raise ValueError('Explicit source differs from frozen training config')
    inventory = json.loads((root/config['split']['inventory_path']).read_text())
    split = read_split_manifest(root/config['split']['manifest_path'], inventory['episodes'],
        config['split']['strata_config'], inventory['dataset_identity'],
        expected_seed=config['split']['seed'], expected_fingerprint=config['split']['manifest_fingerprint'])
    spec = load_robot_spec(root/assets['robot_spec'])
    schema = load_feature_schema(root/assets['feature_schema'])
    policy = load_normalization_policy(root/assets['normalization_policy'], schema)
    actual = dict(robot_spec_fingerprint=robot_spec_fingerprint(spec),
        feature_schema_fingerprint=feature_schema_fingerprint(schema),
        normalization_policy_fingerprint=policy_fingerprint(policy, schema))
    if actual != assets['expected_fingerprints']: raise ValueError('Prepare assets differ from frozen pins')
    if assets['adapter'] != 'smores_ep': raise ValueError('Unsupported configured prepare adapter')
    from . import smores_adapter as sm
    adapter = sm.SMORESAdapter(spec)
    dataset_identity = {'manifest': str((source/'manifest.json').relative_to(root)),
                        'manifest_sha256': config['source']['manifest_sha256']}
    providers = compose_runtime_providers(root, dataset_identity, assets, split['episode_metadata'])
    def metadata(row):
        result = read_task_metadata(row, root, record_reader=_read_indexed_source_records)
        for entry in result['audit']:
            if 'records' in entry:
                entry['sha256'] = sm._source_index(Path(entry['path'])).sha256
        return result
    def labels(row, graph):
        return encode_smores_labels(row, adapter, root)
    files = [args.config, args.prepare_config, root/assets['robot_spec'], root/assets['feature_schema'],
             root/assets['task_context'], root/assets['normalization_policy'],
             root/config['split']['manifest_path'], root/config['split']['inventory_path'],
             Path(sm.__file__), Path(__file__)]
    files.extend(providers.pop('provider_config_files'))
    from . import smores_disconnect_evidence, smores_source_frames, temporal_supervision
    files.extend(Path(module.__file__) for module in (
        smores_disconnect_evidence, smores_source_frames, temporal_supervision))
    prepared = prepare_dataset(source, args.output, adapter, dict(repo_root=root, split_manifest=split,
        expected_split_fingerprint=config['split']['manifest_fingerprint'], spec=spec, schema=schema,
        normalization_policy=policy, task_config=json.loads((root/assets['task_context']).read_text()),
        config_files=files,
        input_hashes={'adapter': file_hash(sm.__file__), 'eligibility_contract': file_hash(root/'drl/IL/bc_v1/policy_inputs.py'),
                      'cli': file_hash(__file__), 'split_file_sha256': file_hash(root/config['split']['manifest_path'])},
        metadata_reader=metadata, label_reader=labels,
        progress=lambda event: print(json.dumps(event, sort_keys=True), flush=True), **providers))
    print(json.dumps({'counts': prepared['counts'], 'cache_fingerprint': prepared['cache_fingerprint'],
        'performance': prepared['performance']}, sort_keys=True), flush=True)


def run_fit_normalizers(args):
    """Compose only generic frozen assets; no adapter or runtime providers."""
    from .bc_v1_contract import load_robot_spec, robot_spec_fingerprint
    from .feature_schema import load_feature_schema, feature_schema_fingerprint
    from .normalization import load_normalization_policy, policy_fingerprint
    from .normalization_prepare import fit_prepared_normalizers
    from .split import read_split_manifest
    root = Path.cwd().resolve()
    config = json.loads(args.config.read_text())
    assets = json.loads(args.prepare_config.read_text())
    inventory = json.loads((root/config['split']['inventory_path']).read_text())
    split = read_split_manifest(root/config['split']['manifest_path'], inventory['episodes'],
        config['split']['strata_config'], inventory['dataset_identity'],
        expected_seed=config['split']['seed'], expected_fingerprint=config['split']['manifest_fingerprint'])
    spec = load_robot_spec(root/assets['robot_spec'])
    schema = load_feature_schema(root/assets['feature_schema'])
    policy = load_normalization_policy(root/assets['normalization_policy'], schema)
    actual = dict(robot_spec_fingerprint=robot_spec_fingerprint(spec),
        feature_schema_fingerprint=feature_schema_fingerprint(schema),
        normalization_policy_fingerprint=policy_fingerprint(policy, schema))
    if actual != assets['expected_fingerprints']: raise ValueError('F2 assets differ from frozen pins')
    expected = dict(actual, source_manifest_sha256=config['source']['manifest_sha256'],
                    split_fingerprint=config['split']['manifest_fingerprint'])
    context = fit_prepared_normalizers(args.cache, args.output, split_manifest=split,
        spec=spec, schema=schema, policy=policy, expected_identity=expected)
    print(json.dumps(context, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
