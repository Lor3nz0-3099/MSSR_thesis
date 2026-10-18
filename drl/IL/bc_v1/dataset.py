"""G1 offline physical caches. No hardware imports, tensors, sampling or scaling.

Only trusted local prepare artifacts may be unpickled. Integrity is checked before
torch.load. Semantic identity is independent of pickle's unordered-set byte order;
individual .pt hashes authenticate bytes separately. Compatibility expectations
are independently pinned by callers; code/config identities are also checked.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
from typing import Mapping

from .bc_v1_contract import PolicyInputs, SampleAudit, StateGraph, SupervisionBundle, robot_spec_fingerprint
from .candidates import CandidateTables, enumerate_candidates
from .causal_topology import CausalBoundary, reconcile_topology
from .feature_schema import FeatureView, extract_features, feature_schema_fingerprint
from .features import FrameGeometryView, transform_inputs
from .normalization import policy_fingerprint
from .policy_inputs import select_policy_inputs
from .pipeline_validation import validate_physical_pipeline
from .split import serialize_split_manifest
from .source_frame_bridges import PipelineDiagnostic, SourceFrameBridgeBatch, frame_config_for
from .task_context import build_task_view
from .temporal_supervision import validate_temporal_audit

VERSION = 'mssr.bc_v1.physical_cache.v2'


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _signature(path):
    stat = Path(path).stat()
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def cache_fingerprint(config: Mapping, input_hashes: Mapping) -> str:
    return _hash({'config': config, 'input_hashes': input_hashes})


def _code_hashes():
    names = ('dataset', 'bc_v1_contract', 'policy_inputs', 'task_context', 'features',
             'feature_schema', 'candidates', 'normalization', 'split',
             'causal_topology', 'source_frame_bridges', 'temporal_supervision', 'pipeline_validation')
    return {name: file_hash(Path(__file__).with_name(name+'.py')) for name in names}


@dataclass(frozen=True)
class CachedInputs:
    policy: PolicyInputs
    features: FeatureView
    candidates: CandidateTables


@dataclass(frozen=True)
class CachedSample:
    inputs: CachedInputs
    supervision: SupervisionBundle
    audit: SampleAudit


@dataclass(frozen=True)
class PreparedPhysicalInputs:
    """Complete deterministic physical derivation; receipts stay outside policy."""
    published_graph: StateGraph
    inputs: CachedInputs
    geometry: FrameGeometryView
    features: FeatureView
    bridges: SourceFrameBridgeBatch
    diagnostics: tuple[PipelineDiagnostic, ...]
    topology_audit: tuple[tuple[str, str], ...]

    def __post_init__(self):
        if self.features is not self.inputs.features:
            raise ValueError('Prepared features must be shared with CachedInputs')


def _routing_provenance(row, config):
    raw = row.get('provenance') or {}
    result = {key: raw[key] for key in ('logical_episode_id', 'canonical_transition_index', 'physical_stamp') if key in raw}
    result['source_records'] = tuple({key: ref[key] for key in (
        '_loader_source_path', '_loader_source_row', '_loader_source_kind') if key in ref}
        for ref in raw.get('source_records', ()))
    if 'dataset_identity' in config:
        result['dataset_identity'] = dict(config['dataset_identity'])
    return result


def derive_physical_inputs(row: Mapping, adapter, config: Mapping,
                           candidate_cache: CurrentCandidateCache) -> PreparedPhysicalInputs:
    """Own adapter → reconciliation → D1/D2 → bridges → E1/E2, once.

    The injected topology provider alone may certify non-applicability (None).
    Missing providers never silently use the published topology/world defaults.
    Neither topology nor frame readers receive the Canonical action or future.
    """
    published = adapter.encode_state(row['graph_t'])
    provenance = _routing_provenance(row, config)
    evidence = config['topology_evidence_provider'](provenance, published)
    if evidence is None:
        topology = reconcile_topology(published, (), CausalBoundary('certified_non_applicable', 0))
        audit = {'applicable': False, 'reason': 'provider_certified_non_applicable'}
    else:
        topology = reconcile_topology(published, evidence.events, evidence.boundary)
        audit = dict(evidence.audit)
    metadata = config['metadata_reader'](row)
    context, observation = build_task_view(row, metadata, config['task_config'])
    inputs = select_policy_inputs(topology.graph, observation, context)
    descriptor = tuple({'name': quantity.name, 'values': quantity.values, 'valid': quantity.valid}
        for quantity in inputs.context.static_geometry
        if quantity.name in ('world.task_frame.origin_xyz_m', 'world.task_frame.yaw_rad')
        or quantity.name.startswith('stage_local.'))
    frame_metadata = {'current_task_geometry': descriptor,
        'audit': tuple({'path': receipt['path'], 'sha256': receipt['sha256']}
            for receipt in metadata.get('audit', ()) if 'path' in receipt and 'sha256' in receipt)}
    source = config['frame_source_reader'](provenance, frame_metadata)
    bridges = config['source_frame_provider'].resolve(inputs, source)
    geometry = transform_inputs(inputs, frame_config_for(bridges))
    features = extract_features(inputs, geometry, config['spec'], config['schema'])
    diagnostics = validate_physical_pipeline(inputs, geometry, features, bridges, config['schema'])
    candidates = candidate_cache.get(topology.graph)
    return PreparedPhysicalInputs(published, CachedInputs(inputs, features, candidates),
        geometry, features, bridges, diagnostics, tuple((key, _json(value)) for key, value in sorted(audit.items())))


class CurrentCandidateCache:
    """Intern immutable tables by exactly E2's current catalog/topology inputs.

    Reset per episode. A bounded lookup avoids retaining unrelated old topologies;
    samples still own their table. Physical measurements are never admission inputs.
    """
    def __init__(self, spec, schema):
        self.spec, self.schema, self._tables = spec, schema, OrderedDict()

    def get(self, graph):
        key = (tuple((m.ref, m.module_type_id) for m in graph.modules),
               tuple((p.ref, p.kind, p.status, p.connected_to) for p in graph.connectors),
               tuple((r.ref, tuple(q.coordinate_id for q in r.coordinates)) for r in graph.resources),
               tuple((e.endpoint_a, e.endpoint_b, e.relation_id) for e in graph.connections))
        if key not in self._tables:
            table = enumerate_candidates(graph, self.spec, self.schema)
            if len(self._tables) == 64: self._tables.popitem(last=False)
            self._tables[key] = table
        self._tables.move_to_end(key)
        return self._tables[key]


def _safe_file(root, relative):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError('Artifact/source path escapes its root')
    return path


def _load_episode(root, eid, entry):
    import torch
    path = _safe_file(root, entry['cache_file'])
    if file_hash(path) != entry['cache_sha256']:
        raise ValueError('Cache file integrity hash mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if type(payload) is not dict or payload.get('schema_version') != VERSION or payload.get('episode_id') != eid:
        raise ValueError('Invalid episode cache schema/identity')
    samples = payload.get('samples')
    if type(samples) is not tuple or len(samples) != entry['samples']:
        raise ValueError('Invalid episode cache count')
    indices = []
    for sample in samples:
        if type(sample) is not CachedSample or type(sample.inputs) is not CachedInputs or type(sample.supervision) is not SupervisionBundle:
            raise ValueError('Invalid typed cache sample')
        if type(sample.inputs.policy) is not PolicyInputs or type(sample.inputs.features) is not FeatureView or type(sample.inputs.candidates) is not CandidateTables:
            raise ValueError('Invalid typed cache inputs')
        if select_policy_inputs(sample.inputs.policy.graph, sample.inputs.policy.observation,
                                sample.inputs.policy.context) != sample.inputs.policy:
            raise ValueError('Cache violates policy input whitelist')
        if sample.audit.episode_id != eid or type(sample.audit.transition_index) is not int or sample.audit.transition_index < 0:
            raise ValueError('Invalid cache routing/index')
        indices.append(sample.audit.transition_index)
    if indices != sorted(set(indices)) or _hash(indices) != entry['transition_indices_fingerprint']:
        raise ValueError('Duplicate/invalid transition indices')
    return samples


def _manifest_fingerprint(manifest):
    return _hash({k: v for k, v in manifest.items() if k != 'manifest_fingerprint'})


def read_cache_manifest(root, *, expected_identity=None):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if (manifest.get('schema_version') != VERSION or manifest.get('complete') is not True
            or manifest.get('normalized') is not False):
        raise ValueError('Unsupported/incomplete cache schema')
    if manifest.get('manifest_fingerprint') != _manifest_fingerprint(manifest):
        raise ValueError('Cache manifest fingerprint mismatch')
    identity = manifest['identity']
    if expected_identity is not None and any(identity.get(k) != v for k, v in expected_identity.items()):
        raise ValueError('Cache stale/incompatible with independent identity expectations')
    if identity['code'] != _code_hashes():
        raise ValueError('Cache stale/incompatible with current code')
    for path, digest in manifest['config_files'].items():
        if file_hash(path) != digest:
            raise ValueError('Cache stale/incompatible with current config')
    if manifest['cache_fingerprint'] != cache_fingerprint(identity, manifest['source_episode_hashes']):
        raise ValueError('Cache semantic fingerprint mismatch')
    seen = set()
    for part in ('train', 'validation', 'test'):
        ids = manifest['splits'][part]
        if len(ids) != len(set(ids)) or seen.intersection(ids):
            raise ValueError('Cache split overlap/duplicate')
        seen.update(ids)
        count = manifest['counts']['splits'][part]
        if count != {'episodes': len(ids), 'samples': sum(manifest['episodes'][e]['samples'] for e in ids)}:
            raise ValueError('Cache split count mismatch')
    if seen != set(manifest['episodes']) or seen.intersection(manifest['excluded_episode_ids']):
        raise ValueError('Cache membership mismatch')
    if manifest['counts']['episodes'] != len(seen) or manifest['counts']['samples'] != sum(e['samples'] for e in manifest['episodes'].values()):
        raise ValueError('Cache total count mismatch')
    return manifest


class BCDataset:
    """Read-only split index; at most one episode is retained in RAM per instance."""
    def __init__(self, cache_root: Path, split_name: str, *, expected_identity=None):
        if split_name not in ('train', 'validation', 'test'):
            raise ValueError('Unknown split name')
        self.cache_root = Path(cache_root)
        self.manifest = read_cache_manifest(self.cache_root, expected_identity=expected_identity)
        self.episode_ids = tuple(self.manifest['splits'][split_name])
        self._ends = [0]
        for eid in self.episode_ids:
            self._ends.append(self._ends[-1]+self.manifest['episodes'][eid]['samples'])
        self._loaded_id, self._loaded = None, None

    def __len__(self):
        return self._ends[-1]

    def __getitem__(self, index):
        if type(index) is not int:
            raise TypeError('Sample index must be an integer')
        if index < 0: index += len(self)
        if not 0 <= index < len(self): raise IndexError(index)
        position = bisect_right(self._ends, index)-1
        eid = self.episode_ids[position]
        if eid != self._loaded_id:
            self._loaded = None
            self._loaded = _load_episode(self.cache_root, eid, self.manifest['episodes'][eid])
            self._loaded_id = eid
        return self._loaded[index-self._ends[position]]


def _representable(bundle, table):
    """Generic current-domain checks; labels never alter candidate enumeration."""
    for target in bundle.resource_targets:
        matches = [x for x in table.resource_candidates if x.resource_ref == target.resource and x.mode.mode_id == target.mode_id]
        if len(matches) != 1 or len(target.payload) != matches[0].mode.payload_size:
            raise ValueError('Resource target not representable by current candidates')
    for target in bundle.structural_targets:
        matches = [b for x in table.structural_candidates if x.action.action_id == target.operation_id
            and x.endpoints == frozenset(target.connectors) for b in x.bindings
            if b.initiator == target.initiator and (b.connectors == target.connectors if target.connectors
                else set(target.participants) <= b.scope_members)]
        if len(matches) != 1:
            raise ValueError('Structural target not representable by current candidates')
        binding = matches[0]
        if not binding.required_scope_members <= set(target.participants) <= binding.scope_members:
            raise ValueError('Structural target scope not representable')
        domains = {d.parameter.name: d for d in binding.arguments}
        for value in target.parameters:
            if value.name not in domains: raise ValueError('Unknown structural argument')
            domain = domains[value.name]; parameter = domain.parameter
            if parameter.kind in ('module_ref', 'connector_ref'):
                if value.value not in domain.options or domain.bound_ref is not None and value.value != domain.bound_ref:
                    raise ValueError('Structural reference not representable')
            elif parameter.kind in ('module_refs', 'connector_refs'):
                if not set(value.value) <= domain.options or parameter.size is not None and len(value.value) != parameter.size:
                    raise ValueError('Structural reference set not representable')
            elif parameter.kind == 'category' and value.value not in parameter.categories:
                raise ValueError('Structural category not representable')
            elif parameter.kind == 'boolean' and type(value.value) is not bool:
                raise ValueError('Structural boolean not representable')
            elif parameter.kind in ('scalar', 'vector'):
                components = value.value if parameter.kind == 'vector' else (value.value,)
                if parameter.kind == 'vector' and len(components) != parameter.size:
                    raise ValueError('Structural vector width mismatch')
                if parameter.limits is not None and any(not parameter.limits[0] <= x <= parameter.limits[1] for x in components):
                    raise ValueError('Structural parameter outside declared domain')


def prepare_dataset(source: Path, output: Path, adapter, config: Mapping) -> Mapping:
    """Explicit offline transaction. Config contains already verified F1 and assets.

    Hardware label/evidence and metadata readers are injected only during prepare.
    Cache files carry physical typed inputs/targets, never F2-transformed values.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    root = Path(config['repo_root']).resolve()
    if output == source or output.is_relative_to(source) or output.is_relative_to(root/'datasets'):
        raise ValueError('Output forbidden inside source/datasets')
    if source.is_relative_to(output):
        raise ValueError('Unsafe output collision with source')
    if output.exists(): raise ValueError('Output collision: cache already exists')
    split = config['split_manifest']; serialize_split_manifest(split)
    if split['manifest_fingerprint'] != config['expected_split_fingerprint']:
        raise ValueError('Frozen split fingerprint incompatible')
    source_digest = file_hash(source/'manifest.json')
    if source_digest != split['dataset_identity']['manifest_sha256']:
        raise ValueError('Source manifest hash mismatch')
    source_manifest = json.loads((source/'manifest.json').read_bytes())
    entries = {e['episode_id']: e for e in source_manifest['episodes']}
    if len(entries) != len(source_manifest['episodes']) or set(entries) != set(split['episode_metadata']):
        raise ValueError('Source episode inventory differs from frozen split')
    for eid, entry in entries.items():
        for key in ('sha256', 'file', 'bc_valid_transitions', 'canonical_transitions', 'bytes'):
            if entry[key] != split['episode_metadata'][eid][key]:
                raise ValueError('Source episode metadata differs from frozen split')
    source_signatures = {e: _signature(_safe_file(source, v['file'])) for e, v in entries.items()}
    spec, schema, policy = config['spec'], config['schema'], config['normalization_policy']
    identity = dict(config['input_hashes'], source_manifest_sha256=source_digest,
        split_fingerprint=split['manifest_fingerprint'], robot_spec_fingerprint=robot_spec_fingerprint(spec),
        feature_schema_fingerprint=feature_schema_fingerprint(schema),
        normalization_policy_fingerprint=policy_fingerprint(policy, schema),
        task_config_fingerprint=_hash(config['task_config']), policy_input_contract=_code_hashes()['policy_inputs'],
        source_frame_provider=dict(config['source_frame_provider'].identity()),
        cache_schema=VERSION, code=_code_hashes())
    config_files = {str(Path(p).resolve()): file_hash(p) for p in config.get('config_files', ())}
    runtime = dict(config, dataset_identity={
        'manifest': str((source/'manifest.json').relative_to(root)), 'manifest_sha256': source_digest})
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'-', dir=output.parent))
    started = time.perf_counter()
    external, cached, source_hashes, supervision_counts = {}, {}, {}, Counter()
    temporal = config.get('temporal_audit')
    temporal_keys = set() if temporal is None else {(x.episode_id, x.transition_index) for x in temporal}
    if temporal is not None: validate_temporal_audit(temporal, temporal_keys)
    seen_temporal = set()
    def tracked_metadata(row):
        metadata = config['metadata_reader'](row)
        for item in metadata['audit']:
            if 'sha256' in item:
                old = external.setdefault(item['path'], item['sha256'])
                if old != item['sha256']: raise ValueError('External metadata changed')
            elif item.get('missing') is True:
                if item['path'] in external and external[item['path']] is not None:
                    raise ValueError('External metadata missingness changed')
                external[item['path']] = None
        return metadata
    runtime['metadata_reader'] = tracked_metadata
    output_reserved = False
    published = False
    try:
        import torch
        import numpy
        (staging/'episodes').mkdir()
        for eid, entry in sorted(entries.items()):
            candidate_cache = CurrentCandidateCache(spec, schema)
            path = _safe_file(source, entry['file'])
            digest = hashlib.sha256(); samples = []; row_count = 0
            before = path.stat()
            if before.st_size != entry['bytes']: raise ValueError(f'Source episode size mismatch: {eid}')
            if entry['bc_valid_transitions'] == 0:
                actual_hash = file_hash(path)
            else:
                with path.open('rb') as stream:
                    for index, line in enumerate(stream):
                        digest.update(line); row_count += 1
                        row = json.loads(line)
                        valid = row['supervision']['valid_for_behavior_cloning']
                        if type(valid) is not bool: raise ValueError('Invalid BC eligibility flag')
                        if not valid: continue
                        try:
                            if temporal is not None:
                                if (eid, index) not in temporal_keys:
                                    raise ValueError('Missing certified temporal BC key')
                                seen_temporal.add((eid, index))
                            derived = derive_physical_inputs(row, adapter, runtime, candidate_cache)
                            label_reader = config.get('label_reader')
                            bundle = (label_reader(row, derived.published_graph) if label_reader else
                                      adapter.encode_labels(row['action_t'], row['supervision'], ()))
                            if not bundle.resource_targets and not bundle.structural_targets:
                                raise ValueError('BC-valid sample has no supervised targets')
                            _representable(bundle, derived.inputs.candidates)
                            for evidence in bundle.label_evidence:
                                old = external.setdefault(evidence.source_path, evidence.source_sha256)
                                if old != evidence.source_sha256: raise ValueError('External evidence changed')
                            supervision_counts.update(t.mode_id for t in bundle.resource_targets)
                            supervision_counts.update(t.operation_id for t in bundle.structural_targets)
                            audit = SampleAudit(eid, index, row['graph_t'].get('stamp'),
                                provenance=entry.get('kind'), task_stratum=split['episode_metadata'][eid].get('stratum'),
                                morphology_stratum=row['graph_t'].get('morphology'),
                                source_hashes=((entry['file'], entry['sha256']),))
                            samples.append(CachedSample(derived.inputs, bundle, audit))
                        except (ValueError, KeyError, TypeError) as exc:
                            raise ValueError(f'{eid} transition {index}: {exc}') from exc
                actual_hash = digest.hexdigest()
                if row_count != entry['canonical_transitions'] or len(samples) != entry['bc_valid_transitions']:
                    raise ValueError('Source/BC count mismatch')
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) or actual_hash != entry['sha256']:
                raise ValueError(f'Source episode hash/integrity mismatch: {eid}')
            source_hashes[eid] = actual_hash
            if samples:
                relative = 'episodes/'+hashlib.sha256(eid.encode()).hexdigest()+'.pt'
                torch.save({'schema_version': VERSION, 'episode_id': eid, 'samples': tuple(samples)}, staging/relative)
                cached[eid] = dict(cache_file=relative, cache_sha256=file_hash(staging/relative),
                    samples=len(samples), source_sha256=actual_hash,
                    transition_indices_fingerprint=_hash([s.audit.transition_index for s in samples]))
            progress = config.get('progress')
            if progress: progress({'phase': 'episode', 'episode_id': eid, 'samples': len(samples), 'completed': len(cached)})
            del samples
        if temporal is not None: validate_temporal_audit(temporal, seen_temporal)
        identity['external_metadata'] = _hash(external)
        manifest = dict(schema_version=VERSION, complete=True, normalized=False, identity=identity,
            cache_fingerprint=cache_fingerprint(identity, source_hashes), source_episode_hashes=source_hashes,
            source_identity=dict(split['dataset_identity'], source_root=str(source)),
            external_metadata_hashes=external, config_files=config_files, episodes=cached,
            splits={p: sorted(split['splits'][p]) for p in ('train', 'validation', 'test')},
            excluded_episode_ids=split['excluded_episode_ids'],
            counts={'episodes': len(cached), 'samples': sum(e['samples'] for e in cached.values()),
                'splits': {p: {'episodes': len(split['splits'][p]), 'samples': sum(cached[e]['samples'] for e in split['splits'][p])}
                           for p in ('train', 'validation', 'test')}},
            supervision_counts=dict(sorted(supervision_counts.items())),
            performance={'prepare_seconds': time.perf_counter()-started},
            versions={'python': platform.python_version(), 'torch': torch.__version__, 'numpy': numpy.__version__})
        manifest['manifest_fingerprint'] = _manifest_fingerprint(manifest)
        (staging/'manifest.json').write_text(_json(manifest)+'\n')
        read_cache_manifest(staging, expected_identity=identity)
        if file_hash(source/'manifest.json') != source_digest: raise ValueError('Source manifest changed during prepare')
        if source_signatures != {e: _signature(_safe_file(source, v['file'])) for e, v in entries.items()}:
            raise ValueError('Source episode changed during prepare')
        for path, digest in external.items():
            if (Path(path).exists() if digest is None else not Path(path).is_file() or file_hash(path) != digest):
                raise ValueError('External metadata/evidence dependency changed during prepare')
        if output.exists(): raise ValueError('Output collision before publication')
        # Reserve the otherwise absent directory without clobbering a concurrent
        # publisher. The replacement below is atomic and only replaces OUR empty
        # reservation. Readers cannot see a complete manifest before that rename.
        output.mkdir()
        output_reserved = True
        os.rename(staging, output)
        published = True
        return manifest
    finally:
        if staging.exists(): shutil.rmtree(staging)
        if not published and output_reserved and output.is_dir() and not any(output.iterdir()):
            output.rmdir()
