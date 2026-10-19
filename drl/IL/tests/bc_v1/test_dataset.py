"""G1: physical typed caches, frozen membership and transactional prepare."""
import copy
from dataclasses import replace
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.bc_v1.feature_schema import load_feature_schema
from drl.IL.bc_v1.normalization import load_normalization_policy, read_normalizers
from drl.IL.bc_v1.split import build_split_manifest, make_episode_split
from drl.IL.tests.bc_v1 import fixtures as f

ROOT = Path(__file__).resolve().parents[4]


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.dataset'), 'Missing G1 dataset/cache engine'
    return importlib.import_module('drl.IL.bc_v1.dataset')


def world_runtime():
    """Explicit non-applicable topology provider for isolated generic fixtures."""
    from drl.IL.bc_v1.source_frame_bridges import (
        BridgeResolution, FrameSourceContext, SourceFrameBridgeBatch)
    class WorldProvider:
        def identity(self):
            return {'provider_id': 'test-world', 'revision': '1'}
        def resolve(self, inputs, source):
            return SourceFrameBridgeBatch((BridgeResolution('world', 'world', 'current',
                None, True, 'implicit_identity', None, ()),), tuple(self.identity().items()))
    return dict(topology_evidence_provider=lambda provenance, graph: None,
        source_frame_provider=WorldProvider(),
        frame_source_reader=lambda provenance, metadata: FrameSourceContext('test', (), ()))


class Adapter:
    def encode_state(self, raw):
        g = f.state_graph(raw['n'], 0)
        return replace(g, resources=()) if raw.get('empty') else g

    def encode_labels(self, action, supervision, evidence):
        m = c.ModuleRef('unit-0')
        targets = () if action.get('empty') else (
            c.ResourceTarget(c.ResourceRef(m, 'joint'), 'rate', (4.,)),)
        return c.SupervisionBundle(targets, (c.StructuralTarget(m, 'gravity_settle', (m,),
            parameters=(c.ParameterValue('duration', 2.), c.ParameterValue('passive_modules', (m,)),
                        c.ParameterValue('clear_on_finish', True))),))


@pytest.fixture
def prepared_inputs(tmp_path):
    source = tmp_path/'datasets'/'canonical'
    source.mkdir(parents=True)
    entries = []
    for eid in ('a', 'b', 'c', 'excluded'):
        rows = [{'schema_version': 'mssr.canonical_transition.v5',
                 'graph_t': {'n': n, 'stamp': float(i), 'target_morphology': 'answer'},
                 'graph_t_plus_1': {'n': 999},
                 'action_t': {'n': n, 'configuration_goal': 'answer'},
                 'supervision': {'valid_for_behavior_cloning': eid != 'excluded' and i != 1},
                 'task_type': 'assembly', 'stage': 'hidden-stage',
                 'configuration_goal': 'answer', 'provenance': {'private': eid}}
                for i, n in enumerate((2, 5, 3))]
        blob = ''.join(json.dumps(row)+'\n' for row in rows).encode()
        (source/f'{eid}.jsonl').write_bytes(blob)
        entries.append(dict(episode_id=eid, file=f'{eid}.jsonl', bytes=len(blob),
            sha256=hashlib.sha256(blob).hexdigest(), canonical_transitions=3,
            bc_valid_transitions=0 if eid == 'excluded' else 2, tag='all', lineage_id=eid))
    manifest = {'dataset_name': 'fixture', 'schema_version': 'fixture.v1', 'episodes': entries}
    (source/'manifest.json').write_text(json.dumps(manifest))
    identity = dict(dataset_name='fixture', schema_version='fixture.v1',
                    manifest_sha256=hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest())
    strata = dict(schema_version='mssr.bc_v1.strata.v1', eligible_count_field='bc_valid_transitions',
                  strata=[dict(name='all', match={'tag': 'all'}, total=3,
                               quotas={'train': 1, 'validation': 1, 'test': 1})])
    split = build_split_manifest(make_episode_split(entries, strata, 20261006),
                                 entries, strata, 20261006, identity)
    spec = f.robot_spec(0)
    spec = replace(spec, actions=(replace(spec.actions[1], action_id='gravity_settle'),)+(c.ActionSpec('rate', 'continuous', 'resource',
        (c.ParameterSpec('value', 'scalar', units='rad/s', module_type_id='unit', resource_id='joint'),)),))
    schema = load_feature_schema(ROOT/'configs/bc_v1/features.json')
    policy = load_normalization_policy(ROOT/'configs/bc_v1/normalization.json', schema)
    policy = copy.deepcopy(policy)
    policy['resource_payloads'] = [{'module_type_id': 'unit', 'resource_id': 'joint',
        'mode_id': 'rate', 'components': [{'parameter': 'value', 'component': 0}]}]
    config = dict(repo_root=tmp_path, split_manifest=split, expected_split_fingerprint=split['manifest_fingerprint'],
        spec=spec, schema=schema, normalization_policy=policy,
        task_config=json.loads((ROOT/'configs/bc_v1/task_context.json').read_text()),
        normalizer_output=tmp_path/'artifacts'/'normalizers.json',
        input_hashes={'eligibility_contract': 'a'*64, 'adapter': 'b'*64},
        metadata_reader=lambda row: {'sources': [], 'audit': []})
    config.update(world_runtime())
    return source, tmp_path/'artifacts'/'cache', Adapter(), config


def prepare(data):
    return api().prepare_dataset(*data)


def test_roundtrip_physical_types_joint_supervision_variable_n_and_audit(prepared_inputs):
    manifest = prepare(prepared_inputs)
    ds = api().BCDataset(prepared_inputs[1], 'train')
    eid = prepared_inputs[3]['split_manifest']['splits']['train'][0]
    assert len(ds) == 2 and ds.episode_ids == (eid,)
    first, last = ds[0], ds[-1]
    assert type(first.inputs.policy) is c.PolicyInputs
    assert type(first.inputs.features).__name__ == 'FeatureView'
    assert type(first.inputs.candidates).__name__ == 'CandidateTables'
    assert len(first.inputs.policy.graph.modules) == 2
    assert len(last.inputs.policy.graph.modules) == 3
    assert not first.inputs.features.connectors and not first.inputs.features.environment
    assert first.supervision == prepared_inputs[2].encode_labels({}, {}, ())
    assert first.supervision.resource_targets[0].payload == (4.,)
    assert first.audit.episode_id == eid and first.audit.transition_index == 0
    assert last.audit.transition_index == 2
    assert first.inputs.policy.context.task_id is None  # Existing D1 ontology: unobserved/unknown.
    assert not hasattr(first.inputs.policy.context, 'stage')
    assert not hasattr(first.inputs, 'audit') and not hasattr(first.inputs, 'supervision')
    assert 'answer' not in repr(first.inputs) and 'hidden-stage' not in repr(first.inputs)
    assert manifest['counts'] == {'episodes': 3, 'samples': 6,
        'splits': {'train': {'episodes': 1, 'samples': 2}, 'validation': {'episodes': 1, 'samples': 2},
                   'test': {'episodes': 1, 'samples': 2}}}
    assert manifest['excluded_episode_ids'] == ['excluded']
    assert manifest['schema_version'] == 'mssr.bc_v1.physical_cache.v2'
    assert 'normalizer' not in manifest
    assert not prepared_inputs[3]['normalizer_output'].exists()


def test_lazy_read_only_no_normalization_and_file_hash_check(prepared_inputs, monkeypatch):
    m = prepare(prepared_inputs)
    import torch
    original = torch.load
    reads = []
    def load(path, **kwargs):
        reads.append(Path(path).name)
        return original(path, **kwargs)
    monkeypatch.setattr(torch, 'load', load)
    ds = api().BCDataset(prepared_inputs[1], 'validation')
    assert not reads
    assert ds[0].supervision.resource_targets[0].payload == (4.,)
    ds[1]
    assert len(reads) == 1
    with pytest.raises(IndexError): ds[2]
    eid = prepared_inputs[3]['split_manifest']['splits']['validation'][0]
    file = prepared_inputs[1]/m['episodes'][eid]['cache_file']
    file.write_bytes(file.read_bytes()+b'corrupt')
    with pytest.raises(ValueError, match='hash|integrity'):
        api().BCDataset(prepared_inputs[1], 'validation')[0]


@pytest.mark.parametrize('key', ['source_manifest_sha256', 'eligibility_contract', 'split_fingerprint',
    'robot_spec_fingerprint', 'feature_schema_fingerprint', 'task_config_fingerprint',
    'policy_input_contract', 'normalization_policy_fingerprint', 'adapter', 'external_metadata'])
def test_stale_compatibility_pin_rejected(prepared_inputs, key):
    m = prepare(prepared_inputs)
    expected = dict(m['identity']); expected[key] = 'f'*64
    with pytest.raises(ValueError, match='stale|incompatible'):
        api().BCDataset(prepared_inputs[1], 'train', expected_identity=expected)


def test_cache_schema_and_manifest_corruption_rejected(prepared_inputs):
    prepare(prepared_inputs)
    path = prepared_inputs[1]/'manifest.json'
    m = json.loads(path.read_text()); m['schema_version'] = 'unsupported'
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='schema|fingerprint'):
        api().BCDataset(prepared_inputs[1], 'train')


def test_source_hash_mismatch_rejected(prepared_inputs):
    source, output, adapter, config = prepared_inputs
    (source/'a.jsonl').write_bytes((source/'a.jsonl').read_bytes()+b'\n')
    with pytest.raises(ValueError, match='source|Source'):
        prepare(prepared_inputs)
    assert not (output/'manifest.json').exists()


@pytest.mark.parametrize('location', ['source', 'child', 'datasets'])
def test_output_inside_source_or_datasets_rejected(prepared_inputs, location):
    source, _, adapter, config = prepared_inputs
    output = {'source': source, 'child': source/'cache', 'datasets': source.parent/'cache'}[location]
    with pytest.raises(ValueError, match='source|datasets|output'):
        prepare((source, output, adapter, config))


def test_collision_preserves_previous_cache(prepared_inputs):
    prepare(prepared_inputs)
    blob = (prepared_inputs[1]/'manifest.json').read_bytes()
    with pytest.raises(ValueError, match='exist|collision'):
        prepare(prepared_inputs)
    assert (prepared_inputs[1]/'manifest.json').read_bytes() == blob
    assert len(api().BCDataset(prepared_inputs[1], 'test')) == 2


def test_late_failure_no_publication_or_normalizer(prepared_inputs):
    source, output, adapter, config = prepared_inputs
    class Failure(Adapter):
        calls = 0
        def encode_state(self, raw):
            self.calls += 1
            assert not (output/'manifest.json').exists()
            if self.calls == 5: raise ValueError('late injected failure')
            return super().encode_state(raw)
    with pytest.raises(ValueError, match='late injected failure'):
        prepare((source, output, Failure(), config))
    assert not output.exists() and not config['normalizer_output'].exists()


def test_cache_fingerprint_order_independent_and_content_sensitive():
    fn = api().cache_fingerprint
    assert fn({'a': 1, 'b': 2}, {'a': 'x', 'b': 'y'}) == fn({'b': 2, 'a': 1}, {'b': 'y', 'a': 'x'})
    assert fn({'a': 1}, {'a': 'x'}) != fn({'a': 2}, {'a': 'x'})


def test_candidate_interning_depends_only_on_current_domains():
    cls = getattr(api(), 'CurrentCandidateCache', None)
    assert cls is not None, 'Missing bounded immutable candidate interning'
    from drl.IL.bc_v1.candidates import enumerate_candidates
    spec = f.robot_spec(1)
    schema = load_feature_schema(ROOT/'configs/bc_v1/features.json')
    cache = cls(spec, schema)
    graph = f.state_graph(2, 1)
    first = cache.get(graph)
    moved = replace(graph, modules=tuple(replace(m, position=(99., 3., 2.)) for m in graph.modules))
    assert cache.get(moved) is first
    changed = replace(graph, connectors=tuple(replace(p, status='connected') for p in graph.connectors))
    assert cache.get(changed) is not first
    assert cache.get(changed) == enumerate_candidates(changed, spec, schema)
    assert cache.get(f.state_graph(3, 1)) is not first


def test_prepare_smores_label_boundary_uses_raw_graph():
    cli = importlib.import_module('drl.IL.bc_v1.__main__')
    fn = getattr(cli, 'encode_smores_labels', None)
    assert callable(fn), 'Missing CLI Canonical-to-B2 label boundary'
    from drl.IL.bc_v1.smores_adapter import SMORESAdapter
    from drl.IL.tests.bc_v1.test_dataset_counts import row
    adapter = SMORESAdapter(c.load_robot_spec(ROOT/'configs/bc_v1/smores_ep.robot.json'))
    current = row('align_faces', wheels=True, internal='tilt_target')
    result = fn(current, adapter, ROOT)
    assert result == adapter.encode_labels(current['action_t'], current['supervision'], ())
    assert len(result.resource_targets) == 2 and len(result.structural_targets) == 1


def test_empty_resource_set_is_cacheable_when_skill_is_supervised(prepared_inputs):
    source, output, adapter, config = prepared_inputs
    class Empty(Adapter):
        def encode_state(self, raw):
            return replace(super().encode_state(raw), resources=())
        def encode_labels(self, action, supervision, evidence):
            return replace(super().encode_labels(action, supervision, evidence), resource_targets=())
    prepare((source, output, Empty(), config))
    sample = api().BCDataset(output, 'train')[0]
    assert not sample.inputs.policy.graph.resources and not sample.inputs.features.resources
    assert not sample.inputs.features.connectors and not sample.inputs.features.environment
    assert not sample.supervision.resource_targets and sample.supervision.structural_targets


def test_manifest_preserves_missing_metadata_identity(prepared_inputs):
    source, output, adapter, config = prepared_inputs
    absent = str(config['repo_root']/'missing-source.jsonl')
    config['metadata_reader'] = lambda row: {'sources': [], 'audit': [{'path': absent, 'missing': True}]}
    m = prepare(prepared_inputs)
    assert absent in m['external_metadata_hashes'], 'Missingness is a causal dependency too'
    assert m['external_metadata_hashes'][absent] is None


def test_prepare_detects_metadata_change_before_publication(prepared_inputs):
    source, output, adapter, config = prepared_inputs
    external = config['repo_root']/'physical-metadata.json'
    external.write_text('{}')
    digest = hashlib.sha256(external.read_bytes()).hexdigest()
    config['metadata_reader'] = lambda row: {'sources': [], 'audit': [{'path': str(external), 'sha256': digest}]}
    def progress(event):
        if event['completed'] == 3: external.write_text('{"changed":true}')
    config['progress'] = progress
    with pytest.raises(ValueError, match='metadata|dependency'):
        prepare(prepared_inputs)
    assert not output.exists() and not config['normalizer_output'].exists()


def test_prepare_detects_previously_scanned_source_change(prepared_inputs):
    source, output, adapter, config = prepared_inputs
    def progress(event):
        if event['completed'] == 3:
            p = source/'a.jsonl'
            p.write_bytes(p.read_bytes().replace(b'999', b'998'))
    config['progress'] = progress
    with pytest.raises(ValueError, match='Source|source'):
        prepare(prepared_inputs)
    assert not output.exists()


def test_dataset_automatic_config_stale_rejection(prepared_inputs):
    source, output, adapter, config = prepared_inputs
    path = config['repo_root']/'input-config.json'; path.write_text('{}')
    config['config_files'] = [path]
    prepare(prepared_inputs)
    path.write_text('{"changed": true}')
    with pytest.raises(ValueError, match='stale|config'):
        api().BCDataset(output, 'train')


def test_failed_physical_publication_leaves_no_cache_or_normalizer(prepared_inputs, monkeypatch):
    def fail(*args): raise OSError('injected publication failure')
    monkeypatch.setattr(api().os, 'rename', fail)
    with pytest.raises(OSError, match='publication failure'):
        prepare(prepared_inputs)
    assert not prepared_inputs[1].exists()
    assert not prepared_inputs[3]['normalizer_output'].exists(), 'Failed transaction leaked a new official normalizer'


def test_physical_prepare_never_touches_existing_normalizer(prepared_inputs):
    official = prepared_inputs[3]['normalizer_output']
    official.parent.mkdir(parents=True, exist_ok=True)
    official.write_bytes(b'existing F2 artifact')
    prepare(prepared_inputs)
    assert official.read_bytes() == b'existing F2 artifact'


def test_task_reader_records_absent_optional_dependencies(tmp_path):
    from drl.IL.bc_v1.task_context import read_task_metadata
    stream = tmp_path/'stream.jsonl'; stream.write_text('{"stamp":0}\n')
    row = {'graph_t': {'stamp': 0}, 'provenance': {'source_records': [
        {'_loader_source_path': str(stream), '_loader_source_row': 1}]}}
    result = read_task_metadata(row, tmp_path)
    absent = {item['path'] for item in result['audit'] if item.get('missing')}
    assert absent == {str(tmp_path/name) for name in ('stream_manifest.json', 'environment_geometry.json', 'course.json')}


def test_collision_at_publication_preserves_newly_existing_directory(prepared_inputs, monkeypatch):
    source, output, adapter, config = prepared_inputs
    original = api().os.rename
    def concurrent_publish(src, dst):
        Path(dst).mkdir()
        return original(src, dst)
    monkeypatch.setattr(api().os, 'rename', concurrent_publish)
    with pytest.raises(FileExistsError): prepare(prepared_inputs)
    # A reserved empty publication directory may be removed only by its owner.
    assert not (output/'manifest.json').exists()


def test_core_import_and_cli_help_without_hardware_or_torch():
    api()
    script = '''
import sys,importlib.abc,runpy
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if any(x in name for x in ('smores','isaac','omni','rclpy','torch')): raise AssertionError(name)
sys.meta_path.insert(0,Block())
import drl.IL.bc_v1.dataset
sys.argv=['bc_v1','--help']
runpy.run_module('drl.IL.bc_v1',run_name='__main__')
'''
    p = subprocess.run([sys.executable, '-c', script], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert 'prepare' in p.stdout


def test_indexed_task_reader_preserves_exact_current_semantics(tmp_path):
    from drl.IL.bc_v1.__main__ import _read_indexed_source_records
    from drl.IL.bc_v1.task_context import read_task_metadata
    path = tmp_path/'stream.jsonl'
    records = [{'stamp': 0., 'task_type': 'assembly'}, {'stamp': 1., 'task_type': 'gap'},
               {'stamp': 2., 'task_type': 'stairs'}]
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    row = {'graph_t': {'stamp': 1.}, 'provenance': {'source_records': [
        {'_loader_source_path': str(path), '_loader_source_row': 2}]}}
    calls = []
    def reader(p, indices):
        calls.append((p, indices))
        return [(i, records[i-1]) for i in indices]
    expected = read_task_metadata(row, tmp_path)
    assert 'record_reader' in __import__('inspect').signature(read_task_metadata).parameters, 'Missing indexed causal read hook'
    assert read_task_metadata(row, tmp_path, record_reader=reader) == expected
    assert calls == [(path, (2,))]
    with pytest.raises(ValueError, match='stamp'):
        read_task_metadata(row, tmp_path, record_reader=lambda p, indices: [(2, records[2])])
    assert read_task_metadata(row, tmp_path, record_reader=_read_indexed_source_records) == expected
    with pytest.raises(ValueError, match='stamp'):
        read_task_metadata({**row, 'graph_t': {'stamp': 2.}}, tmp_path,
            record_reader=_read_indexed_source_records)
    for index in (0, True, 4):
        invalid = {**row, 'provenance': {'source_records': [
            {'_loader_source_path': str(path), '_loader_source_row': index}]}}
        with pytest.raises(ValueError):
            read_task_metadata(invalid, tmp_path, record_reader=_read_indexed_source_records)
    outside = {**row, 'provenance': {'source_records': [
        {'_loader_source_path': str(tmp_path.parent/'outside.jsonl'), '_loader_source_row': 2}]}}
    with pytest.raises(ValueError, match='escapes repo_root'):
        read_task_metadata(outside, tmp_path, record_reader=_read_indexed_source_records)


def derivation_api():
    derive = getattr(api(), 'derive_physical_inputs', None)
    assert callable(derive), 'Missing Task 4 derive_physical_inputs interface'
    return derive


def fixture_row(data):
    return json.loads((data[0]/'a.jsonl').read_text().splitlines()[0])


def derive_one(row, adapter, config):
    return derivation_api()(row, adapter, config,
        api().CurrentCandidateCache(config['spec'], config['schema']))


def test_derive_orders_adapter_topology_d1_d2_bridges_e1_e2(prepared_inputs, monkeypatch):
    derive = derivation_api()
    m = api(); config = dict(prepared_inputs[3]); events = []
    class OrderedAdapter(Adapter):
        def encode_state(self, raw):
            events.append('adapter'); return super().encode_state(raw)
    def traced(name, fn):
        def call(*args, **kwargs):
            events.append(name); return fn(*args, **kwargs)
        return call
    for name, attr in [('topology', 'reconcile_topology'), ('D1', 'build_task_view'),
                       ('D2', 'select_policy_inputs'), ('E1', 'transform_inputs'),
                       ('E2', 'extract_features')]:
        monkeypatch.setattr(m, attr, traced(name, getattr(m, attr)))
    config['frame_source_reader'] = traced('source_reader', config['frame_source_reader'])
    provider = config['source_frame_provider']
    monkeypatch.setattr(provider, 'resolve', traced('bridges', provider.resolve))
    cache = m.CurrentCandidateCache(config['spec'], config['schema'])
    monkeypatch.setattr(cache, 'get', traced('candidates', cache.get))
    result = derive(fixture_row(prepared_inputs), OrderedAdapter(), config, cache)
    assert events == ['adapter', 'topology', 'D1', 'D2', 'source_reader', 'bridges', 'E1', 'E2', 'candidates']
    assert result.features is result.inputs.features
    assert result.diagnostics == ()
    with pytest.raises((AttributeError, TypeError)):
        result.features = None


def test_external_canonical_fixture_absent_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], 'ROOT', tmp_path)
    with pytest.raises(pytest.skip.Exception, match='Canonical v6 external fixture is not installed'):
        real_row('example', 0)


def test_external_canonical_fixture_present_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], 'ROOT', tmp_path)
    source = tmp_path/'datasets/expert_v1_canonical_v6'
    source.mkdir(parents=True)
    (source/'manifest.json').write_text(json.dumps({
        'episodes': [{'episode_id': 'example', 'file': 'episode.jsonl'}]}))
    (source/'episode.jsonl').write_text('{"current": 1}\n{"current": 2}\n')
    assert real_row('example', 1) == {'current': 2}


@pytest.mark.parametrize('content,error', [
    ('{malformed', json.JSONDecodeError),
    ('{}', KeyError),
    ('{"episodes": []}', StopIteration),
])
def test_external_canonical_fixture_invalid_present_fails(tmp_path, monkeypatch, content, error):
    monkeypatch.setattr(sys.modules[__name__], 'ROOT', tmp_path)
    source = tmp_path/'datasets/expert_v1_canonical_v6'
    source.mkdir(parents=True)
    (source/'manifest.json').write_text(content)
    with pytest.raises(error):
        real_row('example', 0)


@pytest.mark.parametrize('kind', ['directory', 'dangling_symlink'])
def test_external_canonical_fixture_invalid_path_fails(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(sys.modules[__name__], 'ROOT', tmp_path)
    manifest = tmp_path/'datasets/expert_v1_canonical_v6/manifest.json'
    manifest.parent.mkdir(parents=True)
    if kind == 'directory':
        manifest.mkdir()
        error = IsADirectoryError
    else:
        manifest.symlink_to('missing.json')
        error = FileNotFoundError
    with pytest.raises(error):
        real_row('example', 0)


def real_row(episode_id, index):
    import itertools
    f.require_canonical_v6_fixture(ROOT)
    source = ROOT/'datasets/expert_v1_canonical_v6'
    manifest = json.loads((source/'manifest.json').read_text())
    entry = next(e for e in manifest['episodes'] if e['episode_id'] == episode_id)
    with (source/entry['file']).open() as stream:
        return json.loads(next(itertools.islice(stream, index, index+1)))


def real_derivation_config():
    from drl.IL.bc_v1.smores_adapter import SMORESAdapter
    from drl.IL.bc_v1.smores_source_frames import SmoresSourceFrameProvider, read_frame_source_context
    from drl.IL.bc_v1.task_context import read_task_metadata
    spec = c.load_robot_spec(ROOT/'configs/bc_v1/smores_ep.robot.json')
    policy = json.loads((ROOT/'configs/bc_v1/source_frames.json').read_text())
    config = dict(repo_root=ROOT, spec=spec,
        schema=load_feature_schema(ROOT/'configs/bc_v1/features.json'),
        task_config=json.loads((ROOT/'configs/bc_v1/task_context.json').read_text()),
        dataset_identity={'manifest': 'datasets/expert_v1_canonical_v6/manifest.json',
            'manifest_sha256': api().file_hash(ROOT/'datasets/expert_v1_canonical_v6/manifest.json')},
        metadata_reader=lambda row: read_task_metadata(row, ROOT),
        # These two tests isolate real frame certification, not historical run certification.
        topology_evidence_provider=lambda provenance, graph: None,
        source_frame_provider=SmoresSourceFrameProvider(ROOT, policy),
        frame_source_reader=lambda provenance, metadata: read_frame_source_context(
            provenance, metadata, repo_root=ROOT, policy=policy))
    return SMORESAdapter(spec), config


def test_real_1725_derivation_certifies_stage_local():
    row = real_row('teleop-0e4e34fe3a5f4ed1b6a86642acf4fac8', 1725)
    before = copy.deepcopy(row)
    adapter, config = real_derivation_config()
    result = derive_one(row, adapter, config)
    bridge = next(x for x in result.bridges.resolutions if x.source_frame == 'stage_local')
    assert bridge.certified and bridge.evidence
    assert bridge.transform.translation == pytest.approx((1.7699752221047802, -2.17605937116128, 0.))
    plane = next(q for q in result.geometry.quantities if q.source_name == 'stage_local.stairs.first_riser_x_m')
    assert plane.valid and plane.kind == 'planes'
    assert plane.values == pytest.approx((0., -1., 0.,
        1.4760593711612797 + result.geometry.frame.origin[1]), abs=1e-9)
    module = result.published_graph.modules[0]
    assert result.geometry.modules[0].position == pytest.approx(tuple(
        p - o for p, o in zip(module.position, result.geometry.frame.origin)))
    assert result.features is result.inputs.features
    assert row == before and 'dataset_identity' not in row['provenance']


def test_real_476_derivation_certifies_legacy_map():
    row = real_row('expert:rc_car:seed-005100', 476)
    adapter, config = real_derivation_config()
    result = derive_one(row, adapter, config)
    bridge = next(x for x in result.bridges.resolutions if x.source_frame == 'map')
    assert bridge.certified and bridge.evidence
    assert bridge.transform.translation == (0., 0., 0.)
    assert bridge.transform.rotation == (1., 0., 0., 0., 1., 0., 0., 0., 1.)
    assert len(result.features.relations) == 24
    assert all(q.valid for q in result.geometry.quantities if q.source_name.startswith('map.'))


def test_composed_1189_keeps_both_targets_and_7_to_6_edges(tmp_path, prepared_inputs):
    from drl.IL.tests.bc_v1.test_smores_disconnect_evidence import real_minimal, catalog_for
    from drl.IL.bc_v1.smores_adapter import SMORESAdapter
    from drl.IL.bc_v1.__main__ import compose_runtime_providers
    identity, _, assets = runtime_assets(prepared_inputs, tmp_path)
    frozen, refs = real_minimal(tmp_path)
    Path(assets['topology']['source_catalog']).write_text(json.dumps(catalog_for(refs, identity, complete=True)))
    runtime = compose_runtime_providers(tmp_path, identity, assets,
        prepared_inputs[3]['split_manifest']['episode_metadata'])
    spec = c.load_robot_spec(ROOT/'configs/bc_v1/smores_ep.robot.json')
    task_config = json.loads((ROOT/'configs/bc_v1/task_context.json').read_text())
    task_config['historical_mission_imputations'] = []  # This fixture isolates topology; no course sidecars.
    config = dict(runtime, spec=spec, schema=load_feature_schema(ROOT/'configs/bc_v1/features.json'),
        task_config=task_config,
        metadata_reader=lambda row: {'sources': [], 'audit': []})
    row = {'graph_t': frozen['published_graph'], 'provenance': {
        'logical_episode_id': 'teleop-39e2f4b4cd684e9ea72734cc21baca94',
        'canonical_transition_index': 1189, 'source_records': [
        {'_loader_source_path': str(tmp_path/x['path']), '_loader_source_row': x['record']}
        for x in refs['source_records']]},
           'graph_t_plus_1': {'forbidden': 'future'}}
    result = derive_one(row, SMORESAdapter(spec), config)
    ref = lambda mid, cid: c.ConnectorRef(c.ModuleRef(mid), cid)
    targets = c.SupervisionBundle((), (
        c.StructuralTarget(c.ModuleRef('smores_05'), 'undock',
            (c.ModuleRef('smores_05'), c.ModuleRef('smores_06')), connectors=(
            ref('smores_05', 'BOTTOM'), ref('smores_06', 'LEFT'))),
        c.StructuralTarget(c.ModuleRef('smores_02'), 'align_faces',
            (c.ModuleRef('smores_02'), c.ModuleRef('smores_01')), connectors=(
            ref('smores_02', 'BOTTOM'), ref('smores_01', 'TOP')))))
    api()._representable(targets, result.inputs.candidates)
    assert len(targets.structural_targets) == 2
    assert len(result.published_graph.connections) == 7
    assert len(result.inputs.policy.graph.connections) == 6
    assert result.published_graph.modules == result.inputs.policy.graph.modules
    audit = {k: json.loads(v) for k, v in result.topology_audit}
    assert audit['admitted'][0]['dispatch_record'] == 227
    assert audit['deferred'][0]['dispatch_record'] == 228
    assert audit['deferred'][0]['reason'] == 'current_bucket_dispatch'


def test_derive_extracts_e2_once_and_candidates_use_reconciled_graph(prepared_inputs, monkeypatch):
    from drl.IL.bc_v1.causal_topology import CausalBoundary, ConfirmedDisconnect
    from drl.IL.bc_v1.smores_disconnect_evidence import DisconnectEvidenceBatch
    m = api(); derivation_api(); config = dict(prepared_inputs[3]); graph = f.state_graph(2, 1)
    a, b = (x.ref for x in graph.connectors)
    published = replace(graph, connectors=(replace(graph.connectors[0], status='connected', connected_to=b),
        replace(graph.connectors[1], status='connected', connected_to=a)),
        connections=(c.ConnectionState(a, b, 'current_connection'),))
    config['spec'] = replace(config['spec'], module_types=f.robot_spec(1).module_types)
    class PublishedAdapter:
        def encode_state(self, raw): return published
    event = ConfirmedDisconnect(published.connections[0].endpoint_a,
        published.connections[0].endpoint_b, 'physical-prior-detach', 'run', 1)
    config['topology_evidence_provider'] = lambda provenance, g: DisconnectEvidenceBatch(
        (event,), CausalBoundary('run', 2), {'proof': 'prior-physical-completion'})
    calls = []
    original = m.extract_features
    def features(*args):
        calls.append(args[0].graph); return original(*args)
    monkeypatch.setattr(m, 'extract_features', features)
    class Candidates:
        def get(self, reconciled):
            assert not reconciled.connections
            assert reconciled == calls[0]
            return 'candidate-sentinel'
    result = derivation_api()(fixture_row(prepared_inputs), PublishedAdapter(), config, Candidates())
    assert calls == [result.inputs.policy.graph]
    assert result.inputs.candidates == 'candidate-sentinel'
    assert result.published_graph is published and result.features is result.inputs.features


def test_runtime_composition_adds_dataset_identity_and_current_task_geometry_without_policy_leakage(prepared_inputs, monkeypatch):
    from drl.IL.bc_v1 import dataset as m
    from drl.IL.bc_v1.source_frame_bridges import BridgeResolution
    row = fixture_row(prepared_inputs); derivation_api()
    row['provenance'].update(logical_episode_id='a', canonical_transition_index=42, physical_stamp=0., source_records=[{
        '_loader_source_path': '/exact/source', '_loader_source_row': 1,
        '_loader_source_kind': 'structural', 'action_t': 'poison'}],
        stage='poison', control_state='poison', FSM='poison', supervision='poison',
        action_t='poison', graph_t_plus_1='poison', target_morphology='poison',
        dataset_identity={'manifest': 'untrusted'})
    before = copy.deepcopy(row)
    config = dict(prepared_inputs[3], dataset_identity={'manifest': 'datasets/canonical/manifest.json',
        'manifest_sha256': 'a'*64})
    provider = config['source_frame_provider']
    resolve = provider.resolve
    def explicit_unresolved(inputs, source):
        batch = resolve(inputs, source)
        return replace(batch, resolutions=batch.resolutions + (BridgeResolution(
            'stage_local', 'world', 'current', None, False,
            'isolated routing fixture has no certified stage-local source', None, ()),))
    monkeypatch.setattr(provider, 'resolve', explicit_unresolved)
    context = c.TaskContext('stairs', (c.NumericFeature('world.task_frame.origin_xyz_m', (1., 2., 0.)),
        c.NumericFeature('world.task_frame.yaw_rad', (0.,)),
        c.NumericFeature('stage_local.stairs.first_riser_x_m', (-.7,)),
        c.NumericFeature('stairs.height_m', (.1,))))
    monkeypatch.setattr(m, 'build_task_view', lambda *args: (context, c.ObservationBundle()))
    seen = []
    def topology(provenance, graph):
        seen.append(provenance)
        assert set(provenance) == {'logical_episode_id', 'canonical_transition_index', 'physical_stamp', 'source_records', 'dataset_identity'}
        assert provenance['logical_episode_id'] == 'a' and provenance['canonical_transition_index'] == 42
        assert provenance['physical_stamp'] == 0.
        assert set(provenance['source_records'][0]) == {'_loader_source_path', '_loader_source_row', '_loader_source_kind'}
        assert provenance['dataset_identity'] == config['dataset_identity']
        return None
    config['topology_evidence_provider'] = topology
    original = config['frame_source_reader']
    def frame(provenance, metadata):
        assert provenance == seen[0]
        assert set(metadata) == {'current_task_geometry', 'audit'}
        assert metadata['audit'] == ()
        assert [x['name'] for x in metadata['current_task_geometry']] == [x.name for x in context.static_geometry[:3]]
        assert all(set(x) == {'name', 'values', 'valid'} for x in metadata['current_task_geometry'])
        return original(provenance, metadata)
    config['frame_source_reader'] = frame
    result = derive_one(row, prepared_inputs[2], config)
    assert 'dataset_identity' not in repr(result.inputs)
    assert 'current_task_geometry' not in repr(result.inputs)
    assert row == before


def test_prepare_passes_published_graph_to_label_codec(prepared_inputs, monkeypatch):
    m = api(); derivation_api(); original = m.derive_physical_inputs; derived = []
    def collect(*args):
        result = original(*args); derived.append(result); return result
    monkeypatch.setattr(m, 'derive_physical_inputs', collect)
    def labels(row, graph):
        assert graph is derived[-1].published_graph
        assert derived[-1].features is derived[-1].inputs.features
        return prepared_inputs[2].encode_labels(row['action_t'], row['supervision'], ())
    prepared_inputs[3]['label_reader'] = labels
    prepare(prepared_inputs)
    assert len(derived) == 6


def test_missing_topology_or_frame_provider_is_not_a_silent_fallback(prepared_inputs):
    for key in ('topology_evidence_provider', 'source_frame_provider', 'frame_source_reader'):
        config = dict(prepared_inputs[3]); del config[key]
        with pytest.raises((ValueError, KeyError), match=key):
            derive_one(fixture_row(prepared_inputs), prepared_inputs[2], config)


def runtime_assets(data, tmp_path, *, dispatch=False):
    from drl.IL.bc_v1.temporal_supervision import TemporalSupervisionAudit, write_temporal_audit
    from drl.IL.tests.bc_v1.test_smores_disconnect_evidence import simple, catalog_for
    source, _, _, config = data
    identity = {'manifest': str((source/'manifest.json').relative_to(tmp_path)),
                'manifest_sha256': api().file_hash(source/'manifest.json')}
    catalog = catalog_for(simple(tmp_path), identity, complete=True)
    catalog_path = tmp_path/'causal_sources.json'; catalog_path.write_text(json.dumps(catalog))
    audit_path = tmp_path/'temporal.json'
    entries = tuple(TemporalSupervisionAudit(e, i, 'single_decision')
                    for e in ('a', 'b', 'c') for i in (0, 2))
    write_temporal_audit(audit_path, entries, identity,
                         expected_keys={(e.episode_id, e.transition_index) for e in entries})
    assets = {'topology': {'policy': str(ROOT/'configs/bc_v1/causal_topology.json'),
                          'source_catalog': str(catalog_path)},
              'source_frames': {'policy': str(ROOT/'configs/bc_v1/source_frames.json')},
              'temporal_audit': str(audit_path)}
    return identity, catalog, assets


def test_cli_composes_serializable_assets_into_runtime_providers(prepared_inputs, tmp_path):
    from drl.IL.bc_v1 import __main__ as cli
    identity, catalog, assets = runtime_assets(prepared_inputs, tmp_path)
    compose = getattr(cli, 'compose_runtime_providers', None)
    assert callable(compose), 'Missing Task 4 CLI runtime provider composition'
    runtime = compose(tmp_path, identity, assets, prepared_inputs[3]['split_manifest']['episode_metadata'])
    assert json.loads(json.dumps(assets)) == assets
    assert runtime['dataset_identity'] == identity
    assert callable(runtime['topology_evidence_provider']) and callable(runtime['frame_source_reader'])
    assert callable(runtime['source_frame_provider'].resolve)
    assert len(runtime['temporal_audit']) == 6
    provenance = {'dataset_identity': identity, 'logical_episode_id': 'a',
        'canonical_transition_index': 0, 'source_records': [{
        '_loader_source_path': str(tmp_path/'dispatch.jsonl'), '_loader_source_row': 1}]}
    assert runtime['topology_evidence_provider'](provenance, f.state_graph(2, 0)) is None
    assert 'topology_evidence_provider' not in assets


def test_cli_late_source_receipts_index_once_without_prefix_rescans(prepared_inputs, tmp_path, monkeypatch):
    """Production frame certification must not reread thousands of prior rows."""
    from drl.IL.bc_v1.__main__ import compose_runtime_providers
    from drl.IL.bc_v1.smores_source_frames import STAGE_LOCAL_CONTRACT_ID
    from drl.IL.bc_v1.task_context import read_task_metadata
    identity, _, assets = runtime_assets(prepared_inputs, tmp_path)
    course = {'frame_id': 'world', 'geometry_sha256': 'geometry', 'mission': {
        'schema_version': 'mssr.composite_mission.v1', 'tasks': [{'parameters': {
            'stage_frame': {'origin_world_xyz_m': [1., 2., 0.], 'yaw_rad': .5},
            'geometry_frame': 'world', 'gap': {'coordinate_frame': 'stage_local', 'width_m': .4}}}]}}
    directory = tmp_path/'teleop'/'episodes'/'a'
    directory.mkdir(parents=True)
    source = directory/'stream.jsonl'
    source.write_text(''.join(json.dumps({'schema_version': 'mssr.expert_transition.v3',
        'graph_t': {'stamp': float(i)}, 'observation': {'schema_version': 'mssr.module_states.v2',
        'robot_family': 'smores_ep', 'stamp': float(i), 'course': course}})+'\n' for i in range(4000)))
    (directory/'course.json').write_text(json.dumps(course))
    (directory/'manifest.json').write_text(json.dumps({'schema_version': 'mssr.curated_teleop_episode.v1',
        'episode_id': 'a', 'geometry_sha256': 'geometry', 'datasets': [{
            'path': 'episodes/a/stream.jsonl', 'source_path': 'recording.jsonl', 'samples': 4000}]}))
    (directory/'source_manifest.json').write_text(json.dumps({
        'schema_version': 'mssr.teleop_recording_manifest.v1', 'episode_id': 'a',
        'structural_streams': [{'path': 'recording.jsonl'}]}))
    frame_policy = {'current_stage_local': {'contract_id': STAGE_LOCAL_CONTRACT_ID,
        'revision': '1.0.0', 'geometry_convention': 'world_z_planar_m_rad',
        'canonical_manifest': {'path': identity['manifest'], 'sha256': identity['manifest_sha256']},
        'source_root': 'teleop', 'source_schema': 'mssr.expert_transition.v3',
        'observation_schema': 'mssr.module_states.v2', 'mission_schema': 'mssr.composite_mission.v1'}}
    policy_path = tmp_path/'frames.json'; policy_path.write_text(json.dumps(frame_policy))
    assets['source_frames']['policy'] = str(policy_path)
    runtime = compose_runtime_providers(tmp_path, identity, assets,
        prepared_inputs[3]['split_manifest']['episode_metadata'])
    original_open = Path.open
    reads = []
    class CountedStream:
        def __init__(self, stream): self.stream = stream
        def __enter__(self): return self
        def __exit__(self, *args): return self.stream.__exit__(*args)
        def __iter__(self): return self
        def __next__(self):
            line = next(self.stream); reads.append(1); return line
        def readline(self, *args):
            line = self.stream.readline(*args)
            if line: reads.append(1)
            return line
        def __getattr__(self, name): return getattr(self.stream, name)
    def counted_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return CountedStream(stream) if path == source else stream
    # Build the same D1 audit/geometry handed to the production frame reader.
    # Instrument only subsequent certification accesses, not fixture setup.
    row = {'graph_t': {'stamp': 3899.}, 'provenance': {'source_records': [{
        '_loader_source_path': str(source), '_loader_source_row': 3900}]}}
    metadata = read_task_metadata(row, tmp_path)
    metadata['current_task_geometry'] = [
        {'name': 'world.task_frame.origin_xyz_m', 'values': [1., 2., 0.], 'valid': True},
        {'name': 'world.task_frame.yaw_rad', 'values': [.5], 'valid': True},
        {'name': 'stage_local.gap.width_m', 'values': [.4], 'valid': True}]
    monkeypatch.setattr(Path, 'open', counted_open)
    receipts = []
    for line in (3900, 3901, 3900, 3901):
        provenance = {**row['provenance'], 'dataset_identity': identity,
            'logical_episode_id': 'a', 'physical_stamp': float(line-1),
            'source_records': [{'_loader_source_path': str(source), '_loader_source_row': line}]}
        receipt = runtime['frame_source_reader'](provenance, metadata)
        assert [c.contract_id for c in receipt.certificates] == [STAGE_LOCAL_CONTRACT_ID]
        assert any(f'#L{line}:current-frame' in e.path for e in receipt.certificates[0].evidence)
        receipts.append(receipt)
    assert receipts[0] == receipts[2] and receipts[1] == receipts[3]
    assert len(reads) <= 4000 + 4, f'Repeated source prefix scans: {len(reads)} row reads'
    from drl.IL.bc_v1.smores_source_frames import read_frame_source_context
    assert receipts[-1] == read_frame_source_context(provenance, metadata,
        repo_root=tmp_path, policy=frame_policy)
    for bad in (
            {**provenance, 'physical_stamp': 3899.},
            {**provenance, 'source_records': [{'_loader_source_path': str(source), '_loader_source_row': 4001}]},
            {**provenance, 'source_records': [{'_loader_source_path': str(directory/'other.jsonl'), '_loader_source_row': 3901}]}):
        assert not runtime['frame_source_reader'](bad, metadata).certificates


def test_cli_catalog_missing_or_foreign_source_fails_closed(prepared_inputs, tmp_path):
    from drl.IL.bc_v1 import __main__ as cli
    compose = getattr(cli, 'compose_runtime_providers', None)
    assert callable(compose), 'Missing Task 4 CLI runtime provider composition'
    identity, catalog, assets = runtime_assets(prepared_inputs, tmp_path)
    foreign = dict(identity, manifest_sha256='f'*64)
    with pytest.raises(ValueError, match='source|identity'):
        compose(tmp_path, foreign, assets, prepared_inputs[3]['split_manifest']['episode_metadata'])
    Path(assets['topology']['source_catalog']).unlink()
    with pytest.raises(ValueError, match='certified.*artifact|artifact.*missing'):
        compose(tmp_path, identity, assets, prepared_inputs[3]['split_manifest']['episode_metadata'])


@pytest.mark.parametrize('fault', ['extra', 'missing'])
def test_sparse_catalog_rejects_21st_case_or_missing_approved_case(prepared_inputs, tmp_path, fault):
    from drl.IL.bc_v1.__main__ import compose_runtime_providers
    identity, catalog, assets = runtime_assets(prepared_inputs, tmp_path)
    if fault == 'extra':
        value = copy.deepcopy(next(iter(catalog['cases'].values())))
        value.update(episode_id='unreviewed', transition_index=0, run_id='unreviewed')
        catalog['cases']['unreviewed::0'] = value
    else:
        catalog['cases'].pop(next(iter(catalog['cases'])))
    Path(assets['topology']['source_catalog']).write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match='approved|20|case'):
        compose_runtime_providers(tmp_path, identity, assets,
            prepared_inputs[3]['split_manifest']['episode_metadata'])


def test_sparse_catalog_rejects_duplicate_json_key(prepared_inputs, tmp_path):
    from drl.IL.bc_v1.__main__ import compose_runtime_providers
    identity, catalog, assets = runtime_assets(prepared_inputs, tmp_path)
    key = next(iter(catalog['cases']))
    payload = json.dumps(catalog)
    marker = '"cases": {'
    payload = payload.replace(marker, marker + json.dumps(key) + ': ' + json.dumps(catalog['cases'][key]) + ', ', 1)
    Path(assets['topology']['source_catalog']).write_text(payload)
    with pytest.raises(ValueError, match='[Dd]uplicate'):
        compose_runtime_providers(tmp_path, identity, assets,
            prepared_inputs[3]['split_manifest']['episode_metadata'])


def test_prepare_temporal_keys_are_checked_without_changing_targets(prepared_inputs):
    from drl.IL.bc_v1.temporal_supervision import TemporalSupervisionAudit
    config = prepared_inputs[3]
    config['temporal_audit'] = tuple(TemporalSupervisionAudit(e, i,
        'temporally_coalesced_sequential' if i == 0 else 'ambiguous')
        for e in ('a', 'b', 'c') for i in (0, 2))
    prepare(prepared_inputs)
    sample = api().BCDataset(prepared_inputs[1], 'train')[0]
    assert sample.supervision == prepared_inputs[2].encode_labels({}, {}, ())
    assert 'temporally_coalesced' not in repr(sample.inputs)


def test_prepare_rejects_foreign_temporal_key(prepared_inputs):
    from drl.IL.bc_v1.temporal_supervision import TemporalSupervisionAudit
    prepared_inputs[3]['temporal_audit'] = (TemporalSupervisionAudit('foreign', 0, 'single_decision'),)
    with pytest.raises(ValueError, match='temporal|Temporal'):
        prepare(prepared_inputs)
    assert not prepared_inputs[1].exists()


def test_cache_identity_contains_frame_provider_semantics(prepared_inputs):
    manifest = prepare(prepared_inputs)
    assert manifest['identity'].get('source_frame_provider') == prepared_inputs[3]['source_frame_provider'].identity()
    incompatible = dict(manifest['identity'], source_frame_provider={'provider_id': 'different', 'revision': '2'})
    with pytest.raises(ValueError, match='stale|incompatible'):
        api().BCDataset(prepared_inputs[1], 'train', expected_identity=incompatible)


@pytest.mark.parametrize('setting', ['absent', 'null'])
def test_cli_composition_allows_absent_temporal_audit_without_changing_physical_runtime(prepared_inputs, tmp_path, monkeypatch, setting):
    from drl.IL.bc_v1 import __main__ as cli
    from drl.IL.bc_v1 import dataset
    identity, _, assets = runtime_assets(prepared_inputs, tmp_path)
    episodes = prepared_inputs[3]['split_manifest']['episode_metadata']
    with_audit = cli.compose_runtime_providers(tmp_path, identity, assets, episodes)
    optional_assets = copy.deepcopy(assets)
    if setting == 'absent':
        del optional_assets['temporal_audit']
    else:
        optional_assets['temporal_audit'] = None
    without_audit = cli.compose_runtime_providers(tmp_path, identity, optional_assets, episodes)
    assert without_audit['temporal_audit'] is None
    assert callable(without_audit['topology_evidence_provider'])
    assert callable(without_audit['source_frame_provider'].resolve)
    assert callable(without_audit['frame_source_reader'])
    assert Path(assets['temporal_audit']) not in without_audit['provider_config_files']
    assert len(without_audit['provider_config_files']) == 3
    row = fixture_row(prepared_inputs)
    row['provenance'].update(logical_episode_id='a', canonical_transition_index=0)
    row['provenance']['source_records'] = [{
        '_loader_source_path': str(tmp_path/'dispatch.jsonl'), '_loader_source_row': 1}]
    monkeypatch.setattr(dataset, 'build_task_view', lambda *args: (
        c.TaskContext('navigation', (c.NumericFeature('world.goal.center_xyz_m', (2., 3., 0.)),)),
        c.ObservationBundle()))
    derived = []
    targets = []
    for runtime in (with_audit, without_audit):
        config = dict(prepared_inputs[3]); config.update(runtime)
        result = derive_one(row, prepared_inputs[2], config)
        target = prepared_inputs[2].encode_labels(row['action_t'], row['supervision'], ())
        dataset._representable(target, result.inputs.candidates)
        derived.append(result); targets.append(target)
    assert derived[0] == derived[1]
    assert derived[0].inputs.features.environment and derived[0].inputs.features.relations
    assert targets[0] == targets[1]
    # Required physical providers do not become optional with the temporal audit.
    for section, key in (('topology', 'source_catalog'), ('source_frames', 'policy')):
        broken = copy.deepcopy(optional_assets)
        broken[section][key] = str(tmp_path/'missing-required.json')
        with pytest.raises(ValueError, match='artifact missing'):
            cli.compose_runtime_providers(tmp_path, identity, broken, episodes)


@pytest.mark.parametrize('fault', ['missing', 'foreign_source', 'enum', 'count', 'range'])
def test_cli_configured_temporal_audit_remains_strict(prepared_inputs, tmp_path, fault):
    from drl.IL.bc_v1 import __main__ as cli
    from drl.IL.bc_v1.temporal_supervision import TemporalSupervisionAudit, write_temporal_audit
    identity, _, assets = runtime_assets(prepared_inputs, tmp_path)
    path = Path(assets['temporal_audit'])
    if fault == 'missing':
        path.unlink()
    elif fault == 'enum':
        document = json.loads(path.read_text())
        document['entries'][0]['kind'] = 'unsupported-kind'
        path.write_text(json.dumps(document))
    else:
        entries = [TemporalSupervisionAudit(e, i, 'single_decision')
                   for e in ('a', 'b', 'c') for i in (0, 2)]
        source = dict(identity)
        if fault == 'foreign_source':
            source['manifest_sha256'] = 'f'*64
        elif fault == 'count':
            entries.pop()
        else:
            entries[0] = TemporalSupervisionAudit('a', 3, 'single_decision')
        write_temporal_audit(path, entries, source,
            expected_keys={(e.episode_id, e.transition_index) for e in entries})
    with pytest.raises(ValueError):
        cli.compose_runtime_providers(tmp_path, identity, assets,
                                     prepared_inputs[3]['split_manifest']['episode_metadata'])


def test_prepare_never_fits_or_writes_official_normalizer(prepared_inputs, monkeypatch):
    from drl.IL.bc_v1 import normalization
    def forbidden(*args, **kwargs):
        pytest.fail('Physical prepare called F2 fit/write')
    for name in ('fit_normalizers', 'write_normalizers'):
        monkeypatch.setattr(normalization, name, forbidden)
        monkeypatch.setattr(api(), name, forbidden, raising=False)
    manifest = prepare(prepared_inputs)
    assert manifest['complete'] is True and manifest['normalized'] is False
    assert manifest['counts']['samples'] == 6
    assert 'normalizer' not in manifest
    assert 'normalizer_fingerprint' not in manifest['identity']
    assert not (prepared_inputs[1]/'normalizers.json').exists()
    assert not (prepared_inputs[1]/'fit_context.json').exists()
    assert not prepared_inputs[3]['normalizer_output'].exists()
    assert len(api().BCDataset(prepared_inputs[1], 'train')) == 2


def test_physical_prepare_requires_no_normalizer_path(prepared_inputs):
    source, output, adapter, config = prepared_inputs
    config = dict(config); del config['normalizer_output']
    prepare((source, output, adapter, config))
    assert len(api().BCDataset(output, 'test')) == 2


def test_physical_reader_explicitly_rejects_old_v1_cache(prepared_inputs):
    prepare(prepared_inputs)
    path = prepared_inputs[1]/'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['schema_version'] = 'mssr.bc_v1.physical_cache.v1'
    manifest['manifest_fingerprint'] = api()._manifest_fingerprint(manifest)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='schema'):
        api().BCDataset(prepared_inputs[1], 'train')
