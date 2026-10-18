"""Explicit train-only F2 lifecycle on tiny complete physical G1 caches."""
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from drl.IL.bc_v1 import normalization
from drl.IL.tests.bc_v1.test_dataset import prepared_inputs, prepare, api as dataset_api, ROOT


def api():
    name = 'drl.IL.bc_v1.normalization_prepare'
    assert importlib.util.find_spec(name), 'Missing separate F2 cache fitter'
    return importlib.import_module(name)


def args(data, output):
    manifest = dataset_api().read_cache_manifest(data[1])
    config = data[3]
    return dict(cache_root=data[1], output_dir=output, split_manifest=config['split_manifest'],
        spec=config['spec'], schema=config['schema'], policy=config['normalization_policy'],
        expected_identity=manifest['identity'])


def cache_bytes(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_separate_fitter_opens_only_train_and_keeps_g1_manifest_immutable(prepared_inputs, tmp_path, monkeypatch):
    fitter = api()
    prepare(prepared_inputs)
    original_bytes = cache_bytes(prepared_inputs[1])
    manifest = dataset_api().read_cache_manifest(prepared_inputs[1])
    expected_train = sorted(prepared_inputs[3]['split_manifest']['splits']['train'])
    opened = []; factories = []
    load = dataset_api()._load_episode
    def train_only(root, eid, entry):
        assert eid in expected_train, 'Fitter opened a validation/test cache episode'
        opened.append(eid)
        return load(root, eid, entry)
    monkeypatch.setattr(dataset_api(), '_load_episode', train_only)
    fit = fitter.fit_normalizers
    def observe(episodes, *a, **kw):
        factories.extend(sorted(episodes))
        assert all(callable(factory) for factory in episodes.values())
        return fit(episodes, *a, **kw)
    monkeypatch.setattr(fitter, 'fit_normalizers', observe)
    output = tmp_path/'f2'
    context = fitter.fit_prepared_normalizers(**args(prepared_inputs, output))
    assert opened == expected_train == factories
    assert cache_bytes(prepared_inputs[1]) == original_bytes
    assert set(p.name for p in output.iterdir()) == {'normalizers.json', 'fit_context.json'}
    assert context == json.loads((output/'fit_context.json').read_text())
    assert context['g1_cache_fingerprint'] == manifest['cache_fingerprint']
    assert context['source_identity'] == manifest['source_identity']
    assert context['split_fingerprint'] == manifest['identity']['split_fingerprint']
    for key in ('robot_spec_fingerprint', 'feature_schema_fingerprint', 'normalization_policy_fingerprint'):
        assert context[key] == manifest['identity'][key]
    assert context['train_episode_ids'] == expected_train
    assert context['train_episode_count'] == 1 and context['train_sample_count'] == 2
    assert context['complete_train_coverage'] is True
    assert context['sample_counts'] == [[expected_train[0], 2]]
    stats = normalization.read_normalizers(output/'normalizers.json',
                                          expected_metadata=context['normalizer_expected_metadata'])
    assert normalization.normalizer_fingerprint(stats) == context['normalizer_fingerprint']
    assert dataset_api().file_hash(output/'normalizers.json') == context['normalizer_sha256']


@pytest.mark.parametrize('fault', ['fit', 'write', 'publish'])
def test_fitter_failure_never_modifies_g1_or_leaves_partial_f2(prepared_inputs, tmp_path, monkeypatch, fault):
    m = api(); prepare(prepared_inputs); before = cache_bytes(prepared_inputs[1]); output = tmp_path/'f2'
    def fail(*args, **kwargs):
        if fault == 'write':
            Path(args[1]).write_text('partial normalizer')
        raise OSError('injected F2 failure')
    monkeypatch.setattr(m if fault != 'publish' else m.os,
                        {'fit': 'fit_normalizers', 'write': 'write_normalizers', 'publish': 'rename'}[fault], fail)
    with pytest.raises(OSError, match='injected F2 failure'):
        m.fit_prepared_normalizers(**args(prepared_inputs, output))
    assert cache_bytes(prepared_inputs[1]) == before
    assert not output.exists()
    assert not list(tmp_path.glob('.f2-*'))


def test_existing_f2_directory_is_no_clobber_before_fit(prepared_inputs, tmp_path, monkeypatch):
    m = api(); prepare(prepared_inputs); before = cache_bytes(prepared_inputs[1])
    output = tmp_path/'f2'; output.mkdir(); (output/'existing.json').write_bytes(b'keep me')
    def forbidden(*args, **kwargs): pytest.fail('Output collision must fail before fitting')
    monkeypatch.setattr(m, 'fit_normalizers', forbidden)
    with pytest.raises(ValueError, match='exist|collision'):
        m.fit_prepared_normalizers(**args(prepared_inputs, output))
    assert cache_bytes(output) == {'existing.json': b'keep me'}
    assert cache_bytes(prepared_inputs[1]) == before


def test_concurrent_output_directory_is_not_overwritten(prepared_inputs, tmp_path, monkeypatch):
    m = api(); prepare(prepared_inputs); output = tmp_path/'f2'; original = m.write_normalizers
    def concurrent_publish(*a, **kw):
        original(*a, **kw)
        output.mkdir(); (output/'concurrent.txt').write_text('keep')
    monkeypatch.setattr(m, 'write_normalizers', concurrent_publish)
    with pytest.raises(FileExistsError):
        m.fit_prepared_normalizers(**args(prepared_inputs, output))
    assert cache_bytes(output) == {'concurrent.txt': b'keep'}


@pytest.mark.parametrize('fault', ['incomplete', 'v1', 'normalized', 'independent_identity', 'spec', 'split_count'])
def test_incomplete_or_incompatible_g1_fails_before_fitting(prepared_inputs, tmp_path, monkeypatch, fault):
    from dataclasses import replace
    m = api(); prepare(prepared_inputs); kw = args(prepared_inputs, tmp_path/'f2')
    path = prepared_inputs[1]/'manifest.json'; doc = json.loads(path.read_text())
    if fault in ('incomplete', 'v1', 'normalized'):
        if fault == 'incomplete': doc['complete'] = False
        elif fault == 'normalized': doc['normalized'] = True
        else: doc['schema_version'] = 'mssr.bc_v1.physical_cache.v1'
        doc['manifest_fingerprint'] = dataset_api()._manifest_fingerprint(doc)
        path.write_text(json.dumps(doc))
    elif fault == 'independent_identity':
        kw['expected_identity'] = dict(kw['expected_identity'], source_manifest_sha256='f'*64)
    elif fault == 'spec':
        kw['spec'] = replace(kw['spec'], robot_id='another-backend')
    else:
        eid = kw['split_manifest']['splits']['train'][0]
        doc['episodes'][eid]['samples'] = 1
        doc['counts']['samples'] -= 1
        doc['counts']['splits']['train']['samples'] = 1
        doc['manifest_fingerprint'] = dataset_api()._manifest_fingerprint(doc)
        doc['cache_fingerprint'] = dataset_api().cache_fingerprint(doc['identity'], doc['source_episode_hashes'])
        path.write_text(json.dumps(doc))
    def forbidden(*args, **kwargs): pytest.fail('Incompatible cache must fail before fitting')
    monkeypatch.setattr(m, 'fit_normalizers', forbidden)
    with pytest.raises(ValueError): m.fit_prepared_normalizers(**kw)
    assert not kw['output_dir'].exists()


def test_fitter_rejects_partial_train_coverage(prepared_inputs, tmp_path, monkeypatch):
    m = api(); prepare(prepared_inputs); load = dataset_api()._load_episode
    monkeypatch.setattr(dataset_api(), '_load_episode', lambda *a: load(*a)[:1])
    output = tmp_path/'f2'
    with pytest.raises(ValueError, match='coverage|count'):
        m.fit_prepared_normalizers(**args(prepared_inputs, output))
    assert not output.exists()


@pytest.mark.parametrize('location', ['cache', 'source'])
def test_fitter_never_writes_inside_g1_or_canonical(prepared_inputs, monkeypatch, location):
    m = api(); prepare(prepared_inputs)
    output = prepared_inputs[1]/'f2' if location == 'cache' else prepared_inputs[0]/'f2'
    def forbidden(*args, **kwargs): pytest.fail('Unsafe output must fail before fitting')
    monkeypatch.setattr(m, 'fit_normalizers', forbidden)
    with pytest.raises(ValueError): m.fit_prepared_normalizers(**args(prepared_inputs, output))
    assert not output.exists()


def test_separate_fit_preserves_existing_normalization_math(prepared_inputs, tmp_path):
    m = api(); prepare(prepared_inputs); output = tmp_path/'f2'
    m.fit_prepared_normalizers(**args(prepared_inputs, output))
    manifest = dataset_api().read_cache_manifest(prepared_inputs[1]); config = prepared_inputs[3]
    def stream(eid):
        for sample in dataset_api()._load_episode(prepared_inputs[1], eid, manifest['episodes'][eid]):
            yield normalization.NormalizationSample(eid, sample.audit.transition_index,
                                                     sample.inputs.features, sample.supervision)
    baseline = normalization.fit_normalizers({e: lambda e=e: stream(e) for e in manifest['splits']['train']},
        config['split_manifest'], config['spec'], config['schema'], config['normalization_policy'],
        expected_split_fingerprint=config['expected_split_fingerprint'])
    assert (output/'normalizers.json').read_bytes() == normalization.serialize_normalizers(baseline)


def test_both_cli_help_commands_import_without_hardware_or_torch():
    script = '''
import sys,importlib.abc,runpy
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if any(x in name for x in ('smores','isaac','omni','rclpy','torch')): raise AssertionError(name)
sys.meta_path.insert(0,Block())
sys.argv=['bc_v1',sys.argv[1],'--help']
runpy.run_module('drl.IL.bc_v1',run_name='__main__')
'''
    for command in ('prepare', 'fit-normalizers'):
        result = subprocess.run([sys.executable, '-B', '-c', script, command], cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert '--output' in result.stdout
