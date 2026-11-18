"""BC-v5 diagnostics/sampler: small frozen synthetic G2, never production data."""
from dataclasses import replace
import json
from collections import Counter

import pytest

from drl.IL.bc_v1 import bc_v5_census as audit, bc_v5_sampling as sampling
from drl.IL.bc_v1.sidecar import SidecarWriter
from drl.IL.tests.bc_v1.test_sidecar import identity, row
from drl.IL.bc_v1 import dataset as g1


def _source(tmp_path, *, test_count=0):
    episodes = {'train': [('a', 2), ('b', 1)], 'validation': [('v', 1)],
                'test': [('t', test_count)] if test_count else []}
    cat = {k: [{'episode_id': eid, 'samples': n,
                'transition_indices_fingerprint': g1._hash(list(range(7, 7+n)))}
               for eid, n in entries] for k, entries in episodes.items()}
    writer = SidecarWriter(tmp_path/'g2', identity(), cat)
    first = row('a', 7)
    first['routing']['audit'] = replace(first['routing']['audit'],
        morphology_stratum='snake8', task_stratum='gap', provenance='expert')
    second = row('a', 8, structural=True)
    second['routing']['audit'] = replace(second['routing']['audit'],
        morphology_stratum='snake8', task_stratum='gap', provenance='expert')
    third = row('b', 7)
    third['routing']['audit'] = replace(third['routing']['audit'],
        morphology_stratum='rc_car8', task_stratum='navigation', provenance='teleop')
    for i, sample in enumerate((first, second)):
        writer.add(sample, 'train', i)
    writer.add(third, 'train', 2)
    val = row('v', 7)
    val['routing']['audit'] = replace(val['routing']['audit'],
        morphology_stratum='mobile_manipulator8', task_stratum='button', provenance='teleop')
    writer.add(val, 'validation', 0)
    if test_count:
        writer.add(row('t', 7), 'test', 0)
    writer.finalize()
    return tmp_path/'g2'


def test_census_uses_only_bc_eligible_train_val_and_preserves_source(tmp_path):
    root = _source(tmp_path)
    previous_manifest = (root/'manifest.json').read_bytes()
    result = audit.analyze(root)
    assert (root/'manifest.json').read_bytes() == previous_manifest
    assert result['test_payload_access'] == 0
    assert set(result['splits']) == {'train', 'validation'}
    train = result['splits']['train']
    assert train['samples'] == 3 and train['episodes'] == 2
    assert train['samples_by_morphology'] == {'rc_car8': 1, 'snake8': 2}
    assert train['samples_by_provenance'] == {'expert': 2, 'teleop': 1}
    assert train['primitive_sample_rows']['resource:pace'] == 3
    assert any(k.startswith('structural:') and n > 0
               for k, n in train['primitive_target_instances'].items())
    assert result['splits']['validation']['samples'] == 1
    assert result['splits']['validation']['samples_by_task'] == {'button': 1}
    assert any(x['primitive'] == 'resource:pace' and x['sample_rows'] == 2
               for x in train['cross_morphology_task_provenance_primitive'])


def test_census_cli_does_not_overwrite_or_access_test(tmp_path, monkeypatch):
    root = _source(tmp_path)
    output = tmp_path/'census.json'
    assert audit.main(['--g2', str(root), '--output', str(output), '--train-only']) == 0
    payload = json.loads(output.read_text())
    assert payload['test_payload_access'] == 0 and set(payload['splits']) == {'train'}
    with pytest.raises(SystemExit):
        audit.main(['--g2', str(root), '--output', str(output)])


def test_census_rejects_nonempty_test_without_reading_payload(tmp_path):
    root = _source(tmp_path, test_count=1)
    with pytest.raises(ValueError, match='TEST'):
        audit.analyze(root)


def test_balanced_index_and_resume_determinism(tmp_path):
    from drl.IL.bc_v1.sidecar import SidecarDataset
    root = _source(tmp_path)
    with SidecarDataset(root, 'train', expected_identity=identity()) as data:
        index = sampling.build_primitive_train_index(data)
        assert index.sample_count == 3
        assert any(k[0] == 'snake8' and k[3].startswith('structural:')
                   for k in index.group_episode_rows)
        a = sampling.BalancedPrimitiveBatchSampler(index, batch_size=4, balanced_quota=2, seed=3)
        b = sampling.BalancedPrimitiveBatchSampler(index, batch_size=4, balanced_quota=2, seed=3)
        assert list(a) == list(b)
        assert a.last_draw == {'ordinary': 2, 'balanced': 2}
        state = json.loads(json.dumps(a.state_dict()))
        next_a = next(iter(a))
        b.load_state_dict(state)
        next_b = next(iter(b))
        assert next_a == next_b
        state['index_fingerprint'] = 'other'
        with pytest.raises(ValueError):
            b.load_state_dict(state)


def test_balanced_index_rejects_validation_and_invalid_rows(tmp_path):
    from drl.IL.bc_v1.sidecar import SidecarDataset
    root = _source(tmp_path)
    with SidecarDataset(root, 'validation', expected_identity=identity()) as data:
        with pytest.raises(ValueError, match='TRAIN'):
            sampling.build_primitive_train_index(data)
    with pytest.raises(ValueError):
        sampling.PrimitiveTrainIndex({'a': (1,)}, {('snake8', 'gap', 'expert', 'resource:pace'): {'a': (1,)}}, 'fp')


def test_sampler_balances_all_families_and_retains_uniform_exposure():
    # Independent small in-memory representation: 990 frequent, 10 rare.
    rows = {'large': tuple(range(0, 990)), 'small': tuple(range(990, 1000))}
    grouped = {('snake8', 'gap', 'expert', 'resource:wheel_velocity'): {'large': rows['large']},
               ('snake8', 'gap', 'expert', 'resource:tilt_target'): {'small': rows['small']}}
    index = sampling.PrimitiveTrainIndex(rows, grouped, 'synthetic-g2')
    sampler = sampling.BalancedPrimitiveBatchSampler(index, batch_size=10, balanced_quota=5,
                                                      frequency_exponent=0.5, seed=33)
    drawn = [i for batch in sampler for i in batch]
    # Baseline uniform 1%; tempered group balancing raises the rare's share
    # but never makes it dominate like full inverse-frequency sampling.
    rare = sum(i >= 990 for i in drawn)/len(drawn)
    assert 0.03 < rare < 0.2
    assert any(i < 990 for i in drawn) and any(i >= 990 for i in drawn)


def test_probe_simulates_train_exposure_without_training_or_test(tmp_path):
    from drl.IL.bc_v1 import bc_v5_probe
    root = _source(tmp_path)
    before = (root/'manifest.json').read_bytes()
    report = bc_v5_probe.probe(root, batch_size=2, balanced_quota=1, seed=12, epochs=2)
    assert report['train_rows'] == 3 and report['sampled_slots'] == 8
    assert report['test_payload_access'] == 0
    assert report['original_one_pass']['primitive']['resource:pace'] == 3
    assert report['simulated_exposure']['primitive']['resource:pace'] == 8
    assert report['unique_rows_sampled'] <= 3
    assert (root/'manifest.json').read_bytes() == before
