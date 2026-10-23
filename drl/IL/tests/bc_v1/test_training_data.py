"""Synthetic CPU checks for the optional bounded training read path."""
import importlib.util
import threading

import pytest
import torch

from drl.IL.tests.bc_v1.test_sidecar import identity, row, assert_equal
from drl.IL.bc_v1.sidecar import SidecarWriter, SidecarDataset, _hash
from drl.IL.bc_v1.batching import collate_policy_samples


@pytest.fixture(autouse=True, scope='module')
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.training_data'), 'Bounded training read path missing'
    from drl.IL.bc_v1 import training_data
    return training_data


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'g2'
    inventory = {'train': [{'episode_id': 'a', 'samples': 3,
        'transition_indices_fingerprint': _hash([7, 8, 9])}], 'validation': [], 'test': []}
    writer = SidecarWriter(root, identity(), inventory)
    for i in range(3):
        sample = row(transition=7+i)
        sample['inputs']['module_x'][0, 0] = float(i)
        writer.add(sample, 'train', i)
    writer.finalize()
    return root


def test_owned_cache_hits_preserve_every_section_and_isolate_mutation(source):
    a = api()
    with a.TrainingDataset(source, 'train', expected_identity=identity(),
            dynamic_budget_bytes=2**20, template_budget_bytes=2**20) as data:
        with SidecarDataset(source, 'train', expected_identity=identity()) as original:
            expected = original[0]
        actual = data[0]
        assert_equal(actual, expected)
        actual['inputs']['module_x'].fill_(999)
        actual['inputs']['structural']['initiator'].fill_(999)
        actual['routing']['extra'] = 'mutated'
        assert_equal(data[0], expected)
        assert data.stats()['dynamic_hits'] >= 1
        assert data.stats()['dynamic_entries'] == 1
        assert data.train_index().sample_count == 3
    assert data.stats()['retained_bytes'] == 0


def test_shared_templates_borrowed_collation_matches_owned_and_duplicates(source):
    a = api()
    with a.TrainingDataset(source, 'train', expected_identity=identity(),
            dynamic_budget_bytes=2**20, template_budget_bytes=2**20) as data:
        expected = collate_policy_samples([data[i] for i in [2, 0, 2, 1]])
        actual = collate_policy_samples([data.get_for_collate(i) for i in [2, 0, 2, 1]])
        assert_equal(actual, expected)
        assert data.stats()['dynamic_entries'] == 3
        assert data.stats()['template_entries'] == 1
        assert data.stats()['packed_buffers'] <= 12
        actual['inputs']['module_x'].fill_(888)
        assert float(data[2]['inputs']['module_x'][0, 0]) == 2.


@pytest.mark.parametrize('budget', [0, 1, 16000])
def test_small_budgets_bypass_or_evict_without_changing_samples(source, budget):
    with api().TrainingDataset(source, 'train', expected_identity=identity(),
            dynamic_budget_bytes=budget, template_budget_bytes=budget) as data:
        for i in [0, 1, 2, 0, -1]:
            assert data[i]['routing']['audit'].transition_index == 7 + i % 3
            stats = data.stats()
            assert stats['dynamic_bytes'] <= budget
            assert stats['template_bytes'] <= budget
        if budget < 2:
            assert data.stats()['retained_bytes'] == 0


def test_rejects_test_before_open_and_rejects_changed_normalizer(source):
    a = api()
    with pytest.raises(ValueError, match='TEST'):
        a.TrainingDataset('/does/not/exist', 'test', expected_identity={})
    with pytest.raises(ValueError, match='[Ss]tale|[Ii]ncompatible'):
        a.TrainingDataset(source, 'train', expected_identity={**identity(), 'f2_fingerprint': 'new-fold'})
    with a.TrainingDataset(source, 'train', expected_identity=identity()) as data:
        assert data.cache_identity['manifest_fingerprint']
        assert data.cache_identity['identity']['f2_fingerprint'] == identity()['f2_fingerprint']
        with pytest.raises(TypeError):
            data[True]
        with pytest.raises(IndexError):
            data[3]


def test_prefetch_uses_worker_owned_sqlite_and_preserves_order(source):
    a = api()
    owner = threading.get_ident()
    opened = []
    def factory():
        opened.append(threading.get_ident())
        return a.TrainingDataset(source, 'train', expected_identity=identity())
    batches = [(2, 0), (1, 1), (0,)]
    with a.OrderedPrefetch(batches, factory, depth=2) as prefetched:
        result = list(prefetched)
    assert opened and opened[0] != owner
    assert [[r['audit'].transition_index for r in b['routing']['samples']] for b in result] == [[9, 7], [8, 8], [7]]
    assert not prefetched.worker_alive


def test_prefetch_early_close_and_source_errors_do_not_hang(source):
    a = api()
    factory = lambda: a.TrainingDataset(source, 'train', expected_identity=identity())
    with a.OrderedPrefetch([(0,)] * 30, factory, depth=1) as prefetched:
        next(prefetched)
    assert not prefetched.worker_alive
    with a.OrderedPrefetch([(0,), (99,)], factory, depth=1) as prefetched:
        next(prefetched)
        with pytest.raises(IndexError):
            next(prefetched)
    assert not prefetched.worker_alive
    with pytest.raises(TypeError, match='[Ss]equence|precomputed'):
        a.OrderedPrefetch(iter([(0,)]), factory)


def test_new_dynamic_rows_reuse_authenticated_static_template_without_redecode(source, monkeypatch):
    from drl.IL.bc_v1 import sidecar
    decoded_templates = []
    original_decode = sidecar.decode
    def decode(blob):
        value = original_decode(blob)
        if 'supervision' not in value:
            decoded_templates.append(1)
        return value
    monkeypatch.setattr(sidecar, 'decode', decode)
    with api().TrainingDataset(source, 'train', expected_identity=identity()) as data:
        for i in (0, 1, 2):
            assert data.get_for_collate(i)['routing']['audit'].transition_index == 7+i
    assert len(decoded_templates) == 1


def test_authenticated_parts_hook_preserves_owned_source_api(source):
    from drl.IL.bc_v1.sidecar import _join
    with SidecarDataset(source, 'train', expected_identity=identity()) as data:
        assert hasattr(data, 'read_parts'), 'Authenticated sidecar parts hook missing'
        tid, static, dynamic = data.read_parts(0)
        assert tid
        assert_equal(_join(static, dynamic), data[0])
        templates = {}
        def provider(key, load):
            if key not in templates:
                templates[key] = load()
            return templates[key]
        _, _, dynamic = data.read_parts(1, template_reader=provider)
        assert dynamic['routing']['audit'].transition_index == 8


@pytest.mark.parametrize('damage', ['checksum', 'mapping', 'audit', 'upstream'])
def test_cache_cold_reads_preserve_sidecar_authentication(source, damage):
    from dataclasses import replace
    import sqlite3
    from drl.IL.bc_v1.sidecar import decode, encode, _sha
    with sqlite3.connect(source/'data.sqlite3') as db:
        if damage == 'checksum':
            db.execute("UPDATE samples SET checksum='bad' WHERE split='train' AND idx=0")
        elif damage == 'mapping':
            db.execute("UPDATE samples SET episode='foreign' WHERE split='train' AND idx=0")
        else:
            blob = db.execute("SELECT payload FROM samples WHERE split='train' AND idx=0").fetchone()[0]
            dynamic = decode(blob)
            if damage == 'audit':
                dynamic['routing']['audit'] = replace(dynamic['routing']['audit'], transition_index=99)
            else:
                dynamic['routing']['normalizer_fingerprint'] = 'foreign-fold'
            blob = encode(dynamic)
            db.execute("UPDATE samples SET payload=?,checksum=? WHERE split='train' AND idx=0", (blob, _sha(blob)))
    with api().TrainingDataset(source, 'train', expected_identity=identity()) as data:
        with pytest.raises(ValueError, match='checksum|mapping|identity'):
            data[0]


def test_dynamic_lru_eviction_reloads_exact_values_under_one_row_budget(source):
    a = api()
    with a.TrainingDataset(source, 'train', expected_identity=identity()) as probe:
        probe[0]
        first_charge = probe.stats()['dynamic_bytes']
        probe[1]
        second_charge = probe.stats()['dynamic_bytes'] - first_charge
    budget = max(first_charge, second_charge)
    with a.TrainingDataset(source, 'train', expected_identity=identity(), dynamic_budget_bytes=budget) as data:
        for i in [0, 1, 0, 1]:
            assert float(data[i]['inputs']['module_x'][0, 0]) == float(i)
            assert data.stats()['dynamic_entries'] == 1
            assert data.stats()['dynamic_bytes'] <= budget
        assert data.stats()['dynamic_hits'] == 0


def test_dataset_rejects_cross_thread_use_and_closed_reads(source):
    a = api()
    with a.TrainingDataset(source, 'train', expected_identity=identity()) as data:
        errors = []
        def other_thread():
            try:
                data[0]
            except RuntimeError as exc:
                errors.append(str(exc))
        worker = threading.Thread(target=other_thread)
        worker.start()
        worker.join(timeout=2)
        assert errors and not worker.is_alive()
    with pytest.raises(ValueError, match='closed'):
        data[0]


def test_evicted_template_reload_reuses_authenticated_dynamic_record(tmp_path, monkeypatch):
    from drl.IL.bc_v1 import sidecar
    root = tmp_path / 'two_templates'
    inventory = {'train': [{'episode_id': 'a', 'samples': 2,
        'transition_indices_fingerprint': _hash([7, 8])}], 'validation': [], 'test': []}
    samples = [row(transition=7), row(transition=8)]
    samples[1]['inputs']['structural']['peer'][0] = -1
    writer = SidecarWriter(root, identity(), inventory)
    for i, sample in enumerate(samples):
        writer.add(sample, 'train', i)
    assert writer.finalize()['template_count'] == 2
    a = api()
    with a.TrainingDataset(root, 'train', expected_identity=identity()) as probe:
        probe[0]
        first_charge = probe.stats()['template_bytes']
        probe[1]
        second_charge = probe.stats()['template_bytes'] - first_charge
    decoded = {'dynamic': 0, 'template': 0}
    decode = sidecar.decode
    def observed_decode(blob):
        value = decode(blob)
        decoded['dynamic' if 'supervision' in value else 'template'] += 1
        return value
    monkeypatch.setattr(sidecar, 'decode', observed_decode)
    with a.TrainingDataset(root, 'train', expected_identity=identity(),
            template_budget_bytes=max(first_charge, second_charge)) as data:
        for i in [0, 1, 0]:
            assert_equal(data[i], samples[i])
            assert data.stats()['template_entries'] == 1
        assert data.stats()['dynamic_entries'] == 2
        assert decoded == {'dynamic': 2, 'template': 3}
        # The dynamic cache is trusted only because this source is immutable;
        # an evicted static template still has to authenticate its source bytes.
        import sqlite3
        with sqlite3.connect(root/'data.sqlite3') as db:
            tid = db.execute("SELECT template FROM samples WHERE split='train' AND idx=1").fetchone()[0]
            blob = db.execute('SELECT payload FROM templates WHERE id=?', (tid,)).fetchone()[0]
            damaged = bytes([blob[0] ^ 1]) + blob[1:]
            db.execute('UPDATE templates SET payload=? WHERE id=?', (damaged, tid))
        with pytest.raises(ValueError, match='corrupt template'):
            data[1]


def test_unpack_does_not_rewalk_immutable_routing_subtrees():
    import sys
    a = api()

    def unpack_work(length):
        bindings = tuple((str(i), i, (True, None)) for i in range(length))
        packed = a._Packed({'numeric': torch.arange(6).reshape(2, 3),
                            'routing': {'bindings': bindings}})
        visits = []
        previous = sys.getprofile()
        def observe(frame, event, arg):
            if event == 'call' and frame.f_code.co_name == 'visit' and frame.f_globals is a.__dict__:
                visits.append(1)
        try:
            sys.setprofile(observe)
            value = packed.unpack(owned=False)
        finally:
            sys.setprofile(previous)
        assert value['routing']['bindings'] == bindings
        assert torch.equal(value['numeric'], torch.arange(6).reshape(2, 3))
        return len(visits)

    assert unpack_work(2000) == unpack_work(2), 'Immutable routing must not be recursively rebuilt per read'


@pytest.mark.parametrize('owned', [False, True])
def test_opaque_routing_keeps_mutable_containers_and_tensor_ownership(owned):
    from collections import namedtuple
    a = api()
    Pair = namedtuple('Pair', 'left right')
    packed = a._Packed({'numeric': torch.arange(3),
        'routing': {'immutable': ((1, 'peer'), (False, None)),
                    'mutable': ({'items': [1]},),
                    'tensor_tuple': (torch.tensor([7.]),),
                    'tuple_subclass': (Pair(1, 2),)}})
    first, second = packed.unpack(owned=owned), packed.unpack(owned=owned)
    first['routing']['mutable'][0]['items'].append(9)
    assert second['routing']['mutable'][0]['items'] == [1]
    assert packed.unpack(owned=False)['routing']['mutable'][0]['items'] == [1]
    assert type(second['routing']['tuple_subclass'][0]) is tuple
    assert second['routing']['immutable'] == ((1, 'peer'), (False, None))
    assert torch.equal(second['routing']['tensor_tuple'][0], torch.tensor([7.]))
    if owned:
        first['numeric'].fill_(99)
        first['routing']['tensor_tuple'][0].fill_(99)
        assert torch.equal(second['numeric'], torch.arange(3))
        assert torch.equal(packed.unpack(owned=False)['routing']['tensor_tuple'][0], torch.tensor([7.]))
