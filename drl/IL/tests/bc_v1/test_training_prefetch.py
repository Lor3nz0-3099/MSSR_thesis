"""Spawned CPU producer checks over synthetic temporary G2 only."""
import importlib.util
import time

import pytest
import torch

from drl.IL.bc_v1.sidecar import SidecarWriter, _hash
from drl.IL.bc_v1.batching import collate_policy_samples
from drl.IL.tests.bc_v1.test_sidecar import identity, row, assert_equal


@pytest.fixture(autouse=True, scope='module')
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.training_prefetch'), 'Process producer missing'
    from drl.IL.bc_v1 import training_prefetch
    return training_prefetch


@pytest.fixture
def source(tmp_path):
    root = tmp_path/'g2'
    inventory = {'train': [{'episode_id': 'a', 'samples': 2,
        'transition_indices_fingerprint': _hash([7, 8])}], 'validation': [], 'test': []}
    writer = SidecarWriter(root, identity(), inventory)
    samples = [row(transition=7), row(transition=8)]
    samples[1]['inputs']['module_x'][0, 0] = 2.
    for i, sample in enumerate(samples):
        writer.add(sample, 'train', i)
    writer.finalize()
    return root, samples


def kwargs():
    return dict(expected_identity=identity(), dynamic_budget_bytes=2**20,
        template_budget_bytes=2**20, depth=2, cpu_threads=1)


def test_process_fifo_exactness_duplicate_rows_and_child_cache_stats(source):
    root, samples = source
    batches = [(1, 0), (1, 1), (0,)]
    with api().OrderedProcessPrefetch(batches, root, 'train', **kwargs()) as producer:
        actual = list(producer)
        stats = producer.stats()
    assert len(actual) == 3
    for batch, indices in zip(actual, batches):
        assert_equal(batch, collate_policy_samples([samples[i] for i in indices]))
    assert stats['batches'] == 3
    assert stats['serialized_bytes'] > 0
    assert stats['cache']['dynamic_hits'] > 0
    assert stats['cache']['retained_bytes'] <= 2 * 2**20
    assert not producer.worker_alive


def test_process_early_close_does_not_leave_owned_child_running(source):
    root, _ = source
    with api().OrderedProcessPrefetch([(0,)]*100, root, 'train', **kwargs()) as producer:
        next(producer)
    assert not producer.worker_alive


def test_worker_error_is_delivered_and_child_is_closed(source):
    root, _ = source
    with api().OrderedProcessPrefetch([(0,), (99,)], root, 'train', **kwargs()) as producer:
        next(producer)
        with pytest.raises(RuntimeError, match='IndexError'):
            next(producer)
    assert not producer.worker_alive


def test_abrupt_owned_child_exit_is_detected_without_blocking_forever(source):
    root, _ = source
    producer = api().OrderedProcessPrefetch([(0,)]*100, root, 'train', **kwargs())
    try:
        producer.process.terminate()
        producer.process.join(timeout=3)
        start = time.monotonic()
        with pytest.raises(RuntimeError, match='exited|died'):
            next(producer)
        assert time.monotonic() - start < 3
    finally:
        producer.close()
    assert not producer.worker_alive


def test_rejects_test_and_live_sampler_before_starting_child(source):
    a = api()
    with pytest.raises(ValueError, match='TEST'):
        a.OrderedProcessPrefetch([], '/not/a/source', 'test', **kwargs())
    with pytest.raises(TypeError, match='Sequence|precomputed'):
        a.OrderedProcessPrefetch(iter([(0,)]), source[0], 'train', **kwargs())


def test_explicit_cpu_cache_warmup_does_not_add_training_batches(source):
    root,samples=source
    with api().OrderedProcessPrefetch([(1,0)],root,'train',warm_indices=(0,1),**kwargs()) as producer:
        actual=list(producer);stats=producer.stats()
    assert len(actual)==1
    assert_equal(actual[0],collate_policy_samples([samples[1],samples[0]]))
    assert stats['cache']['dynamic_hits']>=2
    assert stats['child_peak_rss_bytes']>0
