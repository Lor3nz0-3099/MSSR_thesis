"""Optional process-owned RAM acceleration over immutable, authenticated G2.

G2 is ALREADY normalized. The complete source identity, including F2, catalog,
layout and manifest, scopes every cache instance. Future folds need their own
normalizers and canonical/G1 source; this cache does not invert normalization.
Nothing preloads or publishes artifacts. Budgets cover retained tensor buffers,
Python routing/descriptors and conservative entry overhead, not transient batches,
SQLite's page cache, allocator fragmentation or the process RSS.
"""
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass
import copy
import os
import queue
import sys
import threading

import torch

from .batching import collate_policy_samples
from .sidecar import SidecarDataset

VERSION = 'mssr.bc_v1.training_numeric_cache.v1'


@dataclass(frozen=True)
class _Leaf:
    dtype: torch.dtype
    start: int
    count: int
    shape: tuple


@dataclass(frozen=True)
class _ImmutableTuple:
    value: tuple


class _Packed:
    """At most one contiguous tensor buffer per dtype, plus original routing."""
    def __init__(self, value):
        chunks, ends = {}, {}
        def visit(item):
            if isinstance(item, torch.Tensor):
                dtype = item.dtype
                start = ends.get(dtype, 0)
                chunks.setdefault(dtype, []).append(item.reshape(-1))
                ends[dtype] = start + item.numel()
                return _Leaf(dtype, start, item.numel(), tuple(item.shape))
            if isinstance(item, dict):
                return {k: visit(v) for k, v in item.items()}
            if isinstance(item, list):
                return [visit(v) for v in item]
            if isinstance(item, tuple):
                packed = tuple(visit(v) for v in item)
                # Opaque leaves already have immutable routing semantics.
                # Preserve plain tuple subtrees with no tensor or mutable
                # container to reconstruct; charge them through _retained_size.
                if type(item) is tuple and not any(isinstance(v, (_Leaf, dict, list, tuple)) for v in packed):
                    return _ImmutableTuple(item)
                return packed
            return item
        self.tree = visit(value)
        self.buffers = {dtype: torch.cat(values) for dtype, values in chunks.items()}
        self.bytes = (_retained_size(self.tree) + _retained_size(self.buffers)
                      + sys.getsizeof(self) + sys.getsizeof(self.__dict__))

    def unpack(self, *, owned):
        buffers = {k: v.clone() for k, v in self.buffers.items()} if owned else self.buffers
        def visit(item):
            if isinstance(item, _ImmutableTuple):
                return item.value
            if isinstance(item, _Leaf):
                return buffers[item.dtype].narrow(0, item.start, item.count).view(item.shape)
            if isinstance(item, dict):
                return {k: visit(v) for k, v in item.items()}
            if isinstance(item, list):
                return [visit(v) for v in item]
            if isinstance(item, tuple):
                return tuple(visit(v) for v in item)
            # G2 typed routing dataclasses are immutable, as in SidecarDataset.
            return item
        return visit(self.tree)


def _retained_size(value, seen=None):
    """Conservative recursive charge, counting immutable routing too."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, torch.Tensor):
        return size + value.numel() * value.element_size()
    if isinstance(value, dict):
        return size + sum(_retained_size(k, seen) + _retained_size(v, seen) for k, v in value.items())
    if isinstance(value, (tuple, list, set, frozenset)):
        return size + sum(_retained_size(v, seen) for v in value)
    if is_dataclass(value):
        size += sys.getsizeof(getattr(value, '__dict__', {}))
        size += sum(_retained_size(getattr(value, f.name), seen) for f in fields(value))
    return size


class _BudgetLRU:
    def __init__(self, budget):
        self.budget, self.bytes, self.hits = budget, 0, 0
        self.entries = OrderedDict()

    def get(self, key):
        entry = self.entries.get(key)
        if entry is not None:
            self.hits += 1
            self.entries.move_to_end(key)
            return entry[0]
        return None

    def put(self, key, value, charge):
        charge += _retained_size(key) + 256  # LRU node/entry overhead.
        if charge > self.budget:
            return
        while self.entries and self.bytes + charge > self.budget:
            _, (_, previous) = self.entries.popitem(last=False)
            self.bytes -= previous
        self.entries[key] = (value, charge)
        self.bytes += charge

    def clear(self):
        self.entries.clear()
        self.bytes = 0


class TrainingDataset:
    """TRAIN/VAL-only, lazy template-aware cache with owned public samples.

    get_for_collate returns borrowed numerical views for immediate read-only
    collation. Do not mutate or retain those views. Public indexing clones the
    compact buffers, preserving SidecarDataset's tensor ownership contract.
    """
    def __init__(self, root, split, *, expected_identity,
                 dynamic_budget_bytes=256*1024**2, template_budget_bytes=128*1024**2):
        if split not in ('train', 'validation'):
            raise ValueError('TrainingDataset forbids TEST before opening a source')
        for budget in (dynamic_budget_bytes, template_budget_bytes):
            if type(budget) is not int or budget < 0:
                raise ValueError('Nonnegative integer RAM byte budgets required')
        self._owner = (os.getpid(), threading.get_ident())
        self._data = SidecarDataset(root, split, expected_identity=expected_identity, template_cache_size=0)
        self.root, self.split, self.manifest = self._data.root, split, self._data.manifest
        self.cache_identity = {'schema_version': VERSION,
            'source_schema_version': self.manifest['schema_version'],
            'manifest_fingerprint': self.manifest['manifest_fingerprint'],
            'identity': copy.deepcopy(self.manifest['identity']), 'split': split}
        self._dynamic, self._templates = _BudgetLRU(dynamic_budget_bytes), _BudgetLRU(template_budget_bytes)
        self._closed = False

    def _check_owner(self):
        if self._owner != (os.getpid(), threading.get_ident()):
            raise RuntimeError('TrainingDataset belongs to its creating process and thread')
        if self._closed:
            raise ValueError('TrainingDataset is closed')

    def __len__(self):
        return len(self._data)

    def _read(self, index, *, owned):
        self._check_owner()
        if type(index) is not int:
            raise TypeError('Integer sidecar index required')
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        entry = self._dynamic.get(index)
        template = self._templates.get(entry[0]) if entry is not None else None
        if entry is None:
            loaded = {}
            def template_reader(tid, load_verified):
                packed = self._templates.get(tid)
                if packed is None:
                    packed = _Packed(load_verified())
                    self._templates.put(tid, packed, packed.bytes)
                loaded[tid] = packed
                return packed.unpack(owned=False)
            tid, _, dynamic = self._data.read_parts(index, template_reader=template_reader)
            template = loaded[tid]
            packed = _Packed(dynamic)
            entry = (tid, packed)
            self._dynamic.put(index, entry, packed.bytes + _retained_size(tid) + sys.getsizeof(entry))
        elif template is None:
            # Dynamic content was already authenticated in this source-scoped
            # cache. Reauthenticate only the evicted template's source bytes.
            tid = entry[0]
            template = _Packed(self._data._read_template(tid))
            self._templates.put(tid, template, template.bytes)
        static, dynamic = template.unpack(owned=owned), entry[1].unpack(owned=owned)
        return {'inputs': {**static['inputs'], **dynamic['inputs']},
            'supervision': dynamic['supervision'],
            'routing': {**static['routing'], **dynamic['routing']}}

    def __getitem__(self, index):
        return self._read(index, owned=True)

    def get_for_collate(self, index):
        return self._read(index, owned=False)

    def train_index(self):
        self._check_owner()
        return self._data.train_index()

    def stats(self):
        caches = (self._dynamic, self._templates)
        return {'dynamic_entries': len(self._dynamic.entries), 'template_entries': len(self._templates.entries),
            'dynamic_bytes': self._dynamic.bytes, 'template_bytes': self._templates.bytes,
            'retained_bytes': sum(c.bytes for c in caches),
            'dynamic_hits': self._dynamic.hits, 'template_hits': self._templates.hits,
            'packed_buffers': sum(len((v[0][1] if c is self._dynamic else v[0]).buffers)
                for c in caches for v in c.entries.values())}

    cache_stats = stats

    def close(self):
        if not self._closed:
            self._check_owner()
            self._data.close()
            self._dynamic.clear()
            self._templates.clear()
            self._closed = True

    def __enter__(self):
        self._check_owner()
        return self

    def __exit__(self, *args):
        self.close()


@dataclass
class _Failure:
    error: BaseException


_END = object()


class OrderedPrefetch:
    """Bounded CPU producer for PRECOMPUTED batches; never advances a sampler.

    The caller supplies a Sequence of index batches and a factory that opens
    its dataset inside the worker. Device transfer stays on the consuming
    thread. Use a context manager to guarantee cleanup on early termination.
    """
    def __init__(self, index_batches, dataset_factory, *, depth=2, collate=collate_policy_samples):
        if not isinstance(index_batches, Sequence):
            raise TypeError('OrderedPrefetch requires a precomputed Sequence, never a live sampler')
        if type(depth) is not int or not 1 <= depth <= 2:
            raise ValueError('Prefetch depth must be one or two')
        self._batches, self._factory, self._collate = index_batches, dataset_factory, collate
        self._queue, self._stop = queue.Queue(maxsize=depth), threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._produce, name='bc-g2-prefetch', daemon=True)
        self._thread.start()

    def _put(self, value):
        while not self._stop.is_set():
            try:
                self._queue.put(value, timeout=.05)
                return True
            except queue.Full:
                pass
        return False

    def _produce(self):
        try:
            with self._factory() as data:
                for indices in self._batches:
                    if self._stop.is_set():
                        return
                    batch = self._collate([data.get_for_collate(i) for i in indices])
                    if not self._put(batch):
                        return
        except BaseException as exc:
            self._put(_Failure(exc))
        finally:
            self._put(_END)

    @property
    def worker_alive(self):
        return self._thread.is_alive()

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        item = self._queue.get()
        if item is _END:
            self.close()
            raise StopIteration
        if isinstance(item, _Failure):
            self.close()
            raise item.error
        return item

    def close(self):
        if not self._closed:
            self._stop.set()
            self._thread.join(timeout=10.)
            if self._thread.is_alive():
                raise RuntimeError('CPU prefetch worker did not finish its current batch within ten seconds')
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
