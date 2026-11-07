"""Finite diagnostics over newly appended structured events, without CUDA-wide sync."""
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import math

import torch


def _increment(counters, name, amount=1):
    counters[name] = counters.get(name, 0) + amount


def finite_sample_trees(trees, counters):
    """Return one finite flag per tree, transferring results once per device.

    Every tensor is inspected, including leaves following a nonfinite scalar.
    CUDA reductions remain on their owning device until the compact per-sample
    flags are copied to the host. No global device synchronization is requested.
    """
    good = [True] * len(trees)
    device_flags = defaultdict(lambda: defaultdict(list))

    def visit(value, sample_index):
        if torch.is_tensor(value):
            _increment(counters, 'structured_tensor_checks')
            device_flags[value.device][sample_index].append(
                torch.isfinite(value.detach()).all())
        elif isinstance(value, Mapping):
            for item in value.values():
                visit(item, sample_index)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item, sample_index)
        elif is_dataclass(value):
            for field in fields(value):
                visit(getattr(value, field.name), sample_index)
        elif isinstance(value, (float, int)) and not math.isfinite(value):
            good[sample_index] = False

    with torch.no_grad():
        for index, tree in enumerate(trees):
            visit(tree, index)
        for device, samples in device_flags.items():
            indexes = list(samples)
            flags = torch.stack([torch.stack(samples[index]).all() for index in indexes])
            values = flags.cpu().tolist()
            _increment(counters, 'structured_device_reductions')
            _increment(counters, 'structured_host_transfers', int(device.type != 'cpu'))
            for index, finite in zip(indexes, values):
                good[index] = good[index] and finite
    return good


class StructuredSampleDiagnostics:
    """Check each append position once within a sidecar reset generation.

    Native sidecars increment ``_diagnostic_generation`` on clear, which also
    distinguishes equal-length refills containing the exact same sample objects.
    List identity and tail checks support older append-only sidecars as well.
    The cursor retains no sample collection after it is replaced.
    """
    def __init__(self, counters):
        self.counters = counters
        self._sidecar = self._samples = self._tail = None
        self._generation = None
        self._offset = 0
        for name in ('structured_samples_checked', 'nonfinite_structured_samples',
                     'structured_tensor_checks', 'structured_device_reductions',
                     'structured_host_transfers', 'structured_no_new_sample_calls',
                     'structured_repeat_samples_skipped'):
            counters.setdefault(name, 0)

    def observe(self, sidecar):
        samples = getattr(sidecar, '_samples', ())
        generation = getattr(sidecar, '_diagnostic_generation', None)
        if (sidecar is not self._sidecar or samples is not self._samples
                or generation != self._generation or len(samples) < self._offset
                or (self._offset and samples[self._offset - 1] is not self._tail)):
            self._offset = 0
        self._sidecar, self._samples, self._generation = sidecar, samples, generation
        _increment(self.counters, 'structured_repeat_samples_skipped', self._offset)
        new_samples = samples[self._offset:]
        if not new_samples:
            _increment(self.counters, 'structured_no_new_sample_calls')
            return
        trees = [(sample.action, sample.graph_snapshot,
                  getattr(sample, 'critic_graph_snapshot', None),
                  getattr(sample, 'next_critic_graph_snapshot', None))
                 for sample in new_samples]
        good = finite_sample_trees(trees, self.counters)
        _increment(self.counters, 'structured_samples_checked', len(new_samples))
        _increment(self.counters, 'nonfinite_structured_samples', sum(not flag for flag in good))
        self._offset = len(samples)
        self._tail = samples[-1]
