"""Bounded, label-free runtime catalog/packing caches. No dynamic tensor inputs.

Keys cover exactly candidate admission metadata; physical features, geometry,
contacts, latents, predictions and parameter values are never cached here.
"""
from collections import OrderedDict
from itertools import count
from dataclasses import dataclass
import json

from .bc_v1_contract import robot_spec_fingerprint
from .candidates import enumerate_candidates

_IDENTITIES = count()


@dataclass(frozen=True)
class DecoderLayout:
    token: tuple
    argument_rows: tuple
    scope_ptr: tuple
    members: tuple
    required: tuple
    resource_count: int


def _clone_tree(value):
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    return value.clone()


def catalog_signature(graph):
    """Fields consumed by candidates.py, including input order and peers."""
    return (
        tuple((m.ref, m.module_type_id) for m in graph.modules),
        tuple((p.ref, p.kind, p.status, p.connected_to) for p in graph.connectors),
        tuple((r.ref, tuple(c.coordinate_id for c in r.coordinates)) for r in graph.resources),
        tuple((e.endpoint_a, e.endpoint_b, e.relation_id) for e in graph.connections),
    )


class RuntimeStaticCache:
    def __init__(self, max_layouts=8):
        if type(max_layouts) is not int or max_layouts < 1:
            raise ValueError('Positive bounded cache size required')
        self.max_layouts = max_layouts
        self._identity = next(_IDENTITIES)
        self._serial = count()
        self._catalogs, self._layouts, self._specs = OrderedDict(), OrderedDict(), OrderedDict()
        self._decoders = OrderedDict()
        self._epoch = None
        self._stats = dict(catalog_hits=0, catalog_builds=0, layout_hits=0, layout_builds=0, epoch_invalidations=0)

    def _spec_key(self, spec):
        # RobotSpec is frozen. Keep a strong reference while memoizing its id.
        entry = self._specs.pop(id(spec), None)
        if entry is None or entry[0] is not spec:
            entry = (spec, robot_spec_fingerprint(spec))
        self._specs[id(spec)] = entry
        while len(self._specs) > self.max_layouts:
            self._specs.popitem(last=False)
        return entry[1]

    def set_epoch(self, epoch):
        if epoch == self._epoch:
            return
        if self._catalogs or self._layouts:
            self._catalogs.clear()
            self._layouts.clear()
            self._decoders.clear()
            self._stats['epoch_invalidations'] += 1
        self._epoch = epoch

    def enumerate(self, graph, spec, schema):
        key = (self._spec_key(spec), json.dumps(schema, sort_keys=True, separators=(',', ':')),
               catalog_signature(graph))
        table = self._catalogs.pop(key, None)
        if table is None:
            table = enumerate_candidates(graph, spec, schema)
            self._stats['catalog_builds'] += 1
        else:
            self._stats['catalog_hits'] += 1
        self._catalogs[key] = table
        while len(self._catalogs) > self.max_layouts:
            self._catalogs.popitem(last=False)
        return table

    def pack(self, table, spec, indices, routing, builder):
        key = (id(table), self._spec_key(spec), routing.get('layout_fingerprint'),
               tuple((name, tuple(indices[name].items())) for name in ('module', 'connector', 'resource')),
               tuple(routing['module_types']))
        entry = self._layouts.pop(key, None)
        if entry is None:
            local = dict(routing)
            resources, structural = builder(table, spec, indices, local)
            names = ('actions', 'parameter_catalog', 'mode_catalog', 'resource_candidates',
                     'structural_bindings', 'argument_domains', 'parameter_kinds')
            metadata = {name: local[name] for name in names}
            metadata['runtime_candidate_layout_token'] = (self._identity, next(self._serial))
            # Strong table reference prevents object-id reuse. Templates stay private.
            entry = (table, resources, structural, metadata)
            self._stats['layout_builds'] += 1
        else:
            self._stats['layout_hits'] += 1
        self._layouts[key] = entry
        while len(self._layouts) > self.max_layouts:
            self._layouts.popitem(last=False)
        _, resources, structural, metadata = entry
        routing.update(metadata)
        # Preserve the old fresh-tensor ownership contract for direct callers.
        return _clone_tree(resources), _clone_tree(structural)

    def metrics(self):
        return dict(self._stats, catalog_entries=len(self._catalogs), layout_entries=len(self._layouts))

    def decoder_layout(self, sample):
        token = sample['routing']['runtime_candidate_layout_token']
        layout = self._decoders.pop(token, None)
        if layout is None:
            st = sample['inputs']['structural']
            rows = [[] for _ in range(len(st['initiator']))]
            for index,binding in enumerate(st['argument_binding'].tolist()):
                rows[binding].append(index)
            layout = DecoderLayout(token, tuple(tuple(row) for row in rows),
                tuple(st['scope_ptr'].tolist()), tuple(st['scope_members'].tolist()),
                tuple(st['scope_required'].tolist()), len(sample['inputs']['resource_candidates']['resource']))
        self._decoders[token] = layout
        while len(self._decoders) > self.max_layouts:
            self._decoders.popitem(last=False)
        return layout
