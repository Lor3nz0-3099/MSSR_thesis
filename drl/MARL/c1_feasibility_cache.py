"""Versioned native BUSY support. Physical/base feasibility always remains live.

No predicted logits, actor features, morphology targets or sampled actions are
cache inputs. Static routing metadata is compared by value, including optional
factor index layout; native claim semantics still come from the executor.
"""
from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter

import torch

from drl.MARL.c1_feasibility import (
    CandidateAvailability, NativeAvailability, OptionalMembershipAvailability,
    apply_physical_readiness, capture_native_availability,
)

INDEX_FIELDS = ('scope_ptr', 'scope_members', 'scope_required', 'argument_binding',
                'argument_domain_ptr', 'argument_domain_index', 'argument_domain_required')
COUNTERS = ('mask_cache_hits', 'mask_cache_misses', 'mask_recomputes',
            'active_claim_version_changes', 'claim_epoch_changes',
            'static_claim_precompute_build_count', 'static_candidate_claim_build_count')


def routing_layout(routing, graph):
    """Read routing indexes in bulk, never scalar-by-scalar from GPU tensors.

    Base hard-mask values and physical graph measurements are deliberately not
    in this static key. Changes to either are handled outside BUSY caching.
    """
    structural = graph.get('structural', {})
    metadata = tuple(tuple(structural[k].detach().cpu().tolist()) if k in structural else None
                     for k in INDEX_FIELDS)
    resources = tuple((c.resource_ref.module.module_id, c.resource_ref.resource_id)
                      for c in routing['resource_candidates'])
    bindings = tuple((action.action_id, binding.initiator.module_id,
        tuple((p.module.module_id, p.connector_id) for p in binding.connectors),
        tuple(sorted(m.module_id for m in binding.required_scope_members)))
        for action, binding in routing['structural_bindings'])
    parameters = tuple((d.parameter.name, d.parameter.kind) for d in routing.get('argument_domains', ()))
    modules = tuple(m.module_id for m in routing['module_refs'])
    return (modules, resources, bindings, parameters, metadata)


@dataclass(frozen=True)
class StaticClaims:
    groups: tuple  # Interned tuples of native (resource key, ownership mode).
    candidate_groups: tuple[int, ...]
    scope_members: tuple  # (candidate, ((domain index, module ID, group index), ...))
    reference_members: tuple  # (candidate, argument, members)


def precompute_claims(layout_key, executor):
    """Resolve each fixed native claim template once for this static layout."""
    from smores_ep.primitives.model import PrimitiveGoal
    modules, resources, bindings, parameters, values = layout_key
    meta = dict(zip(INDEX_FIELDS, values))
    groups, group_ids, candidate_groups = [], {}, []
    scopes, references, passive_groups = [], [], {}

    def intern(modes):
        modes = tuple(sorted(modes))
        if modes not in group_ids:
            group_ids[modes] = len(groups)
            groups.append(modes)
        return group_ids[modes]

    def passive(members):
        members = tuple(sorted(members))
        if members not in passive_groups:
            goal = PrimitiveGoal.from_dict(dict(goal_id='availability-preview', primitive='gravity_settle',
                module_ids=list(members), parameters=dict(duration_s=1., passive_module_ids=list(members))))
            passive_groups[members] = intern(executor.resource_claims_for(goal))
        return passive_groups[members]

    for module, resource in resources:
        candidate_groups.append(intern(((f'{resource}:{module}', 'exclusive'),)))

    args_by_binding = {}
    if meta['argument_binding'] is not None:
        for argument, binding in enumerate(meta['argument_binding']):
            args_by_binding.setdefault(binding, []).append(argument)
    for local, (name, initiator, ports, required_members) in enumerate(bindings):
        index = len(resources) + local
        if name in ('align_faces', 'dock', 'undock'):
            if len(ports) != 2:
                raise ValueError('Native pair availability requires fixed endpoints')
            goal = PrimitiveGoal.from_dict(dict(goal_id='availability-preview', primitive=name,
                module_ids=[p[0] for p in ports], parameters=dict(face_a=ports[0][1], face_b=ports[1][1])))
            candidate_groups.append(intern(executor.resource_claims_for(goal)))
        elif name == 'gravity_settle':
            required = set(required_members) | {initiator}
            start, end = meta['scope_ptr'][local:local+2]
            scopes.append((index, tuple((i, modules[meta['scope_members'][i]],
                passive((modules[meta['scope_members'][i]],))) for i in range(start, end)
                if not meta['scope_required'][i])))
            for argument in args_by_binding.get(local, ()):
                param_name, kind = parameters[argument]
                if param_name != 'passive_module_ids':
                    continue
                if kind != 'module_refs':
                    raise ValueError('Unsupported gravity_settle reference declaration')
                lo, hi = meta['argument_domain_ptr'][argument:argument+2]
                refs, req = meta['argument_domain_index'], meta['argument_domain_required']
                required.update(modules[refs[i]] for i in range(lo, hi) if req[i])
                references.append((index, argument, tuple((i, modules[refs[i]], passive((modules[refs[i]],)))
                    for i in range(lo, hi) if not req[i])))
            candidate_groups.append(passive(required))
        else:
            raise ValueError(f'No native feasibility factorization contract for {name}')
    return StaticClaims(tuple(groups), tuple(candidate_groups), tuple(scopes), tuple(references))


def resolve_busy(layout, executor):
    """Recompute only when native ownership or static routing changes."""
    active = executor.active_resource_snapshot()
    by_goal = {row.goal_id: row for row in active}
    statuses = []
    for modes in layout.groups:
        conflicts = executor.resource_claim_conflicts(dict(modes))
        holders = tuple(by_goal[g] for g in sorted({g for ids in conflicts.values() for g in ids}))
        statuses.append(CandidateAvailability('masked_busy' if holders else 'available', modes, holders))
    reasons = tuple(statuses[g] for g in layout.candidate_groups)
    scope_blocked, reference_blocked, optional = [], [], []
    # Preserve the oracle's per-candidate scope/reference factor ordering.
    reference_by_candidate = {}
    for index, argument, members in layout.reference_members:
        reference_by_candidate.setdefault(index, []).append((argument, members))
    for index, members in layout.scope_members:
        blocked = []
        for domain, module, group in members:
            status = statuses[group]
            if status.code == 'masked_busy':
                blocked.append(domain)
                optional.append(OptionalMembershipAvailability(index, 'scope', None,
                    domain, module, status.resource_modes, status.conflicts_with))
        scope_blocked.append((index, tuple(blocked)))
        for argument, members in reference_by_candidate.get(index, ()):
            blocked = []
            for domain, module, group in members:
                status = statuses[group]
                if status.code == 'masked_busy':
                    blocked.append(domain)
                    optional.append(OptionalMembershipAvailability(index, 'reference_set', argument,
                        domain, module, status.resource_modes, status.conflicts_with))
            reference_blocked.append((index, argument, tuple(blocked)))
    return NativeAvailability(tuple(r.code != 'available' for r in reasons), reasons,
        tuple(scope_blocked), tuple(reference_blocked), active, tuple(optional))


def compose_base(busy, hard_mask):
    """A changing external/base feasibility mask is never hidden by a cache hit."""
    base = torch.as_tensor(hard_mask, dtype=torch.bool).detach().cpu().tolist()
    if len(base) != len(busy.candidate_mask):
        raise ValueError('Native availability candidate count differs from hard mask')
    if not any(base):
        return busy
    reasons = tuple(CandidateAvailability('masked_unavailable', row.resource_modes, row.conflicts_with)
                    if base[i] else row for i, row in enumerate(busy.reasons))
    return NativeAvailability(tuple(bool(base[i] or busy.candidate_mask[i]) for i in range(len(base))),
        reasons, busy.scope_blocked, busy.reference_set_blocked, busy.active_claims, busy.optional_factors)


class NativeAvailabilityCache:
    """One env's bounded static layouts and one immutable native BUSY snapshot."""
    def __init__(self, *, max_layouts=4):
        if max_layouts < 1:
            raise ValueError('Availability cache needs at least one static layout')
        self.max_layouts = max_layouts
        self._layouts = OrderedDict()
        self._last_layout_key = self._last_layout = None
        self._busy_layout = self._busy_version = self._busy = None
        self._executor = None
        self._generation = None
        self._observed_version = 0
        self._stats = dict.fromkeys(COUNTERS, 0)
        self._masking_s = 0.

    def _observe(self, executor, generation):
        version = getattr(executor, 'active_claim_version', None)
        if executor is not self._executor or generation != self._generation:
            if self._executor is not None:
                old_version = getattr(self._executor, 'active_claim_version', self._observed_version)
                self._stats['active_claim_version_changes'] += max(0, old_version - self._observed_version) + 1
                self._stats['claim_epoch_changes'] += 1
            if executor is not self._executor:
                self._stats['active_claim_version_changes'] += version or 0
            self._executor, self._generation = executor, generation
            self._observed_version = version or 0
            self._busy_layout = self._busy_version = self._busy = None
        elif version is not None:
            if version < self._observed_version:
                raise RuntimeError('Native claim version regressed within an executor epoch')
            self._stats['active_claim_version_changes'] += version - self._observed_version
            self._observed_version = version
        return version

    def capture(self, *, routing, graph, hard_mask, executor, generation=None):
        start = perf_counter()
        try:
            version = self._observe(executor, generation)
            if version is None:
                # Test/legacy adapters without versioning must stay uncached.
                self._stats['mask_cache_misses'] += 1
                self._stats['mask_recomputes'] += 1
                return capture_native_availability(routing=routing, graph=graph,
                    hard_mask=hard_mask, executor=executor)
            key = routing_layout(routing, graph)
            if key == self._last_layout_key:
                layout = self._last_layout
            else:
                layout = self._layouts.pop(key, None)
                if layout is None:
                    layout = precompute_claims(key, executor)
                    self._stats['static_claim_precompute_build_count'] += 1
                    self._stats['static_candidate_claim_build_count'] += len(layout.candidate_groups)
                self._layouts[key] = layout
                while len(self._layouts) > self.max_layouts:
                    self._layouts.popitem(last=False)
                self._last_layout_key, self._last_layout = key, layout
            if layout is self._busy_layout and version == self._busy_version:
                self._stats['mask_cache_hits'] += 1
            else:
                self._stats['mask_cache_misses'] += 1
                self._stats['mask_recomputes'] += 1
                self._busy = resolve_busy(layout, executor)
                self._busy_layout, self._busy_version = layout, version
            from drl.MARL.c1_feasibility import apply_prepared_handoff
            availability = apply_physical_readiness(
                compose_base(self._busy, hard_mask), routing, executor
            )
            return apply_prepared_handoff(availability, routing, executor)
        finally:
            self._masking_s += perf_counter() - start

    def metrics(self, *, executor=None, generation=None):
        if executor is not None:
            self._observe(executor, generation)
        values = dict(self._stats)
        # Include last-step mutations even when no following action is captured.
        version = getattr(self._executor, 'active_claim_version', self._observed_version)
        values['active_claim_version_changes'] += max(0, version - self._observed_version)
        return dict(values, feasibility_masking_s=self._masking_s,
            native_active_claim_version=version, generation=self._generation,
            versioning_available=hasattr(self._executor, 'active_claim_version'))
