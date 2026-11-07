"""External native action support; never an actor feature or a post-sample repair.

The current catalog fixes connector endpoints before H2 presence. Only
``gravity_settle`` has optional resource-bearing membership factors. Their
support is restricted before drawing; the immutable support travels with PPO.
"""
from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class CandidateAvailability:
    code: str
    resource_modes: tuple[tuple[str, str], ...] = ()
    conflicts_with: tuple = ()


@dataclass(frozen=True)
class OptionalMembershipAvailability:
    candidate_index: int
    factor: str
    argument_index: int | None
    domain_index: int
    module_id: str
    resource_modes: tuple[tuple[str, str], ...]
    conflicts_with: tuple


@dataclass(frozen=True)
class NativeAvailability:
    candidate_mask: tuple[bool, ...]
    reasons: tuple[CandidateAvailability, ...]
    scope_blocked: tuple[tuple[int, tuple[int, ...]], ...] = ()
    reference_set_blocked: tuple[tuple[int, int, tuple[int, ...]], ...] = ()
    active_claims: tuple = ()
    optional_factors: tuple[OptionalMembershipAvailability, ...] = ()

    def mask_tensor(self, device=None):
        return torch.tensor(self.candidate_mask, dtype=torch.bool, device=device)


def apply_physical_readiness(availability, routing, executor):
    """Mask deterministic dock impossibilities without changing catalog order.

    ``align_faces`` remains available from afar; ``undock`` is already
    enumerated only for current connections. Dock readiness is live physical
    state and is therefore recomputed every policy decision, never stored in
    the BUSY cache.
    """
    resources = routing['resource_candidates']
    dock_rows = []
    for local, (action, binding) in enumerate(routing['structural_bindings']):
        if action.action_id != 'dock':
            continue
        if len(binding.connectors) != 2:
            raise ValueError('Dock readiness requires fixed connector endpoints')
        a, b = binding.connectors
        dock_rows.append((local, (a.module.module_id, a.connector_id,
                                  b.module.module_id, b.connector_id)))
    if not dock_rows or not hasattr(executor, 'dock_readiness_for_pairs'):
        return availability
    readiness = executor.dock_readiness_for_pairs(
        tuple(pair for _, pair in dock_rows)
    )
    if len(readiness) != len(dock_rows):
        raise ValueError('Dock readiness result count differs from dock candidate count')
    mask = list(availability.candidate_mask)
    reasons = list(availability.reasons)
    for (local, _), ready in zip(dock_rows, readiness):
        if ready:
            continue
        index = len(resources) + local
        mask[index] = True
        if reasons[index].code == 'available':
            row = reasons[index]
            reasons[index] = CandidateAvailability(
                'masked_not_ready', row.resource_modes, row.conflicts_with
            )
    return NativeAvailability(
        tuple(mask), tuple(reasons), availability.scope_blocked,
        availability.reference_set_blocked, availability.active_claims,
        availability.optional_factors,
    )



def apply_prepared_handoff(availability, routing, executor):
    """Preserve expert-like ALIGN waves without auto-executing DOCK.

    Prepared components are logical policy reservations, not native resource
    claims. Existing ALIGN goals continue. New actions touching the component
    are hidden until the active ALIGN wave is idle; then only the matching
    prepared DOCK (or an exact re-ALIGN if proximity was lost) is exposed.
    Independent components remain fully asynchronous.
    """
    snapshot = getattr(executor, 'prepared_relation_snapshot', None)
    prepared = tuple(snapshot()) if callable(snapshot) else ()
    if not prepared:
        return availability

    def primitive_name(goal):
        value = getattr(goal.primitive, 'value', goal.primitive)
        return str(value)

    active_edges = []
    for goal in getattr(executor, 'active_goals', ()):
        if primitive_name(goal) != 'align_faces' or len(goal.module_ids) != 2:
            continue
        active_edges.append((str(goal.module_ids[0]), str(goal.module_ids[1])))

    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra

    for relation in prepared:
        union(str(relation.source_module), str(relation.target_module))
    for a, b in active_edges:
        union(a, b)

    # Prepared/active edges describe requested relations; the physical assembly
    # can already contain rigid paths between their endpoints.  Close the union
    # graph over those existing docked components so a sibling branch on the
    # same assembly cannot advance independently.
    connected_component = getattr(executor, 'connected_component_module_ids', None)
    if callable(connected_component):
        seeds = set(parent)
        for module in tuple(seeds):
            for connected in connected_component((module,)):
                union(module, str(connected))

    prepared_roots = {find(str(r.source_module)) for r in prepared}
    components = {}
    for module in tuple(parent):
        root = find(module)
        if root in prepared_roots:
            components.setdefault(root, set()).add(module)
    # Active edges can merge prepared components; normalize roots once more.
    components = {}
    for module in tuple(parent):
        root = find(module)
        if any(find(str(r.source_module)) == root for r in prepared):
            components.setdefault(root, set()).add(module)

    active_roots = {find(a) for a, b in active_edges if find(a) in components}
    undocked = [r for r in prepared if not bool(getattr(r, 'docked', False))]
    pairs = tuple((str(r.source_module), str(r.source_face), str(r.target_module), str(r.target_face))
                  for r in undocked)
    readiness = executor.dock_readiness_for_pairs(pairs) if pairs else ()
    ready = {pair: bool(value) for pair, value in zip(pairs, readiness)}
    relation_by_key = {
        (str(r.source_module), str(r.source_face), str(r.target_module), str(r.target_face)): r
        for r in prepared
    }

    resources = routing['resource_candidates']
    mask = list(availability.candidate_mask)
    reasons = list(availability.reasons)

    def mask_index(index, code):
        if reasons[index].code != 'available':
            return
        row = reasons[index]
        mask[index] = True
        reasons[index] = CandidateAvailability(code, row.resource_modes, row.conflicts_with)

    for index, candidate in enumerate(resources):
        module = str(candidate.resource_ref.module.module_id)
        if module in parent and find(module) in components:
            mask_index(index, 'masked_prepared_conflict')

    for local, (action, binding) in enumerate(routing['structural_bindings']):
        index = len(resources) + local
        connector_key = None
        modules = set()
        connectors = tuple(getattr(binding, 'connectors', ()))
        if len(connectors) == 2:
            a, b = connectors
            connector_key = (str(a.module.module_id), str(a.connector_id),
                             str(b.module.module_id), str(b.connector_id))
            modules.update((connector_key[0], connector_key[2]))
        modules.update(str(m.module_id) for m in getattr(binding, 'required_scope_members', ()))
        initiator = getattr(binding, 'initiator', None)
        if initiator is not None:
            modules.add(str(initiator.module_id))
        roots = {find(module) for module in modules if module in parent and find(module) in components}
        if not roots:
            continue
        root = next(iter(roots))
        action_id = str(action.action_id)
        relation = relation_by_key.get(connector_key) if connector_key is not None else None
        relation_open = relation is not None and not bool(getattr(relation, 'docked', False))
        wave_active = root in active_roots

        if action_id == 'dock' and relation_open:
            if wave_active:
                mask_index(index, 'masked_handoff_wait')
            elif not ready.get(connector_key, False):
                mask_index(index, 'masked_handoff_not_ready')
            continue
        if action_id == 'align_faces' and relation_open:
            if not wave_active and not ready.get(connector_key, False):
                # Geometry was disturbed after preparation: only the exact
                # relation may reacquire alignment before docking.
                continue
        mask_index(index, 'masked_prepared_conflict')

    return NativeAvailability(tuple(mask), tuple(reasons), availability.scope_blocked,
        availability.reference_set_blocked, availability.active_claims,
        availability.optional_factors)

def capture_native_availability(*, routing, graph, hard_mask, executor):
    """Capture current native support using executor admission's claim resolver.

    There is no physics step/admission between snapshot and capture. No Expert,
    task identity, target topology, predicted logits or sampled args are read.
    Unknown future action declarations require an explicit correct mask contract.
    """
    from smores_ep.primitives.model import PrimitiveGoal
    active = executor.active_resource_snapshot()
    by_goal = {row.goal_id: row for row in active}
    modules = routing['module_refs']
    resources = routing['resource_candidates']
    count = len(resources) + len(routing['structural_bindings'])
    base = torch.as_tensor(hard_mask, dtype=torch.bool).detach().cpu().tolist()
    if len(base) != count:
        raise ValueError('Native availability candidate count differs from hard mask')
    reasons = []
    scope_blocked, reference_set_blocked = [], []
    optional_factors = []

    def check(modes):
        modes = tuple(sorted(modes))
        conflicts = executor.resource_claim_conflicts(dict(modes))
        holders = tuple(by_goal[g] for g in sorted({g for ids in conflicts.values() for g in ids}))
        return CandidateAvailability('masked_busy' if holders else 'available', modes, holders)

    def passive(module_ids):
        goal = PrimitiveGoal.from_dict(dict(goal_id='availability-preview', primitive='gravity_settle',
            module_ids=list(module_ids), parameters=dict(duration_s=1., passive_module_ids=list(module_ids))))
        return executor.resource_claims_for(goal)

    for row in resources:
        ref = row.resource_ref
        reasons.append(check(((f'{ref.resource_id}:{ref.module.module_id}', 'exclusive'),)))

    structural = graph.get('structural', {})
    for local, (action, binding) in enumerate(routing['structural_bindings']):
        index = len(resources) + local
        name = action.action_id
        if name in ('align_faces', 'dock', 'undock'):
            if len(binding.connectors) != 2:
                raise ValueError('Native pair availability requires fixed endpoints')
            a, b = binding.connectors
            goal = PrimitiveGoal.from_dict(dict(goal_id='availability-preview', primitive=name,
                module_ids=[a.module.module_id, b.module.module_id],
                parameters=dict(face_a=a.connector_id, face_b=b.connector_id)))
            reasons.append(check(executor.resource_claims_for(goal)))
        elif name == 'gravity_settle':
            required = {m.module_id for m in binding.required_scope_members}
            required.add(binding.initiator.module_id)
            module_status = {i: check(passive((m.module_id,))) for i, m in enumerate(modules)}
            blocked_modules = {i for i, status in module_status.items() if status.code == 'masked_busy'}
            start, end = structural['scope_ptr'][local:local+2].tolist()
            scope_indices = structural['scope_members'].tolist()
            scope_required = structural['scope_required'].tolist()
            scope_blocked.append((index, tuple(i for i in range(start, end)
                if not scope_required[i] and scope_indices[i] in blocked_modules)))
            for domain_index in scope_blocked[-1][1]:
                module = scope_indices[domain_index]
                status = module_status[module]
                optional_factors.append(OptionalMembershipAvailability(index, 'scope', None,
                    domain_index, modules[module].module_id, status.resource_modes, status.conflicts_with))
            for argument, domain in enumerate(routing['argument_domains']):
                if int(structural['argument_binding'][argument]) != local:
                    continue
                if domain.parameter.name != 'passive_module_ids':
                    continue
                if domain.parameter.kind != 'module_refs':
                    raise ValueError('Unsupported gravity_settle reference declaration')
                lo, hi = structural['argument_domain_ptr'][argument:argument+2].tolist()
                refs = structural['argument_domain_index'].tolist()
                req = structural['argument_domain_required'].tolist()
                required.update(modules[refs[i]].module_id for i in range(lo, hi) if req[i])
                reference_set_blocked.append((index, argument, tuple(i for i in range(lo, hi)
                    if not req[i] and refs[i] in blocked_modules)))
                for domain_index in reference_set_blocked[-1][2]:
                    module = refs[domain_index]
                    status = module_status[module]
                    optional_factors.append(OptionalMembershipAvailability(index, 'reference_set', argument,
                        domain_index, modules[module].module_id, status.resource_modes, status.conflicts_with))
            reasons.append(check(passive(tuple(sorted(required)))))
        else:
            raise ValueError(f'No native feasibility factorization contract for {name}')

    masks = tuple(bool(base[i] or row.code != 'available') for i, row in enumerate(reasons))
    reasons = tuple(CandidateAvailability('masked_unavailable', row.resource_modes, row.conflicts_with)
                    if base[i] else row for i, row in enumerate(reasons))
    availability = NativeAvailability(
        masks, reasons, tuple(scope_blocked), tuple(reference_set_blocked), active,
        tuple(optional_factors),
    )
    availability = apply_physical_readiness(availability, routing, executor)
    return apply_prepared_handoff(availability, routing, executor)


def apply_availability(actor_output, availability):
    """Restrict membership support without modifying raw H2 tensors or shapes.

    Filter optional factor views (not action samples). A removed factor is a
    deterministic absent value with zero log_prob/entropy and no RNG draw. The
    same operation on rollout and update reconstructs identical support.
    """
    if availability is None:
        return actor_output
    if hasattr(actor_output['structural_payloads'], 'with_availability'):
        return {**actor_output, 'structural_payloads': actor_output['structural_payloads'].with_availability(availability)}
    payloads = dict(actor_output['structural_payloads'])
    scopes = dict(availability.scope_blocked)
    references = {(i, a): blocked for i, a, blocked in availability.reference_set_blocked}

    for index, payload in payloads.items():
        payloads[index] = restrict_structural_payload(payload, index, scopes, references)
    return {**actor_output, 'structural_payloads': payloads}


def restrict_structural_payload(payload, index, scopes, references):
    """Identical support restriction, also usable by a selected-only lazy view."""
    def restrict(view, blocked):
        if not blocked:
            return view
        blocked = torch.tensor(blocked, dtype=view.optional_domain_indices.dtype,
                               device=view.optional_domain_indices.device)
        keep = ~torch.isin(view.optional_domain_indices, blocked)
        return replace(view, optional_logits=view.optional_logits[keep],
                       optional_domain_indices=view.optional_domain_indices[keep])

    return replace(payload,
        scope=restrict(payload.scope, scopes.get(index, ())) if payload.scope is not None else None,
        reference_set=tuple(restrict(a, references.get((index, a.argument_index), ()))
                            for a in payload.reference_set))
