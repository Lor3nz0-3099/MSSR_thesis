"""Opt-in BC-v5 candidate admission from the CURRENT observed topology.

Does not change BC-v4 candidate enumeration, stored G1/G2, or old checkpoints.
No expert labels, future topology, task phases, or executor FSM are consulted.

An align_faces command *moves its initiator*: only a certified singleton
module can initiate it. Its peer may be a member of an attached assembly.
The peer's face occupancy is deliberately not an admission criterion for
alignment; this is not a claim that docking on that face is feasible.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations

from .bc_v1_contract import RobotSpec, StateGraph, validate_robot_spec
from .candidates import (
    CandidateTables, StructuralCandidate, _arguments, _canonical_action,
    _current_catalog, _pair_bindings, _physical_connections,
    _resource_candidates, _skill_bindings,
)
from .feature_schema import validate_feature_schema


def topology_index(graph: StateGraph, ports):
    """Module component IDs and modules with a *certified* physical attachment.

    Connected_to and current_connection are positive evidence. A connector
    status of connected without known peer is conservatively non-singleton:
    we cannot certify the module is completely detached.
    Contact/rolling_contact alone never creates a docked component.
    """
    edges, occupied = _physical_connections(graph, ports)
    parent = {module.ref: module.ref for module in graph.modules}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a.module), find(b.module)
        if ra != rb:
            parent[rb] = ra
    # Equality of component tokens matters, never token ordering or ID values.
    components = {ref: find(ref) for ref in parent}
    physically_attached = {a.module for edge in edges for a in edge}
    possibly_attached = physically_attached | {
        ref.module for ref, port in ports.items()
        if port.status == 'connected' or port.connected_to is not None
    }
    singletons = frozenset(parent.keys() - possibly_attached)
    return components, frozenset(physically_attached), singletons, occupied


def enumerate_candidates_v5(graph: StateGraph, spec: RobotSpec, schema) -> CandidateTables:
    """Proposed v5 structural admission; use the *same* function in G2 and live.

    Required full-dataset positive-retention audit is deliberately external:
    do not replace a production candidate generator just from synthetic tests.
    """
    if not isinstance(graph, StateGraph):
        raise ValueError('graph: expected current StateGraph')
    validate_robot_spec(spec)
    validate_feature_schema(schema)
    owners, ports, port_specs, resources, resource_specs = _current_catalog(graph, spec)
    connected, occupied = _physical_connections(graph, ports)
    components, attached, singleton, _ = topology_index(graph, ports)
    resources_out = _resource_candidates(owners, ports, resources, resource_specs, spec)

    structural = []
    rules = schema['candidate_policy']['actions']
    for raw in spec.actions:
        if not raw.enabled or raw.action_id not in rules:
            continue
        action = _canonical_action(raw)
        rule = rules[action.action_id]
        if action.kind != rule['kind']:
            raise ValueError('Structural action kind disagrees with candidate policy')
        if rule['endpoints'] == 'none':
            bindings = _skill_bindings(action, owners, ports, port_specs, resource_specs)
            if action.action_id == 'gravity_settle':
                bindings = frozenset(b for b in bindings if b.initiator in attached)
            if bindings:
                structural.append(StructuralCandidate(action, frozenset(), bindings))
            continue
        if action.scope != 'connector_pair':
            raise ValueError('Structural pair requires connector_pair scope')
        domains = _arguments(action, owners, ports)
        if domains is None:
            continue
        if action.action_id == 'undock':
            pairs = connected
        elif action.action_id == 'align_faces':
            # Allow occupied *peer* connectors; alignment only permits a
            # certified singleton initiator, not a moving multi-module group.
            pairs = combinations(ports, 2)
        elif action.action_id == 'dock':
            # A dock must use two unoccupied connector endpoints.
            pairs = combinations((ref for ref in ports if ref not in occupied), 2)
        else:
            # Preserve catalog behavior for unrelated connector_pair actions.
            pairs = (connected if rule['endpoints'] == 'current_connection_pair'
                     else combinations((ref for ref in ports if ref not in occupied), 2))
        for pair in pairs:
            a, b = pair
            if a.module == b.module:
                continue
            if action.action_id in ('align_faces', 'dock') and components[a.module] == components[b.module]:
                continue
            bindings = _pair_bindings(action, pair, owners, ports, port_specs,
                                      action.action_id == 'undock', domains)
            if action.action_id == 'align_faces':
                bindings = frozenset(binding for binding in bindings
                                     if binding.initiator in singleton)
            if bindings:
                structural.append(StructuralCandidate(action, frozenset(pair), bindings))
    return CandidateTables(resources_out, tuple(structural))
