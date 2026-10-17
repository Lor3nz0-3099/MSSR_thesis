"""Label-independent current action domains for BC-v1 (stdlib only).

Tables describe catalog/capability/topology admission, never executor feasibility
or supervision eligibility. A physical connector pair has one row per action;
its bindings retain distinct initiator decisions and ordered argument roles.
Reference domains remain symbolic: no ID encodings, value grids or tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from typing import Literal, Mapping

from .bc_v1_contract import (
    ActionSpec, ConnectorRef, CoordinateSpec, ModuleRef, ParameterSpec,
    ResourceModeSpec, ResourceRef, RobotSpec, StateGraph, validate_robot_spec,
)
from .feature_schema import validate_feature_schema


Reference = ModuleRef | ConnectorRef
CoordinateRef = tuple[ResourceRef, str]


@dataclass(frozen=True)
class ArgumentDomain:
    """An intact catalog declaration, with current reference options if needed.

    Numeric/category domains come from parameter limits/units/categories.
    Reference-set size is exact when declared, otherwise a selectable subset.
    bound_ref fixes an ordered endpoint argument; no label supplies its value.
    Nonlearnable declarations are retained for later configured dispatch.
    """

    parameter: ParameterSpec
    options: frozenset[Reference] = frozenset()
    bound_ref: Reference | None = None


@dataclass(frozen=True)
class ResourceCandidate:
    """One current resource + enabled catalog mode/action, not per-coordinate.

    coordinates is the catalog of observables; coordinate_refs identifies those
    present in the current graph. Modes have independent payload ordering. No
    association between a mode and a coordinate is guessed from either ID.
    """

    resource_ref: ResourceRef
    action: ActionSpec
    mode: ResourceModeSpec
    coordinates: tuple[CoordinateSpec, ...]
    coordinate_refs: frozenset[CoordinateRef]
    capabilities: tuple[str, ...]
    arguments: tuple[ArgumentDomain, ...]


@dataclass(frozen=True)
class StructuralBinding:
    """An initiator/argument choice inside a physical candidate row.

    Pair scope is fixed to both modules. Module skill scope is the singleton;
    multi-module skill scope is a selectable domain containing the initiator,
    further constrained by reference-set arguments. This avoids an exponential
    enumeration of every possible subset. 'available' certifies only current
    catalog/topology admission; it NEVER certifies runtime trajectory feasibility.
    """

    initiator: ModuleRef
    connectors: tuple[ConnectorRef, ...]
    scope_members: frozenset[ModuleRef]
    required_scope_members: frozenset[ModuleRef]
    arguments: tuple[ArgumentDomain, ...]
    applicability: Literal['available', 'unknown']


@dataclass(frozen=True)
class StructuralCandidate:
    """One action/physical pair (or endpoint-free skill), with explicit bindings.

    Endpoints and bindings are unordered immutable routing sets. Ordered source /
    peer roles live inside each binding, not in duplicate reversed pair rows.
    """

    action: ActionSpec
    endpoints: frozenset[ConnectorRef]
    bindings: frozenset[StructuralBinding]


@dataclass(frozen=True)
class CandidateTables:
    resource_candidates: tuple[ResourceCandidate, ...] = ()
    structural_candidates: tuple[StructuralCandidate, ...] = ()


def _index(items, key, path):
    result = {}
    for item in items:
        ref = key(item)
        if ref in result:
            raise ValueError(f'{path}: duplicate current reference')
        result[ref] = item
    return result


def _lookup(items, key, path):
    if key not in items:
        raise ValueError(f'{path}: reference absent from current graph/catalog')
    return items[key]


def _current_catalog(graph, spec):
    """Resolve only membership/catalog metadata; never inspect feature payloads."""
    modules = _index(graph.modules, lambda x:x.ref, 'modules')
    types = {x.type_id:x for x in spec.module_types}
    owners = {ref:_lookup(types,m.module_type_id,'module_type_id') for ref,m in modules.items()}
    ports = _index(graph.connectors, lambda x:x.ref, 'connectors')
    port_specs = {}
    for ref, port in ports.items():
        owner = _lookup(owners,ref.module,'connector.module')
        declaration = _lookup({x.connector_id:x for x in owner.connectors},ref.connector_id,'connector.id')
        if port.kind!=declaration.kind or port.status not in ('available','connected','unknown'):
            raise ValueError('connector: kind/status disagrees with current catalog')
        port_specs[ref] = declaration
    resources = _index(graph.resources, lambda x:x.ref, 'resources')
    resource_specs = {}
    for ref, resource in resources.items():
        owner = _lookup(owners,ref.module,'resource.module')
        declaration = _lookup({x.resource_id:x for x in owner.resources},ref.resource_id,'resource.id')
        current = _index(resource.coordinates, lambda x:x.coordinate_id, 'resource.coordinates')
        coordinates = {x.coordinate_id:x for x in declaration.coordinates}
        for coordinate in current:
            _lookup(coordinates,coordinate,'coordinate.id')
        resource_specs[ref] = declaration
    return owners,ports,port_specs,resources,resource_specs


def _physical_connections(graph, ports):
    """Union of certified connection edges and current connected_to pointers.

    Contact/rolling_contact alone is not a docked attachment. Duplicate/reversed
    connection records collapse physically. A contradictory peer is an error;
    missing/unknown status never fabricates an undock pair.
    """
    pairs,peers = {},{}

    def add(a,b):
        _lookup(ports,a,'connection.endpoint');_lookup(ports,b,'connection.endpoint')
        if a==b or a.module==b.module:
            raise ValueError('connection: endpoints must belong to distinct modules')
        for own,peer in ((a,b),(b,a)):
            if own in peers and peers[own]!=peer:
                raise ValueError('connection: contradictory current peer')
            peers[own] = peer
        pairs.setdefault(frozenset((a,b)),(a,b))

    for edge in graph.connections:
        _lookup(ports,edge.endpoint_a,'connection.endpoint');_lookup(ports,edge.endpoint_b,'connection.endpoint')
        if edge.relation_id=='current_connection':
            add(edge.endpoint_a,edge.endpoint_b)
        elif edge.relation_id not in ('contact','rolling_contact'):
            raise ValueError('connection: unknown current relation')
    for ref,port in ports.items():
        if port.connected_to is not None:
            add(ref,port.connected_to)
    occupied = frozenset((*peers,*(ref for ref,p in ports.items() if p.status=='connected')))
    return tuple(pairs.values()),occupied


def _reference_options(parameter, owners, ports):
    def owner_ok(ref):
        return parameter.module_type_id is None or owners[ref].type_id==parameter.module_type_id
    if parameter.kind in ('module_ref','module_refs'):
        return frozenset(ref for ref in owners if owner_ok(ref))
    if parameter.kind in ('connector_ref','connector_refs'):
        return frozenset(ref for ref in ports if owner_ok(ref.module) and
                         (parameter.connector_id is None or ref.connector_id==parameter.connector_id))
    return frozenset()


def _canonical_action(action):
    # A2 catalogs are unordered and load_robot_spec normalizes by name. Apply
    # that same convention to directly constructed specs, without mutating them.
    # The first two connector-ref slots in canonical order bind source and peer;
    # instance IDs and current graph collection order never choose those roles.
    return replace(action,parameters=tuple(sorted(action.parameters,key=lambda p:p.name)))


def _arguments(action, owners, ports, endpoints=(), domains=None):
    if domains is None:
        domains = tuple(ArgumentDomain(
            parameter, _reference_options(parameter, owners, ports))
            for parameter in action.parameters)
    elif (len(domains) != len(action.parameters)
            or any(domain.parameter != parameter
                   for domain, parameter in zip(domains, action.parameters))):
        raise ValueError('argument domain/action mismatch')

    bound_domains = []
    endpoint_index = 0
    for domain in domains:
        parameter = domain.parameter
        options = domain.options
        bound = None
        if parameter.kind=='connector_ref' and endpoint_index<len(endpoints):
            bound = endpoints[endpoint_index]
            endpoint_index += 1
            if bound not in options:
                return None
        if parameter.kind in ('module_ref','module_refs','connector_ref','connector_refs'):
            if not options or len(options)<(parameter.size or 1):
                return None
        bound_domains.append(domain if bound is None else replace(domain,bound_ref=bound))
    return tuple(bound_domains)


def _resource_candidates(owners, ports, resources, declarations, spec):
    actions = {x.action_id:_canonical_action(x) for x in spec.actions}
    rows = []
    for ref,state in resources.items():
        declaration = declarations[ref]
        owner = owners[ref.module]
        capabilities = set(owner.capabilities) | set(declaration.capabilities)
        for mode in declaration.modes:
            action = actions.get(mode.mode_id)
            if action is None or not action.enabled or action.scope!='resource':
                continue
            if action.kind!=mode.kind:
                raise ValueError('resource action kind disagrees with mode kind')
            if not set(action.required_capabilities)<=capabilities:
                continue
            if any((p.module_type_id is not None and p.module_type_id!=owner.type_id) or
                   (p.resource_id is not None and p.resource_id!=ref.resource_id)
                   for p in action.parameters if p.kind not in ('module_ref','module_refs','connector_ref','connector_refs')):
                continue
            arguments = _arguments(action,owners,ports)
            if arguments is not None:
                rows.append(ResourceCandidate(ref,action,mode,declaration.coordinates,
                    frozenset((ref,x.coordinate_id) for x in state.coordinates),declaration.capabilities,arguments))
    return tuple(rows)


def _pair_bindings(action, endpoints, owners, ports, declarations, connected, domains):
    bindings = []
    for source,peer in (endpoints,endpoints[::-1]):
        # v1 compatibility is capability + typed endpoint-argument admission.
        # No gender/mating matrix, face names, pose or morphology is invented.
        if any(not set(action.required_capabilities)<=
               (set(owners[ref.module].capabilities)|set(declarations[ref].capabilities))
               for ref in (source,peer)):
            continue
        arguments = _arguments(action,owners,ports,(source,peer),domains)
        if arguments is None:
            continue
        scope = frozenset((source.module,peer.module))
        applicability = 'available' if connected or all(ports[r].status=='available' for r in (source,peer)) else 'unknown'
        bindings.append(StructuralBinding(source.module,(source,peer),scope,scope,arguments,applicability))
    return frozenset(bindings)


def _skill_bindings(action, owners, ports, port_specs, resource_specs):
    if action.scope not in ('module','modules'):
        raise ValueError('endpoint-free skill requires module/modules scope')
    eligible = set()
    for ref,owner in owners.items():
        capabilities = set(owner.capabilities)
        capabilities.update(cap for r,s in resource_specs.items() if r.module==ref for cap in s.capabilities)
        capabilities.update(cap for r,s in port_specs.items() if r.module==ref for cap in s.capabilities)
        if set(action.required_capabilities)<=capabilities:
            eligible.add(ref)
    arguments = _arguments(action,owners,ports)
    if arguments is None:
        return frozenset()
    # The v1 multi-module skill's reference-set declarations constrain its scope.
    scope_pool = set(eligible)
    for domain in arguments:
        if domain.parameter.kind=='module_refs':
            scope_pool.intersection_update(domain.options)
    bindings = []
    for initiator in owners:
        if initiator not in scope_pool:
            continue
        scope = frozenset((initiator,)) if action.scope=='module' else frozenset(scope_pool)
        if any(domain.parameter.kind=='module_refs' and domain.parameter.size is not None and
               len(scope & domain.options)<domain.parameter.size for domain in arguments):
            continue
        bindings.append(StructuralBinding(initiator,(),scope,frozenset((initiator,)),arguments,'available'))
    return frozenset(bindings)


def enumerate_candidates(
    graph: StateGraph,
    spec: RobotSpec,
    schema: Mapping[str, object],
) -> CandidateTables:
    """Enumerate current choices without labels, context, future or executor state.

    Tuple routing order follows current collections; sets/bindings have invariant
    semantics under reordering and instance renaming. Pair scope is inter-module.
    Unknown unoccupied endpoints stay possible with explicit unknown admission;
    rollout must still apply its independently computed causal feasibility mask.
    """
    if not isinstance(graph,StateGraph):
        raise ValueError('graph: expected current StateGraph')
    validate_robot_spec(spec)
    validate_feature_schema(schema)
    owners,ports,port_specs,resources,resource_specs = _current_catalog(graph,spec)
    connected,occupied = _physical_connections(graph,ports)
    resource_rows = _resource_candidates(owners,ports,resources,resource_specs,spec)
    rows = []
    rules = schema['candidate_policy']['actions']
    for action in spec.actions:
        if not action.enabled or action.action_id not in rules:
            continue
        action = _canonical_action(action)
        rule = rules[action.action_id]
        if action.kind!=rule['kind']:
            raise ValueError('structural action kind disagrees with candidate policy')
        if rule['endpoints']=='none':
            bindings = _skill_bindings(action,owners,ports,port_specs,resource_specs)
            if bindings:
                rows.append(StructuralCandidate(action,frozenset(),bindings))
            continue
        if action.scope!='connector_pair':
            raise ValueError('structural pair requires connector_pair scope')
        existing = rule['endpoints']=='current_connection_pair'
        domains = _arguments(action,owners,ports)
        if domains is None:
            continue
        pairs = connected if existing else combinations((ref for ref in ports if ref not in occupied),2)
        for pair in pairs:
            if pair[0].module==pair[1].module:
                continue
            bindings = _pair_bindings(action,pair,owners,ports,port_specs,existing,domains)
            if bindings:
                rows.append(StructuralCandidate(action,frozenset(pair),bindings))
    return CandidateTables(resource_rows,tuple(rows))
