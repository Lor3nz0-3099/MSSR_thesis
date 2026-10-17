"""Exact removal of causally admitted physical connections; stdlib only."""
from dataclasses import dataclass, replace
import hashlib
import json
from typing import Mapping

from .bc_v1_contract import ConnectorRef, StateGraph


VERSION = 'mssr.bc_v1.causal_topology.v1'
ALGORITHM = 'exact_pair_removal_and_occupancy.v1'
ORDERING_CONTRACT = 'strict_same_domain_event_order.v1'
MIXED_BUCKET_POLICY = 'previous_bucket_dispatch_completed_before_current_decision.v1'


def _order(domain, position):
    if type(domain) is not str or not domain.strip():
        raise ValueError('Missing causal order domain')
    if type(position) is not int or position < 0:
        raise ValueError('Invalid causal order position')


@dataclass(frozen=True)
class CausalBoundary:
    order_domain: str
    before_order: int

    def __post_init__(self):
        _order(self.order_domain, self.before_order)


@dataclass(frozen=True)
class ConfirmedDisconnect:
    endpoint_a: ConnectorRef
    endpoint_b: ConnectorRef
    evidence_id: str
    order_domain: str
    completed_order: int

    def __post_init__(self):
        _order(self.order_domain, self.completed_order)
        if type(self.endpoint_a) is not ConnectorRef or type(self.endpoint_b) is not ConnectorRef:
            raise ValueError('Expected generic connector endpoints')
        if self.endpoint_a == self.endpoint_b:
            raise ValueError('Disconnect endpoints must differ')
        if type(self.evidence_id) is not str or not self.evidence_id.strip():
            raise ValueError('Missing physical completion evidence identity')


@dataclass(frozen=True)
class ReconciledTopologyResult:
    graph: StateGraph
    applied: tuple[ConfirmedDisconnect, ...] = ()
    already_absent: tuple[ConfirmedDisconnect, ...] = ()


def reconcile_topology(graph: StateGraph, events: tuple[ConfirmedDisconnect, ...],
                       boundary: CausalBoundary) -> ReconciledTopologyResult:
    if type(graph) is not StateGraph or type(boundary) is not CausalBoundary:
        raise ValueError('Expected generic current graph and causal boundary')
    events = tuple(events)
    for event in events:
        if type(event) is not ConfirmedDisconnect:
            raise ValueError('Expected confirmed disconnect')
        if event.order_domain != boundary.order_domain or event.completed_order >= boundary.before_order:
            raise ValueError('Disconnect is not strictly before boundary in the same domain')
    current = graph
    applied, absent = [], []
    for event in events:
        a, b = event.endpoint_a, event.endpoint_b
        ports = {x.ref: x for x in current.connectors}
        if len(ports) != len(current.connectors) or a not in ports or b not in ports:
            raise ValueError('Missing or duplicate current connector endpoint')
        pair = frozenset((a, b))
        exact = [edge for edge in current.connections
                 if frozenset((edge.endpoint_a, edge.endpoint_b)) == pair]
        if len(exact) > 1:
            raise ValueError('Conflict: duplicate physical connection pair')
        for endpoint, peer in ((a, b), (b, a)):
            port = ports[endpoint]
            peers = {edge.endpoint_b if edge.endpoint_a == endpoint else edge.endpoint_a
                     for edge in current.connections
                     if endpoint in (edge.endpoint_a, edge.endpoint_b)}
            if port.connected_to is not None:
                peers.add(port.connected_to)
            if peers - {peer}:
                raise ValueError('Conflict: endpoint has a different current peer')
            if not exact and (peers or port.status == 'connected'):
                raise ValueError('Conflict: occupancy and connection pair are incoherent')
            if exact and port.status == 'available':
                raise ValueError('Conflict: available connector has a physical edge')
        if not exact:
            absent.append(event)
            continue
        current = replace(current,
            connections=tuple(edge for edge in current.connections if edge is not exact[0]),
            connectors=tuple(replace(port, connected_to=None, status='available')
                             if port.ref in pair else port for port in current.connectors))
        applied.append(event)
    return ReconciledTopologyResult(current, tuple(applied), tuple(absent))


def causal_topology_fingerprint(policy: Mapping) -> str:
    if not isinstance(policy, Mapping):
        raise ValueError('Expected causal topology policy mapping')
    payload = {'implementation_contract': VERSION, 'algorithm': ALGORITHM,
               'ordering_contract': ORDERING_CONTRACT, 'policy': dict(policy)}
    data = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()
