"""Generic exact disconnect: no measured channel reconstruction."""
from dataclasses import replace, FrozenInstanceError
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest
from drl.IL.bc_v1 import bc_v1_contract as c


ROOT = Path(__file__).resolve().parents[4]


def api():
    name = 'drl.IL.bc_v1.causal_topology'
    assert importlib.util.find_spec(name) is not None, 'Missing Task 2 causal topology core'
    return importlib.import_module(name)


def graph():
    refs = tuple(c.ModuleRef(n) for n in ('alpha', 'beta', 'gamma', 'delta'))
    ports = tuple(c.ConnectorRef(r, 'socket-X') for r in refs)
    features = (c.NumericFeature('position_body', (1., 2., 3.)),)
    modules = tuple(c.ModuleState(r, 'generic-kit', (i, 2., 3.), (0., 0., 0., 1.),
                    (1., 2., 3.), (4., 5., 6.), True) for i, r in enumerate(refs))
    connectors = tuple(c.ConnectorState(p, 'fixed_port', ports[i ^ 1], 'connected', features)
                       for i, p in enumerate(ports))
    resources = (c.ResourceState(c.ResourceRef(refs[0], 'axis'),
                    (c.CoordinateState('q', .7, .3),)),)
    edges = tuple(c.ConnectionState(ports[i], ports[i+1], 'current_connection',
                    features=(c.NumericFeature('relative', (.1, .2, .3)),)) for i in (0, 2))
    return c.StateGraph(modules, connectors, resources, edges)


def event(g=None, *, reverse=False, order=5, domain='ordered-events', evidence='proof-1'):
    g = graph() if g is None else g
    pair = (g.connectors[0].ref, g.connectors[1].ref)
    if reverse: pair = pair[::-1]
    return api().ConfirmedDisconnect(*pair, evidence, domain, order)


def reconcile(g=None, events=None, boundary=None):
    m = api(); g = graph() if g is None else g
    return m.reconcile_topology(g, (event(g),) if events is None else events,
                               m.CausalBoundary('ordered-events', 10) if boundary is None else boundary)


def test_exact_pair_removal_and_occupancy_coherent():
    g = graph(); result = reconcile(g)
    assert result.graph.connections == g.connections[1:]
    assert all(x.connected_to is None and x.status == 'available' for x in result.graph.connectors[:2])
    assert result.graph.connectors[2:] == g.connectors[2:]
    assert result.applied == (event(g),) and result.already_absent == ()


def test_original_immutable_and_non_topological_channels_unchanged():
    g = graph(); before = replace(g); result = reconcile(g)
    assert g == before and len(g.connections) == 2
    assert result.graph.modules == g.modules and result.graph.resources == g.resources
    assert [x.features for x in result.graph.connectors] == [x.features for x in g.connectors]
    assert result.graph.connections[0] == g.connections[1]


def test_idempotence_duplicate_and_already_absent_noop():
    g = graph(); first = reconcile(g, (event(g), event(g, reverse=True)))
    assert len(first.applied) == 1 and len(first.already_absent) == 1
    again = reconcile(first.graph, (event(g),))
    assert again.graph == first.graph and again.applied == ()
    assert again.already_absent == (event(g),)


def test_endpoint_reversal_has_same_physical_effect():
    assert reconcile().graph == reconcile(events=(event(reverse=True),)).graph


@pytest.mark.parametrize('pointer_only', [False, True])
def test_wrong_peer_conflict_does_not_remove_any_edge(pointer_only):
    m = api(); g = graph()
    wrong = m.ConfirmedDisconnect(g.connectors[0].ref, g.connectors[2].ref, 'wrong', 'ordered-events', 5)
    if pointer_only: g = replace(g, connections=())
    before = replace(g)
    with pytest.raises(ValueError, match='[Cc]onflict|peer|coheren'):
        reconcile(g, (wrong,))
    assert g == before


@pytest.mark.parametrize('order,domain', [(10, 'ordered-events'), (11, 'ordered-events'), (5, 'another-domain')])
def test_future_nonbefore_and_cross_domain_events_rejected(order, domain):
    with pytest.raises(ValueError): reconcile(events=(event(order=order, domain=domain),))


def test_missing_endpoints_and_discordant_occupancy_fail():
    m = api(); g = graph()
    absent = c.ConnectorRef(c.ModuleRef('absent'), 'unknown')
    with pytest.raises(ValueError):
        reconcile(g, (m.ConfirmedDisconnect(g.connectors[0].ref, absent, 'x', 'ordered-events', 5),))
    bad = replace(g, connectors=(replace(g.connectors[0], connected_to=g.connectors[2].ref),)+g.connectors[1:])
    with pytest.raises(ValueError): reconcile(bad)


def test_edge_authority_allows_unknown_occupancy_then_frees_exact_pair():
    g = graph()
    g = replace(g, connectors=tuple(replace(x, connected_to=None, status='unknown') for x in g.connectors))
    result = reconcile(g)
    assert len(result.graph.connections) == 1
    assert all(x.status == 'available' for x in result.graph.connectors[:2])


def test_no_label_stage_fsm_or_future_api_and_frozen_types():
    m = api()
    assert tuple(inspect.signature(m.reconcile_topology).parameters) == ('graph', 'events', 'boundary')
    for name in ('label', 'stage', 'fsm', 'future_graph', 'action_t'):
        with pytest.raises(TypeError): m.reconcile_topology(graph(), (), m.CausalBoundary('x', 1), **{name: None})
    with pytest.raises(FrozenInstanceError): event().completed_order = 99


@pytest.mark.parametrize('domain,order', [('', 1), ('x', True), ('x', -1), ('x', 1.0)])
def test_invalid_order_contract_rejected(domain, order):
    with pytest.raises(ValueError): api().CausalBoundary(domain, order)


def test_causal_fingerprint_deterministic_and_semantically_sensitive():
    m = api(); policy = json.loads((ROOT/'configs/bc_v1/causal_topology.json').read_text())
    before = m.causal_topology_fingerprint(policy)
    assert len(before) == 64
    assert before == m.causal_topology_fingerprint(dict(reversed(list(policy.items()))))
    for key in ('schema_version', 'algorithm', 'ordering_contract', 'mixed_bucket_policy'):
        assert before != m.causal_topology_fingerprint(dict(policy, **{key: 'changed'}))


def test_generic_import_isolation():
    api()
    code = '''
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, name, *args):
  if name.startswith(('numpy','torch','rclpy','isaac','smores','freebot')): raise RuntimeError(name)
sys.meta_path.insert(0, Block())
import drl.IL.bc_v1.causal_topology
'''
    result = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
