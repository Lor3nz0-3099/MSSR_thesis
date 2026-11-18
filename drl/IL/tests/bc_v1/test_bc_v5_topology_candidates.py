"""Topology-only BC-v5 proposal: no phase leaks, no BC-v4 mutation."""
from dataclasses import replace

from drl.IL.bc_v1.bc_v5_topology_candidates import enumerate_candidates_v5
from drl.IL.bc_v1.candidates import enumerate_candidates
from drl.IL.tests.bc_v1.test_candidates import graph, catalog, schema, attach, structural, renamed, routing_semantics
from drl.IL.bc_v1 import bc_v1_contract as c


def v5(g, spec=None):
    return enumerate_candidates_v5(g, catalog(2) if spec is None else spec, schema())


def _binding_initiators(table, name):
    return {b.initiator.module_id for item in structural(table, name) for b in item.bindings}


def test_two_detached_modules_can_align_and_dock_but_not_undock_or_settle():
    g = graph(2, 2)
    result = v5(g)
    assert len(structural(result, 'align_faces')) == 4
    assert len(structural(result, 'dock')) == 4
    assert _binding_initiators(result, 'align_faces') == {'unit-0', 'unit-1'}
    assert not structural(result, 'undock')
    assert not structural(result, 'gravity_settle')
    assert len(result.resource_candidates) == 4


def test_connected_structure_proposes_only_undock_and_gravity():
    g = attach(graph(2, 2), 0, 2)
    result = v5(g)
    assert not structural(result, 'align_faces')
    assert not structural(result, 'dock')
    assert len(structural(result, 'undock')) == 1
    (settle,) = structural(result, 'gravity_settle')
    assert len(settle.bindings) == 2
    assert len(result.resource_candidates) == 4
    # Existing BC-v4 generator remains unchanged, not switched implicitly.
    assert enumerate_candidates(g, catalog(2), schema()) != result


def test_free_initiator_may_align_to_attached_peer_even_on_occupied_face():
    g = attach(graph(3, 2), 0, 2)  # unit-0 and unit-1 linked via socket-0
    result = v5(g)
    assert _binding_initiators(result, 'align_faces') == {'unit-2'}
    # unit-2 (free) may align with either face of unit-0 or unit-1:
    # even socket-0 (occupied) is an admissible alignment target.
    assert len(structural(result, 'align_faces')) == 8
    assert all(len(item.bindings) == 1 for item in structural(result, 'align_faces'))
    assert all(item.bindings.__iter__().__next__().initiator.module_id == 'unit-2'
               for item in structural(result, 'align_faces'))
    assert len(structural(result, 'dock')) == 4  # only free faces on 0 and 1
    assert _binding_initiators(result, 'gravity_settle') == {'unit-0', 'unit-1'}
    assert len(structural(result, 'undock')) == 1


def test_two_connected_subassemblies_cannot_align_by_dragging_either():
    g = graph(4, 2)
    left = (g.connectors[0].ref, g.connectors[2].ref)
    right = (g.connectors[4].ref, g.connectors[6].ref)
    connectors = tuple(replace(p, status='connected',
        connected_to=(left[1] if p.ref == left[0] else left[0]
                      if p.ref == left[1] else right[1]
                      if p.ref == right[0] else right[0]
                      if p.ref == right[1] else None))
        if p.ref in (*left, *right) else p for p in g.connectors)
    g = replace(g, connectors=connectors,
        connections=(c.ConnectionState(*left, 'current_connection'),
                     c.ConnectionState(*right, 'current_connection')))
    result = v5(g)
    assert not structural(result, 'align_faces')
    assert len(structural(result, 'dock')) == 4
    assert len(structural(result, 'undock')) == 2
    assert _binding_initiators(result, 'gravity_settle') == {'unit-0','unit-1','unit-2','unit-3'}


def test_uncertain_attachment_is_not_certified_singleton():
    g = graph(2,2)
    # A status indicating connection without known peer is not positive edge
    # evidence, but is enough to forbid certifying an isolated initiator.
    g = replace(g, connectors=(replace(g.connectors[0],status='connected'), *g.connectors[1:]))
    assert _binding_initiators(v5(g),'align_faces') == {'unit-1'}


def test_no_task_stage_or_future_data_consulted_and_renaming_invariance():
    g = attach(graph(3,2),0,2)
    result = v5(g)
    object.__setattr__(g, 'expert_action', {'align_faces': 'unit-0'})
    object.__setattr__(g, 'behavioral_stage', 'reconfiguring')
    object.__setattr__(g, 'target_morphology', 'anything')
    assert result == v5(g)
    changed,spec,modmap,portmap=renamed(g,catalog(2))
    assert routing_semantics(v5(changed,spec),modmap,portmap) == routing_semantics(result)
