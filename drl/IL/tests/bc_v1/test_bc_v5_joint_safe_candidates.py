"""BC-v5 causal candidate gating without using future state or labels."""
from dataclasses import replace

import pytest

from drl.IL.bc_v1.bc_v1_contract import ModuleRef
from drl.IL.bc_v1.bc_v5_joint_safe_candidates import (
    enumerate_joint_safe_candidates, live_align_is_physically_safe,
)
from drl.IL.bc_v1.causal_topology import CausalBoundary, ConfirmedDisconnect
from drl.IL.tests.bc_v1.test_candidates import attach, catalog, graph, schema, structural


def _case():
    g = attach(graph(3, 2), 0, 2)  # unit-0 / unit-1 joined; unit-2 isolated
    return g, g.connectors[0].ref, g.connectors[2].ref


def _events(a, b, completed=7, domain='runA'):
    return (ConfirmedDisconnect(a, b, 'physical-completion-001', domain, completed),)


def _initiators(result):
    return {binding.initiator.module_id
            for cand in structural(result.candidates, 'align_faces')
            for binding in cand.bindings}


def test_stale_joint_without_prior_physical_proof_blocks_attached_initiator():
    g, _, _ = _case()
    result = enumerate_joint_safe_candidates(g, catalog(2), schema())
    assert _initiators(result) == {'unit-2'}
    assert not result.applied_disconnects
    assert len(structural(result.candidates, 'undock')) == 1
    assert len(structural(result.candidates, 'gravity_settle')) == 1


def test_only_prior_verified_physical_completion_releases_stale_joint():
    g, a, b = _case()
    result = enumerate_joint_safe_candidates(
        g, catalog(2), schema(),
        confirmed_disconnects=_events(a, b),
        decision_boundary=CausalBoundary('runA', 9),
    )
    assert _initiators(result) == {'unit-0', 'unit-1', 'unit-2'}
    assert not structural(result.candidates, 'undock')
    assert not structural(result.candidates, 'gravity_settle')
    assert len(result.applied_disconnects) == 1
    # Never mutate the original physical graph / G2 routing.
    assert len(g.connections) == 1
    assert g.connectors[0].status == 'connected'


def test_same_bucket_or_future_completion_never_enables_alignment():
    g, a, b = _case()
    for completed in (9, 10):
        with pytest.raises(ValueError, match='strictly before boundary'):
            enumerate_joint_safe_candidates(g, catalog(2), schema(),
                confirmed_disconnects=_events(a, b, completed=completed),
                decision_boundary=CausalBoundary('runA', 9))


def test_undock_request_or_executor_override_is_not_physical_completion():
    g, a, b = _case()
    object.__setattr__(g, 'executor_face_available', True)
    object.__setattr__(g, 'prior_undock_commanded', True)
    assert _initiators(enumerate_joint_safe_candidates(g, catalog(2), schema())) == {'unit-2'}
    with pytest.raises(ValueError, match='decision boundary'):
        enumerate_joint_safe_candidates(g, catalog(2), schema(),
            confirmed_disconnects=_events(a,b))


def test_wrong_causal_domain_rejected():
    g, a, b = _case()
    with pytest.raises(ValueError, match='strictly before boundary'):
        enumerate_joint_safe_candidates(g, catalog(2), schema(),
            confirmed_disconnects=_events(a,b,domain='other-run'),
            decision_boundary=CausalBoundary('runA', 9))


def test_live_execution_requires_fresh_confirmed_physical_snapshot():
    g, a, b = _case()
    attached = ModuleRef('unit-0')
    detached = ModuleRef('unit-2')
    assert not live_align_is_physically_safe(attached, g, physical_snapshot_confirmed=True)
    assert not live_align_is_physically_safe(detached, g, physical_snapshot_confirmed=False)
    assert live_align_is_physically_safe(detached, g, physical_snapshot_confirmed=True)
    assert not live_align_is_physically_safe(ModuleRef('missing'), g, physical_snapshot_confirmed=True)


def test_causal_reconciliation_does_not_skip_live_joint_guard():
    g, a, b = _case()
    result = enumerate_joint_safe_candidates(g, catalog(2), schema(),
        confirmed_disconnects=_events(a, b),
        decision_boundary=CausalBoundary('runA', 9))
    # A trainer may route through causally reconciled topology while the live
    # controller still has to query the REAL joint state before execution.
    assert ModuleRef('unit-0').module_id in _initiators(result)
    assert not live_align_is_physically_safe(ModuleRef('unit-0'), g,
        physical_snapshot_confirmed=True)
