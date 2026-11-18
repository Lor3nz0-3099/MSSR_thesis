"""BC-v5 candidate admission with *causally proven* disconnects.

The observation at a decision can contain a stale ``current_connection`` edge
for a joint whose physical removal completed before that decision. The
executor's desired/available face state alone is NOT a physical completion.

This opt-in entry point accepts only exact, ordered ConfirmedDisconnect events
from a trusted physical completion provider. It never uses future graph states,
current action labels, or a task/reconfiguration phase. If completion cannot be
proven, it preserves the attachment and forbids align_faces on its initiator.

BC-v4, existing G2 and the deployed decoder are intentionally unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

from .bc_v1_contract import ModuleRef, StateGraph, RobotSpec
from .bc_v5_topology_candidates import enumerate_candidates_v5, topology_index
from .causal_topology import (
    CausalBoundary, ConfirmedDisconnect, ReconciledTopologyResult, reconcile_topology,
)
from .candidates import CandidateTables


@dataclass(frozen=True)
class CausalCandidateResult:
    candidates: CandidateTables
    graph: StateGraph
    applied_disconnects: tuple[ConfirmedDisconnect, ...]
    already_absent_disconnects: tuple[ConfirmedDisconnect, ...]


def enumerate_joint_safe_candidates(
    graph: StateGraph,
    spec: RobotSpec,
    schema,
    *,
    confirmed_disconnects: tuple[ConfirmedDisconnect, ...] = (),
    decision_boundary: CausalBoundary | None = None,
) -> CausalCandidateResult:
    """Generate v5 choices after optional proven *pre-decision* joint removal.

    If no certificate is given, use the observed graph without any repair.
    The reconciliation function validates same-domain ordering and exact pairs.
    This function is suitable for an offline rebuilt-v5 candidate routing only
    when the evidence source and observation boundary are independently audited.
    """
    if not isinstance(graph, StateGraph):
        raise TypeError('Expected a current StateGraph')
    if not isinstance(confirmed_disconnects, tuple):
        raise TypeError('Confirmed disconnects must be a tuple')
    if confirmed_disconnects:
        if not isinstance(decision_boundary, CausalBoundary):
            raise ValueError('Physical disconnect certificates require a decision boundary')
        reconciled = reconcile_topology(graph, confirmed_disconnects, decision_boundary)
    else:
        if decision_boundary is not None and not isinstance(decision_boundary, CausalBoundary):
            raise TypeError('Invalid causal boundary')
        reconciled = ReconciledTopologyResult(graph)
    return CausalCandidateResult(
        candidates=enumerate_candidates_v5(reconciled.graph, spec, schema),
        graph=reconciled.graph,
        applied_disconnects=reconciled.applied,
        already_absent_disconnects=reconciled.already_absent,
    )


def live_align_is_physically_safe(
    initiator: ModuleRef,
    fresh_physical_graph: StateGraph,
    *,
    physical_snapshot_confirmed: bool,
) -> bool:
    """Fail-closed execution gate; never use a causally repaired graph here.

    The caller must obtain a fresh *physical joint* snapshot from the runtime.
    A prior undock request or executor-face availability must not set the
    ``physical_snapshot_confirmed`` flag. Unknown peers/connected connector
    statuses also block execution.
    """
    if type(physical_snapshot_confirmed) is not bool:
        raise TypeError('physical_snapshot_confirmed must be bool')
    if not physical_snapshot_confirmed:
        return False
    if not isinstance(fresh_physical_graph, StateGraph):
        raise TypeError('Expected a physical StateGraph')
    if not isinstance(initiator, ModuleRef):
        raise TypeError('Expected ModuleRef')
    if initiator not in {m.ref for m in fresh_physical_graph.modules}:
        return False
    ports = {p.ref: p for p in fresh_physical_graph.connectors}
    _, _, singletons, _ = topology_index(fresh_physical_graph, ports)
    return initiator in singletons
