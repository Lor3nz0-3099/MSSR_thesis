"""Policy-confidence physical projection; no task/morphology/planner input.

The raw action remains the PPO action. Projection selects physical execution
only. Absolute raw-logit tie tolerance is 1e-6 (no relative tolerance).
"""
from dataclasses import dataclass
import math
from typing import Any

from drl.IL.bc_v1.action_set import ActionBundle
from drl.IL.bc_v1.bc_v1_contract import ActionChoice, ConnectorRef, ResourceRef

CONFIDENCE_TIE_ATOL = 1e-6


@dataclass(frozen=True)
class ProjectionCandidate:
    candidate_id: str
    candidate_index: int
    raw_presence_logit: float
    owner_ids: tuple[str, ...]
    choice: ActionChoice | None
    claims: tuple = ()
    decoded: Any = None
    invalid_reason: str | None = None


@dataclass(frozen=True)
class ProjectionDrop:
    candidate_id: str
    candidate_index: int
    confidence: float
    owner_ids: tuple[str, ...]
    code: str
    reason: str
    conflicting_candidate_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectionConflict:
    candidate_a: str
    candidate_b: str
    confidence_a: float
    confidence_b: float
    shared_claims: tuple
    confidence_tie: bool
    rule: str = 'reserved.intersection(claims)'


@dataclass(frozen=True)
class ActionProjection:
    sampled_candidates: tuple[ProjectionCandidate, ...]
    original_bundle: ActionBundle
    projected_bundle: ActionBundle
    accepted_candidate_ids: tuple[str, ...]
    dropped: tuple[ProjectionDrop, ...]
    conflicts: tuple[ProjectionConflict, ...]
    raw_sampled_action: Any = None
    tie_atol: float = CONFIDENCE_TIE_ATOL

    @property
    def accepted_candidates(self):
        lookup = {c.candidate_id:c for c in self.sampled_candidates}
        return tuple(lookup[cid] for cid in self.accepted_candidate_ids)

    @property
    def conflict_owner_ids(self):
        return tuple(sorted({o for d in self.dropped if d.code=='RESOURCE_BUSY' for o in d.owner_ids}))

    @property
    def invalid_owner_ids(self):
        return tuple(sorted({o for d in self.dropped if d.code=='INVALID_GOAL' for o in d.owner_ids}))


def project_executable_action_set(candidates, *, availability=None, feasibility=None,
                                  raw_sampled_action=None, tie_atol=CONFIDENCE_TIE_ATOL):
    """Greedy descending policy confidence with whole tied-component rejection.

    ``feasibility(candidate)`` is a read-only current physical admission check;
    it returns None or (RESOURCE_BUSY/INVALID_GOAL, exact reason). It must not
    dispatch actions. Unexpected infrastructure exceptions propagate.

    First remove individual failures. At each iteration remove members blocked
    by accepted claims, then process the highest score. If it belongs to a
    connected component of incompatible equal-confidence candidates, drop that
    entire tied component; no ID chooses a winner. Otherwise accept it. Lower
    unblocked candidates remain eligible after ties are removed. Stable IDs
    order independent equal-score actions and logs only.
    """
    candidates = tuple(candidates)
    if not math.isfinite(tie_atol) or tie_atol < 0:
        raise ValueError('Invalid confidence tie tolerance')
    if len({c.candidate_id for c in candidates}) != len(candidates):
        raise ValueError('Duplicate projection candidate ID')
    if any(not math.isfinite(float(c.raw_presence_logit)) for c in candidates):
        raise ValueError('Nonfinite projection confidence')
    ordered = sorted(candidates,key=lambda c:(-float(c.raw_presence_logit),c.candidate_id))
    dropped = []
    def drop(c,code,reason,peers=()):
        if code not in ('RESOURCE_BUSY','INVALID_GOAL'):
            raise RuntimeError(f'Unexpected infrastructure rejection: {code}: {reason}')
        dropped.append(ProjectionDrop(c.candidate_id,c.candidate_index,float(c.raw_presence_logit),
            c.owner_ids,code,reason,tuple(sorted(peers))))
    eligible = []
    for c in ordered:
        if c.invalid_reason is not None:
            drop(c,'INVALID_GOAL',c.invalid_reason)
            continue
        if (c.choice is None or not isinstance(c.claims,tuple) or not c.claims
                or len(set(c.claims)) != len(c.claims)
                or any(not isinstance(ref,(ResourceRef,ConnectorRef)) for ref in c.claims)
                or c.choice.resource is not None and c.choice.resource not in c.claims):
            drop(c,'INVALID_GOAL','Invalid physical claims')
            continue
        if availability is not None and not availability.free(c.claims):
            drop(c,'RESOURCE_BUSY','Physical resource unavailable')
            continue
        failure = feasibility(c) if feasibility is not None else None
        if failure is not None:
            drop(c,*failure)
            continue
        eligible.append(c)
    conflicts = []
    adjacency = {c.candidate_id:set() for c in eligible}
    ties = {c.candidate_id:set() for c in eligible}
    for i,a in enumerate(eligible):
        for b in eligible[i+1:]:
            shared = set(a.claims).intersection(b.claims)
            if not shared:
                continue
            tie = abs(float(a.raw_presence_logit)-float(b.raw_presence_logit)) <= tie_atol
            conflicts.append(ProjectionConflict(a.candidate_id,b.candidate_id,
                float(a.raw_presence_logit),float(b.raw_presence_logit),tuple(sorted(shared,key=repr)),tie))
            adjacency[a.candidate_id].add(b.candidate_id)
            adjacency[b.candidate_id].add(a.candidate_id)
            if tie:
                ties[a.candidate_id].add(b.candidate_id)
                ties[b.candidate_id].add(a.candidate_id)
    accepted = []
    remaining = eligible
    while remaining:
        accepted_ids = {c.candidate_id for c in accepted}
        unblocked = []
        for c in remaining:
            blockers = adjacency[c.candidate_id] & accepted_ids
            if blockers:
                drop(c,'RESOURCE_BUSY','Conflicting physical resource claims',blockers)
            else:
                unblocked.append(c)
        if not unblocked:
            break
        current = unblocked[0]
        available_ids = {c.candidate_id for c in unblocked}
        component = {current.candidate_id}
        pending = [current.candidate_id]
        while pending:
            node = pending.pop()
            for peer in sorted(ties[node] & available_ids - component):
                component.add(peer)
                pending.append(peer)
        if len(component)>1:
            for c in unblocked:
                if c.candidate_id in component:
                    drop(c,'RESOURCE_BUSY','Conflicting confidence tie',ties[c.candidate_id]&component)
            remaining = [c for c in unblocked if c.candidate_id not in component]
        else:
            accepted.append(current)
            remaining = unblocked[1:]
    return ActionProjection(candidates,ActionBundle(tuple(c.choice for c in candidates if c.choice is not None)),
        ActionBundle(tuple(c.choice for c in accepted)),tuple(c.candidate_id for c in accepted),
        tuple(dropped),tuple(conflicts),raw_sampled_action,tie_atol)
