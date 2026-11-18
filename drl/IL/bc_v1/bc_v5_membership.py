"""BC-v5 tri-state ActionSet supervision (S1), intentionally NOT wired to BC-v4.

A candidate absent from the positive demonstration is UNKNOWN, not a known
negative.  Explicit negative labels must be supplied by independently audited
source evidence before this loss can be used in production training. In
particular, a canonical command mask of False is NOT negative evidence.

No inputs, checkpoints, dataset snapshots, or model architecture are changed.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .action_set_targets import positive_indices

UNKNOWN = -1
NEGATIVE = 0
POSITIVE = 1


@dataclass(frozen=True)
class CertifiedNegative:
    """Traceable negative evidence; the caller must verify source semantics."""
    candidate_index: int
    evidence_id: str
    evidence_kind: str

    def __post_init__(self):
        if type(self.candidate_index) is not int or self.candidate_index < 0:
            raise ValueError('Invalid negative candidate index')
        if not isinstance(self.evidence_id, str) or not self.evidence_id.strip():
            raise ValueError('Negative requires source evidence ID')
        if self.evidence_kind not in ('explicit_no_dispatch', 'exclusive_resource_mode'):
            raise ValueError('Unverified negative evidence kind')


def candidate_statuses(route, *, certified_negatives=()):
    """Return int8 statuses (-1/0/+1) for one physical ActionSet graph.

    No default known negatives: a missing action, failed command mask, or
    unrecorded actuator transition is not evidence of a negative.
    """
    rq = len(route['resource_candidates'])
    sq = len(route['structural_bindings'])
    status = torch.full((rq + sq,), UNKNOWN, dtype=torch.int8)
    positive = positive_indices(route)
    if positive:
        status[list(positive)] = POSITIVE
    seen = set()
    for proof in certified_negatives:
        if type(proof) is not CertifiedNegative:
            raise ValueError('Negative label requires a typed evidence record')
        idx = proof.candidate_index
        if idx >= len(status) or idx in seen:
            raise ValueError('Duplicate/out-of-range negative evidence')
        if int(status[idx]) == POSITIVE:
            raise ValueError('Positive/negative collision at candidate')
        seen.add(idx)
        status[idx] = NEGATIVE
    return status


def masked_presence_loss(logits, status, graph, num_graphs):
    """Balanced positive/known-negative loss; UNKNOWN candidates have zero gradient.

    This is the BC-v5-only objective. It cannot be used to train meaningful
    absence predictions until source-certified negatives have been constructed.
    """
    if (type(num_graphs) is not int or num_graphs < 0
            or not isinstance(logits, torch.Tensor) or logits.ndim != 1
            or not logits.is_floating_point()
            or not isinstance(status, torch.Tensor) or status.dtype != torch.int8
            or status.shape != logits.shape or status.requires_grad
            or not isinstance(graph, torch.Tensor) or graph.dtype != torch.int64
            or graph.shape != logits.shape or graph.requires_grad):
        raise ValueError('Invalid BC-v5 presence ABI')
    if not torch.isfinite(logits).all() or not torch.isin(status, torch.tensor([-1, 0, 1], device=status.device)).all():
        raise ValueError('Invalid BC-v5 presence status/logit')
    if ((graph < 0) | (graph >= num_graphs)).any():
        raise ValueError('Graph ownership outside [0,num_graphs)')
    zero = logits[:0].sum()
    if not num_graphs:
        return zero
    status = status.to(logits.device)
    graph = graph.to(logits.device)
    terms = []
    for value, sign in ((POSITIVE, -1), (NEGATIVE, 1)):
        selected = status == value
        owners = graph[selected]
        loss = F.softplus(sign * logits[selected])
        sums = logits.new_zeros(num_graphs).scatter_add(0, owners, loss)
        counts = logits.new_zeros(num_graphs).scatter_add(0, owners, torch.ones_like(loss))
        terms.append(sums / counts.clamp_min(1))
    return (terms[0] + terms[1]).mean() + zero


def check_negative_coverage(statuses):
    """Fail closed: positive-only evidence is insufficient to train inactivity."""
    if not statuses or any(not isinstance(x, torch.Tensor) or x.dtype != torch.int8 for x in statuses):
        raise ValueError('Missing BC-v5 membership status')
    positive = sum(int((s == POSITIVE).sum()) for s in statuses)
    negative = sum(int((s == NEGATIVE).sum()) for s in statuses)
    unknown = sum(int((s == UNKNOWN).sum()) for s in statuses)
    if not positive or not negative:
        raise ValueError('BC-v5 cannot train presence without positive AND source-certified negative evidence')
    return {'positive': positive, 'certified_negative': negative, 'unknown': unknown}
