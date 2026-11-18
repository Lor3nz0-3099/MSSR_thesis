"""BC-v5 S1: conservatively derive candidate membership from authenticated G2.

This module is opt-in and does not alter BC-v4's positive-vs-absent labels.
Only a physically selected resource mode and an independently certified
complete NAV2 snapshot can establish *negative* candidate membership.
Silence, structural omission, invalid command masks, and unknown authority do
not establish a negative. No underlying G2 or Canonical data is mutated.
"""
from collections import defaultdict
import re

import torch

from .action_set_targets import collate_action_set_samples
from .bc_v5_membership import (
    CertifiedNegative, candidate_statuses, POSITIVE, NEGATIVE, UNKNOWN,
)

_AUTHORITY = 'smores.complete_nav2_snapshot.v1'
_SHA256 = re.compile(r'^[0-9a-f]{64}$')


def certified_negatives_from_route(route):
    """Certify only mutually exclusive resource modes and proven no-dispatch.

    A resource has at most one physical mode in one decision: when present,
    alternative modes for that *same resource* are impossible. This does not
    imply anything about other resources, or structural candidates.
    """
    audit = route['audit']
    provenance = f'{audit.episode_id}:{audit.transition_index}'
    resources = tuple(route['resource_refs'])
    resource_set = set(resources)
    modes = defaultdict(set)
    for row in route['physical_supervision'].resource_targets:
        if row.supervised is not True or row.resource not in resource_set:
            raise ValueError('Invalid physical resource target for BC-v5')
        modes[row.resource].add(row.mode_id)
    if any(len(found) != 1 for found in modes.values()):
        raise ValueError('Multiple modes assigned to one physical resource')

    # Exactly one candidate for each resource+mode (the positive index helper
    # independently enforces this for all positive commands).
    by_resource = defaultdict(set)
    for idx, candidate in enumerate(route['resource_candidates']):
        key = (candidate.resource_ref, candidate.mode.mode_id)
        if key in by_resource[candidate.resource_ref]:
            raise ValueError('Duplicate physical resource candidate')
        by_resource[candidate.resource_ref].add(key)
    certified = []
    for idx, candidate in enumerate(route['resource_candidates']):
        selected = modes.get(candidate.resource_ref)
        if selected and candidate.mode.mode_id not in selected:
            certified.append(CertifiedNegative(idx,
                f'{provenance}:exclusive:{candidate.resource_ref.resource_id}:{next(iter(selected))}',
                'exclusive_resource_mode'))

    authority = route.get('resource_label_authority')
    if authority is None:
        return tuple(certified)
    if not isinstance(authority, dict) or authority.get('policy') != _AUTHORITY \
            or type(authority.get('complete')) is not bool:
        raise ValueError('Unsupported BC-v5 negative authority')
    absent = authority.get('certified_absent_resources')
    if not isinstance(absent, list) or any(type(r) is not int or not 0 <= r < len(resources) for r in absent) \
            or len(set(absent)) != len(absent):
        raise ValueError('Invalid certified-absence resource indices')
    if not authority['complete']:
        if absent:
            raise ValueError('Incomplete authority claims certified negatives')
        return tuple(certified)
    evidence = authority.get('evidence')
    if not isinstance(evidence, list) or not evidence or any(
            not isinstance(entry, dict) or not isinstance(entry.get('path'), str)
            or not entry['path'] or type(entry.get('row')) is not int
            or entry['row'] < 0 or not isinstance(entry.get('sha256'), str)
            or not _SHA256.fullmatch(entry['sha256']) for entry in evidence):
        raise ValueError('Missing certified complete snapshot evidence')
    if any(resources[r] in modes for r in absent):
        raise ValueError('Certified absence collides with physical resource command')
    # A source certificate is bound to the full route identity and audit row;
    # non-member resources are never inferred absent.
    for r in absent:
        for idx, candidate in enumerate(route['resource_candidates']):
            if candidate.resource_ref == resources[r]:
                certified.append(CertifiedNegative(idx,
                    f'{provenance}:complete_nav2:{r}:' + evidence[0]['sha256'],
                    'explicit_no_dispatch'))
    return tuple(certified)


def membership_with_evidence(route):
    proofs = certified_negatives_from_route(route)
    status = candidate_statuses(route, certified_negatives=proofs)
    return status, proofs


def collate_action_set_v5_samples(samples):
    """Produce BC-v5 3-state status without touching BC-v4's target ABI.

    All resource statuses precede all structural statuses in the physical
    candidate ordering, just like ActionSetPolicy's output logits.
    """
    if not samples:
        raise ValueError('Empty BC-v5 batch')
    batch = collate_action_set_samples(samples)
    resource_parts, structural_parts = [], []
    for sample in samples:
        route = sample['routing']
        status, _ = membership_with_evidence(route)
        n = len(route['resource_candidates'])
        resource_parts.append(status[:n])
        structural_parts.append(status[n:])
    membership = torch.cat(resource_parts + structural_parts)
    original = batch['supervision']['action_set']['presence']
    if membership.shape != original.shape or not torch.equal(membership == POSITIVE, original):
        raise ValueError('BC-v5 membership and physical positives disagree')
    batch['supervision']['action_set']['membership'] = membership
    return batch


def membership_summary(statuses):
    """Diagnostic only; no labels generated or inferred from missing values."""
    result = {'positive': 0, 'certified_negative': 0, 'unknown': 0}
    for status in statuses:
        result['positive'] += int((status == POSITIVE).sum())
        result['certified_negative'] += int((status == NEGATIVE).sum())
        result['unknown'] += int((status == UNKNOWN).sum())
    return result
