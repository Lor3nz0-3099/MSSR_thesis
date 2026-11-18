"""S1 regression tests. Fully fixture-based, no RAW, dataset, or simulator."""
from dataclasses import replace

import pytest
import torch

from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.bc_v1.bc_v5_membership import NEGATIVE, POSITIVE, UNKNOWN
from drl.IL.bc_v1.bc_v5_supervision import (
    certified_negatives_from_route, collate_action_set_v5_samples,
    membership_with_evidence,
)
from drl.IL.tests.bc_v1.g2_fixtures import fixture, tensorize


def _example():
    sample, kwargs = fixture()
    return tensorize(sample, kwargs)


def test_selected_mode_certifies_only_other_modes_of_same_resource():
    s = _example()
    route = s['routing']
    membership, evidence = membership_with_evidence(route)
    positive_ref = route['physical_supervision'].resource_targets[0].resource
    positive_mode = route['physical_supervision'].resource_targets[0].mode_id
    assert any(p.evidence_kind == 'exclusive_resource_mode' for p in evidence)
    for idx, row in enumerate(route['resource_candidates']):
        if row.resource_ref == positive_ref:
            assert membership[idx].item() == (POSITIVE if row.mode.mode_id == positive_mode else NEGATIVE)
        else:
            assert membership[idx].item() == UNKNOWN
    assert (membership[len(route['resource_candidates']):] == UNKNOWN).all()


def test_empty_demonstration_produces_no_default_negatives():
    sample, kw = fixture()
    empty = replace(sample, supervision=c.SupervisionBundle())
    status, proof = membership_with_evidence(tensorize(empty, kw)['routing'])
    assert (status == UNKNOWN).all()
    assert proof == ()


def test_complete_source_certifies_specific_absent_resource_only():
    s = _example()
    route = s['routing']
    ref = route['physical_supervision'].resource_targets[0].resource
    absent = next(i for i, x in enumerate(route['resource_refs']) if x != ref)
    copied = dict(route, resource_label_authority={
        'policy': 'smores.complete_nav2_snapshot.v1', 'complete': True,
        'evidence': [{'path':'raw/episode.jsonl', 'row':12, 'sha256':'a'*64}],
        'certified_absent_resources':[absent]})
    status, proof = membership_with_evidence(copied)
    for i, row in enumerate(route['resource_candidates']):
        if row.resource_ref == route['resource_refs'][absent]:
            assert status[i].item() == NEGATIVE
    assert any(p.evidence_kind == 'explicit_no_dispatch' for p in proof)
    assert (status[len(route['resource_candidates']):] == UNKNOWN).all()


def test_missing_evidence_or_incomplete_authority_cannot_certify_absence():
    s = _example(); route = s['routing']
    ref = route['physical_supervision'].resource_targets[0].resource
    absent = next(i for i, x in enumerate(route['resource_refs']) if x != ref)
    authority = {'policy':'smores.complete_nav2_snapshot.v1', 'complete':True,
                 'evidence': [], 'certified_absent_resources':[absent]}
    with pytest.raises(ValueError, match='evidence'):
        membership_with_evidence(dict(route,resource_label_authority=authority))
    with pytest.raises(ValueError, match='Incomplete'):
        membership_with_evidence(dict(route,resource_label_authority={**authority, 'complete':False}))
    with pytest.raises(ValueError, match='collides'):
        membership_with_evidence(dict(route,resource_label_authority={**authority,
            'evidence':[{'path':'raw/episode.jsonl','row':0,'sha256':'b'*64}],
            'certified_absent_resources':[route['resource_refs'].index(ref)]}))


def test_collated_membership_uses_resource_then_structural_candidate_order():
    a = _example(); b = _example()
    batch = collate_action_set_v5_samples([a, b])
    result = batch['supervision']['action_set']['membership']
    r, p = len(a['routing']['resource_candidates']), len(a['routing']['structural_bindings'])
    sr, _ = membership_with_evidence(a['routing'])
    torch.testing.assert_close(result, torch.cat([sr[:r], sr[:r], sr[r:], sr[r:]]))
    torch.testing.assert_close(result == POSITIVE, batch['supervision']['action_set']['presence'])
    assert len(result) == 2*(r+p)


def test_v5_loss_ignores_unknown_logits_and_preserves_positive_codecs():
    from drl.IL.bc_v1.bc_v5_action_set_loss import compute_action_set_v5_loss
    from drl.IL.bc_v1.action_set_loss import compute_action_set_loss
    from drl.IL.tests.bc_v1.test_action_set_loss import parameter_example
    prediction, batch = parameter_example()
    labels = batch['supervision']['action_set']['presence']
    # Seed exactly one certified negative and leave all other absences unknown.
    status = torch.full(labels.shape, UNKNOWN, dtype=torch.int8)
    status[labels] = POSITIVE
    chosen_negative = next(i for i, is_positive in enumerate(labels) if not is_positive)
    status[chosen_negative] = NEGATIVE
    batch['supervision']['action_set']['membership'] = status
    v5, v5_stats = compute_action_set_v5_loss(prediction, batch)
    v4, v4_stats = compute_action_set_loss(prediction, batch)
    for key in v4_stats:
        if key != 'presence':
            torch.testing.assert_close(v5_stats[key]['mean'], v4_stats[key]['mean'])
    v5.backward()
    grad = prediction['presence_logits'].grad
    assert grad[labels].lt(0).all()
    assert grad[chosen_negative] > 0
    assert (grad[status == UNKNOWN] == 0).all()
    assert torch.isfinite(v5)
    # With zero logits, 1 positive and 1 negative the presence loss is 2 ln 2.
    torch.testing.assert_close(v5_stats['presence']['mean'],
        2 * torch.log(torch.tensor(2.)))


def test_v5_loss_rejects_v4_only_batch_without_membership():
    from drl.IL.bc_v1.bc_v5_action_set_loss import compute_action_set_v5_loss
    from drl.IL.tests.bc_v1.test_action_set_loss import parameter_example
    prediction, batch = parameter_example()
    with pytest.raises(ValueError, match='requires explicit three-state'):
        compute_action_set_v5_loss(prediction, batch)


def test_full_v5_collator_and_policy_have_finite_backward():
    from drl.IL.bc_v1.bc_v5_action_set_loss import compute_action_set_v5_loss
    from drl.IL.tests.bc_v1.test_action_set_head import policy_for
    sample, kw = fixture(2)
    batch = collate_action_set_v5_samples([tensorize(sample, kw)])
    policy = policy_for(kw)
    pred = policy(batch['inputs'])
    loss, stats = compute_action_set_v5_loss(pred, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.head.presence_scorer[-1].weight.grad.abs().sum() > 0
    assert stats['resource_payload']['support'] >= 1
