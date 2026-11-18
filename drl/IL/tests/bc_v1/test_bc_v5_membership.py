"""BC-v5 S1: missing actions remain UNKNOWN unless explicitly certified."""
from dataclasses import replace

import pytest
import torch

from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.bc_v1.bc_v5_membership import (
    CertifiedNegative, candidate_statuses, masked_presence_loss,
    check_negative_coverage, NEGATIVE, POSITIVE, UNKNOWN,
)
from drl.IL.tests.bc_v1.g2_fixtures import fixture, tensorize


def test_unobserved_candidates_are_unknown_not_negative():
    sample, kwargs = fixture()
    route = tensorize(sample, kwargs)['routing']
    states = candidate_statuses(route)
    assert states.dtype == torch.int8
    assert int((states == POSITIVE).sum()) == 1
    assert int((states == NEGATIVE).sum()) == 0
    assert int((states == UNKNOWN).sum()) == len(states)-1
    with pytest.raises(ValueError, match='cannot train presence'):
        check_negative_coverage([states])


def test_empty_target_does_not_synthesize_idle_negatives():
    sample, kwargs = fixture()
    empty = replace(sample, supervision=c.SupervisionBundle())
    states = candidate_statuses(tensorize(empty, kwargs)['routing'])
    assert (states == UNKNOWN).all()


def test_certified_negative_with_provenance_does_not_change_unknown():
    sample, kwargs = fixture()
    route = tensorize(sample, kwargs)['routing']
    baseline = candidate_statuses(route)
    free_idx = next(i for i, x in enumerate(baseline) if x == UNKNOWN)
    evidence = CertifiedNegative(free_idx, 'raw:episode:frame:decision', 'explicit_no_dispatch')
    current = candidate_statuses(route, certified_negatives=(evidence,))
    assert current[free_idx] == NEGATIVE
    assert int((current == UNKNOWN).sum()) == int((baseline == UNKNOWN).sum())-1
    assert check_negative_coverage([current])['certified_negative'] == 1
    with pytest.raises(ValueError, match='collision'):
        candidate_statuses(route, certified_negatives=(
            CertifiedNegative(int((baseline == POSITIVE).nonzero()[0]), 'raw', 'explicit_no_dispatch'),))
    with pytest.raises(ValueError, match='Duplicate'):
        candidate_statuses(route, certified_negatives=(evidence, evidence))
    with pytest.raises(ValueError, match='out-of-range'):
        candidate_statuses(route, certified_negatives=(
            CertifiedNegative(len(baseline), 'raw', 'explicit_no_dispatch'),))


def test_unknown_logits_have_exactly_zero_gradient_and_no_penalty():
    logits = torch.zeros(4, requires_grad=True)
    labels = torch.tensor([POSITIVE, NEGATIVE, UNKNOWN, UNKNOWN], dtype=torch.int8)
    value = masked_presence_loss(logits, labels, torch.tensor([0, 0, 0, 0]), 1)
    torch.testing.assert_close(value, torch.tensor(2*torch.log(torch.tensor(2.)).item()))
    value.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([-.5, .5, 0., 0.]))


def test_invalid_tri_state_rejected():
    logits = torch.zeros(2)
    graph = torch.zeros(2, dtype=torch.int64)
    with pytest.raises(ValueError):
        masked_presence_loss(logits, torch.tensor([1, 2],dtype=torch.int8), graph, 1)
    with pytest.raises(ValueError):
        masked_presence_loss(logits, torch.tensor([True,False]), graph, 1)
