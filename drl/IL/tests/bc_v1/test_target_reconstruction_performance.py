"""Structural target lookup must avoid unused work and scalar tensor churn."""
from dataclasses import replace

import pytest
import torch

from drl.IL.bc_v1 import tensorization as t
from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.tests.bc_v1.g2_fixtures import fixture, tensorize, with_structural


def reconstruct(row, routing=None):
    route = row['routing'] if routing is None else routing
    indices = {kind: {ref: i for i, ref in enumerate(route[kind + '_refs'])}
               for kind in ('module', 'connector', 'resource')}
    return t._target_tensors(route['resource_targets'], route['structural_targets'],
                             indices, route, row['inputs'])


def test_resource_only_does_not_read_structural_argument_domains(monkeypatch):
    # Reintroducing an unconditional all-domain lookup must fail this test.
    sample, kwargs = fixture()
    row = tensorize(sample, kwargs)
    reads = []
    snapshots = []
    binding = row['inputs']['structural']['argument_binding']
    original_tolist = torch.Tensor.tolist

    def observed_tolist(self, *args, **kwargs):
        if self is binding:
            snapshots.append(True)
        return original_tolist(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'tolist', observed_tolist)

    class ObservedDomains:
        def __getitem__(self, index):
            reads.append(index)
            return row['routing']['argument_domains'][index]

    result = reconstruct(row, {**row['routing'], 'argument_domains': ObservedDomains()})
    assert reads == [], 'Resource-only reconstruction read unused structural argument domains'
    assert snapshots == [], 'Resource-only reconstruction copied unused argument bindings'
    assert result['resource']['payload'].tolist() == [2., 3.]
    assert result['structural']['candidate'].tolist() == []
    for name in ('scope_membership_ptr', 'argument_value_ptr', 'reference_ptr'):
        assert result['structural'][name].tolist() == [0]
        assert result['structural'][name].dtype == torch.int64


@pytest.mark.parametrize('resource_present', [False, True])
def test_structural_positive_lookup_does_not_unbind_scalar_tensors(monkeypatch, resource_present):
    # Iterating argument_binding into scalar Tensor views is the measured GC bug.
    sample, kwargs = fixture()
    sample = with_structural(sample)
    if not resource_present:
        sample = replace(sample, supervision=replace(sample.supervision, resource_targets=()))
    row = tensorize(sample, kwargs)
    binding = row['inputs']['structural']['argument_binding']
    original = torch.Tensor.unbind
    scalar_views = []

    def observed_unbind(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if self is binding:
            scalar_views.extend(result)
        return result

    monkeypatch.setattr(torch.Tensor, 'unbind', observed_unbind)
    result = reconstruct(row)
    assert scalar_views == [], 'Host lookup created avoidable per-item scalar Tensor views'
    assert len(result['structural']['candidate']) == 2
    assert result['structural']['group'].tolist() == [0, 1]
    assert result['structural']['argument_category'].tolist() == [1, 1]
    assert result['structural']['argument_supervised'].tolist() == [True, True]
    assert result['structural']['argument_value_ptr'].tolist() == [0, 0, 0]
    assert len(result['resource']['candidate']) == int(resource_present)


@pytest.mark.parametrize('skill', [False, True])
def test_structural_target_codecs_preserve_literal_values_and_empty_abi(skill):
    sample, kwargs = fixture()
    sample = with_structural(sample, skill=skill)
    result = reconstruct(tensorize(sample, kwargs))['structural']
    assert result['group'].tolist() == [0, 1]
    assert result['supervised'].tolist() == [True, True]
    if skill:
        # fixture normalization: (duration5 - mean1)/scale2 = 2; clear is nonlearnable.
        assert result['argument_value'].tolist() == [2., 1., 2., 1.]
        assert result['argument_supervised'].tolist() == [True, True, False] * 2
        assert result['reference_index'].tolist() == [0, 1]
    else:
        assert result['argument_category'].tolist() == [1, 1]
        assert result['argument_value'].dtype == torch.float32
        assert result['argument_value'].shape == (0,)
