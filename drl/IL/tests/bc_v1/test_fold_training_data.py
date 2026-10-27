"""Fold normalization from the same physical sample, never by inverting G2."""
from dataclasses import replace
import importlib.util
import pytest
import torch

from drl.IL.tests.bc_v1.g2_fixtures import fixture,tensorize,ROOT
from drl.IL.tests.bc_v1.test_resource_decisions import contract
from drl.IL.bc_v1.resource_decisions import add_resource_decisions
from drl.IL.bc_v1.structural_decisions import add_no_structural_supervision
from drl.IL.bc_v1.tensorization import load_tensor_layout
from drl.IL.bc_v1.normalization import normalizer_fingerprint


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.fold_training_data'), 'Raw fold view missing'
    from drl.IL.bc_v1 import fold_training_data
    return fold_training_data


def test_two_normalizers_share_raw_cache_and_preserve_all_decisions():
    a=api();raw,kw=fixture();c=contract(kw)
    frozen=add_resource_decisions(add_no_structural_supervision(tensorize(raw,kw)),c,authoritative_resources={1})
    class Source:
        split='train'
        manifest={'manifest_fingerprint':'g2','identity':{'g1_fingerprint':'physical'}}
        def __len__(self):return 1
        def __getitem__(self,index):return frozen
    reads=[]
    def physical(eid,index):reads.append((eid,index));return raw
    cache=a.RawSampleCache('physical',budget_bytes=2**20,reader=physical)
    altered=replace(kw['normalizers'],scalers=tuple((k,replace(s,scale=s.scale*2)) for k,s in kw['normalizers'].scalers))
    layout=load_tensor_layout(ROOT/'configs/bc_v1/tensor_layout.v1.json')
    first=a.FoldNormalizedView(Source(),cache,contract=c,**kw,tensor_layout=layout)
    second=a.FoldNormalizedView(Source(),cache,contract=c,**{**kw,'normalizers':altered},tensor_layout=layout)
    one,two=first[0],second[0]
    assert len(reads)==1 and cache.stats()['retained_bytes']<=2**20
    assert not torch.equal(one['inputs']['module_x'],two['inputs']['module_x'])
    assert one['routing']['normalizer_fingerprint']==normalizer_fingerprint(kw['normalizers'])
    assert two['routing']['normalizer_fingerprint']==normalizer_fingerprint(altered)
    assert first.cache_identity!=second.cache_identity
    for family in ('resource','structural'):
        for key in ('decision_candidate','decision_group','decision_supervised'):
            assert torch.equal(one['supervision'][family][key],frozen['supervision'][family][key])
            assert torch.equal(two['supervision'][family][key],frozen['supervision'][family][key])
    # No shared mutable output and no change to immutable normalized G2.
    one['inputs']['module_x'].fill_(999)
    assert not torch.equal(first[0]['inputs']['module_x'],one['inputs']['module_x'])
    assert frozen['routing']['normalizer_fingerprint']==normalizer_fingerprint(kw['normalizers'])


def test_raw_fold_view_rejects_test_foreign_g1_and_wrong_physical_sample():
    a=api();raw,kw=fixture();c=contract(kw)
    frozen=add_resource_decisions(add_no_structural_supervision(tensorize(raw,kw)),c)
    class Source:
        split='test'
        manifest={'manifest_fingerprint':'g2','identity':{'g1_fingerprint':'physical'}}
        def __len__(self):return 1
        def __getitem__(self,index):return frozen
    cache=a.RawSampleCache('physical',budget_bytes=2**20,reader=lambda *_:raw)
    args=dict(contract=c,**kw,tensor_layout=load_tensor_layout(ROOT/'configs/bc_v1/tensor_layout.v1.json'))
    with pytest.raises(ValueError,match='TEST'):a.FoldNormalizedView(Source(),cache,**args)
    Source.split='train'
    foreign=a.RawSampleCache('foreign',budget_bytes=2**20,reader=lambda *_:raw)
    with pytest.raises(ValueError,match='G1'):a.FoldNormalizedView(Source(),foreign,**args)
    wrong=a.RawSampleCache('physical',budget_bytes=2**20,reader=lambda *_:replace(raw,audit=replace(raw.audit,transition_index=888)))
    with pytest.raises(ValueError,match='sample|audit'):a.FoldNormalizedView(Source(),wrong,**args)[0]
