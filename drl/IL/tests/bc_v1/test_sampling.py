import importlib
import importlib.util
import json
from collections import Counter

import pytest

from drl.IL.tests.bc_v1.g2_fixtures import fixture, with_structural


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.sampling'), 'Missing G2 train sampler'
    return importlib.import_module('drl.IL.bc_v1.sampling')


def index():
    return api().TrainRowIndex({'a': tuple(range(100)), 'b': (100,)}, {'a': (1,2,3), 'b': (100,)},
        {'g1': 'g1','split': 'split','split_name': 'train'})


def test_configurable_56_plus_8_and_complete_joint_sample():
    s=api().JointActionBatchSampler(index(),seed=5)
    batch=next(iter(s))
    assert len(batch)==64 and all(i in (1,2,3,100) for i in batch[-8:])
    assert s.last_draw=={'ordinary':56,'structural':8,'fallback':False}
    sample,_=fixture();sample=with_structural(sample)
    dataset=[sample]*101
    assert dataset[batch[-1]].supervision.resource_targets and dataset[batch[-1]].supervision.structural_targets
    changed=api().JointActionBatchSampler(index(),batch_size=5,ordinary_quota=3,structural_quota=2,seed=5)
    assert len(next(iter(changed)))==5 and changed.last_draw['structural']==2


def test_empty_structural_pool_falls_back_to_ordinary():
    ix=api().TrainRowIndex({'a':tuple(range(10))},{},{'g1':'g1','split':'split','split_name':'train'})
    s=api().JointActionBatchSampler(ix,seed=10)
    assert len(next(iter(s)))==64
    assert s.last_draw=={'ordinary':64,'structural':0,'fallback':True}


def test_episode_uniform_then_row_uniform_with_large_deterministic_probe():
    ix=api().TrainRowIndex({'a':tuple(range(10000)),'b':tuple(range(10000,20000))},
        {'a':tuple(range(100)), 'b':(10000,)},{'split_name':'train'})
    s=api().JointActionBatchSampler(ix,batch_size=1,ordinary_quota=0,structural_quota=1,seed=234)
    draws=[batch[0] for batch in s];a=sum(i<10000 for i in draws)
    assert 9000<a<11000, 'Rare sampling must be episode-uniform, not row-uniform'
    counts=Counter(i for i in draws if i<10000)
    assert len(counts)==100 and max(counts.values())<180


def test_reproducibility_json_state_resume_and_identity_guard():
    mod=api();a=mod.JointActionBatchSampler(index(),batch_size=4,ordinary_quota=3,structural_quota=1,seed=33)
    b=mod.JointActionBatchSampler(index(),batch_size=4,ordinary_quota=3,structural_quota=1,seed=33)
    ia,ib=iter(a),iter(b)
    for _ in range(3): assert next(ia)==next(ib)
    state=json.loads(json.dumps(a.state_dict()))
    restarted=mod.JointActionBatchSampler(index(),batch_size=4,ordinary_quota=3,structural_quota=1,seed=33)
    restarted.load_state_dict(state);ir=iter(restarted)
    for _ in range(8):assert next(ia)==next(ir)
    state['index_fingerprint']='foreign'
    with pytest.raises(ValueError):restarted.load_state_dict(state)


def test_train_only_and_sequential_evaluation_without_drop_last():
    mod=api()
    with pytest.raises(ValueError):mod.TrainRowIndex({'a':(0,)},{},{'split_name':'validation'})
    for split in ('validation','test'):
        assert list(mod.evaluation_batches(10,batch_size=4,split_name=split))==[[0,1,2,3],[4,5,6,7],[8,9]]


def test_index_builder_opens_only_train_once(monkeypatch):
    mod=api();from drl.IL.bc_v1 import dataset
    a,_=fixture();b=with_structural(a)
    calls=[]
    manifest={'cache_fingerprint':'frozen','identity':{'split_fingerprint':'split'},
        'splits':{'train':['a'],'validation':['v'],'test':['t']},'episodes':{'a':{'samples':2}}}
    monkeypatch.setattr(dataset,'read_cache_manifest',lambda root:manifest)
    def read(root,eid,entry):
        assert eid=='a';calls.append(eid);return (a,b)
    monkeypatch.setattr(dataset,'_load_episode',read)
    ix=mod.build_train_index('unused',expected_cache_fingerprint='frozen')
    assert calls==['a'] and ix.structural_rows=={'a':(1,)} and ix.sample_count==2
    with pytest.raises(ValueError):mod.build_train_index('unused',expected_cache_fingerprint='wrong')
