"""P0 lossless indexed G2 storage; synthetic tests never open frozen datasets."""
import copy
import importlib
import importlib.util
import json
from dataclasses import replace

import pytest
import torch

from drl.IL.tests.bc_v1.g2_fixtures import fixture, tensorize, with_structural
from drl.IL.bc_v1 import dataset as g1,normalization as f2

OLD_LAYOUT='19d649f2348a286703e7df31e98ce516e3c2eed16cec91b0b3de036ed9c17f2f'
CURRENT_LAYOUT='e5b6a313062746a68bc5c1ba43cd200479af15f1b9d1e3cee24a1675a90deb49'


@pytest.fixture(autouse=True,scope='module')
def cpu_thread_budget():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.sidecar'), 'Missing P0 indexed G2 storage'
    return importlib.import_module('drl.IL.bc_v1.sidecar')


def identity():
    return dict(canonical_sha='canonical',f1_sha='split-bytes',g1_fingerprint='g1',
        f2_fingerprint=f2.normalizer_fingerprint(fixture()[1]['normalizers']),normalizer_sha='normalizer',layout_fingerprint=CURRENT_LAYOUT,
        feature_schema_fingerprint='features',robot_spec_fingerprint='robot',
        tensorizer_fingerprint='tensorizer',storage_fingerprint='storage',split_fingerprint='logical-split')


def assert_equal(a,b):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert isinstance(b,dict) and a.keys()==b.keys()
        for k in a:assert_equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert type(a) is type(b) and len(a)==len(b)
        for x,y in zip(a,b):assert_equal(x,y)
    else:assert a==b


def row(eid='a',transition=7,*,structural=False):
    s,kw=fixture()
    if structural:s=with_structural(s)
    s=replace(s,audit=replace(s.audit,episode_id=eid,transition_index=transition))
    return tensorize(s,kw)


def catalog():
    return {p:[{'episode_id':e,'samples':1,'transition_indices_fingerprint':g1._hash([7])}]
        for p,e in [('train','a'),('validation','v'),('test','t')]}


def built(tmp_path,*,ids=None):
    m=api();root=tmp_path/'sidecar';w=m.SidecarWriter(root,ids or identity(),catalog())
    for split,eid in [('train','a'),('validation','v'),('test','t')]:
        sample=row(eid)
        sample['routing']['layout_fingerprint']=(ids or identity())['layout_fingerprint']
        w.add(sample,split,0)
    w.finalize();return root


def test_valid_manifest_opens_and_resolves_exact_sample(tmp_path):
    root=built(tmp_path)
    with api().SidecarDataset(root,'train',expected_identity=identity()) as data:
        assert len(data)==1 and data.manifest['complete'] is True
        assert_equal(data[0],row())


@pytest.mark.parametrize('key',list(identity()))
def test_stale_upstream_identity_is_rejected(tmp_path,key):
    root=built(tmp_path);expected=identity();expected[key]='foreign'
    with pytest.raises(ValueError,match='[Ii]ncompatible|[Ss]tale'):
        api().SidecarDataset(root,'train',expected_identity=expected)


def test_obsolete_layout_rejected_current_layout_accepted(tmp_path):
    ids=identity();ids['layout_fingerprint']=OLD_LAYOUT;root=built(tmp_path,ids=ids)
    with pytest.raises(ValueError,match='[Ii]ncompatible|[Ss]tale'):
        api().SidecarDataset(root,'train',expected_identity=identity())


@pytest.mark.parametrize('field,value',[('schema_version','wrong'),('complete',False)])
def test_wrong_schema_or_incomplete_manifest_rejected(tmp_path,field,value):
    root=built(tmp_path);p=root/'manifest.json';doc=json.loads(p.read_text());doc[field]=value;p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):api().SidecarDataset(root,'train',expected_identity=identity())


def test_lossless_codec_preserves_all_g2_sections_and_empty_sentinels():
    sample=row(structural=True)
    sample['extra_fixture']={'empty':torch.empty((0,3),dtype=torch.float32),
        'mask':torch.tensor([True,False]),'index':torch.tensor([-1,2],dtype=torch.int64)}
    blob=api().encode(sample);restored=api().decode(blob)
    assert_equal(restored,sample)
    restored['inputs']['module_x'][0,0]+=1
    assert not torch.equal(restored['inputs']['module_x'],sample['inputs']['module_x'])


def test_identical_templates_shared_only_when_content_identical(tmp_path):
    cat={'train':[{'episode_id':'a','samples':3,'transition_indices_fingerprint':g1._hash([7,8,9])}],
         'validation':[],'test':[]}
    writer=api().SidecarWriter(tmp_path/'cache',identity(),cat)
    a=row('a',7);b=row('a',8);c=row('a',9)
    c['inputs']['structural']['peer']=c['inputs']['structural']['peer'].clone()
    c['inputs']['structural']['peer'][0]=-1
    writer.add(a,'train',0);writer.add(b,'train',1);writer.add(c,'train',2)
    manifest=writer.finalize()
    assert manifest['template_count']==2
    with api().SidecarDataset(tmp_path/'cache','train',expected_identity=identity()) as data:
        for i,expected in enumerate([a,b,c]):assert_equal(data[i],expected)
        changed=data[0];changed['inputs']['structural']['peer'].fill_(-99)
        assert_equal(data[0],a)  # Caller mutations cannot poison template cache.


def test_interrupted_build_is_explicit_and_resume_reuses_completed_records(tmp_path):
    w=api().SidecarWriter(tmp_path/'cache',identity(),catalog());w.add(row(),'train',0);w.close()
    assert not (tmp_path/'cache').exists()
    with pytest.raises(ValueError):api().SidecarDataset(tmp_path/'cache.building','train',expected_identity=identity())
    with pytest.raises(ValueError):api().SidecarWriter(tmp_path/'cache',{**identity(),'g1_fingerprint':'foreign'},catalog(),resume=True)
    w=api().SidecarWriter(tmp_path/'cache',identity(),catalog(),resume=True)
    assert w.has('train',0)
    for split,eid in [('validation','v'),('test','t')]:w.add(row(eid),split,0)
    assert w.finalize()['sample_count']==3


def test_lookup_by_logical_identity_and_missing_identity_fail_clearly(tmp_path):
    root=built(tmp_path)
    with api().SidecarDataset(root,'train',expected_identity=identity()) as data:
        assert_equal(data.get('a',7),row())
        with pytest.raises(ValueError,match='a.*999'):data.get('a',999)
        with pytest.raises(ValueError):data.get('v',7)


def test_sampler_same_seed_and_state_keeps_56_plus_8_and_logical_rows(tmp_path):
    from drl.IL.bc_v1.sampling import TrainRowIndex,JointActionBatchSampler
    cat={'train':[{'episode_id':'a','samples':3,'transition_indices_fingerprint':g1._hash([7,8,9])}],
         'validation':[],'test':[]}
    w=api().SidecarWriter(tmp_path/'cache',identity(),cat)
    expected=[row('a',7),row('a',8,structural=True),row('a',9)]
    for i,r in enumerate(expected):w.add(r,'train',i)
    w.finalize()
    with api().SidecarDataset(tmp_path/'cache','train',expected_identity=identity()) as data:
        original=TrainRowIndex({'a':(0,1,2)},{'a':(1,)},{'split_name':'train','g1':'g1',
            'split':'logical-split','f2':identity()['f2_fingerprint'],'tensor_layout':CURRENT_LAYOUT})
        cached=data.train_index();assert cached.fingerprint==original.fingerprint
        a=JointActionBatchSampler(original,seed=17);b=JointActionBatchSampler(cached,seed=17)
        b.load_state_dict(a.state_dict());left=next(iter(a));right=next(iter(b))
        assert left==right and b.last_draw=={'ordinary':56,'structural':8,'fallback':False}
        for i in right:assert_equal(data[i],expected[i])


@pytest.mark.parametrize('split,eid',[('validation','v'),('test','t')])
def test_holdout_iteration_is_sequential_repeatable_and_complete(tmp_path,split,eid):
    from drl.IL.bc_v1.sampling import evaluation_batches
    root=built(tmp_path)
    with api().SidecarDataset(root,split,expected_identity=identity()) as data:
        keys=list(data.sample_keys());assert keys==[(eid,7)] and list(data.sample_keys())==keys
        actual=[data[i]['routing']['audit'].transition_index for batch in evaluation_batches(len(data),batch_size=64,split_name=split) for i in batch]
        assert actual==[7]


@pytest.mark.parametrize('fault',['record','template','missing','mapping','table','truncate'])
def test_storage_corruption_or_missing_reference_never_falls_back(tmp_path,fault):
    import sqlite3
    root=built(tmp_path);path=root/'data.sqlite3'
    if fault=='truncate':
        with path.open('r+b') as f:f.truncate(100)
    else:
        with sqlite3.connect(path) as db:
            if fault=='record':db.execute("UPDATE samples SET payload=? WHERE split='train'",(b'broken',))
            if fault=='template':db.execute('UPDATE templates SET payload=?',(b'broken',))
            if fault=='missing':db.execute("DELETE FROM samples WHERE split='train'")
            if fault=='mapping':db.execute("UPDATE samples SET episode='foreign' WHERE split='train'")
            if fault=='table':db.execute('DROP TABLE templates')
    with pytest.raises(ValueError):
        with api().SidecarDataset(root,'train',expected_identity=identity()) as data:data[0]


def test_deep_verify_covers_all_splits_records_and_templates(tmp_path):
    root=built(tmp_path);report=api().verify_sidecar(root,expected_identity=identity())
    assert report['samples']==3 and report['episodes']==3 and report['duplicates']==0
    assert report['splits']=={'train':1,'validation':1,'test':1}


def test_random_access_never_calls_g1_f2_or_g2(tmp_path,monkeypatch):
    root=built(tmp_path);expected=row()
    from drl.IL.bc_v1 import tensorization,normalization
    def forbid(*a,**kw):raise AssertionError('Training access invoked upstream computation')
    monkeypatch.setattr(g1,'_load_episode',forbid)
    monkeypatch.setattr(tensorization,'tensorize_policy_sample',forbid)
    monkeypatch.setattr(normalization,'normalize_features',forbid)
    with api().SidecarDataset(root,'train',expected_identity=identity()) as data:
        for _ in range(8):assert_equal(data[0],expected)


def test_sample_routing_cannot_claim_obsolete_layout_under_current_manifest(tmp_path):
    w=api().SidecarWriter(tmp_path/'cache',identity(),catalog());bad=row()
    bad['routing']['layout_fingerprint']=OLD_LAYOUT
    with pytest.raises(ValueError,match='identity|fingerprint'):
        w.add(bad,'train',0)
    w.close()


def test_build_commits_bounded_chunks_instead_of_fsync_per_sample(tmp_path):
    cat={'train':[{'episode_id':'a','samples':8,'transition_indices_fingerprint':g1._hash(list(range(8)))}],
         'validation':[],'test':[]}
    w=api().SidecarWriter(tmp_path/'cache',identity(),cat,commit_interval=4)
    statements=[];w.db.set_trace_callback(statements.append)
    for i in range(8):w.add(row('a',i),'train',i)
    assert sum(s=='COMMIT' for s in statements)==2
    w.finalize()


def test_build_uses_transactional_wal_but_published_artifact_needs_no_wal(tmp_path):
    import sqlite3
    w=api().SidecarWriter(tmp_path/'cache',identity(),catalog())
    assert w.db.execute('PRAGMA journal_mode').fetchone()==('wal',)
    for split,eid in [('train','a'),('validation','v'),('test','t')]:w.add(row(eid),split,0)
    w.finalize()
    with sqlite3.connect((tmp_path/'cache'/'data.sqlite3').resolve().as_uri()+'?mode=ro',uri=True) as db:
        assert db.execute('PRAGMA journal_mode').fetchone()==('delete',)
    assert not list((tmp_path/'cache').glob('*-wal'))


def test_episode_worker_import_keeps_global_indices_and_sample_content(tmp_path):
    one={'train':catalog()['train'],'validation':[],'test':[]}
    child=api().SidecarWriter(tmp_path/'episode',identity(),one);child.add(row(),'train',0);child.finalize()
    parent=api().SidecarWriter(tmp_path/'all',identity(),catalog())
    parent.import_episode(tmp_path/'episode','train','a')
    for split,eid in [('validation','v'),('test','t')]:parent.add(row(eid),split,0)
    parent.finalize()
    with api().SidecarDataset(tmp_path/'all','train',expected_identity=identity()) as data:assert_equal(data[0],row())


def test_collate_h1_h2_i1_i2_are_identical_for_joint_and_endpoint_free_samples(tmp_path):
    from drl.IL.bc_v1.batching import collate_policy_samples
    from drl.IL.bc_v1.losses import compute_bc_loss,load_loss_config
    from drl.IL.bc_v1.metrics import BCMetrics
    from drl.IL.tests.bc_v1.test_action_heads import complete_model
    from drl.IL.tests.bc_v1.g2_fixtures import ROOT
    a,kw=fixture();b,kb=fixture(2)
    a=with_structural(a);b=with_structural(b,skill=True)
    a=replace(a,audit=replace(a.audit,episode_id='a',transition_index=7))
    b=replace(b,audit=replace(b.audit,episode_id='a',transition_index=8))
    direct=[tensorize(a,kw),tensorize(b,kb)]
    cat={'train':[{'episode_id':'a','samples':2,'transition_indices_fingerprint':g1._hash([7,8])}],
         'validation':[],'test':[]}
    w=api().SidecarWriter(tmp_path/'cache',identity(),cat)
    for i,s in enumerate(direct):w.add(s,'train',i)
    w.finalize()
    with api().SidecarDataset(tmp_path/'cache','train',expected_identity=identity()) as data:
        cached=[data.get('a',7),data.get('a',8)]
    original=collate_policy_samples(direct);optimized=collate_policy_samples(cached);assert_equal(original,optimized)
    model=complete_model(kw);loss_config=load_loss_config(ROOT/'configs/bc_v1/loss_v1.json')
    with torch.no_grad():
        ha=model.encode(original['inputs']);hb=model.encode(optimized['inputs']);assert_equal(ha,hb)
        pa=model.forward_heads(ha,original['inputs']);pb=model.forward_heads(hb,optimized['inputs']);assert_equal(pa,pb)
        assert_equal(compute_bc_loss(pa,original,loss_config),compute_bc_loss(pb,optimized,loss_config))
    ma=BCMetrics(kw['spec'],kw['normalizers']);mb=BCMetrics(kw['spec'],kw['normalizers'])
    ma.update(pa,original);mb.update(pb,optimized);assert_equal(ma.compute(),mb.compute())


def test_train_index_rejects_missing_tail_instead_of_silently_shrinking_population(tmp_path):
    import sqlite3
    root=tmp_path/'tail';cat=catalog();cat['train'][0].update(samples=2,transition_indices_fingerprint=g1._hash([7,8]))
    w=api().SidecarWriter(root,identity(),cat)
    w.add(row('a',7),'train',0);w.add(row('a',8),'train',1)
    w.add(row('v'),'validation',0);w.add(row('t'),'test',0);w.finalize()
    with sqlite3.connect(root/'data.sqlite3') as db:db.execute("DELETE FROM samples WHERE split='train' AND idx=1")
    with api().SidecarDataset(root,'train',expected_identity=identity()) as data:
        with pytest.raises(ValueError):data.train_index()
