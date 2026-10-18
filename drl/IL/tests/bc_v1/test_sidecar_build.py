"""Builder DI tests use real G2 and no external source artifacts."""
from dataclasses import replace
import importlib
import importlib.util

import pytest
import torch

from drl.IL.tests.bc_v1.g2_fixtures import fixture,tensorize
from drl.IL.tests.bc_v1.test_sidecar import identity,catalog,assert_equal
from drl.IL.bc_v1 import dataset as g1


@pytest.fixture(autouse=True)
def thread_budget():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.sidecar_build'),'Missing P0 rebuild tool'
    return importlib.import_module('drl.IL.bc_v1.sidecar_build')


def sources(tmp_path):
    sample,kwargs=fixture();rows={eid:(replace(sample,audit=replace(sample.audit,episode_id=eid,transition_index=7)),)
        for eid in ('a','v','t')}
    return {'cache_root':tmp_path/'g1','manifest':{'episodes':{eid:{} for eid in rows}},
        'identity':identity(),'inventory':catalog(),'kwargs':kwargs},rows


def test_builder_calls_authoritative_g2_once_per_row_and_loads_each_episode_once(tmp_path):
    src,rows=sources(tmp_path);calls=[]
    def reader(root,eid,entry):calls.append(eid);return rows[eid]
    doc=api().build_sidecar(src,tmp_path/'sidecar',episode_reader=reader)
    assert calls==['a','v','t'] and doc['sample_count']==3
    from drl.IL.bc_v1.sidecar import SidecarDataset
    for split,eid in [('train','a'),('validation','v'),('test','t')]:
        with SidecarDataset(tmp_path/'sidecar',split,expected_identity=identity()) as data:
            assert_equal(data[0],tensorize(rows[eid][0],src['kwargs']))


def test_failed_build_remains_incomplete_and_resume_does_not_reload_committed_episode(tmp_path):
    src,rows=sources(tmp_path);calls=[]
    def fail(root,eid,entry):
        calls.append(eid)
        if eid=='v':raise RuntimeError('source fault')
        return rows[eid]
    with pytest.raises(RuntimeError,match='source fault'):
        api().build_sidecar(src,tmp_path/'sidecar',episode_reader=fail)
    assert not (tmp_path/'sidecar').exists()
    def success(root,eid,entry):calls.append(eid);return rows[eid]
    api().build_sidecar(src,tmp_path/'sidecar',resume=True,episode_reader=success)
    assert calls==['a','v','v','t']


def test_output_inside_upstream_is_rejected_before_reading(tmp_path):
    src,rows=sources(tmp_path)
    with pytest.raises(ValueError):api().build_sidecar(src,src['cache_root']/'illegal')


def test_parallel_episode_build_matches_direct_without_whole_dataset_ram(tmp_path):
    src,rows=sources(tmp_path);src['cache_root'].mkdir()
    for eid,values in rows.items():
        path=src['cache_root']/(eid+'.pt')
        torch.save({'schema_version':g1.VERSION,'episode_id':eid,'samples':values},path)
        src['manifest']['episodes'][eid]={'cache_file':path.name,'cache_sha256':g1.file_hash(path),
            'samples':1,'transition_indices_fingerprint':g1._hash([7])}
    doc=api().build_sidecar(src,tmp_path/'parallel',workers=2)
    assert doc['sample_count']==3
    from drl.IL.bc_v1.sidecar import SidecarDataset
    with SidecarDataset(tmp_path/'parallel','train',expected_identity=identity()) as data:
        assert_equal(data[0],tensorize(rows['a'][0],src['kwargs']))
    assert not (tmp_path/'parallel'/'workers').exists()


def physical_sources(tmp_path,*,partial=False):
    src,rows=sources(tmp_path);src['cache_root'].mkdir()
    if partial:
        base=rows['a'][0]
        rows['a']=tuple(replace(base,audit=replace(base.audit,transition_index=i)) for i in (7,8,9))
        src['inventory']['train'][0]={'episode_id':'a','samples':3,'transition_indices_fingerprint':g1._hash([7,8,9])}
    for eid,values in rows.items():
        p=src['cache_root']/(eid+'.pt');torch.save({'schema_version':g1.VERSION,'episode_id':eid,'samples':values},p)
        src['manifest']['episodes'][eid]={'cache_file':p.name,'cache_sha256':g1.file_hash(p),'samples':len(values),
            'transition_indices_fingerprint':g1._hash([s.audit.transition_index for s in values])}
    return src,rows


def test_parallel_resume_cleans_only_already_imported_owned_episode_artifacts(tmp_path):
    from drl.IL.bc_v1.sidecar import SidecarWriter,_hash
    src,rows=physical_sources(tmp_path);root=tmp_path/'output'
    parent=SidecarWriter(root,identity(),src['inventory']);parent.add(tensorize(rows['a'][0],src['kwargs']),'train',0);parent.close()
    childroot=parent.stage/'workers'/_hash('a')
    child=SidecarWriter(childroot,identity(),{'train':src['inventory']['train'],'validation':[],'test':[]})
    child.add(tensorize(rows['a'][0],src['kwargs']),'train',0);child.finalize()
    doc=api().build_sidecar(src,root,resume=True,workers=2)
    assert doc['sample_count']==3 and not (root/'workers').exists()


def test_partial_serial_episode_can_resume_with_default_parallel_option(tmp_path,monkeypatch):
    from drl.IL.bc_v1.sidecar import SidecarWriter,SidecarDataset
    from drl.IL.bc_v1 import tensorization
    src,rows=physical_sources(tmp_path,partial=True);root=tmp_path/'output'
    w=SidecarWriter(root,identity(),src['inventory']);w.add(tensorize(rows['a'][0],src['kwargs']),'train',0);w.close()
    calls=[];original=tensorization.tensorize_policy_sample
    def observed(sample,**kw):calls.append((sample.audit.episode_id,sample.audit.transition_index));return original(sample,**kw)
    monkeypatch.setattr(tensorization,'tensorize_policy_sample',observed);events=[]
    doc=api().build_sidecar(src,root,resume=True,workers=2,progress=events.append)
    assert doc['sample_count']==5 and ('a',7) not in calls and len(calls)==4
    assert any(e.get('resume_strategy')=='serial_missing_rows' for e in events)
    with SidecarDataset(root,'train',expected_identity=identity()) as data:assert len(data)==3


@pytest.mark.parametrize('reserved',[False,True])
def test_publication_interruption_resumes_without_rereading_sources(tmp_path,monkeypatch,reserved):
    from drl.IL.bc_v1 import sidecar
    src,rows=sources(tmp_path);root=tmp_path/'output';w=sidecar.SidecarWriter(root,identity(),src['inventory'])
    for split,eid in [('train','a'),('validation','v'),('test','t')]:w.add(tensorize(rows[eid][0],src['kwargs']),split,0)
    original=sidecar.os.rename
    def interrupted(a,b):raise OSError('interrupted publication')
    monkeypatch.setattr(sidecar.os,'rename',interrupted)
    with pytest.raises(OSError,match='interrupted publication'):w.finalize()
    w.close();monkeypatch.setattr(sidecar.os,'rename',original)
    if not reserved:root.rmdir()
    def forbid(*a,**kw):raise AssertionError('Complete publication recovery reread source')
    doc=api().build_sidecar(src,root,resume=True,episode_reader=forbid)
    assert doc['complete'] is True and doc['sample_count']==3
