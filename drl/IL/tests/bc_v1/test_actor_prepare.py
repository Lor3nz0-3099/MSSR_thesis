"""Tiny CPU actor-data lifecycle; no real holdout payloads, GPU or full builds."""
from dataclasses import replace
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from drl.IL.bc_v1 import normalization as f2
from drl.IL.bc_v1.bc_v1_contract import robot_spec_fingerprint
from drl.IL.bc_v1.feature_schema import feature_schema_fingerprint
from drl.IL.bc_v1.split import make_episode_split,build_split_manifest
from drl.IL.bc_v1.sidecar import SidecarWriter,SidecarDataset,_hash,assert_exact
from drl.IL.bc_v1.structural_decisions import add_no_structural_supervision
from drl.IL.bc_v1.resource_decisions import add_resource_decisions
from drl.IL.bc_v1.tensorization import load_tensor_layout
from drl.IL.tests.bc_v1.test_actor_visibility import sample_for,policy,ROOT
from drl.IL.tests.bc_v1.g2_fixtures import tensorize
from drl.IL.tests.bc_v1.test_resource_decisions import contract
from drl.IL.tests.bc_v1.test_sidecar import identity


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.actor_prepare'), 'Missing new actor F2/G2 lifecycle'
    from drl.IL.bc_v1 import actor_prepare
    return actor_prepare


@pytest.fixture
def tiny(tmp_path):
    entries=[dict(episode_id=str(i),lineage_id=str(i),sha256=str(i+1)*64,bc_valid_transitions=1,tag='tiny') for i in range(4)]
    strata=dict(schema_version='mssr.bc_v1.strata.v1',eligible_count_field='bc_valid_transitions',strata=[dict(name='tiny',match={'tag':'tiny'},total=4,quotas={'train':2,'validation':1,'test':1})])
    split=build_split_manifest(make_episode_split(entries,strata),entries,strata,20261006,
        dict(dataset_name='tiny',schema_version='tiny',manifest_sha256='a'*64))
    physical={};inventory={s:[] for s in ('train','validation','test')}
    for part in ('train','validation'):
        for i,eid in enumerate(split['splits'][part]):
            sample,kw=sample_for('gap',[('world.gap.near_edge_x_m',(.8,)),('gap.width_m',(.2+i*.2,)),('world.task_frame.origin_xyz_m',(99,0,0))])
            physical[eid]=replace(sample,audit=replace(sample.audit,episode_id=eid))
            inventory[part].append(dict(episode_id=eid,samples=1,transition_indices_fingerprint=_hash([0])))
    c=contract(kw);ids={**identity(),'resource_contract_fingerprint':c.fingerprint,
        'robot_spec_fingerprint':robot_spec_fingerprint(kw['spec']),'feature_schema_fingerprint':feature_schema_fingerprint(kw['feature_schema']),'split_fingerprint':split['manifest_fingerprint']}
    w=SidecarWriter(tmp_path/'old',ids,inventory)
    for part in ('train','validation'):
        for i,entry in enumerate(inventory[part]):
            s=physical[entry['episode_id']]
            w.add(add_resource_decisions(add_no_structural_supervision(tensorize(s,kw)),c,authoritative_resources={1}),part,i)
    doc=w.finalize();w.close()
    return split,physical,kw,c,tmp_path/'old',doc


def fit(tiny):
    split,physical,kw,c,source,doc=tiny;opened=[]
    def stream(eid):
        assert eid in split['splits']['train'],'F2 accessed holdout payload'
        opened.append(eid);return (physical[eid],)
    stats=api().fit_actor_normalizers({eid:lambda eid=eid:stream(eid) for eid in split['splits']['train']},split,
        spec=kw['spec'],schema=kw['feature_schema'],normalization_policy=kw['normalization_policy'],visibility=policy())
    return stats,opened


def test_fresh_actor_f2_fits_only_train_and_identity_is_visibility_bound(tiny,tmp_path):
    a=api();stats,opened=fit(tiny);split,physical,kw,*_=tiny
    assert opened==sorted(split['splits']['train'])
    assert stats.metadata['complete_train_coverage'] is True
    assert stats.metadata['actor_visibility_fingerprint']==policy().fingerprint
    assert f2.normalizer_fingerprint(stats)!=f2.normalizer_fingerprint(kw['normalizers'])
    context=a.publish_actor_normalizers(stats,tmp_path/'f2',source_root=tmp_path/'source',visibility=policy())
    assert (tmp_path/'f2/normalizers.json').is_file() and (tmp_path/'f2/fit_context.json').is_file()
    with pytest.raises(ValueError):a.publish_actor_normalizers(stats,tmp_path/'f2',source_root=tmp_path/'source',visibility=policy())
    assert context['TRAIN_only'] is True and context['TEST_payload_access']==0
    s=physical[split['splits']['train'][0]]
    with pytest.raises(ValueError,match='visibility'):
        a.tensorize_actor_sample(s,visibility=policy(),tensor_layout=load_tensor_layout(ROOT/'configs/bc_v1/tensor_layout.v1.json'),**kw)


def test_tiny_actor_sidecar_roundtrip_preserves_decisions_candidates_and_seals_test(tiny,tmp_path):
    a=api();stats,_=fit(tiny);split,physical,kw,c,source,old=tiny;opened=[]
    def loader(eid):
        assert eid not in split['splits']['test'];opened.append(eid);return (physical[eid],)
    result=a.build_actor_sidecar(source,tmp_path/'new',sample_loader=loader,split_manifest=split,
        visibility=policy(),contract=c,normalizers=stats,spec=kw['spec'],schema=kw['feature_schema'],
        normalization_policy=kw['normalization_policy'],tensor_layout=load_tensor_layout(ROOT/'configs/bc_v1/tensor_layout.v1.json'))
    doc=json.loads((tmp_path/'new/manifest.json').read_text())
    assert result['TEST_payload_access']==0 and doc['inventory']['test']==[]
    assert set(opened)==set(physical)
    assert doc['identity']['f2_fingerprint']==f2.normalizer_fingerprint(stats)
    assert doc['identity']['tensorizer_fingerprint']!=old['identity']['tensorizer_fingerprint']
    assert doc['actor_visibility']['fingerprint']==policy().fingerprint
    for part in ('train','validation'):
        with SidecarDataset(source,part,expected_identity=old['identity']) as olddata,SidecarDataset(tmp_path/'new',part,expected_identity=doc['identity']) as newdata:
            for i in range(len(newdata)):
                x,y=olddata[i],newdata[i]
                assert_exact(x['inputs']['structural'],y['inputs']['structural'])
                assert_exact(x['inputs']['resource_candidates'],y['inputs']['resource_candidates'])
                assert_exact(x['routing']['physical_supervision'],y['routing']['physical_supervision'])
                for family in ('resource','structural'):
                    for key in ('decision_group','decision_candidate','decision_supervised'):assert_exact(x['supervision'][family][key],y['supervision'][family][key])
                assert not any('task_frame' in key for key in y['inputs']['environment'])
    assert a.verify_actor_sidecar(tmp_path/'new',policy(),c)['integrity']=='PASS'
    with pytest.raises(ValueError,match='exist|collision'):
        a.build_actor_sidecar(source,tmp_path/'new',sample_loader=loader,split_manifest=split,
            visibility=policy(),contract=c,normalizers=stats,spec=kw['spec'],schema=kw['feature_schema'],normalization_policy=kw['normalization_policy'],tensor_layout=load_tensor_layout(ROOT/'configs/bc_v1/tensor_layout.v1.json'))


def test_frozen_d1_goal_evidence_requires_source_code_contract_and_binds_current_goal():
    a=api();sample,kw=sample_for('navigation',[('world.goal.position_xyz_m',(8,0,0))])
    manifest=dict(complete=True,normalized=False,cache_fingerprint='pinned-G1',
        identity={'code':{'task_context':a.file_sha(ROOT/'drl/IL/bc_v1/task_context.py')},'task_config_fingerprint':_hash(json.loads((ROOT/'configs/bc_v1/task_context.json').read_text()))},
        episodes={sample.audit.episode_id:{'cache_sha256':'verified-by-loader'}},source_identity={'manifest_sha256':'immutable-source'})
    provider=a.FrozenD1GoalProvider(manifest,ROOT/'configs/bc_v1/task_context.json')
    certificate=provider(sample);assert certificate.source_contract_id=='mssr.bc_v1.d1.current_physical_goal.v1'
    from drl.IL.tests.bc_v1.test_actor_visibility import project
    assert project(sample,kw,assigned_goal=certificate).environment
    bad={**manifest,'identity':{**manifest['identity'],'code':{'task_context':'foreign'}}}
    with pytest.raises(ValueError,match='D1'):a.FrozenD1GoalProvider(bad,ROOT/'configs/bc_v1/task_context.json')


def test_preflight_and_manual_helper_are_safe_metadata_only(tmp_path):
    import subprocess,os,tempfile
    a=api()
    # Production outputs may already exist. Only this test owns these fresh paths;
    # keep them under artifacts as required by the real preflight safety contract.
    with tempfile.TemporaryDirectory(prefix='test-actor-preflight-',dir=ROOT/'artifacts/bc_v1') as directory:
        document=json.loads(a.CONFIG.read_text())
        document.update(sidecar_output=str(Path(directory)/'sidecar'),
                        normalizer_output=str(Path(directory)/'normalizers'))
        # Test the current implementation, not the frozen historical rebuild receipt.
        code='drl/IL/bc_v1/tensorization.py'
        document['file_pins'][code]=a.file_sha(ROOT/code)
        config=tmp_path/'actor-prepare.json';config.write_text(json.dumps(document))
        cfg=a.preflight(config);assert cfg['visibility'].document['radius_m']==1.
        assert not cfg['normalizer_output'].exists() and not cfg['sidecar_output'].exists()
        helper=ROOT/cfg['config']['manual_helper'];assert helper.is_file()
        # Parse shell only; neither the helper nor a rebuild is executed.
        r=subprocess.run(['bash','-n',helper],capture_output=True,text=True,timeout=10)
        assert r.returncode==0,r.stderr
        r=subprocess.run([sys.executable,'-B','-m','drl.IL.bc_v1.actor_prepare','--config',str(config)],cwd=ROOT,
            env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':''},capture_output=True,text=True,timeout=30)
        assert r.returncode==0,r.stdout+r.stderr
        assert 'DRY_RUN' in r.stdout and 'TEST_payload_access' in r.stdout
        assert not cfg['normalizer_output'].exists() and not cfg['sidecar_output'].exists()
        cfg['sidecar_output'].mkdir()
        with pytest.raises(ValueError,match='output exists; no overwrite/resume'):
            a.preflight(config)


def test_f2_identity_binds_goal_provider_semantics_even_without_goal_support(tiny):
    a=api();split,physical,kw,*_=tiny
    class NoGoals:
        def __init__(self,identity):self.identity=identity
        def __call__(self,sample):return None
    def fit_provider(identity):
        return a.fit_actor_normalizers({eid:lambda eid=eid:(physical[eid],) for eid in split['splits']['train']},split,
            spec=kw['spec'],schema=kw['feature_schema'],normalization_policy=kw['normalization_policy'],visibility=policy(),goal_provider=NoGoals(identity))
    first,second=fit_provider('mission-contract-A'),fit_provider('mission-contract-B')
    assert f2.normalizer_fingerprint(first)!=f2.normalizer_fingerprint(second),'Goal certification semantics must invalidate F2 compatibility even with no observed goals'


def test_builder_rejects_foreign_split_normalizers_before_creating_output(tiny,tmp_path):
    a=api();stats,_=fit(tiny);split,physical,kw,c,source,doc=tiny
    metadata={**stats.metadata,'split_fingerprint':'foreign'}
    stats=replace(stats,metadata_items=tuple(sorted(metadata.items())))
    with pytest.raises(ValueError,match='split|F1'):
        a.build_actor_sidecar(source,tmp_path/'foreign',sample_loader=lambda e:(physical[e],),split_manifest=split,
            visibility=policy(),contract=c,normalizers=stats,spec=kw['spec'],schema=kw['feature_schema'],
            normalization_policy=kw['normalization_policy'],tensor_layout=load_tensor_layout(ROOT/'configs/bc_v1/tensor_layout.v1.json'))
    assert not (tmp_path/'foreign').exists() and not (tmp_path/'foreign.building').exists()
