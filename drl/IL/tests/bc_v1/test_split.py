"""F1 grouped holdout: literal quotas and independent membership invariants."""

from collections import Counter
import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import random
import subprocess
import sys

import pytest

from drl.IL.tests.bc_v1.fixtures import require_canonical_v6_fixture

ROOT = Path(__file__).resolve().parents[4]
ARTIFACT = ROOT / 'artifacts/bc_v1/splits/f1_v1'
PARTS = ('train', 'validation', 'test')
QUOTAS = {
    'expert_gap': (12,3,3), 'expert_stairs': (12,3,3),
    'expert_navigation': (9,2,2), 'teleop_button': (6,1,1),
    'teleop_mission': (9,2,2), 'expert_assembly': (1,1,1),
}


def api():
    name = 'drl.IL.bc_v1.split'
    assert importlib.util.find_spec(name) is not None, 'Missing F1 grouped episode split'
    return importlib.import_module(name)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def config():
    return {'schema_version':'mssr.bc_v1.strata.v1', 'eligible_count_field':'bc_valid_transitions',
        'strata':[{'name':name, 'match':{'tag':name}, 'total':sum(q),
                   'quotas':dict(zip(PARTS,q))} for name,q in QUOTAS.items()]}


def episodes():
    rows = []
    for name,q in QUOTAS.items():
        for i in range(sum(q)):
            eid = f'{name}:{i:02d}'
            rows.append({'episode_id':eid, 'lineage_id':eid, 'sha256':digest(eid),
                'tag':name, 'bc_valid_transitions':19 if name=='expert_assembly' else 100+i,
                'kind':'teleop' if name.startswith('teleop') else 'expert',
                'source_hashes':{f'stream/{eid}':digest('raw:'+eid)},
                'lineage_aliases':[f'raw/{eid}',f'compact/{eid}'],
                'supervised_instances':{'gravity_settle':int(i==0), 'wheel_velocity':2*i}})
    for i in range(7):
        eid=f'excluded:{i}'
        rows.append({'episode_id':eid,'lineage_id':eid,'sha256':digest(eid),'bc_valid_transitions':0})
    return rows


def small(weights=(2,2,3), quotas=(3,2,2)):
    rows=[]
    for i,w in enumerate(weights):
        for j in range(w):
            eid=f'g{i}-variant{j}'
            rows.append({'episode_id':eid, 'lineage_id':f'g{i}', 'tag':'arbitrary',
                         'sha256':digest(eid), 'bc_valid_transitions':1})
    cfg={'schema_version':'mssr.bc_v1.strata.v1','eligible_count_field':'bc_valid_transitions',
         'strata':[{'name':'other_robot_task','match':{'tag':'arbitrary'},
                    'total':sum(quotas),'quotas':dict(zip(PARTS,quotas))}]}
    return rows,cfg


def test_exact_73_eligible_and_49_12_12_disjoint_union():
    result=api().make_episode_split(episodes(),config())
    assert tuple(result)==PARTS and tuple(map(len,result.values()))==(49,12,12)
    expected={e['episode_id'] for e in episodes() if e['bc_valid_transitions']>0}
    assert len(expected)==73 and set().union(*map(set,result.values()))==expected
    for a,b in (('train','validation'),('train','test'),('validation','test')):
        assert set(result[a]).isdisjoint(result[b])
    assert all(isinstance(v,tuple) and all(type(eid) is str for eid in v) for v in result.values())


def test_exact_six_quotas_and_assembly_1_1_1_without_manual_choice():
    rows=episodes(); result=api().make_episode_split(rows,config())
    tags={e['episode_id']:e.get('tag') for e in rows}
    for i,part in enumerate(PARTS):
        assert Counter(tags[e] for e in result[part])=={name:q[i] for name,q in QUOTAS.items()}
    assembly={e['episode_id'] for e in rows if e.get('tag')=='expert_assembly'}
    assert len(assembly)==3 and all(len(assembly.intersection(result[p]))==1 for p in PARTS)
    api().validate_split(result,rows,config())


def test_reproducible_under_episode_and_configuration_order_without_mutation():
    rows=episodes(); cfg=config(); before=copy.deepcopy((rows,cfg))
    a=api().make_episode_split(rows,cfg,20261006)
    cfg2=copy.deepcopy(cfg);cfg2['strata'].reverse()
    cfg2['strata']=[dict(reversed(list(s.items()))) for s in cfg2['strata']]
    assert api().make_episode_split(rows[::-1],cfg2,20261006)==a
    assert api().make_episode_split(rows,cfg,20261006)==a
    assert (rows,cfg)==before
    assert api().strata_fingerprint(cfg)==api().strata_fingerprint(cfg2)


def test_split_does_not_touch_global_python_or_numpy_rng():
    api()
    import numpy as np
    python=random.getstate(); numpy=np.random.get_state()
    api().make_episode_split(episodes(),config())
    assert random.getstate()==python
    other=np.random.get_state()
    assert numpy[0]==other[0] and (numpy[1]==other[1]).all() and numpy[2:]==other[2:]


@pytest.mark.parametrize('seed',[0,1,7,20261006])
def test_raw_compact_variants_are_indivisible_and_feasible_weighted_groups_work(seed):
    rows,cfg=small(); result=api().make_episode_split(rows,cfg,seed)
    assert tuple(map(len,result.values()))==(3,2,2)
    owners={eid:p for p,ids in result.items() for eid in ids}
    for lineage in ('g0','g1','g2'):
        assert len({owners[e['episode_id']] for e in rows if e['lineage_id']==lineage})==1
    api().validate_split(result,rows,cfg)


def test_incompatible_group_sizes_raise_without_changing_quotas():
    rows,cfg=small((2,2,2),(3,2,1))
    with pytest.raises(ValueError,match='(?i)group|lineage|quota'):
        api().make_episode_split(rows,cfg)


@pytest.mark.parametrize('fault',['duplicate_id','same_hash_other_lineage','shared_alias_other_lineage',
    'shared_source_other_lineage','source_hash_conflict','group_cross_strata','missing_lineage',
    'invalid_hash','invalid_bc_count','row_instead_of_episode','missing_stratum','ambiguous_stratum'])
def test_incoherent_inventory_fails_explicitly(fault):
    api();rows=episodes();cfg=config()
    if fault=='duplicate_id':rows.append(copy.deepcopy(rows[0]))
    elif fault=='same_hash_other_lineage':rows[1]['sha256']=rows[0]['sha256']
    elif fault=='shared_alias_other_lineage':rows[1]['lineage_aliases']=rows[0]['lineage_aliases']
    elif fault=='shared_source_other_lineage':rows[1]['source_hashes']=rows[0]['source_hashes']
    elif fault=='source_hash_conflict':
        rows[1]['lineage_id']=rows[0]['lineage_id'];rows[1]['source_hashes']={next(iter(rows[0]['source_hashes'])):digest('conflict')}
    elif fault=='group_cross_strata':rows[18]['lineage_id']=rows[0]['lineage_id']
    elif fault=='missing_lineage':del rows[0]['lineage_id']
    elif fault=='invalid_hash':rows[0]['sha256']='not-a-digest'
    elif fault=='invalid_bc_count':rows[0]['bc_valid_transitions']=True
    elif fault=='row_instead_of_episode':rows=[{'provenance':{'logical_episode_id':'foo','canonical_transition_index':0}}]
    elif fault=='missing_stratum':rows[0]['tag']='unclassified'
    elif fault=='ambiguous_stratum':cfg['strata'].append({'name':'overlap','match':{'tag':'expert_gap'},'total':1,'quotas':dict(train=1,validation=0,test=0)})
    with pytest.raises(ValueError):api().make_episode_split(rows,cfg)


@pytest.mark.parametrize('fault',['missing','duplicate','unknown','wrong_counts','lineage_split'])
def test_validate_split_rejects_bad_external_membership(fault):
    rows,cfg=small();valid=api().make_episode_split(rows,cfg)
    bad={k:list(v) for k,v in valid.items()}
    if fault=='missing':bad['train'].pop()
    elif fault=='duplicate':bad['train'].append(bad['validation'][0])
    elif fault=='unknown':bad['train'][0]='unknown'
    elif fault=='wrong_counts':bad['test'].append(bad['train'].pop())
    else:bad['validation'][0],bad['test'][0]=bad['test'][0],bad['validation'][0]
    with pytest.raises(ValueError):api().validate_split(bad,rows,cfg)


def identity():
    return {'dataset_name':'synthetic','schema_version':'canonical.v5','manifest_sha256':digest('manifest')}


def test_manifest_serialization_fingerprints_and_readback_are_deterministic(tmp_path):
    module=api();rows=episodes();cfg=config();s=module.make_episode_split(rows,cfg)
    a=module.build_split_manifest(s,rows,cfg,20261006,identity())
    b=module.build_split_manifest(s,rows[::-1],cfg,20261006,dict(reversed(list(identity().items()))))
    assert a==b and module.serialize_split_manifest(a)==module.serialize_split_manifest(b)
    assert a['schema_version']=='mssr.bc_v1.episode_split.v1'
    assert a['global_counts']==dict(train=49,validation=12,test=12)
    assert a['config_fingerprint']==module.strata_fingerprint(cfg)
    assert len(a['manifest_fingerprint'])==64
    assert len(a['excluded_episode_ids'])==7 and len(a['episode_metadata'])==80
    assert len(a['lineage_groups'])==73
    path=tmp_path/'derived'/'split.json'
    module.write_split_manifest(a,path,source_root=tmp_path/'source')
    assert module.read_split_manifest(path,rows,cfg,identity())==a
    module.write_split_manifest(a,path,source_root=tmp_path/'source')


@pytest.mark.parametrize('fault',['membership','fingerprint','source','new_demo','config'])
def test_frozen_manifest_rejects_tampering_or_new_source(fault,tmp_path):
    module=api();rows=episodes();cfg=config();ident=identity()
    value=module.build_split_manifest(module.make_episode_split(rows,cfg),rows,cfg,20261006,ident)
    if fault=='membership':value['splits']['train'][0],value['splits']['test'][0]=value['splits']['test'][0],value['splits']['train'][0]
    elif fault=='fingerprint':value['manifest_fingerprint']='0'*64
    elif fault=='source':ident['manifest_sha256']=digest('newsource')
    elif fault=='new_demo':rows.append({**rows[0],'episode_id':'new-demo','lineage_id':'new-demo','sha256':digest('new-demo'),'lineage_aliases':[],'source_hashes':{}})
    elif fault=='config':cfg['strata'][0]['match']={'tag':'new'}
    path=tmp_path/'split.json';path.write_text(json.dumps(value))
    with pytest.raises(ValueError):module.read_split_manifest(path,rows,cfg,ident)


def test_immutable_writer_never_replaces_other_split_or_writes_in_source(tmp_path):
    module=api();rows=episodes();cfg=config()
    value=module.build_split_manifest(module.make_episode_split(rows,cfg),rows,cfg,20261006,identity())
    source=tmp_path/'source';source.mkdir();path=tmp_path/'split.json';path.write_text('existing frozen artifact')
    with pytest.raises(ValueError):module.write_split_manifest(value,path,source_root=source)
    assert path.read_text()=='existing frozen artifact'
    with pytest.raises(ValueError):module.write_split_manifest(value,source/'split.json',source_root=source)
    assert not (source/'split.json').exists()


def test_reader_rejects_selfconsistent_replacement_for_different_seed(tmp_path):
    module=api();rows=episodes();cfg=config();seed=20261007
    alternate=module.build_split_manifest(module.make_episode_split(rows,cfg,seed),rows,cfg,seed,identity())
    assert alternate['splits']!={k:list(v) for k,v in module.make_episode_split(rows,cfg).items()}
    path=tmp_path/'alternate.json';path.write_bytes(module.serialize_split_manifest(alternate))
    with pytest.raises(ValueError,match='seed'):
        module.read_split_manifest(path,rows,cfg,identity())


def test_reader_uses_explicit_seed_and_independently_pinned_fingerprint(tmp_path):
    module=api();parameters=inspect.signature(module.read_split_manifest).parameters
    assert {'expected_seed','expected_fingerprint'}<=set(parameters), 'Missing independent freeze expectations'
    rows=episodes();cfg=config();seed=20261007
    value=module.build_split_manifest(module.make_episode_split(rows,cfg,seed),rows,cfg,seed,identity())
    path=tmp_path/'split.json';path.write_bytes(module.serialize_split_manifest(value))
    assert module.read_split_manifest(path,rows,cfg,identity(),expected_seed=seed,
        expected_fingerprint=value['manifest_fingerprint'])==value
    with pytest.raises(ValueError,match='fingerprint'):
        module.read_split_manifest(path,rows,cfg,identity(),expected_seed=seed,expected_fingerprint='0'*64)


def test_diagnostic_audit_uses_supervision_counts_and_does_not_reassign():
    module=api();rows,cfg=small((1,1,1),(1,1,1))
    for i,e in enumerate(rows):e.update(kind='origin',supervised_instances={'rare':i+1},bc_valid_transitions=10+i)
    s=module.make_episode_split(rows,cfg);before=copy.deepcopy(s)
    audit=module.audit_split(s,rows,cfg,action_families=('rare','zero'))
    assert s==before
    assert sum(a['bc_valid_transitions'] for a in audit.values())==33
    assert sum(a['supervised_instances']['rare'] for a in audit.values())==6
    assert all(a['supervised_instances']['zero']==0 for a in audit.values())
    assert all(a['episodes']==1 and a['kind_counts']=={'origin':1} for a in audit.values())


def test_stratification_metadata_does_not_become_policy_features():
    module=api();module.make_episode_split(episodes(),config())
    from drl.IL.bc_v1 import bc_v1_contract as c
    from drl.IL.bc_v1.policy_inputs import select_policy_inputs
    from drl.IL.tests.bc_v1.fixtures import state_graph
    g=state_graph(2,1);ctx=c.TaskContext('button');o=c.ObservationBundle()
    before=select_policy_inputs(g,o,ctx)
    for key in ('stratum','lineage_id','training_split','source_morphology','target_morphology'):
        object.__setattr__(ctx,key,'offline-only')
    assert select_policy_inputs(g,o,ctx)==before


def test_split_core_import_and_execution_are_stdlib_only():
    api()
    script='''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'numpy','torch','rclpy','isaacsim','isaaclab','omni','smores_ep','mssr_expert'} or fullname in {'drl.IL.bc_v1.smores_adapter','drl.IL.bc_v1.features','drl.IL.bc_v1.policy_inputs','drl.IL.dataset_loader'}:
   raise ImportError(fullname)
sys.meta_path.insert(0,Block())
from drl.IL.bc_v1.split import make_episode_split
rows=[{'episode_id':str(i),'lineage_id':str(i),'sha256':format(i,'064x'),'eligible':1,'tag':'unrelated'} for i in range(3)]
cfg={'schema_version':'mssr.bc_v1.strata.v1','eligible_count_field':'eligible','strata':[{'name':'any','match':{'tag':'unrelated'},'total':3,'quotas':{'train':1,'validation':1,'test':1}}]}
assert {k:len(v) for k,v in make_episode_split(rows,cfg).items()}=={'train':1,'validation':1,'test':1}
'''
    r=subprocess.run([sys.executable,'-B','-c',script],cwd=ROOT,capture_output=True,text=True)
    assert r.returncode==0,r.stderr


def test_official_v6_artifact_matches_exact_source_quotas_and_frozen_membership():
    require_canonical_v6_fixture(ROOT)
    module=api();training=json.loads((ROOT/'configs/bc_v1/training.json').read_text())
    inventory=json.loads((ARTIFACT/'episodes.json').read_text());rows=inventory['episodes'];cfg=training['split']['strata_config']
    manifest=json.loads((ROOT/'datasets/expert_v1_canonical_v6/manifest.json').read_text())
    assert len(rows)==80 and sum(e['bc_valid_transitions']>0 for e in rows)==73
    assert {e['episode_id'] for e in rows}=={e['episode_id'] for e in manifest['episodes']}
    eligible={e['episode_id'] for e in rows if e['bc_valid_transitions']>0}
    assert sum(e['bc_valid_transitions'] for e in rows)==119600
    assert 'manifest_fingerprint' in training['split'], 'Missing official fingerprint pin'
    value=module.read_split_manifest(ARTIFACT/'split.json',rows,cfg,inventory['dataset_identity'],
        expected_seed=training['split']['seed'],expected_fingerprint=training['split']['manifest_fingerprint'])
    assert {k:len(v) for k,v in value['splits'].items()}==dict(train=49,validation=12,test=12)
    assert set().union(*map(set,value['splits'].values()))==eligible
    for name,q in QUOTAS.items():assert tuple(value['counts_by_stratum'][name][p] for p in PARTS)==q
    assembly={f'expert:assembly:seed-{s:06d}' for s in (6217,6284,6307)}
    assert all(len(assembly.intersection(value['splits'][p]))==1 for p in PARTS)
    assert module.make_episode_split(rows[::-1],cfg,training['split']['seed'])=={p:tuple(value['splits'][p]) for p in PARTS}
    raw=(ROOT/'datasets/expert_v1_canonical_v6/manifest.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='31765272b6ecd7506c54b25e214b75d7b1b28691803ddfc286dafcce8b4ec562'
