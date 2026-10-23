"""Real TRAIN/VAL source authority; no TEST factory or payload access."""
from pathlib import Path
import importlib.util
import json
import copy
import pytest
from drl.IL.bc_v1.bc_v1_contract import load_robot_spec
from drl.IL.bc_v1.sidecar import SidecarDataset
ROOT=Path(__file__).resolve().parents[4]


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.smores_resource_labels'), 'Missing authoritative resource labels'
    from drl.IL.bc_v1 import smores_resource_labels
    return smores_resource_labels


@pytest.mark.parametrize('split,index',[('train',3064),('validation',790)])
def test_real_rc8_complete_navigation_snapshot_certifies_sparse_inactive(split,index):
    a=api();from drl.IL.bc_v1.resource_decisions import load_resource_contract
    spec=load_robot_spec(ROOT/'configs/bc_v1/smores_ep.robot.json')
    contract=load_resource_contract(ROOT/'configs/bc_v1/resource_decisions_v1.json',spec)
    source=ROOT/'artifacts/bc_v1/sidecar/g2_v2_no_structural';doc=json.loads((source/'manifest.json').read_text())
    with SidecarDataset(source,split,expected_identity=doc['identity']) as data:sample=data[index]
    new=a.SmoresResourceLabelProvider(ROOT,contract)(sample)
    y=new['supervision']['resource'];x=new['inputs']['resource_candidates']
    kinds=x['outcome_kind'][y['decision_candidate']].tolist()
    assert y['decision_supervised'].tolist()==[True]*16
    assert kinds.count(0)==8 and kinds.count(1)==4 and kinds.count(2)==4
    assert len(y['payload'])==4
    assert all(new['routing']['resource_label_authority']['evidence'])


def test_invalid_ambiguous_or_native_snapshot_never_certifies_absence():
    a=api();source={'active_primitive':'nav2_planar_route','action_valid':True,
        'expert_action':{'locomotion':{'m':{'vx':0.,'vy':0.,'yaw_rate':0.,'pan_rate_rad_s':1.}},'magnetic':[],'primitive_goal':None},
        'supervision':{'valid_for_behavior_cloning':True},'debug':{'behavior':'nav2_planar_route'}}
    assert a.complete_navigation_snapshot(source)
    for key,value in [('action_valid',False),('active_primitive','align_faces')]:
        changed=copy.deepcopy(source);changed[key]=value;assert not a.complete_navigation_snapshot(changed)
    changed=copy.deepcopy(source);changed['expert_action']['primitive_goal']={'command':'dock'}
    assert not a.complete_navigation_snapshot(changed)
    changed=copy.deepcopy(source);del changed['expert_action']['locomotion']['m']['pan_rate_rad_s']
    assert not a.complete_navigation_snapshot(changed)


def test_smores_catalog_compatibility_fixed_stop_tilt_and_inactive():
    api();from drl.IL.bc_v1.resource_decisions import load_resource_contract
    spec=load_robot_spec(ROOT/'configs/bc_v1/smores_ep.robot.json')
    c=load_resource_contract(ROOT/'configs/bc_v1/resource_decisions_v1.json',spec)
    def o(name):return next(o for o in c.outcomes if o.outcome_id==name)
    assert not c.compatible([(o('wheel_velocity'),(1.,2.)),(o('pan_velocity'),(3.,))])
    assert c.compatible([(o('WHEEL_STOP_FIXED'),(0.,0.)),(o('pan_velocity'),(3.,))])
    assert c.compatible([(o('wheel_velocity'),(1.,2.)),(o('tilt_target'),(3.,))])
    assert c.compatible([(o('INACTIVE'),()),(o('pan_velocity'),(3.,))])


def test_changed_source_same_stamp_cannot_recertify_inactive(tmp_path):
    import hashlib
    from dataclasses import replace
    from drl.IL.tests.bc_v1.g2_fixtures import fixture,tensorize
    from drl.IL.tests.bc_v1.test_resource_decisions import contract
    a=api();s,kw=fixture();sample=tensorize(s,kw)
    source=tmp_path/'source.jsonl'
    record={'stamp':1.,'active_primitive':'nav2_planar_route','action_valid':True,
        'supervision':{'valid_for_behavior_cloning':True},'debug':{'behavior':'nav2_planar_route'},
        'expert_action':{'locomotion':{'unit-0':{'vx':0.,'vy':0.,'yaw_rate':0.,'pan_rate_rad_s':1.}},'magnetic':[],'primitive_goal':None}}
    source.write_text(json.dumps(record)+'\n');expected=hashlib.sha256(source.read_bytes()).hexdigest()
    cfg=tmp_path/'configs/bc_v1/resource_prepare_v3.json';cfg.parent.mkdir(parents=True)
    cfg.write_text(json.dumps({'authority_sources':{'source.jsonl':expected}}))
    row={'provenance':{'logical_episode_id':s.audit.episode_id,'canonical_transition_index':0,'physical_stamp':1.,
        'source_records':[{'_loader_source_path':str(source),'_loader_source_row':1}]},
        'action_t':{'modules':{}},'supervision':{}}
    folder=tmp_path/'datasets/expert_v1_canonical_v6';folder.mkdir(parents=True)
    path=folder/'episode.jsonl';path.write_text(json.dumps(row)+'\n')
    (folder/'manifest.json').write_text(json.dumps({'episodes':[{'episode_id':s.audit.episode_id,'file':'episode.jsonl',
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}]}))
    provider=a.SmoresResourceLabelProvider(tmp_path,contract(kw))
    provider(sample) # pinned original source accepted
    record['expert_action']['locomotion']={'another-module':{'vx':0.,'vy':0.,'yaw_rate':0.,'pan_rate_rad_s':1.}}
    source.write_text(json.dumps(record)+'\n')
    with pytest.raises(ValueError,match='Source authority hash mismatch'):provider(sample)
