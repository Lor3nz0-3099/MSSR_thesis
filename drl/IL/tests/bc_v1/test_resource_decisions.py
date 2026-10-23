"""Resource v3 contract: absence needs authority; exact zeros never regress."""
import importlib.util
import copy
from dataclasses import replace
import pytest
import torch
from drl.IL.tests.bc_v1.g2_fixtures import fixture, tensorize
from drl.IL.bc_v1.bc_v1_contract import robot_spec_fingerprint


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.resource_decisions'), 'Resource v3 contract missing'
    from drl.IL.bc_v1 import resource_decisions
    return resource_decisions


def contract(kw):
    return api().ResourceDecisionContract.from_dict({'schema_version':'mssr.bc_v1.resource_decisions.v1',
        'revision':'test-1','robot_spec_fingerprint':robot_spec_fingerprint(kw['spec']),
        'fixed':[{'module_type_id':'kit','resource_id':'rotor','outcome_id':'stop','mode_id':'pace',
            'payload':[0.,0.],'evidence':['explicit velocity stop']}], 'tags':[], 'forbidden_tag_pairs':[]},kw['spec'])


def test_authoritative_inactive_fixed_parameterized_and_unknown():
    a=api();s,kw=fixture();old=tensorize(s,kw);snap=copy.deepcopy(old)
    new=a.add_resource_decisions(old,contract(kw),authoritative_resources={1})
    y=new['supervision']['resource'];x=new['inputs']['resource_candidates']
    assert y['decision_supervised'].tolist()==[True,True,False]
    assert y['decision_candidate'][2]==-1
    ids=y['decision_candidate'][:2].tolist()
    assert x['outcome_kind'][ids].tolist()==[2,0]
    assert y['supervised'].tolist()==[True] and y['payload'].tolist()==[2.,3.]
    assert new['supervision']['structural'] is old['supervision']['structural']
    from drl.IL.tests.bc_v1.test_tensorization import assert_tree_equal
    assert_tree_equal(old,snap)


def test_fixed_stop_has_no_payload_target_zero_pose_remains_parameterized():
    a=api();s,kw=fixture();target=s.supervision.resource_targets[0]
    s=replace(s,supervision=replace(s.supervision,resource_targets=(replace(target,payload=(0.,0.)),)))
    out=a.add_resource_decisions(tensorize(s,kw),contract(kw));y=out['supervision']['resource']
    assert y['payload'].numel()==0 and y['candidate'].numel()==0
    assert out['inputs']['resource_candidates']['outcome_kind'][y['decision_candidate'][0]]==1
    s=replace(s,supervision=replace(s.supervision,resource_targets=(replace(target,mode_id='aim',payload=(0.,)),)))
    out=a.add_resource_decisions(tensorize(s,kw),contract(kw));y=out['supervision']['resource']
    assert out['inputs']['resource_candidates']['outcome_kind'][y['decision_candidate'][0]]==2
    assert len(y['payload'])==1


def test_unsupervised_existing_row_is_not_inactive():
    a=api();s,kw=fixture();old=tensorize(s,kw);old['supervision']['resource']['supervised'][0]=False
    new=a.add_resource_decisions(old,contract(kw),authoritative_resources={0})
    assert new['supervision']['resource']['decision_supervised'].tolist()==[False,False,False]


def test_joint_selection_is_exact_and_supports_constraints_and_empty_domains():
    a=api();domains=[[0,1,2],[3,4]];scores={0:0.,1:5.,2:4.,3:0.,4:6.}
    assert a.select_joint(domains,scores,lambda rows: not(1 in rows and 4 in rows))==(2,4)
    assert a.select_joint(domains,scores,lambda rows:True)==(1,4)
    assert a.select_joint([[0],[3]],scores,lambda rows:True)==(0,3)
    with pytest.raises(ValueError):a.select_joint([[]],scores,lambda rows:True)
    with pytest.raises(ValueError):a.select_joint(domains,scores,lambda rows:False)


def test_contract_rejects_foreign_spec_and_invalid_fixed_payload():
    a=api();s,kw=fixture();c=contract(kw);doc=copy.deepcopy(c.document)
    doc['robot_spec_fingerprint']='foreign'
    with pytest.raises(ValueError):a.ResourceDecisionContract.from_dict(doc,kw['spec'])
    doc=copy.deepcopy(c.document);doc['fixed'][0]['payload']=[0.]
    with pytest.raises(ValueError):a.ResourceDecisionContract.from_dict(doc,kw['spec'])


def test_compatibility_uses_compiled_catalog_without_json_parsing(monkeypatch):
    a=api();s,kw=fixture();c=contract(kw);o=next(o for o in c.outcomes if o.kind==2)
    calls=[];original=a.json.loads
    def tracked(*args,**kwargs):calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(a.json,'loads',tracked)
    assert c.compatible([(o,(1.,2.))]) and c.compatible([(o,(0.,0.))])
    assert calls==[], 'Catalog JSON must be parsed at composition, not every assignment'
