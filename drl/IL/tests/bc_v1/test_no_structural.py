"""L1c.2 decision abstention; synthetic G2, no official dataset access."""
from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import torch
import pytest
from drl.IL.tests.bc_v1.g2_fixtures import fixture,tensorize
from drl.IL.bc_v1 import bc_v1_contract as c
ROOT=Path(__file__).resolve().parents[4]

@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)

def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.structural_decisions'), 'Missing per-initiator NO_STRUCTURAL contract'
    from drl.IL.bc_v1 import structural_decisions
    return structural_decisions

def sample(n=8,positive=()):
    s,kw=fixture(n)
    targets=[]
    for i in positive:
        mid=s.inputs.policy.graph.modules[i].ref
        a,b=next((r.action,b) for r in s.inputs.candidates.structural_candidates for b in r.bindings if r.action.action_id=='align_faces' and b.initiator==mid)
        targets.append(c.StructuralTarget(mid,a.action_id,tuple(p.module for p in b.connectors),b.connectors,(c.ParameterValue('turn','quarter'),)))
    s=replace(s,supervision=replace(s.supervision,structural_targets=tuple(targets)))
    return tensorize(s,kw),kw

@pytest.mark.parametrize('positive',[(),(2,),(2,6)])
def test_one_decision_per_initiator_preserves_only_real_positive_initiators(positive):
    a=api();b,kw=sample(positive=positive);old=b['supervision']['structural']['candidate'].clone()
    new=a.add_no_structural_supervision(b);y=new['supervision']['structural']
    assert y['decision_group'].tolist()==list(range(8))
    assert y['decision_supervised'].tolist()==[True]*8
    assert int((y['decision_candidate']>=0).sum())==len(positive)
    assert [i for i,q in enumerate(y['decision_candidate']) if q>=0]==list(positive)
    assert torch.equal(y['candidate'],old)
    assert new['inputs'] is b['inputs']
    assert b['supervision']['resource']['supervised'].any()  # nonzero resource retained
    assert 'decision_group' not in b['supervision']['structural']

def test_duplicate_positive_same_initiator_fails_closed():
    a=api();b,_=sample(positive=(2,));y=b['supervision']['structural']
    for k in ('candidate','group','supervised'):y[k]=y[k].repeat(2)
    with pytest.raises(ValueError,match='Duplicate'):a.add_no_structural_supervision(b)

def test_collation_offsets_positive_bindings_but_preserves_none_sentinel():
    from drl.IL.bc_v1.batching import collate_policy_samples
    a=api();b,_=sample(positive=(2,));b=a.add_no_structural_supervision(b)
    n=len(b['inputs']['structural']['initiator']);joint=collate_policy_samples([b,b]);y=joint['supervision']['structural']
    assert y['decision_group'].tolist()==list(range(16))
    assert y['decision_candidate'][10]==y['decision_candidate'][2]+n
    assert int((y['decision_candidate']==-1).sum())==14

def test_v2_head_predicts_one_learnable_none_logit_per_module():
    from drl.IL.bc_v1.action_heads import ActionHeads
    from drl.IL.bc_v1.model import load_model_config
    from drl.IL.tests.bc_v1.test_model import model_for
    b,kw=sample();h1=load_model_config(ROOT/'configs/bc_v1/gnn_v1.json')
    heads=ActionHeads(kw['spec'],h1,dict(schema_version='mssr.bc_v1.action_heads.v2',hidden_dim=128,activation='silu',dropout=.1))
    p=heads(model_for(kw).encode(b['inputs']),b['inputs'])
    assert p['structural']['no_structural_logits'].shape==(8,)
    assert p['structural']['no_structural_logits'].requires_grad
    assert 'no_structural' not in {a.action_id for a in kw['spec'].actions}

def test_joint_ce_none_provides_nonzero_learning_signal_and_singleton_is_safe():
    from drl.IL.bc_v1.losses import structural_choice_loss
    scores=torch.zeros(2,requires_grad=True);none=torch.zeros(2,requires_grad=True)
    p={'structural':{'candidate_logits':scores,'no_structural_logits':none,'group_ptr':torch.tensor([0,2,2])}}
    y={'candidate':torch.tensor([],dtype=torch.int64),'group':torch.tensor([],dtype=torch.int64),'supervised':torch.tensor([],dtype=torch.bool),
       'decision_candidate':torch.tensor([-1,-1]),'decision_group':torch.tensor([0,1]),'decision_supervised':torch.tensor([True,True])}
    result=structural_choice_loss(p,{'supervision':{'structural':y}})
    assert result['support']==2
    assert float(result['mean'].detach())==pytest.approx(math.log(3)/2)
    result['mean'].backward()
    assert none.grad[0]<0 and (scores.grad>0).all() and none.grad[1]==0

def test_decoder_none_wins_with_many_physical_candidates_without_physical_choice():
    from drl.IL.tests.bc_v1.test_action_decoder import example,decode
    case=list(example());p,b=case[:2]
    p['structural']['no_structural_logits']=torch.full((3,),1000.)
    av=case[3][0];case[3]=(replace(av,resource_mask=(False,)*6,structural_mask=(True,)*len(b['routing']['structural_bindings'])),)
    assert decode(case)==()

def test_structural_metrics_detect_false_and_missed_activation():
    a=api();stats=a.StructuralDecisionMetrics(('align_faces','dock','undock','gravity_settle'))
    for target,prediction,exact in [(None,None,True),(None,'undock',False),('align_faces',None,False),('dock','dock',True)]:stats.add(target,prediction,exact)
    r=stats.compute()
    assert r['no_structural']['support']==2 and r['no_structural']['recall']==.5
    assert r['positive']['support']==2 and r['positive']['recall']==.5
    assert r['false_activation_rate']==.5 and r['missed_activation_rate']==.5
    assert r['by_operation']['align_faces']['activation_recall']==0.


def v2_model(kw):
    from drl.IL.bc_v1.model import BCGNN,load_model_config
    return BCGNN(kw['spec'],kw['feature_schema'],load_model_config(ROOT/'configs/bc_v1/gnn_v1.json'),
        action_head_config=dict(schema_version='mssr.bc_v1.action_heads.v2',hidden_dim=128,activation='silu',dropout=.1)).eval()


def test_streaming_metrics_include_dense_decisions_and_count_none_as_positive_miss():
    from drl.IL.bc_v1.metrics import BCMetrics,load_metrics_config
    b,kw=sample(positive=(2,6));b=api().add_no_structural_supervision(b)
    p=v2_model(kw)(b['inputs']);p['structural']['no_structural_logits']=torch.full((8,),1e6)
    m=BCMetrics(kw['spec'],kw['normalizers'],load_metrics_config(ROOT/'configs/bc_v1/metrics_v1.json'))
    m.update(p,b);result=m.compute()['structural']
    assert result['decisions']['no_structural']['support']==6
    assert result['decisions']['missed_activation_rate']==1.
    assert result['candidate_top1']['value']==0.
    assert result['decisions']['positive']['accuracy']==0.


def test_task_changes_learned_logits_without_changing_physical_domain():
    import copy
    b,kw=sample(positive=());other=copy.deepcopy(b)
    other['inputs']['task_one_hot'].zero_();other['inputs']['task_one_hot'][0,-1]=1.
    model=v2_model(kw);x=model(b['inputs'])['structural'];y=model(other['inputs'])['structural']
    assert torch.equal(b['inputs']['structural']['action'],other['inputs']['structural']['action'])
    assert not torch.equal(x['no_structural_logits'],y['no_structural_logits'])
    assert not torch.equal(x['candidate_logits'],y['candidate_logits'])


def test_module_permutation_preserves_none_equivariance_and_id_renaming_is_routing_only():
    from dataclasses import fields,is_dataclass
    from drl.IL.tests.bc_v1.test_model import permuted_sample
    original,kw=fixture(8);permuted,order=permuted_sample(original,19)
    model=v2_model(kw)
    before=model(tensorize(original,kw)['inputs'])['structural']['no_structural_logits']
    after=model(tensorize(permuted,kw)['inputs'])['structural']['no_structural_logits']
    torch.testing.assert_close(after,before[order['modules']],rtol=3e-6,atol=3e-6)
    def rename(v):
        if isinstance(v,c.ModuleRef):return c.ModuleRef('renamed-'+v.module_id)
        if is_dataclass(v):return replace(v,**{f.name:rename(getattr(v,f.name)) for f in fields(v)})
        if isinstance(v,tuple):return tuple(rename(x) for x in v)
        if isinstance(v,frozenset):return frozenset(rename(x) for x in v)
        return v
    renamed=model(tensorize(rename(original),kw)['inputs'])['structural']['no_structural_logits']
    torch.testing.assert_close(renamed,before,rtol=0,atol=0)


def test_resource_and_physical_structural_can_coexist_on_same_module_with_disjoint_claims():
    from drl.IL.tests.bc_v1.test_action_decoder import example
    from drl.IL.bc_v1.action_decoder import decode_actions
    p,b,kw,available,semantics=example()
    q=next(i for i,(a,binding) in enumerate(b['routing']['structural_bindings']) if a.action_id=='align_faces' and binding.initiator==c.ModuleRef('unit-0'))
    p['structural']['candidate_logits'][q]=10.;p['structural']['no_structural_logits']=torch.zeros(3)
    av=(replace(available[0],structural_mask=tuple(i==q for i in range(len(b['routing']['structural_bindings'])))),)
    out=decode_actions(p,b['inputs'],b['routing'],spec=kw['spec'],normalizers=kw['normalizers'],availability=av,
        execution_semantics=semantics,configured_arguments={},claim_resolver=lambda choice:(choice.resource,) if choice.resource else choice.connectors)
    assert len(out)==4 and len([r for r in out if r.choice.initiator==c.ModuleRef('unit-0')])==2


def test_zero_physical_candidates_has_only_none_finite_loss_and_empty_dispatch_domain():
    from drl.IL.bc_v1.losses import structural_choice_loss
    s,kw=fixture(3);s=replace(s,inputs=replace(s.inputs,candidates=replace(s.inputs.candidates,structural_candidates=())))
    b=api().add_no_structural_supervision(tensorize(s,kw));p=v2_model(kw)(b['inputs'])
    result=structural_choice_loss(p,b)
    assert result['support']==3 and result['mean']==0. and torch.isfinite(result['mean'])
    assert p['structural']['candidate_logits'].numel()==0


def test_v2_has_no_forbidden_learned_input_channels():
    b,_=sample();v2=api().add_no_structural_supervision(b)
    assert v2['inputs'] is b['inputs'] and 'task_one_hot' in v2['inputs']
    banned=('stage','fsm','control_state','assembly','reconfigure','configuration_goal','expert','future','success','reward')
    def paths(value,prefix=''):
        for key,item in value.items():
            path=prefix+'/'+key
            yield path
            if isinstance(item,dict):yield from paths(item,path)
    assert not any(word in key for key in paths(v2['inputs']) for word in banned)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_dense_multi_sample_joint_ce_is_cuda_safe():
    from drl.IL.bc_v1.batching import collate_policy_samples
    from drl.IL.bc_v1.training import move_batch_to_device
    from drl.IL.bc_v1.losses import structural_choice_loss
    b,kw=sample(positive=(2,6));b=api().add_no_structural_supervision(b)
    joint=move_batch_to_device(collate_policy_samples([b,b]),'cuda:0');model=v2_model(kw).to('cuda:0')
    p=model(joint['inputs']);loss=structural_choice_loss(p,joint)
    assert loss['support']==16 and torch.isfinite(loss['mean'])
    loss['mean'].backward()
    assert all(torch.isfinite(v.grad).all() for v in model.action_heads.no_structural_scorer.parameters())


def test_joint_targets_cannot_silently_drop_a_preserved_positive():
    from drl.IL.bc_v1.losses import structural_choice_loss
    b,kw=sample(positive=(2,));b=api().add_no_structural_supervision(b)
    b['supervision']['structural']['decision_candidate'][2]=-1
    p=v2_model(kw)(b['inputs'])
    with pytest.raises(ValueError,match='positive|mapping'):structural_choice_loss(p,b)


def test_v2_supervision_rejects_legacy_head_without_none_logits():
    from drl.IL.bc_v1.losses import structural_choice_loss
    from drl.IL.tests.bc_v1.test_action_heads import complete_model
    b,kw=sample();b=api().add_no_structural_supervision(b)
    with pytest.raises(ValueError,match='v2|NO_STRUCTURAL'):structural_choice_loss(complete_model(kw)(b['inputs']),b)
