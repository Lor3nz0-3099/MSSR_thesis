"""I1 numerical and gradient contracts; real G2/H2 routing, no optimizer."""
import copy
import importlib
import importlib.util
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.bc_v1.batching import collate_policy_samples
from drl.IL.tests.bc_v1.g2_fixtures import fixture,tensorize,with_structural
from drl.IL.tests.bc_v1.test_action_heads import complete_model,enriched_fixture

ROOT=Path(__file__).resolve().parents[4]


@pytest.fixture(autouse=True,scope='module')
def thread_budget():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.losses'), 'Missing I1 losses'
    return importlib.import_module('drl.IL.bc_v1.losses')


def example(n=3,*,skill=False,structural=False):
    sample,kw=fixture(n)
    if structural:sample=with_structural(sample,skill=skill)
    batch=tensorize(sample,kw);model=complete_model(kw)
    return model(batch['inputs']),batch,model


def test_masked_statistics_sum_mean_support_and_padding_gradient():
    values=torch.tensor([2.,9999.,6.,float('nan')],requires_grad=True)
    stats=api().masked_statistics(values,torch.tensor([True,False,True,False]))
    assert stats['sum']==8 and stats['mean']==4 and stats['support']==2
    assert stats['support'].dtype==torch.int64 and not stats['support'].requires_grad
    stats['mean'].backward();torch.testing.assert_close(values.grad,torch.tensor([.5,0.,.5,0.]))


@pytest.mark.parametrize('values',[[],[float('nan'),1000.]])
def test_zero_support_is_exact_finite_differentiable_zero(values):
    x=torch.tensor(values,dtype=torch.float32,requires_grad=True)
    stats=api().masked_statistics(x,torch.zeros(len(x),dtype=torch.bool))
    assert stats['support']==0 and stats['sum']==0 and stats['mean']==0
    stats['mean'].backward();assert torch.equal(x.grad,torch.zeros_like(x))


def test_support_statistics_aggregate_by_sum_not_batch_means():
    x=torch.tensor([1.,3.,7.,99.],requires_grad=True);mask=torch.tensor([True,True,True,False])
    full=api().masked_statistics(x,mask);a=api().masked_statistics(x[:1],mask[:1]);b=api().masked_statistics(x[1:],mask[1:])
    assert (a['sum']+b['sum'])/(a['support']+b['support'])==full['mean']
    assert (a['mean']+b['mean'])/2!=full['mean']


def test_statistics_reject_invalid_mask_or_valid_nonfinite_loss():
    with pytest.raises(ValueError):api().masked_statistics(torch.ones(2),torch.ones(2))
    with pytest.raises(ValueError):api().masked_statistics(torch.tensor([float('nan')]),torch.tensor([True]))


def recatalog(sample,kw,spec):
    from drl.IL.bc_v1.candidates import enumerate_candidates
    kw={**kw,'spec':spec,'normalizers':replace(kw['normalizers'],metadata_items=tuple(sorted({
        **kw['normalizers'].metadata,'robot_spec_fingerprint':c.robot_spec_fingerprint(spec)}.items())))}
    return replace(sample,inputs=replace(sample.inputs,candidates=enumerate_candidates(sample.inputs.policy.graph,spec,kw['feature_schema']))),kw


def test_resource_mode_ce_separate_groups_with_hand_computed_value_and_gradients():
    p,b,_=example();p['resource']['mode_logits']=torch.tensor([0.,math.log(3.),1000.,-1000.,0.,0.],requires_grad=True)
    out=api().resource_mode_loss(p,b)
    assert out['support']==1;assert float(out['mean'].detach())==pytest.approx(math.log(4.))
    out['mean'].backward();torch.testing.assert_close(p['resource']['mode_logits'].grad,torch.tensor([-.75,.75,0.,0.,0.,0.]))
    # Add an independent supervised resource: second CE is log(2).
    r=b['supervision']['resource'];r['resource']=torch.tensor([0,2]);r['candidate']=torch.tensor([0,4]);r['supervised']=torch.tensor([True,True])
    out=api().resource_mode_loss(p,b)
    assert out['support']==2 and float(out['sum'].detach())==pytest.approx(math.log(4.)+math.log(2.))


@pytest.mark.parametrize('magnitude',[100.,1000.])
def test_resource_mode_large_logits_are_stable(magnitude):
    p,b,_=example();p['resource']['mode_logits']=torch.tensor([-magnitude,magnitude,0.,0.,0.,0.],requires_grad=True)
    out=api().resource_mode_loss(p,b);assert float(out['mean'].detach())==pytest.approx(2*magnitude)
    out['mean'].backward();assert torch.isfinite(p['resource']['mode_logits'].grad).all()


def test_singleton_resource_and_unsupervised_resource_have_zero_mode_support():
    sample,kw=fixture();spec=kw['spec'];mt=spec.module_types[0];resource=mt.resources[0]
    sample,kw=recatalog(sample,kw,replace(spec,module_types=(replace(mt,resources=(replace(resource,modes=resource.modes[:1]),)),)))
    b=tensorize(sample,kw);p=complete_model(kw)(b['inputs']);out=api().resource_mode_loss(p,b)
    assert out['support']==0 and out['mean']==0
    p,b,_=example();b['supervision']['resource']['supervised'].fill_(False)
    out=api().resource_mode_loss(p,b);assert out['support']==0 and out['mean']==0


def test_resource_mode_rejects_wrong_group_and_duplicate_decision():
    p,b,_=example();b['supervision']['resource']['candidate'][0]=2
    with pytest.raises(ValueError):api().resource_mode_loss(p,b)
    p,b,_=example();r=b['supervision']['resource']
    for key in ('resource','candidate','supervised'):r[key]=r[key].repeat(2)
    with pytest.raises(ValueError):api().resource_mode_loss(p,b)


def test_payload_huber_is_mean_within_target_then_mean_across_targets():
    p,b,_=example();values=torch.tensor([2.5,5.,1000.,1.,3.,1000.,0.,0.,0.],requires_grad=True)
    p['resource']['payload']=values
    out=api().resource_payload_loss(p,b)
    assert out['support']==1 and float(out['mean'].detach())==pytest.approx(.8125)
    r=b['supervision']['resource'];r.update(resource=torch.tensor([0,1]),candidate=torch.tensor([0,2]),
        supervised=torch.tensor([True,True]),payload=torch.tensor([2.,3.,0.,0.]),
        payload_ptr=torch.tensor([0,2,4]),payload_category=torch.full((4,),-1),payload_valid=torch.ones(4,dtype=torch.bool))
    out=api().resource_payload_loss(p,b)
    assert out['support']==2 and float(out['mean'].detach())==pytest.approx(1.15625)
    out['mean'].backward();assert values.grad[2]==0 and values.grad[5]==0
    assert torch.isfinite(values.grad).all()


def test_payload_component_masks_ignore_extreme_and_nonfinite_padding():
    p,b,_=example();p['resource']['payload']=torch.tensor([2.5,float('nan'),999.,0.,0.,0.,0.,0.,0.],requires_grad=True)
    b['supervision']['resource']['payload_valid'][1]=False;b['supervision']['resource']['payload'][1]=float('nan')
    out=api().resource_payload_loss(p,b);assert float(out['mean'].detach())==pytest.approx(.125)
    out['mean'].backward();assert p['resource']['payload'].grad[1]==0
    b['supervision']['resource']['payload_valid'].fill_(False)
    out=api().resource_payload_loss(p,b);assert out['support']==0 and out['mean']==0


def test_unsupervised_payload_never_creates_zero_targets_for_other_modes():
    p,b,_=example();out=api().resource_payload_loss(p,b);assert out['support']==1
    b['supervision']['resource']['supervised'].fill_(False)
    out=api().resource_payload_loss(p,b);assert out['support']==0 and out['mean']==0


def categorical_resource_example():
    sample,kw=fixture();spec=kw['spec'];mt=spec.module_types[0];res=mt.resources[0]
    modes=tuple(replace(m,codec='categorical',units=('1',),categories=('low','mid','high')) if m.mode_id=='aim' else m for m in res.modes)
    actions=tuple(replace(a,parameters=(c.ParameterSpec('level','category',categories=('low','mid','high'),
        module_type_id='kit',resource_id='rotor'),)) if a.action_id=='aim' else a for a in spec.actions)
    sample,kw=recatalog(sample,kw,replace(spec,module_types=(replace(mt,resources=(replace(res,modes=modes),)),),actions=actions))
    kw={**kw,'normalizers':replace(kw['normalizers'],resource_bindings=tuple(replace(r,codec='categorical',keys=(),units=('1',)) if r.mode_id=='aim' else r for r in kw['normalizers'].resource_bindings))}
    sample=replace(sample,supervision=c.SupervisionBundle((c.ResourceTarget(sample.inputs.policy.graph.resources[0].ref,'aim',(1.,)),)))
    batch=tensorize(sample,kw);return complete_model(kw)(batch['inputs']),batch


def test_categorical_resource_payload_uses_ce_once_without_fake_numeric_target():
    p,b=categorical_resource_example();p['resource']['categorical']['logits']=torch.zeros(9,requires_grad=True)
    out=api().resource_payload_loss(p,b)
    assert out['support']==1 and float(out['mean'].detach())==pytest.approx(math.log(3.))
    p['resource']['payload'].retain_grad();out['mean'].backward()
    assert p['resource']['payload'].grad is None
    assert p['resource']['categorical']['logits'].grad[1]<0


def test_structural_group_ce_counts_two_initiators_not_34_candidate_rows():
    p,b,_=example(structural=True);p['structural']['candidate_logits']=torch.zeros(51,requires_grad=True)
    out=api().structural_choice_loss(p,b)
    assert out['support']==2 and float(out['mean'].detach())==pytest.approx(math.log(17.))
    out['mean'].backward();assert torch.equal(p['structural']['candidate_logits'].grad[34:],torch.zeros(17))
    assert torch.isfinite(p['structural']['candidate_logits'].grad).all()


def test_structural_large_logits_and_masked_target_are_stable():
    p,b,_=example(structural=True);scores=torch.full((51,),1000.);scores[b['supervision']['structural']['candidate']]=-1000.
    p['structural']['candidate_logits']=scores.requires_grad_()
    out=api().structural_choice_loss(p,b)
    assert float(out['mean'].detach())==pytest.approx(2000.+math.log(16.))
    b['supervision']['structural']['supervised'][1]=False
    out=api().structural_choice_loss(p,b);assert out['support']==1


@pytest.mark.parametrize('failure',['outside','missing','duplicate'])
def test_structural_invalid_or_uncertified_duplicate_positive_raises(failure):
    p,b,_=example(structural=True);y=b['supervision']['structural']
    if failure=='outside':y['candidate'][0]=17
    elif failure=='missing':y['candidate'][0]=-1
    else:y['candidate'][1]=y['candidate'][0];y['group'][1]=y['group'][0]
    with pytest.raises(ValueError):api().structural_choice_loss(p,b)


def test_structural_candidate_permutation_and_group_size_preserve_support():
    from drl.IL.tests.bc_v1.test_action_heads import reroute_structural
    p,b,model=example(structural=True);old=api().structural_choice_loss(p,b);order=[]
    for start,end in zip(b['inputs']['structural']['group_ptr'][:-1],b['inputs']['structural']['group_ptr'][1:]):
        order.extend(range(int(end)-1,int(start)-1,-1))
    changed=copy.deepcopy(b);changed['inputs']=reroute_structural(b['inputs'],order)
    inverse=torch.argsort(torch.tensor(order));changed['supervision']['structural']['candidate']=inverse[b['supervision']['structural']['candidate']]
    new=api().structural_choice_loss(model(changed['inputs']),changed)
    torch.testing.assert_close(old['mean'],new['mean']);assert new['support']==2
    # Fewer negatives change CE difficulty, never supervision support.
    target=set(b['supervision']['structural']['candidate'].tolist())
    order=[q for q in range(51) if q%2==0 or q in target]
    changed['inputs']=reroute_structural(b['inputs'],order);lookup={q:i for i,q in enumerate(order)}
    changed['supervision']['structural']['candidate']=torch.tensor([lookup[q] for q in b['supervision']['structural']['candidate'].tolist()])
    assert api().structural_choice_loss(model(changed['inputs']),changed)['support']==2


def test_loss_does_not_read_audit_task_or_future_to_select_structural_support():
    class Forbidden(dict):
        def __getitem__(self,key):raise AssertionError('Audit accessed')
    p,b,_=example(structural=True);before=api().structural_choice_loss(p,b)
    b['routing']=Forbidden();b['inputs']['task_one_hot'].zero_();b['audit']=Forbidden();b['stage']='ignored';b['G_t+1']=Forbidden()
    after=api().structural_choice_loss(p,b)
    assert torch.equal(before['sum'],after['sum']) and before['support']==after['support']


def rich_joint():
    from drl.IL.bc_v1 import normalization as f2
    sample,kw=enriched_fixture();g=sample.inputs.policy.graph;m=g.modules[0].ref;port=g.connectors[0].ref
    _,_,bindings=f2._catalog_plan(kw['spec'],kw['feature_schema'],kw['normalization_policy'])
    old=dict(kw['normalizers'].scalers)
    for binding in bindings:
        for key in binding.keys:old.setdefault(key,f2.Scaler(0.,1.,2,'train_standard',False))
    kw={**kw,'normalizers':replace(kw['normalizers'],argument_bindings=bindings,scalers=tuple(sorted(old.items())))}
    target=c.StructuralTarget(m,'gravity_settle',(m,),(),(
        c.ParameterValue('duration',5.),c.ParameterValue('members',(m,)),c.ParameterValue('clear',True),
        c.ParameterValue('choose_module',m),c.ParameterValue('choose_connector',port),
        c.ParameterValue('extra_connectors',(port,)),c.ParameterValue('decide',True),
        c.ParameterValue('ignored',True),c.ParameterValue('adjust',(1.,3.)),c.ParameterValue('select','b')))
    sample=replace(sample,supervision=replace(sample.supervision,structural_targets=(target,)))
    batch=tensorize(sample,kw);model=complete_model(kw)
    return model(batch['inputs']),batch,model


def test_category_argument_ce_is_one_domain_decision():
    p,b,_=rich_joint();records=p['structural']['arguments']['category']
    records['logits']=torch.zeros_like(records['logits'],requires_grad=True)
    out=api().category_argument_loss(p,b)
    assert out['support']==1 and float(out['mean'].detach())==pytest.approx(math.log(3.))
    out['mean'].backward();assert torch.isfinite(records['logits'].grad).all()
    y=b['supervision']['structural'];row=next(i for i,k in enumerate(y['argument_category'].tolist()) if k>=0)
    y['argument_category'][row]=99
    with pytest.raises(ValueError):api().category_argument_loss(p,b)


def test_numeric_arguments_average_vector_components_then_argument_decisions():
    p,b,_=rich_joint();records=p['structural']['arguments']['numeric']
    records['value']=torch.zeros_like(records['value'],requires_grad=True)
    out=api().numeric_argument_loss(p,b)
    # duration target normalized 2 => 1.5; adjust (1,3) => mean(.5,2.5)=1.5.
    assert out['support']==2 and float(out['mean'].detach())==pytest.approx(1.5)
    y=b['supervision']['structural'];x=b['inputs']['structural']
    vectorrow=next(i for i,u in enumerate(y['argument_domain'].tolist()) if int(x['argument_kind'][u])==1)
    start,end=y['argument_value_ptr'][vectorrow:vectorrow+2].tolist();y['argument_value_valid'][end-1]=False
    y['argument_value'][end-1]=float('nan')
    out=api().numeric_argument_loss(p,b)
    assert out['support']==2 and float(out['mean'].detach())==pytest.approx(1.)


def test_single_reference_ce_uses_typed_current_option_domain():
    p,b,_=rich_joint();records=p['structural']['arguments']['reference']
    records['logits']=torch.zeros_like(records['logits'],requires_grad=True)
    out=api().reference_argument_loss(p,b)
    assert out['support']==2 and float(out['mean'].detach())==pytest.approx((math.log(3.)+math.log(6.))/2)
    y=b['supervision']['structural'];x=b['inputs']['structural']
    row=next(i for i,u in enumerate(y['argument_domain'].tolist()) if int(x['argument_kind'][u])==4)
    y['reference_index'][int(y['reference_ptr'][row])]=99
    with pytest.raises(ValueError):api().reference_argument_loss(p,b)


def test_bound_connectors_and_nonlearnable_arguments_never_contribute():
    p,b,_=example(structural=True)
    for name in ('numeric_argument_loss','reference_argument_loss'):
        out=getattr(api(),name)(p,b);assert out['support']==0 and out['mean']==0
    p,b,_=rich_joint();b['supervision']['structural']['argument_supervised'].fill_(False)
    for name in ('category_argument_loss','numeric_argument_loss','reference_argument_loss'):
        out=getattr(api(),name)(p,b);assert out['support']==0 and out['mean']==0


@pytest.mark.parametrize('n',[1,5])
def test_reference_set_means_optional_members_then_counts_one_decision(n):
    sample,kw=fixture(n,0);sample=with_structural(sample,skill=True)
    sample=replace(sample,supervision=replace(sample.supervision,structural_targets=sample.supervision.structural_targets[:1]))
    b=tensorize(sample,kw);p=complete_model(kw)(b['inputs']);r=p['structural']['arguments']['reference_set']
    r['logits']=torch.zeros_like(r['logits'],requires_grad=True)
    out=api().reference_set_loss(p,b)
    assert out['support']==1 and float(out['mean'].detach())==pytest.approx(math.log(2.))
    out['mean'].backward();assert float(r['logits'].grad[0])==pytest.approx(-.5/n)
    if n>1:assert torch.all(r['logits'].grad[1:n]>0)  # Explicit allowed negatives.
    assert torch.equal(r['logits'].grad[n:],torch.zeros_like(r['logits'].grad[n:]))


def test_scope_and_reference_set_are_distinct_decisions_without_required_member_bce():
    p,b,_=rich_joint();sets=p['structural']['arguments']['reference_set'];scope=p['structural']['scope']
    sets['logits']=torch.zeros_like(sets['logits'],requires_grad=True);scope['logits']=torch.zeros_like(scope['logits'],requires_grad=True)
    a=api().reference_set_loss(p,b);z=api().scope_loss(p,b)
    assert a['support']==2 and z['support']==1  # module set + connector set; one participant scope.
    assert float(a['mean'].detach())==pytest.approx(math.log(2.)) and float(z['mean'].detach())==pytest.approx(math.log(2.))
    assert scope['required_domain_index'].numel()==3 and scope['logits'].numel()==6
    # Required scope labels are masked; changing their values cannot affect loss.
    y=b['supervision']['structural'];y['scope_membership_value'][~y['scope_membership_valid']]=1000.
    torch.testing.assert_close(api().scope_loss(p,b)['mean'],z['mean'])


def test_required_argument_members_do_not_receive_trivial_positive_bce():
    from drl.IL.tests.bc_v1.test_action_heads import heads_for
    from drl.IL.tests.bc_v1.test_model import model_for
    sample,kw=fixture(3,0);sample=with_structural(sample,skill=True);b=tensorize(sample,kw);x=b['inputs']['structural']
    u=next(i for i,k in enumerate(x['argument_kind'].tolist()) if k==6);d=int(x['argument_domain_ptr'][u])
    x['argument_domain_required'][d]=True
    p=heads_for(kw)(model_for(kw).encode(b['inputs']),b['inputs']);r=p['structural']['arguments']['reference_set']
    r['logits']=torch.zeros_like(r['logits'],requires_grad=True)
    out=api().reference_set_loss(p,b);assert out['support']==2 and float(out['mean'].detach())==pytest.approx(math.log(2.))
    assert d not in r['domain_index'].tolist()


def test_boolean_bce_is_one_learnable_decision_and_large_logits_stay_finite():
    p,b,_=rich_joint();r=p['structural']['arguments']['boolean'];r['logits']=torch.full_like(r['logits'],-1000.,requires_grad=True)
    out=api().boolean_argument_loss(p,b)
    assert out['support']==1 and out['mean']==1000.
    out['mean'].backward();assert torch.isfinite(r['logits'].grad).all()
    assert r['logits'].grad[0]==-1 and torch.equal(r['logits'].grad[1:],torch.zeros_like(r['logits'].grad[1:]))


def test_membership_false_mask_and_parent_false_mask_remove_support():
    p,b,_=rich_joint();y=b['supervision']['structural'];y['membership_valid'].fill_(False);y['scope_membership_valid'].fill_(False)
    for name in ('reference_set_loss','scope_loss'):
        result=getattr(api(),name)(p,b);assert result['support']==0 and result['mean']==0
    y['supervised'].fill_(False)
    for name in ('category_argument_loss','numeric_argument_loss','reference_argument_loss','boolean_argument_loss'):
        result=getattr(api(),name)(p,b);assert result['support']==0 and result['mean']==0


def test_fixed_scope_and_empty_boolean_families_are_safe():
    p,b,_=example(structural=True)
    for name in ('reference_set_loss','scope_loss','boolean_argument_loss'):
        result=getattr(api(),name)(p,b);assert result['support']==0 and result['mean']==0


def config():
    return api().load_loss_config(ROOT/'configs/bc_v1/loss_v1.json')


def test_full_objective_supports_count_each_target_once_and_weights_one_family():
    p,b,_=rich_joint();total,stats=api().compute_bc_loss(p,b,config())
    expected={'resource_mode':1,'resource_payload':1,'structural_choice':1,'arg_category':1,
        'arg_numeric':2,'arg_reference':2,'arg_reference_set':2,'arg_boolean':1,'scope':1}
    assert {key:int(value['support']) for key,value in stats.items()}==expected
    torch.testing.assert_close(total,sum(row['mean'] for row in stats.values()))
    changed=copy.deepcopy(config());changed['weights']['arg_numeric']=3.
    updated,new=api().compute_bc_loss(p,b,changed)
    torch.testing.assert_close(updated,total+2*stats['arg_numeric']['mean'])
    for family in stats:
        torch.testing.assert_close(stats[family]['sum'],new[family]['sum'])


def test_full_objective_false_masks_ignore_predictions_and_backward_is_zero():
    p,b,_=rich_joint()
    def leaves(tree):
        for key,value in tree.items():
            if isinstance(value,dict):yield from leaves(value)
            elif value.is_floating_point():
                tree[key]=value.detach().clone().requires_grad_();yield tree[key]
    predictions=list(leaves(p))
    for family in b['supervision'].values():
        for value in family.values():
            if value.dtype==torch.bool:value.fill_(False)
    total,stats=api().compute_bc_loss(p,b,config())
    assert total==0 and all(row['support']==0 for row in stats.values());total.backward()
    assert all(value.grad is not None and torch.equal(value.grad,torch.zeros_like(value)) for value in predictions)


def test_full_objective_ignores_unsupervised_mode_payload_and_audit():
    p,b,_=example(structural=True);original,stats=api().compute_bc_loss(p,b,config())
    p['resource']['payload']=p['resource']['payload'].clone();p['resource']['payload'][2]=float('nan')
    b['routing']={'audit':'secret','stage':'secret'};b['future']=object()
    current,new=api().compute_bc_loss(p,b,config());torch.testing.assert_close(original,current)
    assert new['resource_payload']['support']==1


@pytest.mark.parametrize('invalid',['unknown','negative','nan','delta','smoothing'])
def test_loss_config_rejects_unknown_or_invalid_settings(invalid):
    p,b,_=example();cfg=config()
    if invalid=='unknown':cfg['weights']['internal_value']=1.
    elif invalid=='negative':cfg['weights']['scope']=-1.
    elif invalid=='nan':cfg['weights']['scope']=float('nan')
    elif invalid=='delta':cfg['huber_delta']=2.
    else:cfg['label_smoothing']=.1
    with pytest.raises(ValueError):api().compute_bc_loss(p,b,cfg)


def test_targets_with_gradients_and_malformed_membership_are_contract_errors():
    p,b,_=rich_joint();b['supervision']['resource']['payload'].requires_grad_()
    with pytest.raises(ValueError):api().compute_bc_loss(p,b,config())
    p,b,_=rich_joint();b['supervision']['structural']['membership_domain'][0]=99999
    with pytest.raises(ValueError):api().compute_bc_loss(p,b,config())


def test_full_family_sums_supports_are_invariant_to_batch_partitioning():
    sample,kw=fixture();a=tensorize(with_structural(sample),kw)
    other=replace(sample,supervision=replace(sample.supervision,resource_targets=tuple(
        c.ResourceTarget(r.ref,'pace',(1.,2.)) for r in sample.inputs.policy.graph.resources)))
    b=tensorize(with_structural(other,skill=True),kw);joint=collate_policy_samples([a,b]);model=complete_model(kw)
    _,all_stats=api().compute_bc_loss(model(joint['inputs']),joint,config())
    _,sa=api().compute_bc_loss(model(a['inputs']),a,config());_,sb=api().compute_bc_loss(model(b['inputs']),b,config())
    for family,stats in all_stats.items():
        support=sa[family]['support']+sb[family]['support'];numerator=sa[family]['sum']+sb[family]['sum']
        assert support==stats['support'];torch.testing.assert_close(numerator,stats['sum'],atol=3e-6,rtol=3e-6)
        torch.testing.assert_close(numerator/support.clamp_min(1),stats['mean'],atol=3e-6,rtol=3e-6)


def test_concurrent_resource_and_structural_bc_gradients_reach_h2_and_h1():
    sample,kw=fixture();g=sample.inputs.policy.graph
    sample=replace(sample,supervision=replace(sample.supervision,resource_targets=sample.supervision.resource_targets+(
        c.ResourceTarget(g.resources[1].ref,'aim',(5.,)),)))
    b=tensorize(with_structural(sample,skill=True),kw);model=complete_model(kw);h=model.encode(b['inputs'])
    for name in ('resource_h','module_h','context_h'):h[name].retain_grad()
    p=model.forward_heads(h,b['inputs']);total,stats=api().compute_bc_loss(p,b,config())
    assert stats['resource_mode']['support']==2 and stats['resource_payload']['support']==2
    assert stats['structural_choice']['support']==2 and stats['arg_numeric']['support']==2
    assert torch.isfinite(total);total.backward()
    for layer in (model.action_heads.mode_scorer,model.action_heads.payload_decoder,
            model.action_heads.structural_scorer,model.action_heads.numeric_decoder,
            model.resource_context,model.module_fusion):
        grads=[p.grad for p in layer.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads)>0
    for name in ('resource_h','module_h','context_h'):
        assert torch.isfinite(h[name].grad).all() and h[name].grad.abs().sum()>0


def test_masked_and_unrelated_raw_outputs_have_zero_gradient_in_joint_objective():
    p,b,_=rich_joint()
    def leafify(tree):
        for key,value in tree.items():
            if isinstance(value,dict):leafify(value)
            elif value.is_floating_point():tree[key]=value.detach().clone().requires_grad_()
    leafify(p);total,stats=api().compute_bc_loss(p,b,config());total.backward()
    assert torch.equal(p['resource']['payload'].grad[2:],torch.zeros(7))
    assert torch.equal(p['resource']['mode_logits'].grad[2:],torch.zeros(4))
    assert torch.equal(p['structural']['candidate_logits'].grad[17:],torch.zeros(34))
    assert torch.equal(p['structural']['arguments']['boolean']['logits'].grad[1:],torch.zeros(2))
    assert p['resource']['payload'].grad[:2].abs().sum()>0
    assert p['structural']['candidate_logits'].grad[:17].abs().sum()>0


def test_missing_active_argument_prediction_is_not_silently_dropped():
    p,b,_=rich_joint();p['structural']['arguments']['category']['argument']=torch.empty(0,dtype=torch.int64)
    p['structural']['arguments']['category']['ptr']=torch.tensor([0])
    p['structural']['arguments']['category']['logits']=torch.empty(0)
    with pytest.raises(ValueError):api().compute_bc_loss(p,b,config())


@pytest.mark.parametrize('destination',['single_reference','orphan_set'])
def test_valid_membership_routing_to_wrong_kind_or_orphan_domain_fails_closed(destination):
    p,b,_=rich_joint();x=b['inputs']['structural'];y=b['supervision']['structural']
    if destination=='single_reference':
        u=next(u for u in y['argument_domain'].tolist() if int(x['argument_kind'][u])==4)
    else:
        supervised=set(y['argument_domain'].tolist())
        u=next(u for u,k in enumerate(x['argument_kind'].tolist()) if k==6 and u not in supervised)
    start,end=x['argument_domain_ptr'][u:u+2].tolist();assert end-start==3
    # All indices remain globally valid and unique; only ownership is wrong.
    y['membership_domain'][:3]=torch.arange(start,end)
    assert y['membership_valid'][:3].all()
    with pytest.raises(ValueError):api().compute_bc_loss(p,b,config())


@pytest.mark.parametrize('mask',['argument_supervised','supervised'])
def test_valid_membership_ownership_preserves_argument_and_parent_mask_suppression(mask):
    p,b,_=rich_joint();b['supervision']['structural'][mask].fill_(False)
    _,stats=api().compute_bc_loss(p,b,config())
    assert stats['arg_reference_set']['support']==0 and stats['arg_reference_set']['mean']==0
