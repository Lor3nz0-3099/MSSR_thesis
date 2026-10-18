"""I2 streaming diagnostics over real G2/H2 contracts, never training."""
import copy
import gc
import importlib
import importlib.util
import json
import math
from dataclasses import replace
from pathlib import Path
import weakref

import pytest
import torch

from drl.IL.bc_v1 import bc_v1_contract as c,normalization as f2
from drl.IL.bc_v1.batching import collate_policy_samples
from drl.IL.tests.bc_v1.g2_fixtures import fixture,tensorize,with_structural
from drl.IL.tests.bc_v1.test_action_heads import complete_model,enriched_fixture
from drl.IL.tests.bc_v1.test_losses import example,rich_joint,recatalog

ROOT=Path(__file__).resolve().parents[4]


@pytest.fixture(autouse=True,scope='module')
def thread_budget():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.metrics'),'Missing I2 metrics'
    return importlib.import_module('drl.IL.bc_v1.metrics')


def test_streaming_accuracy_uses_global_support_not_average_of_batch_accuracies():
    a=api().MeanAccumulator();a.add([True,True]);a.add([True]+[False]*99)
    assert a.compute()=={'value':3/102,'support':102,'sum':3.}
    assert a.compute()['value']!=(1.+.01)/2
    a.reset();assert a.compute()=={'value':None,'support':0,'sum':0.}


def test_error_stream_accumulates_components_in_float64_and_global_rmse():
    a=api().ErrorAccumulator();a.add(torch.tensor([1.,3.]),torch.zeros(2))
    a.add(torch.tensor([2.,1000.]),torch.zeros(2),torch.tensor([True,False]))
    out=a.compute();assert out['mae']=={'value':2.,'support':3,'sum':6.}
    assert out['rmse']['value']==pytest.approx(math.sqrt(14/3)) and out['rmse']['support']==3
    assert out['rmse']['sum_squared_error']==14.
    assert isinstance(a.sum_abs,float) and isinstance(a.sum_squared,float)


def test_error_stream_partition_invariance_and_zero_support_null():
    a,b=api().ErrorAccumulator(),api().ErrorAccumulator();x=torch.tensor([1.,2.,5.])
    a.add(x,torch.zeros(3));b.add(x[:1],torch.zeros(1));b.add(x[1:],torch.zeros(2))
    assert a.compute()==b.compute();a.reset()
    assert a.compute()['mae']['value'] is None and a.compute()['mae']['support']==0
    assert a.compute()['rmse']['value'] is None


def test_accumulators_do_not_retain_autograd_or_mutate_predictions_and_targets():
    p=torch.tensor([1.,2.],requires_grad=True);target=torch.zeros(2,requires_grad=True)
    original=p.detach().clone();ref=weakref.ref(p);a=api().ErrorAccumulator();a.add(p,target)
    assert torch.equal(p.detach(),original) and p.grad is None and target.grad is None
    del p;gc.collect();assert ref() is None
    json.dumps(a.compute(),allow_nan=False)


def test_supervised_nonfinite_error_raises_but_masked_values_are_unused():
    a=api().ErrorAccumulator()
    with pytest.raises(ValueError):a.add(torch.tensor([float('nan')]),torch.zeros(1))
    a.add(torch.tensor([float('inf'),2.]),torch.zeros(2),torch.tensor([False,True]))
    assert a.compute()['mae']['value']==2. and a.compute()['mae']['support']==1


@pytest.mark.parametrize('offset,scale,count,strategy',[(1.,2.,2,'train_standard'),(2.,4.,2,'physical_bounds'),(0.,1.,0,'train_standard')])
def test_metric_inverse_reuses_f2_for_target_roundtrip(offset,scale,count,strategy):
    sample,kw=fixture();stats=kw['normalizers'];binding=next(b for b in stats.resource_bindings if b.mode_id=='pace')
    scalers=dict(stats.scalers)
    for key in binding.keys:scalers[key]=f2.Scaler(offset,scale,count,strategy,count==0)
    stats=replace(stats,scalers=tuple(sorted(scalers.items())))
    target=sample.supervision.resource_targets[0];normalized=f2.normalize_target(target,stats,module_type_id='kit')
    assert api().physical_values(normalized.payload,binding.keys,stats)==pytest.approx(target.payload)


def test_physical_error_units_tolerance_and_component_decision_support():
    _,kw=fixture();stats=kw['normalizers'];binding=next(b for b in stats.resource_bindings if b.mode_id=='pace')
    a=api().ContinuousAccumulator(stats,binding.keys,binding.units,{'rad/s':1.})
    a.add(torch.tensor([2.5,5.]),torch.tensor([2.,3.]),torch.tensor([True,True]))
    result=a.compute()
    assert result['normalized_mae']['value']==1.25 and result['normalized_mae']['support']==2
    assert result['normalized_rmse']['value']==pytest.approx(math.sqrt(2.125))
    p=result['physical_by_unit']['rad/s']
    assert p['mae']['value']==2.5 and p['mae']['unit']=='rad/s'
    assert p['rmse']['value']==pytest.approx(math.sqrt(8.5))
    assert p['within_tolerance']['value']==.5 and p['within_tolerance']['support']==2
    assert result['decision_support']==1


def test_unconfigured_tolerance_is_null_and_invalid_component_is_not_inverted():
    _,kw=fixture();binding=next(b for b in kw['normalizers'].resource_bindings if b.mode_id=='pace')
    a=api().ContinuousAccumulator(kw['normalizers'],binding.keys,binding.units,{})
    a.add(torch.tensor([2.,float('nan')]),torch.tensor([2.,float('inf')]),torch.tensor([True,False]))
    out=a.compute();assert out['normalized_mae']['support']==1 and out['decision_support']==1
    assert out['physical_by_unit']['rad/s']['within_tolerance']['value'] is None
    assert out['physical_by_unit']['rad/s']['within_tolerance']['support']==0


def metric(kw,**changes):
    cfg=api().load_metrics_config(ROOT/'configs/bc_v1/metrics_v1.json');cfg.update(changes)
    return api().BCMetrics(kw['spec'],kw['normalizers'],cfg)


def resource_example():
    sample,kw=fixture();batch=tensorize(sample,kw)
    return complete_model(kw)(batch['inputs']),batch,kw


def test_resource_payload_metrics_route_signature_inverse_and_component_support():
    p,b,kw=resource_example();p['resource']['payload']=torch.tensor([2.5,5.,999.,0.,0.,0.,0.,0.,0.],requires_grad=True)
    m=metric(kw,tolerances={'rad/s':1.});m.update_resource_payload(p,b)
    payload=m.compute()['resource']['payload'];row=payload['by_signature']['["kit","rotor","pace"]']
    assert payload['normalized_mae']['value']==1.25 and payload['normalized_mae']['support']==2
    assert payload['decision_support']==1 and row['decision_support']==1
    assert row['physical_by_unit']['rad/s']['mae']['value']==2.5
    assert row['physical_by_unit']['rad/s']['within_tolerance']['value']==.5


def test_resource_payload_masks_ignore_unused_modes_and_missing_components():
    p,b,kw=resource_example();p['resource']['payload']=torch.tensor([2.5,float('nan'),float('nan'),0.,0.,0.,0.,0.,0.])
    b['supervision']['resource']['payload_valid'][1]=False
    m=metric(kw);m.update_resource_payload(p,b);row=m.compute()['resource']['payload']
    assert row['normalized_mae']['value']==.5 and row['normalized_mae']['support']==1
    assert row['by_signature']['["kit","rotor","aim"]']['normalized_mae']['value'] is None
    b['supervision']['resource']['supervised'].fill_(False)
    m.update_resource_payload(p,b);assert m.compute()['resource']['payload']['decision_support']==1


def test_resource_numeric_metric_streaming_equals_repeated_component_observations():
    p,b,kw=resource_example();p['resource']['payload']=torch.tensor([3.,4.,0.,0.,0.,0.,0.,0.,0.])
    m=metric(kw);m.update_resource_payload(p,b);m.update_resource_payload(p,b)
    row=m.compute()['resource']['payload'];assert row['normalized_mae']['value']==1.
    assert row['normalized_rmse']['value']==1. and row['normalized_rmse']['support']==4
    assert row['decision_support']==2


def test_resource_payload_supervised_nonfinite_prediction_is_error():
    p,b,kw=resource_example();p['resource']['payload']=torch.full_like(p['resource']['payload'],float('nan'))
    with pytest.raises(ValueError):metric(kw).update_resource_payload(p,b)


@pytest.mark.parametrize('bad',['unknown','negative','nonfinite','group'])
def test_metric_config_rejects_invalid_diagnostic_settings(bad):
    _,kw=fixture();cfg=api().load_metrics_config(ROOT/'configs/bc_v1/metrics_v1.json')
    if bad=='unknown':cfg['checkpoint_threshold']=.9
    elif bad=='negative':cfg['tolerances']['rad']=-1.
    elif bad=='nonfinite':cfg['tolerances']['rad']=float('nan')
    else:cfg['breakdowns']=['future_success']
    with pytest.raises(ValueError):api().BCMetrics(kw['spec'],kw['normalizers'],cfg)


def test_classification_confusion_macro_f1_and_balanced_accuracy_exclude_absent_class():
    a=api().ConfusionAccumulator(('a','b','unused'))
    for target,predicted in [(0,0),(0,1),(1,1),(1,1)]:a.add(target,predicted)
    out=a.compute();assert out['counts']==[[1,1,0],[0,2,0],[0,0,0]]
    assert out['class_support']==[2,2,0] and out['accuracy']['value']==.75
    assert out['macro_f1']['value']==pytest.approx(((2/3)+.8)/2)
    assert out['balanced_accuracy']['value']==.75 and out['balanced_accuracy']['classes_with_support']==2
    assert out['recall'][2]['value'] is None and out['recall'][2]['support']==0


def test_resource_mode_metrics_exclude_singletons_and_preserve_class_vocabulary():
    p,b,kw=resource_example();p['resource']['mode_logits']=torch.tensor([3.,0.,0.,0.,0.,0.])
    m=metric(kw);m.update_resource_mode(p,b);out=m.compute()['resource']['mode']
    assert out['accuracy']['value']==1. and out['accuracy']['support']==1
    domain=next(iter(out['domains'].values()))
    assert domain['classes']==[('kit','rotor','aim'),('kit','rotor','pace')]
    assert domain['counts']==[[0,0],[0,1]] and domain['balanced_accuracy']['value']==1.
    sample,kw=fixture();spec=kw['spec'];mt=spec.module_types[0];r=mt.resources[0]
    sample,kw=recatalog(sample,kw,replace(spec,module_types=(replace(mt,resources=(replace(r,modes=r.modes[:1]),)),)))
    batch=tensorize(sample,kw);p=complete_model(kw)(batch['inputs']);m=metric(kw);m.update_resource_mode(p,batch)
    assert m.compute()['resource']['mode']['accuracy']=={'value':None,'support':0,'sum':0.}


def test_resource_mode_repeated_batches_have_correct_global_confusion():
    p,b,kw=resource_example();m=metric(kw)
    p['resource']['mode_logits']=torch.tensor([0.,5.,0.,0.,0.,0.]);m.update_resource_mode(p,b)
    p['resource']['mode_logits']=torch.tensor([5.,0.,0.,0.,0.,0.]);m.update_resource_mode(p,b)
    out=m.compute()['resource']['mode'];assert out['accuracy']['value']==.5 and out['accuracy']['support']==2
    assert next(iter(out['domains'].values()))['counts']==[[0,0],[1,1]]


def test_different_current_mode_domains_remain_separate_confusions():
    sample,kw=fixture();spec=kw['spec'];mt=spec.module_types[0];res=mt.resources[0]
    mode=replace(res.modes[1],mode_id='third');action=replace(next(a for a in spec.actions if a.action_id=='aim'),action_id='third')
    policy=copy.deepcopy(kw['normalization_policy']);policy['resource_payloads'].append({
        'module_type_id':'kit','resource_id':'rotor','mode_id':'third','components':[{'parameter':'angle','component':0}]})
    sample,kw=recatalog(sample,kw,replace(spec,module_types=(replace(mt,resources=(replace(res,modes=res.modes+(mode,)),)),),actions=spec.actions+(action,)))
    _,rb,ab=f2._catalog_plan(kw['spec'],kw['feature_schema'],policy)
    kw={**kw,'normalization_policy':policy,'normalizers':replace(kw['normalizers'],resource_bindings=rb,argument_bindings=ab,
        metadata_items=tuple(sorted({**kw['normalizers'].metadata,'policy_fingerprint':f2.policy_fingerprint(policy,kw['feature_schema'])}.items())))}
    sample=replace(sample,supervision=replace(sample.supervision,resource_targets=sample.supervision.resource_targets+(
        c.ResourceTarget(sample.inputs.policy.graph.resources[1].ref,'pace',(5.,7.)),)))
    batch=tensorize(sample,kw);rc=batch['inputs']['resource_candidates'];keep=torch.tensor([0,1,2,3,4,6,7,8])
    for key in ('resource','mode','action'):rc[key]=rc[key][keep]
    rc['group_ptr']=torch.tensor([0,3,5,8]);ptr=[0]
    modes=sorted((t.type_id,r.resource_id,m.mode_id,m.payload_size) for t in kw['spec'].module_types for r in t.resources for m in r.modes)
    for mode_index in rc['mode'].tolist():ptr.append(ptr[-1]+modes[mode_index][3])
    rc['payload_ptr']=torch.tensor(ptr)
    p=complete_model(kw)(batch['inputs']);m=metric(kw);m.update_resource_mode(p,batch)
    domains=m.compute()['resource']['mode']['domains']
    assert sorted(len(d['classes']) for d in domains.values())==[2,3]
    assert [d['accuracy']['support'] for d in domains.values()]==[1,1]


def test_structural_metrics_are_per_initiator_and_endpoints_follow_expert_applicability():
    p,b,_=example(structural=True);_,kw=fixture();s=b['supervision']['structural'];q=s['candidate'].tolist()
    p['structural']['candidate_logits']=torch.full((51,),-10.)
    p['structural']['candidate_logits'][q[0]]=5.;p['structural']['candidate_logits'][33]=5.  # Wrong endpoint-free candidate for initiator1.
    m=metric(kw);m.update_structural(p,b);out=m.compute()['structural']
    for name in ('candidate_top1','binding_exact_match','operation_accuracy','peer_accuracy','connector_a_accuracy','connector_b_accuracy','ordered_connector_pair_accuracy'):
        assert out[name]['value']==.5 and out[name]['support']==2
    assert out['operation_recall']['align_faces']['value']==.5
    assert out['operation_recall']['gravity_settle']['value'] is None


def test_rare_operation_recall_remains_visible_and_macro_excludes_zero_support():
    p,b,_=example(structural=True);_,kw=fixture();m=metric(kw)
    p['structural']['candidate_logits']=torch.zeros(51)  # All exact ties: expert among maxima.
    m.update_structural(p,b)
    p,b,_=example(structural=True,skill=True);b['supervision']['structural']['supervised'][1]=False
    p['structural']['candidate_logits']=torch.zeros(51);p['structural']['candidate_logits'][0]=5.
    m.update_structural(p,b);out=m.compute()['structural']
    assert out['operation_accuracy']['value']==pytest.approx(2/3)
    assert out['operation_recall']['align_faces']['value']==1. and out['operation_recall']['align_faces']['support']==2
    assert out['operation_recall']['gravity_settle']['value']==0. and out['operation_recall']['gravity_settle']['support']==1
    assert out['operation_recall']['undock']['value'] is None
    assert out['operation_macro_recall']['value']==.5 and out['operation_macro_recall']['classes_with_support']==2


def test_endpoint_free_structural_target_has_no_peer_or_connector_support():
    p,b,_=example(structural=True,skill=True);_,kw=fixture();m=metric(kw)
    p['structural']['candidate_logits']=torch.full((51,),-1.);p['structural']['candidate_logits'][b['supervision']['structural']['candidate']]=3.
    m.update_structural(p,b);out=m.compute()['structural']
    assert out['candidate_top1']['value']==1. and out['operation_recall']['gravity_settle']['support']==2
    for name in ('peer_accuracy','connector_a_accuracy','connector_b_accuracy','ordered_connector_pair_accuracy'):
        assert out[name]['value'] is None and out[name]['support']==0


def test_structural_ties_and_consistent_candidate_permutation_are_id_order_independent():
    from drl.IL.tests.bc_v1.test_action_heads import reroute_structural
    p,b,_=example(structural=True);_,kw=fixture();p['structural']['candidate_logits']=torch.zeros(51)
    m=metric(kw);m.update_structural(p,b);before=m.compute()
    order=[i for start in (0,17,34) for i in range(start+16,start-1,-1)]
    changed=copy.deepcopy(b);changed['inputs']=reroute_structural(b['inputs'],order)
    changed['supervision']['structural']['candidate']=torch.argsort(torch.tensor(order))[b['supervision']['structural']['candidate']]
    new=metric(kw);new.update_structural(p,changed)
    assert new.compute()==before and before['structural']['candidate_top1']['value']==1.


@pytest.mark.parametrize('bad',['nonfinite','outside','duplicate'])
def test_structural_metric_contract_errors_are_not_silently_dropped(bad):
    p,b,_=example(structural=True);_,kw=fixture()
    if bad=='nonfinite':p['structural']['candidate_logits']=torch.full((51,),float('nan'))
    elif bad=='outside':b['supervision']['structural']['candidate'][0]=17
    else:
        b['supervision']['structural']['candidate'][1]=b['supervision']['structural']['candidate'][0]
        b['supervision']['structural']['group'][1]=0
    with pytest.raises(ValueError):metric(kw).update_structural(p,b)


def rich_metrics():
    p,b,_=rich_joint();_,kw=enriched_fixture();_,rb,ab=f2._catalog_plan(kw['spec'],kw['feature_schema'],kw['normalization_policy'])
    scalers=dict(kw['normalizers'].scalers)
    for binding in ab:
        for key in binding.keys:scalers.setdefault(key,f2.Scaler(0.,1.,2,'train_standard',False))
    kw={**kw,'normalizers':replace(kw['normalizers'],resource_bindings=rb,argument_bindings=ab,scalers=tuple(sorted(scalers.items())))}
    return p,b,kw


def set_discrete_arguments_perfect(p,b):
    s=b['inputs']['structural'];y=b['supervision']['structural'];args=p['structural']['arguments']
    for family in ('category','reference','reference_set','boolean'):
        args[family]['logits']=torch.full_like(args[family]['logits'],-2.)
    for row,u in enumerate(y['argument_domain'].tolist()):
        if not bool(y['argument_supervised'][row]):continue
        kind=int(s['argument_kind'][u]);family={3:'category',4:'reference',5:'reference',6:'reference_set',7:'reference_set',2:'boolean'}.get(kind)
        if family is None or u not in args[family]['argument'].tolist():continue
        records=args[family];r=records['argument'].tolist().index(u)
        if family=='boolean':records['logits'][r]=2*float(y['argument_value'][int(y['argument_value_ptr'][row])])-1;continue
        a,z=records['ptr'][r:r+2].tolist()
        if family=='category':records['logits'][a+int(y['argument_category'][row])]=2.
        elif family=='reference':
            target=int(y['reference_index'][int(y['reference_ptr'][row])]);options=s['argument_domain_index'][records['domain_index'][a:z]].tolist()
            records['logits'][a+options.index(target)]=2.
        else:
            lookup=dict(zip(y['membership_domain'].tolist(),y['membership_value'].tolist()))
            for j,d in enumerate(records['domain_index'][a:z].tolist()):records['logits'][a+j]=2*lookup[d]-1
    scope=p['structural']['scope'];scope['logits']=torch.full_like(scope['logits'],-2.)
    lookup=dict(zip(y['scope_membership_index'].tolist(),y['scope_membership_value'].tolist()))
    for j,d in enumerate(scope['domain_index'].tolist()):
        if d in lookup:scope['logits'][j]=2*lookup[d]-1


def test_typed_discrete_argument_metrics_and_scope_follow_separate_domains():
    p,b,kw=rich_metrics();set_discrete_arguments_perfect(p,b);m=metric(kw)
    m.update_arguments(p,b);m.update_scope(p,b);out=m.compute()
    args=out['arguments']
    for name,support in [('category_accuracy',1),('reference_top1',2),('boolean_accuracy',1)]:
        assert args[name]['value']==1. and args[name]['support']==support
    assert args['reference_set']['exact_match']['value']==1. and args['reference_set']['jaccard']['support']==2
    assert out['structural']['scope']['exact_match']['value']==1. and out['structural']['scope']['jaccard']['support']==1
    # No positive optional participants, but passive/reference sets contain positives.
    p['structural']['arguments']['reference_set']['logits'].fill_(-1.)
    m=metric(kw);m.update_arguments(p,b);m.update_scope(p,b)
    assert m.compute()['arguments']['reference_set']['exact_match']['value']==0.
    assert m.compute()['structural']['scope']['exact_match']['value']==1.


def test_numeric_argument_metrics_use_actual_f2_binding_units_and_component_support():
    p,b,kw=rich_metrics();p['structural']['arguments']['numeric']['value']=torch.zeros_like(p['structural']['arguments']['numeric']['value'])
    m=metric(kw);m.update_arguments(p,b);out=m.compute()['arguments']['numeric']
    assert out['normalized_mae']['value']==2. and out['normalized_mae']['support']==3 and out['argument_support']==2
    assert out['normalized_rmse']['value']==pytest.approx(math.sqrt(14/3))
    assert out['by_parameter']['["gravity_settle","duration"]']['physical_by_unit']['s']['mae']['value']==4.
    adjust=out['by_parameter']['["gravity_settle","adjust"]']['physical_by_unit']['m']
    assert adjust['mae']['value']==2. and adjust['within_tolerance']['value'] is None


def test_empty_optional_reference_set_jaccard_is_one_and_logit_zero_means_absent():
    sample,kw=fixture(3,0);sample=with_structural(sample,skill=True)
    targets=tuple(replace(t,parameters=tuple(replace(v,value=()) if v.name=='members' else v for v in t.parameters)) for t in sample.supervision.structural_targets)
    sample=replace(sample,supervision=replace(sample.supervision,structural_targets=targets));b=tensorize(sample,kw)
    p=complete_model(kw)(b['inputs']);p['structural']['arguments']['reference_set']['logits']=torch.zeros_like(p['structural']['arguments']['reference_set']['logits'])
    m=metric(kw);m.update_arguments(p,b);out=m.compute()['arguments']['reference_set']
    assert out['exact_match']['value']==1. and out['jaccard']['value']==1. and out['jaccard']['support']==2


def test_required_reference_members_are_excluded_from_optional_set_metrics():
    sample,kw=fixture(3,0);sample=with_structural(sample,skill=True)
    sample=replace(sample,supervision=replace(sample.supervision,structural_targets=sample.supervision.structural_targets[:1]))
    b=tensorize(sample,kw);s=b['inputs']['structural'];u=next(i for i,k in enumerate(s['argument_kind'].tolist()) if k==6)
    s['argument_domain_required'][int(s['argument_domain_ptr'][u])]=True
    p=complete_model(kw)(b['inputs']);p['structural']['arguments']['reference_set']['logits']=torch.full_like(p['structural']['arguments']['reference_set']['logits'],-1.)
    m=metric(kw);m.update_arguments(p,b)
    assert m.compute()['arguments']['reference_set']['exact_match']['value']==1.


def test_absent_arguments_are_null_and_invalid_reference_or_membership_is_error():
    p,b,kw=resource_example();m=metric(kw);m.update_arguments(p,b);out=m.compute()['arguments']
    for name in ('category_accuracy','reference_top1','boolean_accuracy'):assert out[name]['value'] is None and out[name]['support']==0
    p,b,kw=rich_metrics();b['supervision']['structural']['reference_index'][1]=9999
    with pytest.raises(ValueError):metric(kw).update_arguments(p,b)
    p,b,kw=rich_metrics();y=b['supervision']['structural'];s=b['inputs']['structural']
    u=next(u for u in y['argument_domain'].tolist() if int(s['argument_kind'][u])==4);start=int(s['argument_domain_ptr'][u])
    y['membership_domain'][:3]=torch.arange(start,start+3)
    with pytest.raises(ValueError):metric(kw).update_arguments(p,b)


def perfect_structural(p,b):
    p['structural']['candidate_logits']=torch.full_like(p['structural']['candidate_logits'],-2.)
    p['structural']['candidate_logits'][b['supervision']['structural']['candidate']]=2.
    set_discrete_arguments_perfect(p,b)


def test_structural_discrete_tuple_requires_binding_and_all_discrete_choices():
    p,b,kw=rich_metrics();perfect_structural(p,b);m=metric(kw);m.update_tuple(p,b)
    assert m.compute()['structural']['discrete_tuple_exact']=={'value':1.,'support':1,'sum':1.,'excluded_incomplete':0}
    records=p['structural']['arguments']['category'];records['logits'].fill_(0.)
    m=metric(kw);m.update_tuple(p,b)
    assert m.compute()['structural']['discrete_tuple_exact']['value']==0.


def test_structural_tuple_excludes_numeric_values_and_never_fabricates_settle_endpoints():
    p,b,kw=rich_metrics();perfect_structural(p,b)
    p['structural']['arguments']['numeric']['value']=torch.full_like(p['structural']['arguments']['numeric']['value'],1000000.)
    m=metric(kw);m.update_tuple(p,b);assert m.compute()['structural']['discrete_tuple_exact']['value']==1.
    assert int(b['inputs']['structural']['peer'][int(b['supervision']['structural']['candidate'][0])])==-1


def test_incomplete_discrete_tuple_is_unknown_instead_of_false_or_inflated_match():
    p,b,kw=rich_metrics();perfect_structural(p,b);y=b['supervision']['structural'];x=b['inputs']['structural']
    row=next(i for i,u in enumerate(y['argument_domain'].tolist()) if int(x['argument_kind'][u])==3)
    y['argument_supervised'][row]=False
    m=metric(kw);m.update_tuple(p,b);out=m.compute()['structural']['discrete_tuple_exact']
    assert out['value'] is None and out['support']==0 and out['excluded_incomplete']==1


def test_partial_set_supervision_cannot_claim_full_set_or_tuple_exact_match():
    p,b,kw=rich_metrics();perfect_structural(p,b);b['supervision']['structural']['membership_valid'][0]=False
    m=metric(kw);m.update_arguments(p,b);m.update_tuple(p,b);out=m.compute()
    assert out['arguments']['reference_set']['exact_match']['support']==1  # Other fully observed set remains.
    assert out['structural']['discrete_tuple_exact']['value'] is None


def test_fixed_bound_arguments_are_not_missing_tuple_supervision():
    p,b,_=example(structural=True);_,kw=fixture();perfect_structural(p,b)
    m=metric(kw);m.update_tuple(p,b)
    assert m.compute()['structural']['discrete_tuple_exact']['value']==1.
    assert m.compute()['structural']['discrete_tuple_exact']['support']==2


def test_postforward_task_and_morphology_breakdowns_preserve_base_support():
    p,b,kw=resource_example();m=metric(kw,breakdowns=['task','morphology','task_stratum'])
    for morphology in ('snake','mobile'):
        with m.audit_context(b,{'morphology_stratum':morphology,'task_stratum':'teleop_mission'}):
            m.update_resource_payload(p,b)
    out=m.compute();assert out['resource']['payload']['normalized_mae']['support']==4
    assert out['breakdowns']['task']['gap']['resource']['payload']['normalized_mae']['support']==4
    assert out['breakdowns']['task_stratum']['teleop_mission']['resource']['payload']['normalized_mae']['support']==4
    assert 'teleop_mission' not in out['breakdowns']['task']
    for morphology in ('snake','mobile'):
        assert out['breakdowns']['morphology'][morphology]['resource']['payload']['normalized_mae']['support']==2


def test_changed_or_missing_audit_changes_only_breakdown_labels_and_not_predictions():
    p,b,kw=resource_example();before=p['resource']['payload'].detach().clone();a=metric(kw);z=metric(kw)
    with a.audit_context(b,{'morphology_stratum':'snake'}):a.update_resource_payload(p,b)
    with z.audit_context(b,{}):z.update_resource_payload(p,b)
    assert a.compute()['resource']==z.compute()['resource']
    assert 'UNKNOWN' in z.compute()['breakdowns']['morphology']
    assert torch.equal(before,p['resource']['payload'].detach())


def test_batched_audit_breakdowns_use_graph_ownership_and_missing_context_is_unknown():
    sample,kw=fixture();a=tensorize(sample,kw);b=tensorize(sample,kw);batch=collate_policy_samples([a,b]);p=complete_model(kw)(batch['inputs']);m=metric(kw)
    with m.audit_context(batch,[{'morphology_stratum':'a'},{'morphology_stratum':'b'}]):m.update_resource_payload(p,batch)
    out=m.compute()['breakdowns']['morphology']
    assert out['a']['resource']['payload']['decision_support']==1 and out['b']['resource']['payload']['decision_support']==1
    b['routing']={};m=metric(kw)
    with m.audit_context(b,{}):m.update_resource_payload(complete_model(kw)(b['inputs']),b)
    assert 'UNKNOWN' in m.compute()['breakdowns']['task']


def test_full_streaming_api_global_accuracy_is_three_of_102_not_mean_batch_accuracy():
    sample,kw=fixture(1);one=tensorize(sample,kw);model=complete_model(kw);m=metric(kw,breakdowns=[])
    for count,correct in [(2,2),(100,1)]:
        b=collate_policy_samples([one]*count);p=model(b['inputs'])
        p['resource']['mode_logits']=torch.tensor([v for i in range(count) for v in ((1.,0.) if i<correct else (0.,1.))])
        m.update(p,b)
    out=m.compute();assert out['resource']['mode']['accuracy']['value']==pytest.approx(3/102)
    assert out['resource']['mode']['accuracy']['support']==102
    m.reset();empty=m.compute()
    assert empty['resource']['mode']['accuracy']['value'] is None and empty['resource']['payload']['normalized_mae']['support']==0


def test_full_streaming_batch_partition_invariance_all_metric_families():
    sample,kw=fixture();a=tensorize(with_structural(sample),kw);b=tensorize(with_structural(sample,skill=True),kw)
    joint=collate_policy_samples([a,b]);model=complete_model(kw);whole=metric(kw);parts=metric(kw)
    def identical_predictions(batch):
        p=model(batch['inputs'])
        def fix(tree):
            for key,value in tree.items():
                if isinstance(value,dict):fix(value)
                elif value.is_floating_point():tree[key]=torch.ones_like(value)
        fix(p);return p
    # Hold predictions identical; FP32 GEMM batch-size roundoff is not metric
    # aggregation. Routing still comes from actual variable G2/H2 contracts.
    whole.update(identical_predictions(joint),joint)
    parts.update(identical_predictions(a),a);parts.update(identical_predictions(b),b)
    def compare(x,y):
        if isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:compare(x[k],y[k])
        elif isinstance(x,float):assert x==pytest.approx(y,abs=3e-6,rel=3e-6)
        else:assert x==y
    compare(whole.compute(),parts.compute());json.dumps(whole.compute(),allow_nan=False)


def test_full_metrics_do_not_mutate_loss_predictions_targets_or_existing_gradients():
    from drl.IL.bc_v1.losses import compute_bc_loss,load_loss_config
    p,b,kw=resource_example();loss_config=load_loss_config(ROOT/'configs/bc_v1/loss_v1.json')
    loss_before,_=compute_bc_loss(p,b,loss_config)
    original=p['resource']['payload'].detach().clone();p['resource']['payload'].retain_grad()
    p['resource']['payload'].grad=torch.ones_like(p['resource']['payload'])
    b['supervision']['resource']['payload'].requires_grad_()
    m=metric(kw);m.update(p,b,audit={'morphology_stratum':'changed'})
    assert torch.equal(original,p['resource']['payload'].detach())
    assert torch.equal(p['resource']['payload'].grad,torch.ones_like(original))
    assert b['supervision']['resource']['payload'].grad is None
    assert m._labels is None
    b['supervision']['resource']['payload']=b['supervision']['resource']['payload'].detach()
    loss_after,_=compute_bc_loss(p,b,loss_config);torch.testing.assert_close(loss_before,loss_after)
    ref=weakref.ref(p['resource']['payload']);del p,loss_before,loss_after;gc.collect();assert ref() is None


def test_full_metrics_missing_support_is_null_and_zero_accuracy_has_positive_support():
    p,b,kw=resource_example();p['resource']['mode_logits']=torch.tensor([0.,2.,0.,0.,0.,0.]);m=metric(kw);m.update(p,b)
    out=m.compute();assert out['resource']['mode']['accuracy']['value']==0. and out['resource']['mode']['accuracy']['support']==1
    assert out['arguments']['reference_top1']['value'] is None and out['arguments']['reference_top1']['support']==0
    assert out['structural']['operation_recall']['gravity_settle']['value'] is None


def test_full_metrics_reject_normalizer_mismatch_and_error_accumulator_overflow():
    p,b,kw=resource_example();b['routing']['normalizer_fingerprint']='foreign'
    with pytest.raises(ValueError):metric(kw).update(p,b)


def test_error_accumulator_rejects_finite_values_whose_error_overflows_float64():
    with pytest.raises(ValueError):api().ErrorAccumulator().add(torch.tensor([1e308],dtype=torch.float64),torch.tensor([-1e308],dtype=torch.float64))


def test_categorical_resource_payload_has_accuracy_without_fake_continuous_components():
    from drl.IL.tests.bc_v1.test_losses import categorical_resource_example
    p,b=categorical_resource_example();sample,kw=fixture();spec=kw['spec'];mt=spec.module_types[0];res=mt.resources[0]
    modes={r.mode.mode_id:r.mode for r in b['routing']['resource_candidates']}
    spec=replace(spec,actions=b['routing']['actions'],module_types=(replace(mt,resources=(replace(res,modes=tuple(modes[m.mode_id] for m in res.modes)),)),))
    _,kw=recatalog(sample,kw,spec)
    kw={**kw,'normalizers':replace(kw['normalizers'],resource_bindings=tuple(replace(r,codec='categorical',keys=(),units=('1',)) if r.mode_id=='aim' else r for r in kw['normalizers'].resource_bindings))}
    p['resource']['categorical']['logits']=torch.tensor([0.,2.,0.,0.,0.,0.,0.,0.,0.])
    m=metric(kw);m.update(p,b);out=m.compute()['resource']
    assert out['payload_category_accuracy']['value']==1. and out['payload_category_accuracy']['support']==1
    assert out['payload']['normalized_mae']['value'] is None and out['payload']['normalized_mae']['support']==0


def test_ordered_connector_pair_requires_one_coherent_maximizer_at_a_tie():
    p,b,_=example(structural=True);_,kw=fixture();x=b['inputs']['structural'];y=b['supervision']['structural']
    y['supervised'][1]=False;q=int(y['candidate'][0]);start,end=b['inputs']['structural']['group_ptr'][:2].tolist()
    candidates=[i for i in range(start,end) if int(x['action'][i])==int(x['action'][q])]
    a=next(i for i in candidates if x['connector_a'][i]==x['connector_a'][q] and x['connector_b'][i]!=x['connector_b'][q])
    z=next(i for i in candidates if x['connector_a'][i]!=x['connector_a'][q] and x['connector_b'][i]==x['connector_b'][q])
    p['structural']['candidate_logits']=torch.zeros(51);p['structural']['candidate_logits'][[a,z]]=3.
    m=metric(kw);m.update_structural(p,b);out=m.compute()['structural']
    assert out['connector_a_accuracy']['value']==1. and out['connector_b_accuracy']['value']==1.
    assert out['ordered_connector_pair_accuracy']['value']==0. and out['candidate_top1']['value']==0.


def test_mixed_physical_units_are_separate_even_inside_one_vector_signature():
    _,kw=fixture();keys=(f2.argument_key('motion','linear',0,'m'),f2.argument_key('motion','angular',0,'rad'))
    stats=replace(kw['normalizers'],scalers=kw['normalizers'].scalers+tuple(
        (k,f2.Scaler(0.,scale,2,'train_standard',False)) for k,scale in zip(keys,(2.,3.))))
    a=api().ContinuousAccumulator(stats,keys,('m','rad'),{'rad':.05})
    a.add(torch.tensor([1.,2.]),torch.zeros(2),torch.ones(2,dtype=torch.bool));out=a.compute()
    assert out['normalized_mae']['value']==1.5
    assert out['physical_by_unit']['m']['mae']['value']==2. and out['physical_by_unit']['rad']['mae']['value']==6.
    assert out['physical_by_unit']['m']['mae']['support']==1
