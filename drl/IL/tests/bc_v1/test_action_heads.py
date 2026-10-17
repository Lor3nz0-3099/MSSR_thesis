"""H2 raw-output behavior over the generic G2/H1 contracts; no BC losses."""
import copy
import importlib
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.bc_v1.batching import collate_policy_samples
from drl.IL.tests.bc_v1.g2_fixtures import fixture,tensorize
from drl.IL.tests.bc_v1.test_model import model_for,equal_trees

ROOT=Path(__file__).resolve().parents[4]


@pytest.fixture(autouse=True,scope='module')
def small_cpu_thread_budget():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.action_heads'), 'Missing H2 resource heads'
    return importlib.import_module('drl.IL.bc_v1.action_heads')


def heads_for(kw):
    from drl.IL.bc_v1.model import load_model_config
    mod=api();torch.manual_seed(31415)
    return mod.ActionHeads(kw['spec'],load_model_config(ROOT/'configs/bc_v1/gnn_v1.json'),
        mod.load_action_head_config(ROOT/'configs/bc_v1/action_heads_v1.json')).eval()


@pytest.mark.parametrize('n',[1,3,12])
def test_resource_groups_and_catalog_payload_arities(n):
    api();sample,kw=fixture(n);batch=tensorize(sample,kw);model=model_for(kw);heads=heads_for(kw)
    result=heads.forward_resource(model.encode(batch['inputs']),batch['inputs'])
    assert result['mode_logits'].shape==(2*n,) and result['mode_logits'].dtype==torch.float32
    assert result['mode_group_ptr'].tolist()==list(range(0,2*n+1,2))
    assert result['mode_candidate'].tolist()==list(range(2*n))
    assert result['payload'].shape==(3*n,) and result['payload'].dtype==torch.float32
    assert result['payload_candidate'].tolist()==list(range(2*n))
    assert result['payload_ptr'].tolist()==[v for i in range(n) for v in (3*i,3*i+2)]+[3*n]
    assert torch.isfinite(result['mode_logits']).all() and torch.isfinite(result['payload']).all()


def test_single_mode_has_payload_without_trainable_ce_scores():
    api();from drl.IL.bc_v1.candidates import enumerate_candidates
    sample,kw=fixture();spec=kw['spec'];mt=spec.module_types[0];res=mt.resources[0]
    spec=replace(spec,module_types=(replace(mt,resources=(replace(res,modes=res.modes[:1]),)),))
    kw={**kw,'spec':spec,'normalizers':replace(kw['normalizers'],metadata_items=tuple(sorted({
        **kw['normalizers'].metadata,'robot_spec_fingerprint':c.robot_spec_fingerprint(spec)}.items())))}
    sample=replace(sample,inputs=replace(sample.inputs,candidates=enumerate_candidates(sample.inputs.policy.graph,spec,kw['feature_schema'])))
    x=tensorize(sample,kw)['inputs'];heads=heads_for(kw)
    out=heads.forward_resource(model_for(kw).encode(x),x)
    assert out['mode_logits'].shape==(0,) and out['mode_group_ptr'].tolist()==[0,0,0,0]
    assert out['payload'].shape==(6,) and out['payload_ptr'].tolist()==[0,2,4,6]


def test_resource_permutation_equivariance_without_instance_embeddings():
    api();sample,kw=fixture();g=sample.inputs.policy.graph
    changed=replace(sample,inputs=replace(sample.inputs,
        policy=replace(sample.inputs.policy,graph=replace(g,resources=g.resources[::-1])),
        features=replace(sample.inputs.features,resources=sample.inputs.features.resources[::-1])))
    a,b=(tensorize(s,kw)['inputs'] for s in (sample,changed));model=model_for(kw);heads=heads_for(kw)
    before=heads.forward_resource(model.encode(a),a);after=heads.forward_resource(model.encode(b),b)
    torch.testing.assert_close(after['mode_logits'].reshape(3,2),before['mode_logits'].reshape(3,2).flip(0))
    torch.testing.assert_close(after['payload'].reshape(3,3),before['payload'].reshape(3,3).flip(0))
    assert not any(isinstance(m,torch.nn.Embedding) for m in heads.modules())


def test_resource_batch_isolation_and_latents_are_read_only():
    api();a,kw=fixture(2);b,_=fixture(3)
    x=collate_policy_samples([tensorize(a,kw),tensorize(b,kw)])['inputs'];model=model_for(kw);heads=heads_for(kw)
    latent=model.encode(x);snapshot=copy.deepcopy({k:v.detach() for k,v in latent.items()})
    before=heads.forward_resource(latent,x);equal_trees(snapshot,{k:v.detach() for k,v in latent.items()})
    changed=copy.deepcopy(x);changed['module_x'][0,0]+=2
    after=heads.forward_resource(model.encode(changed),changed)
    torch.testing.assert_close(before['mode_logits'][4:],after['mode_logits'][4:])
    torch.testing.assert_close(before['payload'][6:],after['payload'][6:])
    assert not torch.allclose(before['payload'][:6],after['payload'][:6])


def test_resource_outputs_are_raw_unclamped_and_eval_deterministic():
    api();sample,kw=fixture();x=tensorize(sample,kw)['inputs'];model=model_for(kw);heads=heads_for(kw)
    latent=model.encode(x)
    with torch.no_grad():
        heads.mode_scorer[-1].weight.zero_();heads.mode_scorer[-1].bias.fill_(2.)
        heads.payload_decoder[-1].weight.zero_();heads.payload_decoder[-1].bias.fill_(-3.)
    out=heads.forward_resource(latent,x)
    assert out['mode_logits'].tolist()==[2.]*6 and out['payload'].tolist()==[-3.]*9
    equal_trees(out,heads.forward_resource(latent,x))
    assert not any(isinstance(m,(torch.nn.Softmax,torch.nn.LogSoftmax,torch.nn.Sigmoid)) for m in heads.modules())


def reroute_structural(inputs,order):
    """Consistent current-domain permutation/filter, including argument routes."""
    x=copy.deepcopy(inputs);old=inputs['structural'];s=x['structural'];order=torch.tensor(order,dtype=torch.int64)
    inverse=torch.full((len(old['initiator']),),-1,dtype=torch.int64);inverse[order]=torch.arange(len(order))
    for key in ('initiator','action','peer','connector_a','connector_b'):s[key]=old[key][order]
    counts=torch.bincount(s['initiator'],minlength=len(x['module_x']))
    s['group_ptr']=torch.cat((torch.zeros(1,dtype=torch.int64),counts.cumsum(0)))
    def ragged(ptr,names,selected):
        indices=[];new_ptr=[0]
        for i in selected.tolist():
            a,b=old[ptr][i:i+2].tolist();indices.extend(range(a,b));new_ptr.append(len(indices))
        indices=torch.tensor(indices,dtype=torch.int64);s[ptr]=torch.tensor(new_ptr,dtype=torch.int64)
        for name in names:s[name]=old[name][indices]
    ragged('scope_ptr',('scope_members','scope_required'),order)
    args=torch.nonzero(inverse[old['argument_binding']]>=0,as_tuple=False).flatten()
    for key in ('argument_parameter','argument_kind','argument_bound_ref'):s[key]=old[key][args]
    s['argument_binding']=inverse[old['argument_binding'][args]]
    ragged('argument_domain_ptr',('argument_domain_index','argument_domain_required'),args)
    return x


def test_structural_binding_logits_preserve_initiator_groups_and_ordered_roles():
    api();sample,kw=fixture();x=tensorize(sample,kw)['inputs'];model=model_for(kw);heads=heads_for(kw)
    out=heads.forward_structural(model.encode(x),x)
    assert out['candidate_logits'].shape==(51,) and out['candidate_logits'].dtype==torch.float32
    assert out['group_ptr'].tolist()==[0,17,34,51]
    s=x['structural'];b=next(i for i in range(51) if s['peer'][i]>=0)
    reverse=next(i for i in range(51) if s['action'][i]==s['action'][b]
        and s['connector_a'][i]==s['connector_b'][b] and s['connector_b'][i]==s['connector_a'][b])
    assert not torch.allclose(out['candidate_logits'][b],out['candidate_logits'][reverse])


def test_structural_candidate_order_equivariance_and_other_group_count_independence():
    api();sample,kw=fixture();x=tensorize(sample,kw)['inputs'];model=model_for(kw);heads=heads_for(kw)
    h=model.encode(x);before=heads.forward_structural(h,x)
    order=[i for group in range(3) for i in range(17*group+16,17*group-1,-1)]
    changed=reroute_structural(x,order)
    after=heads.forward_structural(h,changed)
    torch.testing.assert_close(after['candidate_logits'],before['candidate_logits'][order])
    changed=reroute_structural(x,list(range(1,51)))
    filtered=heads.forward_structural(h,changed)
    torch.testing.assert_close(before['candidate_logits'][17:],filtered['candidate_logits'][16:])


def test_structural_scores_do_not_read_labels_or_regenerate_candidates(monkeypatch):
    api();sample,kw=fixture();empty=replace(sample,supervision=c.SupervisionBundle(),
        audit=replace(sample.audit,provenance='stage/FSM/future'))
    a,b=(tensorize(s,kw) for s in (sample,empty));model=model_for(kw);heads=heads_for(kw)
    from drl.IL.bc_v1 import candidates
    def forbidden(*a,**kw):raise AssertionError('H2 must not enumerate candidates')
    monkeypatch.setattr(candidates,'enumerate_candidates',forbidden)
    equal_trees(heads.forward_structural(model.encode(a['inputs']),a['inputs']),
        heads.forward_structural(model.encode(b['inputs']),b['inputs']))
    with torch.no_grad():
        heads.structural_scorer[-1].weight.zero_();heads.structural_scorer[-1].bias.fill_(3.)
    assert heads.forward_structural(model.encode(a['inputs']),a['inputs'])['candidate_logits'].tolist()==[3.]*51


def test_structural_batch_isolation():
    api();a,kw=fixture(2);b,_=fixture(3);x=collate_policy_samples([tensorize(a,kw),tensorize(b,kw)])['inputs']
    model=model_for(kw);heads=heads_for(kw);before=heads.forward_structural(model.encode(x),x)
    changed=copy.deepcopy(x);changed['module_x'][0,0]+=2
    after=heads.forward_structural(model.encode(changed),changed)
    boundary=int(x['structural']['group_ptr'][2])
    torch.testing.assert_close(before['candidate_logits'][boundary:],after['candidate_logits'][boundary:])
    assert not torch.allclose(before['candidate_logits'][:boundary],after['candidate_logits'][:boundary])


def test_endpoint_free_settle_has_duration_and_actual_membership_domain():
    api();from drl.IL.tests.bc_v1.g2_fixtures import with_structural
    sample,kw=fixture(5,0);sample=with_structural(sample,skill=True)
    batch=tensorize(sample,kw);x=batch['inputs'];model=model_for(kw);heads=heads_for(kw)
    assert x['connector_x'].shape[0]==0
    out=heads.forward_structural(model.encode(x),x)
    assert out['candidate_logits'].shape==(5,) and torch.isfinite(out['candidate_logits']).all()
    numeric=out['arguments']['numeric'];members=out['arguments']['reference_set']
    assert numeric['value'].shape==(5,) and numeric['ptr'].tolist()==[0,1,2,3,4,5]
    assert members['logits'].shape==(25,) and members['ptr'].tolist()==[0,5,10,15,20,25]
    assert members['required_domain_index'].shape==(0,)
    assert next(b.size for b in kw['normalizers'].argument_bindings if b.name=='members')==1
    assert members['domain_index'].numel()==25  # Domain is N, not F2's binding size.


def test_settle_raw_duration_and_membership_are_not_clamped_or_normalized():
    api();sample,kw=fixture(3,0);x=tensorize(sample,kw)['inputs'];model=model_for(kw);heads=heads_for(kw)
    with torch.no_grad():
        heads.numeric_decoder[-1].weight.zero_();heads.numeric_decoder[-1].bias.fill_(-2.5)
        heads.reference_set_scorer[-1].weight.zero_();heads.reference_set_scorer[-1].bias.fill_(4.)
    out=heads.forward_structural(model.encode(x),x)['arguments']
    assert out['numeric']['value'].tolist()==[-2.5]*3
    assert out['reference_set']['logits'].tolist()==[4.]*9


def test_settle_required_argument_members_remain_fixed_without_a_prediction():
    api();sample,kw=fixture(3,0);x=tensorize(sample,kw)['inputs'];s=x['structural']
    # Exercise the published required-member domain flag, without a label.
    u=next(i for i,k in enumerate(s['argument_kind'].tolist()) if k==6)
    start=int(s['argument_domain_ptr'][u]);s['argument_domain_required'][start]=True
    out=heads_for(kw).forward_structural(model_for(kw).encode(x),x)['arguments']['reference_set']
    assert out['required_domain_index'].tolist()==[start]
    assert start not in out['domain_index'].tolist()
    assert out['logits'].numel()==8


def enriched_fixture():
    from drl.IL.bc_v1.candidates import enumerate_candidates
    sample,kw=fixture();spec=kw['spec']
    extra=(c.ParameterSpec('choose_module','module_ref',module_type_id='kit'),
        c.ParameterSpec('choose_connector','connector_ref',module_type_id='kit'),
        c.ParameterSpec('extra_connectors','connector_refs',module_type_id='kit'),
        c.ParameterSpec('decide','boolean'),c.ParameterSpec('ignored','boolean',learnable=False),
        c.ParameterSpec('adjust','vector','m',size=2),c.ParameterSpec('select','category',categories=('a','b','c')))
    spec=replace(spec,actions=tuple(replace(a,parameters=a.parameters+extra) if a.kind=='skill' else a for a in spec.actions))
    kw={**kw,'spec':spec,'normalizers':replace(kw['normalizers'],metadata_items=tuple(sorted({
        **kw['normalizers'].metadata,'robot_spec_fingerprint':c.robot_spec_fingerprint(spec)}.items())))}
    sample=replace(sample,inputs=replace(sample.inputs,candidates=enumerate_candidates(sample.inputs.policy.graph,spec,kw['feature_schema'])))
    return sample,kw


def test_typed_category_boolean_and_unbound_reference_domains():
    api();sample,kw=enriched_fixture();x=tensorize(sample,kw)['inputs'];heads=heads_for(kw)
    args=heads.forward_structural(model_for(kw).encode(x),x)['arguments']
    assert len(args['category']['argument'])==51  # 48 pair turn + 3 skill select.
    assert args['category']['logits'].shape==(105,)
    assert (args['category']['ptr'][1:]-args['category']['ptr'][:-1]).tolist().count(3)==3
    assert args['boolean']['logits'].shape==(3,)
    assert args['reference']['logits'].shape==(27,)  # 3 initiators x (3 modules + 6 connectors).
    assert len(args['reference']['argument'])==6
    assert args['numeric']['value'].shape==(9,)  # Three duration scalars + three 2-vectors.
    assert args['reference_set']['logits'].shape==(27,)  # 3 x (module set 3 + connector set 6).
    for name in ('reference','reference_set'):
        records=args[name]
        assert records['ptr'][-1]==len(records['logits'])==len(records['domain_index'])


def test_nonlearnable_and_bound_connector_args_have_no_prediction_heads():
    api();sample,kw=fixture();x=tensorize(sample,kw)['inputs'];heads=heads_for(kw)
    args=heads.forward_structural(model_for(kw).encode(x),x)['arguments']
    assert heads.boolean_scorer is None and heads.reference_scorer is None
    assert args['boolean']['logits'].shape==(0,) and args['reference']['logits'].shape==(0,)
    predicted={int(i) for family in args.values() for i in family['argument']}
    s=x['structural']
    for i,pid in enumerate(s['argument_parameter'].tolist()):
        p=heads.parameters_catalog[pid]
        if not p.learnable or int(s['argument_bound_ref'][i])!=-1:assert i not in predicted


def test_argument_row_permutation_preserves_semantic_predictions():
    api();sample,kw=enriched_fixture();x=tensorize(sample,kw)['inputs'];model=model_for(kw);heads=heads_for(kw);h=model.encode(x)
    before=heads.forward_structural(h,x)['arguments'];changed=copy.deepcopy(x);s=changed['structural'];old=x['structural']
    order=torch.arange(len(s['argument_binding'])-1,-1,-1)
    for key in ('argument_binding','argument_parameter','argument_kind','argument_bound_ref'):s[key]=old[key][order]
    domain=[];ptr=[0]
    for u in order.tolist():
        a,b=old['argument_domain_ptr'][u:u+2].tolist();domain.extend(range(a,b));ptr.append(len(domain))
    s['argument_domain_ptr']=torch.tensor(ptr)
    for key in ('argument_domain_index','argument_domain_required'):s[key]=old[key][domain]
    after=heads.forward_structural(h,changed)['arguments']
    for family in ('category','numeric','reference','reference_set'):
        values='value' if family=='numeric' else 'logits'
        for row,u in enumerate(after[family]['argument'].tolist()):
            original_u=int(order[u]);prior=before[family]['argument'].tolist().index(original_u)
            a,b=after[family]['ptr'][row:row+2].tolist();c_,d=before[family]['ptr'][prior:prior+2].tolist()
            torch.testing.assert_close(after[family][values][a:b],before[family][values][c_:d])
    for row,u in enumerate(after['boolean']['argument'].tolist()):
        prior=before['boolean']['argument'].tolist().index(int(order[u]))
        torch.testing.assert_close(after['boolean']['logits'][row],before['boolean']['logits'][prior])


def test_categorical_resource_codec_returns_raw_category_logits():
    api();from drl.IL.bc_v1.candidates import enumerate_candidates
    sample,kw=fixture();spec=kw['spec'];mt=spec.module_types[0];res=mt.resources[0]
    modes=tuple(replace(m,codec='categorical',units=('1',),categories=('low','mid','high')) if m.mode_id=='aim' else m for m in res.modes)
    actions=tuple(replace(a,parameters=(c.ParameterSpec('level','category',categories=('low','mid','high'),
        module_type_id='kit',resource_id='rotor'),)) if a.action_id=='aim' else a for a in spec.actions)
    spec=replace(spec,module_types=(replace(mt,resources=(replace(res,modes=modes),)),),actions=actions)
    kw={**kw,'spec':spec,'normalizers':replace(kw['normalizers'],metadata_items=tuple(sorted({
        **kw['normalizers'].metadata,'robot_spec_fingerprint':c.robot_spec_fingerprint(spec)}.items())))}
    sample=replace(sample,inputs=replace(sample.inputs,candidates=enumerate_candidates(sample.inputs.policy.graph,spec,kw['feature_schema'])))
    x=tensorize(sample,kw)['inputs'];heads=heads_for(kw)
    out=heads.forward_resource(model_for(kw).encode(x),x)
    assert out['payload'].shape==(6,) and out['payload_candidate'].tolist()==[0,2,4]
    assert out['categorical']['logits'].shape==(9,) and out['categorical']['candidate'].tolist()==[1,3,5]
    assert out['categorical']['ptr'].tolist()==[0,3,6,9]


def test_participant_scope_is_independent_of_argument_set_membership():
    from drl.IL.tests.bc_v1.g2_fixtures import with_structural
    sample,kw=fixture(5,0);sample=with_structural(sample,skill=True)
    targets=tuple(replace(t,participants=tuple(m.ref for m in sample.inputs.policy.graph.modules))
        for t in sample.supervision.structural_targets)
    changed=replace(sample,supervision=replace(sample.supervision,structural_targets=targets))
    a,b=(tensorize(s,kw) for s in (sample,changed));equal_trees(a['inputs'],b['inputs'])
    assert not torch.equal(a['supervision']['structural']['scope_membership_value'],
        b['supervision']['structural']['scope_membership_value'])
    heads=heads_for(kw);h=model_for(kw).encode(a['inputs'])
    before=heads.forward_structural(h,a['inputs']);after=heads.forward_structural(h,b['inputs']);equal_trees(before,after)
    scope=before['scope'];members=before['arguments']['reference_set']
    assert scope['logits'].shape==(20,) and members['logits'].shape==(25,)
    assert scope['ptr'].tolist()==[0,4,8,12,16,20]
    assert scope['required_ptr'].tolist()==[0,1,2,3,4,5]
    assert scope['binding'].tolist()==list(range(5))
    assert set(scope['domain_index'].tolist()).isdisjoint(scope['required_domain_index'].tolist())
    with torch.no_grad():
        heads.scope_scorer[-1].weight.zero_();heads.scope_scorer[-1].bias.fill_(-4.)
        heads.reference_set_scorer[-1].weight.zero_();heads.reference_set_scorer[-1].bias.fill_(3.)
    result=heads.forward_structural(h,a['inputs'])
    assert result['scope']['logits'].tolist()==[-4.]*20
    assert result['arguments']['reference_set']['logits'].tolist()==[3.]*25


def test_scope_determined_by_binding_does_not_get_duplicate_predictions():
    sample,kw=fixture();x=tensorize(sample,kw)['inputs']
    # Keep only pairs: all participant members are required by the binding.
    x=reroute_structural(x,torch.nonzero(x['structural']['peer']>=0,as_tuple=False).flatten().tolist())
    result=heads_for(kw).forward_structural(model_for(kw).encode(x),x)['scope']
    assert result['logits'].shape==(0,) and result['binding'].shape==(0,)
    assert result['ptr'].tolist()==[0] and result['required_ptr'].tolist()==[0]


def test_scope_logits_follow_candidate_permutation():
    sample,kw=fixture();x=tensorize(sample,kw)['inputs'];heads=heads_for(kw);h=model_for(kw).encode(x)
    old=heads.forward_structural(h,x)['scope'];order=[];ptr=x['structural']['group_ptr']
    for start,end in zip(ptr[:-1].tolist(),ptr[1:].tolist()):order.extend(range(end-1,start-1,-1))
    changed=reroute_structural(x,order);new=heads.forward_structural(h,changed)['scope']
    for row,binding in enumerate(new['binding'].tolist()):
        prior=old['binding'].tolist().index(order[binding])
        a,b=new['ptr'][row:row+2].tolist();c_,d=old['ptr'][prior:prior+2].tolist()
        torch.testing.assert_close(new['logits'][a:b],old['logits'][c_:d])


def complete_model(kw):
    from drl.IL.bc_v1.model import BCGNN,load_model_config
    torch.manual_seed(2718)
    return BCGNN(kw['spec'],kw['feature_schema'],load_model_config(ROOT/'configs/bc_v1/gnn_v1.json'),
        action_head_config=api().load_action_head_config(ROOT/'configs/bc_v1/action_heads_v1.json')).eval()


def floats(tree):
    for value in tree.values():
        if isinstance(value,dict):yield from floats(value)
        elif value.dtype==torch.float32:yield value


def test_full_forward_preserves_independent_h1_and_latent_input_purity():
    sample,kw=enriched_fixture();x=tensorize(sample,kw)['inputs'];snapshot=copy.deepcopy(x)
    model=complete_model(kw);h=model.encode(x)
    equal_trees(h,model_for(kw).encode(x))  # Optional H2 did not reinitialize H1.
    before=copy.deepcopy({k:v.detach() for k,v in h.items()})
    out=model.forward_heads(h,x);equal_trees(out,model(x))
    equal_trees(before,{k:v.detach() for k,v in h.items()});equal_trees(snapshot,x)
    assert set(out)=={'resource','structural'}
    assert all(torch.isfinite(v).all() for v in floats(out))
    equal_trees(out,model(x))


def test_complete_forward_is_independent_of_all_supervision_and_audit():
    from drl.IL.tests.bc_v1.g2_fixtures import with_structural
    sample,kw=fixture();batch=tensorize(with_structural(sample),kw);changed=copy.deepcopy(batch)
    # Expert candidate/mode, numeric/category/member labels and provenance are
    # routing/supervision only. Normal forward receives neither container.
    def alter(tree):
        for value in tree.values():
            if isinstance(value,dict):alter(value)
            elif isinstance(value,torch.Tensor):
                if value.dtype==torch.bool:value.logical_not_()
                else:value.add_(99)
    alter(changed['supervision']);changed['routing']={'stage':'secret','audit':'changed','G_t+1':object()}
    model=complete_model(kw);equal_trees(model(batch['inputs']),model(changed['inputs']))
    with pytest.raises(ValueError,match='only G2'):model(batch)


@pytest.mark.parametrize('forbidden',['stage','expert_candidate','control_state','G_t+1','audit'])
def test_heads_reject_widened_input_boundary(forbidden):
    sample,kw=fixture();x=tensorize(sample,kw)['inputs'];heads=heads_for(kw);h=model_for(kw).encode(x)
    with pytest.raises(ValueError,match='only G2'):heads(h,{**x,forbidden:object()})


@pytest.mark.parametrize('n,ports',[(1,0),(3,0)])
def test_full_forward_handles_empty_families_and_unsupervised_candidates(n,ports):
    sample,kw=fixture(n,ports);batch=tensorize(sample,kw);model=complete_model(kw);out=model(batch['inputs'])
    assert len(batch['supervision']['structural']['candidate'])==0
    assert out['structural']['candidate_logits'].shape==(n,)  # Endpoint-free only, no connectors.
    assert out['structural']['group_ptr'].shape==(n+1,)
    assert out['structural']['arguments']['category']['logits'].shape==(0,)
    assert out['structural']['arguments']['reference']['ptr'].tolist()==[0]
    assert all(torch.isfinite(v).all() for v in floats(out))


def test_resource_only_robot_has_zero_structural_candidates():
    from drl.IL.bc_v1.candidates import enumerate_candidates
    sample,kw=fixture();spec=replace(kw['spec'],actions=tuple(a for a in kw['spec'].actions if a.scope=='resource'))
    kw={**kw,'spec':spec,'normalizers':replace(kw['normalizers'],metadata_items=tuple(sorted({
        **kw['normalizers'].metadata,'robot_spec_fingerprint':c.robot_spec_fingerprint(spec)}.items())))}
    sample=replace(sample,inputs=replace(sample.inputs,candidates=enumerate_candidates(sample.inputs.policy.graph,spec,kw['feature_schema'])))
    out=complete_model(kw)(tensorize(sample,kw)['inputs'])
    assert out['structural']['candidate_logits'].shape==(0,)
    assert out['structural']['group_ptr'].tolist()==[0,0,0,0]
    assert all(v.numel()==0 for v in floats(out['structural']))
    assert out['resource']['payload'].shape==(9,)


def test_h1_without_heads_remains_encode_only_and_head_config_is_closed():
    sample,kw=fixture();x=tensorize(sample,kw)['inputs'];model=model_for(kw)
    assert model.action_heads is None
    with pytest.raises(ValueError,match='not configured'):model.forward_heads(model.encode(x),x)
    config=api().load_action_head_config(ROOT/'configs/bc_v1/action_heads_v1.json')
    with pytest.raises(ValueError):api().ActionHeads(kw['spec'],model.config,{**config,'dropout':.2})


def test_raw_prediction_backward_reaches_heads_and_h1_without_bc_loss():
    sample,kw=enriched_fixture();x=tensorize(sample,kw)['inputs'];model=complete_model(kw);h=model.encode(x)
    for name in ('module_h','resource_h','connector_h','graph_h','context_h'):h[name].retain_grad()
    out=model.forward_heads(h,x);values=[v for v in floats(out) if v.numel()]
    synthetic=sum(v.square().mean() for v in values)
    assert torch.isfinite(synthetic);synthetic.backward()
    for name in ('module_h','resource_h','connector_h','graph_h','context_h'):
        assert h[name].grad is not None and torch.isfinite(h[name].grad).all()
        assert h[name].grad.abs().sum()>0
    for layer in (model.action_heads.payload_decoder,model.action_heads.mode_scorer,
            model.action_heads.structural_scorer,model.action_heads.argument_category_scorer,
            model.action_heads.reference_set_scorer,model.action_heads.reference_scorer,
            model.action_heads.boolean_scorer,model.action_heads.scope_scorer,
            model.resource_context,model.module_fusion,model.encoders['resource']):
        grads=[p.grad for p in layer.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads)>0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_complete_forward_batch_isolation_covers_every_argument_and_scope():
    a,kw=enriched_fixture();b=copy.deepcopy(a)
    x=collate_policy_samples([tensorize(a,kw),tensorize(b,kw)])['inputs'];model=complete_model(kw)
    before=model(x);changed=copy.deepcopy(x);changed['module_x'][0,0]+=2;after=model(changed)
    assert not torch.allclose(before['resource']['payload'][:9],after['resource']['payload'][:9])
    for field,start in [('mode_logits',6),('payload',9)]:
        torch.testing.assert_close(before['resource'][field][start:],after['resource'][field][start:])
    s=x['structural'];q=int(s['group_ptr'][3]);torch.testing.assert_close(
        before['structural']['candidate_logits'][q:],after['structural']['candidate_logits'][q:])
    for family,old in before['structural']['arguments'].items():
        new=after['structural']['arguments'][family];rows=torch.nonzero(s['argument_binding'][old['argument']]>=q,as_tuple=False).flatten()
        if family=='boolean':torch.testing.assert_close(old['logits'][rows],new['logits'][rows]);continue
        value='value' if family=='numeric' else 'logits'
        for row in rows.tolist():
            start,end=old['ptr'][row:row+2].tolist();torch.testing.assert_close(old[value][start:end],new[value][start:end])
    old=before['structural']['scope'];new=after['structural']['scope']
    for row in torch.nonzero(old['binding']>=q,as_tuple=False).flatten().tolist():
        start,end=old['ptr'][row:row+2].tolist();torch.testing.assert_close(old['logits'][start:end],new['logits'][start:end])


def test_physical_id_renaming_cannot_change_complete_predictions():
    from dataclasses import fields,is_dataclass
    sample,kw=enriched_fixture()
    def rename(value):
        if isinstance(value,c.ModuleRef):return c.ModuleRef('arbitrary-'+value.module_id)
        if is_dataclass(value):return replace(value,**{f.name:rename(getattr(value,f.name)) for f in fields(value)})
        if isinstance(value,tuple):return tuple(rename(v) for v in value)
        if isinstance(value,frozenset):return frozenset(rename(v) for v in value)
        return value
    model=complete_model(kw);equal_trees(model(tensorize(sample,kw)['inputs']),model(tensorize(rename(sample),kw)['inputs']))


def semantic_predictions(prediction,x,sample):
    """Decode routing only in the test, to compare consistently permuted IDs."""
    refs={name:[r.ref for r in getattr(sample.inputs.features,name)] for name in ('modules','connectors','resources')}
    s=x['structural'];r=x['resource_candidates'];out={}
    def binding(q):
        values=[int(s[k][q]) for k in ('initiator','action','peer','connector_a','connector_b')]
        i,operation,p,a,b=values
        return (refs['modules'][i],operation,refs['modules'][p] if p>=0 else None,
            refs['connectors'][a] if a>=0 else None,refs['connectors'][b] if b>=0 else None)
    resource=prediction['resource']
    for row,q in enumerate(resource['mode_candidate'].tolist()):
        out[('mode',refs['resources'][int(r['resource'][q])],int(r['mode'][q]))]=resource['mode_logits'][row]
    for family,records in [('payload',resource),('categorical',resource['categorical'])]:
        queries=records['payload_candidate'] if family=='payload' else records['candidate']
        ptr=records['payload_ptr'] if family=='payload' else records['ptr'];values=records['payload'] if family=='payload' else records['logits']
        for row,q in enumerate(queries.tolist()):
            start,end=ptr[row:row+2].tolist()
            for j,value in enumerate(values[start:end]):out[(family,refs['resources'][int(r['resource'][q])],int(r['mode'][q]),j)]=value
    structural=prediction['structural']
    for q,value in enumerate(structural['candidate_logits']):out[('candidate',binding(q))]=value
    for family,records in structural['arguments'].items():
        for row,u in enumerate(records['argument'].tolist()):
            key=(family,binding(int(s['argument_binding'][u])),int(s['argument_parameter'][u]))
            if family=='boolean':out[key]=records['logits'][row];continue
            start,end=records['ptr'][row:row+2].tolist();values=records['value' if family=='numeric' else 'logits'][start:end]
            for j,value in enumerate(values):
                option=j
                if family in ('reference','reference_set'):
                    d=int(records['domain_index'][start+j]);index=int(s['argument_domain_index'][d])
                    option=refs['connectors' if int(s['argument_kind'][u]) in (5,7) else 'modules'][index]
                out[key+(option,)]=value
    records=structural['scope']
    for row,q in enumerate(records['binding'].tolist()):
        start,end=records['ptr'][row:row+2].tolist()
        for d,value in zip(records['domain_index'][start:end],records['logits'][start:end]):
            out[('scope',binding(q),refs['modules'][int(s['scope_members'][d])])]=value
    return out


def test_complete_entity_permutation_is_equivariant_for_all_prediction_families():
    from drl.IL.tests.bc_v1.test_model import permuted_sample
    sample,kw=enriched_fixture();changed,_=permuted_sample(sample,19);model=complete_model(kw)
    a,b=(tensorize(s,kw)['inputs'] for s in (sample,changed))
    before=semantic_predictions(model(a),a,sample);after=semantic_predictions(model(b),b,changed)
    assert before.keys()==after.keys()
    for key in before:torch.testing.assert_close(before[key],after[key],atol=3e-6,rtol=3e-6)


def test_reordered_connector_parameters_preserve_unbound_learnable_pointer():
    from drl.IL.bc_v1.candidates import enumerate_candidates
    sample,kw=fixture();spec=kw['spec'];align=next(a for a in spec.actions if a.action_id=='align_faces')
    # G2 canonically binds endpoint_a/b, even if z_unbound is declared first.
    a,b,turn=align.parameters
    parameters=(replace(a,name='z_unbound'),replace(a,learnable=False),replace(b,learnable=False),turn)
    spec=replace(spec,actions=tuple(replace(action,parameters=parameters) if action==align else action for action in spec.actions))
    kw={**kw,'spec':spec,'normalizers':replace(kw['normalizers'],metadata_items=tuple(sorted({
        **kw['normalizers'].metadata,'robot_spec_fingerprint':c.robot_spec_fingerprint(spec)}.items())))}
    sample=replace(sample,inputs=replace(sample.inputs,candidates=enumerate_candidates(sample.inputs.policy.graph,spec,kw['feature_schema'])))
    x=tensorize(sample,kw)['inputs'];heads=heads_for(kw);h=model_for(kw).encode(x)
    pid=heads.parameter_catalog.index(('align_faces','z_unbound'))
    unbound=torch.nonzero(x['structural']['argument_parameter']==pid,as_tuple=False).flatten()
    assert (x['structural']['argument_bound_ref'][unbound]==-1).all()
    out=heads.forward_structural(h,x)['arguments']['reference']
    assert out['argument'].tolist()==unbound.tolist() and out['logits'].shape==(144,)
    assert heads.reference_scorer is not None
    sorted_spec=replace(spec,actions=tuple(replace(a,parameters=tuple(sorted(a.parameters,key=lambda p:p.name))) for a in spec.actions))
    reordered=heads_for({**kw,'spec':sorted_spec})
    equal_trees(heads.forward_structural(h,x),reordered.forward_structural(h,x))
