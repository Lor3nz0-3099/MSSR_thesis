"""H1 behavioral gates over real G2 tensors, with generic synthetic robots."""
import copy
import importlib
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.bc_v1.batching import collate_policy_samples
from drl.IL.tests.bc_v1.g2_fixtures import fixture, tensorize

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(autouse=True, scope='module')
def small_cpu_thread_budget():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.model'), 'Missing H1 entity encoders'
    return importlib.import_module('drl.IL.bc_v1.model')


def make(n=3, ports=2, *, environment=True, dynamic=False, connected=False):
    sample, kw = fixture(n, ports, geometry=environment, attached=connected)
    if dynamic:
        from drl.IL.bc_v1.features import transform_inputs, PhysicalFrameConfig
        from drl.IL.bc_v1.feature_schema import extract_features
        policy = replace(sample.inputs.policy,
            context=c.TaskContext('button', (c.NumericFeature('world.button.center_xyz_m',(3.,0.,0.)),)),
            observation=c.ObservationBundle((c.NumericFeature('button.depression_m',(.02,)),)))
        features = extract_features(policy, transform_inputs(policy, PhysicalFrameConfig()), kw['spec'], kw['feature_schema'])
        sample = replace(sample, inputs=replace(sample.inputs, policy=policy, features=features))
    return tensorize(sample, kw), kw


def model_for(kw):
    mod = api()
    torch.manual_seed(2718)
    return mod.BCGNN(kw['spec'], kw['feature_schema'], mod.load_model_config(ROOT/'configs/bc_v1/gnn_v1.json')).eval()


def equal_trees(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal_trees(a[k],b[k])
    else:assert torch.equal(a,b)


@pytest.mark.parametrize('n',[1,3,11])
def test_local_encoder_shapes_variable_entities_and_finite_outputs(n):
    batch,kw=make(n);model=model_for(kw);x=batch['inputs']
    output=model.encode_local(x)
    for name,count,width in [('module_local_h',n,128),('connector_local_h',2*n,128),
            ('resource_local_h',n,128),('coordinate_h',2*n,64),('connection_h',0,64),
            ('relation_h',len(x['relation_x']),64),('environment_h',int(x['environment_ptr'][-1]),64),
            ('task_h',1,8),('context_h',1,64),('observation_h',1,64)]:
        assert output[name].shape==(count,width)
        assert output[name].dtype==torch.float32 and torch.isfinite(output[name]).all()


def test_task_semantic_geometry_and_scope_are_distinguished():
    batch,kw=make();model=model_for(kw);x=batch['inputs']
    a=model.encode_local(x)
    swapped=copy.deepcopy(x)
    slots=list(swapped['environment'])
    v=swapped['environment'][slots[0]]['x'][0,0].clone()
    swapped['environment'][slots[0]]['x'][0,0]=swapped['environment'][slots[1]]['x'][0,0]
    swapped['environment'][slots[1]]['x'][0,0]=v
    assert not torch.allclose(a['context_h'],model.encode_local(swapped)['context_h'])
    changed=copy.deepcopy(x);changed['task_one_hot'].zero_();changed['task_one_hot'][0,4]=1
    assert not torch.allclose(a['context_h'],model.encode_local(changed)['context_h'])
    dyn,_=make(dynamic=True)
    output=model.encode_local(dyn['inputs'])
    assert output['observation_present'].tolist()==[True]
    assert output['context_present'].tolist()==[True]
    assert output['observation_h'].abs().sum()>0


def test_empty_observation_stays_zero_even_after_bias_changes():
    batch,kw=make(environment=False);model=model_for(kw)
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer,torch.nn.Linear):layer.bias.fill_(1.)
    result=model.encode_local(batch['inputs'])
    assert torch.equal(result['observation_h'],torch.zeros(1,64))
    assert result['observation_present'].tolist()==[False]
    assert result['context_present'].tolist()==[False]


def test_eval_deterministic_stateless_and_inputs_unmodified():
    batch,kw=make(dynamic=True);model=model_for(kw);before=copy.deepcopy(batch['inputs'])
    a=model.encode_local(batch['inputs'])
    model.encode_local(make(5)[0]['inputs'])
    equal_trees(a,model.encode_local(batch['inputs']));equal_trees(before,batch['inputs'])
    forbidden=(torch.nn.RNN,torch.nn.LSTM,torch.nn.GRU,torch.nn.Transformer,torch.nn.MultiheadAttention,torch.nn.BatchNorm1d)
    assert not any(isinstance(m,forbidden) for m in model.modules())
    assert any(isinstance(m,torch.nn.Dropout) and m.p==.1 for m in model.modules())


def test_closed_config_and_causal_input_boundary():
    batch,kw=make();mod=api();cfg=mod.load_model_config(ROOT/'configs/bc_v1/gnn_v1.json')
    for change in ({'unknown':True},{'message_passing_layers':2},{'dropout':1.},{'hidden_dim':True},
                   {'activation':'relu'},{'normalization':'batchnorm'},{'use_residual':False}):
        with pytest.raises(ValueError):mod.BCGNN(kw['spec'],kw['feature_schema'],{**cfg,**change})
    model=model_for(kw)
    for key in ('stage','configuration_goal','control_state','current_primitive','supervision','audit','future'):
        with pytest.raises(ValueError):model.encode_local({**batch['inputs'],key:torch.zeros(1)})
    with pytest.raises(ValueError):model.encode_local(batch)


def test_coordinate_information_reaches_only_owning_resource():
    batch,kw=make();model=model_for(kw);x=batch['inputs'];before=model.encode_local(x)
    changed=copy.deepcopy(x);changed['coordinate_x'][0,0]+=1
    after=model.encode_local(changed)
    assert not torch.allclose(before['resource_local_h'][0],after['resource_local_h'][0])
    torch.testing.assert_close(before['resource_local_h'][1:],after['resource_local_h'][1:])
    assert not torch.allclose(before['coordinate_h'][0],after['coordinate_h'][0])


def test_resource_and_connector_information_reaches_owning_module():
    batch,kw=make();model=model_for(kw);x=batch['inputs'];local=model.encode_local(x)
    before=model.initialize_modules(x,local)
    for family in ('resource','connector'):
        changed={**local,family+'_local_h':local[family+'_local_h'].clone()}
        changed[family+'_local_h'][0,0]+=3
        after=model.initialize_modules(x,changed)
        assert not torch.allclose(before[0],after[0])
        torch.testing.assert_close(before[1:],after[1:])


def test_child_pooling_is_order_invariant_with_correct_ownership():
    batch,kw=make();model=model_for(kw);x=batch['inputs']
    before=model.initialize_modules(x,model.encode_local(x))
    changed=copy.deepcopy(x)
    for family in ('connector','resource','coordinate'):
        perm=torch.arange(len(x[family+'_x'])-1,-1,-1);inverse=torch.argsort(perm)
        for suffix in ('x','validity'):changed[family+'_'+suffix]=changed[family+'_'+suffix][perm]
        parent={'connector':'connector_to_module','resource':'resource_to_module','coordinate':'coordinate_to_resource'}[family]
        changed[parent]=changed[parent][perm]
        if family=='resource':changed['coordinate_to_resource']=inverse[changed['coordinate_to_resource']]
        if family=='coordinate':changed['coordinate_slot']=changed['coordinate_slot'][perm]
    after=model.initialize_modules(changed,model.encode_local(changed))
    torch.testing.assert_close(before,after,atol=2e-6,rtol=2e-6)


def test_empty_resource_connector_children_have_finite_module_initialization():
    batch,kw=make(environment=False);model=model_for(kw);x=copy.deepcopy(batch['inputs'])
    for family in ('connector','resource','coordinate'):
        for suffix in ('x','validity'):x[family+'_'+suffix]=x[family+'_'+suffix][:0]
    for name in ('connector_to_module','resource_to_module','coordinate_to_resource','coordinate_slot'):x[name]=x[name][:0]
    local=model.encode_local(x);h=model.initialize_modules(x,local)
    assert h.shape==(3,128) and torch.isfinite(h).all()
    assert local['coordinate_h'].shape==(0,64) and local['resource_local_h'].shape==(0,128)


def chain(n=6, edges=((0,1),(1,2),(2,3),(3,4)), *, dynamic=False, typed=False):
    from drl.IL.bc_v1.features import transform_inputs,PhysicalFrameConfig
    from drl.IL.bc_v1.feature_schema import extract_features
    from drl.IL.bc_v1.candidates import enumerate_candidates
    sample,kw=fixture(n,2,geometry=False);g=sample.inputs.policy.graph
    connections=tuple(c.ConnectionState(g.connectors[2*a+1].ref,g.connectors[2*b].ref,'current_connection',features=(
        c.NumericFeature('relative_position_m',(1.,.2,0.)),
        c.NumericFeature('relative_orientation_xyzw',(0.,0.,0.,1.)))) for a,b in edges)
    partners={}
    for edge in connections:partners.update({edge.endpoint_a:edge.endpoint_b,edge.endpoint_b:edge.endpoint_a})
    g=replace(g,connections=connections,connectors=tuple(replace(p,status='connected',connected_to=partners[p.ref])
        if p.ref in partners else p for p in g.connectors))
    policy=replace(sample.inputs.policy,graph=g)
    if dynamic:
        policy=replace(policy,context=c.TaskContext('button',(c.NumericFeature('world.button.center_xyz_m',(2.,0.,0.)),)),
            observation=c.ObservationBundle((c.NumericFeature('button.depression_m',(.02,)),
                c.NumericFeature('world.button.current_center_xyz_m',(2.,.1,0.)))))
    features=extract_features(policy,transform_inputs(policy,PhysicalFrameConfig()),kw['spec'],kw['feature_schema'])
    sample=replace(sample,inputs=replace(sample.inputs,policy=policy,features=features,
        candidates=enumerate_candidates(g,kw['spec'],kw['feature_schema'])))
    return (sample if typed else tensorize(sample,kw)),kw


def test_three_layer_receptive_field_and_disconnected_components():
    batch,kw=chain();model=model_for(kw);x=batch['inputs'];local=model.encode_local(x)
    assert len(model.message_blocks)==3
    a=model.initialize_modules(x,local);b=a.clone();b[0,0]+=2
    for hop,block in enumerate(model.message_blocks,1):
        a=block(a,x,local);b=block(b,x,local)
        assert not torch.allclose(a[hop],b[hop],atol=1e-6,rtol=1e-6),f'{hop}-hop communication missing'
        torch.testing.assert_close(a[hop+1:5],b[hop+1:5],atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(a[5],b[5],atol=1e-6,rtol=1e-6)
    assert torch.isfinite(a).all()


def test_physical_messages_are_bidirectional_with_canonical_endpoint_roles():
    batch,kw=chain(2,((0,1),));model=model_for(kw);x=batch['inputs'];local=model.encode_local(x)
    h=model.initialize_modules(x,local);block=model.message_blocks[0];seen=[]
    hook=block.physical_message.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().clone()))
    a=block(h,x,local);hook.remove()
    assert seen[0][:,-2:].tolist()==[[0.,1.],[1.,0.]]  # receiver B, then receiver A
    torch.testing.assert_close(seen[0][0,-66:-2],local['connection_h'][0])
    torch.testing.assert_close(seen[0][1,-66:-2],local['connection_h'][0])
    for source,target in ((0,1),(1,0)):
        changed=h.clone();changed[source,0]+=2
        assert not torch.allclose(a[target],block(changed,x,local)[target])


def test_environment_relation_messages_are_local_and_separate_from_physical():
    batch,kw=make(dynamic=True);model=model_for(kw);x=batch['inputs'];local=model.encode_local(x)
    assert len(x['relation_x'])>0
    h=model.initialize_modules(x,local);a=model.propagate(x,local,h)
    changed={**local,'relation_h':local['relation_h'].clone()};changed['relation_h'][0,0]+=2
    b=model.propagate(x,changed,h);owner=int(x['relation_to_module'][0])
    assert not torch.allclose(a[owner],b[owner])
    keep=torch.arange(len(h))!=owner;torch.testing.assert_close(a[keep],b[keep])
    assert model.message_blocks[0].physical_message is not model.message_blocks[0].environment_message


def test_zero_edge_propagation_and_cross_graph_edges_fail_closed():
    batch,kw=make(3,ports=0,environment=False);model=model_for(kw);x=batch['inputs'];local=model.encode_local(x)
    result=model.propagate(x,local,model.initialize_modules(x,local))
    assert result.shape==(3,128) and torch.isfinite(result).all()
    a,kw=chain(2,((0,1),));b,_=chain(2,((0,1),));x=collate_policy_samples([a,b])['inputs']
    x['connection_connectors'][0,1]=4
    with pytest.raises(ValueError,match='Cross-graph'):model_for(kw).encode_local(x)


def complete_through_contextualization(model,x):
    local=model.encode_local(x)
    h=model.propagate(x,local,model.initialize_modules(x,local))
    return model.contextualize(x,local,h)


def test_graph_pool_is_mean_max_and_entity_cardinality_does_not_size_parameters():
    a,kw=make(2);b,_=make(5);model=model_for(kw);x=collate_policy_samples([a,b])['inputs']
    local=model.encode_local(x);h=model.propagate(x,local,model.initialize_modules(x,local));seen=[]
    hook=model.graph_projection.register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().clone()))
    out=model.contextualize(x,local,h);hook.remove()
    expected=torch.stack((torch.cat((h[:2].mean(0),h[:2].amax(0))),torch.cat((h[2:].mean(0),h[2:].amax(0)))))
    torch.testing.assert_close(seen[0],expected)
    assert out['graph_h'].shape==(2,128)
    assert out['connector_h'].shape==(14,128) and out['resource_h'].shape==(7,128)
    count=sum(p.numel() for p in model.parameters())
    other,_=make(11,ports=0);complete_through_contextualization(model,other['inputs'])
    assert sum(p.numel() for p in model.parameters())==count


def test_contextual_outputs_and_graph_pool_are_isolated_between_batch_graphs():
    a,kw=chain(2,((0,1),));b,_=make(3,dynamic=True);model=model_for(kw)
    x=collate_policy_samples([a,b])['inputs'];before=complete_through_contextualization(model,x)
    changed=copy.deepcopy(x);changed['module_x'][0,0]+=2
    after=complete_through_contextualization(model,changed)
    for name,start in [('module_h',2),('connector_h',4),('resource_h',2),('graph_h',1)]:
        torch.testing.assert_close(before[name][start:],after[name][start:])
    assert not torch.allclose(before['graph_h'][0],after['graph_h'][0])


def test_context_observation_ownership_and_empty_graph_scope():
    a,kw=make(2,dynamic=True);b,_=make(4,environment=False);model=model_for(kw)
    x=collate_policy_samples([a,b])['inputs'];before=complete_through_contextualization(model,x)
    assert before['observation_present'].tolist()==[True,False]
    assert before['context_present'].tolist()==[True,False]
    assert torch.equal(before['observation_h'][1],torch.zeros(64))
    changed=copy.deepcopy(x)
    for key,row in changed['environment'].items():
        if key.startswith('o_dynamic:'):row['x'][0,0]+=.1
    after=complete_through_contextualization(model,changed)
    assert not torch.allclose(before['observation_h'][0],after['observation_h'][0])
    for name,start in [('module_h',2),('connector_h',4),('resource_h',2),('graph_h',1)]:
        torch.testing.assert_close(before[name][start:],after[name][start:])


def permuted_sample(sample,seed):
    import random
    rng=random.Random(seed)
    fields=('modules','connectors','resources','coordinates','connections','environment','relations')
    permutations={};changes={}
    for field in fields:
        rows=getattr(sample.inputs.features,field);order=list(range(len(rows)));rng.shuffle(order)
        permutations[field]=order;changes[field]=tuple(rows[i] for i in order)
    graph=sample.inputs.policy.graph
    graph=replace(graph,**{field:tuple(getattr(graph,field)[i] for i in permutations[field])
        for field in ('modules','connectors','resources','connections')})
    graph=replace(graph,resources=tuple(replace(r,coordinates=r.coordinates[::-1]) for r in graph.resources))
    # No physical feature extraction here: all cached values/refs retain their meaning.
    changed=replace(sample,inputs=replace(sample.inputs,
        policy=replace(sample.inputs.policy,graph=graph),features=replace(sample.inputs.features,**changes)))
    return changed,permutations


@pytest.mark.parametrize('seed',[7,19])
def test_full_permutation_equivariance_including_entities_edges_and_relations(seed):
    a,kw=chain(3,((0,1),(1,2)),dynamic=True,typed=True)
    b,_=chain(5,((0,1),(1,2),(3,4)),dynamic=True,typed=True)
    pa,order_a=permuted_sample(a,seed);pb,order_b=permuted_sample(b,seed+1)
    baseline=collate_policy_samples([tensorize(a,kw),tensorize(b,kw)])
    permuted=collate_policy_samples([tensorize(pa,kw),tensorize(pb,kw)])
    model=model_for(kw);before=model.encode(baseline['inputs']);after=model.encode(permuted['inputs'])
    for field,latent in [('modules','module_h'),('connectors','connector_h'),('resources','resource_h'),
            ('coordinates','coordinate_h'),('connections','connection_h'),('environment','environment_h'),('relations','relation_h')]:
        n=len(getattr(a.inputs.features,field))
        order=torch.tensor(order_a[field]+[n+i for i in order_b[field]],dtype=torch.int64)
        torch.testing.assert_close(after[latent],before[latent][order],atol=3e-6,rtol=3e-6)
    for latent in ('graph_h','context_h','observation_h','task_h'):
        torch.testing.assert_close(before[latent],after[latent],atol=3e-6,rtol=3e-6)


def test_model_never_reads_supervision_or_audit_and_id_renaming_is_invariant():
    from dataclasses import fields,is_dataclass
    import inspect
    sample,kw=chain(3,((0,1),(1,2)),dynamic=True,typed=True)
    def rename(v):
        if isinstance(v,c.ModuleRef):return c.ModuleRef('arbitrary-instance-'+v.module_id)
        if is_dataclass(v):return replace(v,**{f.name:rename(getattr(v,f.name)) for f in fields(v)})
        if isinstance(v,tuple):return tuple(rename(r) for r in v)
        if isinstance(v,frozenset):return frozenset(rename(r) for r in v)
        return v
    batch=tensorize(sample,kw);renamed=tensorize(rename(sample),kw);model=model_for(kw)
    before=model.encode(batch['inputs']);equal_trees(before,model.encode(renamed['inputs']))
    alternate={**batch,'supervision':{'secret_target':'snake8'},'routing':{'audit':'stage/future','ids':['secret']}}
    equal_trees(before,model.encode(alternate['inputs']))
    assert tuple(inspect.signature(model.encode).parameters)==('inputs',)
    with pytest.raises(ValueError):model.encode(alternate)
    with pytest.raises(TypeError):model.encode(batch['inputs'],routing=alternate['routing'])


def test_full_encode_is_stateless_with_dropout_and_variable_graphs():
    a,kw=make(dynamic=True);model=model_for(kw);x=a['inputs'];snapshot=copy.deepcopy(x)
    before=model.encode(x);model.encode(make(9,ports=0,environment=False)[0]['inputs'])
    equal_trees(before,model.encode(x));equal_trees(snapshot,x)
    model.train();first=model.encode(x);second=model.encode(x)
    assert not torch.equal(first['module_h'],second['module_h'])
    assert torch.isfinite(first['module_h']).all()


def test_backward_smoke_has_finite_nonzero_encoder_and_message_gradients():
    batch,kw=chain(4,((0,1),(1,2)),dynamic=True);model=model_for(kw)
    outputs=model.encode(batch['inputs'])
    floats=[v for v in outputs.values() if v.is_floating_point()]
    assert all(torch.isfinite(v).all() for v in floats)
    loss=sum((v*torch.linspace(.1,1.,v.shape[-1])).sum()/max(v.numel(),1) for v in floats)
    assert torch.isfinite(loss)
    loss.backward()
    gradients={name:p.grad for name,p in model.named_parameters() if p.grad is not None}
    assert gradients and all(torch.isfinite(g).all() for g in gradients.values())
    assert gradients['encoders.module.0.weight'].abs().sum()>0
    assert gradients['message_blocks.0.physical_message.0.weight'].abs().sum()>0


def test_variable_arity_geometry_and_masked_values_preserve_semantic_component_order():
    batch,kw=make();model=model_for(kw);x=copy.deepcopy(batch['inputs'])
    for n in (2,5,13):
        values=torch.arange(n,dtype=torch.float32).reshape(1,n)+.25
        categories=torch.zeros(1,model.object_width);categories[0,-1]=1
        all_values=torch.cat((values,categories),1);valid=torch.ones_like(all_values,dtype=torch.bool)
        key=f'c_static:stairs.top_heights_m:intrinsic_vector:{n}'
        x['environment']={key:{'x':torch.cat((all_values,valid.float()),1),'validity':valid,
            'token_index':torch.tensor([0]),'object_index':torch.tensor([0]),'graph_index':torch.tensor([0])}}
        x['environment_ptr']=torch.tensor([0,1]);x['object_ptr']=torch.tensor([0,1])
        # No module relation in this intrinsic-quantity fixture.
        original=model.encode(x);swapped=copy.deepcopy(x)
        swapped['environment'][key]['x'][0,:n]=values.flip(1)[0]
        assert not torch.allclose(original['environment_h'],model.encode(swapped)['environment_h'])
        assert torch.isfinite(original['graph_h']).all()
    count=sum(p.numel() for p in model.parameters())
    assert sum(p.numel() for p in model_for(kw).parameters())==count


def test_malformed_environment_ownership_and_mask_storage_fail_closed():
    a,kw=make(2,dynamic=True);b,_=make(3,dynamic=True);model=model_for(kw)
    x=collate_policy_samples([a,b])['inputs'];bad=copy.deepcopy(x)
    row=next(iter(bad['environment'].values()));row['graph_index'][0]=1
    with pytest.raises(ValueError,match='Cross-graph'):model.encode(bad)
    bad=copy.deepcopy(x);bad['module_x'][0,-1]=1-bad['module_x'][0,-1]
    with pytest.raises(ValueError,match='Invalid G2 features'):model.encode(bad)


def test_context_retains_object_attribute_pairing_without_relations():
    from drl.IL.bc_v1.features import transform_inputs,PhysicalFrameConfig
    from drl.IL.bc_v1.feature_schema import extract_features
    sample,kw=fixture(2,2,geometry=False)
    def source(sizes,*,rename=False):
        fields=[]
        for i,(position,size) in enumerate(zip((1.,9.),sizes)):
            name=(9,7)[i] if rename else i
            fields.extend((c.NumericFeature(f'world.obstacles.{name}.center_xyz_m',(position,0.,0.)),
                c.NumericFeature(f'world.obstacles.{name}.size_xyz_m',(size,size,size))))
        if rename:fields.reverse()
        policy=replace(sample.inputs.policy,context=c.TaskContext(None,tuple(fields)))
        features=extract_features(policy,transform_inputs(policy,PhysicalFrameConfig()),kw['spec'],kw['feature_schema'])
        changed=replace(sample,inputs=replace(sample.inputs,policy=policy,features=features))
        return tensorize(changed,kw)['inputs']
    model=model_for(kw);a=source((1.,4.));b=source((4.,1.))
    assert len(a['relation_x'])==len(b['relation_x'])==0
    before=model.encode(a);after=model.encode(b)
    assert not torch.allclose(before['context_h'],after['context_h'],atol=3e-6,rtol=3e-6)
    renamed=model.encode(source((1.,4.),rename=True))
    for name in ('context_h','module_h','graph_h'):
        torch.testing.assert_close(before[name],renamed[name],atol=3e-6,rtol=3e-6)


@pytest.mark.parametrize('ownership',['connector_to_module','resource_to_module','coordinate_to_resource'])
def test_child_ownership_cannot_contradict_its_batch_graph_boundary(ownership):
    a,kw=make(2,environment=False);b,_=make(3,environment=False)
    x=collate_policy_samples([a,b])['inputs'];x[ownership][0]=2
    with pytest.raises(ValueError,match='Cross-graph'):model_for(kw).encode(x)


@pytest.mark.parametrize('family',['connection','relation'])
def test_connection_and_relation_rows_cannot_move_to_another_batch_graph(family):
    a,kw=chain(2,((0,1),),dynamic=True);b,_=chain(2,((0,1),),dynamic=True)
    x=collate_policy_samples([a,b])['inputs'];model=model_for(kw)
    changed=copy.deepcopy(x)
    if family=='connection':changed['connection_connectors'][0]=torch.tensor([5,6])
    else:
        changed['relation_to_module'][0]=2
        start,end=changed['relation_geometry_ptr'][:2].tolist()
        changed['relation_geometry_index'][start:end]=int(changed['environment_ptr'][1])
    with pytest.raises(ValueError,match='Cross-graph'):model.encode(changed)


def test_batched_family_pointers_are_required_and_validated():
    a,kw=make(2);b,_=make(3);x=collate_policy_samples([a,b])['inputs'];model=model_for(kw)
    for family in ('connector','resource','coordinate','connection','relation'):
        key=family+'_ptr';bad=copy.deepcopy(x);del bad[key]
        with pytest.raises(ValueError):model.encode(bad)
        bad=copy.deepcopy(x);bad[key][-1]+=1
        with pytest.raises(ValueError):model.encode(bad)
