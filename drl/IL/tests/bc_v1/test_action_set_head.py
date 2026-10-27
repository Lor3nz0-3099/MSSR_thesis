"""Shared physical presence scores, retaining the generic H1/payload ABI."""
import copy
import importlib
import importlib.util
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import pytest
import torch

from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.bc_v1.batching import collate_policy_samples
from drl.IL.bc_v1.model import load_model_config
from drl.IL.tests.bc_v1.g2_fixtures import fixture, tensorize
from drl.IL.tests.bc_v1.test_model import model_for, equal_trees, permuted_sample
from drl.IL.tests.bc_v1.test_action_heads import enriched_fixture, semantic_predictions, reroute_structural

ROOT = Path(__file__).resolve().parents[4]


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.action_set_head'), 'Missing ActionSet shared presence head'
    return importlib.import_module('drl.IL.bc_v1.action_set_head')


def config():
    return api().load_action_set_head_config(ROOT/'configs/bc_v1/action_set_head_v1.json')


def policy_for(kw):
    torch.manual_seed(2718)
    return api().ActionSetPolicy(kw['spec'], kw['feature_schema'],
        load_model_config(ROOT/'configs/bc_v1/gnn_v1.json'), config()).eval()


def floats(tree):
    if isinstance(tree, dict):
        for value in tree.values():
            yield from floats(value)
    elif isinstance(tree, torch.Tensor) and tree.is_floating_point():
        yield tree


def legacy_codec_view(out):
    """Adapt only test routing to reuse the existing semantic comparator."""
    q = len(out['parameters']['resource']['payload_candidate'])
    resource = {**out['parameters']['resource'],
        'mode_candidate': torch.empty(0, dtype=torch.int64),
        'mode_logits': torch.empty(0)}
    # Fixture numeric modes all have one payload record per candidate.
    return {'resource': resource, 'structural': {
        **out['parameters']['structural'], 'candidate_logits': out['presence_logits'][q:]}}


@pytest.mark.parametrize('sizes,ports', [((1,), 0), ((3,), 2), ((2, 5), 1)])
def test_every_physical_candidate_has_one_shared_presence_score_and_graph(sizes, ports):
    api()
    samples = [fixture(n, ports)[0] for n in sizes]
    _, kw = fixture(sizes[0], ports)
    x = collate_policy_samples([tensorize(s, kw) for s in samples])['inputs']
    model = policy_for(kw)
    seen = []
    handle = model.head.presence_scorer.register_forward_hook(lambda module, args, output: seen.append(args[0].shape[0]))
    out = model(x); handle.remove()
    qr = sum(2*n for n in sizes)
    qs = sum(n*(1 + 2*(n-1)*ports*ports) for n in sizes)
    expected = [i for i,n in enumerate(sizes) for _ in range(2*n)]
    expected += [i for i,n in enumerate(sizes) for _ in range(n*(1+2*(n-1)*ports*ports))]
    assert out.keys() == {'presence_logits', 'candidate_graph', 'parameters'}
    assert out['presence_logits'].shape == (qr+qs,)
    assert out['presence_logits'].dtype == torch.float32
    assert out['candidate_graph'].dtype == torch.int64
    assert out['candidate_graph'].tolist() == expected
    assert seen == [qr+qs]
    assert out['parameters'].keys() == {'resource', 'structural'}
    assert out['parameters']['resource'].keys() == {'payload', 'payload_candidate', 'payload_ptr', 'categorical'}
    assert out['parameters']['structural'].keys() == {'arguments', 'scope'}
    assert all(torch.isfinite(v).all() for v in floats(out))
    assert not any(any(forbidden in name for forbidden in ('mode_scorer', 'structural_scorer', 'outcome_projection'))
        for name,_ in model.head.named_parameters())
    assert not any(isinstance(layer, torch.nn.Embedding) for layer in model.modules())


def test_presence_is_raw_and_legacy_outcome_tables_are_ignored():
    sample,kw=fixture();x=tensorize(sample,kw)['inputs'];model=policy_for(kw)
    with torch.no_grad():
        model.head.presence_scorer[-1].weight.zero_();model.head.presence_scorer[-1].bias.fill_(2.5)
    out=model(x)
    assert out['presence_logits'].tolist()==[2.5]*57
    changed=copy.deepcopy(x)
    changed['resource_candidates'].update(outcome_ptr='legacy-unused',outcome_catalog='legacy-unused',outcome_resource='legacy-unused')
    equal_trees(out,model(changed))
    with pytest.raises(ValueError,match='causal'):
        model.head(model.backbone.encode(x),{**x,'supervision':{}})


def test_payload_arguments_scope_backpropagate_into_unchanged_h1():
    sample,kw=enriched_fixture();x=tensorize(sample,kw)['inputs'];model=policy_for(kw)
    out=model(x);values=[v for v in floats(out) if v.numel()]
    sum(v.square().mean() for v in values).backward()
    for layer in (model.head.presence_scorer,model.head.payload_decoder,
            model.head.numeric_decoder,model.head.argument_category_scorer,
            model.head.boolean_scorer,model.head.reference_scorer,
            model.head.reference_set_scorer,model.head.scope_scorer,
            model.backbone.resource_context,model.backbone.module_fusion):
        grads=[p.grad for p in layer.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum() for g in grads)>0
    assert out['parameters']['structural']['arguments']['numeric']['value'].shape==(9,)
    assert out['parameters']['structural']['scope']['logits'].shape==(6,)


def test_wrapper_preserves_standalone_h1_state_and_latents():
    sample,kw=fixture();x=tensorize(sample,kw)['inputs'];standalone=model_for(kw);policy=policy_for(kw)
    assert policy.backbone.action_heads is None
    assert standalone.state_dict().keys()==policy.backbone.state_dict().keys()
    equal_trees(standalone.state_dict(),policy.backbone.state_dict())
    equal_trees(standalone.encode(x),policy.backbone.encode(x))
    with pytest.raises(ValueError,match='not configured'):policy.backbone(x)
    equal_trees(policy(x),policy.head(policy.backbone.encode(x),x))


def test_batch_isolation_preserves_all_second_graph_predictions():
    a,kw=fixture(2);b,_=fixture(3)
    x=collate_policy_samples([tensorize(a,kw),tensorize(b,kw)])['inputs'];model=policy_for(kw)
    before=model(x);changed=copy.deepcopy(x);changed['module_x'][0,0]+=2.;after=model(changed)
    mask=before['candidate_graph']==1
    torch.testing.assert_close(before['presence_logits'][mask],after['presence_logits'][mask])
    assert not torch.allclose(before['presence_logits'][~mask],after['presence_logits'][~mask])
    torch.testing.assert_close(before['parameters']['resource']['payload'][6:],after['parameters']['resource']['payload'][6:])


def test_entity_permutation_and_physical_id_renaming_preserve_all_codecs():
    sample,kw=enriched_fixture();changed,_=permuted_sample(sample,19);model=policy_for(kw)
    a,b=(tensorize(s,kw)['inputs'] for s in (sample,changed))
    before=semantic_predictions(legacy_codec_view(model(a)),a,sample)
    after=semantic_predictions(legacy_codec_view(model(b)),b,changed)
    assert before.keys()==after.keys()
    for key in before:torch.testing.assert_close(before[key],after[key],atol=3e-6,rtol=3e-6)
    # Resource presence follows the actual physical resource order as well.
    refs=[r.ref for r in sample.inputs.features.resources]
    order=[refs.index(r.ref) for r in changed.inputs.features.resources]
    torch.testing.assert_close(model(b)['presence_logits'][:6].reshape(3,2),
        model(a)['presence_logits'][:6].reshape(3,2)[order],atol=3e-6,rtol=3e-6)
    def rename(value):
        if isinstance(value,c.ModuleRef):return c.ModuleRef('new-id-'+value.module_id)
        if is_dataclass(value):return replace(value,**{f.name:rename(getattr(value,f.name)) for f in fields(value)})
        if isinstance(value,tuple):return tuple(rename(v) for v in value)
        if isinstance(value,frozenset):return frozenset(rename(v) for v in value)
        return value
    equal_trees(model(a),model(tensorize(rename(sample),kw)['inputs']))


def test_empty_structural_and_resource_domains_have_no_synthetic_candidates():
    sample,kw=fixture(2,0);x=tensorize(sample,kw)['inputs'];model=policy_for(kw)
    x=reroute_structural(x,[])
    out=model(x)
    assert out['presence_logits'].shape==(4,)
    assert all(v.numel()==0 for v in floats(out['parameters']['structural']))
    x=copy.deepcopy(x);r=x['resource_candidates']
    r['resource']=r['resource'][:0];r['mode']=r['mode'][:0]
    r['group_ptr']=torch.zeros(3,dtype=torch.int64);r['payload_ptr']=torch.zeros(1,dtype=torch.int64)
    out=model(x)
    assert out['presence_logits'].shape==(0,) and out['candidate_graph'].shape==(0,)
    assert all(v.numel()==0 for v in floats(out))


def test_resource_categorical_payload_is_a_codec_without_mode_decision():
    from drl.IL.bc_v1.candidates import enumerate_candidates
    sample,kw=fixture();spec=kw['spec'];mt=spec.module_types[0];res=mt.resources[0]
    modes=tuple(replace(m,codec='categorical',units=('1',),categories=('low','mid','high')) if m.mode_id=='aim' else m for m in res.modes)
    actions=tuple(replace(a,parameters=(c.ParameterSpec('level','category',categories=('low','mid','high'),module_type_id='kit',resource_id='rotor'),)) if a.action_id=='aim' else a for a in spec.actions)
    spec=replace(spec,module_types=(replace(mt,resources=(replace(res,modes=modes),)),),actions=actions)
    kw={**kw,'spec':spec,'normalizers':replace(kw['normalizers'],metadata_items=tuple(sorted({**kw['normalizers'].metadata,'robot_spec_fingerprint':c.robot_spec_fingerprint(spec)}.items())))}
    sample=replace(sample,inputs=replace(sample.inputs,candidates=enumerate_candidates(sample.inputs.policy.graph,spec,kw['feature_schema'])))
    out=policy_for(kw)(tensorize(sample,kw)['inputs'])['parameters']['resource']
    assert out['payload'].shape==(6,) and out['payload_candidate'].tolist()==[0,2,4]
    assert out['categorical']['candidate'].tolist()==[1,3,5]
    assert out['categorical']['logits'].shape==(9,)
    assert out['categorical']['ptr'].tolist()==[0,3,6,9]


@pytest.mark.parametrize('change',[{'schema_version':'mssr.bc_v1.action_heads.v1'}, {'hidden_dim':64}, {'dropout':0.}, {'dropout':1}, {'activation':'relu'}, {'extra':True}, {'threshold':0.}])
def test_config_schema_is_strict(change):
    _,kw=fixture();cfg=config()
    with pytest.raises(ValueError):api().ActionSetHead(kw['spec'],load_model_config(ROOT/'configs/bc_v1/gnn_v1.json'),{**cfg,**change})
