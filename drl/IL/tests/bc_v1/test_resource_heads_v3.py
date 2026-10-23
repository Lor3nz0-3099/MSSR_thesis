import copy
import math
from dataclasses import replace
import pytest
import torch
from drl.IL.tests.bc_v1.test_resource_decisions import api,contract
from drl.IL.tests.bc_v1.g2_fixtures import fixture,tensorize,ROOT
from drl.IL.bc_v1.model import BCGNN,load_model_config
from drl.IL.bc_v1.batching import collate_policy_samples
from drl.IL.bc_v1.losses import resource_mode_loss,resource_payload_loss,structural_choice_loss,compute_bc_loss,load_loss_config
from drl.IL.bc_v1.structural_decisions import add_no_structural_supervision

@pytest.fixture(autouse=True)
def threads():
    n=torch.get_num_threads();torch.set_num_threads(1);yield;torch.set_num_threads(n)


def example(kind='parameterized'):
    s,kw=fixture();target=s.supervision.resource_targets[0]
    if kind=='fixed':s=replace(s,supervision=replace(s.supervision,resource_targets=(replace(target,payload=(0.,0.)),)))
    elif kind in ('inactive','unknown'):s=replace(s,supervision=replace(s.supervision,resource_targets=()))
    old=add_no_structural_supervision(tensorize(s,kw));c=contract(kw)
    sample=api().add_resource_decisions(old,c,authoritative_resources={0} if kind=='inactive' else set())
    cfg={'schema_version':'mssr.bc_v1.action_heads.v3','hidden_dim':128,'activation':'silu','dropout':.1,'resource_contract':c.document}
    model=BCGNN(kw['spec'],kw['feature_schema'],load_model_config(ROOT/'configs/bc_v1/gnn_v1.json'),action_head_config=cfg).eval()
    b=collate_policy_samples([sample]);return b,model,kw,old


def test_v3_logits_cover_all_outcomes_and_batched_routes():
    b,m,kw,_=example();x=b['inputs']['resource_candidates'];p=m(b['inputs'])
    assert len(p['resource']['outcome_logits'])==12 # 3 x [INACTIVE,stop,pace,aim]
    s,kw=fixture();v=api().add_resource_decisions(tensorize(s,kw),contract(kw),authoritative_resources={1})
    c=collate_policy_samples([v,v]);r=c['inputs']['resource_candidates'];y=c['supervision']['resource']
    assert r['outcome_ptr'].tolist()==[0,4,8,12,16,20,24]
    assert r['outcome_resource'].tolist()==[i for i in range(6) for _ in range(4)]
    assert r['outcome_candidate'][12:].tolist()==[-1,6,7,6,-1,8,9,8,-1,10,11,10]
    assert y['decision_group'].tolist()==list(range(6))
    assert y['decision_candidate'].tolist()==[3,4,-1,15,16,-1]


@pytest.mark.parametrize('kind,ce_support,payload_support',[('inactive',1,0),('fixed',1,0),('parameterized',1,1),('unknown',0,0)])
def test_loss_masks_and_gradients_for_each_outcome(kind,ce_support,payload_support):
    b,m,_,_=example(kind);p=m(b['inputs']);r=p['resource'];r['outcome_logits'].retain_grad();r['payload'].retain_grad()
    ce=resource_mode_loss(p,b);payload=resource_payload_loss(p,b)
    assert int(ce['support'])==ce_support and int(payload['support'])==payload_support
    (ce['mean']+payload['mean']).backward()
    assert bool(r['outcome_logits'].grad.abs().sum()>0)==bool(ce_support)
    assert bool(r['payload'].grad.abs().sum()>0)==bool(payload_support)


def test_structural_v2_loss_unchanged_and_complete_loss_finite():
    b,m,_,old=example();p=m(b['inputs']);before={**b,'supervision':{**b['supervision'],'structural':old['supervision']['structural']}}
    assert torch.equal(structural_choice_loss(p,b)['mean'],structural_choice_loss(p,before)['mean'])
    total,f=compute_bc_loss(p,b,load_loss_config(ROOT/'configs/bc_v1/loss_v1.json'))
    assert torch.isfinite(total) and int(f['resource_mode']['support'])==1


def test_singleton_physical_mode_still_classifies_inactive_and_fixed():
    from drl.IL.tests.bc_v1.test_losses import recatalog
    s,kw=fixture();spec=kw['spec'];mt=spec.module_types[0];r=mt.resources[0]
    s,kw=recatalog(s,kw,replace(spec,module_types=(replace(mt,resources=(replace(r,modes=r.modes[:1]),)),)))
    c=contract(kw);v=api().add_resource_decisions(tensorize(s,kw),c)
    cfg={'schema_version':'mssr.bc_v1.action_heads.v3','hidden_dim':128,'activation':'silu','dropout':.1,'resource_contract':c.document}
    m=BCGNN(kw['spec'],kw['feature_schema'],load_model_config(ROOT/'configs/bc_v1/gnn_v1.json'),action_head_config=cfg)
    p=m(v['inputs']);assert len(p['resource']['outcome_logits'])==9
    assert int(resource_mode_loss(p,v)['support'])==1


def test_resource_metrics_stream_inactive_fixed_and_parameterized():
    from drl.IL.bc_v1.metrics import BCMetrics
    b,m,kw,_=example('inactive');p=m(b['inputs']);p['resource']['outcome_logits']=torch.zeros(12)
    metrics=BCMetrics(kw['spec'],kw['normalizers']);metrics.update(p,b)
    row=metrics.compute()['resource']['decisions']
    assert row['support']==1 and row['inactive']['support']==1
    assert row['inactive']['accuracy']==1 and row['outcome_exact_accuracy']==1
    assert row['false_activation_rate']==0 and row['missed_activation_rate'] is None


def test_v3_metrics_report_compatibility_and_semantic_resource_names():
    from drl.IL.bc_v1.metrics import BCMetrics
    b,m,kw,_=example('fixed');p=m(b['inputs'])
    metrics=BCMetrics(kw['spec'],kw['normalizers']);metrics.update(p,b)
    row=metrics.compute()['resource']['decisions']
    assert row['compatibility']['support']==3
    assert row['compatibility']['post_violations']==0
    assert list(row['by_resource'])==['["kit","rotor"]']
    assert 'stop' in row['by_mode']


def test_corrupt_outcome_routes_and_extra_payload_supervision_fail_closed():
    from drl.IL.bc_v1.resource_decisions import validate_outcome_inputs
    b,m,kw,_=example('parameterized');x=b['inputs']['resource_candidates'];c=contract(kw)
    corrupt=copy.deepcopy(x);corrupt['outcome_candidate'][1]=1 # fixed pace misrouted to aim
    with pytest.raises(ValueError):validate_outcome_inputs(corrupt,c,3)


def test_extra_payload_supervision_for_inactive_fails_closed():
    b,m,kw,_=example('parameterized')
    p=m(b['inputs']);b['supervision']['resource']['decision_candidate'][0]=0 # inactive + retained payload
    with pytest.raises(ValueError):resource_payload_loss(p,b)


def test_v3_structural_predictions_equal_v2_with_identical_common_weights():
    b,m,kw,_=example();v2=BCGNN(kw['spec'],kw['feature_schema'],m.config,action_head_config=dict(
        schema_version='mssr.bc_v1.action_heads.v2',hidden_dim=128,activation='silu',dropout=.1)).eval()
    v2.load_state_dict({k:v for k,v in m.state_dict().items() if k in v2.state_dict()},strict=True)
    from drl.IL.bc_v1.sidecar import assert_exact
    assert_exact(m(b['inputs'])['structural'],v2({**b['inputs'],'resource_candidates':{k:v for k,v in b['inputs']['resource_candidates'].items() if not k.startswith('outcome_')}})['structural'])


def test_collate_refuses_incompatible_resource_contracts():
    s,kw=fixture();v=api().add_resource_decisions(tensorize(s,kw),contract(kw))
    other=copy.deepcopy(v);other['routing']['resource_contract_fingerprint']='foreign'
    with pytest.raises(ValueError):collate_policy_samples([v,other])


def test_metrics_handle_inactive_only_domain_without_physical_candidates():
    from drl.IL.tests.bc_v1.test_losses import recatalog
    from drl.IL.bc_v1.metrics import BCMetrics
    s,kw=fixture();spec=replace(kw['spec'],actions=tuple(replace(a,enabled=False) if a.scope=='resource' else a for a in kw['spec'].actions))
    s,kw=recatalog(s,kw,spec);s=replace(s,supervision=replace(s.supervision,resource_targets=()))
    c=contract(kw);v=api().add_resource_decisions(add_no_structural_supervision(tensorize(s,kw)),c,authoritative_resources={0,1,2})
    b=collate_policy_samples([v]);cfg=dict(schema_version='mssr.bc_v1.action_heads.v3',hidden_dim=128,activation='silu',dropout=.1,resource_contract=c.document)
    m=BCGNN(kw['spec'],kw['feature_schema'],load_model_config(ROOT/'configs/bc_v1/gnn_v1.json'),action_head_config=cfg)
    metrics=BCMetrics(kw['spec'],kw['normalizers']);metrics.update(m(b['inputs']),b)
    assert metrics.compute()['resource']['decisions']['inactive']['support']==3
