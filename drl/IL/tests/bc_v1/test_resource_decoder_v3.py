from dataclasses import replace
import torch
import pytest
from drl.IL.tests.bc_v1.test_resource_heads_v3 import example,threads
from drl.IL.tests.bc_v1.test_resource_decisions import contract
from drl.IL.bc_v1.action_decoder import decode_actions,DispatchAvailability,ResourceStatus,ExecutionSemantics


def run(kind,index,*,physical_available=True):
    b,m,kw,_=example(kind);p=m(b['inputs']);r=b['routing']['samples'][0]
    p['resource']['outcome_logits']=torch.full((12,),-100.)
    p['resource']['outcome_logits'][[index,4,8]]=10.
    p['resource']['payload']=torch.zeros_like(p['resource']['payload'])
    av=DispatchAvailability(tuple((r,ResourceStatus.FREE) for r in r['resource_refs']),
        (physical_available,)*len(r['resource_candidates']),(False,)*len(r['structural_bindings']))
    withdrawals=[]
    out=decode_actions(p,b['inputs'],b['routing'],spec=kw['spec'],normalizers=kw['normalizers'],availability=(av,),
        execution_semantics={a.action_id:ExecutionSemantics.HOLD_UNTIL_EXPIRY for a in kw['spec'].actions},
        configured_arguments={},claim_resolver=lambda c:(c.resource,),resource_contract=contract(kw),
        inactive_sink=lambda sample,ref:withdrawals.append((sample,ref)))
    return out,withdrawals


def test_inactive_no_physical_action_and_fixed_exact_zero_bypasses_inverse_f2():
    out,w=run('inactive',0);assert out==() and len(w)==3
    out,w=run('fixed',1);assert len(out)==1 and out[0].choice.payload==(0.,0.) and len(w)==2
    out,w=run('parameterized',3);assert len(out)==1 and out[0].choice.payload==(1.,1.)


@pytest.mark.parametrize('shared_claim,internal_outcome',[(False,'pan_velocity'),(False,'PAN_HOLD_FIXED'),(True,'pan_velocity')])
def test_joint_decode_preserves_pan_wheels_or_respects_shared_claim(shared_claim,internal_outcome):
    # Real SMORES catalog, synthetic live snapshot, production H1/H2/decoder.
    from drl.IL.tests.bc_v1.test_live_observation import settings,builder,snapshot,context
    from drl.IL.bc_v1.live_observation import encode_smores_snapshot,prepare_live_observation
    from drl.IL.bc_v1.smores_adapter import SMORESAdapter
    from drl.IL.bc_v1.source_frame_bridges import BridgeResolution,SourceFrameBridgeBatch
    from drl.IL.bc_v1.resource_decisions import load_resource_contract,outcome_inputs
    from drl.IL.bc_v1.model import BCGNN,load_model_config
    from drl.IL.tests.bc_v1.g2_fixtures import ROOT
    from drl.IL.bc_v1.batching import collate_policy_samples
    s=settings();c=load_resource_contract(ROOT/'configs/bc_v1/resource_decisions_v1.json',s['spec'])
    state=encode_smores_snapshot(snapshot(),adapter=SMORESAdapter(s['spec']),graph_builder=builder())
    from drl.IL.bc_v1.bc_v1_contract import ObservationBundle
    view=prepare_live_observation(state,context=context(),observation=ObservationBundle(),bridges=SourceFrameBridgeBatch((BridgeResolution('world','world','live',None,True,'world',None,()),),(('id','test'),)),**s)
    one=view.batch['routing']['samples'][0] if 'samples' in view.batch['routing'] else view.batch['routing']
    b=view.batch;x=b['inputs']['resource_candidates']
    # Build local numeric domain on a single-sample collated physical batch.
    x.update(outcome_inputs({'routing':one},c));one['resource_contract_fingerprint']=c.fingerprint
    cfg=dict(schema_version='mssr.bc_v1.action_heads.v3',hidden_dim=128,activation='silu',dropout=.1,resource_contract=c.document)
    m=BCGNN(s['spec'],s['feature_schema'],load_model_config(ROOT/'configs/bc_v1/gnn_v1.json'),action_head_config=cfg).eval()
    p=m(b['inputs']);p['resource']['outcome_logits']=torch.zeros(len(x['outcome_kind']))
    for i,ci in enumerate(x['outcome_catalog'].tolist()):
        o=c.outcomes[ci];p['resource']['outcome_logits'][i]={internal_outcome:10.,'wheel_velocity':9.,'WHEEL_STOP_FIXED':12. if shared_claim else 8.}.get(o.outcome_id,0.)
    # All normalized 1 => finite nonzero physical velocity; fixed stop exact.
    p['resource']['payload']=torch.ones_like(p['resource']['payload'])
    from drl.IL.bc_v1.bc_v1_contract import ResourceRef
    helpers=tuple(ResourceRef(r,'helper') for r in one['module_refs']) if shared_claim else ()
    av=DispatchAvailability(tuple((r,ResourceStatus.FREE) for r in (*one['resource_refs'],*helpers)),
        (True,)*len(one['resource_candidates']),(False,)*len(one['structural_bindings']))
    from drl.IL.bc_v1.resource_decisions import load_decode_policy
    policy=load_decode_policy(ROOT/'configs/bc_v1/resource_decode_policy_v1.json',c,s['spec'])
    out=decode_actions(p,b['inputs'],b['routing'],spec=s['spec'],normalizers=s['normalizers'],availability=(av,),
        execution_semantics={a.action_id:ExecutionSemantics.HOLD_UNTIL_EXPIRY for a in s['spec'].actions},configured_arguments={},
        claim_resolver=lambda choice:((choice.resource,ResourceRef(choice.initiator,'helper')) if shared_claim else (choice.resource,)),resource_contract=c,decode_policy=policy)
    wheels=[r for r in out if r.choice.action_id=='wheel_velocity'];pan=[r for r in out if r.choice.action_id=='pan_velocity']
    assert wheels and bool(pan)==(not shared_claim)
    if shared_claim:
        assert all(r.choice.payload==(0.,0.) for r in wheels)
    else:
        assert all(all(v!=0. for v in r.choice.payload) for r in wheels)
        assert all(r.choice.payload==(0.,) if internal_outcome=='PAN_HOLD_FIXED' else all(v!=0. for v in r.choice.payload) for r in pan)
        assert {r.choice.initiator for r in wheels}=={r.choice.initiator for r in pan}
        assert len({r.choice.resource for r in out})==len(out)  # PAN/TILT share one domain.



def test_inactive_withdraws_only_own_continuous_lease_no_cancel():
    from drl.IL.tests.bc_v1.test_live_controller import setup,arm,finish,observe,pan
    from drl.IL.bc_v1 import live_controller as lc
    assert hasattr(lc,'RuntimeDecision'), 'Explicit per-resource withdrawal missing'
    commands=[(pan(),),lc.RuntimeDecision((),(('m1','internal_motion'),('foreign','resource')))]
    c,t,tr,_=setup(lambda *args:commands.pop(0))
    try:
        arm(c,t);finish(c,t);assert len(c.held)==1
        t.now=.2;observe(c,3.);finish(c,t)
        assert not c.held and not tr.clears and not tr.packets[-1][0] and not tr.packets[-1][1]
    finally:c.close()


def test_inactive_never_releases_busy_foreign_ownership_or_native_goals():
    from drl.IL.tests.bc_v1.test_live_controller import setup,arm,finish,observe,pan
    from drl.IL.bc_v1.live_controller import RuntimeDecision
    commands=[(pan(),),RuntimeDecision((),(('m1','internal_motion'),))]
    c,t,tr,_=setup(lambda *args:commands.pop(0))
    try:
        arm(c,t);finish(c,t);tr.busy.add('internal_motion:m1')
        t.now=.2;observe(c,3.);finish(c,t)
        assert ('m1','internal_motion') in c.held and not tr.clears
        assert all(not goals for _,goals,_ in tr.packets)
    finally:c.close()


def test_masked_physical_domains_fall_back_to_inactive_without_dispatch():
    out,withdrawals=run('parameterized',3,physical_available=False)
    assert out==() and len(withdrawals)==3


def test_decode_policy_bound_identity_and_generic_constraints():
    import copy,json
    from drl.IL.tests.bc_v1.test_live_observation import settings
    from drl.IL.tests.bc_v1.g2_fixtures import ROOT
    from drl.IL.bc_v1 import resource_decisions as rd
    assert hasattr(rd,'ResourceDecodePolicy'), 'Missing explicit decode compatibility identity'
    spec=settings()['spec'];c=rd.load_resource_contract(ROOT/'configs/bc_v1/resource_decisions_v1.json',spec)
    doc=json.loads((ROOT/'configs/bc_v1/resource_decode_policy_v1.json').read_text())
    policy=rd.ResourceDecodePolicy.from_dict(doc,c,spec)
    pan=next(o for o in c.outcomes if o.outcome_id=='pan_velocity')
    wheel=next(o for o in c.outcomes if o.outcome_id=='wheel_velocity')
    assert c.fingerprint=='7a77f9118c81a7ca7a9b80f8f1df78011c92492be9c53a17d3fc00188268d194'
    assert not c.compatible([(pan,(1.,)),(wheel,(2.,2.))])  # Historical, unchanged.
    assert policy.compatible([(pan,(1.,)),(wheel,(2.,2.))])
    assert policy.compatible([(pan,(8.9e-10,)),(wheel,(2.,2.))])
    strict=copy.deepcopy(doc);strict['forbidden_tag_pairs']=c.document['forbidden_tag_pairs']
    strict=rd.ResourceDecodePolicy.from_dict(strict,c,spec)
    assert not strict.compatible([(pan,(1.,)),(wheel,(2.,2.))])
    assert strict.fingerprint!=policy.fingerprint
    assert strict.compatible([(pan,(0.,)),(wheel,(2.,2.))])
    for key,value in [('resource_contract_fingerprint','foreign'),('revision',''),('unknown',True),('tags',[{'bad':'tag'}]),('forbidden_tag_pairs',[['unknown','tag']])]:
        bad=copy.deepcopy(doc);bad[key]=value
        with pytest.raises(ValueError):rd.ResourceDecodePolicy.from_dict(bad,c,spec)
    with pytest.raises(ValueError,match='contract'):
        policy.validate_contract(replace(c,serialized=c.serialized+' '))
