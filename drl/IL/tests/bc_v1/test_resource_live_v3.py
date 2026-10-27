from pathlib import Path
import json
from dataclasses import replace
import pytest
import torch
from drl.IL.tests.bc_v1.test_live_observation import settings,builder,snapshot,context
from drl.IL.tests.bc_v1.test_resource_heads_v3 import threads
from drl.IL.bc_v1.live_policy import LivePolicy,_head_contract
from drl.IL.bc_v1.model import BCGNN,load_model_config
from drl.IL.tests.bc_v1.g2_fixtures import ROOT
from drl.IL.bc_v1.live_controller import RuntimeDecision,PhysicalSnapshot
from drl.IL.bc_v1.sidecar import _hash


def config():
    path=ROOT/'configs/bc_v1/action_heads_resource_v3.json'
    assert path.is_file(),'Missing v3 H2 profile'
    return json.loads(path.read_text())


def test_live_v3_builds_current_outcomes_and_returns_resource_withdrawals():
    cfg=config();s=settings();m=BCGNN(s['spec'],s['feature_schema'],load_model_config(ROOT/'configs/bc_v1/gnn_v1.json'),action_head_config=cfg).eval()
    policy=LivePolicy(m,s,builder(),lambda raw:context(),device='cpu')
    # Force the semantic inactive option through the actual production scorer.
    c=m.action_heads.resource_contract
    with torch.no_grad():
        m.action_heads.mode_scorer[-1].weight.zero_();m.action_heads.mode_scorer[-1].bias.zero_()
        m.action_heads.no_structural_scorer[-1].weight.zero_();m.action_heads.no_structural_scorer[-1].bias.fill_(1e6)
    raw=snapshot();stamp=raw['state']['stamp'] if 'stamp' in raw['state'] else raw['state']['sim_time']
    result=policy(PhysicalSnapshot('epoch',stamp,raw),{'busy':()})
    assert isinstance(result,RuntimeDecision)
    assert len(result.withdrawals)==len(policy.last_observation.batch['inputs']['resource_x'])
    assert all(command.key[1]=='structural' for command in result.commands)


def test_v3_profile_requires_contract_identity_legacy_not_reinterpreted():
    cfg=config();identity={'h2_config_fingerprint':_hash(cfg),'structural_decision_schema':'mssr.bc_v1.structural_decisions.v2'}
    with pytest.raises(ValueError,match='resource'): _head_contract(ROOT/'configs/bc_v1',identity)
    from drl.IL.bc_v1.resource_decisions import load_resource_contract,VERSION
    c=load_resource_contract(ROOT/'configs/bc_v1/resource_decisions_v1.json',settings()['spec'])
    identity.update(resource_decision_schema=VERSION,resource_contract_fingerprint=c.fingerprint)
    h2,profile=_head_contract(ROOT/'configs/bc_v1',identity)
    assert profile=='bc_v1.resource_v3' and h2==cfg


from drl.IL.tests.bc_v1.test_live_checkpoint_loader import synthetic_payload

def test_public_loader_admits_v3_strictly_without_relabeling_legacy(tmp_path,synthetic_payload):
    import copy
    from drl.IL.bc_v1.live_policy import load_frozen_policy
    cfg=config();s=settings();h1=load_model_config(ROOT/'configs/bc_v1/gnn_v1.json')
    m=BCGNN(s['spec'],s['feature_schema'],h1,action_head_config=cfg)
    payload=copy.deepcopy(synthetic_payload);c=m.action_heads.resource_contract
    payload['model']=m.state_dict();payload['identity'].update(h2_config_fingerprint=_hash(cfg),
        resource_contract_fingerprint=c.fingerprint,resource_decision_schema='mssr.bc_v1.resource_decisions.v1')
    path=tmp_path/'v3.pt';torch.save(payload,path)
    p=load_frozen_policy(ROOT,path,builder(),device='cpu')
    assert p.checkpoint_provenance['checkpoint_profile']=='bc_v1.resource_v3'
    assert p.checkpoint_provenance['resource_contract_fingerprint']==c.fingerprint
    assert p.decode_policy is not None
    assert p.checkpoint_provenance['decode_policy_fingerprint']==p.decode_policy.fingerprint
    assert p.checkpoint_provenance['decode_policy']==p.decode_policy.document
    assert p.decode_policy.document['forbidden_tag_pairs']==[]
    assert all(torch.equal(v,p.model.state_dict()[k]) for k,v in m.state_dict().items())
    payload['identity']['resource_contract_fingerprint']='foreign';torch.save(payload,path)
    with pytest.raises(ValueError,match='resource'):load_frozen_policy(ROOT,path,builder(),device='cpu')


def test_existing_smoke_safety_wrapper_preserves_v3_withdrawals():
    from drl.IL.bc_v1.live_controller import RuntimeCommand
    from scripts.smores_ep.tests.test_bc_live_smoke import api
    runner=api();decision=RuntimeDecision((),(('m1','rotor'),))
    assert runner.smoke_inference(lambda *args:decision,None,{}) is decision
    structural=RuntimeCommand(('m1','structural'),{'command':'dock'},('resource:m1',),False)
    with pytest.raises(runner.UnexpectedStructuralProposal):
        runner.smoke_inference(lambda *args:RuntimeDecision((structural,),()),None,{})


def test_live_policy_uses_explicit_corrected_decode_policy():
    from drl.IL.bc_v1 import resource_decisions as rd
    assert hasattr(rd,'load_decode_policy'), 'Missing explicit decode compatibility identity'
    cfg=config();s=settings();m=BCGNN(s['spec'],s['feature_schema'],load_model_config(ROOT/'configs/bc_v1/gnn_v1.json'),action_head_config=cfg).eval()
    c=m.action_heads.resource_contract
    policy=rd.load_decode_policy(ROOT/'configs/bc_v1/resource_decode_policy_v1.json',c,s['spec'])
    events=[]
    live=LivePolicy(m,s,builder(),lambda raw:context(),device='cpu',decode_policy=policy,event_sink=events.append)
    def scores(module,args,prediction):
        x=args[0]['resource_candidates'];p=prediction['resource']
        for i,ci in enumerate(x['outcome_catalog'].tolist()):
            p['outcome_logits'][i]={'pan_velocity':10.,'wheel_velocity':9.,'WHEEL_STOP_FIXED':8.}.get(c.outcomes[ci].outcome_id,0.)
        p['payload'].fill_(1.)
        prediction['structural']['no_structural_logits'].fill_(1e6)
    m.register_forward_hook(scores)
    raw=snapshot();result=live(PhysicalSnapshot('epoch',raw['state']['stamp'],raw),{'busy':()})
    rows={r.choice.resource:r.choice for r in live.last_decoded}
    assert result.commands and len(rows)==len(live.last_decoded)
    assert {r.action_id for r in rows.values()}=={'pan_velocity','wheel_velocity'}
    assert all(all(v!=0 for v in r.payload) for r in rows.values())
    assert events[-1]['decode_policy_fingerprint']==policy.fingerprint
