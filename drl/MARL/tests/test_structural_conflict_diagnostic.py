"""Forensic accounting against the real ActionSet safety guard, no simulator."""
from types import SimpleNamespace

import pytest
import torch

from drl.IL.bc_v1.action_set import ActionBundle
from drl.IL.bc_v1.action_set_decoder import assert_executable
from drl.IL.bc_v1.bc_v1_contract import ActionChoice, ConnectorRef, ModuleRef, ResourceRef
from drl.IL.bc_v1.live_policy import smores_claims
from drl.MARL.structural_conflict_diagnostic import audit_prediction, pair_conflicts, summarize


def pair(action, a, b, first='TOP', second='BOTTOM'):
    modules = tuple(ModuleRef(x) for x in (a,b))
    return ActionChoice(action, modules[0], participants=modules,
        connectors=(ConnectorRef(modules[0],first),ConnectorRef(modules[1],second)))


def test_different_connectors_can_conflict_on_locomotion_and_exact_guard_matches():
    a = pair('align_faces','a','b')
    b = pair('dock','a','c','LEFT','RIGHT')
    _, conflicts = pair_conflicts({4:a,9:b},smores_claims)
    assert len(conflicts)==1
    assert conflicts[0]['shared_claims']==[dict(kind='resource',module='a',resource='locomotion')]
    assert conflicts[0]['categories']==['same physical resource claimed multiple times']
    assert conflicts[0]['is_at_first_conflicting_action']
    with pytest.raises(ValueError,match='^Conflicting physical resource claims$'):
        assert_executable(ActionBundle((a,b)),smores_claims)


def test_resource_action_can_reject_structural_pair_without_connector_overlap():
    resource = ActionChoice('wheel_velocity',ModuleRef('a'),resource=ResourceRef(ModuleRef('a'),'locomotion'))
    structural = pair('dock','b','a')
    _, conflicts = pair_conflicts({0:resource,10:structural},smores_claims)
    assert len(conflicts)==1
    assert conflicts[0]['shared_claims'][0]['resource']=='locomotion'
    assert conflicts[0]['candidate_a']['action_id']=='wheel_velocity'


def test_align_plus_dock_pair_reports_connector_and_resources_once_per_pair():
    a = pair('align_faces','a','b')
    b = pair('dock','a','b')
    claims, conflicts = pair_conflicts({0:a,1:b},smores_claims)
    assert len(conflicts)==1
    assert len(conflicts[0]['shared_claims'])==4
    assert 'align + dock on same pair in same decision' in conflicts[0]['categories']
    assert ResourceRef(ModuleRef('a'),'internal_motion') in claims[0]
    assert ResourceRef(ModuleRef('b'),'internal_motion') not in claims[0]


def test_disjoint_modules_are_executable_no_invented_module_conflict():
    actions = (pair('align_faces','a','b'),pair('dock','c','d'))
    _, conflicts = pair_conflicts(dict(enumerate(actions)),smores_claims)
    assert conflicts==[]
    assert assert_executable(ActionBundle(actions),smores_claims).actions==actions


def fake_binding(choices):
    def decoder(prediction,batch,normalizers,threshold):
        chosen = torch.nonzero(prediction['presence_logits']>=threshold).flatten().tolist()
        return (ActionBundle(tuple(choices[i] for i in chosen)),)
    return SimpleNamespace(_action_set_decoder=decoder,_claim_resolver=smores_claims,
        decode_action_bundle=lambda bundle:assert_executable(bundle,smores_claims))


def test_same_logits_threshold_changes_presence_for_both_families_only():
    a = pair('align_faces','a','b')
    wheel = ActionChoice('wheel_velocity',ModuleRef('a'),resource=ResourceRef(ModuleRef('a'),'locomotion'),payload=(9.,))
    prediction = dict(presence_logits=torch.tensor([5.,1.]),parameters=dict(payload=torch.tensor([9.])))
    before = prediction['parameters']['payload'].clone()
    binding = fake_binding([wheel,a])
    low = audit_prediction(prediction=prediction,batch={},normalizers=None,binding=binding,threshold=0.)
    high = audit_prediction(prediction=prediction,batch={},normalizers=None,binding=binding,threshold=4.5)
    assert low['selected']==[0,1] and not low['executable']
    assert low['exact_reason']=='Conflicting physical resource claims'
    assert high['selected']==[0] and high['executable']
    assert high['choices'][0].payload==(9.,)
    assert torch.equal(prediction['parameters']['payload'],before)


def test_decoder_failure_not_misrepresented_as_composer_rejection():
    binding = fake_binding([pair('align_faces','a','b'),pair('dock','a','b')])
    original = binding._action_set_decoder
    def invalid(prediction,batch,normalizers,threshold):
        if bool((prediction['presence_logits']>=threshold)[0]):
            raise ValueError('Invalid participant scope')
        return original(prediction,batch,normalizers,threshold)
    binding._action_set_decoder = invalid
    audit = audit_prediction(prediction=dict(presence_logits=torch.tensor([5.,5.])),
        batch={},normalizers=None,binding=binding,threshold=0.)
    assert audit['rejection_phase']=='action_set_decoder'
    assert audit['exact_reason']=='Invalid participant scope'
    assert audit['member_errors']=={0:'Invalid participant scope'}
    assert not audit['conflict_enumeration_complete']
    assert audit['conflicts']==[]


def test_no_capture_summary_does_not_claim_zero_measured_conflicts(tmp_path):
    summary = summarize(tmp_path)
    assert summary['status']=='awaiting_offline_capture'
    assert summary['captured_decisions']==0
    assert 'unmeasured' in summary['note']
    assert summary['examples']==[]


def test_observer_accepts_real_checkpoint_h2_schema_without_changing_prediction(tmp_path):
    # Fixture state is solely an integration test, never reported as live evidence.
    from drl.MARL.tests.test_bc_v4_runtime_observation import make_bridge, ROOT
    from drl.IL.tests.bc_v1 import test_live_observation as live_test
    from drl.IL.bc_v1.bc_v1_contract import TaskContext, ObservationBundle
    from drl.MARL.c1_runner import build_models, CHECKPOINT
    from drl.MARL.policy import sample_marl_action_bundle
    from drl.MARL.smores_runtime_binding import SMORESMARLRuntimeBinding, materialize_marl_action_set_prediction, SMORESMARLProposalOutcome
    from drl.MARL.structural_conflict_diagnostic import candidate_rows, write_json
    bridge = make_bridge()
    prepared = bridge.prepare_snapshot(live_test.snapshot(n=8,rc=True),
        task_context=TaskContext('navigation'),observation=ObservationBundle())
    actor,_ = build_models(SimpleNamespace(repository_root=ROOT,checkpoint=ROOT/CHECKPOINT,device='cpu'),bridge)
    owners = tuple(m.module_id for m in prepared.routing_snapshot['module_refs'])
    with torch.no_grad():
        output = actor.forward_all_owners(graph=prepared.actor_graph,routing=prepared.routing_snapshot,possible_agents=owners)
        saved = output['prediction_template']['presence_logits'].clone()
        mask = torch.zeros(len(output['records']),dtype=torch.bool)
        action = sample_marl_action_bundle(**{k:output[k] for k in (
            'records','resource_payloads','structural_payloads','resource_log_std','structural_log_std')},hard_mask=mask)
        materialized = materialize_marl_action_set_prediction(action=action,prediction_template=output['prediction_template'],
            routing=prepared.routing_snapshot,batch=prepared.decoder_batch)
        binding = SMORESMARLRuntimeBinding(adapter=bridge.adapter,claim_resolver=smores_claims)
        rng = torch.random.get_rng_state().clone()
        audit = audit_prediction(prediction=materialized,batch=prepared.decoder_batch,
            normalizers=bridge.normalizers,binding=binding,threshold=0.)
        output['calibrated_threshold']=4.5
        rows = candidate_rows(output=output,context=SimpleNamespace(routing_snapshot=prepared.routing_snapshot,hard_mask=mask),
            action=action,task='navigation',step=0,seed=7017,
            runtime_outcome=SMORESMARLProposalOutcome(audit['executable'],'TEST',audit['exact_reason'] or ''),audit=audit)
        write_json(tmp_path/'rows.json',rows)
        assert len(rows)==len(saved)
        assert torch.equal(saved,output['prediction_template']['presence_logits'])
        assert torch.equal(rng,torch.random.get_rng_state())
        assert sum(r['selected'] for r in rows)==int(action.presence.sum())
        torch.save(dict(raw_prediction=output['prediction_template'],sampled_action=action,
            actor_graph=prepared.actor_graph,routing=prepared.routing_snapshot,
            hard_mask=mask,decoder_batch=prepared.decoder_batch),tmp_path/'decision_000.pt')
        restored = torch.load(tmp_path/'decision_000.pt',map_location='cpu',weights_only=False)
        assert torch.equal(restored['raw_prediction']['presence_logits'],saved)
        replayed = audit_prediction(prediction=restored['raw_prediction'],batch=restored['decoder_batch'],
            normalizers=bridge.normalizers,binding=binding,threshold=4.5)
        assert replayed['selected']==torch.nonzero(saved>=4.5).flatten().tolist()
