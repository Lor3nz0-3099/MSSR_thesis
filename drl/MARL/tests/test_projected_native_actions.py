from types import SimpleNamespace
import pytest
import torch

from drl.IL.bc_v1.bc_v1_contract import ModuleRef,ResourceRef,ActionChoice
from drl.MARL.smores_runtime_binding import SMORESMARLProposalOutcome
from drl.MARL.runtime_signals import collect_runtime_signals
from drl.MARL.action_projection import ProjectionCandidate,project_executable_action_set


def test_partial_acceptance_retains_conflict_and_invalid_penalty_events():
    outcome=SMORESMARLProposalOutcome(True,'ACCEPTED','partial',('a',),
        conflict_owner_ids=('b',),invalid_owner_ids=('c',))
    frame=collect_runtime_signals(owner_ids=('a','b','c'),delta_seconds=1/30,proposal_outcome=outcome)
    assert frame.signals.conflict_owner_ids==frozenset(('b',))
    assert frame.signals.invalid_owner_ids==frozenset(('c',))
    assert frame.availability['conflict']['status']=='measured_present'
    assert frame.availability['invalid']['status']=='measured_present'


def test_projected_native_dispatch_keeps_independent_acceptances_on_one_native_rejection():
    from drl.MARL.tests.test_smores_runtime_binding import _native_dispatcher_fixture
    from drl.MARL.smores_runtime_binding import DecodedSMORESAction
    dispatcher,executor,_=_native_dispatcher_fixture(reject_on_submit=2)
    candidates=[]
    for i,m in enumerate(('a','b','c')):
        choice=ActionChoice('pan_target',ModuleRef(m),resource=ResourceRef(ModuleRef(m),'internal_motion'),payload=(.1,))
        decoded=DecodedSMORESAction(choice,dict(command='set_pan',module_ids=[m],parameters=dict(angle_rad=.1)))
        candidates.append(ProjectionCandidate(m,i,8.-i,(m,),choice,(choice.resource,),decoded))
    projection=project_executable_action_set(candidates)
    dispatcher.dispatch_projected(env_index=0,projection=projection)
    outcome=dispatcher.proposal_outcome(0)
    assert outcome.accepted
    assert outcome.executed_candidate_ids==('a','c')
    assert outcome.conflict_owner_ids==('b',)
    assert outcome.projection.raw_sampled_action is projection.raw_sampled_action
    assert [d.candidate_id for d in outcome.physical_drops]==['b']
    assert outcome.physical_drops[0].reason=='test rejection'
    assert executor.canceled==[]


def test_projected_dispatch_retains_both_continuous_resources_on_same_module():
    from drl.MARL.tests.test_smores_runtime_binding import _native_dispatcher_fixture
    from drl.MARL.smores_runtime_binding import DecodedSMORESAction
    dispatcher,executor,_=_native_dispatcher_fixture()
    rows=[]
    for i,(name,resource,params) in enumerate((
        ('wheel_velocity','locomotion',dict(left_rad_s=1.,right_rad_s=3.)),
        ('pan_velocity','internal_motion',dict(rate_rad_s=.75)))):
        ref=ResourceRef(ModuleRef('module_0'),resource)
        choice=ActionChoice(name,ref.module,resource=ref)
        decoded=DecodedSMORESAction(choice,dict(command=name,module_ids=['module_0'],parameters=params))
        rows.append(ProjectionCandidate(name,i,8.-i,('module_0',),choice,(ref,),decoded))
    dispatcher.dispatch_projected(env_index=0,projection=project_executable_action_set(rows))
    command=dispatcher.continuous_commands(0)['module_0']
    assert command.linear_x_m_s==pytest.approx(.08)
    assert command.angular_z_rad_s==pytest.approx(.5)
    assert command.pan_velocity_rad_s==pytest.approx(.75)
    assert not executor.submitted


def test_projected_dispatch_rolls_back_earlier_admissions_on_infrastructure_failure():
    from drl.MARL.tests.test_smores_runtime_binding import _native_dispatcher_fixture
    from drl.MARL.smores_runtime_binding import DecodedSMORESAction
    dispatcher,executor,_=_native_dispatcher_fixture()
    original_submit=executor.submit
    def fail_second(goal,now):
        if executor.submitted:raise RuntimeError('native infrastructure fault')
        return original_submit(goal,now)
    executor.submit=fail_second
    candidates=[]
    for i,m in enumerate(('a','b')):
        ref=ResourceRef(ModuleRef(m),'internal_motion');choice=ActionChoice('pan_target',ref.module,resource=ref)
        candidates.append(ProjectionCandidate(m,i,8.-i,(m,),choice,(ref,),
            DecodedSMORESAction(choice,dict(command='set_pan',module_ids=[m],parameters=dict(angle_rad=.1)))))
    with pytest.raises(RuntimeError,match='native infrastructure fault'):
        dispatcher.dispatch_projected(env_index=0,projection=project_executable_action_set(candidates))
    assert len(executor.canceled)==1
    assert not dispatcher.continuous_commands(0)
    assert dispatcher.proposal_outcome(0) is None


def test_projected_binding_preserves_raw_sample_and_raw_ppo_likelihood():
    from pathlib import Path
    from drl.MARL.c1_runner import build_observation_bridge
    from drl.MARL.smores_runtime_binding import SMORESMARLRuntimeBinding
    from drl.MARL.projected_runtime import project_marl_action
    from drl.MARL.policy import bind_marl_candidates,compute_owner_action_log_probs
    from drl.IL.bc_v1.live_policy import smores_claims
    root=Path(__file__).resolve().parents[3]
    path=root/'logs/isaaclab_c1_native/redesign/structural_conflict_diagnostic/navigation/decision_000.pt'
    if not path.exists():pytest.skip('local finished diagnostic capture unavailable')
    capture=torch.load(path,map_location='cpu',weights_only=False)
    bridge=build_observation_bridge(root)
    binding=SMORESMARLRuntimeBinding(adapter=bridge.adapter,claim_resolver=smores_claims)
    original=capture['sampled_action']
    presence=original.presence.clone()
    projection=project_marl_action(binding=binding,action=original,prediction_template=capture['raw_prediction'],
        routing=capture['routing'],batch=capture['decoder_batch'],normalizers=bridge.normalizers)
    assert projection.raw_sampled_action is original
    assert torch.equal(original.presence,presence)
    assert 0<len(projection.projected_bundle.actions)<int(presence.sum())
    assert projection.dropped
    # Reevaluation consumes original sampled action, not projected presence.
    records=bind_marl_candidates(routing=capture['routing'],presence_logits=capture['raw_prediction']['presence_logits'],presence_threshold_logit=4.5)
    from drl.MARL.policy import compute_owner_presence_log_probs
    owners=tuple(sorted({r.owner_agent_id for r in records}))
    lp=compute_owner_presence_log_probs(records=records,sampled_presence=original.presence,
        hard_mask=capture['hard_mask'],possible_agents=owners)
    expected=sum(torch.distributions.Bernoulli(logits=r.presence_logit-4.5).log_prob(presence[r.candidate_index]) for r in records)
    assert torch.allclose(sum(lp.values()),expected)
    from drl.MARL.mappo import StructuredRolloutSample,StructuredRolloutSidecar
    sidecar=StructuredRolloutSidecar()
    sidecar.append_step([StructuredRolloutSample(original,capture['hard_mask'],{o:True for o in owners},
        capture['routing'],capture['actor_graph'])])
    stored=sidecar.sample_by_index([0])[0]
    assert stored.action is original
    assert torch.equal(stored.action.presence,presence)


def test_partial_projection_events_feed_existing_raw_penalty_terms():
    from drl.MARL.tests.test_phase1_reward import _owners_at
    from drl.MARL.phase1_reward import compute_phase1_raw_reward
    from smores_ep.isaac.phase1_training_course import phase1_rc_car_training_course
    positions=_owners_at(.5,0.)
    outcome=SMORESMARLProposalOutcome(True,'ACCEPTED','partial',('smores_01',),
        conflict_owner_ids=('smores_02',),invalid_owner_ids=('smores_03',))
    signals=collect_runtime_signals(owner_ids=tuple(positions),delta_seconds=1/30,proposal_outcome=outcome).signals
    raw=compute_phase1_raw_reward(phase1_rc_car_training_course(5102),positions,positions,signals=signals)
    assert raw.owners['smores_01'].conflict==raw.owners['smores_01'].invalid==0.
    assert raw.owners['smores_02'].conflict==1.
    assert raw.owners['smores_03'].invalid==1.
    assert raw.team.conflict==raw.team.invalid==pytest.approx(1/8)


def test_native_primitive_preflight_does_not_admit_or_mutate_trackers():
    from copy import deepcopy
    from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor
    from scripts.smores_ep.tests.test_primitives import _FakeDocking,_FakeStateReader,_goal
    from smores_ep.primitives.model import PrimitiveName
    executor=IsaacPrimitiveExecutor(stage=object(),module_roots={'module_a':'/A','module_b':'/B'},
        states={'module_a':_FakeStateReader(),'module_b':_FakeStateReader()},docking=_FakeDocking())
    before=deepcopy(vars(executor._pan_trackers['module_a']))
    goal=_goal('pan1',PrimitiveName.SET_PAN,('module_a',),dict(angle_rad=.1))
    assert executor.check_admission(goal,0.) is None
    assert not executor._active and not executor._resource_owners
    assert vars(executor._pan_trackers['module_a'])==before
    executor.submit(goal,0.)
    other=_goal('pan2',PrimitiveName.SET_PAN,('module_a',),dict(angle_rad=.2))
    rejected=executor.check_admission(other,.1)
    assert rejected.code=='RESOURCE_BUSY'
    assert len(executor._active)==1


@pytest.mark.parametrize('fault',['nonfinite_template','adapter_schema'])
def test_projection_infrastructure_errors_are_not_policy_penalties(fault):
    from pathlib import Path
    from drl.MARL.c1_runner import build_observation_bridge
    from drl.MARL.smores_runtime_binding import SMORESMARLRuntimeBinding
    from drl.MARL.projected_runtime import project_marl_action
    from drl.IL.bc_v1.live_policy import smores_claims
    root=Path(__file__).resolve().parents[3]
    path=root/'logs/isaaclab_c1_native/redesign/structural_conflict_diagnostic/navigation/decision_000.pt'
    if not path.exists():pytest.skip('finished reference capture unavailable')
    c=torch.load(path,map_location='cpu',weights_only=False)
    bridge=build_observation_bridge(root)
    adapter=bridge.adapter if fault=='nonfinite_template' else SimpleNamespace(decode_action=lambda choice:{'bogus':1})
    binding=SMORESMARLRuntimeBinding(adapter=adapter,claim_resolver=smores_claims)
    if fault=='nonfinite_template':
        c['raw_prediction']['parameters']['resource']['payload'][0]=float('nan')
    with pytest.raises(ValueError):
        project_marl_action(binding=binding,action=c['sampled_action'],prediction_template=c['raw_prediction'],
            routing=c['routing'],batch=c['decoder_batch'],normalizers=bridge.normalizers)
