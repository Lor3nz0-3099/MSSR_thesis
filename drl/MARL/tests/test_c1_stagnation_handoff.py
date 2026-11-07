"""Prepared ALIGN -> DOCK must survive the post-physics terminal boundary."""
from dataclasses import replace
from types import SimpleNamespace as NS
import pytest

from drl.MARL.c1_structure import C1PrimitiveEvent
from drl.MARL.tests.test_c1_stagnation import monitor
from drl.MARL.tests.test_c1_structure import frame

PAIR=(('m0','TOP'),('m1','BOTTOM'))
GOAL=dict(primitive='align_faces',module_ids=('m0','m1'),owner_module_id='m0',connectors=PAIR)


def near_observed_boundary():
    m=monitor()
    for i in range(40):
        m.step(frame(float(i),edges=0),structural_score=0.,decision_opportunity=True)
    m.step(frame(40.3,edges=0),structural_score=1.,decision_opportunity=True)
    for t in (42.,44.,46.,48.,50.,52.):
        m.step(frame(t,edges=0),structural_score=1.,decision_opportunity=True)
    m.step(frame(57.733333333333334,edges=0),structural_score=1.,
        decision_opportunity=True,active_goals={'last-align':GOAL})
    m.step(frame(57.766666666666666,edges=0),structural_score=1.,
        decision_opportunity=True,active_goals={'last-align':GOAL})
    r=m.step(frame(65.6,edges=0),structural_score=1.,active_goals={'last-align':GOAL})
    assert not r.terminate and r.global_stagnant and r.protected_option_ids
    assert r.decision_opportunities==9 and r.no_useful_progress_s==pytest.approx(25.3)
    return m


def test_successful_align_does_not_terminate_before_the_next_policy_decision():
    m=near_observed_boundary()
    f=replace(frame(65.63333333333333,edges=0),primitive_events=(
        C1PrimitiveEvent('last-align','align_faces','succeeded','FACES_IN_CONTACT','m0'),))
    r=m.step(f,structural_score=1.,active_goals={})
    assert not r.terminate
    assert r.decision_opportunities==0
    assert r.recovery_until_s==pytest.approx(f.time_s+5.)


def test_precompletion_decision_cannot_count_as_post_handoff_opportunity():
    m=near_observed_boundary();now=65.63333333333333
    f=replace(frame(now,edges=0),primitive_events=(C1PrimitiveEvent('last-align','align_faces','succeeded','FACES_IN_CONTACT','m0'),))
    r=m.step(f,structural_score=1.,decision_opportunity=True)
    assert not r.terminate and r.decision_opportunities==0
    r=m.step(frame(now+6,edges=0),structural_score=1.)
    assert not r.terminate, 'No post-completion decision has yet been possible'
    for i in range(3):r=m.step(frame(now+7+i,edges=0),structural_score=1.,decision_opportunity=True)
    assert r.terminate, 'A prepared relation with actual chances must not exempt forever'


def test_ready_but_unsampled_dock_grants_once_then_expires_after_real_chances():
    m=near_observed_boundary();now=65.63333333333333
    r=m.step(frame(now,edges=0),structural_score=1.,prepared_relations=(PAIR,))
    assert not r.terminate and r.decision_opportunities==0
    for i in range(1,8):
        r=m.step(frame(now+i,edges=0),structural_score=1.,decision_opportunity=True,prepared_relations=(PAIR,))
    assert r.terminate and r.decision_opportunities==7


def test_new_goal_ids_for_same_successful_align_do_not_renew_handoff_forever():
    m=near_observed_boundary()
    for i in range(10):
        goal_id=f'retry{i}'
        f=replace(frame(65.63333333333333+i,edges=0),primitive_events=(
            C1PrimitiveEvent(goal_id,'align_faces','succeeded','FACES_IN_CONTACT','m0'),))
        r=m.step(f,structural_score=1.,decision_opportunity=True,action_goals={goal_id:GOAL})
    assert r.terminate and r.decision_opportunities>=3


def test_repeated_failed_aligns_do_not_renew_grace_forever():
    m=near_observed_boundary()
    for i in range(10):
        f=replace(frame(65.63333333333333+i,edges=0),primitive_events=(
            C1PrimitiveEvent(f'failed{i}','align_faces','failed','TIMEOUT','m0'),))
        r=m.step(f,structural_score=1.,decision_opportunity=True)
    assert r.terminate


def test_genuine_new_progress_can_earn_a_new_preparation_chance():
    m=near_observed_boundary()
    for i,score in enumerate((1.,2.)):
        now=65.63333333333333+i*10
        event=C1PrimitiveEvent(f'g{i}','align_faces','succeeded','FACES_IN_CONTACT','m0')
        r=m.step(replace(frame(now,edges=0),primitive_events=(event,)),structural_score=score,
            action_goals={f'g{i}':GOAL},prepared_relations=(PAIR,),decision_opportunity=True)
        assert not r.terminate and r.decision_opportunities==0


def test_native_success_frame_preserves_episode_until_dock_choice(tmp_path):
    from drl.MARL.c1_native_runtime import NativeC1RewardRuntime
    from drl.MARL.c1_action_events import C1ActionEvents
    from drl.MARL.tests.test_c1_native_runtime import env_fixture
    env=env_fixture();env.runtime_context_provider=NS(_captured=(NS(rollout_active=True),)*2)
    executor=NS(active_goals=(),prepared_relation_snapshot=lambda:())
    sibling=NS(active_goals=(),prepared_relation_snapshot=lambda:())
    env.native_dispatcher=NS(executor_provider=lambda i:executor if i==0 else sibling,continuous_commands=lambda i:{})
    env.c1_action_events=C1ActionEvents(None,masked_detail_limit=0)
    runtime=NativeC1RewardRuntime(settings_path='configs/marl/c1_guided_phase_c_v1.json',evidence_directory=tmp_path)
    for i in range(1,25):runtime.evaluate(env,0,frame(float(i),edges=0))
    env.c1_action_events.goals[(0,1,'align')]=GOAL
    env.runtime_context_provider._captured=(NS(rollout_active=False),)*2
    executor.active_goals=(NS(goal_id='align',primitive='align_faces',module_ids=('m0','m1')),)
    for i in range(25,33):assert not runtime.evaluate(env,0,frame(float(i),edges=0)).terminated
    executor.active_goals=()
    executor.prepared_relation_snapshot=lambda:(NS(source_module='m0',source_face='TOP',target_module='m1',target_face='BOTTOM',docked=False),)
    f=replace(frame(33.,edges=0),primitive_events=(C1PrimitiveEvent('align','align_faces','succeeded','FACES_IN_CONTACT','m0'),))
    r=runtime.evaluate(env,0,f)
    assert not r.terminated and not r.truncated
    assert not runtime.episode_records
    assert r.stagnation['decision_opportunities']==0
    assert r.stagnation['handoff_relations']
    # The unchanged scheduler can now see readiness in the next pre-step query.
    from drl.MARL.c1_training_options import NativeStructuralOptionController
    ctl=NativeStructuralOptionController(env,mode='event')
    assert ctl.decision_mask()[0]
    env.runtime_context_provider._captured=(NS(rollout_active=True),)*2
    docked=replace(frame(34.,edges=1),primitive_events=(C1PrimitiveEvent('dock','dock','succeeded','DOCKED','m0'),))
    after=runtime.evaluate(env,0,docked)
    assert not after.terminated and after.assessment.valid_connections==1
    # The sibling never sees env0's event or receives its preparation grace.
    env.runtime_context_provider._captured=(NS(rollout_active=True),)*2
    for i in range(1,28):
        other=runtime.evaluate(env,1,frame(float(i),edges=0))
        if other.terminated:break
    assert other.terminated and other.reason=='stagnation'


def test_completed_handoff_reopens_existing_native_clock_after_sleep():
    from drl.MARL.c1_training_options import NativeStructuralOptionController
    from drl.MARL.tests.test_c1_training_option_runtime import _Env,_goal
    env=_Env();clock=NativeStructuralOptionController(env,mode='event')
    m=near_observed_boundary()
    assert clock.decision_mask()[0]
    env.executors[0].active_goals=(_goal('last-align'),)
    assert clock.decision_mask()[0]
    assert not clock.decision_mask()[0]
    env.executors[0].active_goals=()
    env.executors[0].prepared=(NS(source_module='m0',source_face='TOP',target_module='m1',target_face='BOTTOM',docked=False),)
    f=replace(frame(65.63333333333333,edges=0),primitive_events=(C1PrimitiveEvent('last-align','align_faces','succeeded','FACES_IN_CONTACT','m0'),))
    assert not m.step(f,structural_score=1.,prepared_relations=(PAIR,)).terminate
    assert clock.decision_mask()[0], 'Readiness is observed after the completion frame survives'


def test_duplicate_success_event_and_reversed_prepared_pair_do_not_add_grace():
    m=near_observed_boundary();now=65.63333333333333
    event=C1PrimitiveEvent('last-align','align_faces','succeeded','FACES_IN_CONTACT','m0')
    r=m.step(replace(frame(now,edges=0),primitive_events=(event,)),structural_score=1.,prepared_relations=(PAIR,))
    assert len(r.handoff_relations)==1
    for i in range(1,8):
        r=m.step(replace(frame(now+i,edges=0),primitive_events=(event,)),structural_score=1.,
            decision_opportunity=True,prepared_relations=(tuple(reversed(PAIR)),))
        assert not r.handoff_relations
    assert r.terminate
