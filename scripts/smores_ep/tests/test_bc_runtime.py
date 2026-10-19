"""Owner-side BC admission reuses actual native executor and file quarantine."""
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from smores_ep.isaac.primitive_executor import IsaacPrimitiveExecutor
from smores_ep.primitives.file_channel import ActionFileChannel, PrimitiveFileChannel
from smores_ep.primitives.model import PrimitiveGoal, PrimitiveState
from smores_ep.scenarios.parallel_self_assembly import invalidate_module_command_sources


def api():
    name='smores_ep.primitives.bc_runtime'
    assert importlib.util.find_spec(name), 'Missing BC owner admission and quarantine notification'
    return importlib.import_module(name)


class State:
    def read(self):return SimpleNamespace(pan_joint_rad=0.,tilt_joint_rad=0.)


def setup(tmp_path):
    a=api();clock=SimpleNamespace(now=0.)
    action=tmp_path/'actions.json'
    channel=ActionFileChannel(action)
    executor=IsaacPrimitiveExecutor(stage=object(),module_roots={'m1':'/M1'},states={'m1':State()},
        docking=SimpleNamespace(connections=(),module_ids=('m1',)))
    bridge=None
    def invalidate(ids):
        invalidate_module_command_sources(ids,action_channel=channel,held_primitive_commands={})
        if bridge is not None:bridge.invalidate(ids)
    executor.invalidate_module_command_sources_callback=invalidate
    bridge=a.BcRuntimeBridge(tmp_path,'session',channel,executor,invalidate,('m1',),clock=lambda:clock.now)
    return bridge,clock,channel,executor


def request(bridge,kind='update',rows=None,goals=(),**extra):
    value=dict(kind=kind,session='session',epoch=bridge.epoch,state_stamp=1.,state_deadline=.9,
        continuous=[] if rows is None else rows,goals=list(goals))
    value.update(extra)
    PrimitiveFileChannel._write_atomic(bridge.request_file,json.dumps(value))
    bridge.service(1.)


def row(generation='bc:session:1'):
    return dict(key=['m1','internal_motion'],generation=generation,source_stamp=1.,
        payload={'command':'pan_velocity','module_ids':['m1'],'parameters':{'rate_rad_s':2.}})


def armed(tmp_path):
    b,t,c,e=setup(tmp_path);request(b,'arm');assert b.armed
    return b,t,c,e


def test_owner_requires_session_and_explicit_arm(tmp_path):
    b,t,c,e=setup(tmp_path);request(b,rows=[row()])
    assert not b.armed and not c.commands(t.now)


def test_owner_rejects_foreign_epoch(tmp_path):
    b,t,c,e=armed(tmp_path);request(b,rows=[row()],epoch='foreign')
    assert not c.commands(t.now) and b.failure


def test_owner_accepts_physical_pan_and_zero_uses_existing_hold(tmp_path):
    from smores_ep.control.teleop import InternalMotionMode
    b,t,c,e=armed(tmp_path);request(b,rows=[row()])
    assert c.commands(t.now)['m1'].pan_velocity_rad_s==2.
    zero=row('bc:session:2');zero['payload']['parameters']['rate_rad_s']=0.
    request(b,rows=[zero])
    assert c.commands(t.now)['m1'].internal_motion is InternalMotionMode.HOLD


def test_gravity_settle_existing_quarantine_notifies_holder_and_never_resumes(tmp_path):
    b,t,c,e=armed(tmp_path);request(b,rows=[row()]);assert c.commands(0.)
    goal=PrimitiveGoal.from_dict(dict(goal_id='settle',primitive='gravity_settle',module_ids=['m1'],
        parameters={'duration_s':.2,'passive_module_ids':['m1'],'clear_passive_policy_on_finish':True}))
    assert e.submit(goal,1.).state is PrimitiveState.ACCEPTED
    e.step(1.)  # Actual existing native callback, no second settle implementation.
    assert not c.commands(0.)
    assert (('m1','internal_motion'),'bc:session:1') in b.invalidated
    e.step(1.3)
    request(b,rows=[row()])  # Late keepalive of the old generation.
    assert not c.commands(0.)


def test_owner_final_busy_recheck_rejects_a_racing_continuous(tmp_path):
    b,t,c,e=armed(tmp_path)
    goal=PrimitiveGoal.from_dict(dict(goal_id='tilt',primitive='set_tilt',module_ids=['m1'],parameters={'angle_rad':.2}))
    e.submit(goal,1.)
    request(b,rows=[row()])
    assert not c.commands(0.) and b.rejections


def test_owner_deadline_withdraws_cached_action_without_cancelling_native(tmp_path):
    b,t,c,e=armed(tmp_path);request(b,rows=[row()]);assert c.commands(0.)
    t.now=.9;b.service(1.)
    assert b.failure and not c.commands(.9)


def test_owner_repeated_goal_id_is_not_dispatched_again(tmp_path):
    b,t,c,e=armed(tmp_path)
    goal=dict(generation='bc:session:1',key=['m1','internal_motion'],source_stamp=1.,
        payload={'command':'set_tilt','module_ids':['m1'],'parameters':{'angle_rad':.2}})
    request(b,goals=[goal]);request(b,goals=[goal])
    assert len(e.active_goals)==1 and not b.failure


def test_last_owner_recheck_withdraws_pan_if_native_goal_arrives_after_packet(tmp_path):
    b,t,c,e=armed(tmp_path);request(b,rows=[row()])
    baseline=c.commands(0.);assert baseline
    goal=PrimitiveGoal.from_dict(dict(goal_id='tilt-race',primitive='set_tilt',module_ids=['m1'],parameters={'angle_rad':.2}))
    e.submit(goal,1.)
    assert not b.filter_commands(baseline)
    assert (('m1','internal_motion'),'bc:session:1') in b.invalidated


def test_optional_factory_preserves_native_settle_callback_and_notifies_bridge(tmp_path,monkeypatch):
    b,t,c,e=setup(tmp_path)
    monkeypatch.setenv('MSSR_BC_SESSION','factory-session')
    result=api().attach_bc_runtime(tmp_path,c,e,('m1',))
    assert result.session=='factory-session'
    result._held[('m1','internal_motion')]=row()
    e.invalidate_module_command_sources_callback(('m1',))
    assert not result._held
    assert (('m1','internal_motion'),'bc:session:1') in result.invalidated


def test_owner_rejects_non_bc_producer_during_bc_phase(tmp_path):
    b,t,c,e=armed(tmp_path);request(b,rows=[row()])
    PrimitiveFileChannel._write_atomic(tmp_path/'actions.json',json.dumps({'locomotion':{'m1':{'vx':.1}}}))
    baseline=c.commands(0.)
    assert not b.filter_commands(baseline) and b.failure


def test_file_transport_keeps_generation_and_exposes_native_revocation(tmp_path):
    from drl.IL.bc_v1.live_controller import HeldCommand,RuntimeCommand
    b,t,c,e=armed(tmp_path)
    transport=api().FileBcTransport(tmp_path,'session',b.epoch)
    command=RuntimeCommand(('m1','internal_motion'),row()['payload'],('internal_motion:m1',),True)
    held={command.key:HeldCommand(command,'bc:session:1',1.)}
    proof=dict(session='session',epoch=b.epoch,state_stamp=1.,state_deadline=.9)
    transport.publish(held,(),proof);b.service(1.)
    transport.publish(held,(),proof);b.service(1.)
    request_data=json.loads(b.request_file.read_text())
    assert request_data['continuous'][0]['generation']=='bc:session:1'
    e.invalidate_module_command_sources_callback(('m1',));b.publish_status(1.)
    assert (('m1','internal_motion'),'bc:session:1') in transport.poll()['invalidated']


def test_file_transport_session_mismatch_fails_closed(tmp_path):
    b,t,c,e=armed(tmp_path);b.publish_status(1.)
    transport=api().FileBcTransport(tmp_path,'different',b.epoch)
    with pytest.raises(ValueError,match='session'):
        transport.poll()


def test_target_mailbox_keeps_goal_until_actual_owner_ack(tmp_path):
    from drl.IL.bc_v1.live_controller import HeldCommand,RuntimeCommand
    b,t,c,e=armed(tmp_path);transport=api().FileBcTransport(tmp_path,'session',b.epoch)
    command=RuntimeCommand(('m1','internal_motion'),
        {'command':'set_tilt','module_ids':['m1'],'parameters':{'angle_rad':.2}},('internal_motion:m1',),False)
    goal=HeldCommand(command,'bc:session:1',1.)
    proof=dict(session='session',epoch=b.epoch,state_stamp=1.,state_deadline=.9)
    transport.publish({},(goal,),proof)
    transport.publish({},(),proof)  # Holder faster than runtime consumption.
    assert len(json.loads(b.request_file.read_text())['goals'])==1
    b.service(1.);transport.poll();transport.publish({},(),proof)
    assert not json.loads(b.request_file.read_text())['goals']
    assert len(e.active_goals)==1


def test_empty_new_structural_decision_does_not_cancel_active_primitive(tmp_path):
    b,t,c,e=armed(tmp_path)
    goal=dict(generation='bc:session:1',key=['m1','internal_motion'],source_stamp=1.,
        payload={'command':'set_tilt','module_ids':['m1'],'parameters':{'angle_rad':.2}})
    request(b,goals=[goal]);old=e.active_goals[0].goal_id
    request(b,goals=())  # NO_STRUCTURAL is absence, never a cancel/replace packet.
    assert [g.goal_id for g in e.active_goals]==[old]


def test_conflicting_native_proposal_rejected_without_replacing_old_goal(tmp_path):
    b,t,c,e=armed(tmp_path)
    def target(generation):return dict(generation=generation,key=['m1','internal_motion'],source_stamp=1.,
        payload={'command':'set_tilt','module_ids':['m1'],'parameters':{'angle_rad':.2}})
    request(b,goals=[target('bc:session:1')]);old=e.active_goals[0].goal_id
    request(b,goals=[target('bc:session:2')])
    assert [g.goal_id for g in e.active_goals]==[old] and b.rejections


def test_disjoint_native_resource_proposal_runs_in_parallel_without_replacing_old(tmp_path):
    channel=ActionFileChannel(tmp_path/'actions.json')
    executor=IsaacPrimitiveExecutor(stage=object(),module_roots={'m1':'/M1','m2':'/M2'},states={'m1':State(),'m2':State()},
        docking=SimpleNamespace(connections=(),module_ids=('m1','m2')))
    b=api().BcRuntimeBridge(tmp_path,'session',channel,executor,lambda ids:None,('m1','m2'),clock=lambda:0.)
    request(b,'arm')
    def target(mid,generation):return dict(generation=generation,key=[mid,'internal_motion'],source_stamp=1.,
        payload={'command':'set_tilt','module_ids':[mid],'parameters':{'angle_rad':.2}})
    request(b,goals=[target('m1','bc:session:1')]);old=executor.active_goals[0].goal_id
    request(b,goals=[target('m2','bc:session:2')])
    assert len(executor.active_goals)==2 and old in [g.goal_id for g in executor.active_goals]
