#!/usr/bin/env python3
"""One bounded frozen BC-v1 RC8/navigation smoke; deterministic setup ends first.

Standalone Isaac runtime + existing pure assembly planner/executor; no ROS,
Nav2, teleop, dataset recorder, training, or alternative learned preprocessing.
"""
from pathlib import Path
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[2]
for path in (ROOT,ROOT/'scripts/smores_ep/src',ROOT/'mssr_ws/src/mssr_expert',ROOT/'scripts/teleop'):
    if str(path) not in sys.path:sys.path.insert(0,str(path))

from smores_ep.primitives.file_channel import PrimitiveFileChannel
from smores_ep.primitives.bc_runtime import FileBcTransport
from mssr_expert.graph.graph_builder import GraphBuilder
from mssr_expert.graph.serialization import load_attributed_graph
from mssr_expert.planning.smores_ep.parallel_self_assembly_planner import ParallelSelfAssemblyPlanner
from mssr_expert.execution.parallel_assembly_executor import ParallelAssemblyExecutor,physical_posture_groups,physical_fold_push_pairs
from mssr_expert.execution.assembly_policy import DEFAULT_ASSEMBLY_EXECUTION_POLICY
from check_mobile_manipulator import isaac_subprocess_environment


class UnexpectedStructuralProposal(ValueError):pass


def smoke_inference(policy,snapshot,feedback):
    commands=policy(snapshot,feedback)
    if any(command.key[1]=='structural' for command in commands):
        raise UnexpectedStructuralProposal('unexpected structural proposal: '+json.dumps([c.payload for c in commands if c.key[1]=='structural']))
    return commands


def runtime_command(root,directory):
    root,directory=Path(root),Path(directory)
    # Headless --steps 0 becomes 7200 (30 SIM seconds) in the existing CLI.
    # Allow setup to finish; the separate BC window still starts at arming.
    return [str(root/'scripts/smores_ep/run_self_assembly.sh'),'--headless','--performance','--simple-visuals',
        '--module-count','8','--physics-hz','240','--state-publish-hz','30','--steps','240000',
        '--rc-car-planar-test-course','--rc-car-seed','5100','--simulation-speed-factor','1.0',
        '--actuator-effort-scale','4.0','--wheel-friction-scale','1.50','--tilt-effort-scale','8.0',
        '--action-file',str(directory/'actions.json'),'--primitive-goal-file',str(directory/'primitive_goal.json'),
        '--primitive-cancel-file',str(directory/'primitive_cancel.json'),'--primitive-status-file',str(directory/'primitive_status.json')]


def smoke_bound_reached(armed_wall,armed_stamp,*,wall_now,physical_stamp):
    """The L1c evaluation budget excludes all pre-evaluation setup time."""
    return armed_wall is not None and (
        wall_now-armed_wall>=120. or physical_stamp-armed_stamp>=5.)


def make_setup_executor(root,current,execution_id):
    target=load_attributed_graph(Path(root)/'mssr_ws/src/mssr_expert/config/smores_rc_car8.json')
    result=ParallelSelfAssemblyPlanner().plan(current_graph=current,target_graph=target)
    attrs=result.target_graph.global_attributes;assignment=result.assignment.target_to_module
    tilts={assignment[v]:float(x) for v,x in attrs.get('post_assembly_tilt_rad_by_vertex',{}).items()}
    pans={assignment[v]:float(x) for v,x in attrs.get('post_assembly_pan_rad_by_vertex',{}).items()}
    return ParallelAssemblyExecutor(plan=result.assembly_plan,execution_id=execution_id,
        **DEFAULT_ASSEMBLY_EXECUTION_POLICY.executor_kwargs(),
        post_assembly_tilt_by_module=tilts,post_assembly_pan_by_module=pans,
        posture_tilt_tolerance_rad=attrs.get('post_assembly_tilt_tolerance_rad'),
        posture_tilt_max_servo_error_rad=attrs.get('post_assembly_tilt_max_servo_error_rad'),
        coordinate_posture_tilts=bool(attrs.get('coordinate_post_assembly_tilts',False)),
        posture_tilt_groups_by_module=physical_posture_groups(attrs.get('post_assembly_tilt_groups_by_vertex'),assignment,set(tilts)),
        posture_push_by_lifter_module=physical_fold_push_pairs(attrs.get('post_assembly_push_pairs_by_vertex'),assignment,set(tilts)),
        additional_known_module_ids=result.reserve_module_ids)


def read_json(path):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return None


def write_json(path,value):
    PrimitiveFileChannel._write_atomic(Path(path),json.dumps(value,indent=2,allow_nan=False)+'\n')


def serializable_owner_status(status):
    """Serialize the one explicitly set-valued transport field, without mutation."""
    return None if status is None else dict(status,busy=sorted(status['busy']))


def stop_process(process):
    if process is None or process.poll() is not None:return
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=10.)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5.)


def pose(raw):
    modules=raw['state']['modules']
    positions=[m['pose']['position'] for m in modules]
    return dict(centroid=[sum(p[i] for p in positions)/len(positions) for i in range(3)],
        module_poses={m['module_id']:m['pose'] for m in modules})


def run(output,checkpoint,cleanup_manifest,*,expected_checkpoint_sha256=None):
    from drl.IL.bc_v1.live_controller import LiveController,PhysicalSnapshot
    from drl.IL.bc_v1.live_policy import load_frozen_policy
    import torch
    cleanup=read_json(cleanup_manifest)
    if not cleanup or not cleanup.get('passed') or time.monotonic()-cleanup['wall_monotonic']>120.:
        raise ValueError('Fresh scoped cleanup evidence is required before Isaac launch')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    runtime_dir=output/'runtime';runtime_dir.mkdir()
    session=uuid4().hex
    torch.set_num_threads(1)
    events_file=(output/'controller_events.jsonl').open('w',buffering=1)
    def event(row):events_file.write(json.dumps(row,allow_nan=False)+'\n')
    try:
        policy=load_frozen_policy(ROOT,checkpoint,GraphBuilder(),event_sink=event,
            expected_checkpoint_sha256=expected_checkpoint_sha256)
    except Exception:
        events_file.close()
        raise
    config=dict(morphology='rc_car8',task='navigation',seed=5100,policy_max_hz=10.,state_hz_sim=30,
        physics_hz_sim=240,state_stale_s=1.,holder_period_s=.1,sim_bound_s=5.,wall_bound_s=120.,
        checkpoint=str(checkpoint),session=session,structural_safety='terminate before dispatch',
        recorders_enabled=False,nav2_enabled=False,teleop_enabled=False,setup='existing deterministic planner/executor, pre-evaluation only',
        **policy.checkpoint_provenance)
    write_json(output/'run_config.json',config)
    process=None;controller=None;transport=None;initial=None;final=None;armed_wall=None;armed_stamp=None
    failure=None;runtime_status=None;setup_states=[]
    try:
        for name,value in [('actions.json',{'schema_version':'mssr.actions.v2','locomotion':{}}),
                ('primitive_goal.json',{}),('primitive_cancel.json',{})]:write_json(runtime_dir/name,value)
        command=runtime_command(ROOT,runtime_dir)
        env=isaac_subprocess_environment(os.environ);env.update(MSSR_BC_SESSION=session,PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
        log=(output/'runtime.log').open('w')
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(output/'process_manifest.json',dict(runtime_pid=process.pid,command=command,cwd=str(ROOT),
            action_sources={'expert':'setup only, in this process','Nav2':False,'teleop':False},dataset_recorders=False))
        deadline=time.monotonic()+180.
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError('Isaac exited during startup: '+str(process.returncode))
            raw=read_json(runtime_dir/'state_graph.json');status=read_json(runtime_dir/'bc_status.json')
            if raw and status and status.get('published_stamp') is not None:break
            time.sleep(.1)
        else:raise TimeoutError('Isaac state startup timeout')
        setup=make_setup_executor(ROOT,GraphBuilder().build(raw['state'],raw['graph']),'setup-'+session)
        setup_deadline=time.monotonic()+900.;last_goal=None;last_setup_state=None
        event(dict(event='setup_started',wall_monotonic=time.monotonic(),sim_stamp=raw['stamp']))
        while time.monotonic()<setup_deadline:
            if process.poll() is not None:raise RuntimeError('Isaac exited during assembly setup')
            statuses=read_json(runtime_dir/'primitive_status.json') or {}
            decision=setup.step(statuses)
            if decision.state!=last_setup_state:
                last_setup_state=decision.state;setup_states.append(decision.state)
                event(dict(event='setup_state',wall_monotonic=time.monotonic(),state=decision.state,phase=decision.phase))
                write_json(output/'progress.json',dict(phase='SETUP',state=decision.state,wall=time.monotonic()))
            if decision.primitive_goal is not None and decision.primitive_goal.goal_id!=last_goal:
                write_json(runtime_dir/'primitive_goal.json',decision.primitive_goal.to_dict())
                last_goal=decision.primitive_goal.goal_id
            if decision.done:
                if not decision.success:raise RuntimeError('Deterministic setup failed: '+decision.message)
                break
            time.sleep(.05)
        else:raise TimeoutError('Pre-evaluation setup timeout')
        setup=None  # No setup step or expert source is called after this boundary.
        deadline=time.monotonic()+10.
        while time.monotonic()<deadline:
            status=read_json(runtime_dir/'bc_status.json')
            if not status['active_goal_ids']:break
            time.sleep(.05)
        else:raise RuntimeError('Native setup goal still active at BC handoff')
        if status['session']!=session:raise RuntimeError('Runtime session mismatch before handoff')
        transport=FileBcTransport(runtime_dir,session,status['epoch']);transport.arm()
        deadline=time.monotonic()+5.
        while time.monotonic()<deadline:
            status=transport.poll()
            if status.get('failure'):raise RuntimeError(status['failure'])
            if status['armed'] and status['last_ack'] and status['last_ack']['kind']=='arm':break
            time.sleep(.01)
        else:raise RuntimeError('Missing owner handoff ACK')
        handoff_stamp=status['last_ack']['sim_stamp']
        event(dict(event='expert_leakage_gate',wall_monotonic=time.monotonic(),handoff_sim_stamp=handoff_stamp,
            expert_active=False,nav2_active=False,teleop_active=False,setup_executor_released=True,
            native_goals=[],old_sources_quarantined=True,owner_epoch=status['epoch']))
        controller=LiveController(session,status['epoch'],lambda s,f:smoke_inference(policy,s,f),transport,event_sink=event)
        startup_deadline=time.monotonic()+10.
        while True:
            if process.poll() is not None:raise RuntimeError('Isaac exited during BC phase')
            raw=read_json(runtime_dir/'state_graph.json');runtime_status=transport.poll()
            if runtime_status.get('failure'):raise RuntimeError(runtime_status['failure'])
            snap=None
            if raw and raw['stamp']>handoff_stamp and raw['stamp']<=runtime_status['published_stamp']:
                stamp=raw['stamp']
                if raw['schema_version']!='mssr.state_graph.v2' or raw['state']['stamp']!=stamp or raw['graph']['stamp']!=stamp:
                    raise ValueError('Invalid combined live snapshot')
                snap=PhysicalSnapshot(status['epoch'],stamp,raw)
                final=raw
            controller.tick(snap)
            if controller.armed and armed_wall is None:
                arm_event=next(e for e in controller.events if e['event']=='authority_armed')
                armed_wall=arm_event['wall_monotonic'];armed_stamp=arm_event['stamp'];initial=raw
                write_json(output/'progress.json',dict(phase='BC',armed_wall=armed_wall,armed_stamp=armed_stamp))
            if controller.failed:
                raise RuntimeError(controller.events[-1].get('reason','BC controller failed'))
            if armed_wall is not None:
                if smoke_bound_reached(armed_wall,armed_stamp,
                        wall_now=time.monotonic(),physical_stamp=final['stamp']):break
            elif time.monotonic()>startup_deadline:raise TimeoutError('BC physical-progress arming timeout')
            time.sleep(.01)
    except Exception as error:
        failure=str(error);event(dict(event='smoke_failure',wall_monotonic=time.monotonic(),reason=failure))
    finally:
        if transport is not None:
            try:
                runtime_status=transport.poll()
                transport.clear('bounded smoke ended' if failure is None else failure)
                time.sleep(.1)
                runtime_status=transport.poll()
            except Exception as error:
                if failure is None:failure='Final withdrawal error: '+str(error)
        stop_process(process)
        if controller is not None:controller.close()
        if final is not None:write_json(output/'final_state.json',final)
        else:write_json(output/'final_state.json',{'unavailable':True})
        all_events=controller.events if controller is not None else []
        starts=[e for e in all_events if e['event']=='inference_started']
        decisions=[e for e in all_events if e['event']=='decision_completed']
        refresh=[e for e in all_events if e['event']=='holder_refresh']
        intervals=[b['wall_monotonic']-a['wall_monotonic'] for a,b in zip(starts,starts[1:])]
        integration=(failure is None and len(decisions)>=3 and runtime_status is not None and runtime_status['applied_packets']>0)
        behavior='INSUFFICIENT_SMOKE_DURATION'
        observation={}
        if initial and final:
            first,last=pose(initial),pose(final);delta=[b-a for a,b in zip(first['centroid'],last['centroid'])]
            displacement=math.hypot(*delta[:2]);observation=dict(initial=first,final=last,delta_xyz=delta,displacement_xy=displacement)
            goal=initial['state']['course']['physical_track']['goal_xyyaw']
            before=math.dist(first['centroid'][:2],goal[:2]);after=math.dist(last['centroid'][:2],goal[:2])
            observation.update(initial_goal_distance=before,final_goal_distance=after)
            behavior='STATIONARY' if displacement<.005 else 'MOVES_TOWARD_GOAL' if after<before else 'MOVES_BUT_NOT_CLEARLY_TOWARD_GOAL'
        if failure and 'unexpected structural proposal' in failure:behavior='UNEXPECTED_STRUCTURAL_PROPOSAL'
        summary=dict(runtime_integration_success=integration,policy_behavior=behavior,failure=failure,
            session=session,armed_wall=armed_wall,armed_stamp=armed_stamp,decision_count=len(decisions),
            inference_count=len(starts),holder_refresh_count=len(refresh),decision_intervals_s=intervals,
            backlog_count=0,forward_count=policy.forward_count,timings=policy.timings,
            candidate_domain=policy.last_candidate_domain,owner_status=serializable_owner_status(runtime_status),
            physical_observation=observation,setup_states=setup_states,expert_leakage=False,
            TEST_access=0,TRAIN_access=0,VAL_access=0,training=0,optimizer=0,backward=0,
            **policy.checkpoint_provenance)
        write_json(output/'decision_summary.json',summary)
        write_json(output/'progress.json',dict(phase='FINISHED',success=integration,failure=failure))
        (output/'smoke_report.md').write_text('# One bounded RC8 seed5100 smoke\n\n'+
            f'Runtime integration success: {integration}\n\nPolicy behavior: {behavior}\n\nFailure: {failure}\n\n'+
            'See controller_events.jsonl, decision_summary.json and process_manifest.json. No further smoke or tuning performed.\n')
        events_file.close()
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=Path('/home/lorenzo/MSSR_thesis/artifacts/bc_v1/checkpoints/k1_baseline_v1/best_checkpoint.pt'))
    parser.add_argument('--expected-checkpoint-sha256',help='Optional integrity pin checked before runtime launch')
    parser.add_argument('--cleanup-manifest',type=Path,required=True)
    args=parser.parse_args();summary=run(args.output,args.checkpoint,args.cleanup_manifest,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256)
    print(json.dumps({k:summary[k] for k in ('runtime_integration_success','policy_behavior','failure','decision_count')},indent=2))
    return 0 if summary['runtime_integration_success'] else 1


if __name__=='__main__':raise SystemExit(main())
