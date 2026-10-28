"""Optional BC file ingress on the existing Isaac owner thread.

Reuses native goal admission, ActionFileChannel and its quarantine. No expert,
model, scheduler, alternate executor, or general resource arbitration framework.
Only enabled for an explicitly identified BC session in a private runtime dir.
"""
import json
import math
import os
import time
from uuid import uuid4
from pathlib import Path

from smores_ep.config.geometry import SmoresGeometry
from smores_ep.primitives.file_channel import PrimitiveFileChannel
from smores_ep.primitives.model import PrimitiveGoal, PrimitiveState


class BcRuntimeBridge:
    def __init__(self,directory,session,action_channel,executor,invalidate_callback,module_ids,*,clock=time.monotonic,
                 lockstep=False,default_step_ticks=8):
        self.directory=Path(directory);self.session=session;self.epoch=uuid4().hex
        self.request_file=self.directory/'bc_request.json';self.status_file=self.directory/'bc_status.json'
        self.action_file=self.directory/'actions.json'
        self.channel=action_channel;self.executor=executor;self._invalidate=invalidate_callback
        self.modules=tuple(module_ids);self.clock=clock
        self.armed=False;self.failure=None;self.invalidated=set();self.rejections=[]
        self._held={};self._signature=None;self._deadline=None;self._goals=set();self._packet_id=''
        self._published_stamp=None;self.last_ack=None
        self.applied_packets=0;self.last_applied=None;self._applied_generation=None
        self.lockstep=bool(lockstep);self.default_step_ticks=default_step_ticks
        self._lockstep_started=False;self._pending=None;self._seen_commands=set()
        self.last_completed=None;self.needs_state_publish=False

    @property
    def requires_primitive_step(self):
        """Refresh admitted goals before the first acknowledged drive tick."""
        return (self.lockstep and self._pending is not None
                and self._pending['applied_stamp'] is None)

    @property
    def can_advance(self):
        """Expert runs freely; after explicit arm only a bounded command advances."""
        if not self.lockstep:return True
        if self.failure:return False
        if not self._lockstep_started:return True
        return self.armed and self._pending is not None

    def after_physics(self,sim_stamp):
        """Called once after each actual physics tick, before publishing its state."""
        if not self.lockstep or self._pending is None:return False
        pending=self._pending
        if pending['applied_stamp'] is None or sim_stamp<=pending['applied_stamp']:
            self._clear('Physics advanced without native application acknowledgement');return False
        pending['ticks']+=1
        if pending['ticks']<pending['step_ticks']:return False
        rejections=[dict(row) for row in self.rejections[pending['rejection_start']:]]
        code=rejections[0]['code'] if rejections else pending['outcome'].upper()
        self.last_completed=dict(command_id=pending['command_id'],applied_stamp=pending['applied_stamp'],
            completed_stamp=sim_stamp,ticks=pending['ticks'],outcome=pending['outcome'],
            code=code,rejections=rejections)
        self._pending=None;self._deadline=None;self.needs_state_publish=True
        return True

    def claims(self):
        return set(resource for goal in self.executor.active_goals for resource in self.executor._resources_for(goal))

    def invalidate(self,module_ids):
        """Notification AFTER existing native source invalidation, not a second invalidator."""
        for key,row in tuple(self._held.items()):
            if key[0] in module_ids:
                self.invalidated.add((key,row['generation']));self._held.pop(key)

    def _retire(self,key,row,reason,*,code=None):
        self.invalidated.add((key,row['generation']))
        self.rejections.append(dict(key=list(key),generation=row['generation'],reason=reason,
            code=code,owner_ids=[key[0]]))
        if self.lockstep and self._pending is not None:self._pending['outcome']='rejected'

    def _clear(self,reason):
        self.invalidate(self.modules)
        self._invalidate(self.modules)
        self._write_actions()
        self.failure=str(reason);self.armed=False
        if self.lockstep:self._pending=None

    def _write_actions(self):
        commands={};traction=[];geometry=SmoresGeometry()
        for (mid,resource),row in self._held.items():
            payload=row['payload'];params=payload['parameters']
            entry=commands.setdefault(mid,dict(vx=0.,yaw_rate=0.,pan_rate_rad_s=0.))
            if payload['command']=='pan_velocity':
                entry['pan_rate_rad_s']=float(params['rate_rad_s'])
                if abs(entry['pan_rate_rad_s'])>1e-12:traction.append(mid)
            elif payload['command']=='wheel_velocity':
                left,right=float(params['left_rad_s']),float(params['right_rad_s'])
                entry['vx']=geometry.wheel_radius_m*(left+right)/2
                entry['yaw_rate']=geometry.wheel_radius_m*(right-left)/geometry.track_width_m
            else:raise ValueError('Unsupported BC continuous actuator')
        packet={'schema_version':'mssr.actions.v2','source':'bc','locomotion':commands,
            'expert':{'pan_traction_module_ids':traction,'debug':{'command_id':self._packet_id}}}
        PrimitiveFileChannel._write_atomic(self.action_file,json.dumps(packet,allow_nan=False))

    def _validate_row(self,row):
        key=tuple(row['key']);payload=row['payload'];generation=row['generation']
        if (len(key)!=2 or key[0] not in self.modules or not generation.startswith(f'bc:{self.session}:')
                or payload['module_ids'][0]!=key[0]):raise ValueError('Invalid BC command routing')
        return key

    def service(self,sim_stamp):
        now=self.clock()
        if self.armed and self._deadline is not None and now>=self._deadline:
            self._clear('BC state lease expired at owner')
        snapshot=PrimitiveFileChannel._snapshot(self.request_file)
        if snapshot is None or snapshot[0]==self._signature:return
        self._signature,raw=snapshot
        try:
            request=json.loads(raw)
            if request['session']!=self.session or request['epoch']!=self.epoch:
                raise ValueError('BC session/epoch mismatch')
            kind=request['kind']
            if kind=='arm':
                if self.failure or self.armed:raise ValueError('BC auto-rearm prohibited')
                if self.executor.active_goals:raise ValueError('Native setup goals still active')
                self._invalidate(self.modules)
                self.armed=True;self.last_ack={'kind':'arm','wall':now,'sim_stamp':sim_stamp}
                if self.lockstep:self._lockstep_started=True;self.needs_state_publish=True
                self.publish_status();return
            if kind=='revoke':
                self._clear(request.get('reason','BC authority revoked'));self.publish_status();return
            if kind!='update' or not self.armed or self.failure:raise ValueError('BC authority is not armed')
            if self.lockstep:
                command_id=request.get('command_id')
                if (not isinstance(command_id,str) or not command_id.startswith(f'bc:{self.session}:')
                        or not command_id[len(f'bc:{self.session}:'):].isdigit()):
                    raise ValueError('Lockstep requires a unique BC command_id')
                if command_id in self._seen_commands:raise ValueError('Repeated lockstep command_id')
                if self._pending is not None:raise ValueError('Lockstep command still in flight')
                step_ticks=request.get('step_ticks',self.default_step_ticks)
                if type(step_ticks) is not int or not 1<=step_ticks<=240:
                    raise ValueError('Lockstep step_ticks must be in [1, 240]')
                self._seen_commands.add(command_id)
                rejection_start=len(self.rejections)
            deadline=request['state_deadline'];stamp=request['state_stamp']
            if (type(deadline) not in (int,float) or not math.isfinite(deadline) or not now<deadline<=now+1.01
                    or type(stamp) not in (int,float) or not math.isfinite(stamp) or stamp>sim_stamp):
                raise ValueError('Expired/invalid BC state proof')
            if self.lockstep and stamp!=sim_stamp:
                raise ValueError('Lockstep state proof must match the held physical state')
            self._deadline=deadline
            for row in request['goals']:
                key=self._validate_row(row);goal_id=row['generation']+':'+':'.join(key)
                if goal_id in self._goals:continue
                goal_deadline=row.get('dispatch_deadline',deadline)
                if not now<goal_deadline:
                    self._goals.add(goal_id);self._retire(key,row,'Expired native proposal');continue
                payload=row['payload']
                goal=PrimitiveGoal.from_dict(dict(goal_id=goal_id,primitive=payload['command'],
                    module_ids=payload['module_ids'],parameters=payload['parameters'],timeout_s=30.))
                status=self.executor.submit(goal,sim_stamp)
                self._goals.add(goal_id)
                if status.state is PrimitiveState.REJECTED:
                    self._retire(key,row,'native admission rejected',code=str(status.code));continue
                # Retire only overlapping continuous resources. SETTLE's module
                # quarantine remains the native executor callback on its first step.
                claims=set(self.executor._resources_for(goal))
                for oldkey,old in tuple(self._held.items()):
                    if f'{oldkey[1]}:{oldkey[0]}' in claims and payload['command']!='gravity_settle':
                        self.invalidated.add((oldkey,old['generation']));self._held.pop(oldkey)
            busy=self.claims();accepted={}
            for row in request['continuous']:
                key=self._validate_row(row)
                if (key,row['generation']) in self.invalidated:continue
                mid,resource=key
                expected={'pan_velocity':'internal_motion','wheel_velocity':'locomotion'}.get(row['payload']['command'])
                if expected!=resource:raise ValueError('Invalid BC actuator resource')
                physical_busy=f'{resource}:{mid}' in busy
                if expected=='internal_motion':
                    # Conservative PAN traction guard using CURRENT connectivity.
                    component=self.executor._connected_component_module_ids((mid,))
                    physical_busy |= any(f'locomotion:{m}' in busy for m in component)
                if physical_busy:
                    self._retire(key,row,'BUSY at owner dispatch',code='RESOURCE_BUSY');continue
                accepted[key]=row
            self._held=accepted
            self._packet_id=max((r['generation'] for r in accepted.values()),default='bc:'+self.session+':empty')
            if self.lockstep:
                self._packet_id=command_id
                proposed=bool(request['continuous'] or request['goals'])
                outcome='rejected' if len(self.rejections)>rejection_start else ('accepted' if proposed else 'empty')
                self._pending=dict(command_id=command_id,step_ticks=step_ticks,ticks=0,
                    applied_stamp=None,outcome=outcome,rejection_start=rejection_start)
            self._write_actions()
            self.last_ack={'kind':'update','wall':now,'sim_stamp':sim_stamp,
                'generations':sorted({r['generation'] for r in accepted.values()}),'physical_packet':bool(accepted)}
            self.publish_status()
        except (KeyError,TypeError,ValueError,OSError) as error:
            self._clear(error);self.publish_status()

    def filter_commands(self,baseline):
        """Last owner-side check, after native admissions and before drive apply."""
        if self.failure:return {}
        if not self.armed:return baseline
        if self._deadline is not None and self.clock()>=self._deadline:
            self._clear('BC state lease expired at drive boundary');return {}
        if self._deadline is not None and not self.channel.diagnostics.command_id.startswith('bc:'):
            self._clear('Non-BC command source during BC phase');return {}
        busy=self.claims();changed=False
        for key,row in tuple(self._held.items()):
            mid,resource=key
            blocked=f'{resource}:{mid}' in busy
            if resource=='internal_motion':
                blocked |= any(f'locomotion:{m}' in busy for m in self.executor._connected_component_module_ids((mid,)))
            if blocked:
                self._retire(key,row,'BUSY at final drive boundary',code='RESOURCE_BUSY')
                self._held.pop(key);changed=True
        if changed:
            self._write_actions()
            return self.channel.commands(self.clock())
        return baseline

    def record_applied(self,baseline,routed,sim_stamp):
        if not self.armed or (not baseline and not self.lockstep):return
        if self.lockstep:
            if self._pending is None:return
            if self._pending['applied_stamp'] is not None:return
            self._pending['applied_stamp']=sim_stamp
        if self._packet_id!=self._applied_generation:
            self.applied_packets+=1;self._applied_generation=self._packet_id
        self.last_applied=dict(generation=self._packet_id,sim_stamp=sim_stamp,wall_monotonic=self.clock(),
            pan={m:float(routed[m].pan_velocity_rad_s) for m in baseline if m in routed})
        if self.lockstep:
            self.last_applied.update(command_id=self._packet_id,applied_stamp=sim_stamp)
            self.publish_status()

    def publish_status(self,published_stamp=None):
        if published_stamp is not None:
            self._published_stamp=published_stamp;self.needs_state_publish=False
        status=dict(schema_version='mssr.bc_runtime_status.v1',session=self.session,epoch=self.epoch,pid=os.getpid(),
            wall_monotonic=self.clock(),published_stamp=self._published_stamp,armed=self.armed,failure=self.failure,
            busy=sorted(self.claims()),active_goal_ids=[g.goal_id for g in self.executor.active_goals],
            invalidated=[[list(k),g] for k,g in sorted(self.invalidated)],rejections=self.rejections,
            acknowledged_goal_ids=sorted(self._goals),last_ack=self.last_ack,applied_packets=self.applied_packets,last_applied=self.last_applied,
            lockstep=self.lockstep,last_completed=self.last_completed)
        PrimitiveFileChannel._write_atomic(self.status_file,json.dumps(status,allow_nan=False))


def attach_bc_runtime(directory,channel,executor,module_ids):
    session=os.environ.get('MSSR_BC_SESSION')
    if not session:return None
    invalidate=executor.invalidate_module_command_sources_callback
    if invalidate is None:raise ValueError('BC requires existing native source invalidation')
    bridge=BcRuntimeBridge(directory,session,channel,executor,invalidate,module_ids,
        lockstep=os.environ.get('MSSR_C1_LOCKSTEP','').lower() in ('1','true','yes'))
    def notified(ids):
        invalidate(ids)
        bridge.invalidate(ids)
    executor.invalidate_module_command_sources_callback=notified
    bridge.publish_status()
    return bridge


class FileBcTransport:
    """Thin external client. Same generation on refresh; owner is final authority."""
    def __init__(self,directory,session,epoch):
        self.directory=Path(directory);self.session=session;self.epoch=epoch
        self.request_file=self.directory/'bc_request.json';self.status_file=self.directory/'bc_status.json'
        self._pending_goals={}

    def _write(self,value):
        value=dict(value,session=self.session,epoch=self.epoch)
        PrimitiveFileChannel._write_atomic(self.request_file,json.dumps(value,allow_nan=False))

    def arm(self):self._write({'kind':'arm'})

    def poll(self):
        status=json.loads(self.status_file.read_text())
        if status['session']!=self.session or status['epoch']!=self.epoch:
            raise ValueError('BC runtime session/epoch mismatch')
        for goal_id in status.get('acknowledged_goal_ids',()):self._pending_goals.pop(goal_id,None)
        return dict(status,busy=set(status['busy']),
            invalidated=[(tuple(k),g) for k,g in status['invalidated']])

    def publish(self,held,goals,proof):
        def row(lease):
            return dict(key=list(lease.command.key),generation=lease.generation,source_stamp=lease.source_stamp,
                payload=lease.command.payload)
        for goal in goals:
            record=row(goal)
            record['dispatch_deadline']=goal.decision_deadline if goal.decision_deadline is not None else proof['state_deadline']
            self._pending_goals[goal.generation+':'+':'.join(goal.command.key)]=record
        self._write(dict(proof,kind='update',continuous=[row(v) for v in held.values()],goals=list(self._pending_goals.values())))

    def clear(self,reason):
        self._pending_goals.clear()
        self._write({'kind':'revoke','reason':str(reason)})
