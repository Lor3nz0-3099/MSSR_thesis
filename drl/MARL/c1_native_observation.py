"""Measured C1 evidence from native roots, fixed joints and actual course solids.

Read-only: no goals, constraints, forces, targets, masks or poses are written.
"""
import math

from drl.MARL.c1_structure import C1Module,C1Connection,C1Frame,C1PrimitiveEvent,norm,module_joint_status
from smores_ep.docking.model import DockingThresholds


def check_fixed_joint(stage,connection,world_transform):
    from pxr import UsdPhysics,Gf
    prim=stage.GetPrimAtPath(connection.joint_path)
    if not prim or not prim.IsActive() or not prim.IsA(UsdPhysics.FixedJoint):return False,0.,0.
    joint=UsdPhysics.FixedJoint(prim)
    if joint.GetJointEnabledAttr().Get() is False:return False,0.,0.
    bodies=(connection.first_face.rigid_body_path,connection.second_face.rigid_body_path)
    anchors=[]
    for i,body in enumerate(bodies):
        if tuple(map(str,getattr(joint,f'GetBody{i}Rel')().GetTargets()))!=(body,):return False,0.,0.
        body_prim=stage.GetPrimAtPath(body)
        if not body_prim or not body_prim.HasAPI(UsdPhysics.RigidBodyAPI):return False,0.,0.
        position=getattr(joint,f'GetLocalPos{i}Attr')().Get()
        rotation=getattr(joint,f'GetLocalRot{i}Attr')().Get()
        if position is None or rotation is None:return False,0.,0.
        transform=Gf.Transform();transform.SetTranslation(Gf.Vec3d(*position))
        transform.SetRotation(Gf.Rotation(Gf.Quatd(float(rotation.GetReal()),Gf.Vec3d(*rotation.GetImaginary()))))
        anchors.append(transform.GetMatrix()*world_transform(body))
    error=norm(tuple(anchors[0].ExtractTranslation()[k]-anchors[1].ExtractTranslation()[k] for k in range(3)))
    quats=[a.ExtractRotationQuat() for a in anchors]
    vectors=[(float(q.GetReal()),*(float(x) for x in q.GetImaginary())) for q in quats]
    cosine=abs(sum(x*y for x,y in zip(*vectors)))/(norm(vectors[0])*norm(vectors[1]))
    angular=2*math.acos(max(0.,min(1.,cosine)))
    limits=DockingThresholds()
    valid=math.isfinite(error) and math.isfinite(angular) and error<=limits.lateral_offset_tolerance_m and angular<=max(limits.normal_alignment_tolerance_rad,limits.clocking_tolerance_rad)
    return valid,error,angular


def solid_penetration_owners(course,modules):
    """Conservative solid-interior pathology, not a contact-force measurement.

    Only the measured module reference point penetrating beyond the established
    10mm docking/solver tolerance counts. Normal floor/obstacle contact and
    useful button face contact do not count as a collision by proximity alone.
    """
    tolerance=DockingThresholds().lateral_offset_tolerance_m
    owners=set()
    for module in modules:
        point=module.position
        if not all(math.isfinite(v) for v in point):continue
        for box in course.support_boxes:
            if not box.collidable:continue
            yaw=math.radians(box.yaw_deg);pitch=math.radians(box.pitch_deg)
            dx,dy,dz=(point[k]-box.center_xyz_m[k] for k in range(3))
            x=math.cos(yaw)*dx+math.sin(yaw)*dy;y=-math.sin(yaw)*dx+math.cos(yaw)*dy
            local=(math.cos(pitch)*x-math.sin(pitch)*dz,y,math.sin(pitch)*x+math.cos(pitch)*dz)
            if all(abs(local[k])<box.size_xyz_m[k]/2-tolerance for k in range(3)):owners.add(module.module_id)
        for cone in getattr(course,'cones',()):
            z=point[2]
            radius=cone.radius_m*max(0.,1-z/cone.height_m)
            if 0<z<cone.height_m and math.hypot(point[0]-cone.center_xy_m[0],point[1]-cone.center_xy_m[1])<radius-tolerance:
                owners.add(module.module_id)
    return tuple(sorted(owners))


def connection_keys(env,index):
    return tuple(sorted((c.joint_path,c.first_face.module_id,c.first_face.face_name,c.second_face.module_id,c.second_face.face_name)
                        for c in env.docking_managers[index].connections))


def start_step_evidence(env):
    env._c1_step_evidence=[]
    for i in range(env.num_envs):
        points={(m['module_id'],p['connector_id']):tuple(p['position_world']) for m in env._pre_raw[i]['state']['modules'] for p in m['connectors']}
        env._c1_step_evidence.append(dict(ticks=[],before_connections=connection_keys(env,i),events=[],
            last_points=points,last_time_s=env._sim_step_counter*env.physics_dt))


def record_physics_evidence(env,index,actuation):
    """Record every native substep to detect brief dynamics and joint churn."""
    evidence=env._c1_step_evidence[index]
    linear=angular=0.;finite=True;interval_valid=True;measured_modules=[]
    origin=env.scene.env_origins[index].cpu().tolist()
    for owner,reader in env.state_readers_by_env[index].items():
        view=reader._articulation.view
        velocity=view.get_root_velocities().numpy()[0]
        root=view.get_root_transforms().numpy()[0]
        finite=finite and all(math.isfinite(float(x)) for x in (*velocity,*root))
        q=view.get_dof_positions().numpy()[0];qd=view.get_dof_velocities().numpy()[0]
        finite=finite and all(math.isfinite(float(x)) for x in (*q,*qd))
        state=reader.read()
        joints=tuple((f'{name}.{field}',sign*float(getattr(state,source))) for name,sign,pos,vel in (
            ('left_wheel',1.,'left_wheel_rad','left_wheel_rad_s'),('right_wheel',1.,'right_wheel_rad','right_wheel_rad_s'),
            ('pan',1.,'pan_joint_rad','pan_joint_rad_s'),('tilt',-1.,'tilt_joint_rad','tilt_joint_rad_s'))
            for field,source in (('position_rad',pos),('velocity_rad_s',vel)))
        module=C1Module(owner,tuple(float(root[k])-origin[k] for k in range(3)),joints=joints)
        measured_modules.append(module)
        complete,sane,controlled,_=module_joint_status(module)
        interval_valid=interval_valid and complete and sane and controlled
        linear=max(linear,norm(velocity[:3]));angular=max(angular,norm(velocity[3:]))
    # DirectMARLEnv increments the counter BEFORE _apply_action and sim.step.
    # These are the poses after the PREVIOUS physical tick; final post-step
    # points below cover the last (eighth) interval, without a timestamp hole.
    now=(env._sim_step_counter-1)*env.physics_dt
    points={}
    try:
        for owner in env.cfg.possible_agents:
            for pose in env.docking_managers[index].face_poses_for(owner):
                points[(owner,pose.face.face_name)]=tuple(pose.position_world_m[k]-origin[k] for k in range(3))
    except ValueError:
        finite=False
    dt=now-evidence['last_time_s']
    if dt>0:
        for key,point in points.items():
            if key in evidence['last_points']:
                linear=max(linear,norm(tuple(point[k]-evidence['last_points'][key][k] for k in range(3)))/dt)
    evidence['last_points']=points;evidence['last_time_s']=now
    collisions=solid_penetration_owners(env.courses[index],measured_modules)
    for connection in env.docking_managers[index].connections:
        valid,distance,angle=check_fixed_joint(env.sim.stage,connection,env.native_transforms)
        interval_valid=interval_valid and valid
        finite=finite and math.isfinite(distance) and math.isfinite(angle)
    evidence['ticks'].append(dict(linear=linear if finite else math.nan,angular=angular if finite else math.nan,
        connections=connection_keys(env,index),valid=interval_valid,collision_owner_ids=collisions))
    for status in actuation.primitive_statuses:
        primitive=getattr(status.primitive,'value',status.primitive)
        state=getattr(status.state,'value',status.state)
        owner=status.module_ids[0] if status.module_ids else ''
        evidence['events'].append(C1PrimitiveEvent(status.goal_id,str(primitive),str(state),str(status.code),owner))


def initial_frame_from_native(env,index,raw):
    """Instantaneous measured reset baseline; no fictitious physical ticks."""
    return frame_from_native(env,index,raw,initial=True)


def frame_from_native(env,index,raw,*,initial=False):
    modules=[];complete=True
    for item in raw['state']['modules']:
        complete=complete and all(k in item for k in ('position','orientation','linear_velocity','angular_velocity','actuators','connectors'))
        actuators=item.get('actuators',{})
        complete=complete and all(k in actuators for k in ('left_wheel','right_wheel','pan','tilt'))
        complete=complete and all(all(field in actuators.get(name,{}) for field in ('position_rad','velocity_rad_s','max_velocity_rad_s'))
            for name in ('left_wheel','right_wheel','pan','tilt'))
        complete=complete and all(k in actuators.get('tilt',{}) for k in ('lower_limit_rad','upper_limit_rad'))
        ports=item.get('connectors',())
        complete=complete and len(ports)==4 and {p['connector_id'] for p in ports}=={'LEFT','RIGHT','TOP','BOTTOM'}
        joints=tuple((f'{name}.{field}',float(value[field])) for name,value in sorted(actuators.items()) for field in ('position_rad','velocity_rad_s') if field in value)
        modules.append(C1Module(item['module_id'],tuple(item['position']),tuple(item.get('orientation',(0.,0.,0.,0.))),
            tuple(item.get('linear_velocity',(math.nan,)*3)),tuple(item.get('angular_velocity',(math.nan,)*3)),joints,
            tuple((c['connector_id'],tuple(c['position_world'])) for c in ports),
            tuple((name,float(value['max_velocity_rad_s'])) for name,value in actuators.items() if 'max_velocity_rad_s' in value),
            tuple((name,float(value['lower_limit_rad']),float(value['upper_limit_rad'])) for name,value in actuators.items() if 'lower_limit_rad' in value and 'upper_limit_rad' in value)))
    connections=[]
    for connection in env.docking_managers[index].connections:
        valid,distance,angle=check_fixed_joint(env.sim.stage,connection,env.native_transforms)
        a,b=connection.first_face,connection.second_face
        connections.append(C1Connection(connection.joint_path,a.module_id,a.face_name,b.module_id,b.face_name,valid,distance,angle))
    evidence=(dict(ticks=[],before_connections=connection_keys(env,index),events=[],last_points={},last_time_s=float(raw['stamp']))
        if initial else env._c1_step_evidence[index])
    ticks=evidence['ticks']
    if not initial:complete=complete and len(ticks)==env.cfg.decimation and all('valid' in t for t in ticks)
    sequence=[evidence['before_connections'],*(tick['connections'] for tick in ticks),connection_keys(env,index)]
    changed=any(a!=b for a,b in zip(sequence,sequence[1:]))
    changes=[]
    for a,b in zip(sequence,sequence[1:]):
        a_ids={x[0] for x in a};b_ids={x[0] for x in b}
        changes.extend(('added',jid) for jid in sorted(b_ids-a_ids))
        changes.extend(('removed',jid) for jid in sorted(a_ids-b_ids))
    peak=lambda key:math.nan if any(not math.isfinite(t[key]) for t in ticks) else max((t[key] for t in ticks),default=0.)
    depression=raw['state']['course'].get('button',{}).get('depression_m')
    if env.domain_by_env[index]=='manipulation' and depression is None:complete=False
    final_point_speed=0.
    final_dofs_finite=True
    for reader in env.state_readers_by_env[index].values():
        view=reader._articulation.view
        final_dofs_finite=final_dofs_finite and all(math.isfinite(float(x)) for x in (
            *view.get_dof_positions().numpy()[0],*view.get_dof_velocities().numpy()[0]))
    dt=float(raw['stamp'])-evidence['last_time_s']
    if dt>0:
        for module in modules:
            for port,point in module.connectors:
                previous=evidence['last_points'].get((module.module_id,port))
                if previous is not None:final_point_speed=max(final_point_speed,norm(tuple(point[k]-previous[k] for k in range(3)))/dt)
    collision_owners=set(solid_penetration_owners(env.courses[index],modules))
    collision_owners.update(owner for tick in ticks for owner in tick.get('collision_owner_ids',()))
    return C1Frame(float(raw['stamp']),tuple(modules),tuple(connections),max(peak('linear'),final_point_speed) if final_dofs_finite else math.nan,peak('angular'),changed,complete,
        tuple(sorted(collision_owners)),tuple(evidence['events']),depression,tuple(changes),all(t.get('valid',False) for t in ticks))
