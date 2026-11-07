from types import SimpleNamespace
from pxr import Usd,UsdGeom,UsdPhysics,Gf,Sdf
import pytest

from drl.MARL.c1_native_observation import check_fixed_joint,solid_penetration_owners
from drl.MARL.c1_config import C1Settings
from drl.MARL.c1_runner import make_course
from drl.MARL.c1_structure import C1Module


def joint_fixture():
    stage=Usd.Stage.CreateInMemory()
    for path in ('/A','/B'):
        prim=UsdGeom.Xform.Define(stage,path).GetPrim();UsdPhysics.RigidBodyAPI.Apply(prim)
    joint=UsdPhysics.FixedJoint.Define(stage,'/J')
    joint.CreateBody0Rel().SetTargets([Sdf.Path('/A')]);joint.CreateBody1Rel().SetTargets([Sdf.Path('/B')])
    for i in (0,1):
        getattr(joint,f'CreateLocalPos{i}Attr')(Gf.Vec3f(0.))
        getattr(joint,f'CreateLocalRot{i}Attr')(Gf.Quatf(1.))
    connection=SimpleNamespace(joint_path='/J',first_face=SimpleNamespace(rigid_body_path='/A'),second_face=SimpleNamespace(rigid_body_path='/B'))
    transform=lambda path:UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))
    return stage,joint,connection,transform


def test_live_fixed_joint_requires_correct_enabled_bodies_and_anchor_closure():
    stage,joint,c,transform=joint_fixture()
    assert check_fixed_joint(stage,c,transform)[0]
    joint.CreateJointEnabledAttr(False)
    assert not check_fixed_joint(stage,c,transform)[0]
    joint.CreateJointEnabledAttr(True)
    UsdGeom.Xform(stage.GetPrimAtPath('/B')).AddTranslateOp().Set(Gf.Vec3d(.1,0,0))
    assert not check_fixed_joint(stage,c,transform)[0]


def test_contact_candidate_or_proximity_is_not_a_physical_docking_joint():
    stage,joint,c,transform=joint_fixture();stage.RemovePrim('/J')
    assert not check_fixed_joint(stage,c,transform)[0]


def test_geometric_pathology_is_solid_interior_not_floor_contact():
    course=make_course('navigation',7017,'train')
    f=course.cones[0]
    inside=C1Module('inside',(f.center_xy_m[0],f.center_xy_m[1],.04))
    floor=C1Module('floor',(0.,0.,.0315))
    owners=solid_penetration_owners(course,(inside,floor))
    assert 'inside' in owners and 'floor' not in owners


def tick_fixture():
    import numpy as np
    import torch
    from drl.MARL.c1_native_observation import start_step_evidence
    names=('left_wheel','right_wheel','pan','tilt')
    actuator={name:dict(position_rad=0.,velocity_rad_s=0.,max_velocity_rad_s=3.) for name in names}
    actuator['tilt'].update(lower_limit_rad=-1.,upper_limit_rad=1.)
    points={p:(0.,0.,.04) for p in ('LEFT','RIGHT','TOP','BOTTOM')}
    def raw(stamp):return dict(stamp=stamp,state=dict(course={},modules=[dict(module_id='m0',position=(0.,0.,.04),
        orientation=(0.,0.,0.,1.),linear_velocity=(0.,0.,0.),angular_velocity=(0.,0.,0.),actuators=actuator,
        connectors=[dict(connector_id=p,position_world=xyz) for p,xyz in points.items()])]))
    array=lambda data:SimpleNamespace(numpy=lambda:np.array(data,dtype=float))
    view=SimpleNamespace(get_root_velocities=lambda:array([[0.]*6]),get_root_transforms=lambda:array([[0.,0.,.04,0.,0.,0.,1.]]),
        get_dof_positions=lambda:array([[0.]*4]),get_dof_velocities=lambda:array([[0.]*4]))
    joint=SimpleNamespace(**{key:0. for key in ('left_wheel_rad','right_wheel_rad','pan_joint_rad','tilt_joint_rad',
        'left_wheel_rad_s','right_wheel_rad_s','pan_joint_rad_s','tilt_joint_rad_s')})
    reader=SimpleNamespace(_articulation=SimpleNamespace(view=view),read=lambda:joint)
    manager=SimpleNamespace(connections=(),face_poses_for=lambda owner:tuple(SimpleNamespace(face=SimpleNamespace(face_name=p),position_world_m=xyz) for p,xyz in points.items()))
    env=SimpleNamespace(num_envs=1,_sim_step_counter=0,physics_dt=1/240,cfg=SimpleNamespace(possible_agents=('m0',),decimation=8),
        _pre_raw=[raw(0.)],state_readers_by_env=[{'m0':reader}],docking_managers=[manager],native_transforms=lambda path:Gf.Matrix4d(1.),
        sim=SimpleNamespace(stage=None),scene=SimpleNamespace(env_origins=torch.zeros(1,3)),
        courses=[make_course('navigation',7017,'train')],domain_by_env=['navigation'])
    start_step_evidence(env)
    return env,points,raw


def test_pre_physics_counter_does_not_hide_last_interval_connector_motion():
    from drl.MARL.c1_native_observation import record_physics_evidence,frame_from_native
    env,points,raw=tick_fixture()
    for counter in range(1,9):
        env._sim_step_counter=counter
        record_physics_evidence(env,0,SimpleNamespace(primitive_statuses=()))
    assert env._c1_step_evidence[0]['last_time_s']==pytest.approx(7/240)
    points['LEFT']=(.02,0.,.04)
    frame=frame_from_native(env,0,raw(8/240))
    assert frame.peak_linear_speed==pytest.approx(4.8)


def test_empty_joint_metadata_is_not_complete_physical_evidence():
    from drl.MARL.c1_native_observation import record_physics_evidence,frame_from_native
    env,points,raw=tick_fixture()
    for counter in range(1,9):
        env._sim_step_counter=counter;record_physics_evidence(env,0,SimpleNamespace(primitive_statuses=()))
    value=raw(8/240);value['state']['modules'][0]['actuators']={k:{} for k in value['state']['modules'][0]['actuators']}
    value['state']['modules'][0]['connectors']=[]
    assert not frame_from_native(env,0,value).evidence_complete


def test_between_decision_invalid_joint_and_penetration_reset_validity(monkeypatch):
    from drl.MARL.c1_native_observation import record_physics_evidence,frame_from_native
    env,points,raw=tick_fixture()
    face=SimpleNamespace(module_id='m0',face_name='LEFT',rigid_body_path='/body')
    env.docking_managers[0].connections=(SimpleNamespace(joint_path='/joint',first_face=face,second_face=face),)
    monkeypatch.setattr('drl.MARL.c1_native_observation.check_fixed_joint',lambda *a:(env._sim_step_counter!=4,0.,0.))
    monkeypatch.setattr('drl.MARL.c1_native_observation.solid_penetration_owners',lambda *a:('m0',) if env._sim_step_counter==5 else ())
    for counter in range(1,9):
        env._sim_step_counter=counter;record_physics_evidence(env,0,SimpleNamespace(primitive_statuses=()))
    result=frame_from_native(env,0,raw(8/240))
    assert not result.interval_valid and result.pathological_collision_owner_ids==('m0',)
