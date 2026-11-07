"""Synthetic CPU schema/tensor/USD fixtures; never physical capture evidence."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from drl.MARL import c1_expert_snapshots as snapshots


def payload():
    owners = [f'm{i}' for i in range(8)]
    return dict(schema='mssr.c1_expert_snapshot.v1', stage='S0',
        identity=dict(scenario_id='navigation:train:7000', domain='navigation', morphology='rc_car8',
            seed=7000, split='train', geometry_sha256='a'*64),
        physics=dict(version='c1-native-v1', source_sha256='b'*64), owners=owners,
        modules={owner:dict(root_relative_path=f'SMORES_{i}',position_xyz_m=[i,.2,.03],
            quaternion_xyzw=[0,0,0,1],root_velocity=[0]*6,dof_names=['wheel','pan'],
            q=[.1,.2],qd=[.3,.4],control={name:[0,0] for name in (
                'dof_stiffnesses','dof_dampings','dof_position_targets',
                'dof_velocity_targets','dof_actuation_forces')}) for i,owner in enumerate(owners)},
        connections=[],actual_target_edges=0,scenario_state={},
        physical_certificate=dict(expert_generated=True,target_compatible=True,target_edges=0,
            source='synthetic_cpu_fixture'))


def sealed(value):
    value=deepcopy(value)
    value.pop('content_sha256',None)
    value['content_sha256']=hashlib.sha256(json.dumps(value,sort_keys=True,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return value


def test_snapshot_exclusive_save_hash_tamper_and_identity(tmp_path):
    path=tmp_path/'state.json'
    snapshots.save_snapshot(path,payload())
    state=snapshots.load_snapshot(path,expected_identity={'split':'train'})
    assert state['modules']['m3']['position_xyz_m']==[3,.2,.03]
    with pytest.raises(FileExistsError):snapshots.save_snapshot(path,payload())
    with pytest.raises(ValueError,match='identity'):snapshots.load_snapshot(path,expected_identity={'split':'test'})
    state['modules']['m0']['q'][0]=8
    with pytest.raises(ValueError,match='hash'):snapshots.validate_snapshot(state)


@pytest.mark.parametrize('mutation', [
    lambda p:p['modules']['m0']['q'].__setitem__(0,float('nan')),
    lambda p:p['modules']['m0'].__setitem__('root_relative_path','../env_7/SMORES_0'),
    lambda p:p['modules']['m0'].__setitem__('quaternion_xyzw',[0,0,0,2]),
    lambda p:p['modules']['m0'].__setitem__('dof_names',['wheel','wheel']),
    lambda p:p.__setitem__('actual_target_edges',1),
    lambda p:p['physical_certificate'].__setitem__('target_compatible',False),
])
def test_snapshot_rejects_invalid_even_with_recomputed_hash(mutation):
    value=payload();mutation(value)
    if np.isnan(value['modules']['m0']['q'][0]):
        with pytest.raises(ValueError):snapshots.save_snapshot('/never-created',value)
    else:
        with pytest.raises(ValueError):snapshots.validate_snapshot(sealed(value))


def test_bank_checks_paths_bytes_hash_and_cross_split_geometry(tmp_path):
    state=tmp_path/'state.json';snapshots.save_snapshot(state,payload())
    record=dict(path='state.json',sha256=hashlib.sha256(state.read_bytes()).hexdigest(),
        identity=payload()['identity'],stage='S0')
    manifest=tmp_path/'bank.json'
    manifest.write_text(json.dumps(dict(schema='mssr.c1_snapshot_bank.v1',records=[record])))
    bank=snapshots.SnapshotBank(manifest,'train',payload()['physics'])
    assert len(bank.records)==1
    assert bank.load(bank.records[0])['stage']=='S0'
    for bad in ('../outside.json','/tmp/outside.json'):
        record['path']=bad
        manifest.write_text(json.dumps(dict(schema='mssr.c1_snapshot_bank.v1',records=[record])))
        with pytest.raises(ValueError):snapshots.SnapshotBank(manifest,'train')
    record['path']='state.json';record['sha256']='0'*64
    manifest.write_text(json.dumps(dict(schema='mssr.c1_snapshot_bank.v1',records=[record])))
    with pytest.raises(ValueError,match='hash'):snapshots.SnapshotBank(manifest,'train')
    other=payload();other['identity'].update(split='test',seed=9000,scenario_id='navigation:test:9000')
    snapshots.save_snapshot(tmp_path/'test.json',other)
    record['sha256']=hashlib.sha256(state.read_bytes()).hexdigest()
    record2=dict(path='test.json',sha256=hashlib.sha256((tmp_path/'test.json').read_bytes()).hexdigest(),
        identity=other['identity'],stage='S0')
    manifest.write_text(json.dumps(dict(schema='mssr.c1_snapshot_bank.v1',records=[record,record2])))
    with pytest.raises(ValueError,match='split'):snapshots.SnapshotBank(manifest,'train')


def tensor_access():
    from drl.MARL.tests.test_isaaclab_articulation import TensorView,make_access
    view=TensorView()
    for name,values in dict(root_transforms=[[1,2,3,0,0,.6,.8]],root_velocities=[[1,2,3,4,5,6]],
        dof_positions=[[.1,.2,.3]],dof_velocities=[[.4,.5,.6]]).items():view.put(name,values)
    return make_access(view),view


def test_native_full_state_preserves_motion_controls_and_rejects_dof_mismatch():
    access,view=tensor_access()
    state=access.capture_state()
    access.reset_pose([9,8,7],0)
    access.restore_state(state)
    np.testing.assert_allclose(view.get_root_transforms().numpy(),[[1,2,3,0,0,.6,.8]])
    np.testing.assert_allclose(view.get_root_velocities().numpy(),[[1,2,3,4,5,6]])
    np.testing.assert_allclose(view.get_dof_positions().numpy(),[[.1,.2,.3]])
    np.testing.assert_allclose(view.get_dof_velocities().numpy(),[[.4,.5,.6]])
    before=len(view.writes)
    state['dof_names']=['pan','right','left']
    with pytest.raises(ValueError):access.restore_state(state)
    assert len(view.writes)==before


def test_handover_preserves_actual_motion_but_retires_expert_drive_plan():
    access,view=tensor_access()
    view.put('dof_position_targets',[[9,8,7]])
    view.put('dof_velocity_targets',[[6,5,4]])
    view.put('dof_stiffnesses',[[300,300,300]])
    view.put('dof_actuation_forces',[[2,3,4]])
    state=access.capture_state()
    access.restore_state(state,restore_control=False)
    np.testing.assert_allclose(view.get_dof_positions().numpy(),[[.1,.2,.3]])
    np.testing.assert_allclose(view.get_dof_velocities().numpy(),[[.4,.5,.6]])
    np.testing.assert_allclose(view.get_dof_position_targets().numpy(),[[.1,.2,.3]])
    np.testing.assert_array_equal(view.get_dof_velocity_targets().numpy(),[[0,0,0]])
    np.testing.assert_array_equal(view.get_dof_stiffnesses().numpy(),[[0,0,0]])
    np.testing.assert_array_equal(view.get_dof_actuation_forces().numpy(),[[0,0,0]])
    np.testing.assert_allclose(view.get_dof_dampings().numpy(),[[.02,.02,.02]])


def test_physics_fingerprint_resolves_real_sources_without_simulator():
    physics=snapshots.current_snapshot_physics()
    assert len(physics['source_sha256'])==64
    assert physics==snapshots.current_snapshot_physics()


def test_stage_bucket_and_certificate_count_cannot_be_forged_by_rehashing():
    state=payload()
    state['stage']='S3'
    with pytest.raises(ValueError,match='stage'):
        snapshots.validate_snapshot(sealed(state))
    state=payload();state['physical_certificate']['target_edges']=False
    with pytest.raises(ValueError):snapshots.validate_snapshot(sealed(state))


def test_bank_rejects_symlink_escape_even_when_target_has_valid_hash(tmp_path):
    outside=tmp_path/'outside.json';snapshots.save_snapshot(outside,payload())
    bankdir=tmp_path/'bank';bankdir.mkdir()
    (bankdir/'escape.json').symlink_to(outside)
    manifest=bankdir/'manifest.json'
    record=dict(path='escape.json',sha256=hashlib.sha256(outside.read_bytes()).hexdigest(),
        identity=payload()['identity'],stage='S0')
    manifest.write_text(json.dumps(dict(schema='mssr.c1_snapshot_bank.v1',records=[record])))
    with pytest.raises(ValueError,match='containment'):snapshots.SnapshotBank(manifest,'train')


def test_manipulation_snapshot_requires_full_plunger_state():
    state=payload()
    state['identity'].update(domain='manipulation',morphology='mobile_manipulator8')
    with pytest.raises(ValueError,match='plunger'):
        snapshots.validate_snapshot(sealed(state))
    state['scenario_state']={'plunger':dict(root_relative_path='C1Course/Plunger',
        position_xyz_m=[.1,.2,.3],quaternion_xyzw=[0,0,.6,.8],root_velocity=[1,2,3,4,5,6])}
    assert snapshots.validate_snapshot(sealed(state))['scenario_state']['plunger']['root_velocity']==[1,2,3,4,5,6]


@pytest.mark.parametrize('manipulation',[False,True])
def test_capture_restore_remaps_origin_and_rejects_active_goals_before_write(manipulation):
    import torch
    access,view=tensor_access()
    native_root='/World/envs/env_0/SMORES_0'
    view.prim_paths=[native_root]
    owners=[f'm{i}' for i in range(8)]
    accesses={}
    roots={}
    for i,owner in enumerate(owners):
        item,tensor=tensor_access()
        roots[owner]=f'/World/envs/env_0/SMORES_{i}'
        tensor.prim_paths=[roots[owner]]
        tensor.put('root_transforms',[[11+i,22,3,0,0,.6,.8]])
        accesses[owner]=item
    manager=SimpleNamespace(connections=(),export_connections=lambda:[],
        validate_connection_states=lambda states:None,restore_connections=lambda states:None)
    executor=SimpleNamespace(active_goals=())
    env=SimpleNamespace(cfg=SimpleNamespace(possible_agents=owners),
        scene=SimpleNamespace(env_origins=torch.tensor([[10.,20.,0.]])),
        state_readers_by_env=[{owner:SimpleNamespace(_articulation=item) for owner,item in accesses.items()}],
        module_roots_by_env=[roots],docking_managers=[manager],
        physical_expert_backends=[{'primitive_executor':executor}],
        native_transforms=SimpleNamespace(invalidate=lambda:None))
    identity=payload()['identity']
    if manipulation:
        from drl.MARL.tests.test_isaaclab_articulation import TensorView
        plunger=TensorView(root='/World/envs/env_0/C1Course/Plunger')
        plunger.put('transforms',[[10.1,20.2,.3,0,0,.6,.8]])
        plunger.put('velocities',[[1,2,3,4,5,6]])
        env.plunger_paths={0:'/World/envs/env_0/C1Course/Plunger'}
        env.native_transforms.rigid_views={env.plunger_paths[0]:plunger}
        identity.update(domain='manipulation',morphology='mobile_manipulator8')
    state=snapshots.capture_native_snapshot(env,0,'S0',identity=identity,
        physics=payload()['physics'],physical_certificate=payload()['physical_certificate'])
    assert state['modules']['m0']['position_xyz_m']==[1,2,3]
    env.scene.env_origins[0]=torch.tensor([100.,200.,0.])
    snapshots.restore_native_snapshot(env,0,state,expected_identity=identity,
        expected_physics=payload()['physics'])
    np.testing.assert_allclose(accesses['m0'].view.get_root_transforms().numpy()[0,:3],[101,202,3])
    if manipulation:
        np.testing.assert_allclose(plunger.get_transforms().numpy(),[[100.1,200.2,.3,0,0,.6,.8]],atol=1e-5)
        np.testing.assert_array_equal(plunger.get_velocities().numpy(),[[1,2,3,4,5,6]])
    before=len(accesses['m0'].view.writes)
    executor.active_goals=('stale-plan',)
    with pytest.raises(ValueError,match='canceled'):
        snapshots.restore_native_snapshot(env,0,state,expected_identity=identity,
            expected_physics=payload()['physics'])
    assert len(accesses['m0'].view.writes)==before


def docking_manager():
    from pxr import Sdf,Usd,UsdGeom,UsdPhysics
    from smores_ep.isaac.docking import IsaacDockingManager
    stage=Usd.Stage.CreateInMemory()
    roots={f'm{i}':f'/World/envs/env_0/SMORES_{i}' for i in range(2)}
    for root in roots.values():
        UsdGeom.Xform.Define(stage,root)
        body=UsdGeom.Xform.Define(stage,root+'/body').GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        for port in ('LEFT','RIGHT','TOP','BOTTOM'):
            face=UsdGeom.Xform.Define(stage,root+'/body/'+port).GetPrim()
            face.CreateAttribute('smores:face',Sdf.ValueTypeNames.String).Set(port)
    return stage,IsaacDockingManager(stage,roots,joints_root='/World/envs/env_0/DockingJoints')


def test_docking_exact_frames_registry_restore_and_reject_unknown_paths():
    from pxr import Gf,UsdPhysics
    stage,manager=docking_manager()
    edges=[dict(first_owner='m0',first_port='RIGHT',second_owner='m1',second_port='LEFT',
        first_body_relative_path='body',second_body_relative_path='body',
        local_pos0=[.1,.2,.3],local_rot0_xyzw=[0,0,.6,.8],
        local_pos1=[-.2,.4,.5],local_rot1_xyzw=[0,0,0,1],collision_enabled=False,
        exclude_from_articulation=True)]
    manager.restore_connections(edges)
    assert manager.occupied_faces==frozenset({('m0','RIGHT'),('m1','LEFT')})
    exported=manager.export_connections()
    np.testing.assert_allclose(exported[0]['local_pos0'],[.1,.2,.3])
    np.testing.assert_allclose(exported[0]['local_rot0_xyzw'],[0,0,.6,.8])
    joint=UsdPhysics.FixedJoint(stage.GetPrimAtPath(manager.connections[0].joint_path))
    assert list(joint.GetBody1Rel().GetTargets())==['/World/envs/env_0/SMORES_1/body']
    _,fresh=docking_manager()
    for key,bad in [('first_port','WRONG'),('first_body_relative_path','../../evil')]:
        invalid=deepcopy(edges);invalid[0][key]=bad
        with pytest.raises(ValueError):fresh.restore_connections(invalid)
        assert not fresh.connections
