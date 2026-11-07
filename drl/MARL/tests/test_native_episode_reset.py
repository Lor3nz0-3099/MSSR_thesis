"""Selected-row authored state and native collision isolation, without stepping."""
import pytest


def stage_fixture():
    from pxr import Usd,UsdGeom,UsdPhysics,Sdf
    stage=Usd.Stage.CreateInMemory()
    for i in range(2):
        for path in (f'/World/envs/env_{i}/C1Course/Support',f'/World/envs/env_{i}/SMORES_0/body/Collider'):
            prim=UsdGeom.Cube.Define(stage,path).GetPrim();UsdPhysics.CollisionAPI.Apply(prim)
    return stage


def test_course_pairs_exclude_only_foreign_colliders_and_preserve_existing_filters():
    from pxr import UsdPhysics
    from drl.MARL.native_episode_reset import isolate_course_colliders
    stage=stage_fixture();own='/World/envs/env_0';foreign='/World/envs/env_1'
    prim=stage.GetPrimAtPath(own+'/C1Course/Support')
    old=UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
    old.SetTargets([own+'/KeepOwnFilter','/World/KeepGlobalFilter'])
    result=isolate_course_colliders(stage,own,(own,foreign))
    targets={str(p) for p in old.GetTargets()}
    assert targets=={own+'/KeepOwnFilter','/World/KeepGlobalFilter',foreign+'/C1Course/Support',foreign+'/SMORES_0/body/Collider'}
    assert own+'/SMORES_0/body/Collider' not in targets
    assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
    assert not stage.GetPrimAtPath(foreign+'/C1Course/Support').HasAPI(UsdPhysics.FilteredPairsAPI)
    assert result['colliders']==1 and result['foreign_colliders']==2


def test_recreated_foreign_shapes_replace_stale_paths_without_disabling_own_course():
    from pxr import UsdGeom,UsdPhysics
    from drl.MARL.native_episode_reset import isolate_course_colliders
    stage=stage_fixture();roots=('/World/envs/env_0','/World/envs/env_1')
    isolate_course_colliders(stage,roots[0],roots)
    stage.RemovePrim(roots[1]+'/C1Course/Support')
    prim=UsdGeom.Cube.Define(stage,roots[1]+'/C1Course/NewSeedSupport').GetPrim();UsdPhysics.CollisionAPI.Apply(prim)
    isolate_course_colliders(stage,roots[0],roots)
    source=stage.GetPrimAtPath(roots[0]+'/C1Course/Support')
    paths={str(p) for p in UsdPhysics.FilteredPairsAPI(source).GetFilteredPairsRel().GetTargets()}
    assert roots[1]+'/C1Course/NewSeedSupport' in paths
    assert roots[1]+'/C1Course/Support' not in paths
    assert roots[0]+'/SMORES_0/body/Collider' not in paths
    assert UsdPhysics.CollisionAPI(source).GetCollisionEnabledAttr().Get()


def test_isolation_rejects_unknown_owner_before_changes():
    from drl.MARL.native_episode_reset import isolate_course_colliders
    stage=stage_fixture();before=stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError,match='owner'):
        isolate_course_colliders(stage,'/World/Unknown',('/World/envs/env_0','/World/envs/env_1'))
    assert stage.GetRootLayer().ExportToString()==before


def test_authored_wrench_clear_is_selected_and_does_not_clear_sibling():
    from pxr import UsdGeom,Sdf,Gf
    from drl.MARL.native_episode_reset import clear_authored_external_wrenches
    stage=stage_fixture()
    for i in range(2):
        p=stage.GetPrimAtPath(f'/World/envs/env_{i}/SMORES_0/body/Collider')
        p.CreateAttribute('physxForce:force',Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(1.,2.,3.))
        p.CreateAttribute('physxForce:torque',Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(4.,5.,6.))
        p.CreateAttribute('physxForce:forceEnabled',Sdf.ValueTypeNames.Bool).Set(True)
    clear_authored_external_wrenches(stage,('/World/envs/env_0/SMORES_0',))
    selected=stage.GetPrimAtPath('/World/envs/env_0/SMORES_0/body/Collider')
    sibling=stage.GetPrimAtPath('/World/envs/env_1/SMORES_0/body/Collider')
    assert tuple(selected.GetAttribute('physxForce:force').Get())==(0.,0.,0.)
    assert tuple(selected.GetAttribute('physxForce:torque').Get())==(0.,0.,0.)
    assert not selected.GetAttribute('physxForce:forceEnabled').Get()
    assert tuple(sibling.GetAttribute('physxForce:force').Get())==(1.,2.,3.)
    assert sibling.GetAttribute('physxForce:forceEnabled').Get()
