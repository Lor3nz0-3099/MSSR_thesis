"""Real USD-only visual override oracle, no simulation application."""
from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,UsdShade,UsdLux


def physical_snapshot(stage):
    result={}
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith('/World/C1Viewer'):continue
        for attr in prim.GetAttributes():
            name=attr.GetName()
            if name.startswith(('physics:','physx','xformOp:')) or name in ('points','extent','size','radius','height'):
                result[str(attr.GetPath())]=repr(attr.Get())
        for rel in prim.GetRelationships():
            if rel.GetName().startswith(('physics:','material:binding')) and ':preview' not in rel.GetName():
                result[str(rel.GetPath())]=tuple(map(str,rel.GetTargets()))
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            for purpose in ('physics',''):
                material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(purpose)
                result[str(prim.GetPath())+':bound:'+purpose]=str(material.GetPath())
    return result


def fixture():
    stage=Usd.Stage.CreateInMemory();roots={}
    material=UsdShade.Material.Define(stage,'/World/PhysicalMaterial')
    UsdPhysics.MaterialAPI.Apply(material.GetPrim()).CreateDynamicFrictionAttr(.75)
    for i in range(8):
        root=f'/World/envs/env_0/SMORES_{i}';roots[f'smores_{i+1:02d}']=root
        body=UsdGeom.Xform.Define(stage,root+'/body_link');UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(.5)
        cube=UsdGeom.Cube.Define(stage,root+'/body_link/collision');cube.CreateSizeAttr(.05)
        cube.AddTranslateOp().Set(Gf.Vec3d(i*.1,0,0));UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material,materialPurpose='physics')
        UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(material)
    return stage,roots


def test_viewer_lights_and_distinct_preview_materials_preserve_every_physical_attribute():
    from drl.MARL.c1_viewer_visuals import configure_viewer_visuals
    stage,roots=fixture();before=physical_snapshot(stage)
    result=configure_viewer_visuals(stage,roots)
    assert physical_snapshot(stage)==before
    assert len(result['modules'])==8 and len({tuple(m['color']) for m in result['modules']})==8
    assert not any(p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.RigidBodyAPI)
                   for p in Usd.PrimRange(stage.GetPrimAtPath('/World/C1Viewer')))
    assert len([p for p in stage.Traverse() if p.IsA(UsdLux.DomeLight)])==1
    assert len([p for p in stage.Traverse() if p.IsA(UsdLux.DistantLight)])==1
    for root in roots.values():
        prim=stage.GetPrimAtPath(root+'/body_link/collision')
        material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('preview')
        assert str(material.GetPath()).startswith('/World/C1Viewer/Materials/')
        assert UsdGeom.Imageable(prim).ComputeVisibility()=='inherited'
    configure_viewer_visuals(stage,roots)
    assert physical_snapshot(stage)==before  # idempotent startup override
