"""Viewer-only lighting and preview materials. No geometry/physics edits."""
PALETTE=((.20,.65,.95),(.95,.35,.20),(.30,.85,.40),(.95,.75,.15),
         (.70,.40,.95),(.15,.85,.80),(.95,.45,.70),(.70,.75,.85))


def configure_viewer_visuals(stage,module_roots):
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdLux,UsdPhysics,UsdShade
    base='/World/C1Viewer'
    UsdGeom.Scope.Define(stage,base+'/Materials')
    # Same lighting intensities as the working dynamic teleop, without calling
    # configure_dynamic_stage (which also changes physical body properties).
    dome=UsdLux.DomeLight.Define(stage,base+'/Sky');dome.CreateIntensityAttr(500.)
    sun=UsdLux.DistantLight.Define(stage,base+'/Sun');sun.CreateIntensityAttr(3000.);sun.CreateAngleAttr(.5)
    xform=UsdGeom.Xformable(sun.GetPrim());ops=xform.GetOrderedXformOps()
    (ops[0] if ops else xform.AddRotateXYZOp()).Set(Gf.Vec3f(315.,0.,35.))
    modules=[]
    for i,(owner,root) in enumerate(module_roots.items()):
        color=PALETTE[i%len(PALETTE)]
        material=UsdShade.Material.Define(stage,f'{base}/Materials/Module_{i}')
        shader=UsdShade.Shader.Define(stage,f'{material.GetPath()}/Surface')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput('emissiveColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(c*.12 for c in color)))
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.65)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        count=0
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or not prim.IsA(UsdGeom.Gprim):continue
            # Only preview purpose: PhysX resolves purpose='physics' or existing
            # allPurpose bindings, neither of which is replaced or strengthened.
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,materialPurpose=UsdShade.Tokens.preview)
            UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
            count+=1
        modules.append(dict(module=owner,root=root,color=color,preview_proxy_count=count))
    return dict(scope='viewer-only',lighting=dict(dome_intensity=500.,sun_intensity=3000.),
                material_purpose='preview',modules=modules,physics_modified=False)
