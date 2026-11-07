"""USD-only parity probes: no Kit launch, physics step, or policy changes."""
from pathlib import Path
import pytest
from pxr import Usd,UsdPhysics

ROOT=Path(__file__).resolve().parents[3]
ASSET=ROOT/'assets/smores-ep/usd_physics/smores_ep_physics_v1.usd'


def test_every_c1_module_gets_known_good_body_settings_without_material_changes():
    from drl.MARL.c1_physics_profile import configure_c1_module_stage
    stage=Usd.Stage.CreateInMemory()
    roots={f'env{i}':f'/World/envs/env_{i}/SMORES_0' for i in range(2)}
    for root in roots.values():stage.DefinePrim(root,'Xform').GetReferences().AddReference(str(ASSET))
    friction_before=[(stage.GetPrimAtPath(root+'/materials/wheel').GetAttribute(k).Get())
        for root in roots.values() for k in ('physics:staticFriction','physics:dynamicFriction')]
    configure_c1_module_stage(stage,roots)
    keys=('physxRigidBody:solverPositionIterationCount','physxRigidBody:solverVelocityIterationCount',
        'physxRigidBody:maxDepenetrationVelocity','physxRigidBody:linearDamping','physxRigidBody:angularDamping')
    for root in roots.values():
        bodies=[p for p in Usd.PrimRange(stage.GetPrimAtPath(root)) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        assert len(bodies)==5
        for body in bodies:
            # Values traced to configure_dynamic_stage, not inferred from flags.
            for key,value in zip(keys,(32,8,1.,.02,.02),strict=True):
                assert body.GetAttribute(key).Get()==pytest.approx(value)
        wheel=stage.GetPrimAtPath(root+'/materials/wheel')
        assert wheel.GetAttribute('physxMaterial:frictionCombineMode').Get()=='multiply'
        assert stage.GetPrimAtPath(root+'/body_link/visual_offset').GetAttribute('visibility').Get()=='invisible'
    assert friction_before==[stage.GetPrimAtPath(root+'/materials/wheel').GetAttribute(k).Get()
        for root in roots.values() for k in ('physics:staticFriction','physics:dynamicFriction')]


def test_c1_scene_ccd_and_proxy_profile_are_wired_before_physics_initialization():
    import ast
    source=(ROOT/'drl/MARL/isaaclab_c1_env.py').read_text();tree=ast.parse(source)
    configs=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='PhysxCfg']
    assert len(configs)==1
    assert any(k.arg=='enable_ccd' and isinstance(k.value,ast.Constant) and k.value.value is True for k in configs[0].keywords)
    setup=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_setup_scene')
    calls=[n.func.id for n in ast.walk(setup) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    assert 'configure_c1_module_stage' in calls


def test_runtime_artifact_distinguishes_requested_ccd_from_gpu_effective_scene():
    from types import SimpleNamespace
    from pxr import Sdf
    from smores_ep.config.physics import SmoresActuatorConfig
    from drl.MARL.c1_physics_profile import configure_c1_module_stage
    from drl.MARL.c1_structural_physics_diagnostic import runtime_settings
    stage=Usd.Stage.CreateInMemory();root='/World/envs/env_0/SMORES_0'
    stage.DefinePrim(root,'Xform').GetReferences().AddReference(str(ASSET))
    configure_c1_module_stage(stage,{'m':root})
    stage.DefinePrim('/physicsScene','PhysicsScene').CreateAttribute('physxScene:enableCCD',Sdf.ValueTypeNames.Bool).Set(False)
    simcfg=SimpleNamespace(physics_prim_path='/physicsScene',render_interval=8,physics=SimpleNamespace(enable_ccd=True,solver_type=1),device='cuda:0',use_fabric=True,gravity=(0.,0.,-9.81))
    env=SimpleNamespace(cfg=SimpleNamespace(sim=simcfg,decimation=8),sim=SimpleNamespace(stage=stage,is_rendering=False,has_gui=False),
        module_roots_by_env=({'m':root},),scene=SimpleNamespace(sensors={}),physics_dt=1/240,step_dt=1/30,
        state_readers_by_env=({'m':SimpleNamespace(_drive=SmoresActuatorConfig.payload_overdrive())},),seed_by_env=(7033,),
        native_dispatcher=SimpleNamespace(primitive_timeout_s=30.))
    result=runtime_settings(env)
    assert result['requested_ccd'] is True
    assert result['scene_attributes']['physxScene:enableCCD'] is False
    assert result['actuator_profile']['wheel_max_effort_nm']==pytest.approx(4.8)
    assert result['actuator_profile']['tilt_max_effort_nm']==pytest.approx(18.4)
