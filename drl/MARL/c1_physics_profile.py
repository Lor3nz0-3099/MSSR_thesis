"""Explicit known-good SMORES body profile, without a teleop pacing loop.

Matches configure_dynamic_stage's physical overrides. The legacy launcher
applies these after cloning (only to its first module); C1 applies them to
every robot. Wheel material values stay those authored in the shared USD:
the legacy wheel_friction_scale argument has no runtime consumer.
"""


def configure_c1_module_stage(stage, module_roots):
    from pxr import Sdf, Usd, UsdPhysics
    from smores_ep.isaac.multi_module_stage import use_collision_proxy_visuals

    for root in module_roots.values():
        articulation = stage.GetPrimAtPath(root)
        if not articulation:
            raise ValueError(f'Missing C1 module: {root}')
        # The shared physics USD already applies PhysxArticulationAPI.
        for name, kind, value in (
            ('enabledSelfCollisions', Sdf.ValueTypeNames.Bool, False),
            ('solverPositionIterationCount', Sdf.ValueTypeNames.Int, 32),
            ('solverVelocityIterationCount', Sdf.ValueTypeNames.Int, 8),
        ):
            articulation.CreateAttribute('physxArticulation:' + name, kind).Set(value)
        for prim in Usd.PrimRange(articulation):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            for name, kind, value in (
                ('solverPositionIterationCount', Sdf.ValueTypeNames.Int, 32),
                ('solverVelocityIterationCount', Sdf.ValueTypeNames.Int, 8),
                ('maxDepenetrationVelocity', Sdf.ValueTypeNames.Float, 1.),
                ('linearDamping', Sdf.ValueTypeNames.Float, .02),
                ('angularDamping', Sdf.ValueTypeNames.Float, .02),
            ):
                prim.CreateAttribute('physxRigidBody:' + name, kind).Set(value)
    use_collision_proxy_visuals(stage, module_roots)
