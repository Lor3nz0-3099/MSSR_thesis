"""Selected-environment reset barriers; never advance or reset the simulation."""


def flush_pending_physics():
    """Retire authored joint/material/geometry edits before tensor teleports."""
    import omni.physx
    omni.physx.get_physx_simulation_interface().flush_changes()


def isolate_course_colliders(stage, owner_root, environment_roots):
    """Restore row isolation for colliders created after physics has started.

    Root collision-group expansion works at initial scene creation but is not
    sufficient for regenerated course shapes in the installed PhysX version.
    Explicit native filtered pairs retain every own-task collider and exclude
    only foreign environment shapes. No body state or global group is reset.
    """
    from pxr import Sdf,Usd,UsdPhysics
    roots=tuple(Sdf.Path(str(root)) for root in environment_roots)
    owner=Sdf.Path(str(owner_root))
    if owner not in roots:raise ValueError('collision isolation owner is not a native environment')
    foreign=tuple(root for root in roots if root!=owner)
    targets=set()
    for root in foreign:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if prim.HasAPI(UsdPhysics.CollisionAPI):targets.add(str(prim.GetPath()))
    course=stage.GetPrimAtPath(owner.AppendChild('C1Course'))
    count=0
    for prim in Usd.PrimRange(course):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
        relation=UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel()
        keep={str(path) for path in relation.GetTargets()
              if not any(path.HasPrefix(root) for root in foreign)}
        relation.SetTargets(sorted(keep|targets))
        count+=1
    return dict(colliders=count,foreign_colliders=len(targets))


def clear_authored_external_wrenches(stage, module_roots):
    """Clear persistent USD ForceAPI state only under selected module roots.

    Native policy primitives use joint drives, not a global external-wrench
    accumulator. Joint actuation efforts are cleared by reset_pose. A global
    SimulationView.clear_forces would also erase sibling loads and is forbidden.
    """
    from pxr import Usd
    for root in module_roots:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            for name in ('physxForce:force','physxForce:torque'):
                attr=prim.GetAttribute(name)
                if attr and attr.Get() is not None:
                    attr.Set(attr.Get()*0.)
            enabled=prim.GetAttribute('physxForce:forceEnabled')
            if enabled:enabled.Set(False)
