"""Selected native C1 USD geometry updates, retaining live dynamic prim paths.

This edits authored geometry only. The caller must reset the existing native
plunger rigid-body view after this update; this helper never resets simulation.
"""
from __future__ import annotations

import math


def update_native_course(stage, env_root, domain, course, origin):
    """Update ``env_root/C1Course`` and return its persistent plunger path.

    Box/cone coordinates remain environment-local; the parent environment owns
    its translation. A prismatic joint with a world body0 needs a world anchor,
    therefore only that anchor receives ``origin`` here. Domain slots are fixed.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics
    from smores_ep.isaac.obstacle_course import (
        _install_course_boxes, _install_button_prismatic_joint)
    from smores_ep.isaac.phase1_training_course import _install_phase1_static_cones

    if domain not in {'navigation', 'stairs', 'gap', 'manipulation'}:
        raise ValueError(f'Unsupported native C1 domain: {domain!r}')
    if len(origin) != 3 or any(not math.isfinite(float(v)) for v in origin):
        raise ValueError('origin must contain three finite world coordinates')
    env_path = Sdf.Path(str(env_root))
    if not env_path.IsAbsolutePath() or not env_path.IsPrimPath() or env_path == Sdf.Path('/'):
        raise ValueError('env_root must be an absolute environment prim path')
    # The shared legacy installer removes this path. Refuse before any mutation
    # rather than let an environment reset change a global physics collider.
    if stage.GetPrimAtPath('/World/Ground'):
        raise ValueError('shared installer would remove global /World/Ground')
    root_path = str(env_path.AppendChild('C1Course'))
    root = stage.GetPrimAtPath(root_path)
    if root:
        old_domain = root.GetAttribute('mssr:c1Domain').Get()
        if old_domain is not None and old_domain != domain:
            raise ValueError('native C1 environment domain slots are fixed')
    boxes = tuple(course.support_boxes)
    names = [box.name for box in boxes]
    if len(names) != len(set(names)) or any(not Sdf.Path.IsValidIdentifier(name) for name in names):
        raise ValueError('course boxes require unique valid prim names')
    plunger_path = (root_path + '/' + course.button_plunger.name
                    if domain == 'manipulation' else None)
    joint_path = root_path + '/ButtonPrismaticJoint'
    if root:
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and str(prim.GetPath()) != plunger_path:
                raise ValueError(f'cannot remove or alter foreign dynamic body {prim.GetPath()}')
        if plunger_path and stage.GetPrimAtPath(plunger_path):
            scale = stage.GetPrimAtPath(plunger_path).GetAttribute('xformOp:scale').Get()
            if scale is not None and any(abs(float(a) - float(b)) > 1e-6 for a, b in
                                         zip(scale, course.button_plunger.size_xyz_m)):
                raise ValueError('live dynamic plunger collision dimensions must stay fixed')
        for child in list(root.GetChildren()):
            child_path = str(child.GetPath())
            if child_path in {plunger_path, joint_path}:
                continue
            if child.GetName() == 'Cones' and domain == 'navigation':
                stage.RemovePrim(child.GetPath())
            elif child.GetName() not in names and (
                    child.IsA(UsdGeom.Cube) or child.GetAttribute('mssr:obstacleSemantic')):
                stage.RemovePrim(child.GetPath())
    for box in boxes:
        prim = stage.GetPrimAtPath(root_path + '/' + box.name)
        if prim:
            # Existing xform attributes may be reused, but their ordered ops
            # must be cleared before the shared installer calls Add*Op again.
            UsdGeom.Xformable(prim).ClearXformOpOrder()
            if not box.collidable and prim.HasAPI(UsdPhysics.CollisionAPI):
                prim.RemoveAPI(UsdPhysics.CollisionAPI)
    _install_course_boxes(stage, root_path, boxes)
    root = stage.GetPrimAtPath(root_path)
    root.CreateAttribute('mssr:c1Domain', Sdf.ValueTypeNames.String).Set(domain)
    if domain == 'navigation':
        _install_phase1_static_cones(stage, root_path + '/Cones', course.cones)
    if domain == 'manipulation':
        _install_button_prismatic_joint(stage, root_path=root_path,
            plunger_name=course.button_plunger.name,
            center_xyz_m=tuple(float(a) + float(b) for a, b in zip(course.button_center_xyz_m, origin)),
            press_direction_world_xy=course.press_direction_world_xy,
            plunger_yaw_deg=course.button_plunger.yaw_deg,
            joint_name='ButtonPrismaticJoint')
    return plunger_path
