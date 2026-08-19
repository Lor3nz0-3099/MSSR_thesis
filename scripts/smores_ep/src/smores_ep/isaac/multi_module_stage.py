from __future__ import annotations

from typing import Any


def _retarget_relationships(
    stage: Any,
    old_root: str,
    new_root: str,
) -> None:
    from pxr import Sdf, Usd

    old = Sdf.Path(old_root)
    new = Sdf.Path(new_root)
    for prim in Usd.PrimRange(stage.GetPrimAtPath(new_root)):
        for relationship in prim.GetRelationships():
            rewritten = []
            changed = False
            for target in relationship.GetTargets():
                if target.HasPrefix(old):
                    rewritten.append(
                        new.AppendPath(target.MakeRelativePath(old))
                    )
                    changed = True
                else:
                    rewritten.append(target)
            if changed:
                relationship.SetTargets(rewritten)


def clone_module(
    stage: Any,
    source_root: str,
    destination_root: str,
) -> None:
    """Clone one authored module and retarget all internal relationships."""

    from pxr import Sdf

    if not stage.GetPrimAtPath(source_root):
        raise RuntimeError(f"Source module does not exist: {source_root}")
    if stage.GetPrimAtPath(destination_root):
        raise RuntimeError(
            f"Destination module already exists: {destination_root}"
        )
    layer = stage.GetRootLayer()
    Sdf.CopySpec(
        layer,
        Sdf.Path(source_root),
        layer,
        Sdf.Path(destination_root),
    )
    _retarget_relationships(stage, source_root, destination_root)


def set_module_pose(
    stage: Any,
    module_root: str,
    position_world_m: tuple[float, float, float],
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(module_root)
    if not prim:
        raise RuntimeError(f"Module root does not exist: {module_root}")
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position_world_m))
    if yaw_deg:
        xform.AddRotateZOp().Set(float(yaw_deg))
    if pitch_deg:
        xform.AddRotateYOp().Set(float(pitch_deg))


def use_collision_proxy_visuals(
    stage: Any,
    module_roots: dict[str, str],
) -> None:
    """Render only lightweight physics proxies for a multi-module scene.

    The imported CAD is repeated below every independently moving link and
    below each decorative mechanism part. Hiding those subtrees and rendering
    the already-authored convex colliders keeps the articulation and contact
    model unchanged while drastically reducing viewport scene complexity.
    """

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    hidden_relative_paths = (
        "body_link/visual_offset",
        "body_link/mechanism_visuals",
        "left_wheel_link/visual_offset",
        "right_wheel_link/visual_offset",
        "tilt_link/visual_offset",
        "pan_link/visual_offset",
    )
    link_colors = {
        "body_link": Gf.Vec3f(0.18, 0.22, 0.28),
        "left_wheel_link": Gf.Vec3f(0.035, 0.035, 0.04),
        "right_wheel_link": Gf.Vec3f(0.035, 0.035, 0.04),
        "tilt_link": Gf.Vec3f(0.25, 0.32, 0.38),
        "pan_link": Gf.Vec3f(0.95, 0.42, 0.08),
    }

    for module_root in module_roots.values():
        for relative_path in hidden_relative_paths:
            prim = stage.GetPrimAtPath(f"{module_root}/{relative_path}")
            if prim:
                UsdGeom.Imageable(prim).MakeInvisible()

        root_prim = stage.GetPrimAtPath(module_root)
        for prim in Usd.PrimRange(root_prim):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            imageable = UsdGeom.Imageable(prim)
            imageable.CreatePurposeAttr().Set(UsdGeom.Tokens.default_)
            imageable.MakeVisible()

            relative = str(prim.GetPath())[len(module_root) + 1 :]
            link_name = relative.split("/", 1)[0]
            color = link_colors.get(link_name, Gf.Vec3f(0.35, 0.35, 0.38))
            gprim = UsdGeom.Gprim(prim)
            if gprim:
                gprim.CreateDisplayColorAttr().Set([color])
