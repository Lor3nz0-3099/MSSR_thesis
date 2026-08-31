from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from smores_ep.config.geometry import SmoresGeometry
from smores_ep.config.physics import SmoresContactConfig, SmoresMassConfig
from smores_ep.isaac.kinematic_stage import (
    _add_filtered_visual,
    _add_rotating_mechanism_part,
)


PHYSICS_ROOT = "/World/smores_ep"

# Reduced central collision core for the fixed chassis.  The visible CAD has
# deep wheel-side reliefs, so a near wheel-diameter cuboid is a poor proxy when
# the module pitches over a sharp riser.  Keeping this core centred and well
# inside the rolling circle leaves stair-edge contact to the two tire shapes;
# the separately authored TILT/PAN face and rear skid remain collidable.
CHASSIS_PROXY_CENTER_M = (0.0, 0.0, 0.0)
CHASSIS_PROXY_SIZE_M = (0.030, 0.040, 0.030)


def _set_link_pose(prim: Any, position: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*position))


def _apply_rigid_body(
    prim: Any,
    mass_kg: float,
    center_of_mass_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    from pxr import Gf, UsdPhysics

    body = UsdPhysics.RigidBodyAPI.Apply(prim)
    body.CreateRigidBodyEnabledAttr(True)
    mass = UsdPhysics.MassAPI.Apply(prim)
    mass.CreateMassAttr(mass_kg)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*center_of_mass_m))


def _define_material(
    stage: Any,
    path: str,
    static_friction: float,
    dynamic_friction: float,
) -> Any:
    from pxr import Sdf, UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, path)
    prim = material.GetPrim()
    physics = UsdPhysics.MaterialAPI.Apply(prim)
    physics.CreateStaticFrictionAttr(static_friction)
    physics.CreateDynamicFrictionAttr(dynamic_friction)
    physics.CreateRestitutionAttr(0.0)
    # Multiplication preserves the intended contrast between high-traction
    # wheels and the low-friction passive skid when they meet the same floor.
    applied_schemas = list(prim.GetAppliedSchemas())
    if "PhysxMaterialAPI" not in applied_schemas:
        applied_schemas.append("PhysxMaterialAPI")
        prim.SetMetadata(
            "apiSchemas",
            Sdf.TokenListOp.CreateExplicit(applied_schemas),
        )
    prim.CreateAttribute(
        "physxMaterial:frictionCombineMode",
        Sdf.ValueTypeNames.Token,
    ).Set("multiply")
    return material


def _bind_collision_material(prim: Any, material: Any) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material,
        UsdShade.Tokens.weakerThanDescendants,
        "physics",
    )


def _add_box_collider(
    stage: Any,
    path: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    material: Any,
) -> Any:
    from pxr import Gf, UsdGeom, UsdPhysics

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(*center))
    xform.AddScaleOp().Set(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    _bind_collision_material(cube.GetPrim(), material)
    return cube


def _add_convex_cylinder_collider(
    stage: Any,
    path: str,
    axis: str,
    radius_m: float,
    height_m: float,
    material: Any,
    sides: int = 32,
) -> Any:
    """Create an explicit convex wheel shape for reliable dynamic contact."""

    from pxr import Gf, UsdGeom, UsdPhysics

    if axis not in ("X", "Y"):
        raise ValueError(f"Unsupported convex-cylinder axis: {axis}")
    if sides < 8:
        raise ValueError("Convex cylinder requires at least eight sides")

    half_height = 0.5 * height_m
    points = []
    for axial in (-half_height, half_height):
        for index in range(sides):
            angle = 2.0 * math.pi * index / sides
            radial_a = radius_m * math.cos(angle)
            radial_b = radius_m * math.sin(angle)
            point = (
                (axial, radial_a, radial_b)
                if axis == "X"
                else (radial_a, axial, radial_b)
            )
            points.append(Gf.Vec3f(*point))

    face_vertex_counts = []
    face_vertex_indices = []
    for index in range(sides):
        following = (index + 1) % sides
        face_vertex_counts.append(4)
        face_vertex_indices.extend(
            (index, following, sides + following, sides + index)
        )
    face_vertex_counts.extend((sides, sides))
    face_vertex_indices.extend(reversed(range(sides)))
    face_vertex_indices.extend(range(sides, 2 * sides))

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(points))
    mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision.CreateApproximationAttr("convexHull")
    _bind_collision_material(mesh.GetPrim(), material)
    return mesh


def _add_sphere_collider(
    stage: Any,
    path: str,
    center: tuple[float, float, float],
    radius_m: float,
    material: Any,
) -> Any:
    from pxr import Gf, UsdGeom, UsdPhysics

    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.CreateRadiusAttr(radius_m)
    sphere.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(*center))
    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    _bind_collision_material(sphere.GetPrim(), material)
    return sphere


def _define_revolute_joint(
    stage: Any,
    path: str,
    body0_path: str,
    body1_path: str,
    axis: str,
    local_position_body0: tuple[float, float, float],
    lower_limit_deg: float | None = None,
    upper_limit_deg: float | None = None,
    drive_mode: str = "velocity",
    max_force_nm: float = 0.4,
) -> Any:
    from pxr import Gf, UsdPhysics

    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body0_path])
    joint.CreateBody1Rel().SetTargets([body1_path])
    joint.CreateAxisAttr(axis)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*local_position_body0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateCollisionEnabledAttr(False)
    if lower_limit_deg is not None:
        joint.CreateLowerLimitAttr(lower_limit_deg)
    if upper_limit_deg is not None:
        joint.CreateUpperLimitAttr(upper_limit_deg)

    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateMaxForceAttr(max_force_nm)
    if drive_mode == "velocity":
        drive.CreateStiffnessAttr(0.0)
        # Geared DC motors resist free rolling; this gain also prevents the
        # light module from overshooting a commanded wheel speed after contact
        # impulses.
        drive.CreateDampingAttr(0.18)
        drive.CreateTargetVelocityAttr(0.0)
    elif drive_mode == "position":
        drive.CreateStiffnessAttr(8.0)
        drive.CreateDampingAttr(0.8)
        drive.CreateTargetPositionAttr(0.0)
    else:
        raise ValueError(f"Unsupported joint drive mode: {drive_mode}")
    return joint


def _define_docking_frame(
    stage: Any,
    path: str,
    position: tuple[float, float, float],
    rotate_z_deg: float,
    face_name: str,
) -> None:
    from pxr import Gf, Sdf, UsdGeom

    frame = UsdGeom.Xform.Define(stage, path)
    xform = UsdGeom.Xformable(frame)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    if rotate_z_deg:
        xform.AddRotateZOp().Set(rotate_z_deg)
    frame.GetPrim().CreateAttribute(
        "smores:face",
        Sdf.ValueTypeNames.Token,
    ).Set(face_name)
    frame.GetPrim().CreateAttribute(
        "smores:dockingNormalAxis",
        Sdf.ValueTypeNames.Token,
    ).Set("X")


def build_physics_asset(
    output_usd: Path,
    visual_reference: Path,
    geometry: SmoresGeometry | None = None,
    masses: SmoresMassConfig | None = None,
    contacts: SmoresContactConfig | None = None,
) -> Path:
    """Generate the reusable SMORES-EP articulation asset."""

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    dimensions = geometry or SmoresGeometry()
    mass = masses or SmoresMassConfig()
    contact = contacts or SmoresContactConfig()
    if abs(mass.total_kg - dimensions.module_mass_kg) > 1.0e-9:
        raise ValueError(
            f"Link masses sum to {mass.total_kg} kg, expected "
            f"{dimensions.module_mass_kg} kg"
        )

    destination = output_usd.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(destination))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    root = UsdGeom.Xform.Define(stage, PHYSICS_ROOT)
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    # The lightweight offline USD tool has the PhysX Python bindings but does
    # not load Kit's schema registry. Author the applied-schema token directly;
    # Isaac resolves it to PhysxArticulationAPI when the asset is opened.
    applied_schemas = list(root.GetPrim().GetAppliedSchemas())
    applied_schemas.append("PhysxArticulationAPI")
    root.GetPrim().SetMetadata(
        "apiSchemas",
        Sdf.TokenListOp.CreateExplicit(applied_schemas),
    )
    root.GetPrim().CreateAttribute(
        "physxArticulation:articulationEnabled",
        Sdf.ValueTypeNames.Bool,
    ).Set(True)
    root.GetPrim().CreateAttribute(
        "physxArticulation:enabledSelfCollisions",
        Sdf.ValueTypeNames.Bool,
    ).Set(False)
    root.GetPrim().CreateAttribute(
        "physxArticulation:solverPositionIterationCount",
        Sdf.ValueTypeNames.Int,
    ).Set(32)
    root.GetPrim().CreateAttribute(
        "physxArticulation:solverVelocityIterationCount",
        Sdf.ValueTypeNames.Int,
    ).Set(8)
    root.GetPrim().CreateAttribute(
        "smores:sourceVisual",
        Sdf.ValueTypeNames.Asset,
    ).Set(str(visual_reference))
    root.GetPrim().CreateAttribute(
        "smores:moduleMassKg",
        Sdf.ValueTypeNames.Double,
    ).Set(dimensions.module_mass_kg)
    root.GetPrim().CreateAttribute(
        "smores:coordinateConvention",
        Sdf.ValueTypeNames.String,
    ).Set("ROS REP-103: +X forward, +Y left, +Z up")

    material_root = f"{PHYSICS_ROOT}/materials"
    wheel_material = _define_material(
        stage,
        f"{material_root}/wheel",
        contact.wheel_static_friction,
        contact.wheel_dynamic_friction,
    )
    body_material = _define_material(
        stage,
        f"{material_root}/body",
        contact.body_static_friction,
        contact.body_dynamic_friction,
    )
    skid_material = _define_material(
        stage,
        f"{material_root}/passive_skid",
        contact.skid_static_friction,
        contact.skid_dynamic_friction,
    )
    pan_material = _define_material(
        stage,
        f"{material_root}/pan_face",
        contact.pan_static_friction,
        contact.pan_dynamic_friction,
    )

    body_path = f"{PHYSICS_ROOT}/body_link"
    body = UsdGeom.Xform.Define(stage, body_path)
    _apply_rigid_body(
        body.GetPrim(),
        mass.body_kg,
        mass.body_com_body_m,
    )
    _add_filtered_visual(
        stage,
        body_path,
        visual_reference,
        dimensions.source_body_origin_m,
        dimensions.fixed_parts,
        dimensions,
    )
    mechanism_path = f"{body_path}/mechanism_visuals"
    UsdGeom.Scope.Define(stage, mechanism_path)
    mechanism_visuals = (
        (
            "outer_left_pinion",
            dimensions.source_outer_left_pinion_center_m,
            dimensions.outer_left_pinion_parts,
        ),
        (
            "outer_right_pinion",
            dimensions.source_outer_right_pinion_center_m,
            dimensions.outer_right_pinion_parts,
        ),
        (
            "inner_left_gear",
            dimensions.source_inner_left_gear_center_m,
            dimensions.inner_left_gear_parts,
        ),
        (
            "inner_right_gear",
            dimensions.source_inner_right_gear_center_m,
            dimensions.inner_right_gear_parts,
        ),
        (
            "inner_left_pinion",
            dimensions.source_inner_left_pinion_center_m,
            dimensions.inner_left_pinion_parts,
        ),
        (
            "inner_right_pinion",
            dimensions.source_inner_right_pinion_center_m,
            dimensions.inner_right_pinion_parts,
        ),
    )
    for name, source_center, visible_parts in mechanism_visuals:
        _add_rotating_mechanism_part(
            stage,
            mechanism_path,
            name,
            source_center,
            visible_parts,
            visual_reference,
            dimensions,
        )
    # Its X-Z corner radius is about 21.2 mm, leaving 9.8 mm inside the
    # 31.06 mm wheel rolling envelope at every body pitch.  The previous
    # 38 x 40 mm proxy left only 0.6 mm nominal radial margin, which was less
    # than the effective PhysX contact skin at a sharp stair edge.
    _add_box_collider(
        stage,
        f"{body_path}/colliders/chassis",
        CHASSIS_PROXY_CENTER_M,
        CHASSIS_PROXY_SIZE_M,
        body_material,
    )
    # The rear lower edge is the low-friction third support described by
    # Davey et al. Its transformed CAD bottom is z=-29.23 mm; the former
    # proxy extended to -32.5 mm and unloaded the driving wheels.
    for side, y_m in (("left", 0.0255), ("right", -0.0255)):
        _add_sphere_collider(
            stage,
            f"{body_path}/colliders/rear_skid_{side}",
            (-0.028, y_m, -0.02798),
            0.00125,
            skid_material,
        )

    left_path = f"{PHYSICS_ROOT}/left_wheel_link"
    left = UsdGeom.Xform.Define(stage, left_path)
    _set_link_pose(left.GetPrim(), dimensions.left_wheel_center_body_m)
    _apply_rigid_body(left.GetPrim(), mass.left_wheel_kg)
    _add_filtered_visual(
        stage,
        left_path,
        visual_reference,
        dimensions.source_left_wheel_center_m,
        dimensions.left_wheel_parts,
        dimensions,
    )
    _add_convex_cylinder_collider(
        stage,
        f"{left_path}/colliders/tire",
        "Y",
        dimensions.wheel_radius_m,
        dimensions.wheel_width_m,
        wheel_material,
    )

    right_path = f"{PHYSICS_ROOT}/right_wheel_link"
    right = UsdGeom.Xform.Define(stage, right_path)
    _set_link_pose(right.GetPrim(), dimensions.right_wheel_center_body_m)
    _apply_rigid_body(right.GetPrim(), mass.right_wheel_kg)
    _add_filtered_visual(
        stage,
        right_path,
        visual_reference,
        dimensions.source_right_wheel_center_m,
        dimensions.right_wheel_parts,
        dimensions,
    )
    _add_convex_cylinder_collider(
        stage,
        f"{right_path}/colliders/tire",
        "Y",
        dimensions.wheel_radius_m,
        dimensions.wheel_width_m,
        wheel_material,
    )

    tilt_path = f"{PHYSICS_ROOT}/tilt_link"
    tilt = UsdGeom.Xform.Define(stage, tilt_path)
    _apply_rigid_body(
        tilt.GetPrim(),
        mass.tilt_carrier_kg,
        dimensions.tilt_carrier_center_body_m,
    )
    _add_filtered_visual(
        stage,
        tilt_path,
        visual_reference,
        dimensions.source_body_origin_m,
        dimensions.tilt_parts,
        dimensions,
    )
    _add_box_collider(
        stage,
        f"{tilt_path}/colliders/carrier",
        (0.027, 0.0, 0.002),
        (0.032, 0.040, 0.026),
        body_material,
    )

    pan_path = f"{PHYSICS_ROOT}/pan_link"
    pan = UsdGeom.Xform.Define(stage, pan_path)
    _set_link_pose(pan.GetPrim(), dimensions.pan_center_body_m)
    _apply_rigid_body(pan.GetPrim(), mass.pan_face_kg)
    _add_filtered_visual(
        stage,
        pan_path,
        visual_reference,
        dimensions.source_pan_center_m,
        dimensions.pan_parts,
        dimensions,
    )
    _add_convex_cylinder_collider(
        stage,
        f"{pan_path}/colliders/face",
        "X",
        dimensions.pan_face_radius_m,
        dimensions.pan_face_thickness_m,
        pan_material,
    )

    joints_path = f"{PHYSICS_ROOT}/joints"
    UsdGeom.Scope.Define(stage, joints_path)
    _define_revolute_joint(
        stage,
        f"{joints_path}/left_wheel_joint",
        body_path,
        left_path,
        "Y",
        dimensions.left_wheel_center_body_m,
        drive_mode="velocity",
        max_force_nm=1.2,
    )
    _define_revolute_joint(
        stage,
        f"{joints_path}/right_wheel_joint",
        body_path,
        right_path,
        "Y",
        dimensions.right_wheel_center_body_m,
        drive_mode="velocity",
        max_force_nm=1.2,
    )
    _define_revolute_joint(
        stage,
        f"{joints_path}/tilt_joint",
        body_path,
        tilt_path,
        "Y",
        (0.0, 0.0, 0.0),
        lower_limit_deg=-90.0,
        upper_limit_deg=90.0,
        drive_mode="position",
        max_force_nm=2.3,
    )
    _define_revolute_joint(
        stage,
        f"{joints_path}/pan_joint",
        tilt_path,
        pan_path,
        "X",
        dimensions.pan_center_body_m,
        # SMORES-EP PAN is continuous. Runtime closes the position loop in
        # unwrapped coordinates and commands this PhysX velocity drive.
        drive_mode="velocity",
        max_force_nm=1.4,
    )

    docking_root = f"{PHYSICS_ROOT}/docking_faces"
    UsdGeom.Scope.Define(stage, docking_root)
    _define_docking_frame(
        stage,
        f"{left_path}/docking_face",
        (0.0, 0.5 * dimensions.wheel_width_m, 0.0),
        90.0,
        "LEFT",
    )
    _define_docking_frame(
        stage,
        f"{right_path}/docking_face",
        (0.0, -0.5 * dimensions.wheel_width_m, 0.0),
        -90.0,
        "RIGHT",
    )
    _define_docking_frame(
        stage,
        f"{pan_path}/docking_face",
        (0.5 * dimensions.pan_face_thickness_m, 0.0, 0.0),
        0.0,
        "TOP",
    )
    _define_docking_frame(
        stage,
        f"{body_path}/docking_face",
        (dimensions.bottom_face_x_m, 0.0, 0.0),
        180.0,
        "BOTTOM",
    )

    # Document the effective motor mixing on the physics asset.
    for name, expression in (
        ("motorA", "tilt + pan"),
        ("motorB", "tilt - pan"),
    ):
        root.GetPrim().CreateAttribute(
            f"smores:panTiltMixer:{name}",
            Sdf.ValueTypeNames.String,
        ).Set(expression)
    root.GetPrim().CreateAttribute(
        "smores:tiltCommandToJointScale",
        Sdf.ValueTypeNames.Double,
    ).Set(-1.0)

    stage.GetRootLayer().Save()
    return destination
