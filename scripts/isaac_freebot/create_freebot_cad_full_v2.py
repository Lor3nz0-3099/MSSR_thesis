from pathlib import Path
import math

from isaacsim import SimulationApp


# Isaac Sim 6 exposes the USD Python modules after the application plugins
# have initialized.  The generator is headless and does not run physics.
simulation_app = SimulationApp({"headless": True})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


# The real wheel supports are rigidly fixed to the chassis.  Keep this asset
# separate from the previous compliant-mount experiment so runs cannot silently
# pick up the non-physical +/-28 degree wheel-support motion.
OUT_USD = Path("assets/freebot/usd_physics/freebot_cad_full_nearer_wheels_rigid.usd")
VISUAL_USD = Path("../usd_visual/freebot_visual_nearer_wheels.usd")
VISUAL_ROOT_PATH = "/World/FreeBOT_simplified_clean_nearer_wheels/FreeBOT_simplified_clean"

SHELL_CENTER = Gf.Vec3d(0.023103, 0.059745, 0.055943)
INTERNAL_CENTER = Gf.Vec3d(0.023065, 0.060436, 0.026953)
LEFT_WHEEL_CENTER = Gf.Vec3d(0.023042, 0.094171, 0.032054)
RIGHT_WHEEL_CENTER = Gf.Vec3d(0.023043, 0.026872, 0.031846)
CASTER_1_CENTER = Gf.Vec3d(-0.031633, 0.060363, 0.054020)
CASTER_2_CENTER = Gf.Vec3d(0.077764, 0.060363, 0.054020)
MAGNET_CENTER = Gf.Vec3d(0.023121, 0.060587, 0.009453)

SHELL_RADIUS_M = 0.0665

# Paper-matched mass budget.  Table I reports 307.9 g for one complete module.
# One Pololu 32x7 mm wheel weighs 0.11 oz.  Caster balls are modeled as solid
# steel spheres; the remaining mass belongs to chassis, motors, magnet,
# battery and supports.
MODULE_TOTAL_MASS_KG = 0.3079
SHELL_MASS_KG = 0.060
INTERNAL_MECHANISM_TOTAL_MASS_KG = MODULE_TOTAL_MASS_KG - SHELL_MASS_KG
WHEEL_MOUNT_MASS_KG = 0.006
WHEEL_MASS_KG = 0.11 * 0.028349523125
CASTER_BALL_MASS_KG = (
    (4.0 / 3.0) * math.pi * 0.00465**3 * 7850.0
)
USE_COMPLIANT_WHEEL_MOUNTS = False
INTERNAL_LINK_MASS_KG = (
    INTERNAL_MECHANISM_TOTAL_MASS_KG
    - 2.0 * WHEEL_MASS_KG
    - 2.0 * CASTER_BALL_MASS_KG
    - (2.0 * WHEEL_MOUNT_MASS_KG if USE_COMPLIANT_WHEEL_MOUNTS else 0.0)
)

# Safety/end-stop collision only. Runtime stage normalization keeps both balls
# 2 mm away from the fitted inner shell in the nominal carrier pose.
ENABLE_CASTER_COLLIDERS = True
WHEEL_MOUNT_SWING_LIMIT_DEG = 28.0
WHEEL_MOUNT_STIFFNESS = 0.10
WHEEL_MOUNT_DAMPING = 0.035
WHEEL_MOUNT_PIVOT_INBOARD_M = 0.012
WHEEL_TIRE_PARTS = {
    "tn__PololuWheel32x7mmtire_____________1_ya0rECJ",
}
TIRE_CONTACT_OFFSET_M = 0.0015
TIRE_REST_OFFSET_M = 0.0002
# No uncalibrated compliance is added to the CAD tire contact.  PhysX resolves
# the CAD SDF surfaces with the authored Coulomb material directly.
TIRE_COMPLIANT_STIFFNESS = 0.0
TIRE_COMPLIANT_DAMPING = 0.0

SHELL_PARTS = {
    "tn__shell_loweript1_YNX6",
    "tn__shell_upper1_qN",
}
LEFT_WHEEL_PARTS = {
    "tn__PololuWheel32x7mm1_tQCi8",
    "tn__tronco_ruota_21_hT",
    "tn__tronco_ruota_22_hT",
    "tn__tronco_ruota31_kR",
}
RIGHT_WHEEL_PARTS = {
    "tn__PololuWheel32x7mm2_tQCi8",
    "tn__tronco_ruota_23_hT",
    "tn__tronco_ruota_24_hT",
    "tn__tronco_ruota32_kR",
}
CASTER_1_PARTS = {"tn__ball1_gA"}
CASTER_2_PARTS = {"tn__ball2_gA"}
INTERNAL_PARTS = {
    "tn__chassis1_XG",
    "tn__chassis31_zH",
    "tn__chassis32_zH",
    "tn__chassis21_m9l8",
    "tn__chassis22_m9l8",
    "tn__telaiopercasters1_tHAgG",
    "tn__base1_gA",
    "tn__base2_gA",
    "tn__base_telaio_casters1_xc0",
    "tn__base_telaio_casters2_xc0",
    "tn__magnete1_XG",
    "tn__M0578Y01A1_YFA",
    "tn__M0578Y01A2_YFg4",
    "tn__Engrenagemreta28D1_yIy7J",
    "tn__Engrenagemreta28D2_yIy7i8",
}


def define_material(
    stage,
    path,
    static_friction,
    dynamic_friction,
    restitution,
    friction_combine="average",
    restitution_combine="min",
    compliant_stiffness=0.0,
    compliant_damping=0.0,
):
    material = UsdShade.Material.Define(stage, path)
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(static_friction)
    physics.CreateDynamicFrictionAttr(dynamic_friction)
    physics.CreateRestitutionAttr(restitution)
    prim = material.GetPrim()
    prim.CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set(friction_combine)
    prim.CreateAttribute("physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token).Set(restitution_combine)
    prim.CreateAttribute("physxMaterial:compliantContactStiffness", Sdf.ValueTypeNames.Float).Set(compliant_stiffness)
    prim.CreateAttribute("physxMaterial:compliantContactDamping", Sdf.ValueTypeNames.Float).Set(compliant_damping)
    return material


def bind_material(prim, material):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def apply_physx_sdf_api_by_name(prim, resolution=128, margin=0.001, narrow_band=0.003):
    api_schemas = prim.GetMetadata("apiSchemas") or Sdf.TokenListOp()
    explicit_items = list(api_schemas.explicitItems)
    if "PhysxSDFMeshCollisionAPI" not in explicit_items:
        explicit_items.append("PhysxSDFMeshCollisionAPI")
    api_schemas.explicitItems = explicit_items
    prim.SetMetadata("apiSchemas", api_schemas)

    prim.CreateAttribute("physxSDFMeshCollision:sdfResolution", Sdf.ValueTypeNames.Int).Set(resolution)
    prim.CreateAttribute("physxSDFMeshCollision:sdfSubgridResolution", Sdf.ValueTypeNames.Int).Set(6)
    prim.CreateAttribute("physxSDFMeshCollision:sdfBitsPerSubgridPixel", Sdf.ValueTypeNames.Token).Set("BitsPerPixel16")
    prim.CreateAttribute("physxSDFMeshCollision:sdfMargin", Sdf.ValueTypeNames.Float).Set(margin)
    prim.CreateAttribute("physxSDFMeshCollision:sdfNarrowBandThickness", Sdf.ValueTypeNames.Float).Set(narrow_band)
    prim.CreateAttribute("physxSDFMeshCollision:sdfEnableRemeshing", Sdf.ValueTypeNames.Bool).Set(False)


def add_cad_reference(stage, parent_path, reference_name, center, visible_part_names):
    ref = UsdGeom.Xform.Define(stage, f"{parent_path}/{reference_name}")
    ref.AddTranslateOp().Set(-center)
    ref.AddScaleOp().Set(Gf.Vec3f(0.01, 0.01, 0.01))
    ref.GetPrim().GetReferences().AddReference(str(VISUAL_USD), VISUAL_ROOT_PATH)

    for child in ref.GetPrim().GetChildren():
        if child.GetName() not in visible_part_names:
            UsdGeom.Imageable(child).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    return ref


def find_named_prims(root, names):
    found = []
    for prim in Usd.PrimRange(root):
        if prim.GetName() in names:
            found.append(prim)
    return found


def apply_sdf_collision_to_parts(root, part_names, material, resolution=128):
    applied = []
    for part in find_named_prims(root, part_names):
        for prim in Usd.PrimRange(part):
            UsdPhysics.CollisionAPI.Apply(prim)
            mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_api.CreateApproximationAttr("sdf")
            apply_physx_sdf_api_by_name(prim, resolution=resolution)
            bind_material(prim, material)
            applied.append(str(prim.GetPath()))
    return applied


def apply_sdf_collision_to_descendants(
    root,
    container_names,
    collider_names,
    material,
    resolution=128,
    contact_offset=None,
    rest_offset=None,
):
    applied = []
    for container in find_named_prims(root, container_names):
        for prim in Usd.PrimRange(container):
            if prim.GetName() not in collider_names:
                continue
            for collider_prim in Usd.PrimRange(prim):
                UsdPhysics.CollisionAPI.Apply(collider_prim)
                mesh_api = UsdPhysics.MeshCollisionAPI.Apply(collider_prim)
                mesh_api.CreateApproximationAttr("sdf")
                apply_physx_sdf_api_by_name(collider_prim, resolution=resolution)
                if contact_offset is not None:
                    collider_prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(contact_offset)
                if rest_offset is not None:
                    collider_prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(rest_offset)
                bind_material(collider_prim, material)
                applied.append(str(collider_prim.GetPath()))
    return applied


def make_rigid_xform(stage, path, center, mass, kinematic=False):
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(center)
    rb = UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(xform.GetPrim()).CreateMassAttr(mass)
    return xform


def tune_rigid_body_solver(
    prim,
    pos_iters=32,
    vel_iters=8,
    max_depenetration_velocity=1.0,
    linear_damping=0.0,
    angular_damping=0.05,
):
    prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(pos_iters)
    prim.CreateAttribute("physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(vel_iters)
    prim.CreateAttribute("physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float).Set(max_depenetration_velocity)
    prim.CreateAttribute("physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float).Set(linear_damping)
    prim.CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(angular_damping)


def make_revolute_drive(stage, path, parent, child, parent_center, child_center, joint_world_pos, velocity_deg_s):
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(parent)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(child)])
    joint.CreateAxisAttr("Y")
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*(joint_world_pos - parent_center)))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*(joint_world_pos - child_center)))

    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateTargetVelocityAttr(velocity_deg_s)
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(1000.0)
    drive.CreateMaxForceAttr(5.0)
    return joint


def make_limited_compliant_mount(
    stage,
    path,
    parent,
    child,
    parent_center,
    child_center,
    joint_world_pos,
    swing_limit_deg=WHEEL_MOUNT_SWING_LIMIT_DEG,
    stiffness=WHEEL_MOUNT_STIFFNESS,
    damping=WHEEL_MOUNT_DAMPING,
):
    joint = UsdPhysics.Joint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(parent)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(child)])
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*(joint_world_pos - parent_center)))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*(joint_world_pos - child_center)))

    # Generic D6 joint. Translation is locked, spin about Y is locked here, and
    # small rotations about X/Z are spring-centered. The actual wheel spin still
    # happens in the downstream revolute joint.
    for axis in (UsdPhysics.Tokens.transX, UsdPhysics.Tokens.transY, UsdPhysics.Tokens.transZ, UsdPhysics.Tokens.rotY):
        limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
        limit.CreateLowAttr(1.0)
        limit.CreateHighAttr(-1.0)

    for axis in (UsdPhysics.Tokens.rotX, UsdPhysics.Tokens.rotZ):
        limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
        limit.CreateLowAttr(-swing_limit_deg)
        limit.CreateHighAttr(swing_limit_deg)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
        drive.CreateTypeAttr("force")
        drive.CreateTargetPositionAttr(0.0)
        drive.CreateStiffnessAttr(stiffness)
        drive.CreateDampingAttr(damping)
        drive.CreateMaxForceAttr(0.4)

    return joint


def make_spherical_joint(stage, path, parent, child, parent_center, child_center, joint_world_pos):
    joint = UsdPhysics.SphericalJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(parent)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(child)])
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateLocalPos0Attr(Gf.Vec3f(*(joint_world_pos - parent_center)))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*(joint_world_pos - child_center)))
    return joint


def main():
    OUT_USD.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(OUT_USD))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)

    # A/B friction trial derived from the paper's static-balance inequalities:
    # the previous 0.95/0.85 pair fell below the required lower-hemisphere
    # dynamic value (~0.895) as soon as sliding began.
    shell_mat = define_material(stage, "/World/materials/shell_sdf", 1.10, 1.00, 0.0, friction_combine="max")
    rubber_mat = define_material(
        stage,
        "/World/materials/wheel_rubber_sdf",
        2.2,
        1.9,
        0.0,
        friction_combine="max",
        compliant_stiffness=TIRE_COMPLIANT_STIFFNESS,
        compliant_damping=TIRE_COMPLIANT_DAMPING,
    )
    caster_mat = define_material(stage, "/World/materials/caster_sdf", 0.03, 0.02, 0.0, friction_combine="min")
    floor_mat = define_material(stage, "/World/materials/floor", 1.45, 1.25, 0.0, friction_combine="max")

    UsdGeom.Xform.Define(stage, "/World/freebot")
    print(
        "Mass budget: "
        f"shell={SHELL_MASS_KG:.3f} kg, "
        f"internal_total={INTERNAL_MECHANISM_TOTAL_MASS_KG:.3f} kg, "
        f"robot_total={MODULE_TOTAL_MASS_KG:.4f} kg"
    )

    shell = make_rigid_xform(stage, "/World/freebot/shell_link", SHELL_CENTER, SHELL_MASS_KG)
    shell_ref = add_cad_reference(stage, shell.GetPath(), "cad_reference", SHELL_CENTER, SHELL_PARTS)
    print("shell SDF:", len(apply_sdf_collision_to_parts(shell_ref.GetPrim(), SHELL_PARTS, shell_mat, 160)))

    tune_rigid_body_solver(
        shell.GetPrim(),
        pos_iters=64,
        vel_iters=8,
        max_depenetration_velocity=0.6,
        linear_damping=0.0,
        angular_damping=0.015,
    )

    internal = make_rigid_xform(stage, "/World/freebot/internal_link", INTERNAL_CENTER, INTERNAL_LINK_MASS_KG, kinematic=False)
    UsdPhysics.ArticulationRootAPI.Apply(internal.GetPrim())
    tune_rigid_body_solver(
        internal.GetPrim(),
        pos_iters=64,
        vel_iters=12,
        max_depenetration_velocity=0.6,
        linear_damping=0.08,
        angular_damping=0.35,
    )
    internal_ref = add_cad_reference(stage, internal.GetPath(), "cad_reference", INTERNAL_CENTER, INTERNAL_PARTS)
    # The internal chassis, supports and magnet remain visual-only. Their
    # unresolved contact chain is represented by the explicit reduced-order
    # docking model in run_freebot_two_module_climb.py.
    print("internal CAD visual only: colliders disabled for chassis/supports/magnet")
    UsdGeom.Xform.Define(stage, "/World/freebot/internal_link/magnet_frame").AddTranslateOp().Set(MAGNET_CENTER - INTERNAL_CENTER)

    if USE_COMPLIANT_WHEEL_MOUNTS:
        left_mount = make_rigid_xform(stage, "/World/freebot/left_wheel_mount_link", LEFT_WHEEL_CENTER, WHEEL_MOUNT_MASS_KG)
        tune_rigid_body_solver(left_mount.GetPrim(), pos_iters=64, vel_iters=12, max_depenetration_velocity=0.6, angular_damping=0.04)
        right_mount = make_rigid_xform(stage, "/World/freebot/right_wheel_mount_link", RIGHT_WHEEL_CENTER, WHEEL_MOUNT_MASS_KG)
        tune_rigid_body_solver(right_mount.GetPrim(), pos_iters=64, vel_iters=12, max_depenetration_velocity=0.6, angular_damping=0.04)

    left = make_rigid_xform(stage, "/World/freebot/left_wheel_link", LEFT_WHEEL_CENTER, WHEEL_MASS_KG)
    tune_rigid_body_solver(left.GetPrim(), pos_iters=64, vel_iters=12, max_depenetration_velocity=0.6, angular_damping=0.02)
    left_ref = add_cad_reference(stage, left.GetPath(), "cad_reference", LEFT_WHEEL_CENTER, LEFT_WHEEL_PARTS)
    print(
        "left tire SDF:",
        len(
            apply_sdf_collision_to_descendants(
                left_ref.GetPrim(),
                {"tn__PololuWheel32x7mm1_tQCi8"},
                WHEEL_TIRE_PARTS,
                rubber_mat,
                96,
                contact_offset=TIRE_CONTACT_OFFSET_M,
                rest_offset=TIRE_REST_OFFSET_M,
            )
        ),
    )

    right = make_rigid_xform(stage, "/World/freebot/right_wheel_link", RIGHT_WHEEL_CENTER, WHEEL_MASS_KG)
    tune_rigid_body_solver(right.GetPrim(), pos_iters=64, vel_iters=12, max_depenetration_velocity=0.6, angular_damping=0.02)
    right_ref = add_cad_reference(stage, right.GetPath(), "cad_reference", RIGHT_WHEEL_CENTER, RIGHT_WHEEL_PARTS)
    print(
        "right tire SDF:",
        len(
            apply_sdf_collision_to_descendants(
                right_ref.GetPrim(),
                {"tn__PololuWheel32x7mm2_tQCi8"},
                WHEEL_TIRE_PARTS,
                rubber_mat,
                96,
                contact_offset=TIRE_CONTACT_OFFSET_M,
                rest_offset=TIRE_REST_OFFSET_M,
            )
        ),
    )

    caster_1 = make_rigid_xform(stage, "/World/freebot/caster_1_ball_link", CASTER_1_CENTER, CASTER_BALL_MASS_KG)
    tune_rigid_body_solver(caster_1.GetPrim(), pos_iters=32, vel_iters=8, max_depenetration_velocity=0.5, angular_damping=0.02)
    caster_1_ref = add_cad_reference(stage, caster_1.GetPath(), "cad_reference", CASTER_1_CENTER, CASTER_1_PARTS)
    if ENABLE_CASTER_COLLIDERS:
        print("caster 1 SDF:", len(apply_sdf_collision_to_parts(caster_1_ref.GetPrim(), CASTER_1_PARTS, caster_mat, 64)))
    else:
        print("caster 1 visual only: collider disabled for diagnostic test")

    caster_2 = make_rigid_xform(stage, "/World/freebot/caster_2_ball_link", CASTER_2_CENTER, CASTER_BALL_MASS_KG)
    tune_rigid_body_solver(caster_2.GetPrim(), pos_iters=32, vel_iters=8, max_depenetration_velocity=0.5, angular_damping=0.02)
    caster_2_ref = add_cad_reference(stage, caster_2.GetPath(), "cad_reference", CASTER_2_CENTER, CASTER_2_PARTS)
    if ENABLE_CASTER_COLLIDERS:
        print("caster 2 SDF:", len(apply_sdf_collision_to_parts(caster_2_ref.GetPrim(), CASTER_2_PARTS, caster_mat, 64)))
    else:
        print("caster 2 visual only: collider disabled for diagnostic test")

    if USE_COMPLIANT_WHEEL_MOUNTS:
        left_mount_pivot = LEFT_WHEEL_CENTER + Gf.Vec3d(0.0, -WHEEL_MOUNT_PIVOT_INBOARD_M, 0.0)
        right_mount_pivot = RIGHT_WHEEL_CENTER + Gf.Vec3d(0.0, WHEEL_MOUNT_PIVOT_INBOARD_M, 0.0)
        make_limited_compliant_mount(
            stage,
            "/World/freebot/joints/left_wheel_mount_joint",
            "/World/freebot/internal_link",
            "/World/freebot/left_wheel_mount_link",
            INTERNAL_CENTER,
            LEFT_WHEEL_CENTER,
            left_mount_pivot,
        )
        make_limited_compliant_mount(
            stage,
            "/World/freebot/joints/right_wheel_mount_joint",
            "/World/freebot/internal_link",
            "/World/freebot/right_wheel_mount_link",
            INTERNAL_CENTER,
            RIGHT_WHEEL_CENTER,
            right_mount_pivot,
        )
        make_revolute_drive(stage, "/World/freebot/joints/left_wheel_joint", "/World/freebot/left_wheel_mount_link", "/World/freebot/left_wheel_link", LEFT_WHEEL_CENTER, LEFT_WHEEL_CENTER, LEFT_WHEEL_CENTER, 720.0)
        make_revolute_drive(stage, "/World/freebot/joints/right_wheel_joint", "/World/freebot/right_wheel_mount_link", "/World/freebot/right_wheel_link", RIGHT_WHEEL_CENTER, RIGHT_WHEEL_CENTER, RIGHT_WHEEL_CENTER, 720.0)
    else:
        make_revolute_drive(stage, "/World/freebot/joints/left_wheel_joint", "/World/freebot/internal_link", "/World/freebot/left_wheel_link", INTERNAL_CENTER, LEFT_WHEEL_CENTER, LEFT_WHEEL_CENTER, 720.0)
        make_revolute_drive(stage, "/World/freebot/joints/right_wheel_joint", "/World/freebot/internal_link", "/World/freebot/right_wheel_link", INTERNAL_CENTER, RIGHT_WHEEL_CENTER, RIGHT_WHEEL_CENTER, 720.0)
    make_spherical_joint(stage, "/World/freebot/joints/caster_1_joint", "/World/freebot/internal_link", "/World/freebot/caster_1_ball_link", INTERNAL_CENTER, CASTER_1_CENTER, CASTER_1_CENTER)
    make_spherical_joint(stage, "/World/freebot/joints/caster_2_joint", "/World/freebot/internal_link", "/World/freebot/caster_2_ball_link", INTERNAL_CENTER, CASTER_2_CENTER, CASTER_2_CENTER)

    # No joint is authored between shell and internal mechanism. The shell is
    # ferromagnetic, so magnet-shell coupling is modeled as a runtime force in
    # run_freebot_magnetic_locomotion.py, not as a hard kinematic constraint.

    marker = UsdGeom.Cube.Define(stage, "/World/freebot/shell_link/rotation_marker")
    marker.CreateSizeAttr(1.0)
    marker.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, SHELL_RADIUS_M))
    marker.AddScaleOp().Set(Gf.Vec3f(0.004, 0.012, 0.004))
    marker.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.0, 0.0)])

    ground = UsdGeom.Cube.Define(stage, "/World/ground")
    ground.CreateSizeAttr(1.0)
    ground.AddTranslateOp().Set(Gf.Vec3d(0.023, 0.060, SHELL_CENTER[2] - SHELL_RADIUS_M - 0.005))
    ground.AddScaleOp().Set(Gf.Vec3f(5.0, 5.0, 0.01))
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.45, 0.45, 0.45)])
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    bind_material(ground.GetPrim(), floor_mat)

    stage.GetRootLayer().Save()
    print(f"Saved {OUT_USD}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
