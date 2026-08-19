from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from freebot_docking.config.geometry import (
    RunningGearGeometry,
    ShellGeometry,
    WheelRadialComplianceConfig,
)
from freebot_docking.config.mass import ModuleMassConfig
from freebot_docking.isaac.materials import (
    IsaacMaterialConfig,
    assign_freebot_materials,
)
from freebot_docking.physics.contact_geometry import (
    radial_ball_center_shift_m,
    radial_cylinder_center_shift_m,
)


SOURCE_ROOT = "/World/freebot"
ACTIVE_ROOT = "/World/active_module"
PASSIVE_ROOT = "/World/passive_module"
SOURCE_SHELL_CENTER_M = np.array(
    [0.02465733, 0.06062240, 0.06061104],
    dtype=np.float64,
)


@dataclass(frozen=True)
class IsaacStageConfig:
    """Geometry and fixture choices for the first two-module scenario."""

    usd_path: Path
    active_shell_center_world: tuple[float, float, float] = (
        0.0199584,
        0.060,
        0.0633472,
    )
    passive_shell_center_world: tuple[float, float, float] = (
        0.1666528,
        0.060,
        0.0633472,
    )
    active_y_rotation_deg: float = 0.0
    passive_y_rotation_deg: float = 0.0
    passive_fixed: bool = True
    ground_center_world: tuple[float, float, float] = (0.3, 0.06, -0.01)
    ground_size_m: tuple[float, float, float] = (1.8, 0.9, 0.02)
    materials: IsaacMaterialConfig = IsaacMaterialConfig()
    shell_geometry: ShellGeometry = ShellGeometry()
    running_gear: RunningGearGeometry = RunningGearGeometry()
    wheel_radial_compliance: WheelRadialComplianceConfig = (
        WheelRadialComplianceConfig()
    )
    masses: ModuleMassConfig = ModuleMassConfig()

    def __post_init__(self) -> None:
        if (
            self.running_gear.tire_precompression_m > 0.0
            and self.materials.wheel_contact_stiffness_n_per_m <= 0.0
        ):
            raise ValueError(
                "A precompressed tire requires compliant-contact stiffness"
            )
        if (
            self.running_gear.caster_precompression_m > 0.0
            and self.materials.caster_contact_stiffness_n_per_m <= 0.0
        ):
            raise ValueError(
                "A precompressed caster requires compliant-contact stiffness"
            )
        if (
            self.wheel_radial_compliance.enabled
            and 2.0 * self.wheel_radial_compliance.mount_mass_kg
            >= self.masses.internal_link_kg
        ):
            raise ValueError("Wheel mounts exceed the available carrier mass")


@dataclass(frozen=True)
class StageBuildResult:
    stage: Any
    active_root: str
    passive_root: str
    removed_duplicate_colliders: int
    replaced_wheel_colliders: int
    replaced_caster_colliders: int
    radial_suspension_joints: int


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
            targets = relationship.GetTargets()
            rewritten = []
            changed = False
            for target in targets:
                if target.HasPrefix(old):
                    rewritten.append(
                        new.AppendPath(target.MakeRelativePath(old))
                    )
                    changed = True
                else:
                    rewritten.append(target)
            if changed:
                relationship.SetTargets(rewritten)


def _clone_module(stage: Any, source: str, destination: str) -> None:
    from pxr import Sdf

    if not stage.GetPrimAtPath(source):
        raise RuntimeError(f"Source module does not exist: {source}")
    if stage.GetPrimAtPath(destination):
        stage.RemovePrim(destination)
    layer = stage.GetRootLayer()
    Sdf.CopySpec(
        layer,
        Sdf.Path(source),
        layer,
        Sdf.Path(destination),
    )
    _retarget_relationships(stage, source, destination)


def _set_module_pose(
    stage: Any,
    root: str,
    desired_shell_center: tuple[float, float, float],
    y_rotation_deg: float,
) -> None:
    from pxr import Gf, UsdGeom

    rotation = Gf.Rotation(
        Gf.Vec3d(0.0, 1.0, 0.0),
        float(y_rotation_deg),
    )
    rotated_source_center = np.asarray(
        rotation.TransformDir(
            Gf.Vec3d(*(float(value) for value in SOURCE_SHELL_CENTER_M))
        ),
        dtype=np.float64,
    )
    translation = (
        np.asarray(desired_shell_center, dtype=np.float64)
        - rotated_source_center
    )
    transform = Gf.Matrix4d(1.0)
    transform.SetRotate(rotation)
    transform.SetTranslateOnly(
        Gf.Vec3d(*(float(value) for value in translation))
    )
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(root))
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(transform)


def _set_rigid_body_mode(stage: Any, root: str, fixed: bool) -> int:
    from pxr import Sdf, Usd, UsdPhysics

    count = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        body = UsdPhysics.RigidBodyAPI.Apply(prim)
        body.CreateKinematicEnabledAttr(bool(fixed))
        prim.CreateAttribute(
            "physxRigidBody:solverVelocityIterationCount",
            Sdf.ValueTypeNames.Int,
        ).Set(4)
        count += 1
    if count == 0:
        raise RuntimeError(f"Module contains no rigid bodies: {root}")
    return count


def _set_module_masses(
    stage: Any,
    root: str,
    masses: ModuleMassConfig,
    radial: WheelRadialComplianceConfig,
) -> None:
    """Override legacy USD masses with the paper-matched mass budget."""

    from pxr import UsdPhysics

    mount_total = 2.0 * radial.mount_mass_kg if radial.enabled else 0.0
    internal_mass = masses.internal_link_kg - mount_total
    if internal_mass <= 0.0:
        raise ValueError("Wheel mounts exceed the available carrier mass")
    body_masses = {
        "shell_link": masses.shell_kg,
        "internal_link": internal_mass,
        "left_wheel_link": masses.wheel_kg,
        "right_wheel_link": masses.wheel_kg,
        "caster_1_ball_link": masses.caster_ball_kg,
        "caster_2_ball_link": masses.caster_ball_kg,
    }
    if radial.enabled:
        body_masses.update(
            {
                "left_wheel_radial_mount_link": radial.mount_mass_kg,
                "right_wheel_radial_mount_link": radial.mount_mass_kg,
            }
        )
    for body_name, mass_kg in body_masses.items():
        path = f"{root}/{body_name}"
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Mass target rigid body is missing: {path}")
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass_kg)).Set(
            float(mass_kg)
        )


def _set_module_mass_properties(
    stage: Any,
    root: str,
    masses: ModuleMassConfig,
    shell: ShellGeometry,
    gear: RunningGearGeometry,
    radial: WheelRadialComplianceConfig,
) -> None:
    """Author finite COM and inertia values instead of relying on auto defaults."""

    from pxr import Gf, UsdPhysics

    mean_shell_inertia = (
        (2.0 / 5.0)
        * masses.shell_kg
        * (shell.outer_radius_m**5 - shell.inner_radius_m**5)
        / (shell.outer_radius_m**3 - shell.inner_radius_m**3)
    )
    wheel_axial = 0.5 * masses.wheel_kg * gear.tire_outer_radius_m**2
    wheel_transverse = (
        masses.wheel_kg
        * (
            3.0 * gear.tire_outer_radius_m**2
            + (2.0 * gear.tire_half_width_m) ** 2
        )
        / 12.0
    )
    caster_inertia = (
        0.4 * masses.caster_ball_kg * gear.caster_ball_radius_m**2
    )
    mount_total = 2.0 * radial.mount_mass_kg if radial.enabled else 0.0
    internal_mass = masses.internal_link_kg - mount_total
    internal_inertia_scale = internal_mass / masses.internal_link_kg
    properties = {
        "shell_link": (
            shell.center_from_body_origin_m,
            (mean_shell_inertia,) * 3,
        ),
        "internal_link": (
            (0.0, 0.0, 0.0),
            tuple(
                value * internal_inertia_scale
                for value in masses.internal_box_diagonal_inertia_kg_m2
            ),
        ),
        "left_wheel_link": (
            (0.0, 0.0, 0.0),
            (wheel_transverse, wheel_axial, wheel_transverse),
        ),
        "right_wheel_link": (
            (0.0, 0.0, 0.0),
            (wheel_transverse, wheel_axial, wheel_transverse),
        ),
        "caster_1_ball_link": (
            (0.0, 0.0, 0.0),
            (caster_inertia,) * 3,
        ),
        "caster_2_ball_link": (
            (0.0, 0.0, 0.0),
            (caster_inertia,) * 3,
        ),
    }
    if radial.enabled:
        mount_inertia = radial.mount_mass_kg * 0.008**2 / 6.0
        properties.update(
            {
                "left_wheel_radial_mount_link": (
                    (0.0, 0.0, 0.0),
                    (mount_inertia,) * 3,
                ),
                "right_wheel_radial_mount_link": (
                    (0.0, 0.0, 0.0),
                    (mount_inertia,) * 3,
                ),
            }
        )
    for name, (center_of_mass, inertia) in properties.items():
        prim = stage.GetPrimAtPath(f"{root}/{name}")
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*center_of_mass))
        mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*inertia))
        mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0))


def _translate_body_and_parent_joint(
    stage: Any,
    root: str,
    body_name: str,
    joint_name: str,
    shift_m: np.ndarray,
) -> None:
    """Move one rigid running-gear body and its parent-side joint anchor."""

    from pxr import Gf

    body = stage.GetPrimAtPath(f"{root}/{body_name}")
    translation = body.GetAttribute("xformOp:translate")
    current = np.asarray(translation.Get(), dtype=np.float64)
    translation.Set(Gf.Vec3d(*(current + shift_m)))

    joint = stage.GetPrimAtPath(f"{root}/joints/{joint_name}")
    local_position = joint.GetAttribute("physics:localPos0")
    current_anchor = np.asarray(local_position.Get(), dtype=np.float64)
    local_position.Set(Gf.Vec3f(*(current_anchor + shift_m)))


def _align_running_gear(
    stage: Any,
    root: str,
    shell: ShellGeometry,
    gear: RunningGearGeometry,
) -> None:
    """Place wheels and casters at their configured inner-shell clearances."""

    shell_center = SOURCE_SHELL_CENTER_M
    tilt = np.radians(gear.tire_axis_tilt_deg)
    wheel_axis = np.array([0.0, np.cos(tilt), np.sin(tilt)])
    nominal_axis = np.array([0.0, 1.0, 0.0])
    for side, sign in (("left", 1.0), ("right", -1.0)):
        body_name = f"{side}_wheel_link"
        body = stage.GetPrimAtPath(f"{root}/{body_name}")
        body_center = np.asarray(
            body.GetAttribute("xformOp:translate").Get(),
            dtype=np.float64,
        )
        proxy_center = (
            body_center
            + sign * gear.tire_center_axial_offset_m * nominal_axis
        )
        shift = radial_cylinder_center_shift_m(
            proxy_center - shell_center,
            wheel_axis,
            gear.tire_outer_radius_m,
            gear.tire_half_width_m,
            shell.inner_radius_m,
            gear.wheel_nominal_clearance_m,
        )
        _translate_body_and_parent_joint(
            stage,
            root,
            body_name,
            f"{side}_wheel_joint",
            shift,
        )

    for index in (1, 2):
        body_name = f"caster_{index}_ball_link"
        body = stage.GetPrimAtPath(f"{root}/{body_name}")
        body_center = np.asarray(
            body.GetAttribute("xformOp:translate").Get(),
            dtype=np.float64,
        )
        shift = radial_ball_center_shift_m(
            body_center - shell_center,
            gear.caster_ball_radius_m,
            shell.inner_radius_m,
            gear.caster_nominal_clearance_m,
        )
        _translate_body_and_parent_joint(
            stage,
            root,
            body_name,
            f"caster_{index}_joint",
            shift,
        )


def _replace_wheel_joints_with_radial_compliance(
    stage: Any,
    root: str,
    config: WheelRadialComplianceConfig,
) -> int:
    """Insert a prismatic elastic mount upstream of each revolute wheel.

    PhysX reduced-coordinate articulations integrate the two serial one-DOF
    joints robustly: the mount translates in the tread-normal direction and
    the downstream wheel retains its ordinary motor-driven axle rotation.
    """

    if not config.enabled:
        return 0

    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    replaced = 0
    axle = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    for side in ("left", "right"):
        joint_path = f"{root}/joints/{side}_wheel_joint"
        old_joint = stage.GetPrimAtPath(joint_path)
        if not old_joint:
            raise RuntimeError(f"Missing wheel joint: {joint_path}")

        body1_targets = old_joint.GetRelationship("physics:body1").GetTargets()
        local_pos0 = old_joint.GetAttribute("physics:localPos0").Get()
        local_pos1 = old_joint.GetAttribute("physics:localPos1").Get()
        if len(body1_targets) != 1:
            raise RuntimeError(f"Wheel joint bodies are ambiguous: {joint_path}")

        wheel_body = stage.GetPrimAtPath(f"{root}/{side}_wheel_link")
        wheel_center = np.asarray(
            wheel_body.GetAttribute("xformOp:translate").Get(),
            dtype=np.float64,
        )
        shell_radial = wheel_center - SOURCE_SHELL_CENTER_M
        tread_normal = shell_radial - np.dot(shell_radial, axle) * axle
        tread_normal_norm = float(np.linalg.norm(tread_normal))
        if tread_normal_norm <= 1.0e-12:
            raise RuntimeError(f"Wheel tread normal is degenerate: {joint_path}")
        tread_normal /= tread_normal_norm

        # Both rigid-body frames in the generated CAD asset are aligned with
        # the module frame. Rotating local X onto the tread normal about the
        # axle therefore preserves local Y as the physical wheel-spin axis.
        rotation = Gf.Rotation(
            Gf.Vec3d(1.0, 0.0, 0.0),
            Gf.Vec3d(*(float(value) for value in tread_normal)),
        )
        quaternion = rotation.GetQuat()
        joint_frame = Gf.Quatf(
            float(quaternion.GetReal()),
            Gf.Vec3f(
                *(float(value) for value in quaternion.GetImaginary())
            ),
        )

        mount_path = f"{root}/{side}_wheel_radial_mount_link"
        mount = UsdGeom.Xform.Define(stage, mount_path)
        mount.AddTranslateOp().Set(
            Gf.Vec3d(*(float(value) for value in wheel_center))
        )
        UsdPhysics.RigidBodyAPI.Apply(mount.GetPrim())
        UsdPhysics.MassAPI.Apply(mount.GetPrim()).CreateMassAttr(
            float(config.mount_mass_kg)
        )

        radial_joint_path = f"{root}/joints/{side}_wheel_radial_joint"
        radial_joint = UsdPhysics.PrismaticJoint.Define(
            stage,
            radial_joint_path,
        )
        radial_joint.CreateBody0Rel().SetTargets(
            [Sdf.Path(f"{root}/internal_link")]
        )
        radial_joint.CreateBody1Rel().SetTargets([Sdf.Path(mount_path)])
        radial_joint.CreateCollisionEnabledAttr(False)
        radial_joint.CreateAxisAttr("X")
        radial_joint.CreateLocalPos0Attr(local_pos0)
        radial_joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        radial_joint.CreateLocalRot0Attr(joint_frame)
        radial_joint.CreateLocalRot1Attr(joint_frame)
        radial_joint.CreateLowerLimitAttr(-float(config.inward_travel_m))
        radial_joint.CreateUpperLimitAttr(float(config.outward_travel_m))

        radial_drive = UsdPhysics.DriveAPI.Apply(
            radial_joint.GetPrim(),
            "linear",
        )
        radial_drive.CreateTypeAttr("force")
        radial_drive.CreateTargetPositionAttr(float(config.rest_position_m))
        radial_drive.CreateStiffnessAttr(float(config.stiffness_n_per_m))
        radial_drive.CreateDampingAttr(float(config.damping_n_s_per_m))
        radial_drive.CreateMaxForceAttr(float(config.max_force_n))

        stage.RemovePrim(joint_path)
        wheel_joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        wheel_joint.CreateBody0Rel().SetTargets([Sdf.Path(mount_path)])
        wheel_joint.CreateBody1Rel().SetTargets(body1_targets)
        wheel_joint.CreateCollisionEnabledAttr(False)
        wheel_joint.CreateAxisAttr("Y")
        wheel_joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        wheel_joint.CreateLocalPos1Attr(local_pos1)
        identity = Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
        wheel_joint.CreateLocalRot0Attr(identity)
        wheel_joint.CreateLocalRot1Attr(identity)
        wheel_drive = UsdPhysics.DriveAPI.Apply(
            wheel_joint.GetPrim(),
            "angular",
        )
        wheel_drive.CreateTypeAttr("force")
        wheel_drive.CreateTargetVelocityAttr(0.0)
        wheel_drive.CreateStiffnessAttr(0.0)
        wheel_drive.CreateDampingAttr(0.0)
        wheel_drive.CreateMaxForceAttr(0.0)
        radial_joint.GetPrim().CreateAttribute(
            "freebot:radialComplianceAxis",
            Sdf.ValueTypeNames.Float3,
        ).Set(Gf.Vec3f(*(float(value) for value in tread_normal)))
        replaced += 1
    return replaced


def _fixture_shell(stage: Any, root: str) -> None:
    """Fix only the outer shell; keep the internal articulation dynamic."""

    from pxr import UsdPhysics

    shell_path = f"{root}/shell_link"
    shell = stage.GetPrimAtPath(shell_path)
    if not shell or not shell.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"Passive shell rigid body is missing: {shell_path}")
    UsdPhysics.RigidBodyAPI.Apply(shell).CreateKinematicEnabledAttr(True)


def _remove_duplicate_container_colliders(stage: Any, root: str) -> int:
    from pxr import Usd, UsdPhysics

    removed = 0
    for prim in list(Usd.PrimRange(stage.GetPrimAtPath(root))):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        has_descendant = any(
            child != prim and child.HasAPI(UsdPhysics.CollisionAPI)
            for child in Usd.PrimRange(prim)
        )
        if has_descendant:
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            removed += 1
    return removed


def _replace_wheel_colliders(
    stage: Any,
    root: str,
    geometry: RunningGearGeometry,
) -> int:
    """Replace tire mesh/SDF collisions with CAD-fitted cylinders."""

    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    removed = 0
    for prim in list(Usd.PrimRange(stage.GetPrimAtPath(root))):
        path = str(prim.GetPath())
        if (
            "wheel_link" in path
            and "tire" in path.lower()
            and prim.HasAPI(UsdPhysics.CollisionAPI)
        ):
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            removed += 1

    for side in ("left", "right"):
        proxy_path = f"{root}/{side}_wheel_link/tire_collision_proxy"
        cylinder = UsdGeom.Cylinder.Define(stage, proxy_path)
        cylinder.CreateAxisAttr(UsdGeom.Tokens.y)
        cylinder.CreateRadiusAttr(geometry.tire_collision_radius_m)
        cylinder.CreateHeightAttr(2.0 * geometry.tire_half_width_m)
        cylinder.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        outward_sign = 1.0 if side == "left" else -1.0
        cylinder.AddTranslateOp().Set(
            Gf.Vec3d(
                0.0,
                outward_sign * geometry.tire_center_axial_offset_m,
                0.0,
            )
        )
        cylinder.AddRotateXOp().Set(geometry.tire_axis_tilt_deg)
        collider = cylinder.GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider)
        collider.CreateAttribute(
            "physxCollision:contactOffset",
            Sdf.ValueTypeNames.Float,
        ).Set(float(geometry.wheel_contact_offset_m))
        collider.CreateAttribute(
            "physxCollision:restOffset",
            Sdf.ValueTypeNames.Float,
        ).Set(0.0)
    return removed


def _replace_caster_colliders(
    stage: Any,
    root: str,
    geometry: RunningGearGeometry,
) -> int:
    """Replace caster mesh/SDF collisions with CAD-fitted spheres."""

    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    removed = 0
    for caster_name in ("caster_1_ball_link", "caster_2_ball_link"):
        caster_path = f"{root}/{caster_name}"
        caster_prim = stage.GetPrimAtPath(caster_path)
        if not caster_prim:
            raise RuntimeError(f"Missing caster rigid body: {caster_path}")
        for prim in list(Usd.PrimRange(caster_prim)):
            if prim == caster_prim or not prim.HasAPI(
                UsdPhysics.CollisionAPI
            ):
                continue
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            removed += 1
        sphere = UsdGeom.Sphere.Define(
            stage,
            f"{caster_path}/caster_collision_proxy",
        )
        sphere.CreateRadiusAttr(geometry.caster_collision_radius_m)
        sphere.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        collider = sphere.GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider)
        collider.CreateAttribute(
            "physxCollision:contactOffset",
            Sdf.ValueTypeNames.Float,
        ).Set(float(geometry.caster_contact_offset_m))
        collider.CreateAttribute(
            "physxCollision:restOffset",
            Sdf.ValueTypeNames.Float,
        ).Set(0.0)
    return removed


def _create_ground(stage: Any, config: IsaacStageConfig, material: Any) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    path = "/World/freebot_ground"
    if stage.GetPrimAtPath(path):
        stage.RemovePrim(path)
    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    box.AddTranslateOp().Set(Gf.Vec3d(*config.ground_center_world))
    box.AddScaleOp().Set(Gf.Vec3f(*config.ground_size_m))
    box.CreateDisplayColorAttr([Gf.Vec3f(0.5, 0.5, 0.5)])
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(
        material,
        UsdShade.Tokens.strongerThanDescendants,
    )


def build_freebot_stage(
    stage_utils: Any,
    config: IsaacStageConfig,
) -> StageBuildResult:
    """Open one CAD stage and clone it into two identical modules."""

    usd_path = config.usd_path.expanduser().resolve()
    if not usd_path.exists():
        raise FileNotFoundError(usd_path)
    success, stage = stage_utils.open_stage(str(usd_path))
    if not success:
        raise RuntimeError(f"Could not open USD stage: {usd_path}")

    _clone_module(stage, SOURCE_ROOT, ACTIVE_ROOT)
    _clone_module(stage, SOURCE_ROOT, PASSIVE_ROOT)
    stage.RemovePrim(SOURCE_ROOT)
    removed = (
        _remove_duplicate_container_colliders(stage, ACTIVE_ROOT)
        + _remove_duplicate_container_colliders(stage, PASSIVE_ROOT)
    )
    _align_running_gear(
        stage,
        ACTIVE_ROOT,
        config.shell_geometry,
        config.running_gear,
    )
    _align_running_gear(
        stage,
        PASSIVE_ROOT,
        config.shell_geometry,
        config.running_gear,
    )
    radial_suspension_joints = (
        _replace_wheel_joints_with_radial_compliance(
            stage,
            ACTIVE_ROOT,
            config.wheel_radial_compliance,
        )
        + _replace_wheel_joints_with_radial_compliance(
            stage,
            PASSIVE_ROOT,
            config.wheel_radial_compliance,
        )
    )
    replaced_wheels = (
        _replace_wheel_colliders(stage, ACTIVE_ROOT, config.running_gear)
        + _replace_wheel_colliders(stage, PASSIVE_ROOT, config.running_gear)
    )
    replaced_casters = (
        _replace_caster_colliders(stage, ACTIVE_ROOT, config.running_gear)
        + _replace_caster_colliders(stage, PASSIVE_ROOT, config.running_gear)
    )
    _set_module_pose(
        stage,
        ACTIVE_ROOT,
        config.active_shell_center_world,
        config.active_y_rotation_deg,
    )
    _set_module_pose(
        stage,
        PASSIVE_ROOT,
        config.passive_shell_center_world,
        config.passive_y_rotation_deg,
    )
    _set_rigid_body_mode(stage, ACTIVE_ROOT, fixed=False)
    _set_rigid_body_mode(stage, PASSIVE_ROOT, fixed=False)
    _set_module_masses(
        stage,
        ACTIVE_ROOT,
        config.masses,
        config.wheel_radial_compliance,
    )
    _set_module_masses(
        stage,
        PASSIVE_ROOT,
        config.masses,
        config.wheel_radial_compliance,
    )
    _set_module_mass_properties(
        stage,
        ACTIVE_ROOT,
        config.masses,
        config.shell_geometry,
        config.running_gear,
        config.wheel_radial_compliance,
    )
    _set_module_mass_properties(
        stage,
        PASSIVE_ROOT,
        config.masses,
        config.shell_geometry,
        config.running_gear,
        config.wheel_radial_compliance,
    )
    if config.passive_fixed:
        _fixture_shell(stage, PASSIVE_ROOT)
    ground_material = assign_freebot_materials(
        stage,
        (ACTIVE_ROOT, PASSIVE_ROOT),
        config.materials,
    )
    _create_ground(stage, config, ground_material)

    return StageBuildResult(
        stage=stage,
        active_root=ACTIVE_ROOT,
        passive_root=PASSIVE_ROOT,
        removed_duplicate_colliders=removed,
        replaced_wheel_colliders=replaced_wheels,
        replaced_caster_colliders=replaced_casters,
        radial_suspension_joints=radial_suspension_joints,
    )
