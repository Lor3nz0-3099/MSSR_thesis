from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable

from smores_ep.config.geometry import SmoresGeometry, Vector3
from smores_ep.control.differential_drive import PlanarPose


MODULE_ROOT = "/World/SmoresEP"


def _source_to_link_transform(
    source_link_origin_m: Vector3,
) -> Any:
    """Return p_link = Rz(-90 deg) * (p_source - source_link_origin)."""

    from pxr import Gf

    translate = Gf.Matrix4d(1.0)
    translate.SetTranslate(
        Gf.Vec3d(
            -source_link_origin_m[0],
            -source_link_origin_m[1],
            -source_link_origin_m[2],
        )
    )
    rotate = Gf.Matrix4d(1.0)
    rotate.SetRotate(
        Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), -90.0)
    )
    return translate * rotate


def _add_filtered_visual(
    stage: Any,
    link_path: str,
    visual_usd: Path,
    source_link_origin_m: Vector3,
    visible_parts: Iterable[str],
    geometry: SmoresGeometry,
) -> None:
    from pxr import UsdGeom

    visible = set(visible_parts)
    offset_path = f"{link_path}/visual_offset"
    offset = UsdGeom.Xform.Define(stage, offset_path)
    offset.AddTransformOp().Set(
        _source_to_link_transform(source_link_origin_m)
    )
    reference_path = f"{offset_path}/cad"
    reference = UsdGeom.Xform.Define(stage, reference_path)
    reference.GetPrim().GetReferences().AddReference(
        str(visual_usd),
        geometry.visual_root_path,
    )
    stage.Load(reference.GetPath())

    missing = []
    for part_name in geometry.all_parts:
        part_path = (
            f"{reference_path}/{geometry.assembly_path}/{part_name}"
        )
        prim = stage.GetPrimAtPath(part_path)
        if not prim:
            missing.append(part_path)
            continue
        visibility = (
            UsdGeom.Tokens.inherited
            if part_name in visible
            else UsdGeom.Tokens.invisible
        )
        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(visibility)
    if missing:
        joined = "\n  ".join(missing)
        raise RuntimeError(f"Visual CAD prims are missing:\n  {joined}")


def _add_ground(stage: Any) -> None:
    from pxr import Gf, UsdGeom

    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(1.0)
    xform = UsdGeom.Xformable(ground)
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.005))
    xform.AddScaleOp().Set(Gf.Vec3f(4.0, 4.0, 0.01))
    ground.CreateDisplayColorAttr(
        [Gf.Vec3f(0.18, 0.20, 0.22)]
    )


@dataclass
class KinematicModel:
    root_translate_op: Any
    root_yaw_op: Any
    left_wheel_op: Any
    right_wheel_op: Any
    outer_left_pinion_op: Any
    outer_right_pinion_op: Any
    inner_left_gear_op: Any
    inner_right_gear_op: Any
    inner_left_pinion_op: Any
    inner_right_pinion_op: Any
    tilt_op: Any
    pan_op: Any
    spur_to_pinion_ratio: float

    def set_state(
        self,
        pose: PlanarPose,
        spawn_height_m: float,
        left_wheel_angle_rad: float,
        right_wheel_angle_rad: float,
        pan_angle_rad: float,
        tilt_angle_rad: float,
    ) -> None:
        from pxr import Gf

        degrees = 180.0 / math.pi
        self.root_translate_op.Set(
            Gf.Vec3d(pose.x_m, pose.y_m, spawn_height_m)
        )
        self.root_yaw_op.Set(pose.yaw_rad * degrees)
        # Both CAD wheel axes point along body +Y. A negative visual rotation
        # produces positive forward rolling at the ground contact.
        self.left_wheel_op.Set(-left_wheel_angle_rad * degrees)
        self.right_wheel_op.Set(-right_wheel_angle_rad * degrees)
        self.outer_left_pinion_op.Set(
            self.spur_to_pinion_ratio * left_wheel_angle_rad * degrees
        )
        self.outer_right_pinion_op.Set(
            self.spur_to_pinion_ratio * right_wheel_angle_rad * degrees
        )

        # Same-direction inner spur motion produces tilt; opposite-direction
        # motion produces pan. Pinions rotate oppositely by the 48:9 ratio.
        inner_left_angle = tilt_angle_rad + pan_angle_rad
        inner_right_angle = tilt_angle_rad - pan_angle_rad
        self.inner_left_gear_op.Set(inner_left_angle * degrees)
        self.inner_right_gear_op.Set(inner_right_angle * degrees)
        self.inner_left_pinion_op.Set(
            -self.spur_to_pinion_ratio * inner_left_angle * degrees
        )
        self.inner_right_pinion_op.Set(
            -self.spur_to_pinion_ratio * inner_right_angle * degrees
        )

        # User-positive tilt raises the TOP face. A positive +Y USD rotation
        # would lower it, hence the deliberate sign inversion.
        self.tilt_op.Set(-tilt_angle_rad * degrees)
        self.pan_op.Set(pan_angle_rad * degrees)


@dataclass(frozen=True)
class KinematicStageResult:
    stage: Any
    model: KinematicModel


def _add_rotating_mechanism_part(
    stage: Any,
    root_path: str,
    name: str,
    source_center_m: Vector3,
    visible_parts: Iterable[str],
    visual_path: Path,
    geometry: SmoresGeometry,
) -> Any:
    from pxr import Gf, UsdGeom

    path = f"{root_path}/{name}"
    link = UsdGeom.Xform.Define(stage, path)
    link.AddTranslateOp().Set(
        Gf.Vec3d(*geometry.source_point_to_body(source_center_m))
    )
    rotate = link.AddRotateYOp()
    _add_filtered_visual(
        stage,
        path,
        visual_path,
        source_center_m,
        visible_parts,
        geometry,
    )
    return rotate


def build_kinematic_stage(
    stage_utils: Any,
    visual_usd: Path,
    geometry: SmoresGeometry,
) -> KinematicStageResult:
    """Build a clean four-link kinematic representation from the CAD visual."""

    from pxr import Gf, UsdGeom

    visual_path = visual_usd.resolve()
    if not visual_path.is_file():
        raise FileNotFoundError(f"Visual USD does not exist: {visual_path}")

    stage = stage_utils.create_new_stage(template="sunlight")
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    if not stage.GetPrimAtPath("/World"):
        UsdGeom.Xform.Define(stage, "/World")
    _add_ground(stage)

    root = UsdGeom.Xform.Define(stage, MODULE_ROOT)
    root_translate = root.AddTranslateOp()
    root_yaw = root.AddRotateZOp()

    body_path = f"{MODULE_ROOT}/body_link"
    UsdGeom.Xform.Define(stage, body_path)
    _add_filtered_visual(
        stage,
        body_path,
        visual_path,
        geometry.source_body_origin_m,
        geometry.fixed_parts,
        geometry,
    )

    outer_left_pinion_rotate = _add_rotating_mechanism_part(
        stage,
        MODULE_ROOT,
        "outer_left_pinion",
        geometry.source_outer_left_pinion_center_m,
        geometry.outer_left_pinion_parts,
        visual_path,
        geometry,
    )
    outer_right_pinion_rotate = _add_rotating_mechanism_part(
        stage,
        MODULE_ROOT,
        "outer_right_pinion",
        geometry.source_outer_right_pinion_center_m,
        geometry.outer_right_pinion_parts,
        visual_path,
        geometry,
    )
    inner_left_gear_rotate = _add_rotating_mechanism_part(
        stage,
        MODULE_ROOT,
        "inner_left_gear",
        geometry.source_inner_left_gear_center_m,
        geometry.inner_left_gear_parts,
        visual_path,
        geometry,
    )
    inner_right_gear_rotate = _add_rotating_mechanism_part(
        stage,
        MODULE_ROOT,
        "inner_right_gear",
        geometry.source_inner_right_gear_center_m,
        geometry.inner_right_gear_parts,
        visual_path,
        geometry,
    )
    inner_left_pinion_rotate = _add_rotating_mechanism_part(
        stage,
        MODULE_ROOT,
        "inner_left_pinion",
        geometry.source_inner_left_pinion_center_m,
        geometry.inner_left_pinion_parts,
        visual_path,
        geometry,
    )
    inner_right_pinion_rotate = _add_rotating_mechanism_part(
        stage,
        MODULE_ROOT,
        "inner_right_pinion",
        geometry.source_inner_right_pinion_center_m,
        geometry.inner_right_pinion_parts,
        visual_path,
        geometry,
    )

    left_path = f"{MODULE_ROOT}/left_wheel_link"
    left = UsdGeom.Xform.Define(stage, left_path)
    left.AddTranslateOp().Set(
        Gf.Vec3d(*geometry.left_wheel_center_body_m)
    )
    left_rotate = left.AddRotateYOp()
    _add_filtered_visual(
        stage,
        left_path,
        visual_path,
        geometry.source_left_wheel_center_m,
        geometry.left_wheel_parts,
        geometry,
    )

    right_path = f"{MODULE_ROOT}/right_wheel_link"
    right = UsdGeom.Xform.Define(stage, right_path)
    right.AddTranslateOp().Set(
        Gf.Vec3d(*geometry.right_wheel_center_body_m)
    )
    right_rotate = right.AddRotateYOp()
    _add_filtered_visual(
        stage,
        right_path,
        visual_path,
        geometry.source_right_wheel_center_m,
        geometry.right_wheel_parts,
        geometry,
    )

    tilt_path = f"{MODULE_ROOT}/tilt_link"
    tilt = UsdGeom.Xform.Define(stage, tilt_path)
    tilt_rotate = tilt.AddRotateYOp()
    _add_filtered_visual(
        stage,
        tilt_path,
        visual_path,
        geometry.source_body_origin_m,
        geometry.tilt_parts,
        geometry,
    )

    pan_path = f"{tilt_path}/pan_link"
    pan = UsdGeom.Xform.Define(stage, pan_path)
    pan.AddTranslateOp().Set(Gf.Vec3d(*geometry.pan_center_body_m))
    pan_rotate = pan.AddRotateXOp()
    _add_filtered_visual(
        stage,
        pan_path,
        visual_path,
        geometry.source_pan_center_m,
        geometry.pan_parts,
        geometry,
    )

    model = KinematicModel(
        root_translate_op=root_translate,
        root_yaw_op=root_yaw,
        left_wheel_op=left_rotate,
        right_wheel_op=right_rotate,
        outer_left_pinion_op=outer_left_pinion_rotate,
        outer_right_pinion_op=outer_right_pinion_rotate,
        inner_left_gear_op=inner_left_gear_rotate,
        inner_right_gear_op=inner_right_gear_rotate,
        inner_left_pinion_op=inner_left_pinion_rotate,
        inner_right_pinion_op=inner_right_pinion_rotate,
        tilt_op=tilt_rotate,
        pan_op=pan_rotate,
        spur_to_pinion_ratio=geometry.spur_to_pinion_ratio,
    )
    model.set_state(
        pose=PlanarPose(),
        spawn_height_m=geometry.ground_contact_height_m(0.0),
        left_wheel_angle_rad=0.0,
        right_wheel_angle_rad=0.0,
        pan_angle_rad=0.0,
        tilt_angle_rad=0.0,
    )
    stage.SetDefaultPrim(root.GetPrim())
    return KinematicStageResult(stage=stage, model=model)
