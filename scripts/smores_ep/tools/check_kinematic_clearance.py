from __future__ import annotations

import argparse
import math
from pathlib import Path

from pxr import Usd, UsdGeom

from smores_ep.config.geometry import SmoresGeometry
from smores_ep.control.differential_drive import PlanarPose
from smores_ep.isaac.kinematic_stage import (
    MODULE_ROOT,
    KinematicStageResult,
    build_kinematic_stage,
)


class _InMemoryStageUtils:
    @staticmethod
    def create_new_stage(template: str) -> Usd.Stage:
        del template
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/World")
        return stage


def _minimum_world_z(stage: Usd.Stage, prim_path: str) -> float:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise RuntimeError(f"Missing prim: {prim_path}")
    minimum_z = float("inf")
    for descendant in Usd.PrimRange(
        prim,
        Usd.TraverseInstanceProxies(),
    ):
        if not descendant.IsA(UsdGeom.Mesh):
            continue
        if (
            UsdGeom.Imageable(descendant).ComputeVisibility()
            == UsdGeom.Tokens.invisible
        ):
            continue
        mesh = UsdGeom.Mesh(descendant)
        transform = UsdGeom.Xformable(
            descendant
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for point in mesh.GetPointsAttr().Get() or ():
            minimum_z = min(
                minimum_z,
                float(transform.Transform(point)[2]),
            )
    if not math.isfinite(minimum_z):
        raise RuntimeError(f"No visible mesh below {prim_path}")
    return minimum_z


def _check_mechanism_mapping(
    built: KinematicStageResult,
    geometry: SmoresGeometry,
) -> None:
    model = built.model
    left_wheel_rad = 0.25
    right_wheel_rad = -0.50
    pan_rad = 0.30
    tilt_rad = 0.20
    model.set_state(
        pose=PlanarPose(),
        spawn_height_m=geometry.ground_contact_height_m(tilt_rad),
        left_wheel_angle_rad=left_wheel_rad,
        right_wheel_angle_rad=right_wheel_rad,
        pan_angle_rad=pan_rad,
        tilt_angle_rad=tilt_rad,
    )
    degrees = 180.0 / math.pi
    expected_degrees = {
        "left wheel/outer gear": -left_wheel_rad * degrees,
        "right wheel/outer gear": -right_wheel_rad * degrees,
        "outer left pinion": (
            geometry.spur_to_pinion_ratio * left_wheel_rad * degrees
        ),
        "outer right pinion": (
            geometry.spur_to_pinion_ratio * right_wheel_rad * degrees
        ),
        "inner left gear": (tilt_rad + pan_rad) * degrees,
        "inner right gear": (tilt_rad - pan_rad) * degrees,
        "inner left pinion": (
            -geometry.spur_to_pinion_ratio
            * (tilt_rad + pan_rad)
            * degrees
        ),
        "inner right pinion": (
            -geometry.spur_to_pinion_ratio
            * (tilt_rad - pan_rad)
            * degrees
        ),
        "positive tilt raises TOP": -tilt_rad * degrees,
        "pan": pan_rad * degrees,
    }
    actual_degrees = {
        "left wheel/outer gear": float(model.left_wheel_op.Get()),
        "right wheel/outer gear": float(model.right_wheel_op.Get()),
        "outer left pinion": float(model.outer_left_pinion_op.Get()),
        "outer right pinion": float(model.outer_right_pinion_op.Get()),
        "inner left gear": float(model.inner_left_gear_op.Get()),
        "inner right gear": float(model.inner_right_gear_op.Get()),
        "inner left pinion": float(model.inner_left_pinion_op.Get()),
        "inner right pinion": float(model.inner_right_pinion_op.Get()),
        "positive tilt raises TOP": float(model.tilt_op.Get()),
        "pan": float(model.pan_op.Get()),
    }
    for name, expected in expected_degrees.items():
        actual = actual_degrees[name]
        if not math.isclose(actual, expected, abs_tol=1.0e-5):
            raise RuntimeError(
                f"{name}: expected {expected:.6f} deg, got {actual:.6f} deg"
            )
    print("mechanism_mapping=OK ratio=48:9")


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Measure wheel and TOP-face clearance in the kinematic stage"
    )
    parser.add_argument(
        "--visual-usd",
        type=Path,
        default=(
            root
            / "assets/smores-ep/usd_visual/smores_ep_usd_visual_v1.usd"
        ),
    )
    args = parser.parse_args()

    geometry = SmoresGeometry()
    built = build_kinematic_stage(
        _InMemoryStageUtils,
        args.visual_usd,
        geometry,
    )
    paths = {
        "left_wheel": f"{MODULE_ROOT}/left_wheel_link",
        "right_wheel": f"{MODULE_ROOT}/right_wheel_link",
        "top_face": f"{MODULE_ROOT}/tilt_link/pan_link",
    }
    for tilt_deg in (-45.0, 0.0, 45.0):
        tilt_rad = math.radians(tilt_deg)
        built.model.set_state(
            pose=PlanarPose(),
            spawn_height_m=geometry.ground_contact_height_m(tilt_rad),
            left_wheel_angle_rad=0.0,
            right_wheel_angle_rad=0.0,
            pan_angle_rad=0.0,
            tilt_angle_rad=tilt_rad,
        )
        minimums = {
            name: _minimum_world_z(built.stage, path)
            for name, path in paths.items()
        }
        print(
            f"tilt={tilt_deg:+05.1f} deg "
            f"height={geometry.ground_contact_height_m(tilt_rad):.6f} m "
            + " ".join(
                f"{name}_min_z={minimum_z:+.6f} m"
                for name, minimum_z in minimums.items()
            )
        )
    _check_mechanism_mapping(built, geometry)


if __name__ == "__main__":
    main()
