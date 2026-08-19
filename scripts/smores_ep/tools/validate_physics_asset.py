from __future__ import annotations

import argparse
import math
from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _position(stage: object, path: str) -> tuple[float, float, float]:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(path)
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    translation = matrix.ExtractTranslation()
    return tuple(float(value) for value in translation)


def main() -> None:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description="Load and step the generated SMORES-EP articulation"
    )
    parser.add_argument(
        "--usd",
        type=Path,
        default=(
            root
            / "assets/smores-ep/usd_physics/smores_ep_physics_v1.usd"
        ),
    )
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument(
        "--zero-gravity",
        action="store_true",
        help="Isolate the articulation drives from gravity and contacts",
    )
    args = parser.parse_args()

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import isaacsim.core.experimental.utils.app as app_utils
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.experimental.prims import Articulation
        from pxr import Gf, UsdGeom, UsdPhysics

        success, stage = stage_utils.open_stage(str(args.usd.resolve()))
        if not success:
            raise RuntimeError(f"Could not open {args.usd}")

        scene = UsdPhysics.Scene.Define(stage, "/World/physicsScene")
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr(0.0 if args.zero_gravity else 9.81)

        if not args.zero_gravity:
            ground = UsdGeom.Cube.Define(stage, "/World/validation_ground")
            ground.CreateSizeAttr(1.0)
            ground_xform = UsdGeom.Xformable(ground)
            ground_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.01))
            ground_xform.AddScaleOp().Set(Gf.Vec3f(1.5, 1.5, 0.02))
            UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

        module = stage.GetPrimAtPath("/World/smores_ep")
        UsdGeom.Xformable(module).AddTranslateOp().Set(
            Gf.Vec3d(0.0, 0.0, 0.08)
        )
        tilt_drive = UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath("/World/smores_ep/joints/tilt_joint"),
            "angular",
        )
        pan_drive = UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath("/World/smores_ep/joints/pan_joint"),
            "angular",
        )
        tilt_drive.GetTargetPositionAttr().Set(20.0)
        pan_drive.GetTargetPositionAttr().Set(30.0)

        SimulationManager.set_physics_dt(1.0 / 240.0)
        app_utils.play()
        app.update()
        articulation = Articulation("/World/smores_ep")
        tilt_index = articulation.dof_paths[0].index(
            "/World/smores_ep/joints/tilt_joint"
        )
        pan_index = articulation.dof_paths[0].index(
            "/World/smores_ep/joints/pan_joint"
        )
        for _ in range(args.steps):
            SimulationManager.step()
            app.update()

        positions = {
            name: _position(stage, f"/World/smores_ep/{name}")
            for name in (
                "body_link",
                "left_wheel_link",
                "right_wheel_link",
                "tilt_link",
                "pan_link",
            )
        }
        if not all(
            math.isfinite(value)
            for position in positions.values()
            for value in position
        ):
            raise RuntimeError("PhysX produced a non-finite link pose")
        for name, position in positions.items():
            formatted = ", ".join(f"{value:+.6f}" for value in position)
            print(f"{name}: ({formatted}) m")
        tilt_position = float(
            articulation.get_dof_positions(
                dof_indices=tilt_index
            ).numpy().item()
        )
        pan_position = float(
            articulation.get_dof_positions(
                dof_indices=pan_index
            ).numpy().item()
        )
        print(
            f"joint_positions: tilt={tilt_position:+.6f}rad "
            f"pan={pan_position:+.6f}rad"
        )
        print(f"Validated {args.steps} PhysX steps")
        app_utils.stop()
    finally:
        app.close()


if __name__ == "__main__":
    main()
