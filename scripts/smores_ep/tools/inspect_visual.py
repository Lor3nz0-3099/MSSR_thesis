from __future__ import annotations

import argparse
from pathlib import Path

from pxr import Gf, Usd, UsdGeom


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _exact_mesh_bound(prim: Usd.Prim) -> tuple[Gf.Vec3d, Gf.Vec3d]:
    """Transform mesh vertices, avoiding rotated-AABB instance artifacts."""

    minimum = [float("inf")] * 3
    maximum = [float("-inf")] * 3
    found_mesh = False
    for descendant in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not descendant.IsA(UsdGeom.Mesh):
            continue
        found_mesh = True
        mesh = UsdGeom.Mesh(descendant)
        transform = UsdGeom.Xformable(
            descendant
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        for point in mesh.GetPointsAttr().Get() or ():
            world_point = transform.Transform(point)
            for axis in range(3):
                value = float(world_point[axis])
                minimum[axis] = min(minimum[axis], value)
                maximum[axis] = max(maximum[axis], value)
    if not found_mesh:
        raise RuntimeError(f"No mesh below {prim.GetPath()}")
    return Gf.Vec3d(*minimum), Gf.Vec3d(*maximum)


def main() -> None:
    root = repository_root()
    parser = argparse.ArgumentParser(
        description="Inspect SMORES-EP CAD hierarchy and part bounds"
    )
    parser.add_argument(
        "usd",
        type=Path,
        nargs="?",
        default=(
            root
            / "assets/smores-ep/usd_visual/smores_ep_usd_visual_v1.usd"
        ),
    )
    args = parser.parse_args()

    stage = Usd.Stage.Open(str(args.usd.resolve()))
    if stage is None:
        raise RuntimeError(f"Could not open {args.usd}")
    assembly = stage.GetPrimAtPath(
        "/World/SMORES_EP_modulev1/tn__SMORESEP_dC"
    )
    if not assembly:
        raise RuntimeError("Expected SMORES-EP assembly prim is missing")

    print(f"USD: {args.usd.resolve()}")
    print(f"metersPerUnit: {UsdGeom.GetStageMetersPerUnit(stage)}")
    print(f"upAxis: {UsdGeom.GetStageUpAxis(stage)}")
    for prim in assembly.GetChildren():
        minimum, maximum = _exact_mesh_bound(prim)
        size = maximum - minimum
        center = (minimum + maximum) * 0.5
        print(prim.GetName())
        print(
            "  size_m: "
            + ", ".join(f"{float(value):.6f}" for value in size)
        )
        print(
            "  center_m: "
            + ", ".join(f"{float(value):.6f}" for value in center)
        )


if __name__ == "__main__":
    main()
