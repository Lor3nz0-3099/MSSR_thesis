from pathlib import Path
import sys

from pxr import Gf, Usd, UsdGeom


DEFAULT_USD = Path("assets/freebot/usd_visual/freebot_visual_shorter_casters.usd")
TIRE_PROTO_NAME = "tn__PololuWheel32x7mmtire_____________1_ya0rECJ"
WHEEL_TIRE_NAMES = (
    "tn__PololuWheel32x7mm1_tQCi8/tn__PololuWheel32x7mmtire_____________1_ya0rECJ",
    "tn__PololuWheel32x7mm2_tQCi8/tn__PololuWheel32x7mmtire_____________1_ya0rECJ",
)


def vec_size(points):
    mins = [min(p[i] for p in points) for i in range(3)]
    maxs = [max(p[i] for p in points) for i in range(3)]
    return [maxs[i] - mins[i] for i in range(3)]


def format_mm(values):
    return ", ".join(f"{v * 1000.0:.3f}" for v in values)


def format_source_units(values):
    return ", ".join(f"{v:.3f}" for v in values)


def main():
    usd_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_USD
    stage = Usd.Stage.Open(str(usd_path.resolve()))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {usd_path}")

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    proto_root = next(
        (
            prim
            for prim in stage.TraverseAll()
            if prim.GetName() == TIRE_PROTO_NAME and "/Prototypes/" in str(prim.GetPath())
        ),
        None,
    )
    proto_mesh = stage.GetPrimAtPath(f"{proto_root.GetPath()}/Solid1/Mesh") if proto_root else None
    if not proto_root or not proto_mesh:
        raise RuntimeError("Could not find the Pololu tire prototype mesh.")

    proto_inv = cache.GetLocalToWorldTransform(proto_root).GetInverse()
    mesh_to_proto = cache.GetLocalToWorldTransform(proto_mesh) * proto_inv
    proto_points = [
        mesh_to_proto.Transform(Gf.Vec3d(point))
        for point in proto_mesh.GetAttribute("points").Get()
    ]

    print(f"USD: {usd_path.resolve()}")
    print("Prototype tire exact local size [X, Y, Z] in CAD source units:")
    print(f"  {format_source_units(vec_size(proto_points))}")
    print("  Expected: X is tire thickness, Y/Z are tire diameter.")
    print()

    stage_root = stage.GetPseudoRoot()
    instance_paths = []
    for tire_name in WHEEL_TIRE_NAMES:
        matches = [prim for prim in stage.Traverse() if str(prim.GetPath()).endswith(tire_name)]
        if not matches:
            raise RuntimeError(f"Could not find tire instance ending with: {tire_name}")
        instance_paths.append(str(matches[0].GetPath()))

    for path in instance_paths:
        prim = stage.GetPrimAtPath(path)
        inst_world = cache.GetLocalToWorldTransform(prim)
        world_points = [inst_world.Transform(point) for point in proto_points]
        world_size = vec_size(world_points)

        x_axis = Gf.Vec3d(inst_world[0][0], inst_world[0][1], inst_world[0][2]).GetNormalized()
        print(path)
        print(f"  exact transformed tire bbox [X, Y, Z] mm: {format_mm(world_size)}")
        print(
            "  tire spin axis, local X in world: "
            f"({x_axis[0]:+.4f}, {x_axis[1]:+.4f}, {x_axis[2]:+.4f})"
        )
        print()


if __name__ == "__main__":
    main()
