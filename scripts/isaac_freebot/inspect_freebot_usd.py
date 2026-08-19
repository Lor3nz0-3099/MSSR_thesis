from pathlib import Path
import sys

from pxr import Usd, UsdGeom


DEFAULT_USD = Path("assets/freebot/usd_visual/freebot_visual_shorter_casters.usd")
KEYWORDS = (
    "shell",
    "chassis",
    "base",
    "wheel",
    "ruota",
    "ball",
    "caster",
    "mag",
    "m057",
    "engrenagem",
    "telaio",
)


def bounds_for(stage, prim):
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    if box.IsEmpty():
        return None
    size = box.GetSize()
    center = (box.GetMin() + box.GetMax()) * 0.5
    return size, center


def main():
    usd_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_USD
    usd_path = usd_path.resolve()
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {usd_path}")

    print(f"USD: {usd_path}")
    print(f"metersPerUnit: {UsdGeom.GetStageMetersPerUnit(stage)}")
    print(f"upAxis: {UsdGeom.GetStageUpAxis(stage)}")
    print()

    rows = []
    for prim in stage.Traverse():
        name = prim.GetName()
        low = name.lower()
        if not any(keyword in low for keyword in KEYWORDS):
            continue
        result = bounds_for(stage, prim)
        if result is None:
            continue
        size, center = result
        rows.append((str(prim.GetPath()), prim.GetTypeName(), name, size, center))

    for path, type_name, name, size, center in rows:
        print(path)
        print(f"  type: {type_name}")
        print(f"  name: {name}")
        print(f"  size_m:   {size[0]:.6f}, {size[1]:.6f}, {size[2]:.6f}")
        print(f"  center_m: {center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}")
        print(f"  max_cm: {max(size) * 100.0:.2f}")


if __name__ == "__main__":
    main()
