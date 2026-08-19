from pathlib import Path
import sys

from pxr import Gf, Usd, UsdGeom


DEFAULT_USD = Path("assets/freebot/usd_visual/freebot_visual_shorter_casters.usd")
KEYWORDS = ("shell", "sphere", "sfera", "ball", "wheel", "ruota", "base", "chassis", "magnet", "magnete")


def format_vec(vec):
    return f"({vec[0]:.6f}, {vec[1]:.6f}, {vec[2]:.6f}) m"


def compute_bounds(stage, prim):
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    bbox = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    if bbox.IsEmpty():
        return None
    size = bbox.GetSize()
    center = (bbox.GetMin() + bbox.GetMax()) * 0.5
    return size, center


def main():
    usd_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_USD
    usd_path = usd_path.resolve()

    if not usd_path.exists():
        raise FileNotFoundError(f"USD file not found: {usd_path}")

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {usd_path}")

    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    up_axis = UsdGeom.GetStageUpAxis(stage)
    pseudo_root = stage.GetPseudoRoot()

    print(f"USD: {usd_path}")
    print(f"metersPerUnit: {meters_per_unit}")
    print(f"upAxis: {up_axis}")
    print()

    world_bounds = compute_bounds(stage, pseudo_root)
    if world_bounds:
        size, center = world_bounds
        max_dim = max(size)
        print("GLOBAL BOUNDING BOX")
        print(f"  size:   {format_vec(size)}")
        print(f"  center: {format_vec(center)}")
        print(f"  max dimension: {max_dim:.6f} m = {max_dim * 100.0:.2f} cm")
        print()

    print("CANDIDATE PARTS")
    candidates = []
    for prim in stage.Traverse():
        name = prim.GetName().lower()
        if any(keyword in name for keyword in KEYWORDS):
            bounds = compute_bounds(stage, prim)
            if bounds is None:
                continue
            size, center = bounds
            candidates.append((max(size), str(prim.GetPath()), size, center))

    for _, path, size, center in sorted(candidates, reverse=True):
        print(f"{path}")
        print(f"  size:   {format_vec(size)}")
        print(f"  center: {format_vec(center)}")
        print(f"  max dimension: {max(size) * 100.0:.2f} cm")


if __name__ == "__main__":
    main()
