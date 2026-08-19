"""Fit FreeBOT collision primitives to transformed CAD vertices.

This intentionally does not use axis-aligned bounding boxes.  Spheres are fit
by linear least squares and tires by PCA plus a least-squares circle fit in the
plane perpendicular to the spin axis.
"""

from pathlib import Path
import sys

import numpy as np
from pxr import Gf, Usd, UsdGeom


DEFAULT_USD = Path("assets/freebot/usd_visual/freebot_visual_nearer_wheels.usd")
PART_SUFFIXES = {
    "shell_lower": "tn__shell_loweript1_YNX6",
    "shell_upper": "tn__shell_upper1_qN",
    "left_tire": "tn__PololuWheel32x7mm1_tQCi8/tn__PololuWheel32x7mmtire_____________1_ya0rECJ",
    "right_tire": "tn__PololuWheel32x7mm2_tQCi8/tn__PololuWheel32x7mmtire_____________1_ya0rECJ",
    "caster_1": "tn__ball1_gA",
    "caster_2": "tn__ball2_gA",
}


def instance_points(stage, prim):
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    points = []

    def visit(candidate):
        if candidate.IsInstance():
            prototype = candidate.GetPrototype()
            instance_world = cache.GetLocalToWorldTransform(candidate)
            prototype_inv = cache.GetLocalToWorldTransform(prototype).GetInverse()
            for prototype_prim in Usd.PrimRange(prototype):
                if not prototype_prim.IsA(UsdGeom.Mesh):
                    continue
                mesh_to_prototype = cache.GetLocalToWorldTransform(prototype_prim) * prototype_inv
                for point in UsdGeom.Mesh(prototype_prim).GetPointsAttr().Get() or []:
                    points.append(instance_world.Transform(mesh_to_prototype.Transform(Gf.Vec3d(point))))
            return
        if candidate.IsA(UsdGeom.Mesh):
            transform = cache.GetLocalToWorldTransform(candidate)
            for point in UsdGeom.Mesh(candidate).GetPointsAttr().Get() or []:
                points.append(transform.Transform(Gf.Vec3d(point)))
        for child in candidate.GetChildren():
            visit(child)

    visit(prim)
    return np.asarray(points, dtype=np.float64)


def fit_sphere(points):
    matrix = np.column_stack((2.0 * points, np.ones(len(points))))
    rhs = np.einsum("ij,ij->i", points, points)
    solution, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    center = solution[:3]
    radii = np.linalg.norm(points - center, axis=1)
    return center, radii


def fit_cylinder(points):
    centroid = np.mean(points, axis=0)
    _, _, vectors = np.linalg.svd(points - centroid, full_matrices=False)
    axis = vectors[-1]
    basis_u, basis_v = vectors[0], vectors[1]
    axial = (points - centroid) @ axis
    planar = np.column_stack(((points - centroid) @ basis_u, (points - centroid) @ basis_v))
    matrix = np.column_stack((2.0 * planar, np.ones(len(planar))))
    rhs = np.einsum("ij,ij->i", planar, planar)
    solution, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    planar_center = solution[:2]
    center = centroid + planar_center[0] * basis_u + planar_center[1] * basis_v
    radii = np.linalg.norm(planar - planar_center, axis=1)
    center += 0.5 * (np.min(axial) + np.max(axial)) * axis
    return center, axis, radii, 0.5 * (np.max(axial) - np.min(axial))


def fmt(vector):
    return "(" + ", ".join(f"{1e3 * value:+.6f}" for value in vector) + ")mm"


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_USD
    stage = Usd.Stage.Open(str(path.resolve()))
    if stage is None:
        raise RuntimeError(f"Could not open {path}")

    matches = {}
    for label, suffix in PART_SUFFIXES.items():
        candidates = [prim for prim in stage.Traverse() if str(prim.GetPath()).endswith(suffix)]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one {label} matching {suffix}, got {len(candidates)}")
        matches[label] = instance_points(stage, candidates[0])

    shell_points = np.concatenate((matches["shell_lower"], matches["shell_upper"]))
    shell_center, shell_radii = fit_sphere(shell_points)
    rounded = np.round(shell_radii, decimals=7)
    values, counts = np.unique(rounded, return_counts=True)
    print(f"USD={path.resolve()}")
    print(f"shell fit center={fmt(shell_center)} vertices={len(shell_points)}")
    print("shell radial surfaces=" + ", ".join(
        f"{1e3 * value:.6f}mm({count})" for value, count in zip(values, counts) if count >= 20
    ))

    for label in ("left_tire", "right_tire"):
        center, axis, radii, half_width = fit_cylinder(matches[label])
        print(
            f"{label} center={fmt(center)} axis=({axis[0]:+.7f},{axis[1]:+.7f},{axis[2]:+.7f}) "
            f"radius_p50/max={1e3*np.median(radii):.6f}/{1e3*np.max(radii):.6f}mm "
            f"half_width={1e3*half_width:.6f}mm vertices={len(radii)}"
        )

    for label in ("caster_1", "caster_2"):
        center, radii = fit_sphere(matches[label])
        print(
            f"{label} center={fmt(center)} radius_p50/p95/max="
            f"{1e3*np.median(radii):.6f}/{1e3*np.percentile(radii,95):.6f}/{1e3*np.max(radii):.6f}mm "
            f"fit_rms={1e3*np.sqrt(np.mean((radii-np.mean(radii))**2)):.6f}mm vertices={len(radii)}"
        )


if __name__ == "__main__":
    main()
