from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt


def cylinder_inside_sphere_clearance_m(
    center_from_sphere_m: npt.ArrayLike,
    cylinder_axis_world: npt.ArrayLike,
    cylinder_radius_m: float,
    cylinder_half_width_m: float,
    sphere_inner_radius_m: float,
) -> float:
    """Return conservative clearance of a finite cylinder in a sphere.

    Zero denotes first contact and a negative value denotes penetration of
    the analytic inner sphere.
    """

    center = np.asarray(center_from_sphere_m, dtype=np.float64)
    axis = np.asarray(cylinder_axis_world, dtype=np.float64)
    if center.shape != (3,) or axis.shape != (3,):
        raise ValueError("Cylinder centre and axis must be three-vectors")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 0.0:
        raise ValueError("Cylinder axis cannot be zero")
    axis = axis / axis_norm

    axial_offset = abs(float(np.dot(center, axis)))
    perpendicular = center - np.dot(center, axis) * axis
    perpendicular_offset = float(np.linalg.norm(perpendicular))
    envelope_radius = math.hypot(
        axial_offset + float(cylinder_half_width_m),
        perpendicular_offset + float(cylinder_radius_m),
    )
    return float(sphere_inner_radius_m) - envelope_radius


def ball_inside_sphere_clearance_m(
    center_from_sphere_m: npt.ArrayLike,
    ball_radius_m: float,
    sphere_inner_radius_m: float,
) -> float:
    """Return clearance of a ball from the inside of a spherical shell."""

    center = np.asarray(center_from_sphere_m, dtype=np.float64)
    if center.shape != (3,):
        raise ValueError("Ball centre must be a three-vector")
    return (
        float(sphere_inner_radius_m)
        - float(np.linalg.norm(center))
        - float(ball_radius_m)
    )


def radial_ball_center_shift_m(
    center_from_sphere_m: npt.ArrayLike,
    ball_radius_m: float,
    sphere_inner_radius_m: float,
    target_clearance_m: float,
) -> npt.NDArray[np.float64]:
    """Translate an internal ball radially to a requested positive clearance."""

    center = np.asarray(center_from_sphere_m, dtype=np.float64)
    if center.shape != (3,):
        raise ValueError("Ball centre must be a three-vector")
    center_norm = float(np.linalg.norm(center))
    target_radius = (
        float(sphere_inner_radius_m)
        - float(ball_radius_m)
        - float(target_clearance_m)
    )
    if center_norm <= 0.0:
        raise ValueError("Ball centre cannot coincide with sphere centre")
    if target_radius <= 0.0:
        raise ValueError("Requested ball clearance does not fit in sphere")
    return (target_radius / center_norm - 1.0) * center


def radial_cylinder_center_shift_m(
    center_from_sphere_m: npt.ArrayLike,
    cylinder_axis_world: npt.ArrayLike,
    cylinder_radius_m: float,
    cylinder_half_width_m: float,
    sphere_inner_radius_m: float,
    target_clearance_m: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Translate an internal finite cylinder radially to target clearance.

    The CAD wheel axis is retained.  Only the distance from the spherical
    centre changes, so the correction does not introduce a compliant mount.
    """

    center = np.asarray(center_from_sphere_m, dtype=np.float64)
    axis = np.asarray(cylinder_axis_world, dtype=np.float64)
    if center.shape != (3,) or axis.shape != (3,):
        raise ValueError("Cylinder centre and axis must be three-vectors")
    center_norm = float(np.linalg.norm(center))
    if center_norm <= 0.0:
        raise ValueError("Cylinder centre cannot coincide with sphere centre")
    target = float(target_clearance_m)
    if not math.isfinite(target) or target < 0.0:
        raise ValueError("Target clearance must be finite and non-negative")

    direction = center / center_norm

    def residual(distance_m: float) -> float:
        return cylinder_inside_sphere_clearance_m(
            center + distance_m * direction,
            axis,
            cylinder_radius_m,
            cylinder_half_width_m,
            sphere_inner_radius_m,
        ) - target

    radius = float(sphere_inner_radius_m)
    lower = -0.95 * center_norm
    upper = radius
    lower_value = residual(lower)
    upper_value = residual(upper)
    if lower_value < 0.0 or upper_value > 0.0:
        raise ValueError("Cannot bracket requested cylinder clearance")

    for _ in range(80):
        midpoint = 0.5 * (lower + upper)
        if residual(midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint
    distance = 0.5 * (lower + upper)
    return distance * direction
