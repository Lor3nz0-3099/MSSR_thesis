"""Pure geometry helpers for the MobileManipulator8 button IK controller."""
from __future__ import annotations

import math

import numpy as np


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        raise ValueError("cannot normalize a near-zero vector")
    return value / norm


def shortest_angular_delta(actual_rad: float, target_rad: float) -> float:
    """Return the wrapped physical delta that moves actual to target."""

    return math.atan2(
        math.sin(float(target_rad) - float(actual_rad)),
        math.cos(float(target_rad) - float(actual_rad)),
    )



def plan_mm8_reposition_command(
    *,
    current_x_m: float,
    current_y_m: float,
    current_yaw_rad: float,
    target_x_m: float,
    target_y_m: float,
    drive_axis_x: float,
    drive_axis_y: float,
    final_yaw_rad: float | None = None,
    position_tolerance_m: float = 0.005,
    linear_speed_m_s: float = 0.05,
) -> dict[str, float | str]:
    """Plan translation-only MM8 motion along its fixed drive axis."""

    # MM8 cannot steer.  Root/odom yaw is intentionally irrelevant.
    _ = current_yaw_rad
    _ = final_yaw_rad

    axis_norm = math.hypot(
        float(drive_axis_x),
        float(drive_axis_y),
    )
    if axis_norm <= 1.0e-12:
        raise ValueError("MM8 drive axis must be non-zero")

    axis_x = float(drive_axis_x) / axis_norm
    axis_y = float(drive_axis_y) / axis_norm

    dx = float(target_x_m) - float(current_x_m)
    dy = float(target_y_m) - float(current_y_m)

    longitudinal_error = (
        dx * axis_x
        + dy * axis_y
    )

    if abs(longitudinal_error) <= float(position_tolerance_m):
        return {
            "phase": "done",
            "linear_x_m_s": 0.0,
            "angular_z_rad_s": 0.0,
        }

    return {
        "phase": "translate",
        "linear_x_m_s": math.copysign(
            float(linear_speed_m_s),
            longitudinal_error,
        ),
        "angular_z_rad_s": 0.0,
    }


def orientation_residual(
    normal: np.ndarray,
    target_normal: np.ndarray,
    *,
    lever_m: float,
) -> np.ndarray:
    """Continuous orientation residual with a unique zero at target_normal.

    Using ``normal - target_normal`` removes the old +/- normal ambiguity:
    the antiparallel branch has residual magnitude ``2 * lever_m`` instead of
    looking identical to the desired branch.
    """

    current = _normalize(normal)
    target = _normalize(target_normal)
    return float(lever_m) * (current - target)


def vector_angle_deg(vector: np.ndarray, target: np.ndarray) -> float:
    """Unsigned angle in degrees between two non-zero vectors."""

    lhs = _normalize(vector)
    rhs = _normalize(target)
    dot = float(np.clip(lhs @ rhs, -1.0, 1.0))
    return math.degrees(math.acos(dot))


def bounded_position_waypoint(
    current: np.ndarray,
    target: np.ndarray,
    *,
    max_step_m: float,
) -> np.ndarray:
    """Move from current toward target by no more than max_step_m."""

    current_value = np.asarray(current, dtype=float)
    target_value = np.asarray(target, dtype=float)
    delta = target_value - current_value
    distance = float(np.linalg.norm(delta))
    if distance <= float(max_step_m):
        return target_value.copy()
    if distance <= 1.0e-12:
        return current_value.copy()
    return current_value + delta * (float(max_step_m) / distance)


def bounded_normal_waypoint(
    current: np.ndarray,
    target: np.ndarray,
    *,
    max_angle_rad: float,
) -> np.ndarray:
    """Rotate a unit normal toward target by at most max_angle_rad.

    Rodrigues' formula is used for the ordinary case.  A deterministic
    orthogonal axis is selected for the nearly antiparallel case so that the
    controller can leave the old 180-degree branch without a discontinuity.
    """

    current_unit = _normalize(current)
    target_unit = _normalize(target)

    dot = float(np.clip(current_unit @ target_unit, -1.0, 1.0))
    angle = math.acos(dot)
    if angle <= float(max_angle_rad):
        return target_unit.copy()

    axis = np.cross(current_unit, target_unit)
    axis_norm = float(np.linalg.norm(axis))

    if axis_norm <= 1.0e-10:
        basis = np.asarray([1.0, 0.0, 0.0])
        if abs(float(current_unit @ basis)) > 0.9:
            basis = np.asarray([0.0, 1.0, 0.0])
        axis = np.cross(current_unit, basis)
        axis_norm = float(np.linalg.norm(axis))

    axis = axis / axis_norm
    step = float(max_angle_rad)

    rotated = (
        current_unit * math.cos(step)
        + np.cross(axis, current_unit) * math.sin(step)
        + axis * float(axis @ current_unit) * (1.0 - math.cos(step))
    )
    return _normalize(rotated)
