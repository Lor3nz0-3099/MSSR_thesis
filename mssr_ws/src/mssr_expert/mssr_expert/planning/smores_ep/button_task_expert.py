"""Deterministic geometry for the camera-guided button task.

The expert deliberately separates:

1. perceived button geometry,
2. RC-Car8 pre-reconfiguration alignment,
3. MobileManipulator8 longitudinal alignment,
4. arm IK/contact execution.

Only the first three are implemented here.  Arm IK is intentionally a
separate stage because its target depends on the perceived button height
and on the live manipulator configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


class ButtonTaskObservationError(ValueError):
    """Raised when the perceived button state is unavailable or malformed."""


@dataclass(frozen=True)
class ButtonObservation:
    """Perception-equivalent live button observation in the world frame."""

    center_xyz_m: tuple[float, float, float]
    depression_m: float


@dataclass(frozen=True)
class ButtonBaseTarget:
    """Desired planar root pose relative to the observed button."""

    button_center_xyz_m: tuple[float, float, float]
    root_xy_m: tuple[float, float]
    forward_yaw_rad: float
    standoff_m: float
    lateral_bias_m: float = 0.0


@dataclass(frozen=True)
class LongitudinalTrackingError:
    """Target error expressed in the robot's current planar frame."""

    along_m: float
    lateral_m: float
    heading_error_rad: float


def observe_button(
    observation: Mapping[str, Any],
) -> ButtonObservation:
    """Return the live button pose supplied by perception/simulation.

    ``current_center_xyz_m`` is preferred over the nominal fixture centre.
    This makes the expert seed-independent and also follows the moving
    plunger during physical depression.
    """

    course = observation.get("course")
    if not isinstance(course, Mapping):
        raise ButtonTaskObservationError(
            "Course/perception metadata is unavailable."
        )

    button = course.get("button")
    if not isinstance(button, Mapping):
        raise ButtonTaskObservationError(
            "Button perception metadata is unavailable."
        )

    center = button.get(
        "current_center_xyz_m",
        button.get("center_xyz_m"),
    )

    if not isinstance(center, (list, tuple)) or len(center) != 3:
        raise ButtonTaskObservationError(
            "Button centre must contain three world-frame coordinates."
        )

    xyz = tuple(_finite(value, "button centre") for value in center)

    depression = _finite(
        button.get("depression_m", 0.0),
        "button depression",
    )

    return ButtonObservation(
        center_xyz_m=xyz,
        depression_m=depression,
    )


def button_base_target(
    button_center_xyz_m: tuple[float, float, float],
    *,
    standoff_m: float,
    future_forward_yaw_rad: float,
    lateral_bias_m: float = 0.0,
) -> ButtonBaseTarget:
    """Compute the robot root pose required around a perceived button.

    The future MM8 forward axis deliberately points *away* from the button.
    Consequently, after reconfiguration, negative longitudinal velocity
    moves the manipulator backwards toward the button.

    ``lateral_bias_m`` is a calibration hook for any repeatable lateral
    displacement introduced by RC-Car -> MM8 reconfiguration.
    """

    standoff = _finite(standoff_m, "button standoff")
    if standoff <= 0.0:
        raise ValueError("Button standoff must be positive.")

    yaw = _wrap_angle(
        _finite(future_forward_yaw_rad, "future forward yaw")
    )
    lateral_bias = _finite(lateral_bias_m, "lateral bias")

    bx, by, bz = (
        _finite(value, "button centre")
        for value in button_center_xyz_m
    )

    forward_x = math.cos(yaw)
    forward_y = math.sin(yaw)

    left_x = -math.sin(yaw)
    left_y = math.cos(yaw)

    # Root lies on the future forward side of the button.  Thus the
    # manipulator's rear/arm side points toward the button.
    root_x = (
        bx
        + standoff * forward_x
        + lateral_bias * left_x
    )
    root_y = (
        by
        + standoff * forward_y
        + lateral_bias * left_y
    )

    return ButtonBaseTarget(
        button_center_xyz_m=(bx, by, bz),
        root_xy_m=(root_x, root_y),
        forward_yaw_rad=yaw,
        standoff_m=standoff,
        lateral_bias_m=lateral_bias,
    )


def longitudinal_tracking_error(
    root_xyz_m: tuple[float, float, float],
    root_yaw_rad: float,
    target: ButtonBaseTarget,
) -> LongitudinalTrackingError:
    """Express target error along/across the live MM8 longitudinal axis."""

    rx = float(root_xyz_m[0])
    ry = float(root_xyz_m[1])

    dx = target.root_xy_m[0] - rx
    dy = target.root_xy_m[1] - ry

    c = math.cos(root_yaw_rad)
    s = math.sin(root_yaw_rad)

    along = c * dx + s * dy
    lateral = -s * dx + c * dy

    return LongitudinalTrackingError(
        along_m=along,
        lateral_m=lateral,
        heading_error_rad=_wrap_angle(
            target.forward_yaw_rad - root_yaw_rad
        ),
    )


def button_is_physically_pressed(
    observation: Mapping[str, Any],
    *,
    threshold_m: float,
) -> bool:
    """Success is physical plunger depression, never EE proximity."""

    threshold = _finite(threshold_m, "button depression threshold")
    if threshold <= 0.0:
        raise ValueError("Button depression threshold must be positive.")

    return observe_button(observation).depression_m >= threshold


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ButtonTaskObservationError(
            f"{name} must be numeric."
        ) from error

    if not math.isfinite(result):
        raise ButtonTaskObservationError(
            f"{name} must be finite."
        )

    return result


def _wrap_angle(angle_rad: float) -> float:
    return math.atan2(
        math.sin(angle_rad),
        math.cos(angle_rad),
    )
