"""Validated Isaac obstacle-course landmarks used by the ROS task expert."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


class CourseLandmarkError(ValueError):
    """Raised when Isaac does not provide a usable course description."""


@dataclass(frozen=True)
class CourseLandmarks:
    """World-frame geometry exported by the Isaac manual obstacle course."""

    gap_near_x_m: float
    gap_far_x_m: float
    ramp_entry_x_m: float
    ramp_exit_x_m: float
    stair_top_heights_m: tuple[float, ...]
    first_riser_x_m: float
    riser_depth_m: float
    button_center_xyz_m: tuple[float, float, float]
    exit_center_xyz_m: tuple[float, float, float]

    @classmethod
    def from_observation(cls, observation: Mapping[str, Any]) -> "CourseLandmarks":
        raw_course = observation.get("course")
        if not isinstance(raw_course, Mapping):
            raise CourseLandmarkError("Isaac course metadata is unavailable.")
        if raw_course.get("frame_id") != "world":
            raise CourseLandmarkError("Course landmarks must use the world frame.")
        gap = _mapping(raw_course, "gap")
        ramp = _mapping(raw_course, "ramp")
        stairs = _mapping(raw_course, "stairs")
        button = _mapping(raw_course, "button")
        exit_marker = _mapping(raw_course, "exit")
        near = _finite(gap, "near_edge_x_m")
        far = _finite(gap, "far_edge_x_m")
        if far <= near:
            raise CourseLandmarkError("Gap far edge must be beyond the near edge.")
        heights_raw = stairs.get("top_heights_m")
        if not isinstance(heights_raw, list | tuple) or not heights_raw:
            raise CourseLandmarkError("Stair top heights must be a non-empty array.")
        heights = tuple(_finite_value(value, "stair height") for value in heights_raw)
        if tuple(sorted(heights)) != heights:
            raise CourseLandmarkError("Stair top heights must be non-decreasing.")
        riser_depth = _finite(stairs, "riser_depth_m")
        if riser_depth <= 0.0:
            raise CourseLandmarkError("Stair riser depth must be positive.")
        return cls(
            gap_near_x_m=near,
            gap_far_x_m=far,
            ramp_entry_x_m=_finite(ramp, "entry_x_m"),
            ramp_exit_x_m=_finite(ramp, "exit_x_m"),
            stair_top_heights_m=heights,
            first_riser_x_m=_finite(stairs, "first_riser_x_m"),
            riser_depth_m=riser_depth,
            button_center_xyz_m=_vector3(button, "center_xyz_m"),
            exit_center_xyz_m=_vector3(exit_marker, "center_xyz_m"),
        )


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise CourseLandmarkError(f"Course field {key!r} must be an object.")
    return value


def _finite(payload: Mapping[str, Any], key: str) -> float:
    return _finite_value(payload.get(key), key)


def _finite_value(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise CourseLandmarkError(f"{name} must be numeric.") from error
    if not math.isfinite(result):
        raise CourseLandmarkError(f"{name} must be finite.")
    return result


def _vector3(payload: Mapping[str, Any], key: str) -> tuple[float, float, float]:
    value = payload.get(key)
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise CourseLandmarkError(f"{key} must be a three-element array.")
    return tuple(_finite_value(component, key) for component in value)