"""Validated staircase collision geometry for spatial Snake8 gaits."""

from __future__ import annotations

from dataclasses import dataclass
import math

from typing import Any, Mapping



class SnakeStairGaitError(ValueError):
    """Raised when the live robot or course cannot define a safe gait."""


@dataclass(frozen=True)
class UniformStaircase:
    """Uniform +X staircase recognized from Isaac course landmarks."""

    first_riser_x_m: float
    tread_depth_m: float
    top_heights_m: tuple[float, ...]
    rise_m: float
    base_height_m: float = 0.0

    @classmethod
    def from_course(cls, course: Mapping[str, Any]) -> "UniformStaircase":
        if course.get("frame_id") != "world":
            raise SnakeStairGaitError("Stair landmarks must use world frame")
        stairs = course.get("stairs")
        if not isinstance(stairs, Mapping):
            raise SnakeStairGaitError("Course has no stair landmarks")
        if stairs.get("coordinate_frame", "world") != "world":
            raise SnakeStairGaitError("Stage-local stair landmarks require the gait stage_frame adapter")
        try:
            first = float(stairs["first_riser_x_m"])
            depth = float(stairs["riser_depth_m"])
            heights = tuple(
                float(value) for value in stairs["top_heights_m"]
            )
            base_height = float(stairs.get("base_height_m", 0.0))
        except (KeyError, TypeError, ValueError) as error:
            raise SnakeStairGaitError("Invalid stair landmarks") from error
        if not heights or depth <= 0.0 or not all(
            math.isfinite(value)
            for value in (first, depth, base_height, *heights)
        ):
            raise SnakeStairGaitError(
                "Stair dimensions must be positive and finite"
            )
        rises = tuple(
            upper - lower
            for lower, upper in zip(
                (base_height, *heights[:-1]), heights
            )
        )
        if min(rises) <= 0.0:
            raise SnakeStairGaitError("Stair heights must increase strictly")
        rise = sum(rises) / len(rises)
        if any(abs(value - rise) > 0.005 for value in rises):
            raise SnakeStairGaitError(
                "Snake8 gait requires uniform stair rises"
            )
        return cls(first, depth, heights, rise, base_height)


class ConcertinaStaircase(UniformStaircase):
    """Uniform landmarks cross-checked against Isaac collision boxes."""

    @classmethod
    def from_course(cls, course: Mapping[str, Any]) -> "ConcertinaStaircase":
        staircase = super().from_course(course)
        collision_boxes = course.get("collision_boxes")
        if collision_boxes is None:
            return staircase
        if not isinstance(collision_boxes, list | tuple):
            raise SnakeStairGaitError(
                "Course collision_boxes must be a sequence"
            )
        try:
            riser_boxes = sorted(
                (
                    box
                    for box in collision_boxes
                    if isinstance(box, Mapping)
                    and box.get("semantic") == "stair_test_riser"
                ),
                key=lambda box: float(box["center_xyz_m"][0]),
            )
            if len(riser_boxes) != len(staircase.top_heights_m):
                raise SnakeStairGaitError(
                    "Stair landmarks disagree with world collision boxes"
                )
            for index, (box, top_height) in enumerate(
                zip(riser_boxes, staircase.top_heights_m)
            ):
                center = tuple(float(value) for value in box["center_xyz_m"])
                size = tuple(float(value) for value in box["size_xyz_m"])
                front_x = center[0] - 0.5 * size[0]
                top_z = center[2] + 0.5 * size[2]
                expected_front = staircase.first_riser_x_m + (
                    index * staircase.tread_depth_m
                )
                if (
                    abs(front_x - expected_front) > 0.001
                    or abs(top_z - top_height) > 0.001
                ):
                    raise SnakeStairGaitError(
                        "Stair landmarks disagree with world collision boxes"
                    )
        except SnakeStairGaitError:
            raise
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise SnakeStairGaitError(
                "Invalid stair collision-box geometry"
            ) from error
        return staircase
