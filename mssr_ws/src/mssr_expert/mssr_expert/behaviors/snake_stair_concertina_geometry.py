"""Validated staircase collision geometry for spatial Snake8 gaits."""

from __future__ import annotations

from typing import Any, Mapping

from mssr_expert.behaviors.snake_stair_gait import (
    SnakeStairGaitError,
    UniformStaircase,
)


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
