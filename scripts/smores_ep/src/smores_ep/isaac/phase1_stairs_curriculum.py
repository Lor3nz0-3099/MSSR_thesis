"""Unseen procedural stair curriculum for C1 APPO Snake8 robustification.

This module is intentionally independent from the validated expert
``UniformStairSpec`` / ``ConcertinaStaircase`` pipeline.

The expert remains the BC-v4 demonstration source.  APPO receives this
separate distribution, which can include descending and internally
non-uniform staircases.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Mapping

from smores_ep.isaac.obstacle_course import CourseBox


_SPLIT_RANGES = {
    "train": range(12000, 13000),
    "validation": range(13000, 13120),
    "test": range(14000, 14200),
    "challenge": range(14500, 14530),
}


@dataclass(frozen=True)
class StairCurriculumIdentity:
    """Stable identity of one procedural stair episode."""

    seed: int
    split: str
    ordinal: int
    difficulty: str
    direction: str
    profile: str


@dataclass(frozen=True)
class Phase1StairCourse:
    """Physical stair task presented to Snake8 during C1."""

    seed: int
    split: str
    difficulty: str
    direction: str
    profile: str

    step_rises_m: tuple[float, ...]
    tread_depths_m: tuple[float, ...]

    start_height_m: float
    step_top_heights_m: tuple[float, ...]
    goal_height_m: float

    width_m: float
    first_transition_x_m: float
    start_platform_length_m: float
    goal_platform_length_m: float

    support_boxes: tuple[CourseBox, ...]

    @property
    def step_count(self) -> int:
        return len(self.step_rises_m)

    @property
    def start_platform(self) -> CourseBox:
        return self.support_boxes[0]

    @property
    def step_boxes(self) -> tuple[CourseBox, ...]:
        return self.support_boxes[1:-1]

    @property
    def goal_platform(self) -> CourseBox:
        return self.support_boxes[-1]

    @property
    def start_platform_end_x_m(self) -> float:
        return (
            self.start_platform.center_xyz_m[0]
            + 0.5 * self.start_platform.size_xyz_m[0]
        )

    @property
    def goal_platform_start_x_m(self) -> float:
        return (
            self.goal_platform.center_xyz_m[0]
            - 0.5 * self.goal_platform.size_xyz_m[0]
        )

    @property
    def goal_center_xyz_m(self) -> tuple[float, float, float]:
        return (
            self.goal_platform_start_x_m
            + 0.5 * self.goal_platform_length_m,
            0.0,
            self.goal_height_m,
        )

    def to_observation(self) -> dict[str, Any]:
        """Public course geometry; contains no reward/success oracle."""

        return {
            "frame_id": "world",
            "course_profile": "phase1_stairs_curriculum_v1",
            "stairs": {
                "seed": self.seed,
                "split": self.split,
                "difficulty": self.difficulty,
                "direction": self.direction,
                "profile": self.profile,
                "first_transition_x_m": self.first_transition_x_m,
                "start_height_m": self.start_height_m,
                "goal_height_m": self.goal_height_m,
                "step_rises_m": list(self.step_rises_m),
                "tread_depths_m": list(self.tread_depths_m),
                "top_heights_m": list(self.step_top_heights_m),
            },
            "goal": {
                "center_xyz_m": list(self.goal_center_xyz_m),
            },
            "collision_boxes": [
                _box_observation(box)
                for box in self.support_boxes
                if box.collidable
            ],
        }


def stair_curriculum_seeds(
    split: str,
) -> tuple[int, ...]:
    """Return the immutable seed allocation for one split."""

    try:
        values = _SPLIT_RANGES[split]
    except KeyError as error:
        raise ValueError(
            f"Unknown stair curriculum split: {split!r}"
        ) from error

    return tuple(values)


def stair_curriculum_identity(
    seed: int,
    *,
    split: str,
) -> StairCurriculumIdentity:
    """Map one allocated seed to difficulty and structural mode."""

    seeds = stair_curriculum_seeds(split)

    try:
        ordinal = int(seed) - seeds[0]
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid stair seed: {seed!r}"
        ) from error

    if ordinal < 0 or ordinal >= len(seeds):
        raise ValueError(
            f"Seed {seed} does not belong to stair split {split!r}"
        )

    if split == "train":
        if ordinal < 250:
            difficulty = "easy"
        elif ordinal < 650:
            difficulty = "medium"
        else:
            difficulty = "hard"

    elif split == "validation":
        if ordinal < 40:
            difficulty = "easy"
        elif ordinal < 80:
            difficulty = "medium"
        else:
            difficulty = "hard"

    elif split == "test":
        if ordinal < 60:
            difficulty = "easy"
        elif ordinal < 130:
            difficulty = "medium"
        else:
            difficulty = "hard"

    else:
        difficulty = "challenge"

    if difficulty in {"easy", "medium"}:
        direction = "ascent"
        profile = "uniform"

    elif difficulty == "hard":
        mode = ordinal % 4

        direction = (
            "ascent"
            if mode in {0, 1}
            else "descent"
        )

        profile = (
            "uniform"
            if mode in {0, 2}
            else "nonuniform"
        )

    else:
        direction = (
            "ascent"
            if ordinal % 2 == 0
            else "descent"
        )

        profile = "nonuniform"

    return StairCurriculumIdentity(
        seed=int(seed),
        split=split,
        ordinal=ordinal,
        difficulty=difficulty,
        direction=direction,
        profile=profile,
    )


def phase1_stairs_curriculum_course(
    seed: int,
    *,
    split: str,
) -> Phase1StairCourse:
    """Generate one deterministic physical stair course."""

    identity = stair_curriculum_identity(
        seed,
        split=split,
    )

    rng = random.Random(
        int(seed) * 1_000_003 ^ 0x57A1A5
    )

    if identity.difficulty == "easy":
        count = rng.randint(2, 4)
        base_rise = _round_mm(
            rng.uniform(0.050, 0.062)
        )
        base_depth = _round_mm(
            rng.uniform(0.270, 0.340)
        )

        rises = (base_rise,) * count
        depths = (base_depth,) * count

    elif identity.difficulty == "medium":
        count = rng.randint(3, 5)
        base_rise = _round_mm(
            rng.uniform(0.052, 0.068)
        )
        base_depth = _round_mm(
            rng.uniform(0.230, 0.310)
        )

        rises = (base_rise,) * count
        depths = (base_depth,) * count

    elif identity.difficulty == "hard":
        count = rng.randint(3, 6)

        base_rise = rng.uniform(
            0.055,
            0.070,
        )

        base_depth = rng.uniform(
            0.220,
            0.295,
        )

        if identity.profile == "uniform":
            rise = _round_mm(base_rise)
            depth = _round_mm(base_depth)

            rises = (rise,) * count
            depths = (depth,) * count

        else:
            rises = _nonuniform_values(
                rng,
                count=count,
                base=base_rise,
                jitter=0.005,
                minimum=0.048,
                maximum=0.073,
            )

            depths = _nonuniform_values(
                rng,
                count=count,
                base=base_depth,
                jitter=0.025,
                minimum=0.190,
                maximum=0.320,
            )

    else:
        count = rng.randint(4, 6)

        rises = _nonuniform_values(
            rng,
            count=count,
            base=rng.uniform(
                0.060,
                0.070,
            ),
            jitter=0.006,
            minimum=0.050,
            maximum=0.075,
        )

        depths = _nonuniform_values(
            rng,
            count=count,
            base=rng.uniform(
                0.215,
                0.270,
            ),
            jitter=0.030,
            minimum=0.180,
            maximum=0.310,
        )

    total_rise = sum(rises)

    if identity.direction == "ascent":
        start_height = 0.0

        running_height = 0.0
        step_heights = []

        for rise in rises:
            running_height += rise
            step_heights.append(
                _round_mm(running_height)
            )

        goal_height = step_heights[-1]

    else:
        start_height = _round_mm(total_rise)

        running_height = start_height
        step_heights = []

        for index, rise in enumerate(rises):
            if index == len(rises) - 1:
                running_height = 0.0
            else:
                running_height = max(
                    0.0,
                    running_height - rise,
                )

            step_heights.append(
                _round_mm(running_height)
            )

        goal_height = 0.0
        step_heights[-1] = 0.0

    width_m = 1.20

    first_transition_x_m = _round_mm(
        rng.uniform(
            0.58,
            0.72,
        )
    )

    start_platform_length_m = 1.20
    goal_platform_length_m = 1.25

    support_boxes = _build_support_boxes(
        seed=int(seed),
        direction=identity.direction,
        start_height_m=start_height,
        step_top_heights_m=tuple(step_heights),
        tread_depths_m=depths,
        width_m=width_m,
        first_transition_x_m=first_transition_x_m,
        start_platform_length_m=start_platform_length_m,
        goal_platform_length_m=goal_platform_length_m,
        goal_height_m=goal_height,
    )

    return Phase1StairCourse(
        seed=int(seed),
        split=split,
        difficulty=identity.difficulty,
        direction=identity.direction,
        profile=identity.profile,
        step_rises_m=tuple(rises),
        tread_depths_m=tuple(depths),
        start_height_m=start_height,
        step_top_heights_m=tuple(step_heights),
        goal_height_m=goal_height,
        width_m=width_m,
        first_transition_x_m=first_transition_x_m,
        start_platform_length_m=start_platform_length_m,
        goal_platform_length_m=goal_platform_length_m,
        support_boxes=support_boxes,
    )


def _nonuniform_values(
    rng: random.Random,
    *,
    count: int,
    base: float,
    jitter: float,
    minimum: float,
    maximum: float,
) -> tuple[float, ...]:
    values = tuple(
        _round_mm(
            min(
                maximum,
                max(
                    minimum,
                    base
                    + rng.uniform(
                        -jitter,
                        +jitter,
                    ),
                ),
            )
        )
        for _ in range(count)
    )

    if len(set(values)) > 1:
        return values

    # Deterministic fail-safe against the extremely unlikely case where
    # millimetre rounding collapses every sample to the same value.
    mutable = list(values)

    candidate = _round_mm(
        min(
            maximum,
            mutable[-1] + 0.001,
        )
    )

    if candidate == mutable[-1]:
        candidate = _round_mm(
            max(
                minimum,
                mutable[-1] - 0.001,
            )
        )

    mutable[-1] = candidate

    return tuple(mutable)


def _build_support_boxes(
    *,
    seed: int,
    direction: str,
    start_height_m: float,
    step_top_heights_m: tuple[float, ...],
    tread_depths_m: tuple[float, ...],
    width_m: float,
    first_transition_x_m: float,
    start_platform_length_m: float,
    goal_platform_length_m: float,
    goal_height_m: float,
) -> tuple[CourseBox, ...]:
    boxes: list[CourseBox] = []

    start_x_m = (
        first_transition_x_m
        - start_platform_length_m
    )

    boxes.append(
        _support_box(
            name=f"Phase1Stairs_{seed}_Start",
            start_x_m=start_x_m,
            end_x_m=first_transition_x_m,
            top_height_m=start_height_m,
            width_m=width_m,
            semantic="phase1_stair_start",
        )
    )

    cursor_x_m = first_transition_x_m

    for index, (
        depth_m,
        top_height_m,
    ) in enumerate(
        zip(
            tread_depths_m,
            step_top_heights_m,
        ),
        start=1,
    ):
        end_x_m = (
            cursor_x_m
            + depth_m
        )

        boxes.append(
            _support_box(
                name=(
                    f"Phase1Stairs_{seed}_"
                    f"Step{index:02d}"
                ),
                start_x_m=cursor_x_m,
                end_x_m=end_x_m,
                top_height_m=top_height_m,
                width_m=width_m,
                semantic="phase1_stair_step",
            )
        )

        cursor_x_m = end_x_m

    boxes.append(
        _support_box(
            name=f"Phase1Stairs_{seed}_Goal",
            start_x_m=cursor_x_m,
            end_x_m=(
                cursor_x_m
                + goal_platform_length_m
            ),
            top_height_m=goal_height_m,
            width_m=width_m,
            semantic="phase1_stair_goal",
        )
    )

    return tuple(boxes)


def _support_box(
    *,
    name: str,
    start_x_m: float,
    end_x_m: float,
    top_height_m: float,
    width_m: float,
    semantic: str,
) -> CourseBox:
    if not (
        math.isfinite(start_x_m)
        and math.isfinite(end_x_m)
        and math.isfinite(top_height_m)
        and math.isfinite(width_m)
    ):
        raise ValueError(
            "Stair support dimensions must be finite"
        )

    if end_x_m <= start_x_m:
        raise ValueError(
            "Stair support must have positive X length"
        )

    if top_height_m < 0.0:
        raise ValueError(
            "Stair support cannot extend below nominal ground"
        )

    bottom_height_m = -0.020

    return CourseBox(
        name=name,
        center_xyz_m=(
            0.5
            * (
                start_x_m
                + end_x_m
            ),
            0.0,
            0.5
            * (
                top_height_m
                + bottom_height_m
            ),
        ),
        size_xyz_m=(
            end_x_m
            - start_x_m,
            width_m,
            top_height_m
            - bottom_height_m,
        ),
        color_rgb=(
            0.24,
            0.27,
            0.31,
        ),
        collidable=True,
        semantic=semantic,
    )


def _box_observation(
    box: CourseBox,
) -> Mapping[str, Any]:
    return {
        "name": box.name,
        "center_xyz_m": list(
            box.center_xyz_m
        ),
        "size_xyz_m": list(
            box.size_xyz_m
        ),
        "semantic": box.semantic,
        "pitch_deg": box.pitch_deg,
        "yaw_deg": box.yaw_deg,
    }


def _round_mm(
    value: float,
) -> float:
    return round(
        float(value),
        3,
    )


# ============================================================
# C1 STAIRS — ISAAC RUNTIME MATERIALIZATION
# ============================================================


def place_phase1_stairs_spawn_layout(
    layout: Mapping[
        str,
        tuple[
            float,
            float,
            float,
            float,
        ],
    ],
    course: Phase1StairCourse,
) -> dict[
    str,
    tuple[
        float,
        float,
        float,
        float,
    ],
]:
    """Place loose modules on the physical stair start platform.

    The existing validated self-assembly XY/yaw layout is preserved.
    Only world Z changes.

    For ascent:
        start_height = 0 -> legacy spawn unchanged.

    For descent:
        start_height > 0 -> the complete loose swarm starts on the
        elevated upper platform, with the same wheel clearance that the
        normal flat-ground spawn already uses.
    """

    dz = float(
        course.start_height_m
    )

    return {
        module_id: (
            float(x_m),
            float(y_m),
            float(z_m) + dz,
            float(yaw_deg),
        )
        for module_id, (
            x_m,
            y_m,
            z_m,
            yaw_deg,
        ) in layout.items()
    }


def install_phase1_stairs_curriculum_course(
    stage: Any,
    seed: int,
    split: str,
) -> Phase1StairCourse:
    """Materialize one C1 stairs episode into the live Isaac stage."""

    from smores_ep.isaac.obstacle_course import (
        _install_course_boxes,
    )

    course = (
        phase1_stairs_curriculum_course(
            int(seed),
            split=str(split),
        )
    )

    _install_course_boxes(
        stage,
        "/World/Phase1StairsCurriculum",
        course.support_boxes,
    )

    return course
