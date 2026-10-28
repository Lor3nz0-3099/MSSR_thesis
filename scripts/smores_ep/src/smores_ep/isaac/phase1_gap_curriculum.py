"""Procedural C1 Snake8 gap curriculum.

This distribution is intentionally separate from the expert gap fixtures
used for demonstration generation and robustness validation.

The first version remains inside the largest currently validated physical
Snake8 gap width (0.178 m).  Structural novelty comes from varying:

- gap width;
- distance from assembled start morphology to the near edge;
- landing-bank length;
- small lateral displacement of the landing bank.

Banks remain coplanar and unrotated.  This isolates gap-crossing learning
from unrelated orientation complexity.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any

from smores_ep.isaac.obstacle_course import (
    CourseBox,
)


PHYSICALLY_VALIDATED_MAX_GAP_WIDTH_M = 0.178

APPROACH_START_X_M = -1.000
BANK_WIDTH_M = 1.200
BANK_THICKNESS_M = 0.020

_SPLIT_RANGES = {
    "train": (
        16000,
        17000,
    ),
    "validation": (
        17000,
        17120,
    ),
    "test": (
        18000,
        18200,
    ),
    "challenge": (
        18500,
        18530,
    ),
}

_RANGES = {
    "easy": {
        "gap_width_m": (
            0.120,
            0.145,
        ),
        "near_edge_x_m": (
            0.620,
            0.750,
        ),
        "landing_length_m": (
            1.150,
            1.350,
        ),
        "landing_offset_m": (
            0.0,
            0.0,
        ),
    },
    "medium": {
        "gap_width_m": (
            0.140,
            0.160,
        ),
        "near_edge_x_m": (
            0.560,
            0.700,
        ),
        "landing_length_m": (
            1.050,
            1.300,
        ),
        "landing_offset_m": (
            -0.025,
            0.025,
        ),
    },
    "hard": {
        "gap_width_m": (
            0.155,
            0.178,
        ),
        "near_edge_x_m": (
            0.520,
            0.640,
        ),
        "landing_length_m": (
            0.950,
            1.200,
        ),
        "landing_offset_m": (
            -0.060,
            0.060,
        ),
    },
    "challenge": {
        "gap_width_m": (
            0.165,
            0.178,
        ),
        "near_edge_x_m": (
            0.500,
            0.600,
        ),
        "landing_length_m": (
            0.850,
            1.050,
        ),
        "landing_offset_m": (
            -0.080,
            0.080,
        ),
    },
}


@dataclass(frozen=True)
class GapCurriculumIdentity:
    split: str
    ordinal: int
    difficulty: str


@dataclass(frozen=True)
class Phase1GapCourse:
    seed: int
    split: str
    difficulty: str

    gap_width_m: float
    near_edge_x_m: float

    approach_start_x_m: float
    landing_length_m: float

    bank_width_m: float
    landing_center_y_m: float
    bank_thickness_m: float

    support_boxes: tuple[
        CourseBox,
        ...,
    ]

    def __post_init__(
        self,
    ) -> None:
        numeric = (
            self.gap_width_m,
            self.near_edge_x_m,
            self.approach_start_x_m,
            self.landing_length_m,
            self.bank_width_m,
            self.landing_center_y_m,
            self.bank_thickness_m,
        )

        if not all(
            math.isfinite(
                value
            )
            for value in numeric
        ):
            raise ValueError(
                "Gap course dimensions must be finite"
            )

        if not (
            0.080
            <= self.gap_width_m
            <= PHYSICALLY_VALIDATED_MAX_GAP_WIDTH_M
        ):
            raise ValueError(
                "Gap width exceeds the C1 physical envelope"
            )

        if self.near_edge_x_m <= 0.34:
            raise ValueError(
                "Gap near edge overlaps the assembly region"
            )

        if (
            self.near_edge_x_m
            <= self.approach_start_x_m
        ):
            raise ValueError(
                "Gap approach bank has invalid extent"
            )

        if self.landing_length_m < 0.80:
            raise ValueError(
                "Gap landing bank is too short"
            )

        if self.bank_width_m < 0.80:
            raise ValueError(
                "Gap banks are too narrow"
            )

        if not (
            0.005
            <= self.bank_thickness_m
            <= 0.20
        ):
            raise ValueError(
                "Gap bank thickness is invalid"
            )

        if len(
            self.support_boxes
        ) != 2:
            raise ValueError(
                "Gap course requires exactly two support banks"
            )

    @property
    def far_edge_x_m(
        self,
    ) -> float:
        return (
            self.near_edge_x_m
            + self.gap_width_m
        )

    @property
    def approach_length_m(
        self,
    ) -> float:
        return (
            self.near_edge_x_m
            - self.approach_start_x_m
        )

    @property
    def landing_end_x_m(
        self,
    ) -> float:
        return (
            self.far_edge_x_m
            + self.landing_length_m
        )

    @property
    def approach_bank(
        self,
    ) -> CourseBox:
        return self.support_boxes[
            0
        ]

    @property
    def landing_bank(
        self,
    ) -> CourseBox:
        return self.support_boxes[
            1
        ]

    @property
    def goal_center_xyz_m(
        self,
    ) -> tuple[
        float,
        float,
        float,
    ]:
        return (
            self.far_edge_x_m
            + 0.5
            * self.landing_length_m,
            self.landing_center_y_m,
            0.0,
        )

    def to_observation(
        self,
    ) -> dict[
        str,
        Any,
    ]:
        return {
            "frame_id":
                "world",

            "course_profile":
                "phase1_gap_curriculum_v1",

            "gap": {
                "seed":
                    self.seed,

                "split":
                    self.split,

                "difficulty":
                    self.difficulty,

                "width_m":
                    self.gap_width_m,

                "near_edge_x_m":
                    self.near_edge_x_m,

                "far_edge_x_m":
                    self.far_edge_x_m,

                "approach_start_x_m":
                    self.approach_start_x_m,

                "approach_length_m":
                    self.approach_length_m,

                "landing_length_m":
                    self.landing_length_m,

                "landing_center_y_m":
                    self.landing_center_y_m,

                "bank_width_m":
                    self.bank_width_m,
            },

            "goal": {
                "center_xyz_m":
                    list(
                        self.goal_center_xyz_m
                    ),
            },

            "collision_boxes": [
                _box_observation(
                    box
                )
                for box
                in self.support_boxes
            ],
        }


def _box_observation(
    box: CourseBox,
) -> dict[
    str,
    Any,
]:
    return {
        "name":
            box.name,

        "center_xyz_m":
            list(
                box.center_xyz_m
            ),

        "size_xyz_m":
            list(
                box.size_xyz_m
            ),

        "semantic":
            box.semantic,

        "pitch_deg":
            box.pitch_deg,
    }


def gap_curriculum_seeds(
    split: str,
) -> tuple[
    int,
    ...,
]:
    try:
        start, end = (
            _SPLIT_RANGES[
                split
            ]
        )
    except KeyError as error:
        raise ValueError(
            f"Unknown gap curriculum split: {split!r}"
        ) from error

    return tuple(
        range(
            start,
            end,
        )
    )


def gap_curriculum_identity(
    seed: int,
    *,
    split: str,
) -> GapCurriculumIdentity:
    seeds = (
        gap_curriculum_seeds(
            split
        )
    )

    start = seeds[
        0
    ]

    end = seeds[
        -1
    ]

    if (
        isinstance(
            seed,
            bool,
        )
        or not isinstance(
            seed,
            int,
        )
        or seed < start
        or seed > end
    ):
        raise ValueError(
            f"Gap seed {seed!r} is not allocated "
            f"to split {split!r}"
        )

    ordinal = (
        seed - start
    )

    if split == "challenge":
        difficulty = "challenge"

    elif split == "train":
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
        raise ValueError(
            f"Unknown gap curriculum split: {split!r}"
        )

    return GapCurriculumIdentity(
        split=split,
        ordinal=ordinal,
        difficulty=difficulty,
    )


def _sample_mm(
    generator: random.Random,
    bounds: tuple[
        float,
        float,
    ],
) -> float:
    low, high = bounds

    if low == high:
        return float(
            low
        )

    return round(
        generator.uniform(
            low,
            high,
        ),
        3,
    )


def phase1_gap_curriculum_course(
    seed: int,
    *,
    split: str,
) -> Phase1GapCourse:
    identity = (
        gap_curriculum_identity(
            seed,
            split=split,
        )
    )

    ranges = _RANGES[
        identity.difficulty
    ]

    generator = (
        random.Random(
            seed
        )
    )

    gap_width_m = _sample_mm(
        generator,
        ranges[
            "gap_width_m"
        ],
    )

    near_edge_x_m = _sample_mm(
        generator,
        ranges[
            "near_edge_x_m"
        ],
    )

    landing_length_m = _sample_mm(
        generator,
        ranges[
            "landing_length_m"
        ],
    )

    landing_center_y_m = _sample_mm(
        generator,
        ranges[
            "landing_offset_m"
        ],
    )

    # The cap is applied again after rounding to make the physical
    # assumption explicit and fail closed.
    gap_width_m = min(
        gap_width_m,
        PHYSICALLY_VALIDATED_MAX_GAP_WIDTH_M,
    )

    far_edge_x_m = (
        near_edge_x_m
        + gap_width_m
    )

    approach_length_m = (
        near_edge_x_m
        - APPROACH_START_X_M
    )

    platform_color = (
        0.24,
        0.27,
        0.31,
    )

    approach_bank = CourseBox(
        "Phase1GapApproachBank",
        (
            APPROACH_START_X_M
            + 0.5
            * approach_length_m,
            0.0,
            -0.5
            * BANK_THICKNESS_M,
        ),
        (
            approach_length_m,
            BANK_WIDTH_M,
            BANK_THICKNESS_M,
        ),
        platform_color,
        semantic="gap_approach_bank",
    )

    landing_bank = CourseBox(
        "Phase1GapLandingBank",
        (
            far_edge_x_m
            + 0.5
            * landing_length_m,
            landing_center_y_m,
            -0.5
            * BANK_THICKNESS_M,
        ),
        (
            landing_length_m,
            BANK_WIDTH_M,
            BANK_THICKNESS_M,
        ),
        platform_color,
        semantic="gap_landing_bank",
    )

    return Phase1GapCourse(
        seed=seed,
        split=split,
        difficulty=identity.difficulty,

        gap_width_m=gap_width_m,
        near_edge_x_m=near_edge_x_m,

        approach_start_x_m=(
            APPROACH_START_X_M
        ),

        landing_length_m=(
            landing_length_m
        ),

        bank_width_m=(
            BANK_WIDTH_M
        ),

        landing_center_y_m=(
            landing_center_y_m
        ),

        bank_thickness_m=(
            BANK_THICKNESS_M
        ),

        support_boxes=(
            approach_bank,
            landing_bank,
        ),
    )


def install_phase1_gap_curriculum_course(
    stage,
    *,
    seed: int,
    split: str,
) -> Phase1GapCourse:
    """Materialize one procedural C1 gap course in the open USD stage."""

    # Lazy import keeps the pure curriculum generator independent from
    # Isaac/USD imports during unit tests and dataset-side inspection.
    from smores_ep.isaac import obstacle_course

    course = (
        phase1_gap_curriculum_course(
            seed,
            split=split,
        )
    )

    obstacle_course._install_course_boxes(
        stage,
        "/World/Phase1GapCurriculum",
        course.support_boxes,
    )

    return course
