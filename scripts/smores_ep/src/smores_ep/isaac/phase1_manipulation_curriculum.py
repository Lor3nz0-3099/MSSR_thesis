"""Procedural C1 MobileManipulator8 manipulation curriculum.

This generator is intentionally separate from the expert/BC button
generator in obstacle_course.py.

Structural novelty:
- expert fixture family: axis-aligned +/-X and +/-Y press directions;
- C1 MARL fixture family: continuously rotated horizontal normals.

The existing physical fixture dimensions and reachable button-height
envelope are preserved.

No reward terms or success oracle live in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any

from smores_ep.isaac.obstacle_course import (
    BUTTON_ASSEMBLY_SPAWN_RADIUS_M,
    BUTTON_ASSEMBLY_SPAWN_STANDOFF_M,
    BUTTON_PLATFORM_CENTER_XY_M,
    BUTTON_PLATFORM_SIZE_XY_M,
    BUTTON_PLUNGER_DEPTH_M,
    BUTTON_PLUNGER_HEIGHT_M,
    BUTTON_PLUNGER_STROKE_M,
    BUTTON_PLUNGER_TANGENT_M,
    BUTTON_SUPPORT_HEIGHT_M,
    BUTTON_SUPPORT_OFFSET_M,
    BUTTON_SUPPORT_TANGENT_M,
    BUTTON_SUPPORT_THICKNESS_M,
    CourseBox,
)


_SPLIT_BOUNDS = {
    "train": (
        20000,
        21000,
    ),
    "validation": (
        21000,
        21120,
    ),
    "test": (
        22000,
        22200,
    ),
    "challenge": (
        22500,
        22530,
    ),
}

_MANIPULATION_READY_STANDOFF_M = (
    0.325
)

_MAX_BUTTON_Z_M = (
    0.17623346377092517
)

_PLATFORM_CLEARANCE_M = (
    0.08
)


@dataclass(frozen=True)
class ManipulationCurriculumIdentity:
    split: str
    ordinal: int
    difficulty: str


@dataclass(frozen=True)
class Phase1ManipulationCourse:
    seed: int
    split: str
    difficulty: str

    button_center_xyz_m: tuple[
        float,
        float,
        float,
    ]

    press_direction_world_xy: tuple[
        float,
        float,
    ]

    press_yaw_rad: float

    assembly_spawn_center_xy_m: tuple[
        float,
        float,
    ]

    assembly_spawn_yaw_rad: float
    assembly_spawn_radius_m: float

    manipulation_ready_base_xy_m: tuple[
        float,
        float,
    ]

    manipulation_ready_standoff_m: float

    base_drive_axis_xy: tuple[
        float,
        float,
    ]

    support_boxes: tuple[
        CourseBox,
        ...,
    ]

    @property
    def button_wall(
        self,
    ) -> CourseBox:
        return next(
            box
            for box
            in self.support_boxes
            if box.semantic
            == "button_support"
        )

    @property
    def button_plunger(
        self,
    ) -> CourseBox:
        return next(
            box
            for box
            in self.support_boxes
            if box.semantic
            == "button"
        )

    def to_observation(
        self,
    ) -> dict[str, Any]:
        platform_cx, platform_cy = (
            BUTTON_PLATFORM_CENTER_XY_M
        )

        platform_sx, platform_sy = (
            BUTTON_PLATFORM_SIZE_XY_M
        )

        return {
            "frame_id":
                "world",

            "course_profile":
                "phase1_mobile_manipulator8_manipulation",

            "scenario": {
                "generator":
                    "phase1_manipulation_v1",

                "seed":
                    self.seed,

                "split":
                    self.split,

                "difficulty":
                    self.difficulty,
            },

            "button": {
                "center_xyz_m":
                    list(
                        self.button_center_xyz_m
                    ),

                "press_direction_world_xy":
                    list(
                        self.press_direction_world_xy
                    ),

                "press_yaw_rad":
                    self.press_yaw_rad,

                "manipulation_ready_base_xy_m":
                    list(
                        self.manipulation_ready_base_xy_m
                    ),

                "manipulation_ready_standoff_m":
                    self.manipulation_ready_standoff_m,

                "base_drive_axis_xy":
                    list(
                        self.base_drive_axis_xy
                    ),

                "assembly_spawn": {
                    "center_xy_m":
                        list(
                            self.assembly_spawn_center_xy_m
                        ),

                    "yaw_rad":
                        self.assembly_spawn_yaw_rad,

                    "radius_m":
                        self.assembly_spawn_radius_m,
                },

                "plunger_depth_m":
                    BUTTON_PLUNGER_DEPTH_M,

                "plunger_stroke_m":
                    BUTTON_PLUNGER_STROKE_M,

                "plunger_path": (
                    "/World/Phase1ManipulationCurriculum/"
                    "Phase1ManipulationButtonPlunger"
                ),
            },

            "platform": {
                "center_xy_m": [
                    platform_cx,
                    platform_cy,
                ],

                "size_xy_m": [
                    platform_sx,
                    platform_sy,
                ],

                "bounds_xy_m": [
                    platform_cx
                    - 0.5 * platform_sx,

                    platform_cx
                    + 0.5 * platform_sx,

                    platform_cy
                    - 0.5 * platform_sy,

                    platform_cy
                    + 0.5 * platform_sy,
                ],
            },

            "collision_boxes": [
                {
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

                    "yaw_deg":
                        box.yaw_deg,
                }
                for box
                in self.support_boxes
                if box.collidable
            ],
        }


def manipulation_curriculum_seeds(
    split: str,
) -> range:
    try:
        start, stop = (
            _SPLIT_BOUNDS[
                split
            ]
        )
    except KeyError as exc:
        raise ValueError(
            f"unknown manipulation split: {split!r}"
        ) from exc

    return range(
        start,
        stop,
    )


def manipulation_curriculum_identity(
    seed: int,
    *,
    split: str,
) -> ManipulationCurriculumIdentity:
    seeds = (
        manipulation_curriculum_seeds(
            split
        )
    )

    if (
        isinstance(
            seed,
            bool,
        )
        or not isinstance(
            seed,
            int,
        )
        or seed not in seeds
    ):
        raise ValueError(
            f"seed {seed!r} does not belong "
            f"to manipulation split {split!r}"
        )

    ordinal = (
        seed
        - seeds.start
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

    return ManipulationCurriculumIdentity(
        split=split,
        ordinal=ordinal,
        difficulty=difficulty,
    )


def _difficulty_ranges(
    difficulty: str,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    if difficulty == "easy":
        return (
            (8.0, 18.0),
            (0.80, 2.80),
            (-1.00, 1.00),
            (0.115, 0.155),
        )

    if difficulty == "medium":
        return (
            (18.0, 32.0),
            (0.70, 2.90),
            (-1.15, 1.15),
            (0.105, 0.168),
        )

    if difficulty == "hard":
        return (
            (32.0, 44.0),
            (0.65, 2.95),
            (-1.20, 1.20),
            (
                0.100,
                _MAX_BUTTON_Z_M,
            ),
        )

    if difficulty == "challenge":
        return (
            (40.0, 45.0),
            (0.65, 2.95),
            (-1.20, 1.20),
            (
                0.100,
                _MAX_BUTTON_Z_M,
            ),
        )

    raise ValueError(
        f"unknown difficulty: {difficulty!r}"
    )


def _sample_press_yaw(
    generator: random.Random,
    *,
    angle_offset_range_deg: tuple[
        float,
        float,
    ],
    ordinal: int,
    challenge: bool,
) -> float:
    # Force deterministic quadrant coverage in CHALLENGE;
    # ordinary splits remain fully seeded/random.
    if challenge:
        quadrant = (
            ordinal
            % 4
        )
    else:
        quadrant = (
            generator.randrange(
                4
            )
        )

    cardinal_deg = (
        90.0
        * quadrant
    )

    offset_deg = (
        generator.uniform(
            *angle_offset_range_deg
        )
    )

    sign = (
        -1.0
        if (
            (
                ordinal // 4
                if challenge
                else generator.randrange(2)
            )
            % 2
        )
        else +1.0
    )

    yaw_deg = (
        cardinal_deg
        + sign
        * offset_deg
    )

    return math.radians(
        yaw_deg
    )


def _assembly_center_has_clearance(
    center_xy_m: tuple[
        float,
        float,
    ],
) -> bool:
    platform_cx, platform_cy = (
        BUTTON_PLATFORM_CENTER_XY_M
    )

    platform_sx, platform_sy = (
        BUTTON_PLATFORM_SIZE_XY_M
    )

    xmin = (
        platform_cx
        - 0.5 * platform_sx
    )

    xmax = (
        platform_cx
        + 0.5 * platform_sx
    )

    ymin = (
        platform_cy
        - 0.5 * platform_sy
    )

    ymax = (
        platform_cy
        + 0.5 * platform_sy
    )

    margin = (
        BUTTON_ASSEMBLY_SPAWN_RADIUS_M
        + _PLATFORM_CLEARANCE_M
    )

    x_m, y_m = center_xy_m

    return (
        xmin + margin
        <= x_m
        <= xmax - margin

        and

        ymin + margin
        <= y_m
        <= ymax - margin
    )


def phase1_manipulation_curriculum_course(
    seed: int,
    *,
    split: str,
) -> Phase1ManipulationCourse:
    identity = (
        manipulation_curriculum_identity(
            seed,
            split=split,
        )
    )

    generator = random.Random(
        (
            int(seed)
            << 8
        )
        ^ 0x4D4D38
    )

    (
        angle_range_deg,
        x_range_m,
        y_range_m,
        z_range_m,
    ) = _difficulty_ranges(
        identity.difficulty
    )

    press_yaw_rad = (
        _sample_press_yaw(
            generator,
            angle_offset_range_deg=(
                angle_range_deg
            ),
            ordinal=identity.ordinal,
            challenge=(
                identity.difficulty
                == "challenge"
            ),
        )
    )

    nx = math.cos(
        press_yaw_rad
    )

    ny = math.sin(
        press_yaw_rad
    )

    button_x_m = None
    button_y_m = None
    assembly_center = None

    for _ in range(256):
        candidate_x_m = (
            generator.uniform(
                *x_range_m
            )
        )

        candidate_y_m = (
            generator.uniform(
                *y_range_m
            )
        )

        candidate_center = (
            candidate_x_m
            - nx
            * BUTTON_ASSEMBLY_SPAWN_STANDOFF_M,

            candidate_y_m
            - ny
            * BUTTON_ASSEMBLY_SPAWN_STANDOFF_M,
        )

        if _assembly_center_has_clearance(
            candidate_center
        ):
            button_x_m = (
                candidate_x_m
            )

            button_y_m = (
                candidate_y_m
            )

            assembly_center = (
                candidate_center
            )

            break

    if (
        button_x_m is None
        or button_y_m is None
        or assembly_center is None
    ):
        raise RuntimeError(
            "could not sample a manipulation fixture "
            "with safe assembly clearance"
        )

    if (
        identity.difficulty
        == "challenge"
    ):
        if (
            identity.ordinal
            % 2
            == 0
        ):
            button_z_m = (
                generator.uniform(
                    0.100,
                    0.115,
                )
            )

        else:
            button_z_m = (
                generator.uniform(
                    0.161,
                    _MAX_BUTTON_Z_M,
                )
            )

    else:
        button_z_m = (
            generator.uniform(
                *z_range_m
            )
        )

    button_center = (
        button_x_m,
        button_y_m,
        button_z_m,
    )

    wall_center = (
        button_x_m
        + nx
        * BUTTON_SUPPORT_OFFSET_M,

        button_y_m
        + ny
        * BUTTON_SUPPORT_OFFSET_M,

        0.150,
    )

    yaw_deg = math.degrees(
        press_yaw_rad
    )

    platform_cx, platform_cy = (
        BUTTON_PLATFORM_CENTER_XY_M
    )

    platform_sx, platform_sy = (
        BUTTON_PLATFORM_SIZE_XY_M
    )

    support_boxes = (
        CourseBox(
            "Phase1ManipulationPlatform",

            (
                platform_cx,
                platform_cy,
                -0.01,
            ),

            (
                platform_sx,
                platform_sy,
                0.02,
            ),

            (
                0.24,
                0.27,
                0.31,
            ),

            semantic=
                "button_test_platform",
        ),

        # Define local X along the arbitrary physical press normal.
        CourseBox(
            "Phase1ManipulationButtonWall",

            wall_center,

            (
                BUTTON_SUPPORT_THICKNESS_M,
                BUTTON_SUPPORT_TANGENT_M,
                BUTTON_SUPPORT_HEIGHT_M,
            ),

            (
                0.35,
                0.37,
                0.40,
            ),

            semantic=
                "button_support",

            yaw_deg=
                yaw_deg,
        ),

        CourseBox(
            "Phase1ManipulationButtonPlunger",

            button_center,

            (
                BUTTON_PLUNGER_DEPTH_M,
                BUTTON_PLUNGER_TANGENT_M,
                BUTTON_PLUNGER_HEIGHT_M,
            ),

            (
                0.85,
                0.08,
                0.06,
            ),

            semantic=
                "button",

            yaw_deg=
                yaw_deg,
        ),
    )

    ready_base = (
        button_x_m
        - nx
        * _MANIPULATION_READY_STANDOFF_M,

        button_y_m
        - ny
        * _MANIPULATION_READY_STANDOFF_M,
    )

    # The assembled robot's longitudinal +X points AWAY from
    # the fixture, keeping the rear/manipulator side toward it.
    assembly_spawn_yaw = math.atan2(
        -ny,
        -nx,
    )

    return Phase1ManipulationCourse(
        seed=seed,
        split=split,
        difficulty=(
            identity.difficulty
        ),

        button_center_xyz_m=(
            button_center
        ),

        press_direction_world_xy=(
            float(nx),
            float(ny),
        ),

        press_yaw_rad=(
            press_yaw_rad
        ),

        assembly_spawn_center_xy_m=(
            assembly_center
        ),

        assembly_spawn_yaw_rad=(
            assembly_spawn_yaw
        ),

        assembly_spawn_radius_m=(
            BUTTON_ASSEMBLY_SPAWN_RADIUS_M
        ),

        manipulation_ready_base_xy_m=(
            ready_base
        ),

        manipulation_ready_standoff_m=(
            _MANIPULATION_READY_STANDOFF_M
        ),

        base_drive_axis_xy=(
            -float(nx),
            -float(ny),
        ),

        support_boxes=(
            support_boxes
        ),
    )


def install_phase1_manipulation_curriculum_course(
    stage,
    *,
    seed: int,
    split: str,
) -> Phase1ManipulationCourse:
    """Materialize one C1 manipulation fixture in Isaac."""

    from smores_ep.isaac import (
        obstacle_course,
    )

    course = (
        phase1_manipulation_curriculum_course(
            seed,
            split=split,
        )
    )

    root_path = "/World/Phase1ManipulationCurriculum"
    obstacle_course._install_course_boxes(
        stage,
        root_path,
        course.support_boxes,
    )

    obstacle_course._install_button_prismatic_joint(
        stage,
        root_path=root_path,
        plunger_name=course.button_plunger.name,
        center_xyz_m=course.button_center_xyz_m,
        press_direction_world_xy=course.press_direction_world_xy,
        plunger_yaw_deg=course.button_plunger.yaw_deg,
        joint_name="ButtonPrismaticJoint",
    )

    return course
