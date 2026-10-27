"""Unseen Phase-1 RC-Car8 MARL curriculum generator.

This generator is intentionally independent from the Expert/reference
RC-Car planar layout generator.

BC-v4 supplies a behavioral prior learned from demonstrations.
MAPPO/APPO trains on this structurally new distribution instead of
replaying or perturbing Expert stages.

The generator reuses only validated physical contracts:
- RC-Car8 footprint,
- road/support primitives,
- static cones,
- Phase1TrainingCourse reward semantics.

Generator-only centerlines remain private to reward/validation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable

from .obstacle_course import (
    CourseBox,
)
from .phase1_training_course import (
    OrientedPlatform,
    Phase1Cone,
    Phase1TrainingCourse,
    _road_support_boxes,
)


GENERATOR_ID = (
    "phase1_marl_unseen_v1"
)

VEHICLE_LENGTH_M = 0.310
VEHICLE_WIDTH_M = 0.232

CONE_RADIUS_M = 0.050
CONE_HEIGHT_M = 0.170

COLLISION_MARGIN_M = 0.015
CLEARANCE_MARGIN_M = 0.050

PLATFORM_SIZE_XY_M = (
    1.20,
    1.20,
)

_SPLITS = {
    "train": (
        7000,
        1000,
    ),
    "validation": (
        8000,
        120,
    ),
    "test": (
        9000,
        200,
    ),
    "challenge": (
        9500,
        30,
    ),
}

_SPLIT_SALTS = {
    "train": 0x71A1,
    "validation": 0x81A1,
    "test": 0x91A1,
    "challenge": 0x95A1,
}


@dataclass(frozen=True)
class MarlCurriculumIdentity:
    seed: int
    split: str
    ordinal: int
    difficulty: str
    family: str


def curriculum_seeds(
    split: str,
) -> tuple[int, ...]:
    try:
        start, count = _SPLITS[
            str(split)
        ]
    except KeyError as error:
        raise ValueError(
            f"Unknown MARL curriculum split: {split!r}"
        ) from error

    return tuple(
        range(
            start,
            start + count,
        )
    )


def split_for_seed(
    seed: int,
) -> str:
    if type(seed) is not int:
        raise TypeError(
            "MARL curriculum seed must be an integer"
        )

    for split, (
        start,
        count,
    ) in _SPLITS.items():
        if (
            start
            <= seed
            < start + count
        ):
            return split

    raise ValueError(
        f"Seed {seed} does not belong to any "
        "Phase-1 MARL curriculum split"
    )


def _difficulty_for(
    split: str,
    ordinal: int,
) -> str:
    if split == "train":
        if ordinal < 250:
            return "easy"

        if ordinal < 650:
            return "medium"

        return "hard"

    if split == "validation":
        if ordinal < 40:
            return "easy"

        if ordinal < 80:
            return "medium"

        return "hard"

    if split == "test":
        if ordinal < 60:
            return "easy"

        if ordinal < 130:
            return "medium"

        return "hard"

    if split == "challenge":
        return "hard"

    raise ValueError(
        f"Unknown split: {split!r}"
    )


_FAMILIES_BY_DIFFICULTY = {
    "easy": (
        "straight",
        "gentle_curve",
        "s_curve",
    ),
    "medium": (
        "straight",
        "gentle_curve",
        "s_curve",
        "chicane",
        "u_turn",
    ),
    "hard": (
        "s_curve",
        "chicane",
        "u_turn",
        "hairpin",
        "compound",
    ),
}

_CHALLENGE_FAMILIES = (
    "u_turn",
    "hairpin",
    "compound",
)


def curriculum_identity(
    seed: int,
    *,
    split: str,
) -> MarlCurriculumIdentity:
    canonical = split_for_seed(
        seed
    )

    if canonical != split:
        raise ValueError(
            f"Seed {seed} does not belong to split "
            f"{split!r}; it belongs to {canonical!r}"
        )

    start, _ = _SPLITS[
        split
    ]

    ordinal = (
        seed - start
    )

    difficulty = (
        _difficulty_for(
            split,
            ordinal,
        )
    )

    if split == "challenge":
        families = (
            _CHALLENGE_FAMILIES
        )
    else:
        families = (
            _FAMILIES_BY_DIFFICULTY[
                difficulty
            ]
        )

    salt = _SPLIT_SALTS[
        split
    ]

    family = families[
        (
            ordinal
            + salt
        )
        % len(families)
    ]

    return MarlCurriculumIdentity(
        seed=seed,
        split=split,
        ordinal=ordinal,
        difficulty=difficulty,
        family=family,
    )


def _rng(
    identity: MarlCurriculumIdentity,
) -> random.Random:
    return random.Random(
        (
            identity.seed
            * 1_000_003
        )
        ^ _SPLIT_SALTS[
            identity.split
        ]
    )


def _straight(
    length_m: float,
) -> tuple[str, float]:
    return (
        "straight",
        float(length_m),
    )


def _arc(
    radius_m: float,
    angle_rad: float,
) -> tuple[
    str,
    float,
    float,
]:
    return (
        "arc",
        float(radius_m),
        float(angle_rad),
    )


def _route_primitives(
    identity: MarlCurriculumIdentity,
    rng: random.Random,
):
    family = identity.family
    difficulty = (
        identity.difficulty
    )

    sign = (
        +1.0
        if rng.random() < 0.5
        else -1.0
    )

    if family == "straight":
        if difficulty == "easy":
            length = rng.uniform(
                2.6,
                4.0,
            )
        else:
            length = rng.uniform(
                4.0,
                6.8,
            )

        return (
            _straight(
                length
            ),
        )

    if family == "gentle_curve":
        return (
            _straight(
                rng.uniform(
                    0.7,
                    1.3,
                )
            ),
            _arc(
                rng.uniform(
                    1.25,
                    2.10,
                ),
                sign
                * math.radians(
                    rng.uniform(
                        40.0,
                        100.0,
                    )
                ),
            ),
            _straight(
                rng.uniform(
                    1.0,
                    2.6,
                )
            ),
        )

    if family == "s_curve":
        angle_1 = math.radians(
            rng.uniform(
                45.0,
                95.0,
            )
        )

        angle_2 = math.radians(
            rng.uniform(
                45.0,
                95.0,
            )
        )

        return (
            _straight(
                rng.uniform(
                    0.6,
                    1.2,
                )
            ),
            _arc(
                rng.uniform(
                    0.95,
                    1.55,
                ),
                sign * angle_1,
            ),
            _straight(
                rng.uniform(
                    0.45,
                    0.90,
                )
            ),
            _arc(
                rng.uniform(
                    0.95,
                    1.55,
                ),
                -sign * angle_2,
            ),
            _straight(
                rng.uniform(
                    1.0,
                    2.3,
                )
            ),
        )

    if family == "chicane":
        angle = math.radians(
            rng.uniform(
                30.0,
                58.0,
            )
        )

        radius = rng.uniform(
            0.85,
            1.25,
        )

        return (
            _straight(
                rng.uniform(
                    0.7,
                    1.2,
                )
            ),
            _arc(
                radius,
                sign * angle,
            ),
            _straight(
                rng.uniform(
                    0.30,
                    0.65,
                )
            ),
            _arc(
                radius,
                -2.0 * sign * angle,
            ),
            _straight(
                rng.uniform(
                    0.30,
                    0.65,
                )
            ),
            _arc(
                radius,
                sign * angle,
            ),
            _straight(
                rng.uniform(
                    0.9,
                    1.8,
                )
            ),
        )

    if family == "u_turn":
        return (
            _straight(
                rng.uniform(
                    1.0,
                    1.8,
                )
            ),
            _arc(
                rng.uniform(
                    0.90,
                    1.40,
                ),
                sign
                * math.radians(
                    rng.uniform(
                        150.0,
                        180.0,
                    )
                ),
            ),
            _straight(
                rng.uniform(
                    1.2,
                    2.4,
                )
            ),
        )

    if family == "hairpin":
        return (
            _straight(
                rng.uniform(
                    1.2,
                    2.0,
                )
            ),
            _arc(
                rng.uniform(
                    0.78,
                    1.05,
                ),
                sign
                * math.radians(
                    rng.uniform(
                        178.0,
                        202.0,
                    )
                ),
            ),
            _straight(
                rng.uniform(
                    1.5,
                    2.8,
                )
            ),
        )

    if family == "compound":
        first = math.radians(
            rng.uniform(
                55.0,
                85.0,
            )
        )

        second = math.radians(
            rng.uniform(
                95.0,
                135.0,
            )
        )

        third = math.radians(
            rng.uniform(
                45.0,
                75.0,
            )
        )

        return (
            _straight(
                rng.uniform(
                    0.7,
                    1.2,
                )
            ),
            _arc(
                rng.uniform(
                    0.95,
                    1.35,
                ),
                sign * first,
            ),
            _straight(
                rng.uniform(
                    0.65,
                    1.15,
                )
            ),
            _arc(
                rng.uniform(
                    0.90,
                    1.30,
                ),
                -sign * second,
            ),
            _straight(
                rng.uniform(
                    0.70,
                    1.20,
                )
            ),
            _arc(
                rng.uniform(
                    0.95,
                    1.40,
                ),
                sign * third,
            ),
            _straight(
                rng.uniform(
                    1.2,
                    2.2,
                )
            ),
        )

    raise ValueError(
        f"Unknown MARL course family: {family}"
    )


def _integrate_route(
    primitives: Iterable[
        tuple
    ],
    *,
    spacing_m: float = 0.16,
) -> tuple[
    tuple[float, float],
    ...,
]:
    if (
        not math.isfinite(
            spacing_m
        )
        or spacing_m <= 0.0
    ):
        raise ValueError(
            "Route spacing must be positive"
        )

    x_m = 0.0
    y_m = 0.0
    heading = 0.0

    points = [
        (
            x_m,
            y_m,
        )
    ]

    for primitive in primitives:
        kind = primitive[0]

        if kind == "straight":
            length_m = float(
                primitive[1]
            )

            steps = max(
                1,
                int(
                    math.ceil(
                        length_m
                        / spacing_m
                    )
                ),
            )

            distance = (
                length_m
                / steps
            )

            for _ in range(
                steps
            ):
                x_m += (
                    distance
                    * math.cos(
                        heading
                    )
                )
                y_m += (
                    distance
                    * math.sin(
                        heading
                    )
                )

                points.append(
                    (
                        x_m,
                        y_m,
                    )
                )

            continue

        if kind == "arc":
            radius_m = float(
                primitive[1]
            )

            angle_rad = float(
                primitive[2]
            )

            if radius_m <= 0.0:
                raise ValueError(
                    "Arc radius must be positive"
                )

            sign = (
                +1.0
                if angle_rad >= 0.0
                else -1.0
            )

            arc_length = (
                radius_m
                * abs(angle_rad)
            )

            steps = max(
                2,
                int(
                    math.ceil(
                        arc_length
                        / spacing_m
                    )
                ),
            )

            start_heading = (
                heading
            )

            center_x = (
                x_m
                - sign
                * radius_m
                * math.sin(
                    start_heading
                )
            )

            center_y = (
                y_m
                + sign
                * radius_m
                * math.cos(
                    start_heading
                )
            )

            for step in range(
                1,
                steps + 1,
            ):
                fraction = (
                    step
                    / steps
                )

                new_heading = (
                    start_heading
                    + angle_rad
                    * fraction
                )

                x_m = (
                    center_x
                    + sign
                    * radius_m
                    * math.sin(
                        new_heading
                    )
                )

                y_m = (
                    center_y
                    - sign
                    * radius_m
                    * math.cos(
                        new_heading
                    )
                )

                points.append(
                    (
                        x_m,
                        y_m,
                    )
                )

            heading = (
                start_heading
                + angle_rad
            )

            continue

        raise ValueError(
            f"Unknown route primitive: {kind!r}"
        )

    if len(points) < 2:
        raise ValueError(
            "Generated route is empty"
        )

    return tuple(
        (
            float(x),
            float(y),
        )
        for x, y
        in points
    )


def _road_width(
    identity: MarlCurriculumIdentity,
    rng: random.Random,
) -> float:
    if (
        identity.split
        == "challenge"
    ):
        return rng.uniform(
            0.90,
            1.00,
        )

    if (
        identity.difficulty
        == "easy"
    ):
        return rng.uniform(
            1.10,
            1.26,
        )

    if (
        identity.difficulty
        == "medium"
    ):
        return rng.uniform(
            1.00,
            1.18,
        )

    return rng.uniform(
        0.92,
        1.08,
    )


def _cone_count(
    identity: MarlCurriculumIdentity,
    rng: random.Random,
) -> int:
    if (
        identity.split
        == "challenge"
    ):
        return rng.randint(
            10,
            14,
        )

    if (
        identity.difficulty
        == "easy"
    ):
        return rng.randint(
            3,
            5,
        )

    if (
        identity.difficulty
        == "medium"
    ):
        return rng.randint(
            5,
            8,
        )

    return rng.randint(
        8,
        12,
    )


def _navigation_cones(
    centerline_xy_m: tuple[
        tuple[float, float],
        ...,
    ],
    *,
    road_width_m: float,
    identity: MarlCurriculumIdentity,
    rng: random.Random,
) -> tuple[
    Phase1Cone,
    ...,
]:
    requested = _cone_count(
        identity,
        rng,
    )

    # Keep start / target platforms uncluttered.
    first = max(
        3,
        int(
            0.10
            * len(
                centerline_xy_m
            )
        ),
    )

    last = min(
        len(
            centerline_xy_m
        )
        - 4,
        int(
            0.90
            * len(
                centerline_xy_m
            )
        ),
    )

    candidates = list(
        range(
            first,
            last + 1,
        )
    )

    count = min(
        requested,
        len(candidates),
    )

    indexes = sorted(
        rng.sample(
            candidates,
            count,
        )
    )

    minimum_offset = (
        0.5
        * VEHICLE_WIDTH_M
        + CONE_RADIUS_M
        + CLEARANCE_MARGIN_M
        + 0.015
    )

    maximum_offset = (
        0.5
        * road_width_m
        - CONE_RADIUS_M
        - 0.035
    )

    if (
        maximum_offset
        <= minimum_offset
    ):
        raise ValueError(
            "Road is too narrow for safe "
            "Phase-1 cone placement"
        )

    # Some seeds intentionally favour one side. This produces asymmetric
    # obstacle fields instead of expert-like balanced cone pairs.
    preferred_side = (
        +1.0
        if rng.random() < 0.5
        else -1.0
    )

    bias_probability = (
        0.62
        if identity.difficulty
        != "easy"
        else 0.55
    )

    cones = []

    for index in indexes:
        previous = (
            centerline_xy_m[
                max(
                    0,
                    index - 1,
                )
            ]
        )

        following = (
            centerline_xy_m[
                min(
                    len(
                        centerline_xy_m
                    )
                    - 1,
                    index + 1,
                )
            ]
        )

        dx = (
            following[0]
            - previous[0]
        )

        dy = (
            following[1]
            - previous[1]
        )

        norm = math.hypot(
            dx,
            dy,
        )

        if norm <= 1.0e-9:
            continue

        normal_x = (
            -dy / norm
        )

        normal_y = (
            dx / norm
        )

        side = (
            preferred_side
            if rng.random()
            < bias_probability
            else -preferred_side
        )

        offset = rng.uniform(
            minimum_offset,
            maximum_offset,
        )

        center = (
            centerline_xy_m[
                index
            ]
        )

        cones.append(
            Phase1Cone(
                center_xy_m=(
                    center[0]
                    + side
                    * offset
                    * normal_x,
                    center[1]
                    + side
                    * offset
                    * normal_y,
                ),
                radius_m=(
                    CONE_RADIUS_M
                ),
                height_m=(
                    CONE_HEIGHT_M
                ),
                static=True,
            )
        )

    return tuple(
        cones
    )


def _final_yaw(
    centerline_xy_m: tuple[
        tuple[float, float],
        ...,
    ],
) -> float:
    left = centerline_xy_m[
        -2
    ]

    right = centerline_xy_m[
        -1
    ]

    return math.atan2(
        right[1] - left[1],
        right[0] - left[0],
    )


def _platform_box(
    *,
    name: str,
    platform: OrientedPlatform,
    semantic: str,
    target: bool,
) -> CourseBox:
    thickness_m = 0.02

    return CourseBox(
        name=name,
        center_xyz_m=(
            platform.center_xy_m[0],
            platform.center_xy_m[1],
            -0.5
            * thickness_m,
        ),
        size_xyz_m=(
            platform.size_xy_m[0],
            platform.size_xy_m[1],
            thickness_m,
        ),
        color_rgb=(
            (
                0.24,
                0.31,
                0.24,
            )
            if target
            else (
                0.24,
                0.27,
                0.31,
            )
        ),
        collidable=True,
        semantic=semantic,
        yaw_deg=math.degrees(
            platform.yaw_rad
        ),
    )


def phase1_marl_curriculum_course(
    seed: int,
    *,
    split: str,
) -> Phase1TrainingCourse:
    """Generate one structurally-unseen RC-Car8 MARL stage."""

    identity = (
        curriculum_identity(
            int(seed),
            split=split,
        )
    )

    rng = _rng(
        identity
    )

    primitives = (
        _route_primitives(
            identity,
            rng,
        )
    )

    centerline = (
        _integrate_route(
            primitives
        )
    )

    road_width_m = (
        _road_width(
            identity,
            rng,
        )
    )

    required_opening = (
        VEHICLE_WIDTH_M
        + 2.0
        * CLEARANCE_MARGIN_M
    )

    if (
        road_width_m
        <= required_opening
    ):
        raise ValueError(
            "Generated road violates RC-Car8 "
            "clearance contract"
        )

    start = OrientedPlatform(
        center_xy_m=(
            0.0,
            0.0,
        ),
        yaw_rad=0.0,
        size_xy_m=(
            PLATFORM_SIZE_XY_M
        ),
    )

    target = OrientedPlatform(
        center_xy_m=(
            centerline[-1]
        ),
        yaw_rad=(
            _final_yaw(
                centerline
            )
        ),
        size_xy_m=(
            PLATFORM_SIZE_XY_M
        ),
    )

    start_box = _platform_box(
        name=(
            "MarlStartPlatform"
        ),
        platform=start,
        semantic=(
            "phase1_start_platform"
        ),
        target=False,
    )

    target_box = _platform_box(
        name=(
            "MarlTargetPlatform"
        ),
        platform=target,
        semantic=(
            "phase1_target_platform"
        ),
        target=True,
    )

    road_boxes = (
        _road_support_boxes(
            centerline,
            road_width_m=(
                road_width_m
            ),
        )
    )

    cones = (
        _navigation_cones(
            centerline,
            road_width_m=(
                road_width_m
            ),
            identity=identity,
            rng=rng,
        )
    )

    profile = (
        f"marl_unseen_v1/"
        f"{identity.split}/"
        f"{identity.difficulty}/"
        f"{identity.family}"
    )

    return Phase1TrainingCourse(
        seed=identity.seed,

        start_platform=start,
        target_platform=target,

        support_boxes=(
            start_box,
            *road_boxes,
            target_box,
        ),

        cones=cones,

        centerline_xy_m=(
            centerline
        ),

        road_width_m=(
            road_width_m
        ),

        vehicle_length_m=(
            VEHICLE_LENGTH_M
        ),
        vehicle_width_m=(
            VEHICLE_WIDTH_M
        ),

        cone_radius_m=(
            CONE_RADIUS_M
        ),
        collision_margin_m=(
            COLLISION_MARGIN_M
        ),
        clearance_margin_m=(
            CLEARANCE_MARGIN_M
        ),

        has_curve=(
            identity.family
            != "straight"
        ),

        track_profile=(
            profile
        ),

        forward_only=True,
    )


def install_phase1_marl_curriculum_course(
    stage,
    *,
    seed: int,
    split: str,
    box_installer=None,
    cone_installer=None,
) -> Phase1TrainingCourse:
    """Materialize one unseen MARL curriculum stage into Isaac USD."""

    course = (
        phase1_marl_curriculum_course(
            int(seed),
            split=str(split),
        )
    )

    if box_installer is None:
        from .obstacle_course import (
            _install_course_boxes,
        )

        box_installer = (
            _install_course_boxes
        )

    if cone_installer is None:
        from .phase1_training_course import (
            _install_phase1_static_cones,
        )

        cone_installer = (
            _install_phase1_static_cones
        )

    box_installer(
        stage,
        "/World/Phase1MarlCurriculum/Supports",
        course.support_boxes,
    )

    cone_installer(
        stage,
        "/World/Phase1MarlCurriculum/NavigationCones",
        course.cones,
    )

    return course


# ============================================================
# TASK4E3_DENSE_HAZARDS_AND_GRADES_V1
# ============================================================
#
# The original 4E1 generator remains the planar layout backbone.
#
# This hardened layer adds:
# - fixed obstacle density by curriculum difficulty;
# - local four-cone hazard zones;
# - a narrow but physically valid gate in every zone;
# - hard-only smooth hill / dip elevation profiles;
# - flat assembly and goal platforms;
# - no reuse of Expert geometry.
# ============================================================

_TASK4E3_BASE_GENERATOR = (
    phase1_marl_curriculum_course
)


def _task4e3_grade_kind(
    identity: MarlCurriculumIdentity,
) -> str:
    """Assign vertical variation only to hard/challenge stages."""

    if identity.split == "challenge":
        return (
            "hill"
            if identity.ordinal % 2 == 0
            else "dip"
        )

    if identity.difficulty != "hard":
        return "flat"

    selector = (
        identity.ordinal
        % 4
    )

    if selector == 1:
        return "hill"

    if selector == 3:
        return "dip"

    return "flat"


def _task4e3_arc_lengths(
    centerline_xy_m,
):
    cumulative = [
        0.0
    ]

    for left, right in zip(
        centerline_xy_m[:-1],
        centerline_xy_m[1:],
    ):
        cumulative.append(
            cumulative[-1]
            + math.hypot(
                right[0] - left[0],
                right[1] - left[1],
            )
        )

    return tuple(
        cumulative
    )


def _task4e3_elevations(
    centerline_xy_m,
    *,
    grade: str,
    rng: random.Random,
    challenge: bool,
):
    """Smooth hill/dip returning to the original z=0 level."""

    if grade == "flat":
        return tuple(
            0.0
            for _ in centerline_xy_m
        )

    cumulative = (
        _task4e3_arc_lengths(
            centerline_xy_m
        )
    )

    total_length = cumulative[-1]

    if total_length <= 0.0:
        raise ValueError(
            "Cannot grade an empty MARL route"
        )

    # Middle section only.
    #
    # Assembly, early obstacle region, late obstacle region and
    # target remain level.
    start_fraction = 0.34
    end_fraction = 0.66

    active_length = (
        total_length
        * (
            end_fraction
            - start_fraction
        )
    )

    maximum_grade_deg = (
        rng.uniform(
            9.0,
            12.0,
        )
        if challenge
        else rng.uniform(
            8.0,
            11.0,
        )
    )

    # z(s) = A sin²(pi u)
    #
    # max slope ~= A*pi/L.
    amplitude_m = (
        math.tan(
            math.radians(
                maximum_grade_deg
            )
        )
        * active_length
        / math.pi
    )

    amplitude_m = min(
        amplitude_m,
        0.24,
    )

    sign = (
        +1.0
        if grade == "hill"
        else -1.0
    )

    elevations = []

    for distance_m in cumulative:
        fraction = (
            distance_m
            / total_length
        )

        if (
            fraction <= start_fraction
            or fraction >= end_fraction
        ):
            elevations.append(
                0.0
            )

            continue

        u = (
            (
                fraction
                - start_fraction
            )
            / (
                end_fraction
                - start_fraction
            )
        )

        elevation = (
            sign
            * amplitude_m
            * (
                math.sin(
                    math.pi
                    * u
                )
                ** 2
            )
        )

        elevations.append(
            float(
                elevation
            )
        )

    elevations[0] = 0.0
    elevations[-1] = 0.0

    return tuple(
        elevations
    )


def _task4e3_graded_road_boxes(
    centerline_xy_m,
    elevations_m,
    *,
    road_width_m: float,
):
    """Create genuinely pitched physical road tiles."""

    if (
        len(centerline_xy_m)
        != len(elevations_m)
    ):
        raise ValueError(
            "Centerline/elevation size mismatch"
        )

    boxes = []

    thickness_m = 0.02
    overlap_m = 0.045

    for index, (
        left,
        right,
        z_left,
        z_right,
    ) in enumerate(
        zip(
            centerline_xy_m[:-1],
            centerline_xy_m[1:],
            elevations_m[:-1],
            elevations_m[1:],
        )
    ):
        dx = (
            right[0]
            - left[0]
        )

        dy = (
            right[1]
            - left[1]
        )

        horizontal_length = (
            math.hypot(
                dx,
                dy,
            )
        )

        if (
            horizontal_length
            <= 1.0e-9
        ):
            continue

        dz = (
            z_right
            - z_left
        )

        physical_length = (
            math.hypot(
                horizontal_length,
                dz,
            )
        )

        yaw_deg = math.degrees(
            math.atan2(
                dy,
                dx,
            )
        )

        # Existing CourseBox convention:
        # positive world-Z slope along local +X uses negative pitch.
        pitch_rad = -math.atan2(
            dz,
            horizontal_length,
        )

        pitch_deg = math.degrees(
            pitch_rad
        )

        surface_z = (
            0.5
            * (
                z_left
                + z_right
            )
        )

        center_z = (
            surface_z
            - 0.5
            * thickness_m
            * math.cos(
                pitch_rad
            )
        )

        boxes.append(
            CourseBox(
                name=(
                    f"Phase1Road"
                    f"{index:03d}"
                ),
                center_xyz_m=(
                    0.5
                    * (
                        left[0]
                        + right[0]
                    ),
                    0.5
                    * (
                        left[1]
                        + right[1]
                    ),
                    center_z,
                ),
                size_xyz_m=(
                    physical_length
                    + overlap_m,
                    road_width_m,
                    thickness_m,
                ),
                color_rgb=(
                    0.24,
                    0.27,
                    0.31,
                ),
                collidable=True,
                semantic="phase1_road",
                pitch_deg=pitch_deg,
                yaw_deg=yaw_deg,
            )
        )

    if not boxes:
        raise ValueError(
            "Generated graded road is empty"
        )

    return tuple(
        boxes
    )


def _task4e3_pose_at_fraction(
    centerline_xy_m,
    *,
    fraction: float,
):
    """Sample position and tangent by arc-length fraction."""

    cumulative = (
        _task4e3_arc_lengths(
            centerline_xy_m
        )
    )

    total_length = (
        cumulative[-1]
    )

    distance = (
        min(
            1.0,
            max(
                0.0,
                float(
                    fraction
                ),
            ),
        )
        * total_length
    )

    for index in range(
        len(centerline_xy_m)
        - 1
    ):
        s0 = cumulative[
            index
        ]

        s1 = cumulative[
            index + 1
        ]

        if (
            distance > s1
            and index
            < len(
                centerline_xy_m
            )
            - 2
        ):
            continue

        left = (
            centerline_xy_m[
                index
            ]
        )

        right = (
            centerline_xy_m[
                index + 1
            ]
        )

        segment_length = max(
            s1 - s0,
            1.0e-12,
        )

        alpha = min(
            1.0,
            max(
                0.0,
                (
                    distance
                    - s0
                )
                / segment_length,
            ),
        )

        x_m = (
            left[0]
            + alpha
            * (
                right[0]
                - left[0]
            )
        )

        y_m = (
            left[1]
            + alpha
            * (
                right[1]
                - left[1]
            )
        )

        dx = (
            right[0]
            - left[0]
        )

        dy = (
            right[1]
            - left[1]
        )

        norm = max(
            math.hypot(
                dx,
                dy,
            ),
            1.0e-12,
        )

        return (
            x_m,
            y_m,
            dx / norm,
            dy / norm,
            total_length,
        )

    raise RuntimeError(
        "Could not sample MARL route"
    )


def _task4e3_zone_fractions(
    identity: MarlCurriculumIdentity,
    *,
    grade: str,
):
    """Choose separated local obstacle zones."""

    if identity.split == "challenge":
        if grade == "flat":
            return (
                0.10,
                0.29,
                0.50,
                0.71,
                0.90,
            )

        return (
            0.08,
            0.18,
            0.28,
            0.78,
            0.90,
        )

    if identity.difficulty == "easy":
        return (
            0.25,
            0.73,
        )

    if identity.difficulty == "medium":
        return (
            0.19,
            0.50,
            0.81,
        )

    if grade == "flat":
        return (
            0.14,
            0.37,
            0.63,
            0.86,
        )

    # Keep graded centre cone-free for this first version.
    return (
        0.12,
        0.25,
        0.75,
        0.88,
    )


def _task4e3_dense_cones(
    course: Phase1TrainingCourse,
    *,
    identity: MarlCurriculumIdentity,
    grade: str,
    rng: random.Random,
):
    """Build deterministic dense local hazards without cone overlap.

    Every zone contains:
    - one narrow but physically valid two-cone gate;
    - one approach flank cone;
    - one exit flank cone.

    Flank placement is solved locally against already accepted cones,
    rather than assuming that a fixed arc-length separation is safe
    on every curved route.
    """

    zone_fractions = (
        _task4e3_zone_fractions(
            identity,
            grade=grade,
        )
    )

    minimum_side_offset = (
        0.5
        * course.vehicle_width_m
        + course.cone_radius_m
        + course.clearance_margin_m
        + 0.020
    )

    maximum_side_offset = (
        0.5
        * course.road_width_m
        - course.cone_radius_m
        - 0.040
    )

    if (
        maximum_side_offset
        <= minimum_side_offset
    ):
        raise ValueError(
            "Generated road too narrow "
            "for dense MARL hazards"
        )

    clearance_between_cones_m = (
        0.010
    )

    cones = []

    def make_cone(
        *,
        fraction: float,
        lateral_offset: float,
    ) -> Phase1Cone:
        (
            x_m,
            y_m,
            tangent_x,
            tangent_y,
            _,
        ) = _task4e3_pose_at_fraction(
            course.centerline_xy_m,
            fraction=fraction,
        )

        normal_x = -tangent_y
        normal_y = tangent_x

        return Phase1Cone(
            center_xy_m=(
                x_m
                + normal_x
                * lateral_offset,
                y_m
                + normal_y
                * lateral_offset,
            ),
            radius_m=(
                course.cone_radius_m
            ),
            height_m=(
                CONE_HEIGHT_M
            ),
            static=True,
        )

    def sufficiently_clear(
        candidate: Phase1Cone,
        others,
    ) -> bool:
        for existing in others:
            distance = math.hypot(
                candidate.center_xy_m[0]
                - existing.center_xy_m[0],
                candidate.center_xy_m[1]
                - existing.center_xy_m[1],
            )

            required = (
                candidate.radius_m
                + existing.radius_m
                + clearance_between_cones_m
            )

            if distance <= required:
                return False

        return True

    for (
        zone_index,
        nominal_fraction,
    ) in enumerate(
        zone_fractions
    ):
        # ====================================================
        # GATE
        # ====================================================

        physical_opening = (
            course.required_gate_opening_m
            + rng.uniform(
                0.035,
                0.080,
            )
        )

        half_center_spacing = (
            0.5
            * (
                physical_opening
                + 2.0
                * course.cone_radius_m
            )
        )

        nominal_gate_shift = (
            rng.uniform(
                -0.025,
                +0.025,
            )
        )

        gate_pair = None
        accepted_fraction = None

        # Curved/hairpin routes can bring distant route sections
        # spatially close. Permit a very small longitudinal shift
        # of the complete gate while preserving the hazard zone.
        gate_fraction_shifts = (
            0.0,
            +0.010,
            -0.010,
            +0.020,
            -0.020,
        )

        gate_lateral_shifts = (
            nominal_gate_shift,
            0.0,
            +0.020,
            -0.020,
        )

        for fraction_shift in gate_fraction_shifts:
            candidate_fraction = min(
                0.94,
                max(
                    0.06,
                    nominal_fraction
                    + fraction_shift,
                ),
            )

            for gate_shift in gate_lateral_shifts:
                first = make_cone(
                    fraction=(
                        candidate_fraction
                    ),
                    lateral_offset=(
                        gate_shift
                        + half_center_spacing
                    ),
                )

                second = make_cone(
                    fraction=(
                        candidate_fraction
                    ),
                    lateral_offset=(
                        gate_shift
                        - half_center_spacing
                    ),
                )

                if not sufficiently_clear(
                    first,
                    cones,
                ):
                    continue

                if not sufficiently_clear(
                    second,
                    (
                        *cones,
                        first,
                    ),
                ):
                    continue

                gate_pair = (
                    first,
                    second,
                )

                accepted_fraction = (
                    candidate_fraction
                )

                break

            if gate_pair is not None:
                break

        if gate_pair is None:
            raise ValueError(
                "Could not place valid MARL gate: "
                f"seed={identity.seed}, "
                f"zone={zone_index}"
            )

        cones.extend(
            gate_pair
        )

        # ====================================================
        # FLANK CONES
        # ====================================================
        #
        # Search is deterministic.  It first preserves the
        # intended side pattern, then tries the opposite side,
        # several safe lateral offsets and several local
        # longitudinal distances.
        #
        # Maximum distance is kept at 0.30 m so graded courses'
        # flank cones remain outside the central hill/dip zone.
        # ====================================================

        primary_leading_side = (
            +1.0
            if (
                identity.seed
                + zone_index
            )
            % 2
            == 0
            else -1.0
        )

        primary_trailing_side = (
            -primary_leading_side
            if rng.random() < 0.55
            else primary_leading_side
        )

        base_leading_offset = (
            rng.uniform(
                minimum_side_offset,
                maximum_side_offset,
            )
        )

        base_trailing_offset = (
            rng.uniform(
                minimum_side_offset,
                maximum_side_offset,
            )
        )

        middle_offset = (
            0.5
            * (
                minimum_side_offset
                + maximum_side_offset
            )
        )

        def place_flank(
            *,
            direction: float,
            primary_side: float,
            preferred_offset: float,
        ) -> Phase1Cone:
            # Arc-length distances, not raw fractions.
            distances_m = (
                0.22,
                0.26,
                0.30,
            )

            sides = (
                primary_side,
                -primary_side,
            )

            magnitudes = (
                preferred_offset,
                maximum_side_offset,
                middle_offset,
                minimum_side_offset,
            )

            (
                _,
                _,
                _,
                _,
                total_length,
            ) = _task4e3_pose_at_fraction(
                course.centerline_xy_m,
                fraction=(
                    accepted_fraction
                ),
            )

            for distance_m in distances_m:
                fraction_delta = (
                    distance_m
                    / max(
                        total_length,
                        1.0e-9,
                    )
                )

                candidate_fraction = (
                    accepted_fraction
                    + direction
                    * fraction_delta
                )

                if not (
                    0.04
                    < candidate_fraction
                    < 0.96
                ):
                    continue

                for side in sides:
                    for magnitude in magnitudes:
                        candidate = make_cone(
                            fraction=(
                                candidate_fraction
                            ),
                            lateral_offset=(
                                side
                                * magnitude
                            ),
                        )

                        if sufficiently_clear(
                            candidate,
                            cones,
                        ):
                            return candidate

            raise ValueError(
                "Could not place MARL flank cone: "
                f"seed={identity.seed}, "
                f"zone={zone_index}, "
                f"direction={direction:+.0f}"
            )

        leading = place_flank(
            direction=-1.0,
            primary_side=(
                primary_leading_side
            ),
            preferred_offset=(
                base_leading_offset
            ),
        )

        cones.append(
            leading
        )

        trailing = place_flank(
            direction=+1.0,
            primary_side=(
                primary_trailing_side
            ),
            preferred_offset=(
                base_trailing_offset
            ),
        )

        cones.append(
            trailing
        )

    # ========================================================
    # FINAL FAIL-CLOSED GEOMETRY CHECK
    # ========================================================

    for left_index in range(
        len(cones)
    ):
        for right_index in range(
            left_index + 1,
            len(cones),
        ):
            left = cones[
                left_index
            ]

            right = cones[
                right_index
            ]

            distance = math.hypot(
                left.center_xy_m[0]
                - right.center_xy_m[0],
                left.center_xy_m[1]
                - right.center_xy_m[1],
            )

            required = (
                left.radius_m
                + right.radius_m
                + clearance_between_cones_m
            )

            if distance <= required:
                raise ValueError(
                    "Dense MARL hazard overlap "
                    "after placement solver: "
                    f"seed={identity.seed}, "
                    f"cones={left_index},"
                    f"{right_index}, "
                    f"distance={distance:.4f}, "
                    f"required>{required:.4f}"
                )

    expected_count = (
        20
        if identity.split
        == "challenge"
        else {
            "easy": 8,
            "medium": 12,
            "hard": 16,
        }[
            identity.difficulty
        ]
    )

    if len(cones) != expected_count:
        raise AssertionError(
            "Dense MARL hazard count mismatch: "
            f"{len(cones)} != {expected_count}"
        )

    return tuple(
        cones
    )

def phase1_marl_curriculum_course(
    seed: int,
    *,
    split: str,
) -> Phase1TrainingCourse:
    """Hardened unseen curriculum used by Phase-1 MARL."""

    from dataclasses import replace

    identity = (
        curriculum_identity(
            int(seed),
            split=split,
        )
    )

    base = (
        _TASK4E3_BASE_GENERATOR(
            int(seed),
            split=split,
        )
    )

    rng = random.Random(
        (
            int(seed)
            * 1_000_003
        )
        ^ 0x4E3D3E
    )

    grade = (
        _task4e3_grade_kind(
            identity
        )
    )

    elevations = (
        _task4e3_elevations(
            base.centerline_xy_m,
            grade=grade,
            rng=rng,
            challenge=(
                identity.split
                == "challenge"
            ),
        )
    )

    road_boxes = (
        _task4e3_graded_road_boxes(
            base.centerline_xy_m,
            elevations,
            road_width_m=(
                base.road_width_m
            ),
        )
    )

    cones = (
        _task4e3_dense_cones(
            base,
            identity=identity,
            grade=grade,
            rng=rng,
        )
    )

    return replace(
        base,
        support_boxes=(
            base.support_boxes[0],
            *road_boxes,
            base.support_boxes[-1],
        ),
        cones=cones,
        track_profile=(
            f"{base.track_profile}"
            f"/grade_{grade}"
        ),
    )


# ============================================================
# TASK4E4_CENTERLINE_BLOCKERS_V1
# ============================================================
#
# 4E3 solved obstacle density and physical validity.
#
# 4E4 changes the MOTIF:
#
#   NOT:
#       cone     cone
#          \   /
#         straight gate
#
#   BUT:
#
#       guide
#          \
#        CENTER BLOCKER
#              \
#               offset guide
#                    \
#                     correction cone
#
# Obstacles are separated longitudinally so a centerline cone
# never has lateral cones beside it at the same road section.
# Bypass direction alternates between hazard zones.
# ============================================================

_TASK4E4_BASE_GENERATOR = (
    phase1_marl_curriculum_course
)


def _task4e4_clear(
    candidate: Phase1Cone,
    accepted,
    *,
    clearance_m: float = 0.010,
) -> bool:
    for existing in accepted:
        distance = math.hypot(
            candidate.center_xy_m[0]
            - existing.center_xy_m[0],
            candidate.center_xy_m[1]
            - existing.center_xy_m[1],
        )

        required = (
            candidate.radius_m
            + existing.radius_m
            + clearance_m
        )

        if distance <= required:
            return False

    return True


def _task4e4_make_cone(
    course,
    *,
    fraction: float,
    lateral_offset: float,
):
    (
        x_m,
        y_m,
        tangent_x,
        tangent_y,
        _,
    ) = _task4e3_pose_at_fraction(
        course.centerline_xy_m,
        fraction=fraction,
    )

    normal_x = -tangent_y
    normal_y = tangent_x

    return Phase1Cone(
        center_xy_m=(
            x_m
            + normal_x
            * lateral_offset,
            y_m
            + normal_y
            * lateral_offset,
        ),
        radius_m=(
            course.cone_radius_m
        ),
        height_m=(
            CONE_HEIGHT_M
        ),
        static=True,
    )


def _task4e4_place(
    course,
    accepted,
    *,
    nominal_fraction: float,
    nominal_lateral: float,
    center_blocker: bool,
):
    """Solve one cone while preserving its intended motif."""

    fraction_shifts = (
        0.0,
        +0.008,
        -0.008,
        +0.016,
        -0.016,
    )

    if center_blocker:
        lateral_candidates = (
            nominal_lateral,
            0.0,
            +0.015,
            -0.015,
            +0.025,
            -0.025,
        )
    else:
        sign = (
            +1.0
            if nominal_lateral >= 0.0
            else -1.0
        )

        magnitude = abs(
            nominal_lateral
        )

        lateral_candidates = (
            nominal_lateral,
            sign
            * max(
                0.12,
                magnitude
                - 0.035,
            ),
            sign
            * (
                magnitude
                + 0.035
            ),
        )

    for fraction_shift in fraction_shifts:
        fraction = min(
            0.96,
            max(
                0.04,
                nominal_fraction
                + fraction_shift,
            ),
        )

        for lateral in lateral_candidates:
            candidate = (
                _task4e4_make_cone(
                    course,
                    fraction=fraction,
                    lateral_offset=lateral,
                )
            )

            if _task4e4_clear(
                candidate,
                accepted,
            ):
                return candidate

    raise ValueError(
        "Could not solve 4E4 obstacle placement"
    )


def _task4e4_zone_fractions(
    identity: MarlCurriculumIdentity,
    *,
    grade: str,
):
    """Spread centerline hazards across the usable route.

    This distribution belongs exclusively to 4E4/4E5.
    It must not modify the legacy 4E3 dense-gate generator.
    """

    if identity.split == "challenge":
        if grade == "flat":
            return (
                0.08,
                0.28,
                0.50,
                0.72,
                0.92,
            )

        return (
            0.06,
            0.18,
            0.30,
            0.78,
            0.92,
        )

    if identity.difficulty == "easy":
        return (
            0.20,
            0.80,
        )

    if identity.difficulty == "medium":
        return (
            0.15,
            0.50,
            0.85,
        )

    if grade == "flat":
        return (
            0.10,
            0.36,
            0.64,
            0.90,
        )

    # Keep the steepest middle portion cone-free for now.
    return (
        0.08,
        0.30,
        0.70,
        0.92,
    )


def _task4e4_centerline_hazards(
    course,
    *,
    identity,
    grade,
    rng,
):
    """Distribute navigation hazards across all usable flat road.

    The final curriculum no longer builds compact four-cone clusters.

    Instead:
    - every cone receives its own longitudinal position;
    - the full usable road length is exploited;
    - every four-cone sequence contains one centerline blocker;
    - bypass direction alternates;
    - hill/dip sections remain cone-free until cone Z placement is
      explicitly upgraded to 3-D terrain coordinates.
    """

    expected_count = (
        20
        if identity.split == "challenge"
        else {
            "easy": 8,
            "medium": 12,
            "hard": 16,
        }[
            identity.difficulty
        ]
    )

    cumulative = (
        _task4e3_arc_lengths(
            course.centerline_xy_m
        )
    )

    total_length = (
        cumulative[-1]
    )

    if total_length <= 1.0:
        raise ValueError(
            "MARL route is too short for distributed hazards"
        )

    # ========================================================
    # USABLE FLAT ARC-LENGTH INTERVALS
    # ========================================================

    edge_margin_m = 0.24

    if grade == "flat":
        intervals = [
            (
                edge_margin_m,
                total_length
                - edge_margin_m,
            )
        ]

    else:
        # Grade itself lives in [0.34, 0.66].
        #
        # Keep an additional physical buffer so nearest road tiles
        # remain effectively horizontal as required by the 4E3
        # collision model.
        grade_buffer_m = 0.22

        left_end = (
            0.34
            * total_length
            - grade_buffer_m
        )

        right_start = (
            0.66
            * total_length
            + grade_buffer_m
        )

        intervals = [
            (
                edge_margin_m,
                left_end,
            ),
            (
                right_start,
                total_length
                - edge_margin_m,
            ),
        ]

    intervals = [
        (
            float(start_m),
            float(end_m),
        )
        for start_m, end_m
        in intervals
        if (
            end_m
            - start_m
            > 0.15
        )
    ]

    if not intervals:
        raise ValueError(
            "No usable flat road remains "
            "for MARL hazards"
        )

    usable_length = sum(
        end_m - start_m
        for start_m, end_m
        in intervals
    )

    if (
        usable_length
        < expected_count
        * 0.105
    ):
        raise ValueError(
            "Insufficient flat arc length "
            "for requested MARL cone density: "
            f"seed={identity.seed}, "
            f"usable={usable_length:.3f}, "
            f"cones={expected_count}"
        )

    # ========================================================
    # EVENLY SAMPLE ALL USABLE ROAD
    # ========================================================

    target_distances = []

    for index in range(
        expected_count
    ):
        target = (
            (
                index
                + 0.5
            )
            / expected_count
            * usable_length
        )

        cursor = 0.0

        selected = None

        for (
            start_m,
            end_m,
        ) in intervals:
            span = (
                end_m
                - start_m
            )

            if (
                target
                <= cursor
                + span
            ):
                selected = (
                    start_m
                    + (
                        target
                        - cursor
                    )
                )

                break

            cursor += span

        if selected is None:
            selected = (
                intervals[-1][1]
            )

        target_distances.append(
            float(selected)
        )

    def distance_is_allowed(
        distance_m: float,
    ) -> bool:
        return any(
            (
                start_m
                <= distance_m
                <= end_m
            )
            for start_m, end_m
            in intervals
        )

    # ========================================================
    # LATERAL MOTIF
    # ========================================================

    minimum_side = (
        0.5
        * course.vehicle_width_m
        + course.cone_radius_m
        + 0.035
    )

    maximum_side = (
        0.5
        * course.road_width_m
        - course.cone_radius_m
        - 0.055
    )

    if (
        maximum_side
        <= minimum_side
    ):
        raise ValueError(
            "Road too narrow for distributed hazards"
        )

    middle_side = (
        0.5
        * (
            minimum_side
            + maximum_side
        )
    )

    accepted = []

    def make_at_distance(
        distance_m: float,
        lateral_offset: float,
    ):
        fraction = (
            distance_m
            / total_length
        )

        return _task4e4_make_cone(
            course,
            fraction=fraction,
            lateral_offset=(
                lateral_offset
            ),
        )

    def solve_position(
        *,
        target_distance_m: float,
        nominal_lateral: float,
        center_blocker: bool,
    ):
        # Small deterministic local corrections only.
        # The cones remain globally distributed rather than collapsing
        # back into clusters.
        longitudinal_shifts_m = (
            0.0,
            +0.035,
            -0.035,
            +0.070,
            -0.070,
        )

        if center_blocker:
            lateral_candidates = (
                nominal_lateral,
                0.0,
                +0.012,
                -0.012,
                +0.025,
                -0.025,
            )
        else:
            sign = (
                +1.0
                if nominal_lateral >= 0.0
                else -1.0
            )

            magnitude = abs(
                nominal_lateral
            )

            lateral_candidates = (
                nominal_lateral,
                sign
                * middle_side,
                sign
                * minimum_side,
                sign
                * maximum_side,
                -sign
                * middle_side,
            )

        for shift_m in longitudinal_shifts_m:
            distance_m = (
                target_distance_m
                + shift_m
            )

            if not distance_is_allowed(
                distance_m
            ):
                continue

            for lateral in lateral_candidates:
                candidate = (
                    make_at_distance(
                        distance_m,
                        lateral,
                    )
                )

                if _task4e4_clear(
                    candidate,
                    accepted,
                    clearance_m=0.010,
                ):
                    return candidate

        raise ValueError(
            "Could not place distributed MARL cone: "
            f"seed={identity.seed}, "
            f"distance={target_distance_m:.3f}, "
            f"center={center_blocker}"
        )

    for index, target_distance_m in enumerate(
        target_distances
    ):
        group = (
            index // 4
        )

        position_in_group = (
            index % 4
        )

        bypass_side = (
            +1.0
            if (
                identity.seed
                + group
            )
            % 2
            == 0
            else -1.0
        )

        # Sequence:
        #
        # 0   opposite-side guide
        # 1   CENTERLINE BLOCKER
        # 2   bypass-side correction
        # 3   opposite-side correction
        #
        # Because longitudinal positions are globally distributed,
        # these no longer form one compact four-cone blob.
        if position_in_group == 0:
            lateral = (
                -bypass_side
                * rng.uniform(
                    middle_side,
                    maximum_side,
                )
            )

            center_blocker = False

        elif position_in_group == 1:
            lateral = (
                rng.uniform(
                    -0.015,
                    +0.015,
                )
            )

            center_blocker = True

        elif position_in_group == 2:
            lateral = (
                bypass_side
                * rng.uniform(
                    minimum_side,
                    middle_side,
                )
            )

            center_blocker = False

        else:
            lateral = (
                -bypass_side
                * rng.uniform(
                    minimum_side,
                    middle_side,
                )
            )

            center_blocker = False

        accepted.append(
            solve_position(
                target_distance_m=(
                    target_distance_m
                ),
                nominal_lateral=(
                    lateral
                ),
                center_blocker=(
                    center_blocker
                ),
            )
        )

    if (
        len(accepted)
        != expected_count
    ):
        raise AssertionError(
            "Distributed MARL cone count mismatch"
        )

    return tuple(
        accepted
    )

def phase1_marl_curriculum_course(
    seed: int,
    *,
    split: str,
) -> Phase1TrainingCourse:
    """Final Phase-1 unseen curriculum.

    Pipeline:
        4E1 planar geometry
        -> 4E3 physical grade
        -> 4E4/4E5 centerline blockers

    Crucially, the obsolete 4E3 gate/flank cone generator is NOT run
    on this path.
    """

    from dataclasses import replace

    identity = (
        curriculum_identity(
            int(seed),
            split=split,
        )
    )

    # --------------------------------------------------------
    # Start directly from the original 4E1 planar generator.
    # --------------------------------------------------------

    planar = (
        _TASK4E3_BASE_GENERATOR(
            int(seed),
            split=split,
        )
    )

    # --------------------------------------------------------
    # Reuse only 4E3's validated elevation machinery.
    # --------------------------------------------------------

    grade = (
        _task4e3_grade_kind(
            identity
        )
    )

    grade_rng = random.Random(
        (
            int(seed)
            * 1_000_003
        )
        ^ 0x4E3D3E
    )

    elevations = (
        _task4e3_elevations(
            planar.centerline_xy_m,
            grade=grade,
            rng=grade_rng,
            challenge=(
                identity.split
                == "challenge"
            ),
        )
    )

    road_boxes = (
        _task4e3_graded_road_boxes(
            planar.centerline_xy_m,
            elevations,
            road_width_m=(
                planar.road_width_m
            ),
        )
    )

    graded = replace(
        planar,
        support_boxes=(
            planar.support_boxes[0],
            *road_boxes,
            planar.support_boxes[-1],
        ),
        track_profile=(
            f"{planar.track_profile}"
            f"/grade_{grade}"
        ),
    )

    # --------------------------------------------------------
    # Generate ONLY the approved centerline-blocker hazards.
    # --------------------------------------------------------

    hazard_rng = random.Random(
        (
            int(seed)
            * 1_000_003
        )
        ^ 0x4E4C3A
    )

    cones = (
        _task4e4_centerline_hazards(
            graded,
            identity=identity,
            grade=grade,
            rng=hazard_rng,
        )
    )

    return replace(
        graded,
        cones=cones,
        track_profile=(
            f"{graded.track_profile}"
            "/hazards_centerline_v1"
        ),
    )

# TASK4E5_SPREAD_CENTERLINE_HAZARDS_V1


# TASK4E5_DISTRIBUTED_FLAT_HAZARDS_V2
