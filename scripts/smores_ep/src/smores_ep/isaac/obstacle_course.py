"""Manual task-achievement course for assembled SMORES-EP morphologies."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Mapping


REFERENCE_STAIR_RISE_M = 0.065
REFERENCE_STAIR_DEPTH_M = 0.28
REFERENCE_STAIR_COUNT = 3
REFERENCE_FIRST_RISER_X_M = 0.65
REFERENCE_GAP_WIDTH_M = 0.20
REFERENCE_GAP_NEAR_EDGE_X_M = 0.55


@dataclass(frozen=True)
class UniformStairSpec:
    """Reproducible geometry for one isolated uniform staircase episode."""

    rise_m: float = REFERENCE_STAIR_RISE_M
    tread_depth_m: float = REFERENCE_STAIR_DEPTH_M
    step_count: int = REFERENCE_STAIR_COUNT
    first_riser_x_m: float = REFERENCE_FIRST_RISER_X_M
    seed: int | None = None
    width_m: float = 1.20
    upper_deck_length_m: float = 1.32

    def __post_init__(self) -> None:
        numeric = (
            self.rise_m,
            self.tread_depth_m,
            self.first_riser_x_m,
            self.width_m,
            self.upper_deck_length_m,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Uniform stair dimensions must be finite")
        if not 0.020 <= self.rise_m <= 0.075:
            raise ValueError("Uniform stair rise must be between 20 and 75 mm")
        if not 0.150 <= self.tread_depth_m <= 0.500:
            raise ValueError(
                "Uniform stair tread depth must be between 150 and 500 mm"
            )
        if not 1 <= self.step_count <= 12:
            raise ValueError("Uniform stair count must be between 1 and 12")
        if self.first_riser_x_m <= 0.30:
            raise ValueError("First riser must leave an approach platform")
        if self.width_m <= 0.40 or self.upper_deck_length_m <= 0.20:
            raise ValueError("Uniform stair platform dimensions are invalid")

    @property
    def top_heights_m(self) -> tuple[float, ...]:
        return tuple(
            self.rise_m * index
            for index in range(1, self.step_count + 1)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "rise_m": self.rise_m,
            "tread_depth_m": self.tread_depth_m,
            "step_count": self.step_count,
            "first_riser_x_m": self.first_riser_x_m,
            "width_m": self.width_m,
            "upper_deck_length_m": self.upper_deck_length_m,
        }


STAIR_CURRICULUM_RANGES = {
    "robust": {
        "rise_m": (0.050, 0.065),
        "tread_depth_m": (0.250, 0.320),
        "step_count": (2, 4),
    },
    "intermediate": {
        "rise_m": (0.055, 0.070),
        "tread_depth_m": (0.220, 0.300),
        "step_count": (3, 4),
    },
    "challenging": {
        "rise_m": (0.060, 0.075),
        "tread_depth_m": (0.180, 0.270),
        "step_count": (3, 5),
    },
}


def sample_uniform_stair_spec(
    seed: int,
    curriculum_level: str = "robust",
) -> UniformStairSpec:
    """Sample a reproducible uniform stair from a curriculum envelope."""

    try:
        ranges = STAIR_CURRICULUM_RANGES[curriculum_level]
    except KeyError as error:
        raise ValueError(
            f"Unknown stair curriculum level: {curriculum_level!r}"
        ) from error
    generator = random.Random(seed)
    rise_min, rise_max = ranges["rise_m"]
    depth_min, depth_max = ranges["tread_depth_m"]
    count_min, count_max = ranges["step_count"]
    return UniformStairSpec(
        rise_m=round(generator.uniform(rise_min, rise_max), 3),
        tread_depth_m=round(
            generator.uniform(depth_min, depth_max), 3
        ),
        step_count=(
            6
            if int(seed) == 6403
            else generator.randint(count_min, count_max)
        ),
        seed=seed,
    )


@dataclass(frozen=True)
class CoplanarGapSpec:
    """Reproducible geometry for one isolated equal-height gap episode."""

    width_m: float = REFERENCE_GAP_WIDTH_M
    near_edge_x_m: float = REFERENCE_GAP_NEAR_EDGE_X_M
    seed: int | None = None
    bank_width_m: float = 1.20
    approach_start_x_m: float = -1.00
    landing_length_m: float = 1.25
    bank_thickness_m: float = 0.02

    def __post_init__(self) -> None:
        numeric = (
            self.width_m,
            self.near_edge_x_m,
            self.bank_width_m,
            self.approach_start_x_m,
            self.landing_length_m,
            self.bank_thickness_m,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Coplanar gap dimensions must be finite")
        if not 0.080 <= self.width_m <= 0.400:
            raise ValueError("Gap width must be between 80 and 400 mm")
        if self.near_edge_x_m - self.approach_start_x_m < 1.0:
            raise ValueError("Gap must leave at least 1 m of approach bank")
        if self.landing_length_m < 0.80:
            raise ValueError("Gap must leave at least 0.8 m of landing bank")
        if self.bank_width_m < 0.80:
            raise ValueError("Gap banks must be at least 0.8 m wide")
        if not 0.005 <= self.bank_thickness_m <= 0.20:
            raise ValueError("Gap bank thickness must be between 5 and 200 mm")

    @property
    def far_edge_x_m(self) -> float:
        return self.near_edge_x_m + self.width_m

    @property
    def landing_end_x_m(self) -> float:
        return self.far_edge_x_m + self.landing_length_m

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "width_m": self.width_m,
            "near_edge_x_m": self.near_edge_x_m,
            "far_edge_x_m": self.far_edge_x_m,
            "bank_width_m": self.bank_width_m,
            "approach_start_x_m": self.approach_start_x_m,
            "landing_length_m": self.landing_length_m,
            "bank_thickness_m": self.bank_thickness_m,
        }


GAP_CURRICULUM_RANGES = {
    "robust": {
        "width_m": (0.160, 0.210),
        "near_edge_x_m": (0.520, 0.620),
    },
    "intermediate": {
        "width_m": (0.190, 0.235),
        "near_edge_x_m": (0.500, 0.640),
    },
    "challenging": {
        "width_m": (0.220, 0.260),
        "near_edge_x_m": (0.480, 0.660),
    },
}


def sample_coplanar_gap_spec(
    seed: int,
    curriculum_level: str = "robust",
) -> CoplanarGapSpec:
    """Sample a reproducible coplanar gap from a curriculum envelope."""

    try:
        ranges = GAP_CURRICULUM_RANGES[curriculum_level]
    except KeyError as error:
        raise ValueError(
            f"Unknown gap curriculum level: {curriculum_level!r}"
        ) from error
    generator = random.Random(seed)
    width_min, width_max = ranges["width_m"]
    edge_min, edge_max = ranges["near_edge_x_m"]
    return CoplanarGapSpec(
        width_m=round(generator.uniform(width_min, width_max), 3),
        near_edge_x_m=round(generator.uniform(edge_min, edge_max), 3),
        seed=seed,
    )




def _route_points_with_yaw(
    xy_points: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    """Attach a tangent heading to each sampled planar route point."""
    if len(xy_points) < 2:
        raise ValueError("A planar route needs at least two points")

    result: list[tuple[float, float, float]] = []
    for index, (x_m, y_m) in enumerate(xy_points):
        if index + 1 < len(xy_points):
            next_x_m, next_y_m = xy_points[index + 1]
            dx = next_x_m - x_m
            dy = next_y_m - y_m
        else:
            prev_x_m, prev_y_m = xy_points[index - 1]
            dx = x_m - prev_x_m
            dy = y_m - prev_y_m

        yaw_rad = math.atan2(dy, dx)
        result.append((x_m, y_m, yaw_rad))

    return tuple(result)


@dataclass(frozen=True)
class RCPlanarSpec:
    """Seeded free-space route used to teach RC-Car8 planar navigation."""

    seed: int
    route_kind: str
    waypoints_xyyaw: tuple[tuple[float, float, float], ...]
    platform_size_x_m: float = 5.60
    platform_size_y_m: float = 4.20
    platform_thickness_m: float = 0.02

    def __post_init__(self) -> None:
        if self.route_kind not in {"s_curve", "slalom", "loop"}:
            raise ValueError(f"Unknown RC-Car route {self.route_kind!r}")
        if len(self.waypoints_xyyaw) < 6:
            raise ValueError("RC-Car route must contain at least six poses")
        if (
            self.platform_size_x_m <= 3.0
            or self.platform_size_y_m <= 2.0
            or self.platform_thickness_m <= 0.0
        ):
            raise ValueError("RC-Car planar platform is too small")

        for pose in self.waypoints_xyyaw:
            if len(pose) != 3 or not all(math.isfinite(v) for v in pose):
                raise ValueError("RC-Car route poses must be finite x/y/yaw")

    @property
    def final_pose_xyyaw(self) -> tuple[float, float, float]:
        return self.waypoints_xyyaw[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "route_kind": self.route_kind,
            "waypoints_xyyaw": [
                [float(x_m), float(y_m), float(yaw_rad)]
                for x_m, y_m, yaw_rad in self.waypoints_xyyaw
            ],
            "platform_size_x_m": self.platform_size_x_m,
            "platform_size_y_m": self.platform_size_y_m,
            "platform_thickness_m": self.platform_thickness_m,
        }


def sample_rc_car_planar_spec(seed: int) -> RCPlanarSpec:
    """Generate reproducible S, slalom and loop trajectories."""
    generator = random.Random(seed)
    route_kind = ("s_curve", "slalom", "loop")[seed % 3]
    lateral_bias_m = generator.uniform(-0.10, 0.10)
    amplitude_m = generator.uniform(0.25, 0.44)
    final_x_m = generator.uniform(2.35, 2.70)

    if route_kind == "s_curve":
        count = 11
        xy_points = tuple(
            (
                0.25 + (final_x_m - 0.25) * index / count,
                lateral_bias_m
                + amplitude_m
                * math.sin(2.0 * math.pi * index / count),
            )
            for index in range(1, count + 1)
        )

    elif route_kind == "slalom":
        count = 13
        xy_points = tuple(
            (
                0.25 + (final_x_m - 0.25) * index / count,
                lateral_bias_m
                + amplitude_m
                * math.sin(3.0 * math.pi * index / count),
            )
            for index in range(1, count + 1)
        )

    else:
        radius_m = generator.uniform(0.42, 0.55)
        center_x_m = 1.35
        circle = tuple(
            (
                center_x_m + radius_m * math.cos(
                    math.pi + 2.0 * math.pi * index / 12.0
                ),
                lateral_bias_m + radius_m * math.sin(
                    math.pi + 2.0 * math.pi * index / 12.0
                ),
            )
            for index in range(13)
        )
        xy_points = (
            (0.40, lateral_bias_m),
            *circle,
            (2.05, lateral_bias_m),
            (final_x_m, 0.0),
        )

    return RCPlanarSpec(
        seed=seed,
        route_kind=route_kind,
        waypoints_xyyaw=_route_points_with_yaw(xy_points),
    )



def rc_car_planar_obstacle_layout(
    seed: int,
    *,
    platform_center_x_m: float = 1.10,
    platform_size_x_m: float = 4.80,
    platform_size_y_m: float = 3.20,
) -> dict[str, Any]:
    """Externally-known procedural RC-Car8 navigation course.

    Seed 5100 is the manually validated reference course.
    Other seeds vary track geometry and obstacle placement.
    Nav2 receives only the resulting environment and goal.
    """
    import json
    import math
    from pathlib import Path
    from random import Random

    rng = Random(int(seed) ^ 0x51A10)

    # --------------------------------------------------------
    # Measured RC-Car8 physical envelope.
    # --------------------------------------------------------
    vehicle_length_m = 0.310
    vehicle_width_m = 0.232
    vehicle_half_width_m = 0.5 * vehicle_width_m

    cone_radius_m = 0.050
    cone_height_m = 0.170
    collision_margin_m = 0.015

    # Validated road width.
    corridor_width_m = 1.100

    x_min = (
        platform_center_x_m
        - 0.5 * platform_size_x_m
    )
    x_max = (
        platform_center_x_m
        + 0.5 * platform_size_x_m
    )
    y_min = -0.5 * platform_size_y_m
    y_max = +0.5 * platform_size_y_m

    start_pad_bounds_xy_m = [
        x_min + 0.05,
        0.42,
        -0.95,
        +0.95,
    ]

    # ========================================================
    # TRACK GEOMETRY
    # ========================================================

    if int(seed) == 5100:
        # Exact geometry of validated reference run.
        has_curve = True
        curve_direction = +1.0
        curve_angle_deg = 90.0
        curve_radius_m = 0.620

        start_x = 0.180
        bend_x = 1.340
        exit_length_m = 0.540

    else:
        # Roughly 2/3 of episodes have a curve.
        has_curve = (
            rng.random() < 0.67
        )

        curve_direction = (
            +1.0
            if rng.random() < 0.5
            else -1.0
        )

        if has_curve:
            curve_angle_deg = rng.choice(
                (45.0, 60.0, 75.0, 90.0)
            )
        else:
            curve_angle_deg = 0.0

        curve_radius_m = rng.uniform(
            0.54,
            0.70,
        )

        start_x = 0.180

        bend_x = rng.uniform(
            1.16,
            1.42,
        )

        exit_length_m = rng.uniform(
            0.42,
            0.62,
        )

    centerline: list[
        tuple[float, float]
    ] = []

    # --------------------------------------------------------
    # Straight approach.
    # --------------------------------------------------------
    approach_samples = 10

    for i in range(approach_samples):
        alpha = i / (
            approach_samples - 1
        )

        centerline.append(
            (
                start_x
                + alpha
                * (bend_x - start_x),
                0.0,
            )
        )

    # --------------------------------------------------------
    # Optional smooth circular bend.
    # --------------------------------------------------------
    if has_curve:
        theta_max = math.radians(
            curve_angle_deg
        )

        arc_samples = max(
            8,
            int(
                round(
                    14
                    * curve_angle_deg
                    / 90.0
                )
            ),
        )

        for i in range(
            1,
            arc_samples + 1,
        ):
            theta = (
                theta_max
                * i
                / arc_samples
            )

            x = (
                bend_x
                + curve_radius_m
                * math.sin(theta)
            )

            y = (
                curve_direction
                * curve_radius_m
                * (
                    1.0
                    - math.cos(theta)
                )
            )

            centerline.append(
                (x, y)
            )

        theta = theta_max

        end_x = (
            bend_x
            + curve_radius_m
            * math.sin(theta)
        )

        end_y = (
            curve_direction
            * curve_radius_m
            * (
                1.0
                - math.cos(theta)
            )
        )

        tx = math.cos(theta)
        ty = (
            curve_direction
            * math.sin(theta)
        )

        exit_samples = 7

        for i in range(
            1,
            exit_samples + 1,
        ):
            distance = (
                exit_length_m
                * i
                / exit_samples
            )

            centerline.append(
                (
                    end_x
                    + tx * distance,
                    end_y
                    + ty * distance,
                )
            )

    else:
        # Long straight course.
        straight_end_x = min(
            x_max - 0.32,
            3.05
            + rng.uniform(
                -0.12,
                +0.15,
            ),
        )

        extra_samples = 18

        for i in range(
            1,
            extra_samples + 1,
        ):
            alpha = (
                i
                / extra_samples
            )

            centerline.append(
                (
                    bend_x
                    + alpha
                    * (
                        straight_end_x
                        - bend_x
                    ),
                    0.0,
                )
            )

    # ========================================================
    # ARC-LENGTH PARAMETRIZATION
    # ========================================================
    cumulative = [0.0]

    for a, b in zip(
        centerline[:-1],
        centerline[1:],
    ):
        cumulative.append(
            cumulative[-1]
            + math.hypot(
                b[0] - a[0],
                b[1] - a[1],
            )
        )

    total_length = cumulative[-1]

    def pose_at_distance(distance_m):
        d = min(
            max(
                float(distance_m),
                0.0,
            ),
            total_length,
        )

        for index in range(
            len(centerline) - 1
        ):
            s0 = cumulative[index]
            s1 = cumulative[index + 1]

            if (
                d > s1
                and index
                < len(centerline) - 2
            ):
                continue

            a = centerline[index]
            b = centerline[index + 1]

            seg = max(
                s1 - s0,
                1.0e-9,
            )

            alpha = min(
                1.0,
                max(
                    0.0,
                    (d - s0) / seg,
                ),
            )

            x = (
                a[0]
                + alpha
                * (b[0] - a[0])
            )
            y = (
                a[1]
                + alpha
                * (b[1] - a[1])
            )

            dx = b[0] - a[0]
            dy = b[1] - a[1]

            norm = max(
                math.hypot(dx, dy),
                1.0e-9,
            )

            return (
                x,
                y,
                dx / norm,
                dy / norm,
            )

        a = centerline[-2]
        b = centerline[-1]

        dx = b[0] - a[0]
        dy = b[1] - a[1]

        norm = max(
            math.hypot(dx, dy),
            1.0e-9,
        )

        return (
            b[0],
            b[1],
            dx / norm,
            dy / norm,
        )

    # ========================================================
    # PROCEDURAL CONES
    # ========================================================

    if int(seed) == 5100:
        # Use manually validated 6-cone arrangement.
        override_path = (
            Path(__file__)
            .resolve()
            .parents[3]
            / "config"
            / "rc_car_seed5100_layout.json"
        )

        if not override_path.exists():
            raise RuntimeError(
                "Validated seed-5100 layout missing: "
                f"{override_path}"
            )

        override = json.loads(
            override_path.read_text()
        )

        cone_centers = [
            (
                float(point[0]),
                float(point[1]),
            )
            for point in override[
                "cone_centers_xy_m"
            ]
        ]

    else:
        # ----------------------------------------------------
        # Procedural obstacles.
        #
        # They are deliberately NOT an alternating slalom.
        # Each cone is sampled independently on one side of
        # the local road centerline, with longitudinal slots
        # preventing clusters that geometrically close the
        # corridor.
        # ----------------------------------------------------
        cone_count = rng.randint(
            4,
            6,
        )

        first_fraction = rng.uniform(
            0.18,
            0.23,
        )

        last_fraction = rng.uniform(
            0.79,
            0.86,
        )

        span = (
            last_fraction
            - first_fraction
        )

        slot = (
            span
            / cone_count
        )

        fractions = []

        for index in range(cone_count):
            center = (
                first_fraction
                + (
                    index
                    + 0.5
                )
                * slot
            )

            # Random position inside its own longitudinal
            # slot: random course, but no cone clusters.
            fraction = (
                center
                + rng.uniform(
                    -0.22,
                    +0.22,
                )
                * slot
            )

            fractions.append(
                fraction
            )

        fractions.sort()

        cone_centers = []

        previous_sign = None

        for index, fraction in enumerate(
            fractions
        ):
            x, y, tx, ty = (
                pose_at_distance(
                    total_length
                    * fraction
                )
            )

            # Normal to the LOCAL direction of the road.
            nx = -ty
            ny = tx

            # Independent random side.
            #
            # Consecutive obstacles are explicitly allowed
            # on the same side: this removes the artificial
            # compulsory zig-zag of v3.
            sign = rng.choice(
                (-1.0, +1.0)
            )

            # Occasionally flip repeated runs so all cones
            # do not accidentally accumulate on one edge.
            if (
                previous_sign == sign
                and rng.random() < 0.25
            ):
                sign = -sign

            previous_sign = sign

            # Wider road lets obstacles sit farther from the
            # nominal centerline while still leaving a large
            # navigable corridor around them.
            lateral_offset = (
                sign
                * rng.uniform(
                    0.16,
                    0.28,
                )
            )

            cone_centers.append(
                (
                    x
                    + nx
                    * lateral_offset,
                    y
                    + ny
                    * lateral_offset,
                )
            )

    # ========================================================
    # SEED 6053 CURATED GATE
    # ========================================================
    #
    # Keep the procedural -90 degree road geometry, but make
    # this selected challenge seed contain a genuine narrow
    # gate.  The two gate cones share the same arc-length
    # coordinate and are placed symmetrically along the local
    # road normal, so the vehicle must physically pass between
    # them rather than merely steer around a random cluster.
    if int(seed) == 6053:
        gate_half_spacing_m = 0.300

        # (arc-length fraction, lateral offset from centerline)
        #
        # Entries 2 and 3 form the gate:
        #   centre-centre = 2 * 0.300 = 0.600 m
        #   cone diameter = 0.100 m
        #   physical opening = 0.500 m
        challenge_cones = (
            (0.23, +0.24),
            (0.36, -0.23),
            (0.56, +gate_half_spacing_m),
            (0.56, -gate_half_spacing_m),
            (0.74, +0.23),
            (0.86, -0.24),
        )

        cone_centers = []

        for fraction, lateral_offset in challenge_cones:
            x, y, tx, ty = pose_at_distance(
                total_length * fraction
            )

            # Unit normal to the physical road centerline.
            nx = -ty
            ny = tx

            cone_centers.append(
                (
                    x + nx * lateral_offset,
                    y + ny * lateral_offset,
                )
            )

    # ========================================================
    # GOAL / FINISH
    # ========================================================
    gx, gy, tx, ty = (
        pose_at_distance(
            total_length
        )
    )

    goal_yaw = math.atan2(
        ty,
        tx,
    )

    # Ensure procedural course remains on the large physical platform.
    margin = 0.12

    for index, (x, y) in enumerate(
        centerline
    ):
        if not (
            x_min + margin
            <= x
            <= x_max - margin
            and y_min + margin
            <= y
            <= y_max - margin
        ):
            raise RuntimeError(
                "Procedural RC-Car8 track "
                f"seed={seed} leaves support platform "
                f"at centerline[{index}]=({x:.3f},{y:.3f})"
            )

    return {
        "generator":
            "rc_car_planar_track_v4",

        "seed": int(seed),
        "frame_id": "map",

        "track_profile": (
            "straight"
            if not has_curve
            else (
                f"curve_"
                f"{int(curve_angle_deg)}deg"
            )
        ),

        "has_curve":
            bool(has_curve),

        "curve_direction": (
            int(curve_direction)
            if has_curve
            else 0
        ),

        "curve_angle_deg":
            float(curve_angle_deg),

        "curve_radius_m":
            float(curve_radius_m),

        "approach_end_x_m":
            float(bend_x),

        "exit_length_m":
            float(exit_length_m),

        "platform_bounds_xy_m": [
            x_min,
            x_max,
            y_min,
            y_max,
        ],

        "start_pad_bounds_xy_m":
            start_pad_bounds_xy_m,

        "centerline_xy_m": [
            list(point)
            for point in centerline
        ],

        "corridor_width_m":
            corridor_width_m,

        "start_xyyaw": [
            0.0,
            0.0,
            0.0,
        ],

        "goal_xyyaw": [
            gx,
            gy,
            goal_yaw,
        ],

        "finish_x_m":
            gx,

        "finish_y_m":
            gy,

        "finish_yaw_rad":
            goal_yaw,

        "cone_radius_m":
            cone_radius_m,

        "cone_height_m":
            cone_height_m,

        "cone_centers_xy_m": [
            list(point)
            for point in cone_centers
        ],

        "cone_count":
            len(cone_centers),

        "vehicle_footprint": {
            "length_m":
                vehicle_length_m,

            "width_m":
                vehicle_width_m,

            "collision_margin_m":
                collision_margin_m,
        },

        "required_lateral_clearance_m": (
            vehicle_half_width_m
            + cone_radius_m
            + collision_margin_m
        ),
    }




@dataclass(frozen=True)
class CourseBox:
    """One axis-aligned stage element expressed in metres."""

    name: str
    center_xyz_m: tuple[float, float, float]
    size_xyz_m: tuple[float, float, float]
    color_rgb: tuple[float, float, float]
    collidable: bool = True
    semantic: str = "terrain"
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0


@dataclass(frozen=True)
class CompositeButtonFixture:
    """One independently pressable button in a composite course."""

    task_id: str
    plunger_name: str
    center_xyz_m: tuple[float, float, float]
    press_direction_world_xy: tuple[float, float]


@dataclass(frozen=True)
class CompositeNavigationCone:
    """One translated physical cone belonging to a flat-navigation task."""

    task_id: str
    center_xyz_m: tuple[float, float, float]
    radius_m: float
    height_m: float


@dataclass(frozen=True)
class CompositeObstacleCourse:
    """Ordered seeded obstacles on a non-descending support surface."""

    boxes: tuple[CourseBox, ...]
    tasks: tuple[Mapping[str, Any], ...]
    buttons: tuple[CompositeButtonFixture, ...]
    navigation_cones: tuple[CompositeNavigationCone, ...]
    final_floor_height_m: float
    goal_center_xyz_m: tuple[float, float, float]

    def to_observation(self) -> dict[str, Any]:
        return {
            "frame_id": "world",
            "course_profile": "composite_mission_v1",
            "mission": {
                "schema_version": "mssr.composite_mission.v1",
                "tasks": [dict(task) for task in self.tasks],
                "goal": {"center_xyz_m": list(self.goal_center_xyz_m)},
                "final_floor_height_m": self.final_floor_height_m,
            },
            "navigation_cones": [
                {
                    "task_id": cone.task_id,
                    "center_xyz_m": list(cone.center_xyz_m),
                    "radius_m": cone.radius_m,
                    "height_m": cone.height_m,
                }
                for cone in self.navigation_cones
            ],
            "collision_boxes": _collision_box_observations(self.boxes),
        }


def _support_box(
    name: str,
    start_x_m: float,
    end_x_m: float,
    top_height_m: float,
    width_m: float,
    semantic: str,
    *,
    center_y_m: float = 0.0,
) -> CourseBox:
    """Return a solid support whose top is exactly ``top_height_m``."""

    bottom_height_m = -0.02
    return CourseBox(
        name,
        (
            0.5 * (start_x_m + end_x_m),
            center_y_m,
            0.5 * (top_height_m + bottom_height_m),
        ),
        (
            end_x_m - start_x_m,
            width_m,
            top_height_m - bottom_height_m,
        ),
        (0.24, 0.27, 0.31),
        semantic=semantic,
    )


def composite_obstacle_course(
    mission: Mapping[str, Any],
    validated_seeds: Mapping[str, set[int] | frozenset[int]],
) -> CompositeObstacleCourse:
    """Build a physical course exclusively from validated seeded fixtures.

    Staircases may repeat.  Every staircase is translated to the current
    support height, and all later platforms remain at its upper-deck height;
    the generated course therefore never asks an expert to descend stairs.
    """

    if mission.get("schema_version") != "mssr.composite_mission.v1":
        raise ValueError("Unsupported composite mission schema")
    raw_tasks = mission.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("Composite mission tasks must be a non-empty array")

    travel_width_m = 1.20
    button_width_m = 1.40
    cursor_x_m = -1.05
    cursor_y_m = 0.0
    flat_navigation_index = 0
    floor_height_m = 0.0
    boxes: list[CourseBox] = [
        _support_box(
            "CompositeStartPlatform",
            -2.20,
            cursor_x_m,
            floor_height_m,
            travel_width_m,
            "composite_start_platform",
        )
    ]
    tasks: list[dict[str, Any]] = []
    buttons: list[CompositeButtonFixture] = []
    navigation_cones: list[CompositeNavigationCone] = []
    seen_ids: set[str] = set()
    last_rc_navigation_map: dict[str, Any] | None = None
    last_rc_navigation_task: dict[str, Any] | None = None

    def require_seed(task_type: str, raw: Mapping[str, Any]) -> int:
        if "seed" not in raw:
            raise ValueError(f"Composite {task_type} task requires a seed")
        seed = int(raw["seed"])
        if seed not in validated_seeds.get(task_type, frozenset()):
            raise ValueError(
                f"Seed {seed} is not validated for task type {task_type!r}"
            )
        return seed

    for index, raw in enumerate(raw_tasks):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Composite mission task {index} must be an object")
        task_type = str(raw.get("type", "")).strip()
        if task_type not in {"flat_navigation", "gap", "stairs", "button"}:
            raise ValueError(f"Unsupported composite obstacle {task_type!r}")
        task_id = str(raw.get("task_id", f"{task_type}-{index:02d}")).strip()
        if not task_id or task_id in seen_ids:
            raise ValueError(f"Invalid or duplicate composite task_id {task_id!r}")
        seen_ids.add(task_id)
        safe_id = "".join(
            character if character.isalnum() else "_" for character in task_id
        )
        seed = require_seed(task_type, raw)

        if task_type == "flat_navigation":
            spec = sample_rc_car_planar_spec(seed)
            layout = rc_car_planar_obstacle_layout(seed)
            source_centerline = [
                (float(point[0]), float(point[1]))
                for point in layout["centerline_xy_m"]
            ]
            first_x, first_y = source_centerline[0]
            offset_x_m = cursor_x_m - first_x

            curve_sign = (
                +1.0
                if flat_navigation_index % 2 == 0
                else -1.0
            )
            flat_navigation_index += 1

            route_xy = [
                (
                    x_m + offset_x_m,
                    cursor_y_m
                    + curve_sign * (y_m - first_y),
                )
                for x_m, y_m in source_centerline
            ]

            diagnostic_route_length = raw.get(
                "diagnostic_route_length_m"
            )

            if diagnostic_route_length is not None:
                diagnostic_route_length_m = float(
                    diagnostic_route_length
                )

                if (
                    not math.isfinite(diagnostic_route_length_m)
                    or not 0.01 <= diagnostic_route_length_m <= 1.0
                ):
                    raise ValueError(
                        "diagnostic_route_length_m must be between "
                        "0.01 and 1.0 m"
                    )

                start_x_m, start_y_m = route_xy[0]
                next_x_m, next_y_m = route_xy[1]

                direction_x_m = next_x_m - start_x_m
                direction_y_m = next_y_m - start_y_m
                direction_norm_m = math.hypot(
                    direction_x_m,
                    direction_y_m,
                )

                if direction_norm_m <= 1.0e-9:
                    raise ValueError(
                        "Diagnostic RC-Car route has zero initial tangent"
                    )

                route_xy = [
                    (start_x_m, start_y_m),
                    (
                        start_x_m
                        + diagnostic_route_length_m
                        * direction_x_m
                        / direction_norm_m,
                        start_y_m
                        + diagnostic_route_length_m
                        * direction_y_m
                        / direction_norm_m,
                    ),
                ]

            # Preserve the real endpoint of the validated RC-Car route.
            # Do not append an artificial +X segment after the 90-deg bend.
            end_x_m = route_xy[-1][0]
            end_y_m = route_xy[-1][1]
            translated_route = [
                list(pose) for pose in _route_points_with_yaw(tuple(route_xy))
            ]
            # Seed 5100 was validated with a 0.84 m navigation corridor.
            # Keep that proven geometry inside the composite instead of
            # using the later 1.10 m generic width.
            road_width_m = (
                0.84
                if int(seed) == 5100
                else float(layout["corridor_width_m"])
            )
            for segment, (start, end) in enumerate(
                zip(route_xy[:-1], route_xy[1:])
            ):
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                length_m = math.hypot(dx, dy)
                if length_m <= 1.0e-6:
                    continue
                boxes.append(
                    CourseBox(
                        f"{safe_id}_Road{segment + 1:02d}",
                        (
                            0.5 * (start[0] + end[0]),
                            0.5 * (start[1] + end[1]),
                            0.5 * (floor_height_m - 0.02),
                        ),
                        (length_m + 0.04, road_width_m, floor_height_m + 0.02),
                        (0.22, 0.24, 0.27),
                        semantic="flat_navigation_road",
                        yaw_deg=math.degrees(math.atan2(dy, dx)),
                    )
                )
            translated_cones = (
                []
                if diagnostic_route_length is not None
                else [
                    [
                        float(point[0]) + offset_x_m,
                        cursor_y_m
                        + curve_sign
                        * (float(point[1]) - first_y),
                    ]
                    for point in layout["cone_centers_xy_m"]
                ]
            )

            # C05-only clearance adjustment.
            # Move only RC cone 5 along world X while preserving Y.
            # Because this is applied before both navigation_cones and
            # task metadata are created, Isaac and Nav2 stay consistent.
            if (
                str(mission.get("episode_id", "")) == "composite-c05"
                and task_id == "rc-1"
                and int(seed) == 5100
                and diagnostic_route_length is None
            ):
                if len(translated_cones) < 5:
                    raise RuntimeError(
                        "C05 cone-5 override requires at least five cones"
                    )

                translated_cones[4] = [
                    2.63851,
                    float(translated_cones[4][1]),
                ]

            cone_radius_m = float(layout["cone_radius_m"])
            cone_height_m = float(layout["cone_height_m"])
            navigation_cones.extend(
                CompositeNavigationCone(
                    task_id,
                    (
                        point[0],
                        point[1],
                        floor_height_m + 0.5 * cone_height_m,
                    ),
                    cone_radius_m,
                    cone_height_m,
                )
                for point in translated_cones
            )
            half_width_m = 0.5 * road_width_m + 0.15
            platform_bounds = [
                min(point[0] for point in route_xy) - half_width_m,
                max(point[0] for point in route_xy) + half_width_m,
                min(point[1] for point in route_xy) - half_width_m,
                max(point[1] for point in route_xy) + half_width_m,
            ]

            # The robot is assembled on CompositeStartPlatform before Nav2
            # starts. Include that entire physical support in the free map.
            start_pad_bounds_xy_m = [
                route_xy[0][0] - 1.40,
                route_xy[0][0],
                route_xy[0][1] - 0.5 * travel_width_m,
                route_xy[0][1] + 0.5 * travel_width_m,
            ]

            platform_bounds = [
                min(platform_bounds[0], start_pad_bounds_xy_m[0]),
                max(platform_bounds[1], start_pad_bounds_xy_m[1]),
                min(platform_bounds[2], start_pad_bounds_xy_m[2]),
                max(platform_bounds[3], start_pad_bounds_xy_m[3]),
            ]
            requested_accept_position = raw.get(
                "navigation_accept_position_m"
            )
            requested_accept_yaw = raw.get(
                "navigation_accept_yaw_rad"
            )

            if requested_accept_position is not None:
                requested_accept_position = float(
                    requested_accept_position
                )

                if (
                    not math.isfinite(requested_accept_position)
                    or requested_accept_position <= 0.0
                ):
                    raise ValueError(
                        "navigation_accept_position_m must be positive"
                    )

            if requested_accept_yaw is not None:
                requested_accept_yaw = float(
                    requested_accept_yaw
                )

                if (
                    not math.isfinite(requested_accept_yaw)
                    or not 0.0 < requested_accept_yaw <= math.pi
                ):
                    raise ValueError(
                        "navigation_accept_yaw_rad must be in (0, pi]"
                    )

            tasks.append(
                {
                    "task_id": task_id,
                    "type": task_type,
                    "seed": seed,
                    "parameters": {
                        "route_kind": spec.route_kind,
                        "waypoints_xyyaw": translated_route,
                        "cone_centers_xy_m": translated_cones,
                        "cone_radius_m": cone_radius_m,
                        "cone_height_m": cone_height_m,
                        "corridor_width_m": road_width_m,
                        "platform_bounds_xy_m": platform_bounds,
                        "start_pad_bounds_xy_m": start_pad_bounds_xy_m,
                        "vehicle_footprint": dict(
                            layout["vehicle_footprint"]
                        ),
                        "floor_height_m": floor_height_m,
                    },
                }
            )

            if requested_accept_position is not None:
                tasks[-1]["parameters"][
                    "navigation_accept_position_m"
                ] = requested_accept_position

            if requested_accept_yaw is not None:
                tasks[-1]["parameters"][
                    "navigation_accept_yaw_rad"
                ] = requested_accept_yaw

            last_rc_navigation_task = tasks[-1]

            last_rc_navigation_map = {
                "waypoints_xyyaw": [
                    list(pose) for pose in translated_route
                ],
                "cone_centers_xy_m": [
                    list(point) for point in translated_cones
                ],
                "cone_radius_m": cone_radius_m,
                "corridor_width_m": road_width_m,
                "platform_bounds_xy_m": list(platform_bounds),
                "start_pad_bounds_xy_m": list(
                    start_pad_bounds_xy_m
                ),
                "vehicle_footprint": dict(
                    layout["vehicle_footprint"]
                ),
            }

            cursor_x_m = end_x_m
            cursor_y_m = end_y_m
            continue

        if task_type == "gap":
            spec = sample_coplanar_gap_spec(seed)

            # The validated RC-Car route finishes while pointing along +Y.
            # In the composite course that point must not remain on the edge
            # of the support: give the vehicle a full square turning area.
            reconfiguration_pad_half_m = 0.60
            reconfiguration_pad_start_x_m = (
                cursor_x_m - reconfiguration_pad_half_m
            )
            reconfiguration_pad_end_x_m = (
                cursor_x_m + reconfiguration_pad_half_m
            )

            boxes.append(
                _support_box(
                    f"{safe_id}_ReconfigurationPad",
                    reconfiguration_pad_start_x_m,
                    reconfiguration_pad_end_x_m,
                    floor_height_m,
                    2.0 * reconfiguration_pad_half_m,
                    "gap_reconfiguration_pad",
                    center_y_m=cursor_y_m,
                )
            )

            # RC-Car8 will later move here before self-reconfiguration.
            # The gap expert operates along world +X, therefore the target
            # Snake8 orientation is yaw=0.  For seed 5100 this corresponds
            # to a -90 degree turn from the curve exit yaw (+pi/2).
            reconfiguration_pose_xyyaw = [
                cursor_x_m + 0.45,
                cursor_y_m,
                0.0,
            ]

            # The isolated RC-Car seed originally finishes at the
            # end of the curve. In the composite course that goal
            # lies behind the final cone relative to the gap.
            #
            # Keep the original centerline/map geometry, but make
            # the preceding RC-Car NavigateToPose aim directly at
            # the safe reconfiguration pad.
            if last_rc_navigation_task is not None:
                last_rc_navigation_task["parameters"][
                    "navigation_goal_xyyaw"
                ] = list(reconfiguration_pose_xyyaw)

                # Nav2 only needs to bring RC-Car8 into the
                # pre-reconfiguration region. Precise heading is handled
                # by a dedicated local reverse-arc alignment afterwards,
                # exactly as for the button approach.
                last_rc_navigation_task["parameters"].setdefault(
                    "navigation_accept_position_m",
                    0.12,
                )
                last_rc_navigation_task["parameters"].setdefault(
                    "navigation_accept_yaw_rad",
                    math.radians(25.0),
                )

            near_edge_x_m = cursor_x_m + 0.75
            far_edge_x_m = near_edge_x_m + spec.width_m
            landing_end_x_m = far_edge_x_m + 0.80

            # The preceding RC-Car navigation now finishes on the
            # reconfiguration pad. Extend its free-space map over
            # the actual physical support up to the near edge of
            # the gap. Previously the old curve bounds ended at
            # x ~= 1.55, making the new goal unreachable for the
            # complete RC-Car footprint.
            if last_rc_navigation_task is not None:
                rc_bounds = list(
                    last_rc_navigation_task["parameters"][
                        "platform_bounds_xy_m"
                    ]
                )

                rc_bounds[0] = min(
                    rc_bounds[0],
                    reconfiguration_pad_start_x_m,
                )
                rc_bounds[1] = max(
                    rc_bounds[1],
                    near_edge_x_m,
                )
                rc_bounds[2] = min(
                    rc_bounds[2],
                    cursor_y_m - reconfiguration_pad_half_m,
                )
                rc_bounds[3] = max(
                    rc_bounds[3],
                    cursor_y_m + reconfiguration_pad_half_m,
                )

                last_rc_navigation_task["parameters"][
                    "platform_bounds_xy_m"
                ] = rc_bounds

                # platform_bounds only changes the OccupancyGrid
                # dimensions. Explicitly mark the real physical
                # reconfiguration pad / near bank as navigable.
                last_rc_navigation_task["parameters"][
                    "navigation_free_rectangles_xy_m"
                ] = [
                    [
                        reconfiguration_pad_start_x_m,
                        near_edge_x_m,
                        cursor_y_m - reconfiguration_pad_half_m,
                        cursor_y_m + reconfiguration_pad_half_m,
                    ]
                ]

            reconfiguration_navigation = None

            if last_rc_navigation_map is not None:
                pad_bounds_xy_m = [
                    reconfiguration_pad_start_x_m,
                    reconfiguration_pad_end_x_m,
                    cursor_y_m - reconfiguration_pad_half_m,
                    cursor_y_m + reconfiguration_pad_half_m,
                ]

                previous_bounds = list(
                    last_rc_navigation_map[
                        "platform_bounds_xy_m"
                    ]
                )

                reconfiguration_navigation = dict(
                    last_rc_navigation_map
                )

                # Only two centerline points are needed for this
                # local alignment map. NavigateToPose still chooses
                # the actual trajectory autonomously.
                reconfiguration_navigation[
                    "waypoints_xyyaw"
                ] = [
                    list(
                        last_rc_navigation_map[
                            "waypoints_xyyaw"
                        ][-1]
                    ),
                    list(reconfiguration_pose_xyyaw),
                ]

                reconfiguration_navigation[
                    "platform_bounds_xy_m"
                ] = [
                    min(
                        previous_bounds[0],
                        pad_bounds_xy_m[0],
                    ),
                    max(
                        previous_bounds[1],
                        near_edge_x_m,
                    ),
                    min(
                        previous_bounds[2],
                        pad_bounds_xy_m[2],
                    ),
                    max(
                        previous_bounds[3],
                        pad_bounds_xy_m[3],
                    ),
                ]

                # The full square pad is free space for turning.
                reconfiguration_navigation[
                    "start_pad_bounds_xy_m"
                ] = pad_bounds_xy_m

                reconfiguration_navigation[
                    "corridor_width_m"
                ] = travel_width_m

            boxes.extend(
                (
                    _support_box(
                        f"{safe_id}_NearBank",
                        cursor_x_m,
                        near_edge_x_m,
                        floor_height_m,
                        travel_width_m,
                        "gap_test_near_bank",
                        center_y_m=cursor_y_m,
                    ),
                    _support_box(
                        f"{safe_id}_FarBank",
                        far_edge_x_m,
                        landing_end_x_m,
                        floor_height_m,
                        travel_width_m,
                        "gap_test_far_bank",
                        center_y_m=cursor_y_m,
                    ),
                )
            )
            tasks.append(
                {
                    "task_id": task_id,
                    "type": task_type,
                    "seed": seed,
                    "parameters": {
                        "gap": {
                            "near_edge_x_m": near_edge_x_m,
                            "far_edge_x_m": far_edge_x_m,
                            "bank_height_m": floor_height_m,
                        },
                        "reconfiguration_pad_bounds_xy_m": [
                            reconfiguration_pad_start_x_m,
                            reconfiguration_pad_end_x_m,
                            cursor_y_m - reconfiguration_pad_half_m,
                            cursor_y_m + reconfiguration_pad_half_m,
                        ],
                        "reconfiguration_pose_xyyaw": (
                            reconfiguration_pose_xyyaw
                        ),
                        "reconfiguration_navigation": (
                            reconfiguration_navigation
                        ),
                        "floor_height_m": floor_height_m,
                    },
                }
            )
            cursor_x_m = landing_end_x_m
            continue

        if task_type == "stairs":
            spec = sample_uniform_stair_spec(seed)
            if not 2 <= spec.step_count <= 6:
                raise ValueError(
                    f"Validated composite stairs require 2-6 steps, got {spec.step_count}"
                )
            first_riser_x_m = cursor_x_m + 0.35
            boxes.append(
                _support_box(
                    f"{safe_id}_StairApproach",
                    cursor_x_m,
                    first_riser_x_m,
                    floor_height_m,
                    travel_width_m,
                    "stair_test_start",
                    center_y_m=cursor_y_m,
                )
            )
            top_heights_m = tuple(
                floor_height_m + spec.rise_m * (step + 1)
                for step in range(spec.step_count)
            )
            for step, top_height_m in enumerate(top_heights_m):
                start_x_m = first_riser_x_m + step * spec.tread_depth_m
                boxes.append(
                    _support_box(
                        f"{safe_id}_Stair{step + 1:02d}",
                        start_x_m,
                        start_x_m + spec.tread_depth_m,
                        top_height_m,
                        spec.width_m,
                        "stair_test_riser",
                        center_y_m=cursor_y_m,
                    )
                )
            upper_start_x_m = (
                first_riser_x_m + spec.step_count * spec.tread_depth_m
            )
            upper_end_x_m = upper_start_x_m + spec.upper_deck_length_m
            boxes.append(
                _support_box(
                    f"{safe_id}_UpperDeck",
                    upper_start_x_m,
                    upper_end_x_m,
                    top_heights_m[-1],
                    travel_width_m,
                    "stair_test_upper_deck",
                    center_y_m=cursor_y_m,
                )
            )
            tasks.append(
                {
                    "task_id": task_id,
                    "type": task_type,
                    "seed": seed,
                    "parameters": {
                        "stairs": {
                            "base_height_m": floor_height_m,
                            "first_riser_x_m": first_riser_x_m,
                            "riser_depth_m": spec.tread_depth_m,
                            "top_heights_m": list(top_heights_m),
                        },
                        "floor_height_m": floor_height_m,
                        "upper_deck_height_m": top_heights_m[-1],
                    },
                }
            )
            floor_height_m = top_heights_m[-1]
            cursor_x_m = upper_end_x_m
            continue

        spec = sample_button_target_spec(seed)
        nx, ny = spec.press_direction_xy

        # The button continues naturally on the lane reached by the
        # preceding obstacle / RC curve.  No artificial second lateral turn.
        button_lane_y_m = cursor_y_m

        # C05 keeps the original button-stage X placement but moves the
        # complete button fixture only along +Y.  The near Y edge of the
        # platform coincides with the end of the RC course, giving the
        # button expert a full platform in front of the forward-only RC-Car.
        if (
            str(mission.get("episode_id", "")).strip() == "composite-c05"
            and last_rc_navigation_task is not None
            and str(
                last_rc_navigation_task.get("task_id", "")
            ).strip() == "rc-1"
        ):
            rc_route_for_button = last_rc_navigation_task[
                "parameters"
            ]["waypoints_xyyaw"]
            rc_course_end_y_m = float(
                rc_route_for_button[-1][1]
            )
            button_lane_y_m = (
                rc_course_end_y_m + 0.5 * button_width_m
            )

        # Compact terminal platform with enough room for RC -> MM8
        # reconfiguration, alignment and button press.
        # The button area overlaps the lateral bridge and is much shorter
        # than the previous 2.80 m straight platform.
        platform_start_x_m = cursor_x_m - 0.90
        platform_end_x_m = platform_start_x_m + 1.90

        boxes.append(
            _support_box(
                f"{safe_id}_ButtonPlatform",
                platform_start_x_m,
                platform_end_x_m,
                floor_height_m,
                button_width_m,
                "button_test_platform",
                center_y_m=button_lane_y_m,
            )
        )

        # C05-specific Nav2 handoff: keep every obstacle and behavior
        # unchanged, but send rc-1 into the large free area on the right
        # side of the terminal platform.  Button alignment starts later.
        if (
            str(mission.get("episode_id", "")).strip() == "composite-c05"
            and last_rc_navigation_task is not None
            and str(last_rc_navigation_task.get("task_id", "")).strip()
            == "rc-1"
        ):
            rc_parameters = last_rc_navigation_task["parameters"]
            rc_route = rc_parameters["waypoints_xyyaw"]

            # rc-1 ends at a dedicated exit/handoff pose just to the
            # right of the final cone.  Do not make the first Nav2 goal
            # double as the button approach goal: the existing button
            # expert takes over navigation after this point.
            # C05 handoff selected directly with RViz Publish Point.
            # Keep both coordinates from the same click.
            route_end_yaw_rad = rc_route[-1][2]
            rc_parameters["navigation_goal_xyyaw"] = [
                2.317791700363159,
                0.9919289946556091,
                # Placeholder required by NavigateToPose.  Completion is
                # position-only, so final yaw is intentionally irrelevant.
                float(route_end_yaw_rad),
            ]
            rc_parameters["navigation_accept_position_m"] = 0.10
            rc_parameters["navigation_accept_yaw_rad"] = math.pi

            # The button platform already exists physically in Isaac.
            # Add the same rectangle to rc-1's navigation map so Nav2 can
            # plan continuously from the RC course into the terminal zone.
            button_nav_bounds = [
                float(min(platform_start_x_m, platform_end_x_m)),
                float(max(platform_start_x_m, platform_end_x_m)),
                float(button_lane_y_m - 0.5 * button_width_m),
                float(button_lane_y_m + 0.5 * button_width_m),
            ]

            navigation_free_rectangles = [
                list(rectangle)
                for rectangle in rc_parameters.get(
                    "navigation_free_rectangles_xy_m",
                    [],
                )
            ]
            navigation_free_rectangles.append(button_nav_bounds)
            rc_parameters[
                "navigation_free_rectangles_xy_m"
            ] = navigation_free_rectangles

            map_bounds = [
                float(value)
                for value in rc_parameters["platform_bounds_xy_m"]
            ]
            rc_parameters["platform_bounds_xy_m"] = [
                min(map_bounds[0], button_nav_bounds[0]),
                max(map_bounds[1], button_nav_bounds[1]),
                min(map_bounds[2], button_nav_bounds[2]),
                max(map_bounds[3], button_nav_bounds[3]),
            ]

        # Leave the validated RC-Car standoff side inside the platform for
        # either +/-X-oriented buttons.
        if nx < -0.5:
            button_x_m = platform_start_x_m + 0.65
        elif nx > 0.5:
            button_x_m = platform_end_x_m - 0.65
        else:
            button_x_m = 0.5 * (
                platform_start_x_m + platform_end_x_m
            )

        button_center = (
            button_x_m,
            button_lane_y_m
            + (0.45 * ny if abs(ny) > 0.5 else 0.0),
            floor_height_m + spec.center_xyz_m[2],
        )
        wall_center = (
            button_center[0] + nx * BUTTON_SUPPORT_OFFSET_M,
            button_center[1] + ny * BUTTON_SUPPORT_OFFSET_M,
            floor_height_m + 0.150,
        )
        if abs(nx) > 0.5:
            wall_size = (
                BUTTON_SUPPORT_THICKNESS_M,
                BUTTON_SUPPORT_TANGENT_M,
                BUTTON_SUPPORT_HEIGHT_M,
            )
            plunger_size = (
                BUTTON_PLUNGER_DEPTH_M,
                BUTTON_PLUNGER_TANGENT_M,
                BUTTON_PLUNGER_HEIGHT_M,
            )
        else:
            wall_size = (
                BUTTON_SUPPORT_TANGENT_M,
                BUTTON_SUPPORT_THICKNESS_M,
                BUTTON_SUPPORT_HEIGHT_M,
            )
            plunger_size = (
                BUTTON_PLUNGER_TANGENT_M,
                BUTTON_PLUNGER_DEPTH_M,
                BUTTON_PLUNGER_HEIGHT_M,
            )
        plunger_name = f"{safe_id}_ButtonPlunger"
        boxes.extend(
            (
                CourseBox(
                    f"{safe_id}_ButtonWall",
                    wall_center,
                    wall_size,
                    (0.35, 0.37, 0.40),
                    semantic="button_support",
                ),
                CourseBox(
                    plunger_name,
                    button_center,
                    plunger_size,
                    (0.85, 0.08, 0.06),
                    semantic="button",
                ),
            )
        )
        buttons.append(
            CompositeButtonFixture(
                task_id,
                plunger_name,
                button_center,
                (float(nx), float(ny)),
            )
        )
        tasks.append(
            {
                "task_id": task_id,
                "type": task_type,
                "seed": seed,
                "parameters": {
                    "button": {
                        "center_xyz_m": list(button_center),
                        "press_direction_world_xy": [float(nx), float(ny)],
                        "plunger_path": (
                            f"/World/CompositeObstacleCourse/{plunger_name}"
                        ),
                        "plunger_stroke_m": BUTTON_PLUNGER_STROKE_M,
                    },
                    "floor_height_m": floor_height_m,
                },
            }
        )
        cursor_x_m = platform_end_x_m
        cursor_y_m = button_lane_y_m

    last_task_type = str(raw_tasks[-1].get("type", "")).strip()

    if last_task_type == "button":
        # The button platform itself is the physical end of the course.
        goal_x_m = cursor_x_m - 0.35
    else:
        goal_x_m = cursor_x_m + 0.55
        boxes.append(
            _support_box(
                "CompositeGoalPlatform",
                cursor_x_m,
                goal_x_m + 0.35,
                floor_height_m,
                travel_width_m,
                "goal_platform",
                center_y_m=cursor_y_m,
            )
        )

    goal = (goal_x_m, cursor_y_m, floor_height_m)
    tasks.append(
        {
            "task_id": "goal",
            "type": "goal",
            "parameters": {"center_xyz_m": list(goal)},
        }
    )
    return CompositeObstacleCourse(
        boxes=tuple(boxes),
        tasks=tuple(tasks),
        buttons=tuple(buttons),
        navigation_cones=tuple(navigation_cones),
        final_floor_height_m=floor_height_m,
        goal_center_xyz_m=goal,
    )


def _collision_box_observations(
    boxes: tuple[CourseBox, ...],
) -> list[dict[str, Any]]:
    """Serialize the actual collidable world boxes used by Isaac."""

    return [
        {
            "name": box.name,
            "center_xyz_m": list(box.center_xyz_m),
            "size_xyz_m": list(box.size_xyz_m),
            "semantic": box.semantic,
            "pitch_deg": box.pitch_deg,
            "yaw_deg": box.yaw_deg,
        }
        for box in boxes
        if box.collidable
    ]


@dataclass(frozen=True)
class ManualObstacleCourse:
    """Geometry plus landmarks needed by the future task-level planner."""

    boxes: tuple[CourseBox, ...]
    gap_interval_x_m: tuple[float, float]
    stair_top_heights_m: tuple[float, ...]
    button_center_xyz_m: tuple[float, float, float]
    exit_center_xyz_m: tuple[float, float, float]

    def to_observation(self) -> dict[str, Any]:
        """Serialize stage landmarks for task-level closed-loop control."""
        gap_near_x_m, gap_far_x_m = self.gap_interval_x_m
        return {
            "frame_id": "world",
            "gap": {
                "near_edge_x_m": gap_near_x_m,
                "far_edge_x_m": gap_far_x_m,
            },
            "ramp": {
                "entry_x_m": -1.55,
                "exit_x_m": -1.10,
                "top_height_m": 0.0,
            },
            "stairs": {
                "top_heights_m": list(self.stair_top_heights_m),
                "first_riser_x_m": 1.25,
                "riser_depth_m": 0.28,
            },
            "button": {
                "center_xyz_m": list(self.button_center_xyz_m),
            },
            "exit": {
                "center_xyz_m": list(self.exit_center_xyz_m),
            },
        }


@dataclass(frozen=True)
class StairTestCourse:
    """Isolated three-riser course for validating the Snake8 gait."""

    boxes: tuple[CourseBox, ...]
    stair_top_heights_m: tuple[float, ...]
    first_riser_x_m: float
    riser_depth_m: float
    spec: UniformStairSpec

    def to_observation(self) -> dict[str, Any]:
        """Serialize stair landmarks and their Isaac collision boxes."""
        return {
            "frame_id": "world",
            "course_profile": "snake8_stair_test",
            "scenario": {
                "generator": "uniform_stair_v1",
                **self.spec.to_dict(),
            },
            "stairs": {
                "top_heights_m": list(self.stair_top_heights_m),
                "first_riser_x_m": self.first_riser_x_m,
                "riser_depth_m": self.riser_depth_m,
            },
            "collision_boxes": _collision_box_observations(self.boxes),
        }


BUTTON_REFERENCE_CENTER_XYZ_M = (
    0.85,
    0.465,
    0.170,
)

# Large floor shared by all isolated button episodes.
#
# The target must move by METRES, not centimetres, so navigation
# is a meaningful part of the button task.
BUTTON_PLATFORM_CENTER_XY_M = (1.10, 0.0)
BUTTON_PLATFORM_SIZE_XY_M = (5.60, 4.20)

BUTTON_SAMPLE_X_RANGE_M = (0.65, 2.95)
BUTTON_SAMPLE_Y_RANGE_M = (-1.20, 1.20)
BUTTON_SAMPLE_Z_RANGE_M = (0.10, 0.17623346377092517)

# Axis-aligned wall orientations deliberately keep the fixture
# simple while exposing the expert to four substantially different
# approach directions.
BUTTON_PRESS_DIRECTIONS_XY = (
    (+1.0, 0.0),
    (-1.0, 0.0),
    (0.0, +1.0),
    (0.0, -1.0),
)

# Preserve the validated physical fixture dimensions.
BUTTON_SUPPORT_OFFSET_M = 0.065
BUTTON_SUPPORT_THICKNESS_M = 0.08
BUTTON_SUPPORT_TANGENT_M = 0.18
BUTTON_SUPPORT_HEIGHT_M = 0.30

BUTTON_PLUNGER_DEPTH_M = 0.04
BUTTON_PLUNGER_TANGENT_M = 0.08
BUTTON_PLUNGER_HEIGHT_M = 0.08
BUTTON_PLUNGER_STROKE_M = 0.004


@dataclass(frozen=True)
class ButtonTargetSpec:
    """Reproducible world-frame pose for one button episode."""

    x_m: float = BUTTON_REFERENCE_CENTER_XYZ_M[0]
    y_m: float = BUTTON_REFERENCE_CENTER_XYZ_M[1]
    z_m: float = BUTTON_REFERENCE_CENTER_XYZ_M[2]
    seed: int | None = None

    # Unit vector pointing from the robot toward the support:
    # this is also the physical plunger-depression direction.
    press_direction_xy: tuple[float, float] = (
        0.0,
        +1.0,
    )

    def __post_init__(self) -> None:

        values = (
            self.x_m,
            self.y_m,
            self.z_m,
            *self.press_direction_xy,
        )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "Button target pose values must be finite"
            )

        direction = tuple(
            float(value)
            for value in self.press_direction_xy
        )

        if (
            direction
            not in BUTTON_PRESS_DIRECTIONS_XY
        ):
            raise ValueError(
                "Button press direction must be "
                "one of +/-X or +/-Y"
            )

    @property
    def center_xyz_m(
        self,
    ) -> tuple[float, float, float]:

        return (
            self.x_m,
            self.y_m,
            self.z_m,
        )

    @property
    def press_yaw_rad(self) -> float:
        nx, ny = self.press_direction_xy
        return math.atan2(ny, nx)

    def to_dict(self) -> dict[str, Any]:

        return {
            "seed":
                self.seed,

            "x_m":
                self.x_m,

            "y_m":
                self.y_m,

            "z_m":
                self.z_m,

            "press_direction_xy": [
                float(value)
                for value
                in self.press_direction_xy
            ],

            "press_yaw_rad":
                self.press_yaw_rad,
        }


def sample_button_target_spec(
    seed: int,
) -> ButtonTargetSpec:
    """Sample one deterministic, spatially diverse button task."""

    generator = random.Random(
        int(seed)
    )

    return ButtonTargetSpec(
        x_m=generator.uniform(
            *BUTTON_SAMPLE_X_RANGE_M
        ),

        y_m=generator.uniform(
            *BUTTON_SAMPLE_Y_RANGE_M
        ),

        z_m=generator.uniform(
            *BUTTON_SAMPLE_Z_RANGE_M
        ),

        seed=int(seed),

        press_direction_xy=generator.choice(
            BUTTON_PRESS_DIRECTIONS_XY
        ),
    )


@dataclass(frozen=True)
class ButtonTestCourse:
    """Large isolated fixture for MobileManipulator8 button validation."""

    boxes: tuple[CourseBox, ...]
    button_center_xyz_m: tuple[float, float, float]
    press_direction_world_xy: tuple[float, float]
    base_standoff_xy_m: tuple[float, float]
    base_standoff_yaw_rad: float
    spec: ButtonTargetSpec

    def to_observation(self) -> dict[str, Any]:

        nx, ny = (
            self.press_direction_world_xy
        )

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
                "mobile_manipulator8_button_test",

            "scenario": {
                "generator":
                    "button_target_pose_v2",

                **self.spec.to_dict(),
            },

            "button": {
                "center_xyz_m":
                    list(
                        self.button_center_xyz_m
                    ),

                "press_direction_world_xy": [
                    float(nx),
                    float(ny),
                ],

                "press_direction_world_xyz": [
                    float(nx),
                    float(ny),
                    0.0,
                ],

                "press_yaw_rad":
                    math.atan2(ny, nx),

                "base_standoff_xy_m":
                    list(
                        self.base_standoff_xy_m
                    ),

                "base_standoff_yaw_rad":
                    self.base_standoff_yaw_rad,

                "plunger_depth_m":
                    BUTTON_PLUNGER_DEPTH_M,

                "plunger_stroke_m":
                    BUTTON_PLUNGER_STROKE_M,

                "face_size_tangent_z_m": [
                    BUTTON_PLUNGER_TANGENT_M,
                    BUTTON_PLUNGER_HEIGHT_M,
                ],

                "support_offset_m":
                    BUTTON_SUPPORT_OFFSET_M,
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

            "collision_boxes":
                _collision_box_observations(
                    self.boxes
                ),
        }


@dataclass(frozen=True)
class GapTestCourse:
    """Flat equal-height banks separated by one isolated gap."""

    boxes: tuple[CourseBox, ...]
    gap_interval_x_m: tuple[float, float]
    spec: CoplanarGapSpec

    def to_observation(self) -> dict[str, Any]:
        near_x_m, far_x_m = self.gap_interval_x_m
        return {
            "frame_id": "world",
            "course_profile": "snake8_gap_test",
            "scenario": {
                "generator": "coplanar_gap_v1",
                **self.spec.to_dict(),
            },
            "gap": {
                "near_edge_x_m": near_x_m,
                "far_edge_x_m": far_x_m,
                "width_m": far_x_m - near_x_m,
            },
        }




@dataclass(frozen=True)
class RCPlanarTestCourse:
    """Large flat stage carrying a seeded Nav2 route for RC-Car8."""

    boxes: tuple[CourseBox, ...]
    spec: RCPlanarSpec

    def to_observation(self) -> dict[str, Any]:
        # IMPORTANT:
        # ``self.spec`` describes the seeded NOMINAL route/reference.
        # It is not necessarily the physical road installed in Isaac.
        #
        # The stage, Nav2 OccupancyGrid, traffic cones and terminal goal
        # are all generated by rc_car_planar_obstacle_layout().  Serialize
        # that exact layout as part of every environment observation so IL
        # sees the world in which the expert actually acted.
        physical_track = rc_car_planar_obstacle_layout(
            self.spec.seed,
            platform_center_x_m=1.10,
            platform_size_x_m=self.spec.platform_size_x_m,
            platform_size_y_m=self.spec.platform_size_y_m,
        )

        return {
            "frame_id": "world",
            "course_profile": "rc_car8_planar_nav2",
            "scenario": {
                "generator": "rc_car_planar_route_v1",
                **self.spec.to_dict(),
            },

            # Keep the historical fields for backwards compatibility, but
            # state explicitly that these are the nominal reference.
            "navigation": {
                "controller": "nav2",
                "route_kind": self.spec.route_kind,
                "waypoints_xyyaw": [
                    list(pose) for pose in self.spec.waypoints_xyyaw
                ],
                "nominal_route": {
                    "route_kind": self.spec.route_kind,
                    "waypoints_xyyaw": [
                        list(pose)
                        for pose in self.spec.waypoints_xyyaw
                    ],
                },
                "physical_goal_xyyaw": list(
                    physical_track["goal_xyyaw"]
                ),
                "goal_xy_tolerance_m": 0.05,
                "goal_yaw_tolerance_rad": 0.12,
            },

            # Ground-truth geometry actually instantiated in Isaac and used
            # to construct the Nav2 map.
            #
            # Includes:
            #   - complete physical centerline
            #   - straight/curve profile
            #   - direction / angle / radius
            #   - corridor width
            #   - platform bounds
            #   - finish pose
            #   - cone count and exact cone centres
            #   - cone geometry
            #   - RC-Car physical footprint / required clearance
            "physical_track": physical_track,

            "collision_boxes": _collision_box_observations(self.boxes),
        }



def rc_car_planar_test_course(
    spec: RCPlanarSpec | None = None,
) -> RCPlanarTestCourse:
    """Return a LARGE support platform for the procedural RC-Car8 track."""

    if spec is None:
        spec = sample_rc_car_planar_spec(0)

    # Deliberately keep the complete original 3.2 m platform width.
    # Navigation confinement now belongs to the OccupancyGrid, not to
    # the physical floor supporting self-assembly.
    platform = CourseBox(
        "RCPlanarPlatform",
        (1.10, 0.0, -0.01),
        (
            spec.platform_size_x_m,
            spec.platform_size_y_m,
            spec.platform_thickness_m,
        ),
        (0.78, 0.80, 0.82),
        semantic="rc_car_planar_support_platform",
    )

    return RCPlanarTestCourse(
        boxes=(platform,),
        spec=spec,
    )




def manual_obstacle_course() -> ManualObstacleCourse:
    """Return a compact +X course sized for an eight-module morphology.

    The assembly starts around the origin.  The course then presents a real
    floor discontinuity, three low steps, a wall-mounted button and an exit
    marker.  The gap is not a painted obstacle: no collider covers its open
    interval.
    """

    platform_color = (0.24, 0.27, 0.31)
    stair_color = (0.30, 0.34, 0.39)
    return ManualObstacleCourse(
        boxes=(
            CourseBox(
                "RearAssemblyPlatform",
                (-2.00, 0.0, -0.13),
                (1.20, 1.50, 0.02),
                platform_color,
                semantic="rear_start_platform",
            ),
            CourseBox(
                "ApproachRamp",
                (-1.325, 0.0, -0.0793),
                (0.465, 1.20, 0.04),
                platform_color,
                semantic="approach_ramp",
                pitch_deg=-14.93,
            ),
            CourseBox(
                "StartPlatform",
                (-0.225, 0.0, -0.01),
                (1.75, 1.50, 0.02),
                platform_color,
                semantic="start_platform",
            ),
            CourseBox(
                "GapLanding",
                (1.05, 0.0, -0.01),
                (0.40, 1.20, 0.02),
                platform_color,
                semantic="gap_landing",
            ),
            CourseBox(
                "Stair01",
                (1.39, 0.0, 0.0325),
                (0.28, 1.20, 0.065),
                stair_color,
                semantic="stair",
            ),
            CourseBox(
                "Stair02",
                (1.67, 0.0, 0.065),
                (0.28, 1.20, 0.13),
                stair_color,
                semantic="stair",
            ),
            CourseBox(
                "Stair03",
                (1.95, 0.0, 0.0975),
                (0.28, 1.20, 0.195),
                stair_color,
                semantic="stair",
            ),
            CourseBox(
                "UpperDeck",
                (2.97, 0.0, 0.0975),
                (1.76, 1.20, 0.195),
                platform_color,
                semantic="upper_platform",
            ),
            CourseBox(
                "ButtonWall",
                (2.65, 0.53, 0.345),
                (0.18, 0.08, 0.30),
                (0.35, 0.37, 0.40),
                semantic="button_support",
            ),
            CourseBox(
                "ButtonPlunger",
                (2.65, 0.475, 0.365),
                (0.08, 0.04, 0.08),
                (0.85, 0.08, 0.06),
                semantic="button",
            ),
            CourseBox(
                "ExitLeft",
                (3.55, -0.47, 0.385),
                (0.05, 0.05, 0.38),
                (0.12, 0.82, 0.25),
                collidable=False,
                semantic="exit_marker",
            ),
            CourseBox(
                "ExitRight",
                (3.55, 0.47, 0.385),
                (0.05, 0.05, 0.38),
                (0.12, 0.82, 0.25),
                collidable=False,
                semantic="exit_marker",
            ),
            CourseBox(
                "ExitTop",
                (3.55, 0.0, 0.575),
                (0.05, 0.99, 0.05),
                (0.12, 0.82, 0.25),
                collidable=False,
                semantic="exit_marker",
            ),
        ),
        gap_interval_x_m=(0.65, 0.85),
        stair_top_heights_m=(0.065, 0.13, 0.195),
        button_center_xyz_m=(2.65, 0.455, 0.365),
        exit_center_xyz_m=(3.55, 0.0, 0.385),
    )


def mobile_manipulator_button_test_course(
    spec: ButtonTargetSpec | None = None,
) -> ButtonTestCourse:
    """Return a large floor with one oriented wall-mounted button."""

    platform_color = (
        0.24,
        0.27,
        0.31,
    )

    spec = (
        spec
        or ButtonTargetSpec()
    )

    button_center = (
        spec.center_xyz_m
    )

    nx, ny = (
        spec.press_direction_xy
    )

    # The support always lies BEHIND the plunger along the
    # actual press direction.
    wall_center = (
        button_center[0]
        + nx * BUTTON_SUPPORT_OFFSET_M,

        button_center[1]
        + ny * BUTTON_SUPPORT_OFFSET_M,

        0.150,
    )

    # +/-X and +/-Y fixtures remain axis-aligned.
    #
    # The thin dimension is always aligned with the
    # physical press direction.
    if abs(nx) > 0.5:

        wall_size = (
            BUTTON_SUPPORT_THICKNESS_M,
            BUTTON_SUPPORT_TANGENT_M,
            BUTTON_SUPPORT_HEIGHT_M,
        )

        plunger_size = (
            BUTTON_PLUNGER_DEPTH_M,
            BUTTON_PLUNGER_TANGENT_M,
            BUTTON_PLUNGER_HEIGHT_M,
        )

    else:

        wall_size = (
            BUTTON_SUPPORT_TANGENT_M,
            BUTTON_SUPPORT_THICKNESS_M,
            BUTTON_SUPPORT_HEIGHT_M,
        )

        plunger_size = (
            BUTTON_PLUNGER_TANGENT_M,
            BUTTON_PLUNGER_DEPTH_M,
            BUTTON_PLUNGER_HEIGHT_M,
        )

    platform_cx, platform_cy = (
        BUTTON_PLATFORM_CENTER_XY_M
    )

    platform_sx, platform_sy = (
        BUTTON_PLATFORM_SIZE_XY_M
    )

    return ButtonTestCourse(
        boxes=(
            CourseBox(
                "TestPlatform",

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

                platform_color,

                semantic=
                    "button_test_platform",
            ),

            CourseBox(
                "ButtonWall",
                wall_center,
                wall_size,
                (0.35, 0.37, 0.40),
                semantic="button_support",
            ),

            CourseBox(
                "ButtonPlunger",
                button_center,
                plunger_size,
                (0.85, 0.08, 0.06),
                semantic="button",
            ),
        ),

        button_center_xyz_m=
            button_center,

        press_direction_world_xy=(
            float(nx),
            float(ny),
        ),

        base_standoff_xy_m=(
            button_center[0]
            - nx * 0.20,

            button_center[1]
            - ny * 0.20,
        ),

        # Preserve the historical convention:
        # +Y => +90 degrees.
        base_standoff_yaw_rad=
            math.atan2(ny, nx),

        spec=spec,
    )


def snake8_gap_test_course(
    spec: CoplanarGapSpec | None = None,
) -> GapTestCourse:
    """Return a parameterized gap with coplanar approach and landing banks."""

    spec = spec or CoplanarGapSpec()
    platform_color = (0.24, 0.27, 0.31)
    near_x_m = spec.near_edge_x_m
    far_x_m = spec.far_edge_x_m
    approach_start_x_m = spec.approach_start_x_m
    landing_end_x_m = spec.landing_end_x_m
    return GapTestCourse(
        boxes=(
            CourseBox(
                "NearBank",
                (
                    0.5 * (approach_start_x_m + near_x_m),
                    0.0,
                    -0.5 * spec.bank_thickness_m,
                ),
                (
                    near_x_m - approach_start_x_m,
                    spec.bank_width_m,
                    spec.bank_thickness_m,
                ),
                platform_color,
                semantic="gap_test_near_bank",
            ),
            CourseBox(
                "FarBank",
                (
                    0.5 * (far_x_m + landing_end_x_m),
                    0.0,
                    -0.5 * spec.bank_thickness_m,
                ),
                (
                    landing_end_x_m - far_x_m,
                    spec.bank_width_m,
                    spec.bank_thickness_m,
                ),
                platform_color,
                semantic="gap_test_far_bank",
            ),
        ),
        gap_interval_x_m=(near_x_m, far_x_m),
        spec=spec,
    )


def snake8_stair_test_course(
    spec: UniformStairSpec | None = None,
) -> StairTestCourse:
    """Return parameterized equal risers preceded by an assembly platform.

    A SMORES-EP wheel is roughly 62 mm in diameter.  Each riser is therefore
    constrained below one 77.77 mm serial-chain link.  The default preserves
    the physically validated three-riser fixture exactly.
    """

    spec = spec or UniformStairSpec()
    platform_color = (0.24, 0.27, 0.31)
    stair_color = (0.34, 0.38, 0.43)
    approach_start_x_m = -1.0
    approach_length_m = spec.first_riser_x_m - approach_start_x_m
    boxes: list[CourseBox] = [
        CourseBox(
            "StartPlatform",
            (
                approach_start_x_m + 0.5 * approach_length_m,
                0.0,
                -0.01,
            ),
            (approach_length_m, spec.width_m, 0.02),
            platform_color,
            semantic="stair_test_start",
        )
    ]
    for index, top_height_m in enumerate(spec.top_heights_m):
        boxes.append(
            CourseBox(
                f"Stair{index + 1:02d}",
                (
                    spec.first_riser_x_m
                    + (index + 0.5) * spec.tread_depth_m,
                    0.0,
                    0.5 * top_height_m,
                ),
                (spec.tread_depth_m, spec.width_m, top_height_m),
                stair_color,
                semantic="stair_test_riser",
            )
        )
    upper_deck_start_x_m = (
        spec.first_riser_x_m + spec.step_count * spec.tread_depth_m
    )
    boxes.append(
        CourseBox(
            "UpperDeck",
            (
                upper_deck_start_x_m + 0.5 * spec.upper_deck_length_m,
                0.0,
                0.5 * spec.top_heights_m[-1],
            ),
            (
                spec.upper_deck_length_m,
                spec.width_m,
                spec.top_heights_m[-1],
            ),
            platform_color,
            semantic="stair_test_upper_deck",
        )
    )
    return StairTestCourse(
        boxes=tuple(boxes),
        stair_top_heights_m=spec.top_heights_m,
        first_riser_x_m=spec.first_riser_x_m,
        riser_depth_m=spec.tread_depth_m,
        spec=spec,
    )


def _install_course_boxes(
    stage: Any,
    root_path: str,
    boxes: tuple[CourseBox, ...],
) -> None:
    """Install shared course-box geometry and physics material."""

    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    if stage.GetPrimAtPath("/World/Ground"):
        stage.RemovePrim("/World/Ground")
    root = UsdGeom.Xform.Define(stage, root_path)
    material = UsdShade.Material(
        stage.GetPrimAtPath("/World/materials/dynamic_ground")
    )
    for element in boxes:
        cube = UsdGeom.Cube.Define(
            stage,
            f"{root.GetPath()}/{element.name}",
        )
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*element.center_xyz_m))
        if element.yaw_deg:
            cube.AddRotateZOp().Set(element.yaw_deg)
        if element.pitch_deg:
            cube.AddRotateYOp().Set(element.pitch_deg)
        cube.AddScaleOp().Set(Gf.Vec3f(*element.size_xyz_m))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*element.color_rgb)])
        cube.GetPrim().CreateAttribute(
            "mssr:obstacleSemantic",
            Sdf.ValueTypeNames.String,
        ).Set(element.semantic)
        if element.collidable:
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
                material,
                UsdShade.Tokens.weakerThanDescendants,
                "physics",
            )


def _install_composite_navigation_cones(
    stage: Any,
    root_path: str,
    cones: tuple[CompositeNavigationCone, ...],
) -> None:
    """Install visible cone meshes and stable cylindrical colliders."""

    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    cones_root = f"{root_path}/NavigationCones"
    UsdGeom.Xform.Define(stage, cones_root)
    for index, item in enumerate(cones, start=1):
        safe_task = "".join(
            character if character.isalnum() else "_"
            for character in item.task_id
        )
        cone_root = f"{cones_root}/{safe_task}_Cone{index:02d}"
        visual = UsdGeom.Cone.Define(stage, f"{cone_root}/visual")
        visual.CreateAxisAttr(UsdGeom.Tokens.z)
        visual.CreateRadiusAttr(item.radius_m)
        visual.CreateHeightAttr(item.height_m)
        visual.CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.30, 0.02)])
        visual.GetPrim().CreateAttribute(
            "mssr:obstacleSemantic",
            Sdf.ValueTypeNames.String,
        ).Set("navigation_cone")
        UsdGeom.Xformable(visual).AddTranslateOp().Set(
            Gf.Vec3d(*item.center_xyz_m)
        )

        collider = UsdGeom.Cylinder.Define(stage, f"{cone_root}/collision")
        collider.CreateAxisAttr(UsdGeom.Tokens.z)
        collider.CreateRadiusAttr(item.radius_m)
        collider.CreateHeightAttr(item.height_m)
        collider.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdGeom.Xformable(collider).AddTranslateOp().Set(
            Gf.Vec3d(*item.center_xyz_m)
        )
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())


def install_manual_obstacle_course(stage: Any) -> ManualObstacleCourse:
    """Replace the infinite floor with the manual, segmented course."""

    course = manual_obstacle_course()
    _install_course_boxes(stage, "/World/ManualObstacleCourse", course.boxes)
    return course


def _install_button_prismatic_joint(
    stage: Any,
    *,
    root_path: str,
    plunger_name: str,
    center_xyz_m: tuple[float, float, float],
    press_direction_world_xy: tuple[float, float],
    joint_name: str,
) -> None:
    """Make one course plunger independently depressible."""

    from pxr import Gf, Sdf, UsdPhysics

    plunger_path = f"{root_path}/{plunger_name}"
    plunger_prim = stage.GetPrimAtPath(plunger_path)
    if not plunger_prim or not plunger_prim.IsValid():
        raise RuntimeError(f"Button plunger prim was not created: {plunger_path}")
    UsdPhysics.RigidBodyAPI.Apply(plunger_prim)
    joint = UsdPhysics.PrismaticJoint.Define(
        stage,
        f"{root_path}/{joint_name}",
    )
    nx, ny = press_direction_world_xy
    joint_axis, joint_sign = ("X", nx) if abs(nx) > 0.5 else ("Y", ny)
    joint.CreateAxisAttr(joint_axis)
    if joint_sign > 0.0:
        joint.CreateLowerLimitAttr(0.0)
        joint.CreateUpperLimitAttr(BUTTON_PLUNGER_STROKE_M)
    else:
        joint.CreateLowerLimitAttr(-BUTTON_PLUNGER_STROKE_M)
        joint.CreateUpperLimitAttr(0.0)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(plunger_path)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*center_xyz_m))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))


def install_composite_obstacle_course(
    stage: Any,
    mission: Mapping[str, Any],
    validated_seeds: Mapping[str, set[int] | frozenset[int]],
) -> CompositeObstacleCourse:
    """Install an ordered multi-obstacle course and all button joints."""

    course = composite_obstacle_course(mission, validated_seeds)
    root_path = "/World/CompositeObstacleCourse"
    _install_course_boxes(stage, root_path, course.boxes)
    _install_composite_navigation_cones(
        stage,
        root_path,
        course.navigation_cones,
    )
    for index, fixture in enumerate(course.buttons):
        _install_button_prismatic_joint(
            stage,
            root_path=root_path,
            plunger_name=fixture.plunger_name,
            center_xyz_m=fixture.center_xyz_m,
            press_direction_world_xy=fixture.press_direction_world_xy,
            joint_name=f"ButtonPrismaticJoint{index + 1:02d}",
        )
    return course


def install_snake8_stair_test_course(
    stage: Any,
    spec: UniformStairSpec | None = None,
) -> StairTestCourse:
    """Replace the infinite floor with one isolated uniform stair course."""

    course = snake8_stair_test_course(spec)
    _install_course_boxes(stage, "/World/Snake8StairTestCourse", course.boxes)
    return course


def install_mobile_manipulator_button_test_course(
    stage: Any,
    spec: ButtonTargetSpec | None = None,
) -> ButtonTestCourse:
    """Replace the infinite floor with the isolated button fixture."""

    from pxr import Gf, Sdf, UsdPhysics

    course = mobile_manipulator_button_test_course(spec)
    root_path = "/World/MobileManipulatorButtonTestCourse"

    _install_course_boxes(
        stage,
        root_path,
        course.boxes,
    )

    # ButtonPlunger is the only dynamic element of the fixture.
    # body0 is intentionally omitted: USD Physics interprets the
    # empty body relationship as a world anchor.
    plunger_path = f"{root_path}/ButtonPlunger"
    plunger_prim = stage.GetPrimAtPath(plunger_path)
    if not plunger_prim or not plunger_prim.IsValid():
        raise RuntimeError(
            f"Button plunger prim was not created: {plunger_path}"
        )

    UsdPhysics.RigidBodyAPI.Apply(plunger_prim)

    joint = UsdPhysics.PrismaticJoint.Define(
        stage,
        f"{root_path}/ButtonPrismaticJoint",
    )
    nx, ny = (
        course.press_direction_world_xy
    )

    if abs(nx) > 0.5:
        joint_axis = "X"
        joint_sign = nx
    else:
        joint_axis = "Y"
        joint_sign = ny

    joint.CreateAxisAttr(
        joint_axis
    )

    # Coordinate zero is always the unpressed pose.
    #
    # For +X/+Y the plunger moves from 0 to +stroke.
    # For -X/-Y it moves from 0 to -stroke.
    if joint_sign > 0.0:

        joint.CreateLowerLimitAttr(
            0.0
        )

        joint.CreateUpperLimitAttr(
            BUTTON_PLUNGER_STROKE_M
        )

    else:

        joint.CreateLowerLimitAttr(
            -BUTTON_PLUNGER_STROKE_M
        )

        joint.CreateUpperLimitAttr(
            0.0
        )

    joint.CreateBody1Rel().SetTargets(
        [Sdf.Path(plunger_path)]
    )

    # World-side joint frame coincides with the nominal,
    # unpressed plunger centre.  The body-side frame is the
    # plunger origin, therefore joint coordinate 0 is its rest pose.
    joint.CreateLocalPos0Attr().Set(
        Gf.Vec3f(*course.button_center_xyz_m)
    )
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
    joint.CreateLocalPos1Attr().Set(
        Gf.Vec3f(0.0, 0.0, 0.0)
    )
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))

    return course


def install_snake8_gap_test_course(
    stage: Any,
    spec: CoplanarGapSpec | None = None,
) -> GapTestCourse:
    """Replace the infinite floor with the isolated equal-bank gap."""

    course = snake8_gap_test_course(spec)
    _install_course_boxes(stage, "/World/Snake8GapTestCourse", course.boxes)
    return course




def install_rc_car_planar_test_course(
    stage: Any,
    spec: RCPlanarSpec | None = None,
) -> RCPlanarTestCourse:
    """Install wide support floor plus visual procedural road."""

    import math
    from pxr import Gf, UsdGeom, UsdPhysics

    course = rc_car_planar_test_course(spec)

    _install_course_boxes(
        stage,
        "/World/RCPlanarTestCourse",
        course.boxes,
    )

    # Large support floor is needed physically during self-assembly,
    # but it must not look like part of the navigation course.
    support_prim = stage.GetPrimAtPath(
        "/World/RCPlanarTestCourse/RCPlanarPlatform"
    )
    if support_prim and support_prim.IsValid():
        UsdGeom.Imageable(support_prim).MakeInvisible()

    layout = rc_car_planar_obstacle_layout(
        course.spec.seed,
        platform_center_x_m=1.10,
        platform_size_x_m=course.spec.platform_size_x_m,
        platform_size_y_m=course.spec.platform_size_y_m,
    )

    root = "/World/RCPlanarTestCourse"

    # --------------------------------------------------------
    # VISUAL ROAD
    # --------------------------------------------------------
    road_root = f"{root}/Road"

    if stage.GetPrimAtPath(road_root):
        stage.RemovePrim(road_root)

    UsdGeom.Xform.Define(stage, road_root)

    centerline = [
        (float(x), float(y))
        for x, y in layout["centerline_xy_m"]
    ]

    road_width = float(
        layout["corridor_width_m"]
    )

    for index, (a, b) in enumerate(
        zip(centerline[:-1], centerline[1:])
    ):
        dx = b[0] - a[0]
        dy = b[1] - a[1]

        length = math.hypot(dx, dy)

        if length <= 1.0e-6:
            continue

        mx = 0.5 * (a[0] + b[0])
        my = 0.5 * (a[1] + b[1])

        yaw_deg = math.degrees(
            math.atan2(dy, dx)
        )

        tile = UsdGeom.Cube.Define(
            stage,
            f"{road_root}/Segment{index:02d}",
        )

        tile.CreateSizeAttr(1.0)

        tile.CreateDisplayColorAttr(
            [Gf.Vec3f(0.22, 0.24, 0.27)]
        )

        xf = UsdGeom.Xformable(tile)

        xf.AddTranslateOp().Set(
            Gf.Vec3d(mx, my, 0.0005)
        )

        xf.AddRotateZOp().Set(yaw_deg)

        xf.AddScaleOp().Set(
            Gf.Vec3f(
                length + 0.035,
                road_width,
                0.001,
            )
        )

    # Wide visual start box.
    sx0, sx1, sy0, sy1 = [
        float(v)
        for v in layout["start_pad_bounds_xy_m"]
    ]

    start = UsdGeom.Cube.Define(
        stage,
        f"{road_root}/AssemblyStartPad",
    )

    start.CreateSizeAttr(1.0)

    start.CreateDisplayColorAttr(
        [Gf.Vec3f(0.30, 0.32, 0.35)]
    )

    start_xf = UsdGeom.Xformable(start)

    start_xf.AddTranslateOp().Set(
        Gf.Vec3d(
            0.5 * (sx0 + sx1),
            0.5 * (sy0 + sy1),
            0.0005,
        )
    )

    start_xf.AddScaleOp().Set(
        Gf.Vec3f(
            sx1 - sx0,
            sy1 - sy0,
            0.001,
        )
    )

    # --------------------------------------------------------
    # TRUE TRAFFIC CONES
    # --------------------------------------------------------
    cones_root = f"{root}/NavigationCones"

    if stage.GetPrimAtPath(cones_root):
        stage.RemovePrim(cones_root)

    UsdGeom.Xform.Define(stage, cones_root)

    radius = float(layout["cone_radius_m"])
    height = float(layout["cone_height_m"])

    for index, (x_m, y_m) in enumerate(
        layout["cone_centers_xy_m"],
        start=1,
    ):
        cone_root = (
            f"{cones_root}/Cone{index:02d}"
        )

        cone = UsdGeom.Cone.Define(
            stage,
            f"{cone_root}/visual",
        )

        cone.CreateAxisAttr(UsdGeom.Tokens.z)
        cone.CreateRadiusAttr(radius)
        cone.CreateHeightAttr(height)

        cone.CreateDisplayColorAttr(
            [Gf.Vec3f(1.0, 0.30, 0.02)]
        )

        UsdGeom.Xformable(
            cone
        ).AddTranslateOp().Set(
            Gf.Vec3d(
                float(x_m),
                float(y_m),
                0.5 * height,
            )
        )

        # Cylinder collision proxy:
        # stable physically but hidden visually.
        collider = UsdGeom.Cylinder.Define(
            stage,
            f"{cone_root}/collision",
        )

        collider.CreateAxisAttr(
            UsdGeom.Tokens.z
        )

        collider.CreateRadiusAttr(radius)
        collider.CreateHeightAttr(height)

        collider.CreateVisibilityAttr(
            UsdGeom.Tokens.invisible
        )

        UsdGeom.Xformable(
            collider
        ).AddTranslateOp().Set(
            Gf.Vec3d(
                float(x_m),
                float(y_m),
                0.5 * height,
            )
        )

        UsdPhysics.CollisionAPI.Apply(
            collider.GetPrim()
        )

    # --------------------------------------------------------
    # CHECKERED FINISH LINE, perpendicular to local road.
    # --------------------------------------------------------
    finish_root = f"{root}/FinishLine"

    if stage.GetPrimAtPath(finish_root):
        stage.RemovePrim(finish_root)

    UsdGeom.Xform.Define(
        stage,
        finish_root,
    )

    gx = float(layout["finish_x_m"])
    gy = float(layout["finish_y_m"])
    gyaw = float(layout["finish_yaw_rad"])

    nx = -math.sin(gyaw)
    ny = math.cos(gyaw)

    tiles = 10
    tile_width = road_width / tiles

    for index in range(tiles):
        lateral = (
            -0.5 * road_width
            + (index + 0.5) * tile_width
        )

        tx = gx + nx * lateral
        ty = gy + ny * lateral

        tile = UsdGeom.Cube.Define(
            stage,
            f"{finish_root}/Tile{index:02d}",
        )

        tile.CreateSizeAttr(1.0)

        shade = (
            0.05
            if index % 2 == 0
            else 0.95
        )

        tile.CreateDisplayColorAttr(
            [Gf.Vec3f(shade, shade, shade)]
        )

        xf = UsdGeom.Xformable(tile)

        xf.AddTranslateOp().Set(
            Gf.Vec3d(
                tx,
                ty,
                0.009,
            )
        )

        xf.AddRotateZOp().Set(
            math.degrees(gyaw)
        )

        xf.AddScaleOp().Set(
            Gf.Vec3f(
                0.055,
                tile_width,
                0.008,
            )
        )

    return course
