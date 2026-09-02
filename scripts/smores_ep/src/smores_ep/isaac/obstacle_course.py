"""Manual task-achievement course for assembled SMORES-EP morphologies."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any


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
        step_count=generator.randint(count_min, count_max),
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
    platform_size_x_m: float = 4.80
    platform_size_y_m: float = 3.20
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
    corridor_width_m = 0.840

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
        -0.78,
        +0.78,
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
        cone_count = rng.randint(
            5,
            7,
        )

        # Leave some free distance after start and before finish.
        first_fraction = rng.uniform(
            0.20,
            0.25,
        )

        last_fraction = rng.uniform(
            0.80,
            0.87,
        )

        fractions = []

        for index in range(
            cone_count
        ):
            if cone_count == 1:
                alpha = 0.5
            else:
                alpha = (
                    index
                    / (cone_count - 1)
                )

            fraction = (
                first_fraction
                + alpha
                * (
                    last_fraction
                    - first_fraction
                )
            )

            # Small longitudinal perturbation.
            if (
                index > 0
                and index
                < cone_count - 1
            ):
                fraction += rng.uniform(
                    -0.018,
                    +0.018,
                )

            fractions.append(
                fraction
            )

        cone_centers = []

        for index, fraction in enumerate(
            fractions
        ):
            x, y, tx, ty = (
                pose_at_distance(
                    total_length
                    * fraction
                )
            )

            # Local road normal.
            nx = -ty
            ny = tx

            # Alternating side relative to LOCAL track direction.
            sign = (
                +1.0
                if index % 2 == 0
                else -1.0
            )

            lateral_offset = (
                sign
                * rng.uniform(
                    0.055,
                    0.105,
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
            "rc_car_planar_track_v3",

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
            "known_environment": rc_car_planar_obstacle_layout(
                self.spec.seed,
                platform_center_x_m=1.10,
                platform_size_x_m=self.spec.platform_size_x_m,
                platform_size_y_m=self.spec.platform_size_y_m,
            ),
            "collision_boxes": _collision_box_observations(self.boxes),
        }


@dataclass(frozen=True)
class ButtonTestCourse:
    """Flat isolated fixture for MobileManipulator8 button validation."""

    boxes: tuple[CourseBox, ...]
    button_center_xyz_m: tuple[float, float, float]
    base_standoff_xy_m: tuple[float, float]
    base_standoff_yaw_rad: float

    def to_observation(self) -> dict[str, Any]:
        return {
            "frame_id": "world",
            "course_profile": "mobile_manipulator8_button_test",
            "button": {
                "center_xyz_m": list(self.button_center_xyz_m),
                "base_standoff_xy_m": list(self.base_standoff_xy_m),
                "base_standoff_yaw_rad": self.base_standoff_yaw_rad,
            },
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
        return {
            "frame_id": "world",
            "course_profile": "rc_car8_planar_nav2",
            "scenario": {
                "generator": "rc_car_planar_route_v1",
                **self.spec.to_dict(),
            },
            "navigation": {
                "controller": "nav2",
                "route_kind": self.spec.route_kind,
                "waypoints_xyyaw": [
                    list(pose) for pose in self.spec.waypoints_xyyaw
                ],
                "goal_xy_tolerance_m": 0.05,
                "goal_yaw_tolerance_rad": 0.12,
            },
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


def mobile_manipulator_button_test_course() -> ButtonTestCourse:
    """Return a continuous floor with only the wall-mounted button."""

    platform_color = (0.24, 0.27, 0.31)
    button_center = (0.85, 0.475, 0.170)
    return ButtonTestCourse(
        boxes=(
            CourseBox(
                "TestPlatform",
                (0.25, 0.0, -0.01),
                (2.50, 1.80, 0.02),
                platform_color,
                semantic="button_test_platform",
            ),
            CourseBox(
                "ButtonWall",
                (button_center[0], 0.53, 0.150),
                (0.18, 0.08, 0.30),
                (0.35, 0.37, 0.40),
                semantic="button_support",
            ),
            CourseBox(
                "ButtonPlunger",
                button_center,
                (0.08, 0.04, 0.08),
                (0.85, 0.08, 0.06),
                semantic="button",
            ),
        ),
        button_center_xyz_m=button_center,
        base_standoff_xy_m=(button_center[0], button_center[1] - 0.20),
        base_standoff_yaw_rad=0.5 * math.pi,
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


def install_manual_obstacle_course(stage: Any) -> ManualObstacleCourse:
    """Replace the infinite floor with the manual, segmented course."""

    course = manual_obstacle_course()
    _install_course_boxes(stage, "/World/ManualObstacleCourse", course.boxes)
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
) -> ButtonTestCourse:
    """Replace the infinite floor with the isolated button fixture."""

    course = mobile_manipulator_button_test_course()
    _install_course_boxes(
        stage,
        "/World/MobileManipulatorButtonTestCourse",
        course.boxes,
    )
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
