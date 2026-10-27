"""Seeded Phase-1 RC-Car8 MARL training course.

The course deliberately reuses the validated RC-Car8 planar geometry
instead of introducing an independent MARL-only terrain generator.

Generator-only centerline information is retained by the course object for
reward computation, but is deliberately omitted from the public physical
observation so it cannot become an actor/critic oracle path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from .obstacle_course import (
    CourseBox,
    rc_car_planar_obstacle_layout,
)


@dataclass(frozen=True)
class OrientedPlatform:
    """One rectangular support area in the world XY plane."""

    center_xy_m: tuple[float, float]
    yaw_rad: float
    size_xy_m: tuple[float, float]

    def __post_init__(self) -> None:
        values = (
            *self.center_xy_m,
            self.yaw_rad,
            *self.size_xy_m,
        )

        if not all(
            math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                "Platform geometry must be finite"
            )

        if (
            self.size_xy_m[0] <= 0.0
            or self.size_xy_m[1] <= 0.0
        ):
            raise ValueError(
                "Platform dimensions must be positive"
            )

    def contains_xy(
        self,
        point_xy_m: tuple[float, float],
        *,
        margin_m: float = 0.0,
    ) -> bool:
        """Return whether a world XY point lies inside the platform."""

        if (
            not math.isfinite(margin_m)
            or margin_m < 0.0
        ):
            raise ValueError(
                "Platform margin must be finite and nonnegative"
            )

        dx = (
            float(point_xy_m[0])
            - self.center_xy_m[0]
        )
        dy = (
            float(point_xy_m[1])
            - self.center_xy_m[1]
        )

        c = math.cos(self.yaw_rad)
        s = math.sin(self.yaw_rad)

        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy

        half_x = (
            0.5 * self.size_xy_m[0]
            - margin_m
        )
        half_y = (
            0.5 * self.size_xy_m[1]
            - margin_m
        )

        if half_x < 0.0 or half_y < 0.0:
            return False

        return (
            abs(local_x) <= half_x
            and abs(local_y) <= half_y
        )


@dataclass(frozen=True)
class Phase1Cone:
    """One rigid static cone obstacle."""

    center_xy_m: tuple[float, float]
    radius_m: float
    height_m: float
    static: bool = True

    def __post_init__(self) -> None:
        if (
            not all(
                math.isfinite(value)
                for value in (
                    *self.center_xy_m,
                    self.radius_m,
                    self.height_m,
                )
            )
            or self.radius_m <= 0.0
            or self.height_m <= 0.0
        ):
            raise ValueError(
                "Invalid Phase-1 cone geometry"
            )

        if self.static is not True:
            raise ValueError(
                "Phase-1 cones must be static"
            )


@dataclass(frozen=True)
class Phase1TrainingCourse:
    """Pure geometry/semantics contract for one Phase-1 episode."""

    seed: int

    start_platform: OrientedPlatform
    target_platform: OrientedPlatform

    support_boxes: tuple[CourseBox, ...]
    cones: tuple[Phase1Cone, ...]

    centerline_xy_m: tuple[
        tuple[float, float],
        ...
    ]

    road_width_m: float

    vehicle_length_m: float
    vehicle_width_m: float

    cone_radius_m: float
    collision_margin_m: float
    clearance_margin_m: float

    has_curve: bool
    track_profile: str

    forward_only: bool = True

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise TypeError(
                "Phase-1 seed must be an integer"
            )

        if len(self.centerline_xy_m) < 2:
            raise ValueError(
                "Phase-1 road requires a centerline"
            )

        numeric = (
            self.road_width_m,
            self.vehicle_length_m,
            self.vehicle_width_m,
            self.cone_radius_m,
            self.collision_margin_m,
            self.clearance_margin_m,
        )

        if (
            not all(
                math.isfinite(value)
                for value in numeric
            )
            or any(
                value <= 0.0
                for value in numeric
            )
        ):
            raise ValueError(
                "Invalid Phase-1 physical dimensions"
            )

        if self.forward_only is not True:
            raise ValueError(
                "Phase 1 must remain forward-only"
            )

    @property
    def required_gate_opening_m(self) -> float:
        """Minimum cone-surface opening accepted by the Phase-1 contract."""

        return (
            self.vehicle_width_m
            + 2.0 * self.clearance_margin_m
        )

    def target_module_count(
        self,
        module_xy_m: Mapping[
            str,
            tuple[float, float],
        ],
    ) -> int:
        return sum(
            1
            for point in module_xy_m.values()
            if self.target_platform.contains_xy(
                point
            )
        )

    def success(
        self,
        module_xy_m: Mapping[
            str,
            tuple[float, float],
        ],
    ) -> bool:
        """Success is exactly 8/8 module centres on target."""

        return (
            len(module_xy_m) == 8
            and self.target_module_count(
                module_xy_m
            )
            == 8
        )

    def to_observation(self) -> dict[str, Any]:
        """Public physical observation without generator-only reference path."""

        return {
            "frame_id": "world",
            "course_profile":
                "marl_phase1_rc_car8",
            "scenario": {
                "generator":
                    "rc_car_phase1_v1",
                "seed":
                    self.seed,
                "forward_only":
                    True,
                "road_width_m":
                    self.road_width_m,
                "vehicle_footprint": {
                    "length_m":
                        self.vehicle_length_m,
                    "width_m":
                        self.vehicle_width_m,
                    "collision_margin_m":
                        self.collision_margin_m,
                },
            },
            "start_platform": {
                "center_xy_m":
                    list(
                        self.start_platform.center_xy_m
                    ),
                "yaw_rad":
                    self.start_platform.yaw_rad,
                "size_xy_m":
                    list(
                        self.start_platform.size_xy_m
                    ),
            },
            "goal": {
                "position_xyz_m": [
                    self.target_platform.center_xy_m[0],
                    self.target_platform.center_xy_m[1],
                    0.0,
                ],
                "yaw_rad":
                    self.target_platform.yaw_rad,
                "platform_size_xy_m":
                    list(
                        self.target_platform.size_xy_m
                    ),
            },
            "navigation_cones": [
                {
                    "center_xyz_m": [
                        cone.center_xy_m[0],
                        cone.center_xy_m[1],
                        0.5 * cone.height_m,
                    ],
                    "radius_m":
                        cone.radius_m,
                    "height_m":
                        cone.height_m,
                    "static":
                        True,
                }
                for cone in self.cones
            ],
        }


def _road_support_boxes(
    centerline_xy_m: tuple[
        tuple[float, float],
        ...
    ],
    *,
    road_width_m: float,
) -> tuple[CourseBox, ...]:
    """Approximate a curved road by overlapping yawed support boxes."""

    boxes: list[CourseBox] = []

    thickness_m = 0.02
    overlap_m = 0.03

    for index, (
        start,
        end,
    ) in enumerate(
        zip(
            centerline_xy_m[:-1],
            centerline_xy_m[1:],
        )
    ):
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        length_m = math.hypot(
            dx,
            dy,
        )

        if length_m <= 1.0e-9:
            continue

        boxes.append(
            CourseBox(
                name=(
                    f"Phase1Road"
                    f"{index:02d}"
                ),
                center_xyz_m=(
                    0.5
                    * (
                        start[0]
                        + end[0]
                    ),
                    0.5
                    * (
                        start[1]
                        + end[1]
                    ),
                    -0.5 * thickness_m,
                ),
                size_xyz_m=(
                    length_m
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
                yaw_deg=math.degrees(
                    math.atan2(
                        dy,
                        dx,
                    )
                ),
            )
        )

    if not boxes:
        raise ValueError(
            "Phase-1 road contains no valid segment"
        )

    return tuple(boxes)


def phase1_rc_car_training_course(
    seed: int,
    *,
    clearance_margin_m: float = 0.050,
    platform_size_xy_m: tuple[
        float,
        float,
    ] = (
        1.20,
        1.20,
    ),
) -> Phase1TrainingCourse:
    """Build one seeded Phase-1 course from validated RC planar geometry."""

    if (
        not math.isfinite(
            clearance_margin_m
        )
        or clearance_margin_m <= 0.0
    ):
        raise ValueError(
            "Phase-1 clearance margin must be positive"
        )

    layout = (
        rc_car_planar_obstacle_layout(
            int(seed)
        )
    )

    centerline = tuple(
        (
            float(point[0]),
            float(point[1]),
        )
        for point in layout[
            "centerline_xy_m"
        ]
    )

    footprint = layout[
        "vehicle_footprint"
    ]

    vehicle_length_m = float(
        footprint["length_m"]
    )
    vehicle_width_m = float(
        footprint["width_m"]
    )

    cone_radius_m = float(
        layout["cone_radius_m"]
    )

    cone_height_m = float(
        layout["cone_height_m"]
    )

    collision_margin_m = float(
        footprint[
            "collision_margin_m"
        ]
    )

    road_width_m = float(
        layout["corridor_width_m"]
    )

    if (
        road_width_m
        <= vehicle_width_m
        + 2.0
        * clearance_margin_m
    ):
        raise ValueError(
            "Road is too narrow for the "
            "Phase-1 RC-Car8 clearance contract"
        )

    goal = layout["goal_xyyaw"]

    target = OrientedPlatform(
        center_xy_m=(
            float(goal[0]),
            float(goal[1]),
        ),
        yaw_rad=float(goal[2]),
        size_xy_m=(
            float(
                platform_size_xy_m[0]
            ),
            float(
                platform_size_xy_m[1]
            ),
        ),
    )

    start = OrientedPlatform(
        center_xy_m=(
            0.0,
            0.0,
        ),
        yaw_rad=0.0,
        size_xy_m=(
            float(
                platform_size_xy_m[0]
            ),
            float(
                platform_size_xy_m[1]
            ),
        ),
    )

    thickness_m = 0.02

    start_box = CourseBox(
        name="Phase1StartPlatform",
        center_xyz_m=(
            start.center_xy_m[0],
            start.center_xy_m[1],
            -0.5 * thickness_m,
        ),
        size_xyz_m=(
            start.size_xy_m[0],
            start.size_xy_m[1],
            thickness_m,
        ),
        color_rgb=(
            0.24,
            0.27,
            0.31,
        ),
        collidable=True,
        semantic="phase1_start_platform",
    )

    target_box = CourseBox(
        name="Phase1TargetPlatform",
        center_xyz_m=(
            target.center_xy_m[0],
            target.center_xy_m[1],
            -0.5 * thickness_m,
        ),
        size_xyz_m=(
            target.size_xy_m[0],
            target.size_xy_m[1],
            thickness_m,
        ),
        color_rgb=(
            0.24,
            0.31,
            0.24,
        ),
        collidable=True,
        semantic="phase1_target_platform",
        yaw_deg=math.degrees(
            target.yaw_rad
        ),
    )

    road_boxes = (
        _road_support_boxes(
            centerline,
            road_width_m=road_width_m,
        )
    )

    cones = tuple(
        Phase1Cone(
            center_xy_m=(
                float(point[0]),
                float(point[1]),
            ),
            radius_m=cone_radius_m,
            height_m=cone_height_m,
        )
        for point in layout[
            "cone_centers_xy_m"
        ]
    )

    return Phase1TrainingCourse(
        seed=int(seed),

        start_platform=start,
        target_platform=target,

        support_boxes=(
            start_box,
            *road_boxes,
            target_box,
        ),

        cones=cones,

        centerline_xy_m=centerline,
        road_width_m=road_width_m,

        vehicle_length_m=
            vehicle_length_m,
        vehicle_width_m=
            vehicle_width_m,

        cone_radius_m=
            cone_radius_m,
        collision_margin_m=
            collision_margin_m,
        clearance_margin_m=
            clearance_margin_m,

        has_curve=bool(
            layout["has_curve"]
        ),
        track_profile=str(
            layout["track_profile"]
        ),

        forward_only=True,
    )


def _install_phase1_static_cones(
    stage,
    root_path: str,
    cones: tuple[Phase1Cone, ...],
) -> None:
    """Materialize rigid static Phase-1 navigation cones in USD.

    No RigidBodyAPI is applied: each cone is therefore a static collider.
    """

    from pxr import (
        Gf,
        UsdGeom,
        UsdPhysics,
    )

    UsdGeom.Xform.Define(
        stage,
        root_path,
    )

    for index, specification in enumerate(
        cones,
        start=1,
    ):
        path = (
            f"{root_path}/"
            f"Cone{index:02d}"
        )

        cone = UsdGeom.Cone.Define(
            stage,
            path,
        )

        cone.CreateHeightAttr(
            float(
                specification.height_m
            )
        )

        cone.CreateRadiusAttr(
            float(
                specification.radius_m
            )
        )

        cone.CreateDisplayColorAttr(
            [
                Gf.Vec3f(
                    0.90,
                    0.32,
                    0.08,
                )
            ]
        )

        x_m, y_m = (
            specification.center_xy_m
        )

        UsdGeom.XformCommonAPI(
            cone
        ).SetTranslate(
            Gf.Vec3d(
                float(x_m),
                float(y_m),
                0.5
                * float(
                    specification.height_m
                ),
            )
        )

        # Collision only. Absence of RigidBodyAPI keeps it static.
        UsdPhysics.CollisionAPI.Apply(
            cone.GetPrim()
        )


def install_phase1_training_course(
    stage,
    *,
    seed: int,
    box_installer=None,
    cone_installer=None,
) -> Phase1TrainingCourse:
    """Build and materialize one seeded Phase-1 RC-Car8 course.

    The pure generator remains the single source of truth for geometry,
    reward and the public course observation.
    """

    course = (
        phase1_rc_car_training_course(
            int(seed)
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
        cone_installer = (
            _install_phase1_static_cones
        )

    box_installer(
        stage,
        "/World/Phase1TrainingCourse/Supports",
        course.support_boxes,
    )

    cone_installer(
        stage,
        "/World/Phase1TrainingCourse/NavigationCones",
        course.cones,
    )

    return course
