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


def sample_uniform_stair_spec(seed: int) -> UniformStairSpec:
    """Sample the initial, deliberately conservative robustness envelope."""

    generator = random.Random(seed)
    return UniformStairSpec(
        rise_m=round(generator.uniform(0.050, 0.065), 3),
        tread_depth_m=round(generator.uniform(0.250, 0.320), 3),
        step_count=generator.randint(2, 4),
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


def sample_coplanar_gap_spec(seed: int) -> CoplanarGapSpec:
    """Sample the initial conservative Snake8 gap robustness envelope."""

    generator = random.Random(seed)
    return CoplanarGapSpec(
        width_m=round(generator.uniform(0.160, 0.210), 3),
        near_edge_x_m=round(generator.uniform(0.520, 0.620), 3),
        seed=seed,
    )


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
        """Serialize only the landmarks relevant to stair testing."""
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
                (-1.325, 0.0, -0.065),
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
