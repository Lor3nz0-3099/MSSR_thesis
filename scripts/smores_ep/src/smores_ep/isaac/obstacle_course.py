"""Manual task-achievement course for assembled SMORES-EP morphologies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
                "riser_depth_m": 0.30,
            },
            "button": {
                "center_xyz_m": list(self.button_center_xyz_m),
            },
            "exit": {
                "center_xyz_m": list(self.exit_center_xyz_m),
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
                (1.40, 0.0, 0.02),
                (0.30, 1.20, 0.04),
                stair_color,
                semantic="stair",
            ),
            CourseBox(
                "Stair02",
                (1.70, 0.0, 0.04),
                (0.30, 1.20, 0.08),
                stair_color,
                semantic="stair",
            ),
            CourseBox(
                "Stair03",
                (2.00, 0.0, 0.06),
                (0.30, 1.20, 0.12),
                stair_color,
                semantic="stair",
            ),
            CourseBox(
                "UpperDeck",
                (3.00, 0.0, 0.06),
                (1.70, 1.20, 0.12),
                platform_color,
                semantic="upper_platform",
            ),
            CourseBox(
                "ButtonWall",
                (2.65, 0.53, 0.27),
                (0.18, 0.08, 0.30),
                (0.35, 0.37, 0.40),
                semantic="button_support",
            ),
            CourseBox(
                "ButtonPlunger",
                (2.65, 0.475, 0.29),
                (0.08, 0.04, 0.08),
                (0.85, 0.08, 0.06),
                semantic="button",
            ),
            CourseBox(
                "ExitLeft",
                (3.55, -0.47, 0.31),
                (0.05, 0.05, 0.38),
                (0.12, 0.82, 0.25),
                collidable=False,
                semantic="exit_marker",
            ),
            CourseBox(
                "ExitRight",
                (3.55, 0.47, 0.31),
                (0.05, 0.05, 0.38),
                (0.12, 0.82, 0.25),
                collidable=False,
                semantic="exit_marker",
            ),
            CourseBox(
                "ExitTop",
                (3.55, 0.0, 0.50),
                (0.05, 0.99, 0.05),
                (0.12, 0.82, 0.25),
                collidable=False,
                semantic="exit_marker",
            ),
        ),
        gap_interval_x_m=(0.65, 0.85),
        stair_top_heights_m=(0.04, 0.08, 0.12),
        button_center_xyz_m=(2.65, 0.455, 0.29),
        exit_center_xyz_m=(3.55, 0.0, 0.31),
    )


def install_manual_obstacle_course(stage: Any) -> ManualObstacleCourse:
    """Replace the infinite floor with the manual, segmented course."""

    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    course = manual_obstacle_course()
    if stage.GetPrimAtPath("/World/Ground"):
        stage.RemovePrim("/World/Ground")
    root = UsdGeom.Xform.Define(stage, "/World/ManualObstacleCourse")
    material = UsdShade.Material(
        stage.GetPrimAtPath("/World/materials/dynamic_ground")
    )
    for element in course.boxes:
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
    return course
