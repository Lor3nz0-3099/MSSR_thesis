from __future__ import annotations

from dataclasses import dataclass
import math
from typing import ClassVar


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class SmoresGeometry:
    """Geometry measured from the imported CAD and hardware references.

    Source points use the imported CAD frame. Runtime points use the ROS body
    frame: +X forward, +Y left, +Z up.
    """

    # Mean of the two wheel axes in the imported visual. This is the module
    # frame origin and the tilt axis.
    source_body_origin_m: Vector3 = (
        -0.055453,
        -0.073320,
        -0.032324,
    )
    source_left_wheel_center_m: Vector3 = (
        -0.090665,
        -0.073348,
        -0.031969,
    )
    source_right_wheel_center_m: Vector3 = (
        -0.020255,
        -0.073293,
        -0.032678,
    )
    source_pan_center_m: Vector3 = (
        -0.055415,
        -0.039699,
        -0.029454,
    )
    source_tilt_carrier_center_m: Vector3 = (
        -0.055434,
        -0.048000,
        -0.030282,
    )
    source_inner_left_gear_center_m: Vector3 = (
        -0.082810,
        -0.073319,
        -0.032063,
    )
    source_inner_right_gear_center_m: Vector3 = (
        -0.028096,
        -0.073319,
        -0.032571,
    )
    source_outer_left_pinion_center_m: Vector3 = (
        -0.085197,
        -0.095427,
        -0.010052,
    )
    source_outer_right_pinion_center_m: Vector3 = (
        -0.025708,
        -0.095418,
        -0.054549,
    )
    source_inner_left_pinion_center_m: Vector3 = (
        -0.083326,
        -0.095220,
        -0.054105,
    )
    source_inner_right_pinion_center_m: Vector3 = (
        -0.027580,
        -0.095226,
        -0.010632,
    )

    # Both driving wheels reference the same CAD prototype. Its raw mesh is
    # exactly 62 mm in diameter; 31.06 mm includes the sub-0.1 mm tessellation
    # and assembly-orientation envelope measured from the transformed points.
    wheel_radius_m: float = 0.03150
    wheel_width_m: float = 0.0165
    pan_face_radius_m: float = 0.03140
    pan_face_thickness_m: float = 0.02030
    # Exact transformed-vertex envelope of the imported TOP wheel. These are
    # deliberately not derived from its rotated axis-aligned bounding box.
    pan_visual_radius_m: float = 0.03140
    pan_visual_thickness_m: float = 0.02030
    # Exact outer plane of tn__base_chassis1_nP in the body frame. The CAD
    # source-Y extent is -107.319 mm and the body origin is -73.320 mm.
    bottom_face_x_m: float = -0.033999
    visual_ground_preload_m: float = 0.0008
    spur_to_pinion_ratio: float = 48.0 / 9.0
    module_mass_kg: float = 0.454
    tilt_min_rad: float = -math.pi / 2.0
    tilt_max_rad: float = math.pi / 2.0

    visual_root_path: ClassVar[str] = "/World/SMORES_EP_modulev1"
    assembly_path: ClassVar[str] = "tn__SMORESEP_dC"

    fixed_parts: ClassVar[tuple[str, ...]] = (
        "tn__base_chassis1_nP",
        "tn__chassis_part_black1_Vb0",
        "tn__chassis_part_black2_Vb0",
        "tn__motor1_dC",
        "tn__motor2_dC",
        "tn__motor3_dC",
        "tn__motor4_dC",
        "tn__chassis_up_paintpot21_ue0",
        "tn__chassis_up_paintpot22_ue0",
    )
    left_wheel_parts: ClassVar[tuple[str, ...]] = (
        "tn__smores_wheel1_nP",
        "tn__spur_gear_48T4_kR",
    )
    right_wheel_parts: ClassVar[tuple[str, ...]] = (
        "tn__smores_wheel2_nP",
        "tn__spur_gear_48T2_kR",
    )
    outer_left_pinion_parts: ClassVar[tuple[str, ...]] = (
        "tn__pinon_9T2_zH",
    )
    outer_right_pinion_parts: ClassVar[tuple[str, ...]] = (
        "tn__pinon_9T4_zH",
    )
    inner_left_gear_parts: ClassVar[tuple[str, ...]] = (
        "tn__spur_gear_48T1_kR",
    )
    inner_right_gear_parts: ClassVar[tuple[str, ...]] = (
        "tn__spur_gear_48T3_kR",
    )
    inner_left_pinion_parts: ClassVar[tuple[str, ...]] = (
        "tn__pinon_9T1_zH",
    )
    inner_right_pinion_parts: ClassVar[tuple[str, ...]] = (
        "tn__pinon_9T3_zH",
    )
    tilt_parts: ClassVar[tuple[str, ...]] = ("tn__chassis_up21_qN",)
    pan_parts: ClassVar[tuple[str, ...]] = (
        "tn__crown_gear1_tL",
        "tn__smores_wheel3_nP",
    )

    def __post_init__(self) -> None:
        if (
            self.wheel_radius_m <= 0.0
            or self.wheel_width_m <= 0.0
            or self.pan_face_radius_m <= 0.0
            or self.pan_face_thickness_m <= 0.0
            or self.pan_visual_radius_m <= 0.0
            or self.pan_visual_thickness_m <= 0.0
            or self.bottom_face_x_m >= 0.0
        ):
            raise ValueError("Wheel dimensions must be positive")
        if self.visual_ground_preload_m < 0.0:
            raise ValueError("Visual ground preload cannot be negative")
        if self.spur_to_pinion_ratio <= 0.0:
            raise ValueError("Gear ratio must be positive")
        if self.module_mass_kg <= 0.0:
            raise ValueError("Module mass must be positive")
        if self.tilt_min_rad >= self.tilt_max_rad:
            raise ValueError("Tilt limits are inverted")

    @staticmethod
    def source_vector_to_body(vector: Vector3) -> Vector3:
        """Rotate a CAD-frame vector into the ROS body frame."""

        x, y, z = vector
        return (y, -x, z)

    def source_point_to_body(self, point: Vector3) -> Vector3:
        """Translate and rotate a CAD point into the ROS body frame."""

        delta = tuple(
            coordinate - origin
            for coordinate, origin in zip(point, self.source_body_origin_m)
        )
        return self.source_vector_to_body(delta)  # type: ignore[arg-type]

    def _measured_wheel_centers_body_m(
        self,
    ) -> tuple[Vector3, Vector3]:
        """Raw wheel-axis centres measured from the imported CAD."""
        return (
            self.source_point_to_body(self.source_left_wheel_center_m),
            self.source_point_to_body(self.source_right_wheel_center_m),
        )

    @property
    def left_wheel_center_body_m(self) -> Vector3:
        """Symmetric physical left-wheel centre.

        The imported CAD places the two measured wheel axes at slightly
        different X/Z coordinates (about 0.71 mm in Z).  Using those raw
        values as articulation anchors systematically preloads one wheel row
        and unloads the other on a flat floor.

        Preserve the measured track width, but project both physical wheel
        centres onto their common mean axle.
        """
        left, right = self._measured_wheel_centers_body_m()

        axle_x = 0.5 * (left[0] + right[0])
        axle_z = 0.5 * (left[2] + right[2])
        half_track = 0.5 * abs(left[1] - right[1])

        return (axle_x, half_track, axle_z)

    @property
    def right_wheel_center_body_m(self) -> Vector3:
        """Symmetric physical right-wheel centre."""
        left, right = self._measured_wheel_centers_body_m()

        axle_x = 0.5 * (left[0] + right[0])
        axle_z = 0.5 * (left[2] + right[2])
        half_track = 0.5 * abs(left[1] - right[1])

        return (axle_x, -half_track, axle_z)

    @property
    def pan_center_body_m(self) -> Vector3:
        return self.source_point_to_body(self.source_pan_center_m)

    @property
    def top_face_x_m(self) -> float:
        """Outer TOP docking plane at zero tilt in the body frame."""

        return self.pan_center_body_m[0] + 0.5 * self.pan_face_thickness_m

    @property
    def top_to_bottom_spacing_m(self) -> float:
        """Root spacing for tangent TOP-to-BOTTOM modules at zero tilt."""

        return self.top_face_x_m - self.bottom_face_x_m

    @property
    def tilt_carrier_center_body_m(self) -> Vector3:
        return self.source_point_to_body(
            self.source_tilt_carrier_center_m
        )

    @property
    def mechanism_parts(self) -> tuple[str, ...]:
        return (
            self.outer_left_pinion_parts
            + self.outer_right_pinion_parts
            + self.inner_left_gear_parts
            + self.inner_right_gear_parts
            + self.inner_left_pinion_parts
            + self.inner_right_pinion_parts
        )

    def ground_contact_height_m(self, tilt_rad: float) -> float:
        """Keep wheels or the tilted TOP face slightly in the ground.

        A negative user tilt lowers the TOP face. Once it reaches the ground,
        the reduced-order model raises the module instead of allowing the face
        to pass through the plane, matching the support transition described
        in the SMORES paper.
        """

        if not math.isfinite(tilt_rad):
            raise ValueError("Tilt angle must be finite")
        wheel_min_z = min(
            self.left_wheel_center_body_m[2],
            self.right_wheel_center_body_m[2],
        ) - self.wheel_radius_m

        # The displayed tilt rotation is -tilt_rad around body +Y.
        displayed_angle = -tilt_rad
        pan_x, _, pan_z = self.pan_center_body_m
        pan_center_z = (
            -math.sin(displayed_angle) * pan_x
            + math.cos(displayed_angle) * pan_z
        )
        pan_half_extent_z = (
            abs(math.sin(displayed_angle))
            * 0.5
            * self.pan_visual_thickness_m
            + abs(math.cos(displayed_angle)) * self.pan_visual_radius_m
        )
        pan_min_z = pan_center_z - pan_half_extent_z
        lowest_local_z = min(wheel_min_z, pan_min_z)
        return max(0.0, -lowest_local_z - self.visual_ground_preload_m)

    @property
    def track_width_m(self) -> float:
        return abs(
            self.left_wheel_center_body_m[1]
            - self.right_wheel_center_body_m[1]
        )

    @property
    def all_parts(self) -> tuple[str, ...]:
        return (
            self.fixed_parts
            + self.left_wheel_parts
            + self.right_wheel_parts
            + self.mechanism_parts
            + self.tilt_parts
            + self.pan_parts
        )
