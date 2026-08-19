from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

from freebot_docking.diagnostics.contacts import ContactSnapshot
from freebot_docking.physics.external_magnet import ExternalMagneticInteraction
from freebot_docking.physics.state import MagnetState
from freebot_docking.scenarios.two_module_docking import (
    InternalMagneticPreloadInteraction,
)


_EPSILON = 1.0e-12


def arrow_segments(
    start_world: np.ndarray,
    vector_world: np.ndarray,
    scale: float = 1.0,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return a shaft and two arrowhead segments in world coordinates."""

    start = np.asarray(start_world, dtype=np.float64)
    vector = float(scale) * np.asarray(vector_world, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if start.shape != (3,) or vector.shape != (3,):
        raise ValueError("Debug arrows require two three-dimensional vectors")
    if not np.all(np.isfinite(start)) or not np.all(np.isfinite(vector)):
        raise ValueError("Debug arrow coordinates must be finite")
    if length <= _EPSILON:
        return ()

    direction = vector / length
    reference = (
        np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(direction[2]) < 0.9
        else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    side = np.cross(direction, reference)
    side /= np.linalg.norm(side)
    end = start + vector
    head_length = min(0.008, 0.30 * length)
    head_half_width = 0.45 * head_length
    head_base = end - head_length * direction
    first_corner = head_base + head_half_width * side
    second_corner = head_base - head_half_width * side
    return (
        (start, end),
        (end, first_corner),
        (end, second_corner),
    )


class IsaacForceDebugDraw:
    """Display-only force vectors; the authored prims have no physics APIs."""

    def __init__(
        self,
        stage: Any,
        enabled: bool,
        force_scale_m_per_n: float,
    ) -> None:
        self.enabled = bool(enabled)
        self.force_scale = float(force_scale_m_per_n)
        self._curves = None
        self._curve_points = None
        self._curve_counts = None
        self._curve_colors = None
        self._points = None
        self._point_positions = None
        self._point_colors = None
        if not np.isfinite(self.force_scale) or self.force_scale <= 0.0:
            raise ValueError("Debug force scale must be finite and positive")
        if not self.enabled:
            return

        from pxr import UsdGeom, Vt

        root = "/World/freebot_force_debug"
        self._curves = UsdGeom.BasisCurves.Define(stage, f"{root}/vectors")
        self._curves.CreateTypeAttr(UsdGeom.Tokens.linear)
        self._curves.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        self._curve_points = self._curves.CreatePointsAttr()
        self._curve_counts = self._curves.CreateCurveVertexCountsAttr()
        self._curve_colors = self._curves.CreateDisplayColorPrimvar(
            UsdGeom.Tokens.uniform
        )
        self._curves.CreateWidthsAttr().Set(Vt.FloatArray([0.0012]))
        self._curves.SetWidthsInterpolation(UsdGeom.Tokens.constant)

        self._points = UsdGeom.Points.Define(stage, f"{root}/points")
        self._point_positions = self._points.CreatePointsAttr()
        self._point_colors = self._points.CreateDisplayColorPrimvar(
            UsdGeom.Tokens.vertex
        )
        self._points.CreateWidthsAttr().Set(Vt.FloatArray([0.006]))
        self._points.SetWidthsInterpolation(UsdGeom.Tokens.constant)

    def update(
        self,
        *,
        active_magnet: MagnetState,
        passive_magnet: MagnetState,
        external: ExternalMagneticInteraction,
        external_force_target: Literal["active-shell", "active-carrier"],
        active_internal: InternalMagneticPreloadInteraction,
        passive_internal: InternalMagneticPreloadInteraction,
        contacts: Mapping[str, ContactSnapshot],
    ) -> None:
        if not self.enabled or self._curves is None or self._points is None:
            return

        from pxr import Gf, Vt

        yellow = Gf.Vec3f(1.0, 0.9, 0.1)
        white = Gf.Vec3f(1.0, 1.0, 1.0)
        green = Gf.Vec3f(0.1, 1.0, 0.2)
        orange = Gf.Vec3f(1.0, 0.35, 0.05)
        blue = Gf.Vec3f(0.1, 0.35, 1.0)
        cyan = Gf.Vec3f(0.1, 0.9, 1.0)
        red = Gf.Vec3f(1.0, 0.1, 0.1)
        magenta = Gf.Vec3f(1.0, 0.1, 1.0)

        segments: list[tuple[np.ndarray, np.ndarray]] = []
        colors: list[Any] = []
        points: list[np.ndarray] = []
        point_colors: list[Any] = []

        def add_segment(
            first: np.ndarray,
            second: np.ndarray,
            color: Any,
        ) -> None:
            segments.append((first, second))
            colors.append(color)

        def add_arrow(
            start: np.ndarray,
            vector: np.ndarray,
            scale: float,
            color: Any,
        ) -> None:
            for first, second in arrow_segments(start, vector, scale):
                add_segment(first, second, color)

        for magnet in (active_magnet, passive_magnet):
            add_arrow(
                magnet.center_world,
                magnet.axis_world,
                0.035,
                yellow,
            )

        active_external_point = (
            external.active_surface_point_world
            if external_force_target == "active-shell"
            else external.interaction_point_world
        )
        if external.line_of_action_valid:
            add_segment(
                active_external_point,
                external.passive_surface_point_world,
                white,
            )
        add_arrow(
            active_external_point,
            external.force_on_active_world,
            self.force_scale,
            green,
        )
        add_arrow(
            external.passive_surface_point_world,
            external.force_on_passive_world,
            self.force_scale,
            orange,
        )
        points.extend(
            (
                external.interaction_point_world,
                active_external_point,
                external.passive_surface_point_world,
            )
        )
        point_colors.extend((yellow, green, orange))

        for interaction in (active_internal, passive_internal):
            add_arrow(
                interaction.interaction_point_world,
                interaction.force_on_carrier_world,
                self.force_scale,
                blue,
            )
            add_arrow(
                interaction.interaction_point_world,
                interaction.force_on_shell_world,
                self.force_scale,
                cyan,
            )

        for reading in contacts.values():
            if reading.error is not None:
                continue
            if np.linalg.norm(reading.normal_force_world) > _EPSILON:
                add_arrow(
                    reading.normal_application_point_world,
                    reading.normal_force_world,
                    self.force_scale,
                    red,
                )
            if np.linalg.norm(reading.friction_force_world) > _EPSILON:
                add_arrow(
                    reading.friction_application_point_world,
                    reading.friction_force_world,
                    self.force_scale,
                    magenta,
                )

        curve_points = [
            Gf.Vec3f(*(float(value) for value in point))
            for segment in segments
            for point in segment
        ]
        self._curve_counts.Set(Vt.IntArray([2] * len(segments)))
        self._curve_points.Set(Vt.Vec3fArray(curve_points))
        self._curve_colors.Set(Vt.Vec3fArray(colors))
        self._point_positions.Set(
            Vt.Vec3fArray(
                [
                    Gf.Vec3f(*(float(value) for value in point))
                    for point in points
                ]
            )
        )
        self._point_colors.Set(Vt.Vec3fArray(point_colors))
