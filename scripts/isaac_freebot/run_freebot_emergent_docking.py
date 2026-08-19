"""Minimal, force-driven FreeBOT docking experiment for Isaac Sim.

This program intentionally does not import or reuse the legacy two-module
runner.  Docking is not a state and is never imposed: PhysX receives only
magnetic action/reaction pairs, wheel commands, gravity and contact forces.

Magnetic force model
--------------------
The own-shell interaction is a reduced magnetostatic model: an equivalent
point dipole plus Maxwell-pressure integration over a spherical inner patch.
The external interaction uses the digitized FreeBOT Figs. 4--6 curves.
These two interactions use independent geometric distances and are evaluated
at every physics step.
No force cap, capture gate, latch, preload, artificial
tangential force or magnetic damping is used.  The force-law interface can be
replaced by monotone experimental force-gap data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
import time
from typing import Any, Iterable

import numpy as np


EPS = 1.0e-12

ANSI_RESET = "\033[0m"
ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_CYAN = "\033[96m"
ANSI_MAGENTA = "\033[95m"


def colored(text: str, color: str, enabled: bool) -> str:
    return f"{color}{text}{ANSI_RESET}" if enabled else text


def vector3(value: Iterable[float]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got shape {result.shape}")
    return result


def normalized(value: Iterable[float]) -> np.ndarray:
    value = vector3(value)
    norm = float(np.linalg.norm(value))
    if norm <= EPS:
        raise ValueError("Cannot normalize a zero vector")
    return value / norm


def first_vector(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value[0], dtype=np.float64)


def first_quaternion(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value[0], dtype=np.float64)
    if result.shape != (4,):
        raise ValueError(f"Expected an Isaac quaternion with shape (4,), got {result.shape}")
    return result


def quaternion_rotate_wxyz(quaternion: Iterable[float], vector: Iterable[float]) -> np.ndarray:
    """Rotate a vector by an Isaac Sim scalar-first quaternion (w, x, y, z)."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Expected a quaternion, got shape {q.shape}")
    q_norm = float(np.linalg.norm(q))
    if q_norm <= EPS:
        raise ValueError("Cannot use a zero quaternion")
    q = q / q_norm
    w = q[0]
    q_vector = q[1:]
    v = vector3(vector)
    return v + 2.0 * np.cross(q_vector, np.cross(q_vector, v) + w * v)


def quaternion_y_degrees(angle_degrees: float) -> np.ndarray:
    half_angle = 0.5 * np.deg2rad(float(angle_degrees))
    return np.array([np.cos(half_angle), 0.0, np.sin(half_angle), 0.0], dtype=np.float64)


@dataclass(frozen=True)
class MagnetState:
    center_world: np.ndarray
    orientation_wxyz: np.ndarray
    axis_world: np.ndarray
    face_center_world: np.ndarray
    half_length: float
    radius: float


@dataclass(frozen=True)
class ShellState:
    center_world: np.ndarray
    com_world: np.ndarray
    inner_radius: float
    outer_radius: float


@dataclass(frozen=True)
class SurfaceHit:
    point_world: np.ndarray
    method: str


@dataclass(frozen=True)
class MagneticCandidate:
    surface_point_world: np.ndarray
    surface_normal_world: np.ndarray
    gap: float
    force_law_gap: float
    alignment_cosine: float
    raw_coupling: float
    is_system_calibrated: bool
    surface_method: str


@dataclass(frozen=True)
class MagneticInteraction:
    force_on_magnet_world: np.ndarray
    force_on_shell_world: np.ndarray
    magnet_application_point_world: np.ndarray
    shell_application_point_world: np.ndarray
    gap: float
    force_law_gap: float
    surface_normal_world: np.ndarray
    alignment_cosine: float
    raw_coupling: float
    normalized_weight: float
    force_magnitude: float
    surface_method: str
    # Components in the paper's local connection frame.  They are zero for
    # the internal magnet-to-own-shell interaction, which is not covered by
    # FreeBOT Figs. 4--6.
    parallel_force_n: float = 0.0
    perpendicular_force_n: float = 0.0
    paper_theta_deg: float = 0.0
    branch_weight: float = 1.0


@dataclass(frozen=True)
class MagnetConfig:
    remanence_t: float = 1.47
    radius_m: float = 0.010
    length_m: float = 0.010
    local_active_axis: tuple[float, float, float] = (0.0, 0.0, -1.0)
    # Used only by the legacy analytic helper retained for comparison tests;
    # it is deliberately not used by either the patch model or Figs. 5--6.
    alignment_power: float = 2.0
    minimum_field_distance_m: float = 0.003
    internal_patch_half_angle_deg: float = 35.0
    internal_patch_rings: int = 5
    internal_patch_samples_per_ring: int = 16
    internal_pressure_scale: float = 1.0
    # Numerical near-field guard for the point-dipole approximation.  At the
    # nominal ~10 mm face gap it is inactive; below a few millimetres the
    # point dipole otherwise predicts nonphysical multi-tesla fields and
    # >100 N loads.  This is not a fitted climbing force.
    maximum_sample_pressure_pa: float = 1.0e5
    axial_dipole_count: int = 7
    axial_dipole_span_fraction: float = 0.80


@dataclass(frozen=True)
class GeometryConfig:
    # Least-squares fit over all 4356 vertices of both CAD hemispheres.  The
    # vertices lie on two concentric surfaces at 61.3472 and 63.3472 mm.
    shell_outer_radius_m: float = 0.0633472
    shell_inner_radius_m: float = 0.0613472
    # The shell_link origin is not the sphere centre.  This measured local
    # offset is essential for contact clearances, magnetic gaps and theta.
    shell_center_from_body_origin_m: tuple[float, float, float] = (
        0.00155433,
        0.00087740,
        0.00466804,
    )
    # Fits over transformed vertices of the CAD actually referenced by the
    # active USD; no axis-aligned bounding-box dimensions are used here.
    tire_outer_radius_m: float = 0.016000287
    tire_half_width_m: float = 0.003001301
    # The tire instance centre is not the wheel_link origin.  Its signed local
    # Y offset is outward on each side.  The tire axis is tilted 0.167 degrees
    # from local Y in the CAD assembly.
    tire_center_axial_offset_m: float = 0.00105277
    tire_axis_tilt_deg: float = 0.16702
    # Both active-CAD caster meshes are exact 4.650 mm spheres.  The previous
    # 5.935 mm value came from an obsolete filename and was 1.285 mm too large.
    caster_ball_radius_m: float = 0.004650
    wheel_contact_tolerance_m: float = 0.001
    # The collision tessellation reaches contact while the nominal spherical
    # radii still report about 2.4 mm of separation.  Subtract this geometric
    # CAD offset before evaluating the published shell-surface gap curve.
    # This changes only the distance datum; it is not a force activation gate.
    shell_gap_contact_offset_m: float = 0.0025


@dataclass(frozen=True)
class SimulationConfig:
    usd_path: Path
    physics_hz: int = 240
    steps: int = 12_000
    log_interval: int = 240
    # The nominal sphere radius is 63.3335 mm.  The CAD collision meshes have
    # small equatorial/protruding features reaching about 66.5 mm, so the body
    # origins start at z=66.5 mm to avoid ground interpenetration.
    active_start: tuple[float, float, float] = (0.0199584, 0.060, 0.0665)
    # The digitized FEM curve has reached zero at 30 mm. The extra 10 mm lets
    # both fully dynamic articulations settle without entering its tail, so
    # approach remains entirely under teleoperation.
    target_start: tuple[float, float, float] = (0.1866528, 0.060, 0.0665)
    # Both complete CAD articulations must start upright. Rotating the whole
    # module to point its magnet sideways destroys its static equilibrium.
    active_y_rotation_deg: float = 0.0
    target_y_rotation_deg: float = 0.0
    platform_center: tuple[float, float, float] = (0.56, 0.060, 0.044)
    platform_size: tuple[float, float, float] = (0.62, 0.42, 0.088)
    left_wheel_velocity_deg_s: float = 0.0
    right_wheel_velocity_deg_s: float = 0.0
    rolling_test: bool = False
    internal_climb_test: bool = False
    rolling_wheel_velocity_deg_s: float = 360.0
    wheel_damping: float = 500.0
    # PhysX angular-drive maxForce is expected to be torque (N m) for a
    # revolute joint; verify this against the installed Isaac Sim build.
    wheel_max_force: float = 0.686
    external_force_model: str = "exponential"
    external_exponential_distance_scale_m: float = 0.0035
    external_exponential_angle_scale_deg: float = 45.0
    external_exponential_angle_power: float = 2.0
    color_logs: bool = True
    ros2_teleop: bool = False
    cmd_vel_topic: str = "/cmd_vel"
    cmd_timeout_s: float = 1.0
    cmd_linear_scale: float = 900.0
    cmd_angular_scale: float = 360.0
    cmd_linear_sign: float = 1.0
    cmd_angular_sign: float = 1.0
    debug_draw: bool = False
    debug_force_scale_m_per_n: float = 0.005
    contact_debug: bool = False
    caster_colliders_enabled: bool = True
    # Used only to evaluate the paper's static feasibility equations in the
    # diagnostics.  PhysX still resolves the actual normals and friction.
    module_mass_kg: float = 0.360
    wheel_shell_static_friction: float = 2.20
    wheel_shell_dynamic_friction: float = 1.90
    caster_shell_static_friction: float = 0.03
    caster_shell_dynamic_friction: float = 0.02
    shell_shell_static_friction: float = 1.10
    shell_shell_dynamic_friction: float = 1.00
    ground_static_friction: float = 1.50
    ground_dynamic_friction: float = 1.25
    magnet: MagnetConfig = field(default_factory=MagnetConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)


class MagneticGeometry:
    """Pure geometry; replace this class later with PhysX/FEM surface queries."""

    @staticmethod
    def ray_sphere_intersection(
        ray_origin: Iterable[float],
        ray_direction: Iterable[float],
        sphere_center: Iterable[float],
        sphere_radius: float,
    ) -> np.ndarray | None:
        origin = vector3(ray_origin)
        direction = normalized(ray_direction)
        center = vector3(sphere_center)
        offset = origin - center
        b = float(np.dot(offset, direction))
        c = float(np.dot(offset, offset) - sphere_radius * sphere_radius)
        discriminant = b * b - c
        if discriminant < 0.0:
            return None
        root = float(np.sqrt(max(discriminant, 0.0)))
        positive = [distance for distance in (-b - root, -b + root) if distance > EPS]
        if not positive:
            return None
        return origin + min(positive) * direction

    @staticmethod
    def ray_sphere_diagnostics(
        ray_origin: Iterable[float],
        ray_direction: Iterable[float],
        sphere_center: Iterable[float],
    ) -> tuple[float, float]:
        """Return forward projection and axis-to-centre miss distance.

        A positive projection means the sphere centre lies in front of the
        selected magnet face.  The infinite axis intersects a sphere exactly
        when ``miss_distance <= sphere_radius``; the forward ray additionally
        requires a suitable positive intersection distance.
        """
        origin = vector3(ray_origin)
        direction = normalized(ray_direction)
        center_offset = vector3(sphere_center) - origin
        forward_projection = float(np.dot(center_offset, direction))
        perpendicular = center_offset - forward_projection * direction
        return forward_projection, float(np.linalg.norm(perpendicular))

    def find_surface_point(
        self,
        face_center: Iterable[float],
        axis: Iterable[float],
        shell_center: Iterable[float],
        radius: float,
    ) -> SurfaceHit | None:
        point = self.ray_sphere_intersection(face_center, axis, shell_center, radius)
        if point is not None:
            return SurfaceHit(point_world=point, method="ray")
        return None


class CylindricalMagnetForceLaw:
    """Reduced finite-cylinder/equivalent-plane high-permeability model.

    Deliberately omitted physics: saturation, hysteresis, spherical curvature
    and spatial field integration.  Therefore its parameters require
    validation against the real magnet, steel and shell geometry.
    """

    MU0 = 4.0 * np.pi * 1.0e-7

    def __init__(self, config: MagnetConfig):
        if config.remanence_t <= 0.0 or config.radius_m <= 0.0 or config.length_m <= 0.0:
            raise ValueError("Magnet remanence and dimensions must be positive")
        if config.alignment_power <= 0.0:
            raise ValueError("alignment_power must be positive")
        self.config = config

    def axial_field_t(self, gap_m: float) -> float:
        gap = max(float(gap_m), 0.0)
        radius = self.config.radius_m
        length = self.config.length_m
        far_face = (gap + length) / np.sqrt(radius * radius + (gap + length) ** 2)
        near_face = gap / np.sqrt(radius * radius + gap * gap)
        return float(0.5 * self.config.remanence_t * (far_face - near_face))

    def force_n(self, gap_m: float, alignment_cosine: float) -> float:
        alignment = float(np.clip(alignment_cosine, 0.0, 1.0))
        field_t = self.axial_field_t(gap_m)
        pole_area = np.pi * self.config.radius_m**2
        pressure_force = field_t * field_t * pole_area / (2.0 * self.MU0)
        return float(pressure_force * alignment**self.config.alignment_power)


class TabulatedMagnetForceLaw:
    """Linear, non-overshooting interpolation for measured aligned force data."""

    def __init__(self, gap_samples_m: Iterable[float], force_samples_n: Iterable[float], alignment_power: float = 2.0):
        gaps = np.asarray(tuple(gap_samples_m), dtype=np.float64)
        forces = np.asarray(tuple(force_samples_n), dtype=np.float64)
        if gaps.ndim != 1 or forces.shape != gaps.shape or len(gaps) < 2:
            raise ValueError("Experimental gap and force arrays must have the same one-dimensional shape")
        if np.any(np.diff(gaps) <= 0.0) or np.any(forces < 0.0) or np.any(np.diff(forces) > 0.0):
            raise ValueError("Gaps must increase strictly and measured forces must be nonnegative and nonincreasing")
        self.gaps = gaps
        self.forces = forces
        self.alignment_power = float(alignment_power)

    def force_n(self, gap_m: float, alignment_cosine: float) -> float:
        aligned_force = float(np.interp(float(gap_m), self.gaps, self.forces, left=self.forces[0], right=0.0))
        alignment = float(np.clip(alignment_cosine, 0.0, 1.0))
        return aligned_force * alignment**self.alignment_power


class FreeBotFemForceLaw(TabulatedMagnetForceLaw):
    """Digitized ANSYS curve from FreeBOT (IROS 2020), Fig. 4.

    The independent variable is the distance *between spherical shells*, not
    the direct magnet-to-opposite-shell distance.  Fig. 4 and Table I report
    22.6 N at shell contact.  Intermediate samples below were digitized from
    the plotted curve; linear interpolation is monotone and cannot overshoot.

    The runtime evaluates one external source: the active magnet toward the
    passive shell.  Therefore the table contains the complete published force
    directly, including 22.6 N at contact; there is no reciprocal split.
    """

    GAP_SAMPLES_M = 1.0e-3 * np.array(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0],
        dtype=np.float64,
    )
    FORCE_SAMPLES_N = np.array(
        [22.6, 16.51, 12.42, 9.52, 7.14, 5.36, 4.13, 2.34, 1.35, 0.79, 0.36, 0.12, 0.08, 0.0],
        dtype=np.float64,
    )
    is_system_calibrated = True

    def __init__(self, alignment_power: float = 2.0):
        super().__init__(
            self.GAP_SAMPLES_M,
            self.FORCE_SAMPLES_N,
            alignment_power=alignment_power,
        )


class FreeBotAngularForceLaw:
    """Digitized angular FEM cuts from FreeBOT Figs. 5 and 6.

    ``parallel`` is along the line joining the module centres and
    ``perpendicular`` is along the projection of the active magnet axis onto
    the tangent plane of that line.  Fig. 5 is the normal magnet-to-remote-
    shell case.  Fig. 6 is the close, same-pole magnet-to-magnet case and
    therefore contains negative (repulsive) components.

    The paper publishes plots, not raw samples.  Values below were digitized
    at 5 degree intervals from the PDF.  End points explicitly stated by the
    authors (22.6 N aligned attraction, zero transverse force by symmetry)
    are used as anchors.  Linear interpolation avoids invented oscillations.
    """

    ANGLES_DEG = np.arange(0.0, 91.0, 5.0, dtype=np.float64)
    SHELL_PARALLEL_N = np.array(
        [22.6, 15.52, 11.89, 9.72, 7.64, 7.08, 6.70, 6.32, 5.85, 5.28,
         4.81, 4.34, 3.92, 3.49, 3.11, 2.64, 2.36, 2.08, 1.75],
        dtype=np.float64,
    )
    SHELL_PERPENDICULAR_N = np.array(
        [0.0, 0.154, 0.706, 1.245, 1.40, 1.227, 0.919, 0.656, 0.474,
         0.345, 0.257, 0.207, 0.169, 0.151, 0.132, 0.125, 0.119, 0.113, 0.10],
        dtype=np.float64,
    )
    CLOSE_MAGNET_PARALLEL_N = np.array(
        [-28.0, -18.79, 1.99, 16.21, 21.95, 31.25, 30.70, 27.97, 25.78,
         24.41, 23.32, 22.77, 22.23, 21.95, 21.68, 21.68, 21.41, 21.41, 22.6],
        dtype=np.float64,
    )
    CLOSE_MAGNET_PERPENDICULAR_N = np.array(
        [0.0, -14.61, -18.42, -13.50, -7.81, -3.95, -1.66, -0.66,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )

    @classmethod
    def _components(cls, angle_deg: float, parallel: np.ndarray, perpendicular: np.ndarray) -> tuple[float, float]:
        angle = float(np.clip(angle_deg, cls.ANGLES_DEG[0], cls.ANGLES_DEG[-1]))
        return (
            float(np.interp(angle, cls.ANGLES_DEG, parallel)),
            float(np.interp(angle, cls.ANGLES_DEG, perpendicular)),
        )

    def shell_components_n(self, angle_deg: float) -> tuple[float, float]:
        return self._components(angle_deg, self.SHELL_PARALLEL_N, self.SHELL_PERPENDICULAR_N)

    def close_magnet_components_n(self, angle_deg: float) -> tuple[float, float]:
        return self._components(
            angle_deg,
            self.CLOSE_MAGNET_PARALLEL_N,
            self.CLOSE_MAGNET_PERPENDICULAR_N,
        )


@dataclass(frozen=True)
class SphericalPatchSample:
    direction: np.ndarray
    area_m2: float
    ring_index: int
    polar_angle_rad: float


@dataclass(frozen=True)
class WheelSlipDiagnostic:
    wheel_surface_speed_m_s: float
    estimated_contact_relative_tangent_speed_m_s: float
    slip_speed_m_s: float
    slip_ratio: float
    clearance_m: float
    estimated_contact: bool


@dataclass(frozen=True)
class InternalPatchResult:
    """Integrated magnet/own-shell load and numerical diagnostics."""

    interactions: tuple[MagneticInteraction, ...]
    internal_gap_m: float
    total_force_on_shell_world: np.ndarray
    total_torque_on_shell_world: np.ndarray
    total_force_on_internal_world: np.ndarray
    total_torque_on_internal_world: np.ndarray
    peak_pressure_pa: float
    unclamped_peak_pressure_pa: float
    pressure_clamp_count: int
    peak_field_t: float
    minimum_dipole_sample_distance_m: float
    ring_force_norms_n: tuple[float, ...]
    ring_max_pressure_pa: tuple[float, ...]
    sampled_area_m2: float
    cap_area_m2: float
    radial_force_on_internal_n: float
    tangential_force_on_internal_n: float
    sample_points_world: tuple[np.ndarray, ...]


class InternalPatchMagneticModel:
    """Equivalent dipole + pressure integration on an inner spherical cap.

    ``internal_pressure_scale`` is an explicit calibration parameter covering
    finite permeability, saturation, curvature and the point-dipole
    approximation.  It is not relative permeability and defaults to one.
    """

    MU0 = 4.0 * np.pi * 1.0e-7

    def __init__(self, config: MagnetConfig):
        self.config = config
        if config.minimum_field_distance_m <= 0.0:
            raise ValueError("minimum_field_distance_m must be positive")
        if config.internal_patch_rings < 1 or config.internal_patch_samples_per_ring < 3:
            raise ValueError("internal patch sampling requires at least one ring and three samples")
        if not 0.0 < config.internal_patch_half_angle_deg < 90.0:
            raise ValueError("internal_patch_half_angle_deg must lie in (0, 90)")
        if config.internal_pressure_scale < 0.0 or config.maximum_sample_pressure_pa <= 0.0:
            raise ValueError("pressure scale must be nonnegative and pressure limit positive")
        if config.axial_dipole_count < 1:
            raise ValueError("axial_dipole_count must be at least one")
        if not 0.0 <= config.axial_dipole_span_fraction <= 1.0:
            raise ValueError("axial_dipole_span_fraction must lie in [0, 1]")

    @property
    def magnetic_moment_magnitude(self) -> float:
        volume = np.pi * self.config.radius_m**2 * self.config.length_m
        return float(self.config.remanence_t * volume / self.MU0)

    @staticmethod
    def _basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = normalized(np.cross(axis, helper))
        return u, normalized(np.cross(axis, u))

    def _cap_samples(self, cap_axis: np.ndarray, sphere_radius_m: float) -> tuple[SphericalPatchSample, ...]:
        u, v = self._basis(cap_axis)
        half_angle = np.deg2rad(self.config.internal_patch_half_angle_deg)
        delta_theta = half_angle / self.config.internal_patch_rings
        central_outer = 0.5 * delta_theta
        central_area = 2.0 * np.pi * sphere_radius_m**2 * (1.0 - np.cos(central_outer))
        samples = [SphericalPatchSample(cap_axis, central_area, 0, 0.0)]
        for ring in range(1, self.config.internal_patch_rings + 1):
            angle = ring * delta_theta
            theta_inner = max(0.5 * delta_theta, angle - 0.5 * delta_theta)
            theta_outer = min(half_angle, angle + 0.5 * delta_theta)
            ring_area = 2.0 * np.pi * sphere_radius_m**2 * (
                np.cos(theta_inner) - np.cos(theta_outer)
            )
            sample_area = ring_area / self.config.internal_patch_samples_per_ring
            for index in range(self.config.internal_patch_samples_per_ring):
                azimuth = 2.0 * np.pi * index / self.config.internal_patch_samples_per_ring
                direction = (
                    np.cos(angle) * cap_axis
                    + np.sin(angle) * (np.cos(azimuth) * u + np.sin(azimuth) * v)
                )
                samples.append(SphericalPatchSample(
                    normalized(direction), sample_area, ring, angle
                ))
        cap_area = 2.0 * np.pi * sphere_radius_m**2 * (1.0 - np.cos(half_angle))
        sampled_area = float(sum(sample.area_m2 for sample in samples))
        if not np.isclose(sampled_area, cap_area, rtol=1.0e-10, atol=1.0e-14):
            raise RuntimeError(
                f"Spherical-cap quadrature area mismatch: sampled={sampled_area} cap={cap_area}"
            )
        return tuple(samples)

    def axial_dipole_offsets_m(self) -> np.ndarray:
        half_span = 0.5 * self.config.length_m * self.config.axial_dipole_span_fraction
        return np.linspace(-half_span, half_span, self.config.axial_dipole_count, dtype=np.float64)

    def dipole_field_world(self, point_world: np.ndarray, magnet: MagnetState) -> np.ndarray:
        """Legacy single-dipole helper retained only for comparison."""
        displacement = vector3(point_world) - magnet.center_world
        distance = max(float(np.linalg.norm(displacement)), self.config.minimum_field_distance_m)
        if float(np.linalg.norm(displacement)) <= EPS:
            return np.zeros(3, dtype=np.float64)
        r_hat = displacement / float(np.linalg.norm(displacement))
        moment = self.magnetic_moment_magnitude * normalized(magnet.axis_world)
        field = self.MU0 / (4.0 * np.pi * distance**3) * (
            3.0 * r_hat * float(np.dot(moment, r_hat)) - moment
        )
        if not np.all(np.isfinite(field)):
            raise FloatingPointError("Non-finite dipole field")
        return field

    def distributed_dipole_field_world(self, point_world: np.ndarray, magnet: MagnetState) -> np.ndarray:
        axis = normalized(magnet.axis_world)
        per_dipole_moment = self.magnetic_moment_magnitude / self.config.axial_dipole_count
        summed_moment = per_dipole_moment * self.config.axial_dipole_count
        if not np.isclose(summed_moment, self.magnetic_moment_magnitude, rtol=1.0e-14, atol=1.0e-14):
            raise RuntimeError("Distributed dipole moments do not preserve total magnetic moment")
        moment = per_dipole_moment * axis
        total_field = np.zeros(3, dtype=np.float64)
        for offset in self.axial_dipole_offsets_m():
            dipole_position = magnet.center_world + float(offset) * axis
            displacement = vector3(point_world) - dipole_position
            physical_distance = float(np.linalg.norm(displacement))
            if physical_distance <= EPS:
                continue
            effective_distance = max(physical_distance, self.config.minimum_field_distance_m)
            r_hat = displacement / physical_distance
            total_field += self.MU0 / (4.0 * np.pi * effective_distance**3) * (
                3.0 * r_hat * float(np.dot(moment, r_hat)) - moment
            )
        if not np.all(np.isfinite(total_field)):
            raise FloatingPointError("Non-finite distributed dipole field")
        return total_field

    def compute(self, magnet: MagnetState, shell: ShellState, internal_com_world: np.ndarray) -> InternalPatchResult:
        radial = magnet.center_world - shell.center_world
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm <= EPS:
            cap_axis = normalized(magnet.axis_world)
        else:
            cap_axis = radial / radial_norm
        samples = self._cap_samples(cap_axis, shell.inner_radius)
        cap_area = 2.0 * np.pi * shell.inner_radius**2 * (
            1.0 - np.cos(np.deg2rad(self.config.internal_patch_half_angle_deg))
        )
        interactions: list[MagneticInteraction] = []
        shell_force = np.zeros(3, dtype=np.float64)
        shell_torque = np.zeros(3, dtype=np.float64)
        internal_torque = np.zeros(3, dtype=np.float64)
        peak_pressure = 0.0
        unclamped_peak_pressure = 0.0
        pressure_clamp_count = 0
        peak_field = 0.0
        minimum_dipole_distance = np.inf
        ring_forces = [np.zeros(3, dtype=np.float64) for _ in range(self.config.internal_patch_rings + 1)]
        ring_max_pressures = [0.0 for _ in range(self.config.internal_patch_rings + 1)]
        points: list[np.ndarray] = []
        dipole_positions = tuple(
            magnet.center_world + float(offset) * normalized(magnet.axis_world)
            for offset in self.axial_dipole_offsets_m()
        )
        for sample in samples:
            direction = sample.direction
            point = shell.center_world + shell.inner_radius * direction
            # Free-space normal of the inner wall points toward the cavity.
            air_normal = -direction
            field = self.distributed_dipole_field_world(point, magnet)
            minimum_dipole_distance = min(
                minimum_dipole_distance,
                *(float(np.linalg.norm(point - position)) for position in dipole_positions),
            )
            normal_field = float(np.dot(field, air_normal))
            unclamped_pressure = self.config.internal_pressure_scale * normal_field**2 / (2.0 * self.MU0)
            pressure = min(unclamped_pressure, self.config.maximum_sample_pressure_pa)
            if unclamped_pressure >= self.config.maximum_sample_pressure_pa:
                pressure_clamp_count += 1
            # Maxwell pressure is normal to the spherical surface.  On the
            # inner wall ``air_normal`` points from steel into the cavity, so
            # the shell is pulled toward the magnet along air_normal and the
            # equal reaction presses the magnet/mechanism radially outward.
            force_on_shell = pressure * sample.area_m2 * air_normal
            force_on_internal = -force_on_shell
            if not np.all(np.isfinite(force_on_shell)):
                raise FloatingPointError("Non-finite internal patch force")
            interaction = MagneticInteraction(
                force_on_magnet_world=force_on_internal,
                force_on_shell_world=force_on_shell,
                magnet_application_point_world=magnet.center_world.copy(),
                shell_application_point_world=point.copy(),
                gap=float(np.linalg.norm(point - magnet.face_center_world)),
                force_law_gap=float(np.linalg.norm(point - magnet.face_center_world)),
                surface_normal_world=air_normal,
                alignment_cosine=float(np.dot(normalized(magnet.axis_world), direction)),
                raw_coupling=float(np.linalg.norm(force_on_internal)),
                normalized_weight=1.0,
                force_magnitude=float(np.linalg.norm(force_on_internal)),
                surface_method="inner-spherical-pressure-patch",
            )
            interactions.append(interaction)
            points.append(point)
            shell_force += force_on_shell
            shell_torque += np.cross(point - shell.com_world, force_on_shell)
            internal_torque += np.cross(magnet.center_world - internal_com_world, force_on_internal)
            peak_pressure = max(peak_pressure, pressure)
            unclamped_peak_pressure = max(unclamped_peak_pressure, unclamped_pressure)
            peak_field = max(peak_field, float(np.linalg.norm(field)))
            ring_forces[sample.ring_index] += force_on_shell
            ring_max_pressures[sample.ring_index] = max(
                ring_max_pressures[sample.ring_index], pressure
            )

        face_radial = float(np.dot(magnet.face_center_world - shell.center_world, cap_axis))
        internal_gap = max(shell.inner_radius - face_radial, 0.0)
        internal_force = -shell_force
        radial_force = float(np.dot(internal_force, cap_axis))
        tangential_force = float(np.linalg.norm(internal_force - radial_force * cap_axis))
        return InternalPatchResult(
            interactions=tuple(interactions),
            internal_gap_m=internal_gap,
            total_force_on_shell_world=shell_force,
            total_torque_on_shell_world=shell_torque,
            total_force_on_internal_world=internal_force,
            total_torque_on_internal_world=internal_torque,
            peak_pressure_pa=peak_pressure,
            unclamped_peak_pressure_pa=unclamped_peak_pressure,
            pressure_clamp_count=pressure_clamp_count,
            peak_field_t=peak_field,
            minimum_dipole_sample_distance_m=float(minimum_dipole_distance),
            ring_force_norms_n=tuple(float(np.linalg.norm(force)) for force in ring_forces),
            ring_max_pressure_pa=tuple(ring_max_pressures),
            sampled_area_m2=float(sum(sample.area_m2 for sample in samples)),
            cap_area_m2=cap_area,
            radial_force_on_internal_n=radial_force,
            tangential_force_on_internal_n=tangential_force,
            sample_points_world=tuple(points),
        )


class MagneticInteractionModel:
    def __init__(
        self,
        magnet_config: MagnetConfig,
        internal_force_law: Any | None = None,
        external_force_law: Any | None = None,
    ):
        self.geometry = MagneticGeometry()
        self.internal_force_law = internal_force_law or CylindricalMagnetForceLaw(magnet_config)
        self.external_force_law = external_force_law or FreeBotFemForceLaw(
            alignment_power=magnet_config.alignment_power,
        )

    def compute_candidate_geometry(
        self,
        magnet_state: MagnetState,
        shell_state: ShellState,
        surface_type: str,
        force_law_gap: float | None = None,
    ) -> MagneticCandidate | None:
        if surface_type == "inner":
            radius = shell_state.inner_radius
            force_law = self.internal_force_law
        elif surface_type == "outer":
            radius = shell_state.outer_radius
            force_law = self.external_force_law
        else:
            raise ValueError("surface_type must be 'inner' or 'outer'")

        hit = self.geometry.find_surface_point(
            magnet_state.face_center_world,
            magnet_state.axis_world,
            shell_state.center_world,
            radius,
        )
        if hit is None:
            return None
        gap_vector = hit.point_world - magnet_state.face_center_world
        gap = float(np.linalg.norm(gap_vector))
        radial_normal = normalized(hit.point_world - shell_state.center_world)
        # ``surface_normal_world`` points from the air gap into steel.  This is
        # the physically relevant incidence normal for an axis directed from
        # the magnet face toward the steel.  Consequently it is radial on an
        # inner surface and inward-radial on an outer surface.  Using the
        # opposite free-space normal would require -dot(axis, normal).
        surface_normal = radial_normal if surface_type == "inner" else -radial_normal
        alignment = float(np.clip(np.dot(magnet_state.axis_world, surface_normal), 0.0, 1.0))
        evaluated_gap = gap if force_law_gap is None else max(float(force_law_gap), 0.0)
        raw_coupling = force_law.force_n(evaluated_gap, alignment)
        return MagneticCandidate(
            surface_point_world=hit.point_world.copy(),
            surface_normal_world=surface_normal,
            gap=gap,
            force_law_gap=evaluated_gap,
            alignment_cosine=alignment,
            raw_coupling=raw_coupling,
            is_system_calibrated=bool(getattr(force_law, "is_system_calibrated", False)),
            surface_method=hit.method,
        )

    def compute_force_from_candidate(
        self,
        magnet_state: MagnetState,
        candidate: MagneticCandidate,
        normalized_weight: float,
    ) -> MagneticInteraction:
        weight = float(normalized_weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Coupling weight must lie in [0, 1], got {weight}")
        gap_vector = candidate.surface_point_world - magnet_state.face_center_world
        force_direction = magnet_state.axis_world.copy() if candidate.gap <= EPS else gap_vector / candidate.gap
        magnitude = candidate.raw_coupling * weight
        force_on_magnet = magnitude * force_direction
        force_on_shell = -force_on_magnet
        np.testing.assert_allclose(force_on_magnet + force_on_shell, np.zeros(3), atol=1.0e-10)
        return MagneticInteraction(
            force_on_magnet_world=force_on_magnet,
            force_on_shell_world=force_on_shell,
            magnet_application_point_world=magnet_state.face_center_world.copy(),
            shell_application_point_world=candidate.surface_point_world.copy(),
            gap=candidate.gap,
            force_law_gap=candidate.force_law_gap,
            surface_normal_world=candidate.surface_normal_world.copy(),
            alignment_cosine=candidate.alignment_cosine,
            raw_coupling=candidate.raw_coupling,
            normalized_weight=weight,
            force_magnitude=magnitude,
            surface_method=candidate.surface_method,
        )

    def compute_competing_interactions(
        self,
        magnet_state: MagnetState,
        surfaces: dict[str, tuple[ShellState, str] | tuple[ShellState, str, float | None]],
    ) -> dict[str, MagneticInteraction]:
        """Allocate one magnet's coupling through an equivalent magnetic circuit.

        Analytic raw couplings contain gap, effective pole area and local-normal
        alignment and are normalized to a unit total weight.  A force law
        explicitly marked as system-calibrated bypasses this normalization:
        its measured/FEM output already contains the shell-mediated magnetic
        circuit and normalizing it again would count the same flux reduction
        twice.
        """
        candidates: dict[str, MagneticCandidate | None] = {}
        for name, descriptor in surfaces.items():
            shell, surface_type = descriptor[0], descriptor[1]
            force_law_gap = descriptor[2] if len(descriptor) == 3 else None
            candidates[name] = self.compute_candidate_geometry(
                magnet_state,
                shell,
                surface_type,
                force_law_gap,
            )
        visible = {name: candidate for name, candidate in candidates.items() if candidate is not None}
        calibrated = {name: item for name, item in visible.items() if item.is_system_calibrated}
        allocation_pool = {name: item for name, item in visible.items() if not item.is_system_calibrated}
        total_coupling = float(sum(candidate.raw_coupling for candidate in allocation_pool.values()))
        interactions = {
            name: self.compute_force_from_candidate(magnet_state, candidate, 1.0)
            for name, candidate in calibrated.items()
            if candidate.raw_coupling > EPS
        }
        if total_coupling > EPS:
            allocated = {
                name: self.compute_force_from_candidate(
                    magnet_state,
                    candidate,
                    candidate.raw_coupling / total_coupling,
                )
                for name, candidate in allocation_pool.items()
            }
            weight_sum = sum(interaction.normalized_weight for interaction in allocated.values())
            if not np.isclose(weight_sum, 1.0, atol=1.0e-12):
                raise RuntimeError(f"Magnetic-circuit weights sum to {weight_sum}")
            interactions.update(allocated)
        return interactions


class ForcePairApplier:
    @staticmethod
    def apply(interaction: MagneticInteraction, magnet_body: Any, shell_body: Any) -> None:
        residual = interaction.force_on_magnet_world + interaction.force_on_shell_world
        if float(np.linalg.norm(residual)) > 1.0e-9:
            raise RuntimeError(f"Magnetic action/reaction residual is {residual}")
        magnet_body.apply_forces_and_torques_at_pos(
            forces=interaction.force_on_magnet_world.reshape(1, 3),
            positions=interaction.magnet_application_point_world.reshape(1, 3),
        )
        shell_body.apply_forces_and_torques_at_pos(
            forces=interaction.force_on_shell_world.reshape(1, 3),
            positions=interaction.shell_application_point_world.reshape(1, 3),
        )

    @staticmethod
    def torques(interaction: MagneticInteraction, magnet_com: np.ndarray, shell_com: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        magnet_torque = np.cross(
            interaction.magnet_application_point_world - magnet_com,
            interaction.force_on_magnet_world,
        )
        shell_torque = np.cross(
            interaction.shell_application_point_world - shell_com,
            interaction.force_on_shell_world,
        )
        return magnet_torque, shell_torque


@dataclass
class FreeBotModule:
    root_path: str
    shell_body: Any
    internal_body: Any
    magnet_frame: Any
    left_wheel_body: Any
    right_wheel_body: Any
    caster_1_body: Any
    caster_2_body: Any

    @property
    def shell_path(self) -> str:
        return f"{self.root_path}/shell_link"

    def shell_state(self, geometry: GeometryConfig) -> ShellState:
        center = self.sphere_center_world(geometry)
        return ShellState(center, self.body_com_world(self.shell_body), geometry.shell_inner_radius_m, geometry.shell_outer_radius_m)

    def sphere_center_world(self, geometry: GeometryConfig) -> np.ndarray:
        poses = self.shell_body.get_world_poses()
        body_origin = first_vector(poses[0])
        body_orientation = first_quaternion(poses[1])
        local_offset = vector3(geometry.shell_center_from_body_origin_m)
        return body_origin + quaternion_rotate_wxyz(body_orientation, local_offset)

    @staticmethod
    def body_com_world(body: Any) -> np.ndarray:
        """Transform the PhysX local COM returned by RigidPrim.get_coms to world."""
        world_poses = body.get_world_poses()
        body_position = first_vector(world_poses[0])
        body_orientation = first_quaternion(world_poses[1])
        local_com_position = first_vector(body.get_coms()[0])
        return body_position + quaternion_rotate_wxyz(body_orientation, local_com_position)

    def internal_com_world(self) -> np.ndarray:
        return self.body_com_world(self.internal_body)

    def mechanism_angle_deg(self, geometry: GeometryConfig) -> float:
        """Magnet radial angle in the X-Z plane: 0=down, +90=+X side."""
        shell_center = self.sphere_center_world(geometry)
        magnet_center = first_vector(self.magnet_frame.get_world_poses()[0])
        radial = magnet_center - shell_center
        return float(np.degrees(np.arctan2(radial[0], -radial[2])))

    def wheel_proxy_pose_world(self, wheel_body: Any, geometry: GeometryConfig) -> tuple[np.ndarray, np.ndarray]:
        """Return the fitted CAD tire centre and spin axis in world coordinates."""
        wheel_pose = wheel_body.get_world_poses()
        body_center = first_vector(wheel_pose[0])
        orientation = first_quaternion(wheel_pose[1])
        nominal_axis = normalized(quaternion_rotate_wxyz(orientation, [0.0, 1.0, 0.0]))
        shell_radial = body_center - self.sphere_center_world(geometry)
        outward_sign = 1.0 if float(np.dot(shell_radial, nominal_axis)) >= 0.0 else -1.0
        center = body_center + outward_sign * geometry.tire_center_axial_offset_m * nominal_axis
        local_tilt_rad = np.radians(geometry.tire_axis_tilt_deg)
        fitted_local_axis = [0.0, np.cos(local_tilt_rad), np.sin(local_tilt_rad)]
        axis = normalized(quaternion_rotate_wxyz(orientation, fitted_local_axis))
        return center, axis

    def wheel_inner_shell_clearance_m(self, wheel_body: Any, geometry: GeometryConfig) -> float:
        """Conservative CAD tire-envelope clearance from the analytic inner sphere."""
        shell_center = self.sphere_center_world(geometry)
        wheel_center, spin_axis = self.wheel_proxy_pose_world(wheel_body, geometry)
        center_offset = wheel_center - shell_center
        axial_offset = abs(float(np.dot(center_offset, spin_axis)))
        perpendicular_offset = float(
            np.linalg.norm(center_offset - np.dot(center_offset, spin_axis) * spin_axis)
        )
        envelope_radius = np.hypot(
            axial_offset + geometry.tire_half_width_m,
            perpendicular_offset + geometry.tire_outer_radius_m,
        )
        return geometry.shell_inner_radius_m - float(envelope_radius)

    def caster_inner_shell_clearance_m(self, caster_body: Any, geometry: GeometryConfig) -> float:
        """Analytic ball-to-inner-shell clearance; zero is first contact."""
        shell_center = self.sphere_center_world(geometry)
        caster_center = first_vector(caster_body.get_world_poses()[0])
        outermost_radius = float(np.linalg.norm(caster_center - shell_center)) + geometry.caster_ball_radius_m
        return geometry.shell_inner_radius_m - outermost_radius

    @staticmethod
    def wheel_spin_deg_s(wheel_body: Any) -> float:
        orientation = first_quaternion(wheel_body.get_world_poses()[1])
        spin_axis = normalized(quaternion_rotate_wxyz(orientation, [0.0, 1.0, 0.0]))
        angular_velocity = first_vector(wheel_body.get_velocities()[1])
        return float(np.degrees(np.dot(angular_velocity, spin_axis)))

    def wheel_slip_diagnostic(
        self,
        wheel_body: Any,
        shell_state: ShellState,
        geometry: GeometryConfig,
    ) -> WheelSlipDiagnostic:
        wheel_pose = wheel_body.get_world_poses()
        wheel_center, spin_axis = self.wheel_proxy_pose_world(wheel_body, geometry)
        radial = wheel_center - shell_state.center_world
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm <= EPS:
            raise ValueError("Cannot estimate wheel contact at the shell centre")
        radial_dir = radial / radial_norm
        axial_cosine = float(np.dot(radial_dir, spin_axis))
        radial_perpendicular = radial_dir - axial_cosine * spin_axis
        radial_perpendicular_norm = float(np.linalg.norm(radial_perpendicular))
        radial_perpendicular_dir = (
            radial_perpendicular / radial_perpendicular_norm
            if radial_perpendicular_norm > EPS else np.zeros(3, dtype=np.float64)
        )
        axial_sign = 1.0 if axial_cosine >= 0.0 else -1.0
        wheel_contact_offset = (
            geometry.tire_outer_radius_m * radial_perpendicular_dir
            + geometry.tire_half_width_m * axial_sign * spin_axis
        )
        wheel_contact_point = wheel_center + wheel_contact_offset
        shell_contact_point = shell_state.center_world + shell_state.inner_radius * radial_dir

        wheel_velocities = wheel_body.get_velocities()
        wheel_linear_velocity = first_vector(wheel_velocities[0])
        wheel_angular_velocity = first_vector(wheel_velocities[1])
        shell_velocities = self.shell_body.get_velocities()
        shell_linear_velocity = first_vector(shell_velocities[0])
        shell_angular_velocity = first_vector(shell_velocities[1])
        v_wheel_point = wheel_linear_velocity + np.cross(
            wheel_angular_velocity, wheel_contact_point - wheel_center
        )
        v_shell_point = shell_linear_velocity + np.cross(
            shell_angular_velocity, shell_contact_point - shell_state.com_world
        )

        tangent_vector = np.cross(spin_axis, radial_dir)
        if float(np.linalg.norm(tangent_vector)) <= EPS:
            rolling_tangent = np.zeros(3, dtype=np.float64)
        else:
            rolling_tangent = normalized(tangent_vector)
        spin_deg_s = self.wheel_spin_deg_s(wheel_body)
        wheel_surface_speed = float(np.radians(spin_deg_s) * geometry.tire_outer_radius_m)
        relative_tangent_speed = float(np.dot(v_wheel_point - v_shell_point, rolling_tangent))
        shell_tangent_speed = float(np.dot(v_shell_point, rolling_tangent))
        reference_speed = max(abs(wheel_surface_speed), abs(shell_tangent_speed), 0.01)
        radial_support = float(np.dot(wheel_contact_offset, radial_dir))
        clearance = shell_state.inner_radius - (radial_norm + radial_support)
        return WheelSlipDiagnostic(
            wheel_surface_speed_m_s=wheel_surface_speed,
            estimated_contact_relative_tangent_speed_m_s=relative_tangent_speed,
            slip_speed_m_s=relative_tangent_speed,
            slip_ratio=relative_tangent_speed / reference_speed,
            clearance_m=clearance,
            estimated_contact=abs(clearance) < geometry.wheel_contact_tolerance_m,
        )

    def magnet_state(self, config: MagnetConfig) -> MagnetState:
        poses = self.magnet_frame.get_world_poses()
        center = first_vector(poses[0])
        orientation = first_quaternion(poses[1])
        axis = normalized(quaternion_rotate_wxyz(orientation, config.local_active_axis))
        face = center + axis * (0.5 * config.length_m)
        return MagnetState(center, orientation, axis, face, 0.5 * config.length_m, config.radius_m)


class PhysxContactDiagnostics:
    """Report solver contact normals and friction forces for selected pairs."""

    def __init__(self, rigid_prim_type: Any, pairs: Iterable[tuple[str, str, str]]):
        self.pairs = []
        for label, sensor_path, filter_path in pairs:
            sensor = rigid_prim_type(
                paths=sensor_path,
                contact_filter_paths=filter_path,
                max_contact_count=64,
            )
            # Isaac Sim 6 does not implicitly author PhysxContactReportAPI
            # when contact_filter_paths is supplied.  It must exist before
            # the physics-ready event creates the tensor contact view.
            sensor.set_enabled_contact_tracking([True], threshold=0.0)
            self.pairs.append((label, sensor))

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def formatted_lines(self, dt: float) -> list[str]:
        lines = []
        for label, sensor in self.pairs:
            try:
                normal_matrix = self._numpy(sensor.get_contact_force_matrix(dt=dt))
            except (AssertionError, AttributeError, RuntimeError) as error:
                lines.append(f"  CONTACT {label}: unavailable ({type(error).__name__}: {error})")
                continue
            normal = np.asarray(normal_matrix[0, 0], dtype=np.float64)

            friction_forces, _, counts, starts = sensor.get_friction_data(dt=dt)
            friction_forces = self._numpy(friction_forces)
            counts = self._numpy(counts)
            starts = self._numpy(starts)
            friction_count = int(counts[0, 0])
            friction_start = int(starts[0, 0])
            if friction_count:
                friction = np.sum(
                    friction_forces[friction_start : friction_start + friction_count],
                    axis=0,
                    dtype=np.float64,
                )
            else:
                friction = np.zeros(3, dtype=np.float64)

            _, points, _, separations, contact_counts, contact_starts = sensor.get_contact_force_data(dt=dt)
            points = self._numpy(points)
            separations = self._numpy(separations)
            contact_counts = self._numpy(contact_counts)
            contact_starts = self._numpy(contact_starts)
            contact_count = int(contact_counts[0, 0])
            contact_start = int(contact_starts[0, 0])
            if contact_count:
                selected_points = points[contact_start : contact_start + contact_count]
                selected_separations = separations[contact_start : contact_start + contact_count]
                mean_point = np.mean(selected_points, axis=0)
                minimum_separation = float(np.min(selected_separations))
            else:
                mean_point = np.zeros(3, dtype=np.float64)
                minimum_separation = float("nan")

            normal_norm = float(np.linalg.norm(normal))
            friction_norm = float(np.linalg.norm(friction))
            utilization = friction_norm / normal_norm if normal_norm > EPS else float("nan")
            lines.append(
                f"  CONTACT {label}: count={contact_count} "
                f"Fn=({normal[0]:+.3f},{normal[1]:+.3f},{normal[2]:+.3f})N |Fn|={normal_norm:.3f}N "
                f"Ft=({friction[0]:+.3f},{friction[1]:+.3f},{friction[2]:+.3f})N |Ft|={friction_norm:.3f}N "
                f"mu_used={utilization:.3f} min_sep={1e3*minimum_separation:+.3f}mm "
                f"Pmean=({mean_point[0]:+.4f},{mean_point[1]:+.4f},{mean_point[2]:+.4f})m"
            )
        return lines


class DebugLines:
    """World-space diagnostics. These primitives never participate in physics."""

    def __init__(self, stage: Any, enabled: bool, force_scale_m_per_n: float):
        self.enabled = enabled
        self.curves = None
        self.points_attr = None
        self.counts_attr = None
        self.color_primvar = None
        self.force_scale = float(force_scale_m_per_n)
        if self.force_scale <= 0.0:
            raise ValueError("debug force scale must be positive")
        if not enabled:
            return
        from pxr import UsdGeom, Vt

        self.curves = UsdGeom.BasisCurves.Define(stage, "/World/freebot_magnetic_debug")
        self.curves.CreateTypeAttr(UsdGeom.Tokens.linear)
        self.curves.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        self.points_attr = self.curves.CreatePointsAttr()
        self.counts_attr = self.curves.CreateCurveVertexCountsAttr()
        self.color_primvar = self.curves.CreateDisplayColorPrimvar(UsdGeom.Tokens.uniform)
        self.curves.CreateWidthsAttr().Set(Vt.FloatArray([0.0015]))
        self.curves.SetWidthsInterpolation(UsdGeom.Tokens.constant)

    def update(
        self,
        magnets: list[MagnetState],
        interactions: list[tuple[str, MagneticInteraction]],
    ) -> None:
        if not self.enabled or self.curves is None:
            return
        from pxr import Gf, Vt

        yellow = Gf.Vec3f(1.0, 0.9, 0.1)
        white = Gf.Vec3f(1.0, 1.0, 1.0)
        magenta = Gf.Vec3f(1.0, 0.1, 1.0)
        green = Gf.Vec3f(0.1, 1.0, 0.2)
        orange = Gf.Vec3f(1.0, 0.35, 0.05)
        segments: list[tuple[np.ndarray, np.ndarray]] = []
        colors: list[Any] = []
        for magnet in magnets:
            # Long enough to remain visible and cross the other 126.667 mm
            # module.  This is display-only and never enters the force model.
            segments.append((magnet.face_center_world, magnet.face_center_world + 0.200 * magnet.axis_world))
            colors.append(yellow)
        for _, item in interactions:
            segments.append((item.magnet_application_point_world, item.shell_application_point_world))
            colors.append(white)
            segments.append(
                (
                    item.shell_application_point_world,
                    item.shell_application_point_world + 0.020 * item.surface_normal_world,
                )
            )
            colors.append(magenta)
            segments.append(
                (
                    item.magnet_application_point_world,
                    item.magnet_application_point_world + self.force_scale * item.force_on_magnet_world,
                )
            )
            colors.append(green)
            segments.append(
                (
                    item.shell_application_point_world,
                    item.shell_application_point_world + self.force_scale * item.force_on_shell_world,
                )
            )
            colors.append(orange)
        points = [Gf.Vec3f(*(float(x) for x in point)) for segment in segments for point in segment]
        self.counts_attr.Set(Vt.IntArray([2] * len(segments)))
        self.points_attr.Set(Vt.Vec3fArray(points))
        self.color_primvar.Set(Vt.Vec3fArray(colors))


class DiagnosticsLogger:
    @staticmethod
    def format_interaction(
        name: str,
        interaction: MagneticInteraction | None,
        magnet_com: np.ndarray | None = None,
        shell_com: np.ndarray | None = None,
    ) -> str:
        if interaction is None:
            return f"{name}=none"
        residual = interaction.force_on_magnet_world + interaction.force_on_shell_world
        torque_text = ""
        if magnet_com is not None and shell_com is not None:
            magnet_torque, shell_torque = ForcePairApplier.torques(interaction, magnet_com, shell_com)
            torque_text = (
                f" Tmag=({magnet_torque[0]:+.4f},{magnet_torque[1]:+.4f},{magnet_torque[2]:+.4f})Nm"
                f" Tshell=({shell_torque[0]:+.4f},{shell_torque[1]:+.4f},{shell_torque[2]:+.4f})Nm"
            )
        paper_text = ""
        if interaction.surface_method.startswith("paper-"):
            paper_text = (
                f" paper_theta={interaction.paper_theta_deg:.2f}deg"
                f" Aparallel={interaction.parallel_force_n:+.4f}N"
                f" Aperpendicular={interaction.perpendicular_force_n:+.4f}N"
                f" branch_weight={interaction.branch_weight:.4f}"
            )
        return (
            f"{name}:gap={1e3 * interaction.gap:.2f}mm "
            f"law_gap={1e3 * interaction.force_law_gap:.2f}mm "
            f"normal=({interaction.surface_normal_world[0]:+.3f},{interaction.surface_normal_world[1]:+.3f},{interaction.surface_normal_world[2]:+.3f}) "
            f"align={interaction.alignment_cosine:.4f} raw={interaction.raw_coupling:.4f}N "
            f"weight={interaction.normalized_weight:.4f} F={interaction.force_magnitude:.4f}N "
            f"Fvec=({interaction.force_on_magnet_world[0]:+.4f},{interaction.force_on_magnet_world[1]:+.4f},{interaction.force_on_magnet_world[2]:+.4f})N "
            f"residual={np.linalg.norm(residual):.3e}N "
            f"Pmag=({interaction.magnet_application_point_world[0]:+.4f},{interaction.magnet_application_point_world[1]:+.4f},{interaction.magnet_application_point_world[2]:+.4f})m "
            f"Pshell=({interaction.shell_application_point_world[0]:+.4f},{interaction.shell_application_point_world[1]:+.4f},{interaction.shell_application_point_world[2]:+.4f})m "
            f"via={interaction.surface_method}{paper_text}{torque_text}"
        )


class Ros2CliTeleop:
    """Read geometry_msgs/Twist without loading ROS libraries into Isaac Python.

    This deliberately follows the bridge used by the previous two-module
    runner: a ROS 2 CLI subprocess emits CSV while a background thread keeps
    the physics loop non-blocking.
    """

    def __init__(self, topic: str, timeout_s: float):
        if timeout_s <= 0.0:
            raise ValueError("ROS 2 command timeout must be positive")
        self.timeout_s = float(timeout_s)
        self.linear_x = 0.0
        self.angular_z = 0.0
        self.last_message_time = 0.0
        self.received = False
        self._reported_first_message = False
        self._reported_exit = False
        self._lock = threading.Lock()
        command = (
            "source /opt/ros/humble/setup.bash && "
            f"ros2 topic echo {shlex.quote(topic)} geometry_msgs/msg/Twist --csv"
        )
        environment = os.environ.copy()
        environment.pop("PYTHONHOME", None)
        environment.pop("PYTHONPATH", None)
        self._process = subprocess.Popen(
            ["bash", "-lc", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        self._thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._thread.start()

    def _read_stdout(self) -> None:
        if self._process.stdout is None:
            return
        number_pattern = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
        for line in self._process.stdout:
            values = [float(value) for value in re.findall(number_pattern, line)]
            if len(values) < 6:
                stripped = line.strip()
                if stripped and not stripped.startswith(("linear", "---")):
                    print(f"ROS 2 teleop bridge: {stripped}")
                continue
            with self._lock:
                self.linear_x = values[0]
                self.angular_z = values[5]
                self.last_message_time = time.monotonic()
                self.received = True
                if not self._reported_first_message:
                    print(
                        "ROS 2 teleop received first Twist: "
                        f"linear.x={self.linear_x:+.3f}, angular.z={self.angular_z:+.3f}"
                    )
                    self._reported_first_message = True

    def command(self) -> tuple[float, float]:
        return_code = self._process.poll()
        if return_code is not None:
            if not self._reported_exit:
                print(f"ROS 2 teleop bridge exited with code {return_code}; wheel command is zero.")
                self._reported_exit = True
            return 0.0, 0.0
        with self._lock:
            expired = time.monotonic() - self.last_message_time > self.timeout_s
            if not self.received or expired:
                return 0.0, 0.0
            return self.linear_x, self.angular_z

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._process.kill()


def twist_to_wheel_velocities(
    linear_x: float,
    angular_z: float,
    config: SimulationConfig,
) -> tuple[float, float, float, float]:
    """Map Twist to left/right drive targets exactly as in the old runner."""
    if linear_x == 0.0 and angular_z == 0.0:
        return 0.0, 0.0, 0.0, 0.0
    forward = config.cmd_linear_sign * config.cmd_linear_scale * float(linear_x)
    turn = config.cmd_angular_sign * config.cmd_angular_scale * float(angular_z)
    return float(forward - turn), float(forward + turn), float(forward), float(turn)


SOURCE_ROOT = "/World/freebot"
ACTIVE_ROOT = "/World/active_module"
TARGET_ROOT = "/World/target_module"
# Nominal CAD shell centre, identical to the shell_link authored origin in the
# generated USD.  This datum also reproduces the measured ~9 mm internal gap.
SOURCE_SHELL_CENTER = np.array([0.02465733, 0.06062240, 0.06061104], dtype=np.float64)
SOURCE_MAGNET_CENTER = np.array([0.023121, 0.060587, 0.009453], dtype=np.float64)


def source_magnet_face_to_inner_shell_gap_m(geometry: GeometryConfig, magnet_half_length_m: float = 0.005) -> float:
    """CAD datum check for the bottom-facing magnet pole and inner shell."""
    radial_center_distance = float(np.linalg.norm(SOURCE_MAGNET_CENTER - SOURCE_SHELL_CENTER))
    return geometry.shell_inner_radius_m - radial_center_distance - float(magnet_half_length_m)


def _set_root_transform(stage: Any, path: str, desired_shell_center: np.ndarray, y_rotation_deg: float) -> None:
    from pxr import Gf, UsdGeom

    xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    orientation = quaternion_y_degrees(y_rotation_deg)
    rotated_source_center = quaternion_rotate_wxyz(orientation, SOURCE_SHELL_CENTER)
    translation = vector3(desired_shell_center) - rotated_source_center
    transform = Gf.Matrix4d(1.0)
    transform.SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), float(y_rotation_deg)))
    transform.SetTranslateOnly(Gf.Vec3d(*(float(value) for value in translation)))
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(transform)


def _retarget_relationships(stage: Any, old_root: str, new_root: str) -> None:
    from pxr import Sdf, Usd

    old = Sdf.Path(old_root)
    new = Sdf.Path(new_root)
    for prim in Usd.PrimRange(stage.GetPrimAtPath(new_root)):
        for relationship in prim.GetRelationships():
            targets = relationship.GetTargets()
            changed = False
            rewritten = []
            for target in targets:
                if target.HasPrefix(old):
                    rewritten.append(new.AppendPath(target.MakeRelativePath(old)))
                    changed = True
                else:
                    rewritten.append(target)
            if changed:
                relationship.SetTargets(rewritten)


def _clone_module(stage: Any, source: str, destination: str) -> None:
    from pxr import Sdf

    layer = stage.GetRootLayer()
    if stage.GetPrimAtPath(destination):
        stage.RemovePrim(destination)
    Sdf.CopySpec(layer, Sdf.Path(source), layer, Sdf.Path(destination))
    _retarget_relationships(stage, source, destination)


def _make_all_rigid_bodies_dynamic(stage: Any, root: str) -> None:
    from pxr import Sdf, Usd, UsdPhysics

    rigid_body_count = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rigid_body = UsdPhysics.RigidBodyAPI.Apply(prim)
        rigid_body.CreateKinematicEnabledAttr(False)
        # Isaac Sim 6/TGS warns and changes behaviour above four velocity
        # iterations.  Keep the authored high position iterations, but use a
        # supported velocity count to avoid impulsive contact/joint response.
        prim.CreateAttribute(
            "physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int
        ).Set(4)
        if bool(rigid_body.GetKinematicEnabledAttr().Get()):
            raise RuntimeError(f"Rigid body remained kinematic: {prim.GetPath()}")
        rigid_body_count += 1
    if rigid_body_count == 0:
        raise RuntimeError(f"Module {root} contains no rigid bodies")


def _remove_duplicate_container_colliders(stage: Any, root: str) -> int:
    """Remove CollisionAPI from non-leaf CAD containers.

    The generated USD currently authors CollisionAPI on both a part Xform and
    its descendant mesh.  That duplicates the same shell/tire geometry in one
    rigid body and can produce explosive depenetration at first contact.
    """
    from pxr import Usd, UsdPhysics

    removed = 0
    for prim in list(Usd.PrimRange(stage.GetPrimAtPath(root))):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        has_descendant_collider = any(
            child != prim
            and child.HasAPI(UsdPhysics.CollisionAPI)
            for child in Usd.PrimRange(prim)
        )
        if not has_descendant_collider:
            continue
        prim.RemoveAPI(UsdPhysics.CollisionAPI)
        prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        removed += 1
    return removed


def _replace_wheel_sdf_with_cylinder_proxies(stage: Any, root: str, geometry: GeometryConfig) -> int:
    """Use stable cylinders fit to the transformed CAD tire vertices.

    Radius, half-width, centre offset and axis tilt come from the vertex fit;
    axis-aligned bounds are deliberately not used.  Visual geometry is left
    untouched.
    """
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    removed = 0
    for prim in list(Usd.PrimRange(stage.GetPrimAtPath(root))):
        path = str(prim.GetPath())
        if "wheel_link" in path and "tire" in path.lower() and prim.HasAPI(UsdPhysics.CollisionAPI):
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            removed += 1
    for side in ("left", "right"):
        proxy_path = f"{root}/{side}_wheel_link/tire_collision_proxy"
        cylinder = UsdGeom.Cylinder.Define(stage, proxy_path)
        cylinder.CreateAxisAttr(UsdGeom.Tokens.y)
        cylinder.CreateRadiusAttr(float(geometry.tire_outer_radius_m))
        cylinder.CreateHeightAttr(float(2.0 * geometry.tire_half_width_m))
        cylinder.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        outward_sign = 1.0 if side == "left" else -1.0
        cylinder.AddTranslateOp().Set(
            Gf.Vec3d(0.0, outward_sign * geometry.tire_center_axial_offset_m, 0.0)
        )
        cylinder.AddRotateXOp().Set(float(geometry.tire_axis_tilt_deg))
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    return removed


def _replace_caster_sdf_with_sphere_proxies(
    stage: Any,
    root: str,
    geometry: GeometryConfig,
    enabled: bool = True,
) -> int:
    """Replace imported caster SDFs with vertex-fit spherical colliders.

    The visual CAD remains untouched.  Each proxy is authored below its caster
    rigid body, so it follows the existing joint and uses the authored caster
    centre without adding another rigid body or changing its dynamics.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    removed = 0
    for caster_name in ("caster_1_ball_link", "caster_2_ball_link"):
        caster_path = f"{root}/{caster_name}"
        caster_prim = stage.GetPrimAtPath(caster_path)
        for prim in list(Usd.PrimRange(caster_prim)):
            if prim == caster_prim or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            removed += 1

        if enabled:
            sphere = UsdGeom.Sphere.Define(stage, f"{caster_path}/caster_collision_proxy")
            sphere.CreateRadiusAttr(float(geometry.caster_ball_radius_m))
            sphere.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
    return removed


def _position_module(stage: Any, root: str, desired_shell_center: Iterable[float], y_rotation_deg: float) -> None:
    _set_root_transform(stage, root, vector3(desired_shell_center), y_rotation_deg)


def _validate_initial_conditions(config: SimulationConfig) -> float:
    active_center = vector3(config.active_start)
    target_center = vector3(config.target_start)
    center_delta = target_center - active_center
    center_distance = float(np.linalg.norm(center_delta))
    shell_gap = center_distance - 2.0 * config.geometry.shell_outer_radius_m
    if shell_gap < 0.0:
        raise ValueError(f"Initial modules interpenetrate by {-1e3 * shell_gap:.2f} mm")
    if not 0.0 <= shell_gap <= 0.100:
        raise ValueError(f"Initial shell gap must be 0--100 mm, got {1e3 * shell_gap:.2f} mm")
    return shell_gap


def spherical_shell_gap(first: ShellState, second: ShellState) -> float:
    center_distance = float(np.linalg.norm(second.center_world - first.center_world))
    return max(center_distance - first.outer_radius - second.outer_radius, 0.0)


def magnetic_shell_gap_from_cad_gap(cad_gap_m: float, contact_offset_m: float) -> float:
    """Map nominal CAD-sphere gap to the collider surface gap continuously."""
    return max(float(cad_gap_m) - float(contact_offset_m), 0.0)


def paper_required_ground_friction(
    theta_deg: float,
    parallel_force_n: float,
    perpendicular_force_n: float,
    gravity_force_n: float,
    shell_radius_m: float,
    com_radius_m: float,
    shell_shell_friction: float,
) -> float:
    """Equation (7): limiting ground friction for dock/undock motion.

    This is a diagnostic condition, not an applied force.  Returning infinity
    marks a singular or infeasible static balance at the supplied pose.
    """
    theta = np.deg2rad(float(theta_deg))
    radius = float(shell_radius_m)
    com_radius = float(com_radius_m)
    mu1 = float(shell_shell_friction)
    r1 = radius - com_radius * np.cos(theta)
    r2 = com_radius * np.sin(theta)
    r3 = radius - com_radius * np.sin(theta)
    r4 = com_radius * np.cos(theta)
    numerator = (
        parallel_force_n * (r2 + mu1 * r1 + mu1 * r4)
        + (perpendicular_force_n - gravity_force_n) * r4
    )
    denominator = (
        (perpendicular_force_n - gravity_force_n) * (r2 + r3 + mu1 * r1)
        + mu1 * parallel_force_n * r3
    )
    if abs(denominator) <= EPS:
        return float("inf")
    result = float(numerator / denominator)
    return result if result >= 0.0 else float("inf")


def paper_required_connection_friction(
    theta_deg: float,
    magnetic_force_n: float,
    gravity_force_n: float,
    lower_hemisphere: bool,
) -> float:
    """Equations (9) and (12) for a statically connected module."""
    theta = np.deg2rad(float(theta_deg))
    attraction = float(magnetic_force_n)
    gravity = float(gravity_force_n)
    if attraction <= EPS:
        return float("inf")
    normal = attraction - gravity * np.sin(theta) if lower_hemisphere else attraction + gravity * np.sin(theta)
    if normal <= EPS:
        return float("inf")
    return max(float(gravity * np.cos(theta) / normal), 0.0)


def _smoothstep01(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _paper_force_interaction(
    active_magnet: MagnetState,
    target_point_world: np.ndarray,
    force_on_active_world: np.ndarray,
    shell_gap: float,
    connection_normal_world: np.ndarray,
    angle_deg: float,
    parallel_force_n: float,
    perpendicular_force_n: float,
    branch_weight: float,
    method: str,
) -> MagneticInteraction:
    force_on_active = vector3(force_on_active_world)
    target_point = vector3(target_point_world)
    magnitude = float(np.linalg.norm(force_on_active))
    return MagneticInteraction(
        force_on_magnet_world=force_on_active,
        force_on_shell_world=-force_on_active,
        magnet_application_point_world=active_magnet.face_center_world.copy(),
        shell_application_point_world=target_point,
        gap=float(np.linalg.norm(target_point - active_magnet.face_center_world)),
        force_law_gap=max(float(shell_gap), 0.0),
        surface_normal_world=vector3(connection_normal_world),
        alignment_cosine=float(np.cos(np.deg2rad(angle_deg))),
        raw_coupling=magnitude,
        normalized_weight=float(branch_weight),
        force_magnitude=magnitude,
        surface_method=method,
        parallel_force_n=float(parallel_force_n),
        perpendicular_force_n=float(perpendicular_force_n),
        paper_theta_deg=float(angle_deg),
        branch_weight=float(branch_weight),
    )


def compute_exponential_external_interactions(
    active_magnet: MagnetState,
    active_shell: ShellState,
    passive_shell: ShellState,
    shell_gap: float,
    angular_force_law: FreeBotAngularForceLaw,
    distance_scale_m: float = 0.0035,
    angle_scale_deg: float = 45.0,
    angle_power: float = 2.0,
) -> dict[str, MagneticInteraction]:
    """Continuous empirical magnet-to-passive-shell attraction.

    The magnitude is 22.6 N at zero shell gap and exact alignment and remains
    positive for every finite distance and angle.  Fig. 5 is used only for
    the direction ratio of its published parallel/perpendicular components;
    it does not gate or scale the magnitude.  Fig. 6 remains disabled.
    """
    if distance_scale_m <= 0.0 or angle_scale_deg <= 0.0 or angle_power <= 0.0:
        raise ValueError("Exponential external-force scales and power must be positive")
    centre_delta = passive_shell.center_world - active_shell.center_world
    centre_distance = float(np.linalg.norm(centre_delta))
    if centre_distance <= EPS:
        return {}
    n = centre_delta / centre_distance
    active_axis = normalized(active_magnet.axis_world)
    cosine = float(np.clip(np.dot(active_axis, n), -1.0, 1.0))
    theta_deg = float(np.degrees(np.arccos(cosine)))
    tangent_component = active_axis - cosine * n
    tangent_norm = float(np.linalg.norm(tangent_component))
    t = tangent_component / tangent_norm if tangent_norm > EPS else np.zeros(3, dtype=np.float64)

    magnitude = 22.6 * np.exp(-max(float(shell_gap), 0.0) / distance_scale_m) * np.exp(
        -(theta_deg / angle_scale_deg) ** angle_power
    )
    if not np.isfinite(magnitude) or magnitude <= 0.0:
        return {}

    direction_theta = min(theta_deg, 90.0)
    direction_parallel, direction_perpendicular = angular_force_law.shell_components_n(direction_theta)
    direction = direction_parallel * n - direction_perpendicular * t
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= EPS:
        direction = n
        direction_norm = 1.0
    force = float(magnitude) * direction / direction_norm
    parallel = float(np.dot(force, n))
    perpendicular = float(-np.dot(force, t)) if tangent_norm > EPS else 0.0
    patch = MagneticGeometry.ray_sphere_intersection(
        active_magnet.face_center_world,
        force,
        passive_shell.center_world,
        passive_shell.outer_radius,
    )
    if patch is None:
        patch = passive_shell.center_world - passive_shell.outer_radius * n
    return {
        "active_field_to_passive_shell": _paper_force_interaction(
            active_magnet,
            patch,
            force,
            shell_gap,
            n,
            theta_deg,
            parallel,
            perpendicular,
            1.0,
            "continuous-exponential-to-passive-shell",
        )
    }


def compute_paper_external_interactions(
    active_magnet: MagnetState,
    passive_magnet: MagnetState,
    active_shell: ShellState,
    passive_shell: ShellState,
    shell_gap: float,
    gap_force_law: FreeBotFemForceLaw,
    angular_force_law: FreeBotAngularForceLaw,
) -> dict[str, MagneticInteraction]:
    """Evaluate the published FreeBOT external magnetic model.

    Coordinate frame
    ----------------
    ``n`` points from the active sphere centre to the passive sphere centre.
    ``theta`` is the lifting angle between the active magnet's outward axis
    and ``n`` (the angle used in Figs. 5 and 6).  ``t`` is the magnet-axis
    projection perpendicular to ``n``.  Published components form

        F_active = A_parallel(d, theta) n - A_perp(d, theta) t.

    The minus sign follows the arrows in Fig. 5: the transverse component
    pulls the displaced active pole back toward the module contact line.

    Distance dependence
    -------------------
    Fig. 4 uses the *shell-to-shell* distance d and already includes the
    fixed magnet/inner-shell gap and shell thickness of the ANSYS assembly.
    Figs. 5 and 6 are angular cuts at contact.  Since the paper does not
    publish the full two-dimensional FEM surface, both angular components
    are multiplied by A_Fig4(d)/A_Fig4(0).  This separability is the one
    explicit modelling assumption in this function.

    Finite magnetic zone
    --------------------
    There is deliberately no infinitesimal ray activation gate.  Fig. 5
    remains nonzero away from exact alignment.  The direct magnet-to-magnet
    Fig. 6 branch is temporarily disabled: all external load reacts on the
    passive ferromagnetic shell through the Fig. 5 branch.  This avoids the
    close-pole repulsion while preserving the docking attraction.
    """
    centre_delta = passive_shell.center_world - active_shell.center_world
    centre_distance = float(np.linalg.norm(centre_delta))
    if centre_distance <= EPS:
        return {}
    n = centre_delta / centre_distance
    active_axis = normalized(active_magnet.axis_world)
    cosine = float(np.clip(np.dot(active_axis, n), -1.0, 1.0))
    theta_deg = float(np.degrees(np.arccos(cosine)))

    # Figs. 5--6 stop at 90 degrees.  A hard cut there is numerically and
    # physically inappropriate for a finite pole: the CAD pose can fluctuate
    # slightly across 90 degrees.  Continue only over the pole's own angular
    # half-width asin(radius/inner_radius), then decay smoothly to zero.  No
    # force is extrapolated farther onto the unpublished rear hemisphere.
    published_theta_deg = min(theta_deg, 90.0)
    pole_half_angle_deg = float(
        np.degrees(np.arcsin(np.clip(active_magnet.radius / active_shell.inner_radius, 0.0, 1.0)))
    )
    rear_extent = max(pole_half_angle_deg, EPS)
    rear_weight = _smoothstep01(1.0 - max(theta_deg - 90.0, 0.0) / rear_extent)
    if rear_weight <= EPS:
        return {}
    tangent_component = active_axis - cosine * n
    tangent_norm = float(np.linalg.norm(tangent_component))
    t = tangent_component / tangent_norm if tangent_norm > EPS else np.zeros(3, dtype=np.float64)

    aligned_at_gap = float(gap_force_law.force_n(max(float(shell_gap), 0.0), 1.0))
    aligned_at_contact = float(gap_force_law.force_n(0.0, 1.0))
    gap_scale = rear_weight * aligned_at_gap / aligned_at_contact if aligned_at_contact > EPS else 0.0
    if gap_scale <= EPS:
        return {}

    shell_parallel, shell_perpendicular = angular_force_law.shell_components_n(published_theta_deg)
    shell_parallel *= gap_scale
    shell_perpendicular *= gap_scale
    # Fig. 6 magnet-to-magnet coupling is intentionally disabled for the
    # current docking experiments.  Do not attenuate Fig. 5 when the passive
    # pole approaches the contact patch, otherwise attraction would vanish.
    close_magnet_weight = 0.0
    shell_weight = 1.0

    interactions: dict[str, MagneticInteraction] = {}
    if shell_weight > EPS:
        parallel = shell_weight * shell_parallel
        perpendicular = shell_weight * shell_perpendicular
        force = parallel * n - perpendicular * t
        if float(np.linalg.norm(force)) > EPS:
            # The resultant FEM force defines the centre of the finite
            # magnetized patch.  A ray is used only to place the reaction on
            # the passive surface, never to switch the force on or off.
            patch = MagneticGeometry.ray_sphere_intersection(
                active_magnet.face_center_world,
                force,
                passive_shell.center_world,
                passive_shell.outer_radius,
            )
            if patch is None:
                patch = passive_shell.center_world - passive_shell.outer_radius * n
            interactions["active_field_to_passive_shell"] = _paper_force_interaction(
                active_magnet,
                patch,
                force,
                shell_gap,
                n,
                theta_deg,
                parallel,
                perpendicular,
                shell_weight,
                "paper-fig4-fig5-shell-patch",
            )

    return interactions


def _validate_no_fixed_or_inter_module_joints(stage: Any) -> None:
    from pxr import Usd, UsdPhysics

    active_prefix = f"{ACTIVE_ROOT}/"
    target_prefix = f"{TARGET_ROOT}/"
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World")):
        if prim.IsA(UsdPhysics.FixedJoint):
            raise RuntimeError(f"Fixed joint is forbidden in this scenario: {prim.GetPath()}")
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        targets = [str(path) for path in joint.GetBody0Rel().GetTargets() + joint.GetBody1Rel().GetTargets()]
        touches_active = any(path.startswith(active_prefix) for path in targets)
        touches_target = any(path.startswith(target_prefix) for path in targets)
        if touches_active and touches_target:
            raise RuntimeError(f"Inter-module joint is forbidden: {prim.GetPath()}")


def _set_wheel_drive(stage: Any, root: str, left: float, right: float, damping: float, max_force: float) -> None:
    for side, velocity in (("left", left), ("right", right)):
        path = f"{root}/joints/{side}_wheel_joint"
        joint = stage.GetPrimAtPath(path)
        if not joint:
            raise RuntimeError(f"Missing wheel joint: {path}")
        joint.GetAttribute("physics:axis").Set("Y")
        joint.GetAttribute("drive:angular:physics:targetVelocity").Set(float(velocity))
        joint.GetAttribute("drive:angular:physics:damping").Set(float(damping))
        joint.GetAttribute("drive:angular:physics:maxForce").Set(float(max_force))


def _create_environment(stage: Any, config: SimulationConfig) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    if stage.GetPrimAtPath("/World/ground"):
        stage.RemovePrim("/World/ground")

    def material(path: str, static: float, dynamic: float) -> Any:
        item = UsdShade.Material.Define(stage, path)
        physics = UsdPhysics.MaterialAPI.Apply(item.GetPrim())
        physics.CreateStaticFrictionAttr(static)
        physics.CreateDynamicFrictionAttr(dynamic)
        physics.CreateRestitutionAttr(0.0)
        # ``max`` makes the authored rough-surface coefficient the effective
        # value instead of averaging it down against the shell material.
        item.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("max")
        return item

    ground_material = material(
        "/World/materials/emergent_ground",
        config.ground_static_friction,
        config.ground_dynamic_friction,
    )
    platform_material = material(
        "/World/materials/emergent_platform",
        config.ground_static_friction,
        config.ground_dynamic_friction,
    )

    def static_box(path: str, center: tuple[float, float, float], size: tuple[float, float, float], color: tuple[float, float, float], physics_material: Any) -> None:
        box = UsdGeom.Cube.Define(stage, path)
        box.CreateSizeAttr(1.0)
        box.AddTranslateOp().Set(Gf.Vec3d(*center))
        box.AddScaleOp().Set(Gf.Vec3f(*size))
        box.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(physics_material)

    static_box("/World/emergent_ground", (0.30, 0.06, -0.01), (1.8, 0.9, 0.02), (0.55, 0.55, 0.55), ground_material)
    static_box("/World/nonferromagnetic_platform", config.platform_center, config.platform_size, (0.42, 0.42, 0.44), platform_material)


def _assign_module_materials(stage: Any, config: SimulationConfig) -> None:
    """Bind explicit PhysX materials to wheel, caster and shell colliders."""
    from pxr import Sdf, Usd, UsdPhysics, UsdShade

    def material(path: str, static: float, dynamic: float, combine_mode: str = "max") -> Any:
        item = UsdShade.Material.Define(stage, path)
        api = UsdPhysics.MaterialAPI.Apply(item.GetPrim())
        api.CreateStaticFrictionAttr(static)
        api.CreateDynamicFrictionAttr(dynamic)
        api.CreateRestitutionAttr(0.0)
        item.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set(combine_mode)
        return item

    wheel_material = material(
        "/World/materials/freebot_wheel_shell",
        config.wheel_shell_static_friction,
        config.wheel_shell_dynamic_friction,
    )
    shell_material = material(
        "/World/materials/freebot_shell_shell",
        config.shell_shell_static_friction,
        config.shell_shell_dynamic_friction,
    )
    caster_material = material(
        "/World/materials/freebot_caster_shell",
        config.caster_shell_static_friction,
        config.caster_shell_dynamic_friction,
        "min",
    )
    for root in (ACTIVE_ROOT, TARGET_ROOT):
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            path = str(prim.GetPath())
            selected = wheel_material if ("left_wheel_link" in path or "right_wheel_link" in path) else None
            if "caster_1_ball_link" in path or "caster_2_ball_link" in path:
                selected = caster_material
            if "/shell_link/" in path or path.endswith("/shell_link"):
                selected = shell_material
            if selected is not None:
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(selected)


def _module_handles(root: str, rigid_prim_type: Any, xform_prim_type: Any) -> FreeBotModule:
    return FreeBotModule(
        root_path=root,
        shell_body=rigid_prim_type(paths=f"{root}/shell_link"),
        internal_body=rigid_prim_type(paths=f"{root}/internal_link"),
        magnet_frame=xform_prim_type(paths=f"{root}/internal_link/magnet_frame"),
        left_wheel_body=rigid_prim_type(paths=f"{root}/left_wheel_link"),
        right_wheel_body=rigid_prim_type(paths=f"{root}/right_wheel_link"),
        caster_1_body=rigid_prim_type(paths=f"{root}/caster_1_ball_link"),
        caster_2_body=rigid_prim_type(paths=f"{root}/caster_2_ball_link"),
    )


def run(config: SimulationConfig, headless: bool) -> None:
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": headless})
    teleop: Ros2CliTeleop | None = None
    try:
        import isaacsim.core.experimental.utils.app as app_utils
        import isaacsim.core.experimental.utils.stage as stage_utils
        from isaacsim.core.experimental.prims import RigidPrim, XformPrim
        from isaacsim.core.rendering_manager import RenderingManager, ViewportManager
        from isaacsim.core.simulation_manager import SimulationManager

        usd_path = config.usd_path.resolve()
        if not usd_path.exists():
            raise FileNotFoundError(usd_path)
        success, stage = stage_utils.open_stage(str(usd_path))
        if not success:
            raise RuntimeError(f"Could not open USD stage: {usd_path}")

        initial_shell_gap = _validate_initial_conditions(config)
        _clone_module(stage, SOURCE_ROOT, ACTIVE_ROOT)
        _clone_module(stage, SOURCE_ROOT, TARGET_ROOT)
        stage.RemovePrim(SOURCE_ROOT)
        removed_colliders = (
            _remove_duplicate_container_colliders(stage, ACTIVE_ROOT)
            + _remove_duplicate_container_colliders(stage, TARGET_ROOT)
        )
        replaced_wheel_sdf_colliders = (
            _replace_wheel_sdf_with_cylinder_proxies(stage, ACTIVE_ROOT, config.geometry)
            + _replace_wheel_sdf_with_cylinder_proxies(stage, TARGET_ROOT, config.geometry)
        )
        replaced_caster_sdf_colliders = (
            _replace_caster_sdf_with_sphere_proxies(
                stage, ACTIVE_ROOT, config.geometry, config.caster_colliders_enabled
            )
            + _replace_caster_sdf_with_sphere_proxies(
                stage, TARGET_ROOT, config.geometry, config.caster_colliders_enabled
            )
        )
        _make_all_rigid_bodies_dynamic(stage, ACTIVE_ROOT)
        _make_all_rigid_bodies_dynamic(stage, TARGET_ROOT)
        _position_module(stage, ACTIVE_ROOT, config.active_start, config.active_y_rotation_deg)
        _position_module(stage, TARGET_ROOT, config.target_start, config.target_y_rotation_deg)
        if config.internal_climb_test:
            # Test-only fixture: the shell is kinematic and external magnetic
            # interactions are disabled.  The passive clone remains outside
            # the 30 mm field range so the normal runtime bindings stay valid.
            from pxr import UsdPhysics
            UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(f"{ACTIVE_ROOT}/shell_link")).CreateKinematicEnabledAttr(True)
        _validate_no_fixed_or_inter_module_joints(stage)
        _create_environment(stage, config)
        _assign_module_materials(stage, config)
        drive_test = config.rolling_test or config.internal_climb_test
        requested_left = config.rolling_wheel_velocity_deg_s if drive_test else config.left_wheel_velocity_deg_s
        requested_right = config.rolling_wheel_velocity_deg_s if drive_test else config.right_wheel_velocity_deg_s
        initial_left = 0.0 if config.ros2_teleop else requested_left
        initial_right = 0.0 if config.ros2_teleop else requested_right
        _set_wheel_drive(
            stage,
            ACTIVE_ROOT,
            initial_left,
            initial_right,
            config.wheel_damping,
            config.wheel_max_force,
        )
        # The external module is fully dynamic but its motors are unpowered.
        _set_wheel_drive(stage, TARGET_ROOT, 0.0, 0.0, 0.0, 0.0)

        for _ in range(5):
            simulation_app.update()
        active = _module_handles(ACTIVE_ROOT, RigidPrim, XformPrim)
        target = _module_handles(TARGET_ROOT, RigidPrim, XformPrim)
        contact_diagnostics = None
        if config.contact_debug:
            contact_pairs = [
                ("left_wheel-active_shell", f"{ACTIVE_ROOT}/left_wheel_link", f"{ACTIVE_ROOT}/shell_link"),
                ("right_wheel-active_shell", f"{ACTIVE_ROOT}/right_wheel_link", f"{ACTIVE_ROOT}/shell_link"),
                ("active_shell-target_shell", f"{ACTIVE_ROOT}/shell_link", f"{TARGET_ROOT}/shell_link"),
                ("active_shell-ground", f"{ACTIVE_ROOT}/shell_link", "/World/emergent_ground"),
                ("target_shell-platform", f"{TARGET_ROOT}/shell_link", "/World/nonferromagnetic_platform"),
            ]
            if config.caster_colliders_enabled:
                contact_pairs[2:2] = [
                    ("caster_1-active_shell", f"{ACTIVE_ROOT}/caster_1_ball_link", f"{ACTIVE_ROOT}/shell_link"),
                    ("caster_2-active_shell", f"{ACTIVE_ROOT}/caster_2_ball_link", f"{ACTIVE_ROOT}/shell_link"),
                ]
            contact_diagnostics = PhysxContactDiagnostics(RigidPrim, contact_pairs)
        internal_patch_model = InternalPatchMagneticModel(config.magnet)
        external_gap_force_law = FreeBotFemForceLaw(alignment_power=1.0)
        external_angular_force_law = FreeBotAngularForceLaw()
        applier = ForcePairApplier()
        debug = DebugLines(stage, config.debug_draw, config.debug_force_scale_m_per_n)

        dt = 1.0 / config.physics_hz
        SimulationManager.set_physics_dt(dt)
        app_utils.play()
        simulation_app.update()
        if not headless:
            midpoint = 0.5 * (vector3(config.active_start) + vector3(config.target_start))
            ViewportManager.set_camera_view(
                "/OmniverseKit_Persp",
                eye=[float(midpoint[0]), -0.38, 0.24],
                target=[float(midpoint[0]), float(midpoint[1]), 0.060],
            )
        print("Running force-driven FreeBOT emergent docking")
        print(f"USD={usd_path}")
        print(f"duplicate_CAD_container_colliders_removed={removed_colliders} TGS_velocity_iterations=4")
        print(
            f"wheel_SDF_colliders_removed={replaced_wheel_sdf_colliders} "
            f"wheel_cylinder_proxies=4 diameter={2e3*config.geometry.tire_outer_radius_m:.3f}mm "
            f"width={2e3*config.geometry.tire_half_width_m:.3f}mm "
            f"CAD_axial_center_offset={1e3*config.geometry.tire_center_axial_offset_m:.3f}mm "
            f"axis_tilt={config.geometry.tire_axis_tilt_deg:.3f}deg"
        )
        print(
            f"caster_SDF_colliders_removed={replaced_caster_sdf_colliders} "
            f"caster_sphere_proxies={4 if config.caster_colliders_enabled else 0} "
            f"diameter={2e3*config.geometry.caster_ball_radius_m:.3f}mm "
            f"CAD_fitted_diameter=9.300mm "
            f"friction={config.caster_shell_static_friction:.2f}/{config.caster_shell_dynamic_friction:.2f}"
        )
        print(f"PhysX_contact_diagnostics={'on' if config.contact_debug else 'off'}")
        print(
            f"test={'rolling' if config.rolling_test else 'static'} initial_shell_gap={1e3*initial_shell_gap:.2f}mm "
            "modules=fully_dynamic upright_CAD_pose fixed_joint=none"
        )
        print("latch=off preload=off paper_transverse_magnetics=on force_cap=off magnetic_damping=off")
        print(
            "internal_magnet_model=equivalent-dipole+spherical-patch Maxwell pressure; "
            f"Br={config.magnet.remanence_t:.3f}T R={1e3*config.magnet.radius_m:.1f}mm "
            f"L={1e3*config.magnet.length_m:.1f}mm pressure_scale={config.magnet.internal_pressure_scale:.3f}; "
            f"patch={config.magnet.internal_patch_rings}x{config.magnet.internal_patch_samples_per_ring} "
            f"half_angle={config.magnet.internal_patch_half_angle_deg:.1f}deg; "
            "saturation=omitted hysteresis=omitted finite_permeability=calibration_parameter"
        )
        per_dipole_moment = internal_patch_model.magnetic_moment_magnitude / config.magnet.axial_dipole_count
        print(
            f"distributed_dipoles={config.magnet.axial_dipole_count} "
            f"span_fraction={config.magnet.axial_dipole_span_fraction:.3f} "
            f"total_moment={internal_patch_model.magnetic_moment_magnitude:.9f}Am2 "
            f"summed_moment={per_dipole_moment*config.magnet.axial_dipole_count:.9f}Am2"
        )
        print(
            "CAD_geometry: shell_center=fitted_concentric_surface_center "
            f"source_magnet_face_gap={1e3*source_magnet_face_to_inner_shell_gap_m(config.geometry, 0.5*config.magnet.length_m):.2f}mm"
        )
        if config.external_force_model == "exponential":
            print(
                "external_magnet_model=continuous_exponential; "
                f"F0=22.6N distance_scale={1e3*config.external_exponential_distance_scale_m:.2f}mm "
                f"angle_scale={config.external_exponential_angle_scale_deg:.1f}deg "
                f"angle_power={config.external_exponential_angle_power:.2f} "
                "Fig5_direction_only Fig6_disabled reaction=passive_shell "
                f"CAD_contact_gap_offset={1e3*config.geometry.shell_gap_contact_offset_m:.2f}mm"
            )
        else:
            print(
                "external_magnet_model=FreeBOT digitized FEM Figs.4-5; "
                "distance=Fig4_shell_gap angular_components=Fig5_Aparallel+Aperpendicular "
                "close_magnets=Fig6_disabled reaction=passive_shell F_aligned_contact=22.6N "
                f"CAD_contact_gap_offset={1e3*config.geometry.shell_gap_contact_offset_m:.2f}mm"
            )
        if config.debug_draw:
            print(
                "debug: yellow=magnet_axis white=gap magenta=normal "
                f"green=force orange=reaction force_scale={config.debug_force_scale_m_per_n:.4f}m/N"
            )
        if config.ros2_teleop:
            teleop = Ros2CliTeleop(config.cmd_vel_topic, config.cmd_timeout_s)
            print(
                f"ROS 2 teleop=on topic={config.cmd_vel_topic} timeout={config.cmd_timeout_s:.2f}s "
                f"linear_scale={config.cmd_linear_scale:.1f} angular_scale={config.cmd_angular_scale:.1f}"
            )
        else:
            print(f"ROS 2 teleop=off fixed_wheel_command=({initial_left:+.1f},{initial_right:+.1f})deg/s")

        commanded_left = initial_left
        commanded_right = initial_right
        teleop_forward = 0.0
        teleop_turn = 0.0
        color_enabled = config.color_logs and sys.stdout.isatty()

        for step in range(config.steps):
            if teleop is not None:
                linear_x, angular_z = teleop.command()
                commanded_left, commanded_right, teleop_forward, teleop_turn = twist_to_wheel_velocities(
                    linear_x,
                    angular_z,
                    config,
                )
                _set_wheel_drive(
                    stage,
                    ACTIVE_ROOT,
                    commanded_left,
                    commanded_right,
                    config.wheel_damping,
                    config.wheel_max_force,
                )

            active_magnet = active.magnet_state(config.magnet)
            target_magnet = target.magnet_state(config.magnet)
            active_shell = active.shell_state(config.geometry)
            target_shell = target.shell_state(config.geometry)
            module_shell_gap = spherical_shell_gap(active_shell, target_shell)
            magnetic_shell_gap = magnetic_shell_gap_from_cad_gap(
                module_shell_gap,
                config.geometry.shell_gap_contact_offset_m,
            )
            active_internal_com = active.internal_com_world()
            target_internal_com = target.internal_com_world()
            active_patch = internal_patch_model.compute(active_magnet, active_shell, active_internal_com)
            target_patch = internal_patch_model.compute(target_magnet, target_shell, target_internal_com)
            active_internal_force_norm = float(np.linalg.norm(active_patch.total_force_on_internal_world))
            active_internal_torque_norm = float(np.linalg.norm(active_patch.total_torque_on_internal_world))
            if active_internal_force_norm > 50.0 or active_internal_torque_norm > 1.0:
                print(colored(
                    f"WARNING excessive internal magnetic load before application: "
                    f"force={active_internal_force_norm:.3f}N torque={active_internal_torque_norm:.3f}Nm",
                    ANSI_RED,
                    color_enabled,
                ))
            if config.internal_climb_test:
                external_interactions = {}
            elif config.external_force_model == "exponential":
                external_interactions = compute_exponential_external_interactions(
                    active_magnet,
                    active_shell,
                    target_shell,
                    magnetic_shell_gap,
                    external_angular_force_law,
                    config.external_exponential_distance_scale_m,
                    config.external_exponential_angle_scale_deg,
                    config.external_exponential_angle_power,
                )
            else:
                external_interactions = compute_paper_external_interactions(
                    active_magnet,
                    target_magnet,
                    active_shell,
                    target_shell,
                    magnetic_shell_gap,
                    external_gap_force_law,
                    external_angular_force_law,
                )

            # Internal loads attach each mechanism to its own inner shell.
            # The paper's external field has two possible reaction bodies:
            # Fig. 5 magnetizes the passive shell, while Fig. 6 is the direct
            # close-pole interaction and reacts on the passive internal body.
            interaction_bindings = [
                *[(
                    "active_own_inner_patch",
                    interaction,
                    active.internal_body,
                    active.shell_body,
                    active_internal_com,
                    active_shell.com_world,
                ) for interaction in active_patch.interactions],
                (
                    "active_field_to_passive_shell",
                    external_interactions.get("active_field_to_passive_shell"),
                    active.internal_body,
                    target.shell_body,
                    active_internal_com,
                    target_shell.com_world,
                ),
                *[(
                    "target_own_inner_patch",
                    interaction,
                    target.internal_body,
                    target.shell_body,
                    target_internal_com,
                    target_shell.com_world,
                ) for interaction in target_patch.interactions],
            ]
            applied: list[tuple[str, MagneticInteraction]] = []
            global_force_terms: list[np.ndarray] = []
            for name, interaction, magnet_body, shell_body, _, _ in interaction_bindings:
                if interaction is None:
                    continue
                applier.apply(interaction, magnet_body, shell_body)
                applied.append((name, interaction))
                global_force_terms.extend((interaction.force_on_magnet_world, interaction.force_on_shell_world))

            global_force_residual = (
                np.sum(np.stack(global_force_terms), axis=0) if global_force_terms else np.zeros(3, dtype=np.float64)
            )
            if float(np.linalg.norm(global_force_residual)) > 1.0e-9:
                raise RuntimeError(f"Global magnetic force residual is {global_force_residual}")
            global_torque_terms = []
            for _, interaction, _, _, _, _ in interaction_bindings:
                if interaction is not None:
                    global_torque_terms.extend((
                        np.cross(interaction.magnet_application_point_world, interaction.force_on_magnet_world),
                        np.cross(interaction.shell_application_point_world, interaction.force_on_shell_world),
                    ))
            global_torque_residual = (
                np.sum(np.stack(global_torque_terms), axis=0)
                if global_torque_terms else np.zeros(3, dtype=np.float64)
            )

            debug.update([active_magnet, target_magnet], applied)
            SimulationManager.step()
            RenderingManager.render()
            simulation_app.update()

            if config.log_interval > 0 and step % config.log_interval == 0:
                active_center = active_shell.center_world
                target_center = target_shell.center_world
                active_face_radial = active_magnet.face_center_world - active_shell.center_world
                active_face_radius = float(np.linalg.norm(active_face_radial))
                active_axis_radial = (
                    float(np.dot(active_magnet.axis_world, active_face_radial / active_face_radius))
                    if active_face_radius > EPS
                    else 0.0
                )
                active_shell_angular = first_vector(active.shell_body.get_velocities()[1])
                axis_forward, axis_miss = MagneticGeometry.ray_sphere_diagnostics(
                    active_magnet.face_center_world,
                    active_magnet.axis_world,
                    target_shell.center_world,
                )
                paper_items = list(external_interactions.values())
                paper_theta = (
                    paper_items[0].paper_theta_deg
                    if paper_items
                    else float(np.degrees(np.arccos(np.clip(np.dot(active_magnet.axis_world, normalized(target_center - active_center)), -1.0, 1.0))))
                )
                paper_parallel = float(sum(item.parallel_force_n for item in paper_items))
                paper_perpendicular = float(sum(item.perpendicular_force_n for item in paper_items))
                paper_force_vector = (
                    np.sum(np.stack([item.force_on_magnet_world for item in paper_items]), axis=0)
                    if paper_items
                    else np.zeros(3, dtype=np.float64)
                )
                paper_force_magnitude = float(np.linalg.norm(paper_force_vector))
                passive_near_point = target_shell.center_world - target_shell.outer_radius * normalized(
                    target_shell.center_world - active_magnet.face_center_world
                )
                magnet_to_passive_gap = float(np.linalg.norm(passive_near_point - active_magnet.face_center_world))
                active_linear = first_vector(active.shell_body.get_velocities()[0])
                internal_linear = first_vector(active.internal_body.get_velocities()[0])
                left_slip = active.wheel_slip_diagnostic(active.left_wheel_body, active_shell, config.geometry)
                right_slip = active.wheel_slip_diagnostic(active.right_wheel_body, active_shell, config.geometry)
                gravity_force = config.module_mass_kg * 9.81
                internal_com_radius = float(np.linalg.norm(active_internal_com - active_center))
                paper_mu2 = paper_required_ground_friction(
                    paper_theta,
                    paper_parallel,
                    paper_perpendicular,
                    gravity_force,
                    active_shell.outer_radius,
                    internal_com_radius,
                    config.shell_shell_static_friction,
                )
                paper_mu_upper = paper_required_connection_friction(
                    paper_theta, paper_force_magnitude, gravity_force, False
                )
                paper_mu_lower = paper_required_connection_friction(
                    paper_theta, paper_force_magnitude, gravity_force, True
                )
                print(
                    f"t={step*dt:6.2f}s "
                    f"active=({active_center[0]:+.3f},{active_center[1]:+.3f},{active_center[2]:+.3f}) "
                    f"target=({target_center[0]:+.3f},{target_center[1]:+.3f},{target_center[2]:+.3f}) "
                    f"shell_gap={1e3*module_shell_gap:.2f}mm "
                    f"magnetic_shell_gap={1e3*magnetic_shell_gap:.2f}mm "
                    f"magnet_to_passive_surface_gap={1e3*magnet_to_passive_gap:.2f}mm "
                    f"internal_gap={1e3*active_patch.internal_gap_m:.2f}mm "
                    f"internal_patch_force={np.linalg.norm(active_patch.total_force_on_internal_world):.3f}N "
                    f"internal_patch_torque={np.linalg.norm(active_patch.total_torque_on_internal_world):.3e}Nm "
                    f"internal_pressure_peak={active_patch.peak_pressure_pa:.3e}Pa "
                    f"peak_unclamped_pressure={active_patch.unclamped_peak_pressure_pa:.3e}Pa "
                    f"pressure_clamp_count={active_patch.pressure_clamp_count} "
                    f"pressure_clamped={active_patch.pressure_clamp_count > 0} "
                    f"minimum_dipole_sample_distance={1e3*active_patch.minimum_dipole_sample_distance_m:.3f}mm "
                    f"axial_dipole_count={config.magnet.axial_dipole_count} "
                    f"internal_B_peak={active_patch.peak_field_t:.3e}T "
                    f"global_magnetic_force_residual={np.linalg.norm(global_force_residual):.3e}N "
                    f"global_magnetic_torque_residual={np.linalg.norm(global_torque_residual):.3e}Nm "
                    f"wheel_cmd=({commanded_left:+.1f},{commanded_right:+.1f})deg/s "
                    f"wheel_actual=({active.wheel_spin_deg_s(active.left_wheel_body):+.1f},"
                    f"{active.wheel_spin_deg_s(active.right_wheel_body):+.1f})deg/s "
                    f"wheel_inner_clearance=({1e3*active.wheel_inner_shell_clearance_m(active.left_wheel_body, config.geometry):+.2f},"
                    f"{1e3*active.wheel_inner_shell_clearance_m(active.right_wheel_body, config.geometry):+.2f})mm "
                    f"wheel_surface_speed=({left_slip.wheel_surface_speed_m_s:+.3f},{right_slip.wheel_surface_speed_m_s:+.3f})m/s "
                    f"estimated_contact_relative_tangent_speed=({left_slip.estimated_contact_relative_tangent_speed_m_s:+.3f},"
                    f"{right_slip.estimated_contact_relative_tangent_speed_m_s:+.3f})m/s "
                    f"slip_speed=({left_slip.slip_speed_m_s:+.3f},{right_slip.slip_speed_m_s:+.3f})m/s "
                    f"slip_ratio=({left_slip.slip_ratio:+.3f},{right_slip.slip_ratio:+.3f}) "
                    f"wheel_contact_estimate=({left_slip.estimated_contact},{right_slip.estimated_contact}) "
                    f"slip_clearance=({1e3*left_slip.clearance_m:+.2f},{1e3*right_slip.clearance_m:+.2f})mm "
                    f"caster_inner_clearance=({1e3*active.caster_inner_shell_clearance_m(active.caster_1_body, config.geometry):+.2f},"
                    f"{1e3*active.caster_inner_shell_clearance_m(active.caster_2_body, config.geometry):+.2f})mm "
                    f"shell_omega=({active_shell_angular[0]:+.2f},{active_shell_angular[1]:+.2f},"
                    f"{active_shell_angular[2]:+.2f})rad/s "
                    f"internal_velocity=({internal_linear[0]:+.3f},{internal_linear[1]:+.3f},{internal_linear[2]:+.3f})m/s "
                    f"active_shell_velocity=({active_linear[0]:+.3f},{active_linear[1]:+.3f},{active_linear[2]:+.3f})m/s "
                    f"mechanism_angle={active.mechanism_angle_deg(config.geometry):+.1f}deg "
                    f"pole_radius={1e3*active_face_radius:.1f}mm "
                    f"axis_radial={active_axis_radial:+.3f} "
                    f"axis_forward={1e3*axis_forward:+.1f}mm "
                    f"axis_miss={1e3*axis_miss:.1f}mm/{1e3*target_shell.outer_radius:.1f}mm "
                    f"paper_force=(Apar={paper_parallel:+.2f},Aperp={paper_perpendicular:+.2f})N "
                    f"paper_mu_required=(ground={paper_mu2:.3f},upper={paper_mu_upper:.3f},lower={paper_mu_lower:.3f}) "
                    f"mu_available=(ground={config.ground_static_friction:.2f}/{config.ground_dynamic_friction:.2f},"
                    f"shell={config.shell_shell_static_friction:.2f}/{config.shell_shell_dynamic_friction:.2f}) "
                    f"cmd_scaled=({teleop_forward:+.1f},{teleop_turn:+.1f})"
                )
                print(
                    f"  internal_patch_rings: force_norms_N={active_patch.ring_force_norms_n} "
                    f"max_pressure_Pa={active_patch.ring_max_pressure_pa} "
                    f"sampled_area={active_patch.sampled_area_m2:.12e}m2 "
                    f"cap_area={active_patch.cap_area_m2:.12e}m2"
                )
                print(colored(
                    "  INTERNAL  "
                    f"Ftotal={active_internal_force_norm:.3f}N "
                    f"Fradial={active_patch.radial_force_on_internal_n:.3f}N "
                    f"Ftangent={active_patch.tangential_force_on_internal_n:.3f}N "
                    f"T={active_internal_torque_norm:.4f}Nm "
                    f"gap={1e3*active_patch.internal_gap_m:.2f}mm "
                    f"clamps={active_patch.pressure_clamp_count}",
                    ANSI_CYAN,
                    color_enabled,
                ))
                print(colored(
                    "  EXTERNAL  "
                    f"F={paper_force_magnitude:.3f}N "
                    f"theta={paper_theta:.2f}deg "
                    f"cad_gap={1e3*module_shell_gap:.2f}mm "
                    f"magnetic_gap={1e3*magnetic_shell_gap:.2f}mm "
                    f"magnet_surface_gap={1e3*magnet_to_passive_gap:.2f}mm",
                    ANSI_MAGENTA,
                    color_enabled,
                ))
                contacts_estimated = left_slip.estimated_contact and right_slip.estimated_contact
                wheel_color = ANSI_GREEN if contacts_estimated else ANSI_YELLOW
                print(colored(
                    "  WHEELS    "
                    f"cmd=({commanded_left:+.1f},{commanded_right:+.1f})deg/s "
                    f"slip=({left_slip.slip_ratio:+.2f},{right_slip.slip_ratio:+.2f}) "
                    f"clearance=({1e3*left_slip.clearance_m:+.2f},{1e3*right_slip.clearance_m:+.2f})mm "
                    f"estimated_contact=({left_slip.estimated_contact},{right_slip.estimated_contact})",
                    wheel_color,
                    color_enabled,
                ))
                for name, interaction, _, _, magnet_com, shell_com in interaction_bindings:
                    if "inner_patch" not in name:
                        print("  " + DiagnosticsLogger.format_interaction(name, interaction, magnet_com, shell_com))
                if contact_diagnostics is not None:
                    for contact_line in contact_diagnostics.formatted_lines(dt):
                        print(contact_line)
    finally:
        if teleop is not None:
            teleop.close()
        try:
            app_utils.stop()
        except (NameError, RuntimeError):
            pass
        simulation_app.close()


def parse_args() -> tuple[SimulationConfig, bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", type=Path, default=Path("assets/freebot/usd_physics/freebot_cad_full_nearer_wheels_rigid.usd"))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=12_000)
    parser.add_argument("--log-interval", type=int, default=240)
    parser.add_argument(
        "--initial-shell-gap",
        type=float,
        default=0.040,
        help="Initial nominal distance between spherical shells [m]. The FEM force is zero from 0.030 m onward.",
    )
    parser.add_argument("--left-wheel-velocity", type=float, default=0.0)
    parser.add_argument("--right-wheel-velocity", type=float, default=0.0)
    parser.add_argument("--rolling-test", action="store_true", help="Run the second validation case with both wheels driven.")
    parser.add_argument("--climbing-test", action="store_true", help="Alias for post-docking rolling/climbing with both wheels driven.")
    parser.add_argument("--internal-climb-test", action="store_true", help="Fixture the active shell, disable external magnetics and drive both wheels.")
    parser.add_argument("--rolling-wheel-velocity", type=float, default=360.0)
    parser.add_argument("--ros2-teleop", action="store_true")
    parser.add_argument("--cmd-vel-topic", default="/cmd_vel")
    parser.add_argument("--cmd-timeout", type=float, default=1.0)
    parser.add_argument("--cmd-linear-scale", type=float, default=900.0)
    parser.add_argument("--cmd-angular-scale", type=float, default=360.0)
    parser.add_argument("--cmd-linear-sign", type=float, default=1.0)
    parser.add_argument("--cmd-angular-sign", type=float, default=1.0)
    parser.add_argument("--debug-draw", action="store_true")
    parser.add_argument("--debug-force-scale", type=float, default=0.005, help="Displayed force-vector length in metres per newton.")
    parser.add_argument(
        "--contact-debug",
        action="store_true",
        help="Log PhysX normal/friction forces and contact points for wheels, casters, shells and ground.",
    )
    parser.add_argument("--magnet-br", type=float, default=1.47)
    parser.add_argument("--magnet-radius", type=float, default=0.010)
    parser.add_argument("--magnet-length", type=float, default=0.010)
    parser.add_argument("--alignment-power", type=float, default=2.0)
    parser.add_argument("--internal-patch-half-angle", type=float, default=35.0)
    parser.add_argument("--internal-patch-rings", type=int, default=5)
    parser.add_argument("--internal-patch-samples-per-ring", type=int, default=16)
    parser.add_argument("--minimum-field-distance", type=float, default=0.003)
    parser.add_argument("--maximum-sample-pressure", type=float, default=1.0e5)
    parser.add_argument("--internal-pressure-scale", type=float, default=1.0)
    parser.add_argument("--axial-dipole-count", type=int, default=7)
    parser.add_argument("--axial-dipole-span-fraction", type=float, default=0.80)
    parser.add_argument(
        "--shell-gap-contact-offset",
        type=float,
        default=0.0025,
        help="CAD nominal shell-gap reading at actual collider contact [m].",
    )
    parser.add_argument("--external-force-model", choices=("exponential", "paper"), default="exponential")
    parser.add_argument("--external-distance-scale", type=float, default=0.0035)
    parser.add_argument("--external-angle-scale", type=float, default=45.0)
    parser.add_argument("--external-angle-power", type=float, default=2.0)
    parser.add_argument("--no-color-logs", action="store_true")
    parser.add_argument("--wheel-shell-static-friction", type=float, default=2.20)
    parser.add_argument("--wheel-shell-dynamic-friction", type=float, default=1.90)
    parser.add_argument(
        "--caster-proxy-radius",
        type=float,
        default=GeometryConfig.caster_ball_radius_m,
        help="Physics-only caster sphere radius [m]; fitted CAD radius is 0.004650 m.",
    )
    parser.add_argument(
        "--disable-caster-colliders",
        action="store_true",
        help="Diagnostic only: remove caster collision without creating sphere proxies.",
    )
    parser.add_argument("--shell-shell-static-friction", type=float, default=1.10)
    parser.add_argument("--shell-shell-dynamic-friction", type=float, default=1.00)
    parser.add_argument("--wheel-max-torque", type=float, default=0.686)
    args = parser.parse_args()
    if not 0.0 <= args.initial_shell_gap <= 0.100:
        parser.error("--initial-shell-gap must lie in [0.0, 0.100] m")
    if not 0.0 <= args.shell_gap_contact_offset <= 0.010:
        parser.error("--shell-gap-contact-offset must lie in [0, 0.010] m")
    if not 0.001 <= args.caster_proxy_radius <= 0.010:
        parser.error("--caster-proxy-radius must lie in [0.001, 0.010] m")
    friction_values = (
        args.wheel_shell_static_friction,
        args.wheel_shell_dynamic_friction,
        args.shell_shell_static_friction,
        args.shell_shell_dynamic_friction,
    )
    if any(value < 0.0 for value in friction_values):
        parser.error("friction coefficients must be nonnegative")
    geometry = GeometryConfig(
        shell_gap_contact_offset_m=args.shell_gap_contact_offset,
        caster_ball_radius_m=args.caster_proxy_radius,
    )
    platform_left_edge_x = SimulationConfig.platform_center[0] - 0.5 * SimulationConfig.platform_size[0]
    target_start = (
        platform_left_edge_x - geometry.shell_outer_radius_m,
        SimulationConfig.target_start[1],
        SimulationConfig.target_start[2],
    )
    active_start = (
        target_start[0] - 2.0 * geometry.shell_outer_radius_m - args.initial_shell_gap,
        target_start[1],
        target_start[2],
    )
    config = SimulationConfig(
        usd_path=args.usd,
        steps=args.steps,
        log_interval=args.log_interval,
        active_start=active_start,
        target_start=target_start,
        left_wheel_velocity_deg_s=args.left_wheel_velocity,
        right_wheel_velocity_deg_s=args.right_wheel_velocity,
        rolling_test=args.rolling_test or args.climbing_test,
        internal_climb_test=args.internal_climb_test,
        rolling_wheel_velocity_deg_s=args.rolling_wheel_velocity,
        wheel_max_force=args.wheel_max_torque,
        external_force_model=args.external_force_model,
        external_exponential_distance_scale_m=args.external_distance_scale,
        external_exponential_angle_scale_deg=args.external_angle_scale,
        external_exponential_angle_power=args.external_angle_power,
        color_logs=not args.no_color_logs,
        wheel_shell_static_friction=args.wheel_shell_static_friction,
        wheel_shell_dynamic_friction=args.wheel_shell_dynamic_friction,
        shell_shell_static_friction=args.shell_shell_static_friction,
        shell_shell_dynamic_friction=args.shell_shell_dynamic_friction,
        ros2_teleop=args.ros2_teleop,
        cmd_vel_topic=args.cmd_vel_topic,
        cmd_timeout_s=args.cmd_timeout,
        cmd_linear_scale=args.cmd_linear_scale,
        cmd_angular_scale=args.cmd_angular_scale,
        cmd_linear_sign=args.cmd_linear_sign,
        cmd_angular_sign=args.cmd_angular_sign,
        debug_draw=args.debug_draw,
        debug_force_scale_m_per_n=args.debug_force_scale,
        contact_debug=args.contact_debug,
        caster_colliders_enabled=not args.disable_caster_colliders,
        geometry=geometry,
        magnet=MagnetConfig(
            remanence_t=args.magnet_br,
            radius_m=args.magnet_radius,
            length_m=args.magnet_length,
            alignment_power=args.alignment_power,
            minimum_field_distance_m=args.minimum_field_distance,
            internal_patch_half_angle_deg=args.internal_patch_half_angle,
            internal_patch_rings=args.internal_patch_rings,
            internal_patch_samples_per_ring=args.internal_patch_samples_per_ring,
            internal_pressure_scale=args.internal_pressure_scale,
            maximum_sample_pressure_pa=args.maximum_sample_pressure,
            axial_dipole_count=args.axial_dipole_count,
            axial_dipole_span_fraction=args.axial_dipole_span_fraction,
        ),
    )
    return config, args.headless


if __name__ == "__main__":
    simulation_config, is_headless = parse_args()
    run(simulation_config, is_headless)
