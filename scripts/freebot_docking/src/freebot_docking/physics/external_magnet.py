from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from freebot_docking.config.geometry import ShellGeometry
from freebot_docking.config.magnet import MagnetConfig
from freebot_docking.physics.geometry import (
    compute_magnet_active_face_center,
    compute_shell_pair_geometry,
)
from freebot_docking.physics.state import (
    MagnetState,
    ShellState,
    Vector3,
    as_vector3,
)
from freebot_docking.physics.wrench import Wrench


@dataclass(frozen=True)
class TabulatedAlignedForceCurve:
    """Monotone aligned attraction samples as a function of shell gap."""

    shell_gap_m: tuple[float, ...]
    force_n: tuple[float, ...]

    def __post_init__(self) -> None:
        gaps = tuple(float(value) for value in self.shell_gap_m)
        forces = tuple(float(value) for value in self.force_n)

        if len(gaps) != len(forces) or len(gaps) < 2:
            raise ValueError(
                "Force curve gaps and forces must have equal length"
            )
        if gaps[0] != 0.0:
            raise ValueError("Force curve must start at zero gap")
        if not all(math.isfinite(value) for value in gaps + forces):
            raise ValueError("Force curve samples must be finite")
        if any(value < 0.0 for value in gaps + forces):
            raise ValueError("Force curve samples must be non-negative")
        if any(
            right <= left
            for left, right in zip(gaps, gaps[1:])
        ):
            raise ValueError("Force curve gaps must increase strictly")
        if any(
            right > left
            for left, right in zip(forces, forces[1:])
        ):
            raise ValueError("Force curve must be non-increasing")

        object.__setattr__(self, "shell_gap_m", gaps)
        object.__setattr__(self, "force_n", forces)

    def attraction_force_n(self, shell_gap_m: float) -> float:
        """Interpolate without overshoot and return zero past capture."""

        gap = float(shell_gap_m)
        if not math.isfinite(gap) or gap < 0.0:
            raise ValueError("Shell gap must be finite and non-negative")

        return float(
            np.interp(
                gap,
                self.shell_gap_m,
                self.force_n,
                left=self.force_n[0],
                right=0.0,
            )
        )


def freebot_figure4_force_curve() -> TabulatedAlignedForceCurve:
    """Return the digitized FreeBOT Fig. 4 aligned ANSYS curve."""

    return TabulatedAlignedForceCurve(
        shell_gap_m=tuple(
            1.0e-3 * value
            for value in (
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0,
            )
        ),
        force_n=(
            22.6, 16.51, 12.42, 9.52, 7.14, 5.36, 4.13,
            2.34, 1.35, 0.79, 0.36, 0.12, 0.08, 0.0,
        ),
    )


@dataclass(frozen=True)
class AnchoredExponentialForceFit:
    """F(g) = F(0) exp(-g/lambda), identified from force samples."""

    contact_force_n: float
    decay_length_m: float
    root_mean_square_error_n: float
    maximum_absolute_error_n: float

    def attraction_force_n(self, shell_gap_m: float) -> float:
        gap = float(shell_gap_m)
        if not math.isfinite(gap) or gap < 0.0:
            raise ValueError("Shell gap must be finite and non-negative")
        return self.contact_force_n * math.exp(
            -gap / self.decay_length_m
        )


def fit_anchored_exponential_force_curve(
    curve: TabulatedAlignedForceCurve,
) -> AnchoredExponentialForceFit:
    """Fit the paper's stated exponential while preserving F(0)."""

    gaps = np.asarray(curve.shell_gap_m, dtype=np.float64)
    forces = np.asarray(curve.force_n, dtype=np.float64)
    contact_force = float(forces[0])
    fit_mask = (gaps > 0.0) & (forces > 0.0)
    fit_gaps = gaps[fit_mask]
    logarithmic_ratio = np.log(forces[fit_mask] / contact_force)
    inverse_decay_length = -float(
        np.dot(fit_gaps, logarithmic_ratio)
        / np.dot(fit_gaps, fit_gaps)
    )

    if (
        not math.isfinite(inverse_decay_length)
        or inverse_decay_length <= 0.0
    ):
        raise ValueError("Force samples do not define exponential decay")

    decay_length = 1.0 / inverse_decay_length
    residuals = contact_force * np.exp(-gaps / decay_length) - forces

    return AnchoredExponentialForceFit(
        contact_force_n=contact_force,
        decay_length_m=decay_length,
        root_mean_square_error_n=float(
            np.sqrt(np.mean(residuals**2))
        ),
        maximum_absolute_error_n=float(np.max(np.abs(residuals))),
    )


@dataclass(frozen=True)
class TabulatedAngularForceCurve:
    """FreeBOT Fig. 5 force components versus lifting angle.

    The published samples stop at 90 degrees. Values outside that calibrated
    interval are deliberately rejected rather than extrapolated.
    """

    angle_deg: tuple[float, ...]
    parallel_force_n: tuple[float, ...]
    perpendicular_force_n: tuple[float, ...]

    def __post_init__(self) -> None:
        angles = tuple(float(value) for value in self.angle_deg)
        parallel = tuple(float(value) for value in self.parallel_force_n)
        perpendicular = tuple(
            float(value) for value in self.perpendicular_force_n
        )

        if not (
            len(angles) == len(parallel) == len(perpendicular)
            and len(angles) >= 2
        ):
            raise ValueError("Angular curve arrays must have equal length")
        if angles[0] != 0.0 or angles[-1] != 90.0:
            raise ValueError("Angular curve must cover 0 through 90 degrees")
        if any(
            right <= left
            for left, right in zip(angles, angles[1:])
        ):
            raise ValueError("Angular samples must increase strictly")
        if not all(
            math.isfinite(value)
            for value in angles + parallel + perpendicular
        ):
            raise ValueError("Angular force samples must be finite")
        if any(value < 0.0 for value in parallel + perpendicular):
            raise ValueError("Figure 5 force components must be non-negative")

        object.__setattr__(self, "angle_deg", angles)
        object.__setattr__(self, "parallel_force_n", parallel)
        object.__setattr__(self, "perpendicular_force_n", perpendicular)

    def components_n(self, angle_deg: float) -> tuple[float, float]:
        angle = float(angle_deg)
        if not math.isfinite(angle) or not 0.0 <= angle <= 90.0:
            raise ValueError("Lifting angle must lie in [0, 90] degrees")

        return (
            float(np.interp(angle, self.angle_deg, self.parallel_force_n)),
            float(
                np.interp(
                    angle,
                    self.angle_deg,
                    self.perpendicular_force_n,
                )
            ),
        )

def freebot_figure5_angular_force_curve() -> TabulatedAngularForceCurve:
    """Return digitized FreeBOT Fig. 5 magnet-to-shell components."""

    return TabulatedAngularForceCurve(
        angle_deg=tuple(float(value) for value in range(0, 91, 5)),
        parallel_force_n=(
            22.6, 15.52, 11.89, 9.72, 7.64, 7.08, 6.70,
            6.32, 5.85, 5.28, 4.81, 4.34, 3.92, 3.49,
            3.11, 2.64, 2.36, 2.08, 1.75,
        ),
        perpendicular_force_n=(
            0.0, 0.154, 0.706, 1.245, 1.40, 1.227, 0.919,
            0.656, 0.474, 0.345, 0.257, 0.207, 0.169,
            0.151, 0.132, 0.125, 0.119, 0.113, 0.10,
        ),
    )


@dataclass(frozen=True)
class ExternalMagneticInteraction:
    """Reduced-order magnetic force pair calibrated on Figs. 4 and 5."""

    shell_gap_m: float
    lifting_angle_deg: float
    parallel_curve_angle_deg: float
    distance_scale: float
    parallel_force_n: float
    perpendicular_force_n: float
    force_on_active_world: Vector3
    force_on_passive_world: Vector3
    interaction_point_world: Vector3
    active_surface_point_world: Vector3
    active_surface_normal_world: Vector3
    passive_surface_point_world: Vector3
    passive_surface_normal_world: Vector3
    magnet_surface_gap_m: float
    active_carrier_wrench: Wrench
    active_shell_wrench: Wrench
    passive_shell_wrench: Wrench
    in_angular_range: bool
    line_of_action_valid: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "force_on_active_world",
            as_vector3(self.force_on_active_world),
        )
        object.__setattr__(
            self,
            "force_on_passive_world",
            as_vector3(self.force_on_passive_world),
        )
        object.__setattr__(
            self,
            "interaction_point_world",
            as_vector3(self.interaction_point_world),
        )
        object.__setattr__(
            self,
            "active_surface_point_world",
            as_vector3(self.active_surface_point_world),
        )
        object.__setattr__(
            self,
            "active_surface_normal_world",
            as_vector3(self.active_surface_normal_world),
        )
        object.__setattr__(
            self,
            "passive_surface_point_world",
            as_vector3(self.passive_surface_point_world),
        )
        object.__setattr__(
            self,
            "passive_surface_normal_world",
            as_vector3(self.passive_surface_normal_world),
        )


def _first_ray_sphere_intersection(
    ray_origin_world: Vector3,
    ray_direction_world: Vector3,
    sphere_center_world: Vector3,
    sphere_radius_m: float,
) -> Vector3 | None:
    """Return the first forward point where a unit ray enters a sphere."""

    origin = as_vector3(ray_origin_world)
    direction = as_vector3(ray_direction_world)
    center = as_vector3(sphere_center_world)
    radius = float(sphere_radius_m)
    direction_norm = float(np.linalg.norm(direction))
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("Sphere radius must be finite and positive")
    if direction_norm <= 1.0e-12:
        raise ValueError("Ray direction cannot be zero")
    direction = direction / direction_norm

    offset = origin - center
    projected_offset = float(np.dot(offset, direction))
    constant = float(np.dot(offset, offset) - radius * radius)
    discriminant = projected_offset * projected_offset - constant
    if discriminant < -1.0e-12:
        return None
    root = math.sqrt(max(discriminant, 0.0))
    forward_distances = tuple(
        distance
        for distance in (
            -projected_offset - root,
            -projected_offset + root,
        )
        if distance >= 0.0
    )
    if not forward_distances:
        return None
    return origin + min(forward_distances) * direction


def compute_external_magnetic_interaction(
    active_shell_state: ShellState,
    active_shell_geometry: ShellGeometry,
    passive_shell_state: ShellState,
    passive_shell_geometry: ShellGeometry,
    active_magnet_state: MagnetState,
    magnet_config: MagnetConfig,
    distance_curve: TabulatedAlignedForceCurve | None = None,
    angular_curve: TabulatedAngularForceCurve | None = None,
) -> ExternalMagneticInteraction:
    """Compute the conservative force pair for the calibrated external law."""

    distance_law = (
        freebot_figure4_force_curve()
        if distance_curve is None
        else distance_curve
    )
    angular_law = (
        freebot_figure5_angular_force_curve()
        if angular_curve is None
        else angular_curve
    )
    pair = compute_shell_pair_geometry(
        active_shell_state,
        active_shell_geometry,
        passive_shell_state,
        passive_shell_geometry,
    )
    face_center = compute_magnet_active_face_center(
        active_magnet_state,
        magnet_config,
    )
    active_patch_radial = face_center - active_shell_state.center_world
    active_patch_radial /= np.linalg.norm(active_patch_radial)
    active_surface_normal = active_patch_radial
    active_surface_point = (
        active_shell_state.center_world
        + active_shell_geometry.outer_radius_m * active_surface_normal
    )
    face_to_passive_center = (
        passive_shell_state.center_world - face_center
    )
    face_to_passive_distance = float(
        np.linalg.norm(face_to_passive_center)
    )
    if face_to_passive_distance > passive_shell_geometry.outer_radius_m:
        # Nearest-point geometry is retained as the zero-force/fallback
        # diagnostic. The active force below replaces it with the intersection
        # of the measured resultant line and the passive spherical surface.
        passive_surface_normal = (
            face_to_passive_center / face_to_passive_distance
        )
        passive_surface_point = (
            passive_shell_state.center_world
            - passive_shell_geometry.outer_radius_m
            * passive_surface_normal
        )
        magnet_surface_gap = (
            face_to_passive_distance
            - passive_shell_geometry.outer_radius_m
        )
    else:
        # This is only a penetration fallback.  The physical configuration
        # keeps the magnet face outside the passive shell.
        passive_surface_normal = pair.normal_first_to_second_world
        passive_surface_point = pair.point_on_second_world
        magnet_surface_gap = 0.0

    # Figure 5 and Eq. (5) resolve A_parallel and A_perpendicular in the
    # shell-shell contact frame.  With ``t`` defined below as the projection
    # of the active magnet axis away from the centre line, the arrows in
    # Fig. 5 give F_active = A_parallel*n - A_perpendicular*t.  In
    # particular, when the magnet is raised, the transverse term pulls it
    # back toward the shell-shell contact line and produces the positive
    # rolling moment described by the paper.  The moving passive patch above
    # describes where the field is concentrated, but it must not rotate the
    # published components into an uncalibrated local frame.
    normal = pair.normal_first_to_second_world
    alignment_cosine = float(
        np.clip(np.dot(active_magnet_state.axis_world, normal), -1.0, 1.0)
    )
    angle_deg = math.degrees(math.acos(alignment_cosine))
    parallel_curve_angle_deg = angle_deg
    in_angular_range = angle_deg <= 90.0
    effective_gap = max(0.0, pair.signed_gap_m)
    aligned_force = distance_law.attraction_force_n(effective_gap)
    contact_force = distance_law.force_n[0]
    distance_scale = (
        aligned_force / contact_force
        if contact_force > 0.0
        else 0.0
    )

    tangent_projection = (
        active_magnet_state.axis_world
        - alignment_cosine * normal
    )
    tangent_norm = float(np.linalg.norm(tangent_projection))
    tangent = (
        tangent_projection / tangent_norm
        if tangent_norm > 1.0e-12
        else np.zeros(3, dtype=np.float64)
    )

    if aligned_force > 0.0 and in_angular_range:
        contact_parallel, contact_perpendicular = angular_law.components_n(
            angle_deg
        )
        parallel_force = distance_scale * contact_parallel
        perpendicular_force = distance_scale * contact_perpendicular
        force_on_active = (
            parallel_force * normal
            - perpendicular_force * tangent
        )
    else:
        parallel_force = 0.0
        perpendicular_force = 0.0
        force_on_active = np.zeros(3, dtype=np.float64)

    # The measured Fig. 5 resultant defines a ray from the moving magnet face.
    # Its first intersection with the passive sphere is the corresponding
    # magnetized patch. Applying the opposite forces at the two endpoints of
    # this same line conserves both linear and angular momentum without adding
    # an unmeasured magnetic couple. If the measured direction misses the
    # passive sphere, the reduced interaction is outside its valid geometry.
    interaction_point = face_center
    force_norm = float(np.linalg.norm(force_on_active))
    line_of_action_valid = False
    if force_norm > 1.0e-12:
        active_intersection = _first_ray_sphere_intersection(
            interaction_point,
            force_on_active / force_norm,
            active_shell_state.center_world,
            active_shell_geometry.outer_radius_m,
        )
        passive_intersection = _first_ray_sphere_intersection(
            interaction_point,
            force_on_active / force_norm,
            passive_shell_state.center_world,
            passive_shell_geometry.outer_radius_m,
        )
        if active_intersection is None or passive_intersection is None:
            parallel_force = 0.0
            perpendicular_force = 0.0
            force_on_active = np.zeros(3, dtype=np.float64)
        else:
            active_surface_point = active_intersection
            active_surface_normal = (
                active_surface_point - active_shell_state.center_world
            )
            active_surface_normal /= np.linalg.norm(active_surface_normal)
            passive_surface_point = passive_intersection
            inward_normal = (
                passive_shell_state.center_world - passive_surface_point
            )
            passive_surface_normal = inward_normal / np.linalg.norm(
                inward_normal
            )
            magnet_surface_gap = float(
                np.linalg.norm(passive_surface_point - interaction_point)
            )
            line_of_action_valid = True
    force_on_passive = -force_on_active

    return ExternalMagneticInteraction(
        shell_gap_m=pair.signed_gap_m,
        lifting_angle_deg=angle_deg,
        parallel_curve_angle_deg=parallel_curve_angle_deg,
        distance_scale=distance_scale,
        parallel_force_n=parallel_force,
        perpendicular_force_n=perpendicular_force,
        force_on_active_world=force_on_active,
        force_on_passive_world=force_on_passive,
        interaction_point_world=interaction_point,
        active_surface_point_world=active_surface_point,
        active_surface_normal_world=active_surface_normal,
        passive_surface_point_world=passive_surface_point,
        passive_surface_normal_world=passive_surface_normal,
        magnet_surface_gap_m=magnet_surface_gap,
        active_carrier_wrench=Wrench.from_force_at_point(
            force_on_active,
            interaction_point,
            active_magnet_state.carrier_com_world,
        ),
        active_shell_wrench=Wrench.from_force_at_point(
            force_on_active,
            active_surface_point,
            active_shell_state.com_world,
        ),
        passive_shell_wrench=Wrench.from_force_at_point(
            force_on_passive,
            passive_surface_point,
            passive_shell_state.com_world,
        ),
        in_angular_range=in_angular_range,
        line_of_action_valid=line_of_action_valid,
    )
