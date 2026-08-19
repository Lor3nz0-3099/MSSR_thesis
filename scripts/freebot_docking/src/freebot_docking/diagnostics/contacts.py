from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from freebot_docking.physics.external_magnet import (
    ExternalMagneticInteraction,
)


_EPSILON = 1.0e-12


def paper_required_ground_friction(
    angle_deg: float,
    parallel_force_n: float,
    perpendicular_force_n: float,
    gravity_force_n: float,
    shell_radius_m: float,
    mechanism_com_radius_m: float,
    shell_shell_friction_coefficient: float,
) -> float:
    """Return the limiting ground coefficient from FreeBOT Eq. (7).

    Infinity denotes a singular or statically infeasible configuration.  This
    equation is a diagnostic balance and does not generate a contact force.
    """

    values = (
        angle_deg,
        parallel_force_n,
        perpendicular_force_n,
        gravity_force_n,
        shell_radius_m,
        mechanism_com_radius_m,
        shell_shell_friction_coefficient,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Friction-diagnostic inputs must be finite")
    if not 0.0 <= angle_deg <= 90.0:
        raise ValueError("FreeBOT lifting angle must lie in [0, 90] degrees")
    if any(float(value) < 0.0 for value in values[1:]):
        raise ValueError("Forces, lengths and friction must be non-negative")
    if shell_radius_m <= 0.0:
        raise ValueError("Shell radius must be positive")
    if mechanism_com_radius_m > shell_radius_m:
        raise ValueError("Mechanism COM radius cannot exceed shell radius")

    angle = math.radians(float(angle_deg))
    radius = float(shell_radius_m)
    com_radius = float(mechanism_com_radius_m)
    coefficient = float(shell_shell_friction_coefficient)
    parallel = float(parallel_force_n)
    perpendicular_minus_weight = (
        float(perpendicular_force_n) - float(gravity_force_n)
    )
    r1 = radius - com_radius * math.cos(angle)
    r2 = com_radius * math.sin(angle)
    r3 = radius - com_radius * math.sin(angle)
    r4 = com_radius * math.cos(angle)
    numerator = (
        parallel * (r2 + coefficient * r1 + coefficient * r4)
        + perpendicular_minus_weight * r4
    )
    denominator = (
        perpendicular_minus_weight * (r2 + r3 + coefficient * r1)
        + coefficient * parallel * r3
    )

    if abs(denominator) <= _EPSILON:
        return float("inf")
    result = numerator / denominator
    return float(result) if result >= 0.0 else float("inf")


def paper_required_connection_friction(
    angle_deg: float,
    magnetic_force_n: float,
    gravity_force_n: float,
    lower_hemisphere: bool,
) -> float:
    """Return the connection coefficient from FreeBOT Eqs. (9) and (12)."""

    values = (angle_deg, magnetic_force_n, gravity_force_n)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Connection-friction inputs must be finite")
    if not 0.0 <= angle_deg <= 90.0:
        raise ValueError("FreeBOT lifting angle must lie in [0, 90] degrees")
    if magnetic_force_n < 0.0 or gravity_force_n < 0.0:
        raise ValueError("Magnetic and gravity forces must be non-negative")
    if magnetic_force_n <= _EPSILON:
        return float("inf")

    angle = math.radians(float(angle_deg))
    signed_weight_normal = gravity_force_n * math.sin(angle)
    normal_load = (
        magnetic_force_n - signed_weight_normal
        if lower_hemisphere
        else magnetic_force_n + signed_weight_normal
    )
    if normal_load <= _EPSILON:
        return float("inf")
    return max(gravity_force_n * math.cos(angle) / normal_load, 0.0)


@dataclass(frozen=True)
class FreebotFrictionDiagnostics:
    """Static friction requirements evaluated at one magnetic interaction."""

    required_ground_coefficient: float
    required_connection_coefficient: float
    shell_shell_static_coefficient: float
    connection_is_feasible: bool


@dataclass(frozen=True)
class Figure9BalanceResidual:
    """Residuals of the three quasi-static balances in FreeBOT Eq. (5)."""

    vertical_force_n: float
    horizontal_force_n: float
    moment_nm: float


def figure9_balance_residual(
    *,
    gravity_force_n: float,
    perpendicular_force_n: float,
    parallel_force_n: float,
    shell_friction_n: float,
    shell_normal_n: float,
    ground_friction_n: float,
    ground_normal_n: float,
    shell_radius_m: float,
    com_radius_m: float,
    angle_deg: float,
) -> Figure9BalanceResidual:
    """Evaluate Eq. (5) using signed forces in the paper's local frame."""

    angle = math.radians(float(angle_deg))
    radius = float(shell_radius_m)
    com_radius = float(com_radius_m)
    r1 = radius - com_radius * math.cos(angle)
    r2 = com_radius * math.sin(angle)
    r3 = radius - com_radius * math.sin(angle)
    r4 = com_radius * math.cos(angle)
    return Figure9BalanceResidual(
        vertical_force_n=(
            float(shell_friction_n)
            + float(perpendicular_force_n)
            + float(ground_normal_n)
            - float(gravity_force_n)
        ),
        horizontal_force_n=(
            float(parallel_force_n)
            + float(ground_friction_n)
            - float(shell_normal_n)
        ),
        moment_nm=(
            float(shell_friction_n) * r1
            + float(shell_normal_n) * r2
            + float(ground_friction_n) * r3
            - float(ground_normal_n) * r4
        ),
    )


def evaluate_freebot_friction_diagnostics(
    interaction: ExternalMagneticInteraction,
    gravity_force_n: float,
    shell_radius_m: float,
    mechanism_com_radius_m: float,
    shell_shell_static_coefficient: float,
    lower_hemisphere: bool = False,
) -> FreebotFrictionDiagnostics:
    """Evaluate the paper balances using the current Fig. 4--5 forces."""

    required_ground = paper_required_ground_friction(
        angle_deg=interaction.lifting_angle_deg,
        parallel_force_n=interaction.parallel_force_n,
        perpendicular_force_n=interaction.perpendicular_force_n,
        gravity_force_n=gravity_force_n,
        shell_radius_m=shell_radius_m,
        mechanism_com_radius_m=mechanism_com_radius_m,
        shell_shell_friction_coefficient=shell_shell_static_coefficient,
    )
    required_connection = paper_required_connection_friction(
        angle_deg=interaction.lifting_angle_deg,
        magnetic_force_n=math.hypot(
            interaction.parallel_force_n,
            interaction.perpendicular_force_n,
        ),
        gravity_force_n=gravity_force_n,
        lower_hemisphere=lower_hemisphere,
    )
    return FreebotFrictionDiagnostics(
        required_ground_coefficient=required_ground,
        required_connection_coefficient=required_connection,
        shell_shell_static_coefficient=float(
            shell_shell_static_coefficient
        ),
        connection_is_feasible=(
            math.isfinite(required_connection)
            and required_connection <= shell_shell_static_coefficient
        ),
    )


class IsaacContactDiagnostics:
    """Read normal and friction forces resolved by PhysX contact views."""

    def __init__(
        self,
        rigid_prim_type: Any,
        pairs: tuple[tuple[str, str, str], ...],
    ) -> None:
        self._pairs: list[tuple[str, Any]] = []
        for label, sensor_path, filter_path in pairs:
            sensor = rigid_prim_type(
                paths=sensor_path,
                contact_filter_paths=filter_path,
                max_contact_count=64,
            )
            sensor.set_enabled_contact_tracking([True], threshold=0.0)
            self._pairs.append((label, sensor))

    @staticmethod
    def _numpy(value: Any) -> np.ndarray:
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def snapshots(
        self,
        time_step_s: float,
        labels: set[str] | None = None,
    ) -> dict[str, "ContactSnapshot"]:
        """Read every configured pair once and retain vector resultants."""

        result: dict[str, ContactSnapshot] = {}
        for label, sensor in self._pairs:
            if labels is not None and label not in labels:
                continue
            try:
                normal_matrix = self._numpy(
                    sensor.get_contact_force_matrix(dt=time_step_s)
                )
                normal_resultant = np.asarray(
                    normal_matrix[0, 0],
                    dtype=np.float64,
                )
                (
                    normal_scalars,
                    normal_points,
                    normal_directions,
                    separations,
                    contact_counts,
                    contact_starts,
                ) = sensor.get_contact_force_data(dt=time_step_s)
                normal_scalars = self._numpy(normal_scalars)
                normal_points = self._numpy(normal_points)
                normal_directions = self._numpy(normal_directions)
                separations = self._numpy(separations)
                contact_counts = self._numpy(contact_counts)
                contact_starts = self._numpy(contact_starts)
                contact_count = int(contact_counts[0, 0])
                contact_start = int(contact_starts[0, 0])
                contact_slice = slice(
                    contact_start,
                    contact_start + contact_count,
                )

                if contact_count:
                    selected_scalars = np.asarray(
                        normal_scalars[contact_slice],
                        dtype=np.float64,
                    ).reshape(-1, 1)
                    selected_points = np.asarray(
                        normal_points[contact_slice],
                        dtype=np.float64,
                    )
                    selected_directions = np.asarray(
                        normal_directions[contact_slice],
                        dtype=np.float64,
                    )
                    normal_vectors = selected_scalars * selected_directions
                    if np.linalg.norm(
                        np.sum(normal_vectors, axis=0) - normal_resultant
                    ) > np.linalg.norm(
                        -np.sum(normal_vectors, axis=0) - normal_resultant
                    ):
                        normal_vectors = -normal_vectors
                    normal_moment_origin = np.sum(
                        np.cross(selected_points, normal_vectors),
                        axis=0,
                    )
                    normal_weights = np.abs(
                        selected_scalars.reshape(-1)
                    )
                    normal_weight_sum = float(np.sum(normal_weights))
                    normal_application_point = (
                        np.average(
                            selected_points,
                            axis=0,
                            weights=normal_weights,
                        )
                        if normal_weight_sum > _EPSILON
                        else np.mean(selected_points, axis=0)
                    )
                    minimum_separation = float(
                        np.min(separations[contact_slice])
                    )
                else:
                    normal_moment_origin = np.zeros(3, dtype=np.float64)
                    normal_application_point = np.zeros(
                        3,
                        dtype=np.float64,
                    )
                    minimum_separation = float("nan")

                friction_forces, friction_points, counts, starts = (
                    sensor.get_friction_data(dt=time_step_s)
                )
                friction_forces = self._numpy(friction_forces)
                friction_points = self._numpy(friction_points)
                counts = self._numpy(counts)
                starts = self._numpy(starts)
                friction_count = int(counts[0, 0])
                friction_start = int(starts[0, 0])
                friction_slice = slice(
                    friction_start,
                    friction_start + friction_count,
                )
                if friction_count:
                    selected_friction = np.asarray(
                        friction_forces[friction_slice],
                        dtype=np.float64,
                    )
                    selected_friction_points = np.asarray(
                        friction_points[friction_slice],
                        dtype=np.float64,
                    )
                    friction_resultant = np.sum(
                        selected_friction,
                        axis=0,
                        dtype=np.float64,
                    )
                    friction_moment_origin = np.sum(
                        np.cross(
                            selected_friction_points,
                            selected_friction,
                        ),
                        axis=0,
                    )
                    friction_weights = np.linalg.norm(
                        selected_friction,
                        axis=1,
                    )
                    friction_weight_sum = float(np.sum(friction_weights))
                    friction_application_point = (
                        np.average(
                            selected_friction_points,
                            axis=0,
                            weights=friction_weights,
                        )
                        if friction_weight_sum > _EPSILON
                        else np.mean(selected_friction_points, axis=0)
                    )
                else:
                    friction_resultant = np.zeros(3, dtype=np.float64)
                    friction_moment_origin = np.zeros(3, dtype=np.float64)
                    friction_application_point = np.zeros(
                        3,
                        dtype=np.float64,
                    )
            except (AssertionError, AttributeError, RuntimeError) as error:
                result[label] = ContactSnapshot.unavailable(error)
                continue

            result[label] = ContactSnapshot(
                normal_force_world=normal_resultant,
                friction_force_world=friction_resultant,
                normal_moment_about_origin_world=normal_moment_origin,
                friction_moment_about_origin_world=friction_moment_origin,
                normal_application_point_world=normal_application_point,
                friction_application_point_world=friction_application_point,
                contact_count=contact_count,
                minimum_separation_m=minimum_separation,
            )
        return result

    def formatted_lines(
        self,
        time_step_s: float,
        snapshots: dict[str, "ContactSnapshot"] | None = None,
    ) -> list[str]:
        """Return one compact diagnostic line for every configured pair."""

        readings = self.snapshots(time_step_s) if snapshots is None else snapshots
        lines: list[str] = []
        for label, _ in self._pairs:
            reading = readings[label]
            if reading.error is not None:
                lines.append(
                    f"  CONTACT {label}: unavailable "
                    f"({reading.error})"
                )
                continue

            normal_norm = float(np.linalg.norm(reading.normal_force_world))
            friction_norm = float(
                np.linalg.norm(reading.friction_force_world)
            )
            utilization = (
                friction_norm / normal_norm
                if normal_norm > _EPSILON
                else float("nan")
            )
            lines.append(
                f"  CONTACT {label}: count={reading.contact_count} "
                f"Fn={normal_norm:.3f}N "
                f"Ft={friction_norm:.3f}N mu_used={utilization:.3f}"
                f" min_sep={1e3*reading.minimum_separation_m:+.3f}mm"
            )
        return lines


@dataclass(frozen=True)
class ContactSnapshot:
    """One PhysX contact-pair resultant acting on the sensor body."""

    normal_force_world: np.ndarray
    friction_force_world: np.ndarray
    normal_moment_about_origin_world: np.ndarray
    friction_moment_about_origin_world: np.ndarray
    normal_application_point_world: np.ndarray
    friction_application_point_world: np.ndarray
    contact_count: int
    minimum_separation_m: float
    error: str | None = None

    @classmethod
    def unavailable(cls, error: Exception) -> "ContactSnapshot":
        zero = np.zeros(3, dtype=np.float64)
        return cls(
            normal_force_world=zero,
            friction_force_world=zero,
            normal_moment_about_origin_world=zero,
            friction_moment_about_origin_world=zero,
            normal_application_point_world=zero,
            friction_application_point_world=zero,
            contact_count=0,
            minimum_separation_m=float("nan"),
            error=f"{type(error).__name__}: {error}",
        )

    @property
    def total_force_world(self) -> np.ndarray:
        return self.normal_force_world + self.friction_force_world

    def total_moment_about(self, reference_world: np.ndarray) -> np.ndarray:
        reference = np.asarray(reference_world, dtype=np.float64)
        moment_origin = (
            self.normal_moment_about_origin_world
            + self.friction_moment_about_origin_world
        )
        return moment_origin - np.cross(reference, self.total_force_world)
