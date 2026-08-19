from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ModuleMassConfig:
    """Mass budget matched to the 307.9 g FreeBOT prototype.

    Wheel mass is the manufacturer value for one Pololu 32x7 mm wheel.
    Caster mass is calculated for a solid steel ball of radius 4.65 mm.
    The unresolved chassis mass is the residual required to match the total
    module mass published in Table I of the FreeBOT paper.
    """

    module_total_kg: float = 0.3079
    # The paper only reports the complete-module mass.  Allocate a larger
    # share to the ferromagnetic shell so the unresolved residual does not
    # unrealistically overload the moving internal mechanism.
    shell_kg: float = 0.090
    wheel_kg: float = 0.11 * 0.028349523125
    caster_ball_kg: float = (
        (4.0 / 3.0) * math.pi * 0.00465**3 * 7850.0
    )
    # Reduced bounding box for chassis, motors, battery and magnet assembly.
    # It is used only to avoid the zero-inertia rigid body produced when the
    # visual-only internal link has no collision mesh.
    internal_box_size_m: tuple[float, float, float] = (0.060, 0.050, 0.035)

    @property
    def internal_link_kg(self) -> float:
        return (
            self.module_total_kg
            - self.shell_kg
            - 2.0 * self.wheel_kg
            - 2.0 * self.caster_ball_kg
        )

    @property
    def internal_mechanism_total_kg(self) -> float:
        return self.module_total_kg - self.shell_kg

    def __post_init__(self) -> None:
        values = (
            self.module_total_kg,
            self.shell_kg,
            self.wheel_kg,
            self.caster_ball_kg,
            self.internal_link_kg,
            *self.internal_box_size_m,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("Every FreeBOT mass must be finite and positive")

    def scaled(self, factor: float) -> "ModuleMassConfig":
        """Scale every body mass coherently for a diagnostic-only trial.

        Geometry is unchanged, so all explicitly authored inertias scale by
        the same factor when the stage is built. Magnetic forces and motor
        torque deliberately remain unchanged to isolate mass/weight limits.
        """

        scale = float(factor)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("Mass scale must be finite and positive")
        return ModuleMassConfig(
            module_total_kg=scale * self.module_total_kg,
            shell_kg=scale * self.shell_kg,
            wheel_kg=scale * self.wheel_kg,
            caster_ball_kg=scale * self.caster_ball_kg,
            internal_box_size_m=self.internal_box_size_m,
        )

    @property
    def internal_box_diagonal_inertia_kg_m2(self) -> tuple[float, float, float]:
        """Principal inertia of the documented reduced internal-body box."""

        x, y, z = self.internal_box_size_m
        mass = self.internal_link_kg
        return (
            mass * (y * y + z * z) / 12.0,
            mass * (x * x + z * z) / 12.0,
            mass * (x * x + y * y) / 12.0,
        )
