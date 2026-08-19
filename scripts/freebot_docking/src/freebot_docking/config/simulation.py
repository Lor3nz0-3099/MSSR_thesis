from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral

@dataclass(frozen=True)
class AxisymmetricGridConfig:
    """Uniform cell-centered grid in the axisymmetric r-z plane"""

    radial_max_m: float
    axial_min_m: float
    axial_max_m: float
    radial_cells: int
    axial_cells: int

    def __post_init__(self) -> None:
        radial_max = float(self.radial_max_m)
        axial_min = float(self.axial_min_m)
        axial_max = float(self.axial_max_m)

        if not isfinite(radial_max) or radial_max <= 0.0:
            raise ValueError("Radial maximum must be finite and positive")

        if not isfinite(axial_min) or not isfinite(axial_max):
            raise ValueError("Axial bounds must be finite")

        if axial_min >= axial_max:
            raise ValueError("Axial minimum must be smaller than axial maximum")

        if (
            isinstance(self.radial_cells, bool)
            or not isinstance(self.radial_cells, Integral)
            or self.radial_cells < 2
        ):
            raise ValueError(
                "Radial cell count must be an integer of at least 2"
            )
        
        if (
            isinstance(self.axial_cells, bool)
            or not isinstance(self.axial_cells, Integral)
            or self.axial_cells < 2
        ):
            raise ValueError(
                "Axial cell count must be an integer of at least two"
            )

        object.__setattr__(self, "radial_max_m", radial_max)
        object.__setattr__(self, "axial_min_m", axial_min)
        object.__setattr__(self, "axial_max_m", axial_max)
        object.__setattr__(
            self,
            "radial_cells",
            int(self.radial_cells),
        )
        object.__setattr__(
            self,
            "axial_cells",
            int(self.axial_cells),
        )

    @property
    def radial_step_m(self) -> float:
        return self.radial_max_m / self.radial_cells

    @property
    def axial_step_m(self) -> float:
        return (
            self.axial_max_m - self.axial_min_m
        ) / self.axial_cells


@dataclass(frozen=True)
class AxisymmetricNonlinearSolverConfig:
    """Convergence controls for the nonlinear magnetostatic solve."""

    relative_tolerance: float = 1.0e-5
    max_iterations: int = 100
    relaxation_factor: float = 0.3
    linear_relative_tolerance: float = 1.0e-9
    linear_max_iterations: int = 5000

    def __post_init__(self) -> None:
        relative_tolerance = float(self.relative_tolerance)
        relaxation_factor = float(self.relaxation_factor)
        linear_relative_tolerance = float(
            self.linear_relative_tolerance
        )

        if (
            not isfinite(relative_tolerance)
            or relative_tolerance <= 0.0
        ):
            raise ValueError(
                "Nonlinear relative tolerance must be finite and positive"
            )
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, Integral)
            or self.max_iterations < 1
        ):
            raise ValueError(
                "Nonlinear maximum iterations must be a positive integer"
            )
        if (
            not isfinite(relaxation_factor)
            or not 0.0 < relaxation_factor <= 1.0
        ):
            raise ValueError("Relaxation factor must lie in (0, 1]")
        if (
            not isfinite(linear_relative_tolerance)
            or linear_relative_tolerance <= 0.0
        ):
            raise ValueError(
                "Linear relative tolerance must be finite and positive"
            )
        if (
            isinstance(self.linear_max_iterations, bool)
            or not isinstance(self.linear_max_iterations, Integral)
            or self.linear_max_iterations < 1
        ):
            raise ValueError(
                "Linear maximum iterations must be a positive integer"
            )

        object.__setattr__(
            self,
            "relative_tolerance",
            relative_tolerance,
        )
        object.__setattr__(
            self,
            "max_iterations",
            int(self.max_iterations),
        )
        object.__setattr__(
            self,
            "relaxation_factor",
            relaxation_factor,
        )
        object.__setattr__(
            self,
            "linear_relative_tolerance",
            linear_relative_tolerance,
        )
        object.__setattr__(
            self,
            "linear_max_iterations",
            int(self.linear_max_iterations),
        )


@dataclass(frozen=True)
class ShellContactFrictionConfig:
    """Parameters of the reduced shell-to-shell Coulomb contact law.

    The tangential spring stores the elastic displacement required for
    static friction.  Once its force exceeds the static limit, the law
    switches to dynamic Coulomb friction.
    """

    contact_tolerance_m: float = 0.0
    # Existing Isaac baseline values; they still require experimental tuning.
    static_friction_coefficient: float = 1.10
    dynamic_friction_coefficient: float = 1.00
    # Numerical compliance, not a material constant from the FreeBOT paper.
    tangential_stiffness_n_per_m: float = 1200.0
    tangential_damping_n_s_per_m: float = 18.0
    slip_speed_epsilon_m_per_s: float = 1.0e-6

    def __post_init__(self) -> None:
        tolerance = float(self.contact_tolerance_m)
        static_coefficient = float(self.static_friction_coefficient)
        dynamic_coefficient = float(self.dynamic_friction_coefficient)
        stiffness = float(self.tangential_stiffness_n_per_m)
        damping = float(self.tangential_damping_n_s_per_m)
        slip_epsilon = float(self.slip_speed_epsilon_m_per_s)

        values = (
            tolerance,
            static_coefficient,
            dynamic_coefficient,
            stiffness,
            damping,
            slip_epsilon,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("Contact-friction parameters must be finite")
        if tolerance < 0.0:
            raise ValueError("Contact tolerance must be non-negative")
        if static_coefficient < 0.0 or dynamic_coefficient < 0.0:
            raise ValueError("Friction coefficients must be non-negative")
        if dynamic_coefficient > static_coefficient:
            raise ValueError(
                "Dynamic friction cannot exceed static friction"
            )
        if stiffness <= 0.0:
            raise ValueError("Tangential stiffness must be positive")
        if damping < 0.0:
            raise ValueError("Tangential damping must be non-negative")
        if slip_epsilon <= 0.0:
            raise ValueError("Slip-speed epsilon must be positive")

        object.__setattr__(self, "contact_tolerance_m", tolerance)
        object.__setattr__(
            self,
            "static_friction_coefficient",
            static_coefficient,
        )
        object.__setattr__(
            self,
            "dynamic_friction_coefficient",
            dynamic_coefficient,
        )
        object.__setattr__(
            self,
            "tangential_stiffness_n_per_m",
            stiffness,
        )
        object.__setattr__(
            self,
            "tangential_damping_n_s_per_m",
            damping,
        )
        object.__setattr__(
            self,
            "slip_speed_epsilon_m_per_s",
            slip_epsilon,
        )
