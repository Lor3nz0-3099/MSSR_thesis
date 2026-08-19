from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from numbers import Integral

import numpy as np
import numpy.typing as npt

from freebot_docking.config.magnet import (
    MagnetConfig,
    TabulatedBHCurve,
)
from freebot_docking.config.geometry import ShellGeometry
from freebot_docking.config.simulation import (
    AxisymmetricNonlinearSolverConfig,
)
from freebot_docking.physics.geometry import AxisymmetricGrid

MU0_H_PER_M = 4.0 * math.pi * 1e-7 #permeability of free space in H/m


def equivalent_dipole_moment_am2(config: MagnetConfig) -> float:
    """Equivalent dipole moment of a uniformly magnetized cylinder."""
    return config.remanence_t * config.volume_m3 / MU0_H_PER_M


def axial_flux_density_t(
    gap_m: float,
    config: MagnetConfig,
) -> float:
    """
    Magnetic flux density on the cylinder axis.
    
    gap_m is measured outward from the active circular face.
    This is the finite-cylinder expression, not a point-dipole model.
    """

    gap = _validated_gap_m(gap_m)
    
    radius = config.radius_m
    length = config.length_m

    far_edge = (gap + length) / math.sqrt((gap+length)**2 + radius**2)

    near_edge = gap / math.sqrt(gap**2 + radius**2)

    density = 0.5 * config.remanence_t * (far_edge - near_edge)

    return density

def _validated_gap_m(gap_m: float) -> float:
    gap = float(gap_m)

    if not math.isfinite(gap):
        raise ValueError("Gap must be finite")
    if gap < 0.0:
        raise ValueError("Gap must be non-negative")

    return gap

def axial_flux_density_gradient_t_per_m(
    gap_m: float,
    config: MagnetConfig,
) -> float:
    """
    Axial derivative dB/dg outside the active magnet face.

    The result is negative because the field decreases as the gap grows.
    """
    gap = _validated_gap_m(gap_m)

    radius_squared = config.radius_m**2
    far_distance = gap + config.length_m

    far_gradient = radius_squared / (
        far_distance**2 + radius_squared
    ) ** 1.5

    near_gradient = radius_squared / (
        gap**2 + radius_squared
    ) ** 1.5

    return (
        0.5
        * config.remanence_t
        * (far_gradient - near_gradient)
    )


def magnetization_magnitude_a_per_m(
    field_strength_a_per_m: float,
    bh_curve: TabulatedBHCurve,
) -> float:
    """
    Return magnetization magnitude from B = mu0 * (H + M).

    This assumes an isotropic first-magnetization curve, without
    hysteresis. It does not solve the surrounding magnetic field.
    """

    field_strength = float(field_strength_a_per_m)

    flux_density = (bh_curve.flux_density_for_field_strength_t(field_strength))

    return (flux_density / MU0_H_PER_M) - field_strength


def secant_reluctivity_with_saturation_extension_m_per_h(
    flux_density_t: float,
    bh_curve: TabulatedBHCurve,
) -> float:
    """
    Return H/B, extending the final B-H sample with vacuum slope.

    Beyond the tabulated range the magnetization is held constant, so
    additional flux requires dB/dH = mu0. The original table remains
    strict and is not silently extrapolated.
    """

    flux_density = float(flux_density_t)

    if not math.isfinite(flux_density) or flux_density < 0.0:
        raise ValueError(
            "Flux density must be finite and non-negative"
        )

    maximum_tabulated_flux = bh_curve.flux_density_t[-1]
    if flux_density <= maximum_tabulated_flux:
        return bh_curve.secant_reluctivity_m_per_h(
            flux_density
        )

    maximum_tabulated_field = bh_curve.field_strength_a_per_m[-1]
    extended_field = (
        maximum_tabulated_field
        + (
            flux_density - maximum_tabulated_flux
        ) / MU0_H_PER_M
    )

    return extended_field / flux_density

class AxisymmetricMaterialRegion(IntEnum):
    """Region of an axisymmetric grid."""

    AIR = 0
    PERMANENT_MAGNET = 1
    FERROMAGNETIC_SHELL = 2

def build_internal_axisymmetric_material_map(
    grid: AxisymmetricGrid,
    shell_geometry: ShellGeometry,
    magnet_config: MagnetConfig,
    face_gap_m: float,
) -> npt.NDArray[np.uint8]:
    """
    Classify grid cells for the internal magnet-shell problem.

    The shell center is at z=0 and the active magnet axis points
    toward positive z.
    """

    face_gap = float(face_gap_m)

    if not math.isfinite(face_gap) or face_gap < 0.0:
        raise ValueError("Magnet face gap must be finite and non-negative")

    magnet_front_z = (
        shell_geometry.inner_radius_m - face_gap
    )
    magnet_back_z = (
        magnet_front_z - magnet_config.length_m
    )

    farthest_axial_position = max(
        abs(magnet_front_z),
        abs(magnet_back_z),
    )
    farthest_magnet_radius = math.hypot(
        magnet_config.radius_m,
        farthest_axial_position,
    )

    if farthest_magnet_radius > shell_geometry.inner_radius_m:
        raise ValueError(
            "Cylindrical magnet does not fit inside the shell cavity"
        )

    radial_coordinates, axial_coordinates = np.meshgrid(
        grid.radial_centers_m,
        grid.axial_centers_m,
    )

    spherical_radius = np.hypot(
        radial_coordinates,
        axial_coordinates,
    )

    shell_mask = (
        (spherical_radius >= shell_geometry.inner_radius_m)
        & (spherical_radius <= shell_geometry.outer_radius_m)
    )

    magnet_mask = (
        (radial_coordinates <= magnet_config.radius_m)
        & (axial_coordinates >= magnet_back_z)
        & (axial_coordinates <= magnet_front_z)
    )

    material_map = np.full(
        grid.cell_shape,
        AxisymmetricMaterialRegion.AIR,
        dtype=np.uint8,
    )

    material_map[shell_mask] = (
        AxisymmetricMaterialRegion.FERROMAGNETIC_SHELL
    )
    material_map[magnet_mask] = (
        AxisymmetricMaterialRegion.PERMANENT_MAGNET
    )

    material_map.setflags(write=False)

    return material_map


def build_initial_reluctivity_map_m_per_h(
    material_map: npt.NDArray[np.uint8],
    magnet_config: MagnetConfig,
    shell_bh_curve: TabulatedBHCurve,
) -> npt.NDArray[np.float64]:
    """Build the initial cell-centered magnetic reluctivity map."""

    if material_map.ndim != 2:
        raise ValueError("Material map must be two-dimensional")

    valid_regions = {
        int(AxisymmetricMaterialRegion.AIR),
        int(AxisymmetricMaterialRegion.PERMANENT_MAGNET),
        int(AxisymmetricMaterialRegion.FERROMAGNETIC_SHELL),
    }

    if not set(np.unique(material_map)).issubset(valid_regions):
        raise ValueError("Material map contains an unknown region")

    reluctivity = np.full(
        material_map.shape,
        1.0 / MU0_H_PER_M,
        dtype=np.float64,
    )

    magnet_mask = (
        material_map
        == AxisymmetricMaterialRegion.PERMANENT_MAGNET
    )
    shell_mask = (
        material_map
        == AxisymmetricMaterialRegion.FERROMAGNETIC_SHELL
    )

    reluctivity[magnet_mask] = 1.0 / (
        MU0_H_PER_M
        * magnet_config.recoil_relative_permeability
    )

    reluctivity[shell_mask] = (
        shell_bh_curve.secant_reluctivity_m_per_h(0.0)
    )

    reluctivity.setflags(write=False)

    return reluctivity

def build_reluctivity_map_m_per_h(
    material_map: npt.NDArray[np.uint8],
    flux_density_magnitude_t: npt.NDArray[np.float64],
    magnet_config: MagnetConfig,
    shell_bh_curve: TabulatedBHCurve,
) -> npt.NDArray[np.float64]:
    """
    Build cell reluctivities from the current flux-density magnitude.

    Only the ferromagnetic shell is nonlinear. Air and permanent
    magnet retain constant reluctivity.
    """

    flux_density = np.asarray(
        flux_density_magnitude_t,
        dtype=np.float64,
    )

    if flux_density.shape != material_map.shape:
        raise ValueError(
            "Flux-density and material maps must have the same shape"
        )

    if not np.all(np.isfinite(flux_density)):
        raise ValueError("Flux-density values must be finite")

    if np.any(flux_density < 0.0):
        raise ValueError(
            "Flux-density magnitude must be non-negative"
        )

    reluctivity = np.array(
        build_initial_reluctivity_map_m_per_h(
            material_map,
            magnet_config,
            shell_bh_curve,
        ),
        copy=True,
    )

    shell_mask = (
        material_map
        == AxisymmetricMaterialRegion.FERROMAGNETIC_SHELL
    )

    shell_flux_density = flux_density[shell_mask]

    reluctivity[shell_mask] = np.fromiter(
        (
            secant_reluctivity_with_saturation_extension_m_per_h(
                value,
                shell_bh_curve,
            )
            for value in shell_flux_density
        ),
        dtype=np.float64,
        count=shell_flux_density.size,
    )

    reluctivity.setflags(write=False)

    return reluctivity

def build_axial_remanence_map_t(
    material_map: npt.NDArray[np.uint8],
    magnet_config: MagnetConfig,
) -> npt.NDArray[np.float64]:
    """
    Build the axial remanent flux-density map.

    In this axisymmetric model, the active magnet axis points
    toward positive z.
    """

    if material_map.ndim != 2:
        raise ValueError("Material map must be two-dimensional")

    axial_remanence = np.zeros(
        material_map.shape,
        dtype=np.float64,
    )

    magnet_mask = (
        material_map
        == AxisymmetricMaterialRegion.PERMANENT_MAGNET
    )

    axial_remanence[magnet_mask] = magnet_config.remanence_t
    axial_remanence.setflags(write=False)

    return axial_remanence

def flux_density_from_flux_function_t(
    grid: AxisymmetricGrid,
    flux_function_wb: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """
    Compute cell-centered Br and Bz from nodal psi = r * A_phi.

    The flux function is defined at grid vertices, while the returned
    magnetic flux-density components are defined at cell centers.
    """

    flux_function = np.asarray(
        flux_function_wb,
        dtype=np.float64,
    )

    expected_shape = (
        grid.config.axial_cells + 1,
        grid.config.radial_cells + 1,
    )

    if flux_function.shape != expected_shape:
        raise ValueError(
            "Flux function shape must match the grid vertices"
        )

    if not np.all(np.isfinite(flux_function)):
        raise ValueError("Flux function values must be finite")

    radial_step = grid.config.radial_step_m
    axial_step = grid.config.axial_step_m

    radial_derivative = (
        (
            flux_function[:-1, 1:]
            - flux_function[:-1, :-1]
        )
        + (
            flux_function[1:, 1:]
            - flux_function[1:, :-1]
        )
    ) / (2.0 * radial_step)

    axial_derivative = (
        (
            flux_function[1:, :-1]
            - flux_function[:-1, :-1]
        )
        + (
            flux_function[1:, 1:]
            - flux_function[:-1, 1:]
        )
    ) / (2.0 * axial_step)

    radial_coordinates = grid.radial_centers_m[np.newaxis, :]

    radial_flux_density = (
        -axial_derivative / radial_coordinates
    )
    axial_flux_density = (
        radial_derivative / radial_coordinates
    )

    radial_flux_density.setflags(write=False)
    axial_flux_density.setflags(write=False)

    return radial_flux_density, axial_flux_density

def build_flux_function_dirichlet_mask(
    grid: AxisymmetricGrid,
) -> npt.NDArray[np.bool_]:
    """
    Mark vertices where the flux function is fixed to zero.

    Psi is zero on the symmetry axis and on the finite outer
    boundary used to approximate an unbounded air domain.
    """

    boundary_mask = np.zeros(
        (
            grid.config.axial_cells + 1,
            grid.config.radial_cells + 1,
        ),
        dtype=np.bool_,
    )

    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True

    boundary_mask.setflags(write=False)

    return boundary_mask

def build_axisymmetric_flux_coefficient_per_h(
    grid: AxisymmetricGrid,
    reluctivity_map_m_per_h: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Build the cell coefficient k = nu / r for the flux equation."""

    reluctivity = np.asarray(
        reluctivity_map_m_per_h,
        dtype=np.float64,
    )

    if reluctivity.shape != grid.cell_shape:
        raise ValueError(
            "Reluctivity map shape must match the grid cells"
        )

    if not np.all(np.isfinite(reluctivity)):
        raise ValueError("Reluctivity values must be finite")

    if np.any(reluctivity <= 0.0):
        raise ValueError("Reluctivity values must be positive")

    radial_coordinates = grid.radial_centers_m[np.newaxis, :]

    coefficient = reluctivity / radial_coordinates
    coefficient.setflags(write=False)

    return coefficient


def _axisymmetric_element_matrix(
    grid: AxisymmetricGrid,
) -> npt.NDArray[np.float64]:
    """Return the unit-coefficient bilinear element matrix."""

    radial_step = grid.config.radial_step_m
    axial_step = grid.config.axial_step_m

    radial_part = (
        axial_step
        / (6.0 * radial_step)
        * np.array(
            [
                [2.0, -2.0, 1.0, -1.0],
                [-2.0, 2.0, -1.0, 1.0],
                [1.0, -1.0, 2.0, -2.0],
                [-1.0, 1.0, -2.0, 2.0],
            ],
            dtype=np.float64,
        )
    )
    axial_part = (
        radial_step
        / (6.0 * axial_step)
        * np.array(
            [
                [2.0, 1.0, -2.0, -1.0],
                [1.0, 2.0, -1.0, -2.0],
                [-2.0, -1.0, 2.0, 1.0],
                [-1.0, -2.0, 1.0, 2.0],
            ],
            dtype=np.float64,
        )
    )

    return radial_part + axial_part


def _validated_flux_coefficient_per_h(
    grid: AxisymmetricGrid,
    coefficient_per_h: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    coefficient = np.asarray(
        coefficient_per_h,
        dtype=np.float64,
    )

    if coefficient.shape != grid.cell_shape:
        raise ValueError(
            "Flux coefficient shape must match the grid cells"
        )
    if not np.all(np.isfinite(coefficient)):
        raise ValueError("Flux coefficient values must be finite")
    if np.any(coefficient <= 0.0):
        raise ValueError("Flux coefficient values must be positive")

    return coefficient


def apply_axisymmetric_flux_operator_a(
    grid: AxisymmetricGrid,
    coefficient_per_h: npt.NDArray[np.float64],
    flux_function_wb: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Apply the finite-element operator for -div(k grad(psi))."""

    coefficient = _validated_flux_coefficient_per_h(
        grid,
        coefficient_per_h,
    )
    flux_function = np.asarray(
        flux_function_wb,
        dtype=np.float64,
    )
    vertex_shape = (
        grid.config.axial_cells + 1,
        grid.config.radial_cells + 1,
    )

    if flux_function.shape != vertex_shape:
        raise ValueError(
            "Flux function shape must match the grid vertices"
        )
    if not np.all(np.isfinite(flux_function)):
        raise ValueError("Flux function values must be finite")

    boundary_mask = build_flux_function_dirichlet_mask(grid)
    interior_flux = np.array(flux_function, copy=True)
    interior_flux[boundary_mask] = 0.0

    local_flux = np.stack(
        (
            interior_flux[:-1, :-1],
            interior_flux[:-1, 1:],
            interior_flux[1:, :-1],
            interior_flux[1:, 1:],
        ),
        axis=-1,
    )
    element_matrix = _axisymmetric_element_matrix(grid)
    local_operator = coefficient[..., np.newaxis] * np.einsum(
        "ij,...j->...i",
        element_matrix,
        local_flux,
    )

    operator = np.zeros(vertex_shape, dtype=np.float64)
    operator[:-1, :-1] += local_operator[..., 0]
    operator[:-1, 1:] += local_operator[..., 1]
    operator[1:, :-1] += local_operator[..., 2]
    operator[1:, 1:] += local_operator[..., 3]
    operator[boundary_mask] = 0.0
    operator.setflags(write=False)

    return operator


def build_axisymmetric_remanence_source_a(
    grid: AxisymmetricGrid,
    reluctivity_map_m_per_h: npt.NDArray[np.float64],
    axial_remanence_map_t: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Build the nodal source from Hc = nu * Br in the magnet."""

    reluctivity = np.asarray(
        reluctivity_map_m_per_h,
        dtype=np.float64,
    )
    remanence = np.asarray(
        axial_remanence_map_t,
        dtype=np.float64,
    )

    if reluctivity.shape != grid.cell_shape:
        raise ValueError(
            "Reluctivity map shape must match the grid cells"
        )
    if remanence.shape != grid.cell_shape:
        raise ValueError(
            "Remanence map shape must match the grid cells"
        )
    if not np.all(np.isfinite(reluctivity)) or np.any(
        reluctivity <= 0.0
    ):
        raise ValueError(
            "Reluctivity values must be finite and positive"
        )
    if not np.all(np.isfinite(remanence)):
        raise ValueError("Remanence values must be finite")

    coercive_field = reluctivity * remanence
    local_source = (
        0.5 * grid.config.axial_step_m * coercive_field
    )
    vertex_shape = (
        grid.config.axial_cells + 1,
        grid.config.radial_cells + 1,
    )
    source = np.zeros(vertex_shape, dtype=np.float64)

    source[:-1, :-1] -= local_source
    source[:-1, 1:] += local_source
    source[1:, :-1] -= local_source
    source[1:, 1:] += local_source

    boundary_mask = build_flux_function_dirichlet_mask(grid)
    source[boundary_mask] = 0.0
    source.setflags(write=False)

    return source


def _axisymmetric_flux_operator_diagonal_a_per_wb(
    grid: AxisymmetricGrid,
    coefficient_per_h: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    coefficient = _validated_flux_coefficient_per_h(
        grid,
        coefficient_per_h,
    )
    element_diagonal = np.diag(
        _axisymmetric_element_matrix(grid)
    )
    vertex_shape = (
        grid.config.axial_cells + 1,
        grid.config.radial_cells + 1,
    )
    diagonal = np.zeros(vertex_shape, dtype=np.float64)

    diagonal[:-1, :-1] += coefficient * element_diagonal[0]
    diagonal[:-1, 1:] += coefficient * element_diagonal[1]
    diagonal[1:, :-1] += coefficient * element_diagonal[2]
    diagonal[1:, 1:] += coefficient * element_diagonal[3]

    boundary_mask = build_flux_function_dirichlet_mask(grid)
    diagonal[boundary_mask] = 1.0

    return diagonal


@dataclass(frozen=True)
class AxisymmetricLinearSolveResult:
    """Result and convergence diagnostics of the linear flux solve."""

    flux_function_wb: npt.NDArray[np.float64]
    iterations: int
    relative_residual: float


def solve_axisymmetric_flux_function(
    grid: AxisymmetricGrid,
    coefficient_per_h: npt.NDArray[np.float64],
    source_a: npt.NDArray[np.float64],
    relative_tolerance: float = 1.0e-10,
    max_iterations: int = 5000,
    initial_flux_function_wb: npt.NDArray[np.float64] | None = None,
) -> AxisymmetricLinearSolveResult:
    """Solve the linear flux equation with Jacobi-preconditioned CG."""

    coefficient = _validated_flux_coefficient_per_h(
        grid,
        coefficient_per_h,
    )
    source = np.asarray(source_a, dtype=np.float64)
    vertex_shape = (
        grid.config.axial_cells + 1,
        grid.config.radial_cells + 1,
    )

    if source.shape != vertex_shape:
        raise ValueError("Source shape must match the grid vertices")
    if not np.all(np.isfinite(source)):
        raise ValueError("Source values must be finite")

    tolerance = float(relative_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("Relative tolerance must be finite and positive")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, Integral)
        or max_iterations < 1
    ):
        raise ValueError("Maximum iterations must be a positive integer")

    boundary_mask = build_flux_function_dirichlet_mask(grid)
    right_hand_side = np.array(source, copy=True)
    right_hand_side[boundary_mask] = 0.0
    source_norm = float(np.linalg.norm(right_hand_side))

    if initial_flux_function_wb is None:
        flux_function = np.zeros(vertex_shape, dtype=np.float64)
    else:
        flux_function = np.array(
            initial_flux_function_wb,
            dtype=np.float64,
            copy=True,
        )
        if flux_function.shape != vertex_shape:
            raise ValueError(
                "Initial flux function shape must match grid vertices"
            )
        if not np.all(np.isfinite(flux_function)):
            raise ValueError(
                "Initial flux function values must be finite"
            )
        flux_function[boundary_mask] = 0.0

    if source_norm == 0.0:
        flux_function.fill(0.0)
        flux_function.setflags(write=False)
        return AxisymmetricLinearSolveResult(
            flux_function_wb=flux_function,
            iterations=0,
            relative_residual=0.0,
        )

    diagonal = _axisymmetric_flux_operator_diagonal_a_per_wb(
        grid,
        coefficient,
    )
    residual = right_hand_side - apply_axisymmetric_flux_operator_a(
        grid,
        coefficient,
        flux_function,
    )
    initial_relative_residual = (
        float(np.linalg.norm(residual)) / source_norm
    )
    if initial_relative_residual <= tolerance:
        flux_function.setflags(write=False)
        return AxisymmetricLinearSolveResult(
            flux_function_wb=flux_function,
            iterations=0,
            relative_residual=initial_relative_residual,
        )

    preconditioned_residual = residual / diagonal
    search_direction = preconditioned_residual.copy()
    residual_product = float(
        np.sum(residual * preconditioned_residual)
    )

    for iteration in range(1, int(max_iterations) + 1):
        operator_direction = apply_axisymmetric_flux_operator_a(
            grid,
            coefficient,
            search_direction,
        )
        denominator = float(
            np.sum(search_direction * operator_direction)
        )

        if not math.isfinite(denominator) or denominator <= 0.0:
            raise RuntimeError("Conjugate-gradient solver broke down")

        step = residual_product / denominator
        flux_function += step * search_direction
        residual -= step * operator_direction
        residual[boundary_mask] = 0.0

        relative_residual = (
            float(np.linalg.norm(residual)) / source_norm
        )
        if relative_residual <= tolerance:
            flux_function[boundary_mask] = 0.0
            flux_function.setflags(write=False)
            return AxisymmetricLinearSolveResult(
                flux_function_wb=flux_function,
                iterations=iteration,
                relative_residual=relative_residual,
            )

        preconditioned_residual = residual / diagonal
        next_residual_product = float(
            np.sum(residual * preconditioned_residual)
        )
        beta = next_residual_product / residual_product
        search_direction = (
            preconditioned_residual + beta * search_direction
        )
        search_direction[boundary_mask] = 0.0
        residual_product = next_residual_product

    raise RuntimeError(
        "Conjugate-gradient solver did not converge within "
        f"{max_iterations} iterations"
    )


@dataclass(frozen=True)
class AxisymmetricMagnetostaticSolution:
    """Converged nonlinear field and material state."""

    material_map: npt.NDArray[np.uint8]
    reluctivity_map_m_per_h: npt.NDArray[np.float64]
    flux_function_wb: npt.NDArray[np.float64]
    radial_flux_density_t: npt.NDArray[np.float64]
    axial_flux_density_t: npt.NDArray[np.float64]
    flux_density_magnitude_t: npt.NDArray[np.float64]
    nonlinear_iterations: int
    nonlinear_relative_change: float
    total_linear_iterations: int
    final_linear_relative_residual: float


@dataclass(frozen=True)
class MaxwellStressForceResult:
    """Axial force obtained from a closed air-surface stress integral."""

    axial_force_n: float
    top_cap_force_n: float
    bottom_cap_force_n: float
    radial_side_force_n: float
    bottom_axial_face_index: int
    top_axial_face_index: int
    radial_face_index: int


def solve_axisymmetric_magnetostatic_material_map(
    grid: AxisymmetricGrid,
    material_map: npt.NDArray[np.uint8],
    magnet_config: MagnetConfig,
    shell_bh_curve: TabulatedBHCurve,
    solver_config: AxisymmetricNonlinearSolverConfig | None = None,
) -> AxisymmetricMagnetostaticSolution:
    """Solve a nonlinear axisymmetric material map."""

    controls = (
        AxisymmetricNonlinearSolverConfig()
        if solver_config is None
        else solver_config
    )
    remanence_map = build_axial_remanence_map_t(
        material_map,
        magnet_config,
    )
    reluctivity = np.array(
        build_initial_reluctivity_map_m_per_h(
            material_map,
            magnet_config,
            shell_bh_curve,
        ),
        copy=True,
    )
    shell_mask = (
        material_map
        == AxisymmetricMaterialRegion.FERROMAGNETIC_SHELL
    )
    flux_function: npt.NDArray[np.float64] | None = None
    total_linear_iterations = 0

    for nonlinear_iteration in range(
        1,
        controls.max_iterations + 1,
    ):
        coefficient = build_axisymmetric_flux_coefficient_per_h(
            grid,
            reluctivity,
        )
        source = build_axisymmetric_remanence_source_a(
            grid,
            reluctivity,
            remanence_map,
        )
        linear_result = solve_axisymmetric_flux_function(
            grid,
            coefficient,
            source,
            relative_tolerance=(
                controls.linear_relative_tolerance
            ),
            max_iterations=controls.linear_max_iterations,
            initial_flux_function_wb=flux_function,
        )
        total_linear_iterations += linear_result.iterations
        flux_function = linear_result.flux_function_wb
        radial_field, axial_field = (
            flux_density_from_flux_function_t(
                grid,
                flux_function,
            )
        )
        field_magnitude = np.hypot(
            radial_field,
            axial_field,
        )
        target_reluctivity = build_reluctivity_map_m_per_h(
            material_map,
            field_magnitude,
            magnet_config,
            shell_bh_curve,
        )

        if np.any(shell_mask):
            shell_scale = np.maximum(
                np.abs(reluctivity[shell_mask]),
                np.abs(target_reluctivity[shell_mask]),
            )
            relative_change = float(
                np.max(
                    np.abs(
                        target_reluctivity[shell_mask]
                        - reluctivity[shell_mask]
                    )
                    / shell_scale
                )
            )
        else:
            relative_change = 0.0

        if relative_change <= controls.relative_tolerance:
            reluctivity.setflags(write=False)
            field_magnitude.setflags(write=False)
            return AxisymmetricMagnetostaticSolution(
                material_map=material_map,
                reluctivity_map_m_per_h=reluctivity,
                flux_function_wb=flux_function,
                radial_flux_density_t=radial_field,
                axial_flux_density_t=axial_field,
                flux_density_magnitude_t=field_magnitude,
                nonlinear_iterations=nonlinear_iteration,
                nonlinear_relative_change=relative_change,
                total_linear_iterations=total_linear_iterations,
                final_linear_relative_residual=(
                    linear_result.relative_residual
                ),
            )

        relaxed_reluctivity = (
            (1.0 - controls.relaxation_factor) * reluctivity
            + controls.relaxation_factor * target_reluctivity
        )
        reluctivity = np.asarray(
            relaxed_reluctivity,
            dtype=np.float64,
        )

    raise RuntimeError(
        "Nonlinear magnetostatic solver did not converge within "
        f"{controls.max_iterations} iterations"
    )


def solve_internal_axisymmetric_magnetostatics(
    grid: AxisymmetricGrid,
    shell_geometry: ShellGeometry,
    magnet_config: MagnetConfig,
    shell_bh_curve: TabulatedBHCurve,
    face_gap_m: float,
    solver_config: AxisymmetricNonlinearSolverConfig | None = None,
) -> AxisymmetricMagnetostaticSolution:
    """Solve the nonlinear internal magnet-shell field problem."""

    material_map = build_internal_axisymmetric_material_map(
        grid,
        shell_geometry,
        magnet_config,
        face_gap_m,
    )

    return solve_axisymmetric_magnetostatic_material_map(
        grid,
        material_map,
        magnet_config,
        shell_bh_curve,
        solver_config,
    )


def compute_magnet_axial_force_from_maxwell_stress(
    grid: AxisymmetricGrid,
    material_map: npt.NDArray[np.uint8],
    radial_flux_density_t: npt.NDArray[np.float64],
    axial_flux_density_t: npt.NDArray[np.float64],
    air_layers: int = 1,
) -> MaxwellStressForceResult:
    """
    Integrate the vacuum Maxwell stress around the permanent magnet.

    The integration surface is a closed cylinder placed on grid faces
    and separated from the magnet by the requested number of air cells.
    """

    if (
        isinstance(air_layers, bool)
        or not isinstance(air_layers, Integral)
        or air_layers < 1
    ):
        raise ValueError("Air layers must be a positive integer")

    materials = np.asarray(material_map)
    radial_field = np.asarray(
        radial_flux_density_t,
        dtype=np.float64,
    )
    axial_field = np.asarray(
        axial_flux_density_t,
        dtype=np.float64,
    )

    if materials.shape != grid.cell_shape:
        raise ValueError("Material map shape must match grid cells")
    if radial_field.shape != grid.cell_shape:
        raise ValueError("Radial field shape must match grid cells")
    if axial_field.shape != grid.cell_shape:
        raise ValueError("Axial field shape must match grid cells")
    if not np.all(np.isfinite(radial_field)) or not np.all(
        np.isfinite(axial_field)
    ):
        raise ValueError("Flux-density values must be finite")

    magnet_indices = np.argwhere(
        materials
        == AxisymmetricMaterialRegion.PERMANENT_MAGNET
    )
    if magnet_indices.size == 0:
        raise ValueError("Material map does not contain a magnet")

    minimum_axial_cell = int(np.min(magnet_indices[:, 0]))
    maximum_axial_cell = int(np.max(magnet_indices[:, 0]))
    maximum_radial_cell = int(np.max(magnet_indices[:, 1]))

    bottom_face = minimum_axial_cell - int(air_layers)
    top_face = maximum_axial_cell + 1 + int(air_layers)
    radial_face = maximum_radial_cell + 1 + int(air_layers)

    if bottom_face <= 0 or top_face >= grid.config.axial_cells:
        raise ValueError(
            "Not enough axial air cells for the stress surface"
        )
    if radial_face >= grid.config.radial_cells:
        raise ValueError(
            "Not enough radial air cells for the stress surface"
        )

    air_region = int(AxisymmetricMaterialRegion.AIR)
    surface_cells = (
        materials[top_face - 1, :radial_face],
        materials[top_face, :radial_face],
        materials[bottom_face - 1, :radial_face],
        materials[bottom_face, :radial_face],
        materials[bottom_face:top_face, radial_face - 1],
        materials[bottom_face:top_face, radial_face],
    )
    if any(
        np.any(cells != air_region)
        for cells in surface_cells
    ):
        raise ValueError(
            "Maxwell stress surface must lie entirely in air"
        )

    top_radial_field = 0.5 * (
        radial_field[top_face - 1, :radial_face]
        + radial_field[top_face, :radial_face]
    )
    top_axial_field = 0.5 * (
        axial_field[top_face - 1, :radial_face]
        + axial_field[top_face, :radial_face]
    )
    bottom_radial_field = 0.5 * (
        radial_field[bottom_face - 1, :radial_face]
        + radial_field[bottom_face, :radial_face]
    )
    bottom_axial_field = 0.5 * (
        axial_field[bottom_face - 1, :radial_face]
        + axial_field[bottom_face, :radial_face]
    )

    radial_faces = grid.radial_faces_m[: radial_face + 1]
    cap_areas = math.pi * (
        radial_faces[1:] ** 2
        - radial_faces[:-1] ** 2
    )
    top_normal_stress = (
        top_axial_field**2 - top_radial_field**2
    ) / (2.0 * MU0_H_PER_M)
    bottom_normal_stress = (
        bottom_axial_field**2 - bottom_radial_field**2
    ) / (2.0 * MU0_H_PER_M)
    top_cap_force = float(
        np.sum(top_normal_stress * cap_areas)
    )
    bottom_cap_force = -float(
        np.sum(bottom_normal_stress * cap_areas)
    )

    side_radial_field = 0.5 * (
        radial_field[bottom_face:top_face, radial_face - 1]
        + radial_field[bottom_face:top_face, radial_face]
    )
    side_axial_field = 0.5 * (
        axial_field[bottom_face:top_face, radial_face - 1]
        + axial_field[bottom_face:top_face, radial_face]
    )
    side_band_area = (
        2.0
        * math.pi
        * grid.radial_faces_m[radial_face]
        * grid.config.axial_step_m
    )
    radial_side_force = float(
        np.sum(
            side_radial_field
            * side_axial_field
            / MU0_H_PER_M
            * side_band_area
        )
    )
    axial_force = (
        top_cap_force
        + bottom_cap_force
        + radial_side_force
    )

    return MaxwellStressForceResult(
        axial_force_n=axial_force,
        top_cap_force_n=top_cap_force,
        bottom_cap_force_n=bottom_cap_force,
        radial_side_force_n=radial_side_force,
        bottom_axial_face_index=bottom_face,
        top_axial_face_index=top_face,
        radial_face_index=radial_face,
    )


def compute_axisymmetric_body_axial_force_from_maxwell_stress(
    grid: AxisymmetricGrid,
    enclosed_cell_mask: npt.NDArray[np.bool_],
    air_cell_mask: npt.NDArray[np.bool_],
    radial_flux_density_t: npt.NDArray[np.float64],
    axial_flux_density_t: npt.NDArray[np.float64],
    air_layers: int = 1,
) -> MaxwellStressForceResult:
    """Integrate Maxwell stress around an arbitrary enclosed body mask."""

    enclosed = np.asarray(enclosed_cell_mask, dtype=np.bool_)
    air = np.asarray(air_cell_mask, dtype=np.bool_)

    if enclosed.shape != grid.cell_shape:
        raise ValueError("Enclosed-body mask must match grid cells")
    if air.shape != grid.cell_shape:
        raise ValueError("Air mask must match grid cells")
    if not np.any(enclosed):
        raise ValueError("Enclosed-body mask must not be empty")
    if np.any(enclosed & air):
        raise ValueError("Enclosed-body cells cannot also be air")

    integration_regions = np.full(
        grid.cell_shape,
        AxisymmetricMaterialRegion.FERROMAGNETIC_SHELL,
        dtype=np.uint8,
    )
    integration_regions[air] = AxisymmetricMaterialRegion.AIR
    integration_regions[enclosed] = (
        AxisymmetricMaterialRegion.PERMANENT_MAGNET
    )

    return compute_magnet_axial_force_from_maxwell_stress(
        grid,
        integration_regions,
        radial_flux_density_t,
        axial_flux_density_t,
        air_layers,
    )


@dataclass(frozen=True)
class InternalMagnetForceSolution:
    """Nonlinear internal field together with the magnet force."""

    magnetostatic_solution: AxisymmetricMagnetostaticSolution
    maxwell_stress_force: MaxwellStressForceResult


def solve_internal_magnet_force(
    grid: AxisymmetricGrid,
    shell_geometry: ShellGeometry,
    magnet_config: MagnetConfig,
    shell_bh_curve: TabulatedBHCurve,
    face_gap_m: float,
    solver_config: AxisymmetricNonlinearSolverConfig | None = None,
    stress_surface_air_layers: int = 1,
) -> InternalMagnetForceSolution:
    """Solve the nonlinear field and return the axial magnet force."""

    magnetostatic_solution = (
        solve_internal_axisymmetric_magnetostatics(
            grid,
            shell_geometry,
            magnet_config,
            shell_bh_curve,
            face_gap_m,
            solver_config,
        )
    )
    maxwell_stress_force = (
        compute_magnet_axial_force_from_maxwell_stress(
            grid,
            magnetostatic_solution.material_map,
            magnetostatic_solution.radial_flux_density_t,
            magnetostatic_solution.axial_flux_density_t,
            air_layers=stress_surface_air_layers,
        )
    )

    return InternalMagnetForceSolution(
        magnetostatic_solution=magnetostatic_solution,
        maxwell_stress_force=maxwell_stress_force,
    )
