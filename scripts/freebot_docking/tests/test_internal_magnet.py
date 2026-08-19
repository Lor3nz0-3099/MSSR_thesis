from math import pi

import pytest

from freebot_docking.config.geometry import ShellGeometry
from freebot_docking.config.magnet import (
    MagnetConfig,
    TabulatedBHCurve,
    pure_iron_bh_curve,
)

import numpy as np

from freebot_docking.physics.geometry import (
    compute_magnet_active_face_center,
    AxisymmetricGrid,
)
from freebot_docking.physics.state import MagnetState
from freebot_docking.physics.internal_magnet import (
    MU0_H_PER_M,
    axial_flux_density_gradient_t_per_m,
    axial_flux_density_t,
    equivalent_dipole_moment_am2,
    magnetization_magnitude_a_per_m,
    secant_reluctivity_with_saturation_extension_m_per_h,
    AxisymmetricMaterialRegion,
    build_internal_axisymmetric_material_map,
    build_initial_reluctivity_map_m_per_h,
    build_reluctivity_map_m_per_h,
    build_axial_remanence_map_t,
    flux_density_from_flux_function_t,
    build_flux_function_dirichlet_mask,
    build_axisymmetric_flux_coefficient_per_h,
    apply_axisymmetric_flux_operator_a,
    build_axisymmetric_remanence_source_a,
    solve_axisymmetric_flux_function,
    solve_internal_axisymmetric_magnetostatics,
    compute_magnet_axial_force_from_maxwell_stress,
    solve_internal_magnet_force,
)

from freebot_docking.config.simulation import (
    AxisymmetricGridConfig,
    AxisymmetricNonlinearSolverConfig,
)


def test_default_magnet_matches_freebot_dimensions() -> None:
    config = MagnetConfig()

    assert config.radius_m == pytest.approx(0.010)
    assert config.diameter_m == pytest.approx(0.020)
    assert config.length_m == pytest.approx(0.010)
    assert config.remanence_t == pytest.approx(1.47)

    assert config.pole_area_m2 == pytest.approx(pi * 0.010**2)
    assert config.volume_m3 == pytest.approx(pi * 0.010**2 * 0.010)
    assert config.half_length_m == pytest.approx(0.005)


@pytest.mark.parametrize(
    ("radius", "length"),
    [
        (0.0, 0.010),
        (0.010, 0.0),
        (-0.010, 0.010),
        (0.010, -0.010),
    ],
)
def test_magnet_rejects_nonpositive_dimensions(
    radius: float,
    length: float,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        MagnetConfig(
            radius_m=radius,
            length_m=length,
        )


def test_magnet_rejects_nonunit_axis() -> None:
    with pytest.raises(ValueError, match="unit vector"):
        MagnetConfig(active_axis_local=(0.0, 0.0, -2.0))


def test_magnet_rejects_invalid_remanence() -> None:
    with pytest.raises(ValueError, match="remanence"):
        MagnetConfig(remanence_t=0.0)

def stationary_magnet(
    center: list[float],
    axis: list[float],
) -> MagnetState:
    return MagnetState(
        center_world=center,
        axis_world=axis,
        carrier_com_world=center,
        carrier_linear_velocity_world=[0.0, 0.0, 0.0],
        carrier_angular_velocity_world=[0.0, 0.0, 0.0],
    )


def test_magnet_active_face_center() -> None:
    config = MagnetConfig()
    state = stationary_magnet(
        center=[1.0, 2.0, 3.0],
        axis=[0.0, 0.0, -1.0],
    )

    face_center = compute_magnet_active_face_center(state, config)

    np.testing.assert_allclose(
        face_center,
        [1.0, 2.0, 2.995],
    )


def test_magnet_state_rejects_nonunit_world_axis() -> None:
    with pytest.raises(ValueError, match="unit vector"):
        stationary_magnet(
            center=[0.0, 0.0, 0.0],
            axis=[2.0, 0.0, 0.0],
        )


def test_magnet_velocity_uses_carrier_rigid_motion() -> None:
    state = MagnetState(
        center_world=[0.0, 3.0, 0.0],
        axis_world=[1.0, 0.0, 0.0],
        carrier_com_world=[0.0, 0.0, 0.0],
        carrier_linear_velocity_world=[1.0, 0.0, 0.0],
        carrier_angular_velocity_world=[0.0, 0.0, 2.0],
    )

    velocity = state.velocity_at(state.center_world)

    np.testing.assert_allclose(
        velocity,
        [-5.0, 0.0, 0.0],
    )


def test_magnet_rejects_axis_with_wrong_dimension() -> None:
    with pytest.raises(ValueError, match="three"):
        MagnetConfig(active_axis_local=(0.0, -1.0))

def test_equivalent_dipole_moment_is_positive():
    config = MagnetConfig()

    assert equivalent_dipole_moment_am2(config) > 0.0


def test_axial_field_is_finite_at_active_face():
    config = MagnetConfig()

    field = axial_flux_density_t(0.0, config)

    assert field > 0.0


def test_axial_field_decreases_with_gap():
    config = MagnetConfig()

    fields = [
        axial_flux_density_t(gap, config)
        for gap in (0.0, 0.001, 0.005, 0.010, 0.050)
    ]

    assert fields == sorted(fields, reverse=True)


@pytest.mark.parametrize("gap", [-0.001, float("nan"), float("inf")])
def test_axial_field_rejects_invalid_gap(gap):
    with pytest.raises(ValueError):
        axial_flux_density_t(gap, MagnetConfig())

def test_axial_field_gradient_is_negative() -> None:
    gradient = axial_flux_density_gradient_t_per_m(
        0.009,
        MagnetConfig(),
    )

    assert gradient < 0.0


def test_axial_field_gradient_matches_finite_difference() -> None:
    config = MagnetConfig()
    gap = 0.009
    step = 1.0e-7

    numerical_gradient = (
        axial_flux_density_t(gap + step, config)
        - axial_flux_density_t(gap - step, config)
    ) / (2.0 * step)

    analytical_gradient = axial_flux_density_gradient_t_per_m(
        gap,
        config,
    )

    assert analytical_gradient == pytest.approx(
        numerical_gradient,
        rel=1.0e-8,
    )


def test_axial_field_gradient_decays_with_distance() -> None:
    config = MagnetConfig()

    near_gradient = axial_flux_density_gradient_t_per_m(
        0.009,
        config,
    )
    far_gradient = axial_flux_density_gradient_t_per_m(
        0.050,
        config,
    )

    assert abs(far_gradient) < abs(near_gradient)


@pytest.mark.parametrize("gap", [-0.001, float("nan"), float("inf")])
def test_axial_field_gradient_rejects_invalid_gap(gap: float) -> None:
    with pytest.raises(ValueError):
        axial_flux_density_gradient_t_per_m(
            gap,
            MagnetConfig(),
        )

def test_bh_curve_interpolates_between_samples() -> None:
    curve = TabulatedBHCurve(
        field_strength_a_per_m=(0.0, 100.0, 200.0),
        flux_density_t=(0.0, 0.10, 0.15),
    )

    assert curve.flux_density_for_field_strength_t(
        150.0
    ) == pytest.approx(0.125)


def test_bh_curve_returns_exact_sample() -> None:
    curve = TabulatedBHCurve(
        field_strength_a_per_m=(0.0, 100.0, 200.0),
        flux_density_t=(0.0, 0.10, 0.15),
    )

    assert curve.flux_density_for_field_strength_t(
        100.0
    ) == pytest.approx(0.10)


@pytest.mark.parametrize("field_strength", [-1.0, float("nan"), float("inf")])
def test_bh_curve_rejects_invalid_field_strength(
    field_strength: float,
) -> None:
    curve = TabulatedBHCurve(
        field_strength_a_per_m=(0.0, 100.0),
        flux_density_t=(0.0, 0.10),
    )

    with pytest.raises(ValueError):
        curve.flux_density_for_field_strength_t(field_strength)


def test_bh_curve_does_not_silently_extrapolate() -> None:
    curve = TabulatedBHCurve(
        field_strength_a_per_m=(0.0, 100.0),
        flux_density_t=(0.0, 0.10),
    )

    with pytest.raises(ValueError, match="outside"):
        curve.flux_density_for_field_strength_t(101.0)

def test_pure_iron_curve_matches_femm_samples() -> None:
    curve = pure_iron_bh_curve()

    assert curve.flux_density_for_field_strength_t(
        13.8984
    ) == pytest.approx(0.227065)

    assert curve.flux_density_for_field_strength_t(
        318310.0
    ) == pytest.approx(2.56)


def test_pure_iron_curve_represents_saturation() -> None:
    curve = pure_iron_bh_curve()

    low_field_slope = (
        curve.flux_density_t[1]
        / curve.field_strength_a_per_m[1]
    )

    high_field_slope = (
        curve.flux_density_t[-1]
        - curve.flux_density_t[-2]
    ) / (
        curve.field_strength_a_per_m[-1]
        - curve.field_strength_a_per_m[-2]
    )

    assert high_field_slope < low_field_slope

def test_pure_iron_has_zero_magnetization_without_field() -> None:
    magnetization = magnetization_magnitude_a_per_m(
        0.0,
        pure_iron_bh_curve(),
    )

    assert magnetization == pytest.approx(0.0)


def test_magnetization_satisfies_constitutive_relation() -> None:
    curve = pure_iron_bh_curve()
    field_strength = 123355.0

    magnetization = magnetization_magnitude_a_per_m(
        field_strength,
        curve,
    )

    reconstructed_flux_density = MU0_H_PER_M * (
        field_strength + magnetization
    )

    assert reconstructed_flux_density == pytest.approx(
        curve.flux_density_for_field_strength_t(
            field_strength
        )
    )


def test_pure_iron_magnetization_remains_finite_in_saturation() -> None:
    curve = pure_iron_bh_curve()

    magnetization = magnetization_magnitude_a_per_m(
        318310.0,
        curve,
    )

    assert magnetization == pytest.approx(
        2.56 / MU0_H_PER_M - 318310.0
    )
    assert magnetization > 0.0

def test_bh_curve_inverse_interpolates_between_samples() -> None:
    curve = TabulatedBHCurve(
        field_strength_a_per_m=(0.0, 100.0, 200.0),
        flux_density_t=(0.0, 0.10, 0.15),
    )

    assert curve.field_strength_for_flux_density_a_per_m(
        0.125
    ) == pytest.approx(150.0)


@pytest.mark.parametrize(
    "field_strength",
    [0.0, 100.0, 1000.0, 100000.0, 300000.0],
)
def test_pure_iron_bh_round_trip(field_strength: float) -> None:
    curve = pure_iron_bh_curve()

    flux_density = curve.flux_density_for_field_strength_t(
        field_strength
    )
    reconstructed_field_strength = (
        curve.field_strength_for_flux_density_a_per_m(
            flux_density
        )
    )

    assert reconstructed_field_strength == pytest.approx(
        field_strength
    )


def test_bh_inverse_does_not_silently_extrapolate() -> None:
    curve = pure_iron_bh_curve()

    with pytest.raises(ValueError, match="outside"):
        curve.field_strength_for_flux_density_a_per_m(2.57)

def test_bh_curve_reluctivity_at_origin_uses_first_segment() -> None:
    curve = pure_iron_bh_curve()

    expected = (
        curve.field_strength_a_per_m[1]
        / curve.flux_density_t[1]
    )

    assert curve.secant_reluctivity_m_per_h(
        0.0
    ) == pytest.approx(expected)


def test_bh_curve_reluctivity_satisfies_h_equals_nu_b() -> None:
    curve = pure_iron_bh_curve()
    flux_density = 2.29993

    reluctivity = curve.secant_reluctivity_m_per_h(
        flux_density
    )

    reconstructed_field_strength = (
        reluctivity * flux_density
    )

    assert reconstructed_field_strength == pytest.approx(
        curve.field_strength_for_flux_density_a_per_m(
            flux_density
        )
    )


def test_pure_iron_reluctivity_increases_toward_saturation() -> None:
    curve = pure_iron_bh_curve()

    low_field_reluctivity = (
        curve.secant_reluctivity_m_per_h(0.227065)
    )
    saturated_reluctivity = (
        curve.secant_reluctivity_m_per_h(2.56)
    )

    assert saturated_reluctivity > low_field_reluctivity


def test_reluctivity_extension_uses_vacuum_slope_after_saturation() -> None:
    curve = pure_iron_bh_curve()
    extended_flux_density = 3.0

    reluctivity = (
        secant_reluctivity_with_saturation_extension_m_per_h(
            extended_flux_density,
            curve,
        )
    )
    reconstructed_field = reluctivity * extended_flux_density
    expected_field = (
        curve.field_strength_a_per_m[-1]
        + (
            extended_flux_density - curve.flux_density_t[-1]
        ) / MU0_H_PER_M
    )

    assert reconstructed_field == pytest.approx(expected_field)
    assert reluctivity > curve.secant_reluctivity_m_per_h(2.56)

def test_default_magnet_has_ndfeb_recoil_permeability() -> None:
    config = MagnetConfig()

    assert config.recoil_relative_permeability == pytest.approx(
        1.05
    )


@pytest.mark.parametrize(
    "recoil_permeability",
    [0.0, -1.0, float("nan"), float("inf")],
)
def test_magnet_rejects_invalid_recoil_permeability(
    recoil_permeability: float,
) -> None:
    with pytest.raises(ValueError, match="recoil"):
        MagnetConfig(
            recoil_relative_permeability=recoil_permeability,
        )


def test_bh_curve_permeability_at_origin_uses_first_segment() -> None:
    curve = pure_iron_bh_curve()

    expected = (
        curve.flux_density_t[1]
        / curve.field_strength_a_per_m[1]
    )

    assert curve.secant_permeability_h_per_m(
        0.0
    ) == pytest.approx(expected)


def test_permeability_and_reluctivity_are_reciprocal() -> None:
    curve = pure_iron_bh_curve()
    field_strength = 123355.0

    flux_density = curve.flux_density_for_field_strength_t(
        field_strength
    )
    permeability = curve.secant_permeability_h_per_m(
        field_strength
    )
    reluctivity = curve.secant_reluctivity_m_per_h(
        flux_density
    )

    assert permeability * reluctivity == pytest.approx(1.0)

def test_axisymmetric_grid_computes_cell_steps() -> None:
    config = AxisymmetricGridConfig(
        radial_max_m=0.12,
        axial_min_m=-0.12,
        axial_max_m=0.12,
        radial_cells=240,
        axial_cells=480,
    )

    assert config.radial_step_m == pytest.approx(0.0005)
    assert config.axial_step_m == pytest.approx(0.0005)


@pytest.mark.parametrize(
    "radial_max",
    [0.0, -0.1, float("nan"), float("inf")],
)
def test_axisymmetric_grid_rejects_invalid_radial_extent(
    radial_max: float,
) -> None:
    with pytest.raises(ValueError, match="Radial"):
        AxisymmetricGridConfig(
            radial_max_m=radial_max,
            axial_min_m=-0.1,
            axial_max_m=0.1,
            radial_cells=10,
            axial_cells=20,
        )


def test_axisymmetric_grid_rejects_inverted_axial_limits() -> None:
    with pytest.raises(ValueError, match="smaller"):
        AxisymmetricGridConfig(
            radial_max_m=0.1,
            axial_min_m=0.1,
            axial_max_m=-0.1,
            radial_cells=10,
            axial_cells=20,
        )


@pytest.mark.parametrize(
    ("radial_cells", "axial_cells"),
    [
        (1, 20),
        (10, 1),
        (2.5, 20),
        (10, True),
    ],
)
def test_axisymmetric_grid_rejects_invalid_cell_counts(
    radial_cells: object,
    axial_cells: object,
) -> None:
    with pytest.raises(ValueError, match="cell count"):
        AxisymmetricGridConfig(
            radial_max_m=0.1,
            axial_min_m=-0.1,
            axial_max_m=0.1,
            radial_cells=radial_cells,
            axial_cells=axial_cells,
        )

def test_internal_material_map_contains_all_regions() -> None:
    grid = AxisymmetricGrid(
        AxisymmetricGridConfig(
            radial_max_m=0.08,
            axial_min_m=-0.08,
            axial_max_m=0.08,
            radial_cells=80,
            axial_cells=160,
        )
    )

    material_map = build_internal_axisymmetric_material_map(
        grid=grid,
        shell_geometry=ShellGeometry(),
        magnet_config=MagnetConfig(),
        face_gap_m=0.009,
    )

    assert material_map.shape == grid.cell_shape
    assert set(np.unique(material_map)) == {
        AxisymmetricMaterialRegion.AIR,
        AxisymmetricMaterialRegion.PERMANENT_MAGNET,
        AxisymmetricMaterialRegion.FERROMAGNETIC_SHELL,
    }


def test_internal_material_map_places_magnet_on_axis() -> None:
    shell = ShellGeometry()
    magnet = MagnetConfig()

    grid = AxisymmetricGrid(
        AxisymmetricGridConfig(
            radial_max_m=0.08,
            axial_min_m=-0.08,
            axial_max_m=0.08,
            radial_cells=80,
            axial_cells=160,
        )
    )

    material_map = build_internal_axisymmetric_material_map(
        grid,
        shell,
        magnet,
        face_gap_m=0.009,
    )

    magnet_center_z = (
        shell.inner_radius_m
        - 0.009
        - magnet.half_length_m
    )

    axial_index = int(
        np.argmin(
            np.abs(
                grid.axial_centers_m - magnet_center_z
            )
        )
    )

    assert material_map[
        axial_index,
        0,
    ] == AxisymmetricMaterialRegion.PERMANENT_MAGNET


def test_internal_material_map_is_read_only() -> None:
    grid = AxisymmetricGrid(
        AxisymmetricGridConfig(
            radial_max_m=0.08,
            axial_min_m=-0.08,
            axial_max_m=0.08,
            radial_cells=80,
            axial_cells=160,
        )
    )

    material_map = build_internal_axisymmetric_material_map(
        grid,
        ShellGeometry(),
        MagnetConfig(),
        face_gap_m=0.009,
    )

    with pytest.raises(ValueError):
        material_map[0, 0] = 99


def test_internal_material_map_rejects_intersecting_magnet() -> None:
    grid = AxisymmetricGrid(
        AxisymmetricGridConfig(
            radial_max_m=0.08,
            axial_min_m=-0.08,
            axial_max_m=0.08,
            radial_cells=80,
            axial_cells=160,
        )
    )

    with pytest.raises(ValueError, match="does not fit"):
        build_internal_axisymmetric_material_map(
            grid,
            ShellGeometry(),
            MagnetConfig(),
            face_gap_m=0.0,
        )

def test_initial_reluctivity_map_assigns_material_properties() -> None:
    grid = AxisymmetricGrid(
        AxisymmetricGridConfig(
            radial_max_m=0.08,
            axial_min_m=-0.08,
            axial_max_m=0.08,
            radial_cells=80,
            axial_cells=160,
        )
    )

    magnet = MagnetConfig()
    curve = pure_iron_bh_curve()

    material_map = build_internal_axisymmetric_material_map(
        grid,
        ShellGeometry(),
        magnet,
        face_gap_m=0.009,
    )

    reluctivity = build_initial_reluctivity_map_m_per_h(
        material_map,
        magnet,
        curve,
    )

    assert np.all(
        reluctivity[
            material_map == AxisymmetricMaterialRegion.AIR
        ] == pytest.approx(1.0 / MU0_H_PER_M)
    )

    assert np.all(
        reluctivity[
            material_map
            == AxisymmetricMaterialRegion.PERMANENT_MAGNET
        ] == pytest.approx(
            1.0
            / (
                MU0_H_PER_M
                * magnet.recoil_relative_permeability
            )
        )
    )

    assert np.all(
        reluctivity[
            material_map
            == AxisymmetricMaterialRegion.FERROMAGNETIC_SHELL
        ] == pytest.approx(
            curve.secant_reluctivity_m_per_h(0.0)
        )
    )

def test_initial_reluctivity_map_assigns_material_properties() -> None:
    grid = AxisymmetricGrid(
        AxisymmetricGridConfig(
            radial_max_m=0.08,
            axial_min_m=-0.08,
            axial_max_m=0.08,
            radial_cells=80,
            axial_cells=160,
        )
    )

    magnet = MagnetConfig()
    curve = pure_iron_bh_curve()

    material_map = build_internal_axisymmetric_material_map(
        grid,
        ShellGeometry(),
        magnet,
        face_gap_m=0.009,
    )

    reluctivity = build_initial_reluctivity_map_m_per_h(
        material_map,
        magnet,
        curve,
    )

    assert np.all(
        reluctivity[
            material_map == AxisymmetricMaterialRegion.AIR
        ] == pytest.approx(1.0 / MU0_H_PER_M)
    )

    assert np.all(
        reluctivity[
            material_map
            == AxisymmetricMaterialRegion.PERMANENT_MAGNET
        ] == pytest.approx(
            1.0
            / (
                MU0_H_PER_M
                * magnet.recoil_relative_permeability
            )
        )
    )

    assert np.all(
        reluctivity[
            material_map
            == AxisymmetricMaterialRegion.FERROMAGNETIC_SHELL
        ] == pytest.approx(
            curve.secant_reluctivity_m_per_h(0.0)
        )
    )

def test_reluctivity_map_updates_only_ferromagnetic_cells() -> None:
    material_map = np.array(
        [[
            AxisymmetricMaterialRegion.AIR,
            AxisymmetricMaterialRegion.PERMANENT_MAGNET,
            AxisymmetricMaterialRegion.FERROMAGNETIC_SHELL,
            AxisymmetricMaterialRegion.FERROMAGNETIC_SHELL,
        ]],
        dtype=np.uint8,
    )

    flux_density = np.array(
        [[0.5, 0.5, 0.227065, 2.56]],
        dtype=np.float64,
    )

    magnet = MagnetConfig()
    curve = pure_iron_bh_curve()

    reluctivity = build_reluctivity_map_m_per_h(
        material_map,
        flux_density,
        magnet,
        curve,
    )

    assert reluctivity[0, 0] == pytest.approx(
        1.0 / MU0_H_PER_M
    )
    assert reluctivity[0, 1] == pytest.approx(
        1.0
        / (
            MU0_H_PER_M
            * magnet.recoil_relative_permeability
        )
    )
    assert reluctivity[0, 2] == pytest.approx(
        curve.secant_reluctivity_m_per_h(0.227065)
    )
    assert reluctivity[0, 3] == pytest.approx(
        curve.secant_reluctivity_m_per_h(2.56)
    )

    assert reluctivity[0, 3] > reluctivity[0, 2]

def test_axial_remanence_exists_only_inside_magnet() -> None:
    material_map = np.array(
        [[
            AxisymmetricMaterialRegion.AIR,
            AxisymmetricMaterialRegion.PERMANENT_MAGNET,
            AxisymmetricMaterialRegion.FERROMAGNETIC_SHELL,
        ]],
        dtype=np.uint8,
    )

    magnet = MagnetConfig()

    remanence = build_axial_remanence_map_t(
        material_map,
        magnet,
    )

    np.testing.assert_allclose(
        remanence,
        [[0.0, magnet.remanence_t, 0.0]],
    )

    assert not remanence.flags.writeable

def test_flux_function_produces_uniform_axial_field() -> None:
    grid = AxisymmetricGrid(
        AxisymmetricGridConfig(
            radial_max_m=0.04,
            axial_min_m=-0.03,
            axial_max_m=0.03,
            radial_cells=4,
            axial_cells=6,
        )
    )

    expected_axial_field = 0.75

    radial_faces = grid.radial_faces_m[np.newaxis, :]

    flux_function = np.broadcast_to(
        0.5 * expected_axial_field * radial_faces**2,
        (
            grid.config.axial_cells + 1,
            grid.config.radial_cells + 1,
        ),
    )

    radial_field, axial_field = (
        flux_density_from_flux_function_t(
            grid,
            flux_function,
        )
    )

    np.testing.assert_allclose(
        radial_field,
        0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        axial_field,
        expected_axial_field,
        atol=1.0e-14,
    )

    assert radial_field.shape == grid.cell_shape
    assert axial_field.shape == grid.cell_shape
    assert not radial_field.flags.writeable
    assert not axial_field.flags.writeable

def test_flux_function_boundary_mask_marks_only_domain_edges() -> None:
    grid = AxisymmetricGrid(
        AxisymmetricGridConfig(
            radial_max_m=0.04,
            axial_min_m=-0.03,
            axial_max_m=0.03,
            radial_cells=4,
            axial_cells=6,
        )
    )

    boundary_mask = build_flux_function_dirichlet_mask(grid)

    assert boundary_mask.shape == (7, 5)

    assert np.all(boundary_mask[:, 0])
    assert np.all(boundary_mask[:, -1])
    assert np.all(boundary_mask[0, :])
    assert np.all(boundary_mask[-1, :])

    assert not np.any(boundary_mask[1:-1, 1:-1])
    assert not boundary_mask.flags.writeable


def test_axisymmetric_flux_coefficient_equals_reluctivity_over_radius() -> None:
    grid = AxisymmetricGrid(
        AxisymmetricGridConfig(
            radial_max_m=0.04,
            axial_min_m=-0.02,
            axial_max_m=0.02,
            radial_cells=4,
            axial_cells=2,
        )
    )

    reluctivity = np.full(
        grid.cell_shape,
        1.0 / MU0_H_PER_M,
        dtype=np.float64,
    )

    coefficient = build_axisymmetric_flux_coefficient_per_h(
        grid,
        reluctivity,
    )

    expected = (
        reluctivity
        / grid.radial_centers_m[np.newaxis, :]
    )

    np.testing.assert_allclose(coefficient, expected)

    assert coefficient.shape == grid.cell_shape
    assert np.all(coefficient > 0.0)
    assert not coefficient.flags.writeable


def test_axisymmetric_flux_operator_is_symmetric_and_positive() -> None:
    grid = AxisymmetricGrid(
        AxisymmetricGridConfig(
            radial_max_m=0.04,
            axial_min_m=-0.03,
            axial_max_m=0.03,
            radial_cells=5,
            axial_cells=6,
        )
    )
    coefficient = np.ones(grid.cell_shape, dtype=np.float64)
    shape = (grid.config.axial_cells + 1, grid.config.radial_cells + 1)
    boundary = build_flux_function_dirichlet_mask(grid)
    random = np.random.default_rng(7)
    first = random.normal(size=shape)
    second = random.normal(size=shape)
    first[boundary] = 0.0
    second[boundary] = 0.0

    first_operator = apply_axisymmetric_flux_operator_a(
        grid,
        coefficient,
        first,
    )
    second_operator = apply_axisymmetric_flux_operator_a(
        grid,
        coefficient,
        second,
    )

    assert np.sum(first * first_operator) > 0.0
    assert np.sum(first * second_operator) == pytest.approx(
        np.sum(second * first_operator),
        rel=1.0e-12,
        abs=1.0e-12,
    )
    assert np.all(first_operator[boundary] == 0.0)


def test_remanence_source_is_finite_and_internal() -> None:
    grid = AxisymmetricGrid(
        AxisymmetricGridConfig(
            radial_max_m=0.04,
            axial_min_m=-0.04,
            axial_max_m=0.04,
            radial_cells=4,
            axial_cells=8,
        )
    )
    reluctivity = np.full(
        grid.cell_shape,
        1.0 / MU0_H_PER_M,
        dtype=np.float64,
    )
    remanence = np.zeros(grid.cell_shape, dtype=np.float64)
    remanence[3:5, 0] = 1.47

    source = build_axisymmetric_remanence_source_a(
        grid,
        reluctivity,
        remanence,
    )
    boundary = build_flux_function_dirichlet_mask(grid)

    assert np.all(np.isfinite(source))
    assert np.any(source[~boundary] != 0.0)
    assert np.all(source[boundary] == 0.0)
    assert not source.flags.writeable


def test_linear_flux_solver_recovers_manufactured_solution() -> None:
    grid = AxisymmetricGrid(
        AxisymmetricGridConfig(
            radial_max_m=0.05,
            axial_min_m=-0.03,
            axial_max_m=0.03,
            radial_cells=5,
            axial_cells=6,
        )
    )
    coefficient = np.ones(grid.cell_shape, dtype=np.float64)
    shape = (grid.config.axial_cells + 1, grid.config.radial_cells + 1)
    radial = grid.radial_faces_m[np.newaxis, :]
    axial_fraction = (
        (grid.axial_faces_m - grid.config.axial_min_m)
        / (grid.config.axial_max_m - grid.config.axial_min_m)
    )[:, np.newaxis]
    exact_flux = (
        np.sin(np.pi * radial / grid.config.radial_max_m)
        * np.sin(np.pi * axial_fraction)
    )
    assert exact_flux.shape == shape

    source = apply_axisymmetric_flux_operator_a(
        grid,
        coefficient,
        exact_flux,
    )
    result = solve_axisymmetric_flux_function(
        grid,
        coefficient,
        source,
        relative_tolerance=1.0e-12,
        max_iterations=200,
    )

    np.testing.assert_allclose(
        result.flux_function_wb,
        exact_flux,
        rtol=1.0e-11,
        atol=1.0e-12,
    )
    assert result.iterations > 0
    assert result.relative_residual <= 1.0e-12
    assert not result.flux_function_wb.flags.writeable


def test_axisymmetric_solver_matches_finite_cylinder_axis_field() -> None:
    grid = AxisymmetricGrid(
        AxisymmetricGridConfig(
            radial_max_m=0.08,
            axial_min_m=-0.08,
            axial_max_m=0.08,
            radial_cells=80,
            axial_cells=160,
        )
    )
    magnet = MagnetConfig()
    radial, axial = np.meshgrid(
        grid.radial_centers_m,
        grid.axial_centers_m,
    )
    material_map = np.full(
        grid.cell_shape,
        AxisymmetricMaterialRegion.AIR,
        dtype=np.uint8,
    )
    magnet_mask = (
        (radial <= magnet.radius_m)
        & (np.abs(axial) <= magnet.half_length_m)
    )
    material_map[magnet_mask] = (
        AxisymmetricMaterialRegion.PERMANENT_MAGNET
    )
    reluctivity = build_initial_reluctivity_map_m_per_h(
        material_map,
        magnet,
        pure_iron_bh_curve(),
    )
    remanence = build_axial_remanence_map_t(
        material_map,
        magnet,
    )
    coefficient = build_axisymmetric_flux_coefficient_per_h(
        grid,
        reluctivity,
    )
    source = build_axisymmetric_remanence_source_a(
        grid,
        reluctivity,
        remanence,
    )

    result = solve_axisymmetric_flux_function(
        grid,
        coefficient,
        source,
        relative_tolerance=1.0e-9,
        max_iterations=1000,
    )
    radial_field, axial_field = flux_density_from_flux_function_t(
        grid,
        result.flux_function_wb,
    )

    requested_gap = 0.0095
    requested_axial_position = (
        magnet.half_length_m + requested_gap
    )
    axial_index = int(
        np.argmin(
            np.abs(
                grid.axial_centers_m
                - requested_axial_position
            )
        )
    )
    sampled_gap = (
        grid.axial_centers_m[axial_index]
        - magnet.half_length_m
    )

    assert axial_field[axial_index, 0] == pytest.approx(
        axial_flux_density_t(sampled_gap, magnet),
        rel=0.04,
    )

    isolated_force = (
        compute_magnet_axial_force_from_maxwell_stress(
            grid,
            material_map,
            radial_field,
            axial_field,
            air_layers=2,
        )
    )
    assert isolated_force.axial_force_n == pytest.approx(
        0.0,
        abs=1.0e-10,
    )


def test_nonlinear_shell_solution_produces_contour_stable_force() -> None:
    grid = AxisymmetricGrid(
        AxisymmetricGridConfig(
            radial_max_m=0.08,
            axial_min_m=-0.08,
            axial_max_m=0.08,
            radial_cells=80,
            axial_cells=160,
        )
    )
    shell = ShellGeometry()
    magnet = MagnetConfig()
    curve = pure_iron_bh_curve()
    controls = AxisymmetricNonlinearSolverConfig(
        relative_tolerance=1.0e-3,
        max_iterations=50,
        relaxation_factor=0.3,
        linear_relative_tolerance=1.0e-8,
        linear_max_iterations=1000,
    )

    force_solution = solve_internal_magnet_force(
        grid,
        shell,
        magnet,
        curve,
        face_gap_m=0.009,
        solver_config=controls,
        stress_surface_air_layers=2,
    )
    solution = force_solution.magnetostatic_solution

    shell_mask = (
        solution.material_map
        == AxisymmetricMaterialRegion.FERROMAGNETIC_SHELL
    )
    initial_reluctivity = build_initial_reluctivity_map_m_per_h(
        solution.material_map,
        magnet,
        curve,
    )
    assert solution.nonlinear_iterations <= controls.max_iterations
    assert solution.nonlinear_relative_change <= (
        controls.relative_tolerance
    )
    assert np.any(
        solution.reluctivity_map_m_per_h[shell_mask]
        != initial_reluctivity[shell_mask]
    )

    forces = np.array(
        [
            compute_magnet_axial_force_from_maxwell_stress(
                grid,
                solution.material_map,
                solution.radial_flux_density_t,
                solution.axial_flux_density_t,
                air_layers=air_layers,
            ).axial_force_n
            for air_layers in (1, 2, 3)
        ]
    )

    assert np.all(forces > 0.0)
    assert np.ptp(forces) / np.mean(forces) < 0.02
    assert (
        force_solution.maxwell_stress_force.axial_force_n
        == pytest.approx(forces[1])
    )
