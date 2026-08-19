import pytest

from freebot_docking.config.geometry import RunningGearGeometry, ShellGeometry

import numpy as np

from freebot_docking.physics.state import ShellState, MagnetState
from freebot_docking.config.simulation import (
    AxisymmetricGridConfig,
)
from freebot_docking.physics.geometry import (
    AxisymmetricGrid,
    compute_magnet_inner_shell_geometry,
    compute_shell_pair_geometry,
)
from freebot_docking.config.magnet import MagnetConfig
from freebot_docking.isaac.materials import IsaacMaterialConfig

def test_default_shell_geometry_matches_fitted_cad() -> None:
    geometry = ShellGeometry()

    assert geometry.outer_radius_m == pytest.approx(0.0633472)
    assert geometry.inner_radius_m == pytest.approx(0.0613472)
    assert geometry.thickness_m == pytest.approx(0.002)


def test_running_gear_preloads_tires_and_keeps_casters_in_contact() -> None:
    geometry = RunningGearGeometry()

    assert geometry.wheel_nominal_clearance_m == 0.0
    assert geometry.tire_precompression_m == pytest.approx(0.0009)
    assert geometry.tire_collision_radius_m == pytest.approx(
        geometry.tire_outer_radius_m + 0.0009
    )
    assert geometry.caster_nominal_clearance_m == 0.0
    assert geometry.caster_precompression_m == pytest.approx(0.0001)
    assert geometry.caster_collision_radius_m == pytest.approx(
        geometry.caster_ball_radius_m + 0.0001
    )


def test_running_gear_rejects_nonphysical_tire_precompression() -> None:
    with pytest.raises(ValueError, match="precompression"):
        RunningGearGeometry(tire_precompression_m=-0.001)

    with pytest.raises(ValueError, match="smaller"):
        RunningGearGeometry(tire_precompression_m=0.02)


def test_running_gear_validates_optional_positive_caster_clearance() -> None:
    geometry = RunningGearGeometry(caster_nominal_clearance_m=0.002)
    assert geometry.caster_contact_offset_m < geometry.caster_nominal_clearance_m

    with pytest.raises(ValueError, match="contact offset"):
        RunningGearGeometry(caster_nominal_clearance_m=0.0001)


def test_running_gear_rejects_nonphysical_caster_precompression() -> None:
    with pytest.raises(ValueError, match="Caster precompression"):
        RunningGearGeometry(caster_precompression_m=-0.001)

    with pytest.raises(ValueError, match="smaller"):
        RunningGearGeometry(caster_precompression_m=0.005)


def test_default_wheel_material_uses_force_based_tire_compliance() -> None:
    material = IsaacMaterialConfig()

    assert material.wheel_contact_stiffness_n_per_m == pytest.approx(8_000.0)
    assert material.wheel_contact_damping_n_s_per_m == pytest.approx(40.0)
    assert material.caster_contact_stiffness_n_per_m == pytest.approx(2_000.0)
    assert material.caster_contact_damping_n_s_per_m == pytest.approx(15.0)
    assert material.ground_static_friction == pytest.approx(1.25)
    assert material.ground_dynamic_friction == pytest.approx(1.00)


def test_wheel_material_rejects_negative_compliance() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        IsaacMaterialConfig(wheel_contact_stiffness_n_per_m=-1.0)


def test_shell_geometry_converts_values_to_float() -> None:
    geometry = ShellGeometry(
        outer_radius_m=1,
        inner_radius_m=0.5,
        center_from_body_origin_m=(0, 0, 0),
    )

    assert isinstance(geometry.outer_radius_m, float)
    assert isinstance(geometry.inner_radius_m, float)
    assert geometry.center_from_body_origin_m == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    ("outer_radius", "inner_radius"),
    [
        (0.0, 0.05),
        (0.06, 0.0),
        (-0.06, 0.05),
        (0.06, -0.05),
    ],
)
def test_shell_geometry_rejects_nonpositive_radii(
    outer_radius: float,
    inner_radius: float,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        ShellGeometry(
            outer_radius_m=outer_radius,
            inner_radius_m=inner_radius,
        )


def test_shell_geometry_rejects_inverted_radii() -> None:
    with pytest.raises(ValueError, match="smaller"):
        ShellGeometry(
            outer_radius_m=0.06,
            inner_radius_m=0.07,
        )


def test_shell_geometry_rejects_invalid_center_offset() -> None:
    with pytest.raises(ValueError, match="three"):
        ShellGeometry(
            center_from_body_origin_m=(0.0, 0.0),
        )

def test_shell_state_copies_input_vectors() -> None:
    center = np.array([1.0, 2.0, 3.0])

    state = ShellState(
        center_world=center,
        com_world=[1.0, 2.0, 3.0],
        linear_velocity_world=[0.0, 0.0, 0.0],
        angular_velocity_world=[0.0, 0.0, 0.0],
    )

    center[0] = 100.0

    np.testing.assert_array_equal(
        state.center_world,
        [1.0, 2.0, 3.0],
    )


def test_shell_velocity_at_world_point() -> None:
    state = ShellState(
        center_world=[0.0, 0.0, 0.0],
        com_world=[0.0, 0.0, 0.0],
        linear_velocity_world=[1.0, 0.0, 0.0],
        angular_velocity_world=[0.0, 0.0, 2.0],
    )

    velocity = state.velocity_at([0.0, 3.0, 0.0])

    np.testing.assert_allclose(
        velocity,
        [-5.0, 0.0, 0.0],
    )

def stationary_shell(center: list[float]) -> ShellState:
    return ShellState(
        center_world=center,
        com_world=center,
        linear_velocity_world=[0.0, 0.0, 0.0],
        angular_velocity_world=[0.0, 0.0, 0.0],
    )

def test_shell_pair_geometry_for_separated_shells() -> None:
    geometry = ShellGeometry(
        outer_radius_m=0.06,
        inner_radius_m=0.05,
        center_from_body_origin_m=(0.0, 0.0, 0.0),
    )

    pair = compute_shell_pair_geometry(
        stationary_shell([0.0, 0.0, 0.0]),
        geometry,
        stationary_shell([0.15, 0.0, 0.0]),
        geometry,
    )

    assert pair.center_distance_m == pytest.approx(0.15)
    assert pair.signed_gap_m == pytest.approx(0.03)

    np.testing.assert_allclose(
        pair.normal_first_to_second_world,
        [1.0, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        pair.point_on_first_world,
        [0.06, 0.0, 0.0],
    )
    np.testing.assert_allclose(
        pair.point_on_second_world,
        [0.09, 0.0, 0.0],
    )


def test_shell_pair_geometry_preserves_penetration() -> None:
    geometry = ShellGeometry(
        outer_radius_m=0.06,
        inner_radius_m=0.05,
    )

    pair = compute_shell_pair_geometry(
        stationary_shell([0.0, 0.0, 0.0]),
        geometry,
        stationary_shell([0.11, 0.0, 0.0]),
        geometry,
    )

    assert pair.signed_gap_m == pytest.approx(-0.01)


def test_shell_pair_geometry_rejects_coincident_centers() -> None:
    geometry = ShellGeometry()

    with pytest.raises(ValueError, match="coincident"):
        compute_shell_pair_geometry(
            stationary_shell([0.0, 0.0, 0.0]),
            geometry,
            stationary_shell([0.0, 0.0, 0.0]),
            geometry,
        )

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


def test_magnet_inner_shell_geometry_reproduces_nine_mm_gap() -> None:
    shell_geometry = ShellGeometry()
    magnet_config = MagnetConfig()

    magnet_center_radius = (
        shell_geometry.inner_radius_m
        - magnet_config.half_length_m
        - 0.009
    )

    result = compute_magnet_inner_shell_geometry(
        shell_state=stationary_shell([0.0, 0.0, 0.0]),
        shell_geometry=shell_geometry,
        magnet_state=stationary_magnet(
            center=[0.0, 0.0, magnet_center_radius],
            axis=[0.0, 0.0, 1.0],
        ),
        magnet_config=magnet_config,
    )

    assert result.axial_gap_m == pytest.approx(0.009)
    assert result.radial_gap_m == pytest.approx(0.009)
    assert result.alignment_cosine == pytest.approx(1.0)

    np.testing.assert_allclose(
        result.point_on_inner_shell_world,
        [0.0, 0.0, shell_geometry.inner_radius_m],
    )


def test_magnet_inner_shell_geometry_rejects_inward_axis() -> None:
    with pytest.raises(ValueError, match="point toward"):
        compute_magnet_inner_shell_geometry(
            shell_state=stationary_shell([0.0, 0.0, 0.0]),
            shell_geometry=ShellGeometry(),
            magnet_state=stationary_magnet(
                center=[0.0, 0.0, 0.04],
                axis=[0.0, 0.0, -1.0],
            ),
            magnet_config=MagnetConfig(),
        )

def test_axisymmetric_grid_builds_faces_and_centers() -> None:
    grid = AxisymmetricGrid(
        AxisymmetricGridConfig(
            radial_max_m=0.04,
            axial_min_m=-0.02,
            axial_max_m=0.02,
            radial_cells=4,
            axial_cells=4,
        )
    )

    np.testing.assert_allclose(
        grid.radial_faces_m,
        [0.0, 0.01, 0.02, 0.03, 0.04],
    )
    np.testing.assert_allclose(
        grid.radial_centers_m,
        [0.005, 0.015, 0.025, 0.035],
    )
    np.testing.assert_allclose(
        grid.axial_faces_m,
        [-0.02, -0.01, 0.0, 0.01, 0.02],
    )
    np.testing.assert_allclose(
        grid.axial_centers_m,
        [-0.015, -0.005, 0.005, 0.015],
    )

    assert grid.cell_shape == (4, 4)
    assert grid.cell_count == 16


def test_axisymmetric_grid_does_not_place_center_on_axis() -> None:
    grid = AxisymmetricGrid(
        AxisymmetricGridConfig(
            radial_max_m=0.02,
            axial_min_m=-0.01,
            axial_max_m=0.01,
            radial_cells=2,
            axial_cells=2,
        )
    )

    assert grid.radial_faces_m[0] == pytest.approx(0.0)
    assert grid.radial_centers_m[0] > 0.0


def test_axisymmetric_grid_coordinates_are_read_only() -> None:
    grid = AxisymmetricGrid(
        AxisymmetricGridConfig(
            radial_max_m=0.02,
            axial_min_m=-0.01,
            axial_max_m=0.01,
            radial_cells=2,
            axial_cells=2,
        )
    )

    with pytest.raises(ValueError):
        grid.radial_centers_m[0] = 1.0
