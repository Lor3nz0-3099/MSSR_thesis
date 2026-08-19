import numpy as np
import pytest

from freebot_docking.config.geometry import (
    RunningGearGeometry,
    ShellGeometry,
    WheelRadialComplianceConfig,
)
from freebot_docking.config.magnet import MagnetConfig
from freebot_docking.config.simulation import ShellContactFrictionConfig
from freebot_docking.control.wheel_drive import (
    WheelDriveConfig,
    WheelVelocityTargets,
    apply_climb_heading_correction,
    dc_motor_torque_limit_nm,
    signed_heading_error_rad,
    twist_to_wheel_targets,
)
from freebot_docking.cli import _default_usd_path, build_argument_parser
from freebot_docking.diagnostics.contacts import (
    evaluate_freebot_friction_diagnostics,
    figure9_balance_residual,
    paper_required_connection_friction,
)
from freebot_docking.physics.external_magnet import (
    compute_external_magnetic_interaction,
    fit_anchored_exponential_force_curve,
    freebot_figure4_force_curve,
    freebot_figure5_angular_force_curve,
)
from freebot_docking.physics.state import MagnetState, ShellState
from freebot_docking.isaac.module_handles import quaternion_rotate_wxyz
from freebot_docking.isaac.debug_draw import arrow_segments
from freebot_docking.isaac.materials import IsaacMaterialConfig
from freebot_docking.isaac.stage_builder import IsaacStageConfig
from freebot_docking.scenarios.two_module_docking import (
    ContactRegime,
    compute_internal_magnetic_preload_interaction,
    compute_two_module_docking_step,
    update_shell_contact_friction,
)


def stationary_shell(center: list[float]) -> ShellState:
    return ShellState(
        center_world=center,
        com_world=center,
        linear_velocity_world=[0.0, 0.0, 0.0],
        angular_velocity_world=[0.0, 0.0, 0.0],
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


def moving_shell(
    center: list[float],
    linear_velocity: list[float],
    angular_velocity: list[float],
) -> ShellState:
    return ShellState(
        center_world=center,
        com_world=center,
        linear_velocity_world=linear_velocity,
        angular_velocity_world=angular_velocity,
    )


def test_freebot_figure4_curve_matches_digitized_anchors() -> None:
    curve = freebot_figure4_force_curve()

    assert curve.attraction_force_n(0.0) == pytest.approx(22.6)
    assert curve.attraction_force_n(0.002) == pytest.approx(12.42)
    assert curve.attraction_force_n(0.010) == pytest.approx(1.35)
    assert curve.attraction_force_n(0.030) == pytest.approx(0.0)
    assert curve.attraction_force_n(0.031) == pytest.approx(0.0)


def test_figure4_anchored_exponential_fit_reports_residuals() -> None:
    fit = fit_anchored_exponential_force_curve(
        freebot_figure4_force_curve()
    )

    assert fit.contact_force_n == pytest.approx(22.6)
    assert fit.decay_length_m == pytest.approx(0.003920384504310472)
    assert fit.root_mean_square_error_n == pytest.approx(
        0.6778407304505231
    )
    assert fit.maximum_absolute_error_n == pytest.approx(
        1.1491097110581823
    )


def test_freebot_figure5_curve_matches_digitized_anchors() -> None:
    curve = freebot_figure5_angular_force_curve()

    assert curve.components_n(0.0) == pytest.approx((22.6, 0.0))
    assert curve.components_n(20.0) == pytest.approx((7.64, 1.40))
    assert curve.components_n(90.0) == pytest.approx((1.75, 0.10))

    with pytest.raises(ValueError, match=r"\[0, 90\]"):
        curve.components_n(90.0 + 1.0e-9)


def test_aligned_external_interaction_matches_figure4() -> None:
    geometry = ShellGeometry()
    magnet = MagnetConfig()
    center_distance = 2.0 * geometry.outer_radius_m + 0.002
    magnet_center = geometry.inner_radius_m - 0.009 - magnet.half_length_m

    interaction = compute_external_magnetic_interaction(
        stationary_shell([0.0, 0.0, 0.0]),
        geometry,
        stationary_shell([center_distance, 0.0, 0.0]),
        geometry,
        stationary_magnet([magnet_center, 0.0, 0.0], [1.0, 0.0, 0.0]),
        magnet,
    )

    assert interaction.lifting_angle_deg == pytest.approx(0.0)
    assert interaction.parallel_force_n == pytest.approx(12.42)
    assert interaction.perpendicular_force_n == pytest.approx(0.0)
    np.testing.assert_allclose(
        interaction.force_on_active_world,
        [12.42, 0.0, 0.0],
    )


def test_angled_interaction_scales_figure5_with_distance() -> None:
    geometry = ShellGeometry()
    magnet = MagnetConfig()
    center_distance = 2.0 * geometry.outer_radius_m + 0.002
    angle = np.deg2rad(20.0)
    axis = np.array([np.cos(angle), np.sin(angle), 0.0])
    magnet_radius = geometry.inner_radius_m - 0.009 - magnet.half_length_m

    interaction = compute_external_magnetic_interaction(
        stationary_shell([0.0, 0.0, 0.0]),
        geometry,
        stationary_shell([center_distance, 0.0, 0.0]),
        geometry,
        stationary_magnet(list(magnet_radius * axis), list(axis)),
        magnet,
    )

    expected_parallel = (
        freebot_figure5_angular_force_curve().components_n(20.0)[0]
    )
    expected_perpendicular = (
        freebot_figure5_angular_force_curve().components_n(20.0)[1]
    )
    scale = 12.42 / 22.6
    assert interaction.lifting_angle_deg == pytest.approx(
        20.0
    )
    assert interaction.parallel_curve_angle_deg == pytest.approx(20.0)
    assert interaction.parallel_force_n == pytest.approx(
        expected_parallel * scale
    )
    assert interaction.perpendicular_force_n == pytest.approx(
        expected_perpendicular * scale
    )
    assert interaction.force_on_active_world[1] < 0.0


def test_small_angle_uses_the_same_measured_angle_for_both_components() -> None:
    geometry = ShellGeometry()
    magnet = MagnetConfig()
    center_distance = 2.0 * geometry.outer_radius_m
    angle = np.deg2rad(4.0)
    axis = np.array([np.cos(angle), np.sin(angle), 0.0])
    magnet_radius = geometry.inner_radius_m - 0.009 - magnet.half_length_m

    interaction = compute_external_magnetic_interaction(
        stationary_shell([0.0, 0.0, 0.0]),
        geometry,
        stationary_shell([center_distance, 0.0, 0.0]),
        geometry,
        stationary_magnet(list(magnet_radius * axis), list(axis)),
        magnet,
    )

    expected_parallel, expected_perpendicular = (
        freebot_figure5_angular_force_curve().components_n(4.0)
    )
    assert interaction.parallel_curve_angle_deg == pytest.approx(4.0)
    assert interaction.parallel_force_n == pytest.approx(expected_parallel)
    assert interaction.perpendicular_force_n == pytest.approx(
        expected_perpendicular
    )
    assert interaction.perpendicular_force_n > 0.0
    assert interaction.force_on_active_world[1] < 0.0


def test_raised_magnet_transverse_force_generates_positive_climb_moment() -> None:
    """Protect the Fig. 5 sign convention against another inversion."""

    geometry = ShellGeometry()
    magnet = MagnetConfig()
    angle = np.deg2rad(20.0)
    axis = np.array([np.cos(angle), 0.0, np.sin(angle)])
    magnet_radius = geometry.inner_radius_m - 0.009 - magnet.half_length_m

    interaction = compute_external_magnetic_interaction(
        stationary_shell([0.0, 0.0, 0.0]),
        geometry,
        stationary_shell([2.0 * geometry.outer_radius_m, 0.0, 0.0]),
        geometry,
        stationary_magnet(list(magnet_radius * axis), list(axis)),
        magnet,
    )

    # +Y is the climb rotation: +Y x (-X) points upward around the passive
    # sphere.  Fig. 5's transverse force pulls the raised magnet downward
    # toward the contact line, producing a positive moment about +Y.
    assert interaction.force_on_active_world[2] < 0.0
    assert interaction.active_carrier_wrench.expressed_at(
        [0.0, 0.0, 0.0]
    ).torque[1] > 0.0


def test_passive_magnetized_patch_follows_magnet_position() -> None:
    geometry = ShellGeometry()
    magnet = MagnetConfig()
    passive_center = np.array(
        [2.0 * geometry.outer_radius_m, 0.0, 0.0]
    )
    magnet_radius = (
        geometry.inner_radius_m - 0.009 - magnet.half_length_m
    )

    aligned = compute_external_magnetic_interaction(
        stationary_shell([0.0, 0.0, 0.0]),
        geometry,
        stationary_shell(list(passive_center)),
        geometry,
        stationary_magnet(
            [magnet_radius, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ),
        magnet,
    )
    angle = np.deg2rad(30.0)
    axis = np.array([np.cos(angle), np.sin(angle), 0.0])
    moved = compute_external_magnetic_interaction(
        stationary_shell([0.0, 0.0, 0.0]),
        geometry,
        stationary_shell(list(passive_center)),
        geometry,
        stationary_magnet(list(magnet_radius * axis), list(axis)),
        magnet,
    )

    # The resultant slopes back toward the centre line but starts at the
    # raised magnet face, so the corresponding passive patch remains mobile
    # on the raised hemisphere rather than being fixed at the central point.
    assert moved.passive_surface_point_world[1] > (
        aligned.passive_surface_point_world[1]
    )
    assert moved.lifting_angle_deg == pytest.approx(30.0)
    np.testing.assert_allclose(
        np.linalg.norm(
            moved.passive_surface_point_world - passive_center
        ),
        geometry.outer_radius_m,
    )
    np.testing.assert_allclose(
        np.linalg.norm(moved.passive_surface_normal_world),
        1.0,
    )


def test_external_force_pair_has_zero_global_wrench() -> None:
    geometry = ShellGeometry()
    magnet = MagnetConfig()
    center_distance = 2.0 * geometry.outer_radius_m
    angle = np.deg2rad(35.0)
    axis = np.array([np.cos(angle), np.sin(angle), 0.0])
    magnet_radius = geometry.inner_radius_m - 0.009 - magnet.half_length_m

    interaction = compute_external_magnetic_interaction(
        stationary_shell([0.0, 0.0, 0.0]),
        geometry,
        stationary_shell([center_distance, 0.0, 0.0]),
        geometry,
        stationary_magnet(list(magnet_radius * axis), list(axis)),
        magnet,
    )
    total = (
        interaction.active_carrier_wrench.expressed_at([0.0, 0.0, 0.0])
        + interaction.passive_shell_wrench.expressed_at([0.0, 0.0, 0.0])
    )

    np.testing.assert_allclose(total.force, np.zeros(3), atol=1.0e-12)
    np.testing.assert_allclose(total.torque, np.zeros(3), atol=1.0e-12)

    shell_total = (
        interaction.active_shell_wrench.expressed_at([0.0, 0.0, 0.0])
        + interaction.passive_shell_wrench.expressed_at([0.0, 0.0, 0.0])
    )
    np.testing.assert_allclose(shell_total.force, np.zeros(3), atol=1.0e-12)
    np.testing.assert_allclose(shell_total.torque, np.zeros(3), atol=1.0e-12)


def test_active_shell_patch_preserves_the_external_resultant_wrench() -> None:
    geometry = ShellGeometry()
    magnet = MagnetConfig()
    angle = np.deg2rad(30.0)
    axis = np.array([np.cos(angle), np.sin(angle), 0.0])
    magnet_radius = geometry.inner_radius_m - 0.009 - magnet.half_length_m

    interaction = compute_external_magnetic_interaction(
        stationary_shell([0.0, 0.0, 0.0]),
        geometry,
        stationary_shell([2.0 * geometry.outer_radius_m, 0.0, 0.0]),
        geometry,
        stationary_magnet(list(magnet_radius * axis), list(axis)),
        magnet,
    )

    assert np.linalg.norm(interaction.active_surface_point_world) == pytest.approx(
        geometry.outer_radius_m
    )
    line = (
        interaction.active_surface_point_world
        - interaction.interaction_point_world
    )
    np.testing.assert_allclose(
        np.cross(line, interaction.force_on_active_world),
        np.zeros(3),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        interaction.active_shell_wrench.expressed_at([0.0, 0.0, 0.0]).force,
        interaction.active_carrier_wrench.expressed_at([0.0, 0.0, 0.0]).force,
    )
    np.testing.assert_allclose(
        interaction.active_shell_wrench.expressed_at([0.0, 0.0, 0.0]).torque,
        interaction.active_carrier_wrench.expressed_at([0.0, 0.0, 0.0]).torque,
        atol=1.0e-12,
    )


def test_external_resultant_line_terminates_on_mobile_passive_patch() -> None:
    geometry = ShellGeometry()
    magnet = MagnetConfig()
    center_distance = 2.0 * geometry.outer_radius_m
    angle = np.deg2rad(35.0)
    axis = np.array([np.cos(angle), np.sin(angle), 0.0])
    magnet_radius = geometry.inner_radius_m - 0.009 - magnet.half_length_m
    passive_center = np.array([center_distance, 0.0, 0.0])

    interaction = compute_external_magnetic_interaction(
        stationary_shell([0.0, 0.0, 0.0]),
        geometry,
        stationary_shell(list(passive_center)),
        geometry,
        stationary_magnet(list(magnet_radius * axis), list(axis)),
        magnet,
    )

    line = (
        interaction.passive_surface_point_world
        - interaction.interaction_point_world
    )
    assert interaction.line_of_action_valid
    np.testing.assert_allclose(
        np.cross(line, interaction.force_on_active_world),
        np.zeros(3),
        atol=1.0e-12,
    )
    assert np.dot(line, interaction.force_on_active_world) > 0.0
    assert np.linalg.norm(
        interaction.passive_surface_point_world - passive_center
    ) == pytest.approx(geometry.outer_radius_m)


def test_magnet_on_far_hemisphere_does_not_extrapolate_unmeasured_force() -> None:
    geometry = ShellGeometry()
    magnet = MagnetConfig()
    center_distance = 2.0 * geometry.outer_radius_m
    magnet_radius = geometry.inner_radius_m - 0.009 - magnet.half_length_m

    interaction = compute_external_magnetic_interaction(
        stationary_shell([0.0, 0.0, 0.0]),
        geometry,
        stationary_shell([center_distance, 0.0, 0.0]),
        geometry,
        stationary_magnet([-magnet_radius, 0.0, 0.0], [-1.0, 0.0, 0.0]),
        magnet,
    )

    assert not interaction.in_angular_range
    assert interaction.parallel_force_n == 0.0
    assert interaction.perpendicular_force_n == 0.0
    np.testing.assert_allclose(interaction.force_on_active_world, np.zeros(3))


def test_shell_contact_is_free_while_shells_are_separated() -> None:
    geometry = ShellGeometry()
    distance = 2.0 * geometry.outer_radius_m + 0.002

    contact = update_shell_contact_friction(
        stationary_shell([0.0, 0.0, 0.0]),
        geometry,
        moving_shell(
            [distance, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ),
        geometry,
        normal_load_n=12.42,
        time_step_s=0.001,
    )

    assert contact.regime is ContactRegime.FREE
    np.testing.assert_allclose(contact.force_on_first_world, np.zeros(3))


def test_shell_contact_sticks_below_static_coulomb_limit() -> None:
    geometry = ShellGeometry()
    distance = 2.0 * geometry.outer_radius_m

    contact = update_shell_contact_friction(
        stationary_shell([0.0, 0.0, 0.0]),
        geometry,
        moving_shell(
            [distance, 0.0, 0.0],
            [0.0, 0.001, 0.0],
            [0.0, 0.0, 0.0],
        ),
        geometry,
        normal_load_n=22.6,
        time_step_s=0.01,
    )

    assert contact.regime is ContactRegime.STICK
    assert 0.0 < contact.force_on_first_world[1] < 24.86
    np.testing.assert_allclose(
        contact.force_on_first_world + contact.force_on_second_world,
        np.zeros(3),
        atol=1.0e-12,
    )


def test_shell_contact_slips_at_dynamic_coulomb_limit() -> None:
    geometry = ShellGeometry()
    distance = 2.0 * geometry.outer_radius_m
    parameters = ShellContactFrictionConfig(
        static_friction_coefficient=1.1,
        dynamic_friction_coefficient=1.0,
    )

    contact = update_shell_contact_friction(
        stationary_shell([0.0, 0.0, 0.0]),
        geometry,
        moving_shell(
            [distance, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 0.0],
        ),
        geometry,
        normal_load_n=22.6,
        time_step_s=0.01,
        config=parameters,
    )

    assert contact.regime is ContactRegime.SLIP
    assert np.linalg.norm(contact.force_on_first_world) == pytest.approx(22.6)
    assert contact.force_on_first_world[1] > 0.0


def test_counter_rotating_shells_have_zero_contact_slip_velocity() -> None:
    geometry = ShellGeometry()
    distance = 2.0 * geometry.outer_radius_m

    contact = update_shell_contact_friction(
        moving_shell(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ),
        geometry,
        moving_shell(
            [distance, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ),
        geometry,
        normal_load_n=22.6,
        time_step_s=0.001,
    )

    assert contact.regime is ContactRegime.STICK
    np.testing.assert_allclose(
        contact.relative_tangent_velocity_world,
        np.zeros(3),
        atol=1.0e-12,
    )


def test_complete_pre_isaac_step_preserves_global_wrench() -> None:
    geometry = ShellGeometry()
    magnet = MagnetConfig()
    distance = 2.0 * geometry.outer_radius_m
    magnet_center = geometry.inner_radius_m - 0.009 - magnet.half_length_m

    result = compute_two_module_docking_step(
        active_shell_state=stationary_shell([0.0, 0.0, 0.0]),
        active_shell_geometry=geometry,
        passive_shell_state=stationary_shell([distance, 0.0, 0.0]),
        passive_shell_geometry=geometry,
        active_magnet_state=stationary_magnet(
            [magnet_center, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ),
        magnet_config=magnet,
        internal_preload_force_n=9.5,
        time_step_s=1.0 / 120.0,
    )
    residual = result.total_wrench_at(np.zeros(3))

    assert result.external_magnetic.parallel_force_n == pytest.approx(22.6)
    assert result.internal_preload.preload_force_n == pytest.approx(9.5)
    np.testing.assert_allclose(
        result.internal_preload.interaction_point_world,
        [magnet_center + magnet.half_length_m, 0.0, 0.0],
    )
    np.testing.assert_allclose(residual.force, np.zeros(3), atol=1.0e-12)
    np.testing.assert_allclose(residual.torque, np.zeros(3), atol=1.0e-12)

    resolved = compute_two_module_docking_step(
        active_shell_state=stationary_shell([0.0, 0.0, 0.0]),
        active_shell_geometry=geometry,
        passive_shell_state=stationary_shell([distance, 0.0, 0.0]),
        passive_shell_geometry=geometry,
        active_magnet_state=stationary_magnet(
            [magnet_center, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ),
        magnet_config=magnet,
        internal_preload_force_n=9.5,
        time_step_s=1.0 / 120.0,
        resolved_contact_normal_load_n=5.0,
    )
    assert resolved.shell_contact.normal_load_n == pytest.approx(5.0)


def test_internal_preload_follows_shell_radius_and_restores_alignment() -> None:
    geometry = ShellGeometry()
    magnet = MagnetConfig()
    angle = np.deg2rad(20.0)
    state = MagnetState(
        center_world=[0.040, 0.0, 0.0],
        axis_world=[np.cos(angle), np.sin(angle), 0.0],
        carrier_com_world=[0.030, 0.0, 0.0],
        carrier_linear_velocity_world=[0.0, 0.0, 0.0],
        carrier_angular_velocity_world=[0.0, 0.0, 0.0],
    )

    interaction = compute_internal_magnetic_preload_interaction(
        stationary_shell([0.0, 0.0, 0.0]),
        geometry,
        state,
        magnet,
        9.5,
    )
    radial = np.array(interaction.interaction_point_world, copy=True)
    radial /= np.linalg.norm(radial)

    np.testing.assert_allclose(
        interaction.force_on_carrier_world,
        9.5 * radial,
    )
    assert not np.allclose(radial, state.axis_world)
    assert abs(interaction.carrier_wrench.torque[2]) > 0.0


def test_paper_connection_diagnostic_uses_angle_and_weight() -> None:
    required_at_zero = paper_required_connection_friction(
        angle_deg=0.0,
        magnetic_force_n=22.6,
        gravity_force_n=2.943,
        lower_hemisphere=False,
    )

    assert required_at_zero == pytest.approx(2.943 / 22.6)

    geometry = ShellGeometry()
    magnet = MagnetConfig()
    distance = 2.0 * geometry.outer_radius_m
    angle = np.deg2rad(20.0)
    axis = np.array([np.cos(angle), np.sin(angle), 0.0])
    magnet_radius = geometry.inner_radius_m - 0.009 - magnet.half_length_m
    interaction = compute_external_magnetic_interaction(
        stationary_shell([0.0, 0.0, 0.0]),
        geometry,
        stationary_shell([distance, 0.0, 0.0]),
        geometry,
        stationary_magnet(list(magnet_radius * axis), list(axis)),
        magnet,
    )
    diagnostic = evaluate_freebot_friction_diagnostics(
        interaction=interaction,
        gravity_force_n=2.943,
        shell_radius_m=geometry.outer_radius_m,
        mechanism_com_radius_m=0.051,
        shell_shell_static_coefficient=1.1,
    )

    assert diagnostic.connection_is_feasible
    assert diagnostic.required_connection_coefficient < 1.1


def test_figure9_balance_residual_is_zero_at_static_equilibrium() -> None:
    residual = figure9_balance_residual(
        gravity_force_n=10.0,
        perpendicular_force_n=1.0,
        parallel_force_n=4.0,
        shell_friction_n=2.0,
        shell_normal_n=5.0,
        ground_friction_n=1.0,
        ground_normal_n=7.0,
        shell_radius_m=0.06,
        com_radius_m=0.02,
        angle_deg=0.0,
    )

    assert residual.vertical_force_n == pytest.approx(0.0)
    assert residual.horizontal_force_n == pytest.approx(0.0)
    assert residual.moment_nm == pytest.approx(0.0)


def test_external_resultant_is_applied_at_magnet_face() -> None:
    geometry = ShellGeometry()
    magnet = MagnetConfig()
    center_radius = geometry.inner_radius_m - 0.009 - magnet.half_length_m
    state = stationary_magnet([center_radius, 0.0, 0.0], [1.0, 0.0, 0.0])

    interaction = compute_external_magnetic_interaction(
        stationary_shell([0.0, 0.0, 0.0]),
        geometry,
        stationary_shell([2.0 * geometry.outer_radius_m, 0.0, 0.0]),
        geometry,
        state,
        magnet,
    )

    np.testing.assert_allclose(
        interaction.interaction_point_world,
        state.center_world + magnet.half_length_m * state.axis_world,
    )


def test_default_isaac_asset_has_rigid_wheel_mounts() -> None:
    assert _default_usd_path().name == (
        "freebot_cad_full_nearer_wheels_rigid.usd"
    )


def test_default_stage_gap_stays_inside_measured_capture_range() -> None:
    geometry = ShellGeometry()
    config = IsaacStageConfig(usd_path=_default_usd_path())
    center_distance = (
        config.passive_shell_center_world[0]
        - config.active_shell_center_world[0]
    )

    assert center_distance - 2.0 * geometry.outer_radius_m == pytest.approx(
        0.020
    )
    ground_top = (
        config.ground_center_world[2] + 0.5 * config.ground_size_m[2]
    )
    assert (
        config.active_shell_center_world[2] - geometry.outer_radius_m
    ) == pytest.approx(ground_top)


def test_stage_rejects_rigid_overlap_for_precompressed_tire() -> None:
    with pytest.raises(ValueError, match="precompressed tire"):
        IsaacStageConfig(
            usd_path=_default_usd_path(),
            running_gear=RunningGearGeometry(tire_precompression_m=0.0004),
            materials=IsaacMaterialConfig(
                wheel_contact_stiffness_n_per_m=0.0,
            ),
        )


def test_stage_rejects_rigid_overlap_for_precompressed_caster() -> None:
    with pytest.raises(ValueError, match="precompressed caster"):
        IsaacStageConfig(
            usd_path=_default_usd_path(),
            running_gear=RunningGearGeometry(
                tire_precompression_m=0.0,
                caster_precompression_m=0.0003,
            ),
            materials=IsaacMaterialConfig(
                caster_contact_stiffness_n_per_m=0.0,
            ),
        )


def test_radial_wheel_compliance_has_finite_visible_travel() -> None:
    config = WheelRadialComplianceConfig()

    assert config.enabled
    assert config.inward_travel_m == pytest.approx(0.0006)
    assert config.outward_travel_m == pytest.approx(0.0021)
    assert config.rest_position_m == pytest.approx(0.0017)
    assert config.stiffness_n_per_m == pytest.approx(3_500.0)
    assert config.max_force_n == pytest.approx(15.0)
    assert config.mount_mass_kg == pytest.approx(0.004)


def test_radial_wheel_rest_position_must_stay_inside_end_stops() -> None:
    with pytest.raises(ValueError, match="rest position"):
        WheelRadialComplianceConfig(
            inward_travel_m=0.0005,
            outward_travel_m=0.0010,
            rest_position_m=0.0011,
        )


def test_ros_twist_mapping_drives_and_turns_internal_wheels() -> None:
    forward = twist_to_wheel_targets(0.5, 0.0)
    turn = twist_to_wheel_targets(0.0, 0.5)

    assert forward.left_deg_s == pytest.approx(360.0)
    assert forward.right_deg_s == pytest.approx(360.0)
    assert turn.left_deg_s == pytest.approx(-180.0)
    assert turn.right_deg_s == pytest.approx(180.0)


def test_wheel_target_is_limited_to_paper_no_load_speed() -> None:
    targets = twist_to_wheel_targets(10.0, 0.0)

    assert targets.left_deg_s == pytest.approx(360.0)
    assert targets.right_deg_s == pytest.approx(360.0)


def test_dc_motor_torque_follows_linear_speed_curve() -> None:
    config = WheelDriveConfig()

    assert dc_motor_torque_limit_nm(360.0, 0.0, config) == pytest.approx(
        config.stall_torque_nm
    )
    assert dc_motor_torque_limit_nm(360.0, 180.0, config) == pytest.approx(
        0.5 * config.stall_torque_nm
    )
    assert dc_motor_torque_limit_nm(360.0, 360.0, config) == pytest.approx(0.0)


def test_zero_command_motor_brake_retains_static_torque_capacity() -> None:
    config = WheelDriveConfig(zero_command_brake_torque_nm=0.12)

    assert dc_motor_torque_limit_nm(0.0, 0.0, config) == pytest.approx(0.12)
    assert dc_motor_torque_limit_nm(0.0, 180.0, config) == pytest.approx(
        0.5 * config.stall_torque_nm
    )


def test_physical_cli_disables_heading_control_by_default() -> None:
    args = build_argument_parser().parse_args([])

    assert not args.climb_heading
    assert args.motor_brake_torque == pytest.approx(0.12)
    assert args.motor_armature == pytest.approx(0.003)
    assert args.tire_precompression_mm == pytest.approx(0.9)
    assert args.tire_contact_stiffness == pytest.approx(8_000.0)
    assert args.tire_contact_damping == pytest.approx(40.0)
    assert args.wheel_radial_compliance
    assert args.wheel_radial_inward_mm == pytest.approx(0.6)
    assert args.wheel_radial_outward_mm == pytest.approx(2.1)
    assert args.wheel_radial_rest_mm == pytest.approx(1.7)
    assert args.wheel_radial_stiffness == pytest.approx(3_500.0)
    assert args.wheel_radial_damping == pytest.approx(12.0)
    assert args.wheel_radial_max_force == pytest.approx(15.0)
    assert args.caster_clearance_mm == 0.0
    assert args.caster_precompression_mm == pytest.approx(0.1)
    assert args.caster_contact_stiffness == pytest.approx(2_000.0)
    assert args.caster_contact_damping == pytest.approx(15.0)
    assert args.ground_static_friction == pytest.approx(1.25)
    assert args.ground_dynamic_friction == pytest.approx(1.00)
    assert args.external_target == "active-carrier"
    assert args.mass_scale == pytest.approx(1.0)
    assert not args.debug_draw
    assert args.debug_force_scale == pytest.approx(0.003)


def test_debug_arrow_has_scaled_shaft_and_arrowhead() -> None:
    segments = arrow_segments(
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 0.0, 0.0]),
        scale=0.01,
    )

    assert len(segments) == 3
    np.testing.assert_allclose(segments[0][0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(segments[0][1], [1.02, 2.0, 3.0])
    np.testing.assert_allclose(segments[1][0], segments[0][1])
    np.testing.assert_allclose(segments[2][0], segments[0][1])


def test_zero_debug_vector_draws_no_arrow() -> None:
    assert arrow_segments(np.zeros(3), np.zeros(3)) == ()


def test_vertical_plane_heading_uses_differential_wheel_speed() -> None:
    error = signed_heading_error_rad(
        current_axis_world=[0.0, 1.0, 0.0],
        desired_axis_world=[-0.173648, 0.984808, 0.0],
        rotation_axis_world=[0.0, 0.0, 1.0],
    )
    corrected = apply_climb_heading_correction(
        WheelVelocityTargets(180.0, 180.0),
        error,
    )

    assert np.degrees(error) == pytest.approx(10.0, abs=1.0e-4)
    assert corrected.left_deg_s < 180.0
    assert corrected.right_deg_s > 180.0


def test_isaac_scalar_first_quaternion_rotation() -> None:
    half_angle = np.deg2rad(45.0)
    quaternion = np.array(
        [np.cos(half_angle), 0.0, np.sin(half_angle), 0.0]
    )

    rotated = quaternion_rotate_wxyz(quaternion, [0.0, 0.0, -1.0])

    np.testing.assert_allclose(rotated, [-1.0, 0.0, 0.0], atol=1.0e-12)
