"""Pure-Python validation for run_freebot_emergent_docking.py."""

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_freebot_emergent_docking import (
    CylindricalMagnetForceLaw,
    ForcePairApplier,
    FreeBotAngularForceLaw,
    FreeBotFemForceLaw,
    FreeBotModule,
    GeometryConfig,
    InternalPatchMagneticModel,
    MagneticInteractionModel,
    MagnetConfig,
    MagnetState,
    ShellState,
    quaternion_rotate_wxyz,
    SimulationConfig,
    TabulatedMagnetForceLaw,
    compute_paper_external_interactions,
    compute_exponential_external_interactions,
    paper_required_connection_friction,
    paper_required_ground_friction,
    _validate_initial_conditions,
    twist_to_wheel_velocities,
    spherical_shell_gap,
    magnetic_shell_gap_from_cad_gap,
    source_magnet_face_to_inner_shell_gap_m,
)


def magnet(face, axis=(1.0, 0.0, 0.0)):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    return MagnetState(
        center_world=np.asarray(face, dtype=np.float64) - 0.005 * axis,
        orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        axis_world=axis,
        face_center_world=np.asarray(face, dtype=np.float64),
        half_length=0.005,
        radius=0.010,
    )


class GeometryAndPhysicsTests(unittest.TestCase):
    def setUp(self):
        self.config = MagnetConfig()
        self.model = MagneticInteractionModel(self.config)
        self.shell = ShellState(
            center_world=np.zeros(3),
            com_world=np.zeros(3),
            inner_radius=0.0613335,
            outer_radius=0.0633335,
        )

    def single_interaction(self, state, shell=None, surface_type="outer"):
        interactions = self.model.compute_competing_interactions(
            state,
            {"surface": (shell or self.shell, surface_type)},
        )
        return interactions.get("surface")

    def test_quaternion_rotation(self):
        half = np.sqrt(0.5)
        rotated = quaternion_rotate_wxyz([half, 0.0, 0.0, half], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1.0e-12)

    def test_ray_surface_and_gap(self):
        state = magnet([-0.080, 0.0, 0.0])
        interaction = self.single_interaction(state)
        self.assertIsNotNone(interaction)
        self.assertEqual(interaction.surface_method, "ray")
        self.assertAlmostEqual(interaction.gap, 0.0166665, places=12)
        np.testing.assert_allclose(interaction.shell_application_point_world, [-0.0633335, 0.0, 0.0])

    def test_newtons_third_law(self):
        interaction = self.single_interaction(magnet([-0.080, 0.0, 0.0]))
        np.testing.assert_allclose(
            interaction.force_on_magnet_world + interaction.force_on_shell_world,
            np.zeros(3),
            atol=1.0e-12,
        )

    def test_force_decreases_monotonically_with_gap(self):
        law = CylindricalMagnetForceLaw(self.config)
        gaps = np.linspace(0.0, 0.05, 101)
        forces = np.array([law.force_n(gap, 1.0) for gap in gaps])
        self.assertTrue(np.all(np.diff(forces) < 0.0))
        self.assertTrue(np.all(forces >= 0.0))

    def test_alignment_is_continuous_and_monotonic(self):
        law = CylindricalMagnetForceLaw(self.config)
        alignments = np.linspace(0.0, 1.0, 101)
        forces = np.array([law.force_n(0.005, alignment) for alignment in alignments])
        self.assertTrue(np.all(np.diff(forces) >= 0.0))
        self.assertEqual(forces[0], 0.0)

    def test_missing_ray_intersection_produces_no_interaction(self):
        state = magnet([-0.080, 0.080, 0.0], axis=(1.0, 0.0, 0.0))
        self.assertIsNone(self.model.compute_candidate_geometry(state, self.shell, "outer"))
        self.assertIsNone(self.single_interaction(state))

    def test_alignment_uses_local_surface_normal(self):
        aligned = self.model.compute_candidate_geometry(magnet([-0.080, 0.0, 0.0]), self.shell, "outer")
        off_axis = self.model.compute_candidate_geometry(magnet([-0.080, 0.020, 0.0]), self.shell, "outer")
        self.assertAlmostEqual(aligned.alignment_cosine, 1.0, places=12)
        self.assertGreater(off_axis.alignment_cosine, 0.0)
        self.assertLess(off_axis.alignment_cosine, 1.0)
        self.assertFalse(np.allclose(off_axis.surface_normal_world, [1.0, 0.0, 0.0]))

    def test_inner_surface_normal_points_from_gap_into_steel(self):
        state = magnet([0.040, 0.0, 0.0])
        candidate = self.model.compute_candidate_geometry(state, self.shell, "inner")
        np.testing.assert_allclose(candidate.surface_normal_world, [1.0, 0.0, 0.0], atol=1.0e-12)
        self.assertAlmostEqual(candidate.alignment_cosine, 1.0, places=12)

    def test_internal_patch_is_finite_and_action_reaction_balanced(self):
        state = magnet([0.040, 0.0, 0.0])
        result = InternalPatchMagneticModel(self.config).compute(state, self.shell, state.center_world)
        self.assertEqual(len(result.interactions), 1 + 5 * 16)
        self.assertTrue(np.isfinite(result.peak_pressure_pa))
        self.assertTrue(np.isfinite(result.peak_field_t))
        np.testing.assert_allclose(
            result.total_force_on_internal_world + result.total_force_on_shell_world,
            np.zeros(3), atol=1.0e-10,
        )
        for interaction in result.interactions:
            np.testing.assert_allclose(
                interaction.force_on_magnet_world + interaction.force_on_shell_world,
                np.zeros(3), atol=1.0e-12,
            )
            self.assertLess(
                np.linalg.norm(np.cross(
                    interaction.force_on_shell_world,
                    interaction.surface_normal_world,
                )),
                1.0e-12,
            )
            self.assertGreaterEqual(
                float(np.dot(
                    interaction.force_on_shell_world,
                    interaction.surface_normal_world,
                )),
                0.0,
            )
        self.assertTrue(np.isclose(result.sampled_area_m2, result.cap_area_m2, rtol=1e-10, atol=1e-14))
        self.assertGreater(result.radial_force_on_internal_n, 0.0)
        self.assertLess(result.tangential_force_on_internal_n, 1.0e-10)

    def test_distributed_dipoles_preserve_total_moment(self):
        model = InternalPatchMagneticModel(self.config)
        per_dipole = model.magnetic_moment_magnitude / self.config.axial_dipole_count
        self.assertAlmostEqual(
            per_dipole * self.config.axial_dipole_count,
            model.magnetic_moment_magnitude,
            places=14,
        )
        self.assertEqual(len(model.axial_dipole_offsets_m()), 7)
        self.assertAlmostEqual(model.axial_dipole_offsets_m()[0], -0.004, places=14)
        self.assertAlmostEqual(model.axial_dipole_offsets_m()[-1], 0.004, places=14)

    def test_wheel_slip_diagnostic_estimates_rolling_point_velocity(self):
        # Isolate the generic rolling-point kinematics from the CAD-specific
        # axial centre offset and 0.167-degree tire-axis tilt.
        geometry = GeometryConfig(tire_center_axial_offset_m=0.0, tire_axis_tilt_deg=0.0)

        class Body:
            def __init__(self, position, linear, angular):
                self.position = np.asarray(position, dtype=np.float64)
                self.linear = np.asarray(linear, dtype=np.float64)
                self.angular = np.asarray(angular, dtype=np.float64)

            def get_world_poses(self):
                return self.position.reshape(1, 3), np.array([[1.0, 0.0, 0.0, 0.0]])

            def get_velocities(self):
                return self.linear.reshape(1, 3), self.angular.reshape(1, 3)

        wheel = Body(
            [geometry.shell_inner_radius_m - geometry.tire_outer_radius_m, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
        )
        shell_body = Body([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        module = FreeBotModule("/unused", shell_body, None, None, wheel, wheel, None, None)
        shell = ShellState(np.zeros(3), np.zeros(3), geometry.shell_inner_radius_m, geometry.shell_outer_radius_m)
        slip = module.wheel_slip_diagnostic(wheel, shell, geometry)
        self.assertAlmostEqual(slip.clearance_m, 0.0, places=14)
        self.assertTrue(slip.estimated_contact)
        self.assertAlmostEqual(slip.wheel_surface_speed_m_s, 0.16000287, places=6)
        self.assertAlmostEqual(slip.slip_ratio, 1.0, places=12)

    def test_internal_patch_pressure_scale_is_explicit(self):
        state = magnet([0.040, 0.0, 0.0])
        unit = InternalPatchMagneticModel(self.config).compute(state, self.shell, state.center_world)
        doubled_config = MagnetConfig(internal_pressure_scale=2.0)
        doubled = InternalPatchMagneticModel(doubled_config).compute(state, self.shell, state.center_world)
        self.assertAlmostEqual(
            np.linalg.norm(doubled.total_force_on_internal_world),
            2.0 * np.linalg.norm(unit.total_force_on_internal_world), places=10,
        )

    def test_internal_near_field_guard_prevents_explosive_load(self):
        # Active face on the inner wall: a point dipole is outside its valid
        # near-field domain here, so the configurable pressure guard must act.
        center = np.array([self.shell.inner_radius - 0.005, 0.0, 0.0])
        state = MagnetState(
            center_world=center,
            orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            axis_world=np.array([1.0, 0.0, 0.0]),
            face_center_world=center + np.array([0.005, 0.0, 0.0]),
            half_length=0.005,
            radius=0.010,
        )
        result = InternalPatchMagneticModel(self.config).compute(state, self.shell, center)
        self.assertLess(np.linalg.norm(result.total_force_on_internal_world), 20.0)
        self.assertLessEqual(result.peak_pressure_pa, self.config.maximum_sample_pressure_pa)

    def test_competing_surface_weights_sum_to_one(self):
        state = magnet([0.040, 0.0, 0.0])
        analytic_model = MagneticInteractionModel(
            self.config,
            external_force_law=CylindricalMagnetForceLaw(self.config),
        )
        external_shell = ShellState(
            center_world=np.array([0.150, 0.0, 0.0]),
            com_world=np.array([0.150, 0.0, 0.0]),
            inner_radius=0.0613335,
            outer_radius=0.0633335,
        )
        interactions = analytic_model.compute_competing_interactions(
            state,
            {"own": (self.shell, "inner"), "external": (external_shell, "outer")},
        )
        self.assertEqual(set(interactions), {"own", "external"})
        self.assertAlmostEqual(sum(item.normalized_weight for item in interactions.values()), 1.0, places=12)
        for item in interactions.values():
            self.assertAlmostEqual(item.force_magnitude, item.raw_coupling * item.normalized_weight, places=12)

    def test_calibrated_fem_branch_is_not_reduced_twice(self):
        state = magnet([0.040, 0.0, 0.0])
        external_shell = ShellState(
            center_world=np.array([0.126667, 0.0, 0.0]),
            com_world=np.array([0.126667, 0.0, 0.0]),
            inner_radius=0.0613335,
            outer_radius=0.0633335,
        )
        interactions = self.model.compute_competing_interactions(
            state,
            {"own": (self.shell, "inner"), "external": (external_shell, "outer", 0.0)},
        )
        self.assertEqual(interactions["own"].normalized_weight, 1.0)
        self.assertEqual(interactions["external"].normalized_weight, 1.0)
        self.assertAlmostEqual(interactions["external"].force_magnitude, 22.6, places=10)

    def test_global_magnetic_force_balance_for_two_magnets(self):
        left_shell = self.shell
        right_shell = ShellState(
            center_world=np.array([0.150, 0.0, 0.0]),
            com_world=np.array([0.150, 0.0, 0.0]),
            inner_radius=0.0613335,
            outer_radius=0.0633335,
        )
        left = self.model.compute_competing_interactions(
            magnet([0.040, 0.0, 0.0], axis=(1.0, 0.0, 0.0)),
            {"own": (left_shell, "inner"), "other": (right_shell, "outer")},
        )
        right = self.model.compute_competing_interactions(
            magnet([0.110, 0.0, 0.0], axis=(-1.0, 0.0, 0.0)),
            {"own": (right_shell, "inner"), "other": (left_shell, "outer")},
        )
        terms = []
        for interaction in (*left.values(), *right.values()):
            terms.extend((interaction.force_on_magnet_world, interaction.force_on_shell_world))
        np.testing.assert_allclose(np.sum(np.stack(terms), axis=0), np.zeros(3), atol=1.0e-12)

    def test_application_point_torque(self):
        interaction = self.single_interaction(magnet([-0.080, 0.010, 0.0]))
        expected = np.cross(
            interaction.shell_application_point_world - self.shell.com_world,
            interaction.force_on_shell_world,
        )
        _, actual = ForcePairApplier.torques(interaction, np.array([-0.085, 0.010, 0.0]), self.shell.com_world)
        np.testing.assert_allclose(actual, expected, atol=1.0e-12)

    def test_physx_local_com_is_transformed_to_world(self):
        half = np.sqrt(0.5)

        class Body:
            @staticmethod
            def get_world_poses():
                return np.array([[1.0, 2.0, 3.0]]), np.array([[half, 0.0, 0.0, half]])

            @staticmethod
            def get_coms():
                return np.array([[1.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0, 0.0]])

        np.testing.assert_allclose(FreeBotModule.body_com_world(Body()), [1.0, 3.0, 3.0], atol=1.0e-12)

    def test_default_initial_gap_and_axes_are_valid(self):
        config = SimulationConfig(usd_path=Path("unused.usd"))
        self.assertAlmostEqual(_validate_initial_conditions(config), 0.040, places=12)
        platform_left = config.platform_center[0] - 0.5 * config.platform_size[0]
        self.assertAlmostEqual(
            config.target_start[0] + config.geometry.shell_outer_radius_m,
            platform_left,
            places=12,
        )

    def test_source_cad_magnet_face_gap_matches_fitted_sphere(self):
        config = SimulationConfig(usd_path=Path("unused.usd"))
        gap = source_magnet_face_to_inner_shell_gap_m(config.geometry)
        self.assertGreater(gap, 0.004)
        self.assertLess(gap, 0.006)

    def paper_external(self, active, passive, shell_gap=0.0):
        active_shell = ShellState(
            center_world=np.zeros(3),
            com_world=np.zeros(3),
            inner_radius=0.0613335,
            outer_radius=0.0633335,
        )
        passive_shell = ShellState(
            center_world=np.array([0.126667, 0.0, 0.0]),
            com_world=np.array([0.126667, 0.0, 0.0]),
            inner_radius=0.0613335,
            outer_radius=0.0633335,
        )
        return compute_paper_external_interactions(
            active,
            passive,
            active_shell,
            passive_shell,
            shell_gap,
            FreeBotFemForceLaw(),
            FreeBotAngularForceLaw(),
        )

    def test_paper_aligned_shell_branch_is_balanced_22_6_n_at_contact(self):
        active = magnet([0.050, 0.0, 0.0], axis=(1.0, 0.0, 0.0))
        passive_bottom = magnet([0.126667, 0.0, -0.050], axis=(0.0, 0.0, -1.0))
        interactions = self.paper_external(active, passive_bottom)
        interaction = interactions["active_field_to_passive_shell"]
        self.assertAlmostEqual(interaction.force_magnitude, 22.6, places=12)
        np.testing.assert_allclose(
            interaction.force_on_magnet_world + interaction.force_on_shell_world,
            np.zeros(3),
            atol=1.0e-12,
        )
        self.assertEqual(interaction.surface_method, "paper-fig4-fig5-shell-patch")
        np.testing.assert_allclose(interaction.force_on_magnet_world, [22.6, 0.0, 0.0], atol=1.0e-12)

    def test_exponential_external_force_is_22_6_at_aligned_contact(self):
        active_shell = self.shell
        passive_shell = ShellState(
            center_world=np.array([0.126667, 0.0, 0.0]),
            com_world=np.array([0.126667, 0.0, 0.0]),
            inner_radius=0.0613335,
            outer_radius=0.0633335,
        )
        interactions = compute_exponential_external_interactions(
            magnet([0.050, 0.0, 0.0], axis=(1.0, 0.0, 0.0)),
            active_shell,
            passive_shell,
            0.0,
            FreeBotAngularForceLaw(),
        )
        interaction = interactions["active_field_to_passive_shell"]
        self.assertAlmostEqual(interaction.force_magnitude, 22.6, places=12)
        np.testing.assert_allclose(
            interaction.force_on_magnet_world + interaction.force_on_shell_world,
            np.zeros(3), atol=1.0e-12,
        )

    def test_exponential_external_force_remains_positive_off_axis(self):
        active_shell = self.shell
        passive_shell = ShellState(
            center_world=np.array([0.126667, 0.0, 0.0]),
            com_world=np.array([0.126667, 0.0, 0.0]),
            inner_radius=0.0613335,
            outer_radius=0.0633335,
        )
        angle = np.deg2rad(45.0)
        interactions = compute_exponential_external_interactions(
            magnet([0.050, 0.0, 0.0], axis=(np.cos(angle), 0.0, np.sin(angle))),
            active_shell,
            passive_shell,
            0.0,
            FreeBotAngularForceLaw(),
        )
        self.assertAlmostEqual(
            interactions["active_field_to_passive_shell"].force_magnitude,
            22.6 / np.e,
            places=12,
        )

    def test_finite_shell_patch_remains_nonzero_without_exact_axis_hit(self):
        passive_bottom = magnet([0.126667, 0.0, -0.050], axis=(0.0, 0.0, -1.0))
        straight = self.paper_external(magnet([0.050, 0.0, 0.0], axis=(1.0, 0.0, 0.0)), passive_bottom)[
            "active_field_to_passive_shell"
        ]
        tilted = self.paper_external(magnet([0.0, 0.0, -0.050], axis=(0.0, 0.0, -1.0)), passive_bottom)[
            "active_field_to_passive_shell"
        ]
        self.assertGreater(tilted.force_magnitude, 1.7)
        self.assertLess(tilted.force_magnitude, straight.force_magnitude)
        self.assertGreater(tilted.force_on_magnet_world[2], 0.0)
        self.assertNotAlmostEqual(
            straight.shell_application_point_world[2],
            tilted.shell_application_point_world[2],
            places=6,
        )

    def test_finite_pole_removes_hard_cut_at_ninety_degrees(self):
        passive_bottom = magnet([0.126667, 0.0, -0.050], axis=(0.0, 0.0, -1.0))
        angle_91 = np.deg2rad(91.0)
        near_rear_axis = (np.cos(angle_91), 0.0, -np.sin(angle_91))
        near_rear = self.paper_external(magnet([0.0, 0.0, -0.050], axis=near_rear_axis), passive_bottom)
        self.assertIn("active_field_to_passive_shell", near_rear)
        angle_110 = np.deg2rad(110.0)
        far_rear_axis = (np.cos(angle_110), 0.0, -np.sin(angle_110))
        self.assertEqual(self.paper_external(magnet([0.0, 0.0, -0.050], axis=far_rear_axis), passive_bottom), {})

    def test_close_same_poles_keep_shell_attraction_with_fig6_disabled(self):
        active = magnet([0.055, 0.0, 0.0], axis=(1.0, 0.0, 0.0))
        passive_contact_pole = magnet([0.071667, 0.0, 0.0], axis=(-1.0, 0.0, 0.0))
        interactions = self.paper_external(active, passive_contact_pole)
        self.assertNotIn("active_to_passive_magnet", interactions)
        interaction = interactions["active_field_to_passive_shell"]
        self.assertEqual(interaction.surface_method, "paper-fig4-fig5-shell-patch")
        self.assertGreater(interaction.parallel_force_n, 0.0)
        self.assertGreater(interaction.force_on_magnet_world[0], 0.0)
        np.testing.assert_allclose(
            interaction.force_on_magnet_world + interaction.force_on_shell_world,
            np.zeros(3),
            atol=1.0e-12,
        )

    def test_paper_components_scale_with_published_shell_gap_curve(self):
        active = magnet([0.050, 0.0, 0.0], axis=(1.0, 0.0, 0.0))
        passive_bottom = magnet([0.126667, 0.0, -0.050], axis=(0.0, 0.0, -1.0))
        contact = self.paper_external(active, passive_bottom, 0.0)["active_field_to_passive_shell"]
        separated = self.paper_external(active, passive_bottom, 0.010)["active_field_to_passive_shell"]
        self.assertAlmostEqual(separated.force_magnitude / contact.force_magnitude, 1.35 / 22.6, places=12)

    def test_paper_static_friction_diagnostics_are_finite_when_force_supports_weight(self):
        gravity = 0.360 * 9.81
        upper = paper_required_connection_friction(30.0, 22.6, gravity, False)
        lower = paper_required_connection_friction(30.0, 22.6, gravity, True)
        self.assertGreaterEqual(lower, upper)
        self.assertTrue(np.isfinite(lower))
        ground = paper_required_ground_friction(30.0, 6.7, 0.919, gravity, 0.0633335, 0.045, 0.95)
        self.assertTrue(np.isfinite(ground) or np.isinf(ground))

    def test_tabulated_law_is_monotone_and_zero_beyond_data(self):
        law = TabulatedMagnetForceLaw([0.0, 0.01, 0.02], [20.0, 8.0, 2.0])
        self.assertAlmostEqual(law.force_n(0.005, 1.0), 14.0)
        self.assertEqual(law.force_n(0.030, 1.0), 0.0)
        self.assertLess(law.force_n(0.005, 0.5), law.force_n(0.005, 1.0))

    def test_freebot_fem_curve_contains_full_active_to_passive_force(self):
        law = FreeBotFemForceLaw()
        self.assertAlmostEqual(law.force_n(0.0, 1.0), 22.6, places=12)
        self.assertEqual(law.force_n(0.030, 1.0), 0.0)

    def test_spherical_shell_gap_uses_cad_radii(self):
        other = ShellState(
            center_world=np.array([0.146667, 0.0, 0.0]),
            com_world=np.array([0.146667, 0.0, 0.0]),
            inner_radius=0.0613335,
            outer_radius=0.0633335,
        )
        self.assertAlmostEqual(spherical_shell_gap(self.shell, other), 0.020, places=12)

    def test_cad_contact_offset_maps_observed_contact_gap_to_zero(self):
        self.assertEqual(magnetic_shell_gap_from_cad_gap(0.00239, 0.0025), 0.0)
        self.assertAlmostEqual(magnetic_shell_gap_from_cad_gap(0.010, 0.0025), 0.0075)

    def test_cmd_vel_wheel_mapping_matches_previous_runner(self):
        config = SimulationConfig(usd_path=Path("unused.usd"))
        left, right, forward, turn = twist_to_wheel_velocities(0.5, 0.25, config)
        self.assertEqual(forward, 450.0)
        self.assertEqual(turn, 90.0)
        self.assertEqual(left, 360.0)
        self.assertEqual(right, 540.0)

    def test_zero_cmd_vel_stops_both_wheels(self):
        config = SimulationConfig(usd_path=Path("unused.usd"))
        self.assertEqual(twist_to_wheel_velocities(0.0, 0.0, config), (0.0, 0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
