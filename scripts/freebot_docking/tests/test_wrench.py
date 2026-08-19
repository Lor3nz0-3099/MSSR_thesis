import numpy as np

from freebot_docking.physics.wrench import Wrench


def test_force_at_origin_has_no_moment() -> None:
    wrench = Wrench.from_force_at_point(
        force=[10.0, 0.0, 0.0],
        application_point=[0.0, 0.0, 0.0],
        reference_point=[0.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(wrench.force, [10.0, 0.0, 0.0])
    np.testing.assert_allclose(wrench.torque, [0.0, 0.0, 0.0])


def test_offset_force_generates_moment() -> None:
    wrench = Wrench.from_force_at_point(
        force=[10.0, 0.0, 0.0],
        application_point=[0.0, 0.1, 0.0],
        reference_point=[0.0, 0.0, 0.0],
    )

    np.testing.assert_allclose(wrench.torque, [0.0, 0.0, -1.0])


def test_reference_point_change_preserves_physical_wrench() -> None:
    original = Wrench.from_force_at_point(
        force=[10.0, 0.0, 0.0],
        application_point=[0.0, 0.1, 0.0],
        reference_point=[0.0, 0.0, 0.0],
    )

    shifted = original.expressed_at([0.0, 0.1, 0.0])

    np.testing.assert_allclose(shifted.force, [10.0, 0.0, 0.0])
    np.testing.assert_allclose(shifted.torque, [0.0, 0.0, 0.0])


def test_action_reaction_pair_has_zero_total_wrench() -> None:
    point = np.array([0.05, 0.02, 0.01])
    force = np.array([10.0, -2.0, 3.0])
    origin = np.zeros(3)

    action = Wrench.from_force_at_point(force, point, origin)
    reaction = Wrench.from_force_at_point(-force, point, origin)

    total = action + reaction

    np.testing.assert_allclose(total.force, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(total.torque, np.zeros(3), atol=1e-12)

def test_wrench_copies_input_vectors() -> None:
    force = np.array([1.0, 2.0, 3.0])
    point = np.array([0.1, 0.2, 0.3])
    reference = np.zeros(3)

    wrench = Wrench.from_force_at_point(
        force=force,
        application_point=point,
        reference_point=reference,
    )

    force[0] = 100.0
    point[1] = 100.0
    reference[2] = 100.0

    np.testing.assert_array_equal(wrench.force, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(wrench.reference_point, [0.0, 0.0, 0.0])