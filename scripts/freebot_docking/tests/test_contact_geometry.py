import numpy as np
import pytest

from freebot_docking.physics.contact_geometry import (
    ball_inside_sphere_clearance_m,
    cylinder_inside_sphere_clearance_m,
    radial_ball_center_shift_m,
    radial_cylinder_center_shift_m,
)


def test_ball_inside_sphere_clearance() -> None:
    assert ball_inside_sphere_clearance_m(
        [0.0, 0.0, 0.04],
        0.005,
        0.05,
    ) == pytest.approx(0.005)


def test_cylinder_inside_sphere_clearance_for_radial_axis() -> None:
    clearance = cylinder_inside_sphere_clearance_m(
        [0.0, 0.0, 0.03],
        [0.0, 1.0, 0.0],
        cylinder_radius_m=0.01,
        cylinder_half_width_m=0.003,
        sphere_inner_radius_m=0.05,
    )

    assert clearance == pytest.approx(
        0.05 - np.hypot(0.003, 0.04)
    )


def test_cylinder_clearance_rejects_zero_axis() -> None:
    with pytest.raises(ValueError, match="axis"):
        cylinder_inside_sphere_clearance_m(
            [0.0, 0.0, 0.03],
            [0.0, 0.0, 0.0],
            0.01,
            0.003,
            0.05,
        )


def test_radial_ball_shift_preserves_two_mm_caster_clearance() -> None:
    center = np.array([0.04, 0.01, -0.02])
    shift = radial_ball_center_shift_m(
        center,
        ball_radius_m=0.00465,
        sphere_inner_radius_m=0.0613472,
        target_clearance_m=0.002,
    )

    assert ball_inside_sphere_clearance_m(
        center + shift,
        0.00465,
        0.0613472,
    ) == pytest.approx(0.002, abs=1.0e-12)


def test_radial_cylinder_shift_closes_wheel_contact() -> None:
    center = np.array([0.0, 0.035, -0.024])
    axis = np.array([0.0, 1.0, 0.0])
    shift = radial_cylinder_center_shift_m(
        center,
        axis,
        cylinder_radius_m=0.016,
        cylinder_half_width_m=0.003,
        sphere_inner_radius_m=0.0613472,
    )

    assert cylinder_inside_sphere_clearance_m(
        center + shift,
        axis,
        0.016,
        0.003,
        0.0613472,
    ) == pytest.approx(0.0, abs=1.0e-12)
