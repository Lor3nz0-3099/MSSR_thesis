from __future__ import annotations

import math

import pytest

from smores_ep.control.differential_drive import (
    PlanarPose,
    integrate_planar_pose,
    twist_to_wheel_rates,
)


def test_straight_twist_has_equal_wheel_rates() -> None:
    rates = twist_to_wheel_rates(0.2, 0.0, 0.04, 0.08)
    assert rates.left_rad_s == pytest.approx(5.0)
    assert rates.right_rad_s == pytest.approx(5.0)


def test_positive_yaw_speeds_up_right_wheel() -> None:
    rates = twist_to_wheel_rates(0.0, 1.0, 0.04, 0.08)
    assert rates.left_rad_s == pytest.approx(-1.0)
    assert rates.right_rad_s == pytest.approx(1.0)


def test_midpoint_pose_integration() -> None:
    pose = integrate_planar_pose(PlanarPose(), 1.0, math.pi, 0.5)
    assert pose.x_m == pytest.approx(math.sqrt(0.5) * 0.5)
    assert pose.y_m == pytest.approx(math.sqrt(0.5) * 0.5)
    assert pose.yaw_rad == pytest.approx(math.pi / 2.0)

