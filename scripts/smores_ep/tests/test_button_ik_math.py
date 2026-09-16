"""Pure regression tests for button IK geometry and PAN command semantics."""
from __future__ import annotations

import importlib.util
import math

import numpy as np


def test_button_ik_math_module_exists() -> None:
    assert importlib.util.find_spec("smores_ep.control.button_ik_math") is not None


def test_relative_pan_delta_uses_physical_current_angle() -> None:
    from smores_ep.control.button_ik_math import shortest_angular_delta

    actual = math.radians(-86.46)
    target = math.radians(-82.00)
    assert math.degrees(shortest_angular_delta(actual, target)) == pytest.approx(4.46, abs=1e-9)


def test_relative_pan_delta_wraps_across_pi() -> None:
    from smores_ep.control.button_ik_math import shortest_angular_delta

    actual = math.radians(179.0)
    target = math.radians(-179.0)
    assert math.degrees(shortest_angular_delta(actual, target)) == pytest.approx(2.0, abs=1e-9)


def test_orientation_residual_distinguishes_antiparallel_branch() -> None:
    from smores_ep.control.button_ik_math import orientation_residual

    target = np.asarray([-1.0, 0.0, 0.0])
    aligned = orientation_residual(target, target, lever_m=0.08)
    opposite = orientation_residual(-target, target, lever_m=0.08)

    assert np.linalg.norm(aligned) == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(opposite) == pytest.approx(0.16, abs=1e-12)


def test_normal_waypoint_crosses_old_hemisphere_boundary_progressively() -> None:
    from smores_ep.control.button_ik_math import bounded_normal_waypoint, vector_angle_deg

    current = np.asarray([0.3033, -0.0076, 0.9529])
    target = np.asarray([-1.0, 0.0, 0.0])

    initial_error = vector_angle_deg(current, target)
    waypoint = bounded_normal_waypoint(current, target, max_angle_rad=math.radians(15.0))
    next_error = vector_angle_deg(waypoint, target)

    assert initial_error > 90.0
    assert next_error < initial_error
    assert initial_error - next_error == pytest.approx(15.0, abs=1e-6)
    assert np.linalg.norm(waypoint) == pytest.approx(1.0, abs=1e-12)


def test_position_waypoint_is_bounded() -> None:
    from smores_ep.control.button_ik_math import bounded_position_waypoint

    current = np.asarray([1.72672, -1.04570, 0.20778])
    target = np.asarray([1.64252, -1.04570, 0.16355])
    waypoint = bounded_position_waypoint(current, target, max_step_m=0.015)

    assert np.linalg.norm(waypoint - current) == pytest.approx(0.015, abs=1e-12)
    assert np.linalg.norm(target - waypoint) < np.linalg.norm(target - current)


import pytest
