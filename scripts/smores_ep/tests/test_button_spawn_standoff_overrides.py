import math

from smores_ep.isaac.obstacle_course import (
    mobile_manipulator_button_test_course,
    sample_button_target_spec,
)


def _spawn_standoff(seed: int) -> float:
    spec = sample_button_target_spec(seed)
    course = mobile_manipulator_button_test_course(spec)

    bx, by, _ = course.button_center_xyz_m
    sx, sy = course.assembly_spawn_center_xy_m

    return math.hypot(sx - bx, sy - by)


def test_button_05_and_07_spawn_closer_without_changing_other_seeds():
    assert math.isclose(_spawn_standoff(6341), 0.60, abs_tol=1e-9)
    assert math.isclose(_spawn_standoff(6265), 0.60, abs_tol=1e-9)

    # Control seed: existing validated behavior must remain unchanged.
    assert math.isclose(_spawn_standoff(6217), 1.20, abs_tol=1e-9)
