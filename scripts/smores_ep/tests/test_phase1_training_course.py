from __future__ import annotations

import math

import pytest


def test_phase1_course_is_seed_reproducible_and_target_varies():
    from smores_ep.isaac.phase1_training_course import (
        phase1_rc_car_training_course,
    )

    first = phase1_rc_car_training_course(5101)
    repeated = phase1_rc_car_training_course(5101)
    other = phase1_rc_car_training_course(5102)

    assert first == repeated
    assert first.seed == 5101

    assert (
        first.target_platform.center_xy_m
        != other.target_platform.center_xy_m
    )


def test_phase1_course_reuses_measured_rc_car_footprint():
    from smores_ep.isaac.phase1_training_course import (
        phase1_rc_car_training_course,
    )

    course = phase1_rc_car_training_course(5101)

    assert course.vehicle_length_m == pytest.approx(0.310)
    assert course.vehicle_width_m == pytest.approx(0.232)

    assert course.cone_radius_m == pytest.approx(0.050)

    # User-approved "RC-Car width + a few centimetres tolerance".
    assert course.clearance_margin_m == pytest.approx(0.050)

    assert course.required_gate_opening_m == pytest.approx(
        course.vehicle_width_m
        + 2.0 * course.clearance_margin_m
    )


def test_phase1_course_has_start_road_target_and_no_physical_side_walls():
    from smores_ep.isaac.phase1_training_course import (
        phase1_rc_car_training_course,
    )

    course = phase1_rc_car_training_course(5101)

    semantics = {
        box.semantic
        for box in course.support_boxes
    }

    assert "phase1_start_platform" in semantics
    assert "phase1_road" in semantics
    assert "phase1_target_platform" in semantics

    assert not any(
        "wall" in semantic
        or "barrier" in semantic
        for semantic in semantics
    )

    assert all(
        box.collidable
        for box in course.support_boxes
    )


def test_phase1_cones_are_static_and_seeded():
    from smores_ep.isaac.phase1_training_course import (
        phase1_rc_car_training_course,
    )

    course = phase1_rc_car_training_course(5101)

    assert 4 <= len(course.cones) <= 6

    assert all(
        cone.static
        for cone in course.cones
    )

    assert all(
        cone.radius_m == pytest.approx(0.050)
        for cone in course.cones
    )


def test_phase1_supports_curved_forward_only_courses():
    from smores_ep.isaac.phase1_training_course import (
        phase1_rc_car_training_course,
    )

    curved = None

    for seed in range(5101, 5151):
        candidate = phase1_rc_car_training_course(seed)

        if candidate.has_curve:
            curved = candidate
            break

    assert curved is not None
    assert curved.forward_only is True

    road_boxes = [
        box
        for box in curved.support_boxes
        if box.semantic == "phase1_road"
    ]

    assert road_boxes

    assert any(
        abs(box.yaw_deg) > 1.0
        for box in road_boxes
    )


def test_phase1_target_success_requires_all_eight_modules_inside():
    from smores_ep.isaac.phase1_training_course import (
        phase1_rc_car_training_course,
    )

    course = phase1_rc_car_training_course(5101)

    cx, cy = course.target_platform.center_xy_m

    inside = {
        f"smores_{index:02d}": (
            cx + 0.02 * ((index % 3) - 1),
            cy + 0.02 * ((index // 3) - 1),
        )
        for index in range(1, 9)
    }

    assert course.target_module_count(inside) == 8
    assert course.success(inside) is True

    outside = dict(inside)

    yaw = course.target_platform.yaw_rad

    # Move one module past the local +X target-platform edge.
    local_distance = (
        0.5 * course.target_platform.size_xy_m[0]
        + 0.05
    )

    outside["smores_08"] = (
        cx + math.cos(yaw) * local_distance,
        cy + math.sin(yaw) * local_distance,
    )

    assert course.target_module_count(outside) == 7
    assert course.success(outside) is False


def test_phase1_public_observation_does_not_expose_generator_reference_path():
    from smores_ep.isaac.phase1_training_course import (
        phase1_rc_car_training_course,
    )

    course = phase1_rc_car_training_course(5101)

    observation = course.to_observation()

    dumped = repr(observation)

    assert "centerline_xy_m" not in dumped
    assert "reference_path" not in dumped
    assert "safe_path" not in dumped

    assert observation["course_profile"] == "marl_phase1_rc_car8"
    assert "goal" in observation
    assert "navigation_cones" in observation
