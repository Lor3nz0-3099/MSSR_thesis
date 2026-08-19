from __future__ import annotations

import math

import pytest

from smores_ep.config.geometry import SmoresGeometry


def test_cad_axes_are_mapped_to_ros_body_axes() -> None:
    geometry = SmoresGeometry()
    assert geometry.source_vector_to_body((0.0, 1.0, 0.0)) == (
        1.0,
        -0.0,
        0.0,
    )
    assert geometry.source_vector_to_body((-1.0, 0.0, 0.0)) == (
        0.0,
        1.0,
        0.0,
    )
    assert geometry.source_vector_to_body((0.0, 0.0, 1.0)) == (
        0.0,
        -0.0,
        1.0,
    )


def test_measured_wheels_straddle_body_symmetrically() -> None:
    geometry = SmoresGeometry()
    left = geometry.left_wheel_center_body_m
    right = geometry.right_wheel_center_body_m
    assert left[1] > 0.0
    assert right[1] < 0.0
    assert left[0] == pytest.approx(0.0, abs=5.0e-5)
    assert right[0] == pytest.approx(0.0, abs=5.0e-5)
    assert geometry.track_width_m == pytest.approx(0.070410, abs=1.0e-6)


def test_both_driving_wheels_use_the_62_mm_cad_diameter() -> None:
    geometry = SmoresGeometry()
    assert 2.0 * geometry.wheel_radius_m == pytest.approx(
        0.06212,
        abs=2.0e-5,
    )


def test_bottom_docking_plane_matches_base_chassis_cad_surface() -> None:
    assert SmoresGeometry().bottom_face_x_m == pytest.approx(-0.033999)


def test_tangent_top_to_bottom_root_spacing_uses_cad_face_planes() -> None:
    geometry = SmoresGeometry()
    assert geometry.top_face_x_m == pytest.approx(0.043771)
    assert geometry.top_to_bottom_spacing_m == pytest.approx(0.077770)


def test_pan_face_is_forward_and_tilt_limit_is_ninety_degrees() -> None:
    geometry = SmoresGeometry()
    assert geometry.pan_center_body_m[0] > 0.03
    assert geometry.tilt_min_rad == pytest.approx(-math.pi / 2.0)
    assert geometry.tilt_max_rad == pytest.approx(math.pi / 2.0)


def test_each_visual_part_belongs_to_exactly_one_link() -> None:
    geometry = SmoresGeometry()
    assert len(geometry.all_parts) == len(set(geometry.all_parts))


def test_outer_and_inner_pinions_use_opposite_diagonals() -> None:
    geometry = SmoresGeometry()
    outer_left = geometry.source_outer_left_pinion_center_m
    outer_right = geometry.source_outer_right_pinion_center_m
    inner_left = geometry.source_inner_left_pinion_center_m
    inner_right = geometry.source_inner_right_pinion_center_m
    assert outer_left[2] > outer_right[2]
    assert inner_left[2] < inner_right[2]


def test_ground_height_uses_wheels_until_top_face_moves_down() -> None:
    geometry = SmoresGeometry()
    neutral = geometry.ground_contact_height_m(0.0)
    face_up = geometry.ground_contact_height_m(math.pi / 4.0)
    face_down = geometry.ground_contact_height_m(-math.pi / 4.0)
    assert neutral == pytest.approx(face_up)
    assert face_down > neutral + 0.01


def test_visual_pan_envelope_contains_physics_proxy() -> None:
    geometry = SmoresGeometry()
    assert geometry.pan_visual_radius_m >= geometry.pan_face_radius_m
    assert geometry.pan_visual_thickness_m >= geometry.pan_face_thickness_m
