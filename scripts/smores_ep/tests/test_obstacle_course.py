
from __future__ import annotations

import pytest

from smores_ep.isaac.obstacle_course import (
    CoplanarGapSpec,
    GAP_CURRICULUM_RANGES,
    STAIR_CURRICULUM_RANGES,
    UniformStairSpec,
    manual_obstacle_course,
    mobile_manipulator_button_test_course,
    sample_coplanar_gap_spec,
    sample_uniform_stair_spec,
    snake8_gap_test_course,
    snake8_stair_test_course,
)


def test_manual_obstacle_course_exports_world_landmarks() -> None:
    course = manual_obstacle_course()

    assert course.to_observation() == {
        "frame_id": "world",
        "gap": {"near_edge_x_m": 0.65, "far_edge_x_m": 0.85},
        "ramp": {
            "entry_x_m": -1.55,
            "exit_x_m": -1.10,
            "top_height_m": 0.0,
        },
        "stairs": {
            "top_heights_m": [0.065, 0.13, 0.195],
            "first_riser_x_m": 1.25,
            "riser_depth_m": 0.28,
        },
        "button": {"center_xyz_m": [2.65, 0.455, 0.365]},
        "exit": {"center_xyz_m": [3.55, 0.0, 0.385]},
    }


def test_manual_course_has_a_real_gap_and_monotonic_stairs() -> None:
    course = manual_obstacle_course()
    gap_start, gap_end = course.gap_interval_x_m

    assert gap_end > gap_start
    for element in course.boxes:
        if not element.collidable:
            continue
        half_x = 0.5 * element.size_xyz_m[0]
        element_start = element.center_xyz_m[0] - half_x
        element_end = element.center_xyz_m[0] + half_x
        assert element_end <= gap_start or element_start >= gap_end
    assert course.stair_top_heights_m == tuple(
        sorted(course.stair_top_heights_m)
    )
    assert course.stair_top_heights_m == pytest.approx((0.065, 0.13, 0.195))


def test_manual_course_exposes_future_task_landmarks() -> None:
    course = manual_obstacle_course()
    semantics = {element.semantic for element in course.boxes}

    assert {
        "button",
        "exit_marker",
        "stair",
        "gap_landing",
        "approach_ramp",
        "rear_start_platform",
    } <= semantics
    assert course.button_center_xyz_m[2] > course.stair_top_heights_m[-1]
    assert course.exit_center_xyz_m[0] > course.button_center_xyz_m[0]


def test_button_test_course_is_flat_isolated_and_nav2_addressable() -> None:
    course = mobile_manipulator_button_test_course()

    observation = course.to_observation()
    assert observation["frame_id"] == "world"
    assert observation["course_profile"] == "mobile_manipulator8_button_test"
    assert observation["button"]["center_xyz_m"] == pytest.approx(
        [0.85, 0.475, 0.17]
    )
    assert observation["button"]["base_standoff_xy_m"] == pytest.approx(
        [0.85, 0.275]
    )
    assert observation["button"]["base_standoff_yaw_rad"] == pytest.approx(
        0.5 * 3.141592653589793
    )
    semantics = {box.semantic for box in course.boxes}
    assert semantics == {
        "button_test_platform",
        "button_support",
        "button",
    }
    platform = next(
        box for box in course.boxes if box.semantic == "button_test_platform"
    )
    assert platform.center_xyz_m[2] == pytest.approx(-0.01)


def test_gap_test_course_has_no_collider_across_open_interval() -> None:
    course = snake8_gap_test_course()
    near_x_m, far_x_m = course.gap_interval_x_m

    assert far_x_m - near_x_m == pytest.approx(0.20)
    assert course.to_observation()["gap"] == {
        "near_edge_x_m": pytest.approx(0.55),
        "far_edge_x_m": pytest.approx(0.75),
        "width_m": pytest.approx(0.20),
    }
    assert course.to_observation()["scenario"] == {
        "generator": "coplanar_gap_v1",
        "seed": None,
        "width_m": pytest.approx(0.20),
        "near_edge_x_m": pytest.approx(0.55),
        "far_edge_x_m": pytest.approx(0.75),
        "bank_width_m": pytest.approx(1.20),
        "approach_start_x_m": pytest.approx(-1.00),
        "landing_length_m": pytest.approx(1.25),
        "bank_thickness_m": pytest.approx(0.02),
    }
    for box in course.boxes:
        half_x_m = 0.5 * box.size_xyz_m[0]
        start_x_m = box.center_xyz_m[0] - half_x_m
        end_x_m = box.center_xyz_m[0] + half_x_m
        assert end_x_m <= near_x_m or start_x_m >= far_x_m


def test_seeded_gap_sampling_is_reproducible_and_conservative() -> None:
    first = sample_coplanar_gap_spec(17)
    second = sample_coplanar_gap_spec(17)

    assert first == second
    assert first.seed == 17
    assert 0.160 <= first.width_m <= 0.210
    assert 0.520 <= first.near_edge_x_m <= 0.620


def test_parameterized_gap_geometry_and_metadata_share_one_spec() -> None:
    spec = CoplanarGapSpec(
        width_m=0.175,
        near_edge_x_m=0.610,
        seed=23,
        bank_width_m=1.10,
        landing_length_m=1.40,
    )
    course = snake8_gap_test_course(spec)
    boxes = {box.name: box for box in course.boxes}

    assert course.gap_interval_x_m == pytest.approx((0.610, 0.785))
    assert boxes["NearBank"].size_xyz_m == pytest.approx((1.610, 1.10, 0.02))
    assert boxes["FarBank"].size_xyz_m == pytest.approx((1.40, 1.10, 0.02))
    assert boxes["FarBank"].center_xyz_m[0] == pytest.approx(1.485)
    observation = course.to_observation()
    assert observation["scenario"] == {
        "generator": "coplanar_gap_v1",
        **spec.to_dict(),
    }
    assert observation["gap"] == {
        "near_edge_x_m": pytest.approx(0.610),
        "far_edge_x_m": pytest.approx(0.785),
        "width_m": pytest.approx(0.175),
    }


@pytest.mark.parametrize(
    "kwargs",
    (
        {"width_m": 0.050},
        {"near_edge_x_m": -0.10},
        {"landing_length_m": 0.50},
        {"bank_width_m": 0.50},
    ),
)
def test_gap_spec_rejects_unsupported_geometry(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        CoplanarGapSpec(**kwargs)


def test_snake8_stair_test_course_has_three_wheel_high_risers() -> None:
    course = snake8_stair_test_course()

    assert course.first_riser_x_m == pytest.approx(0.65)
    assert course.riser_depth_m == pytest.approx(0.28)
    assert course.stair_top_heights_m == pytest.approx(
        (0.065, 0.13, 0.195)
    )
    assert tuple(
        second - first
        for first, second in zip(
            (0.0, *course.stair_top_heights_m[:-1]),
            course.stair_top_heights_m,
        )
    ) == pytest.approx((0.065, 0.065, 0.065))
    boxes = {box.name: box for box in course.boxes}
    assert boxes["StartPlatform"].center_xyz_m == pytest.approx(
        (-0.175, 0.0, -0.01)
    )
    assert boxes["Stair01"].center_xyz_m == pytest.approx(
        (0.79, 0.0, 0.0325)
    )
    assert boxes["Stair03"].size_xyz_m == pytest.approx(
        (0.28, 1.20, 0.195)
    )
    assert boxes["UpperDeck"].center_xyz_m == pytest.approx(
        (2.15, 0.0, 0.0975)
    )


def test_snake8_stair_test_observation_is_isolated_from_full_course() -> None:
    course = snake8_stair_test_course()

    observation = course.to_observation()
    collision_boxes = observation.pop("collision_boxes")
    assert observation == {
        "frame_id": "world",
        "course_profile": "snake8_stair_test",
        "scenario": {
            "generator": "uniform_stair_v1",
            "seed": None,
            "rise_m": 0.065,
            "tread_depth_m": 0.28,
            "step_count": 3,
            "first_riser_x_m": 0.65,
            "width_m": 1.2,
            "upper_deck_length_m": 1.32,
        },
        "stairs": {
            "top_heights_m": [0.065, 0.13, 0.195],
            "first_riser_x_m": 0.65,
            "riser_depth_m": 0.28,
        },
    }
    assert len(collision_boxes) == 5
    first_riser = next(
        box
        for box in collision_boxes
        if box["semantic"] == "stair_test_riser"
    )
    assert first_riser["center_xyz_m"] == pytest.approx(
        [0.79, 0.0, 0.0325]
    )
    assert first_riser["size_xyz_m"] == pytest.approx(
        [0.28, 1.2, 0.065]
    )
    assert (
        first_riser["center_xyz_m"][0]
        - 0.5 * first_riser["size_xyz_m"][0]
    ) == pytest.approx(0.65)
    assert {box.semantic for box in course.boxes} == {
        "stair_test_start",
        "stair_test_riser",
        "stair_test_upper_deck",
    }


def test_seeded_uniform_stair_sampling_is_reproducible_and_conservative() -> None:
    first = sample_uniform_stair_spec(17)
    second = sample_uniform_stair_spec(17)

    assert first == second
    assert first.seed == 17
    assert 0.050 <= first.rise_m <= 0.065
    assert 0.250 <= first.tread_depth_m <= 0.320
    assert 2 <= first.step_count <= 4


@pytest.mark.parametrize(
    "level", ("robust", "intermediate", "challenging")
)
def test_stair_curriculum_sampling_stays_inside_declared_level(
    level: str,
) -> None:
    spec = sample_uniform_stair_spec(91, level)
    ranges = STAIR_CURRICULUM_RANGES[level]

    assert ranges["rise_m"][0] <= spec.rise_m <= ranges["rise_m"][1]
    assert (
        ranges["tread_depth_m"][0]
        <= spec.tread_depth_m
        <= ranges["tread_depth_m"][1]
    )
    assert (
        ranges["step_count"][0]
        <= spec.step_count
        <= ranges["step_count"][1]
    )


@pytest.mark.parametrize(
    "level", ("robust", "intermediate", "challenging")
)
def test_gap_curriculum_sampling_stays_inside_declared_level(
    level: str,
) -> None:
    spec = sample_coplanar_gap_spec(92, level)
    ranges = GAP_CURRICULUM_RANGES[level]

    assert ranges["width_m"][0] <= spec.width_m <= ranges["width_m"][1]
    assert (
        ranges["near_edge_x_m"][0]
        <= spec.near_edge_x_m
        <= ranges["near_edge_x_m"][1]
    )


def test_curriculum_sampling_rejects_unknown_levels() -> None:
    with pytest.raises(ValueError, match="curriculum level"):
        sample_uniform_stair_spec(1, "impossible")
    with pytest.raises(ValueError, match="curriculum level"):
        sample_coplanar_gap_spec(1, "impossible")


def test_parameterized_stair_geometry_and_metadata_share_one_spec() -> None:
    spec = UniformStairSpec(
        rise_m=0.055,
        tread_depth_m=0.310,
        step_count=4,
        first_riser_x_m=0.700,
        seed=23,
    )
    course = snake8_stair_test_course(spec)
    risers = tuple(
        box for box in course.boxes if box.semantic == "stair_test_riser"
    )

    assert course.stair_top_heights_m == pytest.approx(
        (0.055, 0.110, 0.165, 0.220)
    )
    assert len(risers) == 4
    for index, riser in enumerate(risers):
        assert riser.center_xyz_m[0] == pytest.approx(
            spec.first_riser_x_m + (index + 0.5) * spec.tread_depth_m
        )
        assert riser.center_xyz_m[2] == pytest.approx(
            0.5 * spec.top_heights_m[index]
        )
        assert riser.size_xyz_m[0] == pytest.approx(spec.tread_depth_m)
        assert riser.size_xyz_m[2] == pytest.approx(
            spec.top_heights_m[index]
        )
    observation = course.to_observation()
    assert observation["scenario"] == {
        "generator": "uniform_stair_v1",
        **spec.to_dict(),
    }
    assert observation["stairs"]["top_heights_m"] == pytest.approx(
        spec.top_heights_m
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"rise_m": 0.080},
        {"tread_depth_m": 0.100},
        {"step_count": 0},
    ),
)
def test_uniform_stair_spec_rejects_unsupported_geometry(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        UniformStairSpec(**kwargs)
