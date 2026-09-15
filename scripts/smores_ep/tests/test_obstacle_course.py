
from __future__ import annotations
import math

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
        [0.85, 0.465, 0.17]
    )
    assert observation["button"]["base_standoff_xy_m"] == pytest.approx(
        [0.85, 0.265]
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



def test_button_target_sampling_is_seed_reproducible() -> None:
    from smores_ep.isaac.obstacle_course import (
        sample_button_target_spec,
    )

    first = sample_button_target_spec(6100)
    repeat = sample_button_target_spec(6100)

    assert first == repeat
    assert first.seed == 6100


def test_button_target_sampling_varies_position_and_height() -> None:
    from smores_ep.isaac.obstacle_course import (
        BUTTON_SAMPLE_X_RANGE_M,
        BUTTON_SAMPLE_Y_RANGE_M,
        BUTTON_SAMPLE_Z_RANGE_M,
        sample_button_target_spec,
    )

    first = sample_button_target_spec(6100)
    second = sample_button_target_spec(6101)

    assert first.center_xyz_m != second.center_xyz_m

    assert (
        BUTTON_SAMPLE_X_RANGE_M[0]
        <= first.x_m
        <= BUTTON_SAMPLE_X_RANGE_M[1]
    )
    assert (
        BUTTON_SAMPLE_Y_RANGE_M[0]
        <= first.y_m
        <= BUTTON_SAMPLE_Y_RANGE_M[1]
    )
    assert (
        BUTTON_SAMPLE_Z_RANGE_M[0]
        <= first.z_m
        <= BUTTON_SAMPLE_Z_RANGE_M[1]
    )


def test_button_course_consumes_dynamic_xyz_target() -> None:
    from smores_ep.isaac.obstacle_course import (
        ButtonTargetSpec,
        mobile_manipulator_button_test_course,
    )

    spec = ButtonTargetSpec(
        x_m=0.72,
        y_m=0.51,
        z_m=0.23,
        seed=6199,
    )

    course = mobile_manipulator_button_test_course(spec)
    observation = course.to_observation()

    assert course.button_center_xyz_m == pytest.approx(
        (0.72, 0.51, 0.23)
    )

    # Existing convention: MM8 nominal standoff is 20 cm toward -Y.
    assert course.base_standoff_xy_m == pytest.approx(
        (0.72, 0.31)
    )

    assert observation["button"]["center_xyz_m"] == pytest.approx(
        [0.72, 0.51, 0.23]
    )


def test_default_button_fixture_is_unchanged() -> None:
    from smores_ep.isaac.obstacle_course import (
        mobile_manipulator_button_test_course,
    )

    course = mobile_manipulator_button_test_course()

    assert course.button_center_xyz_m == pytest.approx(
        (0.85, 0.465, 0.170)
    )
    assert course.base_standoff_xy_m == pytest.approx(
        (0.85, 0.265)
    )


def test_composite_gap_rc_button_follows_compact_single_curve_layout() -> None:
    from smores_ep.isaac.obstacle_course import composite_obstacle_course

    mission = {
        "schema_version": "mssr.composite_mission.v1",
        "episode_id": "layout-c05",
        "tasks": [
            {"task_id": "gap-1", "type": "gap", "seed": 4100},
            {"task_id": "rc-1", "type": "flat_navigation", "seed": 5100},
            {"task_id": "button-1", "type": "button", "seed": 6101},
        ],
    }
    validated = {
        "gap": frozenset({4100}),
        "flat_navigation": frozenset({5100}),
        "button": frozenset({6101}),
    }

    course = composite_obstacle_course(mission, validated)
    tasks = {task["task_id"]: task for task in course.tasks}

    gap = tasks["gap-1"]["parameters"]["gap"]

    # Spawn/assembly area must no longer be followed by a very long
    # empty approach before the first gap.
    assert gap["near_edge_x_m"] < 0.0

    route = tasks["rc-1"]["parameters"]["waypoints_xyyaw"]

    # RC starts on the lower lane and finishes on the offset upper lane.
    assert route[0][1] == pytest.approx(0.0)
    assert route[-1][1] > 0.50

    # Regression for the old artificial return-to-y=0 point:
    # the RC task ends where its single physical curve actually ends.
    assert route[-1][1] != pytest.approx(0.0)

    button = tasks["button-1"]["parameters"]["button"]
    button_platform = next(
        box
        for box in course.boxes
        if box.semantic == "button_test_platform"
    )

    # The final button platform continues from the RC upper lane.
    assert button_platform.center_xyz_m[1] == pytest.approx(route[-1][1])

    # Seed 6101 presses along -X.  Therefore the robot must finish on
    # the +X side of the button before RC -> MM8 and the final press.
    # The button is still in the terminal platform, but is slightly
    # behind the RC endpoint along X by design.
    press_direction = button["press_direction_world_xy"]

    assert press_direction[0] < -0.5
    assert abs(press_direction[1]) < 0.5

    assert button["center_xyz_m"][1] == pytest.approx(route[-1][1])

    button_standoff_x_m = (
        route[-1][0] - button["center_xyz_m"][0]
    )

    assert 0.0 < button_standoff_x_m < 0.40

    # Button is the terminal platform: do not create another detached
    # goal platform after it.
    assert not any(
        box.name == "CompositeGoalPlatform"
        for box in course.boxes
    )

    half_x = 0.5 * button_platform.size_xyz_m[0]
    platform_min_x = button_platform.center_xyz_m[0] - half_x
    platform_max_x = button_platform.center_xyz_m[0] + half_x

    # Both the RC endpoint and the synthetic final goal remain inside
    # the same physical terminal button platform.
    assert platform_min_x <= route[-1][0] <= platform_max_x
    assert (
        platform_min_x
        <= button["center_xyz_m"][0]
        <= platform_max_x
    )
    assert platform_min_x <= course.goal_center_xyz_m[0] <= platform_max_x
    assert course.goal_center_xyz_m[1] == pytest.approx(route[-1][1])


def test_composite_repeated_rc_curves_alternate_lanes_instead_of_drifting() -> None:
    from smores_ep.isaac.obstacle_course import composite_obstacle_course

    mission = {
        "schema_version": "mssr.composite_mission.v1",
        "episode_id": "layout-repeated-rc",
        "tasks": [
            {"task_id": "rc-1", "type": "flat_navigation", "seed": 5100},
            {"task_id": "rc-2", "type": "flat_navigation", "seed": 5100},
            {"task_id": "button-1", "type": "button", "seed": 6101},
        ],
    }
    validated = {
        "flat_navigation": frozenset({5100}),
        "button": frozenset({6101}),
    }

    course = composite_obstacle_course(mission, validated)
    tasks = {task["task_id"]: task for task in course.tasks}

    first = tasks["rc-1"]["parameters"]["waypoints_xyyaw"]
    second = tasks["rc-2"]["parameters"]["waypoints_xyyaw"]

    assert first[-1][1] > 0.50

    # Second task starts from the first task's lane...
    assert second[0][1] == pytest.approx(first[-1][1])

    # ...then mirrors the same validated curve instead of moving yet
    # farther away from the course.
    assert abs(second[-1][1]) < 0.15


def test_composite_c05_rc_exit_handoff_starts_button_platform_at_course_end() -> None:
    from smores_ep.isaac.obstacle_course import composite_obstacle_course

    mission = {
        "schema_version": "mssr.composite_mission.v1",
        "episode_id": "composite-c05",
        "tasks": [
            {"task_id": "gap-1", "type": "gap", "seed": 4100},
            {"task_id": "rc-1", "type": "flat_navigation", "seed": 5100},
            {"task_id": "button-1", "type": "button", "seed": 6101},
        ],
    }

    validated = {
        "gap": frozenset({4100}),
        "flat_navigation": frozenset({5100}),
        "button": frozenset({6101}),
    }

    course = composite_obstacle_course(mission, validated)

    rc_task = next(
        task for task in course.tasks
        if task["task_id"] == "rc-1"
    )
    params = rc_task["parameters"]

    route = params["waypoints_xyyaw"]
    goal = params["navigation_goal_xyyaw"]

    button_platform = next(
        box for box in course.boxes
        if box.semantic == "button_test_platform"
    )

    cx, cy, _ = button_platform.center_xyz_m
    sx, sy, _ = button_platform.size_xyz_m

    button_left_x = float(cx) - 0.5 * float(sx)
    button_right_x = float(cx) + 0.5 * float(sx)
    button_low_y = float(cy) - 0.5 * float(sy)
    button_high_y = float(cy) + 0.5 * float(sy)

    # Keep the original longitudinal X placement of the button stage.
    # The button platform is moved only in Y.
    assert button_left_x == pytest.approx(route[-1][0] - 0.90)

    # C05 exits the RC curve at yaw ~= +pi/2, so "behind/in depth" is +Y.
    # The near Y edge of the button platform starts exactly where the
    # RC course ends.
    assert button_low_y == pytest.approx(route[-1][1])

    # C05 uses both coordinates selected directly with RViz
    # Publish Point.  Do not mix them with route-derived coordinates.
    assert goal[0] == pytest.approx(2.317791700363159)
    assert goal[1] == pytest.approx(0.9919289946556091)

    # NavigateToPose still requires a yaw field, but completion is
    # position-only for this handoff.
    assert goal[2] == pytest.approx(route[-1][2])
    assert params["navigation_accept_position_m"] == pytest.approx(0.10)
    assert params["navigation_accept_yaw_rad"] == pytest.approx(
        3.141592653589793
    )

    # The clicked goal remains laterally inside the platform X span,
    # while its Y lies before the near edge of the button platform.
    assert button_left_x <= goal[0] <= button_right_x
    assert goal[1] < button_low_y

    # The platform itself remains available to the following button expert
    # through the extended navigation map.
    assert button_low_y <= float(cy) <= button_high_y

def test_composite_c05_rc_map_includes_button_platform() -> None:
    from smores_ep.isaac.obstacle_course import composite_obstacle_course

    mission = {
        "schema_version": "mssr.composite_mission.v1",
        "episode_id": "composite-c05",
        "tasks": [
            {"task_id": "gap-1", "type": "gap", "seed": 4100},
            {"task_id": "rc-1", "type": "flat_navigation", "seed": 5100},
            {"task_id": "button-1", "type": "button", "seed": 6101},
        ],
    }

    validated = {
        "gap": frozenset({4100}),
        "flat_navigation": frozenset({5100}),
        "button": frozenset({6101}),
    }

    course = composite_obstacle_course(mission, validated)

    rc_task = next(
        task for task in course.tasks
        if task["task_id"] == "rc-1"
    )
    params = rc_task["parameters"]

    button_platform = next(
        box for box in course.boxes
        if box.semantic == "button_test_platform"
    )

    cx, cy, _ = button_platform.center_xyz_m
    sx, sy, _ = button_platform.size_xyz_m

    button_bounds = [
        float(cx) - 0.5 * float(sx),
        float(cx) + 0.5 * float(sx),
        float(cy) - 0.5 * float(sy),
        float(cy) + 0.5 * float(sy),
    ]

    free_rectangles = params.get(
        "navigation_free_rectangles_xy_m",
        [],
    )

    assert any(
        all(
            actual == pytest.approx(expected)
            for actual, expected in zip(rectangle, button_bounds)
        )
        for rectangle in free_rectangles
    )

    map_bounds = params["platform_bounds_xy_m"]

    assert map_bounds[0] <= button_bounds[0]
    assert map_bounds[1] >= button_bounds[1]
    assert map_bounds[2] <= button_bounds[2]
    assert map_bounds[3] >= button_bounds[3]

    gx, gy, _ = params["navigation_goal_xyyaw"]

    # The exact clicked goal remains inside the mapped area and
    # before the near Y edge of the button platform.
    assert button_bounds[0] <= gx <= button_bounds[1]
    assert gy < button_bounds[2]
    assert map_bounds[0] <= gx <= map_bounds[1]
    assert map_bounds[2] <= gy <= map_bounds[3]

    # The OccupancyGrid must nevertheless include both the exit goal and
    # the complete shifted button platform.
    assert map_bounds[0] <= gx <= map_bounds[1]
    assert map_bounds[2] <= gy <= map_bounds[3]


def test_composite_c05_rc_uses_exact_clicked_rviz_goal() -> None:
    from smores_ep.isaac.obstacle_course import composite_obstacle_course

    mission = {
        "schema_version": "mssr.composite_mission.v1",
        "episode_id": "composite-c05",
        "tasks": [
            {"task_id": "gap-1", "type": "gap", "seed": 4100},
            {"task_id": "rc-1", "type": "flat_navigation", "seed": 5100},
            {"task_id": "button-1", "type": "button", "seed": 6101},
        ],
    }

    validated = {
        "gap": frozenset({4100}),
        "flat_navigation": frozenset({5100}),
        "button": frozenset({6101}),
    }

    course = composite_obstacle_course(mission, validated)

    params = next(
        task for task in course.tasks
        if task["task_id"] == "rc-1"
    )["parameters"]

    route = params["waypoints_xyyaw"]
    goal = params["navigation_goal_xyyaw"]

    # Exact point selected with RViz Publish Point.
    assert goal[0] == pytest.approx(2.317791700363159)
    assert goal[1] == pytest.approx(0.9919289946556091)

    # Yaw remains only a NavigateToPose placeholder.
    assert goal[2] == pytest.approx(route[-1][2])

    assert "navigation_via_xyyaw" not in params
    assert params["navigation_accept_position_m"] == pytest.approx(0.10)
    assert params["navigation_accept_yaw_rad"] == pytest.approx(
        3.141592653589793
    )


def test_composite_c05_moves_only_rc_cone5() -> None:
    from smores_ep.isaac.obstacle_course import composite_obstacle_course

    def build(episode_id: str):
        mission = {
            "schema_version": "mssr.composite_mission.v1",
            "episode_id": episode_id,
            "tasks": [
                {"task_id": "gap-1", "type": "gap", "seed": 4100},
                {"task_id": "rc-1", "type": "flat_navigation", "seed": 5100},
                {"task_id": "button-1", "type": "button", "seed": 6101},
            ],
        }

        validated = {
            "gap": frozenset({4100}),
            "flat_navigation": frozenset({5100}),
            "button": frozenset({6101}),
        }

        return composite_obstacle_course(mission, validated)

    c05 = build("composite-c05")
    control = build("not-composite-c05")

    c05_rc = next(
        task for task in c05.tasks
        if task["task_id"] == "rc-1"
    )
    control_rc = next(
        task for task in control.tasks
        if task["task_id"] == "rc-1"
    )

    c05_cones = c05_rc["parameters"]["cone_centers_xy_m"]
    control_cones = control_rc["parameters"]["cone_centers_xy_m"]

    assert len(c05_cones) >= 5
    assert len(control_cones) >= 5

    # C05 alone moves cone 5 to the manually selected Isaac X.
    assert c05_cones[4][0] == pytest.approx(2.63851)

    # Preserve cone 5 Y exactly.
    assert c05_cones[4][1] == pytest.approx(control_cones[4][1])

    # Only cone 5 changes; cones 1-4 and 6+ remain untouched.
    for index in range(len(c05_cones)):
        if index == 4:
            continue
        assert c05_cones[index] == pytest.approx(control_cones[index])

    # The generic seed-5100 layout must remain unchanged.
    assert control_cones[4][0] != pytest.approx(2.63851)

    # Physical composite cone and navigation metadata must agree.
    physical = [
        cone for cone in c05.navigation_cones
        if cone.task_id == "rc-1"
    ]

    assert len(physical) == len(c05_cones)
    assert physical[4].center_xyz_m[0] == pytest.approx(2.63851)
    assert physical[4].center_xyz_m[1] == pytest.approx(c05_cones[4][1])
    assert physical[4].center_xyz_m[2] == pytest.approx(0.085)
