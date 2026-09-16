"""Tests for ordered, non-descending composite course generation."""

import pytest

from smores_ep.isaac.obstacle_course import composite_obstacle_course


VALIDATED = {
    "flat_navigation": frozenset({5100}),
    "gap": frozenset({4100, 4102}),
    "stairs": frozenset({6403, 6922}),
    "button": frozenset({6251}),
}


def _mission(*tasks: dict) -> dict:
    return {
        "schema_version": "mssr.composite_mission.v1",
        "tasks": list(tasks),
    }


def test_multiple_stairs_raise_every_later_obstacle() -> None:
    course = composite_obstacle_course(
        _mission(
            {"task_id": "stairs-six", "type": "stairs", "seed": 6403},
            {"task_id": "gap-high", "type": "gap", "seed": 4100},
            {"task_id": "stairs-four", "type": "stairs", "seed": 6922},
            {"task_id": "button-high", "type": "button", "seed": 6251},
        ),
        VALIDATED,
    )

    tasks = {task["task_id"]: task for task in course.tasks}
    first_stairs = tasks["stairs-six"]["parameters"]
    first_top = first_stairs["upper_deck_height_m"]
    assert len(first_stairs["stairs"]["top_heights_m"]) == 6
    assert tasks["gap-high"]["parameters"]["floor_height_m"] == pytest.approx(
        first_top
    )
    second_stairs = tasks["stairs-four"]["parameters"]
    assert second_stairs["stairs"]["base_height_m"] == pytest.approx(first_top)
    second_top = second_stairs["upper_deck_height_m"]
    assert tasks["button-high"]["parameters"]["floor_height_m"] == pytest.approx(
        second_top
    )
    assert course.final_floor_height_m == pytest.approx(second_top)
    assert course.goal_center_xyz_m[2] == pytest.approx(second_top)


def test_repeated_obstacles_and_curved_navigation_are_preserved() -> None:
    course = composite_obstacle_course(
        _mission(
            {"task_id": "curve", "type": "flat_navigation", "seed": 5100},
            {"task_id": "gap-a", "type": "gap", "seed": 4100},
            {"task_id": "gap-b", "type": "gap", "seed": 4102},
            {"task_id": "button-a", "type": "button", "seed": 6251},
            {"task_id": "button-b", "type": "button", "seed": 6251},
        ),
        VALIDATED,
    )

    assert [task["type"] for task in course.tasks] == [
        "flat_navigation",
        "gap",
        "gap",
        "button",
        "button",
        "goal",
    ]
    route = course.tasks[0]["parameters"]["waypoints_xyyaw"]
    assert len(route) >= 7
    assert max(abs(point[1]) for point in route) > 0.1
    assert len(course.navigation_cones) == 6
    assert course.tasks[0]["parameters"]["cone_centers_xy_m"]
    road = [box for box in course.boxes if box.semantic == "flat_navigation_road"]
    assert road
    # Seed 5100 uses the corridor width from the validated successful run.
    assert all(box.size_xyz_m[1] == pytest.approx(0.84) for box in road)
    assert all(
        box.size_xyz_m[1] <= 1.40
        for box in course.boxes
        if box.semantic != "stair_test_riser"
    )
    assert len(course.buttons) == 2


def test_unvalidated_seed_is_rejected() -> None:
    with pytest.raises(ValueError, match="not validated"):
        composite_obstacle_course(
            _mission({"type": "stairs", "seed": 999999}),
            VALIDATED,
        )


def test_curved_navigation_carries_gap_and_stairs_to_its_exit_lane() -> None:
    course = composite_obstacle_course(
        _mission(
            {"task_id": "curve", "type": "flat_navigation", "seed": 5100},
            {"task_id": "gap", "type": "gap", "seed": 4100},
            {"task_id": "stairs", "type": "stairs", "seed": 6922},
            {"task_id": "button", "type": "button", "seed": 6251},
        ),
        VALIDATED,
    )

    route = course.tasks[0]["parameters"]["waypoints_xyyaw"]
    exit_y = route[-1][1]

    assert abs(exit_y) > 0.05

    gap_boxes = [
        box
        for box in course.boxes
        if box.semantic in {"gap_test_near_bank", "gap_test_far_bank"}
    ]
    assert gap_boxes
    assert all(box.center_xyz_m[1] == pytest.approx(exit_y) for box in gap_boxes)

    stair_boxes = [
        box
        for box in course.boxes
        if box.semantic in {
            "stair_test_start",
            "stair_test_riser",
            "stair_test_upper_deck",
        }
    ]
    assert stair_boxes
    assert all(
        box.center_xyz_m[1] == pytest.approx(exit_y)
        for box in stair_boxes
    )

    # The button zone now continues on the same lane reached by the
    # preceding RC/gap/stairs sequence.  No artificial second fold.
    assert course.goal_center_xyz_m[1] == pytest.approx(exit_y)

    button_platform = next(
        box
        for box in course.boxes
        if box.semantic == "button_test_platform"
    )
    assert button_platform.center_xyz_m[1] == pytest.approx(exit_y)

    assert not any(
        box.semantic == "button_turn_connector"
        for box in course.boxes
    )



def test_button_zone_stays_on_exit_lane_without_second_turn() -> None:
    course = composite_obstacle_course(
        _mission(
            {"task_id": "curve", "type": "flat_navigation", "seed": 5100},
            {"task_id": "gap", "type": "gap", "seed": 4100},
            {"task_id": "stairs", "type": "stairs", "seed": 6922},
            {"task_id": "button", "type": "button", "seed": 6251},
        ),
        VALIDATED,
    )

    route = course.tasks[0]["parameters"]["waypoints_xyyaw"]
    exit_y = route[-1][1]

    upper = next(
        box
        for box in course.boxes
        if box.semantic == "stair_test_upper_deck"
    )
    button_platform = next(
        box
        for box in course.boxes
        if box.semantic == "button_test_platform"
    )

    # Gap, stairs and button remain on the same lane reached by
    # the single RC curve.
    assert upper.center_xyz_m[1] == pytest.approx(exit_y)
    assert button_platform.center_xyz_m[1] == pytest.approx(exit_y)
    assert course.goal_center_xyz_m[1] == pytest.approx(exit_y)

    # The old transverse bridge represented the unwanted second turn.
    assert not any(
        box.semantic == "button_turn_connector"
        for box in course.boxes
    )

    # A button-last mission terminates on the button platform itself.
    assert not any(
        box.name == "CompositeGoalPlatform"
        for box in course.boxes
    )

    half_x = 0.5 * button_platform.size_xyz_m[0]
    min_x = button_platform.center_xyz_m[0] - half_x
    max_x = button_platform.center_xyz_m[0] + half_x

    assert min_x <= course.goal_center_xyz_m[0] <= max_x


def test_diagnostic_short_rc_gap_profile_keeps_nav_local_and_coarse() -> None:
    course = composite_obstacle_course(
        _mission(
            {
                "task_id": "short-rc",
                "type": "flat_navigation",
                "seed": 5100,
                "diagnostic_route_length_m": 0.05,
                "navigation_accept_position_m": 0.20,
                "navigation_accept_yaw_rad": 0.7853981633974483,
            },
            {
                "task_id": "gap",
                "type": "gap",
                "seed": 4100,
            },
        ),
        VALIDATED,
    )

    navigation = course.tasks[0]["parameters"]
    route = navigation["waypoints_xyyaw"]
    goal = navigation["navigation_goal_xyyaw"]

    assert len(route) == 2

    travel_m = (
        (goal[0] - route[0][0]) ** 2
        + (goal[1] - route[0][1]) ** 2
    ) ** 0.5

    assert 0.40 <= travel_m <= 0.60

    assert navigation[
        "navigation_accept_position_m"
    ] == pytest.approx(0.20)

    assert navigation[
        "navigation_accept_yaw_rad"
    ] == pytest.approx(0.7853981633974483)

    assert navigation["cone_centers_xy_m"] == []
    assert not course.navigation_cones
