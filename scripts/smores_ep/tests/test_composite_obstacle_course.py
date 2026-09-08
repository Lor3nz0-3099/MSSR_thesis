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
    assert all(box.size_xyz_m[1] == pytest.approx(1.10) for box in road)
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
