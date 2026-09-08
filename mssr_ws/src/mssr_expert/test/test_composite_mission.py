"""Tests for morphology-aware composite stage expansion."""

from mssr_expert.planning.smores_ep.composite_mission import (
    CompositeMissionPlanner,
)
from mssr_expert.planning.smores_ep.obstacle_course_policy import (
    MissionTask,
    ObstacleCoursePolicy,
)


def _planner() -> CompositeMissionPlanner:
    return CompositeMissionPlanner(
        ObstacleCoursePolicy(
            {
                "rc_car8": {"flat_navigation"},
                "snake8": {"cross_gap", "climb_stairs"},
                "mobile_manipulator8": {"press_button"},
            }
        )
    )


def test_consecutive_snake_tasks_do_not_reconfigure() -> None:
    stages = _planner().build(
        (
            MissionTask("gap-a", "gap"),
            MissionTask("stairs-a", "stairs"),
            MissionTask("gap-b", "gap"),
            MissionTask("goal", "goal"),
        )
    )

    transitions = [stage for stage in stages if stage.kind == "reconfiguration"]
    assert [(stage.source_morphology, stage.target_morphology) for stage in transitions] == [
        ("snake8", "rc_car8")
    ]


def test_button_uses_validated_rc_mm8_rc_route() -> None:
    stages = _planner().build(
        (
            MissionTask("stairs", "stairs"),
            MissionTask("button", "button"),
            MissionTask("goal", "goal"),
        )
    )

    button_stages = [stage for stage in stages if stage.task_id == "button"]
    assert [stage.kind for stage in button_stages] == [
        "reconfiguration",
        "button_rc_alignment",
        "reconfiguration",
        "button_expert",
        "reconfiguration",
    ]
    assert [stage.target_morphology for stage in button_stages] == [
        "rc_car8",
        "rc_car8",
        "mobile_manipulator8",
        "mobile_manipulator8",
        "rc_car8",
    ]
