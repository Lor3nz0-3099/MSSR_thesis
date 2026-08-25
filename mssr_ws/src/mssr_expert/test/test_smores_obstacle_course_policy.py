"""Tests for the deterministic SMORES-EP obstacle-course policy."""

from pathlib import Path

from mssr_expert.behaviors.morphology_library import (
    AssignedModule,
    MorphologyLibrary,
)
from mssr_expert.execution.morphology_behavior_executor import (
    MorphologyBehaviorExecutor,
    MorphologyCommand,
)
from mssr_expert.graph.serialization import load_attributed_graph
from mssr_expert.planning.smores_ep.attributed_adapter import (
    target_roles_from_graph,
)
from mssr_expert.planning.smores_ep.obstacle_course_policy import (
    ObstacleCoursePolicy,
)


def test_policy_selects_one_capable_morphology_per_course_task() -> None:
    steps = ObstacleCoursePolicy().steps()

    assert [step.morphology for step in steps] == (
        ["rc_car8"] * 3
        + ["snake8"] * 7
        + ["mobile_manipulator8"] * 5
        + ["rc_car8"] * 3
    )
    assert next(step for step in steps if step.task == "button").requires_button
    assert next(step for step in steps if step.task == "exit").requires_goal
    assert [step.task for step in steps[:3]] == [
        "assembly",
        "ramp_climb",
        "rc_car_pre_gap",
    ]
    assert all(step.behavior != "straighten" for step in steps)
    assert {step.navigation for step in steps if step.navigation} == {
        "front_before_gap",
        "ramp_exit",
        "safe_before_snake_reconfiguration",
        "rear_past_gap",
        "front_before_stair_1",
        "front_on_upper_deck",
        "button_standoff",
        "button_retreat",
        "cross_exit",
    }
    stair_crawl = next(step for step in steps if step.task == "stairs_crawl")
    assert stair_crawl.behavior == "crawl_stairs_arch_wave"
    assert stair_crawl.parameters == {
        "linear_m_s": 0.040,
        "riser_approach_linear_m_s": 0.060,
        "riser_approach_tolerance_m": 0.010,
        "crawl_goal_tolerance_m": 0.004,
        "profile_substeps": 6,
        "transition_clearance_m": 0.0065,
    }


def test_every_course_behavior_accepts_its_target_roles_and_parameters() -> None:
    package_root = Path(__file__).parents[1]
    library = MorphologyLibrary.load(
        package_root / "config" / "smores_morphology_behaviors.json"
    )
    policy = ObstacleCoursePolicy()

    for step in policy.steps():
        if step.behavior is None:
            continue
        if step.behavior in {"crawl_stairs", "crawl_stairs_arch_wave"}:
            # This behavior is generated from live world poses and course
            # landmarks by SnakeStairGaitPlanner, rather than loaded from the
            # static morphology library.
            continue
        target_graph = load_attributed_graph(
            package_root / "config" / f"smores_{step.morphology}.json"
        )
        roles = target_roles_from_graph(target_graph)
        assignments = tuple(
            AssignedModule(
                module_id=f"module_{vertex}",
                target_vertex_id=vertex,
                target_role=str(attributes["target_role"]),
            )
            for vertex, attributes in sorted(roles.items())
        )
        executor = MorphologyBehaviorExecutor(library)
        neutral_tilts = (
            {assignment.module_id: 0.1 for assignment in assignments}
            if library.uses_captured_neutral(step.morphology)
            else {}
        )

        executor.start(
            MorphologyCommand(
                command_id=f"course-{step.task}",
                morphology=step.morphology,
                behavior=step.behavior,
                parameters=step.parameters or {},
            ),
            assignments,
            neutral_tilts,
        )
