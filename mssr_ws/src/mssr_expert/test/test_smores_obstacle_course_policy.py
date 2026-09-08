"""Tests for the deterministic SMORES-EP obstacle-course policy."""

import pytest

from pathlib import Path

from mssr_expert.behaviors.morphology_library import (
    AssignedModule,
    MorphologyLibrary,
)
from mssr_expert.behaviors.snake_stair_registry import STAIR_GAIT_BEHAVIORS
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
        ["snake8"] * 8
        + ["rc_car8"] * 2
        + ["mobile_manipulator8"] * 6
        + ["rc_car8"] * 3
    )

    assert len(steps) == 19

def test_only_requested_stair_gaits_are_public() -> None:
    assert STAIR_GAIT_BEHAVIORS == {
        "crawl_stairs_arch_wave",
        "crawl_stairs_spatial_concertina",
    }
    assert "crawl_stairs" not in STAIR_GAIT_BEHAVIORS


def test_every_course_behavior_accepts_its_target_roles_and_parameters() -> None:
    package_root = Path(__file__).parents[1]
    library = MorphologyLibrary.load(
        package_root / "config" / "smores_morphology_behaviors.json"
    )
    policy = ObstacleCoursePolicy()

    for step in policy.steps():
        if step.behavior is None:
            continue
        if step.behavior in {
            "crawl_stairs_arch_wave",
            "crawl_stairs_spatial_concertina",
            "gap_crossing",
        }:
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


def test_policy_resolves_button_targeted_start_index() -> None:
    policy = ObstacleCoursePolicy()

    assert policy.step_index(
        "button_rc_car_pre_alignment"
    ) == 9

    assert policy.steps()[9].morphology == "rc_car8"


def test_policy_rejects_unknown_targeted_start_task() -> None:
    policy = ObstacleCoursePolicy()

    with pytest.raises(ValueError):
        policy.step_index("not_a_real_course_task")
