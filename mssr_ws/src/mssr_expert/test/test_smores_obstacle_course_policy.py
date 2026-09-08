"""Tests for the library-driven composite mission policy."""

from pathlib import Path

import pytest

from mssr_expert.graph.serialization import load_attributed_graph
from mssr_expert.planning.smores_ep.obstacle_course_policy import (
    MissionTask,
    ObstacleCoursePolicy,
    ValidatedSeedCatalog,
)


PACKAGE_ROOT = Path(__file__).parents[1]


def _catalog() -> dict:
    return {
        name: load_attributed_graph(
            PACKAGE_ROOT / "config" / f"smores_{name}.json"
        )
        for name in ("rc_car8", "snake8", "mobile_manipulator8")
    }


def _policy() -> ObstacleCoursePolicy:
    return ObstacleCoursePolicy.from_morphology_catalog(_catalog())


def test_policy_reads_capabilities_from_target_graphs() -> None:
    policy = _policy()

    assert policy.choose_morphology("flat_navigation") == "rc_car8"
    assert policy.choose_morphology("cross_gap") == "snake8"
    assert policy.choose_morphology("climb_stairs") == "snake8"
    assert policy.choose_morphology("press_button") == "mobile_manipulator8"


def test_policy_keeps_a_compatible_current_morphology() -> None:
    policy = _policy()
    tasks = (
        MissionTask("gap-a", "gap", {"seed": 4100}),
        MissionTask("stairs-a", "stairs", {"seed": 6403}),
        MissionTask("gap-b", "gap", {"seed": 4102}),
        MissionTask("goal", "goal", {"x_m": 10.0}),
    )

    steps = policy.plan(tasks)

    assert [step.morphology for step in steps] == [
        "snake8",
        "snake8",
        "snake8",
        "rc_car8",
    ]
    assert not policy.requires_reconfiguration("snake8", "snake8")


def test_ramp_is_not_a_supported_task() -> None:
    policy = _policy()

    assert "ramp" not in policy.supported_task_types
    with pytest.raises(ValueError, match="Unsupported"):
        policy.resolve(MissionTask("forbidden", "ramp"))


def test_mission_requires_a_terminal_goal_and_unique_ids() -> None:
    policy = _policy()

    with pytest.raises(ValueError, match="terminate with a goal"):
        policy.plan((MissionTask("gap", "gap"),))
    with pytest.raises(ValueError, match="Duplicate"):
        policy.plan(
            (
                MissionTask("same", "gap"),
                MissionTask("same", "goal"),
            )
        )


def test_seed_catalog_accepts_only_canonical_validated_seeds() -> None:
    catalog = ValidatedSeedCatalog.load(
        PACKAGE_ROOT / "config" / "smores_composite_seed_catalog.json"
    )

    catalog.require("stairs", 6403)
    catalog.require("button", 6251)
    with pytest.raises(ValueError, match="not validated"):
        catalog.require("stairs", 6402)


def test_legacy_expansion_contains_no_ramp() -> None:
    steps = _policy().steps()

    assert all("ramp" not in step.task_type for step in steps)
    assert steps[0].morphology == "rc_car8"
