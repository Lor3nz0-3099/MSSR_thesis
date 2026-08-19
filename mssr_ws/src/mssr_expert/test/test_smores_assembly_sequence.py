"""Tests for parallel SMORES-EP assembly-wave generation."""

from __future__ import annotations

import pytest

from mssr_expert.planning.smores_ep.assembly_sequence import (
    AssemblySequenceError,
    generate_parallel_assembly_plan,
)
from mssr_expert.planning.smores_ep.assignment import (
    AssignmentResult,
)
from mssr_expert.planning.smores_ep.rooting import (
    root_kinematic_tree,
)
from mssr_expert.planning.smores_ep.topology import (
    SmoresKinematicTree,
    SmoresTopologyEdge,
)


def _assignment(mapping: dict[str, str]) -> AssignmentResult:
    return AssignmentResult(
        target_to_module=mapping,
        cost_by_target={
            target_id: 0.0
            for target_id in mapping
        },
        total_cost=0.0,
    )


def test_root_left_right_actions_precede_top_bottom() -> None:
    topology = SmoresKinematicTree(
        vertex_ids=(
            "root",
            "left_child",
            "right_child",
            "top_child",
            "bottom_child",
        ),
        edges=(
            SmoresTopologyEdge(
                "root",
                "LEFT",
                "left_child",
                "RIGHT",
            ),
            SmoresTopologyEdge(
                "root",
                "RIGHT",
                "right_child",
                "LEFT",
            ),
            SmoresTopologyEdge(
                "root",
                "TOP",
                "top_child",
                "BOTTOM",
            ),
            SmoresTopologyEdge(
                "root",
                "BOTTOM",
                "bottom_child",
                "TOP",
            ),
        ),
    )

    rooted = root_kinematic_tree(
        topology,
        root_id="root",
    )

    plan = generate_parallel_assembly_plan(
        tree=rooted,
        assignment=_assignment(
            {
                "root": "m0",
                "left_child": "m1",
                "right_child": "m2",
                "top_child": "m3",
                "bottom_child": "m4",
            }
        ),
    )

    assert len(plan.waves) == 2

    assert plan.waves[0].phase == "ROOT_LEFT_RIGHT"
    assert {
        action.parent_face
        for action in plan.waves[0].actions
    } == {"LEFT", "RIGHT"}

    assert plan.waves[1].phase == "ROOT_TOP_BOTTOM"
    assert {
        action.parent_face
        for action in plan.waves[1].actions
    } == {"TOP", "BOTTOM"}

    assert plan.action_count == 4


def test_actions_are_generated_from_root_to_leaves() -> None:
    topology = SmoresKinematicTree(
        vertex_ids=("v0", "v1", "v2"),
        edges=(
            SmoresTopologyEdge(
                "v0",
                "TOP",
                "v1",
                "BOTTOM",
            ),
            SmoresTopologyEdge(
                "v1",
                "TOP",
                "v2",
                "BOTTOM",
            ),
        ),
    )

    rooted = root_kinematic_tree(
        topology,
        root_id="v0",
    )

    plan = generate_parallel_assembly_plan(
        tree=rooted,
        assignment=_assignment(
            {
                "v0": "m0",
                "v1": "m1",
                "v2": "m2",
            }
        ),
    )

    assert len(plan.waves) == 2
    assert plan.waves[0].depth == 1
    assert plan.waves[1].depth == 2

    first_action = plan.waves[0].actions[0]
    second_action = plan.waves[1].actions[0]

    assert first_action.mobile_module_id == "m1"
    assert first_action.parent_module_id == "m0"

    assert second_action.mobile_module_id == "m2"
    assert second_action.parent_module_id == "m1"


def test_same_depth_actions_are_in_one_parallel_wave() -> None:
    topology = SmoresKinematicTree(
        vertex_ids=("root", "left", "right"),
        edges=(
            SmoresTopologyEdge(
                "root",
                "TOP",
                "left",
                "BOTTOM",
            ),
            SmoresTopologyEdge(
                "root",
                "BOTTOM",
                "right",
                "TOP",
            ),
        ),
    )

    rooted = root_kinematic_tree(
        topology,
        root_id="root",
    )

    plan = generate_parallel_assembly_plan(
        tree=rooted,
        assignment=_assignment(
            {
                "root": "m0",
                "left": "m1",
                "right": "m2",
            }
        ),
    )

    assert len(plan.waves) == 1
    assert plan.waves[0].phase == "ROOT_TOP_BOTTOM"
    assert plan.waves[0].can_execute_in_parallel
    assert len(plan.waves[0].actions) == 2


def test_wheel_face_on_mobile_module_requires_helper() -> None:
    topology = SmoresKinematicTree(
        vertex_ids=("root", "child"),
        edges=(
            SmoresTopologyEdge(
                "root",
                "TOP",
                "child",
                "LEFT",
            ),
        ),
    )

    rooted = root_kinematic_tree(
        topology,
        root_id="root",
    )

    plan = generate_parallel_assembly_plan(
        tree=rooted,
        assignment=_assignment(
            {
                "root": "m0",
                "child": "m1",
            }
        ),
    )

    action = plan.all_actions[0]

    assert action.mobile_face == "LEFT"
    assert action.parent_face == "TOP"
    assert action.requires_helper


def test_top_bottom_mobile_face_does_not_require_helper() -> None:
    topology = SmoresKinematicTree(
        vertex_ids=("root", "child"),
        edges=(
            SmoresTopologyEdge(
                "root",
                "TOP",
                "child",
                "BOTTOM",
            ),
        ),
    )

    rooted = root_kinematic_tree(
        topology,
        root_id="root",
    )

    plan = generate_parallel_assembly_plan(
        tree=rooted,
        assignment=_assignment(
            {
                "root": "m0",
                "child": "m1",
            }
        ),
    )

    assert not plan.all_actions[0].requires_helper


def test_missing_assignment_is_rejected() -> None:
    topology = SmoresKinematicTree(
        vertex_ids=("root", "child"),
        edges=(
            SmoresTopologyEdge(
                "root",
                "TOP",
                "child",
                "BOTTOM",
            ),
        ),
    )

    rooted = root_kinematic_tree(
        topology,
        root_id="root",
    )

    with pytest.raises(
        AssemblySequenceError,
        match="do not match",
    ):
        generate_parallel_assembly_plan(
            tree=rooted,
            assignment=_assignment(
                {
                    "root": "m0",
                }
            ),
        )


def test_single_module_has_no_assembly_actions() -> None:
    topology = SmoresKinematicTree(
        vertex_ids=("root",),
        edges=(),
    )

    rooted = root_kinematic_tree(
        topology,
        root_id="root",
    )

    plan = generate_parallel_assembly_plan(
        tree=rooted,
        assignment=_assignment(
            {
                "root": "m0",
            }
        ),
    )

    assert plan.waves == ()
    assert plan.all_actions == ()
    assert plan.action_count == 0