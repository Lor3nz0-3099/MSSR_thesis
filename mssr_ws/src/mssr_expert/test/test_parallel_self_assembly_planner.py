"""End-to-end tests for the deterministic self-assembly planner."""

from __future__ import annotations

import math

import pytest

from mssr_expert.graph.attributed_robot_graph import (
    AttributedRobotGraph,
    GraphEdge,
    GraphNode,
)
from mssr_expert.planning.smores_ep.assignment import AssignmentResult
from mssr_expert.planning.smores_ep.parallel_self_assembly_planner import (
    ParallelSelfAssemblyPlanner,
    ParallelSelfAssemblyPlannerError,
)
from mssr_expert.planning.smores_ep.rooting import (
    RootedSmoresEdge,
    RootedSmoresTree,
)
from mssr_expert.planning.smores_ep.unfolding import (
    PlanarPose,
    UnfoldedPlanarConfiguration,
)


SPACING = 0.043771 + 0.033999


def _quaternion_from_yaw(
    yaw_rad: float,
) -> list[float]:
    return [
        0.0,
        0.0,
        math.sin(yaw_rad / 2.0),
        math.cos(yaw_rad / 2.0),
    ]


def _physical_node(
    module_id: str,
    x_m: float,
    y_m: float,
    yaw_rad: float,
    robot_family: str = "smores_ep",
) -> GraphNode:
    return GraphNode(
        module_id,
        {
            "node_type": "physical_module",
            "robot_family": robot_family,
            "position": [x_m, y_m, 0.0316],
            "orientation": _quaternion_from_yaw(yaw_rad),
            "control_available": True,
        },
    )


def _current_graph() -> AttributedRobotGraph:
    root_x = 10.0
    root_y = 5.0
    root_yaw = math.pi / 2.0

    return AttributedRobotGraph(
        stamp=1.0,
        nodes=(
            _physical_node(
                "root_module",
                root_x,
                root_y,
                root_yaw,
            ),
            _physical_node(
                "negative_module",
                root_x,
                root_y - SPACING,
                root_yaw,
            ),
            _physical_node(
                "positive_module",
                root_x,
                root_y + SPACING,
                root_yaw,
            ),
        ),
        edges=(),
        global_attributes={
            "task_type": "parallel_self_assembly",
        },
    )


def _target_graph() -> AttributedRobotGraph:
    return AttributedRobotGraph(
        nodes=(
            GraphNode(
                "v0",
                {
                    "target_vertex_id": "v0",
                    "target_role": "left_link",
                    "functional_role": {
                        "name": "link",
                    },
                },
            ),
            GraphNode(
                "v1",
                {
                    "target_vertex_id": "v1",
                    "target_role": "support",
                    "functional_role": {
                        "name": "support",
                    },
                },
            ),
            GraphNode(
                "v2",
                {
                    "target_vertex_id": "v2",
                    "target_role": "right_link",
                    "functional_role": {
                        "name": "link",
                    },
                },
            ),
        ),
        edges=(
            GraphEdge(
                "v0",
                "v1",
                {
                    "relation_type": "target_connection",
                    "is_target_edge": True,
                    "face_a": "TOP",
                    "face_b": "BOTTOM",
                    "clocking_quarter_turns": 0,
                },
            ),
            GraphEdge(
                "v1",
                "v2",
                {
                    "relation_type": "target_connection",
                    "is_target_edge": True,
                    "face_a": "TOP",
                    "face_b": "BOTTOM",
                    "clocking_quarter_turns": 0,
                },
            ),
        ),
        global_attributes={
            "schema_version": "mssr.target_graph.v1",
            "graph_kind": "target_morphology",
        },
    )


def test_complete_parallel_self_assembly_plan() -> None:
    result = ParallelSelfAssemblyPlanner().plan(
        current_graph=_current_graph(),
        target_graph=_target_graph(),
    )

    assert result.rooted_target_tree.root_id == "v1"
    assert result.physical_root_id == "root_module"

    assert result.assignment.target_to_module == {
        "v0": "negative_module",
        "v1": "root_module",
        "v2": "positive_module",
    }

    assert result.assignment.total_cost == pytest.approx(0.0)

    assert result.assembly_plan.root_target_vertex == "v1"
    assert result.assembly_plan.root_module_id == "root_module"
    assert result.assembly_plan.action_count == 2
    assert len(result.assembly_plan.waves) == 1

    assert result.assembly_plan.waves[0].phase == (
        "ROOT_TOP_BOTTOM"
    )

    assert len(
        result.task_graph.edges_matching("target_connection")
    ) == 2

    assert len(
        result.task_graph.edges_matching("assignment")
    ) == 3

    task_nodes = result.task_graph.node_by_id()

    assert task_nodes["target:v1"].attributes[
        "is_target_root"
    ]

    assert task_nodes["root_module"].attributes[
        "target_vertex_id"
    ] == "v1"

    assert task_nodes["root_module"].attributes[
        "target_role"
    ] == "support"

    assert result.task_graph.global_attributes[
        "execution_state"
    ]["phase"] == "PLANNED"


def test_layout_clearance_accumulates_correctly_at_depth_two() -> None:
    planner = ParallelSelfAssemblyPlanner(layout_clearance_m=0.070)
    target = UnfoldedPlanarConfiguration(
        root_id="v0",
        poses_by_vertex={
            "v0": PlanarPose(0.0, 0.0, 0.0),
            "v1": PlanarPose(0.1, 0.0, 0.0),
            "v2": PlanarPose(0.2, 0.0, 0.0),
        },
    )
    rooted_tree = RootedSmoresTree(
        root_id="v0",
        vertex_ids=("v0", "v1", "v2"),
        edges=(
            RootedSmoresEdge("v0", "TOP", "v1", "BOTTOM", 0),
            RootedSmoresEdge("v1", "TOP", "v2", "BOTTOM", 0),
        ),
        parent_by_vertex={"v0": None, "v1": "v0", "v2": "v1"},
        depth_by_vertex={"v0": 0, "v1": 1, "v2": 2},
    )
    assignment = AssignmentResult(
        target_to_module={"v0": "m0", "v1": "m1", "v2": "m2"},
        cost_by_target={},
        total_cost=0.0,
    )

    poses = planner._layout_poses(
        physical_poses={
            "m0": PlanarPose(10.0, 5.0, 0.0),
            "m1": PlanarPose(0.0, 0.0, 0.0),
            "m2": PlanarPose(0.0, 0.0, 0.0),
        },
        physical_root_id="m0",
        target=target,
        assignment=assignment,
        rooted_tree=rooted_tree,
    )

    assert poses["m1"].x_m == pytest.approx(10.17)
    assert poses["m2"].x_m == pytest.approx(10.34)


def test_already_connected_modules_are_rejected() -> None:
    current = _current_graph()

    connected = AttributedRobotGraph(
        stamp=current.stamp,
        nodes=current.nodes,
        edges=(
            GraphEdge(
                "root_module",
                "negative_module",
                {
                    "relation_type": "current_connection",
                    "is_attached": True,
                    "face_a": "TOP",
                    "face_b": "BOTTOM",
                },
            ),
        ),
        global_attributes=current.global_attributes,
    )

    with pytest.raises(
        ParallelSelfAssemblyPlannerError,
        match="initially separated",
    ):
        ParallelSelfAssemblyPlanner().plan(
            current_graph=connected,
            target_graph=_target_graph(),
        )


def test_different_module_counts_are_rejected() -> None:
    current = _current_graph()

    only_two_modules = AttributedRobotGraph(
        stamp=current.stamp,
        nodes=current.nodes[:2],
        edges=(),
        global_attributes=current.global_attributes,
    )

    with pytest.raises(
        ParallelSelfAssemblyPlannerError,
        match="number of physical modules",
    ):
        ParallelSelfAssemblyPlanner().plan(
            current_graph=only_two_modules,
            target_graph=_target_graph(),
        )


def test_non_smores_module_is_rejected() -> None:
    current = _current_graph()

    invalid_nodes = (
        _physical_node(
            "root_module",
            10.0,
            5.0,
            0.0,
            robot_family="freebot",
        ),
        *current.nodes[1:],
    )

    invalid_graph = AttributedRobotGraph(
        stamp=current.stamp,
        nodes=invalid_nodes,
        edges=(),
        global_attributes=current.global_attributes,
    )

    with pytest.raises(
        ParallelSelfAssemblyPlannerError,
        match="unsupported robot family",
    ):
        ParallelSelfAssemblyPlanner().plan(
            current_graph=invalid_graph,
            target_graph=_target_graph(),
        )


def test_explicit_target_root_overrides_graph_center() -> None:
    target = _target_graph()
    explicit_root = AttributedRobotGraph(
        stamp=target.stamp,
        nodes=tuple(
            GraphNode(
                node.module_id,
                {
                    **dict(node.attributes),
                    "is_target_root": node.module_id == "v0",
                },
            )
            for node in target.nodes
        ),
        edges=target.edges,
        global_attributes=target.global_attributes,
    )

    result = ParallelSelfAssemblyPlanner().plan(
        current_graph=_current_graph(),
        target_graph=explicit_root,
    )

    assert result.rooted_target_tree.root_id == "v0"
    assert result.assembly_plan.root_target_vertex == "v0"


def test_inconsistent_target_module_count_is_rejected() -> None:
    target = _target_graph()
    inconsistent = AttributedRobotGraph(
        stamp=target.stamp,
        nodes=target.nodes,
        edges=target.edges,
        global_attributes={
            **dict(target.global_attributes),
            "module_count": 4,
        },
    )

    with pytest.raises(
        ParallelSelfAssemblyPlannerError,
        match="module_count",
    ):
        ParallelSelfAssemblyPlanner().plan(
            current_graph=_current_graph(),
            target_graph=inconsistent,
        )


def test_inconsistent_helper_metadata_is_rejected() -> None:
    target = _target_graph()
    inconsistent = AttributedRobotGraph(
        stamp=target.stamp,
        nodes=target.nodes,
        edges=target.edges,
        global_attributes={
            **dict(target.global_attributes),
            "requires_helping_module": True,
        },
    )

    with pytest.raises(
        ParallelSelfAssemblyPlannerError,
        match="helper metadata",
    ):
        ParallelSelfAssemblyPlanner().plan(
            current_graph=_current_graph(),
            target_graph=inconsistent,
        )
