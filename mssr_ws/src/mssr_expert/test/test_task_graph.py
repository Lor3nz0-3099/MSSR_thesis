from __future__ import annotations

import json

from mssr_expert.dataset.dataset_logger import DatasetLogger
from mssr_expert.experts.expert_output import ExpertOutput
from mssr_expert.graph.attributed_robot_graph import (
    AttributedRobotGraph,
    GraphEdge,
    GraphNode,
)
from mssr_expert.graph.graph_builder import GraphBuilder
from mssr_expert.graph.graph_features import graph_to_features
from mssr_expert.graph.task_graph import TaskGraphBuilder
from mssr_expert.planning.smores_ep.attributed_adapter import (
    target_graph_to_kinematic_tree,
    target_roles_from_graph,
)


def _target_graph() -> AttributedRobotGraph:
    return AttributedRobotGraph(
        nodes=(
            GraphNode(
                "v0",
                {
                    "target_vertex_id": "v0",
                    "target_role": "support",
                    "functional_role": {"name": "support"},
                    "is_target_root": True,
                },
            ),
            GraphNode(
                "v1",
                {
                    "target_vertex_id": "v1",
                    "target_role": "elbow",
                    "functional_role": {
                        "name": "joint",
                        "effective_dof_count": 1,
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
                    "clocking_quarter_turns": 1,
                },
            ),
        ),
        global_attributes={
            "schema_version": "mssr.target_graph.v1",
            "graph_kind": "target_morphology",
            "morphology_name": "test_shape",
            "capabilities": ["test_motion"],
        },
    )


def _current_graph() -> AttributedRobotGraph:
    return AttributedRobotGraph(
        stamp=3.0,
        nodes=(
            GraphNode(
                "m0",
                {
                    "node_type": "physical_module",
                    "robot_family": "smores_ep",
                    "position": [0.0, 0.0, 0.03],
                },
            ),
            GraphNode(
                "m1",
                {
                    "node_type": "physical_module",
                    "robot_family": "smores_ep",
                    "position": [0.1, 0.0, 0.03],
                },
            ),
        ),
        global_attributes={"task_type": "parallel_self_assembly"},
    )


def test_graph_builder_preserves_multiple_relation_types_for_same_pair() -> None:
    observation = {
        "modules": {
            "m0": {"position": [0.0, 0.0, 0.0]},
            "m1": {"position": [0.1, 0.0, 0.0]},
        },
        "contacts": [
            {
                "module_a_id": "m0",
                "module_b_id": "m1",
                "face_a": "TOP",
                "face_b": "BOTTOM",
            }
        ],
        "attachments": [
            {
                "module_a_id": "m0",
                "module_b_id": "m1",
                "face_a": "TOP",
                "face_b": "BOTTOM",
                "status": "connected",
            }
        ],
    }
    target_payload = {
        "edges": [
            {
                "module_a_id": "m0",
                "module_b_id": "m1",
                "face_a": "LEFT",
                "face_b": "RIGHT",
                "relation_type": "target_connection",
                "is_target_edge": True,
            }
        ]
    }

    graph = GraphBuilder().build(observation, target_payload)

    assert {
        edge.relation_type for edge in graph.edges
    } == {"contact", "current_connection", "target_connection"}
    assert len(graph.edge_by_pair()[("m0", "m1")]) == 3
    assert graph.adjacency("contact") == {
        "m0": ("m1",),
        "m1": ("m0",),
    }
    assert len(graph.edges_matching("target_connection")) == 1
    assert len(graph.edges_matching(attached_only=True)) == 1


def test_task_graph_contains_physical_target_and_assignment_relations() -> None:
    task_graph = TaskGraphBuilder().build(
        current_graph=_current_graph(),
        target_graph=_target_graph(),
        assignment={"v0": "m0", "v1": "m1"},
        execution_state={"phase": "ASSIGNED"},
    )

    node_types = {
        node.node_id: node.node_type for node in task_graph.nodes
    }
    assert node_types == {
        "m0": "physical_module",
        "m1": "physical_module",
        "target:v0": "target_slot",
        "target:v1": "target_slot",
    }
    assert len(task_graph.edges_matching("target_connection")) == 1
    assert len(task_graph.edges_matching("assignment")) == 2
    assert task_graph.global_attributes["schema_version"] == (
        "mssr.task_graph.v1"
    )
    assert task_graph.global_attributes["execution_state"] == {
        "phase": "ASSIGNED"
    }
    assert task_graph.global_attributes["target_morphology_name"] == (
        "test_shape"
    )
    assert task_graph.global_attributes["target_capabilities"] == [
        "test_motion"
    ]
    nodes = task_graph.node_by_id()
    assert nodes["m0"].attributes["target_vertex_id"] == "v0"
    assert nodes["m0"].attributes["target_role"] == "support"
    assert nodes["m1"].attributes["target_role"] == "elbow"
    assert nodes["m1"].attributes["target_functional_role"][
        "effective_dof_count"
    ] == 1


def test_attributed_target_projection_preserves_faces_clocking_and_roles() -> None:
    task_graph = TaskGraphBuilder().build(
        current_graph=_current_graph(),
        target_graph=_target_graph(),
        assignment={"v0": "m0", "v1": "m1"},
    )

    tree = target_graph_to_kinematic_tree(task_graph)
    roles = target_roles_from_graph(task_graph)

    assert tree.vertex_ids == ("v0", "v1")
    assert len(tree.edges) == 1
    assert tree.edges[0].face_a == "TOP"
    assert tree.edges[0].face_b == "BOTTOM"
    assert tree.edges[0].clocking_quarter_turns == 1
    assert roles["v0"]["target_role"] == "support"
    assert roles["v1"]["functional_role"]["effective_dof_count"] == 1


def test_task_graph_features_include_every_relation() -> None:
    task_graph = TaskGraphBuilder().build(
        current_graph=_current_graph(),
        target_graph=_target_graph(),
        assignment={"v0": "m0", "v1": "m1"},
    )

    features = graph_to_features(task_graph)

    assert len(features["node_ids"]) == 4
    assert len(features["edge_features"]) == 6
    assert "target:v0" in features["node_ids"]


def test_dataset_record_keeps_current_target_assignment_and_next_graph(
    tmp_path,
) -> None:
    current_graph = _current_graph()
    target_graph = _target_graph()
    assignment = {"v0": "m0", "v1": "m1"}
    task_graph = TaskGraphBuilder().build(
        current_graph=current_graph,
        target_graph=target_graph,
        assignment=assignment,
    )
    next_graph = AttributedRobotGraph(
        stamp=4.0,
        nodes=current_graph.nodes,
        edges=(
            GraphEdge(
                "m0",
                "m1",
                {
                    "relation_type": "current_connection",
                    "is_attached": True,
                    "face_a": "TOP",
                    "face_b": "BOTTOM",
                },
            ),
        ),
    )
    path = tmp_path / "dataset.jsonl"

    DatasetLogger(path).log_step(
        episode_id="episode_1",
        timestep=0,
        observation={"modules": {}},
        graph=current_graph,
        expert_output=ExpertOutput(fsm_state="ASSIGNED"),
        stage_name="self_assembly",
        stage_id=0,
        task_type="parallel_self_assembly",
        difficulty=0.0,
        task_graph=task_graph,
        target_graph=target_graph,
        assignment=assignment,
        next_graph=next_graph,
        next_observation={"modules": {"m0": {"x": 0.1}}},
    )

    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["schema_version"] == "mssr.expert_transition.v3"
    assert record["graph_t"]["stamp"] == 3.0
    assert record["target_graph"]["global_attributes"]["graph_kind"] == (
        "target_morphology"
    )
    assert record["assignment_target_to_module"] == assignment
    assert record["task_graph_t"]["global_attributes"]["graph_kind"] == (
        "task_conditioned"
    )
    assert record["graph_t_plus_1"]["stamp"] == 4.0
    assert record["observation_t_plus_1"]["modules"]["m0"]["x"] == 0.1
    assert record["is_first"] is True
    assert record["is_last"] is False
    assert record["action_valid"] is True
    assert record["supervision"]["label_source"] == "deterministic_expert"
