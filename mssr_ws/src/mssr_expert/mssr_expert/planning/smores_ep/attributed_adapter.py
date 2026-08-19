"""Adapters between attributed task graphs and SMORES planning structures."""
from __future__ import annotations

from typing import Any, Mapping

from mssr_expert.graph.attributed_robot_graph import AttributedRobotGraph, GraphNode
from mssr_expert.planning.smores_ep.topology import (
    SmoresKinematicTree,
    SmoresTopologyEdge,
    TopologyValidationError,
    validate_kinematic_tree,
)


def target_graph_to_kinematic_tree(
    graph: AttributedRobotGraph,
) -> SmoresKinematicTree:
    """Project target slots and target connections into a strict tree.

    The attributed graph remains the canonical representation.  This compact
    tree is only the deterministic planner's internal view.
    """

    selected_nodes = _target_nodes(graph)
    node_id_to_vertex = {
        node.node_id: _target_vertex_id(node)
        for node in selected_nodes
    }
    vertex_ids = tuple(sorted(node_id_to_vertex.values()))
    if len(vertex_ids) != len(set(vertex_ids)):
        raise TopologyValidationError(
            "Target graph contains duplicated target vertex IDs."
        )

    target_edges = graph.edges_matching("target_connection")
    if not target_edges:
        target_edges = tuple(
            edge
            for edge in graph.edges
            if bool(edge.attributes.get("is_target_edge"))
        )

    edges: list[SmoresTopologyEdge] = []
    for edge in target_edges:
        vertex_a = node_id_to_vertex.get(edge.module_a_id)
        vertex_b = node_id_to_vertex.get(edge.module_b_id)
        if vertex_a is None or vertex_b is None:
            raise TopologyValidationError(
                "Target connection references a node that is not a target "
                f"slot: {edge.module_a_id!r}, {edge.module_b_id!r}."
            )
        face_a = edge.attributes.get("connector_a_id") or edge.attributes.get(
            "face_a"
        )
        face_b = edge.attributes.get("connector_b_id") or edge.attributes.get(
            "face_b"
        )
        clocking = edge.attributes.get("clocking_quarter_turns", 0)
        if clocking is None:
            clocking = 0
        edges.append(
            SmoresTopologyEdge(
                vertex_a=vertex_a,
                face_a=str(face_a or ""),
                vertex_b=vertex_b,
                face_b=str(face_b or ""),
                clocking_quarter_turns=clocking,
            )
        )

    tree = SmoresKinematicTree(
        vertex_ids=vertex_ids,
        edges=tuple(
            sorted(
                edges,
                key=lambda edge: (
                    edge.vertex_a,
                    edge.face_a,
                    edge.vertex_b,
                    edge.face_b,
                    edge.clocking_quarter_turns,
                ),
            )
        ),
    )
    validate_kinematic_tree(tree)
    return tree


def target_roles_from_graph(
    graph: AttributedRobotGraph,
) -> dict[str, Mapping[str, Any]]:
    """Return role attributes indexed by logical target vertex."""

    roles: dict[str, Mapping[str, Any]] = {}
    for node in _target_nodes(graph):
        vertex_id = _target_vertex_id(node)
        functional_role = node.attributes.get("functional_role", {})
        if not isinstance(functional_role, Mapping):
            functional_role = {}
        roles[vertex_id] = {
            "target_role": str(
                node.attributes.get(
                    "target_role",
                    node.attributes.get("role", "unassigned"),
                )
            ),
            "functional_role": dict(functional_role),
            "is_target_root": bool(
                node.attributes.get("is_target_root", False)
            ),
        }
    return roles


def _target_nodes(
    graph: AttributedRobotGraph,
) -> tuple[GraphNode, ...]:
    explicit = tuple(
        node
        for node in graph.nodes
        if node.node_type == "target_slot"
        or bool(node.attributes.get("is_target_node"))
    )
    if explicit:
        return explicit
    if str(graph.global_attributes.get("graph_kind", "")) in {
        "target",
        "target_morphology",
    }:
        return graph.nodes
    if all(node.node_type != "physical_module" for node in graph.nodes):
        return graph.nodes
    raise TopologyValidationError(
        "The attributed graph does not contain target-slot nodes."
    )


def _target_vertex_id(node: GraphNode) -> str:
    value = node.attributes.get("target_vertex_id")
    if value is not None:
        return str(value)
    if node.node_id.startswith("target:"):
        return node.node_id[len("target:"):]
    return node.node_id
