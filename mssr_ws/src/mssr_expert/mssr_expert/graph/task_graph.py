"""Compose physical state, target morphology and assignment into one graph."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from mssr_expert.graph.attributed_robot_graph import (
    AttributedRobotGraph,
    GraphEdge,
    GraphNode,
)


class TaskGraphError(ValueError):
    """Raised when a task-conditioned graph cannot be constructed."""


@dataclass(frozen=True)
class TaskGraphBuilder:
    """Build the heterogeneous graph consumed by experts and learning."""

    target_prefix: str = "target:"

    def build(
        self,
        current_graph: AttributedRobotGraph,
        target_graph: AttributedRobotGraph,
        assignment: Mapping[str, str] | None = None,
        execution_state: Mapping[str, Any] | None = None,
    ) -> AttributedRobotGraph:
        """Combine current modules, target slots and selected assignments.

        ``assignment`` follows the planner convention
        ``target_vertex_id -> physical_module_id``.
        """

        assignment = assignment or {}
        execution_state = execution_state or {}
        target_nodes, target_id_map = self._target_nodes(target_graph)
        current_nodes = self._physical_nodes(
            current_graph,
            target_nodes,
            target_id_map,
            assignment,
        )
        current_ids = {node.node_id for node in current_nodes}

        target_edges = self._target_edges(target_graph, target_id_map)
        assignment_edges = self._assignment_edges(
            assignment,
            current_ids,
            target_id_map,
        )
        current_edges = tuple(
            GraphEdge(
                edge.module_a_id,
                edge.module_b_id,
                {
                    **dict(edge.attributes),
                    "relation_type": edge.relation_type,
                },
            )
            for edge in current_graph.edges
        )

        all_edges: dict[tuple[str, ...], GraphEdge] = {}
        for edge in (*current_edges, *target_edges, *assignment_edges):
            all_edges[edge.key] = edge

        global_attributes = {
            **dict(current_graph.global_attributes),
            "schema_version": "mssr.task_graph.v1",
            "graph_kind": "task_conditioned",
            "target_graph_present": True,
            "target_vertex_count": len(target_nodes),
            "assignment_count": len(assignment_edges),
            "target_morphology_name": str(
                target_graph.global_attributes.get(
                    "morphology_name", ""
                )
            ),
            "target_capabilities": list(
                target_graph.global_attributes.get(
                    "capabilities", ()
                )
            ),
            "execution_state": dict(execution_state),
        }
        return AttributedRobotGraph(
            stamp=current_graph.stamp,
            nodes=tuple((*current_nodes, *target_nodes)),
            edges=tuple(all_edges[key] for key in sorted(all_edges)),
            global_attributes=global_attributes,
        )

    def _physical_nodes(
        self,
        graph: AttributedRobotGraph,
        target_nodes: tuple[GraphNode, ...],
        target_id_map: Mapping[str, str],
        assignment: Mapping[str, str],
    ) -> tuple[GraphNode, ...]:
        target_by_node_id = {
            node.node_id: node for node in target_nodes
        }
        target_by_module: dict[str, GraphNode] = {}
        for target_vertex_id, module_id in assignment.items():
            target_node_id = target_id_map.get(target_vertex_id)
            if target_node_id is None:
                target_node_id = target_id_map.get(
                    self._strip_target_prefix(target_vertex_id)
                )
            target_node = target_by_node_id.get(str(target_node_id))
            if target_node is not None:
                target_by_module[module_id] = target_node

        physical_nodes: list[GraphNode] = []
        for node in graph.nodes:
            attributes = {
                **dict(node.attributes),
                "node_type": "physical_module",
                "physical_module_id": node.module_id,
            }
            target_node = target_by_module.get(node.module_id)
            if target_node is not None:
                attributes.update(
                    {
                        "target_vertex_id": target_node.attributes[
                            "target_vertex_id"
                        ],
                        "target_role": target_node.attributes.get(
                            "target_role",
                            "unassigned",
                        ),
                        "target_functional_role": dict(
                            target_node.attributes.get(
                                "functional_role",
                                {},
                            )
                        ),
                        "assignment_status": "selected",
                    }
                )
            physical_nodes.append(
                GraphNode(node.module_id, attributes)
            )
        return tuple(physical_nodes)

    def _target_nodes(
        self,
        graph: AttributedRobotGraph,
    ) -> tuple[tuple[GraphNode, ...], dict[str, str]]:
        nodes: list[GraphNode] = []
        target_id_map: dict[str, str] = {}
        for node in graph.nodes:
            target_vertex_id = str(
                node.attributes.get(
                    "target_vertex_id",
                    self._strip_target_prefix(node.node_id),
                )
            )
            task_node_id = self._target_node_id(target_vertex_id)
            if target_vertex_id in target_id_map:
                raise TaskGraphError(
                    f"Duplicate target vertex {target_vertex_id!r}."
                )
            target_id_map[target_vertex_id] = task_node_id
            target_id_map[node.node_id] = task_node_id
            target_role = str(
                node.attributes.get(
                    "target_role",
                    node.attributes.get("role", "unassigned"),
                )
            )
            nodes.append(
                GraphNode(
                    task_node_id,
                    {
                        **dict(node.attributes),
                        "node_type": "target_slot",
                        "target_vertex_id": target_vertex_id,
                        "target_role": target_role,
                        "role": target_role,
                        "is_target_node": True,
                    },
                )
            )
        return tuple(sorted(nodes, key=lambda node: node.node_id)), target_id_map

    def _target_edges(
        self,
        graph: AttributedRobotGraph,
        target_id_map: Mapping[str, str],
    ) -> tuple[GraphEdge, ...]:
        edges: list[GraphEdge] = []
        for edge in graph.edges:
            endpoint_a = target_id_map.get(edge.module_a_id)
            endpoint_b = target_id_map.get(edge.module_b_id)
            if endpoint_a is None or endpoint_b is None:
                raise TaskGraphError(
                    "A target edge references an unknown target vertex: "
                    f"{edge.module_a_id!r}, {edge.module_b_id!r}."
                )
            edges.append(
                GraphEdge(
                    endpoint_a,
                    endpoint_b,
                    {
                        **dict(edge.attributes),
                        "relation_type": "target_connection",
                        "edge_type": "target_connection",
                        "is_target_edge": True,
                        "is_attached": False,
                        "is_contact": False,
                        "is_assignment": False,
                    },
                )
            )
        return tuple(edges)

    def _assignment_edges(
        self,
        assignment: Mapping[str, str],
        physical_ids: set[str],
        target_id_map: Mapping[str, str],
    ) -> tuple[GraphEdge, ...]:
        edges: list[GraphEdge] = []
        used_modules: set[str] = set()
        used_targets: set[str] = set()
        for target_vertex_id, module_id in sorted(assignment.items()):
            target_node_id = target_id_map.get(target_vertex_id)
            if target_node_id is None:
                target_node_id = target_id_map.get(
                    self._strip_target_prefix(target_vertex_id)
                )
            if target_node_id is None:
                raise TaskGraphError(
                    f"Assignment references unknown target {target_vertex_id!r}."
                )
            if module_id not in physical_ids:
                raise TaskGraphError(
                    f"Assignment references unknown module {module_id!r}."
                )
            if module_id in used_modules or target_node_id in used_targets:
                raise TaskGraphError(
                    "Assignments must be one-to-one."
                )
            used_modules.add(module_id)
            used_targets.add(target_node_id)
            edges.append(
                GraphEdge(
                    module_id,
                    target_node_id,
                    {
                        "relation_type": "assignment",
                        "edge_type": "assignment",
                        "edge_id": f"assignment:{module_id}:{target_node_id}",
                        "is_assignment": True,
                        "is_target_edge": False,
                        "is_attached": False,
                        "is_contact": False,
                        "assignment_status": "selected",
                        "directed": True,
                    },
                )
            )
        return tuple(edges)

    def _target_node_id(self, target_vertex_id: str) -> str:
        stripped = self._strip_target_prefix(target_vertex_id)
        return f"{self.target_prefix}{stripped}"

    def _strip_target_prefix(self, node_id: str) -> str:
        if node_id.startswith(self.target_prefix):
            return node_id[len(self.target_prefix):]
        return node_id
