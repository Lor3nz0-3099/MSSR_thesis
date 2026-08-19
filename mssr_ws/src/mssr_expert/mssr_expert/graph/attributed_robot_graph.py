"""Robot-family-agnostic attributed multigraph for MSSR tasks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class GraphNode:
    """One physical-module or logical-target node."""

    module_id: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def node_id(self) -> str:
        """Return the generic node identifier.

        ``module_id`` is retained for compatibility with the physical graph
        schema.  Task graphs may also contain IDs such as ``target:v0``.
        """

        return self.module_id

    @property
    def node_type(self) -> str:
        """Return the semantic node type."""

        return str(self.attributes.get("node_type", "physical_module"))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the node."""
        return {
            "node_id": self.node_id,
            "module_id": self.module_id,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True)
class GraphEdge:
    """One contact, magnetic attachment, support, or target relation."""

    module_a_id: str
    module_b_id: str
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def pair_key(self) -> tuple[str, str]:
        """Return the deterministic unordered endpoint pair."""

        return tuple(sorted((self.module_a_id, self.module_b_id)))

    @property
    def relation_type(self) -> str:
        """Return the normalized semantic relation type."""

        explicit = self.attributes.get("relation_type")
        if explicit:
            return str(explicit)
        if self.attributes.get("is_assignment"):
            return "assignment"
        if self.attributes.get("is_target_edge"):
            return "target_connection"
        if self.attributes.get("is_attached"):
            return "current_connection"
        if self.attributes.get("is_contact"):
            return "contact"
        return str(self.attributes.get("edge_type", "relation"))

    @property
    def key(self) -> tuple[str, ...]:
        """Return a deterministic multigraph key.

        Relation type and connector endpoints are part of the identity, so a
        contact, a current connection and a target connection between the same
        two nodes are all retained.
        """

        face_a = str(
            self.attributes.get("connector_a_id")
            or self.attributes.get("face_a")
            or ""
        )
        face_b = str(
            self.attributes.get("connector_b_id")
            or self.attributes.get("face_b")
            or ""
        )
        endpoint_a = (self.module_a_id, face_a)
        endpoint_b = (self.module_b_id, face_b)
        first, second = sorted((endpoint_a, endpoint_b))
        relation_id = str(
            self.attributes.get("edge_id")
            or self.attributes.get("connection_id")
            or ""
        )
        return (
            self.relation_type,
            first[0],
            first[1],
            second[0],
            second[1],
            relation_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the edge."""
        payload = {
            "module_a_id": self.module_a_id,
            "module_b_id": self.module_b_id,
            "attributes": {
                **dict(self.attributes),
                "relation_type": self.relation_type,
            },
        }
        return payload


@dataclass(frozen=True)
class AttributedRobotGraph:
    """Canonical dynamic graph snapshot used by experts and learning."""

    stamp: float = 0.0
    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()
    global_attributes: Mapping[str, Any] = field(default_factory=dict)

    def node_by_id(self) -> dict[str, GraphNode]:
        """Return nodes indexed by module id."""
        return {node.module_id: node for node in self.nodes}

    def edge_by_pair(self) -> dict[tuple[str, str], tuple[GraphEdge, ...]]:
        """Return every relation grouped by unordered endpoint pair."""

        grouped: dict[tuple[str, str], list[GraphEdge]] = {}
        for edge in self.edges:
            grouped.setdefault(edge.pair_key, []).append(edge)
        return {
            pair: tuple(sorted(edges, key=lambda edge: edge.key))
            for pair, edges in grouped.items()
        }

    def edges_matching(
        self,
        relation_types: str | Iterable[str] | None = None,
        *,
        attached_only: bool = False,
        target_only: bool = False,
    ) -> tuple[GraphEdge, ...]:
        """Return edges matching semantic and state filters."""

        normalized_relations: set[str] | None
        if relation_types is None:
            normalized_relations = None
        elif isinstance(relation_types, str):
            normalized_relations = {relation_types}
        else:
            normalized_relations = {
                str(relation_type) for relation_type in relation_types
            }

        return tuple(
            edge
            for edge in self.edges
            if (
                normalized_relations is None
                or edge.relation_type in normalized_relations
            )
            and (
                not attached_only
                or bool(edge.attributes.get("is_attached"))
            )
            and (
                not target_only
                or bool(edge.attributes.get("is_target_edge"))
            )
        )

    def adjacency(
        self,
        relation_types: str | Iterable[str] | None = None,
        *,
        attached_only: bool = False,
        target_only: bool = False,
    ) -> dict[str, tuple[str, ...]]:
        """Return filtered undirected adjacency."""

        neighbors: dict[str, set[str]] = {
            node.node_id: set() for node in self.nodes
        }
        edges = self.edges_matching(
            relation_types,
            attached_only=attached_only,
            target_only=target_only,
        )
        for edge in edges:
            neighbors.setdefault(edge.module_a_id, set()).add(edge.module_b_id)
            neighbors.setdefault(edge.module_b_id, set()).add(edge.module_a_id)
        return {
            module_id: tuple(sorted(module_neighbors))
            for module_id, module_neighbors in neighbors.items()
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph."""
        return {
            "schema_version": str(
                self.global_attributes.get(
                    "schema_version",
                    "mssr.attributed_graph.v3",
                )
            ),
            "stamp": self.stamp,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "global_attributes": dict(self.global_attributes),
        }
