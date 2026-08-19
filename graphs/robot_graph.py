"""Graph representation of modular robot configurations.

Legacy status: this builder derives edges from ``SurfaceAttachmentState`` and
backs the spherical FreeBOT prototype (``scripts/main.py``, ``robots/``,
``worlds/task_evaluator.py``). SMORES-EP and future robot-family-agnostic
code use the canonical ``mssr_expert.graph.AttributedRobotGraph`` instead. See
``context/LEGACY_SPHERE_MIGRATION.md`` before deleting or refactoring this
module: it must be migrated only after the FreeBOT adapter reaches parity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from robots.joints import JointType
from robots.module_state import AttachmentStatus, ModuleState, SurfaceAttachmentState, Vector3
from robots.state_registry import RobotStateSnapshot


@dataclass(frozen=True)
class RobotGraphNode:
    """Graph node representing one robot module."""

    module_id: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RobotGraphEdge:
    """Graph edge representing a contact or attachment between two modules."""

    source: str
    target: str
    status: AttachmentStatus
    joint_type: JointType | None = None
    is_magnet_enabled: bool = False
    relative_position: Vector3 | None = None
    local_contact_point: Vector3 | None = None
    local_contact_normal: Vector3 | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> frozenset[str]:
        """Return an unordered edge key."""
        return frozenset((self.source, self.target))


@dataclass(frozen=True)
class RobotGraph:
    """Graph snapshot of a modular robot configuration."""

    timestamp: float
    nodes: tuple[RobotGraphNode, ...]
    edges: tuple[RobotGraphEdge, ...]

    def node_by_id(self) -> dict[str, RobotGraphNode]:
        """Return nodes indexed by module id."""
        return {node.module_id: node for node in self.nodes}

    def edge_by_pair(self) -> dict[frozenset[str], RobotGraphEdge]:
        """Return edges indexed by unordered module pair."""
        return {edge.key: edge for edge in self.edges}

    def adjacency(self) -> dict[str, tuple[str, ...]]:
        """Return an undirected adjacency list."""
        neighbors: dict[str, list[str]] = {node.module_id: [] for node in self.nodes}
        for edge in self.edges:
            neighbors.setdefault(edge.source, []).append(edge.target)
            neighbors.setdefault(edge.target, []).append(edge.source)
        return {
            module_id: tuple(sorted(module_neighbors))
            for module_id, module_neighbors in neighbors.items()
        }


class RobotGraphBuilder:
    """Build graph snapshots from module state snapshots."""

    def build(self, snapshot: RobotStateSnapshot) -> RobotGraph:
        """Convert a robot state snapshot into a graph snapshot."""
        nodes = tuple(self._build_node(module) for module in snapshot.modules)
        edges = self._build_edges(snapshot.modules)
        return RobotGraph(timestamp=snapshot.timestamp, nodes=nodes, edges=edges)

    def _build_node(self, module: ModuleState) -> RobotGraphNode:
        """Build a graph node from one module state."""
        return RobotGraphNode(
            module_id=module.module_id,
            attributes=module.graph_attributes(),
        )

    def _build_edges(self, modules: tuple[ModuleState, ...]) -> tuple[RobotGraphEdge, ...]:
        """Build one undirected edge for each module pair with an attachment state."""
        edges_by_pair: dict[frozenset[str], RobotGraphEdge] = {}
        for module in modules:
            for attachment in module.surface_attachments:
                if attachment.connected_module_id is None:
                    continue

                pair = frozenset((module.module_id, attachment.connected_module_id))
                if pair in edges_by_pair:
                    continue

                edges_by_pair[pair] = self._build_edge(module, attachment)

        return tuple(
            edges_by_pair[pair]
            for pair in sorted(edges_by_pair, key=lambda item: tuple(sorted(item)))
        )

    def _build_edge(
        self,
        module: ModuleState,
        attachment: SurfaceAttachmentState,
    ) -> RobotGraphEdge:
        """Build one graph edge from a module-side attachment state."""
        assert attachment.connected_module_id is not None
        source, target = sorted((module.module_id, attachment.connected_module_id))
        return RobotGraphEdge(
            source=source,
            target=target,
            status=attachment.status,
            joint_type=attachment.joint_type,
            is_magnet_enabled=attachment.is_magnet_enabled,
            relative_position=attachment.relative_position,
            local_contact_point=attachment.local_contact_point,
            local_contact_normal=attachment.local_contact_normal,
            attributes={
                "attachment_id": attachment.attachment_id,
                "status": attachment.status.value,
                "joint_type": attachment.joint_type.value if attachment.joint_type is not None else None,
                "attachment_mode": attachment.attachment_mode,
                "is_magnet_enabled": attachment.is_magnet_enabled,
                "relative_position": attachment.relative_position,
                "local_contact_point": attachment.local_contact_point,
                "local_contact_normal": attachment.local_contact_normal,
                "pivot_axis": attachment.pivot_axis,
                "allows_rotation": attachment.allows_rotation,
                "is_load_bearing": attachment.is_load_bearing,
                "is_temporary": attachment.is_temporary,
                "edge_role": attachment.edge_role,
                "is_support_edge": attachment.is_load_bearing
                or attachment.attachment_mode in ("support_contact", "surface_pivot", "bridge_link"),
            },
        )


def build_robot_graph(snapshot: RobotStateSnapshot) -> RobotGraph:
    """Convenience function for building a robot graph from a snapshot."""
    return RobotGraphBuilder().build(snapshot)
