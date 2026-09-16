"""Validated connected anchors for incremental use of the assembly waves."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from mssr_expert.graph.attributed_robot_graph import AttributedRobotGraph
from mssr_expert.planning.smores_ep.topology import (
    SmoresKinematicTree, SmoresTopologyEdge, VALID_FACES,
)


class PartialAssemblyError(ValueError):
    """An anchor seed does not describe the observed connected structure."""


@dataclass(frozen=True)
class PartialAssemblySeed:
    """Fixed target identities and the exact complementary free module set."""

    anchored_target_to_module: Mapping[str, str]
    completed_target_edges: tuple[SmoresTopologyEdge, ...]
    free_module_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'anchored_target_to_module',
                           MappingProxyType(dict(self.anchored_target_to_module)))
        object.__setattr__(self, 'completed_target_edges', tuple(self.completed_target_edges))
        object.__setattr__(self, 'free_module_ids', tuple(self.free_module_ids))


def connection_key(
    vertex_a: str, face_a: str, vertex_b: str, face_b: str,
    clocking_quarter_turns: int = 0,
) -> tuple[str, str, str, str, int]:
    """Canonical attributed edge; reversing endpoints preserves clocking.

    This follows the existing kinematic rooting convention.
    """
    if face_a not in VALID_FACES or face_b not in VALID_FACES:
        raise PartialAssemblyError('Connection has an invalid face.')
    if (not isinstance(clocking_quarter_turns, int)
            or isinstance(clocking_quarter_turns, bool)
            or clocking_quarter_turns not in {0, 1, 2, 3}):
        raise PartialAssemblyError('Connection has invalid clocking.')
    if vertex_a == vertex_b:
        raise PartialAssemblyError('Connection cannot be a self-loop.')
    if (vertex_a, face_a) > (vertex_b, face_b):
        vertex_a, face_a, vertex_b, face_b = vertex_b, face_b, vertex_a, face_a
    return vertex_a, face_a, vertex_b, face_b, clocking_quarter_turns


def topology_edge_key(edge: SmoresTopologyEdge) -> tuple[str, str, str, str, int]:
    """Canonical key for a logical target connection."""
    return connection_key(edge.vertex_a, edge.face_a, edge.vertex_b, edge.face_b,
                          edge.clocking_quarter_turns)


def current_connection_keys(graph: AttributedRobotGraph) -> frozenset[tuple[str, str, str, str, int]]:
    """Read attached physical face topology.

    Runtime ``clocking_quarter_turns`` is intentionally normalized away here.
    The state-graph value is derived from measured face tangents: an aligned
    TOP/BOTTOM pair can report q=2, while LEFT/RIGHT wheel disks rotate during
    locomotion.  It is therefore not a stable morphology identifier.

    Logical target clocking is still retained on ``SmoresTopologyEdge`` and
    assembly actions; only observed-topology identity ignores it.
    """
    result = set()
    for edge in graph.edges:
        if edge.relation_type != 'current_connection' or not edge.attributes.get('is_attached', True):
            continue
        key = connection_key(
            edge.module_a_id,
            edge.attributes.get('connector_a_id') or edge.attributes.get('face_a'),
            edge.module_b_id,
            edge.attributes.get('connector_b_id') or edge.attributes.get('face_b'),
            0,
        )
        if key in result:
            raise PartialAssemblyError('Duplicate observed connection.')
        result.add(key)
    return frozenset(result)


def validate_partial_assembly_seed(
    seed: PartialAssemblySeed, current_graph: AttributedRobotGraph,
    target_tree: SmoresKinematicTree, root_id: str,
) -> None:
    """Require a connected root-containing seed and isolated complement."""
    mapping = seed.anchored_target_to_module
    modules = {node.module_id for node in current_graph.nodes if node.node_type == 'physical_module'}
    anchors = set(mapping.values())
    free = set(seed.free_module_ids)
    if root_id not in mapping or not set(mapping) <= set(target_tree.vertex_ids):
        raise PartialAssemblyError('Anchors must include the target root and valid target vertices.')
    if len(anchors) != len(mapping):
        raise PartialAssemblyError('Anchor mapping must be injective.')
    if len(free) != len(seed.free_module_ids) or anchors & free or anchors | free != modules:
        raise PartialAssemblyError('Anchors and free modules must partition the physical modules.')
    completed = {topology_edge_key(edge) for edge in seed.completed_target_edges}
    if len(completed) != len(seed.completed_target_edges):
        raise PartialAssemblyError('Duplicate completed target edge.')
    if not completed <= {topology_edge_key(edge) for edge in target_tree.edges}:
        raise PartialAssemblyError('Completed edges must match target faces and clocking.')
    adjacent = {vertex: set() for vertex in mapping}
    expected_observed = set()
    for edge in seed.completed_target_edges:
        if edge.vertex_a not in mapping or edge.vertex_b not in mapping:
            raise PartialAssemblyError('Completed edges must join anchored vertices.')
        adjacent[edge.vertex_a].add(edge.vertex_b)
        adjacent[edge.vertex_b].add(edge.vertex_a)
        expected_observed.add(connection_key(
            mapping[edge.vertex_a],
            edge.face_a,
            mapping[edge.vertex_b],
            edge.face_b,
            0,
        ))
    reached = set()
    pending = [root_id]
    while pending:
        vertex = pending.pop()
        if vertex not in reached:
            reached.add(vertex)
            pending.extend(adjacent[vertex] - reached)
    if reached != set(mapping):
        raise PartialAssemblyError('Anchors must form a connected root-containing structure.')
    if current_connection_keys(current_graph) != expected_observed:
        raise PartialAssemblyError('Observed connections differ from the seed; free modules must be isolated.')
