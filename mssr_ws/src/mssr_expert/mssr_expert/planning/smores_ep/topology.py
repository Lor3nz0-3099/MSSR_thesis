"""Kinematic-tree representation used by the deterministic SMORES-EP experts."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


VALID_FACES = frozenset({"LEFT", "RIGHT", "TOP", "BOTTOM"})


class TopologyValidationError(ValueError):
    """Raised when a SMORES-EP target topology is not valid."""


@dataclass(frozen=True)
class SmoresTopologyEdge:
    """One rigid connection between two SMORES-EP modules."""

    vertex_a: str
    face_a: str
    vertex_b: str
    face_b: str
    clocking_quarter_turns: int = 0


@dataclass(frozen=True)
class SmoresKinematicTree:
    """A SMORES-EP configuration represented as an attributed tree."""

    vertex_ids: tuple[str, ...]
    edges: tuple[SmoresTopologyEdge, ...]


def validate_kinematic_tree(tree: SmoresKinematicTree) -> None:
    """Check that a SMORES-EP topology is a valid kinematic tree."""

    if not tree.vertex_ids:
        raise TopologyValidationError(
            "The topology must contain at least one vertex."
        )

    if len(tree.vertex_ids) != len(set(tree.vertex_ids)):
        raise TopologyValidationError(
            "The topology contains duplicated vertex IDs."
        )

    for vertex_id in tree.vertex_ids:
        if not isinstance(vertex_id, str) or not vertex_id.strip():
            raise TopologyValidationError(
                "Every vertex must have a non-empty string ID."
            )

    vertex_set = set(tree.vertex_ids)
    used_connectors: set[tuple[str, str]] = set()

    adjacency: dict[str, set[str]] = {
        vertex_id: set() for vertex_id in tree.vertex_ids
    }

    for edge in tree.edges:
        if edge.vertex_a not in vertex_set:
            raise TopologyValidationError(
                f"Unknown vertex in edge: {edge.vertex_a!r}."
            )

        if edge.vertex_b not in vertex_set:
            raise TopologyValidationError(
                f"Unknown vertex in edge: {edge.vertex_b!r}."
            )

        if edge.vertex_a == edge.vertex_b:
            raise TopologyValidationError(
                f"Self-loop detected on vertex {edge.vertex_a!r}."
            )

        if edge.face_a not in VALID_FACES:
            raise TopologyValidationError(
                f"Invalid face {edge.face_a!r} on vertex {edge.vertex_a!r}."
            )

        if edge.face_b not in VALID_FACES:
            raise TopologyValidationError(
                f"Invalid face {edge.face_b!r} on vertex {edge.vertex_b!r}."
            )

        if (
            not isinstance(edge.clocking_quarter_turns, int)
            or isinstance(edge.clocking_quarter_turns, bool)
            or edge.clocking_quarter_turns not in {0, 1, 2, 3}
        ):
            raise TopologyValidationError(
                "clocking_quarter_turns must be one of 0, 1, 2 or 3."
            )

        connector_a = (edge.vertex_a, edge.face_a)
        connector_b = (edge.vertex_b, edge.face_b)

        if connector_a in used_connectors:
            raise TopologyValidationError(
                f"Connector {edge.vertex_a}.{edge.face_a} is used more than once."
            )

        if connector_b in used_connectors:
            raise TopologyValidationError(
                f"Connector {edge.vertex_b}.{edge.face_b} is used more than once."
            )

        used_connectors.add(connector_a)
        used_connectors.add(connector_b)

        adjacency[edge.vertex_a].add(edge.vertex_b)
        adjacency[edge.vertex_b].add(edge.vertex_a)

    expected_edge_count = len(tree.vertex_ids) - 1

    if len(tree.edges) != expected_edge_count:
        raise TopologyValidationError(
            "A tree with "
            f"{len(tree.vertex_ids)} vertices must contain "
            f"{expected_edge_count} edges, but {len(tree.edges)} were provided."
        )

    start_vertex = tree.vertex_ids[0]
    visited: set[str] = set()
    queue: deque[str] = deque([start_vertex])

    while queue:
        current = queue.popleft()

        if current in visited:
            continue

        visited.add(current)

        for neighbor in adjacency[current]:
            if neighbor not in visited:
                queue.append(neighbor)

    if visited != vertex_set:
        missing_vertices = sorted(vertex_set - visited)
        raise TopologyValidationError(
            f"The topology is disconnected. Unreachable vertices: {missing_vertices}."
        )


def graph_centers(tree: SmoresKinematicTree) -> tuple[str, ...]:
    """Return the one or two central vertices of the tree."""

    validate_kinematic_tree(tree)

    adjacency = _build_adjacency(tree)
    eccentricities: dict[str, int] = {}

    for vertex_id in tree.vertex_ids:
        distances = _distances_from(vertex_id, adjacency)
        eccentricities[vertex_id] = max(distances.values())

    minimum_eccentricity = min(eccentricities.values())

    centers = [
        vertex_id
        for vertex_id, eccentricity in eccentricities.items()
        if eccentricity == minimum_eccentricity
    ]

    return tuple(sorted(centers))


def choose_target_root(tree: SmoresKinematicTree) -> str:
    """Choose a deterministic root for the target configuration."""

    centers = graph_centers(tree)

    # graph_centers() returns the centers in lexicographical order.
    # If there are two centers, the first ID resolves the ambiguity.
    return centers[0]


def _build_adjacency(
    tree: SmoresKinematicTree,
) -> dict[str, tuple[str, ...]]:
    """Build a deterministic adjacency table."""

    adjacency: dict[str, list[str]] = {
        vertex_id: [] for vertex_id in tree.vertex_ids
    }

    for edge in tree.edges:
        adjacency[edge.vertex_a].append(edge.vertex_b)
        adjacency[edge.vertex_b].append(edge.vertex_a)

    return {
        vertex_id: tuple(sorted(neighbors))
        for vertex_id, neighbors in adjacency.items()
    }


def _distances_from(
    start_vertex: str,
    adjacency: dict[str, tuple[str, ...]],
) -> dict[str, int]:
    """Calculate the number of edges from one vertex to every other vertex."""

    distances = {start_vertex: 0}
    queue: deque[str] = deque([start_vertex])

    while queue:
        current = queue.popleft()

        for neighbor in adjacency[current]:
            if neighbor in distances:
                continue

            distances[neighbor] = distances[current] + 1
            queue.append(neighbor)

    return distances