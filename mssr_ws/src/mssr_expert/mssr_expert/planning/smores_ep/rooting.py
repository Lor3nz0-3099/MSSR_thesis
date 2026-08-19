"""Root selection and tree rooting for deterministic SMORES-EP planning."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Mapping

from mssr_expert.planning.smores_ep.topology import (
    SmoresKinematicTree,
    SmoresTopologyEdge,
    TopologyValidationError,
    choose_target_root,
    validate_kinematic_tree,
)


class PhysicalRootSelectionError(ValueError):
    """Raised when a physical root module cannot be selected."""


@dataclass(frozen=True)
class ModulePosition:
    """Position of one physical SMORES-EP module."""

    module_id: str
    x_m: float
    y_m: float
    z_m: float = 0.0


@dataclass(frozen=True)
class RootedSmoresEdge:
    """A topology edge oriented from parent to child."""

    parent_vertex: str
    parent_face: str
    child_vertex: str
    child_face: str
    clocking_quarter_turns: int


@dataclass(frozen=True)
class RootedSmoresTree:
    """A SMORES-EP topology oriented from a selected root."""

    root_id: str
    vertex_ids: tuple[str, ...]
    edges: tuple[RootedSmoresEdge, ...]
    parent_by_vertex: Mapping[str, str | None]
    depth_by_vertex: Mapping[str, int]


def root_kinematic_tree(
    tree: SmoresKinematicTree,
    root_id: str | None = None,
) -> RootedSmoresTree:
    """Orient every edge of a kinematic tree away from its root."""

    validate_kinematic_tree(tree)

    if root_id is None:
        root_id = choose_target_root(tree)

    if root_id not in tree.vertex_ids:
        raise TopologyValidationError(
            f"Cannot root the tree at unknown vertex {root_id!r}."
        )

    incident_edges: dict[str, list[SmoresTopologyEdge]] = {
        vertex_id: [] for vertex_id in tree.vertex_ids
    }

    for edge in tree.edges:
        incident_edges[edge.vertex_a].append(edge)
        incident_edges[edge.vertex_b].append(edge)

    parent_by_vertex: dict[str, str | None] = {
        root_id: None,
    }
    depth_by_vertex: dict[str, int] = {
        root_id: 0,
    }

    rooted_edges: list[RootedSmoresEdge] = []
    queue: deque[str] = deque([root_id])

    while queue:
        current_vertex = queue.popleft()
        candidates: list[RootedSmoresEdge] = []

        for edge in incident_edges[current_vertex]:
            candidate = _orient_edge_from_parent(
                edge=edge,
                parent_vertex=current_vertex,
            )
            candidates.append(candidate)

        candidates.sort(
            key=lambda edge: (
                edge.child_vertex,
                edge.parent_face,
                edge.child_face,
                edge.clocking_quarter_turns,
            )
        )

        for candidate in candidates:
            child_vertex = candidate.child_vertex

            if child_vertex in parent_by_vertex:
                continue

            parent_by_vertex[child_vertex] = current_vertex
            depth_by_vertex[child_vertex] = (
                depth_by_vertex[current_vertex] + 1
            )

            rooted_edges.append(candidate)
            queue.append(child_vertex)

    if len(parent_by_vertex) != len(tree.vertex_ids):
        raise TopologyValidationError(
            "The complete topology could not be reached from the selected root."
        )

    return RootedSmoresTree(
        root_id=root_id,
        vertex_ids=tuple(sorted(tree.vertex_ids)),
        edges=tuple(rooted_edges),
        parent_by_vertex=dict(parent_by_vertex),
        depth_by_vertex=dict(depth_by_vertex),
    )


def vertices_by_depth(
    tree: RootedSmoresTree,
) -> tuple[tuple[str, ...], ...]:
    """Group the vertices into deterministic breadth-first levels."""

    if not tree.depth_by_vertex:
        return ()

    maximum_depth = max(tree.depth_by_vertex.values())
    levels: list[tuple[str, ...]] = []

    for depth in range(maximum_depth + 1):
        vertices = sorted(
            vertex_id
            for vertex_id, vertex_depth in tree.depth_by_vertex.items()
            if vertex_depth == depth
        )
        levels.append(tuple(vertices))

    return tuple(levels)


def swarm_centroid(
    modules: tuple[ModulePosition, ...],
) -> tuple[float, float, float]:
    """Calculate the centroid of the initial physical module positions."""

    _validate_module_positions(modules)

    module_count = float(len(modules))

    centroid_x = sum(module.x_m for module in modules) / module_count
    centroid_y = sum(module.y_m for module in modules) / module_count
    centroid_z = sum(module.z_m for module in modules) / module_count

    return centroid_x, centroid_y, centroid_z


def choose_physical_root(
    modules: tuple[ModulePosition, ...],
) -> str:
    """Choose the module closest to the initial swarm centroid."""

    centroid_x, centroid_y, _ = swarm_centroid(modules)

    ranked_modules: list[tuple[float, str]] = []

    for module in modules:
        delta_x = module.x_m - centroid_x
        delta_y = module.y_m - centroid_y

        # Self-assembly starts on the ground, so root selection uses
        # distance in the horizontal XY plane.
        distance_squared = delta_x * delta_x + delta_y * delta_y

        ranked_modules.append(
            (distance_squared, module.module_id)
        )

    ranked_modules.sort()

    return ranked_modules[0][1]


def _orient_edge_from_parent(
    edge: SmoresTopologyEdge,
    parent_vertex: str,
) -> RootedSmoresEdge:
    """Orient an undirected topology edge from the given parent."""

    if edge.vertex_a == parent_vertex:
        return RootedSmoresEdge(
            parent_vertex=edge.vertex_a,
            parent_face=edge.face_a,
            child_vertex=edge.vertex_b,
            child_face=edge.face_b,
            clocking_quarter_turns=edge.clocking_quarter_turns,
        )

    if edge.vertex_b == parent_vertex:
        return RootedSmoresEdge(
            parent_vertex=edge.vertex_b,
            parent_face=edge.face_b,
            child_vertex=edge.vertex_a,
            child_face=edge.face_a,
            clocking_quarter_turns=edge.clocking_quarter_turns,
        )

    raise TopologyValidationError(
        f"Vertex {parent_vertex!r} does not belong to the supplied edge."
    )


def _validate_module_positions(
    modules: tuple[ModulePosition, ...],
) -> None:
    """Validate the physical modules used for root selection."""

    if not modules:
        raise PhysicalRootSelectionError(
            "At least one physical module is required."
        )

    module_ids = [module.module_id for module in modules]

    if len(module_ids) != len(set(module_ids)):
        raise PhysicalRootSelectionError(
            "Physical module IDs must be unique."
        )

    for module in modules:
        if not isinstance(module.module_id, str) or not module.module_id.strip():
            raise PhysicalRootSelectionError(
                "Every physical module must have a non-empty ID."
            )

        coordinates = (
            module.x_m,
            module.y_m,
            module.z_m,
        )

        if not all(
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in coordinates
        ):
            raise PhysicalRootSelectionError(
                f"Module {module.module_id!r} has an invalid position."
            )