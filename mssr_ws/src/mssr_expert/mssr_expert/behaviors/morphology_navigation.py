"""Planar virtual-base estimation for assembled morphologies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mssr_expert.behaviors.morphology_library import (
    AssignedModule,
    MorphologyLibraryError,
)
from mssr_expert.graph.attributed_robot_graph import AttributedRobotGraph


@dataclass(frozen=True)
class PlanarMorphologyState:
    x_m: float
    y_m: float
    yaw_rad: float
    vx_m_s: float
    vy_m_s: float
    yaw_rate_rad_s: float


def estimate_planar_morphology_state(
    graph: AttributedRobotGraph,
    assignments: Sequence[AssignedModule],
    navigation_spec: Mapping[str, Sequence[str]],
) -> PlanarMorphologyState:
    """Estimate one planar virtual base from live role-anchored modules."""

    role_to_module = {item.target_role: item.module_id for item in assignments}
    if len(role_to_module) != len(assignments):
        raise MorphologyLibraryError(
            "Navigation frame requires unique target roles"
        )
    nodes = graph.node_by_id()

    def role_nodes(key: str) -> list[Any]:
        raw_roles = navigation_spec.get(key, ())
        if not raw_roles:
            raise MorphologyLibraryError(f"Navigation frame is missing {key}")
        selected = []
        for raw_role in raw_roles:
            role = str(raw_role)
            module_id = role_to_module.get(role)
            if module_id is None or module_id not in nodes:
                raise MorphologyLibraryError(
                    f"Navigation role {role!r} is not available in the live graph"
                )
            selected.append(nodes[module_id])
        return selected

    center_nodes = role_nodes("center_roles")
    rear_nodes = role_nodes("forward_from_roles")
    front_nodes = role_nodes("forward_to_roles")

    center_xyz = _mean_vector(center_nodes, "pose", "position")
    rear_xyz = _mean_vector(rear_nodes, "pose", "position")
    front_xyz = _mean_vector(front_nodes, "pose", "position")
    forward_x = front_xyz[0] - rear_xyz[0]
    forward_y = front_xyz[1] - rear_xyz[1]
    if math.hypot(forward_x, forward_y) <= 1.0e-6:
        raise MorphologyLibraryError(
            "Navigation frame forward anchors collapse in the horizontal plane"
        )
    yaw = math.atan2(forward_y, forward_x)

    world_linear = _mean_vector(center_nodes, "twist", "linear")
    world_angular = _mean_vector(center_nodes, "twist", "angular")
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    body_vx = cosine * world_linear[0] + sine * world_linear[1]
    body_vy = -sine * world_linear[0] + cosine * world_linear[1]

    return PlanarMorphologyState(
        x_m=center_xyz[0],
        y_m=center_xyz[1],
        yaw_rad=yaw,
        vx_m_s=body_vx,
        vy_m_s=body_vy,
        yaw_rate_rad_s=world_angular[2],
    )


def _mean_vector(
    nodes: Sequence[Any],
    outer_key: str,
    inner_key: str,
) -> tuple[float, float, float]:
    vectors: list[tuple[float, float, float]] = []
    for node in nodes:
        outer = node.attributes.get(outer_key, {})
        if not isinstance(outer, Mapping):
            raise MorphologyLibraryError(
                f"Live node {node.module_id!r} has no {outer_key}"
            )
        raw = outer.get(inner_key)
        if not isinstance(raw, list | tuple) or len(raw) < 3:
            raise MorphologyLibraryError(
                f"Live node {node.module_id!r} has no {outer_key}.{inner_key}"
            )
        vector = tuple(float(value) for value in raw[:3])
        if not all(math.isfinite(value) for value in vector):
            raise MorphologyLibraryError(
                f"Live node {node.module_id!r} has non-finite {outer_key}.{inner_key}"
            )
        vectors.append(vector)
    count = float(len(vectors))
    return tuple(
        sum(vector[index] for vector in vectors) / count
        for index in range(3)
    )
