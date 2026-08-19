"""Deterministic planar obstacle avoidance for self-assembly staging."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable


Point2 = tuple[float, float]


@dataclass(frozen=True)
class CircularObstacle:
    """A module footprint inflated by the required centre clearance."""

    module_id: str
    center_xy: Point2
    clearance_m: float


def _distance(first: Point2, second: Point2) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _segment_distance(point: Point2, start: Point2, end: Point2) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 1.0e-16:
        return _distance(point, start)
    fraction = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / length_sq
    fraction = min(1.0, max(0.0, fraction))
    closest = (start[0] + fraction * dx, start[1] + fraction * dy)
    return _distance(point, closest)


def segment_is_clear(
    start: Point2,
    end: Point2,
    obstacles: Iterable[CircularObstacle],
    *,
    allow_start_inside: bool = False,
    epsilon_m: float = 1.0e-6,
) -> bool:
    """Return whether a segment respects every inflated footprint.

    PhysX can leave a module marginally inside an inflated footprint.  The
    first route segment may escape that footprint, but only if its distance
    from the obstacle increases monotonically from the starting point.
    """

    for obstacle in obstacles:
        radius = obstacle.clearance_m
        start_distance = _distance(start, obstacle.center_xy)
        if start_distance < radius - epsilon_m:
            if not allow_start_inside:
                return False
            direction = (end[0] - start[0], end[1] - start[1])
            radial = (
                start[0] - obstacle.center_xy[0],
                start[1] - obstacle.center_xy[1],
            )
            if radial[0] * direction[0] + radial[1] * direction[1] < 0.0:
                return False
            continue
        if (
            _segment_distance(obstacle.center_xy, start, end)
            < radius - epsilon_m
        ):
            return False
    return True


def plan_collision_aware_path(
    start: Point2,
    goal: Point2,
    obstacles: Iterable[CircularObstacle],
    *,
    waypoint_margin_m: float = 0.015,
    angular_samples: int = 16,
) -> tuple[Point2, ...] | None:
    """Plan the shortest deterministic visibility-graph route to ``goal``.

    The returned sequence excludes ``start`` and includes ``goal``.  ``None``
    means that the current planar obstacle model admits no safe route.
    """

    obstacle_tuple = tuple(obstacles)
    if waypoint_margin_m <= 0.0 or not math.isfinite(waypoint_margin_m):
        raise ValueError("Waypoint margin must be finite and positive")
    if angular_samples < 8:
        raise ValueError("At least eight angular samples are required")
    if any(
        obstacle.clearance_m <= 0.0
        or not math.isfinite(obstacle.clearance_m)
        for obstacle in obstacle_tuple
    ):
        raise ValueError("Obstacle clearances must be finite and positive")

    if segment_is_clear(
        start,
        goal,
        obstacle_tuple,
        allow_start_inside=True,
    ):
        return (goal,)

    # Start and goal remain the first two nodes so equal-cost paths have a
    # stable tie break. Ring samples are also generated in a stable order.
    nodes: list[Point2] = [start, goal]
    for obstacle in sorted(obstacle_tuple, key=lambda item: item.module_id):
        radius = obstacle.clearance_m + waypoint_margin_m
        for sample_index in range(angular_samples):
            angle = 2.0 * math.pi * sample_index / angular_samples
            candidate = (
                obstacle.center_xy[0] + radius * math.cos(angle),
                obstacle.center_xy[1] + radius * math.sin(angle),
            )
            if all(
                _distance(candidate, other.center_xy)
                >= other.clearance_m - 1.0e-6
                for other in obstacle_tuple
            ):
                nodes.append(candidate)

    # A staging goal inside an unrelated inflated footprint is intentionally
    # rejected: the straight contact phase is the only phase allowed to enter
    # a module footprint.
    if any(
        _distance(goal, obstacle.center_xy)
        < obstacle.clearance_m - 1.0e-6
        for obstacle in obstacle_tuple
    ):
        return None

    adjacency: list[list[tuple[int, float]]] = [[] for _ in nodes]
    for source_index, source in enumerate(nodes):
        for destination_index, destination in enumerate(nodes):
            if source_index == destination_index:
                continue
            if segment_is_clear(
                source,
                destination,
                obstacle_tuple,
                allow_start_inside=(source_index == 0),
            ):
                adjacency[source_index].append(
                    (destination_index, _distance(source, destination))
                )

    distances = [math.inf] * len(nodes)
    predecessors: list[int | None] = [None] * len(nodes)
    distances[0] = 0.0
    queue: list[tuple[float, int]] = [(0.0, 0)]
    while queue:
        distance, node_index = heapq.heappop(queue)
        if distance > distances[node_index] + 1.0e-12:
            continue
        if node_index == 1:
            break
        for neighbour_index, edge_length in adjacency[node_index]:
            candidate_distance = distance + edge_length
            if candidate_distance < distances[neighbour_index] - 1.0e-12:
                distances[neighbour_index] = candidate_distance
                predecessors[neighbour_index] = node_index
                heapq.heappush(queue, (candidate_distance, neighbour_index))

    if not math.isfinite(distances[1]):
        return None
    indices: list[int] = []
    node_index: int | None = 1
    while node_index is not None and node_index != 0:
        indices.append(node_index)
        node_index = predecessors[node_index]
    if node_index != 0:
        return None
    indices.reverse()
    return tuple(nodes[index] for index in indices)
