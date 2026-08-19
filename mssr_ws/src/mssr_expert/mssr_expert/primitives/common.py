"""Shared helpers for primitive implementations."""
from __future__ import annotations

import math
from typing import Any, Mapping


Vector3 = tuple[float, float, float]


def extract_modules(observation: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return modules indexed by id from supported observation formats."""
    raw_modules = observation.get("modules", {})
    modules: dict[str, Mapping[str, Any]] = {}
    if isinstance(raw_modules, Mapping):
        for module_id, payload in raw_modules.items():
            if isinstance(module_id, str) and isinstance(payload, Mapping):
                modules[module_id] = payload
    elif isinstance(raw_modules, list):
        for item in raw_modules:
            if not isinstance(item, Mapping):
                continue
            module_id = item.get("module_id")
            if isinstance(module_id, str):
                modules[module_id] = item
    return modules


def module_position(module: Mapping[str, Any]) -> Vector3:
    """Read a module position from flat or pose-based payloads."""
    pose = module.get("pose", {})
    position = pose.get("position") if isinstance(pose, Mapping) else None
    if position is None:
        position = module.get("position")
    return vector3(position)


def vector3(value: Any, default: Vector3 = (0.0, 0.0, 0.0)) -> Vector3:
    """Convert a JSON vector to a 3D tuple."""
    if not isinstance(value, list | tuple) or len(value) < 3:
        return default
    return (float(value[0]), float(value[1]), float(value[2]))


def distance_xy(a: Vector3, b: Vector3) -> float:
    """Planar distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def distance_3d(a: Vector3, b: Vector3) -> float:
    """Euclidean distance between two points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def limited_xy_velocity(dx: float, dy: float, max_speed: float) -> dict[str, float]:
    """Return a bounded planar velocity command."""
    norm = math.hypot(dx, dy)
    if norm <= 1e-9 or max_speed <= 0.0:
        return {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0}
    scale = min(max_speed, norm) / norm
    return {"vx": dx * scale, "vy": dy * scale, "yaw_rate": 0.0}


def edge_is_attached(
    graph: Mapping[str, Any],
    module_a_id: str,
    module_b_id: str,
    attachment_mode: str | None = None,
) -> bool:
    """Return whether a graph has an attached edge between two modules."""
    wanted = frozenset((module_a_id, module_b_id))
    for edge in graph.get("edges", []):
        if not isinstance(edge, Mapping):
            continue
        attrs = edge.get("attributes", {})
        if not isinstance(attrs, Mapping):
            attrs = {}
        source = edge.get("module_a_id") or edge.get("source") or attrs.get("module_a_id")
        target = edge.get("module_b_id") or edge.get("target") or attrs.get("module_b_id")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        if frozenset((source, target)) != wanted:
            continue
        attached = bool(
            edge.get("is_attached")
            or edge.get("is_magnet_enabled")
            or attrs.get("is_attached")
            or attrs.get("is_magnet_enabled")
            or edge.get("status") == "connected"
            or attrs.get("status") == "connected"
        )
        mode_matches = attachment_mode is None or (
            edge.get("attachment_mode") == attachment_mode
            or attrs.get("attachment_mode") == attachment_mode
        )
        if attached and mode_matches:
            return True
    return False


def contact_point_between(a: Vector3, b: Vector3, radius_a: float) -> list[float]:
    """Estimate a world contact point on module A toward module B."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm <= 1e-9:
        return [a[0] + radius_a, a[1], a[2]]
    return [
        a[0] + radius_a * dx / norm,
        a[1] + radius_a * dy / norm,
        a[2] + radius_a * dz / norm,
    ]
