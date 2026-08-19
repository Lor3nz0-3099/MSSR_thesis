"""Support planning helpers for obstacle traversal."""
from __future__ import annotations

from typing import Mapping


def obstacle_support_targets(
    module_roles: Mapping[str, str],
    obstacle_x: float,
    obstacle_height: float,
    y: float = 0.0,
    z: float = 0.0,
    spacing: float = 0.8,
) -> dict[str, list[float]]:
    """Return targets for base and anchor modules near a low obstacle."""
    targets: dict[str, list[float]] = {}
    base_index = 0
    for module_id, role in module_roles.items():
        if role == "anchor":
            targets[module_id] = [obstacle_x - spacing, y, z]
        elif role == "base":
            lateral = (base_index - 0.5) * spacing
            targets[module_id] = [obstacle_x - spacing * 0.35, y + lateral, z]
            base_index += 1
        elif role == "stabilizer":
            targets[module_id] = [obstacle_x - spacing * 0.8, y + spacing, z]
    return targets
