"""Support-frontier planning helpers for stair climbing."""
from __future__ import annotations

from typing import Mapping


def stair_support_targets(
    module_roles: Mapping[str, str],
    first_step_x: float,
    step_depth: float,
    current_step: int,
    y: float = 0.0,
    z: float = 0.0,
    spacing: float = 0.8,
) -> dict[str, list[float]]:
    """Return support targets around the current stair frontier."""
    frontier_x = first_step_x + current_step * step_depth
    targets: dict[str, list[float]] = {}
    for module_id, role in module_roles.items():
        if role == "frontier_anchor":
            targets[module_id] = [frontier_x - spacing, y, z]
        elif role == "base":
            targets[module_id] = [frontier_x - spacing * 0.5, y, z]
        elif role == "support_transfer":
            targets[module_id] = [frontier_x, y, z]
    return targets
