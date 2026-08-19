"""Simple deterministic bridge planning for gap crossing."""
from __future__ import annotations

from typing import Mapping


def bridge_targets(
    module_roles: Mapping[str, str],
    gap_center_x: float,
    gap_width: float,
    y: float = 0.0,
    z: float = 0.0,
    spacing: float = 0.8,
) -> dict[str, list[float]]:
    """Return target positions for anchors and bridge modules."""
    bridge_ids = [module_id for module_id, role in module_roles.items() if role == "bridge_part"]
    anchor_ids = [module_id for module_id, role in module_roles.items() if role == "anchor"]
    targets: dict[str, list[float]] = {}
    left_edge = gap_center_x - gap_width * 0.5
    right_edge = gap_center_x + gap_width * 0.5
    if anchor_ids:
        targets[anchor_ids[0]] = [left_edge - spacing * 0.5, y, z]
    if len(anchor_ids) > 1:
        targets[anchor_ids[1]] = [right_edge + spacing * 0.5, y, z]
    count = max(1, len(bridge_ids))
    for index, module_id in enumerate(bridge_ids):
        alpha = (index + 1) / (count + 1)
        x = left_edge * (1.0 - alpha) + right_edge * alpha
        targets[module_id] = [x, y, z]
    return targets
