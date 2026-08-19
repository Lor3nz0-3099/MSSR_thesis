"""Deterministic role assignment for swarm-like expert stages."""
from __future__ import annotations

from typing import Any, Mapping

from mssr_expert.primitives.common import distance_xy, extract_modules, module_position
from mssr_expert.utils.deterministic import sorted_ids


def module_ids_from_observation(observation: Mapping[str, Any]) -> tuple[str, ...]:
    """Return deterministic module ids from an observation."""
    return sorted_ids(extract_modules(observation).keys())


def assign_gap_roles(observation: Mapping[str, Any]) -> dict[str, str]:
    """Assign anchor, bridge, mobile, and recovery roles for gap crossing."""
    module_ids = module_ids_from_observation(observation)
    roles: dict[str, str] = {}
    for index, module_id in enumerate(module_ids):
        if index < 2:
            roles[module_id] = "anchor"
        elif index < max(4, len(module_ids) - 1):
            roles[module_id] = "bridge_part"
        elif index == len(module_ids) - 1:
            roles[module_id] = "recovery"
        else:
            roles[module_id] = "mobile"
    return roles


def assign_obstacle_roles(observation: Mapping[str, Any]) -> dict[str, str]:
    """Assign base, anchor, climber, stabilizer, and mobile roles."""
    module_ids = module_ids_from_observation(observation)
    roles: dict[str, str] = {}
    for index, module_id in enumerate(module_ids):
        if index == 0:
            roles[module_id] = "anchor"
        elif index == 1:
            roles[module_id] = "climber"
        elif index in (2, 3):
            roles[module_id] = "base"
        elif index == 4:
            roles[module_id] = "stabilizer"
        else:
            roles[module_id] = "mobile"
    return roles


def assign_obstacle_roles_spatial(
    observation: Mapping[str, Any],
    obstacle_x: float,
    obstacle_y: float = 0.0,
    support_offset: float = 0.8,
) -> dict[str, str]:
    """Assign obstacle roles from module positions instead of module ids."""
    modules = extract_modules(observation)
    if not modules:
        return {}

    module_ids = sorted_ids(modules)
    positions = {module_id: module_position(modules[module_id]) for module_id in module_ids}
    approach_point = (obstacle_x - support_offset, obstacle_y, 0.0)
    anchor_id = min(
        module_ids,
        key=lambda module_id: (
            distance_xy(positions[module_id], approach_point),
            module_id,
        ),
    )

    remaining = [module_id for module_id in module_ids if module_id != anchor_id]
    anchor_position = positions[anchor_id]
    docking_point = (
        anchor_position[0] + 2.0 * float(modules[anchor_id].get("radius", 0.6)),
        anchor_position[1],
        anchor_position[2],
    )
    climber_id = min(
        remaining,
        key=lambda module_id: (
            distance_xy(positions[module_id], docking_point),
            module_id,
        ),
    ) if remaining else None

    remaining = [module_id for module_id in remaining if module_id != climber_id]
    support_point = (obstacle_x - 0.35 * support_offset, obstacle_y, 0.0)
    base_ids = tuple(
        sorted(
            remaining,
            key=lambda module_id: (
                distance_xy(positions[module_id], support_point),
                module_id,
            ),
        )[:2]
    )
    remaining = [module_id for module_id in remaining if module_id not in base_ids]
    stabilizer_id = min(
        remaining,
        key=lambda module_id: (
            distance_xy(positions[module_id], approach_point),
            module_id,
        ),
    ) if remaining else None

    roles = {module_id: "mobile" for module_id in module_ids}
    roles[anchor_id] = "anchor"
    if climber_id is not None:
        roles[climber_id] = "climber"
    for module_id in base_ids:
        roles[module_id] = "base"
    if stabilizer_id is not None:
        roles[stabilizer_id] = "stabilizer"
    return roles


def assign_stair_roles(observation: Mapping[str, Any]) -> dict[str, str]:
    """Assign roles for repeated support-frontier stair climbing."""
    module_ids = module_ids_from_observation(observation)
    roles: dict[str, str] = {}
    for index, module_id in enumerate(module_ids):
        if index == 0:
            roles[module_id] = "frontier_anchor"
        elif index in (1, 2):
            roles[module_id] = "base"
        elif index == 3:
            roles[module_id] = "climber"
        elif index in (4, 5):
            roles[module_id] = "support_transfer"
        else:
            roles[module_id] = "mobile"
    return roles
