"""Primitive that drives a module tangentially around an attached anchor."""
from __future__ import annotations

import math
from typing import Any, Mapping

from mssr_expert.primitives.base_primitive import BasePrimitive, PrimitiveResult
from mssr_expert.primitives.common import (
    edge_is_attached,
    extract_modules,
    module_position,
)


class RotateAroundAttachedPrimitive(BasePrimitive):
    """Approximate pivot rotation with planar tangential locomotion commands."""

    name = "rotate_around_attached"

    def step(
        self,
        observation: Mapping[str, Any],
        graph: Mapping[str, Any],
        params: Mapping[str, Any],
    ) -> PrimitiveResult:
        modules = extract_modules(observation)
        climber_id = str(params.get("climber_module_id", params.get("module_id", "")))
        anchor_id = str(params.get("anchor_module_id", ""))
        if climber_id not in modules or anchor_id not in modules:
            return PrimitiveResult(debug={"reason": "missing_climber_or_anchor"})

        climber_position = module_position(modules[climber_id])
        anchor_position = module_position(modules[anchor_id])
        attached = edge_is_attached(graph, climber_id, anchor_id, "surface_pivot")
        if not attached:
            return PrimitiveResult(
                metrics={"attached_as_surface_pivot": False},
                debug={"reason": "surface_pivot_edge_missing"},
            )

        dx = climber_position[0] - anchor_position[0]
        dy = climber_position[1] - anchor_position[1]
        radius = max(math.hypot(dx, dy), 1e-6)
        direction = 1.0 if float(params.get("direction", 1.0)) >= 0.0 else -1.0
        tangential_speed = float(params.get("tangential_speed", 0.08))
        target_height = params.get("target_height")
        height_margin = float(params.get("height_margin", 0.0))
        done = False
        if target_height is not None:
            done = climber_position[2] >= float(target_height) - height_margin

        command = {
            "vx": -direction * dy / radius * tangential_speed,
            "vy": direction * dx / radius * tangential_speed,
            "yaw_rate": direction * float(params.get("yaw_rate", 0.2)),
        }
        if done:
            command = {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0}

        return PrimitiveResult(
            locomotion={climber_id: command, anchor_id: {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0}},
            success=done,
            done=done,
            metrics={
                "attached_as_surface_pivot": True,
                "current_height": climber_position[2],
                "target_height": target_height,
                "pivot_radius_xy": radius,
            },
            debug={"climber_module_id": climber_id, "anchor_module_id": anchor_id},
        )
