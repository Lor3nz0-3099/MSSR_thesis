"""Primitive that rolls one or more spherical modules toward targets."""
from __future__ import annotations

from typing import Any, Mapping

from mssr_expert.primitives.base_primitive import BasePrimitive, PrimitiveResult
from mssr_expert.primitives.common import (
    distance_xy,
    extract_modules,
    limited_xy_velocity,
    module_position,
    vector3,
)


class RollToPrimitive(BasePrimitive):
    """Generate planar vx/vy commands toward target points."""

    name = "roll_to"

    def step(
        self,
        observation: Mapping[str, Any],
        graph: Mapping[str, Any],
        params: Mapping[str, Any],
    ) -> PrimitiveResult:
        modules = extract_modules(observation)
        module_ids = params.get("module_ids")
        if isinstance(params.get("module_id"), str):
            module_ids = [params["module_id"]]
        if not isinstance(module_ids, list | tuple):
            module_ids = tuple(modules)
        max_speed = float(params.get("max_speed", 0.2))
        tolerance = float(params.get("tolerance", 0.05))
        targets = params.get("targets", {})
        default_target = vector3(params.get("target", observation.get("goal", [0.0, 0.0, 0.0])))

        locomotion: dict[str, Mapping[str, float]] = {}
        distances: dict[str, float] = {}
        for module_id in module_ids:
            if module_id not in modules:
                continue
            target = default_target
            if isinstance(targets, Mapping) and module_id in targets:
                target = vector3(targets[module_id])
            position = module_position(modules[module_id])
            distances[str(module_id)] = distance_xy(position, target)
            locomotion[str(module_id)] = limited_xy_velocity(
                target[0] - position[0],
                target[1] - position[1],
                max_speed,
            )

        done = bool(distances) and all(distance <= tolerance for distance in distances.values())
        return PrimitiveResult(
            locomotion=locomotion,
            success=done,
            done=done,
            metrics={"distance_to_targets": distances},
            debug={"target": list(default_target), "tolerance": tolerance},
        )
