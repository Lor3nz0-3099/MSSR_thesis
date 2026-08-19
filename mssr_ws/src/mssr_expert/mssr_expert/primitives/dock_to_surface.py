"""Primitive that moves a module toward another sphere's surface."""
from __future__ import annotations

from typing import Any, Mapping

from mssr_expert.primitives.base_primitive import BasePrimitive, PrimitiveResult
from mssr_expert.primitives.common import (
    contact_point_between,
    distance_3d,
    extract_modules,
    limited_xy_velocity,
    module_position,
    vector3,
)


class DockToSurfacePrimitive(BasePrimitive):
    """Approach a target sphere until surface contact is plausible."""

    name = "dock_to_surface"

    def step(
        self,
        observation: Mapping[str, Any],
        graph: Mapping[str, Any],
        params: Mapping[str, Any],
    ) -> PrimitiveResult:
        modules = extract_modules(observation)
        mobile_id = str(
            params.get(
                "mobile_module_id",
                params.get("climber_module_id", params.get("module_id", "")),
            )
        )
        target_id = str(
            params.get(
                "target_module_id",
                params.get("anchor_module_id", params.get("support_module_id", "")),
            )
        )
        if mobile_id not in modules or target_id not in modules:
            return PrimitiveResult(debug={"reason": "missing_mobile_or_target"})

        mobile = modules[mobile_id]
        target = modules[target_id]
        mobile_position = module_position(mobile)
        target_position = module_position(target)
        mobile_radius = float(mobile.get("radius", params.get("module_radius", 0.0)))
        target_radius = float(target.get("radius", params.get("target_radius", mobile_radius)))
        desired_distance = float(
            params.get("surface_distance", mobile_radius + target_radius)
        )
        tolerance = float(params.get("tolerance", 0.05))
        max_speed = float(params.get("max_speed", 0.12))

        offset = vector3(params.get("approach_offset"), (desired_distance, 0.0, 0.0))
        desired_position = (
            target_position[0] + offset[0],
            target_position[1] + offset[1],
            target_position[2] + offset[2],
        )
        center_distance = distance_3d(mobile_position, target_position)
        surface_error = abs(center_distance - desired_distance)
        done = surface_error <= tolerance
        command = {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0}
        if not done:
            command = limited_xy_velocity(
                desired_position[0] - mobile_position[0],
                desired_position[1] - mobile_position[1],
                max_speed,
            )
        return PrimitiveResult(
            locomotion={mobile_id: command},
            success=done,
            done=done,
            metrics={
                "center_distance": center_distance,
                "surface_error": surface_error,
                "contact_point_world": contact_point_between(
                    target_position,
                    mobile_position,
                    target_radius,
                ),
            },
            debug={
                "mobile_module_id": mobile_id,
                "target_module_id": target_id,
                "desired_position": list(desired_position),
            },
        )
