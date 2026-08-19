"""Primitive that requests a surface-pivot magnetic attachment."""
from __future__ import annotations

from typing import Any, Mapping

from mssr_expert.primitives.base_primitive import BasePrimitive, PrimitiveResult
from mssr_expert.primitives.common import (
    contact_point_between,
    distance_3d,
    edge_is_attached,
    extract_modules,
    module_position,
    vector3,
)


class AttachAsPivotPrimitive(BasePrimitive):
    """Attach a climber to an anchor with surface_pivot semantics."""

    name = "attach_as_pivot"

    def step(
        self,
        observation: Mapping[str, Any],
        graph: Mapping[str, Any],
        params: Mapping[str, Any],
    ) -> PrimitiveResult:
        modules = extract_modules(observation)
        climber_id = str(params.get("climber_module_id", params.get("module_a_id", "")))
        anchor_id = str(params.get("anchor_module_id", params.get("module_b_id", "")))
        if climber_id not in modules or anchor_id not in modules:
            return PrimitiveResult(debug={"reason": "missing_climber_or_anchor"})

        climber_position = module_position(modules[climber_id])
        anchor_position = module_position(modules[anchor_id])
        anchor_radius = float(modules[anchor_id].get("radius", params.get("radius", 0.0)))
        climber_radius = float(modules[climber_id].get("radius", params.get("radius", 0.0)))
        attach_distance = float(
            params.get("attach_distance", 1.15 * (anchor_radius + climber_radius))
        )
        distance = distance_3d(climber_position, anchor_position)
        attached = edge_is_attached(graph, climber_id, anchor_id, "surface_pivot")
        close_enough = distance <= attach_distance if attach_distance > 0.0 else True
        pivot_axis = vector3(params.get("pivot_axis"), (0.0, 1.0, 0.0))
        contact_point = params.get("contact_point_world")
        if contact_point is None:
            contact_point = contact_point_between(anchor_position, climber_position, anchor_radius)

        magnetic = ()
        if close_enough and not attached:
            magnetic = (
                {
                    "module_a_id": climber_id,
                    "module_b_id": anchor_id,
                    "command": "attach",
                    "joint_type": str(params.get("joint_type", "hinge")),
                    "attachment_mode": "surface_pivot",
                    "contact_point_world": list(contact_point),
                    "pivot_axis": list(pivot_axis),
                    "allows_rotation": True,
                    "is_load_bearing": True,
                    "is_temporary": True,
                    "role": str(params.get("role", "climber_pivots_on_anchor")),
                },
            )

        return PrimitiveResult(
            magnetic=magnetic,
            success=attached,
            done=attached,
            metrics={
                "distance": distance,
                "close_enough": close_enough,
                "attached_as_surface_pivot": attached,
            },
            debug={"climber_module_id": climber_id, "anchor_module_id": anchor_id},
        )
