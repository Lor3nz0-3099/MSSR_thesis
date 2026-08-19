"""Composite primitive for climbing onto a supporting sphere."""
from __future__ import annotations

from typing import Any, Mapping

from mssr_expert.primitives.attach_as_pivot import AttachAsPivotPrimitive
from mssr_expert.primitives.base_primitive import BasePrimitive, PrimitiveResult
from mssr_expert.primitives.common import edge_is_attached, extract_modules, module_position
from mssr_expert.primitives.dock_to_surface import DockToSurfacePrimitive
from mssr_expert.primitives.rotate_around_attached import RotateAroundAttachedPrimitive


class ClimbOnPrimitive(BasePrimitive):
    """Dock, attach as surface pivot, then rotate until height gain is reached."""

    name = "climb_on"

    def __init__(self) -> None:
        self._dock = DockToSurfacePrimitive()
        self._attach = AttachAsPivotPrimitive()
        self._rotate = RotateAroundAttachedPrimitive()

    def step(
        self,
        observation: Mapping[str, Any],
        graph: Mapping[str, Any],
        params: Mapping[str, Any],
    ) -> PrimitiveResult:
        modules = extract_modules(observation)
        climber_id = str(params.get("climber_module_id", params.get("module_id", "")))
        anchor_id = str(params.get("anchor_module_id", params.get("support_module_id", "")))
        if climber_id not in modules or anchor_id not in modules:
            return PrimitiveResult(debug={"reason": "missing_climber_or_anchor"})

        climber_position = module_position(modules[climber_id])
        anchor_position = module_position(modules[anchor_id])
        anchor_radius = float(modules[anchor_id].get("radius", params.get("radius", 0.0)))
        target_height = float(
            params.get("target_height", anchor_position[2] + max(anchor_radius, 0.0))
        )
        height_margin = float(params.get("height_margin", 0.05))
        climb_success = climber_position[2] >= target_height - height_margin
        attached = edge_is_attached(graph, climber_id, anchor_id, "surface_pivot")

        if climb_success:
            return PrimitiveResult(
                locomotion={
                    climber_id: {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0},
                    anchor_id: {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0},
                },
                success=True,
                done=True,
                metrics={
                    "climb_success": True,
                    "height_reached": climber_position[2],
                    "target_height": target_height,
                },
            )

        if not attached:
            dock_result = self._dock.step(observation, graph, params)
            if not dock_result.done:
                return PrimitiveResult(
                    locomotion=dock_result.locomotion,
                    magnetic=dock_result.magnetic,
                    metrics={**dict(dock_result.metrics), "climb_phase": "dock_to_surface"},
                    debug=dock_result.debug,
                )
            attach_result = self._attach.step(observation, graph, params)
            return PrimitiveResult(
                magnetic=attach_result.magnetic,
                success=attach_result.success,
                done=attach_result.done,
                metrics={**dict(attach_result.metrics), "climb_phase": "attach_as_pivot"},
                debug=attach_result.debug,
            )

        rotate_params = dict(params)
        rotate_params["target_height"] = target_height
        rotate_params["height_margin"] = height_margin
        rotate_result = self._rotate.step(observation, graph, rotate_params)
        return PrimitiveResult(
            locomotion=rotate_result.locomotion,
            magnetic=rotate_result.magnetic,
            success=rotate_result.success,
            done=rotate_result.done,
            metrics={**dict(rotate_result.metrics), "climb_phase": "rotate_around_anchor"},
            debug=rotate_result.debug,
        )
