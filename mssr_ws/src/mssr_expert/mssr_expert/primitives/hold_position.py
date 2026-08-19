"""Primitive that keeps selected modules stationary."""
from __future__ import annotations

from typing import Any, Mapping

from mssr_expert.primitives.base_primitive import BasePrimitive, PrimitiveResult
from mssr_expert.primitives.common import extract_modules


class HoldPositionPrimitive(BasePrimitive):
    """Emit zero locomotion commands for support, anchor, or base modules."""

    name = "hold_position"

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
            role = params.get("role")
            module_ids = [
                module_id
                for module_id, payload in modules.items()
                if role is None or payload.get("role") == role
            ]
        locomotion = {
            str(module_id): {"vx": 0.0, "vy": 0.0, "yaw_rate": 0.0}
            for module_id in module_ids
            if module_id in modules
        }
        return PrimitiveResult(
            locomotion=locomotion,
            success=bool(locomotion),
            done=True,
            metrics={"held_module_count": len(locomotion)},
        )
