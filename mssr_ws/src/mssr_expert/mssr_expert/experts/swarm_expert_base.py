"""Shared utilities for stage experts that compose primitives."""
from __future__ import annotations

from typing import Any, Mapping

from mssr_expert.experts.base_expert import BaseExpert
from mssr_expert.experts.expert_output import ExpertOutput
from mssr_expert.primitives.primitive_registry import registry as primitive_registry


class SwarmExpertBase(BaseExpert):
    """Base class with primitive execution helpers."""

    def __init__(self, seed: int | None = None, max_speed: float = 0.2) -> None:
        super().__init__(seed=seed)
        self.max_speed = max_speed
        self._fsm_state = "INIT"
        self._step_count = 0

    def reset(self, scenario: Mapping[str, Any] | None = None) -> None:
        """Reset stage state."""
        super().reset(scenario)
        self._fsm_state = "INIT"
        self._step_count = 0

    def _run_primitive(
        self,
        primitive_name: str,
        observation: Mapping[str, Any],
        graph: Mapping[str, Any],
        params: Mapping[str, Any],
        module_roles: Mapping[str, str],
        fsm_state: str,
        task_metrics: Mapping[str, Any] | None = None,
    ) -> ExpertOutput:
        primitive = primitive_registry.create(primitive_name)
        if primitive is None:
            return ExpertOutput(
                fsm_state=fsm_state,
                active_primitive=primitive_name,
                module_roles=module_roles,
                task_metrics=task_metrics or {},
                debug={"reason": "primitive_not_registered"},
            )
        result = primitive.step(observation, graph, params)
        return ExpertOutput(
            locomotion=result.locomotion,
            magnetic=result.magnetic,
            fsm_state=fsm_state,
            active_primitive=primitive_name,
            primitive_params=dict(params),
            module_roles=module_roles,
            attachment_modes=self._attachment_modes(result.magnetic),
            task_metrics={**dict(task_metrics or {}), **dict(result.metrics)},
            success=result.success,
            done=result.done,
            debug=result.debug,
        )

    def _attachment_modes(
        self,
        magnetic: tuple[Mapping[str, Any], ...],
    ) -> dict[str, str]:
        modes: dict[str, str] = {}
        for command in magnetic:
            module_a_id = command.get("module_a_id")
            module_b_id = command.get("module_b_id")
            mode = command.get("attachment_mode")
            if isinstance(module_a_id, str) and isinstance(module_b_id, str) and isinstance(mode, str):
                modes[f"{module_a_id}:{module_b_id}"] = mode
        return modes
