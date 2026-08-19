"""Stage 2 expert for multi-step stair climbing with support frontier."""
from __future__ import annotations

from typing import Any, Mapping

from mssr_expert.experts.expert_output import ExpertOutput
from mssr_expert.experts.swarm_expert_base import SwarmExpertBase
from mssr_expert.planning.role_assignment import assign_stair_roles
from mssr_expert.planning.stair_planner import stair_support_targets
from mssr_expert.primitives.common import extract_modules, module_position


class Stage2StairClimbExpert(SwarmExpertBase):
    """Repeat support positioning and climb-on across stair levels."""

    def __init__(self, seed: int | None = None, max_speed: float = 0.2) -> None:
        super().__init__(seed=seed, max_speed=max_speed)
        self._current_step = 0

    def reset(self, scenario: Mapping[str, Any] | None = None) -> None:
        """Reset stage state."""
        super().reset(scenario)
        self._current_step = 0

    def step(self, observation: Mapping[str, Any], graph: Mapping[str, Any]) -> ExpertOutput:
        self._step_count += 1
        modules = extract_modules(observation)
        if not modules:
            return ExpertOutput(fsm_state="WAIT_FOR_OBSERVATION")
        roles = assign_stair_roles(observation)
        stair = observation.get("stair", {})
        if not isinstance(stair, Mapping):
            stair = {}
        step_count = int(stair.get("step_count", 3))
        step_height = float(stair.get("step_height", 0.30))
        step_depth = float(stair.get("step_depth", 1.0))
        first_step_x = float(stair.get("first_step_x", 2.5))
        targets = stair_support_targets(
            roles,
            first_step_x=first_step_x,
            step_depth=step_depth,
            current_step=self._current_step,
            z=self._nominal_z(modules),
        )
        support_ready = self._support_ready(modules, targets)
        climber_id = self._first_role(roles, "climber")
        anchor_id = self._first_role(roles, "frontier_anchor")

        if self._current_step >= step_count:
            self._done = True
            return ExpertOutput(
                fsm_state="SUCCESS",
                module_roles=roles,
                task_metrics={"highest_step_reached": self._current_step},
                success=True,
                done=True,
            )

        if not support_ready:
            self._fsm_state = "MOVE_SUPPORT_FRONTIER"
            return self._run_primitive(
                "roll_to",
                observation,
                graph,
                {
                    "module_ids": list(targets),
                    "targets": targets,
                    "max_speed": self.max_speed,
                    "tolerance": 0.3,
                },
                roles,
                self._fsm_state,
                {"current_step": self._current_step, "support_ready": False},
            )

        if climber_id is None or anchor_id is None:
            return ExpertOutput(
                fsm_state="FAILURE",
                module_roles=roles,
                task_metrics={"reason": "missing_climber_or_frontier_anchor"},
                done=True,
            )

        target_height = (self._current_step + 1) * step_height + 0.5
        self._fsm_state = "CLIMB_NEXT_STEP"
        output = self._run_primitive(
            "climb_on",
            observation,
            graph,
            {
                "climber_module_id": climber_id,
                "anchor_module_id": anchor_id,
                "target_height": target_height,
                "height_margin": 0.08,
                "max_speed": self.max_speed * 0.75,
                "joint_type": "hinge",
                "pivot_axis": [0.0, 1.0, 0.0],
            },
            roles,
            self._fsm_state,
            {"current_step": self._current_step, "target_height": target_height},
        )
        if output.done:
            self._current_step += 1
        return output

    def _support_ready(
        self,
        modules: Mapping[str, Mapping[str, Any]],
        targets: Mapping[str, list[float]],
    ) -> bool:
        if not targets:
            return False
        for module_id, target in targets.items():
            if module_id not in modules:
                return False
            position = module_position(modules[module_id])
            if abs(position[0] - target[0]) > 0.3 or abs(position[1] - target[1]) > 0.3:
                return False
        return True

    def _nominal_z(self, modules: Mapping[str, Mapping[str, Any]]) -> float:
        if not modules:
            return 0.0
        return sum(module_position(module)[2] for module in modules.values()) / float(len(modules))

    def _first_role(self, roles: Mapping[str, str], role: str) -> str | None:
        for module_id, module_role in roles.items():
            if module_role == role:
                return module_id
        return None
