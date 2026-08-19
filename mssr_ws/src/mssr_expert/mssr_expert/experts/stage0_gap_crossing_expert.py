"""Stage 0 expert for gap crossing with temporary bridge roles."""
from __future__ import annotations

from typing import Any, Mapping

from mssr_expert.experts.expert_output import ExpertOutput
from mssr_expert.experts.swarm_expert_base import SwarmExpertBase
from mssr_expert.planning.bridge_planner import bridge_targets
from mssr_expert.planning.role_assignment import assign_gap_roles
from mssr_expert.primitives.common import extract_modules, module_position


class Stage0GapCrossingExpert(SwarmExpertBase):
    """Deterministic expert for forming a simple temporary bridge over a gap."""

    def step(self, observation: Mapping[str, Any], graph: Mapping[str, Any]) -> ExpertOutput:
        self._step_count += 1
        modules = extract_modules(observation)
        if not modules:
            return ExpertOutput(fsm_state="WAIT_FOR_OBSERVATION")
        roles = assign_gap_roles(observation)
        gap = observation.get("gap", {})
        if not isinstance(gap, Mapping):
            gap = {}
        gap_center_x = float(gap.get("center_x", 2.5))
        gap_width = float(gap.get("width", 1.2))
        targets = bridge_targets(
            roles,
            gap_center_x=gap_center_x,
            gap_width=gap_width,
            z=float(observation.get("module_nominal_z", self._nominal_z(modules))),
        )
        bridge_ids = [module_id for module_id, role in roles.items() if role in ("anchor", "bridge_part")]
        mobile_ids = [module_id for module_id, role in roles.items() if role in ("mobile", "recovery")]
        bridge_ready = self._targets_reached(modules, targets, tolerance=0.25)

        if not bridge_ready:
            self._fsm_state = "POSITION_BRIDGE_MODULES"
            return self._run_primitive(
                "roll_to",
                observation,
                graph,
                {
                    "module_ids": bridge_ids,
                    "targets": targets,
                    "max_speed": self.max_speed,
                    "tolerance": 0.25,
                },
                roles,
                self._fsm_state,
                {"gap_width": gap_width, "bridge_ready": False},
            )

        goal = observation.get("goal", [gap_center_x + gap_width + 1.5, 0.0, 0.0])
        self._fsm_state = "MOVE_MOBILE_MODULES_ACROSS"
        output = self._run_primitive(
            "roll_to",
            observation,
            graph,
            {
                "module_ids": mobile_ids,
                "target": goal,
                "max_speed": self.max_speed,
                "tolerance": 0.35,
            },
            roles,
            self._fsm_state,
            {"gap_width": gap_width, "bridge_ready": True},
        )
        if output.done:
            self._done = True
        return output

    def _nominal_z(self, modules: Mapping[str, Mapping[str, Any]]) -> float:
        if not modules:
            return 0.0
        return sum(module_position(module)[2] for module in modules.values()) / float(len(modules))

    def _targets_reached(
        self,
        modules: Mapping[str, Mapping[str, Any]],
        targets: Mapping[str, list[float]],
        tolerance: float,
    ) -> bool:
        if not targets:
            return False
        for module_id, target in targets.items():
            if module_id not in modules:
                return False
            position = module_position(modules[module_id])
            if abs(position[0] - target[0]) > tolerance or abs(position[1] - target[1]) > tolerance:
                return False
        return True
