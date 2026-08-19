"""Stage configuration models for swarm-like MSSR tasks."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class StageConfig:
    """Configuration for one curriculum stage."""

    stage_id: int
    stage_name: str
    task_type: str
    expert_name: str
    difficulty: float = 0.0
    scenario_name: str = ""
    module_count: int = 0
    required_primitives: tuple[str, ...] = ()
    success_metrics: tuple[str, ...] = ()
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "StageConfig":
        """Build a stage config from a dict-like payload."""
        return cls(
            stage_id=int(payload.get("stage_id", 0)),
            stage_name=str(payload.get("stage_name", "")),
            task_type=str(payload.get("task_type", "")),
            expert_name=str(payload.get("expert_name", "")),
            difficulty=float(payload.get("difficulty", 0.0)),
            scenario_name=str(payload.get("scenario_name", "")),
            module_count=int(payload.get("module_count", 0)),
            required_primitives=tuple(
                str(item) for item in payload.get("required_primitives", ())
            ),
            success_metrics=tuple(str(item) for item in payload.get("success_metrics", ())),
            parameters=dict(payload.get("parameters", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config for ROS state messages or dataset records."""
        return {
            "stage_id": self.stage_id,
            "stage_name": self.stage_name,
            "task_type": self.task_type,
            "expert_name": self.expert_name,
            "difficulty": self.difficulty,
            "scenario_name": self.scenario_name,
            "module_count": self.module_count,
            "required_primitives": list(self.required_primitives),
            "success_metrics": list(self.success_metrics),
            "parameters": dict(self.parameters),
        }
