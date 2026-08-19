"""Curriculum stage selection for deterministic experts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from mssr_expert.curriculum.difficulty_scheduler import DifficultyScheduler
from mssr_expert.curriculum.stage_config import StageConfig


@dataclass
class CurriculumState:
    """Current curriculum state for logging and ROS publication."""

    stage: StageConfig
    episode_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the current state."""
        payload = self.stage.to_dict()
        payload["episode_index"] = self.episode_index
        return payload


class CurriculumManager:
    """Select stages and update deterministic difficulty."""

    def __init__(
        self,
        stages: Iterable[StageConfig],
        initial_stage_id: int = 0,
        scheduler: DifficultyScheduler | None = None,
    ) -> None:
        self._stages = {stage.stage_id: stage for stage in stages}
        if not self._stages:
            raise ValueError("CurriculumManager requires at least one stage")
        self._stage_id = initial_stage_id if initial_stage_id in self._stages else 0
        if self._stage_id not in self._stages:
            self._stage_id = min(self._stages)
        self._episode_index = 0
        self._scheduler = scheduler or DifficultyScheduler(
            initial_difficulty=self._stages[self._stage_id].difficulty
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CurriculumManager":
        """Build a manager from a dict-like config."""
        stage_payloads = payload.get("stages", ())
        stages = [
            StageConfig.from_mapping(item)
            for item in stage_payloads
            if isinstance(item, Mapping)
        ]
        scheduler_payload = payload.get("difficulty_scheduler", {})
        if not isinstance(scheduler_payload, Mapping):
            scheduler_payload = {}
        return cls(
            stages=stages,
            initial_stage_id=int(payload.get("initial_stage_id", 0)),
            scheduler=DifficultyScheduler(
                initial_difficulty=float(scheduler_payload.get("initial_difficulty", 0.0)),
                min_difficulty=float(scheduler_payload.get("min_difficulty", 0.0)),
                max_difficulty=float(scheduler_payload.get("max_difficulty", 1.0)),
                success_step=float(scheduler_payload.get("success_step", 0.05)),
                failure_step=float(scheduler_payload.get("failure_step", 0.025)),
            ),
        )

    @property
    def current_stage(self) -> StageConfig:
        """Return the current stage with live difficulty injected."""
        stage = self._stages[self._stage_id]
        return StageConfig(
            stage_id=stage.stage_id,
            stage_name=stage.stage_name,
            task_type=stage.task_type,
            expert_name=stage.expert_name,
            difficulty=self._scheduler.difficulty,
            scenario_name=stage.scenario_name,
            module_count=stage.module_count,
            required_primitives=stage.required_primitives,
            success_metrics=stage.success_metrics,
            parameters=stage.parameters,
        )

    @property
    def state(self) -> CurriculumState:
        """Return serializable current curriculum state."""
        return CurriculumState(stage=self.current_stage, episode_index=self._episode_index)

    def set_stage(self, stage_id: int) -> StageConfig:
        """Switch to a stage by id."""
        if stage_id not in self._stages:
            raise KeyError(f"Unknown stage id {stage_id}")
        self._stage_id = stage_id
        self._scheduler.reset(self._stages[stage_id].difficulty)
        return self.current_stage

    def complete_episode(self, success: bool) -> CurriculumState:
        """Advance episode count and update difficulty."""
        self._episode_index += 1
        self._scheduler.update(success)
        return self.state
