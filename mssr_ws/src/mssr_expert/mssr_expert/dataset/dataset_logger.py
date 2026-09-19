"""JSONL dataset logger shared by all deterministic experts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mssr_expert.experts.expert_output import ExpertOutput
from mssr_expert.graph.attributed_robot_graph import AttributedRobotGraph
from mssr_expert.graph.graph_features import graph_to_features
from mssr_expert.utils.json_io import dumps_json


@dataclass
class DatasetLogger:
    """Append expert transitions to a JSONL dataset."""

    path: Path
    label_source: str = "deterministic_expert"
    executed_action_source: str = "deterministic_expert"

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def build_record(
        self,
        episode_id: str,
        timestep: int,
        observation: Mapping[str, Any],
        graph: AttributedRobotGraph,
        expert_output: ExpertOutput,
        stage_name: str,
        stage_id: int,
        task_type: str,
        difficulty: float,
        task_graph: AttributedRobotGraph | None = None,
        target_graph: AttributedRobotGraph | None = None,
        assignment: Mapping[str, str] | None = None,
        next_graph: AttributedRobotGraph | None = None,
        next_observation: Mapping[str, Any] | None = None,
        episode_done: bool | None = None,
        episode_success: bool | None = None,
    ) -> dict[str, Any]:
        """Build one transition without writing the JSONL file.

        Optional arguments keep legacy experts compatible.  Self-assembly and
        reconfiguration experts provide them so IL receives the full
        task-conditioned graph and the selected graph matching.
        """

        effective_task_graph = task_graph or graph
        assignment = assignment or {}
        terminal = bool(expert_output.done) if episode_done is None else bool(
            episode_done
        )
        terminal_success = (
            bool(expert_output.success)
            if episode_success is None
            else bool(episode_success)
        )
        record = {
            "schema_version": "mssr.expert_transition.v3",
            "episode_id": episode_id,
            "timestep": int(timestep),
            "stamp": graph.stamp,
            "stage_id": int(stage_id),
            "stage_name": stage_name,
            "task_type": task_type,
            "difficulty": float(difficulty),
            "fsm_state": expert_output.fsm_state,
            "is_first": int(timestep) == 0,
            "is_last": terminal,
            "is_terminal": terminal,
            "action_valid": not terminal,
            "reward": (
                1.0
                if terminal and terminal_success
                else 0.0
            ),
            "discount": 0.0 if terminal else 1.0,
            "observation": dict(observation),
            "observation_t_plus_1": (
                dict(next_observation)
                if next_observation is not None
                else None
            ),
            "graph_t": graph.to_dict(),
            "target_graph": (
                target_graph.to_dict()
                if target_graph is not None
                else None
            ),
            "task_graph_t": effective_task_graph.to_dict(),
            "assignment_target_to_module": dict(assignment),
            "graph_t_plus_1": (
                next_graph.to_dict()
                if next_graph is not None
                else None
            ),
            "attributed_graph": graph.to_dict(),
            "attributed_task_graph": effective_task_graph.to_dict(),
            "graph_features": graph_to_features(effective_task_graph),
            "expert_action": {
                "locomotion": {
                    module_id: dict(command)
                    for module_id, command in expert_output.locomotion.items()
                },
                "magnetic": [dict(command) for command in expert_output.magnetic],
                "primitive_goal": (
                    dict(expert_output.primitive_goal)
                    if expert_output.primitive_goal is not None
                    else None
                ),
            },
            "supervision": {
                "label_source": self.label_source,
                "executed_action_source": self.executed_action_source,
                "expert_intervention": False,
                "valid_for_behavior_cloning": not terminal,
            },
            "expert_annotation": {
                "fsm_state": expert_output.fsm_state,
                "active_primitive": expert_output.active_primitive,
                "primitive_params": dict(expert_output.primitive_params),
                "task_metrics": dict(expert_output.task_metrics),
                "debug": dict(expert_output.debug),
            },
            "active_primitive": expert_output.active_primitive,
            "primitive_params": dict(expert_output.primitive_params),
            "module_roles": dict(expert_output.module_roles),
            "attachment_modes": dict(expert_output.attachment_modes),
            "task_metrics": dict(expert_output.task_metrics),
            "success": terminal_success if terminal else expert_output.success,
            "done": terminal,
            "debug": dict(expert_output.debug),
        }
        return record

    def log_step(self, *args: Any, **kwargs: Any) -> None:
        """Build and append one transition to the configured JSONL file."""
        record = self.build_record(*args, **kwargs)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(dumps_json(record) + "\n")
