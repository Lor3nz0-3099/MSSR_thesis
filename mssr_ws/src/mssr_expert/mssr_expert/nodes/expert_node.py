"""ROS 2 node that runs deterministic swarm experts."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from mssr_expert.dataset.dataset_logger import DatasetLogger
from mssr_expert.experts.expert_registry import registry as expert_registry
from mssr_expert.graph.graph_builder import GraphBuilder
from mssr_expert.utils.json_io import dict_to_string_msg, string_msg_to_dict


class ExpertNode(Node):
    """Adapt ROS 2 JSON topics to pure deterministic expert logic."""

    def __init__(self) -> None:
        super().__init__("mssr_expert_node")
        self._declare_parameters()

        expert_name = str(self.get_parameter("expert_name").value)
        expert_cls = expert_registry.get(expert_name)
        if expert_cls is None:
            raise ValueError(f"Unknown expert '{expert_name}'")

        self._stage_id = int(self.get_parameter("stage_id").value)
        self._stage_name = str(self.get_parameter("stage_name").value)
        self._task_type = str(self.get_parameter("task_type").value)
        self._difficulty = float(self.get_parameter("difficulty").value)
        self._expert = expert_cls(
            seed=int(self.get_parameter("seed").value),
            max_speed=float(self.get_parameter("max_speed").value),
        )
        self._graph_builder = GraphBuilder()
        self._dataset_logger = DatasetLogger(
            Path(str(self.get_parameter("dataset_path").value))
        )
        self._episode_id = str(self.get_parameter("episode_id").value)
        self._timestep = 0

        self._latest_observation: dict[str, Any] = {}
        self._latest_graph_payload: dict[str, Any] = {}

        self._combined_action_pub = self.create_publisher(
            String,
            str(self.get_parameter("combined_action_topic").value),
            10,
        )
        self._locomotion_pub = self.create_publisher(
            String,
            str(self.get_parameter("locomotion_command_topic").value),
            10,
        )
        self._magnetic_pub = self.create_publisher(
            String,
            str(self.get_parameter("magnetic_command_topic").value),
            10,
        )
        self._expert_state_pub = self.create_publisher(
            String,
            str(self.get_parameter("expert_state_topic").value),
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("observation_topic").value),
            self._on_observation,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("module_states_topic").value),
            self._on_module_states,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("graph_topic").value),
            self._on_graph,
            10,
        )

        period = 1.0 / max(1e-6, float(self.get_parameter("control_rate_hz").value))
        self._timer = self.create_timer(period, self._step)
        self.get_logger().info(
            f"Running expert '{expert_name}' for stage {self._stage_id}: {self._task_type}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("expert_name", "stage0_gap_crossing")
        self.declare_parameter("stage_id", 0)
        self.declare_parameter("stage_name", "gap_crossing")
        self.declare_parameter("task_type", "gap_crossing_temporary_bridge")
        self.declare_parameter("difficulty", 0.1)
        self.declare_parameter("seed", 100)
        self.declare_parameter("max_speed", 0.2)
        self.declare_parameter("episode_id", "episode_0001")
        self.declare_parameter("observation_topic", "/mssr/state_graph")
        self.declare_parameter("module_states_topic", "/mssr/module_states")
        self.declare_parameter("graph_topic", "/mssr/robot_graph")
        self.declare_parameter("locomotion_command_topic", "/mssr/commands/locomotion")
        self.declare_parameter("magnetic_command_topic", "/mssr/commands/magnetic")
        self.declare_parameter("combined_action_topic", "/mssr/actions")
        self.declare_parameter("expert_state_topic", "/mssr/expert/state")
        self.declare_parameter("dataset_path", "logs/datasets/swarm_expert_curriculum.jsonl")
        self.declare_parameter("dataset_log_period", 5)
        self.declare_parameter("control_rate_hz", 20.0)

    def _on_observation(self, message: String) -> None:
        payload = string_msg_to_dict(message)
        state = payload.get("state")
        graph = payload.get("graph")
        if isinstance(state, Mapping):
            self._latest_observation = dict(state)
        elif payload:
            self._latest_observation = payload
        if isinstance(graph, Mapping):
            self._latest_graph_payload = dict(graph)

    def _on_module_states(self, message: String) -> None:
        payload = string_msg_to_dict(message)
        if payload:
            self._latest_observation = payload

    def _on_graph(self, message: String) -> None:
        payload = string_msg_to_dict(message)
        if payload:
            self._latest_graph_payload = payload

    def _step(self) -> None:
        if not self._latest_observation and not self._latest_graph_payload:
            return

        observation = dict(self._latest_observation)
        observation.setdefault("stage_id", self._stage_id)
        observation.setdefault("stage_name", self._stage_name)
        observation.setdefault("task_type", self._task_type)
        observation.setdefault("difficulty", self._difficulty)
        current_stage_id = int(observation.get("stage_id", self._stage_id))
        current_stage_name = str(observation.get("stage_name", self._stage_name))
        current_task_type = str(observation.get("task_type", self._task_type))
        current_difficulty = float(observation.get("difficulty", self._difficulty))
        graph = self._graph_builder.build(observation, self._latest_graph_payload)
        output = self._expert.step(observation, graph.to_dict())
        log_observation = self._with_expert_roles(observation, output.module_roles)
        log_graph = self._graph_builder.build(log_observation, self._latest_graph_payload)
        stamp = graph.stamp if graph.stamp > 0.0 else time.time()
        action_payload = output.to_action_payload(stamp, current_stage_id, current_task_type)

        self._combined_action_pub.publish(dict_to_string_msg(action_payload))
        self._locomotion_pub.publish(
            dict_to_string_msg(
                {
                    "stamp": stamp,
                    "stage_id": current_stage_id,
                    "task_type": current_task_type,
                    "locomotion": action_payload["locomotion"],
                }
            )
        )
        self._magnetic_pub.publish(
            dict_to_string_msg(
                {
                    "stamp": stamp,
                    "stage_id": current_stage_id,
                    "task_type": current_task_type,
                    "magnetic": action_payload["magnetic"],
                }
            )
        )
        self._expert_state_pub.publish(dict_to_string_msg(action_payload["expert"]))
        dataset_log_period = max(1, int(self.get_parameter("dataset_log_period").value))
        if self._timestep % dataset_log_period == 0 or output.done:
            self._dataset_logger.log_step(
                episode_id=self._episode_id,
                timestep=self._timestep,
                observation=log_observation,
                graph=log_graph,
                expert_output=output,
                stage_name=current_stage_name,
                stage_id=current_stage_id,
                task_type=current_task_type,
                difficulty=current_difficulty,
            )
        self._timestep += 1

    def _with_expert_roles(
        self,
        observation: Mapping[str, Any],
        module_roles: Mapping[str, str],
    ) -> dict[str, Any]:
        """Return an observation copy enriched with roles selected by the expert."""
        enriched = dict(observation)
        modules = enriched.get("modules")
        if isinstance(modules, Mapping):
            enriched_modules: dict[str, Any] = {}
            for module_id, payload in modules.items():
                if isinstance(module_id, str) and isinstance(payload, Mapping):
                    module_payload = dict(payload)
                    if module_id in module_roles:
                        module_payload["role"] = module_roles[module_id]
                        module_payload["target_role"] = module_roles[module_id]
                    enriched_modules[module_id] = module_payload
                else:
                    enriched_modules[str(module_id)] = payload
            enriched["modules"] = enriched_modules
        elif isinstance(modules, list):
            enriched_modules = []
            for payload in modules:
                if not isinstance(payload, Mapping):
                    enriched_modules.append(payload)
                    continue
                module_payload = dict(payload)
                module_id = module_payload.get("module_id")
                if isinstance(module_id, str) and module_id in module_roles:
                    module_payload["role"] = module_roles[module_id]
                    module_payload["target_role"] = module_roles[module_id]
                enriched_modules.append(module_payload)
            enriched["modules"] = enriched_modules
        return enriched


def main(args: list[str] | None = None) -> None:
    """Run the expert node."""
    rclpy.init(args=args)
    node = ExpertNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
