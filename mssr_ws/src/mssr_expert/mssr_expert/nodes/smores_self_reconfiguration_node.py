"""ROS 2 node for deterministic SMORES-EP self-reconfiguration."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from mssr_expert.dataset.dataset_logger import DatasetLogger
from mssr_expert.execution.assembly_policy import (
    AssemblyExecutionPolicy,
    DEFAULT_ASSEMBLY_EXECUTION_POLICY,
)
from mssr_expert.execution.self_reconfiguration_executor import (
    SelfReconfigurationDecision,
    SelfReconfigurationExecutor,
)
from mssr_expert.execution.primitive_protocol import (
    parse_primitive_statuses,
)
from mssr_expert.experts.expert_output import ExpertOutput
from mssr_expert.graph.attributed_robot_graph import AttributedRobotGraph
from mssr_expert.graph.graph_builder import GraphBuilder
from mssr_expert.graph.serialization import load_attributed_graph
from mssr_expert.graph.task_graph import TaskGraphBuilder
from mssr_expert.planning.smores_ep.attributed_adapter import (
    target_roles_from_graph,
)
from mssr_expert.planning.smores_ep.self_reconfiguration_planner import (
    SelfReconfigurationPlan,
    SmoresSelfReconfigurationPlanner,
)
from mssr_expert.planning.smores_ep.assignment import AssignmentResult
from mssr_expert.utils.json_io import dict_to_string_msg, string_msg_to_dict


@dataclass(frozen=True)
class _PendingTransition:
    timestep: int
    observation: Mapping[str, Any]
    graph: AttributedRobotGraph
    task_graph: AttributedRobotGraph
    expert_output: ExpertOutput


def load_morphology_catalog(
    config_directory: Path,
) -> dict[str, AttributedRobotGraph]:
    """Load every named target morphology installed with the package."""

    catalog: dict[str, AttributedRobotGraph] = {}
    for path in sorted(config_directory.glob("smores_*.json")):
        try:
            graph = load_attributed_graph(path)
        except ValueError:
            continue
        name = graph.global_attributes.get("morphology_name")
        if (
            not isinstance(name, str)
            or not name.strip()
            or graph.global_attributes.get("graph_kind")
            != "target_morphology"
        ):
            continue
        if name in catalog:
            raise ValueError(f"Duplicated morphology_name {name!r}.")
        catalog[name] = graph
    if not catalog:
        raise ValueError(
            f"No target morphologies found in {config_directory}."
        )
    return catalog


class SmoresSelfReconfigurationNode(Node):
    """Transform one connected morphology into another connected morphology."""

    def __init__(self) -> None:
        super().__init__("smores_self_reconfiguration_node")
        package_share = Path(get_package_share_directory("mssr_expert"))
        config_directory = package_share / "config"
        self._declare_parameters(
            config_directory / "smores_snake7.json"
        )
        self._morphology_catalog = load_morphology_catalog(
            config_directory
        )
        target_name = str(
            self.get_parameter("target_morphology").value
        ).strip()
        if target_name:
            if target_name not in self._morphology_catalog:
                raise ValueError(
                    f"Unknown target_morphology {target_name!r}; available="
                    f"{sorted(self._morphology_catalog)}."
                )
            self._target_graph_path: Path | None = None
            self._target_graph = self._morphology_catalog[target_name]
        else:
            self._target_graph_path = Path(
                str(self.get_parameter("target_graph_path").value)
            )
            self._target_graph = load_attributed_graph(
                self._target_graph_path
            )
        source_path_value = str(
            self.get_parameter("source_graph_path").value
        ).strip()
        self._source_graph_path: Path | None = None
        self._source_graph: AttributedRobotGraph | None = None
        self._source_assignment: AssignmentResult | None = None
        if source_path_value.lower() != "auto":
            self._source_graph_path = Path(source_path_value)
            self._source_graph = load_attributed_graph(
                self._source_graph_path
            )
        self._planner = SmoresSelfReconfigurationPlanner(
            parallel_path_clearance_m=float(
                self.get_parameter("parallel_path_clearance_m").value
            ),
            max_parallel_actions=int(
                self.get_parameter("max_parallel_actions").value
            ),
            assignment_staging_distance_m=float(
                self.get_parameter(
                    "assignment_staging_distance_m"
                ).value
            ),
            assignment_corridor_clearance_m=float(
                self.get_parameter(
                    "assignment_corridor_clearance_m"
                ).value
            ),
        )
        self._graph_builder = GraphBuilder()
        self._task_graph_builder = TaskGraphBuilder()
        self._plan: SelfReconfigurationPlan | None = None
        self._executor: SelfReconfigurationExecutor | None = None
        self._latest_observation: dict[str, Any] = {}
        self._latest_graph_payload: dict[str, Any] = {}
        self._latest_status_payload: dict[str, Any] = {}
        self._last_planning_error = ""
        self._cancel_requested_goal_id: str | None = None
        self._resolved_execution_id = ""
        self._terminal_reached = False
        self._pending_transition: _PendingTransition | None = None
        self._timestep = 0
        self._dataset_logger = DatasetLogger(
            Path(str(self.get_parameter("dataset_path").value))
        )

        self._goal_publisher = self.create_publisher(
            String,
            str(self.get_parameter("primitive_goal_topic").value),
            10,
        )
        self._cancel_publisher = self.create_publisher(
            String,
            str(self.get_parameter("primitive_cancel_topic").value),
            10,
        )
        self._state_publisher = self.create_publisher(
            String,
            str(self.get_parameter("expert_state_topic").value),
            10,
        )
        task_graph_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._task_graph_publisher = self.create_publisher(
            String,
            str(self.get_parameter("task_graph_topic").value),
            task_graph_qos,
        )

        self.create_subscription(
            String,
            str(self.get_parameter("state_graph_topic").value),
            self._on_state_graph,
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
            str(self.get_parameter("robot_graph_topic").value),
            self._on_robot_graph,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("primitive_status_topic").value),
            self._on_primitive_status,
            10,
        )
        rate_hz = max(
            1.0e-6,
            float(self.get_parameter("control_rate_hz").value),
        )
        self._timer = self.create_timer(1.0 / rate_hz, self._step)
        self.get_logger().info(
            "SMORES self-reconfiguration expert ready. "
            f"Catalog={sorted(self._morphology_catalog)}, "
            f"target={self._target_name()}, source=auto"
            if self._source_graph is None
            else "SMORES self-reconfiguration expert ready. "
            f"target={self._target_name()}, "
            f"source={self._morphology_name(self._source_graph)}"
        )

    def _declare_parameters(self, default_target_path: Path) -> None:
        parameters = {
            "target_graph_path": str(default_target_path),
            "target_morphology": "",
            "source_graph_path": "auto",
            "execution_id": "morphology-transition",
            "episode_id": "smores_reconfiguration_0001",
            "dataset_path": "logs/datasets/smores_reconfiguration.jsonl",
            "control_rate_hz": 20.0,
            "joint_timeout_s": 30.0,
            "undock_timeout_s": 10.0,
            "align_timeout_s": (
                DEFAULT_ASSEMBLY_EXECUTION_POLICY.align_timeout_s
            ),
            "dock_timeout_s": (
                DEFAULT_ASSEMBLY_EXECUTION_POLICY.dock_timeout_s
            ),
            "retry_count": 2,
            "align_retry_count": (
                DEFAULT_ASSEMBLY_EXECUTION_POLICY.align_retry_count
            ),
            "dock_recovery_count": (
                DEFAULT_ASSEMBLY_EXECUTION_POLICY.dock_recovery_count
            ),
            "contact_quality_planar_tolerance_m": (
                DEFAULT_ASSEMBLY_EXECUTION_POLICY
                .contact_quality_planar_tolerance_m
            ),
            "contact_quality_retry_count": (
                DEFAULT_ASSEMBLY_EXECUTION_POLICY.contact_quality_retry_count
            ),
            "top_bottom_contact_tolerance_m": (
                DEFAULT_ASSEMBLY_EXECUTION_POLICY
                .top_bottom_contact_tolerance_m
            ),
            "contact_approach_feedback": (
                DEFAULT_ASSEMBLY_EXECUTION_POLICY.contact_approach_feedback
            ),
            "max_concurrent_alignments_per_wave": (
                DEFAULT_ASSEMBLY_EXECUTION_POLICY
                .max_concurrent_alignments_per_wave
            ),
            "snap_docking_faces_to_nominal": (
                DEFAULT_ASSEMBLY_EXECUTION_POLICY
                .snap_docking_faces_to_nominal
            ),
            "parallel_path_clearance_m": 0.12,
            "max_parallel_actions": 0,
            "assignment_staging_distance_m": 0.070,
            "assignment_corridor_clearance_m": 0.110,
            "state_graph_topic": "/mssr/state_graph",
            "module_states_topic": "/mssr/module_states",
            "robot_graph_topic": "/mssr/robot_graph",
            "primitive_goal_topic": "/mssr/primitives/goal",
            "primitive_status_topic": "/mssr/primitives/status",
            "primitive_cancel_topic": "/mssr/primitives/cancel",
            "expert_state_topic": "/mssr/expert/self_reconfiguration/state",
            "task_graph_topic": "/mssr/expert/task_graph",
        }
        for name, value in parameters.items():
            self.declare_parameter(name, value)

    def _on_state_graph(self, message: String) -> None:
        payload = string_msg_to_dict(message)
        state = payload.get("state")
        graph = payload.get("graph")
        if isinstance(state, Mapping):
            self._latest_observation = dict(state)
        if isinstance(graph, Mapping):
            self._latest_graph_payload = dict(graph)

    def _on_module_states(self, message: String) -> None:
        payload = string_msg_to_dict(message)
        if payload:
            self._latest_observation = payload

    def _on_robot_graph(self, message: String) -> None:
        payload = string_msg_to_dict(message)
        if payload:
            self._latest_graph_payload = payload

    def _on_primitive_status(self, message: String) -> None:
        payload = string_msg_to_dict(message)
        if payload:
            self._latest_status_payload = payload

    def _step(self) -> None:
        if not self._latest_observation and not self._latest_graph_payload:
            return
        current_graph = self._graph_builder.build(
            self._latest_observation,
            self._latest_graph_payload,
        )
        self._flush_pending_transition(current_graph)
        if self._terminal_reached:
            return
        if self._plan is None and self._cancel_stale_primitive_goal():
            return
        if self._plan is None and not self._create_plan(current_graph):
            return
        if self._plan is None or self._executor is None:
            return

        decision = self._executor.step(
            self._latest_status_payload,
            current_graph=current_graph,
        )
        task_graph = self._current_task_graph(current_graph, decision)
        expert_output = self._expert_output(decision)
        self._publish(decision, task_graph)
        self._pending_transition = _PendingTransition(
            timestep=self._timestep,
            observation=dict(self._latest_observation),
            graph=current_graph,
            task_graph=task_graph,
            expert_output=expert_output,
        )
        self._timestep += 1
        if decision.done:
            self._terminal_reached = True
            log = (
                self.get_logger().info
                if decision.success
                else self.get_logger().error
            )
            log(
                "Self-reconfiguration completed."
                if decision.success
                else f"Self-reconfiguration failed: {decision.message}"
            )

    def _cancel_stale_primitive_goal(self) -> bool:
        """Release primitive resources left by the source expert.

        The file-backed runtime retains active feedback for every concurrent
        primitive.  Before taking ownership, cancel one foreign active goal at
        a time so the single cancel file cannot be overwritten by a burst.
        """

        try:
            statuses = parse_primitive_statuses(
                self._latest_status_payload
            )
        except ValueError as error:
            self.get_logger().warning(
                f"Cannot inspect primitive preflight status: {error}"
            )
            return True
        active_goal_ids = tuple(
            sorted(
                status.goal_id
                for status in statuses.values()
                if status.state in {"accepted", "running"}
            )
        )
        if not active_goal_ids:
            if self._cancel_requested_goal_id is not None:
                self.get_logger().info(
                    "Primitive-resource preflight completed."
                )
            self._cancel_requested_goal_id = None
            return False

        requested = self._cancel_requested_goal_id
        if requested in active_goal_ids:
            return True

        goal_id = active_goal_ids[0]
        self._cancel_requested_goal_id = goal_id
        self._cancel_publisher.publish(
            dict_to_string_msg({"goal_id": goal_id})
        )
        self.get_logger().warning(
            "Canceling stale primitive before reconfiguration: "
            f"{goal_id}."
        )
        return True

    def _create_plan(self, current_graph: AttributedRobotGraph) -> bool:
        try:
            if self._source_graph is None:
                (
                    self._source_graph,
                    self._source_assignment,
                ) = self._detect_source_morphology(current_graph)
            elif self._source_assignment is None:
                self._source_assignment = (
                    self._planner.configuration_assignment(
                        current_graph,
                        self._source_graph,
                    )
                )
                if self._source_assignment is None:
                    raise ValueError(
                        "The connected graph does not match the declared "
                        "source "
                        f"{self._morphology_name(self._source_graph)!r}."
                    )
            plan = self._planner.plan(
                current_graph,
                self._target_graph,
                source_graph=self._source_graph,
                source_assignment=self._source_assignment,
            )
        except (KeyError, RuntimeError, ValueError) as error:
            message = str(error)
            if message != self._last_planning_error:
                self.get_logger().warning(
                    "Waiting for a valid connected source morphology: "
                    + message
                )
                self._last_planning_error = message
            return False
        self._plan = plan
        requested_execution_id = str(
            self.get_parameter("execution_id").value
        )
        existing_goal_ids = set()
        try:
            existing_goal_ids = set(
                parse_primitive_statuses(
                    self._latest_status_payload
                )
            )
        except ValueError:
            pass
        execution_id = requested_execution_id
        if any(
            goal_id == requested_execution_id
            or goal_id.startswith(requested_execution_id + "-")
            for goal_id in existing_goal_ids
        ):
            execution_id = (
                f"{requested_execution_id}-{time.time_ns()}"
            )
            self.get_logger().warning(
                "execution_id already exists in the retained primitive "
                f"history; using {execution_id!r}."
            )
        self._resolved_execution_id = execution_id
        assembly_policy = AssemblyExecutionPolicy.from_parameter_getter(
            self.get_parameter
        )
        self._executor = SelfReconfigurationExecutor(
            plan,
            execution_id=execution_id,
            joint_timeout_s=float(
                self.get_parameter("joint_timeout_s").value
            ),
            undock_timeout_s=float(
                self.get_parameter("undock_timeout_s").value
            ),
            retry_count=int(self.get_parameter("retry_count").value),
            assembly_policy=assembly_policy,
        )
        self.get_logger().info(
            f"Reconfiguration plan {plan.source_morphology} -> "
            f"{plan.target_morphology}: "
            f"retained={plan.retained_connection_count}, "
            f"undock={len(plan.detach_actions)}, "
            f"dock={plan.new_connection_count}, "
            f"motion_cost={plan.assignment.total_cost:.3f}m, "
            f"future_blockers={plan.assignment.total_future_blockers}, "
            f"progressive_waves={len(plan.stages)}, "
            f"parallel_wave_sizes="
            f"{[len(stage.mobile_module_ids) for stage in plan.stages]}, "
            f"prepare_tilts={len(plan.prepare_tilt_by_module)}, "
            f"final_tilts={len(plan.final_tilt_by_module)}."
        )
        return True

    def _detect_source_morphology(
        self,
        current_graph: AttributedRobotGraph,
    ) -> tuple[AttributedRobotGraph, AssignmentResult]:
        matches: list[tuple[str, AttributedRobotGraph, AssignmentResult]] = []
        for name, graph in sorted(self._morphology_catalog.items()):
            assignment = self._planner.configuration_assignment(
                current_graph,
                graph,
            )
            if assignment is not None:
                matches.append((name, graph, assignment))
        if not matches:
            raise ValueError(
                "The connected graph does not match any known morphology: "
                f"{sorted(self._morphology_catalog)}."
            )
        if len(matches) > 1:
            raise ValueError(
                "Ambiguous source morphology: "
                f"{[name for name, _, _ in matches]}."
            )
        name, graph, assignment = matches[0]
        self.get_logger().info(
            f"Automatically detected source morphology {name!r}."
        )
        return graph, assignment

    @staticmethod
    def _morphology_name(graph: AttributedRobotGraph) -> str:
        return str(
            graph.global_attributes.get("morphology_name", "unknown")
        )

    def _target_name(self) -> str:
        return self._morphology_name(self._target_graph)

    def _current_task_graph(
        self,
        current_graph: AttributedRobotGraph,
        decision: SelfReconfigurationDecision,
    ) -> AttributedRobotGraph:
        if self._plan is None:
            raise RuntimeError("Cannot build a task graph without a plan.")
        return self._task_graph_builder.build(
            current_graph=current_graph,
            target_graph=self._plan.target_graph,
            assignment=self._plan.assignment.target_to_module,
            execution_state={
                "expert": "self_reconfiguration",
                "state": decision.state,
                "phase": decision.phase,
                "active_goal_ids": list(decision.active_goal_ids),
                "retained_connection_count": (
                    decision.retained_connection_count
                ),
                "completed_operation_count": (
                    decision.completed_operation_count
                ),
                "total_operation_count": decision.total_operation_count,
                "source_morphology": self._plan.source_morphology,
                "target_morphology": self._plan.target_morphology,
                "assignment_motion_cost_m": (
                    self._plan.assignment.total_cost
                ),
                "assignment_future_blockers": (
                    self._plan.assignment.total_future_blockers
                ),
                "done": decision.done,
                "success": decision.success,
                "message": decision.message,
            },
        )

    def _expert_output(
        self,
        decision: SelfReconfigurationDecision,
    ) -> ExpertOutput:
        if self._plan is None:
            raise RuntimeError("Cannot build expert output without a plan.")
        goal = decision.primitive_goal
        target_roles = target_roles_from_graph(self._plan.target_graph)
        module_roles = {
            module_id: str(target_roles[target_id]["target_role"])
            for target_id, module_id in (
                self._plan.assignment.target_to_module.items()
            )
        }
        return ExpertOutput(
            fsm_state=decision.state,
            active_primitive=goal.primitive if goal is not None else None,
            primitive_params=(
                dict(goal.parameters) if goal is not None else {}
            ),
            primitive_goal=decision.primitive_goal_payload,
            module_roles=module_roles,
            task_metrics={
                "retained_connection_count": (
                    decision.retained_connection_count
                ),
                "completed_operation_count": (
                    decision.completed_operation_count
                ),
                "total_operation_count": decision.total_operation_count,
                "source_morphology": self._plan.source_morphology,
                "target_morphology": self._plan.target_morphology,
            },
            success=decision.success,
            done=decision.done,
            debug={
                "phase": decision.phase,
                "active_goal_ids": list(decision.active_goal_ids),
                "message": decision.message,
            },
        )

    def _publish(
        self,
        decision: SelfReconfigurationDecision,
        task_graph: AttributedRobotGraph,
    ) -> None:
        if decision.primitive_goal is not None:
            self._goal_publisher.publish(
                dict_to_string_msg(decision.primitive_goal.to_dict())
            )
        self._state_publisher.publish(
            dict_to_string_msg(
                {
                    "schema_version": "mssr.self_reconfiguration_state.v1",
                    "stamp": time.time(),
                    "state": decision.state,
                    "phase": decision.phase,
                    "active_goal_ids": list(decision.active_goal_ids),
                    "retained_connection_count": (
                        decision.retained_connection_count
                    ),
                    "completed_operation_count": (
                        decision.completed_operation_count
                    ),
                    "total_operation_count": decision.total_operation_count,
                    "source_morphology": self._plan.source_morphology,
                    "target_morphology": self._plan.target_morphology,
                    "done": decision.done,
                    "success": decision.success,
                    "message": decision.message,
                }
            )
        )
        self._task_graph_publisher.publish(
            dict_to_string_msg(task_graph.to_dict())
        )

    def _flush_pending_transition(
        self,
        next_graph: AttributedRobotGraph,
    ) -> None:
        pending = self._pending_transition
        if pending is None or self._plan is None:
            return
        self._dataset_logger.log_step(
            episode_id=str(self.get_parameter("episode_id").value),
            timestep=pending.timestep,
            observation=pending.observation,
            graph=pending.graph,
            expert_output=pending.expert_output,
            stage_name="self_reconfiguration",
            stage_id=0,
            task_type="self_reconfiguration",
            difficulty=0.0,
            task_graph=pending.task_graph,
            target_graph=self._plan.target_graph,
            assignment=self._plan.assignment.target_to_module,
            next_graph=next_graph,
        )
        self._pending_transition = None


def main(args: list[str] | None = None) -> None:
    """Run the SMORES self-reconfiguration expert."""

    rclpy.init(args=args)
    node = SmoresSelfReconfigurationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
