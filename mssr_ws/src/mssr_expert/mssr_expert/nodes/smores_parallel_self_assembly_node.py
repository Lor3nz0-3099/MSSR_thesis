"""ROS 2 node for deterministic parallel SMORES-EP self-assembly."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import rclpy
from ament_index_python.packages import (
    get_package_share_directory,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from mssr_expert.dataset.dataset_logger import DatasetLogger
from mssr_expert.execution.assembly_policy import (
    AssemblyExecutionPolicy,
    DEFAULT_ASSEMBLY_EXECUTION_POLICY,
)
from mssr_expert.execution.parallel_assembly_executor import (
    AssemblyExecutionDecision,
    ParallelAssemblyExecutor,
    physical_fold_push_pairs,
    physical_posture_groups,
)
from mssr_expert.execution.primitive_protocol import (
    parse_primitive_statuses,
)
from mssr_expert.experts.expert_output import ExpertOutput
from mssr_expert.graph.attributed_robot_graph import (
    AttributedRobotGraph,
)
from mssr_expert.graph.graph_builder import GraphBuilder
from mssr_expert.graph.serialization import (
    load_attributed_graph,
)
from mssr_expert.graph.task_graph import TaskGraphBuilder
from mssr_expert.planning.smores_ep.attributed_adapter import (
    target_roles_from_graph,
)
from mssr_expert.planning.smores_ep.parallel_self_assembly_planner import (
    ParallelSelfAssemblyPlanner,
    ParallelSelfAssemblyPlanningResult,
)
from mssr_expert.utils.json_io import (
    dict_to_string_msg,
    string_msg_to_dict,
)


@dataclass(frozen=True)
class _PendingTransition:
    """Dataset transition waiting for the following graph."""

    timestep: int
    observation: Mapping[str, Any]
    graph: AttributedRobotGraph
    task_graph: AttributedRobotGraph
    expert_output: ExpertOutput


class SmoresParallelSelfAssemblyNode(Node):
    """Plan and execute one target SMORES-EP morphology."""

    def __init__(self) -> None:
        super().__init__(
            "smores_parallel_self_assembly_node"
        )

        package_share = Path(
            get_package_share_directory("mssr_expert")
        )

        default_target_path = (
            package_share
            / "config"
            / "smores_three_module_chain.json"
        )

        self._declare_parameters(
            default_target_path
        )

        self._target_graph_path = Path(
            str(
                self.get_parameter(
                    "target_graph_path"
                ).value
            )
        )
        self._target_graph = load_attributed_graph(
            self._target_graph_path
        )

        self._planner = ParallelSelfAssemblyPlanner(
            orientation_weight_m_per_rad=float(
                self.get_parameter(
                    "orientation_weight_m_per_rad"
                ).value
            ),
            require_disconnected_modules=bool(
                self.get_parameter(
                    "require_disconnected_modules"
                ).value
            ),
            layout_clearance_m=float(
                self.get_parameter(
                    "layout_clearance_m"
                ).value
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

        self._planning_result: (
            ParallelSelfAssemblyPlanningResult | None
        ) = None

        self._executor: (
            ParallelAssemblyExecutor | None
        ) = None

        self._dataset_logger = DatasetLogger(
            Path(
                str(
                    self.get_parameter(
                        "dataset_path"
                    ).value
                )
            )
        )

        self._latest_observation: dict[str, Any] = {}
        self._latest_graph_payload: dict[str, Any] = {}
        self._latest_status_payload: dict[str, Any] = {}

        self._pending_transition: (
            _PendingTransition | None
        ) = None

        self._timestep = 0
        self._last_planning_error = ""
        self._terminal_reached = False

        self._goal_publisher = self.create_publisher(
            String,
            str(
                self.get_parameter(
                    "primitive_goal_topic"
                ).value
            ),
            10,
        )

        self._expert_state_publisher = (
            self.create_publisher(
                String,
                str(
                    self.get_parameter(
                        "expert_state_topic"
                    ).value
                ),
                10,
            )
        )

        task_graph_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._task_graph_publisher = (
            self.create_publisher(
                String,
                str(
                    self.get_parameter(
                        "task_graph_topic"
                    ).value
                ),
                task_graph_qos,
            )
        )

        self.create_subscription(
            String,
            str(
                self.get_parameter(
                    "state_graph_topic"
                ).value
            ),
            self._on_state_graph,
            10,
        )

        self.create_subscription(
            String,
            str(
                self.get_parameter(
                    "module_states_topic"
                ).value
            ),
            self._on_module_states,
            10,
        )

        self.create_subscription(
            String,
            str(
                self.get_parameter(
                    "robot_graph_topic"
                ).value
            ),
            self._on_robot_graph,
            10,
        )

        self.create_subscription(
            String,
            str(
                self.get_parameter(
                    "primitive_status_topic"
                ).value
            ),
            self._on_primitive_status,
            10,
        )

        control_rate_hz = max(
            1e-6,
            float(
                self.get_parameter(
                    "control_rate_hz"
                ).value
            ),
        )

        self._timer = self.create_timer(
            1.0 / control_rate_hz,
            self._step,
        )

        self.get_logger().info(
            "SMORES parallel self-assembly expert ready. "
            f"Target={self._target_graph_path}"
        )

    def _declare_parameters(
        self,
        default_target_path: Path,
    ) -> None:
        self.declare_parameter(
            "target_graph_path",
            str(default_target_path),
        )
        self.declare_parameter(
            "execution_id",
            "smores-assembly",
        )
        self.declare_parameter(
            "episode_id",
            "smores_assembly_0001",
        )
        self.declare_parameter(
            "dataset_path",
            "logs/datasets/"
            "smores_parallel_self_assembly.jsonl",
        )
        self.declare_parameter(
            "orientation_weight_m_per_rad",
            0.0,
        )
        self.declare_parameter(
            "require_disconnected_modules",
            True,
        )
        self.declare_parameter(
            "control_rate_hz",
            20.0,
        )
        self.declare_parameter(
            "align_timeout_s",
            DEFAULT_ASSEMBLY_EXECUTION_POLICY.align_timeout_s,
        )
        self.declare_parameter(
            "dock_timeout_s",
            DEFAULT_ASSEMBLY_EXECUTION_POLICY.dock_timeout_s,
        )
        self.declare_parameter(
            "align_retry_count",
            DEFAULT_ASSEMBLY_EXECUTION_POLICY.align_retry_count,
        )
        self.declare_parameter(
            "dock_recovery_count",
            DEFAULT_ASSEMBLY_EXECUTION_POLICY.dock_recovery_count,
        )
        self.declare_parameter(
            "contact_quality_planar_tolerance_m",
            DEFAULT_ASSEMBLY_EXECUTION_POLICY
            .contact_quality_planar_tolerance_m,
        )
        self.declare_parameter(
            "contact_quality_retry_count",
            DEFAULT_ASSEMBLY_EXECUTION_POLICY.contact_quality_retry_count,
        )
        self.declare_parameter(
            "top_bottom_contact_tolerance_m",
            DEFAULT_ASSEMBLY_EXECUTION_POLICY
            .top_bottom_contact_tolerance_m,
        )
        self.declare_parameter(
            "contact_approach_feedback",
            DEFAULT_ASSEMBLY_EXECUTION_POLICY.contact_approach_feedback,
        )
        self.declare_parameter(
            "max_concurrent_alignments_per_wave",
            DEFAULT_ASSEMBLY_EXECUTION_POLICY
            .max_concurrent_alignments_per_wave,
        )
        self.declare_parameter(
            "snap_docking_faces_to_nominal",
            DEFAULT_ASSEMBLY_EXECUTION_POLICY
            .snap_docking_faces_to_nominal,
        )
        self.declare_parameter(
            "layout_clearance_m",
            0.070,
        )
        self.declare_parameter(
            "assignment_staging_distance_m",
            0.070,
        )
        self.declare_parameter(
            "assignment_corridor_clearance_m",
            0.110,
        )
        self.declare_parameter(
            "global_layout_before_docking",
            False,
        )
        self.declare_parameter(
            "enable_borrowed_helper",
            False,
        )
        self.declare_parameter(
            "helper_lift_tilt_rad",
            0.7853981633974483,
        )
        self.declare_parameter(
            "helper_joint_timeout_s",
            30.0,
        )
        self.declare_parameter(
            "state_graph_topic",
            "/mssr/state_graph",
        )
        self.declare_parameter(
            "module_states_topic",
            "/mssr/module_states",
        )
        self.declare_parameter(
            "robot_graph_topic",
            "/mssr/robot_graph",
        )
        self.declare_parameter(
            "primitive_goal_topic",
            "/mssr/primitives/goal",
        )
        self.declare_parameter(
            "primitive_status_topic",
            "/mssr/primitives/status",
        )
        self.declare_parameter(
            "expert_state_topic",
            "/mssr/expert/self_assembly/state",
        )
        self.declare_parameter(
            "task_graph_topic",
            "/mssr/expert/task_graph",
        )

    def _on_state_graph(
        self,
        message: String,
    ) -> None:
        payload = string_msg_to_dict(message)

        state = payload.get("state")
        graph = payload.get("graph")

        if isinstance(state, Mapping):
            self._latest_observation = dict(state)

        if isinstance(graph, Mapping):
            self._latest_graph_payload = dict(graph)

    def _on_module_states(
        self,
        message: String,
    ) -> None:
        payload = string_msg_to_dict(message)

        if payload:
            self._latest_observation = payload

    def _on_robot_graph(
        self,
        message: String,
    ) -> None:
        payload = string_msg_to_dict(message)

        if payload:
            self._latest_graph_payload = payload

    def _on_primitive_status(
        self,
        message: String,
    ) -> None:
        payload = string_msg_to_dict(message)

        if payload:
            self._latest_status_payload = payload

    def _step(self) -> None:
        if (
            not self._latest_observation
            and not self._latest_graph_payload
        ):
            return

        current_graph = self._graph_builder.build(
            observation=self._latest_observation,
            graph_payload=self._latest_graph_payload,
        )

        self._flush_pending_transition(
            next_graph=current_graph
        )

        if self._terminal_reached:
            return

        if self._planning_result is None:
            if not self._create_plan(current_graph):
                return

        if (
            self._planning_result is None
            or self._executor is None
        ):
            return

        decision = self._executor.step(
            self._latest_status_payload
        )

        task_graph = self._current_task_graph(
            current_graph,
            decision,
        )

        expert_output = self._expert_output(
            decision
        )

        self._publish_decision(
            decision,
            task_graph,
        )

        self._pending_transition = (
            _PendingTransition(
                timestep=self._timestep,
                observation=dict(
                    self._latest_observation
                ),
                graph=current_graph,
                task_graph=task_graph,
                expert_output=expert_output,
            )
        )

        self._timestep += 1

        if decision.done:
            self._terminal_reached = True

            if decision.success:
                self.get_logger().info(
                    "Parallel self-assembly completed."
                )
            else:
                self.get_logger().error(
                    "Parallel self-assembly failed: "
                    f"{decision.message}"
                )

    def _create_plan(
        self,
        current_graph: AttributedRobotGraph,
    ) -> bool:
        try:
            result = self._planner.plan(
                current_graph=current_graph,
                target_graph=self._target_graph,
            )
        except (KeyError, RuntimeError, ValueError) as error:
            message = str(error)

            if message != self._last_planning_error:
                self.get_logger().error(
                    "Cannot create assembly plan: "
                    f"{message}"
                )
                self._last_planning_error = message

            return False

        self._planning_result = result

        post_assembly_tilts: dict[str, float] = {}
        raw_posture = result.target_graph.global_attributes.get(
            "post_assembly_tilt_rad_by_vertex",
            {},
        )
        if isinstance(raw_posture, Mapping):
            for target_vertex, angle in raw_posture.items():
                target_vertex = str(target_vertex)
                module_id = result.assignment.target_to_module.get(
                    target_vertex
                )
                if module_id is None:
                    raise ValueError(
                        "Post-assembly posture references unknown target "
                        f"vertex {target_vertex!r}."
                    )
                post_assembly_tilts[module_id] = float(angle)

        post_assembly_pans: dict[str, float] = {}
        raw_pan_posture = result.target_graph.global_attributes.get(
            "post_assembly_pan_rad_by_vertex",
            {},
        )
        if isinstance(raw_pan_posture, Mapping):
            for target_vertex, angle in raw_pan_posture.items():
                target_vertex = str(target_vertex)
                module_id = result.assignment.target_to_module.get(
                    target_vertex
                )
                if module_id is None:
                    raise ValueError(
                        "Post-assembly PAN posture references unknown target "
                        f"vertex {target_vertex!r}."
                    )
                post_assembly_pans[module_id] = float(angle)

        target_requires_helper = bool(
            result.target_graph.global_attributes.get(
                "requires_helping_module",
                False,
            )
        )
        helper_enabled = (
            bool(
                self.get_parameter(
                    "enable_borrowed_helper"
                ).value
            )
            or target_requires_helper
        )

        assembly_policy = AssemblyExecutionPolicy.from_parameter_getter(
            self.get_parameter
        )

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

        self._executor = ParallelAssemblyExecutor(
            plan=result.assembly_plan,
            execution_id=execution_id,
            **assembly_policy.executor_kwargs(),
            enable_borrowed_helper=helper_enabled,
            helper_lift_tilt_rad=float(
                self.get_parameter(
                    "helper_lift_tilt_rad"
                ).value
            ),
            helper_joint_timeout_s=float(
                self.get_parameter(
                    "helper_joint_timeout_s"
                ).value
            ),
            layout_pose_by_module=(
                result.layout_pose_by_module
                if bool(
                    self.get_parameter(
                        "global_layout_before_docking"
                    ).value
                )
                else {}
            ),
            post_assembly_tilt_by_module=post_assembly_tilts,
            post_assembly_pan_by_module=post_assembly_pans,
            posture_tilt_tolerance_rad=(
                None
                if result.target_graph.global_attributes.get(
                    "post_assembly_tilt_tolerance_rad"
                ) is None
                else float(
                    result.target_graph.global_attributes[
                        "post_assembly_tilt_tolerance_rad"
                    ]
                )
            ),
            posture_tilt_max_servo_error_rad=(
                None
                if result.target_graph.global_attributes.get(
                    "post_assembly_tilt_max_servo_error_rad"
                ) is None
                else float(
                    result.target_graph.global_attributes[
                        "post_assembly_tilt_max_servo_error_rad"
                    ]
                )
            ),
            coordinate_posture_tilts=bool(
                result.target_graph.global_attributes.get(
                    "coordinate_post_assembly_tilts",
                    False,
                )
            ),
            posture_tilt_groups_by_module=physical_posture_groups(
                result.target_graph.global_attributes.get(
                    "post_assembly_tilt_groups_by_vertex"
                ),
                result.assignment.target_to_module,
                set(post_assembly_tilts),
            ),
            posture_push_by_lifter_module=physical_fold_push_pairs(
                result.target_graph.global_attributes.get(
                    "post_assembly_push_pairs_by_vertex"
                ),
                result.assignment.target_to_module,
                set(post_assembly_tilts),
            ),
            additional_known_module_ids=result.reserve_module_ids,
        )

        self.get_logger().info(
            "Assembly plan created: "
            f"root={result.physical_root_id}, "
            f"waves={len(result.assembly_plan.waves)}, "
            f"actions={result.assembly_plan.action_count}, "
            f"motion_cost={result.assignment.total_cost:.3f}m, "
            "future_blockers="
            f"{result.assignment.total_future_blockers}, "
            f"reserves={list(result.reserve_module_ids)}"
        )

        helper_actions = tuple(
            action
            for action in result.assembly_plan.all_actions
            if action.requires_helper
        )
        if helper_actions:
            if target_requires_helper:
                self.get_logger().info(
                    "Target metadata requests the helping-module procedure."
                )
            if self._executor.helper_module_id is None:
                self.get_logger().warning(
                    "Assembly plan needs a helping module, but no free future "
                    "target module is available to borrow."
                )
            else:
                self.get_logger().info(
                    "Helping-module procedure enabled: borrowed helper="
                    f"{self._executor.helper_module_id}, assisted_actions="
                    f"{len(helper_actions)}."
                )

        return True

    def _current_task_graph(
        self,
        current_graph: AttributedRobotGraph,
        decision: AssemblyExecutionDecision,
    ) -> AttributedRobotGraph:
        if self._planning_result is None:
            raise RuntimeError(
                "Cannot build task graph without a plan."
            )

        return self._task_graph_builder.build(
            current_graph=current_graph,
            target_graph=(
                self._planning_result.target_graph
            ),
            assignment=(
                self._planning_result
                .assignment
                .target_to_module
            ),
            execution_state={
                "expert": (
                    "parallel_self_assembly"
                ),
                "state": decision.state,
                "phase": decision.phase,
                "wave_index": decision.wave_index,
                "wave_count": decision.wave_count,
                "active_goal_ids": list(
                    decision.active_goal_ids
                ),
                "completed_action_count": (
                    decision.completed_action_count
                ),
                "total_action_count": (
                    decision.total_action_count
                ),
                "assignment_motion_cost_m": (
                    self._planning_result.assignment.total_cost
                ),
                "assignment_future_blockers": (
                    self._planning_result
                    .assignment
                    .total_future_blockers
                ),
                "done": decision.done,
                "success": decision.success,
                "message": decision.message,
            },
        )

    def _expert_output(
        self,
        decision: AssemblyExecutionDecision,
    ) -> ExpertOutput:
        if self._planning_result is None:
            raise RuntimeError(
                "Cannot build expert output without a plan."
            )

        primitive_goal = (
            decision.primitive_goal_payload
        )

        module_roles = self._module_roles()

        active_primitive = (
            decision.primitive_goal.primitive
            if decision.primitive_goal is not None
            else None
        )

        primitive_params = (
            dict(
                decision.primitive_goal.parameters
            )
            if decision.primitive_goal is not None
            else {}
        )

        return ExpertOutput(
            fsm_state=decision.state,
            active_primitive=active_primitive,
            primitive_params=primitive_params,
            primitive_goal=primitive_goal,
            module_roles=module_roles,
            task_metrics={
                "wave_index": decision.wave_index,
                "wave_count": decision.wave_count,
                "completed_action_count": (
                    decision.completed_action_count
                ),
                "total_action_count": (
                    decision.total_action_count
                ),
            },
            success=decision.success,
            done=decision.done,
            debug={
                "phase": decision.phase,
                "active_goal_ids": list(
                    decision.active_goal_ids
                ),
                "message": decision.message,
            },
        )

    def _module_roles(
        self,
    ) -> dict[str, str]:
        if self._planning_result is None:
            return {}

        target_roles = target_roles_from_graph(
            self._planning_result.target_graph
        )

        roles: dict[str, str] = {}

        for (
            target_vertex,
            module_id,
        ) in (
            self._planning_result
            .assignment
            .target_to_module
            .items()
        ):
            role_payload = target_roles.get(
                target_vertex,
                {},
            )

            roles[module_id] = str(
                role_payload.get(
                    "target_role",
                    "unassigned",
                )
            )

        return roles

    def _publish_decision(
        self,
        decision: AssemblyExecutionDecision,
        task_graph: AttributedRobotGraph,
    ) -> None:
        if decision.primitive_goal is not None:
            self._goal_publisher.publish(
                dict_to_string_msg(
                    decision.primitive_goal.to_dict()
                )
            )

        self._expert_state_publisher.publish(
            dict_to_string_msg(
                {
                    "schema_version": (
                        "mssr.self_assembly_state.v1"
                    ),
                    "stamp": time.time(),
                    "state": decision.state,
                    "phase": decision.phase,
                    "wave_index": decision.wave_index,
                    "wave_count": decision.wave_count,
                    "active_goal_ids": list(
                        decision.active_goal_ids
                    ),
                    "completed_action_count": (
                        decision.completed_action_count
                    ),
                    "total_action_count": (
                        decision.total_action_count
                    ),
                    "done": decision.done,
                    "success": decision.success,
                    "message": decision.message,
                }
            )
        )

        self._task_graph_publisher.publish(
            dict_to_string_msg(
                task_graph.to_dict()
            )
        )

    def _flush_pending_transition(
        self,
        next_graph: AttributedRobotGraph,
    ) -> None:
        pending = self._pending_transition

        if (
            pending is None
            or self._planning_result is None
        ):
            return

        self._dataset_logger.log_step(
            episode_id=str(
                self.get_parameter(
                    "episode_id"
                ).value
            ),
            timestep=pending.timestep,
            observation=pending.observation,
            graph=pending.graph,
            expert_output=pending.expert_output,
            stage_name="parallel_self_assembly",
            stage_id=0,
            task_type="parallel_self_assembly",
            difficulty=0.0,
            task_graph=pending.task_graph,
            target_graph=(
                self._planning_result.target_graph
            ),
            assignment=(
                self._planning_result
                .assignment
                .target_to_module
            ),
            next_graph=next_graph,
        )

        self._pending_transition = None


def main(
    args: list[str] | None = None,
) -> None:
    """Run the parallel self-assembly expert node."""

    rclpy.init(args=args)

    node = SmoresParallelSelfAssemblyNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()
