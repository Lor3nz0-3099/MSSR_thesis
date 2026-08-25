"""One ROS 2 node for SMORES-EP obstacle-course task achievement.

It composes the existing deterministic assembly, reconfiguration and
morphology-behavior executors.  Course transitions are buffered and written
only after the button and exit have both been verified, yielding successful,
episode-scoped demonstrations suitable for imitation-learning curation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from pathlib import Path
from typing import Any, Mapping

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from mssr_expert.behaviors.morphology_library import AssignedModule, MorphologyLibrary
from mssr_expert.behaviors.snake_gap_gait import SnakeGapGaitPlanner
from mssr_expert.behaviors.snake_stair_gait import SnakeStairGaitPlanner
from mssr_expert.behaviors.morphology_locomotion import coherent_planar_train_commands
from mssr_expert.dataset.dataset_logger import DatasetLogger
from mssr_expert.execution.assembly_policy import DEFAULT_ASSEMBLY_EXECUTION_POLICY
from mssr_expert.execution.morphology_behavior_executor import (
    MorphologyBehaviorExecutor,
    MorphologyCommand,
)
from mssr_expert.execution.parallel_assembly_executor import (
    ParallelAssemblyExecutor,
    physical_fold_push_pairs,
    physical_posture_groups,
)
from mssr_expert.execution.primitive_protocol import parse_primitive_statuses
from mssr_expert.execution.self_reconfiguration_executor import SelfReconfigurationExecutor
from mssr_expert.experts.expert_output import ExpertOutput
from mssr_expert.graph.graph_builder import GraphBuilder
from mssr_expert.graph.task_graph import TaskGraphBuilder
from mssr_expert.nodes.smores_self_reconfiguration_node import load_morphology_catalog
from mssr_expert.planning.smores_ep.attributed_adapter import target_roles_from_graph
from mssr_expert.planning.smores_ep.course_landmarks import (
    CourseLandmarkError,
    CourseLandmarks,
)
from mssr_expert.planning.smores_ep.obstacle_course_policy import CourseStep, ObstacleCoursePolicy
from mssr_expert.planning.smores_ep.parallel_self_assembly_planner import ParallelSelfAssemblyPlanner
from mssr_expert.planning.smores_ep.self_reconfiguration_planner import SmoresSelfReconfigurationPlanner
from mssr_expert.primitives.common import (
    distance_3d,
    extract_modules,
    logical_tilt_positions,
    module_position,
)
from mssr_expert.utils.json_io import dict_to_string_msg, string_msg_to_dict


def end_effector_contact_point(
    module: Mapping[str, Any],
) -> tuple[float, float, float]:
    """Return the free TOP-face centre, falling back to the module centre."""

    connectors = module.get("connectors", ())
    if isinstance(connectors, list):
        for connector in connectors:
            if not isinstance(connector, Mapping):
                continue
            connector_id = str(
                connector.get("connector_id", connector.get("face_name", ""))
            ).upper()
            position = connector.get("position_world")
            if connector_id == "TOP" and isinstance(position, list | tuple):
                if len(position) >= 3:
                    return tuple(float(value) for value in position[:3])
    return module_position(module)


def button_contact_distance_m(
    module: Mapping[str, Any],
    button_center_xyz_m: tuple[float, float, float],
) -> float:
    """Measure simulated Vicon distance from end-effector face to plunger."""

    return distance_3d(
        end_effector_contact_point(module),
        button_center_xyz_m,
    )


@dataclass(frozen=True)
class _NavigationEngine:
    mode: str


@dataclass(frozen=True)
class _NavigationDecision:
    state: str
    phase: str
    locomotion: Mapping[str, Mapping[str, float]]
    done: bool = False
    success: bool = False
    message: str = ""


class SmoresObstacleCourseNode(Node):
    """Execute assembly, self-reconfiguration and task behaviors in one node."""

    def __init__(self) -> None:
        super().__init__("smores_obstacle_course_node")
        self._declare_parameters()
        package_share = Path(get_package_share_directory("mssr_expert"))
        self._catalog = load_morphology_catalog(package_share / "config")
        self._behavior_library = MorphologyLibrary.load(
            package_share / "config" / "smores_morphology_behaviors.json"
        )
        self._gap_gait_planner = SnakeGapGaitPlanner()
        self._stair_gait_planner = SnakeStairGaitPlanner()
        self._policy = ObstacleCoursePolicy()
        self._steps = self._policy.steps()
        self._assembly_planner = ParallelSelfAssemblyPlanner()
        self._reconfiguration_planner = SmoresSelfReconfigurationPlanner()
        self._graph_builder = GraphBuilder()
        self._task_graph_builder = TaskGraphBuilder()
        self._dataset_logger = DatasetLogger(Path(str(self.get_parameter("dataset_path").value)))

        self._latest_observation: dict[str, Any] = {}
        self._latest_graph_payload: dict[str, Any] = {}
        self._latest_status: dict[str, Any] = {}
        self._step_index = 0
        self._engine: Any | None = None
        self._assignment: dict[str, str] = {}
        self._source_graph: Any | None = None
        self._source_assignment: Any | None = None
        self._active_target: Any | None = None
        self._pending_records: list[
            tuple[Any, Any, ExpertOutput, str, int, Any, dict[str, str], dict[str, Any]]
        ] = []
        self._timestep = 0
        self._terminal = False
        self._awaiting_completion_event = False
        self._primitive_preflight_complete = False
        self._neutral_tilt_rad_by_module: dict[str, float] = {}
        self._neutral_assignment_signature: tuple[
            str, tuple[tuple[str, str], ...]
        ] | None = None

        self._goal_publisher = self.create_publisher(String, "/mssr/primitives/goal", 10)
        self._cancel_publisher = self.create_publisher(String, "/mssr/primitives/cancel", 10)
        self._actions_publisher = self.create_publisher(String, "/mssr/actions", 10)
        self._state_publisher = self.create_publisher(String, "/mssr/expert/obstacle_course/state", 10)
        task_graph_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._task_graph_publisher = self.create_publisher(String, "/mssr/expert/task_graph", task_graph_qos)
        self.create_subscription(String, "/mssr/state_graph", self._on_state_graph, 10)
        self.create_subscription(String, "/mssr/module_states", self._on_module_states, 10)
        self.create_subscription(String, "/mssr/robot_graph", self._on_robot_graph, 10)
        self.create_subscription(String, "/mssr/primitives/status", self._on_primitive_status, 10)
        rate_hz = max(1.0, float(self.get_parameter("control_rate_hz").value))
        self._timer = self.create_timer(1.0 / rate_hz, self._step)

    def _declare_parameters(self) -> None:
        self.declare_parameter("episode_id", "smores_obstacle_course_0001")
        self.declare_parameter("dataset_path", "logs/datasets/smores_obstacle_course_il.jsonl")
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("dataset_log_period", 5)
        self.declare_parameter("button_contact_radius_m", 0.040)
        self.declare_parameter("button_contact_speed_m_s", 0.012)
        self.declare_parameter("button_alignment_tolerance_m", 0.050)
        self.declare_parameter("button_navigation_tolerance_m", 0.025)
        self.declare_parameter("goal_min_modules_past_exit", 4)
        self.declare_parameter("gap_approach_margin_m", 0.06)
        self.declare_parameter("gap_clearance_margin_m", 0.08)
        self.declare_parameter("stair_approach_margin_m", 0.04)
        self.declare_parameter("navigation_speed_m_s", 0.050)
        self.declare_parameter("navigation_yaw_rate_rad_s", 0.25)
        self.declare_parameter("navigation_position_tolerance_m", 0.06)

    def _on_state_graph(self, message: String) -> None:
        payload = string_msg_to_dict(message)
        if isinstance(payload.get("state"), Mapping):
            self._latest_observation = dict(payload["state"])
        if isinstance(payload.get("graph"), Mapping):
            self._latest_graph_payload = dict(payload["graph"])

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
            self._latest_status = payload

    def _step(self) -> None:
        if self._terminal or (not self._latest_observation and not self._latest_graph_payload):
            return
        current_graph = self._graph_builder.build(self._latest_observation, self._latest_graph_payload)
        if not self._primitive_preflight_complete:
            self._cancel_stale_primitive_goals()
            self._primitive_preflight_complete = True
            self.get_logger().info("Primitive-resource preflight completed.")
        if self._step_index >= len(self._steps):
            self._finish(True, current_graph, "Obstacle course completed.")
            return
        course_step = self._steps[self._step_index]
        if self._awaiting_completion_event:
            if course_step.requires_button and not self._button_pressed():
                decision = self._approach_button_contact(current_graph)
                self._publish_actions(course_step, decision.locomotion, decision)
                self._publish_state(
                    course_step,
                    decision.state,
                    False,
                    False,
                    decision.message,
                )
                return
            if course_step.requires_goal and not self._goal_reached():
                self._publish_state(course_step, "WAITING_EXIT", False, False, "Robot has not crossed the exit plane.")
                return
            self._advance(course_step)
            return
        if self._engine is None:
            try:
                self._start_step(course_step, current_graph)
            except (KeyError, RuntimeError, TypeError, ValueError) as error:
                self._publish_state(course_step, "WAITING_FOR_PLAN", False, False, str(error))
                return

        decision, output = self._run_engine(course_step, current_graph)
        task_graph = self._task_graph_builder.build(
            current_graph=current_graph,
            target_graph=self._active_target,
            assignment=self._assignment,
            execution_state={
                "expert": "smores_obstacle_course",
                "course_step": course_step.task,
                "selected_morphology": course_step.morphology,
                "state": decision.state,
                "phase": decision.phase,
                "done": decision.done,
                "success": decision.success,
            },
        )
        self._task_graph_publisher.publish(dict_to_string_msg(task_graph.to_dict()))
        self._publish_state(course_step, decision.state, decision.done, decision.success, decision.message)
        period = max(1, int(self.get_parameter("dataset_log_period").value))
        if self._timestep % period == 0 or decision.done:
            self._pending_records.append(
                (
                    current_graph,
                    task_graph,
                    output,
                    course_step.task,
                    self._timestep,
                    self._active_target,
                    dict(self._assignment),
                    dict(self._latest_observation),
                )
            )
        self._timestep += 1
        if not decision.done:
            return
        if not decision.success:
            self._finish(False, current_graph, decision.message)
            return
        if course_step.requires_button and not self._button_pressed():
            self._awaiting_completion_event = True
            self._publish_state(course_step, "WAITING_BUTTON_CONTACT", False, False, "End effector has not reached the button.")
            return
        if course_step.requires_goal and not self._goal_reached():
            self._awaiting_completion_event = True
            self._publish_state(course_step, "WAITING_EXIT", False, False, "Robot has not crossed the exit plane.")
            return
        self._advance(course_step)

    def _cancel_stale_primitive_goals(self) -> None:
        """Best-effort cancel retained goals from an interrupted session."""
        try:
            statuses = parse_primitive_statuses(self._latest_status)
        except ValueError as error:
            self.get_logger().warning(
                f"Cannot inspect primitive preflight status: {error}"
            )
            return
        active_goal_ids = tuple(
            sorted(
                status.goal_id
                for status in statuses.values()
                if status.state in {"accepted", "running"}
            )
        )
        for goal_id in active_goal_ids:
            self._cancel_publisher.publish(
                dict_to_string_msg({"goal_id": goal_id})
            )
        if active_goal_ids:
            self.get_logger().warning(
                "Requested cancellation of stale primitive goals before "
                "course start: " + ", ".join(active_goal_ids)
            )

    def _start_step(self, course_step: CourseStep, current_graph: Any) -> None:
        self._active_target = self._catalog[course_step.morphology]
        execution_id = f"{self.get_parameter('episode_id').value}-{self._step_index:02d}"
        if course_step.task == "assembly":
            result = self._assembly_planner.plan(current_graph, self._active_target)
            self._assignment = dict(result.assignment.target_to_module)
            post_assembly_tilts = self._post_assembly_tilts(result)
            post_assembly_pans = self._post_assembly_pans(result)
            assembly_kwargs = (
                DEFAULT_ASSEMBLY_EXECUTION_POLICY.executor_kwargs()
            )
            if course_step.morphology == "rc_car8":
                # RC Car8's first topology wave shares chassis parents.
                # The Isaac primitive arbiter reserves those parent resources
                # during REACH, so dispatch it serially within the wave.
                assembly_kwargs["max_concurrent_alignments_per_wave"] = 1
            self._engine = ParallelAssemblyExecutor(
                result.assembly_plan,
                execution_id=execution_id,
                enable_borrowed_helper=True,
                additional_known_module_ids=result.reserve_module_ids,
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
                **assembly_kwargs,
            )
            self._source_graph = self._active_target
            self._source_assignment = result.assignment
            return
        if course_step.navigation is not None:
            self._engine = _NavigationEngine(course_step.navigation)
            return
        if course_step.behavior is None:
            plan = self._reconfiguration_planner.plan(
                current_graph,
                self._active_target,
                source_graph=self._source_graph,
                source_assignment=self._source_assignment,
            )
            self._assignment = dict(plan.assignment.target_to_module)
            self._engine = SelfReconfigurationExecutor(plan, execution_id=execution_id)
            self._source_graph = self._active_target
            self._source_assignment = plan.assignment
            return
        roles = target_roles_from_graph(self._active_target)
        assignments = tuple(
            AssignedModule(module_id=module_id, target_vertex_id=vertex, target_role=str(roles[vertex]["target_role"]))
            for vertex, module_id in sorted(self._assignment.items())
        )
        executor = MorphologyBehaviorExecutor(self._behavior_library)
        neutral_tilts: Mapping[str, float] = {}
        if self._behavior_library.uses_captured_neutral(
            course_step.morphology
        ):
            assignment_signature = (
                course_step.morphology,
                tuple(
                    (item.target_vertex_id, item.module_id)
                    for item in assignments
                ),
            )
            if self._neutral_assignment_signature != assignment_signature:
                latest_tilts = logical_tilt_positions(
                    self._latest_observation
                )
                missing = sorted(
                    item.module_id
                    for item in assignments
                    if item.module_id not in latest_tilts
                )
                if missing:
                    raise ValueError(
                        "Cannot capture the assembled neutral TILT posture; "
                        f"missing module states for {missing}"
                    )
                self._neutral_tilt_rad_by_module = {
                    item.module_id: latest_tilts[item.module_id]
                    for item in assignments
                }
                self._neutral_assignment_signature = assignment_signature
            neutral_tilts = self._neutral_tilt_rad_by_module
        program_override = None
        if course_step.behavior in {
            "crawl_stairs",
            "crawl_stairs_arch_wave",
            "gap_crossing",
        }:
            if course_step.behavior == "gap_crossing":
                planner = self._gap_gait_planner.plan
            else:
                planner = (
                    self._stair_gait_planner.plan_arch_wave
                    if course_step.behavior == "crawl_stairs_arch_wave"
                    else self._stair_gait_planner.plan
                )
            program_override = planner(
                current_graph,
                assignments,
                course_step.parameters or {},
            )
        executor.start(
            MorphologyCommand(
                execution_id,
                course_step.morphology,
                course_step.behavior,
                course_step.parameters or {},
            ),
            assignments,
            neutral_tilts,
            program_override,
        )
        self._engine = (executor, assignments)

    @staticmethod
    def _post_assembly_tilts(result: Any) -> dict[str, float]:
        raw = result.target_graph.global_attributes.get(
            "post_assembly_tilt_rad_by_vertex", {}
        )
        if not isinstance(raw, Mapping):
            return {}
        return {
            str(result.assignment.target_to_module[vertex]): float(angle)
            for vertex, angle in raw.items()
        }

    @staticmethod
    def _post_assembly_pans(result: Any) -> dict[str, float]:
        raw = result.target_graph.global_attributes.get(
            "post_assembly_pan_rad_by_vertex", {}
        )
        if not isinstance(raw, Mapping):
            return {}
        return {
            str(result.assignment.target_to_module[vertex]): float(angle)
            for vertex, angle in raw.items()
        }

    def _run_engine(self, course_step: CourseStep, current_graph: Any) -> tuple[Any, ExpertOutput]:
        if isinstance(self._engine, _NavigationEngine):
            decision = self._run_navigation(course_step, current_graph)
            self._publish_actions(course_step, decision.locomotion, decision)
            output = ExpertOutput(
                fsm_state=decision.state,
                locomotion=decision.locomotion,
                success=decision.success,
                done=decision.done,
                debug={"phase": decision.phase, "message": decision.message},
            )
            return decision, output
        if isinstance(self._engine, ParallelAssemblyExecutor):
            decision = self._engine.step(self._latest_status)
            if decision.primitive_goal is not None:
                self._goal_publisher.publish(dict_to_string_msg(decision.primitive_goal.to_dict()))
            output = ExpertOutput(fsm_state=decision.state, primitive_goal=decision.primitive_goal_payload, success=decision.success, done=decision.done)
            return decision, output
        if isinstance(self._engine, SelfReconfigurationExecutor):
            decision = self._engine.step(self._latest_status, current_graph=current_graph)
            if decision.primitive_goal is not None:
                self._goal_publisher.publish(dict_to_string_msg(decision.primitive_goal.to_dict()))
            output = ExpertOutput(fsm_state=decision.state, primitive_goal=decision.primitive_goal_payload, success=decision.success, done=decision.done)
            return decision, output
        executor, assignments = self._engine
        nodes = current_graph.node_by_id()
        module_positions = {
            assignment.module_id: module_position(
                nodes[assignment.module_id].attributes
            )
            for assignment in assignments
            if assignment.module_id in nodes
        }
        decision = executor.step(
            time.monotonic(),
            self._latest_status,
            module_positions,
        )
        if decision.primitive_goal is not None:
            self._goal_publisher.publish(dict_to_string_msg(decision.primitive_goal.to_dict()))
        locomotion = coherent_planar_train_commands(current_graph, decision.locomotion)
        self._publish_actions(course_step, locomotion, decision)
        output = ExpertOutput(fsm_state=decision.state, locomotion=locomotion, primitive_goal=(decision.primitive_goal.to_dict() if decision.primitive_goal else None), success=decision.success, done=decision.done)
        return decision, output

    def _publish_actions(
        self,
        course_step: CourseStep,
        locomotion: Mapping[str, Mapping[str, float]],
        decision: Any,
    ) -> None:
        self._actions_publisher.publish(dict_to_string_msg({
            "schema_version": "mssr.actions.v2", "stamp": time.time(), "stage_id": self._step_index,
            "task_type": course_step.task, "reset": False, "locomotion": locomotion, "magnetic": [],
            "expert": {"fsm_state": decision.state, "success": decision.success, "done": decision.done},
        }))

    def _run_navigation(
        self,
        course_step: CourseStep,
        current_graph: Any,
    ) -> _NavigationDecision:
        if not isinstance(self._engine, _NavigationEngine):
            raise RuntimeError("Navigation step has no navigation engine.")
        try:
            landmarks = CourseLandmarks.from_observation(self._latest_observation)
            modules = extract_modules(self._latest_observation)
            if not modules:
                raise CourseLandmarkError("Isaac has not published module poses.")
            mode = self._engine.mode
            if mode == "ramp_exit":
                return self._navigate_x(
                    current_graph,
                    modules,
                    landmarks.ramp_exit_x_m,
                    use_rear=True,
                    phase="CLIMB_RAMP",
                )
            if mode == "safe_before_snake_reconfiguration":
                safe_x_m = (
                    landmarks.gap_near_x_m
                    - float(
                        self.get_parameter(
                            "gap_approach_margin_m"
                        ).value
                    )
                )
                front_x_m = max(
                    module_position(module)[0]
                    for module in modules.values()
                )
                if front_x_m <= safe_x_m:
                    return _NavigationDecision(
                        "SUCCEEDED",
                        "SAFE_RECONFIGURATION",
                        {},
                        True,
                        True,
                        "RC Car8 is entirely on the near platform.",
                    )
                return _NavigationDecision(
                    "WAITING_SAFE_RECONFIGURATION",
                    "SAFE_RECONFIGURATION",
                    {},
                    message=(
                        "RC Car8 extends beyond the safe reconfiguration "
                        f"limit x={safe_x_m:.3f}m."
                    ),
                )
            if mode == "safe_before_bridge_reconfiguration":
                safe_x_m = (
                    landmarks.gap_near_x_m
                    - float(
                        self.get_parameter(
                            "gap_approach_margin_m"
                        ).value
                    )
                )
                front_x_m = max(
                    module_position(module)[0]
                    for module in modules.values()
                )
                if front_x_m <= safe_x_m:
                    return _NavigationDecision(
                        "SUCCEEDED",
                        "SAFE_RECONFIGURATION",
                        {},
                        True,
                        True,
                        "Entire Snake8 is on the start platform.",
                    )
                return _NavigationDecision(
                    "WAITING_SAFE_RECONFIGURATION",
                    "SAFE_RECONFIGURATION",
                    {},
                    message=(
                        "Snake8 extends beyond the safe reconfiguration "
                        f"limit x={safe_x_m:.3f}m."
                    ),
                )
            if mode == "front_before_gap":
                target_x = landmarks.gap_near_x_m - float(self.get_parameter("gap_approach_margin_m").value)
                return self._navigate_x(current_graph, modules, target_x, use_rear=False, phase="APPROACH_GAP")
            if mode == "rear_past_gap":
                target_x = landmarks.gap_far_x_m + float(self.get_parameter("gap_clearance_margin_m").value)
                return self._navigate_x(current_graph, modules, target_x, use_rear=True, phase="CLEAR_GAP")
            if mode.startswith("front_before_stair_"):
                stair_index = int(mode.rsplit("_", 1)[1]) - 1
                target_x = self._stair_target_x(landmarks, stair_index)
                return self._navigate_stair(current_graph, modules, landmarks, stair_index, target_x)
            if mode == "front_on_upper_deck":
                stair_index = len(landmarks.stair_top_heights_m) - 1
                target_x = self._stair_target_x(landmarks, stair_index + 1)
                return self._navigate_stair(current_graph, modules, landmarks, stair_index, target_x)
            if mode == "button_standoff":
                x_m, y_m, _ = landmarks.button_center_xyz_m
                return self._navigate_pose(
                    current_graph,
                    modules,
                    (x_m, y_m - 0.20),
                    math.pi / 2.0,
                    "BUTTON_STANDOFF",
                    position_tolerance_m=float(
                        self.get_parameter(
                            "button_navigation_tolerance_m"
                        ).value
                    ),
                )
            if mode == "button_retreat":
                x_m, y_m, _ = landmarks.button_center_xyz_m
                return self._navigate_pose(current_graph, modules, (x_m, y_m - 0.40), -math.pi / 2.0, "BUTTON_RETREAT")
            if mode == "cross_exit":
                exit_x_m, exit_y_m, _ = landmarks.exit_center_xyz_m
                if self._goal_reached():
                    return _NavigationDecision("SUCCEEDED", "EXIT", {}, True, True, "Exit plane crossed.")
                return self._navigate_pose(current_graph, modules, (exit_x_m + 0.10, exit_y_m), 0.0, "CROSS_EXIT")
            raise CourseLandmarkError(f"Unknown navigation mode {mode!r}.")
        except (CourseLandmarkError, ValueError) as error:
            return _NavigationDecision("WAITING_COURSE_GEOMETRY", "WAIT", {}, message=str(error))

    def _navigate_x(
        self,
        current_graph: Any,
        modules: Mapping[str, Mapping[str, Any]],
        target_x_m: float,
        *,
        use_rear: bool,
        phase: str,
    ) -> _NavigationDecision:
        positions = [module_position(module) for module in modules.values()]
        progress_x_m = min(position[0] for position in positions) if use_rear else max(position[0] for position in positions)
        if progress_x_m >= target_x_m:
            return _NavigationDecision("SUCCEEDED", phase, {}, True, True, f"Reached x={progress_x_m:.3f}m.")
        locomotion = self._heading_locomotion(current_graph, modules, 0.0)
        return _NavigationDecision("RUNNING_NAVIGATION", phase, locomotion, message=f"x={progress_x_m:.3f}m, target={target_x_m:.3f}m.")

    def _navigate_stair(
        self,
        current_graph: Any,
        modules: Mapping[str, Mapping[str, Any]],
        landmarks: CourseLandmarks,
        stair_index: int,
        target_x_m: float,
    ) -> _NavigationDecision:
        positions = [module_position(module) for module in modules.values()]
        front_x_m = max(position[0] for position in positions)
        required_height_m = landmarks.stair_top_heights_m[stair_index]
        maximum_height_m = max(position[2] for position in positions)
        if front_x_m >= target_x_m and maximum_height_m >= required_height_m - 0.02:
            return _NavigationDecision("SUCCEEDED", "STAIR_PROGRESS", {}, True, True, f"Reached stair {stair_index + 1}.")
        if front_x_m >= target_x_m:
            return _NavigationDecision("WAITING_STAIR_SUPPORT", "STAIR_PROGRESS", {}, message=f"Need height {required_height_m:.3f}m; observed {maximum_height_m:.3f}m.")
        locomotion = self._heading_locomotion(current_graph, modules, 0.0)
        return _NavigationDecision("RUNNING_NAVIGATION", "STAIR_PROGRESS", locomotion, message=f"front={front_x_m:.3f}m, target={target_x_m:.3f}m.")

    def _navigate_pose(
        self,
        current_graph: Any,
        modules: Mapping[str, Mapping[str, Any]],
        target_xy_m: tuple[float, float],
        final_yaw_rad: float,
        phase: str,
        position_tolerance_m: float | None = None,
    ) -> _NavigationDecision:
        root_position, root_yaw_rad = self._root_pose(modules)
        delta_x_m = target_xy_m[0] - root_position[0]
        delta_y_m = target_xy_m[1] - root_position[1]
        distance_m = math.hypot(delta_x_m, delta_y_m)
        tolerance_m = (
            float(self.get_parameter("navigation_position_tolerance_m").value)
            if position_tolerance_m is None
            else float(position_tolerance_m)
        )
        heading_rad = math.atan2(delta_y_m, delta_x_m) if distance_m > tolerance_m else final_yaw_rad
        heading_error_rad = _wrap_angle(heading_rad - root_yaw_rad)
        if distance_m <= tolerance_m and abs(heading_error_rad) <= 0.15:
            return _NavigationDecision("SUCCEEDED", phase, {}, True, True, "Reached world-frame approach pose.")
        locomotion = self._drive_locomotion(
            current_graph,
            0.0 if abs(heading_error_rad) > 0.15 else float(self.get_parameter("navigation_speed_m_s").value),
            math.copysign(float(self.get_parameter("navigation_yaw_rate_rad_s").value), heading_error_rad) if abs(heading_error_rad) > 0.15 else 0.0,
        )
        return _NavigationDecision("RUNNING_NAVIGATION", phase, locomotion, message=f"distance={distance_m:.3f}m, heading_error={heading_error_rad:.3f}rad.")

    def _heading_locomotion(
        self,
        current_graph: Any,
        modules: Mapping[str, Mapping[str, Any]],
        target_yaw_rad: float,
    ) -> dict[str, dict[str, float]]:
        _, root_yaw_rad = self._root_pose(modules)
        heading_error_rad = _wrap_angle(target_yaw_rad - root_yaw_rad)
        return self._drive_locomotion(
            current_graph,
            0.0 if abs(heading_error_rad) > 0.15 else float(self.get_parameter("navigation_speed_m_s").value),
            math.copysign(float(self.get_parameter("navigation_yaw_rate_rad_s").value), heading_error_rad) if abs(heading_error_rad) > 0.15 else 0.0,
        )

    def _drive_locomotion(
        self,
        current_graph: Any,
        linear_m_s: float,
        yaw_rate_rad_s: float,
    ) -> dict[str, dict[str, float]]:
        if self._active_target is None:
            return {}
        morphology = str(self._active_target.global_attributes["morphology_name"])
        roles = target_roles_from_graph(self._active_target)
        assignments = tuple(
            AssignedModule(module_id=module_id, target_vertex_id=vertex, target_role=str(roles[vertex]["target_role"]))
            for vertex, module_id in sorted(self._assignment.items())
        )
        commands = self._behavior_library.drive_commands(morphology, assignments, linear_m_s, yaw_rate_rad_s)
        return coherent_planar_train_commands(current_graph, commands)

    def _root_pose(self, modules: Mapping[str, Mapping[str, Any]]) -> tuple[tuple[float, float, float], float]:
        if self._active_target is None:
            raise CourseLandmarkError("No active target morphology.")
        roles = target_roles_from_graph(self._active_target)
        root_vertex = next(vertex for vertex, role in roles.items() if role["is_target_root"])
        root_module_id = self._assignment.get(root_vertex)
        root_module = modules.get(root_module_id or "")
        if root_module is None:
            raise CourseLandmarkError("Assigned root module pose is unavailable.")
        pose = root_module.get("pose", {})
        orientation = pose.get("orientation_xyzw", pose.get("orientation")) if isinstance(pose, Mapping) else None
        if not isinstance(orientation, list | tuple) or len(orientation) != 4:
            raise CourseLandmarkError("Root orientation is unavailable.")
        x, y, z, w = (float(value) for value in orientation)
        yaw_rad = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return module_position(root_module), yaw_rad

    def _stair_target_x(self, landmarks: CourseLandmarks, index: int) -> float:
        return landmarks.first_riser_x_m + index * landmarks.riser_depth_m - float(self.get_parameter("stair_approach_margin_m").value)

    def _advance(self, course_step: CourseStep) -> None:
        self.get_logger().info(f"Completed course step {course_step.task} with {course_step.morphology}.")
        self._engine = None
        self._awaiting_completion_event = False
        self._step_index += 1

    def _button_pressed(self) -> bool:
        if self._active_target is None:
            return False
        try:
            center = CourseLandmarks.from_observation(
                self._latest_observation
            ).button_center_xyz_m
        except CourseLandmarkError:
            return False
        roles = target_roles_from_graph(self._active_target)
        end_effector = next((module_id for vertex, module_id in self._assignment.items() if roles[vertex]["target_role"] == "end_effector"), None)
        if end_effector is None:
            return False
        module = extract_modules(self._latest_observation).get(end_effector)
        if module is None:
            return False
        return button_contact_distance_m(module, center) <= float(
            self.get_parameter("button_contact_radius_m").value
        )

    def _approach_button_contact(
        self,
        current_graph: Any,
    ) -> _NavigationDecision:
        """Creep with the pressed posture held until the free face contacts."""

        try:
            center = CourseLandmarks.from_observation(
                self._latest_observation
            ).button_center_xyz_m
        except CourseLandmarkError as error:
            return _NavigationDecision(
                "WAITING_COURSE_GEOMETRY",
                "BUTTON_CONTACT",
                {},
                message=str(error),
            )
        if self._active_target is None:
            return _NavigationDecision(
                "WAITING_BUTTON_ASSIGNMENT",
                "BUTTON_CONTACT",
                {},
                message="Mobile-manipulator target assignment is unavailable.",
            )
        roles = target_roles_from_graph(self._active_target)
        end_effector = next(
            (
                module_id
                for vertex, module_id in self._assignment.items()
                if roles[vertex]["target_role"] == "end_effector"
            ),
            None,
        )
        module = extract_modules(self._latest_observation).get(
            end_effector or ""
        )
        if module is None:
            return _NavigationDecision(
                "WAITING_BUTTON_POSE",
                "BUTTON_CONTACT",
                {},
                message="End-effector world pose is unavailable.",
            )
        contact = end_effector_contact_point(module)
        distance_m = distance_3d(contact, center)
        alignment_error_m = math.hypot(
            contact[0] - center[0],
            contact[2] - center[2],
        )
        alignment_tolerance_m = float(
            self.get_parameter("button_alignment_tolerance_m").value
        )
        if alignment_error_m > alignment_tolerance_m:
            return _NavigationDecision(
                "WAITING_BUTTON_ALIGNMENT",
                "BUTTON_CONTACT",
                {},
                message=(
                    "End-effector cannot safely creep toward the plunger: "
                    f"cross-axis error={alignment_error_m:.3f}m."
                ),
            )
        locomotion = self._drive_locomotion(
            current_graph,
            float(self.get_parameter("button_contact_speed_m_s").value),
            0.0,
        )
        return _NavigationDecision(
            "RUNNING_BUTTON_CONTACT",
            "BUTTON_CONTACT",
            locomotion,
            message=(
                f"End-effector face distance={distance_m:.3f}m; "
                "creeping toward the button."
            ),
        )

    def _goal_reached(self) -> bool:
        try:
            exit_x = CourseLandmarks.from_observation(
                self._latest_observation
            ).exit_center_xyz_m[0]
        except CourseLandmarkError:
            return False
        count = sum(1 for module in extract_modules(self._latest_observation).values() if module_position(module)[0] >= exit_x)
        return count >= int(self.get_parameter("goal_min_modules_past_exit").value)

    def _publish_state(self, course_step: CourseStep, state: str, done: bool, success: bool, message: str) -> None:
        self._state_publisher.publish(dict_to_string_msg({
            "schema_version": "mssr.obstacle_course_state.v1", "stamp": time.time(),
            "episode_id": str(self.get_parameter("episode_id").value), "course_step": course_step.task,
            "morphology": course_step.morphology, "state": state, "done": done,
            "success": success, "message": message,
        }))

    def _finish(self, success: bool, current_graph: Any, message: str) -> None:
        if success:
            for (
                graph,
                task_graph,
                output,
                task,
                timestep,
                target_graph,
                assignment,
                observation,
            ) in self._pending_records:
                self._dataset_logger.log_step(
                    episode_id=str(self.get_parameter("episode_id").value), timestep=timestep,
                    observation=observation, graph=graph, expert_output=output,
                    stage_name="smores_obstacle_course", stage_id=timestep, task_type=task,
                    difficulty=1.0, task_graph=task_graph, target_graph=target_graph,
                    assignment=assignment,
                )
            self.get_logger().info(f"{message} Wrote {len(self._pending_records)} IL transitions.")
        else:
            self.get_logger().error(f"Obstacle course failed; discarded {len(self._pending_records)} incomplete transitions: {message}")
        self._terminal = True


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SmoresObstacleCourseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _wrap_angle(angle_rad: float) -> float:
    """Normalize one heading error to the closed interval [-pi, pi]."""
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))
