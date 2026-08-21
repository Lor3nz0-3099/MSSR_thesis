"""ROS 2 node that operates an assembled SMORES-EP morphology."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from std_msgs.msg import String

from mssr_expert.behaviors.morphology_library import (
    AssignedModule,
    MorphologyLibrary,
)
from mssr_expert.behaviors.morphology_navigation import (
    estimate_planar_morphology_state,
)
from mssr_expert.behaviors.morphology_locomotion import (
    coherent_planar_train_commands,
    validate_locomotion_dofs,
)
from mssr_expert.behaviors.morphology_dof_model import (
    MorphologyDofInventory,
    SmoresMorphologyDofAnalyzer,
)
from mssr_expert.execution.morphology_behavior_executor import (
    MorphologyBehaviorExecutor,
    MorphologyCommand,
)
from mssr_expert.graph.attributed_robot_graph import AttributedRobotGraph
from mssr_expert.graph.serialization import (
    attributed_graph_from_dict,
    load_attributed_graph,
)
from mssr_expert.planning.smores_ep.attributed_adapter import (
    target_roles_from_graph,
)
from mssr_expert.planning.smores_ep.self_reconfiguration_planner import (
    SmoresSelfReconfigurationPlanner,
)
from mssr_expert.primitives.common import logical_tilt_positions
from mssr_expert.utils.json_io import dict_to_string_msg, string_msg_to_dict


def load_behavior_morphology_catalog(
    config_directory: Path,
) -> dict[str, AttributedRobotGraph]:
    """Load named target graphs without importing another ROS node."""

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
    return catalog


def assembly_readiness(
    global_attributes: Mapping[str, Any],
) -> tuple[bool, str]:
    """Return whether the latest self-assembly task graph is operational."""

    execution_state = global_attributes.get("execution_state", {})
    if not isinstance(execution_state, Mapping):
        return False, "UNKNOWN"
    state = str(execution_state.get("state", "UNKNOWN"))
    ready = bool(
        execution_state.get("done", False)
        and execution_state.get("success", False)
    )
    return ready, state


class SmoresMorphologyBehaviorNode(Node):
    """Map high-level morphology commands to posture and cluster motion."""

    def __init__(self) -> None:
        super().__init__("smores_morphology_behavior_node")
        package_share = Path(get_package_share_directory("mssr_expert"))
        default_library = (
            package_share / "config" / "smores_morphology_behaviors.json"
        )
        self.declare_parameter("library_path", str(default_library))
        self.declare_parameter("control_rate_hz", 10.0)
        self.declare_parameter("joint_timeout_s", 20.0)
        self.declare_parameter(
            "command_topic", "/mssr/morphology/command"
        )
        self.declare_parameter(
            "status_topic", "/mssr/morphology/status"
        )
        self.declare_parameter("actions_topic", "/mssr/actions")
        self.declare_parameter(
            "primitive_goal_topic", "/mssr/primitives/goal"
        )
        self.declare_parameter(
            "primitive_status_topic", "/mssr/primitives/status"
        )
        self.declare_parameter(
            "module_states_topic", "/mssr/module_states"
        )
        self.declare_parameter(
            "task_graph_topic", "/mssr/expert/task_graph"
        )
        self.declare_parameter("robot_graph_topic", "/mssr/robot_graph")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("cmd_vel_timeout_s", 0.5)
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")

        library = MorphologyLibrary.load(
            Path(str(self.get_parameter("library_path").value))
        )
        self._library = library
        self._executor = MorphologyBehaviorExecutor(
            library,
            joint_timeout_s=float(
                self.get_parameter("joint_timeout_s").value
            ),
        )
        self._morphology_catalog = load_behavior_morphology_catalog(
            package_share / "config"
        )
        self._topology_matcher = SmoresSelfReconfigurationPlanner()
        self._morphology_name = ""
        self._assignments: tuple[AssignedModule, ...] = ()
        self._assembly_ready = False
        self._assembly_state = "UNASSIGNED"
        self._assignment_source = ""
        self._latest_primitive_status: dict[str, Any] = {}
        self._latest_robot_graph = AttributedRobotGraph()
        self._dof_analyzer = SmoresMorphologyDofAnalyzer()
        self._dof_inventory = MorphologyDofInventory(())
        self._dof_signature: tuple[tuple[str, str, str], ...] = ()
        self._last_terminal_command_id = ""
        self._latest_cmd_vel = (0.0, 0.0, 0.0)
        self._last_cmd_vel_s: float | None = None
        self._cmd_vel_output_active = False
        self._latest_tilt_rad_by_module: dict[str, float] = {}
        self._neutral_tilt_rad_by_module: dict[str, float] = {}
        self._neutral_assignment_signature: tuple[
            str, tuple[tuple[str, str], ...]
        ] | None = None

        self._odom_publisher = self.create_publisher(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            10,
        )
        self._tf_broadcaster = TransformBroadcaster(self)

        self._actions_publisher = self.create_publisher(
            String,
            str(self.get_parameter("actions_topic").value),
            10,
        )
        self._goal_publisher = self.create_publisher(
            String,
            str(self.get_parameter("primitive_goal_topic").value),
            10,
        )
        self._status_publisher = self.create_publisher(
            String,
            str(self.get_parameter("status_topic").value),
            10,
        )
        task_graph_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("task_graph_topic").value),
            self._on_task_graph,
            task_graph_qos,
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
        self.create_subscription(
            String,
            str(self.get_parameter("module_states_topic").value),
            self._on_module_states,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("command_topic").value),
            self._on_command,
            10,
        )
        self.create_subscription(
            Twist,
            str(self.get_parameter("cmd_vel_topic").value),
            self._on_cmd_vel,
            10,
        )
        rate_hz = max(
            1.0, float(self.get_parameter("control_rate_hz").value)
        )
        self._timer = self.create_timer(1.0 / rate_hz, self._step)
        self.get_logger().info(
            "SMORES morphology behavior node ready; waiting for assignment."
        )

    def _on_task_graph(self, message: String) -> None:
        payload = string_msg_to_dict(message)
        global_attributes = payload.get("global_attributes", {})
        if isinstance(global_attributes, Mapping):
            candidate_morphology = str(
                global_attributes.get("target_morphology_name", "")
            )
            if (
                self._assignment_source == "robot_graph"
                and candidate_morphology
                and candidate_morphology != self._morphology_name
            ):
                # A retained task graph for another morphology is not proof
                # that the live rigid topology has changed.  The robot-graph
                # callback will adopt the new assignment as soon as that
                # topology uniquely exists.
                return
            self._morphology_name = candidate_morphology
            (
                self._assembly_ready,
                self._assembly_state,
            ) = assembly_readiness(global_attributes)
        assignments: list[AssignedModule] = []
        raw_nodes = payload.get("nodes", ())
        if isinstance(raw_nodes, list | tuple):
            for raw_node in raw_nodes:
                if not isinstance(raw_node, Mapping):
                    continue
                attributes = raw_node.get("attributes", {})
                if not isinstance(attributes, Mapping):
                    continue
                if str(attributes.get("node_type", "")) != "physical_module":
                    continue
                vertex = attributes.get("target_vertex_id")
                role = attributes.get("target_role")
                module_id = raw_node.get("module_id") or raw_node.get("node_id")
                if module_id is None or vertex is None or role is None:
                    continue
                assignments.append(
                    AssignedModule(
                        module_id=str(module_id),
                        target_vertex_id=str(vertex),
                        target_role=str(role),
                    )
                )
        self._assignments = tuple(
            sorted(assignments, key=lambda item: item.target_vertex_id)
        )
        if self._morphology_name and self._assignments:
            self._assignment_source = "task_graph"

    def _on_robot_graph(self, message: String) -> None:
        """Recover an assignment after the producing expert was restarted."""

        payload = string_msg_to_dict(message)
        try:
            current_graph = attributed_graph_from_dict(payload)
        except (KeyError, RuntimeError, TypeError, ValueError):
            return
        self._latest_robot_graph = current_graph
        self._update_dof_inventory(current_graph)

        # Re-evaluate the physical topology even when a task graph supplied
        # the last assignment.  Several expert publishers can have retained
        # task-graph samples on the same transient-local topic; an older
        # self-assembly sample must not overwrite a completed
        # self-reconfiguration.  The live rigid topology is the final
        # authority once it uniquely matches one catalog morphology.
        try:
            matches = []
            for morphology_name, target_graph in sorted(
                self._morphology_catalog.items()
            ):
                assignment = self._topology_matcher.configuration_assignment(
                    current_graph, target_graph
                )
                if assignment is not None:
                    matches.append(
                        (morphology_name, target_graph, assignment)
                    )
        except (KeyError, RuntimeError, TypeError, ValueError):
            return
        if len(matches) != 1:
            if self._assignment_source == "robot_graph":
                self._assembly_ready = False
                self._assembly_state = "TOPOLOGY_TRANSITION"
            return
        morphology_name, target_graph, assignment = matches[0]
        roles = target_roles_from_graph(target_graph)
        recovered_assignments = tuple(
            sorted(
                (
                    AssignedModule(
                        module_id=module_id,
                        target_vertex_id=target_vertex_id,
                        target_role=str(
                            roles[target_vertex_id]["target_role"]
                        ),
                    )
                    for target_vertex_id, module_id in (
                        assignment.target_to_module.items()
                    )
                ),
                key=lambda item: item.target_vertex_id,
            )
        )
        unchanged = (
            self._morphology_name == morphology_name
            and self._assignments == recovered_assignments
            and self._assembly_ready
        )
        self._morphology_name = morphology_name
        self._assignments = recovered_assignments
        self._assembly_ready = True
        self._assembly_state = "TOPOLOGY_MATCHED"
        self._assignment_source = "robot_graph"
        if not unchanged:
            self.get_logger().info(
                "Recovered unique live topology assignment for "
                f"{morphology_name!r}."
            )
        self._publish_navigation_state(current_graph)

    def _publish_navigation_state(
        self,
        current_graph: AttributedRobotGraph,
    ) -> None:
        """Expose the assembled morphology as one planar Nav2 base."""

        try:
            navigation_spec = self._library.navigation_frame_spec(
                self._morphology_name
            )
            if navigation_spec is None:
                return
            state = estimate_planar_morphology_state(
                current_graph,
                self._assignments,
                navigation_spec,
            )
        except ValueError as error:
            self.get_logger().warning(
                f"Cannot publish morphology odometry: {error}"
            )
            return

        stamp = self.get_clock().now().to_msg()
        odom_frame = str(self.get_parameter("odom_frame").value)
        base_frame = str(self.get_parameter("base_frame").value)
        half_yaw = 0.5 * state.yaw_rad
        qz = math.sin(half_yaw)
        qw = math.cos(half_yaw)

        odometry = Odometry()
        odometry.header.stamp = stamp
        odometry.header.frame_id = odom_frame
        odometry.child_frame_id = base_frame
        odometry.pose.pose.position.x = state.x_m
        odometry.pose.pose.position.y = state.y_m
        odometry.pose.pose.position.z = 0.0
        odometry.pose.pose.orientation.z = qz
        odometry.pose.pose.orientation.w = qw
        odometry.twist.twist.linear.x = state.vx_m_s
        odometry.twist.twist.linear.y = state.vy_m_s
        odometry.twist.twist.angular.z = state.yaw_rate_rad_s
        self._odom_publisher.publish(odometry)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = odom_frame
        transform.child_frame_id = base_frame
        transform.transform.translation.x = state.x_m
        transform.transform.translation.y = state.y_m
        transform.transform.translation.z = 0.0
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self._tf_broadcaster.sendTransform(transform)

    def _update_dof_inventory(
        self,
        current_graph: AttributedRobotGraph,
    ) -> None:
        inventory = self._dof_analyzer.analyze(current_graph)
        self._dof_inventory = inventory
        if inventory.signature == self._dof_signature:
            return
        self._dof_signature = inventory.signature
        self.get_logger().info(
            "Operational DoFs: "
            f"load_bearing={len(inventory.by_mode('load_bearing'))}, "
            "locomotion_candidates="
            f"{len(inventory.by_mode('locomotion_candidate'))}, "
            f"shape_candidates={len(inventory.by_mode('shape_candidate'))}."
        )

    def _on_cmd_vel(self, message: Twist) -> None:
        values = (
            float(message.linear.x),
            float(message.linear.y),
            float(message.angular.z),
        )
        if not all(math.isfinite(value) for value in values):
            self.get_logger().error("Ignoring non-finite /cmd_vel command.")
            return
        self._latest_cmd_vel = values
        self._last_cmd_vel_s = time.monotonic()

    def _on_primitive_status(self, message: String) -> None:
        payload = string_msg_to_dict(message)
        if payload:
            self._latest_primitive_status = payload

    def _on_module_states(self, message: String) -> None:
        positions = logical_tilt_positions(string_msg_to_dict(message))
        if positions:
            self._latest_tilt_rad_by_module = positions

    def _on_command(self, message: String) -> None:
        payload = string_msg_to_dict(message)
        try:
            command = MorphologyCommand.from_mapping(payload)
            if not self._morphology_name or not self._assignments:
                raise ValueError("No self-assembly assignment is available")
            if not self._assembly_ready:
                raise ValueError(
                    "Self-assembly is not complete; current state is "
                    f"{self._assembly_state}"
                )
            if command.morphology != self._morphology_name:
                raise ValueError(
                    f"Command requests {command.morphology}, but the assembled "
                    f"target is {self._morphology_name}"
                )
            if self._executor.active and command.behavior != "stop":
                raise ValueError("Another morphology behavior is active")
            neutral_tilts: Mapping[str, float] = {}
            if (
                command.behavior != "stop"
                and self._library.uses_captured_neutral(
                    self._morphology_name
                )
            ):
                assignment_signature = (
                    self._morphology_name,
                    tuple(
                        (item.target_vertex_id, item.module_id)
                        for item in self._assignments
                    ),
                )
                if (
                    self._neutral_assignment_signature
                    != assignment_signature
                ):
                    missing = sorted(
                        item.module_id
                        for item in self._assignments
                        if item.module_id
                        not in self._latest_tilt_rad_by_module
                    )
                    if missing:
                        raise ValueError(
                            "Cannot capture the assembled neutral TILT "
                            "posture; missing module states for "
                            f"{missing}"
                        )
                    self._neutral_tilt_rad_by_module = {
                        item.module_id: self._latest_tilt_rad_by_module[
                            item.module_id
                        ]
                        for item in self._assignments
                    }
                    self._neutral_assignment_signature = (
                        assignment_signature
                    )
                    self.get_logger().info(
                        "Captured assembled neutral TILT posture for "
                        f"{self._morphology_name!r}."
                    )
                neutral_tilts = self._neutral_tilt_rad_by_module
            self._executor.start(
                command,
                self._assignments,
                neutral_tilts,
            )
            self._last_terminal_command_id = ""
            self._publish_status(
                command.command_id,
                command.morphology,
                command.behavior,
                "ACCEPTED",
                "QUEUED",
                0.0,
                False,
                False,
                "Morphology command accepted.",
            )
        except (KeyError, TypeError, ValueError) as error:
            self.get_logger().error(f"Rejected morphology command: {error}")
            self._publish_status(
                str(payload.get("command_id", "")),
                str(payload.get("morphology", "")),
                str(payload.get("behavior", "")),
                "REJECTED",
                "TERMINAL",
                0.0,
                True,
                False,
                str(error),
            )

    def _step(self) -> None:
        if not self._executor.active and self._step_cmd_vel():
            return

        decision = self._executor.step(
            time.monotonic(), self._latest_primitive_status
        )
        if not decision.command_id:
            return
        if decision.primitive_goal is not None:
            self._goal_publisher.publish(
                dict_to_string_msg(decision.primitive_goal.to_dict())
            )
        locomotion = coherent_planar_train_commands(
            self._latest_robot_graph,
            decision.locomotion,
        )
        self._publish_actions(
            locomotion,
            fsm_state=decision.state,
            active_primitive=(
                decision.primitive_goal.primitive
                if decision.primitive_goal is not None
                else decision.behavior
            ),
            progress=decision.progress,
            success=decision.success,
            done=decision.done,
            message=decision.message,
            task_type="morphology_behavior",
        )
        if (
            decision.done
            and decision.command_id == self._last_terminal_command_id
        ):
            return
        self._publish_status(
            decision.command_id,
            decision.morphology,
            decision.behavior,
            decision.state,
            decision.phase,
            decision.progress,
            decision.done,
            decision.success,
            decision.message,
        )
        if decision.done:
            self._last_terminal_command_id = decision.command_id

    def _step_cmd_vel(self) -> bool:
        """Forward a fresh Nav2-style body twist through the morphology map."""

        now_s = time.monotonic()
        timeout_s = float(self.get_parameter("cmd_vel_timeout_s").value)
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            timeout_s = 0.5
        fresh = (
            self._last_cmd_vel_s is not None
            and now_s - self._last_cmd_vel_s <= timeout_s
        )
        if not fresh and not self._cmd_vel_output_active:
            return False
        if (
            not self._morphology_name
            or not self._assignments
            or not self._assembly_ready
        ):
            return False

        linear, lateral, yaw_rate = (
            self._latest_cmd_vel if fresh else (0.0, 0.0, 0.0)
        )
        try:
            locomotion = self._library.drive_commands(
                self._morphology_name,
                self._assignments,
                linear,
                yaw_rate,
                lateral,
            )
            locomotion = coherent_planar_train_commands(
                self._latest_robot_graph, locomotion
            )
            validate_locomotion_dofs(locomotion, self._dof_inventory)
        except ValueError as error:
            self.get_logger().error(
                f"Rejected /cmd_vel for current morphology: {error}"
            )
            if not self._cmd_vel_output_active:
                return False
            locomotion = {}
            fresh = False

        self._publish_actions(
            locomotion,
            fsm_state="NAV2_DRIVE" if fresh else "NAV2_WATCHDOG_STOP",
            active_primitive="cmd_vel",
            progress=0.0,
            success=False,
            done=False,
            message=(
                "Following /cmd_vel through morphology controller."
                if fresh
                else "/cmd_vel watchdog expired; locomotion stopped."
            ),
            task_type="morphology_velocity",
        )
        self._cmd_vel_output_active = fresh
        return True

    def _publish_actions(
        self,
        locomotion: Mapping[str, Mapping[str, float]],
        *,
        fsm_state: str,
        active_primitive: str,
        progress: float,
        success: bool,
        done: bool,
        message: str,
        task_type: str,
    ) -> None:
        self._actions_publisher.publish(
            dict_to_string_msg(
                {
                    "schema_version": "mssr.actions.v2",
                    "stamp": time.time(),
                    "stage_id": 0,
                    "task_type": task_type,
                    "reset": False,
                    "locomotion": {
                        module_id: dict(command)
                        for module_id, command in locomotion.items()
                    },
                    "magnetic": [],
                    "expert": {
                        "fsm_state": fsm_state,
                        "active_primitive": active_primitive,
                        "primitive_params": {},
                        "module_roles": {
                            item.module_id: item.target_role
                            for item in self._assignments
                        },
                        "task_metrics": {"progress": float(progress)},
                        "success": bool(success),
                        "done": bool(done),
                        "debug": {"message": message},
                    },
                }
            )
        )

    def _publish_status(
        self,
        command_id: str,
        morphology: str,
        behavior: str,
        state: str,
        phase: str,
        progress: float,
        done: bool,
        success: bool,
        message: str,
    ) -> None:
        self._status_publisher.publish(
            dict_to_string_msg(
                {
                    "schema_version": "mssr.morphology_status.v1",
                    "stamp": time.time(),
                    "command_id": command_id,
                    "morphology": morphology,
                    "behavior": behavior,
                    "state": state,
                    "phase": phase,
                    "progress": float(progress),
                    "done": bool(done),
                    "success": bool(success),
                    "message": message,
                }
            )
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SmoresMorphologyBehaviorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
