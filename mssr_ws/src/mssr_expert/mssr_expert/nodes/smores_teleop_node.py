"""External teleoperation shell with separate structure-stop and camera channels."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import yaml

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Joy
from std_msgs.msg import String

from mssr_expert.teleop.config import control_rate, load_teleop_config
from mssr_expert.teleop.coordinator import RuntimeCoordinator
from mssr_expert.teleop.camera import CameraController
from mssr_expert.teleop.input import load_input_config
from mssr_expert.teleop.session import TeleopSession
from mssr_expert.teleop.safety import SafetyDecision
from mssr_expert.teleop.structural_macro import StructuralMacroLauncher
from mssr_expert.teleop.action_transport import ActionTransport, RcCarRuntime
from mssr_expert.teleop.rc_car import load_geometry
from mssr_expert.teleop.snake import SnakeActions, SnakeRuntime
from mssr_expert.teleop.recording import TeleopRecordingController
from mssr_expert.teleop.topology import TeleopTopologyDetector
from mssr_expert.behaviors.morphology_library import MorphologyLibrary
from mssr_expert.graph.serialization import (
    attributed_graph_from_dict,
    load_attributed_graph,
)
from mssr_expert.planning.smores_ep.self_reconfiguration_planner import (
    SmoresSelfReconfigurationPlanner,
)


def _repository_root() -> Path:
    try:
        return Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.SubprocessError):
        return Path.cwd()


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class SmoresTeleopNode(Node):
    def __init__(self) -> None:
        super().__init__("mssr_smores_teleop_node")
        config_dir = Path(get_package_share_directory("mssr_expert")) / "config"
        input_path = self.declare_parameter("input_config_path", str(config_dir / "smores_dualsense.yaml")).value
        teleop_path = self.declare_parameter("teleop_config_path", str(config_dir / "smores_teleop.yaml")).value
        config = load_teleop_config(teleop_path)
        self.session = TeleopSession(
            load_input_config(input_path),
            controller_morphologies={"rc_car8", "snake8"},
        )
        self._structural_macro = StructuralMacroLauncher()

        self.coordinator = RuntimeCoordinator(
            self.session,
            camera=CameraController(radius_m=config.camera_radius_m),
            structural_request_handler=self._start_structural_macro,
        )

        self._topology_detector = TeleopTopologyDetector(
            catalog={
                morphology: load_attributed_graph(
                    config_dir / f"smores_{morphology}.json"
                )
                for morphology in (
                    "rc_car8",
                    "snake8",
                    "mobile_manipulator8",
                )
            },
            matcher=SmoresSelfReconfigurationPlanner(),
        )

        topology_timeout = float(
            self.declare_parameter(
                "topology_observation_timeout_s",
                0.5,
            ).value
        )
        if not 0.0 < topology_timeout <= 10.0:
            raise ValueError(
                "topology_observation_timeout_s must be in (0, 10]"
            )

        self._topology_observation_timeout_s = topology_timeout
        self._topology_received_at: float | None = None
        self._topology_name: str | None = None

        repo_root = _repository_root()
        self._repo_root = repo_root
        self._structural_dataset_root = Path(
            self.declare_parameter(
                "structural_dataset_root",
                str(repo_root / "logs/teleop/structural"),
            ).value
        )

        structural_exit_grace_s = float(
            self.declare_parameter(
                "structural_exit_grace_s",
                0.5,
            ).value
        )

        if not 0.0 <= structural_exit_grace_s <= 10.0:
            raise ValueError(
                "structural_exit_grace_s must be in [0, 10]"
            )

        self._structural_exit_grace_s = structural_exit_grace_s

        recording_root = Path(
            self.declare_parameter(
                "recording_root",
                str(repo_root / "logs/teleop/recordings"),
            ).value
        )
        self._recording = TeleopRecordingController(
            root=recording_root,
            git_commit=_git_commit(repo_root),
            dataset_rate_hz=config.dataset_rate_hz,
        )
        rc_config = yaml.safe_load(Path(teleop_path).read_text()).get("rc_car", {})
        self._rc = RcCarRuntime(MorphologyLibrary.load(config_dir / "smores_morphology_behaviors.json"),
                                load_attributed_graph(config_dir / "smores_rc_car8.json"),
                                geometry=load_geometry(str(teleop_path)), **rc_config)
        snake_config = yaml.safe_load(Path(teleop_path).read_text()).get("snake", {})
        self._snake = SnakeRuntime(MorphologyLibrary.load(config_dir / "smores_morphology_behaviors.json"),
                                   load_attributed_graph(config_dir / "smores_snake8.json"),
                                   geometry=load_geometry(str(teleop_path)), **snake_config)
        self._action_transport = ActionTransport()
        self._last_actuator_controller = None
        self._actions = self.create_publisher(String, "/mssr/actions", 10)
        self._goal = self.create_publisher(String, "/mssr/primitives/goal", 10)
        self._cancel = self.create_publisher(String, "/mssr/primitives/cancel", 10)
        self._graph = self.create_subscription(String, "/mssr/robot_graph", self._on_graph, 10)
        self._primitive_status = self.create_subscription(String, "/mssr/primitives/status", self._on_primitive_status, 10)
        self._self_reconfiguration_state = self.create_subscription(
            String,
            "/mssr/expert/self_reconfiguration/state",
            self._on_self_reconfiguration_state,
            10,
        )
        joy_topic = self.declare_parameter("joy_topic", config.joy_topic).value
        status_topic = self.declare_parameter("status_topic", "/mssr/teleop/status").value
        rate = control_rate(self.declare_parameter("control_rate_hz", config.control_rate_hz).value)
        self._status = self.create_publisher(String, status_topic, 10)
        runtime_request_topic = self.declare_parameter("runtime_request_topic", "/mssr/teleop/runtime_request").value
        runtime_status_topic = self.declare_parameter("runtime_status_topic", "/mssr/teleop/runtime_status").value
        self._runtime_request = self.create_publisher(String, runtime_request_topic, 10)
        self._runtime_status = self.create_subscription(String, runtime_status_topic, self._on_runtime_status, 10)
        self._joy = self.create_subscription(Joy, joy_topic, self._on_joy, qos_profile_sensor_data)
        # Input age and diagnostics use a steady wall clock independent of Isaac simulation time.
        self._wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self._timer = self.create_timer(1.0 / rate, self._tick, clock=self._wall_clock)
        self.get_logger().info(f"RC-Car8/Snake8 teleop at {rate:g} Hz; live topology and runtime safety required")

    def _on_graph(self, message: String) -> None:
        now = time.monotonic()

        try:
            payload = json.loads(message.data)
        except (ValueError, TypeError):
            payload = None

        self._rc.observe_graph(payload, now=now)
        self._snake.observe_graph(payload, now=now)

        try:
            current_graph = attributed_graph_from_dict(payload)
        except (
            AttributeError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ):
            self._topology_name = None
            self._topology_received_at = None
            return

        self._topology_name = self._topology_detector.detect(
            current_graph
        )
        self._topology_received_at = now

    def _detected_topology(self, now: float) -> str | None:
        received_at = self._topology_received_at

        if received_at is None:
            return None

        age = now - received_at

        if (
            age < 0.0
            or age > self._topology_observation_timeout_s
        ):
            return None

        if self._topology_name == "snake8" and self._snake.topology(now) is None:
            return None
        return self._topology_name

    def _on_primitive_status(self, message: String) -> None:
        try:
            self._rc.observe_status(json.loads(message.data))
            self._snake.observe_status(json.loads(message.data))
        except (KeyError, RuntimeError, TypeError, ValueError):
            return

    def _start_structural_macro(self, target_morphology: str) -> bool:
        stamp = time.time_ns()
        execution_id = f"teleop-reconfiguration-{stamp}"

        manager = self._recording.manager
        recording_episode_id = self._recording.episode_id

        if (
            manager is not None
            and manager.recording
            and manager.episode_dir is not None
            and recording_episode_id is not None
        ):
            episode_id = recording_episode_id
            dataset_path = (
                manager.episode_dir
                / "structural"
                / f"{execution_id}.jsonl"
            )
        else:
            episode_id = f"teleop-structural-{stamp}"
            dataset_path = (
                self._structural_dataset_root
                / f"{execution_id}.jsonl"
            )

        try:
            self._structural_macro.start(
                state=self.session.state,
                target_morphology=target_morphology,
                execution_id=execution_id,
                episode_id=episode_id,
                dataset_path=dataset_path,
            )
        except (OSError, RuntimeError, ValueError) as error:
            self.get_logger().error(
                "Cannot start structural macro "
                f"{target_morphology!r}: {error}"
            )
            return False

        # Only a successfully launched deterministic expert belongs in the
        # T4 demonstration manifest.  A failed spawn must never leave a
        # structural-stream reference to data that was never produced.
        if (
            manager is not None
            and manager.recording
            and recording_episode_id is not None
        ):
            try:
                self._recording.register_structural_stream(
                    stream_id=execution_id,
                    phase="self_reconfiguration",
                    path=dataset_path,
                    producer="deterministic_expert",
                )
            except RuntimeError as error:
                # Dataset bookkeeping must not revoke structural authority
                # from an expert that already started controlling the robot.
                self.get_logger().error(
                    "Cannot link structural stream into recording manifest: "
                    f"{error}"
                )

        self.get_logger().info(
            "Started structural macro "
            f"{target_morphology!r} as {execution_id!r}."
        )
        return True

    def _on_self_reconfiguration_state(
        self,
        message: String,
    ) -> None:
        try:
            payload = json.loads(message.data)
        except (ValueError, TypeError):
            return

        consumed = self._structural_macro.observe_expert_state(
            state=self.session.state,
            payload=payload,
        )

        if consumed:
            self.get_logger().info(
                "Structural macro terminal state consumed: "
                f"success={self.session.state.last_macro_success}."
            )

    def _on_joy(self, message: Joy) -> None:
        self.session.update_joy(message.axes, message.buttons, time.monotonic())

    def _on_runtime_status(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (ValueError, TypeError):
            return
        self.coordinator.observe_runtime(payload, time.monotonic())

    def _tick(self) -> None:
        now = time.monotonic()
        self.session.state.observe_topology(
            self._detected_topology(now)
        )
        status, runtime_request = self.coordinator.tick(now)

        if (
            status["authority"] == "ESTOP"
            and self._structural_macro.process is not None
        ):
            cancel_goal_ids = self._structural_macro.interrupt()

            for goal_id in cancel_goal_ids:
                self._cancel.publish(
                    String(
                        data=json.dumps(
                            {"goal_id": goal_id},
                            allow_nan=False,
                        )
                    )
                )

        watchdog_cancel_goal_ids = (
            None
            if status["authority"] == "ESTOP"
            else self._structural_macro.check_process(
                state=self.session.state,
                now=now,
                exit_grace_s=self._structural_exit_grace_s,
            )
        )

        if watchdog_cancel_goal_ids is not None:
            for goal_id in watchdog_cancel_goal_ids:
                self._cancel.publish(
                    String(
                        data=json.dumps(
                            {"goal_id": goal_id},
                            allow_nan=False,
                        )
                    )
                )

            self.get_logger().error(
                "Structural expert exited without an authoritative "
                "terminal state; macro marked failed."
            )

        controller_input = SimpleNamespace(**status["controller_input"])
        decision = SafetyDecision(**status["safety"])
        controller = status["active_controller"] if decision.motion_enabled else None
        disabled = SafetyDecision("NONE", False, True, False)
        rc_output = self._rc.step(controller_input,
                                  safety=decision if controller == "rc_car8" else disabled, now=now)
        snake_output = self._snake.step(controller_input,
                                        safety=decision if controller == "snake8" else disabled, now=now)
        output = snake_output if controller == "snake8" else rc_output
        if controller is not None and output.envelope is not None:
            self._actions.publish(String(data=output.envelope))
            self._last_actuator_controller = controller
        elif self._last_actuator_controller is not None and status["authority"] != "STRUCTURAL_MACRO":
            # An empty action packet clears the native dead-man cache on the
            # same tick as input/topology authority is lost.
            clear = self._action_transport.serialize(
                SnakeActions(), stamp=time.time(),
                command_id="teleop-clear-" + str(time.time_ns()),
                morphology=self._last_actuator_controller,
            )
            self._actions.publish(String(data=clear))
            self._last_actuator_controller = None
        cancellations = []
        for delivery in (rc_output.posture, snake_output.posture):
            if delivery.goal is not None:
                self._goal.publish(String(data=json.dumps(delivery.goal.to_dict(), allow_nan=False)))
            if delivery.cancel_goal_id is not None:
                cancellations.append(delivery.cancel_goal_id)
            cancellations.extend(delivery.cancel_goal_ids)
        if cancellations:
            unique = tuple(dict.fromkeys(cancellations))
            cancel_payload = ({"goal_id": unique[0]} if len(unique) == 1
                              else {"goal_ids": list(unique)})
            self._cancel.publish(String(data=json.dumps(cancel_payload)))
        recording_events = tuple(
            {
                "kind": "command",
                "name": str(name),
                "stamp_monotonic": now,
            }
            for name in status["controller_input"]["command_events"]
        ) + tuple(
            {
                "kind": "session",
                "name": str(name),
                "stamp_monotonic": now,
            }
            for name in status["events"]
        )

        self._recording.update(
            recording_requested=status["recording"],
            authority=status["authority"],
            graph=self._rc.latest_graph if controller == "rc_car8" else self._snake.latest_graph,
            controller_input=status["controller_input"],
            intent=output.actions.intent,
            effective_actions=output.actions.module_actions,
            morphology=status["active_controller"],
            now=now,
            wall_time=time.time(),
            events=recording_events,
        )

        status["stamp_ros"] = self.get_clock().now().nanoseconds * 1e-9
        status.update(
            actuator_commands_enabled=bool(
                status["safety"]["motion_enabled"] and output.envelope
            ),
            topology_verification_ready=(
                self._detected_topology(now) is not None
            ),
            recording_backend_ready=self._recording.backend_ready,
            recording_episode_id=self._recording.episode_id,
            recording_error=self._recording.error,
            rc_car_intent=rc_output.actions.intent,
            rc_car_effective_actions=rc_output.actions.module_actions,
            snake_intent=snake_output.actions.intent,
            snake_effective_actions=snake_output.actions.module_actions,
        )
        message = String()
        message.data = json.dumps(status, allow_nan=False)
        self._status.publish(message)
        request = String()
        request.data = json.dumps(runtime_request, allow_nan=False)
        self._runtime_request.publish(request)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = SmoresTeleopNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
