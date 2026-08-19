"""Command sources for robot controllers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from robots.control import PlanarVelocityCommand


class CommandSource(Protocol):
    """Interface for objects that provide robot velocity commands."""

    def update(self) -> None:
        """Poll external systems and refresh the latest command."""

    def get_command(self) -> PlanarVelocityCommand:
        """Return the most recent command."""

    def close(self) -> None:
        """Release resources owned by the command source."""


class MultiModuleCommandSource(Protocol):
    """Interface for command sources that address multiple modules."""

    def update(self) -> None:
        """Poll external systems and refresh the latest commands."""

    def get_commands(self) -> Mapping[str, PlanarVelocityCommand]:
        """Return the latest command for each addressed module."""

    def close(self) -> None:
        """Release resources owned by the command source."""


class SingleModuleCommandAdapter:
    """Expose a single-module command source as a multi-module source."""

    def __init__(self, module_id: str, source: CommandSource) -> None:
        """Bind a single command source to one module id."""
        self._module_id = module_id
        self._source = source

    def update(self) -> None:
        """Refresh the wrapped source."""
        self._source.update()

    def get_commands(self) -> Mapping[str, PlanarVelocityCommand]:
        """Return the wrapped command under the configured module id."""
        return {self._module_id: self._source.get_command()}

    def close(self) -> None:
        """Release resources owned by the wrapped source."""
        self._source.close()


class ConstantCommandSource:
    """Command source that always returns the same command."""

    def __init__(self, command: PlanarVelocityCommand) -> None:
        """Store a constant command for demo and debugging runs."""
        self._command = command

    def update(self) -> None:
        """No-op because the command is constant."""

    def get_command(self) -> PlanarVelocityCommand:
        """Return the configured constant command."""
        return self._command

    def close(self) -> None:
        """No-op because no external resources are owned."""


class MultiConstantCommandSource:
    """Multi-module command source with one constant command per module."""

    def __init__(self, commands: Mapping[str, PlanarVelocityCommand]) -> None:
        """Store constant commands indexed by module id."""
        self._commands = dict(commands)

    def update(self) -> None:
        """No-op because commands are constant."""

    def get_commands(self) -> Mapping[str, PlanarVelocityCommand]:
        """Return the configured commands."""
        return self._commands

    def close(self) -> None:
        """No-op because no external resources are owned."""


class ScriptedCommandSource:
    """Command source that plays a deterministic sequence of timed commands."""

    def __init__(
        self,
        sequence: tuple[tuple[float, PlanarVelocityCommand], ...],
        dt: float = 1.0 / 60.0,
    ) -> None:
        """Store a timed command sequence.

        Each sequence item is ``(duration_seconds, command)``. Once the
        sequence ends, the source returns a zero command.
        """
        if dt <= 0.0:
            raise ValueError("ScriptedCommandSource requires a positive dt.")
        if any(duration < 0.0 for duration, _ in sequence):
            raise ValueError("Scripted command durations must be non-negative.")

        self._sequence = sequence
        self._dt = dt
        self._elapsed = 0.0
        self._command = PlanarVelocityCommand()

    def update(self) -> None:
        """Advance the scripted clock and refresh the active command."""
        self._command = self._command_at(self._elapsed)
        self._elapsed += self._dt

    def get_command(self) -> PlanarVelocityCommand:
        """Return the active scripted command."""
        return self._command

    def close(self) -> None:
        """No-op because no external resources are owned."""

    def _command_at(self, elapsed: float) -> PlanarVelocityCommand:
        """Return the command active at a given elapsed time."""
        cursor = 0.0
        for duration, command in self._sequence:
            cursor += duration
            if elapsed < cursor:
                return command
        return PlanarVelocityCommand()


def create_attachment_pull_test_command_source(
    approach_speed: float = 0.35,
    reverse_speed: float = -0.25,
    approach_duration: float = 4.0,
    reverse_duration: float = 3.0,
    settle_duration: float = 1.0,
    dt: float = 1.0 / 60.0,
) -> ScriptedCommandSource:
    """Create a command sequence that distinguishes pushing from attachment."""
    return ScriptedCommandSource(
        sequence=(
            (approach_duration, PlanarVelocityCommand(vx=approach_speed)),
            (settle_duration, PlanarVelocityCommand()),
            (reverse_duration, PlanarVelocityCommand(vx=reverse_speed)),
            (settle_duration, PlanarVelocityCommand()),
        ),
        dt=dt,
    )


class Ros2CmdVelSource:
    """ROS 2 command source backed by Isaac Sim's ``ROS2SubscribeTwist`` OmniGraph node."""

    def __init__(
        self,
        topic_name: str = "/cmd_vel",
        graph_path: str = "/ActionGraph/ROS2CmdVel",
        command_timeout: float = 0.5,
    ) -> None:
        """Create an OmniGraph subscriber for planar velocity commands.

        This avoids importing ``rclpy`` directly inside Isaac Python. The ROS 2
        bridge handles subscription internally through native OmniGraph nodes.
        """
        import omni.graph.core as og

        self._og = og
        self._command = PlanarVelocityCommand()
        self._command_timeout = command_timeout
        self._frames_without_message = 0

        topic_without_leading_slash = topic_name.lstrip("/")
        self._subscriber_path = f"{graph_path}/SubscribeTwist"

        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("SubscribeTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "SubscribeTwist.inputs:execIn"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    ("SubscribeTwist.inputs:topicName", topic_without_leading_slash),
                ],
            },
        )

        self._subscriber_node = og.Controller.node(self._subscriber_path)
        self._linear_velocity_attr = self._subscriber_node.get_attribute("outputs:linearVelocity")
        self._angular_velocity_attr = self._subscriber_node.get_attribute("outputs:angularVelocity")
        self._exec_out_attr = self._subscriber_node.get_attribute("outputs:execOut")

    def update(self) -> None:
        """Read the latest Twist values produced by the OmniGraph subscriber."""
        linear_velocity = self._og.Controller.get(self._linear_velocity_attr)
        angular_velocity = self._og.Controller.get(self._angular_velocity_attr)

        self._command = PlanarVelocityCommand(
            vx=float(linear_velocity[0]),
            vy=float(linear_velocity[1]),
            yaw_rate=float(angular_velocity[2]),
        )

        if self._command == PlanarVelocityCommand():
            self._frames_without_message += 1
        else:
            self._frames_without_message = 0

        if self._frames_without_message > max(1, int(self._command_timeout * 60.0)):
            self._command = PlanarVelocityCommand()

    def get_command(self) -> PlanarVelocityCommand:
        """Return the latest command received from ROS 2."""
        return self._command

    def close(self) -> None:
        """No-op because the OmniGraph node is owned by the stage."""
