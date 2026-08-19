from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TwistCommand:
    linear_x: float = 0.0
    angular_z: float = 0.0


class Ros2TwistTeleop:
    """Isaac ROS 2 bridge subscriber backed by ROS2SubscribeTwist."""

    def __init__(
        self,
        topic_name: str = "/cmd_vel",
        graph_path: str = "/ActionGraph/FreebotCmdVel",
        command_timeout_s: float = 0.5,
        physics_hz: float = 240.0,
    ) -> None:
        if not topic_name:
            raise ValueError("ROS 2 topic name cannot be empty")
        if not math.isfinite(command_timeout_s) or command_timeout_s <= 0.0:
            raise ValueError("Command timeout must be finite and positive")
        if not math.isfinite(physics_hz) or physics_hz <= 0.0:
            raise ValueError("Physics frequency must be finite and positive")

        import omni.graph.core as og

        self._og = og
        self._command = TwistCommand()
        self._zero_frame_count = 0
        self._timeout_frames = max(
            1,
            int(round(command_timeout_s * physics_hz)),
        )
        subscriber_path = f"{graph_path}/SubscribeTwist"
        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    (
                        "SubscribeTwist",
                        "isaacsim.ros2.bridge.ROS2SubscribeTwist",
                    ),
                ],
                og.Controller.Keys.CONNECT: [
                    (
                        "OnPlaybackTick.outputs:tick",
                        "SubscribeTwist.inputs:execIn",
                    ),
                ],
                og.Controller.Keys.SET_VALUES: [
                    (
                        "SubscribeTwist.inputs:topicName",
                        topic_name.lstrip("/"),
                    ),
                ],
            },
        )
        subscriber = og.Controller.node(subscriber_path)
        self._linear_attribute = subscriber.get_attribute(
            "outputs:linearVelocity"
        )
        self._angular_attribute = subscriber.get_attribute(
            "outputs:angularVelocity"
        )

    def update(self) -> TwistCommand:
        """Read the latest bridge output and apply a dead-command timeout."""

        linear = self._og.Controller.get(self._linear_attribute)
        angular = self._og.Controller.get(self._angular_attribute)
        command = TwistCommand(
            linear_x=float(linear[0]) if linear is not None else 0.0,
            angular_z=float(angular[2]) if angular is not None else 0.0,
        )
        if command == TwistCommand():
            self._zero_frame_count += 1
        else:
            self._zero_frame_count = 0
        self._command = (
            TwistCommand()
            if self._zero_frame_count > self._timeout_frames
            else command
        )
        return self._command

    @property
    def command(self) -> TwistCommand:
        return self._command

    def close(self) -> None:
        """The OmniGraph node is owned and destroyed by the USD stage."""
