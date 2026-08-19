from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class InternalMotionMode(str, Enum):
    """Mutually exclusive use of the two coupled internal motors."""

    PASSIVE = "passive"
    HOLD = "hold"
    STRUCTURAL_HOLD = "structural_hold"
    PAN = "pan"
    PAN_VELOCITY = "pan_velocity"
    TILT = "tilt"


@dataclass(frozen=True)
class SmoresCommand:
    linear_x_m_s: float = 0.0
    angular_z_rad_s: float = 0.0
    pan_target_rad: float = 0.0
    tilt_target_rad: float = 0.0
    # Real SMORES joints are backdrivable when their motors are not selected.
    # An explicit PAN/TILT/HOLD command is therefore required to energize the
    # internal differential; locomotion alone must not alter the posture.
    internal_motion: InternalMotionMode = InternalMotionMode.PASSIVE
    pan_velocity_rad_s: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.internal_motion, InternalMotionMode):
            raise ValueError("internal_motion must be an InternalMotionMode")
        values = (
            self.linear_x_m_s,
            self.angular_z_rad_s,
            self.pan_target_rad,
            self.tilt_target_rad,
            self.pan_velocity_rad_s,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("SMORES command values must be finite")


class Ros2SmoresTeleop:
    """Read drive, absolute pan/tilt, and relative pan ROS 2 commands."""

    def __init__(
        self,
        cmd_vel_topic: str,
        pan_topic: str,
        pan_delta_topic: str,
        tilt_topic: str,
        graph_path: str = "/ActionGraph/SmoresEPCommands",
    ) -> None:
        import omni.graph.core as og

        self._og = og
        self._command = SmoresCommand()
        graph, _, _, _ = og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_VARIABLES: [
                    ("panTargetRad", "float", 0.0),
                ],
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    (
                        "SubscribeTwist",
                        "isaacsim.ros2.bridge.ROS2SubscribeTwist",
                    ),
                    (
                        "SubscribePan",
                        "isaacsim.ros2.bridge.ROS2Subscriber",
                    ),
                    (
                        "SubscribeTilt",
                        "isaacsim.ros2.bridge.ROS2Subscriber",
                    ),
                    (
                        "SubscribePanDelta",
                        "isaacsim.ros2.bridge.ROS2Subscriber",
                    ),
                    ("ReadPanTarget", "omni.graph.core.ReadVariable"),
                    ("AddPanDelta", "omni.graph.nodes.Add"),
                    (
                        "WritePanAbsolute",
                        "omni.graph.core.WriteVariable",
                    ),
                    ("WritePanDelta", "omni.graph.core.WriteVariable"),
                ],
                og.Controller.Keys.CONNECT: [
                    (
                        "OnPlaybackTick.outputs:tick",
                        "SubscribeTwist.inputs:execIn",
                    ),
                    (
                        "OnPlaybackTick.outputs:tick",
                        "SubscribePan.inputs:execIn",
                    ),
                    (
                        "OnPlaybackTick.outputs:tick",
                        "SubscribeTilt.inputs:execIn",
                    ),
                    (
                        "OnPlaybackTick.outputs:tick",
                        "SubscribePanDelta.inputs:execIn",
                    ),
                ],
                og.Controller.Keys.SET_VALUES: [
                    (
                        "SubscribeTwist.inputs:topicName",
                        cmd_vel_topic.lstrip("/"),
                    ),
                    (
                        "SubscribePan.inputs:topicName",
                        pan_topic.lstrip("/"),
                    ),
                    (
                        "SubscribePan.inputs:messagePackage",
                        "std_msgs",
                    ),
                    ("SubscribePan.inputs:messageSubfolder", "msg"),
                    ("SubscribePan.inputs:messageName", "Float32"),
                    (
                        "SubscribeTilt.inputs:topicName",
                        tilt_topic.lstrip("/"),
                    ),
                    (
                        "SubscribeTilt.inputs:messagePackage",
                        "std_msgs",
                    ),
                    ("SubscribeTilt.inputs:messageSubfolder", "msg"),
                    ("SubscribeTilt.inputs:messageName", "Float32"),
                    (
                        "SubscribePanDelta.inputs:topicName",
                        pan_delta_topic.lstrip("/"),
                    ),
                    (
                        "SubscribePanDelta.inputs:messagePackage",
                        "std_msgs",
                    ),
                    (
                        "SubscribePanDelta.inputs:messageSubfolder",
                        "msg",
                    ),
                    (
                        "SubscribePanDelta.inputs:messageName",
                        "Float32",
                    ),
                    (
                        "ReadPanTarget.inputs:variableName",
                        "panTargetRad",
                    ),
                    (
                        "WritePanAbsolute.inputs:variableName",
                        "panTargetRad",
                    ),
                    (
                        "WritePanDelta.inputs:variableName",
                        "panTargetRad",
                    ),
                    (
                        "AddPanDelta.inputs:a",
                        0.0,
                        "float",
                    ),
                    (
                        "AddPanDelta.inputs:b",
                        0.0,
                        "float",
                    ),
                    (
                        "WritePanAbsolute.inputs:value",
                        0.0,
                        "float",
                    ),
                    (
                        "WritePanDelta.inputs:value",
                        0.0,
                        "float",
                    ),
                ],
            },
        )
        og.Controller.edit(
            graph,
            {
                og.Controller.Keys.CONNECT: [
                    (
                        f"{graph_path}/SubscribePan.outputs:execOut",
                        f"{graph_path}/WritePanAbsolute.inputs:execIn",
                    ),
                    (
                        f"{graph_path}/SubscribePan.outputs:data",
                        f"{graph_path}/WritePanAbsolute.inputs:value",
                    ),
                    (
                        f"{graph_path}/SubscribePanDelta.outputs:execOut",
                        f"{graph_path}/WritePanDelta.inputs:execIn",
                    ),
                    (
                        f"{graph_path}/ReadPanTarget.outputs:value",
                        f"{graph_path}/AddPanDelta.inputs:a",
                    ),
                    (
                        f"{graph_path}/SubscribePanDelta.outputs:data",
                        f"{graph_path}/AddPanDelta.inputs:b",
                    ),
                    (
                        f"{graph_path}/AddPanDelta.outputs:sum",
                        f"{graph_path}/WritePanDelta.inputs:value",
                    ),
                ],
            },
        )
        self._twist_node = og.Controller.node(
            f"{graph_path}/SubscribeTwist"
        )
        self._tilt_node = og.Controller.node(f"{graph_path}/SubscribeTilt")
        self._pan_target_variable = graph.find_variable("panTargetRad")
        self._graph_context = graph.get_context()

    def _read_float(self, node: object, fallback: float) -> float:
        attribute = node.get_attribute("outputs:data")
        if not attribute or not attribute.is_valid():
            return fallback
        value = self._og.Controller.get(attribute)
        if value is None:
            return fallback
        result = float(value)
        return result if math.isfinite(result) else fallback

    def update(self) -> SmoresCommand:
        linear = self._og.Controller.get(
            self._twist_node.get_attribute("outputs:linearVelocity")
        )
        angular = self._og.Controller.get(
            self._twist_node.get_attribute("outputs:angularVelocity")
        )
        linear_x = float(linear[0]) if linear is not None else 0.0
        angular_z = float(angular[2]) if angular is not None else 0.0
        if not math.isfinite(linear_x):
            linear_x = 0.0
        if not math.isfinite(angular_z):
            angular_z = 0.0
        pan_target = self._read_pan_target()
        tilt_target = self._read_float(
            self._tilt_node,
            self._command.tilt_target_rad,
        )
        pan_changed = not math.isclose(
            pan_target,
            self._command.pan_target_rad,
            abs_tol=1.0e-7,
        )
        tilt_changed = not math.isclose(
            tilt_target,
            self._command.tilt_target_rad,
            abs_tol=1.0e-7,
        )
        if pan_changed and tilt_changed:
            # PAN and TILT share the same internal motor pair. Reject an
            # ambiguous simultaneous update until one topic changes alone.
            pan_target = self._command.pan_target_rad
            tilt_target = self._command.tilt_target_rad
            internal_motion = InternalMotionMode.HOLD
        elif pan_changed:
            internal_motion = InternalMotionMode.PAN
        elif tilt_changed:
            internal_motion = InternalMotionMode.TILT
        else:
            internal_motion = self._command.internal_motion
        self._command = SmoresCommand(
            linear_x_m_s=linear_x,
            angular_z_rad_s=angular_z,
            pan_target_rad=pan_target,
            tilt_target_rad=tilt_target,
            internal_motion=internal_motion,
        )
        return self._command

    def _read_pan_target(self) -> float:
        value = float(self._pan_target_variable.get(self._graph_context))
        if math.isfinite(value):
            return value
        return self._command.pan_target_rad

    @property
    def command(self) -> SmoresCommand:
        return self._command
