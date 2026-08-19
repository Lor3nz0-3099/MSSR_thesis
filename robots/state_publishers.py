"""ROS 2 state publishers for simulated robots."""

from __future__ import annotations


class Ros2OdometryPublisher:
    """Publish robot odometry using Isaac Sim's native ROS 2 OmniGraph nodes."""

    def __init__(
        self,
        body_path: str,
        graph_path: str = "/ActionGraph/RobotState",
        topic_name: str = "odom",
        odom_frame_id: str = "odom",
        chassis_frame_id: str = "base_link",
    ) -> None:
        """Create an odometry graph for an existing robot rigid body."""
        import omni.graph.core as og
        import usdrt.Sdf

        keys = og.Controller.Keys
        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("ComputeOdometry", "isaacsim.core.nodes.IsaacComputeOdometry"),
                    ("PublishOdometry", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
                ],
                keys.SET_VALUES: [
                    ("ComputeOdometry.inputs:chassisPrim", [usdrt.Sdf.Path(body_path)]),
                    ("PublishOdometry.inputs:topicName", topic_name.lstrip("/")),
                    ("PublishOdometry.inputs:odomFrameId", odom_frame_id),
                    ("PublishOdometry.inputs:chassisFrameId", chassis_frame_id),
                    ("PublishOdometry.inputs:publishRawVelocities", False),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "ComputeOdometry.inputs:execIn"),
                    ("ComputeOdometry.outputs:execOut", "PublishOdometry.inputs:execIn"),
                    ("ComputeOdometry.outputs:position", "PublishOdometry.inputs:position"),
                    ("ComputeOdometry.outputs:orientation", "PublishOdometry.inputs:orientation"),
                    ("ComputeOdometry.outputs:linearVelocity", "PublishOdometry.inputs:linearVelocity"),
                    ("ComputeOdometry.outputs:angularVelocity", "PublishOdometry.inputs:angularVelocity"),
                    ("ReadSimTime.outputs:simulationTime", "PublishOdometry.inputs:timeStamp"),
                ],
            },
        )

    def close(self) -> None:
        """No-op because the OmniGraph nodes are owned by the stage."""


class Ros2TfPublisher:
    """Publish the robot transform using Isaac Sim's native ROS 2 OmniGraph nodes."""

    def __init__(
        self,
        body_path: str,
        graph_path: str = "/ActionGraph/RobotTf",
        topic_name: str = "tf",
        parent_frame_id: str = "odom",
        child_frame_id: str = "base_link",
    ) -> None:
        """Create a TF graph for an existing robot rigid body."""
        import omni.graph.core as og
        import usdrt.Sdf

        keys = og.Controller.Keys
        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("ComputeOdometry", "isaacsim.core.nodes.IsaacComputeOdometry"),
                    ("PublishTf", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
                ],
                keys.SET_VALUES: [
                    ("ComputeOdometry.inputs:chassisPrim", [usdrt.Sdf.Path(body_path)]),
                    ("PublishTf.inputs:topicName", topic_name.lstrip("/")),
                    ("PublishTf.inputs:parentFrameId", parent_frame_id),
                    ("PublishTf.inputs:childFrameId", child_frame_id),
                    ("PublishTf.inputs:staticPublisher", False),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "ComputeOdometry.inputs:execIn"),
                    ("ComputeOdometry.outputs:execOut", "PublishTf.inputs:execIn"),
                    ("ComputeOdometry.outputs:position", "PublishTf.inputs:translation"),
                    ("ComputeOdometry.outputs:orientation", "PublishTf.inputs:rotation"),
                    ("ReadSimTime.outputs:simulationTime", "PublishTf.inputs:timeStamp"),
                ],
            },
        )

    def close(self) -> None:
        """No-op because the OmniGraph nodes are owned by the stage."""


class Ros2ClockPublisher:
    """Publish Isaac simulation time as ROS 2 ``/clock``."""

    def __init__(self, graph_path: str = "/ActionGraph/Clock", topic_name: str = "clock") -> None:
        """Create a clock graph driven by simulation playback ticks."""
        import omni.graph.core as og

        keys = og.Controller.Keys
        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ],
                keys.SET_VALUES: [
                    ("PublishClock.inputs:topicName", topic_name.lstrip("/")),
                ],
                keys.CONNECT: [
                    ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                    ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ],
            },
        )

    def close(self) -> None:
        """No-op because the OmniGraph nodes are owned by the stage."""
