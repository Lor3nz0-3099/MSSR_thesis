from __future__ import annotations


class Ros2DockingCommandSubscriber:
    """Return each std_msgs/String docking command exactly once."""

    def __init__(
        self,
        topic_name: str = "/smores_ep/docking_command",
        graph_path: str = "/ActionGraph/SmoresEPDockingCommands",
    ) -> None:
        if not topic_name.startswith("/"):
            raise ValueError("Docking command topic must be absolute")

        import omni.graph.core as og

        self._og = og
        graph, _, _, _ = og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_VARIABLES: [
                    ("commandSerial", "int", 0),
                ],
                og.Controller.Keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    (
                        "SubscribeDocking",
                        "isaacsim.ros2.bridge.ROS2Subscriber",
                    ),
                    ("ReadSerial", "omni.graph.core.ReadVariable"),
                    ("IncrementSerial", "omni.graph.nodes.Add"),
                    ("WriteSerial", "omni.graph.core.WriteVariable"),
                ],
                og.Controller.Keys.CONNECT: [
                    (
                        "OnPlaybackTick.outputs:tick",
                        "SubscribeDocking.inputs:execIn",
                    ),
                ],
                og.Controller.Keys.SET_VALUES: [
                    (
                        "SubscribeDocking.inputs:topicName",
                        topic_name.lstrip("/"),
                    ),
                    (
                        "SubscribeDocking.inputs:messagePackage",
                        "std_msgs",
                    ),
                    (
                        "SubscribeDocking.inputs:messageSubfolder",
                        "msg",
                    ),
                    (
                        "SubscribeDocking.inputs:messageName",
                        "String",
                    ),
                    (
                        "ReadSerial.inputs:variableName",
                        "commandSerial",
                    ),
                    (
                        "WriteSerial.inputs:variableName",
                        "commandSerial",
                    ),
                    ("IncrementSerial.inputs:a", 0, "int"),
                    ("IncrementSerial.inputs:b", 1, "int"),
                    ("WriteSerial.inputs:value", 0, "int"),
                ],
            },
        )
        og.Controller.edit(
            graph,
            {
                og.Controller.Keys.CONNECT: [
                    (
                        f"{graph_path}/SubscribeDocking.outputs:execOut",
                        f"{graph_path}/WriteSerial.inputs:execIn",
                    ),
                    (
                        f"{graph_path}/ReadSerial.outputs:value",
                        f"{graph_path}/IncrementSerial.inputs:a",
                    ),
                    (
                        f"{graph_path}/IncrementSerial.outputs:sum",
                        f"{graph_path}/WriteSerial.inputs:value",
                    ),
                ],
            },
        )
        subscriber = og.Controller.node(
            f"{graph_path}/SubscribeDocking"
        )
        self._data_attribute = subscriber.get_attribute("outputs:data")
        self._serial_variable = graph.find_variable("commandSerial")
        self._graph_context = graph.get_context()
        self._last_serial = 0

    def poll(self) -> str | None:
        serial = int(self._serial_variable.get(self._graph_context))
        if serial == self._last_serial:
            return None
        self._last_serial = serial
        value = self._og.Controller.get(self._data_attribute)
        if value is None:
            return None
        text = str(value).strip()
        return text or None
