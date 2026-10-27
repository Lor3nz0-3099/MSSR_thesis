from pathlib import Path

from mssr_expert.teleop.action_transport import RcCarRuntime


ROOT = Path(__file__).resolve().parents[4]
NODE = (
    ROOT
    / "mssr_ws/src/mssr_expert/mssr_expert/nodes/smores_teleop_node.py"
)


def test_rc_runtime_exposes_authoritative_latest_graph():
    assert hasattr(RcCarRuntime, "latest_graph")


def test_teleop_node_wires_effective_actions_into_recording_backend():
    text = NODE.read_text(encoding="utf-8")

    assert "TeleopRecordingController" in text
    assert "recording_requested=status[\"recording\"]" in text
    assert "recording_graphs =" in text
    assert '"rc_car8": self._rc.latest_graph' in text
    assert '"snake8": self._snake.latest_graph' in text
    assert '"mobile_manipulator8": self._mm8.latest_graph' in text
    assert 'graph=recording_graphs.get(status["active_controller"])' in text
    assert "controller_input=status[\"controller_input\"]" in text
    assert "intent=output.actions.intent" in text
    assert "effective_actions=output.actions.module_actions" in text

    # Il vecchio placeholder T1/T3 non deve rimanere.
    assert "recording_backend_ready=False" not in text



def test_structural_ros_nodes_have_explicit_recording_gate():
    nodes = ROOT / "mssr_ws/src/mssr_expert/mssr_expert/nodes"
    for filename in (
        "smores_parallel_self_assembly_node.py",
        "smores_self_reconfiguration_node.py",
    ):
        source = (nodes / filename).read_text(encoding="utf-8")
        assert '"record_dataset"' in source
        assert 'record_dataset = bool(self.get_parameter("record_dataset").value)' in source
        assert 'if record_dataset and dataset_path else None' in source
