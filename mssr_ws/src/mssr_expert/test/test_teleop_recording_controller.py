import json

from mssr_expert.graph.attributed_robot_graph import (
    AttributedRobotGraph,
    GraphNode,
)
from mssr_expert.teleop.recording import TeleopRecordingController


def graph(stamp, x=0.0):
    return AttributedRobotGraph(
        stamp=stamp,
        nodes=(
            GraphNode(
                "smores_01",
                {
                    "position": [x, 0.0, 0.05],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                },
            ),
        ),
    )


def test_start_teleop_sample_and_stop_form_one_episode(tmp_path):
    controller = TeleopRecordingController(
        root=tmp_path,
        git_commit="abc123",
        dataset_rate_hz=10.0,
        queue_capacity=16,
        episode_id_factory=lambda: "demo-001",
    )

    controller.update(
        recording_requested=True,
        authority="TELEOP",
        graph=graph(1.0),
        controller_input={"r2": 1.0},
        intent={"forward": 1.0},
        effective_actions={
            "smores_01": {"pan_rate_rad_s": 0.25},
        },
        morphology="rc_car8",
        now=10.0,
        wall_time=100.0,
    )

    controller.update(
        recording_requested=True,
        authority="TELEOP",
        graph=graph(1.1, 0.01),
        controller_input={"r2": 1.0},
        intent={"forward": 1.0},
        effective_actions={
            "smores_01": {"pan_rate_rad_s": 0.25},
        },
        morphology="rc_car8",
        now=10.04,
        wall_time=100.04,
    )

    controller.update(
        recording_requested=False,
        authority="TELEOP",
        graph=graph(1.2, 0.02),
        controller_input={"r2": 0.0},
        intent={"forward": 0.0},
        effective_actions={
            "smores_01": {"pan_rate_rad_s": 0.0},
        },
        morphology="rc_car8",
        now=10.10,
        wall_time=100.10,
    )

    controller.wait_finalized(timeout=2.0)

    path = tmp_path / "demo-001" / "human_behavior.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    assert rows[0]["graph_t"]["stamp"] == 1.0
    assert rows[0]["graph_t_plus_1"]["stamp"] == 1.1
    assert rows[0]["supervision"]["label_source"] == "human_expert"


def test_structural_authority_does_not_create_human_transitions(tmp_path):
    controller = TeleopRecordingController(
        root=tmp_path,
        git_commit="abc123",
        dataset_rate_hz=10.0,
        episode_id_factory=lambda: "demo-structural",
    )

    controller.update(
        recording_requested=True,
        authority="STRUCTURAL_MACRO",
        graph=graph(1.0),
        controller_input={},
        intent={},
        effective_actions={},
        morphology="rc_car8",
        now=20.0,
        wall_time=200.0,
    )

    controller.update(
        recording_requested=True,
        authority="STRUCTURAL_MACRO",
        graph=graph(2.0),
        controller_input={},
        intent={},
        effective_actions={},
        morphology="snake8",
        now=20.2,
        wall_time=200.2,
    )

    controller.update(
        recording_requested=False,
        authority="TELEOP",
        graph=graph(3.0),
        controller_input={},
        intent={},
        effective_actions={},
        morphology="snake8",
        now=20.3,
        wall_time=200.3,
    )

    controller.wait_finalized(timeout=2.0)

    path = tmp_path / "demo-structural" / "human_behavior.jsonl"
    assert path.read_text() == ""


def test_non_teleop_authority_breaks_pending_human_transition(tmp_path):
    controller = TeleopRecordingController(
        root=tmp_path,
        git_commit="abc123",
        dataset_rate_hz=10.0,
        episode_id_factory=lambda: "demo-authority-boundary",
    )

    # Nasce una transizione human pendente.
    controller.update(
        recording_requested=True,
        authority="TELEOP",
        graph=graph(1.0),
        controller_input={"r2": 1.0},
        intent={"forward": 1.0},
        effective_actions={
            "smores_01": {"pan_rate_rad_s": 0.25},
        },
        morphology="rc_car8",
        now=10.00,
        wall_time=100.00,
    )

    # Entra una macro strutturale: la transizione human precedente
    # non deve sopravvivere a questo confine di autorità.
    controller.update(
        recording_requested=True,
        authority="STRUCTURAL_MACRO",
        graph=graph(2.0),
        controller_input={},
        intent={},
        effective_actions={},
        morphology="rc_car8",
        now=10.05,
        wall_time=100.05,
    )

    # Ritorno alla teleoperazione: deve iniziare un NUOVO campione,
    # non chiudere graph(1.0) con graph(3.0).
    controller.update(
        recording_requested=True,
        authority="TELEOP",
        graph=graph(3.0),
        controller_input={"r2": 0.5},
        intent={"forward": 0.5},
        effective_actions={
            "smores_01": {"pan_rate_rad_s": 0.10},
        },
        morphology="rc_car8",
        now=10.20,
        wall_time=100.20,
    )

    controller.update(
        recording_requested=True,
        authority="TELEOP",
        graph=graph(3.1),
        controller_input={"r2": 0.5},
        intent={"forward": 0.5},
        effective_actions={
            "smores_01": {"pan_rate_rad_s": 0.10},
        },
        morphology="rc_car8",
        now=10.24,
        wall_time=100.24,
    )

    controller.update(
        recording_requested=False,
        authority="TELEOP",
        graph=graph(3.2),
        controller_input={},
        intent={},
        effective_actions={},
        morphology="rc_car8",
        now=10.30,
        wall_time=100.30,
    )

    controller.wait_finalized(timeout=2.0)

    path = (
        tmp_path
        / "demo-authority-boundary"
        / "human_behavior.jsonl"
    )
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]

    assert len(rows) == 1
    assert rows[0]["graph_t"]["stamp"] == 3.0
    assert rows[0]["graph_t_plus_1"]["stamp"] == 3.1
    assert (
        rows[0]["expert_action"]["locomotion"]["smores_01"]
        ["pan_rate_rad_s"]
        == 0.10
    )
