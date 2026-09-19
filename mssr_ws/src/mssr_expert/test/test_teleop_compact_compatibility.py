import importlib.util
import json
import sys
from pathlib import Path

from mssr_expert.graph.attributed_robot_graph import (
    AttributedRobotGraph,
    GraphNode,
)
from mssr_expert.teleop.recording import TeleopRecordingController


ROOT = Path(__file__).resolve().parents[4]


def load_builder():
    path = ROOT / "scripts/dataset_tools/build_expert_v1.py"
    spec = importlib.util.spec_from_file_location("build_expert_v1", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def graph(stamp, x):
    return AttributedRobotGraph(
        stamp=stamp,
        nodes=(
            GraphNode(
                "smores_01",
                {
                    "position": [x, 0.0, 0.05],
                    "position_relative_to_centroid": [0.0, 0.0, 0.0],
                    "orientation": [0.0, 0.0, 0.0, 1.0],
                    "linear_velocity": [0.01, 0.0, 0.0],
                    "angular_velocity": [0.0, 0.0, 0.1],
                    "role": "wheel_front_left",
                },
            ),
        ),
        global_attributes={
            "morphology_name": "rc_car8",
            "task_type": "rc_car8_teleop",
        },
    )


def test_human_record_passes_existing_raw_analyzer_and_compactor(tmp_path):
    controller = TeleopRecordingController(
        root=tmp_path / "recordings",
        git_commit="abc123",
        dataset_rate_hz=10.0,
        episode_id_factory=lambda: "demo-compact",
    )

    controller.update(
        recording_requested=True,
        authority="TELEOP",
        graph=graph(1.0, 0.0),
        controller_input={"r2": 0.7, "right_x": 0.2},
        intent={"forward": 0.7, "steering": 0.2},
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
        controller_input={"r2": 0.7, "right_x": 0.2},
        intent={"forward": 0.7, "steering": 0.2},
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
        controller_input={},
        intent={},
        effective_actions={},
        morphology="rc_car8",
        now=10.1,
        wall_time=100.1,
    )
    controller.wait_finalized(timeout=2.0)

    raw = (
        tmp_path
        / "recordings/demo-compact/human_behavior.jsonl"
    )

    builder = load_builder()

    analyzer, digest, byte_count = builder.analyze_existing(
        raw,
        "behavior",
    )

    summary = analyzer.summary()

    assert digest
    assert byte_count == raw.stat().st_size
    assert summary["records"] == 1
    assert summary["graph_transition_records"] == 1
    assert summary["action_valid_records"] == 1
    assert summary["behavior_cloning_valid_records"] == 1
    assert summary["explicit_action_payload_records"] == 1
    assert summary["il_eligible_records"] == 1
    assert summary["excluded_temporally_reversed_records"] == []

    compact_dir = tmp_path / "compact"
    compact_dir.mkdir()
    compact = compact_dir / "behavior.jsonl"

    count, compact_bytes, compact_sha = builder.compact_file(
        raw,
        compact,
        analyzer,
    )

    assert count == 1
    assert compact_bytes == compact.stat().st_size
    assert compact_sha

    row = json.loads(compact.read_text(encoding="utf-8"))

    # Contratto IL che deve sopravvivere alla compattazione.
    assert row["graph_t"]["stamp"] == 1.0
    assert row["graph_t_plus_1"]["stamp"] == 1.1
    assert (
        row["expert_action"]["locomotion"]["smores_01"]
        ["pan_rate_rad_s"]
        == 0.25
    )

    assert row["supervision"]["label_source"] == "human_expert"
    assert (
        row["supervision"]["executed_action_source"]
        == "human_expert"
    )

    attrs = row["graph_t"]["nodes"][0]["attributes"]
    assert attrs["position"] == [0.0, 0.0, 0.05]
    assert attrs["orientation"] == [0.0, 0.0, 0.0, 1.0]
    assert "linear_velocity" in attrs
    assert "angular_velocity" in attrs
