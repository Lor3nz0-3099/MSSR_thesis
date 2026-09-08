"""Tests for campaign selection and composite runner inputs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "run_composite_course.py"
SPEC = importlib.util.spec_from_file_location("run_composite_course", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_select_episode_materializes_mission_schema() -> None:
    campaign = {
        "schema_version": "mssr.composite_campaign.v1",
        "episodes": [
            {
                "episode_id": "test-1",
                "tasks": [{"task_id": "gap", "type": "gap", "seed": 4100}],
            }
        ],
    }
    mission = MODULE.select_episode(campaign, "test-1")
    assert mission["schema_version"] == "mssr.composite_mission.v1"
    assert mission["tasks"][0]["seed"] == 4100


def test_select_episode_rejects_unknown_id() -> None:
    with pytest.raises(ValueError, match="was not found exactly once"):
        MODULE.select_episode(
            {"schema_version": "mssr.composite_campaign.v1", "episodes": []},
            "missing",
        )


def test_normalize_dataset_creates_one_episode_terminal(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    records = [
        {
            "episode_id": "local-a",
            "timestep": 0,
            "done": True,
            "success": True,
            "graph_t": {"stamp": 1},
            "observation_t": {"phase": "assembly"},
        },
        {
            "episode_id": "local-b",
            "timestep": 0,
            "done": True,
            "success": True,
            "graph_t": {"stamp": 2},
            "observation_t": {"phase": "stairs"},
        },
    ]
    path.write_text("".join(__import__("json").dumps(item) + "\n" for item in records))

    MODULE.normalize_dataset(path, "composite-test", True)

    normalized = [
        __import__("json").loads(line)
        for line in path.read_text().splitlines()
    ]
    assert [item["timestep"] for item in normalized] == [0, 1]
    assert [item["done"] for item in normalized] == [False, True]
    assert normalized[0]["graph_t_plus_1"] == normalized[1]["graph_t"]
    assert normalized[-1]["episode_success"] is True
