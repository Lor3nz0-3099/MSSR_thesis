from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "run_snake_obstacle_dataset_campaign.py"
)
SPEC = importlib.util.spec_from_file_location(
    "snake_obstacle_dataset_campaign", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def test_levels_must_be_unique_and_easiest_first() -> None:
    assert campaign.parse_levels("robust,intermediate,challenging") == (
        "robust",
        "intermediate",
        "challenging",
    )
    with pytest.raises(ValueError, match="easiest to hardest"):
        campaign.parse_levels("challenging,robust")


def test_curriculum_gate_uses_completed_episode_success_rate() -> None:
    gate = campaign.curriculum_gate(
        [
            {"success": True},
            {"success": True},
            {"success": True},
            {"success": True},
            {"success": False},
        ],
        0.8,
    )

    assert gate["success_rate"] == pytest.approx(0.8)
    assert gate["passed"] is True
    assert campaign.curriculum_gate(
        [{"success": True}, {"success": False}], 0.8
    )["passed"] is False


def test_plan_only_materializes_separate_stair_and_gap_curricula(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "campaign"
    result = campaign.main(
        [
            "--output-dir",
            str(output_dir),
            "--episodes-per-level",
            "1",
            "--levels",
            "robust,intermediate",
            "--plan-only",
        ]
    )

    assert result == 0
    summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["execution_order"] == ["stairs", "gap"]
    assert summary["datasets_are_obstacle_specific"] is True
    for obstacle in ("stairs", "gap"):
        assert summary[obstacle]["completed_levels"] == [
            "robust",
            "intermediate",
        ]
        assert (output_dir / obstacle / "robust").is_dir()
        assert (output_dir / obstacle / "intermediate").is_dir()
