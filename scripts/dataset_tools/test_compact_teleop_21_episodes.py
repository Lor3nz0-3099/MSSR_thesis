from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/dataset_tools/compact_teleop_semantic.py"


def load_module():
    sys.path.insert(
        0,
        str(ROOT / "scripts/dataset_tools"),
    )

    spec = importlib.util.spec_from_file_location(
        "compact_teleop_semantic_under_test",
        SCRIPT,
    )

    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def test_main_accepts_21_curated_teleop_episodes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_module()

    source = tmp_path / "expert_v1" / "teleop"
    dest = tmp_path / "expert_v1_compact" / "teleop"

    source.mkdir(parents=True)
    dest.mkdir(parents=True)

    episodes = [
        {
            "episode_id": f"teleop-{index:02d}",
            "course": "BUTTON" if index >= 13 else "TXX",
            "task_success": True,
            "success_scope": "test",
            "manifest": (
                f"episodes/teleop-{index:02d}/manifest.json"
            ),
            "datasets": [],
        }
        for index in range(21)
    ]

    source_manifest = {
        "schema_version": "mssr.teleop_dataset_collection.v1",
        "episodes": episodes,
    }

    (source / "manifest.json").write_text(
        json.dumps(source_manifest) + "\n"
    )

    def fake_compact_episode(entry):
        return {
            "episode_id": entry["episode_id"],
            "course": entry["course"],
            "task_success": entry["task_success"],
            "success_scope": entry["success_scope"],
            "totals": {
                "files": 0,
                "samples": 0,
                "compact_records": 0,
                "collapsed_records": 0,
                "source_bytes": 0,
                "compact_jsonl_bytes": 0,
                "archive_bytes": 0,
            },
        }

    monkeypatch.setattr(
        module,
        "ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        module,
        "SOURCE",
        source,
    )
    monkeypatch.setattr(
        module,
        "DEST",
        dest,
    )
    monkeypatch.setattr(
        module,
        "compact_episode",
        fake_compact_episode,
    )

    module.main()

    result = json.loads(
        (dest / "manifest.json").read_text()
    )

    assert (
        result["schema_version"]
        == "mssr.semantic_teleop_archive_collection.v1"
    )

    assert len(result["episodes"]) == 21
    assert result["totals"]["episodes"] == 21

    assert {
        item["episode_id"]
        for item in result["episodes"]
    } == {
        f"teleop-{index:02d}"
        for index in range(21)
    }
