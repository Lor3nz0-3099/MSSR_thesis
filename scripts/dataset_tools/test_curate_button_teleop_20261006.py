from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/dataset_tools/curate_button_teleop_20261006.py"

EXPECTED = {
    "button-01": ("teleop-c78e0455357147369ad16bc1bf39a0c2", 6217),
    "button-02": ("teleop-1f8b406b56f74377834cd863ebccf696", 6307),
    "button-03": ("teleop-170338d489a64eb397a10cf9910392a1", 6533),
    "button-04": ("teleop-8b55584534d24c86888b350e09c74e7b", 6284),
    "button-05": ("teleop-b1456ab292aa4a319c9d4bcc17e7e57f", 6341),
    "button-06": ("teleop-beaa63ebe5f347c6abc58baff9d1394a", 6443),
    "button-07": ("teleop-dd6bf8b9926d4a2aab6af7572646d026", 6265),
    "button-08": ("teleop-edbefc66cecf40c180864b452aff8dab", 6316),
}


def test_button_curator_check_contract_is_complete_and_read_only() -> None:
    assert SCRIPT.is_file(), f"curator script missing: {SCRIPT}"

    collection = ROOT / "datasets/expert_v1/teleop/manifest.json"
    before = collection.read_bytes()

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    after = collection.read_bytes()

    assert after == before, "--check modified the teleop collection manifest"

    payload = json.loads(result.stdout)

    assert payload["mode"] == "check"
    assert payload["schema_version"] == "mssr.button_teleop_curation_check.v1"

    episodes = payload["episodes"]
    assert len(episodes) == 8

    observed = {
        item["label"]: (
            item["episode_id"],
            item["seed"],
        )
        for item in episodes
    }

    assert observed == EXPECTED

    for item in episodes:
        assert item["source_status"] == "completed"
        assert item["eligible_for_import"] is True
        assert item["unwritten_records"] == 0

        assert item["human_samples"] > 0
        assert item["human_action_valid_samples"] == item["human_samples"]
        assert (
            item["human_valid_for_behavior_cloning_samples"]
            == item["human_samples"]
        )

        assert item["max_button_depression_m"] >= 0.0035

        assert item["structural_streams"] >= 1
        assert item["all_structural_terminal_success"] is True

    assert payload["all_sources_valid"] is True


def test_button_curator_imports_eight_episodes_atomically(tmp_path: Path) -> None:
    dest = tmp_path / "teleop"
    dest.mkdir()
    (dest / "episodes").mkdir()

    collection = {
        "schema_version": "mssr.teleop_dataset_collection.v1",
        "dataset_name": "expert_v1_teleop",
        "created_at": "test",
        "selection_policy": "test fixture",
        "episodes": [],
        "totals": {
            "episodes": 0,
            "human_samples": 0,
            "structural_samples": 0,
            "samples": 0,
            "valid_for_behavior_cloning_samples": 0,
            "bytes": 0,
        },
    }

    (dest / "manifest.json").write_text(
        json.dumps(collection, indent=2) + "\n"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--import",
            "--dest-root",
            str(dest),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    payload = json.loads(result.stdout)

    assert payload["mode"] == "import"
    assert payload["imported_episodes"] == 8
    assert payload["all_sources_valid"] is True

    updated = json.loads((dest / "manifest.json").read_text())

    assert updated["schema_version"] == "mssr.teleop_dataset_collection.v1"
    assert len(updated["episodes"]) == 8
    assert updated["totals"]["episodes"] == 8

    imported_ids = {
        item["episode_id"]
        for item in updated["episodes"]
    }

    expected_ids = {
        episode_id
        for episode_id, _seed in EXPECTED.values()
    }

    assert imported_ids == expected_ids

    for label, (episode_id, seed) in EXPECTED.items():
        source = ROOT / "logs/teleop/recordings" / episode_id
        episode = dest / "episodes" / episode_id

        assert episode.is_dir()
        assert (episode / "human_behavior.jsonl").is_file()
        assert (episode / "source_manifest.json").is_file()
        assert (episode / "manifest.json").is_file()
        assert (episode / "structural").is_dir()

        source_structural = sorted(
            (source / "structural").glob("*.jsonl")
        )
        imported_structural = sorted(
            (episode / "structural").glob("*.jsonl")
        )

        assert [p.name for p in imported_structural] == [
            p.name for p in source_structural
        ]

        curated = json.loads(
            (episode / "manifest.json").read_text()
        )

        assert (
            curated["schema_version"]
            == "mssr.curated_teleop_episode.v1"
        )
        assert curated["episode_id"] == episode_id
        assert curated["task_success"] is True
        assert curated["full_mission_success"] is True
        assert curated["evidence"]["button_label"] == label
        assert curated["evidence"]["button_seed"] == seed
        assert (
            curated["evidence"]["max_button_depression_m"]
            >= 0.0035
        )

        datasets = curated["datasets"]

        assert len(datasets) == 1 + len(source_structural)
        assert datasets[0]["producer"] == "human_expert"

        assert (
            datasets[0]["samples"]
            == next(
                ep["human_samples"]
                for ep in payload["episodes"]
                if ep["episode_id"] == episode_id
            )
        )

    # Import must not leave staging directories behind.
    assert not list((dest / "episodes").glob(".*.staging"))
