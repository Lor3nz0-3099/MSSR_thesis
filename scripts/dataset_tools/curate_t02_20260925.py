#!/usr/bin/env python3
"""Copy the complete successful T02 demo, excluding uncertified assembly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
EPISODE = "teleop-1bea8f28c1c04323a4c3ec7e2829d07c"
SOURCE = ROOT / "logs/teleop/recordings" / EPISODE
COURSE = ROOT / "logs/teleop/runs/teleop-t02-20260925-144049.yJK53B/course.json"
DEST = ROOT / "datasets/expert_v1/teleop"
GEOMETRY = "828e7461e65e02bae3dffed6f1432ac61679b0b38a71cb0d9173ff3fe087fb23"
EXCLUDED = "teleop-self_assembly-1790340068040284813.jsonl"
KEEP = (
    "teleop-self_reconfiguration-1790340276721552774.jsonl",
    "teleop-snake_gap-1790340467042632522.jsonl",
    "teleop-self_reconfiguration-1790340713024094596.jsonl",
    "teleop-self_reconfiguration-1790341290042661895.jsonl",
    "teleop-self_reconfiguration-1790341487275042032.jsonl",
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def copy_rows(source: Path, target: Path, *, human: bool) -> dict:
    digest = hashlib.sha256()
    count = size = cloning = 0
    first = last = None
    with source.open("rb") as reader, target.open("wb") as writer:
        for line in reader:
            row = json.loads(line)
            if len(row["graph_t"]["nodes"]) != 8 or len(row["graph_t_plus_1"]["nodes"]) != 8:
                raise ValueError(f"Incomplete eight-module graph: {source}")
            if human:
                course = row["graph_t"]["global_attributes"]["course"]
                if course.get("geometry_sha256") != GEOMETRY:
                    raise ValueError("T02 course geometry changed")
                if not row["action_valid"]:
                    raise ValueError("Invalid human action")
            writer.write(line)
            digest.update(line)
            count += 1
            size += len(line)
            cloning += bool(row["supervision"]["valid_for_behavior_cloning"])
            stamp = float(row["stamp"])
            if last is not None and stamp < last:
                raise ValueError(f"Nonmonotonic graph stamp: {source}")
            if first is None:
                first = stamp
            last = stamp
    if not count:
        raise ValueError(f"Empty source: {source}")
    if not human and not (row.get("done") and row.get("success")):
        raise ValueError(f"Structural stream lacks terminal success: {source}")
    if human and count != 5512:
        raise ValueError("Human stream changed since audit")
    relative = target.relative_to(DEST / "episodes" / (f".{EPISODE}.staging"))
    return {
        "path": (Path("episodes") / EPISODE / relative).as_posix(),
        "source_path": source.relative_to(ROOT).as_posix(),
        "sha256": digest.hexdigest(),
        "bytes": size,
        "samples": count,
        "valid_for_behavior_cloning_samples": cloning,
        "first_stamp": first,
        "last_stamp": last,
        "producer": "human_expert" if human else "deterministic_expert",
    }


def main() -> None:
    original = json.loads((SOURCE / "manifest.json").read_text())
    stream_names = {Path(item["path"]).name for item in original["structural_streams"]}
    if (original["episode_id"] != EPISODE or original["status"] != "completed"
            or stream_names != set(KEEP) | {EXCLUDED}):
        raise ValueError("T02 source episode or structural inventory changed")
    course = json.loads(COURSE.read_text())
    if course.get("geometry_sha256") != GEOMETRY:
        raise ValueError("T02 course geometry changed")

    final = DEST / "episodes" / EPISODE
    staging = DEST / "episodes" / f".{EPISODE}.staging"
    if final.exists() or staging.exists():
        raise ValueError("Curated T02 output already exists; inspect before rerunning")
    collection_path = DEST / "manifest.json"
    collection = json.loads(collection_path.read_text())
    if any(item["episode_id"] == EPISODE for item in collection["episodes"]):
        raise ValueError("T02 already appears in the collection manifest")

    staging.mkdir()
    (staging / "structural").mkdir()
    entries = [copy_rows(SOURCE / "human_behavior.jsonl", staging / "human_behavior.jsonl", human=True)]
    for name in KEEP:
        entries.append(copy_rows(SOURCE / "structural" / name, staging / "structural" / name, human=False))
    shutil.copyfile(SOURCE / "manifest.json", staging / "source_manifest.json")
    shutil.copyfile(COURSE, staging / "course.json")

    curated = {
        "schema_version": "mssr.curated_teleop_episode.v1",
        "episode_id": EPISODE,
        "course": "T02",
        "geometry_sha256": GEOMETRY,
        "source_manifest": (SOURCE / "manifest.json").relative_to(ROOT).as_posix(),
        "source_manifest_copy": "source_manifest.json",
        "course_geometry": "course.json",
        "task_success": True,
        "success_basis": "physical_graph_gap_button_depression_and_goal_platform_with_structural_terminal_success",
        "success_scope": "complete_t02_gap_cones_button_goal_and_final_rc",
        "full_mission_success": True,
        "excluded_structural_streams": [EXCLUDED],
        "exclusion_reason": "initial_assembly_stream_has_no_terminal_row_despite_seven_docks_and_four_successful_posture_goals_in_isaac_log",
        "datasets": entries,
    }
    write_json(staging / "manifest.json", curated)
    staging.rename(final)

    collection["episodes"].append({
        key: curated[key] for key in (
            "episode_id", "course", "task_success", "success_basis", "success_scope",
            "full_mission_success", "geometry_sha256")
    } | {"manifest": f"episodes/{EPISODE}/manifest.json", "datasets": entries})
    totals = collection["totals"]
    totals["episodes"] = len(collection["episodes"])
    totals["human_samples"] += entries[0]["samples"]
    totals["structural_samples"] += sum(item["samples"] for item in entries[1:])
    totals["samples"] += sum(item["samples"] for item in entries)
    totals["valid_for_behavior_cloning_samples"] += sum(item["valid_for_behavior_cloning_samples"] for item in entries)
    totals["bytes"] += sum(item["bytes"] for item in entries)
    collection["selection_policy"] = (
        "Only successful teleop demonstrations: latest C05, T01 through button press, "
        "and complete T02. Incomplete structural streams and superseded recordings excluded."
    )
    temporary = collection_path.with_suffix(".json.tmp")
    write_json(temporary, collection)
    temporary.replace(collection_path)
    print(json.dumps({"episode_id": EPISODE, "datasets": entries, "totals": totals}, indent=2))


if __name__ == "__main__":
    main()
