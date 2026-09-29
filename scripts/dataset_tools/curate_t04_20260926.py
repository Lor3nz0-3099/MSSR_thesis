#!/usr/bin/env python3
"""Curate the successful T04 stairs/RC prefix before RC-to-Snake."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
EPISODE = "teleop-adcf33433f954669838220b9f617d952"
SOURCE = ROOT / "logs/teleop/recordings" / EPISODE
COURSE = ROOT / "logs/teleop/runs/teleop-t04-20260926-111017.x21y1J/course.json"
DEST = ROOT / "datasets/expert_v1/teleop"
GEOMETRY = "29042120f966af8518505e5af85f6bb36a6a81c3fc57190ad193c7709d2a9d56"
CUTOFF_SIM_S = 345.1583333333333  # First RC-to-Snake macro sample.
KEEP = (
    "teleop-snake_stairs-1790414030635672623.jsonl",
    "teleop-self_reconfiguration-1790414849376724495.jsonl",
)
EXCLUDE = (
    "teleop-self_assembly-1790413843756346162.jsonl",
    "teleop-self_reconfiguration-1790415421918300665.jsonl",
    "teleop-snake_gap-1790415723682418008.jsonl",
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def copy_rows(source: Path, target: Path, *, human: bool) -> dict:
    digest = hashlib.sha256()
    count = size = cloning = 0
    first = last = None
    last_row = None
    with source.open("rb") as reader, target.open("wb") as writer:
        for line in reader:
            row = json.loads(line)
            stamp = float(row["stamp"])
            if human and stamp >= CUTOFF_SIM_S:
                continue
            if float(row["graph_t_plus_1"]["stamp"]) >= CUTOFF_SIM_S:
                raise ValueError(f"Transition crosses RC-to-Snake boundary: {source}")
            if len(row["graph_t"]["nodes"]) != 8 or len(row["graph_t_plus_1"]["nodes"]) != 8:
                raise ValueError(f"Incomplete eight-module graph: {source}")
            if human:
                course = row["graph_t"]["global_attributes"]["course"]
                if course.get("geometry_sha256") != GEOMETRY or not row["action_valid"]:
                    raise ValueError("T04 human stream geometry/action changed")
            if last is not None and stamp < last:
                raise ValueError(f"Nonmonotonic stamp: {source}")
            writer.write(line)
            digest.update(line)
            count += 1
            size += len(line)
            cloning += bool(row["supervision"]["valid_for_behavior_cloning"])
            if first is None:
                first = stamp
            last = stamp
            last_row = row
    if human:
        if count != 2813 or last_row["observation"]["morphology"] != "rc_car8":
            raise ValueError("T04 human prefix changed")
    elif not count or not (last_row.get("done") and last_row.get("success")):
        raise ValueError(f"Structural macro lacks terminal success: {source}")
    relative = target.relative_to(DEST / "episodes" / f".{EPISODE}.staging")
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
    names = {Path(item["path"]).name for item in original["structural_streams"]}
    if (original["episode_id"] != EPISODE or original["status"] != "completed"
            or original["unwritten_records"] != 0
            or names != set(KEEP) | set(EXCLUDE)):
        raise ValueError("T04 source inventory changed")
    course = json.loads(COURSE.read_text())
    if course.get("geometry_sha256") != GEOMETRY:
        raise ValueError("T04 geometry changed")
    tasks = course["mission"]["tasks"]
    if [task["type"] for task in tasks] != [
        "stairs", "flat_navigation", "gap", "button", "goal"
    ]:
        raise ValueError("Unexpected T04 mission")

    final = DEST / "episodes" / EPISODE
    staging = DEST / "episodes" / f".{EPISODE}.staging"
    collection_path = DEST / "manifest.json"
    collection = json.loads(collection_path.read_text())
    if final.exists() or staging.exists() or any(
        item["episode_id"] == EPISODE for item in collection["episodes"]
    ):
        raise ValueError("T04 already curated; inspect before rerunning")

    staging.mkdir()
    (staging / "structural").mkdir()
    entries = [copy_rows(
        SOURCE / "human_behavior.jsonl",
        staging / "human_behavior.jsonl",
        human=True,
    )]
    for name in KEEP:
        entries.append(copy_rows(
            SOURCE / "structural" / name,
            staging / "structural" / name,
            human=False,
        ))
    shutil.copyfile(SOURCE / "manifest.json", staging / "source_manifest.json")
    shutil.copyfile(COURSE, staging / "course.json")
    curated = {
        "schema_version": "mssr.curated_teleop_episode.v1",
        "episode_id": EPISODE,
        "course": "T04",
        "geometry_sha256": GEOMETRY,
        "source_manifest": (SOURCE / "manifest.json").relative_to(ROOT).as_posix(),
        "source_manifest_copy": "source_manifest.json",
        "course_geometry": "course.json",
        "task_success": True,
        "success_basis": "snake_stairs_and_snake_to_rc_macros_terminal_success_with_rc_teleop_to_pre_gap_staging",
        "success_scope": "t04_stairs_and_rc_course_prefix_before_rc_to_snake",
        "full_mission_success": False,
        "cutoff_sim_s_exclusive": CUTOFF_SIM_S,
        "excluded_structural_streams": list(EXCLUDE),
        "exclusion_reason": "assembly_has_no_terminal_row; later_rc_to_snake_and_failed_gap_are_after_requested_cutoff",
        "datasets": entries,
    }
    write_json(staging / "manifest.json", curated)
    staging.rename(final)
    collection["episodes"].append({
        key: curated[key] for key in (
            "episode_id", "course", "task_success", "success_basis", "success_scope",
            "full_mission_success", "geometry_sha256",
        )
    } | {"manifest": f"episodes/{EPISODE}/manifest.json", "datasets": entries})
    totals = collection["totals"]
    totals["episodes"] = len(collection["episodes"])
    totals["human_samples"] += entries[0]["samples"]
    totals["structural_samples"] += sum(item["samples"] for item in entries[1:])
    totals["samples"] += sum(item["samples"] for item in entries)
    totals["valid_for_behavior_cloning_samples"] += sum(
        item["valid_for_behavior_cloning_samples"] for item in entries
    )
    totals["bytes"] += sum(item["bytes"] for item in entries)
    collection["selection_policy"] = (
        "Confirmed successful teleop scopes only: latest C05, T01 through button, "
        "complete T02/T03, and T04 stairs/RC prefix before RC-to-Snake. "
        "Failed, superseded and uncertified segments excluded."
    )
    temporary = collection_path.with_suffix(".json.tmp")
    write_json(temporary, collection)
    temporary.replace(collection_path)
    print(json.dumps({"episode_id": EPISODE, "datasets": entries, "totals": totals}, indent=2))


if __name__ == "__main__":
    main()
