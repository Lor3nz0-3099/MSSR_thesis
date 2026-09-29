#!/usr/bin/env python3
"""Curate the completed T03 course without its uncertified assembly stream."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
EPISODE = "teleop-3aa4f69128774d5eba1f8d851c02cf53"
SOURCE = ROOT / "logs/teleop/recordings" / EPISODE
COURSE = ROOT / "logs/teleop/runs/teleop-t03-20260925-154622.thMRXr/course.json"
DEST = ROOT / "datasets/expert_v1/teleop"
GEOMETRY = "647863e6d88274f50578da8505489be9024547ce13240eb5ccba1eefc840fcb8"
EXCLUDED = "teleop-self_assembly-1790344103204452827.jsonl"
KEEP = (
    "teleop-self_reconfiguration-1790344450144321607.jsonl",
    "teleop-snake_gap-1790344604924100433.jsonl",
    "teleop-self_reconfiguration-1790344832043773135.jsonl",
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def copy_rows(source: Path, target: Path, *, human: bool, far_x: float, goal_box: dict) -> dict:
    digest = hashlib.sha256()
    count = size = cloning = 0
    first = last = None
    gap_crossed = goal_reached = False
    gx, gy = goal_box["center_xyz_m"][:2]
    sx, sy = goal_box["size_xyz_m"][:2]
    with source.open("rb") as reader, target.open("wb") as writer:
        for line in reader:
            row = json.loads(line)
            graph = row["graph_t"]
            if len(graph["nodes"]) != 8 or len(row["graph_t_plus_1"]["nodes"]) != 8:
                raise ValueError(f"Incomplete eight-module graph: {source}")
            if human:
                if graph["global_attributes"]["course"].get("geometry_sha256") != GEOMETRY:
                    raise ValueError("T03 course geometry changed")
                if not row["action_valid"]:
                    raise ValueError("Invalid human action")
                points = [node["attributes"]["position"] for node in graph["nodes"]]
                gap_crossed |= all(point[0] > far_x for point in points)
                goal_reached |= all(
                    abs(point[0] - gx) <= sx / 2 and abs(point[1] - gy) <= sy / 2
                    for point in points
                )
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
    if human:
        if count != 3828 or not gap_crossed or not goal_reached:
            raise ValueError("Human stream no longer confirms complete T03 traversal")
        if row["observation"]["morphology"] != "rc_car8":
            raise ValueError("T03 does not finish in RC car")
    elif not (row.get("done") and row.get("success")):
        raise ValueError(f"Structural stream lacks terminal success: {source}")
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
    stream_names = {Path(item["path"]).name for item in original["structural_streams"]}
    if (original["episode_id"] != EPISODE or original["status"] != "completed"
            or original["unwritten_records"] != 0
            or stream_names != set(KEEP) | {EXCLUDED}):
        raise ValueError("T03 source episode or structural inventory changed")
    course = json.loads(COURSE.read_text())
    if course.get("geometry_sha256") != GEOMETRY:
        raise ValueError("T03 course geometry changed")
    mission = course["mission"]
    if [item["type"] for item in mission["tasks"]] != [
        "flat_navigation", "gap", "flat_navigation", "goal"
    ]:
        raise ValueError("Unexpected T03 mission")
    far_x = mission["tasks"][1]["parameters"]["gap"]["far_edge_center_world_xyz_m"][0]
    goal_box = next(box for box in course["collision_boxes"] if box["semantic"] == "goal_platform")

    final = DEST / "episodes" / EPISODE
    staging = DEST / "episodes" / f".{EPISODE}.staging"
    if final.exists() or staging.exists():
        raise ValueError("Curated T03 output already exists; inspect before rerunning")
    collection_path = DEST / "manifest.json"
    collection = json.loads(collection_path.read_text())
    if any(item["episode_id"] == EPISODE for item in collection["episodes"]):
        raise ValueError("T03 already appears in the collection manifest")

    staging.mkdir()
    (staging / "structural").mkdir()
    entries = [copy_rows(SOURCE / "human_behavior.jsonl", staging / "human_behavior.jsonl",
                         human=True, far_x=far_x, goal_box=goal_box)]
    for name in KEEP:
        entries.append(copy_rows(SOURCE / "structural" / name, staging / "structural" / name,
                                 human=False, far_x=far_x, goal_box=goal_box))
    shutil.copyfile(SOURCE / "manifest.json", staging / "source_manifest.json")
    shutil.copyfile(COURSE, staging / "course.json")
    curated = {
        "schema_version": "mssr.curated_teleop_episode.v1",
        "episode_id": EPISODE,
        "course": "T03",
        "geometry_sha256": GEOMETRY,
        "source_manifest": (SOURCE / "manifest.json").relative_to(ROOT).as_posix(),
        "source_manifest_copy": "source_manifest.json",
        "course_geometry": "course.json",
        "task_success": True,
        "success_basis": "all_modules_cross_gap_and_finish_on_goal_platform_with_structural_terminal_success",
        "success_scope": "complete_t03_first_rc_route_gap_second_rc_route_goal",
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
        "and complete T02 and T03. Incomplete structural streams and superseded recordings excluded."
    )
    temporary = collection_path.with_suffix(".json.tmp")
    write_json(temporary, collection)
    temporary.replace(collection_path)
    print(json.dumps({"episode_id": EPISODE, "datasets": entries, "totals": totals}, indent=2))


if __name__ == "__main__":
    main()
