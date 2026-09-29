#!/usr/bin/env python3
"""Copy the user-confirmed T01 prefix, preserving retained JSONL bytes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
IDENTITY = "teleop-69b37b8db7e141339518d4eef585e4bd"
SOURCE = ROOT / "logs/teleop/recordings" / IDENTITY
DEST = ROOT / "datasets/expert_v1/teleop"
START = 208.79166666666666
STOP = 503.7583333333333
KEEP = (
    "teleop-snake_stairs-1790330243777101836.jsonl",
    "teleop-self_reconfiguration-1790330835248152407.jsonl",
    "teleop-self_reconfiguration-1790331579965171715.jsonl",
)
DROP = (
    "teleop-self_reconfiguration-1790329876527151074.jsonl",
    "teleop-self_reconfiguration-1790330084314205520.jsonl",
    "teleop-self_reconfiguration-1790331560242908003.jsonl",
    "teleop-self_reconfiguration-1790331924665104788.jsonl",
)
EXPECTED_GEOMETRY = "d624316f48f47e73908a6e23441e8ba9566849e6cef57a28091506e215d8e709"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def copy_rows(source, target, human=False):
    digest = hashlib.sha256()
    size = count = bc = 0
    first = last = last_next = None
    terminal_success = False
    with source.open("rb") as reader, target.open("wb") as writer:
        for line in reader:
            row = json.loads(line)
            stamp = float(row["stamp"])
            if human and stamp < START:
                continue
            if stamp >= STOP or float(row["graph_t_plus_1"]["stamp"]) >= STOP:
                if human:
                    continue
                raise ValueError(f"Structural stream exceeds cutoff: {source}")
            if len(row["graph_t"]["nodes"]) != 8 or len(row["graph_t_plus_1"]["nodes"]) != 8:
                raise ValueError(f"Incomplete graph: {source}")
            if human and row["graph_t"]["global_attributes"]["course"].get("geometry_sha256") != EXPECTED_GEOMETRY:
                raise ValueError("Wrong T01 geometry")
            writer.write(line)
            digest.update(line)
            size += len(line)
            count += 1
            bc += bool(row["supervision"]["valid_for_behavior_cloning"])
            first = stamp if first is None else first
            last = stamp
            last_next = float(row["graph_t_plus_1"]["stamp"])
            terminal_success = bool(row.get("done") and row.get("success"))
    if not count or (not human and not terminal_success):
        raise ValueError(f"No successful retained rows: {source}")
    relative = target.relative_to(DEST / "episodes" / ("." + IDENTITY + ".staging"))
    return dict(path=(Path("episodes") / IDENTITY / relative).as_posix(),
                source_path=source.relative_to(ROOT).as_posix(),
                sha256=digest.hexdigest(), bytes=size, samples=count,
                valid_for_behavior_cloning_samples=bc,
                first_stamp=first, last_stamp=last,
                max_transition_stamp=last_next,
                producer="human_expert" if human else "deterministic_expert")


def main():
    original = json.loads((SOURCE / "manifest.json").read_text())
    names = {Path(item["path"]).name for item in original["structural_streams"]}
    if original["episode_id"] != IDENTITY or original["status"] != "completed" or names != set(KEEP) | set(DROP):
        raise ValueError("Source episode or stream inventory changed")
    final = DEST / "episodes" / IDENTITY
    staging = DEST / "episodes" / ("." + IDENTITY + ".staging")
    if final.exists() or staging.exists():
        raise ValueError("Curated output already exists; inspect before rerunning")
    course_source = ROOT / "logs/teleop/runs/teleop-t01-20260925-114404.OsrpQe/course.json"
    if json.loads(course_source.read_text())["geometry_sha256"] != EXPECTED_GEOMETRY:
        raise ValueError("Source course geometry changed")
    staging.mkdir()
    (staging / "structural").mkdir()
    entries = [copy_rows(SOURCE / "human_behavior.jsonl", staging / "human_behavior.jsonl", human=True)]
    for name in KEEP:
        entries.append(copy_rows(SOURCE / "structural" / name, staging / "structural" / name))
    shutil.copyfile(SOURCE / "manifest.json", staging / "source_manifest.json")
    shutil.copyfile(course_source, staging / "course.json")
    curated = dict(schema_version="mssr.curated_teleop_episode.v1",
                   episode_id=IDENTITY, course="T01", geometry_sha256=EXPECTED_GEOMETRY,
                   source_manifest=(SOURCE / "manifest.json").relative_to(ROOT).as_posix(),
                   source_manifest_copy="source_manifest.json", course_geometry="course.json",
                   task_success=True, success_basis="user_confirmed_button_press_2026-09-25",
                   success_scope="stairs_rc_button_press_before_final_mm8_to_rc_reconfiguration",
                   full_mission_success=False,
                   retained_interval_sim_s=dict(start_inclusive=START, end_exclusive=STOP),
                   excluded_structural_streams=list(DROP), datasets=entries)
    write_json(staging / "manifest.json", curated)
    staging.rename(final)
    collection_path = DEST / "manifest.json"
    collection = json.loads(collection_path.read_text())
    collection["episodes"].append({k: curated[k] for k in
                                    ("episode_id", "course", "task_success", "success_basis",
                                     "success_scope", "full_mission_success", "geometry_sha256")}
                                  | {"manifest": f"episodes/{IDENTITY}/manifest.json", "datasets": entries})
    totals = collection["totals"]
    totals["episodes"] = len(collection["episodes"])
    totals["human_samples"] += entries[0]["samples"]
    totals["structural_samples"] += sum(e["samples"] for e in entries[1:])
    totals["samples"] += sum(e["samples"] for e in entries)
    totals["valid_for_behavior_cloning_samples"] += sum(e["valid_for_behavior_cloning_samples"] for e in entries)
    totals["bytes"] += sum(e["bytes"] for e in entries)
    collection["selection_policy"] = ("User-confirmed successful prefixes: latest C05 and T01 through "
                                      "button press. Pre-stairs tests and failed structural streams excluded; "
                                      "original recordings preserved.")
    temporary = collection_path.with_suffix(".json.tmp")
    write_json(temporary, collection)
    temporary.replace(collection_path)
    print(json.dumps({"episode": IDENTITY, "datasets": entries, "totals": totals}, indent=2))


if __name__ == "__main__":
    main()
