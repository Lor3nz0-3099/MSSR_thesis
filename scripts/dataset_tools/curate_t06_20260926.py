#!/usr/bin/env python3
"""Import the completed T06 gap, RC route, button and final RC goal."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
EPISODE = "teleop-a7d7e08f5fd14a59bd3af6afb53ccde1"
SOURCE = ROOT / "logs/teleop/recordings" / EPISODE
COURSE = ROOT / "logs/teleop/runs/teleop-t06-20260926-161251.6XFOl1/course.json"
DEST = ROOT / "datasets/expert_v1/teleop"
GEOMETRY = "5367c477e244c69c1f6e0aa658d0dca59dd021d7b4cf69ded6780d39186063e3"
KEEP = (
    "teleop-self_reconfiguration-1790432260992632686.jsonl",
    "teleop-snake_gap-1790432423631935672.jsonl",
    "teleop-self_reconfiguration-1790432652216957795.jsonl",
    "teleop-self_reconfiguration-1790433134902397672.jsonl",
    "teleop-self_reconfiguration-1790433329997872459.jsonl",
)
EXPECTED_PHASES = (
    ("rc_car8", None),
    ("snake8", None),
    ("rc_car8", None),
    ("mobile_manipulator8", "drive_ready"),
    ("mobile_manipulator8", "to_manipulation_ready"),
    ("mobile_manipulator8", "manipulation_ready"),
    ("mobile_manipulator8", "to_drive_ready"),
    ("mobile_manipulator8", "drive_ready"),
    ("rc_car8", None),
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def copy_stream(source: Path, target: Path, *, human: bool, gap_far_x: float, goal_box: dict) -> tuple[dict, dict]:
    digest = hashlib.sha256()
    stats = Counter()
    first_stamp = last_stamp = None
    phases = []
    gap_crossed_at = button_pressed_at = goal_reached_at = None
    gx, gy = goal_box["center_xyz_m"][:2]
    sx, sy = goal_box["size_xyz_m"][:2]
    last_row = None
    with source.open("rb") as reader, target.open("wb") as writer:
        for line in reader:
            row = json.loads(line)
            stamp = float(row["stamp"])
            if last_stamp is not None and stamp < last_stamp:
                raise ValueError(f"Nonmonotonic T06 stream: {source}")
            if len(row["graph_t"]["nodes"]) != 8 or len(row["graph_t_plus_1"]["nodes"]) != 8:
                raise ValueError(f"Incomplete eight-module graph: {source}")
            if first_stamp is None:
                first_stamp = stamp
            last_stamp = stamp
            last_row = row
            stats["samples"] += 1
            stats["bytes"] += len(line)
            stats["valid_for_behavior_cloning_samples"] += bool(row["supervision"]["valid_for_behavior_cloning"])
            if human:
                if not row["action_valid"]:
                    raise ValueError("Invalid T06 human action")
                course = row["graph_t"]["global_attributes"]["course"]
                next_course = row["graph_t_plus_1"]["global_attributes"]["course"]
                if course.get("geometry_sha256") != GEOMETRY or next_course.get("geometry_sha256") != GEOMETRY:
                    raise ValueError("T06 course geometry changed")
                phase = (row["observation"]["morphology"], row["observation"]["intent"].get("mode"))
                if not phases or phase != phases[-1]:
                    phases.append(phase)
                points = [node["attributes"]["position"] for node in row["graph_t"]["nodes"]]
                if gap_crossed_at is None and all(point[0] > gap_far_x for point in points):
                    gap_crossed_at = stamp
                if goal_reached_at is None and all(
                    abs(point[0] - gx) <= sx / 2 and abs(point[1] - gy) <= sy / 2
                    for point in points
                ):
                    goal_reached_at = stamp
                button = next(task["parameters"]["button"] for task in course["mission"]["tasks"] if task["type"] == "button")
                if button_pressed_at is None and button["depression_m"] >= 0.0035:
                    if phase[0] != "mobile_manipulator8":
                        raise ValueError("T06 button press outside MM8 phase")
                    button_pressed_at = stamp
            writer.write(line)
            digest.update(line)
    if not stats["samples"] or sha256(target) != digest.hexdigest():
        raise ValueError(f"T06 copy differs: {source}")
    evidence = {}
    if human:
        if stats["samples"] != 4430 or tuple(phases) != EXPECTED_PHASES:
            raise ValueError("T06 human stream or morphology sequence changed")
        if not all(value is not None for value in (gap_crossed_at, button_pressed_at, goal_reached_at)):
            raise ValueError("T06 lacks gap, button or goal evidence")
        if not (gap_crossed_at < button_pressed_at < goal_reached_at):
            raise ValueError("T06 task evidence is out of order")
        if last_row["observation"]["morphology"] != "rc_car8":
            raise ValueError("T06 does not finish in RC car")
        evidence = {
            "all_modules_cross_gap_at_sim_s": gap_crossed_at,
            "button_physically_pressed_at_sim_s": button_pressed_at,
            "all_modules_on_goal_platform_at_sim_s": goal_reached_at,
            "morphology_and_mm8_modes": [list(phase) for phase in phases],
            "human_terminal_flag_recorded": bool(last_row["is_terminal"]),
        }
    elif not (last_row.get("done") is True and last_row.get("success") is True):
        raise ValueError(f"T06 macro lacks terminal success: {source}")
    relative = target.relative_to(DEST / "episodes" / f".{EPISODE}.staging")
    return {
        "path": (Path("episodes") / EPISODE / relative).as_posix(),
        "source_path": source.relative_to(ROOT).as_posix(),
        "sha256": digest.hexdigest(),
        "bytes": stats["bytes"],
        "samples": stats["samples"],
        "valid_for_behavior_cloning_samples": stats["valid_for_behavior_cloning_samples"],
        "first_stamp": first_stamp,
        "last_stamp": last_stamp,
        "producer": "human_expert" if human else "deterministic_expert",
    }, evidence


def main() -> None:
    original = json.loads((SOURCE / "manifest.json").read_text())
    if (original["episode_id"] != EPISODE or original["status"] != "completed"
            or original["unwritten_records"] != 0 or original["human_stream"]["records"] != 4430
            or {Path(item["path"]).name for item in original["structural_streams"]} != set(KEEP)):
        raise ValueError("T06 source inventory changed")
    course = json.loads(COURSE.read_text())
    if course["geometry_sha256"] != GEOMETRY or [task["type"] for task in course["mission"]["tasks"]] != [
        "gap", "flat_navigation", "button", "goal"
    ]:
        raise ValueError("T06 course changed")
    gap = next(task["parameters"]["gap"] for task in course["mission"]["tasks"] if task["type"] == "gap")
    far_x = gap["far_edge_center_world_xyz_m"][0]
    goal_box = next(box for box in course["collision_boxes"] if box["semantic"] == "goal_platform")
    final = DEST / "episodes" / EPISODE
    staging = DEST / "episodes" / f".{EPISODE}.staging"
    collection_path = DEST / "manifest.json"
    collection = json.loads(collection_path.read_text())
    if final.exists() or staging.exists() or any(item["episode_id"] == EPISODE for item in collection["episodes"]):
        raise ValueError("T06 already imported; inspect before rerunning")
    staging.mkdir()
    (staging / "structural").mkdir()
    try:
        human, evidence = copy_stream(SOURCE / "human_behavior.jsonl", staging / "human_behavior.jsonl", human=True, gap_far_x=far_x, goal_box=goal_box)
        entries = [human]
        for name in KEEP:
            entry, _ = copy_stream(SOURCE / "structural" / name, staging / "structural" / name, human=False, gap_far_x=far_x, goal_box=goal_box)
            entries.append(entry)
        shutil.copyfile(SOURCE / "manifest.json", staging / "source_manifest.json")
        shutil.copyfile(COURSE, staging / "course.json")
        curated = {
            "schema_version": "mssr.curated_teleop_episode.v1",
            "episode_id": EPISODE,
            "course": "T06",
            "geometry_sha256": GEOMETRY,
            "source_manifest": (SOURCE / "manifest.json").relative_to(ROOT).as_posix(),
            "source_manifest_copy": "source_manifest.json",
            "course_geometry": "course.json",
            "task_success": True,
            "success_basis": "physical_gap_crossing_button_depression_goal_platform_and_all_five_structural_terminal_successes",
            "success_scope": "complete_t06_gap_rc_button_final_rc_goal",
            "full_mission_success": True,
            "evidence": evidence,
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
        totals["valid_for_behavior_cloning_samples"] += sum(item["valid_for_behavior_cloning_samples"] for item in entries)
        totals["bytes"] += sum(item["bytes"] for item in entries)
        collection["selection_policy"] = (
            "Confirmed successful teleop scopes only: latest C05, T01 through button, "
            "complete T02/T03/T05/T06, and T04 stairs/RC prefix before RC-to-Snake. "
            "Failed, superseded and uncertified segments excluded."
        )
        temporary = collection_path.with_suffix(".json.tmp")
        write_json(temporary, collection)
        temporary.replace(collection_path)
        print(json.dumps({"episode_id": EPISODE, "evidence": evidence, "datasets": entries, "totals": totals}, indent=2))
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    main()
