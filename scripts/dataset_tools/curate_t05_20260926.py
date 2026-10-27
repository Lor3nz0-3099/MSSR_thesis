#!/usr/bin/env python3
"""Import the completed, physically verified T05 teleoperation."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
EPISODE = "teleop-5bb5a4712d924617aab5604b6deb20b4"
SOURCE = ROOT / "logs/teleop/recordings" / EPISODE
COURSE = ROOT / "logs/teleop/runs/teleop-t05-20260926-150224.ZCxGOJ/course.json"
DEST = ROOT / "datasets/expert_v1/teleop"
COURSE_GEOMETRY = "6f9ea17f49f82b0680d54e8c4b9ed1427875c8a5da8a444babf5fcedabc083aa"
GRAPH_GEOMETRY = "c56515ccf21f1ab91e0057782efb3a599f377700e3277fac68f328050ff4c2f5"
KEEP = (
    "teleop-self_reconfiguration-1790428537034857531.jsonl",
    "teleop-self_reconfiguration-1790428703478581397.jsonl",
)
EXPECTED_PHASES = (
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


def assert_same_static_geometry(observed: dict, recorded: dict) -> None:
    """Ignore live plunger fields and harmless JSON float roundoff."""
    def walk(left, right, path=""):
        if path in {"/geometry_sha256", "/mission/geometry_sha256"}:
            return
        if path.endswith("/parameters/button/current_center_xyz_m") or path.endswith(
            "/parameters/button/depression_m"
        ):
            return
        if isinstance(left, dict) and isinstance(right, dict):
            extra = set(left) - set(right)
            expected_extra = (
                {"current_center_xyz_m", "depression_m"}
                if path.endswith("/parameters/button") else set()
            )
            if extra != expected_extra or set(right) - set(left):
                raise ValueError(f"T05 static geometry keys differ at {path}")
            for key in right:
                walk(left[key], right[key], path + "/" + key)
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                raise ValueError(f"T05 static geometry length differs at {path}")
            for index, (a, b) in enumerate(zip(left, right)):
                walk(a, b, path + f"/{index}")
        elif isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
            if not math.isclose(left, right, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"T05 static geometry value differs at {path}")
        elif left != right:
            raise ValueError(f"T05 static geometry value differs at {path}")
    walk(observed, recorded)


def copy_stream(source: Path, target: Path, *, human: bool, course: dict) -> tuple[dict, dict]:
    digest = hashlib.sha256()
    stats = Counter()
    first_stamp = last_stamp = None
    phases = []
    first_graph_course = None
    button_pressed_at = goal_reached_at = route1_at = route2_at = None
    goal = next(box for box in course["collision_boxes"] if box["semantic"] == "goal_platform")
    gx, gy = goal["center_xyz_m"][:2]
    sx, sy = goal["size_xyz_m"][:2]
    exits = [boundary["exit_world_xyz_m"][0] for boundary in course["boundaries"][:2]]
    last_row = None
    with source.open("rb") as reader, target.open("wb") as writer:
        for line in reader:
            row = json.loads(line)
            stamp = float(row["stamp"])
            if last_stamp is not None and stamp < last_stamp:
                raise ValueError(f"Nonmonotonic T05 stream: {source}")
            if len(row["graph_t"]["nodes"]) != 8 or len(row["graph_t_plus_1"]["nodes"]) != 8:
                raise ValueError(f"Incomplete T05 graph: {source}")
            if first_stamp is None:
                first_stamp = stamp
            last_stamp = stamp
            last_row = row
            stats["samples"] += 1
            stats["bytes"] += len(line)
            stats["valid_for_behavior_cloning_samples"] += bool(row["supervision"]["valid_for_behavior_cloning"])
            if human:
                if not row["action_valid"]:
                    raise ValueError("Invalid T05 human action")
                graph_course = row["graph_t"]["global_attributes"]["course"]
                next_course = row["graph_t_plus_1"]["global_attributes"]["course"]
                if graph_course.get("geometry_sha256") != GRAPH_GEOMETRY or next_course.get("geometry_sha256") != GRAPH_GEOMETRY:
                    raise ValueError("T05 graph geometry hash changed")
                if first_graph_course is None:
                    first_graph_course = graph_course
                    assert_same_static_geometry(graph_course, course)
                mode = row["observation"]["intent"].get("mode")
                phase = (row["observation"]["morphology"], mode)
                if not phases or phase != phases[-1]:
                    phases.append(phase)
                points = [node["attributes"]["position"] for node in row["graph_t"]["nodes"]]
                if route1_at is None and all(point[0] > exits[0] for point in points):
                    route1_at = stamp
                if route2_at is None and all(point[0] > exits[1] for point in points):
                    route2_at = stamp
                if goal_reached_at is None and all(
                    abs(point[0] - gx) <= sx / 2 and abs(point[1] - gy) <= sy / 2
                    for point in points
                ):
                    goal_reached_at = stamp
                button = next(task["parameters"]["button"] for task in graph_course["mission"]["tasks"] if task["type"] == "button")
                if button_pressed_at is None and button["depression_m"] >= 0.0035:
                    if phase[0] != "mobile_manipulator8":
                        raise ValueError("T05 button press occurred outside MM8 phase")
                    button_pressed_at = stamp
            writer.write(line)
            digest.update(line)
    if not stats["samples"] or sha256(target) != digest.hexdigest():
        raise ValueError(f"Copied T05 stream differs: {source}")
    evidence = {}
    if human:
        if stats["samples"] != 5478 or tuple(phases) != EXPECTED_PHASES:
            raise ValueError("T05 human stream or MM8 return sequence changed")
        if not all(value is not None for value in (route1_at, route2_at, button_pressed_at, goal_reached_at)):
            raise ValueError("T05 lacks route, button or goal evidence")
        if not (route1_at < button_pressed_at < route2_at < goal_reached_at):
            raise ValueError("T05 task evidence is out of order")
        if last_row["observation"]["morphology"] != "rc_car8":
            raise ValueError("T05 does not finish in RC car")
        evidence = {
            "first_route_all_modules_passed_at_sim_s": route1_at,
            "button_physically_pressed_at_sim_s": button_pressed_at,
            "second_route_all_modules_passed_at_sim_s": route2_at,
            "all_modules_on_goal_platform_at_sim_s": goal_reached_at,
            "mm8_modes": [list(phase) for phase in phases],
            "human_terminal_flag_recorded": bool(last_row["is_terminal"]),
        }
    elif not (last_row.get("done") is True and last_row.get("success") is True):
        raise ValueError(f"T05 structural stream lacks terminal success: {source}")
    relative = target.relative_to(DEST / "episodes" / f".{EPISODE}.staging")
    entry = {
        "path": (Path("episodes") / EPISODE / relative).as_posix(),
        "source_path": source.relative_to(ROOT).as_posix(),
        "sha256": digest.hexdigest(),
        "bytes": stats["bytes"],
        "samples": stats["samples"],
        "valid_for_behavior_cloning_samples": stats["valid_for_behavior_cloning_samples"],
        "first_stamp": first_stamp,
        "last_stamp": last_stamp,
        "producer": "human_expert" if human else "deterministic_expert",
    }
    return entry, evidence


def main() -> None:
    original = json.loads((SOURCE / "manifest.json").read_text())
    if (original["episode_id"] != EPISODE or original["status"] != "completed"
            or original["unwritten_records"] != 0 or original["human_stream"]["records"] != 5478
            or {Path(item["path"]).name for item in original["structural_streams"]} != set(KEEP)):
        raise ValueError("T05 source manifest changed")
    course = json.loads(COURSE.read_text())
    if course["geometry_sha256"] != COURSE_GEOMETRY or [task["type"] for task in course["mission"]["tasks"]] != [
        "flat_navigation", "flat_navigation", "button", "goal"
    ]:
        raise ValueError("T05 course changed")
    final = DEST / "episodes" / EPISODE
    staging = DEST / "episodes" / f".{EPISODE}.staging"
    collection_path = DEST / "manifest.json"
    collection = json.loads(collection_path.read_text())
    if final.exists() or staging.exists() or any(item["episode_id"] == EPISODE for item in collection["episodes"]):
        raise ValueError("T05 already imported; inspect before rerunning")
    staging.mkdir()
    (staging / "structural").mkdir()
    try:
        human, evidence = copy_stream(SOURCE / "human_behavior.jsonl", staging / "human_behavior.jsonl", human=True, course=course)
        entries = [human]
        for name in KEEP:
            entry, _ = copy_stream(SOURCE / "structural" / name, staging / "structural" / name, human=False, course=course)
            entries.append(entry)
        shutil.copyfile(SOURCE / "manifest.json", staging / "source_manifest.json")
        shutil.copyfile(COURSE, staging / "course.json")
        curated = {
            "schema_version": "mssr.curated_teleop_episode.v1",
            "episode_id": EPISODE,
            "course": "T05",
            "geometry_sha256": COURSE_GEOMETRY,
            "graph_geometry_sha256": GRAPH_GEOMETRY,
            "geometry_hash_note": "Observed course differs from course.json only by sub-picometer cone-coordinate roundoff and live button state; static geometry verified numerically to 1e-12 m.",
            "source_manifest": (SOURCE / "manifest.json").relative_to(ROOT).as_posix(),
            "source_manifest_copy": "source_manifest.json",
            "course_geometry": "course.json",
            "task_success": True,
            "success_basis": "two_rc_routes_button_depression_mm8_restore_both_reconfigurations_terminal_success_and_goal_platform",
            "success_scope": "complete_t05_two_rc_routes_button_mm8_return_final_rc_goal",
            "full_mission_success": True,
            "evidence": evidence,
            "datasets": entries,
        }
        write_json(staging / "manifest.json", curated)
        staging.rename(final)
        collection["episodes"].append({
            key: curated[key] for key in (
                "episode_id", "course", "task_success", "success_basis", "success_scope",
                "full_mission_success", "geometry_sha256", "graph_geometry_sha256",
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
            "complete T02/T03/T05, and T04 stairs/RC prefix before RC-to-Snake. "
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
