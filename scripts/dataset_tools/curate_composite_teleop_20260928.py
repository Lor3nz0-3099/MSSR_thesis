#!/usr/bin/env python3
"""Import the six obstacle-complete composite teleoperation demonstrations."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from teleop_archive import copy_verified_jsonl, last_jsonl_record, select_successful_streams

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "logs/teleop/recordings"
DEST_ROOT = ROOT / "datasets/expert_v1/teleop"
CAMPAIGN = (
    ("T14", "952c36750c624fb19de8421019325d13", "teleop-t14-20260927-163145.sNCjwM", 3888, 4, 0, ("button", "flat_navigation", "stairs"), (6103, 6017, 3105), "6c75d926fd3a403a00d78b5e7c6e7dd549382b651f6dde2123a3f8737b6dd311"),
    ("T15", "39e2f4b4cd684e9ea72734cc21baca94", "teleop-t15-20260927-170456.qNBeb3", 4396, 4, 1, ("button", "flat_navigation", "gap"), (6101, 5104, 4105), "8420be7b584a4b0106f15120d8f427d1877ee0449f621e7745a7e5e65b376100"),
    ("T10", "0e4e34fe3a5f4ed1b6a86642acf4fac8", "teleop-t10-20260928-111855.K4mABp", 4009, 3, 1, ("flat_navigation", "stairs", "flat_navigation"), (6017, 3105, 5100), "0e2dc553fd4215f409f8de3b3416ba284aa77ffe923a8215e6f7ee04e5307fb8"),
    ("T16", "5e0c4d97db80492bbb561b16ef132949", "teleop-t16-20260928-115957.bRqD0X", 4521, 3, 1, ("flat_navigation", "stairs", "gap"), (5100, 3101, 4103), "568b52da098f01a0be119865b4ae55a341a3258a0da4a81ae74312efa496a66b"),
    ("T09", "ce8293656bb444ac999f87c0520672fc", "teleop-t09-20260928-123435.5Z0MLA", 5049, 6, 1, ("gap", "stairs", "button"), (4107, 3101, 9974), "699232209f8529d01953964f6cf94af2c754d12469d7d1a0a5d1dc0a301b6581"),
    ("T07", "fb69637be28d49ccbff572e721cfae24", "teleop-t07-20260928-133103.jQ18MA", 6588, 5, 1, ("stairs", "flat_navigation", "button"), (6403, 5103, 8936), "b40ac07845b88068d9e49866954a4ee127a978ea56979288c0e9f13cdeed855b"),
)


def write_json_atomic(path: Path, value: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def inspect(config: tuple) -> dict:
    course_name, suffix, run_name, expected_rows, expected_keep, expected_excluded, task_types, seeds, geometry = config
    episode_id = "teleop-" + suffix
    source = SOURCE_ROOT / episode_id
    course_path = ROOT / "logs/teleop/runs" / run_name / "course.json"
    manifest = json.loads((source / "manifest.json").read_text())
    course = json.loads(course_path.read_text())
    if (manifest["episode_id"] != episode_id or manifest["status"] != "completed"
        or manifest["unwritten_records"] != 0
        or manifest["human_stream"]["records"] != expected_rows):
        raise ValueError(f"source manifest mismatch: {episode_id}")
    tasks = [x for x in course["mission"]["tasks"] if x["type"] != "goal"]
    if (course["geometry_sha256"] != geometry
        or tuple(x["type"] for x in tasks) != task_types
        or tuple(x.get("seed") for x in tasks) != seeds):
        raise ValueError(f"course changed: {episode_id}")
    actual_names = {x.name for x in (source / "structural").glob("*.jsonl")}
    recorded_names = {Path(x["path"]).name for x in manifest["structural_streams"]}
    if actual_names != recorded_names:
        raise ValueError(f"structural stream inventory mismatch: {episode_id}")
    keep, excluded = select_successful_streams(source / "structural")
    if len(keep) != expected_keep or len(excluded) != expected_excluded:
        raise ValueError(f"terminal stream inventory changed: {episode_id}")
    last = last_jsonl_record(source / "human_behavior.jsonl")
    graph = last["graph_t"]
    nodes = graph["nodes"]
    if len(nodes) != 8 or graph["global_attributes"]["course"]["geometry_sha256"] != geometry:
        raise ValueError(f"final graph/course mismatch: {episode_id}")
    positions = [x["attributes"]["position"] for x in nodes]
    evidence = {"obstacles": [x["type"] for x in tasks], "seeds": list(seeds),
                "last_human_stamp": last["stamp"], "goal_required": False,
                "human_terminal_flag_recorded": bool(last["done"] or last["is_terminal"])}
    for task in tasks:
        if task["type"] == "button":
            depression = task["parameters"]["button"].get("depression_m")
            observed = next(x["parameters"]["button"]["depression_m"]
                            for x in graph["global_attributes"]["course"]["mission"]["tasks"]
                            if x["task_id"] == task["task_id"])
            if observed < 0.0035:
                raise ValueError(f"button not pressed in final state: {episode_id}")
            evidence["button_depression_m_final"] = observed
        if task["type"] == "gap":
            gap = task["parameters"]["gap"]
            far = gap["far_edge_center_world_xyz_m"]
            direction = gap["crossing_direction_world_xy"]
            progress = [((p[0]-far[0])*direction[0] + (p[1]-far[1])*direction[1]) for p in positions]
            if min(progress) <= 0:
                raise ValueError(f"module has not passed gap far edge: {episode_id}")
            evidence["minimum_distance_past_gap_far_edge_m"] = min(progress)
        if task["type"] == "stairs":
            top = task["parameters"]["stairs"]["top_heights_world_m"][-1]
            if min(p[2] for p in positions) < top + 0.02:
                raise ValueError(f"module not on upper stair level at end: {episode_id}")
            evidence["minimum_final_module_z_m"] = min(p[2] for p in positions)
    goal_boxes = [x for x in course["collision_boxes"] if x["semantic"] == "goal_platform"]
    goal_observed = False
    if goal_boxes:
        box = goal_boxes[0]
        gx, gy = box["center_xyz_m"][:2]
        sx, sy = box["size_xyz_m"][:2]
        goal_observed = all(abs(p[0]-gx) <= sx/2 and abs(p[1]-gy) <= sy/2 for p in positions)
    evidence["all_modules_inside_goal_box_at_last_sample"] = goal_observed
    return {"config": config, "episode_id": episode_id, "source": source,
            "course_path": course_path, "manifest": manifest, "course": course,
            "keep": keep, "excluded": excluded, "evidence": evidence}


def copy_episode(info: dict, collection: dict) -> None:
    episode_id = info["episode_id"]
    final = DEST_ROOT / "episodes" / episode_id
    staging = final.with_name("." + episode_id + ".staging")
    if final.exists() or staging.exists():
        raise ValueError(f"episode destination or staging already exists: {episode_id}")
    staging.mkdir()
    (staging / "structural").mkdir()
    try:
        entries = []
        source = info["source"]
        inputs = [(source / "human_behavior.jsonl", staging / "human_behavior.jsonl", "human_expert", info["manifest"]["human_stream"]["records"])]
        inputs += [(p, staging / "structural" / p.name, "deterministic_expert", None) for p in info["keep"]]
        for source_path, target_path, producer, expected in inputs:
            print(f"[{info['config'][0]}] copying {source_path.name}", flush=True)
            metadata = copy_verified_jsonl(source_path, target_path, producer=producer, expected_records=expected, storage="hardlink")
            metadata.update({
                "path": (Path("episodes") / episode_id / target_path.relative_to(staging)).as_posix(),
                "source_path": source_path.relative_to(ROOT).as_posix(),
            })
            entries.append(metadata)
        shutil.copy2(source / "manifest.json", staging / "source_manifest.json")
        shutil.copy2(info["course_path"], staging / "course.json")
        curated = {
            "schema_version": "mssr.curated_teleop_episode.v1",
            "episode_id": episode_id,
            "course": info["config"][0],
            "geometry_sha256": info["course"]["geometry_sha256"],
            "source_manifest": (source / "manifest.json").relative_to(ROOT).as_posix(),
            "source_manifest_copy": "source_manifest.json",
            "course_geometry": "course.json",
            "task_success": True,
            "success_basis": "operator_confirmed_obstacle_course_with_terminal_macros_and_physical_button_gap_stair_evidence",
            "success_scope": "all_course_obstacles_completed_goal_optional",
            "full_mission_success": info["evidence"]["all_modules_inside_goal_box_at_last_sample"],
            "evidence": info["evidence"],
            "excluded_structural_streams": info["excluded"],
            "datasets": entries,
        }
        write_json_atomic(staging / "manifest.json", curated)
        staging.rename(final)
    except Exception:
        shutil.rmtree(staging)
        raise
    collection["episodes"].append({
        key: curated[key] for key in (
            "episode_id", "course", "task_success", "success_basis",
            "success_scope", "full_mission_success", "geometry_sha256",
        )
    } | {"manifest": f"episodes/{episode_id}/manifest.json", "datasets": entries})
    collection["totals"] = {
        "episodes": len(collection["episodes"]),
        "human_samples": sum(e["datasets"][0]["samples"] for e in collection["episodes"]),
        "structural_samples": sum(sum(d["samples"] for d in e["datasets"][1:]) for e in collection["episodes"]),
        "samples": sum(sum(d["samples"] for d in e["datasets"]) for e in collection["episodes"]),
        "valid_for_behavior_cloning_samples": sum(sum(d["valid_for_behavior_cloning_samples"] for d in e["datasets"]) for e in collection["episodes"]),
        "bytes": sum(sum(d["bytes"] for d in e["datasets"]) for e in collection["episodes"]),
    }
    collection["selection_policy"] = (
        "Confirmed obstacle-complete composite teleop: T02/T03/T05/T06/T14/T15/T10/T16/T09/T07; "
        "goal platform optional for new campaign. C05/T01/T04 are retained successful prefixes. "
        "Superseded, failed, and nonterminal structural streams excluded."
    )
    write_json_atomic(DEST_ROOT / "manifest.json", collection)
    print(f"[{info['config'][0]}] imported {episode_id}: {len(entries)} streams", flush=True)


def main() -> None:
    infos = [inspect(config) for config in CAMPAIGN]
    collection = json.loads((DEST_ROOT / "manifest.json").read_text())
    present = {item["episode_id"] for item in collection["episodes"]}
    for info in infos:
        if info["episode_id"] in present:
            final = DEST_ROOT / "episodes" / info["episode_id"]
            if not final.is_dir():
                raise ValueError(f"collection contains missing episode: {final}")
            print(f"[{info['config'][0]}] already imported", flush=True)
            continue
        copy_episode(info, collection)
        present.add(info["episode_id"])


if __name__ == "__main__":
    main()
