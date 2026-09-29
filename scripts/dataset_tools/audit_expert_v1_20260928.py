#!/usr/bin/env python3
"""Check all curated expert files and both compact collection catalogs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def scan(path: Path, *, count_lines: bool = False) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    size = lines = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
            if count_lines:
                lines += chunk.count(b"\n")
    return digest.hexdigest(), size, lines


def check_file(meta: dict, base: Path, *, count_lines: bool = False,
               record_key: str = "samples") -> tuple[int, int]:
    path = base / meta["path"]
    if not path.is_file():
        raise ValueError(f"missing file: {path}")
    digest, size, lines = scan(path, count_lines=count_lines)
    if digest != meta["sha256"] or ("bytes" in meta and size != meta["bytes"]):
        raise ValueError(f"checksum/size mismatch: {path}")
    if count_lines and lines != meta[record_key]:
        raise ValueError(f"line count mismatch: {path}")
    return size, lines


def audit_curated(*, check_singles: bool = True) -> dict:
    single_root = ROOT / "datasets/expert_v1"
    compact_single_root = ROOT / "datasets/expert_v1_compact"
    teleop_root = single_root / "teleop"
    singles = json.loads((single_root / "manifest.json").read_text())
    compact_singles = json.loads((compact_single_root / "manifest.json").read_text())
    teleop = json.loads((teleop_root / "manifest.json").read_text())
    if len(singles["episodes"]) != 56 or len(compact_singles["episodes"]) != 56:
        raise ValueError("the original 56 single-task episodes are incomplete")
    if len(teleop["episodes"]) != 13:
        raise ValueError("expected 13 curated teleop episodes")
    single_files = single_bytes = compact_single_files = compact_single_bytes = 0
    if check_singles:
        for episode in singles["episodes"]:
            if not episode["canonical_success"]:
                raise ValueError(f"single-task success not certified: {episode['task']}/{episode['seed']}")
            for meta in episode["datasets"].values():
                size, _ = check_file(meta, single_root, count_lines=True)
                single_files += 1
                single_bytes += size
            check_file(episode["outcome"], single_root)
        for episode in compact_singles["episodes"]:
            directory = compact_single_root / episode["path"]
            if not directory.is_dir():
                raise ValueError(f"missing compact single-task episode: {directory}")
            for phase in episode["phases"]:
                phase_meta = json.loads((directory / f"{phase}_manifest.json").read_text())
                size, _ = check_file(phase_meta["compact"], compact_single_root,
                                     count_lines=True, record_key="records")
                compact_single_files += 1
                compact_single_bytes += size
                source = phase_meta["source"]
                source_path = single_root / source["path"]
                if not source_path.is_file() or source_path.stat().st_size != source["bytes"]:
                    raise ValueError(f"compact phase source mismatch: {source_path}")
    teleop_files = teleop_bytes = human = structural = valid_bc = hardlink_files = 0
    courses = set()
    for episode in teleop["episodes"]:
        episode_manifest = json.loads((teleop_root / episode["manifest"]).read_text())
        if episode_manifest["episode_id"] != episode["episode_id"] or ("datasets" in episode_manifest and episode_manifest["datasets"] != episode["datasets"]):
            raise ValueError(f"collection/episode disagreement: {episode['episode_id']}")
        if not episode["task_success"]:
            raise ValueError(f"uncertified teleop episode: {episode['episode_id']}")
        courses.add(episode["course"])
        for dataset in episode["datasets"]:
            size, count = check_file(dataset, teleop_root, count_lines=True)
            if dataset.get("storage") == "hardlink":
                original = ROOT / dataset["source_path"]
                curated = teleop_root / dataset["path"]
                if original.exists() and (original.stat().st_dev, original.stat().st_ino) != (curated.stat().st_dev, curated.stat().st_ino):
                    raise ValueError(f"hardlink provenance mismatch: {curated}")
                hardlink_files += 1
            teleop_files += 1
            teleop_bytes += size
            valid_bc += dataset["valid_for_behavior_cloning_samples"]
            if dataset["producer"] == "human_expert":
                human += count
            elif dataset["producer"] == "deterministic_expert":
                structural += count
            else:
                raise ValueError(f"unknown producer: {dataset['producer']}")
        print(f"[audit] curated {episode['course']}: {len(episode['datasets'])} verified streams", flush=True)
    totals = teleop["totals"]
    if any(totals[key] != actual for key, actual in {
        "episodes": 13, "human_samples": human,
        "structural_samples": structural, "samples": human + structural,
        "valid_for_behavior_cloning_samples": valid_bc, "bytes": teleop_bytes,
    }.items()):
        raise ValueError("teleop collection totals disagree with verified files")
    expected_complete = {"T02", "T03", "T05", "T06", "T14", "T15", "T10", "T16", "T09", "T07"}
    if courses != expected_complete | {"C05", "T01", "T04"}:
        raise ValueError(f"unexpected teleop courses: {courses}")
    return {
        "single_episodes": len(singles["episodes"]), "single_files": single_files,
        "single_bytes": single_bytes, "compact_single_episodes": len(compact_singles["episodes"]),
        "compact_single_files": compact_single_files, "compact_single_bytes": compact_single_bytes,
        "teleop_episodes": 13, "teleop_obstacle_complete": len(expected_complete),
        "teleop_prefixes": 3, "teleop_files": teleop_files,
        "teleop_human_samples": human, "teleop_structural_samples": structural,
        "teleop_behavior_cloning_valid_samples": valid_bc,
        "teleop_bytes": teleop_bytes, "teleop_hardlink_files": hardlink_files,
    }


def audit_compact_teleop() -> dict:
    root = ROOT / "datasets/expert_v1_compact/teleop"
    source = ROOT / "datasets/expert_v1/teleop"
    collection = json.loads((root / "manifest.json").read_text())
    if len(collection["episodes"]) != 13:
        raise ValueError("compact teleop has wrong episode count")
    source_sha, _, _ = scan(source / "manifest.json")
    if source_sha != collection["source_collection_sha256"]:
        raise ValueError("compact teleop catalog source SHA differs")
    files = records = compact_records = collapsed_records = source_bytes = compact_jsonl_bytes = archive_bytes = 0
    for episode in collection["episodes"]:
        manifest = json.loads((root / episode["manifest"]).read_text())
        if manifest["episode_id"] != episode["episode_id"]:
            raise ValueError(f"compact episode catalog mismatch: {episode['episode_id']}")
        for file in manifest["files"]:
            archive = ROOT / file["archive_path"]
            digest, size, _ = scan(archive)
            if digest != file["archive_sha256"] or size != file["archive_bytes"]:
                raise ValueError(f"compact archive checksum mismatch: {archive}")
            files += 1
            records += file["source_records"]
            compact_records += file["compact_records"]
            collapsed_records += file["collapsed_records"]
            source_bytes += file["source_bytes"]
            compact_jsonl_bytes += file["compact_jsonl_bytes"]
            if file["source_records"] != file["compact_records"] + file["collapsed_records"]:
                raise ValueError(f"teleop row count changed: {archive}")
            if file["producer"] == "human_expert" and file["collapsed_records"] != 0:
                raise ValueError(f"human pause was collapsed: {archive}")
            archive_bytes += size
        print(f"[audit] compact {episode['course']}: {len(manifest['files'])} verified archives", flush=True)
    if collection["totals"] != {
        "episodes": 13, "files": files, "samples": records,
        "compact_records": compact_records, "collapsed_records": collapsed_records,
        "source_bytes": source_bytes, "compact_jsonl_bytes": compact_jsonl_bytes,
        "archive_bytes": archive_bytes,
    }:
        raise ValueError("compact teleop totals disagree with verified archives")
    return {"compact_teleop_episodes": 13, "compact_teleop_files": files,
            "compact_teleop_samples": records, "compact_teleop_records": compact_records,
            "compact_teleop_collapsed_records": collapsed_records,
            "compact_teleop_source_bytes": source_bytes,
            "compact_teleop_archive_bytes": archive_bytes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("curated", "teleop", "compact"), required=True)
    arguments = parser.parse_args()
    print(json.dumps(audit_compact_teleop() if arguments.mode == "compact" else audit_curated(check_singles=arguments.mode == "curated"), indent=2))
