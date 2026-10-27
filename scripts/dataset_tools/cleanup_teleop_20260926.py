#!/usr/bin/env python3
"""Remove rejected teleop recordings and unrecorded failed macros.

Successful standalone structural macros are deliberately retained.
Curated C05/T02/T03 raw directories are deleted only after hash checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


ROOT = Path(__file__).resolve().parents[2]
RECORDINGS = ROOT / "logs/teleop/recordings"
LOOSE = ROOT / "logs/teleop/structural"
COLLECTION = ROOT / "datasets/expert_v1/teleop"
AUDIT = ROOT / "logs/teleop/cleanup_20260926.json"
OLD_IDS = {
    "teleop-02834a4e6032413bbde512908369ccda",
    "teleop-1212fbae85c24e2aa7c00dc568ab1220",
    "teleop-147c0090c5c14616a92c8be8ab36f46b",
    "teleop-18c036c604cd43519a6d336838ae69b8",
    "teleop-22c95ec71e4349cbb14404093662d399",
    "teleop-31a6d3009964476e8847e11b26bd8517",
    "teleop-4a51aeb8956949bbb940fe145cf3c6fd",
    "teleop-4c687c3fb7a64cada5c104856ed3430a",
    "teleop-4cacc923cd1e40cdac3ea46ac5534488",
    "teleop-4f93e89c0ac545dc862667b0f8c2dbad",
    "teleop-52a740e82fed4f06b5a25591ccf8c212",
    "teleop-5725226e89f244eeb4b990ed9cd71952",
    "teleop-5bdc7d63d8444e1c8b56ae801e64257a",
    "teleop-6a057860dd6549bc9f693ee271f1bd1b",
    "teleop-6a643e77d765420689f6bb8f7fb9c223",
    "teleop-72fd818cab344c858fd5e10ba418c562",
    "teleop-84f642f112ce4099887b20c2f5d0d697",
    "teleop-917ae978d62543d798007b7b96950b01",
    "teleop-9caf32bec99e49459775e65dbbfeea30",
    "teleop-abf57e09fe3046a28fd1107506f27bcb",
    "teleop-b928cb62435447e7b3fc6291cb917956",
    "teleop-c3258f139ca044c3a708b309152335e8",
    "teleop-d0d44d40ebf242a8970ec7c3fdda2b64",
    "teleop-e3f9c3cd6f21496c88160d05a2c3cbc4",
    "teleop-e537d8359dcf4358ab424e500bd20066",
}
CURATED_PREFIX_ID = "teleop-adcf33433f954669838220b9f617d952"  # T04
EXACT_DUPLICATE_IDS = {
    "teleop-f8888f12c46249fc80ef0af8c56b4dcf",  # latest C05
    "teleop-1bea8f28c1c04323a4c3ec7e2829d07c",  # T02
    "teleop-3aa4f69128774d5eba1f8d851c02cf53",  # T03
}
MAX_LOOSE_MTIME = 1790421000.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def last_jsonl(path: Path) -> dict:
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 2_000_000))
        lines = handle.read().splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except ValueError:
            pass
    raise ValueError(f"No valid JSONL row: {path}")


def folder_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def verify_duplicate(episode_id: str, collection: dict) -> None:
    episode = next(
        item for item in collection["episodes"]
        if item["episode_id"] == episode_id
    )
    manifest = COLLECTION / episode["manifest"]
    metadata = json.loads(manifest.read_text())
    source = RECORDINGS / episode_id
    if episode_id.startswith("teleop-f888"):
        metadata_copies = {
            "manifest.json": "source.manifest.json",
            "manifest.interrupted-original.json": "source.manifest.interrupted-original.json",
            "curation.json": "source.curation.json",
        }
        for original_name, copy_name in metadata_copies.items():
            if (source / original_name).read_bytes() != (manifest.parent / copy_name).read_bytes():
                raise ValueError(f"C05 metadata copy differs: {original_name}")
    elif (source / "manifest.json").read_bytes() != (
        manifest.parent / metadata["source_manifest_copy"]
    ).read_bytes():
        raise ValueError(f"Original manifest differs: {episode_id}")
    for item in episode["datasets"]:
        raw = ROOT / item["source_path"]
        curated = COLLECTION / item["path"]
        if raw.parent != source and source not in raw.parents:
            raise ValueError(f"Source outside episode: {raw}")
        if raw.stat().st_size != item["bytes"]:
            raise ValueError(f"Non-exact raw source: {raw}")
        if sha256(raw) != item["sha256"] or sha256(curated) != item["sha256"]:
            raise ValueError(f"Hash mismatch: {raw}")
    if episode_id.startswith("teleop-f888"):
        return
    if sha256(manifest.parent / metadata["course_geometry"]) != sha256(
        ROOT / "logs/teleop/runs" /
        ("teleop-t02-20260925-144049.yJK53B/course.json"
         if episode_id.startswith("teleop-1bea")
         else "teleop-t03-20260925-154622.thMRXr/course.json")
    ):
        raise ValueError(f"Course copy mismatch: {episode_id}")


def verify_t04_prefix(collection: dict) -> None:
    episode = next(
        item for item in collection["episodes"]
        if item["episode_id"] == CURATED_PREFIX_ID
    )
    manifest = COLLECTION / episode["manifest"]
    metadata = json.loads(manifest.read_text())
    source = RECORDINGS / CURATED_PREFIX_ID
    if metadata["success_scope"] != "t04_stairs_and_rc_course_prefix_before_rc_to_snake":
        raise ValueError("Unexpected T04 curation scope")
    if (source / "manifest.json").read_bytes() != (
        manifest.parent / metadata["source_manifest_copy"]
    ).read_bytes():
        raise ValueError("T04 source manifest copy differs")
    course = ROOT / "logs/teleop/runs/teleop-t04-20260926-111017.x21y1J/course.json"
    if course.read_bytes() != (manifest.parent / metadata["course_geometry"]).read_bytes():
        raise ValueError("T04 course copy differs")
    for item in episode["datasets"]:
        raw = ROOT / item["source_path"]
        curated = COLLECTION / item["path"]
        if sha256(curated) != item["sha256"]:
            raise ValueError(f"T04 dataset hash differs: {curated}")
        if raw.stat().st_size == curated.stat().st_size:
            if sha256(raw) != item["sha256"]:
                raise ValueError(f"T04 structural source differs: {raw}")
        elif item["producer"] == "human_expert":
            with raw.open("rb") as reader, curated.open("rb") as expected:
                while chunk := expected.read(8 * 1024 * 1024):
                    if reader.read(len(chunk)) != chunk:
                        raise ValueError("T04 selected human prefix differs from raw")
                next_row = reader.readline()
            if not next_row or float(json.loads(next_row)["stamp"]) < metadata["cutoff_sim_s_exclusive"]:
                raise ValueError("T04 source boundary differs")
        else:
            raise ValueError(f"T04 selected structural stream is not exact: {raw}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    collection = json.loads((COLLECTION / "manifest.json").read_text())
    curated = {item["episode_id"] for item in collection["episodes"]}
    if OLD_IDS & curated or not EXACT_DUPLICATE_IDS | {CURATED_PREFIX_ID} <= curated:
        raise ValueError("Episode inventory changed")
    if {p.name for p in RECORDINGS.iterdir() if p.is_dir()} - (
        OLD_IDS | curated
    ):
        raise ValueError("Unknown recording directory appeared; review before cleanup")
    for episode_id in EXACT_DUPLICATE_IDS:
        verify_duplicate(episode_id, collection)
    verify_t04_prefix(collection)

    targets = []
    for episode_id in sorted(OLD_IDS | EXACT_DUPLICATE_IDS | {CURATED_PREFIX_ID}):
        path = RECORDINGS / episode_id
        if not path.is_dir() or path.is_symlink():
            raise ValueError(f"Missing or linked raw episode: {path}")
        targets.append({
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": folder_bytes(path),
            "reason": (
                "exact_duplicate_of_curated_episode"
                if episode_id in EXACT_DUPLICATE_IDS
                else "selected_prefix_verified_and_remaining_t04_rejected"
                if episode_id == CURATED_PREFIX_ID
                else "unselected_or_failed_episode"
            ),
        })
    retained_successful_loose = 0
    for path in sorted(LOOSE.glob("*.jsonl")):
        if path.is_symlink() or path.stat().st_mtime > MAX_LOOSE_MTIME:
            raise ValueError(f"Unexpected loose macro file: {path}")
        last = last_jsonl(path)
        if last.get("done") is True and last.get("success") is True:
            retained_successful_loose += 1
            continue
        targets.append({
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "reason": "no_terminal_success_outside_START",
        })

    report = {
        "schema_version": "mssr.teleop_cleanup.v1",
        "created_at": time.time(),
        "applied": args.apply,
        "retained_curated_episode_ids": sorted(curated),
        "retained_successful_standalone_macros": retained_successful_loose,
        "targets": targets,
        "total_bytes": sum(item["bytes"] for item in targets),
    }
    print(json.dumps({
        "apply": args.apply,
        "recording_directories": sum(
            item["path"].startswith("logs/teleop/recordings/")
            for item in targets
        ),
        "failed_loose_macros": sum(
            item["path"].startswith("logs/teleop/structural/")
            for item in targets
        ),
        "GiB": round(report["total_bytes"] / 2**30, 2),
        "retained_successful_standalone_macros": retained_successful_loose,
    }, indent=2))
    if not args.apply:
        return
    if AUDIT.exists():
        raise ValueError(f"Audit already exists: {AUDIT}")
    for item in targets:
        path = ROOT / item["path"]
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    AUDIT.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
