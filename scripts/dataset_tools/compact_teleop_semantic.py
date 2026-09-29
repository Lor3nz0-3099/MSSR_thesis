#!/usr/bin/env python3
"""Compact human transitions individually and collapse exact macro repeats."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from teleop_archive import sha256_file
from teleop_semantic_codec import (TeleopCodec, semantically_equal,
                                   temporal_fields, with_temporal_values)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "datasets/expert_v1/teleop"
DEST = ROOT / "datasets/expert_v1_compact/teleop"


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def compact_stream(source: Path, archive: Path, source_meta: dict) -> dict:
    """Validate each source row and reversibly collapse deterministic repeats."""
    producer = source_meta.get("producer", "human_expert")
    if producer not in ("human_expert", "deterministic_expert"):
        raise ValueError(f"unknown teleop producer: {producer}")
    source_digest = hashlib.sha256()
    compact_digest = hashlib.sha256()
    source_bytes = compact_bytes = records = compact_records = reference_count = 0
    codec = None
    representative = pending = None
    pending_paths = None
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("wb") as writer:
        encoder = subprocess.Popen(["zstd", "-T2", "-3", "-q", "-c"],
                                   stdin=subprocess.PIPE, stdout=writer)
        assert encoder.stdin is not None

        def write_compact(value: dict) -> None:
            nonlocal compact_bytes, compact_records
            packed = (json.dumps(value, ensure_ascii=False,
                                 separators=(",", ":")) + "\n").encode("utf-8")
            compact_digest.update(packed)
            compact_bytes += len(packed)
            encoder.stdin.write(packed)
            compact_records += 1

        try:
            with source.open("rb") as reader:
                for line in reader:
                    if not line.endswith(b"\n") or not line.strip():
                        raise ValueError(f"bad JSONL framing: {source}:{records+1}")
                    source_digest.update(line)
                    source_bytes += len(line)
                    original = json.loads(line)
                    if codec is None:
                        codec = TeleopCodec.from_first_record(original)
                    compact = codec.encode(original)
                    if codec.decode(compact) != original:
                        raise ValueError(f"codec changed a transition: {source}:{records+1}")
                    reference_count += compact["__teleop_compact__"]["mask"].bit_count()

                    if producer == "human_expert":
                        write_compact(compact)
                    else:
                        paths, values = temporal_fields(compact)
                        if (representative is not None
                                and paths == pending_paths
                                and semantically_equal(representative, compact)):
                            restored = with_temporal_values(representative, paths, values)
                            if codec.decode(restored) != original:
                                raise ValueError(f"repeat reconstruction failed: {source}:{records+1}")
                            if pending is representative:
                                pending = dict(representative)
                                pending["__teleop_compact__"] = dict(representative["__teleop_compact__"])
                                pending["__teleop_compact__"]["repeat"] = {
                                    "paths": paths, "rows": [],
                                }
                            pending["__teleop_compact__"]["repeat"]["rows"].append(values)
                        else:
                            if pending is not None:
                                write_compact(pending)
                            representative = pending = compact
                            pending_paths = paths
                    records += 1
            if pending is not None:
                write_compact(pending)
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise ValueError(f"zstd encode failed: {source}")
        except Exception:
            if not encoder.stdin.closed:
                encoder.stdin.close()
            encoder.kill()
            encoder.wait()
            raise
        writer.flush()
        os.fsync(writer.fileno())
    if codec is None or records != source_meta["samples"] or source_bytes != source_meta["bytes"]:
        raise ValueError(f"source count or size changed: {source}")
    if source_digest.hexdigest() != source_meta["sha256"]:
        raise ValueError(f"source SHA-256 changed: {source}")
    decoded_digest = hashlib.sha256()
    decoded_records = decoded_bytes = 0
    decoder = subprocess.Popen(["zstd", "-q", "-dc", str(archive)], stdout=subprocess.PIPE)
    assert decoder.stdout is not None
    for chunk in iter(lambda: decoder.stdout.read(8 * 1024 * 1024), b""):
        decoded_digest.update(chunk)
        decoded_records += chunk.count(b"\n")
        decoded_bytes += len(chunk)
    if decoder.wait() != 0 or decoded_records != compact_records or decoded_bytes != compact_bytes:
        raise ValueError(f"zstd archive did not retain all compact rows: {archive}")
    if decoded_digest.hexdigest() != compact_digest.hexdigest():
        raise ValueError(f"zstd archive changed compact data: {archive}")
    return {
        "source_sha256": source_digest.hexdigest(), "source_records": records,
        "source_bytes": source_bytes, "compact_records": compact_records,
        "collapsed_records": records - compact_records,
        "compact_jsonl_sha256": compact_digest.hexdigest(),
        "compact_jsonl_bytes": compact_bytes,
        "archive_sha256": sha256_file(archive),
        "archive_bytes": archive.stat().st_size,
        "redundant_field_references": reference_count,
        "codec": codec.metadata(),
    }


def compact_episode(entry: dict) -> dict:
    episode_id = entry["episode_id"]
    source_manifest = SOURCE / entry["manifest"]
    manifest = json.loads(source_manifest.read_text())
    if manifest["episode_id"] != episode_id or ("datasets" in manifest and manifest["datasets"] != entry["datasets"]):
        raise ValueError(f"curated manifest and collection differ: {episode_id}")
    final = DEST / "episodes" / episode_id
    staging = final.with_name("." + episode_id + ".staging")
    if final.exists():
        existing = json.loads((final / "manifest.json").read_text())
        if existing["source_manifest_sha256"] != sha256_file(source_manifest):
            raise ValueError(f"curated manifest changed after compaction: {episode_id}")
        for file in existing["files"]:
            archive = ROOT / file["archive_path"]
            if not archive.is_file() or sha256_file(archive) != file["archive_sha256"]:
                raise ValueError(f"missing or altered archive: {archive}")
        print(f"[{entry['course']}] already compacted and verified", flush=True)
        return existing
    if staging.exists():
        raise ValueError(f"incomplete staging directory: {staging}")
    staging.mkdir(parents=True)
    try:
        compact_files = []
        for dataset in entry["datasets"]:
            source = SOURCE / dataset["path"]
            if not source.is_file() or source.stat().st_size != dataset["bytes"]:
                raise ValueError(f"curated file missing or size changed: {source}")
            relative = source.relative_to(SOURCE / "episodes" / episode_id)
            target = staging / relative.with_name(relative.name + ".zst")
            print(f"[{entry['course']}] compacting {relative}", flush=True)
            result = compact_stream(source, target, dataset)
            compact_files.append({
                "source_path": source.relative_to(ROOT).as_posix(),
                "archive_path": (final / relative.with_name(relative.name + ".zst")).relative_to(ROOT).as_posix(),
                "producer": dataset["producer"],
                "valid_for_behavior_cloning_samples": dataset["valid_for_behavior_cloning_samples"],
                **result,
            })
        shutil.copy2(source_manifest, staging / "source_manifest.json")
        compact_manifest = {
            "schema_version": "mssr.semantic_teleop_archive.v1",
            "episode_id": episode_id, "course": entry["course"],
            "task_success": entry["task_success"],
            "success_scope": entry["success_scope"],
            "source_manifest": source_manifest.relative_to(ROOT).as_posix(),
            "source_manifest_sha256": sha256_file(source_manifest),
            "format": "semantic JSONL, zstd frame per curated stream",
            "temporal_policy": "human rows retained; consecutive identical macro transitions grouped with exact temporal reconstruction",
            "files": compact_files,
            "totals": {
                "files": len(compact_files),
                "samples": sum(file["source_records"] for file in compact_files),
                "compact_records": sum(file["compact_records"] for file in compact_files),
                "collapsed_records": sum(file["collapsed_records"] for file in compact_files),
                "source_bytes": sum(file["source_bytes"] for file in compact_files),
                "compact_jsonl_bytes": sum(file["compact_jsonl_bytes"] for file in compact_files),
                "archive_bytes": sum(file["archive_bytes"] for file in compact_files),
            },
        }
        write_json_atomic(staging / "manifest.json", compact_manifest)
        staging.rename(final)
        print(f"[{entry['course']}] compacted {len(compact_files)} streams", flush=True)
        return compact_manifest
    except Exception:
        shutil.rmtree(staging)
        raise


def main() -> None:
    collection = json.loads((SOURCE / "manifest.json").read_text())
    if len(collection["episodes"]) != 13:
        raise ValueError("expected 13 curated teleoperation episodes before compaction")
    (DEST / "episodes").mkdir(parents=True, exist_ok=True)
    summaries = [compact_episode(entry) for entry in collection["episodes"]]
    result = {
        "schema_version": "mssr.semantic_teleop_archive_collection.v1",
        "dataset_name": "expert_v1_compact_teleop",
        "source_collection": (SOURCE / "manifest.json").relative_to(ROOT).as_posix(),
        "source_collection_sha256": sha256_file(SOURCE / "manifest.json"),
        "format": "semantic JSONL, zstd frame per curated stream",
        "temporal_policy": "human rows retained; consecutive identical macro transitions grouped with exact temporal reconstruction",
        "obstacle_stage_labels": "deferred_to_adapter",
        "episodes": [{
            "episode_id": summary["episode_id"], "course": summary["course"],
            "task_success": summary["task_success"],
            "success_scope": summary["success_scope"],
            "manifest": (Path("episodes") / summary["episode_id"] / "manifest.json").as_posix(),
            "totals": summary["totals"],
        } for summary in summaries],
        "totals": {
            "episodes": len(summaries),
            "files": sum(x["totals"]["files"] for x in summaries),
            "samples": sum(x["totals"]["samples"] for x in summaries),
            "compact_records": sum(x["totals"]["compact_records"] for x in summaries),
            "collapsed_records": sum(x["totals"]["collapsed_records"] for x in summaries),
            "source_bytes": sum(x["totals"]["source_bytes"] for x in summaries),
            "compact_jsonl_bytes": sum(x["totals"]["compact_jsonl_bytes"] for x in summaries),
            "archive_bytes": sum(x["totals"]["archive_bytes"] for x in summaries),
        },
    }
    write_json_atomic(DEST / "manifest.json", result)
    print("Semantic teleop collection complete:", result["totals"], flush=True)


if __name__ == "__main__":
    main()
