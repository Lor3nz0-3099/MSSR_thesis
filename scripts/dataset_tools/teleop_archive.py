#!/usr/bin/env python3
"""Lossless, independently verifiable teleoperation stream handling."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def last_jsonl_record(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        handle.seek(0, 2)
        end = handle.tell()
        if not end:
            raise ValueError(f"empty JSONL: {path}")
        size = min(end, 1024 * 1024)
        while True:
            handle.seek(end - size)
            content = handle.read(size).rstrip(b"\r\n")
            if b"\n" in content or size == end:
                return json.loads(content.rsplit(b"\n", 1)[-1])
            size = min(end, size * 2)


def select_successful_streams(directory: Path) -> tuple[list[Path], list[dict[str, str]]]:
    keep: list[Path] = []
    excluded: list[dict[str, str]] = []
    for path in sorted(directory.glob("*.jsonl")):
        last = last_jsonl_record(path)
        if last.get("done") is True and last.get("success") is True:
            keep.append(path)
        else:
            reason = "terminal_failure" if last.get("done") is True else "no_success_terminal"
            excluded.append({"name": path.name, "reason": reason})
    return keep, excluded


def copy_verified_jsonl(
    source: Path,
    destination: Path,
    *,
    producer: str,
    expected_records: int | None = None,
    storage: str = "copy",
) -> dict[str, Any]:
    """Validate a stream and store a physical copy or an exact hard link."""
    if storage not in ("copy", "hardlink"):
        raise ValueError(f"unknown storage mode: {storage}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".copying")
    if temporary.exists() or destination.exists():
        raise ValueError(f"destination already exists: {destination}")
    digest = hashlib.sha256()
    count = valid_bc = valid_actions = 0
    first_stamp = last_stamp = None
    last_row: dict[str, Any] | None = None
    byte_count = 0
    try:
        with source.open("rb") as reader, (temporary.open("wb") if storage == "copy" else open(os.devnull, "wb")) as writer:
            for line in reader:
                if not line.endswith(b"\n") or not line.strip():
                    raise ValueError(f"unterminated or blank JSONL line: {source}:{count+1}")
                row = json.loads(line)
                stamp = float(row["stamp"])
                graph = row["graph_t"]
                next_graph = row["graph_t_plus_1"]
                if len(graph["nodes"]) != 8 or len(next_graph["nodes"]) != 8:
                    raise ValueError(f"incomplete eight-module graph: {source}:{count+1}")
                if not isinstance(row["expert_action"], dict):
                    raise ValueError(f"missing expert action: {source}:{count+1}")
                if float(next_graph["stamp"]) < float(graph["stamp"]):
                    raise ValueError(f"reversed graph transition: {source}:{count+1}")
                if last_stamp is not None and (stamp < last_stamp or
                    (producer == "human_expert" and stamp == last_stamp)):
                    raise ValueError(f"nonmonotonic timestamp: {source}:{count+1}")
                if producer == "human_expert" and not row["action_valid"]:
                    raise ValueError(f"invalid human action: {source}:{count+1}")
                if first_stamp is None:
                    first_stamp = stamp
                last_stamp = stamp
                last_row = row
                count += 1
                valid_actions += bool(row["action_valid"])
                valid_bc += bool(row["supervision"]["valid_for_behavior_cloning"])
                digest.update(line)
                byte_count += len(line)
                writer.write(line)
            if storage == "copy":
                writer.flush()
                os.fsync(writer.fileno())
        if not count or (expected_records is not None and count != expected_records):
            raise ValueError(f"record count mismatch: {source}: {count} != {expected_records}")
        if producer == "deterministic_expert" and not (
            last_row is not None and last_row.get("done") is True
            and last_row.get("success") is True
        ):
            raise ValueError(f"macro has no successful terminal: {source}")
        if storage == "hardlink":
            os.link(source, temporary)
        if sha256_file(temporary) != digest.hexdigest():
            raise ValueError(f"copied bytes differ: {source}")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "sha256": digest.hexdigest(),
        "bytes": byte_count,
        "samples": count,
        "valid_for_behavior_cloning_samples": valid_bc,
        "action_valid_samples": valid_actions,
        "first_stamp": first_stamp,
        "last_stamp": last_stamp,
        "producer": producer,
        "storage": storage,
    }


def archive_jsonl(
    source: Path,
    destination: Path,
    *,
    source_sha256: str,
    source_records: int,
) -> dict[str, Any]:
    """Compress one JSONL and verify exact decompression against its manifest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".compacting")
    if destination.exists() or temporary.exists():
        raise ValueError(f"archive already exists: {destination}")
    try:
        with temporary.open("wb") as writer:
            subprocess.run(
                ["zstd", "-T2", "-3", "-q", "-c", str(source)],
                stdout=writer, check=True,
            )
            writer.flush()
            os.fsync(writer.fileno())
        digest = hashlib.sha256()
        records = 0
        byte_count = 0
        process = subprocess.Popen(
            ["zstd", "-q", "-dc", str(temporary)], stdout=subprocess.PIPE,
        )
        assert process.stdout is not None
        for chunk in iter(lambda: process.stdout.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            records += chunk.count(b"\n")
            byte_count += len(chunk)
        if process.wait() != 0:
            raise ValueError(f"zstd decode failed: {temporary}")
        if digest.hexdigest() != source_sha256 or records != source_records:
            raise ValueError(f"archive roundtrip differs: {source}")
        compact_sha = sha256_file(temporary)
        compact_bytes = temporary.stat().st_size
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "codec": "zstd",
        "source_sha256": source_sha256,
        "source_records": source_records,
        "source_bytes": byte_count,
        "archive_sha256": compact_sha,
        "archive_bytes": compact_bytes,
    }
