from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from teleop_archive import (
    archive_jsonl,
    copy_verified_jsonl,
    select_successful_streams,
)


def row(stamp: float, *, done=False, success=False) -> dict:
    graph = {"nodes": [{"module_id": f"smores_{i:02}"} for i in range(1, 9)],
             "edges": [], "stamp": stamp}
    return {"stamp": stamp, "graph_t": graph,
            "graph_t_plus_1": {**graph, "stamp": stamp + 0.01},
            "expert_action": {"locomotion": {"smores_01": {"vx": 0.1}}},
            "supervision": {"valid_for_behavior_cloning": not done},
            "action_valid": not done, "done": done, "success": success}


def write_rows(path: Path, values: list[dict]) -> bytes:
    data = b"".join(json.dumps(v, separators=(",", ":")).encode() + b"\n" for v in values)
    path.write_bytes(data)
    return data


def test_copy_checks_every_transition_and_preserves_bytes(tmp_path):
    source = tmp_path / "source.jsonl"
    original = write_rows(source, [row(1.0), row(1.1)])
    target = tmp_path / "target.jsonl"
    stats = copy_verified_jsonl(source, target, producer="human_expert",
                                expected_records=2)
    assert target.read_bytes() == original
    assert stats["sha256"] == hashlib.sha256(original).hexdigest()
    assert stats["samples"] == 2
    assert stats["first_stamp"] == 1.0
    assert stats["last_stamp"] == 1.1


def test_copy_rejects_a_reversed_graph_transition(tmp_path):
    source = tmp_path / "source.jsonl"
    bad = row(1.0)
    bad["graph_t_plus_1"]["stamp"] = 0.9
    write_rows(source, [bad])
    with pytest.raises(ValueError, match="reversed graph transition"):
        copy_verified_jsonl(source, tmp_path / "target.jsonl",
                            producer="human_expert", expected_records=1)
    assert not (tmp_path / "target.jsonl").exists()


def test_selection_excludes_failed_and_incomplete_macros(tmp_path):
    d = tmp_path / "structural"
    d.mkdir()
    write_rows(d / "ok.jsonl", [row(1.0), row(1.1, done=True, success=True)])
    write_rows(d / "failed.jsonl", [row(2.0, done=True, success=False)])
    write_rows(d / "incomplete.jsonl", [row(3.0)])
    keep, exclude = select_successful_streams(d)
    assert [p.name for p in keep] == ["ok.jsonl"]
    assert {item["name"]: item["reason"] for item in exclude} == {
        "failed.jsonl": "terminal_failure",
        "incomplete.jsonl": "no_success_terminal",
    }


def test_zstd_archive_roundtrips_exact_bytes(tmp_path):
    source = tmp_path / "source.jsonl"
    original = write_rows(source, [row(1.0), row(1.1)])
    compressed = tmp_path / "source.jsonl.zst"
    result = archive_jsonl(source, compressed,
                           source_sha256=hashlib.sha256(original).hexdigest(),
                           source_records=2)
    assert result["codec"] == "zstd"
    assert result["source_records"] == 2
    assert result["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert subprocess.check_output(["zstd", "-q", "-dc", str(compressed)]) == original


def test_hardlink_checks_source_and_has_independent_directory_entry(tmp_path):
    source = tmp_path / "source.jsonl"
    original = write_rows(source, [row(1.0), row(1.1)])
    target = tmp_path / "target.jsonl"
    stats = copy_verified_jsonl(source, target, producer="human_expert",
                                expected_records=2, storage="hardlink")
    assert stats["storage"] == "hardlink"
    assert target.stat().st_ino == source.stat().st_ino
    assert target.read_bytes() == original
    source.unlink()
    assert target.read_bytes() == original
