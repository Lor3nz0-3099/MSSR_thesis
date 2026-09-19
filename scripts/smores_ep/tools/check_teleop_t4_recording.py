#!/usr/bin/env python3
"""Validate one real T4 teleoperation recording against runtime status."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise RuntimeError(f"{path}:{number}: blank line")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"{path}:{number}: row is not an object")
            rows.append(value)
    return rows


def load_builder(repository_root: Path):
    path = repository_root / "scripts/dataset_tools/build_expert_v1.py"
    spec = importlib.util.spec_from_file_location(
        "_t4_build_expert_v1",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load build_expert_v1.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rate_hz(stamps: list[float]) -> float | None:
    if len(stamps) < 2:
        return None
    duration = stamps[-1] - stamps[0]
    if duration <= 0:
        return None
    return (len(stamps) - 1) / duration


def nearest_status(
    statuses: list[dict[str, Any]],
    stamp: float,
) -> tuple[dict[str, Any] | None, float | None]:
    best = None
    best_error = None
    for row in statuses:
        candidate = row.get("stamp_monotonic")
        if not isinstance(candidate, (int, float)):
            continue
        error = abs(float(candidate) - stamp)
        if best_error is None or error < best_error:
            best = row
            best_error = error
    return best, best_error


def check(condition: bool, name: str, detail: str, results: list[dict]) -> None:
    results.append(
        {
            "name": name,
            "pass": bool(condition),
            "detail": detail,
        }
    )


def analyze(
    repository_root: Path,
    episode_dir: Path,
    status_path: Path,
) -> dict[str, Any]:
    manifest_path = episode_dir / "manifest.json"
    human_path = episode_dir / "human_behavior.jsonl"

    manifest = load_json(manifest_path)
    rows = load_jsonl(human_path)
    statuses = load_jsonl(status_path)

    results: list[dict[str, Any]] = []

    human_meta = manifest.get("human_stream", {})

    check(
        manifest.get("schema_version") == "mssr.teleop_recording_manifest.v1",
        "manifest_schema",
        str(manifest.get("schema_version")),
        results,
    )
    check(
        manifest.get("status") == "completed",
        "manifest_completed",
        str(manifest.get("status")),
        results,
    )
    check(
        manifest.get("error") in (None, ""),
        "no_recording_error",
        repr(manifest.get("error")),
        results,
    )
    check(
        human_meta.get("producer") == "human_expert",
        "human_provenance",
        str(human_meta.get("producer")),
        results,
    )
    check(
        int(human_meta.get("records", -1)) == len(rows),
        "manifest_record_count",
        f"manifest={human_meta.get('records')} actual={len(rows)}",
        results,
    )
    check(
        int(human_meta.get("bytes", -1)) == human_path.stat().st_size,
        "manifest_byte_count",
        f"manifest={human_meta.get('bytes')} actual={human_path.stat().st_size}",
        results,
    )
    check(
        len(rows) >= 5,
        "enough_human_samples",
        f"records={len(rows)}",
        results,
    )

    graph_order_ok = True
    provenance_ok = True
    bc_ok = True
    attributes_ok = True
    effective_match_ok = True
    teleop_authority_ok = True
    worst_status_dt = 0.0
    sample_stamps = []

    required_attrs = {
        "position",
        "orientation",
        "linear_velocity",
        "angular_velocity",
    }

    for index, row in enumerate(rows):
        graph_t = row.get("graph_t", {})
        graph_next = row.get("graph_t_plus_1", {})
        t = graph_t.get("stamp")
        t_next = graph_next.get("stamp")

        if not (
            isinstance(t, (int, float))
            and isinstance(t_next, (int, float))
            and float(t_next) > float(t)
        ):
            graph_order_ok = False

        supervision = row.get("supervision", {})
        if (
            supervision.get("label_source") != "human_expert"
            or supervision.get("executed_action_source") != "human_expert"
        ):
            provenance_ok = False
        if not supervision.get("valid_for_behavior_cloning"):
            bc_ok = False

        for graph in (graph_t, graph_next):
            nodes = graph.get("nodes", [])
            if not nodes:
                attributes_ok = False
                continue
            for node in nodes:
                attrs = node.get("attributes", {})
                if not required_attrs.issubset(attrs):
                    attributes_ok = False

        observation = row.get("observation", {})
        sampled_at = observation.get("sampled_at_wall")
        if isinstance(sampled_at, (int, float)):
            sampled_at = float(sampled_at)
            sample_stamps.append(sampled_at)

            status, error = nearest_status(statuses, sampled_at)
            if status is None or error is None:
                effective_match_ok = False
                teleop_authority_ok = False
            else:
                worst_status_dt = max(worst_status_dt, error)
                if error > 0.03:
                    effective_match_ok = False
                    teleop_authority_ok = False
                else:
                    expected = status.get("rc_car_effective_actions")
                    actual = row.get("expert_action", {}).get("locomotion")
                    if actual != expected:
                        effective_match_ok = False
                    if status.get("authority") != "TELEOP":
                        teleop_authority_ok = False
        else:
            effective_match_ok = False
            teleop_authority_ok = False

    check(
        graph_order_ok,
        "fresh_graph_successors",
        "every graph_t_plus_1 stamp > graph_t stamp",
        results,
    )
    check(
        provenance_ok,
        "all_rows_human_expert",
        "label/executed source",
        results,
    )
    check(
        bc_ok,
        "all_rows_bc_valid",
        "supervision.valid_for_behavior_cloning",
        results,
    )
    check(
        attributes_ok,
        "graph_state_attributes",
        ",".join(sorted(required_attrs)),
        results,
    )
    check(
        effective_match_ok,
        "effective_actions_match_runtime",
        f"worst_status_alignment_s={worst_status_dt:.6f}",
        results,
    )
    check(
        teleop_authority_ok,
        "samples_only_under_teleop_authority",
        f"worst_status_alignment_s={worst_status_dt:.6f}",
        results,
    )

    dataset_rate = rate_hz(sample_stamps)
    check(
        dataset_rate is not None and 8.0 <= dataset_rate <= 12.0,
        "dataset_rate_10hz",
        "n/a" if dataset_rate is None else f"{dataset_rate:.3f} Hz",
        results,
    )

    status_stamps = [
        float(row["stamp_monotonic"])
        for row in statuses
        if isinstance(row.get("stamp_monotonic"), (int, float))
    ]
    control_rate = rate_hz(status_stamps)
    check(
        control_rate is not None and 40.0 <= control_rate <= 60.0,
        "control_rate_50hz",
        "n/a" if control_rate is None else f"{control_rate:.3f} Hz",
        results,
    )

    events = manifest.get("events", [])
    event_names = [
        event.get("name")
        for event in events
        if isinstance(event, dict)
    ]
    check(
        event_names.count("record_toggle") >= 2,
        "start_stop_events_present",
        f"record_toggle_count={event_names.count('record_toggle')}",
        results,
    )

    builder = load_builder(repository_root)
    try:
        analyzer, _digest, _bytes = builder.analyze_existing(
            human_path,
            "behavior",
        )
        summary = analyzer.summary()
        analyzer_ok = (
            summary["records"] == len(rows)
            and summary["behavior_cloning_valid_records"] == len(rows)
            and summary["il_eligible_records"] == len(rows)
        )
        analyzer_detail = json.dumps(summary, sort_keys=True)
    except Exception as exc:
        analyzer_ok = False
        analyzer_detail = f"{type(exc).__name__}: {exc}"

    check(
        analyzer_ok,
        "expert_v1_raw_analyzer",
        analyzer_detail,
        results,
    )

    passed = sum(int(item["pass"]) for item in results)

    return {
        "schema_version": "mssr.teleop_t4_runtime_check.v1",
        "episode_dir": str(episode_dir),
        "status_capture": str(status_path),
        "checks_passed": passed,
        "checks_total": len(results),
        "success": passed == len(results),
        "metrics": {
            "human_records": len(rows),
            "status_records": len(statuses),
            "dataset_rate_hz": dataset_rate,
            "control_rate_hz": control_rate,
            "worst_status_alignment_s": worst_status_dt,
        },
        "checks": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("status_capture", type=Path)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    report = analyze(
        args.repository_root.resolve(),
        args.episode_dir.resolve(),
        args.status_capture.resolve(),
    )

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")

    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
