"""Analyze JSON history from a two-module magnetic joint simulation."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class JointHistoryMetrics:
    """Compact behavior metrics for one two-module joint run."""

    sample_count: int
    connected_sample_count: int
    expected_joint_type: str
    final_status: str | None
    final_joint_type: str | None
    final_magnet_enabled: bool
    initial_distance: float
    final_distance: float
    driven_displacement: float
    follower_displacement: float
    distance_drift: float
    passed: bool


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for history analysis."""
    parser = argparse.ArgumentParser(description="Analyze MSSR joint history JSONL.")
    parser.add_argument(
        "--history-file",
        default="logs/bridge/state_graph_history.jsonl",
        help="JSON Lines file written by scripts/main.py --publish-json-history.",
    )
    parser.add_argument(
        "--module-a-id",
        default="sphere_0",
        help="Driven module id.",
    )
    parser.add_argument(
        "--module-b-id",
        default="sphere_1",
        help="Follower module id.",
    )
    parser.add_argument(
        "--expected-joint",
        choices=("rigid", "spherical", "hinge"),
        default="spherical",
        help="Joint type that should appear on the graph edge.",
    )
    parser.add_argument(
        "--min-follower-displacement",
        type=float,
        default=0.03,
        help="Minimum planar motion required from the follower module.",
    )
    parser.add_argument(
        "--max-distance-drift",
        type=float,
        default=0.15,
        help="Maximum allowed center-distance drift during the run.",
    )
    return parser.parse_args()


def main() -> None:
    """Load the history, compute metrics, and exit non-zero on failure."""
    args = parse_args()
    records = _read_history(Path(args.history_file))
    metrics = analyze_history(
        records=records,
        module_a_id=args.module_a_id,
        module_b_id=args.module_b_id,
        expected_joint_type=args.expected_joint,
        min_follower_displacement=args.min_follower_displacement,
        max_distance_drift=args.max_distance_drift,
    )
    print(format_metrics(metrics))
    if not metrics.passed:
        raise SystemExit(1)


def analyze_history(
    records: tuple[dict[str, Any], ...],
    module_a_id: str,
    module_b_id: str,
    expected_joint_type: str,
    min_follower_displacement: float,
    max_distance_drift: float,
) -> JointHistoryMetrics:
    """Compute dynamic coupling metrics from state/graph history records."""
    if len(records) < 2:
        raise ValueError("The history must contain at least two samples.")

    first_positions = _module_positions(records[0])
    last_positions = _module_positions(records[-1])
    first_a = first_positions[module_a_id]
    first_b = first_positions[module_b_id]
    last_a = last_positions[module_a_id]
    last_b = last_positions[module_b_id]

    connected_edges = tuple(
        edge
        for record in records
        for edge in _matching_edges(record, module_a_id, module_b_id)
        if edge.get("status") == "connected"
    )
    final_edge = _final_matching_edge(records, module_a_id, module_b_id)
    final_status = _optional_string(final_edge, "status")
    final_joint_type = _optional_string(final_edge, "joint_type")
    final_magnet_enabled = bool(final_edge.get("is_magnet_enabled", False)) if final_edge else False

    initial_distance = _distance(first_a, first_b)
    final_distance = _distance(last_a, last_b)
    distance_drift = abs(final_distance - initial_distance)
    driven_displacement = _planar_distance(first_a, last_a)
    follower_displacement = _planar_distance(first_b, last_b)
    passed = (
        bool(connected_edges)
        and final_status == "connected"
        and final_joint_type == expected_joint_type
        and final_magnet_enabled
        and follower_displacement >= min_follower_displacement
        and distance_drift <= max_distance_drift
    )

    return JointHistoryMetrics(
        sample_count=len(records),
        connected_sample_count=len(connected_edges),
        expected_joint_type=expected_joint_type,
        final_status=final_status,
        final_joint_type=final_joint_type,
        final_magnet_enabled=final_magnet_enabled,
        initial_distance=initial_distance,
        final_distance=final_distance,
        driven_displacement=driven_displacement,
        follower_displacement=follower_displacement,
        distance_drift=distance_drift,
        passed=passed,
    )


def format_metrics(metrics: JointHistoryMetrics) -> str:
    """Format metrics as a compact terminal report."""
    status = "OK" if metrics.passed else "FAIL"
    return (
        f"{status}: expected_joint={metrics.expected_joint_type} "
        f"final=({metrics.final_status}, {metrics.final_joint_type}, "
        f"magnet={metrics.final_magnet_enabled}) "
        f"samples={metrics.sample_count} connected_samples={metrics.connected_sample_count} "
        f"driven={metrics.driven_displacement:.3f}m "
        f"follower={metrics.follower_displacement:.3f}m "
        f"distance={metrics.initial_distance:.3f}->{metrics.final_distance:.3f}m "
        f"drift={metrics.distance_drift:.3f}m"
    )


def _read_history(path: Path) -> tuple[dict[str, Any], ...]:
    """Read a JSON Lines history file."""
    if not path.exists():
        raise FileNotFoundError(f"History file not found: {path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as history_file:
        for line_number, line in enumerate(history_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {path}") from exc
    return tuple(records)


def _module_positions(record: dict[str, Any]) -> dict[str, Vector3]:
    """Return module world positions indexed by module id."""
    modules = record.get("state", {}).get("modules", ())
    return {
        str(module["module_id"]): _vector3(module["pose"]["position"])
        for module in modules
    }


def _matching_edges(
    record: dict[str, Any],
    module_a_id: str,
    module_b_id: str,
) -> tuple[dict[str, Any], ...]:
    """Return graph edges connecting the requested unordered module pair."""
    requested_pair = frozenset((module_a_id, module_b_id))
    edges = record.get("graph", {}).get("edges", ())
    return tuple(
        edge
        for edge in edges
        if frozenset((str(edge.get("source")), str(edge.get("target")))) == requested_pair
    )


def _final_matching_edge(
    records: tuple[dict[str, Any], ...],
    module_a_id: str,
    module_b_id: str,
) -> dict[str, Any] | None:
    """Return the last graph edge found for the requested module pair."""
    for record in reversed(records):
        edges = _matching_edges(record, module_a_id, module_b_id)
        if edges:
            return edges[-1]
    return None


def _optional_string(edge: dict[str, Any] | None, key: str) -> str | None:
    """Read an optional string field from an edge dictionary."""
    if edge is None or edge.get(key) is None:
        return None
    return str(edge[key])


def _vector3(values: object) -> Vector3:
    """Convert a JSON sequence to a typed 3D vector."""
    sequence = list(values)  # type: ignore[arg-type]
    return (float(sequence[0]), float(sequence[1]), float(sequence[2]))


def _distance(first: Vector3, second: Vector3) -> float:
    """Return Euclidean distance between two 3D points."""
    return math.sqrt(
        (first[0] - second[0]) ** 2
        + (first[1] - second[1]) ** 2
        + (first[2] - second[2]) ** 2
    )


def _planar_distance(first: Vector3, second: Vector3) -> float:
    """Return XY-plane distance between two 3D points."""
    return math.sqrt((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2)


if __name__ == "__main__":
    main()
