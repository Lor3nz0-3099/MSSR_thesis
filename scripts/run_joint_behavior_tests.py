"""Run end-to-end magnetic joint behavior tests through the main simulator."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_joint_history import (  # noqa: E402
    JointHistoryMetrics,
    analyze_history,
    format_metrics,
)


def parse_args() -> argparse.Namespace:
    """Parse joint behavior test options."""
    parser = argparse.ArgumentParser(description="Run MSSR joint behavior tests.")
    parser.add_argument(
        "--isaac-python",
        default="/home/lorenzo/isaac/python.sh",
        help="Isaac Sim Python launcher.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=180,
        help="Simulation steps per joint test.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.15,
        help="Forward speed commanded to sphere_0.",
    )
    parser.add_argument(
        "--output-root",
        default="",
        help="Directory for generated histories. Defaults to a timestamped log dir.",
    )
    parser.add_argument(
        "--min-follower-displacement",
        type=float,
        default=0.03,
        help="Minimum planar motion required from sphere_1.",
    )
    parser.add_argument(
        "--max-distance-drift",
        type=float,
        default=0.15,
        help="Maximum allowed center-distance drift.",
    )
    parser.add_argument(
        "--show-isaac-output",
        action="store_true",
        help="Print full Isaac Sim output for each joint run.",
    )
    return parser.parse_args()


def main() -> None:
    """Run every supported joint test and report the behavior metrics."""
    args = parse_args()
    output_root = _output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    failures = 0
    for joint_type in ("spherical", "hinge", "rigid"):
        joint_output_dir = output_root / joint_type
        _run_simulation(
            isaac_python=args.isaac_python,
            joint_type=joint_type,
            output_dir=joint_output_dir,
            max_steps=args.max_steps,
            speed=args.speed,
            show_isaac_output=args.show_isaac_output,
        )
        metrics = _analyze(
            output_dir=joint_output_dir,
            joint_type=joint_type,
            min_follower_displacement=args.min_follower_displacement,
            max_distance_drift=args.max_distance_drift,
        )
        print(format_metrics(metrics), flush=True)
        if not metrics.passed:
            failures += 1

    print(f"Joint behavior logs: {output_root}", flush=True)
    if failures:
        raise SystemExit(1)


def _output_root(raw_output_root: str) -> Path:
    """Return the root directory for this test run."""
    if raw_output_root:
        return Path(raw_output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "logs" / "joint_behavior" / f"run_{timestamp}"


def _run_simulation(
    isaac_python: str,
    joint_type: str,
    output_dir: Path,
    max_steps: int,
    speed: float,
    show_isaac_output: bool,
) -> None:
    """Run the main simulator once for a requested joint type."""
    command = [
        isaac_python,
        "scripts/main.py",
        "--headless",
        "--max-steps",
        str(max_steps),
        "--module-count",
        "2",
        "--module-spacing",
        "1.2",
        "--demo-forward-speed",
        str(speed),
        "--auto-attach-on-contact",
        "--auto-attach-joint-type",
        joint_type,
        "--publish-json-dir",
        str(output_dir),
        "--publish-json-history",
        "--json-publish-interval",
        "1",
    ]
    if show_isaac_output:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        return

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        print(_tail(completed.stdout, line_count=80), flush=True)
        completed.check_returncode()


def _analyze(
    output_dir: Path,
    joint_type: str,
    min_follower_displacement: float,
    max_distance_drift: float,
) -> JointHistoryMetrics:
    """Analyze one generated state/graph history."""
    history_file = output_dir / "state_graph_history.jsonl"
    records = tuple(
        json.loads(line)
        for line in history_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    return analyze_history(
        records=records,
        module_a_id="sphere_0",
        module_b_id="sphere_1",
        expected_joint_type=joint_type,
        min_follower_displacement=min_follower_displacement,
        max_distance_drift=max_distance_drift,
    )


def _tail(text: str, line_count: int) -> str:
    """Return the last lines of a long subprocess log."""
    return "\n".join(text.splitlines()[-line_count:])


if __name__ == "__main__":
    main()
