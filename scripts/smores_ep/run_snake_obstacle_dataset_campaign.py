#!/usr/bin/env python3
"""Collect adaptive, obstacle-specific Snake8 IL datasets headlessly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "src"))

import run_gap_headless_batch as gap_batch  # noqa: E402
import run_stair_headless_batch as stair_batch  # noqa: E402
from smores_ep.isaac.obstacle_course import (  # noqa: E402
    CoplanarGapSpec,
    UniformStairSpec,
    sample_coplanar_gap_spec,
    sample_uniform_stair_spec,
)


CURRICULUM_LEVELS = ("robust", "intermediate", "challenging")
DIFFICULTY_BY_LEVEL = {
    "robust": 0.0,
    "intermediate": 0.5,
    "challenging": 1.0,
}


def parse_levels(text: str) -> tuple[str, ...]:
    levels = tuple(item.strip() for item in text.split(",") if item.strip())
    if not levels:
        raise ValueError("At least one curriculum level is required")
    unknown = [item for item in levels if item not in CURRICULUM_LEVELS]
    if unknown:
        raise ValueError(f"Unknown curriculum levels: {unknown}")
    indices = [CURRICULUM_LEVELS.index(item) for item in levels]
    if indices != sorted(set(indices)):
        raise ValueError(
            "Curriculum levels must be unique and ordered easiest to hardest"
        )
    return levels


def curriculum_gate(
    results: Sequence[Mapping[str, Any]],
    minimum_success_rate: float,
) -> dict[str, Any]:
    completed = [item for item in results if item.get("success") is not None]
    successes = sum(item.get("success") is True for item in completed)
    failures = sum(item.get("success") is False for item in completed)
    rate = successes / len(completed) if completed else None
    return {
        "evaluated_episode_count": len(completed),
        "success_count": successes,
        "failure_count": failures,
        "success_rate": rate,
        "minimum_success_rate": minimum_success_rate,
        "passed": None if rate is None else rate >= minimum_success_rate,
    }


def _append_jsonl(source: Path, destination: Path) -> int:
    if not source.exists():
        return 0
    lines = [
        line
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as stream:
        for line in lines:
            stream.write(line + "\n")
    return len(lines)


def _episode_args(
    campaign_args: argparse.Namespace,
    output_dir: Path,
    level: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        output_dir=output_dir,
        plan_only=campaign_args.plan_only,
        simulation_steps=campaign_args.simulation_steps,
        simulation_speed_factor=campaign_args.simulation_speed_factor,
        assembly_wall_timeout_s=campaign_args.assembly_wall_timeout_s,
        behavior_wall_timeout_s=campaign_args.behavior_wall_timeout_s,
        behavior_dataset_log_period=(
            campaign_args.behavior_dataset_log_period
        ),
        behavior_control_rate_hz=campaign_args.behavior_control_rate_hz,
        record_assembly_dataset=campaign_args.record_assembly_dataset,
        curriculum_level=level,
        curriculum_difficulty=DIFFICULTY_BY_LEVEL[level],
    )


def _run_obstacle_curriculum(
    *,
    obstacle_name: str,
    batch_module: ModuleType,
    sampler: Callable[[int, str], UniformStairSpec | CoplanarGapSpec],
    levels: Sequence[str],
    seed_offset: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    obstacle_dir = args.output_dir / obstacle_name
    obstacle_dir.mkdir(parents=True, exist_ok=True)
    all_dataset = obstacle_dir / "all_transitions.jsonl"
    successful_dataset = obstacle_dir / "successful_transitions.jsonl"
    level_summaries: list[dict[str, Any]] = []
    total_transitions = 0
    successful_transitions = 0
    stopped_after_level: str | None = None

    for level_index, level in enumerate(levels):
        level_dir = obstacle_dir / level
        level_dir.mkdir(parents=True, exist_ok=True)
        episode_args = _episode_args(args, level_dir, level)
        results: list[dict[str, Any]] = []
        for episode_index in range(args.episodes_per_level):
            seed = (
                args.base_seed
                + seed_offset
                + level_index * 10_000
                + episode_index
            )
            episode_id = f"{level}-seed-{seed:06d}"
            spec = sampler(seed, level)
            print(
                f"[{obstacle_name}/{level}] episode "
                f"{episode_index + 1}/{args.episodes_per_level}, seed={seed}",
                flush=True,
            )
            result = batch_module.run_episode(
                episode_id,
                spec,
                episode_args,
            )
            results.append(result)
            dataset_path = level_dir / episode_id / "behavior_dataset.jsonl"
            transition_count = _append_jsonl(dataset_path, all_dataset)
            total_transitions += transition_count
            if result.get("success") is True:
                successful_transitions += _append_jsonl(
                    dataset_path,
                    successful_dataset,
                )
            print(
                f"[{obstacle_name}/{level}/{episode_id}] "
                f"{result['state']}",
                flush=True,
            )

        gate = curriculum_gate(results, args.minimum_success_rate)
        level_summary = {
            "level": level,
            "difficulty": DIFFICULTY_BY_LEVEL[level],
            "gate": gate,
            "results": results,
        }
        level_summaries.append(level_summary)
        (level_dir / "summary.json").write_text(
            json.dumps(level_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if gate["passed"] is False:
            stopped_after_level = level
            print(
                f"[{obstacle_name}] curriculum stopped after {level}: "
                f"success_rate={gate['success_rate']:.3f} < "
                f"{args.minimum_success_rate:.3f}",
                flush=True,
            )
            break

    summary = {
        "schema_version": "mssr.snake_obstacle_curriculum.v1",
        "obstacle": obstacle_name,
        "headless": not args.plan_only,
        "episodes_per_level": args.episodes_per_level,
        "minimum_success_rate": args.minimum_success_rate,
        "requested_levels": list(levels),
        "completed_levels": [item["level"] for item in level_summaries],
        "stopped_after_level": stopped_after_level,
        "all_transition_count": total_transitions,
        "successful_transition_count": successful_transitions,
        "all_dataset_path": str(all_dataset),
        "successful_dataset_path": str(successful_dataset),
        "levels": level_summaries,
    }
    (obstacle_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "logs/snake_obstacle_dataset_campaign",
    )
    parser.add_argument("--episodes-per-level", type=int, default=5)
    parser.add_argument("--minimum-success-rate", type=float, default=0.80)
    parser.add_argument(
        "--levels",
        default=",".join(CURRICULUM_LEVELS),
        help="Ordered subset of robust,intermediate,challenging",
    )
    parser.add_argument("--base-seed", type=int, default=1_000)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--simulation-steps", type=int, default=240_000)
    parser.add_argument("--simulation-speed-factor", type=float, default=1.0)
    parser.add_argument("--assembly-wall-timeout-s", type=float, default=600.0)
    parser.add_argument("--behavior-wall-timeout-s", type=float, default=600.0)
    parser.add_argument("--behavior-dataset-log-period", type=int, default=30)
    parser.add_argument("--behavior-control-rate-hz", type=float, default=30.0)
    parser.add_argument("--record-assembly-dataset", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.episodes_per_level < 1:
        raise SystemExit("--episodes-per-level must be positive")
    if not 0.0 < args.minimum_success_rate <= 1.0:
        raise SystemExit("--minimum-success-rate must be in (0, 1]")
    try:
        levels = parse_levels(args.levels)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if shutil.which("ros2") is None and not args.plan_only:
        raise SystemExit("ros2 is unavailable; source ROS and mssr_ws first")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(
            f"Output directory is not empty: {args.output_dir}; use a new path"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The requested order is intentional: validate and collect stairs first,
    # then start an independent gap curriculum even if stairs stopped early.
    stairs = _run_obstacle_curriculum(
        obstacle_name="stairs",
        batch_module=stair_batch,
        sampler=sample_uniform_stair_spec,
        levels=levels,
        seed_offset=0,
        args=args,
    )
    gap = _run_obstacle_curriculum(
        obstacle_name="gap",
        batch_module=gap_batch,
        sampler=sample_coplanar_gap_spec,
        levels=levels,
        seed_offset=1_000_000,
        args=args,
    )
    summary = {
        "schema_version": "mssr.snake_obstacle_dataset_campaign.v1",
        "execution_order": ["stairs", "gap"],
        "datasets_are_obstacle_specific": True,
        "stairs": stairs,
        "gap": gap,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
