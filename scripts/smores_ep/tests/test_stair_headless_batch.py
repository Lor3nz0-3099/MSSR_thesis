from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from smores_ep.isaac.obstacle_course import UniformStairSpec


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "run_stair_headless_batch.py"
SPEC = importlib.util.spec_from_file_location("stair_headless_batch", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


def test_seed_expression_supports_inclusive_ranges() -> None:
    assert batch.parse_seed_expression("0:2,5,7:6") == (0, 1, 2, 5, 7, 6)
    with pytest.raises(ValueError, match="unique"):
        batch.parse_seed_expression("1,1")


def test_batch_specs_preserve_reference_and_seeded_episodes() -> None:
    episodes = batch.build_episode_specs((4, 9), include_reference=True)

    assert tuple(episode_id for episode_id, _ in episodes) == (
        "reference",
        "seed-000004",
        "seed-000009",
    )
    assert episodes[0][1] == UniformStairSpec()
    assert episodes[1][1].seed == 4


def test_reference_only_batch_has_no_random_episode() -> None:
    assert batch.build_episode_specs((), include_reference=True) == (
        ("reference", UniformStairSpec()),
    )


def _graph(center_height_m: float, connections: int = 7) -> dict:
    return {
        "global_attributes": {
            "latched_connection_count": connections,
            "module_geometry": {"wheel_radius_m": 0.03106},
        },
        "nodes": [
            {
                "module_id": f"smores_{index:02d}",
                "attributes": {"position": [1.5, 0.0, center_height_m]},
            }
            for index in range(1, 9)
        ],
    }


def test_independent_top_deck_metric_accepts_valid_final_geometry() -> None:
    spec = UniformStairSpec()
    result = batch.evaluate_top_deck_result(
        _graph(spec.top_heights_m[-1] + 0.03106),
        spec,
    )

    assert result["all_modules_on_top_deck"] is True
    assert result["module_count"] == 8
    assert result["connection_count"] == 7


def test_independent_top_deck_metric_rejects_low_tail_or_broken_chain() -> None:
    spec = UniformStairSpec()

    low = batch.evaluate_top_deck_result(_graph(0.03106), spec)
    broken = batch.evaluate_top_deck_result(
        _graph(spec.top_heights_m[-1] + 0.03106, connections=6),
        spec,
    )

    assert low["all_modules_on_top_deck"] is False
    assert broken["all_modules_on_top_deck"] is False
