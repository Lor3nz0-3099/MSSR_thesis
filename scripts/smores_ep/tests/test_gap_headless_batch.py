from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from smores_ep.isaac.obstacle_course import CoplanarGapSpec


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "run_gap_headless_batch.py"
SPEC = importlib.util.spec_from_file_location(
    "gap_headless_batch", SCRIPT_PATH
)
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
    assert episodes[0][1] == CoplanarGapSpec()
    assert episodes[1][1].seed == 4


def _graph(
    center_x_m: float,
    *,
    center_z_m: float = 0.03106,
    connections: int = 7,
    tilt_rad: float = -0.11,
) -> dict:
    return {
        "global_attributes": {
            "latched_connection_count": connections,
            "module_geometry": {"wheel_radius_m": 0.03106},
        },
        "nodes": [
            {
                "module_id": f"smores_{index:02d}",
                "attributes": {
                    "position": [center_x_m, 0.0, center_z_m],
                    "actuators": {
                        "tilt": {"position_rad": tilt_rad},
                    },
                },
            }
            for index in range(1, 9)
        ],
    }


def test_independent_far_bank_metric_accepts_valid_final_geometry() -> None:
    spec = CoplanarGapSpec()
    result = batch.evaluate_far_bank_result(
        _graph(spec.far_edge_x_m + 0.10),
        spec,
    )

    assert result["all_modules_on_far_bank"] is True
    assert result["module_count"] == 8
    assert result["connection_count"] == 7


@pytest.mark.parametrize(
    "payload",
    (
        _graph(0.74),
        _graph(0.85, center_z_m=0.09),
        _graph(0.85, connections=6),
    ),
)
def test_far_bank_metric_rejects_unsupported_or_broken_chain(
    payload: dict,
) -> None:
    result = batch.evaluate_far_bank_result(payload, CoplanarGapSpec())

    assert result["all_modules_on_far_bank"] is False


def test_far_bank_metric_rejects_non_neutral_tilt_profile() -> None:
    payload = _graph(0.85)
    payload["nodes"][0]["attributes"]["actuators"]["tilt"][
        "position_rad"
    ] = 0.40

    result = batch.evaluate_far_bank_result(payload, CoplanarGapSpec())

    assert result["all_modules_on_far_bank"] is False
