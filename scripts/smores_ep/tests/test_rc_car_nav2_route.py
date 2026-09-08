"""Pure input tests for composite Nav2 waypoint routes."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "run_rc_car_nav2_route.py"
SPEC = importlib.util.spec_from_file_location("run_rc_car_nav2_route", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_load_composite_route_preserves_curve(tmp_path: Path) -> None:
    path = tmp_path / "route.json"
    path.write_text(
        json.dumps(
            {
                "waypoints_xyyaw": [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.4, 0.5],
                    [1.8, 0.8, 0.0],
                ]
            }
        )
    )
    assert MODULE.load_composite_route(path)[1] == (1.0, 0.4, 0.5)


def test_load_composite_route_requires_two_poses(tmp_path: Path) -> None:
    path = tmp_path / "route.json"
    path.write_text(json.dumps({"waypoints_xyyaw": [[0.0, 0.0, 0.0]]}))
    with pytest.raises(ValueError, match="at least two"):
        MODULE.load_composite_route(path)
