from __future__ import annotations

import pytest

from smores_ep.config.simulation import MultiModuleLiftSimulationConfig
from smores_ep.scenarios.multi_module_lift import (
    chain_module_ids,
    chain_module_roots,
)


def test_default_lift_scenario_builds_a_five_module_chain(tmp_path) -> None:
    config = MultiModuleLiftSimulationConfig(
        physics_usd=tmp_path / "physics.usd",
    )
    assert config.chain_module_count == 5
    assert config.active_actuators.tilt_max_effort_nm == pytest.approx(18.4)
    assert chain_module_ids("chain", 5) == (
        "chain_01",
        "chain_02",
        "chain_03",
        "chain_04",
        "chain_05",
    )
    assert chain_module_roots(2) == (
        "/World/smores_ep_chain_01",
        "/World/smores_ep_chain_02",
    )


def test_lift_scenario_rejects_an_empty_chain(tmp_path) -> None:
    with pytest.raises(ValueError):
        MultiModuleLiftSimulationConfig(
            physics_usd=tmp_path / "physics.usd",
            chain_module_count=0,
        )
