from __future__ import annotations

from pathlib import Path

from mssr_expert.behaviors.morphology_dof_model import (
    SmoresMorphologyDofAnalyzer,
)
from mssr_expert.graph.serialization import load_attributed_graph


CONFIG = Path(__file__).parents[1] / "config"


def _analyze(filename: str):
    return SmoresMorphologyDofAnalyzer().analyze(
        load_attributed_graph(CONFIG / filename)
    )


def test_bridge_exposes_every_top_joint_as_load_bearing_shape_dof() -> None:
    inventory = _analyze("smores_bridge8.json")

    load_bearing = inventory.by_mode("load_bearing")
    assert len(load_bearing) == 16
    assert {dof.name for dof in load_bearing} == {"tilt", "pan"}
    assert len(inventory.by_mode("locomotion_candidate")) == 16


def test_rc_car_wheel_modules_keep_height_and_locomotion_dofs() -> None:
    inventory = _analyze("smores_rc_car8.json")
    by_module = {module.module_id: module for module in inventory.modules}

    for module_id in ("v3", "v4", "v5", "v6"):
        modes = {dof.name: dof.mode for dof in by_module[module_id].dofs}
        assert modes == {
            "left_wheel": "locomotion_candidate",
            "right_wheel": "locomotion_candidate",
            "tilt": "shape_candidate",
            "pan": "shape_candidate",
        }


def test_bottom_connection_does_not_consume_an_actuator_coordinate() -> None:
    inventory = _analyze("smores_mobile_manipulator8.json")
    by_module = {module.module_id: module for module in inventory.modules}
    chassis = by_module["v0"]

    assert chassis.body_is_directly_attached
    assert all(dof.affected_face != "BOTTOM" for dof in chassis.dofs)


def test_pan_and_tilt_expose_the_physical_motor_mixing() -> None:
    inventory = _analyze("smores_bridge8.json")
    first = inventory.modules[0]
    by_name = {dof.name: dof for dof in first.dofs}

    assert by_name["tilt"].motor_mix == (
        ("motorA", 1.0),
        ("motorB", 1.0),
    )
    assert by_name["pan"].motor_mix == (
        ("motorA", 1.0),
        ("motorB", -1.0),
    )
