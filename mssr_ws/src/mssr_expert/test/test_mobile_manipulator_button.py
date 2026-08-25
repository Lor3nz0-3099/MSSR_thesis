"""Geometric button-contact checks for MobileManipulator8."""

import pytest

from mssr_expert.nodes.smores_obstacle_course_node import (
    button_contact_distance_m,
    end_effector_contact_point,
)


def _module_with_top_face(position: tuple[float, float, float]) -> dict:
    return {
        "position": [0.0, 0.0, 0.0],
        "connectors": [
            {
                "connector_id": "BOTTOM",
                "position_world": [-1.0, -1.0, -1.0],
            },
            {
                "connector_id": "TOP",
                "position_world": list(position),
            },
        ],
    }


def test_button_contact_uses_free_top_face_not_module_center() -> None:
    module = _module_with_top_face((2.65, 0.455, 0.365))

    assert end_effector_contact_point(module) == pytest.approx(
        (2.65, 0.455, 0.365)
    )
    assert button_contact_distance_m(
        module,
        (2.65, 0.475, 0.365),
    ) == pytest.approx(0.020)


def test_button_contact_falls_back_to_vicon_module_center() -> None:
    module = {"position": [2.65, 0.40, 0.365], "connectors": []}

    assert end_effector_contact_point(module) == pytest.approx(
        (2.65, 0.40, 0.365)
    )
