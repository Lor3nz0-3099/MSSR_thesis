from __future__ import annotations

import math
from pathlib import Path

import pytest

from mssr_expert.behaviors.morphology_library import (
    AssignedModule,
    MorphologyLibrary,
)
from mssr_expert.behaviors.morphology_navigation import (
    estimate_planar_morphology_state,
)
from mssr_expert.graph.attributed_robot_graph import (
    AttributedRobotGraph,
    GraphNode,
)


CONFIG = Path(__file__).parents[1] / "config"


def _node(
    module_id: str,
    x: float,
    y: float,
    vx: float = 0.0,
    vy: float = 0.0,
) -> GraphNode:
    return GraphNode(
        module_id,
        {
            "pose": {"position": [x, y, 0.0]},
            "twist": {
                "linear": [vx, vy, 0.0],
                "angular": [0.0, 0.0, 0.2],
            },
        },
    )


def test_role_anchored_navigation_frame_estimates_pose_and_body_twist() -> None:
    graph = AttributedRobotGraph(
        nodes=(
            _node("center", 2.0, 3.0, 0.0, 1.0),
            _node("rear", 2.0, 2.0),
            _node("front", 2.0, 4.0),
        )
    )
    assignments = (
        AssignedModule("center", "v0", "center"),
        AssignedModule("rear", "v1", "rear"),
        AssignedModule("front", "v2", "front"),
    )
    state = estimate_planar_morphology_state(
        graph,
        assignments,
        {
            "center_roles": ("center",),
            "forward_from_roles": ("rear",),
            "forward_to_roles": ("front",),
        },
    )

    assert state.x_m == pytest.approx(2.0)
    assert state.y_m == pytest.approx(3.0)
    assert state.yaw_rad == pytest.approx(math.pi / 2.0)
    assert state.vx_m_s == pytest.approx(1.0)
    assert state.vy_m_s == pytest.approx(0.0)
    assert state.yaw_rate_rad_s == pytest.approx(0.2)


def test_rc_car8_navigation_forward_matches_pan_rolling_direction() -> None:
    spec = MorphologyLibrary.load(
        CONFIG / "smores_morphology_behaviors.json"
    ).navigation_frame_spec("rc_car8")

    assert spec == {
        "center_roles": (
            "chassis_center_left",
            "chassis_left",
            "chassis_center_right",
            "chassis_right",
        ),
        "forward_from_roles": (
            "wheel_right_front",
            "wheel_right_rear",
        ),
        "forward_to_roles": (
            "wheel_left_front",
            "wheel_left_rear",
        ),
    }

    role_positions = {
        "chassis_center_left": (0.0, 0.0),
        "chassis_left": (0.0, 0.0),
        "chassis_center_right": (0.0, 0.0),
        "chassis_right": (0.0, 0.0),
        "wheel_left_front": (-1.0, 1.0),
        "wheel_left_rear": (-1.0, -1.0),
        "wheel_right_front": (1.0, 1.0),
        "wheel_right_rear": (1.0, -1.0),
    }
    graph = AttributedRobotGraph(
        nodes=tuple(
            _node(role, x, y)
            for role, (x, y) in role_positions.items()
        )
    )
    assignments = tuple(
        AssignedModule(role, f"v{index}", role)
        for index, role in enumerate(role_positions)
    )

    state = estimate_planar_morphology_state(graph, assignments, spec)

    assert abs(state.yaw_rad) == pytest.approx(math.pi)
