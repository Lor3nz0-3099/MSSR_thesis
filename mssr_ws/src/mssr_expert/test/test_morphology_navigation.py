from __future__ import annotations

import math

import pytest

from mssr_expert.behaviors.morphology_library import AssignedModule
from mssr_expert.behaviors.morphology_navigation import (
    estimate_planar_morphology_state,
)
from mssr_expert.graph.attributed_robot_graph import (
    AttributedRobotGraph,
    GraphNode,
)


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
