from __future__ import annotations

import math

from mssr_expert.behaviors.morphology_locomotion import (
    coherent_planar_train_commands,
)
from mssr_expert.graph.attributed_robot_graph import (
    AttributedRobotGraph,
    GraphNode,
)


def _node(module_id: str, yaw_rad: float) -> GraphNode:
    half = yaw_rad / 2.0
    return GraphNode(
        module_id,
        {
            "pose": {
                "orientation_xyzw": [
                    0.0,
                    0.0,
                    math.sin(half),
                    math.cos(half),
                ]
            }
        },
    )


def test_parallel_modules_keep_equal_local_train_signs() -> None:
    graph = AttributedRobotGraph(
        nodes=(_node("rear", 0.0), _node("front", 0.0))
    )
    commands = coherent_planar_train_commands(
        graph,
        {
            "rear": {"vx": 0.03, "vy": 0.0, "yaw_rate": 0.0},
            "front": {"vx": -0.03, "vy": 0.0, "yaw_rate": 0.0},
        },
    )

    assert commands["rear"]["vx"] == 0.03
    assert commands["front"]["vx"] == 0.03


def test_reversed_module_receives_opposite_local_sign() -> None:
    graph = AttributedRobotGraph(
        nodes=(_node("rear", 0.0), _node("front", math.pi))
    )
    commands = coherent_planar_train_commands(
        graph,
        {
            "rear": {"vx": 0.03, "vy": 0.0, "yaw_rate": 0.0},
            "front": {"vx": 0.03, "vy": 0.0, "yaw_rate": 0.0},
        },
    )

    assert commands["rear"]["vx"] == 0.03
    assert commands["front"]["vx"] == -0.03


def test_turning_commands_are_not_projected() -> None:
    graph = AttributedRobotGraph(
        nodes=(_node("a", 0.0), _node("b", math.pi))
    )
    original = {
        "a": {"vx": 0.02, "yaw_rate": 0.2},
        "b": {"vx": 0.02, "yaw_rate": -0.2},
    }

    assert coherent_planar_train_commands(graph, original) == original
