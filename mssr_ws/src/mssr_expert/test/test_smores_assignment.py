"""Tests for optimal SMORES-EP module assignment."""

from __future__ import annotations

import math

import pytest

from mssr_expert.planning.smores_ep.assignment import (
    AssignmentError,
    assign_modules_to_targets,
    solve_linear_assignment,
)
from mssr_expert.planning.smores_ep.rooting import (
    root_kinematic_tree,
)
from mssr_expert.planning.smores_ep.topology import (
    SmoresKinematicTree,
    SmoresTopologyEdge,
)
from mssr_expert.planning.smores_ep.unfolding import (
    PlanarPose,
    unfold_tree_on_plane,
)


def _unfolded_three_module_chain():
    topology = SmoresKinematicTree(
        vertex_ids=("v0", "v1", "v2"),
        edges=(
            SmoresTopologyEdge(
                vertex_a="v0",
                face_a="TOP",
                vertex_b="v1",
                face_b="BOTTOM",
            ),
            SmoresTopologyEdge(
                vertex_a="v1",
                face_a="TOP",
                vertex_b="v2",
                face_b="BOTTOM",
            ),
        ),
    )

    rooted = root_kinematic_tree(
        topology,
        root_id="v1",
    )

    return unfold_tree_on_plane(rooted)


def test_hungarian_algorithm_known_matrix() -> None:
    cost_matrix = (
        (4.0, 1.0, 3.0),
        (2.0, 0.0, 5.0),
        (3.0, 2.0, 2.0),
    )

    selected_columns = solve_linear_assignment(cost_matrix)

    assert selected_columns == (1, 0, 2)

    total_cost = sum(
        cost_matrix[row][column]
        for row, column in enumerate(selected_columns)
    )

    assert total_cost == pytest.approx(5.0)


def test_assignment_preserves_the_selected_root() -> None:
    target = _unfolded_three_module_chain()
    spacing = 0.043771 + 0.033999

    physical_poses = {
        "root_module": PlanarPose(
            x_m=10.0,
            y_m=5.0,
            yaw_rad=math.pi / 2.0,
        ),
        "negative_module": PlanarPose(
            x_m=10.0,
            y_m=5.0 - spacing,
            yaw_rad=math.pi / 2.0,
        ),
        "positive_module": PlanarPose(
            x_m=10.0,
            y_m=5.0 + spacing,
            yaw_rad=math.pi / 2.0,
        ),
    }

    result = assign_modules_to_targets(
        physical_poses=physical_poses,
        physical_root_id="root_module",
        target=target,
    )

    assert result.target_to_module == {
        "v0": "negative_module",
        "v1": "root_module",
        "v2": "positive_module",
    }

    assert result.module_to_target == {
        "negative_module": "v0",
        "root_module": "v1",
        "positive_module": "v2",
    }

    assert result.total_cost == pytest.approx(0.0)


def test_assignment_minimizes_travel_distance() -> None:
    target = _unfolded_three_module_chain()
    spacing = 0.043771 + 0.033999

    physical_poses = {
        "m_root": PlanarPose(0.0, 0.0, 0.0),
        "m_far_right": PlanarPose(spacing, 0.0, 0.0),
        "m_far_left": PlanarPose(-spacing, 0.0, 0.0),
    }

    result = assign_modules_to_targets(
        physical_poses=physical_poses,
        physical_root_id="m_root",
        target=target,
    )

    assert result.target_to_module["v0"] == "m_far_left"
    assert result.target_to_module["v1"] == "m_root"
    assert result.target_to_module["v2"] == "m_far_right"


def test_assignment_is_invariant_to_global_translation() -> None:
    target = _unfolded_three_module_chain()
    spacing = 0.043771 + 0.033999

    physical_poses = {
        "root": PlanarPose(50.0, -30.0, 0.0),
        "left": PlanarPose(50.0 - spacing, -30.0, 0.0),
        "right": PlanarPose(50.0 + spacing, -30.0, 0.0),
    }

    result = assign_modules_to_targets(
        physical_poses=physical_poses,
        physical_root_id="root",
        target=target,
    )

    assert result.target_to_module == {
        "v0": "left",
        "v1": "root",
        "v2": "right",
    }


def test_orientation_can_be_included_in_assignment_cost() -> None:
    target = _unfolded_three_module_chain()
    spacing = 0.043771 + 0.033999

    physical_poses = {
        "root": PlanarPose(0.0, 0.0, 0.0),
        "correct_orientation": PlanarPose(
            -spacing,
            0.0,
            0.0,
        ),
        "wrong_orientation": PlanarPose(
            spacing,
            0.0,
            math.pi,
        ),
    }

    result = assign_modules_to_targets(
        physical_poses=physical_poses,
        physical_root_id="root",
        target=target,
        orientation_weight_m_per_rad=0.01,
    )

    assert result.target_to_module["v0"] == "correct_orientation"
    assert result.target_to_module["v2"] == "wrong_orientation"
    assert result.total_cost > 0.0


def test_hungarian_assignment_moves_future_corridor_occupant_earlier() -> None:
    topology = SmoresKinematicTree(
        vertex_ids=tuple(f"v{index}" for index in range(8)),
        edges=tuple(
            SmoresTopologyEdge(
                f"v{index}",
                "TOP",
                f"v{index + 1}",
                "BOTTOM",
            )
            for index in range(7)
        ),
    )
    rooted = root_kinematic_tree(topology, root_id="v3")
    target = unfold_tree_on_plane(rooted)
    module_ids = tuple(f"smores_{index:02d}" for index in range(1, 9))
    physical_poses = {
        module_ids[0]: PlanarPose(0.0, 0.0, 0.0),
    }
    for index, module_id in enumerate(module_ids[1:]):
        angle = 2.0 * math.pi * index / 7.0
        physical_poses[module_id] = PlanarPose(
            0.34 * math.cos(angle),
            0.34 * math.sin(angle),
            angle + math.pi,
        )

    distance_only = assign_modules_to_targets(
        physical_poses,
        "smores_01",
        target,
    )
    assert distance_only.target_to_module["v6"] == "smores_03"
    assert distance_only.target_to_module["v7"] == "smores_02"

    congestion_aware = assign_modules_to_targets(
        physical_poses,
        "smores_01",
        target,
        target_parent_by_vertex=rooted.parent_by_vertex,
        target_depth_by_vertex=rooted.depth_by_vertex,
    )
    assert congestion_aware.target_to_module["v6"] == "smores_02"
    assert congestion_aware.target_to_module["v7"] == "smores_03"
    assert congestion_aware.total_future_blockers == 0
    assert congestion_aware.total_cost > distance_only.total_cost


def test_different_module_counts_are_rejected() -> None:
    target = _unfolded_three_module_chain()

    physical_poses = {
        "root": PlanarPose(0.0, 0.0, 0.0),
        "other": PlanarPose(1.0, 0.0, 0.0),
    }

    with pytest.raises(
        AssignmentError,
        match="at least the target count",
    ):
        assign_modules_to_targets(
            physical_poses=physical_poses,
            physical_root_id="root",
            target=target,
        )


def test_non_square_cost_matrix_is_rejected() -> None:
    with pytest.raises(
        AssignmentError,
        match="square",
    ):
        solve_linear_assignment(
            (
                (1.0, 2.0, 3.0),
                (4.0, 5.0, 6.0),
            )
        )
