"""Tests for planar unfolding of SMORES-EP target configurations."""

from __future__ import annotations

import math

import pytest

from mssr_expert.planning.smores_ep.rooting import (
    root_kinematic_tree,
)
from mssr_expert.planning.smores_ep.topology import (
    SmoresKinematicTree,
    SmoresTopologyEdge,
)
from mssr_expert.planning.smores_ep.unfolding import (
    PlanarModuleGeometry,
    PlanarPose,
    PlanarUnfoldingError,
    normalize_angle,
    unfold_tree_on_plane,
)


def test_top_bottom_chain_is_unfolded_from_center() -> None:
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

    rooted = root_kinematic_tree(topology, root_id="v1")
    unfolded = unfold_tree_on_plane(rooted)

    poses = unfolded.poses_by_vertex
    spacing = 0.043771 + 0.033999

    assert poses["v1"] == PlanarPose(0.0, 0.0, 0.0)

    assert poses["v0"].x_m == pytest.approx(-spacing)
    assert poses["v0"].y_m == pytest.approx(0.0)
    assert poses["v0"].yaw_rad == pytest.approx(0.0)

    assert poses["v2"].x_m == pytest.approx(spacing)
    assert poses["v2"].y_m == pytest.approx(0.0)
    assert poses["v2"].yaw_rad == pytest.approx(0.0)


def test_top_to_top_connection_reverses_child_orientation() -> None:
    topology = SmoresKinematicTree(
        vertex_ids=("root", "child"),
        edges=(
            SmoresTopologyEdge(
                vertex_a="root",
                face_a="TOP",
                vertex_b="child",
                face_b="TOP",
            ),
        ),
    )

    rooted = root_kinematic_tree(topology, root_id="root")
    unfolded = unfold_tree_on_plane(rooted)

    child_pose = unfolded.poses_by_vertex["child"]

    assert child_pose.x_m == pytest.approx(2.0 * 0.043771)
    assert child_pose.y_m == pytest.approx(0.0)
    assert abs(child_pose.yaw_rad) == pytest.approx(math.pi)


def test_left_to_right_connection_uses_measured_wheel_offsets() -> None:
    topology = SmoresKinematicTree(
        vertex_ids=("root", "child"),
        edges=(
            SmoresTopologyEdge(
                vertex_a="root",
                face_a="LEFT",
                vertex_b="child",
                face_b="RIGHT",
            ),
        ),
    )

    rooted = root_kinematic_tree(topology, root_id="root")
    unfolded = unfold_tree_on_plane(rooted)

    child_pose = unfolded.poses_by_vertex["child"]
    expected_spacing = 0.043462 + 0.043448

    assert child_pose.x_m == pytest.approx(0.0)
    assert child_pose.y_m == pytest.approx(expected_spacing)
    assert child_pose.yaw_rad == pytest.approx(0.0)


def test_root_pose_rotates_the_whole_unfolding() -> None:
    topology = SmoresKinematicTree(
        vertex_ids=("root", "child"),
        edges=(
            SmoresTopologyEdge(
                vertex_a="root",
                face_a="TOP",
                vertex_b="child",
                face_b="BOTTOM",
            ),
        ),
    )

    rooted = root_kinematic_tree(topology, root_id="root")

    unfolded = unfold_tree_on_plane(
        rooted,
        root_pose=PlanarPose(
            x_m=1.0,
            y_m=2.0,
            yaw_rad=math.pi / 2.0,
        ),
    )

    child_pose = unfolded.poses_by_vertex["child"]
    spacing = 0.043771 + 0.033999

    assert child_pose.x_m == pytest.approx(1.0)
    assert child_pose.y_m == pytest.approx(2.0 + spacing)
    assert child_pose.yaw_rad == pytest.approx(math.pi / 2.0)


def test_angle_is_normalized() -> None:
    assert normalize_angle(2.0 * math.pi) == pytest.approx(0.0)
    assert normalize_angle(3.0 * math.pi) == pytest.approx(-math.pi)


def test_invalid_geometry_is_rejected() -> None:
    with pytest.raises(
        PlanarUnfoldingError,
        match="positive",
    ):
        PlanarModuleGeometry(
            top_offset_m=-0.01,
        )