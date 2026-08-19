"""Tests for the deterministic SMORES-EP topology representation."""

from __future__ import annotations

import pytest

from mssr_expert.planning.smores_ep.topology import (
    SmoresKinematicTree,
    SmoresTopologyEdge,
    TopologyValidationError,
    choose_target_root,
    graph_centers,
    validate_kinematic_tree,
)


def test_three_module_chain_is_valid() -> None:
    tree = SmoresKinematicTree(
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

    validate_kinematic_tree(tree)

    assert graph_centers(tree) == ("v1",)
    assert choose_target_root(tree) == "v1"


def test_cycle_is_rejected() -> None:
    tree = SmoresKinematicTree(
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
            SmoresTopologyEdge(
                vertex_a="v2",
                face_a="TOP",
                vertex_b="v0",
                face_b="BOTTOM",
            ),
        ),
    )

    with pytest.raises(TopologyValidationError):
        validate_kinematic_tree(tree)


def test_reused_connector_is_rejected() -> None:
    tree = SmoresKinematicTree(
        vertex_ids=("v0", "v1", "v2"),
        edges=(
            SmoresTopologyEdge(
                vertex_a="v0",
                face_a="TOP",
                vertex_b="v1",
                face_b="BOTTOM",
            ),
            SmoresTopologyEdge(
                vertex_a="v0",
                face_a="TOP",
                vertex_b="v2",
                face_b="BOTTOM",
            ),
        ),
    )

    with pytest.raises(
        TopologyValidationError,
        match="used more than once",
    ):
        validate_kinematic_tree(tree)


def test_four_module_chain_has_two_centers() -> None:
    tree = SmoresKinematicTree(
        vertex_ids=("v0", "v1", "v2", "v3"),
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
            SmoresTopologyEdge(
                vertex_a="v2",
                face_a="TOP",
                vertex_b="v3",
                face_b="BOTTOM",
            ),
        ),
    )

    assert graph_centers(tree) == ("v1", "v2")
    assert choose_target_root(tree) == "v1"


def test_invalid_face_is_rejected() -> None:
    tree = SmoresKinematicTree(
        vertex_ids=("v0", "v1"),
        edges=(
            SmoresTopologyEdge(
                vertex_a="v0",
                face_a="FRONT",
                vertex_b="v1",
                face_b="BOTTOM",
            ),
        ),
    )

    with pytest.raises(
        TopologyValidationError,
        match="Invalid face",
    ):
        validate_kinematic_tree(tree)


def test_disconnected_topology_is_rejected() -> None:
    tree = SmoresKinematicTree(
        vertex_ids=("v0", "v1", "v2", "v3"),
        edges=(
            SmoresTopologyEdge(
                vertex_a="v0",
                face_a="TOP",
                vertex_b="v1",
                face_b="BOTTOM",
            ),
            SmoresTopologyEdge(
                vertex_a="v2",
                face_a="TOP",
                vertex_b="v3",
                face_b="BOTTOM",
            ),
            SmoresTopologyEdge(
                vertex_a="v2",
                face_a="LEFT",
                vertex_b="v3",
                face_b="RIGHT",
            ),
        ),
    )

    with pytest.raises(TopologyValidationError):
        validate_kinematic_tree(tree)


def test_single_module_is_a_valid_tree() -> None:
    tree = SmoresKinematicTree(
        vertex_ids=("v0",),
        edges=(),
    )

    validate_kinematic_tree(tree)

    assert graph_centers(tree) == ("v0",)
    assert choose_target_root(tree) == "v0"