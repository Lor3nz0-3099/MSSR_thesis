"""Tests for SMORES-EP target and physical root selection."""

from __future__ import annotations

import pytest

from mssr_expert.planning.smores_ep.rooting import (
    ModulePosition,
    PhysicalRootSelectionError,
    choose_physical_root,
    root_kinematic_tree,
    swarm_centroid,
    vertices_by_depth,
)
from mssr_expert.planning.smores_ep.topology import (
    SmoresKinematicTree,
    SmoresTopologyEdge,
    TopologyValidationError,
)


def _three_module_chain() -> SmoresKinematicTree:
    return SmoresKinematicTree(
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
                clocking_quarter_turns=1,
            ),
        ),
    )


def test_tree_is_rooted_from_its_center() -> None:
    rooted = root_kinematic_tree(_three_module_chain())

    assert rooted.root_id == "v1"

    assert rooted.parent_by_vertex == {
        "v1": None,
        "v0": "v1",
        "v2": "v1",
    }

    assert rooted.depth_by_vertex == {
        "v1": 0,
        "v0": 1,
        "v2": 1,
    }

    assert vertices_by_depth(rooted) == (
        ("v1",),
        ("v0", "v2"),
    )


def test_connection_faces_are_preserved_when_edge_is_reversed() -> None:
    rooted = root_kinematic_tree(
        _three_module_chain(),
        root_id="v1",
    )

    edges_by_child = {
        edge.child_vertex: edge
        for edge in rooted.edges
    }

    edge_to_v0 = edges_by_child["v0"]

    assert edge_to_v0.parent_vertex == "v1"
    assert edge_to_v0.parent_face == "BOTTOM"
    assert edge_to_v0.child_face == "TOP"

    edge_to_v2 = edges_by_child["v2"]

    assert edge_to_v2.parent_vertex == "v1"
    assert edge_to_v2.parent_face == "TOP"
    assert edge_to_v2.child_face == "BOTTOM"
    assert edge_to_v2.clocking_quarter_turns == 1


def test_unknown_root_is_rejected() -> None:
    with pytest.raises(
        TopologyValidationError,
        match="unknown vertex",
    ):
        root_kinematic_tree(
            _three_module_chain(),
            root_id="missing",
        )


def test_swarm_centroid_is_calculated() -> None:
    modules = (
        ModulePosition("m0", 0.0, 0.0, 0.0),
        ModulePosition("m1", 2.0, 0.0, 0.0),
        ModulePosition("m2", 4.0, 3.0, 0.0),
    )

    centroid = swarm_centroid(modules)

    assert centroid == pytest.approx(
        (2.0, 1.0, 0.0)
    )


def test_physical_root_is_closest_to_centroid() -> None:
    modules = (
        ModulePosition("m0", 0.0, 0.0),
        ModulePosition("m1", 2.0, 0.0),
        ModulePosition("m2", 3.0, 0.0),
    )

    assert choose_physical_root(modules) == "m1"


def test_physical_root_tie_is_resolved_by_module_id() -> None:
    modules = (
        ModulePosition("module_b", -1.0, 0.0),
        ModulePosition("module_a", 1.0, 0.0),
    )

    assert choose_physical_root(modules) == "module_a"


def test_empty_physical_swarm_is_rejected() -> None:
    with pytest.raises(
        PhysicalRootSelectionError,
        match="At least one",
    ):
        choose_physical_root(())


def test_duplicate_physical_module_ids_are_rejected() -> None:
    modules = (
        ModulePosition("m0", 0.0, 0.0),
        ModulePosition("m0", 1.0, 0.0),
    )

    with pytest.raises(
        PhysicalRootSelectionError,
        match="unique",
    ):
        choose_physical_root(modules)