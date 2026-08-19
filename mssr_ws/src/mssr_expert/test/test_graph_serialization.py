"""Tests for attributed graph serialization."""

from __future__ import annotations

import json

import pytest

from mssr_expert.graph.attributed_robot_graph import (
    AttributedRobotGraph,
    GraphEdge,
    GraphNode,
)
from mssr_expert.graph.serialization import (
    GraphSerializationError,
    attributed_graph_from_dict,
    load_attributed_graph,
    save_attributed_graph,
)


def _target_graph() -> AttributedRobotGraph:
    return AttributedRobotGraph(
        nodes=(
            GraphNode(
                "v0",
                {
                    "node_type": "target_slot",
                    "target_vertex_id": "v0",
                    "target_role": "support",
                },
            ),
            GraphNode(
                "v1",
                {
                    "node_type": "target_slot",
                    "target_vertex_id": "v1",
                    "target_role": "link",
                },
            ),
        ),
        edges=(
            GraphEdge(
                "v0",
                "v1",
                {
                    "relation_type": "target_connection",
                    "is_target_edge": True,
                    "face_a": "TOP",
                    "face_b": "BOTTOM",
                    "clocking_quarter_turns": 0,
                },
            ),
        ),
        global_attributes={
            "schema_version": "mssr.target_graph.v1",
            "graph_kind": "target_morphology",
        },
    )


def test_graph_round_trip_preserves_target_information() -> None:
    original = _target_graph()

    restored = attributed_graph_from_dict(
        original.to_dict()
    )

    assert {
        node.node_id for node in restored.nodes
    } == {"v0", "v1"}

    assert restored.nodes[0].attributes[
        "node_type"
    ] == "target_slot"

    assert len(
        restored.edges_matching("target_connection")
    ) == 1

    edge = restored.edges[0]

    assert edge.attributes["face_a"] == "TOP"
    assert edge.attributes["face_b"] == "BOTTOM"
    assert edge.attributes[
        "clocking_quarter_turns"
    ] == 0


def test_graph_can_be_saved_and_loaded(tmp_path) -> None:
    path = tmp_path / "target.json"

    save_attributed_graph(
        _target_graph(),
        path,
    )

    restored = load_attributed_graph(path)

    assert restored.global_attributes[
        "graph_kind"
    ] == "target_morphology"

    assert len(restored.nodes) == 2
    assert len(restored.edges) == 1


def test_unknown_edge_endpoint_is_rejected() -> None:
    payload = _target_graph().to_dict()

    payload["edges"][0][
        "module_b_id"
    ] = "missing"

    with pytest.raises(
        GraphSerializationError,
        match="Unknown edge endpoint",
    ):
        attributed_graph_from_dict(payload)


def test_invalid_json_is_rejected(tmp_path) -> None:
    path = tmp_path / "invalid.json"

    path.write_text(
        "{invalid",
        encoding="utf-8",
    )

    with pytest.raises(
        GraphSerializationError,
        match="Invalid JSON",
    ):
        load_attributed_graph(path)


def test_saved_file_contains_schema(tmp_path) -> None:
    path = tmp_path / "target.json"

    save_attributed_graph(
        _target_graph(),
        path,
    )

    payload = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    assert payload["schema_version"] == (
        "mssr.target_graph.v1"
    )