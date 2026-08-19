"""Serialization helpers for canonical attributed graphs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from mssr_expert.graph.attributed_robot_graph import (
    AttributedRobotGraph,
    GraphEdge,
    GraphNode,
)


class GraphSerializationError(ValueError):
    """Raised when an attributed graph payload is invalid."""


def attributed_graph_from_dict(
    payload: Mapping[str, Any],
) -> AttributedRobotGraph:
    """Deserialize a canonical attributed graph."""

    raw_nodes = payload.get("nodes", ())
    raw_edges = payload.get("edges", ())

    if not isinstance(raw_nodes, list | tuple):
        raise GraphSerializationError(
            "Graph nodes must be an array."
        )

    if not isinstance(raw_edges, list | tuple):
        raise GraphSerializationError(
            "Graph edges must be an array."
        )

    nodes = tuple(
        _node_from_dict(raw_node)
        for raw_node in raw_nodes
    )

    node_ids = {
        node.node_id
        for node in nodes
    }

    if len(node_ids) != len(nodes):
        raise GraphSerializationError(
            "Graph node IDs must be unique."
        )

    edges = tuple(
        _edge_from_dict(
            raw_edge,
            known_node_ids=node_ids,
        )
        for raw_edge in raw_edges
    )

    stamp = float(payload.get("stamp", 0.0))

    if not math.isfinite(stamp):
        raise GraphSerializationError(
            "Graph timestamp must be finite."
        )

    global_attributes = payload.get(
        "global_attributes",
        {},
    )

    if not isinstance(global_attributes, Mapping):
        raise GraphSerializationError(
            "global_attributes must be an object."
        )

    schema_version = payload.get("schema_version")

    if schema_version is not None:
        global_attributes = {
            **dict(global_attributes),
            "schema_version": str(schema_version),
        }

    return AttributedRobotGraph(
        stamp=stamp,
        nodes=nodes,
        edges=edges,
        global_attributes=dict(global_attributes),
    )


def load_attributed_graph(
    path: Path,
) -> AttributedRobotGraph:
    """Load an attributed graph from a JSON file."""

    try:
        text = path.read_text(
            encoding="utf-8",
        )
    except FileNotFoundError as error:
        raise GraphSerializationError(
            f"Graph file does not exist: {path}."
        ) from error

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise GraphSerializationError(
            f"Invalid JSON in graph file {path}: {error}."
        ) from error

    if not isinstance(payload, Mapping):
        raise GraphSerializationError(
            "The graph JSON root must be an object."
        )

    return attributed_graph_from_dict(payload)


def save_attributed_graph(
    graph: AttributedRobotGraph,
    path: Path,
) -> None:
    """Write an attributed graph as formatted JSON."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary_path.write_text(
        json.dumps(
            graph.to_dict(),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(path)


def _node_from_dict(
    payload: Any,
) -> GraphNode:
    """Deserialize one physical or logical graph node."""

    if not isinstance(payload, Mapping):
        raise GraphSerializationError(
            "Every graph node must be an object."
        )

    attributes = payload.get(
        "attributes",
        {},
    )

    if not isinstance(attributes, Mapping):
        raise GraphSerializationError(
            "Node attributes must be an object."
        )

    node_id = (
        payload.get("node_id")
        or payload.get("module_id")
        or payload.get("id")
        or attributes.get("node_id")
        or attributes.get("module_id")
    )

    if (
        not isinstance(node_id, str)
        or not node_id.strip()
    ):
        raise GraphSerializationError(
            "Every graph node requires a non-empty ID."
        )

    merged_attributes = dict(attributes)

    for key, value in payload.items():
        if key not in {
            "node_id",
            "module_id",
            "id",
            "attributes",
        }:
            merged_attributes.setdefault(
                str(key),
                value,
            )

    return GraphNode(
        module_id=node_id,
        attributes=merged_attributes,
    )


def _edge_from_dict(
    payload: Any,
    known_node_ids: set[str],
) -> GraphEdge:
    """Deserialize one attributed multigraph edge."""

    if not isinstance(payload, Mapping):
        raise GraphSerializationError(
            "Every graph edge must be an object."
        )

    attributes = payload.get(
        "attributes",
        {},
    )

    if not isinstance(attributes, Mapping):
        raise GraphSerializationError(
            "Edge attributes must be an object."
        )

    module_a_id = (
        payload.get("module_a_id")
        or payload.get("source")
        or attributes.get("module_a_id")
    )

    module_b_id = (
        payload.get("module_b_id")
        or payload.get("target")
        or attributes.get("module_b_id")
    )

    if (
        not isinstance(module_a_id, str)
        or not isinstance(module_b_id, str)
    ):
        raise GraphSerializationError(
            "Every graph edge requires two endpoint IDs."
        )

    if module_a_id not in known_node_ids:
        raise GraphSerializationError(
            f"Unknown edge endpoint {module_a_id!r}."
        )

    if module_b_id not in known_node_ids:
        raise GraphSerializationError(
            f"Unknown edge endpoint {module_b_id!r}."
        )

    merged_attributes = dict(attributes)

    for key, value in payload.items():
        if key not in {
            "module_a_id",
            "module_b_id",
            "source",
            "target",
            "attributes",
        }:
            merged_attributes.setdefault(
                str(key),
                value,
            )

    edge = GraphEdge(
        module_a_id=module_a_id,
        module_b_id=module_b_id,
        attributes=merged_attributes,
    )

    return GraphEdge(
        module_a_id=edge.module_a_id,
        module_b_id=edge.module_b_id,
        attributes={
            **dict(edge.attributes),
            "relation_type": edge.relation_type,
        },
    )