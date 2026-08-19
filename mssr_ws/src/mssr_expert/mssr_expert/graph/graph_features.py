"""Feature extraction from attributed graphs for future GNN/MARL use."""
from __future__ import annotations

from typing import Any, Mapping

from mssr_expert.graph.attributed_robot_graph import AttributedRobotGraph


ROLE_INDEX = {
    "unassigned": 0.0,
    "mobile": 1.0,
    "anchor": 2.0,
    "frontier_anchor": 3.0,
    "base": 4.0,
    "bridge_part": 5.0,
    "climber": 6.0,
    "stabilizer": 7.0,
    "support_transfer": 8.0,
    "recovery": 9.0,
}

ATTACHMENT_MODE_INDEX = {
    "rolling_contact": 0.0,
    "rigid_lock": 1.0,
    "surface_pivot": 2.0,
    "support_contact": 3.0,
    "bridge_link": 4.0,
    "anchor_link": 5.0,
    "transfer_link": 6.0,
}

NODE_TYPE_INDEX = {
    "physical_module": 0.0,
    "target_slot": 1.0,
}

RELATION_TYPE_INDEX = {
    "relation": 0.0,
    "contact": 1.0,
    "current_connection": 2.0,
    "target_connection": 3.0,
    "assignment": 4.0,
    "support": 5.0,
}


def graph_to_features(graph: AttributedRobotGraph) -> dict[str, Any]:
    """Convert an attributed graph into simple numeric feature arrays."""
    node_ids = [node.node_id for node in graph.nodes]
    node_index = {module_id: index for index, module_id in enumerate(node_ids)}
    node_features = [
        _node_features(
            node.attributes,
            len(graph.adjacency().get(node.node_id, ())),
        )
        for node in graph.nodes
    ]
    edge_index: list[list[int]] = [[], []]
    edge_features = []
    for edge in graph.edges:
        source = node_index.get(edge.module_a_id)
        target = node_index.get(edge.module_b_id)
        if source is None or target is None:
            continue
        edge_index[0].extend([source, target])
        edge_index[1].extend([target, source])
        features = _edge_features(edge.attributes)
        edge_features.extend([features, features])
    return {
        "node_ids": node_ids,
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "global_features": _global_features(graph),
    }


def _node_features(attributes: Any, degree: int) -> list[float]:
    attrs = attributes if isinstance(attributes, Mapping) else {}
    position = _vector(attrs.get("position"), 3)
    rel_position = _vector(attrs.get("position_relative_to_centroid"), 3)
    linear_velocity = _vector(attrs.get("linear_velocity"), 3)
    angular_velocity = _vector(attrs.get("angular_velocity"), 3)
    return [
        *position,
        *rel_position,
        *linear_velocity,
        *angular_velocity,
        float(attrs.get("radius", 0.0)),
        float(attrs.get("mass", 0.0)),
        float(attrs.get("distance_to_goal") or 0.0),
        ROLE_INDEX.get(str(attrs.get("role", "unassigned")), 0.0),
        ATTACHMENT_MODE_INDEX.get(str(attrs.get("attachment_mode", "rolling_contact")), 0.0),
        1.0 if attrs.get("is_anchor") else 0.0,
        1.0 if attrs.get("is_mobile") else 0.0,
        1.0 if attrs.get("is_climber") else 0.0,
        1.0 if attrs.get("is_bridge_part") else 0.0,
        1.0 if attrs.get("is_base") else 0.0,
        float(degree),
        NODE_TYPE_INDEX.get(
            str(attrs.get("node_type", "physical_module")),
            0.0,
        ),
        1.0 if attrs.get("is_target_node") else 0.0,
        1.0 if attrs.get("is_target_root") else 0.0,
    ]


def _edge_features(attributes: Any) -> list[float]:
    attrs = attributes if isinstance(attributes, Mapping) else {}
    contact_normal = _vector(attrs.get("contact_normal"), 3)
    pivot_axis = _vector(attrs.get("pivot_axis"), 3)
    return [
        1.0 if attrs.get("is_contact") else 0.0,
        1.0 if attrs.get("is_attached") else 0.0,
        1.0 if attrs.get("is_target_edge") else 0.0,
        1.0 if attrs.get("is_support_edge") else 0.0,
        1.0 if attrs.get("is_load_bearing") else 0.0,
        1.0 if attrs.get("is_temporary") else 0.0,
        ATTACHMENT_MODE_INDEX.get(str(attrs.get("attachment_mode", "rolling_contact")), 0.0),
        1.0 if attrs.get("allows_rotation") else 0.0,
        float(attrs.get("distance_error") or 0.0),
        float(attrs.get("priority") or 0.0),
        *contact_normal,
        *pivot_axis,
        RELATION_TYPE_INDEX.get(
            str(attrs.get("relation_type", "relation")),
            0.0,
        ),
        1.0 if attrs.get("is_assignment") else 0.0,
    ]


def _global_features(graph: AttributedRobotGraph) -> dict[str, float | str]:
    attrs = dict(graph.global_attributes)
    return {
        "stage_id": float(attrs.get("stage_id", 0.0)),
        "difficulty": float(attrs.get("difficulty", 0.0)),
        "task_type": str(attrs.get("task_type", "")),
        "distance_to_goal": float(attrs.get("distance_to_goal") or 0.0),
        "assembled_ratio": float(attrs.get("assembled_ratio") or 0.0),
        "component_count": float(attrs.get("component_count") or 0.0),
        "max_height": float(attrs.get("max_height") or 0.0),
        "height_gain": float(attrs.get("height_gain") or 0.0),
        "attachment_count": float(attrs.get("attachment_count") or 0.0),
        "support_edge_count": float(attrs.get("support_edge_count") or 0.0),
        "pivot_edge_count": float(attrs.get("pivot_edge_count") or 0.0),
    }


def _vector(value: Any, length: int) -> list[float]:
    if not isinstance(value, list | tuple) or len(value) < length:
        return [0.0 for _ in range(length)]
    return [float(value[index]) for index in range(length)]
