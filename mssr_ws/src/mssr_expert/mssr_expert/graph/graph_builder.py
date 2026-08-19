"""Build attributed graphs from bridge observations and graph payloads."""
from __future__ import annotations

import math
from typing import Any, Mapping

from mssr_expert.graph.attributed_robot_graph import (
    AttributedRobotGraph,
    GraphEdge,
    GraphNode,
)


class GraphBuilder:
    """Normalize bridge JSON payloads into an attributed robot graph."""

    def build(
        self,
        observation: Mapping[str, Any],
        graph_payload: Mapping[str, Any] | None = None,
    ) -> AttributedRobotGraph:
        """Build a graph from observation modules plus graph/contact payloads."""
        graph_payload = graph_payload or {}
        stamp = self._stamp(observation, graph_payload)
        modules = self._extract_modules(observation, graph_payload)
        centroid = self._centroid(modules)
        goal = self._goal_position(observation)
        nodes = tuple(
            self._build_node(module_id, module, centroid, goal)
            for module_id, module in sorted(modules.items())
        )
        edges = self._build_edges(observation, graph_payload)
        global_attributes = self._global_attributes(observation, nodes, edges, goal)
        return AttributedRobotGraph(
            stamp=stamp,
            nodes=nodes,
            edges=edges,
            global_attributes=global_attributes,
        )

    def _extract_modules(
        self,
        observation: Mapping[str, Any],
        graph_payload: Mapping[str, Any],
    ) -> dict[str, Mapping[str, Any]]:
        modules: dict[str, Mapping[str, Any]] = {}
        raw_modules = observation.get("modules")
        if isinstance(raw_modules, Mapping):
            for module_id, payload in raw_modules.items():
                if isinstance(module_id, str) and isinstance(payload, Mapping):
                    modules[module_id] = payload
        elif isinstance(raw_modules, list):
            for item in raw_modules:
                if not isinstance(item, Mapping):
                    continue
                module_id = item.get("module_id")
                if isinstance(module_id, str):
                    modules[module_id] = item

        for item in graph_payload.get("nodes", []):
            if not isinstance(item, Mapping):
                continue
            attributes = item.get("attributes", {})
            if not isinstance(attributes, Mapping):
                attributes = {}
            module_id = item.get("module_id") or item.get("id") or attributes.get("module_id")
            if isinstance(module_id, str) and module_id not in modules:
                modules[module_id] = attributes
        return modules

    def _build_node(
        self,
        module_id: str,
        payload: Mapping[str, Any],
        centroid: tuple[float, float, float],
        goal: tuple[float, float, float] | None,
    ) -> GraphNode:
        position = self._position(payload)
        role = str(
            payload.get("current_role", payload.get("role", "unassigned"))
        )
        attachment_mode = str(payload.get("attachment_mode", "rolling_contact"))
        functional_role = payload.get("functional_role", {})
        if not isinstance(functional_role, Mapping):
            functional_role = {}
        actuators = payload.get("actuators", {})
        if not isinstance(actuators, Mapping):
            actuators = {}
        connectors = payload.get("connectors", [])
        if not isinstance(connectors, list | tuple):
            connectors = []
        simulation_fixtures = payload.get("simulation_fixtures", {})
        if not isinstance(simulation_fixtures, Mapping):
            simulation_fixtures = {}
        design_requirements = payload.get("design_requirements", {})
        if not isinstance(design_requirements, Mapping):
            design_requirements = {}
        distance_to_goal = None
        if goal is not None:
            distance_to_goal = self._distance(position, goal)
        attributes = {
            "module_id": module_id,
            "robot_family": str(payload.get("robot_family", "unknown")),
            "module_type": str(payload.get("module_type", "unknown")),
            "design_profile": str(payload.get("design_profile", "")),
            "design_requirements": dict(design_requirements),
            "position": list(position),
            "position_relative_to_centroid": [
                position[0] - centroid[0],
                position[1] - centroid[1],
                position[2] - centroid[2],
            ],
            "orientation": self._orientation(payload),
            "linear_velocity": self._vector(payload.get("linear_velocity"), 3),
            "angular_velocity": self._vector(payload.get("angular_velocity"), 3),
            "radius": float(payload.get("radius", 0.0)),
            "mass": float(payload.get("mass", 0.0)),
            "distance_to_goal": distance_to_goal,
            "role": role,
            "current_role": role,
            "target_role": str(payload.get("target_role", "")),
            "role_confidence": float(payload.get("role_confidence", 0.0)),
            "role_source": str(payload.get("role_source", "unknown")),
            "functional_role": dict(functional_role),
            "locomotion_state": str(payload.get("locomotion_state", "")),
            "magnet_state": str(payload.get("magnet_state", "")),
            "attachment_mode": attachment_mode,
            "supporting_module": payload.get("supporting_module"),
            "supported_modules": list(payload.get("supported_modules", [])),
            "is_anchor": role in ("anchor", "frontier_anchor"),
            "is_mobile": role in ("mobile", "climber", "recovery", "unassigned"),
            "is_climber": role == "climber",
            "is_bridge_part": role == "bridge_part",
            "is_base": role == "base",
            "capabilities": list(payload.get("capabilities", [])),
            "sensor_capabilities": list(
                payload.get("sensor_capabilities", [])
            ),
            "observation_confidence": float(
                payload.get("observation_confidence", 1.0)
            ),
            "control_available": bool(
                payload.get("control_available", True)
            ),
            "health": str(payload.get("health", "unknown")),
            "actuators": dict(actuators),
            "connectors": [
                dict(connector)
                for connector in connectors
                if isinstance(connector, Mapping)
            ],
            "simulation_fixtures": dict(simulation_fixtures),
        }
        return GraphNode(module_id=module_id, attributes=attributes)

    def _build_edges(
        self,
        observation: Mapping[str, Any],
        graph_payload: Mapping[str, Any],
    ) -> tuple[GraphEdge, ...]:
        edges: dict[tuple[str, ...], GraphEdge] = {}
        for edge in graph_payload.get("edges", []):
            parsed = self._edge_from_payload(edge)
            if parsed is not None:
                edges[parsed.key] = parsed
        for contact in observation.get("contacts", []):
            parsed = self._edge_from_relation(
                contact,
                is_contact=True,
                relation_type="contact",
            )
            if parsed is not None:
                edges[parsed.key] = self._merge_edge(edges.get(parsed.key), parsed)
        for attachment in observation.get("attachments", []):
            parsed = self._edge_from_relation(
                attachment,
                is_attached=True,
                relation_type="current_connection",
            )
            if parsed is not None:
                edges[parsed.key] = self._merge_edge(edges.get(parsed.key), parsed)
        return tuple(edges[key] for key in sorted(edges))

    def _edge_from_payload(self, payload: Any) -> GraphEdge | None:
        if not isinstance(payload, Mapping):
            return None
        attributes = payload.get("attributes", {})
        if not isinstance(attributes, Mapping):
            attributes = {}
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
        if not isinstance(module_a_id, str) or not isinstance(module_b_id, str):
            return None
        merged = dict(attributes)
        for key, value in payload.items():
            if key != "attributes":
                merged.setdefault(str(key), value)
        return GraphEdge(module_a_id, module_b_id, self._edge_attributes(merged))

    def _edge_from_relation(
        self,
        payload: Any,
        is_contact: bool = False,
        is_attached: bool = False,
        relation_type: str | None = None,
    ) -> GraphEdge | None:
        if not isinstance(payload, Mapping):
            return None
        module_a_id = payload.get("module_a_id")
        module_b_id = payload.get("module_b_id")
        if not isinstance(module_a_id, str) or not isinstance(module_b_id, str):
            return None
        normalized_payload = dict(payload)
        if relation_type is not None:
            normalized_payload.setdefault("relation_type", relation_type)
        attributes = self._edge_attributes(normalized_payload)
        attributes["is_contact"] = bool(attributes.get("is_contact", False) or is_contact)
        attributes["is_attached"] = bool(attributes.get("is_attached", False) or is_attached)
        return GraphEdge(module_a_id, module_b_id, attributes)

    def _edge_attributes(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        is_contact = bool(payload.get("is_contact", False))
        is_attached = bool(
            payload.get("is_attached", False)
            or payload.get("is_magnet_enabled", False)
            or payload.get("status") == "connected"
        )
        is_target_edge = bool(payload.get("is_target_edge", False))
        is_assignment = bool(payload.get("is_assignment", False))
        relation_type = payload.get("relation_type")
        if not relation_type:
            if is_assignment:
                relation_type = "assignment"
            elif is_target_edge:
                relation_type = "target_connection"
            elif is_attached:
                relation_type = "current_connection"
            elif is_contact:
                relation_type = "contact"
            else:
                relation_type = payload.get("edge_type", "relation")
        return {
            "relation_type": str(relation_type),
            "edge_id": payload.get("edge_id"),
            "is_contact": is_contact,
            "is_attached": is_attached,
            "is_target_edge": is_target_edge,
            "is_assignment": is_assignment,
            "is_support_edge": bool(payload.get("is_support_edge", False)),
            "is_load_bearing": bool(payload.get("is_load_bearing", False)),
            "is_temporary": bool(payload.get("is_temporary", True)),
            "attachment_mode": str(payload.get("attachment_mode", "rolling_contact")),
            "edge_type": str(payload.get("edge_type", "relation")),
            "connection_id": payload.get("connection_id"),
            "connection_state": str(
                payload.get("connection_state", payload.get("status", ""))
            ),
            "connector_type": str(payload.get("connector_type", "")),
            "connector_a_id": (
                payload.get("connector_a_id") or payload.get("face_a")
            ),
            "connector_b_id": (
                payload.get("connector_b_id") or payload.get("face_b")
            ),
            "face_a": payload.get("face_a") or payload.get("connector_a_id"),
            "face_b": payload.get("face_b") or payload.get("connector_b_id"),
            "joint_type": payload.get("joint_type"),
            "contact_point_world": self._optional_vector(payload.get("contact_point_world"), 3),
            "contact_point_a_local": self._optional_vector(payload.get("contact_point_a_local"), 3),
            "contact_point_b_local": self._optional_vector(payload.get("contact_point_b_local"), 3),
            "contact_normal": self._optional_vector(payload.get("contact_normal"), 3),
            "pivot_axis": self._optional_vector(payload.get("pivot_axis"), 3),
            "allows_rotation": bool(payload.get("allows_rotation", False)),
            "allowed_relative_dofs": list(
                payload.get("allowed_relative_dofs", [])
            ),
            "clocking_quarter_turns": payload.get(
                "clocking_quarter_turns"
            ),
            "clocking_error_rad": payload.get("clocking_error_rad"),
            "relative_transform": payload.get("relative_transform"),
            "health": str(payload.get("health", "unknown")),
            "distance_error": payload.get("distance_error"),
            "edge_role": payload.get("edge_role") or payload.get("support_role", ""),
            "priority": float(payload.get("priority", 0.0)),
            "in_main_component": bool(payload.get("in_main_component", False)),
        }

    def _merge_edge(self, current: GraphEdge | None, update: GraphEdge) -> GraphEdge:
        if current is None:
            return update
        attributes = dict(current.attributes)
        attributes.update(update.attributes)
        return GraphEdge(current.module_a_id, current.module_b_id, attributes)

    def _global_attributes(
        self,
        observation: Mapping[str, Any],
        nodes: tuple[GraphNode, ...],
        edges: tuple[GraphEdge, ...],
        goal: tuple[float, float, float] | None,
    ) -> dict[str, Any]:
        metrics = observation.get("task_metrics", {})
        if not isinstance(metrics, Mapping):
            metrics = {}
        max_height = max(
            (float(node.attributes["position"][2]) for node in nodes),
            default=0.0,
        )
        distance_to_goal = metrics.get("distance_to_goal")
        if distance_to_goal is None and goal is not None and nodes:
            centroid = self._centroid(
                {
                    node.module_id: {"position": node.attributes["position"]}
                    for node in nodes
                }
            )
            distance_to_goal = self._distance(centroid, goal)
        return {
            "stage_id": int(observation.get("stage_id", 0)),
            "difficulty": float(observation.get("difficulty", 0.0)),
            "task_type": str(observation.get("task_type", "")),
            "goal": list(goal) if goal is not None else None,
            "distance_to_goal": distance_to_goal,
            "assembled_ratio": metrics.get("assembled_ratio", 0.0),
            "component_count": metrics.get("component_count"),
            "max_height": max_height,
            "height_gain": metrics.get("height_gain", metrics.get("height_reached")),
            "attachment_count": sum(1 for edge in edges if edge.attributes.get("is_attached")),
            "support_edge_count": sum(1 for edge in edges if edge.attributes.get("is_support_edge")),
            "pivot_edge_count": sum(
                1
                for edge in edges
                if edge.attributes.get("attachment_mode") == "surface_pivot"
            ),
            "task_metrics": dict(metrics),
        }

    def _stamp(
        self,
        observation: Mapping[str, Any],
        graph_payload: Mapping[str, Any],
    ) -> float:
        stamp = observation.get("stamp", observation.get("timestamp"))
        if stamp is None:
            stamp = graph_payload.get("stamp", graph_payload.get("timestamp", 0.0))
        return float(stamp or 0.0)

    def _goal_position(self, observation: Mapping[str, Any]) -> tuple[float, float, float] | None:
        goal = observation.get("goal")
        if isinstance(goal, Mapping):
            goal = goal.get("position")
        if self._is_vector(goal, 3):
            return (float(goal[0]), float(goal[1]), float(goal[2]))
        return None

    def _centroid(self, modules: Mapping[str, Mapping[str, Any]]) -> tuple[float, float, float]:
        if not modules:
            return (0.0, 0.0, 0.0)
        positions = [self._position(module) for module in modules.values()]
        count = float(len(positions))
        return (
            sum(position[0] for position in positions) / count,
            sum(position[1] for position in positions) / count,
            sum(position[2] for position in positions) / count,
        )

    def _position(self, payload: Mapping[str, Any]) -> tuple[float, float, float]:
        pose = payload.get("pose", {})
        position = pose.get("position") if isinstance(pose, Mapping) else None
        if position is None:
            position = payload.get("position")
        return self._vector(position, 3)

    def _orientation(self, payload: Mapping[str, Any]) -> list[float]:
        pose = payload.get("pose", {})
        orientation = pose.get("orientation") if isinstance(pose, Mapping) else None
        if orientation is None and isinstance(pose, Mapping):
            orientation = pose.get("orientation_xyzw")
        if orientation is None:
            orientation = payload.get("orientation")
        return list(self._vector(orientation, 4, default=(0.0, 0.0, 0.0, 1.0)))

    def _vector(
        self,
        value: Any,
        length: int,
        default: tuple[float, ...] | None = None,
    ) -> tuple[float, ...]:
        fallback = default if default is not None else tuple(0.0 for _ in range(length))
        if not self._is_vector(value, length):
            return fallback
        return tuple(float(value[index]) for index in range(length))

    def _optional_vector(self, value: Any, length: int) -> list[float] | None:
        if not self._is_vector(value, length):
            return None
        return [float(value[index]) for index in range(length)]

    def _is_vector(self, value: Any, length: int) -> bool:
        return isinstance(value, list | tuple) and len(value) >= length

    def _distance(
        self,
        position: tuple[float, float, float],
        goal: tuple[float, float, float],
    ) -> float:
        return math.sqrt(
            (position[0] - goal[0]) ** 2
            + (position[1] - goal[1]) ** 2
            + (position[2] - goal[2]) ** 2
        )
