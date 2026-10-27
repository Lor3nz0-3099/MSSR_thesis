"""Topology-authoritative morphology detection for teleoperation."""

from __future__ import annotations

from collections.abc import Mapping

from mssr_expert.teleop.state import ACTIVE_MORPHOLOGIES



def is_authoritative_loose_graph(
    current_graph,
    *,
    expected_module_count: int = 8,
) -> bool:
    """Return True only for a proven all-disconnected physical module set."""

    if current_graph is None:
        return False

    try:
        physical_nodes = tuple(
            node
            for node in current_graph.nodes
            if node.node_type == "physical_module"
        )

        module_ids = {
            node.module_id
            for node in physical_nodes
        }

        if (
            len(physical_nodes) != expected_module_count
            or len(module_ids) != expected_module_count
        ):
            return False

        if any(
            str(
                node.attributes.get(
                    "robot_family",
                    "",
                )
            ).lower()
            not in {
                "smores_ep",
                "smores-ep",
            }
            for node in physical_nodes
        ):
            return False

        attached_edges = tuple(
            edge
            for edge in current_graph.edges
            if (
                edge.relation_type == "current_connection"
                and bool(
                    edge.attributes.get(
                        "is_attached",
                        True,
                    )
                )
            )
        )

        return not attached_edges

    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
    ):
        return False


class TeleopTopologyDetector:
    """Detect exactly one supported morphology from the live robot graph."""

    def __init__(
        self,
        *,
        catalog: Mapping[str, object],
        matcher,
    ) -> None:
        self._catalog = {
            str(name): graph
            for name, graph in catalog.items()
            if str(name) in ACTIVE_MORPHOLOGIES
        }
        self._matcher = matcher

    def detect(self, current_graph) -> str | None:
        if current_graph is None:
            return None

        matches: list[str] = []

        try:
            for morphology_name in sorted(self._catalog):
                target_graph = self._catalog[morphology_name]

                assignment = self._matcher.configuration_assignment(
                    current_graph,
                    target_graph,
                )

                if assignment is not None:
                    matches.append(morphology_name)

        except (KeyError, RuntimeError, TypeError, ValueError):
            return None

        if len(matches) != 1:
            return None

        return matches[0]
