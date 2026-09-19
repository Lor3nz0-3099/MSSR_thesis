"""Topology-authoritative morphology detection for teleoperation."""

from __future__ import annotations

from collections.abc import Mapping

from mssr_expert.teleop.state import ACTIVE_MORPHOLOGIES


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
