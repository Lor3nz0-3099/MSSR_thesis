"""Registry for deterministic expert classes."""
from __future__ import annotations

from typing import Type

from mssr_expert.experts.base_expert import BaseExpert


class ExpertRegistry:
    """Map task names to expert classes."""

    def __init__(self) -> None:
        self._classes: dict[str, Type[BaseExpert]] = {}

    def register(self, name: str, expert_cls: Type[BaseExpert]) -> None:
        """Register one expert class."""
        if name in self._classes:
            raise KeyError(f"Expert '{name}' is already registered")
        self._classes[name] = expert_cls

    def get(self, name: str) -> Type[BaseExpert] | None:
        """Return a registered expert class, if available."""
        return self._classes.get(name)

    def names(self) -> tuple[str, ...]:
        """Return available expert names."""
        return tuple(sorted(self._classes))


registry = ExpertRegistry()


def register_default_experts() -> None:
    """Register shipped experts when their modules exist."""
    from mssr_expert.experts.stage0_gap_crossing_expert import Stage0GapCrossingExpert
    from mssr_expert.experts.stage1_obstacle_traversal_expert import (
        Stage1ObstacleTraversalExpert,
    )
    from mssr_expert.experts.stage2_stair_climb_expert import Stage2StairClimbExpert

    defaults = {
        "stage0_gap_crossing": Stage0GapCrossingExpert,
        "stage1_obstacle_traversal": Stage1ObstacleTraversalExpert,
        "stage2_stair_climb": Stage2StairClimbExpert,
    }
    for name, expert_cls in defaults.items():
        if registry.get(name) is None:
            registry.register(name, expert_cls)


register_default_experts()
