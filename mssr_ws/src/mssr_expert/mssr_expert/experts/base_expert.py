"""Base interface for deterministic swarm-like experts."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from mssr_expert.experts.expert_output import ExpertOutput


class BaseExpert(ABC):
    """Pure expert logic, independent from ROS 2 and Isaac APIs."""

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self._done = False

    def reset(self, scenario: Mapping[str, Any] | None = None) -> None:
        """Reset internal state before a new episode."""
        self._done = False

    @abstractmethod
    def step(self, observation: Mapping[str, Any], graph: Mapping[str, Any]) -> ExpertOutput:
        """Compute one deterministic action from observation and graph."""

    def is_done(self) -> bool:
        """Return whether this expert has completed its current episode."""
        return self._done
