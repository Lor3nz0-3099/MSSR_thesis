"""Base interface for reusable swarm behavior primitives."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class PrimitiveResult:
    """Action fragment produced by a primitive."""

    locomotion: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    magnetic: tuple[Mapping[str, Any], ...] = ()
    success: bool = False
    done: bool = False
    metrics: Mapping[str, Any] = field(default_factory=dict)
    debug: Mapping[str, Any] = field(default_factory=dict)


class BasePrimitive(ABC):
    """Pure primitive logic composed by experts."""

    name = "base"

    @abstractmethod
    def step(
        self,
        observation: Mapping[str, Any],
        graph: Mapping[str, Any],
        params: Mapping[str, Any],
    ) -> PrimitiveResult:
        """Compute one primitive action fragment."""
