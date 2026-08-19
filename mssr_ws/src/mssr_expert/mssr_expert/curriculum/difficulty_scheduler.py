"""Deterministic difficulty scheduling for curriculum stages."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DifficultyScheduler:
    """Adjust difficulty from episode outcomes without randomness."""

    initial_difficulty: float = 0.0
    min_difficulty: float = 0.0
    max_difficulty: float = 1.0
    success_step: float = 0.05
    failure_step: float = 0.025

    def __post_init__(self) -> None:
        self._difficulty = self._clamp(self.initial_difficulty)

    @property
    def difficulty(self) -> float:
        """Return the current difficulty."""
        return self._difficulty

    def update(self, success: bool) -> float:
        """Update difficulty after one episode."""
        delta = self.success_step if success else -self.failure_step
        self._difficulty = self._clamp(self._difficulty + delta)
        return self._difficulty

    def reset(self, difficulty: float | None = None) -> None:
        """Reset to the given difficulty, or the initial one."""
        self._difficulty = self._clamp(
            self.initial_difficulty if difficulty is None else difficulty
        )

    def _clamp(self, value: float) -> float:
        return max(self.min_difficulty, min(self.max_difficulty, float(value)))
