"""Semi-Markov helpers for temporally extended C1 structural options."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping

import torch


def _scalar_tensor(value, *, label: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.numel() != 1 or not value.is_floating_point():
        raise TypeError(f"{label} must be one floating tensor scalar")
    result = value.detach().clone().reshape(())
    if not bool(torch.isfinite(result)):
        raise ValueError(f"{label} must be finite")
    return result

def actor_policy_lag_is_acceptable(
    *,
    start_version: int,
    current_version: int,
    max_lag: int = 1,
) -> bool:
    """Return whether an SMDP actor sample is fresh enough for PPO reuse."""
    for label, value in (
        ("start_version", start_version),
        ("current_version", current_version),
        ("max_lag", max_lag),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{label} must be an integer")
        if value < 0:
            raise ValueError(f"{label} must be non-negative")

    if start_version > current_version:
        raise ValueError("start_version cannot exceed current_version")

    return (current_version - start_version) <= max_lag


@dataclass
class DiscountedOptionAccumulator:
    """Accumulate primitive-step rewards into one SMDP option return."""

    discount: float
    duration_steps: int = 0
    team_reward: torch.Tensor | None = None
    owner_rewards: dict[str, torch.Tensor] = field(default_factory=dict)
    mean_field_rewards: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.discount, bool) or not isinstance(self.discount, (int, float)):
            raise TypeError("discount must be numeric")
        self.discount = float(self.discount)
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must lie in [0, 1]")

    def add(self, *, team_reward, owner_rewards: Mapping[str, torch.Tensor],
            mean_field_rewards: Mapping[str, torch.Tensor]) -> None:
        if set(owner_rewards) != set(mean_field_rewards):
            raise ValueError("owner and mean-field rewards must cover the same owners")

        team = _scalar_tensor(team_reward, label="team_reward")
        owner = {key: _scalar_tensor(value, label=f"owner_rewards[{key}]")
                 for key, value in owner_rewards.items()}
        mean_field = {key: _scalar_tensor(value, label=f"mean_field_rewards[{key}]")
                      for key, value in mean_field_rewards.items()}

        coefficient = self.discount ** self.duration_steps
        if self.team_reward is None:
            self.team_reward = torch.zeros_like(team)
            self.owner_rewards = {key: torch.zeros_like(value) for key, value in owner.items()}
            self.mean_field_rewards = {
                key: torch.zeros_like(value) for key, value in mean_field.items()
            }
        elif set(self.owner_rewards) != set(owner):
            raise ValueError("reward owner set changed while option was active")

        self.team_reward = self.team_reward + coefficient * team
        for key in owner:
            self.owner_rewards[key] = self.owner_rewards[key] + coefficient * owner[key]
            self.mean_field_rewards[key] = (
                self.mean_field_rewards[key] + coefficient * mean_field[key]
            )
        self.duration_steps += 1
