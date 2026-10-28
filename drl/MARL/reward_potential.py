"""Discount-consistent raw shaping, with explicit absorbing terminal states.

Use the learner's discount. ``gamma=1`` preserves the undiscounted diagnostic
contract used by existing callers. A time-limit truncation is not an absorbing
terminal: pass ``terminated=False`` and bootstrap the critic from the last state.
Sparse success/failure events and costs are deliberately separate from shaping.
"""

from __future__ import annotations

import math
from typing import Mapping


def validate_shaping_discount(shaping_gamma: float) -> float:
    gamma = float(shaping_gamma)
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("shaping_gamma must be finite and within [0, 1]")
    return gamma


def potential_difference(
    previous: float,
    next_state: float,
    *,
    shaping_gamma: float,
    terminated: bool,
) -> float:
    """Return gamma Phi(next) - Phi(previous); absorbing states have Phi=0.

    Task potentials are nonnegative and bounded. Hence any ordinary closed-loop
    sum is (gamma-1)*sum(Phi) <= 0. Discounted sums telescope, including the final
    debit on true success/failure termination; resetting is never a transition.
    """
    gamma = validate_shaping_discount(shaping_gamma)
    return gamma * (0.0 if terminated else next_state) - previous


def validate_owner_xyz_positions(
    positions: Mapping[str, tuple[float, float, float]], *, name: str,
) -> tuple[str, ...]:
    """Require exactly eight named owners with finite physical XYZ vectors."""
    if not isinstance(positions, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if len(positions) != 8:
        raise ValueError(f"{name} must contain exactly 8 module owners")
    if any(not isinstance(owner, str) or not owner for owner in positions):
        raise ValueError(f"{name} must contain nonempty string owners")
    for owner, point in positions.items():
        if (not isinstance(point, (tuple, list)) or len(point) != 3
                or not all(math.isfinite(float(value)) for value in point)):
            raise ValueError(f"Invalid XYZ position for {owner}")
    return tuple(sorted(positions))
