"""Small deterministic utilities shared by experts and primitives."""
from __future__ import annotations

import random
import re
from typing import Iterable, TypeVar


T = TypeVar("T")


def make_rng(seed: int | None) -> random.Random:
    """Create an isolated RNG so experts do not touch global random state."""
    return random.Random(0 if seed is None else seed)


def natural_sort_key(value: str) -> tuple[str, int | str]:
    """Sort ids like sphere_2 before sphere_10."""
    match = re.match(r"^(.*?)(\d+)$", value)
    if match is None:
        return (value, value)
    return (match.group(1), int(match.group(2)))


def sorted_ids(values: Iterable[str]) -> tuple[str, ...]:
    """Return deterministically sorted module ids."""
    return tuple(sorted(values, key=natural_sort_key))
