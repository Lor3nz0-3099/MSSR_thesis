"""Registry for behavior primitive classes."""
from __future__ import annotations

from typing import Type

from mssr_expert.primitives.base_primitive import BasePrimitive


class PrimitiveRegistry:
    """Map primitive names to primitive classes."""

    def __init__(self) -> None:
        self._classes: dict[str, Type[BasePrimitive]] = {}

    def register(self, name: str, primitive_cls: Type[BasePrimitive]) -> None:
        """Register one primitive class."""
        if name in self._classes:
            raise KeyError(f"Primitive '{name}' is already registered")
        self._classes[name] = primitive_cls

    def create(self, name: str) -> BasePrimitive | None:
        """Instantiate a primitive by name."""
        primitive_cls = self._classes.get(name)
        return primitive_cls() if primitive_cls is not None else None

    def names(self) -> tuple[str, ...]:
        """Return available primitive names."""
        return tuple(sorted(self._classes))


registry = PrimitiveRegistry()


def register_default_primitives() -> None:
    """Register shipped primitives when their modules exist."""
    from mssr_expert.primitives.attach_as_pivot import AttachAsPivotPrimitive
    from mssr_expert.primitives.climb_on import ClimbOnPrimitive
    from mssr_expert.primitives.dock_to_surface import DockToSurfacePrimitive
    from mssr_expert.primitives.hold_position import HoldPositionPrimitive
    from mssr_expert.primitives.roll_to import RollToPrimitive
    from mssr_expert.primitives.rotate_around_attached import (
        RotateAroundAttachedPrimitive,
    )

    defaults = {
        "roll_to": RollToPrimitive,
        "hold_position": HoldPositionPrimitive,
        "dock_to_surface": DockToSurfacePrimitive,
        "attach_as_pivot": AttachAsPivotPrimitive,
        "rotate_around_attached": RotateAroundAttachedPrimitive,
        "climb_on": ClimbOnPrimitive,
    }
    for name, primitive_cls in defaults.items():
        if registry.create(name) is None:
            registry.register(name, primitive_cls)


register_default_primitives()
