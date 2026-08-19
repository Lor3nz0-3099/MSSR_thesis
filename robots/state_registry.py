"""State registry for collections of modular robot modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from robots.module_state import ModuleState


@dataclass(frozen=True)
class RobotStateSnapshot:
    """Immutable snapshot of all known module states at one simulation time."""

    timestamp: float
    modules: tuple[ModuleState, ...]

    def module_by_id(self) -> dict[str, ModuleState]:
        """Return the snapshot modules indexed by module id."""
        return {module.module_id: module for module in self.modules}

    def node_attributes(self) -> dict[str, dict[str, object]]:
        """Return graph-ready node attributes without constructing a graph."""
        return {module.module_id: module.graph_attributes() for module in self.modules}


class ModuleStateRegistry:
    """Mutable store for the latest state of each simulated module."""

    def __init__(self, modules: Iterable[ModuleState] = ()) -> None:
        """Create a registry from an optional iterable of module states."""
        self._modules: dict[str, ModuleState] = {}
        for module in modules:
            self.upsert(module)

    def upsert(self, module: ModuleState) -> None:
        """Insert or replace the latest state for one module."""
        self._modules[module.module_id] = module

    def update_many(self, modules: Iterable[ModuleState]) -> None:
        """Insert or replace multiple module states."""
        for module in modules:
            self.upsert(module)

    def get(self, module_id: str) -> ModuleState | None:
        """Return a module state, or ``None`` if the module is unknown."""
        return self._modules.get(module_id)

    def require(self, module_id: str) -> ModuleState:
        """Return a module state, raising an error if the module is unknown."""
        try:
            return self._modules[module_id]
        except KeyError as exc:
            raise KeyError(f"Unknown module id: {module_id}") from exc

    def remove(self, module_id: str) -> None:
        """Remove a module from the registry if it exists."""
        self._modules.pop(module_id, None)

    def modules(self) -> tuple[ModuleState, ...]:
        """Return all latest module states sorted by module id."""
        return tuple(self._modules[module_id] for module_id in sorted(self._modules))

    def snapshot(self, timestamp: float) -> RobotStateSnapshot:
        """Freeze the current registry content into a timestamped snapshot."""
        return RobotStateSnapshot(timestamp=timestamp, modules=self.modules())
