"""Typed output returned by deterministic experts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


LocomotionCommand = Mapping[str, float]
MagneticCommand = Mapping[str, Any]


@dataclass(frozen=True)
class ExpertOutput:
    """One expert decision step, independent from ROS 2 and Isaac."""

    locomotion: Mapping[str, LocomotionCommand] = field(default_factory=dict)
    magnetic: tuple[MagneticCommand, ...] = ()
    fsm_state: str = "UNINITIALIZED"
    active_primitive: str | None = None
    primitive_params: Mapping[str, Any] = field(default_factory=dict)
    primitive_goal: Mapping[str, Any] | None = None
    module_roles: Mapping[str, str] = field(default_factory=dict)
    attachment_modes: Mapping[str, str] = field(default_factory=dict)
    task_metrics: Mapping[str, Any] = field(default_factory=dict)
    success: bool = False
    done: bool = False
    reset_requested: bool = False
    debug: Mapping[str, Any] = field(default_factory=dict)

    def to_action_payload(self, stamp: float, stage_id: int, task_type: str) -> dict[str, Any]:
        """Convert expert output into the JSON action schema used by the bridge."""
        return {
            "schema_version": "mssr.actions.v2",
            "stamp": float(stamp),
            "stage_id": int(stage_id),
            "task_type": task_type,
            "reset": bool(self.reset_requested),
            "locomotion": {
                module_id: {
                    "vx": float(command.get("vx", 0.0)),
                    "vy": float(command.get("vy", 0.0)),
                    "yaw_rate": float(command.get("yaw_rate", 0.0)),
                }
                for module_id, command in self.locomotion.items()
            },
            "magnetic": [dict(command) for command in self.magnetic],
            "expert": {
                "fsm_state": self.fsm_state,
                "active_primitive": self.active_primitive,
                "primitive_params": dict(self.primitive_params),
                "primitive_goal": (
                    dict(self.primitive_goal)
                    if self.primitive_goal is not None
                    else None
                ),
                "module_roles": dict(self.module_roles),
                "attachment_modes": dict(self.attachment_modes),
                "task_metrics": dict(self.task_metrics),
                "success": self.success,
                "done": self.done,
                "debug": dict(self.debug),
            },
        }
