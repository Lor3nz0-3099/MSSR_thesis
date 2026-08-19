from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

from smores_ep.control.teleop import InternalMotionMode, SmoresCommand

from smores_ep.primitives.model import (
    PrimitiveGoal,
    PrimitiveStatus,
    PrimitiveStatusBatch,
)


class PrimitiveFileChannel:
    """Atomic file transport between Isaac and the external ROS 2 bridge."""

    def __init__(
        self,
        goal_file: str | Path,
        cancel_file: str | Path,
        status_file: str | Path,
        ignore_existing: bool = True,
    ) -> None:
        self._goal_file = Path(goal_file)
        self._cancel_file = Path(cancel_file)
        self._status_file = Path(status_file)
        existing_goal = self._snapshot(self._goal_file)
        existing_cancel = self._snapshot(self._cancel_file)
        self._last_goal_signature = (
            existing_goal[0]
            if ignore_existing and existing_goal is not None
            else None
        )
        self._last_cancel_signature = (
            existing_cancel[0]
            if ignore_existing and existing_cancel is not None
            else None
        )

    def poll_goal(self) -> PrimitiveGoal | None:
        snapshot = self._snapshot(self._goal_file)
        if snapshot is None:
            return None
        signature, payload = snapshot
        if signature == self._last_goal_signature:
            return None
        self._last_goal_signature = signature
        return PrimitiveGoal.from_json(payload)

    def poll_cancel(self) -> str | None:
        snapshot = self._snapshot(self._cancel_file)
        if snapshot is None:
            return None
        signature, payload = snapshot
        if signature == self._last_cancel_signature:
            return None
        self._last_cancel_signature = signature
        decoded = json.loads(payload)
        if isinstance(decoded, str):
            goal_id = decoded
        elif isinstance(decoded, dict):
            goal_id = str(decoded.get("goal_id", ""))
        else:
            raise ValueError("Primitive cancel payload must name a goal_id")
        if not goal_id:
            raise ValueError("Primitive cancel goal_id cannot be empty")
        return goal_id

    def publish(self, status: PrimitiveStatus) -> None:
        self._write_atomic(self._status_file, status.to_json())

    def publish_many(
        self,
        statuses: tuple[PrimitiveStatus, ...] | list[PrimitiveStatus],
        stamp_s: float,
    ) -> None:
        """Publish one atomic multi-goal status snapshot."""

        batch = PrimitiveStatusBatch(stamp_s, tuple(statuses))
        self._write_atomic(self._status_file, batch.to_json())

    @staticmethod
    def _read(path: Path) -> str | None:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    @staticmethod
    def _snapshot(
        path: Path,
    ) -> tuple[tuple[int, int, int], str] | None:
        """Read a file together with an identity that changes on each write."""

        try:
            stat = path.stat()
            payload = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        signature = (
            int(stat.st_mtime_ns),
            int(stat.st_size),
            int(stat.st_ino),
        )
        return signature, payload

    @staticmethod
    def _write_atomic(path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)


class ActionFileChannel:
    """Read composed locomotion commands with a dead-man timeout."""

    def __init__(
        self,
        action_file: str | Path,
        timeout_s: float = 0.5,
        ignore_existing: bool = True,
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("Action command timeout must be positive")
        self._action_file = Path(action_file)
        self._timeout_s = float(timeout_s)
        existing = PrimitiveFileChannel._snapshot(self._action_file)
        self._last_signature = (
            existing[0]
            if ignore_existing and existing is not None
            else None
        )
        self._last_received_s: float | None = None
        self._commands: dict[str, SmoresCommand] = {}

    def commands(self, now_s: float) -> dict[str, SmoresCommand]:
        """Poll once and return only commands that still satisfy the timeout."""

        if not math.isfinite(now_s):
            raise ValueError("Action polling time must be finite")
        snapshot = PrimitiveFileChannel._snapshot(self._action_file)
        if snapshot is not None:
            signature, payload = snapshot
            if signature != self._last_signature:
                self._last_signature = signature
                self._commands = self._parse(payload)
                self._last_received_s = now_s
        if self._last_received_s is None:
            return {}
        if now_s - self._last_received_s > self._timeout_s:
            return {}
        return dict(self._commands)

    @staticmethod
    def _parse(payload: str) -> dict[str, SmoresCommand]:
        decoded = json.loads(payload)
        if not isinstance(decoded, Mapping):
            raise ValueError("Action payload must be a JSON object")
        schema = decoded.get("schema_version", "mssr.actions.v2")
        if schema != "mssr.actions.v2":
            raise ValueError(f"Unsupported action schema: {schema}")
        locomotion = decoded.get("locomotion", {})
        if not isinstance(locomotion, Mapping):
            raise ValueError("Action locomotion field must be an object")
        commands: dict[str, SmoresCommand] = {}
        for raw_module_id, raw_command in locomotion.items():
            module_id = str(raw_module_id)
            if not module_id or not isinstance(raw_command, Mapping):
                raise ValueError("Every locomotion entry needs a module and command")
            lateral = float(raw_command.get("vy", 0.0))
            if not math.isfinite(lateral) or abs(lateral) > 1.0e-9:
                raise ValueError(
                    "SMORES-EP differential drive does not support lateral vy"
                )
            pan_velocity = float(
                raw_command.get(
                    "pan_rate_rad_s",
                    raw_command.get("pan_rate", 0.0),
                )
            )
            if not math.isfinite(pan_velocity):
                raise ValueError("SMORES PAN velocity must be finite")
            commands[module_id] = SmoresCommand(
                linear_x_m_s=float(raw_command.get("vx", 0.0)),
                angular_z_rad_s=float(raw_command.get("yaw_rate", 0.0)),
                internal_motion=(
                    InternalMotionMode.PAN_VELOCITY
                    if abs(pan_velocity) > 1.0e-12
                    # Locomotion must preserve the assembled shape. HOLD
                    # reuses the controller's current unwrapped PAN and TILT
                    # targets, so all selected train wheels can drive without
                    # releasing the hyper-redundant structure to gravity.
                    else InternalMotionMode.HOLD
                ),
                pan_velocity_rad_s=pan_velocity,
            )
        return commands
