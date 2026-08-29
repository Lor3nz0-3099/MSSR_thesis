from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
from typing import Any, Mapping

from smores_ep.docking.model import normalize_face_name


GOAL_SCHEMA = "mssr.primitive_goal.v1"
STATUS_SCHEMA = "mssr.primitive_status.v1"
STATUS_BATCH_SCHEMA = "mssr.primitive_status_batch.v1"

VALID_FACE_EXECUTION_PHASES = frozenset(
    {
        "full",
        "reach",
        "align",
        "approach",
    }
)


class PrimitiveName(str, Enum):
    DRIVE_TO_POSE = "drive_to_pose"
    ALIGN_FACES = "align_faces"
    ASSISTED_ALIGN_FACES = "assisted_align_faces"
    DOCK = "dock"
    UNDOCK = "undock"
    SET_PAN = "set_pan"
    ROTATE_PAN_BY = "rotate_pan_by"
    SET_TILT = "set_tilt"
    ROTATE_TILT_BY = "rotate_tilt_by"


class PrimitiveState(str, Enum):
    ACCEPTED = "accepted"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"
    REJECTED = "rejected"

    @property
    def terminal(self) -> bool:
        return self in {
            PrimitiveState.SUCCEEDED,
            PrimitiveState.FAILED,
            PrimitiveState.CANCELED,
            PrimitiveState.REJECTED,
        }


@dataclass(frozen=True)
class PrimitiveGoal:
    """Transport-independent request for one behavioral primitive."""

    goal_id: str
    primitive: PrimitiveName
    module_ids: tuple[str, ...]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if not self.goal_id.strip():
            raise ValueError("Primitive goal_id cannot be empty")
        if not self.module_ids or any(not item.strip() for item in self.module_ids):
            raise ValueError("Primitive goals require non-empty module IDs")
        if len(set(self.module_ids)) != len(self.module_ids):
            raise ValueError("Primitive module IDs must be distinct")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0.0:
            raise ValueError("Primitive timeout must be finite and positive")
        object.__setattr__(self, "parameters", dict(self.parameters))
        self._validate_shape()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PrimitiveGoal":
        schema = payload.get("schema_version", GOAL_SCHEMA)
        if schema != GOAL_SCHEMA:
            raise ValueError(f"Unsupported primitive goal schema: {schema}")
        primitive = PrimitiveName(str(payload["primitive"]))
        raw_module_ids = payload.get("module_ids")
        if not isinstance(raw_module_ids, list | tuple):
            raise ValueError("Primitive module_ids must be an array")
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError("Primitive parameters must be an object")
        return cls(
            goal_id=str(payload["goal_id"]),
            primitive=primitive,
            module_ids=tuple(str(item) for item in raw_module_ids),
            parameters=dict(parameters),
            timeout_s=float(payload.get("timeout_s", 30.0)),
        )

    @classmethod
    def from_json(cls, payload: str) -> "PrimitiveGoal":
        decoded = json.loads(payload)
        if not isinstance(decoded, Mapping):
            raise ValueError("Primitive goal payload must be a JSON object")
        return cls.from_dict(decoded)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": GOAL_SCHEMA,
            "goal_id": self.goal_id,
            "primitive": self.primitive.value,
            "module_ids": list(self.module_ids),
            "parameters": dict(self.parameters),
            "timeout_s": self.timeout_s,
        }

    def _validate_shape(self) -> None:
        expected_modules = (
            3
            if self.primitive is PrimitiveName.ASSISTED_ALIGN_FACES
            else
            2
            if self.primitive
            in {
                PrimitiveName.ALIGN_FACES,
                PrimitiveName.DOCK,
                PrimitiveName.UNDOCK,
            }
            else 1
        )
        if len(self.module_ids) != expected_modules:
            raise ValueError(
                f"{self.primitive.value} requires {expected_modules} module ID(s)"
            )

        parameters = dict(self.parameters)
        if self.primitive is PrimitiveName.DRIVE_TO_POSE:
            for key in ("x_m", "y_m", "yaw_rad"):
                self._finite_parameter(parameters, key)
        elif self.primitive in {
            PrimitiveName.ALIGN_FACES,
            PrimitiveName.ASSISTED_ALIGN_FACES,
            PrimitiveName.DOCK,
            PrimitiveName.UNDOCK,
        }:
            parameters["face_a"] = normalize_face_name(
                str(parameters["face_a"])
            )
            parameters["face_b"] = normalize_face_name(
                str(parameters["face_b"])
            )
            if "clocking_quarter_turns" in parameters:
                clocking = int(parameters["clocking_quarter_turns"])
                if clocking not in {0, 1, 2, 3}:
                    raise ValueError(
                        "clocking_quarter_turns must be 0, 1, 2 or 3"
                    )
                parameters["clocking_quarter_turns"] = clocking
            if "top_bottom_contact_tolerance_m" in parameters:
                tolerance = self._finite_parameter(
                    parameters,
                    "top_bottom_contact_tolerance_m",
                )
                if tolerance <= 0.0:
                    raise ValueError(
                        "top_bottom_contact_tolerance_m must be positive"
                    )
            if "contact_approach_feedback" in parameters:
                feedback = parameters["contact_approach_feedback"]
                if not isinstance(feedback, bool):
                    raise ValueError(
                        "contact_approach_feedback must be boolean"
                    )
            if "execution_phase" in parameters:
                execution_phase = str(
                    parameters["execution_phase"]
                ).lower()
                if execution_phase not in VALID_FACE_EXECUTION_PHASES:
                    raise ValueError(
                        "execution_phase must be full, reach, align or "
                        "approach"
                    )
                parameters["execution_phase"] = execution_phase
            if "staging_path_fallback_level" in parameters:
                fallback_level = parameters[
                    "staging_path_fallback_level"
                ]
                if (
                    isinstance(fallback_level, bool)
                    or not isinstance(fallback_level, int)
                    or fallback_level not in {0, 1, 2}
                ):
                    raise ValueError(
                        "staging_path_fallback_level must be 0, 1 or 2"
                    )
            if "snap_to_nominal" in parameters:
                snap = parameters["snap_to_nominal"]
                if not isinstance(snap, bool):
                    raise ValueError("snap_to_nominal must be boolean")
            if self.primitive is PrimitiveName.ALIGN_FACES:
                quality_tolerance = parameters.get(
                    "contact_quality_planar_tolerance_m"
                )
                if quality_tolerance is not None:
                    quality_tolerance = float(quality_tolerance)
                    if (
                        not math.isfinite(quality_tolerance)
                        or quality_tolerance <= 0.0
                    ):
                        raise ValueError(
                            "contact_quality_planar_tolerance_m must be "
                            "finite and positive"
                        )
                    retry_count = int(
                        parameters.get("contact_quality_retry_count", 0)
                    )
                    if retry_count < 0:
                        raise ValueError(
                            "contact_quality_retry_count must be non-negative"
                        )
                    parameters[
                        "contact_quality_planar_tolerance_m"
                    ] = quality_tolerance
                    parameters["contact_quality_retry_count"] = retry_count
        elif self.primitive in {
            PrimitiveName.SET_PAN,
            PrimitiveName.SET_TILT,
        }:
            self._finite_parameter(parameters, "angle_rad")
            if "tolerance_rad" in parameters:
                tolerance = self._finite_parameter(
                    parameters,
                    "tolerance_rad",
                )
                if tolerance <= 0.0:
                    raise ValueError(
                        "Primitive parameter 'tolerance_rad' must be positive"
                    )
            if "max_servo_error_rad" in parameters:
                servo_error = self._finite_parameter(
                    parameters,
                    "max_servo_error_rad",
                )
                if servo_error <= 0.0:
                    raise ValueError(
                        "Primitive parameter 'max_servo_error_rad' must be "
                        "positive"
                    )
            if "max_servo_speed_rad_s" in parameters:
                servo_speed = self._finite_parameter(
                    parameters,
                    "max_servo_speed_rad_s",
                )
                if servo_speed <= 0.0:
                    raise ValueError(
                        "Primitive parameter 'max_servo_speed_rad_s' must be "
                        "positive"
                    )
            if "structural_hold_module_ids" in parameters:
                raw_ids = parameters["structural_hold_module_ids"]
                if not isinstance(raw_ids, list | tuple):
                    raise ValueError(
                        "structural_hold_module_ids must be an array"
                    )
                hold_ids = tuple(str(item).strip() for item in raw_ids)
                if (
                    not hold_ids
                    or any(not item for item in hold_ids)
                    or len(set(hold_ids)) != len(hold_ids)
                ):
                    raise ValueError(
                        "structural_hold_module_ids must contain distinct, "
                        "non-empty module IDs"
                    )
                parameters["structural_hold_module_ids"] = list(hold_ids)
            if "passive_module_ids" in parameters:
                raw_ids = parameters["passive_module_ids"]
                if not isinstance(raw_ids, list | tuple):
                    raise ValueError(
                        "passive_module_ids must be an array"
                    )
                passive_ids = tuple(str(item).strip() for item in raw_ids)
                if (
                    any(not item for item in passive_ids)
                    or len(set(passive_ids)) != len(passive_ids)
                ):
                    raise ValueError(
                        "passive_module_ids must contain distinct, non-empty "
                        "module IDs"
                    )
                hold_set = set(parameters.get("structural_hold_module_ids", ()))
                overlap = hold_set.intersection(passive_ids)
                if overlap:
                    raise ValueError(
                        "passive_module_ids and structural_hold_module_ids "
                        "must be disjoint"
                    )
                parameters["passive_module_ids"] = list(passive_ids)
            if "coordination_group" in parameters:
                group = str(parameters["coordination_group"]).strip()
                if not group:
                    raise ValueError(
                        "Primitive parameter 'coordination_group' cannot be empty"
                    )
                size = int(parameters.get("coordination_size", 1))
                if size < 1:
                    raise ValueError(
                        "Primitive parameter 'coordination_size' must be positive"
                    )
                parameters["coordination_group"] = group
                parameters["coordination_size"] = size
            if "hold_after_group_module_ids" in parameters:
                if self.primitive is not PrimitiveName.SET_TILT:
                    raise ValueError(
                        "hold_after_group_module_ids is valid only for "
                        "set_tilt"
                    )
                raw_ids = parameters["hold_after_group_module_ids"]
                if not isinstance(raw_ids, list | tuple):
                    raise ValueError(
                        "hold_after_group_module_ids must be an array"
                    )
                hold_ids = tuple(str(item).strip() for item in raw_ids)
                if (
                    not hold_ids
                    or any(not item for item in hold_ids)
                    or len(set(hold_ids)) != len(hold_ids)
                ):
                    raise ValueError(
                        "hold_after_group_module_ids must contain distinct, "
                        "non-empty module IDs"
                    )
                parameters["hold_after_group_module_ids"] = list(hold_ids)
            if "stabilize_during_group_module_ids" in parameters:
                if self.primitive is not PrimitiveName.SET_TILT:
                    raise ValueError(
                        "stabilize_during_group_module_ids is valid only for "
                        "set_tilt"
                    )
                raw_ids = parameters[
                    "stabilize_during_group_module_ids"
                ]
                if not isinstance(raw_ids, list | tuple):
                    raise ValueError(
                        "stabilize_during_group_module_ids must be an array"
                    )
                stabilize_ids = tuple(
                    str(item).strip() for item in raw_ids
                )
                if (
                    not stabilize_ids
                    or any(not item for item in stabilize_ids)
                    or len(set(stabilize_ids)) != len(stabilize_ids)
                    or set(stabilize_ids) & set(self.module_ids)
                ):
                    raise ValueError(
                        "stabilize_during_group_module_ids must contain "
                        "distinct non-actuated module IDs"
                    )
                parameters["stabilize_during_group_module_ids"] = list(
                    stabilize_ids
                )
            has_pusher_id = "pusher_module_id" in parameters
            has_pusher_speed = "pusher_linear_m_s" in parameters
            if has_pusher_id != has_pusher_speed:
                raise ValueError(
                    "pusher_module_id and pusher_linear_m_s must be supplied "
                    "together"
                )
            if has_pusher_id:
                if self.primitive is not PrimitiveName.SET_TILT:
                    raise ValueError(
                        "fold pusher parameters are valid only for set_tilt"
                    )
                pusher = str(parameters["pusher_module_id"]).strip()
                if not pusher or pusher in self.module_ids:
                    raise ValueError(
                        "pusher_module_id must be non-empty and differ from "
                        "the TILT module"
                    )
                speed = self._finite_parameter(
                    parameters, "pusher_linear_m_s"
                )
                if abs(speed) <= 1.0e-9:
                    raise ValueError(
                        "pusher_linear_m_s must be non-zero"
                    )
                parameters["pusher_module_id"] = pusher
        elif self.primitive in {
            PrimitiveName.ROTATE_PAN_BY,
            PrimitiveName.ROTATE_TILT_BY,
        }:
            self._finite_parameter(parameters, "delta_rad")
        object.__setattr__(self, "parameters", parameters)

    @staticmethod
    def _finite_parameter(parameters: dict[str, Any], key: str) -> float:
        if key not in parameters:
            raise ValueError(f"Primitive parameter '{key}' is required")
        value = float(parameters[key])
        if not math.isfinite(value):
            raise ValueError(f"Primitive parameter '{key}' must be finite")
        parameters[key] = value
        return value


@dataclass(frozen=True)
class PrimitiveStatus:
    """Action-like lifecycle status and feedback for one goal."""

    goal_id: str
    primitive: PrimitiveName
    state: PrimitiveState
    stamp_s: float
    module_ids: tuple[str, ...]
    phase: str = ""
    progress: float = 0.0
    code: str = ""
    message: str = ""
    feedback: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.stamp_s):
            raise ValueError("Primitive status timestamp must be finite")
        progress = max(0.0, min(1.0, float(self.progress)))
        object.__setattr__(self, "progress", progress)
        object.__setattr__(self, "feedback", dict(self.feedback))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STATUS_SCHEMA,
            "goal_id": self.goal_id,
            "primitive": self.primitive.value,
            "state": self.state.value,
            "stamp_s": self.stamp_s,
            "module_ids": list(self.module_ids),
            "phase": self.phase,
            "progress": self.progress,
            "code": self.code,
            "message": self.message,
            "feedback": dict(self.feedback),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass(frozen=True)
class PrimitiveStatusBatch:
    """Snapshot of all status updates emitted by one executor tick."""

    stamp_s: float
    statuses: tuple[PrimitiveStatus, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.stamp_s):
            raise ValueError("Primitive status batch timestamp must be finite")
        object.__setattr__(self, "statuses", tuple(self.statuses))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STATUS_BATCH_SCHEMA,
            "stamp_s": self.stamp_s,
            "statuses": [status.to_dict() for status in self.statuses],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
