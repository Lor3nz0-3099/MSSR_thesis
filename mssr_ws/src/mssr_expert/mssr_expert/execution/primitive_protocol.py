"""Transport-independent protocol used to control backend primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


GOAL_SCHEMA = "mssr.primitive_goal.v1"
STATUS_SCHEMA = "mssr.primitive_status.v1"
STATUS_BATCH_SCHEMA = "mssr.primitive_status_batch.v1"

TERMINAL_STATES = frozenset(
    {
        "succeeded",
        "failed",
        "canceled",
        "rejected",
    }
)

VALID_PRIMITIVES = frozenset(
    {
        "drive_to_pose",
        "align_faces",
        "assisted_align_faces",
        "dock",
        "undock",
        "set_pan",
        "rotate_pan_by",
        "set_tilt",
        "rotate_tilt_by",
    }
)

VALID_FACES = frozenset(
    {
        "LEFT",
        "RIGHT",
        "TOP",
        "BOTTOM",
    }
)

VALID_FACE_EXECUTION_PHASES = frozenset(
    {
        "full",
        "reach",
        "align",
        "approach",
    }
)


class PrimitiveProtocolError(ValueError):
    """Raised when a primitive goal or status is invalid."""


@dataclass(frozen=True)
class PrimitiveGoalRequest:
    """One action-like primitive goal sent to Isaac."""

    goal_id: str
    primitive: str
    module_ids: tuple[str, ...]
    parameters: Mapping[str, Any] = field(default_factory=dict)
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if not self.goal_id.strip():
            raise PrimitiveProtocolError(
                "Primitive goal_id cannot be empty."
            )

        if self.primitive not in VALID_PRIMITIVES:
            raise PrimitiveProtocolError(
                f"Unknown primitive {self.primitive!r}."
            )

        if (
            not self.module_ids
            or any(not module_id.strip() for module_id in self.module_ids)
        ):
            raise PrimitiveProtocolError(
                "Primitive goals require module IDs."
            )

        if len(self.module_ids) != len(set(self.module_ids)):
            raise PrimitiveProtocolError(
                "Primitive module IDs must be distinct."
            )

        if (
            not math.isfinite(self.timeout_s)
            or self.timeout_s <= 0.0
        ):
            raise PrimitiveProtocolError(
                "Primitive timeout must be positive and finite."
            )

        parameters = dict(self.parameters)
        self._validate_shape(parameters)
        object.__setattr__(self, "parameters", parameters)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the request for /mssr/primitives/goal."""

        return {
            "schema_version": GOAL_SCHEMA,
            "goal_id": self.goal_id,
            "primitive": self.primitive,
            "module_ids": list(self.module_ids),
            "parameters": dict(self.parameters),
            "timeout_s": float(self.timeout_s),
        }

    def _validate_shape(
        self,
        parameters: dict[str, Any],
    ) -> None:
        two_module_primitives = {
            "align_faces",
            "dock",
            "undock",
        }

        expected_module_count = (
            3
            if self.primitive == "assisted_align_faces"
            else
            2
            if self.primitive in two_module_primitives
            else 1
        )

        if len(self.module_ids) != expected_module_count:
            raise PrimitiveProtocolError(
                f"{self.primitive} requires "
                f"{expected_module_count} module ID(s)."
            )

        if self.primitive == "drive_to_pose":
            for key in ("x_m", "y_m", "yaw_rad"):
                parameters[key] = _finite_parameter(
                    parameters,
                    key,
                )

        elif self.primitive in {
            *two_module_primitives,
            "assisted_align_faces",
        }:
            face_a = str(parameters.get("face_a", "")).upper()
            face_b = str(parameters.get("face_b", "")).upper()

            if face_a not in VALID_FACES:
                raise PrimitiveProtocolError(
                    f"Invalid face_a {face_a!r}."
                )

            if face_b not in VALID_FACES:
                raise PrimitiveProtocolError(
                    f"Invalid face_b {face_b!r}."
                )

            parameters["face_a"] = face_a
            parameters["face_b"] = face_b

            if "clocking_quarter_turns" in parameters:
                clocking = int(
                    parameters["clocking_quarter_turns"]
                )

                if clocking not in {0, 1, 2, 3}:
                    raise PrimitiveProtocolError(
                        "clocking_quarter_turns must be 0, 1, 2 or 3."
                    )

                parameters["clocking_quarter_turns"] = clocking

            if "top_bottom_contact_tolerance_m" in parameters:
                tolerance = _finite_parameter(
                    parameters,
                    "top_bottom_contact_tolerance_m",
                )
                if tolerance <= 0.0:
                    raise PrimitiveProtocolError(
                        "top_bottom_contact_tolerance_m must be positive."
                    )
            if "contact_approach_feedback" in parameters:
                feedback = parameters["contact_approach_feedback"]
                if not isinstance(feedback, bool):
                    raise PrimitiveProtocolError(
                        "contact_approach_feedback must be boolean."
                    )
            if "execution_phase" in parameters:
                execution_phase = str(
                    parameters["execution_phase"]
                ).lower()
                if execution_phase not in VALID_FACE_EXECUTION_PHASES:
                    raise PrimitiveProtocolError(
                        "execution_phase must be full, reach, align or "
                        "approach."
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
                    raise PrimitiveProtocolError(
                        "staging_path_fallback_level must be 0, 1 or 2."
                    )
            if "snap_to_nominal" in parameters:
                snap = parameters["snap_to_nominal"]
                if not isinstance(snap, bool):
                    raise PrimitiveProtocolError(
                        "snap_to_nominal must be boolean."
                    )

        elif self.primitive in {"set_pan", "set_tilt"}:
            parameters["angle_rad"] = _finite_parameter(
                parameters,
                "angle_rad",
            )
            if "tolerance_rad" in parameters:
                tolerance = _finite_parameter(
                    parameters,
                    "tolerance_rad",
                )
                if tolerance <= 0.0:
                    raise PrimitiveProtocolError(
                        "tolerance_rad must be positive."
                    )
            if "max_servo_error_rad" in parameters:
                servo_error = _finite_parameter(
                    parameters, "max_servo_error_rad"
                )
                if servo_error <= 0.0:
                    raise PrimitiveProtocolError(
                        "max_servo_error_rad must be positive."
                    )
            if "max_servo_speed_rad_s" in parameters:
                servo_speed = _finite_parameter(
                    parameters, "max_servo_speed_rad_s"
                )
                if servo_speed <= 0.0:
                    raise PrimitiveProtocolError(
                        "max_servo_speed_rad_s must be positive."
                    )
            if "structural_hold_module_ids" in parameters:
                raw_ids = parameters["structural_hold_module_ids"]
                if not isinstance(raw_ids, list | tuple):
                    raise PrimitiveProtocolError(
                        "structural_hold_module_ids must be an array."
                    )
                hold_ids = tuple(str(item).strip() for item in raw_ids)
                if (
                    not hold_ids
                    or any(not item for item in hold_ids)
                    or len(set(hold_ids)) != len(hold_ids)
                ):
                    raise PrimitiveProtocolError(
                        "structural_hold_module_ids must contain distinct, "
                        "non-empty module IDs."
                    )
                parameters["structural_hold_module_ids"] = list(hold_ids)
            if "passive_module_ids" in parameters:
                raw_ids = parameters["passive_module_ids"]
                if not isinstance(raw_ids, list | tuple):
                    raise PrimitiveProtocolError(
                        "passive_module_ids must be an array."
                    )
                passive_ids = tuple(str(item).strip() for item in raw_ids)
                if (
                    any(not item for item in passive_ids)
                    or len(set(passive_ids)) != len(passive_ids)
                ):
                    raise PrimitiveProtocolError(
                        "passive_module_ids must contain distinct, non-empty "
                        "module IDs."
                    )
                hold_set = set(parameters.get("structural_hold_module_ids", ()))
                overlap = hold_set.intersection(passive_ids)
                if overlap:
                    raise PrimitiveProtocolError(
                        "passive_module_ids and structural_hold_module_ids "
                        "must be disjoint."
                    )
                parameters["passive_module_ids"] = list(passive_ids)
            has_pusher_id = "pusher_module_id" in parameters
            has_pusher_speed = "pusher_linear_m_s" in parameters
            if has_pusher_id != has_pusher_speed:
                raise PrimitiveProtocolError(
                    "pusher_module_id and pusher_linear_m_s must be supplied "
                    "together."
                )
            if has_pusher_id:
                if self.primitive != "set_tilt":
                    raise PrimitiveProtocolError(
                        "Fold pusher parameters are valid only for set_tilt."
                    )
                pusher = str(parameters["pusher_module_id"]).strip()
                if not pusher or pusher in self.module_ids:
                    raise PrimitiveProtocolError(
                        "pusher_module_id must differ from the TILT module."
                    )
                speed = _finite_parameter(
                    parameters, "pusher_linear_m_s"
                )
                if abs(speed) <= 1.0e-9:
                    raise PrimitiveProtocolError(
                        "pusher_linear_m_s must be non-zero."
                    )
                parameters["pusher_module_id"] = pusher
            if "hold_after_group_module_ids" in parameters:
                if self.primitive != "set_tilt":
                    raise PrimitiveProtocolError(
                        "hold_after_group_module_ids is valid only for "
                        "set_tilt."
                    )
                raw_ids = parameters["hold_after_group_module_ids"]
                if not isinstance(raw_ids, list | tuple):
                    raise PrimitiveProtocolError(
                        "hold_after_group_module_ids must be an array."
                    )
                hold_ids = tuple(str(item).strip() for item in raw_ids)
                if (
                    not hold_ids
                    or any(not item for item in hold_ids)
                    or len(set(hold_ids)) != len(hold_ids)
                ):
                    raise PrimitiveProtocolError(
                        "hold_after_group_module_ids must contain distinct, "
                        "non-empty module IDs."
                    )
                parameters["hold_after_group_module_ids"] = list(hold_ids)
            if "stabilize_during_group_module_ids" in parameters:
                if self.primitive != "set_tilt":
                    raise PrimitiveProtocolError(
                        "stabilize_during_group_module_ids is valid only for "
                        "set_tilt."
                    )
                raw_ids = parameters[
                    "stabilize_during_group_module_ids"
                ]
                if not isinstance(raw_ids, list | tuple):
                    raise PrimitiveProtocolError(
                        "stabilize_during_group_module_ids must be an array."
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
                    raise PrimitiveProtocolError(
                        "stabilize_during_group_module_ids must contain "
                        "distinct non-actuated module IDs."
                    )
                parameters["stabilize_during_group_module_ids"] = list(
                    stabilize_ids
                )

        elif self.primitive in {
            "rotate_pan_by",
            "rotate_tilt_by",
        }:
            parameters["delta_rad"] = _finite_parameter(
                parameters,
                "delta_rad",
            )


@dataclass(frozen=True)
class PrimitiveStatusView:
    """Normalized status received from Isaac."""

    goal_id: str
    primitive: str
    state: str
    module_ids: tuple[str, ...]
    phase: str = ""
    progress: float = 0.0
    code: str = ""
    message: str = ""
    feedback: Mapping[str, Any] = field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        """Return whether the primitive has finished."""

        return self.state in TERMINAL_STATES

    @property
    def succeeded(self) -> bool:
        """Return whether the primitive completed successfully."""

        return self.state == "succeeded"

    @property
    def failed(self) -> bool:
        """Return whether the primitive ended unsuccessfully."""

        return self.state in {
            "failed",
            "canceled",
            "rejected",
        }


def parse_primitive_statuses(
    payload: Mapping[str, Any] | None,
) -> dict[str, PrimitiveStatusView]:
    """Normalize a single status or a concurrent status batch."""

    if not payload:
        return {}

    raw_statuses: Sequence[Any]

    if payload.get("schema_version") == STATUS_BATCH_SCHEMA:
        raw_statuses = payload.get("statuses", [])

    elif isinstance(payload.get("statuses"), list | tuple):
        raw_statuses = payload["statuses"]

    else:
        raw_statuses = (payload,)

    statuses: dict[str, PrimitiveStatusView] = {}

    for raw_status in raw_statuses:
        if not isinstance(raw_status, Mapping):
            continue

        status = _parse_single_status(raw_status)
        statuses[status.goal_id] = status

    return statuses


def make_drive_to_pose_goal(
    goal_id: str,
    module_id: str,
    x_m: float,
    y_m: float,
    yaw_rad: float,
    timeout_s: float = 30.0,
) -> PrimitiveGoalRequest:
    """Create a world-frame differential-drive goal."""

    return PrimitiveGoalRequest(
        goal_id=goal_id,
        primitive="drive_to_pose",
        module_ids=(module_id,),
        parameters={
            "x_m": x_m,
            "y_m": y_m,
            "yaw_rad": yaw_rad,
        },
        timeout_s=timeout_s,
    )


def make_align_faces_goal(
    goal_id: str,
    mobile_module_id: str,
    mobile_face: str,
    parent_module_id: str,
    parent_face: str,
    clocking_quarter_turns: int,
    timeout_s: float = 30.0,
    contact_quality_planar_tolerance_m: float | None = None,
    contact_quality_retry_count: int = 0,
    top_bottom_contact_tolerance_m: float | None = None,
    contact_approach_feedback: bool = False,
    execution_phase: str | None = None,
    staging_path_fallback_level: int | None = None,
) -> PrimitiveGoalRequest:
    """Create an explicit face-alignment goal."""

    parameters: dict[str, Any] = {
        "face_a": mobile_face,
        "face_b": parent_face,
        "clocking_quarter_turns": clocking_quarter_turns,
    }
    if contact_quality_planar_tolerance_m is not None:
        parameters.update(
            {
                "contact_quality_planar_tolerance_m": (
                    contact_quality_planar_tolerance_m
                ),
                "contact_quality_retry_count": contact_quality_retry_count,
            }
        )
    if top_bottom_contact_tolerance_m is not None:
        parameters["top_bottom_contact_tolerance_m"] = (
            top_bottom_contact_tolerance_m
        )
    if contact_approach_feedback:
        parameters["contact_approach_feedback"] = True
    if execution_phase is not None:
        parameters["execution_phase"] = execution_phase
    if staging_path_fallback_level is not None:
        parameters["staging_path_fallback_level"] = (
            staging_path_fallback_level
        )

    return PrimitiveGoalRequest(
        goal_id=goal_id,
        primitive="align_faces",
        module_ids=(
            mobile_module_id,
            parent_module_id,
        ),
        parameters=parameters,
        timeout_s=timeout_s,
    )


def make_assisted_align_faces_goal(
    goal_id: str,
    payload_module_id: str,
    payload_face: str,
    parent_module_id: str,
    parent_face: str,
    helper_module_id: str,
    clocking_quarter_turns: int,
    timeout_s: float = 30.0,
    top_bottom_contact_tolerance_m: float | None = None,
    contact_approach_feedback: bool = False,
) -> PrimitiveGoalRequest:
    """Align a payload face while a rigidly attached helper drives it."""

    parameters: dict[str, Any] = {
        "face_a": payload_face,
        "face_b": parent_face,
        "clocking_quarter_turns": clocking_quarter_turns,
    }
    if top_bottom_contact_tolerance_m is not None:
        parameters["top_bottom_contact_tolerance_m"] = (
            top_bottom_contact_tolerance_m
        )
    if contact_approach_feedback:
        parameters["contact_approach_feedback"] = True

    return PrimitiveGoalRequest(
        goal_id=goal_id,
        primitive="assisted_align_faces",
        module_ids=(
            payload_module_id,
            parent_module_id,
            helper_module_id,
        ),
        parameters=parameters,
        timeout_s=timeout_s,
    )


def make_dock_goal(
    goal_id: str,
    mobile_module_id: str,
    mobile_face: str,
    parent_module_id: str,
    parent_face: str,
    clocking_quarter_turns: int,
    timeout_s: float = 10.0,
    top_bottom_contact_tolerance_m: float | None = None,
    contact_approach_feedback: bool = False,
    snap_to_nominal: bool = False,
) -> PrimitiveGoalRequest:
    """Create an explicit rigid docking goal."""

    parameters: dict[str, Any] = {
        "face_a": mobile_face,
        "face_b": parent_face,
        "clocking_quarter_turns": clocking_quarter_turns,
    }
    if top_bottom_contact_tolerance_m is not None:
        parameters["top_bottom_contact_tolerance_m"] = (
            top_bottom_contact_tolerance_m
        )
    if contact_approach_feedback:
        parameters["contact_approach_feedback"] = True
    if snap_to_nominal:
        parameters["snap_to_nominal"] = True

    return PrimitiveGoalRequest(
        goal_id=goal_id,
        primitive="dock",
        module_ids=(
            mobile_module_id,
            parent_module_id,
        ),
        parameters=parameters,
        timeout_s=timeout_s,
    )


def make_undock_goal(
    goal_id: str,
    first_module_id: str,
    first_face: str,
    second_module_id: str,
    second_face: str,
    timeout_s: float = 10.0,
) -> PrimitiveGoalRequest:
    """Create an explicit rigid undocking goal."""

    return PrimitiveGoalRequest(
        goal_id=goal_id,
        primitive="undock",
        module_ids=(first_module_id, second_module_id),
        parameters={
            "face_a": first_face,
            "face_b": second_face,
        },
        timeout_s=timeout_s,
    )


def _parse_single_status(
    payload: Mapping[str, Any],
) -> PrimitiveStatusView:
    """Parse one status object."""

    goal_id = str(payload.get("goal_id", "")).strip()
    primitive = str(payload.get("primitive", "")).strip()
    state = str(payload.get("state", "")).lower().strip()

    if not goal_id:
        raise PrimitiveProtocolError(
            "Primitive status has no goal_id."
        )

    if not primitive:
        raise PrimitiveProtocolError(
            f"Status {goal_id!r} has no primitive name."
        )

    valid_states = {
        "accepted",
        "running",
        *TERMINAL_STATES,
    }

    if state not in valid_states:
        raise PrimitiveProtocolError(
            f"Status {goal_id!r} has invalid state {state!r}."
        )

    raw_module_ids = payload.get("module_ids", ())

    if not isinstance(raw_module_ids, list | tuple):
        raise PrimitiveProtocolError(
            "Primitive status module_ids must be an array."
        )

    progress = float(payload.get("progress", 0.0))

    if not math.isfinite(progress):
        raise PrimitiveProtocolError(
            "Primitive progress must be finite."
        )

    feedback = payload.get("feedback", {})

    if not isinstance(feedback, Mapping):
        feedback = {}

    return PrimitiveStatusView(
        goal_id=goal_id,
        primitive=primitive,
        state=state,
        module_ids=tuple(
            str(module_id)
            for module_id in raw_module_ids
        ),
        phase=str(payload.get("phase", "")),
        progress=max(0.0, min(1.0, progress)),
        code=str(payload.get("code", "")),
        message=str(payload.get("message", "")),
        feedback=dict(feedback),
    )


def _finite_parameter(
    parameters: Mapping[str, Any],
    key: str,
) -> float:
    """Read and validate one required numeric parameter."""

    if key not in parameters:
        raise PrimitiveProtocolError(
            f"Primitive parameter {key!r} is required."
        )

    value = float(parameters[key])

    if not math.isfinite(value):
        raise PrimitiveProtocolError(
            f"Primitive parameter {key!r} must be finite."
        )

    return value
