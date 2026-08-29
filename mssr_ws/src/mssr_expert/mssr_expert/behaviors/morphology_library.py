"""Resolve target roles into operational SMORES-EP commands."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class MorphologyLibraryError(ValueError):
    """Raised when a morphology profile or assignment is invalid."""


@dataclass(frozen=True)
class AssignedModule:
    module_id: str
    target_vertex_id: str
    target_role: str


@dataclass(frozen=True)
class JointTarget:
    module_id: str
    joint: str
    angle_rad: float
    target_vertex_id: str
    target_role: str
    tolerance_rad: float | None = None
    coordination_group: str | None = None
    pusher_module_id: str | None = None
    pusher_linear_m_s: float | None = None
    max_servo_error_rad: float | None = None
    max_servo_speed_rad_s: float | None = None
    structural_hold_module_ids: tuple[str, ...] = ()
    angle_reference: str = "absolute"


@dataclass(frozen=True)
class LongitudinalPositionGoal:
    """World-X stop condition for a connected-morphology drive phase."""

    module_id: str
    target_x_m: float
    tolerance_m: float


@dataclass(frozen=True)
class LongitudinalDisplacementGoal:
    """World-X centroid displacement measured from a drive-phase start."""

    module_ids: tuple[str, ...]
    distance_m: float
    tolerance_m: float


@dataclass(frozen=True)
class BehaviorProgramStep:
    """One posture, timed drive, or position-goal drive program phase."""

    phase: str
    posture_targets: tuple[JointTarget, ...] = ()
    duration_s: float | None = None
    linear_m_s: float = 0.0
    yaw_rate_rad_s: float = 0.0
    lateral_m_s: float = 0.0
    active_target_roles: tuple[str, ...] = ()
    position_goal: LongitudinalPositionGoal | None = None
    displacement_goal: LongitudinalDisplacementGoal | None = None
    continuous_with_next: bool = False
    hold_locomotion_until_admitted: bool = True
    posture_reached_linear_m_s: float | None = None

    @property
    def kind(self) -> str:
        has_stop_condition = (
            self.duration_s is not None
            or self.position_goal is not None
            or self.displacement_goal is not None
        )
        if self.posture_targets and has_stop_condition:
            return "posture_drive"
        return (
            "drive"
            if has_stop_condition
            else "posture"
        )


class MorphologyLibrary:
    """Validated JSON-backed behavior definitions indexed by morphology."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        if payload.get("schema_version") != "mssr.morphology_library.v1":
            raise MorphologyLibraryError("Unsupported morphology library schema")
        raw = payload.get("morphologies")
        if not isinstance(raw, Mapping) or not raw:
            raise MorphologyLibraryError("Morphology library is empty")
        self._profiles = {str(name): dict(profile) for name, profile in raw.items()}
        for name, profile in self._profiles.items():
            self._validate_profile(name, profile)

    @classmethod
    def load(cls, path: Path) -> "MorphologyLibrary":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise MorphologyLibraryError(f"Library does not exist: {path}") from error
        except json.JSONDecodeError as error:
            raise MorphologyLibraryError(f"Invalid library JSON: {error}") from error
        if not isinstance(payload, Mapping):
            raise MorphologyLibraryError("Morphology library root must be an object")
        return cls(payload)

    @property
    def morphology_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._profiles))

    def uses_captured_neutral(self, morphology_name: str) -> bool:
        """Return whether any posture restores an observed neutral angle."""

        profile = self._profile(morphology_name)
        postures = profile.get("postures", {})
        if not isinstance(postures, Mapping):
            return False
        return any(
            isinstance(target, Mapping)
            and target.get("angle_reference") == "captured_neutral"
            for targets in postures.values()
            if isinstance(targets, list | tuple)
            for target in targets
        )

    def validate_assignment(
        self,
        morphology_name: str,
        assignments: Sequence[AssignedModule],
    ) -> None:
        profile = self._profile(morphology_name)
        expected = int(profile["module_count"])
        if len(assignments) != expected:
            raise MorphologyLibraryError(
                f"{morphology_name} needs {expected} assigned modules, got {len(assignments)}"
            )
        module_ids = [item.module_id for item in assignments]
        vertices = [item.target_vertex_id for item in assignments]
        if len(set(module_ids)) != expected or len(set(vertices)) != expected:
            raise MorphologyLibraryError("Morphology assignment must be one-to-one")


    def navigation_frame_spec(
        self,
        morphology_name: str,
    ) -> dict[str, tuple[str, ...]] | None:
        """Return the optional role-anchored planar navigation frame."""

        profile = self._profile(morphology_name)
        raw = profile.get("navigation")
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise MorphologyLibraryError(
                f"{morphology_name} navigation profile must be an object"
            )
        result: dict[str, tuple[str, ...]] = {}
        for key in (
            "center_roles",
            "forward_from_roles",
            "forward_to_roles",
        ):
            values = raw.get(key)
            if not isinstance(values, list | tuple) or not values:
                raise MorphologyLibraryError(
                    f"{morphology_name} navigation {key} must be a non-empty array"
                )
            roles = tuple(str(value).strip() for value in values)
            if any(not role for role in roles) or len(set(roles)) != len(roles):
                raise MorphologyLibraryError(
                    f"{morphology_name} navigation {key} contains invalid roles"
                )
            result[key] = roles
        return result

    def behavior_joint_targets(
        self,
        morphology_name: str,
        behavior: str,
        assignments: Sequence[AssignedModule],
    ) -> tuple[JointTarget, ...]:
        self.validate_assignment(morphology_name, assignments)
        profile = self._profile(morphology_name)
        if behavior == "prepare":
            posture_names = (str(profile["ready_posture"]),)
        else:
            behaviors = profile.get("behaviors", {})
            if not isinstance(behaviors, Mapping) or behavior not in behaviors:
                raise MorphologyLibraryError(
                    f"Behavior {behavior!r} is not defined for {morphology_name}"
                )
            raw_names = behaviors[behavior]
            if not isinstance(raw_names, list | tuple):
                raise MorphologyLibraryError("Behavior posture sequence must be an array")
            posture_names = tuple(str(item) for item in raw_names)
        targets: list[JointTarget] = []
        for posture_name in posture_names:
            targets.extend(
                self._resolve_posture(profile, posture_name, assignments)
            )
        return tuple(targets)

    def composite_behavior_steps(
        self,
        morphology_name: str,
        behavior: str,
        assignments: Sequence[AssignedModule],
        parameters: Mapping[str, Any] | None = None,
    ) -> tuple[BehaviorProgramStep, ...]:
        """Resolve a data-driven posture/drive behavior program.

        Legacy behaviors remain arrays of posture-name strings. A composite
        behavior contains objects with exactly one ``posture`` or ``drive``
        entry, allowing locomotion to be interleaved with coordinated joint
        targets without embedding morphology names in the executor.
        """

        self.validate_assignment(morphology_name, assignments)
        profile = self._profile(morphology_name)
        behaviors = profile.get("behaviors", {})
        if not isinstance(behaviors, Mapping) or behavior not in behaviors:
            raise MorphologyLibraryError(
                f"Behavior {behavior!r} is not defined for {morphology_name}"
            )
        raw_steps = behaviors[behavior]
        if not isinstance(raw_steps, list | tuple):
            raise MorphologyLibraryError(
                "Behavior sequence must be an array"
            )
        if not any(isinstance(item, Mapping) for item in raw_steps):
            return ()

        command_parameters = dict(parameters or {})
        assigned_roles = {item.target_role for item in assignments}
        steps: list[BehaviorProgramStep] = []
        for index, raw_step in enumerate(raw_steps):
            if not isinstance(raw_step, Mapping):
                raise MorphologyLibraryError(
                    "Composite behavior steps must all be objects"
                )
            has_posture = "posture" in raw_step
            has_drive = "drive" in raw_step
            if has_posture == has_drive:
                raise MorphologyLibraryError(
                    "Composite step needs exactly one posture or drive"
                )
            phase = str(
                raw_step.get("phase", f"STEP_{index + 1}")
            ).strip()
            if not phase:
                raise MorphologyLibraryError(
                    "Composite behavior phase cannot be empty"
                )
            if has_posture:
                posture_name = str(raw_step["posture"])
                targets = tuple(
                    self._resolve_posture(
                        profile,
                        posture_name,
                        assignments,
                    )
                )
                steps.append(
                    BehaviorProgramStep(
                        phase=phase,
                        posture_targets=targets,
                    )
                )
                continue

            drive = raw_step["drive"]
            if not isinstance(drive, Mapping):
                raise MorphologyLibraryError(
                    "Composite drive step must be an object"
                )
            duration_s = self._program_parameter(
                drive,
                command_parameters,
                value_name="default_duration_s",
                parameter_name="duration_parameter",
            )
            linear_m_s = self._program_parameter(
                drive,
                command_parameters,
                value_name="default_linear_m_s",
                parameter_name="linear_parameter",
                default=0.0,
            )
            yaw_rate_rad_s = self._program_parameter(
                drive,
                command_parameters,
                value_name="default_yaw_rate_rad_s",
                parameter_name="yaw_rate_parameter",
                default=0.0,
            )
            lateral_m_s = self._program_parameter(
                drive,
                command_parameters,
                value_name="default_lateral_m_s",
                parameter_name="lateral_parameter",
                default=0.0,
            )
            if duration_s <= 0.0:
                raise MorphologyLibraryError(
                    "Composite drive duration must be positive"
                )
            raw_roles = drive.get("active_target_roles", ())
            if not isinstance(raw_roles, list | tuple) or not raw_roles:
                raise MorphologyLibraryError(
                    "Composite drive needs active_target_roles"
                )
            active_roles = tuple(str(role) for role in raw_roles)
            if len(set(active_roles)) != len(active_roles):
                raise MorphologyLibraryError(
                    "Composite drive roles must be unique"
                )
            missing_roles = sorted(set(active_roles) - assigned_roles)
            if missing_roles:
                raise MorphologyLibraryError(
                    "Composite drive roles are not assigned: "
                    f"{missing_roles}"
                )
            # Reuse the standard limits and selector validation.
            self.drive_commands(
                morphology_name,
                assignments,
                linear_m_s,
                yaw_rate_rad_s,
                lateral_m_s,
            )
            steps.append(
                BehaviorProgramStep(
                    phase=phase,
                    duration_s=duration_s,
                    linear_m_s=linear_m_s,
                    yaw_rate_rad_s=yaw_rate_rad_s,
                    lateral_m_s=lateral_m_s,
                    active_target_roles=active_roles,
                )
            )
        if not steps:
            raise MorphologyLibraryError(
                "Composite behavior must contain at least one step"
            )
        return tuple(steps)

    def ready_joint_targets(
        self,
        morphology_name: str,
        assignments: Sequence[AssignedModule],
    ) -> tuple[JointTarget, ...]:
        return self.behavior_joint_targets(
            morphology_name, "prepare", assignments
        )

    def drive_joint_targets(
        self,
        morphology_name: str,
        assignments: Sequence[AssignedModule],
        linear_m_s: float,
        yaw_rate_rad_s: float,
        lateral_m_s: float = 0.0,
    ) -> tuple[JointTarget, ...]:
        """Resolve the posture required by this particular drive mode."""

        self.validate_assignment(morphology_name, assignments)
        profile = self._profile(morphology_name)
        drive = profile.get("drive")
        if not isinstance(drive, Mapping):
            raise MorphologyLibraryError(f"{morphology_name} cannot drive")
        if bool(drive.get("preserve_current_posture", False)):
            return ()
        posture_name = str(profile["ready_posture"])
        translation_posture = drive.get("translation_ready_posture")
        turn_posture = drive.get("turn_ready_posture")
        if translation_posture is not None and abs(float(linear_m_s)) > 1.0e-9:
            posture_name = str(translation_posture)
        elif (
            turn_posture is not None
            and abs(float(linear_m_s)) <= 1.0e-9
            and abs(float(yaw_rate_rad_s)) > 1.0e-9
        ):
            posture_name = str(turn_posture)
        return tuple(
            self._resolve_posture(profile, posture_name, assignments)
        )

    def drive_restore_joint_targets(
        self,
        morphology_name: str,
        assignments: Sequence[AssignedModule],
    ) -> tuple[JointTarget, ...]:
        """Resolve the optional posture restored after a timed drive."""

        self.validate_assignment(morphology_name, assignments)
        profile = self._profile(morphology_name)
        drive = profile.get("drive")
        if not isinstance(drive, Mapping):
            raise MorphologyLibraryError(f"{morphology_name} cannot drive")
        posture_name = drive.get("restore_posture_after_drive")
        if posture_name is None:
            return ()
        return tuple(
            self._resolve_posture(profile, str(posture_name), assignments)
        )

    def drive_commands(
        self,
        morphology_name: str,
        assignments: Sequence[AssignedModule],
        linear_m_s: float,
        yaw_rate_rad_s: float,
        lateral_m_s: float = 0.0,
    ) -> dict[str, dict[str, float]]:
        self.validate_assignment(morphology_name, assignments)
        profile = self._profile(morphology_name)
        drive = profile.get("drive")
        if not isinstance(drive, Mapping):
            raise MorphologyLibraryError(f"{morphology_name} cannot drive")
        linear = self._bounded(
            linear_m_s, float(drive["max_linear_m_s"]), "linear speed"
        )
        yaw_rate = self._bounded(
            yaw_rate_rad_s,
            float(drive["max_yaw_rate_rad_s"]),
            "yaw rate",
        )
        lateral = self._bounded(
            lateral_m_s,
            float(drive.get("max_lateral_m_s", 0.0)),
            "lateral speed",
        )
        if (
            not bool(drive.get("allow_combined_translation_and_turn", True))
            and abs(yaw_rate) > 1.0e-9
            and (abs(linear) > 1.0e-9 or abs(lateral) > 1.0e-9)
        ):
            raise MorphologyLibraryError(
                "This morphology requires separate translation and turn commands"
            )
        turn_radius = float(drive.get("turn_radius_m", 0.0))
        actuation = str(drive.get("actuation", "module_wheels"))
        if actuation not in {"module_wheels", "pan"}:
            raise MorphologyLibraryError(
                f"Unsupported drive actuation {actuation!r}"
            )
        effective_radius = float(drive.get("effective_radius_m", 0.0))
        max_pan_rate = float(drive.get("max_pan_rate_rad_s", 0.0))
        if actuation == "pan" and (
            not math.isfinite(effective_radius)
            or effective_radius <= 0.0
            or not math.isfinite(max_pan_rate)
            or max_pan_rate <= 0.0
        ):
            raise MorphologyLibraryError(
                "PAN drive needs positive effective radius and rate limit"
            )
        selectors = drive.get("selectors")
        if not isinstance(selectors, list | tuple):
            raise MorphologyLibraryError("Drive selectors must be an array")
        commands: dict[str, dict[str, float]] = {}
        for selector in selectors:
            if not isinstance(selector, Mapping):
                raise MorphologyLibraryError("Drive selector must be an object")
            active_when = str(selector.get("active_when", "always"))
            if active_when not in {"always", "translation", "pure_turn"}:
                raise MorphologyLibraryError(
                    f"Unsupported drive selector mode {active_when!r}"
                )
            if active_when == "pure_turn" and not (
                abs(linear) <= 1.0e-9 and abs(yaw_rate) > 1.0e-9
            ):
                continue
            if active_when == "translation" and not (
                abs(linear) > 1.0e-9 or abs(lateral) > 1.0e-9
            ):
                continue
            matched = self._matches(selector, assignments)
            if not matched:
                raise MorphologyLibraryError(
                    f"Drive selector matched no module: {dict(selector)}"
                )
            velocity = (
                float(selector.get("linear_scale", 1.0)) * linear
                + float(selector.get("lateral_scale", 0.0)) * lateral
                + float(selector.get("turn_scale", 0.0))
                * yaw_rate
                * turn_radius
            )
            selector_actuation = str(selector.get("actuation", actuation))
            if selector_actuation not in {"module_wheels", "pan"}:
                raise MorphologyLibraryError(
                    f"Unsupported selector actuation {selector_actuation!r}"
                )
            for assignment in matched:
                if selector_actuation == "pan":
                    pan_rate = (
                        float(selector.get("actuator_scale", 1.0))
                        * velocity
                        / effective_radius
                    )
                    commands[assignment.module_id] = {
                        "vx": 0.0,
                        "vy": 0.0,
                        "yaw_rate": 0.0,
                        "pan_rate_rad_s": pan_rate,
                    }
                else:
                    commands[assignment.module_id] = {
                        "vx": velocity,
                        "vy": 0.0,
                        "yaw_rate": (
                            float(selector.get("yaw_actuator_scale", 0.0))
                            * yaw_rate
                        ),
                    }
        if not commands:
            raise MorphologyLibraryError("Drive profile selected no locomotor")

        # Preserve the requested morphology twist when one or more PAN
        # locomotors hit their speed limit.  Clipping each actuator
        # independently changes the relative wheel speeds and therefore the
        # path curvature.  A single scale factor keeps their ratios intact.
        pan_commands = [
            command
            for command in commands.values()
            if "pan_rate_rad_s" in command
        ]
        if pan_commands:
            if not math.isfinite(max_pan_rate) or max_pan_rate <= 0.0:
                raise MorphologyLibraryError(
                    "PAN drive needs a positive rate limit"
                )
            peak_rate = max(
                abs(float(command["pan_rate_rad_s"]))
                for command in pan_commands
            )
            if peak_rate > max_pan_rate:
                scale = max_pan_rate / peak_rate
                for command in pan_commands:
                    command["pan_rate_rad_s"] *= scale
        return commands

    def _resolve_posture(
        self,
        profile: Mapping[str, Any],
        posture_name: str,
        assignments: Sequence[AssignedModule],
    ) -> list[JointTarget]:
        postures = profile.get("postures", {})
        if not isinstance(postures, Mapping) or posture_name not in postures:
            raise MorphologyLibraryError(f"Unknown posture {posture_name!r}")
        raw_targets = postures[posture_name]
        if not isinstance(raw_targets, list | tuple):
            raise MorphologyLibraryError("Posture targets must be an array")
        result: list[JointTarget] = []
        for raw_target in raw_targets:
            if not isinstance(raw_target, Mapping):
                raise MorphologyLibraryError("Joint target must be an object")
            joint = str(raw_target.get("joint", ""))
            if joint not in {"pan", "tilt"}:
                raise MorphologyLibraryError(f"Unsupported joint {joint!r}")
            angle = float(raw_target.get("angle_rad"))
            if not math.isfinite(angle):
                raise MorphologyLibraryError("Joint target must be finite")
            angle_reference = str(
                raw_target.get("angle_reference", "absolute")
            )
            if angle_reference not in {"absolute", "captured_neutral"}:
                raise MorphologyLibraryError(
                    "Joint angle_reference must be absolute or "
                    "captured_neutral"
                )
            if angle_reference == "captured_neutral" and joint != "tilt":
                raise MorphologyLibraryError(
                    "Only tilt targets can use captured_neutral"
                )
            raw_tolerance = raw_target.get("tolerance_rad")
            tolerance = (
                None
                if raw_tolerance is None
                else float(raw_tolerance)
            )
            if tolerance is not None and (
                not math.isfinite(tolerance) or tolerance <= 0.0
            ):
                raise MorphologyLibraryError(
                    "Joint target tolerance must be positive and finite"
                )
            raw_max_servo_error = raw_target.get("max_servo_error_rad")
            max_servo_error = (
                None
                if raw_max_servo_error is None
                else float(raw_max_servo_error)
            )
            if max_servo_error is not None and (
                not math.isfinite(max_servo_error)
                or max_servo_error <= 0.0
            ):
                raise MorphologyLibraryError(
                    "Joint max_servo_error_rad must be positive and finite"
                )
            raw_max_servo_speed = raw_target.get("max_servo_speed_rad_s")
            max_servo_speed = (
                None
                if raw_max_servo_speed is None
                else float(raw_max_servo_speed)
            )
            if max_servo_speed is not None and (
                not math.isfinite(max_servo_speed)
                or max_servo_speed <= 0.0
            ):
                raise MorphologyLibraryError(
                    "Joint max_servo_speed_rad_s must be positive and finite"
                )
            raw_group = raw_target.get("coordination_group")
            coordination_group = None
            if raw_group is not None:
                coordination_group = str(raw_group).strip()
                if not coordination_group:
                    raise MorphologyLibraryError(
                        "Joint coordination_group cannot be empty"
                    )
                if joint != "tilt":
                    raise MorphologyLibraryError(
                        "Only tilt targets can be coordinated"
                    )
                coordination_group = (
                    f"{posture_name}:{coordination_group}"
                )
            raw_pusher_role = raw_target.get("pusher_target_role")
            raw_pusher_speed = raw_target.get("pusher_linear_m_s")
            if (raw_pusher_role is None) != (raw_pusher_speed is None):
                raise MorphologyLibraryError(
                    "pusher_target_role and pusher_linear_m_s must be "
                    "declared together"
                )
            pusher_module_id: str | None = None
            pusher_speed: float | None = None
            if raw_pusher_role is not None:
                if joint != "tilt":
                    raise MorphologyLibraryError(
                        "A posture pusher can only accompany a TILT target"
                    )
                pusher_matches = tuple(
                    item
                    for item in assignments
                    if item.target_role == str(raw_pusher_role)
                )
                if len(pusher_matches) != 1:
                    raise MorphologyLibraryError(
                        "pusher_target_role must match exactly one module"
                    )
                pusher_module_id = pusher_matches[0].module_id
                pusher_speed = float(raw_pusher_speed)
                if (
                    not math.isfinite(pusher_speed)
                    or abs(pusher_speed) <= 1.0e-9
                ):
                    raise MorphologyLibraryError(
                        "pusher_linear_m_s must be finite and non-zero"
                    )
            matched = self._matches(raw_target, assignments)
            if not matched:
                raise MorphologyLibraryError(
                    f"Posture target matched no module: {dict(raw_target)}"
                )
            for assignment in matched:
                result.append(
                    JointTarget(
                        module_id=assignment.module_id,
                        joint=joint,
                        angle_rad=angle,
                        target_vertex_id=assignment.target_vertex_id,
                        target_role=assignment.target_role,
                        tolerance_rad=tolerance,
                        coordination_group=coordination_group,
                        pusher_module_id=pusher_module_id,
                        pusher_linear_m_s=pusher_speed,
                        max_servo_error_rad=max_servo_error,
                        max_servo_speed_rad_s=max_servo_speed,
                        angle_reference=angle_reference,
                    )
                )
        return result

    @staticmethod
    def _matches(
        selector: Mapping[str, Any],
        assignments: Sequence[AssignedModule],
    ) -> tuple[AssignedModule, ...]:
        vertex = selector.get("target_vertex_id")
        role = selector.get("target_role")
        if (vertex is None) == (role is None):
            raise MorphologyLibraryError(
                "A selector needs exactly one target_vertex_id or target_role"
            )
        return tuple(
            item
            for item in assignments
            if (
                item.target_vertex_id == str(vertex)
                if vertex is not None
                else item.target_role == str(role)
            )
        )

    def _profile(self, morphology_name: str) -> Mapping[str, Any]:
        try:
            return self._profiles[morphology_name]
        except KeyError as error:
            raise MorphologyLibraryError(
                f"Unknown morphology {morphology_name!r}"
            ) from error

    @staticmethod
    def _program_parameter(
        specification: Mapping[str, Any],
        command_parameters: Mapping[str, Any],
        *,
        value_name: str,
        parameter_name: str,
        default: float | None = None,
    ) -> float:
        raw_default = specification.get(value_name, default)
        if raw_default is None:
            raise MorphologyLibraryError(
                f"Composite drive is missing {value_name}"
            )
        parameter_key = specification.get(parameter_name)
        raw_value = raw_default
        if parameter_key is not None:
            key = str(parameter_key).strip()
            if not key:
                raise MorphologyLibraryError(
                    f"Composite drive {parameter_name} cannot be empty"
                )
            raw_value = command_parameters.get(key, raw_default)
        value = float(raw_value)
        if not math.isfinite(value):
            raise MorphologyLibraryError(
                f"Composite drive {value_name} must be finite"
            )
        return value

    @staticmethod
    def _bounded(value: float, limit: float, label: str) -> float:
        value = float(value)
        if not math.isfinite(value) or not math.isfinite(limit) or limit < 0.0:
            raise MorphologyLibraryError(f"Invalid {label}")
        if abs(value) > limit + 1.0e-12:
            raise MorphologyLibraryError(
                f"Requested {label} {value:.3f} exceeds limit {limit:.3f}"
            )
        return value

    @staticmethod
    def _validate_profile(name: str, profile: Mapping[str, Any]) -> None:
        count = profile.get("module_count")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise MorphologyLibraryError(f"{name} has invalid module_count")
        ready = profile.get("ready_posture")
        postures = profile.get("postures")
        if not isinstance(ready, str) or not isinstance(postures, Mapping):
            raise MorphologyLibraryError(f"{name} has no ready posture")
        if ready not in postures:
            raise MorphologyLibraryError(f"{name} ready posture is undefined")
