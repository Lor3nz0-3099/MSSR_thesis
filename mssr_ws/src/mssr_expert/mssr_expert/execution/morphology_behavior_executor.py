"""State machine for operational behaviors of assembled morphologies."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from mssr_expert.behaviors.morphology_library import (
    AssignedModule,
    BehaviorProgramStep,
    JointTarget,
    MorphologyLibrary,
    MorphologyLibraryError,
)
from mssr_expert.execution.primitive_protocol import (
    PrimitiveGoalRequest,
    parse_primitive_statuses,
)


@dataclass(frozen=True)
class MorphologyCommand:
    command_id: str
    morphology: str
    behavior: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "MorphologyCommand":
        schema = payload.get(
            "schema_version", "mssr.morphology_command.v1"
        )
        if schema != "mssr.morphology_command.v1":
            raise ValueError(f"Unsupported morphology command schema: {schema}")
        parameters = payload.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError("Morphology command parameters must be an object")
        command = cls(
            command_id=str(payload.get("command_id", "")),
            morphology=str(payload.get("morphology", "")),
            behavior=str(payload.get("behavior", "")),
            parameters=dict(parameters),
        )
        if not command.command_id.strip():
            raise ValueError("Morphology command_id cannot be empty")
        if not command.morphology.strip() or not command.behavior.strip():
            raise ValueError("Morphology and behavior cannot be empty")
        return command


@dataclass(frozen=True)
class MorphologyBehaviorDecision:
    command_id: str
    morphology: str
    behavior: str
    state: str
    phase: str
    primitive_goal: PrimitiveGoalRequest | None = None
    locomotion: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    progress: float = 0.0
    done: bool = False
    success: bool = False
    message: str = ""


class MorphologyBehaviorExecutor:
    """Compose posture primitives and coordinated cluster locomotion."""

    def __init__(
        self,
        library: MorphologyLibrary,
        joint_timeout_s: float = 20.0,
    ) -> None:
        if not math.isfinite(joint_timeout_s) or joint_timeout_s <= 0.0:
            raise ValueError("Joint timeout must be positive")
        self._library = library
        self._joint_timeout_s = float(joint_timeout_s)
        self._command: MorphologyCommand | None = None
        self._assignments: tuple[AssignedModule, ...] = ()
        self._joint_targets: tuple[JointTarget, ...] = ()
        self._next_joint_index = 0
        self._completed_joint_indices: set[int] = set()
        self._active_goal_ids: dict[str, int] = {}
        self._awaiting_admission_goal_id: str | None = None
        self._drive_started_s: float | None = None
        self._drive_restoring = False
        self._program_steps: tuple[BehaviorProgramStep, ...] = ()
        self._program_step_index = 0
        self._program_loaded_posture_index: int | None = None
        self._program_drive_started_s: float | None = None
        self._state = "IDLE"
        self._failure_message = ""
        self._neutral_tilt_rad_by_module: dict[str, float] = {}

    @property
    def active(self) -> bool:
        return self._command is not None and self._state not in {
            "SUCCEEDED",
            "FAILED",
        }

    def start(
        self,
        command: MorphologyCommand,
        assignments: Sequence[AssignedModule],
        neutral_tilt_rad_by_module: Mapping[str, float] | None = None,
    ) -> None:
        assignments = tuple(assignments)
        self._library.validate_assignment(command.morphology, assignments)
        if command.behavior == "drive":
            self._validate_drive_parameters(command)
            targets = self._library.drive_joint_targets(
                command.morphology,
                assignments,
                float(command.parameters.get("linear_m_s", 0.0)),
                float(command.parameters.get("yaw_rate_rad_s", 0.0)),
                float(command.parameters.get("lateral_m_s", 0.0)),
            )
        elif command.behavior == "stop":
            targets = ()
        elif command.behavior == "prepare":
            program_steps = ()
            targets = self._library.behavior_joint_targets(
                command.morphology,
                command.behavior,
                assignments,
            )
        else:
            program_steps = self._library.composite_behavior_steps(
                command.morphology,
                command.behavior,
                assignments,
                command.parameters,
            )
            targets = (
                ()
                if program_steps
                else self._library.behavior_joint_targets(
                    command.morphology,
                    command.behavior,
                    assignments,
                )
            )
        if command.behavior in {"drive", "stop"}:
            program_steps = ()
        neutral_tilts = dict(neutral_tilt_rad_by_module or {})
        targets = self._apply_neutral_reference(targets, neutral_tilts)
        program_steps = tuple(
            replace(
                step,
                posture_targets=self._apply_neutral_reference(
                    step.posture_targets,
                    neutral_tilts,
                ),
            )
            for step in program_steps
        )
        self._command = command
        self._assignments = assignments
        self._joint_targets = targets
        self._next_joint_index = 0
        self._completed_joint_indices = set()
        self._active_goal_ids = {}
        self._awaiting_admission_goal_id = None
        self._drive_started_s = None
        self._drive_restoring = False
        self._program_steps = program_steps
        self._program_step_index = 0
        self._program_loaded_posture_index = None
        self._program_drive_started_s = None
        self._state = "READY"
        self._failure_message = ""
        self._neutral_tilt_rad_by_module = neutral_tilts

    def step(
        self,
        now_s: float,
        status_payload: Mapping[str, Any] | None = None,
    ) -> MorphologyBehaviorDecision:
        if not math.isfinite(now_s):
            raise ValueError("Behavior time must be finite")
        command = self._command
        if command is None:
            return MorphologyBehaviorDecision(
                command_id="",
                morphology="",
                behavior="",
                state="IDLE",
                phase="IDLE",
                message="No active morphology command.",
            )
        if self._state == "FAILED":
            return self._decision(
                phase="TERMINAL", done=True, message=self._failure_message
            )
        if self._state == "SUCCEEDED":
            return self._decision(
                phase="TERMINAL",
                done=True,
                success=True,
                message="Morphology behavior completed.",
            )

        if self._active_goal_ids:
            statuses = parse_primitive_statuses(status_payload)
            awaiting = self._awaiting_admission_goal_id
            if awaiting is not None:
                admitted = statuses.get(awaiting)
                if admitted is None or admitted.state not in {
                    "accepted",
                    "running",
                    "succeeded",
                    "failed",
                    "canceled",
                    "rejected",
                }:
                    self._state = "WAITING_JOINT_ADMISSION"
                    return self._decision(
                        phase=self._active_posture_phase(),
                        progress=self._active_progress(now_s),
                        message=f"Waiting for admission of {awaiting}.",
                    )
                self._awaiting_admission_goal_id = None
            for goal_id, target_index in tuple(
                self._active_goal_ids.items()
            ):
                status = statuses.get(goal_id)
                if status is None or not status.terminal:
                    continue
                if status.failed:
                    self._state = "FAILED"
                    self._failure_message = (
                        f"Joint primitive {status.goal_id} failed: "
                        f"{status.code} {status.message}"
                    ).strip()
                    return self._decision(
                        phase="TERMINAL",
                        done=True,
                        message=self._failure_message,
                    )
                self._completed_joint_indices.add(target_index)
                del self._active_goal_ids[goal_id]

        if self._program_steps:
            return self._step_composite_program(now_s)

        if self._next_joint_index < len(self._joint_targets):
            target = self._joint_targets[self._next_joint_index]
            active_groups = {
                self._joint_targets[index].coordination_group
                for index in self._active_goal_ids.values()
            }
            may_dispatch = not self._active_goal_ids or (
                target.coordination_group is not None
                and active_groups == {target.coordination_group}
            )
            if may_dispatch:
                target_index = self._next_joint_index
                self._next_joint_index += 1
                return self._dispatch_joint_target(
                    command,
                    target,
                    target_index,
                )

        if self._active_goal_ids:
            self._state = "WAITING_JOINT"
            return self._decision(
                phase="POSTURE",
                progress=self._posture_progress(),
                message=(
                    "Waiting for coordinated posture goals: "
                    + ", ".join(sorted(self._active_goal_ids))
                    + "."
                ),
            )

        if self._next_joint_index < len(self._joint_targets):
            raise RuntimeError("Posture dispatcher reached an invalid state")

        if command.behavior == "drive":
            if self._drive_restoring:
                self._state = "SUCCEEDED"
                return self._decision(
                    phase="TERMINAL",
                    done=True,
                    success=True,
                    progress=1.0,
                    message=(
                        "Timed cluster drive completed; locomotion stopped "
                        "and the operational posture was restored."
                    ),
                )
            if self._drive_started_s is None:
                self._drive_started_s = now_s
            duration_s = float(command.parameters["duration_s"])
            elapsed_s = now_s - self._drive_started_s
            if elapsed_s >= duration_s:
                restore_targets = (
                    self._library.drive_restore_joint_targets(
                        command.morphology,
                        self._assignments,
                    )
                )
                restore_targets = self._apply_neutral_reference(
                    restore_targets,
                    self._neutral_tilt_rad_by_module,
                )
                if restore_targets:
                    self._joint_targets = restore_targets
                    self._next_joint_index = 0
                    self._completed_joint_indices = set()
                    self._active_goal_ids = {}
                    self._awaiting_admission_goal_id = None
                    self._drive_restoring = True
                    self._state = "READY_RESTORE"
                    return self._decision(
                        phase="RESTORE_POSTURE",
                        progress=0.0,
                        message=(
                            "Timed cluster drive completed and stopped; "
                            "restoring the operational posture."
                        ),
                    )
                self._state = "SUCCEEDED"
                return self._decision(
                    phase="TERMINAL",
                    done=True,
                    success=True,
                    progress=1.0,
                    message="Timed cluster drive completed and stopped.",
                )
            locomotion = self._library.drive_commands(
                command.morphology,
                self._assignments,
                float(command.parameters.get("linear_m_s", 0.0)),
                float(command.parameters.get("yaw_rate_rad_s", 0.0)),
                float(command.parameters.get("lateral_m_s", 0.0)),
            )
            self._state = "RUNNING_DRIVE"
            return self._decision(
                phase="DRIVE",
                locomotion=locomotion,
                progress=max(0.0, min(1.0, elapsed_s / duration_s)),
                message="Driving connected component.",
            )

        self._state = "SUCCEEDED"
        return self._decision(
            phase="TERMINAL",
            done=True,
            success=True,
            progress=1.0,
            message=(
                "Cluster stopped."
                if command.behavior == "stop"
                else "Requested morphology posture reached."
            ),
        )

    def _dispatch_joint_target(
        self,
        command: MorphologyCommand,
        target: JointTarget,
        target_index: int,
    ) -> MorphologyBehaviorDecision:
        """Admit one posture goal, including its optional group barrier."""

        primitive = f"set_{target.joint}"
        parameters: dict[str, Any] = {
            "angle_rad": target.angle_rad,
        }
        if target.tolerance_rad is not None:
            parameters["tolerance_rad"] = target.tolerance_rad
        if target.max_servo_error_rad is not None:
            parameters["max_servo_error_rad"] = target.max_servo_error_rad
        # A posture command moves exactly one declared joint. Modules outside
        # the same coordinated posture group retain their complete PAN/TILT
        # state while that joint moves. Peers in the same group are excluded:
        # they are about to receive their own target and must not be held by
        # the other members of the barrier.
        coordinated_module_ids = {target.module_id}
        if target.coordination_group is not None:
            coordinated_module_ids = {
                candidate.module_id
                for candidate in self._joint_targets
                if candidate.coordination_group == target.coordination_group
            }
        structural_holds = {
            assignment.module_id
            for assignment in self._assignments
            if assignment.module_id not in coordinated_module_ids
        }
        structural_holds.update(target.structural_hold_module_ids)
        structural_holds.difference_update(coordinated_module_ids)
        if structural_holds:
            parameters["structural_hold_module_ids"] = sorted(
                structural_holds
            )
        if target.coordination_group is not None:
            parameters["coordination_group"] = (
                f"{command.command_id}-{target.coordination_group}"
            )
            parameters["coordination_size"] = sum(
                candidate.coordination_group
                == target.coordination_group
                for candidate in self._joint_targets
            )
        if target.pusher_module_id is not None:
            parameters["pusher_module_id"] = target.pusher_module_id
            parameters["pusher_linear_m_s"] = target.pusher_linear_m_s
            parameters["hold_after_group_module_ids"] = sorted(
                assignment.module_id for assignment in self._assignments
            )
            center_ids = sorted(
                assignment.module_id
                for assignment in self._assignments
                if assignment.target_role == "holonomic_center"
            )
            if center_ids:
                parameters["stabilize_during_group_module_ids"] = center_ids
        goal = PrimitiveGoalRequest(
            goal_id=(
                f"{command.command_id}{self._program_goal_suffix()}"
                f"-posture-{target_index:02d}"
            ),
            primitive=primitive,
            module_ids=(target.module_id,),
            parameters=parameters,
            timeout_s=self._joint_timeout_s,
        )
        self._active_goal_ids[goal.goal_id] = target_index
        self._awaiting_admission_goal_id = goal.goal_id
        self._state = "DISPATCHING_JOINT"
        return self._decision(
            phase=self._active_posture_phase(),
            primitive_goal=goal,
            progress=self._active_progress(0.0),
            message=(
                f"Setting {target.target_role}.{target.joint} "
                f"to {target.angle_rad:.3f} rad."
            ),
        )

    def _step_composite_program(
        self,
        now_s: float,
    ) -> MorphologyBehaviorDecision:
        """Execute a posture/locomotion program with a stop at each barrier."""

        if self._program_step_index >= len(self._program_steps):
            self._state = "SUCCEEDED"
            return self._decision(
                phase="TERMINAL",
                done=True,
                success=True,
                progress=1.0,
                message="Composite morphology behavior completed.",
            )
        step = self._program_steps[self._program_step_index]
        if step.kind == "posture":
            return self._step_program_posture(step, now_s)
        return self._step_program_drive(step, now_s)

    def _step_program_posture(
        self,
        step: BehaviorProgramStep,
        now_s: float,
    ) -> MorphologyBehaviorDecision:
        if self._program_loaded_posture_index != self._program_step_index:
            self._joint_targets = step.posture_targets
            self._next_joint_index = 0
            self._completed_joint_indices = set()
            self._active_goal_ids = {}
            self._awaiting_admission_goal_id = None
            self._program_loaded_posture_index = self._program_step_index

        if self._next_joint_index < len(self._joint_targets):
            target = self._joint_targets[self._next_joint_index]
            active_groups = {
                self._joint_targets[index].coordination_group
                for index in self._active_goal_ids.values()
            }
            may_dispatch = not self._active_goal_ids or (
                target.coordination_group is not None
                and active_groups == {target.coordination_group}
            )
            if may_dispatch:
                target_index = self._next_joint_index
                self._next_joint_index += 1
                return self._dispatch_joint_target(
                    self._command_or_raise(),
                    target,
                    target_index,
                )

        if self._active_goal_ids:
            self._state = "WAITING_JOINT"
            return self._decision(
                phase=step.phase,
                progress=self._active_progress(now_s),
                message=(
                    f"Waiting for {step.phase} posture goals: "
                    + ", ".join(sorted(self._active_goal_ids))
                    + "."
                ),
            )

        if self._next_joint_index < len(self._joint_targets):
            raise RuntimeError(
                "Composite posture dispatcher reached an invalid state"
            )
        completed_phase = step.phase
        self._program_step_index += 1
        self._program_loaded_posture_index = None
        self._joint_targets = ()
        self._next_joint_index = 0
        self._completed_joint_indices = set()
        self._state = "PROGRAM_BARRIER"
        return self._decision(
            phase=f"{completed_phase}_COMPLETE",
            progress=self._active_progress(now_s),
            message=f"{completed_phase} posture reached; locomotion stopped.",
        )

    def _step_program_drive(
        self,
        step: BehaviorProgramStep,
        now_s: float,
    ) -> MorphologyBehaviorDecision:
        if step.duration_s is None:
            raise RuntimeError("Composite drive step has no duration")
        if self._program_drive_started_s is None:
            self._program_drive_started_s = now_s
        elapsed_s = now_s - self._program_drive_started_s
        if elapsed_s >= step.duration_s:
            completed_phase = step.phase
            self._program_step_index += 1
            self._program_drive_started_s = None
            self._state = "PROGRAM_BARRIER"
            return self._decision(
                phase=f"{completed_phase}_STOP",
                progress=self._active_progress(now_s),
                message=f"{completed_phase} completed; locomotion stopped.",
            )

        locomotion = self._library.drive_commands(
            self._command_or_raise().morphology,
            self._assignments,
            step.linear_m_s,
            step.yaw_rate_rad_s,
            step.lateral_m_s,
        )
        allowed_module_ids = {
            assignment.module_id
            for assignment in self._assignments
            if assignment.target_role in step.active_target_roles
        }
        locomotion = {
            module_id: command
            for module_id, command in locomotion.items()
            if module_id in allowed_module_ids
        }
        if not locomotion:
            raise MorphologyLibraryError(
                f"Composite phase {step.phase!r} selected no locomotor"
            )
        self._state = "RUNNING_PROGRAM_DRIVE"
        return self._decision(
            phase=step.phase,
            locomotion=locomotion,
            progress=self._active_progress(now_s),
            message=(
                f"Executing {step.phase} with supports "
                f"{list(step.active_target_roles)}."
            ),
        )

    def _active_posture_phase(self) -> str:
        if (
            self._program_steps
            and self._program_step_index < len(self._program_steps)
        ):
            return self._program_steps[self._program_step_index].phase
        return "POSTURE"

    def _active_progress(self, now_s: float) -> float:
        if not self._program_steps:
            return self._posture_progress()
        step_count = len(self._program_steps)
        if self._program_step_index >= step_count:
            return 1.0
        step = self._program_steps[self._program_step_index]
        if step.kind == "posture":
            local_progress = self._posture_progress()
        elif (
            step.duration_s is not None
            and self._program_drive_started_s is not None
        ):
            local_progress = max(
                0.0,
                min(
                    1.0,
                    (now_s - self._program_drive_started_s)
                    / step.duration_s,
                ),
            )
        else:
            local_progress = 0.0
        return min(
            1.0,
            (self._program_step_index + local_progress) / step_count,
        )

    def _program_goal_suffix(self) -> str:
        if not self._program_steps:
            return ""
        return f"-step-{self._program_step_index:02d}"

    def _command_or_raise(self) -> MorphologyCommand:
        if self._command is None:
            raise RuntimeError("No active morphology command")
        return self._command

    def _decision(
        self,
        *,
        phase: str,
        primitive_goal: PrimitiveGoalRequest | None = None,
        locomotion: Mapping[str, Mapping[str, float]] | None = None,
        progress: float = 0.0,
        done: bool = False,
        success: bool = False,
        message: str = "",
    ) -> MorphologyBehaviorDecision:
        if self._command is None:
            raise RuntimeError("Cannot create decision without a command")
        return MorphologyBehaviorDecision(
            command_id=self._command.command_id,
            morphology=self._command.morphology,
            behavior=self._command.behavior,
            state=self._state,
            phase=phase,
            primitive_goal=primitive_goal,
            locomotion=dict(locomotion or {}),
            progress=progress,
            done=done,
            success=success,
            message=message,
        )

    def _posture_progress(self) -> float:
        if not self._joint_targets:
            return 1.0
        return len(self._completed_joint_indices) / len(
            self._joint_targets
        )

    @staticmethod
    def _apply_neutral_reference(
        targets: Sequence[JointTarget],
        neutral_tilt_rad_by_module: Mapping[str, float],
    ) -> tuple[JointTarget, ...]:
        resolved: list[JointTarget] = []
        for target in targets:
            if target.angle_reference == "absolute":
                resolved.append(target)
                continue
            try:
                neutral = float(
                    neutral_tilt_rad_by_module[target.module_id]
                )
            except KeyError as error:
                raise MorphologyLibraryError(
                    "No captured neutral tilt is available for "
                    f"{target.module_id}"
                ) from error
            if not math.isfinite(neutral):
                raise MorphologyLibraryError(
                    f"Captured neutral tilt for {target.module_id} is not finite"
                )
            resolved.append(
                replace(target, angle_rad=neutral + target.angle_rad)
            )
        return tuple(resolved)

    @staticmethod
    def _validate_drive_parameters(command: MorphologyCommand) -> None:
        duration = float(command.parameters.get("duration_s", 0.0))
        linear = float(command.parameters.get("linear_m_s", 0.0))
        yaw_rate = float(command.parameters.get("yaw_rate_rad_s", 0.0))
        lateral = float(command.parameters.get("lateral_m_s", 0.0))
        if not all(
            math.isfinite(value)
            for value in (duration, linear, lateral, yaw_rate)
        ):
            raise MorphologyLibraryError("Drive parameters must be finite")
        if duration <= 0.0:
            raise MorphologyLibraryError("Drive duration_s must be positive")
