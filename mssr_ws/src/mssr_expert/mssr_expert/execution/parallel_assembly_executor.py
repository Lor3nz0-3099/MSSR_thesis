"""State machine that executes a parallel SMORES-EP assembly plan."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from mssr_expert.execution.assembly_policy import (
    DEFAULT_ASSEMBLY_EXECUTION_POLICY,
)
from mssr_expert.execution.primitive_protocol import (
    PrimitiveGoalRequest,
    PrimitiveStatusView,
    make_assisted_align_faces_goal,
    make_align_faces_goal,
    make_dock_goal,
    make_drive_to_pose_goal,
    make_undock_goal,
    parse_primitive_statuses,
)
from mssr_expert.planning.smores_ep.assembly_sequence import (
    AssemblyAction,
    AssemblyWave,
    ParallelAssemblyPlan,
)
from mssr_expert.planning.smores_ep.unfolding import PlanarPose


class ParallelAssemblyExecutionError(ValueError):
    """Raised when an assembly plan cannot be executed."""


def physical_posture_groups(
    raw_groups: object,
    target_to_module: Mapping[str, str],
    posture_modules: set[str],
    field_name: str = "post_assembly_tilt_groups_by_vertex",
) -> tuple[tuple[str, ...], ...]:
    """Resolve target-vertex posture waves into assigned physical modules."""

    if raw_groups is None:
        return ()
    if not isinstance(raw_groups, list | tuple):
        raise ParallelAssemblyExecutionError(
            f"{field_name} must be an array."
        )
    groups: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for raw_group in raw_groups:
        if not isinstance(raw_group, list | tuple) or not raw_group:
            raise ParallelAssemblyExecutionError(
                f"Every {field_name} group must be non-empty."
            )
        group: list[str] = []
        for raw_vertex in raw_group:
            vertex = str(raw_vertex)
            module_id = target_to_module.get(vertex)
            if module_id is None or module_id not in posture_modules:
                raise ParallelAssemblyExecutionError(
                    f"{field_name} references a vertex without "
                    f"a tilt target: {vertex!r}."
                )
            if module_id in seen:
                raise ParallelAssemblyExecutionError(
                    f"Posture module {module_id!r} occurs in two groups."
                )
            seen.add(module_id)
            group.append(module_id)
        groups.append(tuple(group))
    if seen != posture_modules:
        missing = sorted(posture_modules - seen)
        raise ParallelAssemblyExecutionError(
            f"{field_name} omits modules: {missing}."
        )
    return tuple(groups)


def physical_fold_push_pairs(
    raw_pairs: object,
    target_to_module: Mapping[str, str],
    posture_modules: set[str],
) -> dict[str, tuple[str, float]]:
    """Resolve coupled pusher/lifter fold pairs into physical module IDs."""

    if raw_pairs is None:
        return {}
    if not isinstance(raw_pairs, list | tuple):
        raise ParallelAssemblyExecutionError(
            "post_assembly_push_pairs_by_vertex must be an array."
        )
    result: dict[str, tuple[str, float]] = {}
    used_pushers: set[str] = set()
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, Mapping):
            raise ParallelAssemblyExecutionError(
                "Every post-assembly push pair must be an object."
            )
        pusher_vertex = str(raw_pair.get("pusher_vertex", "")).strip()
        lifter_vertex = str(raw_pair.get("lifter_vertex", "")).strip()
        pusher = target_to_module.get(pusher_vertex)
        lifter = target_to_module.get(lifter_vertex)
        if pusher is None or lifter is None:
            raise ParallelAssemblyExecutionError(
                "Post-assembly push pair references an unknown target "
                f"vertex: {pusher_vertex!r}->{lifter_vertex!r}."
            )
        if lifter not in posture_modules:
            raise ParallelAssemblyExecutionError(
                f"Fold lifter {lifter_vertex!r} has no TILT target."
            )
        if pusher == lifter or pusher in used_pushers or lifter in result:
            raise ParallelAssemblyExecutionError(
                "Fold push pairs must contain distinct one-to-one modules."
            )
        speed = float(raw_pair.get("linear_m_s", 0.0))
        if not math.isfinite(speed) or abs(speed) <= 1.0e-9:
            raise ParallelAssemblyExecutionError(
                "Every fold pusher linear_m_s must be finite and non-zero."
            )
        result[lifter] = (pusher, speed)
        used_pushers.add(pusher)
    return result


@dataclass(frozen=True)
class AssemblyExecutionDecision:
    """One deterministic decision produced by the executor."""

    state: str
    phase: str
    wave_index: int
    wave_count: int

    primitive_goal: PrimitiveGoalRequest | None

    active_goal_ids: tuple[str, ...]
    completed_action_count: int
    total_action_count: int

    done: bool
    success: bool

    message: str = ""

    @property
    def primitive_goal_payload(
        self,
    ) -> dict[str, Any] | None:
        """Return the serialized goal for the ROS publisher."""

        if self.primitive_goal is None:
            return None

        return self.primitive_goal.to_dict()


class ParallelAssemblyExecutor:
    """Execute assembly waves through alignment, contact, and docking."""

    def __init__(
        self,
        plan: ParallelAssemblyPlan,
        execution_id: str = "self-assembly",
        align_timeout_s: float = (
            DEFAULT_ASSEMBLY_EXECUTION_POLICY.align_timeout_s
        ),
        dock_timeout_s: float = (
            DEFAULT_ASSEMBLY_EXECUTION_POLICY.dock_timeout_s
        ),
        align_retry_count: int = (
            DEFAULT_ASSEMBLY_EXECUTION_POLICY.align_retry_count
        ),
        dock_recovery_count: int = (
            DEFAULT_ASSEMBLY_EXECUTION_POLICY.dock_recovery_count
        ),
        contact_quality_planar_tolerance_m: float | None = (
            DEFAULT_ASSEMBLY_EXECUTION_POLICY
            .contact_quality_planar_tolerance_m
        ),
        contact_quality_retry_count: int = (
            DEFAULT_ASSEMBLY_EXECUTION_POLICY.contact_quality_retry_count
        ),
        top_bottom_contact_tolerance_m: float | None = (
            DEFAULT_ASSEMBLY_EXECUTION_POLICY.top_bottom_contact_tolerance_m
        ),
        contact_approach_feedback: bool = (
            DEFAULT_ASSEMBLY_EXECUTION_POLICY.contact_approach_feedback
        ),
        max_concurrent_alignments_per_wave: int = (
            DEFAULT_ASSEMBLY_EXECUTION_POLICY
            .max_concurrent_alignments_per_wave
        ),
        snap_docking_faces_to_nominal: bool = (
            DEFAULT_ASSEMBLY_EXECUTION_POLICY.snap_docking_faces_to_nominal
        ),
        enable_borrowed_helper: bool = False,
        helper_lift_tilt_rad: float = math.pi / 4.0,
        helper_joint_timeout_s: float = 30.0,
        layout_pose_by_module: Mapping[str, PlanarPose] | None = None,
        post_assembly_tilt_by_module: Mapping[str, float] | None = None,
        post_assembly_pan_by_module: Mapping[str, float] | None = None,
        posture_tilt_tolerance_rad: float | None = None,
        posture_tilt_max_servo_error_rad: float | None = None,
        coordinate_posture_tilts: bool = False,
        posture_tilt_groups_by_module: (
            tuple[tuple[str, ...], ...] | None
        ) = None,
        posture_push_by_lifter_module: (
            Mapping[str, tuple[str, float]] | None
        ) = None,
        additional_known_module_ids: tuple[str, ...] = (),
    ) -> None:
        if not execution_id.strip():
            raise ParallelAssemblyExecutionError(
                "execution_id cannot be empty."
            )

        if align_timeout_s <= 0.0:
            raise ParallelAssemblyExecutionError(
                "align_timeout_s must be positive."
            )

        if dock_timeout_s <= 0.0:
            raise ParallelAssemblyExecutionError(
                "dock_timeout_s must be positive."
            )
        if not isinstance(align_retry_count, int) or align_retry_count < 0:
            raise ParallelAssemblyExecutionError(
                "align_retry_count must be a non-negative integer."
            )
        if not isinstance(dock_recovery_count, int) or dock_recovery_count < 0:
            raise ParallelAssemblyExecutionError(
                "dock_recovery_count must be a non-negative integer."
            )
        if contact_quality_planar_tolerance_m is not None and (
            not math.isfinite(contact_quality_planar_tolerance_m)
            or contact_quality_planar_tolerance_m <= 0.0
        ):
            raise ParallelAssemblyExecutionError(
                "contact_quality_planar_tolerance_m must be positive."
            )
        if (
            not isinstance(contact_quality_retry_count, int)
            or contact_quality_retry_count < 0
        ):
            raise ParallelAssemblyExecutionError(
                "contact_quality_retry_count must be a non-negative integer."
            )
        if top_bottom_contact_tolerance_m is not None and (
            not math.isfinite(top_bottom_contact_tolerance_m)
            or top_bottom_contact_tolerance_m <= 0.0
        ):
            raise ParallelAssemblyExecutionError(
                "top_bottom_contact_tolerance_m must be positive."
            )
        if (
            not math.isfinite(helper_lift_tilt_rad)
            or not 0.0 < abs(helper_lift_tilt_rad) <= math.pi / 2.0
        ):
            raise ParallelAssemblyExecutionError(
                "helper_lift_tilt_rad must be finite, non-zero and within "
                "the SMORES tilt limits."
            )
        if (
            not math.isfinite(helper_joint_timeout_s)
            or helper_joint_timeout_s <= 0.0
        ):
            raise ParallelAssemblyExecutionError(
                "helper_joint_timeout_s must be positive."
            )
        self.plan = plan
        self.execution_id = execution_id
        self.align_timeout_s = align_timeout_s
        self.dock_timeout_s = dock_timeout_s
        self.align_retry_count = align_retry_count
        self.dock_recovery_count = dock_recovery_count
        self.contact_quality_planar_tolerance_m = (
            contact_quality_planar_tolerance_m
        )
        self.contact_quality_retry_count = contact_quality_retry_count
        self.top_bottom_contact_tolerance_m = (
            top_bottom_contact_tolerance_m
        )
        self.contact_approach_feedback = bool(contact_approach_feedback)
        if (
            not isinstance(max_concurrent_alignments_per_wave, int)
            or isinstance(max_concurrent_alignments_per_wave, bool)
            or max_concurrent_alignments_per_wave < 0
        ):
            raise ParallelAssemblyExecutionError(
                "max_concurrent_alignments_per_wave must be a non-negative "
                "integer."
            )
        self.max_concurrent_alignments_per_wave = (
            max_concurrent_alignments_per_wave
        )
        self.snap_docking_faces_to_nominal = bool(
            snap_docking_faces_to_nominal
        )
        self.enable_borrowed_helper = enable_borrowed_helper
        self.helper_lift_tilt_rad = helper_lift_tilt_rad
        self.helper_joint_timeout_s = helper_joint_timeout_s
        self.coordinate_posture_tilts = bool(coordinate_posture_tilts)
        self._additional_known_module_ids = set(
            additional_known_module_ids
        )

        self._layout_pose_by_module = dict(layout_pose_by_module or {})
        self._post_assembly_tilt_by_module = dict(
            post_assembly_tilt_by_module or {}
        )
        self._post_assembly_pan_by_module = dict(
            post_assembly_pan_by_module or {}
        )
        if posture_tilt_tolerance_rad is not None and (
            not math.isfinite(posture_tilt_tolerance_rad)
            or posture_tilt_tolerance_rad <= 0.0
        ):
            raise ParallelAssemblyExecutionError(
                "posture_tilt_tolerance_rad must be positive and finite."
            )
        self._posture_tilt_tolerance_rad = posture_tilt_tolerance_rad
        if posture_tilt_max_servo_error_rad is not None and (
            not math.isfinite(posture_tilt_max_servo_error_rad)
            or posture_tilt_max_servo_error_rad <= 0.0
        ):
            raise ParallelAssemblyExecutionError(
                "posture_tilt_max_servo_error_rad must be positive and "
                "finite."
            )
        self._posture_tilt_max_servo_error_rad = (
            posture_tilt_max_servo_error_rad
        )
        self._posture_tilt_groups = tuple(
            tuple(group) for group in (posture_tilt_groups_by_module or ())
        )
        self._posture_push_by_lifter_module = dict(
            posture_push_by_lifter_module or {}
        )
        # Every assembled module that is not an active posture target stays
        # structurally held. This is the stable folding model used by RC-car,
        # manipulator and holonomic morphologies.
        all_assigned_module_ids = {self.plan.root_module_id} | {
            action.mobile_module_id for action in self.plan.all_actions
        }
        self._posture_structural_hold_module_ids = (
            all_assigned_module_ids
            - set(self._post_assembly_tilt_by_module)
            - set(self._post_assembly_pan_by_module)
        ) | {
            self.plan.root_module_id,
            *self._posture_push_by_lifter_module,
            *(
                pusher
                for pusher, _speed in (
                    self._posture_push_by_lifter_module.values()
                )
            ),
        }
        self._validate_pose_targets()

        self._wave_index = 0
        self._phase = (
            "LAYOUT"
            if self._layout_pose_by_module
            else "POSTURE"
            if not plan.waves and (
                self._post_assembly_tilt_by_module
                or self._post_assembly_pan_by_module
            )
            else "REACH"
        )

        self._submitted_goal_by_action: dict[int, str] = {}
        self._succeeded_actions: set[int] = set()
        self._docked_actions: set[int] = set()
        self._retry_by_action: dict[int, int] = {}
        self._staging_path_fallback_by_action: dict[int, int] = {}
        self._align_timeout_retry_by_action: dict[tuple[str, int], int] = {}
        self._deferred_align_retry_actions: list[int] = []
        self._approach_recovery_pending_actions: set[int] = set()
        self._dock_recovery_by_action: dict[int, int] = {}
        self._dock_recovery_pending_actions: set[int] = set()

        self._awaiting_admission_goal_id: str | None = None
        self._completed_action_count = 0

        self._state = (
            "READY_LAYOUT"
            if self._layout_pose_by_module
            else "SUCCEEDED"
            if not plan.waves and not self._post_assembly_tilt_by_module
            and not self._post_assembly_pan_by_module
            else "READY_POSTURE"
            if not plan.waves and (
                self._post_assembly_tilt_by_module
                or self._post_assembly_pan_by_module
            )
            else "READY"
        )

        self._failure_message = ""

        self._helper_module_id = (
            self._select_borrowed_helper()
            if enable_borrowed_helper
            else None
        )
        if plan.requires_helper and not enable_borrowed_helper:
            self._state = "FAILED"
            self._phase = "CONFIGURATION"
            self._failure_message = (
                "The target contains mobile LEFT/RIGHT docking faces and "
                "requires the helping-module procedure, but it is disabled."
            )
        elif plan.requires_helper and self._helper_module_id is None:
            self._state = "FAILED"
            self._phase = "CONFIGURATION"
            self._failure_message = (
                "The helping-module procedure is enabled, but the plan has "
                "neither a dedicated reserve nor a free future target leaf "
                "that can be borrowed."
            )
        self._helper_action_index = 0
        self._helper_phase = "HELPER_APPROACH"
        self._helper_goal_id: str | None = None
        self._helper_goal_sequence = 0
        self._helper_retry_by_key: dict[tuple[int, int, str], int] = {}

        self._layout_goal_by_module: dict[str, str] = {}
        self._layout_succeeded: set[str] = set()
        self._layout_retry_by_module: dict[str, int] = {}
        self._layout_awaiting_goal_id: str | None = None

        self._posture_goal_by_module: dict[str, str] = {}
        self._posture_succeeded: set[str] = set()
        self._posture_retry_by_module: dict[str, int] = {}
        self._posture_awaiting_goal_id: str | None = None

    @property
    def helper_module_id(self) -> str | None:
        """Return the target module temporarily reserved as helper."""

        return self._helper_module_id

    def _validate_pose_targets(self) -> None:
        known_modules = {
            self.plan.root_module_id,
            *(action.mobile_module_id for action in self.plan.all_actions),
            *(action.parent_module_id for action in self.plan.all_actions),
            *self._additional_known_module_ids,
        }
        unknown_layout = set(self._layout_pose_by_module) - known_modules
        unknown_posture = (
            set(self._post_assembly_tilt_by_module)
            | set(self._post_assembly_pan_by_module)
        ) - known_modules
        push_modules = set(self._posture_push_by_lifter_module)
        push_modules.update(
            pusher
            for pusher, _speed in self._posture_push_by_lifter_module.values()
        )
        unknown_modules = (
            unknown_layout | unknown_posture
            | (push_modules - known_modules)
        )
        if unknown_modules:
            raise ParallelAssemblyExecutionError(
                "Layout/posture targets reference unknown modules: "
                f"{sorted(unknown_modules)}"
            )
        for module_id, angle in self._post_assembly_tilt_by_module.items():
            if (
                not math.isfinite(angle)
                or not -math.pi / 2.0 <= angle <= math.pi / 2.0
            ):
                raise ParallelAssemblyExecutionError(
                    f"Invalid post-assembly tilt for {module_id}: {angle}"
                )
        for module_id, angle in self._post_assembly_pan_by_module.items():
            if not math.isfinite(angle):
                raise ParallelAssemblyExecutionError(
                    f"Invalid post-assembly pan for {module_id}: {angle}"
                )
        used_pushers: set[str] = set()
        for lifter, (pusher, speed) in (
            self._posture_push_by_lifter_module.items()
        ):
            if lifter not in self._post_assembly_tilt_by_module:
                raise ParallelAssemblyExecutionError(
                    f"Fold lifter {lifter!r} has no post-assembly TILT target."
                )
            if lifter == pusher or pusher in used_pushers:
                raise ParallelAssemblyExecutionError(
                    "Posture push pairs must be one-to-one and distinct."
                )
            if not math.isfinite(speed) or abs(speed) <= 1.0e-9:
                raise ParallelAssemblyExecutionError(
                    f"Invalid fold push speed for {pusher}: {speed}"
                )
            used_pushers.add(pusher)
        flattened_groups = [
            module_id
            for group in self._posture_tilt_groups
            for module_id in group
        ]
        if len(flattened_groups) != len(set(flattened_groups)):
            raise ParallelAssemblyExecutionError(
                "A module cannot occur in multiple posture tilt groups."
            )
        if self._posture_tilt_groups and set(flattened_groups) != set(
            self._post_assembly_tilt_by_module
        ):
            raise ParallelAssemblyExecutionError(
                "Posture tilt groups must partition every posture module."
            )

    def _step_layout(
        self,
        statuses: Mapping[str, PrimitiveStatusView],
    ) -> AssemblyExecutionDecision:
        """Move every assigned module to the unfolded 2-D layout in parallel."""

        for module_id, goal_id in tuple(self._layout_goal_by_module.items()):
            status = statuses.get(goal_id)
            if status is None:
                continue
            if (
                self._layout_awaiting_goal_id == goal_id
                and status.state in {
                    "accepted", "running", "succeeded", "failed",
                    "canceled", "rejected",
                }
            ):
                self._layout_awaiting_goal_id = None
            if status.failed:
                retries = self._layout_retry_by_module.get(module_id, 0)
                if status.code == "TIMEOUT" and retries < self.align_retry_count:
                    self._layout_retry_by_module[module_id] = retries + 1
                    self._layout_goal_by_module.pop(module_id, None)
                    continue
                self._state = "FAILED"
                self._failure_message = (
                    f"Planar layout goal {goal_id} failed: "
                    f"{status.code} {status.message}"
                ).strip()
                return self._decision(None, self._failure_message)
            if status.succeeded:
                self._layout_succeeded.add(module_id)

        expected = set(self._layout_pose_by_module)
        if self._layout_succeeded == expected:
            self._layout_goal_by_module.clear()
            self._layout_awaiting_goal_id = None
            if self.plan.waves:
                self._phase = "REACH"
                self._state = "PLANAR_LAYOUT_REACHED"
                return self._decision(
                    None,
                    "All modules reached the unfolded planar layout; docking starts.",
                )
            return self._begin_posture_or_finish()

        if self._layout_awaiting_goal_id is not None:
            self._state = "WAITING_LAYOUT_ADMISSION"
            return self._decision(
                None,
                f"Waiting for backend admission of {self._layout_awaiting_goal_id}.",
            )

        for module_id in sorted(expected):
            if (
                module_id in self._layout_succeeded
                or module_id in self._layout_goal_by_module
            ):
                continue
            pose = self._layout_pose_by_module[module_id]
            retry = self._layout_retry_by_module.get(module_id, 0)
            goal_id = f"{self.execution_id}-layout-{module_id}"
            if retry:
                goal_id += f"-r{retry}"
            goal = make_drive_to_pose_goal(
                goal_id,
                module_id,
                pose.x_m,
                pose.y_m,
                pose.yaw_rad,
                self.align_timeout_s,
            )
            self._layout_goal_by_module[module_id] = goal_id
            self._layout_awaiting_goal_id = goal_id
            self._state = "DISPATCHING_LAYOUT"
            return self._decision(
                goal,
                f"Dispatching planar layout goal for {module_id}.",
            )

        self._state = "WAITING_LAYOUT_RESULTS"
        return self._decision(
            None,
            f"Waiting for {len(expected - self._layout_succeeded)} layout goals.",
        )

    def _step_posture(
        self,
        statuses: Mapping[str, PrimitiveStatusView],
    ) -> AssemblyExecutionDecision:
        """Fold the docked planar structure into its final 3-D posture."""

        for module_id, goal_id in tuple(self._posture_goal_by_module.items()):
            status = statuses.get(goal_id)
            if status is None:
                continue
            if (
                self._posture_awaiting_goal_id == goal_id
                and status.state in {
                    "accepted", "running", "succeeded", "failed",
                    "canceled", "rejected",
                }
            ):
                self._posture_awaiting_goal_id = None
            if status.failed:
                retries = self._posture_retry_by_module.get(module_id, 0)
                if retries < self.align_retry_count:
                    self._posture_retry_by_module[module_id] = retries + 1
                    self._posture_goal_by_module.pop(module_id, None)
                    continue
                self._state = "FAILED"
                self._failure_message = (
                    f"Post-assembly posture goal {goal_id} failed: "
                    f"{status.code} {status.message}"
                ).strip()
                return self._decision(None, self._failure_message)
            if status.succeeded:
                self._posture_succeeded.add(module_id)

        tilt_expected = set(self._post_assembly_tilt_by_module)
        pan_expected = set(self._post_assembly_pan_by_module)
        expected = tilt_expected | pan_expected
        if self._posture_succeeded == expected:
            self._state = "SUCCEEDED"
            self._phase = "COMPLETE"
            return self._decision(
                None,
                "Assembly and final morphology posture completed.",
            )

        posture_group_index = 0
        posture_group = tilt_expected - self._posture_succeeded
        posture_joint = "tilt"
        coordinate_group = self.coordinate_posture_tilts
        if tilt_expected <= self._posture_succeeded:
            posture_group = pan_expected - self._posture_succeeded
            posture_joint = "pan"
            coordinate_group = False
        elif self._posture_tilt_groups:
            coordinate_group = True
            for index, configured_group in enumerate(
                self._posture_tilt_groups
            ):
                configured = set(configured_group)
                if not configured <= self._posture_succeeded:
                    posture_group_index = index
                    posture_group = configured
                    break

        if self._posture_awaiting_goal_id is not None:
            self._state = "WAITING_POSTURE_ADMISSION"
            return self._decision(None, "Waiting for posture goal admission.")

        for module_id in sorted(posture_group):
            if (
                module_id in self._posture_succeeded
                or module_id in self._posture_goal_by_module
            ):
                continue
            retry = self._posture_retry_by_module.get(module_id, 0)
            goal_id = f"{self.execution_id}-posture-{module_id}"
            if retry:
                goal_id += f"-r{retry}"
            parameters: dict[str, Any] = {
                "angle_rad": (
                    self._post_assembly_tilt_by_module[module_id]
                    if posture_joint == "tilt"
                    else self._post_assembly_pan_by_module[module_id]
                ),
            }
            if posture_joint == "tilt":
                # Capture every non-target module (for example an RC-Car
                # chassis link) into structural hold once the coordinated
                # group reaches its target.
                parameters["hold_after_group_module_ids"] = sorted(
                    self._posture_structural_hold_module_ids
                )
                push = self._posture_push_by_lifter_module.get(module_id)
                if push is not None:
                    pusher_module_id, pusher_linear_m_s = push
                    parameters.update(
                        {
                            "pusher_module_id": pusher_module_id,
                            "pusher_linear_m_s": pusher_linear_m_s,
                            "stabilize_during_group_module_ids": [
                                self.plan.root_module_id
                            ],
                        }
                    )
            if (
                posture_joint == "tilt"
                and self._posture_tilt_tolerance_rad is not None
            ):
                parameters["tolerance_rad"] = (
                    self._posture_tilt_tolerance_rad
                )
            if (
                posture_joint == "tilt"
                and self._posture_tilt_max_servo_error_rad is not None
            ):
                parameters["max_servo_error_rad"] = (
                    self._posture_tilt_max_servo_error_rad
                )
            if coordinate_group:
                coordination_size = (
                    len(posture_group)
                    if self._posture_tilt_groups
                    else len(tilt_expected)
                )
                parameters.update(
                    {
                        "coordination_group": (
                            f"{self.execution_id}-posture"
                            + (
                                f"-group-{posture_group_index}"
                                if self._posture_tilt_groups
                                else ""
                            )
                        ),
                        "coordination_size": coordination_size,
                    }
                )
            goal = PrimitiveGoalRequest(
                goal_id=goal_id,
                primitive=f"set_{posture_joint}",
                module_ids=(module_id,),
                parameters=parameters,
                timeout_s=self.helper_joint_timeout_s,
            )
            self._posture_goal_by_module[module_id] = goal_id
            self._posture_awaiting_goal_id = goal_id
            self._state = "DISPATCHING_POSTURE"
            return self._decision(
                goal,
                f"Dispatching final {posture_joint} for {module_id}.",
            )

        self._state = "WAITING_POSTURE_RESULTS"
        return self._decision(
            None,
            "Waiting for final morphology posture group "
            f"{posture_group_index + 1}.",
        )

    def _begin_posture_or_finish(self) -> AssemblyExecutionDecision:
        if (
            self._post_assembly_tilt_by_module
            or self._post_assembly_pan_by_module
        ):
            self._phase = "POSTURE"
            self._state = "READY_POSTURE"
            return self._decision(
                None,
                "Docking complete; folding into the final morphology posture.",
            )
        self._phase = "COMPLETE"
        self._state = "SUCCEEDED"
        return self._decision(None, "Assembly plan completed.")

    def step(
        self,
        status_payload: Mapping[str, Any] | None = None,
    ) -> AssemblyExecutionDecision:
        """Consume backend status and possibly emit one new goal."""

        if self._state == "SUCCEEDED":
            return self._decision(
                primitive_goal=None,
                message="Assembly plan completed.",
            )

        if self._state == "FAILED":
            return self._decision(
                primitive_goal=None,
                message=self._failure_message,
            )

        statuses = parse_primitive_statuses(
            status_payload
        )

        if self._phase == "LAYOUT":
            return self._step_layout(statuses)

        if self._phase == "POSTURE":
            return self._step_posture(statuses)

        current_wave = self._current_wave()
        if any(action.requires_helper for action in current_wave.actions):
            return self._step_helper_wave(statuses)

        self._consume_statuses(statuses)

        if self._state == "FAILED":
            return self._decision(
                primitive_goal=None,
                message=self._failure_message,
            )

        self._advance_if_phase_complete()

        if self._state == "SUCCEEDED":
            return self._decision(
                primitive_goal=None,
                message="Assembly plan completed.",
            )

        if self._phase == "POSTURE":
            return self._step_posture(statuses)

        current_wave = self._current_wave()

        awaiting_goal_id = self._awaiting_admission_goal_id
        if awaiting_goal_id is not None:
            self._state = f"WAITING_{self._phase}_ADMISSION"

            return self._decision(
                primitive_goal=None,
                message=(
                    "Waiting for backend admission of "
                    f"{awaiting_goal_id}."
                ),
            )

        next_action_index = self._next_unsubmitted_action_index(
            current_wave
        )

        if next_action_index is None:
            self._state = f"WAITING_{self._phase}_RESULTS"

            return self._decision(
                primitive_goal=None,
                message=(
                    f"Waiting for all {self._phase.lower()} "
                    "goals in the current wave."
                ),
            )

        action = current_wave.actions[next_action_index]
        goal = self._goal_for_action(
            action=action,
            action_index=next_action_index,
        )

        self._submitted_goal_by_action[
            next_action_index
        ] = goal.goal_id

        self._awaiting_admission_goal_id = goal.goal_id
        self._state = f"DISPATCHING_{self._phase}"

        return self._decision(
            primitive_goal=goal,
            message=(
                f"Dispatching {goal.primitive} for "
                f"{action.mobile_module_id}."
            ),
        )

    def _select_borrowed_helper(self) -> str | None:
        """Prefer a dedicated reserve, then borrow a future target leaf.

        A future target leaf can be borrowed while it is still disconnected,
        then released before the wave in which it receives its final role.
        """

        if self._additional_known_module_ids:
            return sorted(self._additional_known_module_ids)[0]

        helper_payloads = {
            action.mobile_module_id
            for action in self.plan.all_actions
            if action.requires_helper
        }
        candidates = sorted(
            {
                action.mobile_module_id
                for action in self.plan.all_actions
                if not action.requires_helper
                and action.mobile_module_id not in helper_payloads
                and action.mobile_module_id != self.plan.root_module_id
            }
        )
        return candidates[0] if candidates else None

    def _step_helper_wave(
        self,
        statuses: Mapping[str, PrimitiveStatusView],
    ) -> AssemblyExecutionDecision:
        """Execute lateral docking actions serially with one borrowed helper."""

        wave = self._current_wave()
        if not all(action.requires_helper for action in wave.actions):
            self._state = "FAILED"
            self._failure_message = (
                "A mixed helper/non-helper wave is unsupported; split the "
                "target assembly wave before execution."
            )
            return self._decision(None, self._failure_message)

        if self._helper_module_id is None:
            self._state = "FAILED"
            self._phase = "CONFIGURATION"
            modules = sorted(
                action.mobile_module_id for action in wave.actions
            )
            self._failure_message = (
                "No free future module can be borrowed as helper for "
                f"mobile modules {modules}."
            )
            return self._decision(
                None,
                self._failure_message,
            )

        if self._helper_goal_id is not None:
            status = statuses.get(self._helper_goal_id)
            if status is None or status.state in {"accepted", "running"}:
                self._state = f"WAITING_{self._helper_phase}"
                return self._decision(
                    None,
                    f"Waiting for {self._helper_goal_id}.",
                )
            if status.failed:
                return self._recover_helper_phase(status)
            if status.succeeded:
                self._helper_goal_id = None
                self._advance_helper_phase()

        if self._state in {"SUCCEEDED", "FAILED"}:
            return self._decision(
                None,
                "Assembly plan completed."
                if self._state == "SUCCEEDED"
                else self._failure_message,
            )

        if not any(
            action.requires_helper
            for action in self._current_wave().actions
        ):
            return self._decision(
                None,
                "Helping-module wave completed; normal parallel assembly "
                "resumes on the next control tick.",
            )

        wave = self._current_wave()
        action = wave.actions[self._helper_action_index]
        goal = self._helper_goal_for_action(action)
        self._helper_goal_id = goal.goal_id
        self._state = f"DISPATCHING_{self._helper_phase}"
        return self._decision(
            goal,
            f"{self._helper_module_id} executes {self._helper_phase.lower()} "
            f"for {action.mobile_module_id}.",
        )

    def _recover_helper_phase(
        self,
        status: PrimitiveStatusView,
    ) -> AssemblyExecutionDecision:
        """Retry one helper subphase or return to assisted alignment."""

        failed_phase = self._helper_phase
        self._helper_goal_id = None
        if (
            failed_phase == "TARGET_DOCK"
            and status.code == "DOCKING_REJECTED"
        ):
            recoveries = self._helper_retry_count("TARGET_DOCK")
            if recoveries < self.dock_recovery_count:
                self._increment_helper_retry("TARGET_DOCK")
                self._helper_phase = "ASSISTED_ALIGN"
                self._state = "HELPER_REALIGNING_AFTER_DOCK_REJECTION"
                return self._decision(
                    None,
                    "Target docking was rejected; the helper will push the "
                    "payload closer and retry without discarding the wave.",
                )

        retry_limit = (
            self.align_retry_count
            if failed_phase in {"HELPER_APPROACH", "ASSISTED_ALIGN"}
            else self.dock_recovery_count
        )
        retries = self._helper_retry_count(failed_phase)
        if retries < retry_limit:
            self._increment_helper_retry(failed_phase)
            if (
                failed_phase == "HELPER_DOCK"
                and status.code == "DOCKING_REJECTED"
            ):
                self._helper_phase = "HELPER_APPROACH"
            self._state = f"RETRYING_{self._helper_phase}"
            return self._decision(
                None,
                f"Retrying helper phase {self._helper_phase} after "
                f"{status.code}: {status.message}",
            )

        self._state = "FAILED"
        self._failure_message = (
            f"Helping-module primitive {status.goal_id} failed in "
            f"{failed_phase}: {status.code} {status.message}"
        ).strip()
        return self._decision(None, self._failure_message)

    def _advance_helper_phase(self) -> None:
        """Advance the paper-inspired attach/lift/push/release sequence."""

        phase_order = {
            "HELPER_APPROACH": "HELPER_DOCK",
            "HELPER_DOCK": "HELPER_LIFT",
            "HELPER_LIFT": "HELPER_LOWER",
            "HELPER_LOWER": "ASSISTED_ALIGN",
            "ASSISTED_ALIGN": "TARGET_DOCK",
            "TARGET_DOCK": "HELPER_RELEASE",
            "HELPER_RELEASE": "HELPER_RESET",
        }
        next_phase = phase_order.get(self._helper_phase)
        if next_phase is not None:
            self._helper_phase = next_phase
            self._phase = next_phase
            return
        if self._helper_phase != "HELPER_RESET":
            raise ParallelAssemblyExecutionError(
                f"Unknown helping-module phase {self._helper_phase!r}."
            )

        self._completed_action_count += 1
        self._helper_action_index += 1
        wave = self._current_wave()
        if self._helper_action_index < len(wave.actions):
            self._helper_phase = "HELPER_APPROACH"
            self._phase = self._helper_phase
            self._state = "READY_NEXT_HELPER_ACTION"
            return

        self._wave_index += 1
        self._helper_action_index = 0
        self._helper_phase = "HELPER_APPROACH"
        self._reset_wave_tracking()
        if self._wave_index >= len(self.plan.waves):
            self._state = "SUCCEEDED"
            self._phase = "COMPLETE"
        else:
            self._state = "READY_NEXT_WAVE"
            self._phase = "ALIGN"

    def _helper_goal_for_action(
        self,
        action: AssemblyAction,
    ) -> PrimitiveGoalRequest:
        """Build the next primitive in one assisted docking action."""

        helper_id = self._helper_module_id
        if helper_id is None:
            raise ParallelAssemblyExecutionError("No helper is reserved.")
        opposite_face = {
            "LEFT": "RIGHT",
            "RIGHT": "LEFT",
        }.get(action.mobile_face)
        if opposite_face is None:
            raise ParallelAssemblyExecutionError(
                "Helping-module actions require a LEFT or RIGHT mobile face."
            )
        self._phase = self._helper_phase
        self._helper_goal_sequence += 1
        goal_id = (
            f"{self.execution_id}-w{self._wave_index}"
            f"-a{self._helper_action_index}"
            f"-{self._helper_phase.lower()}"
            f"-s{self._helper_goal_sequence}"
        )
        if self._helper_phase == "HELPER_APPROACH":
            return make_align_faces_goal(
                goal_id,
                helper_id,
                "TOP",
                action.mobile_module_id,
                opposite_face,
                0,
                self.align_timeout_s,
                top_bottom_contact_tolerance_m=(
                    self.top_bottom_contact_tolerance_m
                ),
                contact_approach_feedback=(
                    self.contact_approach_feedback
                ),
            )
        if self._helper_phase == "HELPER_DOCK":
            return make_dock_goal(
                goal_id,
                helper_id,
                "TOP",
                action.mobile_module_id,
                opposite_face,
                0,
                self.dock_timeout_s,
                top_bottom_contact_tolerance_m=(
                    self.top_bottom_contact_tolerance_m
                ),
                contact_approach_feedback=(
                    self.contact_approach_feedback
                ),
            )
        if self._helper_phase in {"HELPER_LIFT", "HELPER_LOWER", "HELPER_RESET"}:
            angle = (
                self.helper_lift_tilt_rad
                if self._helper_phase == "HELPER_LIFT"
                else 0.0
            )
            return PrimitiveGoalRequest(
                goal_id=goal_id,
                primitive="set_tilt",
                module_ids=(helper_id,),
                parameters={"angle_rad": angle},
                timeout_s=self.helper_joint_timeout_s,
            )
        if self._helper_phase == "ASSISTED_ALIGN":
            return make_assisted_align_faces_goal(
                goal_id,
                action.mobile_module_id,
                action.mobile_face,
                action.parent_module_id,
                action.parent_face,
                helper_id,
                action.clocking_quarter_turns,
                self.align_timeout_s,
                top_bottom_contact_tolerance_m=(
                    self.top_bottom_contact_tolerance_m
                ),
                contact_approach_feedback=(
                    self.contact_approach_feedback
                ),
            )
        if self._helper_phase == "TARGET_DOCK":
            return make_dock_goal(
                goal_id,
                action.mobile_module_id,
                action.mobile_face,
                action.parent_module_id,
                action.parent_face,
                action.clocking_quarter_turns,
                self.dock_timeout_s,
                top_bottom_contact_tolerance_m=(
                    self.top_bottom_contact_tolerance_m
                ),
                contact_approach_feedback=(
                    self.contact_approach_feedback
                ),
                snap_to_nominal=(
                    self.snap_docking_faces_to_nominal
                ),
            )
        if self._helper_phase == "HELPER_RELEASE":
            return make_undock_goal(
                goal_id,
                helper_id,
                "TOP",
                action.mobile_module_id,
                opposite_face,
                self.dock_timeout_s,
            )
        raise ParallelAssemblyExecutionError(
            f"Cannot create a helper goal for {self._helper_phase!r}."
        )

    def _helper_retry_count(self, phase: str) -> int:
        key = (self._wave_index, self._helper_action_index, phase)
        return self._helper_retry_by_key.get(key, 0)

    def _increment_helper_retry(self, phase: str) -> None:
        key = (self._wave_index, self._helper_action_index, phase)
        self._helper_retry_by_key[key] = self._helper_retry_by_key.get(key, 0) + 1

    def _consume_statuses(
        self,
        statuses: Mapping[str, PrimitiveStatusView],
    ) -> None:
        """Update one barrier phase using the latest backend statuses."""

        for action_index, goal_id in tuple(
            self._submitted_goal_by_action.items()
        ):
            status = statuses.get(goal_id)

            if status is None:
                continue

            if (
                self._awaiting_admission_goal_id == goal_id
                and status.state in {
                    "accepted",
                    "running",
                    "succeeded",
                    "failed",
                    "canceled",
                    "rejected",
                }
            ):
                self._awaiting_admission_goal_id = None

            if status.failed:
                retry_key = (self._phase, action_index)
                retries = self._align_timeout_retry_by_action.get(
                    retry_key,
                    0,
                )
                if (
                    self._phase == "APPROACH"
                    and status.code
                    in {
                        "TIMEOUT",
                        "CONTACT_TIMEOUT",
                        "CONTACT_POSE_INVALID",
                    }
                    and retries < self.align_retry_count
                ):
                    self._align_timeout_retry_by_action[retry_key] = (
                        retries + 1
                    )
                    self._submitted_goal_by_action.pop(action_index, None)
                    self._approach_recovery_pending_actions.add(action_index)
                    self._state = "WAITING_APPROACH_RECOVERY_BARRIER"
                    continue
                if (
                    self._phase in {"REACH", "ALIGN"}
                    and status.code
                    in {
                        "TIMEOUT",
                        "CONTACT_TIMEOUT",
                        "NO_COLLISION_FREE_STAGING_PATH",
                        "CONTACT_POSE_INVALID",
                    }
                    and retries < self.align_retry_count
                ):
                    self._align_timeout_retry_by_action[
                        retry_key
                    ] = retries + 1
                    self._retry_by_action[action_index] = (
                        self._retry_by_action.get(action_index, 0) + 1
                    )
                    if status.code == "NO_COLLISION_FREE_STAGING_PATH":
                        self._staging_path_fallback_by_action[
                            action_index
                        ] = min(
                            2,
                            self._staging_path_fallback_by_action.get(
                                action_index,
                                0,
                            )
                            + 1,
                        )
                    self._submitted_goal_by_action.pop(action_index, None)
                    if self._awaiting_admission_goal_id == goal_id:
                        self._awaiting_admission_goal_id = None
                    if (
                        self._phase == "REACH"
                        and status.code == "NO_COLLISION_FREE_STAGING_PATH"
                    ):
                        # A peer can temporarily occupy the only route. Let
                        # the other REACH goals settle, then replan from the
                        # updated live geometry without collapsing the wave
                        # into a permanently serial execution.
                        if (
                            action_index
                            not in self._deferred_align_retry_actions
                        ):
                            self._deferred_align_retry_actions.append(
                                action_index
                            )
                        self._state = "DEFERRED_STAGING_PATH_REPLAN"
                    else:
                        self._state = f"RETRYING_{self._phase}"
                    continue
                recoveries = self._dock_recovery_by_action.get(
                    action_index,
                    0,
                )
                if (
                    status.primitive == "dock"
                    and status.code
                    in {
                        "DOCKING_REJECTED",
                        "CONTACT_ALIGNMENT_LOST",
                        "TIMEOUT",
                    }
                    and recoveries < self.dock_recovery_count
                ):
                    self._dock_recovery_by_action[
                        action_index
                    ] = recoveries + 1
                    self._retry_by_action[action_index] = (
                        self._retry_by_action.get(action_index, 0) + 1
                    )
                    self._submitted_goal_by_action.pop(action_index, None)
                    self._succeeded_actions.discard(action_index)
                    self._dock_recovery_pending_actions.add(action_index)
                    self._state = "WAITING_DOCK_RECOVERY_BARRIER"
                    continue
                self._state = "FAILED"
                self._failure_message = (
                    f"Primitive {goal_id} failed: "
                    f"{status.code} {status.message}"
                ).strip()
                return

            if status.succeeded:
                self._succeeded_actions.add(
                    action_index
                )
                if self._phase == "DOCK":
                    self._docked_actions.add(action_index)

    def _advance_if_phase_complete(self) -> None:
        """Advance the paper-style collective barrier state machine."""

        current_wave = self._current_wave()
        expected_action_indices = set(
            range(len(current_wave.actions))
        )

        if self._phase == "DOCK":
            terminal_docks = (
                self._docked_actions
                | self._dock_recovery_pending_actions
            )
            if terminal_docks != expected_action_indices:
                return
            if self._dock_recovery_pending_actions:
                # Successful peers are now rigidly latched. Only failed
                # actions re-enter ALIGN -> APPROACH -> DOCK, in parallel if
                # more than one dock needs recovery.
                self._phase = "ALIGN"
                self._prepare_next_barrier()
                self._dock_recovery_pending_actions.clear()
                self._state = "REALIGNING_BEFORE_DOCK_RETRY"
                return

            self._completed_action_count += len(current_wave.actions)
            self._wave_index += 1

            if self._wave_index >= len(self.plan.waves):
                self._reset_wave_tracking()
                if (
                    self._post_assembly_tilt_by_module
                    or self._post_assembly_pan_by_module
                ):
                    self._state = "READY_POSTURE"
                    self._phase = "POSTURE"
                else:
                    self._state = "SUCCEEDED"
                    self._phase = "COMPLETE"
                return

            self._phase = "REACH"
            self._reset_wave_tracking()
            self._state = "READY_NEXT_WAVE"
            return

        if self._phase == "APPROACH":
            terminal_approaches = (
                self._succeeded_actions
                | self._approach_recovery_pending_actions
            )
            if terminal_approaches != expected_action_indices:
                return
            if self._approach_recovery_pending_actions:
                # A failed close-contact run must never be resubmitted as
                # another blind push. Back the complete independent wave to
                # ALIGN, then repeat the collective approach barrier.
                for action_index in expected_action_indices:
                    self._retry_by_action[action_index] = (
                        self._retry_by_action.get(action_index, 0) + 1
                    )
                self._phase = "ALIGN"
                self._submitted_goal_by_action.clear()
                self._succeeded_actions.clear()
                self._deferred_align_retry_actions.clear()
                self._approach_recovery_pending_actions.clear()
                self._awaiting_admission_goal_id = None
                self._state = "REALIGNING_AFTER_APPROACH_FAILURE"
                return

        if self._succeeded_actions != expected_action_indices:
            return

        next_phase = {
            "REACH": "ALIGN",
            "ALIGN": "APPROACH",
            "APPROACH": "DOCK",
        }.get(self._phase)
        if next_phase is None:
            raise ParallelAssemblyExecutionError(
                f"Unknown execution phase {self._phase!r}."
            )
        self._phase = next_phase
        self._prepare_next_barrier()
        self._state = f"READY_{next_phase}"

    def _prepare_next_barrier(self) -> None:
        """Start the next wave barrier, preserving already latched peers."""

        self._submitted_goal_by_action.clear()
        self._succeeded_actions = set(self._docked_actions)
        self._deferred_align_retry_actions.clear()
        self._awaiting_admission_goal_id = None

    def _reset_wave_tracking(self) -> None:
        """Clear all per-action state before entering another wave."""

        self._submitted_goal_by_action.clear()
        self._succeeded_actions.clear()
        self._docked_actions.clear()
        self._retry_by_action.clear()
        self._staging_path_fallback_by_action.clear()
        self._align_timeout_retry_by_action.clear()
        self._deferred_align_retry_actions.clear()
        self._approach_recovery_pending_actions.clear()
        self._dock_recovery_by_action.clear()
        self._dock_recovery_pending_actions.clear()
        self._awaiting_admission_goal_id = None

    def _next_unsubmitted_action_index(
        self,
        wave: AssemblyWave,
    ) -> int | None:
        """Return the next action not yet sent to the backend."""

        if (
            self._phase in {"REACH", "ALIGN", "APPROACH"}
            and self.max_concurrent_alignments_per_wave > 0
        ):
            active_motion_count = sum(
                action_index not in self._succeeded_actions
                for action_index in self._submitted_goal_by_action
            )
            if (
                active_motion_count
                >= self.max_concurrent_alignments_per_wave
            ):
                return None

        candidates: list[int] = []
        for action_index in range(len(wave.actions)):
            if (
                action_index
                not in self._submitted_goal_by_action
                and action_index not in self._succeeded_actions
                and action_index
                not in self._approach_recovery_pending_actions
                and action_index not in self._dock_recovery_pending_actions
            ):
                candidates.append(action_index)

        for action_index in candidates:
            if action_index not in self._deferred_align_retry_actions:
                return action_index

        if not candidates:
            return None

        # A blocked REACH retry waits only until its moving peers settle. It
        # remains part of the same parallel wave and subsequent barriers.
        has_active_peer = any(
            action_index not in self._succeeded_actions
            for action_index in self._submitted_goal_by_action
        )
        if has_active_peer:
            return None

        action_index = next(
            deferred
            for deferred in self._deferred_align_retry_actions
            if deferred in candidates
        )
        self._deferred_align_retry_actions.remove(action_index)
        return action_index

    def _goal_for_action(
        self,
        action: AssemblyAction,
        action_index: int,
    ) -> PrimitiveGoalRequest:
        """Build the primitive goal for the current phase."""

        goal_id = self._goal_id(
            action_index=action_index,
            phase=self._phase,
        )

        if self._phase in {"REACH", "ALIGN", "APPROACH"}:
            return make_align_faces_goal(
                goal_id=goal_id,
                mobile_module_id=(
                    action.mobile_module_id
                ),
                mobile_face=action.mobile_face,
                parent_module_id=(
                    action.parent_module_id
                ),
                parent_face=action.parent_face,
                clocking_quarter_turns=(
                    action.clocking_quarter_turns
                ),
                timeout_s=self.align_timeout_s,
                contact_quality_planar_tolerance_m=(
                    self.contact_quality_planar_tolerance_m
                ),
                contact_quality_retry_count=(
                    self.contact_quality_retry_count
                ),
                top_bottom_contact_tolerance_m=(
                    self.top_bottom_contact_tolerance_m
                ),
                contact_approach_feedback=(
                    self.contact_approach_feedback
                ),
                execution_phase=self._phase.lower(),
                staging_path_fallback_level=(
                    self._staging_path_fallback_by_action.get(
                        action_index
                    )
                ),
            )

        if self._phase == "DOCK":
            return make_dock_goal(
                goal_id=goal_id,
                mobile_module_id=(
                    action.mobile_module_id
                ),
                mobile_face=action.mobile_face,
                parent_module_id=(
                    action.parent_module_id
                ),
                parent_face=action.parent_face,
                clocking_quarter_turns=(
                    action.clocking_quarter_turns
                ),
                timeout_s=self.dock_timeout_s,
                top_bottom_contact_tolerance_m=(
                    self.top_bottom_contact_tolerance_m
                ),
                contact_approach_feedback=(
                    self.contact_approach_feedback
                ),
                snap_to_nominal=(
                    self.snap_docking_faces_to_nominal
                ),
            )

        raise ParallelAssemblyExecutionError(
            f"Cannot create a goal for phase {self._phase!r}."
        )

    def _goal_id(
        self,
        action_index: int,
        phase: str,
    ) -> str:
        """Create a deterministic unique primitive goal ID."""

        goal_id = (
            f"{self.execution_id}"
            f"-w{self._wave_index}"
            f"-a{action_index}"
            f"-{phase.lower()}"
        )
        retry = self._retry_by_action.get(action_index, 0)
        if retry:
            goal_id += f"-r{retry}"
        return goal_id

    def _current_wave(self) -> AssemblyWave:
        """Return the currently executing assembly wave."""

        if not 0 <= self._wave_index < len(self.plan.waves):
            raise ParallelAssemblyExecutionError(
                "The current wave index is outside the plan."
            )

        return self.plan.waves[self._wave_index]

    def _active_goal_ids(self) -> tuple[str, ...]:
        """Return submitted goals that have not succeeded."""

        active = {
            goal_id
            for action_index, goal_id
            in self._submitted_goal_by_action.items()
            if action_index not in self._succeeded_actions
        }
        if self._helper_goal_id is not None:
            active.add(self._helper_goal_id)
        active.update(
            goal_id
            for module_id, goal_id in self._layout_goal_by_module.items()
            if module_id not in self._layout_succeeded
        )
        active.update(
            goal_id
            for module_id, goal_id in self._posture_goal_by_module.items()
            if module_id not in self._posture_succeeded
        )
        return tuple(sorted(active))

    def _decision(
        self,
        primitive_goal: PrimitiveGoalRequest | None,
        message: str,
    ) -> AssemblyExecutionDecision:
        """Create an observable executor decision."""

        return AssemblyExecutionDecision(
            state=self._state,
            phase=self._phase,
            wave_index=self._wave_index,
            wave_count=len(self.plan.waves),
            primitive_goal=primitive_goal,
            active_goal_ids=self._active_goal_ids(),
            completed_action_count=(
                self._completed_action_count
            ),
            total_action_count=self.plan.action_count,
            done=self._state in {
                "SUCCEEDED",
                "FAILED",
            },
            success=self._state == "SUCCEEDED",
            message=message,
        )
