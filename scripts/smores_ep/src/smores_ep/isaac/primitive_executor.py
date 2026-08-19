from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping

from smores_ep.config.geometry import SmoresGeometry
from smores_ep.control.differential_drive import PlanarPose
from smores_ep.control.pan_tilt import ContinuousAngleTracker
from smores_ep.control.teleop import InternalMotionMode, SmoresCommand
from smores_ep.docking.model import (
    DockingCommand,
    DockingPairEvaluation,
    DockingThresholds,
    evaluate_face_pair,
)
from smores_ep.isaac.docking import IsaacDockingManager
from smores_ep.isaac.dynamic_stage import ArticulationStateReader
from smores_ep.primitives.model import (
    PrimitiveGoal,
    PrimitiveName,
    PrimitiveState,
    PrimitiveStatus,
)
from smores_ep.primitives.collision_avoidance import (
    CircularObstacle,
    plan_collision_aware_path,
    segment_is_clear,
)
from smores_ep.primitives.pose_control import (
    PoseControllerConfig,
    PoseControlStep,
    axial_pose_adjustment_reference,
    drive_to_pose_step,
    wrap_angle,
)


@dataclass(frozen=True)
class PrimitiveExecutorStep:
    commands: Mapping[str, SmoresCommand]
    statuses: tuple[PrimitiveStatus, ...]

    @property
    def status(self) -> PrimitiveStatus | None:
        """Compatibility accessor for callers that expect one status."""

        return self.statuses[-1] if self.statuses else None


@dataclass
class _ActiveGoal:
    goal: PrimitiveGoal
    started_at_s: float
    resources: frozenset[str]
    resource_modes: Mapping[str, str]
    resolved_target_rad: float | None = None
    initial_metric: float | None = None
    alignment_staging_position_reached: bool = False
    alignment_staging_drive_direction: float | None = None
    alignment_approach_started: bool = False
    alignment_recovery_count: int = 0
    contact_quality_recovery_active: bool = False
    axial_alignment_mode: str = "curve"
    axial_pivot_sample_started_s: float | None = None
    axial_pivot_sample_yaw_rad: float | None = None
    axial_escape_until_s: float | None = None
    axial_stall_recovery_count: int = 0
    collision_route_xy: tuple[tuple[float, float], ...] = ()
    collision_route_goal_xy: tuple[float, float] | None = None
    collision_route_replans: int = 0


class IsaacPrimitiveExecutor:
    """Execute concurrent primitives when their physical resources are disjoint."""

    def __init__(
        self,
        stage: Any,
        module_roots: Mapping[str, str],
        states: Mapping[str, ArticulationStateReader],
        docking: IsaacDockingManager,
        motion_module_ids: tuple[str, ...] | None = None,
        geometry: SmoresGeometry | None = None,
        pose_controller: PoseControllerConfig | None = None,
        desired_face_marker_gap_m: float = 0.0055,
        face_alignment_staging_distance_m: float = 0.070,
        face_alignment_staging_yaw_tolerance_rad: float = math.radians(1.5),
        face_alignment_min_turn_speed_rad_s: float = 0.65,
        staging_collision_avoidance: bool = False,
        staging_center_clearance_m: float = 0.110,
        staging_waypoint_margin_m: float = 0.015,
        joint_tolerance_rad: float = math.radians(2.0),
        tilt_joint_tolerance_rad: float = math.radians(3.0),
    ) -> None:
        if set(module_roots) != set(states):
            raise ValueError("Every primitive-controlled module needs a state reader")
        if set(module_roots) != set(docking.module_ids):
            raise ValueError("Primitive and docking module registries must match")
        if (
            not math.isfinite(desired_face_marker_gap_m)
            or desired_face_marker_gap_m < 0.0
        ):
            raise ValueError("Desired face marker gap must be non-negative")
        if (
            not math.isfinite(face_alignment_staging_distance_m)
            or face_alignment_staging_distance_m <= 0.0
        ):
            raise ValueError(
                "Face-alignment staging distance must be positive"
            )
        if (
            not math.isfinite(face_alignment_staging_yaw_tolerance_rad)
            or face_alignment_staging_yaw_tolerance_rad <= 0.0
        ):
            raise ValueError(
                "Face-alignment staging yaw tolerance must be positive"
            )
        if (
            not math.isfinite(face_alignment_min_turn_speed_rad_s)
            or face_alignment_min_turn_speed_rad_s <= 0.0
        ):
            raise ValueError(
                "Face-alignment minimum turn speed must be positive"
            )
        if (
            not math.isfinite(staging_center_clearance_m)
            or staging_center_clearance_m <= 0.0
        ):
            raise ValueError("Staging centre clearance must be positive")
        if (
            not math.isfinite(staging_waypoint_margin_m)
            or staging_waypoint_margin_m <= 0.0
        ):
            raise ValueError("Staging waypoint margin must be positive")
        if not math.isfinite(joint_tolerance_rad) or joint_tolerance_rad <= 0.0:
            raise ValueError("Joint tolerance must be positive")
        if (
            not math.isfinite(tilt_joint_tolerance_rad)
            or tilt_joint_tolerance_rad <= 0.0
        ):
            raise ValueError("Tilt joint tolerance must be positive")

        self._stage = stage
        self._module_roots = dict(module_roots)
        self._states = dict(states)
        self._docking = docking
        self._docking_thresholds = getattr(
            docking,
            "thresholds",
            DockingThresholds(),
        )
        requested_motion_ids = (
            tuple(module_roots)
            if motion_module_ids is None
            else motion_module_ids
        )
        unknown_motion_ids = set(requested_motion_ids) - set(module_roots)
        if unknown_motion_ids:
            raise ValueError(
                "Unknown motion-controlled module(s): "
                + ", ".join(sorted(unknown_motion_ids))
            )
        self._motion_module_ids = frozenset(requested_motion_ids)
        self._geometry = geometry or SmoresGeometry()
        self._pose_controller = pose_controller or PoseControllerConfig()
        # The general 8 mm navigation tolerance is appropriate while driving
        # through free space, but the last connector approach must enter the
        # docking gate (8 mm marker separation) after accounting for the
        # nominal 5.5 mm CAD marker gap.
        self._face_approach_pose_controller = replace(
            self._pose_controller,
            max_linear_speed_m_s=min(
                self._pose_controller.max_linear_speed_m_s,
                0.025,
            ),
            max_angular_speed_rad_s=min(
                self._pose_controller.max_angular_speed_rad_s,
                0.8,
            ),
            # TOP<->BOTTOM uses the marker-normal gate supplied by the shared
            # assembly policy. Stop the final straight approach only after
            # entering that gate; planar centring remains governed by the
            # separate alignment-quality gate.
            position_tolerance_m=0.00075,
        )
        # Face alignment needs a considerably tighter staging pose than
        # free-space navigation.  Otherwise the straight final approach can
        # inherit several millimetres of lateral error and has no steering
        # authority left to remove it near contact.
        self._face_staging_pose_controller = replace(
            self._pose_controller,
            max_linear_speed_m_s=min(
                self._pose_controller.max_linear_speed_m_s,
                0.045,
            ),
            position_tolerance_m=0.002,
            yaw_tolerance_rad=face_alignment_staging_yaw_tolerance_rad,
        )
        self._desired_face_marker_gap_m = desired_face_marker_gap_m
        self._face_alignment_staging_distance_m = (
            face_alignment_staging_distance_m
        )
        self._face_alignment_staging_yaw_tolerance_rad = (
            face_alignment_staging_yaw_tolerance_rad
        )
        self._face_alignment_min_turn_speed_rad_s = (
            face_alignment_min_turn_speed_rad_s
        )
        self._staging_collision_avoidance = staging_collision_avoidance
        self._staging_center_clearance_m = staging_center_clearance_m
        self._staging_waypoint_margin_m = staging_waypoint_margin_m
        self._joint_tolerance_rad = joint_tolerance_rad
        self._tilt_joint_tolerance_rad = tilt_joint_tolerance_rad
        self._released_joint_groups: set[str] = set()
        self._completed_joint_groups: set[str] = set()
        # Only joints which have explicitly reached a PAN/TILT target are
        # retained. Every other module remains backdrivable.
        self._retained_internal_commands: dict[str, SmoresCommand] = {}
        self._pan_trackers = {
            module_id: ContinuousAngleTracker()
            for module_id in module_roots
        }
        self._active: dict[str, _ActiveGoal] = {}
        self._resource_owners: dict[str, dict[str, str]] = {}
        self._status: PrimitiveStatus | None = None

    @property
    def active_goal(self) -> PrimitiveGoal | None:
        """Compatibility accessor; prefer active_goals for concurrent use."""

        return next(
            (runtime.goal for runtime in self._active.values()),
            None,
        )

    @property
    def active_goals(self) -> tuple[PrimitiveGoal, ...]:
        return tuple(runtime.goal for runtime in self._active.values())

    @property
    def status(self) -> PrimitiveStatus | None:
        return self._status

    def compose_with_baseline(
        self,
        baseline: Mapping[str, SmoresCommand],
        primitive_commands: Mapping[str, SmoresCommand],
    ) -> dict[str, SmoresCommand]:
        """Overlay only resources currently owned by primitive goals.

        This lets, for example, ROS teleoperation keep driving the wheels
        while a concurrent action goal moves TILT.
        """

        if baseline:
            # Locomotion marks the assembled component as operational.  From
            # this point its unselected PAN/TILT joints are structural, not
            # assembly-time passive joints.  Isolated reserve modules are
            # deliberately excluded by the connection traversal.
            self._retain_structure_targets(
                self._connected_component_module_ids(baseline)
            )
        result = dict(baseline)
        for module_id in (
            set(baseline)
            | set(primitive_commands)
            | set(self._retained_internal_commands)
        ):
            base = baseline.get(module_id, SmoresCommand())
            primitive = primitive_commands.get(module_id, SmoresCommand())
            retained = self._retained_internal_commands.get(module_id)
            owns_locomotion = (
                f"locomotion:{module_id}" in self._resource_owners
            )
            owns_internal = (
                f"internal_motion:{module_id}" in self._resource_owners
            )
            internal_source = base
            if owns_internal:
                internal_source = primitive
            elif (
                base.internal_motion is InternalMotionMode.PAN_VELOCITY
                and retained is not None
            ):
                # PAN can remain a locomotor (the original RC-car profile)
                # while TILT keeps the captured operational structure.
                internal_source = SmoresCommand(
                    pan_target_rad=retained.pan_target_rad,
                    tilt_target_rad=retained.tilt_target_rad,
                    internal_motion=InternalMotionMode.PAN_VELOCITY,
                    pan_velocity_rad_s=base.pan_velocity_rad_s,
                )
            elif (
                base.internal_motion is InternalMotionMode.PASSIVE
                and retained is not None
            ):
                internal_source = retained
            result[module_id] = SmoresCommand(
                linear_x_m_s=(
                    primitive.linear_x_m_s
                    if owns_locomotion
                    else base.linear_x_m_s
                ),
                angular_z_rad_s=(
                    primitive.angular_z_rad_s
                    if owns_locomotion
                    else base.angular_z_rad_s
                ),
                pan_target_rad=(
                    internal_source.pan_target_rad
                ),
                tilt_target_rad=(
                    internal_source.tilt_target_rad
                ),
                internal_motion=(
                    internal_source.internal_motion
                ),
                pan_velocity_rad_s=(
                    internal_source.pan_velocity_rad_s
                ),
            )
        return result

    def _connected_component_module_ids(
        self,
        seed_module_ids: Mapping[str, Any],
    ) -> tuple[str, ...]:
        """Return only modules magnetically connected to locomotor seeds."""

        connected = {str(module_id) for module_id in seed_module_ids}
        edges: list[tuple[str, str]] = []
        for connection in getattr(self._docking, "connections", ()):
            first = str(connection.first_face.module_id)
            second = str(connection.second_face.module_id)
            edges.append((first, second))
        changed = True
        while changed:
            changed = False
            for first, second in edges:
                if first in connected and second not in connected:
                    connected.add(second)
                    changed = True
                elif second in connected and first not in connected:
                    connected.add(first)
                    changed = True
        return tuple(sorted(connected))

    def submit(self, goal: PrimitiveGoal, now_s: float) -> PrimitiveStatus:
        if goal.goal_id in self._active:
            return self._make_status(
                goal,
                PrimitiveState.REJECTED,
                now_s,
                phase="admission",
                code="DUPLICATE_GOAL_ID",
                message=f"goal_id {goal.goal_id} is already active",
            )
        referenced_modules = set(goal.module_ids)
        if goal.primitive in {PrimitiveName.SET_PAN, PrimitiveName.SET_TILT}:
            referenced_modules.update(
                str(item)
                for item in goal.parameters.get(
                    "structural_hold_module_ids", ()
                )
            )
        if goal.primitive is PrimitiveName.SET_TILT:
            pusher_module_id = goal.parameters.get("pusher_module_id")
            if pusher_module_id is not None:
                referenced_modules.add(str(pusher_module_id))
            referenced_modules.update(
                str(item)
                for item in goal.parameters.get(
                    "hold_after_group_module_ids", ()
                )
            )
            referenced_modules.update(
                str(item)
                for item in goal.parameters.get(
                    "stabilize_during_group_module_ids", ()
                )
            )
        unknown = sorted(referenced_modules - set(self._module_roots))
        if unknown:
            return self._make_status(
                goal,
                PrimitiveState.REJECTED,
                now_s,
                phase="admission",
                code="UNKNOWN_MODULE",
                message="unknown module(s): " + ", ".join(unknown),
            )
        controlled_module_id = (
            goal.module_ids[2]
            if goal.primitive is PrimitiveName.ASSISTED_ALIGN_FACES
            else goal.module_ids[0]
        )
        if (
            goal.primitive
            not in {PrimitiveName.DOCK, PrimitiveName.UNDOCK}
            and controlled_module_id not in self._motion_module_ids
        ):
            return self._make_status(
                goal,
                PrimitiveState.REJECTED,
                now_s,
                phase="admission",
                code="MODULE_NOT_CONTROLLED",
                message=(
                    f"module {controlled_module_id} has no enabled motion "
                    "controller in this scenario"
                ),
            )
        if (
            goal.primitive is PrimitiveName.SET_TILT
            and goal.parameters.get("pusher_module_id") is not None
            and str(goal.parameters["pusher_module_id"])
            not in self._motion_module_ids
        ):
            return self._make_status(
                goal,
                PrimitiveState.REJECTED,
                now_s,
                phase="admission",
                code="MODULE_NOT_CONTROLLED",
                message=(
                    "fold pusher module "
                    f"{goal.parameters['pusher_module_id']} has no enabled "
                    "motion controller in this scenario"
                ),
            )
        try:
            resolved_target = self._resolve_joint_target(goal)
            resource_modes = self._resources_for(goal)
            resources = frozenset(resource_modes)
        except (KeyError, RuntimeError, ValueError) as error:
            return self._make_status(
                goal,
                PrimitiveState.REJECTED,
                now_s,
                phase="admission",
                code="INVALID_GOAL",
                message=str(error),
            )
        conflicts: dict[str, tuple[str, ...]] = {}
        for resource, requested_mode in resource_modes.items():
            owners = self._resource_owners.get(resource, {})
            if not owners:
                continue
            has_exclusive_owner = any(
                mode == "exclusive" for mode in owners.values()
            )
            if requested_mode == "exclusive" or has_exclusive_owner:
                conflicts[resource] = tuple(sorted(owners))
        if conflicts:
            detail = ", ".join(
                f"{resource} owned by {', '.join(owners)}"
                for resource, owners in sorted(conflicts.items())
            )
            return self._make_status(
                goal,
                PrimitiveState.REJECTED,
                now_s,
                phase="admission",
                code="RESOURCE_BUSY",
                message=detail,
            )
        runtime = _ActiveGoal(
            goal=goal,
            started_at_s=now_s,
            resources=resources,
            resource_modes=dict(resource_modes),
            resolved_target_rad=resolved_target,
        )
        self._active[goal.goal_id] = runtime
        for resource, mode in resource_modes.items():
            self._resource_owners.setdefault(resource, {})[
                goal.goal_id
            ] = mode
        self._status = self._make_status(
            goal,
            PrimitiveState.ACCEPTED,
            now_s,
            phase="accepted",
            code="GOAL_ACCEPTED",
            message="primitive goal accepted",
        )
        return self._status

    def cancel(self, goal_id: str, now_s: float) -> PrimitiveStatus | None:
        runtime = self._active.get(goal_id)
        if runtime is None:
            return None
        return self._finish(
            runtime,
            PrimitiveState.CANCELED,
            now_s,
            code="CANCELED_BY_CLIENT",
            message="primitive goal canceled",
        )

    def step(self, now_s: float) -> PrimitiveExecutorStep:
        merged: dict[str, SmoresCommand] = {}
        statuses: list[PrimitiveStatus] = []
        for goal_id in tuple(self._active):
            runtime = self._active.get(goal_id)
            if runtime is None:
                continue
            goal = runtime.goal
            if now_s - runtime.started_at_s > goal.timeout_s:
                contact_timeout = (
                    goal.primitive is PrimitiveName.ALIGN_FACES
                    and runtime.alignment_approach_started
                )
                statuses.append(
                    self._finish(
                        runtime,
                        PrimitiveState.FAILED,
                        now_s,
                        code=(
                            "CONTACT_TIMEOUT" if contact_timeout else "TIMEOUT"
                        ),
                        message=(
                            "contact approach timed out; alignment will not "
                            "be restarted"
                            if contact_timeout
                            else f"goal exceeded {goal.timeout_s:.3f}s"
                        ),
                    )
                )
                continue
            try:
                commands, status = self._execute(runtime, now_s)
            except (KeyError, RuntimeError, ValueError) as error:
                statuses.append(
                    self._finish(
                        runtime,
                        PrimitiveState.FAILED,
                        now_s,
                        code="EXECUTION_ERROR",
                        message=str(error),
                    )
                )
                continue
            statuses.append(status)
            self._merge_commands(merged, commands, runtime.resources)
        if statuses:
            self._status = statuses[-1]
        return PrimitiveExecutorStep(merged, tuple(statuses))

    def _execute(
        self,
        runtime: _ActiveGoal,
        now_s: float,
    ) -> tuple[dict[str, SmoresCommand], PrimitiveStatus]:
        goal = runtime.goal
        if goal.primitive is PrimitiveName.DRIVE_TO_POSE:
            return self._drive_to_pose(runtime, now_s)
        if goal.primitive is PrimitiveName.ALIGN_FACES:
            return self._align_faces(runtime, now_s)
        if goal.primitive is PrimitiveName.ASSISTED_ALIGN_FACES:
            return self._assisted_align_faces(runtime, now_s)
        if goal.primitive in {PrimitiveName.DOCK, PrimitiveName.UNDOCK}:
            return self._change_docking(runtime, now_s)
        return self._move_joint(runtime, now_s)

    def _drive_to_pose(
        self,
        runtime: _ActiveGoal,
        now_s: float,
    ) -> tuple[dict[str, SmoresCommand], PrimitiveStatus]:
        goal = runtime.goal
        module_id = goal.module_ids[0]
        current = self._planar_pose(module_id)
        target = PlanarPose(
            float(goal.parameters["x_m"]),
            float(goal.parameters["y_m"]),
            float(goal.parameters["yaw_rad"]),
        )
        step = drive_to_pose_step(current, target, self._pose_controller)
        if step.done:
            return {}, self._finish(
                runtime,
                PrimitiveState.SUCCEEDED,
                now_s,
                code="POSE_REACHED",
                message="target pose reached",
            )
        if runtime.initial_metric is None:
            runtime.initial_metric = max(
                step.position_error_m,
                self._pose_controller.position_tolerance_m,
            )
        initial_distance = runtime.initial_metric
        progress = 1.0 - min(1.0, step.position_error_m / initial_distance)
        command = SmoresCommand(
            linear_x_m_s=step.linear_x_m_s,
            angular_z_rad_s=step.angular_z_rad_s,
        )
        return {module_id: command}, self._make_status(
            goal,
            PrimitiveState.RUNNING,
            now_s,
            phase=step.phase,
            progress=progress,
            code="DRIVING",
            message="driving differential base to target pose",
            feedback={
                "position_error_m": step.position_error_m,
                "yaw_error_rad": step.yaw_error_rad,
                "current_pose": [
                    current.x_m,
                    current.y_m,
                    current.yaw_rad,
                ],
                "target_pose": [target.x_m, target.y_m, target.yaw_rad],
            },
        )

    def _align_faces(
        self,
        runtime: _ActiveGoal,
        now_s: float,
    ) -> tuple[dict[str, SmoresCommand], PrimitiveStatus]:
        goal = runtime.goal
        mobile_id, target_id = goal.module_ids
        face_a = str(goal.parameters["face_a"])
        face_b = str(goal.parameters["face_b"])
        execution_phase = str(
            goal.parameters.get("execution_phase", "full")
        ).lower()
        first = self._face_pose(mobile_id, face_a)
        second = self._face_pose(target_id, face_b)
        docking_thresholds = self._docking_thresholds_for_goal(goal)
        evaluation = evaluate_face_pair(
            first,
            second,
            docking_thresholds,
        )
        top_module_id = (
            mobile_id
            if face_a == "TOP"
            else target_id
            if face_b == "TOP"
            else None
        )
        if (
            execution_phase in {"full", "align"}
            and top_module_id is not None
            and evaluation.normal_misalignment_rad
            <= docking_thresholds.normal_alignment_tolerance_rad
            and evaluation.clocking_error_rad
            > docking_thresholds.clocking_tolerance_rad
        ):
            current_pan, current_tilt = self._joint_positions(top_module_id)
            pan_target = current_pan + evaluation.clocking_residual_rad
            return {
                top_module_id: SmoresCommand(
                    pan_target_rad=pan_target,
                    tilt_target_rad=current_tilt,
                    internal_motion=InternalMotionMode.PAN,
                )
            }, self._make_status(
                goal,
                PrimitiveState.RUNNING,
                now_s,
                phase="face_clocking",
                progress=0.0,
                code="ALIGNING_CLOCKING",
                message=(
                    f"rotating {top_module_id}:TOP to align EP-Face clocking"
                ),
                feedback={
                    "clocking_error_rad": evaluation.clocking_error_rad,
                    "clocking_residual_rad": (
                        evaluation.clocking_residual_rad
                    ),
                    "pan_target_rad": pan_target,
                },
            )
        current = self._planar_pose(mobile_id)
        final_target = self._face_alignment_target(
            mobile_id,
            face_a,
            target_id,
            face_b,
        )
        staging_target = self._face_alignment_staging_target(
            final_target,
            target_id,
            face_b,
            distance_m=self._staging_distance_for_goal(goal),
        )
        if execution_phase == "reach":
            return self._reach_face_staging(
                runtime,
                now_s,
                mobile_id,
                current,
                staging_target,
            )
        if execution_phase == "align":
            return self._adjust_face_pose(
                runtime,
                now_s,
                mobile_id,
                face_a,
                target_id,
                face_b,
                current,
                staging_target,
            )
        if execution_phase == "approach":
            # REACH and ALIGN were completed collectively by the whole wave.
            # This goal therefore starts directly with the signed straight
            # approach: BOTTOM-face movers back up while TOP-face movers move
            # forward, both preserving the docking centreline.
            runtime.alignment_approach_started = True
        planar_lateral_error_m = self._planar_face_lateral_offset(
            first,
            second,
        )
        quality_tolerance = goal.parameters.get(
            "contact_quality_planar_tolerance_m"
        )
        quality_retry_limit = int(
            goal.parameters.get("contact_quality_retry_count", 0)
        )
        quality_recovery_started = False

        # Once the module has backed out and begun the next straight
        # approach, the new contact may be assessed again.  While it is still
        # physically touching, keep the recovery active so it cannot be
        # mistaken for an immediately successful retry.
        if (
            runtime.contact_quality_recovery_active
            and runtime.alignment_approach_started
            and not evaluation.eligible
        ):
            runtime.contact_quality_recovery_active = False

        alignment_complete = self._face_alignment_complete(evaluation)
        needs_better_parking = (
            quality_tolerance is not None
            and planar_lateral_error_m > float(quality_tolerance)
        )
        # A loaded TOP<->BOTTOM pair can reach its CAD collision stop a few
        # millimetres before the strict docking gate.  Waiting for
        # ``evaluation.eligible`` before starting the parking recovery makes
        # that recovery unreachable: the differential drive keeps pushing
        # straight into the blocked component until the primitive times out.
        #
        # Enter recovery while the pair is already proximal and the remaining
        # planar error is known to be uncorrectable by a straight approach.
        # The shared assembly policy controls both the quality tolerance and
        # the bounded retry count for every target morphology.
        proximal_quality_failure = (
            runtime.alignment_approach_started
            and not alignment_complete
            and needs_better_parking
            and evaluation.normal_separation_m <= max(
                0.010,
                2.0 * docking_thresholds.top_bottom_contact_tolerance_m,
            )
        )
        if alignment_complete or proximal_quality_failure:
            if runtime.contact_quality_recovery_active:
                # Continue the commanded back-off before accepting the same
                # still-visible contact on the following simulation frame.
                pass
            elif (
                needs_better_parking
                and runtime.alignment_recovery_count < quality_retry_limit
            ):
                runtime.alignment_approach_started = False
                runtime.alignment_staging_position_reached = False
                runtime.alignment_staging_drive_direction = None
                runtime.initial_metric = None
                runtime.axial_alignment_mode = "curve"
                runtime.axial_pivot_sample_started_s = None
                runtime.axial_pivot_sample_yaw_rad = None
                runtime.axial_escape_until_s = None
                runtime.collision_route_xy = ()
                runtime.collision_route_goal_xy = None
                runtime.alignment_recovery_count += 1
                runtime.contact_quality_recovery_active = True
                quality_recovery_started = True
            elif alignment_complete:
                return {}, self._finish(
                    runtime,
                    PrimitiveState.SUCCEEDED,
                    now_s,
                    code="FACES_IN_CONTACT",
                    message=(
                        f"{mobile_id}:{face_a} is aligned and in contact with "
                        f"{target_id}:{face_b}"
                    ),
                )

        if not runtime.alignment_approach_started:
            if runtime.alignment_recovery_count:
                staging_target = self._face_alignment_staging_target(
                    final_target,
                    target_id,
                    face_b,
                    distance_m=min(
                        self._face_alignment_staging_distance_m,
                        0.040,
                    ),
                )
            waypoint, avoidance_active = self._collision_aware_staging_waypoint(
                runtime,
                mobile_id,
                current,
                staging_target,
            )
            if waypoint is None:
                return {}, self._finish(
                    runtime,
                    PrimitiveState.FAILED,
                    now_s,
                    code="NO_COLLISION_FREE_STAGING_PATH",
                    message=(
                        f"no collision-free planar route found for {mobile_id} "
                        "to its face-alignment staging pose"
                    ),
                )
            target = waypoint
            axial_mobile_face = face_a in {"TOP", "BOTTOM"}
            if runtime.contact_quality_recovery_active:
                # The connector is already proximal. Back away along the
                # same corridor while preserving the centreline with small
                # steering corrections; do not pivot in the confined gap.
                target = staging_target
                step = self._straight_face_approach_step(
                    current,
                    staging_target,
                    feedback_enabled=True,
                )
                phase_prefix = "quality_backoff"
                staging_ready = step.done
            elif avoidance_active:
                runtime.alignment_staging_position_reached = False
                runtime.alignment_staging_drive_direction = None
                step = self._staging_pose_step(current, waypoint)
                phase_prefix = "collision_avoidance"
                staging_ready = False
            elif axial_mobile_face:
                step, phase_prefix, staging_ready = (
                    self._axial_face_staging_step(
                        runtime,
                        current,
                        staging_target,
                        now_s,
                    )
                )
            else:
                step, phase_prefix, staging_ready = (
                    self._alignment_staging_step(
                        runtime,
                        current,
                        staging_target,
                    )
                )
            if staging_ready:
                runtime.alignment_approach_started = True
                runtime.initial_metric = None
                target = final_target
                step = self._straight_face_approach_step(
                    current,
                    target,
                    feedback_enabled=bool(
                        goal.parameters.get(
                            "contact_approach_feedback",
                            False,
                        )
                    ),
                )
                progress_offset = 0.5
                progress_scale = 0.5
                phase_prefix = "contact"
            else:
                progress_offset = 0.0
                progress_scale = 0.5
        else:
            target = final_target
            step = self._straight_face_approach_step(
                current,
                target,
                feedback_enabled=bool(
                    goal.parameters.get(
                        "contact_approach_feedback",
                        False,
                    )
                ),
            )
            progress_offset = 0.5
            progress_scale = 0.5
            phase_prefix = "contact"
        if phase_prefix == "contact" and step.done:
            # Never leave an ALIGN_FACES primitive running with a zero wheel
            # command.  At this point the monotonic straight approach has
            # reached its longitudinal target, so an ineligible pair cannot
            # be improved without violating the no-restaging contact policy.
            return {}, self._finish(
                runtime,
                PrimitiveState.FAILED,
                now_s,
                code="CONTACT_POSE_INVALID",
                message=(
                    "straight contact approach completed outside the shared "
                    "docking gate: "
                    f"normal_gap={1e3*evaluation.normal_separation_m:.2f}mm "
                    f"lateral={1e3*evaluation.lateral_offset_m:.2f}mm "
                    "normal_error="
                    f"{math.degrees(evaluation.normal_misalignment_rad):.1f}deg "
                    "clocking_error="
                    f"{math.degrees(evaluation.clocking_error_rad):.1f}deg"
                ),
            )
        command = SmoresCommand(
            linear_x_m_s=step.linear_x_m_s,
            angular_z_rad_s=step.angular_z_rad_s,
        )
        target_distance = math.hypot(
            target.x_m - current.x_m,
            target.y_m - current.y_m,
        )
        if runtime.initial_metric is None:
            runtime.initial_metric = max(
                target_distance,
                self._pose_controller.position_tolerance_m,
            )
        progress = progress_offset + progress_scale * (
            1.0
            - min(
                1.0,
                target_distance / runtime.initial_metric,
            )
        )
        return {mobile_id: command}, self._make_status(
            goal,
            PrimitiveState.RUNNING,
            now_s,
            phase=(
                f"contact_{step.phase}"
                if phase_prefix == "contact"
                else f"alignment_{phase_prefix}_{step.phase}"
            ),
            progress=progress,
            code=(
                "REPOSITIONING_CONTACT"
                if runtime.contact_quality_recovery_active
                else "CLOSING_CONTACT"
                if phase_prefix == "contact"
                else "ALIGNING"
            ),
            message=(
                "backing off for a more accurate connector parking retry"
                if runtime.contact_quality_recovery_active
                else "moving to collision-free face-alignment staging pose"
                if phase_prefix == "staging"
                else "following a collision-free staging waypoint"
                if phase_prefix == "collision_avoidance"
                else "backing away with bounded centreline feedback"
                if phase_prefix == "quality_backoff"
                else "orienting the connector at the staging pose"
                if phase_prefix == "staging_yaw"
                else "correcting axial-face lateral error and yaw together"
                if phase_prefix == "axial_pose_adjustment"
                else "closing the face gap without returning to alignment"
            ),
            feedback={
                "normal_gap_m": evaluation.normal_separation_m,
                "lateral_error_m": evaluation.lateral_offset_m,
                "planar_lateral_error_m": planar_lateral_error_m,
                "normal_error_rad": evaluation.normal_misalignment_rad,
                "clocking_error_rad": evaluation.clocking_error_rad,
                "alignment_subphase": phase_prefix,
                "alignment_recovery_count": runtime.alignment_recovery_count,
                "contact_quality_recovery_active": (
                    runtime.contact_quality_recovery_active
                ),
                "contact_quality_recovery_started": quality_recovery_started,
                "contact_quality_planar_tolerance_m": quality_tolerance,
                "collision_avoidance_active": (
                    phase_prefix == "collision_avoidance"
                ),
                "collision_route_replans": runtime.collision_route_replans,
                "collision_route_waypoints_remaining": len(
                    runtime.collision_route_xy
                ),
                "axial_alignment_mode": runtime.axial_alignment_mode,
                "axial_stall_recovery_count": (
                    runtime.axial_stall_recovery_count
                ),
                "staging_position_error_m": target_distance,
                "staging_yaw_error_rad": step.yaw_error_rad,
                "command_linear_m_s": step.linear_x_m_s,
                "command_angular_rad_s": step.angular_z_rad_s,
                "current_pose": [
                    current.x_m,
                    current.y_m,
                    current.yaw_rad,
                ],
                "target_pose": [target.x_m, target.y_m, target.yaw_rad],
                "final_target_pose": [
                    final_target.x_m,
                    final_target.y_m,
                    final_target.yaw_rad,
                ],
            },
        )

    def _reach_face_staging(
        self,
        runtime: _ActiveGoal,
        now_s: float,
        mobile_id: str,
        current: PlanarPose,
        staging_target: PlanarPose,
    ) -> tuple[dict[str, SmoresCommand], PrimitiveStatus]:
        """Execute only the paper's collision-free navigation phase."""

        goal = runtime.goal
        fallback_level = self._staging_path_fallback_level(goal)
        clearance_m = self._staging_clearance_for_goal(goal)
        waypoint, avoidance_active = self._collision_aware_staging_waypoint(
            runtime,
            mobile_id,
            current,
            staging_target,
        )
        if waypoint is None:
            return {}, self._finish(
                runtime,
                PrimitiveState.FAILED,
                now_s,
                code="NO_COLLISION_FREE_STAGING_PATH",
                message=(
                    f"no collision-free planar route found for {mobile_id} "
                    "to its face-alignment staging pose "
                    f"(fallback level {fallback_level}, "
                    f"clearance {clearance_m:.3f} m)"
                ),
            )

        remaining_m = math.hypot(
            staging_target.x_m - current.x_m,
            staging_target.y_m - current.y_m,
        )
        # Pose adjustment owns the final local 25 mm. REACH deliberately
        # stops outside that region so no participant can start approaching
        # while a peer is still navigating through free space.
        if not avoidance_active and remaining_m <= 0.025:
            return {}, self._finish(
                runtime,
                PrimitiveState.SUCCEEDED,
                now_s,
                code="STAGING_REACHED",
                message=f"{mobile_id} reached the local docking region",
            )

        step = self._staging_pose_step(current, waypoint)
        if runtime.initial_metric is None:
            runtime.initial_metric = max(
                remaining_m,
                self._pose_controller.position_tolerance_m,
            )
        progress = 1.0 - min(
            1.0,
            remaining_m / runtime.initial_metric,
        )
        return {
            mobile_id: SmoresCommand(
                linear_x_m_s=step.linear_x_m_s,
                angular_z_rad_s=step.angular_z_rad_s,
            )
        }, self._make_status(
            goal,
            PrimitiveState.RUNNING,
            now_s,
            phase=("collision_avoidance" if avoidance_active else "reach"),
            progress=progress,
            code="REACHING_STAGING",
            message="following the collision-free route to docking staging",
            feedback={
                "staging_distance_m": remaining_m,
                "staging_path_fallback_level": fallback_level,
                "staging_center_clearance_m": clearance_m,
            },
        )

    def _adjust_face_pose(
        self,
        runtime: _ActiveGoal,
        now_s: float,
        mobile_id: str,
        mobile_face: str,
        target_id: str,
        target_face: str,
        current: PlanarPose,
        staging_target: PlanarPose,
    ) -> tuple[dict[str, SmoresCommand], PrimitiveStatus]:
        """Execute only pose adjustment, without entering contact approach."""

        if mobile_face in {"TOP", "BOTTOM"}:
            step, phase, ready = self._axial_face_staging_step(
                runtime,
                current,
                staging_target,
                now_s,
            )
        else:
            step, phase, ready = self._alignment_staging_step(
                runtime,
                current,
                staging_target,
            )
        if ready:
            return {}, self._finish(
                runtime,
                PrimitiveState.SUCCEEDED,
                now_s,
                code="FACES_ALIGNED",
                message=(
                    f"{mobile_id}:{mobile_face} is aligned with "
                    f"{target_id}:{target_face} and ready to approach"
                ),
            )
        return {
            mobile_id: SmoresCommand(
                linear_x_m_s=step.linear_x_m_s,
                angular_z_rad_s=step.angular_z_rad_s,
            )
        }, self._make_status(
            runtime.goal,
            PrimitiveState.RUNNING,
            now_s,
            phase=phase,
            progress=0.0,
            code="ADJUSTING_FACE_POSE",
            message="adjusting lateral position and yaw on the docking line",
            feedback={
                "position_error_m": step.position_error_m,
                "yaw_error_rad": step.yaw_error_rad,
            },
        )

    def _assisted_align_faces(
        self,
        runtime: _ActiveGoal,
        now_s: float,
    ) -> tuple[dict[str, SmoresCommand], PrimitiveStatus]:
        """Align a payload by driving a helper rigidly docked to it.

        The goal names ``(payload, target, helper)``.  The desired planar
        transform is first computed for the payload connector, then rebased
        onto the helper while preserving their measured rigid transform.
        This is the simulated equivalent of the push phase in Fig. 5 of the
        SMORES-EP parallel self-assembly paper.
        """

        goal = runtime.goal
        payload_id, target_id, helper_id = goal.module_ids
        face_a = str(goal.parameters["face_a"])
        face_b = str(goal.parameters["face_b"])
        payload_face = self._face_pose(payload_id, face_a)
        target_face = self._face_pose(target_id, face_b)
        evaluation = evaluate_face_pair(
            payload_face,
            target_face,
            self._docking_thresholds_for_goal(goal),
        )
        if self._face_alignment_complete(evaluation):
            return {}, self._finish(
                runtime,
                PrimitiveState.SUCCEEDED,
                now_s,
                code="FACES_ALIGNED_BY_HELPER",
                message=(
                    f"{helper_id} aligned {payload_id}:{face_a} with "
                    f"{target_id}:{face_b}"
                ),
            )

        payload_current = self._planar_pose(payload_id)
        helper_current = self._planar_pose(helper_id)
        payload_final = self._face_alignment_target(
            payload_id,
            face_a,
            target_id,
            face_b,
        )
        payload_staging = self._face_alignment_staging_target(
            payload_final,
            target_id,
            face_b,
        )
        helper_final = self._rigid_driver_target(
            payload_current,
            helper_current,
            payload_final,
        )
        helper_staging = self._rigid_driver_target(
            payload_current,
            helper_current,
            payload_staging,
        )

        if not runtime.alignment_approach_started:
            if runtime.alignment_recovery_count:
                payload_staging = self._face_alignment_staging_target(
                    payload_final,
                    target_id,
                    face_b,
                    distance_m=min(
                        self._face_alignment_staging_distance_m,
                        0.040,
                    ),
                )
                helper_staging = self._rigid_driver_target(
                    payload_current,
                    helper_current,
                    payload_staging,
                )
            staging_distance_m = math.hypot(
                helper_staging.x_m - helper_current.x_m,
                helper_staging.y_m - helper_current.y_m,
            )
            target = helper_staging
            step, phase_prefix, staging_ready = self._alignment_staging_step(
                runtime,
                helper_current,
                helper_staging,
            )
            if staging_ready:
                runtime.alignment_approach_started = True
                runtime.initial_metric = None
                target = helper_final
                step = self._straight_face_approach_step(
                    helper_current,
                    target,
                )
                progress_offset = 0.5
                progress_scale = 0.5
                phase_prefix = "approach"
            else:
                progress_offset = 0.0
                progress_scale = 0.5
        else:
            target = helper_final
            step = self._straight_face_approach_step(
                helper_current,
                target,
            )
            progress_offset = 0.5
            progress_scale = 0.5
            phase_prefix = "approach"

        residual_contact_error = self._has_residual_contact_error(
            evaluation,
            self._docking_thresholds_for_goal(goal),
        )
        if phase_prefix == "approach" and (
            step.done or residual_contact_error
        ):
            runtime.alignment_approach_started = False
            runtime.alignment_staging_position_reached = False
            runtime.initial_metric = None
            runtime.alignment_recovery_count += 1
            target = helper_staging
            step = self._staging_pose_step(
                helper_current,
                target,
            )
            progress_offset = 0.0
            progress_scale = 0.5
            phase_prefix = "restaging"

        target_distance = math.hypot(
            target.x_m - helper_current.x_m,
            target.y_m - helper_current.y_m,
        )
        if runtime.initial_metric is None:
            runtime.initial_metric = max(
                target_distance,
                self._pose_controller.position_tolerance_m,
            )
        progress = progress_offset + progress_scale * (
            1.0 - min(1.0, target_distance / runtime.initial_metric)
        )
        command = SmoresCommand(
            linear_x_m_s=step.linear_x_m_s,
            angular_z_rad_s=step.angular_z_rad_s,
        )
        return {helper_id: command}, self._make_status(
            goal,
            PrimitiveState.RUNNING,
            now_s,
            phase=f"assisted_align_{phase_prefix}_{step.phase}",
            progress=progress,
            code="HELPER_PUSHING",
            message=(
                f"{helper_id} is pushing {payload_id} toward {target_id}"
            ),
            feedback={
                "normal_gap_m": evaluation.normal_separation_m,
                "lateral_error_m": evaluation.lateral_offset_m,
                "normal_error_rad": evaluation.normal_misalignment_rad,
                "clocking_error_rad": evaluation.clocking_error_rad,
                "alignment_subphase": phase_prefix,
                "alignment_recovery_count": runtime.alignment_recovery_count,
                "staging_position_error_m": target_distance,
                "staging_yaw_error_rad": step.yaw_error_rad,
                "command_linear_m_s": step.linear_x_m_s,
                "command_angular_rad_s": step.angular_z_rad_s,
                "helper_target_pose": [
                    target.x_m,
                    target.y_m,
                    target.yaw_rad,
                ],
                "payload_final_target_pose": [
                    payload_final.x_m,
                    payload_final.y_m,
                    payload_final.yaw_rad,
                ],
            },
        )

    @staticmethod
    def _rigid_driver_target(
        payload_current: PlanarPose,
        driver_current: PlanarPose,
        payload_target: PlanarPose,
    ) -> PlanarPose:
        """Apply a desired payload transform to its rigidly attached driver."""

        dx = driver_current.x_m - payload_current.x_m
        dy = driver_current.y_m - payload_current.y_m
        cosine = math.cos(-payload_current.yaw_rad)
        sine = math.sin(-payload_current.yaw_rad)
        local_x = cosine * dx - sine * dy
        local_y = sine * dx + cosine * dy
        relative_yaw = wrap_angle(
            driver_current.yaw_rad - payload_current.yaw_rad
        )
        cosine = math.cos(payload_target.yaw_rad)
        sine = math.sin(payload_target.yaw_rad)
        return PlanarPose(
            payload_target.x_m + cosine * local_x - sine * local_y,
            payload_target.y_m + sine * local_x + cosine * local_y,
            wrap_angle(payload_target.yaw_rad + relative_yaw),
        )

    def _straight_face_approach_step(
        self,
        current: PlanarPose,
        target: PlanarPose,
        *,
        feedback_enabled: bool = False,
    ) -> PoseControlStep:
        """Drive the last connector gap, optionally with bounded feedback.

        The paper-inspired mode starts from a merely proximal aligned pose
        and corrects lateral/yaw error while the module rolls towards the
        connector.  In particular, a reverse approach makes the small
        left/right corrections *while backing up*, rather than attempting an
        in-place turn where there is no manoeuvring room.  The correction is
        deliberately limited to six degrees and 0.35 rad/s near contact.

        The legacy straight mode remains available so established
        morphologies are unaffected unless their target graph opts in.
        """

        limits = self._face_approach_pose_controller
        dx_m = target.x_m - current.x_m
        dy_m = target.y_m - current.y_m
        forward_error_m = (
            dx_m * math.cos(target.yaw_rad)
            + dy_m * math.sin(target.yaw_rad)
        )
        yaw_error_rad = wrap_angle(target.yaw_rad - current.yaw_rad)
        if abs(forward_error_m) <= limits.position_tolerance_m:
            return PoseControlStep(
                linear_x_m_s=0.0,
                angular_z_rad_s=0.0,
                position_error_m=abs(forward_error_m),
                yaw_error_rad=yaw_error_rad,
                phase="straight_approach_complete",
                done=True,
            )

        speed_m_s = min(
            limits.max_linear_speed_m_s,
            max(
                0.018,
                limits.linear_gain_s * abs(forward_error_m),
            ),
        )
        linear_x_m_s = math.copysign(speed_m_s, forward_error_m)
        angular_z_rad_s = 0.0
        phase = "straight_approach"
        if feedback_enabled:
            cosine = math.cos(target.yaw_rad)
            sine = math.sin(target.yaw_rad)
            world_error_x = current.x_m - target.x_m
            world_error_y = current.y_m - target.y_m
            lateral_error_m = (
                -sine * world_error_x + cosine * world_error_y
            )
            reference = axial_pose_adjustment_reference(
                lateral_error_m,
                math.copysign(1.0, linear_x_m_s),
                translation_speed_m_s=abs(linear_x_m_s),
                lateral_gain_s=1.5,
                maximum_relative_yaw_rad=math.radians(6.0),
            )
            current_relative_yaw_rad = wrap_angle(
                current.yaw_rad - target.yaw_rad
            )
            steering_error_rad = wrap_angle(
                reference.desired_relative_yaw_rad
                - current_relative_yaw_rad
            )
            angular_z_rad_s = max(
                -0.35,
                min(0.35, 2.5 * steering_error_rad),
            )
            phase = "feedback_approach"

        return PoseControlStep(
            linear_x_m_s=linear_x_m_s,
            angular_z_rad_s=angular_z_rad_s,
            position_error_m=abs(forward_error_m),
            yaw_error_rad=yaw_error_rad,
            phase=phase,
            done=False,
        )

    def _staging_pose_step(
        self,
        current: PlanarPose,
        target: PlanarPose,
    ) -> PoseControlStep:
        """Drive a staging correction above the CAD static-friction floor."""

        step = drive_to_pose_step(
            current,
            target,
            self._face_staging_pose_controller,
        )
        if step.done:
            return step
        if (
            step.phase == "translate"
            and 0.0 < abs(step.linear_x_m_s) < 0.018
        ):
            step = replace(
                step,
                linear_x_m_s=math.copysign(0.018, step.linear_x_m_s),
            )
        if (
            step.phase in {"orient_to_path", "final_yaw"}
            and 0.0 < abs(step.angular_z_rad_s)
            < self._face_alignment_min_turn_speed_rad_s
        ):
            step = replace(
                step,
                angular_z_rad_s=math.copysign(
                    self._face_alignment_min_turn_speed_rad_s,
                    step.angular_z_rad_s,
                ),
            )
        return step

    def _collision_aware_staging_waypoint(
        self,
        runtime: _ActiveGoal,
        mobile_id: str,
        current: PlanarPose,
        staging_target: PlanarPose,
    ) -> tuple[PlanarPose | None, bool]:
        """Return the next safe free-space waypoint before local alignment.

        This planner is deliberately used only before the staging pose.  The
        subsequent straight contact approach remains unchanged, so intended
        connector contact is never interpreted as an obstacle.
        """

        if not self._staging_collision_avoidance:
            return staging_target, False

        clearance_m = self._staging_clearance_for_goal(runtime.goal)
        waypoint_margin_m = self._staging_margin_for_goal(runtime.goal)
        obstacles = tuple(
            CircularObstacle(
                module_id=module_id,
                center_xy=(pose.x_m, pose.y_m),
                clearance_m=clearance_m,
            )
            for module_id in sorted(self._module_roots)
            if module_id != mobile_id
            for pose in (self._planar_pose(module_id),)
        )
        current_xy = (current.x_m, current.y_m)
        goal_xy = (staging_target.x_m, staging_target.y_m)

        # Discard a route when its moving target changed or any segment was
        # invalidated by another concurrently moving module.
        route = runtime.collision_route_xy
        if (
            runtime.collision_route_goal_xy is None
            or math.hypot(
                goal_xy[0] - runtime.collision_route_goal_xy[0],
                goal_xy[1] - runtime.collision_route_goal_xy[1],
            )
            > 0.005
        ):
            route = ()
        if route:
            route_points = (current_xy,) + route
            route_is_clear = all(
                segment_is_clear(
                    route_points[index],
                    route_points[index + 1],
                    obstacles,
                    allow_start_inside=(index == 0),
                )
                for index in range(len(route_points) - 1)
            )
            if not route_is_clear:
                route = ()

        # Once the next segment has opened, remove unnecessary detours. This
        # also naturally advances a ring waypoint without a stop-and-turn.
        if segment_is_clear(
            current_xy,
            goal_xy,
            obstacles,
            allow_start_inside=True,
        ):
            route = (goal_xy,)
        else:
            while (
                len(route) > 1
                and math.hypot(
                    route[0][0] - current.x_m,
                    route[0][1] - current.y_m,
                )
                <= 0.018
            ):
                route = route[1:]

        if not route:
            planned = None
            # Sixteen ring samples are normally sufficient and keep the
            # visibility graph small.  Tightly packed reconfiguration starts
            # can leave narrow exits between overlapping inflated module
            # footprints, so progressively densify the same safe geometric
            # model before declaring that no route exists.
            for angular_samples in (16, 32, 64):
                planned = plan_collision_aware_path(
                    current_xy,
                    goal_xy,
                    obstacles,
                    waypoint_margin_m=waypoint_margin_m,
                    angular_samples=angular_samples,
                )
                runtime.collision_route_replans += 1
                if planned is not None:
                    break
            if planned is None:
                runtime.collision_route_xy = ()
                runtime.collision_route_goal_xy = goal_xy
                return None, False
            route = planned

        runtime.collision_route_xy = route
        runtime.collision_route_goal_xy = goal_xy
        if len(route) == 1:
            return staging_target, False

        waypoint_xy = route[0]
        next_xy = route[1]
        waypoint_yaw = math.atan2(
            next_xy[1] - waypoint_xy[1],
            next_xy[0] - waypoint_xy[0],
        )
        return PlanarPose(
            waypoint_xy[0],
            waypoint_xy[1],
            waypoint_yaw,
        ), True

    def _alignment_staging_step(
        self,
        runtime: _ActiveGoal,
        current: PlanarPose,
        target: PlanarPose,
    ) -> tuple[PoseControlStep, str, bool]:
        """Reach staging position, then rotate only toward connector yaw.

        Navigation toward a pose is deliberately disabled inside the capture
        radius.  Otherwise a residual displacement of only a few millimetres
        can make the non-holonomic controller turn toward the displacement
        bearing, which may be opposite to the connector's required yaw.
        """

        distance_m = math.hypot(
            target.x_m - current.x_m,
            target.y_m - current.y_m,
        )
        capture_tolerance_m = 0.005
        release_tolerance_m = 0.008
        if (
            not runtime.alignment_staging_position_reached
            and distance_m <= capture_tolerance_m
        ):
            runtime.alignment_staging_position_reached = True

        if runtime.alignment_staging_position_reached:
            if distance_m > 0.015:
                runtime.alignment_staging_position_reached = False
            else:
                yaw_target = PlanarPose(
                    current.x_m,
                    current.y_m,
                    target.yaw_rad,
                )
                step = self._staging_pose_step(current, yaw_target)
                if step.done:
                    if distance_m <= release_tolerance_m:
                        return step, "staging_yaw", True
                    runtime.alignment_staging_position_reached = False
                else:
                    return step, "staging_yaw", False

        return self._staging_pose_step(current, target), "staging", False

    def _axial_face_staging_step(
        self,
        runtime: _ActiveGoal,
        current: PlanarPose,
        target: PlanarPose,
        now_s: float | None = None,
    ) -> tuple[PoseControlStep, str, bool]:
        """Align a mobile TOP or BOTTOM face with curve/pivot hysteresis.

        In the target-pose frame, axial-face docking needs the lateral
        coordinate and body yaw to converge before the final straight
        approach. The module first follows a correction arc, then retains the
        physically valid differential-drive pivot once it is close enough to
        the centerline. Separate enter/release thresholds prevent millimetric
        PhysX noise from switching between the two commands every frame.

        A pivot that produces no measured body-yaw progress for 0.8 s is
        released with a short, curvature-bounded arc. Both wheels still roll
        in the same longitudinal direction during that escape, after which
        the local curve/pivot controller continues; global staging is never
        restarted.
        """

        # Keep direct unit-test callers and old serialized runtimes compatible
        # while the real executor uses the explicit _ActiveGoal fields above.
        if not hasattr(runtime, "axial_alignment_mode"):
            runtime.axial_alignment_mode = "curve"
        if not hasattr(runtime, "axial_pivot_sample_started_s"):
            runtime.axial_pivot_sample_started_s = None
        if not hasattr(runtime, "axial_pivot_sample_yaw_rad"):
            runtime.axial_pivot_sample_yaw_rad = None
        if not hasattr(runtime, "axial_escape_until_s"):
            runtime.axial_escape_until_s = None
        if not hasattr(runtime, "axial_stall_recovery_count"):
            runtime.axial_stall_recovery_count = 0

        cosine = math.cos(target.yaw_rad)
        sine = math.sin(target.yaw_rad)
        world_dx = current.x_m - target.x_m
        world_dy = current.y_m - target.y_m
        longitudinal_error_m = cosine * world_dx + sine * world_dy
        lateral_error_m = -sine * world_dx + cosine * world_dy
        yaw_error_rad = wrap_angle(current.yaw_rad - target.yaw_rad)
        distance_m = math.hypot(
            longitudinal_error_m,
            lateral_error_m,
        )

        # Free-space navigation is still preferable outside the local pose
        # adjustment region.  Once captured, keep the direction fixed; a
        # sign flip near the staging point was another source of oscillation.
        if runtime.alignment_staging_drive_direction is None:
            if distance_m > 0.025:
                return self._staging_pose_step(current, target), "staging", False
            runtime.alignment_staging_drive_direction = (
                -1.0 if longitudinal_error_m >= 0.0 else 1.0
            )
            runtime.axial_alignment_mode = "curve"
            runtime.axial_pivot_sample_started_s = None
            runtime.axial_pivot_sample_yaw_rad = None
            runtime.axial_escape_until_s = None
        elif abs(longitudinal_error_m) > 0.035:
            runtime.alignment_staging_drive_direction = None
            runtime.axial_alignment_mode = "curve"
            runtime.axial_pivot_sample_started_s = None
            runtime.axial_pivot_sample_yaw_rad = None
            runtime.axial_escape_until_s = None
            return self._staging_pose_step(current, target), "staging", False

        # Normal staging remains tolerant of small CAD pitch/roll motion. A
        # requested parking recovery instead uses the explicit planar target:
        # unlike the full 3-D marker offset, this error can actually be
        # corrected by the differential drive before the straight approach.
        runtime_goal = getattr(runtime, "goal", None)
        goal_parameters = (
            runtime_goal.parameters if runtime_goal is not None else {}
        )
        quality_tolerance = goal_parameters.get(
            "contact_quality_planar_tolerance_m"
        )
        precision_recovery = quality_tolerance is not None
        feedback_approach = bool(
            goal_parameters.get("contact_approach_feedback", False)
        )
        lateral_tolerance_m = (
            0.005
            if feedback_approach
            else
            min(0.0025, float(quality_tolerance))
            if precision_recovery
            else 0.0025
        )
        pivot_enter_lateral_m = min(0.0020, lateral_tolerance_m)
        pivot_release_lateral_m = (
            max(0.0030, 2.0 * lateral_tolerance_m)
            if precision_recovery
            else 0.0040
        )
        yaw_tolerance_rad = math.radians(
            4.0
            if feedback_approach
            else 1.5
            if precision_recovery
            else 2.0
        )
        longitudinal_release_m = 0.030
        if (
            abs(lateral_error_m) <= lateral_tolerance_m
            and abs(yaw_error_rad) <= yaw_tolerance_rad
            and abs(longitudinal_error_m) <= longitudinal_release_m
        ):
            return (
                PoseControlStep(
                    linear_x_m_s=0.0,
                    angular_z_rad_s=0.0,
                    position_error_m=abs(lateral_error_m),
                    yaw_error_rad=-yaw_error_rad,
                    phase="complete",
                    done=True,
                ),
                "axial_pose_adjustment",
                True,
            )

        direction = runtime.alignment_staging_drive_direction
        if direction is None:
            raise RuntimeError("Axial pose-adjustment direction was not set")

        mode = runtime.axial_alignment_mode
        if mode == "escape":
            if (
                now_s is not None
                and runtime.axial_escape_until_s is not None
                and now_s < runtime.axial_escape_until_s
            ):
                pivot_sign = -yaw_error_rad
                if abs(pivot_sign) < 1.0e-9:
                    pivot_sign = -lateral_error_m * direction
                angular_z_rad_s = math.copysign(0.30, pivot_sign)
                return (
                    PoseControlStep(
                        linear_x_m_s=direction * 0.018,
                        angular_z_rad_s=angular_z_rad_s,
                        position_error_m=abs(lateral_error_m),
                        yaw_error_rad=-yaw_error_rad,
                        phase="unstick_arc",
                        done=False,
                    ),
                    "axial_pose_adjustment",
                    False,
                )
            runtime.axial_alignment_mode = "curve"
            runtime.axial_escape_until_s = None
            runtime.axial_pivot_sample_started_s = None
            runtime.axial_pivot_sample_yaw_rad = None
            mode = "curve"

        # Pivot is useful only while a material yaw correction remains.  The
        # old RC-Car-only rule latched pivot from lateral error alone: on the
        # straight Snake staging line, yaw was already below 0.2 degrees but
        # a 2.8-3.6 mm lateral residual kept both modules rotating forever.
        # Once yaw is aligned, return to an arc so differential drive has the
        # translational component required to remove that residual.
        yaw_needs_pivot = abs(yaw_error_rad) > yaw_tolerance_rad
        if (
            mode == "curve"
            and yaw_needs_pivot
            and abs(lateral_error_m) <= pivot_enter_lateral_m
        ):
            runtime.axial_alignment_mode = "pivot"
            runtime.axial_pivot_sample_started_s = now_s
            runtime.axial_pivot_sample_yaw_rad = current.yaw_rad
            mode = "pivot"
        elif mode == "pivot" and (
            not yaw_needs_pivot
            or abs(lateral_error_m) >= pivot_release_lateral_m
        ):
            runtime.axial_alignment_mode = "curve"
            runtime.axial_pivot_sample_started_s = None
            runtime.axial_pivot_sample_yaw_rad = None
            mode = "curve"

        if mode == "curve":
            translation_speed_m_s = 0.025
            # Eq. (4)-(5) in Liu et al. controls y' and theta' in the target
            # connector frame with K=diag(2, 1). The helper below is the
            # bounded, nonsingular realization of that law: at the requested
            # heading, v*sin(theta') is exactly -2*y' until the safe 25 deg
            # connector-attitude limit is reached.
            paper_reference = axial_pose_adjustment_reference(
                lateral_error_m,
                direction,
                translation_speed_m_s=translation_speed_m_s,
                lateral_gain_s=2.0,
                maximum_relative_yaw_rad=math.radians(25.0),
            )
            desired_relative_yaw = (
                paper_reference.desired_relative_yaw_rad
            )
            steering_error_rad = wrap_angle(
                desired_relative_yaw - yaw_error_rad
            )
            # Do not translate while initially pointing to the wrong side of
            # the correction arc; first acquire a useful steering direction.
            linear_x_m_s = (
                paper_reference.linear_x_m_s
                if abs(steering_error_rad) <= math.radians(35.0)
                else 0.0
            )
            command_phase = "curve"
        else:
            desired_relative_yaw = 0.0
            steering_error_rad = -yaw_error_rad
            linear_x_m_s = 0.0
            command_phase = "pivot"

        angular_z_rad_s = max(
            -0.8,
            min(0.8, 3.0 * steering_error_rad),
        )
        if (
            0.0 < abs(angular_z_rad_s)
            < self._face_alignment_min_turn_speed_rad_s
        ):
            angular_z_rad_s = math.copysign(
                self._face_alignment_min_turn_speed_rad_s,
                angular_z_rad_s,
            )

        # Detect every physically stalled in-place turn from body motion, not
        # merely turns carrying the semantic ``pivot`` mode.  The curve
        # controller also requests an in-place steering acquisition when its
        # initial heading error exceeds 35 degrees.  Treating that command as
        # a normal curve left a static-friction lock with zero translation and
        # no recovery path.
        in_place_turn = (
            abs(linear_x_m_s) < 1.0e-9
            and abs(angular_z_rad_s) > 1.0e-9
        )
        if in_place_turn and now_s is not None:
            sample_started_s = runtime.axial_pivot_sample_started_s
            sample_yaw_rad = runtime.axial_pivot_sample_yaw_rad
            if sample_started_s is None or sample_yaw_rad is None:
                runtime.axial_pivot_sample_started_s = now_s
                runtime.axial_pivot_sample_yaw_rad = current.yaw_rad
            elif now_s - sample_started_s >= 0.8:
                measured_yaw_delta_rad = abs(
                    wrap_angle(current.yaw_rad - sample_yaw_rad)
                )
                if (
                    abs(yaw_error_rad) > yaw_tolerance_rad
                    and measured_yaw_delta_rad < math.radians(0.5)
                ):
                    runtime.axial_alignment_mode = "escape"
                    runtime.axial_escape_until_s = now_s + 0.5
                    runtime.axial_pivot_sample_started_s = None
                    runtime.axial_pivot_sample_yaw_rad = None
                    runtime.axial_stall_recovery_count += 1
                    pivot_sign = -yaw_error_rad
                    angular_z_rad_s = math.copysign(0.30, pivot_sign)
                    linear_x_m_s = direction * 0.018
                    command_phase = "unstick_arc"
                else:
                    runtime.axial_pivot_sample_started_s = now_s
                    runtime.axial_pivot_sample_yaw_rad = current.yaw_rad
        elif mode == "curve":
            # A translating correction arc is not a pivot sample.  Start a
            # fresh measurement if a later steering acquisition stops again.
            runtime.axial_pivot_sample_started_s = None
            runtime.axial_pivot_sample_yaw_rad = None

        return (
            PoseControlStep(
                linear_x_m_s=linear_x_m_s,
                angular_z_rad_s=angular_z_rad_s,
                position_error_m=abs(lateral_error_m),
                yaw_error_rad=-yaw_error_rad,
                phase=command_phase,
                done=False,
            ),
            "axial_pose_adjustment",
            False,
        )

    def _has_residual_contact_error(
        self,
        evaluation: DockingPairEvaluation,
        thresholds: DockingThresholds | None = None,
    ) -> bool:
        """Detect a near-contact pose that a straight drive cannot repair."""

        limits = thresholds or self._docking_thresholds
        return (
            evaluation.normal_separation_m < 0.020
            and (
                evaluation.lateral_offset_m
                > limits.lateral_offset_tolerance_m
                or evaluation.normal_misalignment_rad
                > limits.normal_alignment_tolerance_rad
            )
        )

    def _docking_thresholds_for_goal(
        self,
        goal: PrimitiveGoal,
    ) -> DockingThresholds:
        """Return the physical contact gate requested by the runtime policy."""

        raw_tolerance = goal.parameters.get(
            "top_bottom_contact_tolerance_m"
        )
        if raw_tolerance is None:
            return self._docking_thresholds
        return replace(
            self._docking_thresholds,
            top_bottom_contact_tolerance_m=float(raw_tolerance),
        )

    def _face_alignment_complete(
        self,
        evaluation: DockingPairEvaluation,
    ) -> bool:
        """Use exactly the physical docking gate after contact is reached."""

        # A second set of hard-coded limits used to leave align_faces running
        # with zero wheel command while the docking manager would make a
        # different decision.  Alignment and attach now share one verdict.
        return evaluation.eligible

    @staticmethod
    def _planar_face_lateral_offset(first: Any, second: Any) -> float:
        """Measure connector centre error correctable by planar locomotion.

        ``DockingPairEvaluation.lateral_offset_m`` is a 3-D distance and can
        include marker-height differences caused by CAD pitch and settling.
        Projecting the centre delta onto the horizontal tangent of the target
        face isolates the component the two wheels can actually remove.
        """

        normal_x, normal_y = second.outward_normal_world[:2]
        normal_norm = math.hypot(normal_x, normal_y)
        if normal_norm < 1.0e-6:
            return math.hypot(
                first.position_world_m[0] - second.position_world_m[0],
                first.position_world_m[1] - second.position_world_m[1],
            )
        delta_x = first.position_world_m[0] - second.position_world_m[0]
        delta_y = first.position_world_m[1] - second.position_world_m[1]
        tangent_x = -normal_y / normal_norm
        tangent_y = normal_x / normal_norm
        return abs(delta_x * tangent_x + delta_y * tangent_y)

    def _change_docking(
        self,
        runtime: _ActiveGoal,
        now_s: float,
    ) -> tuple[dict[str, SmoresCommand], PrimitiveStatus]:
        goal = runtime.goal
        first, second = goal.module_ids
        action = (
            "attach"
            if goal.primitive is PrimitiveName.DOCK
            else "detach"
        )
        if action == "attach":
            face_a = str(goal.parameters["face_a"])
            face_b = str(goal.parameters["face_b"])
            docking_thresholds = self._docking_thresholds_for_goal(goal)
            evaluation = evaluate_face_pair(
                self._face_pose(first, face_a),
                self._face_pose(second, face_b),
                docking_thresholds,
            )
            if not self._face_alignment_complete(evaluation):
                current = self._planar_pose(first)
                target = self._face_alignment_target(
                    first,
                    face_a,
                    second,
                    face_b,
                )
                step = self._straight_face_approach_step(
                    current,
                    target,
                    feedback_enabled=bool(
                        goal.parameters.get(
                            "contact_approach_feedback",
                            False,
                        )
                    ),
                )
                if step.done:
                    return {}, self._finish(
                        runtime,
                        PrimitiveState.FAILED,
                        now_s,
                        code="CONTACT_ALIGNMENT_LOST",
                        message=(
                            "contact cannot be closed without changing the "
                            "already completed alignment"
                        ),
                    )
                return {
                    first: SmoresCommand(
                        linear_x_m_s=step.linear_x_m_s,
                        angular_z_rad_s=step.angular_z_rad_s,
                    )
                }, self._make_status(
                    goal,
                    PrimitiveState.RUNNING,
                    now_s,
                    phase="contact",
                    progress=0.5,
                    code="CLOSING_CONTACT",
                    message="closing aligned face gap before docking",
                    feedback={
                        "normal_gap_m": evaluation.normal_separation_m,
                        "lateral_error_m": evaluation.lateral_offset_m,
                        "normal_error_rad": evaluation.normal_misalignment_rad,
                        "command_linear_m_s": step.linear_x_m_s,
                        "command_angular_rad_s": step.angular_z_rad_s,
                    },
                )
        command = DockingCommand(
            action,  # type: ignore[arg-type]
            first,
            second,
            str(goal.parameters["face_a"]),
            str(goal.parameters["face_b"]),
        )
        if action == "attach" and goal.parameters.get(
            "top_bottom_contact_tolerance_m"
        ) is not None:
            result = self._docking.handle(
                command,
                thresholds=self._docking_thresholds_for_goal(goal),
                snap_to_nominal=bool(
                    goal.parameters.get("snap_to_nominal", False)
                ),
            )
        else:
            result = self._docking.handle(command)
        state = (
            PrimitiveState.SUCCEEDED
            if result.accepted
            else PrimitiveState.FAILED
        )
        code = (
            "DOCKED"
            if result.accepted and action == "attach"
            else "UNDOCKED"
            if result.accepted
            else "DOCKING_REJECTED"
        )
        return {}, self._finish(
            runtime,
            state,
            now_s,
            code=code,
            message=result.message,
        )

    def _move_joint(
        self,
        runtime: _ActiveGoal,
        now_s: float,
    ) -> tuple[dict[str, SmoresCommand], PrimitiveStatus]:
        goal = runtime.goal
        module_id = goal.module_ids[0]
        current_pan, current_tilt = self._joint_positions(module_id)
        target = runtime.resolved_target_rad
        if target is None:
            raise RuntimeError("Joint primitive has no resolved target")
        is_pan = goal.primitive in {
            PrimitiveName.SET_PAN,
            PrimitiveName.ROTATE_PAN_BY,
        }
        current = current_pan if is_pan else current_tilt
        self._retain_structure_targets(
            goal.parameters.get("structural_hold_module_ids", ())
        )
        retained = self._retained_internal_commands.get(module_id)
        held_tilt = (
            retained.tilt_target_rad
            if retained is not None
            else current_tilt
        )
        error = target - current
        tolerance = (
            self._joint_tolerance_rad
            if is_pan
            else self._tilt_joint_tolerance_rad
        )
        tolerance = float(goal.parameters.get("tolerance_rad", tolerance))
        coordination_group = goal.parameters.get("coordination_group")
        if not is_pan and coordination_group is not None:
            if not self._coordinated_tilt_ready(runtime):
                command = SmoresCommand(
                    pan_target_rad=current_pan,
                    tilt_target_rad=current_tilt,
                    internal_motion=InternalMotionMode.TILT,
                )
                commands = {module_id: command}
                self._add_group_stabilizers(commands, goal)
                self._add_fold_pusher_command(
                    commands, goal, moving=False
                )
                return commands, self._make_status(
                    goal,
                    PrimitiveState.RUNNING,
                    now_s,
                    phase="tilt_coordination",
                    progress=0.0,
                    code="WAITING_JOINT_GROUP",
                    message="waiting for every coordinated tilt goal",
                    feedback={
                        "coordination_group": str(coordination_group),
                        "current_rad": current_tilt,
                        "target_rad": target,
                    },
                )
        # A coordinated member that is already inside tolerance must still
        # participate in the barrier.  Completing it before the whole group
        # is admitted removes it from ``_active`` and makes the requested
        # coordination_size unreachable for the remaining members.
        if abs(error) <= tolerance:
            if (
                not is_pan
                and coordination_group is not None
                and not self._coordinated_tilt_complete(runtime)
            ):
                commands = {
                    module_id: SmoresCommand(
                        pan_target_rad=current_pan,
                        tilt_target_rad=target,
                        internal_motion=InternalMotionMode.TILT,
                    )
                }
                self._add_fold_pusher_command(
                    commands, goal, moving=True
                )
                self._add_group_stabilizers(commands, goal)
                return commands, self._make_status(
                    goal,
                    PrimitiveState.RUNNING,
                    now_s,
                    phase="tilt_group_completion",
                    progress=1.0,
                    code="WAITING_JOINT_GROUP_COMPLETION",
                    message=(
                        "holding the reached tilt until every coordinated "
                        "fold actuator reaches its target"
                    ),
                    feedback={
                        "coordination_group": str(coordination_group),
                        "current_rad": current_tilt,
                        "target_rad": target,
                    },
                )
            self._retain_joint_target(
                goal,
                target,
                current_pan,
                current_tilt,
            )
            return {}, self._finish(
                runtime,
                PrimitiveState.SUCCEEDED,
                now_s,
                code="JOINT_TARGET_REACHED",
                message=f"{'pan' if is_pan else 'tilt'} target reached",
            )
        commanded_target = target
        if "max_servo_error_rad" in goal.parameters:
            max_servo_error = float(
                goal.parameters["max_servo_error_rad"]
            )
            commanded_target = current + max(
                -max_servo_error,
                min(max_servo_error, error),
            )
        coordination_lead_limited = False
        if (
            not is_pan
            and coordination_group is not None
            and "max_coordination_lead_rad" in goal.parameters
        ):
            commanded_target, coordination_lead_limited = (
                self._limit_coordinated_tilt_lead(
                    runtime,
                    current,
                    commanded_target,
                )
            )
        command = SmoresCommand(
            pan_target_rad=commanded_target if is_pan else current_pan,
            tilt_target_rad=held_tilt if is_pan else commanded_target,
            internal_motion=(
                InternalMotionMode.PAN
                if is_pan
                else InternalMotionMode.TILT
            ),
        )
        commands = {module_id: command}
        if not is_pan:
            self._add_fold_pusher_command(commands, goal, moving=True)
            self._add_group_stabilizers(commands, goal)
        progress = 1.0 - min(
            1.0,
            abs(error)
            / max(
                abs(
                    float(
                        goal.parameters.get(
                            "delta_rad",
                            goal.parameters.get("angle_rad", target),
                        )
                    )
                ),
                tolerance,
            ),
        )
        return commands, self._make_status(
            goal,
            PrimitiveState.RUNNING,
            now_s,
            phase="pan" if is_pan else "tilt",
            progress=progress,
            code="MOVING_JOINT",
            message=f"moving {'pan' if is_pan else 'tilt'} joint",
            feedback={
                "current_rad": current,
                "target_rad": target,
                "commanded_target_rad": commanded_target,
                "error_rad": error,
                "coordination_lead_limited": coordination_lead_limited,
            },
        )

    @staticmethod
    def _add_group_stabilizers(
        commands: dict[str, SmoresCommand],
        goal: PrimitiveGoal,
    ) -> None:
        """Keep declared frame modules square during a coupled fold."""

        for raw_module_id in goal.parameters.get(
            "stabilize_during_group_module_ids", ()
        ):
            module_id = str(raw_module_id)
            commands[module_id] = SmoresCommand(
                pan_target_rad=0.0,
                tilt_target_rad=0.0,
                internal_motion=InternalMotionMode.STRUCTURAL_HOLD,
            )

    def _add_fold_pusher_command(
        self,
        commands: dict[str, SmoresCommand],
        goal: PrimitiveGoal,
        *,
        moving: bool,
    ) -> None:
        """Drive the paired spoke while retaining its structural posture."""

        raw_module_id = goal.parameters.get("pusher_module_id")
        if raw_module_id is None:
            return
        module_id = str(raw_module_id)
        speed = (
            float(goal.parameters["pusher_linear_m_s"])
            if moving
            else 0.0
        )
        retained = self._retained_internal_commands.get(module_id)
        if retained is not None:
            pan_target = retained.pan_target_rad
            tilt_target = retained.tilt_target_rad
        else:
            pan_target, tilt_target = self._joint_positions(module_id)
        commands[module_id] = SmoresCommand(
            linear_x_m_s=speed,
            pan_target_rad=pan_target,
            tilt_target_rad=tilt_target,
            internal_motion=InternalMotionMode.STRUCTURAL_HOLD,
        )

    def _retain_joint_target(
        self,
        goal: PrimitiveGoal,
        target: float,
        current_pan: float,
        current_tilt: float,
    ) -> None:
        """Latch only a successfully commanded posture joint."""

        module_id = goal.module_ids[0]
        is_pan = goal.primitive in {
            PrimitiveName.SET_PAN,
            PrimitiveName.ROTATE_PAN_BY,
        }
        retained = self._retained_internal_commands.get(module_id)
        if (
            retained is not None
            and retained.internal_motion
            is InternalMotionMode.STRUCTURAL_HOLD
        ):
            self._retained_internal_commands[module_id] = SmoresCommand(
                pan_target_rad=(
                    target if is_pan else retained.pan_target_rad
                ),
                tilt_target_rad=(
                    retained.tilt_target_rad if is_pan else target
                ),
                internal_motion=InternalMotionMode.STRUCTURAL_HOLD,
            )
            return
        self._retained_internal_commands[module_id] = SmoresCommand(
            pan_target_rad=target if is_pan else current_pan,
            tilt_target_rad=current_tilt if is_pan else target,
            internal_motion=(
                InternalMotionMode.PAN if is_pan else InternalMotionMode.TILT
            ),
        )

    def _retain_structure_targets(self, module_ids: Any) -> None:
        """Latch missing operational holds without overwriting fold targets."""

        for raw_module_id in module_ids:
            module_id = str(raw_module_id)
            retained = self._retained_internal_commands.get(module_id)
            if (
                retained is not None
                and retained.internal_motion
                is InternalMotionMode.STRUCTURAL_HOLD
            ):
                continue
            if retained is None:
                current_pan, current_tilt = self._joint_positions(module_id)
            else:
                current_pan = retained.pan_target_rad
                current_tilt = retained.tilt_target_rad
            self._retained_internal_commands[module_id] = SmoresCommand(
                pan_target_rad=current_pan,
                tilt_target_rad=current_tilt,
                internal_motion=InternalMotionMode.STRUCTURAL_HOLD,
            )

    def _coordinated_tilt_ready(
        self,
        runtime: _ActiveGoal,
    ) -> bool:
        """Release a connected tilt group in one simulation frame.

        Goals arrive serially through the file bridge. Holding them until the
        whole group is active prevents the first three RC-Car corners from
        loading the chassis before the fourth receives its command. Once the
        barrier opens every member receives its final target immediately; a
        slow progress ramp would only let a mechanically blocked member hold
        the entire morphology below its useful posture.
        """

        group_name = str(runtime.goal.parameters["coordination_group"])
        expected_size = int(runtime.goal.parameters["coordination_size"])
        group = [
            candidate
            for candidate in self._active.values()
            if candidate.goal.primitive is PrimitiveName.SET_TILT
            and candidate.goal.parameters.get("coordination_group") == group_name
        ]
        if group_name not in self._released_joint_groups:
            if len(group) < expected_size:
                return False
            self._released_joint_groups.add(group_name)
        return True

    def _coordinated_tilt_complete(self, runtime: _ActiveGoal) -> bool:
        """Keep every fold actuator energized until the whole group arrives."""

        group_name = str(runtime.goal.parameters["coordination_group"])
        if group_name in self._completed_joint_groups:
            return True
        expected_size = int(runtime.goal.parameters["coordination_size"])
        group = [
            candidate
            for candidate in self._active.values()
            if candidate.goal.primitive is PrimitiveName.SET_TILT
            and candidate.goal.parameters.get("coordination_group")
            == group_name
        ]
        if len(group) < expected_size:
            return False
        for candidate in group:
            target = candidate.resolved_target_rad
            if target is None:
                return False
            _, current_tilt = self._joint_positions(
                candidate.goal.module_ids[0]
            )
            tolerance = float(
                candidate.goal.parameters.get(
                    "tolerance_rad", self._tilt_joint_tolerance_rad
                )
            )
            if abs(target - current_tilt) > tolerance:
                return False
        hold_ids: set[str] = set()
        for candidate in group:
            hold_ids.update(
                str(item)
                for item in candidate.goal.parameters.get(
                    "hold_after_group_module_ids", ()
                )
            )
        for module_id in hold_ids:
            current_pan, current_tilt = self._joint_positions(module_id)
            self._retained_internal_commands[module_id] = SmoresCommand(
                pan_target_rad=current_pan,
                tilt_target_rad=current_tilt,
                internal_motion=InternalMotionMode.STRUCTURAL_HOLD,
            )
        self._completed_joint_groups.add(group_name)
        return True

    def _limit_coordinated_tilt_lead(
        self,
        runtime: _ActiveGoal,
        current_tilt: float,
        commanded_target: float,
    ) -> tuple[float, bool]:
        """Keep a fold group mechanically synchronized while it moves.

        Releasing every goal in the same frame is not sufficient under load:
        PhysX can let two supports rotate quickly, changing the chassis pose
        until the opposite supports lose their useful reaction geometry.  A
        member more than five degrees ahead of the slowest active member is
        therefore held at its measured position until the group catches up.
        The final targets and completion tolerances remain unchanged.
        """

        group_name = str(runtime.goal.parameters["coordination_group"])
        group = [
            candidate
            for candidate in self._active.values()
            if candidate.goal.primitive is PrimitiveName.SET_TILT
            and candidate.goal.parameters.get("coordination_group")
            == group_name
            and candidate.resolved_target_rad is not None
        ]
        if len(group) < 2:
            return commanded_target, False

        remaining_by_goal: dict[str, float] = {}
        for candidate in group:
            _, candidate_tilt = self._joint_positions(
                candidate.goal.module_ids[0]
            )
            remaining_by_goal[candidate.goal.goal_id] = abs(
                float(candidate.resolved_target_rad) - candidate_tilt
            )

        slowest_remaining = max(remaining_by_goal.values())
        current_remaining = abs(
            float(runtime.resolved_target_rad) - current_tilt
        )
        maximum_lead_rad = float(
            runtime.goal.parameters["max_coordination_lead_rad"]
        )
        if not math.isfinite(maximum_lead_rad) or maximum_lead_rad <= 0.0:
            return commanded_target, False
        if current_remaining < slowest_remaining - maximum_lead_rad:
            return current_tilt, True
        return commanded_target, False

    def _resolve_joint_target(self, goal: PrimitiveGoal) -> float | None:
        if goal.primitive not in {
            PrimitiveName.SET_PAN,
            PrimitiveName.ROTATE_PAN_BY,
            PrimitiveName.SET_TILT,
            PrimitiveName.ROTATE_TILT_BY,
        }:
            return None
        pan, tilt = self._joint_positions(goal.module_ids[0])
        if goal.primitive is PrimitiveName.SET_PAN:
            return float(goal.parameters["angle_rad"])
        if goal.primitive is PrimitiveName.ROTATE_PAN_BY:
            return pan + float(goal.parameters["delta_rad"])
        if goal.primitive is PrimitiveName.SET_TILT:
            target = float(goal.parameters["angle_rad"])
        else:
            target = tilt + float(goal.parameters["delta_rad"])
        if not self._geometry.tilt_min_rad <= target <= self._geometry.tilt_max_rad:
            raise ValueError(
                f"tilt target {target:.4f}rad is outside "
                f"[{self._geometry.tilt_min_rad:.4f}, "
                f"{self._geometry.tilt_max_rad:.4f}]"
            )
        return target

    def _joint_positions(self, module_id: str) -> tuple[float, float]:
        state = self._states[module_id].read()
        pan = self._pan_trackers[module_id].update(state.pan_joint_rad)
        return pan, -state.tilt_joint_rad

    def _planar_pose(self, module_id: str) -> PlanarPose:
        from pxr import Gf, Usd, UsdGeom

        body_path = f"{self._module_roots[module_id]}/body_link"
        matrix = UsdGeom.Xformable(
            self._stage.GetPrimAtPath(body_path)
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        position = matrix.ExtractTranslation()
        forward = matrix.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
        return PlanarPose(
            float(position[0]),
            float(position[1]),
            math.atan2(float(forward[1]), float(forward[0])),
        )

    def _face_pose(self, module_id: str, face_name: str) -> Any:
        return next(
            pose
            for pose in self._docking.face_poses_for(module_id)
            if pose.face.face_name == face_name
        )

    def _face_alignment_target(
        self,
        mobile_id: str,
        mobile_face_name: str,
        target_id: str,
        target_face_name: str,
    ) -> PlanarPose:
        mobile_pose = self._planar_pose(mobile_id)
        mobile_face = self._face_pose(mobile_id, mobile_face_name)
        target_face = self._face_pose(target_id, target_face_name)

        mobile_normal_xy = mobile_face.outward_normal_world[:2]
        target_normal_xy = target_face.outward_normal_world[:2]
        if math.hypot(*mobile_normal_xy) < 1.0e-6:
            raise RuntimeError("Mobile face normal has no planar projection")
        if math.hypot(*target_normal_xy) < 1.0e-6:
            raise RuntimeError("Target face normal has no planar projection")

        current_face_normal_yaw = math.atan2(
            mobile_normal_xy[1],
            mobile_normal_xy[0],
        )
        local_face_normal_yaw = wrap_angle(
            current_face_normal_yaw - mobile_pose.yaw_rad
        )
        desired_face_normal_yaw = math.atan2(
            -target_normal_xy[1],
            -target_normal_xy[0],
        )
        desired_body_yaw = wrap_angle(
            desired_face_normal_yaw - local_face_normal_yaw
        )

        offset_x = mobile_face.position_world_m[0] - mobile_pose.x_m
        offset_y = mobile_face.position_world_m[1] - mobile_pose.y_m
        cosine = math.cos(-mobile_pose.yaw_rad)
        sine = math.sin(-mobile_pose.yaw_rad)
        local_offset_x = cosine * offset_x - sine * offset_y
        local_offset_y = sine * offset_x + cosine * offset_y

        target_normal_norm = math.hypot(*target_normal_xy)
        face_pair = frozenset((mobile_face_name, target_face_name))
        desired_gap_m = (
            0.0
            if face_pair == frozenset(("TOP", "BOTTOM"))
            else self._desired_face_marker_gap_m
        )
        desired_face_x = (
            target_face.position_world_m[0]
            + desired_gap_m
            * target_normal_xy[0]
            / target_normal_norm
        )
        desired_face_y = (
            target_face.position_world_m[1]
            + desired_gap_m
            * target_normal_xy[1]
            / target_normal_norm
        )
        cosine = math.cos(desired_body_yaw)
        sine = math.sin(desired_body_yaw)
        world_offset_x = cosine * local_offset_x - sine * local_offset_y
        world_offset_y = sine * local_offset_x + cosine * local_offset_y
        return PlanarPose(
            desired_face_x - world_offset_x,
            desired_face_y - world_offset_y,
            desired_body_yaw,
        )

    def _face_alignment_staging_target(
        self,
        final_target: PlanarPose,
        target_id: str,
        target_face_name: str,
        distance_m: float | None = None,
    ) -> PlanarPose:
        """Offset the final pose away from contact while preserving yaw.

        A differential-drive module cannot safely correct a large yaw error
        after its face is already touching the target. The staging pose gives
        it enough free space to settle orientation first, followed by one
        short straight approach.
        """

        target_face = self._face_pose(
            target_id,
            target_face_name,
        )
        normal_x, normal_y = (
            target_face.outward_normal_world[:2]
        )
        normal_norm = math.hypot(normal_x, normal_y)
        if normal_norm < 1.0e-6:
            raise RuntimeError(
                "Target face normal has no planar projection"
            )
        distance = (
            self._face_alignment_staging_distance_m
            if distance_m is None
            else distance_m
        )
        return PlanarPose(
            final_target.x_m
            + distance * normal_x / normal_norm,
            final_target.y_m
            + distance * normal_y / normal_norm,
            final_target.yaw_rad,
        )

    @staticmethod
    def _staging_path_fallback_level(goal: PrimitiveGoal) -> int:
        """Return the validated progressive staging fallback level."""

        return int(goal.parameters.get("staging_path_fallback_level", 0))

    def _staging_distance_for_goal(self, goal: PrimitiveGoal) -> float:
        """Keep staging useful while moving it out of a false obstacle."""

        level = self._staging_path_fallback_level(goal)
        if level == 1:
            return min(self._face_alignment_staging_distance_m, 0.040)
        if level >= 2:
            return min(self._face_alignment_staging_distance_m, 0.015)
        return self._face_alignment_staging_distance_m

    def _staging_clearance_for_goal(self, goal: PrimitiveGoal) -> float:
        """Relax the circular proxy, never below the physical envelope."""

        level = self._staging_path_fallback_level(goal)
        if level == 1:
            return min(self._staging_center_clearance_m, 0.096)
        if level >= 2:
            return min(self._staging_center_clearance_m, 0.082)
        return self._staging_center_clearance_m

    def _staging_margin_for_goal(self, goal: PrimitiveGoal) -> float:
        """Reduce visibility-graph margin together with its footprint."""

        level = self._staging_path_fallback_level(goal)
        if level == 1:
            return min(self._staging_waypoint_margin_m, 0.010)
        if level >= 2:
            return min(self._staging_waypoint_margin_m, 0.005)
        return self._staging_waypoint_margin_m

    def _finish(
        self,
        runtime: _ActiveGoal,
        state: PrimitiveState,
        now_s: float,
        code: str,
        message: str,
    ) -> PrimitiveStatus:
        goal = runtime.goal
        status = self._make_status(
            goal,
            state,
            now_s,
            phase="terminal",
            progress=1.0 if state is PrimitiveState.SUCCEEDED else 0.0,
            code=code,
            message=message,
        )
        self._status = status
        self._active.pop(goal.goal_id, None)
        for resource in runtime.resources:
            owners = self._resource_owners.get(resource)
            if owners is None:
                continue
            owners.pop(goal.goal_id, None)
            if not owners:
                del self._resource_owners[resource]
        return status

    def _resources_for(self, goal: PrimitiveGoal) -> dict[str, str]:
        """Resolve exclusive and shared physical-resource claims."""

        first = goal.module_ids[0]
        if goal.primitive is PrimitiveName.DRIVE_TO_POSE:
            return {f"locomotion:{first}": "exclusive"}
        if goal.primitive is PrimitiveName.ALIGN_FACES:
            second = goal.module_ids[1]
            resources = {
                f"locomotion:{first}": "exclusive",
                # Multiple mobile modules may align with distinct faces of
                # the same stationary target.  Shared claims freeze the
                # target against writers while allowing those alignments.
                f"locomotion:{second}": "shared",
            }
            face_a = str(goal.parameters["face_a"])
            face_b = str(goal.parameters["face_b"])
            if face_a == "TOP":
                resources[f"internal_motion:{first}"] = "exclusive"
            if face_b == "TOP":
                resources[f"internal_motion:{second}"] = "exclusive"
            return resources
        if goal.primitive is PrimitiveName.ASSISTED_ALIGN_FACES:
            payload, target, helper = goal.module_ids
            return {
                f"locomotion:{payload}": "exclusive",
                f"locomotion:{target}": "shared",
                f"locomotion:{helper}": "exclusive",
            }
        if goal.primitive in {
            PrimitiveName.SET_PAN,
            PrimitiveName.ROTATE_PAN_BY,
            PrimitiveName.SET_TILT,
            PrimitiveName.ROTATE_TILT_BY,
        }:
            resources = {f"internal_motion:{first}": "exclusive"}
            if goal.primitive is PrimitiveName.SET_TILT:
                pusher_module_id = goal.parameters.get("pusher_module_id")
                if pusher_module_id is not None:
                    resources.update(
                        {
                            f"locomotion:{pusher_module_id}": "exclusive",
                            f"internal_motion:{pusher_module_id}": (
                                "exclusive"
                            ),
                        }
                    )
                resources.update(
                    {
                        f"internal_motion:{module_id}": "shared"
                        for module_id in goal.parameters.get(
                            "stabilize_during_group_module_ids", ()
                        )
                    }
                )
            return resources
        face_a = str(goal.parameters["face_a"])
        face_b = str(goal.parameters["face_b"])
        connector_resources = {
            f"connector:{first}:{face_a}": "exclusive",
            f"connector:{goal.module_ids[1]}:{face_b}": "exclusive",
        }
        if goal.primitive is PrimitiveName.DOCK:
            connector_resources.update(
                {
                    f"locomotion:{first}": "exclusive",
                    f"locomotion:{goal.module_ids[1]}": "shared",
                }
            )
        return connector_resources

    @staticmethod
    def _merge_commands(
        merged: dict[str, SmoresCommand],
        commands: Mapping[str, SmoresCommand],
        resources: frozenset[str],
    ) -> None:
        """Compose locomotion and internal motion on the same module."""

        for module_id, incoming in commands.items():
            current = merged.get(module_id, SmoresCommand())
            owns_locomotion = f"locomotion:{module_id}" in resources
            owns_internal = f"internal_motion:{module_id}" in resources
            merged[module_id] = SmoresCommand(
                linear_x_m_s=(
                    incoming.linear_x_m_s
                    if owns_locomotion
                    else current.linear_x_m_s
                ),
                angular_z_rad_s=(
                    incoming.angular_z_rad_s
                    if owns_locomotion
                    else current.angular_z_rad_s
                ),
                pan_target_rad=(
                    incoming.pan_target_rad
                    if owns_internal
                    else current.pan_target_rad
                ),
                tilt_target_rad=(
                    incoming.tilt_target_rad
                    if owns_internal
                    else current.tilt_target_rad
                ),
                internal_motion=(
                    incoming.internal_motion
                    if owns_internal
                    else current.internal_motion
                ),
                pan_velocity_rad_s=(
                    incoming.pan_velocity_rad_s
                    if owns_internal
                    else current.pan_velocity_rad_s
                ),
            )

    @staticmethod
    def _make_status(
        goal: PrimitiveGoal,
        state: PrimitiveState,
        now_s: float,
        phase: str,
        code: str,
        message: str,
        progress: float = 0.0,
        feedback: Mapping[str, Any] | None = None,
    ) -> PrimitiveStatus:
        return PrimitiveStatus(
            goal_id=goal.goal_id,
            primitive=goal.primitive,
            state=state,
            stamp_s=now_s,
            module_ids=goal.module_ids,
            phase=phase,
            progress=progress,
            code=code,
            message=message,
            feedback=feedback or {},
        )
