"""Deterministic profile-following and arch-wave gaits for Snake8 stairs."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from statistics import median
from typing import Any, Mapping, Sequence

from mssr_expert.behaviors.morphology_library import (
    AssignedModule,
    BehaviorProgramStep,
    JointTarget,
    LongitudinalDisplacementGoal,
    LongitudinalPositionGoal,
)
from mssr_expert.behaviors.snake_stair_support import (
    apply_trailing_compliance,
)
from mssr_expert.graph.attributed_robot_graph import AttributedRobotGraph
from mssr_expert.primitives.common import module_position


class SnakeStairGaitError(ValueError):
    """Raised when the live robot or course cannot define a safe gait."""


@dataclass(frozen=True)
class UniformStaircase:
    """Uniform +X staircase recognized from Isaac course landmarks."""

    first_riser_x_m: float
    tread_depth_m: float
    top_heights_m: tuple[float, ...]
    rise_m: float

    @classmethod
    def from_course(cls, course: Mapping[str, Any]) -> "UniformStaircase":
        if course.get("frame_id") != "world":
            raise SnakeStairGaitError("Stair landmarks must use world frame")
        stairs = course.get("stairs")
        if not isinstance(stairs, Mapping):
            raise SnakeStairGaitError("Course has no stair landmarks")
        try:
            first = float(stairs["first_riser_x_m"])
            depth = float(stairs["riser_depth_m"])
            heights = tuple(
                float(value) for value in stairs["top_heights_m"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SnakeStairGaitError("Invalid stair landmarks") from error
        if not heights or depth <= 0.0 or not all(
            math.isfinite(value) for value in (first, depth, *heights)
        ):
            raise SnakeStairGaitError(
                "Stair dimensions must be positive and finite"
            )
        rises = tuple(
            upper - lower
            for lower, upper in zip((0.0, *heights[:-1]), heights)
        )
        if min(rises) <= 0.0:
            raise SnakeStairGaitError("Stair heights must increase strictly")
        rise = sum(rises) / len(rises)
        if any(abs(value - rise) > 0.005 for value in rises):
            raise SnakeStairGaitError(
                "Snake8 gait requires uniform stair rises"
            )
        collision_boxes = course.get("collision_boxes")
        if collision_boxes is not None:
            if not isinstance(collision_boxes, list | tuple):
                raise SnakeStairGaitError(
                    "Course collision_boxes must be a sequence"
                )
            try:
                riser_boxes = sorted(
                    (
                        box for box in collision_boxes
                        if isinstance(box, Mapping)
                        and box.get("semantic") == "stair_test_riser"
                    ),
                    key=lambda box: float(box["center_xyz_m"][0]),
                )
                if len(riser_boxes) != len(heights):
                    raise SnakeStairGaitError(
                        "Stair landmarks disagree with world collision boxes"
                    )
                for index, (box, top_height) in enumerate(
                    zip(riser_boxes, heights)
                ):
                    center = tuple(
                        float(value) for value in box["center_xyz_m"]
                    )
                    size = tuple(float(value) for value in box["size_xyz_m"])
                    front_x = center[0] - 0.5 * size[0]
                    top_z = center[2] + 0.5 * size[2]
                    expected_front = first + index * depth
                    if (
                        abs(front_x - expected_front) > 0.001
                        or abs(top_z - top_height) > 0.001
                    ):
                        raise SnakeStairGaitError(
                            "Stair landmarks disagree with world collision "
                            "boxes"
                        )
            except SnakeStairGaitError:
                raise
            except (KeyError, TypeError, ValueError, IndexError) as error:
                raise SnakeStairGaitError(
                    "Invalid stair collision-box geometry"
                ) from error
        return cls(first, depth, heights, rise)


class SnakeStairGaitPlanner:
    """Generate micro-interleaved posture and traction phases for Snake8."""

    MODULE_COUNT = 8
    INITIAL_RISER_EDGE = 4
    TIMED_PARAMETER_NAMES = frozenset(
        {
            "riser_approach_duration_s",
            "front_pull_duration_s",
            "transfer_pull_duration_s",
            "tread_advance_duration_s",
            "slip_compensation",
        }
    )

    def plan(
        self,
        graph: AttributedRobotGraph,
        assignments: Sequence[AssignedModule],
        parameters: Mapping[str, Any],
        *,
        arch_wave: bool = False,
    ) -> tuple[BehaviorProgramStep, ...]:
        timed_parameters = sorted(
            self.TIMED_PARAMETER_NAMES.intersection(parameters)
        )
        if timed_parameters:
            raise SnakeStairGaitError(
                "crawl_stairs uses geometric world-pose goals and does not "
                "accept timed parameters: " + ", ".join(timed_parameters)
            )
        ordered = tuple(sorted(assignments, key=self._vertex_index))
        if len(ordered) != self.MODULE_COUNT:
            raise SnakeStairGaitError(
                "Deterministic stair gait requires Snake8"
            )
        course = graph.global_attributes.get("course")
        if not isinstance(course, Mapping):
            raise SnakeStairGaitError("Robot graph has no course metadata")
        staircase = UniformStaircase.from_course(course)
        positions = self._ordered_positions(graph, ordered)
        spacing = self._link_spacing(positions)
        wheel_radius = self._wheel_radius(graph)
        usable_tilt_limit = math.inf
        if arch_wave:
            tilt_limit_margin = self._number(
                parameters,
                "tilt_limit_margin_rad",
                0.030,
            )
            if not 0.0 <= tilt_limit_margin <= 0.15:
                raise SnakeStairGaitError(
                    "tilt_limit_margin_rad must be in [0.0, 0.15]"
                )
            usable_tilt_limit = (
                self._symmetric_tilt_limit(graph, ordered)
                - tilt_limit_margin
            )
            if usable_tilt_limit <= 0.0:
                raise SnakeStairGaitError(
                    "SMORES TILT limits leave no usable stair range"
                )
        if staircase.rise_m >= spacing:
            raise SnakeStairGaitError("Stair rise exceeds one Snake8 link")
        heading = math.atan2(
            positions[-1][1] - positions[0][1],
            positions[-1][0] - positions[0][0],
        )
        max_heading = self._number(
            parameters, "max_alignment_error_rad", 0.35
        )
        if max_heading <= 0.0 or max_heading > math.pi:
            raise SnakeStairGaitError(
                "max_alignment_error_rad must be in (0, pi]"
            )
        if abs(heading) > max_heading:
            raise SnakeStairGaitError(
                "Snake8 is not aligned with +X stairs: "
                f"heading={heading:.3f} rad"
            )

        bend_angle = math.asin(staircase.rise_m / spacing)
        arch_clearance = 0.0
        if arch_wave:
            arch_clearance = self._number(
                parameters,
                "arch_clearance_m",
                0.58 * wheel_radius,
            )
            if not 0.008 <= arch_clearance <= 0.025:
                raise SnakeStairGaitError(
                    "arch_clearance_m must be in [0.008, 0.025]"
                )
            if staircase.rise_m + arch_clearance >= 2.0 * spacing:
                raise SnakeStairGaitError(
                    "rise plus arch_clearance_m exceeds two Snake8 links"
                )
        diagonal_run = math.sqrt(spacing**2 - staircase.rise_m**2)
        horizontal_links = max(
            1,
            round((staircase.tread_depth_m - diagonal_run) / spacing),
        )
        stride = horizontal_links + 1
        substeps = self._integer(parameters, "profile_substeps", 6, 1, 6)
        approach_speed = self._speed(
            parameters, "riser_approach_linear_m_s", 0.060
        )
        crawl_speed = self._speed(parameters, "linear_m_s", 0.040)
        crawl_tolerance = self._number(
            parameters, "crawl_goal_tolerance_m", 0.004
        )
        if not 0.001 <= crawl_tolerance <= 0.010:
            raise SnakeStairGaitError(
                "crawl_goal_tolerance_m must be in [0.001, 0.010]"
            )
        transition_clearance = self._number(
            parameters,
            "transition_clearance_m",
            0.10 * staircase.rise_m,
        )
        if not 0.0 <= transition_clearance <= 0.015:
            raise SnakeStairGaitError(
                "transition_clearance_m must be in [0.0, 0.015]"
            )
        default_upper_riser_release = max(
            0.0,
            0.5 * spacing - transition_clearance,
        )
        upper_riser_edge_release_lead = self._number(
            parameters,
            "upper_riser_edge_release_lead_m",
            default_upper_riser_release,
        )
        if not 0.0 <= upper_riser_edge_release_lead <= 0.5 * spacing:
            raise SnakeStairGaitError(
                "upper_riser_edge_release_lead_m must be in "
                "[0.0, half one Snake8 link]"
            )
        head_prelift_lookahead = self._number(
            parameters,
            "head_prelift_lookahead_m",
            spacing + transition_clearance,
        )
        if not 0.040 <= head_prelift_lookahead <= 0.150:
            raise SnakeStairGaitError(
                "head_prelift_lookahead_m must be in [0.040, 0.150]"
            )
        head_prelift_ramp = self._number(
            parameters,
            "head_prelift_ramp_m",
            max(wheel_radius, 0.5 * spacing),
        )
        if not 0.010 <= head_prelift_ramp <= head_prelift_lookahead:
            raise SnakeStairGaitError(
                "head_prelift_ramp_m must be in [0.010, "
                "head_prelift_lookahead_m]"
            )
        head_hook_transfer = self._number(
            parameters,
            "head_hook_transfer_m",
            0.040,
        )
        if not 0.010 <= head_hook_transfer <= spacing:
            raise SnakeStairGaitError(
                "head_hook_transfer_m must be between 0.010 m and one "
                "Snake8 link"
            )
        available_head_clearance = spacing - staircase.rise_m
        default_head_clearance = min(
            0.010,
            max(0.0, available_head_clearance - 0.001),
        )
        head_overstep_clearance = self._number(
            parameters,
            "head_overstep_clearance_m",
            default_head_clearance,
        )
        if (
            head_overstep_clearance < 0.0
            or staircase.rise_m + head_overstep_clearance > spacing
        ):
            raise SnakeStairGaitError(
                "head_overstep_clearance_m must be non-negative and keep "
                "rise plus clearance within one Snake8 link"
            )
        head_lift_angle = math.asin(
            (staircase.rise_m + head_overstep_clearance) / spacing
        )
        approach_tolerance = self._number(
            parameters,
            "riser_approach_tolerance_m",
            0.010,
        )
        if not 0.003 <= approach_tolerance <= 0.030:
            raise SnakeStairGaitError(
                "riser_approach_tolerance_m must be in [0.003, 0.030]"
            )

        steps: list[BehaviorProgramStep] = []
        zero = (0.0,) * self.MODULE_COUNT
        if arch_wave:
            # Keep the assembled ground posture actively energized throughout
            # the initial approach.  These zero offsets are relative to the
            # captured post-assembly neutral values, never physical TILT zero.
            steps.append(
                BehaviorProgramStep(
                    phase="GEOM_LOCK_GROUND_NEUTRAL",
                    posture_targets=tuple(
                        JointTarget(
                            module_id=item.module_id,
                            joint="tilt",
                            angle_rad=0.0,
                            target_vertex_id=item.target_vertex_id,
                            target_role=item.target_role,
                            tolerance_rad=0.08,
                            coordination_group=(
                                "stair:GEOM_LOCK_GROUND_NEUTRAL"
                            ),
                            max_servo_error_rad=0.12,
                            angle_reference="captured_neutral",
                        )
                        for item in ordered
                    ),
                )
            )
            # Stop the flat head before the first riser.  Only after this
            # geometric barrier may the broad compliant rail begin lifting.
            first_riser_flat_lookahead = min(
                0.150,
                head_prelift_lookahead + wheel_radius,
            )
            steps.append(
                BehaviorProgramStep(
                    phase="GEOM_APPROACH_FIRST_RISER",
                    linear_m_s=approach_speed,
                    active_target_roles=tuple(
                        item.target_role for item in ordered
                    ),
                    position_goal=LongitudinalPositionGoal(
                        module_id=ordered[-1].module_id,
                        target_x_m=(
                            staircase.first_riser_x_m
                            - first_riser_flat_lookahead
                        ),
                        tolerance_m=approach_tolerance,
                    ),
                )
            )
        lifted = list(zero)
        if arch_wave:
            # Begin the rail with the same broad three-joint cell used on
            # every later riser.  The old one-link first lift reproduced the
            # validated gait but placed the underside of one cube directly
            # on the corner.  Two inclined links now share the requested rise
            # plus clearance before forward motion reaches the riser.
            first_arch_angle = self._distributed_rise_angle(
                staircase.rise_m + arch_clearance,
                spacing,
            )
            lifted[self.INITIAL_RISER_EDGE - 1] = first_arch_angle
            lifted[self.INITIAL_RISER_EDGE + 1] = -first_arch_angle
        else:
            lifted[self.INITIAL_RISER_EDGE] = bend_angle
        steps.append(
            self._posture(
                "LIFT_FIRST_RISER",
                zero,
                tuple(lifted),
                ordered,
            )
        )

        desired_elevated_center_x = (
            staircase.first_riser_x_m - wheel_radius
        )
        # In the compliant arch gait, use the actual head as the first-riser
        # geometric reference.  The legacy shoulder reference forces the
        # already-lifted cell to travel roughly three link lengths while
        # unsupported on flat ground and can self-jam before the stair.
        first_elevated = (
            ordered[-1]
            if arch_wave
            else ordered[self.INITIAL_RISER_EDGE + 1]
        )
        steps.append(
            BehaviorProgramStep(
                phase="APPROACH_FIRST_RISER",
                linear_m_s=approach_speed,
                active_target_roles=tuple(
                    item.target_role
                    for item in (
                        ordered if arch_wave else ordered[:5]
                    )
                ),
                position_goal=LongitudinalPositionGoal(
                    module_id=first_elevated.module_id,
                    target_x_m=desired_elevated_center_x,
                    tolerance_m=approach_tolerance,
                ),
            )
        )

        current = tuple(lifted)
        hooked = self._gait_offsets(
            phase=0,
            stair_count=len(staircase.top_heights_m),
            stride=stride,
            bend_angle=bend_angle,
            arch_wave=arch_wave,
            upper_bend_angle=self._distributed_rise_angle(
                staircase.rise_m,
                spacing,
            ),
        )
        if not arch_wave:
            steps.append(
                self._posture(
                    "CONFORM_PROFILE_00",
                    current,
                    hooked,
                    ordered,
                )
            )
            current = hooked
        base_current = hooked

        final_phase = self.INITIAL_RISER_EDGE + stride * (
            len(staircase.top_heights_m) - 1
        ) + 1
        micro_distance = spacing / substeps
        if crawl_tolerance >= 0.5 * micro_distance:
            raise SnakeStairGaitError(
                "crawl_goal_tolerance_m must be less than half one "
                "profile-substep distance"
            )
        for phase in range(final_phase):
            following = self._gait_offsets(
                phase=phase + 1,
                stair_count=len(staircase.top_heights_m),
                stride=stride,
                bend_angle=bend_angle,
                arch_wave=arch_wave,
                upper_bend_angle=self._distributed_rise_angle(
                    staircase.rise_m,
                    spacing,
                ),
            )
            active_roles = (
                tuple(item.target_role for item in ordered)
                if arch_wave
                else self._stable_support_roles(
                    phase,
                    phase + 1,
                    len(staircase.top_heights_m),
                    stride,
                    ordered,
                )
            )
            segment_start = base_current
            for substep in range(1, substeps + 1):
                fraction = substep / substeps
                settled_wave_progress = phase + (substep - 1) / substeps
                edge_lead_m = transition_clearance * math.sin(
                    math.pi * fraction
                )
                posture_fraction = min(
                    1.0,
                    fraction + edge_lead_m / spacing,
                )
                if arch_wave:
                    lifted_rise = staircase.rise_m + (
                        arch_clearance * math.sin(math.pi * fraction)
                    )
                    lifted_angle = self._distributed_rise_angle(
                        lifted_rise,
                        spacing,
                    )
                    arch_start = self._gait_offsets(
                        phase=phase,
                        stair_count=len(staircase.top_heights_m),
                        stride=stride,
                        bend_angle=bend_angle,
                        arch_wave=True,
                        upper_bend_angle=lifted_angle,
                    )
                    arch_end = self._gait_offsets(
                        phase=phase + 1,
                        stair_count=len(staircase.top_heights_m),
                        stride=stride,
                        bend_angle=bend_angle,
                        arch_wave=True,
                        upper_bend_angle=lifted_angle,
                    )
                    target = [
                        start + posture_fraction * (end - start)
                        for start, end in zip(arch_start, arch_end)
                    ]
                else:
                    target = [
                        start + posture_fraction * (end - start)
                        for start, end in zip(segment_start, following)
                    ]
                release_fraction = min(
                    1.0,
                    posture_fraction
                    + upper_riser_edge_release_lead
                    * math.sin(math.pi * fraction)
                    / spacing,
                )
                # Preserve the validated first-riser wave exactly.  Once the
                # chain spans multiple levels, release only the module beyond
                # the second (or a later) riser a little sooner.  This targets
                # the upper-edge BOTTOM-face jam without changing CRAWL_00_*.
                for stair_index in range(
                    1 if not arch_wave else len(staircase.top_heights_m),
                    len(staircase.top_heights_m),
                ):
                    outgoing_edge = (
                        self.INITIAL_RISER_EDGE
                        + stride * stair_index
                        - phase
                    )
                    if not 0 <= outgoing_edge <= self.MODULE_COUNT - 3:
                        continue
                    outgoing_module = outgoing_edge + 1
                    start = segment_start[outgoing_module]
                    end = following[outgoing_module]
                    if abs(end) + 1e-9 >= abs(start):
                        continue
                    target[outgoing_module] = (
                        start + release_fraction * (end - start)
                    )
                for stair_index in range(
                    1, len(staircase.top_heights_m)
                ):
                    if arch_wave:
                        arch_preview = self._arch_head_preview_angles(
                            wave_progress=phase + posture_fraction,
                            stair_index=stair_index,
                            stride=stride,
                            tread_depth_m=staircase.tread_depth_m,
                            spacing_m=spacing,
                            wheel_radius_m=wheel_radius,
                            support_guard_m=(
                                transition_clearance + crawl_tolerance
                            ),
                            lookahead_m=head_prelift_lookahead,
                            ramp_m=head_prelift_ramp,
                            hook_transfer_m=head_hook_transfer,
                            distributed_bend_angle_rad=(
                                self._distributed_rise_angle(
                                    staircase.rise_m,
                                    spacing,
                                )
                            ),
                            lift_bend_angle_rad=head_lift_angle,
                        )
                        if arch_preview is not None:
                            # The terminal hook is an absolute local profile,
                            # not an addition to the broad arch.  Overwriting
                            # these three free joints prevents double bends
                            # while it migrates v6/v7 -> v5/v7.
                            broad = max(
                                target[self.MODULE_COUNT - 3],
                                arch_preview[0],
                            )
                            terminal = arch_preview[1]
                            # Keep the opposing terminal target inside the
                            # live SMORES TILT range.  Preserve the head lift
                            # first and trim only the temporary broad overlay;
                            # that overlay returns as the hook migrates.
                            broad = min(
                                broad,
                                max(0.0, usable_tilt_limit - terminal),
                            )
                            target[self.MODULE_COUNT - 3:] = [
                                broad,
                                terminal,
                                -(broad + terminal),
                            ]
                        continue
                    terminal_preview, shoulder_preview = (
                        self._head_preview_angles(
                            wave_progress=settled_wave_progress,
                            profile_progress=phase + posture_fraction,
                            stair_index=stair_index,
                            stride=stride,
                            tread_depth_m=staircase.tread_depth_m,
                            spacing_m=spacing,
                            wheel_radius_m=wheel_radius,
                            support_guard_m=(
                                transition_clearance + crawl_tolerance
                            ),
                            lookahead_m=head_prelift_lookahead,
                            ramp_m=head_prelift_ramp,
                            hook_transfer_m=head_hook_transfer,
                            normal_bend_angle_rad=bend_angle,
                            lift_bend_angle_rad=head_lift_angle,
                        )
                    )
                    # First lift only the terminal module with the v6/v7
                    # pair, leaving v6 wheel-supported.  Once the head reaches
                    # the riser, migrate the hook rearward to v5/v6.
                    target[self.MODULE_COUNT - 2] += terminal_preview
                    target[self.MODULE_COUNT - 1] -= terminal_preview
                    target[self.MODULE_COUNT - 3] += shoulder_preview
                    target[self.MODULE_COUNT - 2] -= shoulder_preview
                target_tuple = tuple(target)
                prefix = "ARCH" if arch_wave else "PROFILE"
                label = f"{prefix}_{phase:02d}_{substep:02d}"
                steps.append(
                    self._posture(
                        label,
                        current,
                        target_tuple,
                        ordered,
                    )
                )
                reference, riser_x_m = self._edge_reference(
                    phase,
                    len(staircase.top_heights_m),
                    stride,
                    staircase,
                    ordered,
                )
                drive_phase = (
                    f"ARCH_DRIVE_{phase:02d}_{substep:02d}"
                    if arch_wave
                    else f"CRAWL_{phase:02d}_{substep:02d}"
                )
                target_x_m = (
                    riser_x_m
                    - wheel_radius
                    + fraction * spacing
                    - edge_lead_m
                )
                if arch_wave and substep == substeps:
                    # Before each upper riser, stop from the actual world-X
                    # pose of snake_head rather than inferring its clearance
                    # from an internal edge module.  The following program
                    # posture starts the recurring v6/v7 prelift.
                    upper_stair_index = phase // stride + 1
                    if (
                        phase == stride * (upper_stair_index - 1)
                        and upper_stair_index
                        < len(staircase.top_heights_m)
                    ):
                        reference = ordered[-1]
                        upper_riser_x_m = (
                            staircase.first_riser_x_m
                            + upper_stair_index
                            * staircase.tread_depth_m
                        )
                        target_x_m = (
                            upper_riser_x_m - head_prelift_lookahead
                        )
                        drive_phase = (
                            f"ARCH_HEAD_GATE_{upper_stair_index:02d}"
                        )
                    if phase == final_phase - 1:
                        # The final posture has already lifted the tail.  End
                        # the climb once the adjacent support wheel is one
                        # radius beyond the last riser; demanding another full
                        # link of travel leaves an elevated tail with no useful
                        # traction and makes all wheels spin on the top deck.
                        reference = ordered[1]
                        final_riser_x_m = (
                            staircase.first_riser_x_m
                            + (len(staircase.top_heights_m) - 1)
                            * staircase.tread_depth_m
                        )
                        target_x_m = final_riser_x_m + wheel_radius
                        drive_phase = "ARCH_TAIL_LIFT_COMPLETE"
                steps.append(
                    BehaviorProgramStep(
                        phase=drive_phase,
                        linear_m_s=crawl_speed,
                        active_target_roles=active_roles,
                        position_goal=LongitudinalPositionGoal(
                            module_id=reference.module_id,
                            target_x_m=target_x_m,
                            tolerance_m=crawl_tolerance,
                        ),
                    )
                )
                current = target_tuple
            base_current = following

        upper_deck_distance = self._number(
            parameters,
            "upper_deck_advance_distance_m",
            0.0 if arch_wave else spacing,
        )
        if arch_wave:
            if not 0.0 <= upper_deck_distance <= staircase.tread_depth_m:
                raise SnakeStairGaitError(
                    "upper_deck_advance_distance_m must be in [0.0, one "
                    "tread depth] for crawl_stairs_arch_wave"
                )
        elif not (
            0.5 * spacing
            <= upper_deck_distance
            <= staircase.tread_depth_m
        ):
            raise SnakeStairGaitError(
                "upper_deck_advance_distance_m must be between half one "
                "link and one tread depth"
            )
        if upper_deck_distance > 0.0:
            steps.append(
                BehaviorProgramStep(
                    phase="UPPER_DECK_ADVANCE",
                    linear_m_s=crawl_speed,
                    active_target_roles=tuple(
                        item.target_role for item in ordered
                    ),
                    displacement_goal=LongitudinalDisplacementGoal(
                        module_ids=tuple(
                            item.module_id for item in ordered
                        ),
                        distance_m=upper_deck_distance,
                        tolerance_m=crawl_tolerance,
                    ),
                )
            )
        return tuple(steps)

    def plan_arch_wave(
        self,
        graph: AttributedRobotGraph,
        assignments: Sequence[AssignedModule],
        parameters: Mapping[str, Any],
        neutral_tilt_rad_by_module: Mapping[str, float] | None = None,
    ) -> tuple[BehaviorProgramStep, ...]:
        """Run a geometry-scaled compliant arch rail with concurrent drive.

        World-X barriers and the phase stride retain the validated geometric
        gait, while every riser now uses a clearance arch distributed over
        two links.  The moving TILT group passes the cell toward the tail.
        A short structural support remains immediately behind the cell while
        the remote tail stays backdrivable to permit load transfer.
        """

        synchronized_speed = self._number(
            parameters,
            "synchronized_linear_m_s",
            self._number(
                parameters,
                "minimum_traction_linear_m_s",
                0.020,
            ),
        )
        if not 0.010 <= synchronized_speed <= 0.040:
            raise SnakeStairGaitError(
                "synchronized_linear_m_s must be in [0.010, 0.040]"
            )
        max_wave_tilt_speed = self._number(
            parameters, "max_wave_tilt_speed_rad_s", 0.45
        )
        if not 0.15 <= max_wave_tilt_speed <= 1.0:
            raise SnakeStairGaitError(
                "max_wave_tilt_speed_rad_s must be in [0.15, 1.0]"
            )
        loaded_tilt_tolerance = self._number(
            parameters, "loaded_tilt_tolerance_rad", 0.025
        )
        if not 0.015 <= loaded_tilt_tolerance <= 0.040:
            raise SnakeStairGaitError(
                "loaded_tilt_tolerance_rad must be in [0.015, 0.040]"
            )

        rail_program = self.plan(
            graph,
            assignments,
            parameters,
            arch_wave=True,
        )
        self._validate_program_neutral_limits(
            graph,
            assignments,
            rail_program,
            neutral_tilt_rad_by_module,
            self._number(parameters, "tilt_limit_margin_rad", 0.030),
        )
        synchronized = self._synchronize_arch_rail_program(
            rail_program,
            synchronized_speed_m_s=synchronized_speed,
            max_tilt_speed_rad_s=max_wave_tilt_speed,
            tilt_tolerance_rad=loaded_tilt_tolerance,
        )
        trailing_support_modules = self._integer(
            parameters, "trailing_support_modules", 1, 0, 3
        )
        return apply_trailing_compliance(
            synchronized,
            assignments,
            trailing_support_modules=trailing_support_modules,
        )

    @staticmethod
    def _synchronize_arch_rail_program(
        program: Sequence[BehaviorProgramStep],
        *,
        synchronized_speed_m_s: float,
        max_tilt_speed_rad_s: float,
        tilt_tolerance_rad: float,
    ) -> tuple[BehaviorProgramStep, ...]:
        """Pair each rail posture with its following geometric drive."""

        synchronized: list[BehaviorProgramStep] = []
        index = 0
        while index < len(program):
            posture = program[index]
            if (
                posture.kind == "posture"
                and index + 1 < len(program)
                and program[index + 1].kind == "drive"
            ):
                drive = program[index + 1]
                rail_motion = drive.phase not in {
                    "GEOM_APPROACH_FIRST_RISER",
                    "APPROACH_FIRST_RISER",
                }
                targets = tuple(
                    replace(
                        target,
                        tolerance_rad=tilt_tolerance_rad,
                        max_servo_error_rad=0.06,
                        max_servo_speed_rad_s=(
                            max_tilt_speed_rad_s
                            if target.max_servo_speed_rad_s is None
                            else min(
                                target.max_servo_speed_rad_s,
                                max_tilt_speed_rad_s,
                            )
                        ),
                    )
                    for target in posture.posture_targets
                )
                synchronized.append(
                    replace(
                        drive,
                        posture_targets=targets,
                        linear_m_s=min(
                            drive.linear_m_s,
                            synchronized_speed_m_s,
                        ),
                        continuous_with_next=(
                            rail_motion
                            and drive.phase != "ARCH_TAIL_LIFT_COMPLETE"
                        ),
                        hold_locomotion_until_admitted=not rail_motion,
                        posture_reached_linear_m_s=drive.linear_m_s,
                    )
                )
                index += 2
                continue
            synchronized.append(posture)
            index += 1
        return tuple(synchronized)

    def _validate_program_neutral_limits(
        self,
        graph: AttributedRobotGraph,
        assignments: Sequence[AssignedModule],
        program: Sequence[BehaviorProgramStep],
        neutral_tilt_rad_by_module: Mapping[str, float] | None,
        tilt_limit_margin_rad: float,
    ) -> None:
        ordered = tuple(sorted(assignments, key=self._vertex_index))
        bounds = self._tilt_offset_bounds(
            graph,
            ordered,
            neutral_tilt_rad_by_module,
            tilt_limit_margin_rad,
        )
        bounds_by_module = {
            assignment.module_id: bound
            for assignment, bound in zip(ordered, bounds)
        }
        for step in program:
            for target in step.posture_targets:
                lower, upper = bounds_by_module[target.module_id]
                if not lower - 1e-9 <= target.angle_rad <= upper + 1e-9:
                    raise SnakeStairGaitError(
                        f"{step.phase} exceeds the captured-neutral TILT "
                        f"range for {target.module_id}"
                    )

    def _tilt_offset_bounds(
        self,
        graph: AttributedRobotGraph,
        assignments: Sequence[AssignedModule],
        neutral_tilt_rad_by_module: Mapping[str, float] | None,
        margin_rad: float,
    ) -> tuple[tuple[float, float], ...]:
        """Return safe relative limits around the captured physical neutral."""

        nodes = graph.node_by_id()
        fallback_limit = self._symmetric_tilt_limit(graph, assignments)
        neutrals = neutral_tilt_rad_by_module or {}
        bounds: list[tuple[float, float]] = []
        for assignment in assignments:
            node = nodes.get(assignment.module_id)
            tilt: Mapping[str, Any] | None = None
            if node is not None:
                actuators = node.attributes.get("actuators")
                if isinstance(actuators, Mapping):
                    raw_tilt = actuators.get("tilt")
                    if isinstance(raw_tilt, Mapping):
                        tilt = raw_tilt
            try:
                lower = float(
                    tilt["lower_limit_rad"]
                    if tilt is not None
                    else -fallback_limit
                )
                upper = float(
                    tilt["upper_limit_rad"]
                    if tilt is not None
                    else fallback_limit
                )
                neutral = float(
                    neutrals.get(
                        assignment.module_id,
                        tilt.get("position_rad", 0.0)
                        if tilt is not None
                        else 0.0,
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise SnakeStairGaitError(
                    "Invalid live TILT limits or captured neutral"
                ) from error
            if not all(
                math.isfinite(value)
                for value in (lower, upper, neutral)
            ):
                raise SnakeStairGaitError(
                    "Non-finite live TILT limits or captured neutral"
                )
            relative_lower = lower + margin_rad - neutral
            relative_upper = upper - margin_rad - neutral
            if relative_lower > 0.0 or relative_upper < 0.0:
                raise SnakeStairGaitError(
                    "Captured neutral posture is outside the safe TILT "
                    f"range for {assignment.module_id}"
                )
            bounds.append((relative_lower, relative_upper))
        return tuple(bounds)

    def _head_preview_angles(
        self,
        *,
        wave_progress: float,
        profile_progress: float,
        stair_index: int,
        stride: int,
        tread_depth_m: float,
        spacing_m: float,
        wheel_radius_m: float,
        support_guard_m: float,
        lookahead_m: float,
        ramp_m: float,
        hook_transfer_m: float,
        normal_bend_angle_rad: float,
        lift_bend_angle_rad: float,
    ) -> tuple[float, float]:
        """Over-lift the head, hold it, then migrate its hook rearward."""
        tread_start = stride * (stair_index - 1)
        head_contact_progress = tread_start + (
            tread_depth_m - 2.0 * spacing_m
        ) / spacing_m
        lookahead_progress = head_contact_progress - (
            lookahead_m / spacing_m
        )
        support_fully_on_tread_progress = tread_start + (
            (2.0 * wheel_radius_m + support_guard_m) / spacing_m
        )
        preview_start = max(
            lookahead_progress,
            support_fully_on_tread_progress,
        )
        lift = self._clamp01(
            (wave_progress - preview_start) * spacing_m / ramp_m
        )
        hook_transfer = self._clamp01(
            (wave_progress - head_contact_progress)
            * spacing_m
            / hook_transfer_m
        )
        terminal_preview = (
            lift_bend_angle_rad * lift * (1.0 - hook_transfer)
        )

        # Cross-fade the migrated hook into the ordinary edge-5 profile
        # instead of adding the same bend twice when the wave catches up.  As
        # that natural bend grows, remove the temporary overstep clearance so
        # the neck wheel settles back onto the tread before the next cycle.
        natural_entry = (
            self.INITIAL_RISER_EDGE
            + stride * stair_index
            - (self.MODULE_COUNT - 3)
        )
        natural = self._clamp01(
            profile_progress - (natural_entry - 1.0)
        )
        settled_hook_angle = lift_bend_angle_rad - natural * (
            lift_bend_angle_rad - normal_bend_angle_rad
        )
        desired_shoulder_hook = (
            settled_hook_angle * lift * hook_transfer
        )
        shoulder_preview = max(
            0.0,
            desired_shoulder_hook - normal_bend_angle_rad * natural,
        )
        return terminal_preview, shoulder_preview

    def _arch_head_preview_angles(
        self,
        *,
        wave_progress: float,
        stair_index: int,
        stride: int,
        tread_depth_m: float,
        spacing_m: float,
        wheel_radius_m: float,
        support_guard_m: float,
        lookahead_m: float,
        ramp_m: float,
        hook_transfer_m: float,
        distributed_bend_angle_rad: float,
        lift_bend_angle_rad: float,
    ) -> list[float] | None:
        """Lift the live head before an upper riser, then broaden its hook.

        Program drives terminate on measured world-X module positions, so
        ``wave_progress`` advances only after a geometric barrier is reached.
        The preview therefore has no time assumption.  Its onset is derived
        from tread depth, chain spacing, wheel radius and the requested
        clearance; it repeats for every upper riser.
        """

        tread_start = stride * (stair_index - 1)
        head_contact_progress = tread_start + (
            tread_depth_m - 2.0 * spacing_m
        ) / spacing_m
        preview_complete_progress = math.ceil(head_contact_progress)
        if wave_progress > preview_complete_progress:
            # The head wheel has crossed the riser under a live world-X goal.
            # Stop pinning the terminal v5/v7 arch: the ordinary moving arch
            # now carries the bend rearward, lowers v7 onto the tread and
            # restores its wheel contact.  Keeping this preview alive would
            # superimpose it on the natural arch and lift the head again.
            return None
        support_fully_on_tread_progress = tread_start + (
            (2.0 * wheel_radius_m + support_guard_m) / spacing_m
        )
        # Do not alter the already validated first-riser transfer.  At its
        # endpoint the head still has about one link of geometric lookahead,
        # after which the terminal module may rise without stealing support
        # from the modules that are climbing the first edge.
        preview_start = max(
            head_contact_progress - lookahead_m / spacing_m,
            support_fully_on_tread_progress,
            tread_start + 1.0,
        )
        lift = self._clamp01(
            (wave_progress - preview_start) * spacing_m / ramp_m
        )
        if lift <= 0.0:
            return None

        # Complete the terminal-to-broad cross-fade by the next integer wave
        # boundary even when the user requests a longer generic hook transfer.
        # This leaves the ordinary +A,0,-A arch as the settled profile.
        available_transfer_m = max(
            1e-6,
            preview_complete_progress * spacing_m
            - head_contact_progress * spacing_m,
        )
        transfer_distance_m = min(
            hook_transfer_m,
            available_transfer_m,
        )
        transfer = self._clamp01(
            (wave_progress - head_contact_progress)
            * spacing_m
            / transfer_distance_m
        )
        terminal = lift_bend_angle_rad * lift * (1.0 - transfer)
        broad = distributed_bend_angle_rad * lift * transfer
        return [broad, terminal, -(broad + terminal)]

    @staticmethod
    def _clamp01(value: float) -> float:
        return min(1.0, max(0.0, value))

    def _edge_reference(
        self,
        phase: int,
        stair_count: int,
        stride: int,
        staircase: UniformStaircase,
        assignments: Sequence[AssignedModule],
    ) -> tuple[AssignedModule, float]:
        """Select the foremost live wheel whose riser transfer is active."""
        candidates: list[tuple[int, int]] = []
        for stair_index in range(stair_count):
            old_edge = (
                self.INITIAL_RISER_EDGE + stride * stair_index - phase
            )
            new_edge = old_edge - 1
            if (
                0 <= old_edge <= self.MODULE_COUNT - 3
                or 0 <= new_edge <= self.MODULE_COUNT - 3
            ):
                reference_index = min(
                    self.MODULE_COUNT - 1,
                    max(0, old_edge + 1),
                )
                candidates.append((stair_index, reference_index))
        if not candidates:
            raise SnakeStairGaitError(
                "No stair edge reference during Snake8 transfer"
            )
        stair_index, reference_index = max(candidates)
        return (
            assignments[reference_index],
            staircase.first_riser_x_m
            + stair_index * staircase.tread_depth_m,
        )

    def profile_offsets(
        self,
        *,
        phase: int,
        stair_count: int,
        stride: int,
        bend_angle: float,
    ) -> tuple[float, ...]:
        """Return all relative TILT angles for one chain-progress phase."""
        offsets = [0.0] * self.MODULE_COUNT
        for stair_index in range(stair_count):
            edge = self.INITIAL_RISER_EDGE + stride * stair_index - phase
            # Two modules are needed beyond a riser: one turns vertically
            # and the next turns back onto the horizontal tread.  The earlier
            # head preview is cross-faded into this ordinary edge-5 profile.
            if 0 <= edge <= self.MODULE_COUNT - 3:
                offsets[edge] += bend_angle
                offsets[edge + 1] -= bend_angle
        return tuple(offsets)

    def arch_wave_offsets(
        self,
        *,
        phase: int,
        stair_count: int,
        stride: int,
        upper_bend_angle: float,
    ) -> tuple[float, ...]:
        """Return a broad moving arch passed rearward like a rail baton.

        Every riser uses ``+angle, 0, -angle``, distributing the height over
        two links and three consecutive TILT targets.  Incrementing ``phase``
        moves that cell one module toward the tail.  At the final tail
        boundary the cell shifts half a link forward, retaining the two-link
        rise instead of collapsing into a near-vertical tail hinge.
        """
        offsets = [0.0] * self.MODULE_COUNT
        for stair_index in range(stair_count):
            edge = self.INITIAL_RISER_EDGE + stride * stair_index - phase
            if 1 <= edge <= self.MODULE_COUNT - 2:
                offsets[edge - 1] += upper_bend_angle
                offsets[edge + 1] -= upper_bend_angle
            elif edge == 0:
                # Shift the last cell half a link forward instead of
                # collapsing it into the old near-vertical tail hinge.  The
                # last two links therefore share the rise while the rail
                # baton exits the chain and the tail settles onto the deck.
                offsets[0] += upper_bend_angle
                offsets[2] -= upper_bend_angle
        return tuple(offsets)

    def _gait_offsets(
        self,
        *,
        phase: int,
        stair_count: int,
        stride: int,
        bend_angle: float,
        arch_wave: bool,
        upper_bend_angle: float,
    ) -> tuple[float, ...]:
        if arch_wave:
            return self.arch_wave_offsets(
                phase=phase,
                stair_count=stair_count,
                stride=stride,
                upper_bend_angle=upper_bend_angle,
            )
        return self.profile_offsets(
            phase=phase,
            stair_count=stair_count,
            stride=stride,
            bend_angle=bend_angle,
        )

    @staticmethod
    def _distributed_rise_angle(rise_m: float, spacing_m: float) -> float:
        ratio = rise_m / (2.0 * spacing_m)
        if not 0.0 <= ratio < 1.0:
            raise SnakeStairGaitError(
                "Distributed stair rise exceeds two Snake8 links"
            )
        return math.asin(ratio)

    def _posture(
        self,
        phase: str,
        previous: tuple[float, ...],
        target: tuple[float, ...],
        assignments: Sequence[AssignedModule],
        *,
        motion_duration_s: float | None = None,
        motion_tolerance_rad: float = 0.025,
    ) -> BehaviorProgramStep:
        changed = [
            index
            for index, (old, new) in enumerate(zip(previous, target))
            if abs(old - new) > 1e-6
        ]
        targets = tuple(
            JointTarget(
                module_id=assignments[index].module_id,
                joint="tilt",
                angle_rad=target[index],
                target_vertex_id=assignments[index].target_vertex_id,
                target_role=assignments[index].target_role,
                tolerance_rad=(
                    motion_tolerance_rad
                    if motion_duration_s is not None
                    else 0.08
                ),
                coordination_group=f"stair:{phase}",
                max_servo_error_rad=(
                    0.06 if motion_duration_s is not None else 0.12
                ),
                max_servo_speed_rad_s=(
                    None
                    if motion_duration_s is None
                    else abs(target[index] - previous[index])
                    / motion_duration_s
                ),
                angle_reference="captured_neutral",
            )
            for index in changed
        )
        return BehaviorProgramStep(phase=phase, posture_targets=targets)

    def _stable_support_roles(
        self,
        first_phase: int,
        second_phase: int,
        stair_count: int,
        stride: int,
        assignments: Sequence[AssignedModule],
    ) -> tuple[str, ...]:
        first = self._support_indices(first_phase, stair_count, stride)
        second = self._support_indices(second_phase, stair_count, stride)
        # Wheels that support either endpoint of the shift must keep rolling.
        # In particular, a wheel entering or leaving riser contact would
        # otherwise become a passive obstacle against the tread edge.
        traction = sorted(first | second)
        if not traction:
            raise SnakeStairGaitError(
                "No stable wheel support during stair crawl"
            )
        return tuple(
            assignments[index].target_role for index in traction
        )

    def _support_indices(
        self, phase: int, stair_count: int, stride: int
    ) -> set[int]:
        rises = {
            self.INITIAL_RISER_EDGE + stride * stair_index - phase
            for stair_index in range(stair_count)
        }
        rises = {
            edge
            for edge in rises
            if 0 <= edge <= self.MODULE_COUNT - 3
        }
        vertical_modules = {edge + 1 for edge in rises}
        return {
            index
            for index in range(self.MODULE_COUNT)
            if index not in vertical_modules
        }

    @staticmethod
    def _ordered_positions(
        graph: AttributedRobotGraph,
        assignments: Sequence[AssignedModule],
    ) -> tuple[tuple[float, float, float], ...]:
        nodes = graph.node_by_id()
        try:
            return tuple(
                module_position(nodes[item.module_id].attributes)
                for item in assignments
            )
        except KeyError as error:
            raise SnakeStairGaitError(
                f"Missing live module pose: {error}"
            ) from error

    @staticmethod
    def _link_spacing(
        positions: Sequence[tuple[float, float, float]],
    ) -> float:
        distances = [
            math.dist(first, second)
            for first, second in zip(positions, positions[1:])
        ]
        spacing = median(distances)
        if not math.isfinite(spacing) or not 0.060 <= spacing <= 0.100:
            raise SnakeStairGaitError(
                f"Invalid Snake8 link spacing {spacing:.4f} m"
            )
        return spacing

    @staticmethod
    def _wheel_radius(graph: AttributedRobotGraph) -> float:
        raw_geometry = graph.global_attributes.get("module_geometry")
        if not isinstance(raw_geometry, Mapping):
            raise SnakeStairGaitError(
                "Robot graph has no module geometry metadata"
            )
        try:
            radius = float(raw_geometry["wheel_radius_m"])
        except (KeyError, TypeError, ValueError) as error:
            raise SnakeStairGaitError(
                "Invalid SMORES wheel radius metadata"
            ) from error
        if not math.isfinite(radius) or not 0.020 <= radius <= 0.050:
            raise SnakeStairGaitError(
                f"Invalid SMORES wheel radius {radius:.4f} m"
            )
        return radius

    @staticmethod
    def _symmetric_tilt_limit(
        graph: AttributedRobotGraph,
        assignments: Sequence[AssignedModule],
    ) -> float:
        """Return the smallest observed absolute TILT limit in the chain."""

        nodes = graph.node_by_id()
        limits: list[float] = []
        for assignment in assignments:
            node = nodes.get(assignment.module_id)
            if node is None:
                continue
            actuators = node.attributes.get("actuators")
            if not isinstance(actuators, Mapping):
                continue
            tilt = actuators.get("tilt")
            if not isinstance(tilt, Mapping):
                continue
            try:
                lower = float(tilt["lower_limit_rad"])
                upper = float(tilt["upper_limit_rad"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(lower) and math.isfinite(upper):
                limits.append(min(-lower, upper))
        limit = min(limits, default=0.5 * math.pi)
        if not 0.5 <= limit <= math.pi:
            raise SnakeStairGaitError(
                f"Invalid SMORES TILT limit {limit:.4f} rad"
            )
        return limit

    @staticmethod
    def _vertex_index(assignment: AssignedModule) -> int:
        try:
            return int(assignment.target_vertex_id.removeprefix("v"))
        except ValueError as error:
            raise SnakeStairGaitError(
                "Snake8 vertices must be v0..v7"
            ) from error

    @staticmethod
    def _number(
        parameters: Mapping[str, Any], name: str, default: float
    ) -> float:
        try:
            value = float(parameters.get(name, default))
        except (TypeError, ValueError) as error:
            raise SnakeStairGaitError(f"{name} must be numeric") from error
        if not math.isfinite(value):
            raise SnakeStairGaitError(f"{name} must be finite")
        return value

    def _speed(
        self, parameters: Mapping[str, Any], name: str, default: float
    ) -> float:
        value = self._number(parameters, name, default)
        if value <= 0.0 or value > 0.060:
            raise SnakeStairGaitError(f"{name} must be in (0, 0.060]")
        return value

    @staticmethod
    def _integer(
        parameters: Mapping[str, Any],
        name: str,
        default: int,
        low: int,
        high: int,
    ) -> int:
        raw = parameters.get(name, default)
        try:
            value = int(raw)
            numeric = float(raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise SnakeStairGaitError(
                f"{name} must be an integer"
            ) from error
        if (
            isinstance(raw, bool)
            or not math.isfinite(numeric)
            or value != numeric
        ):
            raise SnakeStairGaitError(f"{name} must be an integer")
        if not low <= value <= high:
            raise SnakeStairGaitError(f"{name} must be in [{low}, {high}]")
        return value
