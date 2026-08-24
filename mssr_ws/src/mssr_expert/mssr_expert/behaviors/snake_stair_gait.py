"""Deterministic follow-the-leader gait for a serial Snake8 on stairs."""

from __future__ import annotations

from dataclasses import dataclass
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
        crawl_speed = self._speed(parameters, "linear_m_s", 0.030)
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
        head_prelift_lookahead = self._number(
            parameters,
            "head_prelift_lookahead_m",
            0.080,
        )
        if not 0.040 <= head_prelift_lookahead <= 0.150:
            raise SnakeStairGaitError(
                "head_prelift_lookahead_m must be in [0.040, 0.150]"
            )
        head_prelift_ramp = self._number(
            parameters,
            "head_prelift_ramp_m",
            0.040,
        )
        if not 0.010 <= head_prelift_ramp <= head_prelift_lookahead:
            raise SnakeStairGaitError(
                "head_prelift_ramp_m must be in [0.010, "
                "head_prelift_lookahead_m]"
            )

        steps: list[BehaviorProgramStep] = []
        zero = (0.0,) * self.MODULE_COUNT
        lifted = list(zero)
        lifted[self.INITIAL_RISER_EDGE] = bend_angle
        steps.append(
            self._posture("LIFT_FIRST_RISER", zero, tuple(lifted), ordered)
        )

        desired_elevated_center_x = (
            staircase.first_riser_x_m - wheel_radius
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
        first_elevated = ordered[self.INITIAL_RISER_EDGE + 1]
        steps.append(
            BehaviorProgramStep(
                phase="APPROACH_FIRST_RISER",
                linear_m_s=approach_speed,
                active_target_roles=tuple(
                    item.target_role for item in ordered[:5]
                ),
                position_goal=LongitudinalPositionGoal(
                    module_id=first_elevated.module_id,
                    target_x_m=desired_elevated_center_x,
                    tolerance_m=approach_tolerance,
                ),
            )
        )

        current = tuple(lifted)
        hooked = self.profile_offsets(
            phase=0,
            stair_count=len(staircase.top_heights_m),
            stride=stride,
            bend_angle=bend_angle,
        )
        steps.append(
            self._posture("CONFORM_PROFILE_00", current, hooked, ordered)
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
            following = self.profile_offsets(
                phase=phase + 1,
                stair_count=len(staircase.top_heights_m),
                stride=stride,
                bend_angle=bend_angle,
            )
            active_roles = self._stable_support_roles(
                phase,
                phase + 1,
                len(staircase.top_heights_m),
                stride,
                ordered,
            )
            segment_start = base_current
            for substep in range(1, substeps + 1):
                fraction = substep / substeps
                wave_progress = phase + fraction
                edge_lead_m = transition_clearance * math.sin(
                    math.pi * fraction
                )
                posture_fraction = min(
                    1.0,
                    fraction + edge_lead_m / spacing,
                )
                target = [
                    start + posture_fraction * (end - start)
                    for start, end in zip(segment_start, following)
                ]
                for stair_index in range(
                    1, len(staircase.top_heights_m)
                ):
                    preview_fraction = self._head_preview_fraction(
                        wave_progress=wave_progress,
                        profile_progress=phase + posture_fraction,
                        stair_index=stair_index,
                        stride=stride,
                        tread_depth_m=staircase.tread_depth_m,
                        spacing_m=spacing,
                        wheel_radius_m=wheel_radius,
                        lookahead_m=head_prelift_lookahead,
                        ramp_m=head_prelift_ramp,
                    )
                    target[self.MODULE_COUNT - 3] += (
                        bend_angle * preview_fraction
                    )
                    target[self.MODULE_COUNT - 2] -= (
                        bend_angle * preview_fraction
                    )
                target_tuple = tuple(target)
                label = f"PROFILE_{phase:02d}_{substep:02d}"
                steps.append(
                    self._posture(label, current, target_tuple, ordered)
                )
                reference, riser_x_m = self._edge_reference(
                    phase,
                    len(staircase.top_heights_m),
                    stride,
                    staircase,
                    ordered,
                )
                steps.append(
                    BehaviorProgramStep(
                        phase=f"CRAWL_{phase:02d}_{substep:02d}",
                        linear_m_s=crawl_speed,
                        active_target_roles=active_roles,
                        position_goal=LongitudinalPositionGoal(
                            module_id=reference.module_id,
                            target_x_m=(
                                riser_x_m
                                - wheel_radius
                                + fraction * spacing
                                - edge_lead_m
                            ),
                            tolerance_m=crawl_tolerance,
                        ),
                    )
                )
                current = target_tuple
            base_current = following

        upper_deck_distance = self._number(
            parameters, "upper_deck_advance_distance_m", spacing
        )
        if not 0.5 * spacing <= upper_deck_distance <= staircase.tread_depth_m:
            raise SnakeStairGaitError(
                "upper_deck_advance_distance_m must be between half one "
                "link and one tread depth"
            )
        steps.append(
            BehaviorProgramStep(
                phase="UPPER_DECK_ADVANCE",
                linear_m_s=crawl_speed,
                active_target_roles=tuple(
                    item.target_role for item in ordered
                ),
                displacement_goal=LongitudinalDisplacementGoal(
                    module_ids=tuple(item.module_id for item in ordered),
                    distance_m=upper_deck_distance,
                    tolerance_m=crawl_tolerance,
                ),
            )
        )
        return tuple(steps)

    def _head_preview_fraction(
        self,
        *,
        wave_progress: float,
        profile_progress: float,
        stair_index: int,
        stride: int,
        tread_depth_m: float,
        spacing_m: float,
        wheel_radius_m: float,
        lookahead_m: float,
        ramp_m: float,
    ) -> float:
        """Pre-lift neck/head after the shoulder is fully on its tread."""
        tread_start = stride * (stair_index - 1)
        head_contact_progress = tread_start + (
            tread_depth_m - 2.0 * spacing_m
        ) / spacing_m
        lookahead_progress = head_contact_progress - (
            lookahead_m / spacing_m
        )
        shoulder_supported_progress = tread_start + (
            2.0 * wheel_radius_m / spacing_m
        )
        preview_start = max(
            lookahead_progress,
            shoulder_supported_progress,
        )
        preview = self._clamp01(
            (wave_progress - preview_start) * spacing_m / ramp_m
        )

        # Cross-fade the preview into the ordinary edge-5 profile instead of
        # adding the same bend twice when the traveling wave catches up.
        natural_entry = (
            self.INITIAL_RISER_EDGE
            + stride * stair_index
            - (self.MODULE_COUNT - 3)
        )
        natural = self._clamp01(
            profile_progress - (natural_entry - 1.0)
        )
        return max(0.0, preview - natural)

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

    def _posture(
        self,
        phase: str,
        previous: tuple[float, ...],
        target: tuple[float, ...],
        assignments: Sequence[AssignedModule],
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
                tolerance_rad=0.08,
                coordination_group=f"stair:{phase}",
                max_servo_error_rad=0.12,
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
