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

    def plan(
        self,
        graph: AttributedRobotGraph,
        assignments: Sequence[AssignedModule],
        parameters: Mapping[str, Any],
    ) -> tuple[BehaviorProgramStep, ...]:
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
        substeps = self._integer(parameters, "profile_substeps", 3, 1, 6)
        approach_speed = self._speed(
            parameters, "riser_approach_linear_m_s", 0.060
        )
        crawl_speed = self._speed(parameters, "linear_m_s", 0.030)
        slip = self._number(parameters, "slip_compensation", 1.5)
        if slip < 1.0 or slip > 4.0:
            raise SnakeStairGaitError("slip_compensation must be in [1, 4]")

        steps: list[BehaviorProgramStep] = []
        zero = (0.0,) * self.MODULE_COUNT
        lifted = list(zero)
        lifted[self.INITIAL_RISER_EDGE] = bend_angle
        steps.append(
            self._posture("LIFT_FIRST_RISER", zero, tuple(lifted), ordered)
        )

        desired_bend_x = staircase.first_riser_x_m - 0.5 * diagonal_run
        approach_distance = max(
            0.0,
            desired_bend_x - positions[self.INITIAL_RISER_EDGE][0],
        )
        raw_approach_duration = parameters.get("riser_approach_duration_s")
        approach_duration = (
            self._number(parameters, "riser_approach_duration_s", 0.0)
            if raw_approach_duration is not None
            else slip * approach_distance / approach_speed
        )
        if approach_duration < 0.0:
            raise SnakeStairGaitError(
                "riser_approach_duration_s must be non-negative"
            )
        if approach_duration > 1e-3:
            steps.append(
                BehaviorProgramStep(
                    phase="APPROACH_FIRST_RISER",
                    duration_s=approach_duration,
                    linear_m_s=approach_speed,
                    active_target_roles=tuple(
                        item.target_role for item in ordered[:5]
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

        final_phase = self.INITIAL_RISER_EDGE + stride * (
            len(staircase.top_heights_m) - 1
        ) + 1
        micro_distance = spacing / substeps
        micro_duration = slip * micro_distance / crawl_speed
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
            segment_start = current
            for substep in range(1, substeps + 1):
                fraction = substep / substeps
                target = tuple(
                    start + fraction * (end - start)
                    for start, end in zip(segment_start, following)
                )
                label = f"PROFILE_{phase:02d}_{substep:02d}"
                steps.append(self._posture(label, current, target, ordered))
                steps.append(
                    BehaviorProgramStep(
                        phase=f"CRAWL_{phase:02d}_{substep:02d}",
                        duration_s=micro_duration,
                        linear_m_s=crawl_speed,
                        active_target_roles=active_roles,
                    )
                )
                current = target

        final_duration = self._number(
            parameters, "tread_advance_duration_s", 4.0
        )
        if final_duration <= 0.0:
            raise SnakeStairGaitError(
                "tread_advance_duration_s must be positive"
            )
        steps.append(
            BehaviorProgramStep(
                phase="UPPER_DECK_ADVANCE",
                duration_s=final_duration,
                linear_m_s=crawl_speed,
                active_target_roles=tuple(
                    item.target_role for item in ordered
                ),
            )
        )
        return tuple(steps)

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
            # and the next turns back onto the horizontal tread.
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
        stable = sorted(first & second)
        if not stable:
            raise SnakeStairGaitError(
                "No stable wheel support during stair crawl"
            )
        return tuple(assignments[index].target_role for index in stable)

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
        first_edge = self.INITIAL_RISER_EDGE - phase
        minimum = first_edge + 2 if first_edge >= 0 else 0
        vertical_modules = {edge + 1 for edge in rises}
        return {
            index
            for index in range(max(0, minimum), self.MODULE_COUNT)
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
