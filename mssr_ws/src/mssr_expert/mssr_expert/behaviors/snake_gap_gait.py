"""Geometric head-span-land-tail gait for a Snake8 crossing a flat gap."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Any, Mapping, Sequence

from mssr_expert.behaviors.morphology_library import (
    AssignedModule,
    BehaviorProgramStep,
    JointTarget,
    LongitudinalPositionGoal,
)
from mssr_expert.graph.attributed_robot_graph import AttributedRobotGraph
from mssr_expert.primitives.common import module_position


class SnakeGapGaitError(ValueError):
    """Raised when live geometry cannot define a safe Snake8 gap gait."""


@dataclass(frozen=True)
class FlatGap:
    """A world-X gap whose near and far banks have the same height."""

    near_edge_x_m: float
    far_edge_x_m: float

    @property
    def width_m(self) -> float:
        return self.far_edge_x_m - self.near_edge_x_m

    @classmethod
    def from_course(cls, course: Mapping[str, Any]) -> "FlatGap":
        if course.get("frame_id") != "world":
            raise SnakeGapGaitError("Gap landmarks must use world frame")
        raw = course.get("gap")
        if not isinstance(raw, Mapping):
            raise SnakeGapGaitError("Course has no gap landmarks")
        try:
            near = float(raw["near_edge_x_m"])
            far = float(raw["far_edge_x_m"])
        except (KeyError, TypeError, ValueError) as error:
            raise SnakeGapGaitError("Invalid gap landmarks") from error
        if not all(math.isfinite(value) for value in (near, far)) or far <= near:
            raise SnakeGapGaitError("Gap edges must be finite and strictly ordered")
        declared_width = raw.get("width_m")
        if declared_width is not None:
            try:
                width = float(declared_width)
            except (TypeError, ValueError) as error:
                raise SnakeGapGaitError("Invalid declared gap width") from error
            if not math.isfinite(width) or not math.isclose(
                width, far - near, abs_tol=1e-6
            ):
                raise SnakeGapGaitError(
                    "Gap width is inconsistent with its edges"
                )
        return cls(near, far)


class SnakeGapGaitPlanner:
    """Plan lift-head, span, land, advance, lift-tail and clear phases."""

    MODULE_COUNT = 8
    TIMED_PARAMETER_NAMES = frozenset(
        {
            "duration_s",
            "span_duration_s",
            "tail_clear_duration_s",
            "head_lift_duration_s",
            "body_advance_duration_s",
        }
    )

    def plan(
        self,
        graph: AttributedRobotGraph,
        assignments: Sequence[AssignedModule],
        parameters: Mapping[str, Any],
    ) -> tuple[BehaviorProgramStep, ...]:
        timed = sorted(self.TIMED_PARAMETER_NAMES.intersection(parameters))
        if timed:
            raise SnakeGapGaitError(
                "gap_crossing uses geometric world-pose goals and does not "
                "accept timed parameters: " + ", ".join(timed)
            )
        ordered = tuple(sorted(assignments, key=self._vertex_index))
        if len(ordered) != self.MODULE_COUNT:
            raise SnakeGapGaitError("Geometric gap gait requires Snake8")
        if tuple(self._vertex_index(item) for item in ordered) != tuple(range(8)):
            raise SnakeGapGaitError("Snake8 vertices must be exactly v0..v7")

        course = graph.global_attributes.get("course")
        if not isinstance(course, Mapping):
            raise SnakeGapGaitError("Robot graph has no course metadata")
        gap = FlatGap.from_course(course)
        positions = self._ordered_positions(graph, ordered)
        spacing = self._link_spacing(positions)
        wheel_radius = self._wheel_radius(graph)
        heading = math.atan2(
            positions[-1][1] - positions[0][1],
            positions[-1][0] - positions[0][0],
        )
        max_heading = self._number(parameters, "max_alignment_error_rad", 0.35)
        if not 0.0 < max_heading <= math.pi:
            raise SnakeGapGaitError("max_alignment_error_rad must be in (0, pi]")
        if abs(heading) > max_heading:
            raise SnakeGapGaitError(
                "Snake8 is not aligned with the +X gap: "
                f"heading={heading:.3f} rad"
            )

        edge_clearance = self._number(parameters, "edge_clearance_m", 0.006)
        landing_margin = self._number(parameters, "landing_margin_m", 0.006)
        goal_tolerance = self._number(parameters, "gap_goal_tolerance_m", 0.004)
        if not 0.002 <= edge_clearance <= 0.020:
            raise SnakeGapGaitError("edge_clearance_m must be in [0.002, 0.020]")
        if not 0.002 <= landing_margin <= 0.020:
            raise SnakeGapGaitError("landing_margin_m must be in [0.002, 0.020]")
        if not 0.001 <= goal_tolerance <= 0.010:
            raise SnakeGapGaitError(
                "gap_goal_tolerance_m must be in [0.001, 0.010]"
            )

        required_span = (
            gap.width_m
            + 2.0 * wheel_radius
            + edge_clearance
            + landing_margin
        )
        lifted_count = max(2, math.ceil(required_span / spacing))
        maximum_lifted_count = self.MODULE_COUNT - 3
        if lifted_count > maximum_lifted_count:
            maximum_gap = (
                maximum_lifted_count * spacing
                - 2.0 * wheel_radius
                - edge_clearance
                - landing_margin
            )
            raise SnakeGapGaitError(
                f"Gap width {gap.width_m:.3f} m exceeds the safe Snake8 "
                f"drawbridge span {maximum_gap:.3f} m"
            )
        approach_speed = self._speed(parameters, "approach_linear_m_s", 0.050)
        crossing_speed = self._speed(parameters, "linear_m_s", 0.040)

        lift_angle = self._number(
            parameters,
            "drawbridge_lift_angle_rad",
            1.20,
        )
        if lift_angle < 0.85:
            raise SnakeGapGaitError(
                "drawbridge_lift_angle_rad must be at least 0.85 rad"
            )
        tilt_margin = self._number(parameters, "tilt_limit_margin_rad", 0.030)
        if not 0.0 <= tilt_margin <= 0.15:
            raise SnakeGapGaitError("tilt_limit_margin_rad must be in [0, 0.15]")
        usable_limit = self._symmetric_tilt_limit(graph, ordered) - tilt_margin
        if lift_angle > usable_limit:
            raise SnakeGapGaitError(
                f"Required gap lift {lift_angle:.3f} rad exceeds usable "
                f"TILT limit {usable_limit:.3f} rad"
            )

        roles = tuple(item.target_role for item in ordered)
        neutral = (0.0,) * self.MODULE_COUNT
        head_pivot_index = self.MODULE_COUNT - lifted_count - 1
        tail_hinge_index = lifted_count - 1
        front_anchor_index = tail_hinge_index + 1
        head_lift = list(neutral)
        # One large hinge angle raises the complete terminal segment as a
        # rigid drawbridge.  No downstream counter-bend may flatten it.
        head_lift[head_pivot_index] = lift_angle
        tail_lift = list(neutral)
        # Spatial mirror: the already landed front segment anchors an equal
        # number of tail modules while they are carried across the opening.
        tail_lift[tail_hinge_index] = -lift_angle

        pivot_at_near_edge = (
            gap.near_edge_x_m - wheel_radius - edge_clearance
        )
        front_anchor_on_far_bank = (
            gap.far_edge_x_m + wheel_radius + landing_margin
        )
        front_anchor_for_safe_tail_landing = (
            gap.far_edge_x_m
            + lifted_count * spacing
            + wheel_radius
            + landing_margin
        )
        tail_on_far_bank = (
            gap.far_edge_x_m + wheel_radius + landing_margin
        )

        return (
            self._posture("RESTORE_GAP_NEUTRAL", neutral, neutral, ordered, all_targets=True),
            self._posture(
                "LIFT_HEAD_DRAWBRIDGE",
                neutral,
                tuple(head_lift),
                ordered,
            ),
            self._drive_to(
                "ADVANCE_HEAD_PIVOT_TO_EDGE",
                ordered[head_pivot_index],
                pivot_at_near_edge,
                approach_speed,
                roles[: head_pivot_index + 1],
                goal_tolerance,
            ),
            self._posture(
                "LOWER_HEAD_ACROSS_GAP",
                tuple(head_lift),
                neutral,
                ordered,
            ),
            self._drive_to(
                "ADVANCE_BODY_TO_FAR_SUPPORT",
                ordered[front_anchor_index],
                front_anchor_on_far_bank,
                crossing_speed,
                roles,
                goal_tolerance,
            ),
            self._posture(
                "LIFT_TAIL_DRAWBRIDGE",
                neutral,
                tuple(tail_lift),
                ordered,
            ),
            self._drive_to(
                "PULL_TAIL_TO_SAFE_LANDING",
                ordered[front_anchor_index],
                front_anchor_for_safe_tail_landing,
                crossing_speed,
                roles[front_anchor_index:],
                goal_tolerance,
            ),
            self._posture(
                "LOWER_TAIL_ON_FAR_BANK",
                tuple(tail_lift),
                neutral,
                ordered,
            ),
            self._drive_to(
                "CLEAR_FAR_EDGE",
                ordered[0],
                tail_on_far_bank + spacing,
                crossing_speed,
                roles,
                goal_tolerance,
            ),
        )

    @staticmethod
    def _drive_to(
        phase: str,
        reference: AssignedModule,
        target_x_m: float,
        speed_m_s: float,
        active_roles: tuple[str, ...],
        tolerance_m: float,
    ) -> BehaviorProgramStep:
        return BehaviorProgramStep(
            phase=phase,
            linear_m_s=speed_m_s,
            active_target_roles=active_roles,
            position_goal=LongitudinalPositionGoal(
                module_id=reference.module_id,
                target_x_m=target_x_m,
                tolerance_m=tolerance_m,
            ),
        )

    @staticmethod
    def _posture(
        phase: str,
        previous: tuple[float, ...],
        target: tuple[float, ...],
        assignments: Sequence[AssignedModule],
        *,
        all_targets: bool = False,
    ) -> BehaviorProgramStep:
        changed = tuple(
            range(len(target))
            if all_targets
            else (
                index
                for index, (old, new) in enumerate(zip(previous, target))
                if abs(old - new) > 1e-6
            )
        )
        return BehaviorProgramStep(
            phase=phase,
            posture_targets=tuple(
                JointTarget(
                    module_id=assignments[index].module_id,
                    joint="tilt",
                    angle_rad=target[index],
                    target_vertex_id=assignments[index].target_vertex_id,
                    target_role=assignments[index].target_role,
                    tolerance_rad=0.08,
                    coordination_group=f"gap:{phase}",
                    max_servo_error_rad=0.12,
                    angle_reference="captured_neutral",
                )
                for index in changed
            ),
        )

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
            raise SnakeGapGaitError(f"Missing live module pose: {error}") from error

    @staticmethod
    def _link_spacing(positions: Sequence[tuple[float, float, float]]) -> float:
        spacing = median(
            math.dist(first, second)
            for first, second in zip(positions, positions[1:])
        )
        if not math.isfinite(spacing) or not 0.060 <= spacing <= 0.100:
            raise SnakeGapGaitError(f"Invalid Snake8 link spacing {spacing:.4f} m")
        return spacing

    @staticmethod
    def _wheel_radius(graph: AttributedRobotGraph) -> float:
        geometry = graph.global_attributes.get("module_geometry")
        if not isinstance(geometry, Mapping):
            raise SnakeGapGaitError("Robot graph has no module geometry metadata")
        try:
            radius = float(geometry["wheel_radius_m"])
        except (KeyError, TypeError, ValueError) as error:
            raise SnakeGapGaitError("Invalid SMORES wheel radius metadata") from error
        if not math.isfinite(radius) or not 0.020 <= radius <= 0.050:
            raise SnakeGapGaitError(f"Invalid SMORES wheel radius {radius:.4f} m")
        return radius

    @staticmethod
    def _symmetric_tilt_limit(
        graph: AttributedRobotGraph,
        assignments: Sequence[AssignedModule],
    ) -> float:
        nodes = graph.node_by_id()
        limits: list[float] = []
        for assignment in assignments:
            actuators = nodes[assignment.module_id].attributes.get("actuators", {})
            tilt = actuators.get("tilt") if isinstance(actuators, Mapping) else None
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
            raise SnakeGapGaitError(f"Invalid SMORES TILT limit {limit:.4f} rad")
        return limit

    @staticmethod
    def _vertex_index(assignment: AssignedModule) -> int:
        try:
            return int(assignment.target_vertex_id.removeprefix("v"))
        except ValueError as error:
            raise SnakeGapGaitError("Snake8 vertices must be v0..v7") from error

    @staticmethod
    def _number(parameters: Mapping[str, Any], name: str, default: float) -> float:
        try:
            value = float(parameters.get(name, default))
        except (TypeError, ValueError) as error:
            raise SnakeGapGaitError(f"{name} must be numeric") from error
        if not math.isfinite(value):
            raise SnakeGapGaitError(f"{name} must be finite")
        return value

    def _speed(self, parameters: Mapping[str, Any], name: str, default: float) -> float:
        value = self._number(parameters, name, default)
        if not 0.0 < value <= 0.060:
            raise SnakeGapGaitError(f"{name} must be in (0, 0.060]")
        return value
