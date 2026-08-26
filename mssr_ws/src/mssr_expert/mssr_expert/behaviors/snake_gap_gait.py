"""Geometric traveling-backbone gait for a Snake8 crossing a flat gap."""

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
        finite_edges = all(math.isfinite(value) for value in (near, far))
        if not finite_edges or far <= near:
            raise SnakeGapGaitError(
                "Gap edges must be finite and strictly ordered"
            )
        declared_width = raw.get("width_m")
        if declared_width is not None:
            try:
                width = float(declared_width)
            except (TypeError, ValueError) as error:
                raise SnakeGapGaitError(
                    "Invalid declared gap width"
                ) from error
            if not math.isfinite(width) or not math.isclose(
                width, far - near, abs_tol=1e-6
            ):
                raise SnakeGapGaitError(
                    "Gap width is inconsistent with its edges"
                )
        return cls(near, far)


class SnakeGapGaitPlanner:
    """Plan a low backbone wave that travels from head to tail over a gap."""

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
    DEPRECATED_DRAWBRIDGE_PARAMETERS = frozenset(
        {
            "drawbridge_lift_angle_rad",
            "drawbridge_prelift_angle_rad",
            "drawbridge_bias_linear_m_s",
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
        deprecated = sorted(
            self.DEPRECATED_DRAWBRIDGE_PARAMETERS.intersection(parameters)
        )
        if deprecated:
            raise SnakeGapGaitError(
                "The vertical drawbridge gait was removed; omit deprecated "
                "parameters: " + ", ".join(deprecated)
            )
        ordered = tuple(sorted(assignments, key=self._vertex_index))
        if len(ordered) != self.MODULE_COUNT:
            raise SnakeGapGaitError("Geometric gap gait requires Snake8")
        vertex_indices = tuple(self._vertex_index(item) for item in ordered)
        if vertex_indices != tuple(range(8)):
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
            raise SnakeGapGaitError(
                "max_alignment_error_rad must be in (0, pi]"
            )
        if abs(heading) > max_heading:
            raise SnakeGapGaitError(
                "Snake8 is not aligned with the +X gap: "
                f"heading={heading:.3f} rad"
            )

        edge_clearance = self._number(parameters, "edge_clearance_m", 0.006)
        landing_margin = self._number(parameters, "landing_margin_m", 0.006)
        goal_tolerance = self._number(
            parameters,
            "gap_goal_tolerance_m",
            0.004,
        )
        if not 0.002 <= edge_clearance <= 0.020:
            raise SnakeGapGaitError(
                "edge_clearance_m must be in [0.002, 0.020]"
            )
        if not 0.002 <= landing_margin <= 0.020:
            raise SnakeGapGaitError(
                "landing_margin_m must be in [0.002, 0.020]"
            )
        if not 0.001 <= goal_tolerance <= 0.010:
            raise SnakeGapGaitError(
                "gap_goal_tolerance_m must be in [0.001, 0.010]"
            )
        far_transition_links = self._number(
            parameters,
            "far_bank_transition_links",
            1.0,
        )
        if not 0.5 <= far_transition_links <= 2.0:
            raise SnakeGapGaitError(
                "far_bank_transition_links must be in [0.5, 2.0]"
            )
        far_transition_m = far_transition_links * spacing

        required_span = (
            gap.width_m
            + 2.0 * wheel_radius
            + edge_clearance
            + landing_margin
            + far_transition_m
        )
        unsupported_link_count = max(2, math.ceil(required_span / spacing))
        maximum_unsupported_links = self.MODULE_COUNT - 3
        if unsupported_link_count > maximum_unsupported_links:
            maximum_gap = (
                maximum_unsupported_links * spacing
                - 2.0 * wheel_radius
                - edge_clearance
                - landing_margin
                - far_transition_m
            )
            raise SnakeGapGaitError(
                f"Gap width {gap.width_m:.3f} m exceeds the safe Snake8 "
                f"backbone-wave span {maximum_gap:.3f} m"
            )
        approach_speed = self._speed(parameters, "approach_linear_m_s", 0.050)
        crossing_speed = self._speed(parameters, "linear_m_s", 0.040)

        tilt_margin = self._number(parameters, "tilt_limit_margin_rad", 0.030)
        if not 0.0 <= tilt_margin <= 0.15:
            raise SnakeGapGaitError(
                "tilt_limit_margin_rad must be in [0, 0.15]"
            )
        usable_limit = self._symmetric_tilt_limit(graph, ordered) - tilt_margin
        roles = tuple(item.target_role for item in ordered)
        neutral = (0.0,) * self.MODULE_COUNT
        clearance_wheel_radii = self._number(
            parameters,
            "arch_clearance_wheel_radii",
            2.0,
        )
        if not 1.0 <= clearance_wheel_radii <= 3.0:
            raise SnakeGapGaitError(
                "arch_clearance_wheel_radii must be in [1.0, 3.0]"
            )
        landing_clearance = self._number(
            parameters,
            "landing_arch_clearance_m",
            clearance_wheel_radii * wheel_radius + edge_clearance,
        )
        if not 0.006 <= landing_clearance <= 0.110:
            raise SnakeGapGaitError(
                "landing_arch_clearance_m must be in [0.006, 0.110]"
            )
        profile_substeps_raw = self._number(
            parameters,
            "gap_profile_substeps",
            3.0,
        )
        profile_substeps = int(profile_substeps_raw)
        if (
            profile_substeps_raw != profile_substeps
            or not 1 <= profile_substeps <= 8
        ):
            raise SnakeGapGaitError(
                "gap_profile_substeps must be an integer in [1, 8]"
            )

        # The entire robot first reaches the near edge while flat.  A low,
        # positive backbone curve starts at the safe near support and remains
        # high through the far edge.  Its descending branch ends one measured
        # link beyond the safe far support by default, so a module cannot
        # descend into the vertical far-bank face before its wheel clears the
        # corner.  Translating the nominal module positions through that curve
        # makes the bend travel from head to tail without ever exposing a
        # symmetric 4-vs-4 lifting hinge.
        near_support_x = (
            gap.near_edge_x_m - wheel_radius - edge_clearance
        )
        far_support_x = (
            gap.far_edge_x_m + wheel_radius + landing_margin
        )
        far_arch_x = far_support_x + far_transition_m
        initial_nominal_x = tuple(
            near_support_x - (self.MODULE_COUNT - 1 - index) * spacing
            for index in range(self.MODULE_COUNT)
        )
        final_tail_x = far_arch_x
        body_travel = final_tail_x - initial_nominal_x[0]
        profile_step_count = max(
            1,
            math.ceil(body_travel / (spacing / profile_substeps)),
        )
        profile_steps: list[BehaviorProgramStep] = []
        previous_profile = neutral
        for profile_index in range(1, profile_step_count + 1):
            fraction = profile_index / profile_step_count
            translation = fraction * body_travel
            tail_x = initial_nominal_x[0] + translation
            module_x = self._module_x_positions_on_arch(
                tail_x,
                self.MODULE_COUNT,
                near_support_x,
                far_arch_x,
                landing_clearance,
                spacing,
            )
            profile = self._traveling_arch_offsets(
                module_x,
                near_support_x,
                far_arch_x,
                landing_clearance,
                spacing,
            )
            if max(abs(value) for value in profile) > usable_limit:
                raise SnakeGapGaitError(
                    "Traveling gap arch exceeds the live Snake8 TILT limits"
                )
            profile_steps.append(
                self._posture(
                    f"CONFORM_GAP_PROFILE_{profile_index:02d}",
                    previous_profile,
                    profile,
                    ordered,
                )
            )
            previous_profile = profile
            profile_steps.append(
                self._drive_to(
                    f"FOLLOW_GAP_PROFILE_{profile_index:02d}",
                    ordered[-1],
                    module_x[-1],
                    crossing_speed,
                    roles,
                    goal_tolerance,
                )
            )

        return (
            self._posture(
                "RESTORE_GAP_NEUTRAL",
                neutral,
                neutral,
                ordered,
                all_targets=True,
            ),
            self._drive_to(
                "APPROACH_HEAD_TO_NEAR_EDGE",
                ordered[-1],
                near_support_x,
                approach_speed,
                roles,
                goal_tolerance,
            ),
            *profile_steps,
            self._posture(
                "RESTORE_GAP_NEUTRAL_FINAL",
                previous_profile,
                neutral,
                ordered,
                all_targets=True,
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
    def _traveling_arch_offsets(
        module_x_m: Sequence[float],
        start_x_m: float,
        end_x_m: float,
        clearance_m: float,
        spacing_m: float,
    ) -> tuple[float, ...]:
        """Return TILT offsets for an upward arch fixed in the world gap."""

        heights = SnakeGapGaitPlanner._traveling_arch_heights(
            module_x_m,
            start_x_m,
            end_x_m,
            clearance_m,
        )
        link_angles: list[float] = []
        for first_height, second_height in zip(heights, heights[1:]):
            height_delta = second_height - first_height
            ratio = height_delta / spacing_m
            if abs(ratio) >= 1.0:
                raise SnakeGapGaitError(
                    "Traveling gap arch exceeds one-link vertical reach"
                )
            link_angles.append(math.asin(ratio))

        offsets = [0.0] * len(module_x_m)
        incoming_angle = 0.0
        for index, outgoing_angle in enumerate(link_angles):
            offsets[index] = outgoing_angle - incoming_angle
            incoming_angle = outgoing_angle
        return tuple(offsets)

    @staticmethod
    def _module_x_positions_on_arch(
        tail_x_m: float,
        module_count: int,
        start_x_m: float,
        end_x_m: float,
        clearance_m: float,
        spacing_m: float,
    ) -> tuple[float, ...]:
        """Place linked module centers on the arch at their true chord length.

        A curved link cannot span ``spacing_m`` in X while also changing
        height: its horizontal projection must be shorter.  Solving each
        successive chord prevents the head-position goal from asking the
        physical chain to stretch by the accumulated curve contraction.
        """

        if module_count < 1:
            raise SnakeGapGaitError("Gap arch requires at least one module")
        if not math.isfinite(spacing_m) or spacing_m <= 0.0:
            raise SnakeGapGaitError("Gap arch has invalid link spacing")

        positions = [tail_x_m]
        for _ in range(module_count - 1):
            first_x = positions[-1]
            first_height = SnakeGapGaitPlanner._traveling_arch_heights(
                (first_x,),
                start_x_m,
                end_x_m,
                clearance_m,
            )[0]
            lower = first_x
            upper = first_x + spacing_m
            for _ in range(64):
                candidate = 0.5 * (lower + upper)
                candidate_height = (
                    SnakeGapGaitPlanner._traveling_arch_heights(
                        (candidate,),
                        start_x_m,
                        end_x_m,
                        clearance_m,
                    )[0]
                )
                chord_error = (
                    (candidate - first_x) ** 2
                    + (candidate_height - first_height) ** 2
                    - spacing_m**2
                )
                if chord_error < 0.0:
                    lower = candidate
                else:
                    upper = candidate
            positions.append(0.5 * (lower + upper))
        return tuple(positions)

    @staticmethod
    def _traveling_arch_heights(
        module_x_m: Sequence[float],
        start_x_m: float,
        end_x_m: float,
        clearance_m: float,
    ) -> tuple[float, ...]:
        """Sample a non-negative world-frame arch at module centers."""

        width = end_x_m - start_x_m
        if width <= 0.0:
            raise SnakeGapGaitError("Traveling gap arch has invalid supports")
        return tuple(
            (
                clearance_m
                * math.sin(math.pi * (x_m - start_x_m) / width)
                if start_x_m < x_m < end_x_m
                else 0.0
            )
            for x_m in module_x_m
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
            raise SnakeGapGaitError(
                f"Missing live module pose: {error}"
            ) from error

    @staticmethod
    def _link_spacing(
        positions: Sequence[tuple[float, float, float]],
    ) -> float:
        spacing = median(
            math.dist(first, second)
            for first, second in zip(positions, positions[1:])
        )
        if not math.isfinite(spacing) or not 0.060 <= spacing <= 0.100:
            raise SnakeGapGaitError(
                f"Invalid Snake8 link spacing {spacing:.4f} m"
            )
        return spacing

    @staticmethod
    def _wheel_radius(graph: AttributedRobotGraph) -> float:
        geometry = graph.global_attributes.get("module_geometry")
        if not isinstance(geometry, Mapping):
            raise SnakeGapGaitError(
                "Robot graph has no module geometry metadata"
            )
        try:
            radius = float(geometry["wheel_radius_m"])
        except (KeyError, TypeError, ValueError) as error:
            raise SnakeGapGaitError(
                "Invalid SMORES wheel radius metadata"
            ) from error
        if not math.isfinite(radius) or not 0.020 <= radius <= 0.050:
            raise SnakeGapGaitError(
                f"Invalid SMORES wheel radius {radius:.4f} m"
            )
        return radius

    @staticmethod
    def _symmetric_tilt_limit(
        graph: AttributedRobotGraph,
        assignments: Sequence[AssignedModule],
    ) -> float:
        nodes = graph.node_by_id()
        limits: list[float] = []
        for assignment in assignments:
            actuators = nodes[assignment.module_id].attributes.get(
                "actuators",
                {},
            )
            tilt = (
                actuators.get("tilt")
                if isinstance(actuators, Mapping)
                else None
            )
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
            raise SnakeGapGaitError(
                f"Invalid SMORES TILT limit {limit:.4f} rad"
            )
        return limit

    @staticmethod
    def _vertex_index(assignment: AssignedModule) -> int:
        try:
            return int(assignment.target_vertex_id.removeprefix("v"))
        except ValueError as error:
            raise SnakeGapGaitError(
                "Snake8 vertices must be v0..v7"
            ) from error

    @staticmethod
    def _number(
        parameters: Mapping[str, Any],
        name: str,
        default: float,
    ) -> float:
        try:
            value = float(parameters.get(name, default))
        except (TypeError, ValueError) as error:
            raise SnakeGapGaitError(f"{name} must be numeric") from error
        if not math.isfinite(value):
            raise SnakeGapGaitError(f"{name} must be finite")
        return value

    def _speed(
        self,
        parameters: Mapping[str, Any],
        name: str,
        default: float,
    ) -> float:
        value = self._number(parameters, name, default)
        if not 0.0 < value <= 0.060:
            raise SnakeGapGaitError(f"{name} must be in (0, 0.060]")
        return value
