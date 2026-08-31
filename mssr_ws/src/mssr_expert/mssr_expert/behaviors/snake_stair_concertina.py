"""Collider-aware wheel-centre path following for Snake8 stairs."""

from __future__ import annotations

import math
from statistics import median
from typing import Any, Mapping, Sequence

from mssr_expert.behaviors.morphology_library import (
    AssignedModule,
    BehaviorProgramStep,
    JointTarget,
    LongitudinalPositionGoal,
)
from mssr_expert.behaviors.snake_stair_concertina_geometry import (
    ConcertinaStaircase,
)
from mssr_expert.behaviors.snake_stair_gait import SnakeStairGaitError
from mssr_expert.behaviors.snake_stair_path_ik import (
    PathPoint,
    WheelCenterPath,
    reconstruct_centers,
    relative_tilt_ik,
)
from mssr_expert.graph.attributed_robot_graph import AttributedRobotGraph
from mssr_expert.primitives.common import module_position


class SnakeStairConcertinaPlanner:
    """Generate a global stair-profile trajectory and solve it by IK.

    The planner deliberately has no BUILD/GROW/SHIFT state machine. The stair
    collision profile first becomes a smooth wheel-centre path. Eight points
    at the measured rigid module spacing are then placed on that path and
    converted to relative TILT angles. Advancing the head coordinate moves
    the same continuous shape through all eight modules like a tracked belt.
    """

    MODULE_COUNT = 8

    def plan(
        self,
        graph: AttributedRobotGraph,
        assignments: Sequence[AssignedModule],
        parameters: Mapping[str, Any],
        neutral_tilt_rad_by_module: Mapping[str, float] | None = None,
    ) -> tuple[BehaviorProgramStep, ...]:
        ordered = tuple(sorted(assignments, key=self._vertex_index))
        if len(ordered) != self.MODULE_COUNT:
            raise SnakeStairGaitError("Spatial stair path IK requires Snake8")
        course = graph.global_attributes.get("course")
        if not isinstance(course, Mapping):
            raise SnakeStairGaitError("Robot graph has no course metadata")
        staircase = ConcertinaStaircase.from_course(course)
        positions = self._ordered_positions(graph, ordered)
        spacing = self._link_spacing(positions)
        wheel_radius, forward_extent, pan_face_radius = (
            self._module_geometry(graph)
        )

        heading = math.atan2(
            positions[-1][1] - positions[0][1],
            positions[-1][0] - positions[0][0],
        )
        max_heading = self._number(
            parameters, "max_alignment_error_rad", 0.35
        )
        if abs(heading) > max_heading:
            raise SnakeStairGaitError(
                "Snake8 is not aligned with +X stairs: "
                f"heading={heading:.3f} rad"
            )
        if any(
            positions[index + 1][0] <= positions[index][0]
            for index in range(self.MODULE_COUNT - 1)
        ):
            raise SnakeStairGaitError(
                "Snake8 tail-to-head ordering is not monotone in +X"
            )

        edge_safety = self._number(
            parameters, "path_corner_safety_m", 0.005
        )
        if not 0.003 <= edge_safety <= 0.020:
            raise SnakeStairGaitError(
                "path_corner_safety_m must be in [0.003, 0.020]"
            )
        corner_radius = max(
            wheel_radius, forward_extent, pan_face_radius
        ) + edge_safety
        approach_run = self._number(
            parameters, "path_approach_run_m", max(0.135, 1.70 * spacing)
        )
        landing_run = self._number(
            parameters, "path_landing_run_m", max(0.105, 1.30 * spacing)
        )
        if not 0.080 <= approach_run <= 0.180:
            raise SnakeStairGaitError(
                "path_approach_run_m must be in [0.080, 0.180]"
            )
        if not 0.060 <= landing_run <= 0.140:
            raise SnakeStairGaitError(
                "path_landing_run_m must be in [0.060, 0.140]"
            )
        path = WheelCenterPath(
            staircase=staircase,
            wheel_radius_m=wheel_radius,
            corner_clearance_radius_m=corner_radius,
            approach_run_m=approach_run,
            landing_run_m=landing_run,
        )

        trajectory_step = self._number(
            parameters, "trajectory_step_m", 0.012
        )
        if not 0.005 <= trajectory_step <= 0.030:
            raise SnakeStairGaitError(
                "trajectory_step_m must be in [0.005, 0.030]"
            )
        goal_tolerance = self._number(
            parameters, "crawl_goal_tolerance_m", 0.003
        )
        if not 0.001 <= goal_tolerance <= 0.006:
            raise SnakeStairGaitError(
                "crawl_goal_tolerance_m must be in [0.001, 0.006]"
            )
        maximum_speed = self._speed(parameters, "linear_m_s", 0.040)
        tracking_kp = self._number(
            parameters, "trajectory_tracking_kp_s_inv", 2.0
        )
        tracking_kd = self._number(
            parameters, "trajectory_tracking_kd", 0.25
        )
        minimum_speed = self._number(
            parameters, "trajectory_tracking_min_linear_m_s", 0.012
        )
        if not 0.2 <= tracking_kp <= 8.0:
            raise SnakeStairGaitError(
                "trajectory_tracking_kp_s_inv must be in [0.2, 8.0]"
            )
        if not 0.0 <= tracking_kd <= 2.0:
            raise SnakeStairGaitError(
                "trajectory_tracking_kd must be in [0.0, 2.0]"
            )
        if not 0.005 <= minimum_speed <= maximum_speed:
            raise SnakeStairGaitError(
                "trajectory_tracking_min_linear_m_s must be at least 0.005 "
                "and no larger than linear_m_s"
            )

        tilt_margin = self._number(
            parameters, "tilt_limit_margin_rad", 0.030
        )
        if not 0.0 <= tilt_margin <= 0.15:
            raise SnakeStairGaitError(
                "tilt_limit_margin_rad must be in [0.0, 0.15]"
            )
        safe_bounds = self._relative_tilt_bounds(
            graph,
            ordered,
            neutral_tilt_rad_by_module,
            tilt_margin,
        )

        last_edge = path.riser_edges_m[-1]
        tail_landing_inset = self._number(
            parameters,
            "path_tail_landing_inset_m",
            max(landing_run, corner_radius + 0.010),
        )
        if not max(corner_radius, landing_run) <= tail_landing_inset <= 0.120:
            raise SnakeStairGaitError(
                "path_tail_landing_inset_m must cover the corner envelope "
                "and complete the landing curve within 0.120 m"
            )
        final_head_x = path.head_x_for_tail_x(
            tail_x_m=last_edge + tail_landing_inset,
            module_count=self.MODULE_COUNT,
            link_length_m=spacing,
        )
        initial_head_x = positions[-1][0]
        if final_head_x <= initial_head_x:
            raise SnakeStairGaitError(
                "Snake8 is already beyond the planned stair trajectory"
            )

        interval_count = max(
            1, math.ceil((final_head_x - initial_head_x) / trajectory_step)
        )
        head_waypoints = tuple(
            initial_head_x
            + (final_head_x - initial_head_x) * index / interval_count
            for index in range(interval_count + 1)
        )
        solutions = tuple(
            self._solve_waypoint(
                path=path,
                head_x_m=head_x,
                spacing_m=spacing,
                safe_bounds=safe_bounds,
            )
            for head_x in head_waypoints
        )

        all_roles = tuple(item.target_role for item in ordered)
        program: list[BehaviorProgramStep] = []
        _, initial_tilts = solutions[0]
        program.append(
            BehaviorProgramStep(
                phase="PATH_IK_PRELOAD",
                posture_targets=self._posture_targets(
                    phase="PATH_IK_PRELOAD",
                    tilts=initial_tilts,
                    assignments=ordered,
                ),
            )
        )

        head_module_id = ordered[-1].module_id
        for index, (head_x, solution) in enumerate(
            zip(head_waypoints[1:], solutions[1:]), start=1
        ):
            _, tilts = solution
            phase = f"PATH_IK_TRACK_{index:03d}_OF_{interval_count:03d}"
            program.append(
                BehaviorProgramStep(
                    phase=phase,
                    posture_targets=self._posture_targets(
                        phase=phase,
                        tilts=tilts,
                        assignments=ordered,
                    ),
                    linear_m_s=maximum_speed,
                    active_target_roles=all_roles,
                    position_goal=LongitudinalPositionGoal(
                        module_id=head_module_id,
                        target_x_m=head_x,
                        tolerance_m=goal_tolerance,
                    ),
                    continuous_with_next=index < interval_count,
                    hold_locomotion_until_admitted=True,
                    position_tracking_kp_s_inv=tracking_kp,
                    position_tracking_kd=tracking_kd,
                    minimum_tracking_linear_m_s=minimum_speed,
                )
            )

        final_points, final_tilts = solutions[-1]
        if final_points[0].x_m + 1.0e-6 < last_edge + tail_landing_inset:
            raise SnakeStairGaitError("Path IK did not land the tail safely")
        program.append(
            BehaviorProgramStep(
                phase="PATH_IK_UPPER_DECK_SETTLE",
                posture_targets=self._posture_targets(
                    phase="PATH_IK_UPPER_DECK_SETTLE",
                    tilts=final_tilts,
                    assignments=ordered,
                ),
            )
        )
        return tuple(program)

    def _solve_waypoint(
        self,
        *,
        path: WheelCenterPath,
        head_x_m: float,
        spacing_m: float,
        safe_bounds: Sequence[tuple[float, float]],
    ) -> tuple[tuple[PathPoint, ...], tuple[float, ...]]:
        points = path.sample_module_centers(
            head_x_m=head_x_m,
            module_count=self.MODULE_COUNT,
            link_length_m=spacing_m,
        )
        tilts = relative_tilt_ik(points)
        for index, (tilt, bounds) in enumerate(zip(tilts, safe_bounds)):
            if not bounds[0] - 1.0e-9 <= tilt <= bounds[1] + 1.0e-9:
                raise SnakeStairGaitError(
                    "Wheel-centre path IK exceeds the safe TILT range at "
                    f"module {index}: {tilt:.3f} rad not in {bounds}"
                )
        reconstructed = reconstruct_centers(points[0], tilts, spacing_m)
        if any(
            math.hypot(actual.x_m - expected.x_m, actual.z_m - expected.z_m)
            > 1.0e-8
            for actual, expected in zip(reconstructed, points)
        ):
            raise RuntimeError("Snake8 path IK failed its FK consistency check")
        return points, tilts

    @staticmethod
    def _posture_targets(
        *,
        phase: str,
        tilts: Sequence[float],
        assignments: Sequence[AssignedModule],
    ) -> tuple[JointTarget, ...]:
        return tuple(
            JointTarget(
                module_id=assignment.module_id,
                joint="tilt",
                angle_rad=float(tilts[index]),
                target_vertex_id=assignment.target_vertex_id,
                target_role=assignment.target_role,
                tolerance_rad=0.035,
                coordination_group=f"stair-path:{phase}",
                max_servo_error_rad=0.05,
                max_servo_speed_rad_s=0.45,
                passive_module_ids=(),
                angle_reference="captured_neutral",
            )
            for index, assignment in enumerate(assignments)
        )

    @staticmethod
    def _ordered_positions(
        graph: AttributedRobotGraph,
        assignments: Sequence[AssignedModule],
    ) -> tuple[tuple[float, float, float], ...]:
        nodes = graph.node_by_id()
        positions: list[tuple[float, float, float]] = []
        for assignment in assignments:
            node = nodes.get(assignment.module_id)
            if node is None:
                raise SnakeStairGaitError(
                    f"Missing graph node for {assignment.module_id}"
                )
            positions.append(module_position(node.attributes))
        return tuple(positions)

    @staticmethod
    def _link_spacing(
        positions: Sequence[tuple[float, float, float]],
    ) -> float:
        distances = [
            math.dist(lower, upper)
            for lower, upper in zip(positions, positions[1:])
        ]
        spacing = median(distances)
        if not math.isfinite(spacing) or spacing <= 0.0:
            raise SnakeStairGaitError("Invalid Snake8 module spacing")
        if any(abs(distance - spacing) > 0.015 for distance in distances):
            raise SnakeStairGaitError(
                "Snake8 is too distorted to initialize stair path IK"
            )
        return spacing

    @staticmethod
    def _module_geometry(
        graph: AttributedRobotGraph,
    ) -> tuple[float, float, float]:
        geometry = graph.global_attributes.get("module_geometry")
        if not isinstance(geometry, Mapping):
            raise SnakeStairGaitError(
                "Robot graph has no live module collision geometry"
            )
        try:
            wheel = float(geometry["wheel_radius_m"])
            forward = float(geometry["forward_collision_extent_m"])
            pan = float(geometry["pan_face_radius_m"])
        except (KeyError, TypeError, ValueError) as error:
            raise SnakeStairGaitError(
                "Robot graph has incomplete module collision geometry"
            ) from error
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (wheel, forward, pan)
        ):
            raise SnakeStairGaitError("Invalid module collision geometry")
        return wheel, forward, pan

    @staticmethod
    def _relative_tilt_bounds(
        graph: AttributedRobotGraph,
        assignments: Sequence[AssignedModule],
        neutrals: Mapping[str, float] | None,
        margin_rad: float,
    ) -> tuple[tuple[float, float], ...]:
        nodes = graph.node_by_id()
        neutral_map = neutrals or {}
        bounds: list[tuple[float, float]] = []
        for assignment in assignments:
            lower = -0.5 * math.pi
            upper = 0.5 * math.pi
            neutral = float(neutral_map.get(assignment.module_id, 0.0))
            node = nodes.get(assignment.module_id)
            if node is not None:
                actuators = node.attributes.get("actuators")
                if isinstance(actuators, Mapping):
                    tilt = actuators.get("tilt")
                    if isinstance(tilt, Mapping):
                        try:
                            lower = float(tilt["lower_limit_rad"])
                            upper = float(tilt["upper_limit_rad"])
                            neutral = float(
                                neutral_map.get(
                                    assignment.module_id,
                                    tilt.get("position_rad", neutral),
                                )
                            )
                        except (KeyError, TypeError, ValueError) as error:
                            raise SnakeStairGaitError(
                                "Invalid live TILT limits"
                            ) from error
            relative_lower = lower - neutral + margin_rad
            relative_upper = upper - neutral - margin_rad
            if (
                not all(
                    math.isfinite(value)
                    for value in (relative_lower, relative_upper)
                )
                or relative_lower >= relative_upper
            ):
                raise SnakeStairGaitError(
                    f"No safe TILT range remains for {assignment.module_id}"
                )
            bounds.append((relative_lower, relative_upper))
        return tuple(bounds)

    @staticmethod
    def _vertex_index(assignment: AssignedModule) -> int:
        digits = "".join(
            character
            for character in assignment.target_vertex_id
            if character.isdigit()
        )
        if not digits:
            raise SnakeStairGaitError(
                f"Cannot order target vertex {assignment.target_vertex_id!r}"
            )
        return int(digits)

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

    @classmethod
    def _speed(
        cls, parameters: Mapping[str, Any], name: str, default: float
    ) -> float:
        value = cls._number(parameters, name, default)
        if not 0.005 <= value <= 0.080:
            raise SnakeStairGaitError(f"{name} must be in [0.005, 0.080]")
        return value
