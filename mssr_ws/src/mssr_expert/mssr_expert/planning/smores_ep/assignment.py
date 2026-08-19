"""Optimal module-to-target assignment for SMORES-EP self-assembly."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from mssr_expert.planning.smores_ep.unfolding import (
    PlanarPose,
    UnfoldedPlanarConfiguration,
    normalize_angle,
)


class AssignmentError(ValueError):
    """Raised when a valid module assignment cannot be computed."""


@dataclass(frozen=True)
class AssignmentResult:
    """Optimal mapping from target vertices to physical modules."""

    target_to_module: Mapping[str, str]
    cost_by_target: Mapping[str, float]
    total_cost: float
    future_blockers_by_target: Mapping[str, int] = field(
        default_factory=dict
    )
    total_future_blockers: int = 0

    @property
    def module_to_target(self) -> dict[str, str]:
        """Return the inverse one-to-one assignment."""

        return {
            module_id: target_id
            for target_id, module_id in self.target_to_module.items()
        }


def assign_modules_to_targets(
    physical_poses: Mapping[str, PlanarPose],
    physical_root_id: str,
    target: UnfoldedPlanarConfiguration,
    orientation_weight_m_per_rad: float = 0.0,
    target_parent_by_vertex: Mapping[str, str | None] | None = None,
    target_depth_by_vertex: Mapping[str, int] | None = None,
    staging_distance_m: float = 0.070,
    staging_corridor_clearance_m: float = 0.110,
) -> AssignmentResult:
    """Assign modules with minimum congestion first, then minimum travel.

    When rooted target metadata is supplied, a physical module is a future
    blocker if it would remain parked beyond the wave whose docking corridor
    currently contains it.  A deterministic tier added to the Hungarian
    matrix makes fewer blockers lexicographically more important than travel
    distance while the reported ``total_cost`` remains a physical motion
    cost in metres.
    """

    _validate_assignment_input(
        physical_poses=physical_poses,
        physical_root_id=physical_root_id,
        target=target,
        orientation_weight_m_per_rad=orientation_weight_m_per_rad,
    )

    target_root_pose = target.poses_by_vertex[target.root_id]
    physical_root_pose = physical_poses[physical_root_id]

    target_relative_poses = {
        target_id: _pose_relative_to_root(
            pose=pose,
            root_pose=target_root_pose,
        )
        for target_id, pose in target.poses_by_vertex.items()
    }

    physical_relative_poses = {
        module_id: _pose_relative_to_root(
            pose=pose,
            root_pose=physical_root_pose,
        )
        for module_id, pose in physical_poses.items()
    }

    target_ids = sorted(
        target_id
        for target_id in target.poses_by_vertex
        if target_id != target.root_id
    )

    module_ids = sorted(
        module_id
        for module_id in physical_poses
        if module_id != physical_root_id
    )

    motion_cost_matrix: tuple[tuple[float, ...], ...] = tuple(
        tuple(
            _assignment_cost(
                physical_pose=physical_relative_poses[module_id],
                target_pose=target_relative_poses[target_id],
                orientation_weight_m_per_rad=orientation_weight_m_per_rad,
            )
            for module_id in module_ids
        )
        for target_id in target_ids
    )

    future_blockers_by_pair = future_blocker_counts_by_pair(
        physical_poses=physical_poses,
        physical_root_id=physical_root_id,
        target=target,
        target_parent_by_vertex=target_parent_by_vertex,
        target_depth_by_vertex=target_depth_by_vertex,
        staging_distance_m=staging_distance_m,
        staging_corridor_clearance_m=staging_corridor_clearance_m,
    )
    selection_cost_by_pair = congestion_aware_pair_costs(
        motion_cost_by_pair={
            (target_id, module_id): motion_cost_matrix[row_index][
                column_index
            ]
            for row_index, target_id in enumerate(target_ids)
            for column_index, module_id in enumerate(module_ids)
        },
        future_blockers_by_pair=future_blockers_by_pair,
        target_ids=target_ids,
        module_ids=module_ids,
    )
    selection_cost_matrix = tuple(
        tuple(
            selection_cost_by_pair[(target_id, module_id)]
            for module_id in module_ids
        )
        for target_id in target_ids
    )
    selected_columns = solve_rectangular_assignment(selection_cost_matrix)

    target_to_module: dict[str, str] = {
        target.root_id: physical_root_id,
    }

    cost_by_target: dict[str, float] = {
        target.root_id: 0.0,
    }
    blocker_count_by_target: dict[str, int] = {
        target.root_id: 0,
    }

    for row_index, column_index in enumerate(selected_columns):
        target_id = target_ids[row_index]
        module_id = module_ids[column_index]
        cost = motion_cost_matrix[row_index][column_index]

        target_to_module[target_id] = module_id
        cost_by_target[target_id] = cost
        blocker_count_by_target[target_id] = future_blockers_by_pair.get(
            (target_id, module_id),
            0,
        )

    total_cost = sum(cost_by_target.values())

    return AssignmentResult(
        target_to_module=dict(target_to_module),
        cost_by_target=dict(cost_by_target),
        total_cost=total_cost,
        future_blockers_by_target=dict(blocker_count_by_target),
        total_future_blockers=sum(blocker_count_by_target.values()),
    )


def future_blocker_counts_by_pair(
    physical_poses: Mapping[str, PlanarPose],
    physical_root_id: str,
    target: UnfoldedPlanarConfiguration,
    target_parent_by_vertex: Mapping[str, str | None] | None,
    target_depth_by_vertex: Mapping[str, int] | None,
    staging_distance_m: float = 0.070,
    staging_corridor_clearance_m: float = 0.110,
) -> dict[tuple[str, str], int]:
    """Count earlier docking corridors occupied by each future assignment.

    The calculation is separable by target/module pair, so the final
    optimization remains a standard Hungarian linear assignment.
    """

    if target_parent_by_vertex is None or target_depth_by_vertex is None:
        return {}
    if (
        not math.isfinite(staging_distance_m)
        or staging_distance_m <= 0.0
    ):
        raise AssignmentError(
            "The assignment staging distance must be positive and finite."
        )
    if (
        not math.isfinite(staging_corridor_clearance_m)
        or staging_corridor_clearance_m <= 0.0
    ):
        raise AssignmentError(
            "The assignment corridor clearance must be positive and finite."
        )
    target_ids = set(target.poses_by_vertex)
    if set(target_parent_by_vertex) != target_ids:
        raise AssignmentError(
            "Target parent metadata must cover every unfolded vertex."
        )
    if set(target_depth_by_vertex) != target_ids:
        raise AssignmentError(
            "Target depth metadata must cover every unfolded vertex."
        )

    target_root_pose = target.poses_by_vertex[target.root_id]
    physical_root_pose = physical_poses[physical_root_id]
    target_relative = {
        target_id: _pose_relative_to_root(pose, target_root_pose)
        for target_id, pose in target.poses_by_vertex.items()
    }
    physical_relative = {
        module_id: _pose_relative_to_root(pose, physical_root_pose)
        for module_id, pose in physical_poses.items()
    }
    corridors: list[
        tuple[int, tuple[float, float], tuple[float, float]]
    ] = []
    for child_id in sorted(target_ids):
        parent_id = target_parent_by_vertex[child_id]
        if parent_id is None:
            continue
        parent_pose = target_relative[parent_id]
        child_pose = target_relative[child_id]
        delta_x = child_pose.x_m - parent_pose.x_m
        delta_y = child_pose.y_m - parent_pose.y_m
        distance = math.hypot(delta_x, delta_y)
        if distance <= 1.0e-9:
            raise AssignmentError(
                f"Target edge {parent_id!r}->{child_id!r} has no planar "
                "direction."
            )
        final_xy = (child_pose.x_m, child_pose.y_m)
        staging_xy = (
            child_pose.x_m + staging_distance_m * delta_x / distance,
            child_pose.y_m + staging_distance_m * delta_y / distance,
        )
        corridors.append(
            (target_depth_by_vertex[child_id], final_xy, staging_xy)
        )

    result: dict[tuple[str, str], int] = {}
    for target_id in sorted(target_ids):
        target_depth = target_depth_by_vertex[target_id]
        for module_id, module_pose in physical_relative.items():
            if module_id == physical_root_id or target_id == target.root_id:
                result[(target_id, module_id)] = 0
                continue
            point = (module_pose.x_m, module_pose.y_m)
            result[(target_id, module_id)] = sum(
                _point_segment_distance(point, final_xy, staging_xy)
                < staging_corridor_clearance_m
                for corridor_depth, final_xy, staging_xy in corridors
                if corridor_depth < target_depth
            )
    return result


def congestion_aware_pair_costs(
    motion_cost_by_pair: Mapping[tuple[str, str], float],
    future_blockers_by_pair: Mapping[tuple[str, str], int],
    target_ids: Sequence[str],
    module_ids: Sequence[str],
) -> dict[tuple[str, str], float]:
    """Encode blocker count before motion cost in one Hungarian matrix."""

    maximum_total_motion_cost = sum(
        max(
            motion_cost_by_pair[(target_id, module_id)]
            for module_id in module_ids
        )
        for target_id in target_ids
    )
    blocker_tier = maximum_total_motion_cost + 1.0
    return {
        (target_id, module_id): (
            motion_cost_by_pair[(target_id, module_id)]
            + blocker_tier
            * future_blockers_by_pair.get((target_id, module_id), 0)
        )
        for target_id in target_ids
        for module_id in module_ids
    }


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    delta_x = end[0] - start[0]
    delta_y = end[1] - start[1]
    length_squared = delta_x * delta_x + delta_y * delta_y
    if length_squared <= 1.0e-18:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    projection = (
        (point[0] - start[0]) * delta_x
        + (point[1] - start[1]) * delta_y
    ) / length_squared
    projection = min(1.0, max(0.0, projection))
    closest_x = start[0] + projection * delta_x
    closest_y = start[1] + projection * delta_y
    return math.hypot(point[0] - closest_x, point[1] - closest_y)


def solve_linear_assignment(
    cost_matrix: Sequence[Sequence[float]],
) -> tuple[int, ...]:
    """Solve a square minimum-cost assignment with the Hungarian algorithm.

    The returned tuple contains the selected column for every matrix row.
    """

    row_count = len(cost_matrix)

    if row_count == 0:
        return ()

    if any(len(row) != row_count for row in cost_matrix):
        raise AssignmentError(
            "The Hungarian solver requires a square cost matrix."
        )

    for row in cost_matrix:
        for cost in row:
            if (
                not isinstance(cost, int | float)
                or isinstance(cost, bool)
                or not math.isfinite(float(cost))
            ):
                raise AssignmentError(
                    "Every assignment cost must be finite and numeric."
                )

    # Potentials for rows and columns.
    row_potential = [0.0] * (row_count + 1)
    column_potential = [0.0] * (row_count + 1)

    # matched_row_by_column[j] contains the row assigned to column j.
    matched_row_by_column = [0] * (row_count + 1)
    previous_column = [0] * (row_count + 1)

    epsilon = 1e-12

    for row_index in range(1, row_count + 1):
        matched_row_by_column[0] = row_index

        minimum_reduced_cost = [math.inf] * (row_count + 1)
        used_columns = [False] * (row_count + 1)

        current_column = 0

        while True:
            used_columns[current_column] = True
            current_row = matched_row_by_column[current_column]

            delta = math.inf
            next_column = 0

            for column_index in range(1, row_count + 1):
                if used_columns[column_index]:
                    continue

                reduced_cost = (
                    float(cost_matrix[current_row - 1][column_index - 1])
                    - row_potential[current_row]
                    - column_potential[column_index]
                )

                if (
                    reduced_cost
                    < minimum_reduced_cost[column_index] - epsilon
                ):
                    minimum_reduced_cost[column_index] = reduced_cost
                    previous_column[column_index] = current_column

                candidate_cost = minimum_reduced_cost[column_index]

                if (
                    candidate_cost < delta - epsilon
                    or (
                        abs(candidate_cost - delta) <= epsilon
                        and (
                            next_column == 0
                            or column_index < next_column
                        )
                    )
                ):
                    delta = candidate_cost
                    next_column = column_index

            if not math.isfinite(delta):
                raise AssignmentError(
                    "The assignment matrix has no complete solution."
                )

            for column_index in range(row_count + 1):
                if used_columns[column_index]:
                    matched_row = matched_row_by_column[column_index]
                    row_potential[matched_row] += delta
                    column_potential[column_index] -= delta
                else:
                    minimum_reduced_cost[column_index] -= delta

            current_column = next_column

            if matched_row_by_column[current_column] == 0:
                break

        while True:
            next_column = previous_column[current_column]

            matched_row_by_column[current_column] = (
                matched_row_by_column[next_column]
            )

            current_column = next_column

            if current_column == 0:
                break

    selected_column_by_row = [-1] * row_count

    for column_index in range(1, row_count + 1):
        matched_row = matched_row_by_column[column_index]

        if matched_row != 0:
            selected_column_by_row[matched_row - 1] = column_index - 1

    if any(column_index < 0 for column_index in selected_column_by_row):
        raise AssignmentError(
            "The Hungarian solver produced an incomplete assignment."
        )

    return tuple(selected_column_by_row)


def solve_rectangular_assignment(
    cost_matrix: Sequence[Sequence[float]],
) -> tuple[int, ...]:
    """Assign every row to a distinct column when columns may be reserves.

    The existing deterministic Hungarian implementation remains the single
    solver. Zero-cost dummy rows pad a rows-by-columns matrix to square; only
    assignments belonging to real target rows are returned. This lets a
    target morphology use a strict subset of the available physical modules.
    """

    row_count = len(cost_matrix)
    if row_count == 0:
        return ()
    column_count = len(cost_matrix[0])
    if column_count < row_count or any(
        len(row) != column_count for row in cost_matrix
    ):
        raise AssignmentError(
            "Assignment needs at least as many columns as rows"
        )
    padded = [tuple(float(value) for value in row) for row in cost_matrix]
    padded.extend(
        tuple(0.0 for _ in range(column_count))
        for _ in range(column_count - row_count)
    )
    return solve_linear_assignment(tuple(padded))[:row_count]


def _pose_relative_to_root(
    pose: PlanarPose,
    root_pose: PlanarPose,
) -> PlanarPose:
    """Express a planar pose in the coordinate frame of the root."""

    delta_x = pose.x_m - root_pose.x_m
    delta_y = pose.y_m - root_pose.y_m

    cosine = math.cos(root_pose.yaw_rad)
    sine = math.sin(root_pose.yaw_rad)

    relative_x = cosine * delta_x + sine * delta_y
    relative_y = -sine * delta_x + cosine * delta_y

    relative_yaw = normalize_angle(
        pose.yaw_rad - root_pose.yaw_rad
    )

    return PlanarPose(
        x_m=relative_x,
        y_m=relative_y,
        yaw_rad=relative_yaw,
    )


def _assignment_cost(
    physical_pose: PlanarPose,
    target_pose: PlanarPose,
    orientation_weight_m_per_rad: float,
) -> float:
    """Calculate the travel cost from a physical pose to a target pose."""

    position_cost = math.hypot(
        physical_pose.x_m - target_pose.x_m,
        physical_pose.y_m - target_pose.y_m,
    )

    orientation_error = abs(
        normalize_angle(
            physical_pose.yaw_rad - target_pose.yaw_rad
        )
    )

    return (
        position_cost
        + orientation_weight_m_per_rad * orientation_error
    )


def _validate_assignment_input(
    physical_poses: Mapping[str, PlanarPose],
    physical_root_id: str,
    target: UnfoldedPlanarConfiguration,
    orientation_weight_m_per_rad: float,
) -> None:
    """Validate physical and target modules before assignment."""

    if not physical_poses:
        raise AssignmentError(
            "At least one physical module is required."
        )

    if physical_root_id not in physical_poses:
        raise AssignmentError(
            f"Unknown physical root module {physical_root_id!r}."
        )

    if target.root_id not in target.poses_by_vertex:
        raise AssignmentError(
            f"Unknown target root vertex {target.root_id!r}."
        )

    if len(physical_poses) < len(target.poses_by_vertex):
        raise AssignmentError(
            "The physical module count must be at least the target count."
        )

    for module_id, pose in physical_poses.items():
        if not isinstance(module_id, str) or not module_id.strip():
            raise AssignmentError(
                "Every physical module must have a non-empty ID."
            )

        if not isinstance(pose, PlanarPose):
            raise AssignmentError(
                f"Module {module_id!r} does not contain a PlanarPose."
            )

    if (
        not isinstance(orientation_weight_m_per_rad, int | float)
        or isinstance(orientation_weight_m_per_rad, bool)
        or not math.isfinite(float(orientation_weight_m_per_rad))
        or orientation_weight_m_per_rad < 0.0
    ):
        raise AssignmentError(
            "The orientation weight must be finite and non-negative."
        )
