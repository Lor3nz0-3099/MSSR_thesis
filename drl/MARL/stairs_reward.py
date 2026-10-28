"""Raw C1 Snake8 stairs reward terms.

No reward weights live here.

This module translates physical state transitions and shared runtime
outcomes into raw reward terms.  Weight selection belongs to the APPO
training configuration after real-rollout scale calibration.

Design rules:
- wheel/actuator motion is never progress by itself;
- progress is measured only from physical XYZ state;
- ascent and descent share one contract;
- stagnation is consumed from the existing common runtime signal;
- level completion requires both geometric X crossing and the appropriate
  physical Z level, preventing a vertical lift in place from masquerading
  as stair completion.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from drl.MARL.phase1_reward import (
    ActionOutcomeSignals,
)
from drl.MARL.reward_potential import (
    potential_difference,
    validate_owner_xyz_positions as _validate_positions,
    validate_shaping_discount,
)
from smores_ep.config.geometry import (
    SmoresGeometry,
)
from smores_ep.isaac.phase1_stairs_curriculum import (
    Phase1StairCourse,
)


XYZ = tuple[float, float, float]


@dataclass(frozen=True)
class StairsOwnerRewardTerms:
    profile_progress: float
    vertical_progress: float
    level_progress: float

    goal_entry: float
    success: float

    outside_width: float
    below_support: float

    collision: float
    conflict: float
    invalid: float
    redundant_action: float
    control_effort: float

    time: float
    stagnation: float


@dataclass(frozen=True)
class StairsTeamRewardTerms:
    profile_progress: float
    vertical_progress: float
    level_progress: float

    goal_entry: float
    success: float

    outside_width: float
    below_support: float

    collision: float
    conflict: float
    invalid: float
    redundant_action: float
    control_effort: float

    time: float
    stagnation: float


@dataclass(frozen=True)
class StairsRawRewardFrame:
    owners: Mapping[
        str,
        StairsOwnerRewardTerms,
    ]

    team: StairsTeamRewardTerms


def _validate_signals(
    signals: ActionOutcomeSignals,
    owners: tuple[str, ...],
) -> None:
    known = set(owners)

    for values in (
        signals.collision_owner_ids,
        signals.conflict_owner_ids,
        signals.invalid_owner_ids,
    ):
        unknown = set(values) - known

        if unknown:
            raise ValueError(
                "Runtime signal references unknown owner: "
                f"{sorted(unknown)}"
            )

    for mapping, label in (
        (
            signals.redundant_action_count_by_owner,
            "redundant action",
        ),
        (
            signals.control_effort_by_owner,
            "control effort",
        ),
    ):
        unknown = set(mapping) - known

        if unknown:
            raise ValueError(
                f"{label} references unknown owner: "
                f"{sorted(unknown)}"
            )

    for owner, value in (
        signals
        .redundant_action_count_by_owner
        .items()
    ):
        if (
            type(value) is not int
            or value < 0
        ):
            raise ValueError(
                "Redundant action count must be a "
                f"nonnegative integer for {owner}"
            )

    for owner, value in (
        signals
        .control_effort_by_owner
        .items()
    ):
        numeric = float(value)

        if (
            not math.isfinite(numeric)
            or numeric < 0.0
        ):
            raise ValueError(
                "Control effort must be finite and "
                f"nonnegative for {owner}"
            )


def _task_run_length_m(
    course: Phase1StairCourse,
) -> float:
    start_x = (
        course.start_platform_end_x_m
        - course.start_platform_length_m
    )

    goal_x = (
        course.goal_platform_start_x_m
        + course.goal_platform_length_m
    )

    length = goal_x - start_x

    if length <= 1.0e-9:
        raise ValueError(
            "Stairs task has zero longitudinal length"
        )

    return float(length)


def _elevation_change_m(
    course: Phase1StairCourse,
) -> float:
    change = abs(
        float(course.goal_height_m)
        - float(course.start_height_m)
    )

    if change <= 1.0e-9:
        raise ValueError(
            "Stairs task has zero elevation change"
        )

    return change


def _step_start_xs(
    course: Phase1StairCourse,
) -> tuple[float, ...]:
    starts = []
    cursor = float(
        course.first_transition_x_m
    )

    for depth in course.tread_depths_m:
        starts.append(cursor)
        cursor += float(depth)

    return tuple(starts)


def _completed_levels(
    course: Phase1StairCourse,
    point_xyz_m: XYZ,
    *,
    wheel_radius_m: float,
    vertical_tolerance_m: float,
) -> int:
    x_m = float(point_xyz_m[0])
    z_m = float(point_xyz_m[2])

    completed = 0

    for start_x_m, top_height_m in zip(
        _step_start_xs(course),
        course.step_top_heights_m,
    ):
        if x_m < start_x_m:
            continue

        expected_center_z_m = (
            float(top_height_m)
            + wheel_radius_m
        )

        if course.direction == "ascent":
            level_reached = (
                z_m
                >= expected_center_z_m
                - vertical_tolerance_m
            )
        else:
            level_reached = (
                z_m
                <= expected_center_z_m
                + vertical_tolerance_m
            )

        if level_reached:
            completed += 1

    return completed


def _support_height_at_x(
    course: Phase1StairCourse,
    x_m: float,
) -> float:
    if x_m < course.first_transition_x_m:
        return float(
            course.start_height_m
        )

    cursor = float(
        course.first_transition_x_m
    )

    for depth, height in zip(
        course.tread_depths_m,
        course.step_top_heights_m,
    ):
        end_x = cursor + float(depth)

        if x_m < end_x:
            return float(height)

        cursor = end_x

    return float(
        course.goal_height_m
    )


def _local_support_floor_height(
    course: Phase1StairCourse,
    x_m: float,
    *,
    wheel_radius_m: float,
) -> float:
    """Support floor under one wheel-radius neighbourhood.

    Taking the minimum adjacent support prevents a legitimate module
    straddling a stair edge from being labelled as below-support while it
    transfers onto the next tread.
    """

    return min(
        _support_height_at_x(
            course,
            x_m - wheel_radius_m,
        ),
        _support_height_at_x(
            course,
            x_m,
        ),
        _support_height_at_x(
            course,
            x_m + wheel_radius_m,
        ),
    )


def _inside_goal(
    course: Phase1StairCourse,
    point_xyz_m: XYZ,
    *,
    wheel_radius_m: float,
    vertical_tolerance_m: float,
) -> bool:
    x_m, y_m, z_m = (
        float(value)
        for value in point_xyz_m
    )

    goal_start = (
        course.goal_platform_start_x_m
    )

    goal_end = (
        goal_start
        + course.goal_platform_length_m
    )

    inside_xy = (
        goal_start
        <= x_m
        <= goal_end
        and abs(y_m)
        <= 0.5 * course.width_m
    )

    minimum_center_z = (
        course.goal_height_m
        + wheel_radius_m
        - vertical_tolerance_m
    )

    return bool(
        inside_xy
        and z_m >= minimum_center_z
    )


def stairs_task_success(
    course: Phase1StairCourse,
    module_xyz_m: Mapping[str, XYZ],
    *,
    wheel_radius_m: float | None = None,
    vertical_tolerance_m: float = 0.020,
) -> bool:
    """Frozen geometric success predicate; consumes only captured state."""
    owners = _validate_positions(module_xyz_m, name="next state")
    radius = SmoresGeometry().wheel_radius_m if wheel_radius_m is None else wheel_radius_m
    return all(_inside_goal(course, module_xyz_m[owner], wheel_radius_m=radius,
                            vertical_tolerance_m=vertical_tolerance_m) for owner in owners)


def compute_stairs_raw_reward(
    course: Phase1StairCourse,
    previous_module_xyz_m: Mapping[
        str,
        XYZ,
    ],
    next_module_xyz_m: Mapping[
        str,
        XYZ,
    ],
    *,
    signals: ActionOutcomeSignals | None = None,
    wheel_radius_m: float | None = None,
    vertical_tolerance_m: float = 0.020,
    shaping_gamma: float = 1.0,
    terminated: bool = False,
) -> StairsRawRewardFrame:
    """Compute unweighted reward terms for one stairs transition."""

    validate_shaping_discount(shaping_gamma)

    previous_owners = _validate_positions(
        previous_module_xyz_m,
        name="previous state",
    )

    next_owners = _validate_positions(
        next_module_xyz_m,
        name="next state",
    )

    if previous_owners != next_owners:
        raise ValueError(
            "Previous and next state must contain "
            "the same 8 module owners"
        )

    owners = next_owners

    if signals is None:
        signals = ActionOutcomeSignals()

    _validate_signals(
        signals,
        owners,
    )

    if wheel_radius_m is None:
        wheel_radius_m = (
            SmoresGeometry().wheel_radius_m
        )

    wheel_radius_m = float(
        wheel_radius_m
    )

    if (
        not math.isfinite(wheel_radius_m)
        or wheel_radius_m <= 0.0
    ):
        raise ValueError(
            "wheel_radius_m must be finite and positive"
        )

    vertical_tolerance_m = float(
        vertical_tolerance_m
    )

    if (
        not math.isfinite(vertical_tolerance_m)
        or vertical_tolerance_m < 0.0
    ):
        raise ValueError(
            "vertical_tolerance_m must be finite and nonnegative"
        )

    run_length_m = _task_run_length_m(
        course
    )

    elevation_change_m = (
        _elevation_change_m(
            course
        )
    )

    previous_inside = {
        owner: _inside_goal(
            course,
            previous_module_xyz_m[owner],
            wheel_radius_m=wheel_radius_m,
            vertical_tolerance_m=vertical_tolerance_m,
        )
        for owner in owners
    }

    next_inside = {
        owner: _inside_goal(
            course,
            next_module_xyz_m[owner],
            wheel_radius_m=wheel_radius_m,
            vertical_tolerance_m=vertical_tolerance_m,
        )
        for owner in owners
    }

    successful = all(
        next_inside.values()
    )

    owner_terms: dict[
        str,
        StairsOwnerRewardTerms,
    ] = {}

    half_width_m = (
        0.5 * float(course.width_m)
    )

    vertical_sign = (
        +1.0
        if course.direction == "ascent"
        else -1.0
    )

    step_count = float(
        course.step_count
    )

    for owner in owners:
        nx, ny, nz = (
            float(value)
            for value in next_module_xyz_m[
                owner
            ]
        )

        def potentials(point: XYZ, inside_goal: bool) -> tuple[float, ...]:
            x, y, z = (float(value) for value in point)
            support = _local_support_floor_height(course, x, wheel_radius_m=wheel_radius_m)
            start_x = course.start_platform_end_x_m - course.start_platform_length_m
            end_x = course.goal_platform_start_x_m + course.goal_platform_length_m
            valid = (
                start_x <= x <= end_x
                and abs(y) <= half_width_m
                and z >= support + wheel_radius_m - vertical_tolerance_m
            )
            if not valid:
                return 0.0, 0.0, 0.0, 0.0
            profile = min(1.0, max(0.0, (x - start_x) / run_length_m))
            # The tolerance offsets keep potentials nonnegative on legitimate
            # support while retaining the sign of vertical motion. Invalid
            # below-support descent resets potential instead of paying a fall.
            vertical = min(1.0, max(0.0, (
                vertical_sign * (z - course.start_height_m - wheel_radius_m)
                + vertical_tolerance_m
            ) / elevation_change_m))
            levels = _completed_levels(
                course, point, wheel_radius_m=wheel_radius_m,
                vertical_tolerance_m=vertical_tolerance_m,
            ) / step_count
            return profile, vertical, levels, float(inside_goal)

        profile_progress, vertical_progress, level_progress, goal_entry = (
            potential_difference(a, b, shaping_gamma=shaping_gamma, terminated=terminated)
            for a, b in zip(
                potentials(previous_module_xyz_m[owner], previous_inside[owner]),
                potentials(next_module_xyz_m[owner], next_inside[owner]),
            )
        )

        lateral_excess_m = max(
            0.0,
            abs(ny)
            - half_width_m,
        )

        outside_width = (
            lateral_excess_m
            / max(
                half_width_m,
                1.0e-9,
            )
        )

        support_height_m = (
            _local_support_floor_height(
                course,
                nx,
                wheel_radius_m=wheel_radius_m,
            )
        )

        minimum_center_z_m = (
            support_height_m
            + wheel_radius_m
            - vertical_tolerance_m
        )

        below_support = max(
            0.0,
            minimum_center_z_m
            - nz,
        ) / wheel_radius_m

        owner_terms[owner] = (
            StairsOwnerRewardTerms(
                profile_progress=float(
                    profile_progress
                ),
                vertical_progress=float(
                    vertical_progress
                ),
                level_progress=float(
                    level_progress
                ),
                goal_entry=float(
                    goal_entry
                ),
                success=float(
                    successful
                ),
                outside_width=float(
                    outside_width
                ),
                below_support=float(
                    below_support
                ),
                collision=float(
                    owner
                    in signals.collision_owner_ids
                ),
                conflict=float(
                    owner
                    in signals.conflict_owner_ids
                ),
                invalid=float(
                    owner
                    in signals.invalid_owner_ids
                ),
                redundant_action=float(
                    signals
                    .redundant_action_count_by_owner
                    .get(
                        owner,
                        0,
                    )
                ),
                control_effort=float(
                    signals
                    .control_effort_by_owner
                    .get(
                        owner,
                        0.0,
                    )
                ),
                time=1.0,
                stagnation=float(
                    bool(
                        signals.stagnating
                    )
                ),
            )
        )

    count = float(
        len(owners)
    )

    def mean(
        field_name: str,
    ) -> float:
        return sum(
            float(
                getattr(
                    owner_terms[owner],
                    field_name,
                )
            )
            for owner in owners
        ) / count

    team = StairsTeamRewardTerms(
        profile_progress=mean(
            "profile_progress"
        ),
        vertical_progress=mean(
            "vertical_progress"
        ),
        level_progress=mean(
            "level_progress"
        ),
        goal_entry=mean(
            "goal_entry"
        ),
        success=float(
            successful
        ),
        outside_width=mean(
            "outside_width"
        ),
        below_support=mean(
            "below_support"
        ),
        collision=mean(
            "collision"
        ),
        conflict=mean(
            "conflict"
        ),
        invalid=mean(
            "invalid"
        ),
        redundant_action=mean(
            "redundant_action"
        ),
        control_effort=mean(
            "control_effort"
        ),
        time=1.0,
        stagnation=float(
            bool(
                signals.stagnating
            )
        ),
    )

    return StairsRawRewardFrame(
        owners=owner_terms,
        team=team,
    )
