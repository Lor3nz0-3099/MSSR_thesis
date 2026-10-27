"""Raw Phase-1 RC-Car8 reward terms.

This module deliberately does NOT assign reward weights.

It converts one physical transition plus already-audited runtime events
into dimensionless or normalized raw terms that can later be weighted by
the MAPPO/APPO training configuration.

Important:
- active action count is NOT itself penalized;
- redundant_action is supplied explicitly by runtime attribution;
- conflict / invalid / collision remain distinct signals;
- centerline geometry is generator-side reward information, not actor input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

from smores_ep.isaac.phase1_training_course import (
    Phase1TrainingCourse,
)


XY = tuple[float, float]


@dataclass(frozen=True)
class ActionOutcomeSignals:
    """Post-action physical/runtime events attributed to conceptual owners."""

    collision_owner_ids: frozenset[str] = frozenset()
    conflict_owner_ids: frozenset[str] = frozenset()
    invalid_owner_ids: frozenset[str] = frozenset()

    redundant_action_count_by_owner: Mapping[str, int] = field(
        default_factory=dict
    )

    control_effort_by_owner: Mapping[str, float] = field(
        default_factory=dict
    )

    # Stateful stagnation detection belongs to the environment/runtime.
    # The reward engine only consumes the resulting event.
    stagnating: bool = False


@dataclass(frozen=True)
class Phase1OwnerRewardTerms:
    progress: float
    target_entry: float
    success: float

    outside_road: float

    collision: float
    conflict: float
    invalid: float
    redundant_action: float
    control_effort: float

    time: float
    stagnation: float


@dataclass(frozen=True)
class Phase1TeamRewardTerms:
    progress: float
    target_entry: float
    success: float

    outside_road: float

    collision: float
    conflict: float
    invalid: float
    redundant_action: float
    control_effort: float

    time: float
    stagnation: float


@dataclass(frozen=True)
class Phase1RawRewardFrame:
    owners: Mapping[str, Phase1OwnerRewardTerms]
    team: Phase1TeamRewardTerms


@dataclass(frozen=True)
class _Projection:
    s_m: float
    lateral_distance_m: float


def _route_geometry(
    course: Phase1TrainingCourse,
) -> tuple[tuple[float, ...], float]:
    cumulative = [0.0]

    for first, second in zip(
        course.centerline_xy_m[:-1],
        course.centerline_xy_m[1:],
    ):
        segment = math.hypot(
            second[0] - first[0],
            second[1] - first[1],
        )

        if segment <= 1.0e-9:
            continue

        cumulative.append(
            cumulative[-1] + segment
        )

    total = cumulative[-1]

    if total <= 1.0e-9:
        raise ValueError(
            "Phase-1 centerline has zero length"
        )

    return tuple(cumulative), total


def _project_to_centerline(
    course: Phase1TrainingCourse,
    point_xy_m: XY,
) -> _Projection:
    """Closest-point projection onto the piecewise-linear route."""

    px = float(point_xy_m[0])
    py = float(point_xy_m[1])

    if not (
        math.isfinite(px)
        and math.isfinite(py)
    ):
        raise ValueError(
            "Module XY position must be finite"
        )

    best_distance_sq = math.inf
    best_s = 0.0

    cumulative_s = 0.0

    for start, end in zip(
        course.centerline_xy_m[:-1],
        course.centerline_xy_m[1:],
    ):
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        length_sq = dx * dx + dy * dy

        if length_sq <= 1.0e-18:
            continue

        segment_length = math.sqrt(
            length_sq
        )

        alpha = (
            (px - start[0]) * dx
            + (py - start[1]) * dy
        ) / length_sq

        alpha = min(
            1.0,
            max(
                0.0,
                alpha,
            ),
        )

        qx = start[0] + alpha * dx
        qy = start[1] + alpha * dy

        distance_sq = (
            (px - qx) ** 2
            + (py - qy) ** 2
        )

        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_s = (
                cumulative_s
                + alpha * segment_length
            )

        cumulative_s += segment_length

    if not math.isfinite(best_distance_sq):
        raise ValueError(
            "Phase-1 centerline has no valid segment"
        )

    return _Projection(
        s_m=best_s,
        lateral_distance_m=math.sqrt(
            best_distance_sq
        ),
    )


def _validate_module_positions(
    positions: Mapping[str, XY],
    *,
    name: str,
) -> tuple[str, ...]:
    owners = tuple(
        sorted(
            str(owner)
            for owner in positions
        )
    )

    if len(owners) != 8:
        raise ValueError(
            f"{name} must contain exactly 8 module owners"
        )

    if len(set(owners)) != 8:
        raise ValueError(
            f"{name} contains duplicate module owners"
        )

    for owner in owners:
        point = positions[owner]

        if (
            not isinstance(
                point,
                (tuple, list),
            )
            or len(point) != 2
            or not all(
                math.isfinite(float(value))
                for value in point
            )
        ):
            raise ValueError(
                f"Invalid XY position for {owner}"
            )

    return owners


def _validate_signals(
    signals: ActionOutcomeSignals,
    owners: tuple[str, ...],
) -> None:
    known = set(owners)

    event_sets = (
        signals.collision_owner_ids,
        signals.conflict_owner_ids,
        signals.invalid_owner_ids,
    )

    for values in event_sets:
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
        signals.redundant_action_count_by_owner.items()
    ):
        if (
            type(value) is not int
            or value < 0
        ):
            raise ValueError(
                "Redundant action count must be "
                f"a nonnegative integer for {owner}"
            )

    for owner, value in (
        signals.control_effort_by_owner.items()
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


def compute_phase1_raw_reward(
    course: Phase1TrainingCourse,
    previous_module_xy_m: Mapping[str, XY],
    next_module_xy_m: Mapping[str, XY],
    *,
    signals: ActionOutcomeSignals | None = None,
) -> Phase1RawRewardFrame:
    """Compute unweighted reward terms for one Phase-1 transition."""

    previous_owners = _validate_module_positions(
        previous_module_xy_m,
        name="previous state",
    )

    next_owners = _validate_module_positions(
        next_module_xy_m,
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

    _, route_length_m = _route_geometry(
        course
    )

    previous_projection = {
        owner: _project_to_centerline(
            course,
            previous_module_xy_m[owner],
        )
        for owner in owners
    }

    next_projection = {
        owner: _project_to_centerline(
            course,
            next_module_xy_m[owner],
        )
        for owner in owners
    }

    previous_inside = {
        owner:
            course.target_platform.contains_xy(
                previous_module_xy_m[owner]
            )
        for owner in owners
    }

    next_inside = {
        owner:
            course.target_platform.contains_xy(
                next_module_xy_m[owner]
            )
        for owner in owners
    }

    successful = course.success(
        next_module_xy_m
    )

    half_road_width_m = (
        0.5 * course.road_width_m
    )

    owner_terms: dict[
        str,
        Phase1OwnerRewardTerms,
    ] = {}

    for owner in owners:
        progress = (
            next_projection[owner].s_m
            - previous_projection[owner].s_m
        ) / route_length_m

        target_entry = float(
            int(next_inside[owner])
            - int(previous_inside[owner])
        )

        lateral_excess_m = max(
            0.0,
            next_projection[
                owner
            ].lateral_distance_m
            - half_road_width_m,
        )

        # Dimensionless: 1.0 means one half-road-width
        # beyond the nominal road edge.
        outside_road = (
            lateral_excess_m
            / max(
                half_road_width_m,
                1.0e-9,
            )
        )

        owner_terms[owner] = (
            Phase1OwnerRewardTerms(
                progress=float(progress),
                target_entry=float(
                    target_entry
                ),
                success=float(
                    successful
                ),

                outside_road=float(
                    outside_road
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

    team = Phase1TeamRewardTerms(
        progress=mean(
            "progress"
        ),

        target_entry=(
            sum(
                int(next_inside[owner])
                - int(previous_inside[owner])
                for owner in owners
            )
            / count
        ),

        success=float(
            successful
        ),

        outside_road=mean(
            "outside_road"
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

    return Phase1RawRewardFrame(
        owners=owner_terms,
        team=team,
    )
