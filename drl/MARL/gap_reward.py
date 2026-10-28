"""Unweighted raw reward terms for the C1 Snake8 gap task.

No APPO reward weights live in this module.

The gap is treated as three longitudinal phases:

    approach -> crossing -> landing

A landing-bank Y offset remains part of the physical support geometry,
but is deliberately NOT exposed as a lateral-alignment reward target.

Progress is signed so backward motion naturally produces negative raw
progress.  A module below the support plane cannot earn progress merely
by continuing to move in +X while falling.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from drl.MARL.phase1_reward import (
    ActionOutcomeSignals,
    _validate_signals,
)
from drl.MARL.reward_potential import (
    potential_difference,
    validate_owner_xyz_positions as _validate_positions,
    validate_shaping_discount,
)
from smores_ep.config.geometry import (
    SmoresGeometry,
)
from smores_ep.isaac.phase1_gap_curriculum import (
    Phase1GapCourse,
)


XYZ = tuple[
    float,
    float,
    float,
]


@dataclass(frozen=True)
class GapOwnerRewardTerms:
    approach_progress: float = 0.0
    crossing_progress: float = 0.0
    landing_progress: float = 0.0

    goal_entry: float = 0.0
    fall: float = 0.0

    collision: float = 0.0
    conflict: float = 0.0
    invalid: float = 0.0

    redundant_action: float = 0.0
    control_effort: float = 0.0

    time: float = 1.0


@dataclass(frozen=True)
class GapTeamRewardTerms:
    approach_progress: float = 0.0
    crossing_progress: float = 0.0
    landing_progress: float = 0.0

    goal_entry: float = 0.0
    success: float = 0.0
    fall: float = 0.0

    collision: float = 0.0
    conflict: float = 0.0
    invalid: float = 0.0

    redundant_action: float = 0.0
    control_effort: float = 0.0

    time: float = 1.0
    stagnation: float = 0.0


@dataclass(frozen=True)
class GapRawRewardFrame:
    owners: Mapping[
        str,
        GapOwnerRewardTerms,
    ]

    team: GapTeamRewardTerms


def _clamp(
    value: float,
    low: float,
    high: float,
) -> float:
    return max(
        low,
        min(
            high,
            value,
        ),
    )


def _bank_top_z_m(
    course: Phase1GapCourse,
) -> float:
    bank = course.approach_bank

    return (
        float(
            bank.center_xyz_m[2]
        )
        + 0.5
        * float(
            bank.size_xyz_m[2]
        )
    )


def _is_supported_height(
    z_m: float,
    *,
    support_center_z_m: float,
    wheel_radius_m: float,
    tolerance_m: float,
) -> bool:
    expected_center_z_m = (
        support_center_z_m
        + wheel_radius_m
    )

    return (
        z_m
        >= expected_center_z_m
        - tolerance_m
    )


def _inside_landing_xy(
    course: Phase1GapCourse,
    *,
    x_m: float,
    y_m: float,
) -> bool:
    half_width_m = (
        0.5
        * course.bank_width_m
    )

    return (
        course.far_edge_x_m
        <= x_m
        <= course.landing_end_x_m

        and

        abs(
            y_m
            - course.landing_center_y_m
        )
        <= half_width_m
    )


def _on_landing(
    course: Phase1GapCourse,
    *,
    x_m: float,
    y_m: float,
    z_m: float,
    bank_top_z_m: float,
    wheel_radius_m: float,
    tolerance_m: float,
) -> bool:
    return (
        _inside_landing_xy(
            course,
            x_m=x_m,
            y_m=y_m,
        )
        and
        _is_supported_height(
            z_m,
            support_center_z_m=bank_top_z_m,
            wheel_radius_m=wheel_radius_m,
            tolerance_m=tolerance_m,
        )
    )


def _approach_coordinate(
    course: Phase1GapCourse,
    x_m: float,
) -> float:
    return _clamp(
        x_m,
        course.approach_start_x_m,
        course.near_edge_x_m,
    )


def _crossing_coordinate(
    course: Phase1GapCourse,
    x_m: float,
) -> float:
    return _clamp(
        (
            x_m
            - course.near_edge_x_m
        )
        / course.gap_width_m,
        0.0,
        1.0,
    )


def _landing_coordinate(
    course: Phase1GapCourse,
    x_m: float,
) -> float:
    return _clamp(
        x_m,
        course.far_edge_x_m,
        course.landing_end_x_m,
    )


def gap_task_success(
    course: Phase1GapCourse,
    module_xyz_m: Mapping[str, XYZ],
    *,
    wheel_radius_m: float | None = None,
    support_tolerance_m: float = 0.020,
) -> bool:
    """All eight captured owners must be physically supported on landing."""
    owners = _validate_positions(module_xyz_m, name="next state")
    radius = SmoresGeometry().wheel_radius_m if wheel_radius_m is None else wheel_radius_m
    return all(_on_landing(
        course, x_m=float(module_xyz_m[owner][0]), y_m=float(module_xyz_m[owner][1]),
        z_m=float(module_xyz_m[owner][2]), bank_top_z_m=_bank_top_z_m(course),
        wheel_radius_m=radius, tolerance_m=support_tolerance_m,
    ) for owner in owners)


def compute_gap_raw_reward(
    course: Phase1GapCourse,
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
    support_tolerance_m: float = 0.02,
    shaping_gamma: float = 1.0,
    terminated: bool = False,
) -> GapRawRewardFrame:
    """Compute raw, unweighted C1 gap terms."""

    validate_shaping_discount(shaping_gamma)
    # All eight physical owners are required on both sides. No module
    # identity coercion is allowed.
    previous_owners = _validate_positions(previous_module_xyz_m, name="previous state")
    next_owners = _validate_positions(next_module_xyz_m, name="next state")
    if previous_owners != next_owners:
        raise ValueError("previous/next must contain the same 8 owners")

    if wheel_radius_m is None:
        wheel_radius_m = (
            SmoresGeometry()
            .wheel_radius_m
        )

    if (
        not math.isfinite(
            wheel_radius_m
        )
        or wheel_radius_m <= 0.0
    ):
        raise ValueError(
            "wheel_radius_m must be finite and positive"
        )

    if (
        not math.isfinite(
            support_tolerance_m
        )
        or support_tolerance_m < 0.0
    ):
        raise ValueError(
            "support_tolerance_m must be finite and non-negative"
        )

    if signals is None:
        signals = (
            ActionOutcomeSignals()
        )

    if not isinstance(
        signals,
        ActionOutcomeSignals,
    ):
        raise TypeError(
            "signals must be ActionOutcomeSignals"
        )

    _validate_signals(signals, next_owners)

    bank_top_z_m = (
        _bank_top_z_m(
            course
        )
    )

    owners: dict[
        str,
        GapOwnerRewardTerms,
    ] = {}

    for owner in sorted(
        previous_owners
    ):
        previous_xyz = (
            previous_module_xyz_m[
                owner
            ]
        )

        next_xyz = (
            next_module_xyz_m[
                owner
            ]
        )

        if (
            len(previous_xyz) != 3
            or len(next_xyz) != 3
        ):
            raise ValueError(
                "Gap reward states must contain XYZ triples"
            )

        px, py, pz = (
            float(value)
            for value in previous_xyz
        )

        nx, ny, nz = (
            float(value)
            for value in next_xyz
        )

        if not all(
            math.isfinite(value)
            for value in (
                px,
                py,
                pz,
                nx,
                ny,
                nz,
            )
        ):
            raise ValueError(
                "Gap reward state values must be finite"
            )

        next_supported = (
            _is_supported_height(
                nz,
                support_center_z_m=bank_top_z_m,
                wheel_radius_m=wheel_radius_m,
                tolerance_m=support_tolerance_m,
            )
        )

        fall = (
            0.0
            if next_supported
            else 1.0
        )

        def potentials(x: float, y: float, z: float) -> tuple[float, ...]:
            supported = _is_supported_height(
                z, support_center_z_m=bank_top_z_m,
                wheel_radius_m=wheel_radius_m, tolerance_m=support_tolerance_m,
            )
            landing = _on_landing(
                course, x_m=x, y_m=y, z_m=z, bank_top_z_m=bank_top_z_m,
                wheel_radius_m=wheel_radius_m, tolerance_m=support_tolerance_m,
            )
            if not supported:
                return 0.0, 0.0, 0.0, 0.0
            # Longitudinal potentials retain original metre units for approach
            # and landing. No landing-bank centering target is introduced.
            return (
                _approach_coordinate(course, x) - course.approach_start_x_m,
                _crossing_coordinate(course, x),
                _landing_coordinate(course, x) - course.far_edge_x_m,
                float(landing),
            )

        approach_progress, crossing_progress, landing_progress, goal_entry = (
            potential_difference(a, b, shaping_gamma=shaping_gamma, terminated=terminated)
            for a, b in zip(potentials(px, py, pz), potentials(nx, ny, nz))
        )

        owners[
            owner
        ] = GapOwnerRewardTerms(
            approach_progress=(
                approach_progress
            ),

            crossing_progress=(
                crossing_progress
            ),

            landing_progress=(
                landing_progress
            ),

            goal_entry=(
                goal_entry
            ),

            fall=fall,

            collision=(
                1.0
                if owner
                in signals.collision_owner_ids
                else 0.0
            ),

            conflict=(
                1.0
                if owner
                in signals.conflict_owner_ids
                else 0.0
            ),

            invalid=(
                1.0
                if owner
                in signals.invalid_owner_ids
                else 0.0
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
        )

    owner_terms = tuple(
        owners.values()
    )

    count = float(
        len(
            owner_terms
        )
    )

    all_on_landing = all(
        _on_landing(
            course,
            x_m=float(
                next_module_xyz_m[
                    owner
                ][0]
            ),
            y_m=float(
                next_module_xyz_m[
                    owner
                ][1]
            ),
            z_m=float(
                next_module_xyz_m[
                    owner
                ][2]
            ),
            bank_top_z_m=bank_top_z_m,
            wheel_radius_m=wheel_radius_m,
            tolerance_m=support_tolerance_m,
        )
        for owner
        in sorted(
            next_owners
        )
    )

    team = GapTeamRewardTerms(
        approach_progress=(
            sum(
                term.approach_progress
                for term in owner_terms
            )
            / count
        ),

        crossing_progress=(
            sum(
                term.crossing_progress
                for term in owner_terms
            )
            / count
        ),

        landing_progress=(
            sum(
                term.landing_progress
                for term in owner_terms
            )
            / count
        ),

        goal_entry=(
            sum(
                term.goal_entry
                for term in owner_terms
            )
            / count
        ),

        success=(
            1.0
            if all_on_landing
            else 0.0
        ),

        fall=sum(
            term.fall
            for term in owner_terms
        ),

        collision=sum(
            term.collision
            for term in owner_terms
        ),

        conflict=sum(
            term.conflict
            for term in owner_terms
        ),

        invalid=sum(
            term.invalid
            for term in owner_terms
        ),

        redundant_action=sum(
            term.redundant_action
            for term in owner_terms
        ),

        control_effort=sum(
            term.control_effort
            for term in owner_terms
        ),

        time=1.0,

        stagnation=(
            1.0
            if signals.stagnating
            else 0.0
        ),
    )

    return GapRawRewardFrame(
        owners=owners,
        team=team,
    )
