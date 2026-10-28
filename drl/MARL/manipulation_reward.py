"""Raw C1 MobileManipulator8 manipulation reward.

No weights live here.

C1 authority boundary:

    Expert assembly
    -> independent topology verification
    -> MARL_CONTROL

Therefore MARL owns the complete post-assembly task:

    approach
    -> end-effector positioning/orientation
    -> physical button depression

The reward deliberately follows the physical geometry already used
by the validated button expert:

- manipulation-ready landmark: 0.325 m button standoff;
- finite button-face overlap, not button-centre targeting;
- orientation shaping only outside the accepted 30 degree cone;
- physical success at 3.5 mm plunger depression;
- all seven MobileManipulator8 connections must remain present.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from drl.MARL.reward_potential import potential_difference, validate_shaping_discount

from drl.MARL.phase1_reward import (
    ActionOutcomeSignals,
)

from smores_ep.isaac.obstacle_course import (
    BUTTON_PLUNGER_DEPTH_M,
    BUTTON_PLUNGER_STROKE_M,
)

from smores_ep.isaac.phase1_manipulation_curriculum import (
    Phase1ManipulationCourse,
)


XYZ = tuple[
    float,
    float,
    float,
]


# Validated expert constants.
BUTTON_SUCCESS_DEPRESSION_M = (
    0.0035
)

ORIENTATION_DEADBAND_DEG = (
    30.0
)

MANIPULATION_REGION_HALF_WIDTH_M = (
    0.050
)

CONTACT_STARTED_DEPRESSION_M = (
    0.00010
)

# PhysX can report sub-micrometre numerical excursions around
# the hard physical plunger limits.  Values within this tolerance
# are interpreted as the corresponding physical endpoint.
BUTTON_DEPRESSION_NUMERICAL_TOLERANCE_M = (
    1.0e-6
)


# Physical button face:
# 80 x 80 mm button surface plus conservative
# 20 mm effective half-patch for end_effector.BOTTOM.
BUTTON_HALF_TANGENT_M = (
    0.040
)

BUTTON_HALF_Z_M = (
    0.040
)

EE_CONTACT_HALF_TANGENT_M = (
    0.020
)

EE_CONTACT_HALF_Z_M = (
    0.020
)


# A 30 mm change corresponds to the validated PRE gap.
# This keeps the raw EE progress term dimensionless.
EE_PROGRESS_SCALE_M = (
    0.030
)


@dataclass(frozen=True)
class ManipulationObservation:
    """Physical quantities required by the raw reward only."""

    module_xyz_m: Mapping[
        str,
        XYZ,
    ]

    end_effector_bottom_center_xyz_m: XYZ

    end_effector_bottom_normal_world: XYZ

    button_depression_m: float

    # MM8 has eight modules and therefore seven tree connections.
    connection_count: int


@dataclass(frozen=True)
class ManipulationOwnerRewardTerms:
    approach_progress: float
    ee_surface_progress: float
    orientation_progress: float
    depression_progress: float

    success: float
    connection_loss: float

    collision: float
    conflict: float
    invalid: float
    redundant_action: float
    control_effort: float

    time: float
    stagnation: float


@dataclass(frozen=True)
class ManipulationTeamRewardTerms:
    approach_progress: float
    ee_surface_progress: float
    orientation_progress: float
    depression_progress: float

    success: float
    connection_loss: float

    collision: float
    conflict: float
    invalid: float
    redundant_action: float
    control_effort: float

    time: float
    stagnation: float


@dataclass(frozen=True)
class ManipulationRawRewardFrame:
    owners: Mapping[
        str,
        ManipulationOwnerRewardTerms,
    ]

    team: ManipulationTeamRewardTerms


def _finite_xyz(
    value,
    *,
    label: str,
) -> XYZ:
    if (
        not isinstance(
            value,
            (tuple, list),
        )
        or len(value) != 3
    ):
        raise ValueError(
            f"{label} must be a length-3 vector"
        )

    result = tuple(
        float(item)
        for item
        in value
    )

    if not all(
        math.isfinite(item)
        for item
        in result
    ):
        raise ValueError(
            f"{label} must contain finite values"
        )

    return result  # type: ignore[return-value]


def _normalize(
    value,
    *,
    label: str,
) -> XYZ:
    x, y, z = (
        _finite_xyz(
            value,
            label=label,
        )
    )

    norm = math.sqrt(
        x*x
        + y*y
        + z*z
    )

    if norm <= 1.0e-12:
        raise ValueError(
            f"{label} must be nonzero"
        )

    return (
        x / norm,
        y / norm,
        z / norm,
    )


def _validate_modules(
    module_xyz_m: Mapping[
        str,
        XYZ,
    ],
    *,
    label: str,
) -> tuple[str, ...]:
    if not isinstance(
        module_xyz_m,
        Mapping,
    ):
        raise ValueError(
            f"{label} must be a mapping"
        )

    owners = tuple(
        sorted(
            module_xyz_m
        )
    )

    if (
        len(owners) != 8
        or any(
            not isinstance(
                owner,
                str,
            )
            or not owner
            for owner
            in owners
        )
    ):
        raise ValueError(
            f"{label} must contain exactly 8 string owners"
        )

    for owner in owners:
        _finite_xyz(
            module_xyz_m[
                owner
            ],
            label=(
                f"{label}[{owner}]"
            ),
        )

    return owners


def _physical_button_depression_m(
    observation: ManipulationObservation,
    *,
    label: str,
) -> float:
    """Return button depression projected onto its physical stroke."""

    depression = float(
        observation.button_depression_m
    )

    tolerance = (
        BUTTON_DEPRESSION_NUMERICAL_TOLERANCE_M
    )

    if (
        not math.isfinite(depression)
        or depression < -tolerance
        or depression
        > (
            BUTTON_PLUNGER_STROKE_M
            + tolerance
        )
    ):
        raise ValueError(
            f"{label}.button_depression_m is invalid"
        )

    return min(
        BUTTON_PLUNGER_STROKE_M,
        max(
            0.0,
            depression,
        ),
    )


def _validate_observation(
    observation: ManipulationObservation,
    *,
    label: str,
) -> tuple[str, ...]:
    if not isinstance(
        observation,
        ManipulationObservation,
    ):
        raise TypeError(
            f"{label} must be ManipulationObservation"
        )

    owners = (
        _validate_modules(
            observation.module_xyz_m,
            label=(
                f"{label}.module_xyz_m"
            ),
        )
    )

    _finite_xyz(
        observation
        .end_effector_bottom_center_xyz_m,
        label=(
            f"{label}.end_effector_bottom_center_xyz_m"
        ),
    )

    _normalize(
        observation
        .end_effector_bottom_normal_world,
        label=(
            f"{label}.end_effector_bottom_normal_world"
        ),
    )

    _physical_button_depression_m(
        observation,
        label=label,
    )

    if (
        type(
            observation.connection_count
        )
        is not int
        or observation.connection_count
        < 0
        or observation.connection_count
        > 7
    ):
        raise ValueError(
            f"{label}.connection_count must be 0..7"
        )

    return owners


def _validate_signals(
    signals: ActionOutcomeSignals,
    owners: tuple[
        str,
        ...,
    ],
) -> None:
    known = set(
        owners
    )

    for values in (
        signals.collision_owner_ids,
        signals.conflict_owner_ids,
        signals.invalid_owner_ids,
    ):
        unknown = (
            set(values)
            - known
        )

        if unknown:
            raise ValueError(
                "Runtime signal references unknown owner: "
                f"{sorted(unknown)}"
            )

    for mapping, label in (
        (
            signals
            .redundant_action_count_by_owner,
            "redundant action",
        ),
        (
            signals
            .control_effort_by_owner,
            "control effort",
        ),
    ):
        unknown = (
            set(mapping)
            - known
        )

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
            type(value)
            is not int
            or value < 0
        ):
            raise ValueError(
                "Redundant action count must be "
                f"nonnegative for {owner}"
            )

    for owner, value in (
        signals
        .control_effort_by_owner
        .items()
    ):
        numeric = float(
            value
        )

        if (
            not math.isfinite(
                numeric
            )
            or numeric < 0.0
        ):
            raise ValueError(
                "Control effort must be finite "
                f"and nonnegative for {owner}"
            )


def _task_axes(
    course: Phase1ManipulationCourse,
) -> tuple[
    XYZ,
    XYZ,
]:
    nx = float(
        course
        .press_direction_world_xy[0]
    )

    ny = float(
        course
        .press_direction_world_xy[1]
    )

    norm = math.hypot(
        nx,
        ny,
    )

    if norm <= 1.0e-12:
        raise ValueError(
            "Manipulation course has zero press direction"
        )

    nx /= norm
    ny /= norm

    normal = (
        nx,
        ny,
        0.0,
    )

    tangent = (
        -ny,
        nx,
        0.0,
    )

    return (
        normal,
        tangent,
    )


def _dot(
    first: XYZ,
    second: XYZ,
) -> float:
    return (
        first[0] * second[0]
        + first[1] * second[1]
        + first[2] * second[2]
    )


def _centroid_xy(
    module_xyz_m: Mapping[
        str,
        XYZ,
    ],
) -> tuple[
    float,
    float,
]:
    count = float(
        len(
            module_xyz_m
        )
    )

    return (
        sum(
            float(
                value[0]
            )
            for value
            in module_xyz_m.values()
        )
        / count,

        sum(
            float(
                value[1]
            )
            for value
            in module_xyz_m.values()
        )
        / count,
    )


def _standoff_m(
    course: Phase1ManipulationCourse,
    observation: ManipulationObservation,
) -> float:
    cx, cy = (
        _centroid_xy(
            observation
            .module_xyz_m
        )
    )

    bx, by, _ = (
        course
        .button_center_xyz_m
    )

    normal, _ = (
        _task_axes(
            course
        )
    )

    return (
        (bx - cx)
        * normal[0]
        + (by - cy)
        * normal[1]
    )


def _approach_error_m(
    course: Phase1ManipulationCourse,
    observation: ManipulationObservation,
) -> float:
    return abs(
        _standoff_m(
            course,
            observation,
        )
        - float(
            course
            .manipulation_ready_standoff_m
        )
    )


def _approach_span_m(
    course: Phase1ManipulationCourse,
) -> float:
    bx, by, _ = (
        course
        .button_center_xyz_m
    )

    sx, sy = (
        course
        .assembly_spawn_center_xy_m
    )

    normal, _ = (
        _task_axes(
            course
        )
    )

    assembly_standoff = (
        (bx - sx)
        * normal[0]
        + (by - sy)
        * normal[1]
    )

    return max(
        abs(
            assembly_standoff
            - float(
                course
                .manipulation_ready_standoff_m
            )
        ),
        1.0e-9,
    )


def _inside_manipulation_region(
    course: Phase1ManipulationCourse,
    observation: ManipulationObservation,
) -> bool:
    return (
        _approach_error_m(
            course,
            observation,
        )
        <= MANIPULATION_REGION_HALF_WIDTH_M
    )


def _surface_error_m(
    course: Phase1ManipulationCourse,
    observation: ManipulationObservation,
) -> float:
    """Distance to the admissible finite resting button-face region."""

    center = (
        _finite_xyz(
            observation
            .end_effector_bottom_center_xyz_m,
            label="end effector center",
        )
    )

    button = tuple(
        float(value)
        for value
        in course
        .button_center_xyz_m
    )

    normal, tangent = (
        _task_axes(
            course
        )
    )

    # Same physical definition as the expert:
    # robot-side resting face of the plunger.
    rest_front = tuple(
        button[index]
        - (
            0.5
            * BUTTON_PLUNGER_DEPTH_M
            * normal[index]
        )
        for index
        in range(3)
    )

    relative = (
        center[0]
        - rest_front[0],
        center[1]
        - rest_front[1],
        center[2]
        - rest_front[2],
    )

    tangent_coordinate = (
        _dot(
            relative,
            tangent,
        )
    )

    normal_coordinate = (
        _dot(
            relative,
            normal,
        )
    )

    tangent_half_extent = (
        BUTTON_HALF_TANGENT_M
        + EE_CONTACT_HALF_TANGENT_M
    )

    z_half_extent = (
        BUTTON_HALF_Z_M
        + EE_CONTACT_HALF_Z_M
    )

    tangent_excess = max(
        0.0,
        abs(
            tangent_coordinate
        )
        - tangent_half_extent,
    )

    z_excess = max(
        0.0,
        abs(
            relative[2]
        )
        - z_half_extent,
    )

    return math.sqrt(
        tangent_excess
        * tangent_excess
        + normal_coordinate
        * normal_coordinate
        + z_excess
        * z_excess
    )


def _normal_error_deg(
    course: Phase1ManipulationCourse,
    observation: ManipulationObservation,
) -> float:
    actual = (
        _normalize(
            observation
            .end_effector_bottom_normal_world,
            label="end effector normal",
        )
    )

    target, _ = (
        _task_axes(
            course
        )
    )

    cosine = min(
        1.0,
        max(
            -1.0,
            _dot(
                actual,
                target,
            ),
        ),
    )

    return math.degrees(
        math.acos(
            cosine
        )
    )


def manipulation_task_success(observation: ManipulationObservation) -> bool:
    """Physical 3.5 mm press with seven intended connections; no retreat."""
    _validate_observation(observation, label="next")
    depression = _physical_button_depression_m(
        observation,
        label="next",
    )

    return bool(
        depression
        >= BUTTON_SUCCESS_DEPRESSION_M
        and observation.connection_count == 7
    )


def compute_manipulation_raw_reward(
    course: Phase1ManipulationCourse,
    previous: ManipulationObservation,
    next_state: ManipulationObservation,
    *,
    signals: ActionOutcomeSignals | None = None,
    shaping_gamma: float = 1.0,
    terminated: bool = False,
) -> ManipulationRawRewardFrame:
    """Compute unweighted C1 manipulation reward terms."""

    validate_shaping_discount(shaping_gamma)

    previous_owners = (
        _validate_observation(
            previous,
            label="previous",
        )
    )

    next_owners = (
        _validate_observation(
            next_state,
            label="next",
        )
    )

    if (
        previous_owners
        != next_owners
    ):
        raise ValueError(
            "Previous and next manipulation states "
            "must contain the same 8 owners"
        )

    owners = (
        next_owners
    )

    if signals is None:
        signals = (
            ActionOutcomeSignals()
        )

    _validate_signals(
        signals,
        owners,
    )

    # Gate each endpoint potential, never the delta across two states. The
    # approach potential plateaus throughout the accepted region so ordinary
    # arm articulation does not create centroid progress. A continuous taper
    # activates the finer EE terms toward the 0.325 m geometric landmark.
    def potentials(observation: ManipulationObservation) -> tuple[float, ...]:
        error = _approach_error_m(course, observation)
        approach = max(0.0, 1.0 - max(
            0.0, error - MANIPULATION_REGION_HALF_WIDTH_M
        ) / _approach_span_m(course))
        region = max(0.0, 1.0 - error / MANIPULATION_REGION_HALF_WIDTH_M)
        surface_quality = 1.0 / (1.0 + _surface_error_m(course, observation) / EE_PROGRESS_SCALE_M)
        physical_depression = (
            _physical_button_depression_m(
                observation,
                label="reward",
            )
        )
        contact = min(
            1.0,
            physical_depression
            / CONTACT_STARTED_DEPRESSION_M,
        )
        # Once contact begins, EE quality smoothly saturates instead of turning
        # off and discarding the debit on a later release/retreat.
        surface = region * (contact + (1.0 - contact) * surface_quality)
        excess = max(0.0, _normal_error_deg(course, observation) - ORIENTATION_DEADBAND_DEG)
        orientation = region * (1.0 - excess / (180.0 - ORIENTATION_DEADBAND_DEG))
        depression = (
            physical_depression
            / BUTTON_SUCCESS_DEPRESSION_M
        )
        return approach, surface, orientation, depression

    previous_potential = potentials(previous)
    next_potential = potentials(next_state)
    approach_progress, ee_surface_progress, orientation_progress, depression_progress = (
        potential_difference(a, b, shaping_gamma=shaping_gamma, terminated=terminated)
        for a, b in zip(previous_potential, next_potential)
    )
    next_depression = (
        _physical_button_depression_m(
            next_state,
            label="next",
        )
    )

    connection_loss = (
        max(
            0,
            7
            - int(
                next_state
                .connection_count
            ),
        )
        / 7.0
    )

    successful = bool(
        next_depression
        >= BUTTON_SUCCESS_DEPRESSION_M
        and next_state.connection_count
        == 7
    )

    owner_terms: dict[
        str,
        ManipulationOwnerRewardTerms,
    ] = {}

    for owner in owners:
        owner_terms[
            owner
        ] = (
            ManipulationOwnerRewardTerms(
                approach_progress=float(
                    approach_progress
                ),

                ee_surface_progress=float(
                    ee_surface_progress
                ),

                orientation_progress=float(
                    orientation_progress
                ),

                depression_progress=float(
                    depression_progress
                ),

                success=float(
                    successful
                ),

                connection_loss=float(
                    connection_loss
                ),

                collision=float(
                    owner
                    in signals
                    .collision_owner_ids
                ),

                conflict=float(
                    owner
                    in signals
                    .conflict_owner_ids
                ),

                invalid=float(
                    owner
                    in signals
                    .invalid_owner_ids
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
        len(
            owners
        )
    )

    def mean(
        field_name: str,
    ) -> float:
        return (
            sum(
                float(
                    getattr(
                        owner_terms[
                            owner
                        ],
                        field_name,
                    )
                )
                for owner
                in owners
            )
            / count
        )

    team = (
        ManipulationTeamRewardTerms(
            approach_progress=float(
                approach_progress
            ),

            ee_surface_progress=float(
                ee_surface_progress
            ),

            orientation_progress=float(
                orientation_progress
            ),

            depression_progress=float(
                depression_progress
            ),

            success=float(
                successful
            ),

            connection_loss=float(
                connection_loss
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
    )

    return (
        ManipulationRawRewardFrame(
            owners=owner_terms,
            team=team,
        )
    )
