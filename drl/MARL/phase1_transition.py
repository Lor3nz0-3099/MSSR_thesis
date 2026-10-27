"""Phase-1 physical transition evaluation for DirectMARLEnv.

This module owns reward/done attribution only.

It receives already captured physical states and native runtime outcomes.
It neither advances Isaac physics nor exposes generator-only centerline
information to the actor.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from typing import Any

from drl.MARL.phase1_reward import (
    ActionOutcomeSignals,
    Phase1RawRewardFrame,
    compute_phase1_raw_reward,
)


XY = tuple[
    float,
    float,
]


@dataclass(frozen=True)
class Phase1TransitionEvaluation:
    """Unweighted Phase-1 reward and episode boundary for one env."""

    raw_reward: (
        Phase1RawRewardFrame
    )

    signals: (
        ActionOutcomeSignals
    )

    terminated: bool
    truncated: bool


def phase1_signals_from_native_step(
    actuation_result,
    *,
    collision_owner_ids=(),
    redundant_action_count_by_owner=None,
    control_effort_by_owner=None,
    stagnating=False,
) -> ActionOutcomeSignals:
    """Translate native admission outcomes into Phase-1 reward events.

    RESOURCE_BUSY -> conflict
    INVALID_GOAL -> invalid

    Infrastructure failures must never arrive here: they remain exceptions
    rather than silently becoming reward penalties.
    """

    conflict_owner_ids = (
        frozenset()
    )

    invalid_owner_ids = (
        frozenset()
    )

    proposal_outcome = getattr(
        actuation_result,
        "proposal_outcome",
        None,
    )

    if (
        proposal_outcome
        is not None
        and not bool(
            getattr(
                proposal_outcome,
                "accepted",
                False,
            )
        )
    ):
        code = str(
            getattr(
                proposal_outcome,
                "code",
                "",
            )
        )

        owner_ids = frozenset(
            str(
                owner_id
            )
            for owner_id
            in getattr(
                proposal_outcome,
                "owner_ids",
                (),
            )
        )

        if code == "RESOURCE_BUSY":
            conflict_owner_ids = (
                owner_ids
            )

        elif code == "INVALID_GOAL":
            invalid_owner_ids = (
                owner_ids
            )

        else:
            raise ValueError(
                "Unexpected policy rejection reached "
                f"Phase-1 reward attribution: {code}"
            )

    return ActionOutcomeSignals(
        collision_owner_ids=(
            frozenset(
                str(
                    owner_id
                )
                for owner_id
                in collision_owner_ids
            )
        ),
        conflict_owner_ids=(
            conflict_owner_ids
        ),
        invalid_owner_ids=(
            invalid_owner_ids
        ),
        redundant_action_count_by_owner=(
            {}
            if redundant_action_count_by_owner
            is None
            else dict(
                redundant_action_count_by_owner
            )
        ),
        control_effort_by_owner=(
            {}
            if control_effort_by_owner
            is None
            else dict(
                control_effort_by_owner
            )
        ),
        stagnating=bool(
            stagnating
        ),
    )


class Phase1TransitionEvaluator:
    """Evaluate the true s_t -> s_t+1 Phase-1 physical transition."""

    def __init__(
        self,
        *,
        courses: Sequence[Any],
        max_episode_steps: int,
        signals_provider: Callable[
            [
                int,
                Any,
                Mapping[
                    str,
                    XY,
                ],
                Mapping[
                    str,
                    XY,
                ],
            ],
            ActionOutcomeSignals,
        ] | None = None,
    ) -> None:
        self._courses = tuple(
            courses
        )

        if not self._courses:
            raise ValueError(
                "Phase1TransitionEvaluator requires at least one course"
            )

        if (
            isinstance(
                max_episode_steps,
                bool,
            )
            or not isinstance(
                max_episode_steps,
                int,
            )
            or max_episode_steps <= 0
        ):
            raise ValueError(
                "max_episode_steps must be a positive integer"
            )

        if (
            signals_provider
            is not None
            and not callable(
                signals_provider
            )
        ):
            raise TypeError(
                "signals_provider must be callable"
            )

        self.max_episode_steps = (
            max_episode_steps
        )

        self._signals_provider = (
            signals_provider
        )

    def __call__(
        self,
        *,
        env_index: int,
        previous_state: Mapping[
            str,
            XY,
        ],
        next_state: Mapping[
            str,
            XY,
        ],
        actuation_result,
        physics_result,
        episode_step: int,
    ) -> Phase1TransitionEvaluation:
        del physics_result

        if (
            isinstance(
                env_index,
                bool,
            )
            or not isinstance(
                env_index,
                int,
            )
            or env_index < 0
            or env_index >= len(
                self._courses
            )
        ):
            raise IndexError(
                f"invalid Phase-1 env_index: {env_index}"
            )

        if (
            isinstance(
                episode_step,
                bool,
            )
            or not isinstance(
                episode_step,
                int,
            )
            or episode_step <= 0
        ):
            raise ValueError(
                "episode_step must be a positive integer"
            )

        if (
            self._signals_provider
            is None
        ):
            signals = (
                phase1_signals_from_native_step(
                    actuation_result
                )
            )

        else:
            signals = (
                self._signals_provider(
                    env_index,
                    actuation_result,
                    previous_state,
                    next_state,
                )
            )

        if not isinstance(
            signals,
            ActionOutcomeSignals,
        ):
            raise TypeError(
                "signals_provider must return ActionOutcomeSignals"
            )

        course = (
            self._courses[
                env_index
            ]
        )

        raw_reward = (
            compute_phase1_raw_reward(
                course,
                previous_state,
                next_state,
                signals=signals,
            )
        )

        terminated = bool(
            course.success(
                next_state
            )
        )

        truncated = bool(
            not terminated
            and episode_step
            >= self.max_episode_steps
        )

        return (
            Phase1TransitionEvaluation(
                raw_reward=raw_reward,
                signals=signals,
                terminated=terminated,
                truncated=truncated,
            )
        )
