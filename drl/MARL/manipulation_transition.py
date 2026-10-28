"""C1 MobileManipulator8 transition evaluation.

This layer connects one real physical transition:

    s_t
    -> admitted native actuation
    -> physics
    -> s_(t+1)

to the unweighted manipulation reward ontology.

It deliberately contains:
- no PPO/APPO logic;
- no reward weights;
- no policy sampling;
- no deterministic Expert reposition.

The Expert authority boundary is already over before this evaluator
is active: MARL owns every post-assembly transition.
"""

from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
)
import json
from pathlib import Path
from typing import (
    Any,
    Callable,
)

from drl.MARL.reward_potential import validate_shaping_discount

from drl.MARL.manipulation_reward import (
    ActionOutcomeSignals,
    ManipulationObservation,
    ManipulationRawRewardFrame,
    compute_manipulation_raw_reward,
    manipulation_task_success,
)

from smores_ep.isaac.phase1_manipulation_curriculum import (
    Phase1ManipulationCourse,
)


MANIPULATION_RAW_TERMS_SCHEMA = (
    "mssr.manipulation_raw_terms.v1"
)


@dataclass(frozen=True)
class ManipulationTransitionEvaluation:
    """Unweighted result for one physical C1 manipulation transition."""

    env_index: int
    episode_step: int

    raw_reward: (
        ManipulationRawRewardFrame
    )

    success: bool
    terminated: bool

    failure_reason: (
        str
        | None
    )


class ManipulationRawTermsJsonlLogger:
    """Append-only logger for raw manipulation reward terms."""

    def __init__(
        self,
        path: Path,
    ) -> None:
        self.path = (
            Path(path)
            .expanduser()
            .resolve()
        )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def append(
        self,
        *,
        course: Phase1ManipulationCourse,
        evaluation: ManipulationTransitionEvaluation,
    ) -> None:
        record = {
            "schema_version":
                MANIPULATION_RAW_TERMS_SCHEMA,

            "env_index":
                int(
                    evaluation.env_index
                ),

            "episode_step":
                int(
                    evaluation.episode_step
                ),

            "course": {
                "seed":
                    int(
                        course.seed
                    ),

                "split":
                    str(
                        course.split
                    ),

                "difficulty":
                    str(
                        course.difficulty
                    ),
            },

            "terminal":
                bool(
                    evaluation.terminated
                ),

            "success":
                bool(
                    evaluation.success
                ),

            "failure_reason":
                evaluation.failure_reason,

            "team_raw_terms":
                asdict(
                    evaluation
                    .raw_reward
                    .team
                ),

            "owner_raw_terms": {
                str(owner):
                    asdict(terms)
                for owner, terms
                in evaluation
                .raw_reward
                .owners
                .items()
            },
        }

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as stream:
            stream.write(
                json.dumps(
                    record,
                    separators=(
                        ",",
                        ":",
                    ),
                    sort_keys=True,
                )
                + "\n"
            )


class ManipulationTransitionEvaluator:
    """Evaluate one post-assembly MobileManipulator8 MARL transition."""

    def __init__(
        self,
        *,
        course: Phase1ManipulationCourse,
        observation_builder: (
            Callable[
                [Any],
                ManipulationObservation,
            ]
            | None
        ) = None,
        signal_builder: (
            Callable[..., ActionOutcomeSignals]
            | None
        ) = None,
        logger: (
            ManipulationRawTermsJsonlLogger
            | None
        ) = None,
        terminate_on_connection_loss: bool = True,
        shaping_gamma: float = 1.0,
    ) -> None:
        if not isinstance(
            course,
            Phase1ManipulationCourse,
        ):
            raise TypeError(
                "course must be Phase1ManipulationCourse"
            )

        if (
            observation_builder
            is not None
            and not callable(
                observation_builder
            )
        ):
            raise TypeError(
                "observation_builder must be callable"
            )

        if (
            signal_builder
            is not None
            and not callable(
                signal_builder
            )
        ):
            raise TypeError(
                "signal_builder must be callable"
            )

        if (
            logger is not None
            and not isinstance(
                logger,
                ManipulationRawTermsJsonlLogger,
            )
        ):
            raise TypeError(
                "logger must be ManipulationRawTermsJsonlLogger"
            )

        self.shaping_gamma = validate_shaping_discount(shaping_gamma)
        self.course = course

        self.observation_builder = (
            observation_builder
        )

        self.signal_builder = (
            signal_builder
        )

        self.logger = logger

        self.terminate_on_connection_loss = bool(
            terminate_on_connection_loss
        )

    def _observation(
        self,
        state: Any,
    ) -> ManipulationObservation:
        if self.observation_builder is None:
            observation = state
        else:
            observation = (
                self.observation_builder(
                    state
                )
            )

        if not isinstance(
            observation,
            ManipulationObservation,
        ):
            raise TypeError(
                "Manipulation observation builder "
                "must return ManipulationObservation"
            )

        return observation

    def _signals(
        self,
        *,
        env_index,
        previous_state,
        next_state,
        actuation_result,
        physics_result,
        episode_step,
    ) -> ActionOutcomeSignals:
        if self.signal_builder is None:
            return (
                ActionOutcomeSignals()
            )

        signals = (
            self.signal_builder(
                env_index=env_index,
                previous_state=previous_state,
                next_state=next_state,
                actuation_result=actuation_result,
                physics_result=physics_result,
                episode_step=episode_step,
            )
        )

        if not isinstance(
            signals,
            ActionOutcomeSignals,
        ):
            raise TypeError(
                "Manipulation signal builder "
                "must return ActionOutcomeSignals"
            )

        return signals

    def __call__(
        self,
        *,
        env_index,
        previous_state,
        next_state,
        actuation_result,
        physics_result,
        episode_step,
    ) -> ManipulationTransitionEvaluation:
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
        ):
            raise ValueError(
                "env_index must be a nonnegative integer"
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
            or episode_step < 0
        ):
            raise ValueError(
                "episode_step must be a nonnegative integer"
            )

        previous_observation = (
            self._observation(
                previous_state
            )
        )

        next_observation = (
            self._observation(
                next_state
            )
        )

        signals = (
            self._signals(
                env_index=env_index,
                previous_state=previous_state,
                next_state=next_state,
                actuation_result=actuation_result,
                physics_result=physics_result,
                episode_step=episode_step,
            )
        )

        success = manipulation_task_success(next_observation)

        connection_failed = bool(
            self.terminate_on_connection_loss
            and next_observation
            .connection_count
            < 7
        )

        failure_reason = (
            "connection_loss"
            if (
                connection_failed
                and not success
            )
            else None
        )

        terminated = bool(
            success
            or connection_failed
        )

        raw_reward = (
            compute_manipulation_raw_reward(
                self.course,
                previous_observation,
                next_observation,
                signals=signals,
                shaping_gamma=self.shaping_gamma,
                terminated=terminated,
            )
        )

        evaluation = (
            ManipulationTransitionEvaluation(
                env_index=env_index,
                episode_step=episode_step,
                raw_reward=raw_reward,
                success=success,
                terminated=terminated,
                failure_reason=failure_reason,
            )
        )

        if self.logger is not None:
            self.logger.append(
                course=self.course,
                evaluation=evaluation,
            )

        return evaluation
