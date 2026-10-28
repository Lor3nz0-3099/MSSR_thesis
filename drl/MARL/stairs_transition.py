"""C1 Snake8 stairs transition evaluation.

This module deliberately parallels the already-validated
``Phase1TransitionEvaluator`` used by RC-Car8.

It does not alter DirectMARLEnv, the physics step, state caching,
authority handoff, or the existing RC reward path.

The evaluator receives exactly the physical ``s_t`` and ``s_(t+1)``
already owned by DirectMARLEnv and converts them into:
- shared ActionOutcomeSignals;
- unweighted stairs raw reward terms;
- terminated/truncated flags.

Optional JSONL logging records raw terms only, for empirical reward-scale
calibration before APPO weights are selected.
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
    Mapping,
    Sequence,
)

from drl.MARL.reward_potential import validate_shaping_discount

from drl.MARL.phase1_reward import (
    ActionOutcomeSignals,
)
from drl.MARL.phase1_transition import (
    phase1_signals_from_native_step,
)
from drl.MARL.stairs_reward import (
    StairsRawRewardFrame,
    compute_stairs_raw_reward,
    stairs_task_success,
)
from smores_ep.isaac.phase1_stairs_curriculum import (
    Phase1StairCourse,
)


XYZ = tuple[
    float,
    float,
    float,
]

SignalsProvider = Callable[
    [
        int,
        Any,
        Mapping[str, XYZ],
        Mapping[str, XYZ],
    ],
    ActionOutcomeSignals,
]

RawTermLogger = Callable[
    [
        Mapping[str, Any],
    ],
    None,
]


@dataclass(frozen=True)
class StairsTransitionEvaluation:
    """Result of one physical C1 stairs transition."""

    raw_reward: StairsRawRewardFrame
    signals: ActionOutcomeSignals
    terminated: bool
    truncated: bool


class StairsRawTermJSONLLogger:
    """Append auditable unweighted stairs terms to JSONL."""

    def __init__(
        self,
        path: Path | str,
    ) -> None:
        self.path = Path(path)

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def __call__(
        self,
        record: Mapping[str, Any],
    ) -> None:
        payload = (
            json.dumps(
                record,
                sort_keys=True,
                separators=(
                    ",",
                    ":",
                ),
            )
            + "\n"
        )

        with self.path.open(
            "a",
            encoding="utf-8",
        ) as stream:
            stream.write(
                payload
            )
            stream.flush()


class StairsTransitionEvaluator:
    """DirectMARLEnv-compatible evaluator for Snake8 stairs."""

    def __init__(
        self,
        *,
        courses: Sequence[
            Phase1StairCourse
        ],
        max_episode_steps: int,
        shaping_gamma: float = 1.0,
        signals_provider: (
            SignalsProvider
            | None
        ) = None,
        raw_term_logger: (
            RawTermLogger
            | None
        ) = None,
    ) -> None:
        self.shaping_gamma = validate_shaping_discount(shaping_gamma)
        self._courses = tuple(
            courses
        )

        if not self._courses:
            raise ValueError(
                "courses must not be empty"
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
                "max_episode_steps must be "
                "a positive integer"
            )

        if (
            signals_provider is not None
            and not callable(
                signals_provider
            )
        ):
            raise TypeError(
                "signals_provider must be callable"
            )

        if (
            raw_term_logger is not None
            and not callable(
                raw_term_logger
            )
        ):
            raise TypeError(
                "raw_term_logger must be callable"
            )

        self._max_episode_steps = (
            max_episode_steps
        )

        self._signals_provider = (
            signals_provider
        )

        self._raw_term_logger = (
            raw_term_logger
        )

    def __call__(
        self,
        *,
        env_index: int,
        previous_state: Mapping[
            str,
            XYZ,
        ],
        next_state: Mapping[
            str,
            XYZ,
        ],
        actuation_result: Any,
        physics_result: Any,
        episode_step: int,
    ) -> StairsTransitionEvaluation:
        # DirectMARLEnv already owns the accepted post-step state.
        # physics_result is deliberately not used to perform another
        # asynchronous read.
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
            or env_index
            >= len(
                self._courses
            )
        ):
            raise IndexError(
                f"invalid env_index: {env_index}"
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
                "episode_step must be "
                "a positive integer"
            )

        course = self._courses[
            env_index
        ]

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
                "signals_provider must return "
                "ActionOutcomeSignals"
            )

        terminated = stairs_task_success(course, next_state)

        raw_reward = (
            compute_stairs_raw_reward(
                course,
                previous_state,
                next_state,
                signals=signals,
                shaping_gamma=self.shaping_gamma,
                terminated=terminated,
            )
        )

        truncated = bool(
            not terminated
            and episode_step
            >= self._max_episode_steps
        )

        evaluation = (
            StairsTransitionEvaluation(
                raw_reward=raw_reward,
                signals=signals,
                terminated=terminated,
                truncated=truncated,
            )
        )

        if (
            self._raw_term_logger
            is not None
        ):
            self._raw_term_logger(
                self._build_raw_record(
                    env_index=env_index,
                    episode_step=episode_step,
                    course=course,
                    evaluation=evaluation,
                )
            )

        return evaluation

    @staticmethod
    def _build_raw_record(
        *,
        env_index: int,
        episode_step: int,
        course: Phase1StairCourse,
        evaluation: StairsTransitionEvaluation,
    ) -> dict[str, Any]:
        signals = (
            evaluation.signals
        )

        return {
            "schema_version":
                "mssr.stairs_raw_terms.v1",

            "task":
                "stairs",

            "env_index":
                env_index,

            "episode_step":
                episode_step,

            "course": {
                "seed":
                    course.seed,

                "split":
                    course.split,

                "difficulty":
                    course.difficulty,

                "direction":
                    course.direction,

                "profile":
                    course.profile,

                "step_count":
                    course.step_count,
            },

            "team":
                asdict(
                    evaluation
                    .raw_reward
                    .team
                ),

            "owners": {
                owner:
                    asdict(
                        terms
                    )
                for owner, terms
                in sorted(
                    evaluation
                    .raw_reward
                    .owners
                    .items()
                )
            },

            "signals": {
                "collision_owner_ids":
                    sorted(
                        signals
                        .collision_owner_ids
                    ),

                "conflict_owner_ids":
                    sorted(
                        signals
                        .conflict_owner_ids
                    ),

                "invalid_owner_ids":
                    sorted(
                        signals
                        .invalid_owner_ids
                    ),

                "redundant_action_count_by_owner":
                    dict(
                        sorted(
                            signals
                            .redundant_action_count_by_owner
                            .items()
                        )
                    ),

                "control_effort_by_owner":
                    dict(
                        sorted(
                            signals
                            .control_effort_by_owner
                            .items()
                        )
                    ),

                "stagnating":
                    bool(
                        signals.stagnating
                    ),
            },

            "terminated":
                evaluation.terminated,

            "truncated":
                evaluation.truncated,
        }
