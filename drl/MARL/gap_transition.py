"""C1 Snake8 gap transition evaluation.

This module parallels the validated RC-Car8 and stairs transition paths.

DirectMARLEnv remains responsible for the physical transition ordering:

    s_t -> actuation -> physics -> s_(t+1)

The evaluator consumes that already accepted state pair.  It does not
perform another asynchronous simulator read.

No APPO weights are applied here.
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

from drl.MARL.gap_reward import (
    GapRawRewardFrame,
    compute_gap_raw_reward,
    gap_task_success,
)
from drl.MARL.phase1_reward import (
    ActionOutcomeSignals,
)
from drl.MARL.phase1_transition import (
    phase1_signals_from_native_step,
)
from smores_ep.isaac.phase1_gap_curriculum import (
    Phase1GapCourse,
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
class GapTransitionEvaluation:
    """Result of one physical C1 gap transition."""

    raw_reward: GapRawRewardFrame
    signals: ActionOutcomeSignals
    terminated: bool
    truncated: bool


class GapRawTermJSONLLogger:
    """Append unweighted gap reward terms to JSONL."""

    def __init__(
        self,
        path: Path | str,
    ) -> None:
        self.path = Path(
            path
        )

        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def __call__(
        self,
        record: Mapping[
            str,
            Any,
        ],
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


class GapTransitionEvaluator:
    """DirectMARLEnv-compatible evaluator for Snake8 gap episodes."""

    def __init__(
        self,
        *,
        courses: Sequence[
            Phase1GapCourse
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
    ) -> GapTransitionEvaluation:
        # The physical result has already been incorporated into next_state
        # by DirectMARLEnv. Never perform a second state read here.
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

        terminated = gap_task_success(course, next_state)

        raw_reward = (
            compute_gap_raw_reward(
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
            GapTransitionEvaluation(
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
        course: Phase1GapCourse,
        evaluation: GapTransitionEvaluation,
    ) -> dict[
        str,
        Any,
    ]:
        signals = (
            evaluation.signals
        )

        return {
            "schema_version":
                "mssr.gap_raw_terms.v1",

            "task":
                "gap",

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

                "gap_width_m":
                    course.gap_width_m,

                "near_edge_x_m":
                    course.near_edge_x_m,

                "far_edge_x_m":
                    course.far_edge_x_m,

                "landing_length_m":
                    course.landing_length_m,

                # Geometry only. This is NOT a separate reward target.
                "landing_center_y_m":
                    course.landing_center_y_m,
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
