"""Conservative quality gate for C1 actor-critic training.

Fit against TD(lambda) returns is an in-sample diagnostic,
not proof of out-of-sample critic accuracy.
Thresholds are provisional pilot defaults.
"""

from dataclasses import dataclass, replace
import math

from drl.MARL.c1_longtrain_schedule import C1LongtrainSchedule


@dataclass(frozen=True)
class GateConfig:
    min_frozen_updates: int = 4
    max_frozen_updates: int = 12
    consecutive_good: int = 3
    min_samples_per_critic: int = 16
    min_explained_variance: float = 0.0
    max_normalized_rmse: float = 1.5

    def __post_init__(self):
        if not (
            1 <= self.min_frozen_updates < self.max_frozen_updates
        ):
            raise ValueError("Invalid warm-up bounds")

        if (
            self.consecutive_good < 2
            or self.min_samples_per_critic < 2
        ):
            raise ValueError("Insufficient gate evidence")

        if not -1.0 <= self.min_explained_variance <= 1.0:
            raise ValueError("Invalid EV threshold")

        if (
            not math.isfinite(self.max_normalized_rmse)
            or self.max_normalized_rmse <= 0
        ):
            raise ValueError("Invalid normalized RMSE bound")


class C1CriticQualityGate:
    SCHEMA = "mssr.c1.critic_quality_gate.v1"

    def __init__(self, config=None):
        self.config = config if config is not None else GateConfig()
        self.last_completed = 0
        self.good_streak = 0
        self.first_actor_update = None
        self.last_ok = False
        self.last_reason = "awaiting_first_update"

    @property
    def exhausted(self):
        return (
            self.first_actor_update is None
            and self.last_completed >= self.config.max_frozen_updates
        )

    def step_for_update(self, update):
        if (
            type(update) is not int
            or update != self.last_completed + 1
        ):
            raise ValueError("Gate/update sequence mismatch")

        if self.exhausted:
            raise RuntimeError(
                "Critic warm-up exhausted: actor remains frozen"
            )

        schedule = C1LongtrainSchedule()

        if self.first_actor_update is None:
            return replace(
                schedule.for_update(1),
                update=update,
            )

        # La rampa comincia dal primo vero update dell'actor.
        relative = update - self.first_actor_update + 5

        return replace(
            schedule.for_update(relative),
            update=update,
        )

    def observe_completed(self, update, metrics):
        if (
            type(update) is not int
            or update != self.last_completed + 1
        ):
            raise ValueError(
                "Nonconsecutive completed PPO update"
            )

        if self.first_actor_update is not None:
            self.last_completed = update
            return

        bad = []
        config = self.config

        for prefix in ("owner", "mean_field"):
            keys = (
                f"{prefix}_fit_count",
                f"{prefix}_fit_evaluable",
                f"{prefix}_fit_explained_variance",
                f"{prefix}_fit_rmse",
                f"{prefix}_fit_return_std",
            )

            if any(key not in metrics for key in keys):
                bad.append(prefix + ":missing")
                continue

            n, evaluable, ev, rmse, std = (
                float(metrics[key]) for key in keys
            )

            if not all(
                math.isfinite(x)
                for x in (n, evaluable, ev, rmse, std)
            ):
                raise ValueError(
                    "Nonfinite critic gate metric"
                )

            if (
                n < config.min_samples_per_critic
                or evaluable != 1
                or std <= 1e-5
            ):
                bad.append(prefix + ":insufficient_data")

            elif (
                ev < config.min_explained_variance
                or rmse / std > config.max_normalized_rmse
            ):
                bad.append(prefix + ":poor_fit")

        self.last_ok = not bad

        self.last_reason = (
            "both_critics_fit"
            if not bad
            else ",".join(bad)
        )

        self.good_streak = (
            self.good_streak + 1
            if self.last_ok
            else 0
        )

        self.last_completed = update

        if (
            update >= config.min_frozen_updates
            and self.good_streak >= config.consecutive_good
        ):
            self.first_actor_update = update + 1
            self.last_reason = "actor_unlock_scheduled"

    def metrics(self):
        return {
            "c1_gate_good_streak": self.good_streak,
            "c1_gate_both_ok": int(self.last_ok),
            "c1_gate_unlocked": int(
                self.first_actor_update is not None
            ),
            "c1_gate_exhausted": int(self.exhausted),
        }

    def state_dict(self):
        return {
            "schema": self.SCHEMA,
            "config": vars(self.config).copy(),
            "last_completed": self.last_completed,
            "good_streak": self.good_streak,
            "first_actor_update": self.first_actor_update,
            "last_ok": self.last_ok,
            "last_reason": self.last_reason,
        }

    @classmethod
    def from_state_dict(cls, state):
        if (
            not isinstance(state, dict)
            or state.get("schema") != cls.SCHEMA
        ):
            raise ValueError(
                "Missing or incompatible critic gate checkpoint"
            )

        config = GateConfig(**state["config"])
        result = cls(config)

        n = state["last_completed"]
        streak = state["good_streak"]
        first = state["first_actor_update"]

        if (
            type(n) is not int
            or n < 0
            or type(streak) is not int
            or streak < 0
            or (
                first is not None
                and (
                    type(first) is not int
                    or first < config.min_frozen_updates + 1
                    or first > n + 1
                )
            )
        ):
            raise ValueError("Corrupted critic gate progress")

        result.last_completed = n
        result.good_streak = streak
        result.first_actor_update = first
        result.last_ok = state["last_ok"]
        result.last_reason = state["last_reason"]

        return result
