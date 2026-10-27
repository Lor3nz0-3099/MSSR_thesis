"""Exact-frame BC-v4 observation source for the Phase-1 Isaac runtime.

This module does not read Isaac files directly.

The ``FileBackedIsaacControlStep`` is the sole owner of physical-frame
synchronization.  This source consumes only its cached atomic
``state_graph.json`` frame and transforms that exact frame through the
already-certified BC-v4 runtime observation bridge.

This keeps:

    physical s_t
        ==
    BC-v4 actor s_t

even if Isaac publishes newer frames while policy inference is running.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
)
from dataclasses import dataclass
import math
from typing import Any

from drl.MARL.bc_v4_runtime_observation import (
    BCV4RuntimeObservation,
)


@dataclass(frozen=True)
class _PreparedEntry:
    stamp: float
    observation: BCV4RuntimeObservation


class SynchronizedBCV4ObservationSource:
    """Prepare BC-v4 actor input from exact synchronizer-owned frames."""

    def __init__(
        self,
        *,
        frame_synchronizer,
        observation_bridge,
        task_context_provider: Callable[
            [int],
            Any,
        ],
        observation_provider: Callable[
            [int],
            Any,
        ]
        | None = None,
        assigned_goal_provider: Callable[
            [int],
            Any,
        ]
        | None = None,
    ) -> None:
        num_envs = getattr(
            frame_synchronizer,
            "num_envs",
            None,
        )

        if (
            isinstance(
                num_envs,
                bool,
            )
            or not isinstance(
                num_envs,
                int,
            )
            or num_envs <= 0
        ):
            raise ValueError(
                "frame_synchronizer.num_envs "
                "must be a positive integer"
            )

        if not callable(
            getattr(
                frame_synchronizer,
                "cached_raw_snapshot",
                None,
            )
        ):
            raise TypeError(
                "frame_synchronizer must expose "
                "cached_raw_snapshot()"
            )

        if not callable(
            getattr(
                observation_bridge,
                "prepare_snapshot",
                None,
            )
        ):
            raise TypeError(
                "observation_bridge must expose "
                "prepare_snapshot()"
            )

        if not callable(
            task_context_provider
        ):
            raise TypeError(
                "task_context_provider must be callable"
            )

        if (
            observation_provider
            is not None
            and not callable(
                observation_provider
            )
        ):
            raise TypeError(
                "observation_provider must be callable or None"
            )

        if (
            assigned_goal_provider
            is not None
            and not callable(
                assigned_goal_provider
            )
        ):
            raise TypeError(
                "assigned_goal_provider must be callable or None"
            )

        self.frame_synchronizer = (
            frame_synchronizer
        )

        self.observation_bridge = (
            observation_bridge
        )

        self.task_context_provider = (
            task_context_provider
        )

        self.observation_provider = (
            observation_provider
        )

        self.assigned_goal_provider = (
            assigned_goal_provider
        )

        self._prepared: list[
            _PreparedEntry | None
        ] = [
            None
            for _ in range(
                num_envs
            )
        ]

    @property
    def num_envs(
        self,
    ) -> int:
        return int(
            self.frame_synchronizer
            .num_envs
        )

    def _validate_env_index(
        self,
        env_index: int,
    ) -> int:
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
            or env_index >= self.num_envs
        ):
            raise IndexError(
                f"invalid env_index: {env_index}"
            )

        return env_index

    @staticmethod
    def _raw_stamp(
        raw_snapshot,
    ) -> float:
        if not isinstance(
            raw_snapshot,
            dict,
        ):
            raise TypeError(
                "cached Isaac snapshot must be a dict"
            )

        stamp = raw_snapshot.get(
            "stamp"
        )

        if (
            isinstance(
                stamp,
                bool,
            )
            or not isinstance(
                stamp,
                (int, float),
            )
            or not math.isfinite(
                float(stamp)
            )
        ):
            raise ValueError(
                "cached Isaac snapshot must contain "
                "a finite stamp"
            )

        return float(
            stamp
        )

    def prepared_stamp(
        self,
        env_index: int,
    ) -> float | None:
        env_index = (
            self._validate_env_index(
                env_index
            )
        )

        entry = self._prepared[
            env_index
        ]

        return (
            None
            if entry is None
            else entry.stamp
        )

    def prepare_env(
        self,
        env_index: int,
    ) -> BCV4RuntimeObservation:
        """Return BC-v4 tensors for the exact current cached Isaac frame."""

        env_index = (
            self._validate_env_index(
                env_index
            )
        )

        raw_snapshot = (
            self.frame_synchronizer
            .cached_raw_snapshot(
                env_index
            )
        )

        stamp = self._raw_stamp(
            raw_snapshot
        )

        existing = self._prepared[
            env_index
        ]

        if (
            existing is not None
            and existing.stamp
            == stamp
        ):
            return (
                existing.observation
            )

        task_context = (
            self.task_context_provider(
                env_index
            )
        )

        kwargs = {
            "task_context":
                task_context,
        }

        if (
            self.observation_provider
            is not None
        ):
            kwargs[
                "observation"
            ] = (
                self.observation_provider(
                    env_index
                )
            )

        if (
            self.assigned_goal_provider
            is not None
        ):
            kwargs[
                "assigned_goal"
            ] = (
                self.assigned_goal_provider(
                    env_index
                )
            )

        prepared = (
            self.observation_bridge
            .prepare_snapshot(
                raw_snapshot,
                **kwargs,
            )
        )

        physical_stamp = getattr(
            prepared.physical_state,
            "stamp",
            None,
        )

        if (
            isinstance(
                physical_stamp,
                bool,
            )
            or not isinstance(
                physical_stamp,
                (int, float),
            )
            or not math.isfinite(
                float(
                    physical_stamp
                )
            )
        ):
            raise ValueError(
                "BC-v4 physical state has no finite stamp"
            )

        if (
            float(
                physical_stamp
            )
            != stamp
        ):
            raise ValueError(
                "BC-v4 preprocessing changed physical stamp: "
                f"{physical_stamp} != {stamp}"
            )

        entry = _PreparedEntry(
            stamp=stamp,
            observation=prepared,
        )

        self._prepared[
            env_index
        ] = entry

        return (
            entry.observation
        )

    def prepare_active(
        self,
        env_indices,
    ) -> tuple[
        BCV4RuntimeObservation,
        ...,
    ]:
        return tuple(
            self.prepare_env(
                int(env_index)
            )
            for env_index
            in env_indices
        )

    def reset(
        self,
        env_indices=None,
    ) -> None:
        """Drop preprocessing cache without changing physical synchronization."""

        if env_indices is None:
            indexes = range(
                self.num_envs
            )
        else:
            indexes = tuple(
                self._validate_env_index(
                    int(env_index)
                )
                for env_index
                in env_indices
            )

        for env_index in indexes:
            self._prepared[
                env_index
            ] = None
