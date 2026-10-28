"""File-backed control-step synchronization for the real Isaac runtime.

The Isaac simulator runs in a separate process and advances physics
continuously.  A MARL "physics step" therefore means waiting for a
strictly newer atomically-published physical state frame rather than
calling SimulationManager.step() from the learning process.

The producer already publishes ``state_graph.json`` atomically with a
simulation-time ``stamp`` and all module poses.  This adapter turns that
stream into the callbacks expected by ``DirectMARLEnv``:

    state_reader(env_index)
    physics_step()

The exact frame which satisfies the freshness barrier is cached so the
subsequent post-step state read cannot race with a later Isaac publish.
"""

from __future__ import annotations

from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any


_STATE_GRAPH_SCHEMA = "mssr.state_graph.v2"
_MODULE_STATE_SCHEMA = "mssr.module_states.v2"


@dataclass(frozen=True)
class _PhysicalSnapshot:
    stamp: float
    module_xy_m: dict[
        str,
        tuple[float, float],
    ]
    raw_payload: dict[str, Any]


@dataclass(frozen=True)
class IsaacControlStepResult:
    """Freshness stamps bounding one external MARL control step."""

    previous_stamp_by_env: Mapping[int, float]
    next_stamp_by_env: Mapping[int, float]


class FileBackedIsaacControlStep:
    """Synchronize DirectMARLEnv transitions with Isaac state publications."""

    def __init__(
        self,
        *,
        runtime_dirs: Sequence[str | Path],
        state_filename: str = "state_graph.json",
        expected_module_count: int | None = 8,
        expected_module_ids: Sequence[str] | None = None,
        timeout_s: float = 2.0,
        poll_interval_s: float = 0.005,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        application_source: Callable[[int], Mapping[str, Any]] | None = None,
    ) -> None:
        if not runtime_dirs:
            raise ValueError(
                "at least one Isaac runtime directory is required"
            )

        self.runtime_dirs = tuple(
            Path(path).expanduser().resolve()
            for path in runtime_dirs
        )

        if (
            not isinstance(state_filename, str)
            or not state_filename
        ):
            raise ValueError(
                "state_filename must be a non-empty string"
            )

        self.state_filename = state_filename

        if expected_module_count is not None:
            if (
                isinstance(expected_module_count, bool)
                or not isinstance(expected_module_count, int)
                or expected_module_count <= 0
            ):
                raise ValueError(
                    "expected_module_count must be a positive integer or None"
                )

        self.expected_module_count = (
            expected_module_count
        )

        if expected_module_ids is not None:
            normalized_ids = tuple(
                expected_module_ids
            )

            if (
                not normalized_ids
                or any(
                    not isinstance(module_id, str)
                    or not module_id
                    for module_id in normalized_ids
                )
                or len(set(normalized_ids))
                != len(normalized_ids)
            ):
                raise ValueError(
                    "expected_module_ids must contain unique non-empty IDs"
                )

            if (
                expected_module_count is not None
                and len(normalized_ids)
                != expected_module_count
            ):
                raise ValueError(
                    "expected_module_ids length must match "
                    "expected_module_count"
                )

            self.expected_module_ids = (
                frozenset(normalized_ids)
            )

        else:
            self.expected_module_ids = None

        for label, value in (
            ("timeout_s", timeout_s),
            ("poll_interval_s", poll_interval_s),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(
                    f"{label} must be finite and positive"
                )

        if not callable(monotonic):
            raise TypeError(
                "monotonic must be callable"
            )

        if not callable(sleep):
            raise TypeError(
                "sleep must be callable"
            )

        self.timeout_s = float(timeout_s)
        self.poll_interval_s = float(
            poll_interval_s
        )

        self._monotonic = monotonic
        self._sleep = sleep
        if application_source is not None and not callable(application_source):
            raise TypeError('application_source must be callable')
        self._application_source = application_source
        self._expected_actions: dict[int, str] = {}


        # Per-environment state machine:
        #
        # IDLE
        #   state_reader() -> ARMED with exact pre-state
        #
        # ARMED
        #   physics_step() waits for stamp > baseline -> READY
        #
        # READY
        #   state_reader() returns cached exact post-state -> IDLE
        self._phase = [
            "IDLE"
            for _ in self.runtime_dirs
        ]

        self._pre_snapshot: list[
            _PhysicalSnapshot | None
        ] = [
            None
            for _ in self.runtime_dirs
        ]

        self._post_snapshot: list[
            _PhysicalSnapshot | None
        ] = [
            None
            for _ in self.runtime_dirs
        ]

        # The accepted s_(t+1) remains available after DirectMARLEnv
        # consumes its XY view. It is both:
        #   - the exact post-step critic frame
        #   - the exact pre-state for the following Markov transition.
        self._last_completed_snapshot: list[
            _PhysicalSnapshot | None
        ] = [
            None
            for _ in self.runtime_dirs
        ]

    @property
    def num_envs(self) -> int:
        return len(self.runtime_dirs)

    def expect_action(self, env_index: int, command_id: str) -> None:
        """Bind this transition to the existing native BC command identifier."""
        env_index = self._validate_env_index(env_index)
        if self._application_source is None:
            raise RuntimeError('expected action requires application_source')
        if self._phase[env_index] != 'ARMED':
            raise RuntimeError('expected action requires an armed pre-state')
        if not isinstance(command_id, str) or not command_id:
            raise ValueError('command_id must be nonempty')
        if env_index in self._expected_actions:
            raise RuntimeError('expected action already registered')
        self._expected_actions[env_index] = command_id

    def _causal_frame(self, env_index: int, snapshot: _PhysicalSnapshot, previous: float) -> bool:
        if self._application_source is None:
            return snapshot.stamp > previous
        status = self._application_source(env_index)
        if status.get('failure'):
            raise RuntimeError(f"Native action application failed: {status['failure']}")
        completed = status.get('last_completed')
        if not isinstance(completed, Mapping) or completed.get('command_id') != self._expected_actions[env_index]:
            return False
        applied = self._finite_number(completed.get('applied_stamp'), label='action applied stamp')
        post = self._finite_number(completed.get('completed_stamp'), label='action completed stamp')
        return applied >= previous and post > applied and snapshot.stamp == post

    @property
    def armed_env_indices(
        self,
    ) -> tuple[int, ...]:
        return tuple(
            env_index
            for env_index, phase
            in enumerate(self._phase)
            if phase == "ARMED"
        )

    def _validate_env_index(
        self,
        env_index: int,
    ) -> int:
        if (
            isinstance(env_index, bool)
            or not isinstance(env_index, int)
            or env_index < 0
            or env_index >= self.num_envs
        ):
            raise IndexError(
                f"invalid env_index: {env_index}"
            )

        return env_index

    def _path(
        self,
        env_index: int,
    ) -> Path:
        return (
            self.runtime_dirs[env_index]
            / self.state_filename
        )

    @staticmethod
    def _finite_number(
        value: Any,
        *,
        label: str,
    ) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
        ):
            raise ValueError(
                f"{label} must be numeric"
            )

        result = float(value)

        if not math.isfinite(result):
            raise ValueError(
                f"{label} must be finite"
            )

        return result

    def _decode_snapshot(
        self,
        payload: Any,
        *,
        path: Path,
    ) -> _PhysicalSnapshot:
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"Isaac state graph {path} must contain an object"
            )

        if (
            payload.get("schema_version")
            != _STATE_GRAPH_SCHEMA
        ):
            raise ValueError(
                "unsupported Isaac state graph schema: "
                f"{payload.get('schema_version')!r}"
            )

        stamp = self._finite_number(
            payload.get("stamp"),
            label="Isaac state finite stamp",
        )

        state = payload.get("state")

        if not isinstance(state, Mapping):
            raise ValueError(
                "Isaac state graph is missing state payload"
            )

        if (
            state.get("schema_version")
            != _MODULE_STATE_SCHEMA
        ):
            raise ValueError(
                "unsupported Isaac module-state schema: "
                f"{state.get('schema_version')!r}"
            )

        state_stamp = self._finite_number(
            state.get("stamp"),
            label="Isaac module-state finite stamp",
        )

        if state_stamp != stamp:
            raise ValueError(
                "state_graph stamp and module-state stamp differ"
            )

        modules = state.get("modules")

        if not isinstance(modules, list):
            raise ValueError(
                "Isaac module state must contain a modules list"
            )

        if (
            self.expected_module_count
            is not None
            and len(modules)
            != self.expected_module_count
        ):
            raise ValueError(
                "Isaac physical state expected "
                f"{self.expected_module_count} modules, "
                f"got {len(modules)}"
            )

        module_xy_m: dict[
            str,
            tuple[float, float],
        ] = {}

        for module in modules:
            if not isinstance(module, Mapping):
                raise ValueError(
                    "Isaac module record must be an object"
                )

            module_id = module.get(
                "module_id"
            )

            if (
                not isinstance(module_id, str)
                or not module_id
            ):
                raise ValueError(
                    "Isaac module record has invalid module_id"
                )

            if module_id in module_xy_m:
                raise ValueError(
                    f"duplicate Isaac module_id: {module_id}"
                )

            position = module.get(
                "position"
            )

            if (
                not isinstance(position, (list, tuple))
                or len(position) < 2
            ):
                raise ValueError(
                    f"Isaac module {module_id} has invalid position"
                )

            x_m = self._finite_number(
                position[0],
                label=f"{module_id} x position",
            )

            y_m = self._finite_number(
                position[1],
                label=f"{module_id} y position",
            )

            module_xy_m[module_id] = (
                x_m,
                y_m,
            )

        if (
            self.expected_module_ids
            is not None
            and set(module_xy_m)
            != self.expected_module_ids
        ):
            missing = sorted(
                self.expected_module_ids
                - set(module_xy_m)
            )

            unexpected = sorted(
                set(module_xy_m)
                - self.expected_module_ids
            )

            raise ValueError(
                "Isaac module IDs do not match expected physical robot: "
                f"missing={missing}, unexpected={unexpected}"
            )

        return _PhysicalSnapshot(
            stamp=stamp,
            module_xy_m=module_xy_m,
            raw_payload=deepcopy(
                dict(payload)
            ),
        )

    def _read_snapshot(
        self,
        env_index: int,
        *,
        missing_ok: bool,
    ) -> _PhysicalSnapshot | None:
        path = self._path(
            env_index
        )

        try:
            raw = path.read_text(
                encoding="utf-8"
            )

        except FileNotFoundError:
            if missing_ok:
                return None

            raise FileNotFoundError(
                "Isaac state publication is missing: "
                f"{path}"
            )

        except OSError as error:
            raise RuntimeError(
                "Cannot read Isaac state publication "
                f"{path}: {error}"
            ) from error

        try:
            payload = json.loads(
                raw
            )

        except json.JSONDecodeError as error:
            # The producer uses atomic tmp+replace writes, therefore a
            # malformed latest file is an actual contract violation.
            raise ValueError(
                f"Invalid Isaac state JSON: {path}"
            ) from error

        return self._decode_snapshot(
            payload,
            path=path,
        )

    @staticmethod
    def _copy_state(
        snapshot: _PhysicalSnapshot,
    ) -> dict[
        str,
        tuple[float, float],
    ]:
        return dict(
            snapshot.module_xy_m
        )

    def baseline_stamp(
        self,
        env_index: int,
    ) -> float | None:
        env_index = self._validate_env_index(
            env_index
        )

        snapshot = self._pre_snapshot[
            env_index
        ]

        return (
            None
            if snapshot is None
            else snapshot.stamp
        )

    def cached_raw_snapshot(
        self,
        env_index: int,
    ) -> dict[str, Any]:
        """Return the exact cached state_graph frame for this cycle.

        This never reads the producer file. It is therefore safe to use
        for actor/critic preprocessing without racing a newer Isaac
        publication.
        """

        env_index = self._validate_env_index(
            env_index
        )

        phase = self._phase[
            env_index
        ]

        if phase == "ARMED":
            snapshot = self._pre_snapshot[
                env_index
            ]

        elif phase == "READY":
            snapshot = self._post_snapshot[
                env_index
            ]

        elif phase == "CARRY":
            snapshot = (
                self._last_completed_snapshot[
                    env_index
                ]
            )

        else:
            snapshot = (
                self._last_completed_snapshot[
                    env_index
                ]
            )

        if snapshot is None:
            raise RuntimeError(
                "No cached Isaac physical frame "
                f"for env {env_index}"
            )

        return deepcopy(
            snapshot.raw_payload
        )

    def state_reader(
        self,
        env_index: int,
    ) -> dict[
        str,
        tuple[float, float],
    ]:
        """Return pre/post state according to the current control cycle."""

        env_index = self._validate_env_index(
            env_index
        )

        phase = self._phase[
            env_index
        ]

        if phase == "READY":
            snapshot = self._post_snapshot[
                env_index
            ]

            if snapshot is None:
                raise RuntimeError(
                    "READY Isaac control step has no cached post-state"
                )

            result = self._copy_state(
                snapshot
            )

            self._last_completed_snapshot[
                env_index
            ] = snapshot

            # Do not return to IDLE here. The exact accepted post-state
            # must become the next transition's pre-state.
            self._phase[
                env_index
            ] = "CARRY"

            self._pre_snapshot[
                env_index
            ] = None

            self._post_snapshot[
                env_index
            ] = None

            return result

        if phase == "CARRY":
            snapshot = (
                self._last_completed_snapshot[
                    env_index
                ]
            )

            if snapshot is None:
                raise RuntimeError(
                    "CARRY Isaac control step has no completed snapshot"
                )

            self._pre_snapshot[
                env_index
            ] = snapshot

            self._phase[
                env_index
            ] = "ARMED"

            return self._copy_state(
                snapshot
            )

        if phase == "ARMED":
            snapshot = self._pre_snapshot[
                env_index
            ]

            if snapshot is None:
                raise RuntimeError(
                    "ARMED Isaac control step has no cached pre-state"
                )

            # Repeated reads before the physical barrier must not move the
            # baseline stamp forward.
            return self._copy_state(
                snapshot
            )

        if phase != "IDLE":
            raise RuntimeError(
                f"Unknown Isaac control-step phase: {phase}"
            )

        snapshot = self._read_snapshot(
            env_index,
            missing_ok=False,
        )

        assert snapshot is not None

        self._pre_snapshot[
            env_index
        ] = snapshot

        self._phase[
            env_index
        ] = "ARMED"

        return self._copy_state(
            snapshot
        )

    def physics_step(
        self,
    ) -> IsaacControlStepResult:
        """Wait until every armed environment publishes a newer frame."""

        armed = self.armed_env_indices

        if not armed:
            raise RuntimeError(
                "Isaac control step has no armed environments"
            )
        if self._application_source is not None and any(index not in self._expected_actions for index in armed):
            raise RuntimeError('Isaac control step has no expected action for an armed environment')

        previous = {
            env_index: (
                self._pre_snapshot[
                    env_index
                ].stamp
            )
            for env_index in armed
            if self._pre_snapshot[
                env_index
            ] is not None
        }

        if len(previous) != len(armed):
            raise RuntimeError(
                "Isaac control step is missing a pre-state baseline"
            )

        deadline = (
            self._monotonic()
            + self.timeout_s
        )

        unresolved = set(
            armed
        )

        accepted: dict[
            int,
            _PhysicalSnapshot,
        ] = {}

        while unresolved:
            for env_index in tuple(
                sorted(unresolved)
            ):
                snapshot = self._read_snapshot(
                    env_index,
                    missing_ok=True,
                )

                if snapshot is None:
                    continue

                if self._causal_frame(env_index, snapshot, previous[env_index]):
                    accepted[
                        env_index
                    ] = snapshot

                    unresolved.remove(
                        env_index
                    )

            if not unresolved:
                break

            now = self._monotonic()

            if now >= deadline:
                details = ", ".join(
                    (
                        f"env {env_index} "
                        f"baseline {previous[env_index]}"
                    )
                    for env_index
                    in sorted(unresolved)
                )

                raise TimeoutError(
                    "Timed out waiting for a strictly newer "
                    f"Isaac publication: {details}"
                )

            self._sleep(
                min(
                    self.poll_interval_s,
                    max(
                        0.0,
                        deadline - now,
                    ),
                )
            )

        for env_index in armed:
            snapshot = accepted[
                env_index
            ]

            self._post_snapshot[
                env_index
            ] = snapshot

            self._phase[
                env_index
            ] = "READY"
            self._expected_actions.pop(env_index, None)

        return IsaacControlStepResult(
            previous_stamp_by_env=dict(
                previous
            ),
            next_stamp_by_env={
                env_index: accepted[
                    env_index
                ].stamp
                for env_index in armed
            },
        )

    def reset(
        self,
        env_indices: Sequence[int] | None = None,
    ) -> None:
        """Discard pending/cached frames without touching Isaac itself."""

        if env_indices is None:
            indexes = tuple(
                range(self.num_envs)
            )

        else:
            indexes = tuple(
                self._validate_env_index(
                    env_index
                )
                for env_index in env_indices
            )

        for env_index in indexes:
            self._expected_actions.pop(env_index, None)
            self._phase[
                env_index
            ] = "IDLE"

            self._pre_snapshot[
                env_index
            ] = None

            self._post_snapshot[
                env_index
            ] = None

            self._last_completed_snapshot[
                env_index
            ] = None
