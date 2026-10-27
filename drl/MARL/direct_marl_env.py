from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from drl.MARL.runtime_handoff import (
    DirectMARLHandoff,
    RuntimePhase,
)


class DirectMARLEnv:
    """Thin runtime orchestrator for the MSSR MARL environment.

    This layer owns only episode/control orchestration:

        reset
        -> launch Expert assembly
        -> Expert terminal status
        -> independent topology verification
        -> MARL authority
        -> native actuation

    It deliberately does not contain PPO, GAE, BC H1/H2 logic,
    reward shaping, or robot-specific action semantics.
    """

    def __init__(
        self,
        *,
        handoffs: Sequence[
            DirectMARLHandoff
        ],
        expert_launcher: Callable[
            [int, str],
            bool | None,
        ],
        expert_status_reader: Callable[
            [int],
            bool | None,
        ],
        topology_observer: Callable[
            [int],
            str | None,
        ],
        actuation_bridge: Any,
        target_morphology: str = "rc_car8",
    ) -> None:
        self._handoffs = tuple(
            handoffs
        )

        if not self._handoffs:
            raise ValueError(
                "DirectMARLEnv requires at least one environment"
            )

        if any(
            not isinstance(
                handoff,
                DirectMARLHandoff,
            )
            for handoff
            in self._handoffs
        ):
            raise TypeError(
                "handoffs must contain DirectMARLHandoff instances"
            )

        for (
            label,
            callback,
        ) in (
            (
                "expert_launcher",
                expert_launcher,
            ),
            (
                "expert_status_reader",
                expert_status_reader,
            ),
            (
                "topology_observer",
                topology_observer,
            ),
        ):
            if not callable(
                callback
            ):
                raise TypeError(
                    f"{label} must be callable"
                )

        step = getattr(
            actuation_bridge,
            "step",
            None,
        )

        if not callable(
            step
        ):
            raise TypeError(
                "actuation_bridge must expose step(env_index)"
            )

        if (
            type(
                target_morphology
            )
            is not str
            or not target_morphology
        ):
            raise ValueError(
                "target_morphology must be a non-empty string"
            )

        self._expert_launcher = (
            expert_launcher
        )

        self._expert_status_reader = (
            expert_status_reader
        )

        self._topology_observer = (
            topology_observer
        )

        self._actuation_bridge = (
            actuation_bridge
        )

        self.target_morphology = (
            target_morphology
        )

    @property
    def num_envs(
        self,
    ) -> int:
        return len(
            self._handoffs
        )

    @property
    def handoffs(
        self,
    ) -> tuple[
        DirectMARLHandoff,
        ...,
    ]:
        return self._handoffs

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

    def _normalize_env_ids(
        self,
        env_ids=None,
    ) -> tuple[int, ...]:
        if env_ids is None:
            return tuple(
                range(
                    self.num_envs
                )
            )

        if isinstance(
            env_ids,
            int,
        ):
            env_ids = (
                env_ids,
            )

        result = tuple(
            self._validate_env_index(
                int(
                    env_index
                )
            )
            for env_index
            in env_ids
        )

        if len(
            set(
                result
            )
        ) != len(
            result
        ):
            raise ValueError(
                "env_ids must not contain duplicates"
            )

        return result

    def reset(
        self,
        env_ids=None,
    ) -> tuple[
        RuntimePhase,
        ...,
    ]:
        """Reset selected environments and start Expert assembly.

        Critical ordering:

            revoke previous authority
            -> launch Expert
            -> only if launch succeeded, claim EXPERT_ASSEMBLY
        """
        env_ids = (
            self._normalize_env_ids(
                env_ids
            )
        )

        phases = []

        for env_index in env_ids:
            handoff = (
                self._handoffs[
                    env_index
                ]
            )

            # Revoke MARL/Expert authority from the previous episode
            # before starting a new external process.
            handoff.reset()

            launched = (
                self._expert_launcher(
                    env_index,
                    self.target_morphology,
                )
            )

            if launched is False:
                raise RuntimeError(
                    "Expert assembly launch failed "
                    f"for env {env_index}"
                )

            # Authority is granted only after successful launch.
            handoff.begin_expert(
                self.target_morphology
            )

            phases.append(
                handoff.phase
            )

        return tuple(
            phases
        )

    def observe_expert_terminal(
        self,
        env_index: int,
        *,
        success: bool,
    ) -> RuntimePhase:
        env_index = (
            self._validate_env_index(
                env_index
            )
        )

        handoff = (
            self._handoffs[
                env_index
            ]
        )

        handoff.observe_expert_terminal(
            success=success
        )

        return handoff.phase

    def observe_topology(
        self,
        env_index: int,
        topology: str | None,
    ) -> RuntimePhase:
        env_index = (
            self._validate_env_index(
                env_index
            )
        )

        handoff = (
            self._handoffs[
                env_index
            ]
        )

        handoff.observe_topology(
            topology
        )

        return handoff.phase

    def advance_handoff(
        self,
        env_index: int,
    ) -> RuntimePhase:
        """Advance at most one authority transition.

        Keeping one transition per call makes the safety boundary explicit:

            ASSEMBLING
              --Expert terminal-->
            VERIFYING_TOPOLOGY
              --independent observation-->
            MARL_CONTROL

        Expert success and topology verification are therefore never
        collapsed into the same authority event.
        """
        env_index = (
            self._validate_env_index(
                env_index
            )
        )

        handoff = (
            self._handoffs[
                env_index
            ]
        )

        phase = handoff.phase

        if phase is RuntimePhase.ASSEMBLING:
            status = (
                self._expert_status_reader(
                    env_index
                )
            )

            if status is None:
                return handoff.phase

            if type(
                status
            ) is not bool:
                raise TypeError(
                    "Expert terminal status must be bool or None"
                )

            handoff.observe_expert_terminal(
                success=status
            )

            return handoff.phase

        if (
            phase
            is RuntimePhase.VERIFYING_TOPOLOGY
        ):
            topology = (
                self._topology_observer(
                    env_index
                )
            )

            if (
                topology is not None
                and type(
                    topology
                )
                is not str
            ):
                raise TypeError(
                    "topology observer must return str or None"
                )

            handoff.observe_topology(
                topology
            )

            return handoff.phase

        return handoff.phase

    def advance_all_handoffs(
        self,
    ) -> tuple[
        RuntimePhase,
        ...,
    ]:
        return tuple(
            self.advance_handoff(
                env_index
            )
            for env_index
            in range(
                self.num_envs
            )
        )

    def step_marl_actuation(
        self,
        env_index: int,
    ):
        """Apply native MARL commands only under MARL authority."""
        env_index = (
            self._validate_env_index(
                env_index
            )
        )

        handoff = (
            self._handoffs[
                env_index
            ]
        )

        if not handoff.marl_active:
            return None

        return (
            self._actuation_bridge
            .step(
                env_index
            )
        )

    def step_marl_actuation_all(
        self,
    ) -> tuple[Any, ...]:
        return tuple(
            self.step_marl_actuation(
                env_index
            )
            for env_index
            in range(
                self.num_envs
            )
        )
