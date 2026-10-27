"""Frame-coherent Phase-1 MARL interaction orchestration.

This runner owns no robot semantics, reward definition, PPO update logic,
or Isaac implementation.

It only orders the already existing boundaries:

    exact current physical frame
    -> actor/critic runtime context
    -> stochastic MAPPO decision
    -> robot-specific action staging
    -> DirectMARLEnv physical transition
    -> next privileged critic frame

The sampled action is never regenerated here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Phase1VerticalSliceStep:
    """One complete vectorized Phase-1 interaction step."""

    active_env_indices: tuple[int, ...]
    contexts: tuple[Any, ...]
    decisions: tuple[Any | None, ...]
    env_actions: Any
    transitions: tuple[Any | None, ...]
    next_critic_graphs: tuple[Any, ...]


class Phase1VerticalSliceRunner:
    """Order one authority-safe, frame-coherent MARL interaction."""

    def __init__(
        self,
        *,
        env,
        runtime_context,
        decision_builder,
        frame_synchronizer,
    ) -> None:
        env_num = getattr(
            env,
            "num_envs",
            None,
        )

        context_num = getattr(
            runtime_context,
            "num_envs",
            None,
        )

        sync_num = getattr(
            frame_synchronizer,
            "num_envs",
            None,
        )

        if (
            isinstance(env_num, bool)
            or not isinstance(env_num, int)
            or env_num <= 0
        ):
            raise ValueError(
                "env.num_envs must be a positive integer"
            )

        if context_num != env_num:
            raise ValueError(
                "runtime_context batch size does not match env"
            )

        if sync_num != env_num:
            raise ValueError(
                "frame_synchronizer batch size does not match env"
            )

        handoffs = getattr(
            env,
            "handoffs",
            None,
        )

        if (
            handoffs is None
            or len(tuple(handoffs))
            != env_num
        ):
            raise TypeError(
                "env must expose one handoff per environment"
            )

        for label, obj, method in (
            (
                "runtime_context",
                runtime_context,
                "current_action_contexts",
            ),
            (
                "runtime_context",
                runtime_context,
                "encode_env_actions",
            ),
            (
                "runtime_context",
                runtime_context,
                "current_critic_graphs",
            ),
            (
                "env",
                env,
                "step_marl_transitions",
            ),
            (
                "frame_synchronizer",
                frame_synchronizer,
                "state_reader",
            ),
        ):
            if not callable(
                getattr(
                    obj,
                    method,
                    None,
                )
            ):
                raise TypeError(
                    f"{label} must expose {method}()"
                )

        build_decision = getattr(
            decision_builder,
            "build_online_rollout_decision",
            None,
        )

        if not callable(
            build_decision
        ):
            raise TypeError(
                "decision_builder must expose "
                "build_online_rollout_decision()"
            )

        self.env = env
        self.runtime_context = (
            runtime_context
        )
        self.decision_builder = (
            decision_builder
        )
        self.frame_synchronizer = (
            frame_synchronizer
        )

    @property
    def num_envs(self) -> int:
        return int(
            self.env.num_envs
        )

    def _active_env_indices(
        self,
    ) -> tuple[int, ...]:
        active = []

        for env_index, handoff in enumerate(
            self.env.handoffs
        ):
            marl_active = getattr(
                handoff,
                "marl_active",
                None,
            )

            if type(marl_active) is not bool:
                raise TypeError(
                    "handoff.marl_active must be bool"
                )

            if marl_active:
                active.append(
                    env_index
                )

        return tuple(
            active
        )

    @staticmethod
    def _build_decision(
        builder,
        context,
    ):
        required = (
            "actor_graph",
            "critic_graph",
            "routing_snapshot",
            "hard_mask",
        )

        missing = [
            name
            for name in required
            if not hasattr(
                context,
                name,
            )
        ]

        if missing:
            raise TypeError(
                "runtime context is missing: "
                + ", ".join(
                    missing
                )
            )

        return (
            builder
            .build_online_rollout_decision(
                actor_graph=(
                    context.actor_graph
                ),
                critic_graph=(
                    context.critic_graph
                ),
                routing_snapshot=(
                    context.routing_snapshot
                ),
                hard_mask=(
                    context.hard_mask
                ),
            )
        )

    def step(
        self,
    ) -> Phase1VerticalSliceStep | None:
        """Execute one complete physical MARL interaction.

        No physical frame, policy decision or action staging occurs before
        verified MARL authority.
        """

        active = (
            self._active_env_indices()
        )

        if not active:
            return None

        # Arm the exact physical pre-state BEFORE runtime observation
        # preprocessing. File-backed graph/context sources can now consume
        # frame_synchronizer.cached_raw_snapshot(env_index) without racing
        # a newer Isaac publication.
        for env_index in active:
            self.frame_synchronizer.state_reader(
                env_index
            )

        contexts = tuple(
            self.runtime_context
            .current_action_contexts()
        )

        if len(contexts) != self.num_envs:
            raise ValueError(
                "runtime context batch size changed during interaction"
            )

        decisions: list[
            Any | None
        ] = [
            None
            for _ in range(
                self.num_envs
            )
        ]

        for env_index in active:
            decisions[
                env_index
            ] = self._build_decision(
                self.decision_builder,
                contexts[
                    env_index
                ],
            )

        # For active environments this is the single semantic path:
        # sampled MARL bundle -> existing ActionSet materialization/decoder
        # -> existing robot-specific dispatcher.
        #
        # Inactive environments are neutralized by the runtime context
        # provider and never invoke its MARL action encoder.
        env_actions = (
            self.runtime_context
            .encode_env_actions(
                decisions
            )
        )

        transitions = tuple(
            self.env
            .step_marl_transitions()
        )

        if len(transitions) != self.num_envs:
            raise ValueError(
                "DirectMARLEnv transition batch size mismatch"
            )

        for env_index in active:
            if transitions[
                env_index
            ] is None:
                raise RuntimeError(
                    "MARL-active environment produced no physical transition: "
                    f"env {env_index}"
                )

        # The provider deliberately captures the post-step privileged graph
        # once and reuses it as the next actor state.
        next_critic_graphs = tuple(
            self.runtime_context
            .current_critic_graphs()
        )

        if (
            len(
                next_critic_graphs
            )
            != self.num_envs
        ):
            raise ValueError(
                "next critic graph batch size mismatch"
            )

        return Phase1VerticalSliceStep(
            active_env_indices=active,
            contexts=contexts,
            decisions=tuple(
                decisions
            ),
            env_actions=env_actions,
            transitions=transitions,
            next_critic_graphs=(
                next_critic_graphs
            ),
        )
