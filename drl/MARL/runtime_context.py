"""Runtime context boundary between MSSR physics and structured MAPPO.

This module deliberately contains no Isaac- or SMORES-specific logic.
Robot-specific graph acquisition, actor visibility, routing, masking and
action encoding are injected dependencies.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from drl.MARL.mappo import OnlineRolloutContext
from drl.MARL.runtime_handoff import DirectMARLHandoff


@dataclass(frozen=True)
class _CapturedRuntimeContext:
    actor_graph: Any
    critic_graph: Any
    routing_snapshot: Any
    hard_mask: Any


class DirectMARLRuntimeContextProvider:
    """Provide synchronized actor/critic views to MSSRMAPPO.

    A physical graph is captured once for each runtime state. The
    privileged critic snapshot is preserved before actor visibility
    filtering. The next privileged snapshot captured after env.step()
    is reused by the following act(), preventing a second physical-state
    acquisition for the same state.
    """

    def __init__(
        self,
        *,
        graph_source: Callable[[], Sequence[Any]],
        actor_visibility: Callable[[Any], Any],
        routing_builder: Callable[[Any], Any],
        hard_mask_builder: Callable[[Any, Any], Any],
        handoffs: Sequence[DirectMARLHandoff],
        action_encoder: Callable[
            [Any, int, OnlineRolloutContext],
            dict[str, torch.Tensor],
        ],
        neutral_action: Callable[
            [int, OnlineRolloutContext],
            dict[str, torch.Tensor],
        ],
    ) -> None:
        self._graph_source = graph_source
        self._actor_visibility = actor_visibility
        self._routing_builder = routing_builder
        self._hard_mask_builder = hard_mask_builder
        self._handoffs = tuple(handoffs)
        self._action_encoder = action_encoder
        self._neutral_action = neutral_action

        if not self._handoffs:
            raise ValueError(
                "handoffs must contain at least one environment"
            )

        if not all(
            isinstance(
                handoff,
                DirectMARLHandoff,
            )
            for handoff in self._handoffs
        ):
            raise TypeError(
                "handoffs must contain DirectMARLHandoff objects"
            )

        self.reset()

    @property
    def num_envs(self) -> int:
        return len(
            self._handoffs
        )

    def reset(self) -> None:
        """Drop cached physical-state views after an environment reset."""
        self._captured: tuple[
            _CapturedRuntimeContext,
            ...
        ] | None = None

        # False means the cached graph describes the current physical
        # state and may be reused by act().
        self._awaiting_post_step = False

    def _capture(self) -> tuple[
        _CapturedRuntimeContext,
        ...
    ]:
        source_graphs = list(
            self._graph_source()
        )

        if len(source_graphs) != self.num_envs:
            raise ValueError(
                "graph source batch size does not match handoff batch: "
                f"{len(source_graphs)} != {self.num_envs}"
            )

        captured = []

        for graph in source_graphs:
            # Freeze the privileged current-state snapshot first.
            critic_graph = deepcopy(
                graph
            )

            # Actor visibility receives a separate object so even a
            # destructive filtering implementation cannot modify the
            # privileged critic snapshot.
            actor_input = deepcopy(
                critic_graph
            )

            actor_graph = self._actor_visibility(
                actor_input
            )

            routing_snapshot = self._routing_builder(
                actor_graph
            )

            hard_mask = self._hard_mask_builder(
                actor_graph,
                routing_snapshot,
            )

            captured.append(
                _CapturedRuntimeContext(
                    actor_graph=actor_graph,
                    critic_graph=critic_graph,
                    routing_snapshot=routing_snapshot,
                    hard_mask=hard_mask,
                )
            )

        self._captured = tuple(
            captured
        )

        return self._captured

    def _ensure_current_capture(
        self,
    ) -> tuple[
        _CapturedRuntimeContext,
        ...
    ]:
        if self._captured is None:
            return self._capture()

        return self._captured

    @staticmethod
    def _public_context(
        captured: _CapturedRuntimeContext,
    ) -> OnlineRolloutContext:
        return OnlineRolloutContext(
            actor_graph=captured.actor_graph,
            critic_graph=captured.critic_graph,
            routing_snapshot=captured.routing_snapshot,
            hard_mask=captured.hard_mask,
        )

    def current_action_contexts(
        self,
    ) -> list[OnlineRolloutContext]:
        captured = self._ensure_current_capture()

        return [
            self._public_context(
                item
            )
            for item in captured
        ]

    @staticmethod
    def _validate_action_row(
        *,
        actions,
        env_index: int,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(
            actions,
            dict,
        ) or not actions:
            raise TypeError(
                "encoded runtime action must be a non-empty dict"
            )

        validated = {}

        for owner, tensor in actions.items():
            if not isinstance(
                owner,
                str,
            ) or not owner:
                raise TypeError(
                    "runtime action owner IDs must be non-empty strings"
                )

            if not torch.is_tensor(
                tensor
            ):
                raise TypeError(
                    f"runtime action for {owner} in env {env_index} "
                    "must be a torch.Tensor"
                )

            if tensor.ndim != 1:
                raise ValueError(
                    f"runtime action for {owner} in env {env_index} "
                    "must be a 1-D per-environment row"
                )

            validated[
                owner
            ] = tensor

        return validated

    def encode_env_actions(
        self,
        decisions,
    ) -> dict[str, torch.Tensor]:
        """Encode one structured decision per environment.

        The MARL encoder is never invoked for an environment that does
        not currently hold MARL authority.
        """
        decisions = list(
            decisions
        )

        if len(decisions) != self.num_envs:
            raise ValueError(
                "decision batch size does not match runtime batch: "
                f"{len(decisions)} != {self.num_envs}"
            )

        captured = self._ensure_current_capture()

        rows = []

        for env_index, (
            decision,
            snapshot,
            handoff,
        ) in enumerate(
            zip(
                decisions,
                captured,
                self._handoffs,
            )
        ):
            context = self._public_context(
                snapshot
            )

            if handoff.marl_active:
                action_row = self._action_encoder(
                    decision,
                    env_index,
                    context,
                )
            else:
                action_row = self._neutral_action(
                    env_index,
                    context,
                )

            rows.append(
                self._validate_action_row(
                    actions=action_row,
                    env_index=env_index,
                )
            )

        expected_owners = tuple(
            rows[0].keys()
        )

        expected_owner_set = set(
            expected_owners
        )

        for env_index, row in enumerate(
            rows[1:],
            start=1,
        ):
            if set(
                row.keys()
            ) != expected_owner_set:
                raise ValueError(
                    "runtime action owners must be identical across "
                    f"vectorized environments; mismatch at env {env_index}"
                )

        batched = {}

        for owner in expected_owners:
            shapes = {
                tuple(
                    row[owner].shape
                )
                for row in rows
            }

            if len(shapes) != 1:
                raise ValueError(
                    f"runtime action shape for {owner} differs "
                    "across vectorized environments"
                )

            batched[
                owner
            ] = torch.stack(
                [
                    row[owner]
                    for row in rows
                ],
                dim=0,
            )

        # SequentialTrainer will now call env.step(). The next request
        # for critic graphs therefore belongs to the post-step state.
        self._awaiting_post_step = True

        return batched

    def current_critic_graphs(
        self,
    ) -> list[Any]:
        """Return current privileged graphs.

        After encode_env_actions() this is the first post-step access, so
        capture the new physical state exactly once. Repeated reads and the
        next current_action_contexts() reuse that same snapshot.
        """
        if (
            self._captured is None
            or self._awaiting_post_step
        ):
            self._capture()
            self._awaiting_post_step = False

        assert self._captured is not None

        return [
            item.critic_graph
            for item in self._captured
        ]
