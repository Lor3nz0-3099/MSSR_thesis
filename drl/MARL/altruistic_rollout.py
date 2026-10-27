from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from drl.MARL.altruism import (
    build_radius_neighbourhoods,
    compute_mean_field_rewards,
)


@dataclass(frozen=True)
class AltruisticRolloutBatchPayload:
    """Runtime-owned causal data for one vectorized transition.

    Team reward is intentionally NOT part of this payload.
    APPO local rewards are supplied separately from the existing
    team-MAPPO reward channel.
    """

    module_ids_by_row: Sequence[
        Sequence[str]
    ]

    owner_to_module_id: Sequence[
        Mapping[
            str,
            str,
        ]
    ]

    module_positions_by_id: Sequence[
        Mapping[
            str,
            torch.Tensor,
        ]
    ]

    owner_rewards: Mapping[
        str,
        torch.Tensor,
    ]

    next_module_ids_by_row: (
        Sequence[
            Sequence[str]
        ]
        | None
    ) = None

    next_owner_to_module_id: (
        Sequence[
            Mapping[
                str,
                str,
            ]
        ]
        | None
    ) = None


@dataclass(frozen=True)
class AltruisticRolloutSample:
    """One environment at one rollout timestep."""

    owner_row_index: dict[
        str,
        int,
    ]

    owner_rewards: dict[
        str,
        torch.Tensor,
    ]

    mean_field_rewards: dict[
        str,
        torch.Tensor,
    ]

    neighbourhoods: dict[
        str,
        tuple[str, ...],
    ]

    next_owner_row_index: (
        dict[
            str,
            int,
        ]
        | None
    ) = None


def _agent_ids(
    possible_agents: Sequence[str],
) -> tuple[str, ...]:
    agents = tuple(
        possible_agents
    )

    if (
        not agents
        or any(
            type(agent) is not str
            or not agent
            for agent in agents
        )
        or len(
            set(
                agents
            )
        )
        != len(
            agents
        )
    ):
        raise ValueError(
            "possible_agents must be non-empty unique strings"
        )

    return agents


def _module_rows(
    module_ids_by_row: Sequence[str],
) -> tuple[str, ...]:
    rows = tuple(
        module_ids_by_row
    )

    if (
        not rows
        or any(
            type(module_id) is not str
            or not module_id
            for module_id in rows
        )
        or len(
            set(
                rows
            )
        )
        != len(
            rows
        )
    ):
        raise ValueError(
            "module_ids_by_row must be non-empty unique strings"
        )

    return rows


def _scalar_reward(
    value: Any,
    *,
    owner: str,
) -> torch.Tensor:
    if not torch.is_tensor(
        value
    ):
        raise TypeError(
            f"reward for {owner} must be a tensor"
        )

    if value.numel() != 1:
        raise ValueError(
            f"reward for {owner} must contain exactly one scalar"
        )

    if not value.is_floating_point():
        raise TypeError(
            f"reward for {owner} must use floating dtype"
        )

    result = (
        value.detach()
        .clone()
        .reshape(())
    )

    if not bool(
        torch.isfinite(
            result
        )
    ):
        raise ValueError(
            f"reward for {owner} must be finite"
        )

    return result


def build_altruistic_rollout_sample(
    *,
    possible_agents: Sequence[str],
    module_ids_by_row: Sequence[str],
    owner_to_module_id: Mapping[
        str,
        str,
    ],
    module_positions_by_id: Mapping[
        str,
        torch.Tensor,
    ],
    owner_rewards: Mapping[
        str,
        torch.Tensor,
    ],
    radius_m: float,
    next_module_ids_by_row: (
        Sequence[str]
        | None
    ) = None,
    next_owner_to_module_id: (
        Mapping[
            str,
            str,
        ]
        | None
    ) = None,
) -> AltruisticRolloutSample:
    """Build one causal APPO sidecar sample.

    `module_ids_by_row` is the exact row order used by the critic graph.
    `owner_to_module_id` is supplied explicitly by the runtime.

    No owner name is interpreted as a tensor index.
    """
    agents = _agent_ids(
        possible_agents
    )

    rows = _module_rows(
        module_ids_by_row
    )

    expected = set(
        agents
    )

    if (
        not isinstance(
            owner_to_module_id,
            Mapping,
        )
        or set(
            owner_to_module_id
        )
        != expected
    ):
        raise ValueError(
            "owner_to_module_id must cover exactly possible_agents"
        )

    mapped_modules = tuple(
        owner_to_module_id[
            agent
        ]
        for agent in agents
    )

    if any(
        type(module_id) is not str
        or not module_id
        for module_id in mapped_modules
    ):
        raise ValueError(
            "owner module IDs must be non-empty strings"
        )

    if len(
        set(
            mapped_modules
        )
    ) != len(
        mapped_modules
    ):
        raise ValueError(
            "multiple owners cannot bind the same module"
        )

    row_lookup = {
        module_id: index
        for index, module_id
        in enumerate(
            rows
        )
    }

    missing_rows = (
        set(
            mapped_modules
        )
        - set(
            rows
        )
    )

    if missing_rows:
        raise ValueError(
            "owner binding references module absent from graph rows"
        )

    if (
        not isinstance(
            module_positions_by_id,
            Mapping,
        )
    ):
        raise TypeError(
            "module_positions_by_id must be a mapping"
        )

    if (
        not isinstance(
            owner_rewards,
            Mapping,
        )
        or set(
            owner_rewards
        )
        != expected
    ):
        raise ValueError(
            "owner_rewards must cover exactly possible_agents"
        )

    owner_positions = {}

    for agent in agents:
        module_id = (
            owner_to_module_id[
                agent
            ]
        )

        if (
            module_id
            not in module_positions_by_id
        ):
            raise ValueError(
                f"missing current position for module {module_id}"
            )

        position = (
            module_positions_by_id[
                module_id
            ]
        )

        if not torch.is_tensor(
            position
        ):
            raise TypeError(
                "module positions must be tensors"
            )

        owner_positions[
            agent
        ] = (
            position.detach()
            .clone()
        )

    stored_rewards = {
        agent: _scalar_reward(
            owner_rewards[
                agent
            ],
            owner=agent,
        )
        for agent in agents
    }

    neighbourhoods = (
        build_radius_neighbourhoods(
            owner_positions,
            radius_m=radius_m,
        )
    )

    mean_field_rewards = (
        compute_mean_field_rewards(
            stored_rewards,
            neighbourhoods,
        )
    )

    # Ensure the sidecar owns its tensors independently.
    mean_field_rewards = {
        owner: value.detach().clone()
        for owner, value
        in mean_field_rewards.items()
    }

    next_owner_row_index = None

    if (
        (
            next_module_ids_by_row
            is None
        )
        != (
            next_owner_to_module_id
            is None
        )
    ):
        raise ValueError(
            "next graph rows and next owner binding "
            "must be provided together"
        )

    if (
        next_module_ids_by_row
        is not None
    ):
        next_rows = _module_rows(
            next_module_ids_by_row
        )

        if (
            not isinstance(
                next_owner_to_module_id,
                Mapping,
            )
            or set(
                next_owner_to_module_id
            )
            != expected
        ):
            raise ValueError(
                "next_owner_to_module_id must cover "
                "exactly possible_agents"
            )

        next_modules = tuple(
            next_owner_to_module_id[
                agent
            ]
            for agent in agents
        )

        if (
            any(
                type(module_id)
                is not str
                or not module_id
                for module_id
                in next_modules
            )
            or len(
                set(
                    next_modules
                )
            )
            != len(
                next_modules
            )
        ):
            raise ValueError(
                "next owner bindings must reference "
                "unique non-empty module IDs"
            )

        next_row_lookup = {
            module_id: index
            for index, module_id
            in enumerate(
                next_rows
            )
        }

        if (
            set(
                next_modules
            )
            - set(
                next_rows
            )
        ):
            raise ValueError(
                "next owner binding references module "
                "absent from next graph rows"
            )

        next_owner_row_index = {
            agent: next_row_lookup[
                next_owner_to_module_id[
                    agent
                ]
            ]
            for agent in agents
        }

    return AltruisticRolloutSample(
        owner_row_index={
            agent: row_lookup[
                owner_to_module_id[
                    agent
                ]
            ]
            for agent in agents
        },
        owner_rewards=stored_rewards,
        mean_field_rewards=(
            mean_field_rewards
        ),
        neighbourhoods={
            owner: tuple(
                neighbours
            )
            for owner, neighbours
            in neighbourhoods.items()
        },
        next_owner_row_index=(
            next_owner_row_index
        ),
    )


def build_altruistic_rollout_batch(
    *,
    possible_agents: Sequence[str],
    module_ids_by_row: Sequence[
        Sequence[str]
    ],
    owner_to_module_id: Sequence[
        Mapping[
            str,
            str,
        ]
    ],
    module_positions_by_id: Sequence[
        Mapping[
            str,
            torch.Tensor,
        ]
    ],
    owner_rewards: Mapping[
        str,
        torch.Tensor,
    ],
    radius_m: float,
    next_module_ids_by_row: (
        Sequence[
            Sequence[str]
        ]
        | None
    ) = None,
    next_owner_to_module_id: (
        Sequence[
            Mapping[
                str,
                str,
            ]
        ]
        | None
    ) = None,
) -> tuple[
    AltruisticRolloutSample,
    ...,
]:
    """Convert one vectorized env transition into per-env APPO samples."""
    agents = _agent_ids(
        possible_agents
    )

    env_rows = tuple(
        module_ids_by_row
    )

    env_bindings = tuple(
        owner_to_module_id
    )

    env_positions = tuple(
        module_positions_by_id
    )

    if (
        (
            next_module_ids_by_row
            is None
        )
        != (
            next_owner_to_module_id
            is None
        )
    ):
        raise ValueError(
            "next graph rows and next owner bindings "
            "must be provided together"
        )

    env_next_rows = (
        None
        if next_module_ids_by_row
        is None
        else tuple(
            next_module_ids_by_row
        )
    )

    env_next_bindings = (
        None
        if next_owner_to_module_id
        is None
        else tuple(
            next_owner_to_module_id
        )
    )

    batch_size = len(
        env_rows
    )

    if batch_size < 1:
        raise ValueError(
            "altruistic rollout batch must contain at least one environment"
        )

    if (
        len(
            env_bindings
        )
        != batch_size
        or len(
            env_positions
        )
        != batch_size
    ):
        raise ValueError(
            "vectorized altruistic metadata must share environment count"
        )

    if (
        env_next_rows
        is not None
        and (
            len(
                env_next_rows
            )
            != batch_size
            or len(
                env_next_bindings
            )
            != batch_size
        )
    ):
        raise ValueError(
            "vectorized next-state metadata must share "
            "environment count"
        )

    if (
        not isinstance(
            owner_rewards,
            Mapping,
        )
        or set(
            owner_rewards
        )
        != set(
            agents
        )
    ):
        raise ValueError(
            "owner_rewards must cover exactly possible_agents"
        )

    for agent in agents:
        tensor = owner_rewards[
            agent
        ]

        if (
            not torch.is_tensor(
                tensor
            )
            or tensor.ndim != 2
            or tensor.shape
            != (
                batch_size,
                1,
            )
        ):
            raise ValueError(
                "batched owner reward tensors must have shape (batch, 1)"
            )

    samples = []

    for env_index in range(
        batch_size
    ):
        samples.append(
            build_altruistic_rollout_sample(
                possible_agents=agents,
                module_ids_by_row=(
                    env_rows[
                        env_index
                    ]
                ),
                owner_to_module_id=(
                    env_bindings[
                        env_index
                    ]
                ),
                module_positions_by_id=(
                    env_positions[
                        env_index
                    ]
                ),
                owner_rewards={
                    agent: owner_rewards[
                        agent
                    ][
                        env_index,
                        0,
                    ]
                    for agent in agents
                },
                radius_m=radius_m,
                next_module_ids_by_row=(
                    None
                    if env_next_rows
                    is None
                    else env_next_rows[
                        env_index
                    ]
                ),
                next_owner_to_module_id=(
                    None
                    if env_next_bindings
                    is None
                    else env_next_bindings[
                        env_index
                    ]
                ),
            )
        )

    return tuple(
        samples
    )


class AltruisticRolloutSidecar:
    """Time-major storage aligned with vectorized MARL rollout steps."""

    def __init__(
        self,
    ) -> None:
        self._steps: list[
            tuple[
                AltruisticRolloutSample,
                ...,
            ]
        ] = []

        self._num_envs: (
            int
            | None
        ) = None

    def __len__(
        self,
    ) -> int:
        return len(
            self._steps
        )

    @property
    def num_envs(
        self,
    ) -> int | None:
        return self._num_envs

    def append_step(
        self,
        samples: Sequence[
            AltruisticRolloutSample
        ],
    ) -> None:
        samples = tuple(
            samples
        )

        if not samples:
            raise ValueError(
                "altruistic rollout step must contain at least one environment"
            )

        if not all(
            isinstance(
                sample,
                AltruisticRolloutSample,
            )
            for sample in samples
        ):
            raise TypeError(
                "altruistic rollout step contains invalid sample"
            )

        if self._num_envs is None:
            self._num_envs = len(
                samples
            )

        elif len(
            samples
        ) != self._num_envs:
            raise ValueError(
                "altruistic rollout environment count changed"
            )

        # Samples already own detached/cloned tensors.
        self._steps.append(
            samples
        )

    def flatten(
        self,
    ) -> tuple[
        AltruisticRolloutSample,
        ...,
    ]:
        """Return time-major, then environment-major samples."""
        return tuple(
            sample
            for step in self._steps
            for sample in step
        )

    def sample_by_index(
        self,
        indexes,
    ):
        flat = self.flatten()

        return tuple(
            flat[
                int(
                    index
                )
            ]
            for index in indexes
        )

    def sample_minibatches(
        self,
        *,
        indexes,
        mini_batches: int,
    ):
        if (
            isinstance(
                mini_batches,
                bool,
            )
            or not isinstance(
                mini_batches,
                int,
            )
            or mini_batches <= 0
        ):
            raise ValueError(
                "mini_batches must be a positive integer"
            )

        indexes = (
            indexes.tolist()
            if hasattr(
                indexes,
                "tolist",
            )
            else list(
                indexes
            )
        )

        quotient, remainder = divmod(
            len(
                indexes
            ),
            mini_batches,
        )

        batches = []

        start = 0

        for batch_index in range(
            mini_batches
        ):
            size = (
                quotient
                + (
                    1
                    if batch_index
                    < remainder
                    else 0
                )
            )

            stop = (
                start
                + size
            )

            batches.append(
                self.sample_by_index(
                    indexes[
                        start:stop
                    ]
                )
            )

            start = stop

        return batches

    def clear(
        self,
    ) -> None:
        self._steps.clear()
        self._num_envs = None
