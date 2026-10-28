from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch


def _owner_ids(
    mapping: Mapping[str, Any],
    *,
    label: str,
) -> tuple[str, ...]:
    if (
        not isinstance(
            mapping,
            Mapping,
        )
        or not mapping
    ):
        raise ValueError(
            f"{label} must be a non-empty mapping"
        )

    owners = tuple(
        mapping
    )

    if any(
        type(owner) is not str
        or not owner
        for owner in owners
    ):
        raise ValueError(
            f"{label} owner IDs must be non-empty strings"
        )

    return owners


def _finite_alpha(
    value: Any,
    *,
    owner: str | None = None,
) -> float:
    if (
        isinstance(
            value,
            bool,
        )
        or type(value)
        not in (
            int,
            float,
        )
    ):
        prefix = (
            ""
            if owner is None
            else f" for {owner}"
        )

        raise TypeError(
            "fixed altruism alpha"
            f"{prefix} must be numeric"
        )

    value = float(
        value
    )

    if (
        not math.isfinite(
            value
        )
        or value < -1.0
        or value > 1.0
    ):
        prefix = (
            ""
            if owner is None
            else f" for {owner}"
        )

        raise ValueError(
            "fixed altruism alpha"
            f"{prefix} must lie in [-1, 1]"
        )

    return value


def build_radius_neighbourhoods(
    positions: Mapping[
        str,
        torch.Tensor,
    ],
    *,
    radius_m: float,
) -> dict[
    str,
    tuple[str, ...],
]:
    """Build the causal current-time neighbour sets N_i,t.

    Only positions supplied for the current state are consumed. No future
    state, reward or action enters this function.
    """
    owners = _owner_ids(
        positions,
        label="positions",
    )

    if (
        isinstance(
            radius_m,
            bool,
        )
        or type(radius_m)
        not in (
            int,
            float,
        )
    ):
        raise TypeError(
            "radius_m must be numeric"
        )

    radius_m = float(
        radius_m
    )

    if (
        not math.isfinite(
            radius_m
        )
        or radius_m <= 0.0
    ):
        raise ValueError(
            "radius_m must be finite and positive"
        )

    vectors = {}

    for owner in owners:
        value = positions[
            owner
        ]

        if not torch.is_tensor(
            value
        ):
            raise TypeError(
                "module positions must be tensors"
            )

        if value.shape != (
            3,
        ):
            raise ValueError(
                "module positions must have shape (3,)"
            )

        detached = (
            value.detach()
            .to(
                dtype=torch.float64,
            )
        )

        if not bool(
            torch.isfinite(
                detached
            ).all()
        ):
            raise ValueError(
                "module positions must be finite"
            )

        vectors[
            owner
        ] = detached

    result = {
        owner: []
        for owner in owners
    }

    # Pairwise construction guarantees symmetry by construction and
    # excludes self from every N_i,t.
    for index, first in enumerate(
        owners
    ):
        for second in owners[
            index + 1:
        ]:
            distance = float(
                torch.linalg.vector_norm(
                    vectors[
                        first
                    ]
                    - vectors[
                        second
                    ]
                ).item()
            )

            if distance <= radius_m:
                result[
                    first
                ].append(
                    second
                )

                result[
                    second
                ].append(
                    first
                )

    return {
        owner: tuple(
            result[
                owner
            ]
        )
        for owner in owners
    }


def compute_mean_field_rewards(
    owner_rewards: Mapping[
        str,
        torch.Tensor,
    ],
    neighbourhoods: Mapping[
        str,
        Sequence[str],
    ],
) -> dict[
    str,
    torch.Tensor,
]:
    """Compute R_MF_i,t as the mean reward of current neighbours.

    The source paper leaves |N_i,t| = 0 undefined. In this implementation,
    an empty neighbourhood contributes exactly zero mean-field reward.
    """
    owners = _owner_ids(
        owner_rewards,
        label="owner_rewards",
    )

    if set(
        neighbourhoods
    ) != set(
        owners
    ):
        raise ValueError(
            "neighbourhoods must cover exactly the reward owners"
        )

    for owner in owners:
        reward = owner_rewards[
            owner
        ]

        if not torch.is_tensor(
            reward
        ):
            raise TypeError(
                "owner rewards must be tensors"
            )

        if not reward.is_floating_point():
            raise TypeError(
                "owner rewards must use floating dtype"
            )

        if not bool(
            torch.isfinite(
                reward
            ).all()
        ):
            raise ValueError(
                "owner rewards must be finite"
            )

    result = {}

    for owner in owners:
        neighbours = tuple(
            neighbourhoods[
                owner
            ]
        )

        if len(
            set(
                neighbours
            )
        ) != len(
            neighbours
        ):
            raise ValueError(
                "neighbourhood contains duplicate owners"
            )

        if owner in neighbours:
            raise ValueError(
                "an owner cannot be its own neighbour"
            )

        unknown = (
            set(
                neighbours
            )
            - set(
                owners
            )
        )

        if unknown:
            raise ValueError(
                "neighbourhood references unknown owner"
            )

        base = owner_rewards[
            owner
        ]

        if not neighbours:
            result[
                owner
            ] = torch.zeros_like(
                base
            )
            continue

        neighbour_rewards = []

        for neighbour in neighbours:
            value = owner_rewards[
                neighbour
            ]

            if (
                value.shape
                != base.shape
                or value.dtype
                != base.dtype
                or value.device
                != base.device
            ):
                raise ValueError(
                    "neighbour reward tensors must match owner shape/dtype/device"
                )

            neighbour_rewards.append(
                value
            )

        result[
            owner
        ] = torch.stack(
            neighbour_rewards,
            dim=0,
        ).mean(
            dim=0
        )

    return result


def _alpha_by_owner(
    owners: tuple[str, ...],
    alpha: float
    | Mapping[
        str,
        float,
    ],
) -> dict[
    str,
    float,
]:
    if isinstance(
        alpha,
        Mapping,
    ):
        if set(
            alpha
        ) != set(
            owners
        ):
            raise ValueError(
                "personalized fixed alpha must cover exactly all owners"
            )

        return {
            owner: _finite_alpha(
                alpha[
                    owner
                ],
                owner=owner,
            )
            for owner in owners
        }

    value = _finite_alpha(
        alpha
    )

    return {
        owner: value
        for owner in owners
    }


def combine_altruistic_advantages(
    owner_advantages: Mapping[
        str,
        torch.Tensor,
    ],
    mean_field_advantages: Mapping[
        str,
        torch.Tensor,
    ],
    *,
    alpha: float
    | Mapping[
        str,
        float,
    ],
) -> dict[
    str,
    torch.Tensor,
]:
    """Compute A_AS_i = A_i + alpha_i * A_MF_i for fixed alpha."""
    owners = _owner_ids(
        owner_advantages,
        label="owner_advantages",
    )

    if set(
        mean_field_advantages
    ) != set(
        owners
    ):
        raise ValueError(
            "mean_field_advantages must cover exactly all owners"
        )

    factors = _alpha_by_owner(
        owners,
        alpha,
    )

    result = {}

    for owner in owners:
        own = owner_advantages[
            owner
        ]

        mean_field = (
            mean_field_advantages[
                owner
            ]
        )

        if (
            not torch.is_tensor(
                own
            )
            or not torch.is_tensor(
                mean_field
            )
        ):
            raise TypeError(
                "advantages must be tensors"
            )

        if (
            own.shape
            != mean_field.shape
            or own.dtype
            != mean_field.dtype
            or own.device
            != mean_field.device
        ):
            raise ValueError(
                "owner and mean-field advantages must match shape/dtype/device"
            )

        result[
            owner
        ] = (
            own
            + factors[
                owner
            ]
            * mean_field
        )

    return result


def select_policy_advantages(
    *,
    mode: str,
    possible_agents: Sequence[
        str,
    ],
    team_advantages: torch.Tensor | None,
    owner_advantages: Mapping[
        str,
        torch.Tensor,
    ]
    | None,
    mean_field_advantages: Mapping[
        str,
        torch.Tensor,
    ]
    | None,
    alpha: float
    | Mapping[
        str,
        float,
    ],
) -> dict[
    str,
    torch.Tensor,
]:
    """Keep team-MAPPO and altruistic APPO as explicit separate modes."""
    agents = tuple(
        possible_agents
    )

    if (
        not agents
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
            "possible_agents must be non-empty and unique"
        )

    if mode == "team":
        if not torch.is_tensor(
            team_advantages
        ):
            raise TypeError(
                "team mode requires team_advantages tensor"
            )

        return {
            owner: team_advantages
            for owner in agents
        }

    if mode != "altruistic":
        raise ValueError(
            "advantage mode must be 'team' or 'altruistic'"
        )

    if (
        owner_advantages is None
        or mean_field_advantages
        is None
    ):
        raise ValueError(
            "altruistic mode requires owner and mean-field advantages"
        )

    if set(
        owner_advantages
    ) != set(
        agents
    ):
        raise ValueError(
            "owner_advantages do not match possible_agents"
        )

    if set(
        mean_field_advantages
    ) != set(
        agents
    ):
        raise ValueError(
            "mean_field_advantages do not match possible_agents"
        )

    return combine_altruistic_advantages(
        owner_advantages,
        mean_field_advantages,
        alpha=alpha,
    )


# ---------------------------------------------------------------------------
# Separate owner / mean-field value and GAE channels
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass(frozen=True)
class GAEChannelResult:
    advantages: torch.Tensor
    returns: torch.Tensor


@dataclass(frozen=True)
class AltruisticGAEResult:
    owner_advantages: dict[
        str,
        torch.Tensor,
    ]
    owner_returns: dict[
        str,
        torch.Tensor,
    ]
    mean_field_advantages: dict[
        str,
        torch.Tensor,
    ]
    mean_field_returns: dict[
        str,
        torch.Tensor,
    ]
    altruistic_advantages: dict[
        str,
        torch.Tensor,
    ]


def _unit_interval(
    value: Any,
    *,
    label: str,
) -> float:
    if (
        isinstance(
            value,
            bool,
        )
        or type(value)
        not in (
            int,
            float,
        )
    ):
        raise TypeError(
            f"{label} must be numeric"
        )

    result = float(
        value
    )

    if (
        not math.isfinite(
            result
        )
        or result < 0.0
        or result > 1.0
    ):
        raise ValueError(
            f"{label} must lie in [0, 1]"
        )

    return result


def compute_gae_channel(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    dones: torch.Tensor,
    discount: float,
    gae_lambda: float,
    terminated: torch.Tensor | None = None,
    next_values: torch.Tensor | None = None,
) -> GAEChannelResult:
    """Compute one independent GAE/value-target channel.

    The leading dimension is time. `bootstrap_value` is V(s_T), while
    `values[t]` is V(s_t). A true entry in `dones[t]` prevents any value
    or GAE propagation across that transition.

    Time-limit bootstrap, when desired, should be handled by the caller
    consistently with the existing MAPPO rollout semantics before this
    function is called.
    """
    discount = _unit_interval(
        discount,
        label="discount",
    )

    gae_lambda = _unit_interval(
        gae_lambda,
        label="gae_lambda",
    )

    for label, tensor in (
        (
            "rewards",
            rewards,
        ),
        (
            "values",
            values,
        ),
        (
            "bootstrap_value",
            bootstrap_value,
        ),
        (
            "dones",
            dones,
        ),
    ):
        if not torch.is_tensor(
            tensor
        ):
            raise TypeError(
                f"{label} must be a tensor"
            )

    if (
        rewards.ndim < 1
        or rewards.shape[0] < 1
    ):
        raise ValueError(
            "rewards must contain at least one timestep"
        )

    if values.shape != rewards.shape:
        raise ValueError(
            "values shape must match rewards shape"
        )

    if dones.shape != rewards.shape:
        raise ValueError(
            "dones shape must match rewards shape"
        )

    if (
        bootstrap_value.shape
        != rewards.shape[1:]
    ):
        raise ValueError(
            "bootstrap_value shape must match one reward timestep"
        )

    if (
        not rewards.is_floating_point()
        or not values.is_floating_point()
        or not bootstrap_value.is_floating_point()
    ):
        raise TypeError(
            "reward/value tensors must use floating dtype"
        )

    if (
        rewards.dtype
        != values.dtype
        or rewards.dtype
        != bootstrap_value.dtype
        or rewards.device
        != values.device
        or rewards.device
        != bootstrap_value.device
    ):
        raise ValueError(
            "reward/value tensors must match dtype/device"
        )

    if not bool(
        torch.isfinite(
            rewards
        ).all()
    ):
        raise ValueError(
            "rewards must be finite"
        )

    if not bool(
        torch.isfinite(
            values
        ).all()
    ):
        raise ValueError(
            "values must be finite"
        )

    if not bool(
        torch.isfinite(
            bootstrap_value
        ).all()
    ):
        raise ValueError(
            "bootstrap_value must be finite"
        )

    done_mask = dones.to(
        dtype=torch.bool,
        device=rewards.device,
    )

    # Stop the trace at both terminal and collection/reset boundaries, but
    # bootstrap time limits from THEIR captured next state. Never use the
    # following episode's initial state as that value.
    terminal_mask = done_mask
    if terminated is not None:
        if terminated.shape != rewards.shape:
            raise ValueError('terminated shape must match rewards')
        terminal_mask = terminated.to(dtype=torch.bool, device=rewards.device)
        if bool((terminal_mask & ~done_mask).any()):
            raise ValueError('terminated must also be an episode boundary')
        if bool((done_mask & ~terminal_mask).any()) and next_values is None:
            raise ValueError('time-limit bootstrap requires captured next_values')
    if next_values is not None:
        if (next_values.shape != rewards.shape or next_values.dtype != rewards.dtype
                or next_values.device != rewards.device or not torch.isfinite(next_values).all()):
            raise ValueError('next_values must be finite and match rewards shape/dtype/device')

    advantages = torch.zeros_like(
        rewards
    )

    next_value = bootstrap_value
    next_gae = torch.zeros_like(
        bootstrap_value
    )

    for timestep in range(
        rewards.shape[0] - 1,
        -1,
        -1,
    ):
        nonterminal = (
            ~done_mask[
                timestep
            ]
        ).to(
            dtype=rewards.dtype
        )

        delta = (
            rewards[
                timestep
            ]
            + discount
            * (next_value if next_values is None else next_values[timestep])
            * (~terminal_mask[timestep]).to(dtype=rewards.dtype)
            - values[
                timestep
            ]
        )

        next_gae = (
            delta
            + discount
            * gae_lambda
            * nonterminal
            * next_gae
        )

        advantages[
            timestep
        ] = next_gae

        next_value = values[
            timestep
        ]

    return GAEChannelResult(
        advantages=advantages,
        returns=(
            advantages
            + values
        ),
    )


def _validate_altruistic_value_maps(
    *,
    owner_rewards,
    mean_field_rewards,
    owner_values,
    mean_field_values,
    owner_bootstrap_values,
    mean_field_bootstrap_values,
) -> tuple[str, ...]:
    owners = _owner_ids(
        owner_rewards,
        label="owner_rewards",
    )

    expected = set(
        owners
    )

    mappings = (
        (
            "mean_field_rewards",
            mean_field_rewards,
        ),
        (
            "owner_values",
            owner_values,
        ),
        (
            "mean_field_values",
            mean_field_values,
        ),
        (
            "owner_bootstrap_values",
            owner_bootstrap_values,
        ),
        (
            "mean_field_bootstrap_values",
            mean_field_bootstrap_values,
        ),
    )

    for label, mapping in mappings:
        if (
            not isinstance(
                mapping,
                Mapping,
            )
            or set(
                mapping
            )
            != expected
        ):
            raise ValueError(
                f"{label} must cover exactly all owners"
            )

    return owners


def compute_altruistic_gae(
    *,
    owner_rewards: Mapping[
        str,
        torch.Tensor,
    ],
    mean_field_rewards: Mapping[
        str,
        torch.Tensor,
    ],
    owner_values: Mapping[
        str,
        torch.Tensor,
    ],
    mean_field_values: Mapping[
        str,
        torch.Tensor,
    ],
    owner_bootstrap_values: Mapping[
        str,
        torch.Tensor,
    ],
    mean_field_bootstrap_values: Mapping[
        str,
        torch.Tensor,
    ],
    dones: torch.Tensor,
    discount: float,
    gae_lambda: float,
    alpha: float
    | Mapping[
        str,
        float,
    ],
) -> AltruisticGAEResult:
    """Compute the two critic channels and the APPO actor advantage.

    Critic targets remain separate:
      - owner_returns trains V_i
      - mean_field_returns trains V_i^MF

    Only the actor receives:
      A_i^AS = A_i + alpha_i * A_i^MF
    """
    owners = _validate_altruistic_value_maps(
        owner_rewards=(
            owner_rewards
        ),
        mean_field_rewards=(
            mean_field_rewards
        ),
        owner_values=(
            owner_values
        ),
        mean_field_values=(
            mean_field_values
        ),
        owner_bootstrap_values=(
            owner_bootstrap_values
        ),
        mean_field_bootstrap_values=(
            mean_field_bootstrap_values
        ),
    )

    owner_advantages = {}
    owner_returns = {}

    mean_field_advantages = {}
    mean_field_returns = {}

    for owner in owners:
        own = compute_gae_channel(
            rewards=owner_rewards[
                owner
            ],
            values=owner_values[
                owner
            ],
            bootstrap_value=(
                owner_bootstrap_values[
                    owner
                ]
            ),
            dones=dones,
            discount=discount,
            gae_lambda=gae_lambda,
        )

        mean_field = compute_gae_channel(
            rewards=mean_field_rewards[
                owner
            ],
            values=mean_field_values[
                owner
            ],
            bootstrap_value=(
                mean_field_bootstrap_values[
                    owner
                ]
            ),
            dones=dones,
            discount=discount,
            gae_lambda=gae_lambda,
        )

        owner_advantages[
            owner
        ] = own.advantages

        owner_returns[
            owner
        ] = own.returns

        mean_field_advantages[
            owner
        ] = (
            mean_field.advantages
        )

        mean_field_returns[
            owner
        ] = (
            mean_field.returns
        )

    altruistic_advantages = (
        combine_altruistic_advantages(
            owner_advantages,
            mean_field_advantages,
            alpha=alpha,
        )
    )

    return AltruisticGAEResult(
        owner_advantages=(
            owner_advantages
        ),
        owner_returns=(
            owner_returns
        ),
        mean_field_advantages=(
            mean_field_advantages
        ),
        mean_field_returns=(
            mean_field_returns
        ),
        altruistic_advantages=(
            altruistic_advantages
        ),
    )
