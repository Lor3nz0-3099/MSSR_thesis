from __future__ import annotations

import pytest
import torch


def test_radius_neighbourhoods_are_current_symmetric_and_exclude_self():
    from drl.MARL.altruism import (
        build_radius_neighbourhoods,
    )

    positions = {
        "module_0": torch.tensor(
            [0.0, 0.0, 0.0]
        ),
        "module_1": torch.tensor(
            [0.3, 0.0, 0.0]
        ),
        "module_2": torch.tensor(
            [1.0, 0.0, 0.0]
        ),
    }

    result = build_radius_neighbourhoods(
        positions,
        radius_m=0.5,
    )

    assert result == {
        "module_0": (
            "module_1",
        ),
        "module_1": (
            "module_0",
        ),
        "module_2": (),
    }


def test_radius_neighbourhood_includes_exact_distance_boundary():
    from drl.MARL.altruism import (
        build_radius_neighbourhoods,
    )

    result = build_radius_neighbourhoods(
        {
            "a": torch.tensor(
                [0.0, 0.0, 0.0]
            ),
            "b": torch.tensor(
                [0.5, 0.0, 0.0]
            ),
        },
        radius_m=0.5,
    )

    assert result == {
        "a": (
            "b",
        ),
        "b": (
            "a",
        ),
    }


@pytest.mark.parametrize(
    "positions,radius",
    (
        (
            {},
            1.0,
        ),
        (
            {
                "a": torch.tensor(
                    [0.0, 0.0]
                )
            },
            1.0,
        ),
        (
            {
                "a": torch.tensor(
                    [
                        float("nan"),
                        0.0,
                        0.0,
                    ]
                )
            },
            1.0,
        ),
        (
            {
                "a": torch.zeros(
                    3
                )
            },
            0.0,
        ),
        (
            {
                "a": torch.zeros(
                    3
                )
            },
            float("inf"),
        ),
    ),
)
def test_radius_neighbourhood_rejects_malformed_current_geometry(
    positions,
    radius,
):
    from drl.MARL.altruism import (
        build_radius_neighbourhoods,
    )

    with pytest.raises(
        (
            TypeError,
            ValueError,
        )
    ):
        build_radius_neighbourhoods(
            positions,
            radius_m=radius,
        )


def test_mean_field_reward_is_average_of_current_neighbours_only():
    from drl.MARL.altruism import (
        compute_mean_field_rewards,
    )

    rewards = {
        "module_0": torch.tensor(
            [[1.0], [2.0]]
        ),
        "module_1": torch.tensor(
            [[3.0], [4.0]]
        ),
        "module_2": torch.tensor(
            [[5.0], [8.0]]
        ),
    }

    neighbours = {
        "module_0": (
            "module_1",
            "module_2",
        ),
        "module_1": (
            "module_0",
        ),
        "module_2": (
            "module_0",
        ),
    }

    result = compute_mean_field_rewards(
        rewards,
        neighbours,
    )

    torch.testing.assert_close(
        result["module_0"],
        torch.tensor(
            [[4.0], [6.0]]
        ),
    )

    torch.testing.assert_close(
        result["module_1"],
        rewards[
            "module_0"
        ],
    )

    torch.testing.assert_close(
        result["module_2"],
        rewards[
            "module_0"
        ],
    )


def test_mean_field_reward_is_zero_when_owner_has_no_current_neighbours():
    from drl.MARL.altruism import (
        compute_mean_field_rewards,
    )

    reward = torch.tensor(
        [[2.0], [-3.0]]
    )

    result = compute_mean_field_rewards(
        {
            "module_0": reward,
        },
        {
            "module_0": (),
        },
    )

    torch.testing.assert_close(
        result["module_0"],
        torch.zeros_like(
            reward
        ),
    )


def test_mean_field_reward_rejects_self_unknown_and_duplicate_neighbours():
    from drl.MARL.altruism import (
        compute_mean_field_rewards,
    )

    rewards = {
        "a": torch.tensor(
            [1.0]
        ),
        "b": torch.tensor(
            [2.0]
        ),
    }

    invalid = (
        {
            "a": (
                "a",
            ),
            "b": (),
        },
        {
            "a": (
                "missing",
            ),
            "b": (),
        },
        {
            "a": (
                "b",
                "b",
            ),
            "b": (
                "a",
            ),
        },
    )

    for neighbourhoods in invalid:
        with pytest.raises(
            ValueError
        ):
            compute_mean_field_rewards(
                rewards,
                neighbourhoods,
            )


def test_alpha_zero_is_exactly_selfish_owner_advantage():
    from drl.MARL.altruism import (
        combine_altruistic_advantages,
    )

    own = {
        "module_0": torch.tensor(
            [[1.0], [-2.0]]
        ),
        "module_1": torch.tensor(
            [[3.0], [4.0]]
        ),
    }

    mean_field = {
        "module_0": torch.tensor(
            [[100.0], [100.0]]
        ),
        "module_1": torch.tensor(
            [[-100.0], [-100.0]]
        ),
    }

    result = combine_altruistic_advantages(
        own,
        mean_field,
        alpha=0.0,
    )

    for uid in own:
        torch.testing.assert_close(
            result[
                uid
            ],
            own[
                uid
            ],
        )


def test_positive_fixed_alpha_changes_advantage_in_neighbour_benefit_direction():
    from drl.MARL.altruism import (
        combine_altruistic_advantages,
    )

    own = {
        "a": torch.tensor(
            [1.0]
        ),
        "b": torch.tensor(
            [1.0]
        ),
    }

    mean_field = {
        "a": torch.tensor(
            [4.0]
        ),
        "b": torch.tensor(
            [-2.0]
        ),
    }

    result = combine_altruistic_advantages(
        own,
        mean_field,
        alpha=0.5,
    )

    torch.testing.assert_close(
        result["a"],
        torch.tensor(
            [3.0]
        ),
    )

    torch.testing.assert_close(
        result["b"],
        torch.tensor(
            [0.0]
        ),
    )


def test_fixed_alpha_can_be_personalized_without_being_learnable_yet():
    from drl.MARL.altruism import (
        combine_altruistic_advantages,
    )

    own = {
        "a": torch.tensor(
            [2.0]
        ),
        "b": torch.tensor(
            [2.0]
        ),
    }

    mean_field = {
        "a": torch.tensor(
            [1.0]
        ),
        "b": torch.tensor(
            [1.0]
        ),
    }

    result = combine_altruistic_advantages(
        own,
        mean_field,
        alpha={
            "a": 1.0,
            "b": -1.0,
        },
    )

    torch.testing.assert_close(
        result["a"],
        torch.tensor(
            [3.0]
        ),
    )

    torch.testing.assert_close(
        result["b"],
        torch.tensor(
            [1.0]
        ),
    )


@pytest.mark.parametrize(
    "alpha",
    (
        1.01,
        -1.01,
        float("nan"),
        float("inf"),
        True,
    ),
)
def test_fixed_alpha_rejects_values_outside_supported_range(
    alpha,
):
    from drl.MARL.altruism import (
        combine_altruistic_advantages,
    )

    with pytest.raises(
        (
            TypeError,
            ValueError,
        )
    ):
        combine_altruistic_advantages(
            {
                "a": torch.tensor(
                    [1.0]
                )
            },
            {
                "a": torch.tensor(
                    [1.0]
                )
            },
            alpha=alpha,
        )


def test_advantage_mode_keeps_team_mappo_as_separate_ablation():
    from drl.MARL.altruism import (
        select_policy_advantages,
    )

    team = torch.tensor(
        [[7.0], [8.0]]
    )

    own = {
        "a": torch.tensor(
            [[1.0], [2.0]]
        ),
        "b": torch.tensor(
            [[3.0], [4.0]]
        ),
    }

    mean_field = {
        "a": torch.tensor(
            [[5.0], [6.0]]
        ),
        "b": torch.tensor(
            [[7.0], [8.0]]
        ),
    }

    team_result = select_policy_advantages(
        mode="team",
        possible_agents=(
            "a",
            "b",
        ),
        team_advantages=team,
        owner_advantages=None,
        mean_field_advantages=None,
        alpha=0.5,
    )

    assert (
        team_result[
            "a"
        ]
        is team
    )

    assert (
        team_result[
            "b"
        ]
        is team
    )

    altruistic = select_policy_advantages(
        mode="altruistic",
        possible_agents=(
            "a",
            "b",
        ),
        team_advantages=None,
        owner_advantages=own,
        mean_field_advantages=mean_field,
        alpha=0.5,
    )

    torch.testing.assert_close(
        altruistic["a"],
        own["a"]
        + 0.5
        * mean_field["a"],
    )

    torch.testing.assert_close(
        altruistic["b"],
        own["b"]
        + 0.5
        * mean_field["b"],
    )


# --- SEPARATE OWNER AND MEAN-FIELD GAE ---

def test_gae_channel_one_step_matches_td_advantage():
    import torch

    from drl.MARL.altruism import (
        compute_gae_channel,
    )

    result = compute_gae_channel(
        rewards=torch.tensor(
            [[2.0]]
        ),
        values=torch.tensor(
            [[0.5]]
        ),
        bootstrap_value=torch.tensor(
            [1.0]
        ),
        dones=torch.tensor(
            [[False]]
        ),
        discount=0.9,
        gae_lambda=0.95,
    )

    # δ = r + γ V(s') - V(s)
    expected = torch.tensor(
        [[2.4]]
    )

    torch.testing.assert_close(
        result.advantages,
        expected,
    )

    torch.testing.assert_close(
        result.returns,
        expected
        + torch.tensor(
            [[0.5]]
        ),
    )


def test_gae_channel_terminal_transition_cuts_bootstrap():
    import torch

    from drl.MARL.altruism import (
        compute_gae_channel,
    )

    result = compute_gae_channel(
        rewards=torch.tensor(
            [[2.0]]
        ),
        values=torch.tensor(
            [[0.5]]
        ),
        bootstrap_value=torch.tensor(
            [100.0]
        ),
        dones=torch.tensor(
            [[True]]
        ),
        discount=0.9,
        gae_lambda=0.95,
    )

    # Terminal: δ = r - V(s), bootstrap must not leak through.
    torch.testing.assert_close(
        result.advantages,
        torch.tensor(
            [[1.5]]
        ),
    )


def test_gae_channel_multistep_uses_only_causal_trajectory_successor_values():
    import torch

    from drl.MARL.altruism import (
        compute_gae_channel,
    )

    rewards = torch.tensor(
        [
            [1.0],
            [2.0],
        ]
    )

    values = torch.tensor(
        [
            [0.5],
            [0.25],
        ]
    )

    result = compute_gae_channel(
        rewards=rewards,
        values=values,
        bootstrap_value=torch.tensor(
            [0.0]
        ),
        dones=torch.tensor(
            [
                [False],
                [True],
            ]
        ),
        discount=1.0,
        gae_lambda=1.0,
    )

    # t=1: 2 - .25 = 1.75
    # t=0: 1 + .25 - .5 + 1.75 = 2.5
    torch.testing.assert_close(
        result.advantages,
        torch.tensor(
            [
                [2.5],
                [1.75],
            ]
        ),
    )


def test_altruistic_gae_uses_distinct_owner_and_mean_field_value_signals():
    import torch

    from drl.MARL.altruism import (
        compute_altruistic_gae,
    )

    result = compute_altruistic_gae(
        owner_rewards={
            "module_0": torch.tensor(
                [[1.0]]
            ),
        },
        mean_field_rewards={
            "module_0": torch.tensor(
                [[3.0]]
            ),
        },
        owner_values={
            "module_0": torch.tensor(
                [[0.2]]
            ),
        },
        mean_field_values={
            "module_0": torch.tensor(
                [[-1.0]]
            ),
        },
        owner_bootstrap_values={
            "module_0": torch.tensor(
                [0.4]
            ),
        },
        mean_field_bootstrap_values={
            "module_0": torch.tensor(
                [2.0]
            ),
        },
        dones=torch.tensor(
            [[False]]
        ),
        discount=0.5,
        gae_lambda=1.0,
        alpha=0.25,
    )

    # Own:
    # 1 + .5*.4 - .2 = 1.0
    torch.testing.assert_close(
        result.owner_advantages[
            "module_0"
        ],
        torch.tensor(
            [[1.0]]
        ),
    )

    # Mean field:
    # 3 + .5*2 - (-1) = 5
    torch.testing.assert_close(
        result.mean_field_advantages[
            "module_0"
        ],
        torch.tensor(
            [[5.0]]
        ),
    )

    # APPO:
    # 1 + .25*5 = 2.25
    torch.testing.assert_close(
        result.altruistic_advantages[
            "module_0"
        ],
        torch.tensor(
            [[2.25]]
        ),
    )


def test_altruistic_gae_alpha_zero_is_exact_selfish_gae():
    import torch

    from drl.MARL.altruism import (
        compute_altruistic_gae,
    )

    result = compute_altruistic_gae(
        owner_rewards={
            "a": torch.tensor(
                [
                    [1.0],
                    [2.0],
                ]
            ),
        },
        mean_field_rewards={
            "a": torch.tensor(
                [
                    [100.0],
                    [100.0],
                ]
            ),
        },
        owner_values={
            "a": torch.zeros(
                2,
                1,
            ),
        },
        mean_field_values={
            "a": torch.zeros(
                2,
                1,
            ),
        },
        owner_bootstrap_values={
            "a": torch.zeros(
                1
            ),
        },
        mean_field_bootstrap_values={
            "a": torch.zeros(
                1
            ),
        },
        dones=torch.tensor(
            [
                [False],
                [True],
            ]
        ),
        discount=0.99,
        gae_lambda=0.95,
        alpha=0.0,
    )

    torch.testing.assert_close(
        result.altruistic_advantages[
            "a"
        ],
        result.owner_advantages[
            "a"
        ],
    )


def test_altruistic_gae_keeps_two_separate_critic_return_targets():
    import torch

    from drl.MARL.altruism import (
        compute_altruistic_gae,
    )

    result = compute_altruistic_gae(
        owner_rewards={
            "a": torch.tensor(
                [[1.0]]
            ),
        },
        mean_field_rewards={
            "a": torch.tensor(
                [[10.0]]
            ),
        },
        owner_values={
            "a": torch.tensor(
                [[0.0]]
            ),
        },
        mean_field_values={
            "a": torch.tensor(
                [[5.0]]
            ),
        },
        owner_bootstrap_values={
            "a": torch.tensor(
                [0.0]
            ),
        },
        mean_field_bootstrap_values={
            "a": torch.tensor(
                [0.0]
            ),
        },
        dones=torch.tensor(
            [[True]]
        ),
        discount=0.99,
        gae_lambda=0.95,
        alpha=1.0,
    )

    # V_i target and V_i^MF target are deliberately distinct.
    torch.testing.assert_close(
        result.owner_returns[
            "a"
        ],
        torch.tensor(
            [[1.0]]
        ),
    )

    torch.testing.assert_close(
        result.mean_field_returns[
            "a"
        ],
        torch.tensor(
            [[10.0]]
        ),
    )

    # The altruistic advantage is an ACTOR signal, not a replacement
    # target for either critic.
    torch.testing.assert_close(
        result.altruistic_advantages[
            "a"
        ],
        torch.tensor(
            [[6.0]]
        ),
    )


def test_altruistic_gae_supports_multiple_owners_without_cross_mixing():
    import torch

    from drl.MARL.altruism import (
        compute_altruistic_gae,
    )

    result = compute_altruistic_gae(
        owner_rewards={
            "a": torch.tensor(
                [[1.0]]
            ),
            "b": torch.tensor(
                [[4.0]]
            ),
        },
        mean_field_rewards={
            "a": torch.tensor(
                [[10.0]]
            ),
            "b": torch.tensor(
                [[20.0]]
            ),
        },
        owner_values={
            "a": torch.zeros(
                1,
                1,
            ),
            "b": torch.zeros(
                1,
                1,
            ),
        },
        mean_field_values={
            "a": torch.zeros(
                1,
                1,
            ),
            "b": torch.zeros(
                1,
                1,
            ),
        },
        owner_bootstrap_values={
            "a": torch.zeros(
                1
            ),
            "b": torch.zeros(
                1
            ),
        },
        mean_field_bootstrap_values={
            "a": torch.zeros(
                1
            ),
            "b": torch.zeros(
                1
            ),
        },
        dones=torch.tensor(
            [[True]]
        ),
        discount=1.0,
        gae_lambda=1.0,
        alpha={
            "a": 0.5,
            "b": -0.5,
        },
    )

    torch.testing.assert_close(
        result.altruistic_advantages[
            "a"
        ],
        torch.tensor(
            [[6.0]]
        ),
    )

    torch.testing.assert_close(
        result.altruistic_advantages[
            "b"
        ],
        torch.tensor(
            [[-6.0]]
        ),
    )


@pytest.mark.parametrize(
    "discount,gae_lambda",
    (
        (-0.1, 0.95),
        (1.01, 0.95),
        (0.99, -0.1),
        (0.99, 1.01),
        (float("nan"), 0.95),
        (0.99, float("inf")),
    ),
)
def test_gae_rejects_invalid_discount_or_lambda(
    discount,
    gae_lambda,
):
    import torch

    from drl.MARL.altruism import (
        compute_gae_channel,
    )

    with pytest.raises(
        ValueError
    ):
        compute_gae_channel(
            rewards=torch.zeros(
                1,
                1,
            ),
            values=torch.zeros(
                1,
                1,
            ),
            bootstrap_value=torch.zeros(
                1
            ),
            dones=torch.zeros(
                1,
                1,
                dtype=torch.bool,
            ),
            discount=discount,
            gae_lambda=gae_lambda,
        )


def test_gae_rejects_shape_mismatch():
    import torch

    from drl.MARL.altruism import (
        compute_gae_channel,
    )

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        compute_gae_channel(
            rewards=torch.zeros(
                2,
                1,
            ),
            values=torch.zeros(
                1,
                1,
            ),
            bootstrap_value=torch.zeros(
                1
            ),
            dones=torch.zeros(
                2,
                1,
                dtype=torch.bool,
            ),
            discount=0.99,
            gae_lambda=0.95,
        )
