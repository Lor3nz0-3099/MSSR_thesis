from __future__ import annotations

import pytest
import torch


def test_sample_maps_owner_to_actual_graph_row_not_owner_name():
    from drl.MARL.altruistic_rollout import (
        build_altruistic_rollout_sample,
    )

    sample = build_altruistic_rollout_sample(
        possible_agents=(
            "owner_a",
            "owner_b",
            "owner_c",
        ),
        module_ids_by_row=(
            "smores_07",
            "smores_02",
            "smores_05",
        ),
        owner_to_module_id={
            "owner_a": "smores_05",
            "owner_b": "smores_07",
            "owner_c": "smores_02",
        },
        module_positions_by_id={
            "smores_07": torch.tensor(
                [0.0, 0.0, 0.0]
            ),
            "smores_02": torch.tensor(
                [2.0, 0.0, 0.0]
            ),
            "smores_05": torch.tensor(
                [4.0, 0.0, 0.0]
            ),
        },
        owner_rewards={
            "owner_a": torch.tensor(
                1.0
            ),
            "owner_b": torch.tensor(
                2.0
            ),
            "owner_c": torch.tensor(
                3.0
            ),
        },
        radius_m=0.5,
    )

    assert sample.owner_row_index == {
        "owner_a": 2,
        "owner_b": 0,
        "owner_c": 1,
    }


def test_sample_builds_current_neighbourhood_and_mean_field_reward():
    from drl.MARL.altruistic_rollout import (
        build_altruistic_rollout_sample,
    )

    sample = build_altruistic_rollout_sample(
        possible_agents=(
            "a",
            "b",
            "c",
        ),
        module_ids_by_row=(
            "m2",
            "m0",
            "m1",
        ),
        owner_to_module_id={
            "a": "m0",
            "b": "m1",
            "c": "m2",
        },
        module_positions_by_id={
            "m0": torch.tensor(
                [0.0, 0.0, 0.0]
            ),
            "m1": torch.tensor(
                [0.2, 0.0, 0.0]
            ),
            "m2": torch.tensor(
                [2.0, 0.0, 0.0]
            ),
        },
        owner_rewards={
            "a": torch.tensor(
                1.0
            ),
            "b": torch.tensor(
                5.0
            ),
            "c": torch.tensor(
                9.0
            ),
        },
        radius_m=0.5,
    )

    assert sample.neighbourhoods == {
        "a": (
            "b",
        ),
        "b": (
            "a",
        ),
        "c": (),
    }

    torch.testing.assert_close(
        sample.mean_field_rewards[
            "a"
        ],
        torch.tensor(
            5.0
        ),
    )

    torch.testing.assert_close(
        sample.mean_field_rewards[
            "b"
        ],
        torch.tensor(
            1.0
        ),
    )

    torch.testing.assert_close(
        sample.mean_field_rewards[
            "c"
        ],
        torch.tensor(
            0.0
        ),
    )


def test_sample_stores_detached_cloned_owner_rewards():
    from drl.MARL.altruistic_rollout import (
        build_altruistic_rollout_sample,
    )

    reward = torch.tensor(
        3.0,
        requires_grad=True,
    )

    sample = build_altruistic_rollout_sample(
        possible_agents=(
            "a",
        ),
        module_ids_by_row=(
            "m0",
        ),
        owner_to_module_id={
            "a": "m0",
        },
        module_positions_by_id={
            "m0": torch.tensor(
                [0.0, 0.0, 0.0]
            ),
        },
        owner_rewards={
            "a": reward,
        },
        radius_m=1.0,
    )

    assert (
        sample.owner_rewards[
            "a"
        ].requires_grad
        is False
    )

    reward.data.fill_(
        99.0
    )

    torch.testing.assert_close(
        sample.owner_rewards[
            "a"
        ],
        torch.tensor(
            3.0
        ),
    )


def test_vectorized_builder_preserves_env_alignment():
    from drl.MARL.altruistic_rollout import (
        build_altruistic_rollout_batch,
    )

    samples = build_altruistic_rollout_batch(
        possible_agents=(
            "a",
            "b",
        ),
        module_ids_by_row=(
            (
                "env0_b",
                "env0_a",
            ),
            (
                "env1_a",
                "env1_b",
            ),
        ),
        owner_to_module_id=(
            {
                "a": "env0_a",
                "b": "env0_b",
            },
            {
                "a": "env1_a",
                "b": "env1_b",
            },
        ),
        module_positions_by_id=(
            {
                "env0_a": torch.tensor(
                    [0.0, 0.0, 0.0]
                ),
                "env0_b": torch.tensor(
                    [0.2, 0.0, 0.0]
                ),
            },
            {
                "env1_a": torch.tensor(
                    [0.0, 0.0, 0.0]
                ),
                "env1_b": torch.tensor(
                    [2.0, 0.0, 0.0]
                ),
            },
        ),
        owner_rewards={
            "a": torch.tensor(
                [
                    [1.0],
                    [10.0],
                ]
            ),
            "b": torch.tensor(
                [
                    [3.0],
                    [30.0],
                ]
            ),
        },
        radius_m=0.5,
    )

    assert len(
        samples
    ) == 2

    assert samples[
        0
    ].owner_row_index == {
        "a": 1,
        "b": 0,
    }

    assert samples[
        1
    ].owner_row_index == {
        "a": 0,
        "b": 1,
    }

    assert samples[
        0
    ].neighbourhoods[
        "a"
    ] == (
        "b",
    )

    assert samples[
        1
    ].neighbourhoods[
        "a"
    ] == ()

    torch.testing.assert_close(
        samples[
            0
        ].mean_field_rewards[
            "a"
        ],
        torch.tensor(
            3.0
        ),
    )

    torch.testing.assert_close(
        samples[
            1
        ].mean_field_rewards[
            "a"
        ],
        torch.tensor(
            0.0
        ),
    )


def test_sidecar_keeps_time_major_env_major_alignment():
    from drl.MARL.altruistic_rollout import (
        AltruisticRolloutSidecar,
        build_altruistic_rollout_batch,
    )

    def batch(
        first,
        second,
    ):
        return build_altruistic_rollout_batch(
            possible_agents=(
                "a",
            ),
            module_ids_by_row=(
                (
                    "m0",
                ),
                (
                    "m1",
                ),
            ),
            owner_to_module_id=(
                {
                    "a": "m0",
                },
                {
                    "a": "m1",
                },
            ),
            module_positions_by_id=(
                {
                    "m0": torch.zeros(
                        3
                    ),
                },
                {
                    "m1": torch.zeros(
                        3
                    ),
                },
            ),
            owner_rewards={
                "a": torch.tensor(
                    [
                        [first],
                        [second],
                    ],
                    dtype=torch.float32,
                ),
            },
            radius_m=1.0,
        )

    sidecar = (
        AltruisticRolloutSidecar()
    )

    sidecar.append_step(
        batch(
            1.0,
            2.0,
        )
    )

    sidecar.append_step(
        batch(
            3.0,
            4.0,
        )
    )

    flat = sidecar.flatten()

    assert len(
        flat
    ) == 4

    values = [
        float(
            sample.owner_rewards[
                "a"
            ].item()
        )
        for sample in flat
    ]

    assert values == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]


def test_sidecar_requires_same_env_count_for_every_step():
    from drl.MARL.altruistic_rollout import (
        AltruisticRolloutSidecar,
        build_altruistic_rollout_batch,
    )

    sidecar = (
        AltruisticRolloutSidecar()
    )

    common = dict(
        possible_agents=(
            "a",
        ),
        radius_m=1.0,
    )

    one = build_altruistic_rollout_batch(
        **common,
        module_ids_by_row=(
            (
                "m0",
            ),
        ),
        owner_to_module_id=(
            {
                "a": "m0",
            },
        ),
        module_positions_by_id=(
            {
                "m0": torch.zeros(
                    3
                ),
            },
        ),
        owner_rewards={
            "a": torch.tensor(
                [[1.0]]
            ),
        },
    )

    two = build_altruistic_rollout_batch(
        **common,
        module_ids_by_row=(
            (
                "m0",
            ),
            (
                "m1",
            ),
        ),
        owner_to_module_id=(
            {
                "a": "m0",
            },
            {
                "a": "m1",
            },
        ),
        module_positions_by_id=(
            {
                "m0": torch.zeros(
                    3
                ),
            },
            {
                "m1": torch.zeros(
                    3
                ),
            },
        ),
        owner_rewards={
            "a": torch.tensor(
                [
                    [1.0],
                    [2.0],
                ]
            ),
        },
    )

    sidecar.append_step(
        one
    )

    with pytest.raises(
        ValueError,
        match="environment count",
    ):
        sidecar.append_step(
            two
        )


@pytest.mark.parametrize(
    "module_ids",
    (
        (
            "m0",
            "m0",
        ),
        (),
    ),
)
def test_sample_rejects_invalid_graph_module_row_identity(
    module_ids,
):
    from drl.MARL.altruistic_rollout import (
        build_altruistic_rollout_sample,
    )

    with pytest.raises(
        ValueError
    ):
        build_altruistic_rollout_sample(
            possible_agents=(
                "a",
            ),
            module_ids_by_row=module_ids,
            owner_to_module_id={
                "a": "m0",
            },
            module_positions_by_id={
                "m0": torch.zeros(
                    3
                ),
            },
            owner_rewards={
                "a": torch.tensor(
                    1.0
                ),
            },
            radius_m=1.0,
        )


def test_sample_rejects_missing_or_duplicate_owner_module_binding():
    from drl.MARL.altruistic_rollout import (
        build_altruistic_rollout_sample,
    )

    base = dict(
        possible_agents=(
            "a",
            "b",
        ),
        module_ids_by_row=(
            "m0",
            "m1",
        ),
        module_positions_by_id={
            "m0": torch.zeros(
                3
            ),
            "m1": torch.ones(
                3
            ),
        },
        owner_rewards={
            "a": torch.tensor(
                1.0
            ),
            "b": torch.tensor(
                2.0
            ),
        },
        radius_m=1.0,
    )

    with pytest.raises(
        ValueError
    ):
        build_altruistic_rollout_sample(
            **base,
            owner_to_module_id={
                "a": "m0",
            },
        )

    with pytest.raises(
        ValueError
    ):
        build_altruistic_rollout_sample(
            **base,
            owner_to_module_id={
                "a": "m0",
                "b": "m0",
            },
        )


def test_sample_rejects_missing_reward_or_position():
    from drl.MARL.altruistic_rollout import (
        build_altruistic_rollout_sample,
    )

    with pytest.raises(
        ValueError
    ):
        build_altruistic_rollout_sample(
            possible_agents=(
                "a",
            ),
            module_ids_by_row=(
                "m0",
            ),
            owner_to_module_id={
                "a": "m0",
            },
            module_positions_by_id={},
            owner_rewards={
                "a": torch.tensor(
                    1.0
                ),
            },
            radius_m=1.0,
        )

    with pytest.raises(
        ValueError
    ):
        build_altruistic_rollout_sample(
            possible_agents=(
                "a",
            ),
            module_ids_by_row=(
                "m0",
            ),
            owner_to_module_id={
                "a": "m0",
            },
            module_positions_by_id={
                "m0": torch.zeros(
                    3
                ),
            },
            owner_rewards={},
            radius_m=1.0,
        )
