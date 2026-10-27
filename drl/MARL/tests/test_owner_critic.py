from __future__ import annotations

import pytest
import torch
from skrl.models.torch import Model


class FakeGraphEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Parameter included intentionally so that critic-pair tests verify
        # encoder independence, not only independence of the value MLP.
        self.scale = torch.nn.Parameter(
            torch.tensor(1.0)
        )

    def encode(self, graph):
        return {
            "module_h": (
                graph["module_latents"]
                * self.scale
            ),
        }


def make_graph(
    module_latents,
    graph_ptr=None,
):
    module_latents = torch.as_tensor(
        module_latents,
        dtype=torch.float32,
    )

    if graph_ptr is None:
        graph_ptr = [
            0,
            len(module_latents),
        ]

    return {
        "module_latents": module_latents,
        "graph_ptr": torch.tensor(
            graph_ptr,
            dtype=torch.long,
        ),
    }


@pytest.mark.parametrize(
    "n_modules",
    (
        1,
        3,
        8,
        11,
    ),
)
def test_owner_critic_returns_one_value_per_module_and_supports_variable_n(
    n_modules,
):
    from drl.MARL.owner_critic import (
        OwnerConditionedCritic,
    )

    critic = OwnerConditionedCritic(
        encoder=FakeGraphEncoder(),
        module_latent_dim=4,
        value_hidden_dim=8,
        device="cpu",
    )

    graph = make_graph(
        torch.arange(
            n_modules * 4,
            dtype=torch.float32,
        ).reshape(
            n_modules,
            4,
        )
        / 10.0
    )

    values = critic.forward_all_owners(
        graph
    )

    assert values.shape == (
        n_modules,
        1,
    )

    assert torch.isfinite(
        values
    ).all()


def test_owner_critic_is_permutation_equivariant_when_owner_rows_are_permuted():
    from drl.MARL.owner_critic import (
        OwnerConditionedCritic,
    )

    torch.manual_seed(
        2026
    )

    critic = OwnerConditionedCritic(
        encoder=FakeGraphEncoder(),
        module_latent_dim=3,
        value_hidden_dim=8,
        device="cpu",
    ).eval()

    modules = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 0.5, 2.0],
            [-1.0, 7.0, 1.0],
            [2.5, -2.0, 6.0],
        ]
    )

    permutation = torch.tensor(
        [
            2,
            0,
            3,
            1,
        ]
    )

    original = make_graph(
        modules
    )

    permuted = make_graph(
        modules[
            permutation
        ]
    )

    with torch.no_grad():
        original_values = (
            critic.forward_all_owners(
                original
            )
        )

        permuted_values = (
            critic.forward_all_owners(
                permuted
            )
        )

    # The global context is permutation invariant; only the owner row moves.
    torch.testing.assert_close(
        permuted_values,
        original_values[
            permutation
        ],
        rtol=1e-6,
        atol=1e-6,
    )


def test_owner_critic_handles_multiple_graphs_without_cross_graph_pooling():
    from drl.MARL.owner_critic import (
        OwnerConditionedCritic,
    )

    critic = OwnerConditionedCritic(
        encoder=FakeGraphEncoder(),
        module_latent_dim=2,
        value_hidden_dim=8,
        device="cpu",
    )

    graph = make_graph(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [10.0, 1.0],
            [20.0, 1.0],
            [30.0, 1.0],
        ],
        graph_ptr=[
            0,
            2,
            5,
        ],
    )

    values = critic.forward_all_owners(
        graph
    )

    assert values.shape == (
        5,
        1,
    )

    # Compare against the same graphs evaluated separately. This catches
    # accidental pooling across graph boundaries.
    first = critic.forward_all_owners(
        make_graph(
            [
                [1.0, 0.0],
                [2.0, 0.0],
            ]
        )
    )

    second = critic.forward_all_owners(
        make_graph(
            [
                [10.0, 1.0],
                [20.0, 1.0],
                [30.0, 1.0],
            ]
        )
    )

    torch.testing.assert_close(
        values,
        torch.cat(
            (
                first,
                second,
            ),
            dim=0,
        ),
    )


def test_forward_owner_selects_local_owner_index_in_each_graph_segment():
    from drl.MARL.owner_critic import (
        OwnerConditionedCritic,
    )

    critic = OwnerConditionedCritic(
        encoder=FakeGraphEncoder(),
        module_latent_dim=2,
        value_hidden_dim=8,
        device="cpu",
    )

    graph = make_graph(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [10.0, 1.0],
            [20.0, 1.0],
            [30.0, 1.0],
        ],
        graph_ptr=[
            0,
            2,
            5,
        ],
    )

    all_values = (
        critic.forward_all_owners(
            graph
        )
    )

    selected = critic.forward_owner(
        graph,
        owner_index=torch.tensor(
            [
                1,
                2,
            ],
            dtype=torch.long,
        ),
    )

    torch.testing.assert_close(
        selected,
        torch.stack(
            (
                all_values[
                    1
                ],
                all_values[
                    4
                ],
            )
        ),
    )


def test_forward_owner_rejects_out_of_range_local_index():
    from drl.MARL.owner_critic import (
        OwnerConditionedCritic,
    )

    critic = OwnerConditionedCritic(
        encoder=FakeGraphEncoder(),
        module_latent_dim=2,
        value_hidden_dim=4,
        device="cpu",
    )

    graph = make_graph(
        [
            [1.0, 0.0],
            [2.0, 0.0],
        ]
    )

    with pytest.raises(
        ValueError,
        match="owner_index",
    ):
        critic.forward_owner(
            graph,
            owner_index=2,
        )


def test_owner_critic_rejects_empty_graph_segment():
    from drl.MARL.owner_critic import (
        OwnerConditionedCritic,
    )

    critic = OwnerConditionedCritic(
        encoder=FakeGraphEncoder(),
        module_latent_dim=2,
        value_hidden_dim=4,
        device="cpu",
    )

    graph = make_graph(
        torch.empty(
            (
                1,
                2,
            )
        ),
        graph_ptr=[
            0,
            0,
            1,
        ],
    )

    with pytest.raises(
        ValueError,
        match="empty module set",
    ):
        critic.forward_all_owners(
            graph
        )


def test_owner_critic_rejects_malformed_graph_ptr():
    from drl.MARL.owner_critic import (
        OwnerConditionedCritic,
    )

    critic = OwnerConditionedCritic(
        encoder=FakeGraphEncoder(),
        module_latent_dim=2,
        value_hidden_dim=4,
        device="cpu",
    )

    graph = make_graph(
        [
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        graph_ptr=[
            0,
            1,
            3,
        ],
    )

    with pytest.raises(
        ValueError,
        match="graph_ptr",
    ):
        critic.forward_all_owners(
            graph
        )


def test_owner_critic_satisfies_skrl_model_contract():
    from drl.MARL.owner_critic import (
        OwnerConditionedCritic,
    )

    critic = OwnerConditionedCritic(
        encoder=FakeGraphEncoder(),
        module_latent_dim=2,
        value_hidden_dim=4,
        device="cpu",
    )

    assert isinstance(
        critic,
        Model,
    )

    graph = make_graph(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    compute_values, compute_extra = (
        critic.compute(
            {
                "states": graph,
                "owner_index": torch.tensor(
                    [1],
                    dtype=torch.long,
                ),
            },
            role="value",
        )
    )

    act_values, act_extra = (
        critic.act(
            {
                "states": graph,
                "owner_index": torch.tensor(
                    [1],
                    dtype=torch.long,
                ),
            },
            role="value",
        )
    )

    assert compute_values.shape == (
        1,
        1,
    )

    assert act_values.shape == (
        1,
        1,
    )

    torch.testing.assert_close(
        compute_values,
        act_values,
    )

    assert compute_extra == {}
    assert act_extra == {}


def test_owner_critic_parameter_shapes_do_not_depend_on_module_count():
    from drl.MARL.owner_critic import (
        OwnerConditionedCritic,
    )

    critic = OwnerConditionedCritic(
        encoder=FakeGraphEncoder(),
        module_latent_dim=3,
        value_hidden_dim=7,
        device="cpu",
    )

    before = {
        name: tuple(
            parameter.shape
        )
        for name, parameter
        in critic.named_parameters()
    }

    critic.forward_all_owners(
        make_graph(
            torch.randn(
                3,
                3,
            )
        )
    )

    critic.forward_all_owners(
        make_graph(
            torch.randn(
                11,
                3,
            )
        )
    )

    after = {
        name: tuple(
            parameter.shape
        )
        for name, parameter
        in critic.named_parameters()
    }

    assert after == before


def test_altruistic_critic_pair_has_zero_parameter_sharing():
    from drl.MARL.owner_critic import (
        build_altruistic_critic_pair,
    )

    pair = build_altruistic_critic_pair(
        encoder_factory=FakeGraphEncoder,
        module_latent_dim=3,
        value_hidden_dim=8,
        device="cpu",
    )

    owner_ids = {
        id(
            parameter
        )
        for parameter
        in pair.owner.parameters()
    }

    mean_field_ids = {
        id(
            parameter
        )
        for parameter
        in pair.mean_field.parameters()
    }

    assert owner_ids
    assert mean_field_ids

    assert owner_ids.isdisjoint(
        mean_field_ids
    )

    assert (
        pair.owner.encoder
        is not
        pair.mean_field.encoder
    )


def test_altruistic_critic_pair_produces_same_owner_shape_for_both_channels():
    from drl.MARL.owner_critic import (
        build_altruistic_critic_pair,
    )

    pair = build_altruistic_critic_pair(
        encoder_factory=FakeGraphEncoder,
        module_latent_dim=2,
        value_hidden_dim=4,
        device="cpu",
    )

    graph = make_graph(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    owner = (
        pair.owner
        .forward_all_owners(
            graph
        )
    )

    mean_field = (
        pair.mean_field
        .forward_all_owners(
            graph
        )
    )

    assert owner.shape == (
        3,
        1,
    )

    assert mean_field.shape == (
        3,
        1,
    )


# --- REAL REPOSITORY BCGNN FACTORIES ---

def _real_repository_fixture():
    import importlib
    from pathlib import Path

    root = (
        Path(
            __file__
        )
        .resolve()
        .parents[3]
    )

    source = importlib.import_module(
        "drl.IL.tests.bc_v1."
        "test_action_set_candidate_universe"
    )

    kwargs = source.load_sources()[
        "kwargs"
    ]

    real_cases = (
        source.real_cases.__wrapped__()
    )

    _, batch, _ = real_cases[
        0
    ]

    return (
        root,
        source,
        kwargs,
        batch,
    )


def test_real_owner_critic_factory_uses_fresh_bcgnn_and_real_graph():
    import torch

    from drl.IL.bc_v1.model import (
        BCGNN,
    )
    from drl.MARL.owner_critic import (
        OwnerConditionedCritic,
    )

    (
        root,
        _,
        kwargs,
        batch,
    ) = _real_repository_fixture()

    torch.manual_seed(
        10101
    )

    critic = (
        OwnerConditionedCritic
        .from_repository_config(
            spec=kwargs[
                "spec"
            ],
            feature_schema=kwargs[
                "feature_schema"
            ],
            repo_root=root,
            device="cpu",
        )
        .eval()
    )

    assert isinstance(
        critic.encoder,
        BCGNN,
    )

    assert (
        critic.module_latent_dim
        == 128
    )

    with torch.no_grad():
        values = (
            critic.forward_all_owners(
                batch[
                    "inputs"
                ]
            )
        )

    # Real fixture currently contains eight SMORES modules.
    assert values.shape == (
        8,
        1,
    )

    assert torch.isfinite(
        values
    ).all()


def test_repository_altruistic_pair_uses_two_independent_fresh_bcgnns():
    import torch

    from drl.IL.bc_v1.model import (
        BCGNN,
    )
    from drl.MARL.owner_critic import (
        build_repository_altruistic_critic_pair,
    )

    (
        root,
        _,
        kwargs,
        batch,
    ) = _real_repository_fixture()

    torch.manual_seed(
        20261026
    )

    pair = (
        build_repository_altruistic_critic_pair(
            spec=kwargs[
                "spec"
            ],
            feature_schema=kwargs[
                "feature_schema"
            ],
            repo_root=root,
            device="cpu",
        )
    )

    assert isinstance(
        pair.owner.encoder,
        BCGNN,
    )

    assert isinstance(
        pair.mean_field.encoder,
        BCGNN,
    )

    assert (
        pair.owner.encoder
        is not
        pair.mean_field.encoder
    )

    owner_parameter_ids = {
        id(
            parameter
        )
        for parameter
        in pair.owner.parameters()
    }

    mean_field_parameter_ids = {
        id(
            parameter
        )
        for parameter
        in pair.mean_field.parameters()
    }

    assert owner_parameter_ids
    assert mean_field_parameter_ids

    assert (
        owner_parameter_ids
        .isdisjoint(
            mean_field_parameter_ids
        )
    )

    owner_encoder = (
        pair.owner.encoder
        .state_dict()
    )

    mean_field_encoder = (
        pair.mean_field.encoder
        .state_dict()
    )

    common = (
        set(
            owner_encoder
        )
        & set(
            mean_field_encoder
        )
    )

    comparable = [
        name
        for name in common
        if (
            owner_encoder[
                name
            ].dtype.is_floating_point
            and owner_encoder[
                name
            ].shape
            == mean_field_encoder[
                name
            ].shape
        )
    ]

    assert comparable

    # They are independent random initializations, not copies.
    assert any(
        not torch.equal(
            owner_encoder[
                name
            ],
            mean_field_encoder[
                name
            ],
        )
        for name in comparable
    )

    with torch.no_grad():
        owner_values = (
            pair.owner
            .forward_all_owners(
                batch[
                    "inputs"
                ]
            )
        )

        mf_values = (
            pair.mean_field
            .forward_all_owners(
                batch[
                    "inputs"
                ]
            )
        )

    assert owner_values.shape == (
        8,
        1,
    )

    assert mf_values.shape == (
        8,
        1,
    )


def test_repository_altruistic_critics_share_no_parameters_or_weights_with_bc_actor():
    import torch

    from drl.MARL.bc_v4_actor import (
        BCV4SharedActorWrapper,
        BC_V4_REFERENCE_SHA256,
    )
    from drl.MARL.owner_critic import (
        build_repository_altruistic_critic_pair,
    )

    (
        root,
        _,
        kwargs,
        _,
    ) = _real_repository_fixture()

    checkpoint = (
        root
        / "artifacts"
        / "bc_v1"
        / "runs"
        / (
            "20261022T232532_854802073Z_207309_"
            "bc_v4_actor_visible_r1m_scratch_final"
        )
        / "best.pt"
    )

    if not checkpoint.is_file():
        pytest.skip(
            "Real BC-v4 checkpoint unavailable"
        )

    actor = (
        BCV4SharedActorWrapper
        .from_repository_checkpoint(
            spec=kwargs[
                "spec"
            ],
            feature_schema=kwargs[
                "feature_schema"
            ],
            repo_root=root,
            checkpoint_path=checkpoint,
            expected_sha256=(
                BC_V4_REFERENCE_SHA256
            ),
        )
    )

    pair = (
        build_repository_altruistic_critic_pair(
            spec=kwargs[
                "spec"
            ],
            feature_schema=kwargs[
                "feature_schema"
            ],
            repo_root=root,
            device="cpu",
        )
    )

    actor_ids = {
        id(
            parameter
        )
        for parameter
        in actor.parameters()
    }

    owner_ids = {
        id(
            parameter
        )
        for parameter
        in pair.owner.parameters()
    }

    mf_ids = {
        id(
            parameter
        )
        for parameter
        in pair.mean_field.parameters()
    }

    assert actor_ids.isdisjoint(
        owner_ids
    )

    assert actor_ids.isdisjoint(
        mf_ids
    )

    assert owner_ids.isdisjoint(
        mf_ids
    )

    # Stronger warm-start guard:
    # same architectural tensor names may exist, but critic encoder
    # values must not simply equal the trained BC-v4 H1 weights.
    actor_backbone = (
        actor.policy.backbone
        .state_dict()
    )

    for critic_encoder in (
        pair.owner.encoder,
        pair.mean_field.encoder,
    ):
        state = (
            critic_encoder
            .state_dict()
        )

        common = (
            set(
                actor_backbone
            )
            & set(
                state
            )
        )

        comparable = [
            name
            for name in common
            if (
                actor_backbone[
                    name
                ].dtype.is_floating_point
                and actor_backbone[
                    name
                ].shape
                == state[
                    name
                ].shape
            )
        ]

        assert comparable

        assert any(
            not torch.equal(
                actor_backbone[
                    name
                ],
                state[
                    name
                ],
            )
            for name in comparable
        )
