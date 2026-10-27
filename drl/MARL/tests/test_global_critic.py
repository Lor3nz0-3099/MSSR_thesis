from pathlib import Path
import importlib
import json

import pytest
import torch

from skrl.models.torch import Model


class FakeGraphEncoder(torch.nn.Module):
    def encode(self, graph):
        return {
            "module_h": graph["module_latents"],
        }


def make_graph(module_latents, graph_ptr=None):
    module_latents = torch.as_tensor(
        module_latents,
        dtype=torch.float32,
    )

    if graph_ptr is None:
        graph_ptr = [0, len(module_latents)]

    return {
        "module_latents": module_latents,
        "graph_ptr": torch.tensor(
            graph_ptr,
            dtype=torch.long,
        ),
    }


@pytest.mark.parametrize(
    "n_modules",
    [1, 3, 8, 11],
)
def test_global_critic_supports_variable_n_and_returns_one_value(
    n_modules,
):
    from drl.MARL.global_critic import GlobalCritic

    torch.manual_seed(101)

    critic = GlobalCritic(
        encoder=FakeGraphEncoder(),
        module_latent_dim=4,
        value_hidden_dim=8,
        device="cpu",
    )

    graph = make_graph(
        torch.arange(
            n_modules * 4,
            dtype=torch.float32,
        ).reshape(n_modules, 4)
        / 10.0
    )

    values = critic.forward_global(graph)

    assert values.shape == (1, 1)
    assert torch.isfinite(values).all()


def test_global_critic_handles_multiple_graphs_in_one_batch():
    from drl.MARL.global_critic import GlobalCritic

    critic = GlobalCritic(
        encoder=FakeGraphEncoder(),
        module_latent_dim=3,
        value_hidden_dim=8,
        device="cpu",
    )

    graph = make_graph(
        [
            [1.0, 0.0, 2.0],
            [2.0, 1.0, 0.0],
            [3.0, 0.0, 1.0],
            [4.0, 1.0, 2.0],
            [5.0, 2.0, 3.0],
        ],
        graph_ptr=[0, 2, 5],
    )

    values = critic.forward_global(graph)

    assert values.shape == (2, 1)
    assert torch.isfinite(values).all()


def test_global_critic_mean_max_pooling_is_permutation_invariant():
    from drl.MARL.global_critic import GlobalCritic

    torch.manual_seed(2026)

    critic = GlobalCritic(
        encoder=FakeGraphEncoder(),
        module_latent_dim=3,
        value_hidden_dim=8,
        device="cpu",
    ).eval()

    modules = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 0.5, 2.0],
        [-1.0, 7.0, 1.0],
        [2.5, -2.0, 6.0],
    ])

    original = make_graph(modules)

    permuted = make_graph(
        modules[
            torch.tensor([2, 0, 3, 1])
        ]
    )

    with torch.no_grad():
        value_a = critic.forward_global(original)
        value_b = critic.forward_global(permuted)

    torch.testing.assert_close(
        value_a,
        value_b,
        rtol=1e-6,
        atol=1e-6,
    )


def test_global_critic_rejects_empty_graph_segment():
    from drl.MARL.global_critic import GlobalCritic

    critic = GlobalCritic(
        encoder=FakeGraphEncoder(),
        module_latent_dim=3,
        value_hidden_dim=8,
        device="cpu",
    )

    graph = make_graph(
        torch.empty((0, 3)),
        graph_ptr=[0, 0],
    )

    with pytest.raises(
        ValueError,
        match="empty module set",
    ):
        critic.forward_global(graph)


def test_global_critic_satisfies_skrl_model_contract():
    from drl.MARL.global_critic import GlobalCritic

    critic = GlobalCritic(
        encoder=FakeGraphEncoder(),
        module_latent_dim=2,
        value_hidden_dim=4,
        device="cpu",
    )

    assert isinstance(critic, Model)

    graph = make_graph([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    compute_value, compute_extra = critic.compute(
        {"states": graph},
        role="value",
    )

    act_value, act_extra = critic.act(
        {"states": graph},
        role="value",
    )

    assert compute_value.shape == (1, 1)
    assert act_value.shape == (1, 1)

    torch.testing.assert_close(
        compute_value,
        act_value,
    )

    assert compute_extra == {}
    assert act_extra == {}


def test_real_global_critic_factory_runs_real_bcgnn_graph():
    from drl.MARL.global_critic import GlobalCritic
    from drl.IL.bc_v1.model import BCGNN

    root = Path(__file__).resolve().parents[3]

    source = (
        root
        / "artifacts/bc_v1/sidecar"
        / "g2_v3_resource_contract"
    )

    manifest_path = source / "manifest.json"

    if not manifest_path.is_file():
        pytest.skip(
            "Real BC-v4 sidecar unavailable"
        )

    source_module = importlib.import_module(
        "drl.IL.tests.bc_v1."
        "test_action_set_candidate_universe"
    )

    kwargs = source_module.load_sources()["kwargs"]

    critic = GlobalCritic.from_repository_config(
        spec=kwargs["spec"],
        feature_schema=kwargs["feature_schema"],
        repo_root=root,
        device="cpu",
    ).eval()

    assert isinstance(
        critic.encoder,
        BCGNN,
    )

    manifest = json.loads(
        manifest_path.read_text()
    )

    with source_module.SidecarDataset(
        source,
        "train",
        expected_identity=manifest["identity"],
    ) as dataset:
        row = dataset[0]

    batch = source_module.collate_action_set_samples(
        [row]
    )

    with torch.no_grad():
        values = critic.forward_global(
            batch["inputs"]
        )

    assert values.shape == (1, 1)
    assert torch.isfinite(values).all()


def test_real_actor_and_critic_have_disjoint_parameters_and_weights():
    from drl.MARL.bc_v4_actor import (
        BCV4SharedActorWrapper,
        BC_V4_REFERENCE_SHA256,
    )
    from drl.MARL.global_critic import GlobalCritic

    root = Path(__file__).resolve().parents[3]

    checkpoint_path = (
        root
        / "artifacts/bc_v1/runs"
        / "20261022T232532_854802073Z_207309_"
          "bc_v4_actor_visible_r1m_scratch_final"
        / "best.pt"
    )

    if not checkpoint_path.is_file():
        pytest.skip(
            "Real BC-v4 checkpoint unavailable"
        )

    source_module = importlib.import_module(
        "drl.IL.tests.bc_v1."
        "test_action_set_candidate_universe"
    )

    kwargs = source_module.load_sources()["kwargs"]

    actor = BCV4SharedActorWrapper.from_repository_checkpoint(
        spec=kwargs["spec"],
        feature_schema=kwargs["feature_schema"],
        repo_root=root,
        checkpoint_path=checkpoint_path,
        expected_sha256=BC_V4_REFERENCE_SHA256,
    )

    critic = GlobalCritic.from_repository_config(
        spec=kwargs["spec"],
        feature_schema=kwargs["feature_schema"],
        repo_root=root,
        device="cpu",
    )

    actor_parameter_ids = {
        id(parameter)
        for parameter in actor.parameters()
    }

    critic_parameter_ids = {
        id(parameter)
        for parameter in critic.parameters()
    }

    assert actor_parameter_ids.isdisjoint(
        critic_parameter_ids
    )

    actor_backbone = (
        actor.policy.backbone.state_dict()
    )
    critic_encoder = (
        critic.encoder.state_dict()
    )

    common = (
        set(actor_backbone)
        & set(critic_encoder)
    )

    assert common

    comparable = [
        name
        for name in common
        if (
            actor_backbone[name].dtype.is_floating_point
            and actor_backbone[name].shape
            == critic_encoder[name].shape
        )
    ]

    assert comparable

    assert any(
        not torch.equal(
            actor_backbone[name],
            critic_encoder[name],
        )
        for name in comparable
    )


# --- LIST OF PRIVILEGED GRAPH SNAPSHOTS ---

def test_global_critic_accepts_list_of_graph_snapshots():
    from drl.MARL.global_critic import GlobalCritic

    critic = GlobalCritic(
        encoder=FakeGraphEncoder(),
        module_latent_dim=2,
        value_hidden_dim=4,
        device="cpu",
    )

    graphs = [
        make_graph([
            [1.0, 2.0],
            [3.0, 4.0],
        ]),
        make_graph([
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
        ]),
    ]

    values, extra = critic.act(
        {
            "states": graphs,
        },
        role="value",
    )

    assert values.shape == (2, 1)
    assert torch.isfinite(values).all()
    assert extra == {}
