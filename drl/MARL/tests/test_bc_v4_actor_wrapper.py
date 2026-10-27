import hashlib

import pytest
import torch

from drl.MARL.mappo import MSSRMAPPO


def test_bc_v4_wrapper_strictly_restores_policy_and_keeps_ppo_log_stds_trainable(
    tmp_path,
):
    from drl.MARL.bc_v4_actor import BCV4SharedActorWrapper

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2)

        def forward(self, inputs):
            return self.linear(inputs)

    source = TinyPolicy()

    with torch.no_grad():
        source.linear.weight.copy_(
            torch.tensor([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ])
        )
        source.linear.bias.copy_(
            torch.tensor([7.0, 8.0])
        )

    checkpoint_path = tmp_path / "best.pt"

    torch.save(
        {
            "schema_version": "mssr.bc_v1.checkpoint.action_set.v1",
            "model_contract": "action_set_v1",
            "model": source.state_dict(),
            "identity": {"name": "test"},
            "config": {"initialization": "scratch"},
            "step": 123,
        },
        checkpoint_path,
    )

    expected_sha256 = hashlib.sha256(
        checkpoint_path.read_bytes()
    ).hexdigest()

    target = TinyPolicy()

    with torch.no_grad():
        target.linear.weight.zero_()
        target.linear.bias.zero_()

    wrapper = BCV4SharedActorWrapper.from_checkpoint(
        policy=target,
        checkpoint_path=checkpoint_path,
        expected_sha256=expected_sha256,
    )

    for name, tensor in source.state_dict().items():
        assert torch.equal(
            wrapper.policy.state_dict()[name],
            tensor,
        )

    assert wrapper.checkpoint_sha256 == expected_sha256
    assert wrapper.checkpoint_step == 123

    assert wrapper.resource_log_std.requires_grad
    assert wrapper.structural_log_std.requires_grad

    parameter_ids = {
        id(parameter)
        for parameter in wrapper.parameters()
    }

    # Wrapper optimizer surface includes both BC H1/H2 and new PPO log-std.
    assert id(wrapper.resource_log_std) in parameter_ids
    assert id(wrapper.structural_log_std) in parameter_ids

    for parameter in wrapper.policy.parameters():
        assert id(parameter) in parameter_ids


def test_bc_v4_wrapper_rejects_wrong_checkpoint_hash(tmp_path):
    from drl.MARL.bc_v4_actor import BCV4SharedActorWrapper

    policy = torch.nn.Linear(2, 1)

    checkpoint_path = tmp_path / "best.pt"

    torch.save(
        {
            "schema_version": "mssr.bc_v1.checkpoint.action_set.v1",
            "model_contract": "action_set_v1",
            "model": policy.state_dict(),
        },
        checkpoint_path,
    )

    with pytest.raises(ValueError, match="SHA256"):
        BCV4SharedActorWrapper.from_checkpoint(
            policy=torch.nn.Linear(2, 1),
            checkpoint_path=checkpoint_path,
            expected_sha256="0" * 64,
        )


@pytest.mark.parametrize("n_agents", [1, 3, 8, 11])
def test_bc_v4_wrapper_forward_is_module_count_agnostic(
    monkeypatch,
    n_agents,
):
    import drl.MARL.bc_v4_actor as wrapper_module
    from drl.MARL.bc_v4_actor import BCV4SharedActorWrapper

    possible_agents = [
        f"module_{i}"
        for i in range(n_agents)
    ]

    presence_logits = torch.tensor(
        [0.25, -0.5],
        requires_grad=True,
    )

    resource_parameters = {
        "resource_raw": torch.tensor(1.0),
    }

    structural_arguments = {
        "numeric": {"n": 1},
        "categorical": {"c": 2},
        "boolean": {"b": 3},
        "reference": {"r": 4},
        "reference_set": {"rs": 5},
    }

    structural_scope = {"scope": 6}

    class FakeActionSetPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(
                torch.tensor(1.0)
            )
            self.calls = []

        def forward(self, inputs):
            self.calls.append(inputs)

            return {
                "presence_logits": (
                    presence_logits * self.scale
                ),
                "parameters": {
                    "resource": resource_parameters,
                    "structural": {
                        "arguments": structural_arguments,
                        "scope": structural_scope,
                    },
                },
            }

    policy = FakeActionSetPolicy()
    wrapper = BCV4SharedActorWrapper(policy=policy)

    sentinel_records = ("record_0", "record_1")
    sentinel_resource = {0: "resource_view"}
    sentinel_local_structural = {
        0: "structural_local"
    }
    sentinel_global_structural = {
        1: "structural_global"
    }

    calls = {}

    def fake_bind_marl_candidates(*, routing, presence_logits):
        calls["bind"] = (
            routing,
            presence_logits,
        )
        return sentinel_records

    def fake_route_resource_payloads(
        resource_parameters,
    ):
        calls["resource"] = resource_parameters
        return sentinel_resource

    def fake_compose_structural_payload_views(
        *,
        binding_count,
        numeric,
        categorical,
        boolean,
        reference,
        reference_set,
        scope,
    ):
        calls["structural"] = {
            "binding_count": binding_count,
            "numeric": numeric,
            "categorical": categorical,
            "boolean": boolean,
            "reference": reference,
            "reference_set": reference_set,
            "scope": scope,
        }
        return sentinel_local_structural

    def fake_map_structural_payloads_to_candidates(
        *,
        records,
        structural_payloads,
    ):
        calls["map"] = (
            records,
            structural_payloads,
        )
        return sentinel_global_structural

    monkeypatch.setattr(
        wrapper_module,
        "bind_marl_candidates",
        fake_bind_marl_candidates,
    )
    monkeypatch.setattr(
        wrapper_module,
        "route_resource_payloads",
        fake_route_resource_payloads,
    )
    monkeypatch.setattr(
        wrapper_module,
        "compose_structural_payload_views",
        fake_compose_structural_payload_views,
    )
    monkeypatch.setattr(
        wrapper_module,
        "map_structural_payloads_to_candidates",
        fake_map_structural_payloads_to_candidates,
    )

    graph = {
        "graph_ptr": torch.tensor([0, n_agents]),
        "num_modules": n_agents,
    }

    routing = {
        "resource_candidates": ["r0"],
        "structural_bindings": ["s0"],
    }

    output = wrapper.forward_all_owners(
        graph=graph,
        routing=routing,
        possible_agents=possible_agents,
    )

    assert policy.calls == [graph]

    assert calls["bind"][0] is routing
    assert torch.equal(
        calls["bind"][1],
        presence_logits,
    )

    assert calls["resource"] is resource_parameters

    assert calls["structural"] == {
        "binding_count": 1,
        "numeric": structural_arguments["numeric"],
        "categorical": structural_arguments["categorical"],
        "boolean": structural_arguments["boolean"],
        "reference": structural_arguments["reference"],
        "reference_set": structural_arguments[
            "reference_set"
        ],
        "scope": structural_scope,
    }

    assert calls["map"] == (
        sentinel_records,
        sentinel_local_structural,
    )

    assert output["records"] is sentinel_records
    assert output["resource_payloads"] is sentinel_resource

    assert (
        output["structural_payloads"]
        is sentinel_global_structural
    )

    assert (
        output["resource_log_std"]
        is wrapper.resource_log_std
    )
    assert (
        output["structural_log_std"]
        is wrapper.structural_log_std
    )


def test_mssr_mappo_forwards_graph_and_routing_to_shared_actor():
    calls = []

    class Actor:
        def forward_all_owners(
            self,
            *,
            graph,
            routing,
            possible_agents,
        ):
            calls.append(
                (graph, routing, tuple(possible_agents))
            )
            return {"ok": True}

    graph = {"graph": 1}
    routing = {"routing": 2}

    result = MSSRMAPPO.forward_shared_actor(
        actor=Actor(),
        graph=graph,
        routing=routing,
        possible_agents=["module_0", "module_1"],
    )

    assert result == {"ok": True}

    assert calls == [
        (
            graph,
            routing,
            ("module_0", "module_1"),
        )
    ]


# --- REAL BC-V4 FACTORY INTEGRATION ---

def test_real_bc_v4_factory_builds_and_strictly_restores_action_set_policy():
    from pathlib import Path
    import importlib

    import torch

    from drl.MARL.bc_v4_actor import (
        BCV4SharedActorWrapper,
        BC_V4_REFERENCE_SHA256,
    )
    from drl.IL.bc_v1.action_set_head import ActionSetPolicy

    root = Path(__file__).resolve().parents[3]

    checkpoint_path = (
        root
        / "artifacts/bc_v1/runs"
        / "20261022T232532_854802073Z_207309_"
          "bc_v4_actor_visible_r1m_scratch_final"
        / "best.pt"
    )

    if not checkpoint_path.is_file():
        pytest.skip("Real BC-v4 checkpoint unavailable")

    source_module = importlib.import_module(
        "drl.IL.tests.bc_v1."
        "test_action_set_candidate_universe"
    )

    sources = source_module.load_sources()
    kwargs = sources["kwargs"]

    wrapper = BCV4SharedActorWrapper.from_repository_checkpoint(
        spec=kwargs["spec"],
        feature_schema=kwargs["feature_schema"],
        repo_root=root,
        checkpoint_path=checkpoint_path,
        expected_sha256=BC_V4_REFERENCE_SHA256,
    )

    assert isinstance(
        wrapper.policy,
        ActionSetPolicy,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    assert len(checkpoint["model"]) == 260

    actual_state = wrapper.policy.state_dict()

    assert set(actual_state) == set(
        checkpoint["model"]
    )

    for name, expected in checkpoint["model"].items():
        assert torch.equal(
            actual_state[name],
            expected,
        )

    assert (
        wrapper.robot_spec_fingerprint
        == checkpoint["identity"][
            "robot_spec_fingerprint"
        ]
    )

    assert (
        wrapper.feature_schema_fingerprint
        == checkpoint["identity"][
            "feature_schema_fingerprint"
        ]
    )


def test_real_bc_v4_wrapper_forward_matches_real_sidecar_candidate_universe():
    from pathlib import Path
    import importlib
    import json

    import torch

    from drl.MARL.bc_v4_actor import (
        BCV4SharedActorWrapper,
        BC_V4_REFERENCE_SHA256,
    )

    root = Path(__file__).resolve().parents[3]

    checkpoint_path = (
        root
        / "artifacts/bc_v1/runs"
        / "20261022T232532_854802073Z_207309_"
          "bc_v4_actor_visible_r1m_scratch_final"
        / "best.pt"
    )

    source = (
        root
        / "artifacts/bc_v1/sidecar"
        / "g2_v3_resource_contract"
    )

    manifest_path = source / "manifest.json"

    if (
        not checkpoint_path.is_file()
        or not manifest_path.is_file()
    ):
        pytest.skip(
            "Real BC-v4 checkpoint/sidecar unavailable"
        )

    source_module = importlib.import_module(
        "drl.IL.tests.bc_v1."
        "test_action_set_candidate_universe"
    )

    sources = source_module.load_sources()
    kwargs = sources["kwargs"]

    wrapper = BCV4SharedActorWrapper.from_repository_checkpoint(
        spec=kwargs["spec"],
        feature_schema=kwargs["feature_schema"],
        repo_root=root,
        checkpoint_path=checkpoint_path,
        expected_sha256=BC_V4_REFERENCE_SHA256,
    ).eval()

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

    inputs = batch["inputs"]
    routing = row["routing"]

    with torch.no_grad():
        direct = wrapper.policy(inputs)

        adapted = wrapper.forward_all_owners(
            graph=inputs,
            routing=routing,
            possible_agents=["unused_runtime_owner_list"],
        )

    expected_candidate_count = (
        len(routing["resource_candidates"])
        + len(routing["structural_bindings"])
    )

    assert len(adapted["records"]) == (
        expected_candidate_count
    )

    adapted_presence = torch.stack([
        record.presence_logit
        for record in adapted["records"]
    ])

    torch.testing.assert_close(
        adapted_presence,
        direct["presence_logits"],
    )

    assert len(
        adapted["resource_payloads"]
    ) == len(
        routing["resource_candidates"]
    )

    structural_candidate_indexes = {
        record.candidate_index
        for record in adapted["records"]
        if record.candidate_kind == "structural"
    }

    assert set(
        adapted["structural_payloads"]
    ).issubset(
        structural_candidate_indexes
    )

    assert (
        adapted["resource_log_std"]
        is wrapper.resource_log_std
    )

    assert (
        adapted["structural_log_std"]
        is wrapper.structural_log_std
    )


def test_mssr_mappo_actor_optimizer_includes_real_wrapper_log_stds():
    from pathlib import Path
    import importlib

    from skrl.multi_agents.torch.mappo import MAPPO_CFG
    from drl.MARL.tests.test_mssr_mappo import _MinimalSkrlModel

    from drl.MARL.bc_v4_actor import (
        BCV4SharedActorWrapper,
        BC_V4_REFERENCE_SHA256,
    )

    root = Path(__file__).resolve().parents[3]

    checkpoint_path = (
        root
        / "artifacts/bc_v1/runs"
        / "20261022T232532_854802073Z_207309_"
          "bc_v4_actor_visible_r1m_scratch_final"
        / "best.pt"
    )

    if not checkpoint_path.is_file():
        pytest.skip("Real BC-v4 checkpoint unavailable")

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

    critic = _MinimalSkrlModel.make()

    possible_agents = [
        f"module_{i}"
        for i in range(8)
    ]

    models = MSSRMAPPO.build_shared_model_registry(
        possible_agents=possible_agents,
        actor=actor,
        critic=critic,
    )

    agent = MSSRMAPPO(
        possible_agents=possible_agents,
        models=models,
        memories=None,
        observation_spaces=None,
        state_spaces=None,
        action_spaces=None,
        device="cpu",
        cfg=MAPPO_CFG(
            learning_rate=(1e-4, 3e-4),
            learning_rate_scheduler=None,
            learning_rate_scheduler_kwargs={
                uid: {}
                for uid in possible_agents
            },
            observation_preprocessor_kwargs={
                uid: {}
                for uid in possible_agents
            },
            state_preprocessor_kwargs={
                uid: {}
                for uid in possible_agents
            },
            value_preprocessor_kwargs={
                uid: {}
                for uid in possible_agents
            },
        ),
    )

    actor_optimizer_parameter_ids = {
        id(parameter)
        for group in agent.actor_optimizer.param_groups
        for parameter in group["params"]
    }

    assert id(
        actor.resource_log_std
    ) in actor_optimizer_parameter_ids

    assert id(
        actor.structural_log_std
    ) in actor_optimizer_parameter_ids

    for parameter in actor.policy.parameters():
        assert id(
            parameter
        ) in actor_optimizer_parameter_ids


# --- SKRL MODEL CONTRACT ---

def test_bc_v4_wrapper_satisfies_skrl_model_device_contract():
    import torch
    from skrl.models.torch import Model

    from drl.MARL.bc_v4_actor import BCV4SharedActorWrapper

    policy = torch.nn.Linear(3, 2)

    wrapper = BCV4SharedActorWrapper(
        policy=policy,
        device="cpu",
    )

    assert isinstance(wrapper, Model)
    assert torch.device(wrapper.device).type == "cpu"

    assert all(
        parameter.device.type == "cpu"
        for parameter in wrapper.parameters()
    )

    output, extra = wrapper.compute(
        torch.ones((1, 3)),
        role="policy",
    )

    assert output.shape == (1, 2)
    assert extra == {}


# --- RAW ACTIONSET PREDICTION TEMPLATE CONTRACT ---

def test_forward_all_owners_exposes_same_raw_actionset_prediction(monkeypatch):
    import torch

    import drl.MARL.bc_v4_actor as module

    prediction = {
        "presence_logits": torch.tensor(
            [0.25]
        ),
        "candidate_graph": torch.tensor(
            [0],
            dtype=torch.long,
        ),
        "parameters": {
            "resource": {
                "sentinel": torch.tensor(
                    [1.5]
                ),
            },
            "structural": {
                "arguments": {},
                "scope": {},
            },
        },
    }

    class FakePolicy:
        def __call__(
            self,
            graph,
        ):
            return prediction

    wrapper = object.__new__(
        module.BCV4SharedActorWrapper
    )

    torch.nn.Module.__init__(
        wrapper
    )

    wrapper.policy = FakePolicy()

    wrapper.resource_log_std = (
        torch.nn.Parameter(
            torch.tensor(
                0.0
            )
        )
    )

    wrapper.structural_log_std = (
        torch.nn.Parameter(
            torch.tensor(
                0.0
            )
        )
    )

    # This test isolates only the wrapper ABI. The routing helpers
    # themselves already have their own policy.py tests.
    monkeypatch.setattr(
        module,
        "bind_marl_candidates",
        lambda **kwargs: (),
    )

    monkeypatch.setattr(
        module,
        "route_resource_payloads",
        lambda value: {},
    )

    monkeypatch.setattr(
        module,
        "compose_structural_payload_views",
        lambda **kwargs: {},
    )

    monkeypatch.setattr(
        module,
        "map_structural_payloads_to_candidates",
        lambda **kwargs: {},
    )

    result = wrapper.forward_all_owners(
        graph={
            "actor": True,
        },
        routing={
            "resource_candidates": [],
            "structural_bindings": [],
        },
        possible_agents=[
            "module_0",
        ],
    )

    # Exact raw output from the SAME H1/H2 forward.
    assert (
        result[
            "prediction_template"
        ]
        is prediction
    )
