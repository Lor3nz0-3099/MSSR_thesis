import sys
import types

import pytest
import torch


def _ensure_skrl_stub():
    try:
        import skrl.models.torch  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    skrl = types.ModuleType("skrl")
    models = types.ModuleType("skrl.models")
    models_torch = types.ModuleType("skrl.models.torch")

    class Model(torch.nn.Module):
        def __init__(self, observation_space=None, state_space=None, action_space=None, device=None):
            super().__init__()
            self.device = torch.device(device or "cpu")

    models_torch.Model = Model
    skrl.models = models
    models.torch = models_torch
    sys.modules["skrl"] = skrl
    sys.modules["skrl.models"] = models
    sys.modules["skrl.models.torch"] = models_torch


_ensure_skrl_stub()

from drl.IL.tests.bc_v1.g2_fixtures import fixture, tensorize, with_structural
from drl.MARL.bc_v4_actor import BCV4SharedActorWrapper
import drl.MARL.bc_v4_actor as actor_module


def test_forward_all_owners_batch_runs_backbone_once_and_splits_h1_latents(monkeypatch):
    a, ka = fixture(2, 2, attached=True)
    b, kb = fixture(5, 2, geometry=False)
    sa = tensorize(a, ka)
    sb = tensorize(with_structural(b, skill=True), kb)
    graphs = [sa["inputs"], sb["inputs"]]
    routings = [sa["routing"], sb["routing"]]

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def encode(self, inputs):
            self.calls += 1
            assert inputs["graph_ptr"].tolist() == [0, 2, 7]
            device = inputs["module_x"].device
            return {
                "module_h": torch.arange(7, device=device, dtype=torch.float32).reshape(7, 1),
                "connector_h": torch.arange(len(inputs["connector_x"]), device=device, dtype=torch.float32).reshape(-1, 1),
                "resource_h": torch.arange(len(inputs["resource_x"]), device=device, dtype=torch.float32).reshape(-1, 1),
                "graph_h": torch.tensor([[10.0], [20.0]], device=device),
                "context_h": torch.tensor([[30.0], [40.0]], device=device),
                "observation_h": torch.tensor([[50.0], [60.0]], device=device),
            }

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, latents, inputs):
            self.calls.append((latents, inputs))
            count = len(inputs["resource_candidates"]["resource"]) + len(inputs["structural"]["initiator"])
            return {
                "presence_logits": torch.zeros(count),
                "parameters": {
                    "resource": {},
                    "structural": {"arguments": {}, "scope": {}},
                },
            }

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = Backbone()
            self.head = Head()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    monkeypatch.setattr(actor_module, "bind_marl_candidates", lambda **kwargs: ())
    monkeypatch.setattr(actor_module, "route_resource_payloads", lambda value: {})
    monkeypatch.setattr(actor_module, "compose_structural_payload_views", lambda **kwargs: {})
    monkeypatch.setattr(actor_module, "map_structural_payloads_to_candidates", lambda **kwargs: {})

    wrapper = BCV4SharedActorWrapper(policy=Policy())
    outputs = wrapper.forward_all_owners_batch(
        graphs=graphs,
        routings=routings,
        possible_agents=[f"module_{i}" for i in range(8)],
    )

    assert len(outputs) == 2
    assert wrapper.policy.backbone.calls == 1
    assert len(wrapper.policy.head.calls) == 2

    first_latents, first_graph = wrapper.policy.head.calls[0]
    second_latents, second_graph = wrapper.policy.head.calls[1]

    assert first_graph is graphs[0]
    assert second_graph is graphs[1]
    assert first_latents["module_h"].flatten().tolist() == [0.0, 1.0]
    assert second_latents["module_h"].flatten().tolist() == [2.0, 3.0, 4.0, 5.0, 6.0]
    assert first_latents["graph_h"].item() == 10.0
    assert second_latents["graph_h"].item() == 20.0


def test_real_action_set_batch_matches_serial_predictions_within_numeric_tolerance():
    from pathlib import Path
    from drl.IL.bc_v1.action_set_head import ActionSetPolicy, load_action_set_head_config
    from drl.IL.bc_v1.model import load_model_config

    root = Path(__file__).resolve().parents[3]
    a, ka = fixture(2, 2, attached=True)
    b, kb = fixture(5, 2, geometry=False)
    sa = tensorize(a, ka)
    sb = tensorize(with_structural(b, skill=True), kb)

    torch.manual_seed(123)
    policy = ActionSetPolicy(
        ka["spec"],
        ka["feature_schema"],
        load_model_config(root / "configs/bc_v1/gnn_v1.json"),
        load_action_set_head_config(root / "configs/bc_v1/action_set_head_v1.json"),
    ).eval()
    wrapper = BCV4SharedActorWrapper(policy=policy)

    serial = [
        wrapper.forward_all_owners(
            graph=sample["inputs"],
            routing=sample["routing"],
            possible_agents=["module_0"],
        )["prediction_template"]
        for sample in (sa, sb)
    ]
    batched = wrapper.forward_all_owners_batch(
        graphs=[sa["inputs"], sb["inputs"]],
        routings=[sa["routing"], sb["routing"]],
        possible_agents=["module_0"],
    )

    def assert_close_tree(left, right):
        if torch.is_tensor(left):
            if left.is_floating_point():
                assert torch.allclose(left, right, atol=1e-5, rtol=1e-5)
            else:
                assert torch.equal(left, right)
        elif isinstance(left, dict):
            assert set(left) == set(right)
            for key in left:
                assert_close_tree(left[key], right[key])
        else:
            assert left == right

    for expected, actual in zip(serial, batched, strict=True):
        assert_close_tree(expected, actual["prediction_template"])
