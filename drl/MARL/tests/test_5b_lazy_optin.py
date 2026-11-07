import torch

from drl.MARL.mappo import MSSRMAPPO


def test_mappo_forwards_lazy_capability():
    class Actor:
        prefer_lazy_payloads = True

        def forward_all_owners_batch(
            self, *, graphs, routings, possible_agents,
            lazy_payloads=False
        ):
            assert lazy_payloads is True
            return [{"lazy": lazy_payloads} for _ in graphs]

    result = MSSRMAPPO.forward_shared_actor_batch(
        actor=Actor(),
        graphs=[{"env": 0}, {"env": 1}],
        routings=[{}, {}],
        possible_agents=["module_0"],
    )

    assert len(result) == 2
    assert all(row["lazy"] for row in result)


def test_real_bc_v4_batch_uses_lazy_payloads():
    from drl.IL.tests.bc_v1.g2_fixtures import fixture, tensorize
    from drl.IL.bc_v1.action_set_head import (
        ActionSetPolicy,
        load_action_set_head_config,
    )
    from drl.IL.bc_v1.model import load_model_config
    from drl.MARL.bc_v4_actor import BCV4SharedActorWrapper
    from drl.MARL.runtime_payloads import LazyPayloadMapping

    sample, kwargs = fixture(2, 2, attached=False)
    prepared = tensorize(sample, kwargs)

    torch.manual_seed(123)

    policy = ActionSetPolicy(
        kwargs["spec"],
        kwargs["feature_schema"],
        load_model_config("configs/bc_v1/gnn_v1.json"),
        load_action_set_head_config(
            "configs/bc_v1/action_set_head_v1.json"
        ),
    ).eval()

    actor = BCV4SharedActorWrapper(
        policy=policy,
        device="cpu",
    ).eval()

    assert actor.prefer_lazy_payloads is True

    graphs = [prepared["inputs"]]
    routings = [prepared["routing"]]

    output = MSSRMAPPO.forward_shared_actor_batch(
        actor=actor,
        graphs=graphs,
        routings=routings,
        possible_agents=["module_0"],
    )[0]

    assert isinstance(
        output["resource_payloads"], LazyPayloadMapping
    )
    assert isinstance(
        output["structural_payloads"], LazyPayloadMapping
    )

    eager = actor.forward_all_owners_batch(
        graphs=graphs,
        routings=routings,
        possible_agents=["module_0"],
        lazy_payloads=False,
    )[0]

    assert isinstance(eager["resource_payloads"], dict)
    assert isinstance(eager["structural_payloads"], dict)

    torch.testing.assert_close(
        eager["prediction_template"]["presence_logits"],
        output["prediction_template"]["presence_logits"],
    )

    assert len(eager["records"]) == len(output["records"])
