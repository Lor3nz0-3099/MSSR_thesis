import sys
import types

import torch


def _install_skrl_stubs():
    skrl = types.ModuleType("skrl")
    multi_agents = types.ModuleType("skrl.multi_agents")
    multi_agents_torch = types.ModuleType("skrl.multi_agents.torch")
    base = types.ModuleType("skrl.multi_agents.torch.base")
    mappo_pkg = types.ModuleType("skrl.multi_agents.torch.mappo")
    mappo_mod = types.ModuleType("skrl.multi_agents.torch.mappo.mappo")
    utils = types.ModuleType("skrl.utils")

    class MultiAgent:
        pass

    class MAPPO(MultiAgent):
        pass

    class MAPPO_CFG(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.__dict__.update(kwargs)

    class ScopedTimer:
        def __init__(self, *args, **kwargs):
            self.elapsed_time_ms = 0.0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    base.MultiAgent = MultiAgent
    mappo_pkg.MAPPO = MAPPO
    mappo_pkg.MAPPO_CFG = MAPPO_CFG
    mappo_mod.compute_gae = lambda **kwargs: None
    utils.ScopedTimer = ScopedTimer

    sys.modules.update({
        "skrl": skrl,
        "skrl.multi_agents": multi_agents,
        "skrl.multi_agents.torch": multi_agents_torch,
        "skrl.multi_agents.torch.base": base,
        "skrl.multi_agents.torch.mappo": mappo_pkg,
        "skrl.multi_agents.torch.mappo.mappo": mappo_mod,
        "skrl.utils": utils,
    })


# Do not replace the real skrl package during pytest collection.
# Global stubs contaminate other MARL tests.

from drl.MARL.mappo import MSSRMAPPO


def test_forward_shared_actor_batch_prefers_actor_batch_api():
    class Actor:
        def __init__(self):
            self.batch_calls = 0
            self.serial_calls = 0

        def forward_all_owners_batch(self, *, graphs, routings, possible_agents):
            self.batch_calls += 1
            return [
                {"graph": graph, "routing": routing}
                for graph, routing in zip(graphs, routings, strict=True)
            ]

        def forward_all_owners(self, *, graph, routing, possible_agents):
            self.serial_calls += 1
            return {"graph": graph, "routing": routing}

    actor = Actor()
    outputs = MSSRMAPPO.forward_shared_actor_batch(
        actor=actor,
        graphs=[{"g": 0}, {"g": 1}],
        routings=[{"r": 0}, {"r": 1}],
        possible_agents=["module_0"],
    )

    assert actor.batch_calls == 1
    assert actor.serial_calls == 0
    assert [row["graph"]["g"] for row in outputs] == [0, 1]


def test_act_batches_actor_forward_before_per_env_sampling():
    from types import MethodType
    from drl.MARL.mappo import (
        OnlineRolloutContext,
        OnlineRolloutDecision,
        StructuredRolloutSample,
    )

    class Actor:
        def __init__(self):
            self.batch_calls = 0

        def forward_all_owners_batch(self, *, graphs, routings, possible_agents):
            self.batch_calls += 1
            return [
                {"marker": graph["env"]}
                for graph in graphs
            ]

    class Provider:
        def current_action_contexts(self):
            return [
                OnlineRolloutContext(
                    actor_graph={"env": 0},
                    critic_graph={"env": 0},
                    routing_snapshot={"env": 0},
                    hard_mask=torch.tensor([False]),
                ),
                OnlineRolloutContext(
                    actor_graph={"env": 1},
                    critic_graph={"env": 1},
                    routing_snapshot={"env": 1},
                    hard_mask=torch.tensor([False]),
                ),
            ]

        def encode_env_actions(self, decisions):
            return {
                "module_0": torch.zeros((2, 1)),
            }

    agent = object.__new__(MSSRMAPPO)
    agent.possible_agents = ["module_0"]
    agent.device = torch.device("cpu")
    agent.shared_actor = Actor()
    agent._rollout_context_provider = Provider()
    agent.skip_expert_policy_inference = False
    agent._current_online_decisions = None

    seen_outputs = []

    def fake_build(self, *, actor_graph, critic_graph, routing_snapshot, hard_mask,
                   availability_snapshot=None, actor_output=None):
        seen_outputs.append(actor_output)
        sample = StructuredRolloutSample(
            action=None,
            hard_mask=hard_mask,
            owner_active_mask={"module_0": True},
            routing_snapshot=routing_snapshot,
            graph_snapshot=actor_graph,
            critic_graph_snapshot=critic_graph,
        )
        return OnlineRolloutDecision(
            action=None,
            owner_log_probs={"module_0": torch.tensor([0.0])},
            global_value=torch.tensor([[0.0]]),
            structured_sample=sample,
        )

    agent.build_online_rollout_decision = MethodType(fake_build, agent)

    observations = {"module_0": torch.zeros((2, 1))}
    states = {"module_0": torch.zeros((2, 1))}

    agent.act(observations, states, timestep=0, timesteps=10)

    assert agent.shared_actor.batch_calls == 1
    assert [row["marker"] for row in seen_outputs] == [0, 1]


def test_reevaluate_structured_minibatch_batches_actor_forward(monkeypatch):
    import drl.MARL.mappo as mappo_module
    from drl.MARL.mappo import StructuredRolloutSample

    class Actor:
        def __init__(self):
            self.batch_calls = 0
            self.serial_calls = 0

        @staticmethod
        def _output(marker):
            return {
                "records": [],
                "resource_payloads": {},
                "structural_payloads": {},
                "resource_log_std": torch.tensor(0.0),
                "structural_log_std": torch.tensor(0.0),
                "marker": marker,
            }

        def forward_all_owners_batch(self, *, graphs, routings, possible_agents):
            self.batch_calls += 1
            return [self._output(graph["env"]) for graph in graphs]

        def forward_all_owners(self, *, graph, routing, possible_agents):
            self.serial_calls += 1
            return self._output(graph["env"])

    agent = object.__new__(MSSRMAPPO)
    agent.possible_agents = ["module_0"]
    agent.device = torch.device("cpu")
    agent.shared_actor = Actor()
    agent.skip_expert_policy_inference = False

    monkeypatch.setattr(
        mappo_module,
        "compute_owner_action_log_probs",
        lambda **kwargs: {"module_0": torch.tensor(0.0)},
    )
    monkeypatch.setattr(
        mappo_module,
        "compute_owner_action_entropies",
        lambda **kwargs: {"module_0": torch.tensor(0.0)},
    )

    batch = [
        StructuredRolloutSample(
            action=None,
            hard_mask=torch.zeros(0, dtype=torch.bool),
            owner_active_mask={"module_0": True},
            routing_snapshot={"env": i},
            graph_snapshot={"env": i},
        )
        for i in range(2)
    ]

    agent.reevaluate_structured_minibatch(
        structured_batch=batch,
        owner_tensor_batch={},
        evaluate_global_critic=False,
    )

    assert agent.shared_actor.batch_calls == 1
    assert agent.shared_actor.serial_calls == 0
