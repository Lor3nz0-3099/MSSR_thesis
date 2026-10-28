import pytest
from dataclasses import replace

from drl.MARL.mappo import MSSRMAPPO, StructuredRolloutSample


@pytest.mark.parametrize("n_agents", [1, 3, 8, 11])
def test_shared_model_registry_uses_same_actor_and_critic_for_variable_n(n_agents):
    possible_agents = [f"module_{i}" for i in range(n_agents)]

    actor = object()
    critic = object()

    models = MSSRMAPPO.build_shared_model_registry(
        possible_agents=possible_agents,
        actor=actor,
        critic=critic,
    )

    assert list(models) == possible_agents

    for uid in possible_agents:
        assert set(models[uid]) == {"policy", "value"}
        assert models[uid]["policy"] is actor
        assert models[uid]["value"] is critic

    assert len({id(models[uid]["policy"]) for uid in possible_agents}) == 1
    assert len({id(models[uid]["value"]) for uid in possible_agents}) == 1


class _FakeGlobalActor:
    def __init__(self):
        self.forward_calls = 0
        self.last_graph = None
        self.last_agents = None

    def forward_all_owners(self, *, graph, possible_agents):
        self.forward_calls += 1
        self.last_graph = graph
        self.last_agents = list(possible_agents)

        return {
            uid: {
                "action": f"action:{uid}",
                "log_prob": float(index),
            }
            for index, uid in enumerate(possible_agents)
        }


@pytest.mark.parametrize("n_agents", [1, 3, 8, 11])
def test_shared_actor_runs_one_global_forward_and_returns_outputs_per_owner(n_agents):
    possible_agents = [f"module_{i}" for i in range(n_agents)]
    graph = object()
    actor = _FakeGlobalActor()

    outputs = MSSRMAPPO.forward_shared_actor(
        actor=actor,
        graph=graph,
        possible_agents=possible_agents,
    )

    assert actor.forward_calls == 1
    assert actor.last_graph is graph
    assert actor.last_agents == possible_agents

    assert list(outputs) == possible_agents

    for index, uid in enumerate(possible_agents):
        assert outputs[uid]["action"] == f"action:{uid}"
        assert outputs[uid]["log_prob"] == float(index)


def test_shared_actor_and_global_critic_use_two_separate_optimizers():
    import torch

    actor = torch.nn.Linear(4, 3)
    critic = torch.nn.Linear(6, 1)

    actor_optimizer, critic_optimizer = MSSRMAPPO.build_shared_optimizers(
        actor=actor,
        critic=critic,
        actor_learning_rate=1e-4,
        critic_learning_rate=3e-4,
    )

    assert isinstance(actor_optimizer, torch.optim.Adam)
    assert isinstance(critic_optimizer, torch.optim.Adam)
    assert actor_optimizer is not critic_optimizer

    actor_param_ids = {
        id(parameter)
        for group in actor_optimizer.param_groups
        for parameter in group["params"]
    }
    critic_param_ids = {
        id(parameter)
        for group in critic_optimizer.param_groups
        for parameter in group["params"]
    }

    assert actor_param_ids == {id(parameter) for parameter in actor.parameters()}
    assert critic_param_ids == {id(parameter) for parameter in critic.parameters()}
    assert actor_param_ids.isdisjoint(critic_param_ids)

    assert actor_optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert critic_optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)


def test_coordinated_update_averages_owner_losses_and_steps_each_optimizer_once():
    import torch

    class CountingSGD(torch.optim.SGD):
        def __init__(self, params, *, lr):
            super().__init__(params, lr=lr)
            self.step_calls = 0

        def step(self, closure=None):
            self.step_calls += 1
            return super().step(closure)

    actor_parameter = torch.nn.Parameter(torch.tensor(1.0))
    critic_parameter = torch.nn.Parameter(torch.tensor(1.0))

    actor_optimizer = CountingSGD([actor_parameter], lr=0.1)
    critic_optimizer = CountingSGD([critic_parameter], lr=0.1)

    # Mean policy gradient:
    # mean([1*w, 2*w, 3*w]) = 2*w
    # grad at w=1 is therefore 2.
    owner_policy_losses = {
        "module_0": 1.0 * actor_parameter,
        "module_1": 2.0 * actor_parameter,
        "module_2": 3.0 * actor_parameter,
    }

    # d(v^2)/dv at v=1 is 2.
    value_loss = critic_parameter.square()

    result = MSSRMAPPO.coordinated_optimizer_step(
        owner_policy_losses=owner_policy_losses,
        value_loss=value_loss,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
    )

    assert actor_optimizer.step_calls == 1
    assert critic_optimizer.step_calls == 1

    assert actor_parameter.item() == pytest.approx(0.8)
    assert critic_parameter.item() == pytest.approx(0.8)

    assert result["policy_loss"].item() == pytest.approx(2.0)
    assert result["value_loss"].item() == pytest.approx(1.0)


class _MinimalSkrlModel:
    """Small factory namespace to keep skrl imports local to this test."""

    @staticmethod
    def make():
        import torch
        from skrl.models.torch import Model

        class MinimalModel(Model):
            def __init__(self):
                super().__init__(device="cpu")
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def compute(self, inputs, role):
                return self.weight.reshape(1, 1), {}

            def act(self, inputs, role):
                return self.compute(inputs, role)

        return MinimalModel()


@pytest.mark.parametrize("n_agents", [3, 8])
def test_real_mssr_mappo_initialization_uses_shared_models_and_only_two_optimizers(n_agents):
    from skrl.multi_agents.torch.mappo import MAPPO_CFG

    possible_agents = [f"module_{i}" for i in range(n_agents)]

    actor = _MinimalSkrlModel.make()
    critic = _MinimalSkrlModel.make()

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
            learning_rate_scheduler_kwargs={uid: {} for uid in possible_agents},
            observation_preprocessor_kwargs={uid: {} for uid in possible_agents},
            state_preprocessor_kwargs={uid: {} for uid in possible_agents},
            value_preprocessor_kwargs={uid: {} for uid in possible_agents},
        ),
    )

    assert agent.shared_actor is actor
    assert agent.shared_critic is critic

    for uid in possible_agents:
        assert agent.policies[uid] is actor
        assert agent.values[uid] is critic

    assert agent.actor_optimizer is not agent.critic_optimizer

    actor_optimizer_params = {
        id(parameter)
        for group in agent.actor_optimizer.param_groups
        for parameter in group["params"]
    }
    critic_optimizer_params = {
        id(parameter)
        for group in agent.critic_optimizer.param_groups
        for parameter in group["params"]
    }

    assert actor_optimizer_params == {id(parameter) for parameter in actor.parameters()}
    assert critic_optimizer_params == {id(parameter) for parameter in critic.parameters()}

    # No optimizer must exist once per module/UID.
    assert set(agent.optimizers) == {"actor", "critic"}
    assert agent.optimizers["actor"] is agent.actor_optimizer
    assert agent.optimizers["critic"] is agent.critic_optimizer


@pytest.mark.parametrize("n_agents", [1, 3, 8, 11])
def test_post_interaction_triggers_one_coordinated_update_not_one_update_per_uid(n_agents):
    from skrl.multi_agents.torch.mappo import MAPPO_CFG

    possible_agents = [f"module_{i}" for i in range(n_agents)]

    actor = _MinimalSkrlModel.make()
    critic = _MinimalSkrlModel.make()

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
            rollouts=4,
            learning_starts=0,
            learning_rate=(1e-4, 3e-4),
            learning_rate_scheduler=None,
            learning_rate_scheduler_kwargs={uid: {} for uid in possible_agents},
            observation_preprocessor_kwargs={uid: {} for uid in possible_agents},
            state_preprocessor_kwargs={uid: {} for uid in possible_agents},
            value_preprocessor_kwargs={uid: {} for uid in possible_agents},
        ),
    )

    per_uid_update_calls = []
    coordinated_update_calls = []

    def fake_per_uid_update(*, timestep, timesteps, uid):
        per_uid_update_calls.append(uid)

    def fake_coordinated_update(*, timestep, timesteps):
        coordinated_update_calls.append((timestep, timesteps))

    agent.update = fake_per_uid_update
    agent.coordinated_update = fake_coordinated_update

    agent.training = True

    # One interaction away from a complete rollout.
    agent._rollout = agent.cfg.rollouts - 1

    agent.post_interaction(
        timestep=0,
        timesteps=100,
    )

    assert per_uid_update_calls == []
    assert coordinated_update_calls == [(0, 100)]
    assert agent._rollout == agent.cfg.rollouts


def test_shared_sampling_reuses_identical_indexes_for_all_owner_memories():
    import torch

    class FakeMemory:
        def __init__(self, offset):
            self.offset = offset
            self.sampling_indexes = None
            self.sample_calls = 0
            self.sample_by_index_calls = []

        def __len__(self):
            return 12

        def sample(self, names, *, batch_size, mini_batches):
            self.sample_calls += 1
            self.sampling_indexes = torch.tensor([7, 2, 10, 1, 5, 9])
            return self.sample_by_index(
                names=names,
                indexes=self.sampling_indexes,
                mini_batches=mini_batches,
            )

        def sample_by_index(self, names, *, indexes, mini_batches):
            indexes = torch.as_tensor(indexes).clone()
            self.sample_by_index_calls.append(indexes)

            chunks = torch.tensor_split(indexes, mini_batches)
            return [
                [
                    chunk + self.offset
                    for _ in names
                ]
                for chunk in chunks
            ]

    memories = {
        "module_0": FakeMemory(offset=0),
        "module_1": FakeMemory(offset=100),
        "module_2": FakeMemory(offset=200),
    }

    batches = MSSRMAPPO.sample_aligned_owner_batches(
        memories=memories,
        names=["observations", "actions"],
        batch_size=6,
        mini_batches=2,
    )

    # Only the first owner is allowed to generate random indexes.
    assert memories["module_0"].sample_calls == 1
    assert memories["module_1"].sample_calls == 0
    assert memories["module_2"].sample_calls == 0

    expected_indexes = torch.tensor([7, 2, 10, 1, 5, 9])

    for memory in memories.values():
        assert torch.equal(
            memory.sample_by_index_calls[-1],
            expected_indexes,
        )

    assert set(batches) == {"module_0", "module_1", "module_2"}
    assert len(batches["module_0"]) == 2
    assert len(batches["module_1"]) == 2
    assert len(batches["module_2"]) == 2

    # Same temporal/env samples, different owner-local tensor contents.
    assert torch.equal(
        batches["module_0"][0][0],
        torch.tensor([7, 2, 10]),
    )
    assert torch.equal(
        batches["module_1"][0][0],
        torch.tensor([107, 102, 110]),
    )
    assert torch.equal(
        batches["module_2"][0][0],
        torch.tensor([207, 202, 210]),
    )


def test_team_gae_is_computed_once_and_broadcast_to_all_owner_memories(monkeypatch):
    import torch
    import drl.MARL.mappo as mappo_module

    class FakeMemory:
        def __init__(self):
            self.data = {
                "rewards": torch.tensor([[1.0], [2.0], [3.0]]),
                "terminated": torch.tensor([[False], [False], [True]]),
                "truncated": torch.tensor([[False], [False], [False]]),
                "values": torch.tensor([[0.1], [0.2], [0.3]]),
            }
            self.writes = {}

        def get_tensor_by_name(self, name):
            return self.data[name]

        def set_tensor_by_name(self, name, value):
            self.writes[name] = value.clone()

    memories = {
        "module_0": FakeMemory(),
        "module_1": FakeMemory(),
        "module_2": FakeMemory(),
    }

    compute_calls = []

    expected_returns = torch.tensor([[10.0], [20.0], [30.0]])
    expected_advantages = torch.tensor([[1.0], [0.0], [-1.0]])

    def fake_compute_gae(**kwargs):
        compute_calls.append(kwargs)
        return expected_returns, expected_advantages

    monkeypatch.setattr(
        mappo_module,
        "compute_gae",
        fake_compute_gae,
        raising=False,
    )

    returns, advantages = MSSRMAPPO.compute_and_broadcast_team_gae(
        memories=memories,
        last_values=torch.tensor([[0.5]]),
        discount_factor=0.99,
        lambda_coefficient=0.95,
        time_limit_bootstrap=False,
    )

    assert len(compute_calls) == 1

    assert torch.equal(returns, expected_returns)
    assert torch.equal(advantages, expected_advantages)

    for memory in memories.values():
        assert torch.equal(memory.writes["returns"], expected_returns)
        assert torch.equal(memory.writes["advantages"], expected_advantages)


def test_policy_loss_uses_separate_ppo_ratio_and_clipping_for_each_owner():
    import math
    import torch

    advantages = torch.tensor([[1.0]])

    old_log_probs = {
        "module_0": torch.tensor([[0.0]]),
        "module_1": torch.tensor([[0.0]]),
        "module_2": torch.tensor([[0.0]]),
    }

    # Ratios will be respectively 1.5, 0.5 and 1.0.
    new_log_probs = {
        "module_0": torch.tensor([[math.log(1.5)]]),
        "module_1": torch.tensor([[math.log(0.5)]]),
        "module_2": torch.tensor([[0.0]]),
    }

    losses = MSSRMAPPO.compute_owner_clipped_policy_losses(
        new_log_probs=new_log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        ratio_clip=0.2,
    )

    assert set(losses) == {
        "module_0",
        "module_1",
        "module_2",
    }

    # module_0:
    # ratio=1.5 -> clipped to 1.2 -> loss=-1.2
    assert losses["module_0"].item() == pytest.approx(-1.2)

    # module_1:
    # ratio=0.5, for positive A min(0.5, 0.8)=0.5 -> loss=-0.5
    assert losses["module_1"].item() == pytest.approx(-0.5)

    # module_2:
    # ratio=1.0 -> loss=-1.0
    assert losses["module_2"].item() == pytest.approx(-1.0)


def test_global_value_loss_matches_skrl_clipped_value_objective():
    import torch

    sampled_values = torch.tensor([[1.0], [2.0]])
    sampled_returns = torch.tensor([[2.0], [0.0]])

    # Raw predictions:
    # sample 0: 2.0 -> clipped around old value 1.0 to 1.2
    # sample 1: 1.0 -> clipped around old value 2.0 to 1.8
    predicted_values = torch.tensor([[2.0], [1.0]])

    loss = MSSRMAPPO.compute_global_value_loss(
        predicted_values=predicted_values,
        sampled_values=sampled_values,
        sampled_returns=sampled_returns,
        value_clip=0.2,
        value_loss_scale=2.5,
    )

    clipped_predictions = torch.tensor([[1.2], [1.8]])
    expected_mse = torch.mean(
        (sampled_returns - clipped_predictions) ** 2
    )
    expected_loss = 2.5 * expected_mse

    assert loss.item() == pytest.approx(expected_loss.item())


def test_coordinated_optimizer_step_averages_policy_loss_only_over_active_owners():
    import torch

    class CountingSGD(torch.optim.SGD):
        def __init__(self, params, *, lr):
            super().__init__(params, lr=lr)
            self.step_calls = 0

        def step(self, closure=None):
            self.step_calls += 1
            return super().step(closure)

    actor_parameter = torch.nn.Parameter(torch.tensor(1.0))
    critic_parameter = torch.nn.Parameter(torch.tensor(1.0))

    actor_optimizer = CountingSGD([actor_parameter], lr=0.1)
    critic_optimizer = CountingSGD([critic_parameter], lr=0.1)

    owner_policy_losses = {
        "module_0": 1.0 * actor_parameter,
        "module_1": 2.0 * actor_parameter,
        "module_2": 100.0 * actor_parameter,
    }

    value_loss = critic_parameter.square()

    result = MSSRMAPPO.coordinated_optimizer_step(
        owner_policy_losses=owner_policy_losses,
        active_owners=("module_0", "module_1"),
        value_loss=value_loss,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
    )

    # Active-owner mean:
    # mean([1*w, 2*w]) = 1.5*w
    # grad at w=1 is 1.5 -> SGD(0.1) gives 0.85.
    assert actor_parameter.item() == pytest.approx(0.85)

    # Critic remains one global update:
    # d(v^2)/dv = 2 at v=1 -> 0.8.
    assert critic_parameter.item() == pytest.approx(0.8)

    assert actor_optimizer.step_calls == 1
    assert critic_optimizer.step_calls == 1

    assert result["policy_loss"].item() == pytest.approx(1.5)
    assert result["value_loss"].item() == pytest.approx(1.0)


def test_owner_actor_losses_combine_clipped_ppo_and_entropy_bonus():
    import math
    import torch

    advantages = torch.tensor([[1.0]])

    old_log_probs = {
        "module_0": torch.tensor([[0.0]]),
        "module_1": torch.tensor([[0.0]]),
        "module_2": torch.tensor([[0.0]]),
    }

    # PPO ratios: 1.5, 1.0, 0.5
    new_log_probs = {
        "module_0": torch.tensor([[math.log(1.5)]]),
        "module_1": torch.tensor([[0.0]]),
        "module_2": torch.tensor([[math.log(0.5)]]),
    }

    owner_entropies = {
        "module_0": torch.tensor(0.4),
        "module_1": torch.tensor(0.8),
        "module_2": torch.tensor(1.2),
    }

    losses = MSSRMAPPO.compute_owner_actor_losses(
        new_log_probs=new_log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        owner_entropies=owner_entropies,
        ratio_clip=0.2,
        entropy_loss_scale=0.01,
    )

    # PPO clipped losses:
    # module_0 -> -1.2
    # module_1 -> -1.0
    # module_2 -> -0.5
    #
    # actor loss = PPO loss - entropy_scale * entropy
    assert losses["module_0"].item() == pytest.approx(-1.204)
    assert losses["module_1"].item() == pytest.approx(-1.008)
    assert losses["module_2"].item() == pytest.approx(-0.512)


def test_policy_loss_averages_only_over_active_owner_timesteps():
    import math
    import torch

    advantages = torch.tensor([
        [1.0],
        [4.0],
    ])

    old_log_probs = {
        "module_0": torch.tensor([[0.0], [0.0]]),
        "module_1": torch.tensor([[0.0], [0.0]]),
    }

    new_log_probs = {
        # ratios: 1.5, 0.5
        "module_0": torch.tensor([
            [math.log(1.5)],
            [math.log(0.5)],
        ]),
        # ratios: 1.0, 1.0
        "module_1": torch.tensor([
            [0.0],
            [0.0],
        ]),
    }

    owner_active_masks = {
        # module_0 makes a real stochastic decision only at timestep 0.
        "module_0": torch.tensor([
            [True],
            [False],
        ]),
        # module_1 is active at both timesteps.
        "module_1": torch.tensor([
            [True],
            [True],
        ]),
    }

    losses = MSSRMAPPO.compute_owner_clipped_policy_losses(
        new_log_probs=new_log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        ratio_clip=0.2,
        owner_active_masks=owner_active_masks,
    )

    # module_0:
    # only timestep 0 counts.
    # ratio 1.5 -> clipped to 1.2, A=1 -> loss=-1.2
    assert losses["module_0"].item() == pytest.approx(-1.2)

    # module_1:
    # both timesteps count, ratio=1.
    # mean([-1, -4]) = -2.5
    assert losses["module_1"].item() == pytest.approx(-2.5)


def test_structured_rollout_sidecar_uses_same_flattened_indexes_as_skrl_memory():
    from drl.MARL.mappo import StructuredRolloutSidecar

    sidecar = StructuredRolloutSidecar()

    # Same logical ordering as Memory.tensors_view:
    #
    # timestep 0 -> env 0, env 1
    # timestep 1 -> env 0, env 1
    sidecar.append_step(
        (
            {"sample_id": "t0_e0"},
            {"sample_id": "t0_e1"},
        )
    )
    sidecar.append_step(
        (
            {"sample_id": "t1_e0"},
            {"sample_id": "t1_e1"},
        )
    )

    assert len(sidecar) == 4

    sampled = sidecar.sample_by_index(
        [3, 0, 2]
    )

    assert [sample["sample_id"] for sample in sampled] == [
        "t1_e1",
        "t0_e0",
        "t1_e0",
    ]


def test_structured_rollout_sample_preserves_action_masks_and_routing_snapshot():
    from drl.MARL.mappo import StructuredRolloutSample

    action = object()
    routing_snapshot = {"candidate_ids": ("c0", "c1")}
    graph_snapshot = {"num_nodes": 8}

    sample = StructuredRolloutSample(
        action=action,
        hard_mask=(False, True),
        owner_active_mask={
            "module_0": True,
            "module_1": False,
        },
        routing_snapshot=routing_snapshot,
        graph_snapshot=graph_snapshot,
    )

    assert sample.action is action
    assert sample.hard_mask == (False, True)

    assert sample.owner_active_mask == {
        "module_0": True,
        "module_1": False,
    }

    assert sample.routing_snapshot is routing_snapshot
    assert sample.graph_snapshot is graph_snapshot


def test_structured_rollout_sidecar_can_be_cleared_between_rollouts():
    from drl.MARL.mappo import StructuredRolloutSidecar

    sidecar = StructuredRolloutSidecar()
    sidecar.append_step(
        (
            {"sample_id": "t0_e0"},
            {"sample_id": "t0_e1"},
        )
    )

    assert len(sidecar) == 2

    sidecar.clear()

    assert len(sidecar) == 0

    # New rollout must restart flattened indexing from zero.
    sidecar.append_step(
        (
            {"sample_id": "next_t0_e0"},
            {"sample_id": "next_t0_e1"},
        )
    )

    assert sidecar.sample_by_index([0, 1]) == [
        {"sample_id": "next_t0_e0"},
        {"sample_id": "next_t0_e1"},
    ]


def test_mssr_mappo_initializes_empty_structured_rollout_sidecar():
    from skrl.multi_agents.torch.mappo import MAPPO_CFG
    from drl.MARL.mappo import StructuredRolloutSidecar

    possible_agents = ["module_0", "module_1"]

    actor = _MinimalSkrlModel.make()
    critic = _MinimalSkrlModel.make()

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
            learning_rate_scheduler_kwargs={uid: {} for uid in possible_agents},
            observation_preprocessor_kwargs={uid: {} for uid in possible_agents},
            state_preprocessor_kwargs={uid: {} for uid in possible_agents},
            value_preprocessor_kwargs={uid: {} for uid in possible_agents},
        ),
    )

    assert isinstance(
        agent._structured_rollout_sidecar,
        StructuredRolloutSidecar,
    )
    assert len(agent._structured_rollout_sidecar) == 0


def test_mssr_mappo_records_structured_rollout_step_in_sidecar():
    from skrl.multi_agents.torch.mappo import MAPPO_CFG
    from drl.MARL.mappo import StructuredRolloutSample

    possible_agents = ["module_0", "module_1"]

    actor = _MinimalSkrlModel.make()
    critic = _MinimalSkrlModel.make()

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
            learning_rate_scheduler_kwargs={uid: {} for uid in possible_agents},
            observation_preprocessor_kwargs={uid: {} for uid in possible_agents},
            state_preprocessor_kwargs={uid: {} for uid in possible_agents},
            value_preprocessor_kwargs={uid: {} for uid in possible_agents},
        ),
    )

    samples = (
        StructuredRolloutSample(
            action="action_env0",
            hard_mask=(False,),
            owner_active_mask={"module_0": True, "module_1": False},
            routing_snapshot={"env": 0},
            graph_snapshot={"env": 0},
        ),
        StructuredRolloutSample(
            action="action_env1",
            hard_mask=(True,),
            owner_active_mask={"module_0": False, "module_1": True},
            routing_snapshot={"env": 1},
            graph_snapshot={"env": 1},
        ),
    )

    agent.record_structured_rollout_step(samples)

    assert len(agent._structured_rollout_sidecar) == 2

    sampled = agent._structured_rollout_sidecar.sample_by_index([1, 0])

    assert sampled[0].action == "action_env1"
    assert sampled[1].action == "action_env0"


def test_post_interaction_clears_structured_sidecar_only_after_coordinated_update():
    from skrl.multi_agents.torch.mappo import MAPPO_CFG

    possible_agents = ["module_0"]

    actor = _MinimalSkrlModel.make()
    critic = _MinimalSkrlModel.make()

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
            rollouts=4,
            learning_starts=0,
            learning_rate=(1e-4, 3e-4),
            learning_rate_scheduler=None,
            learning_rate_scheduler_kwargs={uid: {} for uid in possible_agents},
            observation_preprocessor_kwargs={uid: {} for uid in possible_agents},
            state_preprocessor_kwargs={uid: {} for uid in possible_agents},
            value_preprocessor_kwargs={uid: {} for uid in possible_agents},
        ),
    )

    agent.record_structured_rollout_step(
        (
            {"sample_id": "rollout_sample"},
        )
    )

    sidecar_lengths_seen_during_update = []

    def fake_coordinated_update(*, timestep, timesteps):
        sidecar_lengths_seen_during_update.append(
            len(agent._structured_rollout_sidecar)
        )

    agent.coordinated_update = fake_coordinated_update
    agent.training = True
    agent._rollout = agent.cfg.rollouts - 1

    agent.post_interaction(
        timestep=0,
        timesteps=100,
    )

    assert sidecar_lengths_seen_during_update == [1]
    assert len(agent._structured_rollout_sidecar) == 0


def test_structured_rollout_sidecar_splits_same_indexes_into_minibatches():
    from drl.MARL.mappo import StructuredRolloutSidecar

    sidecar = StructuredRolloutSidecar()
    sidecar.append_step(
        (
            {"sample_id": "k0"},
            {"sample_id": "k1"},
        )
    )
    sidecar.append_step(
        (
            {"sample_id": "k2"},
            {"sample_id": "k3"},
        )
    )

    batches = sidecar.sample_minibatches(
        indexes=[3, 0, 2, 1],
        mini_batches=2,
    )

    assert [
        [sample["sample_id"] for sample in batch]
        for batch in batches
    ] == [
        ["k3", "k0"],
        ["k2", "k1"],
    ]


def test_sample_coordinated_batches_aligns_owner_memories_and_structured_sidecar():
    import torch

    from drl.MARL.mappo import StructuredRolloutSidecar

    class FakeMemory:
        def __init__(self, offset):
            self.offset = offset
            self.sampling_indexes = None
            self.sample_by_index_calls = []

        def sample(self, *, names, batch_size, mini_batches):
            assert names == ["log_prob", "advantages"]
            assert batch_size == 4
            assert mini_batches == 2

            self.sampling_indexes = torch.tensor([3, 0, 2, 1])

            return [
                [
                    torch.tensor([[3 + self.offset], [0 + self.offset]]),
                    torch.tensor([[30 + self.offset], [0 + self.offset]]),
                ],
                [
                    torch.tensor([[2 + self.offset], [1 + self.offset]]),
                    torch.tensor([[20 + self.offset], [10 + self.offset]]),
                ],
            ]

        def sample_by_index(self, *, names, indexes, mini_batches):
            self.sample_by_index_calls.append(
                (names, indexes.clone(), mini_batches)
            )

            values = indexes + self.offset
            advantages = indexes * 10 + self.offset

            return [
                [
                    values[:2].unsqueeze(-1),
                    advantages[:2].unsqueeze(-1),
                ],
                [
                    values[2:].unsqueeze(-1),
                    advantages[2:].unsqueeze(-1),
                ],
            ]

    memories = {
        "module_0": FakeMemory(offset=0),
        "module_1": FakeMemory(offset=100),
    }

    sidecar = StructuredRolloutSidecar()
    sidecar.append_step(
        (
            {"sample_id": "k0"},
            {"sample_id": "k1"},
        )
    )
    sidecar.append_step(
        (
            {"sample_id": "k2"},
            {"sample_id": "k3"},
        )
    )

    owner_batches, structured_batches = MSSRMAPPO.sample_coordinated_batches(
        memories=memories,
        sidecar=sidecar,
        names=["log_prob", "advantages"],
        batch_size=4,
        mini_batches=2,
    )

    assert list(owner_batches) == ["module_0", "module_1"]

    assert [
        [sample["sample_id"] for sample in batch]
        for batch in structured_batches
    ] == [
        ["k3", "k0"],
        ["k2", "k1"],
    ]

    assert torch.equal(
        memories["module_1"].sample_by_index_calls[0][1],
        torch.tensor([3, 0, 2, 1]),
    )


# --- MACRO TDD: owner activity masking ---

def test_build_owner_activity_masks_tracks_active_timesteps_and_owners():
    import torch

    from drl.MARL.mappo import MSSRMAPPO, StructuredRolloutSample

    batch = [
        StructuredRolloutSample(
            action=None,
            hard_mask=(),
            owner_active_mask={
                "module_0": True,
                "module_1": False,
                "module_2": True,
            },
            routing_snapshot=None,
            graph_snapshot=None,
        ),
        StructuredRolloutSample(
            action=None,
            hard_mask=(),
            owner_active_mask={
                "module_0": False,
                "module_1": False,
                "module_2": True,
            },
            routing_snapshot=None,
            graph_snapshot=None,
        ),
        StructuredRolloutSample(
            action=None,
            hard_mask=(),
            owner_active_mask={
                "module_0": False,
                "module_1": False,
                "module_2": False,
            },
            routing_snapshot=None,
            graph_snapshot=None,
        ),
    ]

    masks, active_owners = MSSRMAPPO.build_owner_activity_masks(
        structured_batch=batch,
        possible_agents=("module_0", "module_1", "module_2"),
        device="cpu",
    )

    assert torch.equal(
        masks["module_0"],
        torch.tensor([True, False, False]),
    )
    assert torch.equal(
        masks["module_1"],
        torch.tensor([False, False, False]),
    )
    assert torch.equal(
        masks["module_2"],
        torch.tensor([True, True, False]),
    )
    assert active_owners == ("module_0", "module_2")


def test_owner_actor_losses_mask_inactive_timesteps_for_policy_and_entropy():
    import torch

    from drl.MARL.mappo import MSSRMAPPO

    losses = MSSRMAPPO.compute_owner_actor_losses(
        new_log_probs={
            "module_0": torch.zeros((3, 1), requires_grad=True),
        },
        old_log_probs={
            "module_0": torch.zeros((3, 1)),
        },
        advantages=torch.tensor([
            [1.0],
            [3.0],
            [100.0],
        ]),
        owner_entropies={
            "module_0": torch.tensor([
                [0.2],
                [0.4],
                [9.9],
            ]),
        },
        ratio_clip=0.2,
        entropy_loss_scale=0.5,
        owner_active_masks={
            "module_0": torch.tensor([True, True, False]),
        },
    )

    # PPO loss with ratio=1 on active timesteps:
    # mean(-1, -3) = -2
    #
    # active entropy:
    # mean(0.2, 0.4) = 0.3
    #
    # total = -2 - 0.5 * 0.3 = -2.15
    assert torch.allclose(
        losses["module_0"],
        torch.tensor(-2.15),
    )


def test_owner_clipped_policy_losses_skip_owner_with_no_active_timesteps():
    import torch

    from drl.MARL.mappo import MSSRMAPPO

    losses = MSSRMAPPO.compute_owner_clipped_policy_losses(
        new_log_probs={
            "module_0": torch.zeros((2, 1), requires_grad=True),
        },
        old_log_probs={
            "module_0": torch.zeros((2, 1)),
        },
        advantages=torch.tensor([
            [1.0],
            [2.0],
        ]),
        ratio_clip=0.2,
        owner_active_masks={
            "module_0": torch.tensor([False, False]),
        },
    )

    assert losses == {}


# --- MACRO TDD: coordinated update orchestration ---

def test_compute_global_bootstrap_value_uses_shared_critic_once():
    import torch
    from skrl.multi_agents.torch.mappo import MAPPO_CFG

    possible_agents = ["module_0", "module_1"]

    actor = _MinimalSkrlModel.make()
    critic = _MinimalSkrlModel.make()

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
            learning_rate_scheduler_kwargs={uid: {} for uid in possible_agents},
            observation_preprocessor_kwargs={uid: {} for uid in possible_agents},
            state_preprocessor_kwargs={uid: {} for uid in possible_agents},
            value_preprocessor_kwargs={uid: {} for uid in possible_agents},
        ),
    )

    calls = []

    class FakeCritic:
        def act(self, inputs, role):
            calls.append((inputs, role))
            return torch.tensor([[2.5]]), {}

    agent.shared_critic = FakeCritic()

    agent._current_next_observations = {
        "module_0": torch.tensor([[1.0]]),
        "module_1": torch.tensor([[9.0]]),
    }
    agent._current_next_states = {
        "module_0": torch.tensor([[5.0, 6.0]]),
        "module_1": torch.tensor([[99.0, 99.0]]),
    }

    agent._observation_preprocessor = {
        uid: (lambda x: x)
        for uid in possible_agents
    }
    agent._state_preprocessor = {
        uid: (lambda x: x)
        for uid in possible_agents
    }

    def value_preprocessor(x, inverse=False):
        return x + 10.0 if inverse else x

    agent._value_preprocessor = {
        uid: value_preprocessor
        for uid in possible_agents
    }

    value = agent.compute_global_bootstrap_value()

    assert torch.allclose(value, torch.tensor([[12.5]]))
    assert len(calls) == 1
    assert calls[0][1] == "value"
    assert torch.equal(
        calls[0][0]["states"],
        torch.tensor([[5.0, 6.0]]),
    )


def test_coordinated_update_runs_team_gae_once_and_one_joint_step_per_minibatch():
    import torch
    from skrl.multi_agents.torch.mappo import MAPPO_CFG
    from drl.MARL.mappo import StructuredRolloutSample

    possible_agents = ["module_0", "module_1", "module_2"]

    actor = _MinimalSkrlModel.make()
    critic = _MinimalSkrlModel.make()

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
            learning_epochs=2,
            mini_batches=2,
            ratio_clip=0.2,
            value_clip=0.2,
            entropy_loss_scale=0.01,
            value_loss_scale=1.0,
            learning_rate=(1e-4, 3e-4),
            learning_rate_scheduler=None,
            learning_rate_scheduler_kwargs={uid: {} for uid in possible_agents},
            observation_preprocessor_kwargs={uid: {} for uid in possible_agents},
            state_preprocessor_kwargs={uid: {} for uid in possible_agents},
            value_preprocessor_kwargs={uid: {} for uid in possible_agents},
        ),
    )

    class FakeMemory:
        def __len__(self):
            return 4

    agent.memories = {
        uid: FakeMemory()
        for uid in possible_agents
    }

    structured_batches = [
        [
            StructuredRolloutSample(
                action=None,
                hard_mask=(),
                owner_active_mask={
                    "module_0": True,
                    "module_1": True,
                    "module_2": False,
                },
                routing_snapshot=None,
                graph_snapshot=None,
            ),
            StructuredRolloutSample(
                action=None,
                hard_mask=(),
                owner_active_mask={
                    "module_0": True,
                    "module_1": False,
                    "module_2": False,
                },
                routing_snapshot=None,
                graph_snapshot=None,
            ),
        ],
        [
            StructuredRolloutSample(
                action=None,
                hard_mask=(),
                owner_active_mask={
                    "module_0": False,
                    "module_1": True,
                    "module_2": False,
                },
                routing_snapshot=None,
                graph_snapshot=None,
            ),
            StructuredRolloutSample(
                action=None,
                hard_mask=(),
                owner_active_mask={
                    "module_0": True,
                    "module_1": True,
                    "module_2": False,
                },
                routing_snapshot=None,
                graph_snapshot=None,
            ),
        ],
    ]

    def make_owner_batches():
        batches = {}

        for uid in possible_agents:
            batches[uid] = [
                [
                    torch.zeros((2, 2)),       # states
                    torch.zeros((2, 1)),       # old log_prob
                    torch.zeros((2, 1)),       # sampled values
                    torch.ones((2, 1)),        # returns
                    torch.ones((2, 1)),        # advantages
                ],
                [
                    torch.zeros((2, 2)),
                    torch.zeros((2, 1)),
                    torch.zeros((2, 1)),
                    torch.ones((2, 1)),
                    torch.ones((2, 1)),
                ],
            ]

        return batches

    gae_calls = []
    sampling_calls = []
    evaluation_calls = []
    optimizer_calls = []

    agent.compute_global_bootstrap_value = lambda: torch.tensor([[0.0]])

    def fake_gae(**kwargs):
        gae_calls.append(kwargs)
        return torch.zeros((4, 1)), torch.ones((4, 1))

    agent.compute_and_broadcast_team_gae = fake_gae

    def fake_sampling(**kwargs):
        sampling_calls.append(kwargs)
        return make_owner_batches(), structured_batches

    agent.sample_coordinated_batches = fake_sampling

    def fake_evaluation(*, structured_batch, owner_tensor_batch):
        evaluation_calls.append(
            (structured_batch, owner_tensor_batch)
        )

        batch_size = len(structured_batch)

        return (
            {
                uid: torch.zeros(
                    (batch_size, 1),
                    requires_grad=True,
                )
                for uid in possible_agents
            },
            {
                uid: torch.full(
                    (batch_size, 1),
                    0.5,
                )
                for uid in possible_agents
            },
            torch.zeros(
                (batch_size, 1),
                requires_grad=True,
            ),
        )

    agent.reevaluate_structured_minibatch = fake_evaluation

    def fake_optimizer_step(**kwargs):
        optimizer_calls.append(kwargs)
        return {}

    agent.coordinated_optimizer_step = fake_optimizer_step

    agent.coordinated_update(
        timestep=10,
        timesteps=100,
    )

    # GAE is team/global: once per completed rollout.
    assert len(gae_calls) == 1

    # New random/aligned sampling once per learning epoch.
    assert len(sampling_calls) == 2

    # 2 epochs x 2 minibatches.
    assert len(evaluation_calls) == 4

    # Crucially: 4 optimizer steps total, NOT 4 x number_of_agents.
    assert len(optimizer_calls) == 4


def test_coordinated_update_excludes_owner_inactive_for_entire_minibatch():
    import torch
    from skrl.multi_agents.torch.mappo import MAPPO_CFG
    from drl.MARL.mappo import StructuredRolloutSample

    possible_agents = ["module_0", "module_1", "module_2"]

    actor = _MinimalSkrlModel.make()
    critic = _MinimalSkrlModel.make()

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
            learning_epochs=1,
            mini_batches=1,
            ratio_clip=0.2,
            value_clip=0.2,
            entropy_loss_scale=0.01,
            value_loss_scale=1.0,
            learning_rate=(1e-4, 3e-4),
            learning_rate_scheduler=None,
            learning_rate_scheduler_kwargs={uid: {} for uid in possible_agents},
            observation_preprocessor_kwargs={uid: {} for uid in possible_agents},
            state_preprocessor_kwargs={uid: {} for uid in possible_agents},
            value_preprocessor_kwargs={uid: {} for uid in possible_agents},
        ),
    )

    class FakeMemory:
        def __len__(self):
            return 2

    agent.memories = {
        uid: FakeMemory()
        for uid in possible_agents
    }

    structured_batch = [
        StructuredRolloutSample(
            action=None,
            hard_mask=(),
            owner_active_mask={
                "module_0": True,
                "module_1": False,
                "module_2": False,
            },
            routing_snapshot=None,
            graph_snapshot=None,
        ),
        StructuredRolloutSample(
            action=None,
            hard_mask=(),
            owner_active_mask={
                "module_0": False,
                "module_1": True,
                "module_2": False,
            },
            routing_snapshot=None,
            graph_snapshot=None,
        ),
    ]

    owner_batches = {
        uid: [[
            torch.zeros((2, 2)),  # states
            torch.zeros((2, 1)),  # old log_prob
            torch.zeros((2, 1)),  # values
            torch.ones((2, 1)),   # returns
            torch.ones((2, 1)),   # advantages
        ]]
        for uid in possible_agents
    }

    agent.compute_global_bootstrap_value = lambda: torch.tensor([[0.0]])
    agent.compute_and_broadcast_team_gae = (
        lambda **kwargs: (
            torch.zeros((2, 1)),
            torch.ones((2, 1)),
        )
    )
    agent.sample_coordinated_batches = (
        lambda **kwargs: (
            owner_batches,
            [structured_batch],
        )
    )

    agent.reevaluate_structured_minibatch = (
        lambda **kwargs: (
            {
                uid: torch.zeros(
                    (2, 1),
                    requires_grad=True,
                )
                for uid in possible_agents
            },
            {
                uid: torch.ones((2, 1))
                for uid in possible_agents
            },
            torch.zeros(
                (2, 1),
                requires_grad=True,
            ),
        )
    )

    optimizer_calls = []

    def fake_optimizer_step(**kwargs):
        optimizer_calls.append(kwargs)
        return {}

    agent.coordinated_optimizer_step = fake_optimizer_step

    agent.coordinated_update(
        timestep=10,
        timesteps=100,
    )

    assert len(optimizer_calls) == 1

    call = optimizer_calls[0]

    assert call["active_owners"] == (
        "module_0",
        "module_1",
    )

    assert set(call["owner_policy_losses"]) == {
        "module_0",
        "module_1",
    }

    assert "module_2" not in call["owner_policy_losses"]


# --- MACRO TDD: structured actor critic reevaluation ---

def test_reevaluate_structured_minibatch_runs_global_actor_per_graph_and_critic_once():
    import torch
    from skrl.multi_agents.torch.mappo import MAPPO_CFG

    from drl.MARL.mappo import StructuredRolloutSample
    from drl.MARL.policy import (
        MARLActionBundle,
        MARLCandidateRecord,
        ResourcePayloadView,
    )

    possible_agents = ["module_0", "module_1"]

    base_actor = _MinimalSkrlModel.make()
    base_critic = _MinimalSkrlModel.make()

    models = MSSRMAPPO.build_shared_model_registry(
        possible_agents=possible_agents,
        actor=base_actor,
        critic=base_critic,
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
            learning_rate_scheduler_kwargs={uid: {} for uid in possible_agents},
            observation_preprocessor_kwargs={uid: {} for uid in possible_agents},
            state_preprocessor_kwargs={uid: {} for uid in possible_agents},
            value_preprocessor_kwargs={uid: {} for uid in possible_agents},
        ),
    )

    class FakeStructuredActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.presence_logit = torch.nn.Parameter(
                torch.tensor(0.25)
            )
            self.calls = []

        def forward_all_owners(self, *, graph, possible_agents):
            self.calls.append((graph, tuple(possible_agents)))

            records = (
                MARLCandidateRecord(
                    candidate_index=0,
                    candidate_kind="resource",
                    candidate_id="resource|module_0|wheel|drive|velocity",
                    owner_agent_id="module_0",
                    participant_agent_ids=(),
                    presence_logit=self.presence_logit,
                ),
            )

            return {
                "records": records,
                "resource_payloads": {
                    0: ResourcePayloadView(
                        numeric=None,
                        categorical_logits=None,
                    )
                },
                "structural_payloads": {},
                "resource_log_std": torch.tensor(0.0),
                "structural_log_std": torch.tensor(0.0),
            }

    class FakeGlobalCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(2.0))
            self.calls = []

        def act(self, inputs, role):
            self.calls.append((inputs, role))
            return inputs["states"][:, :1] * self.scale, {}

    actor = FakeStructuredActor()
    critic = FakeGlobalCritic()

    agent.shared_actor = actor
    agent.shared_critic = critic

    def make_action():
        return MARLActionBundle(
            presence=torch.tensor([1.0]),
            resource_numeric={},
            resource_categorical={},
            structural_numeric={},
            structural_boolean={},
            structural_categorical={},
            structural_reference={},
            structural_reference_set={},
            structural_scope={},
        )

    structured_batch = [
        StructuredRolloutSample(
            action=make_action(),
            hard_mask=torch.tensor([False]),
            owner_active_mask={
                "module_0": True,
                "module_1": False,
            },
            routing_snapshot={"sample": 0},
            graph_snapshot={"graph": 0},
        ),
        StructuredRolloutSample(
            action=make_action(),
            hard_mask=torch.tensor([False]),
            owner_active_mask={
                "module_0": True,
                "module_1": False,
            },
            routing_snapshot={"sample": 1},
            graph_snapshot={"graph": 1},
        ),
    ]

    states = torch.tensor([
        [1.0, 10.0],
        [3.0, 20.0],
    ])

    owner_tensor_batch = {
        uid: [states]
        for uid in possible_agents
    }

    (
        new_log_probs,
        owner_entropies,
        predicted_values,
    ) = agent.reevaluate_structured_minibatch(
        structured_batch=structured_batch,
        owner_tensor_batch=owner_tensor_batch,
    )

    # H1/H2 global actor forward: exactly once per graph/sample,
    # not once per owner.
    assert len(actor.calls) == 2
    assert actor.calls[0][0] == {"graph": 0}
    assert actor.calls[1][0] == {"graph": 1}

    # Centralized critic evaluates the whole minibatch once.
    assert len(critic.calls) == 1
    assert critic.calls[0][1] == "value"

    assert new_log_probs["module_0"].shape == (2, 1)
    assert new_log_probs["module_1"].shape == (2, 1)

    assert owner_entropies["module_0"].shape == (2, 1)
    assert owner_entropies["module_1"].shape == (2, 1)

    assert torch.allclose(
        predicted_values,
        torch.tensor([
            [2.0],
            [6.0],
        ]),
    )

    # module_1 owns no stochastic candidate in this test.
    assert torch.equal(
        new_log_probs["module_1"],
        torch.zeros((2, 1)),
    )
    assert torch.equal(
        owner_entropies["module_1"],
        torch.zeros((2, 1)),
    )

    # Both actor and critic graphs must remain differentiable.
    loss = (
        new_log_probs["module_0"].sum()
        + owner_entropies["module_0"].sum()
        + predicted_values.sum()
    )
    loss.backward()

    assert actor.presence_logit.grad is not None
    assert torch.isfinite(actor.presence_logit.grad)

    assert critic.scale.grad is not None
    assert torch.isfinite(critic.scale.grad)


def test_coordinated_update_uses_real_structured_reevaluation_path():
    import torch
    from skrl.multi_agents.torch.mappo import MAPPO_CFG

    from drl.MARL.mappo import StructuredRolloutSample
    from drl.MARL.policy import (
        MARLActionBundle,
        MARLCandidateRecord,
        ResourcePayloadView,
    )

    possible_agents = ["module_0", "module_1"]

    base_actor = _MinimalSkrlModel.make()
    base_critic = _MinimalSkrlModel.make()

    models = MSSRMAPPO.build_shared_model_registry(
        possible_agents=possible_agents,
        actor=base_actor,
        critic=base_critic,
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
            learning_epochs=1,
            mini_batches=1,
            ratio_clip=0.2,
            value_clip=0.2,
            entropy_loss_scale=0.01,
            value_loss_scale=1.0,
            learning_rate=(1e-4, 3e-4),
            learning_rate_scheduler=None,
            learning_rate_scheduler_kwargs={uid: {} for uid in possible_agents},
            observation_preprocessor_kwargs={uid: {} for uid in possible_agents},
            state_preprocessor_kwargs={uid: {} for uid in possible_agents},
            value_preprocessor_kwargs={uid: {} for uid in possible_agents},
        ),
    )

    class FakeMemory:
        def __len__(self):
            return 1

    agent.memories = {
        uid: FakeMemory()
        for uid in possible_agents
    }

    class FakeStructuredActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.presence_logit = torch.nn.Parameter(
                torch.tensor(0.1)
            )
            self.calls = 0

        def forward_all_owners(self, *, graph, possible_agents):
            self.calls += 1

            return {
                "records": (
                    MARLCandidateRecord(
                        candidate_index=0,
                        candidate_kind="resource",
                        candidate_id="resource|module_0|wheel|drive|velocity",
                        owner_agent_id="module_0",
                        participant_agent_ids=(),
                        presence_logit=self.presence_logit,
                    ),
                ),
                "resource_payloads": {
                    0: ResourcePayloadView(
                        numeric=None,
                        categorical_logits=None,
                    )
                },
                "structural_payloads": {},
                "resource_log_std": torch.tensor(0.0),
                "structural_log_std": torch.tensor(0.0),
            }

    class FakeGlobalCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.5))
            self.calls = 0

        def act(self, inputs, role):
            self.calls += 1
            return inputs["states"][:, :1] * self.scale, {}

    actor = FakeStructuredActor()
    critic = FakeGlobalCritic()

    agent.shared_actor = actor
    agent.shared_critic = critic

    action = MARLActionBundle(
        presence=torch.tensor([1.0]),
        resource_numeric={},
        resource_categorical={},
        structural_numeric={},
        structural_boolean={},
        structural_categorical={},
        structural_reference={},
        structural_reference_set={},
        structural_scope={},
    )

    structured_batch = [
        StructuredRolloutSample(
            action=action,
            hard_mask=torch.tensor([False]),
            owner_active_mask={
                "module_0": True,
                "module_1": False,
            },
            routing_snapshot={"sample": 0},
            graph_snapshot={"graph": 0},
        )
    ]

    owner_batches = {
        uid: [[
            torch.tensor([[2.0, 7.0]]),  # states
            torch.zeros((1, 1)),         # old log_prob
            torch.zeros((1, 1)),         # sampled values
            torch.ones((1, 1)),          # returns
            torch.ones((1, 1)),          # advantages
        ]]
        for uid in possible_agents
    }

    agent.compute_global_bootstrap_value = (
        lambda: torch.tensor([[0.0]])
    )

    agent.compute_and_broadcast_team_gae = (
        lambda **kwargs: (
            torch.zeros((1, 1)),
            torch.ones((1, 1)),
        )
    )

    agent.sample_coordinated_batches = (
        lambda **kwargs: (
            owner_batches,
            [structured_batch],
        )
    )

    optimizer_calls = []

    def capture_optimizer_step(**kwargs):
        optimizer_calls.append(kwargs)
        return {}

    agent.coordinated_optimizer_step = capture_optimizer_step

    agent.coordinated_update(
        timestep=10,
        timesteps=100,
    )

    assert actor.calls == 1
    assert critic.calls == 1
    assert len(optimizer_calls) == 1

    call = optimizer_calls[0]

    assert call["active_owners"] == ("module_0",)
    assert set(call["owner_policy_losses"]) == {"module_0"}

    assert call["owner_policy_losses"]["module_0"].requires_grad
    assert call["value_loss"].requires_grad


# --- GLOBAL CRITIC PRIVILEGED SNAPSHOT ---

def test_structured_rollout_sidecar_preserves_privileged_critic_graph():
    from drl.MARL.mappo import (
        StructuredRolloutSample,
        StructuredRolloutSidecar,
    )

    actor_graph = {
        "visibility": "actor",
    }

    critic_graph = {
        "visibility": "privileged",
        "critic_only": {
            "stairs.step_count": 4,
        },
    }

    sample = StructuredRolloutSample(
        action=None,
        hard_mask=(),
        owner_active_mask={"module_0": True},
        routing_snapshot=None,
        graph_snapshot=actor_graph,
        critic_graph_snapshot=critic_graph,
    )

    sidecar = StructuredRolloutSidecar()
    sidecar.append_step([sample])

    restored = sidecar.sample_by_index([0])[0]

    assert restored.graph_snapshot is actor_graph
    assert restored.critic_graph_snapshot is critic_graph
    assert (
        restored.critic_graph_snapshot
        is not restored.graph_snapshot
    )


# --- STRUCTURED GLOBAL CRITIC MAPPO INTEGRATION ---

def test_structured_critic_minibatch_uses_privileged_graphs_in_one_act_call():
    import torch

    from drl.MARL.mappo import (
        MSSRMAPPO,
        StructuredRolloutSample,
    )

    class FakeCritic:
        def __init__(self):
            self.calls = []

        def act(self, inputs, *, role=""):
            self.calls.append((inputs, role))

            graphs = inputs["states"]

            return (
                torch.tensor(
                    [
                        [float(graph["value"])]
                        for graph in graphs
                    ],
                    requires_grad=True,
                ),
                {},
            )

    agent = object.__new__(MSSRMAPPO)
    agent.shared_critic = FakeCritic()

    structured_batch = [
        StructuredRolloutSample(
            action=None,
            hard_mask=(),
            owner_active_mask={"module_0": True},
            routing_snapshot=None,
            graph_snapshot={"actor": 1},
            critic_graph_snapshot={"value": 2},
        ),
        StructuredRolloutSample(
            action=None,
            hard_mask=(),
            owner_active_mask={"module_0": True},
            routing_snapshot=None,
            graph_snapshot={"actor": 2},
            critic_graph_snapshot={"value": 5},
        ),
    ]

    values = agent.evaluate_structured_critic_batch(
        structured_batch=structured_batch,
        owner_tensor_batch=None,
    )

    assert len(agent.shared_critic.calls) == 1

    inputs, role = agent.shared_critic.calls[0]

    assert role == "value"

    assert inputs["states"] == [
        {"value": 2},
        {"value": 5},
    ]

    torch.testing.assert_close(
        values,
        torch.tensor([
            [2.0],
            [5.0],
        ]),
    )


def test_global_bootstrap_prefers_explicit_privileged_next_graph():
    import torch
    from types import SimpleNamespace

    from drl.MARL.mappo import MSSRMAPPO

    class FakeCritic:
        def __init__(self):
            self.calls = []

        def act(self, inputs, *, role=""):
            self.calls.append((inputs, role))

            assert inputs["states"] is next_graph

            return (
                torch.tensor([[7.5]]),
                {},
            )

    class IdentityValuePreprocessor:
        def __call__(
            self,
            value,
            inverse=False,
        ):
            assert inverse is True
            return value

    next_graph = {
        "privileged": True,
        "stairs.step_count": 4,
    }

    agent = object.__new__(MSSRMAPPO)

    agent.possible_agents = ["module_0"]
    agent.shared_critic = FakeCritic()
    agent._current_next_critic_graph = next_graph

    agent._value_preprocessor = {
        "module_0": IdentityValuePreprocessor(),
    }

    agent._device_type = "cpu"

    agent.cfg = SimpleNamespace(
        mixed_precision=False,
    )

    values = agent.compute_global_bootstrap_value()

    assert len(agent.shared_critic.calls) == 1

    inputs, role = agent.shared_critic.calls[0]

    assert role == "value"
    assert inputs["states"] is next_graph

    torch.testing.assert_close(
        values,
        torch.tensor([[7.5]]),
    )


def test_set_current_next_critic_graph_updates_bootstrap_state():
    from drl.MARL.mappo import MSSRMAPPO

    agent = object.__new__(MSSRMAPPO)
    agent._current_next_critic_graph = None

    graph = {
        "full_current_state": True,
    }

    agent.set_current_next_critic_graph(
        graph
    )

    assert (
        agent._current_next_critic_graph
        is graph
    )


# --- ONLINE STRUCTURED ROLLOUT ABI ---

def test_online_rollout_decision_keeps_actor_and_critic_views_separate():
    import torch

    from drl.MARL.mappo import (
        MSSRMAPPO,
        OnlineRolloutDecision,
    )
    from drl.MARL.policy import (
        MARLCandidateRecord,
        ResourcePayloadView,
    )

    actor_graph = {
        "view": "actor",
    }

    critic_graph = {
        "view": "privileged",
        "critic_only": {
            "stairs.step_count": 4,
        },
    }

    routing = {
        "routing": "snapshot",
    }

    record = MARLCandidateRecord(
        candidate_index=0,
        candidate_kind="resource",
        candidate_id="resource:module_0:test",
        owner_agent_id="module_0",
        participant_agent_ids=("module_0",),
        presence_logit=torch.tensor(0.0),
    )

    class FakeActor:
        def __init__(self):
            self.calls = []

        def forward_all_owners(
            self,
            *,
            graph,
            possible_agents,
            routing=None,
        ):
            self.calls.append(
                (
                    graph,
                    tuple(possible_agents),
                    routing,
                )
            )

            return {
                "records": [record],
                "resource_payloads": {
                    0: ResourcePayloadView(),
                },
                "structural_payloads": {},
                "resource_log_std": torch.tensor(0.0),
                "structural_log_std": torch.tensor(0.0),
            }

    class FakeCritic:
        def __init__(self):
            self.calls = []

        def act(self, inputs, *, role=""):
            self.calls.append(
                (
                    inputs,
                    role,
                )
            )

            return (
                torch.tensor([[3.25]]),
                {},
            )

    agent = object.__new__(MSSRMAPPO)

    agent.shared_actor = FakeActor()
    agent.shared_critic = FakeCritic()

    agent.possible_agents = [
        "module_0",
        "module_1",
    ]

    decision = agent.build_online_rollout_decision(
        actor_graph=actor_graph,
        critic_graph=critic_graph,
        routing_snapshot=routing,
        hard_mask=torch.tensor(
            [False],
            dtype=torch.bool,
        ),
    )

    assert isinstance(
        decision,
        OnlineRolloutDecision,
    )

    assert len(
        agent.shared_actor.calls
    ) == 1

    seen_actor_graph, seen_agents, seen_routing = (
        agent.shared_actor.calls[0]
    )

    assert seen_actor_graph is actor_graph
    assert seen_routing is routing

    assert seen_agents == (
        "module_0",
        "module_1",
    )

    assert len(
        agent.shared_critic.calls
    ) == 1

    critic_inputs, critic_role = (
        agent.shared_critic.calls[0]
    )

    assert critic_role == "value"

    assert (
        critic_inputs["states"]
        is critic_graph
    )

    assert (
        decision.structured_sample.graph_snapshot
        is actor_graph
    )

    assert (
        decision.structured_sample.critic_graph_snapshot
        is critic_graph
    )

    assert (
        decision.structured_sample.routing_snapshot
        is routing
    )

    assert (
        decision.structured_sample.action
        is decision.action
    )

    assert decision.global_value.shape == (
        1,
        1,
    )

    torch.testing.assert_close(
        decision.global_value,
        torch.tensor([[3.25]]),
    )

    assert set(
        decision.owner_log_probs
    ) == {
        "module_0",
        "module_1",
    }

    assert (
        decision.structured_sample.owner_active_mask
        == {
            "module_0": True,
            "module_1": False,
        }
    )


def test_online_rollout_hard_mask_makes_owner_inactive_and_zero_presence():
    import torch

    from drl.MARL.mappo import MSSRMAPPO
    from drl.MARL.policy import (
        MARLCandidateRecord,
        ResourcePayloadView,
    )

    record = MARLCandidateRecord(
        candidate_index=0,
        candidate_kind="resource",
        candidate_id="resource:module_0:test",
        owner_agent_id="module_0",
        participant_agent_ids=("module_0",),
        presence_logit=torch.tensor(100.0),
    )

    class FakeActor:
        def forward_all_owners(
            self,
            *,
            graph,
            possible_agents,
            routing=None,
        ):
            return {
                "records": [record],
                "resource_payloads": {
                    0: ResourcePayloadView(),
                },
                "structural_payloads": {},
                "resource_log_std": torch.tensor(0.0),
                "structural_log_std": torch.tensor(0.0),
            }

    class FakeCritic:
        def act(self, inputs, *, role=""):
            return (
                torch.tensor([[0.0]]),
                {},
            )

    agent = object.__new__(MSSRMAPPO)

    agent.shared_actor = FakeActor()
    agent.shared_critic = FakeCritic()

    agent.possible_agents = [
        "module_0",
    ]

    decision = agent.build_online_rollout_decision(
        actor_graph={"actor": True},
        critic_graph={"critic": True},
        routing_snapshot=None,
        hard_mask=torch.tensor(
            [True],
            dtype=torch.bool,
        ),
    )

    assert (
        decision.structured_sample.owner_active_mask
        == {
            "module_0": False,
        }
    )

    assert decision.action.presence.tolist() == [
        0.0
    ]

    torch.testing.assert_close(
        decision.owner_log_probs["module_0"],
        torch.zeros_like(
            decision.owner_log_probs["module_0"]
        ),
    )


def test_online_rollout_decision_is_detached_from_interaction_graph():
    import torch

    from drl.MARL.mappo import MSSRMAPPO
    from drl.MARL.policy import (
        MARLCandidateRecord,
        ResourcePayloadView,
    )

    logit = torch.tensor(
        0.5,
        requires_grad=True,
    )

    record = MARLCandidateRecord(
        candidate_index=0,
        candidate_kind="resource",
        candidate_id="resource:module_0:test",
        owner_agent_id="module_0",
        participant_agent_ids=("module_0",),
        presence_logit=logit,
    )

    class FakeActor:
        def forward_all_owners(
            self,
            *,
            graph,
            possible_agents,
            routing=None,
        ):
            return {
                "records": [record],
                "resource_payloads": {
                    0: ResourcePayloadView(),
                },
                "structural_payloads": {},
                "resource_log_std": torch.tensor(
                    0.0,
                    requires_grad=True,
                ),
                "structural_log_std": torch.tensor(
                    0.0,
                    requires_grad=True,
                ),
            }

    class FakeCritic:
        def act(self, inputs, *, role=""):
            return (
                torch.tensor(
                    [[1.0]],
                    requires_grad=True,
                ),
                {},
            )

    agent = object.__new__(MSSRMAPPO)

    agent.shared_actor = FakeActor()
    agent.shared_critic = FakeCritic()

    agent.possible_agents = [
        "module_0",
    ]

    decision = agent.build_online_rollout_decision(
        actor_graph={},
        critic_graph={},
        routing_snapshot=None,
        hard_mask=torch.tensor(
            [False],
            dtype=torch.bool,
        ),
    )

    assert (
        decision.owner_log_probs[
            "module_0"
        ].requires_grad
        is False
    )

    assert (
        decision.global_value.requires_grad
        is False
    )


# --- ONLINE ROLLOUT MEMORY BRIDGE ---

def _make_real_marl_memory(*, num_envs=2):
    import torch
    from skrl.memories.torch import RandomMemory

    memory = RandomMemory(
        memory_size=4,
        num_envs=num_envs,
        device="cpu",
        replacement=False,
    )

    memory.create_tensor(
        name="observations",
        size=3,
        dtype=torch.float32,
    )
    memory.create_tensor(
        name="states",
        size=4,
        dtype=torch.float32,
    )
    memory.create_tensor(
        name="actions",
        size=2,
        dtype=torch.float32,
    )
    memory.create_tensor(
        name="rewards",
        size=1,
        dtype=torch.float32,
    )
    memory.create_tensor(
        name="terminated",
        size=1,
        dtype=torch.bool,
    )
    memory.create_tensor(
        name="truncated",
        size=1,
        dtype=torch.bool,
    )
    memory.create_tensor(
        name="log_prob",
        size=1,
        dtype=torch.float32,
    )
    memory.create_tensor(
        name="values",
        size=1,
        dtype=torch.float32,
    )
    memory.create_tensor(
        name="returns",
        size=1,
        dtype=torch.float32,
    )
    memory.create_tensor(
        name="advantages",
        size=1,
        dtype=torch.float32,
    )

    return memory


def _make_memory_decision(
    *,
    index,
    log_prob_0,
    log_prob_1,
    value,
):
    import torch

    from drl.MARL.mappo import (
        OnlineRolloutDecision,
        StructuredRolloutSample,
    )

    action = {
        "structured_action_index": index,
    }

    sample = StructuredRolloutSample(
        action=action,
        hard_mask=torch.tensor(
            [],
            dtype=torch.bool,
        ),
        owner_active_mask={
            "module_0": True,
            "module_1": True,
        },
        routing_snapshot={
            "env": index,
        },
        graph_snapshot={
            "actor_env": index,
        },
        critic_graph_snapshot={
            "critic_env": index,
        },
    )

    return OnlineRolloutDecision(
        action=action,
        owner_log_probs={
            "module_0": torch.tensor(
                log_prob_0,
            ),
            "module_1": torch.tensor(
                log_prob_1,
            ),
        },
        global_value=torch.tensor(
            [[value]],
        ),
        structured_sample=sample,
    )


def test_record_online_rollout_batch_writes_real_skrl_memory_and_sidecar():
    import torch

    from drl.MARL.mappo import (
        MSSRMAPPO,
        StructuredRolloutSidecar,
    )

    agent = object.__new__(MSSRMAPPO)

    agent.possible_agents = [
        "module_0",
        "module_1",
    ]

    agent.memories = {
        uid: _make_real_marl_memory(
            num_envs=2,
        )
        for uid in agent.possible_agents
    }

    agent._structured_rollout_sidecar = (
        StructuredRolloutSidecar()
    )

    agent._current_next_observations = {}
    agent._current_next_states = {}
    agent._current_next_critic_graph = None

    decisions = [
        _make_memory_decision(
            index=0,
            log_prob_0=0.10,
            log_prob_1=0.20,
            value=1.25,
        ),
        _make_memory_decision(
            index=1,
            log_prob_0=0.30,
            log_prob_1=0.40,
            value=2.50,
        ),
    ]

    observations = {
        "module_0": torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]),
        "module_1": torch.tensor([
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]),
    }

    states = {
        "module_0": torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]),
        "module_1": torch.tensor([
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]),
    }

    # Deliberately non-zero: these must NOT become PPO actions.
    env_actions = {
        "module_0": torch.tensor([
            [9.0, 8.0],
            [7.0, 6.0],
        ]),
        "module_1": torch.tensor([
            [5.0, 4.0],
            [3.0, 2.0],
        ]),
    }

    team_rewards = torch.tensor([
        [1.0],
        [2.0],
    ])

    terminated = torch.tensor([
        [False],
        [True],
    ])

    truncated = torch.tensor([
        [False],
        [False],
    ])

    next_observations = {
        uid: value + 100.0
        for uid, value in observations.items()
    }

    next_states = {
        uid: value + 100.0
        for uid, value in states.items()
    }

    next_critic_graphs = [
        {"next_critic_env": 0},
        {"next_critic_env": 1},
    ]

    agent.record_online_rollout_batch(
        decisions=decisions,
        observations=observations,
        states=states,
        env_actions=env_actions,
        team_rewards=team_rewards,
        next_observations=next_observations,
        next_states=next_states,
        terminated=terminated,
        truncated=truncated,
        next_critic_graphs=next_critic_graphs,
    )

    # One environment-step containing 2 vectorized envs.
    assert len(
        agent.memories["module_0"]
    ) == 2

    assert len(
        agent.memories["module_1"]
    ) == 2

    expected_values = torch.tensor([
        [1.25],
        [2.50],
    ])

    expected_log_prob_0 = torch.tensor([
        [0.10],
        [0.30],
    ])

    expected_log_prob_1 = torch.tensor([
        [0.20],
        [0.40],
    ])

    for uid in agent.possible_agents:
        memory = agent.memories[uid]

        # Structured action is sidecar-only.
        torch.testing.assert_close(
            memory.get_tensor_by_name(
                "actions"
            )[0],
            torch.zeros(
                (2, 2),
            ),
        )

        # Global/team quantities are identical for every owner.
        torch.testing.assert_close(
            memory.get_tensor_by_name(
                "rewards"
            )[0],
            team_rewards,
        )

        torch.testing.assert_close(
            memory.get_tensor_by_name(
                "values"
            )[0],
            expected_values,
        )

        torch.testing.assert_close(
            memory.get_tensor_by_name(
                "terminated"
            )[0],
            terminated,
        )

        torch.testing.assert_close(
            memory.get_tensor_by_name(
                "truncated"
            )[0],
            truncated,
        )

    torch.testing.assert_close(
        agent.memories[
            "module_0"
        ].get_tensor_by_name(
            "log_prob"
        )[0],
        expected_log_prob_0,
    )

    torch.testing.assert_close(
        agent.memories[
            "module_1"
        ].get_tensor_by_name(
            "log_prob"
        )[0],
        expected_log_prob_1,
    )

    assert len(
        agent._structured_rollout_sidecar
    ) == 2

    sidecar_samples = (
        agent._structured_rollout_sidecar
        .sample_by_index([0, 1])
    )

    assert (
        sidecar_samples[0]
        == replace(decisions[0].structured_sample,
            next_critic_graph_snapshot=next_critic_graphs[0])
    )

    assert (
        sidecar_samples[1]
        == replace(decisions[1].structured_sample,
            next_critic_graph_snapshot=next_critic_graphs[1])
    )

    assert (
        agent._current_next_observations
        is next_observations
    )

    assert (
        agent._current_next_states
        is next_states
    )

    assert (
        agent._current_next_critic_graph
        == next_critic_graphs
    )

    assert all(
        stored is original
        for stored, original in zip(
            agent._current_next_critic_graph,
            next_critic_graphs,
        )
    )


def test_record_online_rollout_batch_rejects_tensor_batch_mismatch():
    import pytest
    import torch

    from drl.MARL.mappo import (
        MSSRMAPPO,
        StructuredRolloutSidecar,
    )

    agent = object.__new__(MSSRMAPPO)

    agent.possible_agents = [
        "module_0",
        "module_1",
    ]

    agent.memories = {
        uid: _make_real_marl_memory(
            num_envs=2,
        )
        for uid in agent.possible_agents
    }

    agent._structured_rollout_sidecar = (
        StructuredRolloutSidecar()
    )

    decision = _make_memory_decision(
        index=0,
        log_prob_0=0.1,
        log_prob_1=0.2,
        value=1.0,
    )

    with pytest.raises(
        ValueError,
        match="decision batch",
    ):
        agent.record_online_rollout_batch(
            decisions=[decision],
            observations={
                "module_0": torch.zeros(
                    (2, 3)
                ),
                "module_1": torch.zeros(
                    (2, 3)
                ),
            },
            states={
                "module_0": torch.zeros(
                    (2, 4)
                ),
                "module_1": torch.zeros(
                    (2, 4)
                ),
            },
            env_actions={
                "module_0": torch.zeros(
                    (2, 2)
                ),
                "module_1": torch.zeros(
                    (2, 2)
                ),
            },
            team_rewards=torch.zeros(
                (2, 1)
            ),
            next_observations={
                "module_0": torch.zeros(
                    (2, 3)
                ),
                "module_1": torch.zeros(
                    (2, 3)
                ),
            },
            next_states={
                "module_0": torch.zeros(
                    (2, 4)
                ),
                "module_1": torch.zeros(
                    (2, 4)
                ),
            },
            terminated=torch.zeros(
                (2, 1),
                dtype=torch.bool,
            ),
            truncated=torch.zeros(
                (2, 1),
                dtype=torch.bool,
            ),
            next_critic_graphs=[
                {},
                {},
            ],
        )


def test_record_online_rollout_batch_rejects_missing_owner_logprob():
    import pytest
    import torch

    from drl.MARL.mappo import (
        MSSRMAPPO,
        OnlineRolloutDecision,
        StructuredRolloutSample,
        StructuredRolloutSidecar,
    )

    agent = object.__new__(MSSRMAPPO)

    agent.possible_agents = [
        "module_0",
        "module_1",
    ]

    agent.memories = {
        uid: _make_real_marl_memory(
            num_envs=1,
        )
        for uid in agent.possible_agents
    }

    agent._structured_rollout_sidecar = (
        StructuredRolloutSidecar()
    )

    sample = StructuredRolloutSample(
        action={"bundle": True},
        hard_mask=torch.tensor(
            [],
            dtype=torch.bool,
        ),
        owner_active_mask={
            "module_0": True,
            "module_1": True,
        },
        routing_snapshot=None,
        graph_snapshot={},
        critic_graph_snapshot={},
    )

    decision = OnlineRolloutDecision(
        action=sample.action,
        owner_log_probs={
            "module_0": torch.tensor(0.1),
        },
        global_value=torch.tensor(
            [[1.0]]
        ),
        structured_sample=sample,
    )

    with pytest.raises(
        ValueError,
        match="owner log-prob",
    ):
        agent.record_online_rollout_batch(
            decisions=[decision],
            observations={
                "module_0": torch.zeros(
                    (1, 3)
                ),
                "module_1": torch.zeros(
                    (1, 3)
                ),
            },
            states={
                "module_0": torch.zeros(
                    (1, 4)
                ),
                "module_1": torch.zeros(
                    (1, 4)
                ),
            },
            env_actions={
                "module_0": torch.zeros(
                    (1, 2)
                ),
                "module_1": torch.zeros(
                    (1, 2)
                ),
            },
            team_rewards=torch.zeros(
                (1, 1)
            ),
            next_observations={
                "module_0": torch.zeros(
                    (1, 3)
                ),
                "module_1": torch.zeros(
                    (1, 3)
                ),
            },
            next_states={
                "module_0": torch.zeros(
                    (1, 4)
                ),
                "module_1": torch.zeros(
                    (1, 4)
                ),
            },
            terminated=torch.zeros(
                (1, 1),
                dtype=torch.bool,
            ),
            truncated=torch.zeros(
                (1, 1),
                dtype=torch.bool,
            ),
            next_critic_graphs=[
                {},
            ],
        )


# --- STANDARD SKRL STRUCTURED ROLLOUT BRIDGE ---

def _make_standard_bridge_agent(*, num_envs=2):
    import torch
    from types import SimpleNamespace

    from drl.MARL.mappo import (
        MSSRMAPPO,
        StructuredRolloutSidecar,
    )
    from drl.MARL.policy import (
        MARLCandidateRecord,
        ResourcePayloadView,
    )

    class FakeActor:
        def forward_all_owners(
            self,
            *,
            graph,
            possible_agents,
            routing=None,
        ):
            env_index = graph["env"]

            record = MARLCandidateRecord(
                candidate_index=0,
                candidate_kind="resource",
                candidate_id=f"resource:module_0:{env_index}",
                owner_agent_id="module_0",
                participant_agent_ids=("module_0",),
                presence_logit=torch.tensor(0.0),
            )

            return {
                "records": [record],
                "resource_payloads": {
                    0: ResourcePayloadView(),
                },
                "structural_payloads": {},
                "resource_log_std": torch.tensor(0.0),
                "structural_log_std": torch.tensor(0.0),
            }

    class FakeCritic:
        def act(
            self,
            inputs,
            *,
            role="",
        ):
            graph = inputs["states"]

            if isinstance(graph, list):
                values = [
                    float(item["value"])
                    for item in graph
                ]
            else:
                values = [
                    float(graph["value"])
                ]

            return (
                torch.tensor(
                    values,
                    dtype=torch.float32,
                ).reshape(-1, 1),
                {},
            )

    agent = object.__new__(
        MSSRMAPPO
    )

    agent.possible_agents = [
        "module_0",
        "module_1",
    ]

    agent.shared_actor = FakeActor()
    agent.shared_critic = FakeCritic()

    agent.memories = {
        uid: _make_real_marl_memory(
            num_envs=num_envs,
        )
        for uid in agent.possible_agents
    }

    agent._structured_rollout_sidecar = (
        StructuredRolloutSidecar()
    )

    agent._current_online_decisions = None
    agent._current_next_critic_graph = None
    agent._current_next_observations = {}
    agent._current_next_states = {}

    agent.training = True
    agent.write_interval = 0

    agent.cfg = SimpleNamespace(
        rewards_shaper=None,
        time_limit_bootstrap={
            uid: False
            for uid in agent.possible_agents
        },
        discount_factor={
            uid: 0.99
            for uid in agent.possible_agents
        },
    )

    return agent


def test_standard_act_uses_context_provider_and_returns_env_action_dict():
    import torch

    from drl.MARL.mappo import (
        OnlineRolloutContext,
    )

    agent = _make_standard_bridge_agent(
        num_envs=2,
    )

    class Provider:
        def __init__(self):
            self.encoded = None

        def current_action_contexts(self):
            return [
                OnlineRolloutContext(
                    actor_graph={
                        "env": 0,
                    },
                    critic_graph={
                        "env": 0,
                        "value": 1.25,
                    },
                    routing_snapshot={
                        "env": 0,
                    },
                    hard_mask=torch.tensor(
                        [False],
                        dtype=torch.bool,
                    ),
                ),
                OnlineRolloutContext(
                    actor_graph={
                        "env": 1,
                    },
                    critic_graph={
                        "env": 1,
                        "value": 2.50,
                    },
                    routing_snapshot={
                        "env": 1,
                    },
                    hard_mask=torch.tensor(
                        [False],
                        dtype=torch.bool,
                    ),
                ),
            ]

        def encode_env_actions(
            self,
            decisions,
        ):
            self.encoded = list(
                decisions
            )

            return {
                "module_0": torch.tensor([
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]),
                "module_1": torch.tensor([
                    [5.0, 6.0],
                    [7.0, 8.0],
                ]),
            }

        def current_critic_graphs(self):
            raise AssertionError(
                "act() must not request next critic graphs"
            )

    provider = Provider()

    agent.set_rollout_context_provider(
        provider
    )

    observations = {
        "module_0": torch.zeros(
            (2, 3)
        ),
        "module_1": torch.zeros(
            (2, 3)
        ),
    }

    states = {
        "module_0": torch.zeros(
            (2, 4)
        ),
        "module_1": torch.zeros(
            (2, 4)
        ),
    }

    actions, outputs = agent.act(
        observations,
        states,
        timestep=0,
        timesteps=32,
    )

    assert set(actions) == {
        "module_0",
        "module_1",
    }

    assert actions[
        "module_0"
    ].shape == (2, 2)

    assert actions[
        "module_1"
    ].shape == (2, 2)

    assert provider.encoded is not None
    assert len(
        provider.encoded
    ) == 2

    assert len(
        agent._current_online_decisions
    ) == 2

    for uid in agent.possible_agents:
        assert uid in outputs

        assert outputs[
            uid
        ]["log_prob"].shape == (
            2,
            1,
        )

        assert outputs[
            uid
        ]["mean_actions"] is actions[
            uid
        ]


def test_standard_record_transition_uses_post_step_privileged_graphs():
    import torch

    from drl.MARL.mappo import (
        OnlineRolloutContext,
    )

    agent = _make_standard_bridge_agent(
        num_envs=2,
    )

    next_critic_graphs = [
        {
            "env": 0,
            "value": 3.0,
        },
        {
            "env": 1,
            "value": 4.0,
        },
    ]

    class Provider:
        def current_action_contexts(self):
            return [
                OnlineRolloutContext(
                    actor_graph={
                        "env": 0,
                    },
                    critic_graph={
                        "env": 0,
                        "value": 1.0,
                    },
                    routing_snapshot=None,
                    hard_mask=torch.tensor(
                        [False],
                        dtype=torch.bool,
                    ),
                ),
                OnlineRolloutContext(
                    actor_graph={
                        "env": 1,
                    },
                    critic_graph={
                        "env": 1,
                        "value": 2.0,
                    },
                    routing_snapshot=None,
                    hard_mask=torch.tensor(
                        [False],
                        dtype=torch.bool,
                    ),
                ),
            ]

        def encode_env_actions(
            self,
            decisions,
        ):
            return {
                uid: torch.zeros(
                    (2, 2)
                )
                for uid in agent.possible_agents
            }

        def current_critic_graphs(self):
            return next_critic_graphs

    agent.set_rollout_context_provider(
        Provider()
    )

    observations = {
        uid: torch.zeros(
            (2, 3)
        )
        for uid in agent.possible_agents
    }

    states = {
        uid: torch.zeros(
            (2, 4)
        )
        for uid in agent.possible_agents
    }

    actions, _ = agent.act(
        observations,
        states,
        timestep=0,
        timesteps=32,
    )

    rewards = {
        uid: torch.tensor([
            [1.0],
            [2.0],
        ])
        for uid in agent.possible_agents
    }

    terminated = {
        uid: torch.tensor([
            [False],
            [True],
        ])
        for uid in agent.possible_agents
    }

    truncated = {
        uid: torch.tensor([
            [False],
            [False],
        ])
        for uid in agent.possible_agents
    }

    next_observations = {
        uid: torch.ones(
            (2, 3)
        )
        for uid in agent.possible_agents
    }

    next_states = {
        uid: torch.ones(
            (2, 4)
        )
        for uid in agent.possible_agents
    }

    agent.record_transition(
        observations=observations,
        states=states,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        next_states=next_states,
        terminated=terminated,
        truncated=truncated,
        infos={},
        timestep=0,
        timesteps=32,
    )

    assert len(
        agent.memories["module_0"]
    ) == 2

    assert len(
        agent.memories["module_1"]
    ) == 2

    assert len(
        agent._structured_rollout_sidecar
    ) == 2

    assert (
        agent._current_next_critic_graph
        == next_critic_graphs
    )

    assert (
        agent._current_online_decisions
        is None
    )

    expected_rewards = torch.tensor([
        [1.0],
        [2.0],
    ])

    for uid in agent.possible_agents:
        torch.testing.assert_close(
            agent.memories[
                uid
            ].get_tensor_by_name(
                "rewards"
            )[0],
            expected_rewards,
        )


def test_standard_record_transition_rejects_non_team_reward():
    import pytest
    import torch

    from drl.MARL.mappo import (
        OnlineRolloutContext,
    )

    agent = _make_standard_bridge_agent(
        num_envs=1,
    )

    class Provider:
        def current_action_contexts(self):
            return [
                OnlineRolloutContext(
                    actor_graph={
                        "env": 0,
                    },
                    critic_graph={
                        "env": 0,
                        "value": 1.0,
                    },
                    routing_snapshot=None,
                    hard_mask=torch.tensor(
                        [False],
                        dtype=torch.bool,
                    ),
                )
            ]

        def encode_env_actions(
            self,
            decisions,
        ):
            return {
                uid: torch.zeros(
                    (1, 2)
                )
                for uid in agent.possible_agents
            }

        def current_critic_graphs(self):
            return [
                {
                    "env": 0,
                    "value": 2.0,
                }
            ]

    agent.set_rollout_context_provider(
        Provider()
    )

    observations = {
        uid: torch.zeros(
            (1, 3)
        )
        for uid in agent.possible_agents
    }

    states = {
        uid: torch.zeros(
            (1, 4)
        )
        for uid in agent.possible_agents
    }

    actions, _ = agent.act(
        observations,
        states,
        timestep=0,
        timesteps=32,
    )

    rewards = {
        "module_0": torch.tensor([
            [1.0]
        ]),
        "module_1": torch.tensor([
            [2.0]
        ]),
    }

    common_bool = {
        uid: torch.zeros(
            (1, 1),
            dtype=torch.bool,
        )
        for uid in agent.possible_agents
    }

    with pytest.raises(
        ValueError,
        match="team reward",
    ):
        agent.record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=observations,
            next_states=states,
            terminated=common_bool,
            truncated=common_bool,
            infos={},
            timestep=0,
            timesteps=32,
        )


# --- ONLINE RAW ACTIONSET TEMPLATE ---

def test_online_rollout_decision_carries_detached_actionset_prediction_template():
    import torch

    from drl.MARL.mappo import (
        MSSRMAPPO,
    )
    from drl.MARL.policy import (
        MARLCandidateRecord,
        ResourcePayloadView,
    )

    raw_presence = torch.tensor(
        [0.25],
        requires_grad=True,
    )

    raw_payload = torch.tensor(
        [2.5, -1.5],
        requires_grad=True,
    )

    raw_template = {
        "presence_logits": (
            raw_presence
        ),
        "candidate_graph": torch.tensor(
            [0],
            dtype=torch.long,
        ),
        "parameters": {
            "resource": {
                "payload": (
                    raw_payload
                ),
            },
            "structural": {
                "arguments": {},
                "scope": {},
            },
        },
    }

    record = MARLCandidateRecord(
        candidate_index=0,
        candidate_kind="resource",
        candidate_id=(
            "resource|module_0|"
            "test|drive|numeric"
        ),
        owner_agent_id="module_0",
        participant_agent_ids=(),
        # Force deterministic admission for this ABI-only test.
        presence_logit=torch.tensor(
            100.0
        ),
    )

    class FakeActor:
        def forward_all_owners(
            self,
            *,
            graph,
            routing=None,
            possible_agents,
        ):
            return {
                "records": (
                    record,
                ),
                "resource_payloads": {
                    0: ResourcePayloadView(
                        numeric=None,
                        categorical_logits=None,
                    ),
                },
                "structural_payloads": {},
                "resource_log_std": torch.tensor(
                    0.0
                ),
                "structural_log_std": torch.tensor(
                    0.0
                ),
                "prediction_template": (
                    raw_template
                ),
            }

    class FakeCritic:
        def act(
            self,
            inputs,
            *,
            role="",
        ):
            return (
                torch.tensor(
                    [[1.0]]
                ),
                {},
            )

    agent = object.__new__(
        MSSRMAPPO
    )

    agent.shared_actor = (
        FakeActor()
    )

    agent.shared_critic = (
        FakeCritic()
    )

    agent.possible_agents = [
        "module_0",
    ]

    decision = (
        agent.build_online_rollout_decision(
            actor_graph={
                "actor": True,
            },
            critic_graph={
                "critic": True,
            },
            routing_snapshot={
                "routing": True,
            },
            hard_mask=torch.tensor(
                [False],
                dtype=torch.bool,
            ),
        )
    )

    stored = (
        decision
        .action_set_prediction_template
    )

    assert stored is not None
    assert stored is not raw_template

    assert set(
        stored
    ) == {
        "presence_logits",
        "candidate_graph",
        "parameters",
    }

    torch.testing.assert_close(
        stored[
            "presence_logits"
        ],
        raw_presence.detach(),
    )

    torch.testing.assert_close(
        stored[
            "parameters"
        ]["resource"]["payload"],
        raw_payload.detach(),
    )

    assert not stored[
        "presence_logits"
    ].requires_grad

    assert not stored[
        "parameters"
    ]["resource"]["payload"].requires_grad

    assert (
        stored[
            "presence_logits"
        ]
        is not raw_presence
    )

    assert (
        stored[
            "parameters"
        ]["resource"]["payload"]
        is not raw_payload
    )

    # Interaction decision must be snapshot-stable even if the original
    # actor output is subsequently modified.
    with torch.no_grad():
        raw_presence.fill_(
            99.0
        )

        raw_payload.fill_(
            99.0
        )

    torch.testing.assert_close(
        stored[
            "presence_logits"
        ],
        torch.tensor(
            [0.25]
        ),
    )

    torch.testing.assert_close(
        stored[
            "parameters"
        ]["resource"]["payload"],
        torch.tensor(
            [2.5, -1.5]
        ),
    )


# --- ALTRUISTIC ROLLOUT RECORDING BRIDGE ---

def _make_altruistic_runtime_payload():
    import torch

    from drl.MARL.altruistic_rollout import (
        AltruisticRolloutBatchPayload,
    )

    return AltruisticRolloutBatchPayload(
        module_ids_by_row=(
            (
                "env0_m1",
                "env0_m0",
            ),
            (
                "env1_m0",
                "env1_m1",
            ),
        ),
        owner_to_module_id=(
            {
                "agent_0": "env0_m0",
                "agent_1": "env0_m1",
            },
            {
                "agent_0": "env1_m0",
                "agent_1": "env1_m1",
            },
        ),
        module_positions_by_id=(
            {
                "env0_m0": torch.tensor(
                    [0.0, 0.0, 0.0]
                ),
                "env0_m1": torch.tensor(
                    [0.2, 0.0, 0.0]
                ),
            },
            {
                "env1_m0": torch.tensor(
                    [0.0, 0.0, 0.0]
                ),
                "env1_m1": torch.tensor(
                    [2.0, 0.0, 0.0]
                ),
            },
        ),
        owner_rewards={
            "agent_0": torch.tensor(
                [
                    [1.0],
                    [10.0],
                ]
            ),
            "agent_1": torch.tensor(
                [
                    [3.0],
                    [30.0],
                ]
            ),
        },
    )


def test_altruistic_rollout_bridge_records_provider_payload():
    from drl.MARL.altruistic_rollout import (
        AltruisticRolloutSidecar,
    )
    from drl.MARL.mappo import (
        MSSRMAPPO,
    )

    class Provider:
        def current_altruistic_rollout_batch(
            self,
        ):
            return (
                _make_altruistic_runtime_payload()
            )

    class Harness:
        possible_agents = (
            "agent_0",
            "agent_1",
        )

        _altruistic_rollout_provider = (
            Provider()
        )

        _altruism_radius_m = 0.5

        _altruistic_rollout_sidecar = (
            AltruisticRolloutSidecar()
        )

    harness = Harness()

    MSSRMAPPO._record_altruistic_rollout_step(
        harness,
        expected_num_envs=2,
    )

    flat = (
        harness
        ._altruistic_rollout_sidecar
        .flatten()
    )

    assert len(
        flat
    ) == 2

    assert flat[
        0
    ].owner_row_index == {
        "agent_0": 1,
        "agent_1": 0,
    }

    assert flat[
        1
    ].owner_row_index == {
        "agent_0": 0,
        "agent_1": 1,
    }

    # env0 agents are neighbours
    assert flat[
        0
    ].neighbourhoods[
        "agent_0"
    ] == (
        "agent_1",
    )

    # env1 agents are outside radius
    assert flat[
        1
    ].neighbourhoods[
        "agent_0"
    ] == ()


def test_altruistic_rollout_uses_local_rewards_not_team_reward():
    import torch

    from drl.MARL.altruistic_rollout import (
        AltruisticRolloutSidecar,
    )
    from drl.MARL.mappo import (
        MSSRMAPPO,
    )

    class Provider:
        def current_altruistic_rollout_batch(
            self,
        ):
            return (
                _make_altruistic_runtime_payload()
            )

    class Harness:
        possible_agents = (
            "agent_0",
            "agent_1",
        )

        _altruistic_rollout_provider = (
            Provider()
        )

        _altruism_radius_m = 0.5

        _altruistic_rollout_sidecar = (
            AltruisticRolloutSidecar()
        )

    harness = Harness()

    MSSRMAPPO._record_altruistic_rollout_step(
        harness,
        expected_num_envs=2,
    )

    samples = (
        harness
        ._altruistic_rollout_sidecar
        .flatten()
    )

    torch.testing.assert_close(
        samples[
            0
        ].owner_rewards[
            "agent_0"
        ],
        torch.tensor(
            1.0
        ),
    )

    torch.testing.assert_close(
        samples[
            0
        ].owner_rewards[
            "agent_1"
        ],
        torch.tensor(
            3.0
        ),
    )

    # MF reward for agent_0 is explicitly agent_1's local reward.
    torch.testing.assert_close(
        samples[
            0
        ].mean_field_rewards[
            "agent_0"
        ],
        torch.tensor(
            3.0
        ),
    )


def test_altruistic_rollout_bridge_is_noop_when_provider_is_disabled():
    from drl.MARL.mappo import (
        MSSRMAPPO,
    )

    class Harness:
        possible_agents = (
            "agent_0",
        )

        _altruistic_rollout_provider = None

    harness = Harness()

    result = (
        MSSRMAPPO
        ._record_altruistic_rollout_step(
            harness,
            expected_num_envs=1,
        )
    )

    assert result is None

    assert not hasattr(
        harness,
        "_altruistic_rollout_sidecar",
    )


def test_altruistic_rollout_bridge_rejects_wrong_environment_count():
    import pytest

    from drl.MARL.altruistic_rollout import (
        AltruisticRolloutBatchPayload,
        AltruisticRolloutSidecar,
    )
    from drl.MARL.mappo import (
        MSSRMAPPO,
    )

    payload = (
        _make_altruistic_runtime_payload()
    )

    one_env_payload = (
        AltruisticRolloutBatchPayload(
            module_ids_by_row=(
                payload.module_ids_by_row[
                    0
                ],
            ),
            owner_to_module_id=(
                payload.owner_to_module_id[
                    0
                ],
            ),
            module_positions_by_id=(
                payload.module_positions_by_id[
                    0
                ],
            ),
            owner_rewards={
                owner: value[
                    0:1
                ]
                for owner, value
                in payload.owner_rewards.items()
            },
        )
    )

    class Provider:
        def current_altruistic_rollout_batch(
            self,
        ):
            return one_env_payload

    class Harness:
        possible_agents = (
            "agent_0",
            "agent_1",
        )

        _altruistic_rollout_provider = (
            Provider()
        )

        _altruism_radius_m = 0.5

        _altruistic_rollout_sidecar = (
            AltruisticRolloutSidecar()
        )

    with pytest.raises(
        ValueError,
        match="environment count",
    ):
        MSSRMAPPO._record_altruistic_rollout_step(
            Harness(),
            expected_num_envs=2,
        )


def test_record_online_rollout_batch_invokes_altruistic_recording_bridge():
    import torch
    from types import SimpleNamespace

    from drl.MARL.altruistic_rollout import (
        AltruisticRolloutSidecar,
    )
    from drl.MARL.mappo import (
        MSSRMAPPO,
        StructuredRolloutSidecar,
    )

    class FakeMemory:
        def __init__(
            self,
        ):
            self.calls = []

        def add_samples(
            self,
            **kwargs,
        ):
            self.calls.append(
                kwargs
            )

    class Provider:
        def current_altruistic_rollout_batch(
            self,
        ):
            return (
                _make_altruistic_runtime_payload()
            )

    class Harness:
        possible_agents = (
            "agent_0",
            "agent_1",
        )

        memories = {
            "agent_0": FakeMemory(),
            "agent_1": FakeMemory(),
        }

        _structured_rollout_sidecar = (
            StructuredRolloutSidecar()
        )

        _altruistic_rollout_provider = (
            Provider()
        )

        _altruism_radius_m = 0.5

        _altruistic_rollout_sidecar = (
            AltruisticRolloutSidecar()
        )

        # This test invokes record_online_rollout_batch as an unbound
        # MSSRMAPPO method on a minimal harness. Bind the real helper too,
        # matching the method available on a real MSSRMAPPO instance.
        _record_altruistic_rollout_step = (
            MSSRMAPPO._record_altruistic_rollout_step
        )

    harness = Harness()

    decisions = [
        SimpleNamespace(
            owner_log_probs={
                "agent_0": torch.tensor(
                    -0.1
                ),
                "agent_1": torch.tensor(
                    -0.2
                ),
            },
            global_value=torch.tensor(
                0.5
            ),
            structured_sample=StructuredRolloutSample(
                action=None, hard_mask=None, owner_active_mask={}, routing_snapshot=None, graph_snapshot=None),
        ),
        SimpleNamespace(
            owner_log_probs={
                "agent_0": torch.tensor(
                    -0.3
                ),
                "agent_1": torch.tensor(
                    -0.4
                ),
            },
            global_value=torch.tensor(
                0.6
            ),
            structured_sample=StructuredRolloutSample(
                action=None, hard_mask=None, owner_active_mask={}, routing_snapshot=None, graph_snapshot=None),
        ),
    ]

    rows = {
        owner: torch.zeros(
            2,
            1,
        )
        for owner in harness.possible_agents
    }

    MSSRMAPPO.record_online_rollout_batch(
        harness,
        decisions=decisions,
        observations=rows,
        states=rows,
        env_actions=rows,
        team_rewards=torch.tensor(
            [
                [100.0],
                [200.0],
            ]
        ),
        terminated=torch.zeros(
            2,
            1,
            dtype=torch.bool,
        ),
        truncated=torch.zeros(
            2,
            1,
            dtype=torch.bool,
        ),
        next_observations=rows,
        next_states=rows,
        next_critic_graphs=[
            object(),
            object(),
        ],
    )

    # Existing team-MAPPO Memory path remains unchanged.
    for owner in harness.possible_agents:
        assert len(
            harness.memories[
                owner
            ].calls
        ) == 1

        torch.testing.assert_close(
            harness.memories[
                owner
            ].calls[
                0
            ][
                "rewards"
            ],
            torch.tensor(
                [
                    [100.0],
                    [200.0],
                ]
            ),
        )

    # Separate APPO sidecar contains local rewards instead.
    altruistic = (
        harness
        ._altruistic_rollout_sidecar
        .flatten()
    )

    assert len(
        altruistic
    ) == 2

    torch.testing.assert_close(
        altruistic[
            0
        ].owner_rewards[
            "agent_0"
        ],
        torch.tensor(
            1.0
        ),
    )

    torch.testing.assert_close(
        altruistic[
            0
        ].owner_rewards[
            "agent_1"
        ],
        torch.tensor(
            3.0
        ),
    )


# --- PER-OWNER ALTRUISTIC PPO ADVANTAGES ---

def test_owner_clipped_policy_loss_accepts_distinct_advantage_per_owner():
    import torch

    from drl.MARL.mappo import (
        MSSRMAPPO,
    )

    losses = (
        MSSRMAPPO
        .compute_owner_clipped_policy_losses(
            new_log_probs={
                "module_0": torch.zeros(
                    (2, 1)
                ),
                "module_1": torch.zeros(
                    (2, 1)
                ),
            },
            old_log_probs={
                "module_0": torch.zeros(
                    (2, 1)
                ),
                "module_1": torch.zeros(
                    (2, 1)
                ),
            },
            advantages={
                "module_0": torch.full(
                    (2, 1),
                    2.0,
                ),
                "module_1": torch.full(
                    (2, 1),
                    -3.0,
                ),
            },
            ratio_clip=0.2,
        )
    )

    # ratio == 1, therefore PPO loss == -advantage.
    torch.testing.assert_close(
        losses[
            "module_0"
        ],
        torch.tensor(
            -2.0
        ),
    )

    torch.testing.assert_close(
        losses[
            "module_1"
        ],
        torch.tensor(
            3.0
        ),
    )


def test_owner_actor_loss_keeps_existing_shared_advantage_abi():
    import torch

    from drl.MARL.mappo import (
        MSSRMAPPO,
    )

    losses = (
        MSSRMAPPO
        .compute_owner_actor_losses(
            new_log_probs={
                "module_0": torch.zeros(
                    (2, 1)
                ),
                "module_1": torch.zeros(
                    (2, 1)
                ),
            },
            old_log_probs={
                "module_0": torch.zeros(
                    (2, 1)
                ),
                "module_1": torch.zeros(
                    (2, 1)
                ),
            },
            advantages=torch.full(
                (2, 1),
                4.0,
            ),
            owner_entropies={
                "module_0": torch.zeros(
                    (2, 1)
                ),
                "module_1": torch.zeros(
                    (2, 1)
                ),
            },
            ratio_clip=0.2,
            entropy_loss_scale=0.0,
        )
    )

    # Existing team-MAPPO behavior remains unchanged.
    torch.testing.assert_close(
        losses[
            "module_0"
        ],
        torch.tensor(
            -4.0
        ),
    )

    torch.testing.assert_close(
        losses[
            "module_1"
        ],
        torch.tensor(
            -4.0
        ),
    )


def test_altruistic_advantage_changes_real_shared_actor_optimizer_update():
    import torch

    from drl.MARL.altruism import (
        combine_altruistic_advantages,
    )
    from drl.MARL.mappo import (
        MSSRMAPPO,
    )

    def run_update(
        alpha,
    ):
        actor_parameter = (
            torch.nn.Parameter(
                torch.tensor(
                    0.0
                )
            )
        )

        critic_parameter = (
            torch.nn.Parameter(
                torch.tensor(
                    1.0
                )
            )
        )

        actor_optimizer = (
            torch.optim.SGD(
                [
                    actor_parameter
                ],
                lr=0.1,
            )
        )

        critic_optimizer = (
            torch.optim.SGD(
                [
                    critic_parameter
                ],
                lr=0.1,
            )
        )

        individual = {
            "module_0": torch.tensor(
                [[1.0]]
            ),
        }

        mean_field = {
            "module_0": torch.tensor(
                [[2.0]]
            ),
        }

        altruistic = (
            combine_altruistic_advantages(
                individual,
                mean_field,
                alpha=alpha,
            )
        )

        # Treat the trainable scalar as the reevaluated log-prob.
        new_log_probs = {
            "module_0": (
                actor_parameter
                .reshape(
                    1,
                    1,
                )
            ),
        }

        old_log_probs = {
            "module_0": torch.zeros(
                (1, 1)
            ),
        }

        actor_losses = (
            MSSRMAPPO
            .compute_owner_actor_losses(
                new_log_probs=(
                    new_log_probs
                ),
                old_log_probs=(
                    old_log_probs
                ),
                advantages=(
                    altruistic
                ),
                owner_entropies={
                    "module_0": torch.zeros(
                        (1, 1)
                    ),
                },
                ratio_clip=0.2,
                entropy_loss_scale=0.0,
            )
        )

        MSSRMAPPO.coordinated_optimizer_step(
            owner_policy_losses=(
                actor_losses
            ),
            value_loss=(
                critic_parameter
                .square()
            ),
            actor_optimizer=(
                actor_optimizer
            ),
            critic_optimizer=(
                critic_optimizer
            ),
            active_owners=(
                "module_0",
            ),
        )

        return (
            actor_parameter
            .detach()
            .clone()
        )

    selfish = run_update(
        alpha=0.0
    )

    altruistic = run_update(
        alpha=1.0
    )

    # Same actor, same own advantage, same PPO machinery.
    # The only difference is the neighbour contribution.
    assert not torch.equal(
        selfish,
        altruistic,
    )

    # Positive neighbour advantage increases the positive update magnitude.
    assert (
        altruistic.item()
        >
        selfish.item()
        >
        0.0
    )


# --- ALTRUISTIC ROLLOUT TO GAE TARGETS ---

def _make_two_step_altruistic_sidecar():
    import torch

    from drl.MARL.altruistic_rollout import (
        AltruisticRolloutSidecar,
        build_altruistic_rollout_batch,
    )

    sidecar = AltruisticRolloutSidecar()

    def append(
        reward_a,
        reward_b,
    ):
        samples = build_altruistic_rollout_batch(
            possible_agents=(
                "module_0",
                "module_1",
            ),
            module_ids_by_row=(
                (
                    "physical_b",
                    "physical_a",
                ),
            ),
            owner_to_module_id=(
                {
                    "module_0": "physical_a",
                    "module_1": "physical_b",
                },
            ),
            module_positions_by_id=(
                {
                    "physical_a": torch.tensor(
                        [0.0, 0.0, 0.0]
                    ),
                    "physical_b": torch.tensor(
                        [0.2, 0.0, 0.0]
                    ),
                },
            ),
            owner_rewards={
                "module_0": torch.tensor(
                    [[reward_a]],
                    dtype=torch.float32,
                ),
                "module_1": torch.tensor(
                    [[reward_b]],
                    dtype=torch.float32,
                ),
            },
            radius_m=0.5,
        )

        sidecar.append_step(
            samples
        )

    append(
        1.0,
        3.0,
    )

    append(
        2.0,
        4.0,
    )

    return sidecar


def _zero_altruistic_values():
    import torch

    return {
        "module_0": torch.zeros(
            2,
            1,
            1,
        ),
        "module_1": torch.zeros(
            2,
            1,
            1,
        ),
    }


def _zero_altruistic_bootstrap():
    import torch

    return {
        "module_0": torch.zeros(
            1,
            1,
        ),
        "module_1": torch.zeros(
            1,
            1,
        ),
    }


def test_recorded_rollout_automatically_builds_owner_and_mean_field_gae():
    import torch

    from drl.MARL.mappo import (
        MSSRMAPPO,
    )

    result = (
        MSSRMAPPO
        .compute_altruistic_rollout_targets(
            sidecar=(
                _make_two_step_altruistic_sidecar()
            ),
            possible_agents=(
                "module_0",
                "module_1",
            ),
            owner_values=(
                _zero_altruistic_values()
            ),
            mean_field_values=(
                _zero_altruistic_values()
            ),
            owner_bootstrap_values=(
                _zero_altruistic_bootstrap()
            ),
            mean_field_bootstrap_values=(
                _zero_altruistic_bootstrap()
            ),
            dones=torch.zeros(
                2,
                1,
                1,
                dtype=torch.bool,
            ),
            discount_factor=0.0,
            lambda_coefficient=0.95,
            alpha=0.5,
        )
    )

    # gamma=0 and V=0:
    # A_i is exactly the local reward.
    torch.testing.assert_close(
        result.owner_advantages[
            "module_0"
        ],
        torch.tensor(
            [
                [[1.0]],
                [[2.0]],
            ]
        ),
    )

    torch.testing.assert_close(
        result.owner_advantages[
            "module_1"
        ],
        torch.tensor(
            [
                [[3.0]],
                [[4.0]],
            ]
        ),
    )

    # Both modules are neighbours:
    # MF(module_0) receives module_1 reward and vice versa.
    torch.testing.assert_close(
        result.mean_field_advantages[
            "module_0"
        ],
        torch.tensor(
            [
                [[3.0]],
                [[4.0]],
            ]
        ),
    )

    torch.testing.assert_close(
        result.mean_field_advantages[
            "module_1"
        ],
        torch.tensor(
            [
                [[1.0]],
                [[2.0]],
            ]
        ),
    )

    # alpha = 0.5
    torch.testing.assert_close(
        result.altruistic_advantages[
            "module_0"
        ],
        torch.tensor(
            [
                [[2.5]],
                [[4.0]],
            ]
        ),
    )

    torch.testing.assert_close(
        result.altruistic_advantages[
            "module_1"
        ],
        torch.tensor(
            [
                [[3.5]],
                [[5.0]],
            ]
        ),
    )


def test_rollout_alpha_zero_is_exact_selfish_training_target():
    import torch

    from drl.MARL.mappo import (
        MSSRMAPPO,
    )

    result = (
        MSSRMAPPO
        .compute_altruistic_rollout_targets(
            sidecar=(
                _make_two_step_altruistic_sidecar()
            ),
            possible_agents=(
                "module_0",
                "module_1",
            ),
            owner_values=(
                _zero_altruistic_values()
            ),
            mean_field_values=(
                _zero_altruistic_values()
            ),
            owner_bootstrap_values=(
                _zero_altruistic_bootstrap()
            ),
            mean_field_bootstrap_values=(
                _zero_altruistic_bootstrap()
            ),
            dones=torch.zeros(
                2,
                1,
                1,
                dtype=torch.bool,
            ),
            discount_factor=0.0,
            lambda_coefficient=0.95,
            alpha=0.0,
        )
    )

    for owner in (
        "module_0",
        "module_1",
    ):
        torch.testing.assert_close(
            result.altruistic_advantages[
                owner
            ],
            result.owner_advantages[
                owner
            ],
        )


def test_rollout_targets_keep_owner_and_mean_field_critic_returns_separate():
    import torch

    from drl.MARL.mappo import (
        MSSRMAPPO,
    )

    result = (
        MSSRMAPPO
        .compute_altruistic_rollout_targets(
            sidecar=(
                _make_two_step_altruistic_sidecar()
            ),
            possible_agents=(
                "module_0",
                "module_1",
            ),
            owner_values=(
                _zero_altruistic_values()
            ),
            mean_field_values=(
                _zero_altruistic_values()
            ),
            owner_bootstrap_values=(
                _zero_altruistic_bootstrap()
            ),
            mean_field_bootstrap_values=(
                _zero_altruistic_bootstrap()
            ),
            dones=torch.zeros(
                2,
                1,
                1,
                dtype=torch.bool,
            ),
            discount_factor=0.0,
            lambda_coefficient=0.95,
            alpha=1.0,
        )
    )

    # V_i target follows R_i.
    torch.testing.assert_close(
        result.owner_returns[
            "module_0"
        ],
        torch.tensor(
            [
                [[1.0]],
                [[2.0]],
            ]
        ),
    )

    # V_i^MF target follows neighbours' reward, not altruistic sum.
    torch.testing.assert_close(
        result.mean_field_returns[
            "module_0"
        ],
        torch.tensor(
            [
                [[3.0]],
                [[4.0]],
            ]
        ),
    )


def test_recorded_neighbour_reward_changes_real_actor_update():
    import torch

    from drl.MARL.mappo import (
        MSSRMAPPO,
    )

    def run(
        alpha,
    ):
        targets = (
            MSSRMAPPO
            .compute_altruistic_rollout_targets(
                sidecar=(
                    _make_two_step_altruistic_sidecar()
                ),
                possible_agents=(
                    "module_0",
                    "module_1",
                ),
                owner_values=(
                    _zero_altruistic_values()
                ),
                mean_field_values=(
                    _zero_altruistic_values()
                ),
                owner_bootstrap_values=(
                    _zero_altruistic_bootstrap()
                ),
                mean_field_bootstrap_values=(
                    _zero_altruistic_bootstrap()
                ),
                dones=torch.zeros(
                    2,
                    1,
                    1,
                    dtype=torch.bool,
                ),
                discount_factor=0.0,
                lambda_coefficient=0.95,
                alpha=alpha,
            )
        )

        actor_parameter = torch.nn.Parameter(
            torch.tensor(
                0.0
            )
        )

        critic_parameter = torch.nn.Parameter(
            torch.tensor(
                1.0
            )
        )

        actor_optimizer = torch.optim.SGD(
            [
                actor_parameter
            ],
            lr=0.1,
        )

        critic_optimizer = torch.optim.SGD(
            [
                critic_parameter
            ],
            lr=0.1,
        )

        # Use timestep 0 / env 0 for module_0.
        advantage = (
            targets
            .altruistic_advantages[
                "module_0"
            ][
                0
            ]
        )

        actor_losses = (
            MSSRMAPPO
            .compute_owner_actor_losses(
                new_log_probs={
                    "module_0": (
                        actor_parameter
                        .reshape(
                            1,
                            1,
                        )
                    ),
                },
                old_log_probs={
                    "module_0": torch.zeros(
                        1,
                        1,
                    ),
                },
                advantages={
                    "module_0": (
                        advantage
                    ),
                },
                owner_entropies={
                    "module_0": torch.zeros(
                        1,
                        1,
                    ),
                },
                ratio_clip=0.2,
                entropy_loss_scale=0.0,
            )
        )

        MSSRMAPPO.coordinated_optimizer_step(
            owner_policy_losses=(
                actor_losses
            ),
            value_loss=(
                critic_parameter
                .square()
            ),
            actor_optimizer=(
                actor_optimizer
            ),
            critic_optimizer=(
                critic_optimizer
            ),
            active_owners=(
                "module_0",
            ),
        )

        return (
            actor_parameter
            .detach()
            .clone()
        )

    selfish = run(
        0.0
    )

    altruistic = run(
        1.0
    )

    assert (
        altruistic.item()
        >
        selfish.item()
        >
        0.0
    )


# --- REAL ALTRUISTIC CRITIC ROLLOUT VALUES ---

def _make_real_test_owner_critic(seed):
    import torch

    from drl.MARL.owner_critic import (
        OwnerConditionedCritic,
    )

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()

            # Keep a real trainable encoder parameter so this is still
            # a genuine neural critic, not a hand-written value function.
            self.scale = torch.nn.Parameter(
                torch.tensor(1.0)
            )

        def encode(
            self,
            graph,
        ):
            module_h = (
                graph[
                    "module_h"
                ]
                * self.scale
            )

            return {
                "module_h": module_h,
                "graph_ptr": graph[
                    "graph_ptr"
                ],
            }

    torch.manual_seed(
        seed
    )

    return (
        OwnerConditionedCritic(
            encoder=Encoder(),
            module_latent_dim=2,
            value_hidden_dim=8,
            device="cpu",
        )
        .eval()
    )


def _make_structured_critic_rollout():
    import torch

    from drl.MARL.mappo import (
        StructuredRolloutSample,
    )

    graphs = (
        {
            "module_h": torch.tensor(
                [
                    [1.0, 0.0],   # physical_b -> row 0
                    [0.0, 2.0],   # physical_a -> row 1
                ]
            ),
            "graph_ptr": torch.tensor(
                [0, 2],
                dtype=torch.long,
            ),
        },
        {
            "module_h": torch.tensor(
                [
                    [3.0, 0.0],   # physical_b -> row 0
                    [0.0, 4.0],   # physical_a -> row 1
                ]
            ),
            "graph_ptr": torch.tensor(
                [0, 2],
                dtype=torch.long,
            ),
        },
    )

    return tuple(
        StructuredRolloutSample(
            action=None,
            hard_mask=(),
            owner_active_mask={
                "module_0": True,
                "module_1": True,
            },
            routing_snapshot=None,
            graph_snapshot=None,
            critic_graph_snapshot=graph,
        )
        for graph in graphs
    )


def test_real_owner_conditioned_critics_evaluate_recorded_rollout_rows():
    import torch

    from drl.MARL.mappo import (
        MSSRMAPPO,
    )

    owner_critic = (
        _make_real_test_owner_critic(
            6101
        )
    )

    mean_field_critic = (
        _make_real_test_owner_critic(
            6102
        )
    )

    structured = (
        _make_structured_critic_rollout()
    )

    altruistic = (
        _make_two_step_altruistic_sidecar()
    )

    (
        owner_values,
        mean_field_values,
    ) = (
        MSSRMAPPO
        .evaluate_altruistic_rollout_values(
            structured_samples=structured,
            altruistic_sidecar=altruistic,
            possible_agents=(
                "module_0",
                "module_1",
            ),
            owner_critic=owner_critic,
            mean_field_critic=(
                mean_field_critic
            ),
        )
    )

    assert owner_values[
        "module_0"
    ].shape == (
        2,
        1,
        1,
    )

    assert owner_values[
        "module_1"
    ].shape == (
        2,
        1,
        1,
    )

    assert mean_field_values[
        "module_0"
    ].shape == (
        2,
        1,
        1,
    )

    # The altruistic sidecar says:
    # module_0 -> physical_a -> graph row 1
    # module_1 -> physical_b -> graph row 0
    for timestep, sample in enumerate(
        structured
    ):
        graph = (
            sample
            .critic_graph_snapshot
        )

        expected_owner_0 = (
            owner_critic
            .forward_owner(
                graph,
                owner_index=1,
            )
            .detach()
            .reshape(
                1
            )
        )

        expected_owner_1 = (
            owner_critic
            .forward_owner(
                graph,
                owner_index=0,
            )
            .detach()
            .reshape(
                1
            )
        )

        torch.testing.assert_close(
            owner_values[
                "module_0"
            ][
                timestep,
                0,
            ],
            expected_owner_0,
        )

        torch.testing.assert_close(
            owner_values[
                "module_1"
            ][
                timestep,
                0,
            ],
            expected_owner_1,
        )


def test_altruistic_rollout_value_mapping_never_uses_owner_name_as_row():
    import torch

    from drl.MARL.mappo import (
        MSSRMAPPO,
    )

    class RowEchoCritic:
        def forward_owner(
            self,
            graph,
            *,
            owner_index,
        ):
            return (
                graph[
                    "row_values"
                ][
                    owner_index
                ]
                .reshape(
                    1,
                    1,
                )
            )

    structured = (
        _make_structured_critic_rollout()
    )

    # Add an explicit row value that makes wrong row selection obvious.
    structured = tuple(
        type(sample)(
            action=sample.action,
            hard_mask=sample.hard_mask,
            owner_active_mask=(
                sample.owner_active_mask
            ),
            routing_snapshot=(
                sample.routing_snapshot
            ),
            graph_snapshot=(
                sample.graph_snapshot
            ),
            critic_graph_snapshot={
                **sample.critic_graph_snapshot,
                "row_values": torch.tensor(
                    [
                        10.0,
                        20.0,
                    ]
                ),
            },
        )
        for sample in structured
    )

    critic = RowEchoCritic()

    owner_values, _ = (
        MSSRMAPPO
        .evaluate_altruistic_rollout_values(
            structured_samples=structured,
            altruistic_sidecar=(
                _make_two_step_altruistic_sidecar()
            ),
            possible_agents=(
                "module_0",
                "module_1",
            ),
            owner_critic=critic,
            mean_field_critic=critic,
        )
    )

    # module_0 is mapped to physical_a == row 1.
    torch.testing.assert_close(
        owner_values[
            "module_0"
        ],
        torch.tensor(
            [
                [[20.0]],
                [[20.0]],
            ]
        ),
    )

    # module_1 is mapped to physical_b == row 0.
    torch.testing.assert_close(
        owner_values[
            "module_1"
        ],
        torch.tensor(
            [
                [[10.0]],
                [[10.0]],
            ]
        ),
    )


def test_altruistic_rollout_values_are_detached_gae_targets():
    from drl.MARL.mappo import (
        MSSRMAPPO,
    )

    owner_critic = (
        _make_real_test_owner_critic(
            6201
        )
    )

    mean_field_critic = (
        _make_real_test_owner_critic(
            6202
        )
    )

    owner_values, mean_field_values = (
        MSSRMAPPO
        .evaluate_altruistic_rollout_values(
            structured_samples=(
                _make_structured_critic_rollout()
            ),
            altruistic_sidecar=(
                _make_two_step_altruistic_sidecar()
            ),
            possible_agents=(
                "module_0",
                "module_1",
            ),
            owner_critic=owner_critic,
            mean_field_critic=(
                mean_field_critic
            ),
        )
    )

    for mapping in (
        owner_values,
        mean_field_values,
    ):
        for value in mapping.values():
            assert (
                value.requires_grad
                is False
            )


def test_altruistic_rollout_value_evaluation_rejects_alignment_or_missing_graph():
    import pytest

    from drl.MARL.mappo import (
        MSSRMAPPO,
        StructuredRolloutSample,
    )

    critic = (
        _make_real_test_owner_critic(
            6301
        )
    )

    altruistic = (
        _make_two_step_altruistic_sidecar()
    )

    with pytest.raises(
        ValueError,
        match="alignment",
    ):
        MSSRMAPPO.evaluate_altruistic_rollout_values(
            structured_samples=(
                _make_structured_critic_rollout()[
                    :1
                ]
            ),
            altruistic_sidecar=(
                altruistic
            ),
            possible_agents=(
                "module_0",
                "module_1",
            ),
            owner_critic=critic,
            mean_field_critic=critic,
        )

    bad = list(
        _make_structured_critic_rollout()
    )

    bad[
        0
    ] = StructuredRolloutSample(
        action=None,
        hard_mask=(),
        owner_active_mask={
            "module_0": True,
            "module_1": True,
        },
        routing_snapshot=None,
        graph_snapshot=None,
        critic_graph_snapshot=None,
    )

    with pytest.raises(
        ValueError,
        match="critic graph",
    ):
        MSSRMAPPO.evaluate_altruistic_rollout_values(
            structured_samples=bad,
            altruistic_sidecar=(
                altruistic
            ),
            possible_agents=(
                "module_0",
                "module_1",
            ),
            owner_critic=critic,
            mean_field_critic=critic,
        )


# --- ALTRUISTIC NEXT-STATE BOOTSTRAP ---

def _make_altruistic_sidecar_with_next_mapping():
    import torch

    from drl.MARL.altruistic_rollout import (
        AltruisticRolloutSidecar,
        build_altruistic_rollout_batch,
    )

    sidecar = AltruisticRolloutSidecar()

    samples = build_altruistic_rollout_batch(
        possible_agents=(
            "module_0",
            "module_1",
        ),

        # Current graph:
        # physical_b -> row 0
        # physical_a -> row 1
        module_ids_by_row=(
            (
                "physical_b",
                "physical_a",
            ),
        ),

        owner_to_module_id=(
            {
                "module_0": "physical_a",
                "module_1": "physical_b",
            },
        ),

        module_positions_by_id=(
            {
                "physical_a": torch.tensor(
                    [0.0, 0.0, 0.0]
                ),
                "physical_b": torch.tensor(
                    [0.2, 0.0, 0.0]
                ),
            },
        ),

        owner_rewards={
            "module_0": torch.tensor(
                [[1.0]]
            ),
            "module_1": torch.tensor(
                [[2.0]]
            ),
        },

        radius_m=0.5,

        # NEXT graph deliberately changes row order:
        # physical_a -> row 0
        # physical_b -> row 1
        next_module_ids_by_row=(
            (
                "physical_a",
                "physical_b",
            ),
        ),

        next_owner_to_module_id=(
            {
                "module_0": "physical_a",
                "module_1": "physical_b",
            },
        ),
    )

    sidecar.append_step(
        samples
    )

    return sidecar


def test_altruistic_sample_stores_explicit_next_state_owner_rows():
    sidecar = (
        _make_altruistic_sidecar_with_next_mapping()
    )

    sample = sidecar.flatten()[
        0
    ]

    # Current:
    # module_0 -> row 1
    # module_1 -> row 0
    assert sample.owner_row_index == {
        "module_0": 1,
        "module_1": 0,
    }

    # Next state has changed graph row order.
    assert sample.next_owner_row_index == {
        "module_0": 0,
        "module_1": 1,
    }


def test_real_altruistic_bootstrap_uses_next_state_mapping_not_current_mapping():
    import torch

    from drl.MARL.mappo import (
        MSSRMAPPO,
    )

    class RowEchoCritic:
        def forward_owner(
            self,
            graph,
            *,
            owner_index,
        ):
            return (
                graph[
                    "row_values"
                ][
                    owner_index
                ]
                .reshape(
                    1,
                    1,
                )
            )

    critic = RowEchoCritic()

    next_graphs = (
        {
            "row_values": torch.tensor(
                [
                    100.0,
                    200.0,
                ]
            ),
        },
    )

    (
        owner_bootstrap,
        mean_field_bootstrap,
    ) = (
        MSSRMAPPO
        .evaluate_altruistic_bootstrap_values(
            next_critic_graphs=(
                next_graphs
            ),
            altruistic_sidecar=(
                _make_altruistic_sidecar_with_next_mapping()
            ),
            possible_agents=(
                "module_0",
                "module_1",
            ),
            owner_critic=critic,
            mean_field_critic=critic,
        )
    )

    # NEXT mapping:
    # module_0 -> row 0
    # module_1 -> row 1
    #
    # If the implementation incorrectly reused the current mapping,
    # these two values would be swapped.
    torch.testing.assert_close(
        owner_bootstrap[
            "module_0"
        ],
        torch.tensor(
            [[100.0]]
        ),
    )

    torch.testing.assert_close(
        owner_bootstrap[
            "module_1"
        ],
        torch.tensor(
            [[200.0]]
        ),
    )

    torch.testing.assert_close(
        mean_field_bootstrap[
            "module_0"
        ],
        torch.tensor(
            [[100.0]]
        ),
    )


def test_real_owner_and_mean_field_bootstrap_use_independent_critics():
    import torch

    from drl.MARL.mappo import (
        MSSRMAPPO,
    )

    class ScaledRowCritic:
        def __init__(
            self,
            scale,
        ):
            self.scale = scale

        def forward_owner(
            self,
            graph,
            *,
            owner_index,
        ):
            return (
                graph[
                    "row_values"
                ][
                    owner_index
                ]
                .reshape(
                    1,
                    1,
                )
                * self.scale
            )

    (
        owner_bootstrap,
        mf_bootstrap,
    ) = (
        MSSRMAPPO
        .evaluate_altruistic_bootstrap_values(
            next_critic_graphs=(
                {
                    "row_values": torch.tensor(
                        [
                            2.0,
                            5.0,
                        ]
                    ),
                },
            ),
            altruistic_sidecar=(
                _make_altruistic_sidecar_with_next_mapping()
            ),
            possible_agents=(
                "module_0",
                "module_1",
            ),
            owner_critic=(
                ScaledRowCritic(
                    10.0
                )
            ),
            mean_field_critic=(
                ScaledRowCritic(
                    100.0
                )
            ),
        )
    )

    torch.testing.assert_close(
        owner_bootstrap[
            "module_0"
        ],
        torch.tensor(
            [[20.0]]
        ),
    )

    torch.testing.assert_close(
        mf_bootstrap[
            "module_0"
        ],
        torch.tensor(
            [[200.0]]
        ),
    )


def test_altruistic_bootstrap_rejects_missing_next_mapping():
    import pytest

    from drl.MARL.mappo import (
        MSSRMAPPO,
    )

    class Critic:
        def forward_owner(
            self,
            graph,
            *,
            owner_index,
        ):
            import torch

            return torch.tensor(
                [[0.0]]
            )

    with pytest.raises(
        ValueError,
        match="next-state owner-row",
    ):
        MSSRMAPPO.evaluate_altruistic_bootstrap_values(
            next_critic_graphs=(
                {
                    "graph": 1,
                },
            ),
            altruistic_sidecar=(
                _make_two_step_altruistic_sidecar()
            ),
            possible_agents=(
                "module_0",
                "module_1",
            ),
            owner_critic=Critic(),
            mean_field_critic=Critic(),
        )


# --- COMPLETE FIXED-ALPHA APPO UPDATE ---

def test_altruistic_sidecar_minibatches_follow_exact_sampling_indexes():
    import torch

    sidecar = _make_two_step_altruistic_sidecar()

    batches = sidecar.sample_minibatches(
        indexes=torch.tensor(
            [1, 0]
        ),
        mini_batches=2,
    )

    assert len(batches) == 2

    assert (
        batches[0][0]
        .owner_rewards["module_0"]
        .item()
        == 2.0
    )

    assert (
        batches[1][0]
        .owner_rewards["module_0"]
        .item()
        == 1.0
    )


def test_owner_conditioned_value_loss_keeps_targets_per_owner():
    import torch

    from drl.MARL.mappo import MSSRMAPPO

    loss = (
        MSSRMAPPO
        .compute_owner_conditioned_value_loss(
            predicted_values={
                "module_0": torch.tensor(
                    [[1.0]],
                    requires_grad=True,
                ),
                "module_1": torch.tensor(
                    [[2.0]],
                    requires_grad=True,
                ),
            },
            sampled_values={
                "module_0": torch.tensor(
                    [[0.0]]
                ),
                "module_1": torch.tensor(
                    [[0.0]]
                ),
            },
            sampled_returns={
                "module_0": torch.tensor(
                    [[3.0]]
                ),
                "module_1": torch.tensor(
                    [[6.0]]
                ),
            },
            value_clip=0.0,
            value_loss_scale=1.0,
        )
    )

    # mean(
    #   (3 - 1)^2,
    #   (6 - 2)^2
    # ) = mean(4, 16) = 10
    torch.testing.assert_close(
        loss,
        torch.tensor(10.0),
    )


def test_altruistic_optimizer_steps_actor_and_both_critics():
    import torch

    from drl.MARL.mappo import MSSRMAPPO

    actor = torch.nn.Parameter(
        torch.tensor(0.0)
    )

    owner_critic = torch.nn.Parameter(
        torch.tensor(0.0)
    )

    mf_critic = torch.nn.Parameter(
        torch.tensor(0.0)
    )

    actor_optimizer = torch.optim.SGD(
        [actor],
        lr=0.1,
    )

    owner_optimizer = torch.optim.SGD(
        [owner_critic],
        lr=0.1,
    )

    mf_optimizer = torch.optim.SGD(
        [mf_critic],
        lr=0.1,
    )

    MSSRMAPPO.coordinated_altruistic_optimizer_step(
        owner_policy_losses={
            "module_0": -actor,
        },
        owner_value_loss=(
            owner_critic - 2.0
        ).square(),
        mean_field_value_loss=(
            mf_critic - 3.0
        ).square(),
        actor_optimizer=actor_optimizer,
        owner_critic_optimizer=(
            owner_optimizer
        ),
        mean_field_critic_optimizer=(
            mf_optimizer
        ),
        active_owners=(
            "module_0",
        ),
    )

    assert actor.item() > 0.0
    assert owner_critic.item() > 0.0
    assert mf_critic.item() > 0.0


def test_coordinated_update_dispatches_explicit_altruistic_mode():
    from drl.MARL.mappo import MSSRMAPPO

    agent = object.__new__(
        MSSRMAPPO
    )

    agent.memories = {
        "module_0": object(),
    }

    agent._training_advantage_mode = (
        "altruistic"
    )

    calls = []

    def altruistic_update(
        *,
        timestep,
        timesteps,
    ):
        calls.append(
            (
                timestep,
                timesteps,
            )
        )

    agent.coordinated_altruistic_update = (
        altruistic_update
    )

    # These must never be reached in altruistic mode.
    agent.compute_global_bootstrap_value = (
        lambda: (_ for _ in ()).throw(
            AssertionError(
                "team bootstrap must not run"
            )
        )
    )

    agent.compute_and_broadcast_team_gae = (
        lambda **kwargs: (
            (_ for _ in ()).throw(
                AssertionError(
                    "team GAE must not run"
                )
            )
        )
    )

    MSSRMAPPO.coordinated_update(
        agent,
        timestep=12,
        timesteps=100,
    )

    assert calls == [
        (
            12,
            100,
        )
    ]


def test_real_offline_fixed_alpha_appo_update_steps_three_networks():
    import torch
    from types import SimpleNamespace

    from drl.MARL.altruistic_rollout import (
        AltruisticRolloutSidecar,
        build_altruistic_rollout_batch,
    )

    from drl.MARL.mappo import (
        MSSRMAPPO,
        StructuredRolloutSample,
        StructuredRolloutSidecar,
    )

    possible_agents = (
        "module_0",
        "module_1",
    )

    # ---------------------------------------------------------
    # Minimal skrl-like memory
    # ---------------------------------------------------------

    class FakeMemory:
        def __init__(self):
            self.sampling_indexes = None

            self._states = torch.zeros(
                2,
                1,
            )

            self._log_prob = torch.zeros(
                2,
                1,
            )

            self._terminated = torch.zeros(
                2,
                1,
                1,
                dtype=torch.bool,
            )

            self._truncated = torch.zeros(
                2,
                1,
                1,
                dtype=torch.bool,
            )

        def __len__(self):
            return 2

        def get_tensor_by_name(
            self,
            name,
        ):
            return {
                "terminated": self._terminated,
                "truncated": self._truncated,
            }[
                name
            ]

        def sample(
            self,
            *,
            names,
            batch_size,
            mini_batches,
        ):
            assert batch_size == 2
            assert mini_batches == 1

            # Deliberately reverse order.
            self.sampling_indexes = (
                torch.tensor(
                    [1, 0]
                )
            )

            return self.sample_by_index(
                names=names,
                indexes=(
                    self.sampling_indexes
                ),
                mini_batches=mini_batches,
            )

        def sample_by_index(
            self,
            *,
            names,
            indexes,
            mini_batches,
        ):
            assert mini_batches == 1

            tensors = {
                "states": self._states,
                "log_prob": self._log_prob,
            }

            return [
                [
                    tensors[
                        name
                    ][
                        indexes
                    ]
                    for name in names
                ]
            ]

    # ---------------------------------------------------------
    # Real trainable owner-conditioned critic interface
    # ---------------------------------------------------------

    class BiasCritic(
        torch.nn.Module
    ):
        def __init__(self):
            super().__init__()

            self.bias = (
                torch.nn.Parameter(
                    torch.tensor(0.0)
                )
            )

        def forward_owner(
            self,
            graph,
            *,
            owner_index,
        ):
            del graph
            del owner_index

            return (
                self.bias
                .reshape(
                    1,
                    1,
                )
            )

    # ---------------------------------------------------------
    # Agent skeleton
    # ---------------------------------------------------------

    agent = object.__new__(
        MSSRMAPPO
    )

    agent.possible_agents = list(
        possible_agents
    )

    agent.memories = {
        uid: FakeMemory()
        for uid in possible_agents
    }

    agent.cfg = SimpleNamespace(
        learning_epochs={
            uid: 1
            for uid in possible_agents
        },
        mini_batches={
            uid: 1
            for uid in possible_agents
        },
        ratio_clip={
            uid: 0.2
            for uid in possible_agents
        },
        value_clip={
            uid: 0.0
            for uid in possible_agents
        },
        entropy_loss_scale={
            uid: 0.0
            for uid in possible_agents
        },
        value_loss_scale={
            uid: 1.0
            for uid in possible_agents
        },
        discount_factor={
            uid: 0.0
            for uid in possible_agents
        },
        gae_lambda={
            uid: 0.95
            for uid in possible_agents
        },
        time_limit_bootstrap={
            uid: False
            for uid in possible_agents
        },
    )

    # ---------------------------------------------------------
    # Actor parameter used by the real PPO loss
    # ---------------------------------------------------------

    actor_parameter = (
        torch.nn.Parameter(
            torch.tensor(0.0)
        )
    )

    agent.actor_optimizer = (
        torch.optim.SGD(
            [
                actor_parameter
            ],
            lr=0.05,
        )
    )

    # Global critic must stay untouched.
    global_critic_parameter = (
        torch.nn.Parameter(
            torch.tensor(7.0)
        )
    )

    agent.critic_optimizer = (
        torch.optim.SGD(
            [
                global_critic_parameter
            ],
            lr=0.1,
        )
    )

    # ---------------------------------------------------------
    # Two independent APPO critics
    # ---------------------------------------------------------

    agent.altruistic_owner_critic = (
        BiasCritic()
    )

    agent.altruistic_mean_field_critic = (
        BiasCritic()
    )

    agent.owner_critic_optimizer = (
        torch.optim.SGD(
            agent
            .altruistic_owner_critic
            .parameters(),
            lr=0.05,
        )
    )

    agent.mean_field_critic_optimizer = (
        torch.optim.SGD(
            agent
            .altruistic_mean_field_critic
            .parameters(),
            lr=0.05,
        )
    )

    agent.altruistic_alpha = 1.0

    # ---------------------------------------------------------
    # Structured rollout: 2 timesteps x 1 env
    # ---------------------------------------------------------

    structured = (
        StructuredRolloutSidecar()
    )

    for timestep in range(2):
        structured.append_step(
            [
                StructuredRolloutSample(
                    action=None,
                    hard_mask=(),
                    owner_active_mask={
                        "module_0": True,
                        "module_1": True,
                    },
                    routing_snapshot=None,
                    graph_snapshot={
                        "t": timestep,
                    },
                    critic_graph_snapshot={
                        "t": timestep,
                    },
                )
            ]
        )

    agent._structured_rollout_sidecar = (
        structured
    )

    # ---------------------------------------------------------
    # Altruistic rollout
    # ---------------------------------------------------------

    altruistic = (
        AltruisticRolloutSidecar()
    )

    for timestep, (
        reward_0,
        reward_1,
    ) in enumerate(
        (
            (
                1.0,
                3.0,
            ),
            (
                2.0,
                4.0,
            ),
        )
    ):
        samples = (
            build_altruistic_rollout_batch(
                possible_agents=(
                    possible_agents
                ),
                module_ids_by_row=(
                    (
                        "physical_0",
                        "physical_1",
                    ),
                ),
                owner_to_module_id=(
                    {
                        "module_0":
                            "physical_0",
                        "module_1":
                            "physical_1",
                    },
                ),
                module_positions_by_id=(
                    {
                        "physical_0":
                            torch.tensor(
                                [
                                    0.0,
                                    0.0,
                                    0.0,
                                ]
                            ),
                        "physical_1":
                            torch.tensor(
                                [
                                    0.2,
                                    0.0,
                                    0.0,
                                ]
                            ),
                    },
                ),
                owner_rewards={
                    "module_0":
                        torch.tensor(
                            [[reward_0]]
                        ),
                    "module_1":
                        torch.tensor(
                            [[reward_1]]
                        ),
                },
                radius_m=0.5,
                next_module_ids_by_row=(
                    (
                        "physical_1",
                        "physical_0",
                    ),
                ),
                next_owner_to_module_id=(
                    {
                        "module_0":
                            "physical_0",
                        "module_1":
                            "physical_1",
                    },
                ),
            )
        )

        altruistic.append_step(
            samples
        )

    agent._altruistic_rollout_sidecar = (
        altruistic
    )

    agent._current_next_critic_graph = [
        {
            "final": True,
        }
    ]

    # ---------------------------------------------------------
    # Actor reevaluation:
    # keep the existing PPO machinery, but avoid requiring
    # a full ActionSet actor in this focused integration test.
    # ---------------------------------------------------------

    def reevaluate(
        *,
        structured_batch,
        owner_tensor_batch,
        evaluate_global_critic=True,
    ):
        del owner_tensor_batch

        assert (
            evaluate_global_critic
            is False
        )

        batch = len(
            structured_batch
        )

        new_log_probs = {
            uid: (
                actor_parameter
                .reshape(
                    1,
                    1,
                )
                .expand(
                    batch,
                    1,
                )
            )
            for uid in possible_agents
        }

        entropies = {
            uid: torch.zeros(
                batch,
                1,
            )
            for uid in possible_agents
        }

        return (
            new_log_probs,
            entropies,
            None,
        )

    agent.reevaluate_structured_minibatch = (
        reevaluate
    )

    # ---------------------------------------------------------
    # Run the actual complete APPO branch.
    # ---------------------------------------------------------

    MSSRMAPPO.coordinated_altruistic_update(
        agent,
        timestep=10,
        timesteps=100,
    )

    # Actor learned.
    assert (
        actor_parameter.item()
        > 0.0
    )

    # Both independent critics learned.
    assert (
        agent
        .altruistic_owner_critic
        .bias
        .item()
        != 0.0
    )

    assert (
        agent
        .altruistic_mean_field_critic
        .bias
        .item()
        != 0.0
    )

    # Team/global critic is untouched in altruistic mode.
    assert (
        global_critic_parameter.item()
        == 7.0
    )
