"""Actual optimizer/checkpoint evidence for the existing fixed-alpha learner."""
import math

import pytest
import torch
from skrl.models.torch import Model

from drl.MARL.mappo import MSSRMAPPO, StructuredRolloutSample
from drl.MARL.altruistic_rollout import AltruisticRolloutSidecar, build_altruistic_rollout_batch


class BiasModel(Model):
    def __init__(self, value=1.):
        super().__init__(device="cpu")
        self.weight = torch.nn.Parameter(torch.tensor(float(value)))

    def compute(self, inputs, role):
        return self.weight.reshape(1, 1), {}

    def act(self, inputs, role=""):
        return self.compute(inputs, role)

    def forward_owner(self, graph, *, owner_index):
        return self.weight.reshape(1, 1)


def make_agent():
    actor, global_critic = BiasModel(0.), BiasModel(7.)
    agent = MSSRMAPPO(possible_agents=["m0", "m1"],
        models=MSSRMAPPO.build_shared_model_registry(possible_agents=["m0", "m1"], actor=actor, critic=global_critic),
        device="cpu", cfg={"learning_epochs": 1, "mini_batches": 1, "discount_factor": 0.,
                           "entropy_loss_scale": 0., "learning_rate": (0.05, .05), "value_clip": 0.,
            **{name: {uid: {} for uid in ("m0", "m1")} for name in
               ("learning_rate_scheduler_kwargs", "observation_preprocessor_kwargs", "state_preprocessor_kwargs", "value_preprocessor_kwargs")}})
    agent.configure_fixed_alpha_appo(owner_critic=BiasModel(0.), mean_field_critic=BiasModel(0.), alpha=.5)
    return agent


def test_checkpoint_roundtrip_keeps_each_model_optimizer_once(tmp_path):
    agent = make_agent()
    weights = [agent.shared_actor.weight, agent.altruistic_owner_critic.weight, agent.altruistic_mean_field_critic.weight]
    MSSRMAPPO.coordinated_altruistic_optimizer_step(owner_policy_losses={"m0": -weights[0]},
        owner_value_loss=(weights[1]-2).square(), mean_field_value_loss=(weights[2]-3).square(),
        actor_optimizer=agent.actor_optimizer, owner_critic_optimizer=agent.owner_critic_optimizer,
        mean_field_critic_optimizer=agent.mean_field_critic_optimizer)
    path = tmp_path / "APPO_SMOKE.pt"
    agent.save(str(path))
    checkpoint = torch.load(path, weights_only=False)
    model_keys = [key for modules in checkpoint.values() for key in modules if key in {"policy", "value", "owner_critic", "mean_field_critic"}]
    assert sorted(model_keys) == ["mean_field_critic", "owner_critic", "policy", "value"]
    restored = make_agent()
    restored.load(str(path))
    assert restored.shared_actor.weight.item() == pytest.approx(weights[0].item())
    assert restored.shared_critic.weight.item() == 7.
    assert restored.altruistic_owner_critic.weight.item() == pytest.approx(weights[1].item())
    assert restored.altruistic_mean_field_critic.weight.item() == pytest.approx(weights[2].item())
    for optimizer in (restored.actor_optimizer, restored.owner_critic_optimizer, restored.mean_field_critic_optimizer):
        assert len(optimizer.state) == 1
        assert next(iter(optimizer.state.values()))["step"].item() == 1


def test_optimizer_diagnostics_measure_actual_pre_step_gradient_norms():
    params = [torch.nn.Parameter(torch.tensor(1.)) for _ in range(3)]
    optimizers = [torch.optim.SGD([p], lr=.1) for p in params]
    result = MSSRMAPPO.coordinated_altruistic_optimizer_step(owner_policy_losses={"m0": 2*params[0]},
        owner_value_loss=2*params[1].square(), mean_field_value_loss=3*params[2].square(),
        actor_optimizer=optimizers[0], owner_critic_optimizer=optimizers[1], mean_field_critic_optimizer=optimizers[2])
    assert result["actor_grad_norm"] == pytest.approx(2.)
    assert result["owner_critic_grad_norm"] == pytest.approx(4.)
    assert result["mean_field_critic_grad_norm"] == pytest.approx(6.)
    assert result["actor_optimizer_steps"] == result["owner_critic_optimizer_steps"] == result["mean_field_critic_optimizer_steps"] == 1
    assert [p.item() for p in params] == pytest.approx([.8, .6, .4])


class Memory:
    """Small deterministic tensor store; optimization/GAE remain production code."""
    def __init__(self):
        self.sampling_indexes = torch.tensor([1, 0])

    def __len__(self):
        return 2

    def get_tensor_by_name(self, name):
        return torch.zeros(2, 1, 1, dtype=torch.bool)

    def sample(self, *, names, batch_size, mini_batches):
        return self.sample_by_index(names=names, indexes=self.sampling_indexes, mini_batches=mini_batches)

    def sample_by_index(self, *, names, indexes, mini_batches):
        return [[torch.zeros(2, 1) for _ in names]]


def test_full_fixed_alpha_update_exposes_finite_losses_policy_diagnostics_and_steps():
    agent = make_agent()
    agent.memories = {uid: Memory() for uid in agent.possible_agents}
    agent._altruistic_rollout_sidecar = AltruisticRolloutSidecar()
    for step in range(2):
        agent._structured_rollout_sidecar.append_step([StructuredRolloutSample(action=None, hard_mask=(),
            owner_active_mask={"m0": True, "m1": True}, routing_snapshot=None, graph_snapshot={}, critic_graph_snapshot={})])
        agent._altruistic_rollout_sidecar.append_step(build_altruistic_rollout_batch(
            possible_agents=agent.possible_agents, module_ids_by_row=(("p0", "p1"),),
            owner_to_module_id=({"m0": "p0", "m1": "p1"},),
            module_positions_by_id=({"p0": torch.tensor([0., 0., 0.]), "p1": torch.tensor([.2, 0., 0.])},),
            owner_rewards={"m0": torch.tensor([[1.+step]]), "m1": torch.tensor([[3.+step]])}, radius_m=.5,
            next_module_ids_by_row=(("p0", "p1"),), next_owner_to_module_id=({"m0": "p0", "m1": "p1"},)))
    agent._current_next_critic_graph = [{}]
    # A trainable scalar actor isolates diagnostics from graph/action encoding.
    def reevaluate(*, structured_batch, owner_tensor_batch, evaluate_global_critic):
        return ({uid: agent.shared_actor.weight.reshape(1, 1).expand(2, 1) for uid in agent.possible_agents},
                {uid: torch.full((2, 1), .7) for uid in agent.possible_agents}, None)
    agent.reevaluate_structured_minibatch = reevaluate
    agent.coordinated_update(timestep=2, timesteps=2)
    metrics = agent.last_update_metrics
    assert metrics["actor_optimizer_steps"] == metrics["owner_critic_optimizer_steps"] == metrics["mean_field_critic_optimizer_steps"] == 1
    assert metrics["global_critic_optimizer_steps"] == 0
    assert metrics["actor_loss"] < 0
    assert metrics["owner_critic_loss"] > 0
    assert metrics["mean_field_critic_loss"] > 0
    assert metrics["entropy"] == pytest.approx(.7)
    assert metrics["ratio_mean"] == pytest.approx(1.)
    assert metrics["approx_kl"] == pytest.approx(0.)
    assert metrics["clip_fraction"] == 0.
    assert all(math.isfinite(value) for value in metrics.values())
    assert all(metrics[key] > 0 for key in ("actor_grad_norm", "owner_critic_grad_norm", "mean_field_critic_grad_norm"))
    assert agent.shared_critic.weight.item() == 7.


def test_optimizer_profile_preserves_parameters_gradients_and_event_order():
    def run(profile_sink=None):
        params = [torch.nn.Parameter(torch.tensor(1.)) for _ in range(3)]
        optimizers = [torch.optim.SGD([p], lr=.1) for p in params]
        kwargs = {} if profile_sink is None else {"profile_sink": profile_sink}
        metrics = MSSRMAPPO.coordinated_altruistic_optimizer_step(
            owner_policy_losses={"m0": 2*params[0]}, owner_value_loss=2*params[1].square(),
            mean_field_value_loss=3*params[2].square(), actor_optimizer=optimizers[0],
            owner_critic_optimizer=optimizers[1], mean_field_critic_optimizer=optimizers[2], **kwargs)
        return params, metrics

    baseline, baseline_metrics = run()
    events = []
    profiled, metrics = run(lambda name, seconds: events.append((name, seconds)))
    assert [name for name, _ in events] == [
        "update.actor_backward_optimizer", "update.owner_backward_optimizer",
        "update.mean_field_backward_optimizer"]
    assert all(seconds > 0 for _, seconds in events)
    for original, timed in zip(baseline, profiled, strict=True):
        assert torch.equal(original, timed)
        assert torch.equal(original.grad, timed.grad)
    for key in ("actor_grad_norm", "owner_critic_grad_norm", "mean_field_critic_grad_norm"):
        assert metrics[key] == baseline_metrics[key]
