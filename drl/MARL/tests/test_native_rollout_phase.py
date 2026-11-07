from types import SimpleNamespace

import torch
import pytest

from drl.MARL.mappo import MSSRMAPPO, OnlineRolloutContext
from drl.MARL.tests.test_mssr_mappo import _make_standard_bridge_agent


def test_mixed_native_act_retains_sample_but_excludes_assembly_owner_activity():
    agent = _make_standard_bridge_agent(num_envs=2)
    encoded = []

    class Provider:
        def current_action_contexts(self):
            return [OnlineRolloutContext(
                actor_graph={'env': i}, critic_graph={'env': i, 'value': 2.5},
                routing_snapshot={'env': i}, hard_mask=torch.tensor([False]),
                rollout_active=i == 0,
            ) for i in range(2)]

        def encode_env_actions(self, decisions):
            encoded.extend(decisions)
            return {uid: torch.zeros(2, 1) for uid in agent.possible_agents}

    agent.set_rollout_context_provider(Provider())
    observations = {uid: torch.zeros(2, 3) for uid in agent.possible_agents}
    agent.act(observations, observations, timestep=0, timesteps=4)
    assert encoded[0].structured_sample.rollout_active
    assert not encoded[1].structured_sample.rollout_active
    assert any(encoded[0].structured_sample.owner_active_mask.values())
    assert not any(encoded[1].structured_sample.owner_active_mask.values())
    assert encoded[1].global_value.item() == 2.5
    assert encoded[1].structured_sample.action is encoded[1].action


def test_owner_critic_loss_masks_assembly_rows_without_resource_activity_mask():
    predicted = torch.tensor([[1.0], [1000.0]], requires_grad=True)
    loss = MSSRMAPPO.compute_owner_conditioned_value_loss(
        predicted_values={'m0': predicted}, sampled_values={'m0': torch.zeros(2, 1)},
        sampled_returns={'m0': torch.tensor([[3.0], [-1000.0]])},
        value_clip=0.0, value_loss_scale=1.0, sample_mask=torch.tensor([True, False]),
    )
    assert loss.item() == 4.0
    loss.backward()
    assert predicted.grad.tolist() == [[-4.0], [0.0]]


def test_all_inactive_value_loss_has_no_gradient_contribution():
    predicted = torch.tensor([[1.0], [1000.0]], requires_grad=True)
    loss = MSSRMAPPO.compute_owner_conditioned_value_loss(
        predicted_values={'m0': predicted}, sampled_values={'m0': torch.zeros(2, 1)},
        sampled_returns={'m0': torch.ones(2, 1)},
        value_clip=0.2, value_loss_scale=1.0, sample_mask=torch.zeros(2, dtype=torch.bool),
    )
    assert loss.item() == 0
    loss.backward()
    assert predicted.grad.tolist() == [[0.0], [0.0]]


def test_native_post_interaction_preserves_rollout_distribution_with_dropout_and_gradients():
    from skrl.models.torch import Model

    class DropoutPolicy(Model):
        def __init__(self):
            super().__init__(device='cpu')
            self.weight = torch.nn.Parameter(torch.ones(64))
            self.dropout = torch.nn.Dropout(.5)

        def compute(self, inputs, role):
            return self.dropout(self.weight), {}

        def act(self, inputs, role=''):
            return self.compute(inputs, role)

    actor, critic = DropoutPolicy(), DropoutPolicy()
    agent = MSSRMAPPO(possible_agents=['m0'],
        models=MSSRMAPPO.build_shared_model_registry(possible_agents=['m0'], actor=actor, critic=critic),
        device='cpu', deterministic_update_mode=True,
        cfg=dict(rollouts=1, learning_starts=0, learning_rate=(.01, .01),
                 **{name: {'m0': {}} for name in (
                     'learning_rate_scheduler_kwargs', 'observation_preprocessor_kwargs',
                     'state_preprocessor_kwargs', 'value_preprocessor_kwargs')}))
    owner, mean_field = DropoutPolicy(), DropoutPolicy()
    assert agent.skip_expert_policy_inference is False
    agent.configure_fixed_alpha_appo(owner_critic=owner, mean_field_critic=mean_field, alpha=.5)
    for model in (actor, critic, owner, mean_field):
        model.eval()
    with torch.no_grad():
        rollout_distribution = torch.distributions.Bernoulli(logits=actor.compute({}, '')[0])
        action = torch.ones(64)
        old_log_prob = rollout_distribution.log_prob(action)
    initial_weight = actor.weight.detach().clone()
    called = []

    def update(**kwargs):
        called.append(kwargs)
        assert torch.is_grad_enabled()
        distribution = torch.distributions.Bernoulli(logits=actor.compute({}, '')[0])
        ratio = (distribution.log_prob(action) - old_log_prob).exp()
        assert torch.equal(ratio, torch.ones_like(ratio))
        assert all(not model.training for model in (actor, critic, owner, mean_field))
        MSSRMAPPO.coordinated_altruistic_optimizer_step(
            owner_policy_losses={'m0': -ratio.mean()},
            owner_value_loss=owner.compute({}, '')[0].square().mean(),
            mean_field_value_loss=mean_field.compute({}, '')[0].square().mean(),
            actor_optimizer=agent.actor_optimizer, owner_critic_optimizer=agent.owner_critic_optimizer,
            mean_field_critic_optimizer=agent.mean_field_critic_optimizer)

    agent.coordinated_update = update
    agent.training, agent._rollout = True, 0
    agent.post_interaction(timestep=0, timesteps=2)
    assert len(called) == 1
    assert actor.weight.grad is not None and actor.weight.grad.abs().sum().item() > 0
    assert not torch.equal(actor.weight.detach(), initial_weight)


def _expert_skip_agent(active_flags):
    from dataclasses import replace
    agent = _make_standard_bridge_agent(num_envs=len(active_flags))
    agent.skip_expert_policy_inference = True
    agent.device = torch.device('cpu')
    weight = torch.nn.Parameter(torch.tensor(.3))
    actor_calls, critic_calls, encoded = [], [], []
    actor_forward = agent.shared_actor.forward_all_owners
    critic_forward = agent.shared_critic.act
    def actor(**kwargs):
        assert kwargs['graph']['active']
        actor_calls.append(kwargs['graph']['env'])
        output = actor_forward(**kwargs)
        output['records'] = [replace(record, presence_logit=weight) for record in output['records']]
        return output
    def critic(inputs, **kwargs):
        assert inputs['states']['active']
        critic_calls.append(inputs['states']['env'])
        return critic_forward(inputs, **kwargs)
    agent.shared_actor.forward_all_owners = actor
    agent.shared_critic.act = critic
    contexts = [OnlineRolloutContext(actor_graph={'env': i, 'active': active},
        critic_graph={'env': i, 'active': active, 'value': 2.5}, routing_snapshot={'row': i},
        hard_mask=torch.tensor([False]), rollout_active=active) for i, active in enumerate(active_flags)]
    class Provider:
        def current_action_contexts(self):
            return contexts
        def current_critic_graphs(self):
            return [dict(context.critic_graph, next=True) for context in contexts]
        def encode_env_actions(self, decisions):
            encoded.extend(decisions)
            return {uid: torch.zeros(len(contexts), 2) for uid in agent.possible_agents}
    agent.set_rollout_context_provider(Provider())
    return agent, weight, actor_calls, critic_calls, encoded, contexts


@pytest.mark.parametrize('active_flags', [[False, False], [True, False]])
def test_opt_in_expert_rows_skip_models_rng_and_preserve_real_rollout_storage(active_flags):
    agent, _, actor_calls, critic_calls, encoded, contexts = _expert_skip_agent(active_flags)
    observations = {uid: torch.zeros(2, 3) for uid in agent.possible_agents}
    rng = torch.random.get_rng_state().clone()
    actions, outputs = agent.act(observations, observations, timestep=0, timesteps=2)
    if not any(active_flags):
        assert torch.equal(rng, torch.random.get_rng_state())
    assert len(actor_calls) == len(critic_calls) == sum(active_flags)
    for decision, context in zip(encoded, contexts):
        assert decision.structured_sample.graph_snapshot is context.actor_graph
        assert decision.structured_sample.critic_graph_snapshot is context.critic_graph
        assert decision.structured_sample.routing_snapshot is context.routing_snapshot
        assert torch.equal(decision.structured_sample.hard_mask, context.hard_mask)
        if context.rollout_active:
            assert decision.action is not None
            assert decision.global_value.item() == 2.5
            continue
        assert decision.action is None
        assert decision.global_value.item() == 0.
        assert not decision.structured_sample.rollout_active
        assert not any(decision.structured_sample.owner_active_mask.values())
    zeros = {uid: torch.zeros(2, 1) for uid in agent.possible_agents}
    dones = {uid: torch.tensor([[True], [False]]) for uid in agent.possible_agents}
    states = {uid: torch.full((2, 4), 7.) for uid in agent.possible_agents}
    agent.record_transition(observations=observations, states=states, actions=actions,
        rewards=zeros, next_observations=observations, next_states=states,
        terminated=dones, truncated={uid: torch.zeros(2, 1, dtype=torch.bool) for uid in agent.possible_agents},
        infos={}, timestep=0, timesteps=2)
    assert len(agent._structured_rollout_sidecar) == 2
    for memory in agent.memories.values():
        assert len(memory) == 2
        assert memory.get_tensor_by_name('terminated')[0, :, 0].tolist() == [True, False]
        assert memory.get_tensor_by_name('values')[0, :, 0].tolist() == [2.5 if active else 0. for active in active_flags]
        assert torch.equal(memory.get_tensor_by_name('states')[0], states[agent.possible_agents[0]])
    for index, active in enumerate(active_flags):
        if not active:
            assert all(output['log_prob'][index].item() == 0. for output in outputs.values())


def test_mixed_neutral_rows_preserve_active_sampling_and_policy_gradients():
    def run(flags):
        agent, weight, actor_calls, critic_calls, encoded, _ = _expert_skip_agent(flags)
        observations = {uid: torch.zeros(len(flags), 3) for uid in agent.possible_agents}
        torch.manual_seed(819)
        agent.act(observations, observations, timestep=0, timesteps=2)
        assert len(actor_calls) == len(critic_calls) == sum(flags)
        samples = [decision.structured_sample for decision in encoded]
        old = {uid: torch.stack([decision.owner_log_probs[uid].reshape(1) for decision in encoded])
               for uid in agent.possible_agents}
        new, entropies, values = agent.reevaluate_structured_minibatch(
            structured_batch=samples, owner_tensor_batch={}, evaluate_global_critic=False)
        assert values is None
        assert len(actor_calls) == 2*sum(flags)
        masks, _ = agent.build_owner_activity_masks(structured_batch=samples, possible_agents=agent.possible_agents)
        advantages = {uid: torch.ones(len(flags), 1) for uid in agent.possible_agents}
        losses = agent.compute_owner_actor_losses(new_log_probs=new, old_log_probs=old,
            advantages=advantages, owner_entropies=entropies, ratio_clip=.2,
            entropy_loss_scale=0., owner_active_masks=masks)
        sum(losses.values()).backward()
        assert torch.isfinite(weight.grad)
        for index, active in enumerate(flags):
            if not active:
                assert all(new[uid][index].item() == entropies[uid][index].item() == 0. for uid in new)
        return weight.grad, {uid: new[uid][torch.tensor(flags)].detach() for uid in new}
    baseline_gradient, baseline_probs = run([True, True])
    mixed_gradient, mixed_probs = run([True, False, True, False])
    torch.testing.assert_close(mixed_gradient, baseline_gradient)
    for uid in baseline_probs:
        torch.testing.assert_close(mixed_probs[uid], baseline_probs[uid])
