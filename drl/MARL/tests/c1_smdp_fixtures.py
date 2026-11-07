"""Small CPU collector fixtures; no simulator or campaign models."""
from types import SimpleNamespace

import torch

from drl.MARL.altruistic_rollout import AltruisticRolloutSample, AltruisticRolloutSidecar
from drl.MARL.mappo import (
    MSSRMAPPO, SMDPEventMemory, OnlineRolloutDecision, StructuredRolloutSample, StructuredRolloutSidecar,
)


class ScalarCritic:
    def __init__(self, scale=1.):
        self.scale = scale

    def forward_all_owners(self, graph):
        return self.scale * graph


def make_learner(capacity=2, num_envs=2):
    learner = MSSRMAPPO.__new__(MSSRMAPPO)
    learner.possible_agents = ['a']
    learner.cfg = SimpleNamespace(discount_factor={'a': .9}, gae_lambda={'a': .95},
        rollouts=capacity, learning_starts=0, grad_norm_clip=.5)
    memory = SMDPEventMemory(memory_size=capacity, num_envs=num_envs, device='cpu', replacement=False)
    for name in ('observations', 'states', 'actions', 'rewards', 'log_prob', 'values'):
        memory.create_tensor(name=name, size=1, dtype=torch.float32)
    for name in ('terminated', 'truncated'):
        memory.create_tensor(name=name, size=1, dtype=torch.bool)
    learner.memories = {'a': memory}
    learner._structured_rollout_sidecar = StructuredRolloutSidecar()
    learner._altruistic_rollout_sidecar = AltruisticRolloutSidecar()
    learner.configure_smdp_options()
    learner._smdp_transition_count = learner._smdp_event_step_count = 0
    learner._rollout = 0
    learner.training = True
    learner.deterministic_update_mode = False
    learner.altruistic_owner_critic = ScalarCritic()
    learner.altruistic_mean_field_critic = ScalarCritic(3.)
    learner.owner_critic_optimizer = learner.mean_field_critic_optimizer = None
    learner.altruistic_alpha = .5
    learner.enable_models_training_mode = lambda *_: None
    learner.track_data = lambda *_: None
    return learner


def frame(learner, active, state, reward, *, terminated=None, truncated=None):
    n = len(active)
    decisions = []
    for value, enabled in zip(state, active):
        graph = torch.tensor([value], dtype=torch.float32)
        sample = StructuredRolloutSample(action=None, hard_mask=torch.zeros(0, dtype=torch.bool),
            owner_active_mask={'a': enabled}, routing_snapshot=None, graph_snapshot=graph,
            critic_graph_snapshot=graph, rollout_active=enabled)
        decisions.append(OnlineRolloutDecision(action=None, owner_log_probs={'a': torch.zeros(1)},
            global_value=graph.reshape(1, 1), structured_sample=sample))
    samples = [AltruisticRolloutSample(owner_row_index={'a': 0},
        owner_rewards={'a': torch.tensor(r, dtype=torch.float32)},
        mean_field_rewards={'a': torch.tensor(2*r, dtype=torch.float32)},
        neighbourhoods={'a': ()}, next_owner_row_index={'a': 0}) for r in reward]
    learner._current_online_decisions = decisions
    learner._build_altruistic_rollout_step = lambda **_: samples
    obs = {'a': torch.tensor(state, dtype=torch.float32).reshape(n, 1)}
    nxt = {'a': obs['a'] + 1}
    learner._record_smdp_transition(observations=obs, states=obs,
        actions={'a': torch.zeros(n, 1)}, next_observations=nxt, next_states=nxt,
        terminated=torch.tensor(terminated or [False]*n).reshape(n, 1),
        truncated=torch.tensor(truncated or [False]*n).reshape(n, 1),
        decisions=decisions, next_critic_graphs=[torch.tensor([s+1.]) for s in state],
        team_rewards=torch.tensor(reward, dtype=torch.float32).reshape(n, 1))


def capture_update_targets(learner):
    """Run the real update target path, stop before loss/optimizer processing."""
    class Captured(Exception):
        pass
    result = {}
    def capture(**kwargs):
        result['targets'] = MSSRMAPPO.compute_altruistic_rollout_targets(**kwargs)
        result['owner_bootstrap'] = kwargs['owner_bootstrap_values']
        result['mf_bootstrap'] = kwargs['mean_field_bootstrap_values']
        raise Captured
    learner.compute_altruistic_rollout_targets = capture
    try:
        learner.coordinated_altruistic_update(timestep=0, timesteps=10)
    except Captured:
        return result
    raise AssertionError('Target construction was not reached')
