"""Actual SKRL eval trainer with a stub native bank, without Isaac."""
import json
from types import SimpleNamespace

import pytest
import torch

from drl.MARL.isaaclab_c1_evaluation import run_native_evaluation, evaluation_protocol


class BaseLearner:
    def post_interaction(self, **kwargs):
        assert not self.training


class Learner(BaseLearner):
    def __init__(self, bad=None):
        self.shared_actor = torch.nn.Linear(1, 1)
        self.shared_critic = torch.nn.Linear(1, 1)
        self.altruistic_owner_critic = torch.nn.Linear(1, 1)
        self.altruistic_mean_field_critic = torch.nn.Linear(1, 1)
        self._structured_rollout_sidecar = []
        self._altruistic_rollout_sidecar = []
        self.memory = []
        self.memories = {'owner': SimpleNamespace(reset=self.memory.clear)}
        self._current_next_critic_graph = object()
        self.context = self._current_next_critic_graph
        self.bad = bad
        self.calls = 0
    def init(self, **kwargs):
        pass
    def enable_training_mode(self, mode):
        self.training = mode
    def pre_interaction(self, **kwargs):
        pass
    def track_data(self, *args):
        pass
    def act(self, observations, states, **kwargs):
        assert self._current_next_critic_graph is self.context
        assert self.deterministic_policy_actions
        return torch.ones(2, 1), {'mean_actions': torch.zeros(2, 1)}
    def record_transition(self, **kwargs):
        assert not self.memory and not self._structured_rollout_sidecar and not self._altruistic_rollout_sidecar
        self.memory.append(1)
        self._structured_rollout_sidecar.append(1)
        self._altruistic_rollout_sidecar.append(1)
        self.calls += 1
        if self.bad == 'update':
            self.coordinated_update()
        if self.bad == 'mutate':
            self.altruistic_mean_field_critic.weight.add_(1)
    def coordinated_update(self, **kwargs):
        raise AssertionError('original optimizer must never run')


def setup(tmp_path, bad=None):
    args = SimpleNamespace(evaluate=True, evaluation_split='validation', max_vector_steps=3,
        output=tmp_path, train=False, benchmark=False, smoke_train=False, controlled_update=False,
        checkpoint='bc.pt', policy_checkpoint=None)
    env = SimpleNamespace(domain_by_env=('navigation', 'stairs'), seed_by_env=(8000, 13000),
        cfg=SimpleNamespace(resample_train=False, course_split='validation'))
    class Wrapper:
        num_envs = 2
        def __init__(self):
            self.actions_seen = []
            self.resets = 0
        def reset(self):
            self.resets += 1
            return torch.zeros(2, 1), {}
        def state(self):
            return None
        def step(self, actions):
            assert torch.equal(actions, torch.ones(2, 1))  # stochastic actions, not means
            self.actions_seen.append(actions.clone())
            for i, (domain, seed) in enumerate(zip(env.domain_by_env, env.seed_by_env)):
                env.training_telemetry.observe_transition(domain=domain, seed=seed, generation=1,
                    env_index=i, reward=1., terminated=False, truncated=False, success=False, active=True)
            return torch.zeros(2, 1), torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.bool), torch.zeros(2, 1, dtype=torch.bool), {}
    return args, env, Wrapper(), Learner(bad)


def test_actual_skrl_eval_is_frozen_and_clears_storage_and_restores(tmp_path):
    args, env, wrapper, learner = setup(tmp_path)
    original = learner.record_transition, learner.coordinated_update
    result = run_native_evaluation(args, env, wrapper, learner)
    assert result['status'] == 'pass'
    assert result['optimizer_updates'] == 0
    assert result['frozen_parameter_proof']['all_unchanged']
    assert len(result['frozen_parameter_proof']['models']) == 4
    assert result['vector_steps'] == learner.calls == 3
    assert result['row_transitions'] == 6
    assert len(result['seed_bank']) == 2
    assert result['split'] == 'validation'
    assert not learner.memory and not learner._structured_rollout_sidecar and not learner._altruistic_rollout_sidecar
    assert original == (learner.record_transition, learner.coordinated_update)
    assert not hasattr(env, 'training_telemetry')
    assert not hasattr(learner,'deterministic_policy_actions')
    assert result['stochastic_evaluation'] is False
    assert json.loads((tmp_path / 'result.json').read_text()) == result


@pytest.mark.parametrize('bad', ['update', 'mutate'])
def test_eval_fails_closed_on_update_attempt_or_model_mutation(tmp_path, bad):
    args, env, wrapper, learner = setup(tmp_path, bad)
    original = learner.record_transition, learner.coordinated_update
    with pytest.raises(RuntimeError):
        run_native_evaluation(args, env, wrapper, learner)
    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['status'] == 'fail'
    assert original == (learner.record_transition, learner.coordinated_update)


@pytest.mark.parametrize('bad', ['seed', 'split', 'train', 'resample', 'limit', 'evaluate'])
def test_invalid_configuration_rejected_before_actions(tmp_path, bad):
    args, env, wrapper, learner = setup(tmp_path)
    if bad == 'seed': env.seed_by_env = (7000, 13000)
    if bad == 'split': args.evaluation_split = 'other'
    if bad == 'train': args.train = True
    if bad == 'resample': env.cfg.resample_train = True
    if bad == 'limit': args.max_vector_steps = float('inf')
    if bad == 'evaluate': args.evaluate = False
    with pytest.raises(ValueError):
        run_native_evaluation(args, env, wrapper, learner)
    assert not wrapper.actions_seen and wrapper.resets == 0


def test_protocol_preserves_exact_paired_bank_and_split_roles():
    bank = [dict(domain='navigation', seed=8000, split='validation')]
    protocol = evaluation_protocol(bank)
    assert protocol['seed_bank'] == bank
    assert protocol['train']['optimizer_updates_allowed']
    assert protocol['validation']['model_selection']
    assert not protocol['test']['tuning_allowed']
    assert protocol['challenge']['independent_stress']


def test_missing_reward_telemetry_cannot_claim_evaluation_pass(tmp_path):
    args, env, wrapper, learner = setup(tmp_path)
    original_step = wrapper.step
    def no_telemetry(actions):
        actual = env.training_telemetry
        env.training_telemetry = SimpleNamespace(observe_transition=lambda **kwargs: None)
        try:
            return original_step(actions)
        finally:
            env.training_telemetry = actual
    wrapper.step = no_telemetry
    with pytest.raises(RuntimeError, match='telemetry'):
        run_native_evaluation(args, env, wrapper, learner)
    assert json.loads((tmp_path / 'result.json').read_text())['status'] == 'fail'
