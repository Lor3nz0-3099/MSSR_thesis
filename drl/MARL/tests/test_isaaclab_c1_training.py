from pathlib import Path
from types import SimpleNamespace
import json
import time

import pytest
import torch


@pytest.mark.parametrize('use_real_trainer', [False, True])
@pytest.mark.parametrize('failure_mode', [None, 'nonfinite', 'alignment', 'inactive'])
def test_harness_uses_trainer_and_stops_after_update_and_restores_hooks(tmp_path, monkeypatch, use_real_trainer, failure_mode):
    from drl.MARL import isaaclab_c1_training as training
    class Learner:
        def __init__(self):
            self.shared_actor = torch.nn.Linear(1, 1)
            self._rollout = 0
            self.cfg = SimpleNamespace(rollouts=2, learning_starts=0)
            self.training = True
            self.write_interval = self.checkpoint_interval = 0
            self._structured_rollout_sidecar = SimpleNamespace(_samples=[])
            self._altruistic_rollout_sidecar = SimpleNamespace(flatten=lambda:
                [] if failure_mode == 'alignment' else self._structured_rollout_sidecar._samples)
            self._structured_rollout_sidecar.clear = self._structured_rollout_sidecar._samples.clear
            self._altruistic_rollout_sidecar.clear = lambda: None
            self.memories = {}
            self.last_update_metrics = {}
        def init(self, **kwargs):
            pass
        def enable_training_mode(self, mode):
            pass
        def enable_models_training_mode(self, mode):
            pass
        def pre_interaction(self, **kwargs):
            pass
        def track_data(self, *args):
            pass
        def act(self, *args, **kwargs):
            return torch.zeros(2, 1), {}
        def record_transition(self, **kwargs):
            self._structured_rollout_sidecar._samples.extend([SimpleNamespace(
                action=torch.ones(1), graph_snapshot={'x': torch.tensor([float('nan')]) if failure_mode == 'nonfinite' else torch.ones(1)}, critic_graph_snapshot=None,
                owner_active_mask={'m': failure_mode != 'inactive'}) for _ in range(2)])
        def coordinated_update(self, **kwargs):
            if failure_mode == 'inactive':
                self.last_update_metrics = {'minibatches': 0, 'actor_optimizer_steps': 0}
                return
            with torch.no_grad():
                self.shared_actor.weight.add_(1.)
            self.last_update_metrics = {'minibatches': 1, 'actor_loss': 1., 'actor_optimizer_steps': 1}
        def post_interaction(self, **kwargs):
            if failure_mode == 'inactive':
                from drl.MARL.mappo import MSSRMAPPO
                return MSSRMAPPO.post_interaction(self, **kwargs)
            self._rollout += 1
            if self._rollout % 2 == 0:
                self.coordinated_update(**kwargs)
                self._structured_rollout_sidecar._samples.clear()
        def save(self, path):
            Path(path).write_text('checkpoint')
    cache_metrics=dict(totals=dict(mask_cache_hits=7,mask_cache_misses=1,
        mask_recomputes=1,feasibility_masking_s=.125),by_env={'0':dict(mask_cache_hits=7)})
    env = SimpleNamespace(assembly_records=[], tasks=[SimpleNamespace(domain='gap')],
        runtime_context_provider=SimpleNamespace(feasibility_metrics=lambda:cache_metrics))
    class Wrapper:
        num_envs = 2
        def reset(self):
            return torch.zeros(2, 1), {}
        def state(self):
            return None
        def step(self, actions):
            for index in range(2):
                env.training_telemetry.observe_transition(domain='gap', seed=7+index, generation=0,
                    env_index=index, reward=1., terminated=False, truncated=False, success=False, active=failure_mode != 'inactive')
            return torch.zeros(2, 1), torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.bool), torch.zeros(2, 1, dtype=torch.bool), {}
    learner, wrapper = Learner(), Wrapper()
    originals = learner.act, learner.post_interaction, wrapper.step
    class Trainer:
        def __init__(self, *, env, agents, cfg):
            assert cfg['timesteps'] == 10
            assert cfg['disable_progressbar'] and not cfg['close_environment_at_exit']
            self.env, self.agents = env, agents
        def train(self):
            for index in range(10):
                self.env.step(self.agents.act())
                self.agents.record_transition()
                self.agents.post_interaction(timestep=index, timesteps=10)
    if not use_real_trainer:
        monkeypatch.setattr(training, 'SequentialTrainer', Trainer)
    args = SimpleNamespace(output=tmp_path, total_transitions=4, max_vector_steps=10,
        rollout_steps=2, learning_epochs=1, mini_batches=1, device='cpu', num_envs=2,
        benchmark=True, smoke_train=False, train=False, checkpoint_every_transitions=0)
    result = training.run_native_training(args, env, wrapper, learner,
        startup_s=.1, assembly_s=.2, run_started=time.monotonic())
    assert result['status'] == ('fail' if failure_mode else 'pass')
    assert result['diagnostics']['feasibility_cache']==cache_metrics
    assert result['checkpoint_bytes'] == len('checkpoint')
    assert Path(result['checkpoint_path']).is_file()
    assert originals == (learner.act, learner.post_interaction, wrapper.step)
    json.loads((tmp_path / 'result.json').read_text())
    if failure_mode == 'inactive':
        assert result['active_marl_transitions'] == 0
        assert result['vector_steps'] == 10
        assert result['optimizer']['completed_ppo_updates'] == 0
        assert result['optimizer']['inactive_update_calls'] == 5
        assert result['error'] is None
        assert result['stop_reason'] == 'vector_step_limit'
        return
    assert result['active_marl_transitions'] == 4
    if failure_mode:
        assert result['error']['type'] == 'ValueError'
        assert result['stop_reason'] == 'error'
        return
    assert result['optimizer']['completed_ppo_updates'] == 1
    assert result['diagnostics']['actor_weights_changed']
    assert result['diagnostics']['sidecar_alignment_checks'] == 1
    assert len((tmp_path / 'updates.jsonl').read_text().splitlines()) == 1


def test_smoke_coverage_requires_distinct_ended_seeds_in_each_domain():
    from drl.MARL.isaaclab_c1_training import _coverage_ready
    episodes = [dict(domain=domain, seed=seed) for domain in ('gap', 'stairs') for seed in range(4)]
    assert _coverage_ready(episodes, ['gap', 'stairs'], 4)
    assert not _coverage_ready(episodes, ['gap', 'stairs', 'manipulation'], 4)
    assert not _coverage_ready(episodes + episodes, ['gap', 'stairs'], 5)


def _campaign_fixture(tmp_path, monkeypatch, *, inactive=False, train=True):
    """Small real-SKRL environment with a controlled process wall clock."""
    from drl.MARL import isaaclab_c1_training as training
    clock = {'elapsed': 100.}
    monkeypatch.setattr(training.time, 'monotonic', lambda: clock['elapsed'])
    env = SimpleNamespace(tasks=[SimpleNamespace(domain='gap')], assembly_records=[])
    class Learner:
        def __init__(self):
            self.shared_actor = torch.nn.Linear(1, 1)
            self.actor_optimizer = torch.optim.SGD(self.shared_actor.parameters(), lr=.1)
            self.optimizers = {'actor': self.actor_optimizer}
            self.owner_critic = torch.nn.Linear(1, 1)
            self.mean_field_critic = torch.nn.Linear(1, 1)
            self.owner_critic_optimizer = torch.optim.SGD(self.owner_critic.parameters(), lr=.1)
            self.mean_field_critic_optimizer = torch.optim.SGD(self.mean_field_critic.parameters(), lr=.1)
            self.training = True
            self.write_interval = self.checkpoint_interval = 0
            self._structured_rollout_sidecar = SimpleNamespace(_samples=[])
            self.memories, self.last_update_metrics = {}, {}
            self.updates, self.posts = 0, 0
        def init(self, **kwargs): pass
        def enable_training_mode(self, mode): pass
        def enable_models_training_mode(self, mode): pass
        def pre_interaction(self, **kwargs): pass
        def track_data(self, *args): pass
        def act(self, *args, **kwargs): return torch.zeros(2, 1), {}
        def record_transition(self, **kwargs):
            self._structured_rollout_sidecar._samples.extend([SimpleNamespace(action=torch.ones(1),
                graph_snapshot={}, owner_active_mask={'m': not inactive}) for _ in range(2)])
        def coordinated_update(self, **kwargs):
            self.updates += 1
            self.shared_actor.weight.sum().backward()
            self.actor_optimizer.step()
            self.owner_critic.weight.sum().backward()
            self.mean_field_critic.weight.sum().backward()
            self.last_update_metrics = {'minibatches': 1, 'actor_optimizer_steps': 1}
            clock['elapsed'] += 20.
        def post_interaction(self, **kwargs):
            self.posts += 1
            if self.posts % 2 == 0:
                self.coordinated_update(**kwargs)
                self._structured_rollout_sidecar._samples.clear()
        def save(self, path): Path(path).write_text('checkpoint')
    class Wrapper:
        num_envs = 2
        def reset(self): return torch.zeros(2, 1), {}
        def state(self): return None
        def step(self, actions):
            for i in range(2):
                env.training_telemetry.observe_transition(domain='gap', seed=16000+i, generation=0,
                    env_index=i, reward=1., terminated=False, truncated=False, success=False, active=not inactive)
            return torch.zeros(2, 1), torch.ones(2, 1), torch.zeros(2, 1, dtype=torch.bool), torch.zeros(2, 1, dtype=torch.bool), {}
    args = SimpleNamespace(output=tmp_path, total_transitions=1000, max_vector_steps=10,
        rollout_steps=2, learning_epochs=1, mini_batches=1, device='cpu', num_envs=2,
        benchmark=not train, smoke_train=False, train=train, checkpoint_every_transitions=0,
        max_wall_seconds=1000. if train else None, early_stopping_patience=0,
        early_stopping_min_delta=0., validation_every_seconds=10.,
        checkpoint_reserve_seconds=10., update_reserve_seconds=30.)
    learner = Learner()
    return training, args, env, Wrapper(), learner, clock


@pytest.mark.parametrize('inactive', [False, True])
def test_wall_guard_counts_startup_and_assembly_and_skips_original_post(tmp_path, monkeypatch, inactive):
    training, args, env, wrapper, learner, clock = _campaign_fixture(tmp_path, monkeypatch, inactive=inactive)
    clock['elapsed'] = 990.
    result = training.run_native_training(args, env, wrapper, learner,
        startup_s=400., assembly_s=590., run_started=0.)
    assert result['stop_reason'] == 'wall_budget'
    assert learner.posts == learner.updates == 0
    assert result['error'] is None
    assert Path(result['checkpoint_path']).is_file()


def test_wall_stop_after_real_update_is_success_without_target_budget(tmp_path, monkeypatch):
    training, args, env, wrapper, learner, clock = _campaign_fixture(tmp_path, monkeypatch)
    args.max_wall_seconds = 125.
    args.update_reserve_seconds = 1.
    result = training.run_native_training(args, env, wrapper, learner,
        startup_s=30., assembly_s=70., run_started=0.)
    assert result['status'] == 'pass'
    assert result['stop_reason'] == 'wall_budget'
    assert result['active_marl_transitions'] < args.total_transitions
    assert learner.updates == 1


def test_validation_baseline_then_patience_stops_before_further_gradients(tmp_path, monkeypatch):
    training, args, env, wrapper, learner, clock = _campaign_fixture(tmp_path, monkeypatch)
    args.early_stopping_patience = 1
    monkeypatch.setattr(training.time, 'perf_counter', lambda: clock['elapsed'])
    calls = []
    class Callback:
        expected_seed_manifest = [dict(env_index=i, domain=domain, seed=seed, split='validation')
            for i, (domain, seed) in enumerate(zip(('navigation', 'stairs', 'gap', 'manipulation'),
                                                  (8057, 13091, 17018, 21063)))]
        def evaluate(self, checkpoint, event_index, remaining_wall_s):
            calls.append((str(checkpoint), event_index, remaining_wall_s))
            clock['elapsed'] += 30.
            assert all(parameter.grad is None for model in
                (learner.shared_actor, learner.owner_critic, learner.mean_field_critic)
                for parameter in model.parameters())
            return dict(split='validation', optimizer_updates=0,
                parameter_sha256_before='a'*64, parameter_sha256_after='a'*64,
                checkpoint_path=str(checkpoint), score_by_domain=dict.fromkeys(
                    ('navigation', 'stairs', 'gap', 'manipulation'), 1. if event_index == 0 else 0.),
                success_by_domain=dict.fromkeys(('navigation', 'stairs', 'gap', 'manipulation'),
                                                .25 if event_index == 0 else 0.),
                seed_manifest=self.expected_seed_manifest, command=['frozen-evaluation'],
                peak_gpu_bytes=1, available_gpu_bytes=2)
    monkeypatch.setattr(training, '_make_validation_callback', lambda *a: Callback(), raising=False)
    result = training.run_native_training(args, env, wrapper, learner,
        startup_s=30., assembly_s=70., run_started=0.)
    assert result['status'] == 'pass' and result['stop_reason'] == 'validation_patience'
    assert learner.updates == 1 and len(calls) == 2
    assert result['best_validation_checkpoint'] == calls[0][0]
    assert result['early_stopping_state']['validation_count'] == 2
    assert result['early_stopping_state']['preserve_domain_floor']
    assert result['early_stopping_state']['best_macro_success'] == .25
    assert result['early_stopping_state']['baseline_success_by_domain']['gap'] == .25
    assert result['validation_events'][-1]['early_stopping']['guard_violations']['gap'] == 'success_below_bc_baseline'
    assert result['timings_s']['validation'] == 60.
    assert result['timings_s']['rollout'] == 0.
    assert result['timings_s']['total'] == 180.


def test_no_validation_during_benchmark_even_with_patience_configured(tmp_path, monkeypatch):
    training, args, env, wrapper, learner, clock = _campaign_fixture(tmp_path, monkeypatch, train=False)
    args.early_stopping_patience = 1
    args.total_transitions = 4
    def forbidden(*args): raise AssertionError('benchmark must never construct validation subprocess')
    monkeypatch.setattr(training, '_make_validation_callback', forbidden, raising=False)
    result = training.run_native_training(args, env, wrapper, learner,
        startup_s=.1, assembly_s=.2, run_started=0.)
    assert result['status'] == 'pass' and learner.updates == 1


def test_wrong_split_from_validation_callback_rejected_before_training(tmp_path, monkeypatch):
    training, args, env, wrapper, learner, clock = _campaign_fixture(tmp_path, monkeypatch)
    args.early_stopping_patience = 1
    callback = SimpleNamespace(evaluate=lambda *a: dict(split='test', score_by_domain={}))
    monkeypatch.setattr(training, '_make_validation_callback', lambda *a: callback, raising=False)
    result = training.run_native_training(args, env, wrapper, learner,
        startup_s=.1, assembly_s=.2, run_started=0.)
    assert result['status'] == 'fail' and result['error']['type'] == 'ValueError'
    assert learner.updates == 0


def test_validation_gpu_skip_saves_and_stops_without_inventing_a_score(tmp_path, monkeypatch):
    training, args, env, wrapper, learner, clock = _campaign_fixture(tmp_path, monkeypatch)
    args.early_stopping_patience = 1
    callback = SimpleNamespace(evaluate=lambda *a: dict(status='skipped', skipped=True,
                                                       reason='insufficient_gpu_headroom'))
    monkeypatch.setattr(training, '_make_validation_callback', lambda *a: callback, raising=False)
    result = training.run_native_training(args, env, wrapper, learner,
        startup_s=.1, assembly_s=.2, run_started=0.)
    assert result['stop_reason'] == 'validation_unavailable'
    assert result['error'] is None and learner.updates == 0
    assert result['early_stopping_state']['validation_count'] == 0
    assert result['best_validation_checkpoint'] is None
    assert result['wall_stop_info']['reason'] == 'insufficient_gpu_headroom'
    assert Path(result['checkpoint_path']).is_file()


def test_configured_update_reserve_remains_after_a_fast_measured_update(tmp_path, monkeypatch):
    training, args, env, wrapper, learner, clock = _campaign_fixture(tmp_path, monkeypatch)
    args.max_wall_seconds = 200.
    args.update_reserve_seconds = 80.
    result = training.run_native_training(args, env, wrapper, learner,
        startup_s=30., assembly_s=70., run_started=0.)
    assert result['status'] == 'pass' and result['stop_reason'] == 'wall_budget'
    assert learner.updates == 1
    assert result['wall_budget_state']['costs']['update'] >= 80.
