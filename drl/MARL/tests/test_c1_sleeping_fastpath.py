"""Offline physical fixtures for privileged-only native post-step preparation."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from drl.IL.bc_v1.bc_v1_contract import TaskContext, ObservationBundle
from drl.IL.tests.bc_v1.test_live_observation import snapshot
from drl.MARL.isaaclab_c1_tasks import NativeC1TaskHooks
from drl.MARL.native_policy_bridge import NativePreparedRuntimeContextProvider
from drl.MARL.tests.test_bc_v4_runtime_observation import make_bridge
from drl.MARL.tests.test_c1_runtime_static_cache import equal_tree
from drl.MARL.tests.test_native_policy_bridge import active_handoff


@pytest.mark.parametrize('connected', [False, True])
def test_privileged_only_matches_complete_view_without_actor_work_or_rng(monkeypatch, connected):
    from drl.MARL import bc_v4_runtime_observation as runtime
    bridge = make_bridge()
    raw = snapshot(n=8, rc=connected)
    kwargs = dict(task_context=TaskContext('navigation'), include_action_candidates=False)
    reference = bridge.prepare_snapshot(raw, **kwargs)
    rng = torch.get_rng_state().clone()

    def forbidden(*args, **kwargs):
        raise AssertionError('privileged-only preparation must not construct an actor view')

    monkeypatch.setattr(runtime, 'tensorize_actor_sample', forbidden)
    reduced = bridge.prepare_snapshot(raw, include_actor=False, **kwargs)
    equal_tree(reference.privileged_graph, reduced.privileged_graph)
    assert reduced.physical_state == reference.physical_state
    assert reduced.actor_graph is reduced.routing_snapshot is reduced.decoder_batch is None
    assert torch.equal(rng, torch.get_rng_state())


def native_fixture(monkeypatch, n=8):
    from drl.MARL import c1_runner
    monkeypatch.setattr(c1_runner, 'physical_task_view',
                        lambda *args: (TaskContext('navigation'), ObservationBundle()))
    current = [snapshot(n=8, rc=bool(i % 2)) for i in range(n)]
    for raw in current:
        raw['state']['course'] = {}
    env = SimpleNamespace(num_envs=n, device=torch.device('cpu'), generation=[1] * n,
        observation_bridge=make_bridge(), _prepared_cache={}, _marl_pre=[True] * n,
        domain_by_env=['navigation'] * n, task_context_config={},
        handoffs=[active_handoff() for _ in range(n)],
        _post_raw=deepcopy(current), current=current)
    env.cached_raw_snapshot = current.__getitem__
    env.prepare_env = lambda i, **kw: NativeC1TaskHooks.prepare_env(env, i, **kw)
    return env


def test_native_privileged_cache_keeps_actor_requests_and_legacy_sources_separate(monkeypatch):
    env = native_fixture(monkeypatch, n=1)
    reduced = env.prepare_env(0, post_step=True, include_candidates_override=False, include_actor=False)
    full = env.prepare_env(0, post_step=True, include_candidates_override=False)
    assert reduced.actor_graph is None
    assert full.actor_graph is not None
    assert reduced is env.prepare_env(0, post_step=True, include_candidates_override=False, include_actor=False)
    equal_tree(reduced.privileged_graph, full.privileged_graph)
    real_bridge = env.observation_bridge

    # Existing injected bridges do not advertise the new optional capability.
    class LegacyBridge:
        def prepare_snapshot(self, raw, *, task_context, observation, include_action_candidates):
            return real_bridge.prepare_snapshot(raw, task_context=task_context,
                observation=observation, include_action_candidates=include_action_candidates)

    env.observation_bridge = LegacyBridge()
    env._prepared_cache.clear()
    fallback = env.prepare_env(0, post_step=True, include_candidates_override=False, include_actor=False)
    assert fallback.actor_graph is not None
    equal_tree(reduced.privileged_graph, fallback.privileged_graph)


def test_sleeping_omission_retains_original_snapshots_for_legacy_source():
    from drl.MARL.runtime_handoff import DirectMARLHandoff
    from drl.MARL.tests.test_native_policy_bridge import make_bridge as legacy_bridge
    bundle, _, _, _ = legacy_bridge([active_handoff(), DirectMARLHandoff()])
    bundle.provider.set_sleeping_actor_omission(True)
    contexts = bundle.provider.current_action_contexts()
    assert contexts[1].actor_graph is not None
    assert contexts[1].routing_snapshot['owner'] == 1
    assert not contexts[1].rollout_active
    assert bundle.provider.preparation_metrics()['sleeping_actor_omitted_rows'] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires offline CUDA tensor placement')
def test_legacy_post_step_skips_actor_device_copies_and_counts_real_transfers():
    from drl.MARL.tests.test_native_policy_bridge import make_bridge as legacy_bridge
    bundle, env, observations, _ = legacy_bridge()
    env.device = torch.device('cuda:0')
    bundle.provider.current_action_contexts()
    before = bundle.provider.preparation_metrics()
    decision = SimpleNamespace(action=object(), action_set_prediction_template={})
    bundle.provider.encode_env_actions([decision, decision])
    terminal = [observations.prepare_snapshot({'stamp': 2., 'env': i}, task_context=i)
                for i in range(2)]
    bundle.provider.post_step_observation_provider = terminal.__getitem__
    result = bundle.provider.current_critic_graphs()
    assert all(graph['privileged'].device == env.device for graph in result)
    assert all(item.actor_graph['visible'].device.type == 'cpu' for item in terminal)
    after = bundle.provider.preparation_metrics()
    assert after['actor_transfer_tensors'] == before['actor_transfer_tensors'] == 2
    assert after['critic_transfer_tensors'] - before['critic_transfer_tensors'] == 4
    assert after['post_actor_views'] == after['post_critic_views'] == 2


def physical_provider(env, decision_mask):
    class Source:
        num_envs = env.num_envs
        def prepare_env_with_candidates(self, i, *, include_action_candidates):
            return env.prepare_env(i, include_candidates_override=include_action_candidates)
        def prepare_env_privileged(self, i, *, include_action_candidates=False):
            return env.prepare_env(i, include_candidates_override=include_action_candidates, include_actor=False)
        def reset(self):
            env._prepared_cache.clear()

    provider = NativePreparedRuntimeContextProvider(observation_source=Source(), native_env=env,
        post_step_observation_provider=lambda i: env.prepare_env(i, post_step=True,
            include_candidates_override=False, include_actor=False),
        graph_source=lambda: (), actor_visibility=lambda x: x, routing_builder=lambda x: {},
        hard_mask_builder=lambda actor, routing: torch.zeros(
            len(routing['resource_candidates']) + len(routing['structural_bindings']), dtype=torch.bool),
        handoffs=env.handoffs,
        action_encoder=lambda decision, i, context: {'m0': torch.zeros(1)},
        neutral_action=lambda i, context: {'m0': torch.zeros(1)})
    provider._structural_options = SimpleNamespace(decision_mask=decision_mask,
        set_continuous_control=lambda *args: None)
    return provider


def test_expert_inactive_omission_preserves_original_critic_candidate_tables(monkeypatch):
    from drl.MARL.runtime_handoff import DirectMARLHandoff
    env = native_fixture(monkeypatch)
    env.handoffs[1] = DirectMARLHandoff()
    provider = physical_provider(env, lambda: (True,) * 8)
    reference = provider.current_action_contexts()[1].critic_graph
    assert reference['structural']['initiator'].numel() > 0
    provider.reset()
    provider.set_sleeping_actor_omission(True)
    omitted = provider.current_action_contexts()[1]
    assert not omitted.rollout_active
    assert omitted.actor_graph is None
    equal_tree(reference, omitted.critic_graph)


@pytest.mark.parametrize('omit_sleeping', [False, True])
def test_mixed_eight_rows_keep_sleeping_views_terminal_graphs_and_fresh_epochs(monkeypatch, omit_sleeping):
    env = native_fixture(monkeypatch)
    mask = (True, False, True, False, True, False, True, False)
    provider = physical_provider(env, lambda: mask)
    provider.set_sleeping_actor_omission(omit_sleeping)
    rng = torch.get_rng_state().clone()
    before = provider.current_action_contexts()
    assert tuple(c.rollout_active for c in before) == mask
    for i, context in enumerate(before):
        if omit_sleeping and not mask[i]:
            assert context.actor_graph is context.routing_snapshot is None
            assert context.hard_mask.numel() == 0
        elif not mask[i]:
            assert context.routing_snapshot['resource_candidates'] == ()
            assert context.routing_snapshot['structural_bindings'] == ()
        reference = env.prepare_env(i, include_candidates_override=mask[i])
        equal_tree(context.critic_graph, reference.privileged_graph)
        if mask[i] or not omit_sleeping:
            equal_tree(context.actor_graph, reference.actor_graph)
    decision = SimpleNamespace(action=object())
    provider.encode_env_actions([decision if active else None for active in mask])
    terminal_reference = [env.prepare_env(i, post_step=True,
        include_candidates_override=False).privileged_graph for i in range(8)]
    # One terminal row auto-resets to a different topology at the same stamp.
    env.generation[0] += 1
    reset_raw = snapshot(n=8, attached=True)
    reset_raw['state']['course'] = {}
    env.current[0] = reset_raw
    final = provider.current_critic_graphs()
    for expected, actual in zip(terminal_reference, final, strict=True):
        equal_tree(expected, actual)
    assert provider.current_critic_graphs() is final
    assert provider._captured is None
    assert not env._prepared_cache
    after = provider.current_action_contexts()
    assert after[0].critic_graph is not final[0]
    assert len(after[0].routing_snapshot['structural_bindings']) != len(before[0].routing_snapshot['structural_bindings'])
    equal_tree(before[1].critic_graph, after[1].critic_graph)
    assert after[1].critic_graph is not before[1].critic_graph
    if not omit_sleeping:
        equal_tree(before[1].actor_graph, after[1].actor_graph)
        assert after[1].actor_graph is not before[1].actor_graph
    assert torch.equal(rng, torch.get_rng_state())
    metrics = provider.preparation_metrics()
    assert metrics['active_rows'] == metrics['sleeping_rows'] == 8
    assert metrics['actor_views'] == (8 if omit_sleeping else 16)
    assert metrics['critic_views'] == 16
    assert metrics['sleeping_actor_omitted_rows'] == (8 if omit_sleeping else 0)
    assert metrics['post_actor_views'] == 0
    assert metrics['post_critic_views'] == 8
    assert metrics['actor_transfer_tensors'] == metrics['critic_transfer_tensors'] == 0


def test_real_physical_sleeping_omission_preserves_smdp_padding_epochs_and_targets(monkeypatch):
    from dataclasses import fields
    from drl.MARL.altruistic_rollout import AltruisticRolloutSample
    from drl.MARL.mappo import OnlineRolloutDecision, StructuredRolloutSample
    from drl.MARL.tests.c1_smdp_fixtures import make_learner, capture_update_targets

    class PhysicalCritic:
        def __init__(self, scale):
            self.scale = scale
        def forward_all_owners(self, graph):
            return self.scale * graph['module_x'].sum(dim=1)

    def collect(omit):
        env = native_fixture(monkeypatch)
        mask = [True] * 8
        provider = physical_provider(env, lambda: tuple(mask))
        learner = make_learner(capacity=3, num_envs=8)
        learner.skip_expert_policy_inference = omit
        learner._rollout_context_provider = provider
        learner.configure_smdp_options(enabled=True)
        learner.altruistic_owner_critic = PhysicalCritic(1.)
        learner.altruistic_mean_field_critic = PhysicalCritic(3.)
        masks = [[True] * 8, [i % 2 == 0 for i in range(8)], [i % 2 == 1 for i in range(8)]]
        for step, frame_mask in enumerate(masks):
            mask[:] = frame_mask
            contexts = provider.current_action_contexts()
            decisions = []
            for context in contexts:
                sample = StructuredRolloutSample(action=None, hard_mask=context.hard_mask,
                    owner_active_mask={'a': context.rollout_active},
                    routing_snapshot=context.routing_snapshot, graph_snapshot=context.actor_graph,
                    critic_graph_snapshot=context.critic_graph, rollout_active=context.rollout_active)
                decisions.append(OnlineRolloutDecision(action=None, owner_log_probs={'a': torch.zeros(1)},
                    global_value=torch.zeros(1, 1), structured_sample=sample))
            learner._current_online_decisions = decisions
            rewards = [AltruisticRolloutSample(owner_row_index={'a': 0},
                owner_rewards={'a': torch.tensor(float(step + 1))},
                mean_field_rewards={'a': torch.tensor(float(2 * step + 1))},
                neighbourhoods={'a': ()}, next_owner_row_index={'a': 0}) for _ in range(8)]
            learner._build_altruistic_rollout_step = lambda **_: rewards
            actions = {'a': provider.encode_env_actions(decisions)['m0']}
            env._post_raw = deepcopy(env.current)
            if step == 1:
                env.generation[0] += 1
                env.current[0] = snapshot(n=8, attached=True)
                env.current[0]['state']['course'] = {}
            next_graphs = provider.current_critic_graphs()
            obs = {'a': torch.full((8, 1), float(step))}
            nxt = {'a': obs['a'] + 1}
            truncated = torch.zeros(8, 1, dtype=torch.bool)
            truncated[0] = step == 1
            learner._record_smdp_transition(observations=obs, states=obs, actions=actions,
                next_observations=nxt, next_states=nxt, terminated=torch.zeros_like(truncated),
                truncated=truncated, decisions=decisions, next_critic_graphs=next_graphs,
                team_rewards=torch.full((8, 1), float(step + 1)))
        return learner, capture_update_targets(learner), provider.preparation_metrics()

    rng = torch.get_rng_state().clone()
    old, old_targets, old_metrics = collect(False)
    new, new_targets, new_metrics = collect(True)
    assert torch.equal(torch.get_rng_state(), rng)
    for name in ('owner_bootstrap', 'mf_bootstrap'):
        equal_tree(old_targets[name], new_targets[name])
    for field in fields(old_targets['targets']):
        equal_tree(getattr(old_targets['targets'], field.name), getattr(new_targets['targets'], field.name))
    for left, right in zip(old._smdp_pending, new._smdp_pending, strict=True):
        assert (left is None) == (right is None)
        if left is not None:
            assert left.rewards.duration_steps == right.rewards.duration_steps
            equal_tree(left.decision.structured_sample.graph_snapshot,
                       right.decision.structured_sample.graph_snapshot)
            equal_tree(left.rewards.owner_rewards, right.rewards.owner_rewards)
    old_samples = old._structured_rollout_sidecar._samples
    new_samples = new._structured_rollout_sidecar._samples
    assert len(old_samples) == len(new_samples) == 24
    assert any(not sample.rollout_active and sample.graph_snapshot is None for sample in new_samples)
    for left, right in zip(old_samples, new_samples, strict=True):
        assert left.duration_steps == right.duration_steps
        assert left.owner_active_mask == right.owner_active_mask
        equal_tree(left.critic_graph_snapshot, right.critic_graph_snapshot)
        equal_tree(left.next_critic_graph_snapshot, right.next_critic_graph_snapshot)
        if right.rollout_active:
            assert right.graph_snapshot is not None
            equal_tree(left.graph_snapshot, right.graph_snapshot)
    assert old_metrics['sleeping_actor_omitted_rows'] == 0
    assert new_metrics['sleeping_actor_omitted_rows'] == 8
    assert old_metrics['post_actor_views'] == new_metrics['post_actor_views'] == 0
