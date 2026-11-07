"""Opt-in performance probes count actual calls and restore every method."""
from types import SimpleNamespace

import pytest
import torch


def test_real_h1_h2_counts_distinguish_sleeping_act_and_restore():
    from drl.MARL.c1_performance import PerformanceObserver
    class Backbone:
        def encode(self, graph):
            return graph + 1
    class Head(torch.nn.Module):
        def forward(self, graph):
            return graph * 2
    backbone, head = Backbone(), Head()
    def act(graph=None):
        return None if graph is None else head(backbone.encode(graph))
    learner = SimpleNamespace(shared_actor=SimpleNamespace(policy=SimpleNamespace(backbone=backbone, head=head)),
                              act=act, record_transition=lambda: None, _update=lambda: None)
    rng = torch.get_rng_state().clone()
    probe = PerformanceObserver("cpu")
    with probe:
        probe.attach_learner(learner)
        assert learner.act() is None
        torch.testing.assert_close(learner.act(torch.tensor(2)), torch.tensor(6))
        learner.record_transition()
        learner._update()
    report = probe.report()
    assert report["calls"]["rollout_actor"] == 2
    assert report["calls"]["h1_encode"] == report["calls"]["h2_forward"] == 1
    assert report["explicit_cuda_synchronizations"] == 0
    assert learner.act is act
    assert "encode" not in backbone.__dict__ and "forward" not in head.__dict__
    assert torch.equal(rng, torch.get_rng_state())


def test_probe_restores_on_failure_and_counts_critic_phase():
    from drl.MARL.c1_performance import PerformanceObserver
    class Critic:
        def forward_all_owners(self, graph):
            return graph
    critic = Critic()
    learner = SimpleNamespace(altruistic_owner_critic=critic,
                              _update=lambda: critic.forward_all_owners(torch.ones(1)))
    probe = PerformanceObserver("cpu")
    with pytest.raises(ValueError), probe:
        probe.attach_learner(learner)
        learner._update()
        raise ValueError("intentional")
    report = probe.report()
    assert report["calls"]["ppo/owner_critic_graph"] == 1
    assert "forward_all_owners" not in critic.__dict__


def test_baseline_and_fastpath_view_and_padding_counters_use_actual_requests():
    from drl.MARL.c1_performance import PerformanceObserver
    class Bridge:
        def prepare_snapshot(self, **kwargs):
            return kwargs
    bridge = Bridge()
    provider = SimpleNamespace(native_env=SimpleNamespace(observation_bridge=bridge),
        current_action_contexts=lambda: bridge.prepare_snapshot(include_actor=False),
        current_critic_graphs=lambda: bridge.prepare_snapshot(include_actor=False))
    learner = SimpleNamespace(_rollout_context_provider=provider,
        evaluate_altruistic_critic_minibatch=lambda **kw: kw["structured_batch"])
    probe = PerformanceObserver("cpu")
    with probe:
        probe.attach_learner(learner)
        provider.current_action_contexts()
        provider.current_critic_graphs()
        learner.evaluate_altruistic_critic_minibatch(
            structured_batch=[SimpleNamespace(rollout_active=x) for x in [False, True, False]],
            skip_inactive=True)
    report = probe.report()
    assert report["rows"]["preparation_pre/actor_views"] == 0
    assert report["rows"]["preparation_post/critic_views"] == 1
    assert report["rows"]["ppo_critic_valid"] == 1
    assert report["rows"]["ppo_critic_padding_skipped"] == 2
    assert report["calls"]["preparation_pre/observation_snapshot"] == 1


def test_native_mssr_actual_update_entrypoint_is_coordinated_update():
    from drl.MARL.c1_performance import PerformanceObserver
    learner = SimpleNamespace(coordinated_update=lambda **kw: kw)
    probe = PerformanceObserver("cpu")
    with probe:
        probe.attach_learner(learner)
        assert learner.coordinated_update(timestep=2) == {"timestep": 2}
    assert probe.report()["calls"]["ppo"] == 1


def test_per_row_preparation_timings_classify_without_querying_scheduler():
    from drl.MARL.c1_performance import PerformanceObserver
    env = SimpleNamespace(handoffs=[SimpleNamespace(marl_active=True), SimpleNamespace(marl_active=False)],
                          prepare_env=lambda i, **kw: (i, kw))
    learner = SimpleNamespace(_rollout_context_provider=SimpleNamespace(native_env=env))
    probe = PerformanceObserver("cpu")
    with probe:
        probe.attach_learner(learner)
        env.prepare_env(0, include_candidates_override=True)
        env.prepare_env(0, include_candidates_override=False)
        env.prepare_env(1, include_candidates_override=True)
        env.prepare_env(0, post_step=True)
    calls = probe.report()["calls"]
    assert calls["preparation_row_active"] == calls["preparation_row_post"] == 1
    assert calls["preparation_row_sleeping"] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_events_are_explicit_bounded_and_flush_only_at_report():
    from drl.MARL.c1_performance import PerformanceObserver
    probe = PerformanceObserver("cuda:0", cuda_events=True, max_event_pairs=1)
    obj = SimpleNamespace(work=lambda: torch.ones(10, device="cuda:0") * 3)
    with probe:
        probe.wrap(obj, "work", "work")
        obj.work()
        obj.work()
        assert probe.explicit_cuda_synchronizations == 0
        result = probe.report()
        assert result["explicit_cuda_synchronizations"] == 1
        assert result["cuda_event_pairs"] == 1
        assert result["cuda_event_pairs_dropped"] == 1
        assert result["cuda_stream_ms"]["work"] >= 0
