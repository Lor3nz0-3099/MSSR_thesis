import pytest


def test_timing_distribution_reports_tail_latency_without_numpy():
    from drl.MARL.c1_gui_viewer import summarize_timing_samples
    assert summarize_timing_samples([]) == dict(count=0, mean=0.0, p50=0.0, p95=0.0, max=0.0)
    result = summarize_timing_samples([0.1, 0.2, 0.3, 1.0])
    assert result['count'] == 4
    assert result['mean'] == pytest.approx(0.4)
    assert result['p50'] == pytest.approx(0.25)
    assert result['p95'] == pytest.approx(1.0)
    assert result['max'] == pytest.approx(1.0)


def test_viewer_status_reports_not_ready_mask_count(tmp_path):
    from types import SimpleNamespace
    from drl.MARL.c1_feasibility import CandidateAvailability, NativeAvailability
    from drl.MARL.c1_viewer_events import ViewerEvents

    recorder = ViewerEvents(tmp_path, 'navigation', classifier=SimpleNamespace())
    record = SimpleNamespace(candidate_id='structural|dock|a|a:TOP>b:BOTTOM',
                             candidate_index=0, candidate_kind='structural')
    availability = NativeAvailability((True,), (CandidateAvailability('masked_not_ready'),))
    recorder.observe_context(0, 0, (record,), availability, 0.0)
    assert recorder.latest_mask == dict(available=0, busy=0, unavailable=0, not_ready=1, handoff_wait=0, prepared_conflict=0)
    recorder.close()


def test_rollout_does_not_flush_sparse_event_writer_every_decision(monkeypatch):
    from collections import deque
    from types import SimpleNamespace
    import torch
    import drl.MARL.c1_gui_viewer as viewer

    context = SimpleNamespace(
        routing_snapshot={'resource_candidates': (), 'structural_bindings': ()},
        availability_snapshot=SimpleNamespace(),
    )
    decision = SimpleNamespace(action=SimpleNamespace(), action_set_prediction_template={})
    monkeypatch.setattr(viewer, 'sample_contexts',
                        lambda *args, **kwargs: [(decision, {'records': ()})])

    class Provider:
        def current_action_contexts(self): return [context]
        def encode_env_actions(self, _): return object()
        def reset(self): pass

    class Recorder:
        def __init__(self): self.flush_calls = 0; self.decision_step = -1
        def observe_context(self, *args, **kwargs): pass
        def observe_transition(self, *args, **kwargs): pass
        def slow_status(self, *args, **kwargs): return None
        def flush(self): self.flush_calls += 1

    evaluator = SimpleNamespace(last_frame=None)
    runtime = SimpleNamespace(recent_transitions=deque(), evaluators=[(0, evaluator)])
    class Env:
        generation = [0]
        num_envs = 1
        physics_dt = 1.0 / 240.0
        _sim_step_counter = 0
        c1_runtime = runtime
        def step(self, _carrier):
            self._sim_step_counter += 8
            runtime.recent_transitions.append({'env_index': 0, 'generation': 0})
            return None, {'reward': torch.tensor(0.0)}, None

    recorder = Recorder()
    result = viewer.rollout_viewer(
        Env(), object(), Provider(), recorder, ('smores_01',),
        steps=2, max_wall_seconds=60.0, accelerated=True, application=None)
    assert result['decisions_completed'] == 2
    assert recorder.flush_calls == 0


def test_low_latency_viewer_events_do_not_flush_or_write_progress_each_status(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from drl.MARL.c1_viewer_events import ViewerEvents
    recorder = ViewerEvents(tmp_path, 'navigation', classifier=SimpleNamespace(), low_latency=True)
    flushes = []
    monkeypatch.setattr(recorder.writer, 'flush', lambda: flushes.append(True))
    recorder.latest_mask = dict(available=1, busy=0, unavailable=0, not_ready=0)
    recorder.slow_status(1.0, now=10.0, force=True)
    assert flushes == []
    assert not (tmp_path / 'progress.json').exists()
    recorder.close()


def test_event_structural_scheduler_skips_policy_while_option_runs_but_keeps_stepping(monkeypatch):
    from collections import deque
    from types import SimpleNamespace
    import torch
    import drl.MARL.c1_gui_viewer as viewer

    context = SimpleNamespace(
        routing_snapshot={'resource_candidates': (), 'structural_bindings': ()},
        availability_snapshot=SimpleNamespace(mask_tensor=lambda device=None: torch.zeros(0, dtype=torch.bool)),
        hard_mask=torch.zeros(0, dtype=torch.bool),
    )
    decision = SimpleNamespace(action=SimpleNamespace(), action_set_prediction_template={})
    samples = []
    def sample(*args, **kwargs):
        samples.append(True)
        return [(decision, {'records': ()})]
    monkeypatch.setattr(viewer, 'sample_contexts', sample)
    import sys
    monkeypatch.setitem(sys.modules, 'drl.MARL.policy', SimpleNamespace(
        bind_marl_candidates=lambda **kwargs: ()))

    class Provider:
        def current_action_contexts(self): return [context]
        def encode_env_actions(self, _): return {'smores_01': torch.ones((1, 1))}
        def reset(self): pass

    class Recorder:
        decision_step = -1
        def observe_context(self, *args, **kwargs): pass
        def observe_transition(self, *args, **kwargs): pass
        def slow_status(self, *args, **kwargs): pass

    goal = SimpleNamespace(goal_id='align-wave')
    executor = SimpleNamespace(active_goals=(), prepared_relation_snapshot=lambda: ())
    evaluator = SimpleNamespace(last_frame=None)
    runtime = SimpleNamespace(recent_transitions=deque(maxlen=16), evaluators={0: (0, evaluator)})

    class Env:
        generation = [0]
        num_envs = 1
        physics_dt = 1.0 / 240.0
        _sim_step_counter = 0
        device = 'cpu'
        native_dispatcher = SimpleNamespace(executor_provider=lambda i: executor)
        docking_managers = (SimpleNamespace(connections=()),)
        c1_runtime = runtime
        step_calls = 0
        carriers = []
        def step(self, carrier):
            self.carriers.append(carrier)
            self.step_calls += 1
            self._sim_step_counter += 8
            if self.step_calls == 1:
                executor.active_goals = (goal,)
            runtime.recent_transitions.append({'env_index': 0, 'generation': 0})
            return None, {'smores_01': torch.tensor(0.0)}, None

    env = Env()
    result = viewer.rollout_viewer(
        env, object(), Provider(), Recorder(), ('smores_01',),
        steps=3, max_wall_seconds=60.0, accelerated=True, application=None,
        structural_scheduler='event')

    assert env.step_calls == 3
    # First decision starts the option; the second is the one-frame wave-fill
    # saturation check.  Only after no new goal appears does the scheduler sleep.
    assert len(samples) == 2
    assert result['policy_evaluations'] == 2
    assert result['idle_control_frames'] == 1
    assert env.carriers[0]['smores_01'].item() == 1.0
    assert env.carriers[1]['smores_01'].item() == 1.0
    assert env.carriers[2]['smores_01'].item() == 0.0
