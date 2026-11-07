"""Read-only decision provenance must preserve the complete event clock trace."""
from types import SimpleNamespace

import pytest
import torch

from drl.MARL.c1_performance import PerformanceObserver
from drl.MARL.c1_training_options import NativeStructuralOptionController
from drl.MARL.tests.test_c1_training_option_runtime import _Env, _goal


def trace(observe=False):
    env = _Env()
    env.step_dt = 1/30
    env.marl_steps = [0, 0]
    env.common_step_counter = 0
    controller = NativeStructuralOptionController(env, mode='event')
    provider = SimpleNamespace(native_env=env, _structural_options=controller)
    learner = SimpleNamespace(_rollout_context_provider=provider)
    rows, states = [], []
    probe = PerformanceObserver('cpu', decision_sink=rows.append) if observe else None
    def run():
        for i in range(65):
            env.common_step_counter = i
            env.marl_steps[:] = [i, i]
            if i == 1:
                env.executors[0].active_goals = (_goal('align-a'),)
            if i == 4:
                env.executors[0].active_goals = ()
                env.executors[0].prepared = (SimpleNamespace(source_module='m0', source_face='TOP',
                    target_module='m1', target_face='BOTTOM', docked=False),)
            if i in (8, 12):
                controller.set_continuous_control(0, i == 8)
            if i == 20:
                env.generation[0] += 1
            mask = controller.decision_mask()
            states.append((mask, [(c.scheduler._force_decision, c.scheduler._idle_frames,
                c.scheduler._continuous_control) for c in controller._clocks],
                [e.seals for e in env.executors]))
    if observe:
        with probe:
            probe.attach_learner(learner)
            run()
    else:
        run()
    return states, rows, probe


def test_four_reasons_are_logged_without_altering_masks_clocks_or_rng(monkeypatch):
    before, _, _ = trace()
    from drl.MARL.c1_gui_viewer import StructuralOptionScheduler
    original = StructuralOptionScheduler.should_decide
    calls = []
    def counted(scheduler, state):
        calls.append(True)
        return original(scheduler, state)
    monkeypatch.setattr(StructuralOptionScheduler, 'should_decide', counted)
    rng = torch.get_rng_state().clone()
    after, rows, probe = trace(True)
    assert before == after
    assert torch.equal(rng, torch.get_rng_state())
    assert len(rows) == 130
    assert len(calls) == 130
    selected = [r for r in rows if r['decide']]
    assert {r['reason'] for r in selected} == {
        'continuous_resource_control', 'prepared_handoff', 'idle_fallback', 'structural_decision'}
    assert rows[0]['reason'] == 'structural_decision'
    assert rows[5*2]['reason'] == 'prepared_handoff'
    assert rows[8*2]['reason'] == 'continuous_resource_control'
    assert rows[15*2+1]['reason'] == 'idle_fallback'
    assert rows[20*2]['reason'] == 'structural_decision'
    assert rows[20*2]['generation'] == 1
    assert rows[8*2]['decision_time_s'] == 8/30
    assert rows[8*2]['vector_frame'] == 9
    assert all(r['reason'] is None for r in rows if not r['decide'])
    counts = probe.report()['decision_reason_counts']
    assert sum(counts.values()) == len(selected)


def test_observer_restores_scheduler_method_even_if_recording_is_interrupted():
    from drl.MARL.c1_gui_viewer import StructuralOptionScheduler
    env = _Env()
    ctl = NativeStructuralOptionController(env, mode='event')
    learner = SimpleNamespace(_rollout_context_provider=SimpleNamespace(native_env=env, _structural_options=ctl))
    original = StructuralOptionScheduler.should_decide
    with pytest.raises(RuntimeError), PerformanceObserver('cpu', decision_sink=lambda row: None) as observer:
        observer.attach_learner(learner)
        assert ctl.decision_mask() == (True, True)
        raise RuntimeError('intentional')
    assert StructuralOptionScheduler.should_decide is original
