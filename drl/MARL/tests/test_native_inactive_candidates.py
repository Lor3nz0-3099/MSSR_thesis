"""Native candidate policy follows pre-step phase for terminal critic snapshots."""
from types import SimpleNamespace

from drl.MARL.isaaclab_c1_tasks import NativeC1TaskHooks


def test_native_current_and_post_step_candidate_flags_preserve_terminal_identity(monkeypatch):
    from drl.MARL import c1_runner
    monkeypatch.setattr(c1_runner, 'physical_task_view', lambda *args: ('context', 'observation'))
    calls = []
    def prepare(raw, **kwargs):
        calls.append((raw['identity'], kwargs['include_action_candidates']))
        return dict(identity=raw['identity'], candidates=kwargs['include_action_candidates'])
    current = [dict(identity='reset0', stamp=2., state={'course': 'c0'}),
               dict(identity='active1', stamp=2., state={'course': 'c1'})]
    terminal = [dict(identity='terminal0', stamp=1., state={'course': 'c0'}),
                dict(identity='expert1', stamp=1., state={'course': 'c1'})]
    env = SimpleNamespace(_prepared_cache={}, generation=[2, 1], _post_raw=terminal,
        cached_raw_snapshot=lambda i: current[i], domain_by_env=['gap', 'stairs'],
        task_context_config={}, handoffs=[SimpleNamespace(marl_active=False), SimpleNamespace(marl_active=True)],
        _marl_pre=[True, False], observation_bridge=SimpleNamespace(prepare_snapshot=prepare))
    for post_step in (False, True):
        for index in (0, 1):
            NativeC1TaskHooks.prepare_env(env, index, post_step=post_step)
    assert calls == [('reset0', False), ('active1', True), ('terminal0', True), ('expert1', False)]
    # A phase change in the same raw frame must not reuse an Expert-only cache.
    env.handoffs[0].marl_active = True
    result = NativeC1TaskHooks.prepare_env(env, 0)
    assert result['candidates'] is True
    assert calls[-1] == ('reset0', True)
