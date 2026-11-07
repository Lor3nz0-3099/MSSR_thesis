from types import SimpleNamespace


def state(*, active=(), prepared=(), connections=0):
    from drl.MARL.c1_gui_viewer import StructuralOptionState
    return StructuralOptionState(tuple(active), tuple(prepared), connections)


def test_event_scheduler_waits_while_a_sealed_structural_wave_remains_active():
    from drl.MARL.c1_gui_viewer import StructuralOptionScheduler
    scheduler = StructuralOptionScheduler(mode='event', control_hz=30.0, fallback_hz=2.0)
    idle = state()
    two_active = state(active=('align-a', 'align-b'))
    one_active = state(active=('align-b',), prepared=(('m5', 'TOP', 'm1', 'BOTTOM', False),))

    assert scheduler.should_decide(idle)
    scheduler.observe_step(idle, two_active)
    # New members joined: one more policy evaluation is allowed to fill the wave.
    assert scheduler.should_decide(two_active)
    scheduler.observe_step(two_active, two_active)
    # No member joined on that evaluation, so the wave is sealed.
    assert not scheduler.should_decide(two_active)
    scheduler.observe_step(two_active, one_active)
    assert not scheduler.should_decide(one_active)


def test_event_scheduler_reopens_only_after_active_wave_becomes_idle():
    from drl.MARL.c1_gui_viewer import StructuralOptionScheduler
    scheduler = StructuralOptionScheduler(mode='event', control_hz=30.0, fallback_hz=2.0)
    idle = state()
    active = state(active=('align-b',), prepared=(('m5', 'TOP', 'm1', 'BOTTOM', False),))
    prepared = state(prepared=(('m5', 'TOP', 'm1', 'BOTTOM', False),
                               ('m7', 'BOTTOM', 'm1', 'TOP', False)))

    assert scheduler.should_decide(idle)
    scheduler.observe_step(idle, active)
    # One saturation decision is allowed after the first member starts.
    assert scheduler.should_decide(active)
    scheduler.observe_step(active, active)
    assert not scheduler.should_decide(active)
    scheduler.observe_step(active, prepared)
    assert scheduler.should_decide(prepared)


def test_event_scheduler_detects_instant_topology_change_even_without_visible_active_goal():
    from drl.MARL.c1_gui_viewer import StructuralOptionScheduler
    scheduler = StructuralOptionScheduler(mode='event', control_hz=30.0, fallback_hz=2.0)
    prepared = state(prepared=(('m7', 'BOTTOM', 'm1', 'TOP', False),))
    docked = state(prepared=(('m7', 'BOTTOM', 'm1', 'TOP', True),), connections=1)

    assert scheduler.should_decide(prepared)
    scheduler.observe_step(prepared, docked)
    assert scheduler.should_decide(docked)


def test_event_scheduler_idle_fallback_is_two_hz_not_thirty_hz():
    from drl.MARL.c1_gui_viewer import StructuralOptionScheduler
    scheduler = StructuralOptionScheduler(mode='event', control_hz=30.0, fallback_hz=2.0)
    idle = state()
    assert scheduler.should_decide(idle)
    scheduler.observe_step(idle, idle)
    for _ in range(14):
        assert not scheduler.should_decide(idle)
        scheduler.observe_step(idle, idle)
    assert scheduler.should_decide(idle)


def test_fixed_scheduler_preserves_policy_evaluation_every_control_frame():
    from drl.MARL.c1_gui_viewer import StructuralOptionScheduler
    scheduler = StructuralOptionScheduler(mode='fixed', control_hz=30.0, fallback_hz=2.0)
    active = state(active=('align-a',))
    for _ in range(5):
        assert scheduler.should_decide(active)
        scheduler.observe_step(active, active)


def test_structural_state_snapshot_reads_executor_prepared_and_connections():
    from drl.MARL.c1_gui_viewer import structural_option_state
    goal = SimpleNamespace(goal_id='g1')
    relation = SimpleNamespace(source_module='m5', source_face='TOP', target_module='m1',
                               target_face='BOTTOM', docked=False)
    executor = SimpleNamespace(active_goals=(goal,), prepared_relation_snapshot=lambda: (relation,))
    env = SimpleNamespace(
        num_envs=1,
        native_dispatcher=SimpleNamespace(executor_provider=lambda i: executor),
        docking_managers=(SimpleNamespace(connections=('joint',)),),
    )
    snapshot = structural_option_state(env)
    assert snapshot.active_goal_ids == ('0:g1',)
    assert snapshot.prepared_relations == ((0, 'm5', 'TOP', 'm1', 'BOTTOM', False),)
    assert snapshot.connection_count == 1


def test_structural_scheduler_cli_defaults_event_for_accelerated_and_fixed_for_legacy():
    from drl.MARL.c1_gui_viewer import parse_args
    accelerated = parse_args(['--preflight', '--device', 'cpu'])
    assert accelerated.structural_scheduler == 'event'
    legacy = parse_args(['--preflight', '--device', 'cpu', '--runtime-mode', 'legacy'])
    assert legacy.structural_scheduler == 'fixed'
    explicit = parse_args(['--preflight', '--device', 'cpu', '--structural-scheduler', 'fixed'])
    assert explicit.structural_scheduler == 'fixed'


def test_event_scheduler_keeps_fixed_rate_when_continuous_resource_control_is_active():
    from drl.MARL.c1_gui_viewer import StructuralOptionScheduler
    scheduler = StructuralOptionScheduler(mode='event', control_hz=30.0, fallback_hz=2.0)
    active = state(active=('align-a',))
    assert scheduler.should_decide(state())
    scheduler.set_continuous_control(True)
    scheduler.observe_step(state(), active)
    assert scheduler.should_decide(active)
    assert scheduler.should_decide(active)
    scheduler.set_continuous_control(False)
    assert not scheduler.should_decide(active)


def test_sampled_resource_control_detects_selected_resource_prefix_only():
    import torch
    from types import SimpleNamespace
    from drl.MARL.c1_gui_viewer import sampled_resource_control
    context = SimpleNamespace(routing_snapshot={
        'resource_candidates': (object(), object()),
        'structural_bindings': ((object(), object()),),
    })
    resource = SimpleNamespace(action=SimpleNamespace(presence=torch.tensor([False, True, False])))
    structural = SimpleNamespace(action=SimpleNamespace(presence=torch.tensor([False, False, True])))
    assert sampled_resource_control([(resource, {})], [context]) is True
    assert sampled_resource_control([(structural, {})], [context]) is False


def test_event_scheduler_fills_parallel_wave_until_a_policy_step_adds_no_new_option():
    from drl.MARL.c1_gui_viewer import StructuralOptionScheduler
    scheduler = StructuralOptionScheduler(mode='event', control_hz=30.0, fallback_hz=2.0)
    idle = state()
    first = state(active=('align-07-01',))
    parallel = state(active=('align-05-01', 'align-07-01'))

    # Initial structural decision starts the first member of the wave.
    assert scheduler.should_decide(idle)
    scheduler.observe_step(idle, first)

    # Do not sleep yet: give H2 another chance to add a compatible member.
    assert scheduler.should_decide(first)
    scheduler.observe_step(first, parallel)

    # A second member joined, so allow one more fill decision.
    assert scheduler.should_decide(parallel)
    scheduler.observe_step(parallel, parallel)

    # No new option joined on the last decision: the wave is now sealed.
    assert not scheduler.should_decide(parallel)


def test_event_scheduler_keeps_pending_prepared_handoff_hot_until_next_option_starts():
    from drl.MARL.c1_gui_viewer import StructuralOptionScheduler
    scheduler = StructuralOptionScheduler(mode='event', control_hz=30.0, fallback_hz=2.0)
    pending = state(prepared=(('m5', 'TOP', 'm1', 'BOTTOM', False),))
    docking = state(active=('dock-05-01',), prepared=(('m5', 'TOP', 'm1', 'BOTTOM', False),))

    # A pending ALIGN->DOCK handoff is not generic idle: keep asking H2 every
    # control frame until a matching DOCK/re-ALIGN actually starts.
    assert scheduler.should_decide(pending)
    scheduler.observe_step(pending, pending)
    assert scheduler.should_decide(pending)
    scheduler.observe_step(pending, pending)
    assert scheduler.should_decide(pending)

    # As soon as an option starts, normal wave-fill/sealing semantics resume.
    scheduler.observe_step(pending, docking)
    assert scheduler.should_decide(docking)
    scheduler.observe_step(docking, docking)
    assert not scheduler.should_decide(docking)


def test_event_scheduler_does_not_treat_already_docked_prepared_row_as_pending_handoff():
    from drl.MARL.c1_gui_viewer import StructuralOptionScheduler
    scheduler = StructuralOptionScheduler(mode='event', control_hz=30.0, fallback_hz=2.0)
    docked_only = state(prepared=(('m7', 'BOTTOM', 'm1', 'TOP', True),), connections=1)

    assert scheduler.should_decide(docked_only)
    scheduler.observe_step(docked_only, docked_only)
    # No unresolved handoff remains, so this is ordinary idle and must fall
    # back to the low-rate safety poll rather than evaluating every frame.
    assert not scheduler.should_decide(docked_only)


def test_viewer_seals_native_align_groups_only_after_wave_fill_saturates():
    from drl.MARL.c1_gui_viewer import seal_active_align_waves

    calls = []
    executor = SimpleNamespace(seal_active_align_wave=lambda: calls.append(True) or (("a", "b"),))
    env = SimpleNamespace(
        num_envs=1,
        native_dispatcher=SimpleNamespace(executor_provider=lambda _i: executor),
    )

    assert seal_active_align_waves(env) == ((0, ("a", "b")),)
    assert calls == [True]


def test_wave_fill_saturates_only_after_policy_frame_adds_no_new_goal():
    from drl.MARL.c1_gui_viewer import structural_wave_fill_saturated

    idle = state()
    first = state(active=("a",))
    parallel = state(active=("a", "b"))

    assert not structural_wave_fill_saturated(True, idle, first)
    assert not structural_wave_fill_saturated(True, first, parallel)
    assert structural_wave_fill_saturated(True, parallel, parallel)
    assert not structural_wave_fill_saturated(False, parallel, parallel)
