"""Read-only characterization of existing event-clock high-frequency branches."""
import pytest


@pytest.mark.parametrize('reason,active,prepared,continuous,expected', [
    ('prepared_handoff', (), (('m5', 'TOP', 'm1', 'BOTTOM', False),), False, 60),
    ('continuous_resources', ('align-a',), (), True, 60),
    ('idle_fallback', (), (), False, 4),
    ('sealed_active_wave', ('align-a',), (), False, 0),
])
def test_existing_event_scheduler_high_frequency_causes(reason, active, prepared, continuous, expected):
    # Read-only characterization: these branches must not be throttled by a cache change.
    from drl.MARL.c1_gui_viewer import StructuralOptionScheduler, StructuralOptionState
    scheduler = StructuralOptionScheduler(mode='event', control_hz=30., fallback_hz=2.)
    state = StructuralOptionState(active, prepared, 0)
    assert scheduler.should_decide(state)
    scheduler.observe_step(state, state)
    scheduler.set_continuous_control(continuous)
    decisions = 0
    for _ in range(60):
        decisions += scheduler.should_decide(state)
        scheduler.observe_step(state, state)
    assert decisions == expected, reason
