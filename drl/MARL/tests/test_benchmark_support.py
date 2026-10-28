"""Instrumentation uses measured monotonic time and never touches policy state."""
import pytest


def test_profile_separates_nested_exclusive_from_inclusive_and_records_failures():
    from drl.MARL.benchmark_support import MonotonicProfile
    ticks=iter([0.,1.,3.,5.,6.,9.])
    profile=MonotonicProfile(clock=lambda:next(ticks))
    with profile.measure('outer'):
        with profile.measure('inner'):
            pass
    with pytest.raises(RuntimeError):
        with profile.measure('failed'):
            raise RuntimeError('unit')
    assert profile.snapshot()['outer']==dict(calls=1,inclusive_s=5.,exclusive_s=3.)
    assert profile.snapshot()['inner']['exclusive_s']==2.
    assert profile.snapshot()['failed']['inclusive_s']==3.


def test_profile_summary_uses_global_wall_and_task_denominators():
    from drl.MARL.benchmark_support import MonotonicProfile
    profile=MonotonicProfile(clock=lambda:10.)
    profile.observe('navigation',reward=2.,progress=.1,success=False,conflict=True,invalid=False,accepted=False)
    profile.observe('navigation',reward=4.,progress=.3,success=True,conflict=False,invalid=True,accepted=False)
    summary=profile.summary(transitions=2,updates=1,wall_s=20.)
    assert summary['transitions_per_s']==.1
    assert summary['eta_seconds']['10000']==100000.
    assert summary['domains']['navigation']['reward_mean']==3.
    assert summary['domains']['navigation']['conflict_rate']==.5
    assert summary['domains']['navigation']['progress_mean']==pytest.approx(.2)
