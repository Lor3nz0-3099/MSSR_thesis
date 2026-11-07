from dataclasses import replace
from types import SimpleNamespace
import pytest

from drl.MARL.tests.test_c1_structure import frame
from drl.MARL.c1_structure import C1PrimitiveEvent


def monitor():
    from drl.MARL.c1_stagnation import EpisodeStagnationMonitor
    return EpisodeStagnationMonitor(SimpleNamespace(values={'stagnation': dict(enabled=True,
        strategic_window_s=20., global_window_s=25., option_window_s=10., option_grace_s=32.,
        recovery_grace_s=5., min_decision_opportunities=3)}))


def test_idle_global_stagnation_is_terminal_only_after_recovery_opportunities():
    m=monitor()
    for i in range(30):
        result=m.step(frame(float(i),edges=0),structural_score=0.,decision_opportunity=i%2==0)
    assert result.strategic and result.terminate and result.reason=='stagnation'


def test_time_without_policy_opportunities_is_not_enough_to_declare_stagnation():
    m=monitor()
    for i in range(60):
        result=m.step(frame(float(i),edges=0),structural_score=0.,decision_opportunity=False)
    assert not result.terminate


def test_one_failed_align_does_not_reset_episode_and_useful_recovery_clears_stagnation():
    m=monitor()
    for i in range(24):
        f=frame(float(i),edges=0)
        if i==1:f=replace(f,primitive_events=(C1PrimitiveEvent('g','align_faces','failed','NO_PATH','m0'),))
        assert not m.step(f,structural_score=0.,decision_opportunity=True).terminate
    assert not m.step(frame(24.,edges=1),structural_score=1/7,decision_opportunity=True).terminate
    assert not m.step(frame(40.,edges=1),structural_score=1/7,decision_opportunity=True).terminate


def test_long_productive_align_is_not_aborted_by_global_clock():
    m=monitor();base=frame(edges=0)
    goal={'primitive':'align_faces','connectors':(('m0','TOP'),('m1','BOTTOM')),'module_ids':('m0','m1')}
    for i in range(70):
        modules=list(base.modules)
        ports=dict(modules[1].connectors);ports['BOTTOM']=(2.-i*.02,0.,.1)
        modules[1]=replace(modules[1],connectors=tuple(ports.items()),position=(2.-i*.02,0.,.1))
        result=m.step(replace(base,time_s=float(i),modules=tuple(modules)),structural_score=0.,
            decision_opportunity=True,active_goals={'g':goal})
        assert not result.terminate


def test_oscillations_and_new_goal_ids_do_not_reset_progress_forever():
    m=monitor();base=frame(edges=0)
    goal={'primitive':'align_faces','connectors':(('m0','TOP'),('m1','BOTTOM')),'module_ids':('m0','m1')}
    for i in range(50):
        modules=tuple(replace(x,position=(x.position[0]+(.02 if i%2 else 0),*x.position[1:])) for x in base.modules)
        result=m.step(replace(base,time_s=float(i),modules=modules),structural_score=0.,
            decision_opportunity=True,active_goals={f'g{i//5}':goal})
    assert result.terminate


@pytest.mark.parametrize('task',['navigation','stairs','gap','manipulation'])
def test_functional_progress_resets_monitor_independent_of_task(task):
    m=monitor()
    for i in range(60):
        result=m.step(frame(float(i),edges=0),structural_score=0.,functional_progress=i*.01,
            decision_opportunity=False)
        assert not result.terminate
