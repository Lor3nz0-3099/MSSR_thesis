from dataclasses import replace
import pytest

from drl.MARL.c1_guided_feedback import TargetGraphProgress
from drl.MARL.tests.test_c1_guided_feedback import frame, physical_target_edge, load_target


def target():
    return load_target('navigation')


def connected_frame():
    return frame(0., tuple(physical_target_edge(target(),i) for i in range(7)))


def tracker():
    f=connected_frame()
    return TargetGraphProgress(target(),tuple(m.module_id for m in f.modules))


def value(parts):
    return sum(parts.values())


def test_target_shaping_signed_and_telescopes_through_dock_undock_cycle():
    t=tracker();f=connected_frame();empty=replace(f,connections=())
    first=value(t.shaping(f,discount_factor=.9,incompatible_weight=1.))
    second=value(t.shaping(empty,discount_factor=.9,incompatible_weight=1.))
    assert first>0 and second<0
    assert first+.9*second==pytest.approx(0.)


def test_holding_correct_edges_discount_and_terminal_have_correct_potential():
    t=tracker();f=connected_frame()
    value(t.shaping(f,discount_factor=.9))
    assert value(t.shaping(f,discount_factor=.9))<0
    assert value(t.shaping(f,discount_factor=.9,terminal=True))<0


def test_physical_shaping_discounted_once_equals_smdp_potential_difference():
    t=tracker();f=connected_frame();total=0.
    for j in range(4):
        parts=t.shaping(f,discount_factor=.9)
        total+=.9**j*value(parts)
    assert total==pytest.approx(.9**4*t.current_potential)


def test_duplicate_physical_edge_cannot_inflate_target_progress():
    f=connected_frame();t=tracker()
    one=replace(f,connections=f.connections[:1])
    duplicate=replace(one,connections=one.connections*2)
    assert t.observe(duplicate)[1] == pytest.approx(1/len(t.target_edges))


def test_incompatible_addition_is_negative_and_removal_is_restorative_without_farming():
    t=tracker();f=connected_frame()
    # RC-Car target has no BOTTOM--BOTTOM edge under any module relabeling.
    wrong=replace(f.connections[0],connector_a='BOTTOM',connector_b='BOTTOM')
    bad=replace(f,connections=(wrong,));empty=replace(f,connections=())
    a=value(t.shaping(bad,discount_factor=.99))
    assert a<0 and t.current_potential<0
    b=value(t.shaping(empty,discount_factor=.99))
    assert b>0 and a+.99*b==pytest.approx(0.)


def test_truncation_retains_potential_while_absorbing_terminal_cancels_it():
    f=connected_frame();a=tracker();b=tracker()
    x=value(a.shaping(f,discount_factor=.99))
    y=value(b.shaping(f,discount_factor=.99))
    assert x==y and a.current_potential==1.
    # Learner bootstraps time-limit final state. True terminal must use Phi=0.
    assert value(a.shaping(f,discount_factor=.99))==pytest.approx(-.01)
    assert value(b.shaping(f,discount_factor=.99,terminal=True))==pytest.approx(-1.)


def test_real_smdp_accumulator_telescopes_across_two_async_option_boundaries():
    import torch
    from drl.MARL.c1_smdp import DiscountedOptionAccumulator
    t=tracker();f=connected_frame();empty=replace(f,connections=())
    accumulators=[]
    for frames in ([f,f],[f,empty]):
        acc=DiscountedOptionAccumulator(.9)
        for state in frames:
            r=torch.tensor(value(t.shaping(state,discount_factor=.9)),dtype=torch.float64)
            acc.add(team_reward=r,owner_rewards={'a':r},mean_field_rewards={'a':r})
        accumulators.append(acc)
    assert accumulators[0].duration_steps==accumulators[1].duration_steps==2
    assert (accumulators[0].team_reward+.9**2*accumulators[1].team_reward).item()==pytest.approx(0.)
