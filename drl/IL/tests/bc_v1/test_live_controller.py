"""L1c temporal safety with a fake monotonic clock; no datasets or simulator."""
from concurrent.futures import TimeoutError
import importlib
import importlib.util
import threading

import pytest


def api():
    name = 'drl.IL.bc_v1.live_controller'
    assert importlib.util.find_spec(name), 'L1c temporal controller behavior is not implemented'
    return importlib.import_module(name)


class Clock:
    def __init__(self): self.now = 0.
    def __call__(self): return self.now


class Transport:
    def __init__(self):
        self.packets = []; self.clears = []; self.busy = set(); self.invalidated = []
    def poll(self):
        result = dict(busy=set(self.busy), invalidated=self.invalidated[:])
        self.invalidated.clear()
        return result
    def publish(self, held, goals, proof):
        self.packets.append((dict(held), tuple(goals), dict(proof)))
    def clear(self, reason): self.clears.append(reason)


def pan(value=2.):
    a = api()
    return a.RuntimeCommand(('m1', 'internal_motion'),
        {'command':'pan_velocity', 'module_ids':['m1'], 'parameters':{'rate_rad_s':value}},
        ('internal_motion:m1',), True)


def setup(infer=None):
    a = api(); clock = Clock(); transport = Transport(); calls = []
    def prediction(snapshot, availability):
        calls.append(snapshot.stamp)
        return (pan(),) if infer is None else infer(snapshot, availability)
    controller = a.LiveController('session', 'epoch', prediction, transport, clock=clock)
    return controller, clock, transport, calls


def observe(c, stamp, *, epoch='epoch'):
    c.tick(api().PhysicalSnapshot(epoch, stamp, {'synthetic':True}))


def arm(c, clock):
    observe(c, 1.)
    clock.now = .01
    observe(c, 2.)


def finish(c, clock, at=None):
    assert c.inflight is not None
    c.inflight.result(timeout=3.)
    if at is not None: clock.now = at
    c.tick()


def test_policy_clock_limits_starts_to_ten_hz():
    c, t, tr, calls = setup()
    try:
        arm(c,t); finish(c,t)
        for at, stamp in ((.05,3.),(.109,4.),(.111,5.)):
            t.now=at; observe(c,stamp)
            if c.inflight is not None: finish(c,t)
        assert calls == [2.,5.]
    finally: c.close()


def test_same_stamp_never_triggers_a_second_inference():
    c,t,tr,calls=setup()
    try:
        arm(c,t); finish(c,t); t.now=.2; observe(c,2.)
        assert calls == [2.] and c.inflight is None
    finally:c.close()


def test_new_stamp_allows_next_inference():
    c,t,tr,calls=setup()
    try:
        arm(c,t);finish(c,t);t.now=.2;observe(c,3.);finish(c,t)
        assert calls == [2.,3.]
    finally:c.close()


def test_missed_periods_do_not_create_backlog_or_catchup():
    c,t,tr,calls=setup()
    try:
        arm(c,t);finish(c,t);t.now=.7;observe(c,30.);finish(c,t)
        observe(c,31.)
        assert calls == [2.,30.] and c.inflight is None
    finally:c.close()


def test_stale_at_one_second_revokes_continuous_and_blocks_inference():
    c,t,tr,calls=setup()
    try:
        arm(c,t);finish(c,t);assert c.held
        t.now=1.01;c.tick()
        assert c.failed and not c.held and tr.clears and calls == [2.]
        observe(c,100.); assert c.inflight is None
    finally:c.close()


def test_stale_stops_keepalive_including_zero():
    c,t,tr,calls=setup(lambda *_:(pan(0.),))
    try:
        arm(c,t);finish(c,t);t.now=1.01;c.tick();n=len(tr.packets)
        t.now=1.5;c.tick()
        assert len(tr.packets)==n and not c.held
    finally:c.close()


def test_keepalive_does_not_infer_and_preserves_generation():
    c,t,tr,calls=setup()
    try:
        arm(c,t);finish(c,t);generation=next(iter(c.held.values())).generation
        t.now=.2;c.tick();t.now=.31;c.tick()
        assert calls == [2.] and len(tr.packets)>=3
        assert all(next(iter(p[0].values())).generation==generation for p in tr.packets)
    finally:c.close()


def test_identical_new_decision_receives_new_generation():
    c,t,tr,calls=setup()
    try:
        arm(c,t);finish(c,t);old=next(iter(c.held.values())).generation
        t.now=.2;observe(c,3.);finish(c,t)
        assert next(iter(c.held.values())).generation != old
    finally:c.close()


def test_zero_is_an_explicit_continuous_decision():
    c,t,tr,calls=setup(lambda *_:(pan(0.),))
    try:
        arm(c,t);finish(c,t)
        assert c.held[('m1','internal_motion')].command.payload['parameters']['rate_rad_s']==0.
        assert tr.packets and not tr.clears
    finally:c.close()


def test_old_startup_file_does_not_arm():
    c,t,tr,calls=setup()
    try:
        observe(c,200.);t.now=.2;observe(c,200.)
        assert not c.armed and not calls and not tr.packets
    finally:c.close()


def test_two_increasing_session_stamps_arm():
    c,t,tr,calls=setup()
    try:
        arm(c,t);finish(c,t)
        assert c.armed and calls==[2.]
    finally:c.close()


def test_regression_fails_closed_without_rearm():
    c,t,tr,calls=setup()
    try:
        arm(c,t);finish(c,t);t.now=.2;observe(c,1.5)
        assert c.failed and not c.held and tr.clears
        observe(c,5.); assert c.inflight is None
    finally:c.close()


def test_busy_at_result_admission_rejects_without_same_tick_retry():
    c,t,tr,calls=setup()
    try:
        arm(c,t);tr.busy.add('internal_motion:m1');finish(c,t)
        assert not c.held and calls==[2.] and c.inflight is None
        assert any(e['event']=='proposal_rejected' for e in c.events)
    finally:c.close()


@pytest.mark.parametrize('new_value',[1.2,0.])
def test_continuous_replacement(new_value):
    c,t,tr,calls=setup(lambda s,_:(pan(2. if s.stamp==2. else new_value),))
    try:
        arm(c,t);finish(c,t);t.now=.2;observe(c,3.);finish(c,t)
        assert c.held[('m1','internal_motion')].command.payload['parameters']['rate_rad_s']==new_value
    finally:c.close()


def test_runtime_invalidation_stops_old_generation_without_terminal_restore():
    c,t,tr,calls=setup()
    try:
        arm(c,t);finish(c,t)
        lease=c.held[('m1','internal_motion')]
        tr.invalidated=[(lease.command.key,lease.generation)]
        t.now=.2;c.tick();assert not c.held
        t.now=.4;c.tick();assert not c.held
        assert not tr.packets[-1][0]
    finally:c.close()


def test_completion_driven_dispatched_once_not_by_keepalive():
    a=api()
    target=a.RuntimeCommand(('m1','internal_motion'),
        {'command':'set_tilt','module_ids':['m1'],'parameters':{'angle_rad':.2}},
        ('internal_motion:m1',),False)
    c,t,tr,calls=setup(lambda *_:(target,))
    try:
        arm(c,t);finish(c,t);t.now=.2;c.tick();t.now=.4;c.tick()
        assert sum(len(goals) for _,goals,_ in tr.packets)==1 and not c.held
    finally:c.close()


def test_holder_and_safety_operate_while_inference_is_blocked():
    gate=threading.Event();entered=threading.Event()
    def infer(s,_):
        if s.stamp>2.:entered.set();gate.wait(3.)
        return (pan(),)
    c,t,tr,calls=setup(infer)
    try:
        arm(c,t);finish(c,t);t.now=.2;observe(c,3.);assert entered.wait(1.)
        count=len(tr.packets);t.now=.4;c.tick()
        assert len(tr.packets)>count and c.inflight is not None
        t.now=1.2;c.tick();assert c.failed and not c.held and tr.clears
        gate.set();c.inflight.result(timeout=3.);c.tick();assert not c.held
    finally:gate.set();c.close()


def test_nonfinite_command_result_fails_closed():
    c,t,tr,calls=setup(lambda *_:(pan(float('nan')),))
    try:
        arm(c,t)
        try:c.inflight.result(timeout=3.)
        except ValueError:pass
        c.tick();assert c.failed and not c.held and not tr.packets
    finally:c.close()


def test_expired_inference_result_is_never_dispatched_even_if_stream_recovers():
    c,t,tr,calls=setup()
    try:
        arm(c,t);c.inflight.result(timeout=3.)
        t.now=1.02;observe(c,3.)
        assert c.failed and not c.held and not tr.packets
    finally:c.close()


def test_session_mismatch_never_dispatches():
    c,t,tr,calls=setup()
    try:
        observe(c,1.,epoch='foreign')
        assert c.failed and not calls and not tr.packets
    finally:c.close()


def test_transport_failure_fails_closed():
    c,t,tr,calls=setup()
    def broken(*_):raise OSError('transport unavailable')
    try:
        arm(c,t);tr.publish=broken;finish(c,t)
        assert c.failed and not c.held
    finally:c.close()


def test_delayed_busy_reject_cannot_start_another_inference_in_same_tick():
    gate=threading.Event();entered=threading.Event()
    def infer(s,_):
        entered.set();gate.wait(2.);return (pan(),)
    c,t,tr,calls=setup(infer)
    try:
        arm(c,t);assert entered.wait(1.)
        tr.busy.add('internal_motion:m1');t.now=.2
        gate.set();c.inflight.result(timeout=2.)
        observe(c,3.)
        assert c.inflight is None and calls==[2.]
    finally:gate.set();c.close()


def test_native_only_phase_refreshes_state_proof_without_redispatching_goal():
    target=api().RuntimeCommand(('m1','internal_motion'),{'command':'set_tilt'},('internal_motion:m1',),False)
    c,t,tr,calls=setup(lambda *_:(target,))
    try:
        arm(c,t);finish(c,t)
        for at in (.2,.4,.6):t.now=at;c.tick()
        assert sum(bool(goals) for _,goals,_ in tr.packets)==1
        assert len(tr.packets)>=4 and not c.held
        assert all(not goals for _,goals,_ in tr.packets[1:])
    finally:c.close()
