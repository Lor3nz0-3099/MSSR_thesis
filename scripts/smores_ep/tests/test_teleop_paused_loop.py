"""Production-loop servicing boundary with controlled app updates, no Isaac."""
from types import SimpleNamespace

from smores_ep.scenarios import parallel_self_assembly


def service():
    function = getattr(parallel_self_assembly, "_service_teleop_runtime", None)
    assert function is not None, "missing paused runtime servicing in production scenario"
    return function


def test_paused_loop_polls_requests_and_updates_app_without_running_robot_work():
    calls = []
    runtime = SimpleNamespace(poll=lambda: calls.append("poll") or {"timeline_playing": False})
    app = SimpleNamespace(update=lambda: calls.append("app_update"))
    assert service()(runtime, app) is False
    assert calls == ["poll", "app_update"]


def test_resume_ack_allows_normal_iteration_without_extra_app_step():
    calls = []
    runtime = SimpleNamespace(poll=lambda: calls.append("poll") or {"timeline_playing": True})
    app = SimpleNamespace(update=lambda: calls.append("app_update"))
    assert service()(runtime, app) is True
    assert calls == ["poll"]


def test_resume_during_app_update_is_polled_before_next_robot_iteration():
    calls, playing = [], [False]
    runtime = SimpleNamespace(poll=lambda: calls.append("poll") or {"timeline_playing": playing[0]})

    def update():
        calls.append("app_update")
        playing[0] = True

    app = SimpleNamespace(update=update)
    assert service()(runtime, app) is False
    assert service()(runtime, app) is True
    assert calls == ["poll", "app_update", "poll"]
