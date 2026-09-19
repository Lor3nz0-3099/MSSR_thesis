"""A launched structural expert must be stopped as one scoped process group."""

import signal

from mssr_expert.teleop.structural_macro import _default_terminate


class FakeProcess:
    def __init__(self, pid=None):
        self.pid = pid
        self.terminate_calls = 0

    def terminate(self):
        self.terminate_calls += 1


def test_default_terminate_targets_spawned_posix_process_group(monkeypatch):
    process = FakeProcess(pid=43210)
    calls = []

    monkeypatch.setattr(
        "mssr_expert.teleop.structural_macro.os.killpg",
        lambda pgid, sig: calls.append((pgid, sig)),
    )

    _default_terminate(process)

    assert calls == [(43210, signal.SIGTERM)]
    assert process.terminate_calls == 0


def test_default_terminate_falls_back_to_process_terminate_without_pid():
    process = FakeProcess(pid=None)

    _default_terminate(process)

    assert process.terminate_calls == 1
