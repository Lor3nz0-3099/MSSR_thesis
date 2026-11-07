"""Linux wall-clock supervisor for one explicitly supplied campaign command.

The learner's WallBudget should save and stop before this emergency deadline.
This outer process owns no learner state and never attempts a checkpoint after
a timeout. Existing periodic/best/final artifacts remain untouched. SIGKILL of
the supervisor itself, or tasks stuck in uninterruptible kernel sleep, cannot
be repaired by a userspace supervisor.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


def _processes():
    """Return pid -> (parent, process group, start ticks, state) from /proc."""
    result = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # comm can contain spaces and parentheses; fields after its final
            # ')' start with state (field 3), and starttime is field 22.
            fields = (entry / 'stat').read_text().rsplit(') ', 1)[1].split()
            result[int(entry.name)] = (int(fields[1]), int(fields[2]), int(fields[19]), fields[0])
        except (OSError, ValueError, IndexError):
            continue
    return result


def _discover(owned, snapshot):
    """Extend a recorded lineage, including previously observed orphan roots."""
    live = {pid for pid, start in owned.items()
            if pid in snapshot and snapshot[pid][2] == start}
    changed = True
    while changed:
        changed = False
        for pid, (parent, _group, start, _state) in snapshot.items():
            if parent in live and pid not in live:
                owned[pid] = start
                live.add(pid)
                changed = True
    return live


def _signal_identity(pid, start, sig):
    # pidfd pins identity across the interval between checking /proc and kill,
    # closing the PID-reuse race of an ordinary os.kill(pid, signal).
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return False
    try:
        row = _processes().get(pid)
        if row is None or row[2] != start or row[3] == 'Z':
            return False
        signal.pidfd_send_signal(fd, sig)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(fd)


def _cleanup(owned, root_pid):
    """Freeze lineage before its final scan, then kill only pinned identities.

    Individual pidfds deliberately cover every member of the owned process
    groups, including detached validation sessions; killpg's numeric group ID
    has no equivalent identity pin and could race group/PID reuse. No group
    outside the recorded ancestry is signalled.
    """
    stopped = set()
    killed = []
    freeze_deadline = time.monotonic() + .2
    # Freeze the root first, then newly discovered descendants. Once all live
    # lineage members are stopped none can spawn a child during the kill scan.
    if root_pid in owned:
        _signal_identity(root_pid, owned[root_pid], signal.SIGSTOP)
        stopped.add((root_pid, owned[root_pid]))
    while True:
        snapshot = _processes()
        live = _discover(owned, snapshot)
        pending = [(pid, owned[pid]) for pid in live
                   if snapshot[pid][3] != 'Z' and (pid, owned[pid]) not in stopped]
        if not pending:
            running = [pid for pid in live if snapshot[pid][3] not in ('T', 't', 'Z')]
            if not running or time.monotonic() >= freeze_deadline:
                break
            time.sleep(.001)
            continue
        for pid, start in pending:
            _signal_identity(pid, start, signal.SIGSTOP)
            stopped.add((pid, start))
    # Children first; stopped root cannot create replacements or reap a child
    # into a recycled PID while identities are being selected.
    for pid, start in sorted(stopped, key=lambda item: item[0] == root_pid):
        if _signal_identity(pid, start, signal.SIGKILL):
            killed.append(pid)
    # SIGKILL delivery is asynchronous. Observe completion briefly without
    # waiting for orphan zombies to be reaped by their external init process.
    kill_deadline = time.monotonic() + .2
    while time.monotonic() < kill_deadline:
        snapshot = _processes()
        if not any(pid in snapshot and snapshot[pid][2] == owned[pid]
                   and snapshot[pid][3] != 'Z' for pid in killed):
            break
        time.sleep(.001)
    return killed


class _ParentSignal(BaseException):
    def __init__(self, signum):
        self.signum = signum


def supervise(command, wall_seconds, output, *, poll_seconds=1., cwd=None):
    """Run a shell-free command; return child status, or 124 on hard timeout.

    The monotonic clock starts before Popen and includes application startup,
    assembly, updates, validation and normal child teardown. Polling never
    sleeps beyond the remaining deadline. Signalable descendants observed
    while the root lives remain owned after reparenting. Trusted children must
    not deliberately double-fork and exit between ancestry scans.
    """
    if (isinstance(wall_seconds, bool) or not isinstance(wall_seconds, (int, float))
            or not math.isfinite(wall_seconds) or not 0 < wall_seconds <= 86400):
        raise ValueError('wall_seconds must be finite, positive and at most 86400')
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError('poll_seconds must be finite and positive')
    if not isinstance(command, (list, tuple)) or not command or any(
            not isinstance(item, str) or not item for item in command):
        raise ValueError('command must be a nonempty argument list; shell strings are forbidden')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('watchdog output already exists; choose a new report path')
    started = time.monotonic()
    report = dict(schema='mssr.c1_campaign_watchdog.v1', command=list(command),
        wall_seconds=float(wall_seconds), status='starting', returncode=None,
        emergency_checkpoint_attempted=False,
        checkpoint_policy='Existing checkpoint artifacts are preserved. Emergency termination cannot save a hung learner.',
        killed_pids=[])
    child = None
    owned = {}
    previous_handlers = {}

    def interrupted(signum, frame):
        raise _ParentSignal(signum)

    try:
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[sig] = signal.signal(sig, interrupted)
        child = subprocess.Popen(command, cwd=cwd, start_new_session=True)
        row = _processes().get(child.pid)
        if row is not None:
            owned[child.pid] = row[2]
        report['root_pid'] = child.pid
        while True:
            _discover(owned, _processes())
            code = child.poll()
            if code is not None:
                report.update(status='completed', returncode=code)
                return code
            remaining = wall_seconds - (time.monotonic() - started)
            if remaining <= 0:
                report.update(status='wall_timeout', returncode=124)
                return 124
            time.sleep(min(poll_seconds, remaining))
    except _ParentSignal as stopped:
        code = 128 + stopped.signum
        report.update(status='supervisor_interrupted', returncode=code, signal=stopped.signum)
        return code
    except BaseException as error:
        report.update(status='supervisor_error', error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        # Repeated termination signals cannot interrupt cleanup midway.
        for sig in previous_handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if child is not None:
                report['killed_pids'] = _cleanup(owned, child.pid)
                try:
                    child.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    report['root_still_present_after_kill'] = True
            snapshot = _processes()
            report['remaining_owned_live_pids'] = [pid for pid, start in owned.items()
                if pid in snapshot and snapshot[pid][2] == start and snapshot[pid][3] != 'Z']
            report['elapsed_seconds'] = time.monotonic() - started
            report['owned_processes'] = [dict(pid=pid, start_ticks=start) for pid, start in owned.items()]
            output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        finally:
            for sig, previous in previous_handlers.items():
                signal.signal(sig, previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wall-seconds', type=float, default=86400.)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    try:
        code = supervise(command, args.wall_seconds, args.output)
    except ValueError as error:
        parser.error(str(error))
    # Conventional shell status for a normally exited child killed by signal.
    return 128 - code if code < 0 else code


if __name__ == '__main__':
    raise SystemExit(main())
