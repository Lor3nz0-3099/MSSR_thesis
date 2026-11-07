import json
from pathlib import Path
import subprocess
import signal
import sys
import time

import pytest

from drl.MARL.c1_campaign_watchdog import supervise, _processes, _signal_identity


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def test_normal_exit_and_checkpoint_preserved(tmp_path):
    checkpoint = tmp_path / 'best.pt'
    checkpoint.write_bytes(b'already saved')
    output = tmp_path / 'watchdog.json'
    assert supervise([sys.executable, '-c', 'raise SystemExit(7)'], .8, output,
                     poll_seconds=.02) == 7
    report = json.loads(output.read_text())
    assert report['status'] == 'completed'
    assert report['returncode'] == 7
    assert checkpoint.read_bytes() == b'already saved'


def test_deadline_kills_separate_session_descendant_only(tmp_path):
    pidfile = tmp_path / 'child.pid'
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])
    code = ('import subprocess,sys,time,pathlib; '
            'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(10)"],start_new_session=True); '
            f'pathlib.Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(10)')
    output = tmp_path / 'watchdog.json'
    try:
        started = time.monotonic()
        assert supervise([sys.executable, '-c', code], .35, output, poll_seconds=.02) == 124
        assert time.monotonic() - started < 2
        assert not alive(int(pidfile.read_text()))
        assert unrelated.poll() is None
        report = json.loads(output.read_text())
        assert report['status'] == 'wall_timeout'
        assert report['emergency_checkpoint_attempted'] is False
        assert report['elapsed_seconds'] >= .35
    finally:
        unrelated.kill()
        unrelated.wait()


def test_normal_exit_cleans_previously_observed_orphan(tmp_path):
    pidfile = tmp_path / 'orphan.pid'
    code = ('import subprocess,sys,time,pathlib; '
            'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(10)"],start_new_session=True); '
            f'pathlib.Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(.15)')
    assert supervise([sys.executable, '-c', code], .8, tmp_path/'report.json', poll_seconds=.01) == 0
    assert not alive(int(pidfile.read_text()))


def test_supervisor_sigterm_cleans_detached_descendant(tmp_path):
    pidfile = tmp_path / 'detached.pid'
    output = tmp_path / 'interrupted.json'
    code = ('import subprocess,sys,time,pathlib; '
            'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(10)"],start_new_session=True); '
            f'pathlib.Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(10)')
    parent = subprocess.Popen([sys.executable, '-m', 'drl.MARL.c1_campaign_watchdog',
        '--wall-seconds', '2', '--output', str(output), '--', sys.executable, '-c', code])
    try:
        deadline = time.monotonic() + 1
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert pidfile.exists()
        parent.send_signal(signal.SIGTERM)
        assert parent.wait(timeout=1) == 143
        assert not alive(int(pidfile.read_text()))
        assert json.loads(output.read_text())['status'] == 'supervisor_interrupted'
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()


@pytest.mark.parametrize('seconds', [0, -1, 86401, float('nan')])
def test_invalid_budget_rejected_before_launch(tmp_path, seconds):
    with pytest.raises(ValueError):
        supervise([sys.executable, '-c', 'pass'], seconds, tmp_path/'report.json')


def test_stale_process_identity_cannot_signal_live_process():
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)'])
    try:
        start = _processes()[child.pid][2]
        assert not _signal_identity(child.pid, start + 1, signal.SIGKILL)
        assert child.poll() is None
    finally:
        child.kill()
        child.wait()
