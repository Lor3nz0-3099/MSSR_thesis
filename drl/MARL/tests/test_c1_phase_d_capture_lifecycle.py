"""CPU-only regression tests for diagnostic persistence prior to Kit close."""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import json
import sys
import pytest

from drl.MARL import c1_phase_d_capture as capture_module


def _invoke(monkeypatch, tmp_path, *, result):
    app_module=ModuleType('isaaclab.app')
    isaac_module=ModuleType('isaaclab')
    seen=[]
    class FakeAppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            pass
        def __init__(self, args):
            self.app=SimpleNamespace(close=lambda: seen.append('close'))
    app_module.AppLauncher=FakeAppLauncher
    isaac_module.app=app_module
    monkeypatch.setitem(sys.modules,'isaaclab',isaac_module)
    monkeypatch.setitem(sys.modules,'isaaclab.app',app_module)
    monkeypatch.setattr(capture_module,'_check_args',lambda args:None)
    def fake_capture(args):
        if isinstance(result,Exception):
            raise result
        if result=='success':
            (args.output/'bank.json').write_text('ok')
    monkeypatch.setattr(capture_module,'capture',fake_capture)
    run=tmp_path/'run';run.mkdir()
    args=['--reward-config',str(tmp_path/'any.json'),
          '--seeds','7000','12000','16000','20000',
          '--output',str(run/'bank')]
    return args,run,seen


def test_fatal_diagnostic_is_written_before_shutdown(monkeypatch,tmp_path):
    args,run,seen=_invoke(monkeypatch,tmp_path,result=RuntimeError('injected setup failure'))
    with pytest.raises(RuntimeError,match='injected setup failure'):
        capture_module.main(args)
    diagnostic=json.loads((run/'capture.diagnostic.json').read_text())
    assert diagnostic['phase']=='fatal_before_shutdown'
    assert diagnostic['error_type']=='RuntimeError'
    assert 'injected setup failure' in diagnostic['traceback']
    assert seen==['close']


def test_incomplete_capture_does_not_claim_success(monkeypatch,tmp_path):
    args,run,seen=_invoke(monkeypatch,tmp_path,result='incomplete')
    with pytest.raises(RuntimeError,match='without bank.json'):
        capture_module.main(args)
    diagnostic=json.loads((run/'capture.diagnostic.json').read_text())
    assert diagnostic['phase']=='fatal_before_shutdown'
    assert seen==['close']


def test_complete_capture_records_ready_before_shutdown(monkeypatch,tmp_path):
    args,run,seen=_invoke(monkeypatch,tmp_path,result='success')
    capture_module.main(args)
    diagnostic=json.loads((run/'capture.diagnostic.json').read_text())
    assert diagnostic['phase']=='bank_ready'
    assert seen==['close']
