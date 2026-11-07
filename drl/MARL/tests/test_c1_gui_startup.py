"""Startup contract against local IsaacLab resolvers, without launching Kit."""
import ast
import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[3]
LAUNCHER=Path('/home/lorenzo/IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py')


def local_resolver():
    # Compile ONLY the actual pure argument resolvers. Never import isaaclab.app
    # or call AppLauncher.__init__ / SimulationApp. This catches upstream default
    # changes rather than pretending that headless=False guarantees a window.
    tree=ast.parse(LAUNCHER.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AppLauncher')
    names={'_parse_visualizer_csv','_normalize_visualizer_intent','_resolve_visualizer_settings',
           '_resolve_headless_settings','_resolve_viewport_settings','_rendering_enabled'}
    cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    module=ast.Module(body=[cls],type_ignores=[])
    namespace=dict(os=os,argparse=argparse,Any=object,logger=logging.getLogger(__name__))
    exec(compile(ast.fix_missing_locations(module),str(LAUNCHER),'exec'),namespace)
    C=namespace['AppLauncher'];C._APPLAUNCHER_CFG_INFO={'headless':([bool],False)}
    resolver=C();resolver._livestream=0;resolver._xr=False;resolver._enable_cameras=False
    return resolver


@pytest.mark.parametrize('headless',[False,True])
def test_gui_intent_survives_actual_local_app_launcher_resolution(monkeypatch,headless):
    from drl.MARL.c1_gui_viewer import parse_args,launcher_options
    monkeypatch.delenv('HEADLESS',raising=False)
    args=parse_args(['--headless'] if headless else [])
    options=launcher_options(args)
    r=local_resolver();r._resolve_visualizer_settings(options)
    r._resolve_headless_settings(options,0,0);r._resolve_viewport_settings(options)
    assert r._headless is headless
    assert r._render_viewport is (not headless)
    assert r._rendering_enabled() is (not headless)
    assert r._cli_visualizer_types==([] if headless else ['kit'])


def test_old_headless_false_without_kit_is_forced_headless(monkeypatch):
    monkeypatch.delenv('HEADLESS',raising=False)
    r=local_resolver();options=dict(headless=False)
    r._resolve_visualizer_settings(options);r._resolve_headless_settings(options,0,0)
    assert options['headless'] is True  # Precise reproduction of the reported cause.


@pytest.mark.parametrize('headless',[False,True])
def test_effective_app_mode_published_before_environment_creation(headless):
    from drl.MARL.c1_gui_viewer import configure_viewer_display
    values={'/isaaclab/render/active_viewport':not headless}
    settings=SimpleNamespace(get=lambda name:values.get(name),
                             set_bool=lambda name,value:values.__setitem__(name,value),
                             set_int=lambda name,value:values.__setitem__(name,value))
    actual=configure_viewer_display(SimpleNamespace(config={'headless':headless}),settings,requested_gui=not headless)
    assert actual['requested_gui'] is (not headless)
    assert actual['effective_headless'] is headless
    assert actual['rendering_enabled'] is (not headless)
    assert values['/isaaclab/has_gui'] is (not headless)
    if not headless:
        from drl.MARL.c1_gui_viewer import VIEWER_WIDTH, VIEWER_HEIGHT
        assert values['/app/window/width'] == VIEWER_WIDTH
        assert values['/app/window/height'] == VIEWER_HEIGHT
        assert values['/app/renderer/resolution/width'] == VIEWER_WIDTH
        assert values['/app/renderer/resolution/height'] == VIEWER_HEIGHT
    with pytest.raises(RuntimeError):
        configure_viewer_display(SimpleNamespace(config={'headless':not headless}),settings,requested_gui=not headless)


def test_gui_preflight_retains_single_env_actor_only_tier1_contract(monkeypatch):
    from drl.MARL.c1_gui_viewer import parse_args,preflight
    monkeypatch.delenv('HEADLESS',raising=False)
    for extra,gui in (([],True),(['--headless'],False)):
        cfg=preflight(parse_args(['--device','cpu',*extra]))
        assert cfg['requested_gui'] is gui and cfg['effective_headless'] is (not gui)
        assert cfg['rendering_enabled'] is gui
        assert cfg['num_envs']==1 and cfg['task']=='navigation'
        assert cfg['runtime_mode']=='accelerated' and cfg['actor_only'] and cfg['readonly_state_cache']
        assert cfg['optimizer_updates']==0 and cfg['training_memory_constructed'] is False
        assert cfg['presence_threshold_logit']==4.5


def test_launcher_and_display_setup_precede_environment_and_no_top_level_kit_imports():
    import drl.MARL.c1_gui_viewer as viewer
    tree=ast.parse(Path(viewer.__file__).read_text())
    imports=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    names=[n.module if isinstance(n,ast.ImportFrom) else n.names[0].name for n in imports]
    assert not any(name.startswith(('isaaclab','isaacsim','omni','pxr')) for name in names)
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    calls={}
    for node in ast.walk(run):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):calls.setdefault(node.func.id,node)
    assert calls['AppLauncher'].lineno<calls['configure_viewer_display'].lineno<calls['MSSRC1IsaacLabEnv'].lineno
    assert any(k.arg is None and isinstance(k.value,ast.Call) and isinstance(k.value.func,ast.Name)
               and k.value.func.id=='launcher_options' for k in calls['AppLauncher'].keywords)
    assert any(k.arg=='before_app_close' and isinstance(k.value,ast.Name)
               and k.value.id=='persist_result' for k in calls['finalize_viewer'].keywords)


def test_summary_persisted_before_fast_shutdown_at_one_second_cutoff(tmp_path):
    # SimulationApp.close() can terminate via os._exit; a subprocess lets us
    # reproduce that shutdown contract safely, with NO simulator imports.
    code=r'''
import json,os,sys
from pathlib import Path
from types import SimpleNamespace
from drl.MARL.c1_gui_viewer import finalize_viewer
folder=Path(sys.argv[1]);result=dict(status='completed',decisions_completed=30,cutoff_reason='horizon_reached')
app=SimpleNamespace(close=lambda:os._exit(0))
env=SimpleNamespace(_sim_step_counter=240,physics_dt=1/240,reset_records=[],close=lambda:None)
def persist():
    (folder/'summary.json').write_text(json.dumps(result))
finalize_viewer(env=env,application=app,actor=None,recorder=None,before=None,sim_start=0.,result=result,folder=folder,
    write_json=lambda *a:None,actor_digest=None,before_app_close=persist)
'''
    child=subprocess.run([sys.executable,'-c',code,str(tmp_path)],cwd=ROOT,capture_output=True,text=True,timeout=20)
    assert child.returncode==0,child.stderr
    result=json.loads((tmp_path/'summary.json').read_text())
    assert result['simulated_seconds']==1. and result['decisions_completed']==30
    assert result['status']=='completed'


def test_preflight_import_chain_has_no_isaac_or_omniverse_initialization():
    code="""
import sys,json
from drl.MARL.c1_gui_viewer import parse_args,preflight
preflight(parse_args(['--preflight','--device','cpu']))
forbidden={'isaaclab','isaacsim','omni','carb','pxr','usdrt'}
print(json.dumps([n for n in sys.modules if n.split('.')[0] in forbidden]))
"""
    child=subprocess.run([sys.executable,'-c',code],cwd=ROOT,capture_output=True,text=True,timeout=20)
    assert child.returncode==0,child.stderr
    assert json.loads(child.stdout)==[]


def test_summary_callback_failure_still_closes_application_and_keeps_original_error(tmp_path):
    from drl.MARL.c1_gui_viewer import finalize_viewer
    closed=[];result=dict(status='error',error='original error')
    def fail():raise OSError('summary unavailable')
    finalize_viewer(env=None,application=SimpleNamespace(close=lambda:closed.append(True)),actor=None,recorder=None,
        before=None,sim_start=None,result=result,folder=tmp_path,write_json=None,actor_digest=None,before_app_close=fail)
    assert closed==[True] and result['error']=='original error'
    assert result['finalization_errors'][0]['operation']=='before_app_close'
