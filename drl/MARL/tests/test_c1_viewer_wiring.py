import sys
from types import ModuleType, SimpleNamespace


def test_viewer_scene_wiring_is_gui_only(monkeypatch):
    from drl.MARL.c1_gui_viewer import configure_viewer_scene
    calls=[]
    fake = ModuleType('drl.MARL.c1_viewer_visuals')
    fake.configure_viewer_visuals = lambda stage, roots: calls.append((stage, roots)) or {'enabled': True}
    monkeypatch.setitem(sys.modules, 'drl.MARL.c1_viewer_visuals', fake)
    env=SimpleNamespace(sim=SimpleNamespace(stage='stage'), module_roots_by_env=({'m':'/m'},))
    assert configure_viewer_scene(env, headless=True) == {'enabled': False, 'reason': 'headless'}
    assert not calls
    assert configure_viewer_scene(env, headless=False) == {'enabled': True}
    assert calls == [('stage', {'m':'/m'})]
