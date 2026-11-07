import builtins
import sys
from pathlib import Path
import pytest

from drl.MARL.c1_phase_c_validation import parse_args,preflight


@pytest.mark.parametrize('mode',['guided','discovery'])
def test_real_checkpoint_preflight_separates_training_and_runtime_contract(tmp_path,monkeypatch,mode):
    original=builtins.__import__
    def guard(name,*args,**kwargs):
        if name.split('.')[0] in ('isaaclab','isaacsim','omni'):
            raise AssertionError('Simulator import in CPU preflight')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guard)
    out=tmp_path/'uncreated'
    cfg=preflight(parse_args(['--output',str(out),'--mode',mode,'--preflight','--device','cpu']))
    assert cfg['completed_updates']==78 and cfg['policy_version']==67
    assert cfg['runtime_probe_config']!=cfg['training_provenance_probe_config']
    assert cfg['episode_seconds']==120 and cfg['physics_hz']==240 and cfg['decision_hz']==30
    assert cfg['actor_only'] and cfg['structural_scheduler']=='event'
    assert not out.exists()


def test_driver_rejects_output_reuse(tmp_path):
    with pytest.raises(ValueError,match='NEW'):
        preflight(parse_args(['--output',str(tmp_path),'--preflight']))


def test_real_actor_restore_with_phase_c_config_uses_all_mappo_weights(tmp_path,monkeypatch):
    import torch
    from drl.MARL.c1_runner import build_observation_bridge
    from drl.MARL.c1_viewer_policy import build_viewer_actor
    def forbidden(*a,**k):raise AssertionError('Training operation in frozen validation')
    monkeypatch.setattr(torch.optim.Optimizer,'__init__',forbidden)
    monkeypatch.setattr(torch.Tensor,'backward',forbidden)
    cfg=preflight(parse_args(['--output',str(tmp_path/'new'),'--device','cpu']))
    actor=build_viewer_actor(build_observation_bridge(Path.cwd()),cfg)
    state=torch.load(cfg['checkpoint'],weights_only=True,map_location='cpu')['smores_01']['policy']
    assert all(torch.equal(v,actor.state_dict()[k]) for k,v in state.items())
    assert all(not p.requires_grad and p.grad is None for p in actor.parameters())
    assert not any(n=='isaaclab' or n.startswith('isaaclab.') for n in sys.modules)
