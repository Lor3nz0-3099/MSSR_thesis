import json
from pathlib import Path
import pytest
from drl.MARL.c1_config import C1Settings


@pytest.mark.parametrize('section,key,value',[
    ('stagnation','orientation_quantization',0.),('stagnation','progress_epsilon',float('nan')),
    ('stagnation','enabled','yes'),('stagnation','min_decision_opportunities',0),
    ('redundancy','continuous_window_s',-1.),('redundancy','enabled','yes'),
])
def test_invalid_monitor_configuration_fails_before_simulator(tmp_path,section,key,value):
    cfg=json.loads(Path('configs/marl/c1_guided_phase_c_v1.json').read_text())
    cfg[section][key]=value
    p=tmp_path/'bad.json';p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):C1Settings.load(p)


@pytest.mark.parametrize('change',['discovery_target','guided_generic','unequal_scales'])
def test_invalid_reward_contract_fails_closed(tmp_path,change):
    cfg=json.loads(Path('configs/marl/c1_guided_phase_c_v1.json').read_text())
    if change=='discovery_target':cfg['episode_mode']='discovery'
    elif change=='guided_generic':cfg['weights']['reward_valid_docking']=1.
    else:cfg['weights']['reward_target_graph_discount']=5.
    p=tmp_path/'bad.json';p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):C1Settings.load(p)
