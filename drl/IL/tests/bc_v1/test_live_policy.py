"""Live wiring/whitelist on synthetic observations; no dataset sample reads."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from drl.IL.tests.bc_v1.test_live_observation import settings,snapshot,builder,context
from drl.IL.bc_v1.model import BCGNN,load_model_config
from drl.IL.bc_v1.action_heads import load_action_head_config
from drl.IL.bc_v1.live_controller import PhysicalSnapshot

ROOT=Path(__file__).resolve().parents[4]


def api():
    name='drl.IL.bc_v1.live_policy'
    assert importlib.util.find_spec(name), 'Missing frozen L1b-to-L1a live inference composition'
    return importlib.import_module(name)


def test_full_live_preprocessing_precedes_model_and_decoder():
    a=api();kw=settings();cfg=ROOT/'configs/bc_v1';events=[]
    torch.manual_seed(0)
    model=BCGNN(kw['spec'],kw['feature_schema'],load_model_config(cfg/'gnn_v1.json'),
        action_head_config=load_action_head_config(cfg/'action_heads_v1.json'))
    policy=a.LivePolicy(model,kw,builder(),lambda _:context(),event_sink=events.append)
    raw=snapshot(8,rc=True)
    with torch.inference_mode():out=policy(PhysicalSnapshot('epoch',12.5,raw),{'busy':set()})
    assert [e['event'] for e in events]==['live_preprocessed','model_forward','actions_decoded']
    assert out and all(c.payload['command'] for c in out)
    assert policy.forward_count==1 and not model.training


def test_navigation_context_uses_physical_goal_and_excludes_route_and_shortcuts():
    a=api()
    course={'frame_id':'world','physical_track':{'goal_xyyaw':[5.,4.,.2],
        'cone_centers_xy_m':[[1.,2.]],'cone_radius_m':.05,'cone_height_m':.2},
        'navigation':{'waypoints_xyyaw':[[999.,999.,0.]],'cmd_vel':[42.]},
        'stage':4,'configuration_goal':'snake8','control_state':{'expert':True},'reward':999}
    task_config=json.loads((ROOT/'configs/bc_v1/task_context.json').read_text())
    ctx=a.navigation_context(course,task_config)
    fields={f.name:f.values for f in ctx.static_geometry}
    assert ctx.task_id=='navigation'
    assert fields['world.goal.xyyaw']==(5.,4.,.2)
    assert fields['world.cones.0.center_xy_m']==(1.,2.)
    assert not any(any(word in name for word in ('stage','configuration','control','expert','reward','waypoint')) for name in fields)
    changed=copy.deepcopy(course);changed['navigation']['waypoints_xyyaw']=[[0.,0.,0.]]
    changed['stage']=999
    assert a.navigation_context(changed,task_config)==ctx


def test_nonfinite_model_output_is_rejected_before_decode():
    a=api();kw=settings()
    class BadModel:
        def eval(self):return self
        def __call__(self,_):return {'broken':torch.tensor([float('nan')])}
    policy=a.LivePolicy(BadModel(),kw,builder(),lambda _:context())
    with pytest.raises(ValueError,match='Nonfinite'):
        policy(PhysicalSnapshot('epoch',12.5,snapshot()),{'busy':set()})
    assert policy.forward_count==1


def test_missing_physical_goal_cannot_use_nominal_waypoint_as_fallback():
    a=api();cfg=json.loads((ROOT/'configs/bc_v1/task_context.json').read_text())
    with pytest.raises(ValueError,match='physical goal'):
        a.navigation_context({'navigation':{'waypoints_xyyaw':[[1.,2.,0.]]}},cfg)


def test_frozen_loader_rejects_wrong_checkpoint_before_deserializing(tmp_path):
    a=api();bad=tmp_path/'bad.pt';bad.write_bytes(b'not the frozen checkpoint')
    with pytest.raises(ValueError,match='checkpoint SHA'):
        a.load_frozen_policy(ROOT,bad,builder(),device='cpu')


def test_runtime_pair_serialization_preserves_ordered_connectors_and_scope_members():
    from drl.IL.bc_v1 import bc_v1_contract as c
    from drl.IL.bc_v1.smores_adapter import SMORESAdapter
    a=api();spec=settings()['spec'];initiator,peer=c.ModuleRef('module-z'),c.ModuleRef('module-a')
    choice=c.ActionChoice('undock',initiator,participants=(peer,initiator),
        connectors=(c.ConnectorRef(initiator,'TOP'),c.ConnectorRef(peer,'BOTTOM')))
    result=a.decode_runtime_action(SMORESAdapter(spec),choice)
    assert result['module_ids']==['module-z','module-a']
    assert result['parameters']['face_a']=='TOP' and result['parameters']['face_b']=='BOTTOM'
    assert choice.participants==(peer,initiator)


def test_online_availability_is_derived_before_model_forward(monkeypatch):
    a=api();kw=settings();cfg=ROOT/'configs/bc_v1';trace=[]
    model=BCGNN(kw['spec'],kw['feature_schema'],load_model_config(cfg/'gnn_v1.json'),
        action_head_config=load_action_head_config(cfg/'action_heads_v1.json'))
    model.register_forward_pre_hook(lambda *_:trace.append('model'))
    original=a.DispatchAvailability
    def available(*args,**kwargs):
        trace.append('availability');return original(*args,**kwargs)
    monkeypatch.setattr(a,'DispatchAvailability',available)
    # Selection is covered separately; this test isolates the causal ordering
    # around a real H1/H2 forward from random synthetic structural proposals.
    monkeypatch.setattr(a,'decode_actions',lambda *args,**kwargs:())
    policy=a.LivePolicy(model,kw,builder(),lambda _:context())
    policy(PhysicalSnapshot('epoch',12.5,snapshot()),{'busy':set()})
    assert trace==['availability','model']
