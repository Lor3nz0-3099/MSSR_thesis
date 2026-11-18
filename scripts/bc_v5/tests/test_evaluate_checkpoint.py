"""Pure synthetic metric tests. Never read G2 or model checkpoints."""
import math
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

# Avoid ROS 2 Humble's installed top-level `scripts` package, which shadows
# MSSR_thesis/scripts (a namespace directory, not a Python package).
# Load the in-repository evaluator by its exact file path.
EVALUATOR_PATH = Path(__file__).resolve().parents[1] / "evaluate_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("mssr_bc_v5_local_evaluator", EVALUATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)
MembershipAccumulator = EVALUATOR.MembershipAccumulator
NumericAccumulator = EVALUATOR.NumericAccumulator
batch_tags = EVALUATOR.batch_tags


def test_masked_candidate_never_counts_as_false_positive():
    m=MembershipAccumulator()
    m.update(torch.tensor([1.,10.,-1.,-1.,1.,10.]),
             torch.tensor([True,False,False,True,False,False]),
             torch.tensor([True,False,True,True,True,False]),
             torch.tensor([0,0,0,1,1,1]),
             ['resource:wheel_velocity','structural:dock','resource:pan_velocity',
              'structural:undock','resource:tilt_target','structural:align_faces'],
             ['navigation','reconfiguration'],['rc_car8','snake8'],0.)
    r=m.report()
    assert r['global']['tp']==1
    assert r['global']['fn']==1
    assert r['global']['fp']==1
    assert r['global']['masked_candidates']==2
    assert r['global']['exact_sets']==1
    assert r['global']['exact_set_accuracy']==.5
    assert r['primitive']['structural:align_faces'] if 'structural:align_faces' in r['primitive'] else True
    assert r['primitive']['resource:tilt_target']['fp']==1
    assert 'structural:dock' not in r['primitive']


def test_oracle_regression_uses_component_units_and_range_normalization():
    b=NumericAccumulator()
    b.add('oracle_positive_physical','resource:wheel_velocity','payload[0]','rad/s',3,1)
    b.add('oracle_positive_physical','resource:wheel_velocity','payload[0]','rad/s',6,2)
    r=b.rows()
    assert len(r)==1
    assert r[0]['count']==2
    assert r[0]['mae']==3
    assert r[0]['rmse']==math.sqrt(10)
    assert r[0]['nrmse_range']==math.sqrt(10)/1


def test_batch_tags_preserves_resource_then_structural_order():
    a=NS(task_stratum='navigation',morphology_stratum='rc_car8')
    b=NS(task_stratum='assembly',morphology_stratum='snake8')
    def make(a, modes, actions, args):
        route={'audit':a,'resource_candidates':[NS(mode=NS(mode_id=x)) for x in modes],
               'structural_bindings':[(NS(action_id=x),None) for x in actions],
               'argument_domains':list(range(args))}
        return {'routing':route}
    tags,tasks,morphs,r_map,s_map,arg_map=batch_tags([
        make(a,['wheel_velocity'],['dock'],2),
        make(b,['tilt_target','pan_velocity'],['align_faces'],1)])
    assert tags==['resource:wheel_velocity','resource:tilt_target','resource:pan_velocity',
                  'structural:dock','structural:align_faces']
    assert r_map==[(0,0),(1,0),(1,1)]
    assert s_map==[(0,0),(1,0)]
    assert arg_map==[(0,0),(0,1),(1,0)]
    assert tasks==['navigation','assembly']
    assert morphs==['rc_car8','snake8']


def test_real_bc_v5_tensor_abi_resource_and_structural_numeric():
    """Tiny built-in fixture; model forward + real supervision/codec, no data files read."""
    from dataclasses import replace
    from pathlib import Path
    from drl.IL.tests.bc_v1.g2_fixtures import fixture, tensorize, with_structural
    from drl.IL.bc_v1.bc_v5_binary import collate_binary_v5, compute_binary_v5_loss
    from drl.IL.bc_v1.model import load_model_config
    from drl.IL.bc_v1.action_set_head import ActionSetPolicy, load_action_set_head_config
    resource_regression = EVALUATOR.resource_regression
    structural_regression = EVALUATOR.structural_regression

    root = Path(__file__).resolve().parents[3]
    src, kwargs = fixture(3, 2)
    alternate = replace(with_structural(src, skill=True),
                        audit=replace(src.audit,transition_index=1))
    samples = [tensorize(src,kwargs),tensorize(alternate,kwargs)]
    overlay={ (s['routing']['audit'].episode_id,s['routing']['audit'].transition_index): {
        'episode_id':s['routing']['audit'].episode_id,
        'transition_index':s['routing']['audit'].transition_index,
        'exclude':False,'blocked':[]}
        for s in samples }
    batch = collate_binary_v5(samples, overlay)
    h1=load_model_config(root/'configs/bc_v1/gnn_v1.json')
    h2=load_action_set_head_config(root/'configs/bc_v1/action_set_head_v1.json')
    model=ActionSetPolicy(kwargs['spec'],kwargs['feature_schema'],h1,h2)
    model.eval()
    with torch.no_grad():
        prediction=model(batch['inputs'])
        loss, _=compute_binary_v5_loss(prediction,batch)
    assert torch.isfinite(loss)
    tags,tasks,morphs,r_map,s_map,a_map=batch_tags(samples)
    accumulator=MembershipAccumulator()
    accumulator.update(prediction['presence_logits'],batch['supervision']['action_set']['presence'],
                       batch['supervision']['action_set']['valid'],prediction['candidate_graph'],
                       tags,tasks,morphs,0.)
    selected=(prediction['presence_logits'].detach().cpu()>=0).tolist()
    resource_regression(prediction,batch,samples,r_map,accumulator,kwargs['normalizers'],selected)
    structural_regression(prediction,batch,samples,s_map,a_map,len(r_map),accumulator,kwargs['normalizers'],selected)
    metrics=accumulator.report()
    assert metrics['global']['examples']==2
    assert any(row['primitive']=='resource:pace' and row['unit']=='rad/s' and row['count']==2
               for row in metrics['regression'])
    assert any(row['primitive']=='structural:gravity_settle' and row['unit']=='s'
               for row in metrics['regression'])
