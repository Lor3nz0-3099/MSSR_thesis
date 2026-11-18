"""Small, offline, deterministic threshold-sweep tests; never open G2 or checkpoints."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))  # avoid ROS2's external top-level `scripts` package
from calibrate_decoder import DecoderSweep, recommend_threshold, thresholds_from_text
from evaluate_checkpoint import MembershipAccumulator


def test_sweep_matches_existing_evaluator_for_thresholds_and_masks():
    logits=np.array([.2,.8,2.,-1.,1.8,.4,3.0],dtype=np.float32)
    labels=np.array([1,0,0,1,0,0,0],dtype=np.bool_)
    valid=np.array([1,1,0,1,1,1,1],dtype=np.bool_)
    owner=np.array([0,0,0,1,1,1,1],dtype=np.int64)
    tags=['resource:wheel_velocity','resource:wheel_velocity','structural:dock',
          'structural:undock','structural:dock','resource:pan_target','structural:align_faces']
    suite=DecoderSweep((-1.,0.,.5,1.,2.,4.))
    suite.update(logits,labels,valid,owner,tags,2)
    for row in suite.summary():
        threshold=row['threshold_logit']
        reference=MembershipAccumulator()
        reference.update(torch.tensor(logits),torch.tensor(labels),torch.tensor(valid),torch.tensor(owner),
                         tags,['navigation','reconfiguration'],['rc_car8','snake8'],threshold)
        original=reference.report()['global']
        for k in ('tp','fp','fn','tn','exact_sets'):
            assert row[k] == original.get(k,0)
        assert row['exact_set_accuracy'] == original['exact_set_accuracy']
    assert suite.masked==1
    assert suite.primitive_counts['structural:dock']==1
    # candidate with valid=False must not appear in the per-primitive total
    assert sum(x['tp']+x['fn'] for x in suite.per_primitive_summary()
               if x['primitive']=='structural:dock' and x['threshold_logit']==0) == 0


def test_recall_guard_protects_original_threshold():
    summary=[{'threshold_logit':0.,'recall':.99,'exact_set_accuracy':.70,'f1':.82},
             {'threshold_logit':1.,'recall':.96,'exact_set_accuracy':.85,'f1':.88},
             {'threshold_logit':.5,'recall':.985,'exact_set_accuracy':.73,'f1':.83}]
    base,selected,floor=recommend_threshold(summary,.01)
    assert base['threshold_logit']==0
    assert selected['threshold_logit']==.5
    assert floor==pytest.approx(.98)


def test_threshold_grid_requires_zero_and_unique():
    assert thresholds_from_text('0.5, 0,-0.5')==(-.5,0.,.5)
    for bad in ('1,2','0,0','0,inf','0,xxx',''):
        with pytest.raises(ValueError):thresholds_from_text(bad)


def test_no_positive_candidate_in_mask():
    suite=DecoderSweep((0.,1.))
    with pytest.raises(ValueError):
        suite.update([0.],[True],[False],[0],['resource:wheel_velocity'],1)
