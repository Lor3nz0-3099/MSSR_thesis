"""SMDP next-state graphs are isolated once at their final storage boundary."""
from copy import deepcopy as original_deepcopy
from unittest.mock import patch
import torch

from drl.MARL.tests.c1_smdp_fixtures import make_learner,frame
from drl.MARL.mappo import StructuredRolloutSidecar


def test_sidecar_clear_has_an_explicit_diagnostic_epoch():
    sidecar=StructuredRolloutSidecar()
    epoch=sidecar._diagnostic_generation
    sidecar.append_step([object()]);sidecar.clear()
    assert sidecar._diagnostic_generation==epoch+1


def test_stored_next_graph_has_one_deepcopy_and_does_not_alias_future_state():
    learner=make_learner()
    frame(learner,[True,True],[0.,10.],[1.,2.])
    copied=[]
    def record(value,*a,**kw):
        if torch.is_tensor(value):copied.append(value.detach().clone())
        return original_deepcopy(value,*a,**kw)
    with patch('drl.MARL.mappo.deepcopy',record):
        frame(learner,[True,False],[2.,11.],[3.,4.])
    # One pre-step closure plus sibling padding: final storage owns one next
    # graph copy for each vector slot, never an overwritten intermediate copy.
    assert len(copied)==2
    samples=learner._structured_rollout_sidecar._samples
    saved=[s.next_critic_graph_snapshot.clone() for s in samples]
    for pending in learner._smdp_pending:
        if pending is not None and torch.is_tensor(pending.decision.structured_sample.critic_graph_snapshot):
            pending.decision.structured_sample.critic_graph_snapshot.fill_(999.)
    frame(learner,[True,True],[500.,600.],[1.,1.])
    assert all(torch.equal(s.next_critic_graph_snapshot,v) for s,v in zip(samples[:2],saved))
