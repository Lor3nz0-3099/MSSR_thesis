"""B1: bootstrap at each env's last valid epoch, never later padding."""
import pytest
import torch
from dataclasses import replace

from drl.MARL.tests.c1_smdp_fixtures import make_learner, frame, capture_update_targets


def test_async_padding_bootstraps_owner_and_mean_field_at_closed_epoch():
    learner = make_learner()
    frame(learner, [True, True], [0., 0.], [1., 1.])
    frame(learner, [True, False], [2., 10.], [1., 1.])
    frame(learner, [False, True], [100., 20.], [1., 1.])
    result = capture_update_targets(learner)
    assert result['targets'].owner_returns['a'][0, 0].item() == pytest.approx(2.8)
    assert result['owner_bootstrap']['a'][0].item() == 2.
    assert result['mf_bootstrap']['a'][0].item() == 6.
    assert result['targets'].mean_field_returns['a'][0, 0].item() == pytest.approx(7.4)
    assert result['targets'].altruistic_advantages['a'][0, 0].item() == pytest.approx(6.5)
    assert learner._smdp_pending[0].rewards.duration_steps == 2


def test_many_sleeping_padding_rows_do_not_move_bootstrap():
    learner = make_learner(capacity=3)
    frame(learner, [True, True], [0., 0.], [1., 1.])
    frame(learner, [True, False], [2., 10.], [1., 1.])
    frame(learner, [False, True], [100., 20.], [1., 1.])
    frame(learner, [False, True], [999., 30.], [1., 1.])
    result = capture_update_targets(learner)
    assert result['targets'].owner_returns['a'][0, 0].item() == pytest.approx(2.8)
    assert result['targets'].owner_advantages['a'][1:, 0].eq(0).all()


def test_env_with_no_closed_transition_has_zero_bootstrap_and_zero_actor_credit():
    learner = make_learner()
    frame(learner, [True, True], [0., 0.], [1., 1.])
    frame(learner, [True, False], [2., 100.], [1., 1.])
    frame(learner, [True, False], [3., 999.], [1., 1.])
    result = capture_update_targets(learner)
    assert result['owner_bootstrap']['a'][1].item() == 0.
    assert result['mf_bootstrap']['a'][1].item() == 0.
    assert result['targets'].altruistic_advantages['a'][:, 1].eq(0).all()


def test_bootstrap_uses_owner_mapping_from_the_same_closed_transition():
    learner = make_learner()
    frame(learner, [True, True], [0., 0.], [1., 1.])
    frame(learner, [True, False], [2., 10.], [1., 1.])
    frame(learner, [False, True], [100., 20.], [1., 1.])
    samples = learner._structured_rollout_sidecar._samples
    samples[0] = replace(samples[0], next_critic_graph_snapshot=torch.tensor([2., 20.]))
    learner._altruistic_rollout_sidecar.flatten()[0].next_owner_row_index['a'] = 1
    result = capture_update_targets(learner)
    assert result['owner_bootstrap']['a'][0].item() == 20.
    assert result['mf_bootstrap']['a'][0].item() == 60.
    assert result['targets'].owner_returns['a'][0, 0].item() == pytest.approx(19.)


@pytest.mark.parametrize('terminal, expected_owner, expected_mf', [(True, 7., 14.), (False, 9.7, 22.1)])
def test_terminal_and_truncated_epoch_use_captured_final_graph_before_reset(terminal, expected_owner, expected_mf):
    learner = make_learner(capacity=3)
    frame(learner, [True, True], [0., 0.], [1., 1.])
    frame(learner, [True, False], [2., 10.], [7., 1.],
        terminated=[terminal, False], truncated=[not terminal, False])
    frame(learner, [False, True], [999., 20.], [0., 1.])
    result = capture_update_targets(learner)
    assert result['targets'].owner_returns['a'][1, 0].item() == pytest.approx(expected_owner)
    assert result['targets'].mean_field_returns['a'][1, 0].item() == pytest.approx(expected_mf)
    assert result['owner_bootstrap']['a'][0].item() == 3.
    assert result['mf_bootstrap']['a'][0].item() == 9.
