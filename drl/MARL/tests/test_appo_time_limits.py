"""A collection limit bootstraps its own final state, never the reset state."""
import torch
import pytest
from drl.MARL.altruism import compute_gae_channel


@pytest.mark.parametrize('terminated, expected', [(False, 2.8), (True, 1.)])
def test_time_limit_and_terminal_have_different_bootstrap(terminated, expected):
    result = compute_gae_channel(rewards=torch.tensor([[[1.]], [[100.]]]),
        values=torch.zeros(2,1,1), bootstrap_value=torch.zeros(1,1),
        dones=torch.tensor([[[True]], [[True]]]),
        terminated=torch.tensor([[[terminated]], [[True]]]),
        next_values=torch.tensor([[[2.]], [[999.]]]), discount=.9, gae_lambda=.95)
    assert result.returns[0].item() == pytest.approx(expected)
    assert result.returns[1].item() == pytest.approx(100.)


def test_time_limit_requires_captured_next_value():
    with pytest.raises(ValueError, match='next_values'):
        compute_gae_channel(rewards=torch.ones(1,1,1),values=torch.zeros(1,1,1),
            bootstrap_value=torch.zeros(1,1),dones=torch.ones(1,1,1,dtype=torch.bool),
            terminated=torch.zeros(1,1,1,dtype=torch.bool),discount=.9,gae_lambda=.95)
