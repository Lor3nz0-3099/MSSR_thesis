import torch


def test_discounted_option_accumulator_keeps_one_reward_for_many_physical_steps():
    from drl.MARL.c1_smdp import DiscountedOptionAccumulator

    acc = DiscountedOptionAccumulator(discount=0.9)
    acc.add(
        team_reward=torch.tensor(1.0),
        owner_rewards={"a": torch.tensor(1.0), "b": torch.tensor(2.0)},
        mean_field_rewards={"a": torch.tensor(3.0), "b": torch.tensor(4.0)},
    )
    acc.add(
        team_reward=torch.tensor(2.0),
        owner_rewards={"a": torch.tensor(2.0), "b": torch.tensor(3.0)},
        mean_field_rewards={"a": torch.tensor(4.0), "b": torch.tensor(5.0)},
    )
    acc.add(
        team_reward=torch.tensor(3.0),
        owner_rewards={"a": torch.tensor(3.0), "b": torch.tensor(4.0)},
        mean_field_rewards={"a": torch.tensor(5.0), "b": torch.tensor(6.0)},
    )

    assert acc.duration_steps == 3
    assert torch.allclose(acc.team_reward, torch.tensor(1 + 0.9 * 2 + 0.9**2 * 3))
    assert torch.allclose(acc.owner_rewards["a"], torch.tensor(1 + 0.9 * 2 + 0.9**2 * 3))
    assert torch.allclose(acc.mean_field_rewards["b"], torch.tensor(4 + 0.9 * 5 + 0.9**2 * 6))


def test_smdp_gae_uses_gamma_power_k_and_skips_padding_rows():
    from drl.MARL.altruism import compute_gae_channel

    rewards = torch.tensor([[[1.0]], [[0.0]], [[2.0]]])
    values = torch.tensor([[[0.5]], [[99.0]], [[1.0]]])
    bootstrap = torch.tensor([[3.0]])
    dones = torch.zeros_like(rewards, dtype=torch.bool)
    durations = torch.tensor([[[2]], [[0]], [[3]]], dtype=torch.long)

    result = compute_gae_channel(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap,
        dones=dones,
        discount=0.9,
        gae_lambda=0.95,
        duration_steps=durations,
    )

    delta_last = 2.0 + 0.9**3 * 3.0 - 1.0
    delta_first = 1.0 + 0.9**2 * 1.0 - 0.5
    expected_first = delta_first + (0.9**2) * 0.95 * delta_last

    assert torch.allclose(result.advantages[2], torch.tensor([[delta_last]]), atol=1e-6)
    assert torch.equal(result.advantages[1], torch.zeros(1, 1))
    assert torch.allclose(result.advantages[0], torch.tensor([[expected_first]]), atol=1e-6)
    assert torch.equal(result.returns[1], values[1])


def test_smdp_gae_rejects_negative_duration():
    from drl.MARL.altruism import compute_gae_channel

    rewards = torch.zeros(1, 1, 1)
    values = torch.zeros_like(rewards)
    dones = torch.zeros_like(rewards, dtype=torch.bool)

    try:
        compute_gae_channel(
            rewards=rewards,
            values=values,
            bootstrap_value=torch.zeros(1, 1),
            dones=dones,
            discount=0.99,
            gae_lambda=0.95,
            duration_steps=torch.tensor([[[-1]]]),
        )
    except ValueError as exc:
        assert "duration_steps" in str(exc)
    else:
        raise AssertionError("negative duration must be rejected")



def test_smdp_actor_policy_lag_accepts_current_and_one_update_old_only():
    from drl.MARL.c1_smdp import actor_policy_lag_is_acceptable

    assert actor_policy_lag_is_acceptable(
        start_version=3,
        current_version=3,
    )
    assert actor_policy_lag_is_acceptable(
        start_version=2,
        current_version=3,
    )
    assert not actor_policy_lag_is_acceptable(
        start_version=1,
        current_version=3,
    )


def test_smdp_actor_policy_lag_rejects_future_or_negative_versions():
    from drl.MARL.c1_smdp import actor_policy_lag_is_acceptable

    for start_version, current_version in (
        (-1, 0),
        (2, 1),
    ):
        try:
            actor_policy_lag_is_acceptable(
                start_version=start_version,
                current_version=current_version,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "invalid policy versions must be rejected"
            )
