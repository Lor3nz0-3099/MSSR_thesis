
import torch
from types import SimpleNamespace
from drl.MARL.c1_longtrain_schedule import C1LongtrainSchedule

def test_c1_phase_boundaries():
    s = C1LongtrainSchedule()
    assert s.for_update(1).actor_lr == 0.
    assert s.for_update(4).actor_enabled is False
    assert s.for_update(5).actor_lr == 2e-6
    assert abs(
        s.for_update(12).actor_lr - 1.5e-5
    ) < 1e-14
    assert s.for_update(13).phase == "joint_refinement"

def test_c1_real_actor_freeze_and_unfreeze():
    from drl.MARL.mappo import MSSRMAPPO

    a, o, m = (
        torch.nn.Linear(2, 1) for _ in range(3)
    )

    learner = SimpleNamespace(
        actor_optimizer=torch.optim.Adam(
            a.parameters(), lr=1e-5
        ),
        owner_critic_optimizer=torch.optim.Adam(
            o.parameters(), lr=3e-4
        ),
        mean_field_critic_optimizer=torch.optim.Adam(
            m.parameters(), lr=3e-4
        ),
        _training_advantage_mode="altruistic",
    )

    schedule = C1LongtrainSchedule()

    actor_before = [
        p.detach().clone() for p in a.parameters()
    ]

    owner_before = [
        p.detach().clone() for p in o.parameters()
    ]

    x = torch.ones(3, 2)

    for update in (1, 5):
        schedule.apply(
            learner, schedule.for_update(update)
        )

        result = MSSRMAPPO.coordinated_altruistic_optimizer_step(
            owner_policy_losses={
                "owner": a(x).square().mean()
            },
            owner_value_loss=o(x).square().mean(),
            mean_field_value_loss=m(x).square().mean(),
            actor_optimizer=learner.actor_optimizer,
            owner_critic_optimizer=learner.owner_critic_optimizer,
            mean_field_critic_optimizer=learner.mean_field_critic_optimizer,
            actor_updates_enabled=learner.c1_actor_updates_enabled,
        )

        expected = 0 if update == 1 else 1

        assert result["actor_optimizer_steps"] == expected
        assert result["owner_critic_optimizer_steps"] == 1
        assert result["mean_field_critic_optimizer_steps"] == 1

        if update == 1:
            assert all(
                torch.equal(p, q)
                for p, q in zip(
                    actor_before, a.parameters()
                )
            )

            assert not learner.actor_optimizer.state

            assert any(
                not torch.equal(p, q)
                for p, q in zip(
                    owner_before, o.parameters()
                )
            )

    assert any(
        not torch.equal(p, q)
        for p, q in zip(
            actor_before, a.parameters()
        )
    )
