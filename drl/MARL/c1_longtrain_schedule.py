
from dataclasses import dataclass

@dataclass(frozen=True)
class C1Step:
    update: int
    phase: str
    phase_id: int
    actor_lr: float
    critic_lr: float
    actor_enabled: bool

class C1LongtrainSchedule:

    def for_update(self, n):
        if type(n) is not int or n < 1:
            raise ValueError("PPO update non valido")

        if n <= 4:
            return C1Step(
                n, "critic_warmup", 1, 0., 3e-4, False
            )

        if n <= 12:
            t = (n - 5) / 7
            return C1Step(
                n, "actor_ramp", 2,
                2e-6 + t * 1.3e-5,
                3e-4 - t * 1e-4,
                True
            )

        return C1Step(
            n, "joint_refinement", 3,
            1.5e-5, 2e-4, True
        )

    def apply(self, learner, step):
        if getattr(
            learner, "_training_advantage_mode", None
        ) != "altruistic":
            raise RuntimeError(
                "Scheduler richiede APPO altruistico"
            )

        optimizers = (
            (learner.actor_optimizer, step.actor_lr),
            (learner.owner_critic_optimizer, step.critic_lr),
            (
                learner.mean_field_critic_optimizer,
                step.critic_lr
            ),
        )

        for optimizer, lr in optimizers:
            for group in optimizer.param_groups:
                group["lr"] = lr

        learner.c1_actor_updates_enabled = step.actor_enabled

        return step
