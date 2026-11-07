from types import SimpleNamespace

from drl.MARL import mappo


def test_smdp_policy_version_advances_after_real_actor_update(monkeypatch):
    """A completed actor optimizer update must advance the SMDP behavior version."""

    learner = mappo.MSSRMAPPO.__new__(mappo.MSSRMAPPO)

    learner.training = True
    learner._smdp_options_enabled = True
    learner._smdp_event_rows_last_interaction = 1
    learner._rollout = 0
    learner._smdp_policy_version = 0
    learner._smdp_max_policy_lag = 0

    learner.cfg = SimpleNamespace(
        rollouts=1,
        learning_starts=0,
    )

    learner.deterministic_update_mode = False
    learner._structured_rollout_sidecar = []
    learner._altruistic_rollout_sidecar = []
    learner._current_next_critic_graph = None
    learner._last_update_metrics = {}

    learner.enable_models_training_mode = lambda *_args, **_kwargs: None
    learner.track_data = lambda *_args, **_kwargs: None

    def coordinated_update(*, timestep, timesteps):
        learner._last_update_metrics = {
            "actor_optimizer_steps": 1,
            "owner_critic_optimizer_steps": 1,
            "mean_field_critic_optimizer_steps": 1,
        }

    learner.coordinated_update = coordinated_update

    monkeypatch.setattr(
        mappo.MultiAgent,
        "post_interaction",
        lambda self, **kwargs: None,
    )

    learner.post_interaction(
        timestep=1,
        timesteps=10,
    )

    assert learner.smdp_policy_version == 1
