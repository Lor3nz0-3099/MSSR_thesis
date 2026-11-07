"""Synthetic UNIT evidence only: real SKRL vector storage and APPO optimizers.

These rows are never calibration evidence or simulator/physical evidence.
"""
from types import SimpleNamespace
import math

import pytest
import torch
from skrl.models.torch import Model

from drl.MARL.isaaclab_c1_runner import build_parser, create_native_learner, rollout_summary
from drl.MARL.mappo import StructuredRolloutSample
from drl.MARL.altruistic_rollout import build_altruistic_rollout_batch


class ScalarModel(Model):
    def __init__(self, value=0.):
        super().__init__(device="cpu")
        self.weight = torch.nn.Parameter(torch.tensor(float(value)))

    def compute(self, inputs, role):
        return self.weight.reshape(1, 1), {}

    def act(self, inputs, role=""):
        return self.compute(inputs, role)

    def forward_owner(self, graph, *, owner_index):
        return self.weight.reshape(1, 1)


def make_learner(*, rollout_steps=4):
    args = SimpleNamespace(num_envs=2, rollout_steps=rollout_steps, device="cpu",
        learning_epochs=2, mini_batches=2, gamma=.99, alpha=.5, altruism_radius_m=.5)
    learner = create_native_learner(args, None, ScalarModel(), ScalarModel(7.), ("m0", "m1"),
        provider=SimpleNamespace(current_altruistic_rollout_batch=lambda: None),
        critic_pair=SimpleNamespace(owner=ScalarModel(), mean_field=ScalarModel()))
    return learner


def populate(learner, *, steps):
    for timestep in range(steps):
        for memory in learner.memories.values():
            zeros = torch.zeros(2, 1)
            memory.add_samples(observations=zeros, states=zeros, actions=zeros, rewards=zeros,
                log_prob=zeros, values=zeros, terminated=zeros.bool(), truncated=zeros.bool())
        learner._structured_rollout_sidecar.append_step([
            StructuredRolloutSample(action=None, hard_mask=(), owner_active_mask={"m0": True, "m1": True},
                routing_snapshot=None, graph_snapshot={"env": env}, critic_graph_snapshot={"env": env})
            for env in range(2)])
        learner._altruistic_rollout_sidecar.append_step(build_altruistic_rollout_batch(
            possible_agents=learner.possible_agents, module_ids_by_row=(("p0", "p1"),) * 2,
            owner_to_module_id=({"m0": "p0", "m1": "p1"},) * 2,
            module_positions_by_id=tuple({"p0": torch.tensor([10.*env, 0., 0.]),
                "p1": torch.tensor([10.*env + .2, 0., 0.])} for env in range(2)),
            owner_rewards={"m0": torch.tensor([[1.+timestep], [2.+timestep]]),
                           "m1": torch.tensor([[3.+timestep], [4.+timestep]])}, radius_m=.5,
            next_module_ids_by_row=(("p0", "p1"),) * 2,
            next_owner_to_module_id=({"m0": "p0", "m1": "p1"},) * 2))
    learner._current_next_critic_graph = [{"env": env} for env in range(2)]


def test_native_cli_defaults_are_bounded_and_optimizer_is_explicit():
    args = build_parser().parse_args(["--reward-config", "/unused/unit.json"])
    assert not args.train and not args.controlled_update
    assert args.num_envs == 2 and args.rollout_steps == 4
    assert args.learning_epochs == args.mini_batches == 2
    assert args.domains == ["navigation", "stairs", "gap", "manipulation"]


def test_native_partial_vector_memory_retains_time_environment_axes_and_sidecars():
    learner = make_learner(rollout_steps=256)
    assert learner.deterministic_update_mode is True
    assert learner.skip_expert_policy_inference is True
    populate(learner, steps=3)
    summary = rollout_summary(learner, num_envs=2, vector_steps=3)
    assert summary["transitions"] == summary["structured_rows"] == summary["altruistic_rows"] == 6
    for memory in learner.memories.values():
        assert memory.memory_size == 256 and memory.num_envs == 2
        assert memory.get_tensor_by_name("log_prob").shape == (3, 2, 1)
    assert [row.graph_snapshot["env"] for row in learner._structured_rollout_sidecar.sample_by_index(range(6))] == [0, 1, 0, 1, 0, 1]


def test_native_two_env_two_epoch_two_minibatch_update_measures_natural_policy_change():
    torch.manual_seed(7)
    learner = make_learner()
    populate(learner, steps=4)
    # Larger UNIT-only step makes diagnostic drift visible in float32.
    for group in learner.actor_optimizer.param_groups:
        group["lr"] = .05
    # Unit scalar models isolate vector storage, APPO targets and optimizer math
    # from the physical scene. No patched losses, KL, ratio or gradient values.
    def reevaluate(*, structured_batch, owner_tensor_batch, evaluate_global_critic):
        batch_size = len(structured_batch)
        return ({uid: learner.shared_actor.weight.reshape(1, 1).expand(batch_size, 1)
                 for uid in learner.possible_agents},
                {uid: torch.full((batch_size, 1), .7) for uid in learner.possible_agents}, None)
    learner.reevaluate_structured_minibatch = reevaluate
    before = [model.weight.detach().clone() for model in (
        learner.shared_actor, learner.altruistic_owner_critic, learner.altruistic_mean_field_critic)]
    learner.coordinated_update(timestep=4, timesteps=4)
    metrics = learner.last_update_metrics
    assert metrics["actor_optimizer_steps"] == metrics["owner_critic_optimizer_steps"] == metrics["mean_field_critic_optimizer_steps"] == 4
    assert metrics["global_critic_optimizer_steps"] == 0
    assert metrics["approx_kl"] > 0 and metrics["ratio_mean"] != 1.
    assert all(math.isfinite(value) for value in metrics.values())
    for value, model in zip(before, (learner.shared_actor,
            learner.altruistic_owner_critic, learner.altruistic_mean_field_critic), strict=True):
        assert not torch.equal(value, model.weight)
    assert learner.shared_critic.weight.item() == 7.
    rollout_summary(learner, num_envs=2, vector_steps=4)


def test_native_learner_accepts_skrl_initialization_without_losing_vector_storage(tmp_path):
    learner = make_learner()
    learner.cfg.experiment.directory = str(tmp_path)
    learner.cfg.experiment.write_interval = 0
    learner.cfg.experiment.checkpoint_interval = 0
    from skrl.trainers.torch.sequential import SequentialTrainerCfg
    learner.init(trainer_cfg=SequentialTrainerCfg(timesteps=4))
    populate(learner, steps=4)
    assert rollout_summary(learner, num_envs=2, vector_steps=4)["transitions"] == 8


@pytest.mark.parametrize("assembled_observations", [False, True])
def test_bounded_collector_uses_existing_act_and_record_hooks_for_two_envs(assembled_observations):
    from drl.MARL.isaaclab_c1_runner import collect_native_rollout
    from drl.MARL.mappo import OnlineRolloutContext, OnlineRolloutDecision
    from drl.MARL.altruistic_rollout import AltruisticRolloutBatchPayload

    learner = make_learner()
    owners = learner.possible_agents
    carrier = {owner: torch.zeros(2, 1) for owner in owners}
    clocks = [0, 0]

    def graph(env):
        return {"env": env, "stamp": clocks[env]}

    provider = SimpleNamespace(
        current_action_contexts=lambda: [OnlineRolloutContext(graph(env), graph(env), None, ()) for env in range(2)],
        current_critic_graphs=lambda: [graph(env) for env in range(2)],
        encode_env_actions=lambda decisions: {owner: torch.zeros(2, 1) for owner in owners},
        current_altruistic_rollout_batch=lambda: AltruisticRolloutBatchPayload(
            module_ids_by_row=(("p0", "p1"),) * 2,
            owner_to_module_id=({"m0": "p0", "m1": "p1"},) * 2,
            module_positions_by_id=tuple({"p0": torch.tensor([10.*env, 0., 0.]),
                "p1": torch.tensor([10.*env + .2, 0., 0.])} for env in range(2)),
            owner_rewards={owner: torch.tensor([[1.], [2.]]) for owner in owners},
            next_module_ids_by_row=(("p0", "p1"),) * 2,
            next_owner_to_module_id=({"m0": "p0", "m1": "p1"},) * 2))
    learner.set_rollout_context_provider(provider)
    learner.set_altruistic_rollout_provider(provider, radius_m=.5)

    # Unit-only semantic sample; act(), record_transition(), storage and
    # provider timing are the real learner implementation.
    def decision(*, actor_graph, critic_graph, routing_snapshot, hard_mask, actor_output):
        assert actor_output == {}
        return OnlineRolloutDecision(action=None,
            owner_log_probs={owner: torch.zeros(1) for owner in owners}, global_value=torch.zeros(1, 1),
            structured_sample=StructuredRolloutSample(action=None, hard_mask=(),
                owner_active_mask={owner: True for owner in owners}, routing_snapshot=None,
                graph_snapshot=actor_graph, critic_graph_snapshot=critic_graph))
    learner.build_online_rollout_decision = decision
    # This collector test verifies vector storage and timing, not actor
    # inference. The semantic decision above is deliberately synthetic.
    # Let the new batch boundary use the same inert unit-only actor.
    learner.shared_actor.forward_all_owners = lambda **kwargs: {}

    resets = []

    class Wrapper:
        num_envs = 2

        def reset(self):
            resets.append(True)
            return carrier, {}

        def state(self):
            return carrier

        def step(self, actions):
            clocks[:] = [value + 1 for value in clocks]
            return carrier, carrier, {owner: value.bool() for owner, value in carrier.items()}, {owner: value.bool() for owner, value in carrier.items()}, {}

    initial = dict(initial_observations=carrier, initial_states=carrier) if assembled_observations else {}
    callback_steps = []
    def before_step(timestep, observations, states):
        callback_steps.append((timestep, clocks[0]))
        return observations, states
    assert collect_native_rollout(Wrapper(), learner, rollout_steps=4, before_step=before_step, **initial)["transitions"] == 8
    assert len(resets) == (0 if assembled_observations else 1)
    assert callback_steps == [(0, 0), (1, 1), (2, 2), (3, 3)]
    samples = learner._structured_rollout_sidecar.sample_by_index(range(8))
    assert [sample.graph_snapshot["stamp"] for sample in samples] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [sample.next_critic_graph_snapshot["stamp"] for sample in samples] == [1, 1, 2, 2, 3, 3, 4, 4]


@pytest.mark.parametrize("disturb_other", [False, True])
def test_partial_reset_probe_checks_measured_pose_isolation(disturb_other):
    """UNIT backend only: prove the checker rejects unrelated physical changes."""
    import numpy as np
    from drl.MARL.isaaclab_c1_runner import make_partial_reset_probe
    from drl.MARL.runtime_handoff import DirectMARLHandoff

    poses = [np.zeros((1, 7)), np.ones((1, 7))]
    resets = []

    class Array:
        def __init__(self, values):
            self.values = values

        def numpy(self):
            return self.values

    def reader(i):
        view = SimpleNamespace(get_root_transforms=lambda: Array(poses[i]),
                               get_dof_positions=lambda: Array(np.zeros((1, 3))))
        return SimpleNamespace(_articulation=SimpleNamespace(view=view))

    handoffs = [DirectMARLHandoff(), DirectMARLHandoff()]
    env = SimpleNamespace(num_envs=2, generation=[1, 1], handoffs=handoffs,
        state_readers_by_env=[{"m0": reader(0)}, {"m0": reader(1)}],
        docking_managers=[SimpleNamespace(connections=[]), SimpleNamespace(connections=[])],
        runtime_context_provider=SimpleNamespace(reset=lambda: resets.append("provider")),
        _get_observations=lambda: "fresh observations")

    def reset(ids):
        assert ids == [0]
        env.generation[0] += 1
        env.handoffs[0].begin_expert("rc_car8")
        if disturb_other:
            poses[1][0, 0] += .1
    env._reset_idx = reset
    truncations = torch.zeros(2, 2, 1, dtype=torch.bool)
    learner = SimpleNamespace(memories={"m0": SimpleNamespace(get_tensor_by_name=lambda name: truncations)})
    probe, proof = make_partial_reset_probe(env, SimpleNamespace(state=lambda: "fresh states"), learner, reset_timestep=2)
    assert probe(1, None, None) is None and not proof
    if disturb_other:
        with pytest.raises(RuntimeError, match="unrelated env 1 physical pose"):
            probe(2, None, None)
    else:
        assert probe(2, None, None) == ("fresh observations", "fresh states")
        assert proof["selected_phase"] == "ASSEMBLING"
        assert proof["unaffected_physical_poses_unchanged"]
        assert proof["generation_after"] == proof["generation_before"] + 1
        assert resets == ["provider"]
        assert proof["collector_truncated_previous_transition"]
        assert truncations[:, :, 0].tolist() == [[False, False], [True, False]]


def test_training_cli_requires_explicit_active_transition_budget_and_train_sampler():
    from drl.MARL.isaaclab_c1_runner import configure_run_arguments
    args = build_parser().parse_args(['--train','--reward-config','unit.json','--output','unit-out'])
    with pytest.raises(ValueError, match='total-transitions'):
        configure_run_arguments(args)
    args.total_transitions = 512
    args.num_envs = 4
    configure_run_arguments(args)
    assert args.train_sampler == 'catalog'
    assert args.total_transitions == 512
    args.train_sampler = 'fixed'
    with pytest.raises(ValueError, match='resampling'):
        configure_run_arguments(args)


def test_scaling_cli_uses_same_aggregate_experience_and_explicit_vector_units():
    from drl.MARL.isaaclab_c1_runner import configure_run_arguments
    for count in (4,8,16):
        args = build_parser().parse_args(['--benchmark','--reward-config','unit.json',
            '--output','unit-out','--num-envs',str(count),'--total-transitions','128'])
        configure_run_arguments(args)
        assert args.rollout_steps * count == 128
        assert args.max_vector_steps == args.rollout_steps
        assert args.timesteps is None


def test_campaign_wall_and_validation_stopping_are_explicit_and_bounded():
    from drl.MARL.isaaclab_c1_runner import configure_run_arguments
    args=build_parser().parse_args(['--train','--num-envs','4','--total-transitions','1024',
        '--output','unit-out','--reward-config','unit.json','--max-wall-seconds','86400',
        '--early-stopping-patience','3','--validation-every-seconds','14400'])
    configure_run_arguments(args)
    assert args.max_wall_seconds == 86400 and args.early_stopping_patience == 3
    args.max_wall_seconds = 86401
    with pytest.raises(ValueError,match='24'):
        configure_run_arguments(args)


def test_non_train_modes_cannot_enable_validation_model_selection():
    from drl.MARL.isaaclab_c1_runner import configure_run_arguments
    args=build_parser().parse_args(['--benchmark','--num-envs','4','--total-transitions','128',
        '--output','unit-out','--reward-config','unit.json','--early-stopping-patience','3'])
    with pytest.raises(ValueError,match='only.*train'):
        configure_run_arguments(args)


def test_global_seed_reaches_skrl_wrapper_reset_key():
    from skrl import config
    from drl.MARL.isaaclab_c1_runner import seed_native_training
    config.torch.key=0
    seed_native_training(137)
    assert config.torch.key == 137
    first=torch.rand(4)
    seed_native_training(137)
    assert torch.equal(first,torch.rand(4))


def test_real_campaign_cannot_reuse_restricted_smoke_seed_manifest():
    from drl.MARL.isaaclab_c1_runner import configure_run_arguments
    args=build_parser().parse_args(['--train','--num-envs','4','--total-transitions','1024',
        '--output','unit-out','--reward-config','unit.json','--seed-manifest','smoke.json'])
    with pytest.raises(ValueError,match='full TRAIN'):
        configure_run_arguments(args)


def test_real_campaign_defaults_to_one_day_without_optional_flags():
    from drl.MARL.isaaclab_c1_runner import configure_run_arguments
    args=build_parser().parse_args(['--train','--num-envs','4','--total-transitions','1024',
        '--output','unit-out','--reward-config','unit.json'])
    configure_run_arguments(args)
    assert args.max_wall_seconds == 86400


@pytest.mark.parametrize('domains', [
    ['navigation'], ['navigation', 'stairs', 'gap'],
    ['navigation', 'stairs', 'gap', 'gap'],
    ['navigation', 'stairs', 'gap', 'manipulation', 'navigation'],
])
def test_real_campaign_requires_exactly_four_unique_c1_domains(domains):
    from drl.MARL.isaaclab_c1_runner import configure_run_arguments
    args = build_parser().parse_args(['--train', '--num-envs', '8',
        '--total-transitions', '1024', '--output', 'unit-out', '--reward-config', 'unit.json',
        '--domains', *domains])
    with pytest.raises(ValueError, match='four.*unique|unique.*four'):
        configure_run_arguments(args)


def test_real_campaign_defaults_validation_patience_and_accepts_reordered_domains():
    from drl.MARL.isaaclab_c1_runner import configure_run_arguments
    args = build_parser().parse_args(['--train', '--num-envs', '4',
        '--total-transitions', '1024', '--output', 'unit-out', '--reward-config', 'unit.json',
        '--domains', 'manipulation', 'gap', 'stairs', 'navigation'])
    assert args.early_stopping_patience is None
    configure_run_arguments(args)
    assert args.early_stopping_patience == 3
    assert args.domains == ['manipulation', 'gap', 'stairs', 'navigation']


@pytest.mark.parametrize('patience', [0, -1])
def test_real_campaign_rejects_explicit_nonpositive_validation_patience(patience):
    from drl.MARL.isaaclab_c1_runner import configure_run_arguments
    args = build_parser().parse_args(['--train', '--num-envs', '4',
        '--total-transitions', '1024', '--output', 'unit-out', '--reward-config', 'unit.json',
        '--early-stopping-patience', str(patience)])
    with pytest.raises(ValueError, match='patience'):
        configure_run_arguments(args)


@pytest.mark.parametrize('mode', ['check', 'smoke-train', 'benchmark', 'evaluate'])
def test_non_campaign_default_patience_remains_zero_and_domains_can_be_restricted(mode):
    from drl.MARL.isaaclab_c1_runner import configure_run_arguments
    argv = ['--reward-config', 'unit.json', '--domains', 'navigation', '--num-envs', '4']
    if mode in ('smoke-train', 'benchmark'):
        argv += ['--' + mode, '--total-transitions', '128', '--output', 'unit-out']
    elif mode == 'evaluate':
        argv += ['--evaluate', '--seeds', '8000', '8001', '8002', '8003',
                 '--max-vector-steps', '10', '--output', 'unit-out']
    args = build_parser().parse_args(argv)
    configure_run_arguments(args)
    assert args.early_stopping_patience == 0
    assert args.domains == ['navigation']
