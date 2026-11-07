"""Native C1 pre-training boundary; default is a bounded check, never a campaign.

Launch with ``python -m drl.MARL.isaaclab_c1_runner`` in the Isaac Lab Python
runtime. Isaac imports deliberately follow AppLauncher construction.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def seed_native_training(seed):
    """Seed the framework reset key as well as Python, NumPy and Torch RNGs."""
    from skrl.utils import set_seed
    return set_seed(seed)


def create_native_learner(args, bridge, actor, critic, owners, provider=None, *, critic_pair=None):
    """Configure the existing learner with T vector slots (T*N transitions)."""
    import torch
    from drl.MARL.c1_runner import make_rollout_memory
    from drl.MARL.mappo import MSSRMAPPO, SMDPEventMemory
    from drl.MARL.owner_critic import build_repository_altruistic_critic_pair

    if args.num_envs < 2 or args.rollout_steps < 1:
        raise ValueError("native pre-training requires num_envs >= 2 and positive rollout_steps")
    owners = tuple(owners)
    smdp_enabled = (provider is not None
        and getattr(args, 'structural_scheduler', 'fixed') == 'event'
        and (bool(getattr(args, 'smoke_train', False)) or bool(getattr(args, 'train', False))))
    memory_factory = SMDPEventMemory if smdp_enabled else make_rollout_memory
    memories = {owner: memory_factory(memory_size=args.rollout_steps,
                num_envs=args.num_envs, device=args.device, replacement=False) for owner in owners}
    for memory in memories.values():
        for name in ("observations", "states", "actions", "rewards", "log_prob", "values"):
            memory.create_tensor(name=name, size=1, dtype=torch.float32)
        for name in ("terminated", "truncated"):
            memory.create_tensor(name=name, size=1, dtype=torch.bool)
    cfg = dict(learning_epochs=args.learning_epochs, mini_batches=args.mini_batches,
        rollouts=args.rollout_steps, learning_rate=(1e-5, 3e-4),
        discount_factor=args.gamma, gae_lambda=.95, entropy_loss_scale=.001,
        learning_rate_scheduler=None,
        **{name: {owner: {} for owner in owners} for name in (
            "learning_rate_scheduler_kwargs", "observation_preprocessor_kwargs",
            "state_preprocessor_kwargs", "value_preprocessor_kwargs")})
    cfg["experiment"] = dict(write_interval=0, checkpoint_interval=0)
    if getattr(args, 'train', False) and getattr(args, 'output', None):
        cfg['experiment'].update(directory=str(args.output / 'skrl'), write_interval=100)
    learner = MSSRMAPPO(possible_agents=owners,
        models=MSSRMAPPO.build_shared_model_registry(possible_agents=owners, actor=actor, critic=critic),
        memories=memories, observation_spaces={owner: 1 for owner in owners},
        state_spaces={owner: 1 for owner in owners}, action_spaces={owner: 1 for owner in owners},
        device=args.device, cfg=cfg, deterministic_update_mode=True,
        skip_expert_policy_inference=True)
    if critic_pair is None:
        critic_pair = build_repository_altruistic_critic_pair(spec=bridge.spec,
            feature_schema=bridge.feature_schema, repo_root=args.repository_root, device=args.device)
    learner.configure_fixed_alpha_appo(owner_critic=critic_pair.owner,
        mean_field_critic=critic_pair.mean_field, alpha=args.alpha)
    for model in (learner.shared_actor, learner.shared_critic,
                  learner.altruistic_owner_critic, learner.altruistic_mean_field_critic):
        model.eval()
    if provider is not None:
        learner.set_rollout_context_provider(provider)
        learner.set_altruistic_rollout_provider(provider, radius_m=args.altruism_radius_m)
        learner.configure_smdp_options(enabled=smdp_enabled)
    return learner


def rollout_summary(learner, *, num_envs, vector_steps):
    """Fail closed on tensor/structured/owner-reward storage misalignment."""
    expected = num_envs * vector_steps
    lengths = {owner: len(memory) for owner, memory in learner.memories.items()}
    structured = len(learner._structured_rollout_sidecar)
    altruistic = len(learner._altruistic_rollout_sidecar.flatten())
    if set(lengths.values()) != {expected} or structured != expected or altruistic != expected:
        raise RuntimeError(f"native rollout storage misalignment: {lengths}, {structured}, {altruistic}")
    for memory in learner.memories.values():
        if memory.get_tensor_by_name("terminated").shape[:2] != (vector_steps, num_envs):
            raise RuntimeError("native memory does not preserve (time, environment) axes")
    samples = learner._structured_rollout_sidecar.sample_by_index(range(expected))
    phases = [[bool(samples[step * num_envs + env].rollout_active)
               for env in range(num_envs)] for step in range(vector_steps)]
    return dict(vector_steps=vector_steps, num_envs=num_envs, transitions=expected,
                structured_rows=structured, altruistic_rows=altruistic,
                rollout_active_by_step=phases, active_learning_rows=sum(map(sum, phases)))


def collect_native_rollout(wrapper, learner, *, rollout_steps,
                           initial_observations=None, initial_states=None, before_step=None):
    """Use the standard SKRL interaction hooks without running an optimizer."""
    import torch
    if initial_observations is None:
        observations, _ = wrapper.reset()
    else:
        observations = initial_observations
    states = wrapper.state() if initial_states is None else initial_states
    for timestep in range(rollout_steps):
        if before_step is not None:
            refreshed = before_step(timestep, observations, states)
            if refreshed is not None:
                observations, states = refreshed
        with torch.no_grad():
            actions, _ = learner.act(observations, states, timestep=timestep, timesteps=rollout_steps)
            next_observations, rewards, terminated, truncated, infos = wrapper.step(actions)
            next_states = wrapper.state()
            learner.record_transition(observations=observations, states=states, actions=actions,
                rewards=rewards, next_observations=next_observations, next_states=next_states,
                terminated=terminated, truncated=truncated, infos=infos,
                timestep=timestep, timesteps=rollout_steps)
        # Native provider preserves terminal next graphs and marks Expert rows
        # inactive; the existing learner masks their optimizer losses.
        observations, states = next_observations, next_states
    return rollout_summary(learner, num_envs=wrapper.num_envs, vector_steps=rollout_steps)


def make_partial_reset_probe(env, wrapper, learner, *, reset_timestep):
    """Reset only env0 and prove isolation against actual PhysX tensor views."""
    import numpy as np
    proof = {}

    def measured(env_index):
        return {owner: (reader._articulation.view.get_root_transforms().numpy().copy(),
                        reader._articulation.view.get_dof_positions().numpy().copy())
                for owner, reader in env.state_readers_by_env[env_index].items()}

    def topology(env_index):
        return tuple(sorted(tuple(sorted(((c.first_face.module_id, c.first_face.face_name),
                                          (c.second_face.module_id, c.second_face.face_name))))
                            for c in env.docking_managers[env_index].connections))

    def before_step(timestep, observations, states):
        if timestep != reset_timestep:
            return None
        if proof:
            raise RuntimeError("partial reset probe invoked more than once")
        before = {i: measured(i) for i in range(1, env.num_envs)}
        edges = {i: topology(i) for i in range(1, env.num_envs)}
        generation = env.generation[0]
        # This explicit collector reset truncates the preceding real physical
        # transition. Its already stored next graph is the correct bootstrap.
        for memory in learner.memories.values():
            previous = memory.get_tensor_by_name("truncated")
            if previous.shape[0] != timestep:
                raise RuntimeError("partial reset probe memory is not aligned with vector time")
            previous[-1, 0] = True
        env._reset_idx([0])
        if env.generation[0] != generation + 1:
            raise RuntimeError("partial reset did not advance selected generation")
        if env.docking_managers[0].connections or env.handoffs[0].marl_active:
            raise RuntimeError("partial reset retained docking/MARL authority")
        if env.handoffs[0].phase.value != "ASSEMBLING":
            raise RuntimeError("partial reset did not restore Expert assembly authority")
        for i, row in before.items():
            if topology(i) != edges[i]:
                raise RuntimeError(f"partial reset changed unrelated env {i} topology")
            after = measured(i)
            for owner, values in row.items():
                if any(not np.array_equal(old, new) for old, new in zip(values, after[owner], strict=True)):
                    raise RuntimeError(f"partial reset changed unrelated env {i} physical pose: {owner}")
        env.runtime_context_provider.reset()
        proof.update(timestep=timestep, reset_env=0, generation_before=generation,
            generation_after=env.generation[0], selected_connections=0,
            selected_phase=env.handoffs[0].phase.value,
            collector_truncated_previous_transition=True,
            unaffected_envs=list(before),
            unaffected_connection_counts={str(i): len(edges[i]) for i in before},
            unaffected_physical_poses_unchanged=True,
            unaffected_topology_unchanged=True)
        return env._get_observations(), wrapper.state()

    return before_step, proof


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="default: bounded rollout, no optimizer")
    mode.add_argument("--train", action="store_true", help="explicitly start SKRL training")
    parser.add_argument("--c1-staged-training", action="store_true")
    parser.add_argument("--c1-integration-probe", action="store_true",
                        help="bounded staged C1 integration test only")
    parser.add_argument("--resume-checkpoint", type=Path, default=None,
                        help="resume C1 staged training from a verified checkpoint; new output directory required")
    mode.add_argument('--benchmark', action='store_true', help='one bounded native SKRL scaling rollout/update')
    mode.add_argument('--smoke-train', action='store_true', help='bounded multi-seed TRAIN learning smoke')
    mode.add_argument('--evaluate', action='store_true', help='frozen BC/APPO on an explicit fixed split bank; no updates')
    parser.add_argument("--check-partial-reset", action="store_true",
                        help="bounded check: reset env0 halfway through and prove isolation")
    parser.add_argument("--check-reassembly", action="store_true",
                        help="after partial-reset check, finish physical Expert reassembly")
    parser.add_argument("--controlled-update", action="store_true", help="one explicit bounded optimizer test")
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--reward-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--domains", nargs="+",
                        choices=("navigation", "stairs", "gap", "manipulation"),
                        default=["navigation", "stairs", "gap", "manipulation"])
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="optional frozen TRAIN curriculum seed per environment")
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--structural-scheduler", choices=("event", "fixed"), default="event",
                        help="C1 structural policy clock; event mirrors the accelerated GUI runtime")
    parser.add_argument("--learning-epochs", type=int, default=2)
    parser.add_argument("--mini-batches", type=int, default=2)
    parser.add_argument('--total-transitions', type=int, help='aggregate active MARL environment transitions; stop at update boundary')
    parser.add_argument('--vector-decision-steps', '--timesteps', dest='timesteps', type=int,
                        help='bounded-check vector iterations including Expert rows; training uses --total-transitions')
    parser.add_argument('--max-vector-steps', type=int, help='finite upper bound on SKRL vector iterations')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--global-seed', type=int, default=20261029)
    parser.add_argument('--train-sampler', choices=('catalog','smoke-cycle','fixed'))
    parser.add_argument('--initialization-mode', choices=('loose_modules','assembled_state','expert_validation'))
    parser.add_argument('--snapshot-bank', type=Path, default=None,
                        help='versioned C1 Expert snapshot manifest for assembled_state')
    parser.add_argument('--snapshot-stage-weights', type=str, default=None,
                        help='JSON object with S0/S1/S2/S3 curriculum probabilities')
    parser.add_argument('--curriculum', choices=('catalog','expert_validation','c1'), default='catalog',
                        help='reuse existing TRAIN difficulty allocation: 250 easy / 400 medium / 350 hard')
    parser.add_argument('--seed-manifest', type=Path, help='optional TRAIN-only pools from a saved dispersed smoke manifest')
    parser.add_argument('--episode-seconds', type=float, default=60., help='MARL horizon, excludes Expert assembly')
    parser.add_argument(
        '--c1-probe-config', type=Path, default=None,
        help='C1 assembly/probe reward settings'
    )
    parser.add_argument('--checkpoint-every-transitions', type=int, default=0,
                        help='checkpoint cadence in active environment transitions; zero disables periodic saves')
    parser.add_argument('--min-updates', type=int, default=2)
    parser.add_argument('--min-completed-seeds-per-domain', type=int, default=4)
    parser.add_argument('--evaluation-split', choices=('train','validation','test','challenge'), default='validation')
    parser.add_argument('--policy-checkpoint', type=Path, help='frozen learned checkpoint for --evaluate; default is frozen BC-v4')
    parser.add_argument('--max-wall-seconds', type=float, help='entire campaign wall limit, at most 86400 seconds')
    parser.add_argument('--early-stopping-patience', type=int, default=None,
                        help='validation events without improvement; defaults to 3 for --train, 0 otherwise; --train requires a positive value')
    parser.add_argument('--early-stopping-min-delta', type=float, default=0.)
    parser.add_argument('--validation-every-seconds', type=float, default=14400.)
    parser.add_argument('--validation-vector-steps', type=int, default=300)
    parser.add_argument('--validation-timeout-seconds', type=float, default=1800.)
    parser.add_argument('--validation-memory-bytes', type=int, default=3*1024**3)
    parser.add_argument('--checkpoint-reserve-seconds', type=float, default=60.)
    parser.add_argument('--update-reserve-seconds', type=float, default=600.)
    parser.add_argument("--assembly-max-steps", type=int, default=18000)
    parser.add_argument("--alpha", type=float, default=.5)
    parser.add_argument("--gamma", type=float, default=.99)
    parser.add_argument("--altruism-radius-m", type=float, default=.5)
    return parser


def configure_run_arguments(args):
    """Disambiguate experience units before constructing a simulator."""
    from drl.MARL.c1_initialization import resolve_initialization
    args.initialization_mode=resolve_initialization(args.curriculum,args.initialization_mode).value
    if args.resume_checkpoint is not None:
        if not (args.train and args.c1_staged_training and args.train_sampler in (None, 'catalog')):
            raise ValueError('--resume-checkpoint requires --train --c1-staged-training and catalog sampling')
        if not args.resume_checkpoint.is_file():
            raise ValueError('resume checkpoint does not exist')
        if args.output is None or args.output.resolve() == args.resume_checkpoint.parent.resolve():
            raise ValueError('resume requires a NEW output directory')
    if args.initialization_mode=='assembled_state':
        if not args.train or args.curriculum != 'c1':
            raise ValueError('assembled_state requires explicit policy-controlled staged C1 training')
        if (not args.c1_staged_training or args.structural_scheduler != 'event'
                or args.c1_probe_config is None):
            raise ValueError('assembled_state requires staged/event guided C1 probe configuration')
        if args.snapshot_bank is None or not args.snapshot_bank.is_file():
            raise ValueError('assembled_state requires an existing --snapshot-bank manifest')
        if args.resume_checkpoint is not None:
            raise ValueError('assembled_state cannot resume historical MAPPO checkpoints')
        if not math.isfinite(args.episode_seconds) or not 0 < args.episode_seconds <= 180.:
            raise ValueError('assembled_state requires episode duration in (0,180] seconds')
        from drl.MARL.c1_config import C1Settings
        settings=C1Settings.load(args.c1_probe_config)
        if settings.values.get('episode_mode') != 'guided':
            raise ValueError('assembled_state C1 requires the guided episode contract')
        if args.snapshot_stage_weights is not None:
            from drl.MARL.c1_snapshot_sampling import STAGES
            try:
                weights=json.loads(args.snapshot_stage_weights)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError('Invalid JSON --snapshot-stage-weights') from exc
            if (not isinstance(weights, dict) or set(weights) != set(STAGES)
                    or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                           for v in weights.values()) or not sum(weights.values())):
                raise ValueError('Invalid --snapshot-stage-weights mapping')
            args.snapshot_stage_weights=weights
    elif args.snapshot_bank is not None or args.snapshot_stage_weights is not None:
        raise ValueError('--snapshot-bank and weights require assembled_state')
    if args.initialization_mode=='loose_modules':
        if (args.train and not args.c1_integration_probe) or args.benchmark or args.evaluate:
            if args.benchmark or args.evaluate:
                raise ValueError(
                    'Benchmark/evaluation loose_modules non autorizzati da questo guard'
                )
            if args.train:
                if not (
                    getattr(args, 'c1_staged_training', False)
                    and args.structural_scheduler == 'event'
                    and getattr(args, 'c1_probe_config', None) is not None
                ):
                    raise ValueError(
                        'C1 train richiede staged training, scheduler event, '
                        'guided reward e episodi di 180s'
                    )
                probe_path = Path(args.c1_probe_config)
                probe = json.loads(probe_path.read_text())
                if probe.get('episode_mode') in ('guided','discovery'):
                    from drl.MARL.c1_config import C1Settings
                    C1Settings.load(probe_path)  # validate the explicit new contract
                    if not math.isfinite(args.episode_seconds) or not 0 < args.episode_seconds <= 180.:
                        raise ValueError('C1 phase C episode duration requires (0,180] seconds')
                elif probe.get('target_graph_shaping_mode') != 'temporary' or args.episode_seconds != 180.:
                    raise ValueError('Legacy C1 train requires guided reward temporanea e episodi di 180s')
        if args.check_partial_reset or args.check_reassembly or args.controlled_update:
            raise ValueError('legacy Expert checks cannot control the new C1 diagnostic')
    if getattr(args, "c1_staged_training", False) and not args.train:
        raise ValueError("--c1-staged-training requires --train")
    training = args.train or args.benchmark or args.smoke_train
    if args.early_stopping_patience is None:
        args.early_stopping_patience = 3 if args.train else 0
    if args.train:
        domains = {'navigation', 'stairs', 'gap', 'manipulation'}
        if len(args.domains) != len(domains) or set(args.domains) != domains:
            raise ValueError('--train requires exactly four unique C1 domains: navigation, stairs, gap, manipulation')
        if args.initialization_mode == 'assembled_state':
            # The old frozen validation child starts from Expert-assembled
            # courses; it is NOT an honest evaluation of the new partial-bank
            # training objective.  Disable it explicitly until a separate,
            # split-safe partial-bank evaluation is implemented. Retain bounded
            # wall time and periodic, recoverable model checkpoints.
            if args.early_stopping_patience != 0:
                raise ValueError('snapshot C1 requires explicit --early-stopping-patience 0; '
                                 'legacy frozen validation is not comparable')
            if args.checkpoint_every_transitions <= 0:
                raise ValueError('snapshot C1 without validation requires periodic checkpoints')
        elif args.early_stopping_patience <= 0:
            raise ValueError('--train requires positive validation early-stopping patience')
    if args.train and args.max_wall_seconds is None:
        args.max_wall_seconds = 86400.
    if args.max_wall_seconds is not None and (not math.isfinite(args.max_wall_seconds) or not 0 < args.max_wall_seconds <= 86400):
        raise ValueError('campaign wall limit must be positive and at most 24 hours (86400 seconds)')
    if args.early_stopping_patience < 0 or args.early_stopping_min_delta < 0:
        raise ValueError('early stopping patience and min delta must be nonnegative')
    if args.early_stopping_patience and not args.train:
        raise ValueError('validation model selection is allowed only with --train')
    if args.validation_every_seconds <= 0 or args.validation_vector_steps < 1 or args.validation_timeout_seconds <= 0:
        raise ValueError('validation cadence, vector budget and timeout must be positive')
    if args.train_sampler is None:
        args.train_sampler = 'catalog' if training else 'fixed'

    if args.c1_integration_probe:
        if not (
            args.train
            and args.c1_staged_training
            and args.curriculum == "c1"
            and args.initialization_mode == "loose_modules"
        ):
            raise ValueError(
                "Integration probe requires staged C1 training"
            )

        if (
            args.num_envs != 8
            or args.rollout_steps != 32
            or args.learning_epochs != 2
            or args.mini_batches != 8
            or args.structural_scheduler != "event"
        ):
            raise ValueError(
                "Probe requires 8env/roll32/2epochs/8minibatches/event"
            )

        if args.total_transitions not in (32, 64):
            raise ValueError(
                "Probe permits only 32 or 64 SMDP transitions"
            )

        if (
            args.max_vector_steps is None
            or not 1 <= args.max_vector_steps <= 1000
            or args.max_wall_seconds is None
            or not 900 <= args.max_wall_seconds <= 3600
        ):
            raise ValueError(
                "Probe requires bounded vector steps and wall time"
            )

        if (
            args.train_sampler != "catalog"
            or args.seed_manifest is not None
            or args.validation_every_seconds < args.max_wall_seconds
            or args.checkpoint_every_transitions != 0
        ):
            raise ValueError(
                "Probe requires full catalog and no extra validation"
            )

    if args.train and args.train_sampler != 'catalog':
        raise ValueError('real C1 requires full TRAIN catalog resampling')
    if args.train and args.seed_manifest is not None:
        raise ValueError('real C1 requires the full TRAIN catalog; smoke seed manifests are forbidden')
    if args.evaluate:
        if args.train_sampler != 'fixed' or not args.seeds:
            raise ValueError('evaluation requires explicit fixed split seeds, without resampling')
        if args.total_transitions is not None or args.controlled_update or args.check_partial_reset:
            raise ValueError('evaluation forbids optimizer budgets, controlled updates and diagnostic resets')
        if args.max_vector_steps is None or args.max_vector_steps < 1:
            raise ValueError('evaluation requires a positive bounded --max-vector-steps')
        if args.output is None:
            raise ValueError('evaluation requires --output')
    elif args.policy_checkpoint is not None:
        raise ValueError('--policy-checkpoint is a frozen evaluation input')
    elif not training and args.timesteps is not None:
        if args.timesteps < 1:
            raise ValueError('bounded vector decision budget must be positive')
        args.rollout_steps = args.timesteps
    if training:
        if args.total_transitions is None or args.total_transitions <= 0:
            raise ValueError('--total-transitions must specify a positive active environment budget')
        if args.output is None:
            raise ValueError('--output is required for measured training modes')
        if (args.output / 'result.json').exists():
            raise ValueError('--output already contains result.json; use a new run directory')
        if args.timesteps is not None:
            raise ValueError('use --total-transitions and --max-vector-steps; --timesteps counts vector iterations')
        if args.episode_seconds <= 0 or args.checkpoint_every_transitions < 0:
            raise ValueError('episode horizon must be positive and checkpoint cadence nonnegative')
        if args.num_envs < len(args.domains):
            raise ValueError('all requested domains require at least one native environment each')
        if args.learning_epochs < 2 or args.mini_batches < 2:
            raise ValueError('training diagnostics require at least two epochs and two minibatches')
        if args.benchmark:
            if args.total_transitions % args.num_envs:
                raise ValueError('benchmark transition budget must be divisible by num-envs')
            args.rollout_steps = args.total_transitions // args.num_envs
            args.max_vector_steps = args.rollout_steps
        elif args.max_vector_steps is None:
            # A guard, not an alternative budget: Expert occupancy is measured
            # and can make this upper bound expire before the active target.
            args.max_vector_steps = max(18000, 8 * math.ceil(args.total_transitions / args.num_envs))
    return args


def main(argv=None):
    run_started = time.monotonic()
    from isaaclab.app import AppLauncher
    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    try:
        configure_run_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    if args.num_envs < 2 or min(args.rollout_steps, args.learning_epochs, args.mini_batches) < 1:
        parser.error("requires num_envs >= 2 and positive rollout/epoch/minibatch counts")
    if args.seeds is not None and len(args.seeds) != args.num_envs:
        parser.error("--seeds requires exactly one seed per environment")
    if args.mini_batches > args.rollout_steps * args.num_envs:
        parser.error("mini-batches exceeds rollout transition count")
    if args.train and args.num_envs < len(args.domains):
        parser.error("--train requires num-envs >= domain count for the fixed per-env curriculum bank")
    if args.check_partial_reset and (args.train or args.rollout_steps < 2):
        parser.error("--check-partial-reset requires a bounded rollout of at least two steps")
    if args.check_reassembly and not args.check_partial_reset:
        parser.error("--check-reassembly requires --check-partial-reset")
    if args.train and args.controlled_update:
        parser.error("--controlled-update is a bounded check, incompatible with --train")
    monitor = None
    if args.output is not None:
        from drl.MARL.c1_resource_monitor import GPUMemoryMonitor
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'run_arguments.json').write_text(json.dumps(vars(args),default=str,indent=2))
        monitor = GPUMemoryMonitor(args.output/'gpu_samples.csv').start()
    application = AppLauncher(args).app
    env = None
    try:
        # Repository-owned libraries need no ROS overlay in the native path.
        import sys
        import torch
        seed_native_training(args.global_seed)
        args.repository_root = args.repository_root.resolve()
        for source in (args.repository_root / "mssr_ws/src/mssr_expert",
                       args.repository_root / "scripts/smores_ep/src"):
            sys.path.insert(0, str(source))
        from drl.MARL.c1_runner import CHECKPOINT, build_models, build_observation_bridge
        from drl.MARL.isaaclab_c1_env import MSSRC1IsaacLabEnv, make_c1_isaaclab_env_cfg
        from isaaclab_rl.skrl import SkrlVecEnvWrapper
        # Kit caches importable filenames. Load the complete selected path
        # before the expensive physical assembly; fail immediately on imports.
        training_runtime = evaluation_runtime = None
        if args.train or args.benchmark or args.smoke_train:
            from drl.MARL.isaaclab_c1_training import run_native_training
            training_runtime = run_native_training
        if args.evaluate:
            from drl.MARL.isaaclab_c1_evaluation import run_native_evaluation
            evaluation_runtime = run_native_evaluation
        args.repository_root = args.repository_root.resolve()
        args.checkpoint = args.checkpoint or args.repository_root / CHECKPOINT
        bridge = build_observation_bridge(args.repository_root)
        actor, critic = build_models(args, bridge)
        owners = tuple(f"smores_{index:02d}" for index in range(1, 9))
        cfg = make_c1_isaaclab_env_cfg(possible_agents=owners, num_envs=args.num_envs, device=args.device)
        cfg.reward_config_path = str(args.reward_config.resolve())
        cfg.domains = tuple(args.domains)
        cfg.shaping_gamma = args.gamma
        cfg.seed = args.global_seed
        cfg.episode_length_s = args.episode_seconds
        cfg.c1_probe_config_path=(
            str(args.c1_probe_config.resolve())
            if args.c1_probe_config is not None else None
        )
        cfg.resample_train = args.train_sampler != 'fixed'
        cfg.train_sampler_seed = args.global_seed
        cfg.smoke_seed_cycle = args.train_sampler == 'smoke-cycle'
        cfg.course_split = args.evaluation_split if args.evaluate else 'train'
        cfg.initialization_mode = args.initialization_mode
        if args.initialization_mode in ('loose_modules','assembled_state'):
            cfg.c1_evidence_directory=str(Path(args.output)/'c1_discovery')
            # Same read-only native state cache used by the accelerated GUI.
            # Actor/critic weights and dynamic policy embeddings are never cached.
            cfg.c1_readonly_state_cache = args.structural_scheduler == 'event'
        if args.initialization_mode=='assembled_state':
            cfg.snapshot_bank_path=str(args.snapshot_bank.resolve())
            cfg.snapshot_stage_weights=args.snapshot_stage_weights
        if args.seed_manifest is not None:
            cfg.train_seed_pools = json.loads(args.seed_manifest.read_text())['seed_pools']
        if args.seeds is not None:
            cfg.seeds = tuple(args.seeds)
        env = MSSRC1IsaacLabEnv(cfg)
        if args.train:
            env._wall_deadline_s = run_started + args.max_wall_seconds - args.checkpoint_reserve_seconds
        wrapper = SkrlVecEnvWrapper(env, ml_framework="torch")
        initial_observations, _ = wrapper.reset()
        startup_s = time.monotonic() - run_started
        assembly_started = time.monotonic()
        if args.initialization_mode=='expert_validation':
            env.assemble_until_ready(max_steps=args.assembly_max_steps)
        assembly_s = time.monotonic() - assembly_started
        # Standard wrapper.step refreshes its cached initial observation after
        # direct Expert assembly, before any MARL transition is eligible.
        import torch
        if args.initialization_mode=='expert_validation':
            initial_observations, _, _, _, _ = wrapper.step(
                {owner: torch.zeros(args.num_envs, 1, device=env.device) for owner in owners})
        env.initialize_policy_bridge(observation_bridge=bridge,
            structural_scheduler=args.structural_scheduler, bulk_host_reads=True)
        learner = create_native_learner(args, bridge, actor, critic, owners, env.runtime_context_provider)
        if args.resume_checkpoint is not None:
            from drl.MARL.c1_longtrain_resume_integration import resume_fresh_native_episodes
            resume_fresh_native_episodes(args, env, wrapper, learner)
        startup_s += time.monotonic() - assembly_started - assembly_s
        if args.evaluate:
            if args.policy_checkpoint is not None:
                learner.load(str(args.policy_checkpoint))
            result=evaluation_runtime(args,env,wrapper,learner)
            print('NATIVE_C1_EVALUATION_PASS: '+json.dumps({key:result.get(key) for key in ('status','split','optimizer_updates','parameters_unchanged')}),flush=True)
            return 0
        if args.train or args.benchmark or args.smoke_train:
            result = training_runtime(args,env,wrapper,learner,
                startup_s=startup_s,assembly_s=assembly_s,run_started=run_started)
            if monitor is not None:
                result['gpu_monitor'] = monitor.stop()
                result['memory_bytes'].update({key:value for key,value in result['gpu_monitor'].items()
                    if key.endswith('_bytes')})
            result['bc_checkpoint'] = str(args.checkpoint)
            from drl.MARL.c1_runner import BC_SHA256
            result['bc_sha256'] = BC_SHA256
            result['vector_decision_steps'] = result['vector_steps']
            result['simulated_vector_seconds'] = result['vector_steps']*env.step_dt
            result['simulated_active_environment_seconds'] = result['active_marl_transitions']*env.step_dt
            result['physics_steps'] = env._sim_step_counter
            result['configured_episode_seconds'] = args.episode_seconds
            result['structural_scheduler'] = args.structural_scheduler
            if hasattr(env.runtime_context_provider, 'structural_option_metrics'):
                result['structural_option_metrics'] = env.runtime_context_provider.structural_option_metrics()
            for domain,row in result['domains'].items():
                row['reward_sum'] = sum(e['active_return'] for e in row['episodes'])
                row['reward_per_active_transition'] = row['reward_sum']/row['active_marl_transitions'] if row['active_marl_transitions'] else None
            (args.output/'result.json').write_text(json.dumps(result,indent=2))
            sentinel = 'NATIVE_C1_TRAIN_PASS' if result['status']=='pass' else 'NATIVE_C1_TRAIN_FAIL'
            print(sentinel+': '+json.dumps({key:result.get(key) for key in (
                'status','num_envs','active_marl_transitions','row_transitions','stop_reason','timings_s','error')}),flush=True)
            if result['status'] != 'pass':
                raise RuntimeError('Native measured training gate failed; see result.json')
            return 0
        before_step = reset_proof = None
        if args.check_partial_reset:
            before_step, reset_proof = make_partial_reset_probe(
                env, wrapper, learner, reset_timestep=args.rollout_steps // 2)
        summary = collect_native_rollout(wrapper, learner, rollout_steps=args.rollout_steps,
            initial_observations=initial_observations, initial_states=wrapper.state(), before_step=before_step)
        if reset_proof is not None:
            if not reset_proof:
                raise RuntimeError("partial reset probe was not exercised")
            reset_row = summary["rollout_active_by_step"][args.rollout_steps // 2]
            if reset_row[0]:
                raise RuntimeError("reset Expert row incorrectly remained active for learning")
            if not all(reset_row[1:]):
                raise RuntimeError("partial reset check requires unaffected environments to continue MARL")
            summary["partial_reset"] = reset_proof
        if args.controlled_update:
            learner.coordinated_update(timestep=args.rollout_steps, timesteps=args.rollout_steps)
            metrics = dict(learner.last_update_metrics)
            if not metrics or any(not math.isfinite(float(value)) for value in metrics.values()):
                raise RuntimeError("missing or nonfinite controlled optimizer diagnostics")
            summary["controlled_update"] = metrics
        if args.check_reassembly:
            env.assemble_until_ready(max_steps=args.assembly_max_steps)
            summary["reassembly"] = dict(
                env_index=0, generation=env.generation[0],
                physical_connections=len(env.docking_managers[0].connections),
                topology_verified=bool(env.assembly_verified[0]),
                phase=env.handoffs[0].phase.value,
            )
            if not summary["reassembly"]["topology_verified"] or summary["reassembly"]["physical_connections"] != 7:
                raise RuntimeError("partial reset did not finish physically verified reassembly")
        print("NATIVE_C1_PRETRAIN_PASS: " + json.dumps(summary), flush=True)
        return 0
    except BaseException as error:
        import traceback
        traceback.print_exc()
        print('NATIVE_C1_PRETRAIN_FAIL: '+json.dumps(dict(type=type(error).__name__,message=str(error))),flush=True)
        if args.output is not None and not (args.output/'result.json').exists():
            (args.output/'result.json').write_text(json.dumps(dict(status='fail',num_envs=args.num_envs,
                requested_total_transitions=args.total_transitions,error=dict(type=type(error).__name__,message=str(error))),indent=2))
        raise
    finally:
        if monitor is not None:
            snapshot=monitor.stop()
            (args.output/'gpu_monitor.json').write_text(json.dumps(snapshot,indent=2))
        if env is not None:
            env.close()
        application.close()


if __name__ == "__main__":
    raise SystemExit(main())
