"""Frozen paired native C1 evaluation through the standard SKRL eval trainer."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
from skrl.trainers.torch import SequentialTrainer

from drl.MARL.c1_train_sampling import catalog_seeds
from drl.MARL.c1_training_telemetry import TrainingTelemetry

SPLITS = ('train', 'validation', 'test', 'challenge')
MODEL_NAMES = ('shared_actor', 'shared_critic', 'altruistic_owner_critic',
               'altruistic_mean_field_critic')


def evaluation_protocol(banklist):
    """Document the exact paired bank and the separate roles of dataset splits."""
    return dict(schema='mssr.c1_evaluation_protocol.v1', seed_bank=deepcopy(list(banklist)),
        paired_policies=['frozen_bc_v4', 'frozen_learned_appo'],
        pairing='same seed bank, split, vector budget, simulator configuration and calibrated deterministic policy rule',
        stochastic_evaluation=False,
        train=dict(optimizer_updates_allowed=True, role='TRAIN-only policy optimization'),
        validation=dict(optimizer_updates_allowed=False, model_selection=True, role='checkpoint selection'),
        test=dict(optimizer_updates_allowed=False, tuning_allowed=False, role='frozen final comparison'),
        challenge=dict(optimizer_updates_allowed=False, independent_stress=True,
                       role='separate stress report, never pooled into model selection or TEST'))


def _snapshot_models(learner):
    return {name: {key: parameter.detach().cpu().clone()
                   for key, parameter in getattr(learner, name).named_parameters()}
            for name in MODEL_NAMES}


def _digest(parameters):
    digest = hashlib.sha256()
    for name, value in sorted(parameters.items()):
        digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode())
        digest.update(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _checkpoint(path):
    if path is None:
        return None
    path = Path(path)
    sha = None
    if path.is_file():
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        sha = digest.hexdigest()
    return dict(path=str(path), sha256=sha)


def run_native_evaluation(args, env, wrapper, learner):
    """Run a finite fixed-bank evaluation, fail closed on any parameter update.

    The caller constructs the requested split's geometry and loads an optional
    learned checkpoint before entry. This function never loads or saves models.
    """
    split = getattr(args, 'evaluation_split', None)
    limit = getattr(args, 'max_vector_steps', None)
    if not getattr(args, 'evaluate', False) or split not in SPLITS:
        raise ValueError('evaluation requires --evaluate and an allocated evaluation split')
    if type(limit) is not int or limit <= 0:
        raise ValueError('evaluation requires a finite positive integer max_vector_steps')
    if getattr(args, 'output', None) is None:
        raise ValueError('evaluation output directory is required')
    if any(getattr(args, name, False) for name in ('train', 'benchmark', 'smoke_train', 'controlled_update')):
        raise ValueError('evaluation cannot enable a TRAIN optimizer mode')
    if getattr(env.cfg, 'resample_train', False):
        raise ValueError('evaluation requires fixed courses with TRAIN resampling disabled')
    if getattr(env.cfg, 'course_split', split) != split:
        raise ValueError('configured course split differs from evaluation split')
    domains, seeds = tuple(env.domain_by_env), tuple(env.seed_by_env)
    if len(domains) != len(seeds) or len(seeds) != wrapper.num_envs:
        raise ValueError('evaluation bank requires exactly one domain/seed per native environment')
    bank = [dict(env_index=i, domain=domain, seed=seed, split=split)
            for i, (domain, seed) in enumerate(zip(domains, seeds))]
    for row in bank:
        if type(row['seed']) is not int or row['seed'] not in catalog_seeds(row['domain'], split):
            raise ValueError(f'evaluation seed outside {split} catalog: {row}')

    before = _snapshot_models(learner)
    telemetry = TrainingTelemetry()
    original = []
    counts = dict(vector_steps=0, optimizer_attempts=0, recorded_steps=0)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    def install(obj, name, replacement):
        original.append((obj, name, name in vars(obj), vars(obj).get(name)))
        setattr(obj, name, replacement)

    def assert_bank():
        if tuple(env.seed_by_env) != seeds or tuple(env.domain_by_env) != domains:
            raise RuntimeError('fixed evaluation bank changed during evaluation')

    def forbidden_update(*positional, **keywords):
        counts['optimizer_attempts'] += 1
        raise RuntimeError('optimizer updates are forbidden during frozen evaluation')

    def clear_storage():
        for memory in learner.memories.values():
            memory.reset()
        for name in ('_structured_rollout_sidecar', '_altruistic_rollout_sidecar'):
            sidecar = getattr(learner, name, None)
            if sidecar is not None:
                sidecar.clear()

    original_record = learner.record_transition
    original_step = wrapper.step

    def passive_record(*positional, **keywords):
        try:
            result = original_record(*positional, **keywords)
            counts['recorded_steps'] += 1
            return result
        finally:
            # Keep provider, online decisions and next critic context intact.
            clear_storage()

    def fixed_step(*positional, **keywords):
        assert_bank()
        result = original_step(*positional, **keywords)
        counts['vector_steps'] += 1
        assert_bank()
        return result

    error = None
    install(env, 'training_telemetry', telemetry)
    install(learner, 'coordinated_update', forbidden_update)
    install(learner, 'record_transition', passive_record)
    install(learner, 'deterministic_policy_actions', True)
    install(wrapper, 'step', fixed_step)
    try:
        clear_storage()
        trainer = SequentialTrainer(env=wrapper, agents=learner,
            cfg=dict(timesteps=limit, headless=True, disable_progressbar=True,
                     close_environment_at_exit=False, stochastic_evaluation=True))
        # SKRL must pass through act()'s fixed-shape carrier. The structured
        # sampler above makes the actual semantic action deterministic; SKRL
        # must not replace it with another tensor encoding/mean action.
        trainer.eval()
    except Exception as caught:
        error = caught
    finally:
        clear_storage()
        after = _snapshot_models(learner)
        for obj, name, existed, value in reversed(original):
            if existed:
                setattr(obj, name, value)
            else:
                delattr(obj, name)
    proof = {}
    for name in MODEL_NAMES:
        old, new = before[name], after[name]
        unchanged = old.keys() == new.keys() and all(torch.equal(value, new[key]) for key, value in old.items())
        proof[name] = dict(unchanged=unchanged, parameter_tensors=len(old),
                           before_sha256=_digest(old), after_sha256=_digest(new))
    all_unchanged = all(row['unchanged'] for row in proof.values())
    if not all_unchanged and error is None:
        error = RuntimeError('frozen evaluation changed model parameters')
    result = telemetry.snapshot()
    if result['row_transitions'] != counts['vector_steps'] * wrapper.num_envs and error is None:
        error = RuntimeError('evaluation reward telemetry does not cover every native row transition')
    if any(row['nonfinite'] for row in result['finite_checks'].values()) and error is None:
        error = RuntimeError('nonfinite evaluation transition evidence')
    result.update(schema='mssr.c1_native_evaluation.v1', status='fail' if error else 'pass',
        split=split, seed_bank=bank, protocol=evaluation_protocol(bank),
        policy='learned_appo' if getattr(args, 'policy_checkpoint', None) else 'bc_v4',
        bc_checkpoint=_checkpoint(getattr(args, 'checkpoint', None)),
        policy_checkpoint=_checkpoint(getattr(args, 'policy_checkpoint', None)),
        global_seed=getattr(args, 'global_seed', None), stochastic_evaluation=False,
        vector_steps=counts['vector_steps'], recorded_steps=counts['recorded_steps'],
        optimizer_updates=0, optimizer_attempts=counts['optimizer_attempts'],
        frozen_parameter_proof=dict(all_unchanged=all_unchanged, models=proof),
        error=None if error is None else dict(type=type(error).__name__, message=str(error)))
    (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if error is not None:
        raise RuntimeError(f'frozen native evaluation failed: {error}') from error
    return result
