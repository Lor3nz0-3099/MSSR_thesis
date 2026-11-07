"""Explicit, offline-verified continuation of one manifest-less APPO artifact.

This is deliberately separate from staged resume manifests. Historical RNG,
physical episodes, pending options and sampler state were not checkpointed.
The gate is reconstructed by replay; it was not present in the artifact.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

import torch

from drl.MARL.c1_critic_quality_gate import C1CriticQualityGate
from drl.MARL.c1_longtrain_checkpoint import sha256_file, validate_counters
from drl.MARL.c1_longtrain_resume_core import _check_empty_rollout
from drl.MARL.c1_longtrain_schedule import C1LongtrainSchedule

HISTORICAL_SCHEMA = 'mssr.c1.historical_appo.v1'
# These are evidence anchors for the explicitly authorized historical run,
# not a general escape hatch for checkpoints with absent resume manifests.
_HISTORICAL_HASHES = {
    'checkpoint_5048.pt': '746f186933766009414c94dce5e41b5295f07f22c5e2922bb740c7a87903285b',
    'updates.jsonl': 'b4d35cc58965de6e5ca37b94be5ce6baead4533aa1b0dc3d3f301c9403adb8f2',
    'result.json': '4b7ea06014d72784a086108b26b1b503c2a926374b39c1bb9cbfbc340a3c36fd',
    'run_arguments.json': '331d4204d9adde651553eab6246a9a779569437831d829f3d99f6d2d50d47b56',
    'training_telemetry.json': '391a9788d22e1c9f8bd77725423a521c4ba1b2f9ef8e8b245f8d64889bd4ec1d',
}
_CONFIG_HASHES = {
    'reward_config': 'd9c74751a479630421f80600b37409944db3c393d2eef8963d0624914c7c2020',
    'c1_probe_config': 'ecd966e7ac2d750a99a78ca70c9ec3b1a161f6212b4db39a18468d2b78b905cd',
    'snapshot_bank': '95d78aa35317b0d41de9437b5844f80bcadf3ce0a4c818bbc1d981c071a2a106',
}
_OPTIMIZER_MODELS = {
    'actor_optimizer': 'policy', 'critic_optimizer': 'value',
    'owner_critic_optimizer': 'owner_critic',
    'mean_field_critic_optimizer': 'mean_field_critic',
}


def _finite(value, label):
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f'Nonfinite tensor: {label}')
    elif isinstance(value, dict):
        for key, child in value.items():
            _finite(child, f'{label}.{key}')
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite(child, f'{label}[{index}]')
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'Nonfinite value: {label}')


def _equal(actual, expected, label):
    if torch.is_tensor(expected):
        valid = (torch.is_tensor(actual) and actual.shape == expected.shape
                 and actual.dtype == expected.dtype
                 and torch.equal(actual.detach().cpu(), expected.detach().cpu()))
    elif isinstance(expected, dict):
        valid = isinstance(actual, dict) and actual.keys() == expected.keys()
        if valid:
            for key in expected:
                _equal(actual[key], expected[key], f'{label}.{key}')
    elif isinstance(expected, (list, tuple)):
        valid = type(actual) is type(expected) and len(actual) == len(expected)
        if valid:
            for index, child in enumerate(expected):
                _equal(actual[index], child, f'{label}[{index}]')
    else:
        valid = actual == expected
    if not valid:
        raise ValueError(f'Full state restore equality mismatch: {label}')


def _state_digest(state):
    """Stable tensor/content hash, independent of torch archive metadata."""
    digest = hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b'{')
            for key in sorted(value, key=str):
                digest.update(repr(key).encode())
                visit(value[key])
            digest.update(b'}')
        elif isinstance(value, (tuple, list)):
            digest.update(type(value).__name__.encode())
            for child in value:
                visit(child)
        else:
            digest.update(repr(value).encode())
    visit(state)
    return digest.hexdigest()


def _json(path):
    def reject_constant(value):
        raise ValueError(f'Nonfinite JSON constant: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject_constant)


def _safe_checkpoint(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError('Checkpoint requires a nonempty module registry')
    _finite(state, 'checkpoint')
    return state


def _validate_adam(optimizer, saved, model, label):
    if not isinstance(optimizer, torch.optim.Adam):
        raise ValueError(f'Expected Adam optimizer: {label}')
    if set(saved) != {'state', 'param_groups'}:
        raise ValueError(f'Invalid Adam state dictionary: {label}')
    groups = saved['param_groups']
    if len(groups) != len(optimizer.param_groups):
        raise ValueError(f'Adam parameter group coverage mismatch: {label}')
    flat_parameters, identifiers = [], []
    for current, source in zip(optimizer.param_groups, groups):
        if len(current['params']) != len(source['params']):
            raise ValueError(f'Adam parameter group size mismatch: {label}')
        # LRs are saved progression; every other optimizer setting must match.
        for key in set(current) | set(source):
            if key not in ('params', 'lr', 'initial_lr') and current.get(key) != source.get(key):
                raise ValueError(f'Adam configuration mismatch {label}.{key}')
        flat_parameters.extend(current['params'])
        identifiers.extend(source['params'])
        if not isinstance(source['lr'], (int, float)) or source['lr'] < 0:
            raise ValueError(f'Invalid saved learning rate: {label}')
    if (len(set(identifiers)) != len(identifiers)
            or any(type(identifier) is not int for identifier in identifiers)
            or len({id(p) for p in flat_parameters}) != len(flat_parameters)):
        raise ValueError(f'Adam parameter aliases or identifiers inconsistent: {label}')
    if [id(p) for p in flat_parameters] != [id(p) for p in model.parameters()]:
        raise ValueError(f'Adam/model parameter coverage or order mismatch: {label}')
    mapping = dict(zip(identifiers, flat_parameters))
    if not set(saved['state']).issubset(mapping):
        raise ValueError(f'Adam state contains unknown parameter: {label}')
    for identifier, entry in saved['state'].items():
        required = {'step', 'exp_avg', 'exp_avg_sq'}
        group = next(group for group in groups if identifier in group['params'])
        if group.get('amsgrad'):
            required.add('max_exp_avg_sq')
        if set(entry) != required:
            raise ValueError(f'Adam moment coverage mismatch: {label}.{identifier}')
        parameter = mapping[identifier]
        step = entry['step']
        if (not torch.is_tensor(step) or step.numel() != 1 or step.ndim != 0
                or float(step) < 1 or not float(step).is_integer()):
            raise ValueError(f'Invalid Adam step: {label}.{identifier}')
        for key in required - {'step'}:
            value = entry[key]
            if (not torch.is_tensor(value) or value.shape != parameter.shape
                    or value.dtype != parameter.dtype):
                raise ValueError(f'Adam moment shape/dtype mismatch: {label}.{identifier}.{key}')
            if key in ('exp_avg_sq', 'max_exp_avg_sq') and bool((value < 0).any()):
                raise ValueError(f'Negative Adam second moment: {label}.{identifier}.{key}')
    return {
        'state_entries': len(saved['state']), 'parameters': len(identifiers),
        'step_values': sorted({int(float(entry['step'])) for entry in saved['state'].values()}),
        'group_lrs': [group['lr'] for group in groups],
        'sha256': _state_digest(saved), 'equal': True,
    }


def restore_full_state(learner, checkpoint, expected_checkpoint_sha256=None):
    """Strictly restore all registered models and Adam state, then compare all.

    SKRL's loader historically requests unrestricted unpickling. Keep its actual
    load implementation but force every nested torch.load to weights_only=True.
    This is a synchronous offline/startup boundary, never a training-thread API.
    """
    path = Path(checkpoint).resolve()
    checkpoint_hash = sha256_file(path)
    if expected_checkpoint_sha256 is not None and checkpoint_hash != expected_checkpoint_sha256:
        raise ValueError('Checkpoint SHA256 hash mismatch')
    saved = _safe_checkpoint(path)
    registry = learner.checkpoint_modules
    if not isinstance(registry, dict) or set(registry) != set(saved):
        raise ValueError('Full checkpoint agent registry coverage mismatch')
    models, optimizers, aliases, module_ids, parameter_owners = {}, {}, {}, {}, {}
    for uid, modules in registry.items():
        if not isinstance(saved[uid], dict) or set(modules) != set(saved[uid]):
            raise ValueError(f'Full checkpoint module registry coverage mismatch: {uid}')
        for name, module in modules.items():
            label = f'{uid}:{name}'
            source = saved[uid][name]
            if id(module) in module_ids:
                alias = module_ids[id(module)]
                alias_uid, alias_name = alias.split(':', 1)
                _equal(source, saved[alias_uid][alias_name], label + '.alias')
                aliases[label] = alias
            else:
                module_ids[id(module)] = label
            if isinstance(module, torch.optim.Optimizer):
                model_name = _OPTIMIZER_MODELS.get(name)
                if model_name is None or model_name not in modules:
                    raise ValueError(f'Unknown optimizer/model relationship: {label}')
                optimizers[label] = _validate_adam(module, source, modules[model_name], label)
                continue
            if not isinstance(module, torch.nn.Module):
                raise ValueError(f'Unsupported checkpoint registry module: {label}')
            current = module.state_dict()
            if set(current) != set(source):
                raise ValueError(f'Model architecture key coverage mismatch: {label}')
            for key, tensor in current.items():
                if (not torch.is_tensor(source[key]) or source[key].shape != tensor.shape
                        or source[key].dtype != tensor.dtype):
                    raise ValueError(f'Model architecture shape/dtype mismatch: {label}.{key}')
            for parameter in module.parameters():
                previous = parameter_owners.setdefault(id(parameter), id(module))
                if previous != id(module):
                    raise ValueError(f'Unexpected shared model parameter alias: {label}')
            models[label] = {'tensors': len(source), 'sha256': _state_digest(source), 'equal': True}
    if not models or not optimizers:
        raise ValueError('Full state requires both models and optimizers')
    actual_torch_load = torch.load
    def safe_load(*args, **kwargs):
        kwargs['weights_only'] = True
        return actual_torch_load(*args, **kwargs)
    with patch.object(torch, 'load', safe_load):
        learner.load(str(path))
    for uid, modules in registry.items():
        for name, module in modules.items():
            _equal(module.state_dict(), saved[uid][name], f'{uid}:{name}')
    if sha256_file(path) != checkpoint_hash:
        raise ValueError('Checkpoint hash changed during restore')
    return {
        'schema': 'mssr.c1.full_state_audit.v1', 'checkpoint_path': str(path),
        'checkpoint_sha256': checkpoint_hash, 'all_equal': True,
        'models': models, 'optimizers': optimizers, 'aliases': aliases,
        'initial_models_sha256': _state_digest({label: row['sha256'] for label, row in models.items()}),
        'weights_only': True, 'actual_learner_load_called': True,
    }


def _replay_history(rows, telemetry, result, arguments):
    gate = C1CriticQualityGate()
    expected_updates = result['optimizer']['completed_ppo_updates']
    if len(rows) != expected_updates or len(telemetry['updates']) != expected_updates:
        raise ValueError('Historical update history row coverage mismatch')
    totals = {name: 0 for name in _OPTIMIZER_MODELS}
    actor_updates = 0
    previous_vector = previous_transitions = 0
    for update, row in enumerate(rows, 1):
        if row['update'] != update:
            raise ValueError('Historical updates must be consecutive and complete')
        vector, transitions = row['vector_step'], row['active_marl_transitions']
        if (type(vector) is not int or type(transitions) is not int
                or vector <= previous_vector or transitions <= previous_transitions):
            raise ValueError('Historical counters are not strictly increasing')
        previous_vector, previous_transitions = vector, transitions
        metrics = row['metrics']
        _finite(metrics, f'update {update}')
        if metrics != telemetry['updates'][update - 1]:
            raise ValueError('Historical telemetry/update metrics mismatch')
        step = gate.step_for_update(update)
        if (row['c1_phase'] != step.phase or metrics['c1_phase_id'] != step.phase_id
                or not math.isclose(metrics['c1_actor_lr'], step.actor_lr, rel_tol=1e-12, abs_tol=1e-16)
                or not math.isclose(metrics['c1_critic_lr'], step.critic_lr, rel_tol=1e-12, abs_tol=1e-16)):
            raise ValueError(f'Historical phase or LR disagrees with gate replay at {update}')
        minibatches = metrics['minibatches']
        if (type(minibatches) is not int or minibatches < 1
                or minibatches > arguments['learning_epochs'] * arguments['mini_batches']):
            raise ValueError('Historical minibatch coverage inconsistent')
        expected = {
            'actor_optimizer': minibatches if step.actor_enabled else 0,
            'critic_optimizer': 0, 'owner_critic_optimizer': minibatches,
            'mean_field_critic_optimizer': minibatches,
        }
        for name, count in expected.items():
            metric = 'global_critic_optimizer_steps' if name == 'critic_optimizer' else name + '_steps'
            if metrics[metric] != count:
                raise ValueError(f'Historical optimizer step count mismatch: {name}')
            totals[name] += count
        actor_updates += int(step.actor_enabled)
        gate.observe_completed(update, metrics)
        if any(metrics[key] != value for key, value in gate.metrics().items()):
            raise ValueError(f'Historical critic gate metrics disagree with replay at {update}')
    counters = validate_counters({
        'completed_updates': expected_updates, 'vector_steps': result['vector_steps'],
        'smdp_transition_count': result['smdp_transitions'],
        'smdp_event_step_count': result['smdp_event_steps'],
        'smdp_policy_version': result['smdp_policy_version'],
    })
    if (previous_vector != counters['vector_steps']
            or previous_transitions != counters['smdp_transition_count']
            or result['active_marl_transitions'] != previous_transitions
            or result['optimizer']['rollout_active_transitions'] != previous_transitions
            or result['smdp_policy_version'] != actor_updates
            or counters['smdp_event_step_count'] > previous_vector
            or telemetry['active_marl_transitions'] != previous_transitions):
        raise ValueError('Historical result/history counter mismatch')
    last_recorded = result['diagnostics']['last_update_metrics']
    if any(rows[-1]['metrics'].get(key) != value for key, value in last_recorded.items()):
        raise ValueError('Historical final diagnostic metrics mismatch')
    return gate, counters, totals


def _scientific_config(arguments, root):
    config = {key: arguments[key] for key in (
        'num_envs', 'domains', 'rollout_steps', 'learning_epochs', 'mini_batches',
        'global_seed', 'seeds', 'train_sampler', 'initialization_mode', 'snapshot_stage_weights',
        'curriculum', 'episode_seconds', 'alpha', 'gamma', 'altruism_radius_m',
        'structural_scheduler', 'assembly_max_steps',
    )}
    config['role_schedule'] = 'critic_quality_gate_then_actor_ramp_then_joint_refinement'
    config['recorded_arguments'] = arguments
    source_files = {}
    bank_path = None
    for key, expected in _CONFIG_HASHES.items():
        path = Path(arguments[key])
        # Preserve the recorded repository-relative identity when relocated.
        recorded_root = Path(arguments['repository_root'])
        if path.is_relative_to(recorded_root):
            path = root / path.relative_to(recorded_root)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f'Historical scientific configuration hash mismatch: {key}')
        config[key] = str(path.resolve())
        config[key + '_sha256'] = expected
        source_files[key] = {'path': str(path.resolve()), 'sha256': expected}
        if key == 'snapshot_bank':
            bank_path = path
    bank = _json(bank_path)
    snapshot_physics = []
    snapshot_hashes = {}
    identities = []
    for record in bank['records']:
        path = (bank_path.parent / record['path']).resolve()
        if not path.is_relative_to(bank_path.parent.resolve()):
            raise ValueError('Historical snapshot path escapes bank')
        if sha256_file(path) != record['sha256']:
            raise ValueError('Historical snapshot hash mismatch')
        snapshot = _json(path)
        if snapshot['identity'] != record['identity'] or snapshot['stage'] != record['stage']:
            raise ValueError('Historical snapshot identity mismatch')
        if snapshot['physics'] not in snapshot_physics:
            snapshot_physics.append(snapshot['physics'])
        if record['identity'] not in identities:
            identities.append(record['identity'])
        snapshot_hashes[record['path']] = record['sha256']
    config['snapshot_physics'] = snapshot_physics
    config['snapshot_identities'] = identities
    config['snapshot_hashes'] = snapshot_hashes
    # Current source hashes are reproducibility evidence, not a claim that the
    # exact same files were recorded at the old launch (no source manifest).
    for relative in (
        'drl/MARL/mappo.py', 'drl/MARL/c1_critic_quality_gate.py',
        'drl/MARL/c1_longtrain_schedule.py', 'drl/MARL/bc_v4_actor.py',
        'drl/MARL/owner_critic.py', 'drl/MARL/global_critic.py',
        'configs/bc_v1/gnn_v1.json', 'configs/bc_v1/action_set_head_v1.json',
    ):
        path = root / relative
        source_files[relative] = {'path': str(path), 'sha256': sha256_file(path),
                                  'evidence': 'current_replay_source_not_historical_launch_manifest'}
    return config, source_files


def inspect_historical_checkpoint(path, repository_root):
    """Authenticate the authorized artifact/history and reconstruct its boundary."""
    path, root = Path(path).resolve(), Path(repository_root).resolve()
    if path.name != 'checkpoint_5048.pt':
        raise ValueError('Unsupported explicit historical checkpoint scope')
    hashes = {}
    for name, expected in _HISTORICAL_HASHES.items():
        source = path if name == path.name else path.parent / name
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f'Historical integrity hash mismatch or missing file: {name}')
        hashes[name] = {'path': str(source), 'sha256': expected}
    arguments = _json(path.parent / 'run_arguments.json')
    result = _json(path.parent / 'result.json')
    telemetry = _json(path.parent / 'training_telemetry.json')
    rows = [json.loads(line) for line in (path.parent / 'updates.jsonl').read_text().splitlines()]
    gate, counters, totals = _replay_history(rows, telemetry, result, arguments)
    config, sources = _scientific_config(arguments, root)
    checkpoint = _safe_checkpoint(path)
    expected_names = set(_OPTIMIZER_MODELS) | set(_OPTIMIZER_MODELS.values())
    expected_uids = {f'smores_{n:02}' for n in range(1, 9)}
    if (set(checkpoint) != expected_uids or set(checkpoint['smores_01']) != expected_names
            or any(checkpoint[uid] for uid in expected_uids - {'smores_01'})):
        raise ValueError('Historical checkpoint registry coverage mismatch')
    adam = {}
    registry = checkpoint['smores_01']
    for name, entries in {'actor_optimizer': 245, 'critic_optimizer': 0,
                          'owner_critic_optimizer': 180, 'mean_field_critic_optimizer': 180}.items():
        optimizer = registry[name]
        states = optimizer['state']
        steps = sorted({int(float(entry['step'])) for entry in states.values()})
        if len(states) != entries or (steps and max(steps) != totals[name]):
            raise ValueError('Historical Adam state coverage/step history mismatch')
        adam[name] = {'state_entries': len(states), 'step_values': steps,
                      'history_step_total': totals[name], 'sha256': _state_digest(optimizer)}
    return {
        'schema': HISTORICAL_SCHEMA, 'scope': 'explicit_historical_appo_continuation',
        'checkpoint_path': str(path), 'checkpoint_sha256': _HISTORICAL_HASHES[path.name],
        'repository_root': str(root), 'counters': counters,
        'critic_gate_state': gate.state_dict(),
        'provenance': {
            'hashes': hashes, 'source_files': sources, 'scientific_config': config,
            'history': {'replayed_updates': len(rows), 'optimizer_step_totals': totals,
                        'first_actor_update': gate.first_actor_update, 'adam': adam},
            'gate_evidence': {'config': 'inferred_current_defaults_validated_against_every_recorded_update',
                              'last_reason': 'replayed_not_recorded',
                              'metrics': 'recorded_and_replayed_for_all_updates'},
            'limitations': [
                'No historical resume manifest; this is an explicitly scoped historical continuation.',
                'Gate configuration is inferred from current defaults; equivalent unseen settings cannot be excluded.',
                'Gate last_reason is reconstructed, never recorded; good_streak/last_ok freeze after unlock.',
                'Historical RNG, pending options, physical episodes, provider epochs and sampler RNG were not saved.',
                'Fresh episodes and fresh provider epochs; policy version is restored, maximum lag starts at zero.',
                'Source hashes identify current replay code; a historical source revision was not recorded.',
                'In-sample critic fit and checkpoint continuity do not establish task success.',
            ],
        },
    }


def restore_historical_learner(learner, context):
    """Restore exact tensors/Adam plus reconstructed counters at fresh episodes."""
    if not isinstance(context, dict) or context.get('schema') != HISTORICAL_SCHEMA:
        raise ValueError('Invalid explicit historical continuation context')
    # Never accept mutable caller-provided reconstructed metadata as authority.
    verified = inspect_historical_checkpoint(context['checkpoint_path'], context['repository_root'])
    if context != verified:
        raise ValueError('Historical continuation context/counter integrity mismatch')
    _check_empty_rollout(learner)
    gate = C1CriticQualityGate.from_state_dict(verified['critic_gate_state'])
    counters = verified['counters']
    next_step = gate.step_for_update(counters['completed_updates'] + 1)
    if next_step.update != 110 or next_step.phase != 'joint_refinement':
        raise ValueError('Historical gate next phase mismatch')
    audit = restore_full_state(learner, verified['checkpoint_path'], verified['checkpoint_sha256'])
    _check_empty_rollout(learner)
    for optimizer, lr in (
        (learner.actor_optimizer, next_step.actor_lr),
        (learner.owner_critic_optimizer, next_step.critic_lr),
        (learner.mean_field_critic_optimizer, next_step.critic_lr),
    ):
        if any(group['lr'] != lr for group in optimizer.param_groups):
            raise ValueError('Historical loaded learning rate differs from next gate schedule')
    learner._smdp_policy_version = counters['smdp_policy_version']
    learner._smdp_transition_count = counters['smdp_transition_count']
    learner._smdp_event_step_count = counters['smdp_event_step_count']
    learner._smdp_event_rows_last_interaction = 0
    learner._smdp_pending = None
    learner._smdp_max_policy_lag = 0
    learner._rollout = 0
    # Applying next stage is allowed only after proving it preserves saved LRs.
    C1LongtrainSchedule().apply(learner, next_step)
    return {
        'schema': 'mssr.c1.historical_boundary_audit.v1', 'full_state': audit,
        'counters': counters, 'critic_gate_state': gate.state_dict(),
        'next_update': next_step.update, 'next_phase': next_step.phase,
        'next_actor_lr': next_step.actor_lr, 'next_critic_lr': next_step.critic_lr,
        'initial_models_sha256': audit['initial_models_sha256'],
        'physical_episode_reused': False, 'pending_options_reused': False,
        'provider_epochs': 'fresh_episode_provider_epochs_not_checkpointed',
        'rng_restored': False, 'sampler_rng_restored': False,
        'fresh_max_policy_lag': 0,
    }
