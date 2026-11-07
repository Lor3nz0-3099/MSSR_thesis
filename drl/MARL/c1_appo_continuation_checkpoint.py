"""Snapshot APPO continuation artifacts at completed learner updates.

This schema is deliberately separate from the catalog-only staged resume schema.
It saves learner/optimizer state and sampling history, not live physical episodes
or in-flight SMDP options. Resume reconstructs fresh episodes before restoring
process RNG; it cannot reproduce the interrupted physical trajectory bitwise.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch

from drl.MARL.c1_critic_quality_gate import C1CriticQualityGate
from drl.MARL.c1_expert_snapshots import SnapshotBank, current_snapshot_physics
from drl.MARL.c1_longtrain_checkpoint import capture_rng, sha256_file, validate_counters
from drl.MARL.c1_longtrain_resume_artifacts import training_fingerprint
from drl.MARL.c1_longtrain_resume_core import _check_empty_rollout
from drl.MARL.c1_snapshot_sampling import DEFAULT_WEIGHTS, STAGES, SnapshotSampler

SCHEMA = 'mssr.c1.appo_snapshot_continuation.v1'
SAMPLER_SCHEMA = 'mssr.c1.snapshot_sampler.v1'
PHYSICS_RESUME = 'new_episode_not_bitwise_continuation'
IDENTITY_KEYS = ('schema', 'seed', 'split', 'domains_by_env', 'stage_weights', 'bank_sha256')


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _owned(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else value


def continuation_fingerprint(args):
    """Scientific settings plus snapshot bank, physics, and continuation code."""
    result = training_fingerprint(args)
    weights = getattr(args, 'snapshot_stage_weights', None)
    if isinstance(weights, str):
        weights = json.loads(weights)
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if (set(weights) != set(STAGES) or any(type(v) not in (int, float) or not math.isfinite(v)
            or v < 0 for v in weights.values()) or sum(weights.values()) <= 0):
        raise ValueError('Invalid snapshot stage weights')
    root = Path(args.repository_root).resolve()
    sources = ('c1_appo_continuation_checkpoint.py', 'c1_expert_snapshots.py',
               'c1_snapshot_sampling.py', 'c1_snapshot_runtime.py')
    result.update(continuation_schema=SCHEMA,
        snapshot_bank_sha256=sha256_file(args.snapshot_bank),
        snapshot_stage_weights=weights, snapshot_physics=current_snapshot_physics(),
        continuation_source_sha256={f'drl/MARL/{name}': sha256_file(root/'drl/MARL'/name)
                                    for name in sources})
    return _owned(result)


def _validated_counters(counters):
    result = validate_counters(counters)
    update, version = result['completed_updates'], result['smdp_policy_version']
    if update < 1:
        raise ValueError('Continuation requires a completed update')
    if version > update or (update <= 4 and version != 0):
        raise ValueError('Inconsistent completed update/policy version')
    return result


def _gate(state, updates):
    try:
        gate = C1CriticQualityGate.from_state_dict(state)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Invalid critic gate state') from error
    if (gate.last_completed != updates or gate.good_streak > updates
            or type(gate.last_ok) is not bool or not isinstance(gate.last_reason, str)):
        raise ValueError('Critic gate/update mismatch')
    return gate


def _provenance(value):
    if not isinstance(value, dict) or any(not _digest(value.get(key)) for key in (
            'historical_checkpoint_sha256', 'previous_checkpoint_sha256')):
        raise ValueError('Invalid continuation provenance checkpoint hashes')
    for key in ('discarded_pending_options', 'discarded_physical_episodes'):
        if key in value and (type(value[key]) is not int or value[key] < 0):
            raise ValueError('Invalid continuation provenance discard counts')
    return _owned(value)


def _validate_rng(state):
    try:
        if not isinstance(state, dict) or set(state) != {'python', 'numpy', 'torch_cpu', 'torch_cuda'}:
            raise ValueError('incomplete state')
        random.Random().setstate(_tuples(state['python']))
        n = state['numpy']
        np.random.RandomState().set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
        for row in [state['torch_cpu'], *state['torch_cuda']]:
            if not isinstance(row, list) or not row or any(type(v) is not int or not 0 <= v <= 255 for v in row):
                raise ValueError('invalid Torch generator bytes')
        torch.Generator(device='cpu').set_state(torch.tensor(state['torch_cpu'], dtype=torch.uint8))
    except (KeyError, IndexError, TypeError, ValueError, RuntimeError) as error:
        raise ValueError('Invalid global RNG state') from error


def _validated_rows(saved, bank):
    """Validate all state before assigning anything into a live sampler."""
    if not isinstance(saved, dict) or saved.get('schema') != SAMPLER_SCHEMA:
        raise ValueError('Invalid snapshot sampler schema')
    if saved.get('split') != 'train' or type(saved.get('seed')) is not int:
        raise ValueError('Snapshot continuation requires TRAIN sampler identity')
    if not _digest(saved.get('bank_sha256')) or saved['bank_sha256'] != bank.sha256:
        raise ValueError('Snapshot bank identity changed')
    domains = saved.get('domains_by_env')
    rows = saved.get('environments')
    if not isinstance(domains, list) or not domains or not isinstance(rows, list) or len(rows) != len(domains):
        raise ValueError('Snapshot sampler environment count changed')
    weights = saved.get('stage_weights')
    if (not isinstance(weights, dict) or set(weights) != set(STAGES)
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in weights.values())
            or sum(weights.values()) <= 0):
        raise ValueError('Invalid snapshot stage weights')
    rngs, generations, currents, counts = [], [], [], []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or type(row.get('env_index')) is not int or row['env_index'] != index:
            raise ValueError('Invalid snapshot sampler environment index')
        generation, current, stage_counts = row.get('generation'), row.get('current'), row.get('stage_counts')
        if type(generation) is not int or generation < 0:
            raise ValueError('Invalid snapshot sampler generation')
        if (not isinstance(stage_counts, dict) or not set(stage_counts) <= set(STAGES)
                or any(type(v) is not int or v < 0 for v in stage_counts.values())
                or sum(stage_counts.values()) != generation):
            raise ValueError('Invalid snapshot sampler stage counts')
        if generation == 0:
            if current is not None:
                raise ValueError('Snapshot bank current record without a generation')
        elif (not isinstance(current, dict) or current not in bank.records
                or current.get('identity', {}).get('domain') != domains[index]
                or current.get('identity', {}).get('split') != saved['split']
                or not stage_counts.get(current.get('stage'), 0)):
            raise ValueError('Snapshot bank current record is outside selected environment bank')
        try:
            rng = random.Random()
            rng.setstate(_tuples(row['rng_state']))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError('Invalid snapshot sampler RNG state') from error
        rngs.append(rng); generations.append(generation)
        currents.append(deepcopy(current)); counts.append(Counter(stage_counts))
    return rngs, generations, currents, counts


def restore_snapshot_sampler(sampler, savedmanifest):
    """Restore full per-row RNG/history transactionally, before fresh resets."""
    saved = _owned(savedmanifest)
    fresh = sampler.manifest()
    for key in IDENTITY_KEYS:
        if saved.get(key) != fresh.get(key):
            raise ValueError(f'Snapshot sampler identity changed: {key}')
    rngs, generations, currents, counts = _validated_rows(saved, sampler.bank)
    # Manifest roundtrip is checked on a temporary object before touching sampler.
    probe = SnapshotSampler(sampler.bank, sampler.domains, sampler.seed,
                            weights=sampler.weights, split=sampler.split)
    probe.rngs, probe.generations, probe.current, probe.counts = rngs, generations, currents, counts
    if probe.manifest() != saved:
        raise ValueError('Snapshot sampler state failed exact manifest roundtrip')
    sampler.rngs[:], sampler.generations[:], sampler.current[:], sampler.counts[:] = rngs, generations, currents, counts
    return sampler.manifest()


def _registry(payload):
    if not isinstance(payload, dict) or not payload:
        raise ValueError('Missing full learner checkpoint registry')
    if any(not isinstance(uid, str) or not isinstance(modules, dict)
           or any(not isinstance(name, str) or not isinstance(state, dict) for name, state in modules.items())
           for uid, modules in payload.items()):
        raise ValueError('Invalid full learner checkpoint registry')
    return {uid: sorted(modules) for uid, modules in payload.items()}


def _read_payload(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except Exception as error:
        raise ValueError('Unreadable learner checkpoint registry') from error


def _check_update_boundary(learner):
    # In-flight options continue in the current run. Only a future resume discards
    # them, so do not call the resume helper (which rejects pending options) here.
    if int(getattr(learner, '_rollout', 0)) != 0:
        raise RuntimeError('Continuation checkpoint requires a completed PPO rollout')
    for memory in getattr(learner, 'memories', {}).values():
        if len(memory):
            raise RuntimeError('Continuation checkpoint requires empty rollout memories')
    structured = getattr(learner, '_structured_rollout_sidecar', None)
    altruistic = getattr(learner, '_altruistic_rollout_sidecar', None)
    if ((structured is not None and len(structured))
            or (altruistic is not None and len(altruistic.flatten()))):
        raise RuntimeError('Continuation checkpoint requires empty rollout sidecars')


def save_continuation_checkpoint(args, output, learner, sampler, *, counters,
                                 critic_gate_state, provenance):
    """Exclusively publish full learner checkpoint, then atomic JSON metadata."""
    if not (getattr(args, 'train', False) and getattr(args, 'c1_staged_training', False)):
        raise ValueError('Explicit staged C1 training is required')
    counters = _validated_counters(counters)
    _gate(critic_gate_state, counters['completed_updates'])
    provenance = _provenance(provenance)
    _check_update_boundary(learner)
    fingerprint = continuation_fingerprint(args)
    saved = sampler.manifest()
    _validated_rows(saved, sampler.bank)
    if (len(saved['environments']) != args.num_envs
            or saved['bank_sha256'] != fingerprint['snapshot_bank_sha256']
            or saved['stage_weights'] != fingerprint['snapshot_stage_weights']):
        raise ValueError('Snapshot sampler/bank configuration differs from fingerprint')
    path = Path(output).resolve()/(
        f"c1_continuation_u{counters['completed_updates']:06d}_v{counters['smdp_policy_version']:06d}.pt")
    metadata = Path(str(path)+'.continuation.json')
    if path.exists() or metadata.exists():
        raise FileExistsError(f'Continuation checkpoint already exists: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary = path.with_name(path.name+'.'+token+'.incomplete')
    temporary_metadata = metadata.with_name(metadata.name+'.'+token+'.incomplete')
    try:
        # Save into an exclusively claimed file; never allow an old artifact to
        # be replaced, including a concurrent publisher using the same update.
        with temporary.open('xb'):
            pass
        learner.save(str(temporary))
        if not temporary.stat().st_size:
            raise RuntimeError('Learner checkpoint was not created')
        payload = _read_payload(temporary)
        registry = _registry(payload)
        expected_registry = {uid: sorted(modules) for uid, modules in learner.checkpoint_modules.items()}
        if registry != expected_registry:
            raise ValueError('Saved checkpoint registry differs from learner registry')
        for uid, modules in learner.checkpoint_modules.items():
            for name, module in modules.items():
                if isinstance(module, torch.optim.Optimizer):
                    # An optimizer registry entry without its complete state is
                    # never a resumable artifact, even if model weights exist.
                    value = payload[uid][name]
                    if set(value) != {'state', 'param_groups'}:
                        raise ValueError('Incomplete optimizer checkpoint registry state')
        pending = getattr(learner, '_smdp_pending', None)
        pending_count = sum(v is not None for v in pending) if pending is not None else 0
        state = dict(schema=SCHEMA, checkpoint_name=path.name,
            checkpoint_sha256=sha256_file(temporary), checkpoint_bytes=temporary.stat().st_size,
            checkpoint_registry=registry, counters=counters, fingerprint=fingerprint,
            sampler_manifest=saved, snapshot_bank_path=str(Path(args.snapshot_bank).resolve()),
            critic_gate_state=_owned(critic_gate_state), provenance=provenance,
            rng=capture_rng(), physics_resume=PHYSICS_RESUME,
            boundary_contract=dict(completed_update=True, rollout_serialized=False,
                physical_episode_serialized=False, pending_options_serialized=False,
                pending_options_to_discard_on_resume=pending_count,
                physical_episodes_to_discard_on_resume=len(saved['environments'])))
        _validate_rng(state['rng'])
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        with temporary_metadata.open('x', encoding='utf-8') as stream:
            json.dump(state, stream, sort_keys=True, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        # link() is atomic and fails if destination exists (replace() does not).
        os.link(temporary, path)
        os.link(temporary_metadata, metadata)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        read_continuation_manifest(path, expected_fingerprint=fingerprint)
        return path
    finally:
        temporary.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)


def read_continuation_manifest(path, expected_fingerprint=None):
    """Validate integrity and semantic state before touching the learner."""
    path = Path(path).resolve()
    state = json.loads(Path(str(path)+'.continuation.json').read_text(encoding='utf-8'))
    _owned(state)
    if state.get('schema') != SCHEMA:
        raise ValueError('Unsupported continuation checkpoint schema')
    if state.get('checkpoint_name') != path.name:
        raise ValueError('Continuation checkpoint filename mismatch')
    fingerprint = state.get('fingerprint')
    if not isinstance(fingerprint, dict) or fingerprint.get('continuation_schema') != SCHEMA:
        raise ValueError('Missing continuation fingerprint')
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise ValueError('Configuration fingerprint changed: unsafe continuation')
    counters = _validated_counters(state.get('counters', {}))
    _gate(state.get('critic_gate_state'), counters['completed_updates'])
    _provenance(state.get('provenance'))
    _validate_rng(state.get('rng'))
    boundary = state.get('boundary_contract', {})
    if (state.get('physics_resume') != PHYSICS_RESUME
            or boundary.get('completed_update') is not True
            or any(boundary.get(key) is not False for key in ('rollout_serialized',
                'physical_episode_serialized', 'pending_options_serialized'))
            or any(type(boundary.get(key)) is not int or boundary[key] < 0 for key in (
                'pending_options_to_discard_on_resume', 'physical_episodes_to_discard_on_resume'))):
        raise ValueError('Invalid fresh-episode continuation boundary contract')
    saved = state.get('sampler_manifest', {})
    if (saved.get('bank_sha256') != fingerprint.get('snapshot_bank_sha256')
            or saved.get('stage_weights') != fingerprint.get('snapshot_stage_weights')):
        raise ValueError('Snapshot sampler bank/weights fingerprint mismatch')
    bank_path = state.get('snapshot_bank_path')
    if not isinstance(bank_path, str) or not Path(bank_path).is_file():
        raise ValueError('Snapshot bank manifest unavailable')
    bank = SnapshotBank(bank_path, 'train')
    _validated_rows(saved, bank)
    if (len(saved['environments']) != fingerprint.get('num_envs')
            or boundary['physical_episodes_to_discard_on_resume'] != len(saved['environments'])):
        raise ValueError('Snapshot sampler environment boundary mismatch')
    if (type(state.get('checkpoint_bytes')) is not int
            or state['checkpoint_bytes'] != path.stat().st_size
            or state.get('checkpoint_sha256') != sha256_file(path)):
        raise ValueError('Continuation checkpoint incomplete or corrupted hash')
    if _registry(_read_payload(path)) != state.get('checkpoint_registry'):
        raise ValueError('Continuation checkpoint registry mismatch')
    return state


def restore_continuation_learner(learner, path, *, fingerprint, fresh_sampler_manifest):
    """Restore registered models/optimizers and counters, with fresh physics next."""
    state = read_continuation_manifest(path, expected_fingerprint=fingerprint)
    saved = state['sampler_manifest']
    for key in IDENTITY_KEYS:
        if fresh_sampler_manifest.get(key) != saved.get(key):
            raise ValueError(f'Snapshot sampler identity changed: {key}')
    if len(fresh_sampler_manifest.get('environments', [])) != len(saved['environments']):
        raise ValueError('Snapshot sampler environment count changed')
    _check_empty_rollout(learner)
    from drl.MARL.c1_historical_appo import restore_full_state
    restore_full_state(learner, path, expected_checkpoint_sha256=state['checkpoint_sha256'])
    _check_empty_rollout(learner)
    counters = state['counters']
    learner._smdp_policy_version = counters['smdp_policy_version']
    learner._smdp_transition_count = counters['smdp_transition_count']
    learner._smdp_event_step_count = counters['smdp_event_step_count']
    learner._smdp_event_rows_last_interaction = 0
    learner._smdp_pending = None
    learner._smdp_max_policy_lag = 0
    learner._rollout = 0
    gate = _gate(state['critic_gate_state'], counters['completed_updates'])
    next_step = gate.step_for_update(counters['completed_updates']+1)
    return dict(counters=counters, critic_gate_state=gate.state_dict(), provenance=state['provenance'],
        checkpoint_path=str(Path(path).resolve()), checkpoint_sha256=state['checkpoint_sha256'],
        next_update=counters['completed_updates']+1, next_phase=next_step.phase,
        next_actor_lr=next_step.actor_lr, next_critic_lr=next_step.critic_lr,
        rng_to_restore_after_physics_reset=state['rng'],
        sampler_to_reconcile_with_fresh_physics=saved, physical_episode_reused=False)
