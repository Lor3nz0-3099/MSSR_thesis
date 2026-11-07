"""Offline restore/provenance regressions. Never launch Isaac or optimize."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / 'logs/isaaclab_c1_native/redesign/c1_5B_long5h_0XmSGcdL'
CHECKPOINT = HISTORY / 'checkpoint_5048.pt'


@pytest.fixture(scope='module')
def real_learner():
    from drl.IL.bc_v1.action_set_head import ActionSetPolicy
    from drl.IL.bc_v1.sidecar_build import load_sources
    from drl.MARL.bc_v4_actor import BCV4SharedActorWrapper
    from drl.MARL.global_critic import GlobalCritic
    from drl.MARL.owner_critic import build_repository_altruistic_critic_pair
    from drl.MARL.mappo import MSSRMAPPO
    kwargs = load_sources()['kwargs']
    spec, schema = kwargs['spec'], kwargs['feature_schema']
    policy = ActionSetPolicy(spec, schema,
        json.loads((ROOT / 'configs/bc_v1/gnn_v1.json').read_text()),
        json.loads((ROOT / 'configs/bc_v1/action_set_head_v1.json').read_text()))
    actor = BCV4SharedActorWrapper(policy=policy, device='cpu')
    critic = GlobalCritic.from_repository_config(spec=spec, feature_schema=schema,
                                                 repo_root=ROOT, device='cpu')
    owners = tuple(f'smores_{n:02}' for n in range(1, 9))
    cfg = dict(learning_epochs=2, mini_batches=8, rollouts=32,
               learning_rate=(1e-5, 3e-4), discount_factor=.999,
               gae_lambda=.95, entropy_loss_scale=.001, learning_rate_scheduler=None)
    for key in ('learning_rate_scheduler_kwargs', 'observation_preprocessor_kwargs',
                'state_preprocessor_kwargs', 'value_preprocessor_kwargs'):
        cfg[key] = {owner: {} for owner in owners}
    cfg['experiment'] = dict(write_interval=0, checkpoint_interval=0)
    learner = MSSRMAPPO(possible_agents=owners,
        models=MSSRMAPPO.build_shared_model_registry(possible_agents=owners, actor=actor, critic=critic),
        device='cpu', cfg=cfg, deterministic_update_mode=True, skip_expert_policy_inference=True)
    pair = build_repository_altruistic_critic_pair(spec=spec, feature_schema=schema,
                                                   repo_root=ROOT, device='cpu')
    learner.configure_fixed_alpha_appo(owner_critic=pair.owner, mean_field_critic=pair.mean_field, alpha=.5)
    learner._rollout = 0
    return learner


def test_real_historical_provenance_replays_all_updates():
    from drl.MARL.c1_historical_appo import inspect_historical_checkpoint
    context = inspect_historical_checkpoint(CHECKPOINT, ROOT)
    assert context['counters'] == dict(completed_updates=109, vector_steps=4737,
        smdp_transition_count=5048, smdp_event_step_count=3488, smdp_policy_version=99)
    gate = context['critic_gate_state']
    assert (gate['last_completed'], gate['first_actor_update'], gate['good_streak'],
            gate['last_ok'], gate['last_reason']) == (109, 11, 3, True, 'actor_unlock_scheduled')
    assert context['provenance']['history']['replayed_updates'] == 109
    assert context['provenance']['history']['optimizer_step_totals']['actor_optimizer'] == 1580
    assert context['provenance']['history']['optimizer_step_totals']['owner_critic_optimizer'] == 1740
    assert context['provenance']['gate_evidence']['last_reason'] == 'replayed_not_recorded'
    json.dumps(context, allow_nan=False)


def test_real_full_restore_uses_actual_loader_safely_and_compares_every_module(real_learner):
    from drl.MARL.c1_historical_appo import restore_full_state
    original_load = torch.load
    calls = []
    def tracked_load(*args, **kwargs):
        calls.append(kwargs.get('weights_only'))
        return original_load(*args, **kwargs)
    with patch.object(torch, 'load', tracked_load), patch.object(real_learner, 'load', wraps=real_learner.load) as loader:
        audit = restore_full_state(real_learner, CHECKPOINT)
    assert loader.call_count == 1
    assert calls and all(value is True for value in calls)
    assert audit['all_equal'] is True
    assert len(audit['models']) == 4
    assert [audit['optimizers'][f'smores_01:{key}']['state_entries'] for key in (
        'actor_optimizer', 'owner_critic_optimizer', 'mean_field_critic_optimizer', 'critic_optimizer')] == [245, 180, 180, 0]
    assert audit['optimizers']['smores_01:owner_critic_optimizer']['step_values'] == [1740]
    assert len(audit['initial_models_sha256']) == 64


def test_real_boundary_restores_version_and_preserves_loaded_lrs(real_learner):
    from drl.MARL.c1_historical_appo import inspect_historical_checkpoint, restore_historical_learner
    context = inspect_historical_checkpoint(CHECKPOINT, ROOT)
    audit = restore_historical_learner(real_learner, context)
    assert audit['next_update'] == 110 and audit['next_phase'] == 'joint_refinement'
    assert real_learner._smdp_policy_version == 99
    assert real_learner._smdp_transition_count == 5048
    assert real_learner._smdp_event_step_count == 3488
    assert real_learner._smdp_pending is None
    assert real_learner._smdp_max_policy_lag == 0
    assert real_learner.actor_optimizer.param_groups[0]['lr'] == 1.5e-5
    assert real_learner.owner_critic_optimizer.param_groups[0]['lr'] == 2e-4
    assert audit['full_state']['all_equal']


def test_history_tampering_or_missing_rows_is_rejected(tmp_path):
    from drl.MARL.c1_historical_appo import inspect_historical_checkpoint
    import shutil
    target = tmp_path / HISTORY.name
    target.mkdir()
    for name in ('checkpoint_5048.pt', 'updates.jsonl', 'result.json', 'run_arguments.json', 'training_telemetry.json'):
        shutil.copyfile(HISTORY / name, target / name)
    (target / 'updates.jsonl').write_text('\n'.join((target / 'updates.jsonl').read_text().splitlines()[1:]) + '\n')
    with pytest.raises(ValueError, match='hash|historical|integrity'):
        inspect_historical_checkpoint(target / CHECKPOINT.name, ROOT)


def test_historical_context_tampering_is_rejected_before_load(real_learner):
    from drl.MARL.c1_historical_appo import inspect_historical_checkpoint, restore_historical_learner
    context = inspect_historical_checkpoint(CHECKPOINT, ROOT)
    context['counters']['smdp_policy_version'] = 0
    with patch.object(real_learner, 'load') as loader, pytest.raises(ValueError, match='context|counter'):
        restore_historical_learner(real_learner, context)
    assert not loader.called


def test_nonempty_rollout_rejected_before_load(real_learner):
    from drl.MARL.c1_historical_appo import inspect_historical_checkpoint, restore_historical_learner
    context = inspect_historical_checkpoint(CHECKPOINT, ROOT)
    real_learner._rollout = 1
    try:
        with patch.object(real_learner, 'load') as loader, pytest.raises(RuntimeError, match='empty'):
            restore_historical_learner(real_learner, context)
        assert not loader.called
    finally:
        real_learner._rollout = 0


@pytest.mark.parametrize('corruption', ['missing_registry', 'extra_registry', 'adam_shape', 'adam_nonfinite',
    'model_nonfinite', 'param_groups', 'unexpected_adam_id', 'model_shape', 'optimizer_hyperparameter'])
def test_full_restore_rejects_incomplete_or_corrupt_checkpoint_before_load(real_learner, tmp_path, corruption):
    from drl.MARL.c1_historical_appo import restore_full_state
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    registry = state['smores_01']
    adam = registry['actor_optimizer']
    if corruption == 'missing_registry': del registry['owner_critic_optimizer']
    elif corruption == 'extra_registry': registry['hidden_model'] = {}
    elif corruption == 'adam_shape': next(iter(adam['state'].values()))['exp_avg'] = torch.zeros(20)
    elif corruption == 'adam_nonfinite': next(iter(adam['state'].values()))['step'].fill_(float('nan'))
    elif corruption == 'model_nonfinite': registry['policy']['resource_log_std'].fill_(float('nan'))
    elif corruption == 'param_groups': adam['param_groups'][0]['params'].pop()
    elif corruption == 'unexpected_adam_id': adam['state'][999999] = deepcopy(next(iter(adam['state'].values())))
    elif corruption == 'model_shape': registry['policy']['resource_log_std'] = torch.zeros(3)
    elif corruption == 'optimizer_hyperparameter': adam['param_groups'][0]['betas'] = (.8, .99)
    path = tmp_path / 'corrupt.pt'
    torch.save(state, path)
    with patch.object(real_learner, 'load') as loader, pytest.raises((ValueError, RuntimeError)):
        restore_full_state(real_learner, path)
    assert not loader.called


def test_loader_silent_skip_is_detected(real_learner):
    from drl.MARL.c1_historical_appo import restore_full_state
    with torch.no_grad(): real_learner.shared_actor.resource_log_std.add_(1)
    with patch.object(real_learner, 'load'), pytest.raises(ValueError, match='equal|mismatch|restor'):
        restore_full_state(real_learner, CHECKPOINT)


def test_hash_mismatch_is_rejected_before_load(real_learner):
    from drl.MARL.c1_historical_appo import restore_full_state
    with patch.object(real_learner, 'load') as loader, pytest.raises(ValueError, match='hash|SHA'):
        restore_full_state(real_learner, CHECKPOINT, expected_checkpoint_sha256='0' * 64)
    assert not loader.called
