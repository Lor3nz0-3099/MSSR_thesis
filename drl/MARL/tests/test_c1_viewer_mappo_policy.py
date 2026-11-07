"""Real checkpoint actor-only restoration; no Isaac Sim imports or learner."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from drl.MARL.c1_gui_rc_car_180s import ROOT, parse_args, preflight

MAPPO = ROOT / 'logs/isaaclab_c1_native/redesign/c1_guided_resume_u13_20261105_113402/c1_resume_u000078_v000067.pt'
PROBE = ROOT / 'configs/marl/c1_assembly_probe_guided_v1.json'


def config():
    return preflight(parse_args(['--preflight', '--device', 'cpu', '--policy-checkpoint', str(MAPPO),
                                '--c1-probe-config', str(PROBE), '--episode-seconds', '180']))


@pytest.fixture(scope='module')
def restored():
    from drl.MARL.c1_runner import build_observation_bridge
    from drl.MARL.c1_viewer_policy import build_viewer_actor
    cfg = config()
    def forbidden(*args, **kwargs):
        pytest.fail('Actor-only restore invoked training machinery')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.optim.Optimizer, '__init__', forbidden)
        patch.setattr(torch.Tensor, 'backward', forbidden)
        patch.setattr(torch.autograd, 'backward', forbidden)
        actor = build_viewer_actor(build_observation_bridge(ROOT), cfg)
    return actor, cfg


def test_real_mappo_restores_every_tensor_and_trained_heads(restored):
    actor, cfg = restored
    state = torch.load(MAPPO, map_location='cpu', weights_only=True)['smores_01']['policy']
    assert set(state) == set(actor.state_dict())
    assert all(torch.equal(v, actor.state_dict()[k]) for k, v in state.items())
    audit = actor.viewer_restore_audit
    assert audit['completed_updates'] == 78 and audit['policy_version'] == 67
    assert audit['all_checkpoint_tensors_equal']
    assert audit['changed_from_bc']['h1'] > 0
    assert audit['changed_from_bc']['h2'] > 0
    assert audit['changed_from_bc']['distribution'] == 2
    assert not actor.training and all(not p.requires_grad and p.grad is None for p in actor.parameters())
    assert actor.presence_threshold_logit == 4.5
    assert cfg['policy_kind'] == 'mappo' and cfg['checkpoint'] == str(MAPPO)
    assert not any(name == 'isaaclab' or name.startswith('isaaclab.') for name in sys.modules)


def test_bc_default_remains_original():
    from drl.MARL.c1_runner import build_observation_bridge
    from drl.MARL.c1_viewer_policy import build_viewer_actor
    cfg = preflight(parse_args(['--preflight', '--device', 'cpu']))
    actor = build_viewer_actor(build_observation_bridge(ROOT), cfg)
    original = torch.load(cfg['checkpoint'], map_location='cpu', weights_only=False)['model']
    assert all(torch.equal(v, actor.policy.state_dict()[k]) for k, v in original.items())
    assert actor.resource_log_std.item() == -2.
    assert cfg['policy_kind'] == 'bc_v4'


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'preprocessor', 'duplicate', 'shape', 'dtype', 'nan'])
def test_reject_incomplete_or_ambiguous_policy(restored, mutation):
    from drl.MARL.c1_viewer_policy import restore_actor_state
    actor, _ = restored
    state = dict(actor.state_dict())
    bundle = {'smores_01': {'policy': state}, 'smores_02': {}}
    if mutation == 'missing':
        state.pop('structural_log_std')
    elif mutation == 'extra':
        state['unknown_adapter.weight'] = torch.zeros(1)
    elif mutation == 'preprocessor':
        bundle['smores_01']['observation_preprocessor'] = {'mean': torch.ones(1)}
    elif mutation == 'duplicate':
        bundle['smores_02']['policy'] = state
    elif mutation == 'shape':
        state['resource_log_std'] = torch.ones(2)
    elif mutation == 'dtype':
        state['resource_log_std'] = state['resource_log_std'].double()
    else:
        state['resource_log_std'] = torch.tensor(float('nan'))
    with pytest.raises((ValueError, RuntimeError)):
        restore_actor_state(actor, bundle)


def test_manifest_integrity_and_probe_mismatch(tmp_path):
    from drl.MARL.c1_viewer_policy import inspect_policy_checkpoint
    cfg = config()
    with pytest.raises(ValueError, match='probe'):
        inspect_policy_checkpoint(MAPPO, dict(cfg, probe_config=str(ROOT/'configs/marl/c1_assembly_probe.json')))
    path = tmp_path / MAPPO.name
    path.write_bytes(b'corrupt')
    manifest = Path(str(MAPPO) + '.resume.json').read_text()
    Path(str(path) + '.resume.json').write_text(manifest)
    with pytest.raises(ValueError, match='corrupt'):
        inspect_policy_checkpoint(path, cfg)


def test_real_mappo_forward_and_stochastic_viewer_sampling_without_sim(restored):
    from drl.IL.bc_v1.bc_v1_contract import TaskContext
    from drl.IL.tests.bc_v1.test_live_observation import snapshot
    from drl.MARL.c1_runner import build_observation_bridge
    from drl.MARL.c1_gui_rc_car_180s import sample_contexts
    from drl.MARL.c1_feasibility import NativeAvailability, CandidateAvailability
    from drl.MARL.tests.test_c1_tier1_policy import equal
    actor, _ = restored
    prepared = build_observation_bridge(ROOT).prepare_snapshot(
        snapshot(n=8, rc=True), task_context=TaskContext('navigation'), include_privileged=False)
    count = len(prepared.routing_snapshot['resource_candidates']) + len(prepared.routing_snapshot['structural_bindings'])
    availability = NativeAvailability((False,) * count, tuple(CandidateAvailability('available') for _ in range(count)))
    context = SimpleNamespace(actor_graph=prepared.actor_graph, routing_snapshot=prepared.routing_snapshot,
                              hard_mask=torch.zeros(count, dtype=torch.bool), availability_snapshot=availability)
    owners = tuple(prepared.routing_snapshot['module_refs'])
    before = {key: value.clone() for key, value in actor.state_dict().items()}
    torch.manual_seed(3103)
    reference = sample_contexts(actor, [context], owners, accelerated=False)[0][0].action
    rng = torch.random.get_rng_state()
    torch.manual_seed(3103)
    accelerated = sample_contexts(actor, [context], owners, accelerated=True)[0][0].action
    equal(reference, accelerated)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(value, actor.state_dict()[key]) for key, value in before.items())
    assert all(p.grad is None for p in actor.parameters())


def test_manual_command_settings_match_original_run():
    import json
    args = parse_args(['--task', 'navigation', '--steps', '5400', '--episode-seconds', '180',
                      '--device', 'cuda:0', '--global-seed', '3103', '--runtime-mode', 'accelerated',
                      '--structural-scheduler', 'event', '--status-seconds', '5', '--max-wall-seconds', '14400',
                      '--policy-checkpoint', str(MAPPO), '--c1-probe-config', str(PROBE)])
    cfg = preflight(args)
    previous = json.loads((ROOT/'logs/isaaclab_c1_native/redesign/rc_car_guided_180s_20261105_024045/run_config.json').read_text())
    for key in ('task', 'initial_train_seed', 'global_seed', 'initialization_mode', 'num_envs', 'physics_hz',
                'decision_hz', 'steps', 'episode_seconds', 'structural_scheduler', 'structural_fallback_hz',
                'sampling', 'runtime_mode', 'reward_config', 'probe_config', 'headless', 'launcher_options'):
        assert cfg[key] == previous[key], key
