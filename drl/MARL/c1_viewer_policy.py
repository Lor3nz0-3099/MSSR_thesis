"""Viewer-only frozen actor restore. No learner, critic or optimizer creation."""
import hashlib
import json
from pathlib import Path


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inspect_policy_checkpoint(path, config):
    """Validate the native resume envelope without restoring training state."""
    path = Path(path).resolve()
    manifest = json.loads(Path(str(path) + '.resume.json').read_text())
    if manifest.get('schema') != 'mssr.c1.staged_resume.v1':
        raise ValueError('Unsupported MAPPO resume manifest schema')
    if (manifest.get('checkpoint_name') != path.name
            or manifest.get('checkpoint_bytes') != path.stat().st_size
            or manifest.get('checkpoint_sha256') != _sha256(path)):
        raise ValueError('MAPPO checkpoint incomplete or corrupt')
    fingerprint = manifest['fingerprint']
    if fingerprint.get('bc_v4_checkpoint_sha256') != config['expected_bc_sha256']:
        raise ValueError('MAPPO BC-v4 architecture reference differs')
    for key, field in (('reward_config', 'reward_config_sha256'),
                       ('probe_config', 'c1_probe_config_sha256')):
        if fingerprint.get(field) != _sha256(config[key]):
            raise ValueError(f'MAPPO {key} differs from training manifest')
    return dict(policy_kind='mappo', checkpoint=str(path),
                expected_checkpoint_sha256=manifest['checkpoint_sha256'],
                completed_updates=manifest['counters']['completed_updates'],
                policy_version=manifest['counters']['smdp_policy_version'])


def restore_actor_state(actor, bundle):
    """Restore exactly one shared actor; fail closed on unsupported preprocessors."""
    import torch
    if not isinstance(bundle, dict) or any(not isinstance(v, dict) for v in bundle.values()):
        raise ValueError('Expected native MAPPO agent registries')
    policies = [(uid, registry['policy']) for uid, registry in bundle.items() if 'policy' in registry]
    if len(policies) != 1:
        raise ValueError('Expected exactly one shared MAPPO policy registry')
    for registry in bundle.values():
        if any(key in registry for key in ('observation_preprocessor', 'state_preprocessor')):
            raise ValueError('Learned observation/state preprocessing is unsupported by this viewer')
    uid, state = policies[0]
    expected = actor.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError('MAPPO actor state keys differ from complete H1/H2 wrapper')
    for key, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != expected[key].shape
                or value.dtype != expected[key].dtype or not torch.isfinite(value).all()):
            raise ValueError(f'Invalid MAPPO actor tensor: {key}')
    changed = dict(h1=0, h2=0, distribution=0)
    for key, value in state.items():
        if not torch.equal(value.cpu(), expected[key].detach().cpu()):
            group = 'h1' if key.startswith('policy.backbone.') else 'h2' if key.startswith('policy.head.') else 'distribution'
            changed[group] += 1
    actor.load_state_dict(state, strict=True)
    equal = all(torch.equal(value.cpu(), actor.state_dict()[key].detach().cpu()) for key, value in state.items())
    if not equal:
        raise RuntimeError('Loaded actor tensors do not match MAPPO checkpoint')
    return dict(shared_actor_registry=uid, tensor_count=len(state),
                all_checkpoint_tensors_equal=equal, changed_from_bc=changed)


def build_viewer_actor(bridge, config):
    """Use the BC factory for the original architecture; MAPPO restore is LAST.

    The native checkpoint stores the full wrapper, including policy.backbone
    (H1), policy.head (H2), and both learned Gaussian log std parameters. The
    SMORES adapter, frozen normalizers and presence calibration are external,
    identical to the native training bridge. They are not optimizer state.
    """
    import torch
    from drl.MARL.c1_structural_physics_diagnostic import build_frozen_actor
    if config.get('policy_kind', 'bc_v4') == 'bc_v4':
        return build_frozen_actor(bridge, config)
    metadata = inspect_policy_checkpoint(config['checkpoint'], config)
    if metadata['expected_checkpoint_sha256'] != config['expected_checkpoint_sha256']:
        raise ValueError('MAPPO checkpoint changed after preflight')
    base = dict(config, checkpoint=config['bc_checkpoint'])
    actor = build_frozen_actor(bridge, base)
    # Reading the bundle does not instantiate any serialized optimizer/model.
    bundle = torch.load(config['checkpoint'], map_location='cpu', weights_only=True)
    audit = restore_actor_state(actor, bundle)
    actor.to(config['device']).eval().requires_grad_(False)
    actor.checkpoint_path = config['checkpoint']
    actor.checkpoint_sha256 = metadata['expected_checkpoint_sha256']
    actor.checkpoint_step = metadata['completed_updates']
    actor.viewer_restore_audit = dict(metadata, **audit,
        resource_log_std=float(actor.resource_log_std.detach().cpu()),
        structural_log_std=float(actor.structural_log_std.detach().cpu()),
        presence_threshold_logit=actor.presence_threshold_logit,
        observation_normalization='frozen BC-v4 actor-visible repository normalizers; no learned preprocessor',
        bc_checkpoint_role='architecture/identity scaffold; every tensor overwritten by strict MAPPO restore')
    return actor
