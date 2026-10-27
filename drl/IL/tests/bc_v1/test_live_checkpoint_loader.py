"""CPU-only checkpoint admission. Real artifacts are read, never copied/saved."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from drl.IL.bc_v1 import live_policy as live
from drl.IL.bc_v1.model import BCGNN, load_model_config
from drl.IL.bc_v1.action_heads import load_action_head_config
from drl.IL.bc_v1.dataset import _hash
from drl.IL.bc_v1.training import CHECKPOINT_SCHEMA
from drl.IL.tests.bc_v1.test_live_observation import settings, builder, snapshot, context
from drl.IL.bc_v1.live_controller import PhysicalSnapshot

ROOT = Path(__file__).resolve().parents[4]
FREEZE = ROOT/'artifacts/bc_v1/frozen_runs/20261020T104018_733358Z_k2_no_structural_v1'
CASES = [
    (ROOT/'artifacts/bc_v1/checkpoints/k1_baseline_v1/best_checkpoint.pt',
     '35737c025e7fa53492cb12e59ccf4c4f3f6210a6d6f0882c245bf1847975d06d', 5, 6120, 'bc_v1.legacy_v1', False),
    (FREEZE/'best_checkpoint.pt', '091c65918370967269e0a88e69914f70e4df16855141f3c0e2dc124ce83a38d6',
     13, 15912, 'bc_v1.no_structural_v2', True),
    (FREEZE/'last_checkpoint.pt', '3a4d5b6038e06d415663994ceef780b54c464af2c0d9e28275acc2220b920182',
     21, 25704, 'bc_v1.no_structural_v2', True),
]


@pytest.fixture(autouse=True, scope='module')
def cpu_threads():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('path,digest,epoch,step,profile,has_none', CASES)
@pytest.mark.parametrize('pin_sha', [False, True])
def test_official_checkpoint_loads_strict_on_cpu(path, digest, epoch, step, profile, has_none, pin_sha):
    if not path.is_file(): pytest.skip('External official checkpoint unavailable')
    policy = live.load_frozen_policy(ROOT, path, builder(), device='cpu',
        expected_checkpoint_sha256=digest if pin_sha else None)
    meta = policy.checkpoint_provenance
    assert meta['checkpoint_path'] == str(path.resolve())
    assert meta['checkpoint_sha256'] == digest
    assert meta['checkpoint_epoch'] == epoch and meta['checkpoint_global_step'] == step
    assert meta['checkpoint_profile'] == profile and meta['checkpoint_schema'] == CHECKPOINT_SCHEMA
    assert meta['state_dict_strict'] is True
    assert bool(policy.model.action_heads.no_structural_scorer is not None) == has_none
    original = torch.load(path, map_location='cpu', weights_only=False)
    assert policy.model.state_dict().keys() == original['model'].keys()
    assert all(torch.equal(value, original['model'][name]) for name, value in policy.model.state_dict().items())
    assert all(p.device.type == 'cpu' and not p.requires_grad for p in policy.model.parameters())
    assert not policy.model.training


@pytest.fixture(scope='module')
def synthetic_payload():
    # Independent weights, not a copy or re-save of any real checkpoint.
    kw = settings(); cfg = ROOT/'configs/bc_v1'
    h1 = load_model_config(cfg/'gnn_v1.json')
    h2 = load_action_head_config(cfg/'action_heads_no_structural_v2.json')
    model = BCGNN(kw['spec'], kw['feature_schema'], h1, action_head_config=h2)
    from drl.IL.bc_v1.bc_v1_contract import robot_spec_fingerprint
    from drl.IL.bc_v1.feature_schema import feature_schema_fingerprint
    from drl.IL.bc_v1.normalization import normalizer_fingerprint
    config = {'schema_version': 'synthetic.future_training.v1', 'seed': 19}
    identity = {'robot_spec_fingerprint': robot_spec_fingerprint(kw['spec']),
        'feature_schema_fingerprint': feature_schema_fingerprint(kw['feature_schema']),
        'layout_fingerprint': kw['tensor_layout']['fingerprint'],
        'normalizer_sha': hashlib.sha256((ROOT/'artifacts/bc_v1/normalization/f2_v1_optimized/normalizers.json').read_bytes()).hexdigest(),
        'f2_fingerprint': normalizer_fingerprint(kw['normalizers']),
        'h1_config_fingerprint': _hash(h1), 'h2_config_fingerprint': _hash(h2),
        'training_config_fingerprint': _hash(config),
        'structural_decision_schema': 'mssr.bc_v1.structural_decisions.v2'}
    return {'schema_version': CHECKPOINT_SCHEMA, 'identity': identity, 'config': config,
            'summary': {'epoch': 37}, 'step': 999, 'model': model.state_dict()}


def save_fixture(tmp_path, payload):
    path = tmp_path/'synthetic.pt'; torch.save(payload, path)
    return path


def test_compatible_contract_is_not_an_epoch_path_or_sha_allowlist(tmp_path, synthetic_payload):
    path = save_fixture(tmp_path, synthetic_payload)
    policy = live.load_frozen_policy(ROOT, path, builder(), device='cpu')
    assert policy.checkpoint_provenance['checkpoint_epoch'] == 37
    assert policy.checkpoint_provenance['checkpoint_global_step'] == 999
    assert policy.checkpoint_provenance['checkpoint_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('damage', ['missing', 'unexpected', 'shape', 'dtype', 'nonfinite'])
def test_invalid_state_dict_fails_closed(tmp_path, synthetic_payload, damage):
    payload = copy.deepcopy(synthetic_payload); state = payload['model']; key = next(iter(state))
    if damage == 'missing': state.pop(key)
    elif damage == 'unexpected': state['unexpected.weight'] = torch.ones(1)
    elif damage == 'shape': state[key] = torch.ones(1)
    elif damage == 'dtype': state[key] = state[key].double()
    else: state[key].flatten()[0] = float('nan')
    with pytest.raises(ValueError, match='state_dict|Nonfinite'):
        live.load_frozen_policy(ROOT, save_fixture(tmp_path, payload), builder(), device='cpu')


@pytest.mark.parametrize('field', ['robot_spec_fingerprint', 'feature_schema_fingerprint',
    'layout_fingerprint', 'normalizer_sha', 'f2_fingerprint', 'h1_config_fingerprint',
    'h2_config_fingerprint', 'training_config_fingerprint', 'structural_decision_schema'])
def test_incompatible_identity_fails_closed(tmp_path, synthetic_payload, field):
    payload = copy.deepcopy(synthetic_payload); payload['identity'][field] = 'incompatible'
    with pytest.raises(ValueError, match=field):
        live.load_frozen_policy(ROOT, save_fixture(tmp_path, payload), builder(), device='cpu')


@pytest.mark.parametrize('damage', ['schema', 'missing_identity', 'epoch', 'step'])
def test_bad_checkpoint_metadata_fails_closed(tmp_path, synthetic_payload, damage):
    payload = copy.deepcopy(synthetic_payload)
    if damage == 'schema': payload['schema_version'] = 'unsupported.v99'
    elif damage == 'missing_identity': payload['identity'].pop('layout_fingerprint')
    elif damage == 'epoch': payload['summary']['epoch'] = -1
    else: payload['step'] = True
    with pytest.raises(ValueError, match='metadata|schema|identity|contract'):
        live.load_frozen_policy(ROOT, save_fixture(tmp_path, payload), builder(), device='cpu')


def test_expected_sha_mismatch_precedes_deserialization(tmp_path, monkeypatch):
    path = tmp_path/'bad.pt'; path.write_bytes(b'not a checkpoint')
    monkeypatch.setattr(live.torch, 'load', lambda *a, **kw: pytest.fail('Must check SHA before torch.load'))
    with pytest.raises(ValueError, match='checkpoint SHA'):
        live.load_frozen_policy(ROOT, path, builder(), device='cpu', expected_checkpoint_sha256='0'*64)


def test_missing_v2_decision_contract_is_not_silently_legacy(tmp_path, synthetic_payload):
    payload = copy.deepcopy(synthetic_payload); payload['identity'].pop('structural_decision_schema')
    with pytest.raises(ValueError, match='structural_decision_schema'):
        live.load_frozen_policy(ROOT, save_fixture(tmp_path, payload), builder(), device='cpu')


def test_normalization_policy_must_match_frozen_f2(tmp_path, synthetic_payload):
    root = tmp_path/'runtime'; cfg = root/'configs/bc_v1'; cfg.mkdir(parents=True)
    for path in (ROOT/'configs/bc_v1').iterdir():
        if path.name != 'normalization.json': (cfg/path.name).symlink_to(path)
    policy = json.loads((ROOT/'configs/bc_v1/normalization.json').read_text())
    policy['fields']['module']['relative_position'] = 'preserve'
    (cfg/'normalization.json').write_text(json.dumps(policy))
    normalizers = root/'artifacts/bc_v1/normalization/f2_v1_optimized/normalizers.json'
    normalizers.parent.mkdir(parents=True)
    normalizers.symlink_to(ROOT/'artifacts/bc_v1/normalization/f2_v1_optimized/normalizers.json')
    with pytest.raises(ValueError, match='policy_fingerprint'):
        live.load_frozen_policy(root, save_fixture(tmp_path, synthetic_payload), builder(), device='cpu')


def test_no_structural_live_policy_has_no_dispatch_or_active_goal_cancellation():
    kw = settings(); cfg = ROOT/'configs/bc_v1'
    model = BCGNN(kw['spec'], kw['feature_schema'], load_model_config(cfg/'gnn_v1.json'),
        action_head_config=load_action_head_config(cfg/'action_heads_no_structural_v2.json'))
    def choose_none(module, args, prediction):
        prediction['structural']['no_structural_logits'].fill_(1e6)
    model.register_forward_hook(choose_none)
    policy = live.LivePolicy(model, kw, builder(), lambda _: context())
    from drl.IL.bc_v1.live_policy import claim_key
    from drl.IL.bc_v1 import bc_v1_contract as c
    busy = {claim_key(c.ResourceRef(c.ModuleRef('module-'+str(i)), r.resource_id))
        for i in range(2) for r in kw['spec'].module_types[0].resources}
    feedback = {'busy': busy, 'active_goal_ids': ['existing-primitive']}; before = copy.deepcopy(feedback)
    commands = policy(PhysicalSnapshot('epoch', 12.5, snapshot(2)), feedback)
    assert commands == () and policy.last_decoded == ()
    assert feedback == before and feedback['active_goal_ids'] == ['existing-primitive']
