"""J plumbing over the unchanged generic G2/H1/H2/I1 contracts."""
import copy
import importlib
import importlib.util
from pathlib import Path

import pytest
import torch

from drl.IL.bc_v1.losses import load_loss_config, compute_bc_loss
from drl.IL.tests.bc_v1.g2_fixtures import fixture, tensorize, with_structural
from drl.IL.tests.bc_v1.test_action_heads import complete_model
from drl.IL.tests.bc_v1.test_model import equal_trees
from drl.IL.tests.bc_v1.test_sidecar import assert_equal

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(autouse=True, scope='module')
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.training'), 'Missing J training helpers'
    return importlib.import_module('drl.IL.bc_v1.training')


def case():
    sample, kw = fixture(2)
    batch = tensorize(with_structural(sample), kw)
    return complete_model(kw), batch, kw


def loss_config():
    return load_loss_config(ROOT/'configs/bc_v1/loss_v1.json')


def test_optimizer_groups_cover_every_trainable_parameter_once():
    mod = api()
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.LayerNorm(4), torch.nn.Linear(4, 2))
    model.register_parameter('extra_weight', torch.nn.Parameter(torch.ones(2)))
    model.register_parameter('frozen', torch.nn.Parameter(torch.ones(1), requires_grad=False))
    optimizer = mod.build_optimizer(model)
    groups = {g['weight_decay']: {id(p) for p in g['params']} for g in optimizer.param_groups}
    assert groups[0.] == {id(model[0].bias), id(model[1].weight), id(model[1].bias), id(model[2].bias)}
    assert groups[1e-4] == {id(model[0].weight), id(model[2].weight), id(model.extra_weight)}
    ids = [id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(ids) == len(set(ids)) == len([p for p in model.parameters() if p.requires_grad])
    assert optimizer.defaults['lr'] == 3e-4
    assert optimizer.defaults['betas'] == (.9, .999) and optimizer.defaults['eps'] == 1e-8


def test_train_step_changes_h1_h2_and_all_supported_head_gradients():
    mod = api()
    model, batch, _ = case()
    before = copy.deepcopy(model.state_dict())
    result = mod.train_step(model, batch, mod.build_optimizer(model), loss_config(), audit=True)
    assert model.training and result['optimizer_step_completed']
    assert result['total_loss'] > 0 and result['grad_norm_before_clip'] > 0
    assert result['gradients']['h1']['nonzero']
    supported = {k for k, v in result['families'].items() if v['support']}
    assert supported == {'resource_mode', 'resource_payload', 'structural_choice', 'arg_category'}
    for family in supported:
        assert result['gradients'][family]['nonzero'] and result['gradients'][family]['finite']
    assert result['parameter_delta']['h1']['value'] > 0
    assert result['parameter_delta']['h2']['value'] > 0
    assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_train_step_clips_large_global_gradients_to_one():
    mod = api()
    model, batch, _ = case()
    handles = [p.register_hook(lambda grad: grad*1e6) for p in model.parameters() if p.requires_grad]
    try:
        result = mod.train_step(model, batch, mod.build_optimizer(model), loss_config())
    finally:
        for handle in handles:
            handle.remove()
    norm = torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None))
    assert result['grad_norm_before_clip'] > 1000 and result['clip_threshold'] == 1.
    assert 0 < float(norm) <= 1.00001


def test_only_inputs_enter_forward_and_device_move_preserves_routing():
    mod = api()
    model, batch, _ = case()
    batch['routing']['private_audit_poison'] = object()
    moved = mod.move_batch_to_device(batch, torch.device('cpu'))
    assert moved['routing'] is batch['routing']
    seen = []
    def boundary(module, args):
        assert args[0] is moved['inputs']
        assert 'supervision' not in args[0] and 'routing' not in args[0]
        seen.append(True)
    hook = model.register_forward_pre_hook(boundary)
    try:
        mod.train_step(model, moved, mod.build_optimizer(model), loss_config())
    finally:
        hook.remove()
    assert seen == [True]
    assert moved['supervision']['resource']['supervised'].dtype == torch.bool
    assert moved['inputs']['structural']['initiator'].dtype == torch.int64


def test_nonfinite_gradients_fail_before_optimizer_update():
    mod = api()
    model, batch, _ = case()
    before = copy.deepcopy(model.state_dict())
    optimizer = mod.build_optimizer(model)
    hook = next(model.parameters()).register_hook(lambda grad: grad*float('inf'))
    try:
        with pytest.raises((ValueError, RuntimeError), match='finite'):
            mod.train_step(model, batch, optimizer, loss_config())
    finally:
        hook.remove()
    equal_trees(before, model.state_dict())
    assert not optimizer.state


def checkpoint_case(tmp_path, device='cpu'):
    mod = api()
    model, batch, kw = case()
    model = model.to(device)
    batch = mod.move_batch_to_device(batch, torch.device(device))
    optimizer = mod.build_optimizer(model)
    mod.train_step(model, batch, optimizer, loss_config())
    identity = {'layout_fingerprint': 'current-layout', 'sidecar_manifest_fingerprint': 'current-sidecar'}
    path = tmp_path/'checkpoint.pt'
    mod.save_checkpoint(path, model, optimizer, step=1, identity=identity,
                        config={'seed': 20261006}, selection=[['train-fixture', 0]], seed=20261006)
    return mod, path, model, optimizer, batch, kw, identity


def test_checkpoint_roundtrip_new_model_optimizer_and_all_outputs(tmp_path):
    mod, path, model, optimizer, batch, kw, identity = checkpoint_case(tmp_path)
    model.eval()
    with torch.no_grad():
        expected = model(batch['inputs'])
    restored = complete_model(kw)
    fresh_optimizer = mod.build_optimizer(restored)
    payload = mod.load_checkpoint(path, restored, fresh_optimizer, expected_identity=identity)
    equal_trees(model.state_dict(), restored.state_dict())
    assert_equal(optimizer.state_dict(), fresh_optimizer.state_dict())
    restored.eval()
    with torch.no_grad():
        actual = restored(batch['inputs'])
    equal_trees(expected, actual)
    torch.testing.assert_close(compute_bc_loss(expected, batch, loss_config())[0],
                               compute_bc_loss(actual, batch, loss_config())[0], rtol=1e-6, atol=1e-7)
    assert payload['step'] == 1 and payload['seed'] == 20261006
    assert payload['selection'] == [['train-fixture', 0]] and payload['rng_state']
    assert not list(tmp_path.glob('*.tmp'))


def test_stale_checkpoint_rejected_before_partial_loading(tmp_path):
    mod, path, _, _, _, kw, identity = checkpoint_case(tmp_path)
    fresh = complete_model(kw)
    before = copy.deepcopy(fresh.state_dict())
    optimizer = mod.build_optimizer(fresh)
    with pytest.raises(ValueError, match='identity'):
        mod.load_checkpoint(path, fresh, optimizer,
                            expected_identity={**identity, 'layout_fingerprint': 'OLD-invalid-layout'})
    equal_trees(before, fresh.state_dict())
    assert not optimizer.state


def test_failed_checkpoint_save_does_not_replace_previous_checkpoint(tmp_path, monkeypatch):
    mod, path, model, optimizer, _, _, identity = checkpoint_case(tmp_path)
    before = path.read_bytes()
    def fail(*args, **kwargs):
        args[1].write(b'partial')
        raise OSError('simulated interrupted write')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError, match='interrupted'):
        mod.save_checkpoint(path, model, optimizer, step=2, identity=identity,
                            config={}, selection=[], seed=20261006)
    assert path.read_bytes() == before and not list(tmp_path.glob('*.tmp'))


def runner_api():
    assert importlib.util.find_spec('drl.IL.bc_v1.small_overfit'), 'Missing J diagnostic runner'
    return importlib.import_module('drl.IL.bc_v1.small_overfit')


def test_selection_is_unique_deterministic_train_only_and_uses_sidecar(tmp_path):
    mod = runner_api()
    from dataclasses import replace
    from drl.IL.bc_v1.sidecar import SidecarDataset, SidecarWriter
    from drl.IL.bc_v1.dataset import _hash
    from drl.IL.tests.bc_v1.test_sidecar import identity
    ids = identity()
    inventory = {split: [{'episode_id': prefix+str(i), 'samples': 1,
                         'transition_indices_fingerprint': _hash([0])} for i in range(n)]
                 for split, prefix, n in [('train', 'a', 9), ('validation', 'v', 1), ('test', 't', 1)]}
    root = tmp_path/'selection-sidecar'
    writer = SidecarWriter(root, ids, inventory)
    for split, episodes in inventory.items():
        for index, entry in enumerate(episodes):
            sample, kw = fixture(2)
            if index < 2:
                sample = with_structural(sample)
            sample = replace(sample, audit=replace(sample.audit, episode_id=entry['episode_id']))
            writer.add(tensorize(sample, kw), split, index)
    writer.finalize()
    with SidecarDataset(root, 'train', expected_identity=ids) as data:
        a = mod.select_micro_set(data, required_keys=(('a0', 0), ('a1', 0)))
        b = mod.select_micro_set(data, required_keys=(('a0', 0), ('a1', 0)))
    keys = lambda rows: [(s['routing']['audit'].episode_id, s['routing']['audit'].transition_index) for s in rows]
    assert keys(a) == keys(b) == [('a'+str(i), 0) for i in range(8)]
    with SidecarDataset(root, 'validation', expected_identity=ids) as data:
        with pytest.raises(ValueError, match='TRAIN'):
            mod.select_micro_set(data)


def test_success_requires_loss_both_supported_accuracies_and_continuous_improvement():
    mod = runner_api()
    def snapshot(loss, mode, structural, rmse):
        return {'total_loss': loss, 'selected_metrics': {
            'resource_mode': {'value': mode, 'support': 1},
            'structural_top1': {'value': structural, 'support': 4},
            'resource_payload_nrmse': {'value': rmse, 'support': 8},
            'arg_numeric_nrmse': {'value': None, 'support': 0}}}
    initial = snapshot(8., .1, 0., 2.)
    assert mod.success_gates(initial, snapshot(1.9, 1., 1., .9))['passed']
    for bad in [snapshot(2.1, 1., 1., .9), snapshot(1., .5, 1., .9),
                snapshot(1., 1., .5, .9), snapshot(1., 1., 1., 1.1)]:
        assert not mod.success_gates(initial, bad)['passed']
    assert not mod.success_gates(snapshot(8., 1., 0., 2.), snapshot(1., 1., 1., .9))['passed']


def test_selection_support_uses_i1_singleton_and_parent_masks():
    mod = runner_api()
    from drl.IL.bc_v1.batching import collate_policy_samples
    sample, kw = fixture(2)
    batch = collate_policy_samples([tensorize(with_structural(sample), kw), tensorize(sample, kw)])
    prediction = complete_model(kw)(batch['inputs'])
    support = mod.per_sample_support(prediction, batch, loss_config(), 2)
    assert support[0]['resource_mode'] == support[1]['resource_mode'] == 1
    assert support[0]['resource_payload'] == support[1]['resource_payload'] == 1
    assert support[0]['structural_choice'] == support[0]['arg_category'] == 2
    assert support[1]['structural_choice'] == support[1]['arg_category'] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA checkpoint placement regression')
def test_cuda_checkpoint_restores_adamw_step_on_cpu_and_moments_on_cuda(tmp_path):
    mod, path, model, optimizer, batch, kw, identity = checkpoint_case(tmp_path, 'cuda:0')
    restored = complete_model(kw).to('cuda:0')
    fresh_optimizer = mod.build_optimizer(restored)
    mod.load_checkpoint(path, restored, fresh_optimizer, expected_identity=identity)
    for before, after in zip(optimizer.state.values(), fresh_optimizer.state.values()):
        assert before['step'].device.type == after['step'].device.type == 'cpu'
        assert before['exp_avg'].device.type == after['exp_avg'].device.type == 'cuda'
        assert before['exp_avg_sq'].device.type == after['exp_avg_sq'].device.type == 'cuda'
    assert_equal(optimizer.state_dict(), fresh_optimizer.state_dict())
    model.eval()
    restored.eval()
    with torch.no_grad():
        equal_trees(model(batch['inputs']), restored(batch['inputs']))


@pytest.mark.parametrize('failure', ['loss', 'gradient'])
def test_nonfinite_failure_preserves_current_family_and_gradient_diagnostics(tmp_path, failure):
    import json
    mod = api()
    runner = runner_api()
    model, batch, _ = case()
    optimizer = mod.build_optimizer(model)
    before = copy.deepcopy(model.state_dict())
    hook = None
    if failure == 'loss':
        batch['supervision']['resource']['payload'][0] = float('nan')
    else:
        hook = next(model.parameters()).register_hook(lambda g: g*float('inf'))
    try:
        with pytest.raises((ValueError, RuntimeError)) as caught:
            mod.train_step(model, batch, optimizer, loss_config())
    finally:
        if hook is not None:
            hook.remove()
    diagnostics = caught.value.diagnostics
    assert diagnostics['stage'] == failure
    assert diagnostics['families']['resource_mode']['support'] == 1
    if failure == 'loss':
        assert 'Nonfinite' in diagnostics['families']['resource_payload']['error']
    else:
        assert diagnostics['families']['resource_payload']['support'] == 1
        assert not diagnostics['gradients']['finite']
        assert diagnostics['nonfinite_gradient_parameters']
    path = tmp_path/'failure.json'
    runner._write_json(path, {'step': 7, 'current_step_diagnostics': diagnostics})
    assert json.loads(path.read_text())['current_step_diagnostics'] == diagnostics
    equal_trees(before, model.state_dict())
    assert not optimizer.state


@pytest.mark.parametrize('previous,warn_only', [(False, False), (True, False), (True, True), (False, True)])
def test_deterministic_verification_context_restores_previous_setting(previous, warn_only):
    mod = api()
    original = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
    try:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)
        with mod.deterministic_verification():
            assert torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert torch.are_deterministic_algorithms_enabled() == previous
        assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
    finally:
        torch.use_deterministic_algorithms(original[0], warn_only=original[1])


@pytest.mark.parametrize('previous', [False, True])
def test_deterministic_verification_context_restores_on_exception(previous):
    mod = api()
    original = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(previous)
        with pytest.raises(RuntimeError, match='inside verification'):
            with mod.deterministic_verification():
                raise RuntimeError('inside verification')
        assert torch.are_deterministic_algorithms_enabled() == previous
    finally:
        torch.use_deterministic_algorithms(original)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA equivalence verification'))])
def test_checkpoint_verification_api_uses_eval_no_grad_and_determinism(device):
    mod = api()
    model, batch, kw = case()
    model.to(device).train()
    restored = complete_model(kw).to(device).train()
    restored.load_state_dict(model.state_dict(), strict=True)
    batch = mod.move_batch_to_device(batch, torch.device(device))
    seen = []
    def boundary(module, args):
        assert not module.training and not torch.is_grad_enabled()
        assert torch.are_deterministic_algorithms_enabled()
        assert args[0] is batch['inputs']
        seen.append(module)
    hooks = [m.register_forward_pre_hook(boundary) for m in (model, restored)]
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        result = mod.verify_checkpoint_equivalence(model, restored, batch, loss_config())
    finally:
        for h in hooks:
            h.remove()
    assert seen == [model, restored]
    assert result['output_tensors'] == result['bitwise_equal_count'] == 35
    assert result['max_output_absolute_difference'] == 0.
    assert result['loss_absolute_difference'] == result['loss_relative_difference'] == 0.
    assert result['assert_close'] == 'PASS' and result['rtol'] == 1e-6 and result['atol'] == 1e-7
    assert torch.are_deterministic_algorithms_enabled() == previous
    assert model.training and restored.training
    assert all(p.grad is None for m in (model, restored) for p in m.parameters())


@pytest.mark.parametrize('fault', ['keys', 'dtype', 'value'])
def test_checkpoint_verification_rejects_different_outputs_and_restores_context(fault):
    mod = api()
    model, batch, kw = case()
    restored = complete_model(kw)
    def corrupt(module, args, output):
        resource = output['resource']
        if fault == 'keys':
            resource.pop('payload')
        elif fault == 'dtype':
            resource['payload'] = resource['payload'].double()
        else:
            resource['payload'] = resource['payload']+1.
        return output
    hook = restored.register_forward_hook(corrupt)
    previous = torch.are_deterministic_algorithms_enabled()
    try:
        with pytest.raises((ValueError, AssertionError)):
            mod.verify_checkpoint_equivalence(model, restored, batch, loss_config())
        assert torch.are_deterministic_algorithms_enabled() == previous
    finally:
        hook.remove()
