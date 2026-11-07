"""Phase 5B: H1 batched vs serial parity for outputs and actor gradients.

Unit test only: synthetic tensorized G2 fixtures, no IsaacLab or training.
"""

import torch

from drl.IL.tests.bc_v1.g2_fixtures import fixture, tensorize, with_structural
from drl.IL.bc_v1.action_set_head import ActionSetPolicy, load_action_set_head_config
from drl.IL.bc_v1.model import load_model_config
from drl.MARL.bc_v4_actor import BCV4SharedActorWrapper
from pathlib import Path


def _tensor_leaves(tree):
    if torch.is_tensor(tree):
        yield tree
    elif isinstance(tree, dict):
        for value in tree.values():
            yield from _tensor_leaves(value)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            yield from _tensor_leaves(value)


def _assert_same_tree(serial, batched):
    if torch.is_tensor(serial):
        if serial.is_floating_point():
            torch.testing.assert_close(serial, batched, atol=1e-5, rtol=1e-5)
        else:
            torch.testing.assert_close(serial, batched, rtol=0, atol=0)
    elif isinstance(serial, dict):
        assert isinstance(batched, dict) and serial.keys() == batched.keys()
        for key in serial:
            _assert_same_tree(serial[key], batched[key])
    elif isinstance(serial, (tuple, list)):
        assert len(serial) == len(batched)
        for left, right in zip(serial, batched, strict=True):
            _assert_same_tree(left, right)
    else:
        assert serial == batched


def _differentiable_loss(outputs):
    terms = []
    for output in outputs:
        for tensor in _tensor_leaves(output):
            if tensor.is_floating_point() and tensor.requires_grad:
                assert torch.isfinite(tensor).all()
                # Nonlinear loss exercises forward and backward, including H2.
                terms.append(tensor.square().sum() / max(tensor.numel(), 1))
    assert terms, 'No differentiable outputs: gradient parity would be vacuous'
    return torch.stack(terms).sum()


def test_5b_real_h1_h2_serial_vs_batch_forward_and_backward():
    root = Path(__file__).resolve().parents[3]
    a, ka = fixture(2, 2, attached=True)
    b, kb = fixture(5, 2, geometry=False)
    sa = tensorize(a, ka)
    sb = tensorize(with_structural(b, skill=True), kb)
    samples = [sa, sb]

    torch.manual_seed(123)
    policy = ActionSetPolicy(
        ka['spec'],
        ka['feature_schema'],
        load_model_config(root / 'configs/bc_v1/gnn_v1.json'),
        load_action_set_head_config(root / 'configs/bc_v1/action_set_head_v1.json'),
    ).eval()  # Deterministic mode still permits gradients.
    wrapper = BCV4SharedActorWrapper(policy=policy)

    serial = [
        wrapper.forward_all_owners(
            graph=sample['inputs'], routing=sample['routing'],
            possible_agents=['module_0'],
        )['prediction_template']
        for sample in samples
    ]
    batched = [
        row['prediction_template']
        for row in wrapper.forward_all_owners_batch(
            graphs=[sample['inputs'] for sample in samples],
            routings=[sample['routing'] for sample in samples],
            possible_agents=['module_0'],
        )
    ]

    for expected, actual in zip(serial, batched, strict=True):
        _assert_same_tree(expected, actual)

    parameters = list(wrapper.policy.named_parameters())
    assert parameters
    serial_grads = torch.autograd.grad(
        _differentiable_loss(serial), [param for _, param in parameters],
        allow_unused=True,
    )
    batch_grads = torch.autograd.grad(
        _differentiable_loss(batched), [param for _, param in parameters],
        allow_unused=True,
    )

    checked = 0
    checked_backbone = 0
    for (name, _), serial_grad, batch_grad in zip(
        parameters, serial_grads, batch_grads, strict=True
    ):
        assert (serial_grad is None) == (batch_grad is None), name
        if serial_grad is None:
            continue
        assert torch.isfinite(serial_grad).all(), name
        assert torch.isfinite(batch_grad).all(), name
        torch.testing.assert_close(
            serial_grad, batch_grad, atol=1e-4, rtol=1e-4,
            msg=lambda message, name=name: f'{name}: {message}',
        )
        checked += 1
        if name.startswith('backbone.'):
            checked_backbone += 1
    assert checked > 0 and checked_backbone > 0
    print(f'H1/H2 gradient parity: {checked} parameter tensors; H1: {checked_backbone}')
