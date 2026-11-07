"""Integration parity of H2 batched against H2 serial after the SAME batched H1.

Run with the real project environment/configs; no Isaac simulation required.
"""

import sys
import types

import pytest
import torch


def _ensure_skrl_stub():
    try:
        import skrl.models.torch  # noqa
        return
    except ModuleNotFoundError:
        pass
    skrl = types.ModuleType('skrl')
    models = types.ModuleType('skrl.models')
    models_torch = types.ModuleType('skrl.models.torch')

    class Model(torch.nn.Module):
        def __init__(self, observation_space=None, state_space=None,
                     action_space=None, device=None):
            super().__init__()
            self.device = torch.device(device or 'cpu')

    models_torch.Model = Model
    sys.modules.update({'skrl':skrl,'skrl.models':models,
                        'skrl.models.torch':models_torch})


_ensure_skrl_stub()

from drl.IL.bc_v1.action_set_head import ActionSetPolicy, load_action_set_head_config
from drl.IL.bc_v1.model import load_model_config
from drl.IL.tests.bc_v1.g2_fixtures import fixture, tensorize, with_structural
from drl.MARL.bc_v4_actor import BCV4SharedActorWrapper
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _batch():
    specs = (
        (2, True, True),
        (5, False, True),
        (8, False, False),
    )
    samples = []
    kw = None
    for n, attached, skill in specs:
        sample, kw = fixture(n, 2, attached=attached)
        if skill:
            sample = with_structural(sample, skill=True)
        samples.append(tensorize(sample, kw))
    policy = ActionSetPolicy(
        kw['spec'], kw['feature_schema'],
        load_model_config(ROOT / 'configs/bc_v1/gnn_v1.json'),
        load_action_set_head_config(ROOT / 'configs/bc_v1/action_set_head_v1.json'),
    ).eval()
    return BCV4SharedActorWrapper(policy=policy).eval(), samples


def _compare(a, b, *, atol=3e-5, rtol=1e-4):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            _compare(a[k], b[k], atol=atol, rtol=rtol)
    elif torch.is_tensor(a):
        if a.is_floating_point():
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
        else:
            torch.testing.assert_close(a, b, atol=0, rtol=0)
    else:
        assert a == b


def _float_values(x):
    if isinstance(x, dict):
        for value in x.values():
            yield from _float_values(value)
    elif torch.is_tensor(x) and x.is_floating_point():
        yield x


def _run(actor, samples, *, batched: bool, lazy: bool):
    actor.prefer_batched_h2 = batched
    return actor.forward_all_owners_batch(
        graphs=[s['inputs'] for s in samples],
        routings=[s['routing'] for s in samples],
        possible_agents=['module_0'],
        lazy_payloads=lazy,
    )


@pytest.mark.parametrize('lazy', [False, True])
def test_h2_batched_predicts_identical_per_graph_ABI(lazy):
    actor, samples = _batch()
    with torch.inference_mode():
        serial = _run(actor, samples, batched=False, lazy=lazy)
        batched = _run(actor, samples, batched=True, lazy=lazy)
    assert len(serial) == len(batched) == 3
    for a,b in zip(serial,batched):
        _compare(a['prediction_template'],b['prediction_template'])
        assert len(a['records']) == len(b['records'])
        assert set(a['resource_payloads']) == set(b['resource_payloads'])
        assert set(a['structural_payloads']) == set(b['structural_payloads'])


@pytest.mark.parametrize("seed", [123, 3103, 7311])
def test_h2_batched_preserves_actor_and_h1_gradients(seed):
    # Reproducible policy initialization, without changing other tests' RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor, samples = _batch()
    parameters = tuple(actor.parameters())

    def evaluate(batched):
        outputs = _run(actor, samples, batched=batched, lazy=True)
        loss = sum(
            (v.square().sum() for x in outputs
             for v in _float_values(x['prediction_template'])),
            torch.zeros((), dtype=torch.float32),
        )
        grads = torch.autograd.grad(loss, parameters, allow_unused=True)
        return loss, grads

    serial_loss, serial_grad = evaluate(False)
    batched_loss, batched_grad = evaluate(True)
    torch.testing.assert_close(serial_loss, batched_loss, atol=1e-2, rtol=1e-4)
    assert len(serial_grad) == len(batched_grad)

    for i, (ga, gb) in enumerate(zip(serial_grad, batched_grad)):
        assert (ga is None) == (gb is None), f"Grad {i} presence mismatch"
        if ga is None:
            continue
        assert torch.isfinite(ga).all(), f"Grad {i}: serial non-finite"
        assert torch.isfinite(gb).all(), f"Grad {i}: batched non-finite"

        # Summation order differs between independent H2 forwards and
        # one large batched H2. A near-zero element can exceed a local
        # elementwise relative tolerance even when the overall derivative
        # is equivalent to float32 precision.
        a64, b64 = ga.double(), gb.double()
        delta = a64 - b64
        rel_l2 = (
            torch.linalg.vector_norm(delta)
            / torch.maximum(
                torch.linalg.vector_norm(a64),
                torch.linalg.vector_norm(b64),
            ).clamp_min(1.0)
        )
        assert rel_l2.item() <= 1e-5, (
            f"seed={seed}, grad={i}: relative L2={rel_l2.item():.3g}"
        )

        max_abs = delta.abs().amax().item()
        largest_grad = torch.maximum(a64.abs().amax(), b64.abs().amax()).item()
        max_allowed = 1e-3 + 1e-4 * max(largest_grad, 1.0)
        assert max_abs <= max_allowed, (
            f"seed={seed}, grad={i}: max error={max_abs:.3g}, "
            f"bound={max_allowed:.3g}"
        )
