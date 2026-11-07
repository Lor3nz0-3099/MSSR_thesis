"""Masked graph work may be omitted; every consumed APPO quantity must agree."""
from copy import deepcopy
import ast
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from drl.MARL.mappo import MSSRMAPPO
from drl.MARL.owner_critic import OwnerConditionedCritic


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(3))
        self.calls = 0

    def encode(self, graph):
        self.calls += 1
        return {"module_h": graph["latents"] * self.scale}


DEVICES = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])


def samples(device, active=(False, True, False, True, False)):
    structured, altruistic = [], []
    for i, valid in enumerate(active):
        graph = {"latents": torch.arange(9, device=device, dtype=torch.float32).reshape(3, 3) / 7 + i,
                 "graph_ptr": torch.tensor([0, 3], device=device)}
        structured.append(SimpleNamespace(critic_graph_snapshot=graph, rollout_active=valid))
        altruistic.append(SimpleNamespace(owner_row_index={"a": 2, "b": 0, "c": 1}))
    return structured, altruistic


def critics(device):
    return [OwnerConditionedCritic(encoder=Encoder(), module_latent_dim=3,
                                  value_hidden_dim=8, device=device) for _ in range(2)]


@cache
def original_evaluator():
    # Use the saved implementation, rather than assuming that two modes of the
    # edited function are independent numerical references.
    root = Path(__file__).resolve().parents[3]
    path = root/'docs/experiments/c1_5b_perf_20261107_petrlqrb/before/drl/MARL/mappo.py'
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'MSSRMAPPO')
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'evaluate_altruistic_critic_minibatch')
    function.decorator_list = []
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[function.name]


def evaluate(pair, structured, altruistic, **kwargs):
    function = original_evaluator() if kwargs.pop('reference_original', False) else MSSRMAPPO.evaluate_altruistic_critic_minibatch
    return function(
        structured_batch=structured, altruistic_batch=altruistic,
        possible_agents=("a", "b", "c"), owner_critic=pair[0], mean_field_critic=pair[1], **kwargs)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("clip", [0., .2])
def test_valid_predictions_loss_gradients_clipping_and_adam_equal(device, clip):
    torch.manual_seed(7033)
    original = critics(device)
    optimized = deepcopy(original)
    structured, altruistic = samples(device)
    mask = torch.tensor([s.rollout_active for s in structured], device=device)
    reference = torch.zeros(len(structured), 1, device=device)
    before = evaluate(original, structured, altruistic, reference_original=True)
    counts = {}
    after = evaluate(optimized, structured, altruistic, skip_inactive=True,
                     padding_reference=reference, diagnostics=counts)
    assert [c.encoder.calls for c in original] == [5, 5]
    assert [c.encoder.calls for c in optimized] == [2, 2]
    assert counts["valid_rows"] == 2 and counts["padding_rows_skipped"] == 3
    for pair in zip(before, after, original, optimized):
        expected, actual, old_model, new_model = pair
        for uid in expected:
            torch.testing.assert_close(actual[uid][mask], expected[uid][mask], rtol=0, atol=0)
        kwargs = dict(sampled_values={uid: torch.zeros_like(t) for uid, t in expected.items()},
                      sampled_returns={uid: torch.full_like(t, -3.) for uid, t in expected.items()},
                      sample_mask=mask, value_clip=clip, value_loss_scale=.7)
        losses = [MSSRMAPPO.compute_owner_conditioned_value_loss(predicted_values=p, **kwargs)
                  for p in (expected, actual)]
        torch.testing.assert_close(*losses, rtol=0, atol=0)
        optimizers = [torch.optim.Adam(m.parameters(), lr=3e-4) for m in (old_model, new_model)]
        norms = []
        for model, loss, optimizer in zip((old_model, new_model), losses, optimizers):
            loss.backward()
            norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), .5))
            optimizer.step()
        torch.testing.assert_close(*norms, rtol=0, atol=0)
        for old, new in zip(old_model.parameters(), new_model.parameters()):
            torch.testing.assert_close(old.grad, new.grad, rtol=0, atol=0)
            torch.testing.assert_close(old, new, rtol=0, atol=0)
        for old_state, new_state in zip(optimizers[0].state.values(), optimizers[1].state.values()):
            for key in old_state:
                torch.testing.assert_close(old_state[key], new_state[key], rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_all_padding_never_invokes_either_network(device):
    pair = critics(device)
    structured, altruistic = samples(device, (False, False))
    result = evaluate(pair, structured, altruistic, skip_inactive=True,
                      padding_reference=torch.zeros(2, 1, device=device))
    assert [c.encoder.calls for c in pair] == [0, 0]
    for values in result:
        assert all(t.shape == (2, 1) and t.device.type == torch.device(device).type
                   and torch.count_nonzero(t) == 0 for t in values.values())


def test_padding_still_validates_sidecar_owner_alignment():
    pair = critics("cpu")
    structured, altruistic = samples("cpu", (False,))
    altruistic[0].owner_row_index = {"a": 0}
    with pytest.raises(ValueError, match="owner-row mapping"):
        evaluate(pair, structured, altruistic, skip_inactive=True, padding_reference=torch.zeros(1))


@pytest.mark.parametrize("deterministic", [False, True])
def test_production_skip_is_restricted_to_deterministic_native_updates(deterministic):
    # Generic train-mode encoders can have dropout: even masked forwards consume
    # RNG. Native C1 explicitly uses deterministic_update_mode and eval mode.
    from drl.MARL.tests.c1_smdp_fixtures import make_learner, frame
    learner = make_learner()
    learner.deterministic_update_mode = deterministic
    for name, value in [("learning_epochs", 1), ("mini_batches", 1), ("ratio_clip", .2),
                        ("value_clip", .2), ("entropy_loss_scale", .001), ("value_loss_scale", 1.)]:
        setattr(learner.cfg, name, {"a": value})
    for state in range(3):
        frame(learner, [True, True], [float(state)] * 2, [1., 1.])
    learner.reevaluate_structured_minibatch = lambda **kw: (
        {"a": torch.zeros(len(kw["structured_batch"]), 1)},
        {"a": torch.zeros(len(kw["structured_batch"]), 1)}, None)
    class Captured(Exception):
        pass
    def capture(**kwargs):
        assert kwargs["skip_inactive"] is deterministic
        raise Captured
    learner.evaluate_altruistic_critic_minibatch = capture
    with pytest.raises(Captured):
        learner.coordinated_altruistic_update(timestep=0, timesteps=10)


@pytest.mark.parametrize("device", DEVICES)
def test_real_bc_h1_critic_encoders_preserve_gradients_and_optimizer(device):
    from pathlib import Path
    from drl.IL.bc_v1.action_set_head import ActionSetPolicy, load_action_set_head_config
    from drl.IL.bc_v1.model import load_model_config
    from drl.IL.tests.bc_v1.g2_fixtures import fixture, tensorize
    from drl.IL.bc_v1.training import move_batch_to_device
    root = Path(__file__).resolve().parents[3]
    raw, config = fixture(3, 2, attached=True)
    graph = move_batch_to_device({"inputs": tensorize(raw, config)["inputs"], "supervision": {}},
                                 torch.device(device))["inputs"]
    torch.manual_seed(3103)
    policy = ActionSetPolicy(config['spec'], config['feature_schema'],
        load_model_config(root/'configs/bc_v1/gnn_v1.json'),
        load_action_set_head_config(root/'configs/bc_v1/action_set_head_v1.json'))
    old_pair = [OwnerConditionedCritic(encoder=deepcopy(policy.backbone), module_latent_dim=128,
                                     device=device).eval() for _ in range(2)]
    new_pair = deepcopy(old_pair)
    structured = [SimpleNamespace(critic_graph_snapshot=deepcopy(graph), rollout_active=a)
                  for a in (False, True, False, True, False)]
    altruistic = [SimpleNamespace(owner_row_index={"a": 2, "b": 0, "c": 1}) for _ in structured]
    mask = torch.tensor([s.rollout_active for s in structured], device=device)
    rng = torch.get_rng_state().clone()
    before = evaluate(old_pair, structured, altruistic, reference_original=True)
    after = evaluate(new_pair, structured, altruistic, skip_inactive=True)
    assert torch.equal(rng, torch.get_rng_state())
    for old_values, new_values, old_model, new_model in zip(before, after, old_pair, new_pair):
        kwargs = dict(sampled_values={uid: torch.zeros_like(v) for uid, v in old_values.items()},
            sampled_returns={uid: torch.full_like(v, 3.) for uid, v in old_values.items()},
            value_clip=0., value_loss_scale=1., sample_mask=mask)
        for uid in old_values:
            torch.testing.assert_close(old_values[uid][mask], new_values[uid][mask], rtol=0, atol=0)
        losses = [MSSRMAPPO.compute_owner_conditioned_value_loss(predicted_values=v, **kwargs)
                  for v in (old_values, new_values)]
        torch.testing.assert_close(*losses, rtol=0, atol=0)
        old_opt, new_opt = [torch.optim.Adam(m.parameters(), lr=3e-4) for m in (old_model, new_model)]
        for model, opt, loss in zip((old_model, new_model), (old_opt, new_opt), losses):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
            opt.step()
        for old, new in zip(old_model.parameters(), new_model.parameters()):
            assert (old.grad is None) == (new.grad is None)
            if old.grad is not None:
                torch.testing.assert_close(old.grad, new.grad, rtol=0, atol=0)
            torch.testing.assert_close(old, new, rtol=0, atol=0)
        for old_state, new_state in zip(old_opt.state.values(), new_opt.state.values()):
            for key in old_state:
                torch.testing.assert_close(old_state[key], new_state[key], rtol=0, atol=0)
