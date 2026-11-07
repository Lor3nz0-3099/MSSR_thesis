"""Phase 5B: serial/batched MAPPO PPO reevaluation, masks and gradients.

Synthetic unit case: real MSSRMAPPO reevaluation and probability functions,
no Isaac Sim, no checkpoints, no optimizer steps. Run with installed SKRL.
"""

import torch

from drl.MARL.mappo import MSSRMAPPO, StructuredRolloutSample
from drl.MARL.policy import MARLActionBundle, MARLCandidateRecord, ResourcePayloadView


OWNERS = ("module_0", "module_1")


class TinyStructuredActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.43))
        self.resource_log_std = torch.nn.Parameter(torch.tensor(-0.22))
        self.batch_calls = 0
        self.serial_calls = 0
        self.batch_sizes = []

    def _forward(self, graph):
        scales = graph["scales"]
        records = tuple(
            MARLCandidateRecord(
                candidate_index=index,
                candidate_kind="resource",
                candidate_id=f"resource|{owner}|wheel|drive|velocity",
                owner_agent_id=owner,
                participant_agent_ids=(),
                presence_logit=self.weight * scales[index],
            )
            for index, owner in enumerate(OWNERS)
        )
        return {
            "records": records,
            "resource_payloads": {
                index: ResourcePayloadView(
                    numeric=(self.weight * scales[index]).reshape(1),
                    categorical_logits=None,
                )
                for index in range(len(OWNERS))
            },
            "structural_payloads": {},
            "resource_log_std": self.resource_log_std,
            "structural_log_std": torch.tensor(0.0),
        }

    def forward_all_owners(self, *, graph, possible_agents, routing=None):
        assert tuple(possible_agents) == OWNERS
        self.serial_calls += 1
        return self._forward(graph)

    def forward_all_owners_batch(self, *, graphs, routings, possible_agents):
        assert len(graphs) == len(routings)
        assert tuple(possible_agents) == OWNERS
        self.batch_calls += 1
        self.batch_sizes.append(len(graphs))
        return [self._forward(graph) for graph in graphs]


def _action(presence, numeric):
    return MARLActionBundle(
        presence=torch.tensor(presence, dtype=torch.float32),
        resource_numeric={i: torch.tensor([value]) for i, value in numeric.items()},
        resource_categorical={},
        structural_numeric={},
        structural_boolean={},
        structural_categorical={},
        structural_reference={},
        structural_reference_set={},
        structural_scope={},
    )


def _sample(scales, mask, presence, numeric, active_owners, *, rollout_active=True):
    return StructuredRolloutSample(
        action=_action(presence, numeric) if rollout_active else None,
        hard_mask=torch.tensor(mask, dtype=torch.bool),
        owner_active_mask={uid: uid in active_owners for uid in OWNERS},
        routing_snapshot={"test": True},
        graph_snapshot={"scales": torch.tensor(scales, dtype=torch.float32)},
        rollout_active=rollout_active,
    )


def _agent(actor):
    agent = object.__new__(MSSRMAPPO)
    agent.possible_agents = list(OWNERS)
    agent.device = torch.device("cpu")
    agent.shared_actor = actor
    agent.skip_expert_policy_inference = True
    return agent


def _loss_with_ppo_clipping(probs, entropies, old_probs, samples):
    # A synthetic clipped PPO objective applied to real reevaluated log-probs.
    advantages = {
        "module_0": torch.tensor([[0.8], [-0.4], [0.3], [0.0]]),
        "module_1": torch.tensor([[-0.2], [0.7], [-0.6], [0.0]]),
    }
    total = torch.zeros((), dtype=torch.float32)
    for uid in OWNERS:
        active = torch.tensor(
            [[float(row.rollout_active and row.owner_active_mask[uid])] for row in samples]
        )
        ratio = (probs[uid] - old_probs[uid]).exp()
        clipped = ratio.clamp(0.8, 1.2)
        surrogate = torch.minimum(ratio * advantages[uid], clipped * advantages[uid])
        total = total - (surrogate * active).sum() / active.sum().clamp_min(1)
        total = total - 0.02 * (entropies[uid] * active).sum() / active.sum().clamp_min(1)
    return total


def test_5b_ppo_reevaluation_batch_matches_serial_with_mask_busy_and_gradients():
    samples = [
        # One candidate masked, other candidate selected and numeric payload active.
        _sample([0.8, -1.3], [False, True], [1.0, 0.0], {0: 0.1}, {"module_0"}),
        _sample([-0.2, 1.6], [True, False], [0.0, 0.0], {}, {"module_1"}),
        _sample([1.1, 1.4], [False, False], [0.0, 1.0], {1: -0.3}, set(OWNERS)),
        # Expert/BUSY row must neither enter H1 nor contribute to the PPO loss.
        _sample([8.0, -9.0], [True, True], [0.0, 0.0], {}, set(), rollout_active=False),
    ]

    serial_actor = TinyStructuredActor()
    batch_actor = TinyStructuredActor()
    batch_actor.load_state_dict(serial_actor.state_dict())
    # Disable batch capability only for the serial baseline.
    serial_actor.forward_all_owners_batch = None

    serial = _agent(serial_actor)
    batched = _agent(batch_actor)
    serial_probs, serial_entropy, serial_value = serial.reevaluate_structured_minibatch(
        structured_batch=samples, owner_tensor_batch={}, evaluate_global_critic=False,
    )
    batch_probs, batch_entropy, batch_value = batched.reevaluate_structured_minibatch(
        structured_batch=samples, owner_tensor_batch={}, evaluate_global_critic=False,
    )

    assert serial_value is batch_value is None
    assert serial_actor.serial_calls == 3
    assert batch_actor.batch_calls == 1 and batch_actor.batch_sizes == [3]
    assert batch_actor.serial_calls == 0

    for uid in OWNERS:
        assert serial_probs[uid].shape == batch_probs[uid].shape == (4, 1)
        torch.testing.assert_close(serial_probs[uid], batch_probs[uid], atol=1e-7, rtol=1e-7)
        torch.testing.assert_close(serial_entropy[uid], batch_entropy[uid], atol=1e-7, rtol=1e-7)
        assert torch.equal(batch_probs[uid][-1], torch.zeros(1))
        assert torch.equal(batch_entropy[uid][-1], torch.zeros(1))

    # A candidate's full owner probability and entropy are zero if it is masked.
    assert batch_probs["module_1"][0].item() == 0.0
    assert batch_entropy["module_1"][0].item() == 0.0
    assert batch_probs["module_0"][1].item() == 0.0
    assert batch_entropy["module_0"][1].item() == 0.0

    old_probs = {uid: serial_probs[uid].detach().clone() for uid in OWNERS}
    serial_loss = _loss_with_ppo_clipping(serial_probs, serial_entropy, old_probs, samples)
    batch_loss = _loss_with_ppo_clipping(batch_probs, batch_entropy, old_probs, samples)
    torch.testing.assert_close(serial_loss, batch_loss, atol=1e-7, rtol=1e-7)

    serial_grads = torch.autograd.grad(serial_loss, list(serial_actor.parameters()))
    batch_grads = torch.autograd.grad(batch_loss, list(batch_actor.parameters()))
    for expected, actual in zip(serial_grads, batch_grads, strict=True):
        assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
        torch.testing.assert_close(expected, actual, atol=1e-7, rtol=1e-7)
    assert any(grad.abs().max() > 0 for grad in batch_grads)
    print('Phase 5B PPO reevaluation mask/BUSY/gradient parity: passed')
