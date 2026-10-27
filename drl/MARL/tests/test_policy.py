from types import SimpleNamespace

import pytest
import torch

from drl.IL.bc_v1.bc_v1_contract import ConnectorRef, ModuleRef, ResourceRef
from drl.MARL.policy import bind_marl_candidates


def test_candidate_binding_preserves_h2_order_and_assigns_owner_and_participants():
    m0 = ModuleRef("module_0")
    m1 = ModuleRef("module_1")
    m2 = ModuleRef("module_2")

    resource_candidates = (
        SimpleNamespace(
            resource_ref=ResourceRef(m0, "left_wheel"),
            action=SimpleNamespace(action_id="wheel_velocity"),
            mode=SimpleNamespace(mode_id="velocity"),
        ),
        SimpleNamespace(
            resource_ref=ResourceRef(m1, "pan"),
            action=SimpleNamespace(action_id="pan_velocity"),
            mode=SimpleNamespace(mode_id="velocity"),
        ),
    )

    structural_bindings = (
        (
            SimpleNamespace(action_id="dock"),
            SimpleNamespace(
                initiator=m0,
                connectors=(
                    ConnectorRef(m0, "front"),
                    ConnectorRef(m2, "back"),
                ),
                scope_members=frozenset((m0, m2)),
            ),
        ),
        (
            SimpleNamespace(action_id="stabilize"),
            SimpleNamespace(
                initiator=m1,
                connectors=(),
                scope_members=frozenset((m0, m1, m2)),
            ),
        ),
    )

    routing = {
        "resource_candidates": resource_candidates,
        "structural_bindings": structural_bindings,
    }

    presence_logits = torch.tensor([0.1, 0.2, 0.3, 0.4])

    records = bind_marl_candidates(
        routing=routing,
        presence_logits=presence_logits,
    )

    assert len(records) == 4

    assert [record.candidate_index for record in records] == [0, 1, 2, 3]
    assert [record.presence_logit.item() for record in records] == pytest.approx(
        [0.1, 0.2, 0.3, 0.4]
    )

    # Resource candidates come first, exactly as in ActionSetHead.forward().
    assert records[0].candidate_kind == "resource"
    assert records[0].candidate_id == (
        "resource|module_0|left_wheel|wheel_velocity|velocity"
    )
    assert records[0].owner_agent_id == "module_0"
    assert records[0].participant_agent_ids == ()

    assert records[1].candidate_kind == "resource"
    assert records[1].candidate_id == (
        "resource|module_1|pan|pan_velocity|velocity"
    )
    assert records[1].owner_agent_id == "module_1"
    assert records[1].participant_agent_ids == ()

    # Structural bindings follow resource candidates in their routing order.
    assert records[2].candidate_kind == "structural"
    assert records[2].candidate_id == (
        "structural|dock|module_0|module_0:front>module_2:back"
    )
    assert records[2].owner_agent_id == "module_0"
    assert records[2].participant_agent_ids == ("module_2",)

    # Endpoint-free multi-module skill: owner is initiator; other scope members
    # are participants, deterministically sorted.
    assert records[3].candidate_kind == "structural"
    assert records[3].candidate_id == (
        "structural|stabilize|module_1|none"
    )
    assert records[3].owner_agent_id == "module_1"
    assert records[3].participant_agent_ids == ("module_0", "module_2")


def test_resource_payload_routing_uses_h2_candidate_indexes_and_ptrs():
    import drl.MARL.policy as policy

    # Three resource candidates:
    #   candidate 0 -> numeric payload of width 2
    #   candidate 1 -> categorical payload with 3 logits
    #   candidate 2 -> numeric payload of width 1
    resource_parameters = {
        "payload": torch.tensor(
            [10.0, 11.0, 30.0],
            requires_grad=True,
        ),
        "payload_candidate": torch.tensor([0, 2]),
        "payload_ptr": torch.tensor([0, 2, 3]),
        "categorical": {
            "candidate": torch.tensor([1]),
            "ptr": torch.tensor([0, 3]),
            "logits": torch.tensor(
                [0.1, 0.2, 0.3],
                requires_grad=True,
            ),
        },
    }

    views = policy.route_resource_payloads(resource_parameters)

    assert set(views) == {0, 1, 2}

    assert torch.equal(
        views[0].numeric,
        torch.tensor([10.0, 11.0]),
    )
    assert views[0].categorical_logits is None

    assert views[1].numeric is None
    assert torch.equal(
        views[1].categorical_logits,
        torch.tensor([0.1, 0.2, 0.3]),
    )

    assert torch.equal(
        views[2].numeric,
        torch.tensor([30.0]),
    )
    assert views[2].categorical_logits is None

    # Routing must not detach H2 from autograd:
    # PPO must be able to backpropagate into H1/H2.
    assert views[0].numeric.requires_grad
    assert views[1].categorical_logits.requires_grad
    assert views[2].numeric.requires_grad


def test_structural_numeric_argument_routing_uses_argument_binding_table():
    import drl.MARL.policy as policy

    # argument domain -> structural binding:
    #   argument 0 -> binding 0
    #   argument 1 -> binding 2
    #   argument 2 -> binding 0
    structural_inputs = {
        "argument_binding": torch.tensor([0, 2, 0]),
    }

    # H2 predicts only arguments 2 and 1, in this order.
    # argument 2 has width 2, argument 1 has width 1.
    numeric = {
        "argument": torch.tensor([2, 1]),
        "value": torch.tensor(
            [20.0, 21.0, 10.0],
            requires_grad=True,
        ),
        "ptr": torch.tensor([0, 2, 3]),
    }

    views = policy.route_structural_numeric_arguments(
        numeric=numeric,
        structural_inputs=structural_inputs,
    )

    assert set(views) == {0, 2}

    assert len(views[0]) == 1
    assert views[0][0].argument_index == 2
    assert torch.equal(
        views[0][0].values,
        torch.tensor([20.0, 21.0]),
    )

    assert len(views[2]) == 1
    assert views[2][0].argument_index == 1
    assert torch.equal(
        views[2][0].values,
        torch.tensor([10.0]),
    )

    # PPO must retain the H2 autograd graph.
    assert views[0][0].values.requires_grad
    assert views[2][0].values.requires_grad


def test_structural_scope_routing_uses_binding_and_ptr_tables():
    import drl.MARL.policy as policy

    scope = {
        # Solo binding 0 e 2 hanno scope variabile.
        "binding": torch.tensor([0, 2]),
        "logits": torch.tensor(
            [0.1, 0.2, 0.3],
            requires_grad=True,
        ),
        "ptr": torch.tensor([0, 2, 3]),
        "domain_index": torch.tensor([4, 5, 9]),
        "required_ptr": torch.tensor([0, 1, 3]),
        "required_domain_index": torch.tensor([1, 7, 8]),
    }

    views = policy.route_structural_scopes(scope)

    assert set(views) == {0, 2}

    assert torch.equal(
        views[0].optional_logits,
        torch.tensor([0.1, 0.2]),
    )
    assert torch.equal(
        views[0].optional_domain_indices,
        torch.tensor([4, 5]),
    )
    assert torch.equal(
        views[0].required_domain_indices,
        torch.tensor([1]),
    )

    assert torch.equal(
        views[2].optional_logits,
        torch.tensor([0.3]),
    )
    assert torch.equal(
        views[2].optional_domain_indices,
        torch.tensor([9]),
    )
    assert torch.equal(
        views[2].required_domain_indices,
        torch.tensor([7, 8]),
    )

    # PPO deve mantenere il collegamento autograd con H2.
    assert views[0].optional_logits.requires_grad
    assert views[2].optional_logits.requires_grad


def test_structural_categorical_argument_routing_uses_argument_binding_and_ptrs():
    import drl.MARL.policy as policy

    structural_inputs = {
        # argument 0 -> binding 1
        # argument 1 -> binding 0
        # argument 2 -> binding 1
        "argument_binding": torch.tensor([1, 0, 1]),
    }

    categorical = {
        # H2 predicts argument 2 first, then argument 1.
        "argument": torch.tensor([2, 1]),
        "ptr": torch.tensor([0, 3, 5]),
        "logits": torch.tensor(
            [0.1, 0.2, 0.3, 1.0, 2.0],
            requires_grad=True,
        ),
    }

    views = policy.route_structural_categorical_arguments(
        categorical=categorical,
        structural_inputs=structural_inputs,
    )

    assert set(views) == {0, 1}

    assert len(views[1]) == 1
    assert views[1][0].argument_index == 2
    assert torch.equal(
        views[1][0].logits,
        torch.tensor([0.1, 0.2, 0.3]),
    )

    assert len(views[0]) == 1
    assert views[0][0].argument_index == 1
    assert torch.equal(
        views[0][0].logits,
        torch.tensor([1.0, 2.0]),
    )

    # No detach: PPO must backpropagate into H2.
    assert views[1][0].logits.requires_grad
    assert views[0][0].logits.requires_grad


def test_structural_boolean_argument_routing_uses_argument_binding_table():
    import drl.MARL.policy as policy

    structural_inputs = {
        # argument 0 -> binding 2
        # argument 1 -> binding 0
        # argument 2 -> binding 2
        "argument_binding": torch.tensor([2, 0, 2]),
    }

    boolean = {
        # H2 predicts argument 2 first, then argument 1.
        "argument": torch.tensor([2, 1]),
        "logits": torch.tensor(
            [0.25, -0.75],
            requires_grad=True,
        ),
    }

    views = policy.route_structural_boolean_arguments(
        boolean=boolean,
        structural_inputs=structural_inputs,
    )

    assert set(views) == {0, 2}

    assert len(views[2]) == 1
    assert views[2][0].argument_index == 2
    assert views[2][0].logit.item() == pytest.approx(0.25)

    assert len(views[0]) == 1
    assert views[0][0].argument_index == 1
    assert views[0][0].logit.item() == pytest.approx(-0.75)

    # No detach: PPO must backpropagate into H2.
    assert views[2][0].logit.requires_grad
    assert views[0][0].logit.requires_grad


def test_structural_reference_argument_routing_preserves_domain_indexes():
    import drl.MARL.policy as policy

    structural_inputs = {
        # argument 0 -> binding 1
        # argument 1 -> binding 0
        # argument 2 -> binding 1
        "argument_binding": torch.tensor([1, 0, 1]),
    }

    reference = {
        # H2 predicts argument 2 first, then argument 1.
        "argument": torch.tensor([2, 1]),
        "logits": torch.tensor(
            [0.1, 0.2, 0.3, 1.0, 2.0],
            requires_grad=True,
        ),
        "ptr": torch.tensor([0, 3, 5]),
        "domain_index": torch.tensor([4, 6, 9, 2, 7]),
    }

    views = policy.route_structural_reference_arguments(
        reference=reference,
        structural_inputs=structural_inputs,
    )

    assert set(views) == {0, 1}

    assert len(views[1]) == 1
    assert views[1][0].argument_index == 2
    assert torch.equal(
        views[1][0].logits,
        torch.tensor([0.1, 0.2, 0.3]),
    )
    assert torch.equal(
        views[1][0].domain_indices,
        torch.tensor([4, 6, 9]),
    )

    assert len(views[0]) == 1
    assert views[0][0].argument_index == 1
    assert torch.equal(
        views[0][0].logits,
        torch.tensor([1.0, 2.0]),
    )
    assert torch.equal(
        views[0][0].domain_indices,
        torch.tensor([2, 7]),
    )

    # No detach: PPO must backpropagate into H2.
    assert views[1][0].logits.requires_grad
    assert views[0][0].logits.requires_grad


def test_structural_reference_set_routing_separates_optional_and_required_members():
    import drl.MARL.policy as policy

    structural_inputs = {
        # argument 0 -> binding 1
        # argument 1 -> binding 0
        # argument 2 -> binding 1
        "argument_binding": torch.tensor([1, 0, 1]),
    }

    reference_set = {
        # H2 predicts argument 2 first, then argument 1.
        "argument": torch.tensor([2, 1]),

        # Optional members:
        # argument 2 -> 3 optional members
        # argument 1 -> 1 optional member
        "logits": torch.tensor(
            [0.1, 0.2, 0.3, 1.0],
            requires_grad=True,
        ),
        "ptr": torch.tensor([0, 3, 4]),
        "domain_index": torch.tensor([4, 6, 9, 2]),

        # Required members:
        # argument 2 -> 1 required member
        # argument 1 -> 2 required members
        "required_ptr": torch.tensor([0, 1, 3]),
        "required_domain_index": torch.tensor([5, 7, 8]),
    }

    views = policy.route_structural_reference_set_arguments(
        reference_set=reference_set,
        structural_inputs=structural_inputs,
    )

    assert set(views) == {0, 1}

    assert len(views[1]) == 1
    assert views[1][0].argument_index == 2
    assert torch.equal(
        views[1][0].optional_logits,
        torch.tensor([0.1, 0.2, 0.3]),
    )
    assert torch.equal(
        views[1][0].optional_domain_indices,
        torch.tensor([4, 6, 9]),
    )
    assert torch.equal(
        views[1][0].required_domain_indices,
        torch.tensor([5]),
    )

    assert len(views[0]) == 1
    assert views[0][0].argument_index == 1
    assert torch.equal(
        views[0][0].optional_logits,
        torch.tensor([1.0]),
    )
    assert torch.equal(
        views[0][0].optional_domain_indices,
        torch.tensor([2]),
    )
    assert torch.equal(
        views[0][0].required_domain_indices,
        torch.tensor([7, 8]),
    )

    # Solo gli opzionali sono stochastic/learnable.
    assert views[1][0].optional_logits.requires_grad
    assert views[0][0].optional_logits.requires_grad


def test_structural_payload_composition_keeps_one_view_per_binding():
    import drl.MARL.policy as policy

    numeric_0 = policy.StructuralNumericArgumentView(
        argument_index=0,
        values=torch.tensor([1.0], requires_grad=True),
    )

    categorical_2 = policy.StructuralCategoricalArgumentView(
        argument_index=3,
        logits=torch.tensor([0.1, 0.2], requires_grad=True),
    )

    boolean_0 = policy.StructuralBooleanArgumentView(
        argument_index=4,
        logit=torch.tensor(0.5, requires_grad=True),
    )

    reference_2 = policy.StructuralReferenceArgumentView(
        argument_index=5,
        logits=torch.tensor([1.0, 2.0], requires_grad=True),
        domain_indices=torch.tensor([7, 9]),
    )

    reference_set_2 = policy.StructuralReferenceSetArgumentView(
        argument_index=6,
        optional_logits=torch.tensor([0.3], requires_grad=True),
        optional_domain_indices=torch.tensor([11]),
        required_domain_indices=torch.tensor([12]),
    )

    scope_2 = policy.StructuralScopeView(
        optional_logits=torch.tensor([0.4], requires_grad=True),
        optional_domain_indices=torch.tensor([5]),
        required_domain_indices=torch.tensor([6]),
    )

    views = policy.compose_structural_payload_views(
        binding_count=3,
        numeric={0: (numeric_0,)},
        categorical={2: (categorical_2,)},
        boolean={0: (boolean_0,)},
        reference={2: (reference_2,)},
        reference_set={2: (reference_set_2,)},
        scope={2: scope_2},
    )

    # Every structural binding must exist, even if it has no stochastic payload.
    assert set(views) == {0, 1, 2}

    assert views[0].numeric == (numeric_0,)
    assert views[0].boolean == (boolean_0,)
    assert views[0].categorical == ()
    assert views[0].reference == ()
    assert views[0].reference_set == ()
    assert views[0].scope is None

    # Binding 1 is presence-only.
    assert views[1].numeric == ()
    assert views[1].categorical == ()
    assert views[1].boolean == ()
    assert views[1].reference == ()
    assert views[1].reference_set == ()
    assert views[1].scope is None

    assert views[2].numeric == ()
    assert views[2].categorical == (categorical_2,)
    assert views[2].boolean == ()
    assert views[2].reference == (reference_2,)
    assert views[2].reference_set == (reference_set_2,)
    assert views[2].scope is scope_2

    # Composition must reuse the original H2 views, not copy/detach them.
    assert views[0].numeric[0] is numeric_0
    assert views[2].categorical[0] is categorical_2
    assert views[2].scope is scope_2


def test_owner_presence_log_prob_is_factored_by_owner_and_ignores_hard_masked_candidates():
    import math
    import drl.MARL.policy as policy

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="resource",
            candidate_id="c0",
            owner_agent_id="module_0",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0, requires_grad=True),
        ),
        policy.MARLCandidateRecord(
            candidate_index=1,
            candidate_kind="structural",
            candidate_id="c1",
            owner_agent_id="module_0",
            participant_agent_ids=("module_1",),
            presence_logit=torch.tensor(10.0, requires_grad=True),
        ),
        policy.MARLCandidateRecord(
            candidate_index=2,
            candidate_kind="resource",
            candidate_id="c2",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=torch.tensor(math.log(3.0), requires_grad=True),
        ),
    )

    # c0 present, c1 absent but physically hard-masked, c2 present.
    sampled_presence = torch.tensor([1.0, 0.0, 1.0])
    hard_mask = torch.tensor([False, True, False])

    log_probs = policy.compute_owner_presence_log_probs(
        records=records,
        sampled_presence=sampled_presence,
        hard_mask=hard_mask,
        possible_agents=("module_0", "module_1", "module_2"),
    )

    assert set(log_probs) == {"module_0", "module_1", "module_2"}

    # module_0 receives only c0:
    # Bernoulli(logits=0), P(present)=0.5 -> log(0.5)
    assert log_probs["module_0"].item() == pytest.approx(math.log(0.5))

    # module_1 receives c2:
    # logit=log(3) -> sigmoid=0.75 -> log(0.75)
    assert log_probs["module_1"].item() == pytest.approx(math.log(0.75))

    # An agent with no available candidates contributes the neutral log-prob 0.
    assert log_probs["module_2"].item() == pytest.approx(0.0)

    # The hard-masked c1 must contribute nothing to PPO.
    total = sum(log_probs.values())
    total.backward()

    assert records[0].presence_logit.grad is not None
    assert records[2].presence_logit.grad is not None
    assert records[1].presence_logit.grad is None


def test_resource_categorical_log_prob_is_owned_and_only_counts_when_candidate_is_present():
    import math
    import drl.MARL.policy as policy

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="resource",
            candidate_id="c0",
            owner_agent_id="module_0",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=1,
            candidate_kind="resource",
            candidate_id="c1",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=2,
            candidate_kind="resource",
            candidate_id="c2",
            owner_agent_id="module_0",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0),
        ),
    )

    resource_payloads = {
        0: policy.ResourcePayloadView(
            categorical_logits=torch.tensor(
                [0.0, math.log(3.0)],
                requires_grad=True,
            ),
        ),
        1: policy.ResourcePayloadView(
            categorical_logits=torch.tensor(
                [2.0, -2.0],
                requires_grad=True,
            ),
        ),
        2: policy.ResourcePayloadView(
            categorical_logits=torch.tensor(
                [5.0, -5.0],
                requires_grad=True,
            ),
        ),
    }

    # c0 present -> category 1 contributes.
    # c1 absent -> its payload must contribute zero.
    # c2 present but hard-masked -> contributes zero.
    sampled_presence = torch.tensor([1.0, 0.0, 1.0])
    sampled_categories = {
        0: torch.tensor(1),
        1: torch.tensor(0),
        2: torch.tensor(0),
    }
    hard_mask = torch.tensor([False, False, True])

    log_probs = policy.compute_owner_resource_categorical_log_probs(
        records=records,
        resource_payloads=resource_payloads,
        sampled_presence=sampled_presence,
        sampled_categories=sampled_categories,
        hard_mask=hard_mask,
        possible_agents=("module_0", "module_1", "module_2"),
    )

    assert set(log_probs) == {"module_0", "module_1", "module_2"}

    # logits [0, log(3)] -> probabilities [0.25, 0.75]
    assert log_probs["module_0"].item() == pytest.approx(math.log(0.75))

    assert log_probs["module_1"].item() == pytest.approx(0.0)
    assert log_probs["module_2"].item() == pytest.approx(0.0)

    total = sum(log_probs.values())
    total.backward()

    # Only c0 participated in the stochastic payload decision.
    assert resource_payloads[0].categorical_logits.grad is not None
    assert resource_payloads[1].categorical_logits.grad is None
    assert resource_payloads[2].categorical_logits.grad is None


def test_resource_numeric_log_prob_uses_h2_as_gaussian_mean_and_only_counts_present_candidates():
    import math
    import drl.MARL.policy as policy

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="resource",
            candidate_id="c0",
            owner_agent_id="module_0",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=1,
            candidate_kind="resource",
            candidate_id="c1",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=2,
            candidate_kind="resource",
            candidate_id="c2",
            owner_agent_id="module_0",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0),
        ),
    )

    resource_payloads = {
        0: policy.ResourcePayloadView(
            numeric=torch.tensor(
                [0.0, 1.0],
                requires_grad=True,
            ),
        ),
        1: policy.ResourcePayloadView(
            numeric=torch.tensor(
                [2.0],
                requires_grad=True,
            ),
        ),
        2: policy.ResourcePayloadView(
            numeric=torch.tensor(
                [3.0],
                requires_grad=True,
            ),
        ),
    }

    sampled_presence = torch.tensor([1.0, 0.0, 1.0])

    sampled_numeric = {
        0: torch.tensor([1.0, 1.0]),
        1: torch.tensor([2.5]),
        2: torch.tensor([3.5]),
    }

    hard_mask = torch.tensor([False, False, True])

    # Initial std = exp(0) = 1 in normalized payload space.
    log_std = torch.tensor(0.0, requires_grad=True)

    log_probs = policy.compute_owner_resource_numeric_log_probs(
        records=records,
        resource_payloads=resource_payloads,
        sampled_presence=sampled_presence,
        sampled_numeric=sampled_numeric,
        hard_mask=hard_mask,
        possible_agents=("module_0", "module_1", "module_2"),
        log_std=log_std,
    )

    assert set(log_probs) == {"module_0", "module_1", "module_2"}

    # c0:
    # mean = [0, 1], action = [1, 1], std = 1
    # sum log-prob = -0.5 - log(2*pi)
    expected = -0.5 - math.log(2.0 * math.pi)

    assert log_probs["module_0"].item() == pytest.approx(expected)
    assert log_probs["module_1"].item() == pytest.approx(0.0)
    assert log_probs["module_2"].item() == pytest.approx(0.0)

    total = sum(log_probs.values())
    total.backward()

    # Only the active, non-masked candidate participates.
    assert resource_payloads[0].numeric.grad is not None
    assert resource_payloads[1].numeric.grad is None
    assert resource_payloads[2].numeric.grad is None

    # Exploration scale itself must be trainable by PPO.
    assert log_std.grad is not None


def test_structural_payloads_are_mapped_from_local_binding_to_global_candidate_index():
    import drl.MARL.policy as policy

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="resource",
            candidate_id="r0",
            owner_agent_id="module_0",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=1,
            candidate_kind="resource",
            candidate_id="r1",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=2,
            candidate_kind="structural",
            candidate_id="s0",
            owner_agent_id="module_0",
            participant_agent_ids=("module_1",),
            presence_logit=torch.tensor(0.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=3,
            candidate_kind="structural",
            candidate_id="s1",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0),
        ),
    )

    payload_0 = policy.StructuralPayloadView()
    payload_1 = policy.StructuralPayloadView()

    global_views = policy.map_structural_payloads_to_candidates(
        records=records,
        structural_payloads={
            0: payload_0,
            1: payload_1,
        },
    )

    assert set(global_views) == {2, 3}
    assert global_views[2] is payload_0
    assert global_views[3] is payload_1


def test_structural_boolean_log_prob_is_owned_and_only_counts_for_active_candidate():
    import math
    import drl.MARL.policy as policy

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="resource",
            candidate_id="r0",
            owner_agent_id="module_0",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=1,
            candidate_kind="resource",
            candidate_id="r1",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=2,
            candidate_kind="structural",
            candidate_id="s0",
            owner_agent_id="module_0",
            participant_agent_ids=("module_1",),
            presence_logit=torch.tensor(0.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=3,
            candidate_kind="structural",
            candidate_id="s1",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=4,
            candidate_kind="structural",
            candidate_id="s2",
            owner_agent_id="module_0",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0),
        ),
    )

    active_logit = torch.tensor(math.log(3.0), requires_grad=True)
    absent_logit = torch.tensor(2.0, requires_grad=True)
    masked_logit = torch.tensor(5.0, requires_grad=True)

    structural_payloads = {
        2: policy.StructuralPayloadView(
            boolean=(
                policy.StructuralBooleanArgumentView(
                    argument_index=4,
                    logit=active_logit,
                ),
            ),
        ),
        3: policy.StructuralPayloadView(
            boolean=(
                policy.StructuralBooleanArgumentView(
                    argument_index=5,
                    logit=absent_logit,
                ),
            ),
        ),
        4: policy.StructuralPayloadView(
            boolean=(
                policy.StructuralBooleanArgumentView(
                    argument_index=6,
                    logit=masked_logit,
                ),
            ),
        ),
    }

    sampled_presence = torch.tensor([0.0, 0.0, 1.0, 0.0, 1.0])
    hard_mask = torch.tensor([False, False, False, False, True])

    sampled_booleans = {
        2: {4: torch.tensor(1.0)},
        3: {5: torch.tensor(1.0)},
        4: {6: torch.tensor(1.0)},
    }

    log_probs = policy.compute_owner_structural_boolean_log_probs(
        records=records,
        structural_payloads=structural_payloads,
        sampled_presence=sampled_presence,
        sampled_booleans=sampled_booleans,
        hard_mask=hard_mask,
        possible_agents=("module_0", "module_1", "module_2"),
    )

    # sigmoid(log(3)) = 0.75
    assert log_probs["module_0"].item() == pytest.approx(math.log(0.75))
    assert log_probs["module_1"].item() == pytest.approx(0.0)
    assert log_probs["module_2"].item() == pytest.approx(0.0)

    total = sum(log_probs.values())
    total.backward()

    assert active_logit.grad is not None
    assert absent_logit.grad is None
    assert masked_logit.grad is None


def test_structural_categorical_log_prob_is_owned_and_only_counts_for_active_candidate():
    import math
    import drl.MARL.policy as policy

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="structural",
            candidate_id="s0",
            owner_agent_id="module_0",
            participant_agent_ids=("module_1",),
            presence_logit=torch.tensor(0.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=1,
            candidate_kind="structural",
            candidate_id="s1",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=2,
            candidate_kind="structural",
            candidate_id="s2",
            owner_agent_id="module_0",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0),
        ),
    )

    active_logits = torch.tensor(
        [0.0, math.log(3.0)],
        requires_grad=True,
    )
    absent_logits = torch.tensor(
        [2.0, -2.0],
        requires_grad=True,
    )
    masked_logits = torch.tensor(
        [5.0, -5.0],
        requires_grad=True,
    )

    structural_payloads = {
        0: policy.StructuralPayloadView(
            categorical=(
                policy.StructuralCategoricalArgumentView(
                    argument_index=3,
                    logits=active_logits,
                ),
            ),
        ),
        1: policy.StructuralPayloadView(
            categorical=(
                policy.StructuralCategoricalArgumentView(
                    argument_index=4,
                    logits=absent_logits,
                ),
            ),
        ),
        2: policy.StructuralPayloadView(
            categorical=(
                policy.StructuralCategoricalArgumentView(
                    argument_index=5,
                    logits=masked_logits,
                ),
            ),
        ),
    }

    sampled_presence = torch.tensor([1.0, 0.0, 1.0])
    hard_mask = torch.tensor([False, False, True])

    sampled_categories = {
        0: {3: torch.tensor(1)},
        1: {4: torch.tensor(0)},
        2: {5: torch.tensor(0)},
    }

    log_probs = policy.compute_owner_structural_categorical_log_probs(
        records=records,
        structural_payloads=structural_payloads,
        sampled_presence=sampled_presence,
        sampled_categories=sampled_categories,
        hard_mask=hard_mask,
        possible_agents=("module_0", "module_1", "module_2"),
    )

    # logits [0, log(3)] -> probabilities [0.25, 0.75]
    assert log_probs["module_0"].item() == pytest.approx(math.log(0.75))
    assert log_probs["module_1"].item() == pytest.approx(0.0)
    assert log_probs["module_2"].item() == pytest.approx(0.0)

    total = sum(log_probs.values())
    total.backward()

    assert active_logits.grad is not None
    assert absent_logits.grad is None
    assert masked_logits.grad is None


def test_structural_numeric_log_prob_uses_h2_as_gaussian_mean_and_only_counts_active_candidate():
    import math
    import drl.MARL.policy as policy

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="structural",
            candidate_id="set_tilt_0",
            owner_agent_id="module_0",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=1,
            candidate_kind="structural",
            candidate_id="set_tilt_1",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=2,
            candidate_kind="structural",
            candidate_id="set_tilt_2",
            owner_agent_id="module_0",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0),
        ),
    )

    active_values = torch.tensor(
        [0.0, 1.0],
        requires_grad=True,
    )
    absent_values = torch.tensor(
        [2.0],
        requires_grad=True,
    )
    masked_values = torch.tensor(
        [3.0],
        requires_grad=True,
    )

    structural_payloads = {
        0: policy.StructuralPayloadView(
            numeric=(
                policy.StructuralNumericArgumentView(
                    argument_index=4,
                    values=active_values,
                ),
            ),
        ),
        1: policy.StructuralPayloadView(
            numeric=(
                policy.StructuralNumericArgumentView(
                    argument_index=5,
                    values=absent_values,
                ),
            ),
        ),
        2: policy.StructuralPayloadView(
            numeric=(
                policy.StructuralNumericArgumentView(
                    argument_index=6,
                    values=masked_values,
                ),
            ),
        ),
    }

    sampled_presence = torch.tensor([1.0, 0.0, 1.0])
    hard_mask = torch.tensor([False, False, True])

    sampled_numeric = {
        0: {4: torch.tensor([1.0, 1.0])},
        1: {5: torch.tensor([2.5])},
        2: {6: torch.tensor([3.5])},
    }

    # sigma = exp(0) = 1 nello spazio normalizzato.
    log_std = torch.tensor(0.0, requires_grad=True)

    log_probs = policy.compute_owner_structural_numeric_log_probs(
        records=records,
        structural_payloads=structural_payloads,
        sampled_presence=sampled_presence,
        sampled_numeric=sampled_numeric,
        hard_mask=hard_mask,
        possible_agents=("module_0", "module_1", "module_2"),
        log_std=log_std,
    )

    # mean=[0,1], sample=[1,1], sigma=1
    expected = -0.5 - math.log(2.0 * math.pi)

    assert log_probs["module_0"].item() == pytest.approx(expected)
    assert log_probs["module_1"].item() == pytest.approx(0.0)
    assert log_probs["module_2"].item() == pytest.approx(0.0)

    total = sum(log_probs.values())
    total.backward()

    assert active_values.grad is not None
    assert absent_values.grad is None
    assert masked_values.grad is None

    assert log_std.grad is not None


def test_structural_reference_log_prob_maps_sampled_domain_index_to_categorical_position():
    import math
    import drl.MARL.policy as policy

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="structural",
            candidate_id="dock_0",
            owner_agent_id="module_0",
            participant_agent_ids=("module_1",),
            presence_logit=torch.tensor(0.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=1,
            candidate_kind="structural",
            candidate_id="dock_1",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0),
        ),
    )

    active_logits = torch.tensor(
        [0.0, math.log(3.0), 0.0],
        requires_grad=True,
    )
    absent_logits = torch.tensor(
        [2.0, -2.0],
        requires_grad=True,
    )

    structural_payloads = {
        0: policy.StructuralPayloadView(
            reference=(
                policy.StructuralReferenceArgumentView(
                    argument_index=7,
                    logits=active_logits,
                    domain_indices=torch.tensor([4, 9, 12]),
                ),
            ),
        ),
        1: policy.StructuralPayloadView(
            reference=(
                policy.StructuralReferenceArgumentView(
                    argument_index=8,
                    logits=absent_logits,
                    domain_indices=torch.tensor([3, 6]),
                ),
            ),
        ),
    }

    sampled_presence = torch.tensor([1.0, 0.0])
    hard_mask = torch.tensor([False, False])

    # We store the actual selected domain index, not its local categorical slot.
    sampled_references = {
        0: {7: torch.tensor(9)},
        1: {8: torch.tensor(3)},
    }

    log_probs = policy.compute_owner_structural_reference_log_probs(
        records=records,
        structural_payloads=structural_payloads,
        sampled_presence=sampled_presence,
        sampled_references=sampled_references,
        hard_mask=hard_mask,
        possible_agents=("module_0", "module_1", "module_2"),
    )

    # logits [0, log(3), 0] -> probabilities [0.2, 0.6, 0.2]
    # domain index 9 corresponds to local categorical position 1.
    assert log_probs["module_0"].item() == pytest.approx(math.log(0.6))
    assert log_probs["module_1"].item() == pytest.approx(0.0)
    assert log_probs["module_2"].item() == pytest.approx(0.0)

    total = sum(log_probs.values())
    total.backward()

    assert active_logits.grad is not None
    assert absent_logits.grad is None


def test_structural_reference_set_log_prob_counts_only_optional_members():
    import math
    import drl.MARL.policy as policy

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="structural",
            candidate_id="s0",
            owner_agent_id="module_0",
            participant_agent_ids=("module_1",),
            presence_logit=torch.tensor(0.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=1,
            candidate_kind="structural",
            candidate_id="s1",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0),
        ),
    )

    active_logits = torch.tensor(
        [0.0, math.log(3.0)],
        requires_grad=True,
    )
    absent_logits = torch.tensor(
        [2.0, -2.0],
        requires_grad=True,
    )

    structural_payloads = {
        0: policy.StructuralPayloadView(
            reference_set=(
                policy.StructuralReferenceSetArgumentView(
                    argument_index=10,
                    optional_logits=active_logits,
                    optional_domain_indices=torch.tensor([4, 9]),
                    required_domain_indices=torch.tensor([12]),
                ),
            ),
        ),
        1: policy.StructuralPayloadView(
            reference_set=(
                policy.StructuralReferenceSetArgumentView(
                    argument_index=11,
                    optional_logits=absent_logits,
                    optional_domain_indices=torch.tensor([3, 6]),
                    required_domain_indices=torch.tensor([8]),
                ),
            ),
        ),
    }

    sampled_presence = torch.tensor([1.0, 0.0])
    hard_mask = torch.tensor([False, False])

    # Store actual selected domain indices, including deterministic required ones.
    sampled_reference_sets = {
        0: {10: torch.tensor([9, 12])},
        1: {11: torch.tensor([3, 8])},
    }

    log_probs = policy.compute_owner_structural_reference_set_log_probs(
        records=records,
        structural_payloads=structural_payloads,
        sampled_presence=sampled_presence,
        sampled_reference_sets=sampled_reference_sets,
        hard_mask=hard_mask,
        possible_agents=("module_0", "module_1", "module_2"),
    )

    # optional domain 4: not selected, sigmoid(0)=0.5 -> log(0.5)
    # optional domain 9: selected, sigmoid(log(3))=0.75 -> log(0.75)
    # required domain 12 contributes zero.
    assert log_probs["module_0"].item() == pytest.approx(
        math.log(0.5) + math.log(0.75)
    )
    assert log_probs["module_1"].item() == pytest.approx(0.0)
    assert log_probs["module_2"].item() == pytest.approx(0.0)

    total = sum(log_probs.values())
    total.backward()

    assert active_logits.grad is not None
    assert absent_logits.grad is None


def test_structural_scope_log_prob_counts_only_optional_members():
    import math
    import drl.MARL.policy as policy

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="structural",
            candidate_id="s0",
            owner_agent_id="module_0",
            participant_agent_ids=("module_1", "module_2"),
            presence_logit=torch.tensor(0.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=1,
            candidate_kind="structural",
            candidate_id="s1",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0),
        ),
    )

    active_logits = torch.tensor(
        [0.0, math.log(3.0)],
        requires_grad=True,
    )
    absent_logits = torch.tensor(
        [2.0, -2.0],
        requires_grad=True,
    )

    structural_payloads = {
        0: policy.StructuralPayloadView(
            scope=policy.StructuralScopeView(
                optional_logits=active_logits,
                optional_domain_indices=torch.tensor([4, 9]),
                required_domain_indices=torch.tensor([12]),
            ),
        ),
        1: policy.StructuralPayloadView(
            scope=policy.StructuralScopeView(
                optional_logits=absent_logits,
                optional_domain_indices=torch.tensor([3, 6]),
                required_domain_indices=torch.tensor([8]),
            ),
        ),
    }

    sampled_presence = torch.tensor([1.0, 0.0])
    hard_mask = torch.tensor([False, False])

    # Store actual selected scope members, including deterministic required ones.
    sampled_scopes = {
        0: torch.tensor([9, 12]),
        1: torch.tensor([3, 8]),
    }

    log_probs = policy.compute_owner_structural_scope_log_probs(
        records=records,
        structural_payloads=structural_payloads,
        sampled_presence=sampled_presence,
        sampled_scopes=sampled_scopes,
        hard_mask=hard_mask,
        possible_agents=("module_0", "module_1", "module_2"),
    )

    # optional 4: not selected -> log(0.5)
    # optional 9: selected -> log(0.75)
    # required 12: deterministic -> zero contribution
    assert log_probs["module_0"].item() == pytest.approx(
        math.log(0.5) + math.log(0.75)
    )
    assert log_probs["module_1"].item() == pytest.approx(0.0)
    assert log_probs["module_2"].item() == pytest.approx(0.0)

    total = sum(log_probs.values())
    total.backward()

    assert active_logits.grad is not None
    assert absent_logits.grad is None


def test_owner_log_prob_components_are_summed_without_breaking_autograd():
    import drl.MARL.policy as policy

    presence_0 = torch.tensor(-0.2, requires_grad=True)
    resource_0 = torch.tensor(-0.3, requires_grad=True)
    structural_0 = torch.tensor(-0.5, requires_grad=True)

    presence_1 = torch.tensor(-0.7, requires_grad=True)
    structural_1 = torch.tensor(-0.4, requires_grad=True)

    combined = policy.combine_owner_log_prob_components(
        possible_agents=("module_0", "module_1", "module_2"),
        components=(
            {
                "module_0": presence_0,
                "module_1": presence_1,
                "module_2": torch.tensor(0.0),
            },
            {
                "module_0": resource_0,
                "module_1": torch.tensor(0.0),
                "module_2": torch.tensor(0.0),
            },
            {
                "module_0": structural_0,
                "module_1": structural_1,
                "module_2": torch.tensor(0.0),
            },
        ),
    )

    assert combined["module_0"].item() == pytest.approx(-1.0)
    assert combined["module_1"].item() == pytest.approx(-1.1)
    assert combined["module_2"].item() == pytest.approx(0.0)

    total = sum(combined.values())
    total.backward()

    assert presence_0.grad is not None
    assert resource_0.grad is not None
    assert structural_0.grad is not None
    assert presence_1.grad is not None
    assert structural_1.grad is not None


def test_marl_action_bundle_preserves_semantic_sampled_actions():
    import drl.MARL.policy as policy

    bundle = policy.MARLActionBundle(
        presence=torch.tensor([1.0, 0.0]),
        resource_numeric={
            0: torch.tensor([0.25]),
        },
        resource_categorical={
            0: torch.tensor(2),
        },
        structural_numeric={
            1: {
                4: torch.tensor([0.5]),
            },
        },
        structural_boolean={
            1: {
                5: torch.tensor(1.0),
            },
        },
        structural_categorical={
            1: {
                6: torch.tensor(2),
            },
        },
        structural_reference={
            1: {
                7: torch.tensor(9),
            },
        },
        structural_reference_set={
            1: {
                8: torch.tensor([3, 12]),
            },
        },
        structural_scope={
            1: torch.tensor([4, 12]),
        },
    )

    assert torch.equal(
        bundle.presence,
        torch.tensor([1.0, 0.0]),
    )

    assert bundle.resource_numeric[0].item() == pytest.approx(0.25)
    assert bundle.resource_categorical[0].item() == 2

    assert bundle.structural_numeric[1][4].item() == pytest.approx(0.5)
    assert bundle.structural_boolean[1][5].item() == pytest.approx(1.0)
    assert bundle.structural_categorical[1][6].item() == 2

    # References are stored semantically as real domain indices.
    assert bundle.structural_reference[1][7].item() == 9
    assert torch.equal(
        bundle.structural_reference_set[1][8],
        torch.tensor([3, 12]),
    )
    assert torch.equal(
        bundle.structural_scope[1],
        torch.tensor([4, 12]),
    )


def test_compute_owner_action_log_probs_composes_presence_resource_and_structural_terms():
    import math
    import drl.MARL.policy as policy

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="resource",
            candidate_id="wheel_0",
            owner_agent_id="module_0",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0, requires_grad=True),
        ),
        policy.MARLCandidateRecord(
            candidate_index=1,
            candidate_kind="structural",
            candidate_id="tilt_0",
            owner_agent_id="module_0",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0, requires_grad=True),
        ),
        policy.MARLCandidateRecord(
            candidate_index=2,
            candidate_kind="structural",
            candidate_id="idle_candidate_1",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=torch.tensor(0.0, requires_grad=True),
        ),
    )

    resource_mean = torch.tensor([0.0], requires_grad=True)
    resource_category_logits = torch.tensor(
        [0.0, 0.0],
        requires_grad=True,
    )
    structural_boolean_logit = torch.tensor(
        0.0,
        requires_grad=True,
    )

    resource_payloads = {
        0: policy.ResourcePayloadView(
            numeric=resource_mean,
            categorical_logits=resource_category_logits,
        ),
    }

    structural_payloads = {
        1: policy.StructuralPayloadView(
            boolean=(
                policy.StructuralBooleanArgumentView(
                    argument_index=4,
                    logit=structural_boolean_logit,
                ),
            ),
        ),
        2: policy.StructuralPayloadView(),
    }

    action = policy.MARLActionBundle(
        presence=torch.tensor([1.0, 1.0, 0.0]),
        resource_numeric={
            0: torch.tensor([0.0]),
        },
        resource_categorical={
            0: torch.tensor(0),
        },
        structural_numeric={},
        structural_boolean={
            1: {
                4: torch.tensor(1.0),
            },
        },
        structural_categorical={},
        structural_reference={},
        structural_reference_set={},
        structural_scope={},
    )

    resource_log_std = torch.tensor(0.0, requires_grad=True)
    structural_log_std = torch.tensor(0.0, requires_grad=True)

    log_probs = policy.compute_owner_action_log_probs(
        records=records,
        resource_payloads=resource_payloads,
        structural_payloads=structural_payloads,
        action=action,
        hard_mask=torch.tensor([False, False, False]),
        possible_agents=("module_0", "module_1", "module_2"),
        resource_log_std=resource_log_std,
        structural_log_std=structural_log_std,
    )

    gaussian_at_mean = -0.5 * math.log(2.0 * math.pi)

    # module_0:
    #   resource presence      log(0.5)
    #   structural presence    log(0.5)
    #   resource categorical   log(0.5)
    #   resource numeric       N(0,1) evaluated at 0
    #   structural boolean     log(0.5)
    expected_module_0 = (
        4.0 * math.log(0.5)
        + gaussian_at_mean
    )

    # module_1 owns one absent structural candidate:
    # presence=0 under Bernoulli(logit=0) -> log(0.5)
    expected_module_1 = math.log(0.5)

    assert log_probs["module_0"].item() == pytest.approx(
        expected_module_0
    )
    assert log_probs["module_1"].item() == pytest.approx(
        expected_module_1
    )
    assert log_probs["module_2"].item() == pytest.approx(0.0)

    total = sum(log_probs.values())
    total.backward()

    assert records[0].presence_logit.grad is not None
    assert records[1].presence_logit.grad is not None
    assert records[2].presence_logit.grad is not None
    assert resource_mean.grad is not None
    assert resource_category_logits.grad is not None
    assert structural_boolean_logit.grad is not None
    assert resource_log_std.grad is not None

    # No structural numeric action was active in this sample.
    assert structural_log_std.grad is None


def test_sample_marl_action_bundle_samples_presence_and_forces_hard_mask_absent():
    import drl.MARL.policy as policy

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="resource",
            candidate_id="active",
            owner_agent_id="module_0",
            participant_agent_ids=(),
            presence_logit=torch.tensor(50.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=1,
            candidate_kind="structural",
            candidate_id="inactive",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=torch.tensor(-50.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=2,
            candidate_kind="structural",
            candidate_id="masked",
            owner_agent_id="module_2",
            participant_agent_ids=(),
            presence_logit=torch.tensor(50.0),
        ),
    )

    action = policy.sample_marl_action_bundle(
        records=records,
        resource_payloads={},
        structural_payloads={},
        hard_mask=torch.tensor([False, False, True]),
        resource_log_std=torch.tensor(0.0),
        structural_log_std=torch.tensor(0.0),
    )

    assert isinstance(action, policy.MARLActionBundle)

    assert torch.equal(
        action.presence,
        torch.tensor([1.0, 0.0, 0.0]),
    )

    assert action.resource_numeric == {}
    assert action.resource_categorical == {}
    assert action.structural_numeric == {}
    assert action.structural_boolean == {}
    assert action.structural_categorical == {}
    assert action.structural_reference == {}
    assert action.structural_reference_set == {}
    assert action.structural_scope == {}


def test_sample_marl_action_bundle_samples_resource_numeric_only_for_present_candidate():
    import drl.MARL.policy as policy

    active_mean = torch.tensor([0.2, -0.4])
    absent_mean = torch.tensor([9.0])

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="resource",
            candidate_id="active_resource",
            owner_agent_id="module_0",
            participant_agent_ids=(),
            presence_logit=torch.tensor(50.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=1,
            candidate_kind="resource",
            candidate_id="absent_resource",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=torch.tensor(-50.0),
        ),
    )

    resource_payloads = {
        0: policy.ResourcePayloadView(
            numeric=active_mean,
        ),
        1: policy.ResourcePayloadView(
            numeric=absent_mean,
        ),
    }

    log_std = torch.tensor(-0.7)

    # Reproduce the expected RNG sequence:
    # two Bernoulli presence samples, then one Normal payload sample.
    torch.manual_seed(1234)

    torch.distributions.Bernoulli(
        logits=records[0].presence_logit
    ).sample()
    torch.distributions.Bernoulli(
        logits=records[1].presence_logit
    ).sample()

    expected_numeric = torch.distributions.Normal(
        loc=active_mean,
        scale=torch.exp(log_std),
    ).sample()

    torch.manual_seed(1234)

    action = policy.sample_marl_action_bundle(
        records=records,
        resource_payloads=resource_payloads,
        structural_payloads={},
        hard_mask=torch.tensor([False, False]),
        resource_log_std=log_std,
        structural_log_std=torch.tensor(0.0),
    )

    assert torch.equal(
        action.presence,
        torch.tensor([1.0, 0.0]),
    )

    assert set(action.resource_numeric) == {0}
    assert torch.equal(
        action.resource_numeric[0],
        expected_numeric,
    )

    # presence=0 means the payload is not sampled at all.
    assert 1 not in action.resource_numeric


def test_sample_marl_action_bundle_samples_resource_categorical_only_for_present_candidate():
    import drl.MARL.policy as policy

    active_logits = torch.tensor([-50.0, 50.0])
    absent_logits = torch.tensor([50.0, -50.0])

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="resource",
            candidate_id="active_resource",
            owner_agent_id="module_0",
            participant_agent_ids=(),
            presence_logit=torch.tensor(50.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=1,
            candidate_kind="resource",
            candidate_id="absent_resource",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=torch.tensor(-50.0),
        ),
    )

    resource_payloads = {
        0: policy.ResourcePayloadView(
            categorical_logits=active_logits,
        ),
        1: policy.ResourcePayloadView(
            categorical_logits=absent_logits,
        ),
    }

    action = policy.sample_marl_action_bundle(
        records=records,
        resource_payloads=resource_payloads,
        structural_payloads={},
        hard_mask=torch.tensor([False, False]),
        resource_log_std=torch.tensor(0.0),
        structural_log_std=torch.tensor(0.0),
    )

    assert torch.equal(
        action.presence,
        torch.tensor([1.0, 0.0]),
    )

    assert set(action.resource_categorical) == {0}

    # logits [-50, 50] make category 1 effectively deterministic.
    assert action.resource_categorical[0].item() == 1

    # presence=0 -> categorical payload is not sampled.
    assert 1 not in action.resource_categorical


def test_sample_marl_action_bundle_samples_structural_numeric_only_for_present_candidate():
    import drl.MARL.policy as policy

    active_mean = torch.tensor([0.25, -0.5])
    absent_mean = torch.tensor([8.0])

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="structural",
            candidate_id="active_tilt",
            owner_agent_id="module_0",
            participant_agent_ids=(),
            presence_logit=torch.tensor(50.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=1,
            candidate_kind="structural",
            candidate_id="absent_tilt",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=torch.tensor(-50.0),
        ),
    )

    structural_payloads = {
        0: policy.StructuralPayloadView(
            numeric=(
                policy.StructuralNumericArgumentView(
                    argument_index=4,
                    values=active_mean,
                ),
            ),
        ),
        1: policy.StructuralPayloadView(
            numeric=(
                policy.StructuralNumericArgumentView(
                    argument_index=5,
                    values=absent_mean,
                ),
            ),
        ),
    }

    log_std = torch.tensor(-0.4)

    # Expected RNG sequence:
    # 2 Bernoulli presence samples, then active structural Normal sample.
    torch.manual_seed(4321)

    torch.distributions.Bernoulli(
        logits=records[0].presence_logit
    ).sample()
    torch.distributions.Bernoulli(
        logits=records[1].presence_logit
    ).sample()

    expected_numeric = torch.distributions.Normal(
        loc=active_mean,
        scale=torch.exp(log_std),
    ).sample()

    torch.manual_seed(4321)

    action = policy.sample_marl_action_bundle(
        records=records,
        resource_payloads={},
        structural_payloads=structural_payloads,
        hard_mask=torch.tensor([False, False]),
        resource_log_std=torch.tensor(0.0),
        structural_log_std=log_std,
    )

    assert torch.equal(
        action.presence,
        torch.tensor([1.0, 0.0]),
    )

    assert set(action.structural_numeric) == {0}
    assert set(action.structural_numeric[0]) == {4}

    assert torch.equal(
        action.structural_numeric[0][4],
        expected_numeric,
    )

    # presence=0 -> no numeric payload sampling.
    assert 1 not in action.structural_numeric


def test_sample_marl_action_bundle_samples_structural_boolean_only_for_present_candidate():
    import drl.MARL.policy as policy

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="structural",
            candidate_id="active_structural",
            owner_agent_id="module_0",
            participant_agent_ids=(),
            presence_logit=torch.tensor(50.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=1,
            candidate_kind="structural",
            candidate_id="absent_structural",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=torch.tensor(-50.0),
        ),
    )

    structural_payloads = {
        0: policy.StructuralPayloadView(
            boolean=(
                policy.StructuralBooleanArgumentView(
                    argument_index=4,
                    logit=torch.tensor(50.0),
                ),
            ),
        ),
        1: policy.StructuralPayloadView(
            boolean=(
                policy.StructuralBooleanArgumentView(
                    argument_index=5,
                    logit=torch.tensor(50.0),
                ),
            ),
        ),
    }

    action = policy.sample_marl_action_bundle(
        records=records,
        resource_payloads={},
        structural_payloads=structural_payloads,
        hard_mask=torch.tensor([False, False]),
        resource_log_std=torch.tensor(0.0),
        structural_log_std=torch.tensor(0.0),
    )

    assert torch.equal(
        action.presence,
        torch.tensor([1.0, 0.0]),
    )

    assert set(action.structural_boolean) == {0}
    assert set(action.structural_boolean[0]) == {4}
    assert action.structural_boolean[0][4].item() == pytest.approx(1.0)

    assert 1 not in action.structural_boolean


def test_sample_marl_action_bundle_samples_structural_categorical_only_for_present_candidate():
    import drl.MARL.policy as policy

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="structural",
            candidate_id="active_structural",
            owner_agent_id="module_0",
            participant_agent_ids=(),
            presence_logit=torch.tensor(50.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=1,
            candidate_kind="structural",
            candidate_id="absent_structural",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=torch.tensor(-50.0),
        ),
    )

    structural_payloads = {
        0: policy.StructuralPayloadView(
            categorical=(
                policy.StructuralCategoricalArgumentView(
                    argument_index=4,
                    logits=torch.tensor([-50.0, 50.0]),
                ),
            ),
        ),
        1: policy.StructuralPayloadView(
            categorical=(
                policy.StructuralCategoricalArgumentView(
                    argument_index=5,
                    logits=torch.tensor([50.0, -50.0]),
                ),
            ),
        ),
    }

    action = policy.sample_marl_action_bundle(
        records=records,
        resource_payloads={},
        structural_payloads=structural_payloads,
        hard_mask=torch.tensor([False, False]),
        resource_log_std=torch.tensor(0.0),
        structural_log_std=torch.tensor(0.0),
    )

    assert torch.equal(
        action.presence,
        torch.tensor([1.0, 0.0]),
    )

    assert set(action.structural_categorical) == {0}
    assert set(action.structural_categorical[0]) == {4}

    # logits [-50, 50] -> category 1 effectively deterministic.
    assert action.structural_categorical[0][4].item() == 1

    assert 1 not in action.structural_categorical


def test_sample_marl_action_bundle_samples_structural_reference_as_domain_index():
    import drl.MARL.policy as policy

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="structural",
            candidate_id="active_reference",
            owner_agent_id="module_0",
            participant_agent_ids=("module_1",),
            presence_logit=torch.tensor(50.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=1,
            candidate_kind="structural",
            candidate_id="absent_reference",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=torch.tensor(-50.0),
        ),
    )

    structural_payloads = {
        0: policy.StructuralPayloadView(
            reference=(
                policy.StructuralReferenceArgumentView(
                    argument_index=7,
                    logits=torch.tensor([-50.0, 50.0, -50.0]),
                    domain_indices=torch.tensor([4, 9, 12]),
                ),
            ),
        ),
        1: policy.StructuralPayloadView(
            reference=(
                policy.StructuralReferenceArgumentView(
                    argument_index=8,
                    logits=torch.tensor([50.0, -50.0]),
                    domain_indices=torch.tensor([3, 6]),
                ),
            ),
        ),
    }

    action = policy.sample_marl_action_bundle(
        records=records,
        resource_payloads={},
        structural_payloads=structural_payloads,
        hard_mask=torch.tensor([False, False]),
        resource_log_std=torch.tensor(0.0),
        structural_log_std=torch.tensor(0.0),
    )

    assert torch.equal(
        action.presence,
        torch.tensor([1.0, 0.0]),
    )

    assert set(action.structural_reference) == {0}
    assert set(action.structural_reference[0]) == {7}

    # Local categorical slot 1 corresponds to semantic domain index 9.
    assert action.structural_reference[0][7].item() == 9

    # presence=0 -> reference payload is not sampled.
    assert 1 not in action.structural_reference


def test_sample_marl_action_bundle_samples_structural_reference_set_with_required_members():
    import drl.MARL.policy as policy

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="structural",
            candidate_id="active_reference_set",
            owner_agent_id="module_0",
            participant_agent_ids=("module_1",),
            presence_logit=torch.tensor(50.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=1,
            candidate_kind="structural",
            candidate_id="absent_reference_set",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=torch.tensor(-50.0),
        ),
    )

    structural_payloads = {
        0: policy.StructuralPayloadView(
            reference_set=(
                policy.StructuralReferenceSetArgumentView(
                    argument_index=10,
                    optional_logits=torch.tensor([-50.0, 50.0]),
                    optional_domain_indices=torch.tensor([4, 9]),
                    required_domain_indices=torch.tensor([12]),
                ),
            ),
        ),
        1: policy.StructuralPayloadView(
            reference_set=(
                policy.StructuralReferenceSetArgumentView(
                    argument_index=11,
                    optional_logits=torch.tensor([50.0]),
                    optional_domain_indices=torch.tensor([3]),
                    required_domain_indices=torch.tensor([8]),
                ),
            ),
        ),
    }

    action = policy.sample_marl_action_bundle(
        records=records,
        resource_payloads={},
        structural_payloads=structural_payloads,
        hard_mask=torch.tensor([False, False]),
        resource_log_std=torch.tensor(0.0),
        structural_log_std=torch.tensor(0.0),
    )

    assert torch.equal(
        action.presence,
        torch.tensor([1.0, 0.0]),
    )

    assert set(action.structural_reference_set) == {0}
    assert set(action.structural_reference_set[0]) == {10}

    # optional 4 -> rejected
    # optional 9 -> selected
    # required 12 -> always included
    assert torch.equal(
        action.structural_reference_set[0][10],
        torch.tensor([9, 12]),
    )

    # presence=0 -> no reference-set payload is sampled.
    assert 1 not in action.structural_reference_set


def test_sample_marl_action_bundle_samples_structural_scope_with_required_members():
    import drl.MARL.policy as policy

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="structural",
            candidate_id="active_scope",
            owner_agent_id="module_0",
            participant_agent_ids=("module_1", "module_2"),
            presence_logit=torch.tensor(50.0),
        ),
        policy.MARLCandidateRecord(
            candidate_index=1,
            candidate_kind="structural",
            candidate_id="absent_scope",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=torch.tensor(-50.0),
        ),
    )

    structural_payloads = {
        0: policy.StructuralPayloadView(
            scope=policy.StructuralScopeView(
                optional_logits=torch.tensor([-50.0, 50.0]),
                optional_domain_indices=torch.tensor([4, 9]),
                required_domain_indices=torch.tensor([12]),
            ),
        ),
        1: policy.StructuralPayloadView(
            scope=policy.StructuralScopeView(
                optional_logits=torch.tensor([50.0]),
                optional_domain_indices=torch.tensor([3]),
                required_domain_indices=torch.tensor([8]),
            ),
        ),
    }

    action = policy.sample_marl_action_bundle(
        records=records,
        resource_payloads={},
        structural_payloads=structural_payloads,
        hard_mask=torch.tensor([False, False]),
        resource_log_std=torch.tensor(0.0),
        structural_log_std=torch.tensor(0.0),
    )

    assert torch.equal(
        action.presence,
        torch.tensor([1.0, 0.0]),
    )

    assert set(action.structural_scope) == {0}

    # optional 4 -> rejected
    # optional 9 -> selected
    # required 12 -> always included
    assert torch.equal(
        action.structural_scope[0],
        torch.tensor([9, 12]),
    )

    # presence=0 -> no scope payload is sampled.
    assert 1 not in action.structural_scope


def test_sampled_marl_action_bundle_can_be_reevaluated_end_to_end():
    import drl.MARL.policy as policy

    resource_presence = torch.tensor(50.0, requires_grad=True)
    structural_presence = torch.tensor(50.0, requires_grad=True)

    resource_mean = torch.tensor([0.2], requires_grad=True)
    resource_category_logits = torch.tensor(
        [-1.0, 1.0],
        requires_grad=True,
    )

    structural_mean = torch.tensor([0.3], requires_grad=True)
    structural_boolean_logit = torch.tensor(1.0, requires_grad=True)
    structural_category_logits = torch.tensor(
        [-1.0, 1.0],
        requires_grad=True,
    )
    structural_reference_logits = torch.tensor(
        [-1.0, 1.0],
        requires_grad=True,
    )
    structural_reference_set_logits = torch.tensor(
        [1.0, -1.0],
        requires_grad=True,
    )
    structural_scope_logits = torch.tensor(
        [-1.0, 1.0],
        requires_grad=True,
    )

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="resource",
            candidate_id="resource_0",
            owner_agent_id="module_0",
            participant_agent_ids=(),
            presence_logit=resource_presence,
        ),
        policy.MARLCandidateRecord(
            candidate_index=1,
            candidate_kind="structural",
            candidate_id="structural_0",
            owner_agent_id="module_1",
            participant_agent_ids=("module_2",),
            presence_logit=structural_presence,
        ),
    )

    resource_payloads = {
        0: policy.ResourcePayloadView(
            numeric=resource_mean,
            categorical_logits=resource_category_logits,
        ),
    }

    structural_payloads = {
        1: policy.StructuralPayloadView(
            numeric=(
                policy.StructuralNumericArgumentView(
                    argument_index=4,
                    values=structural_mean,
                ),
            ),
            boolean=(
                policy.StructuralBooleanArgumentView(
                    argument_index=5,
                    logit=structural_boolean_logit,
                ),
            ),
            categorical=(
                policy.StructuralCategoricalArgumentView(
                    argument_index=6,
                    logits=structural_category_logits,
                ),
            ),
            reference=(
                policy.StructuralReferenceArgumentView(
                    argument_index=7,
                    logits=structural_reference_logits,
                    domain_indices=torch.tensor([20, 21]),
                ),
            ),
            reference_set=(
                policy.StructuralReferenceSetArgumentView(
                    argument_index=8,
                    optional_logits=structural_reference_set_logits,
                    optional_domain_indices=torch.tensor([30, 31]),
                    required_domain_indices=torch.tensor([32]),
                ),
            ),
            scope=policy.StructuralScopeView(
                optional_logits=structural_scope_logits,
                optional_domain_indices=torch.tensor([40, 41]),
                required_domain_indices=torch.tensor([42]),
            ),
        ),
    }

    resource_log_std = torch.tensor(-0.5, requires_grad=True)
    structural_log_std = torch.tensor(-0.5, requires_grad=True)

    torch.manual_seed(123)

    action = policy.sample_marl_action_bundle(
        records=records,
        resource_payloads=resource_payloads,
        structural_payloads=structural_payloads,
        hard_mask=torch.tensor([False, False]),
        resource_log_std=resource_log_std,
        structural_log_std=structural_log_std,
    )

    log_probs = policy.compute_owner_action_log_probs(
        records=records,
        resource_payloads=resource_payloads,
        structural_payloads=structural_payloads,
        action=action,
        hard_mask=torch.tensor([False, False]),
        possible_agents=("module_0", "module_1", "module_2"),
        resource_log_std=resource_log_std,
        structural_log_std=structural_log_std,
    )

    assert torch.isfinite(log_probs["module_0"])
    assert torch.isfinite(log_probs["module_1"])
    assert log_probs["module_2"].item() == pytest.approx(0.0)

    total = sum(log_probs.values())
    total.backward()

    assert resource_presence.grad is not None
    assert structural_presence.grad is not None

    assert resource_mean.grad is not None
    assert resource_category_logits.grad is not None
    assert resource_log_std.grad is not None

    assert structural_mean.grad is not None
    assert structural_boolean_logit.grad is not None
    assert structural_category_logits.grad is not None
    assert structural_reference_logits.grad is not None
    assert structural_reference_set_logits.grad is not None
    assert structural_scope_logits.grad is not None
    assert structural_log_std.grad is not None


def test_owner_presence_entropy_sums_unmasked_candidates_by_owner():
    import math
    import drl.MARL.policy as policy

    active_0 = torch.tensor(0.0, requires_grad=True)
    active_1 = torch.tensor(0.0, requires_grad=True)
    masked = torch.tensor(0.0, requires_grad=True)

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="resource",
            candidate_id="r0",
            owner_agent_id="module_0",
            participant_agent_ids=(),
            presence_logit=active_0,
        ),
        policy.MARLCandidateRecord(
            candidate_index=1,
            candidate_kind="structural",
            candidate_id="s0",
            owner_agent_id="module_0",
            participant_agent_ids=(),
            presence_logit=active_1,
        ),
        policy.MARLCandidateRecord(
            candidate_index=2,
            candidate_kind="structural",
            candidate_id="masked",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=masked,
        ),
    )

    entropies = policy.compute_owner_presence_entropies(
        records=records,
        hard_mask=torch.tensor([False, False, True]),
        possible_agents=("module_0", "module_1", "module_2"),
    )

    # Bernoulli(logit=0) has entropy log(2).
    assert entropies["module_0"].item() == pytest.approx(
        2.0 * math.log(2.0)
    )
    assert entropies["module_1"].item() == pytest.approx(0.0)
    assert entropies["module_2"].item() == pytest.approx(0.0)

    total = sum(entropies.values())
    total.backward()

    assert active_0.grad is not None
    assert active_1.grad is not None

    # Hard-masked candidate is not part of the policy distribution.
    assert masked.grad is None


def test_owner_resource_payload_entropy_is_weighted_by_presence_probability():
    import math
    import drl.MARL.policy as policy

    presence_logit = torch.tensor(0.0, requires_grad=True)
    masked_presence_logit = torch.tensor(0.0, requires_grad=True)

    categorical_logits = torch.tensor(
        [0.0, 0.0],
        requires_grad=True,
    )
    masked_categorical_logits = torch.tensor(
        [0.0, 0.0],
        requires_grad=True,
    )

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="resource",
            candidate_id="r0",
            owner_agent_id="module_0",
            participant_agent_ids=(),
            presence_logit=presence_logit,
        ),
        policy.MARLCandidateRecord(
            candidate_index=1,
            candidate_kind="resource",
            candidate_id="masked",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=masked_presence_logit,
        ),
    )

    resource_payloads = {
        0: policy.ResourcePayloadView(
            numeric=torch.tensor([0.0]),
            categorical_logits=categorical_logits,
        ),
        1: policy.ResourcePayloadView(
            numeric=torch.tensor([0.0]),
            categorical_logits=masked_categorical_logits,
        ),
    }

    log_std = torch.tensor(0.0, requires_grad=True)

    entropies = policy.compute_owner_resource_payload_entropies(
        records=records,
        resource_payloads=resource_payloads,
        hard_mask=torch.tensor([False, True]),
        possible_agents=("module_0", "module_1", "module_2"),
        log_std=log_std,
    )

    # presence probability = sigmoid(0) = 0.5
    #
    # numeric Normal(sigma=1):
    # H = 0.5 * log(2*pi*e)
    #
    # categorical [0, 0]:
    # H = log(2)
    expected_payload_entropy = (
        0.5 * math.log(2.0 * math.pi * math.e)
        + math.log(2.0)
    )

    assert entropies["module_0"].item() == pytest.approx(
        0.5 * expected_payload_entropy
    )
    assert entropies["module_1"].item() == pytest.approx(0.0)
    assert entropies["module_2"].item() == pytest.approx(0.0)

    total = sum(entropies.values())
    total.backward()

    # Payload entropy is weighted by P(presence=1).
    assert presence_logit.grad is not None

    # Categorical entropy remains differentiable through H2 logits.
    assert categorical_logits.grad is not None

    # Gaussian entropy trains exploration scale.
    assert log_std.grad is not None

    # Hard-masked candidate does not belong to the policy distribution.
    assert masked_presence_logit.grad is None
    assert masked_categorical_logits.grad is None


def test_owner_structural_payload_entropy_is_weighted_by_presence_probability():
    import math
    import drl.MARL.policy as policy

    presence_logit = torch.tensor(0.0, requires_grad=True)
    masked_presence_logit = torch.tensor(0.0, requires_grad=True)

    numeric_mean = torch.tensor([0.0], requires_grad=True)
    boolean_logit = torch.tensor(0.0, requires_grad=True)
    categorical_logits = torch.tensor([0.0, 0.0], requires_grad=True)
    reference_logits = torch.tensor([0.0, 0.0], requires_grad=True)
    reference_set_logits = torch.tensor([0.0, 0.0], requires_grad=True)
    scope_logits = torch.tensor([0.0, 0.0], requires_grad=True)

    masked_boolean_logit = torch.tensor(0.0, requires_grad=True)

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="structural",
            candidate_id="s0",
            owner_agent_id="module_0",
            participant_agent_ids=("module_1",),
            presence_logit=presence_logit,
        ),
        policy.MARLCandidateRecord(
            candidate_index=1,
            candidate_kind="structural",
            candidate_id="masked",
            owner_agent_id="module_1",
            participant_agent_ids=(),
            presence_logit=masked_presence_logit,
        ),
    )

    structural_payloads = {
        0: policy.StructuralPayloadView(
            numeric=(
                policy.StructuralNumericArgumentView(
                    argument_index=4,
                    values=numeric_mean,
                ),
            ),
            boolean=(
                policy.StructuralBooleanArgumentView(
                    argument_index=5,
                    logit=boolean_logit,
                ),
            ),
            categorical=(
                policy.StructuralCategoricalArgumentView(
                    argument_index=6,
                    logits=categorical_logits,
                ),
            ),
            reference=(
                policy.StructuralReferenceArgumentView(
                    argument_index=7,
                    logits=reference_logits,
                    domain_indices=torch.tensor([20, 21]),
                ),
            ),
            reference_set=(
                policy.StructuralReferenceSetArgumentView(
                    argument_index=8,
                    optional_logits=reference_set_logits,
                    optional_domain_indices=torch.tensor([30, 31]),
                    required_domain_indices=torch.tensor([32]),
                ),
            ),
            scope=policy.StructuralScopeView(
                optional_logits=scope_logits,
                optional_domain_indices=torch.tensor([40, 41]),
                required_domain_indices=torch.tensor([42]),
            ),
        ),
        1: policy.StructuralPayloadView(
            boolean=(
                policy.StructuralBooleanArgumentView(
                    argument_index=9,
                    logit=masked_boolean_logit,
                ),
            ),
        ),
    }

    log_std = torch.tensor(0.0, requires_grad=True)

    entropies = policy.compute_owner_structural_payload_entropies(
        records=records,
        structural_payloads=structural_payloads,
        hard_mask=torch.tensor([False, True]),
        possible_agents=("module_0", "module_1", "module_2"),
        log_std=log_std,
    )

    # P(presence=1) = 0.5
    #
    # numeric:       0.5 * log(2*pi*e)
    # boolean:       log(2)
    # categorical:   log(2)
    # reference:     log(2)
    # reference_set: 2 * log(2), optional only
    # scope:         2 * log(2), optional only
    #
    # required members contribute no entropy.
    expected_payload_entropy = (
        0.5 * math.log(2.0 * math.pi * math.e)
        + 7.0 * math.log(2.0)
    )

    assert entropies["module_0"].item() == pytest.approx(
        0.5 * expected_payload_entropy
    )
    assert entropies["module_1"].item() == pytest.approx(0.0)
    assert entropies["module_2"].item() == pytest.approx(0.0)

    total = sum(entropies.values())
    total.backward()

    assert presence_logit.grad is not None

    # Normal entropy depends on sigma, not on its mean.
    assert numeric_mean.grad is None
    assert log_std.grad is not None

    assert boolean_logit.grad is not None
    assert categorical_logits.grad is not None
    assert reference_logits.grad is not None
    assert reference_set_logits.grad is not None
    assert scope_logits.grad is not None

    # Hard-masked candidate is excluded completely.
    assert masked_presence_logit.grad is None
    assert masked_boolean_logit.grad is None


def test_compute_owner_action_entropies_combines_presence_and_payload_terms():
    import math
    import drl.MARL.policy as policy

    presence_logit = torch.tensor(0.0, requires_grad=True)
    resource_logits = torch.tensor([0.0, 0.0], requires_grad=True)

    records = (
        policy.MARLCandidateRecord(
            candidate_index=0,
            candidate_kind="resource",
            candidate_id="r0",
            owner_agent_id="module_0",
            participant_agent_ids=(),
            presence_logit=presence_logit,
        ),
    )

    resource_payloads = {
        0: policy.ResourcePayloadView(
            numeric=torch.tensor([0.0]),
            categorical_logits=resource_logits,
        ),
    }

    resource_log_std = torch.tensor(0.0, requires_grad=True)
    structural_log_std = torch.tensor(0.0, requires_grad=True)

    entropies = policy.compute_owner_action_entropies(
        records=records,
        resource_payloads=resource_payloads,
        structural_payloads={},
        hard_mask=torch.tensor([False]),
        possible_agents=("module_0", "module_1"),
        resource_log_std=resource_log_std,
        structural_log_std=structural_log_std,
    )

    presence_entropy = math.log(2.0)

    payload_entropy = (
        0.5 * math.log(2.0 * math.pi * math.e)
        + math.log(2.0)
    )

    expected = (
        presence_entropy
        + 0.5 * payload_entropy
    )

    assert entropies["module_0"].item() == pytest.approx(expected)
    assert entropies["module_1"].item() == pytest.approx(0.0)

    total = sum(entropies.values())
    total.backward()

    assert presence_logit.grad is not None
    assert resource_logits.grad is not None
    assert resource_log_std.grad is not None

    # No structural numeric payload exists in this sample.
    assert structural_log_std.grad is None
