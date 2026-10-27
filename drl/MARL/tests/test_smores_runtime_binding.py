from __future__ import annotations

from types import SimpleNamespace

import pytest


def make_choice(
    module_id,
    resource_id,
    *,
    action_id="wheel_velocity",
    payload=(1.0, -1.0),
):
    from drl.IL.bc_v1.bc_v1_contract import (
        ActionChoice,
        ModuleRef,
        ResourceRef,
    )

    module = ModuleRef(
        module_id
    )

    return ActionChoice(
        action_id,
        module,
        resource=ResourceRef(
            module,
            resource_id,
        ),
        payload=tuple(payload),
    )


def make_bundle(*choices):
    from drl.IL.bc_v1.action_set import (
        ActionBundle,
    )

    return ActionBundle(
        tuple(choices)
    )


class EchoAdapter:
    def __init__(self):
        self.calls = []

    def decode_action(self, choice):
        self.calls.append(
            choice
        )

        return {
            "command": choice.action_id,
            "module_ids": [
                choice.initiator.module_id
            ],
            "parameters": {
                "payload": list(
                    choice.payload
                ),
            },
        }


def test_binding_preserves_multi_action_bundle_order():
    from drl.MARL.smores_runtime_binding import (
        SMORESMARLRuntimeBinding,
    )

    first = make_choice(
        "module_0",
        "drive",
        payload=(1.0, 2.0),
    )

    second = make_choice(
        "module_1",
        "drive",
        payload=(3.0, 4.0),
    )

    adapter = EchoAdapter()

    binding = SMORESMARLRuntimeBinding(
        adapter=adapter,
        claim_resolver=lambda choice: (
            choice.resource,
        ),
    )

    decoded = binding.decode_action_bundle(
        make_bundle(
            first,
            second,
        )
    )

    assert adapter.calls == [
        first,
        second,
    ]

    assert [
        row.choice
        for row in decoded
    ] == [
        first,
        second,
    ]

    assert [
        row.command["module_ids"]
        for row in decoded
    ] == [
        ["module_0"],
        ["module_1"],
    ]


def test_binding_rejects_conflicting_actionset_as_a_whole():
    from drl.IL.bc_v1.bc_v1_contract import (
        ModuleRef,
        ResourceRef,
    )
    from drl.MARL.smores_runtime_binding import (
        SMORESMARLRuntimeBinding,
    )

    # Two DISTINCT semantic candidates.
    #
    # Each action owns its own direct resource, but both also require
    # one shared physical resource. Therefore each candidate is valid
    # in isolation while the joint ActionSet is physically conflicting.
    first = make_choice(
        "module_0",
        "drive_left",
    )

    second = make_choice(
        "module_0",
        "drive_right",
        payload=(2.0, 2.0),
    )

    shared = ResourceRef(
        ModuleRef("module_0"),
        "shared_drive_bus",
    )

    adapter = EchoAdapter()

    binding = SMORESMARLRuntimeBinding(
        adapter=adapter,
        claim_resolver=lambda choice: (
            choice.resource,
            shared,
        ),
    )

    with pytest.raises(
        ValueError,
        match="Conflicting physical resource claims",
    ):
        binding.decode_action_bundle(
            make_bundle(
                first,
                second,
            )
        )

    # Runtime is only the fail-safe. It must reject the complete sampled
    # set and must not silently choose which candidate is allowed through.
    assert adapter.calls == []


def test_binding_rejects_currently_unavailable_claim():
    from drl.MARL.smores_runtime_binding import (
        SMORESMARLRuntimeBinding,
    )

    choice = make_choice(
        "module_0",
        "drive",
    )

    class BusyAvailability:
        def free(
            self,
            claims,
        ):
            return False

    adapter = EchoAdapter()

    binding = SMORESMARLRuntimeBinding(
        adapter=adapter,
        claim_resolver=lambda item: (
            item.resource,
        ),
    )

    with pytest.raises(
        ValueError,
        match="Physical resource unavailable",
    ):
        binding.decode_action_bundle(
            make_bundle(
                choice
            ),
            availability=BusyAvailability(),
        )

    assert adapter.calls == []


def test_binding_fails_closed_on_malformed_adapter_command():
    from drl.MARL.smores_runtime_binding import (
        SMORESMARLRuntimeBinding,
    )

    choice = make_choice(
        "module_0",
        "drive",
    )

    class BadAdapter:
        def decode_action(
            self,
            choice,
        ):
            return {
                "command": "wheel_velocity",
                # deliberately missing module_ids / parameters
            }

    binding = SMORESMARLRuntimeBinding(
        adapter=BadAdapter(),
        claim_resolver=lambda item: (
            item.resource,
        ),
    )

    with pytest.raises(
        ValueError,
        match="decoded SMORES command",
    ):
        binding.decode_action_bundle(
            make_bundle(
                choice
            )
        )


def test_binding_can_use_existing_actionset_decoder_contract():
    from drl.MARL.smores_runtime_binding import (
        SMORESMARLRuntimeBinding,
    )

    choice = make_choice(
        "module_0",
        "drive",
    )

    bundle = make_bundle(
        choice
    )

    calls = []

    def decoder(
        prediction,
        batch,
        normalizers,
        *,
        configured_arguments=None,
        threshold=0.0,
    ):
        calls.append(
            (
                prediction,
                batch,
                normalizers,
                configured_arguments,
                threshold,
            )
        )

        return (
            bundle,
        )

    binding = SMORESMARLRuntimeBinding(
        adapter=EchoAdapter(),
        claim_resolver=lambda item: (
            item.resource,
        ),
        action_set_decoder=decoder,
    )

    prediction = object()
    batch = object()
    normalizers = object()

    decoded = (
        binding.decode_action_set_prediction(
            prediction,
            batch,
            normalizers,
            configured_arguments={
                ("foo", "bar"): 1,
            },
            threshold=0.25,
        )
    )

    assert len(
        calls
    ) == 1

    assert calls[0] == (
        prediction,
        batch,
        normalizers,
        {
            ("foo", "bar"): 1,
        },
        0.25,
    )

    assert len(
        decoded
    ) == 1

    assert (
        decoded[0].choice
        is choice
    )


def test_binding_uses_real_smores_adapter_for_wheel_velocity():
    import importlib

    from drl.IL.bc_v1.bc_v1_contract import (
        ActionChoice,
        ModuleRef,
    )
    from drl.IL.bc_v1.smores_adapter import (
        SMORESAdapter,
    )
    from drl.MARL.smores_runtime_binding import (
        SMORESMARLRuntimeBinding,
    )

    # Reuse the repository-certified RobotSpec loader already used by
    # the real ActionSet tests rather than rebuilding a fake spec here.
    source_module = importlib.import_module(
        "drl.IL.tests.bc_v1."
        "test_action_set_candidate_universe"
    )

    sources = source_module.load_sources()
    spec = sources["kwargs"]["spec"]

    adapter = SMORESAdapter(
        spec
    )

    module = ModuleRef(
        "module_0"
    )

    # Use the adapter's physical RobotSpec mapping only to create the
    # correctly typed configured resource reference for this smoke.
    target = adapter._resource_target(
        "module_0",
        "wheel_velocity",
        (
            1.25,
            -0.75,
        ),
        "rad/s",
    )

    choice = ActionChoice(
        "wheel_velocity",
        module,
        resource=target.resource,
        payload=target.payload,
    )

    binding = SMORESMARLRuntimeBinding(
        adapter=adapter,
        claim_resolver=lambda item: (
            item.resource,
        ),
    )

    decoded = binding.decode_action_bundle(
        make_bundle(
            choice
        )
    )

    assert len(
        decoded
    ) == 1

    assert decoded[0].command == {
        "command": "wheel_velocity",
        "module_ids": [
            "module_0"
        ],
        "parameters": {
            "left_rad_s": 1.25,
            "right_rad_s": -0.75,
        },
    }


def test_actor_visibility_projector_delegates_exact_runtime_context():
    from drl.MARL.smores_runtime_binding import (
        SMORESActorVisibilityProjector,
    )

    view = object()
    policy = object()
    schema = object()
    certificate = object()

    calls = []

    def assigned_goal_provider(
        received_view,
    ):
        assert (
            received_view
            is view
        )

        return certificate

    def projector(
        received_view,
        task,
        received_policy,
        received_schema,
        *,
        assigned_goal=None,
    ):
        calls.append(
            (
                received_view,
                task,
                received_policy,
                received_schema,
                assigned_goal,
            )
        )

        return "actor-view"

    runtime_projector = (
        SMORESActorVisibilityProjector(
            task="locomotion",
            policy=policy,
            schema=schema,
            assigned_goal_provider=(
                assigned_goal_provider
            ),
            projector=projector,
        )
    )

    assert (
        runtime_projector(
            view
        )
        == "actor-view"
    )

    assert calls == [
        (
            view,
            "locomotion",
            policy,
            schema,
            certificate,
        )
    ]


# --- MARL ACTIONSET MATERIALIZER ---

def _materializer_fixture():
    import torch

    prediction = {
        "presence_logits": torch.tensor(
            [9.0, 9.0, 9.0, 9.0, 9.0]
        ),
        "candidate_graph": torch.tensor(
            [0, 0, 0, 0, 0],
            dtype=torch.long,
        ),
        "parameters": {
            "resource": {
                "payload": torch.tensor(
                    [9.0, 9.0]
                ),
                "payload_candidate": torch.tensor(
                    [0],
                    dtype=torch.long,
                ),
                "payload_ptr": torch.tensor(
                    [0, 2],
                    dtype=torch.long,
                ),
                "categorical": {
                    "candidate": torch.tensor(
                        [1],
                        dtype=torch.long,
                    ),
                    "ptr": torch.tensor(
                        [0, 3],
                        dtype=torch.long,
                    ),
                    "logits": torch.tensor(
                        [9.0, 9.0, 9.0]
                    ),
                },
            },
            "structural": {
                "arguments": {
                    "numeric": {
                        "argument": torch.tensor(
                            [0],
                            dtype=torch.long,
                        ),
                        "value": torch.tensor(
                            [9.0, 9.0]
                        ),
                        "ptr": torch.tensor(
                            [0, 2],
                            dtype=torch.long,
                        ),
                    },
                    "boolean": {
                        "argument": torch.tensor(
                            [1],
                            dtype=torch.long,
                        ),
                        "logits": torch.tensor(
                            [9.0]
                        ),
                    },
                    "category": {
                        "argument": torch.tensor(
                            [2],
                            dtype=torch.long,
                        ),
                        "ptr": torch.tensor(
                            [0, 3],
                            dtype=torch.long,
                        ),
                        "logits": torch.tensor(
                            [9.0, 9.0, 9.0]
                        ),
                    },
                    "reference": {
                        "argument": torch.tensor(
                            [3],
                            dtype=torch.long,
                        ),
                        "ptr": torch.tensor(
                            [0, 3],
                            dtype=torch.long,
                        ),
                        "logits": torch.tensor(
                            [9.0, 9.0, 9.0]
                        ),
                        "domain_index": torch.tensor(
                            [10, 11, 12],
                            dtype=torch.long,
                        ),
                    },
                    "reference_set": {
                        "argument": torch.tensor(
                            [4],
                            dtype=torch.long,
                        ),
                        "ptr": torch.tensor(
                            [0, 2],
                            dtype=torch.long,
                        ),
                        "logits": torch.tensor(
                            [9.0, 9.0]
                        ),
                        "domain_index": torch.tensor(
                            [20, 22],
                            dtype=torch.long,
                        ),
                        "required_ptr": torch.tensor(
                            [0, 1],
                            dtype=torch.long,
                        ),
                        "required_domain_index": torch.tensor(
                            [21],
                            dtype=torch.long,
                        ),
                    },
                },
                "scope": {
                    "binding": torch.tensor(
                        [2],
                        dtype=torch.long,
                    ),
                    "logits": torch.tensor(
                        [9.0, 9.0]
                    ),
                    "ptr": torch.tensor(
                        [0, 2],
                        dtype=torch.long,
                    ),
                    "domain_index": torch.tensor(
                        [30, 32],
                        dtype=torch.long,
                    ),
                    "required_ptr": torch.tensor(
                        [0, 1],
                        dtype=torch.long,
                    ),
                    "required_domain_index": torch.tensor(
                        [31],
                        dtype=torch.long,
                    ),
                },
            },
        },
    }

    routing = {
        "resource_candidates": [
            object(),
            object(),
        ],
        "structural_bindings": [
            object(),
            object(),
            object(),
        ],
    }

    batch = {
        "structural": {
            # argument 0,1 belong to structural binding 0
            # argument 2,3 belong to structural binding 1
            # argument 4   belongs to structural binding 2
            "argument_binding": torch.tensor(
                [0, 0, 1, 1, 2],
                dtype=torch.long,
            ),
        },
    }

    return prediction, routing, batch


def _full_sampled_marl_action():
    import torch

    from drl.MARL.policy import (
        MARLActionBundle,
    )

    return MARLActionBundle(
        presence=torch.tensor(
            [1.0, 1.0, 1.0, 1.0, 1.0]
        ),

        resource_numeric={
            0: torch.tensor(
                [1.5, -2.0]
            ),
        },

        resource_categorical={
            1: torch.tensor(
                2
            ),
        },

        structural_numeric={
            2: {
                0: torch.tensor(
                    [3.0, 4.0]
                ),
            },
        },

        structural_boolean={
            2: {
                1: torch.tensor(
                    1.0
                ),
            },
        },

        structural_categorical={
            3: {
                2: torch.tensor(
                    1
                ),
            },
        },

        structural_reference={
            3: {
                3: torch.tensor(
                    11
                ),
            },
        },

        structural_reference_set={
            4: {
                4: torch.tensor(
                    [22, 21],
                    dtype=torch.long,
                ),
            },
        },

        structural_scope={
            4: torch.tensor(
                [32, 31],
                dtype=torch.long,
            ),
        },
    )


def test_materializer_preserves_exact_sampled_presence_and_all_payload_families():
    import torch

    from drl.MARL.smores_runtime_binding import (
        materialize_marl_action_set_prediction,
    )

    template, routing, batch = (
        _materializer_fixture()
    )

    action = _full_sampled_marl_action()

    result = materialize_marl_action_set_prediction(
        action=action,
        prediction_template=template,
        routing=routing,
        batch=batch,
    )

    # Membership is no longer probabilistic here.
    torch.testing.assert_close(
        result["presence_logits"],
        torch.ones(5),
    )

    # Candidate graph/routing identity is retained unchanged.
    torch.testing.assert_close(
        result["candidate_graph"],
        template["candidate_graph"],
    )

    resource = (
        result["parameters"]["resource"]
    )

    torch.testing.assert_close(
        resource["payload"],
        torch.tensor(
            [1.5, -2.0]
        ),
    )

    torch.testing.assert_close(
        resource[
            "categorical"
        ]["logits"],
        torch.tensor(
            [-1.0, -1.0, 1.0]
        ),
    )

    arguments = (
        result["parameters"]
        ["structural"]["arguments"]
    )

    torch.testing.assert_close(
        arguments[
            "numeric"
        ]["value"],
        torch.tensor(
            [3.0, 4.0]
        ),
    )

    torch.testing.assert_close(
        arguments[
            "boolean"
        ]["logits"],
        torch.tensor(
            [1.0]
        ),
    )

    torch.testing.assert_close(
        arguments[
            "category"
        ]["logits"],
        torch.tensor(
            [-1.0, 1.0, -1.0]
        ),
    )

    torch.testing.assert_close(
        arguments[
            "reference"
        ]["logits"],
        torch.tensor(
            [-1.0, 1.0, -1.0]
        ),
    )

    # domain 22 selected, optional domain 20 not selected.
    # domain 21 is required and therefore has no learned logit.
    torch.testing.assert_close(
        arguments[
            "reference_set"
        ]["logits"],
        torch.tensor(
            [-1.0, 1.0]
        ),
    )

    scope = (
        result["parameters"]
        ["structural"]["scope"]
    )

    # domain 32 selected, domain 30 yielded.
    # domain 31 is mandatory routing-only membership.
    torch.testing.assert_close(
        scope["logits"],
        torch.tensor(
            [-1.0, 1.0]
        ),
    )

    # Materialization must not mutate the actor's original H2 output.
    torch.testing.assert_close(
        template["presence_logits"],
        torch.full(
            (5,),
            9.0,
        ),
    )

    torch.testing.assert_close(
        template[
            "parameters"
        ]["resource"]["payload"],
        torch.tensor(
            [9.0, 9.0]
        ),
    )


def test_materializer_presence_zero_is_deterministic_implicit_yield():
    import torch

    from drl.MARL.policy import (
        MARLActionBundle,
    )
    from drl.MARL.smores_runtime_binding import (
        materialize_marl_action_set_prediction,
    )

    template, routing, batch = (
        _materializer_fixture()
    )

    action = MARLActionBundle(
        presence=torch.zeros(5),
        resource_numeric={},
        resource_categorical={},
        structural_numeric={},
        structural_boolean={},
        structural_categorical={},
        structural_reference={},
        structural_reference_set={},
        structural_scope={},
    )

    result = materialize_marl_action_set_prediction(
        action=action,
        prediction_template=template,
        routing=routing,
        batch=batch,
    )

    torch.testing.assert_close(
        result["presence_logits"],
        -torch.ones(5),
    )


def test_materializer_rejects_payload_for_absent_candidate():
    import pytest
    import torch

    from drl.MARL.policy import (
        MARLActionBundle,
    )
    from drl.MARL.smores_runtime_binding import (
        materialize_marl_action_set_prediction,
    )

    template, routing, batch = (
        _materializer_fixture()
    )

    action = MARLActionBundle(
        presence=torch.zeros(5),

        # Candidate 0 explicitly yielded, so a payload here is inconsistent.
        resource_numeric={
            0: torch.tensor(
                [1.0, 2.0]
            ),
        },

        resource_categorical={},
        structural_numeric={},
        structural_boolean={},
        structural_categorical={},
        structural_reference={},
        structural_reference_set={},
        structural_scope={},
    )

    with pytest.raises(
        ValueError,
        match="absent candidate",
    ):
        materialize_marl_action_set_prediction(
            action=action,
            prediction_template=template,
            routing=routing,
            batch=batch,
        )


def test_materializer_rejects_missing_payload_for_active_candidate():
    import pytest
    import torch

    from drl.MARL.policy import (
        MARLActionBundle,
    )
    from drl.MARL.smores_runtime_binding import (
        materialize_marl_action_set_prediction,
    )

    template, routing, batch = (
        _materializer_fixture()
    )

    action = MARLActionBundle(
        # Candidate 0 is active and is a width-2 numeric resource candidate.
        presence=torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 0.0]
        ),
        resource_numeric={},
        resource_categorical={},
        structural_numeric={},
        structural_boolean={},
        structural_categorical={},
        structural_reference={},
        structural_reference_set={},
        structural_scope={},
    )

    with pytest.raises(
        ValueError,
        match="Missing sampled resource numeric payload",
    ):
        materialize_marl_action_set_prediction(
            action=action,
            prediction_template=template,
            routing=routing,
            batch=batch,
        )


def test_materializer_rejects_reference_selection_outside_current_domain():
    import pytest
    import torch

    from drl.MARL.policy import (
        MARLActionBundle,
    )
    from drl.MARL.smores_runtime_binding import (
        materialize_marl_action_set_prediction,
    )

    template, routing, batch = (
        _materializer_fixture()
    )

    action = MARLActionBundle(
        presence=torch.tensor(
            [0.0, 0.0, 0.0, 1.0, 0.0]
        ),
        resource_numeric={},
        resource_categorical={},
        structural_numeric={},
        structural_boolean={},
        structural_categorical={
            3: {
                2: torch.tensor(0),
            },
        },
        structural_reference={
            3: {
                # Current routed domain is {10, 11, 12}.
                3: torch.tensor(999),
            },
        },
        structural_reference_set={},
        structural_scope={},
    )

    with pytest.raises(
        ValueError,
        match="reference selection is outside current domain",
    ):
        materialize_marl_action_set_prediction(
            action=action,
            prediction_template=template,
            routing=routing,
            batch=batch,
        )


def test_binding_decodes_materialized_sample_without_resampling():
    from drl.IL.bc_v1.action_set import (
        ActionBundle,
    )
    from drl.MARL.smores_runtime_binding import (
        SMORESMARLRuntimeBinding,
    )

    template, routing, batch = (
        _materializer_fixture()
    )

    action = _full_sampled_marl_action()

    decoder_calls = []

    def decoder(
        prediction,
        received_batch,
        normalizers,
        *,
        configured_arguments=None,
        threshold=0.0,
    ):
        decoder_calls.append(
            (
                prediction,
                received_batch,
                normalizers,
                configured_arguments,
                threshold,
            )
        )

        # This fake decoder is deliberately deterministic.
        # It exists only to prove the binding forwards the already
        # materialized MARL sample instead of invoking another policy.
        return (
            ActionBundle(()),
        )

    binding = SMORESMARLRuntimeBinding(
        adapter=EchoAdapter(),
        claim_resolver=lambda choice: (
            choice.resource,
        ),
        action_set_decoder=decoder,
    )

    normalizers = object()

    decoded = binding.decode_marl_action(
        action=action,
        prediction_template=template,
        routing=routing,
        batch=batch,
        normalizers=normalizers,
    )

    assert decoded == ()
    assert len(decoder_calls) == 1

    prediction = decoder_calls[0][0]

    assert decoder_calls[0][1] is batch
    assert decoder_calls[0][2] is normalizers
    assert decoder_calls[0][4] == 0.0

    # Verify a few independent sampled decisions reached the decoder.
    assert (
        prediction[
            "presence_logits"
        ].tolist()
        == [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]
    )

    assert (
        prediction[
            "parameters"
        ]["resource"]["payload"].tolist()
        == [
            1.5,
            -2.0,
        ]
    )

    assert (
        prediction[
            "parameters"
        ]["structural"]
        ["arguments"]
        ["reference"]
        ["logits"]
        .tolist()
        == [
            -1.0,
            1.0,
            -1.0,
        ]
    )


# --- REAL F2 MARL ACTIONSET ROUNDTRIP ---

def test_real_f2_roundtrip_preserves_sampled_normalized_resource_payload():
    import importlib

    import torch

    from drl.IL.bc_v1.action_set_decoder import (
        decode_action_sets,
    )
    from drl.MARL.policy import (
        MARLActionBundle,
    )
    from drl.MARL.smores_runtime_binding import (
        materialize_marl_action_set_prediction,
    )

    source = importlib.import_module(
        "drl.IL.tests.bc_v1."
        "test_action_set_candidate_universe"
    )

    # Reuse exactly the repository-certified TRAIN fixture used by
    # the ActionSet decoder/model integration tests.
    real_cases = source.real_cases.__wrapped__()
    normalizers = source.load_sources()[
        "kwargs"
    ]["normalizers"]

    def clone_tree(value):
        if torch.is_tensor(value):
            return value.detach().clone()

        if isinstance(value, dict):
            return {
                key: clone_tree(item)
                for key, item in value.items()
            }

        if isinstance(value, tuple):
            return tuple(
                clone_tree(item)
                for item in value
            )

        if isinstance(value, list):
            return [
                clone_tree(item)
                for item in value
            ]

        return value

    selected = None

    # Search the existing real cases for one numeric resource candidate
    # whose zero-normalized F2 value is physically decodable.
    #
    # This avoids assuming a particular candidate id or action name while
    # still exercising the real frozen F2 inverse transform.
    for case_index in (
        0,
        3063,
        9260,
    ):
        row, batch, prediction = real_cases[
            case_index
        ]

        resource = prediction[
            "parameters"
        ]["resource"]

        candidates = (
            resource[
                "payload_candidate"
            ]
            .detach()
            .cpu()
            .tolist()
        )

        ptr = (
            resource[
                "payload_ptr"
            ]
            .detach()
            .cpu()
            .tolist()
        )

        for numeric_row, candidate_raw in enumerate(
            candidates
        ):
            candidate = int(
                candidate_raw
            )

            lo = ptr[
                numeric_row
            ]

            hi = ptr[
                numeric_row + 1
            ]

            if hi <= lo:
                continue

            sample = torch.zeros_like(
                resource[
                    "payload"
                ][lo:hi]
            )

            reference = clone_tree(
                prediction
            )

            reference[
                "presence_logits"
            ].fill_(-1.0)

            reference[
                "presence_logits"
            ][candidate] = 1.0

            reference[
                "parameters"
            ]["resource"]["payload"][
                lo:hi
            ] = sample

            try:
                expected = decode_action_sets(
                    reference,
                    batch,
                    normalizers,
                    threshold=0.0,
                )[0]
            except ValueError:
                continue

            if len(
                expected.actions
            ) != 1:
                continue

            selected = (
                row,
                batch,
                prediction,
                candidate,
                sample,
                expected,
            )

            break

        if selected is not None:
            break

    assert selected is not None, (
        "Real fixture contains no decodable numeric "
        "resource candidate"
    )

    (
        row,
        batch,
        prediction,
        candidate,
        sample,
        expected,
    ) = selected

    total = prediction[
        "presence_logits"
    ].numel()

    presence = torch.zeros(
        total,
        dtype=prediction[
            "presence_logits"
        ].dtype,
        device=prediction[
            "presence_logits"
        ].device,
    )

    presence[
        candidate
    ] = 1.0

    action = MARLActionBundle(
        presence=presence,
        resource_numeric={
            candidate: sample,
        },
        resource_categorical={},
        structural_numeric={},
        structural_boolean={},
        structural_categorical={},
        structural_reference={},
        structural_reference_set={},
        structural_scope={},
    )

    materialized = (
        materialize_marl_action_set_prediction(
            action=action,
            prediction_template=prediction,
            routing=row[
                "routing"
            ],
            # IMPORTANT: this is the REAL full collated decoder batch,
            # not the synthetic batch["structural"] fixture used earlier.
            batch=batch,
        )
    )

    actual = decode_action_sets(
        materialized,
        batch,
        normalizers,
        threshold=0.0,
    )[0]

    # Same semantic/physical ActionChoice after the real frozen F2 inverse.
    assert actual == expected
    assert len(
        actual.actions
    ) == 1

    # And the materializer itself preserved exactly the normalized
    # sample supplied by MARL.
    resource = materialized[
        "parameters"
    ]["resource"]

    rows = (
        resource[
            "payload_candidate"
        ]
        .detach()
        .cpu()
        .tolist()
    )

    row_index = rows.index(
        candidate
    )

    ptr = (
        resource[
            "payload_ptr"
        ]
        .detach()
        .cpu()
        .tolist()
    )

    lo = ptr[
        row_index
    ]

    hi = ptr[
        row_index + 1
    ]

    torch.testing.assert_close(
        resource[
            "payload"
        ][lo:hi],
        sample,
    )


# --- ONLINE DECISION TO SMORES RUNTIME BRIDGE ---

def test_decision_bridge_consumes_exact_sample_template_and_routing_once():
    import torch
    from types import SimpleNamespace

    from drl.MARL.smores_runtime_binding import (
        SMORESMARLDecisionBridge,
    )

    action = object()
    template = {
        "presence_logits": torch.tensor(
            [1.0]
        ),
    }

    routing = {
        "resource_candidates": [
            "candidate-0",
        ],
        "structural_bindings": [],
    }

    actor_graph = {
        "module_x": torch.tensor(
            [[1.0, 2.0]]
        ),
    }

    context = SimpleNamespace(
        actor_graph=actor_graph,
        critic_graph={
            "privileged": True,
        },
        routing_snapshot=routing,
        hard_mask=torch.tensor(
            [False],
            dtype=torch.bool,
        ),
    )

    decision = SimpleNamespace(
        action=action,
        action_set_prediction_template=template,
    )

    normalizers = object()

    binding_calls = []
    decoded = (
        SimpleNamespace(
            command={
                "command": "wheel_velocity",
                "module_ids": [
                    "module_0"
                ],
                "parameters": {
                    "left_rad_s": 1.0,
                    "right_rad_s": 2.0,
                },
            }
        ),
    )

    class FakeBinding:
        def decode_marl_action(
            self,
            **kwargs,
        ):
            binding_calls.append(
                kwargs
            )
            return decoded

    dispatch_calls = []

    def dispatcher(
        *,
        env_index,
        decoded_actions,
        context,
    ):
        dispatch_calls.append(
            (
                env_index,
                decoded_actions,
                context,
            )
        )

    placeholder_calls = []

    def placeholder_action(
        env_index,
        received_context,
    ):
        placeholder_calls.append(
            (
                env_index,
                received_context,
            )
        )

        return {
            "module_0": torch.zeros(
                2
            ),
        }

    bridge = SMORESMARLDecisionBridge(
        binding=FakeBinding(),
        normalizers=normalizers,
        dispatcher=dispatcher,
        placeholder_action=placeholder_action,
    )

    result = bridge(
        decision,
        3,
        context,
    )

    assert len(
        binding_calls
    ) == 1

    call = binding_calls[0]

    assert (
        call["action"]
        is action
    )

    assert (
        call[
            "prediction_template"
        ]
        is template
    )

    assert (
        call["routing"]
        is routing
    )

    assert (
        call["normalizers"]
        is normalizers
    )

    # Default batch construction must reuse the SAME actor input and
    # routing snapshot. It is ABI packaging, not feature recomputation.
    assert (
        call["batch"]["inputs"]
        is actor_graph
    )

    assert (
        call["batch"]["routing"][0]
        is routing
    )

    assert len(
        call["batch"]["routing"]
    ) == 1

    assert dispatch_calls == [
        (
            3,
            decoded,
            context,
        )
    ]

    assert placeholder_calls == [
        (
            3,
            context,
        )
    ]

    torch.testing.assert_close(
        result["module_0"],
        torch.zeros(2),
    )


def test_decision_bridge_never_reruns_actor_or_resamples():
    import torch
    from types import SimpleNamespace

    from drl.MARL.smores_runtime_binding import (
        SMORESMARLDecisionBridge,
    )

    sentinel_action = object()
    sentinel_template = object()

    context = SimpleNamespace(
        actor_graph={
            "already_computed": True,
        },
        critic_graph={},
        routing_snapshot={
            "resource_candidates": [],
            "structural_bindings": [],
        },
        hard_mask=torch.empty(
            0,
            dtype=torch.bool,
        ),
    )

    decision = SimpleNamespace(
        action=sentinel_action,
        action_set_prediction_template=(
            sentinel_template
        ),
    )

    class FakeBinding:
        def __init__(self):
            self.calls = 0

        def decode_marl_action(
            self,
            **kwargs,
        ):
            self.calls += 1

            assert (
                kwargs["action"]
                is sentinel_action
            )

            assert (
                kwargs[
                    "prediction_template"
                ]
                is sentinel_template
            )

            return ()

    binding = FakeBinding()

    dispatched = []

    bridge = SMORESMARLDecisionBridge(
        binding=binding,
        normalizers=object(),
        dispatcher=lambda **kwargs: (
            dispatched.append(
                kwargs
            )
        ),
        placeholder_action=lambda *_: {
            "module_0": torch.zeros(1),
        },
    )

    result = bridge(
        decision,
        0,
        context,
    )

    assert binding.calls == 1
    assert len(
        dispatched
    ) == 1

    torch.testing.assert_close(
        result["module_0"],
        torch.zeros(1),
    )


def test_decision_bridge_fails_closed_without_prediction_template():
    import pytest
    import torch
    from types import SimpleNamespace

    from drl.MARL.smores_runtime_binding import (
        SMORESMARLDecisionBridge,
    )

    binding_called = []
    dispatch_called = []
    placeholder_called = []

    class FakeBinding:
        def decode_marl_action(
            self,
            **kwargs,
        ):
            binding_called.append(
                kwargs
            )
            return ()

    bridge = SMORESMARLDecisionBridge(
        binding=FakeBinding(),
        normalizers=object(),
        dispatcher=lambda **kwargs: (
            dispatch_called.append(
                kwargs
            )
        ),
        placeholder_action=lambda *_: (
            placeholder_called.append(
                True
            )
            or {
                "module_0": torch.zeros(
                    1
                )
            }
        ),
    )

    decision = SimpleNamespace(
        action=object(),
        action_set_prediction_template=None,
    )

    context = SimpleNamespace(
        actor_graph={},
        critic_graph={},
        routing_snapshot={
            "resource_candidates": [],
            "structural_bindings": [],
        },
        hard_mask=torch.empty(
            0,
            dtype=torch.bool,
        ),
    )

    with pytest.raises(
        ValueError,
        match="prediction template",
    ):
        bridge(
            decision,
            0,
            context,
        )

    assert binding_called == []
    assert dispatch_called == []
    assert placeholder_called == []


def test_decision_bridge_dispatch_failure_never_returns_step_placeholder():
    import pytest
    import torch
    from types import SimpleNamespace

    from drl.MARL.smores_runtime_binding import (
        SMORESMARLDecisionBridge,
    )

    decoded = (
        SimpleNamespace(
            command={
                "command": "dock",
                "module_ids": [
                    "module_0",
                    "module_1",
                ],
                "parameters": {},
            }
        ),
    )

    class FakeBinding:
        def decode_marl_action(
            self,
            **kwargs,
        ):
            return decoded

    placeholder_called = []

    def fail_dispatch(
        **kwargs,
    ):
        raise RuntimeError(
            "native owner rejected command"
        )

    bridge = SMORESMARLDecisionBridge(
        binding=FakeBinding(),
        normalizers=object(),
        dispatcher=fail_dispatch,
        placeholder_action=lambda *_: (
            placeholder_called.append(
                True
            )
            or {
                "module_0": torch.zeros(
                    1
                )
            }
        ),
    )

    decision = SimpleNamespace(
        action=object(),
        action_set_prediction_template={
            "presence_logits": torch.tensor(
                [1.0]
            ),
        },
    )

    context = SimpleNamespace(
        actor_graph={},
        critic_graph={},
        routing_snapshot={
            "resource_candidates": [],
            "structural_bindings": [],
        },
        hard_mask=torch.empty(
            0,
            dtype=torch.bool,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="native owner rejected command",
    ):
        bridge(
            decision,
            0,
            context,
        )

    # A failed physical staging must not be reported to the environment
    # as a successfully accepted MARL step.
    assert placeholder_called == []


def test_decision_bridge_supports_runtime_availability_and_configured_arguments():
    import torch
    from types import SimpleNamespace

    from drl.MARL.smores_runtime_binding import (
        SMORESMARLDecisionBridge,
    )

    availability = object()

    configured = {
        (
            "dock",
            "clocking_quarter_turns",
        ): 2,
    }

    context = SimpleNamespace(
        actor_graph={
            "x": torch.ones(
                1
            ),
        },
        critic_graph={},
        routing_snapshot={
            "resource_candidates": [],
            "structural_bindings": [],
        },
        hard_mask=torch.empty(
            0,
            dtype=torch.bool,
        ),
    )

    decision = SimpleNamespace(
        action=object(),
        action_set_prediction_template={
            "presence_logits": torch.empty(
                0
            ),
        },
    )

    calls = []

    class FakeBinding:
        def decode_marl_action(
            self,
            **kwargs,
        ):
            calls.append(
                kwargs
            )
            return ()

    bridge = SMORESMARLDecisionBridge(
        binding=FakeBinding(),
        normalizers=object(),
        dispatcher=lambda **kwargs: None,
        placeholder_action=lambda *_: {
            "module_0": torch.zeros(
                1
            ),
        },
        availability_provider=(
            lambda env_index, received_context:
            availability
        ),
        configured_arguments_provider=(
            lambda env_index, received_context:
            configured
        ),
    )

    bridge(
        decision,
        2,
        context,
    )

    assert len(
        calls
    ) == 1

    assert (
        calls[0]["availability"]
        is availability
    )

    assert (
        calls[0][
            "configured_arguments"
        ]
        is configured
    )


# --- NATIVE SMORES DISPATCHER ---

def _native_decoded(command, module_ids, parameters):
    from types import SimpleNamespace

    return SimpleNamespace(
        command={
            "command": command,
            "module_ids": list(
                module_ids
            ),
            "parameters": dict(
                parameters
            ),
        }
    )


def _native_dispatcher_fixture(
    *,
    reject_on_submit=None,
):
    from dataclasses import dataclass
    from types import SimpleNamespace

    from drl.MARL.smores_runtime_binding import (
        SMORESNativeDispatcher,
    )

    @dataclass(frozen=True)
    class FakeCommand:
        linear_x_m_s: float = 0.0
        angular_z_rad_s: float = 0.0
        pan_target_rad: float = 0.0
        tilt_target_rad: float = 0.0
        internal_motion: str = "passive"
        pan_velocity_rad_s: float = 0.0
        reset_internal_targets: bool = False

    class FakeGoal:
        def __init__(
            self,
            payload,
        ):
            self.goal_id = payload[
                "goal_id"
            ]
            self.primitive = payload[
                "primitive"
            ]
            self.module_ids = tuple(
                payload[
                    "module_ids"
                ]
            )
            self.parameters = dict(
                payload.get(
                    "parameters",
                    {},
                )
            )
            self.timeout_s = float(
                payload.get(
                    "timeout_s",
                    30.0,
                )
            )

    class FakeExecutor:
        def __init__(self):
            self.submitted = []
            self.canceled = []

        def submit(
            self,
            goal,
            now_s,
        ):
            self.submitted.append(
                (
                    goal,
                    now_s,
                )
            )

            index = len(
                self.submitted
            )

            if (
                reject_on_submit
                is not None
                and index
                == reject_on_submit
            ):
                return SimpleNamespace(
                    state="rejected",
                    code="RESOURCE_BUSY",
                    message=(
                        "test rejection"
                    ),
                )

            return SimpleNamespace(
                state="accepted",
                code="GOAL_ACCEPTED",
                message="accepted",
            )

        def cancel(
            self,
            goal_id,
            now_s,
        ):
            self.canceled.append(
                (
                    goal_id,
                    now_s,
                )
            )

            return SimpleNamespace(
                state="canceled",
            )

    executor = FakeExecutor()

    dispatcher = SMORESNativeDispatcher(
        executor_provider=lambda env_index: executor,
        now_provider=lambda env_index: 12.5,
        geometry=SimpleNamespace(
            wheel_radius_m=0.04,
            track_width_m=0.16,
        ),
        command_factory=FakeCommand,
        pan_velocity_mode="pan_velocity",
        primitive_goal_factory=lambda payload: (
            FakeGoal(
                payload
            )
        ),
        accepted_state="accepted",
    )

    return (
        dispatcher,
        executor,
        FakeCommand,
    )


def test_decision_decode_rejection_revokes_previous_continuous_commands():
    from drl.MARL.smores_runtime_binding import SMORESMARLDecisionBridge

    dispatcher, _, _ = _native_dispatcher_fixture()
    for env_index in (0, 1):
        dispatcher(
            env_index=env_index,
            decoded_actions=(
                _native_decoded("wheel_velocity", ("module_0",),
                                {"left_rad_s": 1.0, "right_rad_s": 1.0}),
            ),
            context=None,
        )

    class RejectingBinding:
        def decode_marl_action(self, **kwargs):
            raise ValueError("sampled ActionSet conflicts")

    bridge = SMORESMARLDecisionBridge(
        binding=RejectingBinding(),
        normalizers=object(),
        dispatcher=dispatcher,
        placeholder_action=lambda *_: pytest.fail("rejected action reached physics"),
        decoder_batch_builder=lambda *_: {"inputs": {}, "routing": {}},
    )
    decision = SimpleNamespace(action=object(), action_set_prediction_template={})
    context = SimpleNamespace(routing_snapshot={})

    with pytest.raises(ValueError, match="sampled ActionSet conflicts"):
        bridge(decision, 0, context)

    assert dispatcher.continuous_commands(0) == {}
    assert dispatcher.continuous_commands(1)


def test_native_dispatcher_combines_wheel_and_pan_for_same_module():
    import pytest

    dispatcher, executor, _ = (
        _native_dispatcher_fixture()
    )

    dispatcher(
        env_index=0,
        decoded_actions=(
            _native_decoded(
                "wheel_velocity",
                ("module_0",),
                {
                    "left_rad_s": 1.0,
                    "right_rad_s": 3.0,
                },
            ),
            _native_decoded(
                "pan_velocity",
                ("module_0",),
                {
                    "rate_rad_s": 0.75,
                },
            ),
        ),
        context=None,
    )

    commands = (
        dispatcher.continuous_commands(
            0
        )
    )

    assert set(
        commands
    ) == {
        "module_0",
    }

    command = commands[
        "module_0"
    ]

    # v = r * (wl + wr) / 2
    assert command.linear_x_m_s == pytest.approx(
        0.08
    )

    # w = r * (wr - wl) / track
    assert command.angular_z_rad_s == pytest.approx(
        0.5
    )

    assert (
        command.internal_motion
        == "pan_velocity"
    )

    assert command.pan_velocity_rad_s == pytest.approx(
        0.75
    )

    assert executor.submitted == []


def test_native_dispatcher_empty_proposal_clears_previous_continuous_commands():
    dispatcher, _, _ = (
        _native_dispatcher_fixture()
    )

    dispatcher(
        env_index=0,
        decoded_actions=(
            _native_decoded(
                "wheel_velocity",
                ("module_0",),
                {
                    "left_rad_s": 2.0,
                    "right_rad_s": 2.0,
                },
            ),
        ),
        context=None,
    )

    assert (
        dispatcher.continuous_commands(
            0
        )
    )

    # Presence=0 on the next policy decision is an actual yield.
    # Previous continuous actuation must not latch indefinitely.
    dispatcher(
        env_index=0,
        decoded_actions=(),
        context=None,
    )

    assert (
        dispatcher.continuous_commands(
            0
        )
        == {}
    )


def test_native_dispatcher_submits_structural_actions_as_native_primitive_goals():
    dispatcher, executor, _ = (
        _native_dispatcher_fixture()
    )

    statuses = dispatcher(
        env_index=0,
        decoded_actions=(
            _native_decoded(
                "dock",
                (
                    "module_0",
                    "module_1",
                ),
                {
                    "face_a": "TOP",
                    "face_b": "BOTTOM",
                },
            ),
        ),
        context=None,
    )

    assert len(
        executor.submitted
    ) == 1

    goal, now_s = (
        executor.submitted[0]
    )

    assert (
        goal.primitive
        == "dock"
    )

    assert goal.module_ids == (
        "module_0",
        "module_1",
    )

    assert goal.parameters == {
        "face_a": "TOP",
        "face_b": "BOTTOM",
    }

    assert goal.goal_id.startswith(
        "marl:env0:"
    )

    assert now_s == 12.5

    assert len(
        statuses
    ) == 1

    assert (
        dispatcher.continuous_commands(
            0
        )
        == {}
    )


def test_native_dispatcher_rolls_back_partial_primitive_admission_and_fails_closed():
    import pytest

    dispatcher, executor, _ = (
        _native_dispatcher_fixture(
            reject_on_submit=2,
        )
    )

    # Establish an old continuous command first.
    dispatcher(
        env_index=0,
        decoded_actions=(
            _native_decoded(
                "wheel_velocity",
                ("module_0",),
                {
                    "left_rad_s": 1.0,
                    "right_rad_s": 1.0,
                },
            ),
        ),
        context=None,
    )

    assert (
        dispatcher.continuous_commands(
            0
        )
    )

    with pytest.raises(
        RuntimeError,
        match="RESOURCE_BUSY",
    ):
        dispatcher(
            env_index=0,
            decoded_actions=(
                _native_decoded(
                    "dock",
                    (
                        "module_0",
                        "module_1",
                    ),
                    {
                        "face_a": "TOP",
                        "face_b": "BOTTOM",
                    },
                ),
                _native_decoded(
                    "dock",
                    (
                        "module_2",
                        "module_3",
                    ),
                    {
                        "face_a": "LEFT",
                        "face_b": "RIGHT",
                    },
                ),
            ),
            context=None,
        )

    assert len(
        executor.submitted
    ) == 2

    # First accepted primitive from THIS proposal is canceled.
    first_goal = (
        executor.submitted[0][0]
    )

    assert executor.canceled == [
        (
            first_goal.goal_id,
            12.5,
        )
    ]

    # Fail closed: stale continuous command must not survive.
    assert (
        dispatcher.continuous_commands(
            0
        )
        == {}
    )


def test_native_dispatcher_commits_continuous_only_after_primitive_admission():
    dispatcher, executor, _ = (
        _native_dispatcher_fixture()
    )

    dispatcher(
        env_index=0,
        decoded_actions=(
            _native_decoded(
                "dock",
                (
                    "module_0",
                    "module_1",
                ),
                {
                    "face_a": "TOP",
                    "face_b": "BOTTOM",
                },
            ),
            _native_decoded(
                "wheel_velocity",
                ("module_2",),
                {
                    "left_rad_s": 2.0,
                    "right_rad_s": 2.0,
                },
            ),
        ),
        context=None,
    )

    assert len(
        executor.submitted
    ) == 1

    commands = (
        dispatcher.continuous_commands(
            0
        )
    )

    assert set(
        commands
    ) == {
        "module_2",
    }

    assert commands[
        "module_2"
    ].linear_x_m_s > 0.0


def test_native_dispatcher_default_factories_use_real_smores_runtime_types():
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    root = Path(
        __file__
    ).resolve().parents[3]

    source = (
        root
        / "scripts"
        / "smores_ep"
        / "src"
    )

    if str(
        source
    ) not in sys.path:
        sys.path.insert(
            0,
            str(source),
        )

    from smores_ep.control.teleop import (
        SmoresCommand,
    )
    from smores_ep.primitives.model import (
        PrimitiveGoal,
        PrimitiveState,
    )

    from drl.MARL.smores_runtime_binding import (
        SMORESNativeDispatcher,
    )

    class Executor:
        def __init__(self):
            self.goals = []

        def submit(
            self,
            goal,
            now_s,
        ):
            self.goals.append(
                goal
            )

            return SimpleNamespace(
                state=(
                    PrimitiveState.ACCEPTED
                ),
                code="GOAL_ACCEPTED",
                message="accepted",
            )

        def cancel(
            self,
            goal_id,
            now_s,
        ):
            raise AssertionError(
                "rollback not expected"
            )

    executor = Executor()

    dispatcher = SMORESNativeDispatcher(
        executor_provider=(
            lambda env_index:
            executor
        ),
        now_provider=(
            lambda env_index:
            5.0
        ),
    )

    dispatcher(
        env_index=0,
        decoded_actions=(
            _native_decoded(
                "dock",
                (
                    "module_0",
                    "module_1",
                ),
                {
                    "face_a": "TOP",
                    "face_b": "BOTTOM",
                },
            ),
            _native_decoded(
                "wheel_velocity",
                ("module_2",),
                {
                    "left_rad_s": 1.0,
                    "right_rad_s": 1.0,
                },
            ),
        ),
        context=None,
    )

    assert len(
        executor.goals
    ) == 1

    assert isinstance(
        executor.goals[0],
        PrimitiveGoal,
    )

    command = (
        dispatcher
        .continuous_commands(
            0
        )["module_2"]
    )

    assert isinstance(
        command,
        SmoresCommand,
    )


# --- NATIVE SMORES ACTUATION BRIDGE ---

def test_actuation_bridge_uses_native_composition_order():
    from types import SimpleNamespace

    from drl.MARL.smores_runtime_binding import (
        SMORESNativeActuationBridge,
    )

    calls = []

    baseline = {
        "module_0": "marl-continuous",
    }

    primitive_commands = {
        "module_1": "primitive-command",
    }

    merged = {
        "module_0": "marl-continuous",
        "module_1": "primitive-command",
    }

    statuses = (
        "running",
    )

    class Dispatcher:
        def continuous_commands(
            self,
            env_index,
        ):
            calls.append(
                (
                    "baseline",
                    env_index,
                )
            )

            return dict(
                baseline
            )

    class Executor:
        def step(
            self,
            now_s,
        ):
            calls.append(
                (
                    "step",
                    now_s,
                )
            )

            return SimpleNamespace(
                commands=dict(
                    primitive_commands
                ),
                statuses=statuses,
            )

        def compose_with_baseline(
            self,
            received_baseline,
            received_primitive,
        ):
            calls.append(
                (
                    "compose",
                    received_baseline,
                    received_primitive,
                )
            )

            assert (
                received_baseline
                == baseline
            )

            assert (
                received_primitive
                == primitive_commands
            )

            return dict(
                merged
            )

    class Router:
        def apply(
            self,
            commands,
        ):
            calls.append(
                (
                    "apply",
                    commands,
                )
            )

            assert (
                commands
                == merged
            )

            return {
                "module_0": (
                    1.0,
                    2.0,
                ),
            }

    executor = Executor()
    router = Router()

    bridge = SMORESNativeActuationBridge(
        dispatcher=Dispatcher(),
        executor_provider=(
            lambda env_index:
            executor
        ),
        router_provider=(
            lambda env_index:
            router
        ),
        now_provider=(
            lambda env_index:
            7.5
        ),
    )

    result = bridge.step(
        2
    )

    assert calls == [
        (
            "baseline",
            2,
        ),
        (
            "step",
            7.5,
        ),
        (
            "compose",
            baseline,
            primitive_commands,
        ),
        (
            "apply",
            merged,
        ),
    ]

    assert result.baseline_commands == baseline
    assert result.primitive_commands == primitive_commands
    assert result.applied_commands == merged
    assert result.primitive_statuses == statuses

    assert result.wheel_rates == {
        "module_0": (
            1.0,
            2.0,
        ),
    }


def test_actuation_bridge_steps_active_primitives_with_empty_marl_baseline():
    from types import SimpleNamespace

    from drl.MARL.smores_runtime_binding import (
        SMORESNativeActuationBridge,
    )

    class Dispatcher:
        def continuous_commands(
            self,
            env_index,
        ):
            return {}

    class Executor:
        def __init__(self):
            self.step_calls = 0

        def step(
            self,
            now_s,
        ):
            self.step_calls += 1

            return SimpleNamespace(
                commands={
                    "module_0":
                    "primitive-only",
                },
                statuses=(
                    "running",
                ),
            )

        def compose_with_baseline(
            self,
            baseline,
            primitive,
        ):
            assert baseline == {}

            return dict(
                primitive
            )

    class Router:
        def __init__(self):
            self.received = None

        def apply(
            self,
            commands,
        ):
            self.received = dict(
                commands
            )

            return {}

    executor = Executor()
    router = Router()

    bridge = SMORESNativeActuationBridge(
        dispatcher=Dispatcher(),
        executor_provider=(
            lambda env_index:
            executor
        ),
        router_provider=(
            lambda env_index:
            router
        ),
        now_provider=(
            lambda env_index:
            1.0
        ),
    )

    result = bridge.step(
        0
    )

    assert executor.step_calls == 1

    assert router.received == {
        "module_0":
        "primitive-only",
    }

    assert result.baseline_commands == {}

    assert result.applied_commands == {
        "module_0":
        "primitive-only",
    }


def test_actuation_bridge_never_applies_baseline_before_primitive_overlay():
    from types import SimpleNamespace

    from drl.MARL.smores_runtime_binding import (
        SMORESNativeActuationBridge,
    )

    applied = []

    class Dispatcher:
        def continuous_commands(
            self,
            env_index,
        ):
            return {
                "module_0":
                "baseline",
            }

    class Executor:
        def step(
            self,
            now_s,
        ):
            return SimpleNamespace(
                commands={
                    "module_0":
                    "primitive",
                },
                statuses=(),
            )

        def compose_with_baseline(
            self,
            baseline,
            primitive,
        ):
            return {
                "module_0":
                "composed",
            }

    class Router:
        def apply(
            self,
            commands,
        ):
            applied.append(
                dict(
                    commands
                )
            )
            return {}

    bridge = SMORESNativeActuationBridge(
        dispatcher=Dispatcher(),
        executor_provider=(
            lambda env_index:
            Executor()
        ),
        router_provider=(
            lambda env_index:
            Router()
        ),
        now_provider=(
            lambda env_index:
            2.0
        ),
    )

    bridge.step(
        0
    )

    # There is exactly ONE drive-boundary apply call.
    assert applied == [
        {
            "module_0":
            "composed",
        }
    ]


def test_actuation_bridge_fails_closed_when_runtime_component_missing():
    import pytest

    from drl.MARL.smores_runtime_binding import (
        SMORESNativeActuationBridge,
    )

    class Dispatcher:
        def continuous_commands(
            self,
            env_index,
        ):
            return {}

    missing_executor = (
        SMORESNativeActuationBridge(
            dispatcher=Dispatcher(),
            executor_provider=(
                lambda env_index:
                None
            ),
            router_provider=(
                lambda env_index:
                object()
            ),
            now_provider=(
                lambda env_index:
                1.0
            ),
        )
    )

    with pytest.raises(
        RuntimeError,
        match="primitive executor",
    ):
        missing_executor.step(
            0
        )

    missing_router = (
        SMORESNativeActuationBridge(
            dispatcher=Dispatcher(),
            executor_provider=(
                lambda env_index:
                object()
            ),
            router_provider=(
                lambda env_index:
                None
            ),
            now_provider=(
                lambda env_index:
                1.0
            ),
        )
    )

    with pytest.raises(
        RuntimeError,
        match="command router",
    ):
        missing_router.step(
            0
        )


def test_actuation_bridge_exposes_native_statuses_and_router_result_unchanged():
    from dataclasses import dataclass
    from types import SimpleNamespace

    from drl.MARL.smores_runtime_binding import (
        SMORESNativeActuationBridge,
    )

    @dataclass(frozen=True)
    class Status:
        goal_id: str
        state: str

    status_a = Status(
        "goal-a",
        "running",
    )

    status_b = Status(
        "goal-b",
        "succeeded",
    )

    wheel_rates = {
        "module_0": (
            3.0,
            4.0,
        ),
        "module_1": (
            5.0,
            6.0,
        ),
    }

    class Dispatcher:
        def continuous_commands(
            self,
            env_index,
        ):
            return {}

    class Executor:
        def step(
            self,
            now_s,
        ):
            return SimpleNamespace(
                commands={},
                statuses=(
                    status_a,
                    status_b,
                ),
            )

        def compose_with_baseline(
            self,
            baseline,
            primitive,
        ):
            return {}

    class Router:
        def apply(
            self,
            commands,
        ):
            return wheel_rates

    bridge = SMORESNativeActuationBridge(
        dispatcher=Dispatcher(),
        executor_provider=(
            lambda env_index:
            Executor()
        ),
        router_provider=(
            lambda env_index:
            Router()
        ),
        now_provider=(
            lambda env_index:
            9.0
        ),
    )

    result = bridge.step(
        0
    )

    assert result.primitive_statuses == (
        status_a,
        status_b,
    )

    assert result.wheel_rates == wheel_rates
