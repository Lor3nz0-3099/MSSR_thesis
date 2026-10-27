"""SMORES-specific binding for the generic Direct MARL runtime.

This module deliberately reuses the existing IL/runtime contracts:

ActionSet prediction
    -> existing decode_action_sets
    -> ActionBundle
    -> existing assert_executable
    -> existing SMORESAdapter.decode_action

No Isaac command is applied here. Physical dispatch remains the
responsibility of DirectMARLEnv/runtime integration.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from drl.IL.bc_v1.action_set import (
    ActionBundle,
)
from drl.IL.bc_v1.action_set_decoder import (
    assert_executable,
    decode_action_sets,
)
from drl.IL.bc_v1.actor_visibility import (
    project_actor_features,
)
from drl.IL.bc_v1.bc_v1_contract import (
    ActionChoice,
)


@dataclass(frozen=True)
class DecodedSMORESAction:
    """One semantic ActionChoice and its SMORES-native command."""

    choice: ActionChoice
    command: Mapping[str, object]


class SMORESMARLRuntimeBinding:
    """Translate executable ActionSets through the existing SMORES adapter."""

    def __init__(
        self,
        *,
        adapter,
        claim_resolver: Callable[
            [ActionChoice],
            tuple,
        ],
        action_set_decoder=decode_action_sets,
    ) -> None:
        if not callable(
            getattr(
                adapter,
                "decode_action",
                None,
            )
        ):
            raise TypeError(
                "adapter must expose decode_action(choice)"
            )

        if not callable(
            claim_resolver
        ):
            raise TypeError(
                "claim_resolver must be callable"
            )

        if not callable(
            action_set_decoder
        ):
            raise TypeError(
                "action_set_decoder must be callable"
            )

        self._adapter = adapter
        self._claim_resolver = (
            claim_resolver
        )
        self._action_set_decoder = (
            action_set_decoder
        )

    @staticmethod
    def _validate_decoded_command(
        command,
    ) -> dict[str, object]:
        if not isinstance(
            command,
            Mapping,
        ):
            raise ValueError(
                "decoded SMORES command must be a mapping"
            )

        if set(
            command
        ) != {
            "command",
            "module_ids",
            "parameters",
        }:
            raise ValueError(
                "decoded SMORES command must contain exactly "
                "command/module_ids/parameters"
            )

        primitive = command[
            "command"
        ]

        module_ids = command[
            "module_ids"
        ]

        parameters = command[
            "parameters"
        ]

        if (
            not isinstance(
                primitive,
                str,
            )
            or not primitive
        ):
            raise ValueError(
                "decoded SMORES command has invalid command"
            )

        if (
            not isinstance(
                module_ids,
                list,
            )
            or not module_ids
            or any(
                not isinstance(
                    module_id,
                    str,
                )
                or not module_id
                for module_id
                in module_ids
            )
        ):
            raise ValueError(
                "decoded SMORES command has invalid module_ids"
            )

        if not isinstance(
            parameters,
            Mapping,
        ):
            raise ValueError(
                "decoded SMORES command has invalid parameters"
            )

        # Isolate runtime consumers from mutable objects returned by
        # an adapter implementation.
        return {
            "command": primitive,
            "module_ids": list(
                module_ids
            ),
            "parameters": deepcopy(
                dict(parameters)
            ),
        }

    def decode_action_bundle(
        self,
        bundle: ActionBundle,
        *,
        availability=None,
    ) -> tuple[
        DecodedSMORESAction,
        ...,
    ]:
        if not isinstance(
            bundle,
            ActionBundle,
        ):
            raise TypeError(
                "Expected ActionBundle"
            )

        # Existing ActionSet semantics:
        # reject the complete proposal on overlap/unavailability.
        executable = assert_executable(
            bundle,
            self._claim_resolver,
            availability,
        )

        decoded = []

        for choice in executable.actions:
            raw = self._adapter.decode_action(
                choice
            )

            decoded.append(
                DecodedSMORESAction(
                    choice=choice,
                    command=(
                        self._validate_decoded_command(
                            raw
                        )
                    ),
                )
            )

        return tuple(
            decoded
        )

    def decode_action_set_prediction(
        self,
        prediction,
        batch,
        normalizers,
        *,
        configured_arguments=None,
        threshold=0.0,
        availability=None,
    ) -> tuple[
        DecodedSMORESAction,
        ...,
    ]:
        """Decode one ActionSet sample using the existing pure decoder."""
        bundles = self._action_set_decoder(
            prediction,
            batch,
            normalizers,
            configured_arguments=(
                configured_arguments
            ),
            threshold=threshold,
        )

        if (
            not isinstance(
                bundles,
                tuple,
            )
            or len(
                bundles
            )
            != 1
        ):
            raise ValueError(
                "runtime binding requires exactly one "
                "ActionSet sample per environment"
            )

        return self.decode_action_bundle(
            bundles[0],
            availability=availability,
        )


class SMORESActorVisibilityProjector:
    """Callable wrapper around the certified actor visibility projection."""

    def __init__(
        self,
        *,
        task: str,
        policy,
        schema,
        assigned_goal_provider=None,
        projector=project_actor_features,
    ) -> None:
        if (
            not isinstance(
                task,
                str,
            )
            or not task
        ):
            raise ValueError(
                "task must be a non-empty string"
            )

        if (
            assigned_goal_provider
            is not None
            and not callable(
                assigned_goal_provider
            )
        ):
            raise TypeError(
                "assigned_goal_provider must be callable"
            )

        if not callable(
            projector
        ):
            raise TypeError(
                "projector must be callable"
            )

        self.task = task
        self.policy = policy
        self.schema = schema

        self._assigned_goal_provider = (
            assigned_goal_provider
        )

        self._projector = projector

    def __call__(
        self,
        full_view,
    ):
        assigned_goal = None

        if (
            self._assigned_goal_provider
            is not None
        ):
            assigned_goal = (
                self._assigned_goal_provider(
                    full_view
                )
            )

        return self._projector(
            full_view,
            self.task,
            self.policy,
            self.schema,
            assigned_goal=assigned_goal,
        )


# ---------------------------------------------------------------------------
# Sampled MARL action -> existing ActionSet prediction schema
# ---------------------------------------------------------------------------

def _clone_prediction_tree(value):
    """Detach and clone an ActionSet prediction tree."""
    import torch

    if torch.is_tensor(value):
        return value.detach().clone()

    if isinstance(value, dict):
        return {
            key: _clone_prediction_tree(item)
            for key, item in value.items()
        }

    if isinstance(value, tuple):
        return tuple(
            _clone_prediction_tree(item)
            for item in value
        )

    if isinstance(value, list):
        return [
            _clone_prediction_tree(item)
            for item in value
        ]

    return deepcopy(value)


def _tensor_int_list(value, label):
    import torch

    if (
        not torch.is_tensor(value)
        or value.ndim != 1
    ):
        raise ValueError(
            f"{label} must be a 1-D tensor"
        )

    return [
        int(item)
        for item in value.detach().cpu().tolist()
    ]


def _ptr_list(value, rows, label):
    ptr = _tensor_int_list(
        value,
        label,
    )

    if (
        len(ptr) != rows + 1
        or not ptr
        or ptr[0] != 0
        or any(
            left > right
            for left, right
            in zip(
                ptr,
                ptr[1:],
            )
        )
    ):
        raise ValueError(
            f"Invalid {label}"
        )

    return ptr


def _sample_tensor(value, *, label):
    import torch

    if not torch.is_tensor(value):
        raise TypeError(
            f"{label} must be a torch.Tensor"
        )

    return value.detach()


def _sample_int(value, *, label):
    tensor = _sample_tensor(
        value,
        label=label,
    )

    if tensor.numel() != 1:
        raise ValueError(
            f"{label} must contain one index"
        )

    number = float(
        tensor.reshape(-1)[0].item()
    )

    integer = int(
        number
    )

    if number != integer:
        raise ValueError(
            f"{label} must be an integer index"
        )

    return integer


def _set_choice_logits(
    logits,
    *,
    lo,
    hi,
    position,
    label,
):
    if not (
        0 <= position < hi - lo
    ):
        raise ValueError(
            f"{label} selection is outside current domain"
        )

    logits[
        lo:hi
    ].fill_(-1.0)

    logits[
        lo + position
    ] = 1.0


def _candidate_maps(action):
    return {
        "resource numeric": (
            action.resource_numeric,
            "resource",
        ),
        "resource categorical": (
            action.resource_categorical,
            "resource",
        ),
        "structural numeric": (
            action.structural_numeric,
            "structural",
        ),
        "structural boolean": (
            action.structural_boolean,
            "structural",
        ),
        "structural categorical": (
            action.structural_categorical,
            "structural",
        ),
        "structural reference": (
            action.structural_reference,
            "structural",
        ),
        "structural reference_set": (
            action.structural_reference_set,
            "structural",
        ),
        "structural scope": (
            action.structural_scope,
            "structural",
        ),
    }


def _validate_sample_candidate_keys(
    *,
    action,
    active,
    resource_count,
    structural_count,
):
    total = (
        resource_count
        + structural_count
    )

    for label, (
        mapping,
        family,
    ) in _candidate_maps(
        action
    ).items():
        if not isinstance(
            mapping,
            dict,
        ):
            raise TypeError(
                f"{label} samples must be a dict"
            )

        for candidate in mapping:
            if type(candidate) is not int:
                raise TypeError(
                    f"{label} candidate indexes must be int"
                )

            if not (
                0 <= candidate < total
            ):
                raise ValueError(
                    f"{label} candidate index is outside current ActionSet"
                )

            if (
                family == "resource"
                and candidate >= resource_count
            ):
                raise ValueError(
                    f"{label} references a structural candidate"
                )

            if (
                family == "structural"
                and candidate < resource_count
            ):
                raise ValueError(
                    f"{label} references a resource candidate"
                )

            if not active[
                candidate
            ]:
                raise ValueError(
                    f"{label} supplied for absent candidate {candidate}"
                )


def _argument_binding_map(
    batch,
    *,
    structural_count,
):
    import torch

    try:
        inputs = (
            batch.get(
                "inputs",
                batch,
            )
            if isinstance(
                batch,
                dict,
            )
            else batch
        )

        values = inputs[
            "structural"
        ]["argument_binding"]

    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Missing structural argument_binding"
        ) from exc

    if (
        not torch.is_tensor(values)
        or values.ndim != 1
    ):
        raise ValueError(
            "structural.argument_binding must be a 1-D tensor"
        )

    bindings = [
        int(value)
        for value
        in values.detach().cpu().tolist()
    ]

    if any(
        not 0 <= value < structural_count
        for value in bindings
    ):
        raise ValueError(
            "structural argument binding is outside current candidate set"
        )

    return bindings


def _argument_sample_lookup(
    mapping,
    *,
    family_label,
    active,
    resource_count,
    argument_bindings,
):
    """Flatten {candidate: {argument: sample}} with ownership validation."""
    flattened = {}

    for candidate, rows in mapping.items():
        if not isinstance(
            rows,
            dict,
        ):
            raise TypeError(
                f"{family_label} candidate payload must be a dict"
            )

        local_binding = (
            candidate
            - resource_count
        )

        for argument, sample in rows.items():
            if type(argument) is not int:
                raise TypeError(
                    f"{family_label} argument index must be int"
                )

            if not (
                0 <= argument
                < len(argument_bindings)
            ):
                raise ValueError(
                    f"{family_label} argument index is outside current routing"
                )

            if (
                argument_bindings[
                    argument
                ]
                != local_binding
            ):
                raise ValueError(
                    f"{family_label} argument belongs to another candidate"
                )

            if argument in flattened:
                raise ValueError(
                    f"Duplicate {family_label} argument sample"
                )

            flattened[
                argument
            ] = sample

    return flattened


def _materialize_numeric_arguments(
    head,
    *,
    samples,
    active,
    resource_count,
    argument_bindings,
):
    arguments = _tensor_int_list(
        head["argument"],
        "numeric argument routing",
    )

    ptr = _ptr_list(
        head["ptr"],
        len(arguments),
        "numeric argument ptr",
    )

    values = head[
        "value"
    ]

    seen = set()

    for row, argument in enumerate(
        arguments
    ):
        binding = argument_bindings[
            argument
        ]

        candidate = (
            resource_count
            + binding
        )

        if not active[
            candidate
        ]:
            continue

        if argument not in samples:
            raise ValueError(
                "Missing sampled structural numeric argument"
            )

        sample = _sample_tensor(
            samples[
                argument
            ],
            label="structural numeric sample",
        ).reshape(-1)

        lo, hi = ptr[
            row:row + 2
        ]

        if sample.numel() != hi - lo:
            raise ValueError(
                "Structural numeric payload arity mismatch"
            )

        values[
            lo:hi
        ] = sample.to(
            dtype=values.dtype,
            device=values.device,
        )

        seen.add(
            argument
        )

    if set(
        samples
    ) != seen:
        raise ValueError(
            "Structural numeric sample has no matching prediction row"
        )


def _materialize_boolean_arguments(
    head,
    *,
    samples,
    active,
    resource_count,
    argument_bindings,
):
    arguments = _tensor_int_list(
        head["argument"],
        "boolean argument routing",
    )

    logits = head[
        "logits"
    ]

    if logits.numel() != len(
        arguments
    ):
        raise ValueError(
            "Boolean routing/logit length mismatch"
        )

    seen = set()

    for row, argument in enumerate(
        arguments
    ):
        candidate = (
            resource_count
            + argument_bindings[
                argument
            ]
        )

        if not active[
            candidate
        ]:
            continue

        if argument not in samples:
            raise ValueError(
                "Missing sampled structural boolean argument"
            )

        sample = _sample_tensor(
            samples[
                argument
            ],
            label="structural boolean sample",
        )

        if sample.numel() != 1:
            raise ValueError(
                "Structural boolean sample must be scalar"
            )

        value = float(
            sample.reshape(-1)[0].item()
        )

        if value not in (
            0.0,
            1.0,
        ):
            raise ValueError(
                "Structural boolean sample must be binary"
            )

        logits[
            row
        ] = (
            1.0
            if value
            else -1.0
        )

        seen.add(
            argument
        )

    if set(
        samples
    ) != seen:
        raise ValueError(
            "Structural boolean sample has no matching prediction row"
        )


def _materialize_index_arguments(
    head,
    *,
    samples,
    active,
    resource_count,
    argument_bindings,
    family_label,
    semantic_domain=False,
):
    arguments = _tensor_int_list(
        head["argument"],
        f"{family_label} argument routing",
    )

    ptr = _ptr_list(
        head["ptr"],
        len(arguments),
        f"{family_label} ptr",
    )

    logits = head[
        "logits"
    ]

    domains = (
        _tensor_int_list(
            head["domain_index"],
            f"{family_label} domain",
        )
        if semantic_domain
        else None
    )

    seen = set()

    for row, argument in enumerate(
        arguments
    ):
        candidate = (
            resource_count
            + argument_bindings[
                argument
            ]
        )

        if not active[
            candidate
        ]:
            continue

        if argument not in samples:
            raise ValueError(
                f"Missing sampled structural {family_label} argument"
            )

        selected = _sample_int(
            samples[
                argument
            ],
            label=(
                f"structural {family_label} sample"
            ),
        )

        lo, hi = ptr[
            row:row + 2
        ]

        if semantic_domain:
            current = domains[
                lo:hi
            ]

            if selected not in current:
                raise ValueError(
                    f"{family_label} selection is outside current domain"
                )

            position = current.index(
                selected
            )
        else:
            position = selected

        _set_choice_logits(
            logits,
            lo=lo,
            hi=hi,
            position=position,
            label=family_label,
        )

        seen.add(
            argument
        )

    if set(
        samples
    ) != seen:
        raise ValueError(
            f"Structural {family_label} sample has no matching prediction row"
        )


def _materialize_membership_arguments(
    head,
    *,
    samples,
    active,
    resource_count,
    argument_bindings,
):
    arguments = _tensor_int_list(
        head["argument"],
        "reference_set argument routing",
    )

    ptr = _ptr_list(
        head["ptr"],
        len(arguments),
        "reference_set ptr",
    )

    required_ptr = _ptr_list(
        head["required_ptr"],
        len(arguments),
        "reference_set required ptr",
    )

    optional_domain = _tensor_int_list(
        head["domain_index"],
        "reference_set optional domain",
    )

    required_domain = _tensor_int_list(
        head["required_domain_index"],
        "reference_set required domain",
    )

    logits = head[
        "logits"
    ]

    seen = set()

    for row, argument in enumerate(
        arguments
    ):
        candidate = (
            resource_count
            + argument_bindings[
                argument
            ]
        )

        if not active[
            candidate
        ]:
            continue

        if argument not in samples:
            raise ValueError(
                "Missing sampled structural reference_set argument"
            )

        selected_tensor = _sample_tensor(
            samples[
                argument
            ],
            label="structural reference_set sample",
        ).reshape(-1)

        selected = {
            int(value)
            for value
            in selected_tensor.cpu().tolist()
        }

        lo, hi = ptr[
            row:row + 2
        ]

        rlo, rhi = required_ptr[
            row:row + 2
        ]

        optional = optional_domain[
            lo:hi
        ]

        required = required_domain[
            rlo:rhi
        ]

        allowed = set(
            optional
        ).union(
            required
        )

        if not set(
            required
        ).issubset(
            selected
        ):
            raise ValueError(
                "reference_set sample omits required domain member"
            )

        if not selected.issubset(
            allowed
        ):
            raise ValueError(
                "reference_set selection is outside current domain"
            )

        for offset, domain in enumerate(
            optional
        ):
            logits[
                lo + offset
            ] = (
                1.0
                if domain in selected
                else -1.0
            )

        seen.add(
            argument
        )

    if set(
        samples
    ) != seen:
        raise ValueError(
            "Structural reference_set sample has no matching prediction row"
        )


def _materialize_scope(
    head,
    *,
    samples,
    active,
    resource_count,
    structural_count,
):
    bindings = _tensor_int_list(
        head["binding"],
        "scope binding routing",
    )

    ptr = _ptr_list(
        head["ptr"],
        len(bindings),
        "scope ptr",
    )

    required_ptr = _ptr_list(
        head["required_ptr"],
        len(bindings),
        "scope required ptr",
    )

    optional_domain = _tensor_int_list(
        head["domain_index"],
        "scope optional domain",
    )

    required_domain = _tensor_int_list(
        head["required_domain_index"],
        "scope required domain",
    )

    logits = head[
        "logits"
    ]

    seen = set()

    for row, binding in enumerate(
        bindings
    ):
        if not (
            0 <= binding
            < structural_count
        ):
            raise ValueError(
                "Scope binding is outside current structural candidates"
            )

        candidate = (
            resource_count
            + binding
        )

        if not active[
            candidate
        ]:
            continue

        if candidate not in samples:
            raise ValueError(
                "Missing sampled structural scope"
            )

        selected_tensor = _sample_tensor(
            samples[
                candidate
            ],
            label="structural scope sample",
        ).reshape(-1)

        selected = {
            int(value)
            for value
            in selected_tensor.cpu().tolist()
        }

        lo, hi = ptr[
            row:row + 2
        ]

        rlo, rhi = required_ptr[
            row:row + 2
        ]

        optional = optional_domain[
            lo:hi
        ]

        required = required_domain[
            rlo:rhi
        ]

        allowed = set(
            optional
        ).union(
            required
        )

        if not set(
            required
        ).issubset(
            selected
        ):
            raise ValueError(
                "scope sample omits required participant"
            )

        if not selected.issubset(
            allowed
        ):
            raise ValueError(
                "scope selection is outside current domain"
            )

        for offset, domain in enumerate(
            optional
        ):
            logits[
                lo + offset
            ] = (
                1.0
                if domain in selected
                else -1.0
            )

        seen.add(
            candidate
        )

    if set(
        samples
    ) != seen:
        raise ValueError(
            "Structural scope sample has no matching prediction row"
        )


def materialize_marl_action_set_prediction(
    *,
    action,
    prediction_template,
    routing,
    batch,
):
    """Represent one already-sampled MARL action in ActionSet H2 schema.

    This function performs no sampling and no policy decision. It only
    replaces the stochastic H2 values with deterministic values encoding
    exactly the previously sampled MARLActionBundle.
    """
    import torch

    from drl.MARL.policy import (
        MARLActionBundle,
    )

    if not isinstance(
        action,
        MARLActionBundle,
    ):
        raise TypeError(
            "action must be MARLActionBundle"
        )

    if not isinstance(
        prediction_template,
        dict,
    ) or set(
        prediction_template
    ) != {
        "presence_logits",
        "candidate_graph",
        "parameters",
    }:
        raise ValueError(
            "prediction_template is not an ActionSet prediction"
        )

    try:
        resource_count = len(
            routing[
                "resource_candidates"
            ]
        )

        structural_count = len(
            routing[
                "structural_bindings"
            ]
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "Malformed current candidate routing"
        ) from exc

    total = (
        resource_count
        + structural_count
    )

    presence = _sample_tensor(
        action.presence,
        label="MARL presence sample",
    )

    if (
        presence.ndim != 1
        or presence.numel()
        != total
    ):
        raise ValueError(
            "MARL presence length does not match current ActionSet"
        )

    if not torch.all(
        (presence == 0)
        | (presence == 1)
    ):
        raise ValueError(
            "MARL presence sample must be binary"
        )

    result = _clone_prediction_tree(
        prediction_template
    )

    logits = result[
        "presence_logits"
    ]

    if (
        not torch.is_tensor(
            logits
        )
        or logits.ndim != 1
        or logits.numel()
        != total
    ):
        raise ValueError(
            "ActionSet presence template length mismatch"
        )

    active = [
        bool(value)
        for value
        in presence.bool().cpu().tolist()
    ]

    logits.copy_(
        torch.where(
            presence.to(
                device=logits.device
            ).bool(),
            torch.ones_like(
                logits
            ),
            -torch.ones_like(
                logits
            ),
        )
    )

    _validate_sample_candidate_keys(
        action=action,
        active=active,
        resource_count=resource_count,
        structural_count=structural_count,
    )

    parameters = result[
        "parameters"
    ]

    if not isinstance(
        parameters,
        dict,
    ):
        raise ValueError(
            "Malformed ActionSet parameter template"
        )

    # ------------------------------------------------------------------
    # Resource payloads
    # ------------------------------------------------------------------

    resource = parameters[
        "resource"
    ]

    numeric_candidates = _tensor_int_list(
        resource[
            "payload_candidate"
        ],
        "resource numeric candidate routing",
    )

    numeric_ptr = _ptr_list(
        resource[
            "payload_ptr"
        ],
        len(
            numeric_candidates
        ),
        "resource numeric payload ptr",
    )

    numeric_values = resource[
        "payload"
    ]

    categorical = resource[
        "categorical"
    ]

    categorical_candidates = _tensor_int_list(
        categorical[
            "candidate"
        ],
        "resource categorical candidate routing",
    )

    categorical_ptr = _ptr_list(
        categorical[
            "ptr"
        ],
        len(
            categorical_candidates
        ),
        "resource categorical ptr",
    )

    categorical_logits = categorical[
        "logits"
    ]

    families = set(
        numeric_candidates
    ).union(
        categorical_candidates
    )

    if (
        set(
            numeric_candidates
        ).intersection(
            categorical_candidates
        )
        or families
        != set(
            range(
                resource_count
            )
        )
    ):
        raise ValueError(
            "Resource prediction template does not cover candidates exactly once"
        )

    seen_numeric = set()

    for row, candidate in enumerate(
        numeric_candidates
    ):
        if not active[
            candidate
        ]:
            continue

        lo, hi = numeric_ptr[
            row:row + 2
        ]

        if hi == lo:
            continue

        sample = action.resource_numeric.get(
            candidate
        )

        if sample is None:
            raise ValueError(
                "Missing sampled resource numeric payload"
            )

        sample = _sample_tensor(
            sample,
            label="resource numeric sample",
        ).reshape(-1)

        if sample.numel() != hi - lo:
            raise ValueError(
                "Resource numeric payload arity mismatch"
            )

        numeric_values[
            lo:hi
        ] = sample.to(
            dtype=numeric_values.dtype,
            device=numeric_values.device,
        )

        seen_numeric.add(
            candidate
        )

    if set(
        action.resource_numeric
    ) != seen_numeric:
        raise ValueError(
            "Resource numeric sample has no matching prediction row"
        )

    seen_categorical = set()

    for row, candidate in enumerate(
        categorical_candidates
    ):
        if not active[
            candidate
        ]:
            continue

        sample = action.resource_categorical.get(
            candidate
        )

        if sample is None:
            raise ValueError(
                "Missing sampled resource categorical payload"
            )

        selected = _sample_int(
            sample,
            label="resource categorical sample",
        )

        lo, hi = categorical_ptr[
            row:row + 2
        ]

        _set_choice_logits(
            categorical_logits,
            lo=lo,
            hi=hi,
            position=selected,
            label="resource categorical",
        )

        seen_categorical.add(
            candidate
        )

    if set(
        action.resource_categorical
    ) != seen_categorical:
        raise ValueError(
            "Resource categorical sample has no matching prediction row"
        )

    # ------------------------------------------------------------------
    # Structural arguments
    # ------------------------------------------------------------------

    structural = parameters[
        "structural"
    ]

    arguments = structural[
        "arguments"
    ]

    argument_bindings = (
        _argument_binding_map(
            batch,
            structural_count=(
                structural_count
            ),
        )
        if structural_count
        else []
    )

    numeric_samples = (
        _argument_sample_lookup(
            action.structural_numeric,
            family_label="structural numeric",
            active=active,
            resource_count=resource_count,
            argument_bindings=argument_bindings,
        )
    )

    boolean_samples = (
        _argument_sample_lookup(
            action.structural_boolean,
            family_label="structural boolean",
            active=active,
            resource_count=resource_count,
            argument_bindings=argument_bindings,
        )
    )

    categorical_samples = (
        _argument_sample_lookup(
            action.structural_categorical,
            family_label="structural categorical",
            active=active,
            resource_count=resource_count,
            argument_bindings=argument_bindings,
        )
    )

    reference_samples = (
        _argument_sample_lookup(
            action.structural_reference,
            family_label="structural reference",
            active=active,
            resource_count=resource_count,
            argument_bindings=argument_bindings,
        )
    )

    reference_set_samples = (
        _argument_sample_lookup(
            action.structural_reference_set,
            family_label="structural reference_set",
            active=active,
            resource_count=resource_count,
            argument_bindings=argument_bindings,
        )
    )

    _materialize_numeric_arguments(
        arguments[
            "numeric"
        ],
        samples=numeric_samples,
        active=active,
        resource_count=resource_count,
        argument_bindings=argument_bindings,
    )

    _materialize_boolean_arguments(
        arguments[
            "boolean"
        ],
        samples=boolean_samples,
        active=active,
        resource_count=resource_count,
        argument_bindings=argument_bindings,
    )

    _materialize_index_arguments(
        arguments[
            "category"
        ],
        samples=categorical_samples,
        active=active,
        resource_count=resource_count,
        argument_bindings=argument_bindings,
        family_label="categorical",
        semantic_domain=False,
    )

    _materialize_index_arguments(
        arguments[
            "reference"
        ],
        samples=reference_samples,
        active=active,
        resource_count=resource_count,
        argument_bindings=argument_bindings,
        family_label="reference",
        semantic_domain=True,
    )

    _materialize_membership_arguments(
        arguments[
            "reference_set"
        ],
        samples=reference_set_samples,
        active=active,
        resource_count=resource_count,
        argument_bindings=argument_bindings,
    )

    _materialize_scope(
        structural[
            "scope"
        ],
        samples=action.structural_scope,
        active=active,
        resource_count=resource_count,
        structural_count=structural_count,
    )

    return result


def _decode_marl_action(
    self,
    *,
    action,
    prediction_template,
    routing,
    batch,
    normalizers,
    configured_arguments=None,
    availability=None,
):
    """Decode one already-sampled MARLActionBundle without resampling."""
    prediction = (
        materialize_marl_action_set_prediction(
            action=action,
            prediction_template=prediction_template,
            routing=routing,
            batch=batch,
        )
    )

    return self.decode_action_set_prediction(
        prediction,
        batch,
        normalizers,
        configured_arguments=(
            configured_arguments
        ),
        threshold=0.0,
        availability=availability,
    )


# Keep the class body above compact while exposing the operation as a
# normal runtime-binding method.
SMORESMARLRuntimeBinding.decode_marl_action = (
    _decode_marl_action
)


# ---------------------------------------------------------------------------
# OnlineRolloutDecision -> SMORES runtime staging
# ---------------------------------------------------------------------------

def build_single_env_action_set_decoder_batch(
    *,
    actor_graph,
    routing_snapshot,
):
    """Package existing actor inputs/routing for decode_action_sets.

    No feature extraction, visibility projection, H1/H2 forward or sampling
    occurs here. This only reconstructs the one-environment collate ABI
    expected by the existing ActionSet decoder.
    """
    if actor_graph is None:
        raise ValueError(
            "actor graph is required for ActionSet decoding"
        )

    if routing_snapshot is None:
        raise ValueError(
            "routing snapshot is required for ActionSet decoding"
        )

    return {
        "inputs": actor_graph,
        "routing": (
            routing_snapshot,
        ),
    }


class SMORESMARLDecisionBridge:
    """Callable injected as DirectMARLRuntimeContextProvider.action_encoder.

    The true structured MARL action is decoded and staged through the
    robot-specific runtime path. The returned tensor mapping is only the
    fixed-shape environment step ABI; it is not an encoding of the policy
    action and is never used for PPO reevaluation.
    """

    def __init__(
        self,
        *,
        binding,
        normalizers,
        dispatcher,
        placeholder_action,
        decoder_batch_builder=None,
        availability_provider=None,
        configured_arguments_provider=None,
    ):
        if not callable(
            getattr(
                binding,
                "decode_marl_action",
                None,
            )
        ):
            raise TypeError(
                "binding must expose decode_marl_action"
            )

        if not callable(
            dispatcher
        ):
            raise TypeError(
                "dispatcher must be callable"
            )

        if not callable(
            placeholder_action
        ):
            raise TypeError(
                "placeholder_action must be callable"
            )

        if (
            decoder_batch_builder
            is not None
            and not callable(
                decoder_batch_builder
            )
        ):
            raise TypeError(
                "decoder_batch_builder must be callable"
            )

        for name, provider in (
            (
                "availability_provider",
                availability_provider,
            ),
            (
                "configured_arguments_provider",
                configured_arguments_provider,
            ),
        ):
            if (
                provider is not None
                and not callable(
                    provider
                )
            ):
                raise TypeError(
                    f"{name} must be callable"
                )

        self.binding = binding
        self.normalizers = normalizers
        self.dispatcher = dispatcher
        self.placeholder_action = (
            placeholder_action
        )

        self.decoder_batch_builder = (
            decoder_batch_builder
        )

        self.availability_provider = (
            availability_provider
        )

        self.configured_arguments_provider = (
            configured_arguments_provider
        )

    def _decoder_batch(
        self,
        *,
        env_index,
        context,
    ):
        if (
            self.decoder_batch_builder
            is not None
        ):
            batch = (
                self.decoder_batch_builder(
                    env_index,
                    context,
                )
            )
        else:
            batch = (
                build_single_env_action_set_decoder_batch(
                    actor_graph=(
                        context.actor_graph
                    ),
                    routing_snapshot=(
                        context.routing_snapshot
                    ),
                )
            )

        if not isinstance(
            batch,
            dict,
        ):
            raise TypeError(
                "ActionSet decoder batch must be a dict"
            )

        if (
            "inputs" not in batch
            or "routing" not in batch
        ):
            raise ValueError(
                "ActionSet decoder batch requires inputs and routing"
            )

        return batch

    def __call__(
        self,
        decision,
        env_index,
        context,
    ):
        """Revoke stale continuous actuation when a proposal is rejected."""
        try:
            return self._stage_decision(decision, env_index, context)
        except Exception:
            clear = getattr(self.dispatcher, "clear", None)
            if callable(clear):
                clear(env_index)
            raise

    def _stage_decision(
        self,
        decision,
        env_index,
        context,
    ):
        """Decode and stage one already-sampled online MARL decision."""
        if decision is None:
            raise ValueError(
                "online MARL decision is required"
            )

        action = getattr(
            decision,
            "action",
            None,
        )

        if action is None:
            raise ValueError(
                "online MARL decision has no sampled action"
            )

        prediction_template = getattr(
            decision,
            "action_set_prediction_template",
            None,
        )

        if prediction_template is None:
            raise ValueError(
                "online MARL decision has no ActionSet prediction template"
            )

        batch = self._decoder_batch(
            env_index=env_index,
            context=context,
        )

        availability = None

        if (
            self.availability_provider
            is not None
        ):
            availability = (
                self.availability_provider(
                    env_index,
                    context,
                )
            )

        configured_arguments = None

        if (
            self.configured_arguments_provider
            is not None
        ):
            configured_arguments = (
                self.configured_arguments_provider(
                    env_index,
                    context,
                )
            )

        # This is the ONLY semantic decode after the policy sample.
        # decode_marl_action materializes the sampled bundle and passes it
        # through the existing ActionSet decoder / frozen F2 / SMORESAdapter.
        decoded_actions = (
            self.binding.decode_marl_action(
                action=action,
                prediction_template=(
                    prediction_template
                ),
                routing=(
                    context.routing_snapshot
                ),
                batch=batch,
                normalizers=(
                    self.normalizers
                ),
                configured_arguments=(
                    configured_arguments
                ),
                availability=(
                    availability
                ),
            )
        )

        # Stage the complete proposal as one unit. The dispatcher/native
        # owner remains responsible for actual physical admission/application.
        self.dispatcher(
            env_index=env_index,
            decoded_actions=decoded_actions,
            context=context,
        )

        # Only after successful staging do we return the fixed-shape tensor
        # ABI required by skrl/DirectMARLEnv.step().
        return self.placeholder_action(
            env_index,
            context,
        )


# ---------------------------------------------------------------------------
# Native SMORES runtime dispatcher
# ---------------------------------------------------------------------------

class SMORESNativeDispatcher:
    """Stage decoded MARL actions into the existing native SMORES runtime.

    Continuous commands are held only until the next policy proposal.
    Structural commands are admitted through IsaacPrimitiveExecutor.submit().
    No command is applied directly to PhysX here.
    """

    _CONTINUOUS_COMMANDS = frozenset(
        {
            "wheel_velocity",
            "pan_velocity",
        }
    )

    def __init__(
        self,
        *,
        executor_provider,
        now_provider,
        geometry=None,
        command_factory=None,
        pan_velocity_mode=None,
        primitive_goal_factory=None,
        accepted_state=None,
        primitive_timeout_s=30.0,
    ):
        import math

        if not callable(
            executor_provider
        ):
            raise TypeError(
                "executor_provider must be callable"
            )

        if not callable(
            now_provider
        ):
            raise TypeError(
                "now_provider must be callable"
            )

        if (
            not math.isfinite(
                primitive_timeout_s
            )
            or primitive_timeout_s <= 0.0
        ):
            raise ValueError(
                "primitive_timeout_s must be finite and positive"
            )

        if geometry is None:
            from smores_ep.config.geometry import (
                SmoresGeometry,
            )

            geometry = (
                SmoresGeometry()
            )

        if command_factory is None:
            from smores_ep.control.teleop import (
                InternalMotionMode,
                SmoresCommand,
            )

            command_factory = (
                SmoresCommand
            )

            if (
                pan_velocity_mode
                is None
            ):
                pan_velocity_mode = (
                    InternalMotionMode
                    .PAN_VELOCITY
                )

        if (
            primitive_goal_factory
            is None
            or accepted_state is None
        ):
            from smores_ep.primitives.model import (
                PrimitiveGoal,
                PrimitiveState,
            )

            if (
                primitive_goal_factory
                is None
            ):
                primitive_goal_factory = (
                    PrimitiveGoal
                    .from_dict
                )

            if accepted_state is None:
                accepted_state = (
                    PrimitiveState
                    .ACCEPTED
                )

        if pan_velocity_mode is None:
            raise ValueError(
                "pan_velocity_mode is required"
            )

        for field in (
            "wheel_radius_m",
            "track_width_m",
        ):
            value = getattr(
                geometry,
                field,
                None,
            )

            if (
                type(value)
                not in (
                    int,
                    float,
                )
                or not math.isfinite(
                    value
                )
                or value <= 0.0
            ):
                raise ValueError(
                    f"geometry.{field} must be finite and positive"
                )

        self.executor_provider = (
            executor_provider
        )

        self.now_provider = (
            now_provider
        )

        self.geometry = geometry
        self.command_factory = (
            command_factory
        )

        self.pan_velocity_mode = (
            pan_velocity_mode
        )

        self.primitive_goal_factory = (
            primitive_goal_factory
        )

        self.accepted_state = (
            accepted_state
        )

        self.primitive_timeout_s = float(
            primitive_timeout_s
        )

        self._continuous_by_env = {}
        self._sequence_by_env = {}

    @staticmethod
    def _finite_number(
        value,
        *,
        label,
    ):
        import math

        if (
            isinstance(
                value,
                bool,
            )
            or type(value)
            not in (
                int,
                float,
            )
            or not math.isfinite(
                value
            )
        ):
            raise ValueError(
                f"{label} must be finite"
            )

        return float(
            value
        )

    @staticmethod
    def _command_payload(
        decoded,
    ):
        from collections.abc import (
            Mapping,
        )

        payload = getattr(
            decoded,
            "command",
            None,
        )

        if (
            not isinstance(
                payload,
                Mapping,
            )
            or set(
                payload
            )
            != {
                "command",
                "module_ids",
                "parameters",
            }
        ):
            raise ValueError(
                "decoded SMORES action has invalid command mapping"
            )

        command = payload[
            "command"
        ]

        module_ids = payload[
            "module_ids"
        ]

        parameters = payload[
            "parameters"
        ]

        if (
            not isinstance(
                command,
                str,
            )
            or not command
        ):
            raise ValueError(
                "decoded command name is invalid"
            )

        if (
            not isinstance(
                module_ids,
                list,
            )
            or not module_ids
            or any(
                not isinstance(
                    module_id,
                    str,
                )
                or not module_id
                for module_id
                in module_ids
            )
        ):
            raise ValueError(
                "decoded command module_ids are invalid"
            )

        if not isinstance(
            parameters,
            Mapping,
        ):
            raise ValueError(
                "decoded command parameters are invalid"
            )

        return (
            command,
            tuple(
                module_ids
            ),
            dict(
                parameters
            ),
        )

    def _next_sequence(
        self,
        env_index,
    ):
        sequence = (
            self._sequence_by_env.get(
                env_index,
                0,
            )
            + 1
        )

        self._sequence_by_env[
            env_index
        ] = sequence

        return sequence

    def _continuous_specs(
        self,
        decoded_actions,
    ):
        specs = {}
        primitive_rows = []

        for decoded in decoded_actions:
            (
                command,
                module_ids,
                parameters,
            ) = self._command_payload(
                decoded
            )

            if (
                command
                not in self._CONTINUOUS_COMMANDS
            ):
                primitive_rows.append(
                    (
                        command,
                        module_ids,
                        parameters,
                    )
                )
                continue

            if len(
                module_ids
            ) != 1:
                raise ValueError(
                    f"{command} requires exactly one module"
                )

            module_id = (
                module_ids[0]
            )

            spec = specs.setdefault(
                module_id,
                {
                    "wheel": None,
                    "pan": None,
                },
            )

            if (
                command
                == "wheel_velocity"
            ):
                if spec[
                    "wheel"
                ] is not None:
                    raise ValueError(
                        "duplicate wheel_velocity command for one module"
                    )

                if set(
                    parameters
                ) != {
                    "left_rad_s",
                    "right_rad_s",
                }:
                    raise ValueError(
                        "wheel_velocity parameters must be left_rad_s/right_rad_s"
                    )

                left = (
                    self._finite_number(
                        parameters[
                            "left_rad_s"
                        ],
                        label=(
                            "left_rad_s"
                        ),
                    )
                )

                right = (
                    self._finite_number(
                        parameters[
                            "right_rad_s"
                        ],
                        label=(
                            "right_rad_s"
                        ),
                    )
                )

                spec[
                    "wheel"
                ] = (
                    left,
                    right,
                )

            elif (
                command
                == "pan_velocity"
            ):
                if spec[
                    "pan"
                ] is not None:
                    raise ValueError(
                        "duplicate pan_velocity command for one module"
                    )

                if set(
                    parameters
                ) != {
                    "rate_rad_s",
                }:
                    raise ValueError(
                        "pan_velocity parameters must contain rate_rad_s"
                    )

                spec[
                    "pan"
                ] = (
                    self._finite_number(
                        parameters[
                            "rate_rad_s"
                        ],
                        label=(
                            "rate_rad_s"
                        ),
                    )
                )

        return (
            specs,
            primitive_rows,
        )

    def _build_continuous_commands(
        self,
        specs,
    ):
        commands = {}

        radius = float(
            self.geometry
            .wheel_radius_m
        )

        track = float(
            self.geometry
            .track_width_m
        )

        for (
            module_id,
            spec,
        ) in specs.items():
            linear = 0.0
            angular = 0.0
            pan_rate = 0.0
            kwargs = {}

            if (
                spec[
                    "wheel"
                ] is not None
            ):
                (
                    left,
                    right,
                ) = spec[
                    "wheel"
                ]

                linear = (
                    radius
                    * (
                        left
                        + right
                    )
                    / 2.0
                )

                angular = (
                    radius
                    * (
                        right
                        - left
                    )
                    / track
                )

            if (
                spec[
                    "pan"
                ] is not None
            ):
                pan_rate = float(
                    spec[
                        "pan"
                    ]
                )

                kwargs[
                    "internal_motion"
                ] = (
                    self.pan_velocity_mode
                )

                kwargs[
                    "pan_velocity_rad_s"
                ] = pan_rate

            commands[
                module_id
            ] = self.command_factory(
                linear_x_m_s=linear,
                angular_z_rad_s=angular,
                **kwargs,
            )

        return commands

    def _build_primitive_goal(
        self,
        *,
        env_index,
        sequence,
        action_index,
        command,
        module_ids,
        parameters,
    ):
        payload = {
            "goal_id": (
                f"marl:env{env_index}:"
                f"proposal{sequence}:"
                f"action{action_index}"
            ),
            "primitive": command,
            "module_ids": list(
                module_ids
            ),
            "parameters": dict(
                parameters
            ),
            "timeout_s": (
                self.primitive_timeout_s
            ),
        }

        return (
            self.primitive_goal_factory(
                payload
            )
        )

    def continuous_commands(
        self,
        env_index,
    ):
        """Current continuous proposal for one environment."""
        return dict(
            self._continuous_by_env.get(
                env_index,
                {},
            )
        )

    def clear(
        self,
        env_index=None,
    ):
        if env_index is None:
            self._continuous_by_env.clear()
            return

        self._continuous_by_env.pop(
            env_index,
            None,
        )

    def __call__(
        self,
        *,
        env_index,
        decoded_actions,
        context,
    ):
        """Stage one complete decoded ActionSet atomically where possible."""
        del context

        decoded_actions = tuple(
            decoded_actions
        )

        executor = (
            self.executor_provider(
                env_index
            )
        )

        if executor is None:
            self.clear(
                env_index
            )

            raise RuntimeError(
                f"No primitive executor for env {env_index}"
            )

        now_s = self._finite_number(
            self.now_provider(
                env_index
            ),
            label="runtime timestamp",
        )

        sequence = (
            self._next_sequence(
                env_index
            )
        )

        accepted_goals = []

        try:
            (
                specs,
                primitive_rows,
            ) = self._continuous_specs(
                decoded_actions
            )

            pending_continuous = (
                self._build_continuous_commands(
                    specs
                )
            )

            statuses = []

            for (
                action_index,
                (
                    command,
                    module_ids,
                    parameters,
                ),
            ) in enumerate(
                primitive_rows
            ):
                goal = (
                    self._build_primitive_goal(
                        env_index=env_index,
                        sequence=sequence,
                        action_index=action_index,
                        command=command,
                        module_ids=module_ids,
                        parameters=parameters,
                    )
                )

                status = executor.submit(
                    goal,
                    now_s,
                )

                state = getattr(
                    status,
                    "state",
                    None,
                )

                if (
                    state
                    != self.accepted_state
                ):
                    code = getattr(
                        status,
                        "code",
                        "UNKNOWN_REJECTION",
                    )

                    message = getattr(
                        status,
                        "message",
                        "",
                    )

                    raise RuntimeError(
                        "MARL primitive admission rejected: "
                        f"{code}: {message}"
                    )

                accepted_goals.append(
                    goal
                )

                statuses.append(
                    status
                )

            # Commit continuous actuation only after every primitive from
            # this sampled ActionSet was admitted.
            self._continuous_by_env[
                env_index
            ] = pending_continuous

            return tuple(
                statuses
            )

        except Exception:
            # Presence=0 or rejected proposals must never leave stale drive
            # commands from the preceding policy decision.
            self.clear(
                env_index
            )

            rollback_errors = []

            for goal in reversed(
                accepted_goals
            ):
                try:
                    executor.cancel(
                        goal.goal_id,
                        now_s,
                    )

                except Exception as error:
                    rollback_errors.append(
                        (
                            goal.goal_id,
                            error,
                        )
                    )

            if rollback_errors:
                details = "; ".join(
                    f"{goal_id}: {error}"
                    for (
                        goal_id,
                        error,
                    )
                    in rollback_errors
                )

                raise RuntimeError(
                    "MARL primitive proposal failed and rollback "
                    f"also failed: {details}"
                )

            raise


# ---------------------------------------------------------------------------
# Native SMORES actuation boundary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SMORESNativeActuationStep:
    """One native SMORES drive-boundary result."""

    baseline_commands: Mapping[str, object]
    primitive_commands: Mapping[str, object]
    applied_commands: Mapping[str, object]
    primitive_statuses: tuple[object, ...]
    wheel_rates: Mapping[
        str,
        tuple[float, float],
    ]


class SMORESNativeActuationBridge:
    """Compose MARL baseline + native primitives and apply once via router."""

    def __init__(
        self,
        *,
        dispatcher,
        executor_provider,
        router_provider,
        now_provider,
    ):
        if not callable(
            getattr(
                dispatcher,
                "continuous_commands",
                None,
            )
        ):
            raise TypeError(
                "dispatcher must expose continuous_commands(env_index)"
            )

        for name, provider in (
            (
                "executor_provider",
                executor_provider,
            ),
            (
                "router_provider",
                router_provider,
            ),
            (
                "now_provider",
                now_provider,
            ),
        ):
            if not callable(
                provider
            ):
                raise TypeError(
                    f"{name} must be callable"
                )

        self.dispatcher = dispatcher
        self.executor_provider = (
            executor_provider
        )
        self.router_provider = (
            router_provider
        )
        self.now_provider = (
            now_provider
        )

    @staticmethod
    def _finite_time(
        value,
    ):
        import math

        if (
            isinstance(
                value,
                bool,
            )
            or type(value)
            not in (
                int,
                float,
            )
            or not math.isfinite(
                value
            )
        ):
            raise ValueError(
                "runtime timestamp must be finite"
            )

        return float(
            value
        )

    def step(
        self,
        env_index,
    ) -> SMORESNativeActuationStep:
        executor = (
            self.executor_provider(
                env_index
            )
        )

        if executor is None:
            raise RuntimeError(
                f"No primitive executor for env {env_index}"
            )

        router = (
            self.router_provider(
                env_index
            )
        )

        if router is None:
            raise RuntimeError(
                f"No command router for env {env_index}"
            )

        for name in (
            "step",
            "compose_with_baseline",
        ):
            if not callable(
                getattr(
                    executor,
                    name,
                    None,
                )
            ):
                raise TypeError(
                    f"primitive executor must expose {name}()"
                )

        if not callable(
            getattr(
                router,
                "apply",
                None,
            )
        ):
            raise TypeError(
                "command router must expose apply()"
            )

        now_s = self._finite_time(
            self.now_provider(
                env_index
            )
        )

        # Current policy proposal. Presence=0 from the most recent ActionSet
        # has already removed any stale continuous command in the dispatcher.
        baseline = dict(
            self.dispatcher
            .continuous_commands(
                env_index
            )
        )

        # Progress every admitted temporal primitive, even when MARL has
        # no continuous action in the current policy decision.
        primitive_step = (
            executor.step(
                now_s
            )
        )

        primitive_commands = dict(
            getattr(
                primitive_step,
                "commands",
            )
        )

        primitive_statuses = tuple(
            getattr(
                primitive_step,
                "statuses",
            )
        )

        # Native executor owns the resource arbitration/overlay semantics.
        composed = dict(
            executor.compose_with_baseline(
                baseline,
                primitive_commands,
            )
        )

        # One and only one actuator-boundary call.
        wheel_rates = dict(
            router.apply(
                composed
            )
        )

        return SMORESNativeActuationStep(
            baseline_commands=baseline,
            primitive_commands=(
                primitive_commands
            ),
            applied_commands=composed,
            primitive_statuses=(
                primitive_statuses
            ),
            wheel_rates=wheel_rates,
        )
