"""Frozen pre-Tier1 sampler oracle; test-only, never used by runtime."""
import torch
from drl.MARL.policy import MARLCandidateRecord,ResourcePayloadView,StructuralPayloadView,MARLActionBundle

def sample_reference(
    *,
    records: tuple[MARLCandidateRecord, ...],
    resource_payloads: dict[int, ResourcePayloadView],
    structural_payloads: dict[int, StructuralPayloadView],
    hard_mask: torch.Tensor,
    resource_log_std: torch.Tensor,
    structural_log_std: torch.Tensor,
    deterministic: bool = False,
) -> MARLActionBundle:
    """Sample the calibrated policy; explicit evaluation uses distribution modes.

    Calibration only shifts membership logits, never continuous payloads.
    """
    def draw(distribution):
        if not deterministic:
            return distribution.sample()
        if isinstance(distribution, torch.distributions.Bernoulli):
            return (distribution.logits >= 0).to(distribution.logits.dtype)
        if isinstance(distribution, torch.distributions.Categorical):
            return distribution.logits.argmax(dim=-1)
        return distribution.mean
    if records:
        reference = records[0].presence_logit
        presence = torch.zeros(
            len(records),
            dtype=reference.dtype,
            device=reference.device,
        )
    else:
        presence = torch.zeros(
            0,
            dtype=torch.float32,
            device=hard_mask.device,
        )

    for record in records:
        index = record.candidate_index

        if bool(hard_mask[index].item()):
            presence[index] = 0.0
            continue

        distribution = torch.distributions.Bernoulli(
            logits=record.effective_presence_logit,
        )
        presence[index] = draw(distribution)

    resource_numeric = {}

    for record in records:
        if record.candidate_kind != "resource":
            continue

        index = record.candidate_index

        if not bool(presence[index].item()):
            continue

        payload = resource_payloads.get(index)
        if payload is None or payload.numeric is None:
            continue

        mean = payload.numeric
        std = torch.exp(resource_log_std).to(
            dtype=mean.dtype,
            device=mean.device,
        )

        distribution = torch.distributions.Normal(
            loc=mean,
            scale=std,
        )

        resource_numeric[index] = draw(distribution)

    resource_categorical = {}

    for record in records:
        if record.candidate_kind != "resource":
            continue

        index = record.candidate_index

        if not bool(presence[index].item()):
            continue

        payload = resource_payloads.get(index)
        if payload is None or payload.categorical_logits is None:
            continue

        distribution = torch.distributions.Categorical(
            logits=payload.categorical_logits,
        )

        resource_categorical[index] = draw(distribution)

    structural_numeric = {}

    for record in records:
        if record.candidate_kind != "structural":
            continue

        index = record.candidate_index

        if not bool(presence[index].item()):
            continue

        payload = structural_payloads.get(index)
        if payload is None or not payload.numeric:
            continue

        sampled_arguments = {}

        for argument in payload.numeric:
            mean = argument.values
            std = torch.exp(structural_log_std).to(
                dtype=mean.dtype,
                device=mean.device,
            )

            distribution = torch.distributions.Normal(
                loc=mean,
                scale=std,
            )

            sampled_arguments[
                argument.argument_index
            ] = draw(distribution)

        if sampled_arguments:
            structural_numeric[index] = sampled_arguments

    structural_boolean = {}

    for record in records:
        if record.candidate_kind != "structural":
            continue

        index = record.candidate_index

        if not bool(presence[index].item()):
            continue

        payload = structural_payloads.get(index)
        if payload is None or not payload.boolean:
            continue

        sampled_arguments = {}

        for argument in payload.boolean:
            distribution = torch.distributions.Bernoulli(
                logits=argument.logit,
            )

            sampled_arguments[
                argument.argument_index
            ] = draw(distribution)

        if sampled_arguments:
            structural_boolean[index] = sampled_arguments

    structural_categorical = {}

    for record in records:
        if record.candidate_kind != "structural":
            continue

        index = record.candidate_index

        if not bool(presence[index].item()):
            continue

        payload = structural_payloads.get(index)
        if payload is None or not payload.categorical:
            continue

        sampled_arguments = {}

        for argument in payload.categorical:
            distribution = torch.distributions.Categorical(
                logits=argument.logits,
            )

            sampled_arguments[
                argument.argument_index
            ] = draw(distribution)

        if sampled_arguments:
            structural_categorical[index] = sampled_arguments

    structural_reference = {}

    for record in records:
        if record.candidate_kind != "structural":
            continue

        index = record.candidate_index

        if not bool(presence[index].item()):
            continue

        payload = structural_payloads.get(index)
        if payload is None or not payload.reference:
            continue

        sampled_arguments = {}

        for argument in payload.reference:
            distribution = torch.distributions.Categorical(
                logits=argument.logits,
            )

            local_position = draw(distribution)

            sampled_arguments[
                argument.argument_index
            ] = argument.domain_indices[
                local_position
            ]

        if sampled_arguments:
            structural_reference[index] = sampled_arguments

    structural_reference_set = {}

    for record in records:
        if record.candidate_kind != "structural":
            continue

        index = record.candidate_index

        if not bool(presence[index].item()):
            continue

        payload = structural_payloads.get(index)
        if payload is None or not payload.reference_set:
            continue

        sampled_arguments = {}

        for argument in payload.reference_set:
            distribution = torch.distributions.Bernoulli(
                logits=argument.optional_logits,
            )

            optional_membership = draw(distribution).bool()

            selected_optional = argument.optional_domain_indices[
                optional_membership
            ]

            selected = torch.cat(
                (
                    selected_optional,
                    argument.required_domain_indices,
                )
            )

            sampled_arguments[
                argument.argument_index
            ] = selected

        if sampled_arguments:
            structural_reference_set[index] = sampled_arguments

    structural_scope = {}

    for record in records:
        if record.candidate_kind != "structural":
            continue

        index = record.candidate_index

        if not bool(presence[index].item()):
            continue

        payload = structural_payloads.get(index)
        if payload is None or payload.scope is None:
            continue

        scope = payload.scope

        distribution = torch.distributions.Bernoulli(
            logits=scope.optional_logits,
        )

        optional_membership = draw(distribution).bool()

        selected_optional = scope.optional_domain_indices[
            optional_membership
        ]

        selected = torch.cat(
            (
                selected_optional,
                scope.required_domain_indices,
            )
        )

        structural_scope[index] = selected

    return MARLActionBundle(
        presence=presence,
        resource_numeric=resource_numeric,
        resource_categorical=resource_categorical,
        structural_numeric=structural_numeric,
        structural_boolean=structural_boolean,
        structural_categorical=structural_categorical,
        structural_reference=structural_reference,
        structural_reference_set=structural_reference_set,
        structural_scope=structural_scope,
    )
