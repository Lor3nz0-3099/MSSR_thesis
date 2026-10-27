from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class MARLCandidateRecord:
    candidate_index: int
    candidate_kind: Literal["resource", "structural"]
    candidate_id: str
    owner_agent_id: str
    participant_agent_ids: tuple[str, ...]
    presence_logit: torch.Tensor


@dataclass(frozen=True)
class ResourcePayloadView:
    numeric: torch.Tensor | None = None
    categorical_logits: torch.Tensor | None = None


def route_resource_payloads(
    resource_parameters: dict,
) -> dict[int, ResourcePayloadView]:
    """Route flat H2 resource outputs back to their resource candidates."""
    views: dict[int, ResourcePayloadView] = {}

    payload = resource_parameters["payload"]
    payload_candidates = resource_parameters["payload_candidate"]
    payload_ptr = resource_parameters["payload_ptr"]

    for row, candidate in enumerate(payload_candidates):
        start = int(payload_ptr[row].item())
        end = int(payload_ptr[row + 1].item())
        views[int(candidate.item())] = ResourcePayloadView(
            numeric=payload[start:end],
        )

    categorical = resource_parameters["categorical"]
    category_candidates = categorical["candidate"]
    category_ptr = categorical["ptr"]
    category_logits = categorical["logits"]

    for row, candidate in enumerate(category_candidates):
        start = int(category_ptr[row].item())
        end = int(category_ptr[row + 1].item())
        views[int(candidate.item())] = ResourcePayloadView(
            categorical_logits=category_logits[start:end],
        )

    return views


@dataclass(frozen=True)
class StructuralNumericArgumentView:
    argument_index: int
    values: torch.Tensor


def route_structural_numeric_arguments(
    *,
    numeric: dict,
    structural_inputs: dict,
) -> dict[int, tuple[StructuralNumericArgumentView, ...]]:
    """Route H2 numeric arguments to their structural bindings."""
    grouped: dict[int, list[StructuralNumericArgumentView]] = {}

    arguments = numeric["argument"]
    values = numeric["value"]
    ptr = numeric["ptr"]
    argument_binding = structural_inputs["argument_binding"]

    for row, argument in enumerate(arguments):
        argument_index = int(argument.item())
        binding_index = int(argument_binding[argument_index].item())

        start = int(ptr[row].item())
        end = int(ptr[row + 1].item())

        grouped.setdefault(binding_index, []).append(
            StructuralNumericArgumentView(
                argument_index=argument_index,
                values=values[start:end],
            )
        )

    return {
        binding_index: tuple(items)
        for binding_index, items in grouped.items()
    }


@dataclass(frozen=True)
class StructuralScopeView:
    optional_logits: torch.Tensor
    optional_domain_indices: torch.Tensor
    required_domain_indices: torch.Tensor


def route_structural_scopes(
    scope: dict,
) -> dict[int, StructuralScopeView]:
    """Route H2 structural scope predictions back to structural bindings."""
    views: dict[int, StructuralScopeView] = {}

    bindings = scope["binding"]
    logits = scope["logits"]
    ptr = scope["ptr"]
    domain_index = scope["domain_index"]
    required_ptr = scope["required_ptr"]
    required_domain_index = scope["required_domain_index"]

    for row, binding in enumerate(bindings):
        binding_index = int(binding.item())

        optional_start = int(ptr[row].item())
        optional_end = int(ptr[row + 1].item())

        required_start = int(required_ptr[row].item())
        required_end = int(required_ptr[row + 1].item())

        views[binding_index] = StructuralScopeView(
            optional_logits=logits[optional_start:optional_end],
            optional_domain_indices=domain_index[optional_start:optional_end],
            required_domain_indices=required_domain_index[
                required_start:required_end
            ],
        )

    return views


@dataclass(frozen=True)
class StructuralCategoricalArgumentView:
    argument_index: int
    logits: torch.Tensor


def route_structural_categorical_arguments(
    *,
    categorical: dict,
    structural_inputs: dict,
) -> dict[int, tuple[StructuralCategoricalArgumentView, ...]]:
    """Route H2 categorical arguments to their structural bindings."""
    grouped: dict[int, list[StructuralCategoricalArgumentView]] = {}

    arguments = categorical["argument"]
    ptr = categorical["ptr"]
    logits = categorical["logits"]
    argument_binding = structural_inputs["argument_binding"]

    for row, argument in enumerate(arguments):
        argument_index = int(argument.item())
        binding_index = int(argument_binding[argument_index].item())

        start = int(ptr[row].item())
        end = int(ptr[row + 1].item())

        grouped.setdefault(binding_index, []).append(
            StructuralCategoricalArgumentView(
                argument_index=argument_index,
                logits=logits[start:end],
            )
        )

    return {
        binding_index: tuple(items)
        for binding_index, items in grouped.items()
    }


@dataclass(frozen=True)
class StructuralBooleanArgumentView:
    argument_index: int
    logit: torch.Tensor


def route_structural_boolean_arguments(
    *,
    boolean: dict,
    structural_inputs: dict,
) -> dict[int, tuple[StructuralBooleanArgumentView, ...]]:
    """Route H2 boolean arguments to their structural bindings."""
    grouped: dict[int, list[StructuralBooleanArgumentView]] = {}

    arguments = boolean["argument"]
    logits = boolean["logits"]
    argument_binding = structural_inputs["argument_binding"]

    for row, argument in enumerate(arguments):
        argument_index = int(argument.item())
        binding_index = int(argument_binding[argument_index].item())

        grouped.setdefault(binding_index, []).append(
            StructuralBooleanArgumentView(
                argument_index=argument_index,
                logit=logits[row],
            )
        )

    return {
        binding_index: tuple(items)
        for binding_index, items in grouped.items()
    }


@dataclass(frozen=True)
class StructuralReferenceArgumentView:
    argument_index: int
    logits: torch.Tensor
    domain_indices: torch.Tensor


def route_structural_reference_arguments(
    *,
    reference: dict,
    structural_inputs: dict,
) -> dict[int, tuple[StructuralReferenceArgumentView, ...]]:
    """Route H2 single-reference arguments to their structural bindings."""
    grouped: dict[int, list[StructuralReferenceArgumentView]] = {}

    arguments = reference["argument"]
    logits = reference["logits"]
    ptr = reference["ptr"]
    domain_index = reference["domain_index"]
    argument_binding = structural_inputs["argument_binding"]

    for row, argument in enumerate(arguments):
        argument_index = int(argument.item())
        binding_index = int(argument_binding[argument_index].item())

        start = int(ptr[row].item())
        end = int(ptr[row + 1].item())

        grouped.setdefault(binding_index, []).append(
            StructuralReferenceArgumentView(
                argument_index=argument_index,
                logits=logits[start:end],
                domain_indices=domain_index[start:end],
            )
        )

    return {
        binding_index: tuple(items)
        for binding_index, items in grouped.items()
    }


@dataclass(frozen=True)
class StructuralReferenceSetArgumentView:
    argument_index: int
    optional_logits: torch.Tensor
    optional_domain_indices: torch.Tensor
    required_domain_indices: torch.Tensor


def route_structural_reference_set_arguments(
    *,
    reference_set: dict,
    structural_inputs: dict,
) -> dict[int, tuple[StructuralReferenceSetArgumentView, ...]]:
    """Route H2 reference-set arguments to their structural bindings."""
    grouped: dict[int, list[StructuralReferenceSetArgumentView]] = {}

    arguments = reference_set["argument"]
    logits = reference_set["logits"]
    ptr = reference_set["ptr"]
    domain_index = reference_set["domain_index"]
    required_ptr = reference_set["required_ptr"]
    required_domain_index = reference_set["required_domain_index"]
    argument_binding = structural_inputs["argument_binding"]

    for row, argument in enumerate(arguments):
        argument_index = int(argument.item())
        binding_index = int(argument_binding[argument_index].item())

        optional_start = int(ptr[row].item())
        optional_end = int(ptr[row + 1].item())

        required_start = int(required_ptr[row].item())
        required_end = int(required_ptr[row + 1].item())

        grouped.setdefault(binding_index, []).append(
            StructuralReferenceSetArgumentView(
                argument_index=argument_index,
                optional_logits=logits[optional_start:optional_end],
                optional_domain_indices=domain_index[
                    optional_start:optional_end
                ],
                required_domain_indices=required_domain_index[
                    required_start:required_end
                ],
            )
        )

    return {
        binding_index: tuple(items)
        for binding_index, items in grouped.items()
    }


@dataclass(frozen=True)
class StructuralPayloadView:
    numeric: tuple[StructuralNumericArgumentView, ...] = ()
    categorical: tuple[StructuralCategoricalArgumentView, ...] = ()
    boolean: tuple[StructuralBooleanArgumentView, ...] = ()
    reference: tuple[StructuralReferenceArgumentView, ...] = ()
    reference_set: tuple[StructuralReferenceSetArgumentView, ...] = ()
    scope: StructuralScopeView | None = None


def compose_structural_payload_views(
    *,
    binding_count: int,
    numeric: dict[int, tuple[StructuralNumericArgumentView, ...]],
    categorical: dict[int, tuple[StructuralCategoricalArgumentView, ...]],
    boolean: dict[int, tuple[StructuralBooleanArgumentView, ...]],
    reference: dict[int, tuple[StructuralReferenceArgumentView, ...]],
    reference_set: dict[int, tuple[StructuralReferenceSetArgumentView, ...]],
    scope: dict[int, StructuralScopeView],
) -> dict[int, StructuralPayloadView]:
    """Compose all routed H2 structural outputs into one view per binding."""
    return {
        binding_index: StructuralPayloadView(
            numeric=numeric.get(binding_index, ()),
            categorical=categorical.get(binding_index, ()),
            boolean=boolean.get(binding_index, ()),
            reference=reference.get(binding_index, ()),
            reference_set=reference_set.get(binding_index, ()),
            scope=scope.get(binding_index),
        )
        for binding_index in range(binding_count)
    }


def compute_owner_presence_log_probs(
    *,
    records: tuple[MARLCandidateRecord, ...],
    sampled_presence: torch.Tensor,
    hard_mask: torch.Tensor,
    possible_agents: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Compute Bernoulli presence log-prob independently for each owner."""
    log_probs = {
        uid: torch.zeros(
            (),
            dtype=sampled_presence.dtype,
            device=sampled_presence.device,
        )
        for uid in possible_agents
    }

    for record in records:
        index = record.candidate_index

        if bool(hard_mask[index].item()):
            continue

        distribution = torch.distributions.Bernoulli(
            logits=record.presence_logit,
        )
        contribution = distribution.log_prob(
            sampled_presence[index].to(
                dtype=record.presence_logit.dtype,
                device=record.presence_logit.device,
            )
        )

        log_probs[record.owner_agent_id] = (
            log_probs[record.owner_agent_id] + contribution
        )

    return log_probs


def compute_owner_resource_categorical_log_probs(
    *,
    records: tuple[MARLCandidateRecord, ...],
    resource_payloads: dict[int, ResourcePayloadView],
    sampled_presence: torch.Tensor,
    sampled_categories: dict[int, torch.Tensor],
    hard_mask: torch.Tensor,
    possible_agents: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Compute categorical resource payload log-prob per owner."""
    log_probs = {
        uid: torch.zeros(
            (),
            dtype=sampled_presence.dtype,
            device=sampled_presence.device,
        )
        for uid in possible_agents
    }

    for record in records:
        if record.candidate_kind != "resource":
            continue

        index = record.candidate_index

        if bool(hard_mask[index].item()):
            continue

        if not bool(sampled_presence[index].item()):
            continue

        payload = resource_payloads.get(index)
        if payload is None or payload.categorical_logits is None:
            continue

        distribution = torch.distributions.Categorical(
            logits=payload.categorical_logits,
        )
        contribution = distribution.log_prob(
            sampled_categories[index].to(
                device=payload.categorical_logits.device,
            )
        )

        log_probs[record.owner_agent_id] = (
            log_probs[record.owner_agent_id] + contribution
        )

    return log_probs


def compute_owner_resource_numeric_log_probs(
    *,
    records: tuple[MARLCandidateRecord, ...],
    resource_payloads: dict[int, ResourcePayloadView],
    sampled_presence: torch.Tensor,
    sampled_numeric: dict[int, torch.Tensor],
    hard_mask: torch.Tensor,
    possible_agents: tuple[str, ...],
    log_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute Gaussian resource payload log-prob independently per owner."""
    log_probs = {
        uid: torch.zeros(
            (),
            dtype=sampled_presence.dtype,
            device=sampled_presence.device,
        )
        for uid in possible_agents
    }

    for record in records:
        if record.candidate_kind != "resource":
            continue

        index = record.candidate_index

        if bool(hard_mask[index].item()):
            continue

        if not bool(sampled_presence[index].item()):
            continue

        payload = resource_payloads.get(index)
        if payload is None or payload.numeric is None:
            continue

        mean = payload.numeric
        std = torch.exp(log_std).to(
            dtype=mean.dtype,
            device=mean.device,
        )

        distribution = torch.distributions.Normal(
            loc=mean,
            scale=std,
        )

        action = sampled_numeric[index].to(
            dtype=mean.dtype,
            device=mean.device,
        )

        contribution = distribution.log_prob(action).sum()

        log_probs[record.owner_agent_id] = (
            log_probs[record.owner_agent_id] + contribution
        )

    return log_probs


def map_structural_payloads_to_candidates(
    *,
    records: tuple[MARLCandidateRecord, ...],
    structural_payloads: dict[int, StructuralPayloadView],
) -> dict[int, StructuralPayloadView]:
    """Map local structural binding indexes to global candidate indexes."""
    structural_records = tuple(
        record
        for record in records
        if record.candidate_kind == "structural"
    )

    return {
        structural_records[binding_index].candidate_index: payload
        for binding_index, payload in structural_payloads.items()
    }


def compute_owner_structural_boolean_log_probs(
    *,
    records: tuple[MARLCandidateRecord, ...],
    structural_payloads: dict[int, StructuralPayloadView],
    sampled_presence: torch.Tensor,
    sampled_booleans: dict[int, dict[int, torch.Tensor]],
    hard_mask: torch.Tensor,
    possible_agents: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Compute structural boolean argument log-prob per owner."""
    log_probs = {
        uid: torch.zeros(
            (),
            dtype=sampled_presence.dtype,
            device=sampled_presence.device,
        )
        for uid in possible_agents
    }

    for record in records:
        if record.candidate_kind != "structural":
            continue

        index = record.candidate_index

        if bool(hard_mask[index].item()):
            continue

        if not bool(sampled_presence[index].item()):
            continue

        payload = structural_payloads.get(index)
        if payload is None:
            continue

        for argument in payload.boolean:
            distribution = torch.distributions.Bernoulli(
                logits=argument.logit,
            )

            sampled_value = sampled_booleans[index][
                argument.argument_index
            ].to(
                dtype=argument.logit.dtype,
                device=argument.logit.device,
            )

            contribution = distribution.log_prob(sampled_value)

            log_probs[record.owner_agent_id] = (
                log_probs[record.owner_agent_id] + contribution
            )

    return log_probs


def compute_owner_structural_categorical_log_probs(
    *,
    records: tuple[MARLCandidateRecord, ...],
    structural_payloads: dict[int, StructuralPayloadView],
    sampled_presence: torch.Tensor,
    sampled_categories: dict[int, dict[int, torch.Tensor]],
    hard_mask: torch.Tensor,
    possible_agents: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Compute structural categorical argument log-prob per owner."""
    log_probs = {
        uid: torch.zeros(
            (),
            dtype=sampled_presence.dtype,
            device=sampled_presence.device,
        )
        for uid in possible_agents
    }

    for record in records:
        if record.candidate_kind != "structural":
            continue

        index = record.candidate_index

        if bool(hard_mask[index].item()):
            continue

        if not bool(sampled_presence[index].item()):
            continue

        payload = structural_payloads.get(index)
        if payload is None:
            continue

        for argument in payload.categorical:
            distribution = torch.distributions.Categorical(
                logits=argument.logits,
            )

            sampled_value = sampled_categories[index][
                argument.argument_index
            ].to(device=argument.logits.device)

            contribution = distribution.log_prob(sampled_value)

            log_probs[record.owner_agent_id] = (
                log_probs[record.owner_agent_id] + contribution
            )

    return log_probs


def compute_owner_structural_numeric_log_probs(
    *,
    records: tuple[MARLCandidateRecord, ...],
    structural_payloads: dict[int, StructuralPayloadView],
    sampled_presence: torch.Tensor,
    sampled_numeric: dict[int, dict[int, torch.Tensor]],
    hard_mask: torch.Tensor,
    possible_agents: tuple[str, ...],
    log_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute Gaussian structural numeric-argument log-prob per owner."""
    log_probs = {
        uid: torch.zeros(
            (),
            dtype=sampled_presence.dtype,
            device=sampled_presence.device,
        )
        for uid in possible_agents
    }

    for record in records:
        if record.candidate_kind != "structural":
            continue

        index = record.candidate_index

        if bool(hard_mask[index].item()):
            continue

        if not bool(sampled_presence[index].item()):
            continue

        payload = structural_payloads.get(index)
        if payload is None:
            continue

        for argument in payload.numeric:
            mean = argument.values
            std = torch.exp(log_std).to(
                dtype=mean.dtype,
                device=mean.device,
            )

            distribution = torch.distributions.Normal(
                loc=mean,
                scale=std,
            )

            sampled_value = sampled_numeric[index][
                argument.argument_index
            ].to(
                dtype=mean.dtype,
                device=mean.device,
            )

            contribution = distribution.log_prob(
                sampled_value
            ).sum()

            log_probs[record.owner_agent_id] = (
                log_probs[record.owner_agent_id] + contribution
            )

    return log_probs


def compute_owner_structural_reference_log_probs(
    *,
    records: tuple[MARLCandidateRecord, ...],
    structural_payloads: dict[int, StructuralPayloadView],
    sampled_presence: torch.Tensor,
    sampled_references: dict[int, dict[int, torch.Tensor]],
    hard_mask: torch.Tensor,
    possible_agents: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Compute single-reference structural argument log-prob per owner."""
    log_probs = {
        uid: torch.zeros(
            (),
            dtype=sampled_presence.dtype,
            device=sampled_presence.device,
        )
        for uid in possible_agents
    }

    for record in records:
        if record.candidate_kind != "structural":
            continue

        index = record.candidate_index

        if bool(hard_mask[index].item()):
            continue

        if not bool(sampled_presence[index].item()):
            continue

        payload = structural_payloads.get(index)
        if payload is None:
            continue

        for argument in payload.reference:
            sampled_domain_index = sampled_references[index][
                argument.argument_index
            ].to(device=argument.domain_indices.device)

            matches = torch.nonzero(
                argument.domain_indices == sampled_domain_index,
                as_tuple=False,
            ).flatten()

            if len(matches) != 1:
                raise ValueError(
                    "sampled reference must match exactly one domain index"
                )

            local_position = matches[0].to(
                device=argument.logits.device
            )

            distribution = torch.distributions.Categorical(
                logits=argument.logits,
            )

            contribution = distribution.log_prob(local_position)

            log_probs[record.owner_agent_id] = (
                log_probs[record.owner_agent_id] + contribution
            )

    return log_probs


def compute_owner_structural_reference_set_log_probs(
    *,
    records: tuple[MARLCandidateRecord, ...],
    structural_payloads: dict[int, StructuralPayloadView],
    sampled_presence: torch.Tensor,
    sampled_reference_sets: dict[int, dict[int, torch.Tensor]],
    hard_mask: torch.Tensor,
    possible_agents: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Compute optional reference-set membership log-prob per owner."""
    log_probs = {
        uid: torch.zeros(
            (),
            dtype=sampled_presence.dtype,
            device=sampled_presence.device,
        )
        for uid in possible_agents
    }

    for record in records:
        if record.candidate_kind != "structural":
            continue

        index = record.candidate_index

        if bool(hard_mask[index].item()):
            continue

        if not bool(sampled_presence[index].item()):
            continue

        payload = structural_payloads.get(index)
        if payload is None:
            continue

        for argument in payload.reference_set:
            selected = sampled_reference_sets[index][
                argument.argument_index
            ].to(device=argument.optional_domain_indices.device)

            optional_selected = torch.isin(
                argument.optional_domain_indices,
                selected,
            ).to(
                dtype=argument.optional_logits.dtype,
                device=argument.optional_logits.device,
            )

            distribution = torch.distributions.Bernoulli(
                logits=argument.optional_logits,
            )

            contribution = distribution.log_prob(
                optional_selected
            ).sum()

            log_probs[record.owner_agent_id] = (
                log_probs[record.owner_agent_id] + contribution
            )

    return log_probs


def compute_owner_structural_scope_log_probs(
    *,
    records: tuple[MARLCandidateRecord, ...],
    structural_payloads: dict[int, StructuralPayloadView],
    sampled_presence: torch.Tensor,
    sampled_scopes: dict[int, torch.Tensor],
    hard_mask: torch.Tensor,
    possible_agents: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Compute optional structural-scope membership log-prob per owner."""
    log_probs = {
        uid: torch.zeros(
            (),
            dtype=sampled_presence.dtype,
            device=sampled_presence.device,
        )
        for uid in possible_agents
    }

    for record in records:
        if record.candidate_kind != "structural":
            continue

        index = record.candidate_index

        if bool(hard_mask[index].item()):
            continue

        if not bool(sampled_presence[index].item()):
            continue

        payload = structural_payloads.get(index)
        if payload is None or payload.scope is None:
            continue

        scope = payload.scope

        selected = sampled_scopes[index].to(
            device=scope.optional_domain_indices.device
        )

        optional_selected = torch.isin(
            scope.optional_domain_indices,
            selected,
        ).to(
            dtype=scope.optional_logits.dtype,
            device=scope.optional_logits.device,
        )

        distribution = torch.distributions.Bernoulli(
            logits=scope.optional_logits,
        )

        contribution = distribution.log_prob(
            optional_selected
        ).sum()

        log_probs[record.owner_agent_id] = (
            log_probs[record.owner_agent_id] + contribution
        )

    return log_probs


def combine_owner_log_prob_components(
    *,
    possible_agents: tuple[str, ...],
    components: tuple[dict[str, torch.Tensor], ...],
) -> dict[str, torch.Tensor]:
    """Sum already-computed log-probability components per owner."""
    combined = {}

    for uid in possible_agents:
        values = [
            component[uid]
            for component in components
            if uid in component
        ]

        if not values:
            combined[uid] = torch.tensor(0.0)
            continue

        total = values[0]
        for value in values[1:]:
            total = total + value

        combined[uid] = total

    return combined


@dataclass(frozen=True)
class MARLActionBundle:
    """Semantic action sample produced by the MARL policy."""
    presence: torch.Tensor
    resource_numeric: dict[int, torch.Tensor]
    resource_categorical: dict[int, torch.Tensor]
    structural_numeric: dict[int, dict[int, torch.Tensor]]
    structural_boolean: dict[int, dict[int, torch.Tensor]]
    structural_categorical: dict[int, dict[int, torch.Tensor]]
    structural_reference: dict[int, dict[int, torch.Tensor]]
    structural_reference_set: dict[int, dict[int, torch.Tensor]]
    structural_scope: dict[int, torch.Tensor]


def compute_owner_action_log_probs(
    *,
    records: tuple[MARLCandidateRecord, ...],
    resource_payloads: dict[int, ResourcePayloadView],
    structural_payloads: dict[int, StructuralPayloadView],
    action: MARLActionBundle,
    hard_mask: torch.Tensor,
    possible_agents: tuple[str, ...],
    resource_log_std: torch.Tensor,
    structural_log_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute the complete factorized action log-probability per owner."""
    components = (
        compute_owner_presence_log_probs(
            records=records,
            sampled_presence=action.presence,
            hard_mask=hard_mask,
            possible_agents=possible_agents,
        ),
        compute_owner_resource_numeric_log_probs(
            records=records,
            resource_payloads=resource_payloads,
            sampled_presence=action.presence,
            sampled_numeric=action.resource_numeric,
            hard_mask=hard_mask,
            possible_agents=possible_agents,
            log_std=resource_log_std,
        ),
        compute_owner_resource_categorical_log_probs(
            records=records,
            resource_payloads=resource_payloads,
            sampled_presence=action.presence,
            sampled_categories=action.resource_categorical,
            hard_mask=hard_mask,
            possible_agents=possible_agents,
        ),
        compute_owner_structural_numeric_log_probs(
            records=records,
            structural_payloads=structural_payloads,
            sampled_presence=action.presence,
            sampled_numeric=action.structural_numeric,
            hard_mask=hard_mask,
            possible_agents=possible_agents,
            log_std=structural_log_std,
        ),
        compute_owner_structural_boolean_log_probs(
            records=records,
            structural_payloads=structural_payloads,
            sampled_presence=action.presence,
            sampled_booleans=action.structural_boolean,
            hard_mask=hard_mask,
            possible_agents=possible_agents,
        ),
        compute_owner_structural_categorical_log_probs(
            records=records,
            structural_payloads=structural_payloads,
            sampled_presence=action.presence,
            sampled_categories=action.structural_categorical,
            hard_mask=hard_mask,
            possible_agents=possible_agents,
        ),
        compute_owner_structural_reference_log_probs(
            records=records,
            structural_payloads=structural_payloads,
            sampled_presence=action.presence,
            sampled_references=action.structural_reference,
            hard_mask=hard_mask,
            possible_agents=possible_agents,
        ),
        compute_owner_structural_reference_set_log_probs(
            records=records,
            structural_payloads=structural_payloads,
            sampled_presence=action.presence,
            sampled_reference_sets=action.structural_reference_set,
            hard_mask=hard_mask,
            possible_agents=possible_agents,
        ),
        compute_owner_structural_scope_log_probs(
            records=records,
            structural_payloads=structural_payloads,
            sampled_presence=action.presence,
            sampled_scopes=action.structural_scope,
            hard_mask=hard_mask,
            possible_agents=possible_agents,
        ),
    )

    return combine_owner_log_prob_components(
        possible_agents=possible_agents,
        components=components,
    )


def sample_marl_action_bundle(
    *,
    records: tuple[MARLCandidateRecord, ...],
    resource_payloads: dict[int, ResourcePayloadView],
    structural_payloads: dict[int, StructuralPayloadView],
    hard_mask: torch.Tensor,
    resource_log_std: torch.Tensor,
    structural_log_std: torch.Tensor,
) -> MARLActionBundle:
    """Sample a MARL action bundle from the current policy distributions."""
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
            logits=record.presence_logit,
        )
        presence[index] = distribution.sample()

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

        resource_numeric[index] = distribution.sample()

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

        resource_categorical[index] = distribution.sample()

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
            ] = distribution.sample()

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
            ] = distribution.sample()

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
            ] = distribution.sample()

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

            local_position = distribution.sample()

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

            optional_membership = distribution.sample().bool()

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

        optional_membership = distribution.sample().bool()

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


def compute_owner_presence_entropies(
    *,
    records: tuple[MARLCandidateRecord, ...],
    hard_mask: torch.Tensor,
    possible_agents: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Compute Bernoulli presence entropy per candidate owner."""
    if records:
        reference = records[0].presence_logit
        entropies = {
            uid: torch.zeros(
                (),
                dtype=reference.dtype,
                device=reference.device,
            )
            for uid in possible_agents
        }
    else:
        entropies = {
            uid: torch.zeros(
                (),
                dtype=torch.float32,
                device=hard_mask.device,
            )
            for uid in possible_agents
        }

    for record in records:
        index = record.candidate_index

        if bool(hard_mask[index].item()):
            continue

        distribution = torch.distributions.Bernoulli(
            logits=record.presence_logit,
        )

        entropies[record.owner_agent_id] = (
            entropies[record.owner_agent_id]
            + distribution.entropy()
        )

    return entropies


def compute_owner_resource_payload_entropies(
    *,
    records: tuple[MARLCandidateRecord, ...],
    resource_payloads: dict[int, ResourcePayloadView],
    hard_mask: torch.Tensor,
    possible_agents: tuple[str, ...],
    log_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute expected resource-payload entropy per owner."""
    if records:
        reference = records[0].presence_logit
        entropies = {
            uid: torch.zeros(
                (),
                dtype=reference.dtype,
                device=reference.device,
            )
            for uid in possible_agents
        }
    else:
        entropies = {
            uid: torch.zeros(
                (),
                dtype=torch.float32,
                device=hard_mask.device,
            )
            for uid in possible_agents
        }

    for record in records:
        if record.candidate_kind != "resource":
            continue

        index = record.candidate_index

        if bool(hard_mask[index].item()):
            continue

        payload = resource_payloads.get(index)
        if payload is None:
            continue

        payload_entropy = None

        if payload.numeric is not None:
            mean = payload.numeric
            std = torch.exp(log_std).to(
                dtype=mean.dtype,
                device=mean.device,
            )

            numeric_entropy = torch.distributions.Normal(
                loc=mean,
                scale=std,
            ).entropy().sum()

            payload_entropy = numeric_entropy

        if payload.categorical_logits is not None:
            categorical_entropy = torch.distributions.Categorical(
                logits=payload.categorical_logits,
            ).entropy()

            payload_entropy = (
                categorical_entropy
                if payload_entropy is None
                else payload_entropy + categorical_entropy
            )

        if payload_entropy is None:
            continue

        presence_probability = torch.sigmoid(
            record.presence_logit
        )

        contribution = (
            presence_probability * payload_entropy
        )

        entropies[record.owner_agent_id] = (
            entropies[record.owner_agent_id]
            + contribution
        )

    return entropies


def compute_owner_structural_payload_entropies(
    *,
    records: tuple[MARLCandidateRecord, ...],
    structural_payloads: dict[int, StructuralPayloadView],
    hard_mask: torch.Tensor,
    possible_agents: tuple[str, ...],
    log_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute expected structural-payload entropy per owner."""
    if records:
        reference = records[0].presence_logit
        entropies = {
            uid: torch.zeros(
                (),
                dtype=reference.dtype,
                device=reference.device,
            )
            for uid in possible_agents
        }
    else:
        entropies = {
            uid: torch.zeros(
                (),
                dtype=torch.float32,
                device=hard_mask.device,
            )
            for uid in possible_agents
        }

    for record in records:
        if record.candidate_kind != "structural":
            continue

        index = record.candidate_index

        if bool(hard_mask[index].item()):
            continue

        payload = structural_payloads.get(index)
        if payload is None:
            continue

        payload_entropy = None

        for argument in payload.numeric:
            mean = argument.values
            std = torch.exp(log_std).to(
                dtype=mean.dtype,
                device=mean.device,
            )

            contribution = torch.distributions.Normal(
                loc=mean,
                scale=std,
            ).entropy().sum()

            payload_entropy = (
                contribution
                if payload_entropy is None
                else payload_entropy + contribution
            )

        for argument in payload.boolean:
            contribution = torch.distributions.Bernoulli(
                logits=argument.logit,
            ).entropy().sum()

            payload_entropy = (
                contribution
                if payload_entropy is None
                else payload_entropy + contribution
            )

        for argument in payload.categorical:
            contribution = torch.distributions.Categorical(
                logits=argument.logits,
            ).entropy()

            payload_entropy = (
                contribution
                if payload_entropy is None
                else payload_entropy + contribution
            )

        for argument in payload.reference:
            contribution = torch.distributions.Categorical(
                logits=argument.logits,
            ).entropy()

            payload_entropy = (
                contribution
                if payload_entropy is None
                else payload_entropy + contribution
            )

        for argument in payload.reference_set:
            contribution = torch.distributions.Bernoulli(
                logits=argument.optional_logits,
            ).entropy().sum()

            payload_entropy = (
                contribution
                if payload_entropy is None
                else payload_entropy + contribution
            )

        if payload.scope is not None:
            contribution = torch.distributions.Bernoulli(
                logits=payload.scope.optional_logits,
            ).entropy().sum()

            payload_entropy = (
                contribution
                if payload_entropy is None
                else payload_entropy + contribution
            )

        if payload_entropy is None:
            continue

        presence_probability = torch.sigmoid(
            record.presence_logit
        )

        entropies[record.owner_agent_id] = (
            entropies[record.owner_agent_id]
            + presence_probability * payload_entropy
        )

    return entropies


def compute_owner_action_entropies(
    *,
    records: tuple[MARLCandidateRecord, ...],
    resource_payloads: dict[int, ResourcePayloadView],
    structural_payloads: dict[int, StructuralPayloadView],
    hard_mask: torch.Tensor,
    possible_agents: tuple[str, ...],
    resource_log_std: torch.Tensor,
    structural_log_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute complete hierarchical action entropy per owner."""
    components = (
        compute_owner_presence_entropies(
            records=records,
            hard_mask=hard_mask,
            possible_agents=possible_agents,
        ),
        compute_owner_resource_payload_entropies(
            records=records,
            resource_payloads=resource_payloads,
            hard_mask=hard_mask,
            possible_agents=possible_agents,
            log_std=resource_log_std,
        ),
        compute_owner_structural_payload_entropies(
            records=records,
            structural_payloads=structural_payloads,
            hard_mask=hard_mask,
            possible_agents=possible_agents,
            log_std=structural_log_std,
        ),
    )

    return combine_owner_log_prob_components(
        possible_agents=possible_agents,
        components=components,
    )


def _connector_id(ref) -> str:
    return f"{ref.module.module_id}:{ref.connector_id}"


def bind_marl_candidates(
    *,
    routing: dict,
    presence_logits: torch.Tensor,
) -> tuple[MARLCandidateRecord, ...]:
    """Bind BC-v4 candidate routing to MARL owner/participant semantics."""
    records = []
    index = 0

    for candidate in routing["resource_candidates"]:
        module_id = candidate.resource_ref.module.module_id

        records.append(
            MARLCandidateRecord(
                candidate_index=index,
                candidate_kind="resource",
                candidate_id=(
                    f"resource|{module_id}|"
                    f"{candidate.resource_ref.resource_id}|"
                    f"{candidate.action.action_id}|"
                    f"{candidate.mode.mode_id}"
                ),
                owner_agent_id=module_id,
                participant_agent_ids=(),
                presence_logit=presence_logits[index],
            )
        )
        index += 1

    for action, binding in routing["structural_bindings"]:
        owner_id = binding.initiator.module_id

        participants = tuple(
            sorted(
                module.module_id
                for module in binding.scope_members
                if module != binding.initiator
            )
        )

        connector_path = (
            ">".join(_connector_id(ref) for ref in binding.connectors)
            if binding.connectors
            else "none"
        )

        records.append(
            MARLCandidateRecord(
                candidate_index=index,
                candidate_kind="structural",
                candidate_id=(
                    f"structural|{action.action_id}|"
                    f"{owner_id}|{connector_path}"
                ),
                owner_agent_id=owner_id,
                participant_agent_ids=participants,
                presence_logit=presence_logits[index],
            )
        )
        index += 1

    return tuple(records)
