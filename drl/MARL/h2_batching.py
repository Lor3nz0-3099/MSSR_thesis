"""Lossless, differentiable inverse of G2 collation for ActionSetHead outputs.

The H2 head predicts one large batch with typed resource rows followed by
structural rows. MARL needs the original per-graph ActionSetPolicy ABI. All
indices are localised to the graph's own candidate, argument, scope-domain,
and argument-domain tables. No tensor value is detached or copied to CPU;
only integer slice boundaries are synchronised for Python slicing.
"""

from __future__ import annotations

from itertools import accumulate
from typing import Any

import torch


def _offsets(graphs: list[dict[str, Any]], *keys: str) -> list[int]:
    sizes = []
    for graph in graphs:
        obj = graph
        for key in keys:
            obj = obj[key]
        sizes.append(len(obj))
    return [0, *accumulate(sizes)]


def _bounds(indices: torch.Tensor, offsets: list[int]) -> list[int]:
    """Read only the O(number_of_graphs) row boundaries onto host."""
    if indices.ndim != 1 or indices.dtype != torch.int64:
        raise ValueError("H2 routing must use flat int64 indices")
    cuts = torch.as_tensor(offsets, device=indices.device, dtype=indices.dtype)
    return torch.searchsorted(indices.contiguous(), cuts).cpu().tolist()


def _pointer_boundaries(ptr: torch.Tensor, row_bounds: list[int]) -> list[int]:
    rows = torch.as_tensor(row_bounds, device=ptr.device, dtype=torch.int64)
    return ptr.index_select(0, rows).cpu().tolist()


def _split_indexed(
    head: dict[str, torch.Tensor],
    *,
    indices_key: str,
    offsets: list[int],
    values_key: str,
    pointer_key: str | None = "ptr",
    domain_offsets: list[int] | None = None,
    required: bool = False,
) -> list[dict[str, torch.Tensor]]:
    """Split a packed, ordered H2 family and restore its *local* routing."""
    expected = {indices_key, values_key}
    if pointer_key is not None:
        expected.add(pointer_key)
    if domain_offsets is not None:
        expected.add("domain_index")
    if required:
        expected.update(("required_ptr", "required_domain_index"))
    if set(head) != expected:
        raise ValueError(
            f"Unexpected H2 output ABI for {indices_key}: "
            f"{sorted(set(head) ^ expected)}"
        )

    indices = head[indices_key]
    row_bounds = _bounds(indices, offsets)
    if row_bounds[0] != 0 or row_bounds[-1] != len(indices):
        raise ValueError("H2 batch routing escaped graph ownership")

    if pointer_key is not None:
        ptr = head[pointer_key]
        if len(ptr) != len(indices) + 1:
            raise ValueError("Invalid H2 pointer length")
        flat_bounds = _pointer_boundaries(ptr, row_bounds)
    else:
        ptr = None
        flat_bounds = row_bounds

    if required:
        rptr = head["required_ptr"]
        if len(rptr) != len(indices) + 1:
            raise ValueError("Invalid H2 required pointer length")
        required_bounds = _pointer_boundaries(rptr, row_bounds)

    results = []
    for i, (a, b) in enumerate(zip(row_bounds, row_bounds[1:])):
        start, stop = flat_bounds[i:i + 2]
        local = {
            indices_key: indices[a:b] - offsets[i],
            values_key: head[values_key][start:stop],
        }
        if pointer_key is not None:
            local[pointer_key] = ptr[a:b + 1] - ptr[a]
        if domain_offsets is not None:
            local["domain_index"] = (
                head["domain_index"][start:stop] - domain_offsets[i]
            )
        if required:
            ra, rb = required_bounds[i:i + 2]
            local["required_ptr"] = rptr[a:b + 1] - rptr[a]
            local["required_domain_index"] = (
                head["required_domain_index"][ra:rb] - domain_offsets[i]
            )
        results.append(local)
    return results


def split_action_set_h2_batch(
    *,
    packed: dict[str, Any],
    graphs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reconstruct exact per-graph H2 outputs from one batched H2 call.

    Keeps autograd through every predicted logit/value. Supports variable
    module counts, variable candidate lists, empty families, numeric and
    categorical resource payloads, structural numeric/category/boolean/
    reference/reference_set arguments, and optional/required scopes.
    """
    if not graphs:
        return []
    if set(packed) != {"presence_logits", "candidate_graph", "parameters"}:
        raise ValueError("Unexpected ActionSetHead output ABI")

    n = len(graphs)
    resource_offsets = _offsets(graphs, "resource_candidates", "resource")
    structural_offsets = _offsets(graphs, "structural", "initiator")
    argument_offsets = _offsets(graphs, "structural", "argument_binding")
    arg_domain_offsets = _offsets(graphs, "structural", "argument_domain_index")
    scope_domain_offsets = _offsets(graphs, "structural", "scope_members")
    total_resource = resource_offsets[-1]
    total_structural = structural_offsets[-1]

    presence = packed["presence_logits"]
    candidate_graph = packed["candidate_graph"]
    if len(presence) != total_resource + total_structural or len(candidate_graph) != len(presence):
        raise ValueError("Invalid candidate count in batched H2 prediction")

    params = packed["parameters"]
    if set(params) != {"resource", "structural"}:
        raise ValueError("Unexpected H2 parameter families")
    resource = params["resource"]
    if set(resource) != {"payload", "payload_candidate", "payload_ptr", "categorical"}:
        raise ValueError("Unexpected H2 resource parameter ABI")
    categorical = resource["categorical"]
    structural = params["structural"]
    if set(structural) != {"arguments", "scope"}:
        raise ValueError("Unexpected H2 structural parameter ABI")
    arguments = structural["arguments"]
    if set(arguments) != {"numeric", "reference_set", "reference", "boolean", "category"}:
        raise ValueError("Unexpected H2 argument families")

    numeric_resource = _split_indexed(
        {"candidate": resource["payload_candidate"],
         "ptr": resource["payload_ptr"],
         "value": resource["payload"]},
        indices_key="candidate", offsets=resource_offsets,
        values_key="value",
    )
    categorical_resource = _split_indexed(
        categorical, indices_key="candidate", offsets=resource_offsets,
        values_key="logits",
    )

    struct_splits = {
        "numeric": _split_indexed(
            arguments["numeric"], indices_key="argument", offsets=argument_offsets,
            values_key="value",
        ),
        "category": _split_indexed(
            arguments["category"], indices_key="argument", offsets=argument_offsets,
            values_key="logits",
        ),
        "boolean": _split_indexed(
            arguments["boolean"], indices_key="argument", offsets=argument_offsets,
            values_key="logits", pointer_key=None,
        ),
        "reference": _split_indexed(
            arguments["reference"], indices_key="argument", offsets=argument_offsets,
            values_key="logits", domain_offsets=arg_domain_offsets,
        ),
        "reference_set": _split_indexed(
            arguments["reference_set"], indices_key="argument", offsets=argument_offsets,
            values_key="logits", domain_offsets=arg_domain_offsets, required=True,
        ),
    }
    scopes = _split_indexed(
        structural["scope"], indices_key="binding", offsets=structural_offsets,
        values_key="logits", domain_offsets=scope_domain_offsets, required=True,
    )

    outputs = []
    for i in range(n):
        r0, r1 = resource_offsets[i:i + 2]
        s0, s1 = structural_offsets[i:i + 2]
        serial_indices = torch.cat((
            candidate_graph[r0:r1],
            candidate_graph[total_resource + s0:total_resource + s1],
        ), dim=0) - i
        numeric = numeric_resource[i]
        outputs.append({
            "presence_logits": torch.cat((
                presence[r0:r1],
                presence[total_resource + s0:total_resource + s1],
            ), dim=0),
            "candidate_graph": serial_indices,
            "parameters": {
                "resource": {
                    "payload": numeric["value"],
                    "payload_candidate": numeric["candidate"],
                    "payload_ptr": numeric["ptr"],
                    "categorical": categorical_resource[i],
                },
                "structural": {
                    "arguments": {
                        family: per_graph[i]
                        for family, per_graph in struct_splits.items()
                    },
                    "scope": scopes[i],
                },
            },
        })
    return outputs
