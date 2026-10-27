"""Bridge chronological compact episodes to canonical MSSR transitions.

``dataset_loader`` restores one logical, chronological episode from immutable
compact sources. ``preprocessing`` owns the canonical semantics of G_t, o_t,
a_t and G_t+1. This module only connects the two layers; it does not redefine
state/action semantics and it never synthesizes phase-boundary actions.
"""
from __future__ import annotations

from pathlib import Path
import copy
import json
from typing import Any, Iterator, Mapping

from dataset_loader import EpisodeSpec, iter_episode_records
from preprocessing import (
    CanonicalTimestepBuilder, PreprocessingError,
    iter_physical_timestep_buckets,
)


def _annotate_episode(
    transition: Mapping[str, Any],
    spec: EpisodeSpec,
    *,
    canonical_index: int,
) -> dict[str, Any]:
    # Builder outputs are fresh dictionaries.  Keep the large canonical graphs
    # shared inside this result and copy only the small provenance mapping.
    result = dict(transition)
    raw_provenance = result.get("provenance")
    if not isinstance(raw_provenance, Mapping):
        raise PreprocessingError("Canonical transition provenance must be a mapping.")
    provenance = dict(raw_provenance)
    result["provenance"] = provenance

    provenance.update(
        {
            "logical_episode_id": spec.episode_id,
            "canonical_transition_index": int(canonical_index),
        }
    )
    return result


def iter_canonical_episode(
    spec: EpisodeSpec,
    *,
    repo_root: Path | str | None = None,
) -> Iterator[dict[str, Any]]:
    """Build one canonical transition per unique physical ``graph_t`` snapshot.

    All expert/FSM records that observed the same robot state are aggregated into
    one joint per-module action.  The next transition's pre-action graph is the
    current transition's G_t+1, so executor availability is temporally consistent.
    """

    root = Path.cwd() if repo_root is None else Path(repo_root)
    episode_path = Path(spec.path)
    if not episode_path.is_absolute():
        episode_path = root / episode_path
    geometry_path = episode_path / "environment_geometry.json"
    episode_environment = None
    if geometry_path.is_file():
        episode_environment = json.loads(geometry_path.read_text(encoding="utf-8"))
        if not isinstance(episode_environment, Mapping):
            raise PreprocessingError(f"Episode geometry must be a mapping: {geometry_path}")
        if episode_environment.get("task") not in (None, spec.task):
            raise PreprocessingError(f"Episode geometry task does not match {spec.episode_id}.")
        if spec.seed is not None and episode_environment.get("seed") not in (None, spec.seed):
            raise PreprocessingError(f"Episode geometry seed does not match {spec.episode_id}.")

    if spec.kind not in {"expert", "teleop"}:
        raise PreprocessingError(f"Unsupported logical episode kind: {spec.kind!r}.")

    builder = CanonicalTimestepBuilder(
        repo_root=root, episode_environment=episode_environment
    )
    builder.reset()

    records: Any = iter_episode_records(spec, repo_root=root)
    legacy_button = False
    if spec.kind == "expert" and spec.task == "button":
        # Existing button experts contain interval-level IK primitive_sequence
        # records without intermediate PAN/TILT states and also miss the RC->MM8
        # structural stream.  Keep them readable for audit, but never let them
        # silently supervise the new low-level contract.  A re-recorded button
        # episode without aggregated primitive_sequence rows automatically exits
        # this quarantine.
        records = list(records)
        legacy_button = any(
            isinstance((record.get("expert_action") or {}).get("primitive_sequence"), list)
            and bool((record.get("expert_action") or {}).get("primitive_sequence"))
            for record in records
        )

    buckets = iter_physical_timestep_buckets(records)

    def quarantine_legacy_button(transition: dict[str, Any]) -> None:
        if not legacy_button:
            return
        masks = transition["supervision"]["module_resource_masks"]
        for resource_masks in masks.values():
            for resource in resource_masks:
                resource_masks[resource] = False
        transition["supervision"]["valid_for_behavior_cloning"] = False
        transition["supervision"]["episode_exclusion_reason"] = (
            "legacy_button_requires_low_level_rerecording"
        )
        transition["provenance"]["legacy_button_quarantined"] = True

    pending: dict[str, Any] | None = None
    index = 0
    for bucket in buckets:
        current = builder.build_bucket(bucket)
        quarantine_legacy_button(current)
        if pending is not None:
            pending["graph_t_plus_1"] = copy.deepcopy(current["graph_t"])
            pending.setdefault("episode_state", {})["done"] = False
            yield _annotate_episode(pending, spec, canonical_index=index)
            index += 1
        pending = current

    if pending is not None:
        pending.setdefault("episode_state", {})["done"] = True
        if pending.get("graph_t_plus_1") is None:
            pending["graph_t_plus_1"] = copy.deepcopy(pending["graph_t"])
            pending["provenance"]["terminal_next_state_fallback"] = "self_state_no_later_snapshot"
        yield _annotate_episode(pending, spec, canonical_index=index)


def canonical_episode_summary(
    spec: EpisodeSpec,
    *,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Small audit summary without materializing the episode in memory."""

    count = 0
    bc_valid = 0
    action_types: dict[str, int] = {}
    conflicts = 0
    unresolved_sequences = 0

    for transition in iter_canonical_episode(spec, repo_root=repo_root):
        count += 1
        supervision = transition.get("supervision") or {}
        bc_valid += int(supervision.get("valid_for_behavior_cloning") is True)
        conflicts += len(supervision.get("conflicts") or [])
        unresolved_sequences += len(
            (transition.get("provenance") or {}).get("unresolved_recorded_sequences") or []
        )
        modules = (transition.get("action_t") or {}).get("modules") or {}
        for action in modules.values():
            if (action.get("wheels") or {}).get("commanded"):
                action_types["wheel_velocity"] = action_types.get("wheel_velocity", 0) + 1
            internal = action.get("internal") or {}
            if internal.get("commanded") and internal.get("mode"):
                name = str(internal["mode"])
                action_types[name] = action_types.get(name, 0) + 1
            structural = action.get("structural") or {}
            if structural.get("commanded") and structural.get("primitive"):
                name = str(structural["primitive"])
                action_types[name] = action_types.get(name, 0) + 1

    return {
        "episode_id": spec.episode_id,
        "kind": spec.kind,
        "task": spec.task,
        "seed": spec.seed,
        "canonical_transitions": count,
        "bc_valid_transitions": bc_valid,
        "action_types": dict(sorted(action_types.items())),
        "resource_conflicts": conflicts,
        "unresolved_recorded_sequences": unresolved_sequences,
    }
