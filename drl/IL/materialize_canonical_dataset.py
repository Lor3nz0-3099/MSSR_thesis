"""Materialize the validated canonical MSSR dataset without changing semantics.

This writer is intentionally downstream-only:

    compact collection
        -> dataset_loader
        -> canonical_dataset.iter_canonical_episode
        -> one canonical JSONL file per logical episode

It does not split episodes, tensorize features, normalize values, expand expert
RLE weights, or reinterpret actions. The canonical transition dictionaries are
written exactly as produced by the already validated bridge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from canonical_dataset import iter_canonical_episode
from dataset_loader import EpisodeSpec, discover_compact_episode_specs


MANIFEST_SCHEMA = "mssr.canonical_dataset_collection.v5"


def _manifest_path(path: Path, repo_root: Path) -> str:
    """Prefer portable repository paths while accepting external collections."""
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_line(value: Mapping[str, Any]) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def _safe_episode_filename(episode_id: str) -> str:
    name = episode_id.replace(":", "__")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    if not name:
        raise ValueError(f"Cannot derive filename from episode id {episode_id!r}.")
    return f"{name}.jsonl"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _transition_stats(
    transition: Mapping[str, Any],
    *,
    expected_index: int,
) -> tuple[bool, int, str, tuple[str, ...], str | None]:
    for key in (
        "graph_t", "observation_t", "action_t", "graph_t_plus_1",
        "provenance", "supervision",
    ):
        if key not in transition:
            raise ValueError(f"Canonical transition missing required key {key!r}.")

    if transition.get("schema_version") != "mssr.canonical_transition.v5":
        raise ValueError("Materializer requires canonical transition v5.")

    provenance = _mapping(transition.get("provenance"))
    actual_index = provenance.get("canonical_transition_index")
    if actual_index != expected_index:
        raise ValueError(
            "Non-contiguous canonical_transition_index: "
            f"expected {expected_index}, got {actual_index!r}."
        )

    graph_t = _mapping(transition.get("graph_t"))
    node_ids = {str(node.get("module_id")) for node in graph_t.get("nodes", [])}
    action = _mapping(transition.get("action_t"))
    modules = action.get("modules")
    if not isinstance(modules, Mapping) or set(modules) != node_ids:
        raise ValueError("Canonical joint action must contain exactly one action object per graph module.")

    supervision = _mapping(transition.get("supervision"))
    masks = supervision.get("module_resource_masks")
    if not isinstance(masks, Mapping) or set(masks) != node_ids:
        raise ValueError("module_resource_masks must cover exactly the graph modules.")
    for module_id, resource_masks in masks.items():
        if not isinstance(resource_masks, Mapping):
            raise ValueError(f"Invalid resource mask for {module_id}.")
        if set(resource_masks) != {"wheels", "internal", "structural"}:
            raise ValueError(f"Incomplete resource mask for {module_id}.")
        if not all(type(value) is bool for value in resource_masks.values()):
            raise ValueError(f"Resource masks must be boolean for {module_id}.")

    bc_valid = supervision.get("valid_for_behavior_cloning") is True
    expected_bc = any(value for resource_masks in masks.values() for value in resource_masks.values())
    if bc_valid != expected_bc:
        raise ValueError("BC validity does not match module resource masks.")

    coverage = provenance.get("source_repeat_coverage", 1)
    if not isinstance(coverage, int) or coverage < 1:
        raise ValueError(f"Invalid source_repeat_coverage={coverage!r}.")

    action_types = set()
    for module_action in modules.values():
        wheels = _mapping(module_action.get("wheels"))
        if wheels.get("commanded"):
            action_types.add("wheel_velocity")
        internal = _mapping(module_action.get("internal"))
        if internal.get("commanded") and internal.get("mode"):
            action_types.add(str(internal["mode"]))
        structural = _mapping(module_action.get("structural"))
        if structural.get("commanded") and structural.get("primitive"):
            action_types.add(str(structural["primitive"]))

    morphology_raw = graph_t.get("morphology")
    morphology = None if morphology_raw is None else str(morphology_raw)
    return bc_valid, coverage, "physical_timestep", tuple(sorted(action_types)), morphology


def _materialize_episode(
    spec: EpisodeSpec,
    *,
    repo_root: Path,
    episodes_dir: Path,
) -> dict[str, Any]:
    filename = _safe_episode_filename(spec.episode_id)
    final_path = episodes_dir / filename
    temporary_path = episodes_dir / f".{filename}.tmp"

    counts = Counter()
    transition_kinds = Counter()
    command_types = Counter()
    command_instances = Counter()
    morphologies = Counter()
    source_paths: set[str] = set()
    digest = hashlib.sha256()
    previous_decision_stamp: float | None = None

    try:
        with temporary_path.open("wb") as handle:
            for canonical_index, transition in enumerate(
                iter_canonical_episode(spec, repo_root=repo_root)
            ):
                (
                    bc_valid,
                    repeat_count,
                    transition_kind,
                    channels,
                    morphology,
                ) = _transition_stats(
                    transition,
                    expected_index=canonical_index,
                )

                counts["canonical_transitions"] += 1
                counts["effective_weight"] += repeat_count
                if bc_valid:
                    counts["bc_valid_transitions"] += 1
                    counts["bc_effective_weight"] += repeat_count
                supervision = transition["supervision"]
                masks = supervision["module_resource_masks"]
                counts["command_bc_transitions"] += int(bc_valid)
                counts["configuration_bc_transitions"] += 0
                counts["resource_conflicts"] += len(supervision.get("conflicts") or [])
                counts["unresolved_recorded_sequences"] += len(
                    transition["provenance"].get("unresolved_recorded_sequences") or []
                )

                transition_kinds[transition_kind] += 1
                for channel in channels:
                    command_types[channel] += 1
                for module_action in transition["action_t"]["modules"].values():
                    if module_action["wheels"]["commanded"]:
                        command_instances["wheel_velocity"] += 1
                    if module_action["internal"]["commanded"]:
                        command_instances[str(module_action["internal"]["mode"])] += 1
                    if module_action["structural"]["commanded"]:
                        command_instances[str(module_action["structural"]["primitive"])] += 1
                if morphology is not None:
                    morphologies[morphology] += 1

                provenance = _mapping(transition.get("provenance"))
                decision_stamp = provenance.get("decision_stamp")
                if isinstance(decision_stamp, (int, float)):
                    decision_stamp = float(decision_stamp)
                    if (previous_decision_stamp is not None and
                            decision_stamp < previous_decision_stamp - 1e-9):
                        raise ValueError(
                            f"Episode {spec.episode_id!r} has reversed decision time "
                            f"at canonical index {canonical_index}: "
                            f"{previous_decision_stamp} -> {decision_stamp}."
                        )
                    previous_decision_stamp = decision_stamp
                for field in (
                    "loader_source_path",
                ):
                    value = provenance.get(field)
                    if isinstance(value, str) and value:
                        source_paths.add(value)

                payload = _json_line(transition)
                handle.write(payload)
                digest.update(payload)

            handle.flush()
            os.fsync(handle.fileno())

        if counts["canonical_transitions"] == 0:
            raise ValueError(
                f"Episode {spec.episode_id!r} produced zero canonical transitions."
            )

        temporary_path.replace(final_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    entry = {
        "episode_id": spec.episode_id,
        "kind": spec.kind,
        "task": spec.task,
        "seed": spec.seed,
        "file": f"episodes/{filename}",
        "sha256": digest.hexdigest(),
        "bytes": final_path.stat().st_size,
        "canonical_transitions": counts["canonical_transitions"],
        "bc_valid_transitions": counts["bc_valid_transitions"],
        "command_bc_transitions": counts["command_bc_transitions"],
        "configuration_bc_transitions": counts["configuration_bc_transitions"],
        "effective_weight": counts["effective_weight"],
        "bc_effective_weight": counts["bc_effective_weight"],
        "resource_conflicts": counts["resource_conflicts"],
        "unresolved_recorded_sequences": counts["unresolved_recorded_sequences"],
        "transition_kinds": dict(sorted(transition_kinds.items())),
        "command_types": dict(sorted(command_types.items())),
        "command_instances": dict(sorted(command_instances.items())),
        "morphologies": dict(sorted(morphologies.items())),
        "source_paths": sorted(source_paths),
    }
    geometry_path = Path(spec.path) / "environment_geometry.json"
    if geometry_path.is_file():
        entry["environment_geometry"] = {
            "path": _manifest_path(geometry_path.resolve(), repo_root),
            "sha256": _sha256_file(geometry_path),
        }
    return entry


def _sum_nested_counts(
    episodes: Iterable[Mapping[str, Any]],
    field: str,
) -> dict[str, int]:
    result = Counter()
    for episode in episodes:
        value = episode.get(field)
        if isinstance(value, Mapping):
            for key, count in value.items():
                result[str(key)] += int(count)
    return dict(sorted(result.items()))


def _code_hashes(repo_root: Path) -> dict[str, str]:
    relative_paths = (
        Path("drl/IL/preprocessing.py"),
        Path("drl/IL/dataset_loader.py"),
        Path("drl/IL/canonical_dataset.py"),
        Path("drl/IL/materialize_canonical_dataset.py"),
        Path("mssr_ws/src/mssr_expert/mssr_expert/execution/parallel_assembly_executor.py"),
    )
    return {
        str(path): _sha256_file(repo_root / path)
        for path in relative_paths
    }


def _publish_materialized_dataset(
    staging_root: Path,
    output_root: Path,
    *,
    overwrite: bool,
) -> None:
    """Publish a complete collection at the requested path, restoring on failure."""
    if not staging_root.is_dir():
        raise FileNotFoundError(f"Staging dataset not found: {staging_root}")
    if not output_root.exists():
        staging_root.replace(output_root)
        return
    if not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output_root}")

    backup_root = output_root.with_name(f".{output_root.name}.backup-{os.getpid()}")
    if backup_root.exists():
        raise FileExistsError(f"Refusing to replace dataset with stale backup: {backup_root}")
    output_root.replace(backup_root)
    try:
        staging_root.replace(output_root)
    except BaseException:
        backup_root.replace(output_root)
        raise
    shutil.rmtree(backup_root)


def materialize_canonical_dataset(
    *,
    repo_root: Path,
    source_root: Path,
    output_root: Path,
    episode_ids: set[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    source_root = source_root.resolve()
    output_root = output_root.resolve()

    if output_root.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing canonical dataset: {output_root}"
        )
    if output_root.exists() and episode_ids is not None:
        raise ValueError(
            "Replacing an existing collection requires all episodes; "
            "omit --episode-id or choose a new output path."
        )

    compact_manifest_path = source_root / "manifest.json"
    if not compact_manifest_path.is_file():
        raise FileNotFoundError(
            f"Compact collection manifest not found: {compact_manifest_path}"
        )

    compact_manifest = json.loads(
        compact_manifest_path.read_text(encoding="utf-8")
    )

    specs = discover_compact_episode_specs(source_root)
    if episode_ids is not None:
        specs = [spec for spec in specs if spec.episode_id in episode_ids]
        missing = episode_ids.difference(spec.episode_id for spec in specs)
        if missing:
            raise ValueError(
                "Unknown requested episode IDs: " + ", ".join(sorted(missing))
            )

    specs = sorted(specs, key=lambda spec: spec.episode_id)
    if not specs:
        raise ValueError("No logical episodes selected for materialization.")

    temporary_root = output_root.with_name(
        f".{output_root.name}.tmp-{os.getpid()}"
    )
    if temporary_root.exists():
        raise FileExistsError(
            f"Refusing to remove an existing staging dataset: {temporary_root}"
        )

    episodes_dir = temporary_root / "episodes"
    episodes_dir.mkdir(parents=True)

    episode_entries: list[dict[str, Any]] = []

    try:
        for index, spec in enumerate(specs, start=1):
            entry = _materialize_episode(
                spec,
                repo_root=repo_root,
                episodes_dir=episodes_dir,
            )
            episode_entries.append(entry)
            print(
                f"[{index:02d}/{len(specs):02d}] "
                f"{spec.episode_id}  "
                f"canonical={entry['canonical_transitions']}  "
                f"BC={entry['bc_valid_transitions']}  "
                f"bytes={entry['bytes']}"
            )

        aggregate = Counter()
        kind_counts = Counter()
        for entry in episode_entries:
            aggregate["canonical_transitions"] += int(
                entry["canonical_transitions"]
            )
            aggregate["bc_valid_transitions"] += int(
                entry["bc_valid_transitions"]
            )
            aggregate["effective_weight"] += int(entry["effective_weight"])
            aggregate["bc_effective_weight"] += int(
                entry["bc_effective_weight"]
            )
            for name in ("command_bc_transitions", "configuration_bc_transitions",
                         "resource_conflicts", "unresolved_recorded_sequences"):
                aggregate[name] += entry[name]
            kind_counts[str(entry["kind"])] += 1

        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "dataset_name": output_root.name,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "format": "one UTF-8 canonical JSONL file per logical episode",
            "contract": {
                "transition": "one unique physical snapshot: (G_t, o_t, joint a_t, G_t_plus_1)",
                "episode_boundary_policy": "logical episodes preserved; source phase terminals are provenance only",
                "timestep_policy": (
                    "all decoded records sharing one graph_t.stamp are one policy timestep; "
                    "parallel module commands are fused and the last continuous revision per resource wins"
                ),
                "action_policy": (
                    "exactly one action object per module; wheels plus one coupled internal PAN-or-TILT resource "
                    "plus one initiated structural skill; omitted resources are unsupervised, never implicit WAIT/STOP"
                ),
                "internal_resource_policy": (
                    "PAN and TILT are mutually exclusive for one module at one physical timestep; "
                    "parallel internal actions on different modules are allowed"
                ),
                "structural_skill_policy": (
                    "align_faces, dock, undock and gravity_settle remain executor tools; "
                    "align_faces phases/retries are executor internals, not separate policy targets"
                ),
                "executor_state_policy": (
                    "G_t node control_state exposes inferred resource availability, active primitives and inferred "
                    "structural_hold; active_goal_ids is used only when explicitly recorded and other lifecycle "
                    "evidence is labelled rather than assumed"
                ),
                "legacy_sequence_policy": (
                    "old button IK primitive_sequence records are retained in provenance but never supervise "
                    "low-level BC because their intermediate states were not recorded; re-recording is required"
                ),
                "observation_policy": (
                    "current obstacle identity and root-relative geometry only; static world geometry stays in source metadata"
                ),
                "split_applied": False,
                "tensorization_applied": False,
                "normalization_applied": False,
                "units": {"position": "m", "orientation": "quaternion_xyzw",
                          "joint_position": "rad", "joint_velocity": "rad/s"},
                "observation_frame": "root_module body frame; unavailable during structural transitions",
                "static_context": "read existing compact phase/stream metadata and episode geometry",
                "temporal_policy": (
                    "source decisions ordered by decision_stamp; each G_t_plus_1 is its recorded "
                    "next state; only identical-state same-time records with identical recorded "
                    "next states can coalesce; "
                    "source_records and source_repeat_count retain coverage of the original rows"
                ),
                "split_applied": False,
                "tensorization_applied": False,
                "normalization_applied": False,
            },
            "source_dataset": {
                "path": _manifest_path(source_root, repo_root),
                "manifest": _manifest_path(compact_manifest_path, repo_root),
                "manifest_sha256": _sha256_file(compact_manifest_path),
                "schema_version": compact_manifest.get("schema_version"),
                "source_collection": compact_manifest.get("source_collection"),
                "source_collection_sha256": compact_manifest.get(
                    "source_collection_sha256"
                ),
            },
            "producer_code_sha256": _code_hashes(repo_root),
            "totals": {
                "episodes": len(episode_entries),
                "episodes_by_kind": dict(sorted(kind_counts.items())),
                "canonical_transitions": aggregate["canonical_transitions"],
                "bc_valid_transitions": aggregate["bc_valid_transitions"],
                "command_bc_transitions": aggregate["command_bc_transitions"],
                "configuration_bc_transitions": aggregate["configuration_bc_transitions"],
                "resource_conflicts": aggregate["resource_conflicts"],
                "unresolved_recorded_sequences": aggregate["unresolved_recorded_sequences"],
                "effective_weight": aggregate["effective_weight"],
                "bc_effective_weight": aggregate["bc_effective_weight"],
                "transition_kinds": _sum_nested_counts(
                    episode_entries, "transition_kinds"
                ),
                "command_types": _sum_nested_counts(
                    episode_entries, "command_types"
                ),
                "command_instances": _sum_nested_counts(
                    episode_entries, "command_instances"
                ),
                "morphologies": _sum_nested_counts(
                    episode_entries, "morphologies"
                ),
            },
            "episodes": episode_entries,
        }

        manifest_path = temporary_root / "manifest.json"
        manifest_bytes = (
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        manifest_path.write_bytes(manifest_bytes)

        _publish_materialized_dataset(
            temporary_root, output_root, overwrite=overwrite
        )
        return manifest
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("datasets/expert_v1_compact"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/expert_v1_canonical"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the existing complete canonical collection at --output.",
    )
    parser.add_argument(
        "--episode-id",
        action="append",
        default=None,
        help="Materialize only this logical episode; repeat to select several.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    source_root = (
        args.source if args.source.is_absolute() else repo_root / args.source
    )
    output_root = (
        args.output if args.output.is_absolute() else repo_root / args.output
    )

    manifest = materialize_canonical_dataset(
        repo_root=repo_root,
        source_root=source_root,
        output_root=output_root,
        episode_ids=None if args.episode_id is None else set(args.episode_id),
        overwrite=args.overwrite,
    )

    totals = manifest["totals"]
    print()
    print("MATERIALIZATION COMPLETE")
    print("  output               :", output_root)
    print("  episodes             :", totals["episodes"])
    print("  canonical transitions:", totals["canonical_transitions"])
    print("  BC-valid             :", totals["bc_valid_transitions"])
    print("  effective weight     :", totals["effective_weight"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
