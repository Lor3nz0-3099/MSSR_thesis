"""Chronological loaders for the compact MSSR imitation-learning dataset.

The compact datasets remain immutable.  This module restores only information
that the compact manifests prove constant, preserves RLE/source-span metadata,
and merges the physical substreams that belong to one logical episode.

No action or transition is synthesized at a phase boundary.
"""
from __future__ import annotations

import copy
import heapq
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from preprocessing import PreprocessingError, merge_episode_record_streams


EXPERT_PHASE_ORDER = ("assembly", "behavior", "manipulation")


@dataclass(frozen=True)
class EpisodeSpec:
    episode_id: str
    kind: str
    path: Path
    task: str | None = None
    seed: int | None = None


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreprocessingError(f"Cannot read JSON manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreprocessingError(f"Manifest {path} is not a JSON object.")
    return value


def _merge_constants(
    dynamic: Mapping[str, Any] | None,
    constants: Mapping[str, Any] | None,
    *,
    label: str,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(constants or {}))
    if dynamic is None:
        return result
    if not isinstance(dynamic, Mapping):
        raise PreprocessingError(f"{label} must be a mapping.")
    for key, value in dynamic.items():
        if key in result and result[key] != value:
            raise PreprocessingError(
                f"{label}.{key} conflicts with a manifest constant."
            )
        result[key] = copy.deepcopy(value)
    return result


def _restore_observation(
    row: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    source_key: str,
    next_state: bool,
) -> dict[str, Any] | None:
    raw = row.get(source_key)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise PreprocessingError(f"{source_key} must be a mapping when present.")

    if next_state:
        constants = manifest.get("next_observation_constants")
        if not isinstance(constants, Mapping):
            constants = manifest.get("observation_constants")
        runtime_constants = manifest.get("next_runtime_state_constants")
        if not isinstance(runtime_constants, Mapping):
            runtime_constants = manifest.get("runtime_state_constants")
    else:
        constants = manifest.get("observation_constants")
        runtime_constants = manifest.get("runtime_state_constants")

    result = _merge_constants(raw, constants, label=source_key)

    dynamic_runtime = raw.get("runtime_state")
    if isinstance(runtime_constants, Mapping) or isinstance(dynamic_runtime, Mapping):
        result["runtime_state"] = _merge_constants(
            dynamic_runtime if isinstance(dynamic_runtime, Mapping) else {},
            runtime_constants if isinstance(runtime_constants, Mapping) else {},
            label=f"{source_key}.runtime_state",
        )

    return result


def decode_compact_expert_record(
    compact_record: Mapping[str, Any],
    phase_manifest: Mapping[str, Any],
    *,
    phase: str | None = None,
) -> dict[str, Any]:
    """Restore manifest-proven constants without expanding RLE repeats.

    Fields intentionally removed as graph aliases during compaction (for
    example assembly ``observation.modules`` or manipulation runtime module
    copies) are *not* regenerated: canonical preprocessing consumes the graph
    directly, so recreating duplicate aliases would add memory without adding
    information.
    """
    if not isinstance(compact_record, Mapping):
        raise PreprocessingError("Compact expert record must be a mapping.")

    row = copy.deepcopy(dict(compact_record))
    constants = phase_manifest.get("top_level_constants")
    if isinstance(constants, Mapping):
        for key, value in constants.items():
            if key in row and row[key] != value:
                raise PreprocessingError(
                    f"Compact row field {key!r} conflicts with phase metadata."
                )
            row.setdefault(key, copy.deepcopy(value))

    observation = _restore_observation(
        row,
        phase_manifest,
        source_key="observation",
        next_state=False,
    )
    if observation is not None:
        row["observation"] = observation

    next_observation = _restore_observation(
        row,
        phase_manifest,
        source_key="observation_t_plus_1",
        next_state=True,
    )
    if next_observation is not None:
        row["observation_t_plus_1"] = next_observation

    repeat_count = row.get("_source_repeat_count", 1)
    if not isinstance(repeat_count, int) or repeat_count < 1:
        raise PreprocessingError("_source_repeat_count must be a positive integer.")

    row["_loader_source_kind"] = "expert"
    row["_loader_phase"] = phase or str(phase_manifest.get("phase") or "")
    return row


def iter_compact_expert_phase(
    episode_dir: Path | str,
    phase: str,
) -> Iterator[dict[str, Any]]:
    episode_dir = Path(episode_dir)
    data_path = episode_dir / f"{phase}.jsonl"
    manifest_path = episode_dir / f"{phase}_manifest.json"

    if not data_path.is_file() or not manifest_path.is_file():
        raise PreprocessingError(
            f"Compact phase {phase!r} is incomplete in {episode_dir}."
        )

    manifest = _json(manifest_path)
    previous_start: float | None = None

    with data_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise PreprocessingError(f"Blank line in {data_path}:{line_number}.")
            try:
                compact = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PreprocessingError(
                    f"Invalid JSON in {data_path}:{line_number}: {exc}"
                ) from exc

            row = decode_compact_expert_record(compact, manifest, phase=phase)
            row["_loader_source_path"] = str(data_path)
            row["_loader_source_row"] = line_number
            stamp = _record_start_stamp(row)
            if previous_start is not None and stamp < previous_start:
                raise PreprocessingError(
                    f"Non-monotonic compact phase {data_path} at line {line_number}."
                )
            previous_start = stamp
            yield row


def _record_start_stamp(record: Mapping[str, Any]) -> float:
    value = record.get("_source_stamp_start", record.get("stamp"))
    if not isinstance(value, (int, float)):
        raise PreprocessingError("Episode record has no numeric source/start stamp.")
    return float(value)


def _record_end_stamp(record: Mapping[str, Any]) -> float:
    value = record.get("_source_stamp_end", record.get("stamp"))
    if not isinstance(value, (int, float)):
        raise PreprocessingError("Episode record has no numeric source/end stamp.")
    return float(value)


def _merge_expert_phase_streams(
    streams: Sequence[tuple[str, Iterable[Mapping[str, Any]]]],
) -> Iterator[Mapping[str, Any]]:
    """Chronologically merge compact expert phases by their original spans."""
    priority = {name: index for index, name in enumerate(EXPERT_PHASE_ORDER)}
    heap: list[tuple[float, int, int, int, dict[str, Any]]] = []

    for stream_id, (phase, records) in enumerate(streams):
        iterator = iter(records)
        try:
            row = next(iterator)
        except StopIteration:
            continue
        state = {
            "phase": phase,
            "iterator": iterator,
            "row": row,
            "sequence": 0,
            "last_start": _record_start_stamp(row),
        }
        heapq.heappush(
            heap,
            (
                state["last_start"],
                priority.get(phase, len(priority)),
                stream_id,
                0,
                state,
            ),
        )

    previous_global_start: float | None = None
    previous_global_end: float | None = None
    previous_phase: str | None = None

    while heap:
        start, _, stream_id, _, state = heapq.heappop(heap)
        row = state["row"]
        end = _record_end_stamp(row)

        if end < start:
            raise PreprocessingError(
                f"Source span is reversed in phase {state['phase']!r}: {start} -> {end}."
            )
        if previous_global_start is not None and start < previous_global_start:
            raise PreprocessingError("Merged expert episode is not temporally monotonic.")
        if (
            previous_global_end is not None
            and start < previous_global_end
            and state["phase"] != previous_phase
        ):
            raise PreprocessingError(
                "Expert phase source spans overlap after compaction: "
                f"{previous_phase!r} -> {state['phase']!r} at {start:.6f}s."
            )

        previous_global_start = start
        previous_global_end = max(previous_global_end or end, end)
        previous_phase = str(state["phase"])
        yield row

        try:
            next_row = next(state["iterator"])
        except StopIteration:
            continue

        next_start = _record_start_stamp(next_row)
        if next_start < state["last_start"]:
            raise PreprocessingError(
                f"Phase {state['phase']!r} is not temporally monotonic."
            )
        state["last_start"] = next_start
        state["sequence"] += 1
        state["row"] = next_row
        heapq.heappush(
            heap,
            (
                next_start,
                priority.get(str(state["phase"]), len(priority)),
                stream_id,
                state["sequence"],
                state,
            ),
        )


def iter_compact_expert_episode(
    episode_dir: Path | str,
) -> Iterator[Mapping[str, Any]]:
    """Yield one logical expert episode with all available phases time-merged."""
    episode_dir = Path(episode_dir)
    episode_manifest = _json(episode_dir / "manifest.json")

    declared = episode_manifest.get("phase_order")
    if isinstance(declared, list):
        phases = [str(x) for x in declared]
    else:
        phase_map = episode_manifest.get("phases")
        available = set(phase_map) if isinstance(phase_map, Mapping) else set()
        phases = [phase for phase in EXPERT_PHASE_ORDER if phase in available]

    # Older manifests can lack phase_order.  The physical files remain the
    # authoritative indication that a supported phase is present.
    for phase in EXPERT_PHASE_ORDER:
        if (
            phase not in phases
            and (episode_dir / f"{phase}.jsonl").is_file()
            and (episode_dir / f"{phase}_manifest.json").is_file()
        ):
            phases.append(phase)

    if not phases:
        raise PreprocessingError(f"Expert episode {episode_dir} has no phases.")

    streams = [
        (phase, iter_compact_expert_phase(episode_dir, phase))
        for phase in phases
    ]
    yield from _merge_expert_phase_streams(streams)


def _load_teleop_codec(repo_root: Path):
    tools_dir = repo_root / "scripts/dataset_tools"
    tools_str = str(tools_dir)
    if tools_str not in sys.path:
        sys.path.insert(0, tools_str)
    try:
        from teleop_semantic_codec import iter_decoded_archive
    except ImportError as exc:
        raise PreprocessingError(
            f"Cannot import teleop_semantic_codec from {tools_dir}."
        ) from exc
    return iter_decoded_archive


def _resolve_archive_path(
    raw_path: str,
    *,
    repo_root: Path,
    episode_dir: Path,
) -> Path:
    path = Path(raw_path)
    candidates = [path] if path.is_absolute() else [repo_root / path]
    candidates.append(episode_dir / path.name)
    if "structural" in path.parts:
        candidates.append(episode_dir / "structural" / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise PreprocessingError(
        f"Cannot resolve compact teleop archive {raw_path!r} from {episode_dir}."
    )


def iter_compact_teleop_episode(
    episode_dir: Path | str,
    *,
    repo_root: Path | str | None = None,
) -> Iterator[Mapping[str, Any]]:
    """Yield human + every structural macro of one teleop episode chronologically."""
    episode_dir = Path(episode_dir)
    root = Path.cwd() if repo_root is None else Path(repo_root)
    manifest = _json(episode_dir / "manifest.json")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise PreprocessingError(f"Teleop episode {episode_dir} has no compact files.")

    iter_decoded_archive = _load_teleop_codec(root)
    streams = []

    for meta in files:
        if not isinstance(meta, Mapping):
            raise PreprocessingError("Teleop manifest file entry is not a mapping.")
        archive_raw = meta.get("archive_path")
        if not isinstance(archive_raw, str) or not archive_raw:
            raise PreprocessingError("Teleop manifest file entry has no archive_path.")
        archive_path = _resolve_archive_path(
            archive_raw,
            repo_root=root,
            episode_dir=episode_dir,
        )
        source_path = str(meta.get("source_path") or archive_raw)
        kind = "structural" if "/structural/" in source_path.replace("\\", "/") else "human"

        def annotated_records(
            archive: Path = archive_path,
            metadata: Mapping[str, Any] = meta,
            stream_kind: str = kind,
            source: str = source_path,
        ) -> Iterator[Mapping[str, Any]]:
            for line_number, original in enumerate(iter_decoded_archive(archive, metadata), start=1):
                row = copy.deepcopy(dict(original))
                row["_loader_source_kind"] = stream_kind
                row["_loader_source_path"] = source
                row["_loader_source_row"] = line_number
                yield row

        streams.append((kind, annotated_records()))

    yield from merge_episode_record_streams(streams)


def discover_compact_episode_specs(
    compact_root: Path | str,
) -> list[EpisodeSpec]:
    """Discover expert and teleop logical episodes under expert_v1_compact."""
    compact_root = Path(compact_root)
    specs: list[EpisodeSpec] = []

    dataset_manifest = _json(compact_root / "manifest.json")
    expert_episodes = dataset_manifest.get("episodes")
    if not isinstance(expert_episodes, list):
        raise PreprocessingError("Compact expert manifest has no episodes list.")

    for item in expert_episodes:
        if not isinstance(item, Mapping):
            raise PreprocessingError("Compact expert episode entry is not a mapping.")
        task = str(item.get("task") or "")
        seed = int(item["seed"])
        rel = item.get("path")
        if not isinstance(rel, str) or not rel:
            raise PreprocessingError("Compact expert episode entry has no path.")
        specs.append(
            EpisodeSpec(
                episode_id=f"expert:{task}:seed-{seed:06d}",
                kind="expert",
                task=task,
                seed=seed,
                path=compact_root / rel,
            )
        )

    teleop_root = compact_root / "teleop" / "episodes"
    if teleop_root.is_dir():
        for episode_dir in sorted(p for p in teleop_root.iterdir() if p.is_dir()):
            if not (episode_dir / "manifest.json").is_file():
                continue
            specs.append(
                EpisodeSpec(
                    episode_id=episode_dir.name,
                    kind="teleop",
                    task="teleop",
                    seed=None,
                    path=episode_dir,
                )
            )

    ids = [spec.episode_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise PreprocessingError("Duplicate logical episode IDs were discovered.")

    return sorted(specs, key=lambda spec: spec.episode_id)


def iter_episode_records(
    spec: EpisodeSpec,
    *,
    repo_root: Path | str | None = None,
) -> Iterator[Mapping[str, Any]]:
    if spec.kind == "expert":
        yield from iter_compact_expert_episode(spec.path)
        return
    if spec.kind == "teleop":
        yield from iter_compact_teleop_episode(spec.path, repo_root=repo_root)
        return
    raise PreprocessingError(f"Unsupported episode kind: {spec.kind!r}")
