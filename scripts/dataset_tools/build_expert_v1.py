#!/usr/bin/env python3
"""Discover, validate, import, and compact MSSR expert demonstrations.

The raw ``expert_v1`` dataset is authoritative.  Compact files are derived
without changing an expert transition: every retained row still contains
``graph_t``, ``expert_action``, and ``graph_t_plus_1``.  Only proved aliases,
proved graph-derivable fields, per-stream constants, and consecutive repeated
transitions are removed.

The command is intentionally idempotent.  Completed successful runs already
represented by ``(task, seed)`` are left alone; a later invocation imports
newly completed campaigns.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


PHASES_BY_TASK = {
    "stairs": ("assembly", "behavior"),
    "gap": ("assembly", "behavior"),
    "rc_car": ("assembly", "behavior"),
    "button": ("assembly", "behavior", "manipulation"),
}

# Stair recovery campaigns intentionally omit self-assembly: the Snake8
# assembly task is invariant and is already represented by the canonical
# assembly corpus.  The episode-specific supervision is the stair behavior.
# Keep assembly as a supported phase when a complete run records it, but do
# not reject a successful stairs episode that contains behavior only.
REQUIRED_PHASES_BY_TASK = {
    "stairs": ("behavior",),
    "gap": ("assembly", "behavior"),
    "rc_car": ("assembly", "behavior"),
    "button": ("assembly", "behavior", "manipulation"),
}

RESULT_SCHEMA_TO_TASK = {
    "mssr.stair_headless_episode.v1": "stairs",
    "mssr.gap_headless_episode.v1": "gap",
    "mssr.rc_car_nav2_episode.v1": "rc_car",
}

TOP_LEVEL_CONSTANT_CANDIDATES = (
    "schema_version",
    "episode_id",
    "stage_name",
    "task_type",
    "difficulty",
    "target_graph",
    "assignment_target_to_module",
    "module_roles",
)

EXACT_ALIAS_FIELDS = {
    "attributed_graph": "graph_t",
    "attributed_task_graph": "task_graph_t",
}

RLE_IGNORED_KEYS = frozenset(("stage_id", "stamp", "timestep"))
SOURCE_RLE_PREFIX = "_source_"
SEED_RE = re.compile(r"seed-(\d+)")


class DatasetError(RuntimeError):
    """Raised when a source would make the curated dataset ambiguous."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    else:
        text = json.dumps(value, separators=(",", ":")) + "\n"
    return text.encode("utf-8")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(json_bytes(value, pretty=True))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def record_seed(path: Path, result: dict[str, Any]) -> int | None:
    seed = result.get("seed")
    if isinstance(seed, int):
        return seed
    for key in ("stair", "gap", "physical_track", "route"):
        item = result.get(key)
        if isinstance(item, dict) and isinstance(item.get("seed"), int):
            return int(item["seed"])
    match = SEED_RE.search(path.parent.name)
    return int(match.group(1)) if match else None


@dataclasses.dataclass
class Candidate:
    task: str
    seed: int
    run_dir: Path
    completion_path: Path | None
    result: dict[str, Any]
    phases: tuple[str, ...]
    successful: bool
    stable: bool
    source_gui_run: bool
    completion_mtime_ns: int
    classification: str = "unclassified"
    reason: str = ""

    @property
    def key(self) -> tuple[str, int]:
        return self.task, self.seed

    @property
    def variant_label(self) -> str:
        return self.run_dir.name

    def as_audit(self, root: Path) -> dict[str, Any]:
        return {
            "task": self.task,
            "seed": self.seed,
            "run_dir": relative(self.run_dir, root),
            "completion_path": (
                relative(self.completion_path, root) if self.completion_path else None
            ),
            "completion_schema_version": self.result.get("schema_version"),
            "completion_mtime_ns": self.completion_mtime_ns,
            "phases_present": list(self.phases),
            "successful": self.successful,
            "stable": self.stable,
            "source_gui_run": self.source_gui_run,
            "variant_label": self.variant_label,
            "classification": self.classification,
            "reason": self.reason,
        }


def candidate_from_result(path: Path) -> Candidate | None:
    try:
        result = json_load(path)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(result, dict):
        return None
    task = RESULT_SCHEMA_TO_TASK.get(str(result.get("schema_version")))
    if task is None:
        return None
    seed = record_seed(path, result)
    if seed is None:
        return None
    supported = PHASES_BY_TASK[task]
    required = REQUIRED_PHASES_BY_TASK[task]
    phases = tuple(
        phase
        for phase in supported
        if (path.parent / f"{phase}_dataset.jsonl").is_file()
    )
    successful = bool(result.get("success"))
    if task == "rc_car" and result.get("dataset_terminal_success") is False:
        successful = False
    completion_mtime_ns = path.stat().st_mtime_ns
    phase_paths = [path.parent / f"{phase}_dataset.jsonl" for phase in phases]
    stable = (
        all(phase in phases for phase in required)
        and all(item.stat().st_size > 0 for item in phase_paths)
        and all(item.stat().st_mtime_ns <= completion_mtime_ns for item in phase_paths)
    )
    lower_path = path.parent.as_posix().lower()
    return Candidate(
        task=task,
        seed=seed,
        run_dir=path.parent,
        completion_path=path,
        result=result,
        phases=phases,
        successful=successful,
        stable=stable,
        source_gui_run="gui" in lower_path,
        completion_mtime_ns=completion_mtime_ns,
    )


def candidate_from_button_dir(path: Path) -> Candidate | None:
    match = SEED_RE.match(path.name)
    if match is None:
        return None
    seed = int(match.group(1))
    complete = path / "expert_complete.json"
    result: dict[str, Any] = {}
    completion_mtime_ns = path.stat().st_mtime_ns
    if complete.is_file():
        try:
            loaded = json_load(complete)
            if isinstance(loaded, dict):
                result = loaded
            completion_mtime_ns = complete.stat().st_mtime_ns
        except (OSError, json.JSONDecodeError):
            pass
    required = PHASES_BY_TASK["button"]
    phases = tuple(
        phase for phase in required if (path / f"{phase}_dataset.jsonl").is_file()
    )
    phase_paths = [path / f"{phase}_dataset.jsonl" for phase in phases]
    successful = bool(
        result.get("success")
        and result.get("button_pressed")
        and result.get("final_morphology") == "rc_car8"
        and isinstance(result.get("return_reconfiguration"), dict)
        and result["return_reconfiguration"].get("success")
    )
    stable = (
        complete.is_file()
        and phases == required
        and all(item.stat().st_size > 0 for item in phase_paths)
        and all(item.stat().st_mtime_ns <= completion_mtime_ns for item in phase_paths)
    )
    return Candidate(
        task="button",
        seed=seed,
        run_dir=path,
        completion_path=complete if complete.is_file() else None,
        result=result,
        phases=phases,
        successful=successful,
        stable=stable,
        source_gui_run=False,
        completion_mtime_ns=completion_mtime_ns,
    )


def discover_candidates(log_root: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    for path in sorted(log_root.rglob("result.json")):
        candidate = candidate_from_result(path)
        if candidate is not None:
            candidates.append(candidate)
    button_root = log_root / "button_expert_to_ik"
    if button_root.is_dir():
        for path in sorted(item for item in button_root.iterdir() if item.is_dir()):
            candidate = candidate_from_button_dir(path)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def classify_candidates(
    candidates: list[Candidate], existing_sources: dict[tuple[str, int], Path | None]
) -> list[Candidate]:
    valid_by_key: dict[tuple[str, int], list[Candidate]] = {}
    for candidate in candidates:
        if candidate.key in existing_sources:
            canonical_source = existing_sources[candidate.key]
            if canonical_source is not None and candidate.run_dir.resolve() == canonical_source:
                candidate.classification = "canonical_source"
                candidate.reason = "this run is the canonical source for task/seed"
            elif not candidate.successful:
                candidate.classification = "excluded_failure"
                candidate.reason = "failed attempt for a task/seed that has another canonical run"
            elif not candidate.stable:
                candidate.classification = "excluded_incomplete"
                candidate.reason = "incomplete attempt for a task/seed that is already canonical"
            else:
                candidate.classification = "excluded_noncanonical_duplicate"
                candidate.reason = "another completed run is the immutable canonical source"
        elif not candidate.successful:
            candidate.classification = "excluded_failure"
            candidate.reason = "completion artifact does not certify task success"
        elif not candidate.stable:
            candidate.classification = "excluded_incomplete"
            candidate.reason = "required phases are absent, empty, or newer than completion"
        else:
            valid_by_key.setdefault(candidate.key, []).append(candidate)

    selected: list[Candidate] = []
    for key, group in sorted(valid_by_key.items()):
        group.sort(key=lambda item: (item.completion_mtime_ns, item.run_dir.as_posix()))
        winner = group[-1]
        winner.classification = "selected_for_import"
        winner.reason = "latest complete successful candidate for task/seed"
        selected.append(winner)
        for item in group[:-1]:
            item.classification = "excluded_superseded"
            item.reason = f"superseded by {winner.run_dir.as_posix()}"
    return selected


class MappingTracker:
    """Find keys whose JSON values are exact constants over a stream."""

    def __init__(self) -> None:
        self.records = 0
        self.first_values: dict[str, Any] = {}
        self.all_keys: set[str] = set()
        self.dynamic: set[str] = set()

    def observe(self, value: dict[str, Any]) -> None:
        keys = set(value)
        if self.records == 0:
            self.first_values = copy.deepcopy(value)
            self.all_keys = set(keys)
        else:
            for key in self.all_keys | keys:
                # Once a field is known to be dynamic, comparing its often very
                # large graph/DOF payload again cannot change the conclusion.
                if key in self.dynamic:
                    continue
                if key not in self.first_values or key not in value:
                    self.dynamic.add(key)
                elif self.first_values[key] != value[key]:
                    self.dynamic.add(key)
            self.all_keys.update(keys)
        self.records += 1

    @property
    def constants(self) -> dict[str, Any]:
        return {
            key: self.first_values[key]
            for key in sorted(self.first_values)
            if key not in self.dynamic
        }

    @property
    def dynamic_keys(self) -> list[str]:
        return sorted(self.all_keys - set(self.constants))


def graph_node_map(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            raise DatasetError("graph node is not an object")
        node_id = node.get("module_id") or node.get("node_id") or node.get("id")
        if not isinstance(node_id, str):
            raise DatasetError("graph node has no string module/node id")
        if node_id in result:
            raise DatasetError(f"duplicate graph node id {node_id}")
        attributes = node.get("attributes")
        if not isinstance(attributes, dict):
            raise DatasetError(f"graph node {node_id} has no attributes object")
        result[node_id] = attributes
    if not result:
        raise DatasetError("graph contains no nodes")
    return result


def graph_positions(graph: dict[str, Any]) -> dict[str, Any]:
    positions: dict[str, Any] = {}
    for module_id, attributes in graph_node_map(graph).items():
        if "position" not in attributes:
            raise DatasetError(f"graph node {module_id} has no position")
        positions[module_id] = attributes["position"]
    return positions


def validate_graph(graph: Any, label: str) -> dict[str, Any]:
    if not isinstance(graph, dict):
        raise DatasetError(f"{label} is not an object")
    graph_node_map(graph)
    edges = graph.get("edges")
    if not isinstance(edges, list):
        raise DatasetError(f"{label}.edges is not a list")
    for edge in edges:
        if not isinstance(edge, dict) or not isinstance(edge.get("attributes"), dict):
            raise DatasetError(f"{label} contains an edge without attributes")
    return graph


def module_reconstruction_metadata(
    modules: list[Any], graph: dict[str, Any]
) -> dict[str, Any]:
    nodes = graph_node_map(graph)
    order: list[str] = []
    metadata: dict[str, Any] = {}
    for item in modules:
        if not isinstance(item, dict) or not isinstance(item.get("module_id"), str):
            raise DatasetError("observation.modules contains an invalid module")
        module_id = item["module_id"]
        order.append(module_id)
        if module_id not in nodes:
            raise DatasetError(f"observation module {module_id} is absent from graph")
        module_metadata: dict[str, Any] = {}
        for key in ("body_frame_id", "prim_path"):
            if key in item and key not in nodes[module_id]:
                module_metadata[key] = copy.deepcopy(item[key])
        pose = item.get("pose")
        if isinstance(pose, dict):
            module_metadata["pose_frame_id"] = pose.get("frame_id")
            module_metadata["pose_fields"] = sorted(pose)
        twist = item.get("twist")
        if isinstance(twist, dict):
            module_metadata["twist_fields"] = sorted(twist)
        metadata[module_id] = module_metadata
    if set(order) != set(nodes):
        raise DatasetError("observation.modules and graph nodes have different ids")
    return {"module_order": order, "modules": metadata}


def validate_modules_reconstructible(
    modules: Any,
    graph: dict[str, Any],
    reconstruction: dict[str, Any],
) -> None:
    if not isinstance(modules, list):
        raise DatasetError("observation.modules is not a list")
    nodes = graph_node_map(graph)
    order = reconstruction["module_order"]
    if [item.get("module_id") for item in modules if isinstance(item, dict)] != order:
        raise DatasetError("observation module order changed")
    for item in modules:
        module_id = item["module_id"]
        attributes = nodes.get(module_id)
        if attributes is None:
            raise DatasetError(f"module {module_id} is missing from graph")
        static = reconstruction["modules"][module_id]
        for key, value in item.items():
            if key in attributes:
                if value != attributes[key]:
                    raise DatasetError(f"module {module_id}.{key} differs from graph")
            elif key in ("body_frame_id", "prim_path"):
                if value != static.get(key):
                    raise DatasetError(f"module {module_id}.{key} is not static")
            elif key == "pose":
                expected = {
                    field: (
                        static.get("pose_frame_id")
                        if field == "frame_id"
                        else attributes.get("orientation")
                        if field in ("orientation", "orientation_xyzw")
                        else attributes.get("position")
                        if field == "position"
                        else None
                    )
                    for field in static.get("pose_fields", [])
                }
                if value != expected:
                    raise DatasetError(f"module {module_id}.pose is not graph-derived")
            elif key == "twist":
                expected = {
                    field: (
                        attributes.get("angular_velocity")
                        if field == "angular"
                        else attributes.get("linear_velocity")
                        if field == "linear"
                        else None
                    )
                    for field in static.get("twist_fields", [])
                }
                if value != expected:
                    raise DatasetError(f"module {module_id}.twist is not graph-derived")
            else:
                raise DatasetError(
                    f"module {module_id}.{key} is neither in graph nor reconstruction metadata"
                )


def validate_module_positions(value: Any, graph: dict[str, Any], label: str) -> None:
    if value != graph_positions(graph):
        raise DatasetError(f"{label} is not an exact alias of graph node positions")


def graph_attribute_list(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return [node["attributes"] for node in graph["nodes"]]


@dataclasses.dataclass
class NormalizedObservation:
    body: dict[str, Any]
    runtime_state: dict[str, Any] | None
    removed_aliases: list[str]


def normalize_observation(
    observation: dict[str, Any],
    graph: dict[str, Any],
    phase: str,
    reconstruction: dict[str, Any] | None,
    label: str,
) -> NormalizedObservation:
    body = dict(observation)
    removed: list[str] = []
    runtime: dict[str, Any] | None = None

    if phase == "assembly":
        modules = body.pop("modules", None)
        if reconstruction is None:
            raise DatasetError("assembly reconstruction metadata is absent")
        validate_modules_reconstructible(modules, graph, reconstruction)
        removed.append(f"{label}.modules=graph_nodes+static_metadata")
    else:
        if "module_positions_world_m" in body:
            validate_module_positions(
                body.pop("module_positions_world_m"), graph, f"{label}.module_positions_world_m"
            )
            removed.append(f"{label}.module_positions_world_m=graph_node_positions")

    if phase == "manipulation":
        graph_globals = graph.get("global_attributes")
        if "global_attributes" in body:
            if body.pop("global_attributes") != graph_globals:
                raise DatasetError(f"{label}.global_attributes differs from graph")
            removed.append(f"{label}.global_attributes=graph.global_attributes")

        graph_course = graph_globals.get("course") if isinstance(graph_globals, dict) else None
        if "course" in body and graph_course is not None:
            if body.pop("course") != graph_course:
                raise DatasetError(f"{label}.course differs from graph course")
            removed.append(f"{label}.course=graph.global_attributes.course")

        environment = body.get("environment")
        if isinstance(environment, dict) and "course" in environment and graph_course is not None:
            environment = dict(environment)
            if environment.pop("course") != graph_course:
                raise DatasetError(f"{label}.environment.course differs from graph course")
            body["environment"] = environment
            removed.append(f"{label}.environment.course=graph.global_attributes.course")

        raw_runtime = body.pop("runtime_state", None)
        if raw_runtime is not None:
            if not isinstance(raw_runtime, dict):
                raise DatasetError(f"{label}.runtime_state is not an object")
            runtime = dict(raw_runtime)
            modules = runtime.pop("modules", None)
            if modules != graph_attribute_list(graph):
                raise DatasetError(f"{label}.runtime_state.modules differs from graph nodes")
            removed.append(f"{label}.runtime_state.modules=graph_node_attributes")
            if "course" in runtime and graph_course is not None:
                if runtime.pop("course") != graph_course:
                    raise DatasetError(f"{label}.runtime_state.course differs from graph course")
                removed.append(
                    f"{label}.runtime_state.course=graph.global_attributes.course"
                )

    return NormalizedObservation(body=body, runtime_state=runtime, removed_aliases=removed)


def semantic_equal(left: Any, right: Any) -> bool:
    """Compare two transitions without allocating a second graph-sized tree."""
    if isinstance(left, dict) and isinstance(right, dict):
        left_keys = {
            key
            for key in left
            if key not in RLE_IGNORED_KEYS and not key.startswith(SOURCE_RLE_PREFIX)
        }
        right_keys = {
            key
            for key in right
            if key not in RLE_IGNORED_KEYS and not key.startswith(SOURCE_RLE_PREFIX)
        }
        return left_keys == right_keys and all(
            semantic_equal(left[key], right[key]) for key in left_keys
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            semantic_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def action_has_payload(action: dict[str, Any]) -> bool:
    for key, value in action.items():
        if key == "locomotion" and isinstance(value, dict) and value:
            return True
        if key == "magnetic" and isinstance(value, list) and value:
            return True
        if key not in ("locomotion", "magnetic") and value not in (None, {}, []):
            return True
    return False


class RawAnalyzer:
    def __init__(self, phase: str, source: Path) -> None:
        self.phase = phase
        self.source = source
        self.record_count = 0
        self.first_record: dict[str, Any] | None = None
        self.last_record: dict[str, Any] | None = None
        self.top_tracker = MappingTracker()
        self.observation_tracker = MappingTracker()
        self.next_observation_tracker = MappingTracker()
        self.runtime_tracker = MappingTracker()
        self.next_runtime_tracker = MappingTracker()
        self.observation_kind: str | None = None
        self.next_observation_kind: str | None = None
        self.reconstruction: dict[str, Any] | None = None
        self.removed_aliases: set[str] = set()
        self.task_types: Counter[str] = Counter()
        self.stage_names: Counter[str] = Counter()
        self.node_counts: Counter[int] = Counter()
        self.edge_counts_t: Counter[int] = Counter()
        self.edge_counts_next: Counter[int] = Counter()
        self.action_valid = 0
        self.bc_valid = 0
        self.action_payload = 0
        self.implicit_hold = 0
        self.terminal = 0
        self.invalid_terminal = 0
        self.transition_stamp_regressions = 0
        self.excluded_transition_indices: list[int] = []
        self.segment_rows: list[dict[str, Any]] = []
        self._active_segment: dict[str, Any] | None = None

    def _observe_segments(self, index: int, record: dict[str, Any]) -> None:
        key = (record.get("stage_name"), record.get("task_type"))
        if self._active_segment is None or self._active_segment["key"] != key:
            if self._active_segment is not None:
                self._active_segment["end_record_index"] = index - 1
                self.segment_rows.append(self._active_segment)
            self._active_segment = {
                "key": key,
                "stage_name": key[0],
                "task_type": key[1],
                "start_record_index": index,
                "end_record_index": index,
            }

    def observe(self, record: Any) -> None:
        index = self.record_count
        if not isinstance(record, dict):
            raise DatasetError(f"{self.source}:{index + 1}: record is not an object")
        graph = validate_graph(record.get("graph_t"), "graph_t")
        graph_next = validate_graph(record.get("graph_t_plus_1"), "graph_t_plus_1")
        action = record.get("expert_action")
        if not isinstance(action, dict):
            raise DatasetError(f"{self.source}:{index + 1}: expert_action is absent")
        if "action_valid" not in record or "supervision" not in record:
            raise DatasetError(f"{self.source}:{index + 1}: supervision fields are absent")
        for alias, primary in EXACT_ALIAS_FIELDS.items():
            if alias not in record or primary not in record or record[alias] != record[primary]:
                raise DatasetError(
                    f"{self.source}:{index + 1}: {alias} is not an exact alias of {primary}"
                )
            self.removed_aliases.add(f"{alias}={primary}")

        if self.first_record is None:
            # Parsed records are never mutated by the analyzer, so retaining the
            # object avoids a graph-sized deepcopy on every source row.
            self.first_record = record
            observation = record.get("observation")
            if self.phase == "assembly":
                if not isinstance(observation, dict) or not isinstance(
                    observation.get("modules"), list
                ):
                    raise DatasetError("assembly observation.modules is absent")
                self.reconstruction = module_reconstruction_metadata(
                    observation["modules"], graph
                )
        self.last_record = record

        top = {
            key: record[key]
            for key in TOP_LEVEL_CONSTANT_CANDIDATES
            if key in record
        }
        self.top_tracker.observe(top)

        observation = record.get("observation")
        if not isinstance(observation, dict):
            raise DatasetError(f"{self.source}:{index + 1}: observation is not an object")
        normalized = normalize_observation(
            observation, graph, self.phase, self.reconstruction, "observation"
        )
        self.removed_aliases.update(normalized.removed_aliases)
        body = dict(normalized.body)
        body.pop("runtime_state", None)
        self.observation_tracker.observe(body)
        if normalized.runtime_state is not None:
            self.runtime_tracker.observe(normalized.runtime_state)

        next_observation = record.get("observation_t_plus_1")
        next_kind = "none" if next_observation is None else "object"
        if self.next_observation_kind is None:
            self.next_observation_kind = next_kind
        elif self.next_observation_kind != next_kind:
            raise DatasetError("observation_t_plus_1 mixes null and object records")
        if isinstance(next_observation, dict):
            normalized_next = normalize_observation(
                next_observation,
                graph_next,
                self.phase,
                self.reconstruction,
                "observation_t_plus_1",
            )
            self.removed_aliases.update(normalized_next.removed_aliases)
            body_next = dict(normalized_next.body)
            body_next.pop("runtime_state", None)
            self.next_observation_tracker.observe(body_next)
            if normalized_next.runtime_state is not None:
                self.next_runtime_tracker.observe(normalized_next.runtime_state)

        ids_t = set(graph_node_map(graph))
        ids_next = set(graph_node_map(graph_next))
        if ids_t != ids_next:
            raise DatasetError(f"{self.source}:{index + 1}: node ids change inside transition")
        self.node_counts[len(ids_t)] += 1
        self.edge_counts_t[len(graph["edges"])] += 1
        self.edge_counts_next[len(graph_next["edges"])] += 1
        stamp_t = graph.get("stamp")
        stamp_next = graph_next.get("stamp")
        if isinstance(stamp_t, (int, float)) and isinstance(stamp_next, (int, float)):
            if stamp_next < stamp_t:
                self.transition_stamp_regressions += 1
                self.excluded_transition_indices.append(index)

        action_is_valid = bool(record.get("action_valid"))
        supervision = record.get("supervision")
        bc_is_valid = isinstance(supervision, dict) and bool(
            supervision.get("valid_for_behavior_cloning")
        )
        has_payload = action_has_payload(action)
        is_terminal = bool(record.get("is_terminal") or record.get("done"))
        self.action_valid += int(action_is_valid)
        self.bc_valid += int(bc_is_valid)
        self.action_payload += int(has_payload)
        self.implicit_hold += int(action_is_valid and not has_payload)
        self.terminal += int(is_terminal)
        self.invalid_terminal += int(is_terminal and not action_is_valid)
        self.task_types[str(record.get("task_type"))] += 1
        self.stage_names[str(record.get("stage_name"))] += 1
        self._observe_segments(index, record)
        self.record_count += 1

    def finish(self) -> None:
        if self.record_count == 0:
            raise DatasetError(f"{self.source} is empty")
        if self._active_segment is not None:
            self._active_segment["end_record_index"] = self.record_count - 1
            self.segment_rows.append(self._active_segment)
            self._active_segment = None

    def summary(self) -> dict[str, Any]:
        return {
            "records": self.record_count,
            "required_transition_fields": [
                "graph_t",
                "expert_action",
                "graph_t_plus_1",
            ],
            "graph_transition_records": self.record_count,
            "action_valid_records": self.action_valid,
            "behavior_cloning_valid_records": self.bc_valid,
            "explicit_action_payload_records": self.action_payload,
            "implicit_hold_or_continue_records": self.implicit_hold,
            "terminal_records": self.terminal,
            "invalid_terminal_records": self.invalid_terminal,
            "il_eligible_records": self.record_count
            - len(self.excluded_transition_indices),
            "excluded_temporally_reversed_records": list(
                self.excluded_transition_indices
            ),
            "node_count_distribution": dict(sorted(self.node_counts.items())),
            "edge_count_t_distribution": dict(sorted(self.edge_counts_t.items())),
            "edge_count_t_plus_1_distribution": dict(
                sorted(self.edge_counts_next.items())
            ),
            "task_type_counts": dict(sorted(self.task_types.items())),
            "stage_name_counts": dict(sorted(self.stage_names.items())),
            "segments": [
                {key: value for key, value in row.items() if key != "key"}
                for row in self.segment_rows
            ],
        }


def analyze_and_copy(
    source: Path,
    destination: Path,
    phase: str,
    copy_mode: str,
) -> tuple[RawAnalyzer, str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".copying")
    temporary.unlink(missing_ok=True)
    analyzer = RawAnalyzer(phase, source)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        output_stream = temporary.open("wb") if copy_mode == "physical" else None
        with source.open("rb") as input_stream:
            for line_number, line in enumerate(input_stream, start=1):
                if not line.strip():
                    raise DatasetError(f"{source}:{line_number}: blank JSONL line")
                if not line.endswith(b"\n"):
                    raise DatasetError(f"{source}:{line_number}: unterminated JSONL line")
                digest.update(line)
                byte_count += len(line)
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DatasetError(f"{source}:{line_number}: {error}") from error
                analyzer.observe(record)
                if output_stream is not None:
                    output_stream.write(line)
        if output_stream is not None:
            output_stream.flush()
            os.fsync(output_stream.fileno())
            output_stream.close()
        analyzer.finish()
        if copy_mode == "hardlink":
            try:
                os.link(source, temporary)
            except OSError as error:
                print(
                    f"WARNING: hard link unavailable for {source} ({error}); using physical copy.",
                    file=sys.stderr,
                    flush=True,
                )
                shutil.copy2(source, temporary)
        temporary.replace(destination)
        shutil.copystat(source, destination)
    except Exception:
        if output_stream is not None and not output_stream.closed:
            output_stream.close()
        temporary.unlink(missing_ok=True)
        raise
    return analyzer, digest.hexdigest(), byte_count


def analyze_existing(path: Path, phase: str) -> tuple[RawAnalyzer, str, int]:
    analyzer = RawAnalyzer(phase, path)
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip() or not line.endswith(b"\n"):
                raise DatasetError(f"{path}:{line_number}: invalid JSONL framing")
            digest.update(line)
            byte_count += len(line)
            analyzer.observe(json.loads(line))
    analyzer.finish()
    return analyzer, digest.hexdigest(), byte_count


def strip_constant_keys(value: dict[str, Any], constants: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in constants}


def transform_record(record: dict[str, Any], analyzer: RawAnalyzer) -> dict[str, Any]:
    transformed = dict(record)
    for alias in EXACT_ALIAS_FIELDS:
        transformed.pop(alias, None)
    for key in analyzer.top_tracker.constants:
        transformed.pop(key, None)

    graph = transformed["graph_t"]
    observation = normalize_observation(
        transformed["observation"],
        graph,
        analyzer.phase,
        analyzer.reconstruction,
        "observation",
    )
    body = strip_constant_keys(observation.body, analyzer.observation_tracker.constants)
    if observation.runtime_state is not None:
        runtime = strip_constant_keys(
            observation.runtime_state, analyzer.runtime_tracker.constants
        )
        if runtime:
            body["runtime_state"] = runtime
    transformed["observation"] = body

    next_observation = transformed.get("observation_t_plus_1")
    if next_observation is None:
        transformed.pop("observation_t_plus_1", None)
    else:
        normalized_next = normalize_observation(
            next_observation,
            transformed["graph_t_plus_1"],
            analyzer.phase,
            analyzer.reconstruction,
            "observation_t_plus_1",
        )
        body_next = strip_constant_keys(
            normalized_next.body, analyzer.next_observation_tracker.constants
        )
        if normalized_next.runtime_state is not None:
            runtime_next = strip_constant_keys(
                normalized_next.runtime_state,
                analyzer.next_runtime_tracker.constants,
            )
            if runtime_next:
                body_next["runtime_state"] = runtime_next
        transformed["observation_t_plus_1"] = body_next
    return transformed


def compact_file(
    source: Path,
    destination: Path,
    analyzer: RawAnalyzer,
) -> tuple[int, int, str]:
    temporary = destination.with_name(destination.name + ".compacting")
    digest = hashlib.sha256()
    bytes_written = 0
    compact_records = 0
    current: dict[str, Any] | None = None

    def write_record(stream: Any, value: dict[str, Any]) -> None:
        nonlocal bytes_written, compact_records
        encoded = json_bytes(value)
        stream.write(encoded)
        digest.update(encoded)
        bytes_written += len(encoded)
        compact_records += 1

    try:
        with source.open("r", encoding="utf-8") as input_stream, temporary.open(
            "wb"
        ) as output_stream:
            for index, line in enumerate(input_stream):
                raw = json.loads(line)
                if index in analyzer.excluded_transition_indices:
                    continue
                transformed = transform_record(raw, analyzer)
                if current is None or not semantic_equal(transformed, current):
                    if current is not None:
                        write_record(output_stream, current)
                    current = transformed
                    current["_source_repeat_count"] = 1
                    current["_source_record_index_start"] = index
                    current["_source_record_index_end"] = index
                    current["_source_timestep_start"] = raw.get("timestep")
                    current["_source_timestep_end"] = raw.get("timestep")
                    current["_source_stamp_start"] = raw.get("stamp")
                    current["_source_stamp_end"] = raw.get("stamp")
                else:
                    current["_source_repeat_count"] += 1
                    current["_source_record_index_end"] = index
                    current["_source_timestep_end"] = raw.get("timestep")
                    current["_source_stamp_end"] = raw.get("stamp")
            if current is not None:
                write_record(output_stream, current)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return compact_records, bytes_written, digest.hexdigest()


def phase_manifest(
    analyzer: RawAnalyzer,
    phase: str,
    raw_relative_path: str,
    raw_sha: str,
    raw_bytes: int,
    compact_relative_path: str,
    compact_records: int,
    compact_bytes: int,
    compact_sha: str,
) -> dict[str, Any]:
    next_constants = analyzer.next_observation_tracker.constants
    manifest: dict[str, Any] = {
        "schema_version": "mssr.expert_compact_phase.v2",
        "phase": phase,
        "source": {
            "path": raw_relative_path,
            "records": analyzer.record_count,
            "bytes": raw_bytes,
            "sha256": raw_sha,
        },
        "compact": {
            "path": compact_relative_path,
            "records": compact_records,
            "expanded_records": analyzer.record_count
            - len(analyzer.excluded_transition_indices),
            "bytes": compact_bytes,
            "sha256": compact_sha,
        },
        "top_level_constants": analyzer.top_tracker.constants,
        "dynamic_top_level_keys": analyzer.top_tracker.dynamic_keys,
        "observation_constants": analyzer.observation_tracker.constants,
        "dynamic_observation_keys": analyzer.observation_tracker.dynamic_keys,
        "dynamic_next_observation_keys": analyzer.next_observation_tracker.dynamic_keys,
        "next_observation_constants": (
            None
            if analyzer.next_observation_kind == "none"
            or next_constants == analyzer.observation_tracker.constants
            else next_constants
        ),
        "runtime_state_constants": analyzer.runtime_tracker.constants or None,
        "dynamic_runtime_state_keys": analyzer.runtime_tracker.dynamic_keys,
        "next_runtime_state_constants": (
            analyzer.next_runtime_tracker.constants or None
        ),
        "dynamic_next_runtime_state_keys": analyzer.next_runtime_tracker.dynamic_keys,
        "assembly_module_reconstruction": analyzer.reconstruction,
        "validated_redundancies_removed": sorted(analyzer.removed_aliases),
        "transition_contract": analyzer.summary(),
        "excluded_source_records": [
            {
                "record_index": index,
                "reason": "graph_t_plus_1 stamp precedes graph_t stamp",
            }
            for index in analyzer.excluded_transition_indices
        ],
    }
    return manifest


def edge_topology(graph: dict[str, Any]) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for edge in graph.get("edges", []):
        a = edge.get("module_a_id") or edge.get("source") or edge.get("from")
        b = edge.get("module_b_id") or edge.get("target") or edge.get("to")
        if a is None or b is None:
            raise DatasetError("cannot identify graph edge endpoints")
        result.add(tuple(sorted((str(a), str(b)))))
    return result


def phase_boundary(
    previous: RawAnalyzer, following: RawAnalyzer
) -> dict[str, Any]:
    assert previous.last_record is not None and following.first_record is not None
    previous_graph = previous.last_record["graph_t_plus_1"]
    following_graph = following.first_record["graph_t"]
    previous_edges = edge_topology(previous_graph)
    following_edges = edge_topology(following_graph)
    added = sorted(following_edges - previous_edges)
    removed = sorted(previous_edges - following_edges)
    if not added and not removed:
        kind = "exact_topology"
    elif len(added) == 1 and not removed:
        kind = "single_expected_late_edge"
    else:
        kind = "topology_discontinuity"
    return {
        "kind": kind,
        "from_phase": previous.phase,
        "to_phase": following.phase,
        "from_edge_count": len(previous_edges),
        "to_edge_count": len(following_edges),
        "added_edges": [list(edge) for edge in added],
        "removed_edges": [list(edge) for edge in removed],
        "expert_action_imputed": False,
        "following_first_graph_used_for_boundary_provenance": True,
    }


def copy_small_file(source: Path, destination: Path) -> dict[str, Any]:
    digest = sha256_file(source)
    if destination.exists():
        if sha256_file(destination) != digest:
            raise DatasetError(f"refusing to overwrite different file {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".copying")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    return {"path": destination.name, "bytes": destination.stat().st_size, "sha256": digest}


def make_environment_geometry(
    candidate: Candidate, destination: Path, project_root: Path
) -> dict[str, Any] | None:
    if candidate.task != "rc_car":
        return None
    result = candidate.result
    layout = result.get("physical_track")
    route = result.get("route")
    if not isinstance(layout, dict) or not isinstance(route, dict):
        return None
    bounds = layout.get("platform_bounds_xy_m")
    thickness = float(route.get("platform_thickness_m", 0.02))
    if isinstance(bounds, list) and len(bounds) == 4:
        center_x = (float(bounds[0]) + float(bounds[1])) / 2.0
        center_y = (float(bounds[2]) + float(bounds[3])) / 2.0
        size_x = float(bounds[1]) - float(bounds[0])
        size_y = float(bounds[3]) - float(bounds[2])
    else:
        center_x, center_y = 1.1, 0.0
        size_x = float(route["platform_size_x_m"])
        size_y = float(route["platform_size_y_m"])
    payload = {
        "schema_version": "mssr.rc_car_environment_geometry.v2",
        "task": "rc_car",
        "seed": candidate.seed,
        "stage_frame_id": "world",
        "layout_frame_id": layout.get("frame_id", "map"),
        "platform": {
            "name": "RCPlanarPlatform",
            "semantic": "rc_car_planar_support_platform",
            "center_xyz_m": [center_x, center_y, -thickness / 2.0],
            "size_xyz_m": [size_x, size_y, thickness],
            "pitch_deg": 0.0,
        },
        "layout": layout,
        "provenance": {
            "kind": "captured_success_result",
            "source_result": relative(candidate.completion_path, project_root),
            "generator": layout.get("generator"),
        },
    }
    atomic_write_json(destination, payload)
    return {
        "path": destination.name,
        "schema_version": payload["schema_version"],
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def compact_episode(
    episode_dir: Path,
    compact_dir: Path,
    task: str,
    seed: int,
    analyzers: dict[str, RawAnalyzer],
    raw_metadata: dict[str, tuple[str, int]],
    raw_root: Path,
    compact_root: Path,
) -> dict[str, Any]:
    compact_dir.mkdir(parents=True, exist_ok=True)
    phases: dict[str, Any] = {}
    ordered = [phase for phase in PHASES_BY_TASK[task] if phase in analyzers]
    for phase in ordered:
        analyzer = analyzers[phase]
        raw_path = episode_dir / f"{phase}_dataset.jsonl"
        output_path = compact_dir / f"{phase}.jsonl"
        compact_records, compact_bytes, compact_sha = compact_file(
            raw_path, output_path, analyzer
        )
        raw_sha, raw_bytes = raw_metadata[phase]
        manifest = phase_manifest(
            analyzer,
            phase,
            relative(raw_path, raw_root),
            raw_sha,
            raw_bytes,
            relative(output_path, compact_root),
            compact_records,
            compact_bytes,
            compact_sha,
        )
        atomic_write_json(compact_dir / f"{phase}_manifest.json", manifest)
        phases[phase] = manifest

    boundaries: dict[str, Any] = {}
    for before, after in zip(ordered, ordered[1:]):
        boundaries[f"{before}_to_{after}"] = phase_boundary(
            analyzers[before], analyzers[after]
        )
    source_manifest = episode_dir / "manifest.json"
    episode_manifest = {
        "schema_version": "mssr.expert_compact_episode.v2",
        "task": task,
        "seed": seed,
        "phase_order": ordered,
        "phases": phases,
        "phase_boundaries": boundaries,
        "assembly_behavior_boundary": boundaries.get("assembly_to_behavior"),
        "source_episode_manifest": relative(source_manifest, raw_root),
        "source_episode_manifest_sha256": sha256_file(source_manifest),
        "il_contract": {
            "state_t": "graph_t",
            "expert_action": "expert_action",
            "state_t_plus_1": "graph_t_plus_1",
            "expert_action_imputation": False,
            "graph_node_and_edge_attributes_retained": True,
            "module_roles_and_assignments_retained_or_moved_to_phase_metadata_when_constant": True,
        },
    }
    atomic_write_json(compact_dir / "manifest.json", episode_manifest)
    return episode_manifest


def candidate_episode_manifest(
    candidate: Candidate,
    datasets: dict[str, Any],
    outcome: dict[str, Any],
    environment_geometry: dict[str, Any] | None,
    analyzers: dict[str, RawAnalyzer],
    project_root: Path,
) -> dict[str, Any]:
    result = candidate.result
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "task": candidate.task,
        "seed": candidate.seed,
        "canonical_success": True,
        "original_success": True,
        "relabel_reason": None,
        "source_run": relative(candidate.run_dir, project_root),
        "source_result": relative(candidate.completion_path, project_root),
        "source_gui_run": candidate.source_gui_run,
        "source_variant_label": candidate.variant_label,
        "source_episode_id": result.get("episode_id"),
        "source_schema_version": result.get("schema_version"),
        "curriculum_level": result.get("curriculum_level"),
        "curriculum_difficulty": result.get("curriculum_difficulty"),
        "datasets": datasets,
        "outcome": outcome,
        "phase_task_types": {
            phase: dict(sorted(analyzer.task_types.items()))
            for phase, analyzer in analyzers.items()
        },
        "omitted_reusable_phases": [
            phase for phase in PHASES_BY_TASK[candidate.task] if phase not in datasets
        ],
        "geometry": result.get("geometry"),
        "gap": result.get("gap"),
        "stair": result.get("stair"),
    }
    if candidate.task == "button":
        manifest["button_outcome"] = {
            key: result.get(key)
            for key in (
                "button_pressed",
                "button_xyz_m",
                "final_morphology",
                "final_root_xy_m",
                "final_root_yaw_rad",
                "retreat",
                "return_reconfiguration",
            )
        }
    if environment_geometry is not None:
        manifest["environment_geometry"] = environment_geometry
    return manifest


def backfill_existing_outcomes(raw_root: Path, project_root: Path) -> int:
    count = 0
    for manifest_path in sorted((raw_root / "episodes").glob("*/*/manifest.json")):
        manifest = json_load(manifest_path)
        if manifest.get("outcome"):
            continue
        source_text = manifest.get("source_result")
        if not isinstance(source_text, str):
            continue
        source = project_root / source_text
        if not source.is_file():
            continue
        outcome_path = manifest_path.parent / "outcome.json"
        outcome = copy_small_file(source, outcome_path)
        outcome["source_path"] = relative(source, project_root)
        outcome["path"] = relative(outcome_path, raw_root)
        manifest["outcome"] = outcome
        atomic_write_json(manifest_path, manifest)
        count += 1
    return count


def rebuild_raw_manifest(raw_root: Path, previous: dict[str, Any], audit_path: Path) -> dict[str, Any]:
    episodes = [
        json_load(path)
        for path in sorted((raw_root / "episodes").glob("*/*/manifest.json"))
    ]
    episodes.sort(key=lambda item: (str(item["task"]), int(item["seed"])))
    tasks: dict[str, Any] = {}
    totals: Counter[str] = Counter()
    for task in sorted({str(item["task"]) for item in episodes}):
        selected = [item for item in episodes if item["task"] == task]
        phase_totals: Counter[str] = Counter()
        for item in selected:
            for phase, metadata in item["datasets"].items():
                phase_totals[phase] += int(metadata["samples"])
                totals[f"{phase}_samples"] += int(metadata["samples"])
        task_data: dict[str, Any] = {
            "trajectory_count": len(selected),
            "seeds": [int(item["seed"]) for item in selected],
            "gui_source_seeds": [
                int(item["seed"]) for item in selected if item.get("source_gui_run")
            ],
            "relabeled_seeds": [
                int(item["seed"]) for item in selected if item.get("relabel_reason")
            ],
        }
        task_data.update({f"{phase}_samples": value for phase, value in phase_totals.items()})
        tasks[task] = task_data
    totals["trajectory_count"] = len(episodes)
    manifest = dict(previous)
    manifest.update(
        {
            "schema_version": 2,
            "dataset_name": "expert_v1",
            "description": "Canonical successful deterministic MSSR expert demonstrations for imitation learning.",
            "episodes": episodes,
            "selection_policy": (
                "One canonical successful run per task/seed. Required raw phase JSONL files "
                "are copied byte-for-byte. Stairs recovery episodes may omit the invariant "
                "Snake8 assembly phase; button episodes additionally require a successful "
                "physical press and return to RC-Car8."
            ),
            "source_inventory": relative(audit_path, raw_root.parent.parent),
            "tasks": tasks,
            "totals": dict(sorted(totals.items())),
            "raw_logs_modified": False,
        }
    )
    return manifest


def raw_readme(manifest: dict[str, Any]) -> str:
    totals = manifest["totals"]
    lines = [
        "MSSR Expert Dataset v1",
        "======================",
        "",
        "Canonical byte-for-byte raw expert transitions for imitation learning.",
        "Every sample retains graph_t, expert_action, and graph_t_plus_1.",
        "",
        f"Trajectories: {totals['trajectory_count']}",
    ]
    for phase in ("assembly", "behavior", "manipulation"):
        key = f"{phase}_samples"
        if key in totals:
            lines.append(f"{phase.capitalize()} samples: {totals[key]}")
    for task, data in manifest["tasks"].items():
        lines.extend(
            [
                "",
                f"{task}: {data['trajectory_count']} trajectories",
                f"  seeds: {data['seeds']}",
            ]
        )
        if data.get("gui_source_seeds"):
            lines.append(f"  GUI source seeds: {data['gui_source_seeds']}")
        if data.get("relabeled_seeds"):
            lines.append(f"  canonically relabeled: {data['relabeled_seeds']}")
        for phase in ("assembly", "behavior", "manipulation"):
            key = f"{phase}_samples"
            if key in data:
                lines.append(f"  {phase} samples: {data[key]}")
    return "\n".join(lines) + "\n"


def load_all_compact_episode_manifests(compact_root: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted((compact_root / "episodes").glob("*/*/manifest.json")):
        manifest = json_load(path)
        manifest["_path"] = path
        result.append(manifest)
    return result


def rebuild_compact_manifest(
    compact_root: Path, raw_root: Path, previous: dict[str, Any]
) -> dict[str, Any]:
    episode_manifests = load_all_compact_episode_manifests(compact_root)
    episodes: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    per_task: dict[str, Counter[str]] = {}
    per_phase: dict[str, Counter[str]] = {}
    boundary_counts: Counter[str] = Counter()
    for item in episode_manifests:
        path = item.pop("_path")
        phases = item["phases"]
        source_records = sum(int(value["source"]["records"]) for value in phases.values())
        expanded_records = sum(
            int(value["compact"].get("expanded_records", value["source"]["records"]))
            for value in phases.values()
        )
        compact_records = sum(int(value["compact"]["records"]) for value in phases.values())
        task = str(item.get("task") or path.parent.parent.name)
        seed = int(item.get("seed") or SEED_RE.search(path.parent.name).group(1))
        boundary = item.get("assembly_behavior_boundary") or {}
        boundary_kind = str(boundary.get("kind", "unknown"))
        # v1 used "exact" while v2 makes clear that exactness is topological.
        if boundary_kind == "exact":
            boundary_kind = "exact_topology"
        boundary_counts[boundary_kind] += 1
        episodes.append(
            {
                "task": task,
                "seed": seed,
                "path": relative(path.parent, compact_root),
                "phases": sorted(phases),
                "source_records": source_records,
                "expanded_records": expanded_records,
                "compact_records": compact_records,
                "boundary_kind": boundary_kind,
            }
        )
        totals["trajectory_count"] += 1
        totals["source_records"] += source_records
        totals["expanded_records"] += expanded_records
        totals["excluded_source_records"] += source_records - expanded_records
        totals["compact_records"] += compact_records
        task_counter = per_task.setdefault(task, Counter())
        task_counter["trajectory_count"] += 1
        for phase, value in phases.items():
            source = value["source"]
            compact = value["compact"]
            totals["source_jsonl_bytes"] += int(source["bytes"])
            totals["compact_jsonl_bytes"] += int(compact["bytes"])
            task_counter["source_records"] += int(source["records"])
            phase_expanded = int(compact.get("expanded_records", source["records"]))
            task_counter["expanded_records"] += phase_expanded
            task_counter["excluded_source_records"] += int(source["records"]) - phase_expanded
            task_counter["compact_records"] += int(compact["records"])
            task_counter["compact_bytes"] += int(compact["bytes"])
            phase_counter = per_phase.setdefault(phase, Counter())
            phase_counter["trajectory_count"] += 1
            phase_counter["source_records"] += int(source["records"])
            phase_counter["expanded_records"] += phase_expanded
            phase_counter["excluded_source_records"] += int(source["records"]) - phase_expanded
            phase_counter["compact_records"] += int(compact["records"])
            phase_counter["source_bytes"] += int(source["bytes"])
            phase_counter["compact_bytes"] += int(compact["bytes"])
    totals["collapsed_source_records"] = totals["expanded_records"] - totals["compact_records"]
    for kind, count in boundary_counts.items():
        totals[f"boundary_{kind}"] = count
    totals_json: dict[str, Any] = dict(sorted(totals.items()))
    totals_json["per_task"] = {
        key: dict(sorted(value.items())) for key, value in sorted(per_task.items())
    }
    totals_json["per_phase"] = {
        key: dict(sorted(value.items())) for key, value in sorted(per_phase.items())
    }
    policy = dict(previous.get("compaction_policy", {}))
    policy.update(
        {
            "graph_is_primary_robot_state": True,
            "graph_features_retained": True,
            "exact_aliases_removed": list(EXACT_ALIAS_FIELDS),
            "phase_constants": (
                "Moved to phase metadata only after exact stream-wide validation; dynamic "
                "button subtask labels, assignments, target graphs, and roles stay in records."
            ),
            "rle": {
                "consecutive_only": True,
                "ignored_fields": sorted(RLE_IGNORED_KEYS),
                "is_first_preserved": True,
                "terminal_flags_preserved": True,
                "repeat_count_preserved": True,
            },
            "transition_contract": {
                "state_t": "graph_t",
                "expert_action": "expert_action",
                "state_t_plus_1": "graph_t_plus_1",
                "expert_action_imputation": False,
                "behavior_cloning_filter": (
                    "action_valid == true and "
                    "supervision.valid_for_behavior_cloning == true"
                ),
                "terminal_markers": (
                    "retained for episode boundaries; terminal rows without a valid action "
                    "are not behavior-cloning targets"
                ),
                "temporally_reversed_transitions": (
                    "retained in authoritative raw data, excluded from compact IL rows, and indexed in phase manifests"
                ),
            },
            "manipulation_observation": {
                "module_positions_world_m": "removed after exact graph-position validation",
                "global_attributes": "removed after exact graph-global validation",
                "runtime_state.modules": "removed after exact graph-node validation",
                "duplicated_course_views": "removed after exact graph-course validation",
                "operational_dofs": "retained",
                "runtime_contacts_and_attachments": "retained",
            },
        }
    )
    episodes.sort(key=lambda item: (item["task"], item["seed"]))
    return {
        "schema_version": "mssr.expert_compact_dataset.v2",
        "source_dataset": relative(raw_root, compact_root.parent.parent),
        "source_manifest_sha256": sha256_file(raw_root / "manifest.json"),
        "compaction_policy": policy,
        "episodes": episodes,
        "totals": totals_json,
    }


def compact_readme(manifest: dict[str, Any]) -> str:
    totals = manifest["totals"]
    return f"""MSSR expert_v1_compact
========================

Derived from datasets/expert_v1; the raw canonical dataset remains authoritative.

IL transition contract
----------------------
Every compact row retains graph_t, expert_action, and graph_t_plus_1.  Node and
edge attributes, graph_features, supervision, rewards, terminal flags, active
primitive/FSM state, task graph, operational DOFs, assignments, and module roles
remain either in the row or (only when exactly constant) in the phase manifest.
For behavior cloning, select rows where action_valid is true and
supervision.valid_for_behavior_cloning is true. Terminal rows without a valid
action remain available as episode-boundary metadata, not as action targets.

Validated redundancy removal
----------------------------
- attributed_graph and attributed_task_graph are exact aliases.
- assembly observation.modules is reconstructed from graph nodes plus static
  per-module metadata in assembly_manifest.json.
- behavior/manipulation module_positions_world_m is reconstructed from graph
  node positions.
- manipulation runtime_state.modules/global_attributes/course aliases are
  removed only after exact per-record validation. Contacts and attachments stay.
- constants are moved to metadata only after exact stream-wide comparison.

RLE
---
Only consecutive semantically identical transitions are collapsed. stage_id,
stamp, and timestep are bookkeeping for equality; source index/timestep/stamp
spans and _source_repeat_count preserve temporal provenance. No action is ever
invented at a phase boundary. Source rows whose G_t+1 stamp precedes the G_t
stamp are preserved in raw expert_v1 but explicitly excluded from compact IL.

Current totals
--------------
Trajectories: {totals['trajectory_count']}
Source records: {totals['source_records']}
Compact records: {totals['compact_records']}
Temporally invalid source records excluded: {totals.get('excluded_source_records', 0)}
"""


def duplicate_content_report(raw_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for episode in raw_manifest["episodes"]:
        for phase, item in episode["datasets"].items():
            by_hash.setdefault(str(item["sha256"]), []).append(
                {"task": episode["task"], "seed": episode["seed"], "phase": phase}
            )
    return [
        {"sha256": digest, "occurrences": occurrences}
        for digest, occurrences in sorted(by_hash.items())
        if len(occurrences) > 1
    ]


def pending_run_report(log_root: Path, candidates: list[Candidate], project_root: Path) -> list[dict[str, Any]]:
    known_dirs = {candidate.run_dir.resolve() for candidate in candidates}
    pending: list[dict[str, Any]] = []
    for pattern in ("*stair*", "*campaign*"):
        for path in sorted(log_root.glob(pattern)):
            if not path.is_dir() or path.resolve() in known_dirs:
                continue
            has_marker = any((path / name).exists() for name in ("pid", "campaign.pid", "overnight.log"))
            if has_marker and not any(path.rglob("result.json")):
                pending.append(
                    {
                        "path": relative(path, project_root),
                        "reason": "run marker exists but no completed episode result is available",
                    }
                )
    unique = {item["path"]: item for item in pending}
    return [unique[key] for key in sorted(unique)]


def import_candidate(
    candidate: Candidate,
    project_root: Path,
    raw_root: Path,
    compact_root: Path,
    copy_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Import one episode; safe to run in a separate process."""
    episode_name = f"seed-{candidate.seed:06d}"
    episode_dir = raw_root / "episodes" / candidate.task / episode_name
    compact_dir = compact_root / "episodes" / candidate.task / episode_name
    if (episode_dir / "manifest.json").exists():
        raise DatasetError(f"unexpected existing episode {episode_dir}")
    analyzers: dict[str, RawAnalyzer] = {}
    raw_metadata: dict[str, tuple[str, int]] = {}
    datasets: dict[str, Any] = {}
    validation_summaries: list[dict[str, Any]] = []
    for phase in candidate.phases:
        source = candidate.run_dir / f"{phase}_dataset.jsonl"
        destination = episode_dir / f"{phase}_dataset.jsonl"
        print(
            f"[{candidate.task}:{candidate.seed}] validating/copying {phase}...",
            flush=True,
        )
        analyzer, digest, byte_count = analyze_and_copy(
            source, destination, phase, copy_mode
        )
        analyzers[phase] = analyzer
        raw_metadata[phase] = (digest, byte_count)
        datasets[phase] = {
            "path": relative(destination, raw_root),
            "source_path": relative(source, project_root),
            "samples": analyzer.record_count,
            "bytes": byte_count,
            "sha256": digest,
        }
        validation_summaries.append(
            {
                "task": candidate.task,
                "seed": candidate.seed,
                "phase": phase,
                **analyzer.summary(),
                "validated_redundancies_removed": sorted(analyzer.removed_aliases),
                "constant_top_level_keys": sorted(analyzer.top_tracker.constants),
                "dynamic_top_level_keys": analyzer.top_tracker.dynamic_keys,
            }
        )

    if candidate.completion_path is None:
        raise DatasetError(f"selected candidate {candidate.run_dir} has no completion artifact")
    outcome_path = episode_dir / "outcome.json"
    outcome = copy_small_file(candidate.completion_path, outcome_path)
    outcome["source_path"] = relative(candidate.completion_path, project_root)
    outcome["path"] = relative(outcome_path, raw_root)
    environment_geometry = make_environment_geometry(
        candidate, episode_dir / "environment_geometry.json", project_root
    )
    if environment_geometry is not None:
        environment_geometry["path"] = relative(
            episode_dir / "environment_geometry.json", raw_root
        )
    episode_manifest = candidate_episode_manifest(
        candidate,
        datasets,
        outcome,
        environment_geometry,
        analyzers,
        project_root,
    )
    atomic_write_json(episode_dir / "manifest.json", episode_manifest)
    print(f"[{candidate.task}:{candidate.seed}] compacting...", flush=True)
    compact_episode(
        episode_dir,
        compact_dir,
        candidate.task,
        candidate.seed,
        analyzers,
        raw_metadata,
        raw_root,
        compact_root,
    )
    imported = {
        "task": candidate.task,
        "seed": candidate.seed,
        "source_run": relative(candidate.run_dir, project_root),
        "raw_episode": relative(episode_dir, project_root),
        "compact_episode": relative(compact_dir, project_root),
    }
    print(f"[{candidate.task}:{candidate.seed}] complete.", flush=True)
    return imported, validation_summaries


def run(args: argparse.Namespace) -> int:
    project_root = args.project_root.resolve()
    log_root = (project_root / args.logs).resolve()
    raw_root = (project_root / args.raw).resolve()
    compact_root = (project_root / args.compact).resolve()
    raw_manifest_path = raw_root / "manifest.json"
    compact_manifest_path = compact_root / "manifest.json"
    if not raw_manifest_path.is_file() or not compact_manifest_path.is_file():
        raise DatasetError("expert_v1 and expert_v1_compact manifests must already exist")
    previous_raw = json_load(raw_manifest_path)
    previous_compact = json_load(compact_manifest_path)
    existing_sources: dict[tuple[str, int], Path | None] = {}
    for item in previous_raw["episodes"]:
        key = (str(item["task"]), int(item["seed"]))
        source_run = item.get("source_run")
        existing_sources[key] = (
            (project_root / source_run).resolve()
            if isinstance(source_run, str)
            else None
        )
    # A previous interrupted invocation may have finished an episode before it
    # rebuilt the dataset-level manifest. Treat complete on-disk episode
    # manifests as canonical too; partial directories have no manifest and are
    # safely resumed by overwriting only their temporary files.
    for path in (raw_root / "episodes").glob("*/*/manifest.json"):
        item = json_load(path)
        key = (str(item["task"]), int(item["seed"]))
        source_run = item.get("source_run")
        existing_sources[key] = (
            (project_root / source_run).resolve()
            if isinstance(source_run, str)
            else existing_sources.get(key)
        )

    candidates = discover_candidates(log_root)
    selected = classify_candidates(candidates, existing_sources)
    print(
        f"Discovered {len(candidates)} candidates; importing {len(selected)} new task/seed episodes.",
        flush=True,
    )
    imported: list[dict[str, Any]] = []
    validation_summaries: list[dict[str, Any]] = []
    worker_count = max(1, min(int(args.workers), len(selected) or 1))
    if worker_count == 1:
        for candidate in selected:
            result, summaries = import_candidate(
                candidate, project_root, raw_root, compact_root, args.copy_mode
            )
            candidate.classification = "imported"
            candidate.reason = "validated raw phases copied and compacted"
            imported.append(result)
            validation_summaries.extend(summaries)
    else:
        print(f"Using {worker_count} independent episode workers.", flush=True)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    import_candidate,
                    candidate,
                    project_root,
                    raw_root,
                    compact_root,
                    args.copy_mode,
                ): candidate
                for candidate in selected
            }
            for future in as_completed(futures):
                candidate = futures[future]
                result, summaries = future.result()
                candidate.classification = "imported"
                candidate.reason = "validated raw phases copied and compacted"
                imported.append(result)
                validation_summaries.extend(summaries)

    backfilled = backfill_existing_outcomes(raw_root, project_root)
    audit_path = raw_root / "import_audit.json"
    raw_manifest = rebuild_raw_manifest(raw_root, previous_raw, audit_path)
    atomic_write_json(raw_manifest_path, raw_manifest)
    atomic_write_text(raw_root / "README.txt", raw_readme(raw_manifest))

    compact_manifest = rebuild_compact_manifest(compact_root, raw_root, previous_compact)
    atomic_write_json(compact_manifest_path, compact_manifest)
    atomic_write_text(compact_root / "README.txt", compact_readme(compact_manifest))

    same_seed_groups = []
    by_key: dict[tuple[str, int], list[Candidate]] = {}
    for candidate in candidates:
        by_key.setdefault(candidate.key, []).append(candidate)
    for (task, seed), group in sorted(by_key.items()):
        if len(group) > 1:
            same_seed_groups.append(
                {
                    "task": task,
                    "seed": seed,
                    "candidates": [
                        relative(item.run_dir, project_root)
                        for item in sorted(group, key=lambda value: value.completion_mtime_ns)
                    ],
                    "classifications": [
                        item.classification
                        for item in sorted(group, key=lambda value: value.completion_mtime_ns)
                    ],
                }
            )
    canonical_v2: list[dict[str, Any]] = []
    canonical_validation: list[dict[str, Any]] = []
    for episode in raw_manifest["episodes"]:
        if int(episode.get("schema_version", 1)) < 2:
            continue
        task = str(episode["task"])
        seed = int(episode["seed"])
        canonical_v2.append(
            {
                "task": task,
                "seed": seed,
                "source_run": episode.get("source_run"),
                "raw_episode": f"datasets/expert_v1/episodes/{task}/seed-{seed:06d}",
                "compact_episode": f"datasets/expert_v1_compact/episodes/{task}/seed-{seed:06d}",
            }
        )
        compact_episode_manifest = (
            compact_root / "episodes" / task / f"seed-{seed:06d}" / "manifest.json"
        )
        if compact_episode_manifest.is_file():
            compact_episode_data = json_load(compact_episode_manifest)
            for phase, phase_data in compact_episode_data.get("phases", {}).items():
                contract = phase_data.get("transition_contract")
                if isinstance(contract, dict):
                    contract = dict(contract)
                    contract.setdefault(
                        "il_eligible_records",
                        int(
                            phase_data["compact"].get(
                                "expanded_records", phase_data["source"]["records"]
                            )
                        ),
                    )
                    contract.setdefault(
                        "excluded_temporally_reversed_records",
                        [
                            int(item["record_index"])
                            for item in phase_data.get("excluded_source_records", [])
                        ],
                    )
                    canonical_validation.append(
                        {"task": task, "seed": seed, "phase": phase, **contract}
                    )
    validation_totals = {
        key: sum(int(item.get(key, 0)) for item in canonical_validation)
        for key in (
            "records",
            "il_eligible_records",
            "behavior_cloning_valid_records",
            "action_valid_records",
            "explicit_action_payload_records",
            "implicit_hold_or_continue_records",
            "terminal_records",
            "invalid_terminal_records",
        )
    }
    validation_totals["temporally_reversed_records_excluded"] = sum(
        len(item.get("excluded_temporally_reversed_records", []))
        for item in canonical_validation
    )
    audit = {
        "schema_version": "mssr.expert_dataset_import_audit.v1",
        "generated_at_utc": utc_now(),
        "selection_key": ["task", "seed"],
        "selection_rule": "latest completed successful stable run; existing canonical task/seed is immutable",
        "candidate_count": len(candidates),
        "imported_count": len(canonical_v2),
        "imported": canonical_v2,
        "newly_imported_this_invocation_count": len(imported),
        "newly_imported_this_invocation": imported,
        "existing_outcomes_backfilled": backfilled,
        "candidates": [
            item.as_audit(project_root)
            for item in sorted(
                candidates,
                key=lambda value: (value.task, value.seed, value.completion_mtime_ns),
            )
        ],
        "same_seed_groups": same_seed_groups,
        "exact_duplicate_canonical_payloads": duplicate_content_report(raw_manifest),
        "validation": canonical_validation or validation_summaries,
        "validation_totals": validation_totals,
        "pending_runs": pending_run_report(log_root, candidates, project_root),
        "metadata_findings": [
            {
                "finding": "aliases",
                "resolution": "attributed_graph and attributed_task_graph removed only after exact per-record equality",
            },
            {
                "finding": "button behavior contains multiple task_type segments",
                "resolution": "dynamic task labels/graphs/roles remain in compact records instead of being treated as constants",
            },
            {
                "finding": "GUI and headless labels share result schemas",
                "resolution": "source_gui_run and source_variant_label preserve provenance; task and seed are canonical keys",
            },
            {
                "finding": "valid expert rows may carry an empty immediate action payload",
                "resolution": "these are retained as hold/continue transitions together with active primitive and FSM context",
            },
        ],
        "raw_manifest_sha256": sha256_file(raw_manifest_path),
        "compact_manifest_sha256": sha256_file(compact_manifest_path),
    }
    atomic_write_json(audit_path, audit)
    # The manifest points at this audit but does not hash it, so writing it last is intentional.
    print(
        f"Done: {raw_manifest['totals']['trajectory_count']} raw trajectories, "
        f"{compact_manifest['totals']['compact_records']} compact records. "
        f"Audit: {relative(audit_path, project_root)}",
        flush=True,
    )
    return 0


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--logs", type=Path, default=Path("logs"))
    parser.add_argument("--raw", type=Path, default=Path("datasets/expert_v1"))
    parser.add_argument(
        "--compact", type=Path, default=Path("datasets/expert_v1_compact")
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="episode workers (default: 1; raise only on storage suited to parallel I/O)",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("hardlink", "physical"),
        default="hardlink",
        help="raw payload copy mode (default: hardlink, avoiding duplicate disk blocks)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except DatasetError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
