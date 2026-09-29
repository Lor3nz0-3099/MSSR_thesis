#!/usr/bin/env python3
"""Reversible semantic compaction of individual teleoperation transitions.

The format keeps one output row per input row.  It removes only values that
are either equal to another value in that row or reconstructible from stream
metadata and a per-row difference.  No time or action fields are synthesized.
"""

from __future__ import annotations

import copy
from typing import Any

TOP_CONSTANT_CANDIDATES = (
    "schema_version", "episode_id", "stage_name", "task_type", "difficulty",
    "target_graph", "assignment_target_to_module", "module_roles",
)
GRAPH_ALIASES = (
    ("attributed_graph", "graph_t"),
    ("attributed_task_graph", "task_graph_t"),
    ("task_graph_t", "graph_t"),
)
ANNOTATION_FIELDS = (
    "active_primitive", "debug", "fsm_state", "primitive_params", "task_metrics",
)
PRIVATE_KEY = "__teleop_compact__"


def _get_path(value: dict, path: list[str]) -> tuple[bool, Any]:
    current: Any = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False, None
        current = current[key]
    return True, current


def _delete_path(value: dict, path: list[str]) -> None:
    current = value
    for key in path[:-1]:
        current = current[key]
    del current[path[-1]]


def _set_path(value: dict, path: list[str], item: Any) -> None:
    current = value
    for key in path[:-1]:
        current = current.setdefault(key, {})
    current[path[-1]] = copy.deepcopy(item)


class TeleopCodec:
    def __init__(self, baseline: dict, field_baselines: list[dict],
                 patch_paths: list[list] | None = None) -> None:
        self.baseline = copy.deepcopy(baseline)
        self.field_baselines = copy.deepcopy(field_baselines)
        self.patch_paths = copy.deepcopy(patch_paths) if patch_paths is not None else []
        self.path_ids = {tuple(path): index for index, path in enumerate(self.patch_paths)}

    @classmethod
    def from_first_record(cls, row: dict) -> "TeleopCodec":
        if PRIVATE_KEY in row:
            raise ValueError("source row already contains codec control key")
        baseline = row["graph_t"]["global_attributes"]
        candidates = [[key] for key in TOP_CONSTANT_CANDIDATES if key in row]
        for observation_key in ("observation", "observation_t_plus_1"):
            observation = row.get(observation_key)
            if isinstance(observation, dict):
                candidates.extend([observation_key, key] for key in observation)
        field_baselines = [{"path": path, "value": copy.deepcopy(_get_path(row, path)[1])}
                           for path in candidates]
        return cls(baseline, field_baselines)

    @classmethod
    def from_metadata(cls, metadata: dict) -> "TeleopCodec":
        if metadata["schema_version"] != "mssr.teleop_semantic_codec.v1":
            raise ValueError("unsupported teleop codec")
        return cls(metadata["global_attributes_baseline"],
                   metadata["field_baselines"], metadata["patch_paths"])

    def metadata(self) -> dict:
        return {
            "schema_version": "mssr.teleop_semantic_codec.v1",
            "global_attributes_baseline": self.baseline,
            "field_baselines": self.field_baselines,
            "patch_paths": self.patch_paths,
            "graph_aliases": [list(pair) for pair in GRAPH_ALIASES],
            "annotation_fields": list(ANNOTATION_FIELDS),
            "control_key": PRIVATE_KEY,
        }

    def _path_id(self, path: tuple) -> int:
        if path not in self.path_ids:
            self.path_ids[path] = len(self.patch_paths)
            self.patch_paths.append(list(path))
        return self.path_ids[path]

    def _diff(self, base: Any, value: Any, path: tuple = ()) -> list[list]:
        if base == value:
            return []
        if isinstance(base, dict) and isinstance(value, dict):
            patches = []
            for key in sorted(base.keys() | value.keys()):
                child = path + (key,)
                if key not in value:
                    patches.append([self._path_id(child), 0])
                elif key not in base:
                    patches.append([self._path_id(child), 1, value[key]])
                else:
                    patches.extend(self._diff(base[key], value[key], child))
            return patches
        if isinstance(base, list) and isinstance(value, list) and len(base) == len(value):
            patches = []
            for index, (before, after) in enumerate(zip(base, value)):
                patches.extend(self._diff(before, after, path + (index,)))
            return patches
        return [[self._path_id(path), 1, value]]

    def _apply(self, patches: list[list]) -> dict:
        value = copy.deepcopy(self.baseline)
        for patch in patches:
            path = self.patch_paths[patch[0]]
            if not path:
                if patch[1] != 1:
                    raise ValueError("cannot delete global attributes root")
                value = copy.deepcopy(patch[2])
                continue
            current = value
            for component in path[:-1]:
                current = current[component]
            if patch[1] == 0:
                del current[path[-1]]
            elif patch[1] == 1:
                current[path[-1]] = copy.deepcopy(patch[2])
            else:
                raise ValueError("unknown patch operation")
        return value

    def encode(self, row: dict) -> dict:
        if PRIVATE_KEY in row:
            raise ValueError("source row already contains codec control key")
        output = dict(row)
        patches = []
        for key in ("graph_t", "graph_t_plus_1"):
            graph = dict(row[key])
            globals_value = graph.pop("global_attributes")
            patches.append(self._diff(self.baseline, globals_value))
            output[key] = graph
        for key in ("observation", "observation_t_plus_1"):
            if isinstance(row.get(key), dict):
                output[key] = dict(row[key])

        mask = 0
        for index, field in enumerate(self.field_baselines):
            path = field["path"]
            present, value = _get_path(output, path)
            if present and value == field["value"]:
                _delete_path(output, path)
                mask |= 1 << index
        offset = len(self.field_baselines)
        for index, (alias, primary) in enumerate(GRAPH_ALIASES):
            if alias in output and primary in output and output[alias] == row[primary]:
                del output[alias]
                mask |= 1 << (offset + index)
        if all(key in row for key in ANNOTATION_FIELDS) and row.get("expert_annotation") == {
            key: row[key] for key in ANNOTATION_FIELDS
        }:
            del output["expert_annotation"]
            mask |= 1 << (offset + len(GRAPH_ALIASES))
        if "stage_id" in output and "timestep" in output and output["stage_id"] == output["timestep"]:
            del output["stage_id"]
            mask |= 1 << (offset + len(GRAPH_ALIASES) + 1)
        output[PRIVATE_KEY] = {"mask": mask, "global_patches": patches}
        return output

    def decode(self, compact: dict) -> dict:
        output = dict(compact)
        control = output.pop(PRIVATE_KEY)
        patches = control["global_patches"]
        if len(patches) != 2:
            raise ValueError("expected graph_t and graph_t_plus_1 patches")
        for key, graph_patches in zip(("graph_t", "graph_t_plus_1"), patches):
            graph = dict(output[key])
            graph["global_attributes"] = self._apply(graph_patches)
            output[key] = graph
        for key in ("observation", "observation_t_plus_1"):
            if isinstance(output.get(key), dict):
                output[key] = dict(output[key])
        mask = control["mask"]
        for index, field in enumerate(self.field_baselines):
            if mask & (1 << index):
                _set_path(output, field["path"], field["value"])
        offset = len(self.field_baselines)
        if mask & (1 << (offset + 2)):
            output["task_graph_t"] = copy.deepcopy(output["graph_t"])
        for index, (alias, primary) in enumerate(GRAPH_ALIASES[:2]):
            if mask & (1 << (offset + index)):
                output[alias] = copy.deepcopy(output[primary])
        if mask & (1 << (offset + len(GRAPH_ALIASES))):
            output["expert_annotation"] = {
                key: copy.deepcopy(output[key]) for key in ANNOTATION_FIELDS
            }
        if mask & (1 << (offset + len(GRAPH_ALIASES) + 1)):
            output["stage_id"] = output["timestep"]
        return output



TEMPORAL_KEYS = frozenset(("stage_id", "stamp", "timestep"))


def semantically_equal(left: Any, right: Any) -> bool:
    """Use the single-task repeat rule, ignoring only temporal fields."""
    if isinstance(left, dict) and isinstance(right, dict):
        left_keys = {key for key in left if key not in TEMPORAL_KEYS}
        right_keys = {key for key in right if key not in TEMPORAL_KEYS}
        return left_keys == right_keys and all(
            semantically_equal(left[key], right[key]) for key in left_keys
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            semantically_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def temporal_fields(value: Any) -> tuple[list[list], list[Any]]:
    """Return paths and values required to restore one repeated transition."""
    paths: list[list] = []
    values: list[Any] = []

    def visit(item: Any, path: list) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key in TEMPORAL_KEYS:
                    paths.append(path + [key])
                    values.append(child)
                else:
                    visit(child, path + [key])
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, path + [index])

    visit(value, [])
    return paths, values


def with_temporal_values(value: dict, paths: list[list], values: list[Any]) -> dict:
    if len(paths) != len(values):
        raise ValueError("repeat temporal path/value mismatch")
    restored = copy.deepcopy(value)
    for path, item in zip(paths, values):
        current = restored
        for component in path[:-1]:
            current = current[component]
        current[path[-1]] = item
    return restored


def iter_decoded_archive(archive, file_meta):
    """Yield original-schema JSON records from one semantic .jsonl.zst stream."""
    import json
    import subprocess

    codec = TeleopCodec.from_metadata(file_meta["codec"])
    process = subprocess.Popen(["zstd", "-q", "-dc", str(archive)],
                               stdout=subprocess.PIPE)
    assert process.stdout is not None
    try:
        for line in process.stdout:
            compact = json.loads(line)
            repeat = compact[PRIVATE_KEY].get("repeat")
            yield codec.decode(compact)
            if repeat is not None:
                paths = repeat["paths"]
                for values in repeat["rows"]:
                    yield codec.decode(with_temporal_values(compact, paths, values))
        if process.wait() != 0:
            raise ValueError(f"cannot decode teleop archive: {archive}")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            process.wait()
