"""D1.2 task boundary: assigned problem, static geometry and dynamic values.

None is the A1 representation of UNKNOWN. Spatial NumericFeature names carry
their source frame; missing/unsupported frames retain values with valid=False.
Configuration destinations never enter the model context.
Source paths, line indices, file hashes and missing-source diagnostics live only
in read_task_metadata's audit mapping. Historical task reconstruction is explicit
and versioned in config; there is no next-record fill or coordinate transformation,
robot morphology inference, feature tensorization, or label access here.
"""

from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

from .bc_v1_contract import NumericFeature, ObservationBundle, TaskContext


_TASKS = ["UNKNOWN", "navigation", "gap", "stairs", "button"]
# Parsed metadata is closed to task/environment fields. Hardware, planner,
# action, result, runtime and next-state subtrees never cross this boundary.
_SAFE_KEYS = frozenset({
    "task_type", "task_id", "type", "stage_name", "stage_id", "stage", "phase",
    "current_task_id", "active_task_id", "current_obstacle_id", "task_context",
    "course", "environment", "environment_geometry", "task_parameters", "parameters",
    "mission", "tasks", "frame_id", "coordinate_frame", "geometry_frame", "layout_frame_id",
    "stage_frame", "origin_world_xyz_m", "yaw_rad", "physical_track", "layout", "navigation",
    "current_goal", "position_xyz_m", "xyyaw", "relative_geometry", "gap", "stairs", "button", "scenario",
    "collision_boxes", "center_xyz_m", "size_xyz_m", "yaw_deg", "pitch_deg",
    "width_m", "bank_width_m", "bank_thickness_m", "landing_length_m", "near_edge_x_m",
    "far_edge_x_m", "near_edge_world_xyz_m", "far_edge_world_xyz_m", "riser_depth_m",
    "near_edge_center_world_xyz_m", "far_edge_center_world_xyz_m", "crossing_direction_world_xy",
    "rise_m", "tread_depth_m", "step_count", "first_riser_x_m", "base_height_m",
    "top_heights_m", "first_riser_center_world_xyz_m", "ascent_direction_world_xy",
    "top_heights_world_m", "face_size_tangent_z_m", "plunger_depth_m", "plunger_stroke_m",
    "current_center_xyz_m", "press_direction_world_xy", "press_direction_world_xyz",
    "press_yaw_rad", "depression_m", "cone_centers_xy_m", "cone_radius_m", "cone_height_m",
    "corridor_width_m", "clear_landing_length_m", "floor_height_m", "upper_deck_height_m",
    "platform_bounds_xy_m", "value", "values", "valid",
    "navigation_cones", "radius_m", "height_m", "name", "semantic", "exit_pose_xyyaw",
})


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _clean(value):
    if isinstance(value, Mapping):
        return {k: _clean(v) for k, v in value.items() if k in _SAFE_KEYS}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return copy.deepcopy(value)


def _current(document):
    result = {k: _clean(document[k]) for k in
              ("task_type", "stage_name", "stage_id", "stage", "task_id",
               "current_task_id", "active_task_id", "current_obstacle_id") if k in document}
    for observation_key in ("observation", "observation_t"):
        observation = document.get(observation_key)
        if isinstance(observation, Mapping):
            # Keep sources distinct: duplicates must agree rather than overwrite.
            result[observation_key] = _clean(observation)
    graph = document.get("graph_t")
    globals_ = graph.get("global_attributes") if isinstance(graph, Mapping) else None
    if isinstance(globals_, Mapping) and isinstance(globals_.get("course"), Mapping):
        result["course"] = _clean(globals_["course"])
    return result


def _path(source, root):
    if not isinstance(source, str) or not source:
        raise ValueError("Missing source metadata path")
    path = Path(source)
    path = (root / path if not path.is_absolute() else path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Source metadata path escapes repo_root")
    return path


def _join_current_constants(constants, current, field="observation"):
    """Restore compact CURRENT scope; differing declared constants are errors."""
    if isinstance(constants, Mapping) and isinstance(current, Mapping):
        return {key: _join_current_constants(constants[key], current[key], f"{field}.{key}")
                if key in constants and key in current else copy.deepcopy(
                    current[key] if key in current else constants[key])
                for key in constants.keys() | current.keys()}
    if isinstance(constants, (list, tuple)) and isinstance(current, (list, tuple)):
        if len(constants) != len(current):
            raise ValueError(f"Contradictory current compact constants: {field}")
        return [_join_current_constants(a, b, field) for a, b in zip(constants, current)]
    if constants is None or current is None:
        return copy.deepcopy(current if current is not None else constants)
    if type(constants) in (int, float) and type(current) in (int, float):
        equal = math.isclose(constants, current, rel_tol=1e-9, abs_tol=1e-9)
    else:
        equal = type(constants) is type(current) and constants == current
    if not equal:
        raise ValueError(f"Contradictory current compact constants: {field}")
    return copy.deepcopy(current)


def read_task_metadata(row: Mapping, repo_root: Path, *, record_reader=None) -> Mapping:
    """Join exact CURRENT source lines and explicitly declared phase constants.

    Missing files preserve missingness; malformed joins and contradictory values
    fail. Only referenced lines are decoded, never following/future records.
    Phase semantics come from recorded fields/sidecars, never source filenames,
    loader phases, goal IDs, FSM states or demonstration outcomes.
    """
    row = _mapping(row, "row")
    root = Path(repo_root).resolve()
    references = _mapping(row.get("provenance") or {}, "provenance").get("source_records", ())
    grouped = defaultdict(set)
    for ref in references:
        ref = _mapping(ref, "source reference")
        path = _path(ref.get("_loader_source_path"), root)
        index = ref.get("_loader_source_row")
        if type(index) is not int or index < 1:
            raise ValueError("Source metadata record must be a positive one-based index")
        grouped[path].add(index)
    sources, audit, seen = [], [], set()

    def json_metadata(path, expected_hash=None):
        if path in seen:
            return None
        seen.add(path)
        if not path.is_file():
            audit.append({"path": str(path), "missing": True})
            return None
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected_hash is not None and digest != expected_hash:
            raise ValueError(f"Source metadata hash mismatch: {path}")
        value = _mapping(json.loads(data), "source metadata")
        audit.append({"path": str(path), "sha256": digest})
        return value

    expected_stamp = (row.get("graph_t") or {}).get("stamp")

    for path, indices in grouped.items():
        observation_constants = {}
        sidecar = json_metadata(path.with_name(path.stem + "_manifest.json"))
        if sidecar is not None and sidecar.get("schema_version") in {
                "mssr.expert_compact_phase.v1", "mssr.expert_compact_phase.v2"}:
            constants = _mapping(sidecar.get("top_level_constants") or {}, "phase constants")
            sources.append(_current(constants))
            sources.append({"phase": sidecar.get("phase")})
            observation_constants = _clean(_mapping(
                sidecar.get("observation_constants") or {}, "observation constants"))
        if path.is_file():
            before = path.stat()
            found = set()
            def sequential():
                with path.open(encoding="utf-8") as stream:
                    for index, line in enumerate(stream, 1):
                        if index in indices:
                            yield index, json.loads(line)
                        if index >= max(indices):
                            break
            records = sequential() if record_reader is None else record_reader(path, tuple(sorted(indices)))
            for index, value in records:
                if index not in indices or index in found:
                    raise ValueError("Indexed reader returned an unrequested/duplicate source record")
                record = _mapping(value, "current source record")
                actual_stamp = (record.get("graph_t") or {}).get("stamp", record.get("stamp"))
                if expected_stamp is not None and actual_stamp is not None and not math.isclose(
                        expected_stamp, actual_stamp, rel_tol=0., abs_tol=1e-9):
                    raise ValueError("Referenced source physical stamp disagrees with G_t stamp")
                current = _current(record)
                current["observation"] = _join_current_constants(
                    observation_constants, current.get("observation") or {})
                sources.append(current)
                found.add(index)
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                raise ValueError("Source metadata changed while reading")
            if found != indices:
                raise ValueError("Source metadata record out of bounds")
            audit.append({"path": str(path), "records": sorted(found)})
        else:
            if observation_constants:
                sources.append({"observation": observation_constants})
            audit.append({"path": str(path), "missing": True})
        # Environment metadata beside the CURRENT stream/episode, not another
        # phase's observations or a completion/result file.
        directories = [path.parent]
        if path.parent.name == "structural":
            directories.append(path.parent.parent)
        for directory in directories:
            geometry = json_metadata(directory / "environment_geometry.json")
            if geometry is not None:
                sources.append({"environment_geometry": _clean(geometry)})
            course = json_metadata(directory / "course.json")
            if course is not None:
                def static_declaration(value):
                    if isinstance(value, Mapping):
                        return {k: static_declaration(v) for k, v in value.items()
                                if k not in ("depression_m", "current_center_xyz_m")}
                    if isinstance(value, list):
                        return [static_declaration(v) for v in value]
                    return value
                sources.append({"course": _clean(static_declaration(course))})
    return {"sources": sources, "audit": audit}


def _containers(document):
    yield document
    for key in ("observation", "observation_t", "task_context", "environment"):
        nested = document.get(key)
        if isinstance(nested, Mapping):
            yield from _containers(nested)


def _semantic_containers(document):
    """Inspect course identifiers without dropping inherited geometry frames."""
    for source in _containers(document):
        yield source
        course = source.get("course")
        if isinstance(course, Mapping):
            yield course
        for container in (source, course):
            if isinstance(container, Mapping) and isinstance(container.get("mission"), Mapping):
                yield container["mission"]


def _agree(old, new):
    return len(old.values) == len(new.values) and old.valid == new.valid and all(
        math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) for a, b in zip(old.values, new.values))


def _documents(row, metadata):
    return [_current(row), *metadata.get("sources", ()),
            {k: v for k, v in metadata.items() if k not in ("sources", "audit")}]


def _mission_courses(row, metadata):
    return [source["course"] for doc in _documents(row, metadata)
            for source in _containers(_mapping(doc, "metadata source"))
            if isinstance(source.get("course"), Mapping)
            and isinstance(source["course"].get("mission"), Mapping)]


def resolve_mission_task(row: Mapping, metadata: Mapping, config: Mapping) -> Mapping | None:
    """Audit-only historical intent reconstruction, never a policy feature.

    Intervals are reviewed data in config, not inferred from demonstrated
    actions. Their successors reference the CURRENT source's declared plan.
    Once a boundary has occurred, its successor is a floor on mission progress;
    stale geometric Canonical locators cannot switch back to the predecessor.
    No next record, morphology, executor, action or outcome is read.
    """
    provenance = _mapping(row.get("provenance") or {}, "provenance")
    episode = provenance.get("logical_episode_id", row.get("episode_id"))
    protocols = [p for p in config.get("historical_mission_imputations", ())
                 if p.get("episode_id") == episode]
    if not protocols:
        return None
    if len(protocols) != 1:
        raise ValueError("Contradictory duplicate mission protocol")
    index = provenance.get("canonical_transition_index", row.get("transition_index"))
    if type(index) is not int or index < 0:
        raise ValueError("Historical mission requires a current Canonical index")
    protocol = protocols[0]
    order = protocol["mission_order"]
    ids = [t["task_id"] for t in order]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Contradictory mission task identifiers")
    if any(config["task_aliases"].get(t["type"]) not in _TASKS[1:] for t in order):
        raise ValueError("Unresolved successor mission task semantic")
    courses = _mission_courses(row, metadata)
    if not courses:
        raise ValueError("Current mission source unavailable for historical imputation")
    for course in courses:
        actual = [{"task_id": t.get("task_id"), "type": t.get("type")}
                  for t in course["mission"].get("tasks", ())]
        if actual != order:
            raise ValueError("Contradictory current mission plan and historical protocol")
    intervals = protocol["intervals"]
    previous_end, previous_successor = -1, 0
    active = None
    for interval in intervals:
        start, end = interval["start_index"], interval["end_index"]
        if type(start) is not int or type(end) is not int or start < 0 or end < start:
            raise ValueError("Invalid mission interval")
        if start <= previous_end:
            raise ValueError("Overlapping historical mission intervals")
        successor = interval["successor_task_id"]
        predecessor = interval.get("predecessor_task_id")
        if successor not in ids:
            raise ValueError("Unresolved successor mission task")
        successor_index = ids.index(successor)
        if predecessor is None:
            if successor_index != 0:
                raise ValueError("Initial mission handoff must prepare first task")
        elif predecessor not in ids or ids.index(predecessor) + 1 != successor_index:
            raise ValueError("Historical handoff does not prepare declared mission successor")
        if successor_index < previous_successor:
            raise ValueError("Historical mission progress moves backwards")
        previous_end, previous_successor = end, successor_index
        if start <= index:
            active = interval
    floor = ids.index(active["successor_task_id"]) if active else 0
    selected = floor
    inside = active is not None and index <= active["end_index"]
    explicit = set()
    for doc in _documents(row, metadata):
        for source in _semantic_containers(_mapping(doc, "metadata source")):
            for key in ("current_task_id", "active_task_id", "current_obstacle_id"):
                if source.get(key):
                    explicit.add(source[key])
    if len(explicit) > 1:
        raise ValueError("Contradictory explicit current mission task")
    if explicit:
        task_id = next(iter(explicit))
        if task_id not in ids or ids.index(task_id) < floor or inside and ids.index(task_id) != floor:
            raise ValueError("Explicit current task contradicts historical mission assignment")
        selected = ids.index(task_id)
    elif not inside:
        # Modern Canonical selectors use CURRENT physical exit geometry. The
        # legacy morphology-based fallback lacks these declarations and is not
        # trusted. The cursor floor still comes only from boundaries already met.
        physical_tasks = [t for t in courses[0]["mission"]["tasks"] if t["type"] != "goal"]
        geometric_plan = all(isinstance(t.get("parameters", {}).get("exit_pose_xyyaw"), list)
                             for t in physical_tasks)
        locator = (row.get("observation_t") or {}).get("task_id")
        if geometric_plan and locator in ids:
            selected = max(floor, ids.index(locator))
    task_id = ids[selected]
    return {"task_id": task_id, "task": config["task_aliases"][order[selected]["type"]],
            "evidence_class": "HISTORICAL_PROTOCOL_IMPUTATION",
            "predecessor_task_id": active.get("predecessor_task_id") if active else None,
            "successor_task_id": active["successor_task_id"] if active else ids[0],
            "structural_interval": [active["start_index"], active["end_index"]] if active else None,
            "within_structural_interval": inside}


def build_task_view(row: Mapping, metadata: Mapping, config: Mapping) -> tuple[TaskContext, ObservationBundle]:
    """Build existing A1 values; metadata audit entries never become features.

    Supported physical fields are declared by this task config. UNKNOWN/absent
    semantic values become None. Unobserved groups have no features; an observed
    spatial quantity with an uncertified frame is present but explicitly invalid.
    This is the task metadata boundary, not D2's complete G/o/c input selector.
    """
    row, metadata, config = (_mapping(v, n) for v, n in
                             ((row, "row"), (metadata, "metadata"), (config, "config")))
    if config.get("schema_version") != "mssr.bc_v1.task_context.v2" or config.get("task_vocabulary") != _TASKS:
        raise ValueError("Unsupported task context config/vocabulary")
    task_aliases = _mapping(config["task_aliases"], "task_aliases")
    if any(v not in _TASKS[1:] for v in task_aliases.values()):
        raise ValueError("Unsupported semantic alias category")
    documents = _documents(row, metadata)
    containers = [item for doc in documents for item in _containers(_mapping(doc, "metadata source"))]
    tasks, locators = set(), set()
    assignment = resolve_mission_task(row, metadata, config)
    semantic_sources = [source for doc in documents
                        for source in _semantic_containers(_mapping(doc, "metadata source"))]
    for source in semantic_sources:
        task = source.get("task_type")
        if task in task_aliases:
            tasks.add(task_aliases[task])
        for key in ("task_id", "current_task_id", "active_task_id", "current_obstacle_id"):
            value = source.get(key)
            if isinstance(value, str) and value:
                locators.add(value)
    if assignment:
        raw_tasks = {task_aliases[source["task_type"]]
                     for doc in documents[1:]
                     for source in _semantic_containers(_mapping(doc, "metadata source"))
                     if source.get("task_type") in task_aliases}
        if raw_tasks and raw_tasks != {assignment["task"]}:
            raise ValueError("Contradictory explicit source task semantics and mission assignment")
        tasks, locators = {assignment["task"]}, {assignment["task_id"]}
    if len(tasks) > 1 or len(locators) > 1:
        raise ValueError("Contradictory current task identifiers/semantics")
    locator = next(iter(locators), None)
    if not tasks:
        single_tasks = {key for source in semantic_sources for key in ("gap", "stairs", "button", "navigation")
                        if isinstance(source.get(key), Mapping) and not isinstance(source.get("mission"), Mapping)}
        if len(single_tasks) == 1:
            tasks = single_tasks
    context_features, observation_features, observed_frames, object_counts = {}, {}, {}, {}

    def add(target, name, raw, kind="scalar", frame=None, spatial=False):
        valid = True
        if isinstance(raw, Mapping):
            valid = raw.get("valid", True)
            if type(valid) is not bool:
                raise ValueError("Feature validity must be boolean")
            frame = raw.get("frame_id", frame)
            raw = raw.get("values", raw.get("value"))
        if raw is None:
            return
        if kind == "scalar":
            values = (raw,) if not isinstance(raw, (list, tuple)) else tuple(raw)
            size = 1
        else:
            if not isinstance(raw, (list, tuple)):
                raise ValueError("Expected numeric physical vector")
            values = tuple(raw)
            size = int(kind[6:]) if kind.startswith("vector") else None
        if size is not None and len(values) != size:
            raise ValueError("Invalid physical vector size")
        if not values:
            return
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError("Expected finite numeric physical values")
        if spatial:
            source_frame = frame
            if name in observed_frames and observed_frames[name] != source_frame:
                raise ValueError(f"Contradictory source frame for {name}; D1 cannot transform frames")
            observed_frames[name] = source_frame
            certified = frame in config["frames"]
            name = f"{frame if certified else 'unframed'}.{name}"
            valid = valid and certified
        feature = NumericFeature(name, tuple(float(v) for v in values), valid)
        if name in target and not _agree(target[name], feature):
            raise ValueError(f"Contradictory authoritative metadata: {name}")
        target.setdefault(name, feature)

    def frame_of(source, fallback=None):
        declared = [source[k] for k in ("coordinate_frame", "geometry_frame", "frame_id")
                    if source.get(k) is not None]
        if any(value != declared[0] for value in declared[1:]):
            raise ValueError("Contradictory frame declarations in current geometry")
        return source.get("coordinate_frame", source.get("geometry_frame", source.get("frame_id", fallback)))

    def group(name, source, parent_frame):
        spec = config["geometry"][name]
        frame = frame_of(source, parent_frame)
        for placement, fields in spec.items():
            for key, kind in fields.items():
                if key in source:
                    add(observation_features if placement == "dynamic" else context_features,
                        f"{name}.{key}", source[key], kind,
                        frame="world" if "world" in key else frame,
                        spatial=placement == "spatial" or placement == "dynamic" and kind.startswith("vector"))

    def geometry(source, inherited_frame=None, mission_tasks=()):
        mission_tasks = (source.get("mission") or {}).get("tasks", ()) or mission_tasks
        mission_scope = bool(mission_tasks)
        frame = frame_of(source, inherited_frame)
        for name in ("gap", "stairs", "button"):
            value = source.get(name)
            if isinstance(value, Mapping) and (not tasks or name in tasks):
                group(name, value, frame)
        # Scenario dimensions are current recorded environment configuration.
        scenario = source.get("scenario")
        if isinstance(scenario, Mapping):
            for task in tasks & {"gap", "stairs"}:
                group(task, {k: v for k, v in scenario.items()
                             if k in config["geometry"][task]["context"]}, frame)
        group("environment", source, frame)
        for key in ("physical_track", "layout"):
            physical = source.get(key)
            if isinstance(physical, Mapping):
                geometry(physical, source.get("layout_frame_id", frame), mission_tasks)
        group("cones", source, frame)
        def objects(key, group_name):
            value = source.get(key)
            if value is None:
                return ()
            if not isinstance(value, (list, tuple)):
                raise ValueError("Physical objects must be a list")
            if mission_scope:
                if not locator and value:
                    raise ValueError("Unresolved current mission task: cannot scope physical inventory")
                scope = config["mission_geometry_scope"]
                scoped = []
                for item in value:
                    item = _mapping(item, "physical mission object")
                    name = item.get("name")
                    if name in scope["shared_object_names"]:
                        scoped.append(item)
                        continue
                    owners = set()
                    for task in mission_tasks:
                        task_id = task.get("task_id")
                        if not isinstance(task_id, str) or not task_id:
                            raise ValueError("Invalid mission task identifier for object ownership")
                        prefix = "".join(char if char.isalnum() else "_" for char in task_id) + "_"
                        if item.get("task_id") == task_id or isinstance(name, str) and name.startswith(prefix):
                            owners.add(task_id)
                        if task.get("type") == "goal" and name in scope["terminal_goal_object_names"]:
                            owners.add(task_id)
                    if len(owners) != 1:
                        raise ValueError("Unscoped or ambiguous physical object ownership in mission")
                    if locator in owners:
                        scoped.append(item)
                value = scoped
            if group_name in object_counts and object_counts[group_name] != len(value):
                raise ValueError(f"Contradictory physical object count for {group_name}")
            object_counts[group_name] = len(value)
            return value

        for i, center in enumerate(objects("cone_centers_xy_m", "cones")):
            add(context_features, f"cones.{i}.center_xy_m", center, "vector2", frame, True)
        # A mission inventory is scoped to its current task before exposure.
        # These are actual objects, never a navigation route or waypoint list.
        for collection, spec in config.get("physical_static_objects", {}).items():
            for i, item in enumerate(objects(collection, spec["group"])):
                item = _mapping(item, "physical observation object")
                item_frame = frame_of(item, frame)
                for key, field in spec["fields"].items():
                    if key in item:
                        add(context_features, f"{spec['group']}.{i}.{key}", item[key],
                            field["kind"], item_frame, field["spatial"])
        for i, box in enumerate(objects("collision_boxes", "obstacles")):
            box = _mapping(box, "physical collision box")
            box_frame = frame_of(box, frame)
            for key in ("center_xyz_m", "size_xyz_m"):
                if key in box:
                    add(context_features, f"obstacles.{i}.{key}", box[key], "vector3", box_frame, True)
            for key in ("yaw_deg", "pitch_deg"):
                if key in box:
                    add(context_features, f"obstacles.{i}.{key}", box[key], frame=box_frame, spatial=True)
        stage_frame = source.get("stage_frame")
        if isinstance(stage_frame, Mapping):
            if "origin_world_xyz_m" in stage_frame:
                add(context_features, "task_frame.origin_xyz_m", stage_frame["origin_world_xyz_m"], "vector3", "world", True)
            if "yaw_rad" in stage_frame:
                add(context_features, "task_frame.yaw_rad", stage_frame["yaw_rad"], frame="world", spatial=True)
        goal = source.get("current_goal")
        if isinstance(goal, Mapping):
            for key, kind in (("position_xyz_m", "vector3"), ("xyyaw", "vector3")):
                if key in goal:
                    add(context_features, f"goal.{key}", goal[key], kind, frame_of(goal, frame), True)

    for source in containers:
        geometry(source)
        # Intrinsic current measurement survives loss of Canonical root frame.
        relative = source.get("relative_geometry")
        if isinstance(relative, Mapping) and "depression_m" in relative:
            add(observation_features, "button.depression_m", relative["depression_m"])
        for key in ("course", "environment_geometry", "task_parameters"):
            value = source.get(key)
            if not isinstance(value, Mapping):
                continue
            # Read physical CURRENT course/environment fields regardless of
            # whether a task is assigned. geometry() never descends into mission.
            geometry(value, frame_of(source))
            mission = value.get("mission")
            if isinstance(mission, Mapping):
                matches = [t for t in mission.get("tasks", ()) if t.get("task_id") == locator] if locator else []
                if len(matches) > 1:
                    raise ValueError("Contradictory duplicate current mission task")
                if matches:
                    selected = matches[0]
                    if selected.get("type") in task_aliases:
                        tasks.add(task_aliases[selected["type"]])
                    parameters = _mapping(selected.get("parameters") or {}, "current task parameters")
                    selected_frame = frame_of(value, frame_of(source))
                    if selected.get("type") == "goal":
                        group("goal", parameters, selected_frame)
                    else:
                        geometry(parameters, selected_frame)
                # Unassigned/future task instructions and nominal routes stay
                # excluded; known physical objects above remain observable.
    if len(tasks) > 1:
        raise ValueError("Contradictory authoritative task metadata")
    return (TaskContext(task_id=next(iter(tasks), None),
                        static_geometry=tuple(context_features[k] for k in sorted(context_features))),
            ObservationBundle(tuple(observation_features[k] for k in sorted(observation_features))))
