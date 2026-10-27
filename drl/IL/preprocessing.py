"""Canonical preprocessing for MSSR imitation-learning datasets.

Raw/compact records are not modified.  This module adapts decoded expert
transitions into a platform-aware canonical representation.

Current implementation:
    raw/decoded transition -> canonical (G_t, o_t, a_t, G_t+1)
        - topology preserved
        - stable structural roles/root reconstructed from target morphology
        - self-assembly and self-reconfiguration represented as transient states
        - environment/course separated from robot graph into o_t
        - per-module commands at their recorded dispatch, with no behavior macros
        - legacy IK targets retained for configuration-level BC

Canonical v4 learning contract:
    G_t/G_t+1: module nodes, physical/contact edges, measured dynamic state.
        control_state.active_alignments describes earlier dispatched operations.
    o_t: current obstacle identity and geometry in the root module's body frame.
    a_t.commands: {module_ids, command, parameters}, at this dispatch time.
        wheel_velocity: left_rad_s/right_rad_s
        pan_velocity/tilt_velocity: rate_rad_s
        set_pan/set_tilt: angle_rad
        rotate_pan_by/rotate_tilt_by: delta_rad
        align_faces: one alignment intent; execution phases live in provenance
        dock/undock/etc.: actual executor arguments
    a_t.configuration_target_rad: role.joint -> radians, for recorded IK only.
    supervision.command_mask selects instantaneous command targets.
    supervision.configuration_mask selects a whole IK configuration target.
    supervision.sample_weight counts BC-eligible repeats; source_repeat_count
    counts all source rows, including executor progress without a new decision.
    Earlier differing continuous setpoints on the exact same source input and
    timestamp remain recorded commands with false command masks. Only the last
    revision supervises that resource. Omitted channels are not stop/wait labels.
    An IK recorded_sequence has only interval endpoint states: its command list
    documents execution order, never synthetic intermediate transitions.
    valid_for_behavior_cloning is true if either target kind is supervised;
    a trainer must select the corresponding mask/objective explicitly.
    Metadata/provenance are not automatic network inputs. Static world geometry
    stays in the compact episode metadata. Face vectors and node poses are in
    world coordinates; edge poses are module B expressed in module A's frame.
    Compatible same-time commands on identical physical state/observation and
    recorded next state form one action. Distinct endpoints stay separate.
    Source references/counts remain available after coalescing.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


ACTIVE_MORPHOLOGY_CONFIGS = {
    "snake8": "smores_snake8.json",
    "rc_car8": "smores_rc_car8.json",
    "mobile_manipulator8": "smores_mobile_manipulator8.json",
}


class PreprocessingError(ValueError):
    """Raised when a raw record cannot be adapted consistently."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def morphology_config_dir(repo_root: Path | None = None) -> Path:
    root = repository_root() if repo_root is None else Path(repo_root)
    return root / "mssr_ws/src/mssr_expert/config"


def load_target_graph(
    morphology: str,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    try:
        filename = ACTIVE_MORPHOLOGY_CONFIGS[morphology]
    except KeyError as exc:
        raise PreprocessingError(
            f"Unsupported active morphology: {morphology!r}"
        ) from exc

    path = morphology_config_dir(repo_root) / filename
    return json.loads(path.read_text())


def _module_id(node: Mapping[str, Any]) -> str:
    value = node.get("module_id") or node.get("node_id")
    if not isinstance(value, str) or not value:
        raise PreprocessingError("Graph node has no valid module_id/node_id.")
    return value


def _physical_edges(
    graph: Mapping[str, Any],
) -> list[tuple[str, str, str, str]]:
    """Return attached physical edges as (a, face_a, b, face_b)."""

    result: list[tuple[str, str, str, str]] = []

    for edge in graph.get("edges", []):
        attrs = edge.get("attributes", {})

        # Ignore explicitly non-attached relations/contact candidates.
        if attrs.get("is_attached") is False:
            continue

        a = edge.get("module_a_id")
        b = edge.get("module_b_id")
        fa = attrs.get("face_a")
        fb = attrs.get("face_b")

        if all(isinstance(x, str) and x for x in (a, b, fa, fb)):
            result.append((a, fa, b, fb))

    return result


def _target_edges(
    target: Mapping[str, Any],
) -> list[tuple[str, str, str, str]]:
    result: list[tuple[str, str, str, str]] = []

    for edge in target.get("edges", []):
        attrs = edge.get("attributes", {})
        a = edge.get("module_a_id")
        b = edge.get("module_b_id")
        fa = attrs.get("face_a")
        fb = attrs.get("face_b")

        if not all(isinstance(x, str) and x for x in (a, b, fa, fb)):
            raise PreprocessingError("Malformed target morphology edge.")

        result.append((a, fa, b, fb))

    return result


def _face_signatures(
    node_ids: list[str],
    edges: list[tuple[str, str, str, str]],
) -> dict[str, tuple[str, ...]]:
    faces: dict[str, list[str]] = {node_id: [] for node_id in node_ids}

    for a, fa, b, fb in edges:
        if a in faces:
            faces[a].append(fa)
        if b in faces:
            faces[b].append(fb)

    return {
        node_id: tuple(sorted(local_faces))
        for node_id, local_faces in faces.items()
    }


def _adjacency(
    node_ids: list[str],
    edges: list[tuple[str, str, str, str]],
) -> dict[str, list[tuple[str, str, str]]]:
    adj: dict[str, list[tuple[str, str, str]]] = {
        node_id: [] for node_id in node_ids
    }

    for a, fa, b, fb in edges:
        if a in adj and b in adj:
            adj[a].append((b, fa, fb))
            adj[b].append((a, fb, fa))

    return adj


def topology_fingerprint(graph: Mapping[str, Any]) -> tuple[Any, ...]:
    """Fingerprint only the structural information relevant to role matching."""

    modules = tuple(sorted(_module_id(n) for n in graph.get("nodes", [])))

    edges = tuple(
        sorted(
            min(
                (a, fa, b, fb),
                (b, fb, a, fa),
            )
            for a, fa, b, fb in _physical_edges(graph)
        )
    )

    return modules, edges


def match_target_to_physical(
    graph: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    max_matches: int = 2,
) -> tuple[dict[str, str] | None, str]:
    """Match target vertices to physical modules using topology + docking faces.

    Returns:
        (mapping, "unique")      exactly one mapping
        (None, "ambiguous")     more than one mapping
        (None, "no_match")      no valid mapping

    Pose and module numeric IDs are intentionally NOT used to invent labels.
    """

    physical_ids = [_module_id(n) for n in graph.get("nodes", [])]
    target_ids = [
        str(node["node_id"])
        for node in target.get("nodes", [])
    ]

    p_edges = _physical_edges(graph)
    t_edges = _target_edges(target)

    p_sig = _face_signatures(physical_ids, p_edges)
    t_sig = _face_signatures(target_ids, t_edges)

    p_adj = _adjacency(physical_ids, p_edges)
    t_adj = _adjacency(target_ids, t_edges)

    candidates: dict[str, list[str]] = {
        target_id: [
            physical_id
            for physical_id in physical_ids
            if p_sig[physical_id] == t_sig[target_id]
        ]
        for target_id in target_ids
    }

    if any(not values for values in candidates.values()):
        return None, "no_match"

    # Most constrained target vertices first.
    order = sorted(
        target_ids,
        key=lambda target_id: (
            len(candidates[target_id]),
            -len(t_adj[target_id]),
            target_id,
        ),
    )

    matches: list[dict[str, str]] = []
    assignment: dict[str, str] = {}
    used_physical: set[str] = set()

    def compatible(target_id: str, physical_id: str) -> bool:
        for target_neighbor, own_face, neighbor_face in t_adj[target_id]:
            if target_neighbor not in assignment:
                continue

            physical_neighbor = assignment[target_neighbor]

            if not any(
                neighbor == physical_neighbor
                and local_face == own_face
                and remote_face == neighbor_face
                for neighbor, local_face, remote_face in p_adj[physical_id]
            ):
                return False

        return True

    def search(index: int) -> None:
        if len(matches) >= max_matches:
            return

        if index == len(order):
            matches.append(dict(assignment))
            return

        target_id = order[index]

        for physical_id in candidates[target_id]:
            if physical_id in used_physical:
                continue
            if not compatible(target_id, physical_id):
                continue

            assignment[target_id] = physical_id
            used_physical.add(physical_id)

            search(index + 1)

            used_physical.remove(physical_id)
            del assignment[target_id]

    search(0)

    if not matches:
        return None, "no_match"
    if len(matches) > 1:
        return None, "ambiguous"

    return matches[0], "unique"


def _target_roles(
    target: Mapping[str, Any],
) -> tuple[dict[str, str], str | None]:
    role_by_vertex: dict[str, str] = {}
    root_vertices: list[str] = []

    for node in target.get("nodes", []):
        vertex_id = str(node["node_id"])
        attrs = node.get("attributes", {})

        role = attrs.get("target_role")
        if isinstance(role, str) and role:
            role_by_vertex[vertex_id] = role

        if attrs.get("is_target_root") is True:
            root_vertices.append(vertex_id)

    if len(root_vertices) > 1:
        raise PreprocessingError(
            f"Target morphology declares multiple roots: {root_vertices}"
        )

    root_vertex = root_vertices[0] if root_vertices else None
    return role_by_vertex, root_vertex


class CanonicalAdapter:
    """Stateful adapter with topology-assignment caching."""

    def __init__(self, *, repo_root: Path | None = None) -> None:
        self.repo_root = (
            repository_root() if repo_root is None else Path(repo_root)
        )
        self._target_cache: dict[str, dict[str, Any]] = {}
        self._assignment_cache: dict[
            tuple[str, tuple[Any, ...]],
            tuple[dict[str, str] | None, str],
        ] = {}

    def target_graph(self, morphology: str) -> dict[str, Any]:
        if morphology not in self._target_cache:
            self._target_cache[morphology] = load_target_graph(
                morphology,
                repo_root=self.repo_root,
            )
        return self._target_cache[morphology]

    def structural_assignment(
        self,
        graph: Mapping[str, Any],
        morphology: str,
    ) -> tuple[dict[str, str] | None, str]:
        key = (morphology, topology_fingerprint(graph))

        if key not in self._assignment_cache:
            self._assignment_cache[key] = match_target_to_physical(
                graph,
                self.target_graph(morphology),
            )

        return self._assignment_cache[key]

    def adapt_graph(
        self,
        graph: Mapping[str, Any],
        *,
        morphology: str,
    ) -> dict[str, Any]:
        """Build canonical robot graph G_t or G_{t+1}."""

        # During self-assembly/reconfiguration the physical graph is real and
        # must be preserved, but there is no single stable current morphology.
        # Treat both operations identically at the structural-state level; the
        # distinct labels only retain which transition process is underway.
        if morphology in {"assembling", "reconfiguring"}:
            canonical = copy.deepcopy(dict(graph))

            canonical["schema_version"] = "mssr.canonical_robot_graph.v1"
            canonical["source_schema_version"] = graph.get(
                "schema_version"
            )
            canonical["morphology"] = morphology
            canonical["morphology_stable"] = False
            canonical["root_module_id"] = None
            canonical["structural_assignment_status"] = morphology

            globals_ = canonical.get("global_attributes")
            if isinstance(globals_, dict):
                globals_.pop("course", None)
                globals_["morphology"] = morphology

            for node in canonical.get("nodes", []):
                attributes = node.setdefault("attributes", {})
                attributes["structural_role_current"] = None
                attributes["structural_role_current_valid"] = False
                # Keep the older spelling too until the tensorizer is frozen.
                attributes["structural_role_valid"] = False
                attributes["is_structural_root"] = False

            return canonical


        target = self.target_graph(morphology)
        target_to_physical, status = self.structural_assignment(
            graph,
            morphology,
        )

        role_by_vertex, root_vertex = _target_roles(target)

        role_by_module: dict[str, str] = {}
        root_module_id: str | None = None

        if target_to_physical is not None:
            role_by_module = {
                physical_id: role_by_vertex[target_id]
                for target_id, physical_id in target_to_physical.items()
                if target_id in role_by_vertex
            }

            if root_vertex is not None:
                root_module_id = target_to_physical.get(root_vertex)

        canonical = copy.deepcopy(dict(graph))

        source_schema = canonical.get("schema_version")
        canonical["schema_version"] = "mssr.canonical_robot_graph.v1"
        canonical["source_schema_version"] = source_schema
        canonical["morphology"] = morphology
        canonical["morphology_stable"] = True
        canonical["root_module_id"] = root_module_id
        canonical["structural_assignment_status"] = status

        # Environment belongs to o_t, not to robot graph G_t.
        global_attributes = canonical.setdefault("global_attributes", {})
        global_attributes.pop("course", None)
        global_attributes["morphology"] = morphology

        for node in canonical.get("nodes", []):
            module_id = _module_id(node)
            attrs = node.setdefault("attributes", {})

            role = role_by_module.get(module_id)

            attrs["structural_role_current"] = role
            attrs["structural_role_current_valid"] = role is not None
            attrs["structural_role_valid"] = role is not None
            attrs["is_structural_root"] = (
                root_module_id is not None
                and module_id == root_module_id
            )

        return canonical


def morphology_from_record(record: Mapping[str, Any]) -> str | None:
    """Return stable morphology or an explicit transient structural state."""

    task_type = str(record.get("task_type") or "").strip().lower()
    stage_name = str(record.get("stage_name") or "").strip().lower()

    if task_type in {"parallel_self_assembly", "self_assembly"} or stage_name in {
        "parallel_self_assembly",
        "self_assembly",
    }:
        return "assembling"

    if task_type == "self_reconfiguration" or stage_name == "self_reconfiguration":
        return "reconfiguring"

    observation = record.get("observation")
    if isinstance(observation, Mapping):
        value = observation.get("morphology")
        if isinstance(value, str) and value:
            return value
    return None


# ---------------------------------------------------------------------------
# Canonical task/environment observation o_t
# ---------------------------------------------------------------------------

import math


def _yaw_from_quaternion_xyzw(q: list[float]) -> float:
    if len(q) != 4:
        raise PreprocessingError("Expected quaternion [x, y, z, w].")

    x, y, z, w = q
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _root_pose(
    canonical_graph: Mapping[str, Any],
) -> tuple[str, list[float], float, list[float]]:
    root_id = canonical_graph.get("root_module_id")

    if not isinstance(root_id, str) or not root_id:
        raise PreprocessingError(
            "Canonical graph has no valid structural root."
        )

    for node in canonical_graph.get("nodes", []):
        if _module_id(node) != root_id:
            continue

        attrs = node.get("attributes", {})
        pose = attrs.get("pose", {})

        position = pose.get("position") or attrs.get("position")
        orientation = (
            pose.get("orientation_xyzw")
            or pose.get("orientation")
            or attrs.get("orientation")
        )

        if (
            not isinstance(position, list)
            or len(position) != 3
            or not isinstance(orientation, list)
            or len(orientation) != 4
        ):
            raise PreprocessingError(
                f"Root module {root_id!r} has no valid pose."
            )

        return (
            root_id,
            [float(v) for v in position],
            _yaw_from_quaternion_xyzw(
                [float(v) for v in orientation]
            ),
            [float(v) for v in orientation],
        )

    raise PreprocessingError(
        f"Root module {root_id!r} not present in graph."
    )


def _world_point_to_root(
    xyz: list[float],
    root_xyz: list[float],
    root_orientation: float | list[float],
) -> list[float]:
    dx = float(xyz[0]) - root_xyz[0]
    dy = float(xyz[1]) - root_xyz[1]
    dz = float(xyz[2]) - root_xyz[2]

    if isinstance(root_orientation, (int, float)):
        c = math.cos(root_orientation)
        s = math.sin(root_orientation)
        return [c * dx + s * dy, -s * dx + c * dy, dz]

    if len(root_orientation) != 4:
        raise PreprocessingError("Root orientation must be quaternion xyzw.")
    x, y, z, w = (float(value) for value in root_orientation)
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm <= 1e-12:
        raise PreprocessingError("Root orientation quaternion has zero norm.")
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    return [
        (1 - 2*(y*y + z*z))*dx + 2*(x*y + w*z)*dy + 2*(x*z - w*y)*dz,
        2*(x*y - w*z)*dx + (1 - 2*(x*x + z*z))*dy + 2*(y*z + w*x)*dz,
        2*(x*z + w*y)*dx + 2*(y*z - w*x)*dy + (1 - 2*(x*x + y*y))*dz,
    ]


def _world_direction_to_root(
    xy: list[float],
    root_yaw: float,
) -> list[float]:
    x = float(xy[0])
    y = float(xy[1])

    c = math.cos(root_yaw)
    s = math.sin(root_yaw)

    return [
        c * x + s * y,
        -s * x + c * y,
    ]


def _task_direction_world(
    task: Mapping[str, Any],
) -> list[float] | None:
    params = task.get("parameters", {})

    entry = params.get("entry_pose_xyyaw")
    exit_pose = params.get("exit_pose_xyyaw")

    if (
        isinstance(entry, list)
        and len(entry) >= 2
        and isinstance(exit_pose, list)
        and len(exit_pose) >= 2
    ):
        dx = float(exit_pose[0]) - float(entry[0])
        dy = float(exit_pose[1]) - float(entry[1])
        norm = math.hypot(dx, dy)

        if norm > 1e-9:
            return [dx / norm, dy / norm]

    gap = params.get("gap")
    if isinstance(gap, Mapping):
        direction = gap.get("crossing_direction_world_xy")
        if isinstance(direction, list) and len(direction) == 2:
            return [float(direction[0]), float(direction[1])]

    stairs = params.get("stairs")
    if isinstance(stairs, Mapping):
        direction = stairs.get("ascent_direction_world_xy")
        if isinstance(direction, list) and len(direction) == 2:
            return [float(direction[0]), float(direction[1])]

    return None


def _task_passed(
    task: Mapping[str, Any],
    root_xyz: list[float],
) -> bool:
    """Whether the root has crossed the task exit plane."""

    params = task.get("parameters", {})
    exit_pose = params.get("exit_pose_xyyaw")
    direction = _task_direction_world(task)

    if (
        not isinstance(exit_pose, list)
        or len(exit_pose) < 2
        or direction is None
    ):
        return False

    dx = root_xyz[0] - float(exit_pose[0])
    dy = root_xyz[1] - float(exit_pose[1])

    return (
        dx * direction[0]
        + dy * direction[1]
    ) > 0.0


def _explicit_task_id(
    record: Mapping[str, Any],
    course: Mapping[str, Any],
    mission: Mapping[str, Any],
) -> str | None:
    observation = record.get("observation")
    task_context = (
        observation.get("task_context")
        if isinstance(observation, Mapping) else None
    )
    for container in (observation, task_context, record, course, mission):
        if not isinstance(container, Mapping):
            continue
        for key in ("current_obstacle_id", "active_task_id", "current_task_id"):
            value = container.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _select_active_upcoming_task(
    mission: Mapping[str, Any],
    root_xyz: list[float],
) -> Mapping[str, Any] | None:
    """Select first mission obstacle whose exit has not been crossed."""

    tasks = [
        task
        for task in mission.get("tasks", [])
        if task.get("type") != "goal"
    ]

    if not tasks:
        return None

    for task in tasks:
        if not _task_passed(task, root_xyz):
            return task

    # Keep final physical task active after its geometric exit. This is
    # especially useful for button interaction, whose completion is not
    # determined by base position alone.
    return tasks[-1]



def _navigation_goal_relative(
    goal: list[float],
    *,
    floor_z: float,
    root_xyz: list[float],
    root_yaw: float,
    root_orientation: list[float],
) -> list[float]:
    return [
        *_world_point_to_root(
            [float(goal[0]), float(goal[1]), floor_z],
            root_xyz, root_orientation,
        )[:2],
        _wrap_angle(float(goal[2]) - root_yaw),
    ]


def _navigation_cones_relative(
    centers: list[Any],
    *,
    floor_z: float,
    cone_height_m: float | None,
    root_xyz: list[float],
    root_orientation: list[float],
) -> list[dict[str, Any]]:
    center_z = floor_z + (cone_height_m / 2.0 if cone_height_m is not None else 0.0)
    result = []
    for index, center in enumerate(centers):
        if not isinstance(center, list) or len(center) < 2:
            continue
        relative = _world_point_to_root(
            [float(center[0]), float(center[1]), center_z],
            root_xyz, root_orientation,
        )
        result.append({
            "cone_index": index,
            "center_relative_xyz_m": relative,
            "distance_xy_m": math.hypot(float(center[0]) - root_xyz[0], float(center[1]) - root_xyz[1]),
        })
    return result


def _canonical_task_geometry(
    task: Mapping[str, Any],
    *,
    course: Mapping[str, Any],
    root_xyz: list[float],
    root_yaw: float,
    root_orientation: list[float],
) -> dict[str, Any]:
    params = task.get("parameters", {})
    task_type = task.get("type")

    result: dict[str, Any] = {}

    if task_type == "gap":
        gap = params.get("gap", {})

        near_x = gap.get("near_edge_x_m")
        far_x = gap.get("far_edge_x_m")
        width = gap.get("width_m")
        if width is None and isinstance(near_x, (int, float)) and isinstance(far_x, (int, float)):
            width = abs(float(far_x) - float(near_x))
        result = {
            "width_m": width,
            "bank_height_m": gap.get("bank_height_m"),
        }

        for source, fallback_x, target in (
            ("near_edge_center_world_xyz_m", near_x, "near_edge_relative_xyz_m"),
            ("far_edge_center_world_xyz_m", far_x, "far_edge_relative_xyz_m"),
        ):
            point = gap.get(source)
            if not (isinstance(point, list) and len(point) == 3) and isinstance(fallback_x, (int, float)):
                point = [float(fallback_x), root_xyz[1], float(params.get("floor_height_m", 0.0))]
                result["edge_reference"] = "closest_point_at_root_world_y"
            if isinstance(point, list) and len(point) == 3:
                result[target] = _world_point_to_root(
                    point, root_xyz, root_orientation
                )

        direction = gap.get("crossing_direction_world_xy")
        if not (isinstance(direction, list) and len(direction) == 2) and isinstance(near_x, (int, float)) and isinstance(far_x, (int, float)):
            direction = [1.0 if far_x >= near_x else -1.0, 0.0]
        if isinstance(direction, list) and len(direction) == 2:
            result["crossing_direction_root_xy"] = (
                _world_direction_to_root(direction, root_yaw)
            )

    elif task_type == "stairs":
        stairs = params.get("stairs", {})

        result = {
            "riser_depth_m": stairs.get("riser_depth_m"),
            "top_heights_m": stairs.get("top_heights_m"),
            "upper_deck_height_m": params.get("upper_deck_height_m"),
        }

        first = stairs.get("first_riser_center_world_xyz_m")
        if isinstance(first, list) and len(first) == 3:
            result["first_riser_relative_xyz_m"] = (
                _world_point_to_root(first, root_xyz, root_orientation)
            )

        direction = stairs.get("ascent_direction_world_xy")
        if isinstance(direction, list) and len(direction) == 2:
            result["ascent_direction_root_xy"] = (
                _world_direction_to_root(direction, root_yaw)
            )
        task_id = task.get("task_id")
        boxes = course.get("collision_boxes")
        if isinstance(task_id, str) and isinstance(boxes, list):
            prefix = task_id.replace("-", "_") + "_"
            result["collision_boxes_relative"] = [
                {
                    "name": box.get("name"),
                    "semantic": box.get("semantic"),
                    "center_relative_xyz_m": _world_point_to_root(
                        box["center_xyz_m"], root_xyz, root_orientation
                    ),
                    "size_xyz_m": copy.deepcopy(box.get("size_xyz_m")),
                    "orientation_relative_xyzw": _box_relative_orientation(box, root_orientation),
                }
                for box in boxes
                if isinstance(box, Mapping)
                and isinstance(box.get("name"), str)
                and box["name"].startswith(prefix)
                and isinstance(box.get("center_xyz_m"), list)
                and len(box["center_xyz_m"]) == 3
            ]

    elif task_type == "button":
        button = params.get("button", {})

        center = (
            button.get("current_center_xyz_m")
            or button.get("center_xyz_m")
        )

        if isinstance(center, list) and len(center) == 3:
            result["center_relative_xyz_m"] = (
                _world_point_to_root(center, root_xyz, root_orientation)
            )

        direction = button.get("press_direction_world_xy")
        if isinstance(direction, list) and len(direction) == 2:
            result["press_direction_root_xy"] = (
                _world_direction_to_root(direction, root_yaw)
            )

        result["plunger_stroke_m"] = button.get("plunger_stroke_m")
        result["depression_m"] = button.get("depression_m")

    elif task_type == "flat_navigation":
        floor_z = float(params.get("floor_height_m", root_xyz[2]))
        goal = params.get("navigation_goal_xyyaw")
        goal_source = "navigation_goal_xyyaw"
        if not (isinstance(goal, list) and len(goal) >= 3):
            goal = params.get("exit_pose_xyyaw")
            goal_source = "mission_exit_pose"
        if isinstance(goal, list) and len(goal) >= 3:
            result["goal_relative_xyyaw"] = _navigation_goal_relative(
                goal,
                floor_z=floor_z,
                root_xyz=root_xyz,
                root_yaw=root_yaw,
                root_orientation=root_orientation,
            )
            result["goal_source"] = goal_source
        cones = params.get("cone_centers_xy_m")
        result["cone_positions_available"] = isinstance(cones, list)
        result["vehicle_center_reference"] = "graph_t.root_module_pose"
        if isinstance(cones, list):
            height = params.get("cone_height_m")
            cone_height = float(height) if isinstance(height, (int, float)) else None
            result["cones_relative"] = _navigation_cones_relative(
                cones,
                floor_z=floor_z,
                cone_height_m=cone_height,
                root_xyz=root_xyz,
                root_orientation=root_orientation,
            )
            for key in ("cone_radius_m", "cone_height_m"):
                if key in params:
                    result[key] = copy.deepcopy(params[key])
        for key in ("corridor_width_m", "vehicle_footprint"):
            if key in params:
                result[key] = copy.deepcopy(params[key])

    return result



def _course_from_record(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    graph = record.get("graph_t")
    if isinstance(graph, Mapping):
        globals_ = graph.get("global_attributes")
        if isinstance(globals_, Mapping) and isinstance(globals_.get("course"), Mapping):
            return globals_["course"]
    observation = record.get("observation")
    if isinstance(observation, Mapping) and isinstance(observation.get("course"), Mapping):
        return observation["course"]
    return None


def _expert_course_task(course: Mapping[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Identify a single-task expert course without pretending it is a mission."""
    for task_type in ("gap", "stairs", "button"):
        value = course.get(task_type)
        if isinstance(value, Mapping):
            return task_type, dict(value)
    navigation = course.get("navigation")
    if isinstance(navigation, Mapping):
        return "flat_navigation", dict(navigation)
    return None



def _expert_navigation_physical_layout(
    course: Mapping[str, Any],
    episode_environment: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any] | None, str | None]:
    physical = course.get("physical_track")
    if isinstance(physical, Mapping):
        return physical, "recorded_physical_track"
    if isinstance(episode_environment, Mapping):
        layout = episode_environment.get("layout")
        if isinstance(layout, Mapping):
            return layout, "episode_environment_geometry"
    return None, None


def _expert_course_geometry(
    task_type: str,
    params: Mapping[str, Any],
    course: Mapping[str, Any],
    *,
    episode_environment: Mapping[str, Any] | None,
    root_xyz: list[float],
    root_yaw: float,
    root_orientation: list[float],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if task_type == "gap":
        result["width_m"] = params.get("width_m")
        for source, target in (
            ("near_edge_x_m", "near_edge_relative_xyz_m"),
            ("far_edge_x_m", "far_edge_relative_xyz_m"),
        ):
            x = params.get(source)
            if isinstance(x, (int, float)):
                result[target] = _world_point_to_root(
                    [float(x), root_xyz[1], 0.0], root_xyz, root_orientation
                )
        result["edge_reference"] = "closest_point_at_root_world_y"
    elif task_type == "stairs":
        for key in ("riser_depth_m", "top_heights_m"):
            if key in params:
                result[key] = copy.deepcopy(params[key])
        first_x = params.get("first_riser_x_m")
        if isinstance(first_x, (int, float)):
            result["first_riser_relative_xyz_m"] = _world_point_to_root(
                [float(first_x), root_xyz[1], 0.0], root_xyz, root_orientation
            )
        boxes = course.get("collision_boxes")
        if isinstance(boxes, list):
            result["collision_boxes_relative"] = [
                {
                    "name": box.get("name"),
                    "semantic": box.get("semantic"),
                    "center_relative_xyz_m": _world_point_to_root(
                        box["center_xyz_m"], root_xyz, root_orientation
                    ),
                    "size_xyz_m": copy.deepcopy(box.get("size_xyz_m")),
                    "orientation_relative_xyzw": _box_relative_orientation(box, root_orientation),
                }
                for box in boxes
                if isinstance(box, Mapping)
                and isinstance(box.get("center_xyz_m"), list)
                and len(box["center_xyz_m"]) == 3
            ]
    elif task_type == "button":
        center = params.get("current_center_xyz_m") or params.get("center_xyz_m")
        if isinstance(center, list) and len(center) == 3:
            result["center_relative_xyz_m"] = _world_point_to_root(
                center, root_xyz, root_orientation
            )
        direction = params.get("press_direction_world_xy")
        if isinstance(direction, list) and len(direction) == 2:
            result["press_direction_root_xy"] = _world_direction_to_root(
                direction, root_yaw
            )
        for key in ("depression_m", "plunger_stroke_m", "face_size_tangent_z_m"):
            if key in params:
                result[key] = copy.deepcopy(params[key])
    elif task_type == "flat_navigation":
        physical, source = _expert_navigation_physical_layout(
            course, episode_environment
        )
        result["vehicle_center_reference"] = "graph_t.root_module_pose"
        result["cone_positions_available"] = (
            isinstance(physical, Mapping)
            and isinstance(physical.get("cone_centers_xy_m"), list)
        )
        if isinstance(physical, Mapping):
            goal = physical.get("goal_xyyaw")
            if isinstance(goal, list) and len(goal) >= 3:
                result["goal_relative_xyyaw"] = _navigation_goal_relative(
                    goal,
                    floor_z=0.0,
                    root_xyz=root_xyz,
                    root_yaw=root_yaw,
                    root_orientation=root_orientation,
                )
                result["goal_source"] = source
            cones = physical.get("cone_centers_xy_m")
            if isinstance(cones, list):
                height = physical.get("cone_height_m")
                cone_height = float(height) if isinstance(height, (int, float)) else None
                result["cones_relative"] = _navigation_cones_relative(
                    cones,
                    floor_z=0.0,
                    cone_height_m=cone_height,
                    root_xyz=root_xyz,
                    root_orientation=root_orientation,
                )
                for key in ("cone_radius_m", "cone_height_m", "corridor_width_m", "vehicle_footprint"):
                    if key in physical:
                        result[key] = copy.deepcopy(physical[key])

    return result


def adapt_observation(
    record: Mapping[str, Any],
    canonical_graph: Mapping[str, Any],
    *,
    episode_environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build canonical global environment/task observation o_t."""

    # No root-relative task frame exists while topology/roles are changing.
    # Self-assembly and self-reconfiguration share the same transient graph
    # semantics; only the process label differs.
    transient_state = canonical_graph.get("morphology")
    if transient_state in {"assembling", "reconfiguring"}:
        target_graph = record.get("target_graph")
        target_morphology = None

        if isinstance(target_graph, Mapping):
            target_globals = target_graph.get("global_attributes", {})
            if isinstance(target_globals, Mapping):
                target_morphology = target_globals.get("morphology_name")

        process = (
            "assembly" if transient_state == "assembling"
            else "reconfiguration"
        )

        course = _course_from_record(record)
        expert_task = _expert_course_task(course) if course is not None else None
        active_task = (
            {
                "task_id": course.get("course_profile"),
                "type": expert_task[0],
                "geometry": None,
                "source": "expert_course",
            }
            if expert_task is not None else None
        )
        result = {
            "schema_version": "mssr.canonical_observation.v2",
            "task_valid": False,
            "task_context_valid": active_task is not None or (
                isinstance(course, Mapping) and isinstance(course.get("mission"), Mapping)
            ),
            "morphology_stable": False,
            "reference_frame": None,
            "root_module_id": None,
            "active_task": active_task,
            "structural_transition": {
                "active": True,
                "kind": process,
                "target_morphology": target_morphology,
            },
        }
        mission = course.get("mission") if isinstance(course, Mapping) else None
        if isinstance(mission, Mapping):
            explicit_id = _explicit_task_id(record, course, mission)
            task = next((t for t in mission.get("tasks", [])
                         if t.get("task_id") == explicit_id), None) if explicit_id else None
            if task is not None:
                result["active_task"] = {"task_id": task["task_id"], "type": task.get("type"),
                                         "geometry": None}
        # Named alias kept for readability/provenance in downstream audits.
        result[process] = copy.deepcopy(result["structural_transition"])
        return result


    course = _course_from_record(record)
    if course is None:
        return {
            "schema_version": "mssr.canonical_observation.v2",
            "task_valid": False,
            "task_context_valid": False,
            "active_task": None,
            "missing_reason": "course_unavailable",
        }

    root_id, root_xyz, root_yaw, root_orientation = _root_pose(canonical_graph)
    mission = course.get("mission")
    if not isinstance(mission, Mapping):
        expert_task = _expert_course_task(course)
        if expert_task is None:
            return {
                "schema_version": "mssr.canonical_observation.v2",
                "task_valid": False,
                "task_context_valid": bool(course),
                "active_task": None,
                "unparsed_course_world": copy.deepcopy(dict(course)),
                "missing_reason": "unsupported_course_schema",
            }
        task_type, params = expert_task
        result = {
            "schema_version": "mssr.canonical_observation.v2",
            "task_valid": True,
            "task_context_valid": True,
            "reference_frame": "structural_root",
            "transform_model": "full_se3_points_planar_heading",
            "root_module_id": root_id,
            "active_task": {
                "task_id": course.get("course_profile"),
                "type": task_type,
                "geometry": _expert_course_geometry(
                    task_type, params, course,
                    episode_environment=episode_environment,
                    root_xyz=root_xyz, root_yaw=root_yaw,
                    root_orientation=root_orientation,
                ),
                "source": "expert_course",
            },
        }
        source_observation = record.get("observation")
        if isinstance(source_observation, Mapping):
            task_context = source_observation.get("task_context")
            if isinstance(task_context, Mapping):
                result["source_task_context"] = copy.deepcopy(dict(task_context))
        return result

    explicit_id = _explicit_task_id(record, course, mission)
    task = None
    selection_source = "mission_exit_plane"
    if explicit_id is not None:
        for candidate in mission.get("tasks", []):
            if isinstance(candidate, Mapping) and candidate.get("task_id") == explicit_id:
                task = candidate
                selection_source = "mission_explicit"
                break
        if task is None:
            return {
                "schema_version": "mssr.canonical_observation.v2",
                "task_valid": False,
                "task_context_valid": True,
                "root_module_id": root_id,
                "active_task": None,
                "explicit_task_id": explicit_id,
                "available_task_ids": [
                    candidate.get("task_id")
                    for candidate in mission.get("tasks", [])
                    if isinstance(candidate, Mapping)
                ],
                "missing_reason": "explicit_task_id_not_in_mission",
            }
    if task is None:
        physical_tasks = [
            candidate for candidate in mission.get("tasks", [])
            if isinstance(candidate, Mapping) and candidate.get("type") != "goal"
        ]
        missing_exit = any(
            not isinstance(candidate.get("parameters", {}).get("exit_pose_xyyaw"), list)
            for candidate in physical_tasks
        )
        if missing_exit:
            morphology_types = {
                "snake8": {"gap", "stairs"},
                "rc_car8": {"flat_navigation"},
                "mobile_manipulator8": {"button"},
            }.get(canonical_graph.get("morphology"), set())
            compatible = [
                candidate for candidate in physical_tasks
                if candidate.get("type") in morphology_types
            ]
            if len(compatible) == 1:
                task = compatible[0]
                selection_source = "mission_unique_morphology"
            else:
                return {
                    "schema_version": "mssr.canonical_observation.v2",
                    "task_valid": False,
                    "task_context_valid": bool(physical_tasks),
                    "root_module_id": root_id,
                    "active_task": None,
                    "available_task_ids": [candidate.get("task_id") for candidate in physical_tasks],
                    "missing_reason": "task_selection_unresolved",
                }
        else:
            task = _select_active_upcoming_task(mission, root_xyz)

    if task is None:
        return {
            "schema_version": "mssr.canonical_observation.v2",
            "task_valid": False,
            "task_context_valid": False,
            "root_module_id": root_id,
            "active_task": None,
        }

    params = task.get("parameters", {})
    entry = params.get("entry_pose_xyyaw")
    exit_pose = params.get("exit_pose_xyyaw")

    relative_entry = None
    relative_entry_yaw = None

    if isinstance(entry, list) and len(entry) >= 3:
        floor_z = float(params.get("floor_height_m", root_xyz[2]))

        relative_entry = _world_point_to_root(
            [float(entry[0]), float(entry[1]), floor_z],
            root_xyz,
            root_orientation,
        )
        relative_entry_yaw = _wrap_angle(
            float(entry[2]) - root_yaw
        )

    relative_exit = None
    if isinstance(exit_pose, list) and len(exit_pose) >= 2:
        floor_z = float(params.get("floor_height_m", root_xyz[2]))

        relative_exit = _world_point_to_root(
            [
                float(exit_pose[0]),
                float(exit_pose[1]),
                floor_z,
            ],
            root_xyz,
            root_orientation,
        )

    return {
        "schema_version": "mssr.canonical_observation.v2",
        "task_valid": True,
        "task_context_valid": True,
        "reference_frame": "structural_root",
        "transform_model": "full_se3_points_planar_heading",
        "root_module_id": root_id,
        "active_task": {
            "source": selection_source,
            "task_id": task.get("task_id"),
            "type": task.get("type"),
            "entry_relative_xyz_m": relative_entry,
            "entry_relative_yaw_rad": relative_entry_yaw,
            "exit_relative_xyz_m": relative_exit,
            "geometry": _canonical_task_geometry(
                task,
                course=course,
                root_xyz=root_xyz,
                root_yaw=root_yaw,
                root_orientation=root_orientation,
            ),
        },
    }


# ---------------------------------------------------------------------------
# Canonical multi-agent action a_t
# ---------------------------------------------------------------------------

import sys


JOINT_PRIMITIVES = frozenset({
    "set_pan",
    "set_tilt",
    "rotate_pan_by",
    "rotate_tilt_by",
})

STRUCTURAL_PRIMITIVES = frozenset({
    "drive_to_pose",
    "align_faces",
    "assisted_align_faces",
    "dock",
    "undock",
    "gravity_settle",
    "reset_free_modules",
})


def _load_smores_runtime(repo_root: Path):
    """Reuse the exact SMORES-EP runtime geometry/conventions."""

    src = repo_root / "scripts/smores_ep/src"
    src_str = str(src)

    if src_str not in sys.path:
        sys.path.insert(0, src_str)

    from smores_ep.config.geometry import SmoresGeometry
    from smores_ep.control.differential_drive import twist_to_wheel_rates

    return SmoresGeometry(), twist_to_wheel_rates


def _joint_primitive_target(
    primitive_name: str,
    parameters: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Physical target implied by an elementary joint primitive.

    This is auxiliary information only.  The BC target remains the primitive.
    """

    table = {
        "set_pan": ("PAN", "position_absolute", "angle_rad"),
        "set_tilt": ("TILT", "position_absolute", "angle_rad"),
        "rotate_pan_by": ("PAN", "position_delta", "delta_rad"),
        "rotate_tilt_by": ("TILT", "position_delta", "delta_rad"),
    }

    spec = table.get(primitive_name)
    if spec is None:
        return None

    resource, mode, parameter_name = spec

    if parameter_name not in parameters:
        return None

    return {
        "resource": resource,
        "mode": mode,
        "value_rad": float(parameters[parameter_name]),
    }


def _primitive_identity(goal: Mapping[str, Any]) -> str:
    goal_id = goal.get("goal_id")

    if isinstance(goal_id, str) and goal_id:
        return goal_id

    return json.dumps(
        {
            "primitive": goal.get("primitive"),
            "module_ids": goal.get("module_ids"),
            "parameters": goal.get("parameters"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


class ActionDecisionTracker:
    """Recognize dispatch events independently for each recorded goal ID."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._seen: set[str] = set()

    def primitive_is_start(self, goal: Mapping[str, Any]) -> bool:
        identity = _primitive_identity(goal)
        if identity in self._seen:
            return False
        self._seen.add(identity)
        return True


class AlignmentOperationTracker:
    """Track an alignment intent across recorded phases, scoped to its producer.

    Only a start is a policy target. Phase changes and retries remain execution
    evidence. A dock request ends this alignment interval; docking is a separate
    action. State before dispatch never contains the action being predicted.
    """

    def __init__(self):
        self.active = {}
        self.events = []
        self.serial = 0

    @staticmethod
    def scope(record):
        return str(record.get('_loader_source_path') or record.get('_loader_phase') or
                   record.get('_loader_source_kind') or 'episode')

    @staticmethod
    def operation_base(goal):
        import re
        identity = str(goal.get('goal_id') or '')
        match = re.match(r'^(.*-w\d+-a\d+)-(?:reach|align|approach|clocking|retreat|dock)(?:-r\d+)?$', identity)
        return match.group(1) if match else None

    @staticmethod
    def signature(goal):
        params = goal['parameters']
        return (tuple(goal['module_ids']), params.get('face_a'), params.get('face_b'),
                params.get('clocking_quarter_turns', 0))

    def snapshot(self):
        return {'active_alignments': [copy.deepcopy(item['state']) for item in self.active.values()]}

    def begin_record(self):
        self.events = []

    def process(self, goal, record, is_new):
        if not is_new:
            return None
        scope = self.scope(record)
        signature = self.signature(goal)
        key = (scope, signature)
        phase = str(goal['parameters'].get('execution_phase', 'full')).lower()
        base = self.operation_base(goal)
        existing = self.active.get(key)
        # Known executor IDs retain one operation through r1/r2 recovery goals.
        # A new wave/stage is always a distinct operation, even for the same faces.
        continues = existing is not None and (
            (base is not None and base == existing['base']) or
            (base is None and existing['base'] is None and phase in {'align','approach','clocking','retreat'})
        )
        if not continues:
            if existing is not None:
                self.events.append({'operation_id': existing['id'], 'event': 'superseded'})
            self.serial += 1
            existing = {'id': f'alignment-{self.serial:05d}', 'base': base,
                        'state': {'module_ids': list(goal['module_ids']),
                                  'face_a': signature[1], 'face_b': signature[2],
                                  'clocking_quarter_turns': signature[3]}}
            self.active[key] = existing
        self.events.append({'operation_id': existing['id'],
                            'event': 'phase' if continues else 'start', 'phase': phase,
                            'source_goal_id': goal.get('goal_id'),
                            'parameters': copy.deepcopy(dict(goal['parameters']))})
        if continues:
            return None
        command = _primitive_command(goal)
        command['parameters'].pop('execution_phase', None)
        return command

    def finish_for_dock(self, goal, record):
        key = (self.scope(record), self.signature(goal))
        item = self.active.pop(key, None)
        if item is not None:
            self.events.append({'operation_id': item['id'], 'event': 'handoff_to_dock',
                                'source_goal_id': goal.get('goal_id')})

    def finish_scope(self, record):
        scope = self.scope(record)
        for key in list(self.active):
            if key[0] == scope:
                item = self.active.pop(key)
                self.events.append({'operation_id': item['id'], 'event': 'source_phase_end',
                                    'source_success': record.get('success')})


def command_resource_claims(command):
    """Match actuator ownership; shared anchors may be read by multiple movers."""
    name = command['command']; modules = command['module_ids']; params = command['parameters']
    claims = {}
    def claim(module, resource, mode='exclusive'):
        claims[(module, resource)] = mode
    if name == 'wheel_velocity' or name == 'drive_to_pose':
        claim(modules[0], 'wheels')
    elif name in JOINT_PRIMITIVES or name in {'pan_velocity', 'tilt_velocity'}:
        claim(modules[0], 'internal')
        for module in params.get('stabilize_during_group_module_ids', []):
            claim(module, 'internal', 'shared')
        if params.get('pusher_module_id'):
            claim(params['pusher_module_id'], 'wheels'); claim(params['pusher_module_id'], 'internal')
    elif name in {'align_faces', 'assisted_align_faces', 'dock', 'undock'}:
        a,b = modules[:2]
        claim(a, 'face:'+str(params['face_a'])); claim(b, 'face:'+str(params['face_b']))
        if name != 'undock':
            claim(a, 'wheels'); claim(b, 'wheels', 'shared')
        if name == 'align_faces':
            if params['face_a'] == 'TOP': claim(a, 'internal')
            if params['face_b'] == 'TOP': claim(b, 'internal')
        if name == 'assisted_align_faces': claim(modules[2], 'wheels')
    elif name in {'gravity_settle', 'reset_free_modules'}:
        for module in params.get('passive_module_ids', modules):
            claim(module, 'wheels'); claim(module, 'internal')
    else:
        raise PreprocessingError(f'No resource model for command {name!r}.')
    return claims


def _compatible_parallel_commands(existing, added):
    claims = {}
    for command in existing:
        for key,mode in command_resource_claims(command).items():
            claims.setdefault(key, []).append(mode)
    for command in added:
        if command in existing:
            continue
        for key,mode in command_resource_claims(command).items():
            if key in claims and (mode == 'exclusive' or 'exclusive' in claims[key]):
                return False
            claims.setdefault(key, []).append(mode)
    return True


def _same_decision_state(a, b):
    if a['action_t']['schedule'] != 'per_step' or b['action_t']['schedule'] != 'per_step':
        return False
    pa,pb = a['provenance'], b['provenance']
    if any(pa.get(key) != pb.get(key) for key in
           ('decision_stamp','loader_source_path','loader_phase','loader_source_kind')):
        return False
    for graph_key in ('graph_t', 'graph_t_plus_1'):
        ga = {k:v for k,v in a[graph_key].items() if k != 'control_state'}
        gb = {k:v for k,v in b[graph_key].items() if k != 'control_state'}
        if ga != gb:
            return False
    return a['observation_t'] == b['observation_t']


def resolve_continuous_revisions(transitions):
    """Supervise the last continuous setpoint for one unchanged source input.

    A controller can revise velocities while its cached physical graph remains
    unchanged. Keep all commands and recorded endpoints as execution evidence,
    but mask earlier differing velocity targets for the same actuator resource.
    Discrete primitive starts remain targets. Resolve before coalescing so idle
    or superseded source rows never inflate another command's BC sample weight.
    """
    continuous = {'wheel_velocity', 'pan_velocity', 'tilt_velocity'}

    def resource(command):
        return (tuple(command['module_ids']),
                'wheels' if command['command'] == 'wheel_velocity' else 'internal')

    def finish_window(window):
        groups = []
        for row in window:
            if row['action_t']['schedule'] != 'per_step':
                continue
            provenance = row['provenance']
            scope = tuple(provenance.get(k) for k in
                          ('loader_source_path', 'loader_phase', 'loader_source_kind'))
            group = next((items for key, representative, items in groups
                          if key == scope and representative['graph_t'] == row['graph_t']
                          and representative['observation_t'] == row['observation_t']), None)
            if group is None:
                group = []
                groups.append((scope, row, group))
            group.append(row)
        for _, _, group in groups:
            latest = {}
            for row in group:
                for command, mask in zip(row['action_t']['commands'], row['supervision']['command_mask']):
                    if mask and command['command'] in continuous:
                        latest[resource(command)] = command
            for row in group:
                supervision = row['supervision']
                revised = False
                for index, command in enumerate(row['action_t']['commands']):
                    if (supervision['command_mask'][index] and command['command'] in continuous
                            and command != latest[resource(command)]):
                        supervision['command_mask'][index] = False
                        revised = True
                if revised:
                    supervision['command_exclusion_reason'] = 'superseded_continuous_command'
                    supervision['valid_for_behavior_cloning'] = (
                        any(supervision['command_mask']) or supervision['configuration_mask'])
                    if not supervision['valid_for_behavior_cloning']:
                        supervision['sample_weight'] = 0
                        supervision['exclusion_reason'] = 'superseded_continuous_command'
        yield from window

    window = []
    stamp = None
    for transition in transitions:
        current_stamp = transition['provenance']['decision_stamp']
        if window and (current_stamp is None or current_stamp != stamp):
            yield from finish_window(window)
            window = []
        if current_stamp is None:
            yield transition
        else:
            stamp = current_stamp
            window.append(transition)
    if window:
        yield from finish_window(window)


def coalesce_parallel_transitions(transitions):
    """Fuse same-state dispatches only; preserve every distinct physical state.

    Identical recorded endpoints are required, preserving every distinct state
    and waiting interval. Blank same-clock snapshots remain in source references.
    Conflicting actuator revisions remain separate decisions. No action is moved
    to an older state and no endpoint beyond a recorded next graph is synthesized.
    """
    pending = None
    def refs(row):
        p = row['provenance']
        return p.get('source_records') or [{k:p[k] for k in
            ('loader_source_path','source_row','source_timestep','source_repeat_count') if k in p}]
    for transition in transitions:
        if pending is None:
            pending = transition
            continue
        if not (_same_decision_state(pending, transition) and
                _compatible_parallel_commands(pending['action_t']['commands'], transition['action_t']['commands'])):
            yield pending
            pending = transition
            continue
        p, q = pending['provenance'], transition['provenance']
        source_records = refs(pending) + refs(transition)
        p['source_records'] = source_records
        p['source_repeat_count'] += q['source_repeat_count']
        pending['supervision']['sample_weight'] += transition['supervision']['sample_weight']
        for command, mask in zip(transition['action_t']['commands'], transition['supervision']['command_mask']):
            if command in pending['action_t']['commands']:
                index = pending['action_t']['commands'].index(command)
                pending['supervision']['command_mask'][index] |= mask
            else:
                pending['action_t']['commands'].append(command)
                pending['supervision']['command_mask'].append(mask)
        if q.get('alignment_events'):
            p.setdefault('alignment_events', []).extend(q['alignment_events'])
        if 'control_state' in transition['graph_t_plus_1']:
            pending['graph_t_plus_1']['control_state'] = transition['graph_t_plus_1']['control_state']
        if q.get('source_stamp_span_end') is not None:
            p['source_stamp_span_end'] = q['source_stamp_span_end']
        if q.get('source_record_index_end') is not None:
            p['source_record_index_end'] = q['source_record_index_end']
        if transition.get('episode_state'):
            pending['episode_state'] = transition['episode_state']
        supervision = pending['supervision']
        supervision['valid_for_behavior_cloning'] = any(supervision['command_mask'])
        if supervision['valid_for_behavior_cloning']:
            supervision['exclusion_reason'] = None
    if pending is not None:
        yield pending


def _primitive_command(goal: Mapping[str, Any]) -> dict[str, Any]:
    name = goal.get("primitive")
    if name not in JOINT_PRIMITIVES | STRUCTURAL_PRIMITIVES:
        raise PreprocessingError(f"Unsupported command primitive: {name!r}.")
    modules = goal.get("module_ids")
    parameters = goal.get("parameters")
    if not isinstance(modules, list) or not modules or not all(isinstance(m, str) and m for m in modules):
        raise PreprocessingError(f"Primitive {name!r} has no valid module_ids.")
    if not isinstance(parameters, Mapping):
        raise PreprocessingError(f"Primitive {name!r} has no parameter mapping.")
    if name in JOINT_PRIMITIVES and _joint_primitive_target(name, parameters) is None:
        raise PreprocessingError(f"Joint primitive {name!r} has no target angle.")
    # Retain actual executor arguments: tolerances/assistance can vary per
    # dispatch. Remove only opaque grouping IDs, which are recorder metadata.
    params = {k: copy.deepcopy(v) for k, v in parameters.items()
              if k not in {"coordination_group", "coordination_size"}}
    return {"module_ids": list(modules), "command": name, "parameters": params}


def adapt_action(
    record: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
    decision_tracker: ActionDecisionTracker | None = None,
    alignment_tracker: AlignmentOperationTracker | None = None,
) -> dict[str, Any]:
    """Actual per-module commands at this source decision, with no macro labels.

    A command in progress is not a new policy target. Explicit zero velocities
    remain commands. An old IK record has an ordered physical command sequence
    but no intermediate states; its schedule makes this limitation explicit.
    """
    raw = record.get("expert_action")
    if not isinstance(raw, Mapping):
        raise PreprocessingError("Record has no valid expert_action.")
    unknown = set(raw) - {"locomotion", "magnetic", "primitive_goal", "joint_configuration", "primitive_sequence"}
    if unknown:
        raise PreprocessingError(f"Unmapped action fields: {sorted(unknown)}.")
    commands: list[dict[str, Any]] = []
    locomotion = raw.get("locomotion") or {}
    if not isinstance(locomotion, Mapping):
        raise PreprocessingError("expert_action.locomotion must be a mapping.")
    for module_id, command in locomotion.items():
        if not isinstance(command, Mapping):
            raise PreprocessingError(f"Invalid command for {module_id!r}.")
        unknown = set(command) - {"vx", "vy", "yaw_rate", "pan_rate_rad_s", "pan_rate",
                                  "tilt_rate_rad_s", "tilt_rate", "pan_target_rad", "tilt_target_rad"}
        if unknown:
            raise PreprocessingError(f"Unmapped direct command fields for {module_id}: {sorted(unknown)}.")
        if not all(math.isfinite(float(value)) for value in command.values()):
            raise PreprocessingError(f"Nonfinite direct command for {module_id}.")
        if any(k in command for k in ("vx", "vy", "yaw_rate")):
            if abs(float(command.get("vy", 0.))) > 1e-12:
                raise PreprocessingError("SMORES differential drive does not support vy != 0.")
            geometry, convert = _load_smores_runtime(repo_root or repository_root())
            rates = convert(float(command.get("vx", 0.)), float(command.get("yaw_rate", 0.)),
                            geometry.wheel_radius_m, geometry.track_width_m)
            commands.append({"module_ids": [str(module_id)], "command": "wheel_velocity",
                             "parameters": {"left_rad_s": float(rates.left_rad_s),
                                            "right_rad_s": float(rates.right_rad_s)}})
        internal = []
        for joint in ("pan", "tilt"):
            for keys, name, value_key in (
                ((f"{joint}_rate_rad_s", f"{joint}_rate"), f"{joint}_velocity", "rate_rad_s"),
                ((f"{joint}_target_rad",), f"set_{joint}", "angle_rad"),
            ):
                key = next((k for k in keys if k in command), None)
                if key is not None:
                    internal.append({"module_ids": [str(module_id)], "command": name,
                                     "parameters": {value_key: float(command[key])}})
        if len(internal) > 1:
            raise PreprocessingError("PAN/TILT share one internal resource: conflicting direct commands.")
        commands.extend(internal)

    goal = raw.get("primitive_goal")
    if isinstance(goal, Mapping):
        primitive = _primitive_command(goal)
        is_new = decision_tracker is None or decision_tracker.primitive_is_start(goal)
        if alignment_tracker is not None and goal.get("primitive") == "align_faces":
            command = alignment_tracker.process(goal, record, is_new)
            if command is not None:
                commands.append(command)
        elif is_new:
            if alignment_tracker is not None and goal.get("primitive") == "dock":
                alignment_tracker.finish_for_dock(goal, record)
            commands.append(primitive)

    if isinstance(raw.get("joint_configuration"), Mapping):
        sequence = raw.get("primitive_sequence")
        if not isinstance(sequence, list) or not sequence:
            raise PreprocessingError("IK configuration has no recorded physical primitive_sequence.")
        if commands:
            raise PreprocessingError("An ordered IK sequence cannot also contain step commands.")
        configuration = raw["joint_configuration"]
        targets = configuration.get("target_rad")
        if not isinstance(targets, Mapping) or not targets:
            raise PreprocessingError("IK configuration has no joint targets.")
        targets = {str(name): float(value) for name, value in targets.items()}
        if not all(math.isfinite(value) for value in targets.values()):
            raise PreprocessingError("IK configuration contains a nonfinite target.")
        return {"schedule": "recorded_sequence",
                "configuration_target_rad": targets,
                "commands": [_primitive_command(goal) for goal in sequence]}

    if raw.get("magnetic"):
        raise PreprocessingError("Unmapped magnetic command; explicit dock/undock primitives required.")
    # Check only explicitly commanded exclusive resources, never turn an
    # absent wheel channel into a zero command.
    occupied: set[tuple[str, str]] = set()
    for command in commands:
        name = command["command"]
        if name in JOINT_PRIMITIVES or name in {"pan_velocity", "tilt_velocity"}:
            resource = "internal"
        elif name == "wheel_velocity":
            resource = "wheels"
        else:
            continue
        for module in command["module_ids"]:
            key = (module, resource)
            if key in occupied:
                raise PreprocessingError(f"Conflicting commands for {module} resource {resource}.")
            occupied.add(key)
    return {"schedule": "per_step", "commands": commands}


def _pose_vector(value: Any, length: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise PreprocessingError(f"Invalid {label}.")
    result = [float(v) for v in value]
    if not all(math.isfinite(v) for v in result):
        raise PreprocessingError(f"Nonfinite {label}.")
    return result


def _relative_orientation(a: list[float], b: list[float]) -> list[float]:
    """Quaternion of B in A coordinates, xyzw; no planar approximation."""
    an = math.sqrt(sum(v*v for v in a))
    bn = math.sqrt(sum(v*v for v in b))
    if min(an, bn) < 1e-12:
        raise PreprocessingError("Zero quaternion in graph pose.")
    x, y, z, w = [-a[0]/an, -a[1]/an, -a[2]/an, a[3]/an]
    X, Y, Z, W = [v/bn for v in b]
    return [w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X,
            w*Z+x*Y-y*X+z*W, w*W-x*X-y*Y-z*Z]


def _box_relative_orientation(box: Mapping[str, Any], root: list[float]) -> list[float]:
    yaw = math.radians(float(box.get("yaw_deg", 0.))) / 2.
    pitch = math.radians(float(box.get("pitch_deg") or 0.)) / 2.
    world = [-math.sin(yaw)*math.sin(pitch), math.cos(yaw)*math.sin(pitch),
             math.sin(yaw)*math.cos(pitch), math.cos(yaw)*math.cos(pitch)]
    return _relative_orientation(root, world)


def compact_robot_graph(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Project robot state to one representation per dynamic quantity.

    Face world geometry is retained for docking decisions between currently
    disconnected modules. Connected partners are represented only by edges.
    Static robot specifications and simulator paths stay in the compact source.
    """
    nodes = []
    poses = {}
    for raw_node in graph.get("nodes", []):
        module_id = _module_id(raw_node)
        attrs = raw_node.get("attributes", {})
        pose = attrs.get("pose") or {}
        position = _pose_vector(attrs.get("position", pose.get("position")), 3, "position")
        orientation = _pose_vector(attrs.get("orientation", pose.get("orientation_xyzw", pose.get("orientation"))), 4, "orientation")
        if sum(v*v for v in orientation) < 1e-24:
            raise PreprocessingError("Zero quaternion in graph pose.")
        poses[module_id] = (position, orientation)
        role = attrs.get("structural_role_current")
        if role is None:
            role = attrs.get("current_role")
        if role in {"", "unassigned", "unknown"}:
            role = None
        faces = {}
        for face in attrs.get("connectors", []):
            name = face.get("connector_id") or face.get("face_name")
            state = face.get("status", "unknown")
            entry = {"state": state}
            for source, key in (("position_world", "position"),
                                ("outward_normal_world", "normal"), ("tangent_world", "tangent")):
                if face.get(source) is not None:
                    entry[key] = _pose_vector(face[source], 3, f"face {source}")
            faces[name] = entry
        actuators = {name: {key: copy.deepcopy(state[key])
                            for key in ("position_rad", "velocity_rad_s") if key in state}
                     for name, state in attrs.get("actuators", {}).items()}
        node = {"module_id": module_id, "position": position, "orientation": orientation,
                "role": role, "actuators": actuators, "faces": faces}
        for key in ("linear_velocity", "angular_velocity"):
            if attrs.get(key) is not None:
                node[key] = _pose_vector(attrs[key], 3, key)
        if attrs.get("control_available") is False:
            node["control_available"] = False
        if (attrs.get("simulation_fixtures") or {}).get("ground_support_anchor"):
            node["support_engaged"] = True
        nodes.append(node)
    edges = []
    for raw_edge in graph.get("edges", []):
        a, b = raw_edge.get("module_a_id"), raw_edge.get("module_b_id")
        if a not in poses or b not in poses:
            raise PreprocessingError("Physical graph edge references a missing module.")
        attrs = raw_edge.get("attributes", {})
        relation = attrs.get("relation_type") or ("current_connection" if attrs.get("is_attached") else "contact")
        if relation not in {"current_connection", "contact"}:
            raise PreprocessingError(f"Unexpected robot-state relation {relation!r}.")
        pa, qa = poses[a]
        pb, qb = poses[b]
        edge = {"module_a_id": a, "module_b_id": b,
                "face_a": attrs.get("face_a") or attrs.get("connector_a_id"),
                "face_b": attrs.get("face_b") or attrs.get("connector_b_id"),
                "relation": relation,
                "relative_pose": {"position": _world_point_to_root(pb, pa, qa),
                                  "orientation": _relative_orientation(qa, qb)}}
        if attrs.get("clocking_quarter_turns") is not None:
            edge["clocking_quarter_turns"] = attrs["clocking_quarter_turns"]
        edges.append(edge)
    by_module = {node['module_id']: node for node in nodes}
    # Docking edges are authoritative about occupied faces. Raw asynchronous
    # snapshots can update edges before node connector flags catch up.
    for edge in edges:
        if edge['relation'] == 'current_connection':
            for module, face in ((edge['module_a_id'], edge['face_a']),
                                 (edge['module_b_id'], edge['face_b'])):
                if face not in by_module[module]['faces']:
                    raise PreprocessingError(f'Attached edge references missing face {module}:{face}.')
                by_module[module]['faces'][face]['state'] = 'connected'
    return {"stamp": graph.get("stamp"), "morphology": graph.get("morphology"),
            "root_module_id": graph.get("root_module_id"), "nodes": nodes, "edges": edges}


def compact_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Dynamic task geometry; static world context remains in compact metadata."""
    task = observation.get("active_task") or {}
    geometry = task.get("geometry")
    relative = None
    if isinstance(geometry, Mapping):
        relative = {}
        for key, value in geometry.items():
            if value is not None and ("relative" in key or key.endswith("_root_xy") or key == "depression_m"):
                relative[key] = copy.deepcopy(value)
        if "cones_relative" in relative:
            relative["cones_relative"] = [{k: cone[k] for k in
                ("cone_index", "center_relative_xyz_m", "distance_xy_m") if k in cone}
                for cone in relative["cones_relative"]]
        if "collision_boxes_relative" in relative:
            relative["collision_boxes_relative"] = [{k: box[k] for k in
                ("name", "center_relative_xyz_m", "yaw_relative_rad", "orientation_relative_xyzw")
                if k in box and box[k] is not None} for box in relative["collision_boxes_relative"]]
        if task.get("type") != "flat_navigation":
            for key in ("entry_relative_xyz_m", "entry_relative_yaw_rad", "exit_relative_xyz_m"):
                if task.get(key) is not None:
                    relative[key] = copy.deepcopy(task[key])
    result = {"task_id": task.get("task_id"), "task_type": task.get("type"),
              "relative_geometry": relative}
    structural = observation.get("structural_transition")
    if isinstance(structural, Mapping):
        result["target_morphology"] = structural.get("target_morphology")
    if observation.get("missing_reason"):
        result["missing_reason"] = observation["missing_reason"]
    return result


class CanonicalTransitionBuilder:
    """Adapt a source decision with shared expert/teleop operation semantics.

    Primitive starts use their own recorded G_t, never an earlier wave anchor.
    G_t_plus_1 is the actual next recorded state, not an inferred terminal state.
    Old IK sequences keep their interval endpoints and configuration targets.
    Their commands cannot supervise instantaneous BC without intermediate states.
    """

    def __init__(self, adapter: CanonicalAdapter | None = None, *,
                 repo_root: Path | None = None,
                 episode_environment: Mapping[str, Any] | None = None) -> None:
        self.repo_root = repository_root() if repo_root is None else Path(repo_root)
        self.adapter = adapter or CanonicalAdapter(repo_root=self.repo_root)
        self.episode_environment = episode_environment
        self._tracker = ActionDecisionTracker()
        self._alignments = AlignmentOperationTracker()

    def reset(self) -> None:
        self._tracker.reset()
        self._alignments = AlignmentOperationTracker()

    @staticmethod
    def _stamp(record: Mapping[str, Any]) -> float | None:
        value = record.get("stamp")
        return float(value) if isinstance(value, (int, float)) else None

    def _canonical_graph(self, graph: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
        return self.adapter.adapt_graph(graph, morphology=morphology_from_record(record))

    def build_expert_step(self, record: Mapping[str, Any]) -> dict[str, Any]:
        # Kept as the public entry point for existing callers; applies to teleop too.
        next_graph = record.get("graph_t_plus_1")
        if not isinstance(next_graph, Mapping):
            raise PreprocessingError("Transition has no recorded graph_t_plus_1.")
        g_t = self._canonical_graph(record["graph_t"], record)
        next_state = self._canonical_graph(next_graph, record)
        observation = adapt_observation(record, g_t, episode_environment=self.episode_environment)
        self._alignments.begin_record()
        control_before = self._alignments.snapshot()
        action = adapt_action(record, repo_root=self.repo_root, decision_tracker=self._tracker,
                              alignment_tracker=self._alignments)
        if record.get("done"):
            self._alignments.finish_scope(record)
        control_after = self._alignments.snapshot()
        sequence = action["schedule"] == "recorded_sequence"
        start = self._stamp(record["graph_t"])
        end = self._stamp(next_graph)
        if start is not None and end is not None and end < start - 1e-9:
            raise PreprocessingError(f"Reversed state interval: {start} -> {end}.")
        source_supervision = record.get("supervision") or {}
        action_valid = bool(record.get("action_valid", True))
        source_valid = bool(source_supervision.get("valid_for_behavior_cloning"))
        selection_error = observation.get("missing_reason") in {
            "explicit_task_id_not_in_mission", "task_selection_unresolved"}
        command_mask = [action_valid and source_valid and not sequence and not selection_error
                        and command["command"] != "reset_free_modules"
                        for command in action["commands"]]
        configuration_mask = sequence and action_valid and source_valid and not selection_error
        reason = None
        if not action_valid:
            reason = "source_action_invalid"
        elif not source_valid:
            reason = "source_supervision_invalid"
        elif selection_error:
            reason = observation["missing_reason"]
        elif not any(command_mask) and not configuration_mask:
            reason = "no_policy_decision"
        provenance = {
            "decision_stamp": self._stamp(record),
            "source_stamp_start": start, "source_stamp_end": end,
            "duration_s": None if start is None or end is None else end-start,
            "transition_kind": "recorded_command_sequence" if sequence else "command_step",
            "temporal_model": "interval" if sequence else "step",
            "source_repeat_count": int(record.get("_source_repeat_count", 1)),
        }
        for source, dest in (("timestep", "source_timestep"),
                             ("_loader_phase", "loader_phase"),
                             ("_loader_source_kind", "loader_source_kind"),
                             ("_loader_source_path", "loader_source_path"),
                             ("_loader_source_row", "source_row"),
                             ("_source_record_index_start", "source_record_index_start"),
                             ("_source_record_index_end", "source_record_index_end"),
                             ("_source_stamp_start", "source_stamp_span_start"),
                             ("_source_stamp_end", "source_stamp_span_end")):
            if record.get(source) is not None:
                provenance[dest] = record[source]
        behavior = (record.get("debug") or {}).get("behavior")
        if behavior:
            provenance["behavior_context"] = behavior
        result = {
            "schema_version": "mssr.canonical_transition.v4",
            "graph_t": compact_robot_graph(g_t),
            "observation_t": compact_observation(observation),
            "action_t": action,
            "graph_t_plus_1": compact_robot_graph(next_state),
            "supervision": {"valid_for_behavior_cloning": any(command_mask) or configuration_mask,
                            "command_mask": command_mask,
                            "configuration_mask": configuration_mask,
                            "sample_weight": provenance["source_repeat_count"]
                            if any(command_mask) or configuration_mask else 0,
                            "exclusion_reason": reason},
            "provenance": provenance,
        }
        result["graph_t"]["control_state"] = control_before
        result["graph_t_plus_1"]["control_state"] = control_after
        if self._alignments.events:
            provenance["alignment_events"] = copy.deepcopy(self._alignments.events)
        if sequence:
            result["supervision"]["command_exclusion_reason"] = "intermediate_command_states_unavailable"
        terminal = {k: record[k] for k in ("done", "terminated", "truncated") if k in record}
        if terminal:
            result["episode_state"] = terminal
        return result

    def push(self, record: Mapping[str, Any], *, repo_root: Path | None = None) -> list[dict[str, Any]]:
        return [self.build_expert_step(record)]


# ---------------------------------------------------------------------------
# Canonical v5: physical-timestep joint actions + executor availability
# ---------------------------------------------------------------------------


def _physical_stamp(record: Mapping[str, Any]) -> float:
    graph = record.get("graph_t")
    if not isinstance(graph, Mapping):
        raise PreprocessingError("Record has no graph_t mapping.")
    value = graph.get("stamp")
    if not isinstance(value, (int, float)):
        raise PreprocessingError("graph_t has no numeric stamp.")
    return float(value)


def iter_physical_timestep_buckets(
    records: Iterable[Mapping[str, Any]],
) -> Iterator[list[Mapping[str, Any]]]:
    """Group source records by the physical state snapshot they observed.

    Multiple expert/FSM decisions can share the same cached ``graph_t``.  They
    must become one joint policy decision rather than conflicting BC examples.
    Source order inside a bucket is retained so the final continuous revision
    of a resource is the one exposed to BC.
    """

    bucket: list[Mapping[str, Any]] = []
    stamp: float | None = None
    for record in records:
        current = _physical_stamp(record)
        if stamp is not None and current < stamp - 1e-9:
            raise PreprocessingError(
                f"Physical graph time regressed from {stamp} to {current}."
            )
        if bucket and abs(current - stamp) > 1e-9:
            yield bucket
            bucket = []
        if not bucket:
            stamp = current
        bucket.append(record)
    if bucket:
        yield bucket


def _record_scope(record: Mapping[str, Any]) -> str:
    return str(
        record.get("_loader_source_path")
        or record.get("_loader_phase")
        or record.get("_loader_source_kind")
        or "episode"
    )


def _record_command_id(record: Mapping[str, Any]) -> str | None:
    debug = record.get("debug")
    value = debug.get("command_id") if isinstance(debug, Mapping) else None
    return str(value) if isinstance(value, str) and value else None


def _primitive_goal(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    expert = record.get("expert_action")
    if not isinstance(expert, Mapping):
        return None
    goal = expert.get("primitive_goal")
    return goal if isinstance(goal, Mapping) else None


def _source_bc_eligible(record: Mapping[str, Any]) -> bool:
    supervision = record.get("supervision")
    return bool(record.get("action_valid", True)) and bool(
        isinstance(supervision, Mapping)
        and supervision.get("valid_for_behavior_cloning") is True
    )


def _module_ids_from_graph(graph: Mapping[str, Any]) -> list[str]:
    return sorted(_module_id(node) for node in graph.get("nodes", []))


def _new_module_action() -> dict[str, Any]:
    return {
        "wheels": {
            "commanded": False,
            "left_velocity_rad_s": None,
            "right_velocity_rad_s": None,
        },
        "internal": {
            "commanded": False,
            "mode": None,
            "value": None,
            "unit": None,
        },
        "structural": {
            "commanded": False,
            "primitive": None,
            "parameters": None,
        },
        "participates_in": [],
    }


def _internal_action_from_command(command: Mapping[str, Any]) -> dict[str, Any]:
    name = str(command["command"])
    params = command["parameters"]
    if name == "pan_velocity":
        return {"commanded": True, "mode": "pan_velocity", "value": float(params["rate_rad_s"]), "unit": "rad/s"}
    if name == "tilt_velocity":
        return {"commanded": True, "mode": "tilt_velocity", "value": float(params["rate_rad_s"]), "unit": "rad/s"}
    if name == "set_pan":
        return {"commanded": True, "mode": "pan_target", "value": float(params["angle_rad"]), "unit": "rad"}
    if name == "set_tilt":
        return {"commanded": True, "mode": "tilt_target", "value": float(params["angle_rad"]), "unit": "rad"}
    if name == "rotate_pan_by":
        return {"commanded": True, "mode": "pan_delta", "value": float(params["delta_rad"]), "unit": "rad"}
    if name == "rotate_tilt_by":
        return {"commanded": True, "mode": "tilt_delta", "value": float(params["delta_rad"]), "unit": "rad"}
    raise PreprocessingError(f"Not an internal command: {name!r}.")


def _structural_action_from_command(command: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "commanded": True,
        "primitive": str(command["command"]),
        "parameters": copy.deepcopy(dict(command.get("parameters") or {})),
    }


def _command_primary_resource(command: Mapping[str, Any]) -> tuple[str, str]:
    name = str(command["command"])
    modules = command["module_ids"]
    if name == "wheel_velocity":
        return str(modules[0]), "wheels"
    if name in JOINT_PRIMITIVES or name in {"pan_velocity", "tilt_velocity"}:
        return str(modules[0]), "internal"
    return str(modules[0]), "structural"


class ExecutorAvailabilityTracker:
    """Infer executor-owned resource occupancy from recorded primitive lifecycle.

    Evidence is deliberately labelled.  ``active_goal_ids`` is used when it is
    present (assembly/reconfiguration streams), while coordinated joint programs
    can be closed by their recorded barrier/reached states.  Unknown evidence is
    never silently converted into a learned WAIT action.
    """

    def __init__(self) -> None:
        self.active: dict[str, dict[str, Any]] = {}
        self.alignments: dict[tuple[str, tuple[Any, ...]], str] = {}
        # Logical connector state inferred only from already-dispatched structural
        # actions.  The physical graph can lag one recorder frame behind an
        # accepted DOCK/UNDOCK; this override never looks at the action being
        # predicted for the current timestep.
        self.face_connection_overrides: dict[tuple[str, str], bool] = {}
        self.serial = 0
        self._expert_phase: str | None = None

    @staticmethod
    def _align_signature(goal: Mapping[str, Any]) -> tuple[Any, ...]:
        params = goal.get("parameters") or {}
        modules = tuple(str(x) for x in goal.get("module_ids") or [])
        return (
            modules,
            params.get("face_a"),
            params.get("face_b"),
            params.get("clocking_quarter_turns", 0),
        )

    def _operation_id(self, prefix: str) -> str:
        self.serial += 1
        return f"{prefix}-{self.serial:05d}"

    def _remove(self, operation_id: str) -> None:
        item = self.active.pop(operation_id, None)
        if item is None:
            return
        for key, value in list(self.alignments.items()):
            if value == operation_id:
                del self.alignments[key]

    def _remove_matching_alignment(self, goal: Mapping[str, Any], record: Mapping[str, Any]) -> None:
        key = (_record_scope(record), self._align_signature(goal))
        operation_id = self.alignments.get(key)
        if operation_id is not None:
            self._remove(operation_id)

    def prepare_bucket(self, records: Sequence[Mapping[str, Any]], stamp: float) -> None:
        # Expert compact phases are sequential (assembly -> behavior -> manipulation).
        # A phase change is therefore hard evidence that executor work owned by
        # the previous expert phase has completed, even when its final source row
        # forgot to set done/success.  Teleop has no _loader_phase and is not
        # affected: structural skills may legitimately span human-stream buckets.
        expert_phases = [
            str(record["_loader_phase"])
            for record in records
            if isinstance(record.get("_loader_phase"), str)
            and record.get("_loader_phase")
        ]
        if expert_phases:
            current_phase = expert_phases[-1]
            if self._expert_phase is None:
                self._expert_phase = current_phase
            elif current_phase != self._expert_phase:
                previous_phase = self._expert_phase
                for operation_id, item in list(self.active.items()):
                    if item.get("loader_phase") == previous_phase:
                        self._remove(operation_id)
                self._expert_phase = current_phase

        # Timed executor skills can be closed without an explicit terminal row.
        for operation_id, item in list(self.active.items()):
            expires = item.get("expires_at")
            if isinstance(expires, (int, float)) and stamp >= float(expires) - 1e-9:
                self._remove(operation_id)

        # A dock request is the explicit handoff out of ALIGN_FACES.
        for record in records:
            goal = _primitive_goal(record)
            if isinstance(goal, Mapping) and goal.get("primitive") == "dock":
                self._remove_matching_alignment(goal, record)

        # A newly accepted internal primitive implies any earlier primitive on
        # that module's coupled PAN/TILT resource has completed.
        incoming_internal: set[str] = set()
        for record in records:
            goal = _primitive_goal(record)
            if not isinstance(goal, Mapping) or goal.get("primitive") not in JOINT_PRIMITIVES:
                continue
            modules = goal.get("module_ids") or []
            if modules:
                incoming_internal.add(str(modules[0]))
        for operation_id, item in list(self.active.items()):
            if item.get("primary_resource") == "internal" and item.get("primary_module") in incoming_internal:
                self._remove(operation_id)

        # Where active_goal_ids exists it is direct runtime evidence.  Absence
        # of the field does not mean idle.
        explicit_by_scope: dict[str, set[str]] = {}
        scopes_with_explicit: set[str] = set()
        for record in records:
            debug = record.get("debug")
            if not isinstance(debug, Mapping) or "active_goal_ids" not in debug:
                continue
            scope = _record_scope(record)
            scopes_with_explicit.add(scope)
            values = debug.get("active_goal_ids")
            if isinstance(values, list):
                explicit_by_scope.setdefault(scope, set()).update(
                    str(value) for value in values if isinstance(value, str)
                )
        for operation_id, item in list(self.active.items()):
            scope = item.get("scope")
            if scope not in scopes_with_explicit or item.get("kind") == "align_faces":
                continue
            goal_ids = set(item.get("goal_ids") or [])
            if goal_ids and not (goal_ids & explicit_by_scope.get(str(scope), set())):
                self._remove(operation_id)

        # Coordinated morphology programs explicitly announce that a posture
        # barrier has been reached.  That closes outstanding joint primitives
        # for the same behavior command.
        completed_command_ids: set[str] = set()
        completed_joint_scopes: set[str] = set()
        completed_source_scopes: set[str] = set()
        for record in records:
            fsm = str(record.get("fsm_state") or "").upper()
            debug = record.get("debug")
            message = str(debug.get("message") or "").lower() if isinstance(debug, Mapping) else ""
            if "BARRIER" in fsm and "reached" in message:
                command_id = _record_command_id(record)
                if command_id:
                    completed_command_ids.add(command_id)
                completed_joint_scopes.add(_record_scope(record))
            if record.get("done") is True and record.get("success") is True:
                completed_source_scopes.add(_record_scope(record))

        # A posture barrier closes only coupled PAN/TILT work.
        for operation_id, item in list(self.active.items()):
            if item.get("kind") not in JOINT_PRIMITIVES:
                continue
            if (item.get("command_id") in completed_command_ids
                    or item.get("scope") in completed_joint_scopes):
                self._remove(operation_id)

        # Successful completion of a recorded source stream/phase is stronger:
        # no executor primitive from that scope may remain active in the next
        # phase.  This prevents final assembly DOCK/ALIGN operations from
        # leaking wheel/internal BUSY state into the subsequent behavior stream.
        for operation_id, item in list(self.active.items()):
            if item.get("scope") in completed_source_scopes:
                self._remove(operation_id)

    def observe_dispatches(self, records: Sequence[Mapping[str, Any]], stamp: float) -> None:
        for record in records:
            goal = _primitive_goal(record)
            if not isinstance(goal, Mapping):
                continue
            name = str(goal.get("primitive") or "")
            if name not in JOINT_PRIMITIVES | STRUCTURAL_PRIMITIVES:
                continue
            goal_id = str(goal.get("goal_id") or self._operation_id("goal"))
            scope = _record_scope(record)
            command_id = _record_command_id(record)
            command = _primitive_command(goal)

            if name == "align_faces":
                key = (scope, self._align_signature(goal))
                operation_id = self.alignments.get(key)
                if operation_id is None:
                    operation_id = self._operation_id("align")
                    claims = command_resource_claims(command)
                    self.active[operation_id] = {
                        "operation_id": operation_id,
                        "kind": "align_faces",
                        "scope": scope,
                        "loader_phase": record.get("_loader_phase"),
                        "loader_source_kind": record.get("_loader_source_kind"),
                        "command_id": command_id,
                        "goal_ids": set(),
                        "claims": claims,
                        "primary_module": str(command["module_ids"][0]),
                        "primary_resource": "structural",
                        "parameters": copy.deepcopy(command["parameters"]),
                        "started_at": stamp,
                        "evidence": "primitive_lifecycle",
                    }
                    self.alignments[key] = operation_id
                self.active[operation_id]["goal_ids"].add(goal_id)
                continue

            if goal_id in self.active:
                continue
            claims = command_resource_claims(command)
            primary_module, primary_resource = _command_primary_resource(command)
            item = {
                "operation_id": goal_id,
                "kind": name,
                "scope": scope,
                "loader_phase": record.get("_loader_phase"),
                "loader_source_kind": record.get("_loader_source_kind"),
                "command_id": command_id,
                "goal_ids": {goal_id},
                "claims": claims,
                "primary_module": primary_module,
                "primary_resource": primary_resource,
                "parameters": copy.deepcopy(command["parameters"]),
                "started_at": stamp,
                "evidence": "active_goal_ids_or_fsm" if name != "gravity_settle" else "recorded_duration",
            }
            if name == "gravity_settle":
                duration = command["parameters"].get("duration_s")
                if isinstance(duration, (int, float)):
                    item["expires_at"] = stamp + float(duration)
            self.active[goal_id] = item

            # The graph recorder can expose the old edge for one more frame
            # after an accepted UNDOCK (and analogously can lag a DOCK).  Record
            # the logical connector state only *after* the action at t has been
            # built, so G_t itself remains pre-action and the next timestep can
            # use the executor's already-known structural history without label
            # leakage from a_{t+1}.
            if name in {"dock", "undock"} and len(command["module_ids"]) >= 2:
                params = command["parameters"]
                face_a = params.get("face_a")
                face_b = params.get("face_b")
                if isinstance(face_a, str) and isinstance(face_b, str):
                    connected_after = name == "dock"
                    self.face_connection_overrides[(str(command["module_ids"][0]), face_a)] = connected_after
                    self.face_connection_overrides[(str(command["module_ids"][1]), face_b)] = connected_after

    def _busy_claims(self, module_id: str) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {"wheels": [], "internal": []}
        for item in self.active.values():
            for (module, resource), mode in item.get("claims", {}).items():
                if module != module_id or mode == "shared":
                    continue
                if resource in {"wheels", "internal"}:
                    result[resource].append(str(item["operation_id"]))
        return result

    def _face_busy(self, module_id: str, face: str) -> list[str]:
        result = []
        key = (module_id, f"face:{face}")
        for item in self.active.values():
            mode = item.get("claims", {}).get(key)
            if mode == "exclusive":
                result.append(str(item["operation_id"]))
        return result

    def decorate_graph(self, graph: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(graph)
        connected_faces: dict[str, set[str]] = {str(node["module_id"]): set() for node in result.get("nodes", [])}
        for edge in result.get("edges", []):
            if edge.get("relation") != "current_connection":
                continue
            connected_faces.setdefault(str(edge["module_a_id"]), set()).add(str(edge.get("face_a")))
            connected_faces.setdefault(str(edge["module_b_id"]), set()).add(str(edge.get("face_b")))

        # Drop an override as soon as the measured physical graph catches up.
        # Until then, primitive availability follows the executor-known logical
        # connector state while the physical face/edge data remains untouched.
        for key, expected_connected in list(self.face_connection_overrides.items()):
            module_id, face = key
            measured_connected = face in connected_faces.get(module_id, set())
            if measured_connected == expected_connected:
                del self.face_connection_overrides[key]

        effective_connected_faces: dict[str, set[str]] = {
            module_id: set(faces) for module_id, faces in connected_faces.items()
        }
        for (module_id, face), connected in self.face_connection_overrides.items():
            if connected:
                effective_connected_faces.setdefault(module_id, set()).add(face)
            else:
                effective_connected_faces.setdefault(module_id, set()).discard(face)

        for node in result.get("nodes", []):
            module_id = str(node["module_id"])
            busy = self._busy_claims(module_id)
            faces = {}
            for face in sorted((node.get("faces") or {}).keys()):
                face_busy = self._face_busy(module_id, face)
                physical_connected = face in connected_faces.get(module_id, set())
                connected = face in effective_connected_faces.get(module_id, set())
                override_active = (module_id, face) in self.face_connection_overrides
                faces[face] = {
                    "busy": bool(face_busy),
                    "busy_by": face_busy,
                    "physical_connected": physical_connected,
                    "connected_for_executor": connected,
                    "availability_source": (
                        "executor_structural_history" if override_active else "physical_graph"
                    ),
                    "align_faces_available": not connected and not face_busy,
                    "dock_available": not connected and not face_busy,
                    "undock_available": connected and not face_busy,
                }
            active = []
            for item in self.active.values():
                involved = any(key[0] == module_id for key in item.get("claims", {}))
                if involved:
                    active.append({
                        "operation_id": item["operation_id"],
                        "primitive": item["kind"],
                        "evidence": item["evidence"],
                        "started_at": item["started_at"],
                    })
            connected = bool(effective_connected_faces.get(module_id))
            internal_busy = bool(busy["internal"])
            node["control_state"] = {
                "wheels": {
                    "available": not bool(busy["wheels"]),
                    "busy_by": busy["wheels"],
                },
                "internal": {
                    "available": not internal_busy,
                    "busy_by": busy["internal"],
                    "mode": "primitive_busy" if internal_busy else (
                        "structural_hold" if connected else "passive"
                    ),
                    "mode_source": "primitive_lifecycle" if internal_busy else "executor_default_inferred",
                },
                "faces": faces,
                "gravity_settle_local_available": not bool(busy["wheels"] or busy["internal"]),
                "active_primitives": sorted(active, key=lambda x: x["operation_id"]),
            }
        return result


class CanonicalTimestepBuilder:
    """Build one readable multi-agent action for one physical graph snapshot."""

    def __init__(self, *, repo_root: Path | None = None,
                 episode_environment: Mapping[str, Any] | None = None) -> None:
        self.repo_root = repository_root() if repo_root is None else Path(repo_root)
        self.adapter = CanonicalAdapter(repo_root=self.repo_root)
        self.episode_environment = episode_environment
        self._decision_tracker = ActionDecisionTracker()
        self._alignment_tracker = AlignmentOperationTracker()
        self.executor_state = ExecutorAvailabilityTracker()

    def reset(self) -> None:
        self._decision_tracker.reset()
        self._alignment_tracker = AlignmentOperationTracker()
        self.executor_state = ExecutorAvailabilityTracker()

    @staticmethod
    def _bucket_morphology(records: Sequence[Mapping[str, Any]]) -> str:
        values = [morphology_from_record(record) for record in records]
        if "reconfiguring" in values:
            return "reconfiguring"
        if "assembling" in values:
            return "assembling"
        stable = [value for value in values if isinstance(value, str) and value]
        if not stable:
            raise PreprocessingError("Cannot determine morphology for physical timestep.")
        if len(set(stable)) != 1:
            raise PreprocessingError(f"Conflicting stable morphologies in one timestep: {sorted(set(stable))}.")
        return stable[-1]

    @staticmethod
    def _representative_record(records: Sequence[Mapping[str, Any]], morphology: str) -> Mapping[str, Any]:
        if morphology in {"assembling", "reconfiguring"}:
            for record in records:
                if morphology_from_record(record) == morphology:
                    return record
        # Prefer a human/behavior record for task geometry when the structure is stable.
        for record in records:
            if record.get("_loader_source_kind") == "human" or record.get("_loader_phase") == "behavior":
                return record
        return records[-1]

    def _aggregate_action(
        self,
        records: Sequence[Mapping[str, Any]],
        module_ids: Sequence[str],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        modules = {module_id: _new_module_action() for module_id in module_ids}
        masks = {module_id: {"wheels": False, "internal": False, "structural": False}
                 for module_id in module_ids}
        selected: dict[tuple[str, str], tuple[dict[str, Any], bool, str]] = {}
        conflicts: list[dict[str, Any]] = []
        unresolved_sequences: list[dict[str, Any]] = []

        for source_index, record in enumerate(records):
            adapted = adapt_action(
                record,
                repo_root=self.repo_root,
                decision_tracker=self._decision_tracker,
                alignment_tracker=self._alignment_tracker,
            )
            eligible = _source_bc_eligible(record)
            if adapted.get("schedule") == "recorded_sequence":
                unresolved_sequences.append({
                    "source_index": source_index,
                    "configuration_target_rad": copy.deepcopy(adapted.get("configuration_target_rad")),
                    "commands": copy.deepcopy(adapted.get("commands") or []),
                    "reason": "intermediate_states_not_recorded",
                })
                continue
            for command in adapted.get("commands", []):
                module_id, resource = _command_primary_resource(command)
                if module_id not in modules:
                    raise PreprocessingError(f"Action references module absent from G_t: {module_id}.")
                key = (module_id, resource)
                previous = selected.get(key)
                name = str(command["command"])
                if previous is not None:
                    previous_command, previous_eligible, previous_kind = previous
                    if command == previous_command:
                        selected[key] = (copy.deepcopy(command), previous_eligible or eligible, previous_kind)
                        continue
                    continuous = name in {"wheel_velocity", "pan_velocity", "tilt_velocity"}
                    previous_continuous = previous_kind in {"wheel_velocity", "pan_velocity", "tilt_velocity"}
                    if continuous and previous_continuous:
                        # Last continuous setpoint on the unchanged physical input wins.
                        if eligible or not previous_eligible:
                            selected[key] = (copy.deepcopy(command), eligible, name)
                        continue
                    conflicts.append({
                        "module_id": module_id,
                        "resource": resource,
                        "previous": copy.deepcopy(previous_command),
                        "new": copy.deepcopy(command),
                        "reason": "multiple_discrete_decisions_without_intermediate_state",
                    })
                    selected[key] = (copy.deepcopy(command), False, name)
                    continue
                selected[key] = (copy.deepcopy(command), eligible, name)

        # Fill the fixed per-module action objects.
        for (module_id, resource), (command, eligible, _) in selected.items():
            if resource == "wheels":
                params = command["parameters"]
                modules[module_id]["wheels"] = {
                    "commanded": True,
                    "left_velocity_rad_s": float(params["left_rad_s"]),
                    "right_velocity_rad_s": float(params["right_rad_s"]),
                }
            elif resource == "internal":
                modules[module_id]["internal"] = _internal_action_from_command(command)
            else:
                modules[module_id]["structural"] = _structural_action_from_command(command)
                if len(command.get("module_ids") or []) >= 2:
                    peer = str(command["module_ids"][1])
                    params = command.get("parameters") or {}
                    if peer in modules:
                        modules[peer]["participates_in"].append({
                            "primitive": command["command"],
                            "initiator_module_id": module_id,
                            "own_face": params.get("face_b"),
                            "initiator_face": params.get("face_a"),
                        })
                for passive in (command.get("parameters") or {}).get("passive_module_ids", []):
                    passive = str(passive)
                    if passive in modules and passive != module_id:
                        modules[passive]["participates_in"].append({
                            "primitive": command["command"],
                            "initiator_module_id": module_id,
                            "role": "passive_module",
                        })
            masks[module_id][resource] = bool(eligible)

        # Structural primitives own the actuator resources claimed by the executor.
        # Keep direct commands visible for provenance/readability but do not train a
        # second policy output on an actuator the selected structural skill owns.
        for (module_id, resource), (command, eligible, _) in list(selected.items()):
            if resource != "structural" or not eligible:
                continue
            for (claimed_module, claimed_resource), mode in command_resource_claims(command).items():
                if mode == "shared" or claimed_module not in masks:
                    continue
                if claimed_resource in {"wheels", "internal"}:
                    masks[claimed_module][claimed_resource] = False

        action = {
            "schema_version": "mssr.canonical_joint_action.v1",
            "modules": modules,
        }
        supervision = {
            "module_resource_masks": masks,
            "valid_for_behavior_cloning": any(
                value for module in masks.values() for value in module.values()
            ),
            "conflicts": conflicts,
        }
        audit = {"unresolved_recorded_sequences": unresolved_sequences}
        return action, supervision, audit

    def build_bucket(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not records:
            raise PreprocessingError("Cannot build an empty timestep bucket.")
        stamp = _physical_stamp(records[0])
        if any(abs(_physical_stamp(record) - stamp) > 1e-9 for record in records):
            raise PreprocessingError("Timestep bucket contains multiple physical stamps.")

        morphology = self._bucket_morphology(records)
        representative = self._representative_record(records, morphology)
        canonical_graph = self.adapter.adapt_graph(representative["graph_t"], morphology=morphology)
        compact_graph = compact_robot_graph(canonical_graph)

        # Reconcile completions before exposing availability at G_t.
        self.executor_state.prepare_bucket(records, stamp)
        graph_t = self.executor_state.decorate_graph(compact_graph)

        observation = compact_observation(adapt_observation(
            representative,
            canonical_graph,
            episode_environment=self.episode_environment,
        ))
        module_ids = _module_ids_from_graph(canonical_graph)
        action, supervision, audit = self._aggregate_action(records, module_ids)

        # Dispatches become busy only after the policy state/action at t.
        self.executor_state.observe_dispatches(records, stamp)

        source_records = []
        coverage = 0
        phase_done = False
        terminal = False
        for record in records:
            repeat = int(record.get("_source_repeat_count", 1))
            coverage = max(coverage, repeat)
            source_records.append({
                key: record[key]
                for key in (
                    "_loader_source_path", "_loader_source_row", "_loader_phase",
                    "_loader_source_kind", "timestep", "_source_timestep_start",
                    "_source_timestep_end", "_source_stamp_start", "_source_stamp_end",
                    "_source_repeat_count",
                )
                if record.get(key) is not None
            })
            phase_done |= bool(record.get("done"))
            terminal |= bool(record.get("is_terminal"))

        # Fallback endpoint is used only for the final logical timestep.  Normal
        # G_{t+1} comes from the next physical bucket's G_t.
        endpoint_record = max(
            records,
            key=lambda record: float((record.get("graph_t_plus_1") or {}).get("stamp", stamp)),
        )
        raw_next = endpoint_record.get("graph_t_plus_1")
        fallback_next = None
        if isinstance(raw_next, Mapping):
            fallback_canonical = self.adapter.adapt_graph(raw_next, morphology=morphology)
            fallback_next = self.executor_state.decorate_graph(compact_robot_graph(fallback_canonical))

        provenance = {
            "physical_stamp": stamp,
            "source_records": source_records,
            "source_repeat_coverage": coverage,
            "source_record_count": len(records),
            "timestep_model": "unique_physical_graph_snapshot",
        }
        if audit["unresolved_recorded_sequences"]:
            provenance.update(audit)
        behavior_labels = sorted({
            str((record.get("debug") or {}).get("behavior"))
            for record in records
            if isinstance(record.get("debug"), Mapping)
            and (record.get("debug") or {}).get("behavior")
        })
        if behavior_labels:
            provenance["behavior_context"] = behavior_labels

        return {
            "schema_version": "mssr.canonical_transition.v5",
            "graph_t": graph_t,
            "observation_t": observation,
            "action_t": action,
            "graph_t_plus_1": fallback_next,
            "supervision": supervision,
            "provenance": provenance,
            "episode_state": {
                "phase_done_observed": phase_done,
                "source_terminal_observed": terminal,
                "done": False,
            },
        }


# ---------------------------------------------------------------------------
# Episode temporal merging
# ---------------------------------------------------------------------------

def merge_episode_record_streams(
    streams: Sequence[
        tuple[
            str,
            Iterable[Mapping[str, Any]],
        ]
    ],
) -> Iterator[Mapping[str, Any]]:
    """Merge decoded episode streams into the real chronological sequence.

    Parameters
    ----------
    streams:
        Sequence of ``(kind, iterable)`` pairs.

        Supported kinds:
            - ``"human"``
            - ``"structural"``

    Structural records have priority when timestamps are exactly equal.  This
    lets a structural macro finish first and lets the human stream immediately
    afterwards expose the newly acquired stable morphology.

    Each individual stream must already be monotonic in ``stamp``.
    """

    import heapq

    def _priority(kind: str) -> int:
        if kind == "structural":
            return 0

        if kind == "human":
            return 1

        raise PreprocessingError(
            f"Unsupported episode stream kind: {kind!r}"
        )

    heap = []
    states = []

    for stream_id, (kind, records) in enumerate(streams):
        iterator = iter(records)

        try:
            row = next(iterator)
        except StopIteration:
            continue

        stamp = row.get("stamp")

        if not isinstance(stamp, (int, float)):
            raise PreprocessingError(
                "Episode record has no numeric stamp."
            )

        state = {
            "id": stream_id,
            "kind": kind,
            "iterator": iterator,
            "sequence": 0,
            "last_stamp": float(stamp),
            "row": row,
        }

        states.append(state)

        heapq.heappush(
            heap,
            (
                float(stamp),
                _priority(kind),
                stream_id,
                0,
                state,
            ),
        )

    last_global_stamp = None

    while heap:
        stamp, _, _, _, state = heapq.heappop(heap)

        if (
            last_global_stamp is not None
            and stamp < last_global_stamp
        ):
            raise PreprocessingError(
                "Merged episode stream is not temporally monotonic."
            )

        last_global_stamp = stamp

        yield state["row"]

        try:
            row = next(state["iterator"])
        except StopIteration:
            continue

        next_stamp = row.get("stamp")

        if not isinstance(next_stamp, (int, float)):
            raise PreprocessingError(
                "Episode record has no numeric stamp."
            )

        next_stamp = float(next_stamp)

        if next_stamp < state["last_stamp"]:
            raise PreprocessingError(
                "Individual episode stream is not temporally monotonic."
            )

        state["last_stamp"] = next_stamp
        state["sequence"] += 1
        state["row"] = row

        heapq.heappush(
            heap,
            (
                next_stamp,
                _priority(state["kind"]),
                state["id"],
                state["sequence"],
                state,
            ),
        )


# ---------------------------------------------------------------------------
# Episode-level dataset splitting
# ---------------------------------------------------------------------------


def split_episode_ids(
    episode_ids: Sequence[str],
    *,
    seed: int = 20260930,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, list[str]]:
    """Deterministically split whole episodes into train/validation/test.

    No episode may appear in more than one split.  This prevents temporal
    leakage between controller timesteps originating from the same
    demonstration.

    The caller must pass the complete episode inventory.  In the current
    dataset that means the 56 single-task expert episodes plus 13 teleop
    episodes; this helper deliberately does not hard-code that collection.
    """

    import random

    ids = [str(x) for x in episode_ids]

    if len(ids) != len(set(ids)):
        raise PreprocessingError(
            "Episode IDs must be unique before splitting."
        )

    if not ids:
        raise PreprocessingError(
            "Cannot split an empty episode collection."
        )

    if not (0.0 < train_fraction < 1.0):
        raise PreprocessingError(
            "train_fraction must be between 0 and 1."
        )

    if not (0.0 < validation_fraction < 1.0):
        raise PreprocessingError(
            "validation_fraction must be between 0 and 1."
        )

    if train_fraction + validation_fraction >= 1.0:
        raise PreprocessingError(
            "train + validation fractions must leave room for test."
        )

    shuffled = sorted(ids)

    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n = len(shuffled)

    n_train = round(n * train_fraction)
    n_validation = round(n * validation_fraction)

    # Keep all three partitions non-empty whenever the collection permits it.
    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_validation = max(
            1,
            min(n_validation, n - n_train - 1),
        )

    n_test = n - n_train - n_validation

    if n_test <= 0:
        raise PreprocessingError(
            "Episode split produced an empty test partition."
        )

    train = shuffled[:n_train]
    validation = shuffled[
        n_train:n_train + n_validation
    ]
    test = shuffled[
        n_train + n_validation:
    ]

    return {
        "train": train,
        "validation": validation,
        "test": test,
    }
