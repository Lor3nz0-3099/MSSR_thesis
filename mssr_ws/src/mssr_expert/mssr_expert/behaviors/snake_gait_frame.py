"""Private local planning view for the existing longitudinal Snake8 gaits.

Raw graphs and actuator commands remain in their original conventions. Only
the planner's position/collider view is local; stop goals carry their world axis.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import math
from typing import Mapping

from mssr_expert.primitives.common import module_position


@dataclass(frozen=True)
class GaitFrame:
    origin: tuple[float, float, float]
    yaw: float

    @classmethod
    def parse(cls, raw):
        if not isinstance(raw, Mapping):
            raise ValueError("Local gait landmarks require a stage_frame")
        try:
            origin = tuple(float(v) for v in raw["origin_world_xyz_m"])
            yaw = float(raw["yaw_rad"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid gait stage_frame") from error
        if len(origin) != 3 or not all(math.isfinite(v) for v in (*origin, yaw)):
            raise ValueError("Gait stage_frame must contain finite origin and yaw")
        return cls(origin, yaw)

    @property
    def axis(self):
        # Cardinal frames retain exact axes, avoiding accumulated trig noise.
        return tuple(0. if abs(v) < 1e-12 else v for v in (math.cos(self.yaw), math.sin(self.yaw)))

    def local_point(self, point):
        if len(point) != 3 or not all(math.isfinite(float(v)) for v in point):
            raise ValueError("Invalid world point for gait frame")
        x, y, z = (float(p)-o for p, o in zip(point, self.origin))
        c, s = self.axis
        return (round(c*x+s*y, 12), round(-s*x+c*y, 12), round(z, 12))

    def bind_program(self, program):
        result = []
        for step in program:
            if step.displacement_goal is not None:
                raise ValueError("Oriented Snake gait requires absolute longitudinal goals")
            if step.position_goal is not None:
                goal = replace(step.position_goal, origin_world_xy_m=self.origin[:2], axis_world_xy=self.axis)
                step = replace(step, position_goal=goal)
            result.append(step)
        return tuple(result)


def local_planning_graph(graph, parameters, kind):
    """Return (private projected graph, frame), or None for legacy world-X."""
    course = graph.global_attributes.get("course", {})
    if not isinstance(course, Mapping):
        return None  # the unchanged core planner reports the legacy error
    landmark = parameters.get(kind, course.get(kind))
    raw_frame = parameters.get("stage_frame", course.get("stage_frame"))
    if raw_frame is None:
        return None
    if not isinstance(landmark, Mapping) or landmark.get("coordinate_frame") != "stage_local":
        raise ValueError("stage_frame requires explicitly stage_local gait landmarks")
    frame = GaitFrame.parse(raw_frame)
    local_course = {"frame_id": "world", "planning_view": "private_stage_local",
                    kind: {**copy.deepcopy(landmark), "coordinate_frame": "world"}}
    # Never pass other stages' risers into the active staircase collision check.
    task_id = str(parameters.get("task_id", course.get("active_task_id", "")))
    prefix = "".join(c if c.isalnum() else "_" for c in task_id) + "_"
    if "collision_boxes" in course:
        local_boxes = []
        for original in course["collision_boxes"]:
            if task_id and "mission" in course and not str(original.get("name", "")).startswith(prefix):
                continue
            box = copy.deepcopy(original)
            box["center_xyz_m"] = frame.local_point(box["center_xyz_m"])
            yaw = math.radians(float(box.get("yaw_deg", 0.))) - frame.yaw
            box["yaw_deg"] = math.degrees(math.atan2(math.sin(yaw), math.cos(yaw)))
            if box.get("semantic") == "stair_test_riser" and abs(box["yaw_deg"]) > 1e-6:
                raise ValueError("Stair collider is not aligned with the gait stage_frame")
            local_boxes.append(box)
        local_course["collision_boxes"] = local_boxes

    nodes = []
    for node in graph.nodes:
        attrs = dict(node.attributes)
        position = frame.local_point(module_position(attrs))
        attrs["position"] = position
        if isinstance(attrs.get("pose"), Mapping):
            attrs["pose"] = {**attrs["pose"], "position": position}
        # The existing planners read only module positions and body-local
        # actuator/geometry data. This view must never be published or logged.
        nodes.append(replace(node, attributes=attrs))
    attrs = {**graph.global_attributes, "course": local_course}
    return replace(graph, nodes=tuple(nodes), global_attributes=attrs), frame


def resolve_gait_parameters(graph, assignments, parameters, kind):
    """Pin a composite obstacle once, before planning or recording a command.

    Automatic selection requires the whole snake near the approach lane. An
    ambiguous scene requires an explicit task_id; it never picks by list order.
    Legacy courses and explicitly supplied standalone geometry keep their API.
    """
    course = graph.global_attributes.get("course", {})
    mission = course.get("mission", {}) if isinstance(course, Mapping) else {}
    tasks = mission.get("tasks", []) if isinstance(mission, Mapping) else []
    if not tasks or not any("stage_frame" in t.get("parameters", {}) for t in tasks):
        return dict(parameters)
    requested = parameters.get("task_id")
    candidates = [t for t in tasks if t.get("type") == kind and
                  (not requested or t.get("task_id") == requested)]
    if requested and len(candidates) != 1:
        raise ValueError(f"Unknown or non-unique {kind} task_id: {requested}")
    if not requested:
        ids = {a.module_id for a in assignments}
        positions = [module_position(n.attributes) for n in graph.nodes if n.module_id in ids]
        if len(positions) != len(ids) or not positions:
            raise ValueError("Cannot select gait obstacle without all assigned module positions")
        nearby = []
        for task in candidates:
            p = task.get("parameters", {})
            if "stage_frame" not in p:
                continue
            frame = GaitFrame.parse(p["stage_frame"])
            points = [frame.local_point(pos) for pos in positions]
            landmark = p[kind]
            edge = float(landmark["near_edge_x_m" if kind == "gap" else "first_riser_x_m"])
            if (max(abs(pos[1]) for pos in points) <= .35 and
                    min(pos[0] for pos in points) >= edge-2.0 and
                    max(pos[0] for pos in points) <= edge+.20):
                nearby.append(task)
        candidates = nearby
        if len(candidates) != 1:
            raise ValueError(f"No unique nearby {kind} obstacle (missing or ambiguous); align at the approach or specify task_id")
    task = candidates[0]
    p = task["parameters"]
    # The selected mission owns geometry; tuning parameters remain caller-owned.
    return {**parameters, kind: copy.deepcopy(p[kind]),
            **({"stage_frame": copy.deepcopy(p["stage_frame"])} if "stage_frame" in p else {}),
            "task_id": task["task_id"]}
