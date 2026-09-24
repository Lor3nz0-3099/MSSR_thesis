"""Oriented, connected courses for manual control, built from seeded fixtures.

Legacy task experts assume world +X for stairs/gaps. This opt-in profile keeps
their scalar geometry in an explicitly local frame and is teleoperation-only.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
import math
from typing import Any, Mapping

from .obstacle_course import (
    CompositeObstacleCourse, CourseBox, composite_obstacle_course,
)

from .course_geometry_audit import validate_teleop_course

PROFILE = "teleop_connected_v1"
PAD_SIDE_M = 2.4
CONNECTOR_WIDTH_M = 1.4
CONNECTOR_LENGTH_M = 2.6
MIN_SNAKE_LANDING_M = 1.2


def _point(point, origin, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = point[:2]
    result = [origin[0] + c*x - s*y, origin[1] + s*x + c*y]
    if len(point) == 3:
        result.append(origin[2] + point[2])
    return result


def _angle(yaw):
    return math.atan2(math.sin(yaw), math.cos(yaw))


def _pose(pose, origin, yaw):
    return _point(pose[:2], origin, yaw) + [_angle(pose[2] + yaw)]


def _bounds(bounds, origin, yaw):
    corners = [_point((x, y), origin, yaw)
               for x in bounds[:2] for y in bounds[2:]]
    return [min(p[0] for p in corners), max(p[0] for p in corners),
            min(p[1] for p in corners), max(p[1] for p in corners)]


def _support(name, center, size, yaw, semantic):
    height = center[2]
    return CourseBox(name, (center[0], center[1], (height - .02)/2),
                     (size[0], size[1], height + .02), (.24, .27, .31),
                     semantic=semantic, yaw_deg=math.degrees(yaw))


def _world_box(box, origin, yaw):
    center = _point(box.center_xyz_m, origin, yaw)
    size = box.size_xyz_m
    # Supports extend down to the common bottom plane, including upper decks.
    if box.semantic not in {"button", "button_support"}:
        center[2] -= origin[2]/2
        size = (size[0], size[1], size[2] + origin[2])
    return replace(box, center_xyz_m=tuple(center), size_xyz_m=size,
                   yaw_deg=box.yaw_deg + math.degrees(yaw))


def _parameters(local, origin, yaw):
    p = copy.deepcopy(local)
    p["geometry_frame"] = "world"
    p["stage_frame"] = {"origin_world_xyz_m": list(origin), "yaw_rad": yaw}
    p["floor_height_m"] += origin[2]
    if "upper_deck_height_m" in p:
        p["upper_deck_height_m"] += origin[2]
    for key in ("waypoints_xyyaw",):
        if key in p:
            p[key] = [_pose(v, origin, yaw) for v in p[key]]
    if "cone_centers_xy_m" in p:
        p["cone_centers_xy_m"] = [_point(v, origin, yaw) for v in p["cone_centers_xy_m"]]
    for key in ("platform_bounds_xy_m", "start_pad_bounds_xy_m", "reconfiguration_pad_bounds_xy_m"):
        if key in p:
            p[key] = _bounds(p[key], origin, yaw)
    if "reconfiguration_pose_xyyaw" in p:
        p["reconfiguration_pose_xyyaw"] = _pose(p["reconfiguration_pose_xyyaw"], origin, yaw)
    if "gap" in p:
        gap = p["gap"]
        gap["coordinate_frame"] = "stage_local"
        for side in ("near", "far"):
            gap[f"{side}_edge_center_world_xyz_m"] = _point(
                (gap[f"{side}_edge_x_m"], 0, gap["bank_height_m"]), origin, yaw)
        gap["width_m"] = gap["far_edge_x_m"] - gap["near_edge_x_m"]
        gap["crossing_direction_world_xy"] = [math.cos(yaw), math.sin(yaw)]
    if "stairs" in p:
        stairs = p["stairs"]
        stairs["coordinate_frame"] = "stage_local"
        stairs["first_riser_center_world_xyz_m"] = _point(
            (stairs["first_riser_x_m"], 0, stairs["base_height_m"]), origin, yaw)
        stairs["ascent_direction_world_xy"] = [math.cos(yaw), math.sin(yaw)]
        stairs["top_heights_world_m"] = [v + origin[2] for v in stairs["top_heights_m"]]
    if "button" in p:
        button = p["button"]
        button["center_xyz_m"] = _point(button["center_xyz_m"], origin, yaw)
        button["press_direction_world_xy"] = _point(button["press_direction_world_xy"], (0, 0, 0), yaw)
    return p


def build_teleop_course(mission: Mapping[str, Any], validated_seeds) -> CompositeObstacleCourse:
    if mission.get("schema_version") != "mssr.composite_mission.v1":
        raise ValueError("Unsupported composite mission schema")
    raw_tasks = mission.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("Composite mission tasks must be a non-empty array")
    boxes, tasks, buttons, cones = [], [], [], []
    seen = set()
    cursor = [-1.05, 0., 0.]
    heading = 0.
    boundaries = []
    boxes.append(_support("CompositeStartPlatform", cursor, (PAD_SIDE_M, PAD_SIDE_M), 0, "composite_start_platform"))

    for index, raw in enumerate(raw_tasks):
        if not isinstance(raw, Mapping):
            raise ValueError("Composite task must be an object")
        task_id = str(raw.get("task_id", f"{raw.get('type')}-{index:02d}"))
        safe_id = "".join(c if c.isalnum() else "_" for c in task_id)
        if not task_id.strip() or safe_id in seen or task_id == "goal":
            raise ValueError("Invalid, duplicate, or colliding task_id")
        seen.add(safe_id)
        yaw_deg = float(raw.get("yaw_deg", 90 * round(math.degrees(heading)/90)))
        if not math.isfinite(yaw_deg) or yaw_deg not in (-180, -90, 0, 90, 180):
            raise ValueError("yaw_deg must be a finite cardinal orientation")
        yaw = math.radians(yaw_deg)
        local = composite_obstacle_course({
            "schema_version": "mssr.composite_mission.v1",
            "tasks": [{"task_id": task_id, "type": raw["type"], "seed": raw["seed"]}],
        }, validated_seeds)
        local_task = local.tasks[0]
        local_boxes = [b for b in local.boxes if b.name not in {"CompositeStartPlatform", "CompositeGoalPlatform"}]
        kind = raw["type"]
        # The entire 0.80 m Snake8 must clear the obstacle before a turn.
        # Reserve 1.20 m in the landing itself; connectors do not count.
        landing_semantic = {"gap": "gap_test_far_bank", "stairs": "stair_test_upper_deck"}.get(kind)
        if landing_semantic:
            widened = []
            for b in local_boxes:
                if b.semantic == landing_semantic:
                    extra = max(0., MIN_SNAKE_LANDING_M-b.size_xyz_m[0])
                    b = replace(b, center_xyz_m=(b.center_xyz_m[0]+extra/2, *b.center_xyz_m[1:]),
                                size_xyz_m=(max(MIN_SNAKE_LANDING_M, b.size_xyz_m[0]), *b.size_xyz_m[1:]))
                widened.append(b)
            local_boxes = widened
        if kind == "gap":
            local_start_x = -1.65  # includes the seeded near-bank maneuver pad
        elif kind == "button":
            # Enlarge only support, never button height or mechanism geometry.
            local_boxes = [replace(b, size_xyz_m=(3.0, PAD_SIDE_M, b.size_xyz_m[2]))
                           if b.semantic == "button_test_platform" else b for b in local_boxes]
            platform = next(b for b in local_boxes if b.semantic == "button_test_platform")
            local_start_x = platform.center_xyz_m[0] - platform.size_xyz_m[0]/2
        else:
            local_start_x = -1.05

        if index:
            length = float(raw.get("connector_length_m", CONNECTOR_LENGTH_M))
            if not math.isfinite(length) or not 2.6 <= length <= 8.0:
                raise ValueError("connector_length_m must be between 2.6 and 8 m")
            pad = _point((length, 0, 0), cursor, heading)
            middle = _point((length/2, 0, 0), cursor, heading)
            boxes.append(_support(f"{safe_id}_Connector", middle,
                                  (length + .04, CONNECTOR_WIDTH_M), heading, "teleop_connector"))
            boxes.append(_support(f"{safe_id}_TurnPad", pad,
                                  (PAD_SIDE_M, PAD_SIDE_M), yaw, "teleop_turn_pad"))
        else:
            pad = list(cursor)
        entry = _point((PAD_SIDE_M/2 - .02, 0, 0), pad, yaw)
        origin = _point((-local_start_x, 0, 0), entry, yaw)
        stage_boxes = [_world_box(b, origin, yaw) for b in local_boxes]
        boxes.extend(stage_boxes)
        params = _parameters(local_task["parameters"], origin, yaw)
        if kind == "flat_navigation":
            exit_pose = params["waypoints_xyyaw"][-1]
            exit_point = exit_pose[:2] + [cursor[2]]
            exit_heading = exit_pose[2]
        elif kind == "button":
            exit_point = _point((platform.center_xyz_m[0] + platform.size_xyz_m[0]/2, 0, 0), origin, yaw)
            exit_heading = yaw
        else:
            landing = next(b for b in local_boxes if b.semantic == landing_semantic)
            local_exit_x = landing.center_xyz_m[0] + landing.size_xyz_m[0]/2
            exit_point = _point((local_exit_x, 0, local.final_floor_height_m), origin, yaw)
            params["clear_landing_length_m"] = landing.size_xyz_m[0]
            exit_heading = yaw
        params["entry_pose_xyyaw"] = entry[:2] + [yaw]
        params["exit_pose_xyyaw"] = exit_point[:2] + [exit_heading]
        params["source_seed"] = int(raw["seed"])
        params["execution_mode"] = "teleoperation"
        tasks.append({**local_task, "parameters": params})
        buttons.extend(replace(b, center_xyz_m=tuple(_point(b.center_xyz_m, origin, yaw)),
                               press_direction_world_xy=tuple(_point(b.press_direction_world_xy, (0, 0, 0), yaw)))
                       for b in local.buttons)
        cones.extend(replace(c, center_xyz_m=tuple(_point(c.center_xyz_m, origin, yaw)))
                     for c in local.navigation_cones)
        boundaries.append({"task_id": task_id, "entry_world_xyz_m": entry,
                           "exit_world_xyz_m": exit_point, "pad_center_world_xyz_m": pad,
                           "previous_exit_world_xyz_m": list(cursor) if index else None,
                           "connector_heading_rad": heading, "yaw_deg": yaw_deg})
        cursor, heading = exit_point, exit_heading

    # Every ending has a full maneuver pad and a reachable terminal goal.
    goal = _point((PAD_SIDE_M/2 - .02, 0, 0), cursor, heading)
    boxes.append(_support("CompositeGoalPlatform", goal, (PAD_SIDE_M, PAD_SIDE_M), heading, "goal_platform"))
    tasks.append({"task_id": "goal", "type": "goal", "parameters": {"center_xyz_m": goal}})
    course = CompositeObstacleCourse(tuple(boxes), tuple(tasks), tuple(buttons), tuple(cones),
                                   cursor[2], tuple(goal), {
        "layout_profile": PROFILE, "episode_id": str(mission.get("episode_id", "")),
        "execution_mode": "teleoperation", "layout_version": 1,
        "maneuver_pad_side_m": PAD_SIDE_M, "minimum_snake_landing_m": MIN_SNAKE_LANDING_M,
        "snake_design_length_m": .80, "ramps_allowed": False, "boundaries": boundaries,
        "validation_scope": "seeded fixture dimensions; physical rollout required",
    })

    digest = hashlib.sha256(json.dumps(course.to_observation(), sort_keys=True).encode()).hexdigest()
    return replace(course, layout_metadata={**course.layout_metadata, "geometry_sha256": digest})
