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
    CompositeNavigationCone,
    CompositeObstacleCourse,
    CourseBox,
    composite_obstacle_course,
)

from .course_geometry_audit import (
    box_polygon,
    overlap_area,
    validate_teleop_course,
)

PROFILE = "teleop_connected_v1"
# Compact C-like handoff geometry.  Keep only the free space that is
# operationally useful for turning/reconfiguration instead of inserting
# multi-metre dead zones between every seeded obstacle.
PAD_SIDE_M = 1.20
START_PAD_SIDE_M = 1.60
SHARED_JUNCTION_SIDE_M = 1.20
CONNECTOR_WIDTH_M = 1.20
CONNECTOR_LENGTH_M = 0.80
MIN_SNAKE_LANDING_M = 1.20
BUTTON_PLATFORM_LENGTH_M = 1.90
BUTTON_PLATFORM_WIDTH_M = 1.40


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




def _shared_edge_to_edge_shift_m(
    local_boxes,
    existing_boxes,
    origin,
    yaw,
    *,
    max_shift_m=1.50,
):
    """Minimum forward shift making a shared stage physically edge-to-edge."""

    existing_supports = [
        box
        for box in existing_boxes
        if box.collidable
        and box.semantic not in {
            "button",
            "button_support",
        }
    ]

    local_supports = [
        box
        for box in local_boxes
        if box.collidable
        and box.semantic not in {
            "button",
            "button_support",
        }
    ]

    if not existing_supports or not local_supports:
        return 0.0

    existing_polygons = [
        box_polygon(box)
        for box in existing_supports
    ]

    dx = math.cos(yaw)
    dy = math.sin(yaw)

    # Much stricter than the geometry audit's 1e-6 area threshold.
    overlap_tolerance_m2 = 1.0e-10

    def is_clear(shift_m):
        shifted_origin = (
            origin[0] + dx * shift_m,
            origin[1] + dy * shift_m,
            origin[2],
        )

        new_supports = [
            _world_box(
                box,
                shifted_origin,
                yaw,
            )
            for box in local_supports
        ]

        for new_box in new_supports:
            new_polygon = box_polygon(new_box)

            for old_polygon in existing_polygons:
                if (
                    overlap_area(
                        old_polygon,
                        new_polygon,
                    )
                    > overlap_tolerance_m2
                ):
                    return False

        return True

    if is_clear(0.0):
        return 0.0

    # Coarse scan finds the FIRST clear configuration.
    # 5 mm is small relative to the robot/course geometry.
    step_m = 0.005
    shift_m = step_m

    while shift_m <= max_shift_m + 1.0e-12:
        if is_clear(shift_m):
            low = shift_m - step_m
            high = shift_m

            # Refine to effectively exact physical contact.
            for _ in range(40):
                middle = 0.5 * (low + high)

                if is_clear(middle):
                    high = middle
                else:
                    low = middle

            return high

        shift_m += step_m

    raise RuntimeError(
        "Shared handoff needs more than "
        f"{max_shift_m:.2f} m to avoid physical overlap"
    )

def _sample_polyline_xyyaw(waypoints, fraction):
    """Return x, y and unit tangent at a fraction of a waypoint polyline."""
    points = [
        (float(p[0]), float(p[1]))
        for p in waypoints
    ]

    segments = []

    for a, b in zip(points, points[1:]):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = math.hypot(dx, dy)

        if length > 1.0e-9:
            segments.append(
                (
                    a,
                    b,
                    dx,
                    dy,
                    length,
                )
            )

    if not segments:
        raise ValueError(
            "RC waypoint polyline has no non-zero segment"
        )

    total = sum(s[4] for s in segments)
    target = max(
        0.0,
        min(1.0, float(fraction)),
    ) * total

    walked = 0.0

    for a, b, dx, dy, length in segments:
        if walked + length >= target:
            local = (target - walked) / length

            return (
                a[0] + local * dx,
                a[1] + local * dy,
                dx / length,
                dy / length,
            )

        walked += length

    a, b, dx, dy, length = segments[-1]

    return (
        b[0],
        b[1],
        dx / length,
        dy / length,
    )



def _zigzag_rc_cone_centers(
    waypoints,
    count,
):
    """Return a deterministic teleop-only alternating slalom.

    This does not alter the validated RC route or seed geometry.
    It only replaces cone placement for selected human demonstrations.
    """

    count = int(count)

    if not 4 <= count <= 10:
        raise ValueError(
            "zigzag cone_count must be between 4 and 10"
        )

    # Keep cones away from the immediate start/end handoffs while
    # distributing them uniformly enough that no two become a gate.
    first_fraction = 0.18
    last_fraction = 0.82

    fractions = [
        (
            first_fraction
            + (
                last_fraction
                - first_fraction
            )
            * index
            / (count - 1)
        )
        for index in range(count)
    ]

    # Deliberately close to the centreline:
    # enough intrusion to require steering around each obstacle,
    # but still comfortably inside the validated road width.
    lateral_pattern_m = (
        0.10,
        0.12,
        0.09,
        0.11,
    )

    centers = []

    for index, fraction in enumerate(fractions):
        x, y, tx, ty = _sample_polyline_xyyaw(
            waypoints,
            fraction,
        )

        nx = -ty
        ny = tx

        # Alternate at every obstacle.
        # First cone is on the RIGHT, then LEFT, then RIGHT...
        sign = -1.0 if index % 2 == 0 else +1.0

        lateral = (
            sign
            * lateral_pattern_m[
                index % len(lateral_pattern_m)
            ]
        )

        centers.append(
            (
                x + nx * lateral,
                y + ny * lateral,
            )
        )

    return centers

def _extra_rc_cone_centers(
    waypoints,
    existing_xy,
    count,
):
    """Place extra teleop-only cones without altering the validated route."""
    count = int(count)

    if count < 0 or count > 8:
        raise ValueError(
            "extra_cones must be between 0 and 8"
        )

    if count == 0:
        return []

    occupied = [
        (float(p[0]), float(p[1]))
        for p in existing_xy
    ]

    result = []

    # Spread requested cones through the useful middle 70% of the course.
    base_fractions = [
        0.15 + 0.70 * (i + 1) / (count + 1)
        for i in range(count)
    ]

    shifts = (
        0.00,
        +0.045,
        -0.045,
        +0.090,
        -0.090,
        +0.135,
        -0.135,
    )

    lateral_offsets = (
        0.24,
        0.20,
        0.28,
    )

    for index, base_fraction in enumerate(base_fractions):
        preferred_sign = 1.0 if index % 2 == 0 else -1.0
        selected = None

        for shift in shifts:
            fraction = max(
                0.12,
                min(
                    0.88,
                    base_fraction + shift,
                ),
            )

            x, y, tx, ty = _sample_polyline_xyyaw(
                waypoints,
                fraction,
            )

            nx = -ty
            ny = tx

            for sign in (
                preferred_sign,
                -preferred_sign,
            ):
                for lateral in lateral_offsets:
                    candidate = (
                        x + sign * lateral * nx,
                        y + sign * lateral * ny,
                    )

                    # Keep the new cone clearly distinct from every
                    # seeded/new cone already placed.
                    if all(
                        math.dist(candidate, other) >= 0.20
                        for other in occupied
                    ):
                        selected = candidate
                        break

                if selected is not None:
                    break

            if selected is not None:
                break

        if selected is None:
            raise RuntimeError(
                "Could not place requested extra RC cone "
                "without overlapping existing cones"
            )

        result.append(selected)
        occupied.append(selected)

    return result

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
    boxes.append(
        _support(
            "CompositeStartPlatform",
            cursor,
            (START_PAD_SIDE_M, START_PAD_SIDE_M),
            0,
            "composite_start_platform",
        )
    )

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
        mount_on_previous_support = bool(
            raw.get("mount_on_previous_support", False)
        )

        handoff_mode = str(
            raw.get("handoff_mode", "connector")
        )
        shared_shift_m = 0.0
        shared_junction = False

        if handoff_mode not in {
            "connector",
            "shared",
        }:
            raise ValueError(
                "handoff_mode must be 'connector' or 'shared'"
            )

        button_inset_m = float(
            raw.get("button_inset_m", 0.05)
        )

        if (
            not math.isfinite(button_inset_m)
            or not 0.0 <= button_inset_m <= 0.50
        ):
            raise ValueError(
                "button_inset_m must be between 0 and 0.50 m"
            )

        if mount_on_previous_support and kind != "button":
            raise ValueError(
                "mount_on_previous_support is valid only for button tasks"
            )
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
            platform = next(
                b
                for b in local_boxes
                if b.semantic == "button_test_platform"
            )

            if mount_on_previous_support:
                if (
                    index == 0
                    or not tasks
                    or tasks[-1]["type"] not in {"gap", "stairs"}
                ):
                    raise ValueError(
                        "Mounted button requires a preceding gap or stairs stage"
                    )

                # The landing already exists physically.  Keep only the
                # support + plunger fixture, not another button platform.
                local_boxes = [
                    b
                    for b in local_boxes
                    if b.semantic != "button_test_platform"
                ]

                local_start_x = None

            else:
                # Compact standalone terminal button stage.
                local_boxes = [
                    replace(
                        b,
                        size_xyz_m=(
                            BUTTON_PLATFORM_LENGTH_M,
                            BUTTON_PLATFORM_WIDTH_M,
                            b.size_xyz_m[2],
                        ),
                    )
                    if b.semantic == "button_test_platform"
                    else b
                    for b in local_boxes
                ]

                platform = next(
                    b
                    for b in local_boxes
                    if b.semantic == "button_test_platform"
                )

                local_start_x = (
                    platform.center_xyz_m[0]
                    - platform.size_xyz_m[0] / 2
                )

        else:
            local_start_x = -1.05

        if mount_on_previous_support:
            previous_type = tasks[-1]["type"]

            landing_semantic = {
                "gap": "gap_test_far_bank",
                "stairs": "stair_test_upper_deck",
            }.get(previous_type)

            if landing_semantic is None:
                raise ValueError(
                    "Mounted button requires a physical gap/stairs landing"
                )

            previous_support = next(
                (
                    b
                    for b in reversed(boxes)
                    if b.semantic == landing_semantic
                ),
                None,
            )

            if previous_support is None:
                raise RuntimeError(
                    "Previous landing support was not found"
                )

            fixture = local.buttons[0]

            # World press direction after rotating the seeded fixture.
            press_world = _point(
                fixture.press_direction_world_xy,
                (0, 0, 0),
                yaw,
            )

            nx = float(press_world[0])
            ny = float(press_world[1])

            # Project the previous support half-size onto the press
            # direction. Put the button near its edge, so the 0.65 m
            # MM8 approach point remains ON the existing landing.
            support_yaw = math.radians(
                previous_support.yaw_deg
            )

            ux = math.cos(support_yaw)
            uy = math.sin(support_yaw)

            vx = -uy
            vy = ux

            half_x = 0.5 * previous_support.size_xyz_m[0]
            half_y = 0.5 * previous_support.size_xyz_m[1]

            projected_half_extent = (
                abs(nx * ux + ny * uy) * half_x
                + abs(nx * vx + ny * vy) * half_y
            )

            edge_offset = max(
                0.0,
                projected_half_extent - button_inset_m,
            )

            target_button_xy = (
                previous_support.center_xyz_m[0]
                + nx * edge_offset,
                previous_support.center_xyz_m[1]
                + ny * edge_offset,
            )

            local_button_rotated = _point(
                fixture.center_xyz_m,
                (0, 0, 0),
                yaw,
            )

            origin = [
                target_button_xy[0]
                - local_button_rotated[0],
                target_button_xy[1]
                - local_button_rotated[1],
                cursor[2],
            ]

            # Button-only overlay:
            # the physical course already ended on this landing.
            #
            # Do NOT turn the button approach point into another course
            # segment. The audit checks the real 0.65 m approach
            # independently from the button centre/direction.
            entry = list(cursor)
            pad = list(cursor)

        elif index and handoff_mode == "shared":
            # V4 physical shared handoff:
            # no connector and no turn pad.
            #
            # First align the logical stage entry with the previous exit.
            # Then translate the complete new stage by the MINIMUM amount
            # required for its real support polygons to become edge-to-edge
            # with all support already present.
            entry = list(cursor)

            origin = _point(
                (-local_start_x, 0, 0),
                entry,
                yaw,
            )

            shared_shift_m = _shared_edge_to_edge_shift_m(
                local_boxes,
                boxes,
                origin,
                yaw,
            )

            if shared_shift_m > 0.65:
                raise RuntimeError(
                    f"Dense shared handoff {task_id} requires "
                    f"{shared_shift_m:.3f} m; "
                    "visual-review budget is 0.65 m"
                )

            forward_x = math.cos(yaw)
            forward_y = math.sin(yaw)

            origin = [
                origin[0] + forward_x * shared_shift_m,
                origin[1] + forward_y * shared_shift_m,
                origin[2],
            ]

            entry = [
                entry[0] + forward_x * shared_shift_m,
                entry[1] + forward_y * shared_shift_m,
                entry[2],
            ]

            heading_delta_rad = abs(
                _angle(yaw - heading)
            )

            # Parallel/aligned stages need no extra support.
            # A real heading discontinuity needs a compact common turning
            # area; otherwise two differently-oriented rectangles can meet
            # only at a point/wedge and cannot support the audited 0.60 m
            # traversable band.
            shared_junction = (
                heading_delta_rad
                > math.radians(5.0)
            )

            if shared_junction:
                junction_center = [
                    0.5 * (
                        cursor[0]
                        + entry[0]
                    ),
                    0.5 * (
                        cursor[1]
                        + entry[1]
                    ),
                    cursor[2],
                ]

                boxes.append(
                    _support(
                        f"{safe_id}_SharedJunction",
                        junction_center,
                        (
                            SHARED_JUNCTION_SIDE_M,
                            SHARED_JUNCTION_SIDE_M,
                        ),
                        yaw,
                        "teleop_turn_pad",
                    )
                )

            # No extra linear connector: the boundary is simply the
            # compact previous-exit -> new-entry transition.
            pad = list(entry)

        elif index:
            length = float(
                raw.get(
                    "connector_length_m",
                    CONNECTOR_LENGTH_M,
                )
            )

            if (
                not math.isfinite(length)
                or not 0.8 <= length <= 8.0
            ):
                raise ValueError(
                    "connector_length_m must be between 0.8 and 8 m"
                )

            pad = _point(
                (length, 0, 0),
                cursor,
                heading,
            )

            middle = _point(
                (length / 2, 0, 0),
                cursor,
                heading,
            )

            boxes.append(
                _support(
                    f"{safe_id}_Connector",
                    middle,
                    (
                        length + .04,
                        CONNECTOR_WIDTH_M,
                    ),
                    heading,
                    "teleop_connector",
                )
            )

            boxes.append(
                _support(
                    f"{safe_id}_TurnPad",
                    pad,
                    (
                        PAD_SIDE_M,
                        PAD_SIDE_M,
                    ),
                    yaw,
                    "teleop_turn_pad",
                )
            )

            # Dense concatenation:
            # the next seeded stage begins at the CENTER of the same
            # maneuver pad rather than after crossing the entire pad.
            entry = list(pad)

            origin = _point(
                (-local_start_x, 0, 0),
                entry,
                yaw,
            )

        else:
            pad = list(cursor)

            # Initial obstacle begins at the far side of the full-size
            # loose-module spawn platform.
            entry = _point(
                (
                    START_PAD_SIDE_M / 2 - .02,
                    0,
                    0,
                ),
                pad,
                yaw,
            )

            origin = _point(
                (-local_start_x, 0, 0),
                entry,
                yaw,
            )

        stage_boxes = [
            _world_box(b, origin, yaw)
            for b in local_boxes
        ]
        boxes.extend(stage_boxes)

        params = _parameters(
            local_task["parameters"],
            origin,
            yaw,
        )

        stage_cones = [
            replace(
                c,
                center_xyz_m=tuple(
                    _point(
                        c.center_xyz_m,
                        origin,
                        yaw,
                    )
                ),
            )
            for c in local.navigation_cones
        ]

        if kind == "flat_navigation":
            cone_pattern = raw.get("cone_pattern")

            if cone_pattern is not None:
                if cone_pattern != "zigzag":
                    raise ValueError(
                        "Unsupported teleop cone_pattern: "
                        f"{cone_pattern!r}"
                    )

                if not stage_cones:
                    raise RuntimeError(
                        "Cannot derive physical cone dimensions "
                        "for teleop zig-zag cones"
                    )

                cone_count = int(
                    raw.get("cone_count", 0)
                )

                zigzag_xy = _zigzag_rc_cone_centers(
                    params["waypoints_xyyaw"],
                    cone_count,
                )

                template_cone = stage_cones[0]

                stage_cones = [
                    CompositeNavigationCone(
                        task_id=task_id,
                        center_xyz_m=(
                            x,
                            y,
                            template_cone.center_xyz_m[2],
                        ),
                        radius_m=template_cone.radius_m,
                        height_m=template_cone.height_m,
                    )
                    for x, y in zigzag_xy
                ]

                params["cone_centers_xy_m"] = [
                    [x, y]
                    for x, y in zigzag_xy
                ]

                params["cone_count"] = len(
                    zigzag_xy
                )

                params["cone_pattern"] = "zigzag"

            extra_cone_count = int(
                raw.get("extra_cones", 0)
            )

            if cone_pattern is not None and extra_cone_count:
                raise ValueError(
                    "cone_pattern and extra_cones are mutually exclusive"
                )

            if extra_cone_count:
                existing_xy = [
                    c.center_xyz_m[:2]
                    for c in stage_cones
                ]

                extra_xy = _extra_rc_cone_centers(
                    params["waypoints_xyyaw"],
                    existing_xy,
                    extra_cone_count,
                )

                if not stage_cones:
                    raise RuntimeError(
                        "Cannot derive physical cone dimensions "
                        "for extra RC cones"
                    )

                template_cone = stage_cones[0]

                for x, y in extra_xy:
                    stage_cones.append(
                        CompositeNavigationCone(
                            task_id=task_id,
                            center_xyz_m=(
                                x,
                                y,
                                template_cone.center_xyz_m[2],
                            ),
                            radius_m=template_cone.radius_m,
                            height_m=template_cone.height_m,
                        )
                    )

                params.setdefault(
                    "cone_centers_xy_m",
                    [],
                ).extend(
                    [
                        [x, y]
                        for x, y in extra_xy
                    ]
                )

                params["cone_count"] = len(
                    params["cone_centers_xy_m"]
                )
                params["extra_cones"] = extra_cone_count

            exit_pose = params["waypoints_xyyaw"][-1]
            exit_point = exit_pose[:2] + [cursor[2]]
            exit_heading = exit_pose[2]
        elif kind == "button":
            if mount_on_previous_support:
                # Fixture lives on the already-existing landing.
                # No new stage length is introduced.
                exit_point = list(entry)
                exit_heading = heading
            else:
                exit_point = _point(
                    (
                        platform.center_xyz_m[0]
                        + platform.size_xyz_m[0] / 2,
                        0,
                        0,
                    ),
                    origin,
                    yaw,
                )
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
        buttons.extend(
            replace(
                b,
                center_xyz_m=tuple(
                    _point(b.center_xyz_m, origin, yaw)
                ),
                press_direction_world_xy=tuple(
                    _point(
                        b.press_direction_world_xy,
                        (0, 0, 0),
                        yaw,
                    )
                ),
                yaw_deg=b.yaw_deg + math.degrees(yaw),
            )
            for b in local.buttons
        )
        cones.extend(stage_cones)
        boundaries.append({"task_id": task_id, "entry_world_xyz_m": entry,
                           "exit_world_xyz_m": exit_point, "pad_center_world_xyz_m": pad,
                           "previous_exit_world_xyz_m": list(cursor) if index else None,
                           "connector_heading_rad": heading,
                           "yaw_deg": yaw_deg,
                           "handoff_mode": handoff_mode,
                           "shared_shift_m": shared_shift_m,
                           "shared_junction": shared_junction})
        cursor, heading = exit_point, exit_heading

    terminal_button_overlay = (
        raw_tasks[-1].get("type") == "button"
        and bool(
            raw_tasks[-1].get(
                "mount_on_previous_support",
                False,
            )
        )
    )

    if terminal_button_overlay:
        # The final landing itself is the goal support.
        goal = list(cursor)
    else:
        goal = _point(
            (
                PAD_SIDE_M / 2 - .02,
                0,
                0,
            ),
            cursor,
            heading,
        )

        boxes.append(
            _support(
                "CompositeGoalPlatform",
                goal,
                (
                    PAD_SIDE_M,
                    PAD_SIDE_M,
                ),
                heading,
                "goal_platform",
            )
        )
    tasks.append({"task_id": "goal", "type": "goal", "parameters": {"center_xyz_m": goal}})
    course = CompositeObstacleCourse(tuple(boxes), tuple(tasks), tuple(buttons), tuple(cones),
                                   cursor[2], tuple(goal), {
        "layout_profile": PROFILE, "episode_id": str(mission.get("episode_id", "")),
        "execution_mode": "teleoperation", "layout_version": 5,
        "maneuver_pad_side_m": PAD_SIDE_M,
        "start_pad_side_m": START_PAD_SIDE_M,
        "connector_length_m": CONNECTOR_LENGTH_M,
        "connector_width_m": CONNECTOR_WIDTH_M,
        "minimum_snake_landing_m": MIN_SNAKE_LANDING_M,
        "snake_design_length_m": .80, "ramps_allowed": False, "boundaries": boundaries,
        "validation_scope": "seeded fixture dimensions; physical rollout required",
    })

    digest = hashlib.sha256(json.dumps(course.to_observation(), sort_keys=True).encode()).hexdigest()
    return replace(course, layout_metadata={**course.layout_metadata, "geometry_sha256": digest})
