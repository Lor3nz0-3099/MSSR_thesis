"""Dependency-free checks of support polygons for teleoperated courses.

Polygon intersections are exact up to floating point tolerance. Passage
clearance is sampled at 5 cm intervals; this is not a dynamics certificate.
"""
from __future__ import annotations

import math


def box_polygon(box):
    c, s = math.cos(math.radians(box.yaw_deg)), math.sin(math.radians(box.yaw_deg))
    hx, hy = box.size_xyz_m[0]/2, box.size_xyz_m[1]/2
    return [(box.center_xyz_m[0] + c*x - s*y, box.center_xyz_m[1] + s*x + c*y)
            for x, y in [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]]


def _cross(a, b, p):
    return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])


def overlap_area(subject, clip):
    """Convex polygon clipping (counterclockwise vertices)."""
    result = list(subject)
    for a, b in zip(clip, clip[1:]+clip[:1]):
        previous, result = result, []
        if not previous:
            return 0.
        start = previous[-1]
        for end in previous:
            u, v = _cross(a, b, start), _cross(a, b, end)
            if (u >= 0) != (v >= 0):
                t = u/(u-v)
                result.append((start[0]+t*(end[0]-start[0]), start[1]+t*(end[1]-start[1])))
            if v >= 0:
                result.append(end)
            start = end
    return abs(sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(result, result[1:]+result[:1])))/2


def _inside(point, polygon):
    return all(_cross(a, b, point) >= -1e-8 for a, b in zip(polygon, polygon[1:]+polygon[:1]))


def validate_teleop_course(course):
    errors = []
    boxes = [b for b in course.boxes if b.collidable and b.semantic not in {"button", "button_support"}]
    polygons = {b.name: box_polygon(b) for b in boxes}
    if len(polygons) != len(boxes):
        errors.append("Duplicate physical support names")
    for b in boxes:
        if abs(b.pitch_deg) > 1e-9 or "ramp" in b.semantic.lower():
            errors.append(f"Ramps are not allowed: {b.name}")
        if not all(math.isfinite(v) for v in (*b.center_xyz_m, *b.size_xyz_m, b.yaw_deg)) or min(b.size_xyz_m) <= 0:
            errors.append(f"Invalid support geometry: {b.name}")

    tasks = course.tasks[:-1]
    ids = ["".join(c if c.isalnum() else "_" for c in t["task_id"]) for t in tasks]

    def owner(box):
        if box.name == "CompositeStartPlatform":
            return ("boundary", 0)
        if box.name == "CompositeGoalPlatform":
            return ("boundary", len(tasks))
        for i, prefix in enumerate(ids):
            if box.name.startswith(prefix+"_"):
                return ("boundary" if box.semantic in {"teleop_connector", "teleop_turn_pad"} else "stage", i)
        return ("unknown", -1)

    def adjacent(a, b):
        ka, ia = owner(a); kb, ib = owner(b)
        if ka == kb:
            return ia == ib
        if ka == "stage" and kb == "boundary":
            return ia in (ib-1, ib)
        if ka == "boundary" and kb == "stage":
            return ib in (ia-1, ia)
        return False

    # Non-adjacent overlaps create shortcuts or collide with raised stages.
    for i, a in enumerate(boxes):
        for b in boxes[i+1:]:
            if not adjacent(a, b) and overlap_area(polygons[a.name], polygons[b.name]) > 1e-6:
                errors.append(f"Nonlocal support overlap: {a.name} / {b.name}")

    for task in tasks:
        if task["type"] != "gap":
            continue
        gap = task["parameters"]["gap"]
        near, far = gap["near_edge_center_world_xyz_m"], gap["far_edge_center_world_xyz_m"]
        dx, dy = gap["crossing_direction_world_xy"]
        # Entire physical bank width, not only the center line, stays open.
        half_width = .6
        void = [(near[0]+dy*half_width, near[1]-dx*half_width),
                (far[0]+dy*half_width, far[1]-dx*half_width),
                (far[0]-dy*half_width, far[1]+dx*half_width),
                (near[0]-dy*half_width, near[1]+dx*half_width)]
        for b in boxes:
            if b.center_xyz_m[2]+b.size_xyz_m[2]/2 >= near[2] - 1e-6:
                if overlap_area(polygons[b.name], void) > 1e-6:
                    errors.append(f"Support covers gap {task['task_id']}: {b.name}")

    def supported(point, height):
        return any(abs(b.center_xyz_m[2]+b.size_xyz_m[2]/2-height) < 1e-6
                   and _inside(point, polygons[b.name]) for b in boxes)

    stage_samples = 0

    def check_segment(a, b, height, label):
        nonlocal stage_samples
        length = math.dist(a[:2], b[:2])
        if length <= 1e-8:
            return
        dx, dy = (b[0]-a[0])/length, (b[1]-a[1])/length
        count = max(1, math.ceil(length/.05))
        for i in range(count+1):
            for offset in (-.30, 0, .30):
                point = (a[0]+(b[0]-a[0])*i/count-dy*offset,
                         a[1]+(b[1]-a[1])*i/count+dx*offset)
                stage_samples += 1
                if not supported(point, height):
                    errors.append(f"Unsupported stage {label} at {point}")
                    return

    from .obstacle_course import sample_uniform_stair_spec, sample_coplanar_gap_spec
    for index, task in enumerate(tasks):
        p = task["parameters"]
        height = p["floor_height_m"]
        entry, exit_pose = p["entry_pose_xyyaw"], p["exit_pose_xyyaw"]
        frame = p["stage_frame"]
        c, s = math.cos(frame["yaw_rad"]), math.sin(frame["yaw_rad"])
        ox, oy, oz = frame["origin_world_xyz_m"]

        def local_xy(x, y=0):
            return (ox+c*x-s*y, oy+s*x+c*y)

        kind = task["type"]
        if kind in {"gap", "stairs"}:
            semantic = "gap_test_far_bank" if kind == "gap" else "stair_test_upper_deck"
            landings = [b for b in boxes if owner(b) == ("stage", index) and b.semantic == semantic]
            if len(landings) != 1 or landings[0].size_xyz_m[0] < 1.20-1e-8:
                errors.append(f"Snake landing shorter than 1.20 m: {task['task_id']}")
        if kind == "stairs":
            spec = sample_uniform_stair_spec(task["seed"])
            stairs = p["stairs"]
            risers = [b for b in boxes if owner(b) == ("stage", index) and b.semantic == "stair_test_riser"]
            if (len(risers) != spec.step_count or
                    any(abs(b.size_xyz_m[0]-spec.tread_depth_m)>1e-8 or
                        abs(b.size_xyz_m[1]-spec.width_m)>1e-8 for b in risers)):
                errors.append(f"Stair seed dimensions/completeness mismatch: {task['task_id']}")
            first = stairs["first_riser_x_m"]
            check_segment(entry, local_xy(first), height, task["task_id"])
            for step in range(spec.step_count):
                start = first + step*spec.tread_depth_m
                top = height + (step+1)*spec.rise_m
                check_segment(local_xy(start), local_xy(start+spec.tread_depth_m), top, task["task_id"])
            check_segment(local_xy(first+spec.step_count*spec.tread_depth_m), exit_pose,
                          height+spec.step_count*spec.rise_m, task["task_id"])
        elif kind == "gap":
            gap = p["gap"]
            near, far = gap["near_edge_center_world_xyz_m"], gap["far_edge_center_world_xyz_m"]
            if abs(math.dist(near, far)-sample_coplanar_gap_spec(task["seed"]).width_m)>1e-8:
                errors.append(f"Gap seed width mismatch: {task['task_id']}")
            check_segment(entry, near, height, task["task_id"])
            check_segment(far, exit_pose, height, task["task_id"])
        elif kind == "flat_navigation":
            route = p["waypoints_xyyaw"]
            for a, b in zip(route, route[1:]):
                check_segment(a, b, height, task["task_id"])
        elif kind == "button":
            check_segment(entry, exit_pose, height, task["task_id"])
            button = p["button"]
            center = button["center_xyz_m"]
            direction = button["press_direction_world_xy"]
            approach = [center[i]-.65*direction[i] for i in (0, 1)]
            if not supported(approach, height):
                errors.append(f"Unsupported button approach: {task['task_id']}")
    if boundaries_for_goal := course.layout_metadata.get("boundaries"):
        check_segment(boundaries_for_goal[-1]["exit_world_xyz_m"], course.goal_center_xyz_m,
                      course.goal_center_xyz_m[2], "goal")

    checked_samples = 0
    boundaries = course.layout_metadata.get("boundaries", [])
    for boundary in boundaries:
        previous = boundary["previous_exit_world_xyz_m"]
        pad, entry = boundary["pad_center_world_xyz_m"], boundary["entry_world_xyz_m"]
        segments = [(pad, entry)] if previous is None else [(previous, pad), (pad, entry)]
        for a, b in segments:
            length = math.dist(a[:2], b[:2])
            if length <= 1e-8:
                continue
            dx, dy = (b[0]-a[0])/length, (b[1]-a[1])/length
            count = math.ceil(length/.05)
            for i in range(count+1):
                for offset in (-.30, 0, .30):
                    point = (a[0]+(b[0]-a[0])*i/count-dy*offset,
                             a[1]+(b[1]-a[1])*i/count+dx*offset)
                    checked_samples += 1
                    if not supported(point, a[2]):
                        errors.append(f"Unsupported handoff {boundary['task_id']} at {point}")
                        break
    if not supported(course.goal_center_xyz_m[:2], course.goal_center_xyz_m[2]):
        errors.append("Unsupported goal")
    return {"valid": not errors, "errors": errors, "support_boxes": len(boxes),
            "handoff_samples": checked_samples, "stage_support_samples": stage_samples, "handoff_sample_spacing_m": .05,
            "handoff_checked_width_m": .60,
            "scope": "geometric support and gap preservation; dynamics not certified"}
