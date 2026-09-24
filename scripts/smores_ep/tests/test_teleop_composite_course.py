"""Physical invariants for the oriented teleoperation curriculum."""
import math

import pytest

from smores_ep.isaac.obstacle_course import (
    composite_obstacle_course, sample_coplanar_gap_spec, sample_uniform_stair_spec,
)

VALIDATED = {"gap": {4106}, "stairs": {3102}, "flat_navigation": {6053}, "button": {8568}}


def mission(*tasks):
    return {"schema_version": "mssr.composite_mission.v1", "episode_id": "test-oriented",
            "layout_profile": "teleop_connected_v1", "tasks": list(tasks)}


@pytest.mark.parametrize("yaw", [-90, 90])
def test_rotated_gap_preserves_void_and_records_world_edges(yaw):
    course = composite_obstacle_course(mission({"type": "gap", "seed": 4106, "yaw_deg": yaw}), VALIDATED)
    params = course.tasks[0]["parameters"]
    near, far = params["gap"]["near_edge_center_world_xyz_m"], params["gap"]["far_edge_center_world_xyz_m"]
    assert near[0] == pytest.approx(far[0])
    assert far[1] - near[1] == pytest.approx(math.copysign(sample_coplanar_gap_spec(4106).width_m, yaw))
    assert params["geometry_frame"] == "world"
    assert params["gap"]["coordinate_frame"] == "stage_local"
    assert course.to_observation()["mission"]["episode_id"] == "test-oriented"
    assert course.to_observation()["layout_profile"] == "teleop_connected_v1"


def test_stairs_rotate_and_downstream_support_retains_height():
    course = composite_obstacle_course(mission(
        {"task_id": "s", "type": "stairs", "seed": 3102, "yaw_deg": 90},
        {"task_id": "g", "type": "gap", "seed": 4106, "yaw_deg": 90}), VALIDATED)
    spec = sample_uniform_stair_spec(3102)
    risers = [b for b in course.boxes if b.semantic == "stair_test_riser"]
    assert len(risers) == spec.step_count
    assert all(b.yaw_deg == pytest.approx(90) for b in risers)
    assert risers[1].center_xyz_m[1] - risers[0].center_xyz_m[1] == pytest.approx(spec.tread_depth_m)
    assert course.tasks[1]["parameters"]["floor_height_m"] == pytest.approx(spec.rise_m * spec.step_count)


def test_route_and_button_fixture_share_transformed_world_metadata():
    course = composite_obstacle_course(mission(
        {"task_id": "r", "type": "flat_navigation", "seed": 6053, "yaw_deg": 90},
        {"task_id": "b", "type": "button", "seed": 8568, "yaw_deg": 90}), VALIDATED)
    route = course.tasks[0]["parameters"]["waypoints_xyyaw"]
    assert route[-1][2] == pytest.approx(0, abs=1e-8)
    fixture = course.buttons[0]
    button = course.tasks[1]["parameters"]["button"]
    assert fixture.press_direction_world_xy == pytest.approx((-1, 0))
    assert button["center_xyz_m"] == pytest.approx(fixture.center_xyz_m)
    assert button["press_direction_world_xy"] == pytest.approx(fixture.press_direction_world_xy)
    assert len([b for b in course.boxes if b.semantic == "teleop_turn_pad"]) >= 1


def test_unknown_layout_is_rejected_instead_of_silently_using_legacy():
    data = mission({"type": "gap", "seed": 4106})
    data["layout_profile"] = "typo"
    with pytest.raises(ValueError, match="layout_profile"):
        composite_obstacle_course(data, VALIDATED)


def test_geometry_audit_detects_missing_connector_and_filled_gap():
    from dataclasses import replace
    from smores_ep.isaac.teleop_composite_course import validate_teleop_course
    from smores_ep.isaac.obstacle_course import CourseBox
    course = composite_obstacle_course(mission(
        {"task_id": "g", "type": "gap", "seed": 4106, "yaw_deg": 0},
        {"task_id": "s", "type": "stairs", "seed": 3102, "yaw_deg": 90}), VALIDATED)
    assert validate_teleop_course(course)["valid"]
    broken = replace(course, boxes=tuple(b for b in course.boxes if b.semantic != "teleop_connector"))
    assert not validate_teleop_course(broken)["valid"]
    gap = course.tasks[0]["parameters"]["gap"]
    a, b = gap["near_edge_center_world_xyz_m"], gap["far_edge_center_world_xyz_m"]
    plug = CourseBox("accidental_bridge", ((a[0]+b[0])/2, a[1], -.01),
                     (gap["width_m"], 1.2, .02), (0, 0, 0), semantic="teleop_connector")
    blocked = validate_teleop_course(replace(course, boxes=course.boxes+(plug,)))
    assert any("gap" in error for error in blocked["errors"])


def test_geometry_audit_rejects_course_folding_back_over_prior_stages():
    from smores_ep.isaac.teleop_composite_course import validate_teleop_course
    course = composite_obstacle_course(mission(
        {"task_id": "a", "type": "stairs", "seed": 3102, "yaw_deg": 0},
        {"task_id": "b", "type": "stairs", "seed": 3102, "yaw_deg": 180}), VALIDATED)
    assert not validate_teleop_course(course)["valid"]


def test_campaign_covers_validated_rc_and_button_seeds_and_passes_geometry_audit():
    import json
    from pathlib import Path
    from smores_ep.isaac.teleop_composite_course import validate_teleop_course
    root = Path(__file__).resolve().parents[3]
    config = root / "mssr_ws/src/mssr_expert/config"
    campaign = json.loads((config / "smores_teleop_composite_campaign13.json").read_text())
    catalog = json.loads((config / "smores_composite_seed_catalog.json").read_text())["validated_seeds"]
    used = {"flat_navigation": set(), "button": set()}
    for episode in campaign["episodes"]:
        data = {**episode, "schema_version": "mssr.composite_mission.v1"}
        course = composite_obstacle_course(data, catalog)
        audit = validate_teleop_course(course)
        assert audit["valid"], (episode["episode_id"], audit["errors"])
        for task in episode["tasks"]:
            if task["type"] in used:
                used[task["type"]].add(task["seed"])
    assert len(campaign["episodes"]) == 13
    assert used == {k: set(catalog[k]) for k in used}


def test_invalid_geometry_is_rejected_before_isaac_is_touched():
    from smores_ep.isaac.obstacle_course import install_composite_obstacle_course
    data = mission({"task_id": "a", "type": "stairs", "seed": 3102, "yaw_deg": 0},
                   {"task_id": "b", "type": "stairs", "seed": 3102, "yaw_deg": 180})
    with pytest.raises(ValueError, match="Invalid teleoperation course"):
        install_composite_obstacle_course(None, data, VALIDATED)


def test_geometry_fingerprint_is_reproducible_and_changes_with_orientation():
    a = mission({"type": "gap", "seed": 4106, "yaw_deg": 0})
    b = mission({"type": "gap", "seed": 4106, "yaw_deg": 90})
    first = composite_obstacle_course(a, VALIDATED).to_observation()
    second = composite_obstacle_course(a, VALIDATED).to_observation()
    rotated = composite_obstacle_course(b, VALIDATED).to_observation()
    assert first["geometry_sha256"] == second["geometry_sha256"]
    assert first["geometry_sha256"] != rotated["geometry_sha256"]


@pytest.mark.parametrize("removed_semantic", ["stair_test_riser", "gap_test_far_bank", "flat_navigation_road"])
def test_audit_rejects_missing_support_inside_an_obstacle(removed_semantic):
    from dataclasses import replace
    from smores_ep.isaac.teleop_composite_course import validate_teleop_course
    course = composite_obstacle_course(mission(
        {"task_id": "s", "type": "stairs", "seed": 3102, "yaw_deg": 90},
        {"task_id": "g", "type": "gap", "seed": 4106, "yaw_deg": 90},
        {"task_id": "r", "type": "flat_navigation", "seed": 6053, "yaw_deg": 90}), VALIDATED)
    broken = replace(course, boxes=tuple(b for b in course.boxes if b.semantic != removed_semantic))
    assert not validate_teleop_course(broken)["valid"]


@pytest.mark.parametrize("yaw", [0, 90, -90])
def test_snake_has_120cm_landing_before_turn_or_next_obstacle(yaw):
    course = composite_obstacle_course(mission(
        {"task_id": "g", "type": "gap", "seed": 4106, "yaw_deg": yaw},
        {"task_id": "s", "type": "stairs", "seed": 3102, "yaw_deg": yaw}), VALIDATED)
    landing = next(b for b in course.boxes if b.semantic == "gap_test_far_bank")
    assert landing.size_xyz_m[0] >= 1.20
    assert landing.size_xyz_m[1] >= 1.20
    assert all(b.pitch_deg == 0 and "ramp" not in b.semantic for b in course.boxes)
    p = course.tasks[0]["parameters"]
    far = p["gap"]["far_edge_center_world_xyz_m"]
    exit_pose = p["exit_pose_xyyaw"]
    assert math.dist(far[:2], exit_pose[:2]) >= 1.20 - 1e-8


def test_audit_rejects_ramps_and_shortened_snake_landing():
    from dataclasses import replace
    from smores_ep.isaac.teleop_composite_course import validate_teleop_course
    course = composite_obstacle_course(mission({"type": "gap", "seed": 4106}), VALIDATED)
    ramped = replace(course, boxes=tuple(replace(b, pitch_deg=5) if b.semantic == "teleop_turn_pad" or b.name == "CompositeGoalPlatform" else b for b in course.boxes))
    assert not validate_teleop_course(ramped)["valid"]
    shortened = replace(course, boxes=tuple(replace(b, size_xyz_m=(.8, *b.size_xyz_m[1:])) if b.semantic == "gap_test_far_bank" else b for b in course.boxes))
    assert not validate_teleop_course(shortened)["valid"]
