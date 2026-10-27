"""Physical invariants for the oriented teleoperation curriculum."""
import math

import pytest

from smores_ep.isaac.obstacle_course import (
    composite_obstacle_course, sample_coplanar_gap_spec, sample_uniform_stair_spec,
    rc_car_planar_obstacle_layout,
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
    assert len(campaign["episodes"]) == 16
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



def test_teleop_handoffs_match_compact_c_course_scale():
    """Connected teleop courses keep connectors without multi-metre dead zones."""
    course = composite_obstacle_course(
        mission(
            {
                "task_id": "s",
                "type": "stairs",
                "seed": 3102,
                "yaw_deg": 0,
            },
            {
                "task_id": "r",
                "type": "flat_navigation",
                "seed": 6053,
                "yaw_deg": 0,
            },
            {
                "task_id": "b",
                "type": "button",
                "seed": 8568,
                "yaw_deg": 90,
            },
        ),
        VALIDATED,
    )

    connectors = [
        box
        for box in course.boxes
        if box.semantic == "teleop_connector"
    ]
    pads = [
        box
        for box in course.boxes
        if box.semantic == "teleop_turn_pad"
    ]

    assert connectors
    assert pads

    # C-like compact handoff:
    # nominal connector = 0.80 m (+0.04 overlap tolerance)
    # width = 1.20 m.
    assert all(box.size_xyz_m[0] <= 0.84 + 1e-8 for box in connectors)
    assert all(box.size_xyz_m[1] == pytest.approx(1.20) for box in connectors)

    # Match the 1.20 x 1.20 m reconfiguration scale already used by
    # the legacy C composite courses.
    assert all(box.size_xyz_m[0] == pytest.approx(1.20) for box in pads)
    assert all(box.size_xyz_m[1] == pytest.approx(1.20) for box in pads)

    # Keep the proven full-Snake landing clearance.
    landing_boxes = [
        box
        for box in course.boxes
        if box.semantic in {
            "gap_test_far_bank",
            "stair_test_upper_deck",
        }
    ]
    assert landing_boxes
    assert all(box.size_xyz_m[0] >= 1.20 for box in landing_boxes)

    # Button zone stays C-like instead of becoming a 3 x 2.4 m dead area.
    button_platform = next(
        box
        for box in course.boxes
        if box.semantic == "button_test_platform"
    )
    assert button_platform.size_xyz_m[0] <= 1.90 + 1e-8
    assert button_platform.size_xyz_m[1] <= 1.40 + 1e-8


def test_rotated_button_wall_plunger_and_fixture_share_one_orientation():
    """The red plunger must remain rigidly aligned with its support frame."""
    course = composite_obstacle_course(
        mission(
            {
                "task_id": "b",
                "type": "button",
                "seed": 8568,
                "yaw_deg": 90,
            },
        ),
        VALIDATED,
    )

    wall = next(
        box
        for box in course.boxes
        if box.semantic == "button_support"
    )
    plunger = next(
        box
        for box in course.boxes
        if box.semantic == "button"
    )
    fixture = course.buttons[0]

    assert wall.yaw_deg == pytest.approx(90.0)
    assert plunger.yaw_deg == pytest.approx(90.0)

    # The joint installer needs the same explicit fixture orientation;
    # otherwise PhysX may satisfy an identity joint frame by rotating
    # only the dynamic plunger.
    assert fixture.yaw_deg == pytest.approx(plunger.yaw_deg)



def test_initial_spawn_platform_remains_full_size():
    """Compacting handoffs must not shrink the loose-module spawn area."""
    course = composite_obstacle_course(
        mission(
            {
                "task_id": "s",
                "type": "stairs",
                "seed": 3102,
                "yaw_deg": 0,
            },
            {
                "task_id": "r",
                "type": "flat_navigation",
                "seed": 6053,
                "yaw_deg": 0,
            },
        ),
        VALIDATED,
    )

    start = next(
        box
        for box in course.boxes
        if box.name == "CompositeStartPlatform"
    )

    # This is the proven loose-module assembly area.
    assert start.size_xyz_m[0] == pytest.approx(1.60)
    assert start.size_xyz_m[1] == pytest.approx(1.60)

    # Intermediate maneuver pads must stay compact.
    turn_pads = [
        box
        for box in course.boxes
        if box.semantic == "teleop_turn_pad"
    ]

    assert turn_pads
    assert all(
        box.size_xyz_m[0] == pytest.approx(1.20)
        and box.size_xyz_m[1] == pytest.approx(1.20)
        for box in turn_pads
    )



def _campaign_episode(episode_id):
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    config = (
        root
        / "mssr_ws/src/mssr_expert/config/"
        "smores_teleop_composite_campaign13.json"
    )

    campaign = json.loads(config.read_text(encoding="utf-8"))

    raw = next(
        item
        for item in campaign["episodes"]
        if item["episode_id"] == episode_id
    )

    return {
        **raw,
        "schema_version": "mssr.composite_mission.v1",
    }


def _validated_catalog():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    path = (
        root
        / "mssr_ws/src/mssr_expert/config/"
        "smores_composite_seed_catalog.json"
    )

    return json.loads(
        path.read_text(encoding="utf-8")
    )["validated_seeds"]


def test_t01_to_t10_use_dense_obstacle_handoffs():
    """Ordinary T-course obstacles should read as one continuous course."""
    catalog = _validated_catalog()

    for number in range(1, 11):
        episode_id = f"teleop-t{number:02d}"
        course = composite_obstacle_course(
            _campaign_episode(episode_id),
            catalog,
        )

        boundaries = course.layout_metadata["boundaries"]

        for index, boundary in enumerate(boundaries):
            previous = boundary["previous_exit_world_xyz_m"]

            if previous is None:
                continue

            entry = boundary["entry_world_xyz_m"]

            distance = math.dist(
                previous[:2],
                entry[:2],
            )

            task_id = boundary["task_id"]

            raw_task = next(
                task
                for task in _campaign_episode(
                    episode_id
                )["tasks"]
                if task["task_id"] == task_id
            )

            if raw_task.get(
                "handoff_mode"
            ) == "shared":
                assert 0.0 <= distance <= 0.65 + 1e-8
                continue

            if raw_task.get(
                "mount_on_previous_support",
                False,
            ):
                # Button fixture is physically ON the existing landing:
                # no connector/dead zone is expected.
                assert distance <= 1.20
                continue

            # T09 intentionally keeps extra maneuvering room between its
            # first gap and the following stairs.
            if episode_id == "teleop-t09" and task_id == "stairs-2":
                assert 1.20 - 1e-8 <= distance <= 1.60 + 1e-8
            else:
                # 0.80 m Snake8 is the longest morphology.
                # Allow a little margin, but not metre-scale dead zones.
                assert 0.80 - 1e-8 <= distance <= 1.00 + 1e-8, (
                    episode_id,
                    task_id,
                    distance,
                )


def test_t07_uses_validated_six_step_stairs():
    episode = _campaign_episode("teleop-t07")

    stairs = next(
        task
        for task in episode["tasks"]
        if task["type"] == "stairs"
    )

    assert stairs["seed"] == 6403

    spec = sample_uniform_stair_spec(stairs["seed"])
    assert spec.step_count == 6


def test_t08_builds_u_turn_from_two_validated_quarter_turn_rc_segments():
    episode = _campaign_episode("teleop-t08")

    tasks = episode["tasks"]

    assert [task["type"] for task in tasks] == [
        "gap",
        "flat_navigation",
        "flat_navigation",
        "stairs",
        "button",
    ]

    rc = [
        task
        for task in tasks
        if task["type"] == "flat_navigation"
    ]

    assert [task["seed"] for task in rc] == [5100, 6001]
    assert [task["yaw_deg"] for task in rc] == [0, 90]

    # Two independently validated +90 degree routes form the U.
    for task in rc:
        layout = rc_car_planar_obstacle_layout(task["seed"])
        route = layout["centerline_xy_m"]

        headings = []
        for a, b in zip(route, route[1:]):
            dx = float(b[0]) - float(a[0])
            dy = float(b[1]) - float(a[1])

            if math.hypot(dx, dy) > 1e-9:
                headings.append(math.atan2(dy, dx))

        net = 0.0
        for a, b in zip(headings, headings[1:]):
            net += math.atan2(
                math.sin(b - a),
                math.cos(b - a),
            )

        assert math.degrees(net) == pytest.approx(90.0)


def test_t04_and_t08_button_can_mount_on_existing_terminal_support():
    for episode_id in ("teleop-t04", "teleop-t08"):
        episode = _campaign_episode(episode_id)

        button = next(
            task
            for task in episode["tasks"]
            if task["type"] == "button"
        )

        assert button["mount_on_previous_support"] is True

        course = composite_obstacle_course(
            episode,
            _validated_catalog(),
        )

        # Fixture remains physical.
        assert len(course.buttons) == 1
        assert any(
            box.semantic == "button"
            for box in course.boxes
        )
        assert any(
            box.semantic == "button_support"
            for box in course.boxes
        )

        # But no new standalone button platform is added.
        assert not any(
            box.semantic == "button_test_platform"
            for box in course.boxes
        )


def test_dense_layout_keeps_full_loose_spawn_area():
    course = composite_obstacle_course(
        _campaign_episode("teleop-t01"),
        _validated_catalog(),
    )

    start = next(
        box
        for box in course.boxes
        if box.name == "CompositeStartPlatform"
    )

    assert start.size_xyz_m[:2] == pytest.approx(
        (1.60, 1.60)
    )



def _episode_task(episode_id, task_id):
    episode = _campaign_episode(episode_id)

    return next(
        task
        for task in episode["tasks"]
        if task["task_id"] == task_id
    )


def test_visual_review_v4_uses_shared_physical_handoffs():
    """Most obstacles must share landing/approach support, not add whitespace."""

    expected_shared = {
        "teleop-t01": (
            "flat_navigation-2",
        ),
        "teleop-t02": (
            "flat_navigation-2",
        ),
        "teleop-t03": (
            "gap-2",
            "flat_navigation-3",
        ),
        "teleop-t04": (
            "flat_navigation-2",
            "gap-3",
        ),
        "teleop-t05": (
            "flat_navigation-2",
        ),
        "teleop-t06": (
            "flat_navigation-2",
        ),
        "teleop-t07": (
            "flat_navigation-2",
        ),
        "teleop-t08": (
            "flat_navigation-3",
            "stairs-4",
        ),
        "teleop-t10": (
            "stairs-2",
            "flat_navigation-3",
        ),
    }

    for episode_id, task_ids in expected_shared.items():
        for task_id in task_ids:
            task = _episode_task(
                episode_id,
                task_id,
            )

            assert task.get("handoff_mode") == "shared", (
                episode_id,
                task_id,
            )


def test_t09_keeps_only_its_intentional_extended_gap_to_stairs_handoff():
    stairs = _episode_task(
        "teleop-t09",
        "stairs-2",
    )

    assert stairs.get("handoff_mode") != "shared"
    assert stairs["connector_length_m"] == pytest.approx(1.40)


def test_spawn_platform_is_compact_but_still_larger_than_course_width():
    course = composite_obstacle_course(
        _campaign_episode("teleop-t03"),
        _validated_catalog(),
    )

    start = next(
        box
        for box in course.boxes
        if box.name == "CompositeStartPlatform"
    )

    assert start.size_xyz_m[:2] == pytest.approx(
        (1.60, 1.60)
    )


def test_t01_uses_zigzag_and_t05_requests_extra_rc_cones():
    # T01 V6 supersedes the old "seeded cones + 3 extras" rule
    # with a deterministic seven-cone alternating teleop slalom.
    t01 = _episode_task(
        "teleop-t01",
        "flat_navigation-2",
    )

    assert t01.get("cone_pattern") == "zigzag"
    assert t01.get("cone_count") == 7
    assert "extra_cones" not in t01

    # T05 intentionally keeps the V4 extra-cone augmentation.
    t05_a = _episode_task(
        "teleop-t05",
        "flat_navigation-1",
    )

    t05_b = _episode_task(
        "teleop-t05",
        "flat_navigation-2",
    )

    assert t05_a.get("extra_cones") == 2
    assert t05_b.get("extra_cones") == 2


def test_visual_review_v4_buttons_reuse_previous_support_where_requested():
    mounted = (
        ("teleop-t04", "button-4"),
        ("teleop-t08", "button-5"),
        ("teleop-t09", "button-3"),
    )

    for episode_id, task_id in mounted:
        button = _episode_task(
            episode_id,
            task_id,
        )

        assert button.get(
            "mount_on_previous_support"
        ) is True

        # Keep wall/plunger clearly inside the existing platform.
        assert button.get(
            "button_inset_m"
        ) == pytest.approx(0.20)


def test_consecutive_rc_sections_have_no_separate_white_connector():
    """T05/T08 should read visually as one continuous RC course."""

    for episode_id, second_rc_id in (
        ("teleop-t05", "flat_navigation-2"),
        ("teleop-t08", "flat_navigation-3"),
    ):
        task = _episode_task(
            episode_id,
            second_rc_id,
        )

        assert task["handoff_mode"] == "shared"



def test_v4_shared_handoffs_remove_physical_connector_boxes():
    expected_shared = {
        "teleop-t01": ("flat_navigation-2", "button-3"),
        "teleop-t02": ("flat_navigation-2", "button-3"),
        "teleop-t03": ("gap-2", "flat_navigation-3"),
        "teleop-t04": ("flat_navigation-2", "gap-3"),
        "teleop-t05": ("flat_navigation-2", "button-3"),
        "teleop-t06": ("flat_navigation-2", "button-3"),
        "teleop-t07": ("flat_navigation-2", "button-3"),
        "teleop-t08": (
            "flat_navigation-2",
            "flat_navigation-3",
            "stairs-4",
        ),
        "teleop-t10": ("stairs-2", "flat_navigation-3"),
    }

    catalog = _validated_catalog()

    for episode_id, task_ids in expected_shared.items():
        course = composite_obstacle_course(
            _campaign_episode(episode_id),
            catalog,
        )

        names = {
            box.name
            for box in course.boxes
        }

        boundaries = {
            b["task_id"]: b
            for b in course.layout_metadata["boundaries"]
        }

        for task_id in task_ids:
            safe = "".join(
                c if c.isalnum() else "_"
                for c in task_id
            )

            assert f"{safe}_Connector" not in names
            assert f"{safe}_TurnPad" not in names
            assert boundaries[task_id]["handoff_mode"] == "shared"

            previous = boundaries[task_id][
                "previous_exit_world_xyz_m"
            ]
            entry = boundaries[task_id][
                "entry_world_xyz_m"
            ]

            distance = math.dist(
                entry[:2],
                previous[:2],
            )

            assert 0.0 <= distance <= 0.65 + 1e-8

            assert distance == pytest.approx(
                boundaries[task_id]["shared_shift_m"],
                abs=1e-8,
            )


def test_v4_extra_cones_are_materialized_as_physical_cones():
    expected = (
        ("teleop-t01", "flat_navigation-2", 5101, 3),
        ("teleop-t05", "flat_navigation-1", 6000, 2),
        ("teleop-t05", "flat_navigation-2", 6056, 2),
    )

    catalog = _validated_catalog()

    for episode_id, task_id, seed, extra in expected:
        course = composite_obstacle_course(
            _campaign_episode(episode_id),
            catalog,
        )

        actual = sum(
            cone.task_id == task_id
            for cone in course.navigation_cones
        )

        baseline = composite_obstacle_course(
            {
                "schema_version": "mssr.composite_mission.v1",
                "tasks": [
                    {
                        "task_id": task_id,
                        "type": "flat_navigation",
                        "seed": seed,
                    }
                ],
            },
            catalog,
        )

        baseline_count = sum(
            cone.task_id == task_id
            for cone in baseline.navigation_cones
        )

        assert actual == baseline_count + extra


def test_v4_button_overlays_are_inset_on_existing_landings():
    catalog = _validated_catalog()

    for episode_id in (
        "teleop-t04",
        "teleop-t08",
        "teleop-t09",
    ):
        episode = _campaign_episode(episode_id)

        button_task = next(
            task
            for task in episode["tasks"]
            if task["type"] == "button"
        )

        assert button_task["button_inset_m"] == pytest.approx(0.20)
        assert button_task["mount_on_previous_support"] is True

        course = composite_obstacle_course(
            episode,
            catalog,
        )

        assert not any(
            box.semantic == "button_test_platform"
            for box in course.boxes
        )



def test_v5_heading_changes_use_compact_shared_junctions():
    """Turning shared handoffs need a pad, never a long connector."""

    expected = (
        ("teleop-t01", "button-3"),
        ("teleop-t04", "gap-3"),
        ("teleop-t05", "button-3"),
        ("teleop-t06", "button-3"),
        ("teleop-t07", "button-3"),
        ("teleop-t10", "stairs-2"),
    )

    catalog = _validated_catalog()

    for episode_id, task_id in expected:
        course = composite_obstacle_course(
            _campaign_episode(episode_id),
            catalog,
        )

        safe = "".join(
            c if c.isalnum() else "_"
            for c in task_id
        )

        names = {
            box.name
            for box in course.boxes
        }

        boundary = next(
            item
            for item in course.layout_metadata["boundaries"]
            if item["task_id"] == task_id
        )

        assert boundary["handoff_mode"] == "shared"
        assert boundary["shared_junction"] is True

        assert f"{safe}_SharedJunction" in names
        assert f"{safe}_Connector" not in names
        assert f"{safe}_TurnPad" not in names

        junction = next(
            box
            for box in course.boxes
            if box.name == f"{safe}_SharedJunction"
        )

        assert junction.semantic == "teleop_turn_pad"
        assert junction.size_xyz_m[:2] == pytest.approx(
            (1.20, 1.20)
        )


def test_v5_aligned_shared_handoffs_add_no_junction():
    """Already-aligned stages remain directly edge-to-edge."""

    expected = (
        ("teleop-t01", "flat_navigation-2"),
        ("teleop-t02", "flat_navigation-2"),
        ("teleop-t02", "button-3"),
        ("teleop-t03", "gap-2"),
        ("teleop-t03", "flat_navigation-3"),
        ("teleop-t05", "flat_navigation-2"),
        ("teleop-t08", "flat_navigation-2"),
        ("teleop-t08", "flat_navigation-3"),
        ("teleop-t08", "stairs-4"),
        ("teleop-t10", "flat_navigation-3"),
    )

    catalog = _validated_catalog()

    for episode_id, task_id in expected:
        course = composite_obstacle_course(
            _campaign_episode(episode_id),
            catalog,
        )

        safe = "".join(
            c if c.isalnum() else "_"
            for c in task_id
        )

        names = {
            box.name
            for box in course.boxes
        }

        boundary = next(
            item
            for item in course.layout_metadata["boundaries"]
            if item["task_id"] == task_id
        )

        assert boundary["handoff_mode"] == "shared"
        assert boundary["shared_junction"] is False
        assert f"{safe}_SharedJunction" not in names
        assert f"{safe}_Connector" not in names


def test_v5_mounted_buttons_add_zero_course_handoff():
    """T04/T08/T09 button fixtures live on the preceding landing."""

    catalog = _validated_catalog()

    for episode_id in (
        "teleop-t04",
        "teleop-t08",
        "teleop-t09",
    ):
        course = composite_obstacle_course(
            _campaign_episode(episode_id),
            catalog,
        )

        button_task = next(
            task
            for task in _campaign_episode(
                episode_id
            )["tasks"]
            if task["type"] == "button"
        )

        boundary = next(
            item
            for item in course.layout_metadata["boundaries"]
            if item["task_id"] == button_task["task_id"]
        )

        assert boundary["entry_world_xyz_m"] == pytest.approx(
            boundary["previous_exit_world_xyz_m"]
        )

        assert boundary["pad_center_world_xyz_m"] == pytest.approx(
            boundary["previous_exit_world_xyz_m"]
        )



def _cone_progress_and_lateral(route, x, y):
    cumulative = [0.0]

    for a, b in zip(route, route[1:]):
        cumulative.append(
            cumulative[-1]
            + math.dist(a[:2], b[:2])
        )

    best = None

    for index, (a, b) in enumerate(
        zip(route, route[1:])
    ):
        ax, ay = a[:2]
        bx, by = b[:2]

        dx = bx - ax
        dy = by - ay

        length_sq = dx * dx + dy * dy

        if length_sq <= 1.0e-12:
            continue

        t = (
            (x - ax) * dx
            + (y - ay) * dy
        ) / length_sq

        t = max(
            0.0,
            min(1.0, t),
        )

        qx = ax + t * dx
        qy = ay + t * dy

        ex = x - qx
        ey = y - qy

        distance_sq = ex * ex + ey * ey

        length = math.sqrt(length_sq)

        tx = dx / length
        ty = dy / length

        nx = -ty
        ny = tx

        lateral = ex * nx + ey * ny

        progress = (
            cumulative[index]
            + t * length
        )

        candidate = (
            distance_sq,
            progress,
            lateral,
        )

        if best is None or candidate[0] < best[0]:
            best = candidate

    assert best is not None

    return best[1], best[2]


def test_v6_t01_t02_use_true_teleop_zigzag_cones():
    expected = {
        "teleop-t01": {
            "task_id": "flat_navigation-2",
            "count": 7,
        },
        "teleop-t02": {
            "task_id": "flat_navigation-2",
            "count": 8,
        },
    }

    for episode_id, contract in expected.items():
        task = _episode_task(
            episode_id,
            contract["task_id"],
        )

        assert task.get("cone_pattern") == "zigzag"
        assert task.get("cone_count") == contract["count"]


def test_v6_t01_t02_generated_cones_force_alternating_center_crossing():
    catalog = _validated_catalog()

    expected_counts = {
        "teleop-t01": 7,
        "teleop-t02": 8,
    }

    for episode_id, expected_count in expected_counts.items():
        course = composite_obstacle_course(
            _campaign_episode(episode_id),
            catalog,
        )

        rc_task = next(
            task
            for task in course.tasks
            if task["type"] == "flat_navigation"
        )

        route = rc_task["parameters"][
            "waypoints_xyyaw"
        ]

        cones = [
            cone
            for cone in course.navigation_cones
            if cone.task_id == rc_task["task_id"]
        ]

        assert len(cones) == expected_count

        ordered = sorted(
            (
                *_cone_progress_and_lateral(
                    route,
                    cone.center_xyz_m[0],
                    cone.center_xyz_m[1],
                ),
                cone,
            )
            for cone in cones
        )

        progress = [
            item[0]
            for item in ordered
        ]

        lateral = [
            item[1]
            for item in ordered
        ]

        # No paired "gate": every obstacle occurs at its own
        # longitudinal position.
        total_length = sum(
            math.dist(a[:2], b[:2])
            for a, b in zip(route, route[1:])
        )

        progress_fraction = [
            value / total_length
            for value in progress
        ]

        assert all(
            b - a >= 0.07
            for a, b in zip(
                progress_fraction,
                progress_fraction[1:],
            )
        )

        # Actual slalom: side changes at EVERY successive cone.
        signs = [
            1 if value > 0 else -1
            for value in lateral
        ]

        assert all(
            a != b
            for a, b in zip(
                signs,
                signs[1:],
            )
        )

        # Cones must intrude toward the route centre enough that
        # driving straight down the middle is no longer the easy solution.
        assert all(
            0.06 <= abs(value) <= 0.14
            for value in lateral
        )


def test_new_il_orders_are_compact_and_keep_snake_landings():
    import json
    from pathlib import Path
    from smores_ep.isaac.teleop_composite_course import validate_teleop_course

    root = Path(__file__).resolve().parents[3]
    config = root / "mssr_ws/src/mssr_expert/config"
    campaign = json.loads((config / "smores_teleop_composite_campaign13.json").read_text())
    catalog = json.loads((config / "smores_composite_seed_catalog.json").read_text())["validated_seeds"]
    episodes = {item["episode_id"]: item for item in campaign["episodes"]}
    expected = {
        "teleop-t14": [("button", 6103), ("flat_navigation", 6017), ("stairs", 3105)],
        "teleop-t15": [("button", 6101), ("flat_navigation", 5104), ("gap", 4105)],
        "teleop-t16": [("flat_navigation", 5100), ("stairs", 3101), ("gap", 4103)],
    }
    for episode_id, tasks in expected.items():
        episode = episodes[episode_id]
        assert [(task["type"], task["seed"]) for task in episode["tasks"]] == tasks
        course = composite_obstacle_course(
            {**episode, "schema_version": "mssr.composite_mission.v1"}, catalog
        )
        assert validate_teleop_course(course)["valid"]
        assert all(box.pitch_deg == 0 and "ramp" not in box.semantic for box in course.boxes)
        assert all(box.size_xyz_m[0] >= 1.2 for box in course.boxes
                   if box.semantic in {"gap_test_far_bank", "stair_test_upper_deck"})
