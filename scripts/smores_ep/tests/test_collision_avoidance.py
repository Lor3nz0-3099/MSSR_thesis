from __future__ import annotations

from smores_ep.primitives.collision_avoidance import (
    CircularObstacle,
    plan_collision_aware_path,
    segment_is_clear,
)


def test_free_staging_path_needs_no_waypoint() -> None:
    obstacle = CircularObstacle("smores_02", (0.0, 0.4), 0.110)

    route = plan_collision_aware_path(
        (-0.2, 0.0),
        (0.2, 0.0),
        (obstacle,),
    )

    assert route == ((0.2, 0.0),)


def test_mobile_manipulator_staging_detours_around_root() -> None:
    start = (0.0337700796, -0.0912308292)
    goal = (0.1418618810, 0.0000867031)
    obstacles = (
        CircularObstacle("smores_01", (-0.0059, 0.0004), 0.110),
        CircularObstacle("smores_02", (0.3459, -0.0003), 0.110),
        CircularObstacle("smores_03", (0.1731, 0.2993), 0.110),
        CircularObstacle("smores_04", (-0.0852, 0.0005), 0.110),
        CircularObstacle("smores_05", (-0.3459, 0.0003), 0.110),
        CircularObstacle("smores_07", (0.1726, -0.2996), 0.110),
    )

    assert not segment_is_clear(
        start,
        goal,
        obstacles,
        allow_start_inside=True,
    )
    route = plan_collision_aware_path(start, goal, obstacles)

    assert route is not None
    assert len(route) >= 2
    points = (start,) + route
    assert all(
        segment_is_clear(
            points[index],
            points[index + 1],
            obstacles,
            allow_start_inside=(index == 0),
        )
        for index in range(len(points) - 1)
    )


def test_start_inside_clearance_can_only_escape_outward() -> None:
    obstacle = CircularObstacle("root", (0.0, 0.0), 0.110)

    assert segment_is_clear(
        (0.100, 0.0),
        (0.140, 0.0),
        (obstacle,),
        allow_start_inside=True,
    )
    assert not segment_is_clear(
        (0.100, 0.0),
        (0.0, 0.140),
        (obstacle,),
        allow_start_inside=True,
    )


def test_dense_sampling_finds_snake_to_rc_car_staging_exit() -> None:
    # Captured immediately after undocking the assembled Snake7.  The wheel
    # candidate starts inside overlapping safety footprints in the packed
    # chain, and its RC Car staging pose is below the chassis.
    start = (0.1529994190097993, 0.0027492840999632784)
    goal = (0.06826784902889223, -0.14712662119454065)
    obstacles = (
        CircularObstacle(
            "smores_01",
            (-0.0058339862152934074, 0.0003028709616046399),
            0.110,
        ),
        CircularObstacle(
            "smores_02",
            (0.2318254226446152, 0.006227368954569114),
            0.110,
        ),
        CircularObstacle(
            "smores_03",
            (0.07428672324675711, 0.005681287449901484),
            0.110,
        ),
        CircularObstacle(
            "smores_04",
            (-0.1657201611863202, 0.007000034638897257),
            0.110,
        ),
        CircularObstacle(
            "smores_05",
            (-0.24369422733783724, -0.0010244576260447322),
            0.110,
        ),
        CircularObstacle(
            "smores_06",
            (-0.08572151266561007, 0.0047790185187146506),
            0.110,
        ),
    )

    assert plan_collision_aware_path(
        start,
        goal,
        obstacles,
        angular_samples=16,
    ) is None

    route = plan_collision_aware_path(
        start,
        goal,
        obstacles,
        angular_samples=64,
    )
    assert route is not None
    points = (start,) + route
    assert all(
        segment_is_clear(
            points[index],
            points[index + 1],
            obstacles,
            allow_start_inside=(index == 0),
        )
        for index in range(len(points) - 1)
    )


def test_snake8_false_positive_staging_has_physical_envelope_fallback() -> None:
    # Captured from course-snake8-assembly-01-w2-a1-reach. The final docking
    # corridor is visibly free, but the nominal staging goal lands inside the
    # conservative 110 mm proxy around the isolated smores_02.
    start = (0.21333, 0.26704)
    final_target = (0.2460, -0.0100)
    target_normal = (0.99653, -0.06292)
    centres = {
        "smores_01": (0.00193, 0.00020),
        "smores_02": (0.34583, -0.00029),
        "smores_04": (-0.07976, -0.00033),
        "smores_05": (-0.30224, 0.02878),
        "smores_06": (-0.16094, 0.00043),
        "smores_07": (0.08369, -0.00050),
        "smores_08": (0.16482, -0.00495),
    }

    nominal_goal = (
        final_target[0] + 0.070 * target_normal[0],
        final_target[1] + 0.070 * target_normal[1],
    )
    nominal_obstacles = tuple(
        CircularObstacle(module_id, centre, 0.110)
        for module_id, centre in centres.items()
    )
    assert plan_collision_aware_path(
        start,
        nominal_goal,
        nominal_obstacles,
        waypoint_margin_m=0.015,
        angular_samples=64,
    ) is None

    fallback_goal = (
        final_target[0] + 0.015 * target_normal[0],
        final_target[1] + 0.015 * target_normal[1],
    )
    fallback_obstacles = tuple(
        CircularObstacle(module_id, centre, 0.082)
        for module_id, centre in centres.items()
    )
    route = plan_collision_aware_path(
        start,
        fallback_goal,
        fallback_obstacles,
        waypoint_margin_m=0.005,
        angular_samples=64,
    )
    assert route == (fallback_goal,)
