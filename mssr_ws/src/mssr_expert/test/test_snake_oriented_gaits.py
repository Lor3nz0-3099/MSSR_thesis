"""A rotated course must preserve joint plans and rotate progress feedback."""
import copy
from dataclasses import replace
import math

import pytest

from mssr_expert.behaviors.morphology_library import LongitudinalPositionGoal, BehaviorProgramStep
from mssr_expert.behaviors.snake_gap_gait import SnakeGapGaitPlanner
from mssr_expert.behaviors.snake_stair_concertina import SnakeStairConcertinaPlanner
from mssr_expert.execution.morphology_behavior_executor import MorphologyBehaviorExecutor
from test_snake_gap_gait import _graph as gap_graph, _assignments
from test_snake_stair_concertina import _graph as stair_graph, _library


def oriented(graph, kind, yaw, origin=(3., -2., .4)):
    c, s = round(math.cos(yaw), 12), round(math.sin(yaw), 12)
    def point(p):
        return [origin[0]+c*p[0]-s*p[1], origin[1]+s*p[0]+c*p[1], origin[2]+p[2]]
    attrs = copy.deepcopy(graph.global_attributes)
    course = attrs["course"]
    course[kind]["coordinate_frame"] = "stage_local"
    course["stage_frame"] = {"origin_world_xyz_m": list(origin), "yaw_rad": yaw}
    for box in course.get("collision_boxes", []):
        box["center_xyz_m"] = point(box["center_xyz_m"])
        box["yaw_deg"] = box.get("yaw_deg", 0.) + math.degrees(yaw)
    nodes = tuple(replace(n, attributes={**n.attributes, "position": point(n.attributes["position"])}) for n in graph.nodes)
    return replace(graph, nodes=nodes, global_attributes=attrs)


@pytest.mark.parametrize("kind,factory,planner", [
    ("gap", gap_graph, SnakeGapGaitPlanner),
    ("stairs", stair_graph, SnakeStairConcertinaPlanner),
])
@pytest.mark.parametrize("yaw", [0., math.pi/2, -math.pi/2])
def test_joint_trajectory_is_equivariant_and_world_graph_unchanged(kind, factory, planner, yaw):
    graph = factory()
    world = oriented(graph, kind, yaw)
    before = copy.deepcopy(world)
    baseline = planner().plan(graph, _assignments(), {})
    rotated = planner().plan(world, _assignments(), {})
    assert world == before
    assert len(rotated) == len(baseline)
    for expected, actual in zip(baseline, rotated):
        assert actual.phase == expected.phase
        assert actual.linear_m_s == expected.linear_m_s
        assert len(actual.posture_targets) == len(expected.posture_targets)
        for a, b in zip(actual.posture_targets, expected.posture_targets):
            assert a.module_id == b.module_id
            assert a.angle_rad == pytest.approx(b.angle_rad, abs=1e-8)
        if expected.position_goal:
            assert actual.position_goal.target_x_m == pytest.approx(expected.position_goal.target_x_m, abs=1e-8)
            assert actual.position_goal.axis_world_xy == pytest.approx((math.cos(yaw), math.sin(yaw)))
            assert actual.position_goal.origin_world_xy_m == (3., -2.)


@pytest.mark.parametrize("axis,position", [((0., 1.), (100., 2.98, 0.)), ((0., -1.), (100., 1.02, 0.))])
def test_executor_progress_and_tracking_use_obstacle_axis(axis, position):
    goal = LongitudinalPositionGoal("m0", 1., .005,
                                    origin_world_xy_m=(3., 2.), axis_world_xy=axis)
    reached, _ = MorphologyBehaviorExecutor._position_goal_reached(goal, .04, {"m0": position})
    assert not reached  # world X=100 must not prematurely finish the gait
    endpoint = (3.+axis[0], 2.+axis[1], 0.)
    assert MorphologyBehaviorExecutor._position_goal_reached(goal, .04, {"m0": endpoint})[0]
    executor = MorphologyBehaviorExecutor(_library())
    step = BehaviorProgramStep("test", linear_m_s=.04, position_goal=goal, position_tracking_kp_s_inv=1.)
    speed = executor._position_tracking_speed(step, 0., {"m0": position}, speed_limit_m_s=.04)
    assert speed == pytest.approx(.02)


def test_composite_selection_pins_nearby_obstacle_and_rejects_ambiguity():
    from mssr_expert.behaviors.snake_gait_frame import resolve_gait_parameters
    graph = oriented(gap_graph(), "gap", math.pi/2)
    course = copy.deepcopy(graph.global_attributes["course"])
    parameters = {k: course.pop(k) for k in ("gap", "stage_frame")}
    task = {"task_id": "gap-1", "type": "gap", "parameters": parameters}
    course["mission"] = {"tasks": [task]}
    graph = replace(graph, global_attributes={**graph.global_attributes, "course": course})
    resolved = resolve_gait_parameters(graph, _assignments(), {}, "gap")
    assert resolved["task_id"] == "gap-1"
    assert resolved["stage_frame"] == parameters["stage_frame"]
    course["mission"]["tasks"].append({**task, "task_id": "gap-2"})
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_gait_parameters(graph, _assignments(), {}, "gap")
    assert resolve_gait_parameters(graph, _assignments(), {"task_id": "gap-2"}, "gap")["task_id"] == "gap-2"
    with pytest.raises(ValueError, match="Unknown"):
        resolve_gait_parameters(graph, _assignments(), {"task_id": "missing"}, "gap")


def test_legacy_parameters_are_unchanged():
    from mssr_expert.behaviors.snake_gait_frame import resolve_gait_parameters
    parameters = {"speed_m_s": .04}
    assert resolve_gait_parameters(gap_graph(), _assignments(), parameters, "gap") == parameters


def test_legacy_composite_explicit_geometry_is_preserved():
    from mssr_expert.behaviors.snake_gait_frame import resolve_gait_parameters
    graph = gap_graph()
    attrs = copy.deepcopy(graph.global_attributes)
    attrs["course"]["mission"] = {"tasks": [{"task_id": "gap-1", "type": "gap", "parameters": {"gap": attrs["course"]["gap"]}}]}
    graph = replace(graph, global_attributes=attrs)
    parameters = {"gap": attrs["course"]["gap"]}
    assert resolve_gait_parameters(graph, _assignments(), parameters, "gap") == parameters


def test_node_overlay_preserves_frame_and_filters_other_staircases():
    from mssr_expert.nodes.smores_morphology_behavior_node import graph_with_command_course
    from mssr_expert.behaviors.snake_gait_frame import resolve_gait_parameters
    world = oriented(stair_graph(), "stairs", -math.pi/2)
    attrs = copy.deepcopy(world.global_attributes)
    course = attrs["course"]
    parameters = {k: course.pop(k) for k in ("stairs", "stage_frame")}
    course["mission"] = {"tasks": [{"task_id": "stairs-1", "type": "stairs", "parameters": parameters}]}
    for box in course["collision_boxes"]:
        box["name"] = "stairs_1_" + box["name"]
    other = copy.deepcopy(course["collision_boxes"][0])
    other["name"] = "stairs_2_Stair01"
    other["center_xyz_m"] = [999., 999., 999.]
    course["collision_boxes"].append(other)
    world = replace(world, global_attributes=attrs)
    before = copy.deepcopy(world)
    resolved = resolve_gait_parameters(world, _assignments(), {"task_id": "stairs-1"}, "stairs")
    overlay = graph_with_command_course(world, resolved)
    assert overlay.nodes == world.nodes
    assert overlay.global_attributes["course"]["stage_frame"] == parameters["stage_frame"]
    result = SnakeStairConcertinaPlanner().plan(overlay, _assignments(), resolved)
    assert result and all(step.position_goal.axis_world_xy == (0., -1.) for step in result if step.position_goal)
    assert world == before
