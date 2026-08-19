"""Tests for RC Car7 -> Snake7 topology-preserving reconfiguration."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from mssr_expert.execution.self_reconfiguration_executor import (
    SelfReconfigurationExecutor,
)
from mssr_expert.graph.attributed_robot_graph import (
    AttributedRobotGraph,
    GraphEdge,
    GraphNode,
)
from mssr_expert.graph.serialization import load_attributed_graph
from mssr_expert.planning.smores_ep.self_reconfiguration_planner import (
    ReconfigurationDetachAction,
    SmoresSelfReconfigurationPlanner,
)
from mssr_expert.planning.smores_ep.assembly_sequence import AssemblyAction
from mssr_expert.planning.smores_ep.attributed_adapter import (
    target_graph_to_kinematic_tree,
)
from mssr_expert.planning.smores_ep.parallel_self_assembly_planner import (
    ParallelSelfAssemblyPlanner,
)
from mssr_expert.planning.smores_ep.rooting import root_kinematic_tree
from mssr_expert.planning.smores_ep.topology import (
    SmoresKinematicTree,
    SmoresTopologyEdge,
)
from mssr_expert.planning.smores_ep.unfolding import (
    FACE_ANGLE_RAD,
    PlanarPose,
    unfold_tree_on_plane,
)


MODULES = tuple(f"smores_{index:02d}" for index in range(1, 8))
CONFIG_ROOT = Path(__file__).parents[1] / "config"


def _node(
    module_id: str,
    position: tuple[float, float, float] = (0.0, 0.0, 0.03),
) -> GraphNode:
    return GraphNode(
        module_id,
        {
            "node_type": "physical_module",
            "robot_family": "smores_ep",
            "control_available": True,
            "position": list(position),
            "orientation": [0.0, 0.0, 0.0, 1.0],
        },
    )


def _connection(
    module_a: str,
    face_a: str,
    module_b: str,
    face_b: str,
) -> GraphEdge:
    return GraphEdge(
        module_a,
        module_b,
        {
            "relation_type": "current_connection",
            "is_attached": True,
            "connector_a_id": face_a,
            "connector_b_id": face_b,
            "face_a": face_a,
            "face_b": face_b,
        },
    )


def _rc_car_graph() -> AttributedRobotGraph:
    # A three-module TOP/BOTTOM spine plus four BOTTOM-to-lateral wheels.
    return AttributedRobotGraph(
        nodes=tuple(_node(module_id) for module_id in MODULES),
        edges=(
            _connection(MODULES[1], "TOP", MODULES[0], "BOTTOM"),
            _connection(MODULES[0], "TOP", MODULES[2], "BOTTOM"),
            _connection(MODULES[3], "BOTTOM", MODULES[1], "LEFT"),
            _connection(MODULES[4], "BOTTOM", MODULES[1], "RIGHT"),
            _connection(MODULES[5], "BOTTOM", MODULES[2], "LEFT"),
            _connection(MODULES[6], "BOTTOM", MODULES[2], "RIGHT"),
        ),
    )


def _captured_snake_graph() -> AttributedRobotGraph:
    """Return the packed Snake7 geometry that exposed crossed wheel paths."""

    positions = {
        "smores_01": (-0.0058339862, 0.0003028710, 0.03),
        "smores_02": (0.2318254226, 0.0062273690, 0.03),
        "smores_03": (0.0742867232, 0.0056812874, 0.03),
        "smores_04": (-0.1657201612, 0.0070000346, 0.03),
        "smores_05": (-0.2436942273, -0.0010244576, 0.03),
        "smores_06": (-0.0857215127, 0.0047790185, 0.03),
        "smores_07": (0.1529994319, 0.0027492916, 0.03),
    }
    return AttributedRobotGraph(
        nodes=tuple(
            _node(module_id, positions[module_id])
            for module_id in MODULES
        ),
        edges=(
            _connection("smores_03", "BOTTOM", "smores_01", "TOP"),
            _connection("smores_06", "TOP", "smores_01", "BOTTOM"),
            _connection("smores_07", "BOTTOM", "smores_03", "TOP"),
            _connection("smores_04", "TOP", "smores_06", "BOTTOM"),
            _connection("smores_05", "TOP", "smores_04", "BOTTOM"),
            _connection("smores_02", "BOTTOM", "smores_07", "TOP"),
        ),
    )


def _snake_target() -> AttributedRobotGraph:
    return load_attributed_graph(
        CONFIG_ROOT / "smores_snake7.json"
    )


def _target(morphology: str) -> AttributedRobotGraph:
    return load_attributed_graph(
        CONFIG_ROOT / f"smores_{morphology}.json"
    )


def _physical_graph_from_target(
    target: AttributedRobotGraph,
) -> AttributedRobotGraph:
    tree = target_graph_to_kinematic_tree(target)
    module_ids = tuple(
        f"smores_{index:02d}"
        for index in range(1, len(tree.vertex_ids) + 1)
    )
    mapping = {
        vertex: module_ids[index]
        for index, vertex in enumerate(tree.vertex_ids)
    }
    return AttributedRobotGraph(
        nodes=tuple(_node(module_id) for module_id in module_ids),
        edges=tuple(
            _connection(
                mapping[edge.vertex_a],
                edge.face_a,
                mapping[edge.vertex_b],
                edge.face_b,
            )
            for edge in tree.edges
        ),
    )


def _physical_graph_with_target_layout(
    target: AttributedRobotGraph,
) -> AttributedRobotGraph:
    tree = target_graph_to_kinematic_tree(target)
    rooted = root_kinematic_tree(
        tree,
        root_id=ParallelSelfAssemblyPlanner._declared_target_root(target),
    )
    unfolded = unfold_tree_on_plane(rooted)
    module_ids = tuple(
        f"smores_{index:02d}"
        for index in range(1, len(tree.vertex_ids) + 1)
    )
    mapping = {
        vertex: module_ids[index]
        for index, vertex in enumerate(tree.vertex_ids)
    }
    return AttributedRobotGraph(
        nodes=tuple(
            _node(
                mapping[vertex],
                (
                    unfolded.poses_by_vertex[vertex].x_m,
                    unfolded.poses_by_vertex[vertex].y_m,
                    0.03,
                ),
            )
            for vertex in tree.vertex_ids
        ),
        edges=tuple(
            _connection(
                mapping[edge.vertex_a],
                edge.face_a,
                mapping[edge.vertex_b],
                edge.face_b,
            )
            for edge in tree.edges
        ),
    )


def _physical_target_graph(plan) -> AttributedRobotGraph:
    edges = tuple(
        _connection(
            plan.assignment.target_to_module[edge.vertex_a],
            edge.face_a,
            plan.assignment.target_to_module[edge.vertex_b],
            edge.face_b,
        )
        for edge in plan.target_tree.edges
    )
    module_ids = tuple(
        sorted(
            set(plan.assignment.target_to_module.values())
            | set(plan.reserve_module_ids)
        )
    )
    return AttributedRobotGraph(
        nodes=tuple(_node(module_id) for module_id in module_ids),
        edges=edges,
    )


def _succeeded(goal) -> dict:
    return {
        "schema_version": "mssr.primitive_status.v1",
        "goal_id": goal.goal_id,
        "primitive": goal.primitive,
        "state": "succeeded",
        "module_ids": list(goal.module_ids),
        "phase": "terminal",
        "progress": 1.0,
        "code": "DONE",
        "message": "done",
    }


def _accepted(goal) -> dict:
    return {
        "schema_version": "mssr.primitive_status.v1",
        "goal_id": goal.goal_id,
        "primitive": goal.primitive,
        "state": "accepted",
        "module_ids": list(goal.module_ids),
        "phase": "admission",
        "progress": 0.0,
        "code": "ACCEPTED",
        "message": "accepted",
    }


def test_rc_car_to_snake_keeps_three_module_spine() -> None:
    planner = SmoresSelfReconfigurationPlanner()
    source = _target("rc_car7")
    source_assignment = planner.configuration_assignment(
        _rc_car_graph(), source
    )
    plan = planner.plan(
        _rc_car_graph(),
        _snake_target(),
        source_graph=source,
        source_assignment=source_assignment,
    )

    assert plan.retained_connection_count == 2
    assert len(plan.retained_module_ids) == 3
    assert len(plan.prepare_tilt_by_module) == 4
    assert len(plan.detach_actions) == 4
    assert plan.new_connection_count == 4
    assert [len(wave.actions) for wave in plan.assembly_plan.waves] == [1] * 4
    assert not plan.assembly_plan.requires_helper
    assert plan.source_morphology == "rc_car7"
    assert plan.target_morphology == "snake7"


def test_holonomic9_to_manipulator7_releases_two_leaf_reserves() -> None:
    module_ids = tuple(f"smores_{index:02d}" for index in range(1, 10))
    source = _target("holonomic9")
    tree = target_graph_to_kinematic_tree(source)
    mapping = {
        vertex: module_ids[index]
        for index, vertex in enumerate(tree.vertex_ids)
    }
    current = AttributedRobotGraph(
        nodes=tuple(_node(module_id) for module_id in module_ids),
        edges=tuple(
            _connection(
                mapping[edge.vertex_a],
                edge.face_a,
                mapping[edge.vertex_b],
                edge.face_b,
            )
            for edge in tree.edges
        ),
    )
    planner = SmoresSelfReconfigurationPlanner()
    plan = planner.plan(
        current,
        _target("mobile_manipulator7"),
        source_graph=source,
        source_assignment=planner.configuration_assignment(current, source),
    )

    assert len(plan.reserve_module_ids) == 2
    assert len(plan.reserve_detach_actions) == 2
    assert len(plan.prepare_tilt_by_module) == 4
    assert set(plan.prepare_tilt_by_module.values()) == {0.0}
    assert plan.prepare_stabilize_module_ids == (mapping["v0"],)
    assert all(
        action.module_a_id in plan.reserve_module_ids
        or action.module_b_id in plan.reserve_module_ids
        for action in plan.reserve_detach_actions
    )


def test_rc_car8_stows_all_four_wheels_before_reconfiguration() -> None:
    module_ids = tuple(f"smores_{index:02d}" for index in range(1, 9))
    source = _target("rc_car8")
    tree = target_graph_to_kinematic_tree(source)
    configured_mapping = {
        vertex: module_ids[index]
        for index, vertex in enumerate(tree.vertex_ids)
    }
    current = AttributedRobotGraph(
        nodes=tuple(_node(module_id) for module_id in module_ids),
        edges=tuple(
            _connection(
                configured_mapping[edge.vertex_a],
                edge.face_a,
                configured_mapping[edge.vertex_b],
                edge.face_b,
            )
            for edge in tree.edges
        ),
    )
    planner = SmoresSelfReconfigurationPlanner()
    source_assignment = planner.configuration_assignment(current, source)
    assert source_assignment is not None
    plan = planner.plan(
        current,
        _target("rc_car7"),
        source_graph=source,
        source_assignment=source_assignment,
    )
    mapping = source_assignment.target_to_module
    expected_groups = ((
        mapping["v3"],
        mapping["v4"],
        mapping["v5"],
        mapping["v6"],
    ),)

    assert plan.prepare_tilt_groups_by_module == expected_groups

    executor = SelfReconfigurationExecutor(plan, retry_count=0)
    first = executor.step(current_graph=current)
    assert first.primitive_goal is not None
    assert first.primitive_goal.module_ids[0] in expected_groups[0]
    assert first.primitive_goal.parameters["coordination_size"] == 4
    assert "max_servo_error_rad" not in first.primitive_goal.parameters

    goals = [first.primitive_goal]
    for _ in range(3):
        decision = executor.step(
            _accepted(goals[-1]),
            current_graph=current,
        )
        assert decision.primitive_goal is not None
        goals.append(decision.primitive_goal)

    assert {goal.module_ids[0] for goal in goals} == set(
        expected_groups[0]
    )
    assert {
        goal.parameters["coordination_group"] for goal in goals
    } == {goals[0].parameters["coordination_group"]}

    waiting = executor.step(
        _accepted(goals[-1]),
        current_graph=current,
    )
    assert waiting.primitive_goal is None
    assert set(waiting.active_goal_ids) == {
        goal.goal_id for goal in goals
    }


def test_holonomic9_to_manipulator8_releases_one_leaf_reserve() -> None:
    module_ids = tuple(f"smores_{index:02d}" for index in range(1, 10))
    source = _target("holonomic9")
    tree = target_graph_to_kinematic_tree(source)
    mapping = {
        vertex: module_ids[index]
        for index, vertex in enumerate(tree.vertex_ids)
    }
    current = AttributedRobotGraph(
        nodes=tuple(_node(module_id) for module_id in module_ids),
        edges=tuple(
            _connection(
                mapping[edge.vertex_a],
                edge.face_a,
                mapping[edge.vertex_b],
                edge.face_b,
            )
            for edge in tree.edges
        ),
    )
    planner = SmoresSelfReconfigurationPlanner()
    plan = planner.plan(
        current,
        _target("mobile_manipulator8"),
        source_graph=source,
        source_assignment=planner.configuration_assignment(current, source),
    )

    assert len(plan.reserve_module_ids) == 1
    assert len(plan.reserve_detach_actions) == 1
    assert len(plan.final_pan_by_module) == 1
    assert len(plan.final_tilt_by_module) == 7


def test_manipulator8_plus_one_reserve_can_target_holonomic9() -> None:
    module_ids = tuple(f"smores_{index:02d}" for index in range(1, 10))
    source = _target("mobile_manipulator8")
    tree = target_graph_to_kinematic_tree(source)
    mapping = {
        vertex: module_ids[index]
        for index, vertex in enumerate(tree.vertex_ids)
    }
    current = AttributedRobotGraph(
        nodes=tuple(_node(module_id) for module_id in module_ids),
        edges=tuple(
            _connection(
                mapping[edge.vertex_a],
                edge.face_a,
                mapping[edge.vertex_b],
                edge.face_b,
            )
            for edge in tree.edges
        ),
    )
    planner = SmoresSelfReconfigurationPlanner()
    source_assignment = planner.configuration_assignment(current, source)

    assert source_assignment is not None
    plan = planner.plan(
        current,
        _target("holonomic9"),
        source_graph=source,
        source_assignment=source_assignment,
    )

    assert plan.reserve_module_ids == ()
    assert len(plan.final_push_by_lifter_module) == 4
    assert module_ids[-1] in {
        module_id
        for stage in plan.stages
        if not stage.detach_actions
        for module_id in stage.mobile_module_ids
    }


def test_observed_holonomic_assignment_never_parks_internal_smores_05() -> None:
    """Regression for the live fold reported by the Isaac state graph."""

    source = _target("holonomic9")
    mapping = {
        "v0": "smores_01",
        "v1": "smores_03",
        "v2": "smores_09",
        "v3": "smores_07",
        "v4": "smores_05",
        "v5": "smores_04",
        "v6": "smores_02",
        "v7": "smores_08",
        "v8": "smores_06",
    }
    tree = target_graph_to_kinematic_tree(source)
    current = AttributedRobotGraph(
        nodes=tuple(_node(module_id) for module_id in mapping.values()),
        edges=tuple(
            _connection(
                mapping[edge.vertex_a],
                edge.face_a,
                mapping[edge.vertex_b],
                edge.face_b,
            )
            for edge in tree.edges
        ),
    )
    planner = SmoresSelfReconfigurationPlanner()
    plan = planner.plan(
        current,
        _target("rc_car7_reference"),
        source_graph=source,
        source_assignment=planner.configuration_assignment(current, source),
    )

    assert set(plan.reserve_module_ids) <= {
        "smores_02",
        "smores_04",
        "smores_06",
        "smores_08",
    }
    assert "smores_05" not in plan.reserve_module_ids


def test_manipulator7_plus_two_isolated_reserves_can_target_holonomic9() -> None:
    source = _target("mobile_manipulator7")
    current7 = _physical_graph_from_target(source)
    reserve_ids = ("smores_08", "smores_09")
    current = AttributedRobotGraph(
        nodes=current7.nodes + tuple(_node(item) for item in reserve_ids),
        edges=current7.edges,
    )
    planner = SmoresSelfReconfigurationPlanner()
    source_assignment = planner.configuration_assignment(current, source)
    assert source_assignment is not None

    plan = planner.plan(
        current,
        _target("holonomic9"),
        source_graph=source,
        source_assignment=source_assignment,
    )

    assert plan.reserve_module_ids == ()
    isolated_stage_modules = {
        module_id
        for stage in plan.stages
        if not stage.detach_actions
        for module_id in stage.mobile_module_ids
    }
    assert isolated_stage_modules == set(reserve_ids)
    assert len(plan.final_push_by_lifter_module) == 4
    expected_pushers = {
        plan.assignment.target_to_module[f"v{index}"]
        for index in range(5, 9)
    }
    expected_lifters = {
        plan.assignment.target_to_module[f"v{index}"]
        for index in range(1, 5)
    }
    assert set(plan.final_push_by_lifter_module) == expected_lifters
    assert {
        pusher
        for pusher, speed in plan.final_push_by_lifter_module.values()
        if speed == pytest.approx(0.025)
    } == expected_pushers


def test_snake_to_rc_car_assignment_avoids_crossing_free_modules() -> None:
    planner = SmoresSelfReconfigurationPlanner()
    current = _captured_snake_graph()
    source = _target("snake7")
    plan = planner.plan(
        current,
        _target("rc_car7"),
        source_graph=source,
        source_assignment=planner.configuration_assignment(current, source),
    )

    # The topology-only lexicographic assignment crossed the two end modules:
    # smores_05 was sent to the right chassis and smores_02 to the left.  The
    # live-pose tie-break keeps each detached module on its nearest side.
    assert plan.assignment.target_to_module == {
        "v0": "smores_01",
        "v1": "smores_06",
        "v2": "smores_03",
        "v3": "smores_04",
        "v4": "smores_05",
        "v5": "smores_02",
        "v6": "smores_07",
    }
    assert [
        len(wave.actions) for wave in plan.assembly_plan.waves
    ] == [2, 2]
    assert [stage.mobile_module_ids for stage in plan.stages] == [
        ("smores_02", "smores_05"),
        ("smores_07", "smores_04"),
    ]
    assert [stage.source_depth for stage in plan.stages] == [2, 1]
    assert [
        tuple(
            (action.mobile_module_id, action.parent_module_id)
            for action in stage.assembly_plan.all_actions
        )
        for stage in plan.stages
    ] == [
        (
            ("smores_02", "smores_03"),
            ("smores_05", "smores_06"),
        ),
        (
            ("smores_07", "smores_03"),
            ("smores_04", "smores_06"),
        ),
    ]


def test_parallel_wave_uses_collective_motion_barriers() -> None:
    planner = SmoresSelfReconfigurationPlanner()
    current = _captured_snake_graph()
    source = _target("snake7")
    plan = planner.plan(
        current,
        _target("rc_car7"),
        source_graph=source,
        source_assignment=planner.configuration_assignment(current, source),
    )
    executor = SelfReconfigurationExecutor(plan, retry_count=0)
    status = None
    first_align = None

    for _ in range(80):
        decision = executor.step(status, current_graph=current)
        status = None
        goal = decision.primitive_goal
        if goal is None:
            continue
        if goal.primitive == "align_faces":
            first_align = goal
            break
        status = _succeeded(goal)

    assert first_align is not None
    assert first_align.parameters["execution_phase"] == "reach"
    second_decision = executor.step(
        _accepted(first_align),
        current_graph=current,
    )
    second_reach = second_decision.primitive_goal
    assert second_reach is not None
    assert second_reach.primitive == "align_faces"
    assert second_reach.parameters["execution_phase"] == "reach"

    one_peer_done = executor.step(
        {
            "statuses": [
                _succeeded(first_align),
                _accepted(second_reach),
            ]
        },
        current_graph=current,
    )
    assert one_peer_done.primitive_goal is None

    first_alignment = executor.step(
        _succeeded(second_reach),
        current_graph=current,
    ).primitive_goal
    assert first_alignment is not None
    assert first_alignment.primitive == "align_faces"
    assert first_alignment.parameters["execution_phase"] == "align"

    second_align = executor.step(
        _accepted(first_alignment),
        current_graph=current,
    ).primitive_goal
    assert second_align is not None
    assert second_align.primitive == "align_faces"
    assert second_align.module_ids[0] != first_align.module_ids[0]


def test_rc_car_to_manipulator_parallelizes_diverging_source_siblings() -> None:
    planner = SmoresSelfReconfigurationPlanner()
    source = _target("rc_car7")
    current = _physical_graph_with_target_layout(source)
    plan = planner.plan(
        current,
        _target("mobile_manipulator7"),
        source_graph=source,
        source_assignment=planner.configuration_assignment(current, source),
    )

    assert len(plan.stages) == 1
    assert plan.stages[0].mobile_module_ids == (
        "smores_05",
        "smores_04",
    )
    first_detach, second_detach = plan.stages[0].detach_actions
    shared_source = {
        first_detach.module_a_id,
        first_detach.module_b_id,
    } & {
        second_detach.module_a_id,
        second_detach.module_b_id,
    }
    assert shared_source == {"smores_02"}
    assert planner._detach_face(
        first_detach,
        "smores_02",
    ) != planner._detach_face(second_detach, "smores_02")


def test_parallel_clearance_can_force_the_safe_serial_fallback() -> None:
    planner = SmoresSelfReconfigurationPlanner(
        parallel_path_clearance_m=1.0,
    )
    current = _captured_snake_graph()
    source = _target("snake7")
    plan = planner.plan(
        current,
        _target("rc_car7"),
        source_graph=source,
        source_assignment=planner.configuration_assignment(current, source),
    )

    assert [len(stage.mobile_module_ids) for stage in plan.stages] == [
        1,
        1,
        1,
        1,
    ]


def test_parallel_wave_has_no_artificial_two_module_cap() -> None:
    planner = SmoresSelfReconfigurationPlanner()
    retained = ("root_0", "root_1", "root_2")
    movers = ("mobile_0", "mobile_1", "mobile_2")
    retained_edges = (
        _connection("root_0", "TOP", "root_1", "BOTTOM"),
        _connection("root_1", "TOP", "root_2", "BOTTOM"),
    )
    moving_edges = tuple(
        _connection(module_id, "BOTTOM", parent_id, "LEFT")
        for module_id, parent_id in zip(movers, retained, strict=True)
    )
    detach_actions = tuple(
        ReconfigurationDetachAction(
            edge.module_a_id,
            "BOTTOM",
            edge.module_b_id,
            "LEFT",
        )
        for edge in moving_edges
    )
    actions = tuple(
        AssemblyAction(
            mobile_module_id=module_id,
            mobile_face="BOTTOM",
            parent_module_id=parent_id,
            parent_face="RIGHT",
            mobile_target_vertex=f"target_mobile_{index}",
            parent_target_vertex=f"target_parent_{index}",
            depth=2,
            clocking_quarter_turns=0,
            requires_helper=False,
        )
        for index, (module_id, parent_id) in enumerate(
            zip(movers, retained, strict=True)
        )
    )
    physical_poses = {
        **{
            parent_id: PlanarPose(index * 0.5, 0.8, 0.0)
            for index, parent_id in enumerate(retained)
        },
        **{
            module_id: PlanarPose(index * 0.5, 0.0, 0.0)
            for index, module_id in enumerate(movers)
        },
    }
    target_xy = {
        **{
            parent_id: (index * 0.5, 0.9)
            for index, parent_id in enumerate(retained)
        },
        **{
            module_id: (index * 0.5, 1.0)
            for index, module_id in enumerate(movers)
        },
    }

    waves, _, _ = planner._progressive_action_waves(
        retained + movers,
        retained_edges + moving_edges,
        set(retained),
        detach_actions,
        actions,
        physical_poses,
        target_xy,
    )

    assert [[action.mobile_module_id for action in wave] for wave in waves] == [
        ["mobile_0", "mobile_1", "mobile_2"],
    ]


def test_fixed_assignment_verifies_complete_snake_and_rejects_rc_car() -> None:
    planner = SmoresSelfReconfigurationPlanner()
    plan = planner.plan(_rc_car_graph(), _snake_target())

    assert not planner.target_reached(_rc_car_graph(), plan)
    assert planner.target_reached(_physical_target_graph(plan), plan)


def test_source_morphology_must_match_complete_rc_car_topology() -> None:
    planner = SmoresSelfReconfigurationPlanner()
    source_target = load_attributed_graph(
        CONFIG_ROOT / "smores_rc_car7.json"
    )

    assignment = planner.configuration_assignment(
        _rc_car_graph(),
        source_target,
    )

    assert assignment is not None
    assert planner.configuration_assignment(
        _physical_target_graph(
            planner.plan(_rc_car_graph(), _snake_target())
        ),
        source_target,
    ) is None


@pytest.mark.parametrize(
    "source_name",
    ("snake7", "rc_car7", "mobile_manipulator7"),
)
def test_each_known_source_topology_has_one_catalog_match(
    source_name: str,
) -> None:
    planner = SmoresSelfReconfigurationPlanner()
    current = _physical_graph_from_target(_target(source_name))
    matches = [
        candidate
        for candidate in (
            "snake7",
            "rc_car7",
            "mobile_manipulator7",
        )
        if planner.configuration_assignment(
            current,
            _target(candidate),
        )
        is not None
    ]

    assert matches == [source_name]


@pytest.mark.parametrize(
    "source_name",
    ("rc_car8", "snake8", "bridge8", "mobile_manipulator8"),
)
def test_each_roadmap_source_topology_has_one_catalog_match(
    source_name: str,
) -> None:
    planner = SmoresSelfReconfigurationPlanner()
    current = _physical_graph_from_target(_target(source_name))
    candidates = ("rc_car8", "snake8", "bridge8", "mobile_manipulator8")
    matches = [
        candidate
        for candidate in candidates
        if planner.configuration_assignment(current, _target(candidate))
        is not None
    ]

    assert matches == [source_name]


def test_connected_snake8_is_not_misidentified_as_snake7() -> None:
    planner = SmoresSelfReconfigurationPlanner()
    current = _physical_graph_from_target(_target("snake8"))

    assert planner.configuration_assignment(
        current,
        _target("snake8"),
    ) is not None
    assert planner.configuration_assignment(
        current,
        _target("snake7"),
    ) is None


def test_source_topology_matching_scales_beyond_seven_modules() -> None:
    module_count = 20
    module_ids = tuple(f"module_{index:02d}" for index in range(module_count))
    target_ids = tuple(f"v{index:02d}" for index in range(module_count))
    current = AttributedRobotGraph(
        nodes=tuple(_node(module_id) for module_id in module_ids),
        edges=tuple(
            _connection(
                module_ids[index],
                "TOP",
                module_ids[index + 1],
                "BOTTOM",
            )
            for index in range(module_count - 1)
        ),
    )
    target = AttributedRobotGraph(
        nodes=tuple(
            GraphNode(
                target_id,
                {
                    "node_type": "target_slot",
                    "is_target_node": True,
                    "target_vertex_id": target_id,
                    "is_target_root": index == module_count // 2,
                },
            )
            for index, target_id in enumerate(target_ids)
        ),
        edges=tuple(
            GraphEdge(
                target_ids[index],
                target_ids[index + 1],
                {
                    "relation_type": "target_connection",
                    "is_target_edge": True,
                    "face_a": "TOP",
                    "face_b": "BOTTOM",
                },
            )
            for index in range(module_count - 1)
        ),
        global_attributes={
            "target_root_vertex": target_ids[module_count // 2],
        },
    )

    assignment = SmoresSelfReconfigurationPlanner().configuration_assignment(
        current,
        target,
    )

    assert assignment is not None
    assert len(assignment.target_to_module) == module_count


@pytest.mark.parametrize(
    (
        "source_name",
        "target_name",
        "retained",
        "wave_sizes",
        "prepare_count",
        "final_count",
    ),
    (
        ("snake7", "rc_car7", 2, [1, 1, 1, 1], 7, 4),
        ("snake7", "mobile_manipulator7", 4, [1, 1], 7, 6),
        ("rc_car7", "snake7", 2, [1, 1, 1, 1], 4, 0),
        ("rc_car7", "mobile_manipulator7", 4, [1, 1], 4, 6),
        ("mobile_manipulator7", "snake7", 4, [1, 1], 5, 0),
        ("mobile_manipulator7", "rc_car7", 4, [1, 1], 5, 4),
    ),
)
def test_every_known_morphology_transition_is_plannable(
    source_name: str,
    target_name: str,
    retained: int,
    wave_sizes: list[int],
    prepare_count: int,
    final_count: int,
) -> None:
    planner = SmoresSelfReconfigurationPlanner()
    source = _target(source_name)
    target = _target(target_name)
    current = _physical_graph_from_target(source)
    source_assignment = planner.configuration_assignment(current, source)

    plan = planner.plan(
        current,
        target,
        source_graph=source,
        source_assignment=source_assignment,
    )

    assert plan.source_morphology == source_name
    assert plan.target_morphology == target_name
    assert plan.retained_connection_count == retained
    assert len(plan.detach_actions) == 6 - retained
    assert plan.new_connection_count == 6 - retained
    assert [
        len(wave.actions) for wave in plan.assembly_plan.waves
    ] == wave_sizes
    assert len(plan.prepare_tilt_by_module) == prepare_count
    assert len(plan.final_tilt_by_module) == final_count
    assert len(plan.final_pan_by_module) == (
        1 if target_name == "mobile_manipulator7" else 0
    )
    assert not plan.assembly_plan.requires_helper


@pytest.mark.parametrize(
    ("source_name", "target_name", "expected_wave_sizes"),
    (
        ("snake7", "rc_car7", [2, 2]),
        ("snake7", "mobile_manipulator7", [1, 1]),
        ("rc_car7", "snake7", [2, 2]),
        ("rc_car7", "mobile_manipulator7", [2]),
        ("mobile_manipulator7", "snake7", [1, 1]),
        ("mobile_manipulator7", "rc_car7", [2]),
    ),
)
def test_known_transition_staging_goals_are_not_occupied(
    source_name: str,
    target_name: str,
    expected_wave_sizes: list[int],
) -> None:
    planner = SmoresSelfReconfigurationPlanner()
    source = _target(source_name)
    target = _target(target_name)
    current = _physical_graph_with_target_layout(source)
    plan = planner.plan(
        current,
        target,
        source_graph=source,
        source_assignment=planner.configuration_assignment(current, source),
    )
    assert [
        len(stage.mobile_module_ids) for stage in plan.stages
    ] == expected_wave_sizes
    target_tree = target_graph_to_kinematic_tree(target)
    rooted_target = root_kinematic_tree(
        target_tree,
        root_id=ParallelSelfAssemblyPlanner._declared_target_root(target),
    )
    unfolded_target = unfold_tree_on_plane(rooted_target)
    current_xy = {
        node.module_id: tuple(node.attributes["position"][:2])
        for node in current.nodes
    }
    root_module = plan.assignment.target_to_module[rooted_target.root_id]
    root_xy = current_xy[root_module]
    target_pose_by_module = {
        plan.assignment.target_to_module[vertex]: (
            root_xy[0] + pose.x_m,
            root_xy[1] + pose.y_m,
            pose.yaw_rad,
        )
        for vertex, pose in unfolded_target.poses_by_vertex.items()
    }

    for wave in plan.assembly_plan.waves:
        wave_modules = {
            action.mobile_module_id for action in wave.actions
        }
        for wave_action in wave.actions:
            target_pose = target_pose_by_module[
                wave_action.mobile_module_id
            ]
            parent_pose = target_pose_by_module[
                wave_action.parent_module_id
            ]
            direction = (
                parent_pose[2]
                + FACE_ANGLE_RAD[wave_action.parent_face]
            )
            staging_xy = (
                target_pose[0] + 0.070 * math.cos(direction),
                target_pose[1] + 0.070 * math.sin(direction),
            )
            blockers = sorted(
                module_id
                for module_id, position in current_xy.items()
                if module_id not in wave_modules
                and math.dist(staging_xy, position) < 0.110 - 1.0e-6
            )
            assert blockers == []
        for wave_action in wave.actions:
            current_xy[wave_action.mobile_module_id] = (
                target_pose_by_module[wave_action.mobile_module_id][:2]
            )


@pytest.mark.parametrize(
    ("source_name", "target_name"),
    (
        ("snake7", "rc_car7"),
        ("snake7", "mobile_manipulator7"),
        ("rc_car7", "snake7"),
        ("rc_car7", "mobile_manipulator7"),
        ("mobile_manipulator7", "snake7"),
        ("mobile_manipulator7", "rc_car7"),
        ("rc_car8", "snake8"),
        ("rc_car8", "bridge8"),
        ("rc_car8", "mobile_manipulator8"),
        ("snake8", "rc_car8"),
        ("snake8", "bridge8"),
        ("bridge8", "snake8"),
        ("bridge8", "rc_car8"),
        ("bridge8", "mobile_manipulator8"),
        ("snake8", "mobile_manipulator8"),
        ("mobile_manipulator8", "rc_car8"),
        ("mobile_manipulator8", "snake8"),
        ("mobile_manipulator8", "bridge8"),
    ),
)
def test_every_known_morphology_transition_executes_to_verification(
    source_name: str,
    target_name: str,
) -> None:
    planner = SmoresSelfReconfigurationPlanner()
    source = _target(source_name)
    current = _physical_graph_from_target(source)
    plan = planner.plan(
        current,
        _target(target_name),
        source_graph=source,
        source_assignment=planner.configuration_assignment(
            current,
            source,
        ),
    )
    executor = SelfReconfigurationExecutor(plan, retry_count=0)
    status = None

    for _ in range(160):
        decision = executor.step(status, current_graph=current)
        status = None
        if decision.primitive_goal is not None:
            status = _succeeded(decision.primitive_goal)
        if decision.phase == "VERIFY":
            break
    else:
        raise AssertionError(
            f"{source_name} -> {target_name} did not reach verification."
        )

    verified = executor.step(
        current_graph=_physical_target_graph(plan),
    )
    assert verified.done
    assert verified.success
    assert (
        verified.completed_operation_count
        == verified.total_operation_count
    )

    if source_name.endswith("8") and target_name.endswith("8"):
        assert plan.reserve_module_ids == ()
        assert len(plan.assignment.target_to_module) == 8


def test_snake8_to_bridge8_starts_with_undock_without_a_tilt() -> None:
    planner = SmoresSelfReconfigurationPlanner()
    source = _target("snake8")
    current = _physical_graph_from_target(source)
    plan = planner.plan(
        current,
        _target("bridge8"),
        source_graph=source,
        source_assignment=planner.configuration_assignment(
            current,
            source,
        ),
    )

    assert plan.prepare_tilt_by_module == {}
    assert len(plan.detach_actions) == 1
    assert plan.assembly_plan.action_count == 1

    executor = SelfReconfigurationExecutor(plan, retry_count=0)
    first = executor.step(current_graph=current)

    assert first.primitive_goal is not None
    assert first.primitive_goal.primitive == "undock"


def test_executor_stows_undocks_redocks_and_verifies_graph() -> None:
    planner = SmoresSelfReconfigurationPlanner()
    source = _target("rc_car7")
    plan = planner.plan(
        _rc_car_graph(),
        _snake_target(),
        source_graph=source,
        source_assignment=planner.configuration_assignment(
            _rc_car_graph(), source
        ),
    )
    executor = SelfReconfigurationExecutor(plan, retry_count=0)
    status = None
    emitted = []
    face_execution_phases = []
    prepare_goals = []
    decision = None

    for _ in range(100):
        decision = executor.step(status, current_graph=_rc_car_graph())
        status = None
        if decision.primitive_goal is not None:
            emitted.append(decision.primitive_goal.primitive)
            if decision.primitive_goal.primitive == "align_faces":
                face_execution_phases.append(
                    decision.primitive_goal.parameters["execution_phase"]
                )
            if decision.primitive_goal.primitive == "set_tilt":
                prepare_goals.append(decision.primitive_goal)
            status = _succeeded(decision.primitive_goal)
        if decision.phase == "VERIFY":
            break

    assert decision is not None
    assert decision.phase == "VERIFY"
    assert emitted.count("set_tilt") == 4
    assert emitted.count("undock") == 4
    assert emitted.count("align_faces") == 12
    assert emitted.count("dock") == 4
    assert face_execution_phases.count("reach") == 4
    assert face_execution_phases.count("align") == 4
    assert face_execution_phases.count("approach") == 4
    assert {
        goal.parameters["coordination_group"] for goal in prepare_goals
    } == {"self-reconfiguration-prepare-group-0"}
    assert all(
        goal.parameters["coordination_size"] == 4
        for goal in prepare_goals
    )
    assert not decision.done

    verified = executor.step(
        current_graph=_physical_target_graph(plan),
    )
    assert verified.done
    assert verified.success
    assert verified.completed_operation_count == 12


def test_reverse_transition_applies_rc_car_operational_posture() -> None:
    planner = SmoresSelfReconfigurationPlanner()
    source = _target("snake7")
    current = _physical_graph_from_target(source)
    plan = planner.plan(
        current,
        _target("rc_car7"),
        source_graph=source,
        source_assignment=planner.configuration_assignment(
            current,
            source,
        ),
    )
    executor = SelfReconfigurationExecutor(plan, retry_count=0)
    status = None
    tilt_goals = []

    for _ in range(140):
        decision = executor.step(status, current_graph=current)
        status = None
        if decision.primitive_goal is not None:
            if decision.primitive_goal.primitive == "set_tilt":
                tilt_goals.append(decision.primitive_goal)
            status = _succeeded(decision.primitive_goal)
        if decision.phase == "VERIFY":
            break
    else:
        raise AssertionError("Executor did not reach graph verification.")

    assert len(tilt_goals) == 11
    final_goals = [
        goal
        for goal in tilt_goals
        if goal.parameters["angle_rad"] < 0.0
    ]
    assert len(final_goals) == 4
    assert all(
        goal.parameters["coordination_size"] == 4
        for goal in final_goals
    )
    assert len(
        {
            str(goal.parameters["coordination_group"])
            for goal in final_goals
        }
    ) == 1
    assert all(
        goal.parameters["tolerance_rad"] == pytest.approx(0.08)
        for goal in final_goals
    )
    assert plan.final_tilt_groups_by_module == ()

    verified = executor.step(
        current_graph=_physical_target_graph(plan),
    )
    assert verified.done
    assert verified.success
    assert verified.completed_operation_count == 19


def test_reconfiguration_congestion_cost_only_reassigns_free_movers() -> None:
    topology = SmoresKinematicTree(
        vertex_ids=("v0", "v1", "v2", "v3"),
        edges=(
            SmoresTopologyEdge("v0", "TOP", "v1", "BOTTOM"),
            SmoresTopologyEdge("v1", "TOP", "v2", "BOTTOM"),
            SmoresTopologyEdge("v2", "TOP", "v3", "BOTTOM"),
        ),
    )
    rooted = root_kinematic_tree(topology, root_id="v0")
    unfolded = unfold_tree_on_plane(rooted)
    current_edges = (
        _connection("m0", "TOP", "m1", "BOTTOM"),
    )
    poses = {
        "m0": PlanarPose(0.0, 0.0, 0.0),
        "m1": PlanarPose(0.07777, 0.0, 0.0),
        "m2": PlanarPose(0.15554, 0.25, 0.0),
        "m3": PlanarPose(0.270, 0.0, 0.0),
    }

    assignment, retained = (
        SmoresSelfReconfigurationPlanner()._maximum_common_assignment(
            physical_module_ids=("m0", "m1", "m2", "m3"),
            current_edges=current_edges,
            target_tree=topology,
            target_root="v0",
            physical_poses=poses,
            unfolded_target=unfolded,
        )
    )

    assert assignment.target_to_module["v0"] == "m0"
    assert assignment.target_to_module["v1"] == "m1"
    assert assignment.target_to_module["v2"] == "m3"
    assert assignment.target_to_module["v3"] == "m2"
    assert assignment.total_future_blockers == 0
    assert len(retained) == 1
