"""Tests for the example SMORES target morphology."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from mssr_expert.execution.assembly_policy import (
    TARGET_EXECUTION_POLICY_FIELDS,
)
from mssr_expert.graph.serialization import (
    load_attributed_graph,
)
from mssr_expert.behaviors.morphology_library import (
    AssignedModule,
    MorphologyLibrary,
)
from mssr_expert.planning.smores_ep.attributed_adapter import (
    target_roles_from_graph,
    target_graph_to_kinematic_tree,
)
from mssr_expert.planning.smores_ep.rooting import (
    root_kinematic_tree,
)
from mssr_expert.planning.smores_ep.assignment import AssignmentResult
from mssr_expert.planning.smores_ep.assembly_sequence import (
    generate_parallel_assembly_plan,
)


def test_three_module_target_is_valid() -> None:
    package_root = Path(__file__).parents[1]

    graph = load_attributed_graph(
        package_root
        / "config"
        / "smores_three_module_chain.json"
    )

    tree = target_graph_to_kinematic_tree(
        graph
    )

    rooted = root_kinematic_tree(tree)

    assert tree.vertex_ids == (
        "v0",
        "v1",
        "v2",
    )

    assert rooted.root_id == "v1"
    assert len(tree.edges) == 2


@pytest.mark.parametrize(
    ("filename", "expected_root", "requires_helper"),
    (
        ("smores_snake7.json", "v3", False),
        ("smores_mobile_manipulator7.json", "v4", False),
        ("smores_rc_car7.json", "v0", False),
    ),
)
def test_seven_module_reference_targets_are_valid(
    filename: str,
    expected_root: str,
    requires_helper: bool,
) -> None:
    package_root = Path(__file__).parents[1]
    graph = load_attributed_graph(package_root / "config" / filename)
    tree = target_graph_to_kinematic_tree(graph)
    rooted = root_kinematic_tree(tree)

    assert len(tree.vertex_ids) == 7
    assert len(tree.edges) == 6
    assert rooted.root_id == expected_root
    assignment = AssignmentResult(
        target_to_module={vertex: vertex for vertex in rooted.vertex_ids},
        cost_by_target={vertex: 0.0 for vertex in rooted.vertex_ids},
        total_cost=0.0,
    )
    plan = generate_parallel_assembly_plan(rooted, assignment)

    assert plan.requires_helper is requires_helper
    assert graph.global_attributes["requires_helping_module"] is requires_helper


def test_eight_module_manipulator_has_grounded_first_arm_module() -> None:
    package_root = Path(__file__).parents[1]
    graph = load_attributed_graph(
        package_root / "config" / "smores_mobile_manipulator8.json"
    )
    tree = target_graph_to_kinematic_tree(graph)
    rooted = root_kinematic_tree(tree)
    roles = target_roles_from_graph(graph)

    assert len(tree.vertex_ids) == 8
    assert len(tree.edges) == 7
    assert roles["v0"]["is_target_root"] is True
    assert roles["v4"]["target_role"] == "arm_ground_drive"
    assert graph.global_attributes["post_assembly_tilt_rad_by_vertex"][
        "v4"
    ] == pytest.approx(0.0)
    assert graph.global_attributes["post_assembly_tilt_rad_by_vertex"][
        "v5"
    ] == pytest.approx(0.75)
    assert graph.global_attributes["post_assembly_tilt_rad_by_vertex"][
        "v6"
    ] == pytest.approx(0.65)


def test_eight_module_rc_car_has_four_module_centerline() -> None:
    package_root = Path(__file__).parents[1]
    graph = load_attributed_graph(
        package_root / "config" / "smores_rc_car8.json"
    )
    tree = target_graph_to_kinematic_tree(graph)
    rooted = root_kinematic_tree(tree)
    roles = target_roles_from_graph(graph)

    assert len(tree.vertex_ids) == 8
    assert len(tree.edges) == 7
    assert rooted.root_id == "v0"
    assert {
        roles[vertex]["target_role"]
        for vertex in ("v0", "v1", "v2", "v7")
    } == {
        "chassis_center_left",
        "chassis_left",
        "chassis_center_right",
        "chassis_right",
    }
    wheel_edges = [
        edge
        for edge in tree.edges
        if edge.vertex_a in {"v3", "v4", "v5", "v6"}
        or edge.vertex_b in {"v3", "v4", "v5", "v6"}
    ]
    assert len(wheel_edges) == 4
    assert graph.global_attributes[
        "post_assembly_tilt_groups_by_vertex"
    ] == [["v3", "v4", "v5", "v6"]]
    assert graph.global_attributes[
        "pre_reconfiguration_tilt_groups_by_vertex"
    ] == [["v3", "v4", "v5", "v6"]]
    assert (
        "post_assembly_tilt_max_servo_error_rad"
        not in graph.global_attributes
    )
    assert graph.global_attributes[
        "post_assembly_tilt_tolerance_rad"
    ] == pytest.approx(0.12)
    assert (
        "pre_reconfiguration_tilt_max_servo_error_rad"
        not in graph.global_attributes
    )


@pytest.mark.parametrize(
    ("filename", "expected_root"),
    (
        ("smores_snake8.json", "v3"),
        ("smores_bridge8.json", "v3"),
    ),
)
def test_eight_module_obstacle_targets_use_every_module(
    filename: str,
    expected_root: str,
) -> None:
    package_root = Path(__file__).parents[1]
    graph = load_attributed_graph(package_root / "config" / filename)
    tree = target_graph_to_kinematic_tree(graph)
    rooted = root_kinematic_tree(tree)

    assert len(tree.vertex_ids) == 8
    assert len(tree.edges) == 7
    assert rooted.root_id == expected_root
    assert graph.global_attributes["module_count"] == 8
    assert graph.global_attributes["requires_helping_module"] is False


def test_bridge_uses_top_face_on_both_ground_supports() -> None:
    package_root = Path(__file__).parents[1]
    graph = load_attributed_graph(
        package_root / "config" / "smores_bridge8.json"
    )
    tree = target_graph_to_kinematic_tree(graph)
    endpoint_faces = {}
    for edge in tree.edges:
        endpoint_faces.setdefault(edge.vertex_a, []).append(edge.face_a)
        endpoint_faces.setdefault(edge.vertex_b, []).append(edge.face_b)

    assert endpoint_faces["v0"] == ["TOP"]
    assert endpoint_faces["v7"] == ["TOP"]
    assert any(
        {edge.vertex_a, edge.vertex_b} == {"v6", "v7"}
        and edge.face_a == "TOP"
        and edge.face_b == "TOP"
        for edge in tree.edges
    )


def test_rc_car_uses_two_progressive_waves_and_no_helper() -> None:
    package_root = Path(__file__).parents[1]
    graph = load_attributed_graph(
        package_root / "config" / "smores_rc_car7.json"
    )
    assert graph.global_attributes[
        "post_assembly_tilt_tolerance_rad"
    ] == pytest.approx(0.08)
    rooted = root_kinematic_tree(target_graph_to_kinematic_tree(graph))
    assignment = AssignmentResult(
        target_to_module={vertex: vertex for vertex in rooted.vertex_ids},
        cost_by_target={vertex: 0.0 for vertex in rooted.vertex_ids},
        total_cost=0.0,
    )
    plan = generate_parallel_assembly_plan(rooted, assignment)

    assert [len(wave.actions) for wave in plan.waves] == [2, 4]
    assert not any(action.requires_helper for action in plan.all_actions)
    assert graph.global_attributes["assembly_strategy"] == (
        "progressive_planar_waves_then_fold"
    )
    tilt_targets = graph.global_attributes[
        "post_assembly_tilt_rad_by_vertex"
    ]
    assert len(tilt_targets) == 4
    assert set(tilt_targets) == {"v3", "v4", "v5", "v6"}
    assert all(
        angle == pytest.approx(-math.radians(45.0))
        for angle in tilt_targets.values()
    )


@pytest.mark.parametrize(
    "filename",
    (
        "smores_snake7.json",
        "smores_snake8.json",
        "smores_bridge8.json",
        "smores_rc_car7.json",
        "smores_rc_car8.json",
        "smores_mobile_manipulator7.json",
        "smores_mobile_manipulator8.json",
        "smores_holonomic9.json",
    ),
)
def test_target_graphs_do_not_override_execution_policy(
    filename: str,
) -> None:
    package_root = Path(__file__).parents[1]
    graph = load_attributed_graph(package_root / "config" / filename)

    assert not (
        TARGET_EXECUTION_POLICY_FIELDS
        & set(graph.global_attributes)
    )


def test_mobile_manipulator_folds_to_paper_operational_posture() -> None:
    package_root = Path(__file__).parents[1]
    graph = load_attributed_graph(
        package_root / "config" / "smores_mobile_manipulator7.json"
    )

    tilt_targets = graph.global_attributes[
        "post_assembly_tilt_rad_by_vertex"
    ]
    assert tilt_targets == {
        "v0": pytest.approx(0.0),
        "v1": pytest.approx(-0.20),
        "v2": pytest.approx(0.0),
        "v3": pytest.approx(-0.20),
        "v4": pytest.approx(0.75),
        "v5": pytest.approx(0.65),
    }
    assert graph.global_attributes["post_assembly_pan_rad_by_vertex"] == {
        "v6": pytest.approx(0.0)
    }
    assert graph.global_attributes[
        "post_assembly_tilt_groups_by_vertex"
    ] == [["v0", "v1", "v3"], ["v5"], ["v4"], ["v2"]]
    assert "passive_post_assembly_tilt_vertices" not in graph.global_attributes
    assert graph.global_attributes[
        "post_assembly_tilt_tolerance_rad"
    ] == pytest.approx(0.2)


def test_holonomic_outer_modules_push_inward_as_one_fold_group() -> None:
    package_root = Path(__file__).parents[1]
    graph = load_attributed_graph(
        package_root / "config" / "smores_holonomic9.json"
    )

    assert graph.global_attributes["coordinate_post_assembly_tilts"] is True
    assert graph.global_attributes[
        "post_assembly_tilt_groups_by_vertex"
    ] == [["v1", "v2", "v3", "v4"]]
    assert "passive_post_assembly_tilt_vertices" not in graph.global_attributes
    assert graph.global_attributes[
        "post_assembly_push_pairs_by_vertex"
    ] == [
        {
            "pusher_vertex": f"v{index + 4}",
            "lifter_vertex": f"v{index}",
            "linear_m_s": pytest.approx(0.025),
        }
        for index in range(1, 5)
    ]
    assert graph.global_attributes[
        "post_assembly_tilt_rad_by_vertex"
    ] == {
        vertex: pytest.approx(-1.35)
        for vertex in ("v1", "v2", "v3", "v4")
    }


@pytest.mark.parametrize(
    ("filename", "expected_vertices"),
    (
        ("smores_snake7.json", {f"v{index}" for index in range(7)}),
        ("smores_rc_car7.json", {"v3", "v4", "v5", "v6"}),
        ("smores_rc_car8.json", {"v3", "v4", "v5", "v6"}),
        (
            "smores_mobile_manipulator7.json",
            {"v1", "v2", "v3", "v4", "v5"},
        ),
        (
            "smores_mobile_manipulator8.json",
            {"v1", "v2", "v3", "v4", "v5", "v6"},
        ),
    ),
)
def test_known_morphologies_declare_neutral_reconfiguration_posture(
    filename: str,
    expected_vertices: set[str],
) -> None:
    package_root = Path(__file__).parents[1]
    graph = load_attributed_graph(package_root / "config" / filename)
    posture = graph.global_attributes[
        "pre_reconfiguration_tilt_rad_by_vertex"
    ]

    assert set(posture) == expected_vertices
    assert all(angle == pytest.approx(0.0) for angle in posture.values())


@pytest.mark.parametrize(
    "filename",
    (
        "smores_snake7.json",
        "smores_mobile_manipulator7.json",
        "smores_mobile_manipulator8.json",
        "smores_rc_car7.json",
        "smores_rc_car8.json",
    ),
)
def test_post_assembly_fold_matches_operational_ready_posture(
    filename: str,
) -> None:
    package_root = Path(__file__).parents[1]
    graph = load_attributed_graph(package_root / "config" / filename)
    morphology = str(graph.global_attributes["morphology_name"])
    roles = target_roles_from_graph(graph)
    assignments = tuple(
        AssignedModule(
            module_id=vertex,
            target_vertex_id=vertex,
            target_role=str(attributes["target_role"]),
        )
        for vertex, attributes in sorted(roles.items())
    )
    library = MorphologyLibrary.load(
        package_root / "config" / "smores_morphology_behaviors.json"
    )
    expected_tilts = {
        target.target_vertex_id: target.angle_rad
        for target in library.ready_joint_targets(morphology, assignments)
        if target.joint == "tilt"
    }
    configured = graph.global_attributes.get(
        "post_assembly_tilt_rad_by_vertex",
        {},
    )

    assert set(configured) <= set(expected_tilts)
    for vertex, angle in expected_tilts.items():
        assert configured.get(vertex, 0.0) == pytest.approx(angle)
