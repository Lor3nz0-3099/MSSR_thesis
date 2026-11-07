"""CPU regressions for complete guided topology and measured folding."""

from dataclasses import replace
import math

import pytest

from drl.MARL.c1_guided_feedback import load_target
from drl.MARL.c1_guided_success import GuidedAssemblyGate as gate
from drl.MARL.c1_structure import C1Connection, C1Frame, C1Module


def measured_target(target, *, folded=True, owners=None):
    vertices = tuple(node["node_id"] for node in target["nodes"])
    owners = owners or tuple(f"owner_{index}" for index in range(len(vertices)))
    mapping = dict(zip(vertices, owners))
    attrs = target.get("global_attributes", {})
    modules = []
    for vertex, owner in mapping.items():
        tilt = attrs.get("post_assembly_tilt_rad_by_vertex", {}).get(vertex, 0.0)
        pan = attrs.get("post_assembly_pan_rad_by_vertex", {}).get(vertex, 0.0)
        modules.append(C1Module(owner, (0.0, 0.0, 0.1), joints=(
            ("pan.position_rad", pan if folded else 0.0),
            ("tilt.position_rad", tilt if folded else 0.0),
        )))
    connections = tuple(C1Connection(
        f"joint_{index}", mapping[item["module_a_id"]], item["attributes"]["face_a"],
        mapping[item["module_b_id"]], item["attributes"]["face_b"], True,
    ) for index, item in enumerate(target["edges"]))
    return C1Frame(0.0, tuple(modules), connections), owners, mapping


@pytest.mark.parametrize("domain", ["navigation", "stairs", "gap", "manipulation"])
def test_measured_catalog_target_is_complete_with_permuted_physical_ids(domain):
    target = load_target(domain)
    frame, owners, mapping = measured_target(target, owners=tuple(f"robot_{i}" for i in (6, 1, 4, 3, 7, 0, 2, 5)))
    status = gate(target, owners).assess(frame)
    assert status.target_graph_complete
    assert status.fold_complete
    assert set(status.assignment) == set(mapping)
    assert set(status.assignment.values()) == set(owners)


def test_flat_rc_car_has_complete_topology_but_incomplete_fold():
    target = load_target("navigation")
    frame, owners, _ = measured_target(target, folded=False)
    status = gate(target, owners).assess(frame)
    assert status.target_graph_complete
    assert not status.fold_complete


@pytest.mark.parametrize("damage", ["missing", "extra", "wrong_face", "invalid_joint", "duplicate_joint"])
def test_only_exact_valid_face_topology_can_be_complete(damage):
    target = load_target("navigation")
    frame, owners, _ = measured_target(target)
    edges = list(frame.connections)
    if damage == "missing":
        edges.pop()
    elif damage == "extra":
        edges.append(C1Connection("extra", owners[0], "LEFT", owners[2], "RIGHT", True))
    elif damage == "wrong_face":
        edges[0] = replace(edges[0], connector_a="LEFT")
    elif damage == "invalid_joint":
        edges[0] = replace(edges[0], joint_valid=False)
    else:
        edges[0] = replace(edges[0], joint_id=edges[1].joint_id)
    status = gate(target, owners).assess(replace(frame, connections=tuple(edges)))
    assert not status.target_graph_complete
    assert not status.fold_complete


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), 0.0])
def test_required_fold_measurement_fails_closed(value):
    target = load_target("navigation")
    frame, owners, mapping = measured_target(target)
    modules = []
    for module in frame.modules:
        if module.module_id == mapping["v3"]:
            joints = tuple((name, value if name == "tilt.position_rad" else angle)
                           for name, angle in module.joints if value is not None or name != "tilt.position_rad")
            module = replace(module, joints=joints)
        modules.append(module)
    status = gate(target, owners).assess(replace(frame, modules=tuple(modules)))
    assert status.target_graph_complete
    assert not status.fold_complete


def symmetric_target():
    return {
        "global_attributes": {"graph_kind": "target_morphology",
                              "post_assembly_tilt_rad_by_vertex": {"a": -0.5, "b": 0.5},
                              "post_assembly_pan_rad_by_vertex": {"a": 0.25},
                              "post_assembly_tilt_tolerance_rad": 0.12},
        "nodes": [{"node_id": vertex, "attributes": {"node_type": "target_slot"}} for vertex in ("a", "b")],
        "edges": [{"module_a_id": "a", "module_b_id": "b", "attributes": {
            "relation_type": "target_connection", "face_a": "TOP", "face_b": "TOP"}}],
    }


def test_selects_fold_compatible_global_assignment_instead_of_first_topology_match():
    target = symmetric_target()
    frame, owners, _ = measured_target(target, owners=("zz", "aa"))
    status = gate(target, owners).assess(frame)
    assert status.target_graph_complete and status.fold_complete
    assert status.assignment == {"a": "zz", "b": "aa"}


def test_fold_joints_must_agree_under_one_global_assignment():
    target = symmetric_target()
    frame, owners, _ = measured_target(target, owners=("aa", "zz"))
    modules = tuple(replace(module, joints=tuple(
        (name, 0.0 if module.module_id == "aa" else 0.25) if name == "pan.position_rad" else (name, angle)
        for name, angle in module.joints)) for module in frame.modules)
    # TILT matches a->aa, while PAN matches a->zz: separate matches are invalid.
    status = gate(target, owners).assess(replace(frame, modules=modules))
    assert status.target_graph_complete
    assert not status.fold_complete


def test_pan_is_periodic_but_tilt_is_not():
    target = symmetric_target()
    frame, owners, _ = measured_target(target, owners=("aa", "zz"))
    modules = tuple(replace(module, joints=tuple(
        (name, angle + 4 * math.pi) if name == "pan.position_rad" else (name, angle)
        for name, angle in module.joints)) for module in frame.modules)
    evaluator = gate(target, owners)
    assert evaluator.assess(replace(frame, modules=modules)).fold_complete
    modules = tuple(replace(module, joints=tuple(
        (name, angle + 2 * math.pi) if name == "tilt.position_rad" else (name, angle)
        for name, angle in module.joints)) for module in modules)
    assert not evaluator.assess(replace(frame, modules=modules)).fold_complete


def test_cached_topology_rechecks_fold_and_invalidated_connections(monkeypatch):
    target = load_target("navigation")
    flat, owners, _ = measured_target(target, folded=False)
    folded, _, _ = measured_target(target)
    evaluator = gate(target, owners)
    assert not evaluator.assess(flat).fold_complete

    def unexpected_match(*args, **kwargs):
        raise AssertionError("Unchanged topology must reuse cached assignment")

    from mssr_expert.planning.smores_ep.self_reconfiguration_planner import SmoresSelfReconfigurationPlanner
    monkeypatch.setattr(SmoresSelfReconfigurationPlanner, "configuration_assignment", unexpected_match)
    monkeypatch.setattr(SmoresSelfReconfigurationPlanner, "_common_subtree_match", unexpected_match)
    assert evaluator.assess(replace(folded, connections=tuple(reversed(folded.connections)))).fold_complete
    damaged = (replace(folded.connections[0], joint_valid=False), *folded.connections[1:])
    assert not evaluator.assess(replace(folded, connections=damaged)).target_graph_complete


def test_target_without_posture_targets_needs_no_extra_joint_controls():
    target = symmetric_target()
    target["global_attributes"] = {"graph_kind": "target_morphology"}
    frame, owners, _ = measured_target(target)
    frame = replace(frame, modules=tuple(replace(module, joints=()) for module in frame.modules))
    assert gate(target, owners).assess(frame).fold_complete


@pytest.mark.parametrize("joint,error,complete", [
    ("tilt", 0.11, True), ("tilt", 0.13, False),
    ("pan", 0.03, True), ("pan", 0.04, False),
])
def test_fold_uses_target_tilt_tolerance_and_existing_pan_default(joint, error, complete):
    target = symmetric_target()
    frame, owners, _ = measured_target(target, owners=("aa", "zz"))
    modules = tuple(replace(module, joints=tuple(
        (name, angle + error) if name == f"{joint}.position_rad" else (name, angle)
        for name, angle in module.joints)) for module in frame.modules)
    status = gate(target, owners).assess(replace(frame, modules=modules))
    assert status.target_graph_complete
    assert status.fold_complete is complete


def test_custom_pan_tolerance_is_respected():
    target = symmetric_target()
    target["global_attributes"]["post_assembly_pan_tolerance_rad"] = 0.1
    frame, owners, _ = measured_target(target, owners=("aa", "zz"))
    modules = tuple(replace(module, joints=tuple(
        (name, angle + 0.08) if name == "pan.position_rad" else (name, angle)
        for name, angle in module.joints)) for module in frame.modules)
    assert gate(target, owners).assess(replace(frame, modules=modules)).fold_complete


def test_shared_physical_connector_cannot_satisfy_individually_matching_edges():
    target = load_target("navigation")
    frame, owners, _ = measured_target(target)
    connections = list(frame.connections)
    # Both chassis end modules have a BOTTOM->LEFT wheel relation. Reusing
    # one endpoint preserves local face types but violates the global target.
    connections[-2] = replace(connections[-2], module_b=connections[-4].module_b)
    status = gate(target, owners).assess(replace(frame, connections=tuple(connections)))
    assert not status.target_graph_complete
    assert not status.fold_complete


def test_duplicate_joint_measurement_is_not_silently_overwritten():
    target = symmetric_target()
    frame, owners, _ = measured_target(target, owners=("aa", "zz"))
    module = frame.modules[0]
    module = replace(module, joints=(*module.joints, ("tilt.position_rad", -0.5)))
    status = gate(target, owners).assess(replace(frame, modules=(module, *frame.modules[1:])))
    assert status.target_graph_complete
    assert not status.fold_complete


def test_wrapped_target_payload_and_attributed_graph_are_supported():
    target = symmetric_target()
    frame, owners, _ = measured_target(target)
    from mssr_expert.graph.serialization import attributed_graph_from_dict
    assert gate({"target": target}, owners).assess(frame).fold_complete
    assert gate(attributed_graph_from_dict(target), owners).assess(frame).fold_complete


@pytest.mark.parametrize("damage", ["missing", "duplicate", "foreign"])
def test_exact_expected_physical_owners_are_required(damage):
    target = load_target("navigation")
    frame, owners, _ = measured_target(target)
    modules = list(frame.modules)
    if damage == "missing":
        modules.pop()
    else:
        modules[-1] = replace(modules[-1], module_id=owners[0] if damage == "duplicate" else "foreign")
    status = gate(target, owners).assess(replace(frame, modules=tuple(modules)))
    assert not status.target_graph_complete
    assert not status.fold_complete
