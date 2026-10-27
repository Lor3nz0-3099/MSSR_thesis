"""B1: physical inputs and label-only reconstruction, with no runtime imports."""

import copy
import importlib
import importlib.util
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
import subprocess
import sys

import pytest

from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.tests.bc_v1 import fixtures as f

ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "configs/bc_v1/smores_ep.robot.json"


def api():
    # A real behavior RED, not a missing import or environment failure.
    name = "drl.IL.bc_v1.smores_adapter"
    assert importlib.util.find_spec(name) is not None, "Missing B1 SMORESAdapter behavior"
    module = importlib.import_module(name)
    assert callable(getattr(module, "SMORESAdapter", None))
    assert callable(getattr(module, "read_label_evidence", None))
    return module


def adapter(spec=None):
    module = api()
    return module.SMORESAdapter(c.load_robot_spec(CONFIG) if spec is None else spec)


def source_evidence(tmp_path, records):
    path = tmp_path / "source.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    row = {"provenance": {"source_records": [
        {"_loader_source_path": "source.jsonl", "_loader_source_row": i + 1}
        for i in range(len(records))]}}
    return api().read_label_evidence(row, tmp_path), path, row


def choice(target):
    if isinstance(target, c.ResourceTarget):
        return c.ActionChoice(target.mode_id, target.resource.module,
                              resource=target.resource, payload=target.payload)
    return c.ActionChoice(target.operation_id, target.initiator,
                          participants=target.participants, connectors=target.connectors,
                          parameters=target.parameters)


def test_config_validates_and_declares_shared_resource_and_enabled_settle():
    api()
    spec = c.load_robot_spec(CONFIG)
    schema = json.loads((CONFIG.parent / "robot_spec.schema.json").read_text())
    assert f.schema_accepts(json.loads(CONFIG.read_text()), schema)
    assert next(a for a in spec.actions if a.action_id == "gravity_settle").enabled
    internal = next(r for r in spec.module_types[0].resources if r.resource_id == "internal_motion")
    assert {x.coordinate_id for x in internal.coordinates} == {"pan", "tilt"}
    assert "mutually_exclusive_modes" in internal.capabilities
    assert all(x.codec == "continuous" for x in internal.modes)
    geom = {x.name: x.values for x in spec.module_types[0].geometry}
    assert geom["wheel_radius_m"] == (0.03150,)  # current geometry.py, not legacy manifest


def test_wheels_map_order_units_and_payload():
    a = adapter()
    action, masks = f.smores_label(wheels=True)
    target, = a.encode_labels(action, masks, ()).resource_targets
    assert target == c.ResourceTarget(c.ResourceRef(c.ModuleRef("moving"), "locomotion"),
                                     "wheel_velocity", (0.8, -0.2))
    resource = next(r for r in a.spec.module_types[0].resources if r.resource_id == "locomotion")
    assert resource.modes[0].units == ("rad/s", "rad/s")
    assert a.decode_action(choice(target)) == {"command": "wheel_velocity",
        "module_ids": ["moving"], "parameters": {"left_rad_s": 0.8, "right_rad_s": -0.2}}


@pytest.mark.parametrize("mode,command,key", [
    ("pan_velocity", "pan_velocity", "rate_rad_s"),
    ("tilt_velocity", "tilt_velocity", "rate_rad_s"),
    ("pan_target", "set_pan", "angle_rad"), ("tilt_target", "set_tilt", "angle_rad"),
    ("pan_delta", "rotate_pan_by", "delta_rad"), ("tilt_delta", "rotate_tilt_by", "delta_rad"),
])
def test_internal_modes_map_to_one_configured_resource(mode, command, key):
    a = adapter()
    action, masks = f.smores_label(internal=mode)
    target, = a.encode_labels(action, masks, ()).resource_targets
    assert target.resource.resource_id == "internal_motion"
    assert target.mode_id == mode and target.payload == (0.3,)
    assert a.decode_action(choice(target)) == {"command": command,
        "module_ids": ["moving"], "parameters": {key: 0.3}}


def test_simultaneous_resources_and_omitted_resource_is_not_zero():
    a = adapter()
    action, masks = f.smores_label(wheels=True, internal="tilt_target")
    targets = a.encode_labels(action, masks, ()).resource_targets
    assert len(targets) == 2 and all(t.supervised for t in targets)
    masks["module_resource_masks"]["moving"]["wheels"] = False
    assert len(a.encode_labels(action, masks, ()).resource_targets) == 1
    action, masks = f.smores_label()
    assert not a.encode_labels(action, masks, ()).resource_targets


def test_current_geometry_becomes_body_local_and_preserves_joint_observations():
    graph = adapter().encode_state(f.smores_graph())
    assert len(graph.modules) == 2
    top = next(x for x in graph.connectors if x.ref == c.ConnectorRef(c.ModuleRef("moving"), "TOP"))
    features = {x.name: x for x in top.features}
    assert features["position_body_m"].values == pytest.approx((.05, 0., .02))
    assert features["normal_body"].values == pytest.approx((1., 0., 0.))
    assert features["tangent_body"].values == pytest.approx((0., 1., 0.))
    resource = next(x for x in graph.resources if x.ref == c.ResourceRef(c.ModuleRef("moving"), "internal_motion"))
    assert resource.coordinates == (c.CoordinateState("pan", .4, .1), c.CoordinateState("tilt", -.3, None))
    assert not graph.modules[0].features


def test_config_fallback_is_only_valid_for_body_fixed_ports():
    graph = adapter().encode_state(f.smores_graph())
    bottom = next(x for x in graph.connectors if x.ref.module.module_id == "moving" and x.ref.connector_id == "BOTTOM")
    assert next(x for x in bottom.features if x.name == "position_body_m").values == (-.033999, 0., 0.)
    assert all(x.valid for x in bottom.features)
    left = next(x for x in graph.connectors if x.ref.module.module_id == "moving" and x.ref.connector_id == "LEFT")
    assert not any(x.valid for x in left.features)  # wheel angle/port geometry absent


@pytest.mark.parametrize("relation", ["current_connection", "contact"])
def test_edges_route_physical_endpoints_without_using_control_state(relation):
    raw = f.smores_graph()
    raw["edges"] = [{"module_a_id": "moving", "face_a": "TOP",
        "module_b_id": "peer", "face_b": "BOTTOM", "relation": relation,
        "relative_pose": {"position": [1., 0., 0.], "orientation": [0., 0., 0., 1.]}}]
    graph = adapter().encode_state(raw)
    edge, = graph.connections
    assert edge.relation_id == relation
    assert edge.endpoint_a == c.ConnectorRef(c.ModuleRef("moving"), "TOP")
    port = next(x for x in graph.connectors if x.ref == edge.endpoint_a)
    assert port.connected_to == (edge.endpoint_b if relation == "current_connection" else None)


@pytest.mark.parametrize("primitive", ["align_faces", "dock", "undock"])
def test_structural_roundtrip_from_canonical_only(primitive):
    a = adapter()
    action, masks = f.smores_label(primitive)
    missing_source = (c.LabelEvidence("/unavailable/old/raw.jsonl", 1, "0" * 64),)
    target, = a.encode_labels(action, masks, missing_source).structural_targets
    assert target.participants == (c.ModuleRef("moving"), c.ModuleRef("peer"))
    assert target.connectors == (c.ConnectorRef(c.ModuleRef("moving"), "TOP"),
                                 c.ConnectorRef(c.ModuleRef("peer"), "BOTTOM"))
    decoded = a.decode_action(choice(target))
    assert decoded == {"command": primitive, "module_ids": ["moving", "peer"],
                       "parameters": action["modules"]["moving"]["structural"]["parameters"]}
    json.dumps(decoded)


def test_gravity_settle_scope_and_no_invented_joint_zeros():
    a = adapter()
    action, masks = f.smores_label("gravity_settle")
    bundle = a.encode_labels(action, masks, ())
    target, = bundle.structural_targets
    assert not bundle.resource_targets and not target.connectors
    assert target.participants == (c.ModuleRef("moving"), c.ModuleRef("peer"))
    assert dict((p.name, p.value) for p in target.parameters)["passive_module_ids"] == target.participants
    assert a.decode_action(choice(target)) == {"command": "gravity_settle", "module_ids": ["moving"],
        "parameters": {"duration_s": 1.5, "passive_module_ids": ["moving", "peer"]}}
    action["modules"]["moving"]["structural"]["parameters"]["passive_module_ids"] = ["moving"]
    assert a.encode_labels(action, masks, ()).structural_targets[0].participants == (c.ModuleRef("moving"),)


def test_structural_peer_is_recovered_from_decision_only(tmp_path):
    a = adapter()
    evidence, _, _ = source_evidence(tmp_path, [f.smores_source_goal(), f.smores_source_goal(goal_id="same-pair")])
    action, masks = f.smores_label("align_faces", peer=False)
    action["target_graph"] = {"peer": "wrong"}
    target, = a.encode_labels(action, masks, evidence).structural_targets
    assert target.participants == (c.ModuleRef("moving"), c.ModuleRef("peer"))
    assert not any(p.name in {"goal_id", "execution_phase"} for p in target.parameters)
    assert evidence[0].source_record == 1 and evidence[0].phase == "REACH"


@pytest.mark.parametrize("fault", ["missing", "ambiguous", "parameters", "initiator"])
def test_missing_or_ambiguous_peer_evidence_fails(tmp_path, fault):
    a = adapter()
    action, masks = f.smores_label("dock", peer=False)
    records = [f.smores_source_goal("dock")]
    if fault == "missing": records = []
    if fault == "ambiguous":
        action["modules"]["other"] = {"participates_in": []}
        records.append(f.smores_source_goal("dock", "other"))
    if fault == "parameters": records[0]["expert_action"]["primitive_goal"]["parameters"]["clocking_quarter_turns"] = 2
    if fault == "initiator": records[0]["expert_action"]["primitive_goal"]["module_ids"][0] = "peer"
    evidence, _, _ = source_evidence(tmp_path, records)
    with pytest.raises(ValueError, match="peer|scope|evidence|parameters"):
        a.encode_labels(action, masks, evidence)


def test_contradictory_canonical_peers_fail_without_source_recovery():
    action, masks = f.smores_label("dock")
    action["modules"]["other"] = copy.deepcopy(action["modules"]["peer"])
    with pytest.raises(ValueError, match="(?i)ambiguous|contradictory"):
        adapter().encode_labels(action, masks, ())


def test_canonical_explicit_module_scope_is_sufficient():
    action, masks = f.smores_label("dock", peer=False)
    action["modules"]["moving"]["structural"]["module_ids"] = ["moving", "peer"]
    assert adapter().encode_labels(action, masks, ()).structural_targets[0].participants[-1] == c.ModuleRef("peer")


def test_adapter_does_not_mutate_source():
    graph = f.smores_graph()
    action, masks = f.smores_label("align_faces", wheels=True, internal="pan_velocity")
    originals = copy.deepcopy((graph, action, masks))
    a = adapter()
    a.encode_state(graph)
    a.encode_labels(action, masks, ())
    assert (graph, action, masks) == originals


def test_evidence_and_forbidden_fields_never_enter_policy_inputs(tmp_path):
    a = adapter()
    graph = f.smores_graph()
    baseline = c.PolicyInputs(a.encode_state(graph), c.ObservationBundle(), c.TaskContext())
    graph.update({"action_t": "FORBIDDEN", "provenance": "FORBIDDEN", "supervision": "FORBIDDEN",
                  "graph_t_plus_1": "FORBIDDEN", "success": True, "target_graph": "FORBIDDEN"})
    for node in graph["nodes"]:
        node["control_state"] = {"next_action": "FORBIDDEN"}
        node["role"] = "changed-template-role"
    evidence, _, _ = source_evidence(tmp_path, [f.smores_source_goal()])
    action, masks = f.smores_label("align_faces", peer=False)
    a.encode_labels(action, masks, evidence)
    inputs = c.PolicyInputs(a.encode_state(graph), c.ObservationBundle(), c.TaskContext())
    assert inputs == baseline
    assert "FORBIDDEN" not in json.dumps(asdict(inputs))
    assert "source_path" not in json.dumps(asdict(inputs))


def test_evidence_index_cache_and_integrity(tmp_path):
    evidence, path, row = source_evidence(tmp_path, [f.smores_source_goal()])
    assert api().read_label_evidence(row, tmp_path) == evidence
    assert evidence[0].source_path == str(path)
    assert len(evidence[0].source_sha256) == 64
    path.write_text(json.dumps(f.smores_source_goal(peer="other")) + "\n")
    action, masks = f.smores_label("align_faces", peer=False)
    with pytest.raises(ValueError, match="hash|changed|integrity"):
        adapter().encode_labels(action, masks, evidence)


def test_reading_evidence_does_not_mistake_access_time_for_mutation(tmp_path):
    api()
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps(f.smores_source_goal()) + "\n")
    os.utime(path, (1, 2))  # a real read may update atime, never label content
    row = {"provenance": {"source_records": [{"_loader_source_path": str(path), "_loader_source_row": 1}]}}
    assert len(api().read_label_evidence(row, tmp_path)) == 1


def test_settle_recovers_passive_scope_from_canonical_participation_first():
    action, masks = f.smores_label("gravity_settle")
    del action["modules"]["moving"]["structural"]["parameters"]["passive_module_ids"]
    action["modules"]["peer"]["participates_in"] = [{"primitive": "gravity_settle",
        "initiator_module_id": "moving", "role": "passive_module"}]
    target, = adapter().encode_labels(action, masks, ()).structural_targets
    assert target.participants == (c.ModuleRef("moving"), c.ModuleRef("peer"))


def test_pair_recovers_missing_faces_from_canonical_participation_first():
    action, masks = f.smores_label("align_faces")
    del action["modules"]["moving"]["structural"]["parameters"]["face_b"]
    target, = adapter().encode_labels(action, masks, ()).structural_targets
    assert target.connectors[-1].connector_id == "BOTTOM"


@pytest.mark.parametrize("source", ["canonical", "evidence"])
def test_null_face_is_recovered_like_a_missing_face(tmp_path, source):
    action, masks = f.smores_label("align_faces", peer=source == "canonical")
    action["modules"]["moving"]["structural"]["parameters"]["face_b"] = None
    evidence = source_evidence(tmp_path, [f.smores_source_goal()])[0] if source == "evidence" else ()
    target, = adapter().encode_labels(action, masks, evidence).structural_targets
    assert target.connectors[-1].connector_id == "BOTTOM"


def test_settle_rejects_contradictory_canonical_passive_scope():
    action, masks = f.smores_label("gravity_settle")
    action["modules"]["moving"]["structural"]["parameters"]["passive_module_ids"] = ["moving"]
    action["modules"]["peer"]["participates_in"] = [{"primitive": "gravity_settle",
        "initiator_module_id": "moving", "role": "passive_module"}]
    with pytest.raises(ValueError, match="(?i)contradictory.*scope"):
        adapter().encode_labels(action, masks, ())


def test_settle_rejects_contradictory_decode_scope():
    a = adapter()
    action, masks = f.smores_label("gravity_settle")
    target, = a.encode_labels(action, masks, ()).structural_targets
    with pytest.raises(ValueError, match="(?i)contradictory.*scope"):
        a.decode_action(replace(choice(target), participants=(c.ModuleRef("moving"),)))


@pytest.mark.parametrize("fault", ["units", "missing_units", "nan", "unknown_mode", "masked_uncommanded"])
def test_invalid_resource_labels_fail_explicitly(fault):
    action, masks = f.smores_label(internal="pan_velocity")
    command = action["modules"]["moving"]["internal"]
    if fault == "units": command["unit"] = "deg/s"
    if fault == "missing_units": del command["unit"]
    if fault == "nan": command["value"] = float("nan")
    if fault == "unknown_mode": command["mode"] = "unknown"
    if fault == "masked_uncommanded": command["commanded"] = False
    with pytest.raises(ValueError): adapter().encode_labels(action, masks, ())


def test_settle_can_be_explicitly_disabled_only_through_config():
    a = adapter()
    spec = replace(a.spec, actions=tuple(replace(x, enabled=False) if x.action_id == "gravity_settle" else x
                                       for x in a.spec.actions))
    action, masks = f.smores_label("gravity_settle")
    bundle = adapter(spec).encode_labels(action, masks, ())
    assert not bundle.structural_targets
    assert "disabled_action:gravity_settle" in bundle.exclusions


def test_adapter_import_is_runtime_and_framework_independent():
    api()
    script = '''
import importlib.abc, sys
blocked = {'rclpy', 'isaacsim', 'isaaclab', 'omni', 'pxr', 'smores_ep', 'mssr_expert', 'numpy', 'torch'}
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked: raise ImportError(fullname)
sys.meta_path.insert(0, Guard())
import drl.IL.bc_v1.smores_adapter
assert not blocked.intersection(sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
