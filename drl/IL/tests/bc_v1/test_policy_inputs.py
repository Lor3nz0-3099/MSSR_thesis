"""D2: a closed policy view, independent of targets/audit and robot hardware."""

import copy
from dataclasses import asdict, fields, replace
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.bc_v1.smores_adapter import SMORESAdapter
from drl.IL.bc_v1.task_context import build_task_view, read_task_metadata
from drl.IL.tests.bc_v1 import fixtures as f

ROOT = Path(__file__).resolve().parents[4]
TASK_CONFIG = ROOT / "configs/bc_v1/task_context.json"
ROBOT_CONFIG = ROOT / "configs/bc_v1/smores_ep.robot.json"


def selector():
    name = "drl.IL.bc_v1.policy_inputs"
    assert importlib.util.find_spec(name) is not None, "Missing D2 policy input selector"
    module = importlib.import_module(name)
    assert callable(getattr(module, "select_policy_inputs", None)), "Missing D2 whitelist behavior"
    return module.select_policy_inputs


def physical_view(task="gap"):
    return (f.state_graph(2, 1), c.ObservationBundle(), c.TaskContext(task, (
        c.NumericFeature("gap.width_m", (.2,)),
        c.NumericFeature("world.gap.near_edge_x_m", (1.,)),
    ) if task == "gap" else ()))


def raw_view_fixture(task="button"):
    row = {**f.task_row(task), "graph_t": f.smores_graph(),
           "graph_t_plus_1": {"nodes": [{"module_id": "future"}]},
           "action_t": {}, "supervision": {}, "episode_id": "synthetic-d2"}
    metadata = {"course": {"frame_id": "world", "button": {
        "center_xyz_m": [8., 0., .2], "current_center_xyz_m": [7.999, 0., .2],
        "depression_m": .001, "plunger_stroke_m": .004}}}
    if task == "navigation":
        metadata = {"course": {"frame_id": "world", "navigation": {
            "waypoints_xyyaw": [[100., 200., 0.]]}, "physical_track": {
            "cone_centers_xy_m": [[1., 2.]], "cone_radius_m": .05}}}
    return row, metadata


def from_raw(row, metadata):
    # This is test caller composition, not an API that passes raw data into D2.
    graph = SMORESAdapter(c.load_robot_spec(ROBOT_CONFIG)).encode_state(row["graph_t"])
    context, observation = build_task_view(row, metadata, json.loads(TASK_CONFIG.read_text()))
    return selector()(graph, observation, context)


def test_valid_physical_view_is_preserved_and_contains_only_three_input_structures():
    graph, observation, context = physical_view()
    selected = selector()(graph, observation, context)
    assert type(selected) is c.PolicyInputs
    assert set(asdict(selected)) == {"graph", "observation", "context"}
    assert selected.graph == graph and selected.context == context
    assert selected.graph is not graph and selected.context is not context
    assert selected.graph.modules[0] is not graph.modules[0]
    assert selected.context.static_geometry[0].values == (.2,)


@pytest.mark.parametrize("modules,ports,surface", [(2, 0, True), (5, 7, False), (12, 2, False)])
def test_selector_preserves_variable_entities_and_surface_connectors(modules, ports, surface):
    graph = f.state_graph(modules, ports, surface=surface)
    selected = selector()(graph, c.ObservationBundle(), c.TaskContext())
    assert len(selected.graph.modules) == modules
    assert len(selected.graph.connectors) == modules * (ports + int(surface))
    assert len(selected.graph.resources) == modules
    assert selected.graph == graph


@pytest.mark.parametrize("key,value", [
    ("stage", "assembly"), ("stage_id", "reconfiguration"),
    ("stage_name", "task_execution"), ("stage", "navigation"),
    ("configuration_goal", "snake8"), ("desired_morphology", "mobile_manipulator8"),
    ("target_configuration", "rc_car8"), ("morphology_goal", "snake8"),
    ("structural_goal", "mm8"), ("target_morphology", "snake8"),
    ("target_graph", {"nodes": ["future"]}), ("final_topology", {"edges": ["future"]}),
    ("graph_t_plus_1", {"nodes": ["future", "opposite"]}),
    ("next_state", {"task_type": "stairs"}), ("future_task", "gap"),
    ("observation_t_plus_1", {"task_type": "gap", "depression_m": 99.}),
    ("next_record", {"task_type": "stairs"}),
    ("action_t", {"modules": {"moving": {"structural": {"primitive": "undock"}}}}),
    ("resource_labels", {"velocity": 99.}), ("structural_labels", {"primitive": "dock"}),
    ("supervision", {"valid_for_behavior_cloning": False, "loss_weights": [99.],
                     "module_resource_masks": {"moving": {"structural": False}}}),
    ("command_masks", {"moving": True}), ("bc_validity_mask", False),
    ("reward", 999.), ("success", False), ("failure", True),
    ("terminal", True), ("outcome", "future-success"),
    ("episode_id", "opposite-episode"), ("dataset_kind", "different"),
    ("provenance", {"logical_episode_id": "opposite", "canonical_transition_index": 9000,
                    "source_records": [{"_loader_source_path": "missing.jsonl", "_loader_source_row": 9000}]}),
    ("decision_stamp", 12345.), ("stream_id", "other"),
    ("control_state", {"busy": True, "phase": "APPROACH"}),
    ("fsm_state", "DISPATCHING_ALIGN"), ("executor_phase", "CLOCKING"),
    ("secret_future_hint", {"nested": {"target": "future"}}),
])
def test_forbidden_raw_metadata_cannot_change_selected_inputs(key, value):
    row, metadata = raw_view_fixture()
    before = from_raw(row, metadata)
    changed = copy.deepcopy(row)
    changed[key] = value
    assert from_raw(changed, metadata) == before


@pytest.mark.parametrize("key", ["root_module_id", "role", "morphology", "template_id", "target_morphology"])
def test_semantic_graph_annotations_cannot_change_physical_view(key):
    row, metadata = raw_view_fixture()
    before = from_raw(row, metadata)
    changed = copy.deepcopy(row)
    changed["graph_t"][key] = "opposite-template"
    for node in changed["graph_t"]["nodes"]:
        node[key] = "opposite-template"
    assert from_raw(changed, metadata) == before


@pytest.mark.parametrize("name", ["stage_id", "role", "episode_id", "configuration_goal",
                                   "desired_morphology", "graph_t_plus_1", "mask", "reward",
                                   "root_module_id", "action_target_id", "decision_stamp", "module_id"])
def test_open_numeric_feature_slots_cannot_smuggle_semantics(name):
    graph, observation, context = physical_view()
    baseline = selector()(graph, observation, context)
    poisoned = c.NumericFeature(name, (99.,))
    graph = replace(graph, modules=(replace(graph.modules[0], features=(poisoned,)), *graph.modules[1:]),
                    connectors=(replace(graph.connectors[0], features=(poisoned,)), *graph.connectors[1:]),
                    resources=(replace(graph.resources[0], features=(poisoned,)), *graph.resources[1:]))
    selected = selector()(graph, c.ObservationBundle((poisoned,)),
                          replace(context, static_geometry=(*context.static_geometry, poisoned)))
    assert selected == baseline


def test_selection_removes_undeclared_attributes_and_detaches_all_objects():
    graph, observation, context = physical_view()
    object.__setattr__(graph.modules[0], "secret_future_hint", {"target": "future"})
    object.__setattr__(context, "stage_id", "assembly")
    selected = selector()(graph, observation, context)
    assert not hasattr(selected.context, "stage_id")
    assert not hasattr(selected.graph.modules[0], "secret_future_hint")
    object.__setattr__(graph.modules[0], "position", (99., 99., 99.))
    assert selected.graph.modules[0].position == (0., 0., 0.)


@pytest.mark.parametrize("slot", ["graph", "observation", "context"])
def test_raw_dictionaries_are_rejected_even_by_direct_policyinputs_constructor(slot):
    values = dict(zip(("graph", "observation", "context"), physical_view()))
    values[slot] = {"secret_future_hint": 99.}
    with pytest.raises(TypeError):
        c.PolicyInputs(**values)
    with pytest.raises(TypeError):
        selector()(**values)


def test_direct_policyinputs_cannot_bypass_feature_whitelist():
    graph, observation, context = physical_view()
    with pytest.raises(ValueError, match="whitelist|selected|feature"):
        c.PolicyInputs(graph, c.ObservationBundle((c.NumericFeature("stage_id", (1.,)),)), context)
    with pytest.raises(ValueError, match="whitelist|selected|feature"):
        c.PolicyInputs(graph, observation, replace(context, static_geometry=(c.NumericFeature("button.depression_m", (.004,)),)))


@pytest.mark.parametrize("kwargs", [{"supervision": c.SupervisionBundle()},
                                     {"audit": c.SampleAudit("ep", 1)}, {"graph_t_plus_1": c.StateGraph()},
                                     {"action_t": {}}, {"raw": {}}])
def test_selector_cannot_accept_supervision_audit_raw_or_future_arguments(kwargs):
    with pytest.raises(TypeError):
        selector()(*physical_view(), **kwargs)


def test_same_inputs_with_opposite_resource_and_structural_targets_stay_equal():
    graph, observation, context = physical_view()
    one = c.PolicySample(selector()(graph, observation, context), c.SupervisionBundle(
        resource_targets=(c.ResourceTarget(graph.resources[0].ref, "rate", (1.,), True),),
        structural_targets=(c.StructuralTarget(graph.modules[0].ref, "connect", supervised=True),)),
        c.SampleAudit("first", 1))
    two = c.PolicySample(selector()(graph, observation, context), c.SupervisionBundle(
        resource_targets=(c.ResourceTarget(graph.resources[0].ref, "rate", (-1.,), False),),
        structural_targets=(c.StructuralTarget(graph.modules[0].ref, "disconnect", supervised=False),),
        exclusions=("no-loss",)), c.SampleAudit("other", 999, provenance="different"))
    assert one.inputs == two.inputs and one.supervision != two.supervision


def test_current_physical_changes_affect_inputs():
    graph, observation, context = physical_view()
    baseline = selector()(graph, observation, context)
    changed = replace(graph, modules=(replace(graph.modules[0], position=(.1, 0., 0.)), *graph.modules[1:]))
    assert selector()(changed, observation, context) != baseline
    changed = replace(graph, connectors=(replace(graph.connectors[0], status="available"), *graph.connectors[1:]))
    assert selector()(changed, observation, context) != baseline
    changed = replace(graph, resources=(replace(graph.resources[0], coordinates=(c.CoordinateState("rotation", .2, .3),)), *graph.resources[1:]))
    assert selector()(changed, observation, context) != baseline


def test_assigned_task_and_approved_static_geometry_are_sensitive():
    graph, observation, context = physical_view()
    baseline = selector()(graph, observation, context)
    assert selector()(graph, observation, c.TaskContext("stairs")) != baseline
    assert selector()(graph, observation, replace(context, static_geometry=(c.NumericFeature("gap.width_m", (.3,)),))) != baseline
    # Same graph/geometry, only semantic task differs.
    empty = c.TaskContext("gap")
    assert selector()(graph, observation, empty) != selector()(graph, observation, replace(empty, task_id="stairs"))


def test_current_dynamic_measurements_are_sensitive_and_static_dynamic_owners_are_closed():
    graph, _, context = physical_view()
    first = c.ObservationBundle((c.NumericFeature("button.depression_m", (.001,)),))
    second = c.ObservationBundle((c.NumericFeature("button.depression_m", (.002,)),))
    assert selector()(graph, first, context) != selector()(graph, second, context)
    bad = c.NumericFeature("gap.width_m", (.2,))
    assert selector()(graph, replace(first, features=(*first.features, bad)), context).observation == first
    dynamic = c.NumericFeature("world.button.current_center_xyz_m", (8., 0., .2))
    static = c.TaskContext("button", (c.NumericFeature("world.button.center_xyz_m", (8., 0., .2)), dynamic))
    selected = selector()(graph, c.ObservationBundle((dynamic,)), static)
    assert [x.name for x in selected.context.static_geometry] == ["world.button.center_xyz_m"]
    assert selected.observation.features == (dynamic,)


def test_future_task_typed_semantics_and_geometry_cannot_bypass_current_task_whitelist():
    graph, observation, context = physical_view()
    poisoned = replace(context, static_geometry=(*context.static_geometry,
                       c.NumericFeature("button.plunger_stroke_m", (.004,)),
                       c.NumericFeature("mission.next_task", (3.,))))
    assert selector()(graph, observation, poisoned) == selector()(graph, observation, context)
    with pytest.raises(ValueError, match="task"):
        selector()(graph, observation, c.TaskContext("reconfiguration"))


def test_current_navigation_cones_change_inputs_nominal_routes_do_not():
    row, metadata = raw_view_fixture("navigation")
    baseline = from_raw(row, metadata)
    assert baseline.context.task_id == "navigation"
    assert any(x.name == "world.cones.0.center_xy_m" for x in baseline.context.static_geometry)
    changed = copy.deepcopy(metadata)
    changed["course"]["navigation"].update(path=[[99., 99.]], waypoints_xyyaw=[[99., 99., 0.]],
                                             nominal_route={"future": [99.]})
    changed["course"]["physical_track"]["centerline_xy_m"] = [[99., 99.]]
    assert from_raw(row, changed) == baseline
    changed["course"]["physical_track"]["cone_centers_xy_m"] = [[1.1, 2.]]
    assert from_raw(row, changed) != baseline


def test_ids_remain_opaque_references_without_generated_numeric_features():
    graph, observation, context = physical_view()
    def rename(ref):
        return c.ModuleRef("opaque-renamed-" + ref.module_id)
    renamed = c.StateGraph(
        tuple(replace(m, ref=rename(m.ref)) for m in graph.modules),
        tuple(replace(p, ref=c.ConnectorRef(rename(p.ref.module), "port-opaque")) for p in graph.connectors),
        tuple(replace(r, ref=c.ResourceRef(rename(r.ref.module), "resource-opaque"),
                      coordinates=(c.CoordinateState("coordinate-opaque", .1, 0.),)) for r in graph.resources))
    selected = selector()(renamed, observation, context)
    assert [m.position for m in selected.graph.modules] == [m.position for m in graph.modules]
    assert all(not m.features for m in selected.graph.modules)
    assert all(not r.features for r in selected.graph.resources)
    assert selected.graph.modules[0].ref.module_id == "opaque-renamed-unit-0"
    assert selected.graph.resources[0].coordinates[0].position == .1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "1.0"])
def test_physical_values_must_be_finite_numeric_scalars(bad):
    graph, observation, context = physical_view()
    graph = replace(graph, modules=(replace(graph.modules[0], position=(bad, 0., 0.)), *graph.modules[1:]))
    with pytest.raises((ValueError, TypeError)):
        selector()(graph, observation, context)


def test_approved_feature_shape_and_validity_are_preserved_and_checked():
    graph, observation, context = physical_view()
    invalid_frame = c.NumericFeature("unframed.gap.near_edge_x_m", (1.,), False)
    selected = selector()(graph, observation, replace(context, static_geometry=(invalid_frame,)))
    assert selected.context.static_geometry == (invalid_frame,)
    with pytest.raises(ValueError, match="shape|size|length"):
        selector()(graph, c.ObservationBundle((c.NumericFeature("world.button.current_center_xyz_m", (1., 2.)),)), context)
    with pytest.raises(ValueError, match="valid"):
        selector()(graph, observation, replace(context, static_geometry=(replace(invalid_frame, valid=1),)))


def test_core_selector_imports_without_robot_simulator_or_tensor_libraries():
    selector()
    script = """
import importlib.abc, sys
blocked={'rclpy','rospy','isaacsim','isaaclab','omni','torch','numpy','smores_ep','mssr_expert'}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ImportError('forbidden dependency '+fullname)
sys.meta_path.insert(0,Block())
from drl.IL.bc_v1.policy_inputs import select_policy_inputs
from drl.IL.bc_v1.bc_v1_contract import StateGraph,ObservationBundle,TaskContext
x=select_policy_inputs(StateGraph(),ObservationBundle(),TaskContext())
assert x.context.task_id is None
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_current_connection_payload_and_allowed_relative_dofs_are_preserved():
    graph, observation, context = physical_view()
    a, b = (port.ref for port in graph.connectors)
    connection = c.ConnectionState(a, b, "rolling_contact", ("roll", "tangent"), (
        c.NumericFeature("relative_position_m", (.1, .2, .3)),
        c.NumericFeature("relative_orientation_xyzw", (0., 0., 0., 1.)),
        c.NumericFeature("clocking_quarter_turns", (1.,)),
        c.NumericFeature("executor_phase", (3.,))))
    selected = selector()(replace(graph, connections=(connection,)), observation, context)
    assert selected.graph.connections[0].allowed_relative_dofs == ("roll", "tangent")
    assert [x.name for x in selected.graph.connections[0].features] == [
        "relative_position_m", "relative_orientation_xyzw", "clocking_quarter_turns"]
    changed = replace(connection, features=(c.NumericFeature("relative_position_m", (.4, .2, .3)),))
    assert selector()(replace(graph, connections=(changed,)), observation, context) != selected
    with pytest.raises(ValueError, match="relation"):
        selector()(replace(graph, connections=(replace(connection, relation_id="DISPATCHING_ALIGN"),)), observation, context)


def test_direct_constructor_rejects_hidden_attributes_in_typed_contracts():
    graph, observation, context = physical_view()
    object.__setattr__(context, "stage_id", "reconfiguration")
    with pytest.raises(ValueError, match="attribute|selected"):
        c.PolicyInputs(graph, observation, context)


@pytest.mark.parametrize("slot,value", [("kind", "fixed_port"), ("status", "available"),
                                      ("relation_id", "current_connection")])
@pytest.mark.parametrize("direct", [False, True])
def test_physical_enum_tokens_cannot_carry_mutable_subclass_metadata(slot, value, direct):
    class PoisonedToken(str):
        pass
    token = PoisonedToken(value)
    token.stage_id = {"stage": "reconfiguration", "future_graph": {"nodes": ["future"]}}
    graph, observation, context = physical_view()
    if slot == "relation_id":
        graph = replace(graph, connections=(c.ConnectionState(
            graph.connectors[0].ref, graph.connectors[1].ref, token),))
    else:
        graph = replace(graph, connectors=(replace(graph.connectors[0], **{slot: token}), *graph.connectors[1:]))
    with pytest.raises((ValueError, TypeError), match="physical|connector|relation"):
        (c.PolicyInputs if direct else selector())(graph, observation, context)


def test_selector_is_independent_of_source_and_label_modules():
    selector()
    script = """
import importlib.abc, sys
class BlockLabels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'drl.IL.bc_v1.smores_adapter','drl.IL.bc_v1.task_context','drl.IL.preprocessing'}:
            raise ImportError('input selector must not call source or label code')
sys.meta_path.insert(0,BlockLabels())
from drl.IL.bc_v1.policy_inputs import select_policy_inputs
from drl.IL.bc_v1.bc_v1_contract import StateGraph,ObservationBundle,TaskContext
assert select_policy_inputs(StateGraph(),ObservationBundle(),TaskContext('navigation')).context.task_id=='navigation'
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


_REAL_CASES = (
    ("assembly", "expert:assembly:seed-006217", 0, "button", "button."),
    ("gap", "teleop-1bea8f28c1c04323a4c3ec7e2829d07c", 655, "gap", "gap."),
    ("stairs", "teleop-0e4e34fe3a5f4ed1b6a86642acf4fac8", 1733, "stairs", "stairs."),
    ("button", "teleop-1bea8f28c1c04323a4c3ec7e2829d07c", 7228, "button", "button."),
    ("navigation", "teleop-1bea8f28c1c04323a4c3ec7e2829d07c", 3673, "navigation", "environment_cones."),
    ("terminal-goal", "teleop-1bea8f28c1c04323a4c3ec7e2829d07c", 8604, "navigation", "goal."),
)


@pytest.fixture(scope="module")
def real_policy_rows(tmp_path_factory):
    from collections import defaultdict
    root = ROOT / "datasets/expert_v1_canonical_v6"
    if not (root / "manifest.json").is_file():
        pytest.skip("Canonical v6 unavailable; synthetic D2 boundary tests remain active")
    manifest = json.loads((root / "manifest.json").read_text())
    paths = {episode["episode_id"]: root / episode["file"] for episode in manifest["episodes"]}
    def signature(path):
        st = path.stat()
        return st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns
    before = {path: signature(path) for path in [root / "manifest.json", *paths.values()]}
    wanted, rows = defaultdict(set), {}
    for _, episode, index, _, _ in _REAL_CASES:
        wanted[episode].add(index)
    for episode, indices in wanted.items():
        with paths[episode].open() as stream:
            for index, line in enumerate(stream):
                if index in indices:
                    rows[episode, index] = json.loads(line)
                if index >= max(indices):
                    break
    adapter = SMORESAdapter(c.load_robot_spec(ROBOT_CONFIG))
    config = json.loads(TASK_CONFIG.read_text())
    selected, audit = {}, []
    for case, episode, index, expected_task, group in _REAL_CASES:
        row = rows[episode, index]
        assert row["provenance"]["canonical_transition_index"] == index
        assert row["supervision"]["valid_for_behavior_cloning"] is True
        assert "graph_t_plus_1" in row
        metadata = read_task_metadata(row, ROOT)
        context, observation = build_task_view(row, metadata, config)
        graph = adapter.encode_state(row["graph_t"])
        inputs = selector()(graph, observation, context)
        selected[case] = (row, graph, observation, context, inputs)
        audit.append({"case": case, "episode_id": episode, "canonical_index": index,
                      "expected_task": expected_task, "canonical_morphology_audit_only": row["graph_t"].get("morphology"),
                      "source_records": row["provenance"]["source_records"],
                      "context": asdict(inputs.context), "observation": asdict(inputs.observation),
                      "graph": asdict(inputs.graph), "source_audit": metadata["audit"]})
    out = tmp_path_factory.mktemp("d2-targeted-real-rows") / "audit.json"
    out.write_text(json.dumps(audit, indent=2))
    yield selected
    assert before == {path: signature(path) for path in before}, "Canonical v6 changed during targeted regression"


@pytest.mark.parametrize("case,episode,index,task,geometry", _REAL_CASES)
def test_six_targeted_real_v6_rows_keep_current_physical_task_inputs(real_policy_rows, case, episode, index, task, geometry):
    row, graph, observation, context, selected = real_policy_rows[case]
    assert selected.graph == graph and selected.context == context and selected.observation == observation
    assert selected.context.task_id == task
    assert any(geometry in feature.name for feature in selected.context.static_geometry)
    assert set(asdict(selected.context)) == {"task_id", "static_geometry"}
    assert not hasattr(selected, "graph_t_plus_1")
    changed = copy.deepcopy(row)
    changed.update(graph_t_plus_1={"nodes": []}, target_morphology="OPPOSITE", action_t={"primitive": "OPPOSITE"},
                   supervision={"valid_for_behavior_cloning": False}, success=False,
                   stage_name="OPPOSITE", control_state={"phase": "APPROACH"}, secret_future_hint={"task": "OPPOSITE"})
    changed["graph_t"].update(root_module_id="OPPOSITE", morphology="OPPOSITE", target_graph={"nodes": []})
    for node in changed["graph_t"]["nodes"]:
        node.update(role="OPPOSITE", template_id="OPPOSITE")
    assert selector()(SMORESAdapter(c.load_robot_spec(ROBOT_CONFIG)).encode_state(changed["graph_t"]), observation, context) == selected
    # Next graph was deliberately changed only in a detached test copy.
    assert "graph_t_plus_1" in row and row["graph_t_plus_1"] != changed["graph_t_plus_1"]
