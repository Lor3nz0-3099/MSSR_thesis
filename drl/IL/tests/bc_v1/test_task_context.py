"""D1: causal semantic context and framed physical values, without tensors."""

import copy
from dataclasses import asdict
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.tests.bc_v1 import fixtures as f

ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "configs/bc_v1/task_context.json"


def api():
    name = "drl.IL.bc_v1.task_context"
    assert importlib.util.find_spec(name) is not None, "Missing D1 causal task-view behavior"
    module = importlib.import_module(name)
    assert callable(getattr(module, "read_task_metadata", None))
    assert callable(getattr(module, "build_task_view", None))
    return module


def view(row=None, metadata=None):
    module = api()
    config = json.loads(CONFIG.read_text())
    return module.build_task_view(row or {}, metadata or {}, config)


def mission_view(index=100, *, mutate=None):
    row, metadata, protocol = f.mission_task_fixture(index)
    config = json.loads(CONFIG.read_text())
    config["historical_mission_imputations"] = [protocol]
    if mutate:
        mutate(row, metadata, config)
    return api().build_task_view(row, metadata, config)


@pytest.mark.parametrize("index,task", [(100, "gap"), (199, "gap"), (300, "stairs"),
                                       (399, "stairs"), (500, "navigation"), (599, "navigation"), (700, "button"), (799, "button"),
                                       (900, "navigation"), (999, "navigation")])
def test_historical_mission_successor_is_constant_across_structural_interval(index, task):
    assert mission_view(index)[0].task_id == task


def test_historical_task_switch_does_not_revert_to_predecessor_after_interval():
    def stale_current(row, metadata, config):
        row["observation_t"].update(task_type="navigation", task_id="nav-1")
    assert mission_view(99, mutate=stale_current)[0].task_id == "navigation"
    assert mission_view(100, mutate=stale_current)[0].task_id == "gap"
    assert mission_view(200, mutate=stale_current)[0].task_id == "gap"
    assert mission_view(299, mutate=stale_current)[0].task_id == "gap"


def test_current_geometric_mission_locator_can_advance_after_interval_without_future_backfill(tmp_path):
    def current_stairs(row, metadata, config):
        row["observation_t"].update(task_id="stairs-3", task_type="stairs")
        for task in metadata["course"]["mission"]["tasks"]:
            if task["type"] != "goal":
                task["parameters"]["exit_pose_xyyaw"] = [10., 0., 0.]
        raw = {"stamp": 1., "observation": {"course": metadata["course"]}}
        (tmp_path / "current.jsonl").write_text(json.dumps(raw) + "\n")
        row["provenance"]["source_records"] = [{"_loader_source_path": "current.jsonl",
                                                "_loader_source_row": 1}]
        metadata.clear()
        metadata.update(api().read_task_metadata(row, tmp_path))
    assert mission_view(250, mutate=current_stairs)[0].task_id == "stairs"
    assert mission_view(150, mutate=current_stairs)[0].task_id == "gap"
    assert mission_view(250)[0].task_id == "gap"


@pytest.mark.parametrize("location", ["course", "mission"])
def test_explicit_course_cursor_is_read_without_losing_inherited_geometry_frame(location):
    def current_stairs(row, metadata, config):
        target = metadata["course"] if location == "course" else metadata["course"]["mission"]
        target["current_task_id"] = "stairs-3"
    assert mission_view(250, mutate=current_stairs)[0].task_id == "stairs"
    with pytest.raises(ValueError, match="contradicts historical"):
        mission_view(150, mutate=current_stairs)


@pytest.mark.parametrize("stage", ["assembly", "reconfiguration", "task_execution", "navigation"])
def test_audit_stage_is_inaccessible_from_policy_context(stage):
    context, observation = view(f.task_row("gap", stage), f.gap_metadata())
    serialized = asdict(c.PolicyInputs(c.StateGraph(), observation, context))
    assert set(serialized["context"]) == {"task_id", "static_geometry"}
    assert stage not in str(serialized) or stage == "gap"


@pytest.mark.parametrize("task", ["gap", "stairs", "navigation", "button"])
def test_single_task_current_course_certifies_expert_assembly_task(task):
    context, _ = view(f.task_row(stage="assembly"), {"course": {"frame_id": "world", task: {}}})
    assert context.task_id == task


def test_current_task_static_geometry_and_dynamic_button_measurement_have_separate_owners():
    context, observation = mission_view(700)
    static = {x.name: x for x in context.static_geometry}
    dynamic = features(observation)
    assert static["world.button.center_xyz_m"].values == (9., 0., .2)
    assert static["button.plunger_stroke_m"].values == (.004,)
    assert dynamic["world.button.current_center_xyz_m"].values == (8.998, 0., .2)
    assert dynamic["button.depression_m"].values == (.002,)
    assert not any("depression" in name or "current_center" in name for name in static)


def test_current_segment_excludes_future_fixtures_and_semantics():
    context, observation = mission_view(100)
    values = {x.name: x.values for x in context.static_geometry}
    assert values["world.gap.near_edge_world_xyz_m"] == (3., 0., 0.)
    assert values["world.obstacles.0.center_xyz_m"] == (3., 0., -.1)
    assert not any("stairs" in name or "button" in name for name in values)
    assert not observation.features
    assert "mission-fixture" not in repr(context)


@pytest.mark.parametrize("container", ["physical_track", "layout"])
def test_nested_mission_physical_inventory_retains_current_task_ownership(container):
    def nested_inventory(row, metadata, config):
        course = metadata["course"]
        course[container] = {"collision_boxes": course.pop("collision_boxes")}
    context, _ = mission_view(100, mutate=nested_inventory)
    values = [x.values for x in context.static_geometry if x.name.endswith("center_xyz_m")]
    assert values == [(3., 0., -.1)]


@pytest.mark.parametrize("placement", ["action_t", "target_graph", "graph_t_plus_1",
                                        "next_record", "control_state", "configuration_goal",
                                        "target_morphology", "final_topology", "success", "reward"])
def test_imputation_never_reads_actions_targets_future_or_outcome(placement):
    def mutate(row, metadata, config):
        row[placement] = {"task_type": "button", "task_id": "button-4", "morphology": "FORBIDDEN"}
    assert mission_view(100, mutate=mutate) == mission_view(100)


def test_protocol_imputation_is_audit_evidence_not_explicit_raw_task():
    row, metadata, protocol = f.mission_task_fixture()
    config = json.loads(CONFIG.read_text());config["historical_mission_imputations"] = [protocol]
    resolver = getattr(api(), "resolve_mission_task", None)
    assert callable(resolver), "Missing D1.2 auditable mission-protocol resolver"
    evidence = resolver(row, metadata, config)
    assert evidence["task_id"] == "gap-2"
    assert evidence["evidence_class"] == "HISTORICAL_PROTOCOL_IMPUTATION"
    assert "evidence_class" not in asdict(api().build_task_view(row, metadata, config)[0])


@pytest.mark.parametrize("mutation", ["missing_successor", "overlap", "wrong_plan"])
def test_ambiguous_or_conflicting_mission_bindings_fail(mutation):
    def change(row, metadata, config):
        p = config["historical_mission_imputations"][0]
        if mutation == "missing_successor":
            p["intervals"][0]["successor_task_id"] = "absent"
        elif mutation == "overlap":
            p["intervals"].append({**p["intervals"][0], "successor_task_id": "stairs-3"})
        else:
            p["mission_order"][1]["type"] = "button"
    with pytest.raises(ValueError, match="mission|[Cc]ontradict|[Oo]verlap|successor"):
        mission_view(100, mutate=change)


def test_imputation_does_not_mutate_raw_metadata_or_config():
    row, metadata, protocol = f.mission_task_fixture()
    config = json.loads(CONFIG.read_text());config["historical_mission_imputations"] = [protocol]
    before = copy.deepcopy((row, metadata, config))
    api().build_task_view(row, metadata, config)
    assert (row, metadata, config) == before


def test_explicit_raw_problem_semantics_conflicting_with_imputation_fail():
    def raw_button(row, metadata, config):
        metadata["sources"] = [{"task_type": "button"}]
    with pytest.raises(ValueError, match="[Cc]ontradict.*task"):
        mission_view(100, mutate=raw_button)


def features(bundle):
    return {x.name: x for x in bundle.features}


def statics(context):
    return {x.name: x for x in context.static_geometry}


def test_unassigned_multi_task_geometry_fails_instead_of_exposing_future_inventory():
    _, metadata, _ = f.mission_task_fixture()
    with pytest.raises(ValueError, match="[Uu]nresolved.*mission|[Uu]nassigned.*mission"):
        view(metadata=metadata)


def test_current_mission_requires_certified_physical_object_ownership():
    def mutate(row, metadata, config):
        metadata["course"]["collision_boxes"].append({"center_xyz_m": [99., 0., 0.]})
    with pytest.raises(ValueError, match="ownership|[Uu]nscoped"):
        mission_view(100, mutate=mutate)


@pytest.mark.parametrize("task", ["UNKNOWN", "navigation", "gap", "stairs", "button"])
def test_task_vocabulary(task):
    context, observation = view(f.task_row(task))
    assert isinstance(context, c.TaskContext) and isinstance(observation, c.ObservationBundle)
    assert context.task_id == (None if task == "UNKNOWN" else task)


@pytest.mark.parametrize("stage", ["UNKNOWN", "assembly", "reconfiguration", "task_execution"])
def test_stage_vocabulary(stage):
    context, _ = view(f.task_row(stage=stage))
    assert not hasattr(context, "stage_id")


def test_missing_context_does_not_use_next_record_or_action():
    row = {"action_t": {"primitive": "press_button", "task_type": "button"},
           "graph_t_plus_1": {"course": f.gap_metadata()["course"]},
           "observation_t_plus_1": {"task_type": "gap", "stage_name": "assembly"},
           "next_record": f.task_row("stairs", "task_execution"),
           "target_graph": {"task_type": "navigation"}, "target_morphology": "snake8",
           "supervision": {"task_type": "button"}, "reward": 1.0, "success": True}
    assert view(row) == (c.TaskContext(), c.ObservationBundle())


@pytest.mark.parametrize("name", ["rc_car8", "snake8", "mobile_manipulator8",
                                  "DISPATCHING_ALIGN", "self_reconfiguration"])
def test_nonsemantic_names_are_not_task_categories(name):
    context, _ = view(f.task_row(name))
    assert context.task_id is None


def test_no_dispatching_or_control_state_inference():
    row = {"fsm_state": "DISPATCHING_GAP", "stage_name": "DISPATCHING_ALIGN",
           "graph_t": {"nodes": [{"control_state": {"task_type": "button",
                        "active_primitives": ["gravity_settle"]}}]},
           "debug": {"phase": "REACH"}}
    assert view(row) == (c.TaskContext(), c.ObservationBundle())


def test_current_geometry_preserves_values_validity_and_source_frame():
    context, observation = view(f.task_row("gap", "task_execution"), f.gap_metadata())
    assert statics(context)["gap.width_m"].values == (.2,)
    assert statics(context)["world.gap.near_edge_x_m"] == c.NumericFeature(
        "world.gap.near_edge_x_m", (1.0,))
    assert statics(context)["world.gap.far_edge_x_m"].values == (1.2,)


def test_missing_frame_does_not_become_world():
    context, observation = view(f.task_row("gap"), f.gap_metadata(frame=None))
    assert statics(context)["unframed.gap.near_edge_x_m"].valid is False


def test_missing_geometry_stays_missing():
    context, observation = view(f.task_row("stairs", "task_execution"))
    assert not context.static_geometry
    assert observation.features == ()


def test_equivalent_duplicate_metadata_agrees():
    row = f.task_row("gap_crossing", "behavior")
    metadata = {"sources": [{"task_type": "gap", **f.gap_metadata()},
                            {"task_type": "gap_crossing", **f.gap_metadata(.20000000000000004)}]}
    assert view(row, metadata) == view(row, f.gap_metadata())


@pytest.mark.parametrize("metadata", [
    {"sources": [f.gap_metadata(), f.gap_metadata(.3)]},
    {"task_type": "stairs"},
])
def test_contradictory_authoritative_metadata_fails(metadata):
    with pytest.raises(ValueError, match="[Cc]ontradict"):
        view(f.task_row("gap", "task_execution"), metadata)


@pytest.mark.parametrize("schema", ["mssr.expert_compact_phase.v1", "mssr.expert_compact_phase.v2"])
def test_explicit_phase_constants_are_scoped_and_never_temporally_filled(tmp_path, schema):
    source = tmp_path / "behavior.jsonl"
    source.write_text(json.dumps({"stamp": 1., "observation": {}}) + "\n")
    (tmp_path / "behavior_manifest.json").write_text(json.dumps({
        "schema_version": schema, "phase": "behavior",
        "top_level_constants": {"task_type": "gap_crossing"},
        "observation_constants": f.gap_metadata(),
        "next_observation_constants": {"course": {"gap": {"width_m": 99.}}}}))
    row = {**f.task_row(), "provenance": {"source_records": [
        {"_loader_source_path": "behavior.jsonl", "_loader_source_row": 1}]}}
    metadata = api().read_task_metadata(row, tmp_path)
    assert view(row, metadata)[0].task_id == "gap"
    assert not hasattr(view(row, metadata)[0], "stage_id")
    assert view(f.task_row()) == (c.TaskContext(), c.ObservationBundle())
    assert view(f.task_row("stairs", "task_execution"))[0].task_id == "stairs"


def test_standalone_assembly_does_not_invent_target_task():
    row = {"task_type": "parallel_self_assembly", "stage_name": "parallel_self_assembly",
           "observation_t": {"target_morphology": "mobile_manipulator8"}}
    context, _ = view(row)
    assert context.task_id is None and not hasattr(context, "stage_id")


def test_reconfiguration_is_explicit_current_stage_not_morphology():
    context, _ = view({"task_type": "self_reconfiguration"})
    assert context.task_id is None and not hasattr(context, "stage_id")
    assert view({"graph_t": {"morphology": "reconfiguring"}})[0].task_id is None


def test_button_depression_changes_observation_but_future_success_does_not():
    row = f.task_row("button", "task_execution")
    row["observation_t"]["relative_geometry"] = {"depression_m": .001}
    context, observation = view(row)
    assert features(observation)["button.depression_m"].values == (.001,)
    changed = copy.deepcopy(row)
    changed.update(success=True, reward=99., observation_t_plus_1={"depression_m": .004})
    assert view(changed) == (context, observation)
    changed["observation_t"]["relative_geometry"]["depression_m"] = .002
    assert view(changed)[1] != observation


def test_cones_are_physical_and_never_waypoints():
    row = f.task_row("navigation")
    metadata = {"course": {"frame_id": "world", "navigation": {
        "waypoints_xyyaw": [[9., 8., 0.]], "nominal_route": {"waypoints_xyyaw": [[7., 6., 0.]]}},
        "physical_track": {"frame_id": "map", "cone_centers_xy_m": [[1., 2.], [3., 4.]],
                           "centerline_xy_m": [[20., 20.]], "cone_radius_m": .05}}}
    result = view(row, metadata)
    assert statics(result[0])["map.cones.0.center_xy_m"].values == (1., 2.)
    changed = copy.deepcopy(metadata)
    changed["course"]["navigation"] = {"waypoints_xyyaw": [[99., 99., 0.]]}
    changed["course"]["physical_track"]["centerline_xy_m"] = [[30., 30.]]
    assert view(row, changed) == result
    assert view(row, {"course": {"navigation": changed["course"]["navigation"]}})[1].features == ()


def test_mission_lookup_requires_current_task_id_and_does_not_expose_future_route():
    course = {"frame_id": "world", "mission": {"tasks": [
        {"task_id": "current", "type": "gap", "parameters": f.gap_metadata()["course"]},
        {"task_id": "later", "type": "button", "parameters": {"button": {"depression_m": .004}}}]}}
    row = f.task_row("gap")
    row["observation_t"]["task_id"] = "current"
    context, observation = view(row, {"course": course})
    assert context.task_id == "gap"
    assert not any("button" in x.name for x in observation.features)
    assert view(f.task_row(), {"course": course}) == (c.TaskContext(), c.ObservationBundle())


def test_no_frame_transform_and_stage_frame_is_preserved():
    metadata = {"task_parameters": {"geometry_frame": "world",
        "gap": {"coordinate_frame": "stage_local", "near_edge_x_m": 1., "width_m": .2},
        "stage_frame": {"origin_world_xyz_m": [4., 5., 0.], "yaw_rad": 1.}}}
    context, observation = view(f.task_row("gap"), metadata)
    assert statics(context)["stage_local.gap.near_edge_x_m"].values == (1.,)
    assert statics(context)["world.task_frame.origin_xyz_m"].values == (4., 5., 0.)


def test_raw_inputs_not_mutated_and_audit_data_not_features(tmp_path):
    row = f.task_row("gap", "task_execution")
    row["provenance"] = {"episode_id": "SECRET", "source_records": []}
    metadata = {**f.gap_metadata(), "audit": {"path": "SECRET", "hash": "SECRET"}}
    before = copy.deepcopy((row, metadata))
    context, observation = view(row, metadata)
    assert (row, metadata) == before
    assert "SECRET" not in repr((asdict(context), asdict(observation)))


def test_missing_source_preserves_missingness_and_does_not_infer_from_filename(tmp_path):
    row = {"provenance": {"source_records": [{"_loader_source_path": "assembly.jsonl",
            "_loader_source_row": 1, "_loader_phase": "assembly"}]}}
    metadata = api().read_task_metadata(row, tmp_path)
    assert view(row, metadata) == (c.TaskContext(), c.ObservationBundle())


def test_source_record_join_reads_only_referenced_current_record(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps({"stamp": 1., "task_type": "gap",
        "stage_name": "behavior", "observation": f.gap_metadata()}) + "\n" +
        json.dumps({"stamp": 2., "task_type": "button", "success": True}) + "\n")
    row = {**f.task_row(), "provenance": {"source_records": [
        {"_loader_source_path": "records.jsonl", "_loader_source_row": 1}]}}
    metadata = api().read_task_metadata(row, tmp_path)
    assert view(row, metadata)[0].task_id == "gap"
    assert "success" not in repr(metadata)
    assert "expert_action" not in repr(metadata)


def test_source_record_with_different_physical_stamp_is_rejected(tmp_path):
    (tmp_path / "records.jsonl").write_text(json.dumps({"graph_t": {"stamp": 2.},
                                                       "task_type": "button"}) + "\n")
    row = {**f.task_row(), "provenance": {"source_records": [
        {"_loader_source_path": "records.jsonl", "_loader_source_row": 1}]}}
    with pytest.raises(ValueError, match="stamp"):
        api().read_task_metadata(row, tmp_path)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), "0.2"])
def test_invalid_physical_numeric_values_fail(value):
    with pytest.raises(ValueError, match="finite|numeric"):
        view(f.task_row("gap"), f.gap_metadata(value))


def test_import_isolation():
    api()
    code = """
import sys
class Block:
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'rclpy', 'rospy', 'isaacsim', 'isaaclab', 'omni', 'smores_ep', 'numpy', 'torch'}:
            raise AssertionError('Forbidden dependency: ' + name)
sys.meta_path.insert(0, Block())
from drl.IL.bc_v1.task_context import build_task_view, read_task_metadata
"""
    subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, check=True)


def test_conflicting_source_frames_fail_without_attempting_transformation():
    with pytest.raises(ValueError, match="[Cc]ontradict.*frame"):
        view(f.task_row("gap"), {"sources": [f.gap_metadata(frame="world"),
                                             f.gap_metadata(frame="map")]})


def test_contradictory_physical_object_counts_fail():
    def source(centers):
        return {"course": {"physical_track": {"frame_id": "world",
                                               "cone_centers_xy_m": centers}}}
    with pytest.raises(ValueError, match="[Cc]ontradict"):
        view(f.task_row("navigation"), {"sources": [source([[1., 2.]]),
                                                     source([[1., 2.], [3., 4.]])]})


def test_current_goal_requires_explicit_physical_field_and_keeps_its_frame():
    row = f.task_row("navigation", "task_execution")
    metadata = {"current_goal": {"frame_id": "map", "xyyaw": [1., 2., .3]}}
    assert statics(view(row, metadata)[0])["map.goal.xyyaw"].values == (1., 2., .3)


def test_input_validity_is_preserved():
    metadata = f.gap_metadata()
    metadata["course"]["gap"]["near_edge_x_m"] = {"value": 1., "valid": False}
    assert not statics(view(f.task_row("gap"), metadata)[0])["world.gap.near_edge_x_m"].valid


def test_uncertified_canonical_root_geometry_is_not_treated_as_world():
    row = f.task_row("gap")
    row["observation_t"]["relative_geometry"] = {"near_edge_relative_xyz_m": [1., 2., 3.]}
    assert not view(row)[1].features


def test_compact_current_constants_contradicting_current_row_fail(tmp_path):
    (tmp_path / "behavior.jsonl").write_text(json.dumps({"stamp": 1., "task_type": "stairs"}) + "\n")
    (tmp_path / "behavior_manifest.json").write_text(json.dumps({
        "schema_version": "mssr.expert_compact_phase.v2", "phase": "behavior",
        "top_level_constants": {"task_type": "gap"}}))
    row = {**f.task_row(), "provenance": {"source_records": [
        {"_loader_source_path": "behavior.jsonl", "_loader_source_row": 1}]}}
    with pytest.raises(ValueError, match="[Cc]ontradict"):
        view(row, api().read_task_metadata(row, tmp_path))


def test_conflicting_frame_aliases_in_one_geometry_fail():
    metadata = f.gap_metadata()
    metadata["course"]["gap"].update(frame_id="world", coordinate_frame="map")
    with pytest.raises(ValueError, match="[Cc]ontradict.*frame"):
        view(f.task_row("gap"), metadata)


def test_two_unsupported_frame_labels_do_not_silently_agree():
    with pytest.raises(ValueError, match="[Cc]ontradict.*frame"):
        view(f.task_row("gap"), {"sources": [f.gap_metadata(frame="unknown-A"),
                                             f.gap_metadata(frame="unknown-B")]})


def test_collision_box_uses_its_explicit_frame_instead_of_parent_frame():
    metadata = {"environment_geometry": {"frame_id": "world", "collision_boxes": [
        {"frame_id": "map", "center_xyz_m": [1., 2., 3.]}]}}
    assert statics(view(metadata=metadata)[0])["map.obstacles.0.center_xyz_m"].valid


def test_duplicate_collision_boxes_with_conflicting_frames_fail():
    def source(frame):
        return {"environment_geometry": {"frame_id": "world", "collision_boxes": [
            {"frame_id": frame, "center_xyz_m": [1., 2., 3.]}]}}
    with pytest.raises(ValueError, match="[Cc]ontradict.*frame"):
        view(metadata={"sources": [source("world"), source("map")]})


def test_course_inherits_explicit_current_observation_frame():
    row = f.task_row("gap")
    row["observation_t"].update(frame_id="map", course={"gap": {"near_edge_x_m": 1.}})
    assert statics(view(row)[0])["map.gap.near_edge_x_m"].valid


@pytest.mark.parametrize("schema", ["mssr.expert_compact_phase.v1", "mssr.expert_compact_phase.v2"])
def test_raw_and_compact_current_observations_have_identical_frame_scope(tmp_path, schema):
    raw = {"stamp": 1., "observation": {"frame_id": "map", "course": {
        "gap": {"near_edge_x_m": 1.}}}}
    compact = copy.deepcopy(raw)
    del compact["observation"]["frame_id"]
    (tmp_path / "raw.jsonl").write_text(json.dumps(raw) + "\n")
    (tmp_path / "compact.jsonl").write_text(json.dumps(compact) + "\n")
    (tmp_path / "compact_manifest.json").write_text(json.dumps({
        "schema_version": schema, "observation_constants": {"frame_id": "map"},
        "next_observation_constants": {"frame_id": "world"}}))
    def loaded(file):
        row = {**f.task_row("gap"), "provenance": {"source_records": [
            {"_loader_source_path": file, "_loader_source_row": 1}]}}
        return view(row, api().read_task_metadata(row, tmp_path))
    assert loaded("raw.jsonl") == loaded("compact.jsonl")
    assert statics(loaded("compact.jsonl")[0])["map.gap.near_edge_x_m"].valid


def test_compact_observation_frame_constants_conflicting_with_current_row_fail(tmp_path):
    (tmp_path / "compact.jsonl").write_text(json.dumps({"stamp": 1., "observation": {
        "frame_id": "world", "course": {"gap": {"near_edge_x_m": 1.}}}}) + "\n")
    (tmp_path / "compact_manifest.json").write_text(json.dumps({
        "schema_version": "mssr.expert_compact_phase.v2", "observation_constants": {"frame_id": "map"}}))
    row = {**f.task_row("gap"), "provenance": {"source_records": [
        {"_loader_source_path": "compact.jsonl", "_loader_source_row": 1}]}}
    with pytest.raises(ValueError, match="[Cc]ontradict.*frame"):
        view(row, api().read_task_metadata(row, tmp_path))


def test_current_mission_task_geometry_inherits_observation_frame():
    row = f.task_row("gap")
    row["observation_t"].update(frame_id="map", task_id="current", course={"mission": {"tasks": [
        {"task_id": "current", "type": "gap", "parameters": {"gap": {"near_edge_x_m": 1.}}}]}})
    assert statics(view(row)[0])["map.gap.near_edge_x_m"].valid


def test_real_composite_gap_preserves_world_borders_and_stage_local_edges():
    metadata = {"task_parameters": {"geometry_frame": "world", "gap": {
        "coordinate_frame": "stage_local", "near_edge_x_m": -.3,
        "near_edge_center_world_xyz_m": [1.08, 0., 0.],
        "far_edge_center_world_xyz_m": [1.258, 0., 0.],
        "crossing_direction_world_xy": [1., 0.], "width_m": .178}}}
    context, observation = view(f.task_row("gap"), metadata)
    assert statics(context)["world.gap.near_edge_center_world_xyz_m"].values == (1.08, 0., 0.)
    assert statics(context)["stage_local.gap.near_edge_x_m"].values == (-.3,)
    assert statics(context)["gap.width_m"].values == (.178,)


# Configuration destinations are forbidden BC-v1 inputs, even when causal.
@pytest.mark.parametrize("stage", ["assembly", "reconfiguration", "task_execution"])
@pytest.mark.parametrize("placement", ["row", "observation", "metadata", "source", "task_context"])
def test_configuration_goal_cannot_change_task_or_observation(stage, placement):
    row = f.task_row("stairs", stage)
    metadata = f.gap_metadata()
    baseline = view(row, metadata)
    before = copy.deepcopy((row, metadata))
    for destination in ["snake8", "rc_car8", "mobile_manipulator8", "arbitrary_robot"]:
        changed_row, changed_metadata = copy.deepcopy(before)
        if placement == "row":
            changed_row["configuration_goal"] = destination
        elif placement == "observation":
            changed_row["observation_t"]["configuration_goal"] = destination
        elif placement == "metadata":
            changed_metadata["configuration_goal"] = destination
        elif placement == "source":
            changed_metadata["sources"] = [{"configuration_goal": destination}]
        else:
            changed_metadata["task_context"] = {"configuration_goal": destination}
        snapshot = copy.deepcopy((changed_row, changed_metadata))
        assert view(changed_row, changed_metadata) == baseline
        assert (changed_row, changed_metadata) == snapshot
    assert (row, metadata) == before


def test_removed_goal_rules_cannot_reintroduce_policy_input():
    config = json.loads(CONFIG.read_text())
    config.update(configuration_goal_vocabulary=["UNKNOWN", "snake"],
                  configuration_goal_aliases={"snake8": "snake"})
    assert api().build_task_view(f.task_row(stage="reconfiguration"),
        {"configuration_goal": "snake8"}, config) == view(f.task_row(stage="reconfiguration"))


def referenced_row(path, index=1, stamp=1.):
    return {"graph_t": {"stamp": stamp}, "provenance": {"source_records": [
        {"_loader_source_path": path, "_loader_source_row": index}]}}


@pytest.mark.parametrize("stage,task_type", [("assembly", "parallel_self_assembly"),
                                            ("reconfiguration", "self_reconfiguration")])
def test_source_reader_drops_recorded_destination_but_keeps_environment(tmp_path, stage, task_type):
    source = tmp_path / "current.jsonl"
    original = {"schema_version": "mssr.expert_transition.v3", "task_type": task_type,
        "stamp": 1., "configuration_goal": "snake8",
        "task_metrics": {"target_morphology": "snake8", "completed_operation_count": 3},
        "observation": {"configuration_goal": "snake8", "course": {"frame_id": "world",
            "collision_boxes": [{"center_xyz_m": [1., 2., .3], "size_xyz_m": [.4,.5,.6]}]}},
        "expert_action": {"configuration_goal": "snake8"},
        "target_graph": {"global_attributes": {"morphology_name": "snake8"}}}
    source.write_text(json.dumps(original) + "\n")
    row = referenced_row("current.jsonl")
    first_metadata = api().read_task_metadata(row, tmp_path)
    context, observation = view(row, first_metadata)
    assert not hasattr(context, "stage_id") and context.task_id is None
    assert statics(context)["world.obstacles.0.center_xyz_m"].values == (1., 2., .3)
    assert "configuration_goal" not in repr(first_metadata["sources"])
    assert "target_morphology" not in repr(first_metadata["sources"])
    changed = copy.deepcopy(original)
    changed["configuration_goal"] = "mobile_manipulator8"
    changed["task_metrics"]["target_morphology"] = "mobile_manipulator8"
    source.write_text(json.dumps(changed) + "\n")
    assert view(row, api().read_task_metadata(row, tmp_path)) == (context, observation)


def test_reader_does_not_consume_configuration_commands_or_startup_logs(tmp_path):
    folder = tmp_path / "structural"; folder.mkdir()
    (folder / "chosen.jsonl").write_text(json.dumps({"stamp": 1., "stage_name": "self_assembly"}) + "\n")
    (tmp_path / "source_manifest.json").write_text(json.dumps({
        "schema_version": "mssr.teleop_recording_manifest.v1", "structural_streams": [
            {"stream_id": "chosen", "path": "/original/chosen.jsonl", "phase": "self_assembly"}],
        "events": [{"kind": "command", "name": "select_mm8", "stamp_monotonic": .5}]}))
    row = referenced_row("structural/chosen.jsonl")
    # Load the real config as the public reader would, including any stale rules.
    config_dir = tmp_path / "configs/bc_v1"; config_dir.mkdir(parents=True)
    (config_dir / "task_context.json").write_text(CONFIG.read_text())
    metadata = api().read_task_metadata(row, tmp_path)
    assert view(row, metadata) == (c.TaskContext(), c.ObservationBundle())
    assert all(not entry["path"].endswith("source_manifest.json") for entry in metadata["audit"])


def test_active_context_config_has_no_destination_vocabulary_or_imputation():
    config = json.loads(CONFIG.read_text())
    assert not any(key.startswith("configuration_goal") for key in config)
    assert "assembly_intent" not in config and "recorded_intent_rules" not in config


def test_dynamic_geometry_changes_observation_without_changing_task_stage():
    row = f.task_row("button", "reconfiguration")
    metadata = {"course": {"frame_id": "world", "button": {"center_xyz_m": [1., 2., .1], "depression_m": .001}}}
    first = view(row, metadata)
    metadata["course"]["button"]["depression_m"] = .002
    changed = view(row, metadata)
    assert first[0] == changed[0] and first[1] != changed[1]


@pytest.mark.parametrize('stage', ['assembly', 'reconfiguration', 'task_execution'])
def test_unassigned_mission_is_rejected_independently_of_audit_stage(stage):
    _, metadata, _ = f.mission_task_fixture()
    before = copy.deepcopy(metadata)
    with pytest.raises(ValueError, match="Unresolved current mission task"):
        view(f.task_row(stage=stage), metadata)
    assert metadata == before


def test_current_task_geometry_and_owned_physical_inventory_coexist():
    row = f.task_row('navigation', 'reconfiguration')
    row['observation_t']['task_id'] = 'active'
    metadata = {'course': {'frame_id': 'world',
        'navigation_cones': [{'task_id': 'active', 'center_xyz_m': [1., 2., .1]},
                             {'task_id': 'future', 'center_xyz_m': [3., 4., .1]}],
        'mission': {'tasks': [{'task_id': 'active', 'type': 'navigation', 'parameters': {
            'cone_centers_xy_m': [[1., 2.]], 'waypoints_xyyaw': [[8., 8., 0.]]}},
            {'task_id': 'future', 'type': 'navigation', 'parameters': {}}]}}}
    context, obs = view(row, metadata)
    assert context.task_id == 'navigation'
    assert statics(context)['world.environment_cones.0.center_xyz_m'].values == (1., 2., .1)
    assert 'world.environment_cones.1.center_xyz_m' not in statics(context)
    assert statics(context)['world.cones.0.center_xy_m'].values == (1., 2.)
    assert not any('waypoint' in feature.name for feature in context.static_geometry)


def test_physical_course_missing_frame_is_preserved_but_uncertified():
    metadata = {'course': {'mission': {'tasks': []},
                          'navigation_cones': [{'center_xyz_m': [1., 2., 3.]}],
                          'collision_boxes': [{'center_xyz_m': [4., 5., 6.]}]}}
    context, obs = view(f.task_row(stage='assembly'), metadata)
    assert not statics(context)['unframed.environment_cones.0.center_xyz_m'].valid
    assert not statics(context)['unframed.obstacles.0.center_xyz_m'].valid
