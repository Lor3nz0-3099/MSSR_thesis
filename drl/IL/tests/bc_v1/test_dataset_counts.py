"""B2: independently frozen v6 expectations and small adversarial collections.

The session fixture performs at most ONE real census per pytest invocation.
MSSR_BC_V1_SKIP_REAL_CENSUS=1 explicitly selects only synthetic/unit gates;
the default includes the full immutable source when it is present.
"""

import copy
from collections import Counter
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import pytest

from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.bc_v1 import smores_adapter as sm
from drl.IL.tests.bc_v1 import fixtures as f

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "datasets/expert_v1_canonical_v6"
COUNTS = ROOT / "configs/bc_v1/counts.v1.json"


def test_g1_materialized_cache_has_frozen_v6_split_counts():
    """Independent G1 gate; real prepare must run explicitly before this check."""
    from drl.IL.bc_v1.dataset import BCDataset, read_cache_manifest
    path = ROOT/'artifacts/bc_v1/cache/g1_v1'
    if not (path/'manifest.json').is_file():
        pytest.skip('Official G1 cache has not been materialized')
    manifest = read_cache_manifest(path)
    assert manifest['schema_version'] == 'mssr.bc_v1.physical_cache.v2'
    assert 'normalizer' not in manifest and 'normalizer_fingerprint' not in manifest['identity']
    assert manifest['counts'] == {'episodes': 73, 'samples': 119600, 'splits': {
        'train': {'episodes': 49, 'samples': 78302}, 'validation': {'episodes': 12, 'samples': 22762},
        'test': {'episodes': 12, 'samples': 18536}}}
    split = json.loads((ROOT/'artifacts/bc_v1/splits/f1_v1/split.json').read_text())
    assert manifest['splits'] == split['splits']
    assert manifest['excluded_episode_ids'] == split['excluded_episode_ids']
    assert len(manifest['excluded_episode_ids']) == 7
    for part, episodes, samples in [('train', 49, 78302), ('validation', 12, 22762), ('test', 12, 18536)]:
        dataset = BCDataset(path, part)
        assert len(dataset.episode_ids) == episodes and len(dataset) == samples


def api():
    assert callable(getattr(sm, "audit_canonical_source", None)), "Missing B2 audit_canonical_source behavior"
    return sm.audit_canonical_source


def adapter():
    return sm.SMORESAdapter(c.load_robot_spec(ROOT / "configs/bc_v1/smores_ep.robot.json"))


@pytest.fixture(scope="session")
def frozen():
    return json.loads(COUNTS.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def v6_audit(tmp_path_factory):
    audit = api()  # Missing behavior must fail before any environment skip.
    if os.environ.get("MSSR_BC_V1_SKIP_REAL_CENSUS") == "1":
        pytest.skip("Explicit synthetic-only gate; real census runs in the final suite")
    if not (SOURCE / "manifest.json").is_file():
        pytest.skip("Canonical v6 is not installed; synthetic audit tests still run")
    result = audit(SOURCE, adapter())
    # An ephemeral report lets the final review inspect this SAME census.
    report = tmp_path_factory.mktemp("b2-audit") / "audit.json"
    report.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def test_unfiltered_source_matches_frozen_counts(v6_audit, frozen):
    assert v6_audit["source"] == frozen["source"]
    for key in ("episodes", "episodes_by_kind", "bc_useful_episodes", "canonical_rows",
                "bc_valid_rows", "supervised_instances", "command_instances", "command_types",
                "morphology_rows", "bc_morphology_rows", "per_episode"):
        assert v6_audit[key] == frozen["expected"][key], key


def test_all_supervised_structural_arguments_are_resolvable(v6_audit, frozen):
    expected = frozen["expected"]
    assert v6_audit["structural_counts"] == expected["structural_counts"]
    assert v6_audit["structural_targets_checked"] == expected["structural_targets_checked"]
    assert v6_audit["unresolved_supervised_structural_targets"] == 0
    assert v6_audit["evidence_fallback_rows"] <= v6_audit["bc_valid_rows"]
    assert v6_audit["structural_resolved_without_evidence"] + v6_audit["structural_resolved_with_evidence"] == expected["structural_targets_checked"]


def test_source_fingerprints_unchanged_after_audit(v6_audit, frozen):
    assert v6_audit["immutability"]["before"] == v6_audit["immutability"]["after"]
    assert v6_audit["immutability"]["before"]["manifest_sha256"] == frozen["source"]["manifest_sha256"]
    assert len(v6_audit["immutability"]["before"]["files"]) == frozen["expected"]["episodes"]


def test_all_settle_targets_enabled_and_no_supervised_labels_discarded(v6_audit, frozen):
    assert v6_audit["structural_counts"]["gravity_settle"] == 24
    assert v6_audit["settle_targets_checked"] == 24
    assert v6_audit["supervised_instances"] == frozen["expected"]["supervised_instances"]
    assert not ({"reach", "align", "clocking", "approach"} & set(v6_audit["supervised_instances"]))
    for resource, modes in frozen["expected"]["generic_resource_instances"].items():
        for mode, count in modes.items():
            assert v6_audit["generic_resource_instances"].get(resource, {}).get(mode, 0) == count


def test_known_diagnostics_preserve_other_valid_resources(v6_audit, frozen):
    assert v6_audit["diagnostics"] == frozen["expected"]["diagnostics"]
    assert v6_audit["diagnostics"]["conflicting_resources_supervised"] == 0
    assert v6_audit["diagnostics"]["unresolved_sequences_on_bc_rows"] == 0


def test_all_three_new_assembly_episodes_are_included(v6_audit):
    by_id = {e["episode_id"]: e for e in v6_audit["per_episode"]}
    for seed, rows in ((6217, 2178), (6284, 2162), (6307, 2133)):
        e = by_id[f"expert:assembly:seed-{seed:06d}"]
        assert e["canonical_rows"] == rows and e["bc_valid_rows"] == 19
        assert e["supervised_instances"] == {"align_faces": 7, "dock": 7, "tilt_target": 8}


def row(primitive=None, *, wheels=False, internal=None, peer=True):
    action, supervision = f.smores_label(primitive, wheels=wheels, internal=internal, peer=peer)
    supervision.update(valid_for_behavior_cloning=any(any(mask.values()) for mask in supervision["module_resource_masks"].values()), conflicts=[])
    graph = f.smores_graph()
    graph["morphology"] = "transient"
    return {"schema_version": "mssr.canonical_transition.v5", "graph_t": graph,
            "action_t": action, "supervision": supervision, "provenance": {}}


def collection(tmp_path, rows):
    """Tiny Canonical-shaped source; manifest declarations built without audit."""
    root = tmp_path / "canonical"
    root.mkdir()
    payload = "".join(json.dumps(r) + "\n" for r in rows).encode()
    (root / "episode.jsonl").write_bytes(payload)
    commands, command_rows = Counter(), Counter()
    for r in rows:
        present = set()
        for action in r["action_t"]["modules"].values():
            for resource in ("wheels", "internal", "structural"):
                command = action.get(resource, {})
                if command.get("commanded") is True:
                    name = "wheel_velocity" if resource == "wheels" else command["mode" if resource == "internal" else "primitive"]
                    commands[name] += 1
                    present.add(name)
        command_rows.update(present)
    episode = {"episode_id": "synthetic", "file": "episode.jsonl", "kind": "expert", "task": "assembly",
        "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "canonical_transitions": len(rows),
        "bc_valid_transitions": sum(r["supervision"]["valid_for_behavior_cloning"] for r in rows),
        "command_instances": dict(commands), "command_types": dict(command_rows)}
    manifest = {"dataset_name": "synthetic", "schema_version": "mssr.canonical_dataset_collection.v5", "episodes": [episode]}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_synthetic_counts_distinguish_rows_commands_and_supervised_instances(tmp_path):
    audit = api()
    active = row(wheels=True, internal="pan_velocity")
    masked = row(internal="pan_velocity")
    masked["supervision"]["module_resource_masks"]["moving"]["internal"] = False
    masked["supervision"]["valid_for_behavior_cloning"] = False
    root = collection(tmp_path, [active, masked, row("align_faces"), row()])
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    result = audit(root, adapter())
    assert result["canonical_rows"] == 4 and result["bc_valid_rows"] == 2
    assert result["bc_useful_episodes"] == 1
    assert result["supervised_instances"] == {"wheel_velocity": 1, "pan_velocity": 1, "align_faces": 1}
    assert result["command_instances"]["pan_velocity"] == 2
    assert result["command_types"]["pan_velocity"] == 2
    assert result["generic_resource_instances"] == {"locomotion": {"wheel_velocity": 1}, "internal_motion": {"pan_velocity": 1}}
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before
    json.dumps(result, allow_nan=False)


def test_complete_canonical_label_needs_no_source_and_ignores_future(tmp_path):
    audit = api()
    r = row("dock")
    r["graph_t_plus_1"] = {"invalid": "must never be inspected"}
    r["provenance"]["source_records"] = [{"_loader_source_path": "missing.jsonl", "_loader_source_row": 1}]
    result = audit(collection(tmp_path, [r]), adapter())
    assert result["structural_targets_checked"] == 1
    assert result["evidence_fallback_rows"] == 0 and result["source_evidence_files"] == []


def test_exact_evidence_fallback_is_label_only_and_deduplicates(tmp_path):
    audit = api()
    r = row("align_faces", peer=False)
    source = tmp_path / "decision.jsonl"
    source.write_text(json.dumps(f.smores_source_goal()) + "\n" + json.dumps(f.smores_source_goal(goal_id="equivalent")) + "\n")
    r["provenance"]["source_records"] = [{"_loader_source_path": str(source), "_loader_source_row": i} for i in (1, 2)]
    result = audit(collection(tmp_path, [r]), adapter())
    assert result["structural_counts"] == {"align_faces": 1}
    assert result["evidence_fallback_rows"] == 1
    assert result["structural_resolved_with_evidence"] == 1
    assert len(result["source_evidence_files"]) == 1


def test_evidence_usage_counts_only_the_structural_target_needing_recovery(tmp_path):
    audit = api()
    r = row("align_faces")
    modules = r["action_t"]["modules"]
    modules["second"] = copy.deepcopy(modules["moving"])
    modules["second_peer"] = {"participates_in": []}
    masks = r["supervision"]["module_resource_masks"]
    masks["second"] = {"wheels": False, "internal": False, "structural": True}
    masks["second_peer"] = {"wheels": False, "internal": False, "structural": False}
    for name in ("second", "second_peer"):
        node = copy.deepcopy(r["graph_t"]["nodes"][0])
        node["module_id"] = name
        r["graph_t"]["nodes"].append(node)
    goal = f.smores_source_goal()
    goal["expert_action"]["primitive_goal"]["module_ids"] = ["second", "second_peer"]
    source = tmp_path / "second.jsonl"
    source.write_text(json.dumps(goal) + "\n")
    r["provenance"]["source_records"] = [{"_loader_source_path": str(source), "_loader_source_row": 1}]
    result = audit(collection(tmp_path, [r]), adapter())
    assert result["structural_targets_checked"] == 2
    assert result["evidence_fallback_rows"] == 1
    assert result["structural_resolved_with_evidence"] == 1
    assert result["structural_resolved_without_evidence"] == 1


@pytest.mark.parametrize("fault", ["missing", "ambiguous", "contradictory"])
def test_structural_evidence_failures_are_not_silently_dropped(tmp_path, fault):
    audit = api()
    r = row("align_faces", peer=False)
    if fault != "missing":
        r["action_t"]["modules"]["other"] = {"participates_in": []}
        other = copy.deepcopy(r["graph_t"]["nodes"][0])
        other["module_id"] = "other"
        r["graph_t"]["nodes"].append(other)
        goals = [f.smores_source_goal(), f.smores_source_goal(peer="other")]
        if fault == "contradictory": goals[-1]["expert_action"]["primitive_goal"]["parameters"]["clocking_quarter_turns"] = 2
        source = tmp_path / "decisions.jsonl"
        source.write_text("".join(json.dumps(g) + "\n" for g in goals))
        r["provenance"]["source_records"] = [{"_loader_source_path": str(source), "_loader_source_row": i} for i in (1, 2)]
    with pytest.raises(ValueError, match="synthetic.*transition 0"):
        audit(collection(tmp_path, [r]), adapter())


def test_disabled_settle_is_an_audit_failure_not_a_silent_count_change(tmp_path):
    audit = api()
    a = adapter()
    spec = replace(a.spec, actions=tuple(replace(x, enabled=False) if x.action_id == "gravity_settle" else x for x in a.spec.actions))
    with pytest.raises(ValueError, match="discard|supervised|excluded"):
        audit(collection(tmp_path, [row("gravity_settle")]), sm.SMORESAdapter(spec))


def test_settle_scope_is_checked_without_invented_internal_targets(tmp_path):
    result = api()(collection(tmp_path, [row("gravity_settle")]), adapter())
    assert result["supervised_instances"] == {"gravity_settle": 1}
    assert result["generic_resource_instances"] == {}
    assert result["settle_targets_checked"] == 1


@pytest.mark.parametrize("clear", [True, False])
def test_supervised_settle_flag_is_declared_and_preserved(clear):
    api()
    a = adapter()
    action_spec = next(x for x in a.spec.actions if x.action_id == "gravity_settle")
    parameters = {p.name: p for p in action_spec.parameters}
    assert "clear_passive_policy_on_finish" in parameters, "Missing real v6 SETTLE parameter in RobotSpec"
    flag = parameters["clear_passive_policy_on_finish"]
    assert flag.kind == "boolean" and flag.learnable is False
    r = row("gravity_settle")
    r["action_t"]["modules"]["moving"]["structural"]["parameters"][flag.name] = clear
    bundle = a.encode_labels(r["action_t"], r["supervision"], ())
    target, = bundle.structural_targets
    assert not bundle.resource_targets
    assert next(p.value for p in target.parameters if p.name == flag.name) is clear
    decoded = a.decode_action(c.ActionChoice(target.operation_id, target.initiator,
        participants=target.participants, connectors=target.connectors, parameters=target.parameters))
    assert decoded["parameters"][flag.name] is clear


def test_existing_contract_supports_settle_flag_with_explicit_catalog_entry():
    """Positive control: the discovered gap is config-only, not a core defect."""
    api()
    a = adapter()
    flag = c.ParameterSpec("clear_passive_policy_on_finish", "boolean", learnable=False)
    spec = replace(a.spec, actions=tuple(replace(x, parameters=x.parameters + (flag,))
        if x.action_id == "gravity_settle" and not any(p.name == flag.name for p in x.parameters) else x
        for x in a.spec.actions))
    r = row("gravity_settle")
    r["action_t"]["modules"]["moving"]["structural"]["parameters"][flag.name] = True
    target, = sm.SMORESAdapter(spec).encode_labels(r["action_t"], r["supervision"], ()).structural_targets
    assert next(p.value for p in target.parameters if p.name == flag.name) is True


@pytest.mark.parametrize("primitive,params", [
    ("align_faces", {"initial_straight_clearance_m": .06,
                     "initial_straight_clearance_reference_module_id": "clearance_anchor"}),
    ("dock", {"retain_mobile_structure_after_dock": True}),
])
def test_real_executor_options_are_declared_and_roundtrip_without_becoming_scope(primitive, params):
    api()
    a = adapter()
    catalog = {p.name: p for p in a.actions[primitive].parameters}
    assert set(params).issubset(catalog), "Missing real v6 executor options in RobotSpec"
    assert all(catalog[name].learnable is False for name in params)
    if primitive == "align_faces":
        assert catalog["initial_straight_clearance_m"].units == "m"
        assert catalog["initial_straight_clearance_reference_module_id"].kind == "module_ref"
    r = row(primitive)
    r["action_t"]["modules"]["clearance_anchor"] = {"participates_in": []}
    r["action_t"]["modules"]["moving"]["structural"]["parameters"].update(params)
    target, = a.encode_labels(r["action_t"], r["supervision"], ()).structural_targets
    assert target.participants == (c.ModuleRef("moving"), c.ModuleRef("peer"))
    if primitive == "align_faces":
        assert next(p.value for p in target.parameters if p.name.endswith("reference_module_id")) == c.ModuleRef("clearance_anchor")
    decoded = a.decode_action(c.ActionChoice(target.operation_id, target.initiator,
        participants=target.participants, connectors=target.connectors, parameters=target.parameters))
    for name, value in params.items(): assert decoded["parameters"][name] == value
    json.dumps(decoded, allow_nan=False)


def test_existing_module_ref_codec_is_supported_without_core_changes():
    api()
    a = adapter()
    name = "initial_straight_clearance_reference_module_id"
    declaration = c.ParameterSpec(name, "module_ref", learnable=False,
                                  module_type_id=a.module_type.type_id)
    spec = replace(a.spec, actions=tuple(replace(x, parameters=x.parameters + (declaration,))
        if x.action_id == "align_faces" and not any(p.name == name for p in x.parameters) else x for x in a.spec.actions))
    r = row("align_faces")
    r["action_t"]["modules"]["moving"]["structural"]["parameters"][name] = "peer"
    target, = sm.SMORESAdapter(spec).encode_labels(r["action_t"], r["supervision"], ()).structural_targets
    assert next(p.value for p in target.parameters if p.name == name) == c.ModuleRef("peer")


def test_conflict_preserves_unaffected_supervision(tmp_path):
    audit = api()
    r = row(wheels=True, internal="pan_target")
    r["supervision"]["module_resource_masks"]["moving"]["internal"] = False
    r["supervision"]["conflicts"] = [{"module_id": "moving", "resource": "internal", "reason": "conflicting decisions"}]
    result = audit(collection(tmp_path, [r]), adapter())
    assert result["supervised_instances"] == {"wheel_velocity": 1}
    assert result["diagnostics"]["conflicts_on_bc_rows"] == 1
    assert result["diagnostics"]["conflicting_resources_supervised"] == 0


@pytest.mark.parametrize("fault", ["inventory", "size", "rows", "unknown_mode", "false_bc", "orphan_mask"])
def test_source_and_supervision_drift_fail_explicitly(tmp_path, fault):
    audit = api()
    r = row(internal="pan_velocity")
    if fault == "unknown_mode": r["action_t"]["modules"]["moving"]["internal"]["mode"] = "REACH"
    if fault == "false_bc": r["supervision"]["valid_for_behavior_cloning"] = False
    if fault == "orphan_mask": r["supervision"]["module_resource_masks"]["absent"] = {"internal": True}
    root = collection(tmp_path, [r])
    if fault == "inventory": (root / "extra.jsonl").write_text("{}\n")
    if fault in {"size", "rows"}:
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["episodes"][0]["bytes" if fault == "size" else "canonical_transitions"] += 1
        (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError): audit(root, adapter())


def test_audit_rejects_source_mutation_during_adapter_call(tmp_path):
    audit = api()
    root = collection(tmp_path, [row(wheels=True)])
    class MutatingAdapter:
        def encode_labels(self, action, supervision, evidence):
            with (root / "episode.jsonl").open("ab") as stream: stream.write(b"\n")
            return adapter().encode_labels(action, supervision, evidence)
    with pytest.raises(ValueError, match="changed|mutat"):
        audit(root, MutatingAdapter())


def test_audit_rejects_manifest_stat_drift_even_with_identical_bytes(tmp_path):
    audit = api()
    root = collection(tmp_path, [row(wheels=True)])
    manifest = root / "manifest.json"
    class TouchingAdapter:
        def encode_labels(self, action, supervision, evidence):
            stat = manifest.stat()
            os.utime(manifest, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            return adapter().encode_labels(action, supervision, evidence)
    with pytest.raises(ValueError, match="changed|mutat"):
        audit(root, TouchingAdapter())


def test_audit_rejects_adapter_that_discards_or_invents_targets(tmp_path):
    audit = api()
    root = collection(tmp_path, [row(wheels=True)])
    class DroppingAdapter:
        def encode_labels(self, action, supervision, evidence): return c.SupervisionBundle()
    with pytest.raises(ValueError, match="supervised|discard|target"):
        audit(root, DroppingAdapter())
