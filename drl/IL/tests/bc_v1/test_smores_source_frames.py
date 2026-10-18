"""Current planar and certified legacy-map bridges; small source proofs only."""

import copy
from dataclasses import asdict, replace
import importlib
import importlib.util
import hashlib
import json
import itertools
import io
from pathlib import Path
import subprocess
import sys

import pytest

from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.bc_v1.features import transform_inputs
from drl.IL.bc_v1.policy_inputs import select_policy_inputs
from drl.IL.bc_v1.source_frame_bridges import (
    BridgeEvidence, FrameSourceContext, SourceContractCertificate, frame_config_for,
)
from drl.IL.tests.bc_v1.fixtures import require_canonical_v6_fixture

ROOT = Path(__file__).resolve().parents[4]
CONTRACT = "mssr.smores.current_task_frame.planar.v1"
ORIGIN = (1.7699752221047802, -2.17605937116128, 0.)
YAW = -1.5707963267948966
ORIGIN_NAME = "world.task_frame.origin_xyz_m"
YAW_NAME = "world.task_frame.yaw_rad"
EXPECTED_WORLD = (1.7699752221047802, -1.4760593711612797, 0.)


def api():
    name = "drl.IL.bc_v1.smores_source_frames"
    assert importlib.util.find_spec(name) is not None, "Missing Task 2 current SMORES stage_local resolver"
    return importlib.import_module(name)


def inputs():
    graph = c.StateGraph(modules=(
        c.ModuleState(c.ModuleRef("physical-unit"), "generic", (0., 0., 0.), (0., 0., 0., 1.), pose_valid=True),
    ))
    return select_policy_inputs(graph, c.ObservationBundle(), c.TaskContext("stairs", (
        c.NumericFeature(ORIGIN_NAME, ORIGIN), c.NumericFeature(YAW_NAME, (YAW,)),
        c.NumericFeature("stage_local.stairs.first_riser_x_m", (-.7000000000000001,)),
    )))


def source():
    # Synthetic verified receipt; no production evidence is fabricated or read.
    cert = SourceContractCertificate(CONTRACT, (BridgeEvidence("current-source.json", "a" * 64),))
    return FrameSourceContext("smores_ep", (("source", "immutable-current-source"), ("record", "current")), (cert,))


def poisoned_feature(selected, name, **changes):
    result = copy.deepcopy(selected)
    features = tuple(replace(f, **changes) if f.name == name else f for f in result.context.static_geometry)
    # D2 refuses these at construction; exercise resolver defense on a corrupted copy.
    object.__setattr__(result.context, "static_geometry", features)
    return result


def test_stage_local_1725_exact_current_frame():
    p = api()
    selected, receipt = inputs(), source()
    before = asdict(selected)
    resolved = p.resolve_stage_local(selected, receipt)
    assert resolved.certified and resolved.source_frame == "stage_local" and resolved.target_frame == "world"
    assert resolved.transform.translation == ORIGIN
    assert resolved.transform.rotation == pytest.approx((0., 1., 0., -1., 0., 0., 0., 0., 1.), abs=1e-15)
    assert resolved.contract_id == CONTRACT and resolved.evidence == receipt.certificates[0].evidence
    r, t = resolved.transform.rotation, resolved.transform.translation
    point = (-.7000000000000001, 0., 0.)
    measured_world = tuple(sum(r[3*i+j]*point[j] for j in range(3)) + t[i] for i in range(3))
    assert measured_world == pytest.approx(EXPECTED_WORLD, abs=1e-9)
    provider = p.SmoresSourceFrameProvider(ROOT, {})
    bridges = provider.resolve(selected, receipt)
    assert resolved in bridges.resolutions
    geometry = transform_inputs(selected, frame_config_for(bridges))
    plane = next(q for q in geometry.quantities if q.source_name == "stage_local.stairs.first_riser_x_m")
    assert plane.valid and plane.kind == "planes"
    assert plane.values == pytest.approx((0., -1., 0., 1.4760593711612797), abs=1e-9)
    assert asdict(selected) == before


@pytest.mark.parametrize("name,changes", [
    (ORIGIN_NAME, {"values": (1., 2.)}), (YAW_NAME, {"values": (0., 1.)}),
    (YAW_NAME, {"values": "not-a-vector"}), (YAW_NAME, {"values": (True,)}),
    (ORIGIN_NAME, {"values": (float("nan"), 0., 0.)}),
    (ORIGIN_NAME, {"values": (0., float("inf"), 0.)}),
    (YAW_NAME, {"values": (float("nan"),)}), (YAW_NAME, {"values": (float("inf"),)}),
    (YAW_NAME, {"values": (-float("inf"),)}),
    (ORIGIN_NAME, {"valid": False}), (YAW_NAME, {"valid": False}), (YAW_NAME, {"valid": 1}),
])
def test_stage_local_bad_current_frame_fails_closed(name, changes):
    result = api().resolve_stage_local(poisoned_feature(inputs(), name, **changes), source())
    assert not result.certified and result.transform is None and name in result.reason
    assert frame_config_for(api().SmoresSourceFrameProvider(ROOT, {}).resolve(
        poisoned_feature(inputs(), name, **changes), source())).transforms == ()


@pytest.mark.parametrize("missing", [ORIGIN_NAME, YAW_NAME])
def test_stage_local_missing_current_frame_fails_closed(missing):
    selected = inputs()
    selected = replace(selected, context=replace(selected.context, static_geometry=tuple(
        f for f in selected.context.static_geometry if f.name != missing)))
    result = api().resolve_stage_local(selected, source())
    assert not result.certified and result.transform is None and missing in result.reason


@pytest.mark.parametrize("name", [ORIGIN_NAME, YAW_NAME])
def test_stage_local_conflicting_current_evidence_rejected(name):
    selected = inputs()
    duplicate = c.NumericFeature(name, (9., 8., 7.) if name == ORIGIN_NAME else (1.,))
    object.__setattr__(selected.context, "static_geometry", selected.context.static_geometry + (duplicate,))
    result = api().resolve_stage_local(selected, source())
    assert not result.certified and result.transform is None
    assert "conflict" in result.reason.lower() and name in result.reason


def test_stage_local_contradictory_receipts_rejected_and_coherent_proofs_preserved():
    p = api()
    receipt = source()
    contradictory = SourceContractCertificate(CONTRACT, (BridgeEvidence("current-source.json", "b" * 64),))
    result = p.resolve_stage_local(inputs(), replace(receipt, certificates=receipt.certificates + (contradictory,)))
    assert not result.certified and result.transform is None and "conflict" in result.reason.lower()
    extra = SourceContractCertificate(CONTRACT, (BridgeEvidence("other-current-proof.json", "c" * 64),))
    receipts = receipt.certificates + (extra,)
    first = p.resolve_stage_local(inputs(), replace(receipt, certificates=receipts))
    second = p.resolve_stage_local(inputs(), replace(receipt, certificates=receipts[::-1]))
    assert first.certified and first == second and len(first.evidence) == 2


@pytest.mark.parametrize("change", ["missing_certificate", "wrong_contract", "wrong_backend", "missing_source_identity"])
def test_stage_local_requires_applicable_current_source_certificate(change):
    receipt = source()
    if change == "missing_certificate":
        receipt = replace(receipt, certificates=())
    elif change == "wrong_contract":
        receipt = replace(receipt, certificates=(replace(receipt.certificates[0], contract_id="other.v1"),))
    elif change == "wrong_backend":
        receipt = replace(receipt, backend_id="freebot")
    else:
        receipt = replace(receipt, source_identity=())
    result = api().resolve_stage_local(inputs(), receipt)
    assert not result.certified and result.transform is None and result.reason


def test_stage_local_ignores_stage_fsm_labels_and_future():
    p = api()
    forbidden = {"stage", "fsm", "control_state", "action_t", "label", "target_morphology", "target_graph", "graph_t_plus_1", "future", "supervision", "success"}

    class GuardedSource(FrameSourceContext):
        def __getattribute__(self, name):
            if name in forbidden:
                raise AssertionError("Forbidden metadata read: " + name)
            return super().__getattribute__(name)

    receipt = source()
    guarded = GuardedSource(receipt.backend_id, receipt.source_identity, receipt.certificates)
    provider = p.SmoresSourceFrameProvider(ROOT, {})
    before = provider.resolve(inputs(), receipt)
    assert provider.resolve(inputs(), guarded) == before
    for value in ("OPPOSITE", {"future": "unreachable", "action_t": "opposite"}):
        altered = copy.deepcopy(receipt)
        for name in forbidden:
            object.__setattr__(altered, name, value)
        assert provider.resolve(inputs(), altered) == before
    # Task ID is not a selector for the physical bridge either.
    selected = inputs()
    without_task = replace(selected, context=c.TaskContext(None, selected.context.static_geometry[:2]))
    assert p.resolve_stage_local(without_task, receipt) == p.resolve_stage_local(selected, receipt)


def test_provider_identity_deterministic_nonempty_and_detached_from_mutable_policy(tmp_path):
    p = api()
    policy = {"revision": "fixture.1", "configuration": {"b": 2, "a": 1}}
    provider = p.SmoresSourceFrameProvider(ROOT, policy)
    identity = provider.identity()
    reordered = {"configuration": {"a": 1, "b": 2}, "revision": "fixture.1"}
    assert identity and identity == p.SmoresSourceFrameProvider(tmp_path, reordered).identity()
    assert identity != p.SmoresSourceFrameProvider(ROOT, {**policy, "revision": "fixture.2"}).identity()
    policy["configuration"]["a"] = 99
    assert provider.identity() == identity
    identity["changed"] = True
    assert "changed" not in provider.identity()
    assert provider.resolve(inputs(), source()).provider_identity


def test_provider_does_not_resolve_map_or_transform_local_intrinsic_channels():
    p = api()
    selected = inputs()
    selected = replace(selected, context=c.TaskContext("button", (
        c.NumericFeature("map.button.center_xyz_m", (1., 2., 3.)),
        c.NumericFeature("body_local.button.center_xyz_m", (3., 2., 1.)),
        c.NumericFeature("button.plunger_stroke_m", (.01,)),
    )))
    bridges = p.SmoresSourceFrameProvider(ROOT, {}).resolve(selected, source())
    assert all(r.source_frame != "body_local" for r in bridges.resolutions)
    result = next(r for r in bridges.resolutions if r.source_frame == "map")
    assert not result.certified and result.transform is None and result.reason
    assert frame_config_for(bridges).transforms == ()
    world = next(r for r in bridges.resolutions if r.source_frame == "world")
    assert world.certified and world.transform is None


def test_provider_import_without_hardware_torch_readers_or_cache():
    api()
    script = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'rclpy', 'isaacsim', 'isaaclab', 'omni', 'smores_ep', 'mssr_expert'} or fullname in {'drl.IL.bc_v1.dataset', 'drl.IL.bc_v1.smores_adapter', 'drl.IL.bc_v1.task_context'}:
            raise ImportError('Provider isolation: ' + fullname)
sys.meta_path.insert(0, Block())
from drl.IL.bc_v1.smores_source_frames import SmoresSourceFrameProvider
from pathlib import Path
assert SmoresSourceFrameProvider(Path('.'), {}).identity()
'''
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


LEGACY_CONTRACT = "mssr.smores.isaac_nav2.legacy_rc_map.v1"


@pytest.fixture
def legacy_source(tmp_path):
    """Small source proofs shaped like the real manifests; no trajectory reads."""
    catalog = {}

    def write(path, document):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(document, sort_keys=True))
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        catalog[path] = digest
        return {"path": path, "sha256": digest}

    compact_root = "datasets/expert_v1_compact"
    directory = compact_root + "/episodes/rc_car/source-A"
    platform = {"semantic": "rc_car_planar_support_platform", "center_xyz_m": [1.1, 0., -.01], "size_xyz_m": [4.8, 3.2, .02]}
    env = write(directory + "/environment_geometry.json", {
        "layout_frame_id": "map", "layout": {"frame_id": "map", "platform_bounds_xy_m": [-1.3, 3.5, -1.6, 1.6]},
        "platform": platform,
    })
    phases = {}
    for phase in ("assembly", "behavior"):
        phases[phase] = {"schema_version": "mssr.expert_compact_phase.v1", "phase": phase,
            "compact": {"path": "episodes/rc_car/source-A/" + phase + ".jsonl", "records": 10, "sha256": "d" * 64},
            "observation_constants": {"course": {"frame_id": "world", "collision_boxes": [platform]}}}
        if phase == "assembly":
            phases[phase]["observation_constants"]["robot_family"] = "smores_ep"
        write(directory + "/" + phase + "_manifest.json", phases[phase])
    write(directory + "/manifest.json", {"schema_version": "mssr.expert_compact_episode.v1", "task": "rc_car",
        "environment_geometry": {"path": "environment_geometry.json", "sha256": env["sha256"]}, "phases": phases})
    compact = write(compact_root + "/manifest.json", {"schema_version": "mssr.expert_compact_dataset.v2",
        "episodes": [{"path": "episodes/rc_car/source-A", "task": "rc_car", "phases": ["assembly", "behavior"]}]})
    canonical = write("datasets/expert_v1_canonical_v6/manifest.json", {
        "dataset_name": "expert_v1_canonical_v6", "source_dataset": {"manifest": compact["path"], "manifest_sha256": compact["sha256"]},
        "episodes": [{"episode_id": "expert:rc_car:synthetic", "kind": "expert", "task": "rc_car", "environment_geometry": env}]})
    launch = write("historical_launch.txt", {"declaration": "fixture historical Isaac-world / map->odom identity proof"})
    policy = {"schema_version": "mssr.bc_v1.source_frames.v1", "missingness_policy": "unresolved",
        "legacy_rc_map": {"contract_id": LEGACY_CONTRACT, "revision": "1.0.0", "backend_id": "smores_ep", "source_kind": "expert",
            "canonical": {"manifest": canonical, "dataset_name": "expert_v1_canonical_v6"},
            "compact": {"manifest": compact, "schema_version": "mssr.expert_compact_dataset.v2"},
            "launch": {**launch, "odom_pose_frame": "isaac_world", "map_to_odom": "identity"},
            "evidence_catalog": {k: v for k, v in catalog.items() if k.startswith(directory)},
            "certification_policy": "pinned_source_membership_and_geometry_and_historical_launch"}}
    provenance = {"logical_episode_id": "expert:rc_car:synthetic",
        "dataset_identity": {"manifest": canonical["path"], "manifest_sha256": canonical["sha256"]},
        "source_records": [{"_loader_source_path": str(tmp_path / directory / "behavior.jsonl"), "_loader_source_row": 2, "_loader_source_kind": "expert"}]}
    return tmp_path, policy, provenance, {"backend_id": "smores_ep"}


def legacy_context(fixture):
    root, policy, provenance, metadata = fixture
    reader = getattr(api(), "read_frame_source_context", None)
    assert callable(reader), "Missing Task 3 verified legacy source reader"
    return reader(provenance, metadata, repo_root=root, policy=policy)


def test_certified_legacy_rc_map_identity(legacy_source):
    p = api()
    receipt = legacy_context(legacy_source)
    result = p.resolve_legacy_rc_map(receipt)
    assert result.source_frame == "map" and result.target_frame == "world" and result.certified
    assert result.transform.translation == (0., 0., 0.)
    assert result.transform.rotation == (1., 0., 0., 0., 1., 0., 0., 0., 1.)
    assert result.contract_id == LEGACY_CONTRACT and len(result.evidence) == 7
    root, policy, _, _ = legacy_source
    selected = replace(inputs(), context=c.TaskContext("button", (c.NumericFeature("map.button.center_xyz_m", (1., 2., 3.)),)))
    bridges = p.SmoresSourceFrameProvider(root, policy).resolve(selected, receipt)
    assert result in bridges.resolutions
    view = transform_inputs(selected, frame_config_for(bridges))
    assert next(q for q in view.quantities if q.source_name == "map.button.center_xyz_m").values == (1., 2., 3.)


@pytest.mark.parametrize("only_hint", ["map", "rc_car8", "episode", "nav2", "freebot", "missing_dataset_receipt"])
def test_arbitrary_map_and_other_backend_never_inherit_legacy_identity(legacy_source, only_hint):
    root, policy, provenance, metadata = legacy_source
    if only_hint == "freebot":
        metadata = {"backend_id": "freebot"}
    elif only_hint == "missing_dataset_receipt":
        provenance = {k: v for k, v in provenance.items() if k != "dataset_identity"}
    else:
        provenance = {"logical_episode_id": "expert:rc_car:synthetic"} if only_hint == "episode" else {}
        metadata = {"frame_id": "map"} if only_hint == "map" else {"morphology": "rc_car8"} if only_hint == "rc_car8" else {"navigation": {"controller": "nav2"}} if only_hint == "nav2" else {}
    receipt = api().read_frame_source_context(provenance, metadata, repo_root=root, policy=policy)
    result = api().resolve_legacy_rc_map(receipt)
    assert not result.certified and result.transform is None and result.reason
    # Even a valid SMORES certificate does not transfer to another backend.
    if only_hint == "freebot":
        verified = legacy_context(legacy_source)
        assert not api().resolve_legacy_rc_map(replace(verified, backend_id="freebot")).certified


@pytest.mark.parametrize("proof", ["canonical", "compact", "launch", "episode", "phase", "assembly", "environment"])
@pytest.mark.parametrize("failure", ["missing", "changed_digest"])
def test_legacy_identity_requires_all_pinned_source_evidence(legacy_source, proof, failure):
    root, policy, provenance, metadata = legacy_source
    contract = policy["legacy_rc_map"]
    paths = {"canonical": contract["canonical"]["manifest"]["path"], "compact": contract["compact"]["manifest"]["path"], "launch": contract["launch"]["path"],
        "episode": "datasets/expert_v1_compact/episodes/rc_car/source-A/manifest.json",
        "phase": "datasets/expert_v1_compact/episodes/rc_car/source-A/behavior_manifest.json",
        "assembly": "datasets/expert_v1_compact/episodes/rc_car/source-A/assembly_manifest.json",
        "environment": "datasets/expert_v1_compact/episodes/rc_car/source-A/environment_geometry.json"}
    target = root / paths[proof]
    target.unlink() if failure == "missing" else target.write_text(target.read_text() + " ")
    with pytest.raises(ValueError, match="evidence|digest|missing|hash"):
        api().read_frame_source_context(provenance, metadata, repo_root=root, policy=policy)


@pytest.mark.parametrize("failure", ["new_dataset_identity", "copied_compact_source", "wrong_episode", "wrong_stream", "outside_row", "wrong_source_kind"])
def test_legacy_reader_rejects_unapproved_membership(legacy_source, failure):
    root, policy, provenance, metadata = legacy_source
    provenance = copy.deepcopy(provenance)
    if failure == "new_dataset_identity":
        provenance["dataset_identity"]["manifest"] = "datasets/new_collection/manifest.json"
    elif failure == "copied_compact_source":
        provenance["source_records"][0]["_loader_source_path"] = str(root / "datasets/new_compact/episodes/rc_car/source-A/behavior.jsonl")
    elif failure == "wrong_episode":
        provenance["logical_episode_id"] = "expert:rc_car:other"
    elif failure == "wrong_stream":
        provenance["source_records"][0]["_loader_source_path"] = str(root / "datasets/expert_v1_compact/episodes/rc_car/source-A/unapproved.jsonl")
    elif failure == "outside_row":
        provenance["source_records"][0]["_loader_source_row"] = 999
    else:
        provenance["source_records"][0]["_loader_source_kind"] = "teleop"
    try:
        result = api().resolve_legacy_rc_map(api().read_frame_source_context(provenance, metadata, repo_root=root, policy=policy))
    except ValueError:
        return  # Explicit proof failure is also fail-closed.
    assert not result.certified and result.transform is None


def test_teleop_world_navigation_never_requests_legacy_map(monkeypatch):
    p = api()
    def forbidden(*args, **kwargs):
        raise AssertionError("WORLD teleop requested legacy certification")
    monkeypatch.setattr(p, "resolve_legacy_rc_map", forbidden)
    selected = replace(inputs(), context=c.TaskContext("navigation", (
        c.NumericFeature("world.cones.0.center_xy_m", (1., 2.)),)))
    bridges = p.SmoresSourceFrameProvider(ROOT, {}).resolve(selected, FrameSourceContext("smores_ep", (("source_kind", "teleop"),), ()))
    assert [r.source_frame for r in bridges.resolutions] == ["world"]


def test_legacy_reader_no_large_file_reads_and_no_forbidden_metadata(legacy_source, monkeypatch):
    root, policy, provenance, metadata = legacy_source
    forbidden = {"action_t", "label", "target_morphology", "target_graph", "graph_t_plus_1", "stage", "fsm", "control_state", "reward", "success", "supervision", "feasibility", "canonical_transition_index"}
    class Guard(dict):
        def get(self, key, default=None):
            assert key not in forbidden, key
            return super().get(key, default)
        def __getitem__(self, key):
            assert key not in forbidden, key
            return super().__getitem__(key)
    original_open = Path.open
    def bounded_open(path, *args, **kwargs):
        assert path.suffix != ".jsonl", "Trajectory opened by source reader"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", bounded_open)
    before = legacy_context(legacy_source)
    changed = Guard({**provenance, **{key: "OPPOSITE" for key in forbidden}})
    altered_metadata = Guard({**metadata, **{key: "OPPOSITE" for key in forbidden}})
    after = api().read_frame_source_context(changed, altered_metadata, repo_root=root, policy=policy)
    assert after == before


def test_legacy_and_current_planar_contracts_are_independent(legacy_source):
    p = api()
    legacy = legacy_context(legacy_source)
    assert not p.resolve_stage_local(inputs(), legacy).certified
    assert not p.resolve_legacy_rc_map(source()).certified
    combined = replace(legacy, certificates=legacy.certificates + source().certificates)
    selected = inputs()
    selected = replace(selected, context=replace(selected.context, static_geometry=selected.context.static_geometry + (
        c.NumericFeature("map.environment.floor_height_m", (0.,)),)))
    root, policy, _, _ = legacy_source
    results = p.SmoresSourceFrameProvider(root, policy).resolve(selected, combined).resolutions
    assert {r.source_frame for r in results if r.certified} == {"world", "stage_local", "map"}


def test_real_legacy_005100_small_evidence_chain():
    require_canonical_v6_fixture(ROOT)
    p = api()
    path = ROOT / "configs/bc_v1/source_frames.json"
    assert path.is_file(), "Missing pinned historical source-frame policy"
    policy = json.loads(path.read_text())
    provenance = {"logical_episode_id": "expert:rc_car:seed-005100",
        "dataset_identity": {"manifest": "datasets/expert_v1_canonical_v6/manifest.json", "manifest_sha256": "31765272b6ecd7506c54b25e214b75d7b1b28691803ddfc286dafcce8b4ec562"},
        "source_records": [{"_loader_source_path": str(ROOT / "datasets/expert_v1_compact/episodes/rc_car/seed-005100/behavior.jsonl"), "_loader_source_row": 702, "_loader_source_kind": "expert"}]}
    receipt = p.read_frame_source_context(provenance, {}, repo_root=ROOT, policy=policy)
    assert p.resolve_legacy_rc_map(receipt).certified
    evidence = {e.path: e.sha256 for e in receipt.certificates[0].evidence}
    assert evidence["datasets/expert_v1_compact/episodes/rc_car/seed-005100/environment_geometry.json"] == "360be45bab514717098e04ee627e4b017353b9ca4c0179f6f8199ec4f947e562"


def test_legacy_provider_requires_matching_policy_identity(legacy_source):
    root, policy, _, _ = legacy_source
    receipt = legacy_context(legacy_source)
    selected = replace(inputs(), context=c.TaskContext("button", (c.NumericFeature("map.button.center_xyz_m", (1., 2., 3.)),)))
    altered = {**policy, "missingness_policy": "different-policy-revision"}
    provider = api().SmoresSourceFrameProvider(root, altered)
    result = next(r for r in provider.resolve(selected, receipt).resolutions if r.source_frame == "map")
    assert not result.certified and result.transform is None and "policy" in result.reason


@pytest.mark.parametrize("declaration", ["map_frame", "world_frame", "platform", "source_link", "launch_convention"])
def test_pinned_files_alone_do_not_override_contradictory_declarations(legacy_source, declaration):
    root, policy, provenance, metadata = legacy_source
    contract = policy["legacy_rc_map"]
    if declaration == "launch_convention":
        contract["launch"]["map_to_odom"] = "nonidentity"
    else:
        name = "environment_geometry.json" if declaration == "map_frame" else "behavior_manifest.json" if declaration in {"world_frame", "platform"} else None
        target = root / ("datasets/expert_v1_compact/episodes/rc_car/source-A/" + name) if name else root / contract["canonical"]["manifest"]["path"]
        document = json.loads(target.read_text())
        if declaration == "map_frame":
            document["layout"]["frame_id"] = "unrelated_map"
        elif declaration == "world_frame":
            document["observation_constants"]["course"]["frame_id"] = "map"
        elif declaration == "platform":
            document["observation_constants"]["course"]["collision_boxes"][0]["center_xyz_m"][0] = 99.
        else:
            document["source_dataset"]["manifest_sha256"] = "e" * 64
        target.write_text(json.dumps(document, sort_keys=True))
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        relative = str(target.relative_to(root))
        if name:
            contract["evidence_catalog"][relative] = digest
        else:
            contract["canonical"]["manifest"]["sha256"] = digest
            provenance["dataset_identity"]["manifest_sha256"] = digest
    with pytest.raises(ValueError):
        api().read_frame_source_context(provenance, metadata, repo_root=root, policy=policy)


def real_current_stage_chain():
    require_canonical_v6_fixture(ROOT)
    from drl.IL.bc_v1.smores_adapter import SMORESAdapter
    from drl.IL.bc_v1.task_context import read_task_metadata, build_task_view
    source_root = ROOT / "datasets/expert_v1_canonical_v6"
    manifest = json.loads((source_root / "manifest.json").read_text())
    entry = next(e for e in manifest["episodes"] if e["episode_id"] == "teleop-0e4e34fe3a5f4ed1b6a86642acf4fac8")
    with (source_root / entry["file"]).open() as stream:
        row = json.loads(next(itertools.islice(stream, 1725, 1726)))
    metadata = read_task_metadata(row, ROOT)
    task_config = json.loads((ROOT / "configs/bc_v1/task_context.json").read_text())
    context, observation = build_task_view(row, metadata, task_config)
    adapter = SMORESAdapter(c.load_robot_spec(ROOT / "configs/bc_v1/smores_ep.robot.json"))
    selected = select_policy_inputs(adapter.encode_state(row["graph_t"]), observation, context)
    # Caller composition supplies already-selected physical data, never task IDs.
    metadata["current_task_geometry"] = tuple(asdict(f) for f in selected.context.static_geometry
        if f.name in (ORIGIN_NAME, YAW_NAME) or f.name.startswith("stage_local."))
    provenance = {**row["provenance"], "dataset_identity": {
        "manifest": "datasets/expert_v1_canonical_v6/manifest.json",
        "manifest_sha256": hashlib.sha256((source_root / "manifest.json").read_bytes()).hexdigest(),
    }}
    policy = json.loads((ROOT / "configs/bc_v1/source_frames.json").read_text())
    return provenance, metadata, policy, selected


def test_real_stage_local_1725_reader_to_resolver_certified():
    provenance, metadata, policy, selected = real_current_stage_chain()
    receipt = api().read_frame_source_context(provenance, metadata, repo_root=ROOT, policy=policy)
    assert CONTRACT in {cert.contract_id for cert in receipt.certificates}, "Missing production current stage_local certificate"
    assert receipt.backend_id == "smores_ep"
    result = api().resolve_stage_local(selected, receipt)
    assert result.certified and result.transform.translation == ORIGIN
    point = (-.7000000000000001, 0., 0.)
    r, t = result.transform.rotation, result.transform.translation
    assert tuple(sum(r[3*i+j]*point[j] for j in range(3))+t[i] for i in range(3)) == pytest.approx(EXPECTED_WORLD, abs=1e-9)
    assert result.evidence
    assert all("legacy_rc" not in cert.contract_id for cert in receipt.certificates)
    stage_only_policy = {k: v for k, v in policy.items() if k != "legacy_rc_map"}
    independent = api().read_frame_source_context(provenance, metadata, repo_root=ROOT, policy=stage_only_policy)
    assert api().resolve_stage_local(selected, independent).certified


def test_arbitrary_stage_local_fields_do_not_certify_without_current_source_proof():
    provenance, metadata, policy, _ = real_current_stage_chain()
    arbitrary = {"current_task_geometry": metadata["current_task_geometry"], "backend_id": "smores_ep", "sources": metadata["sources"]}
    receipt = api().read_frame_source_context({}, arbitrary, repo_root=ROOT, policy=policy)
    assert CONTRACT not in {cert.contract_id for cert in receipt.certificates}
    receipt = api().read_frame_source_context({**provenance, "source_records": []}, metadata, repo_root=ROOT, policy=policy)
    assert not receipt.certificates


def test_current_stage_reader_does_not_consume_inapplicable_legacy_rc_proofs():
    provenance, metadata, policy, selected = real_current_stage_chain()
    policy["legacy_rc_map"]["compact"]["manifest"]["sha256"] = "f" * 64
    receipt = api().read_frame_source_context(provenance, metadata, repo_root=ROOT, policy=policy)
    assert api().resolve_stage_local(selected, receipt).certified


@pytest.mark.parametrize("bad", ["missing_current_frame", "invalid_current_frame", "other_frame", "wrong_current_stamp", "unreferenced_source"])
def test_current_stage_reader_fails_closed_on_noncurrent_or_invalid_evidence(bad):
    provenance, metadata, policy, _ = real_current_stage_chain()
    if bad == "missing_current_frame":
        metadata.pop("current_task_geometry")
    elif bad == "invalid_current_frame":
        metadata["current_task_geometry"][0]["valid"] = False
    elif bad == "other_frame":
        origin = next(f for f in metadata["current_task_geometry"] if f["name"] == ORIGIN_NAME)
        origin["values"] = (100., 200., 300.)
    elif bad == "wrong_current_stamp":
        provenance["physical_stamp"] += 1.
    else:
        provenance["source_records"][0]["_loader_source_row"] = 999999
    try:
        receipt = api().read_frame_source_context(provenance, metadata, repo_root=ROOT, policy=policy)
    except ValueError:
        return
    assert CONTRACT not in {cert.contract_id for cert in receipt.certificates}


def test_current_stage_reader_ignores_forbidden_metadata():
    provenance, metadata, policy, selected = real_current_stage_chain()
    forbidden = {"stage", "fsm", "control_state", "action_t", "target_graph", "target_morphology", "graph_t_plus_1", "success", "reward", "supervision", "canonical_transition_index"}
    class Guard(dict):
        def get(self, key, default=None):
            assert key not in forbidden, key
            return super().get(key, default)
        def __getitem__(self, key):
            assert key not in forbidden, key
            return super().__getitem__(key)
    before = api().read_frame_source_context(provenance, metadata, repo_root=ROOT, policy=policy)
    assert CONTRACT in {cert.contract_id for cert in before.certificates}
    guarded_p = Guard({**provenance, **{key: "OPPOSITE" for key in forbidden}})
    guarded_m = Guard({**metadata, **{key: "OPPOSITE" for key in forbidden}})
    after = api().read_frame_source_context(guarded_p, guarded_m, repo_root=ROOT, policy=policy)
    assert after == before and api().resolve_stage_local(selected, after).certified


def test_current_stage_row_receipt_ignores_actions_and_future_payloads(monkeypatch):
    provenance, metadata, policy, _ = real_current_stage_chain()
    before = api().read_frame_source_context(provenance, metadata, repo_root=ROOT, policy=policy)
    path = ROOT / provenance["source_records"][0]["_loader_source_path"]
    with path.open("rb") as stream:
        rows = [json.loads(line) for line in itertools.islice(stream, 3)]
    for row in rows:
        for key in ("expert_action", "stage_name", "fsm_state", "target_graph", "graph_t_plus_1", "success", "reward", "supervision"):
            row[key] = {"forbidden": "OPPOSITE"}
    payload = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    original = Path.open
    def modified_current_rows(target, mode="r", *args, **kwargs):
        if target == path and mode == "rb":
            return io.BytesIO(payload)
        return original(target, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", modified_current_rows)
    after = api().read_frame_source_context(provenance, metadata, repo_root=ROOT, policy=policy)
    assert after == before
