"""Generic certified bridge contracts; geometry is still owned by E1."""

import copy
from dataclasses import FrozenInstanceError, asdict, fields, replace
import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.bc_v1.features import PhysicalFrameConfig, SourceFrameTransform, transform_inputs
from drl.IL.bc_v1.policy_inputs import select_policy_inputs

ROOT = Path(__file__).resolve().parents[4]


def api():
    name = "drl.IL.bc_v1.source_frame_bridges"
    assert importlib.util.find_spec(name) is not None, "Missing Task 1 source-frame bridge contract"
    return importlib.import_module(name)


def inputs(frame="world"):
    graph = c.StateGraph(modules=(
        c.ModuleState(c.ModuleRef("a"), "generic", (10., 2., 3.), (0., 0., 0., 1.), pose_valid=True),
        c.ModuleState(c.ModuleRef("b"), "generic", (14., 4., 5.), (0., 0., 0., 1.), pose_valid=True),
    ))
    return select_policy_inputs(graph, c.ObservationBundle((
        c.NumericFeature("button.depression_m", (.1,)),
    )), c.TaskContext("button", (
        c.NumericFeature(frame + ".button.center_xyz_m", (20., 6., 8.)),
    )))


def resolution(frame="world", *, transform=None, certified=True, scope="current-source"):
    b = api()
    return b.BridgeResolution(
        source_frame=frame, target_frame="world", scope_id=scope,
        transform=transform, certified=certified,
        reason="declared current world" if frame == "world" else "current source certificate",
        contract_id=None if frame == "world" else "source-contract.v1",
        evidence=() if frame == "world" else (b.BridgeEvidence("source.json", "a" * 64),),
    )


def batch(*resolutions):
    return api().SourceFrameBridgeBatch(tuple(resolutions), (("provider", "generic"), ("revision", "1")))


def test_world_identity_reuses_e1_without_redundant_transform():
    b = api()
    result = resolution()
    assert result.certified and result.transform is None and result.reason
    config = b.frame_config_for(batch(result))
    assert type(config) is PhysicalFrameConfig
    assert config.graph_frame == config.axes_id == "world" and config.transforms == ()
    selected = inputs()
    view = transform_inputs(selected, config)
    assert view.frame.origin == (12., 3., 4.)
    assert [m.position for m in view.modules] == [(-2., -1., -1.), (2., 1., 1.)]
    point = next(q for q in view.quantities if q.source_scope == "c_static")
    assert point.valid and point.values == (8., 3., 4.)
    assert view == transform_inputs(selected, PhysicalFrameConfig())


@pytest.mark.parametrize("invalid_case", [
    "missing_contract", "missing_evidence", "missing_contract_and_evidence",
    "missing_provider", "empty_certificate",
])
def test_certified_named_bridge_requires_contract_evidence_and_provider_identity(invalid_case):
    b = api()
    world = resolution()
    assert world.certified and world.contract_id is None and world.evidence == () and world.transform is None
    assert b.frame_config_for(batch(world)).transforms == ()
    assert b.frame_config_for(b.SourceFrameBridgeBatch((), ())).transforms == ()

    named = resolution("map", transform=SourceFrameTransform("map", (1., 2., 3.)))
    # Identity key names are provider-owned, not a generic core schema.
    attributed = b.SourceFrameBridgeBatch((named,), (("implementation", "generic.v1"),))
    assert b.frame_config_for(attributed).transforms == (named.transform,)

    failed_attempt = replace(named, certified=False, transform=None, reason="evidence failed verification")
    assert failed_attempt.contract_id == "source-contract.v1" and failed_attempt.evidence == named.evidence
    assert b.frame_config_for(batch(failed_attempt)).transforms == ()
    unattributed_attempt = replace(failed_attempt, contract_id=None, evidence=())
    assert not unattributed_attempt.certified and b.frame_config_for(batch(unattributed_attempt)).transforms == ()

    if invalid_case == "missing_contract":
        with pytest.raises(ValueError, match="contract_id"):
            replace(named, contract_id=None)
    elif invalid_case == "missing_evidence":
        with pytest.raises(ValueError, match="evidence"):
            replace(named, evidence=())
    elif invalid_case == "missing_contract_and_evidence":
        with pytest.raises(ValueError, match="contract_id"):
            replace(named, contract_id=None, evidence=())
    elif invalid_case == "missing_provider":
        for result in (named, world, failed_attempt):
            with pytest.raises(ValueError, match="provider_identity"):
                b.SourceFrameBridgeBatch((result,), ())
    else:
        with pytest.raises(ValueError, match="evidence"):
            b.SourceContractCertificate("source-contract.v1", ())


@pytest.mark.parametrize("frame", ["unknown_global", "map"])
def test_unknown_frame_unresolved_with_reason(frame):
    b = api()
    result = replace(resolution(frame, certified=False), reason="source contract not certified")
    config = b.frame_config_for(batch(result))
    assert not result.certified and result.transform is None and result.reason
    assert config.transforms == ()
    if frame == "map":
        view = transform_inputs(inputs("map"), config)
        point = next(q for q in view.quantities if q.source_scope == "c_static")
        assert not point.valid and point.values is None
    with pytest.raises(ValueError, match="reason"):
        replace(result, reason="")
    if frame == "map":
        with pytest.raises(ValueError, match="uncertified"):
            replace(result, transform=SourceFrameTransform("map", (0., 0., 0.)))


@pytest.mark.parametrize("change", ["translation", "rotation", "contract"])
def test_conflicting_frame_scope_certificates_rejected(change):
    b = api()
    first = resolution("map", transform=SourceFrameTransform("map", (1., 2., 3.)))
    if change == "translation":
        second = replace(first, transform=SourceFrameTransform("map", (4., 5., 6.)))
    elif change == "rotation":
        second = replace(first, transform=SourceFrameTransform(
            "map", (1., 2., 3.), (0., -1., 0., 1., 0., 0., 0., 0., 1.)))
    else:
        second = replace(first, contract_id="different-contract.v1")
    for ordered in ((first, second), (second, first)):
        with pytest.raises(ValueError, match="[Cc]onflict"):
            b.frame_config_for(batch(*ordered))


def test_coherent_duplicates_compile_once_and_scope_disagreement_fails_closed():
    b = api()
    first = resolution("map", transform=SourceFrameTransform("map", (1., 2., 3.)))
    same = replace(first, evidence=(b.BridgeEvidence("other-proof.json", "b" * 64),))
    config = b.frame_config_for(batch(first, same))
    assert config.transforms == (first.transform,)
    # E1 indexes transforms by frame, so differing scopes cannot choose a winner either.
    different = replace(first, scope_id="other-source", transform=SourceFrameTransform("map", (4., 5., 6.)))
    with pytest.raises(ValueError, match="[Cc]onflict"):
        b.frame_config_for(batch(first, different))
    with pytest.raises(ValueError, match="[Cc]onflict"):
        b.frame_config_for(batch(first, replace(first, certified=False, transform=None, reason="not certified")))


def test_bridge_contract_frozen_and_not_policy_input():
    b = api()
    evidence = b.BridgeEvidence("source.json", "a" * 64)
    certificate = b.SourceContractCertificate("contract.v1", (evidence,))
    source = b.FrameSourceContext("generic-backend", (("dataset", "immutable.v1"),), (certificate,))
    result = resolution()
    bridges = batch(result)
    diagnostic = b.PipelineDiagnostic("bridge", "c_static", "map.button.center_xyz_m", "FRAME_SEMANTICS_UNKNOWN", "no proof")
    records = (evidence, certificate, source, result, bridges, diagnostic)
    for record in records:
        name = fields(record)[0].name
        with pytest.raises(FrozenInstanceError):
            setattr(record, name, "changed")
        hash(record)  # No mutable nested containers inside these value records.
    with pytest.raises(TypeError):
        replace(certificate, evidence=[evidence])
    with pytest.raises(TypeError):
        replace(source, source_identity={"dataset": "immutable.v1"})
    with pytest.raises(TypeError):
        replace(source, certificates=[certificate])
    with pytest.raises(TypeError):
        replace(bridges, provider_identity={"provider": "generic"})
    with pytest.raises(TypeError):
        replace(bridges, resolutions=[result])
    selected = inputs()
    before = asdict(selected)
    assert [f.name for f in fields(c.PolicyInputs)] == ["graph", "observation", "context"]
    alternate_audit = replace(bridges, provider_identity=(("provider", "other-proof-provider"),))
    assert transform_inputs(selected, b.frame_config_for(bridges)) == transform_inputs(selected, b.frame_config_for(alternate_audit))
    assert asdict(selected) == before
    with pytest.raises(TypeError):
        c.PolicyInputs(selected.graph, selected.observation, selected.context, bridges=bridges)
    poisoned = copy.deepcopy(selected)
    object.__setattr__(poisoned, "bridges", bridges)
    with pytest.raises(ValueError, match="Unselected"):
        transform_inputs(poisoned, b.frame_config_for(bridges))


def test_bridge_policy_fingerprint_deterministic_sensitive():
    b = api()
    policy = {"provider": {"id": "generic", "revision": "1"}, "configuration": {"translation": [1., 2., 3.]},
              "source_contract": "contract.v1", "evidence_policy": "pinned", "source": "immutable.v1", "missingness": "closed"}
    reordered = {key: value for key, value in reversed(tuple(policy.items()))}
    reordered["provider"] = {"revision": "1", "id": "generic"}
    digest = b.bridge_policy_fingerprint(policy)
    assert len(digest) == 64 and digest == b.bridge_policy_fingerprint(reordered)
    for key in ("source_contract", "evidence_policy", "source", "missingness"):
        assert digest != b.bridge_policy_fingerprint({**policy, key: "changed"})
    assert digest != b.bridge_policy_fingerprint({**policy, "provider": {"id": "generic", "revision": "2"}})
    # Transform component order is significant, unlike mapping insertion order.
    assert digest != b.bridge_policy_fingerprint({**policy, "configuration": {"translation": [3., 2., 1.]}})
    for bad in ({"x": float("nan")}, {"x": float("inf")}, {1: "coerced key"}):
        with pytest.raises((ValueError, TypeError)):
            b.bridge_policy_fingerprint(bad)


@pytest.mark.parametrize("frame", ["module_body", "body_local", "endpoint_a_body", "connection_intrinsic", "resource_local", "intrinsic"])
def test_local_intrinsic_semantics_are_not_global_bridge_candidates(frame):
    with pytest.raises(ValueError, match="local|intrinsic"):
        resolution(frame)


@pytest.mark.parametrize("changes", [
    {"source_frame": "stage_local"}, {"target_frame": "map"}, {"certified": 1}, {"scope_id": ""},
])
def test_malformed_bridge_cannot_reach_e1(changes):
    with pytest.raises((ValueError, TypeError)):
        api().frame_config_for(batch(replace(resolution(), **changes)))


def test_named_certified_bridge_reuses_e1_validation_and_math():
    b = api()
    transform = SourceFrameTransform("map", (1., 2., 3.))
    config = b.frame_config_for(batch(resolution("map", transform=transform)))
    point = next(q for q in transform_inputs(inputs("map"), config).quantities if q.source_scope == "c_static")
    assert point.valid and point.values == (9., 5., 7.)
    with pytest.raises(ValueError, match="frame"):
        replace(resolution("map", transform=transform), source_frame="stage_local")
    # Even a frozen E1 value poisoned through low-level mutation must be revalidated.
    poisoned = copy.deepcopy(transform)
    object.__setattr__(poisoned, "rotation", (0.,) * 9)
    with pytest.raises(ValueError):
        b.frame_config_for(batch(resolution("map", transform=poisoned)))


def test_generic_bridge_import_has_no_hardware_torch_or_pipeline_dependencies():
    api()
    script = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'rclpy', 'rospy', 'isaacsim', 'isaaclab', 'omni', 'smores_ep'} or fullname in {'drl.IL.bc_v1.smores_adapter', 'drl.IL.bc_v1.task_context', 'drl.IL.bc_v1.dataset', 'drl.IL.bc_v1.normalization', 'drl.IL.bc_v1.feature_schema'}:
            raise ImportError('Bridge isolation: ' + fullname)
sys.meta_path.insert(0, Block())
from drl.IL.bc_v1.source_frame_bridges import SourceFrameProvider, SourceFrameBridgeBatch, frame_config_for
assert frame_config_for(SourceFrameBridgeBatch((), ())).transforms == ()
assert {'resolve', 'identity'} <= set(vars(SourceFrameProvider))
'''
    result = subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
