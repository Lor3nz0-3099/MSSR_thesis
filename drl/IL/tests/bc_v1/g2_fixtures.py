"""Small generic G2 fixtures; no RAW/cache scan or simulator dependency."""
from dataclasses import replace
import copy
from pathlib import Path

from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.bc_v1 import normalization as f2
from drl.IL.bc_v1.candidates import enumerate_candidates
from drl.IL.bc_v1.dataset import CachedInputs, CachedSample
from drl.IL.bc_v1.feature_schema import extract_features, load_feature_schema, feature_schema_fingerprint
from drl.IL.bc_v1.features import PhysicalFrameConfig, transform_inputs
from drl.IL.tests.bc_v1.test_candidates import catalog, graph

ROOT = Path(__file__).resolve().parents[4]


def fixture(n=3, ports=2, *, geometry=True, attached=False):
    from drl.IL.tests.bc_v1.test_candidates import attach
    spec = catalog(ports or 1)
    state = graph(n, ports)
    if attached:
        state = attach(state, 0, ports)
    context = c.TaskContext('gap', (c.NumericFeature('gap.width_m', (2.,)),
        c.NumericFeature('gap.landing_length_m', (4.,)))) if geometry else c.TaskContext()
    inputs = c.PolicyInputs(state, c.ObservationBundle(), context)
    schema = load_feature_schema(ROOT/'configs/bc_v1/features.json')
    features = extract_features(inputs, transform_inputs(inputs, PhysicalFrameConfig()), spec, schema)
    table = enumerate_candidates(state, spec, schema)
    policy = copy.deepcopy(f2.load_normalization_policy(ROOT/'configs/bc_v1/normalization.json', schema))
    policy['resource_payloads'] = [
        {'module_type_id': 'kit', 'resource_id': 'rotor', 'mode_id': 'pace', 'components': [
            {'parameter': 'speeds', 'component': 0}, {'parameter': 'speeds', 'component': 1}]},
        {'module_type_id': 'kit', 'resource_id': 'rotor', 'mode_id': 'aim', 'components': [
            {'parameter': 'angle', 'component': 0}]}]
    _, rb, ab = f2._catalog_plan(spec, schema, policy)
    scales = [(f2.feature_key('module', 'relative_position', 0, 'm'), f2.Scaler(1., 2., 2, 'train_standard', False))]
    scales += [(key, f2.Scaler(1., 2., 2, 'train_standard', False)) for b in rb for key in b.keys]
    scales += [(key, f2.Scaler(1., 2., 2, 'train_standard', False)) for b in ab for key in b.keys]
    stats = f2.NormalizerStats(tuple(sorted({'feature_schema_fingerprint': feature_schema_fingerprint(schema),
        'policy_fingerprint': f2.policy_fingerprint(policy, schema),
        'robot_spec_fingerprint': c.robot_spec_fingerprint(spec)}.items())), tuple(scales), rb, ab)
    target = c.ResourceTarget(state.resources[0].ref, 'pace', (5., 7.)) if n else None
    supervision = c.SupervisionBundle((target,) if target else ())
    sample = CachedSample(CachedInputs(inputs, features, table), supervision,
        c.SampleAudit('train-fixture', 0, provenance='private', morphology_stratum='private'))
    return sample, dict(spec=spec, feature_schema=schema, normalization_policy=policy, normalizers=stats)


def with_structural(sample, *, skill=False):
    rows = [r for r in sample.inputs.candidates.structural_candidates
            if r.action.kind == ('skill' if skill else 'prepare_connection')]
    targets = []
    for mid in sample.inputs.policy.graph.modules[:2]:
        chosen = next((r, b) for r in rows for b in r.bindings if b.initiator == mid.ref)
        row, binding = chosen
        if skill:
            targets.append(c.StructuralTarget(mid.ref, row.action.action_id, (mid.ref,), (), (
                c.ParameterValue('duration', 5.), c.ParameterValue('members', (mid.ref,)),
                c.ParameterValue('clear', True))))
        else:
            targets.append(c.StructuralTarget(mid.ref, row.action.action_id,
                tuple(p.module for p in binding.connectors), binding.connectors,
                (c.ParameterValue('turn', 'quarter'),)))
    return replace(sample, supervision=replace(sample.supervision, structural_targets=tuple(targets)))


def tensorize(sample, kwargs):
    from drl.IL.bc_v1 import tensorization as t
    return t.tensorize_policy_sample(sample, tensor_layout=t.load_tensor_layout(
        ROOT/'configs/bc_v1/tensor_layout.v1.json'), **kwargs)
