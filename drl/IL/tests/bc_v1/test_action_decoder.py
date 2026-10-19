"""L1a: actual G2/H2 ABI, synthetic states only; never opens dataset samples."""
import copy
import importlib
import importlib.util
from dataclasses import replace

import pytest
import torch

from drl.IL.bc_v1 import bc_v1_contract as c, normalization as f2
from drl.IL.bc_v1.batching import collate_policy_samples
from drl.IL.tests.bc_v1.g2_fixtures import fixture, tensorize
from drl.IL.tests.bc_v1.test_action_heads import complete_model, enriched_fixture


def api():
    name = 'drl.IL.bc_v1.action_decoder'
    assert importlib.util.find_spec(name), 'Missing L1a pure action decoder'
    return importlib.import_module(name)


@pytest.fixture(autouse=True, scope='module')
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def example(n=3, *, enriched=False, batch=False):
    a = api()
    sample, kw = enriched_fixture() if enriched else fixture(n)
    if enriched:
        _, rb, ab = f2._catalog_plan(kw['spec'], kw['feature_schema'], kw['normalization_policy'])
        keys = {key for binding in (*rb, *ab) for key in binding.keys}
        kw['normalizers'] = replace(kw['normalizers'], resource_bindings=rb, argument_bindings=ab,
            scalers=tuple((key, f2.Scaler(1., 2., 2, 'train_standard', False)) for key in sorted(keys)))
    one = tensorize(sample, kw)
    b = collate_policy_samples([one, one]) if batch else one
    with torch.no_grad():
        pred = complete_model(kw)(b['inputs'])
    # Known, hand-computable predictions; full production H2 routing retained.
    pred['resource']['mode_logits'].fill_(0.)
    pred['resource']['payload'].fill_(0.)
    pred['structural']['candidate_logits'].fill_(0.)
    for family, row in pred['structural']['arguments'].items():
        if 'value' in row: row['value'].fill_(0.)
        if 'logits' in row: row['logits'].fill_(0.)
    pred['structural']['scope']['logits'].fill_(-1.)
    routes = b['routing'].get('samples', (b['routing'],))
    availability = tuple(a.DispatchAvailability(
        states=tuple((ref, a.ResourceStatus.FREE) for ref in (*r['resource_refs'], *r['connector_refs'])),
        resource_mask=(True,) * len(r['resource_candidates']),
        structural_mask=(False,) * len(r['structural_bindings'])) for r in routes)
    semantics = {action.action_id: (a.ExecutionSemantics.HOLD_UNTIL_EXPIRY
        if action.kind == 'continuous' else a.ExecutionSemantics.COMPLETION_DRIVEN)
        for action in kw['spec'].actions}
    return pred, b, kw, availability, semantics


def decode(case, **extra):
    p, b, kw, available, semantics = case
    return api().decode_actions(p, b['inputs'], b['routing'], spec=kw['spec'],
        normalizers=kw['normalizers'], availability=available, execution_semantics=semantics,
        configured_arguments={('gravity_settle', 'clear'): True,
                              ('gravity_settle', 'ignored'): False},
        claim_resolver=lambda choice: ((choice.resource,) if choice.resource else
            tuple(c.ResourceRef(m, 'rotor') for m in choice.participants)), **extra)


def structural_case(*, enriched=False):
    case = example(enriched=enriched)
    p, b, kw, avail, semantics = case
    route = b['routing']
    target = next(i for i, (act, bound) in enumerate(route['structural_bindings'])
                  if act.action_id == 'gravity_settle' and bound.initiator == route['module_refs'][0])
    mask = tuple(i == target for i in range(len(route['structural_bindings'])))
    avail = (replace(avail[0], resource_mask=(False,) * len(avail[0].resource_mask), structural_mask=mask),)
    # The scope is singleton, and the independently predicted membership agrees.
    sets = p['structural']['arguments']['reference_set']
    sets['logits'].fill_(-1.)
    s = b['inputs']['structural']
    for i, arg in enumerate(sets['argument'].tolist()):
        if int(s['argument_binding'][arg]) != target: continue
        for j in range(int(sets['ptr'][i]), int(sets['ptr'][i+1])):
            domain = int(sets['domain_index'][j])
            if int(s['argument_domain_index'][domain]) == 0: sets['logits'][j] = 1.
    return (p, b, kw, avail, semantics), target


def test_mode_selection_and_inverse_f2_use_selected_payload():
    case = example(); p = case[0]['resource']
    p['mode_logits'][:] = torch.tensor([8., 0., 0., 9., 8., 0.])
    p['payload'][:] = torch.tensor([1., 2., 99., 99., 99., 0., 7., 8., 99.])
    out = decode(case)
    assert [x.choice.action_id for x in out] == ['pace', 'aim', 'pace']
    assert [x.choice.payload for x in out] == [(3., 5.), (1.,), (15., 17.)]
    assert out[0].execution is api().ExecutionSemantics.HOLD_UNTIL_EXPIRY
    assert out[0].duration_s is None
    assert out[1].execution is api().ExecutionSemantics.COMPLETION_DRIVEN


def test_busy_shared_internal_resource_cannot_redispatch_but_other_modules_can():
    case = list(example()); a = api(); avail = case[3][0]
    case[3] = (replace(avail, states=tuple((ref, a.ResourceStatus.BUSY if ref ==
        case[1]['routing']['resource_refs'][0] else state) for ref, state in avail.states)),)
    out = decode(case)
    assert {x.choice.initiator for x in out} == {c.ModuleRef('unit-1'), c.ModuleRef('unit-2')}


def test_resource_masks_filter_before_argmax_and_no_wait_is_created():
    case = list(example()); avail = case[3][0]
    case[0]['resource']['mode_logits'][:] = torch.tensor([99., 1., 99., 1., 99., 1.])
    case[3] = (replace(avail, resource_mask=(False, True, False, False, False, False)),)
    out = decode(case)
    assert len(out) == 1 and out[0].choice.action_id == 'aim'


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_predictions_fail_closed_even_when_not_selected(bad):
    case = example(); case[0]['structural']['candidate_logits'][0] = bad
    with pytest.raises(ValueError, match='finite'): decode(case)


def test_unknown_mode_route_fails_closed():
    case = example(); case[1]['inputs']['resource_candidates']['mode'][0] = 999
    with pytest.raises(ValueError): decode(case)


def test_payload_arity_mismatch_fails_closed():
    case = example(); case[0]['resource']['payload_ptr'][1] += 1
    with pytest.raises(ValueError): decode(case)


def test_unknown_ownership_is_not_free():
    case = list(example()); case[3] = (replace(case[3][0], states=()),)
    assert decode(case) == ()


def test_structural_top1_respects_initiator_group_and_current_mask():
    case = list(example()); p, b = case[:2]; route = b['routing']
    chosen = next(i for i, (act, bound) in enumerate(route['structural_bindings'])
        if act.action_id == 'dock' and bound.initiator == route['module_refs'][1])
    p['structural']['candidate_logits'].fill_(100.)
    p['structural']['candidate_logits'][chosen] = 1.
    case[3] = (replace(case[3][0], resource_mask=(False,) * 6,
        structural_mask=tuple(i == chosen for i in range(len(route['structural_bindings'])))),)
    out = decode(case)
    assert len(out) == 1
    assert out[0].candidate_index == chosen
    assert out[0].choice.action_id == 'dock'
    assert out[0].choice.initiator == c.ModuleRef('unit-1')
    assert len(out[0].choice.connectors) == 2
    assert out[0].execution is api().ExecutionSemantics.COMPLETION_DRIVEN


def test_settle_is_endpoint_free_and_uses_config_not_teacher_arguments():
    case, target = structural_case()
    out = decode(case)
    assert len(out) == 1 and out[0].candidate_index == target
    choice = out[0].choice
    assert choice.action_id == 'gravity_settle' and choice.connectors == ()
    assert choice.participants == (c.ModuleRef('unit-0'),)
    assert dict((p.name, p.value) for p in choice.parameters) == {
        'clear': True, 'duration': 1., 'members': (c.ModuleRef('unit-0'),)}


def test_learned_argument_kinds_decode_without_metric_accumulator():
    case, _ = structural_case(enriched=True)
    out = decode(case); args = {p.name: p.value for p in out[0].choice.parameters}
    assert args['adjust'] == (1., 1.) and args['select'] == 'a'
    assert args['decide'] is False and args['ignored'] is False
    assert isinstance(args['choose_module'], c.ModuleRef)
    assert isinstance(args['choose_connector'], c.ConnectorRef)
    assert all(isinstance(v, c.ConnectorRef) for v in args['extra_connectors'])


def test_ties_are_deterministic_and_inputs_are_immutable():
    case = example(); before = copy.deepcopy(case[:2])
    first = decode(case); second = decode(case)
    assert first == second
    assert [x.choice.action_id for x in first] == ['pace'] * 3
    def equal(a, b):
        if isinstance(a, torch.Tensor): assert torch.equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a: equal(a[key], b[key])
        elif isinstance(a, (tuple, list)):
            assert len(a) == len(b)
            for x, y in zip(a, b): equal(x, y)
        else: assert a == b
    equal(before, case[:2])


def test_two_samples_with_same_physical_ids_stay_separate():
    case = list(example(batch=True)); a = api(); av = case[3][0]
    case[3] = (replace(av, states=tuple((ref, a.ResourceStatus.BUSY) for ref, _ in av.states)), case[3][1])
    out = decode(case)
    assert len(out) == 3 and {r.sample_index for r in out} == {1}


def test_decoder_rejects_supervision_as_an_input():
    case = example(); contaminated = dict(case[1]['inputs'], supervision=case[1]['supervision'])
    case[1]['inputs'] = contaminated
    with pytest.raises(ValueError): decode(case)


def test_delegated_busy_remains_ineligible_across_repeated_physics_observations():
    case = list(example()); a = api(); av = case[3][0]
    busy = replace(av, states=tuple((ref, a.ResourceStatus.BUSY) for ref, _ in av.states))
    case[3] = (busy,)
    for _ in range(4): assert decode(case) == ()
    # Executor release is supplied externally, never guessed from elapsed ticks.
    case[3] = (av,)
    assert len(decode(case)) == 3


def test_f2_identity_mismatch_fails_closed():
    case = example(); case[1]['routing']['normalizer_fingerprint'] = 'foreign'
    with pytest.raises(ValueError): decode(case)


def test_physical_payload_outside_declared_parameter_limits_is_rejected_not_clipped():
    case = example(); case[0]['resource']['mode_logits'][:] = torch.tensor([0., 9.] * 3)
    case[0]['resource']['payload'].fill_(2.)  # inverse gives 5 rad, aim limit is [-2, 2].
    with pytest.raises(ValueError, match='limit'): decode(case)


def test_structural_endpoints_must_agree_with_current_tensor_routing():
    case, target = structural_case()
    case[1]['inputs']['structural']['connector_a'][target] = 0  # SETTLE must be -1.
    with pytest.raises(ValueError): decode(case)


def test_empty_or_duplicate_backend_claims_are_rejected():
    case = example(); p, b, kw, available, semantics = case
    for claims in ((), (b['routing']['resource_refs'][0],) * 2):
        with pytest.raises(ValueError):
            api().decode_actions(p, b['inputs'], b['routing'], spec=kw['spec'],
                normalizers=kw['normalizers'], availability=available,
                execution_semantics=semantics, configured_arguments={}, claim_resolver=lambda _: claims)


def test_physical_wheel_pair_uses_real_smores_adapter_without_twist_conversion():
    from pathlib import Path
    from drl.IL.bc_v1.smores_adapter import SMORESAdapter
    from drl.IL.bc_v1.dataset import CachedInputs, CachedSample
    from drl.IL.bc_v1.candidates import enumerate_candidates
    from drl.IL.bc_v1.feature_schema import extract_features, feature_schema_fingerprint
    from drl.IL.bc_v1.features import transform_inputs, PhysicalFrameConfig
    root = Path(__file__).resolve().parents[4]
    _, kw = fixture(1)
    spec = c.load_robot_spec(root / 'configs/bc_v1/smores_ep.robot.json')
    adapter = SMORESAdapter(spec)
    graph = adapter.encode_state({'nodes': [{'module_id': 'physical-module',
        'position': [0., 0., 0.], 'orientation': [0., 0., 0., 1.]}]})
    policy = c.PolicyInputs(graph, c.ObservationBundle(), c.TaskContext('navigation'))
    features = extract_features(policy, transform_inputs(policy, PhysicalFrameConfig()), spec, kw['feature_schema'])
    normal_policy = f2.load_normalization_policy(root / 'configs/bc_v1/normalization.json', kw['feature_schema'])
    _, rb, ab = f2._catalog_plan(spec, kw['feature_schema'], normal_policy)
    stats = f2.NormalizerStats(tuple(sorted({'robot_spec_fingerprint': c.robot_spec_fingerprint(spec),
        'feature_schema_fingerprint': feature_schema_fingerprint(kw['feature_schema']),
        'policy_fingerprint': f2.policy_fingerprint(normal_policy, kw['feature_schema'])}.items())),
        tuple((key, f2.Scaler(1., 2., 2, 'train_standard', False)) for bind in (*rb, *ab) for key in bind.keys), rb, ab)
    kw.update(spec=spec, normalization_policy=normal_policy, normalizers=stats)
    sample = CachedSample(CachedInputs(policy, features, enumerate_candidates(graph, spec, kw['feature_schema'])),
        c.SupervisionBundle(), c.SampleAudit('synthetic', 0))
    b = tensorize(sample, kw)
    with torch.no_grad(): p = complete_model(kw)(b['inputs'])
    route = b['routing']; a = api()
    q = next(i for i, row in enumerate(route['resource_candidates']) if row.mode.mode_id == 'wheel_velocity')
    prediction_row = p['resource']['payload_candidate'].tolist().index(q)
    lo, hi = p['resource']['payload_ptr'][prediction_row:prediction_row+2].tolist()
    p['resource']['payload'][lo:hi] = torch.tensor([2., -1.])
    available = (a.DispatchAvailability(tuple((r.ref, a.ResourceStatus.FREE) for r in graph.resources),
        tuple(i == q for i in range(len(route['resource_candidates']))), (False,) * len(route['structural_bindings'])),)
    out = a.decode_actions(p, b['inputs'], route, spec=spec, normalizers=stats,
        availability=available, execution_semantics={'wheel_velocity': a.ExecutionSemantics.HOLD_UNTIL_EXPIRY},
        configured_arguments={}, claim_resolver=lambda choice: (choice.resource,))
    assert len(out) == 1 and out[0].choice.payload == (5., -1.)
    assert adapter.decode_action(out[0].choice) == {'command': 'wheel_velocity',
        'module_ids': ['physical-module'], 'parameters': {'left_rad_s': 5., 'right_rad_s': -1.}}


def test_decoder_ignores_teacher_fields_incidentally_retained_in_g2_routing():
    case = example(); expected = decode(case)
    route = case[1]['routing']
    for key in ('audit', 'resource_targets', 'structural_targets', 'physical_supervision'):
        route[key] = object()
    assert decode(case) == expected


def test_selected_structural_action_cannot_redispatch_busy_claim():
    case, _ = structural_case(); case = list(case); a = api(); av = case[3][0]
    case[3] = (replace(av, states=tuple((ref, a.ResourceStatus.BUSY) for ref, _ in av.states)),)
    assert decode(case) == ()


def test_actions_in_same_decode_do_not_conflict_on_shared_physical_claims():
    case = list(example()); av = case[3][0]
    case[3] = (replace(av, structural_mask=(True,) * len(av.structural_mask)),)
    out = decode(case)
    claims = [claim for action in out for claim in action.claims]
    assert len(claims) == len(set(claims))


def test_singleton_resource_mode_needs_no_mode_logits():
    from drl.IL.tests.bc_v1.test_losses import recatalog
    sample, kw = fixture(2)
    mt = kw['spec'].module_types[0]; resource = mt.resources[0]
    spec = replace(kw['spec'], module_types=(replace(mt, resources=(replace(resource, modes=resource.modes[:1]),)),))
    sample, kw = recatalog(sample, kw, spec)
    b = tensorize(sample, kw)
    with torch.no_grad(): p = complete_model(kw)(b['inputs'])
    p['resource']['payload'].fill_(0.)
    a = api(); route = b['routing']
    av = a.DispatchAvailability(tuple((ref, a.ResourceStatus.FREE) for ref in route['resource_refs']),
        (True,) * len(route['resource_candidates']), (False,) * len(route['structural_bindings']))
    out = a.decode_actions(p, b['inputs'], route, spec=spec, normalizers=kw['normalizers'],
        availability=(av,), execution_semantics={'pace': a.ExecutionSemantics.HOLD_UNTIL_EXPIRY},
        configured_arguments={}, claim_resolver=lambda choice: (choice.resource,))
    assert p['resource']['mode_logits'].numel() == 0
    assert len(out) == 2 and all(x.choice.action_id == 'pace' for x in out)


def test_generic_decoder_import_needs_no_ros_isaac_smores_or_metrics():
    import subprocess
    import sys
    script = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('rclpy', 'isaacsim', 'omni', 'smores_ep', 'mssr_expert') or fullname.endswith('.metrics'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from drl.IL.bc_v1.action_decoder import decode_actions
assert callable(decode_actions)
'''
    result = subprocess.run([sys.executable, '-B', '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
