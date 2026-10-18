from dataclasses import replace
import importlib
import importlib.util
import copy

import pytest
import torch

from drl.IL.bc_v1 import normalization as f2
from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.tests.bc_v1.g2_fixtures import fixture, tensorize, with_structural


def api():
    assert importlib.util.find_spec('drl.IL.bc_v1.tensorization'), 'Missing G2 tensorization contract'
    return importlib.import_module('drl.IL.bc_v1.tensorization')


def assert_tree_equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: assert_tree_equal(a[key], b[key])
    else: assert a == b


def test_f2_once_immutable_no_fit_and_literal_values(monkeypatch):
    api(); sample, kwargs = fixture(); original = copy.deepcopy(sample)
    calls = {'features': 0, 'target': 0, 'structural': 0}
    for name, key in [('normalize_features', 'features'), ('normalize_target', 'target'),
                      ('normalize_structural_target', 'structural')]:
        original_fn = getattr(f2, name)
        def counted(*args, _fn=original_fn, _key=key, **kw):
            calls[_key] += 1; return _fn(*args, **kw)
        monkeypatch.setattr(f2, name, counted)
    def forbid(*a, **kw): raise AssertionError('F2 fit must never run in G2')
    monkeypatch.setattr(f2, 'fit_normalizers', forbid)
    from drl.IL.bc_v1 import features, feature_schema
    monkeypatch.setattr(features, 'transform_inputs', forbid)
    monkeypatch.setattr(feature_schema, 'extract_features', forbid)
    result = tensorize(sample, kwargs)
    assert calls == {'features': 1, 'target': 1, 'structural': 0}
    assert sample == original
    assert result['inputs']['module_x'][0, 0].item() == pytest.approx(-1.)
    assert result['supervision']['resource']['payload'].tolist() == [2., 3.]


def test_preserved_channels_missing_dtype_and_variable_width():
    api(); sample, kwargs = fixture(3)
    row = sample.inputs.features.modules[0]
    row = replace(row, values=(99.,)+row.values[1:], validity=(False,)+row.validity[1:])
    sample = replace(sample, inputs=replace(sample.inputs,
        features=replace(sample.inputs.features, modules=(row,)+sample.inputs.features.modules[1:])))
    result = tensorize(sample, kwargs); x = result['inputs']['module_x']
    assert x.shape == (3, 30) and x.dtype == torch.float32
    assert x[0, 0] == 0 and not result['inputs']['module_validity'][0, 0]
    assert x[0, 3:9].tolist() == list(row.values[3:9])
    assert result['inputs']['module_validity'].dtype == torch.bool
    assert result['inputs']['connector_to_module'].dtype == torch.int64
    larger, k = fixture(11)
    assert tensorize(larger, k)['inputs']['module_x'].shape == (11, 30)


def test_supervision_audit_changes_cannot_change_inputs():
    api(); sample, kwargs = fixture(); before = tensorize(sample, kwargs)
    changed = replace(sample, supervision=c.SupervisionBundle(),
        audit=replace(sample.audit, provenance='stage-control-state-secret', notes=('future',)))
    assert_tree_equal(before['inputs'], tensorize(changed, kwargs)['inputs'])
    forbidden = {'stage','configuration_goal','control_state','FSM','current_primitive',
                 'future','target','audit','provenance','supervision'}
    def visit(value):
        assert isinstance(value, (dict, torch.Tensor))
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for v in value.values(): visit(v)
    visit(before['inputs'])
    assert before['routing']['module_refs'] == tuple(m.ref for m in sample.inputs.policy.graph.modules)


def test_geometry_semantic_slots_distinguish_scalars_and_ignore_object_ordinals():
    t = api(); sample, kwargs = fixture(); result = tensorize(sample, kwargs)
    assert {v['semantic_slot'] for v in result['routing']['environment_layouts'].values()} == {
        'gap.width_m', 'gap.landing_length_m'}
    layout = t.load_tensor_layout(__import__('drl.IL.tests.bc_v1.g2_fixtures',fromlist=['ROOT']).ROOT/'configs/bc_v1/tensor_layout.v1.json')
    assert t.semantic_geometry_slot('world.obstacles.12.size_xyz_m', layout) == 'obstacles.*.size_xyz_m'
    assert t.semantic_geometry_slot('world.obstacles.99.size_xyz_m', layout) == 'obstacles.*.size_xyz_m'
    with pytest.raises(ValueError): t.semantic_geometry_slot('stage.fake_future', layout)


def navigation_cone_sample(*, valid=True):
    from drl.IL.bc_v1.features import PhysicalFrameConfig, transform_inputs
    from drl.IL.bc_v1.feature_schema import extract_features
    from drl.IL.bc_v1.policy_inputs import select_policy_inputs
    sample, kwargs = fixture()
    context = c.TaskContext('navigation', (
        c.NumericFeature('world.cones.0.center_xy_m', (4., 6.), valid=valid),
        c.NumericFeature('world.environment_cones.1.center_xyz_m', (8., 9., 7.)),
        c.NumericFeature('environment_cones.1.radius_m', (.2,)),
        c.NumericFeature('environment_cones.1.height_m', (1.,))))
    inputs = select_policy_inputs(sample.inputs.policy.graph, c.ObservationBundle(), context)
    assert inputs.context == context  # Exercise actual D2 admission before G2.
    geometry = transform_inputs(inputs, PhysicalFrameConfig())
    features = extract_features(inputs, geometry, kwargs['spec'], kwargs['feature_schema'])
    return replace(sample, inputs=replace(sample.inputs, policy=inputs, features=features)), kwargs


def test_navigation_cone_xy_semantic_slot_removes_frame_and_instance_only():
    from drl.IL.tests.bc_v1.g2_fixtures import ROOT
    layout = api().load_tensor_layout(ROOT/'configs/bc_v1/tensor_layout.v1.json')
    for name in ('map.cones.0.center_xy_m', 'world.cones.19.center_xy_m'):
        assert api().semantic_geometry_slot(name, layout) == 'cones.*.center_xy_m'
    assert api().semantic_geometry_slot('map.cones.0.center_xyz_m', layout) == 'cones.*.center_xyz_m'


@pytest.mark.parametrize('valid', [True, False])
def test_navigation_cone_xy_preserves_line_xy_unknown_height_and_validity(valid):
    sample, kwargs = navigation_cone_sample(valid=valid)
    result = tensorize(sample, kwargs)
    row = result['inputs']['environment']['c_static:cones.*.center_xy_m:line:6']
    assert row['x'].dtype == torch.float32 and row['x'].shape == (1, 30)
    assert row['validity'].dtype == torch.bool and row['validity'].shape == (1, 15)
    assert row['validity'][0, :6].tolist() == [valid]*6
    # Centroid is (1,0,0). The first three values are a LINE parameter origin,
    # not a measured 3D cone center; the final three retain its free direction.
    assert row['x'][0, :6].tolist() == ([3., 6., 0., 0., 0., 1.] if valid else [0.]*6)
    assert torch.equal(row['x'][:, 15:], row['validity'].float())
    assert torch.isfinite(row['x']).all()
    point = result['inputs']['environment']['c_static:environment_cones.*.center_xyz_m:point3:3']
    assert point['x'].shape == (1, 24) and point['validity'].shape == (1, 12)
    assert point['x'][0, :3].tolist() == [7., 9., 7.]
    for name in ('radius_m', 'height_m'):
        assert any('environment_cones.*.'+name+':' in key for key in result['inputs']['environment'])


def test_navigation_cone_layout_stays_fail_closed():
    from drl.IL.tests.bc_v1.g2_fixtures import ROOT
    layout = api().load_tensor_layout(ROOT/'configs/bc_v1/tensor_layout.v1.json')
    for name in ('map.cones.0.invented_future_field', 'map.environment_cones.0.center_xy_m'):
        with pytest.raises(ValueError, match='Unapproved semantic geometry slot'):
            api().semantic_geometry_slot(name, layout)


def test_navigation_cone_mixed_batch_h1_width_is_derived_and_old_weights_rejected():
    from drl.IL.bc_v1.batching import collate_policy_samples
    from drl.IL.tests.bc_v1.test_action_heads import complete_model, floats
    sample, kwargs = navigation_cone_sample()
    ordinary, ordinary_kwargs = fixture(2)
    batch = collate_policy_samples([tensorize(sample, kwargs), tensorize(ordinary, ordinary_kwargs)])
    model = complete_model(kwargs).eval()
    assert len(model.slots) == 54
    first = next(layer for layer in model.environment_encoder.modules() if isinstance(layer, torch.nn.Linear))
    assert first.in_features == 147 and first.weight.shape == (128, 147)
    old = model.state_dict()
    old['environment_encoder.0.weight'] = old['environment_encoder.0.weight'][:, :-1].clone()
    with pytest.raises(RuntimeError, match='size mismatch for environment_encoder.0.weight'):
        model.load_state_dict(old)
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.no_grad():
            prediction = model(batch['inputs'])
        assert all(torch.isfinite(value).all() for value in floats(prediction))
    finally:
        torch.set_num_threads(previous)


def test_rename_and_permutation_preserve_numeric_features_and_bindings():
    api(); from dataclasses import fields, is_dataclass
    sample, kwargs = fixture()
    def rename(value):
        if isinstance(value, c.ModuleRef): return c.ModuleRef('renamed-'+value.module_id)
        if is_dataclass(value): return replace(value, **{f.name:rename(getattr(value,f.name)) for f in fields(value)})
        if isinstance(value, tuple): return tuple(rename(v) for v in value)
        if isinstance(value, frozenset): return frozenset(rename(v) for v in value)
        return value
    assert_tree_equal(tensorize(sample,kwargs)['inputs'], tensorize(rename(sample),kwargs)['inputs'])
    reversed_graph = replace(sample.inputs.policy.graph, modules=tuple(reversed(sample.inputs.policy.graph.modules)))
    p = replace(sample.inputs.policy, graph=reversed_graph)
    v = replace(sample.inputs.features, modules=tuple(reversed(sample.inputs.features.modules)))
    reordered = replace(sample, inputs=replace(sample.inputs, policy=p, features=v))
    a,b=tensorize(sample,kwargs),tensorize(reordered,kwargs)
    assert torch.equal(a['inputs']['module_x'].flip(0), b['inputs']['module_x'])
    assert b['inputs']['connector_to_module'].tolist() == [2,2,1,1,0,0]


def test_unknown_task_and_no_train_support_diagnostics():
    api(); sample, kwargs = fixture(geometry=False)
    out=tensorize(sample,kwargs)
    assert out['inputs']['task_one_hot'].tolist() == [[1.,0.,0.,0.,0.]]
    assert out['inputs']['task_valid'].tolist() == [False]
    assert 'no_train_support' in out['routing']


def test_structural_groups_are_initiator_bindings_not_global_rows():
    api(); sample, kwargs=fixture(); sample=with_structural(sample)
    out=tensorize(sample,kwargs)
    assert 'structural' in out['inputs'], 'Missing initiator binding groups'
    s=out['inputs']['structural']; y=out['supervision']['structural']
    assert s['group_ptr'].tolist() == [0,17,34,51]
    assert y['group'].tolist() == [0,1]
    assert y['candidate'].numel() == 2
    for i,target in enumerate(sample.supervision.structural_targets):
        b=y['candidate'][i].item()
        assert s['group_ptr'][i] <= b < s['group_ptr'][i+1]
        refs=out['routing']['connector_refs']
        assert (refs[s['connector_a'][b]],refs[s['connector_b'][b]]) == target.connectors


def test_settle_reference_set_uses_domain_not_normalizer_size():
    api(); sample, kwargs=fixture(5,0); sample=with_structural(sample,skill=True)
    out=tensorize(sample,kwargs)
    assert 'structural' in out['inputs'], 'Missing typed structural domains'
    s=out['inputs']['structural']; y=out['supervision']['structural']
    assert s['connector_a'].tolist() == [-1]*5 and s['connector_b'].tolist() == [-1]*5
    assert s['peer'].tolist() == [-1]*5
    assert s['group_ptr'].tolist() == list(range(6))
    domains=out['routing']['argument_domains']
    index=next(i for i,d in enumerate(domains) if d.parameter.kind=='module_refs')
    assert next(b.size for b in kwargs['normalizers'].argument_bindings if b.name=='members') == 1
    a,z=s['argument_domain_ptr'][index:index+2].tolist()
    assert z-a == 5 and s['argument_domain_required'][a:z].sum() == 0
    assert s['scope_required'][:5].sum() == 1
    assert y['membership_value'].numel() == 10
    assert y['membership_value'].sum() == 2
    assert y['argument_value'][y['argument_value_valid']].tolist() == [2.,2.]


def test_candidate_groups_independent_of_targets_and_keep_generic_resource_modes():
    api(); sample, kwargs=fixture(); chosen=with_structural(sample)
    a,b=tensorize(sample,kwargs),tensorize(chosen,kwargs)
    assert 'structural' in a['inputs'], 'Missing label-independent groups'
    assert_tree_equal(a['inputs'],b['inputs'])
    r=a['inputs']['resource_candidates']
    assert r['group_ptr'].tolist() == [0,2,4,6]
    assert a['supervision']['resource']['candidate'].tolist() == [0]
    assert r['payload_ptr'].tolist() == [0,2,3,5,6,8,9]


def test_reference_domain_and_parameter_order_are_entity_order_equivariant():
    api(); sample, kwargs=fixture(3,0); a=tensorize(with_structural(sample,skill=True),kwargs)
    assert 'structural' in a['inputs'], 'Missing structural binding references'
    s=a['inputs']['structural']
    assert s['initiator'].tolist() == [0,1,2]
    assert s['scope_members'].tolist() == [0,1,2]*3
    assert s['scope_required'].tolist() == [True,False,False,False,True,False,False,False,True]
    assert s['argument_domain_index'].dtype == torch.int64
    assert s['scope_required'].dtype == torch.bool


def test_coordinate_slots_are_stable_under_equivalent_catalog_permutation():
    sample, kwargs = fixture()
    before = tensorize(sample, kwargs)
    spec = kwargs['spec']
    module_type = spec.module_types[0]
    resource = module_type.resources[0]
    swapped_resource = replace(resource, coordinates=tuple(reversed(resource.coordinates)))
    swapped_type = replace(module_type, resources=(swapped_resource,)+module_type.resources[1:])
    swapped = replace(spec, module_types=(swapped_type,)+spec.module_types[1:])
    assert c.robot_spec_fingerprint(swapped) == c.robot_spec_fingerprint(spec)
    assert_tree_equal(before['inputs'], tensorize(sample, {**kwargs, 'spec': swapped})['inputs'])


def test_reference_set_membership_does_not_inherit_participant_requirements():
    sample, kwargs = fixture(3, 0)
    sample = with_structural(sample, skill=True)
    first = sample.supervision.structural_targets[0]
    first = replace(first, parameters=tuple(replace(p, value=(sample.inputs.policy.graph.modules[1].ref,))
        if p.name == 'members' else p for p in first.parameters))
    sample = replace(sample, supervision=replace(sample.supervision,
        structural_targets=(first,)+sample.supervision.structural_targets[1:]))
    out = tensorize(sample, kwargs)
    assert out['supervision']['structural']['membership_value'][:3].tolist() == [0.,1.,0.]
    assert out['supervision']['structural']['membership_valid'][:3].tolist() == [True,True,True]
    assert out['inputs']['structural']['scope_required'][:3].tolist() == [True,False,False]


def test_participant_scope_has_independent_membership_supervision():
    sample, kwargs = fixture(3, 0)
    sample = with_structural(sample, skill=True)
    before = tensorize(sample, kwargs)
    first = sample.supervision.structural_targets[0]
    changed = replace(first, participants=first.participants+(sample.inputs.policy.graph.modules[2].ref,))
    changed_sample = replace(sample, supervision=replace(sample.supervision,
        structural_targets=(changed,)+sample.supervision.structural_targets[1:]))
    after = tensorize(changed_sample, kwargs)
    assert_tree_equal(before['inputs'], after['inputs'])
    y = after['supervision']['structural']
    assert y['scope_membership_ptr'].tolist() == [0,3,6]
    assert y['scope_membership_index'][:3].tolist() == [0,1,2]
    assert y['scope_membership_value'][:3].tolist() == [1.,0.,1.]
    assert y['scope_membership_valid'][:3].tolist() == [False,True,True]
    assert before['supervision']['structural']['scope_membership_value'][:3].tolist() == [1.,0.,0.]
