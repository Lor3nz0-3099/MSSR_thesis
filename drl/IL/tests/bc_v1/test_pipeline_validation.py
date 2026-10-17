"""Bounded post-E2 physical-loss guard; faults never change production assets."""
from dataclasses import replace
import importlib
import importlib.util
from pathlib import Path

import pytest

from drl.IL.bc_v1 import bc_v1_contract as c
from drl.IL.bc_v1.feature_schema import extract_features, load_feature_schema
from drl.IL.bc_v1.features import transform_inputs
from drl.IL.bc_v1.policy_inputs import select_policy_inputs
from drl.IL.bc_v1.source_frame_bridges import (
    BridgeEvidence, BridgeResolution, SourceFrameBridgeBatch, frame_config_for)
from drl.IL.bc_v1.features import SourceFrameTransform
from drl.IL.tests.bc_v1 import fixtures as f

ROOT = Path(__file__).resolve().parents[4]


def api():
    name = 'drl.IL.bc_v1.pipeline_validation'
    assert importlib.util.find_spec(name), 'Missing Task 5 physical pipeline validator'
    return importlib.import_module(name)


def bridges(frame='world', *, certified=True):
    world = BridgeResolution('world', 'world', 'current', None, True,
                             'implicit_world_identity', None, ())
    resolutions = (world,)
    if frame != 'world':
        resolutions += (BridgeResolution(frame, 'world', 'current',
            SourceFrameTransform(frame, (0., 0., 0.)) if certified else None, certified,
            'certified_current_source' if certified else 'no_verified_source_contract',
            'test.current-source.v1' if certified else None,
            (BridgeEvidence('current-source.json', 'a'*64),) if certified else ()),)
    return SourceFrameBridgeBatch(resolutions, (('provider_id', 'test'), ('revision', '1')))


def prepared(task='navigation', static=None, dynamic=(), batch=None, graph=None):
    if static is None:
        static = (c.NumericFeature('map.goal.center_xyz_m', (2., 3., 0.)),)
    batch = bridges('map') if batch is None else batch
    graph = f.state_graph(2, 1) if graph is None else graph
    inputs = select_policy_inputs(graph, c.ObservationBundle(dynamic), c.TaskContext(task, static))
    geometry = transform_inputs(inputs, frame_config_for(batch))
    schema = load_feature_schema(ROOT/'configs/bc_v1/features.json')
    features = extract_features(inputs, geometry, f.robot_spec(1), schema)
    return inputs, geometry, features, batch, schema


def validate(data):
    return api().validate_physical_pipeline(*data)


@pytest.mark.parametrize('frame', ['world', 'map', 'stage_local'])
def test_certified_named_frame_cannot_silently_invalidate_in_e1(frame):
    validator = api()
    inputs, geometry, features, batch, schema = prepared(
        static=(c.NumericFeature(frame+'.goal.center_xyz_m', (2., 3., 0.)),), batch=bridges(frame))
    corrupted = replace(geometry, quantities=(replace(geometry.quantities[0], valid=False, values=None),))
    with pytest.raises(ValueError, match='UNEXPECTED_E1_INVALIDATION'):
        validator.validate_physical_pipeline(inputs, corrupted, features, batch, schema)


def test_unresolved_named_frame_has_explicit_reason_and_no_identity_fallback():
    data = prepared(batch=bridges('map', certified=False))
    diagnostics = validate(data)
    assert [(d.classification, d.reason) for d in diagnostics] == [
        ('EXPECTED_UNRESOLVED_FRAME', 'no_verified_source_contract')]
    assert not data[1].quantities[0].valid and data[1].quantities[0].values is None
    resolved = prepared()
    with pytest.raises(ValueError, match='FRAME_SEMANTICS_UNKNOWN'):
        validate((*resolved[:3], data[3], data[4]))
    with pytest.raises(ValueError, match='FRAME_SEMANTICS_UNKNOWN'):
        validate((*data[:3], bridges(), data[4]))


@pytest.mark.parametrize('fault', ['remove_object', 'invalidate_geometry'])
def test_valid_environment_object_cannot_silently_disappear_in_e2(fault):
    inputs, geometry, features, batch, schema = prepared()
    if fault == 'remove_object':
        corrupted = replace(features, environment=())
    else:
        row = features.environment[0]
        # Leave valid object-type one-hot channels; these are not physical geometry.
        corrupted_row = replace(row, validity=(False,)*3 + row.validity[3:])
        corrupted = replace(features, environment=(corrupted_row,))
    with pytest.raises(ValueError, match='UNEXPECTED_E2_LOSS'):
        validate((inputs, geometry, corrupted, batch, schema))


def test_intrinsic_and_local_channels_do_not_trigger_global_frame_loss():
    graph = f.state_graph(2, 1)
    graph = replace(graph, connectors=(replace(graph.connectors[0], features=(
        c.NumericFeature('position_body_m', (.1, .2, .3)),)),)+graph.connectors[1:])
    data = prepared('button', (
        c.NumericFeature('body_local.button.center_xyz_m', (1., 2., 3.)),
        c.NumericFeature('map.obstacles.0.size_xyz_m', (1., 2., 3.)),
        c.NumericFeature('button.plunger_depth_m', (.04,))),
        (c.NumericFeature('button.depression_m', (.01,)),), bridges('map', certified=False), graph)
    diagnostics = validate(data)
    assert all(d.classification == 'EXPECTED_SCHEMA_VALIDITY' for d in diagnostics)
    assert data[1].quantities[1].valid  # Intrinsic box dimensions need no global bridge.
    local = next(q for q in data[1].local_geometry if q.source_name == 'position_body_m')
    assert local.values == (.1, .2, .3) and local.source_frame == local.output_frame == 'module_body'
    assert any(q.source_frame == q.output_frame == 'resource_local' for q in data[1].local_geometry)


@pytest.mark.parametrize('case', ['gap', 'stairs', 'cones'])
def test_existing_physical_dedup_is_accepted_without_one_row_per_alias(case):
    q = c.NumericFeature
    if case == 'gap':
        static = (q('world.gap.near_edge_x_m', (2.,)), q('world.gap.near_edge_world_xyz_m', (2., 0., 0.)),
                  q('world.gap.near_edge_center_world_xyz_m', (2., 0., 0.)),
                  q('world.gap.crossing_direction_world_xy', (1., 0.)))
        task = 'gap'
    elif case == 'stairs':
        static = (q('world.stairs.first_riser_x_m', (2.,)),
                  q('world.stairs.first_riser_center_world_xyz_m', (2., 0., 0.)),
                  q('world.stairs.ascent_direction_world_xy', (1., 0.)),
                  q('world.stairs.top_heights_m', (.1, .2)), q('world.stairs.top_heights_world_m', (.1, .2)))
        task = 'stairs'
    else:
        static = (q('world.cones.0.center_xy_m', (1., 2.)),
                  q('world.environment_cones.9.center_xyz_m', (1., 2., .2)),
                  q('environment_cones.9.height_m', (.4,)), q('environment_cones.9.radius_m', (.1,)),
                  q('cones.cone_height_m', (.4,)), q('cones.cone_radius_m', (.1,)),
                  q('world.environment.floor_height_m', (0.,)))
        task = 'navigation'
    data = prepared(task, static, batch=bridges())
    diagnostics = validate(data)
    assert any(d.classification == 'EXPECTED_DEDUP_OR_ALIAS' for d in diagnostics)
    assert all(not d.classification.startswith('UNEXPECTED') for d in diagnostics)


def test_different_object_or_invalid_alias_cannot_explain_e2_loss():
    q = c.NumericFeature
    data = prepared(static=(q('world.goal.center_xyz_m', (2., 3., 0.)),
                            q('world.obstacles.0.center_xyz_m', (2., 3., 0.))), batch=bridges())
    inputs, geometry, features, batch, schema = data
    corrupted = replace(features, environment=tuple(r for r in features.environment if r.object_type != 'goal'))
    with pytest.raises(ValueError, match='UNEXPECTED_E2_LOSS'):
        validate((inputs, geometry, corrupted, batch, schema))


def test_unobserved_source_and_missing_centroid_have_explicit_schema_reasons():
    q = c.NumericFeature
    data = prepared(static=(q('map.goal.center_xyz_m', (0., 0., 0.), False),), batch=bridges('map', certified=False))
    assert validate(data)[0].classification == 'EXPECTED_SCHEMA_VALIDITY'
    graph = f.state_graph(2, 1)
    graph = replace(graph, modules=tuple(replace(m, pose_valid=False) for m in graph.modules))
    data = prepared(graph=graph)
    result = validate(data)
    assert result[0].classification == 'EXPECTED_SCHEMA_VALIDITY'
    assert 'centroid' in result[0].reason


@pytest.mark.parametrize('fault', ['missing_quantity', 'wrong_frame', 'wrong_kind'])
def test_missing_or_relabelled_e1_quantity_is_not_expected_missingness(fault):
    inputs, geometry, features, batch, schema = prepared()
    quantities = () if fault == 'missing_quantity' else (
        replace(geometry.quantities[0], **({'source_frame': 'world'} if fault == 'wrong_frame' else {'kind': 'intrinsic'})),)
    with pytest.raises(ValueError, match='UNEXPECTED_E1_INVALIDATION|FRAME_SEMANTICS_UNKNOWN'):
        validate((inputs, replace(geometry, quantities=quantities), features, batch, schema))


def test_known_map_476_and_stage_local_1725_have_no_unexpected_diagnostics():
    from drl.IL.tests.bc_v1.test_dataset import real_row, real_derivation_config, derive_one
    for episode, index, frame in (
        ('expert:rc_car:seed-005100', 476, 'map'),
        ('teleop-0e4e34fe3a5f4ed1b6a86642acf4fac8', 1725, 'stage_local')):
        result = derive_one(real_row(episode, index), *real_derivation_config())
        assert result.diagnostics
        assert all(not d.classification.startswith('UNEXPECTED') and d.classification != 'FRAME_SEMANTICS_UNKNOWN'
                   for d in result.diagnostics)
        assert any(d.classification == 'CERTIFIED_VALID' and d.source_name.startswith(frame+'.')
                   for d in result.diagnostics)
        assert all(q.valid for q in result.geometry.quantities if q.source_frame == frame)
        if index == 476:
            assert len(result.features.relations) == 24


def test_derive_fault_raises_before_return_without_repeating_e1_e2(monkeypatch):
    from drl.IL.bc_v1 import dataset
    from drl.IL.tests.bc_v1.test_dataset import Adapter, world_runtime
    schema = load_feature_schema(ROOT/'configs/bc_v1/features.json')
    config = dict(world_runtime(), spec=f.robot_spec(1), schema=schema, task_config={},
                  metadata_reader=lambda row: {'sources': [], 'audit': []})
    monkeypatch.setattr(dataset, 'build_task_view', lambda *args: (
        c.TaskContext('navigation', (c.NumericFeature('world.goal.center_xyz_m', (1., 2., 0.)),)),
        c.ObservationBundle()))
    calls = []; e1, e2 = dataset.transform_inputs, dataset.extract_features
    def transform(*args):
        calls.append('E1'); return e1(*args)
    def extract(*args):
        calls.append('E2'); return replace(e2(*args), environment=())
    monkeypatch.setattr(dataset, 'transform_inputs', transform)
    monkeypatch.setattr(dataset, 'extract_features', extract)
    with pytest.raises(ValueError, match='UNEXPECTED_E2_LOSS'):
        dataset.derive_physical_inputs({'graph_t': {'n': 2}}, Adapter(), config,
                                      dataset.CurrentCandidateCache(config['spec'], schema))
    assert calls == ['E1', 'E2']
