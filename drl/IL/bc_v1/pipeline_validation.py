"""Bounded post-E2 physical-loss gate, using only completed current outputs.

No source reads, observers, labels, transforms or FeatureView recomputation.
E2's existing schema codecs/equivalence rules certify representation aliases;
small E1-derived witnesses are audit probes, never new learned feature rows.
"""
from .feature_schema import (
    _deduplicate_environment, _environment_identity, _environment_layout,
    _environment_row, _geometry_payload, _physical_values, _same_numbers, _source_parts)
from .features import _geometry_kind
from .source_frame_bridges import PipelineDiagnostic

_GLOBAL = frozenset({'world', 'map', 'stage_local'})
_LOCAL = frozenset({'module_body', 'endpoint_a_body', 'connection_intrinsic', 'resource_local'})
_NEEDS_CENTROID = frozenset({'point', 'line', 'planes', 'planar_pose'})


def _fail(classification, scope, name, reason):
    raise ValueError(f'{classification}: {scope}/{name}: {reason}')


def _e2_representation(quantity, source, features, schema):
    """Require valid physical geometry or an existing E2 dedup certificate.

    Do not treat object-type one-hot channels as geometric representation.
    Witnesses use already-transformed E1 values and the existing E2 codecs;
    no new extraction/dedup algorithm or physical transformation is introduced.
    """
    _, base = _source_parts(quantity.source_name)
    object_ref, object_type = _environment_identity(base)
    kind, width, units, parts = _environment_layout(quantity, source, schema)
    alias = False
    for part in range(parts):
        witness = _environment_row(quantity, object_ref, object_type, kind, width, units, part, schema)
        payload = _geometry_payload(witness)
        direct = any(row.ref == witness.ref and row.output_frame == witness.output_frame
            and row.geometry_kind == witness.geometry_kind and row.object_type == object_type
            and _same_numbers(_geometry_payload(row), payload) for row in features.environment)
        if direct:
            continue
        # The original rule must certify removal of THIS witness in favor of
        # actual valid output geometry. Equal numbers from another object alone
        # are insufficient; unresolved/invalid alternatives cannot certify it.
        retained = _deduplicate_environment((witness, *features.environment))
        if any(row.ref == witness.ref for row in retained):
            _fail('UNEXPECTED_E2_LOSS', quantity.source_scope, quantity.source_name,
                  'valid E1 geometry has no valid E2 representation or certified existing alias')
        alias = True
    return alias


def validate_physical_pipeline(inputs, geometry, features, bridges, schema):
    """Validate the completed D2/E1/E2 derivation without executing it again.

    Explicit schema exceptions: unobserved source values; intrinsic channels;
    unowned body-local task geometry; unframed geometry; missing centroid for
    position-dependent codecs. No such exception admits an identity fallback.
    """
    if features.frame != geometry.frame:
        _fail('FRAME_SEMANTICS_UNKNOWN', 'g_current', 'physical_frame', 'E1/E2 frames differ')
    for local in geometry.local_geometry:
        if local.source_frame not in _LOCAL or local.output_frame != local.source_frame:
            _fail('FRAME_SEMANTICS_UNKNOWN', 'g_current', local.source_name,
                  'local/intrinsic/resource geometry must retain its local frame')
    sources = {(scope, q.name): q for scope, quantities in (
        ('c_static', inputs.context.static_geometry), ('o_dynamic', inputs.observation.features))
        for q in quantities}
    quantities = {(q.source_scope, q.source_name): q for q in geometry.quantities}
    if len(quantities) != len(geometry.quantities) or set(quantities) != set(sources):
        _fail('UNEXPECTED_E1_INVALIDATION', 'c_static/o_dynamic', 'quantities',
              'E1 must retain exactly the current D2 quantities')
    diagnostics = []
    for key, source in sources.items():
        quantity = quantities[key]
        frame, base = _source_parts(source.name)
        expected_kind = _geometry_kind(base) if frame is not None else 'intrinsic'
        output = 'intrinsic' if expected_kind in ('intrinsic', 'dimensions') else 'robot_centroid'
        if (quantity.source_frame != frame or quantity.kind != expected_kind
                or quantity.output_frame != output):
            _fail('FRAME_SEMANTICS_UNKNOWN', *key, 'E1 changed the source frame/kind semantics')
        classification, reason = 'CERTIFIED_VALID', 'current certified spatial geometry retained'
        if not source.valid:
            if quantity.valid or quantity.values is not None:
                _fail('FRAME_SEMANTICS_UNKNOWN', *key, 'unobserved source cannot become valid E1 geometry')
            classification, reason = 'EXPECTED_SCHEMA_VALIDITY', 'current source validity is false'
        elif expected_kind == 'intrinsic':
            classification, reason = 'EXPECTED_SCHEMA_VALIDITY', 'intrinsic channel requires no global frame bridge'
        elif expected_kind == 'dimensions':
            if not quantity.valid:
                _fail('UNEXPECTED_E1_INVALIDATION', *key, 'valid intrinsic dimensions must remain valid')
            classification, reason = 'EXPECTED_SCHEMA_VALIDITY', 'intrinsic dimensions require no global frame bridge'
        elif frame not in _GLOBAL:
            if quantity.valid or quantity.values is not None:
                _fail('FRAME_SEMANTICS_UNKNOWN', *key, 'unframed/unowned body-local geometry cannot use a global fallback')
            classification, reason = 'EXPECTED_SCHEMA_VALIDITY', 'unframed or body-local task geometry lacks a certified origin/owner'
        else:
            resolutions = tuple(r for r in bridges.resolutions if r.source_frame == frame and r.target_frame == 'world')
            certified = frame == 'world' or any(r.certified for r in resolutions)
            if not certified:
                unresolved = tuple(r for r in resolutions if not r.certified and r.reason.strip())
                if not unresolved or quantity.valid or quantity.values is not None:
                    _fail('FRAME_SEMANTICS_UNKNOWN', *key,
                          'unresolved global frame needs an explicit bridge reason and invalid geometry; no identity fallback')
                classification, reason = 'EXPECTED_UNRESOLVED_FRAME', unresolved[0].reason
            elif not geometry.frame.valid and expected_kind in _NEEDS_CENTROID:
                if quantity.valid or quantity.values is not None:
                    _fail('FRAME_SEMANTICS_UNKNOWN', *key, 'position-dependent geometry requires an observed centroid')
                classification, reason = 'EXPECTED_SCHEMA_VALIDITY', 'no valid current module pose defines the physical centroid'
            elif quantity.valid is not True or quantity.values is None:
                _fail('UNEXPECTED_E1_INVALIDATION', *key, 'valid certified spatial source became invalid in E1')
        if quantity.valid and expected_kind != 'intrinsic':
            kind, width, _, parts = _environment_layout(quantity, source, schema)
            _physical_values(quantity.values, width*parts, 'pipeline.E1.geometry')
            if _e2_representation(quantity, source, features, schema):
                classification, reason = 'EXPECTED_DEDUP_OR_ALIAS', 'existing E2 physical equivalence retains valid representation'
        diagnostics.append(PipelineDiagnostic('post_e2', key[0], key[1], classification, reason))
    return tuple(diagnostics)
