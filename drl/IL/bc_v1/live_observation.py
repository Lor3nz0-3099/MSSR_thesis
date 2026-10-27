"""L1b current physical snapshot -> unchanged D2/E1/E2/F2/G2 inputs.

No source files, historical catalogs, targets, clock, scheduler or simulator
are read here. A runtime caller supplies the already-acquired atomic payload,
current typed task geometry, and any certified named-frame bridges. The caller
must acquire a fresh snapshot at a future decision point; this module cannot
prove freshness or choose when a policy is queried.

encode_smores_snapshot is the SMORES publication adapter. Its GraphBuilder
dependency is injected and its compact projection is the existing offline
function. The typed preparation path has no ROS/Isaac/SMORES-package imports.
Only navigation is certified in L1b. Historical Canonical role/root decoration
is intentionally not reconstructed: BC D2 discards those fields, while their
underlying physical state is preserved by compact_robot_graph.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

from .bc_v1_contract import (
    ObservationBundle, PolicyInputs, SampleAudit, StateGraph, SupervisionBundle,
    TaskContext, validate_robot_spec,
)
from .batching import collate_policy_samples
from .candidates import enumerate_candidates
from .dataset import CachedInputs, CachedSample
from .feature_schema import extract_features
from .features import FrameGeometryView, quaternion_to_matrix, transform_inputs
from .pipeline_validation import validate_physical_pipeline
from .policy_inputs import select_policy_inputs
from .source_frame_bridges import PipelineDiagnostic, SourceFrameBridgeBatch, frame_config_for
from .tensorization import tensorize_policy_sample


def _mapping(value, name):
    if not isinstance(value, Mapping): raise ValueError(f'Missing/invalid {name} mapping')
    return value


def _sequence(value, name):
    if not isinstance(value, (list, tuple)): raise ValueError(f'Missing/invalid {name} sequence')
    return value


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _vector(value, size, name):
    values = _sequence(value, name)
    if len(values) != size: raise ValueError(f'Invalid {name} dimension')
    return tuple(_number(v, name) for v in values)


def _identifier(value, name):
    if type(value) is not str or not value or any(c.isspace() for c in value):
        raise ValueError(f'Missing/invalid {name}')
    return value


@dataclass(frozen=True)
class LivePhysicalState:
    """Current WORLD physical state; stamp is audit-only, never an input column."""
    stamp: float
    graph: StateGraph

    def __post_init__(self):
        _number(self.stamp, 'physical stamp')
        if type(self.graph) is not StateGraph: raise TypeError('Expected typed current StateGraph')


@dataclass(frozen=True)
class LiveObservation:
    state: LivePhysicalState
    inputs: CachedInputs
    geometry: FrameGeometryView
    bridges: SourceFrameBridgeBatch
    diagnostics: tuple[PipelineDiagnostic, ...]
    batch: Mapping


def _module(raw, module_type):
    """Validate BEFORE GraphBuilder's legacy missing-value defaults can apply."""
    raw = _mapping(raw, 'module'); mid = _identifier(raw.get('module_id'), 'module ID')
    pose = _mapping(raw.get('pose'), 'current pose')
    if pose.get('frame_id') != 'world': raise ValueError('SMORES publisher pose must explicitly declare world')
    position = _vector(pose.get('position'), 3, 'pose position')
    orientation = _vector(pose.get('orientation_xyzw', pose.get('orientation')), 4, 'pose quaternion XYZW')
    quaternion_to_matrix(orientation)  # Existing validation; no new orientation codec.
    for source, key, expected in ((raw, 'position', position), (raw, 'orientation', orientation),
                                 (pose, 'orientation', orientation), (pose, 'orientation_xyzw', orientation)):
        if key in source and _vector(source[key], len(expected), key) != expected:
            raise ValueError('Conflicting current pose aliases')
    linear = _vector(raw.get('linear_velocity'), 3, 'linear velocity')
    angular = _vector(raw.get('angular_velocity'), 3, 'angular velocity')
    if 'twist' in raw:
        twist = _mapping(raw['twist'], 'current twist')
        if (_vector(twist.get('linear'), 3, 'linear twist') != linear
                or _vector(twist.get('angular'), 3, 'angular twist') != angular):
            raise ValueError('Conflicting current velocity aliases')
    actuators = _mapping(raw.get('actuators'), 'actuators'); selected = {}
    for resource in module_type.resources:
        for coordinate in resource.coordinates:
            if coordinate.units != 'rad': raise ValueError('SMORES publisher actuator contract requires radian coordinates')
            observed = _mapping(actuators.get(coordinate.coordinate_id), 'actuator '+coordinate.coordinate_id)
            selected[coordinate.coordinate_id] = {
                key: _number(observed.get(key), 'actuator '+key) for key in ('position_rad', 'velocity_rad_s')}
    ports = _sequence(raw.get('connectors'), 'connectors'); declared = {p.connector_id for p in module_type.connectors}
    connectors = []; seen = set()
    for port in ports:
        port = _mapping(port, 'connector'); name = _identifier(port.get('connector_id'), 'connector ID')
        if name not in declared or name in seen: raise ValueError('Unknown/duplicate current connector')
        seen.add(name)
        if 'face_name' in port and port['face_name'] != name: raise ValueError('Conflicting connector aliases')
        status = port.get('status')
        if status not in ('available', 'connected', 'in_contact', 'unknown'): raise ValueError('Invalid current connector status')
        result = {'connector_id':name, 'face_name':name, 'status':status}
        for key in ('position_world', 'outward_normal_world', 'tangent_world'):
            result[key] = list(_vector(port.get(key), 3, 'connector '+key))
        for key in ('connected_module_id', 'connected_connector_id'):
            value = port.get(key)
            result[key] = None if value is None else _identifier(value, key)
        if status == 'connected' and any(result[k] is None for k in ('connected_module_id', 'connected_connector_id')):
            raise ValueError('Connected connector is missing its current peer')
        connectors.append(result)
    if seen != declared: raise ValueError('Missing required current connector geometry')
    return {'module_id':mid, 'pose':{'frame_id':'world', 'position':list(position), 'orientation_xyzw':list(orientation)},
        'position':list(position), 'orientation':list(orientation), 'linear_velocity':list(linear),
        'angular_velocity':list(angular), 'actuators':selected, 'connectors':connectors}


def _relations(raw, known_modules, port_ids, *, forced=None):
    result = []
    for relation in _sequence(raw, 'current physical relations'):
        r = _mapping(relation, 'physical relation')
        if r.get('is_target_edge') or r.get('is_assignment') or r.get('relation_type') in ('target_connection', 'assignment'):
            raise ValueError('Future/target topology is not current physical connectivity')
        a, b = (_identifier(r.get(key), key) for key in ('module_a_id', 'module_b_id'))
        if a not in known_modules or b not in known_modules or a == b:
            raise ValueError('Physical edge references invalid current modules')
        fa, fb = r.get('connector_a_id', r.get('face_a')), r.get('connector_b_id', r.get('face_b'))
        if fa not in port_ids or fb not in port_ids: raise ValueError('Physical edge references unknown connector')
        for key, value in (('face_a',fa), ('face_b',fb)):
            if key in r and r[key] != value: raise ValueError('Conflicting physical endpoint aliases')
        for key in ('is_attached', 'is_contact'):
            if key in r and type(r[key]) is not bool: raise ValueError('Physical relation flags must be booleans')
        current = r.get('is_attached') is True or r.get('relation_type') == 'current_connection' or r.get('edge_type') == 'docking_connection'
        kind = forced or ('current_connection' if current else 'contact')
        if forced == 'contact' and current: raise ValueError('Contact is marked as attached')
        if r.get('relation_type') not in (None, kind): raise ValueError('Unknown/contradictory current relation')
        row = {'module_a_id':a, 'module_b_id':b, 'connector_a_id':fa, 'connector_b_id':fb,
            'face_a':fa, 'face_b':fb, 'relation_type':kind,
            'is_attached':kind == 'current_connection', 'is_contact':True}
        if r.get('clocking_quarter_turns') is not None:
            clocking = _number(r['clocking_quarter_turns'], 'clocking quarter turns')
            if clocking not in (0.,1.,2.,3.): raise ValueError('Invalid measured clocking')
            row['clocking_quarter_turns'] = clocking
        if r.get('connection_id') is not None: row['connection_id'] = _identifier(r['connection_id'], 'connection ID')
        result.append(row)
    return result


def _pair(relation):
    return frozenset(((relation['module_a_id'],relation['face_a']), (relation['module_b_id'],relation['face_b'])))


def encode_smores_snapshot(snapshot, *, adapter, graph_builder) -> LivePhysicalState:
    """Decode one combined mssr.state_graph.v2 payload, entirely without I/O.

    graph_builder is the existing pure runtime GraphBuilder, injected so that
    importing the BC preparation core does not require a runtime installation.
    Both embedded publications must describe the SAME stamp and physical data.
    Extra expert/role/course fields are never forwarded to the physical builder.
    """
    raw = _mapping(snapshot, 'combined snapshot')
    if raw.get('schema_version') != 'mssr.state_graph.v2': raise ValueError('Unsupported live snapshot schema')
    state, graph = _mapping(raw.get('state'), 'state snapshot'), _mapping(raw.get('graph'), 'graph snapshot')
    if state.get('schema_version') != 'mssr.module_states.v2' or graph.get('schema_version') != 'mssr.robot_graph.v2':
        raise ValueError('Unsupported embedded physical publication schema')
    if state.get('robot_family') != 'smores_ep': raise ValueError('Wrong physical source backend')
    stamp = _number(raw.get('stamp'), 'snapshot stamp')
    if _number(state.get('stamp'), 'state stamp') != stamp or _number(graph.get('stamp'), 'graph stamp') != stamp:
        raise ValueError('Incoherent physical snapshot stamps')
    module_type = adapter.module_type
    modules = [_module(m, module_type) for m in _sequence(state.get('modules'), 'current modules')]
    ids = [m['module_id'] for m in modules]
    if not ids or len(ids) != len(set(ids)): raise ValueError('Missing/duplicate current modules')
    by_id = {m['module_id']:m for m in modules}; nodes = []
    for node in _sequence(graph.get('nodes'), 'current graph nodes'):
        node = _mapping(node, 'graph node'); mid = _identifier(node.get('module_id'), 'graph module ID')
        decoded = _module(node.get('attributes'), module_type)
        if mid != decoded['module_id'] or by_id.get(mid) != decoded:
            raise ValueError('State and graph physical module data differ')
        nodes.append({'module_id':mid, 'attributes':decoded})
    if len(nodes) != len(ids) or {node['module_id'] for node in nodes} != set(ids):
        raise ValueError('State and graph module inventories differ')
    port_ids = {p.connector_id for p in module_type.connectors}
    edges = _relations(graph.get('edges'), set(ids), port_ids)
    attachments = _relations(state.get('attachments'), set(ids), port_ids, forced='current_connection')
    contacts = _relations(state.get('contacts'), set(ids), port_ids, forced='contact')
    graph_attached = {_pair(r) for r in edges if r['is_attached']}
    if graph_attached != {_pair(r) for r in attachments}: raise ValueError('Incoherent physical attachment inventories')
    for module in modules:
        for port in module['connectors']:
            if port['status'] == 'connected':
                pair = frozenset(((module['module_id'],port['connector_id']),
                    (port['connected_module_id'],port['connected_connector_id'])))
                if pair not in graph_attached: raise ValueError('Connected port has no current physical edge')
    observation = {'stamp':stamp, 'modules':modules, 'attachments':attachments, 'contacts':contacts}
    published = {'stamp':stamp, 'nodes':nodes, 'edges':edges}
    attributed = graph_builder.build(observation, published).to_dict()
    # Exact frozen projection, not a new relative-pose or graph algorithm.
    from drl.IL.preprocessing import compact_robot_graph
    encoded = adapter.encode_state(compact_robot_graph(attributed))
    return LivePhysicalState(stamp, encoded)


def _require_current_measurements(graph, spec):
    """The typed entry point has the same completeness gate as publication IO."""
    types = {t.type_id:t for t in spec.module_types}
    if not graph.modules: raise ValueError('Missing required current modules')
    resources = {r.ref:r for r in graph.resources}
    connectors = {p.ref:p for p in graph.connectors}
    for module in graph.modules:
        if (not module.pose_valid or module.linear_velocity is None or module.angular_velocity is None):
            raise ValueError('Missing required current module pose/twist')
        module_type = types.get(module.module_type_id)
        if module_type is None: raise ValueError('Unknown required current module type')
        for declaration in module_type.resources:
            matches = [r for ref,r in resources.items() if ref.module == module.ref and ref.resource_id == declaration.resource_id]
            if len(matches) != 1: raise ValueError('Missing required current resource')
            coordinates = {co.coordinate_id:co for co in matches[0].coordinates}
            for co in declaration.coordinates:
                observed = coordinates.get(co.coordinate_id)
                if observed is None or observed.position is None or observed.velocity is None:
                    raise ValueError('Missing required current coordinate observation')
        for declaration in module_type.connectors:
            matches = [p for ref,p in connectors.items() if ref.module == module.ref and ref.connector_id == declaration.connector_id]
            if len(matches) != 1: raise ValueError('Missing required current connector')
            fields = {f.name:f for f in matches[0].features}
            if any(name not in fields or not fields[name].valid for name in ('position_body_m','normal_body','tangent_body')):
                raise ValueError('Missing required current body-local connector geometry')


def prepare_live_observation(state: LivePhysicalState, *, context: TaskContext,
        observation: ObservationBundle, bridges: SourceFrameBridgeBatch,
        spec, feature_schema, normalization_policy, normalizers, tensor_layout) -> LiveObservation:
    """One pure current-state derivation; F2 runs once inside frozen G2.

    Current task/geometry must already be scoped and declared by the caller.
    No goal is inferred from route/waypoints or robot configuration. Optional
    absent geometry remains absent, as in training. Missing required robot
    measurements are rejected by the snapshot adapter before legacy defaults.
    Historical topology repair/source readers do not apply to an atomic current
    publication; attachment inventories above must agree instead.
    """
    if type(state) is not LivePhysicalState: raise TypeError('Expected current LivePhysicalState')
    validate_robot_spec(spec)
    _require_current_measurements(state.graph, spec)
    if type(context) is not TaskContext or context.task_id != 'navigation':
        raise ValueError('L1b certifies only an explicit current navigation task')
    if type(observation) is not ObservationBundle or observation.features:
        raise ValueError('No navigation-specific dynamic environment channels are certified in L1b')
    # Strict constructor rejects unknown context/observation names; do not
    # silently drop a caller's requested stage/future/expert feature.
    strict = PolicyInputs(state.graph, observation, context)
    inputs = select_policy_inputs(strict.graph, strict.observation, strict.context)
    geometry = transform_inputs(inputs, frame_config_for(bridges))
    for source, derived in zip((*inputs.context.static_geometry, *inputs.observation.features), geometry.quantities):
        if source.valid and not derived.valid: raise ValueError('Uncertified or missing physical geometry frame')
    features = extract_features(inputs, geometry, spec, feature_schema)
    diagnostics = validate_physical_pipeline(inputs, geometry, features, bridges, feature_schema)
    current = CachedInputs(inputs, features, enumerate_candidates(inputs.graph, spec, feature_schema))
    # Empty supervision is an ABI placeholder, not synthesized teacher data.
    sample = CachedSample(current, SupervisionBundle(), SampleAudit('live-current', 0))
    tensor = tensorize_policy_sample(sample, spec=spec, feature_schema=feature_schema,
        normalization_policy=normalization_policy, normalizers=normalizers, tensor_layout=tensor_layout)
    return LiveObservation(state, current, geometry, bridges, diagnostics, collate_policy_samples([tensor]))
