"""G2: immutable physical CachedSample -> F2 -> explicit tensor/routing sections.

No E1/E2 extraction, source IO, fitting, sampling, learned embeddings or model.
Opaque entity references live only in routing. Geometry names select closed
semantic layouts; object ordinals never become semantic channels.
"""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import re

import torch

from . import normalization as f2
from . import policy_inputs as d2
from .bc_v1_contract import CoordinateSpec
from .feature_schema import _resolved_fields
from .dataset import _representable
from .bc_v1_contract import ModuleRef, ConnectorRef

VERSION = 'mssr.bc_v1.tensor_sample.v1'
FAMILIES = {'module': 'modules', 'connector': 'connectors', 'resource': 'resources',
    'coordinate': 'coordinates', 'connection': 'connections', 'relation': 'relations'}
SCHEMA_FAMILIES = {'coordinate': 'actuator_coordinate', 'relation': 'module_environment_relation'}
PARAMETER_KINDS = ('scalar', 'vector', 'boolean', 'category', 'module_ref', 'connector_ref', 'module_refs', 'connector_refs')


def load_tensor_layout(path):
    path = Path(path)
    config = json.loads(path.read_text())
    if config.get('schema_version') != 'mssr.bc_v1.tensor_layout.v1':
        raise ValueError('Unsupported tensor layout')
    if (config.get('policy_input_contract') != d2.POLICY_INPUTS_SCHEMA_VERSION
            or config.get('packing') != 'all_values_then_matching_validity'
            or config.get('semantic_geometry') != 'approved_d2_fields_with_object_ordinals_removed'
            or config.get('dtypes') != {'features': 'float32', 'indices': 'int64', 'masks': 'bool'}):
        raise ValueError('Incompatible tensor layout semantics')
    # Resolve the configured ontology against the repository, not the caller cwd.
    root = Path(__file__).resolve().parents[3]
    task_config = json.loads((root/config['task_ontology']).read_text())
    vocabulary = tuple(task_config['task_vocabulary'])
    if vocabulary[0] != 'UNKNOWN' or len(set(vocabulary)) != len(vocabulary) or set(vocabulary[1:]) != d2._TASKS:
        raise ValueError('Task vocabulary differs from frozen D2')
    slots = set(d2._COMMON_STATIC) | set(d2._DYNAMIC)
    slots.update(k for fields in d2._TASK_STATIC.values() for k in fields)
    slots.update('obstacles.*.'+f for f in d2._OBSTACLE_FIELDS)
    slots.update(group+'.*.'+f for group in ('cones', 'environment_cones') for f in d2._CONE_FIELDS)
    # D2 separately admits legacy planar cone centers for navigation.
    # E1/E2 retain them as lines with unknown height, distinct from 3D centers.
    slots.add('cones.*.center_xy_m')
    resolved = {**config, 'task_vocabulary': vocabulary, 'semantic_slots': tuple(sorted(slots))}
    return {**resolved, 'fingerprint': hashlib.sha256(json.dumps(resolved, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()}


def semantic_geometry_slot(name, layout):
    first, _, remainder = name.partition('.')
    base = remainder if first in d2._FRAMES else name
    base = re.sub(r'^(obstacles|cones|environment_cones)\.[0-9]+\.', r'\1.*.', base)
    if base not in layout['semantic_slots']:
        raise ValueError('Unapproved semantic geometry slot')
    return base


def _int(values, shape=None):
    out = torch.tensor(values, dtype=torch.int64)
    return out.reshape(shape) if shape is not None else out


def _index(refs):
    result = {ref: i for i, ref in enumerate(refs)}
    if len(result) != len(refs): raise ValueError('Duplicate entity routing reference')
    return result


def _matrix(rows, width):
    if any(len(row.values) != width or len(row.validity) != width for row in rows):
        raise ValueError('Inconsistent feature width/validity')
    validity = torch.tensor([row.validity for row in rows], dtype=torch.bool).reshape(len(rows), width)
    values = torch.tensor([[v if valid else 0. for v, valid in zip(row.values, row.validity)]
        for row in rows], dtype=torch.float32).reshape(len(rows), width)
    if not torch.isfinite(values).all(): raise ValueError('Feature outside finite float32 range')
    return torch.cat((values, validity.float()), dim=1), validity


def _width(spec, schema, family):
    vocab = {'robot_spec.connector_capabilities': tuple(sorted({cap for t in spec.module_types
        for p in t.connectors for cap in p.capabilities})),
        'robot_spec.resource_capabilities': tuple(sorted({cap for t in spec.module_types
        for r in t.resources for cap in r.capabilities}))}
    coordinate = next((c for t in spec.module_types for r in t.resources for c in r.coordinates), CoordinateSpec('unused', '1'))
    return sum(f.width for f in _resolved_fields(schema, SCHEMA_FAMILIES.get(family, family), vocab, coordinate))


def _candidate_inputs(table, spec, indices, routing):
    """Catalog/group/domain routing, built before reading any expert targets."""
    actions = tuple(sorted(spec.actions, key=lambda a: a.action_id))
    action_index = {a.action_id:i for i,a in enumerate(actions)}
    parameter_catalog = tuple((a.action_id,p.name) for a in actions for p in sorted(a.parameters,key=lambda p:p.name))
    parameter_index = _index(parameter_catalog)
    resource_rows = sorted(table.resource_candidates,
        key=lambda r: indices['resource'][r.resource_ref])
    mode_catalog = tuple(sorted((t.type_id,r.resource_id,m.mode_id) for t in spec.module_types
        for r in t.resources for m in r.modes))
    mode_index = _index(mode_catalog)
    type_by_module = dict(zip(routing['module_refs'], routing['module_types']))
    rp, payload_ptr = [0], [0]
    for ref in routing['resource_refs']:
        rp.append(rp[-1]+sum(r.resource_ref==ref for r in resource_rows))
    for row in resource_rows: payload_ptr.append(payload_ptr[-1]+row.mode.payload_size)
    resources = {'group_ptr': _int(rp), 'resource': _int([indices['resource'][r.resource_ref] for r in resource_rows]),
        'mode': _int([mode_index[(type_by_module[r.resource_ref.module], r.resource_ref.resource_id, r.mode.mode_id)] for r in resource_rows]),
        'action': _int([action_index[r.action.action_id] for r in resource_rows]), 'payload_ptr': _int(payload_ptr)}
    bound_rows = [(row.action,binding) for row in table.structural_candidates for binding in row.bindings]
    bound_rows.sort(key=lambda ab:(indices['module'][ab[1].initiator], action_index[ab[0].action_id],
        tuple(indices['connector'][p] for p in ab[1].connectors)))
    groups = [0]
    for ref in routing['module_refs']: groups.append(groups[-1]+sum(b.initiator==ref for _,b in bound_rows))
    scope_members, scope_required, scope_ptr = [], [], [0]
    domains, arg_binding, arg_parameter, arg_kind = [], [], [], []
    domain_refs, required, domain_ptr, bound_ref = [], [], [0], []
    for i,(action,binding) in enumerate(bound_rows):
        scope = sorted(binding.scope_members,key=indices['module'].__getitem__)
        scope_members.extend(indices['module'][r] for r in scope)
        scope_required.extend(r in binding.required_scope_members for r in scope)
        scope_ptr.append(len(scope_members))
        for domain in binding.arguments:
            parameter = domain.parameter
            domains.append(domain); arg_binding.append(i)
            arg_parameter.append(parameter_index[(action.action_id,parameter.name)])
            arg_kind.append(PARAMETER_KINDS.index(parameter.kind))
            index = indices['connector'] if parameter.kind in ('connector_ref','connector_refs') else indices['module']
            refs = sorted(domain.options,key=index.__getitem__)
            domain_refs.extend(index[r] for r in refs)
            # Participant scope and parameter membership are separate domains.
            # Only an explicit argument binding establishes a required reference.
            required.extend(r==domain.bound_ref for r in refs)
            domain_ptr.append(len(domain_refs)); bound_ref.append(-1 if domain.bound_ref is None else index[domain.bound_ref])
    structural = {'group_ptr': _int(groups), 'initiator': _int([indices['module'][b.initiator] for _,b in bound_rows]),
        'action': _int([action_index[a.action_id] for a,_ in bound_rows]),
        'peer': _int([indices['module'][b.connectors[1].module] if len(b.connectors)>1 else -1 for _,b in bound_rows]),
        'connector_a': _int([indices['connector'][b.connectors[0]] if b.connectors else -1 for _,b in bound_rows]),
        'connector_b': _int([indices['connector'][b.connectors[1]] if len(b.connectors)>1 else -1 for _,b in bound_rows]),
        'scope_ptr': _int(scope_ptr), 'scope_members': _int(scope_members),
        'scope_required': torch.tensor(scope_required,dtype=torch.bool),
        'argument_binding': _int(arg_binding), 'argument_parameter': _int(arg_parameter),
        'argument_kind': _int(arg_kind), 'argument_bound_ref': _int(bound_ref),
        'argument_domain_ptr': _int(domain_ptr), 'argument_domain_index': _int(domain_refs),
        'argument_domain_required': torch.tensor(required,dtype=torch.bool)}
    routing.update(actions=actions, parameter_catalog=parameter_catalog, mode_catalog=mode_catalog,
        resource_candidates=tuple(resource_rows), structural_bindings=tuple(bound_rows),
        argument_domains=tuple(domains), parameter_kinds=PARAMETER_KINDS)
    return resources, structural


def _target_tensors(resource_targets, structural_targets, indices, routing, inputs):
    resource_lookup = _index(tuple((r.resource_ref,r.mode.mode_id) for r in routing['resource_candidates']))
    payload, ptr, valid, categories = [], [0], [], []
    for target in resource_targets:
        row = routing['resource_candidates'][resource_lookup[(target.resource,target.mode_id)]]
        if len(target.payload)!=row.mode.payload_size: raise ValueError('Target payload width mismatch')
        for value in target.payload:
            if row.mode.codec=='categorical':
                if value != int(value) or not 0 <= int(value) < len(row.mode.categories):
                    raise ValueError('Invalid categorical resource payload')
                payload.append(0.); categories.append(int(value))
            else: payload.append(value); categories.append(-1)
            valid.append(target.supervised)
        ptr.append(len(payload))
    resource = {'resource': _int([indices['resource'][t.resource] for t in resource_targets]),
        'candidate': _int([resource_lookup[(t.resource,t.mode_id)] for t in resource_targets]),
        'payload': torch.tensor(payload,dtype=torch.float32), 'payload_ptr': _int(ptr),
        'payload_category': _int(categories), 'payload_valid': torch.tensor(valid,dtype=torch.bool),
        'supervised': torch.tensor([t.supervised for t in resource_targets],dtype=torch.bool)}
    selected, groups, supervised = [], [], []
    arg_domain, arg_supervised, arg_value, arg_value_valid, arg_ptr, arg_category = [],[],[],[],[0],[]
    refs, ref_ptr, member_domain, member_value, member_valid = [],[0],[],[],[]
    scope_index, scope_value, scope_valid, scope_ptr = [], [], [], [0]
    st = inputs['structural']
    arg_lookup = {(int(b),routing['argument_domains'][i].parameter.name):i for i,b in enumerate(st['argument_binding'])}
    for target in structural_targets:
        matches = [i for i,(a,b) in enumerate(routing['structural_bindings']) if a.action_id==target.operation_id
            and b.initiator==target.initiator and b.connectors==target.connectors]
        if len(matches)!=1: raise ValueError('Structural target has no unique binding')
        candidate=matches[0]; selected.append(candidate); groups.append(indices['module'][target.initiator]); supervised.append(target.supervised)
        start,end=st['scope_ptr'][candidate:candidate+2].tolist()
        participants={indices['module'][r] for r in target.participants}
        for j in range(start,end):
            scope_index.append(j);scope_value.append(float(int(st['scope_members'][j]) in participants))
            scope_valid.append(target.supervised and not bool(st['scope_required'][j]))
        scope_ptr.append(len(scope_index))
        for p in target.parameters:
            u = arg_lookup[(candidate,p.name)]; domain = routing['argument_domains'][u]; declaration=domain.parameter
            enabled=target.supervised and declaration.learnable
            arg_domain.append(u); arg_supervised.append(enabled); category=-1
            values = (p.value,) if declaration.kind in ('scalar','boolean') else p.value if declaration.kind=='vector' else ()
            arg_value.extend(values); arg_value_valid.extend([enabled]*len(values)); arg_ptr.append(len(arg_value))
            if declaration.kind=='category': category=declaration.categories.index(p.value)
            arg_category.append(category)
            if declaration.kind in ('module_ref','module_refs','connector_ref','connector_refs'):
                index=indices['module'] if declaration.kind.startswith('module') else indices['connector']
                actual=(p.value,) if declaration.kind.endswith('_ref') else p.value
                refs.extend(index[r] for r in actual)
                if declaration.kind.endswith('_refs'):
                    start,end=st['argument_domain_ptr'][u:u+2].tolist()
                    actual_indices={index[r] for r in actual}
                    for j in range(start,end):
                        member_domain.append(j);member_value.append(float(int(st['argument_domain_index'][j]) in actual_indices))
                        member_valid.append(enabled and not bool(st['argument_domain_required'][j]))
            ref_ptr.append(len(refs))
    structural = {'candidate': _int(selected), 'group': _int(groups), 'supervised': torch.tensor(supervised,dtype=torch.bool),
        'scope_membership_index': _int(scope_index), 'scope_membership_ptr': _int(scope_ptr),
        'scope_membership_value': torch.tensor(scope_value,dtype=torch.float32),
        'scope_membership_valid': torch.tensor(scope_valid,dtype=torch.bool),
        'argument_domain': _int(arg_domain), 'argument_supervised': torch.tensor(arg_supervised,dtype=torch.bool),
        'argument_value': torch.tensor(arg_value,dtype=torch.float32), 'argument_value_valid': torch.tensor(arg_value_valid,dtype=torch.bool),
        'argument_value_ptr': _int(arg_ptr), 'argument_category': _int(arg_category),
        'reference_index': _int(refs), 'reference_ptr': _int(ref_ptr),
        'membership_domain': _int(member_domain), 'membership_value': torch.tensor(member_value,dtype=torch.float32),
        'membership_valid': torch.tensor(member_valid,dtype=torch.bool)}
    for tree in (resource,structural):
        if any(t.is_floating_point() and not torch.isfinite(t).all() for t in tree.values()):
            raise ValueError('Target outside finite float32 range')
    return {'resource': resource, 'structural': structural}


def tensorize_policy_sample(sample, *, spec, feature_schema, normalization_policy, normalizers, tensor_layout):
    """Inputs are label-independent. Supervision is normalized separately once."""
    d2._validate_selected_inputs(sample.inputs.policy)
    view = f2.normalize_features(sample.inputs.features, normalizers, feature_schema, normalization_policy, spec)
    routing = {'schema_version': VERSION, 'layout_fingerprint': tensor_layout['fingerprint'],
        'normalizer_fingerprint': f2.normalizer_fingerprint(normalizers), 'audit': sample.audit,
        'no_train_support': tuple(k for k, s in normalizers.scalers if s.no_train_support)}
    inputs = {}
    indices = {}
    for family, collection in FAMILIES.items():
        rows = getattr(view, collection)
        refs = tuple(row.ref for row in rows)
        routing[family+'_refs'] = refs
        indices[family] = _index(refs)
        inputs[family+'_x'], inputs[family+'_validity'] = _matrix(rows, _width(spec, feature_schema, family))
        routing[family+'_fields'] = rows[0].fields if rows else ()
    inputs['graph_ptr'] = _int([0, len(view.modules)])
    inputs['connector_to_module'] = _int([indices['module'][r.ref.module] for r in view.connectors])
    inputs['resource_to_module'] = _int([indices['module'][r.ref.module] for r in view.resources])
    inputs['coordinate_to_resource'] = _int([indices['resource'][r.ref[0]] for r in view.coordinates])
    types = {t.type_id: t for t in spec.module_types}
    module_types = {r.ref: r.catalog_ref for r in view.modules}
    inputs['coordinate_slot'] = _int([next(i for i,c in enumerate(sorted(next(r for r in
        types[module_types[row.ref[0].module]].resources if r.resource_id == row.ref[0].resource_id).coordinates,
        key=lambda c:c.coordinate_id))
        if c.coordinate_id == row.ref[1]) for row in view.coordinates])
    inputs['connection_connectors'] = _int([[indices['connector'][p] for p in row.ref] for row in view.connections], (-1, 2))
    task = sample.inputs.policy.context.task_id
    task_index = tensor_layout['task_vocabulary'].index(task if task is not None else 'UNKNOWN')
    inputs['task_one_hot'] = torch.zeros((1, len(tensor_layout['task_vocabulary'])), dtype=torch.float32)
    inputs['task_one_hot'][0, task_index] = 1.
    inputs['task_valid'] = torch.tensor([task is not None], dtype=torch.bool)
    environment, objects, buckets, layouts = {}, {}, defaultdict(list), {}
    for i, row in enumerate(view.environment):
        slot = semantic_geometry_slot(row.source_name, tensor_layout)
        width = row.fields[0].width
        key = row.source_scope+':'+slot+':'+row.geometry_kind+':'+str(width)
        buckets[key].append((i, row))
        object_key = (row.source_scope, row.object_ref)
        objects.setdefault(object_key, len(objects))
        layouts[key] = {'semantic_slot': slot, 'source_scope': row.source_scope,
            'geometry_kind': row.geometry_kind, 'fields': row.fields}
    for key, records in buckets.items():
        rows = [r for _,r in records]
        x, validity = _matrix(rows, len(rows[0].values))
        environment[key] = {'x': x, 'validity': validity, 'token_index': _int([i for i,_ in records]),
            'object_index': _int([objects[(r.source_scope,r.object_ref)] for r in rows]),
            'graph_index': torch.zeros(len(rows), dtype=torch.int64)}
    inputs['environment'] = environment
    inputs['environment_ptr'] = _int([0, len(view.environment)])
    inputs['object_ptr'] = _int([0, len(objects)])
    inputs['relation_to_module'] = _int([indices['module'][row.module_ref] for row in view.relations])
    env_index = _index(tuple(row.ref for row in view.environment))
    sources, ptr = [], [0]
    for row in view.relations:
        sources.extend(env_index[ref] for ref in row.geometry_sources); ptr.append(len(sources))
    inputs['relation_geometry_index'], inputs['relation_geometry_ptr'] = _int(sources), _int(ptr)
    routing.update(environment_refs=tuple(r.ref for r in view.environment), environment_layouts=layouts,
        object_refs=tuple(objects), task_vocabulary=tensor_layout['task_vocabulary'])
    routing['module_types'] = tuple(row.catalog_ref for row in view.modules)
    inputs['resource_candidates'], inputs['structural'] = _candidate_inputs(sample.inputs.candidates, spec, indices, routing)
    _representable(sample.supervision, sample.inputs.candidates)
    resource_targets = tuple(f2.normalize_target(t, normalizers, module_type_id=module_types[t.resource.module])
        for t in sample.supervision.resource_targets)
    structural_targets = tuple(f2.normalize_structural_target(t, normalizers) for t in sample.supervision.structural_targets)
    supervision = _target_tensors(resource_targets, structural_targets, indices, routing, inputs)
    routing.update(resource_targets=resource_targets, structural_targets=structural_targets,
        physical_supervision=sample.supervision)
    return {'inputs': inputs, 'supervision': supervision, 'routing': routing}
