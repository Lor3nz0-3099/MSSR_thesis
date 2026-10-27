"""F2 immutable, train-only normalization; no source IO, batching or backend.

The caller supplies certified E2 views in episode-keyed streams/factories and
an independently pinned F1 manifest. Only train streams are ever opened.
Explicit policy metadata selects components; IDs/masks/topology are untouched.
Exact streaming rational moments avoid cancellation and order-dependent floats.
No observations are buffered, clipped or used to invent physical bounds.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

from .bc_v1_contract import ResourceTarget, StructuralTarget, SupervisionBundle, robot_spec_fingerprint
from .feature_schema import FeatureView, feature_schema_fingerprint, _resolved_fields
from .split import serialize_split_manifest

VERSION = 'mssr.bc_v1.normalizers.v1'
POLICY_VERSION = 'mssr.bc_v1.normalization_policy.v1'
_FAMILIES = {'modules':'module', 'connectors':'connector', 'resources':'resource',
             'coordinates':'actuator_coordinate', 'connections':'connection',
             'environment':'environment', 'relations':'module_environment_relation'}
_PRESERVE_CODECS = {'one_hot','multi_hot','boolean','r6','quaternion_r6','sin_cos','sincos','categorical'}


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def feature_key(family, field, component, unit, *, geometry_kind='', module_type_id='', resource_id='', coordinate_id=''):
    """Opaque scaler lookup, never a learned feature or instance reference."""
    return _json(['feature',family,field,geometry_kind,module_type_id,resource_id,coordinate_id,component,unit])


def resource_target_key(module_type_id,resource_id,mode_id,component,unit):
    return _json(['resource_target',module_type_id,resource_id,mode_id,component,unit])


def argument_key(action_id,name,component,unit):
    return _json(['action_argument',action_id,name,component,unit])


@dataclass(frozen=True)
class Scaler:
    offset: float
    scale: float
    count: int
    strategy: str
    no_train_support: bool

    def __post_init__(self):
        if any(isinstance(x,bool) or not isinstance(x,Real) or not math.isfinite(x) for x in (self.offset,self.scale)) or self.scale<=0:
            raise ValueError('Scaler requires finite offset and positive finite scale')
        if type(self.count) is not int or self.count<0 or type(self.no_train_support) is not bool or self.no_train_support!=(self.count==0):
            raise ValueError('Invalid scaler support diagnostic')
        if self.strategy not in ('train_standard','physical_bounds'):
            raise ValueError('Unknown scaler strategy')
        if self.no_train_support and (self.offset!=0 or self.scale!=1):
            raise ValueError('No train support requires identity scaler')


@dataclass(frozen=True)
class ResourceBinding:
    module_type_id: str
    resource_id: str
    mode_id: str
    codec: str
    units: tuple[str,...]
    keys: tuple[str,...]


@dataclass(frozen=True)
class ArgumentBinding:
    action_id: str
    name: str
    kind: str
    size: int
    learnable: bool
    keys: tuple[str,...]


@dataclass(frozen=True)
class NormalizerStats:
    metadata_items: tuple[tuple[str,object],...]
    scalers: tuple[tuple[str,Scaler],...]
    resource_bindings: tuple[ResourceBinding,...] = ()
    argument_bindings: tuple[ArgumentBinding,...] = ()

    @property
    def metadata(self):
        # Detached JSON tree: callers cannot mutate the frozen artifact.
        return json.loads(_json(dict(self.metadata_items)))

    def get(self,key):
        return dict(self.scalers).get(key,Scaler(0.,1.,0,'train_standard',True))


@dataclass(frozen=True)
class NormalizationSample:
    episode_id: str
    transition_index: int
    features: FeatureView
    supervision: SupervisionBundle = SupervisionBundle()


def validate_normalization_policy(policy,schema):
    expected={'schema_version','feature_schema_fingerprint','fields','geometry','resource_payloads'}
    if not isinstance(policy,Mapping) or set(policy)!=expected or policy['schema_version']!=POLICY_VERSION:
        raise ValueError('Unsupported normalization policy schema/keys')
    if policy['feature_schema_fingerprint']!=feature_schema_fingerprint(schema):
        raise ValueError('Normalization policy incompatible with feature schema')
    if set(policy['fields'])!=set(schema['entities']):
        raise ValueError('Normalization policy requires all schema families')
    for family,entity in schema['entities'].items():
        rules=policy['fields'][family]
        if set(rules)!={f['name'] for f in entity['fields']}:
            raise ValueError('Normalization policy requires exact field names')
        for f in entity['fields']:
            rule=rules[f['name']]
            if rule not in ('train','preserve','coordinate_bounds','geometry'):
                raise ValueError('Unknown normalization rule')
            # Special vector semantics are declared in metadata, not ID/name heuristics.
            if (f['codec'] in _PRESERVE_CODECS or f['units']=='1' and f['codec'] in ('vector3','vector2')) and rule!='preserve':
                raise ValueError('Special codec/direction must be preserved')
            if (rule=='geometry') != (f['codec']=='geometry_kind'):
                raise ValueError('Geometry requires component policy')
            if rule=='coordinate_bounds' and f['source']!='coordinate.position':
                raise ValueError('Coordinate bounds apply only to coordinate position')
    if set(policy['geometry'])!=set(schema['geometry_kinds']):
        raise ValueError('Missing/unknown geometry policy')
    for kind,entry in schema['geometry_kinds'].items():
        rules=policy['geometry'][kind];width=entry['dimension']
        if width=='source_components' or entry['units']==['source_quantity_units']:
            if rules not in ('train','preserve'):
                raise ValueError('Intrinsic geometry requires a uniform component policy')
            continue
        if type(rules) is not list or len(rules)!=width or any(r not in ('train','preserve') for r in rules):
            raise ValueError('Invalid geometry component policy')
        if any(u=='1' and r!='preserve' for u,r in zip(entry['units'],rules)):
            raise ValueError('Geometry direction/basis components must be preserved')
    if type(policy['resource_payloads']) is not list:
        raise ValueError('Resource payload bindings must be a list')
    seen=set()
    for binding in policy['resource_payloads']:
        if not isinstance(binding,Mapping) or set(binding)!={'module_type_id','resource_id','mode_id','components'}:
            raise ValueError('Invalid resource payload binding keys')
        identity=tuple(binding[k] for k in ('module_type_id','resource_id','mode_id'))
        if any(type(x) is not str or not x for x in identity) or identity in seen:
            raise ValueError('Duplicate/invalid resource payload binding')
        seen.add(identity)
        if type(binding['components']) is not list or not binding['components']:
            raise ValueError('Missing resource payload components')
        for item in binding['components']:
            if not isinstance(item,Mapping) or set(item)!={'parameter','component'} or type(item['parameter']) is not str or type(item['component']) is not int or item['component']<0:
                raise ValueError('Invalid ordered payload parameter/component binding')
    _json(policy)


def load_normalization_policy(path,schema):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    validate_normalization_policy(value,schema)
    return value


def policy_fingerprint(policy,schema):
    validate_normalization_policy(policy,schema)
    value=dict(policy)
    value['resource_payloads']=sorted(policy['resource_payloads'],key=lambda b:(b['module_type_id'],b['resource_id'],b['mode_id']))
    return _hash(value)


class _Moments:
    """Constant-memory exact binary-float moments; no cancellation or reorder."""
    def __init__(self,bounds=None):
        self.n=0;self.total=Fraction();self.squares=Fraction();self.bounds=bounds

    def add(self,value):
        if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value):
            raise ValueError('Valid normalization value must be finite numeric')
        self.n+=1
        if self.bounds is None:
            x=Fraction(float(value));self.total+=x;self.squares+=x*x

    def scaler(self):
        if self.bounds is not None:
            lower,upper=map(Fraction,self.bounds)
            if lower>=upper:raise ValueError('Physical bounds require lower < upper')
            center=float((lower+upper)/2);scale=float((upper-lower)/2)
            Scaler(center,scale,1,'physical_bounds',False) # Validate even unsupported limits.
            return Scaler(center,scale,self.n,'physical_bounds',False) if self.n else Scaler(0.,1.,0,'physical_bounds',True)
        if not self.n:return Scaler(0.,1.,0,'train_standard',True)
        mean=self.total/self.n;variance=self.squares/self.n-mean*mean
        try:
            # sqrt in rational scale avoids overflow of a representable std's square.
            if variance:
                exponent=(variance.numerator.bit_length()-variance.denominator.bit_length())//2
                factor=Fraction(2)**exponent
                scale=math.ldexp(math.sqrt(float(variance/(factor*factor))),exponent)
            else:scale=1.
            return Scaler(float(mean),scale,self.n,'train_standard',False)
        except OverflowError as exc:
            raise ValueError('Normalization statistics exceed finite float range') from exc


def _catalog_plan(spec,schema,policy):
    robot_spec_fingerprint(spec);validate_normalization_policy(policy,schema)
    moments={};resources=[];arguments=[]
    for family,entity in schema['entities'].items():
        if family in ('actuator_coordinate','environment'):continue
        for f in entity['fields']:
            if policy['fields'][family][f['name']]=='train':
                for i in range(f['dimension']):
                    moments[feature_key(family,f['name'],i,f['units'])]=_Moments()
    for kind,entry in schema['geometry_kinds'].items():
        rules=policy['geometry'][kind]
        if isinstance(rules,list):
            for i,(rule,unit) in enumerate(zip(rules,entry['units'])):
                if rule=='train':moments[feature_key('environment','geometry',i,unit,geometry_kind=kind)]=_Moments()
    bindings={(b['module_type_id'],b['resource_id'],b['mode_id']):b for b in policy['resource_payloads']}
    used=set();actions={a.action_id:a for a in spec.actions}
    for mt in spec.module_types:
        for resource in mt.resources:
            for coordinate in resource.coordinates:
                for field in schema['entities']['actuator_coordinate']['fields']:
                    rule=policy['fields']['actuator_coordinate'][field['name']]
                    if rule=='preserve':continue
                    unit=coordinate.units+('/s' if field['units']=='coordinate_units_per_s' else '')
                    key=feature_key('actuator_coordinate',field['name'],0,unit,module_type_id=mt.type_id,resource_id=resource.resource_id,coordinate_id=coordinate.coordinate_id)
                    moments[key]=_Moments(coordinate.limits if rule=='coordinate_bounds' else None)
                    moments[key].scaler() # Validate bounds before observing any data.
            for mode in resource.modes:
                identity=(mt.type_id,resource.resource_id,mode.mode_id)
                if mode.codec=='categorical':
                    resources.append(ResourceBinding(*identity,mode.codec,mode.units,()))
                    continue
                binding=bindings.get(identity)
                if binding is None or len(binding['components'])!=mode.payload_size:
                    raise ValueError('Numeric resource mode needs explicit ordered parameter binding')
                used.add(identity);action=actions.get(mode.mode_id)
                if action is None or action.scope!='resource':raise ValueError('Unresolved resource action binding')
                params={p.name:p for p in action.parameters};keys=[];seen=set()
                for i,item in enumerate(binding['components']):
                    p=params.get(item['parameter']);j=item['component']
                    if p is None or p.kind not in ('scalar','vector') or not p.learnable or j>=(p.size or 1) or (p.name,j) in seen:
                        raise ValueError('Invalid learnable numeric payload binding')
                    if (p.units or '1')!=mode.units[i] or p.module_type_id not in (None,mt.type_id) or p.resource_id not in (None,resource.resource_id):
                        raise ValueError('Payload binding units/owner incompatible')
                    seen.add((p.name,j))
                    key=resource_target_key(*identity,i,mode.units[i]);keys.append(key)
                    moments[key]=_Moments(p.limits);moments[key].scaler()
                resources.append(ResourceBinding(*identity,mode.codec,mode.units,tuple(keys)))
    if used!=set(bindings):raise ValueError('Dangling resource payload binding')
    for action in spec.actions:
        if action.scope=='resource':continue
        for p in action.parameters:
            keys=()
            if p.learnable and p.kind in ('scalar','vector'):
                keys=tuple(argument_key(action.action_id,p.name,i,p.units or '1') for i in range(p.size or 1))
                for key in keys:
                    moments[key]=_Moments(p.limits);moments[key].scaler()
            arguments.append(ArgumentBinding(action.action_id,p.name,p.kind,p.size or 1,p.learnable,keys))
    return moments,tuple(sorted(resources,key=lambda r:(r.module_type_id,r.resource_id,r.mode_id))),tuple(sorted(arguments,key=lambda a:(a.action_id,a.name)))


def _feature_components(view,spec,schema,policy):
    if type(view) is not FeatureView or view.schema_version!=schema['schema_version'] or view.schema_fingerprint!=feature_schema_fingerprint(schema):
        raise ValueError('Feature view incompatible with schema')
    module_types={r.ref:r.catalog_ref for r in view.modules}
    types={t.type_id:t for t in spec.module_types}
    vocabularies={
        'robot_spec.connector_capabilities':tuple(sorted({cap for t in spec.module_types for p in t.connectors for cap in p.capabilities})),
        'robot_spec.resource_capabilities':tuple(sorted({cap for t in spec.module_types for r in t.resources for cap in r.capabilities})),
    }
    for collection,family in _FAMILIES.items():
        declarations={f['name']:f for f in schema['entities'][family]['fields']}
        for row in getattr(view,collection):
            if len(row.values)!=len(row.validity) or sum(f.width for f in row.fields)!=len(row.values) or any(type(v) is not bool for v in row.validity):
                raise ValueError('Invalid feature value/validity layout')
            if [f.name for f in row.fields]!=[f['name'] for f in schema['entities'][family]['fields']]:
                raise ValueError('Feature fields incompatible with schema')
            cursor=0
            for field in row.fields:
                if type(field.width) is not int or field.width<0 or len(field.units)!=field.width:
                    raise ValueError('Invalid feature component units/width')
                rule=policy['fields'][family][field.name];context={};bounds=None;coordinate=None
                if family=='actuator_coordinate':
                    resource_ref,coordinate_id=row.ref
                    mt_id=module_types.get(resource_ref.module);mt=types.get(mt_id)
                    if mt is None:raise ValueError('Coordinate module catalog absent')
                    resource=next((r for r in mt.resources if r.resource_id==resource_ref.resource_id),None)
                    coordinate=next((c for c in resource.coordinates if c.coordinate_id==coordinate_id),None) if resource else None
                    if coordinate is None:raise ValueError('Coordinate catalog absent')
                    context={'module_type_id':mt_id,'resource_id':resource.resource_id,'coordinate_id':coordinate_id}
                    if rule=='coordinate_bounds':bounds=coordinate.limits
                rules=[rule]*field.width
                if rule=='geometry':
                    rules=policy['geometry'].get(row.geometry_kind)
                    if rules is None:raise ValueError('Unknown geometry kind')
                    rules=[rules]*field.width if isinstance(rules,str) else rules
                    if len(rules)!=field.width:raise ValueError('Geometry component width incompatible')
                    context={'geometry_kind':row.geometry_kind}
                declaration=declarations[field.name]
                if family!='environment':
                    expected=next(f for f in _resolved_fields(schema,family,vocabularies,coordinate) if f.name==field.name)
                    if field!=expected:raise ValueError('Feature field layout units/frame/channels incompatible with schema')
                elif declaration['codec']!='geometry_kind':
                    channels=tuple(declaration['categories'])
                    if field.width!=len(channels) or field.units!=('1',)*len(channels) or field.channels!=channels or field.frame!=declaration['frame']:
                        raise ValueError('Environment categorical layout incompatible with schema')
                else:
                    entry=schema['geometry_kinds'][row.geometry_kind]
                    declared_units=entry['units']
                    if declared_units in ('source_quantity_units',['source_quantity_units']):
                        if not field.units or len(set(field.units))!=1 or field.units[0] not in ('m','1'):
                            raise ValueError('Intrinsic geometry units incompatible with schema')
                    elif field.units!=tuple(declared_units):
                        raise ValueError('Geometry units incompatible with schema')
                    if field.channels:raise ValueError('Geometric field must not encode categorical channels')
                if type(declaration['dimension']) is int and field.width!=declaration['dimension']:
                    raise ValueError('Feature component width incompatible with schema')
                for i,(strategy,unit) in enumerate(zip(rules,field.units)):
                    idx=cursor+i
                    if row.validity[idx] and (isinstance(row.values[idx],bool) or not isinstance(row.values[idx],Real) or not math.isfinite(row.values[idx])):
                        raise ValueError('Valid feature components must be finite numeric')
                    key=feature_key(family,field.name,i,unit,**context) if strategy!='preserve' else None
                    yield collection,row,idx,key,bounds
                cursor+=field.width


def _resource_components(target,bindings,module_type_id):
    if not target.supervised:return
    binding=next((b for b in bindings if (b.module_type_id,b.resource_id,b.mode_id)==(module_type_id,target.resource.resource_id,target.mode_id)),None)
    if binding is None:raise ValueError('Resource target catalog/mode absent')
    if binding.codec=='categorical':return
    if len(target.payload)!=len(binding.keys):raise ValueError('Numeric resource target payload width incompatible')
    for i,key in enumerate(binding.keys):yield i,key,target.payload[i]


def _argument_components(target,bindings):
    if not target.supervised:return
    domains={b.name:b for b in bindings if b.action_id==target.operation_id}
    for i,param in enumerate(target.parameters):
        binding=domains.get(param.name)
        if binding is None:raise ValueError('Structural argument catalog absent')
        if not binding.keys:continue
        values=(param.value,) if binding.kind=='scalar' else param.value
        if not isinstance(values,tuple) or len(values)!=binding.size:raise ValueError('Numeric action argument width incompatible')
        for j,key in enumerate(binding.keys):yield i,j,key,values[j],binding.kind


def fit_normalizers(episodes,split_manifest,spec,schema,policy,*,expected_split_fingerprint):
    """Fit only the verified F1 train membership, never open holdout streams.

    episodes maps IDs to iterables or zero-argument factories of E2 samples.
    Partial smoke fits are explicitly marked; G1 will supply full BC iterators.
    Caller reads F1 with its existing independent source/config/seed pins first.
    """
    serialize_split_manifest(split_manifest)
    if split_manifest['manifest_fingerprint']!=expected_split_fingerprint:
        raise ValueError('Frozen split fingerprint incompatible with expected split')
    train=tuple(sorted(split_manifest['splits']['train']))
    if not train or len(train)!=len(set(train)):raise ValueError('Invalid train membership')
    if not isinstance(episodes,Mapping) or any(eid not in episodes for eid in train):
        raise ValueError('Missing required train episode stream')
    moments,resource_bindings,argument_bindings=_catalog_plan(spec,schema,policy)
    counts=[]
    for eid in train:
        stream=episodes[eid];stream=stream() if callable(stream) else stream
        seen=set()
        for sample in stream:
            if type(sample) is not NormalizationSample or sample.episode_id!=eid:
                raise ValueError('Normalization sample episode identity mismatch')
            if type(sample.transition_index) is not int or sample.transition_index<0 or sample.transition_index in seen:
                raise ValueError('Invalid/duplicate sample transition index')
            seen.add(sample.transition_index)
            for _,row,i,key,bounds in _feature_components(sample.features,spec,schema,policy):
                if key is not None:
                    accumulator=moments.setdefault(key,_Moments(bounds))
                    if row.validity[i]:accumulator.add(row.values[i])
            module_types={r.ref:r.catalog_ref for r in sample.features.modules}
            for target in sample.supervision.resource_targets:
                mt=module_types.get(target.resource.module)
                for _,key,value in _resource_components(target,resource_bindings,mt):moments[key].add(value)
            for target in sample.supervision.structural_targets:
                for _,_,key,value,_ in _argument_components(target,argument_bindings):moments[key].add(value)
        counts.append((eid,len(seen)))
    source=split_manifest['dataset_identity']
    count_field=split_manifest['strata_config']['eligible_count_field']
    expected={eid:split_manifest['episode_metadata'][eid][count_field] for eid in train}
    if any(n>expected[eid] for eid,n in counts):raise ValueError('Observed train sample count exceeds frozen BC count')
    metadata={
        'split_fingerprint':expected_split_fingerprint,
        'dataset_manifest_sha256':source['manifest_sha256'],
        'dataset_identity_fingerprint':_hash(source),
        'train_episode_ids':train,'train_episode_count':len(train),
        'train_membership_fingerprint':_hash(train),
        'train_source_fingerprint':_hash([(eid,split_manifest['episode_metadata'][eid]['sha256']) for eid in train]),
        'robot_spec_fingerprint':robot_spec_fingerprint(spec),
        'feature_schema_fingerprint':feature_schema_fingerprint(schema),
        'policy_fingerprint':policy_fingerprint(policy,schema),
        'sample_counts':tuple(counts),'complete_train_coverage':all(n==expected[eid] for eid,n in counts),
        'unseen_train_component_policy':'identity_with_no_train_support',
        'moments':'exact_streaming_population_ddof_0',
    }
    return NormalizerStats(tuple(sorted(metadata.items())),tuple((k,m.scaler()) for k,m in sorted(moments.items())),resource_bindings,argument_bindings)


def normalize_features(features,stats,schema,policy,spec):
    """Return an immutable E2 view; only valid, authorized values change."""
    expected={'feature_schema_fingerprint':feature_schema_fingerprint(schema),
              'policy_fingerprint':policy_fingerprint(policy,schema),'robot_spec_fingerprint':robot_spec_fingerprint(spec)}
    if any(stats.metadata.get(k)!=v for k,v in expected.items()):
        raise ValueError('Normalizer incompatible with feature/schema/spec/policy')
    scalers=dict(stats.scalers);values={}
    for collection,row,i,key,bounds in _feature_components(features,spec,schema,policy):
        if key is not None and row.validity[i]:
            scaler=scalers.get(key,_Moments(bounds).scaler())
            values.setdefault(id(row),list(row.values))[i]=_affine(row.values[i],scaler)
    return replace(features,**{name:tuple(replace(row,values=tuple(values[id(row)])) if id(row) in values else row for row in getattr(features,name)) for name in _FAMILIES})


def normalize_target(target: ResourceTarget,stats: NormalizerStats,*,module_type_id: str):
    if not target.supervised:return target
    result=list(target.payload);changed=False
    for i,key,value in _resource_components(target,stats.resource_bindings,module_type_id):
        if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value):raise ValueError('Valid target must be finite numeric')
        scaler=stats.get(key);result[i]=_affine(value,scaler);changed=True
    return replace(target,payload=tuple(result)) if changed else target


def normalize_structural_target(target: StructuralTarget,stats: NormalizerStats):
    values={};kinds={}
    for i,j,key,value,kind in _argument_components(target,stats.argument_bindings):
        if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value):raise ValueError('Valid action argument must be finite numeric')
        if i not in values:values[i]=[target.parameters[i].value] if kind=='scalar' else list(target.parameters[i].value)
        scaler=stats.get(key);values[i][j]=_affine(value,scaler);kinds[i]=kind
    return replace(target,parameters=tuple(replace(p,value=values[i][0] if kinds[i]=='scalar' else tuple(values[i])) if i in values else p for i,p in enumerate(target.parameters))) if values else target


def _affine(value,scaler,*,inverse=False):
    """Fast float path, exact fallback for representable overflow cancellation."""
    if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value):
        raise ValueError('Affine value must be finite numeric')
    value=float(value)
    result=value*scaler.scale+scaler.offset if inverse else (value-scaler.offset)/scaler.scale
    if not math.isfinite(result):
        x,offset,scale=map(Fraction,(value,scaler.offset,scaler.scale))
        try:
            result=float(x*scale+offset if inverse else (x-offset)/scale)
        except OverflowError as exc:
            raise ValueError('Affine result exceeds finite float range') from exc
    return result


def denormalize_values(values,key,stats):
    """Restore physical units, retaining scalar/tuple/list/ndarray shape."""
    scaler=stats.get(key)
    if isinstance(values,tuple):return tuple(denormalize_values(x,key,stats) for x in values)
    if isinstance(values,list):return [denormalize_values(x,key,stats) for x in values]
    if isinstance(values,Real):return _affine(values,scaler,inverse=True)
    # Arrays are a caller choice, never a core import dependency.
    import numpy as np
    if not isinstance(values,np.ndarray) or values.dtype.kind not in 'fiu':
        raise TypeError('Expected numeric scalar, sequence or ndarray')
    array=np.asarray(values,dtype=float)
    if not np.all(np.isfinite(array)):raise ValueError('Affine values must be finite numeric')
    with np.errstate(over='ignore',invalid='ignore'):
        result=np.asarray(array*scaler.scale+scaler.offset)
    flat_result=result.reshape(-1);flat_array=array.reshape(-1)
    for index in np.flatnonzero(~np.isfinite(flat_result)):
        flat_result[index]=_affine(flat_array[index],scaler,inverse=True)
    return result


def _document(stats):
    return {'schema_version':VERSION,'metadata':stats.metadata,
            'scalers':[{'key':k,**asdict(v)} for k,v in stats.scalers],
            'resource_bindings':[asdict(b) for b in stats.resource_bindings],
            'argument_bindings':[asdict(b) for b in stats.argument_bindings]}


def normalizer_fingerprint(stats):
    return _hash(_document(stats))


def serialize_normalizers(stats):
    doc=_document(stats);doc['fingerprint']=normalizer_fingerprint(stats)
    return (_json(doc)+'\n').encode('utf-8')


def write_normalizers(stats,path,*,source_root):
    path=Path(path)
    if path.resolve().is_relative_to(Path(source_root).resolve()):raise ValueError('Normalizer artifact forbidden inside source dataset')
    blob=serialize_normalizers(stats);path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with path.open('xb') as stream:stream.write(blob)
    except FileExistsError:
        if path.read_bytes()!=blob:raise ValueError('Existing normalizer differs; overwrite forbidden')


def _freeze(value):
    return tuple(_freeze(v) for v in value) if isinstance(value,list) else value


def read_normalizers(path,*,expected_metadata):
    doc=json.loads(Path(path).read_text(encoding='utf-8'))
    if set(doc)!={'schema_version','metadata','scalers','resource_bindings','argument_bindings','fingerprint'} or doc['schema_version']!=VERSION:
        raise ValueError('Unsupported normalizer schema/version')
    if _hash({k:v for k,v in doc.items() if k!='fingerprint'})!=doc['fingerprint']:
        raise ValueError('Normalizer fingerprint mismatch')
    required={'split_fingerprint','dataset_manifest_sha256','train_membership_fingerprint','robot_spec_fingerprint','feature_schema_fingerprint','policy_fingerprint'}
    if not required<=set(expected_metadata):raise ValueError('Independent normalizer compatibility expectations required')
    if any(_json(doc['metadata'].get(k))!=_json(v) for k,v in expected_metadata.items()):raise ValueError('Normalizer incompatible with expected metadata')
    scalers=tuple((s['key'],Scaler(**{k:v for k,v in s.items() if k!='key'})) for s in doc['scalers'])
    resources=tuple(ResourceBinding(**{k:_freeze(v) for k,v in b.items()}) for b in doc['resource_bindings'])
    arguments=tuple(ArgumentBinding(**{k:_freeze(v) for k,v in b.items()}) for b in doc['argument_bindings'])
    stats=NormalizerStats(tuple((k,_freeze(v)) for k,v in sorted(doc['metadata'].items())),scalers,resources,arguments)
    if normalizer_fingerprint(stats)!=doc['fingerprint']:raise ValueError('Invalid normalizer canonical fingerprint')
    return stats
