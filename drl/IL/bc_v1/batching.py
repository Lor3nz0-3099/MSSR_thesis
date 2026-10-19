"""G2 variable concatenation; no normalization, sampler or model."""
import torch

from .tensorization import FAMILIES


def _cat(values):
    return torch.cat(values, dim=0)


def _shift(values, offset):
    return torch.where(values == -1, values, values+offset)


def _ptr(values):
    parts, offset = [torch.zeros(1,dtype=torch.int64)], 0
    for value in values:
        if value.dtype!=torch.int64 or value[0]!=0 or (value[1:]<value[:-1]).any():
            raise ValueError('Invalid local pointer')
        parts.append(value[1:]+offset); offset += int(value[-1])
    return _cat(parts)


def collate_policy_samples(samples):
    """Collate already tensorized samples; preserve whole joint supervision."""
    if not samples: raise ValueError('Cannot collate an empty batch')
    for sample in samples:
        for key in ('layout_fingerprint','normalizer_fingerprint'):
            if sample['routing'][key] != samples[0]['routing'][key]:
                raise ValueError('incompatible tensor samples')
    xs=[s['inputs'] for s in samples]
    result={}
    offsets=[]; current={k:0 for k in ('module','connector','resource','coordinate','connection','relation','environment','object','resource_candidate','structural','argument','domain','scope')}
    for x in xs:
        offsets.append(dict(current))
        for family in FAMILIES: current[family]+=len(x[family+'_x'])
        current['environment']+=int(x['environment_ptr'][-1]);current['object']+=int(x['object_ptr'][-1])
        current['resource_candidate']+=len(x['resource_candidates']['resource'])
        current['structural']+=len(x['structural']['initiator'])
        current['argument']+=len(x['structural']['argument_binding'])
        current['domain']+=len(x['structural']['argument_domain_index'])
        current['scope']+=len(x['structural']['scope_members'])
    for family in FAMILIES:
        result[family+'_x']=_cat([x[family+'_x'] for x in xs])
        result[family+'_validity']=_cat([x[family+'_validity'] for x in xs])
        result[family+'_ptr']=_ptr([torch.tensor([0,len(x[family+'_x'])],dtype=torch.int64) for x in xs])
    result['graph_ptr']=result.pop('module_ptr')
    result['module_batch']=_cat([torch.full((len(x['module_x']),),i,dtype=torch.int64) for i,x in enumerate(xs)])
    for name,target in [('connector_to_module','module'),('resource_to_module','module'),
                        ('coordinate_to_resource','resource'),('connection_connectors','connector'),('relation_to_module','module')]:
        result[name]=_cat([_shift(x[name],o[target]) for x,o in zip(xs,offsets)])
    result['coordinate_slot']=_cat([x['coordinate_slot'] for x in xs])
    for name in ('task_one_hot','task_valid'): result[name]=_cat([x[name] for x in xs])
    for family in ('environment','object'): result[family+'_ptr']=_ptr([x[family+'_ptr'] for x in xs])
    result['relation_geometry_ptr']=_ptr([x['relation_geometry_ptr'] for x in xs])
    result['relation_geometry_index']=_cat([x['relation_geometry_index']+o['environment'] for x,o in zip(xs,offsets)])
    environment={}
    for key in sorted({k for x in xs for k in x['environment']}):
        records=[(i,x['environment'][key],offsets[i]) for i,x in enumerate(xs) if key in x['environment']]
        environment[key]={name:_cat([row[name] for _,row,_ in records]) for name in ('x','validity')}
        for name,target in [('token_index','environment'),('object_index','object')]:
            environment[key][name]=_cat([row[name]+o[target] for _,row,o in records])
        environment[key]['graph_index']=_cat([torch.full((len(row['x']),),i,dtype=torch.int64) for i,row,_ in records])
    result['environment']=environment
    resources={}
    for key in xs[0]['resource_candidates']:
        values=[x['resource_candidates'][key] for x in xs]
        resources[key]=_ptr(values) if key.endswith('_ptr') else _cat([
            _shift(v,o['resource']) if key=='resource' else v for v,o in zip(values,offsets)])
    result['resource_candidates']=resources
    structural={}
    for key in xs[0]['structural']:
        values=[x['structural'][key] for x in xs]
        if key.endswith('_ptr'): structural[key]=_ptr(values);continue
        shifted=[]
        for v,x,o in zip(values,xs,offsets):
            target={'initiator':'module','peer':'module','connector_a':'connector','connector_b':'connector',
                    'scope_members':'module','argument_binding':'structural'}.get(key)
            if target: v=_shift(v,o[target])
            elif key in ('argument_bound_ref','argument_domain_index'):
                kind=x['structural']['argument_kind']
                if key=='argument_domain_index':
                    kind=torch.repeat_interleave(kind,x['structural']['argument_domain_ptr'][1:]-x['structural']['argument_domain_ptr'][:-1])
                shift=torch.where((kind==5)|(kind==7),o['connector'],o['module'])
                v=torch.where(v==-1,v,v+shift)
            shifted.append(v)
        structural[key]=_cat(shifted)
    result['structural']=structural
    supervision={}
    for family in ('resource','structural'):
        out={}
        for key in samples[0]['supervision'][family]:
            values=[s['supervision'][family][key] for s in samples]
            if key.endswith('_ptr'): out[key]=_ptr(values);continue
            shifted=[]
            for v,s,o in zip(values,samples,offsets):
                target=({'resource':'resource','candidate':'resource_candidate'} if family=='resource' else {
                    'candidate':'structural','group':'module','decision_candidate':'structural','decision_group':'module','argument_domain':'argument','membership_domain':'domain',
                    'scope_membership_index':'scope'}).get(key)
                if target:v=_shift(v,o[target])
                elif key=='reference_index':
                    y=s['supervision']['structural'];st=s['inputs']['structural']
                    kind=st['argument_kind'][y['argument_domain']]
                    kind=torch.repeat_interleave(kind,y['reference_ptr'][1:]-y['reference_ptr'][:-1])
                    v=v+torch.where((kind==5)|(kind==7),o['connector'],o['module'])
                shifted.append(v)
            out[key]=_cat(shifted)
        supervision[family]=out
    return {'inputs':result,'supervision':supervision,'routing':{'samples':tuple(s['routing'] for s in samples),
        'layout_fingerprint':samples[0]['routing']['layout_fingerprint'],
        'normalizer_fingerprint':samples[0]['routing']['normalizer_fingerprint']}}
