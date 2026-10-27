"""I1 losses over frozen G2 supervision and H2 raw predictions only.

Each family returns tensor sum/mean/support (support counts decisions, not
negatives or scalar components). Total is the weighted sum of family means.
For later global validation aggregate each family's sums and supports before
dividing. No fitting, sampling, model forward, audit, trainer or metrics here.
"""
from collections.abc import Mapping
from pathlib import Path
import json
import math
import torch
from torch.nn import functional as F

from .tensorization import PARAMETER_KINDS


def masked_statistics(values,valid):
    """Sum and support for decision-level global aggregation, with safe zero.

    Select before reducing: invalid/padded values need not be finite and cannot
    poison either the numerator or its gradient through multiplication by zero.
    """
    if values.ndim!=1 or not values.is_floating_point() or valid.dtype!=torch.bool or valid.shape!=values.shape:
        raise ValueError('Invalid masked loss vector')
    selected=values[valid]
    if not torch.isfinite(selected).all():raise ValueError('Nonfinite supervised loss')
    numerator=selected.sum();support=valid.sum()
    return {'sum':numerator,'support':support,'mean':numerator/support.clamp_min(1)}


def _statistics(items,anchor):
    values=torch.stack(items) if items else anchor.reshape(-1)[:0]
    return masked_statistics(values,torch.ones_like(values,dtype=torch.bool))


def _anchor(prediction):
    tensors=[]
    def collect(tree):
        for value in tree.values():
            if isinstance(value,dict):collect(value)
            elif value.is_floating_point():tensors.append(value)
    collect(prediction)
    # Empty slices avoid NaN*0 and connect zero-support backward to every raw
    # head output. Masked heads have exactly zero gradients, not fabricated labels.
    return sum(v.reshape(-1)[:0].sum() for v in tensors)


def _routes(indices):
    if indices.dtype!=torch.int64 or indices.ndim!=1:raise ValueError('Invalid prediction route')
    result={int(index):row for row,index in enumerate(indices.tolist())}
    if len(result)!=len(indices):raise ValueError('Duplicate prediction route')
    return result


def _bounds(ptr,row):
    if ptr.dtype!=torch.int64 or not 0<=row<len(ptr)-1:raise ValueError('Invalid group reference')
    start,end=ptr[row:row+2].tolist()
    if not 0<=start<=end:raise ValueError('Invalid group pointer')
    return start,end


def _ce(logits,target):
    if not 0<=target<len(logits):raise ValueError('Target outside categorical domain')
    if not torch.isfinite(logits).all():raise ValueError('Nonfinite supervised logits')
    return torch.logsumexp(logits,0)-logits[target]


def resource_mode_loss(prediction,batch):
    if 'outcome_logits' in prediction['resource']:
        from .resource_decisions import supervised_decisions
        return _statistics([_ce(logits,target) for _,logits,target in supervised_decisions(prediction,batch)],_anchor(prediction))
    if 'decision_group' in batch['supervision']['resource']:raise ValueError('Resource v3 targets require v3 heads')
    p=prediction['resource'];x=batch['inputs']['resource_candidates'];y=batch['supervision']['resource']
    routes=_routes(p['mode_candidate']);items=[];seen=set()
    for row,valid in enumerate(y['supervised'].tolist()):
        if not valid:continue
        resource=int(y['resource'][row]);candidate=int(y['candidate'][row]);start,end=_bounds(x['group_ptr'],resource)
        if not start<=candidate<end:raise ValueError('Resource target outside group')
        if resource in seen:raise ValueError('Duplicate supervised resource decision')
        seen.add(resource)
        if end-start==1:continue
        lo,hi=_bounds(p['mode_group_ptr'],resource)
        if candidate not in routes or not lo<=routes[candidate]<hi or hi-lo!=end-start:
            raise ValueError('Missing/misaligned mode prediction')
        items.append(_ce(p['mode_logits'][lo:hi],routes[candidate]-lo))
    return _statistics(items,_anchor(prediction))


def _numeric(predicted,target,valid):
    if predicted.shape!=target.shape or valid.shape!=target.shape or valid.dtype!=torch.bool:
        raise ValueError('Numeric target arity/mask mismatch')
    if target.requires_grad:raise ValueError('Targets must not require gradients')
    if not valid.any():return None
    predicted,target=predicted[valid],target[valid]
    if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():raise ValueError('Nonfinite supervised numeric value')
    return F.huber_loss(predicted,target,reduction='mean',delta=1.)


def resource_payload_loss(prediction,batch):
    if 'outcome_logits' in prediction['resource']:
        from .resource_decisions import validate_decision_targets
        validate_decision_targets(batch)
    p=prediction['resource'];y=batch['supervision']['resource'];x=batch['inputs']['resource_candidates']
    numeric=_routes(p['payload_candidate']);categorical=_routes(p['categorical']['candidate']);items=[];seen=set()
    for row,valid in enumerate(y['supervised'].tolist()):
        if not valid:continue
        resource=int(y['resource'][row]);candidate=int(y['candidate'][row]);start,end=_bounds(x['group_ptr'],resource)
        if not start<=candidate<end:raise ValueError('Resource payload outside group')
        if resource in seen:raise ValueError('Duplicate supervised resource payload')
        seen.add(resource);a,b=_bounds(y['payload_ptr'],row);mask=y['payload_valid'][a:b]
        if candidate in numeric:
            lo,hi=_bounds(p['payload_ptr'],numeric[candidate])
            loss=_numeric(p['payload'][lo:hi],y['payload'][a:b],mask)
        elif candidate in categorical:
            if b-a!=1:raise ValueError('Categorical resource payload must have one category')
            lo,hi=_bounds(p['categorical']['ptr'],categorical[candidate])
            loss=_ce(p['categorical']['logits'][lo:hi],int(y['payload_category'][a])) if mask.any() else None
        else:raise ValueError('Missing payload prediction')
        if loss is not None:items.append(loss)
    return _statistics(items,_anchor(prediction))


def structural_choice_loss(prediction,batch):
    """Unique-positive decision CE; G2 has no certified equivalence sets.

    Duplicate targets within one initiator group cannot be assumed equivalent.
    A separate future contract would be needed for marginal multi-positive CE.
    """
    if 'no_structural_logits' in prediction['structural']:
        from .structural_decisions import joint_decisions
        return _statistics([_ce(logits,target) for _,logits,target in joint_decisions(prediction,batch)],_anchor(prediction))
    if 'decision_group' in batch['supervision']['structural']:
        raise ValueError('v2 supervision requires NO_STRUCTURAL logits')
    p=prediction['structural'];y=batch['supervision']['structural'];items=[];seen=set()
    for row,valid in enumerate(y['supervised'].tolist()):
        if not valid:continue
        group=int(y['group'][row]);candidate=int(y['candidate'][row]);start,end=_bounds(p['group_ptr'],group)
        if not start<=candidate<end:raise ValueError('Structural target absent/outside initiator group')
        if group in seen:raise ValueError('Duplicate structural decision; no certified positive-set contract')
        seen.add(group);items.append(_ce(p['candidate_logits'][start:end],candidate-start))
    return _statistics(items,_anchor(prediction))


def _argument_rows(batch,kinds):
    x=batch['inputs']['structural'];y=batch['supervision']['structural'];seen=set()
    parents={int(q):bool(valid) for q,valid in zip(y['candidate'],y['supervised'])}
    for row,valid in enumerate(y['argument_supervised'].tolist()):
        if not valid:continue
        u=int(y['argument_domain'][row])
        if not 0<=u<len(x['argument_kind']):raise ValueError('Argument target outside current table')
        binding=int(x['argument_binding'][u])
        if binding not in parents:raise ValueError('Argument has no structural decision target')
        if not parents[binding]:continue
        if u in seen:raise ValueError('Duplicate supervised argument')
        seen.add(u)
        if PARAMETER_KINDS[int(x['argument_kind'][u])] in kinds:yield row,u


def _argument_prediction(records,u):
    routes=_routes(records['argument'])
    if u not in routes:raise ValueError('Missing learnable argument prediction')
    row=routes[u];start,end=_bounds(records['ptr'],row)
    return row,start,end


def category_argument_loss(prediction,batch):
    records=prediction['structural']['arguments']['category'];y=batch['supervision']['structural'];items=[]
    for row,u in _argument_rows(batch,('category',)):
        _,start,end=_argument_prediction(records,u)
        items.append(_ce(records['logits'][start:end],int(y['argument_category'][row])))
    return _statistics(items,_anchor(prediction))


def numeric_argument_loss(prediction,batch):
    records=prediction['structural']['arguments']['numeric'];y=batch['supervision']['structural'];items=[]
    for row,u in _argument_rows(batch,('scalar','vector')):
        _,start,end=_argument_prediction(records,u);a,b=_bounds(y['argument_value_ptr'],row)
        loss=_numeric(records['value'][start:end],y['argument_value'][a:b],y['argument_value_valid'][a:b])
        if loss is not None:items.append(loss)
    return _statistics(items,_anchor(prediction))


def reference_argument_loss(prediction,batch):
    records=prediction['structural']['arguments']['reference'];x=batch['inputs']['structural'];y=batch['supervision']['structural'];items=[]
    for row,u in _argument_rows(batch,('module_ref','connector_ref')):
        a,b=_bounds(y['reference_ptr'],row)
        if b-a!=1:raise ValueError('Single reference target must contain one reference')
        target=int(y['reference_index'][a]);bound=int(x['argument_bound_ref'][u])
        if bound!=-1:
            if target!=bound:raise ValueError('Target differs from fixed bound reference')
            continue
        _,start,end=_argument_prediction(records,u)
        options=x['argument_domain_index'][records['domain_index'][start:end]].tolist()
        if target not in options:raise ValueError('Reference target outside current allowed domain')
        items.append(_ce(records['logits'][start:end],options.index(target)))
    return _statistics(items,_anchor(prediction))


def _bce(logits,targets):
    if targets.requires_grad:raise ValueError('Targets must not require gradients')
    if (not torch.isfinite(logits).all() or not torch.isfinite(targets).all()
            or not ((targets==0)|(targets==1)).all()):raise ValueError('Invalid supervised membership/boolean value')
    return F.binary_cross_entropy_with_logits(logits,targets,reduction='mean')


def _membership(records,row,indices,values,valid,required,domain_range):
    """Average optional valid members of ONE decision, never domain-size weight."""
    if indices.shape!=values.shape or valid.shape!=values.shape or valid.dtype!=torch.bool:
        raise ValueError('Invalid membership target mask/shape')
    start,end=_bounds(records['ptr'],row);routes=_routes(records['domain_index'][start:end])
    selected=[];labels=[];seen=set();lo,hi=domain_range
    for i in torch.nonzero(valid,as_tuple=False).flatten().tolist():
        d=int(indices[i])
        if not lo<=d<hi:raise ValueError('Membership target outside decision domain')
        if d in seen:raise ValueError('Duplicate supervised membership')
        seen.add(d)
        if bool(required[d]):continue
        if d not in routes:raise ValueError('Missing optional membership prediction')
        selected.append(start+routes[d]);labels.append(i)
    if not selected:return None
    return _bce(records['logits'][selected],values[labels])


def _validate_membership_owners(batch):
    """Check ownership before filtering; globally valid is not enough.

    A valid membership belongs to a declared set-argument target. That target
    may itself be masked (or have a masked parent); ownership still exists and
    the normal argument mask intersection suppresses its loss afterwards.
    """
    x=batch['inputs']['structural'];y=batch['supervision']['structural']
    indices=y['membership_domain'][y['membership_valid']]
    if not len(indices):return
    if ((indices<0)|(indices>=len(x['argument_domain_index']))).any():
        raise ValueError('Membership outside current argument domains')
    owners=torch.searchsorted(x['argument_domain_ptr'][1:],indices,right=True)
    kinds=x['argument_kind'][owners]
    if ((kinds!=PARAMETER_KINDS.index('module_refs'))&(kinds!=PARAMETER_KINDS.index('connector_refs'))).any():
        raise ValueError('Membership target belongs to a non-set argument')
    declared=set(y['argument_domain'].tolist())
    if not set(owners.tolist())<=declared:raise ValueError('Membership has no corresponding set-argument target')


def reference_set_loss(prediction,batch):
    _validate_membership_owners(batch)
    records=prediction['structural']['arguments']['reference_set'];x=batch['inputs']['structural'];y=batch['supervision']['structural'];items=[]
    for row,u in _argument_rows(batch,('module_refs','connector_refs')):
        route,_,_=_argument_prediction(records,u);start,end=_bounds(x['argument_domain_ptr'],u)
        a,b=_bounds(y['reference_ptr'],row);options=set(x['argument_domain_index'][start:end].tolist())
        if not set(y['reference_index'][a:b].tolist())<=options:raise ValueError('Set target outside allowed reference domain')
        indices=y['membership_domain'];selected=(indices>=start)&(indices<end)
        loss=_membership(records,route,indices[selected],y['membership_value'][selected],y['membership_valid'][selected],
            x['argument_domain_required'],(start,end))
        if loss is not None:items.append(loss)
    return _statistics(items,_anchor(prediction))


def boolean_argument_loss(prediction,batch):
    records=prediction['structural']['arguments']['boolean'];routes=_routes(records['argument']);y=batch['supervision']['structural'];items=[]
    for row,u in _argument_rows(batch,('boolean',)):
        start,end=_bounds(y['argument_value_ptr'],row)
        if end-start!=1:raise ValueError('Boolean target must have one value')
        if not bool(y['argument_value_valid'][start]):continue
        if u not in routes:raise ValueError('Missing learnable boolean prediction')
        items.append(_bce(records['logits'][routes[u]:routes[u]+1],y['argument_value'][start:end]))
    return _statistics(items,_anchor(prediction))


def scope_loss(prediction,batch):
    records=prediction['structural']['scope'];routes=_routes(records['binding']);x=batch['inputs']['structural'];y=batch['supervision']['structural'];items=[]
    for row,valid in enumerate(y['supervised'].tolist()):
        if not valid:continue
        candidate=int(y['candidate'][row]);start,end=_bounds(x['scope_ptr'],candidate)
        if x['scope_required'][start:end].all():continue
        if candidate not in routes:raise ValueError('Missing variable participant-scope prediction')
        a,b=_bounds(y['scope_membership_ptr'],row)
        loss=_membership(records,routes[candidate],y['scope_membership_index'][a:b],y['scope_membership_value'][a:b],
            y['scope_membership_valid'][a:b],x['scope_required'],(start,end))
        if loss is not None:items.append(loss)
    return _statistics(items,_anchor(prediction))


LOSS_FUNCTIONS={
    'resource_mode':resource_mode_loss,'resource_payload':resource_payload_loss,
    'structural_choice':structural_choice_loss,'arg_category':category_argument_loss,
    'arg_numeric':numeric_argument_loss,'arg_reference':reference_argument_loss,
    'arg_reference_set':reference_set_loss,'arg_boolean':boolean_argument_loss,'scope':scope_loss}


def _validate_config(config):
    if (not isinstance(config,Mapping) or set(config)!={'schema_version','huber_delta','weights'}
            or config['schema_version']!='mssr.bc_v1.loss.v1'
            or type(config['huber_delta']) not in (int,float) or config['huber_delta']!=1.
            or not isinstance(config['weights'],Mapping) or set(config['weights'])!=set(LOSS_FUNCTIONS)
            or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in config['weights'].values())):
        raise ValueError('Unsupported I1 loss configuration')


def load_loss_config(path):
    config=json.loads(Path(path).read_text());_validate_config(config)
    return config


def _pointer_shape(ptr,rows,total):
    if len(ptr)!=rows+1 or int(ptr[0])!=0 or int(ptr[-1])!=total or (ptr[1:]<ptr[:-1]).any():
        raise ValueError('Invalid supervision pointer/count')


def _validate_supervision(batch):
    """Tensor ABI and integrity checks; never inspect audit or physical IDs."""
    masks={'supervised','decision_supervised','payload_valid','argument_supervised','argument_value_valid','membership_valid','scope_membership_valid'}
    floating={'payload','argument_value','membership_value','scope_membership_value'}
    for family in ('resource','structural'):
        for key,value in batch['supervision'][family].items():
            expected=torch.bool if key in masks else torch.float32 if key in floating else torch.int64
            if value.ndim!=1 or value.dtype!=expected or value.requires_grad:
                raise ValueError('Invalid supervision dtype/shape or target gradient')
    r=batch['supervision']['resource'];s=batch['supervision']['structural'];x=batch['inputs']['structural']
    groups=[(r,('resource','candidate','supervised')),(r,('payload','payload_category','payload_valid')),
        (s,('candidate','group','supervised')),(s,('argument_domain','argument_supervised','argument_category')),
        (s,('argument_value','argument_value_valid')),(s,('membership_domain','membership_value','membership_valid')),
        (s,('scope_membership_index','scope_membership_value','scope_membership_valid'))]
    if any(len({len(tree[key]) for key in keys})!=1 for tree,keys in groups):raise ValueError('Supervision row count mismatch')
    _pointer_shape(r['payload_ptr'],len(r['candidate']),len(r['payload']))
    _pointer_shape(s['argument_value_ptr'],len(s['argument_domain']),len(s['argument_value']))
    _pointer_shape(s['reference_ptr'],len(s['argument_domain']),len(s['reference_index']))
    _pointer_shape(s['scope_membership_ptr'],len(s['candidate']),len(s['scope_membership_index']))
    for indices,mask,limit in ((s['membership_domain'],s['membership_valid'],len(x['argument_domain_index'])),
            (s['scope_membership_index'],s['scope_membership_valid'],len(x['scope_members']))):
        if ((indices[mask]<0)|(indices[mask]>=limit)).any():raise ValueError('Supervised membership outside current domain')


def compute_bc_loss(prediction,batch,loss_config):
    """Raw H2 + G2 labels -> scalar total, per-family tensor statistics.

    Mean each payload/vector/set/scope within its valid components first, then
    mean supervised decisions within a family. Coefficients do not renormalize
    with active-family count. Device/dtype follow H2. All target tensors must be
    nondifferentiable; only supervision masks and current routing select labels.
    """
    _validate_config(loss_config);_validate_supervision(batch)
    total=_anchor(prediction);stats={}
    for family,function in LOSS_FUNCTIONS.items():
        row=function(prediction,batch);row['weighted']=loss_config['weights'][family]*row['mean']
        total=total+row['weighted'];stats[family]=row
    return total,stats
