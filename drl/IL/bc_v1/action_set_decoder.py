"""Decode membership, then validate physical claims without semantic repair.

No per-resource or initiator argmax, replacement, inactive sentinel or planner.
The proposed bundle is observable before a separate all-or-nothing safety guard.
"""
from collections import defaultdict
import math
import torch
from .action_set import ActionBundle,ActionInstance
from .action_decoder import _Arguments,_routes,_inverse,_finite_tree
from .normalization import normalizer_fingerprint
from .model import _owners


@torch.no_grad()
def decode_action_sets(prediction,batch,normalizers,*,configured_arguments=None,threshold=0.):
    if type(threshold) not in (float,int) or not math.isfinite(threshold):raise ValueError('Invalid presence threshold')
    if set(prediction)!={'presence_logits','candidate_graph','parameters'}:raise ValueError('Not an ActionSet prediction')
    if batch['routing']['normalizer_fingerprint']!=normalizer_fingerprint(normalizers):raise ValueError('Frozen F2 mismatch')
    _finite_tree(prediction)
    x=batch['inputs'];routes=batch['routing'].get('samples',(batch['routing'],))
    rq=len(x['resource_candidates']['resource']);sq=len(x['structural']['initiator'])
    logits=prediction['presence_logits'];graph=prediction['candidate_graph']
    owners=_owners(x['graph_ptr']);expected=torch.cat((owners[x['resource_to_module'][x['resource_candidates']['resource']]],owners[x['structural']['initiator']]))
    if (logits.shape!=(rq+sq,) or not logits.is_floating_point() or graph.dtype!=torch.int64
            or not torch.equal(graph,expected)):raise ValueError('ActionSet candidate routing mismatch')
    chosen=set(torch.nonzero(logits>=threshold).flatten().cpu().tolist())
    values=prediction['parameters'];resource=values['resource'];structural=values['structural']
    numeric=_routes(dict(resource,ptr=resource['payload_ptr']),'payload_candidate','payload')
    categorical=_routes(resource['categorical'],'candidate','logits')
    modules=tuple(m for r in routes for m in r['module_refs']);connectors=tuple(m for r in routes for m in r['connector_refs'])
    domains=tuple(d for r in routes for d in r['argument_domains'])
    args=_Arguments(structural,x['structural'],modules,connectors,domains,normalizers)
    bindings={(b.module_type_id,b.resource_id,b.mode_id):b for b in normalizers.resource_bindings}
    argument_rows=defaultdict(list)
    for i,binding in enumerate(x['structural']['argument_binding'].cpu().tolist()):argument_rows[binding].append(i)
    st=x['structural'];scope_ptr=st['scope_ptr'].cpu().tolist();members=st['scope_members'].cpu().tolist();required=st['scope_required'].cpu().tolist()
    results=[];ro=so=0;configured={} if configured_arguments is None else configured_arguments
    for route in routes:
        actions=[]
        for local,row in enumerate(route['resource_candidates']):
            q=ro+local
            if q not in chosen:continue
            if row.mode.codec=='categorical':
                scores=categorical[q];payload=(float(max(range(len(scores)),key=scores.__getitem__)),)
            else:
                tid=route['module_types'][route['module_refs'].index(row.resource_ref.module)]
                binding=bindings.get((tid,row.resource_ref.resource_id,row.mode.mode_id))
                if binding is None or binding.units!=row.mode.units or binding.codec!=row.mode.codec:raise ValueError('Missing F2 physical binding')
                payload=_inverse(numeric[q],binding.keys,normalizers)
            if len(payload)!=row.mode.payload_size:raise ValueError('Physical payload arity mismatch')
            limited=[p for p in row.action.parameters if p.limits is not None]
            if limited:
                numbers=[p for p in row.action.parameters if p.kind in ('scalar','vector')]
                if len(numbers)!=1 or (numbers[0].size or 1)!=len(payload):raise ValueError('Ambiguous payload limits')
                if any(not limited[0].limits[0]<=v<=limited[0].limits[1] for v in payload):raise ValueError('Physical payload outside limits')
            actions.append(ActionInstance(row.action.action_id,row.resource_ref.module,resource=row.resource_ref,payload=payload))
        for local,(action,bound) in enumerate(route['structural_bindings']):
            q=so+local
            if rq+q not in chosen:continue
            a,b=scope_ptr[q:q+2]
            positions=list(range(a,b)) if all(required[a:b]) else args.members(structural['scope'],q,a,b,required)
            participants=tuple(modules[members[i]] for i in positions)
            if bound.initiator not in participants or not set(participants)<=bound.scope_members:raise ValueError('Invalid participant scope')
            parameters=tuple(p for i in argument_rows[q] if (p:=args.parameter(i,action,configured)) is not None)
            actions.append(ActionInstance(action.action_id,bound.initiator,participants=participants,
                                          connectors=bound.connectors,parameters=parameters))
        results.append(ActionBundle(tuple(actions)));ro+=len(route['resource_candidates']);so+=len(route['structural_bindings'])
    if ro!=rq or so!=sq:raise ValueError('ActionSet routing length mismatch')
    return tuple(results)


def assert_executable(bundle,claim_resolver,availability=None):
    """Reject the proposal as a whole on conflict; never select substitutes.

    Backend claims/online availability are causal runtime arguments, not learned
    targets. The original proposed set remains available for metrics/diagnostics.
    """
    from .bc_v1_contract import ResourceRef,ConnectorRef
    reserved=set()
    for action in bundle.actions:
        claims=claim_resolver(action)
        if (not isinstance(claims,tuple) or not claims or len(set(claims))!=len(claims)
                or any(not isinstance(ref,(ResourceRef,ConnectorRef)) for ref in claims)
                or action.resource is not None and action.resource not in claims):raise ValueError('Invalid physical claims')
        if reserved.intersection(claims):raise ValueError('Conflicting physical resource claims')
        if availability is not None and not availability.free(claims):raise ValueError('Physical resource unavailable')
        reserved.update(claims)
    return bundle
