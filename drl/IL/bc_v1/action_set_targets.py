"""Positive-only deterministic view over frozen G2-v3, never a dataset rewrite.

Legacy dense absence/authority labels do not enter v4. Normalized values from
original G2 routing include fixed-zero targets discarded by v3 payload loss.
Physical original supervision retains exact zero/output semantics separately.
"""
from collections import defaultdict
import torch
from .action_set import ActionBundle,ActionInstance
from .tensorization import _target_tensors
from .batching import collate_policy_samples


def candidate_index(route):
    result=defaultdict(list)
    for i,row in enumerate(route['resource_candidates']):
        result[('resource',row.resource_ref,row.mode.mode_id)].append(i)
    offset=len(route['resource_candidates'])
    for i,(action,binding) in enumerate(route['structural_bindings']):
        result[('structural',action.action_id,binding.initiator,binding.connectors)].append(offset+i)
    return dict(result)


def positive_indices(route,index=None):
    lookup=candidate_index(route) if index is None else index;found=[]
    source=route['physical_supervision']
    for target in source.resource_targets:
        found.append(lookup.get(('resource',target.resource,target.mode_id),()))
    for target in source.structural_targets:
        found.append(lookup.get(('structural',target.operation_id,target.initiator,target.connectors),()))
    if any(len(matches)!=1 for matches in found):raise ValueError('Positive target has no unique candidate identity')
    result=tuple(matches[0] for matches in found)
    if len(set(result))!=len(result):raise ValueError('Duplicate target candidate identity')
    return result


def target_bundle(sample):
    route=sample['routing'];positive_indices(route);source=route['physical_supervision']
    by_resource={(r.resource_ref,r.mode.mode_id):r.action.action_id for r in route['resource_candidates']}
    actions=tuple(ActionInstance(by_resource[(t.resource,t.mode_id)],t.resource.module,resource=t.resource,payload=t.payload)
                  for t in source.resource_targets)
    actions+=tuple(ActionInstance(t.operation_id,t.initiator,participants=t.participants,
                                 connectors=t.connectors,parameters=t.parameters) for t in source.structural_targets)
    return ActionBundle(actions)


def target_view(sample):
    route=sample['routing'];positive_indices(route)
    source=route['physical_supervision']
    for normalized,physical,key in (
            (route['resource_targets'],source.resource_targets,lambda t:(t.resource,t.mode_id)),
            (route['structural_targets'],source.structural_targets,lambda t:(t.operation_id,t.initiator,t.connectors))):
        if (len(normalized)!=len(physical) or {key(t) for t in normalized}!={key(t) for t in physical}
                or any(not t.supervised for t in normalized)):
            raise ValueError('Normalized positive targets differ from physical positive source')
    indices={key:{ref:i for i,ref in enumerate(route[key+'_refs'])} for key in ('module','connector','resource')}
    # Restore all original normalized positives, including exact-stop/hold whose
    # regression rows were removed by the v3 loss contract. No inverse fitting.
    supervision=_target_tensors(route['resource_targets'],route['structural_targets'],indices,route,sample['inputs'])
    return {**sample,'supervision':supervision}


def collate_action_set_samples(samples):
    batch=collate_policy_samples([target_view(sample) for sample in samples])
    rq=len(batch['inputs']['resource_candidates']['resource']);sq=len(batch['inputs']['structural']['initiator'])
    positive=torch.cat((batch['supervision']['resource']['candidate'],rq+batch['supervision']['structural']['candidate']))
    if positive.unique().numel()!=positive.numel():raise ValueError('Duplicate positive action identity')
    labels=torch.zeros(rq+sq,dtype=torch.bool);labels[positive]=True
    batch['supervision']['action_set']={'presence':labels}
    return batch
