"""BC-v5 H2: one *binary presence logit* per candidate, evidence-aware masks.

Unselected candidates are negatives EXCEPT when Canonical explicitly carries an
unlabelled command on that module/resource.  This is not a third class: logits
remain one-dimensional, with ordinary BCE and zero gradient on suspect rows.
No BC-v4 module is modified; this implementation is explicit opt-in.
"""
from collections import Counter

import torch
from torch.nn import functional as F

from .action_set_loss import compute_action_set_loss, _candidate_graph
from .action_set_targets import collate_action_set_samples, positive_indices

_RESOURCE_IDS = {'locomotion': 'wheels', 'internal_motion': 'internal'}


def index_overlay(report, *, split):
    if not report.get('ready_for_training') or split not in ('train','validation'):
        raise ValueError('BC-v5 requires a complete authenticated TRAIN/VAL Canonical overlay')
    found = {}
    for row in report['rows']:
        if row['split'] != split:
            continue
        key = row['episode_id'], row['transition_index']
        if key in found:
            raise ValueError('Duplicate BC-v5 overlay key')
        kind = row.get('temporal_kind')
        if report.get('temporal_audit_provided') is False:
            # No external classifier: exclusions still come from Canonical v8 flags.
            if kind is not None:
                raise ValueError('Unexpected external temporal category in native-only overlay')
        elif kind not in ('single_decision','joint_single_point',
                          'temporally_coalesced_sequential','ambiguous'):
            raise ValueError('Missing/invalid optional temporal decision')
        found[key] = row
    if len(found) != report['expected_rows'][split]:
        raise ValueError('Incomplete split overlay')
    return found


def candidate_validity(route, overlay_entry):
    """True for supervised membership candidates in a safe graph."""
    if overlay_entry['exclude']:
        raise ValueError('Temporally unsafe graph must be excluded before batching')
    audit = route['audit']
    if (audit.episode_id, audit.transition_index) != (overlay_entry['episode_id'],
                                                       overlay_entry['transition_index']):
        raise ValueError('Overlay row does not match physical sample')
    blocked = set()
    for pair in overlay_entry['blocked']:
        if (not isinstance(pair, list) or len(pair) != 2
                or any(not isinstance(p, str) or not p for p in pair)):
            raise ValueError('Invalid blocked module/resource tuple')
        if tuple(pair) in blocked:
            raise ValueError('Duplicate blocked resource')
        blocked.add(tuple(pair))
    rmask = []
    for item in route['resource_candidates']:
        resource = _RESOURCE_IDS.get(item.resource_ref.resource_id)
        if resource is None:
            # Generic backends without an SMORES-specific Canonical exclusion
            # remain valid; if a module is blocked, unknown resource mapping
            # must fail closed rather than silently certify it as absent.
            if any(mid == item.resource_ref.module.module_id for mid, _ in blocked):
                raise ValueError('BC-v5 unknown resource category: '+item.resource_ref.resource_id)
            rmask.append(True)
        else:
            rmask.append((item.resource_ref.module.module_id, resource) not in blocked)
    # v8: label-scoped uncertainty does NOT alter the candidate catalogue.
    # Mask only the BCE contribution for admissible align choices on the
    # 17 explicitly ambiguous initiators. All other labels remain closed-world.
    uncertain = route.get('bc_v5_masked_align_initiators', ())
    if (not isinstance(uncertain, tuple) or len(set(uncertain)) != len(uncertain)
            or any(not isinstance(x,str) or not x for x in uncertain)
            or route.get('bc_v5_masked_align_count', len(uncertain)) != len(uncertain)):
        raise ValueError('Invalid v8 align supervision mask provenance')
    uncertain = set(uncertain)
    smask = [(binding.initiator.module_id, 'structural') not in blocked
             and not (action.action_id == 'align_faces' and binding.initiator.module_id in uncertain)
             for action, binding in route['structural_bindings']]
    result = torch.tensor(rmask + smask, dtype=torch.bool)
    if result.numel() != len(rmask)+len(smask):
        raise ValueError('Bad candidate validity width')
    if any(not bool(result[i]) for i in positive_indices(route)):
        raise ValueError('Overlay attempts to exclude a valid positive target')
    return result


def collate_binary_v5(samples, overlay):
    """Preserve physical ActionSet target encoding and candidate ordering."""
    if not samples:
        raise ValueError('Empty binary v5 batch')
    resource_masks, structural_masks = [], []
    for sample in samples:
        route = sample['routing']
        key = (route['audit'].episode_id, route['audit'].transition_index)
        if key not in overlay:
            raise ValueError('Missing authenticated overlay row')
        mask = candidate_validity(route, overlay[key])
        r = len(route['resource_candidates'])
        resource_masks.append(mask[:r])
        structural_masks.append(mask[r:])
    batch = collate_action_set_samples(samples)
    valid = torch.cat((*resource_masks, *structural_masks))
    if valid.shape != batch['supervision']['action_set']['presence'].shape:
        raise ValueError('BC-v5 candidate ordering/width mismatch')
    labels = batch['supervision']['action_set']['presence']
    if torch.any(labels & ~valid):
        raise ValueError('BC-v5 masked a positive target')
    batch['supervision']['action_set']['valid'] = valid
    return batch


def binary_presence_loss(logits, labels, valid, graph, num_graphs):
    """BCE positive + negative group means, per graph; masked rows have zero gradient."""
    if (type(num_graphs) is not int or num_graphs < 1
            or logits.ndim != 1 or not logits.is_floating_point()
            or labels.shape != logits.shape or labels.dtype != torch.bool
            or valid.shape != logits.shape or valid.dtype != torch.bool
            or graph.shape != logits.shape or graph.dtype != torch.int64):
        raise ValueError('Invalid BC-v5 binary presence ABI')
    if not torch.isfinite(logits).all() or ((graph < 0)|(graph >= num_graphs)).any():
        raise ValueError('Invalid logits/graph owners')
    labels, valid, graph = labels.to(logits.device), valid.to(logits.device), graph.to(logits.device)
    if (labels & ~valid).any():
        raise ValueError('Masked positive supervision')
    active_graphs = torch.zeros(num_graphs, device=logits.device, dtype=torch.bool)
    active_graphs.scatter_(0, graph[labels], True)
    if not active_graphs.all():
        raise ValueError('BC-v5 graph without positive targets: filter/exclude it')
    terms=[]
    for selected, sign in ((labels & valid, -1), (~labels & valid, 1)):
        owners=graph[selected]
        value=F.softplus(sign*logits[selected])
        numerator=logits.new_zeros(num_graphs).scatter_add(0,owners,value)
        denominator=logits.new_zeros(num_graphs).scatter_add(0,owners,torch.ones_like(value))
        terms.append(numerator/denominator.clamp_min(1))
    return (terms[0]+terms[1]).mean() + logits[:0].sum()


def compute_binary_v5_loss(prediction, batch):
    """Reuse unmodified positive parameter codecs; replace only presence term."""
    total, stats=compute_action_set_loss(prediction,batch)
    logits=prediction['presence_logits']
    graph,count=_candidate_graph(batch['inputs'])
    valid=batch['supervision']['action_set']['valid']
    labels=batch['supervision']['action_set']['presence']
    new=binary_presence_loss(logits,labels,valid,graph,count)
    before=stats['presence']['weighted']
    # No double-gradient: remove exactly the prior scalar presence contribution.
    total=total-before+new
    stats['presence']={'sum':new*count,'support':torch.tensor(count,device=logits.device),
                       'mean':new,'weighted':new}
    return total,stats


def binary_presence_metrics(prediction,batch,*,threshold=0.):
    """Metrics exclude only explicitly invalid candidates, never label absences."""
    logits=prediction['presence_logits'].detach().cpu()
    expected=batch['supervision']['action_set']['presence'].detach().cpu()
    valid=batch['supervision']['action_set']['valid'].detach().cpu()
    predicted=logits>=threshold
    tp=int((predicted & expected & valid).sum())
    fp=int((predicted & ~expected & valid).sum())
    fn=int((~predicted & expected & valid).sum())
    return {'tp':tp,'fp':fp,'fn':fn,'masked_candidates':int((~valid).sum()),
            'valid_candidates':int(valid.sum())}


def aggregate_binary_metrics(rows):
    totals=Counter()
    for row in rows:totals.update(row)
    tp,fp,fn=(totals[k] for k in ('tp','fp','fn'))
    p=tp/(tp+fp) if tp+fp else 0.
    r=tp/(tp+fn) if tp+fn else 0.
    return {**dict(totals), 'precision':p,'recall':r,
            'f1':2*p*r/(p+r) if p+r else 0.}
