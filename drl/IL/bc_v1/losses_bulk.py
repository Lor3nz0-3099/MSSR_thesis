"""Opt-in BC reductions with host routing and grouped device arithmetic.

Routing is snapshotted afresh once per tensor, never cached across calls. Python
validation/selection sees CPU metadata only; differentiable predictions and loss
arithmetic remain on their original device. The reference loss module remains
the contract and the legacy resource-mode fallback.
"""
from collections import defaultdict

import torch
from torch.nn import functional as F

from . import losses as reference
from .resource_decisions import supervised_decisions, validate_decision_targets
from .structural_decisions import NO_STRUCTURAL, _positive_targets


def _host(tree, *, routes_only=False):
    return {key: (_host(value, routes_only=routes_only) if isinstance(value, dict)
                  else value if routes_only and value.is_floating_point()
                  else value.to(device='cpu')) for key, value in tree.items()}


def _indices(values, tensor):
    return torch.tensor(values, dtype=torch.int64, device=tensor.device)


def _categorical(logits, decisions):
    """One stable CE per (flat option indices, local target), grouped by arity."""
    buckets = defaultdict(list)
    for row, (indices, target) in enumerate(decisions):
        if not 0 <= target < len(indices):
            raise ValueError('Target outside categorical domain')
        buckets[len(indices)].append((row, indices, target))
    values, order = [], []
    for width, rows in buckets.items():
        indices = [index for _, options, _ in rows for index in options]
        if logits.ndim != 1 or min(indices) < 0 or max(indices) >= len(logits):
            raise ValueError('Missing/misaligned categorical prediction')
        scores = logits[_indices(indices, logits)].reshape(len(rows), width)
        if not torch.isfinite(scores).all():
            raise ValueError('Nonfinite supervised logits')
        targets = _indices([target for _, _, target in rows], logits)
        values.append(torch.logsumexp(scores, 1) - scores.gather(1, targets[:, None]).flatten())
        order.extend(row for row, _, _ in rows)
    if not values:
        return logits.reshape(-1)[:0]
    inverse = sorted(range(len(order)), key=order.__getitem__)
    result = torch.cat(values)
    return result[_indices(inverse, result)]


def _components(predicted, targets, decisions, *, binary=False):
    """Select first, reduce components within each nonempty decision second."""
    prediction_indices, target_indices, owners, counts = [], [], [], []
    for p, t in decisions:
        if len(p) != len(t):
            raise ValueError('Numeric target arity/mask mismatch')
        if not p:
            continue
        owners.extend([len(counts)] * len(p))
        counts.append(len(p))
        prediction_indices.extend(p)
        target_indices.extend(t)
    if not counts:
        return predicted.reshape(-1)[:0]
    p = predicted[_indices(prediction_indices, predicted)]
    t = targets[_indices(target_indices, targets)]
    if t.requires_grad:
        raise ValueError('Targets must not require gradients')
    if not torch.isfinite(p).all() or not torch.isfinite(t).all():
        raise ValueError('Nonfinite supervised numeric value')
    if binary:
        if not ((t == 0) | (t == 1)).all():
            raise ValueError('Invalid supervised membership/boolean value')
        components = F.binary_cross_entropy_with_logits(p, t, reduction='none')
    else:
        components = F.huber_loss(p, t, reduction='none', delta=1.)
    sums = components.new_zeros(len(counts)).scatter_add(0, _indices(owners, components), components)
    return sums / components.new_tensor(counts)


def _statistics(values, anchor):
    if not len(values):
        values = anchor.reshape(-1)[:0]
    return reference.masked_statistics(values, torch.ones_like(values, dtype=torch.bool))


def _resource_mode(prediction, batch, host_prediction, host_batch):
    p = prediction['resource']
    if 'outcome_logits' not in p:
        return reference.resource_mode_loss(host_prediction, host_batch)
    # This generator performs metadata validation only; all its scalar reads
    # address the fresh host snapshot. Device slices are not evaluated here.
    rows = list(supervised_decisions(host_prediction, host_batch))
    ptr = host_batch['inputs']['resource_candidates']['outcome_ptr'].tolist()
    decisions = [(range(ptr[row], ptr[row+1]), target) for row, _, target in rows]
    return _statistics(_categorical(p['outcome_logits'], decisions), reference._anchor(prediction))


def _payload(prediction, batch, host_prediction, host_batch):
    p = prediction['resource']; hp = host_prediction['resource']
    y = host_batch['supervision']['resource']; x = host_batch['inputs']['resource_candidates']
    if 'outcome_logits' in p:
        validate_decision_targets(host_batch)
    numeric = reference._routes(hp['payload_candidate'])
    category = reference._routes(hp['categorical']['candidate'])
    numeric_rows, category_rows, seen = [], [], set()
    for row, valid in enumerate(y['supervised'].tolist()):
        if not valid:
            continue
        resource, candidate = int(y['resource'][row]), int(y['candidate'][row])
        start, end = reference._bounds(x['group_ptr'], resource)
        if not start <= candidate < end:
            raise ValueError('Resource payload outside group')
        if resource in seen:
            raise ValueError('Duplicate supervised resource payload')
        seen.add(resource)
        a, b = reference._bounds(y['payload_ptr'], row)
        valid_components = y['payload_valid'][a:b].tolist()
        if candidate in numeric:
            lo, hi = reference._bounds(hp['payload_ptr'], numeric[candidate])
            if p['payload'].ndim != 1 or len(p['payload'][lo:hi]) != b-a:
                raise ValueError('Numeric target arity/mask mismatch')
            numeric_rows.append(([lo+i for i, flag in enumerate(valid_components) if flag],
                                 [a+i for i, flag in enumerate(valid_components) if flag]))
        elif candidate in category:
            if b-a != 1:
                raise ValueError('Categorical resource payload must have one category')
            lo, hi = reference._bounds(hp['categorical']['ptr'], category[candidate])
            if any(valid_components):
                category_rows.append((range(lo, hi), int(y['payload_category'][a])))
        else:
            raise ValueError('Missing payload prediction')
    values = torch.cat((_components(p['payload'], batch['supervision']['resource']['payload'], numeric_rows),
                        _categorical(p['categorical']['logits'], category_rows)))
    return _statistics(values, reference._anchor(prediction))


def _structural(prediction, host_prediction, host_batch):
    p = prediction['structural']; hp = host_prediction['structural']
    y = host_batch['supervision']['structural']; ptr = hp['group_ptr']; count = len(ptr)-1
    decisions = []
    if 'no_structural_logits' in p:
        none = p['no_structural_logits']
        if none.shape != (count,):
            raise ValueError('Invalid NO_STRUCTURAL logit groups')
        for key, dtype in [('decision_group', torch.int64), ('decision_candidate', torch.int64),
                           ('decision_supervised', torch.bool)]:
            if key not in y or y[key].shape != (count,) or y[key].dtype != dtype:
                raise ValueError('Missing/invalid dense decision supervision')
        if not torch.equal(y['decision_group'], torch.arange(count, device=ptr.device)):
            raise ValueError('Decision groups must cover every initiator once')
        if not torch.equal(y['decision_candidate'], _positive_targets(y, ptr)):
            raise ValueError('Dense decision mapping changed preserved positive targets')
        for group, (candidate, valid) in enumerate(zip(y['decision_candidate'].tolist(), y['decision_supervised'].tolist())):
            if not valid:
                continue
            lo, hi = reference._bounds(ptr, group)
            if candidate != NO_STRUCTURAL and not lo <= candidate < hi:
                raise ValueError('Structural target outside joint domain')
            decisions.append(([group, *range(count+lo, count+hi)],
                              0 if candidate == NO_STRUCTURAL else candidate-lo+1))
        logits = torch.cat((none, p['candidate_logits']))
    else:
        if 'decision_group' in y:
            raise ValueError('v2 supervision requires NO_STRUCTURAL logits')
        seen = set()
        for group, candidate, valid in zip(y['group'].tolist(), y['candidate'].tolist(), y['supervised'].tolist()):
            if not valid:
                continue
            lo, hi = reference._bounds(ptr, group)
            if not lo <= candidate < hi:
                raise ValueError('Structural target absent/outside initiator group')
            if group in seen:
                raise ValueError('Duplicate structural decision; no certified positive-set contract')
            seen.add(group)
            decisions.append((range(lo, hi), candidate-lo))
        logits = p['candidate_logits']
    return _statistics(_categorical(logits, decisions), reference._anchor(prediction))


def _argument_losses(prediction, batch, host_prediction, host_batch):
    """Shared host routing for learnable arguments, optional sets, and scope."""
    p = prediction['structural']; hp = host_prediction['structural']
    x = host_batch['inputs']['structural']; y = host_batch['supervision']['structural']
    targets = batch['supervision']['structural']; result = {}; anchor = reference._anchor(prediction)
    for name, kinds in [('category', ('category',)), ('numeric', ('scalar', 'vector')),
                        ('reference', ('module_ref', 'connector_ref')),
                        ('boolean', ('boolean',))]:
        records, host_records = p['arguments'][name], hp['arguments'][name]
        routes = reference._routes(host_records['argument'])
        decisions = []
        for row, argument in reference._argument_rows(host_batch, kinds):
            if name == 'reference':
                a, b = reference._bounds(y['reference_ptr'], row)
                if b-a != 1:
                    raise ValueError('Single reference target must contain one reference')
                target = int(y['reference_index'][a]); bound = int(x['argument_bound_ref'][argument])
                if bound != -1:
                    if target != bound:
                        raise ValueError('Target differs from fixed bound reference')
                    continue
            if name == 'boolean':
                a, b = reference._bounds(y['argument_value_ptr'], row)
                if b-a != 1:
                    raise ValueError('Boolean target must have one value')
                if not bool(y['argument_value_valid'][a]):
                    continue
            if argument not in routes:
                raise ValueError('Missing learnable argument prediction')
            route = routes[argument]
            if name == 'boolean':
                decisions.append(([route], [a]))
                continue
            lo, hi = reference._bounds(host_records['ptr'], route)
            if name == 'category':
                decisions.append((range(lo, hi), int(y['argument_category'][row])))
            elif name == 'reference':
                options = x['argument_domain_index'][host_records['domain_index'][lo:hi]].tolist()
                if target not in options:
                    raise ValueError('Reference target outside current allowed domain')
                decisions.append((range(lo, hi), options.index(target)))
            else:
                a, b = reference._bounds(y['argument_value_ptr'], row)
                if records['value'].ndim != 1 or len(records['value'][lo:hi]) != b-a:
                    raise ValueError('Numeric target arity/mask mismatch')
                valid = y['argument_value_valid'][a:b].tolist()
                decisions.append(([lo+i for i, flag in enumerate(valid) if flag],
                                  [a+i for i, flag in enumerate(valid) if flag]))
        if name in ('category', 'reference'):
            values = _categorical(records['logits'], decisions)
        else:
            values = _components(records['logits' if name == 'boolean' else 'value'],
                                 targets['argument_value'], decisions, binary=name == 'boolean')
        result['arg_'+name] = _statistics(values, anchor)

    reference._validate_membership_owners(host_batch)
    records = p['arguments']['reference_set']; host_records = hp['arguments']['reference_set']
    routes = reference._routes(host_records['argument']); decisions = []
    for row, argument in reference._argument_rows(host_batch, ('module_refs', 'connector_refs')):
        if argument not in routes:
            raise ValueError('Missing learnable argument prediction')
        lo, hi = reference._bounds(x['argument_domain_ptr'], argument)
        a, b = reference._bounds(y['reference_ptr'], row)
        if not set(y['reference_index'][a:b].tolist()) <= set(x['argument_domain_index'][lo:hi].tolist()):
            raise ValueError('Set target outside allowed reference domain')
        labels = [i for i, domain in enumerate(y['membership_domain'].tolist()) if lo <= domain < hi]
        decisions.append(_membership(host_records, routes[argument], y['membership_domain'],
                                     y['membership_valid'], x['argument_domain_required'], (lo, hi), labels))
    result['arg_reference_set'] = _statistics(_components(records['logits'], targets['membership_value'], decisions, binary=True), anchor)

    records = p['scope']; host_records = hp['scope']; routes = reference._routes(host_records['binding']); decisions = []
    for row, valid in enumerate(y['supervised'].tolist()):
        if not valid:
            continue
        candidate = int(y['candidate'][row]); lo, hi = reference._bounds(x['scope_ptr'], candidate)
        if x['scope_required'][lo:hi].all():
            continue
        if candidate not in routes:
            raise ValueError('Missing variable participant-scope prediction')
        a, b = reference._bounds(y['scope_membership_ptr'], row)
        decisions.append(_membership(host_records, routes[candidate], y['scope_membership_index'],
                                     y['scope_membership_valid'], x['scope_required'], (lo, hi), range(a, b)))
    result['scope'] = _statistics(_components(records['logits'], targets['scope_membership_value'], decisions, binary=True), anchor)
    return result


def _membership(records, row, indices, valid, required, domain_range, labels):
    start, end = reference._bounds(records['ptr'], row)
    routes = reference._routes(records['domain_index'][start:end])
    selected, targets, seen = [], [], set(); lo, hi = domain_range
    for label in labels:
        if not bool(valid[label]):
            continue
        domain = int(indices[label])
        if not lo <= domain < hi:
            raise ValueError('Membership target outside decision domain')
        if domain in seen:
            raise ValueError('Duplicate supervised membership')
        seen.add(domain)
        if bool(required[domain]):
            continue
        if domain not in routes:
            raise ValueError('Missing optional membership prediction')
        selected.append(start+routes[domain]); targets.append(label)
    return selected, targets


def compute_bc_loss_bulk(prediction, batch, loss_config):
    """Reference-compatible opt-in objective; no per-decision CUDA routing read."""
    reference._validate_config(loss_config)
    host_batch = {'inputs': {**batch['inputs'],
                            'resource_candidates': _host(batch['inputs']['resource_candidates']),
                            'structural': _host(batch['inputs']['structural'])},
                  'supervision': _host(batch['supervision'])}
    host_prediction = _host(prediction, routes_only=True)
    reference._validate_supervision(host_batch)
    stats = {'resource_mode': _resource_mode(prediction, batch, host_prediction, host_batch),
             'resource_payload': _payload(prediction, batch, host_prediction, host_batch),
             'structural_choice': _structural(prediction, host_prediction, host_batch)}
    stats.update(_argument_losses(prediction, batch, host_prediction, host_batch))
    total = reference._anchor(prediction)
    # Preserve family accumulation order, independent of routing implementation.
    for family in reference.LOSS_FUNCTIONS:
        row = stats[family]
        row['weighted'] = loss_config['weights'][family] * row['mean']
        total = total + row['weighted']
    return total, {family: stats[family] for family in reference.LOSS_FUNCTIONS}
