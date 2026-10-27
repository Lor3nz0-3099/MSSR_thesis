"""Balanced physical-candidate membership with positive-only F2 codecs.

Each graph contributes one positive mean and one negative mean. Parameter
families retain the existing component-then-decision support normalization.
"""
import torch
from torch.nn import functional as F

from . import losses as reference
from . import losses_bulk as bulk


def presence_loss(logits, labels, graph, num_graphs):
    """Mean graph loss, with differentiable zero for every missing group."""
    if (type(num_graphs) is not int or num_graphs < 0
            or not isinstance(logits, torch.Tensor) or logits.ndim != 1
            or not logits.is_floating_point()
            or not isinstance(labels, torch.Tensor) or labels.dtype != torch.bool
            or labels.shape != logits.shape or labels.requires_grad
            or not isinstance(graph, torch.Tensor) or graph.dtype != torch.int64
            or graph.shape != logits.shape):
        raise ValueError('Invalid presence dtype/shape or graph count')
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite presence logits')
    if ((graph < 0) | (graph >= num_graphs)).any():
        raise ValueError('Presence candidate outside graph range')
    zero = logits[:0].sum()
    if not num_graphs:
        return zero
    labels = labels.to(logits.device); graph = graph.to(logits.device)
    terms = []
    for selected, sign in ((labels, -1), (~labels, 1)):
        owners = graph[selected]
        sums = logits.new_zeros(num_graphs).scatter_add(0, owners, F.softplus(sign * logits[selected]))
        counts = logits.new_zeros(num_graphs).scatter_add(0, owners, torch.ones_like(logits[selected]))
        terms.append(sums / counts.clamp_min(1))
    return (terms[0] + terms[1]).mean() + zero


def _candidate_graph(inputs):
    """Derive physical ownership from current G2 inputs, never audit IDs."""
    ptr = inputs['graph_ptr'].detach().cpu()
    if (ptr.dtype != torch.int64 or ptr.ndim != 1 or not len(ptr) or ptr[0] != 0
            or (ptr[1:] < ptr[:-1]).any()):
        raise ValueError('Invalid graph pointer')
    count = len(ptr) - 1
    module_graph = torch.repeat_interleave(torch.arange(count), ptr[1:] - ptr[:-1])
    parent = inputs['resource_to_module'].detach().cpu()
    resource = inputs['resource_candidates']['resource'].detach().cpu()
    initiator = inputs['structural']['initiator'].detach().cpu()
    for values, bound in ((parent, len(module_graph)), (resource, len(parent)), (initiator, len(module_graph))):
        if (values.dtype != torch.int64 or values.ndim != 1
                or ((values < 0) | (values >= bound)).any()):
            raise ValueError('Invalid physical candidate ownership')
    return torch.cat((module_graph[parent[resource]], module_graph[initiator])), count


def _validate_positive_targets(batch, labels):
    x = batch['inputs']; y = batch['supervision']
    resource_count = len(x['resource_candidates']['resource'])
    structural_count = len(x['structural']['initiator'])
    expected = torch.zeros(resource_count + structural_count, dtype=torch.bool)
    for family, count, offset in (('resource', resource_count, 0), ('structural', structural_count, resource_count)):
        target = y[family]; candidates = target['candidate']
        if not target['supervised'].all():
            raise ValueError('ActionSet requires positive-only parameter targets')
        if ((candidates < 0) | (candidates >= count)).any() or len(set(candidates.tolist())) != len(candidates):
            raise ValueError('Invalid/duplicate positive candidate target')
        expected[offset + candidates] = True
        actual_owner = x['resource_candidates']['resource'][candidates] if family == 'resource' else x['structural']['initiator'][candidates]
        owner = target['resource'] if family == 'resource' else target['group']
        if not torch.equal(owner, actual_owner):
            raise ValueError('Positive candidate owner mismatch')
    if not torch.equal(labels.detach().cpu(), expected):
        raise ValueError('Presence and positive parameter targets disagree')


def compute_action_set_loss(prediction, batch):
    """Return unit-weight presence and positive codec family statistics.

    No resource mode, inactivity outcome, or structural choice loss is used.
    Supervision is the reconstructed v1 positive tensor ABI; physical source
    targets in routing are deliberately outside this objective.
    """
    logits = prediction['presence_logits']; graph = prediction['candidate_graph']
    expected_graph, count = _candidate_graph(batch['inputs'])
    if graph.dtype != torch.int64 or graph.ndim != 1 or not torch.equal(graph.detach().cpu(), expected_graph):
        raise ValueError('Presence candidate graph differs from physical ownership')
    labels = batch['supervision']['action_set']['presence']
    mean = presence_loss(logits, labels, graph, count)
    parameters = prediction['parameters']
    if any(k in parameters['resource'] for k in ('mode_logits', 'outcome_logits')) or any(
            k in parameters['structural'] for k in ('candidate_logits', 'no_structural_logits')):
        raise ValueError('ActionSet parameters cannot contain mandatory decision logits')
    host_batch = {'inputs': {**batch['inputs'],
                            'resource_candidates': bulk._host(batch['inputs']['resource_candidates']),
                            'structural': bulk._host(batch['inputs']['structural'])},
                  'supervision': bulk._host({k: batch['supervision'][k] for k in ('resource', 'structural')})}
    reference._validate_supervision(host_batch)
    _validate_positive_targets(host_batch, labels)
    host_prediction = bulk._host(parameters, routes_only=True)
    support = torch.tensor(count, dtype=torch.int64, device=logits.device)
    stats = {'presence': {'sum': mean * count, 'support': support, 'mean': mean}}
    stats['resource_payload'] = bulk._payload(parameters, batch, host_prediction, host_batch)
    stats.update(bulk._argument_losses(parameters, batch, host_prediction, host_batch))
    total = reference._anchor(prediction)
    for row in stats.values():
        row['weighted'] = row['mean']
        total = total + row['weighted']
    return total, stats
