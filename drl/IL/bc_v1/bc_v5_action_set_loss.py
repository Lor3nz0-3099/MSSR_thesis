"""BC-v5 S1 loss: same positive codecs as BC-v4, tri-state presence.

A new named entry point prevents accidentally changing BC-v4 checkpoints,
training, metrics, or decoding. Unknown candidates have no membership gradient.
"""
import torch

from .action_set_loss import _candidate_graph, _validate_positive_targets
from .bc_v5_membership import masked_presence_loss, POSITIVE
from . import losses as reference
from . import losses_bulk as bulk


def compute_action_set_v5_loss(prediction, batch):
    logits = prediction['presence_logits']
    graph = prediction['candidate_graph']
    expected_graph, count = _candidate_graph(batch['inputs'])
    if graph.dtype != torch.int64 or graph.ndim != 1 or not torch.equal(
            graph.detach().cpu(), expected_graph):
        raise ValueError('Presence candidate graph differs from physical ownership')
    y = batch['supervision']['action_set']
    membership = y.get('membership')
    if membership is None:
        raise ValueError('BC-v5 requires explicit three-state membership')
    labels = y['presence']
    if not torch.equal((membership == POSITIVE).detach().cpu(), labels.detach().cpu()):
        raise ValueError('BC-v5 and BC-v4 physical positive labels disagree')
    mean = masked_presence_loss(logits, membership, graph, count)
    parameters = prediction['parameters']
    if any(k in parameters['resource'] for k in ('mode_logits', 'outcome_logits')) or any(
            k in parameters['structural'] for k in ('candidate_logits', 'no_structural_logits')):
        raise ValueError('ActionSet parameters cannot contain mandatory decision logits')
    host_batch = {'inputs': {**batch['inputs'],
                    'resource_candidates': bulk._host(batch['inputs']['resource_candidates']),
                    'structural': bulk._host(batch['inputs']['structural'])},
                  'supervision': bulk._host({k: batch['supervision'][k]
                                             for k in ('resource', 'structural')})}
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
