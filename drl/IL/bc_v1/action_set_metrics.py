"""Streaming physical action-set diagnostics without retained predictions.

Membership ignores values but preserves ordered connector bindings. Numeric
observations require a declared unit, and use only identities matched in both
sets. These offline metrics never claim executor feasibility or task success.
"""
import math

import torch


from .action_set import action_identity


def _scores(tp, fp, fn):
    # Both empty is a correct set; a unilateral empty set has zero agreement.
    empty = tp + fp + fn == 0
    return {'precision': tp/(tp+fp) if tp+fp else float(empty),
            'recall': tp/(tp+fn) if tp+fn else float(empty),
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.}


def _values(value):
    if isinstance(value, torch.Tensor):
        if value.ndim > 1:
            raise ValueError('Numeric metrics require scalar or vector values')
        return value.detach().double().cpu().reshape(-1).tolist()
    if type(value) in (float, int):
        return [float(value)]
    if isinstance(value, (tuple, list)) and all(type(v) in (float, int) for v in value):
        return [float(v) for v in value]
    raise ValueError('Numeric metrics require numeric scalar or vector values')


class ActionSetMetrics:
    """Accumulate membership plus optional matched numeric measurements.

    ``parameter_units`` maps ``(action_id, parameter_name)`` or parameter names
    to a unit; ``payload`` names resource payloads. Tolerances are keyed by
    unit. Missing units skip numeric measurements. Explicit normalized values
    may be recorded with ``update_numeric(..., normalized=True)``.
    """
    def __init__(self, parameter_units=None, tolerances=None):
        self.parameter_units = dict(parameter_units or {})
        self.tolerances = dict(tolerances or {})
        if (any(not isinstance(v, str) or not v for v in self.parameter_units.values())
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in self.tolerances.values())):
            raise ValueError('Invalid metric units or tolerances')
        self.reset()

    def reset(self):
        self.samples = self.tp = self.fp = self.fn = 0
        self.exact = self.target_count = self.predicted_count = self.cardinality_error = 0
        self.jaccard_sum = 0.
        self.target_min = self.target_max = self.predicted_min = self.predicted_max = None
        self.per_action = {}
        self.numeric = {}

    def update_numeric(self, key, prediction, target, unit, tolerance=None, normalized=False):
        """Record one matched scalar/vector observation, separately per unit."""
        if not isinstance(key, str) or not key or not isinstance(unit, str) or not unit or type(normalized) is not bool:
            raise ValueError('Invalid numeric metric key/unit')
        if tolerance is None:
            tolerance = self.tolerances.get(unit)
        if tolerance is not None and (type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0):
            raise ValueError('Invalid numeric metric tolerance')
        p, t = _values(prediction), _values(target)
        if len(p) != len(t) or not all(math.isfinite(v) for v in p+t):
            raise ValueError('Nonfinite or misaligned numeric metric values')
        errors = [abs(a-b) for a, b in zip(p, t)]
        if not all(math.isfinite(v*v) for v in errors):
            raise ValueError('Numeric metric exceeds finite accumulator range')
        bucket = self.numeric.setdefault((key, unit, normalized),
            {'support': 0, 'absolute': 0., 'squared': 0., 'within': 0, 'tolerance': tolerance})
        if bucket['tolerance'] != tolerance:
            raise ValueError('Metric tolerance changed within a stream')
        bucket['support'] += len(errors)
        bucket['absolute'] = math.fsum((bucket['absolute'], *errors))
        bucket['squared'] = math.fsum((bucket['squared'], *(v*v for v in errors)))
        if tolerance is not None:
            bucket['within'] += sum(v <= tolerance for v in errors)

    def _matched_numeric(self, target, predicted):
        target_values = {'payload': target.payload, **{p.name: p.value for p in target.parameters}}
        predicted_values = {'payload': predicted.payload, **{p.name: p.value for p in predicted.parameters}}
        for name in target_values.keys() & predicted_values.keys():
            unit = self.parameter_units.get((target.action_id, name), self.parameter_units.get(name))
            if unit is None:
                continue
            value = target_values[name]
            if type(value) not in (int, float) and not (isinstance(value, (tuple, list)) and all(type(v) in (int, float) for v in value)):
                continue
            self.update_numeric(f'{target.action_id}.{name}', predicted_values[name], value, unit)

    def update(self, target_bundle, predicted_bundle, *, numeric=True):
        """Consume one sample; optionally record membership without values."""
        if type(numeric) is not bool:
            raise ValueError('Numeric observation flag must be boolean')
        target = {action_identity(a): a for a in target_bundle.actions}
        predicted = {action_identity(a): a for a in predicted_bundle.actions}
        if len(target) != len(target_bundle.actions) or len(predicted) != len(predicted_bundle.actions):
            raise ValueError('Duplicate action candidate identity')
        truth, proposed = set(target), set(predicted)
        matched = truth & proposed; extra = proposed - truth; missing = truth - proposed
        if numeric:
            for identity in matched:
                self._matched_numeric(target[identity], predicted[identity])
        self.samples += 1
        self.tp += len(matched); self.fp += len(extra); self.fn += len(missing)
        self.exact += truth == proposed
        self.jaccard_sum += len(matched)/len(truth | proposed) if truth | proposed else 1.
        self.target_count += len(truth); self.predicted_count += len(proposed)
        self.cardinality_error += abs(len(truth)-len(proposed))
        self.target_min = min(self.target_min, len(truth)) if self.target_min is not None else len(truth)
        self.target_max = max(self.target_max, len(truth)) if self.target_max is not None else len(truth)
        self.predicted_min = min(self.predicted_min, len(proposed)) if self.predicted_min is not None else len(proposed)
        self.predicted_max = max(self.predicted_max, len(proposed)) if self.predicted_max is not None else len(proposed)
        for identities, field in ((matched, 'tp'), (extra, 'fp'), (missing, 'fn')):
            for identity in identities:
                row = self.per_action.setdefault(identity[0], {'tp': 0, 'fp': 0, 'fn': 0})
                row[field] += 1

    def compute(self):
        """Return plain sufficient-statistic summaries; no tensors retained."""
        count = self.samples
        result = {'samples': count, 'true_positive': self.tp, 'false_positive': self.fp,
                  'false_negative': self.fn, **_scores(self.tp, self.fp, self.fn),
                  'exact_match': self.exact/count if count else None,
                  'jaccard': self.jaccard_sum/count if count else None,
                  'target_cardinality': self.target_count/count if count else None,
                  'predicted_cardinality': self.predicted_count/count if count else None,
                  'cardinality_absolute_error': self.cardinality_error/count if count else None,
                  'target_cardinality_min': self.target_min, 'target_cardinality_max': self.target_max,
                  'predicted_cardinality_min': self.predicted_min, 'predicted_cardinality_max': self.predicted_max,
                  'per_action': {}, 'numeric': {}}
        for action, row in sorted(self.per_action.items()):
            result['per_action'][action] = {**row, **_scores(row['tp'], row['fp'], row['fn']),
                                           'support': row['tp']+row['fn']}
        normalized_squared = 0.; normalized_support = 0
        for (key, unit, normalized), row in sorted(self.numeric.items()):
            n = row['support']
            namespace = 'normalized_numeric' if normalized else 'numeric'
            result.setdefault(namespace, {}).setdefault(key, {})[unit] = {
                'support': n, 'mae': row['absolute']/n if n else None,
                'rmse': math.sqrt(row['squared']/n) if n else None,
                'within_tolerance': row['within']/n if n and row['tolerance'] is not None else None,
                'tolerance': row['tolerance'], 'unit': unit}
            if normalized and (key == 'payload' or key.endswith('.payload')):
                normalized_squared += row['squared']; normalized_support += n
        result['normalized_payload_rmse'] = math.sqrt(normalized_squared/normalized_support) if normalized_support else None
        for alias, source in (('set_exact_match', 'exact_match'), ('set_precision', 'precision'),
                              ('set_recall', 'recall'), ('set_f1', 'f1'), ('set_jaccard', 'jaccard'),
                              ('target_cardinality_mean', 'target_cardinality'),
                              ('predicted_cardinality_mean', 'predicted_cardinality'),
                              ('false_positive_actions', 'false_positive'), ('false_negative_actions', 'false_negative')):
            result[alias] = result[source]
        return result
