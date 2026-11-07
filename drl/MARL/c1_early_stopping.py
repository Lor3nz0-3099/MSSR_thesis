"""Serializable validation-only stopping decisions for a bounded C1 campaign.

Domain scores must already use a declared comparable evaluation metric. This
controller applies equal domain weights; it does not invent reward rescaling.
The caller supplies elapsed process wall time including startup, assembly,
updates, checkpointing, and validation. No simulator or subprocess runs here.
"""
from __future__ import annotations

import math

DOMAINS = ('navigation', 'stairs', 'gap', 'manipulation')


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return float(value)


def _balanced_score(split, score_by_domain):
    if split != 'validation':
        raise ValueError('Early stopping accepts only the explicit validation split')
    if set(score_by_domain) != set(DOMAINS):
        raise ValueError('Scores must contain exactly the four C1 domains')
    values = []
    for domain in DOMAINS:
        value = score_by_domain[domain]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{domain} score must be finite')
        values.append(float(value))
    return math.fsum(values) / len(values)


class ValidationEarlyStopping:
    """Select balanced validation outcomes; patience counts validation events.

    With preserve_domain_floor=True, macro success is primary, macro return is
    the tiebreaker, and each domain must retain its initial BC success. At equal
    domain success, its return must also retain the initial BC value. The floor
    has zero tolerance; an average gain cannot mask a domain's regression.

    min_delta=0 requires a strictly improved score. Ties consume patience. The
    best checkpoint always identifies a real evaluated candidate supplied by the
    caller. Restoring a checkpoint and pausing training remain caller operations.
    """
    def __init__(self, patience, min_delta=0., preserve_domain_floor=False):
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 1:
            raise ValueError('patience must be a positive integer')
        self.patience = patience
        self.min_delta = _nonnegative(min_delta, 'min_delta')
        self.preserve_domain_floor = bool(preserve_domain_floor)
        self.baseline_score_by_domain = None
        self.baseline_success_by_domain = None
        self.best_macro_success = None
        self.last_selection_proof = None
        self.best_score = None
        self.best_checkpoint = None
        self.bad_validations = 0
        self.validation_count = 0

    def update(self, checkpoint_path, split, score_by_domain, success_by_domain=None):
        score = _balanced_score(split, score_by_domain)
        success = dict.fromkeys(DOMAINS, 0.) if success_by_domain is None else dict(success_by_domain)
        macro_success = _balanced_score(split, success)
        if any(not 0 <= value <= 1 for value in success.values()):
            raise ValueError('Domain success values must be probabilities in [0, 1]')
        if not str(checkpoint_path or '').strip():
            raise ValueError('An evaluated checkpoint_path is required')
        violations = {}
        if self.baseline_score_by_domain is None:
            # The native campaign's first frozen candidate is the initial BC
            # policy. Preserve its per-domain skills with explicit zero slack.
            self.baseline_score_by_domain = dict(score_by_domain)
            self.baseline_success_by_domain = dict(success)
        if self.preserve_domain_floor:
            for domain in DOMAINS:
                if success[domain] < self.baseline_success_by_domain[domain]:
                    violations[domain] = 'success_below_bc_baseline'
                elif (success[domain] == self.baseline_success_by_domain[domain] and
                      score_by_domain[domain] < self.baseline_score_by_domain[domain]):
                    violations[domain] = 'return_below_bc_baseline_at_equal_success'
        use_success = self.preserve_domain_floor or success_by_domain is not None
        if self.best_score is None:
            improved = True
        elif use_success:
            improved = (macro_success > self.best_macro_success + self.min_delta or
                        (macro_success == self.best_macro_success and score > self.best_score + self.min_delta))
        else:
            improved = score > self.best_score + self.min_delta
        improved = improved and not violations
        self.last_selection_proof = dict(
            metric='macro_domain_success_then_macro_return' if use_success else 'macro_domain_return',
            success_source='observed_fixed_window' if success_by_domain is not None else 'missing_assumed_zero',
            baseline_score_by_domain=dict(self.baseline_score_by_domain),
            baseline_success_by_domain=dict(self.baseline_success_by_domain),
            candidate_score_by_domain=dict(score_by_domain), candidate_success_by_domain=success,
            preserve_domain_floor=self.preserve_domain_floor, tolerance=0., guard_violations=violations)
        self.validation_count += 1
        if improved:
            self.best_score, self.best_checkpoint = score, str(checkpoint_path)
            self.best_macro_success = macro_success
            self.bad_validations = 0
        else:
            self.bad_validations += 1
        return dict(stop=self.bad_validations >= self.patience, improved=improved,
                    balanced_score=score, macro_success=macro_success,
                    best_macro_success=self.best_macro_success, guard_violations=violations,
                    selection_proof=self.last_selection_proof, best_score=self.best_score,
                    best_checkpoint=self.best_checkpoint, bad_validations=self.bad_validations,
                    validation_count=self.validation_count,
                    reason='validation_patience' if self.bad_validations >= self.patience else None)

    def state_dict(self):
        return dict(schema='mssr.c1_validation_early_stopping.v1', patience=self.patience,
                    min_delta=self.min_delta, best_score=self.best_score,
                    best_checkpoint=self.best_checkpoint, bad_validations=self.bad_validations,
                    validation_count=self.validation_count,
                    preserve_domain_floor=self.preserve_domain_floor,
                    baseline_score_by_domain=self.baseline_score_by_domain,
                    baseline_success_by_domain=self.baseline_success_by_domain,
                    best_macro_success=self.best_macro_success,
                    last_selection_proof=self.last_selection_proof)

    @classmethod
    def from_state_dict(cls, state):
        if state.get('schema') != 'mssr.c1_validation_early_stopping.v1':
            raise ValueError('Unsupported early stopping state schema')
        instance = cls(state['patience'], state['min_delta'], state.get('preserve_domain_floor', False))
        for name in ('baseline_score_by_domain', 'baseline_success_by_domain', 'best_macro_success', 'last_selection_proof'):
            setattr(instance, name, state.get(name))
        for name in ('best_score', 'best_checkpoint', 'bad_validations', 'validation_count'):
            setattr(instance, name, state[name])
        if instance.best_score is not None and instance.best_macro_success is None:
            instance.best_macro_success = 0.
        if instance.best_score is not None and not math.isfinite(instance.best_score):
            raise ValueError('Restored best_score must be finite')
        for name in ('bad_validations', 'validation_count'):
            value = getattr(instance, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('Restored counters must be nonnegative integers')
        if instance.bad_validations > instance.validation_count:
            raise ValueError('Bad validation count exceeds total validations')
        return instance


class WallBudget:
    """Reserve measured operation cost and final checkpoint before starting work.

    Unknown required costs fail closed. Reservations use the most recently
    measured duration of each operation; overruns remain possible, so this is an
    admission guard, not a real-time deadline guarantee. A caller may pass a
    stricter max_wall_s below 24h to reserve an additional operational margin.
    """
    def __init__(self, max_wall_s=24 * 60 * 60):
        self.max_wall_s = _nonnegative(max_wall_s, 'max_wall_s')
        if self.max_wall_s == 0:
            raise ValueError('max_wall_s must be positive')
        self.costs = dict(update=None, checkpoint=None, validation=None)

    def record_costs(self, *, update_s=None, checkpoint_s=None, validation_s=None):
        for name, value in dict(update=update_s, checkpoint=checkpoint_s, validation=validation_s).items():
            if value is not None:
                self.costs[name] = _nonnegative(value, name + '_s')

    def decision(self, elapsed_s, next_operation='update'):
        elapsed = _nonnegative(elapsed_s, 'elapsed_s')
        if next_operation not in self.costs:
            raise ValueError('next_operation must be update, checkpoint, or validation')
        needed = [next_operation] if next_operation == 'checkpoint' else [next_operation, 'checkpoint']
        unknown = [name for name in needed if self.costs[name] is None]
        reserve = None if unknown else sum(self.costs[name] for name in needed)
        remaining = max(0., self.max_wall_s - elapsed)
        reason = ('wall_budget_exhausted' if elapsed >= self.max_wall_s else
                  'missing_measured_cost:' + ','.join(unknown) if unknown else
                  'insufficient_wall_reserve' if reserve >= remaining else None)
        return dict(stop=reason is not None, reason=reason, elapsed_s=elapsed,
                    max_wall_s=self.max_wall_s, remaining_s=remaining,
                    reserved_s=reserve, next_operation=next_operation)

    def state_dict(self):
        return dict(schema='mssr.c1_wall_budget.v1', max_wall_s=self.max_wall_s, costs=dict(self.costs))

    @classmethod
    def from_state_dict(cls, state):
        if state.get('schema') != 'mssr.c1_wall_budget.v1':
            raise ValueError('Unsupported wall budget state schema')
        instance = cls(state['max_wall_s'])
        instance.record_costs(**{name + '_s': value for name, value in state['costs'].items()})
        return instance


def validate_frozen_evaluation(result, *, expected_seed_manifest=None):
    """Check evidence from a paused-training frozen native validation callback.

    Caller is responsible for constructing the fixed validation bank from the
    authoritative curriculum and measuring available VRAM before launching a
    separate native evaluation subprocess. Exact bank equality can be enforced
    with expected_seed_manifest. TRAIN, TEST and challenge scores never enter
    this controller. Unchanged actual parameter hashes and zero optimizer calls
    are required; a filename or an 'eval' flag alone is insufficient evidence.
    """
    score = _balanced_score(result.get('split'), result.get('score_by_domain', {}))
    if result.get('optimizer_updates') != 0 or isinstance(result.get('optimizer_updates'), bool):
        raise ValueError('Frozen validation must perform exactly zero optimizer updates')
    before, after = result.get('parameter_sha256_before'), result.get('parameter_sha256_after')
    if not isinstance(before, str) or len(before) != 64 or any(c not in '0123456789abcdef' for c in before) or after != before:
        raise ValueError('Frozen validation requires equal before/after parameter SHA256 hashes')
    for field in ('checkpoint_path', 'seed_manifest', 'command'):
        if not result.get(field):
            raise ValueError(f'Frozen validation evidence missing {field}')
    if expected_seed_manifest is not None and result['seed_manifest'] != expected_seed_manifest:
        raise ValueError('Validation seed manifest differs from the fixed validation bank')
    available, peak = result.get('available_gpu_bytes'), result.get('peak_gpu_bytes')
    if available is None or peak is None:
        raise ValueError('Frozen evaluation needs measured available and peak GPU memory')
    available = _nonnegative(available, 'available_gpu_bytes')
    peak = _nonnegative(peak, 'peak_gpu_bytes')
    if available == 0 or peak == 0 or peak > available:
        raise ValueError('Frozen validation does not fit measured available GPU memory')
    return dict(balanced_score=score, checkpoint_path=str(result['checkpoint_path']),
                split='validation', optimizer_updates=0, frozen_parameters_verified=True,
                seed_manifest=result['seed_manifest'])
