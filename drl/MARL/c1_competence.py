"""Training-only competence evidence for physical-time stagnation patience.

Defaults are a conservative engineering choice, not a scientific optimum.
One observation is an independent next-target-edge attempt, never a PPO update
or primitive retry. This module neither commands actions nor terminates episodes.
"""
from collections import Counter, deque
import math


DEFAULT_CONFIG = dict(window_size=64, min_observations=20, min_successes=10,
    min_success_lower_bound=.6, wilson_z=1.96, duration_quantile=.9,
    duration_multiplier=2., handoff_margin_s=5., fallback_patience_s=60.,
    min_patience_s=20., max_patience_s=90., baseline_max_s=120.)


def _finite(value, name, minimum=0.):
    value = float(value)
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f'{name} must be finite and >= {minimum}')
    return value


def _integer(value, name, minimum=0):
    number = _finite(value, name, minimum)
    if number != int(number) or isinstance(value, bool):
        raise ValueError(f'{name} must be an integer')
    return int(number)


def _key(domain, stage):
    if not isinstance(domain, str) or not domain.strip():
        raise ValueError('domain must be a nonempty string')
    if not isinstance(stage, (str, int)) or isinstance(stage, bool) or not str(stage).strip():
        raise ValueError('stage must be a nonempty string or integer')
    return domain, str(stage)


def _attempt(elapsed_s, success, primitive_failures=0, recoveries=0,
             opportunities=0, censored=False):
    elapsed = _finite(elapsed_s, 'elapsed_s')
    if not isinstance(success, bool) or not isinstance(censored, bool):
        raise ValueError('success and censored must be boolean')
    if success and (censored or elapsed == 0.):
        raise ValueError('success requires positive elapsed time and cannot be censored')
    return dict(elapsed_s=elapsed, success=success, censored=censored,
        primitive_failures=_integer(primitive_failures, 'primitive_failures'),
        recoveries=_integer(recoveries, 'recoveries'),
        opportunities=_integer(opportunities, 'opportunities'))


def _wilson(successes, attempts, z):
    if not attempts:
        return 0.
    fraction = successes / attempts
    return ((fraction + z*z/(2*attempts)
        - z * math.sqrt(fraction*(1-fraction)/attempts + z*z/(4*attempts*attempts)))
        / (1 + z*z/attempts))


def _duration_quantiles(records, quantile):
    successes = sorted(r['elapsed_s'] for r in records if r['success'])
    if not successes:
        return None, None
    # Nearest-rank quantile avoids interpolating an unsupported fast duration.
    observed = successes[max(0, math.ceil(quantile*len(successes))-1)]
    # Treat every unsuccessful attempt as right-censored for time-to-next-edge,
    # even when its separate reported censor flag is false. Never call a failed
    # primitive's duration a successful duration. Events precede censors at ties.
    events, removals = Counter(), Counter()
    for record in records:
        when = record['elapsed_s']
        removals[when] += 1
        if record['success']:
            events[when] += 1
    at_risk, survival = len(records), 1.
    for when in sorted(removals):
        survival *= 1. - events[when] / at_risk
        if 1. - survival >= quantile - 1e-12:
            return observed, when
        at_risk -= removals[when]
    # The requested tail is unresolved: keeping fallback avoids fitting only
    # the fast successful subset. KM still assumes noninformative censoring;
    # this is a conservative guard, not an unbiased scientific estimator.
    return observed, None


class CompetencePatience:
    """Bounded sliding windows keyed by domain and stage; evaluation is read-only."""

    def __init__(self, config=None, *, training=True):
        supplied = dict(config or {})
        unknown = supplied.keys() - DEFAULT_CONFIG.keys()
        if unknown:
            raise ValueError(f'Unknown competence config keys: {sorted(unknown)}')
        self.config = {**DEFAULT_CONFIG, **supplied}
        c = self.config
        for name in ('window_size', 'min_observations', 'min_successes'):
            c[name] = _integer(c[name], name, 1)
        if not c['min_successes'] <= c['min_observations'] <= c['window_size']:
            raise ValueError('Require min_successes <= min_observations <= window_size')
        for name in DEFAULT_CONFIG.keys() - {'window_size', 'min_observations', 'min_successes'}:
            c[name] = _finite(c[name], name)
        if not (0. < c['min_success_lower_bound'] < 1.
                and 0. < c['duration_quantile'] <= 1.
                and c['wilson_z'] > 0. and c['duration_multiplier'] >= 1.):
            raise ValueError('Invalid probability, confidence, or duration multiplier')
        if not (0. < c['min_patience_s'] <= c['max_patience_s'] <= c['baseline_max_s']
                and c['min_patience_s'] <= c['fallback_patience_s'] <= c['baseline_max_s']):
            raise ValueError('Patience bounds must be positive and within baseline_max_s')
        self.training = bool(training)
        self._windows = {}

    def observe(self, domain, stage, *, elapsed_s, success, primitive_failures=0,
                recoveries=0, opportunities=0, censored=False, split='TRAIN'):
        """Append one closed attempt; VALIDATION/TEST never alter TRAIN evidence."""
        if not self.training or str(split).upper() != 'TRAIN':
            return False
        key = _key(domain, stage)
        record = _attempt(elapsed_s, success, primitive_failures, recoveries,
                          opportunities, censored)
        self._windows.setdefault(key, deque(maxlen=self.config['window_size'])).append(record)
        return True

    def _summary(self, domain, stage):
        key = _key(domain, stage)
        records = self._windows.get(key, ())
        c, n = self.config, len(records)
        successes = sum(r['success'] for r in records)
        lower = _wilson(successes, n, c['wilson_z'])
        observed, guarded = _duration_quantiles(records, c['duration_quantile'])
        enough = (n >= c['min_observations'] and successes >= c['min_successes']
                  and lower >= c['min_success_lower_bound'])
        mastered = enough and guarded is not None
        patience = c['fallback_patience_s']
        if mastered:
            patience = min(c['max_patience_s'], max(c['min_patience_s'],
                max(observed, guarded)*c['duration_multiplier'] + c['handoff_margin_s']))
        return dict(domain=key[0], stage=key[1], attempts=n, successes=successes,
            failures=n-successes, censored=sum(r['censored'] for r in records),
            success_fraction=successes/n if n else 0., wilson_lower_bound=lower,
            primitive_failures=sum(r['primitive_failures'] for r in records),
            recoveries=sum(r['recoveries'] for r in records),
            opportunities=sum(r['opportunities'] for r in records),
            successful_duration_quantile_s=observed,
            censor_aware_duration_quantile_s=guarded, enough_evidence=enough,
            mastered=mastered, patience_s=patience)

    def patience(self, domain, stage):
        return self._summary(domain, stage)['patience_s']

    def snapshot(self, domain=None, stage=None):
        if domain is not None and stage is not None:
            return self._summary(domain, stage)
        if domain is not None or stage is not None:
            raise ValueError('Provide both domain and stage, or neither')
        return dict(config=dict(self.config),
            stages=[self._summary(*key) for key in sorted(self._windows)])

    def state_dict(self):
        return dict(version=1, config=dict(self.config), stages=[dict(domain=key[0],
            stage=key[1], records=[dict(r) for r in self._windows[key]])
            for key in sorted(self._windows)])

    @classmethod
    def from_state_dict(cls, state, *, training=True):
        if state.get('version') != 1:
            raise ValueError('Unsupported competence state version')
        policy = cls(state['config'], training=True)
        for item in state['stages']:
            key = _key(item['domain'], item['stage'])
            if key in policy._windows or len(item['records']) > policy.config['window_size']:
                raise ValueError('Duplicate or oversized competence window')
            for record in item['records']:
                policy.observe(*key, **record)
        policy.training = bool(training)
        return policy


def competence_stage(target_count):
    """Stage of the next required edge, based on real matched target counts."""
    count = _integer(target_count, 'target_count')
    if count > 7:
        raise ValueError('target_count must be within 0..7')
    return ('S0' if count == 0 else 'S1' if count <= 2 else
            'S2' if count <= 4 else 'S3' if count <= 6 else 'FOLD')


class CompetenceEpisodeTracker:
    """Close attempts at new target-count records or one terminal censor.

    A jump closes one observation, rather than inventing independent attempts
    for unobserved intermediate edges. Regressions, retries and oscillations do
    not renew the physical clock. The caller supplies an actual target matching
    count, not raw graph edges or a potential score.
    """

    def __init__(self, domain, initial_target_count, start_s, policy, split='train'):
        self.stage = competence_stage(initial_target_count)
        _key(domain, self.stage)
        self.domain, self.policy, self.split = domain, policy, split
        self.best_count = int(initial_target_count)
        self.started = self.last_time = _finite(start_s, 'start_s')
        self.failures = self.opportunities = 0
        self.seen_terminal = set()
        self.finished = False

    def observe(self, time_s, target_count, primitive_events=(),
                decision_opportunity=False, finished=False, truncated=False):
        if self.finished:
            return []
        now = _finite(time_s, 'time_s')
        competence_stage(target_count)
        if now < self.last_time:
            raise ValueError('Episode time cannot move backwards')
        self.last_time = now
        self.opportunities += int(bool(decision_opportunity))
        for event in primitive_events:
            if event.state not in ('failed', 'canceled', 'rejected', 'succeeded'):
                continue
            if event.goal_id in self.seen_terminal:
                continue
            self.seen_terminal.add(event.goal_id)
            self.failures += int(event.state in ('failed', 'canceled', 'rejected'))
        progress = target_count > self.best_count
        terminal = bool(finished or truncated)
        records = []
        if progress or terminal:
            # A zero-time terminal failure is legitimate censor evidence; a
            # zero-time edge jump cannot provide a meaningful success duration.
            record = dict(domain=self.domain, stage=self.stage,
                elapsed_s=now-self.started, success=bool(progress),
                censored=not progress, primitive_failures=self.failures,
                recoveries=int(progress and self.failures > 0),
                opportunities=self.opportunities,
                target_count_before=self.best_count, target_count_after=int(target_count))
            self.policy.observe(self.domain, self.stage,
                **{k: record[k] for k in ('elapsed_s', 'success', 'censored',
                    'primitive_failures', 'recoveries', 'opportunities')}, split=self.split)
            records.append(record)
            if progress:
                self.best_count = int(target_count)
                self.stage = competence_stage(self.best_count)
                self.started = now
                self.failures = self.opportunities = 0
        self.finished = terminal
        return records
